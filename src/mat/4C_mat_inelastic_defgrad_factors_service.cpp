// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_mat_inelastic_defgrad_factors_service.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor_generators.hpp"
#include "4C_linalg_utils_quaternion_interpolation.hpp"
#include "4C_linalg_utils_scalar_interpolation.hpp"
#include "4C_linalg_utils_tensor_interpolation.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>


FOUR_C_NAMESPACE_OPEN

using namespace Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;
namespace AEI =
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolation;


namespace
{

  // elastic and plastic predictor locations
  constexpr double ELASTIC_PREDICTOR_LOCATION = 0.0;
  constexpr double PLASTIC_PREDICTOR_LOCATION = 1.0;
  const std::vector<Core::LinAlg::Matrix<1, 1>> ELASTIC_AND_PLASTIC_PREDICTOR_LOCATIONS = []()
  {
    Core::LinAlg::Matrix<1, 1> elast(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<1, 1> plast(Core::LinAlg::Initialization::zero);
    elast(0, 0) = ELASTIC_PREDICTOR_LOCATION;
    plast(0, 0) = PLASTIC_PREDICTOR_LOCATION;
    return std::vector<Core::LinAlg::Matrix<1, 1>>{elast, plast};
  }();


  const Core::LinAlg::Matrix<4, 1> UNIT_QUATERNION = []()
  {
    Core::LinAlg::Matrix<4, 1> uq{Core::LinAlg::Initialization::zero};
    uq(3) = 1.0;

    return uq;
  }();

  // creates the eigenvalue interpolator used for the Adaptive Estimate Interpolation
  Core::LinAlg::ScalarInterpolator<1> create_eigenvalue_interpolator()
  {
    Core::LinAlg::ScalarInterpolationType interp_type =
        Core::LinAlg::ScalarInterpolationType::logarithmic_weighted_average;
    Core::LinAlg::ScalarInterpolationWeightingFunction weight_func =
        Core::LinAlg::ScalarInterpolationWeightingFunction::inverse_distance;
    Core::LinAlg::ScalarInterpolationParams interp_params;

    return {interp_type, weight_func, interp_params};
  }


  // Adaptive Estimate Interpolation: compute the elastic deformation gradient, using the
  // interpolated eigenvalues and rotation contributions (quaternions) with respect to the elastic
  // deformation gradient within the elastic predictor
  Core::LinAlg::Matrix<3, 3> compute_elast_defgrad_wrt_elast_predictor(
      const std::vector<double>& interp_eigenval,
      const Core::LinAlg::Matrix<3, 3>& eigenvect_rot_elast_pred,
      const Core::LinAlg::Matrix<4, 1>& interp_rel_eigenvect_rot_quat,
      const Core::LinAlg::Matrix<3, 3>& rot_elast_pred,
      const Core::LinAlg::Matrix<4, 1>& interp_rel_rot_quat)
  {
    Core::LinAlg::Matrix<3, 3> out{Core::LinAlg::Initialization::zero};

    // construct diagonal eigenvalue matrix
    Core::LinAlg::Matrix<3, 3> eigenval_matrix{Core::LinAlg::Initialization::zero};
    for (unsigned int i = 0; i < 3; ++i)
    {
      eigenval_matrix(i, i) = interp_eigenval[i];
    }

    // construct eigenvector matrix
    Core::LinAlg::Matrix<3, 3> rel_interp_eigenvect_matrix{Core::LinAlg::Initialization::zero};
    Core::LargeRotations::quaterniontotriad(
        interp_rel_eigenvect_rot_quat, rel_interp_eigenvect_matrix);
    Core::LinAlg::Matrix<3, 3> interp_eigenvect_matrix{Core::LinAlg::Initialization::zero};
    interp_eigenvect_matrix.multiply_nn(
        1.0, eigenvect_rot_elast_pred, rel_interp_eigenvect_matrix, 0.0);


    // construct rotation matrix
    Core::LinAlg::Matrix<3, 3> rel_interp_rot_matrix{Core::LinAlg::Initialization::zero};
    Core::LargeRotations::quaterniontotriad(interp_rel_rot_quat, rel_interp_rot_matrix);
    Core::LinAlg::Matrix<3, 3> interp_rot_matrix{Core::LinAlg::Initialization::zero};
    interp_rot_matrix.multiply_nn(1.0, rot_elast_pred, rel_interp_rot_matrix, 0.0);


    // multiply contributions
    Core::LinAlg::Matrix<3, 3> LQ{Core::LinAlg::Initialization::zero};
    LQ.multiply(1.0, eigenval_matrix, interp_eigenvect_matrix, 0.0);
    Core::LinAlg::Matrix<3, 3> QTLQ{Core::LinAlg::Initialization::zero};
    QTLQ.multiply_tn(1.0, interp_eigenvect_matrix, LQ, 0.0);
    out.multiply(1.0, interp_rot_matrix, QTLQ, 0.0);

    return out;
  }


  // precondition matrix: absolute values smaller than a set tolerance are set to 0.0
  Core::LinAlg::Matrix<3, 3> precondition_matrix(
      const Core::LinAlg::Matrix<3, 3>& matrix, const double tol)
  {
    Core::LinAlg::Matrix<3, 3> out_matrix{matrix};

    for (unsigned i = 0; i < 3; ++i)
    {
      for (unsigned j = 0; j < 3; ++j)
      {
        if (std::abs(matrix(i, j)) < tol)
        {
          out_matrix(i, j) = 0.0;
        }
      }
    }
    return out_matrix;
  }


  //! calculates the starting point for the Adaptive Estimate Interpolation based on the equivalent
  //! stress of the previous solution between its both predictors
  double calculate_equiv_stress_starting_point(
      const AEI::InputEquivStressStartingPoint& input_equiv_stress_starting_point)
  {
    // set to elastic predictor if the stress of the elastic predictor is numerically 0.0 ->
    // this is theoretically
    // possible for viscoplastic laws without yield surfaces, which may exhibit plastic flow
    // even in this case; however, the determination of the starting point requires dividing
    // over this stress value, which will not be possible in this specific case.
    // Same goes for the case where the elastic predictor and the plastic predictor are
    // associated with effectively the same stress value (e.g., during stress relaxation) -> set
    // starting point as elastic predictor in these particular cases
    if (input_equiv_stress_starting_point.equiv_stress_elast_pred <= 1.0e-12 ||
        std::abs(input_equiv_stress_starting_point.equiv_stress_plast_pred -
                 input_equiv_stress_starting_point.equiv_stress_elast_pred) /
                input_equiv_stress_starting_point.equiv_stress_elast_pred <
            1.0e-8)
    {
      return ELASTIC_PREDICTOR_LOCATION;
    }


    // compute starting point based on the equivalent stress: we clamp between the elastic and
    // plastic predictors because in some special cases such as stress relaxation, the starting
    // point may be slightly out of this interval (machine precision)
    return std::clamp((input_equiv_stress_starting_point.equiv_stress_solution -
                          input_equiv_stress_starting_point.equiv_stress_elast_pred) /
                          (input_equiv_stress_starting_point.equiv_stress_plast_pred -
                              input_equiv_stress_starting_point.equiv_stress_elast_pred),
        ELASTIC_PREDICTOR_LOCATION, PLASTIC_PREDICTOR_LOCATION);
  }

  // Validate that AEIParams is internally consistent and return it unchanged.
  // --> throws before any dependent member gets constructed from a bad state.
  const AEI::AEIParams& validate_aei_params(const AEI::AEIParams& aei_params)
  {
    const auto& interp = aei_params.estimate_interpolation;

    if (interp.starting_point_type == AEI::StartingPointType::constant)
    {
      // for constant starting points, the user-set starting point is required
      FOUR_C_ASSERT_ALWAYS(interp.user_set_starting_point.has_value(),
          "No user-set starting point has been specified! This is required for the used "
          "starting point type {}",
          EnumTools::enum_name(AEI::StartingPointType::constant));
    }
    else
    {
      // for other starting point types, specifying the user-set starting point is not allowed!
      FOUR_C_ASSERT_ALWAYS(!interp.user_set_starting_point.has_value(),
          "No user-set starting point should be specified for the used starting "
          "point type {}",
          EnumTools::enum_name(interp.starting_point_type));
    }

    return aei_params;
  }


}  // namespace


void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ThermoMechanicalCouplingCache::resize(
    const unsigned int numgp)
{
  std::apply([numgp](auto&... quantity) { (quantity.resize(numgp), ...); }, quantities());
}

void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ThermoMechanicalCouplingCache::reset(
    const int gp)
{
  std::apply([gp](auto&... quantity) { (quantity.reset(gp), ...); }, quantities());
}


std::string
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::get_detailed_error_message_for_error_type(
    ErrorType err_type)
{
  switch (err_type)
  {
    case ErrorType::negative_plastic_strain:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: negative plastic strain!";
    case ErrorType::overflow_error:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: overflow error related to "
             "the evaluation of the plastic strain increment!";
    case ErrorType::failed_solution_linear_system_lnl:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: solution of the linear "
             "system in the Local Newton Loop failed!";
    case ErrorType::no_convergence_local_newton:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Local Newton Loop did not "
             "converge for the given loop settings!";
    case ErrorType::singular_jacobian:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: singular Jacobian after "
             "converged Local Newton Loop, which does not allow for the analytical evaluation "
             "of the linearization!";
    case ErrorType::failed_solution_analytic_linearization:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: solution of the linear "
             "system in the analytical linearization failed";
    case ErrorType::failed_computation_flow_resistance:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed while computing "
             "the flow resistance for the viscoplasticity law";
    case ErrorType::failed_computation_flow_resistance_derivs:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed while computing "
             "the derivatives of the flow resistance for the viscoplasticity law";
    case ErrorType::failed_matrix_log_evaluation:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed in evaluating the "
             "matrix logarithm or its derivative with respect to the argument";
    case ErrorType::failed_matrix_exp_evaluation:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed in evaluating the "
             "matrix exponential or its derivative with respect to the argument";
    case ErrorType::failed_right_cg_interpolation:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed in interpolating "
             "the right Cauchy-Green deformation tensor";
    case ErrorType::under_yield_surface:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: we are 'under' the yield "
             "surface, sigma < sigma_yield!";
    default:
      FOUR_C_THROW("to_string(ErrorType): {}: No error message provided!", err_type);
  }
}

Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ConstNonMatTensors::ConstNonMatTensors()
{  // auxiliaries
  Core::LinAlg::Matrix<3, 3> unit3x3(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) unit3x3(i, i) = 1.0;
  Core::LinAlg::Matrix<6, 6> temp6x6(Core::LinAlg::Initialization::zero);

  // set constant non-material tensors

  // 3x3 identity
  id3x3.update(1.0, unit3x3, 0.0);

  // Voigt stress form of 3x3 identity
  Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
      id3x3, id6x1);

  // symmetric identity four tensor
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(id4_6x6, 1.0, id3x3, id3x3, 0.0);

  // deviatoric operator
  Core::LinAlg::FourTensor<3> dev_op_four_tensor =
      Core::LinAlg::setup_deviatoric_projection_tensor<3>();
  Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, dev_op_four_tensor);
  dev_op = Core::LinAlg::Voigt::modify_voigt_representation(temp6x6, 1.0, 2.0);

  // identity four tensor
  id4_9x9.clear();
  Core::LinAlg::FourTensorOperations::add_non_symmetric_product(1.0, id3x3, id3x3, id4_9x9);

  // 10x10 identity
  id10x10.clear();
  for (int i = 0; i < 10; ++i) id10x10(i, i) = 1.0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ConstMatTensors::
    set_material_const_tensors(const Core::LinAlg::Matrix<3, 1>& m)
{
  // get instance of constant non-material tensors
  const auto& const_non_mat_tensors = ConstNonMatTensors::instance();

  // set material-dependent tensors (fiber orientation)

  // structural tensor
  mm.multiply_nt(1.0, m, m, 0.0);

  // deviatoric part of the structural tensor
  double tr_mm_ = mm(0, 0) + mm(1, 1) + mm(2, 2);
  mm_dev.update(1.0, mm, -1.0 / 3.0 * tr_mm_, const_non_mat_tensors.id3x3);

  // dyadic product of structural tensors
  Core::LinAlg::Matrix<6, 1> mm_V(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
      mm, mm_V);
  mm_dyad_mm.multiply_nt(1.0, mm_V, mm_V, 0.0);

  // dyadic product of deviatoric structural tensor with the structural tensor
  Core::LinAlg::Matrix<6, 1> mm_dev_V(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
      mm_dev, mm_dev_V);
  mm_dev_dyad_mm.multiply_nt(1.0, mm_dev_V, mm_V, 0.0);

  // dyadic product of identity with the structural tensor
  id_dyad_mm.multiply_nt(1.0, const_non_mat_tensors.id6x1, mm_V, 0.0);

  // sum of identity with the structural tensor
  id_plus_mm.update(1.0, const_non_mat_tensors.id3x3, 1.0, mm, 0.0);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalSubsteppingUtils::reset(
    const double dt)
{
  t_ = 0.0;
  substep_counter_ = 1;
  curr_dt_ = dt;
  time_step_halving_counter_ = 0;
  total_num_of_substeps_ = 1;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalSubsteppingUtils::
    increment_substep()
{
  t_ += curr_dt_;
  substep_counter_++;
};

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalSubsteppingUtils::halve_substep()
{
  curr_dt_ *= 1.0 / 2.0;
  time_step_halving_counter_ += 1;
  total_num_of_substeps_ += (total_num_of_substeps_ - substep_counter_ + 1);
};


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::init(
    const double ref_temperature)
{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> id3x3{Core::LinAlg::Initialization::zero};
  for (unsigned int i = 0; i < 3; ++i)
  {
    id3x3(i, i) = 1.0;
  }

  // ----- set last_ and current_ variables referring to values at different time instants
  // ----- for now: the number of Gauss points is unknown -> we set the values only for 1
  // Gauss point and update the number of Gauss points in the setup method

  // default values of the inverse plastic deformation gradient: unit tensor
  last_plastic_defgrad_inverse.resize(1, id3x3);
  current_plastic_defgrad_inverse.resize(1, id3x3);  // value irrelevant at this point
  last_substep_plastic_defgrad_inverse.resize(1, id3x3);

  // update last_ and current_ values of the plastic strain
  last_plastic_strain.resize(1, 0.0);
  current_plastic_strain.resize(1, 0.0);  // value irrelevant at this point
  last_substep_plastic_strain.resize(1, 0.0);

  // update last_ and current_ values of the temperature
  last_temperature.resize(1, ref_temperature);
  current_temperature.resize(1, ref_temperature);  // value irrelevant at this point

  // update last_ and current_ values of the equivalent stress
  last_equiv_stress.resize(1, 0.0);
  current_equiv_stress.resize(1, 0.0);  // value irrelevant at this point

  // default values of the right CG tensor: unit tensor
  last_rightCG.resize(1, id3x3);
  current_rightCG.resize(1, id3x3);  // value irrelevant at this point

  // default value for the current deformation gradient: zero tensor \f$ \boldsymbol{0} f$ (to make
  // sure that the inverse inelastic deformation gradient is evaluated in the first method call)
  last_defgrad.resize(1, Core::LinAlg::Matrix<3, 3>{id3x3});
  current_defgrad.resize(1, Core::LinAlg::Matrix<3, 3>{Core::LinAlg::Initialization::zero});
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::resize(
    const unsigned int numgp)
{
  FOUR_C_ASSERT_ALWAYS(!resize_called,
      "You already called resize for the time step quantities! The number of current GP is {} and "
      "you attempt to set it to {}",
      last_plastic_strain.size(), numgp);


  // default values of the inverse plastic deformation gradient for ALL Gauss Points
  last_plastic_defgrad_inverse.resize(numgp, last_plastic_defgrad_inverse[0]);
  current_plastic_defgrad_inverse.resize(numgp,
      last_plastic_defgrad_inverse[0]);  // value irrelevant at this point
  last_substep_plastic_defgrad_inverse.resize(numgp, last_substep_plastic_defgrad_inverse[0]);

  // default values of the plastic strain for ALL Gauss Points
  last_plastic_strain.resize(numgp, last_plastic_strain[0]);
  current_plastic_strain.resize(numgp, last_plastic_strain[0]);  // value irrelevant at this point
  last_substep_plastic_strain.resize(numgp, last_substep_plastic_strain[0]);

  // default values of the right CG deformation tensor for ALL Gauss Points
  last_rightCG.resize(numgp, last_rightCG[0]);
  current_rightCG.resize(numgp, last_rightCG[0]);  // value irrelevant at this point

  // default values of the equivalent stress for ALL Gauss Points
  last_equiv_stress.resize(numgp, last_equiv_stress[0]);
  current_equiv_stress.resize(numgp, current_equiv_stress[0]);  // value irrelevant at this point

  // default values of the deformation gradient
  last_defgrad.resize(numgp, last_defgrad[0]);
  current_defgrad.resize(numgp, current_defgrad[0]);

  // default values of the temperature for ALL Gauss Points
  last_temperature.resize(numgp, last_temperature[0]);        // value irrelevant at this point
  current_temperature.resize(numgp, current_temperature[0]);  // value irrelevant at this point

  resize_called = true;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::pre_evaluate(
    const unsigned int gp)
{
  FOUR_C_ASSERT_ALWAYS(gp < last_plastic_defgrad_inverse.size(),
      "You try to pre-evaluate the time step quantities at GP {}, but the object has only {} Gauss "
      "points",
      gp, last_plastic_defgrad_inverse.size());

  // set consistent last substep values
  last_substep_plastic_defgrad_inverse[gp] = last_plastic_defgrad_inverse[gp];
  last_substep_plastic_strain[gp] = last_plastic_strain[gp];
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::update(
    const unsigned int gp)
{
  FOUR_C_ASSERT_ALWAYS(gp < last_plastic_defgrad_inverse.size(),
      "You try to update the time step quantities at GP {}, but the object has only {} Gauss "
      "points",
      gp, last_plastic_defgrad_inverse.size());

  // update history variables for the next time step
  last_defgrad[gp] = current_defgrad[gp];
  last_rightCG[gp] = current_rightCG[gp];
  last_plastic_defgrad_inverse[gp] = current_plastic_defgrad_inverse[gp];
  last_substep_plastic_defgrad_inverse[gp] = current_plastic_defgrad_inverse[gp];
  last_plastic_strain[gp] = current_plastic_strain[gp];
  last_substep_plastic_strain[gp] = current_plastic_strain[gp];
  last_equiv_stress[gp] = current_equiv_stress[gp];
  last_temperature[gp] = current_temperature[gp];
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::pack(
    Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, last_defgrad);
  add_to_pack(data, last_rightCG);
  add_to_pack(data, last_plastic_defgrad_inverse);
  add_to_pack(data, last_plastic_strain);
  add_to_pack(data, last_equiv_stress);
  add_to_pack(data, last_substep_plastic_defgrad_inverse);
  add_to_pack(data, last_substep_plastic_strain);
  add_to_pack(data, last_temperature);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  // extract last values
  extract_from_pack(buffer, last_defgrad);
  extract_from_pack(buffer, last_rightCG);
  extract_from_pack(buffer, last_plastic_defgrad_inverse);
  extract_from_pack(buffer, last_plastic_strain);
  extract_from_pack(buffer, last_equiv_stress);
  extract_from_pack(buffer, last_substep_plastic_defgrad_inverse);
  extract_from_pack(buffer, last_substep_plastic_strain);
  extract_from_pack(buffer, last_temperature);

  // fill current_ values with the last_ values
  current_rightCG.resize(last_rightCG.size(),
      last_rightCG[0]);  // value irrelevant
  current_plastic_defgrad_inverse.resize(last_plastic_defgrad_inverse.size(),
      last_plastic_defgrad_inverse[0]);  // value irrelevant
  current_plastic_strain.resize(last_plastic_strain.size(),
      last_plastic_strain[0]);  // value irrelevant
  current_equiv_stress.resize(last_equiv_stress.size(),
      last_equiv_stress[0]);  // value irrelevant
  current_temperature.resize(last_temperature.size(),
      0.0);  // value irrelevant

  // set evaluated deformation gradient to 0, to make sure that the inverse inelastic deformation
  // gradient is evaluated fully after the restart
  current_defgrad.resize(last_substep_plastic_defgrad_inverse.size(),
      Core::LinAlg::Matrix<3, 3>{Core::LinAlg::Initialization::zero});
}

Core::LinAlg::Matrix<1, 6>
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::compute_taylor_quinney_wrt_cauchygreen(
    const double taylor_quinney_coefficient, const ThermoMechanicalCouplingState& state,
    const ThermoMechanicalCouplingStateDerivatives& state_derivatives,
    const HistoryVariablesDerivativesWrtCauchyGreen& history_variables_derivatives)
{
  Core::LinAlg::Matrix<1, 6> total_dequiv_stress_dC =
      state_derivatives.equiv_stress_wrt_cauchy_green;
  total_dequiv_stress_dC.multiply(1.0, state_derivatives.equiv_stress_wrt_inverse_plastic_defgrad,
      history_variables_derivatives.inv_plastic_defgrad_wrt_cauchy_green, 1.0);

  Core::LinAlg::Matrix<1, 6> total_dpsr_dC{Core::LinAlg::Initialization::zero};
  total_dpsr_dC.update(state_derivatives.plastic_strain_rate_derivs.deriv_plastic_strain,
      history_variables_derivatives.plastic_strain_wrt_cauchy_green, 0.0);
  total_dpsr_dC.update(
      state_derivatives.plastic_strain_rate_derivs.deriv_equiv_stress, total_dequiv_stress_dC, 1.0);

  Core::LinAlg::Matrix<1, 6> dR_TQ_dCV{Core::LinAlg::Initialization::zero};
  dR_TQ_dCV.update(
      taylor_quinney_coefficient * state.plastic_strain_rate, total_dequiv_stress_dC, 0.0);
  dR_TQ_dCV.update(taylor_quinney_coefficient * state.equiv_stress, total_dpsr_dC, 1.0);

  return dR_TQ_dCV;
}

double
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::compute_taylor_quinney_wrt_temperature(
    const double taylor_quinney_coefficient, const ThermoMechanicalCouplingState& state,
    const ThermoMechanicalCouplingStateDerivatives& state_derivatives,
    const HistoryVariablesDerivativesWrtTemperature& history_variables_derivatives)
{
  double total_dequiv_stress_dT = state_derivatives.equiv_stress_wrt_temperature;
  for (int i = 0; i < 9; ++i)
  {
    total_dequiv_stress_dT += state_derivatives.equiv_stress_wrt_inverse_plastic_defgrad(i) *
                              history_variables_derivatives.inv_plastic_defgrad_wrt_temperature(i);
  }

  const double total_dpsr_dT =
      state_derivatives.plastic_strain_rate_derivs.deriv_temperature +
      state_derivatives.plastic_strain_rate_derivs.deriv_plastic_strain *
          history_variables_derivatives.plastic_strain_wrt_temperature +
      state_derivatives.plastic_strain_rate_derivs.deriv_equiv_stress * total_dequiv_stress_dT;

  return taylor_quinney_coefficient *
         (total_dequiv_stress_dT * state.plastic_strain_rate + state.equiv_stress * total_dpsr_dT);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::LocalNewtonManager(
    const LocalNewtonParams& lnl_params)
    : params_(lnl_params)
{
  // set number of Gauss points to 1 temporarily, since we don't
  // know it at this point in time
  curr_num_iters_.resize(1, 0);

  // set initial number of iterations
  iter_ = 0;

  // initialize solution vector and convergence quantities with dummy values; they will be set
  // anyway to more meaningful values when starting the local Newton within the material model
  sol_ = Core::LinAlg::Matrix<10, 1>(Core::LinAlg::Initialization::zero);
  convergence_quantities_.residual_norm = 0.0;
  convergence_quantities_.increment_norm = 0.0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::resize(
    const unsigned int numgp)
{
  FOUR_C_ASSERT_ALWAYS(!resize_called_,
      "You already called resize for the Local Newton manager! The number of current GP is {} and "
      "you attempt to set it to {}",
      curr_num_iters_.size(), numgp);

  // resize arrays
  curr_num_iters_.resize(numgp, curr_num_iters_[0]);


  resize_called_ = true;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::
    update_after_local_newton(const unsigned int gp)
{
  // increment number of local Newton iterations for the current timestep at the
  // current GP
  curr_num_iters_[gp] += iter_;
}



/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::
    reset_curr_num_iters(const unsigned int gp)
{
  FOUR_C_ASSERT_ALWAYS(gp < curr_num_iters_.size(),
      "You try to reset the current number of iterations within the Local Newton manager at Gauss "
      "point {}, but the object only has {} Gauss points",
      gp, curr_num_iters_.size());

  curr_num_iters_[gp] = 0;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::
    save_init_estimate_and_reset_convergence_quantities(
        const Core::LinAlg::Matrix<10, 1>& init_estimate)
{
  // --> set initial estimate
  sol_ = init_estimate;

  // --> set quantities used for convergence checks

  // residual norm
  convergence_quantities_.residual_norm = 0.0;
  // if the convergence check requires verifying the residual norm, we must ensure that the value
  // set here is larger than the tolerance, to perform the check at least once, in the next
  // iteration
  if (params_.conv_check == LocalNewtonConvCheck::residual ||
      params_.conv_check == LocalNewtonConvCheck::residual_and_increment_ratio)
  {
    convergence_quantities_.residual_norm = 2.0 * params_.res_tol;
  }

  // increment norm: ratio of increment to current solution
  convergence_quantities_.increment_norm = 0.0;
  // if the convergence check requires verifying the increment norm, we must ensure that the value
  // set here is larger than the tolerance, to perform the check at least once, in the next
  // iteration
  if (params_.conv_check == LocalNewtonConvCheck::increment_ratio ||
      params_.conv_check == LocalNewtonConvCheck::residual_and_increment_ratio)
  {
    convergence_quantities_.increment_norm = 2.0 * params_.incr_tol;
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::
    increment_solution_vector(const Core::LinAlg::Matrix<10, 1>& delta_sol)
{
  sol_.update(1.0, delta_sol, 1.0);

  const double sol_norm = sol_.norm2();
  const double delta_sol_norm = delta_sol.norm2();
  FOUR_C_ASSERT_ALWAYS(sol_norm >= 1.0e-8,
      "The solution vector in local iteration {} is nearly 0, with 2-norm: {}! Something went "
      "wrong, since such mechanical states are not expected!",
      iter_, sol_norm);
  convergence_quantities_.increment_norm = delta_sol_norm / sol_norm;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::
    is_local_newton_converged() const
{
  // check for convergence
  switch (params_.conv_check)
  {
    case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::residual:
      return (convergence_quantities_.residual_norm <= params_.res_tol);
      break;
    case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::increment_ratio:
      return (convergence_quantities_.increment_norm <= params_.incr_tol);
      break;
    case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::
        residual_and_increment_ratio:
      return (convergence_quantities_.residual_norm <= params_.res_tol &&
              convergence_quantities_.increment_norm <= params_.incr_tol);
      break;
    default:
      FOUR_C_THROW("You should not be here (convergence checking of the Local Newton Loop)");
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::
    is_local_newton_stuck() const
{
  // check for "stuck" Local Newton, i.e., the increment does not change much but there is not a
  // converged state (check only feasible after the first iteration, since dx must be available)
  if ((iter_ > 0) && (convergence_quantities_.increment_norm < 1.0e-15))
  {
    // only in the case that the residual is verified, we set an
    // error status
    switch (params_.conv_check)
    {
      case LocalNewtonConvCheck::residual:
      case LocalNewtonConvCheck::residual_and_increment_ratio:
      {
        return (convergence_quantities_.residual_norm > params_.res_tol);
      }
      case LocalNewtonConvCheck::increment_ratio:
      {
        return false;
      }
      default:
        FOUR_C_THROW(
            "You should not be here with convergence check type {} (check: is Local Newton "
            "stuck?)",
            EnumTools::enum_name(params_.conv_check));
    }
  }

  return false;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::pack(
    Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, curr_num_iters_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  // extract last values
  extract_from_pack(buffer, curr_num_iters_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalIntegrationInput::
    LocalIntegrationInput(const Config& cfg)
{
  defgrad = cfg.defgrad;
  inv_defgrad.invert(defgrad);
  right_cg.multiply_tn(1.0, defgrad, defgrad, 0.0);
  elastic_predictor_inverse_plastic_defgrad = cfg.last_inv_inelastic_defgrad;
  elastic_predictor_elastic_defgrad.multiply(
      1.0, defgrad, elastic_predictor_inverse_plastic_defgrad, 0.0);
  temperature = cfg.temperature;
  last_plastic_strain = cfg.last_plastic_strain;
  step = cfg.step;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
AEI::PredictorInterpolator::PredictorInterpolator()
    : ref_predictor_locs_(ELASTIC_AND_PLASTIC_PREDICTOR_LOCATIONS),
      eigenval_interpolator_(create_eigenvalue_interpolator())
{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> unit_3x3{Core::LinAlg::Initialization::zero};
  for (unsigned int i = 0; i < 3; ++i)
  {
    unit_3x3(i, i) = 1.0;
  }
  std::vector<std::vector<double>> vector_of_ones(2, {1.0, 1.0, 1.0});

  // initialize variables for a single Gauss point
  eigenval_elast_pred_.resize(1, unit_3x3);
  eigenval_plast_pred_.resize(1, unit_3x3);
  scalar_interp_eigenval_.resize(1, vector_of_ones);
  eigenvect_rot_elast_pred_.resize(1, unit_3x3);
  rel_eigenvect_rot_plast_pred_.resize(1, UNIT_QUATERNION);
  rot_elast_pred_.resize(1, unit_3x3);
  rel_rot_plast_pred_.resize(1, UNIT_QUATERNION);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::PredictorInterpolator::resize(const unsigned int numgp)
{
  FOUR_C_ASSERT(!resize_called_,
      "You already called resize for the predictor interpolator! The number of current GP is {} "
      "and "
      "you attempt to set it to {}",
      eigenval_elast_pred_.size(), numgp);

  eigenval_elast_pred_.resize(numgp, eigenval_elast_pred_[0]);
  eigenval_plast_pred_.resize(numgp, eigenval_plast_pred_[0]);
  scalar_interp_eigenval_.resize(numgp, scalar_interp_eigenval_[0]);
  eigenvect_rot_elast_pred_.resize(numgp, eigenvect_rot_elast_pred_[0]);
  rel_eigenvect_rot_plast_pred_.resize(numgp, rel_eigenvect_rot_plast_pred_[0]);
  rot_elast_pred_.resize(numgp, rot_elast_pred_[0]);
  rel_rot_plast_pred_.resize(numgp, rel_rot_plast_pred_[0]);

  resize_called_ = true;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::PredictorInterpolator::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::add_to_pack(data, eigenval_elast_pred_);
  Core::Communication::add_to_pack(data, eigenval_plast_pred_);
  Core::Communication::add_to_pack(data, scalar_interp_eigenval_);
  Core::Communication::add_to_pack(data, eigenvect_rot_elast_pred_);
  Core::Communication::add_to_pack(data, rel_eigenvect_rot_plast_pred_);
  Core::Communication::add_to_pack(data, rot_elast_pred_);
  Core::Communication::add_to_pack(data, rel_rot_plast_pred_);
  Core::Communication::add_to_pack(data, resize_called_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::PredictorInterpolator::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_from_pack(buffer, eigenval_elast_pred_);
  Core::Communication::extract_from_pack(buffer, eigenval_plast_pred_);
  Core::Communication::extract_from_pack(buffer, scalar_interp_eigenval_);
  Core::Communication::extract_from_pack(buffer, eigenvect_rot_elast_pred_);
  Core::Communication::extract_from_pack(buffer, rel_eigenvect_rot_plast_pred_);
  Core::Communication::extract_from_pack(buffer, rot_elast_pred_);
  Core::Communication::extract_from_pack(buffer, rel_rot_plast_pred_);
  Core::Communication::extract_from_pack(buffer, resize_called_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::PredictorInterpolator::construct_prelim_plastic_pred(const unsigned int gp,
    const Core::LinAlg::Matrix<3, 3>& elastic_defgrad_elastic_pred,
    const double elastic_predictor_zero_component_threshold,
    const PlasticPredictorConstructionParams& plastic_predictor_construction_params)
{
  // consistency checks
  FOUR_C_ASSERT(gp < eigenval_elast_pred_.size(),
      "Inconsistent Gauss point index {}, with set Gauss point size {}", gp,
      eigenval_elast_pred_.size());

  // get (preconditioned) elastic deformation gradient to be considered as elastic predictor
  Core::LinAlg::Matrix<3, 3> precond_elastic_defgrad_elastic_pred{elastic_defgrad_elastic_pred};
  precond_elastic_defgrad_elastic_pred =
      precondition_matrix(elastic_defgrad_elastic_pred, elastic_predictor_zero_component_threshold);

  //  perform polar-spectral decomposition of elastic defgrad within elastic predictor
  Core::LinAlg::Matrix<3, 3> material_stretch_elast_pred{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<3, 3> eigenval_elast_pred_temp{
      Core::LinAlg::Initialization::zero};  // needed here for the function call, but we will grab
                                            // the ordered eigenvalues from the spectral pairs
  std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> spectral_pairs_elast_pred;
  Core::LinAlg::matrix_3x3_polar_decomposition(precond_elastic_defgrad_elastic_pred,
      rot_elast_pred_[gp], material_stretch_elast_pred, eigenval_elast_pred_temp,
      spectral_pairs_elast_pred);
  for (int i = 0; i < 3; ++i)
  {
    FOUR_C_ASSERT_ALWAYS(spectral_pairs_elast_pred[i].first >= 1.0e-8,
        "The eigenvalue {} of the elastic deformation gradient within the elastic predictor at GP "
        "{} is {}, "
        "such that its logarithm can not be computed!",
        i, gp, spectral_pairs_elast_pred[i].first);
    eigenval_elast_pred_[gp](i, i) = spectral_pairs_elast_pred[i].first;
    for (int j = 0; j < 3; ++j)
    {
      eigenvect_rot_elast_pred_[gp](i, j) = spectral_pairs_elast_pred[i].second(j);
    }
  }

  // -->  construct a preliminary plastic predictor based on the parameter specifications

  // elastic stretch eigenvectors
  switch (plastic_predictor_construction_params.elastic_stretch_eigenvect_type)
  {
    case AEI::PrelimPlasticPredictor::ElasticStretchEigenvectType::from_elastic_predictor:
    {
      rel_eigenvect_rot_plast_pred_[gp].update(1.0, UNIT_QUATERNION, 0.0);
      break;
    }
    default:
    {
      // other eigenvector rotation types not yet enabled; in case of multiple eigenvalues, a
      // canonicalization approach for the related eigenvectors must be first implemented for the
      // spectral decomposition to avoid artificial rotation contributions
      FOUR_C_THROW(
          "Elastic stretch eigenvector type {} not yet enabled for the preliminary plastic "
          "predictor",
          EnumTools::enum_name(
              plastic_predictor_construction_params.elastic_stretch_eigenvect_type));
    }
  }

  // elastic rotation
  switch (plastic_predictor_construction_params.elastic_rotation_type)
  {
    case AEI::PrelimPlasticPredictor::ElasticRotationType::from_elastic_predictor:
    {
      rel_rot_plast_pred_[gp].update(1.0, UNIT_QUATERNION, 0.0);
      break;
    }
    default:
    {
      // same as in the case of the elastic stretch eigenvectors
      FOUR_C_THROW("Elastic rotation type {} not yet enabled for the plastic predictor!",
          EnumTools::enum_name(plastic_predictor_construction_params.elastic_rotation_type));
    }
  }

  // elastic stretch eigenvalues
  const double detF = precond_elastic_defgrad_elastic_pred.determinant();
  FOUR_C_ASSERT_ALWAYS(detF > 1.0e-8,
      "The determinant of the deformation gradient is {}, which is physically and numerically "
      "invalid!",
      detF);

  switch (plastic_predictor_construction_params.elastic_stretch_eigenval_type)
  {
    case AEI::PrelimPlasticPredictor::ElasticStretchEigenvalType::scale_unit:
    {
      const double scaled_detF = std::pow(detF, 1.0 / 3.0);
      for (unsigned int i = 0; i < 3; ++i)
      {
        eigenval_plast_pred_[gp](i, i) = scaled_detF;
      }

      break;
    }
    default:
    {
      FOUR_C_THROW("Elastic stretch eigenvalue type {} not yet enabled for the plastic predictor",
          EnumTools::enum_name(
              plastic_predictor_construction_params.elastic_stretch_eigenval_type));
    }
  }

  // store eigenvalues such that they can be directly used within the scalar interpolator
  scalar_interp_eigenval_[gp] = {{eigenval_elast_pred_[gp](0, 0), eigenval_elast_pred_[gp](1, 1),
                                     eigenval_elast_pred_[gp](2, 2)},
      {eigenval_plast_pred_[gp](0, 0), eigenval_plast_pred_[gp](1, 1),
          eigenval_plast_pred_[gp](2, 2)}};
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::PredictorInterpolator::interpolate_elastic_defgrad_contributions(const unsigned int gp,
    const double interp_loc, Core::LinAlg::Matrix<4, 1>& interp_rel_rot_quat,
    std::vector<double>& interp_eigenval,
    Core::LinAlg::Matrix<4, 1>& interp_rel_eigenvect_rot_quat) const
{
  // consistency checks
  FOUR_C_ASSERT(
      ELASTIC_PREDICTOR_LOCATION <= interp_loc && interp_loc <= PLASTIC_PREDICTOR_LOCATION,
      "Interpolation is constrained to the interval between the elastic and the plastic "
      "predictors! The current plastic predictor location "
      "{} is out of these bounds: [{}, {}]",
      interp_loc, ELASTIC_PREDICTOR_LOCATION, PLASTIC_PREDICTOR_LOCATION);
  FOUR_C_ASSERT(gp < eigenval_elast_pred_.size(),
      "Inconsistent Gauss point index {}, with set Gauss point size {}", gp,
      eigenval_elast_pred_.size());

  // auxiliaries
  Core::LinAlg::Matrix<1, 1> matrix_interp_loc{Core::LinAlg::Initialization::zero};
  matrix_interp_loc(0) = interp_loc;

  // interpolate eigenvalues
  interp_eigenval = eigenval_interpolator_.get_interpolated_scalar(
      scalar_interp_eigenval_[gp], ref_predictor_locs_, matrix_interp_loc);

  // interpolate quaternions
  interp_rel_eigenvect_rot_quat = Core::LinAlg::spherical_linear_interpolation(
      UNIT_QUATERNION, rel_eigenvect_rot_plast_pred_[gp], interp_loc);
  interp_rel_rot_quat = Core::LinAlg::spherical_linear_interpolation(
      UNIT_QUATERNION, rel_rot_plast_pred_[gp], interp_loc);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 3> AEI::PredictorInterpolator::interpolate_elastic_defgrad(
    const unsigned int gp, const double interp_loc) const
{
  // consistency checks
  FOUR_C_ASSERT(
      ELASTIC_PREDICTOR_LOCATION <= interp_loc && interp_loc <= PLASTIC_PREDICTOR_LOCATION,
      "Interpolation is constrained to the interval between the elastic and the plastic "
      "predictors! The current plastic predictor location "
      "{} is out of these bounds: [{}, {}]",
      interp_loc, ELASTIC_PREDICTOR_LOCATION, PLASTIC_PREDICTOR_LOCATION);
  FOUR_C_ASSERT(gp < eigenval_elast_pred_.size(),
      "Inconsistent Gauss point index {}, with set Gauss point size {}", gp,
      eigenval_elast_pred_.size());

  // interpolate contributions
  Core::LinAlg::Matrix<4, 1> interp_rel_rot_quat{Core::LinAlg::Initialization::zero};
  std::vector<double> interp_eigenval;
  Core::LinAlg::Matrix<4, 1> interp_rel_eigenvect_rot_quat{Core::LinAlg::Initialization::zero};
  interpolate_elastic_defgrad_contributions(
      gp, interp_loc, interp_rel_rot_quat, interp_eigenval, interp_rel_eigenvect_rot_quat);

  return compute_elast_defgrad_wrt_elast_predictor(interp_eigenval, eigenvect_rot_elast_pred_[gp],
      interp_rel_eigenvect_rot_quat, rot_elast_pred_[gp], interp_rel_rot_quat);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::PredictorInterpolator::set_plastic_predictor_after_construction_algo(
    const unsigned int gp, const double plastic_pred_loc)
{
  // consistency checks
  FOUR_C_ASSERT(ELASTIC_PREDICTOR_LOCATION <= plastic_pred_loc &&
                    plastic_pred_loc <= PLASTIC_PREDICTOR_LOCATION,
      "Interpolation constrained to the interval between the elastic and the plastic "
      "predictors! The current plastic predictor location "
      "{} is out of these bounds: [{}, {}]",
      plastic_pred_loc, ELASTIC_PREDICTOR_LOCATION, PLASTIC_PREDICTOR_LOCATION);
  FOUR_C_ASSERT(gp < eigenval_elast_pred_.size(),
      "Inconsistent Gauss point index {}, with set Gauss point size {}", gp,
      eigenval_elast_pred_.size());

  // set all quantities relevant for the plastic predictor
  std::vector<double> interp_eigenval;
  interpolate_elastic_defgrad_contributions(gp, plastic_pred_loc, rel_rot_plast_pred_[gp],
      interp_eigenval, rel_eigenvect_rot_plast_pred_[gp]);
  for (unsigned int i = 0; i < 3; ++i)
  {
    eigenval_plast_pred_[gp](i, i) = interp_eigenval[i];
  }
  scalar_interp_eigenval_[gp] = {{eigenval_elast_pred_[gp](0, 0), eigenval_elast_pred_[gp](1, 1),
                                     eigenval_elast_pred_[gp](2, 2)},
      {eigenval_plast_pred_[gp](0, 0), eigenval_plast_pred_[gp](1, 1),
          eigenval_plast_pred_[gp](2, 2)}};
}

AEI::InterpolationPointContainer::InterpolationPointContainer(
    const EstimateInterpolationParams& estimate_interpolation_params)
{
  lower_interp_bounds.resize(1, ELASTIC_PREDICTOR_LOCATION);
  upper_interp_bounds.resize(1, PLASTIC_PREDICTOR_LOCATION);
  switch (estimate_interpolation_params.starting_point_type)
  {
    case AEI::StartingPointType::constant:
    {
      FOUR_C_ASSERT(estimate_interpolation_params.user_set_starting_point.has_value(),
          "The user-set starting point is not specified!");

      starting_points.resize(1, estimate_interpolation_params.user_set_starting_point.value());
      break;
    }
    default:
    {
      starting_points.resize(1, estimate_interpolation_params.interval_scanning_param);
      break;
    }
  }
  current_interp_points.resize(1, starting_points[0]);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::InterpolationPointContainer::reset_bounds_and_current_interp_point(const unsigned int gp)
{
  // consistency checks
  FOUR_C_ASSERT(gp < current_interp_points.size(),
      "Inconsistent Gauss point index {}, with set Gauss point size {}", gp,
      current_interp_points.size());

  current_interp_points[gp] = starting_points[gp];
  lower_interp_bounds[gp] = ELASTIC_PREDICTOR_LOCATION;
  upper_interp_bounds[gp] = PLASTIC_PREDICTOR_LOCATION;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::InterpolationPointContainer::resize(const unsigned int numgp)
{
  FOUR_C_ASSERT(!resize_called,
      "You already called resize for the interpolation point container! The number of "
      "current GP is "
      "{} and "
      "you attempt to set it to {}",
      current_interp_points.size(), numgp);

  current_interp_points.resize(numgp, current_interp_points[0]);
  lower_interp_bounds.resize(numgp, lower_interp_bounds[0]);
  upper_interp_bounds.resize(numgp, upper_interp_bounds[0]);
  starting_points.resize(numgp, starting_points[0]);

  resize_called = true;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::InterpolationPointContainer::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::add_to_pack(data, current_interp_points);
  Core::Communication::add_to_pack(data, lower_interp_bounds);
  Core::Communication::add_to_pack(data, upper_interp_bounds);
  Core::Communication::add_to_pack(data, starting_points);
  Core::Communication::add_to_pack(data, resize_called);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::InterpolationPointContainer::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_from_pack(buffer, current_interp_points);
  Core::Communication::extract_from_pack(buffer, lower_interp_bounds);
  Core::Communication::extract_from_pack(buffer, upper_interp_bounds);
  Core::Communication::extract_from_pack(buffer, starting_points);
  Core::Communication::extract_from_pack(buffer, resize_called);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
AEI::AEIManager::AEIManager(const AEI::AEIParams& aei_params)
    : params_(validate_aei_params(aei_params)),
      interp_point_container_(aei_params.estimate_interpolation),
      predictor_interpolator_()
{
  // initialize class variables
  num_plastic_pred_construct_iters_ = 0;
  num_estimate_interp_iters_ = 0;
  num_reestimations_ = 0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::resize(const unsigned int num_gp)
{
  FOUR_C_ASSERT(!resize_called_,
      "You already called resize for the adaptive estimate interpolation manager! You attempt to "
      "set it to {}",
      num_gp);

  interp_point_container_.resize(num_gp);
  predictor_interpolator_.resize(num_gp);

  resize_called_ = true;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::reset_and_construct_prelim_plastic_pred(
    const unsigned int gp, const LocalIntegrationInput& local_integration_input)
{
  // reset tracking variables
  num_plastic_pred_construct_iters_ = 0;
  num_estimate_interp_iters_ = 0;
  num_reestimations_ = 0;
  interp_point_container_.reset_bounds_and_current_interp_point(gp);

  // construct the preliminary predictor
  predictor_interpolator_.construct_prelim_plastic_pred(gp,
      local_integration_input.elastic_predictor_elastic_defgrad,
      params_.elastic_predictor_zero_component_threshold, params_.plastic_predictor_construction);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::pack(Core::Communication::PackBuffer& data) const
{
  interp_point_container_.pack(data);
  predictor_interpolator_.pack(data);
  Core::Communication::add_to_pack(data, resize_called_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::unpack(Core::Communication::UnpackBuffer& buffer)
{
  interp_point_container_.unpack(buffer);
  predictor_interpolator_.unpack(buffer);
  Core::Communication::extract_from_pack(buffer, resize_called_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 3> AEI::AEIManager::interpolate_inverse_inelastic_defgrad(
    const unsigned int gp, const Core::LinAlg::Matrix<3, 3>& inv_defgrad)
{
  Core::LinAlg::Matrix<3, 3> interp_elastic_defgrad =
      predictor_interpolator_.interpolate_elastic_defgrad(
          gp, interp_point_container_.current_interp_points[gp]);

  Core::LinAlg::Matrix<3, 3> inv_inelastic_defgrad{Core::LinAlg::Initialization::zero};
  inv_inelastic_defgrad.multiply(1.0, inv_defgrad, interp_elastic_defgrad);

  return inv_inelastic_defgrad;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 3> AEI::AEIManager::get_inverse_inelastic_defgrad_plastic_pred(
    const unsigned int gp, const Core::LinAlg::Matrix<3, 3>& inv_defgrad)
{
  // the plastic predictor lies at the location 1.0
  Core::LinAlg::Matrix<3, 3> interp_elastic_defgrad =
      predictor_interpolator_.interpolate_elastic_defgrad(gp, 1.0);

  Core::LinAlg::Matrix<3, 3> inv_inelastic_defgrad{Core::LinAlg::Initialization::zero};
  inv_inelastic_defgrad.multiply(1.0, inv_defgrad, interp_elastic_defgrad);

  return inv_inelastic_defgrad;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::set_plastic_predictor_after_construction_algo(const unsigned int gp)
{
  // set the plastic predictor quantities
  predictor_interpolator_.set_plastic_predictor_after_construction_algo(
      gp, interp_point_container_.current_interp_points[gp]);

  // reset the interpolation point container
  interp_point_container_.reset_bounds_and_current_interp_point(gp);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::adapt_interpolation_interval(
    const unsigned int gp, const InterpolationIntervalShift& interval_shift)
{
  switch (interval_shift)
  {
    case InterpolationIntervalShift::towards_elastic_pred:
    {
      interp_point_container_.upper_interp_bounds[gp] =
          interp_point_container_.current_interp_points[gp];
      break;
    }
    case InterpolationIntervalShift::towards_plastic_pred:
    {
      interp_point_container_.lower_interp_bounds[gp] =
          interp_point_container_.current_interp_points[gp];
      break;
    }
    default:
    {
      FOUR_C_THROW(
          "You should not be here in the interpolation routine! The shift direction {} is not "
          "supported!",
          EnumTools::enum_name(interval_shift));
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::set_current_interp_point(
    const unsigned int gp, const CurrentInterpPointPreset preset)
{
  FOUR_C_ASSERT(gp < interp_point_container_.current_interp_points.size(), "GP index out of range");
  switch (preset)
  {
    case CurrentInterpPointPreset::plastic_pred_construct_update:
    {
      interp_point_container_.current_interp_points[gp] =
          interp_point_container_.lower_interp_bounds[gp] +
          params_.plastic_predictor_construction.interval_scanning_param *
              (interp_point_container_.upper_interp_bounds[gp] -
                  interp_point_container_.lower_interp_bounds[gp]);
      return;
    }
    case CurrentInterpPointPreset::estimate_interpolation_update:
    {
      interp_point_container_.current_interp_points[gp] =
          interp_point_container_.lower_interp_bounds[gp] +
          params_.estimate_interpolation.interval_scanning_param *
              (interp_point_container_.upper_interp_bounds[gp] -
                  interp_point_container_.lower_interp_bounds[gp]);
      return;
    }
    case CurrentInterpPointPreset::lower_interp_bound:
    {
      interp_point_container_.current_interp_points[gp] =
          interp_point_container_.lower_interp_bounds[gp];
      return;
    }
    case CurrentInterpPointPreset::upper_interp_bound:
    {
      interp_point_container_.current_interp_points[gp] =
          interp_point_container_.upper_interp_bounds[gp];
      return;
    }
    case CurrentInterpPointPreset::elastic_predictor:
    {
      interp_point_container_.current_interp_points[gp] = ELASTIC_PREDICTOR_LOCATION;
      return;
    }
    case CurrentInterpPointPreset::plastic_predictor:
    {
      interp_point_container_.current_interp_points[gp] = PLASTIC_PREDICTOR_LOCATION;
      return;
    }
    case CurrentInterpPointPreset::starting_point:
    {
      interp_point_container_.current_interp_points[gp] =
          interp_point_container_.starting_points[gp];
      return;
    }
    case CurrentInterpPointPreset::intermediate_point:
    {
      interp_point_container_.current_interp_points[gp] =
          interp_point_container_.lower_interp_bounds[gp] +
          params_.reestimation.interval_scanning_param *
              (interp_point_container_.current_interp_points[gp] -
                  interp_point_container_.lower_interp_bounds[gp]);
      return;
    }
    default:
      FOUR_C_THROW(
          "Unsupported current interpolation point preset {}", EnumTools::enum_name(preset));
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::set_user_starting_point(const unsigned gp)
{
  FOUR_C_ASSERT(gp < interp_point_container_.starting_points.size(), "GP index out of range");
  FOUR_C_ASSERT_ALWAYS(
      params_.estimate_interpolation.starting_point_type == AEI::StartingPointType::constant,
      "Setter should only be called for user-set starting points, not for {}",
      EnumTools::enum_name(params_.estimate_interpolation.starting_point_type));
  FOUR_C_ASSERT(params_.estimate_interpolation.user_set_starting_point.has_value(),
      "User-set starting point not specified!");

  interp_point_container_.starting_points[gp] =
      params_.estimate_interpolation.user_set_starting_point.value();
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::set_stress_based_starting_point(
    const unsigned gp, InputEquivStressStartingPoint input_equiv_stress_starting_point)
{
  FOUR_C_ASSERT(gp < interp_point_container_.starting_points.size(), "GP index out of range");
  FOUR_C_ASSERT_ALWAYS(params_.estimate_interpolation.starting_point_type ==
                           AEI::StartingPointType::equiv_stress_history,
      "Setter should only be called for stress-based starting points, not for {}",
      EnumTools::enum_name(params_.estimate_interpolation.starting_point_type));

  interp_point_container_.starting_points[gp] =
      calculate_equiv_stress_starting_point(input_equiv_stress_starting_point);
}

FOUR_C_NAMESPACE_CLOSE
