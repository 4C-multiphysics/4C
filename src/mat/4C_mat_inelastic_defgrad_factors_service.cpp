// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_mat_inelastic_defgrad_factors_service.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor_generators.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <string>
#include <tuple>


FOUR_C_NAMESPACE_OPEN

using namespace Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;


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
  // set consistent last substep values
  last_substep_plastic_defgrad_inverse[gp] = last_plastic_defgrad_inverse[gp];
  last_substep_plastic_strain[gp] = last_plastic_strain[gp];
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::update()
{
  // update history variables for the next time step
  last_defgrad = current_defgrad;
  last_rightCG = current_rightCG;
  last_plastic_defgrad_inverse = current_plastic_defgrad_inverse;
  last_substep_plastic_defgrad_inverse = current_plastic_defgrad_inverse;
  last_plastic_strain = current_plastic_strain;
  last_equiv_stress = current_equiv_stress;
  last_substep_plastic_strain = current_plastic_strain;
  last_temperature = current_temperature;
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

  // set initial number of iterations to 0
  iter_ = 0;
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
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::reset()
{
  std::ranges::fill(curr_num_iters_, 0);
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



FOUR_C_NAMESPACE_CLOSE
