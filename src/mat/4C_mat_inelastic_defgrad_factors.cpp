// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_inelastic_defgrad_factors.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_legacy_enum_definitions_materials.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_utils_densematrix_funct.hpp"
#include "4C_linalg_utils_scalar_interpolation.hpp"
#include "4C_linalg_utils_tensor_interpolation.hpp"
#include "4C_mat_elast_couptransverselyisotropic.hpp"
#include "4C_mat_elasthyper_service.hpp"
#include "4C_mat_electrode.hpp"
#include "4C_mat_inelastic_defgrad_factors_service.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper_service.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_mat_vplast_law.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <utility>

FOUR_C_NAMESPACE_OPEN
namespace
{
  namespace ViscoplastUtils = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;

  // declare file-scope instance of the constant non-material tensors
  static ViscoplastUtils::ConstNonMatTensors const_non_mat_tensors =
      ViscoplastUtils::ConstNonMatTensors::instance();

  double defgrad_difference_norm(
      const Core::LinAlg::Matrix<3, 3>& lhs, const Core::LinAlg::Matrix<3, 3>& rhs)
  {
    Core::LinAlg::Matrix<3, 3> diff{Core::LinAlg::Initialization::zero};
    diff.update(1.0, lhs, -1.0, rhs, 0.0);
    return diff.norm2();
  }

  /// Holds Kinematic quantities reduced by the preceding inelastic factors
  struct ReducedKinematics
  {
    /// The deformation gradient reduced by the preceding inelastic factors
    Core::LinAlg::Matrix<3, 3> defgrad{Core::LinAlg::Initialization::zero};
    /// The right cauchy gree tensor obtained from the reduced deformation gradient
    Core::LinAlg::Matrix<3, 3> right_cauchy_green{Core::LinAlg::Initialization::zero};
  };

  ReducedKinematics evaluate_reduced_kinematics(const Core::LinAlg::Matrix<3, 3>& defgrad,
      const Core::LinAlg::Matrix<3, 3>& inverse_other_inelastic_defgrad)
  {
    ReducedKinematics result;
    result.defgrad.multiply_nn(1.0, defgrad, inverse_other_inelastic_defgrad, 0.0);
    result.right_cauchy_green.multiply_tn(1.0, result.defgrad, result.defgrad, 0.0);
    return result;
  }

  // read input parameter container of parent material (i.e., underlying
  // multiplicative_split_defgrad_elasthyper material)
  Core::IO::InputParameterContainer get_parameters_of_parent_material(const int mat_id,
      const std::map<int, Core::Utils::LazyPtr<Core::Mat::PAR::Parameter>>& material_map)
  {
    // go over material map
    for (const auto& material_parameters : material_map | std::views::values)
    {
      // check type of parent material
      if (material_parameters->type() == Core::Materials::m_multiplicative_split_defgrad_elasthyper)
      {
        auto inel_defgrad_facids_ =
            material_parameters->raw_parameters().get<std::vector<int>>("INELDEFGRADFACIDS");
        // check if the inelastic defgrad factor ids of the found multiplicative split material
        // contain the id of the considered factor
        if (std::ranges::find(inel_defgrad_facids_.begin(), inel_defgrad_facids_.end(), mat_id) !=
            inel_defgrad_facids_.end())
        {
          return material_parameters->raw_parameters();
        }
      }
    }

    FOUR_C_THROW("No parent material found for inelastic defgrad factor ID {}", mat_id);
  }

  // assemble Jacobian from components (helper function:
  // InelasticDefgradTransvIsotropElastViscoplast)
  Core::LinAlg::Matrix<10, 10> assemble_jacobian_from_components(
      const Core::LinAlg::Matrix<9, 9>& J_FdF, const Core::LinAlg::Matrix<9, 1>& J_FdS,
      const Core::LinAlg::Matrix<1, 9>& J_SdF, const double J_SdS)
  {
    // declare output Jacobian
    Core::LinAlg::Matrix<10, 10> J(Core::LinAlg::Initialization::zero);

    // set its components
    J(0, 0) = J_FdF(0, 0);
    J(0, 1) = J_FdF(0, 1);
    J(0, 2) = J_FdF(0, 2);
    J(0, 3) = J_FdF(0, 3);
    J(0, 4) = J_FdF(0, 4);
    J(0, 5) = J_FdF(0, 5);
    J(0, 6) = J_FdF(0, 6);
    J(0, 7) = J_FdF(0, 7);
    J(0, 8) = J_FdF(0, 8);
    J(0, 9) = J_FdS(0, 0);

    J(1, 0) = J_FdF(1, 0);
    J(1, 1) = J_FdF(1, 1);
    J(1, 2) = J_FdF(1, 2);
    J(1, 3) = J_FdF(1, 3);
    J(1, 4) = J_FdF(1, 4);
    J(1, 5) = J_FdF(1, 5);
    J(1, 6) = J_FdF(1, 6);
    J(1, 7) = J_FdF(1, 7);
    J(1, 8) = J_FdF(1, 8);
    J(1, 9) = J_FdS(1, 0);

    J(2, 0) = J_FdF(2, 0);
    J(2, 1) = J_FdF(2, 1);
    J(2, 2) = J_FdF(2, 2);
    J(2, 3) = J_FdF(2, 3);
    J(2, 4) = J_FdF(2, 4);
    J(2, 5) = J_FdF(2, 5);
    J(2, 6) = J_FdF(2, 6);
    J(2, 7) = J_FdF(2, 7);
    J(2, 8) = J_FdF(2, 8);
    J(2, 9) = J_FdS(2, 0);

    J(3, 0) = J_FdF(3, 0);
    J(3, 1) = J_FdF(3, 1);
    J(3, 2) = J_FdF(3, 2);
    J(3, 3) = J_FdF(3, 3);
    J(3, 4) = J_FdF(3, 4);
    J(3, 5) = J_FdF(3, 5);
    J(3, 6) = J_FdF(3, 6);
    J(3, 7) = J_FdF(3, 7);
    J(3, 8) = J_FdF(3, 8);
    J(3, 9) = J_FdS(3, 0);

    J(4, 0) = J_FdF(4, 0);
    J(4, 1) = J_FdF(4, 1);
    J(4, 2) = J_FdF(4, 2);
    J(4, 3) = J_FdF(4, 3);
    J(4, 4) = J_FdF(4, 4);
    J(4, 5) = J_FdF(4, 5);
    J(4, 6) = J_FdF(4, 6);
    J(4, 7) = J_FdF(4, 7);
    J(4, 8) = J_FdF(4, 8);
    J(4, 9) = J_FdS(4, 0);

    J(5, 0) = J_FdF(5, 0);
    J(5, 1) = J_FdF(5, 1);
    J(5, 2) = J_FdF(5, 2);
    J(5, 3) = J_FdF(5, 3);
    J(5, 4) = J_FdF(5, 4);
    J(5, 5) = J_FdF(5, 5);
    J(5, 6) = J_FdF(5, 6);
    J(5, 7) = J_FdF(5, 7);
    J(5, 8) = J_FdF(5, 8);
    J(5, 9) = J_FdS(5, 0);

    J(6, 0) = J_FdF(6, 0);
    J(6, 1) = J_FdF(6, 1);
    J(6, 2) = J_FdF(6, 2);
    J(6, 3) = J_FdF(6, 3);
    J(6, 4) = J_FdF(6, 4);
    J(6, 5) = J_FdF(6, 5);
    J(6, 6) = J_FdF(6, 6);
    J(6, 7) = J_FdF(6, 7);
    J(6, 8) = J_FdF(6, 8);
    J(6, 9) = J_FdS(6, 0);

    J(7, 0) = J_FdF(7, 0);
    J(7, 1) = J_FdF(7, 1);
    J(7, 2) = J_FdF(7, 2);
    J(7, 3) = J_FdF(7, 3);
    J(7, 4) = J_FdF(7, 4);
    J(7, 5) = J_FdF(7, 5);
    J(7, 6) = J_FdF(7, 6);
    J(7, 7) = J_FdF(7, 7);
    J(7, 8) = J_FdF(7, 8);
    J(7, 9) = J_FdS(7, 0);

    J(8, 0) = J_FdF(8, 0);
    J(8, 1) = J_FdF(8, 1);
    J(8, 2) = J_FdF(8, 2);
    J(8, 3) = J_FdF(8, 3);
    J(8, 4) = J_FdF(8, 4);
    J(8, 5) = J_FdF(8, 5);
    J(8, 6) = J_FdF(8, 6);
    J(8, 7) = J_FdF(8, 7);
    J(8, 8) = J_FdF(8, 8);
    J(8, 9) = J_FdS(8, 0);

    J(9, 0) = J_SdF(0, 0);
    J(9, 1) = J_SdF(0, 1);
    J(9, 2) = J_SdF(0, 2);
    J(9, 3) = J_SdF(0, 3);
    J(9, 4) = J_SdF(0, 4);
    J(9, 5) = J_SdF(0, 5);
    J(9, 6) = J_SdF(0, 6);
    J(9, 7) = J_SdF(0, 7);
    J(9, 8) = J_SdF(0, 8);
    J(9, 9) = J_SdS;

    return J;
  }

  // assemble additional Cmat RHS (helper function: InelasticDefgradTransvIsotropElastViscoplast)
  Core::LinAlg::Matrix<10, 6> assemble_rhs_additional_cmat(
      const Core::LinAlg::Matrix<9, 6>& min_dResFdCV,
      const Core::LinAlg::Matrix<1, 6>& min_dResSdCV)
  {
    // declare output matrix
    Core::LinAlg::Matrix<10, 6> B(Core::LinAlg::Initialization::zero);

    // set its components
    B(0, 0) = min_dResFdCV(0, 0);
    B(0, 1) = min_dResFdCV(0, 1);
    B(0, 2) = min_dResFdCV(0, 2);
    B(0, 3) = min_dResFdCV(0, 3);
    B(0, 4) = min_dResFdCV(0, 4);
    B(0, 5) = min_dResFdCV(0, 5);

    B(1, 0) = min_dResFdCV(1, 0);
    B(1, 1) = min_dResFdCV(1, 1);
    B(1, 2) = min_dResFdCV(1, 2);
    B(1, 3) = min_dResFdCV(1, 3);
    B(1, 4) = min_dResFdCV(1, 4);
    B(1, 5) = min_dResFdCV(1, 5);

    B(2, 0) = min_dResFdCV(2, 0);
    B(2, 1) = min_dResFdCV(2, 1);
    B(2, 2) = min_dResFdCV(2, 2);
    B(2, 3) = min_dResFdCV(2, 3);
    B(2, 4) = min_dResFdCV(2, 4);
    B(2, 5) = min_dResFdCV(2, 5);

    B(3, 0) = min_dResFdCV(3, 0);
    B(3, 1) = min_dResFdCV(3, 1);
    B(3, 2) = min_dResFdCV(3, 2);
    B(3, 3) = min_dResFdCV(3, 3);
    B(3, 4) = min_dResFdCV(3, 4);
    B(3, 5) = min_dResFdCV(3, 5);

    B(4, 0) = min_dResFdCV(4, 0);
    B(4, 1) = min_dResFdCV(4, 1);
    B(4, 2) = min_dResFdCV(4, 2);
    B(4, 3) = min_dResFdCV(4, 3);
    B(4, 4) = min_dResFdCV(4, 4);
    B(4, 5) = min_dResFdCV(4, 5);

    B(5, 0) = min_dResFdCV(5, 0);
    B(5, 1) = min_dResFdCV(5, 1);
    B(5, 2) = min_dResFdCV(5, 2);
    B(5, 3) = min_dResFdCV(5, 3);
    B(5, 4) = min_dResFdCV(5, 4);
    B(5, 5) = min_dResFdCV(5, 5);

    B(6, 0) = min_dResFdCV(6, 0);
    B(6, 1) = min_dResFdCV(6, 1);
    B(6, 2) = min_dResFdCV(6, 2);
    B(6, 3) = min_dResFdCV(6, 3);
    B(6, 4) = min_dResFdCV(6, 4);
    B(6, 5) = min_dResFdCV(6, 5);

    B(7, 0) = min_dResFdCV(7, 0);
    B(7, 1) = min_dResFdCV(7, 1);
    B(7, 2) = min_dResFdCV(7, 2);
    B(7, 3) = min_dResFdCV(7, 3);
    B(7, 4) = min_dResFdCV(7, 4);
    B(7, 5) = min_dResFdCV(7, 5);

    B(8, 0) = min_dResFdCV(8, 0);
    B(8, 1) = min_dResFdCV(8, 1);
    B(8, 2) = min_dResFdCV(8, 2);
    B(8, 3) = min_dResFdCV(8, 3);
    B(8, 4) = min_dResFdCV(8, 4);
    B(8, 5) = min_dResFdCV(8, 5);

    B(9, 0) = min_dResSdCV(0, 0);
    B(9, 1) = min_dResSdCV(0, 1);
    B(9, 2) = min_dResSdCV(0, 2);
    B(9, 3) = min_dResSdCV(0, 3);
    B(9, 4) = min_dResSdCV(0, 4);
    B(9, 5) = min_dResSdCV(0, 5);

    return B;
  }

  // extract the derivative of the inverse inelastic defgrad w.r.t. right CG tensor from the
  // solution of the linear system of equations. This SoE is used in the additional cmat
  // calculation. (helper function: InelasticDefgradTransvIsotropElastViscoplast)
  Core::LinAlg::Matrix<9, 6> extract_derivative_of_inv_inelastic_defgrad(
      const Core::LinAlg::Matrix<10, 6>& SOL)
  {
    // declare output derivative
    Core::LinAlg::Matrix<9, 6> diFin_dC_V(Core::LinAlg::Initialization::zero);

    // set its components
    diFin_dC_V(0, 0) = SOL(0, 0);
    diFin_dC_V(0, 1) = SOL(0, 1);
    diFin_dC_V(0, 2) = SOL(0, 2);
    diFin_dC_V(0, 3) = SOL(0, 3);
    diFin_dC_V(0, 4) = SOL(0, 4);
    diFin_dC_V(0, 5) = SOL(0, 5);

    diFin_dC_V(1, 0) = SOL(1, 0);
    diFin_dC_V(1, 1) = SOL(1, 1);
    diFin_dC_V(1, 2) = SOL(1, 2);
    diFin_dC_V(1, 3) = SOL(1, 3);
    diFin_dC_V(1, 4) = SOL(1, 4);
    diFin_dC_V(1, 5) = SOL(1, 5);

    diFin_dC_V(2, 0) = SOL(2, 0);
    diFin_dC_V(2, 1) = SOL(2, 1);
    diFin_dC_V(2, 2) = SOL(2, 2);
    diFin_dC_V(2, 3) = SOL(2, 3);
    diFin_dC_V(2, 4) = SOL(2, 4);
    diFin_dC_V(2, 5) = SOL(2, 5);

    diFin_dC_V(3, 0) = SOL(3, 0);
    diFin_dC_V(3, 1) = SOL(3, 1);
    diFin_dC_V(3, 2) = SOL(3, 2);
    diFin_dC_V(3, 3) = SOL(3, 3);
    diFin_dC_V(3, 4) = SOL(3, 4);
    diFin_dC_V(3, 5) = SOL(3, 5);

    diFin_dC_V(4, 0) = SOL(4, 0);
    diFin_dC_V(4, 1) = SOL(4, 1);
    diFin_dC_V(4, 2) = SOL(4, 2);
    diFin_dC_V(4, 3) = SOL(4, 3);
    diFin_dC_V(4, 4) = SOL(4, 4);
    diFin_dC_V(4, 5) = SOL(4, 5);

    diFin_dC_V(5, 0) = SOL(5, 0);
    diFin_dC_V(5, 1) = SOL(5, 1);
    diFin_dC_V(5, 2) = SOL(5, 2);
    diFin_dC_V(5, 3) = SOL(5, 3);
    diFin_dC_V(5, 4) = SOL(5, 4);
    diFin_dC_V(5, 5) = SOL(5, 5);

    diFin_dC_V(6, 0) = SOL(6, 0);
    diFin_dC_V(6, 1) = SOL(6, 1);
    diFin_dC_V(6, 2) = SOL(6, 2);
    diFin_dC_V(6, 3) = SOL(6, 3);
    diFin_dC_V(6, 4) = SOL(6, 4);
    diFin_dC_V(6, 5) = SOL(6, 5);

    diFin_dC_V(7, 0) = SOL(7, 0);
    diFin_dC_V(7, 1) = SOL(7, 1);
    diFin_dC_V(7, 2) = SOL(7, 2);
    diFin_dC_V(7, 3) = SOL(7, 3);
    diFin_dC_V(7, 4) = SOL(7, 4);
    diFin_dC_V(7, 5) = SOL(7, 5);

    diFin_dC_V(8, 0) = SOL(8, 0);
    diFin_dC_V(8, 1) = SOL(8, 1);
    diFin_dC_V(8, 2) = SOL(8, 2);
    diFin_dC_V(8, 3) = SOL(8, 3);
    diFin_dC_V(8, 4) = SOL(8, 4);
    diFin_dC_V(8, 5) = SOL(8, 5);

    return diFin_dC_V;
  }

  /**
   * @brief Extract the derivative of the plastic strain w.r.t. right CG tensor from the
   solution of the linear system of equations arising in
   evaluate_history_variables_wrt_cauchy_green.
   *
   * @param SOL Solution arising from the linear system of equations in
   evaluate_history_variables_wrt_cauchy_green
   * @return Core::LinAlg::Matrix<1, 6> \f[ \frac{\mathrm{d} \varepsilon_p}{\mathrm{d}
   \boldsymbol{C}}
   \f] in Voigt Notation
   */
  Core::LinAlg::Matrix<1, 6> extract_derivative_of_plastic_strain(
      const Core::LinAlg::Matrix<10, 6>& SOL)
  {
    // declare output derivative
    Core::LinAlg::Matrix<1, 6> depsp_dC_V(Core::LinAlg::Initialization::zero);

    // set its components
    depsp_dC_V(0, 0) = SOL(9, 0);
    depsp_dC_V(0, 1) = SOL(9, 1);
    depsp_dC_V(0, 2) = SOL(9, 2);
    depsp_dC_V(0, 3) = SOL(9, 3);
    depsp_dC_V(0, 4) = SOL(9, 4);
    depsp_dC_V(0, 5) = SOL(9, 5);

    return depsp_dC_V;
  }

  // wrap inverse inelastic defgrad and plastic strain to a vector of unknowns for the Local
  // Newton Loop (helper function: InelasticDefgradTransvIsotropElastViscoplast)
  Core::LinAlg::Matrix<10, 1> wrap_unknowns(
      const Core::LinAlg::Matrix<3, 3>& iFinM, const double& plastic_strain)
  {
    Core::LinAlg::Matrix<10, 1> x(Core::LinAlg::Initialization::zero);
    x(0) = iFinM(0, 0);
    x(1) = iFinM(1, 1);
    x(2) = iFinM(2, 2);
    x(3) = iFinM(0, 1);
    x(4) = iFinM(1, 2);
    x(5) = iFinM(0, 2);
    x(6) = iFinM(1, 0);
    x(7) = iFinM(2, 1);
    x(8) = iFinM(2, 0);
    x(9) = plastic_strain;

    return x;
  }

  // extract the inverse inelastic defgrad from the vector of unknowns used in the Local Newton
  // Loop (helper function: InelasticDefgradTransvIsotropElastViscoplast)
  Core::LinAlg::Matrix<3, 3> extract_inverse_inelastic_defgrad(const Core::LinAlg::Matrix<10, 1>& x)
  {
    Core::LinAlg::Matrix<3, 3> iFinM(Core::LinAlg::Initialization::zero);
    iFinM(0, 0) = x(0);
    iFinM(1, 1) = x(1);
    iFinM(2, 2) = x(2);
    iFinM(0, 1) = x(3);
    iFinM(1, 2) = x(4);
    iFinM(0, 2) = x(5);
    iFinM(1, 0) = x(6);
    iFinM(2, 1) = x(7);
    iFinM(2, 0) = x(8);


    return iFinM;
  }

  // Initialize the second-order tensor interpolator for
  // InelasticDefgradTransvIsotropElastViscoplast Currently, we consider R - LOG interpolation with
  // a set exponential decay factor.
  Core::LinAlg::SecondOrderTensorInterpolator<1> init_tensor_interpolator()
  {
    // initialize interpolation parameter list
    Core::LinAlg::ScalarInterpolationParams interp_param_list;
    /// add exponential decay factor for weighting
    interp_param_list.exponential_decay_c = 20.0;

    // return corresponding tensor interpolator
    return Core::LinAlg::SecondOrderTensorInterpolator<1>{1,
        Core::LinAlg::RotationInterpolationType::RotationVector,
        Core::LinAlg::EigenvalInterpolationType::LOG, interp_param_list};
  }

  Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams
  retrieve_local_newton_params(const Core::Mat::PAR::Parameter::Data& matdata)
  {
    const auto local_newton_params = ViscoplastUtils::LocalNewtonParams{
        .res_tol = matdata.parameters.group("LOCAL_NEWTON").get<double>("RES_TOL"),
        .incr_tol = matdata.parameters.group("LOCAL_NEWTON").get<double>("INCR_TOL"),
        .conv_check = matdata.parameters.group("LOCAL_NEWTON")
            .get<ViscoplastUtils::LocalNewtonConvCheck>("CONV_CHECK"),
        .diver_cont = matdata.parameters.group("LOCAL_NEWTON")
            .get<ViscoplastUtils::LocalNewtonDiverCont>("DIVER_CONT"),
        .max_iter = static_cast<unsigned int>(
            matdata.parameters.group("LOCAL_NEWTON").get<int>("MAX_ITER")),
        .max_exceedance_fact_res_tol =
            matdata.parameters.group("LOCAL_NEWTON").get<double>("MAX_EXCEEDANCE_FACT_RES_TOL"),
        .max_exceedance_fact_incr_tol =
            matdata.parameters.group("LOCAL_NEWTON").get<double>("MAX_EXCEEDANCE_FACT_INCR_TOL"),
    };

    return local_newton_params;
  }

  bool show_warnings(const unsigned int ele_gid)
  {
    // get structure discretization
    const auto structure_dis = Global::Problem::instance()->does_exist_dis("structure")
                                   ? Global::Problem::instance()->get_dis("structure")
                                   : nullptr;
    if (structure_dis == nullptr)
    {
      // We display a warning if the structure discretization could not be found. For instance, this
      // is currently the case for our unit tests, and we don't want to construct strange dummy
      // discretizations only for the warning and error messages in these tests.
      // In normal runs / simulations, these warnings signal to the user that something may be wrong
      // with the simulated problem if this discretization cannot be found.
      std::cout << "Discretization 'structure' could not be detected by "
                   "InelasticDefgradTransvIsotropElastViscoplast! Hence, warnings and error "
                   "messages via std::cout will not be shown! \n";
      return false;
    }
    else
    {
      FOUR_C_ASSERT_ALWAYS(structure_dis->have_global_element(ele_gid),
          "Inconsistency: the element with global id {} cannot be found in the structure "
          "discretization!",
          ele_gid);

      return structure_dis->element_row_map()->lid(ele_gid) >= 0;
    }
  }
}  // namespace



/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradNoGrowth::InelasticDefgradNoGrowth(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata)
{
  // do nothing here
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradScalar::InelasticDefgradScalar(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      scalar1_(matdata.parameters.get<int>("SCALAR1")),
      scalar1_ref_conc_(matdata.parameters.get<double>("SCALAR1_RefConc"))
{
  // safety checks
  // in case not all scatra dofs are transported scalars, the last scatra dof is a potential and
  // can not be treated as a concentration but it is treated like that in so3_scatra_evaluate.cpp
  // in the pre_evaluate method!
  if (scalar1_ != 1) FOUR_C_THROW("At the moment it is only possible that SCALAR1 induces growth");
  if (matdata.parameters.get<double>("SCALAR1_RefConc") < 0.0)
    FOUR_C_THROW("The reference concentration of SCALAR1 can't be negative");
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradLinScalar::InelasticDefgradLinScalar(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : InelasticDefgradScalar(matdata),
      scalar1_molar_growth_fac_(matdata.parameters.get<double>("SCALAR1_MolarGrowthFac"))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradIntercalFrac::InelasticDefgradIntercalFrac(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : InelasticDefgradScalar(matdata)
{
  // get matid
  const int matid = matdata.parameters.get<int>("MATID");

  // Check if the material specified by user with MATID is an electrode material
  if (matid > 0)
  {
    // retrieve problem instance to read from
    const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
    // retrieve validated input line of material ID in question
    const auto* current_material =
        Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
    switch (current_material->type())
    {
      case Core::Materials::m_electrode:
      {
        // Get C_max and Chi_max of electrode material
        c_max_ = current_material->raw_parameters().get<double>("C_MAX");
        chi_max_ = current_material->raw_parameters().get<double>("CHI_MAX");
        break;
      }
      default:
        FOUR_C_THROW("The material you have specified by MATID has to be an electrode material!");
    }
  }
  else
  {
    FOUR_C_THROW("You have to enter a valid MATID for the corresponding electrode material!");
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradPolyIntercalFrac::InelasticDefgradPolyIntercalFrac(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : InelasticDefgradIntercalFrac(matdata),
      poly_coeffs_(matdata.parameters.get<std::vector<double>>("POLY_PARAMS")),
      x_max_(matdata.parameters.get<double>("X_max")),
      x_min_(matdata.parameters.get<double>("X_min"))
{
  // safety check
  if (poly_coeffs_.size() !=
      static_cast<unsigned int>(matdata.parameters.get<int>("POLY_PARA_NUM")))
  {
    FOUR_C_THROW(
        "Number of coefficients POLY_PARA_NUM you entered in input file has to match the size "
        "of coefficient vector POLY_PARAMS");
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradLinScalarAniso::InelasticDefgradLinScalarAniso(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : InelasticDefgradLinScalar(matdata),
      growth_dir_(InelasticDeformationDirection(
          matdata.parameters.get<std::vector<double>>("GrowthDirection")))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradPolyIntercalFracAniso::InelasticDefgradPolyIntercalFracAniso(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : InelasticDefgradPolyIntercalFrac(matdata),
      growth_dir_(InelasticDeformationDirection(
          matdata.parameters.get<std::vector<double>>("GrowthDirection")))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDeformationDirection::InelasticDeformationDirection(
    const std::vector<double>& growthdirection)
{
  FOUR_C_ASSERT_ALWAYS(growthdirection.size() == 3,
      "Since we have a 3D problem here, vector that defines the growth direction also needs to "
      "have the size 3!");

  // fill matrix that determines the growth direction
  const double growthdir_vec_norm_sq = growthdirection[0] * growthdirection[0] +
                                       growthdirection[1] * growthdirection[1] +
                                       growthdirection[2] * growthdirection[2];
  FOUR_C_ASSERT_ALWAYS(
      growthdir_vec_norm_sq > 0.0, "Growth direction vector must not be the zero vector!");

  const double inv_quadr_growthdir_vec_norm = 1.0 / growthdir_vec_norm_sq;

  // loop over all rows and columns to fill the matrix and scale it correctly on the fly
  for (unsigned i = 0; i < growthdirection.size(); ++i)
  {
    for (unsigned j = 0; j < growthdirection.size(); ++j)
    {
      growth_dir_tensor_(i, j) =
          inv_quadr_growthdir_vec_norm * growthdirection[i] * growthdirection[j];
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradLinTempIso::InelasticDefgradLinTempIso(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      ref_temp_(matdata.parameters.get<double>("RefTemp")),
      temp_growth_fac_(matdata.parameters.get<double>("Temp_GrowthFac"))
{
  // safety checks
  if (ref_temp_ < 0.0) FOUR_C_THROW("Avoid negative reference temperatures");
  if (temp_growth_fac_ == 0.0)
  {
    FOUR_C_THROW(
        "Do not use 'MAT_InelasticDefgradLinTempIso' with a growth factor of 0.0. Use "
        "'MAT_InelasticDefgradNoGrowth' instead!");
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradTransvIsotropElastViscoplast::
    InelasticDefgradTransvIsotropElastViscoplast(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      viscoplastic_law_id_(matdata.parameters.get<int>("VISCOPLAST_LAW_ID")),
      taylor_quinney_coefficient_(matdata.parameters.get<double>("TAYLOR_QUINNEY_COEFFICIENT")),
      fiber_reader_gid_(matdata.parameters.get<int>("FIBER_READER_ID")),
      yield_cond_a_(matdata.parameters.get<std::optional<double>>("YIELD_COND_A").value_or(0.0)),
      yield_cond_b_(matdata.parameters.get<std::optional<double>>("YIELD_COND_B").value_or(0.0)),
      yield_cond_f_(matdata.parameters.get<std::optional<double>>("YIELD_COND_F").value_or(0.0)),
      mat_behavior_(matdata.parameters.get<ViscoplastUtils::MatBehavior>("MAT_BEHAVIOR")),
      timint_type_(
          matdata.parameters.get<ViscoplastUtils::TimIntType>("TIME_INTEGRATION_HIST_VARS")),
      linearization_type_(
          matdata.parameters.get<ViscoplastUtils::LinearizationType>("LINEARIZATION")),
      max_plastic_strain_incr_(matdata.parameters.get<double>("MAX_PLASTIC_STRAIN_INCR")),
      max_plastic_strain_deriv_incr_(
          matdata.parameters.get<double>("MAX_PLASTIC_STRAIN_DERIV_INCR")),
      use_local_substepping_(
          matdata.parameters.group("LOCAL_SUBSTEPPING").get<bool>("USE_SUBSTEPPING")),
      max_local_substepping_halve_num_(static_cast<unsigned int>(
          matdata.parameters.group("LOCAL_SUBSTEPPING").get<int>("MAX_SUBSTEPPING_HALVE_NUM"))),
      mat_exp_calc_method_(
          matdata.parameters.get<Core::LinAlg::MatrixExpCalcMethod>("MATRIX_EXP_CALC_METHOD")),
      mat_exp_deriv_calc_method_(
          matdata.parameters.get<Core::LinAlg::GenMatrixExpFirstDerivCalcMethod>(
              "MATRIX_EXP_DERIV_CALC_METHOD")),
      mat_log_calc_method_(
          matdata.parameters.get<Core::LinAlg::MatrixLogCalcMethod>("MATRIX_LOG_CALC_METHOD")),
      mat_log_deriv_calc_method_(
          matdata.parameters.get<Core::LinAlg::GenMatrixLogFirstDerivCalcMethod>(
              "MATRIX_LOG_DERIV_CALC_METHOD")),
      local_newton_params_(retrieve_local_newton_params(matdata))
{
  // consistency check: yield parameters in case of transversely-isotropic behavior
  const bool all_yield_cond_param_specified =
      matdata.parameters.get<std::optional<double>>("YIELD_COND_A").has_value() &&
      matdata.parameters.get<std::optional<double>>("YIELD_COND_B").has_value() &&
      matdata.parameters.get<std::optional<double>>("YIELD_COND_F").has_value();
  if (mat_behavior_ ==
          InelasticDefgradTransvIsotropElastViscoplastUtils::MatBehavior::transv_isotropic &&
      !all_yield_cond_param_specified)
  {
    FOUR_C_THROW(
        "You are attempting to simulate transversely isotropic behavior but have not specified "
        "all yield function parameters!");
  }
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradFactors::InelasticDefgradFactors(Core::Mat::PAR::Parameter* params)
    : params_(params)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
std::shared_ptr<Mat::InelasticDefgradFactors> Mat::InelasticDefgradFactors::factory(int matnum)
{
  // for the sake of safety
  if (Global::Problem::instance()->materials() == nullptr)
    FOUR_C_THROW("List of materials cannot be accessed in the global problem instance.");

  // another safety check
  if (Global::Problem::instance()->materials()->num() == 0)
    FOUR_C_THROW("List of materials in the global problem instance is empty.");

  // check correct masslin type
  const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
  if (Teuchos::getIntegralValue<Inpar::Solid::MassLin>(sdyn, "MASSLIN") !=
      Inpar::Solid::MassLin::ml_none)
  {
    FOUR_C_THROW(
        "If you use the material 'InelasticDefgradFactors' please set 'MASSLIN' in the "
        "STRUCTURAL DYNAMIC Section to 'None', or feel free to implement other "
        "possibilities!");
  }

  // retrieve problem instance to read from
  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
  // retrieve validated input line of material ID in question
  auto* current_material =
      Global::Problem::instance(probinst)->materials()->parameter_by_id(matnum);

  // get material type and call corresponding constructors
  switch (current_material->type())
  {
    case Core::Materials::mfi_no_growth:
    {
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradNoGrowth*>(current_material);

      return std::make_shared<InelasticDefgradNoGrowth>(params);
    }
    case Core::Materials::mfi_lin_scalar_aniso:
    {
      // get pointer to parameter class
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradLinScalarAniso*>(current_material);

      // return pointer to inelastic deformation gradient object
      return std::make_shared<InelasticDefgradLinScalarAniso>(params);
    }
    case Core::Materials::mfi_lin_scalar_iso:
    {
      // get pointer to parameter class
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradScalar*>(current_material);

      // return pointer to inelastic deformation gradient object
      return std::make_shared<InelasticDefgradLinScalarIso>(params);
    }
    case Core::Materials::mfi_poly_intercal_frac_aniso:
    {
      // get pointer to parameter class
      auto* params =
          dynamic_cast<Mat::PAR::InelasticDefgradPolyIntercalFracAniso*>(current_material);

      // return pointer to inelastic deformation gradient object
      return std::make_shared<InelasticDefgradPolyIntercalFracAniso>(params);
    }
    case Core::Materials::mfi_poly_intercal_frac_iso:
    {
      // get pointer to parameter class
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradPolyIntercalFrac*>(current_material);

      // return pointer to inelastic deformation gradient object
      return std::make_shared<InelasticDefgradPolyIntercalFracIso>(params);
    }

    case Core::Materials::mfi_lin_temp_iso:
    {
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradLinTempIso*>(current_material);
      return std::make_shared<InelasticDefgradLinTempIso>(params);
    }
    case Core::Materials::mfi_time_funct_aniso:
    {
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradTimeFunctAniso*>(current_material);
      return std::make_shared<InelasticDefgradTimeFunctAniso>(params);
    }
    case Core::Materials::mfi_time_funct_iso:
    {
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradTimeFunct*>(current_material);
      return std::make_shared<InelasticDefgradTimeFunctIso>(params);
    }
    case Core::Materials::mfi_transv_isotrop_elast_viscoplast:
    {
      // read material map of the global problem
      std::map<int, Core::Utils::LazyPtr<Core::Mat::PAR::Parameter>> material_map =
          Global::Problem::instance(probinst)->materials()->map();

      // retrieve parameter container of parent material
      Core::IO::InputParameterContainer parentmat_input_params =
          get_parameters_of_parent_material(matnum, material_map);

      // create parameter class
      auto* params =
          dynamic_cast<Mat::PAR::InelasticDefgradTransvIsotropElastViscoplast*>(current_material);

      // create viscoplastic law
      auto viscoplastic_law = Mat::Viscoplastic::Law::factory(params->viscoplastic_law_id());

      // construct fiber reader
      auto* fiber_reader_params = Global::Problem::instance(probinst)->materials()->parameter_by_id(
          params->fiber_reader_gid());
      FOUR_C_ASSERT_ALWAYS(
          fiber_reader_params->type() == Core::Materials::mes_couptransverselyisotropic,
          "Provided fiber reader material is not of the correct type (hyperelastic, transversely "
          "isotropic: ELAST_CoupTransverselyIsotropic)!");
      Mat::Elastic::CoupTransverselyIsotropic fiber_reader{
          dynamic_cast<Mat::Elastic::PAR::CoupTransverselyIsotropic*>(fiber_reader_params)};

      // retrieve elastic materials
      std::vector<std::shared_ptr<Mat::Elastic::Summand>> potsumel;
      std::vector<std::shared_ptr<Mat::Elastic::CoupTransverselyIsotropic>> potsumel_transviso;
      for (const int matid_elastic : parentmat_input_params.get<std::vector<int>>("MATIDSEL"))
      {
        // create elastic component
        auto elastic_summand = Mat::Elastic::Summand::factory(matid_elastic);
        FOUR_C_ASSERT_ALWAYS(elastic_summand != nullptr, "Failed to allocate");
        // add to the list of elastic components
        if (elastic_summand->material_type() == Core::Materials::mes_couptransverselyisotropic)
        {
          potsumel_transviso.push_back(
              std::dynamic_pointer_cast<Mat::Elastic::CoupTransverselyIsotropic>(elastic_summand));
        }
        else
        {
          potsumel.push_back(elastic_summand);
        }
      }

      const double thermal_expansion_coefficient =
          parentmat_input_params.get<double>("THERMAL_EXPANSION_COEFFICIENT");

      const auto ref_temperature = parentmat_input_params.get<double>("REF_TEMPERATURE");

      // return shared pointer to the inelastic factor
      return std::make_shared<InelasticDefgradTransvIsotropElastViscoplast>(params,
          viscoplastic_law, fiber_reader, potsumel, potsumel_transviso,
          thermal_expansion_coefficient, ref_temperature);
    }

    default:
      FOUR_C_THROW("cannot deal with type {}", current_material->type());
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradScalar::InelasticDefgradScalar(Core::Mat::PAR::Parameter* params)
    : InelasticDefgradFactors(params), concentrations_(nullptr)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradScalar::pre_evaluate(const Teuchos::ParameterList& params,
    const EvaluationContext<3>& context, const int gp, const int eleGID)
{
  // store scalars of current gauss point
  concentrations_ = params.get<std::shared_ptr<std::vector<double>>>("scalars");
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradScalar::set_concentration_gp(const double concentration)
{
  const int scalar1 = parameter()->scalar1();
  concentrations_->at(scalar1 - 1) = concentration;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradPolyIntercalFrac::InelasticDefgradPolyIntercalFrac(
    Core::Mat::PAR::Parameter* params)
    : InelasticDefgradScalar(params),
      polynomial_growth_(InelasticDefgradPolynomialShape(
          parameter()->poly_coeffs(), parameter()->x_min(), parameter()->x_max()))
{
  // get reference intercalation fraction
  const double x_ref = Mat::Electrode::compute_intercalation_fraction(
      parameter()->scalar1_ref_conc(), parameter()->chimax(), parameter()->cmax(), 1.0);

  // set the polynomial value in the reference configuration
  parameter()->set_polynom_reference_value(polynomial_growth_.compute_polynomial(x_ref));
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::InelasticDefgradPolyIntercalFrac::evaluate_polynomial(
    const double concentration, const double detjacobian) const
{
  // get intercalation fraction
  const double x = Mat::Electrode::compute_intercalation_fraction(
      concentration, parameter()->chimax(), parameter()->cmax(), detjacobian);

  // check bounds of validity of polynomial
  polynomial_growth_.check_polynomial_bounds(x);

  // calculate and return the value of the polynomial
  return polynomial_growth_.compute_polynomial(x);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::InelasticDefgradPolyIntercalFrac::evaluate_polynomial_derivative(
    const double concentration, const double detjacobian) const
{
  // get intercalation fraction
  const double x = Mat::Electrode::compute_intercalation_fraction(
      concentration, parameter()->chimax(), parameter()->cmax(), detjacobian);

  return polynomial_growth_.compute_polynomial_derivative(x);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticSource Mat::InelasticDefgradPolyIntercalFrac::get_inelastic_source()
{
  return Mat::PAR::InelasticSource::concentration;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradLinScalarIso::InelasticDefgradLinScalarIso(Core::Mat::PAR::Parameter* params)
    : InelasticDefgradScalar(params),
      linear_growth_(InelasticDefgradLinearShape(
          parameter()->scalar1_molar_growth_fac(), parameter()->scalar1_ref_conc()))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticSource Mat::InelasticDefgradLinScalarIso::get_inelastic_source()
{
  return Mat::PAR::InelasticSource::concentration;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarIso::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  // get parameter
  const int sc1 = parameter()->scalar1();
  const double material_concentration = get_concentration_gp().at(sc1 - 1) * defgrad->determinant();

  // get growth factor
  const double growth_factor = linear_growth_.evaluate_linear_growth(material_concentration);

  const double isoinelasticdefo = std::pow(1.0 + growth_factor, (1.0 / 3.0));

  // calculate inverse inelastic deformation gradient
  for (int i = 0; i < 3; ++i) iFinM(i, i) = 1.0 / isoinelasticdefo;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarIso::evaluate_additional_cmat(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 1>& iCV,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 6>& cmatadd)
{
  // static variables
  static Core::LinAlg::Matrix<9, 6> diFinjdC(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 1> id9x1(Core::LinAlg::Initialization::zero);

  // prepare id9x1 (identity matrix written as a 9x1 vector)
  for (int i = 0; i < 3; ++i) id9x1(i) = 1.0;

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double sc1GrowthFac = linear_growth_.growth_fac();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double detjacobian = defgrad->determinant();

  // get growth factor
  const double growth_factor = linear_growth_.evaluate_linear_growth(concentration * detjacobian);

  // evaluate scaling factor
  const double scalefac =
      -sc1GrowthFac * concentration * detjacobian / 6.0 * std::pow(1 + growth_factor, -4.0 / 3.0);

  // calculate diFindC
  diFinjdC.multiply_nt(scalefac, id9x1, iCV, 0.0);

  // cmatadd = 2 dSdiFinj : diFinjdC
  cmatadd.multiply_nn(2.0, dSdiFinj, diFinjdC, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarIso::evaluate_od_stiff_mat(
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
    Core::LinAlg::Matrix<6, 1>& dstressdc)
{
  static Core::LinAlg::Matrix<9, 1> id9x1(Core::LinAlg::Initialization::zero);
  // prepare id9x1 (identity matrix written as a 9x1 vector)
  for (int i = 0; i < 3; ++i) id9x1(i) = 1.0;

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double sc1GrowthFac = linear_growth_.growth_fac();
  const double detjacobian = defgrad->determinant();
  const double material_concentration = get_concentration_gp().at(sc1 - 1) * detjacobian;

  // get growth factor
  const double growth_factor = linear_growth_.evaluate_linear_growth(material_concentration);

  // calculate scalefac
  const double scalefac =
      -sc1GrowthFac / 3.0 * detjacobian * std::pow(1 + growth_factor, -4.0 / 3.0);

  // calculate diFindc and add contribution to dstressdc = dSdiFinj : diFinjdc
  dstressdc.multiply_nn(scalefac, dSdiFinj, id9x1, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarIso::evaluate_inelastic_def_grad_derivative(
    const double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx)
{
  // get parameters
  const int sc1 = parameter()->scalar1();
  const double material_concentration = get_concentration_gp().at(sc1 - 1) * detjacobian;

  // get growth factor
  const double growth_factor = linear_growth_.evaluate_linear_growth(material_concentration);
  // calculate the scale factor needed to calculate the derivative below
  const double scalefac = 1.0 / 3.0 * std::pow(1 + growth_factor, -2.0 / 3.0) *
                          linear_growth_.growth_fac() * detjacobian;

  dFindx =
      Core::LinAlg::get_full(scalefac * Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradLinScalarAniso::InelasticDefgradLinScalarAniso(
    Core::Mat::PAR::Parameter* params)
    : InelasticDefgradScalar(params),
      linear_growth_(InelasticDefgradLinearShape(
          parameter()->scalar1_molar_growth_fac(), parameter()->scalar1_ref_conc()))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticSource Mat::InelasticDefgradLinScalarAniso::get_inelastic_source()
{
  return Mat::PAR::InelasticSource::concentration;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarAniso::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  // init and clear variable
  static Core::LinAlg::Matrix<3, 3> FinM(Core::LinAlg::Initialization::zero);
  FinM.clear();
  Core::LinAlg::TensorView<double, 3, 3> Fin = Core::LinAlg::make_tensor_view(FinM);

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double material_concentration = get_concentration_gp().at(sc1 - 1) * defgrad->determinant();

  // get growth factor
  const double growth_factor = linear_growth_.evaluate_linear_growth(material_concentration);

  // calculate inelastic deformation gradient and its inverse
  for (int i = 0; i < 3; ++i) FinM(i, i) = 1.0;

  // finalize inelastic deformation gradient matrix (FinM is calculated, such that the volume
  // change is a linear function of the scalar (mapped to reference frame) that causes it)
  Fin += growth_factor * Core::LinAlg::get_full(parameter()->growth_dir_tensor());

  // calculate inverse of inelastic deformation gradient matrix
  iFinM.invert(FinM);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarAniso::evaluate_additional_cmat(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 1>& iCV,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 6>& cmatadd)
{
  static Core::LinAlg::Matrix<3, 3> temp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iFinjGiFinj(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 1> iFinjGiFinj9x1(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 6> diFinjdC(Core::LinAlg::Initialization::zero);

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double sc1GrowthFac = linear_growth_.growth_fac();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double detjacobian = defgrad->determinant();

  // prepare scalefac
  const double scalefac = -sc1GrowthFac * concentration * detjacobian / 2.0;

  // calculate F_{in,j}^{-1} . G . F_{in,j}^{-1} with F_{in,j}, the j-th factor of F_{in}
  temp.multiply_nn(1.0, iFinjM,
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(parameter()->growth_dir_tensor())), 0.0);
  iFinjGiFinj.multiply_nn(1.0, temp, iFinjM, 0.0);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(iFinjGiFinj, iFinjGiFinj9x1);

  // calculate diFinjdC
  diFinjdC.multiply_nt(scalefac, iFinjGiFinj9x1, iCV, 0.0);

  // cmatadd = 2 dSdiFinj : diFinjdC
  cmatadd.multiply_nn(2.0, dSdiFinj, diFinjdC, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarAniso::evaluate_od_stiff_mat(
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
    Core::LinAlg::Matrix<6, 1>& dstressdc)
{
  // static variables
  static Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> diFinjdcM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 1> diFinjdc9x1(Core::LinAlg::Initialization::zero);

  // get parameters
  const double sc1GrowthFac = linear_growth_.growth_fac();
  const double detjacobian = defgrad->determinant();

  // prepare scalefac
  const double scalefac = -sc1GrowthFac * detjacobian;

  // calculate diFinjdc
  tmp.multiply_nn(1.0, iFinjM,
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(parameter()->growth_dir_tensor())), 0.0);
  diFinjdcM.multiply_nn(scalefac, tmp, iFinjM, 0.0);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(diFinjdcM, diFinjdc9x1);

  // dstressdc = dSdiFinj : diFinjdc
  dstressdc.multiply_nn(1.0, dSdiFinj, diFinjdc9x1, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinScalarAniso::evaluate_inelastic_def_grad_derivative(
    const double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx)
{
  const double scalefac = linear_growth_.growth_fac() * detjacobian;

  dFindx = Core::LinAlg::get_full(parameter()->growth_dir_tensor());
  dFindx *= scalefac;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradPolyIntercalFracIso::InelasticDefgradPolyIntercalFracIso(
    Core::Mat::PAR::Parameter* params)
    : InelasticDefgradPolyIntercalFrac(params)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracIso::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  // get parameters
  const int sc1 = parameter()->scalar1();
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get polynomial
  const double polynomValue =
      evaluate_polynomial(get_concentration_gp().at(sc1 - 1), defgrad->determinant());

  // calculate growth
  const double isoInelasticDefo =
      std::pow((1.0 + polynomValue) / (1.0 + polynomReferenceValue), (1.0 / 3.0));
  // calculate inverse inelastic deformation gradient
  for (int i = 0; i < 3; ++i) iFinM(i, i) = 1.0 / isoInelasticDefo;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracIso::evaluate_additional_cmat(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 1>& iCV,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 6>& cmatadd)
{
  // static variables
  static Core::LinAlg::Matrix<9, 6> diFinjdC(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 1> id9x1(Core::LinAlg::Initialization::zero);

  // prepare id9x1 (identity matrix written as a 9x1 vector)
  for (int i = 0; i < 3; ++i) id9x1(i) = 1.0;

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double chi_max = parameter()->chimax();
  const double c_max = parameter()->cmax();
  const double detjacobian = defgrad->determinant();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get polynomials
  const double polynomValue = evaluate_polynomial(concentration, detjacobian);
  const double polynomDerivativeValue = evaluate_polynomial_derivative(concentration, detjacobian);

  // prepare scalefac
  const double scalefac = -1.0 / (6.0 * c_max) * concentration * chi_max * detjacobian *
                          std::pow(1.0 + polynomValue, -4.0 / 3.0) * polynomDerivativeValue *
                          std::pow(1.0 + polynomReferenceValue, 1.0 / 3.0);

  // calculate diFinjdC
  diFinjdC.multiply_nt(scalefac, id9x1, iCV, 0.0);

  // cmatadd = 2 dSdiFinj : diFinjdC
  cmatadd.multiply_nn(2.0, dSdiFinj, diFinjdC, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracIso::evaluate_od_stiff_mat(
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
    Core::LinAlg::Matrix<6, 1>& dstressdc)
{
  static Core::LinAlg::Matrix<9, 1> id9x1(Core::LinAlg::Initialization::zero);
  // prepare id9x1 (identity matrix written as a 9x1 vector)
  for (int i = 0; i < 3; ++i) id9x1(i) = 1.0;

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double detjacobian = defgrad->determinant();
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get polynomial and derivatives
  const double polynomValue = evaluate_polynomial(concentration, detjacobian);
  const double polynomDerivativeValue = evaluate_polynomial_derivative(concentration, detjacobian);
  const double dChidc = Mat::Electrode::compute_d_intercalation_fraction_d_concentration(
      parameter()->chimax(), parameter()->cmax(), detjacobian);

  // prepare scalefac
  const double scalefac = -1.0 / 3.0 * std::pow(1.0 + polynomValue, -4.0 / 3.0) *
                          std::pow(1.0 + polynomReferenceValue, 1.0 / 3.0) *
                          polynomDerivativeValue * dChidc;

  // calculate diFinjdc and add contribution to dstressdc = dSdiFinj : diFinjdc
  dstressdc.multiply_nn(scalefac, dSdiFinj, id9x1, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracIso::evaluate_inelastic_def_grad_derivative(
    const double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx)
{
  // get parameters
  const int sc1 = parameter()->scalar1();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get polynomial and its derivative
  const double polynomValue = evaluate_polynomial(concentration, detjacobian);
  const double polynomDerivativeValue = evaluate_polynomial_derivative(concentration, detjacobian);

  // calculate the scale factor needed to get the derivative later
  const double denominator = 1.0 / (polynomReferenceValue + 1.0);
  const double base = (polynomValue + 1.0) * denominator;
  const double dChidc = Mat::Electrode::compute_d_intercalation_fraction_d_concentration(
      parameter()->chimax(), parameter()->cmax(), detjacobian);
  const double scalefac =
      1.0 / 3.0 * std::pow(base, -2.0 / 3.0) * polynomDerivativeValue * denominator * dChidc;

  // here dFindc is zeroed out and filled with the current value
  dFindx =
      Core::LinAlg::get_full(scalefac * Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradPolyIntercalFracAniso::InelasticDefgradPolyIntercalFracAniso(
    Core::Mat::PAR::Parameter* params)
    : InelasticDefgradPolyIntercalFrac(params)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracAniso::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  // init and clear variable
  static Core::LinAlg::Matrix<3, 3> FinM(Core::LinAlg::Initialization::zero);
  FinM.clear();

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get polynomials
  const double polynomValue =
      evaluate_polynomial(get_concentration_gp().at(sc1 - 1), defgrad->determinant());

  // calculate growth factor
  const double growth_factor =
      (polynomValue - polynomReferenceValue) / (polynomReferenceValue + 1.0);

  // calculate inelastic deformation gradient and its inverse
  for (int i = 0; i < 3; ++i) FinM(i, i) = 1.0;

  // add the growth part
  FinM.update(growth_factor,
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(parameter()->growth_dir_tensor())), 1.0);

  // calculate inverse of inelastic deformation gradient matrix
  iFinM.invert(FinM);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracAniso::evaluate_additional_cmat(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 1>& iCV,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 6>& cmatadd)
{
  static Core::LinAlg::Matrix<3, 3> temp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iFinjGiFinj(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 1> iFinjGiFinj9x1(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 6> diFinjdC(Core::LinAlg::Initialization::zero);

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double chi_max = parameter()->chimax();
  const double c_max = parameter()->cmax();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double detjacobian = defgrad->determinant();
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get first derivative of polynomial
  const double polynomDerivativeValue = evaluate_polynomial_derivative(concentration, detjacobian);

  // prepare scalefac
  const double scalefac = -detjacobian * concentration * chi_max * polynomDerivativeValue /
                          (2.0 * c_max * (polynomReferenceValue + 1.0));

  // calculate F_{in,j}^{-1} . G . F_{in,j}^{-1} with F_{in,j}, the j-th factor of F_{in}
  temp.multiply_nn(1.0, iFinjM,
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(parameter()->growth_dir_tensor())), 0.0);
  iFinjGiFinj.multiply_nn(1.0, temp, iFinjM, 0.0);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(iFinjGiFinj, iFinjGiFinj9x1);

  // calculate diFinjdC
  diFinjdC.multiply_nt(scalefac, iFinjGiFinj9x1, iCV, 0.0);

  // cmatadd = 2 dSdiFinj : diFinjdC
  cmatadd.multiply_nn(2.0, dSdiFinj, diFinjdC, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracAniso::evaluate_od_stiff_mat(
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
    Core::LinAlg::Matrix<6, 1>& dstressdc)
{
  // static variables
  static Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> diFinjdcM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 1> diFinjdc9x1(Core::LinAlg::Initialization::zero);

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double detjacobian = defgrad->determinant();
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get derivatives
  const double polynomDerivativeValue =
      evaluate_polynomial_derivative(get_concentration_gp().at(sc1 - 1), detjacobian);
  const double dChidc = Mat::Electrode::compute_d_intercalation_fraction_d_concentration(
      parameter()->chimax(), parameter()->cmax(), detjacobian);

  // prepare scalefac
  const double scalefac = -polynomDerivativeValue / (polynomReferenceValue + 1.0) * dChidc;

  // calculate diFinjdc
  tmp.multiply_nn(1.0, iFinjM,
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(parameter()->growth_dir_tensor())), 0.0);
  diFinjdcM.multiply_nn(scalefac, tmp, iFinjM, 0.0);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(diFinjdcM, diFinjdc9x1);

  // dstressdc = dSdiFinj : diFinjdc
  dstressdc.multiply_nn(1.0, dSdiFinj, diFinjdc9x1, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolyIntercalFracAniso::evaluate_inelastic_def_grad_derivative(
    const double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx)
{
  // get parameters
  const int sc1 = parameter()->scalar1();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double polynomReferenceValue = parameter()->get_polynom_reference_value();

  // get polynomial derivative
  const double polynomDerivativeValue = evaluate_polynomial_derivative(concentration, detjacobian);

  const double dChidc = Mat::Electrode::compute_d_intercalation_fraction_d_concentration(
      parameter()->chimax(), parameter()->cmax(), detjacobian);
  const double scalefac = polynomDerivativeValue / (polynomReferenceValue + 1.0) * dChidc;

  // here dFindc is zeroed out and filled with the current value
  dFindx = Core::LinAlg::get_full(scalefac * parameter()->growth_dir_tensor());
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradLinearShape::InelasticDefgradLinearShape(
    const double growth_fac, const double reference_value)
    : growth_fac_(growth_fac), reference_value_(reference_value)
{
  // safety checks
  FOUR_C_ASSERT_ALWAYS(
      growth_fac >= 0.0, "Growth factor can not be negative, please check your input file!");

  FOUR_C_ASSERT_ALWAYS(growth_fac != 0.0,
      "Do not use linear growth laws with a growth factor of 0.0. Use "
      "'MAT_InelasticDefgradNoGrowth' instead!");
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::InelasticDefgradLinearShape::evaluate_linear_growth(const double value) const
{
  // calculate and return the linear growth factor
  return growth_fac_ * (value - reference_value_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradPolynomialShape::InelasticDefgradPolynomialShape(
    std::vector<double> poly_coeffs, const double x_min, const double x_max)
    : poly_coeffs_(std::move(poly_coeffs)), x_min_(x_min), x_max_(x_max)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::InelasticDefgradPolynomialShape::compute_polynomial(const double x) const
{
  // initialize the variable for the evaluation of the polynomial
  double polynom(0.0);

  // compute polynomial
  for (unsigned i = 0; i < poly_coeffs_.size(); ++i) polynom += poly_coeffs_[i] * std::pow(x, i);

  return polynom;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::InelasticDefgradPolynomialShape::compute_polynomial_derivative(const double x) const
{
  // initialize the variable for the derivative of the polynomial
  double polynomDerivative(0.0);

  // compute first derivative of polynomial
  for (unsigned i = 1; i < poly_coeffs_.size(); ++i)
    polynomDerivative += i * poly_coeffs_[i] * std::pow(x, i - 1);

  return polynomDerivative;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradPolynomialShape::check_polynomial_bounds(const double x) const
{
  // safety check for validity of polynomial
  if ((x < x_min_) or (x > x_max_))
  {
    std::cout << "WARNING: Polynomial is evaluated outside its range of validity!\n";
    std::cout << "Evaluation at: " << x << " Lower bound is " << x_min_ << " Upper bound is "
              << x_max_ << std::endl;
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradLinTempIso::InelasticDefgradLinTempIso(Core::Mat::PAR::Parameter* params)
    : InelasticDefgradFactors(params)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinTempIso::pre_evaluate(const Teuchos::ParameterList& params,
    const EvaluationContext<3>& context, const int gp, const int eleGID)
{
  temperature_ = params.get<double>("temperature");
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinTempIso::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  // get parameters
  const double tempgrowthfac = parameter()->get_temp_growth_fac();
  const double reftemp = parameter()->ref_temp();

  const double growthfactor = 1.0 + tempgrowthfac * (temperature_ - reftemp);
  FOUR_C_ASSERT_ALWAYS(growthfactor > 0.0,
      "Determinant of inelastic deformation must be positive, please check your input file!");
  const double isoinelasticdefo = std::cbrt(growthfactor);

  // calculate inverse inelastic deformation gradient
  for (int i = 0; i < 3; ++i) iFinM(i, i) = 1.0 / isoinelasticdefo;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinTempIso::evaluate_inelastic_def_grad_derivative(
    double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx)
{
  // get parameters
  const double tempgrowthfac = parameter()->get_temp_growth_fac();
  const double reftemp = parameter()->ref_temp();

  const double growthfactor = 1.0 + tempgrowthfac * (temperature_ - reftemp);
  const double scalefac = tempgrowthfac / 3.0 * std::pow(growthfactor, -2.0 / 3.0);

  // here dFindT is zeroed out and filled with the current value
  dFindx =
      Core::LinAlg::get_full(scalefac * Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinTempIso::evaluate_additional_cmat(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 1>& iCV,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 6>& cmatadd)
{
  // nothing to do so far, as current growth model is not a function of displacements (and thus C)
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradLinTempIso::evaluate_od_stiff_mat(
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
    Core::LinAlg::Matrix<6, 1>& dstressdT)
{
  static Core::LinAlg::Matrix<9, 1> id9x1(Core::LinAlg::Initialization::zero);
  // prepare id9x1 (identity matrix written as a 9x1 vector)
  for (int i = 0; i < 3; ++i) id9x1(i) = 1.0;

  // get parameters from parameter class
  const double tempgrowthfac = parameter()->get_temp_growth_fac();
  const double reftemp = parameter()->ref_temp();

  const double growthfactor = 1.0 + tempgrowthfac * (temperature_ - reftemp);
  if (growthfactor <= 0.0) FOUR_C_THROW("Determinante of growth must not become negative");

  const double scalefac = -tempgrowthfac / (3.0 * std::pow(growthfactor, 4.0 / 3.0));

  // dstressdT = dSdiFinj : diFinjdT
  // diFinjdT = - growthfac/(3*[1 + growthfac*(T-T_{ref})]^(4/3)) * I
  dstressdT.multiply_nn(scalefac, dSdiFinj, id9x1, 1.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticSource Mat::InelasticDefgradLinTempIso::get_inelastic_source()
{
  return PAR::InelasticSource::temperature;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradNoGrowth::evaluate_additional_cmat(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 1>& iCV,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 6>& cmatadd)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradNoGrowth::evaluate_inelastic_def_grad_derivative(
    double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradNoGrowth::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  iFinM = identity_;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradNoGrowth::evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
    const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 1>& dstressdx)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticSource Mat::InelasticDefgradNoGrowth::get_inelastic_source()
{
  return PAR::InelasticSource::none;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradNoGrowth::InelasticDefgradNoGrowth(Core::Mat::PAR::Parameter* params)
    : InelasticDefgradFactors(params), identity_(Core::LinAlg::Initialization::zero)
{
  // add 1.0 to main diagonal
  identity_(0, 0) = identity_(1, 1) = identity_(2, 2) = 1.0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradNoGrowth::pre_evaluate(const Teuchos::ParameterList& params,
    const EvaluationContext<3>& context, const int gp, const int eleGID)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradTimeFunct::InelasticDefgradTimeFunct(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata), funct_num_(matdata.parameters.get<int>("FUNCT_NUM"))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTimeFunct::InelasticDefgradTimeFunct(Core::Mat::PAR::Parameter* params)
    : InelasticDefgradFactors(params), funct_value_(0.0)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTimeFunct::pre_evaluate(const Teuchos::ParameterList& params,
    const EvaluationContext<3>& context, const int gp, const int eleGID)
{
  // evaluate function value for current time step.
  const auto& funct = Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfTime>(
      parameter()->funct_num());
  FOUR_C_ASSERT(context.total_time, "Time not given in evaluation context.");
  const double time = *context.total_time;
  funct_value_ = funct.evaluate(time);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradTimeFunctAniso::InelasticDefgradTimeFunctAniso(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : InelasticDefgradTimeFunct(matdata),
      growth_dir_(InelasticDeformationDirection(
          matdata.parameters.get<std::vector<double>>("GrowthDirection")))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticSource Mat::InelasticDefgradTimeFunctAniso::get_inelastic_source()
{
  return PAR::InelasticSource::none;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTimeFunctAniso::InelasticDefgradTimeFunctAniso(
    Core::Mat::PAR::Parameter* params)
    : InelasticDefgradTimeFunct(params),
      identity_(Core::LinAlg::TensorGenerators::identity<double, 3, 3>)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTimeFunctAniso::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  Core::LinAlg::Matrix<3, 3> FinM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::TensorView<double, 3, 3> Fin = Core::LinAlg::make_tensor_view(FinM);

  Fin += Core::LinAlg::get_full(identity_);
  Fin += funct_value() * Core::LinAlg::get_full(parameter()->growth_dir_tensor());

  iFinM.invert(FinM);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticSource Mat::InelasticDefgradTimeFunctIso::get_inelastic_source()
{
  return PAR::InelasticSource::none;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTimeFunctIso::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  iFinM.clear();
  Core::LinAlg::TensorView<double, 3, 3> iFin = Core::LinAlg::make_tensor_view(iFinM);

  const double idetFin = std::pow(1.0 + funct_value(), -1.0 / 3.0);
  iFin += idetFin * Core::LinAlg::get_full(identity_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTimeFunctIso::InelasticDefgradTimeFunctIso(Core::Mat::PAR::Parameter* params)
    : InelasticDefgradTimeFunct(params),
      identity_(Core::LinAlg::TensorGenerators::identity<double, 3, 3>)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplast::InelasticDefgradTransvIsotropElastViscoplast(
    Core::Mat::PAR::Parameter* params, std::shared_ptr<Mat::Viscoplastic::Law> viscoplastic_law,
    Mat::Elastic::CoupTransverselyIsotropic fiber_reader,
    std::vector<std::shared_ptr<Mat::Elastic::Summand>> pot_sum_el,
    std::vector<std::shared_ptr<Mat::Elastic::CoupTransverselyIsotropic>> pot_sum_el_transv_iso,
    const double thermal_expansion_coefficient, const double ref_temperature)
    : InelasticDefgradFactors(params),
      thermal_expansion_coefficient_(thermal_expansion_coefficient),
      ref_temperature_(ref_temperature),
      potsumel_(std::move(pot_sum_el)),
      potsumel_transviso_(std::move(pot_sum_el_transv_iso)),
      viscoplastic_law_(std::move(viscoplastic_law)),
      fiber_reader_(std::move(fiber_reader)),
      state_quantities_(),
      state_quantity_derivatives_(),
      tensor_interpolator_(init_tensor_interpolator()),
      local_substepping_utils_(0.0),
      local_newton_manager_(parameter()->local_newton_params())
{
  // set time step size to 0.0 (this is set to the correct and current value in the
  // preevaluate method)
  time_step_tracker_.dt = 0.0;
  // set minimum substep length
  time_step_tracker_.min_dt = 0.0;

  // initialize time step quantities
  time_step_quantities_.init(ref_temperature_);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::pre_evaluate(
    const Teuchos::ParameterList& params, const EvaluationContext<3>& context, const int gp,
    const int eleGID)
{
  // save parameter list
  params_ = params;

  // set Gauss Point
  gp_ = gp;

  // set element ID
  ele_gid_ = eleGID;


  const double previous_temperature = time_step_quantities_.current_temperature[gp];

  // set current absolute temperature
  time_step_quantities_.current_temperature[gp] = ref_temperature_;
  if (params.isParameter("temperature"))
  {
    time_step_quantities_.current_temperature[gp] = params.get<double>("temperature");
  }

  // invalidate the caches if the temperature changed
  if (std::abs(time_step_quantities_.current_temperature[gp] - previous_temperature) >
      ViscoplastUtils::thermo_mechanical_state_equality_tolerance)
  {
    thermo_mechanical_coupling_cache_.reset(gp);

    // We use the current defgrad stored in time_step_quantities in
    // evalate_inverse_inelastic_defgrad to determine whether or not there is a new state. Since a
    // changed temperature alone is enough to have a new state, we invalidate the stored current
    // defgrad here by setting it to a physically meaningless value.
    time_step_quantities_.current_defgrad[gp].clear();
  }

  // set time step
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  time_step_tracker_.dt = *context.time_step_size;
  FOUR_C_ASSERT(context.total_time, "Total time not given in evaluation context.");
  time_step_tracker_.tnp = *context.total_time;

  // set minimum substep length
  time_step_tracker_.min_dt = time_step_tracker_.dt;
  if (parameter()->use_local_substepping())  // if substepping is applied, then we account for the
                                             // maximum number of halving procedures
  {
    time_step_tracker_.min_dt /= std::pow(2.0, parameter()->max_local_substepping_halve_num());
  }

  // call pre_evaluate method of the time step quantities
  time_step_quantities_.pre_evaluate(gp);
  // call preevaluate method of the viscoplastic law
  viscoplastic_law_->pre_evaluate(params, gp);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::prepare_return_mapping()
{
  // pre-evaluate viscoplastic law
  viscoplastic_law_->pre_evaluate(params_, gp_);

  // reset local iteration count
  local_newton_manager_.set_iteration_count(0);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::calculate_gamma_delta(
    const Core::LinAlg::Matrix<3, 3>& CeM, Core::LinAlg::Matrix<3, 1>& gamma,
    Core::LinAlg::Matrix<8, 1>& delta) const
{
  // compute principal values
  Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> CeV_strain(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Strains::matrix_to_vector(CeM, CeV_strain);
  Core::LinAlg::Voigt::Strains::invariants_principal(prinv, CeV_strain);

  // compute derivatives of principle invariants
  Core::LinAlg::Matrix<3, 1> dPIe(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 1> ddPIIe(Core::LinAlg::Initialization::zero);
  // clear variables
  dPIe.clear();
  ddPIIe.clear();

  // loop over map of associated potential summands
  // derivatives of strain energy function w.r.t. principal invariants
  for (const auto& p : potsumel_)  // only isotropic
  {
    p->add_derivatives_principal(dPIe, ddPIIe, prinv, gp_, ele_gid_);
  }

  // compose coefficients
  Mat::calculate_gamma_delta(gamma, delta, prinv, dPIe, ddPIIe);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
ViscoplastUtils::StateQuantities
Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_state_quantities(
    const Core::LinAlg::Matrix<3, 3>& CM, const double temperature,
    const Core::LinAlg::Matrix<3, 3>& iFinM, const double plastic_strain,
    ViscoplastUtils::ErrorType& err_status, const double dt,
    const ViscoplastUtils::StateQuantityEvalType& eval_type) const
{
  ViscoplastUtils::StateQuantities state_quantities{};
  state_quantities.eval_type = eval_type;


  // auxiliaries
  Core::LinAlg::Matrix<1, 1> temp1x1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 3> temp1x3(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> temp3x3;

  // compute right elastic CG tensor
  temp3x3.multiply_nn(1.0, CM, iFinM, 0.0);
  state_quantities.curr_CeM.multiply_tn(1.0, iFinM, temp3x3, 0.0);
  Core::LinAlg::SymmetricTensor<double, 3, 3> CeV =
      Core::LinAlg::assume_symmetry(Core::LinAlg::make_tensor(state_quantities.curr_CeM));

  // compose isotropic elastic coefficients (Holzapfel, Nonlinear Solid Mechanics, 2000)
  calculate_gamma_delta(
      state_quantities.curr_CeM, state_quantities.curr_gamma, state_quantities.curr_delta);
  state_quantities.curr_SeM.clear();
  state_quantities.curr_dSedCe.clear();
  // compute additional 2nd elastic PK stress and elastic stiffness for the transversely
  // isotropic components (additive split assumed, as for CoupTransverselyIsotropic)
  if (parameter()->mat_behavior() == ViscoplastUtils::MatBehavior::transv_isotropic)
  {
    // initialize empty parameter list
    Teuchos::ParameterList param_list{};

    // loop through all transversely isotropic parts, and compute the additional elastic
    // stress and elastic stiffness
    Core::LinAlg::SymmetricTensor<double, 3, 3> SeV{};

    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> dSedCe{};
    for (const auto& p : potsumel_transviso_)
    {
      FOUR_C_ASSERT_ALWAYS(std::abs(temperature - ref_temperature_) <= 1.0e-16,
          "Thermoelastic coupling with transversely isotropic elastic summands is not supported "
          "yet. Use isotropic elastic summands only.");
      p->add_stress_aniso_principal(CeV, dSedCe, SeV, param_list, gp_, ele_gid_);
    }
    state_quantities.curr_dSedCe.clear();
    state_quantities.curr_dSedCe += Core::LinAlg::make_stress_like_voigt_view(dSedCe);
    state_quantities.curr_SeM = Core::LinAlg::make_matrix(Core::LinAlg::get_full(SeV));
  }

  // Ce * Ce tensor
  Core::LinAlg::Matrix<3, 3> CeCeM(Core::LinAlg::Initialization::zero);
  CeCeM.multiply_nn(1.0, state_quantities.curr_CeM, state_quantities.curr_CeM, 0.0);

  // thermal quantities and stress factors
  Mat::ThermalQuantities thermal_quantities =
      Mat::evaluate_thermal_quantities(temperature - ref_temperature_,
          thermal_expansion_coefficient_, iFinM, gp_, ele_gid_, potsumel_);
  Mat::StressFactors thermal_stress_factors;
  Mat::calculate_gamma_delta(thermal_stress_factors.gamma, thermal_stress_factors.delta,
      thermal_quantities.prinv, thermal_quantities.dPI, thermal_quantities.ddPII);

  // compute mixed thermo-elastic tensors required for reducing the Mandel stress based on
  // temperature
  Core::LinAlg::Matrix<3, 3> CT{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Voigt::Stresses::vector_to_matrix(thermal_quantities.CTV, CT);
  Core::LinAlg::Matrix<3, 3> iCT{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Voigt::Stresses::vector_to_matrix(thermal_quantities.iCTV, iCT);
  Core::LinAlg::Matrix<3, 3> CeCT{Core::LinAlg::Initialization::zero};
  CeCT.multiply_nn(1.0, state_quantities.curr_CeM, CT, 0.0);
  Core::LinAlg::Matrix<3, 3> CeiCT{Core::LinAlg::Initialization::zero};
  CeiCT.multiply_nn(1.0, state_quantities.curr_CeM, iCT, 0.0);


  /**
  compute symmetric part of thermo-elastic Mandel stress tensor
  \f[\boldsymbol{M}_{\theta,\text{sym}}  = \boldsymbol{C}_e \cdot \left( \boldsymbol{S}_{e} -
  \boldsymbol{S}_T \right)\f]
   */
  Core::LinAlg::Matrix<3, 3> Mtheta_sym_M(Core::LinAlg::Initialization::zero);
  Mtheta_sym_M.update(state_quantities.curr_gamma(0), state_quantities.curr_CeM,
      state_quantities.curr_gamma(1), CeCeM, 0.0);
  Mtheta_sym_M.update(state_quantities.curr_gamma(2), const_non_mat_tensors.id3x3, 1.0);
  Mtheta_sym_M.update(-1.0 * thermal_stress_factors.gamma(0), state_quantities.curr_CeM, 1.0);
  Mtheta_sym_M.update(-1.0 * thermal_stress_factors.gamma(1), CeCT, 1.0);
  Mtheta_sym_M.update(-1.0 * thermal_stress_factors.gamma(2), CeiCT, 1.0);
  if (parameter()->mat_behavior() == ViscoplastUtils::MatBehavior::transv_isotropic)
  {
    Core::LinAlg::Matrix<3, 3> addMeM(Core::LinAlg::Initialization::zero);
    temp3x3.multiply_nn(1.0, state_quantities.curr_CeM, state_quantities.curr_SeM, 0.0);
    addMeM.update(1.0 / 2.0, temp3x3, 0.0);
    temp3x3.multiply_tn(1.0, state_quantities.curr_SeM, state_quantities.curr_CeM, 0.0);
    addMeM.update(1.0 / 2.0, temp3x3, 1.0);
    Mtheta_sym_M.update(1.0, addMeM, 1.0);
  }

  // calculate deviatoric part of the symmetric Mandel stress
  double trMtheta = 0.0;
  for (int i = 0; i < 3; ++i) trMtheta += Mtheta_sym_M(i, i);
  state_quantities.curr_Mtheta_dev_sym_M.update(
      1.0, Mtheta_sym_M, -1.0 / 3.0 * trMtheta, const_non_mat_tensors.id3x3, 0.0);

  // for transverse isotropy: we use the Hill 1949 yield condition, adapted for transversely
  // isotropic materials --> get yield function parameters A, B, F
  const double A = parameter()->yield_cond_a();
  const double B = parameter()->yield_cond_b();
  const double F = parameter()->yield_cond_f();

  // determine scalar quantities of invariants / pseudoinvariants needed to compute the
  // equivalent tensile stress
  double Mtheta_dev_sym_contract_Mtheta_dev_sym =
      Core::LinAlg::FourTensorOperations::contract_matrix_matrix(
          state_quantities.curr_Mtheta_dev_sym_M, state_quantities.curr_Mtheta_dev_sym_M);
  Core::LinAlg::Matrix<3, 3> Mtheta_dev_sym_squared_M(Core::LinAlg::Initialization::zero);
  Mtheta_dev_sym_squared_M.multiply_nn(
      1.0, state_quantities.curr_Mtheta_dev_sym_M, state_quantities.curr_Mtheta_dev_sym_M, 0.0);
  temp1x3.multiply_tn(1.0, m_, Mtheta_dev_sym_squared_M, 0.0);
  temp1x1.multiply_nn(1.0, temp1x3, m_, 0.0);
  double mTMtheta_dev_sym_squared_m = temp1x1(0);
  temp1x3.multiply_tn(1.0, m_, state_quantities.curr_Mtheta_dev_sym_M, 0.0);
  temp1x1.multiply_nn(1.0, temp1x3, m_, 0.0);
  double mTMtheta_dev_sym_m = temp1x1(0);

  // calculate equivalent tensile stress
  if (parameter()->mat_behavior() == ViscoplastUtils::MatBehavior::transv_isotropic)
  {
    state_quantities.curr_equiv_stress =
        std::sqrt((A + 2 * B) * Mtheta_dev_sym_contract_Mtheta_dev_sym +
                  2 * (F - A - 2 * B) * mTMtheta_dev_sym_squared_m +
                  (5 * A + B - 2 * F) * std::pow(mTMtheta_dev_sym_m, 2.0));
  }
  else if (parameter()->mat_behavior() == ViscoplastUtils::MatBehavior::isotropic)
  {
    state_quantities.curr_equiv_stress =
        std::sqrt(3.0 / 2.0 * Mtheta_dev_sym_contract_Mtheta_dev_sym);
  }
  else
  {
    FOUR_C_THROW("Inconsistent material behavior {} in evaluating equivalent stress",
        EnumTools::enum_name(parameter()->mat_behavior()));
  }

  // check if either stress or plastic strain are NaN -> in this case,
  // return with overflow error
  if (std::isnan(state_quantities.curr_equiv_stress) || std::isnan(plastic_strain))
  {
    // return with error
    err_status = ViscoplastUtils::ErrorType::overflow_error;
    return state_quantities;
  }

  if (eval_type == ViscoplastUtils::StateQuantityEvalType::equiv_stress_only)
  {
    return state_quantities;
  }


  // calculate equivalent plastic strain rate using the viscoplastic law
  state_quantities.curr_equiv_plastic_strain_rate =
      viscoplastic_law_->evaluate_plastic_strain_rate(state_quantities.curr_equiv_stress,
          plastic_strain, dt, parameter()->max_plastic_strain_incr(), err_status, update_hist_var_);

  if (eval_type == ViscoplastUtils::StateQuantityEvalType::plastic_strain_rate_only)
  {
    return state_quantities;
  }

  // return if we get an error, all other calculations are useless since substepping is
  // triggered
  if (err_status != ViscoplastUtils::ErrorType::no_errors)
  {
    // return with error
    return state_quantities;
  }

  // calculate plastic flow direction
  if (parameter()->mat_behavior() == ViscoplastUtils::MatBehavior::transv_isotropic)
  {
    // determine required components for the computation of the plastic flow direction
    Core::LinAlg::Matrix<3, 1> Mtheta_dev_sym_m(Core::LinAlg::Initialization::zero);
    Mtheta_dev_sym_m.multiply_nn(1.0, state_quantities.curr_Mtheta_dev_sym_M, m_, 0.0);
    Core::LinAlg::Matrix<3, 3> m_dyad_Mtheta_dev_sym_m(Core::LinAlg::Initialization::zero);
    m_dyad_Mtheta_dev_sym_m.multiply_nt(1.0, m_, Mtheta_dev_sym_m, 0.0);
    Core::LinAlg::Matrix<3, 3> Mtheta_dev_sym_A_dyad_A(Core::LinAlg::Initialization::zero);
    Mtheta_dev_sym_A_dyad_A.multiply_nt(1.0, Mtheta_dev_sym_m, m_, 0.0);

    state_quantities.curr_NpM.clear();
    state_quantities.curr_dpM.clear();
    if (state_quantities.curr_equiv_stress > 0.0)
    {
      // build the plastic flow direction from its tensor parts
      state_quantities.curr_NpM.update(-2.0 / (3.0 * state_quantities.curr_equiv_stress) *
                                           (F - A - 2.0 * B) * mTMtheta_dev_sym_m,
          const_non_mat_tensors.id3x3, 0.0);
      state_quantities.curr_NpM.update(
          1.0 / (1.0 * state_quantities.curr_equiv_stress) * (A + 2.0 * B),
          state_quantities.curr_Mtheta_dev_sym_M, 1.0);
      state_quantities.curr_NpM.update(
          1.0 / (2.0 * state_quantities.curr_equiv_stress) * 2.0 * (F - A - 2.0 * B),
          m_dyad_Mtheta_dev_sym_m, 1.0);
      state_quantities.curr_NpM.update(
          1.0 / (2.0 * state_quantities.curr_equiv_stress) * 2.0 * (F - A - 2.0 * B),
          Mtheta_dev_sym_A_dyad_A, 1.0);
      state_quantities.curr_NpM.update(1.0 / (2.0 * state_quantities.curr_equiv_stress) *
                                           (5.0 * A + B - 2.0 * F) * 2.0 * mTMtheta_dev_sym_m,
          const_mat_tensors_.mm_dev, 1.0);

      // calculate plastic stretching tensor (deformation rate tensor)
      state_quantities.curr_dpM.update(
          state_quantities.curr_equiv_plastic_strain_rate, state_quantities.curr_NpM, 0.0);
    }

    // calculate plastic velocity gradient tensor
    state_quantities.curr_lpM.multiply_nn(
        1.0, const_mat_tensors_.id_plus_mm, state_quantities.curr_dpM, 0.0);
    state_quantities.curr_lpM.multiply_nn(
        -1.0, state_quantities.curr_dpM, const_mat_tensors_.mm, 1.0);
  }
  else if (parameter()->mat_behavior() == ViscoplastUtils::MatBehavior::isotropic)
  {
    state_quantities.curr_NpM.clear();
    state_quantities.curr_dpM.clear();
    if (state_quantities.curr_equiv_stress > 0.0)
    {
      // build the plastic flow direction from its tensor parts
      state_quantities.curr_NpM.update(3.0 / (2.0 * state_quantities.curr_equiv_stress),
          state_quantities.curr_Mtheta_dev_sym_M, 0.0);

      // calculate plastic stretching tensor (deformation rate tensor)
      state_quantities.curr_dpM.update(
          state_quantities.curr_equiv_plastic_strain_rate, state_quantities.curr_NpM, 0.0);
    }

    // calculate plastic velocity gradient tensor
    state_quantities.curr_lpM.update(1.0, state_quantities.curr_dpM, 0.0);
  }
  else
  {
    FOUR_C_THROW("Inconsistent material behavior {} for evaluating plastic flow direction",
        EnumTools::enum_name(parameter()->mat_behavior()));
  }

  // calculate plastic update tensor (only required, and computed, for standard substepping)
  if (parameter()->timint_type() == ViscoplastUtils::TimIntType::standard)
  {
    temp3x3.update(-dt, state_quantities.curr_lpM, 0.0);
    auto exp_err_status = Core::LinAlg::MatrixFunctErrorType::no_errors;
    state_quantities.curr_EpM =
        Core::LinAlg::matrix_exp(temp3x3, exp_err_status, parameter()->mat_exp_calc_method());
    if (exp_err_status != Core::LinAlg::MatrixFunctErrorType::no_errors)
    {
      err_status = ViscoplastUtils::ErrorType::failed_matrix_exp_evaluation;
      return state_quantities;
    }

    return state_quantities;
  }
  else if (parameter()->timint_type() == ViscoplastUtils::TimIntType::logarithmic)
  {
    // nothing else to be done for logarithmic time integration
    return state_quantities;
  }
  else
  {
    FOUR_C_THROW("Inconsistent time integration type {} in evaluating plastic update tensor",
        EnumTools::enum_name(parameter()->timint_type()));
  }

  return state_quantities;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
ViscoplastUtils::StateQuantityDerivatives
Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_state_quantity_derivatives(
    const Core::LinAlg::Matrix<3, 3>& CM, const double temperature,
    const Core::LinAlg::Matrix<3, 3>& iFinM, const double plastic_strain,
    ViscoplastUtils::ErrorType& err_status, const double dt,
    const ViscoplastUtils::StateQuantityDerivEvalType& eval_type, const bool eval_state) const
{
  ViscoplastUtils::StateQuantityDerivatives state_quantity_derivatives{};
  state_quantity_derivatives.eval_type = eval_type;

  // auxiliaries
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 6> temp6x6(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 9> temp6x9(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<9, 6> temp9x6(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<9, 9> temp9x9(Core::LinAlg::Initialization::zero);
  Core::LinAlg::FourTensor<3> temp_four_tensor(true);

  // check whether to reevaluate the state or to keep the available state quantity values
  ViscoplastUtils::StateQuantities relevant_state_quantities = state_quantities_;
  if (eval_state)
  {
    relevant_state_quantities = evaluate_state_quantities(CM, temperature, iFinM, plastic_strain,
        err_status, dt, ViscoplastUtils::StateQuantityEvalType::full_eval);
  }

  // get the state quantities
  const Core::LinAlg::Matrix<3, 3> CeM = relevant_state_quantities.curr_CeM;
  const Core::LinAlg::Matrix<3, 1> gamma = relevant_state_quantities.curr_gamma;
  const Core::LinAlg::Matrix<8, 1> delta = relevant_state_quantities.curr_delta;
  const Core::LinAlg::Matrix<3, 3> SeM = relevant_state_quantities.curr_SeM;
  const Core::LinAlg::Matrix<6, 6> dSedCe = relevant_state_quantities.curr_dSedCe;
  const Core::LinAlg::Matrix<3, 3> Mtheta_dev_sym_M =
      relevant_state_quantities.curr_Mtheta_dev_sym_M;
  const double equiv_stress = relevant_state_quantities.curr_equiv_stress;
  const double equiv_plastic_strain_rate = relevant_state_quantities.curr_equiv_plastic_strain_rate;
  const Core::LinAlg::Matrix<3, 3> NpM = relevant_state_quantities.curr_NpM;
  const Core::LinAlg::Matrix<3, 3> dpM = relevant_state_quantities.curr_dpM;
  const Core::LinAlg::Matrix<3, 3> lpM = relevant_state_quantities.curr_lpM;
  const Core::LinAlg::Matrix<3, 3> EpM = relevant_state_quantities.curr_EpM;


  // compute the relevant derivatives of the elastic right Cauchy-Green deformation tensor
  Mat::elast_hyper_get_derivs_of_elastic_right_cg_tensor(Core::LinAlg::make_tensor(iFinM),
      Core::LinAlg::assume_symmetry(Core::LinAlg::make_tensor(CM)),
      state_quantity_derivatives.curr_dCedC, state_quantity_derivatives.curr_dCediFin);
  // save these also as four tensors
  Core::LinAlg::FourTensor<3> dCediFin_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x9_voigt_matrix(
      dCediFin_FourTensor, state_quantity_derivatives.curr_dCediFin);
  Core::LinAlg::FourTensor<3> dCedC_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix(
      dCedC_FourTensor, state_quantity_derivatives.curr_dCedC);

  // inverse inelastic right Cauchy-Green deformation tensor
  Core::LinAlg::Matrix<3, 3> iCinM(Core::LinAlg::Initialization::zero);
  iCinM.multiply_nt(1.0, iFinM, iFinM, 0.0);
  Core::LinAlg::Matrix<6, 1> iCinV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCinM, iCinV);

  // elastic right Cauchy-Green tensor in stress form
  Core::LinAlg::Matrix<6, 1> CeV(Core::LinAlg::Initialization::zero);  // stress-form
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(CeM, CeV);

  // inverse elastic right Cauchy-Green tensor
  Core::LinAlg::Matrix<3, 3> iCeM(Core::LinAlg::Initialization::zero);
  iCeM.invert(CeM);
  Core::LinAlg::Matrix<6, 1> iCeV(Core::LinAlg::Initialization::zero);  // stress-form
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCeM, iCeV);

  // inverse transposed inelastic defgrad
  Core::LinAlg::Matrix<3, 3> iFinTM(Core::LinAlg::Initialization::zero);
  iFinTM.multiply_tn(1.0, iFinM, const_non_mat_tensors.id3x3, 0.0);

  // calculate various other helper tensors required for subsequent computation
  Core::LinAlg::Matrix<3, 3> CiFinM(Core::LinAlg::Initialization::zero);
  CiFinM.multiply_nn(1.0, CM, iFinM, 0.0);
  Core::LinAlg::Matrix<9, 1> CiFinV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(CiFinM, CiFinV);

  Core::LinAlg::Matrix<3, 3> iFinTCM(Core::LinAlg::Initialization::zero);
  iFinTCM.multiply_tn(1.0, iFinM, CM, 0.0);

  Core::LinAlg::Matrix<3, 3> CeiFinTCM(Core::LinAlg::Initialization::zero);
  temp3x3.multiply_nt(1.0, CeM, iFinM, 0.0);
  CeiFinTCM.multiply_nn(1.0, temp3x3, CM, 0.0);

  Core::LinAlg::Matrix<3, 3> CiFinCeM(Core::LinAlg::Initialization::zero);
  CiFinCeM.multiply_nn(1.0, CiFinM, CeM, 0.0);
  Core::LinAlg::Matrix<9, 1> CiFinCeV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(CiFinCeM, CiFinCeV);

  Core::LinAlg::Matrix<3, 3> CeCeM(Core::LinAlg::Initialization::zero);
  CeCeM.multiply_nn(1.0, CeM, CeM, 0.0);
  Core::LinAlg::Matrix<6, 1> CeCeV(Core::LinAlg::Initialization::zero);  // stress-form
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(CeCeM, CeCeV);

  Core::LinAlg::Matrix<3, 3> CiFiniCeM(Core::LinAlg::Initialization::zero);
  CiFiniCeM.multiply_nn(1.0, CiFinM, iCeM, 0.0);
  Core::LinAlg::Matrix<9, 1> CiFiniCeV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(CiFiniCeM, CiFiniCeV);

  Core::LinAlg::Matrix<3, 3> CeiFinTM(Core::LinAlg::Initialization::zero);
  CeiFinTM.multiply_nn(1.0, CeM, iFinTM, 0.0);

  Core::LinAlg::Matrix<3, 3> iCinCiCinM(Core::LinAlg::Initialization::zero);
  temp3x3.multiply_nn(1.0, CM, iCinM, 0.0);
  iCinCiCinM.multiply_nn(1.0, iCinM, temp3x3, 0.0);
  Core::LinAlg::Matrix<6, 1> iCinCiCinV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCinCiCinM, iCinCiCinV);

  Core::LinAlg::Matrix<3, 3> iCM(Core::LinAlg::Initialization::zero);
  iCM.invert(CM);
  Core::LinAlg::Matrix<6, 1> iCV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(iCM, iCV);

  // thermal quantities and stress factors
  Mat::ThermalQuantities thermal_quantities =
      Mat::evaluate_thermal_quantities(temperature - ref_temperature_,
          thermal_expansion_coefficient_, iFinM, gp_, ele_gid_, potsumel_);
  Core::LinAlg::SymmetricTensor<double, 3, 3> hyperelast_stress_CT{};
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> hyperelast_stiffness_CT{};
  Mat::elast_hyper_add_isotropic_stress_cmat(hyperelast_stress_CT, hyperelast_stiffness_CT,
      Core::LinAlg::make_symmetric_tensor_from_stress_like_voigt_matrix(thermal_quantities.CTV),
      Core::LinAlg::make_symmetric_tensor_from_stress_like_voigt_matrix(thermal_quantities.iCTV),
      thermal_quantities.prinv, thermal_quantities.dPI, thermal_quantities.ddPII);
  const Core::LinAlg::Matrix<3, 3> S_T =
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(hyperelast_stress_CT));

  // compute the relevant derivatives of the symmetric part of the Mandel stress

  /**
   \f$ \frac{\partial \boldsymbol{M}_{\theta, \text{sym}} }{\partial
   \boldsymbol{F}_{\text{in}}^{-1}}  =
   \frac{\partial}{\partial \boldsymbol{F}_{\text{in}}^{-1}}\left(\boldsymbol{C}_e \cdot
   \boldsymbol{S}_{e}\right)
   - \frac{\partial}{\partial \boldsymbol{F}_{\text{in}}^{-1}}\left(\boldsymbol{C}_e \cdot
   \boldsymbol{S}_{T}\right)\f$ (Voigt stress-form)
  */
  Core::LinAlg::Matrix<6, 9> dMtheta_sym_diFin(Core::LinAlg::Initialization::zero);
  dMtheta_sym_diFin.clear();
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dMtheta_sym_diFin, iFinTCM, const_non_mat_tensors.id3x3, gamma(0));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dMtheta_sym_diFin, iFinTCM, CeM, gamma(1));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dMtheta_sym_diFin, CeiFinTCM, const_non_mat_tensors.id3x3, gamma(1));
  dMtheta_sym_diFin.multiply_nt(delta(0), CeV, CiFinV, 1.0);
  dMtheta_sym_diFin.multiply_nt(delta(1), CeV, CiFinCeV, 1.0);
  dMtheta_sym_diFin.multiply_nt(delta(1), CeCeV, CiFinV, 1.0);
  dMtheta_sym_diFin.multiply_nt(delta(2), CeV, CiFiniCeV, 1.0);
  dMtheta_sym_diFin.multiply_nt(delta(2), const_non_mat_tensors.id6x1, CiFinV, 1.0);
  dMtheta_sym_diFin.multiply_nt(delta(3), CeCeV, CiFinCeV, 1.0);
  dMtheta_sym_diFin.multiply_nt(delta(4), CeCeV, CiFiniCeV, 1.0);
  dMtheta_sym_diFin.multiply_nt(delta(4), const_non_mat_tensors.id6x1, CiFinCeV, 1.0);
  dMtheta_sym_diFin.multiply_nt(delta(5), const_non_mat_tensors.id6x1, CiFiniCeV, 1.0);
  // thermal contribution
  temp_four_tensor.clear();
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor_by_second_index<3>(
      temp_four_tensor, S_T, dCediFin_FourTensor, true);
  Core::LinAlg::Matrix<6, 9> dMT_sym_dFin{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(dMT_sym_dFin, temp_four_tensor);
  dMtheta_sym_diFin.update(-1.0, dMT_sym_dFin, 1.0);

  /**
   \f$ \frac{\partial \boldsymbol{M}_{\theta, \text{sym}} }{\partial
   \boldsymbol{C}}  =
   \frac{\partial}{\partial \boldsymbol{C}}\left(\boldsymbol{C}_e \cdot \boldsymbol{S}_{e}\right)
   - \frac{\partial}{\partial \boldsymbol{C}}\left(\boldsymbol{C}_e \cdot
   \boldsymbol{S}_{T}\right)\f$ (Voigt stress-stress form)
  */
  Core::LinAlg::Matrix<6, 6> dMtheta_sym_dC(Core::LinAlg::Initialization::zero);
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
      dMtheta_sym_dC, gamma(0), iFinTM, iFinTM, 0.0);
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
      dMtheta_sym_dC, gamma(1), iFinTM, CeiFinTM, 1.0);
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
      dMtheta_sym_dC, gamma(1), CeiFinTM, iFinTM, 1.0);
  dMtheta_sym_dC.multiply_nt(delta(0) / 2.0, CeV, iCinV, 1.0);
  dMtheta_sym_dC.multiply_nt(delta(1) / 2.0, CeV, iCinCiCinV, 1.0);
  dMtheta_sym_dC.multiply_nt(delta(1) / 2.0, CeCeV, iCinV, 1.0);
  dMtheta_sym_dC.multiply_nt(delta(2) / 2.0, CeV, iCV, 1.0);
  dMtheta_sym_dC.multiply_nt(delta(2) / 2.0, const_non_mat_tensors.id6x1, iCinV, 1.0);
  dMtheta_sym_dC.multiply_nt(delta(3) / 2.0, CeCeV, iCinCiCinV, 1.0);
  dMtheta_sym_dC.multiply_nt(delta(4) / 2.0, CeCeV, iCV, 1.0);
  dMtheta_sym_dC.multiply_nt(delta(4) / 2.0, const_non_mat_tensors.id6x1, iCinCiCinV, 1.0);
  dMtheta_sym_dC.multiply_nt(delta(5) / 2.0, const_non_mat_tensors.id6x1, iCV, 1.0);
  // thermal contribution
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor_by_second_index<3>(
      temp_four_tensor, S_T, dCedC_FourTensor, true);
  Core::LinAlg::Matrix<6, 6> dMT_sym_dC{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(dMT_sym_dC, temp_four_tensor);
  dMtheta_sym_dC.update(-1.0, dMT_sym_dC, 1.0);

  /**
   \f$ \frac{\partial \boldsymbol{M}_{\theta, \text{sym}} }{\partial T}  =
   - \boldsymbol{C}_e \cdot \frac{\partial\boldsymbol{S}_{T}}{\partial T}\f$ (Voigt stress-form)
  */
  Core::LinAlg::Matrix<6, 1> dST_dT_V{Core::LinAlg::Initialization::zero};
  dST_dT_V.multiply_nn(0.5, Core::LinAlg::make_stress_like_voigt_view(hyperelast_stiffness_CT),
      thermal_quantities.dCTdTV, 0.0);
  Core::LinAlg::Matrix<3, 3> dST_dT_M{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Voigt::Stresses::vector_to_matrix(dST_dT_V, dST_dT_M);
  Core::LinAlg::Matrix<3, 3> dMtheta_sym_dT_M(Core::LinAlg::Initialization::zero);
  dMtheta_sym_dT_M.multiply_nn(-1.0, CeM, dST_dT_M, 0.0);
  Core::LinAlg::Matrix<6, 1> dMtheta_sym_dT_V(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(dMtheta_sym_dT_M, dMtheta_sym_dT_V);


  // compute derivative of the additional transversely isotropic stress (w.r.t. right
  // elastic Cauchy-Green deformation tensor) in stress-strain notation
  temp6x6.update(1.0, dSedCe, 0.0);
  Core::LinAlg::Matrix<6, 6> dSedCe_stress_strain =
      Core::LinAlg::Voigt::modify_voigt_representation(temp6x6, 1.0, 2.0);

  // \f$ \frac{\partial \boldsymbol{S}^{\text{e}}_{\text{trn}} }{\partial
  // \boldsymbol{F}^{\text{in}^{-1}}_{}} \f$ (Voigt stress-form)
  Core::LinAlg::Matrix<6, 9> dSediFin(Core::LinAlg::Initialization::zero);
  dSediFin.multiply_nn(1.0, dSedCe_stress_strain, state_quantity_derivatives.curr_dCediFin, 0.0);
  Core::LinAlg::FourTensor<3> dSediFin_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x9_voigt_matrix(dSediFin_FourTensor, dSediFin);

  // \f$ \frac{\partial \boldsymbol{S}^{\text{e}}_{\text{trn}} }{\partial
  // \boldsymbol{C}^{}_{}} \f$ (Voigt stress-stress form)
  Core::LinAlg::Matrix<6, 6> dSedC(Core::LinAlg::Initialization::zero);
  dSedC.multiply_nn(1.0, dSedCe_stress_strain, state_quantity_derivatives.curr_dCedC, 0.0);
  Core::LinAlg::FourTensor<3> dSedC_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix(dSedC_FourTensor, dSedC);

  // compute additional components of the elastic transversely isotropic components for the
  // derivatives of the symmetric Mandel stress
  if (parameter()->mat_behavior() == ViscoplastUtils::MatBehavior::transv_isotropic)
  {
    Core::LinAlg::FourTensor<3> CedSediFin_FourTensor(true);
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        CedSediFin_FourTensor, CeM, dSediFin_FourTensor, true);
    Core::LinAlg::FourTensor<3> CedSediFin_T12_FourTensor(true);
    CedSediFin_T12_FourTensor.transpose_12(CedSediFin_FourTensor);
    Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(temp6x9, CedSediFin_FourTensor);
    dMtheta_sym_diFin.update(1.0 / 2.0, temp6x9, 1.0);
    Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(
        temp6x9, CedSediFin_T12_FourTensor);
    dMtheta_sym_diFin.update(1.0 / 2.0, temp6x9, 1.0);
    Core::LinAlg::FourTensor<3> SedCediFin_FourTensor(true);
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        SedCediFin_FourTensor, SeM, dCediFin_FourTensor, true);
    Core::LinAlg::FourTensor<3> SedCediFin_T12_FourTensor(true);
    SedCediFin_T12_FourTensor.transpose_12(SedCediFin_FourTensor);
    Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(temp6x9, SedCediFin_FourTensor);
    dMtheta_sym_diFin.update(1.0 / 2.0, temp6x9, 1.0);
    Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(
        temp6x9, SedCediFin_T12_FourTensor);
    dMtheta_sym_diFin.update(1.0 / 2.0, temp6x9, 1.0);

    Core::LinAlg::FourTensor<3> CedSedC_FourTensor(true);
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        CedSedC_FourTensor, CeM, dSedC_FourTensor, true);
    Core::LinAlg::FourTensor<3> CedSedC_T12_FourTensor(true);
    CedSedC_T12_FourTensor.transpose_12(CedSedC_FourTensor);
    Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, CedSedC_FourTensor);
    dMtheta_sym_dC.update(1.0 / 2.0, temp6x6, 1.0);
    Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, CedSedC_T12_FourTensor);
    dMtheta_sym_dC.update(1.0 / 2.0, temp6x6, 1.0);
    Core::LinAlg::FourTensor<3> SedCedC_FourTensor(true);
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        SedCedC_FourTensor, SeM, dCedC_FourTensor, true);
    Core::LinAlg::FourTensor<3> SedCedC_T12_FourTensor(true);
    SedCedC_T12_FourTensor.transpose_12(SedCedC_FourTensor);
    Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, SedCedC_FourTensor);
    dMtheta_sym_dC.update(1.0 / 2.0, temp6x6, 1.0);
    Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, SedCedC_T12_FourTensor);
    dMtheta_sym_dC.update(1.0 / 2.0, temp6x6, 1.0);
  }

  // compute derivatives of the deviatoric, symmetric part of the Mandel stress

  /**
  \f$ \frac{\partial \boldsymbol{M}^{\theta}_{\text{dev, sym}} }{\partial
  \boldsymbol{F}^{\text{in}^{-1}}_{}} \f$ (Voigt stress-form)
  */
  state_quantity_derivatives.curr_dMtheta_dev_sym_diFin.multiply_nn(
      1.0, const_non_mat_tensors.dev_op, dMtheta_sym_diFin, 0.0);
  /**
  \f$ \frac{\partial \boldsymbol{M}^{\theta}_{\text{dev,sym}} }{\partial
  \boldsymbol{C}^{}_{}} \f$ (Voigt stress-stress form)
  */
  state_quantity_derivatives.curr_dMtheta_dev_sym_dC.multiply_nn(
      1.0, const_non_mat_tensors.dev_op, dMtheta_sym_dC, 0.0);
  /**
  \f$ \frac{\partial \boldsymbol{M}^{\theta}_{\text{dev,sym}} }{\partial
  T} \f$ (Voigt stress form)
  */
  state_quantity_derivatives.curr_dMtheta_dev_sym_dT.multiply_nn(
      1.0, const_non_mat_tensors.dev_op, dMtheta_sym_dT_V, 0.0);

  // plastic flow direction in Voigt strain notation
  Core::LinAlg::Matrix<6, 1> NpVstrainform(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Strains::matrix_to_vector(NpM, NpVstrainform);

  // compute derivatives of the equivalent stress

  /// \f$ \frac{\partial \overline{\sigma} }{\partial
  /// \boldsymbol{F}^{\text{in}^{-1}}_{}} \f$ (Voigt stress-form)
  state_quantity_derivatives.curr_dequiv_stress_diFin.multiply_tn(
      1.0, NpVstrainform, state_quantity_derivatives.curr_dMtheta_dev_sym_diFin, 0.0);
  /// \f$ \frac{\partial \overline{\sigma} }{\partial
  /// \boldsymbol{C}^{}} \f$ (Voigt stress-form)
  state_quantity_derivatives.curr_dequiv_stress_dC.multiply_tn(
      1.0, NpVstrainform, state_quantity_derivatives.curr_dMtheta_dev_sym_dC, 0.0);
  /// \f$ \frac{\partial \overline{\sigma} }{\partial
  /// T} \f$
  Core::LinAlg::Matrix<1, 1> temp1x1{Core::LinAlg::Initialization::zero};
  temp1x1.multiply_tn(1.0, NpVstrainform, state_quantity_derivatives.curr_dMtheta_dev_sym_dT, 0.0);
  state_quantity_derivatives.curr_dequiv_stress_dT = temp1x1(0);



  // recompute flow direction in stress form
  Core::LinAlg::Matrix<6, 1> NpVstressform(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(NpM, NpVstressform);

  // we use the Hill 1949 yield condition, adapted for transversely isotropic materials ->
  // get yield condition parameters A, B, and F
  const double A = parameter()->yield_cond_a();
  const double B = parameter()->yield_cond_b();
  const double F = parameter()->yield_cond_f();

  // compute required derivative of the plastic flow direction (w.r.t. dev., sym. part of
  // the Mandel stress)
  /// \f$ \frac{\partial \boldsymbol{N}_{\text{p}} }{\partial
  /// \boldsymbol{M}_{\theta,\text{dev,sym}}} \f$ (Voigt stress-stress form)
  Core::LinAlg::Matrix<6, 6> dNpdMtheta_sym_dev(Core::LinAlg::Initialization::zero);
  if (parameter()->mat_behavior() == ViscoplastUtils::MatBehavior::transv_isotropic)
  {
    dNpdMtheta_sym_dev.multiply_nt(-1.0 / equiv_stress, NpVstressform, NpVstressform, 0.0);
    dNpdMtheta_sym_dev.update(-1.0 / 2.0 * 1.0 / equiv_stress * 4.0 / 3.0 * (F - A - 2.0 * B),
        const_mat_tensors_.id_dyad_mm, 1.0);
    dNpdMtheta_sym_dev.update(1.0 / equiv_stress * (A + 2 * B), const_non_mat_tensors.id4_6x6, 1.0);
    Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(dNpdMtheta_sym_dev,
        1.0 / equiv_stress * (F - A - 2 * B), const_mat_tensors_.mm, const_non_mat_tensors.id3x3,
        1.0);
    Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(dNpdMtheta_sym_dev,
        1.0 / equiv_stress * (F - A - 2 * B), const_non_mat_tensors.id3x3, const_mat_tensors_.mm,
        1.0);
    dNpdMtheta_sym_dev.update(
        1.0 / equiv_stress * (5 * A + B - 2 * F), const_mat_tensors_.mm_dev_dyad_mm, 1.0);
  }
  else if (parameter()->mat_behavior() == ViscoplastUtils::MatBehavior::isotropic)
  {
    dNpdMtheta_sym_dev.multiply_nt(-1.0 / equiv_stress, NpVstressform, NpVstressform, 0.0);
    dNpdMtheta_sym_dev.update(1.0 / equiv_stress * 3.0 / 2.0, const_non_mat_tensors.id4_6x6, 1.0);
  }
  else
  {
    FOUR_C_THROW(
        "Inconsistent material behavior {} in evaluating derivative of plastic flow direction",
        EnumTools::enum_name(parameter()->mat_behavior()));
  }

  // compute the relevant derivatives of the plastic strain rate
  InelasticDefgradTransvIsotropElastViscoplastUtils::PlasticStrainRateDerivs evoEqFunctionDers =
      viscoplastic_law_->evaluate_derivatives_of_plastic_strain_rate(equiv_stress, plastic_strain,
          dt, parameter()->max_plastic_strain_deriv_incr(), err_status);

  // return if we get an error, all other calculations are useless since substepping is
  // triggered
  if (err_status != ViscoplastUtils::ErrorType::no_errors)
  {
    // return with error
    return ViscoplastUtils::StateQuantityDerivatives{};
  }

  if (eval_type == ViscoplastUtils::StateQuantityDerivEvalType::equiv_stress_derivs_only)
  {
    return state_quantity_derivatives;
  }

  // compute derivatives of the plastic strain rate
  state_quantity_derivatives.curr_dpsr_dequiv_stress = evoEqFunctionDers.deriv_equiv_stress;
  state_quantity_derivatives.curr_dpsr_depsp = evoEqFunctionDers.deriv_plastic_strain;
  state_quantity_derivatives.curr_dpsr_dT = evoEqFunctionDers.deriv_temperature;


  if (eval_type == ViscoplastUtils::StateQuantityDerivEvalType::plastic_strain_rate_derivs_only)
  {
    return state_quantity_derivatives;
  }

  // convert derivative to Voigt stress-strain form
  temp6x6 = dNpdMtheta_sym_dev;
  dNpdMtheta_sym_dev = Core::LinAlg::Voigt::modify_voigt_representation(temp6x6, 1.0, 2.0);

  Core::LinAlg::Matrix<6, 6> Np_dyad_Np_V(
      Core::LinAlg::Initialization::zero);  // in stress-strain form
  Np_dyad_Np_V.multiply_nt(1.0, NpVstressform, NpVstrainform, 0.0);
  temp6x6.update(state_quantity_derivatives.curr_dpsr_dequiv_stress, Np_dyad_Np_V,
      equiv_plastic_strain_rate, dNpdMtheta_sym_dev, 0.0);

  /** Now: \f[ \texttt{temp6x6} = \frac{\partial \boldsymbol{D}_p}{\partial
  \boldsymbol{M}_{\theta,\text{symm, dev}}} =
  \frac{\partial\dot{\varepsilon}_p}{\partial\sigma_\text{eq}}
  \boldsymbol{N}_p \otimes \boldsymbol{N}_p + \dot{\varepsilon}_p \frac{\partial
  \boldsymbol{N}_p}{\boldsymbol{M}_{\theta,\text{symm, dev}}}\f]
  (in stress-strain form)
  */

  // compute derivatives of the plastic stretching tensor...
  // ... w.r.t. invese inelastic defgrad
  state_quantity_derivatives.curr_ddpdiFin.multiply_nn(
      1.0, temp6x6, state_quantity_derivatives.curr_dMtheta_dev_sym_diFin, 0.0);
  // ... w.r.t. plastic strain
  state_quantity_derivatives.curr_ddpdepsp.update(
      state_quantity_derivatives.curr_dpsr_depsp, NpVstressform, 0.0);
  // ... w.r.t. right CG
  state_quantity_derivatives.curr_ddpdC.multiply_nn(
      1.0, temp6x6, state_quantity_derivatives.curr_dMtheta_dev_sym_dC, 0.0);
  // ... w.r.t. temperature
  state_quantity_derivatives.curr_ddpdT.multiply_nn(
      1.0, temp6x6, state_quantity_derivatives.curr_dMtheta_dev_sym_dT, 0.0);
  state_quantity_derivatives.curr_ddpdT.update(
      state_quantity_derivatives.curr_dpsr_dT, NpVstressform, 1.0);

  // compute derivatives of the plastic velocity gradient ...

  // ... w.r.t. invese inelastic defgrad
  Core::LinAlg::FourTensor<3> ddpdiFin_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x9_voigt_matrix(
      ddpdiFin_FourTensor, state_quantity_derivatives.curr_ddpdiFin);
  Core::LinAlg::FourTensor<3> id_plus_mm_ddpdiFin_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      id_plus_mm_ddpdiFin_FourTensor, const_mat_tensors_.id_plus_mm, ddpdiFin_FourTensor, true);
  Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(
      temp9x9, id_plus_mm_ddpdiFin_FourTensor);
  state_quantity_derivatives.curr_dlpdiFin.update(1.0, temp9x9, 0.0);
  Core::LinAlg::FourTensor<3> mm_ddpdiFin_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      mm_ddpdiFin_FourTensor, const_mat_tensors_.mm, ddpdiFin_FourTensor, true);
  Core::LinAlg::FourTensor<3> mm_ddpdiFin_T12_FourTensor(true);
  mm_ddpdiFin_T12_FourTensor.transpose_12(mm_ddpdiFin_FourTensor);
  Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(temp9x9, mm_ddpdiFin_T12_FourTensor);
  state_quantity_derivatives.curr_dlpdiFin.update(-1.0, temp9x9, 1.0);

  // ... w.r.t. right CG
  Core::LinAlg::FourTensor<3> ddpdC_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix(
      ddpdC_FourTensor, state_quantity_derivatives.curr_ddpdC);
  Core::LinAlg::FourTensor<3> id_plus_mm_ddpdC_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      id_plus_mm_ddpdC_FourTensor, const_mat_tensors_.id_plus_mm, ddpdC_FourTensor, true);
  Core::LinAlg::Voigt::setup_9x6_voigt_matrix_from_four_tensor(
      temp9x6, id_plus_mm_ddpdC_FourTensor);
  state_quantity_derivatives.curr_dlpdC.update(1.0, temp9x6, 0.0);
  Core::LinAlg::FourTensor<3> mm_ddpdC_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      mm_ddpdC_FourTensor, const_mat_tensors_.mm, ddpdC_FourTensor, true);
  Core::LinAlg::FourTensor<3> mm_ddpdC_T12_FourTensor(true);
  mm_ddpdC_T12_FourTensor.transpose_12(mm_ddpdC_FourTensor);
  Core::LinAlg::Voigt::setup_9x6_voigt_matrix_from_four_tensor(temp9x6, mm_ddpdC_T12_FourTensor);
  state_quantity_derivatives.curr_dlpdC.update(-1.0, temp9x6, 1.0);

  // ... w.r.t. plastic strain
  Core::LinAlg::Matrix<3, 3> ddpdepsp_M(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::vector_to_matrix(
      state_quantity_derivatives.curr_ddpdepsp, ddpdepsp_M);
  Core::LinAlg::Matrix<3, 3> dlpdepsp_M(Core::LinAlg::Initialization::zero);
  dlpdepsp_M.multiply_nn(1.0, const_mat_tensors_.id_plus_mm, ddpdepsp_M, 0.0);
  dlpdepsp_M.multiply_nn(-1.0, ddpdepsp_M, const_mat_tensors_.mm, 1.0);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(dlpdepsp_M, state_quantity_derivatives.curr_dlpdepsp);

  // ... w.r.t. temperature
  Core::LinAlg::Matrix<3, 3> ddpdT_M(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::vector_to_matrix(state_quantity_derivatives.curr_ddpdT, ddpdT_M);
  Core::LinAlg::Matrix<3, 3> dlpdT_M(Core::LinAlg::Initialization::zero);
  dlpdT_M.multiply_nn(1.0, const_mat_tensors_.id_plus_mm, ddpdT_M, 0.0);
  dlpdT_M.multiply_nn(-1.0, ddpdT_M, const_mat_tensors_.mm, 1.0);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(dlpdT_M, state_quantity_derivatives.curr_dlpdT);

  // compute derivatives of the update tensor (only required for standard substepping)
  if (parameter()->timint_type() == ViscoplastUtils::TimIntType::standard)
  {
    // compute argument
    Core::LinAlg::Matrix<3, 3> min_dt_lpM(Core::LinAlg::Initialization::zero);
    min_dt_lpM.update(-1.0 * dt, lpM, 0.0);

    // compute derivative of exponential ...

    // ... w.r.t. its argument
    auto exp_err_status = Core::LinAlg::MatrixFunctErrorType::no_errors;
    Core::LinAlg::Matrix<9, 9> expderivV = Core::LinAlg::matrix_3x3_exp_1st_deriv(
        min_dt_lpM, exp_err_status, parameter()->mat_exp_deriv_calc_method());
    if (exp_err_status != Core::LinAlg::MatrixFunctErrorType::no_errors)
    {
      err_status = ViscoplastUtils::ErrorType::failed_matrix_exp_evaluation;
      return state_quantity_derivatives;
    }

    // ... w.r.t. inverse inelastic defgrad
    state_quantity_derivatives.curr_dEpdiFin.multiply_nn(
        -dt, expderivV, state_quantity_derivatives.curr_dlpdiFin, 0.0);

    // ... w.r.t. right CG
    state_quantity_derivatives.curr_dEpdC.multiply_nn(
        -dt, expderivV, state_quantity_derivatives.curr_dlpdC, 0.0);

    // ... w.r.t. plastic strain
    state_quantity_derivatives.curr_dEpdepsp.multiply_nn(
        -dt, expderivV, state_quantity_derivatives.curr_dlpdepsp, 0.0);

    // ... w.r.t. temperature
    state_quantity_derivatives.curr_dEpdT.multiply_nn(
        -dt, expderivV, state_quantity_derivatives.curr_dlpdT, 0.0);

    return state_quantity_derivatives;
  }
  if (parameter()->timint_type() == ViscoplastUtils::TimIntType::logarithmic)
  {
    // nothing else to be done for logarithmic time integration
    return state_quantity_derivatives;
  }

  return state_quantity_derivatives;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_additional_cmat(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 1>& iCV,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 6>& cmatadd)
{
  const ReducedKinematics reduced_kinematics = evaluate_reduced_kinematics(*defgrad, iFin_other);
  const double temperature = time_step_quantities_.current_temperature[gp_];

  if (parameter()->linearization_type() == ViscoplastUtils::LinearizationType::perturbation_based)
  {
    // compute additional stiffness contribution using perturbation-based approach
    evaluate_additional_cmat_perturb_based(
        reduced_kinematics.defgrad, temperature, cmatadd, dSdiFinj);
    return;
  }

  // declare error status (no errors)
  ViscoplastUtils::ErrorType err_status = ViscoplastUtils::ErrorType::no_errors;
  const auto& diFinjdC = evaluate_history_variables_wrt_cauchy_green(
      reduced_kinematics.right_cauchy_green, temperature, err_status)
                             .inv_plastic_defgrad_wrt_cauchy_green;

  FOUR_C_ASSERT_ALWAYS(err_status == ViscoplastUtils::ErrorType::no_errors,
      "Could not evaluate additional stiffness matrix: {}",
      ViscoplastUtils::get_detailed_error_message_for_error_type(err_status));

  // compute additional term to stiffness matrix additional_cmat
  cmatadd.multiply_nn(2.0, dSdiFinj, diFinjdC, 1.0);
}

ViscoplastUtils::HistoryVariablesDerivativesWrtTemperature
Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_history_variables_wrt_temperature(
    const Core::LinAlg::Matrix<3, 3>& CredM, const double temperature,
    InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status)
{
  if (thermo_mechanical_coupling_cache_.history_variables_wrt_temperature.is_evaluated(gp_))
  {
    err_status = ViscoplastUtils::ErrorType::no_errors;
    return thermo_mechanical_coupling_cache_.history_variables_wrt_temperature.value(gp_);
  }

  state_quantities_ = evaluate_state_quantities(CredM, temperature,
      time_step_quantities_.current_plastic_defgrad_inverse[gp_],
      time_step_quantities_.current_plastic_strain[gp_], err_status, time_step_tracker_.dt,
      ViscoplastUtils::StateQuantityEvalType::full_eval);

  thermo_mechanical_coupling_cache_.state.set(gp_, {state_quantities_});

  // calculate linearization term only if we have plastic strain
  if (std::abs(state_quantities_.curr_equiv_plastic_strain_rate * time_step_tracker_.dt) >
      ViscoplastUtils::zero_plastic_strain_increment)
  {
    // calculate Jacobian
    Core::LinAlg::Matrix<10, 1> current_sol =
        wrap_unknowns(time_step_quantities_.current_plastic_defgrad_inverse[gp_],
            time_step_quantities_.current_plastic_strain[gp_]);

    Core::LinAlg::Matrix<10, 10> jacMat(Core::LinAlg::Initialization::zero);
    viscoplastic_law_->pre_evaluate(params_, gp_);  // set last_substep <- last_
    jacMat = evaluate_local_newton_jacobian(CredM, temperature, current_sol,
        time_step_quantities_.last_plastic_strain[gp_],
        time_step_quantities_.last_plastic_defgrad_inverse[gp_], time_step_tracker_.dt, err_status);
    FOUR_C_ASSERT_ALWAYS(
        err_status == InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors,
        "Could not evaluate Jacobian in off-diagonal stiffness evaluation!");

    // Assert that jacobian is not singular
    FOUR_C_ASSERT_ALWAYS(abs(jacMat.determinant()) > 1.0e-10,
        "Singular Jacobian in off-diagonal stiffness evaluation! Jacobian determinant: {}",
        abs(jacMat.determinant()));

    // declare right-hand side (RHS) terms of the linear system of equations related to the
    // analytical linearization
    Core::LinAlg::Matrix<9, 1> rhs_iFin_V(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<1, 1> rhs_epsp_V(Core::LinAlg::Initialization::zero);

    if (parameter()->timint_type() == ViscoplastUtils::TimIntType::standard)
    // standard time integration
    {
      FOUR_C_THROW(
          "Off-diagonal stiffness integration not yet implemented for standard local time "
          "integration! See evaluate_additional_cmat for an implementation guideline!");
    }
    else if (parameter()->timint_type() == ViscoplastUtils::TimIntType::logarithmic)
    // logarithmic substepping
    {
      // calculate RHS of the equation for the plastic deformation gradient
      rhs_iFin_V.update(-time_step_tracker_.dt, state_quantity_derivatives_.curr_dlpdT, 0.0);

      // calculate RHS of the equation for the plastic strain
      /** \f$\texttt{rhs\_epsp\_V} = - \frac{\partial r_{\varepsilon_{\text{p}}}}{\partial T_{n+1}}
      = \Delta t \cdot \left(\underbrace{\frac{\partial \dot{\varepsilon}_{\text{p}}}{\partial
      \sigma_{\text{yield}}} \cdot
      \frac{\partial\sigma_{\text{yield}}}{\partial T}}_\texttt{curr\_dpsr\_dT} + \frac{\partial
      v_{\text{p}}}{\partial \sigma_\text{eq}}
      \cdot \frac{\partial \sigma_\text{eq}}{\partial T}\right)\f$
      */
      rhs_epsp_V(0) =
          time_step_tracker_.dt * (state_quantity_derivatives_.curr_dpsr_dT +
                                      state_quantity_derivatives_.curr_dpsr_dequiv_stress *
                                          state_quantity_derivatives_.curr_dequiv_stress_dT);
    }
    else
    {
      FOUR_C_THROW("You should not be here");
    }

    // assemble the RHS from its components
    Core::LinAlg::Matrix<10, 1> RHS(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 9; ++i) RHS(i) = rhs_iFin_V(i);
    RHS(9) = rhs_epsp_V(0);

    // solve the linear system of equations
    Core::LinAlg::Matrix<10, 1> SOL(Core::LinAlg::Initialization::zero);
    Core::LinAlg::FixedSizeSerialDenseSolver<10, 10, 1> solver;
    solver.set_matrix(jacMat);     // set A = jacM
    solver.set_vectors(SOL, RHS);  // set X=SOL, B=RHS
    solver.factor_with_equilibration(true);
    int err2 = solver.factor();
    int err = solver.solve();  // X = A^-1 B

    if ((err != 0) || (err2 != 0))
    {
      err_status = ViscoplastUtils::ErrorType::failed_solution_analytic_linearization;
      FOUR_C_THROW("Evaluation of linear system for off-diagonal stiffness has failed: {}",
          get_detailed_error_message_for_error_type(err_status));
    }

    // disassemble the solution vector
    auto extract_inv_plastic_defgrad_wrt_temperature = [&SOL]()
    {
      Core::LinAlg::Matrix<9, 1> result{Core::LinAlg::Initialization::zero};
      for (int i = 0; i < 9; ++i) result(i) = SOL(i);
      return result;
    };

    thermo_mechanical_coupling_cache_.state_derivatives.set(gp_, {state_quantity_derivatives_});
    thermo_mechanical_coupling_cache_.history_variables_wrt_temperature.set(
        gp_, {.inv_plastic_defgrad_wrt_temperature = extract_inv_plastic_defgrad_wrt_temperature(),
                 .plastic_strain_wrt_temperature = SOL(9)});

    return thermo_mechanical_coupling_cache_.history_variables_wrt_temperature.value(gp_);
  }
  else
  {
    thermo_mechanical_coupling_cache_.state_derivatives.set(gp_, {});
    thermo_mechanical_coupling_cache_.history_variables_wrt_temperature.set(gp_, {});

    return thermo_mechanical_coupling_cache_.history_variables_wrt_temperature.value(gp_);
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_od_stiff_mat(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
    Core::LinAlg::Matrix<6, 1>& dstressdT)
{
  const ReducedKinematics reduced_kinematics = evaluate_reduced_kinematics(*defgrad, iFin_other);
  const double temperature = time_step_quantities_.current_temperature[gp_];

  if (parameter()->linearization_type() == ViscoplastUtils::LinearizationType::perturbation_based)
  {
    // compute additional stiffness contribution using perturbation-based approach
    evaluate_od_stiff_mat_perturb_based(
        reduced_kinematics.defgrad, temperature, dstressdT, dSdiFinj);
    return;
  }

  ViscoplastUtils::ErrorType err_status = ViscoplastUtils::ErrorType::no_errors;
  const auto& diFinjTV = evaluate_history_variables_wrt_temperature(
      reduced_kinematics.right_cauchy_green, temperature, err_status)
                             .inv_plastic_defgrad_wrt_temperature;

  FOUR_C_ASSERT_ALWAYS(err_status == ViscoplastUtils::ErrorType::no_errors,
      "Could not evaluate off-diagonal stiffness matrix: {}",
      ViscoplastUtils::get_detailed_error_message_for_error_type(err_status));

  // compute off-diagonal stiffness contribution
  dstressdT.multiply_nn(1.0, dSdiFinj, diFinjTV, 1.0);
}

Mat::HeatSource
Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_taylor_quinney_heat_source(
    const Mat::EvaluationContext<3>& context, const int gp, const int eleGID,
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    const double temperature)
{
  const ReducedKinematics reduced_kinematics = evaluate_reduced_kinematics(*defgrad, iFin_other);

  ViscoplastUtils::ErrorType err_status = ViscoplastUtils::ErrorType::no_errors;

  const double tol = ViscoplastUtils::thermo_mechanical_state_equality_tolerance;

  // First determine current state, then pre_evaluate, because pre_evaluate sets state.
  const bool is_current_state =
      defgrad_difference_norm(
          reduced_kinematics.defgrad, time_step_quantities_.current_defgrad[gp]) < tol &&
      std::abs(temperature - time_step_quantities_.current_temperature[gp]) < tol;

  // set the new temperature and pre-evaluate
  params_.set<double>("temperature", temperature);
  pre_evaluate(params_, context, gp, eleGID);

  if (parameter()->linearization_type() == ViscoplastUtils::LinearizationType::perturbation_based)
  {
    return evaluate_taylor_quinney_heat_source_perturb_based(
        reduced_kinematics.defgrad, temperature);
  }

  // Note: In monolithic tsi, the thermo-predictor comes in with the last state,
  // so it would be possible to store the computed dissipation of the last step and
  // return it instead of re-running returnmapping. This is probably a problem of the
  // tsi predictor though, so not addressed here.

  if (not is_current_state)
  {
    // recompute the current history variables for the incoming state. This also sets the new state.
    return_mapping(reduced_kinematics.defgrad, temperature);
  }

  // evaluate the relevant linearizations, using cached values when available
  const auto history_variables_wrt_cauchy_green = evaluate_history_variables_wrt_cauchy_green(
      reduced_kinematics.right_cauchy_green, temperature, err_status);
  FOUR_C_ASSERT_ALWAYS(err_status == ViscoplastUtils::ErrorType::no_errors,
      "Could not evaluate history variable derivatives for mechanical dissipation evaluation! "
      "Error: {}",
      ViscoplastUtils::get_detailed_error_message_for_error_type(err_status));

  const auto history_variables_wrt_temperature = evaluate_history_variables_wrt_temperature(
      reduced_kinematics.right_cauchy_green, temperature, err_status);
  FOUR_C_ASSERT_ALWAYS(err_status == ViscoplastUtils::ErrorType::no_errors,
      "Could not evaluate history variable derivatives for mechanical dissipation evaluation! "
      "Error: {}",
      ViscoplastUtils::get_detailed_error_message_for_error_type(err_status));

  const auto thermo_mechanical_coupling_state = evaluate_thermo_mechanical_coupling_state(
      reduced_kinematics.right_cauchy_green, temperature, err_status);
  FOUR_C_ASSERT_ALWAYS(err_status == ViscoplastUtils::ErrorType::no_errors,
      "Could not evaluate thermo-mechanical coupling state for mechanical dissipation evaluation! "
      "Error: {}",
      ViscoplastUtils::get_detailed_error_message_for_error_type(err_status));
  const auto thermo_mechanical_coupling_state_derivatives =
      evaluate_thermo_mechanical_coupling_state_derivatives(
          reduced_kinematics.right_cauchy_green, temperature, err_status);
  FOUR_C_ASSERT_ALWAYS(err_status == ViscoplastUtils::ErrorType::no_errors,
      "Could not evaluate thermo-mechanical coupling state derivatives for mechanical dissipation "
      "evaluation! Error: {}",
      ViscoplastUtils::get_detailed_error_message_for_error_type(err_status));

  HeatSource heat_source;
  heat_source.value = parameter()->taylor_quinney_coefficient() *
                      thermo_mechanical_coupling_state.equiv_stress *
                      thermo_mechanical_coupling_state.plastic_strain_rate;
  heat_source.derivative_wrt_cauchy_green = compute_taylor_quinney_wrt_cauchygreen(
      parameter()->taylor_quinney_coefficient(), thermo_mechanical_coupling_state,
      thermo_mechanical_coupling_state_derivatives, history_variables_wrt_cauchy_green);
  heat_source.derivative_wrt_temperature = compute_taylor_quinney_wrt_temperature(
      parameter()->taylor_quinney_coefficient(), thermo_mechanical_coupling_state,
      thermo_mechanical_coupling_state_derivatives, history_variables_wrt_temperature);

  return heat_source;
}

void Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  const ReducedKinematics reduced_kinematics = evaluate_reduced_kinematics(*defgrad, iFin_other);

  // check whether we have already evaluated the inverse inelastic deformation gradient for
  // the given reduced deformation gradient (this check should only be
  // performed for the first repetition, since we want to repeat the
  // return mapping under the same conditions, and not simply return the
  // computed value)
  if (defgrad_difference_norm(
          reduced_kinematics.defgrad, time_step_quantities_.current_defgrad[gp_]) <= 1.0e-16)
  {
    // just set the already computed value, no further computation
    iFinM = time_step_quantities_.current_plastic_defgrad_inverse[gp_];
    return;
  }
  // If this is a "new" deformation gradient, evaluate the inverse inelastic deformation gradient
  // via return mapping.

  // temperature was set by pre_evaluate
  const double temperature =
      params_.isParameter("temperature") ? params_.get<double>("temperature") : ref_temperature_;
  iFinM = return_mapping(reduced_kinematics.defgrad, temperature).inv_plastic_defgrad;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
ViscoplastUtils::ThermoMechanicalCouplingState
Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_thermo_mechanical_coupling_state(
    const Core::LinAlg::Matrix<3, 3>& CredM, const double temperature,
    ViscoplastUtils::ErrorType& err_status)
{
  if (thermo_mechanical_coupling_cache_.state.is_evaluated(gp_))
  {
    return thermo_mechanical_coupling_cache_.state.value(gp_);
  }

  const auto& state_quantities = evaluate_state_quantities(CredM, temperature,
      time_step_quantities_.current_plastic_defgrad_inverse[gp_],
      time_step_quantities_.current_plastic_strain[gp_], err_status, time_step_tracker_.dt,
      ViscoplastUtils::StateQuantityEvalType::full_eval);
  if (err_status != ViscoplastUtils::ErrorType::no_errors) return {};

  thermo_mechanical_coupling_cache_.state.set(gp_, {state_quantities});

  return thermo_mechanical_coupling_cache_.state.value(gp_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
ViscoplastUtils::ThermoMechanicalCouplingStateDerivatives
Mat::InelasticDefgradTransvIsotropElastViscoplast::
    evaluate_thermo_mechanical_coupling_state_derivatives(const Core::LinAlg::Matrix<3, 3>& CredM,
        const double temperature, ViscoplastUtils::ErrorType& err_status)
{
  if (thermo_mechanical_coupling_cache_.state_derivatives.is_evaluated(gp_))
  {
    return thermo_mechanical_coupling_cache_.state_derivatives.value(gp_);
  }

  const auto& state_quantity_derivatives = evaluate_state_quantity_derivatives(CredM, temperature,
      time_step_quantities_.current_plastic_defgrad_inverse[gp_],
      time_step_quantities_.current_plastic_strain[gp_], err_status, time_step_tracker_.dt,
      ViscoplastUtils::StateQuantityDerivEvalType::full_eval, true);
  if (err_status != ViscoplastUtils::ErrorType::no_errors) return {};

  thermo_mechanical_coupling_cache_.state_derivatives.set(gp_, {state_quantity_derivatives});

  return thermo_mechanical_coupling_cache_.state_derivatives.value(gp_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/

ViscoplastUtils::HistoryVariablesDerivativesWrtCauchyGreen
Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_history_variables_wrt_cauchy_green(
    const Core::LinAlg::Matrix<3, 3>& CredM, const double temperature,
    ViscoplastUtils::ErrorType& err_status)
{
  if (thermo_mechanical_coupling_cache_.history_variables_wrt_cauchy_green.is_evaluated(gp_))
  {
    // return the cached values.
    err_status = ViscoplastUtils::ErrorType::no_errors;
    return thermo_mechanical_coupling_cache_.history_variables_wrt_cauchy_green.value(gp_);
  }

  // auxiliaries
  Core::LinAlg::FourTensor<3> tempFourTensor(true);

  state_quantities_ = evaluate_state_quantities(CredM, temperature,
      time_step_quantities_.current_plastic_defgrad_inverse[gp_],
      time_step_quantities_.current_plastic_strain[gp_], err_status, time_step_tracker_.dt,
      ViscoplastUtils::StateQuantityEvalType::full_eval);

  thermo_mechanical_coupling_cache_.state.set(gp_, {state_quantities_});

  // calculate linearization term only if we have plastic strain
  if (std::abs(state_quantities_.curr_equiv_plastic_strain_rate * time_step_tracker_.dt) >
      ViscoplastUtils::zero_plastic_strain_increment)
  {
    // ----- perturbation-based linearization ----- //
    if (err_status != ViscoplastUtils::ErrorType::no_errors) return {};

    // calculate Jacobian
    Core::LinAlg::Matrix<10, 1> current_sol =
        wrap_unknowns(time_step_quantities_.current_plastic_defgrad_inverse[gp_],
            time_step_quantities_.current_plastic_strain[gp_]);

    Core::LinAlg::Matrix<10, 10> jacMat(Core::LinAlg::Initialization::zero);
    viscoplastic_law_->pre_evaluate(params_, gp_);  // set last_substep <- last_
    jacMat = evaluate_local_newton_jacobian(CredM, temperature, current_sol,
        time_step_quantities_.last_plastic_strain[gp_],
        time_step_quantities_.last_plastic_defgrad_inverse[gp_], time_step_tracker_.dt, err_status);

    // if we get singular Jacobian: throw exception -> go to FD-based linearization
    if (abs(jacMat.determinant()) < 1.0e-10)
      err_status = ViscoplastUtils::ErrorType::singular_jacobian;

    if (err_status != ViscoplastUtils::ErrorType::no_errors) return {};


    // declare right-hand side (RHS) terms of the linear system of equations related to the
    // analytical linearization
    Core::LinAlg::Matrix<9, 6> rhs_iFin_V(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<1, 6> rhs_epsp_V(Core::LinAlg::Initialization::zero);

    if (parameter()->timint_type() == ViscoplastUtils::TimIntType::standard)
    // standard time integration
    {
      // calculate RHS of the equation for the plastic deformation gradient
      Core::LinAlg::FourTensor<3> dEpdC_FourTensor(true);
      Core::LinAlg::Voigt::setup_four_tensor_from_9x6_voigt_matrix(
          dEpdC_FourTensor, state_quantity_derivatives_.curr_dEpdC);
      Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(tempFourTensor,
          time_step_quantities_.last_plastic_defgrad_inverse[gp_], dEpdC_FourTensor);
      Core::LinAlg::Voigt::setup_9x6_voigt_matrix_from_four_tensor(rhs_iFin_V, tempFourTensor);

      // calculate RHS of the equation for the plastic strain
      rhs_epsp_V.update(time_step_tracker_.dt * state_quantity_derivatives_.curr_dpsr_dequiv_stress,
          state_quantity_derivatives_.curr_dequiv_stress_dC, 0.0);
    }
    else if (parameter()->timint_type() == ViscoplastUtils::TimIntType::logarithmic)
    // logarithmic substepping
    {
      // calculate RHS of the equation for the plastic deformation gradient
      rhs_iFin_V.update(-time_step_tracker_.dt, state_quantity_derivatives_.curr_dlpdC, 0.0);

      // calculate RHS of the equation for the plastic strain
      rhs_epsp_V.update(time_step_tracker_.dt * state_quantity_derivatives_.curr_dpsr_dequiv_stress,
          state_quantity_derivatives_.curr_dequiv_stress_dC, 0.0);
    }
    else
    {
      FOUR_C_THROW("Inconsistent time integration type {} in evaluating material linearization",
          EnumTools::enum_name(parameter()->timint_type()));
    }

    // assemble the RHS from its components
    Core::LinAlg::Matrix<10, 6> RHS = assemble_rhs_additional_cmat(rhs_iFin_V, rhs_epsp_V);

    // solve the linear system of equations
    Core::LinAlg::Matrix<10, 6> SOL(Core::LinAlg::Initialization::zero);
    Core::LinAlg::FixedSizeSerialDenseSolver<10, 10, 6> solver;
    solver.set_matrix(jacMat);     // set A = jacM
    solver.set_vectors(SOL, RHS);  // set X=SOL, B=RHS
    solver.factor_with_equilibration(true);
    const int err2 = solver.factor();
    const int err = solver.solve();  // X = A^-1 B
    if ((err != 0) || (err2 != 0))
    {
      err_status = ViscoplastUtils::ErrorType::failed_solution_analytic_linearization;
      return {};
    }

    thermo_mechanical_coupling_cache_.state_derivatives.set(gp_, {state_quantity_derivatives_});
    thermo_mechanical_coupling_cache_.history_variables_wrt_cauchy_green.set(gp_,
        {.inv_plastic_defgrad_wrt_cauchy_green = extract_derivative_of_inv_inelastic_defgrad(SOL),
            .plastic_strain_wrt_cauchy_green = extract_derivative_of_plastic_strain(SOL)});

    return thermo_mechanical_coupling_cache_.history_variables_wrt_cauchy_green.value(gp_);
  }
  else
  {
    // linearization terms are zero if no plastic strain increment
    thermo_mechanical_coupling_cache_.state_derivatives.set(gp_, {});
    thermo_mechanical_coupling_cache_.history_variables_wrt_cauchy_green.set(gp_, {});

    return thermo_mechanical_coupling_cache_.history_variables_wrt_cauchy_green.value(gp_);
  }
}

Mat::InelasticDefgradTransvIsotropElastViscoplast::HistoryVariables
Mat::InelasticDefgradTransvIsotropElastViscoplast::return_mapping(
    const Core::LinAlg::Matrix<3, 3>& FredM, const double temperature)
{
  // declare output: history variables (after return mapping)
  HistoryVariables result;

  // compute right CG tensor corresponding to the given deformation gradient
  Core::LinAlg::Matrix<3, 3> CredM(Core::LinAlg::Initialization::zero);
  CredM.multiply_tn(1.0, FredM, FredM, 0.0);

  // perform non-repeatable pre-evaluation tasks (non-repeatable: not
  // called in the redundant evaluate call, which is already handled -> direct return
  // without calling this function)
  prepare_return_mapping();

  // set predictor: assume purely elastic behavior in this time step
  Core::LinAlg::Matrix<3, 3> iFinM_pred(Core::LinAlg::Initialization::zero);
  iFinM_pred.update(1.0, time_step_quantities_.last_plastic_defgrad_inverse[gp_], 0.0);
  double plastic_strain_pred = time_step_quantities_.last_plastic_strain[gp_];
  // declare error status of evaluation (no errors)
  ViscoplastUtils::ErrorType err_status = ViscoplastUtils::ErrorType::no_errors;

  // set current defgrad and current right CG tensor
  time_step_quantities_.current_defgrad[gp_] = FredM;
  time_step_quantities_.current_rightCG[gp_] = CredM;
  time_step_quantities_.current_temperature[gp_] = temperature;
  thermo_mechanical_coupling_cache_.reset(gp_);
  // check whether the predictor is the solution (no plastic strain during this time step)
  bool pred_is_sol =
      check_elastic_predictor(CredM, temperature, iFinM_pred, plastic_strain_pred, err_status);
  if ((err_status == ViscoplastUtils::ErrorType::no_errors) && (pred_is_sol))
  {
    // update inverse inelastic defgrad and plastic strain
    result.inv_plastic_defgrad = iFinM_pred;
    result.plastic_strain = plastic_strain_pred;
  }
  else  // predictor does not suffice
  {
    // perform local time integration
    Core::LinAlg::Matrix<10, 1> x = wrap_unknowns(iFinM_pred, plastic_strain_pred);
    Core::LinAlg::Matrix<10, 1> sol = viscoplastic_correction(FredM, temperature, x, err_status);
    // throw error if the Local Newton Loop cannot be evaluated with the given substepping
    // settings
    if (err_status != ViscoplastUtils::ErrorType::no_errors)
    {
      // output error and then throw (in order to display the error on
      // the right processor)
      const std::string extended_message =
          get_error_info(Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
                  get_detailed_error_message_for_error_type(err_status));
      FOUR_C_THROW("{}", extended_message);
    }

    // update inverse inelastic defgrad and plastic strain
    result.inv_plastic_defgrad = extract_inverse_inelastic_defgrad(sol);
    result.plastic_strain = sol(9);
  }

  // update history variables of material
  if (update_hist_var_)
  {
    time_step_quantities_.current_plastic_defgrad_inverse[gp_] = result.inv_plastic_defgrad;
    time_step_quantities_.current_plastic_strain[gp_] = result.plastic_strain;
    time_step_quantities_.current_equiv_stress[gp_] = state_quantities_.curr_equiv_stress;
    time_step_quantities_.current_rightCG[gp_] = CredM;
    time_step_quantities_.current_defgrad[gp_] = FredM;
  }

  return result;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::update()
{
  // update history variables for the next time step
  time_step_quantities_.update();
  // call update method of the viscoplastic law
  viscoplastic_law_->update();
  // reset Local Newton-Raphson manager
  local_newton_manager_.reset();
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::setup(const int numgp,
    const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // auxiliaries
  Core::LinAlg::Matrix<6, 1> temp_6x1(Core::LinAlg::Initialization::zero);

  // resize time step quantities according to the number of Gauss points
  time_step_quantities_.resize(numgp);
  thermo_mechanical_coupling_cache_.resize(numgp);

  // call corresponding method of the viscoplastic law
  viscoplastic_law_->setup(numgp, fibers, coord_system);

  // setup the Local Newton data tracker with the correct number
  // of Gauss points
  local_newton_manager_.resize(numgp);

  // read fiber and structural tensor in the case of transverse isotropy
  if (parameter()->mat_behavior() == ViscoplastUtils::MatBehavior::transv_isotropic)
  {
    std::vector<Core::LinAlg::Tensor<double, 3>> temp_vec;
    // read fiber via the fiber reader (hyperelastic transversely isotropic material)
    fiber_reader_.setup(numgp, fibers, coord_system);
    fiber_reader_.get_fiber_vecs(temp_vec);
    m_ = Core::LinAlg::make_matrix<3, 1>(temp_vec.back());
  }
  else
  {
    m_.scale(0.0);
  }
  // set material dependent constant tensors
  const_mat_tensors_.set_material_const_tensors(m_);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::pack_inelastic(
    Core::Communication::PackBuffer& data) const
{
  // pack history variables of this specific inelastic factor
  if (parameter() != nullptr)
  {
    // pack viscoplastic law
    viscoplastic_law_->pack_viscoplastic_law(data);

    // pack fiber direction
    add_to_pack(data, m_);

    // pack time_step_quantities_
    time_step_quantities_.pack(data);

    // pack Local Newton manager
    local_newton_manager_.pack(data);
  }
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::unpack_inelastic(
    Core::Communication::UnpackBuffer& buffer)
{
  // NOTE: factory method is called during assign_to_source in the unpack method of the
  // multiplicative split framework --> material created with its params (as well as the
  // viscoplastic law with its params), we only need to unpack the history variablesj
  if (parameter() != nullptr)
  {
    // unpack viscoplastic law
    viscoplastic_law_->unpack_viscoplastic_law(buffer);
    // unpack fiber direction
    extract_from_pack(buffer, m_);
    // unpack time step quantities
    time_step_quantities_.unpack(buffer);
    // unpack the Local Newton manager
    local_newton_manager_.unpack(buffer);
  }

  // now that the fiber direction is available, we set the material-dependent constant tensors
  // with it
  const_mat_tensors_.set_material_const_tensors(m_);

  // resize the thermo-mechanical coupling cache to the correct number of Gauss points
  thermo_mechanical_coupling_cache_.resize(
      time_step_quantities_.current_plastic_defgrad_inverse.size());
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<10, 1>
Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_local_newton_residual(
    const Core::LinAlg::Matrix<3, 3>& CM, const double temperature,
    const Core::LinAlg::Matrix<10, 1>& x, const double last_plastic_strain,
    const Core::LinAlg::Matrix<3, 3>& last_iFinM, const double dt,
    ViscoplastUtils::ErrorType& err_status)
{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);

  // extract inverse inelastic defgrad and plastic strain from input vector
  const Core::LinAlg::Matrix<3, 3> iFinM = extract_inverse_inelastic_defgrad(x);
  const double plastic_strain = x(9);

  // evaluate state variables
  state_quantities_ = evaluate_state_quantities(CM, temperature, iFinM, plastic_strain, err_status,
      dt, ViscoplastUtils::StateQuantityEvalType::full_eval);

  // declare residuals of the LNL
  Core::LinAlg::Matrix<3, 3> resFM(Core::LinAlg::Initialization::zero);
  double resepsp = 0.0;

  // compute residuals (standard time integration)
  if (parameter()->timint_type() == ViscoplastUtils::TimIntType::standard)
  {
    // calculate residual of the equation for inelastic defgrad
    temp3x3.multiply_nn(1.0, last_iFinM, state_quantities_.curr_EpM, 0.0);
    resFM.update(1.0, iFinM, -1.0, temp3x3, 0.0);

    // calculate residual of the equation for plastic strain
    resepsp = plastic_strain - last_plastic_strain -
              dt * state_quantities_.curr_equiv_plastic_strain_rate;
  }
  // compute residuals (logarithmic time integration)
  else if (parameter()->timint_type() == ViscoplastUtils::TimIntType::logarithmic)
  {
    // calculate the tensor logarithm involved in the residual
    Core::LinAlg::Matrix<3, 3> last_FinM(Core::LinAlg::Initialization::zero);
    last_FinM.invert(last_iFinM);
    Core::LinAlg::Matrix<3, 3> T(Core::LinAlg::Initialization::zero);
    T.multiply_nn(1.0, last_FinM, iFinM, 0.0);
    auto log_err_status = Core::LinAlg::MatrixFunctErrorType::no_errors;
    Core::LinAlg::Matrix<3, 3> logT{Core::LinAlg::Initialization::zero};
    if (parameter()->mat_log_calc_method() ==
        Core::LinAlg::MatrixLogCalcMethod::inv_scal_square)  // evaluation using the inverse
                                                             // scaling and squaring
                                                             // method
    {
      // when computing the matrix logarithm with the inverse scaling
      // and squaring, we also save the resulting Pade
      // order via the dedicated pointer. This will be helpful when we
      // compute the derivative - we want consistent Pade orders for the
      // evaluations of functions and their derivatives.
      logT = Core::LinAlg::matrix_log(
          T, log_err_status, matrix_exp_log_utils_.pade_order, parameter()->mat_log_calc_method());
    }
    else  // evaluation using other provided methods
    {
      logT = Core::LinAlg::matrix_log(T, log_err_status, parameter()->mat_log_calc_method());
    }
    if (log_err_status != Core::LinAlg::MatrixFunctErrorType::no_errors)
    {
      err_status = ViscoplastUtils::ErrorType::failed_matrix_log_evaluation;
      return Core::LinAlg::Matrix<10, 1>{Core::LinAlg::Initialization::zero};
    }

    // calculate residual of the equation for inelastic defgrad
    resFM.update(1.0, logT, dt, state_quantities_.curr_lpM, 0.0);

    // calculate residual of the equation for plastic strain
    resepsp = plastic_strain - last_plastic_strain -
              dt * state_quantities_.curr_equiv_plastic_strain_rate;
  }
  else
  {
    FOUR_C_THROW("Inconsistent time integration type {} in evaluating Local Newton residual",
        EnumTools::enum_name(parameter()->timint_type()));
  }

  // return 10x1 residual vector
  Core::LinAlg::Matrix<10, 1> residual;
  residual(0) = resFM(0, 0);
  residual(1) = resFM(1, 1);
  residual(2) = resFM(2, 2);
  residual(3) = resFM(0, 1);
  residual(4) = resFM(1, 2);
  residual(5) = resFM(0, 2);
  residual(6) = resFM(1, 0);
  residual(7) = resFM(2, 1);
  residual(8) = resFM(2, 0);
  residual(9) = resepsp;

  return residual;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<10, 10>
Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_local_newton_jacobian(
    const Core::LinAlg::Matrix<3, 3>& CM, const double temperature,
    const Core::LinAlg::Matrix<10, 1>& x, const double last_plastic_strain,
    const Core::LinAlg::Matrix<3, 3>& last_iFinM, const double dt,
    ViscoplastUtils::ErrorType& err_status)
{
  // auxiliaries
  Core::LinAlg::FourTensor<3> tempFourTensor(true);
  Core::LinAlg::Matrix<9, 9> temp9x9(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);

  // extract inverse inelastic defgrad and plastic strain from input vector
  const Core::LinAlg::Matrix<3, 3> iFinM = extract_inverse_inelastic_defgrad(x);
  const double plastic_strain = x(9);

  // evaluate state derivatives
  state_quantity_derivatives_ = evaluate_state_quantity_derivatives(CM, temperature, iFinM,
      plastic_strain, err_status, dt,
      ViscoplastUtils::StateQuantityDerivEvalType::full_eval);  // we do not reevaluate the state
                                                                // quantities, this was done in the
                                                                // residual computation already

  // get derivative of update tensor wrt inverse inelastic defgrad (in FourTensor form)
  Core::LinAlg::FourTensor<3> dEpdiFin_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(
      dEpdiFin_FourTensor, state_quantity_derivatives_.curr_dEpdiFin);

  // declare Jacobian component blocks

  // derivative of residual for inelastic deformation gradient w.r.t. inelastic deformation
  // gradient
  Core::LinAlg::Matrix<9, 9> J_iFin_iFin(Core::LinAlg::Initialization::zero);
  // derivative of residual for inelastic deformation gradient w.r.t. plastic strain
  Core::LinAlg::Matrix<9, 1> J_iFin_epsp(Core::LinAlg::Initialization::zero);
  // derivative of residual for plastic strain w.r.t. inelastic deformation gradient
  Core::LinAlg::Matrix<1, 9> J_epsp_iFin(Core::LinAlg::Initialization::zero);
  // derivative of residual for plastic strain w.r.t. plastic strain
  double J_epsp_epsp = 0.0;

  // standard time integration
  if (parameter()->timint_type() == ViscoplastUtils::TimIntType::standard)
  {
    // compute 9x9 north-west component block of the Jacobian (derivative of residual for
    // inelastic deformation gradient w.r.t. inelastic deformation gradient)
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        tempFourTensor, last_iFinM, dEpdiFin_FourTensor, true);
    Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(temp9x9, tempFourTensor);
    J_iFin_iFin.update(1.0, const_non_mat_tensors.id4_9x9, -1.0, temp9x9, 0.0);

    // compute derivative of update tensor wrt plastic strain in matrix form
    Core::LinAlg::Matrix<3, 3> dEpdepsp_M(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::matrix_9x1_to_3x3(state_quantity_derivatives_.curr_dEpdepsp, dEpdepsp_M);

    // compute 9x1 north-east component block of the Jacobian (derivative of residual for
    // inelastic deformation gradient w.r.t. plastic strain)
    temp3x3.multiply_nn(-1.0, last_iFinM, dEpdepsp_M, 0.0);
    Core::LinAlg::Voigt::matrix_3x3_to_9x1(temp3x3, J_iFin_epsp);

    // compute 1x9 south-west component block of the Jacobian (derivative of residual for
    // plastic strain w.r.t. inelastic deformation gradient)
    J_epsp_iFin.update(-dt * state_quantity_derivatives_.curr_dpsr_dequiv_stress,
        state_quantity_derivatives_.curr_dequiv_stress_diFin, 0.0);

    // compute south-east component of the Jacobian (derivative of residual for plastic
    // strain w.r.t. plastic strain)
    J_epsp_epsp = 1.0 - dt * state_quantity_derivatives_.curr_dpsr_depsp;
  }
  else if (parameter()->timint_type() == ViscoplastUtils::TimIntType::logarithmic)
  // logarithmic time integration
  {
    // compute 9x9 north-west component block of the Jacobian (derivative of residual for
    // inelastic deformation gradient w.r.t. inelastic deformation gradient)
    Core::LinAlg::Matrix<3, 3> last_FinM(Core::LinAlg::Initialization::zero);
    last_FinM.invert(last_iFinM);
    Core::LinAlg::Matrix<3, 3> T(Core::LinAlg::Initialization::zero);
    T.multiply_nn(1.0, last_FinM, iFinM, 0.0);
    auto log_err_status = Core::LinAlg::MatrixFunctErrorType::no_errors;
    Core::LinAlg::Matrix<9, 9> dlogTdT{Core::LinAlg::Initialization::zero};
    if ((parameter()->mat_log_deriv_calc_method() ==
            Core::LinAlg::GenMatrixLogFirstDerivCalcMethod::
                pade_part_fract))  // evaluation using the Pade partial fraction
                                   // expansion?...
    {
      // check whether the logarithm was evaluated with the inverse
      // scaling and squaring method, for which we have also determined
      // a suitable Pade order -> if not so, then we throw error, since
      // this is the only implemented case for now!
      FOUR_C_ASSERT_ALWAYS(
          parameter()->mat_log_calc_method() == Core::LinAlg::MatrixLogCalcMethod::inv_scal_square,
          "Combination of logarithm evaluation methods not implemented yet!");

      dlogTdT = Core::LinAlg::matrix_3x3_log_1st_deriv(T, log_err_status,
          matrix_exp_log_utils_.pade_order, parameter()->mat_log_deriv_calc_method());
    }
    else  // evaluation using other provided methods?...
    {
      dlogTdT = Core::LinAlg::matrix_3x3_log_1st_deriv(
          T, log_err_status, parameter()->mat_log_deriv_calc_method());
    }

    if (log_err_status != Core::LinAlg::MatrixFunctErrorType::no_errors)
    {
      err_status = ViscoplastUtils::ErrorType::failed_matrix_log_evaluation;
      return Core::LinAlg::Matrix<10, 10>{Core::LinAlg::Initialization::zero};
    }
    Core::LinAlg::Matrix<9, 9> dTdiFin(Core::LinAlg::Initialization::zero);
    Core::LinAlg::FourTensorOperations::add_non_symmetric_product(
        1.0, last_FinM, const_non_mat_tensors.id3x3, dTdiFin);
    Core::LinAlg::Matrix<9, 9> dlogTdiFin(Core::LinAlg::Initialization::zero);
    dlogTdiFin.multiply_nn(1.0, dlogTdT, dTdiFin, 0.0);
    J_iFin_iFin.update(1.0, dlogTdiFin, dt, state_quantity_derivatives_.curr_dlpdiFin, 0.0);

    // compute 9x1 north-east component block of the Jacobian (derivative of residual for
    // inelastic deformation gradient w.r.t. plastic strain)
    J_iFin_epsp.update(dt, state_quantity_derivatives_.curr_dlpdepsp, 0.0);

    // compute 1x9 south-west component block of the Jacobian (derivative of residual for
    // plastic strain w.r.t. inelastic deformation gradient)
    J_epsp_iFin.update(-dt * state_quantity_derivatives_.curr_dpsr_dequiv_stress,
        state_quantity_derivatives_.curr_dequiv_stress_diFin, 0.0);

    // compute south-east component of the Jacobian (derivative of residual for plastic
    // strain w.r.t. plastic strain)
    J_epsp_epsp = 1.0 - dt * state_quantity_derivatives_.curr_dpsr_depsp;
  }
  else
  {
    FOUR_C_THROW("Inconsistent time integration type {} in evaluating Jacobian",
        EnumTools::enum_name(parameter()->timint_type()));
  }

  // assemble and return the Jacobian
  return assemble_jacobian_from_components(J_iFin_iFin, J_iFin_epsp, J_epsp_iFin, J_epsp_epsp);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<10, 1>
Mat::InelasticDefgradTransvIsotropElastViscoplast::viscoplastic_correction(
    const Core::LinAlg::Matrix<3, 3>& defgrad, const double temperature,
    const Core::LinAlg::Matrix<10, 1>& x, ViscoplastUtils::ErrorType& err_status)
{
  // calculate right Cauchy-Green deformation tensor
  Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
  CM.multiply_tn(1.0, defgrad, defgrad, 0.0);

  // define solution vector
  Core::LinAlg::Matrix<10, 1> sol = x;

  // declare current right CG (tensor interpolated later on in each substep)
  Core::LinAlg::Matrix<3, 3> curr_CM(Core::LinAlg::Initialization::zero);
  // declare current temperature (interpolated later on in each substep)
  double curr_temp = 0.0;

  // reset substep parameters
  if (parameter()->use_local_substepping())
  {
    local_substepping_utils_.reset(time_step_tracker_.dt);
  }

  // initialize tensor interpolation error status
  Core::LinAlg::TensorInterpolationErrorType tensor_interp_err_status =
      Core::LinAlg::TensorInterpolationErrorType::NoErrors;

  // local substepping procedures
  if (parameter()->use_local_substepping())
  {
    while (!local_substepping_utils_.end_substepping())
    {
      // interpolate right Cauchy-Green tensor if we use local substepping
      curr_CM = tensor_interpolator_.get_interpolated_matrix(
          {time_step_quantities_.last_rightCG[gp_], CM}, {0.0, 1.0},
          local_substepping_utils_.get_normalized_next_time_param(time_step_tracker_.dt),
          tensor_interp_err_status);
      FOUR_C_ASSERT_ALWAYS(
          tensor_interp_err_status == Core::LinAlg::TensorInterpolationErrorType::NoErrors,
          "Tensor interpolation failed with err: {}",
          Core::LinAlg::make_error_message(tensor_interp_err_status));
      // interpolate temperature
      curr_temp = std::lerp(time_step_quantities_.last_temperature[gp_], temperature,
          local_substepping_utils_.get_normalized_next_time_param(time_step_tracker_.dt));

      // perform substep local Newton loop
      local_newton_loop(curr_CM, curr_temp, time_step_quantities_.last_substep_plastic_strain[gp_],
          time_step_quantities_.last_substep_plastic_defgrad_inverse[gp_],
          local_substepping_utils_.get_substep_size(), sol, err_status);
      // update Local Newton quantities
      local_newton_manager_.update_after_local_newton(gp_);

      // update substep
      if (err_status == InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors)
      {
        // this means the current substep has converged: we increment the substep and update the
        // values relevant for substepping, including history data
        local_substepping_utils_.increment_substep();

        // update the values of history variables at the last converged state
        time_step_quantities_.last_substep_plastic_defgrad_inverse[gp_] =
            extract_inverse_inelastic_defgrad(sol);
        time_step_quantities_.last_substep_plastic_strain[gp_] = sol(9);
        // update last substep history variables of the viscoplastic flow rule
        viscoplastic_law_->update_gp_state(gp_);
      }
      else
      {
        // halve and prepare a new substep
        bool halving_success = halve_and_prepare_new_substep(sol, CM);
        // if the halving number was exceeded --> return with error
        if (!halving_success)
        {
          const std::string extended_message =
              get_error_info(Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
                      get_detailed_error_message_for_error_type(err_status));
          FOUR_C_THROW("{}", extended_message);
        }
      }
    }
  }
  // one-step correction
  else
  {
    // perform local Newton loop
    local_newton_loop(CM, temperature, time_step_quantities_.last_plastic_strain[gp_],
        time_step_quantities_.last_plastic_defgrad_inverse[gp_], time_step_tracker_.dt, sol,
        err_status);

    // update Local Newton quantities and reset iteration counter
    local_newton_manager_.update_after_local_newton(gp_);
  }


  // return the obtained solution
  return sol;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::local_newton_loop(
    const Core::LinAlg::Matrix<3, 3>& CM, const double temperature,
    const double last_plastic_strain, const Core::LinAlg::Matrix<3, 3>& last_iFinM, const double dt,
    Core::LinAlg::Matrix<10, 1>& sol,
    InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status)
{
  // auxiliaries
  Core::LinAlg::Matrix<10, 1> temp10x1(Core::LinAlg::Initialization::zero);

  // Jacobian matrix
  Core::LinAlg::Matrix<10, 10> jacMat(Core::LinAlg::Initialization::zero);
  // increment of the solution variables
  Core::LinAlg::Matrix<10, 1> dx(Core::LinAlg::Initialization::zero);
  // residual of both equations
  Core::LinAlg::Matrix<10, 1> residual(Core::LinAlg::Initialization::zero);

  // initialize quantities checked for convergence
  ViscoplastUtils::LocalNewtonConvQuantities conv_quantities{
      .residual_norm = 1.0, .increment_norm = 1.0};

  // initialize evaluation management action
  ViscoplastUtils::EvaluationAction eval_action{
      ViscoplastUtils::EvaluationAction::continue_current_iteration};

  // reset Local Newton iteration count
  local_newton_manager_.set_iteration_count(0);

  // local Newton-Raphson loop
  while (true)
  {
    // set error status to no_errors
    err_status = ViscoplastUtils::ErrorType::no_errors;

    // increment iteration counter
    local_newton_manager_.increment_iteration_count();

    // evaluate residual
    residual = evaluate_local_newton_residual(
        CM, temperature, sol, last_plastic_strain, last_iFinM, dt, err_status);

    // error management after residual evaluation
    temp10x1.update(1.0, sol, 0.0);
    manage_evaluation(err_status, eval_action);
    switch (eval_action)
    {
      case (ViscoplastUtils::EvaluationAction::continue_current_iteration):
      {
        // continue evaluation
        break;
      }
      case (ViscoplastUtils::EvaluationAction::continue_with_next_iteration):
      {
        // recompute dx after conducting adjustments to solution vector
        dx.update(1.0, sol, -1.0, temp10x1, 0.0);

        // proceed with next iteration after performing adjustments due
        // to errors
        continue;
      }
      case (ViscoplastUtils::EvaluationAction::exit_with_error):
      {
        // exit with the set error status
        return;
      }
      default:
      {
        FOUR_C_THROW("Invalid evaluation action {} for error status {} after residual evaluation",
            EnumTools::enum_name(eval_action), EnumTools::enum_name(err_status));
      }
    }


    // if we continue, then the residual evaluation was successful

    // verify convergence
    conv_quantities.residual_norm = residual.norm2();
    const bool is_converged = is_local_newton_converged(conv_quantities);

    // exit in case of convergence
    if (is_converged)
    {
      return;
    }

    // check if maximum iteration is exceeded
    if (local_newton_manager_.iter() > local_newton_manager_.params().max_iter)
    {
      // set non-convergence error
      err_status =
          InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_convergence_local_newton;


      // irrespective of divergence continuation strategy: for local substepping, we should look
      // for a smaller substep size; hence, we return with the set error status
      if (parameter()->use_local_substepping())
      {
        return;
      }
      // for one-step processes, we account for the set divergence continuation strategy
      else
      {
        verify_local_newton_exit(conv_quantities, err_status);
        return;
      }
    }
    else
    {
      // check whether the Local Newton is 'stuck'
      if (is_local_newton_stuck(conv_quantities))
      {
        // error management routine after the 'stuck' verification
        err_status = InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::
            no_convergence_local_newton;
        temp10x1.update(1.0, sol, 0.0);
        manage_evaluation(err_status, eval_action);
        switch (eval_action)
        {
          case (ViscoplastUtils::EvaluationAction::continue_current_iteration):
          {
            // continue evaluation
            break;
          }
          case (ViscoplastUtils::EvaluationAction::continue_with_next_iteration):
          {
            // recompute dx after conducting adjustments to solution vector
            dx.update(1.0, sol, -1.0, temp10x1, 0.0);

            // proceed with next iteration after performing adjustments due
            // to errors
            continue;
          }
          case (ViscoplastUtils::EvaluationAction::exit_with_error):
          {
            // exit with the set error status
            return;
          }
          default:
          {
            FOUR_C_THROW(
                "Invalid evaluation action {} for error status {} after verification of stuck "
                "Local Newton",
                EnumTools::enum_name(eval_action), EnumTools::enum_name(err_status));
          }
        }
      }
    }

    // evaluate Jacobian
    jacMat = evaluate_local_newton_jacobian(
        CM, temperature, sol, last_plastic_strain, last_iFinM, dt, err_status);
    // error management after Jacobian evaluation
    manage_evaluation(err_status, eval_action);
    switch (eval_action)
    {
      case (ViscoplastUtils::EvaluationAction::continue_current_iteration):
      {
        // continue evaluation
        break;
      }
      case (ViscoplastUtils::EvaluationAction::continue_with_next_iteration):
      {
        // proceed with next iteration after performing adjustments due
        // to errors
        continue;
      }
      case (ViscoplastUtils::EvaluationAction::exit_with_error):
      {
        // exit with the set error status
        return;
      }
      default:
      {
        FOUR_C_THROW("Invalid evaluation action {} for error status {} after Jacobian evaluation",
            EnumTools::enum_name(eval_action), EnumTools::enum_name(err_status));
      }
    }


    // solve linear system
    const bool successful_solve = solve_local_newton_linear_system(residual, jacMat, dx);
    if (!successful_solve)
    {
      err_status = ViscoplastUtils::ErrorType::failed_solution_linear_system_lnl;
      // error management after linear system solution
      manage_evaluation(err_status, eval_action);
      switch (eval_action)
      {
        case (ViscoplastUtils::EvaluationAction::continue_current_iteration):
        {
          // continue evaluation
          break;
        }
        case (ViscoplastUtils::EvaluationAction::continue_with_next_iteration):
        {
          // proceed with next iteration after performing adjustments due
          // to errors
          continue;
        }
        case (ViscoplastUtils::EvaluationAction::exit_with_error):
        {
          // exit with the set error status
          return;
        }
        default:
        {
          FOUR_C_THROW(
              "Invalid evaluation action {} for error status {} after solving linear system",
              EnumTools::enum_name(eval_action), EnumTools::enum_name(err_status));
        }
      }
    }

    // update solution vector and relative increment
    sol.update(1.0, dx, 1.0);
    const double sol_norm = sol.norm2();
    const double dx_norm = dx.norm2();
    FOUR_C_ASSERT_ALWAYS(sol_norm >= 1.0e-8,
        "The solution vector in local iteration {} is nearly 0, with 2-norm: {}! Something went "
        "wrong, since such mechanical states are not expected!",
        local_newton_manager_.iter(), sol_norm);
    conv_quantities.increment_norm = dx_norm / sol_norm;
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplast::is_local_newton_converged(
    const InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvQuantities&
        conv_quantities)
{
  // check for convergence
  switch (local_newton_manager_.params().conv_check)
  {
    case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::residual:
      return (conv_quantities.residual_norm <= local_newton_manager_.params().res_tol);
      break;
    case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::increment_ratio:
      return (conv_quantities.increment_norm <= local_newton_manager_.params().incr_tol);
      break;
    case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::
        residual_and_increment_ratio:
      return (conv_quantities.residual_norm <= local_newton_manager_.params().res_tol &&
              conv_quantities.increment_norm <= local_newton_manager_.params().incr_tol);
      break;
    default:
      FOUR_C_THROW("You should not be here (convergence checking of the Local Newton Loop)");
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplast::is_local_newton_stuck(
    const ViscoplastUtils::LocalNewtonConvQuantities& conv_quantities)
{
  // check for "stuck" Local Newton, i.e., the increment does not change much but there is not a
  // converged state (check only feasible after the first iteration, since dx must be available)
  if ((local_newton_manager_.iter() > 1) && (conv_quantities.increment_norm < 1.0e-15))
  {
    // only in the case that the residual is verified, we set an
    // error status
    switch (local_newton_manager_.params().conv_check)
    {
      case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::residual:
      case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::
          residual_and_increment_ratio:
      {
        return (conv_quantities.residual_norm > local_newton_manager_.params().res_tol);
      }
      case ViscoplastUtils::LocalNewtonConvCheck::increment_ratio:
      {
        return false;
      }
      default:
        FOUR_C_THROW(
            "You should not be here with convergence check type {} (check: is Local Newton "
            "stuck?)",
            EnumTools::enum_name(local_newton_manager_.params().conv_check));
    }
  }

  return false;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::verify_local_newton_exit(
    const ViscoplastUtils::LocalNewtonConvQuantities& conv_quantities,
    InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status)
{
  switch (local_newton_manager_.params().diver_cont)
  {
    case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonDiverCont::stop:
    {
      // throw error: there is no convergence
      const std::string extended_message =
          get_error_info(Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
                  get_detailed_error_message_for_error_type(err_status));
      FOUR_C_THROW("{}", extended_message);
    }
    case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonDiverCont::
        continue_sim:
    case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonDiverCont::
        continue_sim_with_safeguard:
    {
      // show warning that convergence could not be reached
      if (show_warnings(ele_gid_))
      {
        std::cout << std::format(
            "WARNING: The Local Newton Loop for ele_gid = {}, gp = {} did not reach "
            "convergence after {} iterations: residual = {}, increment = {}\n",
            ele_gid_, gp_, local_newton_manager_.iter(), conv_quantities.residual_norm,
            conv_quantities.increment_norm);
      }

      // safeguard check: is the current solution within the bounds posed by the
      // maximum exceedance?
      if (local_newton_manager_.params().diver_cont ==
          InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonDiverCont::
              continue_sim_with_safeguard)
      {
        const bool residual_within_bounds =
            conv_quantities.residual_norm <
            (local_newton_manager_.params().res_tol *
                local_newton_manager_.params().max_exceedance_fact_res_tol);
        const bool incr_ratio_within_bounds =
            conv_quantities.increment_norm <
            (local_newton_manager_.params().incr_tol *
                local_newton_manager_.params().max_exceedance_fact_incr_tol);

        switch (local_newton_manager_.params().conv_check)
        {
          case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::
              residual:
          {
            FOUR_C_ASSERT_ALWAYS(residual_within_bounds,
                "Residual {} exceeds the residual tolerance {} by more than the set "
                "exceedance tolerance factor {}!",
                conv_quantities.residual_norm, local_newton_manager_.params().res_tol,
                local_newton_manager_.params().max_exceedance_fact_res_tol);

            break;
          }
          case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::
              increment_ratio:
          {
            FOUR_C_ASSERT_ALWAYS(incr_ratio_within_bounds,
                "Relative increment {} exceeds the increment tolerance {} by more "
                "than the set exceedance tolerance factor {}!",
                conv_quantities.increment_norm, local_newton_manager_.params().incr_tol,
                local_newton_manager_.params().max_exceedance_fact_incr_tol);

            break;
          }
          case FourC::Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::
              residual_and_increment_ratio:
          {
            FOUR_C_ASSERT_ALWAYS(residual_within_bounds && incr_ratio_within_bounds,
                "Residual {} and relative increment {} exceed the tolerances {} and {} by "
                "more than the set exceedance tolerance factors {} and {}!",
                conv_quantities.residual_norm, conv_quantities.increment_norm,
                local_newton_manager_.params().res_tol, local_newton_manager_.params().incr_tol,
                local_newton_manager_.params().max_exceedance_fact_res_tol,
                local_newton_manager_.params().max_exceedance_fact_incr_tol);

            break;
          }
          default:
            FOUR_C_THROW("Invalid convergence check {} (verification of safe Local Newton exit)",
                EnumTools::enum_name(local_newton_manager_.params().conv_check));
        }
      }

      // if we have reached this stage, then we set the error status to no errors (in order to
      // safely proceed) and return
      err_status = InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors;
      return;
    }
    default:
      FOUR_C_THROW(
          "You should not be here (divergence management strategy for Local Newton "
          "Loop)");
  }

  // safeguard for the function: each path must either return of throw
  FOUR_C_THROW("The Local Newton scheme cannot be safely exited! Uncaught exception with error {}",
      err_status);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplast::solve_local_newton_linear_system(
    const Core::LinAlg::Matrix<10, 1>& residual, const Core::LinAlg::Matrix<10, 10>& jacobian,
    Core::LinAlg::Matrix<10, 1>& dx)
{
  // auxiliaries: use copies of the residual and jacobian to avoid modifying the original variables
  Core::LinAlg::Matrix<10, 1> temp_negative_residual(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<10, 10> temp_jacobian(Core::LinAlg::Initialization::zero);

  // store residual and jacobian
  temp_negative_residual.update(-1.0, residual, 0.0);
  temp_jacobian.update(1.0, jacobian, 0.0);

  // solve linear system
  Core::LinAlg::FixedSizeSerialDenseSolver<10, 10, 1> solver_10_10_1;
  dx.clear();                                              // reset
  solver_10_10_1.set_matrix(temp_jacobian);                // set A=jacMat
  solver_10_10_1.set_vectors(dx, temp_negative_residual);  // set dx=increment, residual=RHS
  solver_10_10_1.factor_with_equilibration(true);          // "some easy type of preconditioning"
  int err2 = solver_10_10_1.factor();                      // factoring
  int err = solver_10_10_1.solve();                        // X = A^-1 B
  return (err == 0) && (err2 == 0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplast::check_elastic_predictor(
    const Core::LinAlg::Matrix<3, 3>& CM, const double temperature,
    const Core::LinAlg::Matrix<3, 3>& iFinM_pred, const double plastic_strain_pred,
    ViscoplastUtils::ErrorType& err_status)
{
  // evaluate state with this elastic predictor and the minimum possible time step
  state_quantities_ = evaluate_state_quantities(CM, temperature, iFinM_pred, plastic_strain_pred,
      err_status, time_step_tracker_.min_dt,
      ViscoplastUtils::StateQuantityEvalType::plastic_strain_rate_only);


  // check if the predicted plastic strain rate is 0 -> for flow rules with yield functions,
  // this means that the predictor is correct
  return (state_quantities_.curr_equiv_plastic_strain_rate * time_step_tracker_.dt <=
          ViscoplastUtils::zero_plastic_strain_increment);
}

bool Mat::InelasticDefgradTransvIsotropElastViscoplast::halve_and_prepare_new_substep(
    Core::LinAlg::Matrix<10, 1>& sol, const Core::LinAlg::Matrix<3, 3>& curr_CM)
{
  // the current iteration vector has reached a numerically inevaluable state -> we halve
  // the time step and apply substepping

  // halve the current time step
  local_substepping_utils_.halve_substep();

  // check if we have halved the time step too many times
  if (local_substepping_utils_.get_halving_counter() >
      parameter()->max_local_substepping_halve_num())
  {
    return false;
  }

  // reset the predictor to the last converged state
  sol = wrap_unknowns(time_step_quantities_.last_substep_plastic_defgrad_inverse[gp_],
      time_step_quantities_.last_substep_plastic_strain[gp_]);

  return true;  // no error
}

void Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_additional_cmat_perturb_based(
    const Core::LinAlg::Matrix<3, 3>& FredM, const double temperature,
    Core::LinAlg::Matrix<6, 6>& cmatadd, const Core::LinAlg::Matrix<6, 9>& dSdiFinj)
{
  // ----- FD-based linearization ----- //
  // approximation using perturbations of the right Cauchy-Green deformation tensor,
  // inspired by the procedure described in Miehe et al. (1995)

  // auxiliaries
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);

  // set update boolean to false
  update_hist_var_ = false;  // no update of the current_ values during the upcoming evaluation
  // of perturbed states
  const auto unperturbed_state_quantities = state_quantities_;
  const auto unperturbed_time_step_quantities = time_step_quantities_;

  // inverse of the reduced deformation gradient
  Core::LinAlg::Matrix<3, 3> iFredM(Core::LinAlg::Initialization::zero);
  iFredM.invert(FredM);

  // Voigt representation of the inverse inelastic defgrad
  Core::LinAlg::Matrix<9, 1> iFinV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(
      time_step_quantities_.current_plastic_defgrad_inverse[gp_], iFinV);

  // derivative of inverse inelastic deformation gradient w.r.t. right Cauchy-Green
  // deformation tensor, to be evaluated in the FD-based procedure
  Core::LinAlg::Matrix<9, 6> diFindC_FD(Core::LinAlg::Initialization::zero);

  // declare perturbed variables
  Core::LinAlg::Matrix<3, 3> perturbed_FM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> perturbed_CM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> perturbed_iFinM(Core::LinAlg::Initialization::zero);

  // define the delta perturbed deformation gradients
  std::vector<Core::LinAlg::Matrix<3, 3>> delta_perturbed_defgrads(6);
  const double pert_fact = 1.0e-5;  // perturbation factor \f$ \epsilon \f$
  const std::vector<std::tuple<int, int>> indices_array = {
      {0, 0}, {1, 1}, {2, 2}, {0, 1}, {1, 2}, {0, 2}};

  // vary deformation gradient (and therefore the right Cauchy-Green tensor), calculate
  // resulting inverse inelastic defgrad, and compute the contribution to the required
  // derivative
  for (int i = 0; i < static_cast<int>(indices_array.size()); i++)
  {
    // set perturbation of the form
    // \f$ \Delta F_{pert(CD)} = \epsilon/2 F^{-T} (E_{C} \otimes  E_{D} + E_{D} \otimes
    // E_{C} ) \f$
    temp3x3.clear();
    temp3x3(std::get<0>(indices_array[i]), std::get<1>(indices_array[i])) += pert_fact / 2.0;
    temp3x3(std::get<1>(indices_array[i]), std::get<0>(indices_array[i])) += pert_fact / 2.0;
    delta_perturbed_defgrads[i].multiply_tn(1.0, iFredM, temp3x3, 0.0);

    for (const double delta_sign : {1.0, -1.0})
    {
      // get perturbed defgrad
      perturbed_FM.update(1.0, FredM, delta_sign, delta_perturbed_defgrads[i], 0.0);

      // calculate perturbed right CG tensor
      perturbed_CM.multiply_tn(1.0, perturbed_FM, perturbed_FM, 0.0);

      // get corresponding inverse inelastic defgrad
      perturbed_iFinM = return_mapping(perturbed_FM, temperature).inv_plastic_defgrad;
      Core::LinAlg::Matrix<9, 1> perturbed_iFinV(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Voigt::matrix_3x3_to_9x1(perturbed_iFinM, perturbed_iFinV);

      // update components of the required derivative
      for (int j = 0; j < 9; ++j)
      {
        diFindC_FD(j, i) += delta_sign / (4.0 * pert_fact) * (perturbed_iFinV(j, 0) - iFinV(j, 0));
      }
    }
  }

  // compute additional term to stiffness matrix additional_cmat
  cmatadd.multiply_nn(2.0, dSdiFinj, diFindC_FD, 1.0);

  // restore the state quantities to the unperturbed state
  state_quantities_ = unperturbed_state_quantities;
  time_step_quantities_ = unperturbed_time_step_quantities;
  // reset boolean for the history update
  update_hist_var_ = true;
}

void Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_od_stiff_mat_perturb_based(
    const Core::LinAlg::Matrix<3, 3>& FredM, const double temperature,
    Core::LinAlg::Matrix<6, 1>& dstressdT, const Core::LinAlg::Matrix<6, 9>& dSdiFinj)
{
  update_hist_var_ = false;
  const auto unperturbed_state_quantities = state_quantities_;
  const auto unperturbed_time_step_quantities = time_step_quantities_;

  const double perturbation_factor = 1.0e-7;
  const double delta_T_perturbation = std::max(perturbation_factor * temperature,
      perturbation_factor);  // ensure a minimum perturbation size for small temperatures

  const std::array<double, 2> perturbed_temperatures = {temperature + delta_T_perturbation,
      std::max(
          0.0, temperature - delta_T_perturbation)};  // ensure non-negative absolute temperature

  std::array<double, 2> delta_signs = {1.0, -1.0};
  const double delta_T_perturbation_used = perturbed_temperatures[0] - perturbed_temperatures[1];


  Core::LinAlg::Matrix<9, 1> diFindT_FD(Core::LinAlg::Initialization::zero);

  for (int index : {0, 1})
  {
    const double perturbed_temperature = perturbed_temperatures[index];

    // set the perturbed temperature in the viscoplastic law.
    params_.set<double>("temperature", perturbed_temperature);
    viscoplastic_law_->pre_evaluate(params_, gp_);

    // run return-mapping with the perturbed temperature
    const HistoryVariables perturbed_history_variables =
        return_mapping(FredM, perturbed_temperature);

    Core::LinAlg::Matrix<9, 1> perturbed_iFinV(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::matrix_3x3_to_9x1(
        perturbed_history_variables.inv_plastic_defgrad, perturbed_iFinV);

    diFindT_FD.update(delta_signs[index] / delta_T_perturbation_used, perturbed_iFinV, 1.0);
  }

  // update dstressdT by the contribution of the variation of the inverse inelastic defgrad with
  // respect to temperature
  dstressdT.multiply_nn(1.0, dSdiFinj, diFindT_FD, 1.0);

  // reset everything to the unperturbed state
  time_step_quantities_ = unperturbed_time_step_quantities;
  params_.set<double>("temperature", temperature);
  viscoplastic_law_->pre_evaluate(params_, gp_);
  state_quantities_ = unperturbed_state_quantities;
  update_hist_var_ = true;
}

Mat::HeatSource Mat::InelasticDefgradTransvIsotropElastViscoplast::
    evaluate_taylor_quinney_heat_source_perturb_based(
        const Core::LinAlg::Matrix<3, 3>& FredM, const double temperature)
{
  Core::LinAlg::Matrix<3, 3> CredM{Core::LinAlg::Initialization::zero};
  CredM.multiply_tn(1.0, FredM, FredM, 0.0);

  Mat::HeatSource result{};

  const auto evaluate_taylor_quinney = [this](const ViscoplastUtils::StateQuantities& state)
  {
    return parameter()->taylor_quinney_coefficient() * state.curr_equiv_plastic_strain_rate *
           state.curr_equiv_stress;
  };

  // Evaluate the incoming state
  params_.set<double>("temperature", temperature);
  viscoplastic_law_->pre_evaluate(params_, gp_);
  return_mapping(FredM, temperature);

  update_hist_var_ = false;  // no update of the current values during perturbed evaluations
  const auto unperturbed_state_quantities = state_quantities_;
  const auto unperturbed_time_step_quantities = time_step_quantities_;

  result.value = evaluate_taylor_quinney(unperturbed_state_quantities);

  if (std::abs(unperturbed_state_quantities.curr_equiv_plastic_strain_rate *
               time_step_tracker_.dt) <= ViscoplastUtils::zero_plastic_strain_increment)
  {
    // no dissipation if plastic strain rate is 0.
    update_hist_var_ = true;
    return result;
  }

  // Finite-difference linearization w.r.t. the right Cauchy-Green tensor.
  {
    Core::LinAlg::Matrix<3, 3> iFredM(Core::LinAlg::Initialization::zero);
    iFredM.invert(FredM);

    Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);

    Core::LinAlg::Matrix<3, 3> perturbed_FM(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 3> perturbed_CM(Core::LinAlg::Initialization::zero);

    std::vector<Core::LinAlg::Matrix<3, 3>> delta_perturbed_defgrads(6);
    const double pert_fact = 1.0e-5;  // perturbation factor \f$ \epsilon \f$
    const std::vector<std::tuple<int, int>> indices_array = {
        {0, 0}, {1, 1}, {2, 2}, {0, 1}, {1, 2}, {0, 2}};

    for (int i = 0; i < static_cast<int>(indices_array.size()); i++)
    {
      // set perturbation of the form
      // \f$ \Delta F_{pert(CD)} = \epsilon/2 F^{-T} (E_{C} \otimes  E_{D} + E_{D} \otimes
      // E_{C} ) \f$
      temp3x3.clear();
      temp3x3(std::get<0>(indices_array[i]), std::get<1>(indices_array[i])) += pert_fact / 2.0;
      temp3x3(std::get<1>(indices_array[i]), std::get<0>(indices_array[i])) += pert_fact / 2.0;
      delta_perturbed_defgrads[i].multiply_tn(1.0, iFredM, temp3x3, 0.0);

      for (const double delta_sign : {1.0, -1.0})
      {
        perturbed_FM.update(1.0, FredM, delta_sign, delta_perturbed_defgrads[i], 0.0);
        perturbed_CM.multiply_tn(1.0, perturbed_FM, perturbed_FM, 0.0);

        // run return-mapping with the perturbed deformation gradient. This populates the
        // state_quantities_
        return_mapping(perturbed_FM, temperature);

        result.derivative_wrt_cauchy_green(i) +=
            delta_sign / (4.0 * pert_fact) * evaluate_taylor_quinney(state_quantities_);
      }
    }
  }

  // Finite-difference linearization w.r.t. temperature.
  {
    const double pert_fact = 1.0e-7;
    double delta_T_perturbation = std::max(pert_fact * std::abs(temperature),
        pert_fact);  // avoid 0 perturbation

    for (double delta_sign : {1.0, -1.0})
    {
      double perturbed_temperature = temperature + delta_T_perturbation * delta_sign;

      // The viscoplastic law caches temperature-dependent data during pre-evaluation.
      params_.set<double>("temperature", perturbed_temperature);
      viscoplastic_law_->pre_evaluate(params_, gp_);

      // run return-mapping with the perturbed temperature. This populates the state_quantities_
      return_mapping(FredM, perturbed_temperature);

      result.derivative_wrt_temperature +=
          delta_sign / (2 * delta_T_perturbation) * evaluate_taylor_quinney(state_quantities_);
    }
  }

  // reset everything to the unperturbed state
  time_step_quantities_ = unperturbed_time_step_quantities;
  state_quantities_ = unperturbed_state_quantities;

  // reset the temperature in the viscoplastic law
  params_.set<double>("temperature", temperature);
  viscoplastic_law_->pre_evaluate(params_, gp_);
  update_hist_var_ = true;

  return result;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::manage_evaluation(
    const InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status,
    InelasticDefgradTransvIsotropElastViscoplastUtils::EvaluationAction& eval_action) const
{
  // default evaluation action: continue iteration
  eval_action = InelasticDefgradTransvIsotropElastViscoplastUtils::EvaluationAction::
      continue_current_iteration;

  // return directly if there is no evaluation error
  if (err_status == ViscoplastUtils::ErrorType::no_errors)
  {
    return;
  }
  else
  {
    // ERROR MANAGEMENT STRATEGY 1: substepping -> just exit, and see if a new halved
    // substep size is feasible
    if (parameter()->use_local_substepping())
    {
      eval_action = ViscoplastUtils::EvaluationAction::exit_with_error;
      return;
    }
    else
    {
      FOUR_C_THROW(
          "The Local Newton evaluation has failed with err status {} and there is no evaluation "
          "management strategy "
          "selected!",
          err_status);
    }
  }
}



/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
std::string Mat::InelasticDefgradTransvIsotropElastViscoplast::get_error_info(
    const std::string& base_error_string) const
{
  // auxiliaries
  std::ostringstream temp_ostream;

  // set output format for the numbers -> we can set it here for the
  // entire error message
  temp_ostream << std::fixed << std::setprecision(16) << std::endl;

  // declare the extended error message
  std::string extended_error_string{local_substepping_utils_.get_info()};

  // get relevant error info
  extended_error_string += "BASE ERROR: \n";
  extended_error_string += base_error_string + "\n";
  extended_error_string +=
      "-> At EleID: " + std::to_string(ele_gid_) + ". At GP: " + std::to_string(gp_) + ".\n";
  extended_error_string += std::string(10, '.') + "\n";

  // add the material parameters
  extended_error_string += "PARAMETERS: \n";
  parameter()->raw_parameters().print(temp_ostream);
  viscoplastic_law_->parameter()->raw_parameters().print(temp_ostream);
  temp_ostream << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += std::string(10, '.') + "\n";

  // add the relevant last_ values
  extended_error_string += "LAST_ VALUES: \n";
  extended_error_string += "last_plastic_defgrad_inverse: \n";
  time_step_quantities_.last_plastic_defgrad_inverse[gp_].print(temp_ostream);
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += "last_substep_plastic_defgrad_inverse: \n";
  time_step_quantities_.last_substep_plastic_defgrad_inverse[gp_].print(temp_ostream);
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += "last_plastic_strain: \n";
  extended_error_string += "Double<1,1> \n";
  temp_ostream << time_step_quantities_.last_plastic_strain[gp_] << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");

  extended_error_string += "last_substep_plastic_strain: \n";
  extended_error_string += "Double<1,1> \n";
  temp_ostream << time_step_quantities_.last_substep_plastic_strain[gp_] << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");


  extended_error_string += "last_rightCG: \n";
  time_step_quantities_.last_rightCG[gp_].print(temp_ostream);
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  // add the current values
  extended_error_string += "CURRENT_ VALUES: \n";
  extended_error_string += "current_defgrad: \n";
  time_step_quantities_.current_defgrad[gp_].print(temp_ostream);
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += "current_rightCG: \n";
  time_step_quantities_.current_rightCG[gp_].print(temp_ostream);
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += "current_temperature: \n";
  extended_error_string += "Double<1,1> \n";
  temp_ostream << time_step_quantities_.current_temperature[gp_] << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");

  extended_error_string += std::string(10, '.');


  // add the current values
  extended_error_string += "OTHER VALUES: \n";
  extended_error_string += "dt: \n";
  extended_error_string += "Double<1,1> \n";
  temp_ostream << time_step_tracker_.dt << std::endl;
  extended_error_string += temp_ostream.str();
  temp_ostream.str("");
  extended_error_string += std::string(10, '.');


  return extended_error_string;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  names_and_size["inverse_plastic_defgrad"] = 9;
  names_and_size["plastic_strain"] = 1;
  names_and_size["equiv_stress"] = 1;
  names_and_size["defgrad"] = 9;
  names_and_size["rightCG"] = 9;
  names_and_size["local_newton_iters"] = 1;
  viscoplastic_law_->register_output_data_names(names_and_size);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  // auxiliaries
  Core::LinAlg::Matrix<9, 1> temp9x1{Core::LinAlg::Initialization::zero};

  if (name == "inverse_plastic_defgrad")
  {
    for (int gp = 0;
        gp < static_cast<int>(time_step_quantities_.current_plastic_defgrad_inverse.size()); ++gp)
    {
      Core::LinAlg::Voigt::matrix_3x3_to_9x1(
          time_step_quantities_.current_plastic_defgrad_inverse[gp], temp9x1);

      for (int col = 0; col < 9; ++col)
      {
        data(gp, col) = temp9x1(col);
      }
    }
    return true;
  }
  else if (name == "plastic_strain")
  {
    for (int gp = 0; gp < static_cast<int>(time_step_quantities_.current_plastic_strain.size());
        ++gp)
    {
      data(gp, 0) = time_step_quantities_.current_plastic_strain[gp];
    }
    return true;
  }
  else if (name == "equiv_stress")
  {
    for (int gp = 0; gp < static_cast<int>(time_step_quantities_.current_equiv_stress.size()); ++gp)
    {
      data(gp, 0) = time_step_quantities_.current_equiv_stress[gp];
    }
    return true;
  }
  else if (name == "defgrad")
  {
    for (int gp = 0; gp < static_cast<int>(time_step_quantities_.current_defgrad.size()); ++gp)
    {
      Core::LinAlg::Voigt::matrix_3x3_to_9x1(time_step_quantities_.current_defgrad[gp], temp9x1);

      for (int col = 0; col < 9; ++col)
      {
        data(gp, col) = temp9x1(col);
      }
    }
    return true;
  }
  else if (name == "rightCG")
  {
    for (int gp = 0; gp < static_cast<int>(time_step_quantities_.current_rightCG.size()); ++gp)
    {
      Core::LinAlg::Voigt::matrix_3x3_to_9x1(time_step_quantities_.current_rightCG[gp], temp9x1);


      for (int col = 0; col < 9; ++col)
      {
        data(gp, col) = temp9x1(col);
      }
    }
    return true;
  }
  else if (name == "local_newton_iters")
  {
    for (int gp = 0; gp < static_cast<int>(local_newton_manager_.curr_num_iters().size()); ++gp)
    {
      data(gp, 0) = local_newton_manager_.curr_num_iters()[gp];
    }
    return true;
  }


  return viscoplastic_law_->evaluate_output_data(name, data);
}

FOUR_C_NAMESPACE_CLOSE
