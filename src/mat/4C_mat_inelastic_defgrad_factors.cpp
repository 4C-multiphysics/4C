// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_inelastic_defgrad_factors.hpp"

#include "4C_global_data.hpp"
#include "4C_legacy_enum_definitions_materials.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor_generators.hpp"
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
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_vplast_law.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

FOUR_C_NAMESPACE_OPEN
namespace
{


  // struct: constant non-material tensors
  struct ConstNonMatTensors
  {
    static const ConstNonMatTensors& instance()
    {
      static ConstNonMatTensors instance;
      return instance;
    }

    // constructor
    ConstNonMatTensors()
    {  // auxiliaries
      Core::LinAlg::Matrix<3, 3> id3x3(Core::LinAlg::Initialization::zero);
      for (int i = 0; i < 3; ++i) id3x3(i, i) = 1.0;
      Core::LinAlg::Matrix<6, 6> temp6x6(Core::LinAlg::Initialization::zero);

      // set constant non-material tensors

      // 3x3 identity
      id3x3_.update(1.0, id3x3, 0.0);

      // Voigt stress form of 3x3 identity
      Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
          id3x3_, id6x1_);

      // symmetric identity four tensor
      Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
          id4_6x6_, 1.0, id3x3, id3x3, 0.0);

      // deviatoric operator
      Core::LinAlg::FourTensor<3> dev_op_four_tensor =
          Core::LinAlg::setup_deviatoric_projection_tensor<3>();
      Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, dev_op_four_tensor);
      dev_op_ = Core::LinAlg::Voigt::modify_voigt_representation(temp6x6, 1.0, 2.0);

      // identity four tensor
      id4_9x9_.clear();
      Core::LinAlg::FourTensorOperations::add_non_symmetric_product(1.0, id3x3_, id3x3_, id4_9x9_);

      // 10x10 identity
      id10x10_.clear();
      for (int i = 0; i < 10; ++i) id10x10_(i, i) = 1.0;
    }

    // second-order 3x3 identity tensor in matrix form \f$ \boldsymbol{I} \f$
    Core::LinAlg::Matrix<3, 3> id3x3_{Core::LinAlg::Initialization::zero};
    // second-order 3x3 identity in Voigt stress form \f$ \boldsymbol{I} \f$
    Core::LinAlg::Matrix<6, 1> id6x1_{Core::LinAlg::Initialization::zero};
    // symmetric identity four tensor of dimension 3 \f$ \mathbb{I}_\text{S} \f$
    Core::LinAlg::Matrix<6, 6> id4_6x6_{Core::LinAlg::Initialization::zero};
    // deviatoric operator \f$ \mathbb{P}_{\text{dev}}  =  \mathbb{I}_\text{S} -
    // \frac{1}{3} \boldsymbol{I} \otimes \boldsymbol{I} \f$
    Core::LinAlg::Matrix<6, 6> dev_op_{Core::LinAlg::Initialization::zero};
    // identity fourth-order tensor in Voigt notation: delta_AC delta_BD in index notation
    Core::LinAlg::Matrix<9, 9> id4_9x9_{Core::LinAlg::Initialization::zero};
    // second-order 10x10 identity tensor in matrix form
    Core::LinAlg::Matrix<10, 10> id10x10_{Core::LinAlg::Initialization::zero};
  };

  // declare file-scope instance of the constant non-material tensors
  static ConstNonMatTensors const_non_mat_tensors = ConstNonMatTensors::instance();

  // read input parameter container of parent material (i.e., underlying
  // multiplicative_split_defgrad_elasthyper material)
  Core::IO::InputParameterContainer get_parameters_of_parent_material(const int mat_id,
      const std::map<int, Core::Utils::LazyPtr<Core::Mat::PAR::Parameter>>& material_map)
  {
    std::map<int, Core::Utils::LazyPtr<Core::Mat::PAR::Parameter>>::const_iterator m;
    // go over material map
    for (m = material_map.begin(); m != material_map.end(); ++m)
    {
      // check type of parent material
      if (m->second->type() == Core::Materials::m_multiplicative_split_defgrad_elasthyper)
      {
        std::vector<int> inel_defgrad_facids_ =
            m->second->raw_parameters().get<std::vector<int>>("INELDEFGRADFACIDS");
        // check if the inelastic defgrad factor ids of the found multiplicative split material
        // contain the id of the considered factor
        if (std::find((inel_defgrad_facids_).begin(), (inel_defgrad_facids_).end(), mat_id) !=
            (inel_defgrad_facids_).end())
        {
          return m->second->raw_parameters();
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
    auto* curmat = Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
    switch (curmat->type())
    {
      case Core::Materials::m_electrode:
      {
        // Get C_max and Chi_max of electrode material
        c_max_ = curmat->raw_parameters().get<double>("C_MAX");
        chi_max_ = curmat->raw_parameters().get<double>("CHI_MAX");
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
      growth_dir_(std::make_shared<InelasticDeformationDirection>(
          matdata.parameters.get<std::vector<double>>("GrowthDirection")))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradPolyIntercalFracAniso::InelasticDefgradPolyIntercalFracAniso(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : InelasticDefgradPolyIntercalFrac(matdata),
      growth_dir_(std::make_shared<InelasticDeformationDirection>(
          matdata.parameters.get<std::vector<double>>("GrowthDirection")))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDeformationDirection::InelasticDeformationDirection(
    std::vector<double> growthdirection)
    : growth_dir_mat_()
{
  if (growthdirection.size() != 3)
  {
    FOUR_C_THROW(
        "Since we have a 3D problem here, vector that defines the growth direction also needs to "
        "have the size 3!");
  }

  // fill matrix that determines the growth direction
  const double growthdirvecnorm =
      std::sqrt(std::pow(growthdirection[0], 2.0) + std::pow(growthdirection[1], 2.0) +
                std::pow(growthdirection[2], 2.0));
  const double invquadrgrowthdirvecnorm = 1.0 / (growthdirvecnorm * growthdirvecnorm);

  // loop over all rows and columns to fill the matrix and scale it correctly on the fly
  for (unsigned i = 0; i < growthdirection.size(); ++i)
  {
    for (unsigned j = 0; j < growthdirection.size(); ++j)
    {
      growth_dir_mat_(i, j) = invquadrgrowthdirvecnorm * growthdirection[i] * growthdirection[j];
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
Mat::PAR::InelasticDefgradTimeFunct::InelasticDefgradTimeFunct(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata), funct_num_(matdata.parameters.get<int>("FUNCT_NUM"))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticDefgradTransvIsotropElastViscoplast::
    InelasticDefgradTransvIsotropElastViscoplast(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      viscoplastic_law_id_(matdata.parameters.get<int>("VISCOPLAST_LAW_ID")),
      fiber_reader_gid_(matdata.parameters.get<int>("FIBER_READER_ID")),
      yield_cond_a_(matdata.parameters.get<double>("YIELD_COND_A")),
      yield_cond_b_(matdata.parameters.get<double>("YIELD_COND_B")),
      yield_cond_f_(matdata.parameters.get<double>("YIELD_COND_F")),
      bool_transv_isotropy_(
          read_anisotropy_type(matdata.parameters.get<std::string>("ANISOTROPY"))),
      bool_log_substepping_(matdata.parameters.get<bool>("LOG_SUBSTEP")),
      max_halve_number_(matdata.parameters.get<int>("MAX_HALVE_NUM_SUBSTEP"))
{
  if (max_halve_number_ < 0) FOUR_C_THROW("Parameter MAX_HALVE_NUM_SUBSTEP must be >= 0!");
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
        "STRUCTURAL DYNAMIC Section to 'None', or feel free to implement other possibility!");
  }

  // retrieve problem instance to read from
  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
  // retrieve validated input line of material ID in question
  auto* curmat = Global::Problem::instance(probinst)->materials()->parameter_by_id(matnum);

  // get material type and call corresponding constructors
  const Core::Materials::MaterialType currentMaterialType = curmat->type();
  switch (currentMaterialType)
  {
    case Core::Materials::mfi_no_growth:
    {
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradNoGrowth*>(curmat);

      return std::make_shared<InelasticDefgradNoGrowth>(params);
    }
    case Core::Materials::mfi_lin_scalar_aniso:
    {
      // get pointer to parameter class
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradLinScalarAniso*>(curmat);

      // return pointer to inelastic deformation gradient object
      return std::make_shared<InelasticDefgradLinScalarAniso>(params);
    }
    case Core::Materials::mfi_lin_scalar_iso:
    {
      // get pointer to parameter class
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradScalar*>(curmat);

      // return pointer to inelastic deformation gradient object
      return std::make_shared<InelasticDefgradLinScalarIso>(params);
    }
    case Core::Materials::mfi_poly_intercal_frac_aniso:
    {
      // get pointer to parameter class
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradPolyIntercalFracAniso*>(curmat);

      // return pointer to inelastic deformation gradient object
      return std::make_shared<InelasticDefgradPolyIntercalFracAniso>(params);
    }
    case Core::Materials::mfi_poly_intercal_frac_iso:
    {
      // get pointer to parameter class
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradPolyIntercalFrac*>(curmat);

      // return pointer to inelastic deformation gradient object
      return std::make_shared<InelasticDefgradPolyIntercalFracIso>(params);
    }

    case Core::Materials::mfi_lin_temp_iso:
    {
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradLinTempIso*>(curmat);
      return std::make_shared<InelasticDefgradLinTempIso>(params);
    }
    case Core::Materials::mfi_time_funct:
    {
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradTimeFunct*>(curmat);
      return std::make_shared<InelasticDefgradTimeFunct>(params);
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
      auto* params = dynamic_cast<Mat::PAR::InelasticDefgradTransvIsotropElastViscoplast*>(curmat);

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
      for (int matid_elastic : parentmat_input_params.get<std::vector<int>>("MATIDSEL"))
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

      // return shared pointer to the inelastic factor
      return std::make_shared<InelasticDefgradTransvIsotropElastViscoplast>(
          params, viscoplastic_law, fiber_reader, potsumel, potsumel_transviso);
    }

    default:
      FOUR_C_THROW("cannot deal with type {}", curmat->type());
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
    const EvaluationContext& context, const int gp, const int eleGID)
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
    : InelasticDefgradScalar(params)
{
  polynomial_growth_ = std::make_shared<InelasticDefgradPolynomialShape>(
      parameter()->poly_coeffs(), parameter()->x_min(), parameter()->x_max());

  // get reference intercalation fraction
  const double x_ref = Mat::Electrode::compute_intercalation_fraction(
      parameter()->scalar1_ref_conc(), parameter()->chimax(), parameter()->cmax(), 1.0);

  // set the polynomial value in the reference configuration
  parameter()->set_polynom_reference_value(polynomial_growth_->compute_polynomial(x_ref));
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::InelasticDefgradPolyIntercalFrac::evaluate_polynomial(
    const double concentration, const double detjacobian)
{
  // get intercalation fraction
  const double x = Mat::Electrode::compute_intercalation_fraction(
      concentration, parameter()->chimax(), parameter()->cmax(), detjacobian);

  // check bounds of validity of polynomial
  polynomial_growth_->check_polynomial_bounds(x);

  // calculate and return the value of the polynomial
  return polynomial_growth_->compute_polynomial(x);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::InelasticDefgradPolyIntercalFrac::evaluate_polynomial_derivative(
    const double concentration, const double detjacobian)
{
  // get intercalation fraction
  const double x = Mat::Electrode::compute_intercalation_fraction(
      concentration, parameter()->chimax(), parameter()->cmax(), detjacobian);

  return polynomial_growth_->compute_polynomial_derivative(x);
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
    : InelasticDefgradScalar(params)
{
  linear_growth_ = std::make_shared<InelasticDefgradLinearShape>(
      parameter()->scalar1_molar_growth_fac(), parameter()->scalar1_ref_conc());
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
  const double growth_factor = linear_growth_->evaluate_linear_growth(material_concentration);

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
  const double sc1GrowthFac = linear_growth_->growth_fac();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double detjacobian = defgrad->determinant();

  // get growth factor
  const double growth_factor = linear_growth_->evaluate_linear_growth(concentration * detjacobian);

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
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<3, 3>& iFinjM,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 1>& dstressdc)
{
  static Core::LinAlg::Matrix<9, 1> id9x1(Core::LinAlg::Initialization::zero);
  // prepare id9x1 (identity matrix written as a 9x1 vector)
  for (int i = 0; i < 3; ++i) id9x1(i) = 1.0;

  // get parameters
  const int sc1 = parameter()->scalar1();
  const double sc1GrowthFac = linear_growth_->growth_fac();
  const double detjacobian = defgrad->determinant();
  const double material_concentration = get_concentration_gp().at(sc1 - 1) * detjacobian;

  // get growth factor
  const double growth_factor = linear_growth_->evaluate_linear_growth(material_concentration);

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
  const double growth_factor = linear_growth_->evaluate_linear_growth(material_concentration);
  // calculate the scale factor needed to calculate the derivative below
  const double scalefac = 1.0 / 3.0 * std::pow(1 + growth_factor, -2.0 / 3.0) *
                          linear_growth_->growth_fac() * detjacobian;

  dFindx =
      Core::LinAlg::get_full(scalefac * Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradLinScalarAniso::InelasticDefgradLinScalarAniso(
    Core::Mat::PAR::Parameter* params)
    : InelasticDefgradScalar(params)
{
  linear_growth_ = std::make_shared<InelasticDefgradLinearShape>(
      parameter()->scalar1_molar_growth_fac(), parameter()->scalar1_ref_conc());
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
  const double growth_factor = linear_growth_->evaluate_linear_growth(material_concentration);

  // calculate inelastic deformation gradient and its inverse
  for (int i = 0; i < 3; ++i) FinM(i, i) = 1.0;

  // finalize inelastic deformation gradient matrix (FinM is calculated, such that the volume
  // change is a linear function of the scalar (mapped to reference frame) that causes it)
  Fin += growth_factor * Core::LinAlg::get_full(parameter()->growth_dir_mat());

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
  const double sc1GrowthFac = linear_growth_->growth_fac();
  const double concentration = get_concentration_gp().at(sc1 - 1);
  const double detjacobian = defgrad->determinant();

  // prepare scalefac
  const double scalefac = -sc1GrowthFac * concentration * detjacobian / 2.0;

  // calculate F_{in,j}^{-1} . G . F_{in,j}^{-1} with F_{in,j}, the j-th factor of F_{in}
  temp.multiply_nn(1.0, iFinjM,
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(parameter()->growth_dir_mat())), 0.0);
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
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<3, 3>& iFinjM,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 1>& dstressdc)
{
  // static variables
  static Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> diFinjdcM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 1> diFinjdc9x1(Core::LinAlg::Initialization::zero);

  // get parameters
  const double sc1GrowthFac = linear_growth_->growth_fac();
  const double detjacobian = defgrad->determinant();

  // prepare scalefac
  const double scalefac = -sc1GrowthFac * detjacobian;

  // calculate diFinjdc
  tmp.multiply_nn(1.0, iFinjM,
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(parameter()->growth_dir_mat())), 0.0);
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
  const double scalefac = linear_growth_->growth_fac() * detjacobian;

  dFindx = Core::LinAlg::get_full(parameter()->growth_dir_mat());
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
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<3, 3>& iFinjM,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 1>& dstressdc)
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
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(parameter()->growth_dir_mat())), 1.0);

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
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(parameter()->growth_dir_mat())), 0.0);
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
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<3, 3>& iFinjM,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 1>& dstressdc)
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
      Core::LinAlg::make_matrix(Core::LinAlg::get_full(parameter()->growth_dir_mat())), 0.0);
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
  dFindx = Core::LinAlg::get_full(scalefac * parameter()->growth_dir_mat());
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradLinearShape::InelasticDefgradLinearShape(
    const double growth_fac, const double reference_value)
    : growth_fac_(growth_fac), reference_value_(reference_value)
{
  // safety checks
  if (growth_fac < 0.0)
    FOUR_C_THROW("Growth factor can not be negative, please check your input file!");
  if (growth_fac == 0.0)
  {
    FOUR_C_THROW(
        "Do not use linear growth laws with a growth factor of 0.0. Use "
        "'MAT_InelasticDefgradNoGrowth' instead!");
  }
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
double Mat::InelasticDefgradPolynomialShape::compute_polynomial(const double x)
{
  // initialize the variable for the evaluation of the polynomial
  double polynom(0.0);

  // compute polynomial
  for (unsigned i = 0; i < poly_coeffs_.size(); ++i) polynom += poly_coeffs_[i] * std::pow(x, i);

  return polynom;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::InelasticDefgradPolynomialShape::compute_polynomial_derivative(const double x)
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
    std::cout << "WARNING: Polynomial is evaluated outside its range of validity!" << std::endl;
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
    const EvaluationContext& context, const int gp, const int eleGID)
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
  if (growthfactor <= 0.0) FOUR_C_THROW("Determinante of growth must not become negative");
  const double isoinelasticdefo = std::pow(growthfactor, (1.0 / 3.0));

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
    const Core::LinAlg::Matrix<3, 3>* const defgrad, const Core::LinAlg::Matrix<3, 3>& iFinjM,
    const Core::LinAlg::Matrix<6, 9>& dSdiFinj, Core::LinAlg::Matrix<6, 1>& dstressdT)
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
    const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
    Core::LinAlg::Matrix<6, 1>& dstressdx)
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
    const EvaluationContext& context, const int gp, const int eleGID)
{
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTimeFunct::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  const double idetFin = std::pow(funct_value_, -1.0 / 3.0);
  iFinM.update(idetFin, identity_, 0.0);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::InelasticSource Mat::InelasticDefgradTimeFunct::get_inelastic_source()
{
  return PAR::InelasticSource::none;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTimeFunct::InelasticDefgradTimeFunct(Core::Mat::PAR::Parameter* params)
    : InelasticDefgradFactors(params),
      funct_value_(0.0),
      identity_(Core::LinAlg::Initialization::zero)
{
  // add 1.0 to main diagonal
  identity_(0, 0) = identity_(1, 1) = identity_(2, 2) = 1.0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTimeFunct::pre_evaluate(const Teuchos::ParameterList& params,
    const EvaluationContext& context, const int gp, const int eleGID)
{
  // evaluate function value for current time step.
  auto& funct = Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfTime>(
      parameter()->funct_num());
  FOUR_C_ASSERT(context.total_time, "Time not given in evaluation context.");
  const double time = *context.total_time;
  funct_value_ = funct.evaluate(time);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplast::InelasticDefgradTransvIsotropElastViscoplast(
    Core::Mat::PAR::Parameter* params, std::shared_ptr<Mat::Viscoplastic::Law> viscoplastic_law,
    Mat::Elastic::CoupTransverselyIsotropic fiber_reader,
    std::vector<std::shared_ptr<Mat::Elastic::Summand>> pot_sum_el,
    std::vector<std::shared_ptr<Mat::Elastic::CoupTransverselyIsotropic>> pot_sum_el_transv_iso)
    : InelasticDefgradFactors(params),
      potsumel_(std::move(pot_sum_el)),
      potsumel_transviso_(std::move(pot_sum_el_transv_iso)),
      viscoplastic_law_(std::move(viscoplastic_law)),
      fiber_reader_(std::move(fiber_reader)),
      tensor_interpolator_(init_tensor_interpolator())
{
  // set time step size to 0.0 (this is set to the correct and current value in the preevaluate
  // method)
  time_step_settings_.dt_ = 0.0;
  // set minimum substep length
  time_step_settings_.min_dt_ = 0.0;

  // ----- set last_ and current_ variables referring to values at different time instants -----
  // for now: the number of Gauss points is unknown -> we set the values only for 1 Gauss point
  // and update the number of Gauss points in the setup method

  // default values of the inverse plastic deformation gradient: unit tensor
  time_step_quantities_.last_plastic_defgrd_inverse_.resize(1, const_non_mat_tensors.id3x3_);
  time_step_quantities_.current_plastic_defgrd_inverse_.resize(
      1, const_non_mat_tensors.id3x3_);  // value irrelevant at this point
  time_step_quantities_.last_substep_plastic_defgrd_inverse_.resize(
      1, const_non_mat_tensors.id3x3_);

  // update last_ and current_ values of the plastic strain
  time_step_quantities_.last_plastic_strain_.resize(1, 0.0);
  time_step_quantities_.current_plastic_strain_.resize(1, 0.0);  // value irrelevant at this point
  time_step_quantities_.last_substep_plastic_strain_.resize(1, 0.0);

  // default values of the right CG tensor: unit tensor
  time_step_quantities_.last_rightCG_.resize(1, const_non_mat_tensors.id3x3_);
  time_step_quantities_.current_rightCG_.resize(
      1, const_non_mat_tensors.id3x3_);  // value irrelevant at this point

  // default value for the current deformation gradient: zero tensor \f$ \boldsymbol{0} f$ (to make
  // sure that the inverse inelastic deformation gradient is evaluated in the first method call)
  time_step_quantities_.current_defgrad_.resize(
      1, Core::LinAlg::Matrix<3, 3>{Core::LinAlg::Initialization::zero});
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::pre_evaluate(
    const Teuchos::ParameterList& params, const EvaluationContext& context, const int gp,
    const int eleGID)
{
  // set Gauss Point
  gp_ = gp;

  // set element ID
  ele_gid_ = eleGID;

  // set time step
  FOUR_C_ASSERT(context.time_step_size, "Time step size not given in evaluation context.");
  time_step_settings_.dt_ = *context.time_step_size;
  // set minimum substep length
  time_step_settings_.min_dt_ =
      time_step_settings_.dt_ / std::pow(2.0, parameter()->max_halve_number());

  // set last substep values (last converged state) as the last time step values --> required, as
  // these are used in the EvaluateAdditionalCMat method (in the case where there is no plastic
  // deformation, these would not be updated correctly otherwise)
  time_step_quantities_.last_substep_plastic_defgrd_inverse_[gp_] =
      time_step_quantities_.last_plastic_defgrd_inverse_[gp_];
  time_step_quantities_.last_substep_plastic_strain_[gp_] =
      time_step_quantities_.last_plastic_strain_[gp_];

  // call preevaluate method of the viscoplastic law
  viscoplastic_law_->pre_evaluate(gp);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::calculate_gamma_delta(
    const Core::LinAlg::Matrix<3, 3>& CeM, Core::LinAlg::Matrix<3, 1>& gamma,
    Core::LinAlg::Matrix<8, 1>& delta)
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
  for (auto& p : potsumel_)  // only isotropic
  {
    p->add_derivatives_principal(dPIe, ddPIIe, prinv, gp_, ele_gid_);
  }

  // compose coefficients
  Mat::calculate_gamma_delta(gamma, delta, prinv, dPIe, ddPIIe);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplast::StateQuantities
Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_state_quantities(
    const Core::LinAlg::Matrix<3, 3>& CM, const Core::LinAlg::Matrix<3, 3>& iFinM,
    const double plastic_strain, Mat::ViscoplastErrorType& err_status, const double dt)
{
  StateQuantities state_quantities{};

  // auxiliaries
  Core::LinAlg::Matrix<1, 1> temp1x1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 3> temp1x3(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> temp3x3;

  // compute right elastic CG tensor
  temp3x3.multiply_nn(1.0, CM, iFinM, 0.0);
  state_quantities.curr_CeM_.multiply_tn(1.0, iFinM, temp3x3, 0.0);
  Core::LinAlg::SymmetricTensor<double, 3, 3> CeV =
      Core::LinAlg::assume_symmetry(Core::LinAlg::make_tensor(state_quantities.curr_CeM_));

  // compose isotropic elastic coefficients (Holzapfel, Nonlinear Solid Mechanics, 2000)
  calculate_gamma_delta(
      state_quantities.curr_CeM_, state_quantities.curr_gamma_, state_quantities.curr_delta_);
  state_quantities.curr_SeM_.clear();
  state_quantities.curr_dSedCe_.clear();
  // compute additional 2nd elastic PK stress and elastic stiffness for the transversely isotropic
  // components (additive split assumed, as for CoupTransverselyIsotropic)
  if (parameter()->bool_transv_isotropy())
  {
    // initialize empty parameter list
    Teuchos::ParameterList param_list{};

    // loop through all transversely isotropic parts, and compute the additional elastic stress
    // and elastic stiffness
    Core::LinAlg::SymmetricTensor<double, 3, 3> SeV{};

    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> dSedCe{};
    for (auto& p : potsumel_transviso_)
    {
      p->add_stress_aniso_principal(CeV, dSedCe, SeV, param_list, gp_, ele_gid_);
    }
    state_quantities.curr_dSedCe_.clear();
    state_quantities.curr_dSedCe_ += Core::LinAlg::make_stress_like_voigt_view(dSedCe);
    state_quantities.curr_SeM_ = Core::LinAlg::make_matrix(Core::LinAlg::get_full(SeV));
  }

  // Ce * Ce tensor
  Core::LinAlg::Matrix<3, 3> CeCeM(Core::LinAlg::Initialization::zero);
  CeCeM.multiply_nn(1.0, state_quantities.curr_CeM_, state_quantities.curr_CeM_, 0.0);

  // compute symmetric part of Mandel stress tensor
  Core::LinAlg::Matrix<3, 3> Me_sym_M(Core::LinAlg::Initialization::zero);
  Me_sym_M.update(state_quantities.curr_gamma_(0), state_quantities.curr_CeM_,
      state_quantities.curr_gamma_(1), CeCeM, 0.0);
  Me_sym_M.update(state_quantities.curr_gamma_(2), const_non_mat_tensors.id3x3_, 1.0);
  if (parameter()->bool_transv_isotropy())
  {
    Core::LinAlg::Matrix<3, 3> addMeM(Core::LinAlg::Initialization::zero);
    temp3x3.multiply_nn(1.0, state_quantities.curr_CeM_, state_quantities.curr_SeM_, 0.0);
    addMeM.update(1.0 / 2.0, temp3x3, 0.0);
    temp3x3.multiply_tn(1.0, state_quantities.curr_SeM_, state_quantities.curr_CeM_, 0.0);
    addMeM.update(1.0 / 2.0, temp3x3, 1.0);
    Me_sym_M.update(1.0, addMeM, 1.0);
  }

  // calculate deviatoric part of the symmetric Mandel stress
  double trMe = 0.0;
  for (int i = 0; i < 3; ++i) trMe += Me_sym_M(i, i);
  state_quantities.curr_Me_dev_sym_M_.update(
      1.0, Me_sym_M, -1.0 / 3.0 * trMe, const_non_mat_tensors.id3x3_, 0.0);

  // for transverse isotropy: we use the Hill 1949 yield condition, adapted for transversely
  // isotropic materials --> get yield function parameters A, B, F
  const double A = parameter()->yield_cond_a();
  const double B = parameter()->yield_cond_b();
  const double F = parameter()->yield_cond_f();

  // determine scalar quantities of invariants / pseudoinvariants needed to compute the equivalent
  // tensile stress
  double Me_dev_sym_contract_Me_dev_sym =
      Core::LinAlg::FourTensorOperations::contract_matrix_matrix(
          state_quantities.curr_Me_dev_sym_M_, state_quantities.curr_Me_dev_sym_M_);
  Core::LinAlg::Matrix<3, 3> Me_dev_sym_squared_M(Core::LinAlg::Initialization::zero);
  Me_dev_sym_squared_M.multiply_nn(
      1.0, state_quantities.curr_Me_dev_sym_M_, state_quantities.curr_Me_dev_sym_M_, 0.0);
  temp1x3.multiply_tn(1.0, m_, Me_dev_sym_squared_M, 0.0);
  temp1x1.multiply_nn(1.0, temp1x3, m_, 0.0);
  double mTMe_dev_sym_squared_m = temp1x1(0);
  temp1x3.multiply_tn(1.0, m_, state_quantities.curr_Me_dev_sym_M_, 0.0);
  temp1x1.multiply_nn(1.0, temp1x3, m_, 0.0);
  double mTMe_dev_sym_m = temp1x1(0);

  // calculate equivalent tensile stress
  if (parameter()->bool_transv_isotropy())
  {
    state_quantities.curr_equiv_stress_ =
        std::sqrt((A + 2 * B) * Me_dev_sym_contract_Me_dev_sym +
                  2 * (F - A - 2 * B) * mTMe_dev_sym_squared_m +
                  (5 * A + B - 2 * F) * std::pow(mTMe_dev_sym_m, 2.0));
  }
  else
  {
    state_quantities.curr_equiv_stress_ = std::sqrt(3.0 / 2.0 * Me_dev_sym_contract_Me_dev_sym);
  }

  // calculate equivalent plastic strain rate using the viscoplastic law
  state_quantities.curr_equiv_plastic_strain_rate_ =
      viscoplastic_law_->evaluate_plastic_strain_rate(state_quantities.curr_equiv_stress_,
          plastic_strain, dt, parameter()->bool_log_substepping(), err_status, update_hist_var_);

  // return if we get an error, all other calculations are useless since substepping is triggered
  if (err_status != Mat::ViscoplastErrorType::NoErrors)
  {
    // return with error
    return StateQuantities{};
  }

  // calculate plastic flow direction
  if (parameter()->bool_transv_isotropy())
  {
    // determine required components for the computation of the plastic flow direction
    Core::LinAlg::Matrix<3, 1> Me_dev_sym_m(Core::LinAlg::Initialization::zero);
    Me_dev_sym_m.multiply_nn(1.0, state_quantities.curr_Me_dev_sym_M_, m_, 0.0);
    Core::LinAlg::Matrix<3, 3> m_dyad_Me_dev_sym_m(Core::LinAlg::Initialization::zero);
    m_dyad_Me_dev_sym_m.multiply_nt(1.0, m_, Me_dev_sym_m, 0.0);
    Core::LinAlg::Matrix<3, 3> Me_dev_sym_A_dyad_A(Core::LinAlg::Initialization::zero);
    Me_dev_sym_A_dyad_A.multiply_nt(1.0, Me_dev_sym_m, m_, 0.0);

    state_quantities.curr_NpM_.clear();
    state_quantities.curr_dpM_.clear();
    if (state_quantities.curr_equiv_stress_ > 0.0)
    {
      // build the plastic flow direction from its tensor parts
      state_quantities.curr_NpM_.update(
          -2.0 / (3.0 * state_quantities.curr_equiv_stress_) * (F - A - 2.0 * B) * mTMe_dev_sym_m,
          const_non_mat_tensors.id3x3_, 0.0);
      state_quantities.curr_NpM_.update(
          1.0 / (1.0 * state_quantities.curr_equiv_stress_) * (A + 2.0 * B),
          state_quantities.curr_Me_dev_sym_M_, 1.0);
      state_quantities.curr_NpM_.update(
          1.0 / (2.0 * state_quantities.curr_equiv_stress_) * 2.0 * (F - A - 2.0 * B),
          m_dyad_Me_dev_sym_m, 1.0);
      state_quantities.curr_NpM_.update(
          1.0 / (2.0 * state_quantities.curr_equiv_stress_) * 2.0 * (F - A - 2.0 * B),
          Me_dev_sym_A_dyad_A, 1.0);
      state_quantities.curr_NpM_.update(1.0 / (2.0 * state_quantities.curr_equiv_stress_) *
                                            (5.0 * A + B - 2.0 * F) * 2.0 * mTMe_dev_sym_m,
          const_mat_tensors_.mm_dev_, 1.0);

      // calculate plastic stretching tensor (deformation rate tensor)
      state_quantities.curr_dpM_.update(
          state_quantities.curr_equiv_plastic_strain_rate_, state_quantities.curr_NpM_, 0.0);
    }

    // calculate plastic velocity gradient tensor
    state_quantities.curr_lpM_.multiply_nn(
        1.0, const_mat_tensors_.id_plus_mm_, state_quantities.curr_dpM_, 0.0);
    state_quantities.curr_lpM_.multiply_nn(
        -1.0, state_quantities.curr_dpM_, const_mat_tensors_.mm_, 1.0);
  }
  else
  {
    state_quantities.curr_NpM_.clear();
    state_quantities.curr_dpM_.clear();
    if (state_quantities.curr_equiv_stress_ > 0.0)
    {
      // build the plastic flow direction from its tensor parts
      state_quantities.curr_NpM_.update(3.0 / (2.0 * state_quantities.curr_equiv_stress_),
          state_quantities.curr_Me_dev_sym_M_, 0.0);

      // calculate plastic stretching tensor (deformation rate tensor)
      state_quantities.curr_dpM_.update(
          state_quantities.curr_equiv_plastic_strain_rate_, state_quantities.curr_NpM_, 0.0);
    }

    // calculate plastic velocity gradient tensor
    state_quantities.curr_lpM_.update(1.0, state_quantities.curr_dpM_, 0.0);
  }

  // calculate plastic update tensor (only required, and computed, for standard substepping)
  if (!parameter()->bool_log_substepping())
  {
    temp3x3.update(-dt, state_quantities.curr_lpM_, 0.0);
    Core::LinAlg::MatrixFunctErrorType exp_err_status{
        Core::LinAlg::MatrixFunctErrorType::no_errors};
    state_quantities.curr_EpM_ = Core::LinAlg::matrix_exp(temp3x3, exp_err_status);
    FOUR_C_ASSERT_ALWAYS(exp_err_status == Core::LinAlg::MatrixFunctErrorType::no_errors,
        "Matrix exponential evaluation failed!");
  }

  return state_quantities;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplast::StateQuantityDerivatives
Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_state_quantity_derivatives(
    const Core::LinAlg::Matrix<3, 3>& CM, const Core::LinAlg::Matrix<3, 3>& iFinM,
    const double plastic_strain, Mat::ViscoplastErrorType& err_status, const double dt,
    const bool eval_state)
{
  StateQuantityDerivatives state_quantity_derivatives{};

  // auxiliaries
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 6> temp6x6(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<6, 9> temp6x9(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<9, 6> temp9x6(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<9, 9> temp9x9(Core::LinAlg::Initialization::zero);

  // check whether to reevaluate the state or to keep the available state quantity values
  StateQuantities relevant_state_quantities = state_quantities_;
  if (eval_state)
  {
    relevant_state_quantities =
        evaluate_state_quantities(CM, iFinM, plastic_strain, err_status, dt);
  }

  // get the state quantities
  const Core::LinAlg::Matrix<3, 3> CeM = relevant_state_quantities.curr_CeM_;
  const Core::LinAlg::Matrix<3, 1> gamma = relevant_state_quantities.curr_gamma_;
  const Core::LinAlg::Matrix<8, 1> delta = relevant_state_quantities.curr_delta_;
  const Core::LinAlg::Matrix<3, 3> SeM = relevant_state_quantities.curr_SeM_;
  const Core::LinAlg::Matrix<6, 6> dSedCe = relevant_state_quantities.curr_dSedCe_;
  const Core::LinAlg::Matrix<3, 3> Me_dev_sym_M = relevant_state_quantities.curr_Me_dev_sym_M_;
  const double equiv_stress = relevant_state_quantities.curr_equiv_stress_;
  const double equiv_plastic_strain_rate =
      relevant_state_quantities.curr_equiv_plastic_strain_rate_;
  const Core::LinAlg::Matrix<3, 3> NpM = relevant_state_quantities.curr_NpM_;
  const Core::LinAlg::Matrix<3, 3> dpM = relevant_state_quantities.curr_dpM_;
  const Core::LinAlg::Matrix<3, 3> lpM = relevant_state_quantities.curr_lpM_;
  const Core::LinAlg::Matrix<3, 3> EpM = relevant_state_quantities.curr_EpM_;


  // compute the relevant derivatives of the elastic right Cauchy-Green deformation tensor
  Mat::elast_hyper_get_derivs_of_elastic_right_cg_tensor(Core::LinAlg::make_tensor(iFinM),
      Core::LinAlg::assume_symmetry(Core::LinAlg::make_tensor(CM)),
      state_quantity_derivatives.curr_dCedC_, state_quantity_derivatives.curr_dCediFin_);
  // save these also as four tensors
  Core::LinAlg::FourTensor<3> dCediFin_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x9_voigt_matrix(
      dCediFin_FourTensor, state_quantity_derivatives.curr_dCediFin_);
  Core::LinAlg::FourTensor<3> dCedC_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix(
      dCedC_FourTensor, state_quantity_derivatives.curr_dCedC_);

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
  iFinTM.multiply_tn(1.0, iFinM, const_non_mat_tensors.id3x3_, 0.0);

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

  // compute the relevant derivatives of the symmetric part of the Mandel stress

  // \f$ \frac{\partial \boldsymbol{M}^{\text{e}}_{\text{sym}} }{\partial
  // \boldsymbol{F}^{\text{in}^{-1}}_{}} \f$ (Voigt stress-form)
  Core::LinAlg::Matrix<6, 9> dMe_sym_diFin(Core::LinAlg::Initialization::zero);
  dMe_sym_diFin.clear();
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dMe_sym_diFin, iFinTCM, const_non_mat_tensors.id3x3_, gamma(0));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dMe_sym_diFin, iFinTCM, CeM, gamma(1));
  Core::LinAlg::FourTensorOperations::add_right_non_symmetric_holzapfel_product(
      dMe_sym_diFin, CeiFinTCM, const_non_mat_tensors.id3x3_, gamma(1));
  dMe_sym_diFin.multiply_nt(delta(0), CeV, CiFinV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(1), CeV, CiFinCeV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(1), CeCeV, CiFinV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(2), CeV, CiFiniCeV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(2), const_non_mat_tensors.id6x1_, CiFinV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(3), CeCeV, CiFinCeV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(4), CeCeV, CiFiniCeV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(4), const_non_mat_tensors.id6x1_, CiFinCeV, 1.0);
  dMe_sym_diFin.multiply_nt(delta(5), const_non_mat_tensors.id6x1_, CiFiniCeV, 1.0);

  // \f$ \frac{\partial \boldsymbol{M}^{\text{e}}_{\text{sym}} }{\partial
  // \boldsymbol{C}^{}_{}} \f$ (Voigt stress-stress form)
  Core::LinAlg::Matrix<6, 6> dMe_sym_dC(Core::LinAlg::Initialization::zero);
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
      dMe_sym_dC, gamma(0), iFinTM, iFinTM, 0.0);
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
      dMe_sym_dC, gamma(1), iFinTM, CeiFinTM, 1.0);
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(
      dMe_sym_dC, gamma(1), CeiFinTM, iFinTM, 1.0);
  dMe_sym_dC.multiply_nt(delta(0) / 2.0, CeV, iCinV, 1.0);
  dMe_sym_dC.multiply_nt(delta(1) / 2.0, CeV, iCinCiCinV, 1.0);
  dMe_sym_dC.multiply_nt(delta(1) / 2.0, CeCeV, iCinV, 1.0);
  dMe_sym_dC.multiply_nt(delta(2) / 2.0, CeV, iCV, 1.0);
  dMe_sym_dC.multiply_nt(delta(2) / 2.0, const_non_mat_tensors.id6x1_, iCinV, 1.0);
  dMe_sym_dC.multiply_nt(delta(3) / 2.0, CeCeV, iCinCiCinV, 1.0);
  dMe_sym_dC.multiply_nt(delta(4) / 2.0, CeCeV, iCV, 1.0);
  dMe_sym_dC.multiply_nt(delta(4) / 2.0, const_non_mat_tensors.id6x1_, iCinCiCinV, 1.0);
  dMe_sym_dC.multiply_nt(delta(5) / 2.0, const_non_mat_tensors.id6x1_, iCV, 1.0);

  // compute derivative of the additional transversely isotropic stress (w.r.t. right elastic
  // Cauchy-Green deformation tensor) in stress-strain notation
  temp6x6.update(1.0, dSedCe, 0.0);
  Core::LinAlg::Matrix<6, 6> dSedCe_stress_strain =
      Core::LinAlg::Voigt::modify_voigt_representation(temp6x6, 1.0, 2.0);

  // \f$ \frac{\partial \boldsymbol{S}^{\text{e}}_{\text{trn}} }{\partial
  // \boldsymbol{F}^{\text{in}^{-1}}_{}} \f$ (Voigt stress-form)
  Core::LinAlg::Matrix<6, 9> dSediFin(Core::LinAlg::Initialization::zero);
  dSediFin.multiply_nn(1.0, dSedCe_stress_strain, state_quantity_derivatives.curr_dCediFin_, 0.0);
  Core::LinAlg::FourTensor<3> dSediFin_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x9_voigt_matrix(dSediFin_FourTensor, dSediFin);

  // \f$ \frac{\partial \boldsymbol{S}^{\text{e}}_{\text{trn}} }{\partial
  // \boldsymbol{C}^{}_{}} \f$ (Voigt stress-stress form)
  Core::LinAlg::Matrix<6, 6> dSedC(Core::LinAlg::Initialization::zero);
  dSedC.multiply_nn(1.0, dSedCe_stress_strain, state_quantity_derivatives.curr_dCedC_, 0.0);
  Core::LinAlg::FourTensor<3> dSedC_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix(dSedC_FourTensor, dSedC);

  // compute additional components of the elastic transversely isotropic components for the
  // derivatives of the symmetric Mandel stress
  if (parameter()->bool_transv_isotropy())
  {
    Core::LinAlg::FourTensor<3> CedSediFin_FourTensor(true);
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        CedSediFin_FourTensor, CeM, dSediFin_FourTensor, true);
    Core::LinAlg::FourTensor<3> CedSediFin_T12_FourTensor(true);
    CedSediFin_T12_FourTensor.transpose_12(CedSediFin_FourTensor);
    Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(temp6x9, CedSediFin_FourTensor);
    dMe_sym_diFin.update(1.0 / 2.0, temp6x9, 1.0);
    Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(
        temp6x9, CedSediFin_T12_FourTensor);
    dMe_sym_diFin.update(1.0 / 2.0, temp6x9, 1.0);
    Core::LinAlg::FourTensor<3> SedCediFin_FourTensor(true);
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        SedCediFin_FourTensor, SeM, dCediFin_FourTensor, true);
    Core::LinAlg::FourTensor<3> SedCediFin_T12_FourTensor(true);
    SedCediFin_T12_FourTensor.transpose_12(SedCediFin_FourTensor);
    Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(temp6x9, SedCediFin_FourTensor);
    dMe_sym_diFin.update(1.0 / 2.0, temp6x9, 1.0);
    Core::LinAlg::Voigt::setup_6x9_voigt_matrix_from_four_tensor(
        temp6x9, SedCediFin_T12_FourTensor);
    dMe_sym_diFin.update(1.0 / 2.0, temp6x9, 1.0);

    Core::LinAlg::FourTensor<3> CedSedC_FourTensor(true);
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        CedSedC_FourTensor, CeM, dSedC_FourTensor, true);
    Core::LinAlg::FourTensor<3> CedSedC_T12_FourTensor(true);
    CedSedC_T12_FourTensor.transpose_12(CedSedC_FourTensor);
    Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, CedSedC_FourTensor);
    dMe_sym_dC.update(1.0 / 2.0, temp6x6, 1.0);
    Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, CedSedC_T12_FourTensor);
    dMe_sym_dC.update(1.0 / 2.0, temp6x6, 1.0);
    Core::LinAlg::FourTensor<3> SedCedC_FourTensor(true);
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        SedCedC_FourTensor, SeM, dCedC_FourTensor, true);
    Core::LinAlg::FourTensor<3> SedCedC_T12_FourTensor(true);
    SedCedC_T12_FourTensor.transpose_12(SedCedC_FourTensor);
    Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, SedCedC_FourTensor);
    dMe_sym_dC.update(1.0 / 2.0, temp6x6, 1.0);
    Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, SedCedC_T12_FourTensor);
    dMe_sym_dC.update(1.0 / 2.0, temp6x6, 1.0);
  }

  // compute derivatives of the deviatoric, symmetric part of the Mandel stress

  // \f$ \frac{\partial \boldsymbol{M}^{\text{e}}_{\text{dev, sym}} }{\partial
  // \boldsymbol{F}^{\text{in}^{-1}}_{}} \f$ (Voigt stress-form)
  state_quantity_derivatives.curr_dMe_dev_sym_diFin_.multiply_nn(
      1.0, const_non_mat_tensors.dev_op_, dMe_sym_diFin, 0.0);
  // \f$ \frac{\partial \boldsymbol{M}^{\text{e}}_{\text{dev,sym}} }{\partial
  // \boldsymbol{C}^{}_{}} \f$ (Voigt stress-stress form)
  state_quantity_derivatives.curr_dMe_dev_sym_dC_.multiply_nn(
      1.0, const_non_mat_tensors.dev_op_, dMe_sym_dC, 0.0);

  // plastic flow direction in Voigt strain notation
  Core::LinAlg::Matrix<6, 1> NpV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Strains::matrix_to_vector(NpM, NpV);

  // compute derivatives of the equivalent stress

  // \f$ \frac{\partial \overline{\sigma} }{\partial
  // \boldsymbol{F}^{\text{in}^{-1}}_{}} \f$ (Voigt stress-form)
  state_quantity_derivatives.curr_dequiv_stress_diFin_.multiply_tn(
      1.0, NpV, state_quantity_derivatives.curr_dMe_dev_sym_diFin_, 0.0);
  // \f$ \frac{\partial \overline{\sigma} }{\partial
  // \boldsymbol{C}^{}} \f$ (Voigt stress-form)
  state_quantity_derivatives.curr_dequiv_stress_dC_.multiply_tn(
      1.0, NpV, state_quantity_derivatives.curr_dMe_dev_sym_dC_, 0.0);

  // recompute flow direction in stress form
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(NpM, NpV);

  // we use the Hill 1949 yield condition, adapted for transversely isotropic materials -> get
  // yield condition parameters A, B, and F
  const double A = parameter()->yield_cond_a();
  const double B = parameter()->yield_cond_b();
  const double F = parameter()->yield_cond_f();

  // compute required derivative of the plastic flow direction (w.r.t. dev., sym. part of the
  // Mandel stress)
  // \f$ \frac{\partial \boldsymbol{N}^{\text{p}}_{} }{\partial
  // \partial \boldsymbol{M}^{\text{e}}_{\text{dev,sym}}} \f$ (Voigt stress-stress form)
  Core::LinAlg::Matrix<6, 6> dNpdMe_sym_dev(Core::LinAlg::Initialization::zero);
  if (parameter()->bool_transv_isotropy())
  {
    dNpdMe_sym_dev.multiply_nt(-1.0 / equiv_stress, NpV, NpV, 0.0);
    dNpdMe_sym_dev.update(-1.0 / 2.0 * 1.0 / equiv_stress * 4.0 / 3.0 * (F - A - 2.0 * B),
        const_mat_tensors_.id_dyad_mm_, 1.0);
    dNpdMe_sym_dev.update(
        1.0 / 1.0 * 1.0 / equiv_stress * (A + 2 * B), const_non_mat_tensors.id4_6x6_, 1.0);
    Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(dNpdMe_sym_dev,
        1.0 / equiv_stress * (F - A - 2 * B), const_mat_tensors_.mm_, const_non_mat_tensors.id3x3_,
        1.0);
    Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(dNpdMe_sym_dev,
        1.0 / equiv_stress * (F - A - 2 * B), const_non_mat_tensors.id3x3_, const_mat_tensors_.mm_,
        1.0);
    dNpdMe_sym_dev.update(
        1.0 / equiv_stress * (5 * A + B - 2 * F), const_mat_tensors_.mm_dev_dyad_mm_, 1.0);
  }
  else
  {
    dNpdMe_sym_dev.multiply_nt(-1.0 / equiv_stress, NpV, NpV, 0.0);
    dNpdMe_sym_dev.update(1.0 / equiv_stress * 3.0 / 2.0, const_non_mat_tensors.id4_6x6_, 1.0);
  }
  // convert derivative to Voigt stress-strain form
  temp6x6 = dNpdMe_sym_dev;
  dNpdMe_sym_dev = Core::LinAlg::Voigt::modify_voigt_representation(temp6x6, 1.0, 2.0);

  // compute the relevant derivatives of the plastic strain rate
  Core::LinAlg::Matrix<2, 1> evoEqFunctionDers =
      viscoplastic_law_->evaluate_derivatives_of_plastic_strain_rate(
          equiv_stress, plastic_strain, dt, parameter()->bool_log_substepping(), err_status);

  // return if we get an error, all other calculations are useless since substepping is triggered
  if (err_status != Mat::ViscoplastErrorType::NoErrors)
  {
    // return with error
    return StateQuantityDerivatives{};
  }

  state_quantity_derivatives.curr_dpsr_dequiv_stress_ = evoEqFunctionDers(0);
  state_quantity_derivatives.curr_dpsr_depsp_ = evoEqFunctionDers(1);

  // compute derivatives of the plastic stretching tensor...
  Core::LinAlg::Matrix<6, 6> Np_dyad_Np_V(
      Core::LinAlg::Initialization::zero);  // in stress-strain form
  temp6x6.multiply_nt(1.0, NpV, NpV, 0.0);
  Np_dyad_Np_V = Core::LinAlg::Voigt::modify_voigt_representation(temp6x6, 1.0, 2.0);
  temp6x6.update(state_quantity_derivatives.curr_dpsr_dequiv_stress_, Np_dyad_Np_V,
      equiv_plastic_strain_rate, dNpdMe_sym_dev, 0.0);

  // ... w.r.t. invese inelastic defgrad
  state_quantity_derivatives.curr_ddpdiFin_.multiply_nn(
      1.0, temp6x6, state_quantity_derivatives.curr_dMe_dev_sym_diFin_, 0.0);
  // ... w.r.t. plastic strain
  state_quantity_derivatives.curr_ddpdepsp_.update(
      state_quantity_derivatives.curr_dpsr_depsp_, NpV, 0.0);
  // ... w.r.t. right CG
  state_quantity_derivatives.curr_ddpdC_.multiply_nn(
      1.0, temp6x6, state_quantity_derivatives.curr_dMe_dev_sym_dC_, 0.0);

  // compute derivatives of the plastic velocity gradient ...

  // ... w.r.t. invese inelastic defgrad
  Core::LinAlg::FourTensor<3> ddpdiFin_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x9_voigt_matrix(
      ddpdiFin_FourTensor, state_quantity_derivatives.curr_ddpdiFin_);
  Core::LinAlg::FourTensor<3> id_plus_mm_ddpdiFin_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      id_plus_mm_ddpdiFin_FourTensor, const_mat_tensors_.id_plus_mm_, ddpdiFin_FourTensor, true);
  Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(
      temp9x9, id_plus_mm_ddpdiFin_FourTensor);
  state_quantity_derivatives.curr_dlpdiFin_.update(1.0, temp9x9, 0.0);
  Core::LinAlg::FourTensor<3> mm_ddpdiFin_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      mm_ddpdiFin_FourTensor, const_mat_tensors_.mm_, ddpdiFin_FourTensor, true);
  Core::LinAlg::FourTensor<3> mm_ddpdiFin_T12_FourTensor(true);
  mm_ddpdiFin_T12_FourTensor.transpose_12(mm_ddpdiFin_FourTensor);
  Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(temp9x9, mm_ddpdiFin_T12_FourTensor);
  state_quantity_derivatives.curr_dlpdiFin_.update(-1.0, temp9x9, 1.0);

  // ... w.r.t. right CG
  Core::LinAlg::FourTensor<3> ddpdC_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_6x6_voigt_matrix(
      ddpdC_FourTensor, state_quantity_derivatives.curr_ddpdC_);
  Core::LinAlg::FourTensor<3> id_plus_mm_ddpdC_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      id_plus_mm_ddpdC_FourTensor, const_mat_tensors_.id_plus_mm_, ddpdC_FourTensor, true);
  Core::LinAlg::Voigt::setup_9x6_voigt_matrix_from_four_tensor(
      temp9x6, id_plus_mm_ddpdC_FourTensor);
  state_quantity_derivatives.curr_dlpdC_.update(1.0, temp9x6, 0.0);
  Core::LinAlg::FourTensor<3> mm_ddpdC_FourTensor(true);
  Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
      mm_ddpdC_FourTensor, const_mat_tensors_.mm_, ddpdC_FourTensor, true);
  Core::LinAlg::FourTensor<3> mm_ddpdC_T12_FourTensor(true);
  mm_ddpdC_T12_FourTensor.transpose_12(mm_ddpdC_FourTensor);
  Core::LinAlg::Voigt::setup_9x6_voigt_matrix_from_four_tensor(temp9x6, mm_ddpdC_T12_FourTensor);
  state_quantity_derivatives.curr_dlpdC_.update(-1.0, temp9x6, 1.0);

  // ... w.r.t. plastic strain
  Core::LinAlg::Matrix<3, 3> ddpdepsp_M(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::vector_to_matrix(
      state_quantity_derivatives.curr_ddpdepsp_, ddpdepsp_M);
  Core::LinAlg::Matrix<3, 3> dlpdepsp_M(Core::LinAlg::Initialization::zero);
  dlpdepsp_M.multiply_nn(1.0, const_mat_tensors_.id_plus_mm_, ddpdepsp_M, 0.0);
  dlpdepsp_M.multiply_nn(-1.0, ddpdepsp_M, const_mat_tensors_.mm_, 1.0);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(dlpdepsp_M, state_quantity_derivatives.curr_dlpdepsp_);


  // compute derivatives of the update tensor (only required for standard substepping)
  if (!parameter()->bool_log_substepping())
  {
    // compute argument
    Core::LinAlg::Matrix<3, 3> min_dt_lpM(Core::LinAlg::Initialization::zero);
    min_dt_lpM.update(-1.0 * dt, lpM, 0.0);

    // compute derivative of exponential ...

    // ... w.r.t. its argument
    Core::LinAlg::MatrixFunctErrorType exp_deriv_err_status{
        Core::LinAlg::MatrixFunctErrorType::no_errors};
    Core::LinAlg::Matrix<9, 9> expderivV =
        Core::LinAlg::matrix_3x3_exp_1st_deriv(min_dt_lpM, exp_deriv_err_status);
    FOUR_C_ASSERT_ALWAYS(exp_deriv_err_status == Core::LinAlg::MatrixFunctErrorType::no_errors,
        "Matrix exponential derivative evaluation failed!");


    // ... w.r.t. inverse inelastic defgrad
    state_quantity_derivatives.curr_dEpdiFin_.multiply_nn(
        -dt, expderivV, state_quantity_derivatives.curr_dlpdiFin_, 0.0);

    // ... w.r.t. right CG
    state_quantity_derivatives.curr_dEpdC_.multiply_nn(
        -dt, expderivV, state_quantity_derivatives.curr_dlpdC_, 0.0);

    // ... w.r.t. plastic strain
    state_quantity_derivatives.curr_dEpdepsp_.multiply_nn(
        -dt, expderivV, state_quantity_derivatives.curr_dlpdepsp_, 0.0);
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
  // reduced deformation gradient FredM, taking into account all the already computed inelastic
  // factors
  //    \f$ \boldsymbol{F_{\text{red}}} = \boldsymbol{F} \boldsymbol{F_{\text{in,other}}^{-1}} \f$
  //      with \f$\boldsymbol{F}_{\text{in,other}}^{-1} = \boldsymbol{F}_{\text{in},1}^{-1}
  //      \boldsymbol{F}_{\text{in},2}^{-1} \dots \f$ up to the current inelastic factor
  Core::LinAlg::Matrix<3, 3> FredM(Core::LinAlg::Initialization::zero);
  FredM.multiply_nn(1.0, *defgrad, iFin_other, 0.0);

  // reduced right Cauchy-Green deformation tensor
  Core::LinAlg::Matrix<3, 3> CredM(Core::LinAlg::Initialization::zero);
  CredM.multiply_tn(1.0, FredM, FredM, 0.0);

  // auxiliaries
  Core::LinAlg::FourTensor<3> tempFourTensor(true);

  // declare error status (no errors)
  Mat::ViscoplastErrorType err_status = Mat::ViscoplastErrorType::NoErrors;

  // calculate linearization term only if we have plastic strain
  if (std::abs(time_step_quantities_.current_plastic_strain_[gp_] -
               time_step_quantities_.last_plastic_strain_[gp_]) > 0.0)
  {
    // ----- analytical linearization ----- //
    // if error encountered: perform perturbation-based linearization

    // calculate Jacobian
    Core::LinAlg::Matrix<10, 1> current_sol =
        wrap_unknowns(time_step_quantities_.current_plastic_defgrd_inverse_[gp_],
            time_step_quantities_.current_plastic_strain_[gp_]);
    Core::LinAlg::Matrix<10, 10> jacMat(Core::LinAlg::Initialization::zero);
    viscoplastic_law_->pre_evaluate(gp_);  // set last_substep <- last_
    jacMat = calculate_jacobian(CredM, current_sol,
        time_step_quantities_.last_plastic_defgrd_inverse_[gp_],
        time_step_quantities_.last_plastic_strain_[gp_], time_step_settings_.dt_, err_status);
    if (err_status != Mat::ViscoplastErrorType::NoErrors)
    {
      evaluate_additional_cmat_perturb_based(FredM, cmatadd, iFin_other, dSdiFinj);
      return;
    }

    // if we get singular Jacobian: throw exception -> go to FD-based linearization
    if (abs(jacMat.determinant()) < 1.0e-10)
    {
      err_status = ViscoplastErrorType::SingularJacobian;
      evaluate_additional_cmat_perturb_based(FredM, cmatadd, iFin_other, dSdiFinj);
      return;
    }

    // declare right-hand side (RHS) terms of the linear system of equations related to the
    // analytical linearization
    Core::LinAlg::Matrix<9, 6> rhs_iFin_V(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<1, 6> rhs_epsp_V(Core::LinAlg::Initialization::zero);

    if (!(parameter()->bool_log_substepping()))
    // standard substepping
    {
      // calculate RHS of the equation for the plastic deformation gradient
      Core::LinAlg::FourTensor<3> dEpdC_FourTensor(true);
      Core::LinAlg::Voigt::setup_four_tensor_from_9x6_voigt_matrix(
          dEpdC_FourTensor, state_quantity_derivatives_.curr_dEpdC_);
      Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(tempFourTensor,
          time_step_quantities_.last_plastic_defgrd_inverse_[gp_], dEpdC_FourTensor);
      Core::LinAlg::Voigt::setup_9x6_voigt_matrix_from_four_tensor(rhs_iFin_V, tempFourTensor);

      // calculate RHS of the equation for the plastic strain
      rhs_epsp_V.update(
          time_step_settings_.dt_ * state_quantity_derivatives_.curr_dpsr_dequiv_stress_,
          state_quantity_derivatives_.curr_dequiv_stress_dC_, 0.0);
    }
    else
    // logarithmic substepping
    {
      // calculate RHS of the equation for the plastic deformation gradient
      rhs_iFin_V.update(-time_step_settings_.dt_, state_quantity_derivatives_.curr_dlpdC_, 0.0);

      // calculate RHS of the equation for the plastic strain
      rhs_epsp_V.update(
          time_step_settings_.dt_ * state_quantity_derivatives_.curr_dpsr_dequiv_stress_,
          state_quantity_derivatives_.curr_dequiv_stress_dC_, 0.0);
    }

    // assemble the RHS from its components
    Core::LinAlg::Matrix<10, 6> RHS = assemble_rhs_additional_cmat(rhs_iFin_V, rhs_epsp_V);

    // solve the linear system of equations
    Core::LinAlg::Matrix<10, 6> SOL(Core::LinAlg::Initialization::zero);
    Core::LinAlg::FixedSizeSerialDenseSolver<10, 10, 6> solver;
    solver.set_matrix(jacMat);     // set A = jacM
    solver.set_vectors(SOL, RHS);  // set X=SOL, B=RHS
    solver.factor_with_equilibration(true);
    int err = solver.solve();  // X = A^-1 B
    int err2 = solver.factor();
    if ((err != 0) || (err2 != 0))
    {
      err_status = ViscoplastErrorType::FailedSolAnalytLinearization;
      evaluate_additional_cmat_perturb_based(FredM, cmatadd, iFin_other, dSdiFinj);
      return;
    }

    // disassemble the solution vector
    Core::LinAlg::Matrix<9, 6> diFinjdCV = extract_derivative_of_inv_inelastic_defgrad(SOL);

    // compute additional term to stiffness matrix additional_cmat
    cmatadd.multiply_nn(2.0, dSdiFinj, diFinjdCV, 1.0);
  }
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_inverse_inelastic_def_grad(
    const Core::LinAlg::Matrix<3, 3>* defgrad, const Core::LinAlg::Matrix<3, 3>& iFin_other,
    Core::LinAlg::Matrix<3, 3>& iFinM)
{
  // reduced deformation gradient FredM, taking into account all the already computed inelastic
  // factors
  //    \f$ \boldsymbol{F_{\text{red}}} = \boldsymbol{F} \boldsymbol{F_{\text{in,other}}^{-1}} \f$
  //      with \f$\boldsymbol{F}_{\text{in,other}}^{-1} = \boldsymbol{F}_{\text{in},1}^{-1}
  //      \boldsymbol{F}_{\text{in},2}^{-1} \dots \f$ up to the current inelastic factor
  Core::LinAlg::Matrix<3, 3> FredM(Core::LinAlg::Initialization::zero);
  FredM.multiply_nn(1.0, *defgrad, iFin_other, 0.0);
  Core::LinAlg::Matrix<3, 3> CredM(Core::LinAlg::Initialization::zero);
  CredM.multiply_tn(1.0, FredM, FredM, 0.0);

  // check whether we have already evaluated the inverse inelastic deformation gradient for the
  // given reduced deformation gradient
  Core::LinAlg::Matrix<3, 3> diff_defgrad{Core::LinAlg::Initialization::zero};
  diff_defgrad.update(1.0, FredM, -1.0, time_step_quantities_.current_defgrad_[gp_], 0.0);
  if (diff_defgrad.norm2() == 0.0)
  {
    // return the already computed current_ value
    iFinM = time_step_quantities_.current_plastic_defgrd_inverse_[gp_];
    return;
  }


  // set predictor: assume purely elastic behavior in this time step
  Core::LinAlg::Matrix<3, 3> iFinM_pred(Core::LinAlg::Initialization::zero);
  iFinM_pred.update(1.0, time_step_quantities_.last_plastic_defgrd_inverse_[gp_], 0.0);
  double plastic_strain_pred = time_step_quantities_.last_plastic_strain_[gp_];

  // declare error status of evaluation (no errors)
  Mat::ViscoplastErrorType err_status = Mat::ViscoplastErrorType::NoErrors;

  // check whether the predictor is the solution (no plastic strain during this time step)
  bool pred_is_sol = check_predictor(CredM, iFinM_pred, plastic_strain_pred, err_status);
  if ((err_status == Mat::ViscoplastErrorType::NoErrors) && (pred_is_sol))
  {
    // update inverse inelastic defgrad
    iFinM = iFinM_pred;

    // update history variables of material
    if (update_hist_var_)
    {
      time_step_quantities_.current_plastic_defgrd_inverse_[gp_] = iFinM;
      time_step_quantities_.current_plastic_strain_[gp_] = plastic_strain_pred;
      time_step_quantities_.current_rightCG_[gp_] = CredM;
      time_step_quantities_.current_defgrad_[gp_] = FredM;
    }
  }
  else  // predictor does not suffice
  {
    // perform time integration via the Local Newton-Raphson Loop (LNL), using the elastic
    // predictor
    Core::LinAlg::Matrix<10, 1> x = wrap_unknowns(iFinM_pred, plastic_strain_pred);
    Core::LinAlg::Matrix<10, 1> sol = local_newton_loop(FredM, x, err_status);

    // throw error if the Local Newton Loop cannot be evaluated with the given substepping
    // settings
    if (err_status != Mat::ViscoplastErrorType::NoErrors)
      FOUR_C_THROW("{}", Mat::to_string(err_status));

    // extract the inverse inelastic defgrad from the LNL solution
    iFinM = extract_inverse_inelastic_defgrad(sol);

    // update history variables of material
    if (update_hist_var_)
    {
      time_step_quantities_.current_plastic_defgrd_inverse_[gp_] = iFinM;
      time_step_quantities_.current_plastic_strain_[gp_] = sol(9);
      time_step_quantities_.current_rightCG_[gp_] = CredM;
      time_step_quantities_.current_defgrad_[gp_] = FredM;
    }
  }
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::update()
{
  // update history variables for the next time step
  time_step_quantities_.last_rightCG_ = time_step_quantities_.current_rightCG_;
  time_step_quantities_.last_plastic_defgrd_inverse_ =
      time_step_quantities_.current_plastic_defgrd_inverse_;
  time_step_quantities_.last_plastic_strain_ = time_step_quantities_.current_plastic_strain_;

  // call update method of the viscoplastic law
  viscoplastic_law_->update();
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplast::setup(const int numgp,
    const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // auxiliaries
  std::vector<Core::LinAlg::Tensor<double, 3>> temp_vec;
  Core::LinAlg::Matrix<6, 1> temp_6x1(Core::LinAlg::Initialization::zero);

  // default values of the inverse plastic deformation gradient for ALL Gauss Points
  time_step_quantities_.last_plastic_defgrd_inverse_.resize(
      numgp, time_step_quantities_.last_plastic_defgrd_inverse_[0]);
  time_step_quantities_.current_plastic_defgrd_inverse_.resize(numgp,
      time_step_quantities_.last_plastic_defgrd_inverse_[0]);  // value irrelevant at this point
  time_step_quantities_.last_substep_plastic_defgrd_inverse_.resize(
      numgp, time_step_quantities_.last_substep_plastic_defgrd_inverse_[0]);

  // default values of the plastic strain for ALL Gauss Points
  time_step_quantities_.last_plastic_strain_.resize(
      numgp, time_step_quantities_.last_plastic_strain_[0]);
  time_step_quantities_.current_plastic_strain_.resize(
      numgp, time_step_quantities_.last_plastic_strain_[0]);  // value irrelevant at this point
  time_step_quantities_.last_substep_plastic_strain_.resize(
      numgp, time_step_quantities_.last_substep_plastic_strain_[0]);

  // default values of the right CG deformation tensor for ALL Gauss Points
  time_step_quantities_.last_rightCG_.resize(numgp, time_step_quantities_.last_rightCG_[0]);
  time_step_quantities_.current_rightCG_.resize(
      numgp, time_step_quantities_.last_rightCG_[0]);  // value irrelevant at this point

  // default values of the deformation gradient
  time_step_quantities_.current_defgrad_.resize(numgp, time_step_quantities_.current_defgrad_[0]);

  // call corresponding method of the viscoplastic law
  viscoplastic_law_->setup(numgp, fibers, coord_system);

  // read fiber and structural tensor in the case of transverse isotropy
  if (parameter()->bool_transv_isotropy())
  {
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

    // pack last_ values inside time_step_quantities_
    add_to_pack(data, time_step_quantities_.last_rightCG_);
    add_to_pack(data, time_step_quantities_.last_plastic_defgrd_inverse_);
    add_to_pack(data, time_step_quantities_.last_plastic_strain_);
    add_to_pack(data, time_step_quantities_.last_substep_plastic_defgrd_inverse_);
    add_to_pack(data, time_step_quantities_.last_substep_plastic_strain_);
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

    // unpack last_ values inside time_step_quantities_
    extract_from_pack(buffer, time_step_quantities_.last_rightCG_);
    extract_from_pack(buffer, time_step_quantities_.last_plastic_defgrd_inverse_);
    extract_from_pack(buffer, time_step_quantities_.last_plastic_strain_);
    extract_from_pack(buffer, time_step_quantities_.last_substep_plastic_defgrd_inverse_);
    extract_from_pack(buffer, time_step_quantities_.last_substep_plastic_strain_);
  }

  // fill current_ values with the last_ values
  time_step_quantities_.current_rightCG_.resize(time_step_quantities_.last_rightCG_.size(),
      time_step_quantities_.last_rightCG_[0]);  // value irrelevant
  time_step_quantities_.current_plastic_defgrd_inverse_.resize(
      time_step_quantities_.last_plastic_defgrd_inverse_.size(),
      time_step_quantities_.last_plastic_defgrd_inverse_[0]);  // value irrelevant
  time_step_quantities_.current_plastic_strain_.resize(
      time_step_quantities_.last_plastic_strain_.size(),
      time_step_quantities_.last_plastic_strain_[0]);  // value irrelevant

  // set evaluated deformation gradient to 0, to make sure that the inverse inelastic deformation
  // gradient is evaluated fully after the restart
  time_step_quantities_.current_defgrad_.resize(
      time_step_quantities_.last_substep_plastic_defgrd_inverse_.size(),
      Core::LinAlg::Matrix<3, 3>{Core::LinAlg::Initialization::zero});


  // now that the fiber direction is available, we set the material-dependent constant tensors
  // with it
  const_mat_tensors_.set_material_const_tensors(m_);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<10, 1>
Mat::InelasticDefgradTransvIsotropElastViscoplast::calculate_local_newton_loop_residual(
    const Core::LinAlg::Matrix<3, 3>& CM, const Core::LinAlg::Matrix<10, 1>& x,
    const Core::LinAlg::Matrix<3, 3>& last_iFinM, const double last_plastic_strain, const double dt,
    Mat::ViscoplastErrorType& err_status)
{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);

  // extract inverse inelastic defgrad and plastic strain from input vector
  Core::LinAlg::Matrix<3, 3> iFinM = extract_inverse_inelastic_defgrad(x);
  double plastic_strain = x(9);

  // evaluate state variables
  state_quantities_ = evaluate_state_quantities(CM, iFinM, plastic_strain, err_status, dt);

  // declare residuals of the LNL
  Core::LinAlg::Matrix<3, 3> resFM(Core::LinAlg::Initialization::zero);
  double resepsp = 0.0;

  // compute residuals (standard substepping)
  if (!(parameter()->bool_log_substepping()))
  {
    // calculate residual of the equation for inelastic defgrad
    temp3x3.multiply_nn(1.0, last_iFinM, state_quantities_.curr_EpM_, 0.0);
    resFM.update(1.0, iFinM, -1.0, temp3x3, 0.0);

    // calculate residual of the equation for plastic strain
    resepsp = plastic_strain - last_plastic_strain -
              dt * state_quantities_.curr_equiv_plastic_strain_rate_;
  }
  else
  // compute residuals (logarithmic substepping)
  {
    // calculate the tensor logarithm involved in the residual
    Core::LinAlg::Matrix<3, 3> last_FinM(Core::LinAlg::Initialization::zero);
    last_FinM.invert(last_iFinM);
    Core::LinAlg::Matrix<3, 3> T(Core::LinAlg::Initialization::zero);
    T.multiply_nn(1.0, last_FinM, iFinM, 0.0);
    Core::LinAlg::MatrixFunctErrorType log_err_status{
        Core::LinAlg::MatrixFunctErrorType::no_errors};
    Core::LinAlg::Matrix<3, 3> logT = Core::LinAlg::matrix_log(T, log_err_status);
    FOUR_C_ASSERT_ALWAYS(log_err_status == Core::LinAlg::MatrixFunctErrorType::no_errors,
        "Matrix logarithm evaluation failed!");

    // calculate residual of the equation for inelastic defgrad
    resFM.update(1.0, logT, dt, state_quantities_.curr_lpM_, 0.0);

    // calculate residual of the equation for plastic strain
    resepsp = plastic_strain - last_plastic_strain -
              dt * state_quantities_.curr_equiv_plastic_strain_rate_;
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
Core::LinAlg::Matrix<10, 10> Mat::InelasticDefgradTransvIsotropElastViscoplast::calculate_jacobian(
    const Core::LinAlg::Matrix<3, 3>& CM, const Core::LinAlg::Matrix<10, 1>& x,
    const Core::LinAlg::Matrix<3, 3>& last_iFinM, const double last_plastic_strain, const double dt,
    Mat::ViscoplastErrorType& err_status)
{
  // auxiliaries
  Core::LinAlg::FourTensor<3> tempFourTensor(true);
  Core::LinAlg::Matrix<9, 9> temp9x9(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);

  // extract inverse inelastic defgrad and plastic strain from input vector
  Core::LinAlg::Matrix<3, 3> iFinM = extract_inverse_inelastic_defgrad(x);
  double plastic_strain = x(9);

  // evaluate state derivatives
  state_quantity_derivatives_ = evaluate_state_quantity_derivatives(CM, iFinM, plastic_strain,
      err_status, dt);  // we do not reevaluate the state quantities, this
                        // was done in the residual computation already

  // get derivative of update tensor wrt inverse inelastic defgrad (in FourTensor form)
  Core::LinAlg::FourTensor<3> dEpdiFin_FourTensor(true);
  Core::LinAlg::Voigt::setup_four_tensor_from_9x9_voigt_matrix(
      dEpdiFin_FourTensor, state_quantity_derivatives_.curr_dEpdiFin_);

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

  // standard substepping
  if (!parameter()->bool_log_substepping())
  {
    // compute 9x9 north-west component block of the Jacobian (derivative of residual for
    // inelastic deformation gradient w.r.t. inelastic deformation gradient)
    Core::LinAlg::FourTensorOperations::multiply_matrix_four_tensor<3>(
        tempFourTensor, last_iFinM, dEpdiFin_FourTensor, true);
    Core::LinAlg::Voigt::setup_9x9_voigt_matrix_from_four_tensor(temp9x9, tempFourTensor);
    J_iFin_iFin.update(1.0, const_non_mat_tensors.id4_9x9_, -1.0, temp9x9, 0.0);

    // compute derivative of update tensor wrt plastic strain in matrix form
    Core::LinAlg::Matrix<3, 3> dEpdepsp_M(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::matrix_9x1_to_3x3(state_quantity_derivatives_.curr_dEpdepsp_, dEpdepsp_M);

    // compute 9x1 north-east component block of the Jacobian (derivative of residual for
    // inelastic deformation gradient w.r.t. plastic strain)
    temp3x3.multiply_nn(-1.0, last_iFinM, dEpdepsp_M, 0.0);
    Core::LinAlg::Voigt::matrix_3x3_to_9x1(temp3x3, J_iFin_epsp);

    // compute 1x9 south-west component block of the Jacobian (derivative of residual for plastic
    // strain w.r.t. inelastic deformation gradient)
    J_epsp_iFin.update(-dt * state_quantity_derivatives_.curr_dpsr_dequiv_stress_,
        state_quantity_derivatives_.curr_dequiv_stress_diFin_, 0.0);

    // compute south-east component of the Jacobian (derivative of residual for plastic
    // strain w.r.t. plastic strain)
    J_epsp_epsp = 1.0 - dt * state_quantity_derivatives_.curr_dpsr_depsp_;
  }
  else
  // logarithmic substepping
  {
    // compute 9x9 north-west component block of the Jacobian (derivative of residual for
    // inelastic deformation gradient w.r.t. inelastic deformation gradient)
    Core::LinAlg::Matrix<3, 3> last_FinM(Core::LinAlg::Initialization::zero);
    last_FinM.invert(last_iFinM);
    Core::LinAlg::Matrix<3, 3> T(Core::LinAlg::Initialization::zero);
    T.multiply_nn(1.0, last_FinM, iFinM, 0.0);
    Core::LinAlg::MatrixFunctErrorType log_deriv_err_status{
        Core::LinAlg::MatrixFunctErrorType::no_errors};
    Core::LinAlg::Matrix<9, 9> dlogTdT =
        Core::LinAlg::matrix_3x3_log_1st_deriv(T, log_deriv_err_status);
    FOUR_C_ASSERT_ALWAYS(log_deriv_err_status == Core::LinAlg::MatrixFunctErrorType::no_errors,
        "Matrix logarithm derivative evaluation failed!");
    Core::LinAlg::Matrix<9, 9> dTdiFin(Core::LinAlg::Initialization::zero);
    Core::LinAlg::FourTensorOperations::add_non_symmetric_product(
        1.0, last_FinM, const_non_mat_tensors.id3x3_, dTdiFin);
    Core::LinAlg::Matrix<9, 9> dlogTdiFin(Core::LinAlg::Initialization::zero);
    dlogTdiFin.multiply_nn(1.0, dlogTdT, dTdiFin, 0.0);
    J_iFin_iFin.update(1.0, dlogTdiFin, dt, state_quantity_derivatives_.curr_dlpdiFin_, 0.0);

    // compute 9x1 north-east component block of the Jacobian (derivative of residual for
    // inelastic deformation gradient w.r.t. plastic strain)
    J_iFin_epsp.update(dt, state_quantity_derivatives_.curr_dlpdepsp_, 0.0);

    // compute 1x9 south-west component block of the Jacobian (derivative of residual for plastic
    // strain w.r.t. inelastic deformation gradient)
    J_epsp_iFin.update(-dt * state_quantity_derivatives_.curr_dpsr_dequiv_stress_,
        state_quantity_derivatives_.curr_dequiv_stress_diFin_, 0.0);

    // compute south-east component of the Jacobian (derivative of residual for plastic
    // strain w.r.t. plastic strain)
    J_epsp_epsp = 1.0 - dt * state_quantity_derivatives_.curr_dpsr_depsp_;
  }

  // assemble and return the Jacobian
  return assemble_jacobian_from_components(J_iFin_iFin, J_iFin_epsp, J_epsp_iFin, J_epsp_epsp);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<10, 1> Mat::InelasticDefgradTransvIsotropElastViscoplast::local_newton_loop(
    const Core::LinAlg::Matrix<3, 3>& defgrad, const Core::LinAlg::Matrix<10, 1>& x,
    Mat::ViscoplastErrorType& err_status)
{
  // auxiliaries
  Core::LinAlg::Matrix<10, 10> temp10x10(Core::LinAlg::Initialization::zero);

  // calculate right Cauchy-Green deformation tensor
  Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
  CM.multiply_tn(1.0, defgrad, defgrad, 0.0);

  // set convergence tolerance for LNL and declare LNL matrices, vectors
  const double tolNR = 1.0e-12;
  const unsigned max_iter = 100;
  // Jacobian matrix
  Core::LinAlg::Matrix<10, 10> jacMat(Core::LinAlg::Initialization::zero);
  // increment of the solution variables
  Core::LinAlg::Matrix<10, 1> dx(Core::LinAlg::Initialization::zero);
  // residual of both equations
  Core::LinAlg::Matrix<10, 1> residual(Core::LinAlg::Initialization::zero);
  double residualNorm2(0.0);

  // declare solvers
  Core::LinAlg::FixedSizeSerialDenseSolver<10, 10, 1> solver_10_10_1;

  // define solution vector
  Core::LinAlg::Matrix<10, 1> sol = x;

  // declare current right CG (tensor interpolated later on in each substep)
  Core::LinAlg::Matrix<3, 3> curr_CM(Core::LinAlg::Initialization::zero);

  // initialize substep parameters
  SubstepParams substep_params = {
      .t = 0,                // t = 0 (time parameter)
      .substep_counter = 1,  // substep_counter = 1 (evaluation of first substep)
      .curr_dt = time_step_settings_
          .dt_,  // curr_dt = time_step_settings_.dt_ (first substep length = full time step)
      .time_step_halving_counter =
          0,  // time_step_halving_counter = 0 (full time step, therefore no substep halving yet)
      .total_num_of_substeps =
          1,      // total_num_of_substeps = 1 (1 substep to evaluate: full time step)
      .iter = 0,  // iter = 0 (0 LNL iterations for the current substep)
  };

  // set reference matrices for interpolation
  ref_matrices_ = {time_step_quantities_.last_rightCG_[gp_], CM};

  // declare error status for the new substep, to check whether we have halved the time step too
  // many times (false) or if a new substep is possible (true)
  bool new_substep_status = true;

  // declare error status for tensor interpolation
  Core::LinAlg::TensorInterpolationErrorType tensor_interpolator_err_status{
      Core::LinAlg::TensorInterpolationErrorType::NoErrors};

  // substepping procedure
  while (substep_params.substep_counter <= substep_params.total_num_of_substeps)
  {
    // reset iteration counter
    substep_params.iter = 0;

    // get the right Cauchy-Green tensor of the current substep
    curr_CM = tensor_interpolator_.get_interpolated_matrix(ref_matrices_, ref_locs_,
        (substep_params.t + substep_params.curr_dt) / time_step_settings_.dt_,
        tensor_interpolator_err_status);
    FOUR_C_ASSERT_ALWAYS(
        tensor_interpolator_err_status == Core::LinAlg::TensorInterpolationErrorType::NoErrors,
        "Tensor interpolation failed with err: {}",
        Core::LinAlg::make_error_message(tensor_interpolator_err_status));

    // Newton-Raphson scheme for the current substep
    while (true)
    {
      err_status = Mat::ViscoplastErrorType::NoErrors;

      // increment iteration counter
      ++substep_params.iter;

      // check the determinant of the obtained inverse inelastic defgrad: if far from 1.0, then
      // restart with smaller time step
      if (std::abs(extract_inverse_inelastic_defgrad(sol).determinant() - 1.0) > 0.05)
      {
        err_status = ViscoplastErrorType::NoPlasticIncompressibility;
        new_substep_status = prepare_new_substep(substep_params, sol, curr_CM);
        if (!new_substep_status) return sol;  // return with error
        continue;
      }

      // compute residual
      residual = calculate_local_newton_loop_residual(curr_CM, sol,
          time_step_quantities_.last_substep_plastic_defgrd_inverse_[gp_],
          time_step_quantities_.last_substep_plastic_strain_[gp_], substep_params.curr_dt,
          err_status);
      if (err_status != Mat::ViscoplastErrorType::NoErrors)
      {
        new_substep_status = prepare_new_substep(substep_params, sol, curr_CM);
        if (!new_substep_status) return sol;  // return with error
        continue;
      }

      // 2-norm of the residual
      residualNorm2 = residual.norm2();

      // check convergence
      if (residualNorm2 < tolNR)
      {
        // this means the current substep has converged: we need to update values of the
        // last_substep_ quantities, the time parameter, the substep count and to
        // break out of the loop of the current substep

        // update time parameter and substep count
        substep_params.t += substep_params.curr_dt;
        substep_params.substep_counter += 1;

        // update the values of history variables at the last converged state (if we have not
        // reached the last step yet)
        if (substep_params.substep_counter <= substep_params.total_num_of_substeps)
        {
          time_step_quantities_.last_substep_plastic_defgrd_inverse_[gp_] =
              extract_inverse_inelastic_defgrad(sol);
          time_step_quantities_.last_substep_plastic_strain_[gp_] = sol(9);
          // update last substep history variables of the viscoplastic flow rule
          viscoplastic_law_->update_gp_state(gp_);
        }

        // break out of the substep NR loop
        break;
      }

      // check if maximum iteration is reached: if we have halved the time step the maximum
      // number of times, throw error and finish execution. Otherwise throw exception and
      // proceed with a smaller time step in the substepping scheme!
      if (substep_params.iter > max_iter)
      {
        new_substep_status = prepare_new_substep(substep_params, sol, curr_CM);
        // if the halving number was exceeded --> return with error
        if (!new_substep_status)
        {
          err_status = ViscoplastErrorType::NoConvergenceLNL;
          return sol;  // return with error
        }
        continue;
      }

      // compute Jacobian
      jacMat = calculate_jacobian(curr_CM, sol,
          time_step_quantities_.last_substep_plastic_defgrd_inverse_[gp_],
          time_step_quantities_.last_substep_plastic_strain_[gp_], substep_params.curr_dt,
          err_status);
      if (err_status != Mat::ViscoplastErrorType::NoErrors)
      {
        new_substep_status = prepare_new_substep(substep_params, sol, curr_CM);
        if (!new_substep_status) return sol;  // return with error
        continue;
      }

      // scale residual by -1.0, in order to use it for the solution of the loop equation
      residual.scale(-1.0);

      // solve loop equation
      dx.clear();                                      // reset
      solver_10_10_1.set_matrix(jacMat);               // set A=jacMat
      solver_10_10_1.set_vectors(dx, residual);        // set dx=increment, residual=RHS
      solver_10_10_1.factor_with_equilibration(true);  // "some easy type of preconditioning"
      int err2 = solver_10_10_1.factor();              // factoring
      int err = solver_10_10_1.solve();                // X = A^-1 B
      if ((err != 0) || (err2 != 0))
      {
        err_status = ViscoplastErrorType::FailedSolLinSystLNL;
        new_substep_status = prepare_new_substep(substep_params, sol, curr_CM);
        if (!new_substep_status) return sol;  // return with error
        continue;
      }

      // update solution vector
      sol.update(1.0, dx, 1.0);
    }
  }


  // return the obtained solution
  return sol;
}

void Mat::InelasticDefgradTransvIsotropElastViscoplast::ConstMatTensors::set_material_const_tensors(
    const Core::LinAlg::Matrix<3, 1>& m)
{
  // set material-dependent tensors (fiber orientation)

  // structural tensor
  mm_.multiply_nt(1.0, m, m, 0.0);

  // deviatoric part of the structural tensor
  double tr_mm_ = mm_(0, 0) + mm_(1, 1) + mm_(2, 2);
  mm_dev_.update(1.0, mm_, -1.0 / 3.0 * tr_mm_, const_non_mat_tensors.id3x3_);

  // dyadic product of structural tensors
  Core::LinAlg::Matrix<6, 1> mm_V(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
      mm_, mm_V);
  mm_dyad_mm_.multiply_nt(1.0, mm_V, mm_V, 0.0);

  // dyadic product of deviatoric structural tensor with the structural tensor
  Core::LinAlg::Matrix<6, 1> mm_dev_V(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
      mm_dev_, mm_dev_V);
  mm_dev_dyad_mm_.multiply_nt(1.0, mm_dev_V, mm_V, 0.0);

  // dyadic product of identity with the structural tensor
  id_dyad_mm_.multiply_nt(1.0, const_non_mat_tensors.id6x1_, mm_V, 0.0);

  // sum of identity with the structural tensor
  id_plus_mm_.update(1.0, const_non_mat_tensors.id3x3_, 1.0, mm_, 0.0);
}

bool Mat::InelasticDefgradTransvIsotropElastViscoplast::check_predictor(
    const Core::LinAlg::Matrix<3, 3>& CM, const Core::LinAlg::Matrix<3, 3>& iFinM_pred,
    const double plastic_strain_pred, Mat::ViscoplastErrorType& err_status)
{
  // evaluate state with this elastic predictor and the minimum possible time step
  state_quantities_ = evaluate_state_quantities(
      CM, iFinM_pred, plastic_strain_pred, err_status, time_step_settings_.min_dt_);

  // check if the predicted plastic strain rate is 0 -> for flow rules with yield functions,
  // this means that the predictor is correct
  return (state_quantities_.curr_equiv_plastic_strain_rate_ < 1.0e-15);
}

bool Mat::InelasticDefgradTransvIsotropElastViscoplast::prepare_new_substep(
    SubstepParams& substep_params, Core::LinAlg::Matrix<10, 1>& sol,
    Core::LinAlg::Matrix<3, 3>& curr_CM)
{
  // extract substep parameters
  const double& t = substep_params.t;
  const unsigned int& substep_counter = substep_params.substep_counter;
  double& curr_dt = substep_params.curr_dt;
  unsigned int& time_step_halving_counter = substep_params.time_step_halving_counter;
  unsigned int& total_num_of_substeps = substep_params.total_num_of_substeps;
  unsigned int& iter = substep_params.iter;

  // the current iteration vector has reached a numerically inevaluable state -> we halve
  // the time step and apply substepping

  // halve the current time step
  curr_dt *= 1.0 / 2.0;
  time_step_halving_counter += 1;
  total_num_of_substeps += (total_num_of_substeps - substep_counter + 1);

  // check if we have halved the time step too many times
  if (time_step_halving_counter > parameter()->max_halve_number())
  {
    return false;
  }

  // reset the predictor to the last converged state
  sol = wrap_unknowns(time_step_quantities_.last_substep_plastic_defgrd_inverse_[gp_],
      time_step_quantities_.last_substep_plastic_strain_[gp_]);

  // initialize the error status for the tensor interpolation
  Core::LinAlg::TensorInterpolationErrorType tensor_interpolator_err_status{
      Core::LinAlg::TensorInterpolationErrorType::NoErrors};

  // recompute the current right CG
  curr_CM = tensor_interpolator_.get_interpolated_matrix(ref_matrices_, ref_locs_,
      (t + curr_dt) / time_step_settings_.dt_, tensor_interpolator_err_status);
  FOUR_C_ASSERT_ALWAYS(
      tensor_interpolator_err_status == Core::LinAlg::TensorInterpolationErrorType::NoErrors,
      "Tensor interpolation failed with err: {}",
      Core::LinAlg::make_error_message(tensor_interpolator_err_status));


  // reset iteration counter to 0, as we restart the Newton-Raphson Loop
  iter = 0;

  return true;  // no error
}

void Mat::InelasticDefgradTransvIsotropElastViscoplast::evaluate_additional_cmat_perturb_based(
    const Core::LinAlg::Matrix<3, 3>& FredM, Core::LinAlg::Matrix<6, 6>& cmatadd,
    const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<6, 9>& dSdiFinj)
{
  // ----- FD-based linearization ----- //
  // approximation using perturbations of the right Cauchy-Green deformation tensor, inspired
  // by the procedure described in Miehe et al. (1995)

  // auxiliaries
  Core::LinAlg::Matrix<3, 3> temp3x3(Core::LinAlg::Initialization::zero);

  // set update boolean to false
  update_hist_var_ = false;  // no update of the current_ values during the upcoming evaluation
  // of perturbed states

  // inverse of the reduced deformation gradient
  Core::LinAlg::Matrix<3, 3> iFredM(Core::LinAlg::Initialization::zero);
  iFredM.invert(FredM);

  // Voigt representation of the inverse inelastic defgrad
  Core::LinAlg::Matrix<9, 1> iFinV(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(
      time_step_quantities_.current_plastic_defgrd_inverse_[gp_], iFinV);

  // derivative of inverse inelastic deformation gradient w.r.t. right Cauchy-Green
  // deformation tensor, to be evaluated in the FD-based procedure
  Core::LinAlg::Matrix<9, 6> diFindC_FD(Core::LinAlg::Initialization::zero);

  // declare perturbed variables
  Core::LinAlg::Matrix<3, 3> perturbed_FM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3>* pointer_perturbed_FM = &perturbed_FM;
  Core::LinAlg::Matrix<3, 3> perturbed_CM(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> perturbed_iFinM(Core::LinAlg::Initialization::zero);

  // define the delta perturbed deformation gradients
  std::vector<Core::LinAlg::Matrix<3, 3>> delta_perturbed_defgrads(6);
  const double pert_fact = 1.0e-9;  // perturbation factor \f$ \epsilon \f$
  std::vector<std::tuple<int, int>> indices_array = {
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

    // get perturbed defgrad
    perturbed_FM.update(1.0, FredM, 1.0, delta_perturbed_defgrads[i], 0.0);

    // calculate perturbed right CG tensor
    perturbed_CM.multiply_tn(1.0, perturbed_FM, perturbed_FM, 0.0);

    // get corresponding inverse inelastic defgrad
    evaluate_inverse_inelastic_def_grad(pointer_perturbed_FM, iFin_other, perturbed_iFinM);
    Core::LinAlg::Matrix<9, 1> perturbed_iFinV(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Voigt::matrix_3x3_to_9x1(perturbed_iFinM, perturbed_iFinV);

    // update components of the required derivative
    for (int j = 0; j < 9; ++j)
    {
      diFindC_FD(j, i) += 1.0 / 2.0 * 1.0 / pert_fact * (perturbed_iFinV(j, 0) - iFinV(j, 0));
    }
  }

  // compute additional term to stiffness matrix additional_cmat
  cmatadd.multiply_nn(2.0, dSdiFinj, diFindC_FD, 1.0);

  // reset boolean for the history update
  update_hist_var_ = true;
}



FOUR_C_NAMESPACE_CLOSE
