// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_stvenantkirchhoff_orthotropic.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

#include <Teuchos_ParameterList.hpp>

namespace
{
  using namespace FourC;

  class StVenantKirchhoffOrthotropicTest : public ::testing::Test
  {
   protected:
    void SetUp() override
    {
      Core::IO::InputParameterContainer container;
      // add material parameters to container
      container.add("YOUNG", young_);
      container.add("SHEAR", shear_);
      container.add("NUE", nu_);
      container.add("DENS", rho_);

      // initialize parameter class for StVenantKirchhoffOrthotropic material with container
      parameters_stvenantkirchhofforthotropic_ =
          std::make_shared<Mat::PAR::StVenantKirchhoffOrthotropic>(
              Core::Mat::PAR::Parameter::Data{.parameters = container});

      // initialize stvenantkirchhofforthotropic material with parameter class
      stvenantkirchhofforthotropic_ = std::make_shared<Mat::StVenantKirchhoffOrthotropic>(
          parameters_stvenantkirchhofforthotropic_.get());
    }

    //! material parameters
    const std::vector<double> young_ = {162.0, 10.0, 10.0};
    const std::vector<double> shear_ = {3.5, 5.2, 5.2};
    const std::vector<double> nu_ = {0.35, 0.49, 0.35};
    const double rho_ = 1.0;  // dummy value (needed for construction)
    std::shared_ptr<Mat::PAR::StVenantKirchhoffOrthotropic>
        parameters_stvenantkirchhofforthotropic_;

    //! material class
    std::shared_ptr<Mat::StVenantKirchhoffOrthotropic> stvenantkirchhofforthotropic_;
  };

  TEST_F(StVenantKirchhoffOrthotropicTest, TestEvaluateStressLinearization)
  {
    auto cmat = stvenantkirchhofforthotropic_->evaluate_stress_linearization(young_, shear_, nu_);
    Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

    Core::LinAlg::Matrix<6, 6> cmat_reference(Core::LinAlg::Initialization::zero);
    cmat_reference(0, 0) = 166.950729699389;
    cmat_reference(0, 1) = 7.072471027841;
    cmat_reference(0, 2) = 7.072471027841;
    cmat_reference(1, 0) = 7.072471027841;
    cmat_reference(1, 1) = 13.459234709699;
    cmat_reference(1, 2) = 6.747825290301;
    cmat_reference(2, 0) = 7.072471027841;
    cmat_reference(2, 1) = 6.747825290301;
    cmat_reference(2, 2) = 13.459234709699;
    cmat_reference(3, 3) = 3.500000000000;
    cmat_reference(4, 4) = 5.200000000000;
    cmat_reference(5, 5) = 5.200000000000;

    FOUR_C_EXPECT_NEAR(cmat_view, cmat_reference, 1e-06);
  }

  TEST_F(StVenantKirchhoffOrthotropicTest, TestEvaluateTensor)
  {
    const Core::LinAlg::SymmetricTensor<double, 3, 3> ref_stress = Core::LinAlg::assume_symmetry(
        Core::LinAlg::Tensor<double, 3, 3>{{{181.095671697643, 3.500000000000, 5.200000000000},
            {3.500000000000, 27.279530696632, 5.200000000000},
            {5.200000000000, 5.200000000000, 27.279530696632}}});

    Core::LinAlg::SymmetricTensor<double, 3, 3> input_glstrain = Core::LinAlg::assume_symmetry(
        Core::LinAlg::Tensor<double, 3, 3>{{{1.0, 0.5, 0.5}, {0.5, 1.0, 0.5}, {0.5, 0.5, 1.0}}});

    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> result_cmat{};
    Core::LinAlg::SymmetricTensor<double, 3, 3> result_stress{};

    Teuchos::ParameterList params{};

    const double total_time = 0.0;
    const double time_step_size = 1.0;
    Mat::EvaluationContext context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};

    stvenantkirchhofforthotropic_->evaluate(
        nullptr, input_glstrain, params, context, result_stress, result_cmat, 0, 0);

    FOUR_C_EXPECT_NEAR(result_stress, ref_stress, 1.0e-4);
  }

  TEST_F(StVenantKirchhoffOrthotropicTest, TestStrainEnergy)
  {
    const double ref_strain_energy = 124.7773669;

    Core::LinAlg::SymmetricTensor<double, 3, 3> input_glstrain = Core::LinAlg::assume_symmetry(
        Core::LinAlg::Tensor<double, 3, 3>{{{1.0, 0.5, 0.5}, {0.5, 1.0, 0.5}, {0.5, 0.5, 1.0}}});

    const Core::LinAlg::Matrix<6, 1> test_glstrain(input_glstrain.data(), false);

    const int eleGID = 1;
    const double total_time = 0.0;
    const double time_step_size = 1.0;
    Mat::EvaluationContext context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};

    double result_psi =
        stvenantkirchhofforthotropic_->strain_energy(input_glstrain, context, 0, eleGID);

    EXPECT_NEAR(result_psi, ref_strain_energy, 1.0e-4);
  }
}  // namespace
