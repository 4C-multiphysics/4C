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
#include "4C_mat_stvenantkirchhoff.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

#include <Teuchos_ParameterList.hpp>

namespace
{
  using namespace FourC;

  class StVenantKirchhoffTest : public ::testing::Test
  {
   protected:
    void SetUp() override
    {
      Core::IO::InputParameterContainer container;
      // add material parameters to container
      container.add("YOUNG", young_);
      container.add("NUE", nu_);
      container.add("DENS", rho_);

      // initialize parameter class for StVenantKirchhoff material with container
      parameters_stvenantkirchhoff_ = std::make_shared<Mat::PAR::StVenantKirchhoff>(
          Core::Mat::PAR::Parameter::Data{.parameters = container});

      // initialize stvenantkirchhoff material with parameter class
      stvenantkirchhoff_ =
          std::make_shared<Mat::StVenantKirchhoff>(parameters_stvenantkirchhoff_.get());
    }

    //! material parameters
    const double young_ = 210.;
    const double nu_ = 0.3;
    const double rho_ = 1.0;  // dummy value (needed for construction)
    std::shared_ptr<Mat::PAR::StVenantKirchhoff> parameters_stvenantkirchhoff_;

    //! material class
    std::shared_ptr<Mat::StVenantKirchhoff> stvenantkirchhoff_;

    //! Test Green-Lagrange Strain
    Core::LinAlg::SymmetricTensor<double, 3, 3> input_glstrain_ = Core::LinAlg::assume_symmetry(
        Core::LinAlg::Tensor<double, 3, 3>{{{1.0, 0.5, 0.5}, {0.5, 1.0, 0.5}, {0.5, 0.5, 1.0}}});

    //! calculate reference results for stress
    const double ref_stress_normal_ =
        (young_ / ((1.0 + nu_) * (1.0 - (2.0 * nu_)))) * ((1.0 - nu_) + nu_ + nu_);

    const double ref_stress_shear_ =
        (young_ / ((1.0 + nu_) * (1.0 - (2.0 * nu_)))) * ((1.0 - (2.0 * nu_)) / 2.0);

    const Core::LinAlg::SymmetricTensor<double, 3, 3> ref_stress_ =
        Core::LinAlg::assume_symmetry(Core::LinAlg::Tensor<double, 3, 3>{
            {{ref_stress_normal_, ref_stress_shear_, ref_stress_shear_},
                {ref_stress_shear_, ref_stress_normal_, ref_stress_shear_},
                {ref_stress_shear_, ref_stress_shear_, ref_stress_normal_}}});
  };

  TEST_F(StVenantKirchhoffTest, TestEvaluateTensor)
  {
    // Resulting material stiffness matrix
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> result_cmat{};

    // Resulting stress
    Core::LinAlg::SymmetricTensor<double, 3, 3> result_stress{};

    Teuchos::ParameterList params{};

    // Call evaluate function with test strain
    double total_time = 0.0;
    double time_step_size = 1.0;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};
    stvenantkirchhoff_->evaluate(
        nullptr, input_glstrain_, params, context, result_stress, result_cmat, 0, 0);

    // Test member function results using reference stress values
    FOUR_C_EXPECT_NEAR(result_stress, ref_stress_, 1.0e-4);
  }

  TEST_F(StVenantKirchhoffTest, TestStrainEnergy)
  {
    // define reference result for strain energy
    const double ref_strain_energy = 908.6538;

    // Input strain
    const Core::LinAlg::Matrix<6, 1> test_glstrain(input_glstrain_.data(), false);

    // result strain energy
    ;

    int eleGID = 1;

    double total_time = 0.0;
    double time_step_size = 1.0;
    Mat::EvaluationContext<3> context{.total_time = &total_time,
        .time_step_size = &time_step_size,
        .xi = {},
        .ref_coords = nullptr};

    // Call evaluate function with test strain
    double result_psi = stvenantkirchhoff_->strain_energy(input_glstrain_, context, 0, eleGID);

    // test result with respect to reference result
    EXPECT_NEAR(result_psi, ref_strain_energy, 1.0e-4);
  }
}  // namespace
