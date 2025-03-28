// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_serialdensevector.hpp"
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
    std::array<double, 6> input_glstrain_ = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    //! calculate reference results for stress
    const double ref_stress_normal_ =
        (young_ / ((1.0 + nu_) * (1.0 - (2.0 * nu_)))) * ((1.0 - nu_) + nu_ + nu_);

    const double ref_stress_shear_ =
        (young_ / ((1.0 + nu_) * (1.0 - (2.0 * nu_)))) * ((1.0 - (2.0 * nu_)) / 2.0);

    const std::array<double, 6> ref_stress_ = {ref_stress_normal_, ref_stress_normal_,
        ref_stress_normal_, ref_stress_shear_, ref_stress_shear_, ref_stress_shear_};
  };

  TEST_F(StVenantKirchhoffTest, TestEvaluateEpetraSerialDenseMatrix)
  {
    // Input strain
    const Core::LinAlg::SerialDenseVector input_glstrain(Teuchos::Copy, input_glstrain_.data(), 6);

    // Resulting material stiffness matrix
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> result_cmat =
        std::make_shared<Core::LinAlg::SerialDenseMatrix>(6, 6);

    // Resulting stress
    std::shared_ptr<Core::LinAlg::SerialDenseVector> result_stress =
        std::make_shared<Core::LinAlg::SerialDenseVector>(6);

    // Call evaluate function with test strain
    stvenantkirchhoff_->evaluate(&input_glstrain, result_cmat.get(), result_stress.get());

    // Test member function results using reference stress values
    FOUR_C_EXPECT_ITERABLE_NEAR(result_stress->values(), ref_stress_.data(), 6, 1.0e-4);
  }

  TEST_F(StVenantKirchhoffTest, TestEvaluateLinalgMatrix)
  {
    // Resulting stress
    Core::LinAlg::Matrix<6, 1> result_stress(Core::LinAlg::Initialization::zero);

    // Resulting material stiffness matrix
    Core::LinAlg::Matrix<6, 6> result_cmat(Core::LinAlg::Initialization::zero);

    // Input deformation gradient, which is not used here
    Core::LinAlg::Matrix<3, 3> defgrad(Core::LinAlg::Initialization::zero);

    // ParameterList, also not used here
    Teuchos::ParameterList paras;

    // Input strain
    const Core::LinAlg::Matrix<6, 1> input_strain(input_glstrain_.data(), false);

    // Reference stress
    const Core::LinAlg::Matrix<6, 1> ref_stress(ref_stress_.data(), false);

    // Call evaluate function with test strain
    stvenantkirchhoff_->evaluate(
        &defgrad, &input_strain, paras, &result_stress, &result_cmat, 0, 0);

    // Test member function results using reference stress values
    FOUR_C_EXPECT_NEAR(result_stress, ref_stress, 1.0e-4);
  }

  TEST_F(StVenantKirchhoffTest, TestStrainEnergy)
  {
    // define reference result for strain energy
    const double ref_strain_energy = 908.6538;

    // Input strain
    const Core::LinAlg::Matrix<6, 1> test_glstrain(input_glstrain_.data(), false);

    // result strain energy
    double result_psi;

    int eleGID = 1;

    // Call evaluate function with test strain
    stvenantkirchhoff_->strain_energy(test_glstrain, result_psi, 0, eleGID);

    // test result with respect to reference result
    EXPECT_NEAR(result_psi, ref_strain_energy, 1.0e-4);
  }
}  // namespace
