// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_beam3_kirchhoff.hpp"
#include "4C_beam3_reissner.hpp"
#include "4C_comm_pack_buffer.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_FADmatrix_utils.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_plasticdruckerprager.hpp"
#include "4C_mat_service.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_singleton_owner.hpp"

namespace
{
  using namespace FourC;

  class DruckerPragerTest : public ::testing::Test
  {
   protected:
    void SetUp() override
    {
      Core::IO::InputParameterContainer container;
      container.add("YOUNG", 1.0);
      container.add("NUE", 0.25);
      container.add("DENS", 0.0);
      container.add("ISOHARD", 1.0);
      container.add("TOL", 1.e-12);
      container.add("C", 1.);
      container.add("ETA", 1.);
      container.add("XI", 1.);
      container.add("ETABAR", 1.);
      container.add("MAXITER", 50);
      container.add("TANG", std::string("consistent"));

      param_druckprag_ = std::shared_ptr(
          Mat::make_parameter(1, Core::Materials::MaterialType::m_pldruckprag, container));

      Global::Problem& problem = (*Global::Problem::instance());
      problem.materials()->set_read_from_problem(0);
      problem.materials()->insert(1, param_druckprag_);
      druckprag_ = std::make_shared<Mat::PlasticDruckerPrager>(
          dynamic_cast<Mat::PAR::PlasticDruckerPrager*>(param_druckprag_.get()));
    }

    std::shared_ptr<Core::Mat::PAR::Parameter> param_druckprag_;
    Core::Communication::PackBuffer data;
    std::shared_ptr<Mat::PlasticDruckerPrager> druckprag_;

    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;
  };

  //! test member function Pack and unpack
  TEST_F(DruckerPragerTest, TestPackUnpack)
  {
    Core::IO::InputParameterContainer container;
    druckprag_->setup(1, container);
    Core::LinAlg::Matrix<6, 1> input_strain;
    for (int i = 0; i < 6; ++i) input_strain(i) = .1;
    Teuchos::ParameterList paras;
    Core::LinAlg::Matrix<3, 3> defgrad(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> ref_stress(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; ++i)
      ref_stress(i) =
          (1.0 / ((1.0 + 0.25) * (1.0 - (2.0 * 0.25)))) * ((1.0 - 0.25) + 0.25 + 0.25) * .1;
    for (int i = 3; i < 6; ++i)
      ref_stress(i) =
          (1.0 / ((1.0 + 0.25) * (1.0 - (2.0 * 0.25)))) * ((1.0 - (2.0 * 0.25)) / 2.0) * .1;
    Core::LinAlg::Matrix<6, 6> result_cmat(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> result_stress(Core::LinAlg::Initialization::zero);
    druckprag_->pack(data);
    std::vector<char> dataSend;
    swap(dataSend, data());
    FourC::Mat::PlasticDruckerPrager plastic;
    Core::Communication::UnpackBuffer buffer(dataSend);
    plastic.unpack(buffer);
    plastic.evaluate(&defgrad, &input_strain, paras, &result_stress, &result_cmat, 0, 0);
    FOUR_C_EXPECT_NEAR(result_stress, ref_stress, 1.0e-12);
  };

  //! test member function Evaluate
  TEST_F(DruckerPragerTest, TestEvaluate)
  {
    Core::IO::InputParameterContainer container;
    druckprag_->setup(1, container);
    Core::LinAlg::Matrix<6, 1> input_strain;
    for (int i = 0; i < 6; ++i) input_strain(i) = .1;
    Teuchos::ParameterList paras;
    Core::LinAlg::Matrix<3, 3> defgrad(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> ref_stress(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; ++i)
      ref_stress(i) =
          (1.0 / ((1.0 + 0.25) * (1.0 - (2.0 * 0.25)))) * ((1.0 - 0.25) + 0.25 + 0.25) * .1;
    for (int i = 3; i < 6; ++i)
      ref_stress(i) =
          (1.0 / ((1.0 + 0.25) * (1.0 - (2.0 * 0.25)))) * ((1.0 - (2.0 * 0.25)) / 2.0) * .1;
    Core::LinAlg::Matrix<6, 6> result_cmat(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> result_stress(Core::LinAlg::Initialization::zero);
    druckprag_->evaluate(&defgrad, &input_strain, paras, &result_stress, &result_cmat, 0, 0);
    FOUR_C_EXPECT_NEAR(result_stress, ref_stress, 1.0e-12);
  };

  //! test member function Evaluate for Return to Cone
  TEST_F(DruckerPragerTest, TestEvaluateReturnToCone)
  {
    Core::IO::InputParameterContainer container;
    druckprag_->setup(1, container);
    Core::LinAlg::Matrix<6, 1> input_strain;
    for (int i = 0; i < 3; ++i) input_strain(i) = 0.0;
    for (int i = 3; i < 6; ++i) input_strain(i) = 2.2;
    Teuchos::ParameterList paras;
    Core::LinAlg::Matrix<3, 3> defgrad(Core::LinAlg::Initialization::zero);
    double Dgamma = (2.2 * sqrt(3) / 2.5 - 1.0) / 31.0 * 15.0;
    Core::LinAlg::Matrix<6, 1> ref_stress;
    for (int i = 0; i < 3; ++i) ref_stress(i) = -(Dgamma * (1.0 / (3.0 * (1.0 - (2.0 * 0.25)))));
    for (int i = 3; i < 6; ++i)
      ref_stress(i) = (1.0 / (2 * (1.0 + 0.25))) *
                      (1 - ((1.0 / (2 * (1.0 + 0.25))) * Dgamma / (2.2 * sqrt(3) / 2.5))) * 2.2;
    Core::LinAlg::Matrix<6, 6> result_cmat(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> result_stress(Core::LinAlg::Initialization::zero);
    druckprag_->evaluate(&defgrad, &input_strain, paras, &result_stress, &result_cmat, 0, 0);
    FOUR_C_EXPECT_NEAR(result_stress, ref_stress, 1.0e-12);
  };

  //! test member function Evaluate for Return to Apex
  TEST_F(DruckerPragerTest, TestEvaluateReturnToApex)
  {
    Core::IO::InputParameterContainer container;
    druckprag_->setup(1, container);
    Core::LinAlg::Matrix<6, 1> input_strain;
    for (int i = 0; i < 3; ++i) input_strain(i) = 1.0;
    for (int i = 3; i < 6; ++i) input_strain(i) = 0.0;
    Teuchos::ParameterList paras;
    Core::LinAlg::Matrix<3, 3> defgrad(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> ref_stress(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; ++i) ref_stress(i) = 2.0 - (10. / 15.) * (3. / 5.);
    Core::LinAlg::Matrix<6, 6> result_cmat(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> result_stress(Core::LinAlg::Initialization::zero);
    druckprag_->evaluate(&defgrad, &input_strain, paras, &result_stress, &result_cmat, 0, 0);
    FOUR_C_EXPECT_NEAR(result_stress, ref_stress, 1.0e-12);
  };

  //! test member function Evaluate for History and elastic unloading
  TEST_F(DruckerPragerTest, TestEvaluateHistory)
  {
    Core::IO::InputParameterContainer container;
    druckprag_->setup(1, container);
    Core::LinAlg::Matrix<6, 1, FAD> input_strain;
    for (int i = 0; i < 3; ++i) input_strain(i) = FAD(6, i, 0.1);
    for (int i = 3; i < 6; ++i) input_strain(i) = FAD(6, i, 0.1);
    Teuchos::ParameterList paras;
    Core::LinAlg::Matrix<3, 3> defgrad(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1, FAD> ref_stress(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 6> result_cmat(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1, FAD> result_stress(Core::LinAlg::Initialization::zero);
    druckprag_->evaluate(&defgrad, &input_strain, paras, &result_stress, &result_cmat, 0, 0);
    Core::LinAlg::Matrix<6, 6> ref_cmat(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++)
      {
        ref_cmat(i, j) = result_stress(i).dx(j);
      }
    }
    FOUR_C_EXPECT_NEAR(result_cmat, ref_cmat, 1.0e-12);
    druckprag_->update();
    for (int i = 0; i < 3; ++i) input_strain(i) = FAD(6, i, 1.0);
    for (int i = 3; i < 6; ++i) input_strain(i) = FAD(6, i, 0.0);
    druckprag_->evaluate(&defgrad, &input_strain, paras, &result_stress, &result_cmat, 0, 0);
    for (int i = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++)
      {
        ref_cmat(i, j) = result_stress(i).dx(j);
      }
    }
    FOUR_C_EXPECT_NEAR(result_cmat, ref_cmat, 1.0e-12);
    druckprag_->update();
    for (int i = 0; i < 3; ++i) input_strain(i) = FAD(6, i, 0.2);
    for (int i = 3; i < 6; ++i) input_strain(i) = FAD(6, i, 0.0);
    druckprag_->evaluate(&defgrad, &input_strain, paras, &result_stress, &result_cmat, 0, 0);
    for (int i = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++)
      {
        ref_cmat(i, j) = result_stress(i).dx(j);
      }
    }
    FOUR_C_EXPECT_NEAR(Core::FADUtils::cast_to_double(result_stress),
        Core::FADUtils::cast_to_double(ref_stress), 1.0e-12);
  };

  //! test member function Evaluate for arbitrary values
  TEST_F(DruckerPragerTest, TestEvaluateRandomStrain)
  {
    Core::IO::InputParameterContainer container;
    druckprag_->setup(1, container);
    Core::LinAlg::Matrix<6, 1> input_strain;
    input_strain(0) = 1.1;
    input_strain(1) = 2.0;
    input_strain(2) = 0.1;
    input_strain(3) = 2.5;
    input_strain(4) = 1.4;
    input_strain(5) = 1.0;
    Teuchos::ParameterList paras;
    Core::LinAlg::Matrix<3, 3> defgrad(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> ref_stress(Core::LinAlg::Initialization::zero);
    ref_stress(0) = 1.3231031817668;
    ref_stress(1) = 1.7934880206154;
    ref_stress(2) = 0.8004533608238;
    ref_stress(3) = 0.6533122761787;
    ref_stress(4) = 0.3658548746601;
    ref_stress(5) = 0.2613249104715;
    Core::LinAlg::Matrix<6, 6> result_cmat(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> result_stress(Core::LinAlg::Initialization::zero);
    druckprag_->evaluate(&defgrad, &input_strain, paras, &result_stress, &result_cmat, 0, 0);
    FOUR_C_EXPECT_NEAR(result_stress, ref_stress, 1.0e-12);
  };

  //! test member function Evaluate
  TEST_F(DruckerPragerTest, TestEvaluateCmat)
  {
    Core::IO::InputParameterContainer container;
    druckprag_->setup(1, container);
    Core::LinAlg::Matrix<6, 1, FAD> input_strain;
    for (int i = 0; i < 6; ++i) input_strain(i) = FAD(6, i, .1 * i);
    Teuchos::ParameterList paras;
    Core::LinAlg::Matrix<3, 3> defgrad(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1, FAD> ref_stress(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; ++i)
      ref_stress(i) =
          FAD((1.0 / ((1.0 + 0.25) * (1.0 - (2.0 * 0.25)))) * ((1.0 - 0.25) + 0.25 + 0.25) * .1);
    for (int i = 3; i < 6; ++i)
      ref_stress(i) =
          FAD((1.0 / ((1.0 + 0.25) * (1.0 - (2.0 * 0.25)))) * ((1.0 - (2.0 * 0.25)) / 2.0) * .1);
    Core::LinAlg::Matrix<6, 6> result_cmat(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1, FAD> result_stress(Core::LinAlg::Initialization::zero);
    druckprag_->evaluate(&defgrad, &input_strain, paras, &result_stress, &result_cmat, 0, 0);
    Core::LinAlg::Matrix<6, 6> ref_cmat(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++)
      {
        ref_cmat(i, j) = result_stress(i).dx(j);
      }
    }
    FOUR_C_EXPECT_NEAR(result_cmat, ref_cmat, 1.0e-12);
  };

  //! test CMAT matrix for Return to Cone
  TEST_F(DruckerPragerTest, TestEvaluateReturnToConeCmat)
  {
    Core::IO::InputParameterContainer container;
    druckprag_->setup(1, container);
    Core::LinAlg::Matrix<6, 1, FAD> input_strain;
    for (int i = 0; i < 3; ++i) input_strain(i) = FAD(6, i, 0.1 * i);
    for (int i = 3; i < 6; ++i) input_strain(i) = FAD(6, i, 2.2 * i);
    Teuchos::ParameterList paras;
    Core::LinAlg::Matrix<3, 3> defgrad(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 6> result_cmat(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1, FAD> result_stress(Core::LinAlg::Initialization::zero);
    druckprag_->evaluate(&defgrad, &input_strain, paras, &result_stress, &result_cmat, 0, 0);
    Core::LinAlg::Matrix<6, 6> ref_cmat(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++)
      {
        ref_cmat(i, j) = result_stress(i).dx(j);
      }
    }
    FOUR_C_EXPECT_NEAR(result_cmat, ref_cmat, 1.0e-12);
  };
  TEST_F(DruckerPragerTest, TestEvaluateReturnToApexCmat)
  {
    Core::IO::InputParameterContainer container;
    druckprag_->setup(1, container);
    Core::LinAlg::Matrix<6, 1, FAD> input_strain;
    for (int i = 0; i < 3; ++i) input_strain(i) = FAD(6, i, 1.0);
    for (int i = 3; i < 6; ++i) input_strain(i) = FAD(6, i, 0.0);
    Teuchos::ParameterList paras;
    Core::LinAlg::Matrix<3, 3> defgrad(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 6> result_cmat(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1, FAD> result_stress(Core::LinAlg::Initialization::zero);
    druckprag_->evaluate(&defgrad, &input_strain, paras, &result_stress, &result_cmat, 0, 0);
    Core::LinAlg::Matrix<6, 6> ref_cmat(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++)
      {
        ref_cmat(i, j) = result_stress(i).dx(j);
      }
    }
    FOUR_C_EXPECT_NEAR(result_cmat, ref_cmat, 1.0e-12);
  };

  //! test CMAT matrix for Return to Apex
  TEST_F(DruckerPragerTest, TestEvaluateRandomStrainCmat)
  {
    Core::IO::InputParameterContainer container;
    druckprag_->setup(1, container);
    Core::LinAlg::Matrix<6, 1, FAD> input_strain;
    input_strain(0) = FAD(6, 0, 1.1);
    input_strain(1) = FAD(6, 1, 2.0);
    input_strain(2) = FAD(6, 2, 0.1);
    input_strain(3) = FAD(6, 3, 2.5);
    input_strain(4) = FAD(6, 4, 1.4);
    input_strain(5) = FAD(6, 5, 1.0);
    Teuchos::ParameterList paras;
    Core::LinAlg::Matrix<3, 3> defgrad(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1, FAD> ref_stress(Core::LinAlg::Initialization::zero);
    ref_stress(0) = FAD(1.4142412329012);
    ref_stress(1) = FAD(1.8571566160540);
    ref_stress(2) = FAD(0.9221130293981);
    ref_stress(3) = FAD(0.6151602543789);
    ref_stress(4) = FAD(0.3444897424522);
    ref_stress(5) = FAD(0.2460641017516);
    Core::LinAlg::Matrix<6, 6> result_cmat(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1, FAD> result_stress(Core::LinAlg::Initialization::zero);
    druckprag_->evaluate(&defgrad, &input_strain, paras, &result_stress, &result_cmat, 0, 0);
    Core::LinAlg::Matrix<6, 6> ref_cmat(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++)
      {
        ref_cmat(i, j) = result_stress(i).dx(j);
      }
    }
    FOUR_C_EXPECT_NEAR(result_cmat, ref_cmat, 1.0e-12);
  };
}  // namespace
