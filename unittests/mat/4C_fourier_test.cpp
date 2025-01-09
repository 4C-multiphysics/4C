// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_comm_pack_buffer.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_fourier.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_singleton_owner.hpp"

namespace
{
  using namespace FourC;

  enum TensorType
  {
    scalar,
    diagonal,
    full
  };


  class IsotropicTest : public ::testing::TestWithParam<TensorType>
  {
   protected:
    void SetUp() override
    {
      auto tensor_type = GetParam();

      switch (tensor_type)
      {
        case TensorType::scalar:
        {
          conduct_para_num_ = 1;
          conduct_ = {1.0};
          break;
        }
        case TensorType::diagonal:
        {
          conduct_para_num_ = 3;
          conduct_ = {1.0, 1.0, 1.0};
          break;
        }
        case TensorType::full:
        {
          conduct_para_num_ = 9;
          conduct_ = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
          break;
        }
      }

      Core::IO::InputParameterContainer container;
      container.add("CONDUCT_PARA_NUM", conduct_para_num_);
      container.add("CAPA", capa_);
      container.add("CONDUCT", conduct_);

      parameters_fourier_ = std::shared_ptr(
          Mat::make_parameter(1, Core::Materials::MaterialType::m_thermo_fourier, container));

      Global::Problem* problem = Global::Problem::instance();
      problem->materials()->set_read_from_problem(0);
      problem->materials()->insert(1, parameters_fourier_);

      fourier_ = std::make_shared<Mat::Fourier>(
          dynamic_cast<Mat::PAR::Fourier*>(parameters_fourier_.get()));
    }

    const double capa_ = 420.0;
    int conduct_para_num_;
    std::vector<double> conduct_;

    std::shared_ptr<Core::Mat::PAR::Parameter> parameters_fourier_;
    std::shared_ptr<Mat::Fourier> fourier_;

    Core::Communication::PackBuffer data_;

    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard_;
  };

  //! test member function pack and unpack
  TEST_P(IsotropicTest, TestPackUnpack)
  {
    Core::LinAlg::Matrix<3, 1> ref_heatflux(true);
    ref_heatflux(0, 0) = 4.0;
    ref_heatflux(1, 0) = 20.0;
    ref_heatflux(2, 0) = 100.0;

    Core::LinAlg::Matrix<3, 3> ref_cmat(true);
    ref_cmat(0, 0) = 1.0;
    ref_cmat(1, 1) = 1.0;
    ref_cmat(2, 2) = 1.0;

    Core::LinAlg::Matrix<3, 1> gradtemp(true);
    gradtemp(0, 0) = 4.0;
    gradtemp(1, 0) = 20.0;
    gradtemp(2, 0) = 100.0;

    Core::LinAlg::Matrix<3, 1> result_heatflux(true);
    Core::LinAlg::Matrix<3, 3> result_cmat(true);

    fourier_->pack(data_);
    std::vector<char> dataSend;
    swap(dataSend, data_());
    FourC::Mat::Fourier aniso;
    Core::Communication::UnpackBuffer buffer(dataSend);
    aniso.unpack(buffer);

    aniso.evaluate(gradtemp, result_cmat, result_heatflux);

    FOUR_C_EXPECT_NEAR(result_cmat, ref_cmat, 1.0e-12);
    FOUR_C_EXPECT_NEAR(result_heatflux, ref_heatflux, 1.0e-12);
  }

  //! test member function evaluate
  TEST_P(IsotropicTest, TestEvaluate)
  {
    Core::LinAlg::Matrix<3, 1> ref_heatflux(true);
    ref_heatflux(0, 0) = 4.0;
    ref_heatflux(1, 0) = 20.0;
    ref_heatflux(2, 0) = 100.0;

    Core::LinAlg::Matrix<3, 3> ref_cmat(true);
    ref_cmat(0, 0) = 1.0;
    ref_cmat(1, 1) = 1.0;
    ref_cmat(2, 2) = 1.0;

    Core::LinAlg::Matrix<3, 1> gradtemp(true);
    gradtemp(0, 0) = 4.0;
    gradtemp(1, 0) = 20.0;
    gradtemp(2, 0) = 100.0;

    Core::LinAlg::Matrix<3, 1> result_heatflux(true);
    Core::LinAlg::Matrix<3, 3> result_cmat(true);

    fourier_->evaluate(gradtemp, result_cmat, result_heatflux);

    FOUR_C_EXPECT_NEAR(result_cmat, ref_cmat, 1.0e-12);
    FOUR_C_EXPECT_NEAR(result_heatflux, ref_heatflux, 1.0e-12);
  }

  INSTANTIATE_TEST_SUITE_P(Fourier, IsotropicTest,
      ::testing::Values(TensorType::scalar, TensorType::diagonal, TensorType::full));


  class AnisotropicTest : public ::testing::TestWithParam<TensorType>
  {
   protected:
    void SetUp() override
    {
      auto tensor_type = GetParam();

      switch (tensor_type)
      {
        case TensorType::scalar:
        {
          FOUR_C_THROW("Scalar tensor case not available for anisotropic material behavior.");
        }
        case TensorType::diagonal:
        {
          conduct_para_num_ = 3;
          conduct_ = {1.0, 10.0, 100.0};
          break;
        }
        case TensorType::full:
        {
          conduct_para_num_ = 9;
          conduct_ = {1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 100.0};
          break;
        }
      }

      Core::IO::InputParameterContainer container;
      container.add("CONDUCT_PARA_NUM", conduct_para_num_);
      container.add("CAPA", capa_);
      container.add("CONDUCT", conduct_);

      parameters_fourier_ = std::shared_ptr(
          Mat::make_parameter(1, Core::Materials::MaterialType::m_thermo_fourier, container));

      Global::Problem* problem = Global::Problem::instance();
      problem->materials()->set_read_from_problem(0);
      problem->materials()->insert(1, parameters_fourier_);

      fourier_ = std::make_shared<Mat::Fourier>(
          dynamic_cast<Mat::PAR::Fourier*>(parameters_fourier_.get()));
    }

    const double capa_ = 420.0;
    int conduct_para_num_;
    std::vector<double> conduct_;

    std::shared_ptr<Core::Mat::PAR::Parameter> parameters_fourier_;
    std::shared_ptr<Mat::Fourier> fourier_;

    Core::Communication::PackBuffer data_;

    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard_;
  };

  //! test member function pack and unpack
  TEST_P(AnisotropicTest, TestPackUnpack)
  {
    Core::LinAlg::Matrix<3, 1> ref_heatflux(true);
    ref_heatflux(0, 0) = 4.0;
    ref_heatflux(1, 0) = 20.0;
    ref_heatflux(2, 0) = 100.0;

    Core::LinAlg::Matrix<3, 3> ref_cmat(true);
    ref_cmat(0, 0) = 1.0;
    ref_cmat(1, 1) = 10.0;
    ref_cmat(2, 2) = 100.0;

    Core::LinAlg::Matrix<3, 1> gradtemp(true);
    gradtemp(0, 0) = 4.0;
    gradtemp(1, 0) = 2.0;
    gradtemp(2, 0) = 1.0;

    Core::LinAlg::Matrix<3, 1> result_heatflux(true);
    Core::LinAlg::Matrix<3, 3> result_cmat(true);

    fourier_->pack(data_);
    std::vector<char> dataSend;
    swap(dataSend, data_());
    FourC::Mat::Fourier aniso;
    Core::Communication::UnpackBuffer buffer(dataSend);
    aniso.unpack(buffer);

    aniso.evaluate(gradtemp, result_cmat, result_heatflux);

    FOUR_C_EXPECT_NEAR(result_cmat, ref_cmat, 1.0e-12);
    FOUR_C_EXPECT_NEAR(result_heatflux, ref_heatflux, 1.0e-12);
  }

  //! test member function evaluate
  TEST_P(AnisotropicTest, TestEvaluate)
  {
    Core::LinAlg::Matrix<3, 1> ref_heatflux(true);
    ref_heatflux(0, 0) = 4.0;
    ref_heatflux(1, 0) = 20.0;
    ref_heatflux(2, 0) = 100.0;

    Core::LinAlg::Matrix<3, 3> ref_cmat(true);
    ref_cmat(0, 0) = 1.0;
    ref_cmat(1, 1) = 10.0;
    ref_cmat(2, 2) = 100.0;

    Core::LinAlg::Matrix<3, 1> gradtemp(true);
    gradtemp(0, 0) = 4.0;
    gradtemp(1, 0) = 2.0;
    gradtemp(2, 0) = 1.0;

    Core::LinAlg::Matrix<3, 1> result_heatflux(true);
    Core::LinAlg::Matrix<3, 3> result_cmat(true);

    fourier_->evaluate(gradtemp, result_cmat, result_heatflux);

    FOUR_C_EXPECT_NEAR(result_cmat, ref_cmat, 1.0e-12);
    FOUR_C_EXPECT_NEAR(result_heatflux, ref_heatflux, 1.0e-12);
  }

  INSTANTIATE_TEST_SUITE_P(
      Fourier, AnisotropicTest, ::testing::Values(TensorType::diagonal, TensorType::full));

}  // namespace