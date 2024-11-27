// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_comm_pack_buffer.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_fourieraniso.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_singleton_owner.hpp"

namespace
{
  using namespace FourC;

  class FourierAnisoTest : public ::testing::Test
  {
   protected:
    void SetUp() override
    {
      Core::IO::InputParameterContainer container;
      container.add("NUMSPACEDIM", num_space_dim_);
      container.add("CAPA", capa_);
      container.add("CONDUCT", conduct_);

      parameters_fourieraniso_ = std::shared_ptr(Mat::make_parameter(1, Core::Materials::MaterialType::m_th_fourier_aniso, container));

      Global::Problem* problem = Global::Problem::instance();
      problem->materials()->set_read_from_problem(0);
      problem->materials()->insert(1, parameters_fourieraniso_);

      fourieraniso_ = std::make_shared<Mat::FourierAniso>(dynamic_cast<Mat::PAR::FourierAniso*>(parameters_fourieraniso_.get()));
    }

    const int num_space_dim_ = 3;
    const double capa_ = 420.0;
    const std::vector<double> conduct_ = {1.0, 10.0, 100.0};

    std::shared_ptr<Core::Mat::PAR::Parameter> parameters_fourieraniso_;
    std::shared_ptr<Mat::FourierAniso> fourieraniso_;

    Core::Communication::PackBuffer data_;

    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard_;
  };

  //! test member function pack and unpack
  TEST_F(FourierAnisoTest, TestPackUnpack)
  {
    Core::LinAlg::Matrix<3, 1> ref_heatflux(true);
    ref_heatflux(0, 0) = 4.0;
    ref_heatflux(1, 0) = 20.0;
    ref_heatflux(2, 0) = 100.0;

    Core::LinAlg::Matrix<3, 3> ref_cmat(true);
    ref_cmat(0, 0) = conduct_[0];
    ref_cmat(1, 1) = conduct_[1];
    ref_cmat(2, 2) = conduct_[2];

    Core::LinAlg::Matrix<3, 1> gradtemp(true);
    gradtemp(0, 0) = 4.0;
    gradtemp(1, 0) = 2.0;
    gradtemp(2, 0) = 1.0;

    Core::LinAlg::Matrix<3, 1> result_heatflux(true);
    Core::LinAlg::Matrix<3, 3> result_cmat(true);

    fourieraniso_->pack(data_);
    std::vector<char> dataSend;
    swap(dataSend, data_());
    FourC::Mat::FourierAniso aniso;
    Core::Communication::UnpackBuffer buffer(dataSend);
    aniso.unpack(buffer);

    aniso.evaluate(gradtemp, result_cmat, result_heatflux);

    FOUR_C_EXPECT_NEAR(result_cmat, ref_cmat, 1.0e-12);
    FOUR_C_EXPECT_NEAR(result_heatflux, ref_heatflux, 1.0e-12);
  }

  //! test member function evaluate
  TEST_F(FourierAnisoTest, TestEvaluate)
  {
    Core::LinAlg::Matrix<3, 1> ref_heatflux(true);
    ref_heatflux(0, 0) = 4.0;
    ref_heatflux(1, 0) = 20.0;
    ref_heatflux(2, 0) = 100.0;

    Core::LinAlg::Matrix<3, 3> ref_cmat(true);
    ref_cmat(0, 0) = conduct_[0];
    ref_cmat(1, 1) = conduct_[1];
    ref_cmat(2, 2) = conduct_[2];

    Core::LinAlg::Matrix<3, 1> gradtemp(true);
    gradtemp(0, 0) = 4.0;
    gradtemp(1, 0) = 2.0;
    gradtemp(2, 0) = 1.0;

    Core::LinAlg::Matrix<3, 1> result_heatflux(true);
    Core::LinAlg::Matrix<3, 3> result_cmat(true);

    fourieraniso_->evaluate(gradtemp, result_cmat, result_heatflux);

    FOUR_C_EXPECT_NEAR(result_cmat, ref_cmat, 1.0e-12);
    FOUR_C_EXPECT_NEAR(result_heatflux, ref_heatflux, 1.0e-12);
  }
}  // namespace