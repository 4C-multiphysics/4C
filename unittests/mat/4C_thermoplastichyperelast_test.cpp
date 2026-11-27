// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_einstein.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_thermoplastichyperelast.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_ParameterList.hpp>

#include <algorithm>
#include <cmath>
#include <memory>

namespace
{
  using namespace FourC;

  constexpr int gp = 0;
  constexpr int ele_gid = 0;
  constexpr double time_step_size = 0.1;
  /// differs from the reference temperature, so that the thermal stress and the temperature
  /// dependence of the yield stress are active
  constexpr double temperature = 330.0;

  /// state of the Gauss point
  enum class PlasticState
  {
    elastic,  //< stress below the yield stress, no plastic flow
    plastic   //< stress well above the yield stress
  };

  /// Green-Lagrange strain \f$ \boldsymbol{E} = \frac{1}{2} (\boldsymbol{F}^\mathrm{T}
  /// \boldsymbol{F} - \boldsymbol{I}) \f$
  Core::LinAlg::SymmetricTensor<double, 3, 3> green_lagrange(
      const Core::LinAlg::Tensor<double, 3, 3>& F)
  {
    return 0.5 * (Core::LinAlg::assume_symmetry(Core::LinAlg::transpose(F) * F) -
                     Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
  }

  /// largest absolute entry of @p tensor, e.g., to scale the tolerance of a comparison against
  /// finite differences, whose errors scale with the largest entry
  double max_abs(const auto& tensor)
  {
    return std::abs(std::ranges::max(
        tensor.container(), {}, [](const double value) { return std::abs(value); }));
  }

  Teuchos::ParameterList create_parameter_list(const double T)
  {
    Teuchos::ParameterList params;
    params.set("temperature", T);
    return params;
  }

  /// stress and stiffness tensor of a material evaluation
  struct StressResponse
  {
    Core::LinAlg::SymmetricTensor<double, 3, 3> stress;
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat;
  };

  /*!
   * Checks the tangents of ThermoPlasticHyperElast through its public interface in an elastic and
   * a plastic state:
   *
   * - the stiffness tensor equals the finite difference of the stress w.r.t. F
   * - the derivative of the stress w.r.t. the temperature equals its finite difference
   *
   * The history is not updated, i.e., all evaluations start from the initial state.
   */
  class ThermoPlasticHyperElastTest : public ::testing::TestWithParam<PlasticState>
  {
   protected:
    void SetUp() override
    {
      Core::IO::InputParameterContainer container;
      container.add("YOUNG", 2.1e5);
      container.add("NUE", 0.3);
      container.add("DENS", 1.0);
      container.add("CTE", 1.2e-5);
      container.add("INITTEMP", 290.0);
      container.add("YIELD", 300.0);
      container.add("ISOHARD", 100.0);
      container.add("SATHARDENING", 400.0);
      container.add("HARDEXPO", 10.0);
      container.add("YIELDSOFT", 0.002);
      container.add("HARDSOFT", 0.002);
      // tight tolerance of the return mapping, so that it does not spoil the finite differences
      container.add("TOL", 1.0e-12);

      parameters_ = std::make_shared<Mat::PAR::ThermoPlasticHyperElast>(
          Core::Mat::PAR::Parameter::Data{.parameters = container});
      material_ = std::make_shared<Mat::ThermoPlasticHyperElast>(parameters_.get());
      material_->setup(1, {}, {});

      const double scale = GetParam() == PlasticState::elastic ? 1.0e-4 : 1.0e-2;
      F_ = Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>) +
           scale * Core::LinAlg::Tensor<double, 3, 3>{
                       {{2.0, 1.5, -0.5}, {0.5, -1.5, 1.0}, {-0.5, 0.4, 3.0}}};
    }

    /// evaluate the stress and stiffness tensor at F and T
    [[nodiscard]] StressResponse evaluate(
        const Core::LinAlg::Tensor<double, 3, 3>& F, const double T)
    {
      StressResponse response{};
      material_->evaluate(&F, green_lagrange(F), create_parameter_list(T), context_,
          response.stress, response.cmat, gp, ele_gid);
      return response;
    }

    const Mat::EvaluationContext<3> context_{.total_time = &time_step_size,
        .time_step_size = &time_step_size,
        .xi = nullptr,
        .ref_coords = nullptr};
    Core::LinAlg::Tensor<double, 3, 3> F_;

    std::shared_ptr<Mat::PAR::ThermoPlasticHyperElast> parameters_;
    std::shared_ptr<Mat::ThermoPlasticHyperElast> material_;
  };

  TEST_P(ThermoPlasticHyperElastTest, CmatEqualsFiniteDifference)
  {
    // analytic: dS_ij/dF_kl = 1/2 C_ijmn dC_mn/dF_kl = C_ijml F_km
    const Core::LinAlg::Tensor<double, 3, 3, 3, 3> dSdF =
        Core::LinAlg::einsum<"ijml", "km">(evaluate(F_, temperature).cmat, F_);

    // finite difference
    constexpr double h = 1.0e-6;
    Core::LinAlg::Tensor<double, 3, 3, 3, 3> dSdF_fd{};
    for (std::size_t k = 0; k < 3; ++k)
    {
      for (std::size_t l = 0; l < 3; ++l)
      {
        Core::LinAlg::Tensor<double, 3, 3> F_plus = F_;
        Core::LinAlg::Tensor<double, 3, 3> F_minus = F_;
        F_plus(k, l) += h;
        F_minus(k, l) -= h;
        const Core::LinAlg::SymmetricTensor<double, 3, 3> dS =
            (evaluate(F_plus, temperature).stress - evaluate(F_minus, temperature).stress) /
            (2.0 * h);
        for (std::size_t i = 0; i < 3; ++i)
          for (std::size_t j = 0; j < 3; ++j) dSdF_fd(i, j, k, l) = dS(i, j);
      }
    }

    ASSERT_GT(max_abs(dSdF_fd), 0.0) << "the finite difference reference vanishes";
    FOUR_C_EXPECT_NEAR(dSdF, dSdF_fd, 1.0e-6 * max_abs(dSdF_fd));
  }

  TEST_P(ThermoPlasticHyperElastTest, DStressDTemperatureEqualsFiniteDifference)
  {
    // analytic, the plastic part is computed in evaluate()
    (void)evaluate(F_, temperature);
    const Core::LinAlg::SymmetricTensor<double, 3, 3> dSdT = material_->evaluate_d_stress_d_scalar(
        F_, green_lagrange(F_), create_parameter_list(temperature), context_, gp, ele_gid);

    // finite difference
    constexpr double h = 1.0e-3;
    const Core::LinAlg::SymmetricTensor<double, 3, 3> dSdT_fd =
        (evaluate(F_, temperature + h).stress - evaluate(F_, temperature - h).stress) / (2.0 * h);

    ASSERT_GT(max_abs(dSdT_fd), 0.0) << "the finite difference reference vanishes";
    FOUR_C_EXPECT_NEAR(dSdT, dSdT_fd, 1.0e-6 * max_abs(dSdT_fd));
  }

  INSTANTIATE_TEST_SUITE_P(PlasticStates, ThermoPlasticHyperElastTest,
      ::testing::ValuesIn(EnumTools::enum_values<PlasticState>()),
      [](const auto& info) { return std::string(EnumTools::enum_name(info.param)); });
}  // namespace
