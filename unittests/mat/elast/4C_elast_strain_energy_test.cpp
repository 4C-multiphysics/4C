// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_mat_elast_coupblatzko.hpp"
#include "4C_mat_elast_volpow.hpp"
#include "4C_material_parameter_base.hpp"

#include <cmath>

namespace
{
  using namespace FourC;

  double evaluate_blatz_ko_energy(
      Mat::Elastic::CoupBlatzKo& summand, Core::LinAlg::Matrix<3, 1> prinv)
  {
    const Core::LinAlg::Matrix<3, 1> modinv(Core::LinAlg::Initialization::zero);
    const Core::LinAlg::SymmetricTensor<double, 3, 3> glstrain{};
    double psi = 0.0;
    summand.add_strain_energy(psi, prinv, modinv, glstrain, 0, 0);
    return psi;
  }

  TEST(ElasticStrainEnergyTest, BlatzKoEnergyIsConsistentWithPrincipalDerivative)
  {
    Core::IO::InputParameterContainer parameters;
    parameters.add("MUE", 2.3);
    parameters.add("NUE", 0.2);
    parameters.add("F", 0.35);
    Mat::Elastic::PAR::CoupBlatzKo material_parameters(
        Core::Mat::PAR::Parameter::Data{.parameters = parameters});
    Mat::Elastic::CoupBlatzKo summand(&material_parameters);

    Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::zero);
    prinv(0) = 4.2;
    prinv(1) = 5.1;
    prinv(2) = 1.44;

    Core::LinAlg::Matrix<3, 1> dpsi(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> ddpsi(Core::LinAlg::Initialization::zero);
    summand.add_derivatives_principal(dpsi, ddpsi, prinv, 0, 0);

    constexpr double step = 1.0e-6;
    auto prinv_minus = prinv;
    auto prinv_plus = prinv;
    prinv_minus(2) -= step;
    prinv_plus(2) += step;
    const double numerical_derivative = (evaluate_blatz_ko_energy(summand, prinv_plus) -
                                            evaluate_blatz_ko_energy(summand, prinv_minus)) /
                                        (2.0 * step);

    EXPECT_NEAR(numerical_derivative, dpsi(2), 1.0e-9);
  }

  TEST(ElasticStrainEnergyTest, BlatzKoEnergyUsesLogarithmicLimitAtZeroPoissonRatio)
  {
    Core::IO::InputParameterContainer parameters;
    parameters.add("MUE", 2.3);
    parameters.add("NUE", 0.0);
    parameters.add("F", 0.35);
    Mat::Elastic::PAR::CoupBlatzKo material_parameters(
        Core::Mat::PAR::Parameter::Data{.parameters = parameters});
    Mat::Elastic::CoupBlatzKo summand(&material_parameters);

    Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::zero);
    prinv(0) = 4.2;
    prinv(1) = 5.1;
    prinv(2) = 1.44;

    const double expected = 0.5 * 2.3 *
                            (0.35 * (prinv(0) - 3.0 - std::log(prinv(2))) +
                                0.65 * (prinv(1) / prinv(2) - 3.0 + std::log(prinv(2))));

    EXPECT_NEAR(evaluate_blatz_ko_energy(summand, prinv), expected, 1.0e-14);
  }

  TEST(ElasticStrainEnergyTest, VolPowEnergyIsConsistentWithFirstDerivative)
  {
    Core::IO::InputParameterContainer parameters;
    parameters.add("A", 2.5);
    parameters.add("EXPON", 4.0);
    Mat::Elastic::PAR::VolPow material_parameters(
        Core::Mat::PAR::Parameter::Data{.parameters = parameters});
    Mat::Elastic::VolPow summand(&material_parameters);

    const Core::LinAlg::Matrix<3, 1> prinv(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1> modinv(Core::LinAlg::Initialization::zero);
    modinv(2) = 1.2;
    const Core::LinAlg::SymmetricTensor<double, 3, 3> glstrain{};

    Core::LinAlg::Matrix<3, 1> dpsi(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<6, 1> ddpsi(Core::LinAlg::Initialization::zero);
    summand.add_derivatives_modified(dpsi, ddpsi, modinv, 0, 0);

    const auto evaluate_energy = [&](const double j)
    {
      auto evaluation_modinv = modinv;
      evaluation_modinv(2) = j;
      double psi = 0.0;
      summand.add_strain_energy(psi, prinv, evaluation_modinv, glstrain, 0, 0);
      return psi;
    };

    constexpr double step = 1.0e-6;
    const double numerical_derivative =
        (evaluate_energy(modinv(2) + step) - evaluate_energy(modinv(2) - step)) / (2.0 * step);

    EXPECT_NEAR(numerical_derivative, dpsi(2), 1.0e-9);
  }
}  // namespace
