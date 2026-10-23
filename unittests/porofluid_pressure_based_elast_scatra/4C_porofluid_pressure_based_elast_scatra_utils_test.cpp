// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_porofluid_pressure_based_elast_scatra_utils.hpp"

#include "4C_unittest_utils_assertions_test.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace
{
  using namespace FourC;

  class TestCalculationOxyPartialPressure : public ::testing::Test
  {
   protected:
    /**
     * helper function to get oxygen concentration from oxygen partial pressure, this is the known
     * forward relation
     *
     * @param Pb [in]: oxygen partial pressure
     * @param CaO2_max [in]: maximum oxygen concentration
     * @param Pb50 [in]: partial pressure at 50% maximum oxygen concentration
     * @param n [in]: exponent in Hill equation
     * @param alpha_eff [in]: effective solubility of oxygen in blood
     * @return CaO2: oxygen concentration
     */
    double get_oxy_concentration_from_partial_pressure(const double& Pb, const double& CaO2_max,
        const double& Pb50, const double& n, const double& alpha_eff)
    {
      // Hill equation for saturation
      const double saturation_hill = std::pow(Pb, n) / (std::pow(Pb, n) + std::pow(Pb50, n));
      // oxygen dissolved in plasma + bound to hemoglobin
      const double CaO2 = alpha_eff * Pb + CaO2_max * saturation_hill;

      return CaO2;
    }
  };

  //! test corner value Pb = 0 mmHg
  TEST_F(TestCalculationOxyPartialPressure, Pb0)
  {
    // Parameters
    const double n = 3.0;
    const double Pb50 = 3.6;
    const double alpha_eff = 0.00021;
    const double CaO2_max = 0.25;


    // get oxygen concentration for this partial pressure
    const double Pb = 0.0;
    const double CaO2 =
        get_oxy_concentration_from_partial_pressure(Pb, CaO2_max, Pb50, n, alpha_eff);

    // invert to get partial pressure for this concentration
    double inverted_Pb = 0.0;
    PoroPressureBased::get_oxy_partial_pressure_from_concentration<double>(
        inverted_Pb, CaO2, CaO2_max, Pb50, n, alpha_eff);

    // check if partial pressure and numerically inverted partial pressure are equal
    EXPECT_NEAR(0.0, inverted_Pb, 1e-14);
  }

  //! test value Pb = 25 mmHg
  TEST_F(TestCalculationOxyPartialPressure, Pb25)
  {
    // Parameters
    const double n = 2.5;
    const double Pb50 = 45.0;
    const double alpha_eff = 3.1e-5;
    const double CaO2_max = 0.225;
    const double Pb = 25.0;

    // get oxygen concentration for this partial pressure
    const double CaO2 =
        get_oxy_concentration_from_partial_pressure(Pb, CaO2_max, Pb50, n, alpha_eff);

    // invert to get partial pressure for this concentration
    double inverted_Pb = 0.0;
    PoroPressureBased::get_oxy_partial_pressure_from_concentration<double>(
        inverted_Pb, CaO2, CaO2_max, Pb50, n, alpha_eff);

    // check if partial pressure and numerically inverted partial pressure are equal
    EXPECT_NEAR(Pb, inverted_Pb, 1e-14);
  }

  //! test value Pb = 50 mmHg
  TEST_F(TestCalculationOxyPartialPressure, Pb50)
  {
    // Parameters
    const double n = 2.7;
    const double Pb50 = 37.0;
    const double alpha_eff = 3.1e-5;
    const double CaO2_max = 0.225;
    const double Pb = 50.0;

    // get oxygen concentration for this partial pressure
    const double CaO2 =
        get_oxy_concentration_from_partial_pressure(Pb, CaO2_max, Pb50, n, alpha_eff);

    // invert to get partial pressure for this concentration
    double inverted_Pb = 0.0;
    PoroPressureBased::get_oxy_partial_pressure_from_concentration<double>(
        inverted_Pb, CaO2, CaO2_max, Pb50, n, alpha_eff);

    // check if partial pressure and numerically inverted partial pressure are equal
    EXPECT_NEAR(Pb, inverted_Pb, 1e-14);
  }

  //! test value Pb = 100 mmHg
  TEST_F(TestCalculationOxyPartialPressure, Pb100)
  {
    // Parameters
    const double n = 2.2;
    const double Pb50 = 42.0;
    const double alpha_eff = 3.7e-5;
    const double CaO2_max = 0.225;
    const double Pb = 100.0;

    // get oxygen concentration for this partial pressure
    const double CaO2 =
        get_oxy_concentration_from_partial_pressure(Pb, CaO2_max, Pb50, n, alpha_eff);

    // invert to get partial pressure for this concentration
    double inverted_Pb = 0.0;
    PoroPressureBased::get_oxy_partial_pressure_from_concentration<double>(
        inverted_Pb, CaO2, CaO2_max, Pb50, n, alpha_eff);

    // check if partial pressure and numerically inverted partial pressure are equal
    EXPECT_NEAR(Pb, inverted_Pb, 1e-14 * Pb);  // relative tolerance
  }

  //! test native double and Sacado values/derivatives against the forward relation for several
  //! pressures and exponents
  TEST_F(TestCalculationOxyPartialPressure, ForwardRelationDoubleAndFad)
  {
    using FAD = Sacado::Fad::DFad<double>;

    const double Pb50 = 3.6;
    const double alpha_eff = 0.00021;
    const double CaO2_max = 0.25;

    for (const double n : {1.0, 2.2, 3.0, 4.0})
    {
      for (const double Pb : {0.0, 1.0, 3.6, 10.0, 20.0, 100.0, 1000.0})
      {
        const std::string label = "n = " + std::to_string(n) + ", Pb = " + std::to_string(Pb);
        const double CaO2 =
            get_oxy_concentration_from_partial_pressure(Pb, CaO2_max, Pb50, n, alpha_eff);
        const double tol = 1.0e-12 * std::max(1.0, Pb);

        // native double
        double inverted_Pb = -1.0;
        PoroPressureBased::get_oxy_partial_pressure_from_concentration<double>(
            inverted_Pb, CaO2, CaO2_max, Pb50, n, alpha_eff);
        EXPECT_NEAR(Pb, inverted_Pb, tol) << "double, " << label;
        EXPECT_GE(inverted_Pb, 0.0) << label;

        // Sacado: value and derivative dPb/dCaO2 = 1 / (dCaO2/dPb) of the forward relation
        FAD CaO2_fad(1, 0, CaO2);
        FAD inverted_Pb_fad = 0.0;
        PoroPressureBased::get_oxy_partial_pressure_from_concentration<FAD>(
            inverted_Pb_fad, CaO2_fad, CaO2_max, Pb50, n, alpha_eff);
        EXPECT_NEAR(Pb, inverted_Pb_fad.val(), tol) << "FAD value, " << label;

        const double Pb_n = std::pow(Pb, n);
        const double Pb50_n = std::pow(Pb50, n);
        double dCdP = alpha_eff;
        if (Pb > 0.0)
          dCdP += CaO2_max * n * Pb_n / Pb * Pb50_n / ((Pb_n + Pb50_n) * (Pb_n + Pb50_n));
        else if (n == 1.0)
          dCdP += CaO2_max / Pb50;
        EXPECT_NEAR(1.0 / dCdP, inverted_Pb_fad.dx(0), 1.0e-12 / dCdP)
            << "FAD derivative, " << label;
      }
    }
  }

  //! test concentration above the Hill plateau (see issue
  //! https://github.com/4C-multiphysics/4C/issues/2354)
  TEST_F(TestCalculationOxyPartialPressure, ConcentrationAbovePlateau)
  {
    const double n = 3.0;
    const double Pb50 = 3.6;
    const double alpha_eff = 0.00021;
    const double CaO2_max = 0.25;
    const double CaO2 = 0.26;

    double inverted_Pb = -1.0;
    PoroPressureBased::get_oxy_partial_pressure_from_concentration<double>(
        inverted_Pb, CaO2, CaO2_max, Pb50, n, alpha_eff);

    EXPECT_GT(inverted_Pb, 0.0);
    EXPECT_NEAR(CaO2,
        get_oxy_concentration_from_partial_pressure(inverted_Pb, CaO2_max, Pb50, n, alpha_eff),
        1.0e-12);
  }

  //! test that invalid states are rejected
  TEST_F(TestCalculationOxyPartialPressure, InvalidInput)
  {
    double Pb = 0.0;
    // negative concentration
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        PoroPressureBased::get_oxy_partial_pressure_from_concentration<double>(
            Pb, -0.1, 0.25, 3.6, 3.0, 0.00021),
        Core::Exception, "invalid oxygen concentration");
    // non-finite concentration
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        PoroPressureBased::get_oxy_partial_pressure_from_concentration<double>(
            Pb, std::numeric_limits<double>::quiet_NaN(), 0.25, 3.6, 3.0, 0.00021),
        Core::Exception, "invalid oxygen concentration");
    // non-positive maximum concentration
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        PoroPressureBased::get_oxy_partial_pressure_from_concentration<double>(
            Pb, 0.1, 0.0, 3.6, 3.0, 0.00021),
        Core::Exception, "CaO2_max and Pb50 must be positive");
    // non-positive P50
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        PoroPressureBased::get_oxy_partial_pressure_from_concentration<double>(
            Pb, 0.1, 0.25, 0.0, 3.0, 0.00021),
        Core::Exception, "CaO2_max and Pb50 must be positive");
  }

}  // namespace