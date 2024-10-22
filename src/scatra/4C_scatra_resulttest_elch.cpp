// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_resulttest_elch.hpp"

#include "4C_scatra_timint_elch.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                               fang 03/15 |
 *----------------------------------------------------------------------*/
ScaTra::ElchResultTest::ElchResultTest(Teuchos::RCP<ScaTraTimIntElch> elchtimint)
    : ScaTraResultTest::ScaTraResultTest(elchtimint)
{
  return;
}


/*----------------------------------------------------------------------*
 | get special result to be tested                           fang 03/15 |
 *----------------------------------------------------------------------*/
double ScaTra::ElchResultTest::result_special(const std::string quantity) const
{
  // initialize variable for result
  double result(0.);

  if (quantity == "meanc" or quantity == "meanc1" or quantity == "meanc2")
  {
    auto it = elch_tim_int()->electrode_conc().begin();
    if (quantity == "meanc2") ++it;
    result = it->second;
  }
  else if (quantity == "meaneta" or quantity == "meaneta1" or quantity == "meaneta2")
  {
    auto it = elch_tim_int()->electrode_eta().begin();
    if (quantity == "meaneta2") ++it;
    result = it->second;
  }
  else if (quantity == "meancur" or quantity == "meancur1" or quantity == "meancur2")
  {
    auto it = elch_tim_int()->electrode_curr().begin();
    if (quantity == "meancur2") ++it;
    result = it->second;
  }
  else if (quantity == "soc" or quantity == "soc1" or quantity == "soc2")
  {
    auto it = elch_tim_int()->electrode_soc().begin();
    if (quantity == "soc2") ++it;
    result = it->second;
  }
  else if (quantity == "c-rate" or quantity == "c-rate1" or quantity == "c-rate2")
  {
    auto it = elch_tim_int()->electrode_c_rates().begin();
    if (quantity == "c-rate2") ++it;
    result = it->second;
  }
  else if (quantity == "cellvoltage")
    result = elch_tim_int()->cell_voltage();
  else if (quantity == "temperature")
    result = elch_tim_int()->get_current_temperature();
  else
    result = ScaTraResultTest::result_special(quantity);

  return result;
}  // ScaTra::ElchResultTest::result_special

FOUR_C_NAMESPACE_CLOSE
