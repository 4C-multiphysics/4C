// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_RESULTTEST_HPP
#define FOUR_C_REDUCED_LUNG_RESULTTEST_HPP

#include "4C_config.hpp"

#include "4C_utils_result_test.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  template <typename T>
  class Vector;
}

namespace ReducedLung::Airways
{
  struct AirwayContainer;
}

namespace ReducedLung::TerminalUnits
{
  struct TerminalUnitContainer;
}

namespace ReducedLung
{
  /**
   * @brief Numerical result checks for the final reduced-lung state.
   */
  class ResultTest : public Core::Utils::ResultTest
  {
   public:
    ResultTest(const Airways::AirwayContainer& airways,
        const TerminalUnits::TerminalUnitContainer& terminal_units,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs);

    void test_element(
        const Core::IO::InputParameterContainer& container, int& nerr, int& test_count) override;

   private:
    const Airways::AirwayContainer& airways_;
    const TerminalUnits::TerminalUnitContainer& terminal_units_;
    const Core::LinAlg::Vector<double>& locally_relevant_dofs_;
  };
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
