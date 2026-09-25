// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_resulttest.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_reduced_lung_airways.hpp"
#include "4C_reduced_lung_terminal_unit.hpp"
#include "4C_utils_exceptions.hpp"

#include <optional>
#include <string>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace
  {
    std::optional<double> airway_result(const Airways::AirwayContainer& airways,
        const Core::LinAlg::Vector<double>& dofs, int global_element_id,
        const std::string& quantity)
    {
      const auto dof_values = dofs.local_values_as_span();
      for (const auto& model : airways.models)
      {
        const auto& data = model.data;
        for (std::size_t i = 0; i < data.number_of_elements(); ++i)
        {
          if (data.global_element_id[i] != global_element_id) continue;

          if (quantity == "q_in") return dof_values[data.lid_q1[i]];
          FOUR_C_THROW("Unsupported airway result quantity '{}'.", quantity);
        }
      }
      return std::nullopt;
    }

    std::optional<double> terminal_unit_result(const TerminalUnits::TerminalUnitContainer& units,
        int global_element_id, const std::string& quantity)
    {
      for (const auto& model : units.models)
      {
        const auto& data = model.data;
        for (std::size_t i = 0; i < data.number_of_elements(); ++i)
        {
          if (data.global_element_id[i] != global_element_id) continue;

          if (quantity == "volume") return data.volume_v[i];
          FOUR_C_THROW("Unsupported terminal-unit result quantity '{}'.", quantity);
        }
      }
      return std::nullopt;
    }
  }  // namespace

  ResultTest::ResultTest(const Airways::AirwayContainer& airways,
      const TerminalUnits::TerminalUnitContainer& terminal_units,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs)
      : Core::Utils::ResultTest("REDUCED_LUNG"),
        airways_(airways),
        terminal_units_(terminal_units),
        locally_relevant_dofs_(locally_relevant_dofs)
  {
  }

  void ResultTest::test_element(
      const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
  {
    const int global_element_id = container.get<int>("ELEMENT") - 1;
    FOUR_C_ASSERT_ALWAYS(global_element_id >= 0, "Reduced-lung result element ids are one-based.");
    const std::string quantity = container.get<std::string>("QUANTITY");

    std::optional<double> result =
        airway_result(airways_, locally_relevant_dofs_, global_element_id, quantity);
    if (!result)
    {
      result = terminal_unit_result(terminal_units_, global_element_id, quantity);
    }
    if (!result) return;

    nerr += compare_values(*result, "ELEMENT", container);
    ++test_count;
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE
