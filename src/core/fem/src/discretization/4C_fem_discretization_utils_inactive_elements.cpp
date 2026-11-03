// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <array>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  // Collect whether the row elements of the discretization are active in a vector based on the
  // element row map (1: active, 0: inactive)
  Core::LinAlg::Vector<int> active_flags_of_row_elements(
      const Core::FE::Discretization& discretization)
  {
    Core::LinAlg::Vector<int> active_flags(*discretization.element_row_map(), false);
    std::span<int> values = active_flags.get_local_values();
    for (int lid = 0; lid < discretization.num_my_row_elements(); ++lid)
      values[lid] = discretization.l_row_element(lid)->is_active() ? 1 : 0;

    return active_flags;
  }

  // Set whether the elements in the map of the given vector are active
  void set_active_flags(
      Core::FE::Discretization& discretization, const Core::LinAlg::Vector<int>& active_flags)
  {
    const Core::LinAlg::Map& map = active_flags.get_map();
    std::span<const int> values = active_flags.get_local_values();
    for (int lid = 0; lid < map.num_my_elements(); ++lid)
    {
      FOUR_C_ASSERT_ALWAYS(values[lid] == 0 or values[lid] == 1,
          "No active flag available for element {} of discretization '{}'.", map.gid(lid),
          discretization.name());
      discretization.g_element(map.gid(lid))->set_active(values[lid] == 1);
    }
  }
}  // namespace

void Core::FE::Discretization::deactivate_elements(std::span<const int> row_element_ids)
{
  FOUR_C_ASSERT_ALWAYS(filled(), "fill_complete() was not called");

  for (const int gid : row_element_ids)
  {
    FOUR_C_ASSERT(element_row_map()->my_gid(gid),
        "Element {} is not a row element of discretization '{}'.", gid, name());
    g_element(gid)->set_active(false);
  }

  // make the active flags of the ghosted elements consistent with the owned ones
  Core::LinAlg::Vector<int> col_active_flags(*element_col_map(), false);
  Core::LinAlg::export_to(active_flags_of_row_elements(*this), col_active_flags);
  set_active_flags(*this, col_active_flags);
}

int Core::FE::copy_element_active_flags(const Discretization& source, Discretization& target)
{
  FOUR_C_ASSERT_ALWAYS(source.filled(), "fill_complete() was not called on '{}'", source.name());
  FOUR_C_ASSERT_ALWAYS(target.filled(), "fill_complete() was not called on '{}'", target.name());

  // the active flags are exported to the owned and ghosted elements of the target at once, elements
  // of the target which are not available in the source keep the marker -1
  Core::LinAlg::Vector<int> target_active_flags(*target.element_col_map(), false);
  target_active_flags.put_value(-1);
  Core::LinAlg::export_to(active_flags_of_row_elements(source), target_active_flags);

  // Count deactivated row elements once, but reject reactivation of owned or ghosted elements.
  std::array<int, 3> local_counts{};
  for (auto ele : target.my_col_element_range())
  {
    const bool was_active = ele.user_element()->is_active();
    const int source_flag = target_active_flags.get_local_values()[ele.local_id()];
    if (source_flag == 0 and was_active and target.element_row_map()->my_gid(ele.global_id()))
      ++local_counts[0];
    if (source_flag == 1 and not was_active) ++local_counts[1];
    if (source_flag != 0 and source_flag != 1) ++local_counts[2];
  }

  const auto [num_deactivated, num_reactivated_copies, num_missing_copies] =
      Core::Communication::sum_all(local_counts, target.get_comm());
  FOUR_C_ASSERT_ALWAYS(num_missing_copies == 0,
      "No source active flag available for {} element copy/copies of discretization '{}'.",
      num_missing_copies, target.name());
  FOUR_C_ASSERT_ALWAYS(num_reactivated_copies == 0,
      "Cannot reactivate {} element copy/copies of discretization '{}' while copying active flags.",
      num_reactivated_copies, target.name());

  set_active_flags(target, target_active_flags);
  return num_deactivated;
}

Core::LinAlg::Map Core::FE::dofs_without_active_elements(
    const Discretization& discretization, const unsigned nds)
{
  FOUR_C_ASSERT_ALWAYS(discretization.filled(), "fill_complete() was not called");
  FOUR_C_ASSERT_ALWAYS(discretization.have_dofs(), "assign_degrees_of_freedom() was not called");

  const Core::LinAlg::Map& dof_row_map = *discretization.dof_row_map(nds);

  std::vector<int> dofs;
  for (auto node : discretization.my_row_node_range())
  {
    const auto adjacent_elements = node.adjacent_elements();
    if (adjacent_elements.begin() == adjacent_elements.end()) continue;

    // all adjacent elements of a row node are available in the element column map
    const bool has_active_element = std::ranges::any_of(
        adjacent_elements, [](auto element) { return element.user_element()->is_active(); });
    if (has_active_element) continue;

    for (const int dof : discretization.dof(nds, node))
      if (dof_row_map.my_gid(dof)) dofs.push_back(dof);
  }

  return Core::LinAlg::Map(
      -1, static_cast<int>(dofs.size()), dofs.data(), 0, discretization.get_comm());
}

FOUR_C_NAMESPACE_CLOSE
