// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_tree_linearization.hpp"

#include "4C_utils_exceptions.hpp"

#include <algorithm>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  void TreeCoefficientAssemblyTarget::replace_values(std::span<const int> local_row_ids,
      std::span<const int> local_dof_ids, std::span<const double> values)
  {
    FOUR_C_ASSERT_ALWAYS(
        local_row_ids.size() == local_dof_ids.size() && local_row_ids.size() == values.size(),
        "Tree coefficient batch replacement size mismatch: rows {}, dofs {}, values {}.",
        local_row_ids.size(), local_dof_ids.size(), values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
    {
      replace_value(local_row_ids[i], local_dof_ids[i], values[i]);
    }
  }

  void TreeLinearization::reset(int num_rows, int num_dofs)
  {
    FOUR_C_ASSERT_ALWAYS(num_rows >= 0, "Tree linearization row count must be non-negative.");
    FOUR_C_ASSERT_ALWAYS(num_dofs >= 0, "Tree linearization dof count must be non-negative.");

    num_rows_ = num_rows;
    num_dofs_ = num_dofs;
    rows_.assign(static_cast<std::size_t>(num_rows_), {});
  }

  void TreeLinearization::clear_values()
  {
    for (auto& row : rows_)
    {
      row.clear();
    }
  }

  void TreeLinearization::reserve_row_entries(int local_row_id, int entry_count)
  {
    FOUR_C_ASSERT_ALWAYS(local_row_id >= 0 && local_row_id < num_rows_,
        "Tree linearization row {} is outside [0, {}).", local_row_id, num_rows_);
    FOUR_C_ASSERT_ALWAYS(entry_count >= 0,
        "Tree linearization row capacity must be non-negative, got {}.", entry_count);

    rows_[static_cast<std::size_t>(local_row_id)].reserve(static_cast<std::size_t>(entry_count));
  }

  void TreeLinearization::set_value(int local_row_id, int local_dof_id, double value)
  {
    FOUR_C_ASSERT_ALWAYS(local_row_id >= 0 && local_row_id < num_rows_,
        "Tree linearization row {} is outside [0, {}).", local_row_id, num_rows_);
    FOUR_C_ASSERT_ALWAYS(local_dof_id >= 0 && local_dof_id < num_dofs_,
        "Tree linearization dof {} is outside [0, {}).", local_dof_id, num_dofs_);

    auto& row = rows_[static_cast<std::size_t>(local_row_id)];
    const auto entry = std::find_if(row.begin(), row.end(),
        [local_dof_id](const auto& coefficient) { return coefficient.first == local_dof_id; });
    if (entry != row.end())
    {
      entry->second = value;
      return;
    }

    row.emplace_back(local_dof_id, value);
  }

  void TreeLinearization::append_value(int local_row_id, int local_dof_id, double value)
  {
    FOUR_C_ASSERT_ALWAYS(local_row_id >= 0 && local_row_id < num_rows_,
        "Tree linearization row {} is outside [0, {}).", local_row_id, num_rows_);
    FOUR_C_ASSERT_ALWAYS(local_dof_id >= 0 && local_dof_id < num_dofs_,
        "Tree linearization dof {} is outside [0, {}).", local_dof_id, num_dofs_);

    rows_[static_cast<std::size_t>(local_row_id)].emplace_back(local_dof_id, value);
  }

  void TreeLinearization::replace_value(int local_row_id, int local_dof_id, double value)
  {
    FOUR_C_ASSERT_ALWAYS(local_row_id >= 0 && local_row_id < num_rows_,
        "Tree linearization row {} is outside [0, {}).", local_row_id, num_rows_);
    FOUR_C_ASSERT_ALWAYS(local_dof_id >= 0 && local_dof_id < num_dofs_,
        "Tree linearization dof {} is outside [0, {}).", local_dof_id, num_dofs_);

    auto& row = rows_[static_cast<std::size_t>(local_row_id)];
    const auto entry = std::find_if(row.begin(), row.end(),
        [local_dof_id](const auto& coefficient) { return coefficient.first == local_dof_id; });
    FOUR_C_ASSERT_ALWAYS(entry != row.end(),
        "Tree linearization row {} does not contain dof {} for replacement.", local_row_id,
        local_dof_id);
    entry->second = value;
  }

  void TreeLinearization::replace_values(std::span<const int> local_row_ids,
      std::span<const int> local_dof_ids, std::span<const double> values)
  {
    FOUR_C_ASSERT_ALWAYS(
        local_row_ids.size() == local_dof_ids.size() && local_row_ids.size() == values.size(),
        "Tree linearization batch replacement size mismatch: rows {}, dofs {}, values {}.",
        local_row_ids.size(), local_dof_ids.size(), values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
    {
      TreeLinearization::replace_value(local_row_ids[i], local_dof_ids[i], values[i]);
    }
  }

  double TreeLinearization::value(int local_row_id, int local_dof_id) const
  {
    FOUR_C_ASSERT_ALWAYS(local_row_id >= 0 && local_row_id < num_rows_,
        "Tree linearization row {} is outside [0, {}).", local_row_id, num_rows_);
    FOUR_C_ASSERT_ALWAYS(local_dof_id >= 0 && local_dof_id < num_dofs_,
        "Tree linearization dof {} is outside [0, {}).", local_dof_id, num_dofs_);

    const auto& row = rows_[static_cast<std::size_t>(local_row_id)];
    const auto entry = std::find_if(row.begin(), row.end(),
        [local_dof_id](const auto& coefficient) { return coefficient.first == local_dof_id; });
    if (entry == row.end())
    {
      return 0.0;
    }
    return entry->second;
  }

  const std::vector<std::pair<int, double>>& TreeLinearization::entries(int local_row_id) const
  {
    FOUR_C_ASSERT_ALWAYS(local_row_id >= 0 && local_row_id < num_rows_,
        "Tree linearization row {} is outside [0, {}).", local_row_id, num_rows_);

    return rows_[static_cast<std::size_t>(local_row_id)];
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE
