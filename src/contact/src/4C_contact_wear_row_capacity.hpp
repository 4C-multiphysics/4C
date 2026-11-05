// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_WEAR_ROW_CAPACITY_HPP
#define FOUR_C_CONTACT_WEAR_ROW_CAPACITY_HPP

#include "4C_config.hpp"

#include <map>
#include <set>

FOUR_C_NAMESPACE_OPEN

namespace Wear
{
  namespace Internal
  {
    /// Collect all columns that gap and implicit-wear derivatives may insert into an S row.
    inline std::set<int> s_derivative_columns(
        const std::map<int, double>& gap, const std::map<int, double>& wear)
    {
      std::set<int> columns;
      for (const auto& [column, value] : gap) columns.insert(column);
      for (const auto& [column, value] : wear) columns.insert(column);
      return columns;
    }
  }  // namespace Internal
}  // namespace Wear

FOUR_C_NAMESPACE_CLOSE

#endif
