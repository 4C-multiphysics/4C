// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FS3I_UTILS_HPP
#define FOUR_C_FS3I_UTILS_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace FS3I::Utils
{
  //! Reattach the structural material on matching-grid TFSI scatra elements.
  void set_material_pointers_matching_grid(
      const Core::FE::Discretization& sourcedis, const Core::FE::Discretization& targetdis);
}  // namespace FS3I::Utils

FOUR_C_NAMESPACE_CLOSE

#endif
