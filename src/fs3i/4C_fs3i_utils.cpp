// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fs3i_utils.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"

FOUR_C_NAMESPACE_OPEN

void FS3I::Utils::set_material_pointers_matching_grid(
    const Core::FE::Discretization& sourcedis, const Core::FE::Discretization& targetdis)
{
  const int numelements = targetdis.num_my_col_elements();

  for (int i = 0; i < numelements; ++i)
  {
    Core::Elements::Element* targetele = targetdis.l_col_element(i);
    const int gid = targetele->id();

    Core::Elements::Element* sourceele = sourcedis.g_element(gid);

    // For matching-grid TFSI coupling we need the structural material on the scatra element.
    targetele->set_material(1, sourceele->material());
  }
}

FOUR_C_NAMESPACE_CLOSE
