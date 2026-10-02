// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_ELEMENT_CENTER_HPP
#define FOUR_C_FEM_GENERAL_ELEMENT_CENTER_HPP

#include "4C_config.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Elements
{
  class Element;
}

namespace Core::FE
{
  // return center coordinates of @p ele. Note that the average of all nodal coordinates is
  // computed.
  std::vector<double> element_center_refe_coords(const Core::Elements::Element& ele);
}  // namespace Core::FE


FOUR_C_NAMESPACE_CLOSE

#endif
