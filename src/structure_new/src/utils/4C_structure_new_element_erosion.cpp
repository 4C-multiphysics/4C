// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_element_erosion.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_element_center.hpp"
#include "4C_solid_ele.hpp"
#include "4C_solid_scatra_ele.hpp"
#include "4C_structure_new_input.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

Solid::ErosionResult Solid::erode_elements(Core::FE::Discretization& discretization,
    const ElementErosionParameters& parameters,
    const Core::Utils::FunctionOfSpaceTime* erosion_function, const double time)
{
  const auto reached_threshold = [&](const Core::Elements::Element& element)
  {
    return erosion_function != nullptr and
           erosion_function->evaluate(Core::FE::element_center_refe_coords(element), time, 0) >=
               parameters.threshold;
  };
  const auto has_failed = [&](const Core::Elements::Element& element)
  {
    if (not parameters.deactivate_failed_elements) return false;
    if (const auto* solid = dynamic_cast<const Discret::Elements::Solid<2>*>(&element))
      return solid->material_failed();
    if (const auto* solid = dynamic_cast<const Discret::Elements::Solid<3>*>(&element))
      return solid->material_failed();
    if (const auto* solid = dynamic_cast<const Discret::Elements::SolidScatra<2>*>(&element))
      return solid->material_failed();
    if (const auto* solid = dynamic_cast<const Discret::Elements::SolidScatra<3>*>(&element))
      return solid->material_failed();
    return false;
  };

  // inactive elements stay inactive
  ErosionResult change;
  for (auto ele : discretization.my_row_element_range())
  {
    const Core::Elements::Element& element = *ele.user_element();
    if (element.is_active() and (reached_threshold(element) or has_failed(element)))
      change.row_element_ids.push_back(ele.global_id());
  }

  change.global_count = Core::Communication::sum_all(
      static_cast<int>(change.row_element_ids.size()), discretization.get_comm());

  if (change.global_count > 0) discretization.deactivate_elements(change.row_element_ids);

  return change;
}

FOUR_C_NAMESPACE_CLOSE
