// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_MODEL_REGISTRY_HPP
#define FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_MODEL_REGISTRY_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_terminal_unit.hpp"

#include <functional>
#include <map>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::TerminalUnits::ModelRegistry
{
  /**
   * @brief Rheology selection enum used as registry key component.
   */
  using RheologicalModelType =
      ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType;

  /**
   * @brief Elasticity selection enum used as registry key component.
   */
  using ElasticityModelType =
      ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType;

  /**
   * @brief Factory callback that appends one terminal-unit element to the proper model block.
   */
  using TerminalUnitFactory =
      std::function<void(TerminalUnitContainer& terminal_units, int global_element_id,
          int local_element_id, double ref_length, const ReducedLungParameters& parameters)>;

  /**
   * @brief Composite registry key: (rheology type, elasticity type).
   */
  using TerminalUnitModelKey = std::pair<RheologicalModelType, ElasticityModelType>;

  /**
   * @brief Mapping from model key to concrete terminal-unit element factory.
   */
  using TerminalUnitFactoryMap = std::map<TerminalUnitModelKey, TerminalUnitFactory>;

  /**
   * @brief Add a terminal-unit element by resolving the requested model key in the registry.
   *
   * @param terminal_units Container of terminal-unit model blocks.
   * @param global_element_id Global element id.
   * @param local_element_id Local element id in element row map.
   * @param ref_length Reference length of the element, i.e. the distance between its two nodes.
   * @param parameters Reduced-lung input parameters.
   * @param rheological_model_type Selected rheology model type.
   * @param elasticity_model_type Selected elasticity model type.
   */
  void add_terminal_unit_with_model_selection(TerminalUnitContainer& terminal_units,
      int global_element_id, int local_element_id, double ref_length,
      const ReducedLungParameters& parameters, RheologicalModelType rheological_model_type,
      ElasticityModelType elasticity_model_type);
}  // namespace ReducedLung::TerminalUnits::ModelRegistry

FOUR_C_NAMESPACE_CLOSE

#endif
