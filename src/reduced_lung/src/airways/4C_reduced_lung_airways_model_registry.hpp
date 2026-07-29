// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_AIRWAYS_MODEL_REGISTRY_HPP
#define FOUR_C_REDUCED_LUNG_AIRWAYS_MODEL_REGISTRY_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_airways.hpp"

#include <functional>
#include <map>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::Airways::ModelRegistry
{
  /**
   * @brief Flow model selection enum used as registry-key component.
   */
  using FlowModelType = FlowResistance::FlowModelType;

  /**
   * @brief Wall model selection enum used as registry-key component.
   */
  using WallModelType = WallMechanics::WallModelType;

  /**
   * @brief Factory callback adding one airway element to its model block.
   *
   * @return Number of state equations for the selected model pair.
   */
  using AirwayFactory = std::function<int(AirwayContainer& airways, int global_element_id,
      int local_element_id, double ref_length, const ReducedLungParameters& parameters)>;

  /**
   * @brief Composite registry key: (flow model type, wall model type).
   */
  using AirwayModelKey = std::pair<FlowModelType, WallModelType>;

  /**
   * @brief Mapping from model key to concrete airway element factory.
   */
  using AirwayFactoryMap = std::map<AirwayModelKey, AirwayFactory>;

  /**
   * @brief Add one airway element by resolving the selected model pair in the registry.
   *
   * @param ref_length Reference length of the element, i.e. the distance between its two nodes.
   *
   * @return Number of state equations for the selected model pair.
   */
  int add_airway_with_model_selection(AirwayContainer& airways, int global_element_id,
      int local_element_id, double ref_length, const ReducedLungParameters& parameters,
      FlowModelType flow_model_type, WallModelType wall_model_type);
}  // namespace ReducedLung::Airways::ModelRegistry

FOUR_C_NAMESPACE_CLOSE

#endif
