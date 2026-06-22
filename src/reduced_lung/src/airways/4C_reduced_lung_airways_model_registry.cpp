// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_airways_model_registry.hpp"

#include "4C_utils_exceptions.hpp"

#include <cmath>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace
  {
    using namespace Airways::ModelRegistry;

    const std::vector<AirwayModelKey>& compatible_airway_model_pairs()
    {
      static const std::vector<AirwayModelKey> compatible_pairs = {
          {FlowModelType::Linear, WallModelType::Rigid},
          {FlowModelType::Linear, WallModelType::KelvinVoigt},
          {FlowModelType::NonLinear, WallModelType::Rigid},
          {FlowModelType::NonLinear, WallModelType::KelvinVoigt},
      };

      return compatible_pairs;
    }

    template <typename FlowModelT, typename WallModelT>
    Airways::AirwayModel& register_or_access_airway_model(Airways::AirwayContainer& airways)
    {
      for (auto& model : airways.models)
      {
        if (std::holds_alternative<FlowModelT>(model.flow_model) &&
            std::holds_alternative<WallModelT>(model.wall_model))
        {
          return model;
        }
      }

      airways.models.emplace_back();
      auto& model = airways.models.back();
      model.flow_model = FlowModelT{};
      model.wall_model = WallModelT{};
      model.data.n_state_equations =
          Airways::FlowResistance::FlowModelStateCount<FlowModelT>::value +
          Airways::WallMechanics::WallModelStateCount<WallModelT>::value;

      return model;
    }

    template <typename FlowModelT, typename WallModelT>
    void add_airway_element(Airways::AirwayContainer& airways, const int global_element_id,
        const int local_element_id, const ReducedLungParameters& parameters)
    {
      auto& model = register_or_access_airway_model<FlowModelT, WallModelT>(airways);

      model.data.global_element_id.push_back(global_element_id);
      model.data.local_element_id.push_back(local_element_id);

      const auto& node_ids =
          parameters.lung_tree.topology.element_nodes.at(global_element_id, "element_nodes");
      const int node_in = node_ids[0] - 1;
      const int node_out = node_ids[1] - 1;
      const auto& coords_node_1 =
          parameters.lung_tree.topology.node_coordinates.at(node_in, "node_coordinates");
      const auto& coords_node_2 =
          parameters.lung_tree.topology.node_coordinates.at(node_out, "node_coordinates");
      const double length =
          std::sqrt((coords_node_1[0] - coords_node_2[0]) * (coords_node_1[0] - coords_node_2[0]) +
                    (coords_node_1[1] - coords_node_2[1]) * (coords_node_1[1] - coords_node_2[1]) +
                    (coords_node_1[2] - coords_node_2[2]) * (coords_node_1[2] - coords_node_2[2]));
      model.data.ref_length.push_back(length);

      const double radius = parameters.lung_tree.airways.radius.at(global_element_id, "radius");
      const double area = radius * radius * M_PI;
      model.data.air_properties.dynamic_viscosity = parameters.air_properties.dynamic_viscosity;
      model.data.air_properties.density = parameters.air_properties.density;
      model.data.ref_area.push_back(area);

      model.data.q1_n.push_back(0.0);
      model.data.q2_n.push_back(0.0);
      model.data.p1_n.push_back(0.0);
      model.data.p2_n.push_back(0.0);

      Airways::FlowResistance::append_model_parameters(
          model.flow_model, global_element_id, parameters.lung_tree.airways.flow_model);
      Airways::WallMechanics::append_model_parameters(
          model.wall_model, global_element_id, parameters.lung_tree.airways.wall_model, area);
    }

    const AirwayFactoryMap& airway_factory_registry()
    {
      static const AirwayFactoryMap registry = []
      {
        AirwayFactoryMap factories;

        for (const auto& compatible_pair : compatible_airway_model_pairs())
        {
          const FlowModelType flow_model_type = compatible_pair.first;
          const WallModelType wall_model_type = compatible_pair.second;
          const auto [it, inserted] = factories.emplace(
              AirwayModelKey{flow_model_type, wall_model_type},
              [flow_model_type, wall_model_type](Airways::AirwayContainer& airways,
                  int global_element_id, int local_element_id,
                  const ReducedLungParameters& parameters) -> int
              {
                int n_state_equations = 0;
                Airways::FlowResistance::dispatch_flow_model_type(flow_model_type,
                    [&]<typename FlowModelT>()
                    {
                      Airways::WallMechanics::dispatch_wall_model_type(wall_model_type,
                          [&]<typename WallModelT>()
                          {
                            add_airway_element<FlowModelT, WallModelT>(
                                airways, global_element_id, local_element_id, parameters);
                            n_state_equations =
                                Airways::FlowResistance::FlowModelStateCount<FlowModelT>::value +
                                Airways::WallMechanics::WallModelStateCount<WallModelT>::value;
                          });
                    });
                return n_state_equations;
              });

          if (!inserted)
          {
            FOUR_C_THROW("Duplicate airway model registration for (flow='{}', wall='{}').",
                Airways::FlowResistance::flow_model_name(flow_model_type),
                Airways::WallMechanics::wall_model_name(wall_model_type));
          }
          FOUR_C_ASSERT_ALWAYS(it != factories.end(), "Invalid airway registry insertion.");
        }

        return factories;
      }();

      return registry;
    }
  }  // namespace

  namespace Airways::ModelRegistry
  {
    int add_airway_with_model_selection(AirwayContainer& airways, int global_element_id,
        int local_element_id, const ReducedLungParameters& parameters,
        FlowModelType flow_model_type, WallModelType wall_model_type)
    {
      const AirwayModelKey key{flow_model_type, wall_model_type};
      const auto& registry = airway_factory_registry();
      const auto factory_it = registry.find(key);
      if (factory_it == registry.end())
      {
        FOUR_C_THROW("Airway model combination not implemented (flow='{}', wall='{}').",
            Airways::FlowResistance::flow_model_name(flow_model_type),
            Airways::WallMechanics::wall_model_name(wall_model_type));
      }

      return factory_it->second(airways, global_element_id, local_element_id, parameters);
    }
  }  // namespace Airways::ModelRegistry
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE
