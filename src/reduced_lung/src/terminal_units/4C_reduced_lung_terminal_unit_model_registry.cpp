// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_terminal_unit_model_registry.hpp"

#include "4C_reduced_lung_terminal_unit_elasticity.hpp"
#include "4C_reduced_lung_terminal_unit_rheology.hpp"
#include "4C_utils_exceptions.hpp"

#include <cmath>
#include <numbers>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace
  {
    using namespace TerminalUnits::ModelRegistry;

    /**
     * Compatibility whitelist for terminal-unit model pairings.
     */
    const std::vector<TerminalUnitModelKey>& compatible_terminal_unit_model_pairs()
    {
      static const std::vector<TerminalUnitModelKey> compatible_pairs = {
          {RheologicalModelType::KelvinVoigt, ElasticityModelType::Linear},
          {RheologicalModelType::KelvinVoigt, ElasticityModelType::Ogden},
          {RheologicalModelType::FourElementMaxwell, ElasticityModelType::Linear},
          {RheologicalModelType::FourElementMaxwell, ElasticityModelType::Ogden},
      };

      return compatible_pairs;
    }

    /**
     * Return existing model block for a constitutive pair or create a new one.
     */
    template <typename RheologicalModel, typename ElasticityModel>
    TerminalUnits::TerminalUnitModel& register_or_access_terminal_unit_model(
        TerminalUnits::TerminalUnitContainer& terminal_units)
    {
      for (auto& model : terminal_units.models)
      {
        if (std::holds_alternative<RheologicalModel>(model.rheological_model) &&
            std::holds_alternative<ElasticityModel>(model.elasticity_model))
        {
          return model;
        }
      }

      terminal_units.models.emplace_back();
      auto& model = terminal_units.models.back();
      model.rheological_model = RheologicalModel{};
      model.elasticity_model = ElasticityModel{};
      return model;
    }

    /**
     * Add one element to the selected terminal-unit model block and append model parameters.
     */
    template <typename RheologicalModel, typename ElasticityModel>
    void add_terminal_unit_element(TerminalUnits::TerminalUnitContainer& terminal_units,
        const int global_element_id, const int local_element_id, const double ref_length,
        const ReducedLungParameters& parameters)
    {
      auto& model =
          register_or_access_terminal_unit_model<RheologicalModel, ElasticityModel>(terminal_units);

      model.data.global_element_id.push_back(global_element_id);
      model.data.local_element_id.push_back(local_element_id);

      // The terminal unit is modelled as a sphere whose radius is the element length.
      const double radius = ref_length;
      const double volume = (4.0 / 3.0) * std::numbers::pi * radius * radius * radius;
      model.data.volume_v.push_back(volume);
      model.data.reference_volume_v0.push_back(volume);

      TerminalUnits::Rheology::append_model_parameters(model.rheological_model, global_element_id,
          parameters.lung_tree.terminal_units.rheological_model);
      TerminalUnits::Elasticity::append_model_parameters(model.elasticity_model, global_element_id,
          parameters.lung_tree.terminal_units.elasticity_model);
    }

    /**
     * Build singleton factory registry from compatibility pairs.
     */
    const TerminalUnitFactoryMap& terminal_unit_factory_registry()
    {
      static const TerminalUnitFactoryMap registry = []
      {
        TerminalUnitFactoryMap factories;

        for (const auto& [rheological_model_type, elasticity_model_type] :
            compatible_terminal_unit_model_pairs())
        {
          const auto [it, inserted] = factories.emplace(
              TerminalUnitModelKey{rheological_model_type, elasticity_model_type},
              [rheological_model_type, elasticity_model_type](
                  TerminalUnits::TerminalUnitContainer& terminal_units, int global_element_id,
                  int local_element_id, double ref_length, const ReducedLungParameters& parameters)
              {
                TerminalUnits::Rheology::dispatch_rheological_model_type(rheological_model_type,
                    [&]<typename RheologicalModel>()
                    {
                      TerminalUnits::Elasticity::dispatch_elasticity_model_type(
                          elasticity_model_type,
                          [&]<typename ElasticityModel>()
                          {
                            add_terminal_unit_element<RheologicalModel, ElasticityModel>(
                                terminal_units, global_element_id, local_element_id, ref_length,
                                parameters);
                          });
                    });
              });

          if (!inserted)
          {
            FOUR_C_THROW(
                "Duplicate terminal-unit model registration for (rheological='{}', "
                "elasticity='{}').",
                TerminalUnits::Rheology::rheological_model_name(rheological_model_type),
                TerminalUnits::Elasticity::elasticity_model_name(elasticity_model_type));
          }
          FOUR_C_ASSERT_ALWAYS(it != factories.end(), "Invalid terminal-unit registry insertion.");
        }

        return factories;
      }();

      return registry;
    }
  }  // namespace

  namespace TerminalUnits::ModelRegistry
  {
    void add_terminal_unit_with_model_selection(TerminalUnitContainer& terminal_units,
        int global_element_id, int local_element_id, double ref_length,
        const ReducedLungParameters& parameters, RheologicalModelType rheological_model_type,
        ElasticityModelType elasticity_model_type)
    {
      const TerminalUnitModelKey key{rheological_model_type, elasticity_model_type};
      const auto& registry = terminal_unit_factory_registry();
      const auto factory_it = registry.find(key);
      if (factory_it == registry.end())
      {
        FOUR_C_THROW(
            "Terminal unit model combination not implemented (rheological='{}', elasticity='{}').",
            TerminalUnits::Rheology::rheological_model_name(rheological_model_type),
            TerminalUnits::Elasticity::elasticity_model_name(elasticity_model_type));
      }

      factory_it->second(
          terminal_units, global_element_id, local_element_id, ref_length, parameters);
    }
  }  // namespace TerminalUnits::ModelRegistry
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE
