// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_terminal_unit.hpp"

#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_terminal_unit_elasticity.hpp"
#include "4C_reduced_lung_terminal_unit_rheology.hpp"

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace TerminalUnits
  {
    /**
     * Loop over all model blocks and assemble residual contributions.
     */
    void update_residual_vector(Core::LinAlg::Vector<double>& res_vector,
        TerminalUnitContainer& terminal_units,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (auto& model : terminal_units.models)
      {
        model.residual_evaluator(model.data, res_vector, locally_relevant_dofs, dt);
      }
    }

    /**
     * Loop over all model blocks and assemble Jacobian contributions.
     */
    void update_jacobian(Core::LinAlg::SparseMatrix& jac, TerminalUnitContainer& terminal_units,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (auto& model : terminal_units.models)
      {
        model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);
      }
    }

    /**
     * Assign contiguous local equation ids to all terminal-unit equations.
     */
    void assign_local_equation_ids(TerminalUnitContainer& terminal_units, int& n_local_equations)
    {
      for (auto& model : terminal_units.models)
      {
        model.data.local_row_id.clear();
        model.data.local_row_id.reserve(model.data.number_of_elements());
        for (size_t i = 0; i < model.data.number_of_elements(); i++)
        {
          model.data.local_row_id.push_back(n_local_equations);
          n_local_equations++;
        }
      }
    }

    /**
     * Map global dof ids to local dof ids needed for local assembly.
     */
    void assign_local_dof_ids(
        const Core::LinAlg::Map& locally_relevant_dof_map, TerminalUnitContainer& terminal_units)
    {
      for (auto& model : terminal_units.models)
      {
        model.data.lid_p1.clear();
        model.data.lid_p2.clear();
        model.data.lid_q.clear();
        model.data.lid_p1.reserve(model.data.number_of_elements());
        model.data.lid_p2.reserve(model.data.number_of_elements());
        model.data.lid_q.reserve(model.data.number_of_elements());

        for (size_t i = 0; i < model.data.number_of_elements(); i++)
        {
          model.data.lid_p1.push_back(locally_relevant_dof_map.lid(model.data.gid_p1[i]));
          model.data.lid_p2.push_back(locally_relevant_dof_map.lid(model.data.gid_p2[i]));
          model.data.lid_q.push_back(locally_relevant_dof_map.lid(model.data.gid_q[i]));
        }
      }
    }

    /**
     * Synchronize nonlinear-iteration internal states for all model blocks.
     */
    void update_internal_state_vectors(TerminalUnitContainer& terminal_units,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (auto& model : terminal_units.models)
      {
        model.internal_state_updater(model.data, locally_relevant_dofs, dt);
      }
    }

    /**
     * Execute end-of-step updates (volume + model-specific history variables).
     */
    void end_of_timestep_routine(TerminalUnitContainer& terminal_units,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
    {
      for (auto& model : terminal_units.models)
      {
        for (size_t i = 0; i < model.data.number_of_elements(); i++)
        {
          model.data.volume_v[i] +=
              locally_relevant_dofs.local_values_as_span()[model.data.lid_q[i]] * dt;
        }
        model.end_of_timestep_routine(model.data, locally_relevant_dofs, dt);
      }
    }

    /**
     * Emit terminal-unit volume output.
     */
    void append_volume_output(const TerminalUnitData& data,
        ReducedLung::RuntimeOutputCollector& collector,
        ReducedLungParameters::OutputVerbosity verbosity)
    {
      if (verbosity >= ReducedLungParameters::OutputVerbosity::medium)
      {
        auto& volume_vec = collector.get_or_create_vector("volume");
        for (size_t i = 0; i < data.number_of_elements(); ++i)
        {
          volume_vec.replace_local_value(data.local_element_id[i], data.volume_v[i]);
        }
      }
    }

    /**
     * Compose elasticity and rheology callbacks into final model evaluators.
     */
    void create_evaluators(TerminalUnitContainer& terminal_units)
    {
      for (auto& model : terminal_units.models)
      {
        auto elastic_pressure_evaluator =
            Elasticity::make_elastic_pressure_evaluator(model.elasticity_model);
        auto elastic_pressure_gradient_evaluator =
            Elasticity::make_elastic_pressure_gradient_evaluator(model.elasticity_model);

        model.residual_evaluator =
            Rheology::make_residual_evaluator(model.rheological_model, elastic_pressure_evaluator);
        model.jacobian_evaluator = Rheology::make_jacobian_evaluator(
            model.rheological_model, elastic_pressure_gradient_evaluator);
        model.internal_state_updater =
            Rheology::make_internal_state_updater(model.rheological_model);
        model.end_of_timestep_routine =
            Rheology::make_end_of_timestep_routine(model.rheological_model);
        auto elasticity_output_evaluator =
            Elasticity::make_output_evaluator(model.elasticity_model);
        auto rheology_output_evaluator = Rheology::make_output_evaluator(model.rheological_model);
        const OutputEvaluator volume_output_evaluator = append_volume_output;
        model.output_evaluator = [elasticity_output_evaluator, rheology_output_evaluator,
                                     volume_output_evaluator](const TerminalUnitData& data,
                                     ReducedLung::RuntimeOutputCollector& collector,
                                     ReducedLungParameters::OutputVerbosity verbosity)
        {
          elasticity_output_evaluator(data, collector, verbosity);
          rheology_output_evaluator(data, collector, verbosity);
          volume_output_evaluator(data, collector, verbosity);
        };
      }
    }

  }  // namespace TerminalUnits
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE
