// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_reduced_lung_airways.hpp"

#include "4C_reduced_lung_airways_flow_resistance.hpp"
#include "4C_reduced_lung_airways_wall_mechanics.hpp"

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::Airways
{
  void update_jacobian(Core::LinAlg::SparseMatrix& jac, AirwayContainer& airways,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
  {
    for (auto& model : airways.models)
    {
      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);
    }
  }

  void update_residual_vector(Core::LinAlg::Vector<double>& res_vector, AirwayContainer& airways,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
  {
    for (auto& model : airways.models)
    {
      model.residual_evaluator(model.data, res_vector, locally_relevant_dofs, dt);
    }
  }

  void assign_local_equation_ids(AirwayContainer& airways, int& n_local_equations)
  {
    for (auto& model : airways.models)
    {
      model.data.local_row_id.clear();
      model.data.local_row_id.reserve(model.data.number_of_elements());
      for (size_t i = 0; i < model.data.number_of_elements(); i++)
      {
        model.data.local_row_id.push_back(n_local_equations);
        n_local_equations += model.data.n_state_equations;
      }
    }
  }

  void assign_local_dof_ids(
      const Core::LinAlg::Map& locally_relevant_dof_map, AirwayContainer& airways)
  {
    for (auto& model : airways.models)
    {
      model.data.lid_p1.clear();
      model.data.lid_p2.clear();
      model.data.lid_q1.clear();
      model.data.lid_q2.clear();
      model.data.lid_p1.reserve(model.data.number_of_elements());
      model.data.lid_p2.reserve(model.data.number_of_elements());
      model.data.lid_q1.reserve(model.data.number_of_elements());
      model.data.lid_q2.reserve(model.data.number_of_elements());

      for (size_t i = 0; i < model.data.number_of_elements(); i++)
      {
        model.data.lid_p1.push_back(locally_relevant_dof_map.lid(model.data.gid_p1[i]));
        model.data.lid_p2.push_back(locally_relevant_dof_map.lid(model.data.gid_p2[i]));
        model.data.lid_q1.push_back(locally_relevant_dof_map.lid(model.data.gid_q1[i]));
        if (model.data.n_state_equations == 2)
        {
          model.data.lid_q2.push_back(locally_relevant_dof_map.lid(model.data.gid_q2[i]));
        }
      }
    }
  }

  void update_internal_state_vectors(AirwayContainer& airways,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
  {
    for (auto& model : airways.models)
    {
      model.internal_state_updater(model.data, locally_relevant_dofs, dt);
    }
  }

  void end_of_timestep_routine(AirwayContainer& airways,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt)
  {
    for (auto& model : airways.models)
    {
      for (size_t i = 0; i < model.data.number_of_elements(); i++)
      {
        model.data.p1_n[i] = locally_relevant_dofs.local_values_as_span()[model.data.lid_p1[i]];
        model.data.p2_n[i] = locally_relevant_dofs.local_values_as_span()[model.data.lid_p2[i]];
        model.data.q1_n[i] = locally_relevant_dofs.local_values_as_span()[model.data.lid_q1[i]];
        if (model.data.n_state_equations == 2)
        {
          model.data.q2_n[i] = locally_relevant_dofs.local_values_as_span()[model.data.lid_q2[i]];
        }
      }

      model.end_of_timestep_routine(model.data, locally_relevant_dofs, dt);
    }
  }

  void create_evaluators(AirwayContainer& airways)
  {
    for (auto& model : airways.models)
    {
      model.residual_evaluator =
          WallMechanics::make_residual_evaluator(model.wall_model, model.flow_model);
      model.jacobian_evaluator =
          WallMechanics::make_jacobian_evaluator(model.wall_model, model.flow_model);
      model.internal_state_updater = WallMechanics::make_internal_state_updater(
          model.wall_model, FlowResistance::make_internal_state_updater(model.flow_model));
      model.end_of_timestep_routine = WallMechanics::make_end_of_timestep_routine(model.wall_model);
      auto flow_output_evaluator = FlowResistance::make_output_evaluator(model.flow_model);
      auto wall_output_evaluator = WallMechanics::make_output_evaluator(model.wall_model);
      model.output_evaluator = [flow_output_evaluator, wall_output_evaluator](
                                   const AirwayData& data,
                                   ReducedLung::RuntimeOutputCollector& collector,
                                   ReducedLungParameters::OutputVerbosity verbosity)
      {
        flow_output_evaluator(data, collector, verbosity);
        wall_output_evaluator(data, collector, verbosity);
      };
    }
  }
}  // namespace ReducedLung::Airways

FOUR_C_NAMESPACE_CLOSE
