// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_AIRWAYS_HPP
#define FOUR_C_REDUCED_LUNG_AIRWAYS_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_airways_common.hpp"
#include "4C_reduced_lung_airways_flow_resistance.hpp"
#include "4C_reduced_lung_airways_wall_mechanics.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::Airways
{
  /**
   * @brief One airway model block (data + selected constitutive pair + evaluator callbacks).
   */
  struct AirwayModel
  {
    AirwayData data;
    FlowModel flow_model;
    WallModel wall_model;
    ResidualEvaluator residual_evaluator;
    JacobianEvaluator jacobian_evaluator;
    InternalStateUpdater internal_state_updater;
    EndOfTimestepRoutine end_of_timestep_routine;
    OutputEvaluator output_evaluator;
  };

  /**
   * @brief Container for all airway model blocks in the local partition.
   */
  struct AirwayContainer
  {
    std::vector<AirwayModel> models;
  };

  /**
   * @brief Assemble airway residual contributions for all local model blocks.
   */
  void update_residual_vector(Core::LinAlg::Vector<double>& res_vector, AirwayContainer& airways,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

  /**
   * @brief Assemble airway Jacobian contributions for all local model blocks.
   */
  void update_jacobian(Core::LinAlg::SparseMatrix& jac, AirwayContainer& airways,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

  /**
   * @brief Build residual/Jacobian/state callback objects for all model blocks.
   */
  void create_evaluators(AirwayContainer& airways);

  /**
   * @brief Assign local row ids for airway equations.
   */
  void assign_local_equation_ids(AirwayContainer& airways, int& n_local_equations);

  /**
   * @brief Assign local dof ids from the locally relevant dof map.
   */
  void assign_local_dof_ids(
      const Core::LinAlg::Map& locally_relevant_dof_map, AirwayContainer& airways);

  /**
   * @brief Synchronize model-internal state from converged dofs.
   */
  void update_internal_state_vectors(AirwayContainer& airways,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

  /**
   * @brief Run end-of-step updates, including history-variable advancement.
   */
  void end_of_timestep_routine(AirwayContainer& airways,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);
}  // namespace ReducedLung::Airways

FOUR_C_NAMESPACE_CLOSE

#endif
