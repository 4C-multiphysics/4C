// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_HPP
#define FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_HPP

#include "4C_config.hpp"

#include "4C_linalg_map.hpp"
#include "4C_reduced_lung_terminal_unit_common.hpp"
#include "4C_reduced_lung_terminal_unit_elasticity.hpp"
#include "4C_reduced_lung_terminal_unit_recruitment.hpp"
#include "4C_reduced_lung_terminal_unit_rheology.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung::TerminalUnits
{
  /**
   * @brief One terminal-unit model block (data + selected constitutive pair + evaluator callbacks).
   */
  struct TerminalUnitModel
  {
    TerminalUnitData data;
    RheologicalModel rheological_model;
    ElasticityModel elasticity_model;
    RecruitmentModel recruitment_model;
    ResidualEvaluator residual_evaluator;
    JacobianEvaluator jacobian_evaluator;
    StaticTreeLinearizationEvaluator
        static_tree_linearization_evaluator;  ///< One-time structured coefficient pattern callback.
    TreeLinearizationEvaluator
        tree_linearization_evaluator;  ///< Dynamic structured coefficient refresh callback.
    InternalStateUpdater internal_state_updater;
    EndOfTimestepRoutine end_of_timestep_routine;
    OutputEvaluator output_evaluator;
  };

  /**
   * @brief Container for all terminal-unit model blocks in the local partition.
   */
  struct TerminalUnitContainer
  {
    std::vector<TerminalUnitModel> models;
  };

  /**
   * @brief Assemble terminal-unit residual contributions for all local model blocks.
   */
  void update_residual_vector(Core::LinAlg::Vector<double>& res_vector,
      TerminalUnitContainer& terminal_units,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

  /**
   * @brief Assemble terminal-unit Jacobian contributions for all local model blocks.
   */
  void update_jacobian(Core::LinAlg::SparseMatrix& jac, TerminalUnitContainer& terminal_units,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

  /**
   * @brief Assemble static terminal-unit structured tree-linearization row patterns.
   *
   * Appends p1, p2, and q placeholders consumed by the direct and generic structured tree assembly
   * paths.
   */
  void update_static_tree_linearization(
      TreeCoefficientAssemblyTarget& target, TerminalUnitContainer& terminal_units);

  /**
   * @brief Update dynamic terminal-unit structured tree-linearization coefficients.
   *
   * Replaces all three coefficients in the row pattern. Coupled recruitment makes the pressure
   * coefficients state-dependent as well as the q coefficient.
   */
  void update_tree_linearization(TreeCoefficientAssemblyTarget& target,
      TerminalUnitContainer& terminal_units,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

  /**
   * @brief Build residual/Jacobian/state callback objects for all model blocks.
   *
   * This routine composes elasticity and rheology module callbacks into executable evaluator
   * functions stored in each @ref TerminalUnitModel.
   */
  void create_evaluators(TerminalUnitContainer& terminal_units);

  /**
   * @brief Assign local row ids for terminal-unit equations.
   */
  void assign_local_equation_ids(TerminalUnitContainer& terminal_units, int& n_local_equations);

  /**
   * @brief Assign local dof ids from the locally relevant dof map.
   */
  void assign_local_dof_ids(
      const Core::LinAlg::Map& locally_relevant_dof_map, TerminalUnitContainer& terminal_units);

  /**
   * @brief Synchronize model-internal state from converged dofs.
   */
  void update_internal_state_vectors(TerminalUnitContainer& terminal_units,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);

  /**
   * @brief Run end-of-step updates, including volume state advancement.
   */
  void end_of_timestep_routine(TerminalUnitContainer& terminal_units,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double dt);
}  // namespace ReducedLung::TerminalUnits

FOUR_C_NAMESPACE_CLOSE

#endif
