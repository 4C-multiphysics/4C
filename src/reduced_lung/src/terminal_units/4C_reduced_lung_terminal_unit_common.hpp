// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_COMMON_HPP
#define FOUR_C_REDUCED_LUNG_TERMINAL_UNIT_COMMON_HPP

#include "4C_config.hpp"

#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_reduced_lung_input.hpp"

#include <functional>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  struct RuntimeOutputCollector;
}

namespace ReducedLung::TerminalUnits
{
  /**
   * @brief Shared geometric, equation, and state data for a block of terminal units.
   *
   * One terminal-unit model block can represent multiple elements sharing the same constitutive
   * model pair (rheology + elasticity). This struct stores element-wise indices and physical
   * states required by assembly and time stepping.
   */
  struct TerminalUnitData
  {
    ///< Global element ids.
    std::vector<int> global_element_id;
    ///< Local element ids in the reduced-lung discretization.
    std::vector<int> local_element_id;
    ///< Local row ids in the residual/Jacobian row map.
    std::vector<int> local_row_id;
    ///< Global dof ids of p1.
    std::vector<int> gid_p1;
    ///< Global dof ids of p2.
    std::vector<int> gid_p2;
    ///< Global dof ids of q.
    std::vector<int> gid_q;
    ///< Local ids in the locally-relevant dof map for p1.
    std::vector<int> lid_p1;
    ///< Local ids in the locally-relevant dof map for p2.
    std::vector<int> lid_p2;
    ///< Local ids in the locally-relevant dof map for q.
    std::vector<int> lid_q;
    ///< Current terminal-unit volumes.
    std::vector<double> volume_v;
    ///< Reference terminal-unit volumes.
    std::vector<double> reference_volume_v0;

    /**
     * @brief Number of terminal-unit elements in this model block.
     */
    [[nodiscard]] size_t number_of_elements() const { return global_element_id.size(); }
  };

  ///< Callback type for residual block assembly.
  using ResidualEvaluator = std::function<void(TerminalUnitData& model_data,
      Core::LinAlg::Vector<double>& target_vector,
      const Core::LinAlg::Vector<double>& locally_relevant_dof_vector, double time_step_size_dt)>;

  ///< Callback type for Jacobian block assembly.
  using JacobianEvaluator = std::function<void(TerminalUnitData& model_data,
      Core::LinAlg::SparseMatrix& target_matrix,
      const Core::LinAlg::Vector<double>& locally_relevant_dof_vector, double time_step_size_dt)>;

  ///< Callback type for nonlinear-iteration internal state synchronization.
  using InternalStateUpdater = std::function<void(TerminalUnitData& model_data,
      const Core::LinAlg::Vector<double>& locally_relevant_dof_vector, double time_step_size_dt)>;

  ///< Callback type for end-of-timestep history updates.
  using EndOfTimestepRoutine = std::function<void(TerminalUnitData& model_data,
      const Core::LinAlg::Vector<double>& locally_relevant_dof_vector, double time_step_size_dt)>;

  ///< Callback type for collecting additional runtime output.
  using OutputEvaluator = std::function<void(const TerminalUnitData& model_data,
      RuntimeOutputCollector& collector, ReducedLungParameters::OutputVerbosity verbosity)>;

}  // namespace ReducedLung::TerminalUnits

FOUR_C_NAMESPACE_CLOSE

#endif
