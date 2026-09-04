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
#include <limits>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  struct RuntimeOutputCollector;
}

namespace ReducedLung::TerminalUnits
{
  /**
   * @brief Effective reference volume of one terminal-unit element and its pressure derivatives.
   *
   * Recruitment makes the reference volume depend on the transmural pressure. The derivatives
   * vanish wherever the reference volume is constant within a Newton step, i.e. for terminal
   * units without recruitment and for the frozen linearization.
   */
  struct ReferenceVolumeContext
  {
    ///< Effective reference volume.
    double v0_eff = std::numeric_limits<double>::quiet_NaN();
    ///< Reciprocal of the effective reference volume, cached because assembly mostly needs it.
    double inv_v0_eff = std::numeric_limits<double>::quiet_NaN();
    ///< Derivative of the effective reference volume w.r.t. the transmural pressure.
    double dv0_dp = std::numeric_limits<double>::quiet_NaN();
  };

  /**
   * @brief Assemble a reference volume context from the reference volume and its derivative.
   */
  [[nodiscard]] inline ReferenceVolumeContext make_reference_volume_context(
      const double v0_eff, const double dv0_dp)
  {
    const double inv_v0_eff = 1.0 / v0_eff;
    return {.v0_eff = v0_eff, .inv_v0_eff = inv_v0_eff, .dv0_dp = dv0_dp};
  }

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
    ///< Current physical terminal-unit gas volumes.
    std::vector<double> volume_v;
    ///< Effective reference volume of each element, refreshed whenever the dof vector changes.
    std::vector<ReferenceVolumeContext> reference_volume_context;

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
