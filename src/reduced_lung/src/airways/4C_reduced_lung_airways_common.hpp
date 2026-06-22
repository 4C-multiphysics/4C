// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_AIRWAYS_COMMON_HPP
#define FOUR_C_REDUCED_LUNG_AIRWAYS_COMMON_HPP

#include "4C_config.hpp"

#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_reduced_lung_input.hpp"

#include <cstddef>
#include <functional>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  struct RuntimeOutputCollector;
}
namespace ReducedLung::Airways
{
  /**
   * @brief Shared geometric, equation, and state data for one airway model block.
   *
   * One airway model block can represent multiple elements sharing the same constitutive model
   * pair (flow resistance + wall mechanics). This struct stores element-wise indices and physical
   * states required by assembly and time stepping.
   */
  struct AirwayData
  {
    ///< Global element ids.
    std::vector<int> global_element_id;
    ///< Local element ids in the reduced-lung discretization.
    std::vector<int> local_element_id;
    ///< Local row ids in the residual/Jacobian row map.
    std::vector<int> local_row_id;
    ///< Global dof ids of p1, p2, q1, q2.
    std::vector<int> gid_p1, gid_p2, gid_q1, gid_q2;
    ///< Local ids in the locally-relevant dof map for p1, p2, q1, q2.
    std::vector<int> lid_p1, lid_p2, lid_q1, lid_q2;

    ///< Reference airway lengths.
    std::vector<double> ref_length;
    ///< Reference airway cross-sectional areas.
    std::vector<double> ref_area;
    struct AirProperties
    {
      double dynamic_viscosity;
      double density;
    } air_properties;

    ///< Previous-step flow history in branch q1.
    std::vector<double> q1_n;
    ///< Previous-step flow history in branch q2.
    std::vector<double> q2_n;
    ///< Previous-step pressure history at node p1.
    std::vector<double> p1_n;
    ///< Previous-step pressure history at node p2.
    std::vector<double> p2_n;

    ///< Number of state equations contributed per airway element in this block.
    int n_state_equations;

    /**
     * @brief Number of airway elements in this model block.
     */
    [[nodiscard]] size_t number_of_elements() const { return global_element_id.size(); }
  };

  ///< Callback type for residual block assembly.
  using ResidualEvaluator =
      std::function<void(const AirwayData& data, Core::LinAlg::Vector<double>& target_vector,
          const Core::LinAlg::Vector<double>& locally_relevant_dofs, double time_step_size_dt)>;

  ///< Callback type for Jacobian block assembly.
  using JacobianEvaluator =
      std::function<void(const AirwayData& data, Core::LinAlg::SparseMatrix& target_mat,
          const Core::LinAlg::Vector<double>& locally_relevant_dofs, double time_step_size_dt)>;

  ///< Callback type for nonlinear-iteration internal state synchronization.
  using InternalStateUpdater = std::function<void(AirwayData& data,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double time_step_size_dt)>;

  ///< Callback type for flow-model-only internal state synchronization.
  using FlowModelInternalStateUpdater = std::function<void(
      AirwayData& data, const Core::LinAlg::Vector<double>& locally_relevant_dofs)>;

  ///< Callback type for end-of-timestep history updates.
  using EndOfTimestepRoutine = std::function<void(AirwayData& data,
      const Core::LinAlg::Vector<double>& locally_relevant_dofs, double time_step_size_dt)>;

  ///< Callback type for collecting additional runtime output.
  using OutputEvaluator = std::function<void(const AirwayData& data,
      RuntimeOutputCollector& collector, ReducedLungParameters::OutputVerbosity verbosity)>;
}  // namespace ReducedLung::Airways

FOUR_C_NAMESPACE_CLOSE

#endif
