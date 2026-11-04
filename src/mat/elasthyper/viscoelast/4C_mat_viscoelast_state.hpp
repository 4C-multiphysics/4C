// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_VISCOELAST_STATE_HPP
#define FOUR_C_MAT_VISCOELAST_STATE_HPP

#include "4C_config.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_material_factory.hpp"

#include <cstddef>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  /**
   * \brief Per-Gauss-point history state for all active Mat::ViscoElastHyper contributions.
   *
   * ViscoElastState owns the time-dependent internal variables that must survive nonlinear
   * iterations, time-step advancement, communication, and restart. It is intentionally independent
   * of the constitutive kernels: kernels receive read-only previous state and write current state
   * through narrow accessors.
   *
   * Storage layout:
   * - Iso-rate state stores previous/current right Cauchy-Green and modified right Cauchy-Green
   *   tensors per Gauss point.
   * - Generalized Maxwell state stores previous/current viscous and elastic branch stresses per
   *   Gauss point and branch.
   * - Quasi-linear generalized Maxwell state stores previous/current branch stresses plus the
   *   Green-Lagrange strain history used by the optional parallel dashpot.
   * - FSLS state stores the current artificial stress per Gauss point plus the converged artificial
   *   stress history used by the fractional update.
   *
   * Lifecycle contract:
   * - initialize_from_setup() or deserialize_state() activates the requested model sub-states and
   *   sizes their containers.
   * - Evaluation writes current-step values with the set_*_current_* accessors.
   * - advance_time_step() commits current values to previous history and resets current storage for
   *   the next time step.
   * - serialize_state() and deserialize_state() preserve only active model histories.
   * - Access to inactive or uninitialized model state is rejected with explicit diagnostics.
   */
  struct ViscoElastState
  {
    /// Fixed-size stress-like vector used by all visco history containers.
    using StressVector = Core::LinAlg::Matrix<NUM_STRESS_3D, 1>;
    /// One Gauss-point history vector, indexed by branch or stored time level depending on model.
    using PointHistory = std::vector<StressVector>;
    /// Generalized Maxwell container indexed by Gauss point and branch.
    using BranchHistory = std::vector<PointHistory>;
    /// FSLS container indexed by Gauss point and stored artificial-stress history entry.
    using FslsHistory = std::vector<PointHistory>;

    /// \brief Previous/current kinematic history for the iso-rate visco model.
    struct IsoRateState
    {
      /// Right Cauchy-Green tensor \f$C\f$ at current time level, per GP.
      PointHistory scg_current_;
      /// Right Cauchy-Green tensor \f$C\f$ at previous converged time level, per GP.
      PointHistory scg_previous_;
      /// Modified right Cauchy-Green tensor \f$\bar{C}\f$ at current time level, per GP.
      PointHistory modrcg_current_;
      /// Modified right Cauchy-Green tensor \f$\bar{C}\f$ at previous time level, per GP.
      PointHistory modrcg_previous_;
    };

    /// \brief Previous/current branch stress history for generalized Maxwell visco model.
    struct GeneralizedMaxwellState
    {
      /// Viscous branch stress at current time level, indexed by GP and branch.
      BranchHistory branch_stress_current_;
      /// Viscous branch stress at previous time level, indexed by GP and branch.
      BranchHistory branch_stress_previous_;
      /// Elastic branch stress at current time level, indexed by GP and branch.
      BranchHistory branch_elastic_stress_current_;
      /// Elastic branch stress at previous time level, indexed by GP and branch.
      BranchHistory branch_elastic_stress_previous_;
    };

    /// \brief Previous/current branch and dashpot history for quasi-linear generalized Maxwell.
    struct QuasiLinearGeneralizedMaxwellState
    {
      /// Viscous branch stress at current time level, indexed by GP and branch.
      BranchHistory branch_stress_current_;
      /// Viscous branch stress at previous time level, indexed by GP and branch.
      BranchHistory branch_stress_previous_;
      /// Elastic branch stress at current time level, indexed by GP and branch.
      BranchHistory branch_elastic_stress_current_;
      /// Elastic branch stress at previous time level, indexed by GP and branch.
      BranchHistory branch_elastic_stress_previous_;
      /// Green-Lagrange strain at current time level for optional parallel dashpots, per GP.
      PointHistory dashpot_strain_current_;
      /// Green-Lagrange strain at previous time level for optional parallel dashpots, per GP.
      PointHistory dashpot_strain_previous_;
      /// Parallel dashpot stress at current time level, per GP.
      PointHistory dashpot_stress_current_;
      /// Parallel dashpot stress at previous time level, per GP.
      PointHistory dashpot_stress_previous_;
    };

    /// \brief Previous/current history for FSLS artificial stress state.
    struct FslsState
    {
      /// Artificial FSLS stress at current time level, per GP.
      PointHistory artificial_stress_current_;
      /// Full converged FSLS history per GP (time history of artificial stress).
      FslsHistory artificial_stress_previous_history_;
    };

    /// \brief Read-only per-GP view of previous iso-rate kinematic state.
    struct IsoRatePrevPointState
    {
      const StressVector& scg;
      const StressVector& modrcg;
    };

    /// \brief Activation flags for the optional model sub-states owned by this aggregate.
    struct ActiveModels
    {
      bool iso_rate = false;
      bool generalized_maxwell = false;
      bool quasi_linear_generalized_maxwell = false;
      bool fsls = false;
    };

    /// Iso-rate model state containers.
    IsoRateState iso_rate_;
    /// Generalized Maxwell model state containers.
    GeneralizedMaxwellState generalized_maxwell_;
    /// Quasi-linear generalized Maxwell model state containers.
    QuasiLinearGeneralizedMaxwellState quasi_linear_generalized_maxwell_;
    /// FSLS model state containers.
    FslsState fsls_;

    /// True once state was initialized via setup or deserialization.
    bool isinitvis_ = false;
    /// Number of Gauss points represented by this state instance.
    int gp_count_ = 0;
    /// Activation flag for iso-rate visco model state.
    bool has_iso_rate_state_ = false;
    /// Activation flag for generalized Maxwell model state.
    bool has_generalized_maxwell_state_ = false;
    /// Activation flag for quasi-linear generalized Maxwell model state.
    bool has_quasi_linear_generalized_maxwell_state_ = false;
    /// Activation flag for FSLS model state.
    bool has_fsls_state_ = false;

    /// \brief Returns whether the state has been initialized.
    [[nodiscard]] bool initialized() const;

    /**
     * \brief Serialize active model histories to a communication buffer.
     *
     * The active model descriptor is supplied by the owning material and must match the configured
     * state when the state is initialized.
     */
    void serialize_state(
        Core::Communication::PackBuffer& data, const ActiveModels& active_models) const;

    /// Deserialize active model histories and configure this state accordingly.
    void deserialize_state(
        Core::Communication::UnpackBuffer& buffer, const ActiveModels& active_models);

    /// Initialize state containers from setup metadata and active model selection.
    void initialize_from_setup(int gp_count, const ActiveModels& active_models,
        std::size_t generalized_maxwell_branch_count,
        std::size_t quasi_linear_generalized_maxwell_branch_count);

    /// Advance active histories from current to previous and reset current state containers.
    void advance_time_step(
        const ActiveModels& active_models, unsigned int fsls_max_history_size, int visco_mat_id);

    /// \brief Return read-only previous iso-rate state for one Gauss point.
    IsoRatePrevPointState iso_rate_prev_point(int gp) const;
    /// \brief Store current iso-rate state for one Gauss point.
    void set_iso_rate_current_point(int gp, const StressVector& scg, const StressVector& modrcg);

    /// \brief Return read-only previous generalized Maxwell elastic branch stresses for one GP.
    const PointHistory& generalized_maxwell_prev_branch_elastic_stress(int gp) const;
    /// \brief Return read-only previous generalized Maxwell viscous branch stresses for one GP.
    const PointHistory& generalized_maxwell_prev_branch_stress(int gp) const;
    /// \brief Store current generalized Maxwell state for one Gauss point.
    void set_generalized_maxwell_current_point(
        int gp, const PointHistory& branch_elastic_stress, const PointHistory& branch_stress);

    /// \brief Return previous QLV elastic branch stresses for one GP.
    const PointHistory& quasi_linear_generalized_maxwell_prev_branch_elastic_stress(int gp) const;
    /// \brief Return previous QLV viscous branch stresses for one GP.
    const PointHistory& quasi_linear_generalized_maxwell_prev_branch_stress(int gp) const;
    /// \brief Return previous Green-Lagrange strain for QLV dashpot terms.
    const StressVector& quasi_linear_generalized_maxwell_prev_dashpot_strain(int gp) const;
    /// \brief Return previous QLV parallel dashpot stress for one GP.
    const StressVector& quasi_linear_generalized_maxwell_prev_dashpot_stress(int gp) const;
    /// \brief Store current QLV branch state for one Gauss point.
    void set_quasi_linear_generalized_maxwell_current_point(
        int gp, const PointHistory& branch_elastic_stress, const PointHistory& branch_stress);
    /// \brief Store current Green-Lagrange strain for QLV dashpot terms.
    void set_quasi_linear_generalized_maxwell_current_dashpot_strain(
        int gp, const StressVector& strain);
    /// \brief Store current QLV parallel dashpot stress for one GP.
    void set_quasi_linear_generalized_maxwell_current_dashpot_stress(
        int gp, const StressVector& stress);

    /// \brief Return full converged FSLS history (per GP and time step).
    const FslsHistory& fsls_previous_history() const;
    /// \brief Store current FSLS artificial stress for one Gauss point.
    void set_fsls_current_artificial_stress(int gp, const StressVector& value);

    /// \brief Reset all state containers and activation metadata.
    void clear();

   private:
    void mark_initialized(bool initialized);
    void configure_active_models(const ActiveModels& active_models);

    [[nodiscard]] ActiveModels configured_active_models() const;

    [[nodiscard]] int serialized_gp_count() const;
    [[nodiscard]] int gp_count() const;

    void clear_iso_rate_state();
    void clear_generalized_maxwell_state();
    void clear_quasi_linear_generalized_maxwell_state();
    void clear_fsls_state();

    void ensure_initialized(const char* context) const;
    void ensure_model_active(
        bool model_is_active, const char* model_name, const char* context) const;
    void ensure_gp_in_range(int gp, const char* model_name, const char* context) const;

    void serialize_iso_rate_state(Core::Communication::PackBuffer& data, int gp_count) const;
    void serialize_generalized_maxwell_state(
        Core::Communication::PackBuffer& data, int gp_count) const;
    void serialize_quasi_linear_generalized_maxwell_state(
        Core::Communication::PackBuffer& data, int gp_count) const;
    void serialize_fsls_state(Core::Communication::PackBuffer& data, int gp_count) const;
    void deserialize_iso_rate_state(Core::Communication::UnpackBuffer& buffer, int gp_count);
    void deserialize_generalized_maxwell_state(
        Core::Communication::UnpackBuffer& buffer, int gp_count);
    void deserialize_quasi_linear_generalized_maxwell_state(
        Core::Communication::UnpackBuffer& buffer, int gp_count);
    void deserialize_fsls_state(Core::Communication::UnpackBuffer& buffer, int gp_count);

    void initialize_iso_rate_state_for_deserialize(int gp_count);
    void initialize_generalized_maxwell_state_for_deserialize(int gp_count);
    void initialize_quasi_linear_generalized_maxwell_state_for_deserialize(int gp_count);
    void initialize_fsls_state_for_deserialize(int gp_count, std::size_t history_size);

    void initialize_iso_rate_state(int gp_count);
    void initialize_generalized_maxwell_state(int gp_count, std::size_t branch_count);
    void initialize_quasi_linear_generalized_maxwell_state(int gp_count, std::size_t branch_count);
    void initialize_fsls_state(int gp_count);

    void commit_iso_rate_to_previous();
    void append_fsls_current_to_previous_history(unsigned int max_history_size);
    void reset_current_state(int gp_count, bool has_iso_rate_state, bool has_fsls_state);
    std::size_t rotate_generalized_maxwell_to_previous(int gp_count, int visco_mat_id);
    void reset_generalized_maxwell_current_state(int gp_count, std::size_t branch_count);
    std::size_t rotate_quasi_linear_generalized_maxwell_to_previous(int gp_count, int visco_mat_id);
    void reset_quasi_linear_generalized_maxwell_current_state(
        int gp_count, std::size_t branch_count);
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
