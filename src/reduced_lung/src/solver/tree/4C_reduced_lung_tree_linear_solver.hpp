// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_TREE_LINEAR_SOLVER_HPP
#define FOUR_C_REDUCED_LUNG_TREE_LINEAR_SOLVER_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_newton_linear_solver.hpp"
#include "4C_reduced_lung_tree_linearization.hpp"
#include "4C_reduced_lung_tree_metadata.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  struct TreeNewtonLinearSolverProfile;

  /**
   * @brief Location of one tree-solver coefficient in row-oriented structured storage.
   */
  struct TreeCoefficientLocation
  {
    int local_row = -1;               ///< Local residual row id.
    int local_dof = -1;               ///< Local dof id on the locally relevant dof map.
    int structured_entry_index = -1;  ///< Entry index inside the structured row storage.
  };

  /**
   * @brief Debug/value view of a structured coefficient stored by the direct tree target.
   */
  struct TreeStructuredCoefficientValue
  {
    int local_row = -1;             ///< Local residual row id.
    int local_dof = -1;             ///< Local dof id on the locally relevant dof map.
    double value = 0.0;             ///< Stored coefficient value.
    const char* context = nullptr;  ///< Human-readable coefficient context.
  };

  /**
   * @brief Coefficient source used by the serial tree Newton linear solver.
   */
  enum class TreeNewtonLinearSolverCoefficientSource
  {
    SparseJacobian,  ///< Validation reference: read coefficients from a completed sparse Jacobian.
    StructuredTreeBlocks,  ///< Runtime path: read structured tree-linearization blocks.
  };

  /**
   * @brief Context for the serial tree-based Newton correction solver.
   *
   * The batch-forcing and fallback-error controls exist for deterministic validation of optimized
   * paths. Production callers should retain their default values.
   */
  struct TreeNewtonLinearSolverContext
  {
    const ReducedLungTreeMetadata& tree_metadata;  ///< Directed tree metadata for the solve.
    double pivot_tolerance = 1.0e-12;              ///< Dense pivot tolerance.
    TreeNewtonLinearSolverCoefficientSource coefficient_source =
        TreeNewtonLinearSolverCoefficientSource::SparseJacobian;  ///< Validation or runtime source.
    Core::LinAlg::SparseMatrix* validation_jacobian =
        nullptr;  ///< Validation-only matrix for the sparse coefficient source.
    TreeNewtonLinearSolverProfile* profile = nullptr;  ///< Optional profile sink.
    bool force_batch_tree_solve = false;   ///< Validation only: force batches for small trees.
    bool error_on_dense_fallback = false;  ///< Validation only: reject dense fallback.
  };

  /**
   * @brief Serial tree-based solver for reduced-lung Newton correction systems.
   *
   * Uses tree metadata to condense subtrees bottom-up into inlet relations and recover the Newton
   * correction top-down. Runtime NewtonTree uses structured coefficients; sparse coefficients
   * remain available for validation.
   */
  class TreeNewtonLinearSolver : public NewtonLinearSolver, public TreeCoefficientAssemblyTarget
  {
   public:
    /**
     * @brief Construct the serial tree Newton correction solver.
     *
     * @param context Tree metadata, coefficient source, tolerances, and optional profile sink.
     */
    explicit TreeNewtonLinearSolver(const TreeNewtonLinearSolverContext& context);

    /**
     * @brief Return the linearization representation required by this solver.
     *
     * @return Sparse or structured tree-block linearization type.
     */
    [[nodiscard]] NewtonLinearizationType linearization_type() const override;

    /**
     * @brief Return the validation sparse Jacobian when the sparse reference source is selected.
     */
    [[nodiscard]] Core::LinAlg::SparseMatrix* sparse_jacobian_target() override;

    /**
     * @brief Provide generic structured coefficients for validation/reference solves.
     *
     * Runtime NewtonTree assembles directly into solver-owned storage instead. This setter remains
     * available to compare the generic representation with the direct production path.
     *
     * @param tree_linearization Structured coefficients assembled for the current Newton state.
     */
    void set_tree_linearization(const TreeLinearization& tree_linearization) override;

    /**
     * @brief Return this solver as a direct structured coefficient assembly target.
     *
     * @return Direct coefficient target for structured runtime assembly.
     */
    [[nodiscard]] TreeCoefficientAssemblyTarget* direct_tree_coefficient_target() override;

    /**
     * @brief Return stored structured coefficients for validation diagnostics.
     *
     * This view supports coefficient-by-coefficient equivalence tests and is not used by the
     * production solve.
     *
     * @return Flat list of direct coefficient values and their row/dof locations.
     */
    [[nodiscard]] std::vector<TreeStructuredCoefficientValue> structured_coefficient_values() const;

    /**
     * @brief Solve one Newton correction system with the tree algorithm.
     *
     * @param residual Residual vector for the current nonlinear state.
     * @param x Current nonlinear solution vector.
     * @param metadata Current time-step and Newton-iteration metadata.
     * @param delta Output Newton correction vector.
     */
    void solve(const Core::LinAlg::Vector<double>& residual, const Core::LinAlg::Vector<double>& x,
        const NewtonLinearSystemMetadata& metadata, Core::LinAlg::Vector<double>& delta) override;

   private:
    /**
     * @brief Contiguous group of elements with identical local block shape.
     */
    struct ElementGroup
    {
      int begin = 0;        ///< First grouped element index.
      int end = 0;          ///< One-past-last grouped element index.
      int block_size = 0;   ///< Element local block size.
      int child_count = 0;  ///< Number of child interfaces.
    };

    /**
     * @brief Precompute topology, row, dof, coefficient-location, grouping, and workspace data.
     */
    void build_symbolic_plan();

    /**
     * @brief Resolve structured row-entry indices for all coefficients used by the tree solver.
     *
     * @param tree_linearization Structured coefficient storage for the current layout.
     */
    void resolve_structured_coefficient_locations(const TreeLinearization& tree_linearization);

    /**
     * @brief Append a direct structured coefficient value to this solver target.
     *
     * @param local_row_id Local residual row id.
     * @param local_dof_id Local dof id on the locally relevant dof map.
     * @param value Coefficient value.
     */
    void append_value(int local_row_id, int local_dof_id, double value) override;

    /**
     * @brief Replace a direct structured coefficient value in this solver target.
     *
     * @param local_row_id Local residual row id.
     * @param local_dof_id Local dof id on the locally relevant dof map.
     * @param value New coefficient value.
     */
    void replace_value(int local_row_id, int local_dof_id, double value) override;

    /**
     * @brief Replace a batch of direct structured coefficient values.
     *
     * @param local_row_ids Local residual row ids.
     * @param local_dof_ids Local dof ids on the locally relevant dof map.
     * @param values New coefficient values.
     */
    void replace_values(std::span<const int> local_row_ids, std::span<const int> local_dof_ids,
        std::span<const double> values) override;

    /**
     * @brief Set one direct structured coefficient value.
     *
     * @param local_row_id Local residual row id.
     * @param local_dof_id Local dof id on the locally relevant dof map.
     * @param value Coefficient value.
     * @param operation Human-readable operation name for diagnostics.
     */
    void set_direct_coefficient_value(
        int local_row_id, int local_dof_id, double value, const char* operation);

    const ReducedLungTreeMetadata& tree_metadata_;
    double pivot_tolerance_;
    TreeNewtonLinearSolverCoefficientSource coefficient_source_;
    Core::LinAlg::SparseMatrix* validation_jacobian_ = nullptr;
    const TreeLinearization* tree_linearization_ = nullptr;
    TreeNewtonLinearSolverProfile* profile_ = nullptr;
    bool force_batch_tree_solve_ = false;   ///< Validation override; disabled in production.
    bool error_on_dense_fallback_ = false;  ///< Validation check; disabled in production.

    BoundaryConditions::ConstrainedVariable root_boundary_variable_ =
        BoundaryConditions::ConstrainedVariable::Pressure;
    int root_boundary_row_ = -1;
    int root_boundary_local_dof_ = -1;
    std::vector<int> global_element_id_;
    std::vector<int> inlet_pressure_local_dof_;
    std::vector<int> inlet_flow_unknown_index_;
    std::vector<int> outlet_pressure_unknown_index_;
    std::vector<int> block_size_;
    std::vector<int> child_interface_count_;
    std::vector<unsigned char> is_leaf_;

    std::vector<int> unknown_offset_;
    std::vector<int> equation_offset_;
    std::vector<int> child_interface_offset_;
    std::vector<int> matrix_offset_;

    std::vector<int> unknown_global_dof_ids_;
    std::vector<int> unknown_local_dof_ids_;
    std::vector<int> equation_rows_;
    std::vector<int> inlet_pressure_correction_local_dof_ids_;
    std::vector<int> unknown_correction_local_dof_ids_;
    bool correction_local_dof_ids_initialized_ = false;

    std::vector<int> child_element_index_;
    std::vector<int> pressure_row_;
    std::vector<int> parent_outlet_pressure_local_dof_;
    std::vector<int> child_inlet_pressure_local_dof_;
    std::vector<int> child_inlet_flow_local_dof_;
    std::vector<int> parent_outlet_pressure_unknown_index_;

    TreeCoefficientLocation root_boundary_coefficient_;
    std::vector<TreeCoefficientLocation> equation_inlet_pressure_coefficients_;
    std::vector<TreeCoefficientLocation> matrix_coefficients_;
    std::vector<TreeCoefficientLocation> child_pressure_parent_coefficients_;
    std::vector<TreeCoefficientLocation> child_pressure_child_coefficients_;
    std::vector<TreeCoefficientLocation> child_flow_coefficients_;

    double root_boundary_coefficient_value_ = 0.0;
    std::vector<double> equation_inlet_pressure_coefficient_values_;
    std::vector<double> matrix_coefficient_values_;
    std::vector<double> child_pressure_parent_coefficient_values_;
    std::vector<double> child_pressure_child_coefficient_values_;
    std::vector<double> child_flow_coefficient_values_;

    struct DirectCoefficientEntry
    {
      int local_dof = -1;
      double* value = nullptr;
      const char* context = nullptr;
    };
    std::vector<int> direct_coefficient_row_offsets_;
    std::vector<DirectCoefficientEntry> direct_coefficient_entries_;

    std::vector<int> grouped_element_indices_;
    std::vector<int> grouped_unknown_begin_;
    std::vector<int> grouped_matrix_begin_;
    std::vector<int> grouped_equation_begin_;
    std::vector<int> grouped_child_begin_;
    std::vector<std::vector<ElementGroup>> bottom_up_layer_groups_;
    std::vector<std::vector<ElementGroup>> top_down_layer_groups_;
    bool use_scalar_tree_solve_ = false;

    std::vector<double> workspace_matrix_;
    std::vector<double> workspace_rhs_constant_;
    std::vector<double> workspace_rhs_inlet_pressure_;
    std::vector<double> workspace_intercept_;
    std::vector<double> workspace_slope_;
    std::vector<double> child_pressure_slope_;
    std::vector<double> child_pressure_intercept_;
    std::vector<double> batch_2x2_a00_;
    std::vector<double> batch_2x2_a01_;
    std::vector<double> batch_2x2_a10_;
    std::vector<double> batch_2x2_a11_;
    std::vector<double> batch_2x2_rhs_constant0_;
    std::vector<double> batch_2x2_rhs_constant1_;
    std::vector<double> batch_2x2_rhs_inlet_pressure0_;
    std::vector<double> batch_2x2_rhs_inlet_pressure1_;
    std::vector<double> batch_2x2_intercept0_;
    std::vector<double> batch_2x2_intercept1_;
    std::vector<double> batch_2x2_slope0_;
    std::vector<double> batch_2x2_slope1_;
    std::vector<int> batch_2x2_fallback_lanes_;
    std::vector<double> batch_3x3_a00_;
    std::vector<double> batch_3x3_a01_;
    std::vector<double> batch_3x3_a02_;
    std::vector<double> batch_3x3_a10_;
    std::vector<double> batch_3x3_a11_;
    std::vector<double> batch_3x3_a12_;
    std::vector<double> batch_3x3_a20_;
    std::vector<double> batch_3x3_a21_;
    std::vector<double> batch_3x3_a22_;
    std::vector<double> batch_3x3_rhs_constant0_;
    std::vector<double> batch_3x3_rhs_constant1_;
    std::vector<double> batch_3x3_rhs_constant2_;
    std::vector<double> batch_3x3_rhs_inlet_pressure0_;
    std::vector<double> batch_3x3_rhs_inlet_pressure1_;
    std::vector<double> batch_3x3_rhs_inlet_pressure2_;
    std::vector<double> batch_3x3_intercept0_;
    std::vector<double> batch_3x3_intercept1_;
    std::vector<double> batch_3x3_intercept2_;
    std::vector<double> batch_3x3_slope0_;
    std::vector<double> batch_3x3_slope1_;
    std::vector<double> batch_3x3_slope2_;
    std::vector<int> batch_3x3_fallback_lanes_;
    std::vector<double> top_down_inlet_pressure_;
    std::vector<double> top_down_unknown_values_;
    std::vector<double> top_down_outlet_pressure_;
    std::vector<double> top_down_child_pressure_;
    std::vector<std::string> element_context_;

    std::vector<double> subtree_relation_G_;
    std::vector<double> subtree_relation_h_;
    std::vector<double> inlet_pressure_by_element_;
    std::vector<int> inlet_pressure_stamp_;
    int current_solve_stamp_ = 0;
  };

}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
