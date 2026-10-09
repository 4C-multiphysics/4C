// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_TREE_LINEAR_SOLVER_HPP
#define FOUR_C_REDUCED_LUNG_TREE_LINEAR_SOLVER_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_tree_metadata.hpp"

#include <array>
#include <span>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  template <typename T>
  class Vector;
}  // namespace Core::LinAlg

namespace ReducedLung
{
  /**
   * @brief Location of one tree-solver coefficient in row-oriented structured storage.
   */
  struct TreeCoefficientLocation
  {
    int local_row = -1;  ///< Local residual row id.
    int local_dof = -1;  ///< Local dof id on the locally relevant dof map.
  };

  /**
   * @brief Context for the serial tree-based Newton correction solver.
   */
  struct TreeNewtonLinearSolverContext
  {
    const ReducedLungTreeMetadata& tree_metadata;  ///< Directed tree metadata for the solve.
    double pivot_tolerance = 1.0e-12;              ///< Dense pivot tolerance.
  };

  /**
   * @brief Serial tree-based solver for reduced-lung Newton correction systems.
   *
   * Uses tree metadata to recursively statically condense the directed tree from leaves to root.
   * For each element/subtree, the inlet pressure correction is kept as the interface parameter and
   * all remaining local unknown corrections are eliminated, producing the affine inlet-flow
   * relation `delta_q_in = G * delta_p_in + h`. Child subtree relations are inserted into the
   * parent local block, which is the tree-structured Schur-complement/static-condensation step.
   * Once the root boundary equation determines the root inlet-pressure correction, all remaining
   * Newton corrections are recovered top-down. Coefficients are assembled directly into
   * solver-owned structured storage.
   */
  class TreeNewtonLinearSolver
  {
   public:
    /**
     * @brief Construct the serial tree Newton correction solver.
     *
     * @param context Tree metadata and tolerances.
     */
    explicit TreeNewtonLinearSolver(const TreeNewtonLinearSolverContext& context);

    /**
     * @brief Set a structured coefficient value in this solver target.
     *
     * @param local_row_id Local residual row id.
     * @param local_dof_id Local dof id on the locally relevant dof map.
     * @param value Coefficient value.
     */
    void set_value(int local_row_id, int local_dof_id, double value);

    /**
     * @brief Set a batch of structured coefficient values.
     *
     * @param local_row_ids Local residual row ids.
     * @param local_dof_ids Local dof ids on the locally relevant dof map.
     * @param values Coefficient values.
     */
    void set_values(std::span<const int> local_row_ids, std::span<const int> local_dof_ids,
        std::span<const double> values);

    /**
     * @brief Solve one Newton correction system with the tree algorithm.
     *
     * @param residual Residual vector for the current nonlinear state.
     * @param delta Output Newton correction vector.
     */
    void solve(const Core::LinAlg::Vector<double>& residual, Core::LinAlg::Vector<double>& delta);

   private:
    /**
     * @brief Precompute topology, row, dof, coefficient-location, and workspace data.
     */
    void build_symbolic_plan();

    /**
     * @brief Validate coefficient sources and vector maps before a solve.
     */
    void validate_solve_inputs(const Core::LinAlg::Vector<double>& residual,
        const Core::LinAlg::Vector<double>& delta) const;

    /**
     * @brief Resolve correction-vector local ids once for repeated tree solves.
     */
    void initialize_correction_local_dof_ids(const Core::LinAlg::Vector<double>& delta);

    /**
     * @brief Execute one tree solve with solver-owned structured coefficients.
     */
    void solve_with_coefficients(
        const Core::LinAlg::Vector<double>& residual, Core::LinAlg::Vector<double>& delta);

    /**
     * @brief Close the condensed root system and return the root inlet-pressure correction.
     */
    [[nodiscard]] double solve_root_inlet_pressure(
        const Core::LinAlg::Vector<double>& residual) const;

    // Immutable metadata and numerical tolerance used by every correction solve.
    const ReducedLungTreeMetadata& tree_metadata_;
    double pivot_tolerance_;

    // Root boundary row and dof data used to close the condensed tree system.
    BoundaryConditions::ConstrainedVariable root_boundary_variable_ =
        BoundaryConditions::ConstrainedVariable::Pressure;
    int root_boundary_row_ = -1;
    int root_boundary_local_dof_ = -1;

    // Per-element symbolic data derived from tree metadata. The inlet pressure is the interface
    // parameter; the other element dofs form the local block condensed in the bottom-up pass.
    std::vector<int> global_element_id_;
    std::vector<int> inlet_pressure_local_dof_;
    std::vector<int> inlet_flow_unknown_index_;
    std::vector<int> outlet_pressure_unknown_index_;
    std::vector<int> block_size_;
    std::vector<int> child_interface_count_;
    std::vector<unsigned char> is_leaf_;

    // Offsets into the flattened unknown, equation, child-interface, and local matrix arrays.
    std::vector<int> unknown_offset_;
    std::vector<int> equation_offset_;
    std::vector<int> child_interface_offset_;
    std::vector<int> matrix_offset_;

    // Flattened local unknown and equation ids used to assemble and recover element corrections.
    // Correction-vector local ids are resolved lazily from the solve output map.
    std::vector<int> unknown_global_dof_ids_;
    std::vector<int> unknown_local_dof_ids_;
    std::vector<int> equation_rows_;
    std::vector<int> inlet_pressure_correction_local_dof_ids_;
    std::vector<int> unknown_correction_local_dof_ids_;
    bool correction_local_dof_ids_initialized_ = false;

    // Flattened child-interface data. These arrays connect parent outlet unknowns to child inlet
    // pressure and flow dofs while incorporating child subtree relations into the parent block.
    std::vector<int> child_element_index_;
    std::vector<int> pressure_row_;
    std::vector<int> parent_outlet_pressure_local_dof_;
    std::vector<int> child_inlet_pressure_local_dof_;
    std::vector<int> child_inlet_flow_local_dof_;
    std::vector<int> parent_outlet_pressure_unknown_index_;

    // Symbolic coefficient locations used by physics assembly callbacks.
    TreeCoefficientLocation root_boundary_coefficient_;
    std::vector<TreeCoefficientLocation> equation_inlet_pressure_coefficients_;
    std::vector<TreeCoefficientLocation> matrix_coefficients_;
    std::vector<TreeCoefficientLocation> child_pressure_parent_coefficients_;
    std::vector<TreeCoefficientLocation> child_pressure_child_coefficients_;
    std::vector<TreeCoefficientLocation> child_flow_coefficients_;

    // Direct coefficient values matching the symbolic locations above. Static coefficients are
    // written once; dynamic coefficients are replaced before each tree solve.
    double root_boundary_coefficient_value_ = 0.0;
    std::vector<double> equation_inlet_pressure_coefficient_values_;
    std::vector<double> matrix_coefficient_values_;
    std::vector<double> child_pressure_parent_coefficient_values_;
    std::vector<double> child_pressure_child_coefficient_values_;
    std::vector<double> child_flow_coefficient_values_;

    // Row-indexed lookup table from assembly row/dof pairs to direct coefficient storage.
    struct DirectCoefficientEntry
    {
      int local_dof = -1;
      double* value = nullptr;
      const char* context = nullptr;
    };
    std::vector<int> direct_coefficient_row_offsets_;
    std::vector<DirectCoefficientEntry> direct_coefficient_entries_;

    // Reusable work arrays for local block assembly and solves.
    std::vector<double> workspace_matrix_;
    std::vector<double> workspace_rhs_constant_;
    std::vector<double> workspace_rhs_inlet_pressure_;
    std::vector<double> workspace_intercept_;
    std::vector<double> workspace_slope_;
    std::vector<double> child_pressure_slope_;
    std::vector<double> child_pressure_intercept_;
    std::vector<std::string> element_context_;

    // Per-subtree affine relation delta_q_in = G * delta_p_in + h and top-down inlet-pressure
    // corrections. Stamps mark which element corrections are valid in the current solve.
    std::vector<double> subtree_relation_g_;
    std::vector<double> subtree_relation_h_;
    std::vector<double> inlet_pressure_by_element_;
    std::vector<int> inlet_pressure_stamp_;
    int current_solve_stamp_ = 0;
  };

}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
