// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_TREE_LINEARIZATION_HPP
#define FOUR_C_REDUCED_LUNG_TREE_LINEARIZATION_HPP

#include "4C_config.hpp"

#include <span>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  /**
   * @brief Interface for assembling structured tree-linearization coefficients.
   */
  class TreeCoefficientAssemblyTarget
  {
   public:
    /**
     * @brief Destroy the tree coefficient assembly target interface.
     */
    virtual ~TreeCoefficientAssemblyTarget() = default;

    /**
     * @brief Append a coefficient entry to a local residual row.
     *
     * @param local_row_id Local residual row id.
     * @param local_dof_id Local dof id on the locally relevant dof map.
     * @param value Coefficient value.
     */
    virtual void append_value(int local_row_id, int local_dof_id, double value) = 0;

    /**
     * @brief Replace an existing coefficient entry in a local residual row.
     *
     * @param local_row_id Local residual row id.
     * @param local_dof_id Local dof id on the locally relevant dof map.
     * @param value New coefficient value.
     */
    virtual void replace_value(int local_row_id, int local_dof_id, double value) = 0;

    /**
     * @brief Replace a batch of existing coefficient entries.
     *
     * @param local_row_ids Local residual row ids.
     * @param local_dof_ids Local dof ids on the locally relevant dof map.
     * @param values New coefficient values.
     */
    virtual void replace_values(std::span<const int> local_row_ids,
        std::span<const int> local_dof_ids, std::span<const double> values);
  };

  /**
   * @brief Structured local derivative storage for the tree Newton linear solver.
   *
   * Coefficients are addressed by local residual row id and local dof id from the reduced-lung
   * locally-relevant dof map. This mirrors the coefficient lookup that the validation tree path
   * currently performs on the completed sparse Jacobian, but avoids constructing that sparse
   * matrix. The generic container is retained as a validation/reference representation; production
   * `NewtonTree` assembles directly into solver-owned structured storage.
   */
  class TreeLinearization : public TreeCoefficientAssemblyTarget
  {
   public:
    /**
     * @brief Construct an empty tree linearization.
     */
    TreeLinearization() = default;

    /**
     * @brief Construct tree-linearization storage with the given dimensions.
     *
     * @param num_rows Number of local residual rows.
     * @param num_dofs Number of local dofs on the locally relevant dof map.
     */
    TreeLinearization(int num_rows, int num_dofs) { reset(num_rows, num_dofs); }

    /**
     * @brief Resize storage and remove all existing coefficient entries.
     *
     * @param num_rows Number of local residual rows.
     * @param num_dofs Number of local dofs on the locally relevant dof map.
     */
    void reset(int num_rows, int num_dofs);

    /**
     * @brief Clear all coefficient entries while keeping the current dimensions.
     */
    void clear_values();

    /**
     * @brief Reserve coefficient storage for one local residual row.
     *
     * @param local_row_id Local residual row id.
     * @param entry_count Number of coefficient entries to reserve.
     */
    void reserve_row_entries(int local_row_id, int entry_count);

    /**
     * @brief Set a coefficient, replacing an existing entry or appending a new one.
     *
     * @param local_row_id Local residual row id.
     * @param local_dof_id Local dof id on the locally relevant dof map.
     * @param value Coefficient value.
     */
    void set_value(int local_row_id, int local_dof_id, double value);

    /**
     * @brief Append a coefficient entry to a local residual row.
     *
     * @param local_row_id Local residual row id.
     * @param local_dof_id Local dof id on the locally relevant dof map.
     * @param value Coefficient value.
     */
    void append_value(int local_row_id, int local_dof_id, double value) override;

    /**
     * @brief Replace an existing coefficient entry in a local residual row.
     *
     * @param local_row_id Local residual row id.
     * @param local_dof_id Local dof id on the locally relevant dof map.
     * @param value New coefficient value.
     */
    void replace_value(int local_row_id, int local_dof_id, double value) override;

    /**
     * @brief Replace a batch of existing coefficient entries.
     *
     * @param local_row_ids Local residual row ids.
     * @param local_dof_ids Local dof ids on the locally relevant dof map.
     * @param values New coefficient values.
     */
    void replace_values(std::span<const int> local_row_ids, std::span<const int> local_dof_ids,
        std::span<const double> values) override;

    /**
     * @brief Return one coefficient value.
     *
     * @param local_row_id Local residual row id.
     * @param local_dof_id Local dof id on the locally relevant dof map.
     * @return Stored coefficient value, or zero if no entry exists.
     */
    [[nodiscard]] double value(int local_row_id, int local_dof_id) const;

    /**
     * @brief Return all coefficient entries for one local residual row.
     *
     * @param local_row_id Local residual row id.
     * @return Stored `(local dof id, coefficient)` entries for the row.
     */
    [[nodiscard]] const std::vector<std::pair<int, double>>& entries(int local_row_id) const;

    /**
     * @brief Number of local residual rows.
     *
     * @return Current row count.
     */
    [[nodiscard]] int num_rows() const { return num_rows_; }

    /**
     * @brief Number of local dofs on the locally relevant dof map.
     *
     * @return Current dof count.
     */
    [[nodiscard]] int num_dofs() const { return num_dofs_; }

   private:
    int num_rows_ = 0;
    int num_dofs_ = 0;
    std::vector<std::vector<std::pair<int, double>>> rows_;
  };
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
