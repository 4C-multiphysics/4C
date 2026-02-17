// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_BOUNDARY_CONDITIONS_HPP
#define FOUR_C_REDUCED_LUNG_BOUNDARY_CONDITIONS_HPP

#include "4C_config.hpp"

#include "4C_reduced_lung_input.hpp"

#include <functional>
#include <map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace Core::LinAlg
{
  class Map;
  class SparseMatrix;
  template <typename T>
  class Vector;
}  // namespace Core::LinAlg

namespace Core::Utils
{
  class FunctionOfTime;
}

namespace ReducedLung
{
  namespace BoundaryConditions
  {
    /**
     * @brief Boundary condition equation types for the reduced lung model.
     */
    enum class Type
    {
      Pressure,
      Flow
    };

    /**
     * @brief Source of the boundary condition value.
     */
    enum class ValueSource
    {
      function_id,
      constant_value
    };

    struct BoundaryConditionModel;

    /**
     * @brief Shared data container for all boundary condition entities.
     *
     * Stores identifiers and dof mappings for the boundary condition equations. This uses a
     * struct-of-arrays layout for efficient assembly loops.
     */
    struct BoundaryConditionData
    {
      // Global node id for each boundary condition.
      std::vector<int> node_id;
      // Global element id attached to the boundary node.
      std::vector<int> global_element_id;
      // Input index for diagnostics (zero-based).
      std::vector<int> input_bc_id;
      // Local boundary condition id.
      std::vector<int> local_bc_id;
      // Local equation id.
      std::vector<int> local_equation_id;
      // Global equation id.
      std::vector<int> global_equation_id;
      // Global dof id for the constrained variable.
      std::vector<int> global_dof_id;
      // Local dof id for the constrained variable.
      std::vector<int> local_dof_id;

      /**
       * @brief Number of entries stored in this data block.
       */
      [[nodiscard]] size_t size() const { return node_id.size(); }
      /**
       * @brief Clear all data arrays.
       */
      void clear();
      /**
       * @brief Reserve storage for @p count entries in all arrays.
       */
      void reserve(size_t count);
      /**
       * @brief Append a new boundary condition entry.
       */
      void add_entry(int node_id_value, int element_id_value, int local_bc_id_value,
          int global_dof_id_value, int input_bc_id_value);
    };

    /**
     * @brief Function handle for evaluating the negative residuals.
     */
    using NegativeResidualEvaluator = std::function<void(const BoundaryConditionModel& model,
        Core::LinAlg::Vector<double>& rhs, const Core::LinAlg::Vector<double>& dofs, double time)>;
    /**
     * @brief Function handle for evaluating the boundary condition Jacobian.
     */
    using JacobianEvaluator = std::function<void(
        const BoundaryConditionModel& model, Core::LinAlg::SparseMatrix& sysmat)>;

    /**
     * @brief Boundary condition model containing a homogeneous set of equations.
     *
     * Boundary conditions are grouped by type and value source. Constant values are stored per
     * entry, whereas function-based models share a function pointer evaluated at assembly time.
     */
    struct BoundaryConditionModel
    {
      Type type;
      ValueSource value_source;
      int function_id = 0;
      // Cached function pointer for time-dependent boundary conditions.
      const Core::Utils::FunctionOfTime* function = nullptr;
      // Per-entry constant values (only used for constant value sources or potential mixed
      // conditions in the future).
      std::vector<double> values;
      BoundaryConditionData data;
      NegativeResidualEvaluator negative_residual_evaluator;
      JacobianEvaluator jacobian_evaluator;

      /**
       * @brief Add a boundary condition entry to this model.
       */
      void add_condition(int node_id_value, int element_id_value, int local_bc_id_value,
          int global_dof_id_value, int input_bc_id_value, double value);
    };

    /**
     * @brief Container for all boundary condition models.
     */
    struct BoundaryConditionContainer
    {
      std::vector<BoundaryConditionModel> models;
    };

    /*!
     * @brief Create boundary condition models from the input parameters and topology.
     *
     * This function groups boundary conditions by type and value source, maps boundary nodes to
     * the connected element, and assigns the constrained dof id.
     */
    void create_boundary_conditions(const Core::FE::Discretization& discretization,
        const ReducedLungParameters& parameters,
        const std::map<int, std::vector<int>>& global_ele_ids_per_node,
        const std::map<int, int>& global_dof_per_ele,
        const std::map<int, int>& first_global_dof_of_ele,
        BoundaryConditionContainer& boundary_conditions);

    /**
     * @brief Count the total number of boundary condition equations.
     */
    int count_boundary_conditions(const BoundaryConditionContainer& boundary_conditions);

    /**
     * @brief Assign local equation ids for all boundary condition equations.
     */
    void assign_local_equation_ids(
        BoundaryConditionContainer& boundary_conditions, int& n_local_equations);

    /**
     * @brief Assign global equation ids from the row map.
     */
    void assign_global_equation_ids(
        const Core::LinAlg::Map& row_map, BoundaryConditionContainer& boundary_conditions);

    /**
     * @brief Assign local dof ids from the locally relevant dof map.
     */
    void assign_local_dof_ids(const Core::LinAlg::Map& locally_relevant_dof_map,
        BoundaryConditionContainer& boundary_conditions);

    /**
     * @brief Create evaluator function objects for all models.
     */
    void create_evaluators(BoundaryConditionContainer& boundary_conditions);

    /**
     * @brief Assemble the boundary condition residual contributions.
     */
    void update_negative_residual_vector(Core::LinAlg::Vector<double>& rhs,
        const BoundaryConditionContainer& boundary_conditions,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double time);

    /**
     * @brief Assemble the boundary condition Jacobian contributions once.
     */
    void update_jacobian(
        Core::LinAlg::SparseMatrix& sysmat, const BoundaryConditionContainer& boundary_conditions);
  }  // namespace BoundaryConditions
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
