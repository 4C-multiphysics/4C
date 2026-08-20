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

#include <mpi.h>

#include <cstdint>
#include <functional>
#include <map>
#include <variant>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace ReducedLung::TerminalUnits
{
  struct TerminalUnitContainer;
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
  class FunctionManager;
  class FunctionOfTime;
}  // namespace Core::Utils

namespace ReducedLung
{
  namespace BoundaryConditions
  {
    /**
     * @brief The solution variable a boundary condition constrains.
     *
     * At most one condition may constrain a given variable on a given node.
     */
    enum class ConstrainedVariable : std::uint8_t
    {
      Pressure,
      Flow,
    };

    struct BoundaryConditionModel;

    /**
     * @brief State the boundary condition evaluators may depend on.
     */
    struct AssemblyContext
    {
      double time = 0.0;
      double total_terminal_unit_volume = 0.0;
    };

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
      void add_entry(
          int node_id_value, int element_id_value, int local_bc_id_value, int global_dof_id_value);
    };

    /**
     * @brief Function handle for evaluating the residuals.
     */
    using ResidualEvaluator =
        std::function<void(const BoundaryConditionModel& model, Core::LinAlg::Vector<double>& rhs,
            const Core::LinAlg::Vector<double>& dofs, const AssemblyContext& assembly_context)>;
    /**
     * @brief Function handle for evaluating the boundary condition Jacobian.
     */
    using JacobianEvaluator =
        std::function<void(const BoundaryConditionModel& model, Core::LinAlg::SparseMatrix& sysmat,
            const Core::LinAlg::Vector<double>& dofs, const AssemblyContext& assembly_context)>;

    /**
     * @brief A prescribed value that follows a function of time.
     */
    struct TimeFunction
    {
      int function_id = 0;
      //! Never null after create_boundary_conditions().
      const Core::Utils::FunctionOfTime* function = nullptr;
    };

    /**
     * @brief A pleural pressure evaluated from the total volume of all terminal units.
     */
    struct VolumeDependentPleuralPressure
    {
      ReducedLungParameters::BoundaryConditions::VolumeDependentPleuralPressureDefinition
          parameters;
    };

    /**
     * @brief How a boundary condition computes the value it prescribes.
     */
    using ValueModel = std::variant<TimeFunction, VolumeDependentPleuralPressure>;

    /**
     * @brief Boundary condition model containing a homogeneous set of equations.
     *
     * All equations of one model evaluate to the same value, so a model holds the entries of
     * exactly one input definition.
     */
    struct BoundaryConditionModel
    {
      //! The `id` of the input definition all entries of this model come from.
      int definition_id = 0;
      ConstrainedVariable constrained_variable = ConstrainedVariable::Pressure;
      ValueModel value_model;
      BoundaryConditionData data;
      ResidualEvaluator residual_evaluator;
      JacobianEvaluator jacobian_evaluator;
      //! False if the Jacobian row depends on the current iterate and must be reassembled on
      //! every call.
      bool has_constant_jacobian = true;

      /**
       * @brief Add a boundary condition entry to this model.
       */
      void add_condition(
          int node_id_value, int element_id_value, int local_bc_id_value, int global_dof_id_value);
    };

    /**
     * @brief Container for all boundary condition models.
     */
    struct BoundaryConditionContainer
    {
      std::vector<BoundaryConditionModel> models;

      //! Total volume of all terminal units of the last converged timestep.
      double total_terminal_unit_volume = 0.0;
      //! Whether any condition needs the volume; the collective reduction is skipped otherwise.
      bool requires_total_terminal_unit_volume = false;
    };

    /*!
     * @brief Create boundary condition models from the input parameters and the mesh.
     *
     * Every definition of the input applies to all nodes that carry its id in the `bc_id` point
     * data of the mesh. This function creates one model per definition, maps boundary nodes to
     * the connected element, and assigns the constrained dof id.
     */
    void create_boundary_conditions(const Core::FE::Discretization& discretization,
        const ReducedLungParameters& parameters, const std::map<int, std::vector<int>>& bc_nodes,
        const std::map<int, std::vector<int>>& global_ele_ids_per_node,
        const std::map<int, int>& global_dof_per_ele,
        const std::map<int, int>& first_global_dof_of_ele,
        const Core::Utils::FunctionManager& function_manager,
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
    void update_residual_vector(Core::LinAlg::Vector<double>& rhs,
        const BoundaryConditionContainer& boundary_conditions,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double time);

    /**
     * @brief Assemble the boundary condition Jacobian contributions.
     */
    void update_jacobian(Core::LinAlg::SparseMatrix& sysmat,
        const BoundaryConditionContainer& boundary_conditions,
        const Core::LinAlg::Vector<double>& locally_relevant_dofs, double time);

    /**
     * @brief Refresh the total terminal unit volume the boundary conditions evaluate against.
     *
     * Must be called once per timestep, before the Newton solve, and collectively on all ranks.
     */
    void refresh_total_terminal_unit_volume(BoundaryConditionContainer& boundary_conditions,
        const TerminalUnits::TerminalUnitContainer& terminal_units, MPI_Comm comm);

    /**
     * @brief Sum the volume of all terminal units across all ranks. Collective.
     */
    double compute_total_terminal_unit_volume(
        const TerminalUnits::TerminalUnitContainer& terminal_units, MPI_Comm comm);
  }  // namespace BoundaryConditions
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif
