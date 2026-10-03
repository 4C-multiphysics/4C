// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_tree_linear_solver.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace
  {
    double rhs_value(const Core::LinAlg::Vector<double>& residual, int local_row)
    {
      FOUR_C_ASSERT(local_row >= 0 && local_row < residual.local_length(),
          "TreeNewtonLinearSolver row {} is not locally available.", local_row);
      return -residual.local_values_as_span()[static_cast<std::size_t>(local_row)];
    }

    double required_matrix_value(double value, double tolerance, const std::string& context)
    {
      FOUR_C_ASSERT(std::abs(value) > tolerance,
          "TreeNewtonLinearSolver missing or near-zero matrix coefficient for {}.", context);
      return value;
    }

    /**
     * Solve one dense local block for two right-hand sides: constant term and inlet-pressure slope.
     */
    template <unsigned int n>
    void solve_fixed_size_system(std::span<const double> matrix,
        std::span<const double> rhs_constant, std::span<const double> rhs_inlet_pressure,
        std::span<double> intercept, std::span<double> slope, double pivot_tolerance,
        const std::string& context)
    {
      Core::LinAlg::Matrix<n, n> local_matrix(Core::LinAlg::Initialization::uninitialized);
      constexpr int block_size = static_cast<int>(n);
      FOUR_C_ASSERT(matrix.size() == n * n && rhs_constant.size() == n &&
                        rhs_inlet_pressure.size() == n && intercept.size() == n &&
                        slope.size() == n,
          "TreeNewtonLinearSolver fixed-size system has inconsistent span sizes for {}.", context);
      for (int row = 0; row < block_size; ++row)
      {
        for (int col = 0; col < block_size; ++col)
        {
          local_matrix(static_cast<unsigned int>(row), static_cast<unsigned int>(col)) =
              matrix[static_cast<std::size_t>(row * block_size + col)];
        }
      }

      const double determinant = local_matrix.invert();
      FOUR_C_ASSERT_ALWAYS(std::abs(determinant) > pivot_tolerance,
          "TreeNewtonLinearSolver found a singular or underconstrained local block for {}.",
          context);

      for (int row = 0; row < block_size; ++row)
      {
        double value_a = 0.0;
        double value_b = 0.0;
        for (int col = 0; col < block_size; ++col)
        {
          value_a += local_matrix(static_cast<unsigned int>(row), static_cast<unsigned int>(col)) *
                     rhs_constant[static_cast<std::size_t>(col)];
          value_b += local_matrix(static_cast<unsigned int>(row), static_cast<unsigned int>(col)) *
                     rhs_inlet_pressure[static_cast<std::size_t>(col)];
        }
        intercept[static_cast<std::size_t>(row)] = value_a;
        slope[static_cast<std::size_t>(row)] = value_b;
      }
    }

    void solve_dense_system(std::span<double> matrix, std::span<double> rhs_constant,
        std::span<double> rhs_inlet_pressure, std::span<double> intercept, std::span<double> slope,
        double pivot_tolerance, const std::string& context)
    {
      const int n = static_cast<int>(rhs_constant.size());
      FOUR_C_ASSERT(n > 0, "TreeNewtonLinearSolver dense system has invalid size for {}.", context);
      FOUR_C_ASSERT(rhs_inlet_pressure.size() == rhs_constant.size() &&
                        intercept.size() == rhs_constant.size() &&
                        slope.size() == rhs_constant.size() &&
                        matrix.size() == rhs_constant.size() * rhs_constant.size(),
          "TreeNewtonLinearSolver dense system has inconsistent span sizes for {}.", context);

      if (n == 2)
      {
        solve_fixed_size_system<2>(
            matrix, rhs_constant, rhs_inlet_pressure, intercept, slope, pivot_tolerance, context);
        return;
      }

      if (n == 3)
      {
        solve_fixed_size_system<3>(
            matrix, rhs_constant, rhs_inlet_pressure, intercept, slope, pivot_tolerance, context);
        return;
      }

      FOUR_C_THROW("NewtonTree does not support local block size {} for {}.", n, context);
    }

    int unknown_index_for_global_dof(const std::vector<int>& unknown_global_dof_ids,
        int unknown_begin, int unknown_end, int global_dof_id)
    {
      const auto first = unknown_global_dof_ids.begin() + unknown_begin;
      const auto last = unknown_global_dof_ids.begin() + unknown_end;
      const auto it = std::find(first, last, global_dof_id);
      FOUR_C_ASSERT_ALWAYS(it != last,
          "TreeNewtonLinearSolver recovery data does not contain global dof {}.", global_dof_id);
      return static_cast<int>(std::distance(first, it));
    }

  }  // namespace

  TreeNewtonLinearSolver::TreeNewtonLinearSolver(const TreeNewtonLinearSolverContext& context)
      : tree_metadata_(context.tree_metadata), pivot_tolerance_(context.pivot_tolerance)
  {
    FOUR_C_ASSERT_ALWAYS(pivot_tolerance_ > 0.0,
        "TreeNewtonLinearSolver requires a positive pivot tolerance, got {}.", pivot_tolerance_);
    build_symbolic_plan();
  }

  void TreeNewtonLinearSolver::build_symbolic_plan()
  {
    const auto& root_boundary = tree_root_inlet_boundary(tree_metadata_);
    root_boundary_variable_ = root_boundary.constrained_variable;
    root_boundary_row_ = root_boundary.local_equation_id;
    root_boundary_local_dof_ = root_boundary.local_dof_id;

    const int element_count = static_cast<int>(tree_metadata_.elements.size());
    constexpr int scalar_tree_element_threshold = 7;
    use_scalar_tree_solve_ = element_count <= scalar_tree_element_threshold;
    global_element_id_.assign(static_cast<std::size_t>(element_count), -1);
    inlet_pressure_local_dof_.assign(static_cast<std::size_t>(element_count), -1);
    inlet_flow_unknown_index_.assign(static_cast<std::size_t>(element_count), -1);
    outlet_pressure_unknown_index_.assign(static_cast<std::size_t>(element_count), -1);
    block_size_.assign(static_cast<std::size_t>(element_count), 0);
    child_interface_count_.assign(static_cast<std::size_t>(element_count), 0);
    is_leaf_.assign(static_cast<std::size_t>(element_count), 0);

    unknown_offset_.assign(static_cast<std::size_t>(element_count + 1), 0);
    equation_offset_.assign(static_cast<std::size_t>(element_count + 1), 0);
    child_interface_offset_.assign(static_cast<std::size_t>(element_count + 1), 0);
    matrix_offset_.assign(static_cast<std::size_t>(element_count + 1), 0);

    unknown_global_dof_ids_.clear();
    unknown_local_dof_ids_.clear();
    equation_rows_.clear();
    inlet_pressure_correction_local_dof_ids_.clear();
    unknown_correction_local_dof_ids_.clear();
    correction_local_dof_ids_initialized_ = false;
    child_element_index_.clear();
    pressure_row_.clear();
    parent_outlet_pressure_local_dof_.clear();
    child_inlet_pressure_local_dof_.clear();
    child_inlet_flow_local_dof_.clear();
    parent_outlet_pressure_unknown_index_.clear();
    element_context_.assign(static_cast<std::size_t>(element_count), std::string{});

    subtree_relation_g_.assign(static_cast<std::size_t>(element_count), 0.0);
    subtree_relation_h_.assign(static_cast<std::size_t>(element_count), 0.0);
    inlet_pressure_by_element_.assign(
        static_cast<std::size_t>(element_count), std::numeric_limits<double>::quiet_NaN());
    inlet_pressure_stamp_.assign(static_cast<std::size_t>(element_count), 0);
    current_solve_stamp_ = 0;
    // The tree solve treats each element inlet pressure as the external variable and solves the
    // remaining element dofs as a local dense block parameterized by that inlet pressure.
    int matrix_entry_count = 0;
    for (std::size_t element_index = 0; element_index < tree_metadata_.elements.size();
        ++element_index)
    {
      const int element_index_int = static_cast<int>(element_index);
      const auto& element = tree_metadata_.elements[element_index];
      global_element_id_[element_index] = element.global_element_id;
      inlet_pressure_local_dof_[element_index] = element.local_dof_ids[0];
      is_leaf_[element_index] = element.is_leaf() ? 1u : 0u;
      element_context_[element_index] = "element " + std::to_string(element.global_element_id + 1);

      const int unknown_begin = static_cast<int>(unknown_global_dof_ids_.size());
      unknown_offset_[element_index] = unknown_begin;
      for (int i = 1; i < element.num_dofs; ++i)
      {
        unknown_global_dof_ids_.push_back(element.global_dof_ids[static_cast<std::size_t>(i)]);
        unknown_local_dof_ids_.push_back(element.local_dof_ids[static_cast<std::size_t>(i)]);
      }
      const int unknown_end = static_cast<int>(unknown_global_dof_ids_.size());
      block_size_[element_index] = unknown_end - unknown_begin;
      inlet_flow_unknown_index_[element_index] = unknown_index_for_global_dof(
          unknown_global_dof_ids_, unknown_begin, unknown_end, element.global_dof_ids[2]);
      outlet_pressure_unknown_index_[element_index] = unknown_index_for_global_dof(
          unknown_global_dof_ids_, unknown_begin, unknown_end, element.global_dof_ids[1]);

      matrix_offset_[element_index] = matrix_entry_count;
      matrix_entry_count += block_size_[element_index] * block_size_[element_index];
      matrix_offset_[element_index + 1] = matrix_entry_count;

      equation_offset_[element_index] = static_cast<int>(equation_rows_.size());
      for (int row_offset = 0; row_offset < element.num_state_equations; ++row_offset)
      {
        equation_rows_.push_back(element.first_local_state_equation_id + row_offset);
      }

      child_interface_offset_[element_index] = static_cast<int>(child_element_index_.size());
      if (element.is_leaf())
      {
        const auto& outlet_boundary =
            tree_outlet_boundary_for_element(tree_metadata_, element_index_int);
        equation_rows_.push_back(outlet_boundary.local_equation_id);
      }
      else
      {
        const TreeJunctionMetadata& junction =
            tree_junction_for_parent(tree_metadata_, element_index_int);

        equation_rows_.push_back(junction.first_local_equation_id + junction.child_count);
        child_interface_count_[element_index] = junction.child_count;
        for (int child_slot = 0; child_slot < junction.child_count; ++child_slot)
        {
          const int child_element_index =
              junction.child_element_indices[static_cast<std::size_t>(child_slot)];
          const auto& child =
              tree_metadata_.elements[static_cast<std::size_t>(child_element_index)];
          child_element_index_.push_back(child_element_index);
          pressure_row_.push_back(junction.first_local_equation_id + child_slot);
          parent_outlet_pressure_local_dof_.push_back(element.local_dof_ids[1]);
          child_inlet_pressure_local_dof_.push_back(child.local_dof_ids[0]);
          child_inlet_flow_local_dof_.push_back(child.local_dof_ids[2]);
          parent_outlet_pressure_unknown_index_.push_back(
              outlet_pressure_unknown_index_[element_index]);
        }
      }

      equation_offset_[element_index + 1] = static_cast<int>(equation_rows_.size());
      child_interface_offset_[element_index + 1] = static_cast<int>(child_element_index_.size());
      unknown_offset_[element_index + 1] = static_cast<int>(unknown_global_dof_ids_.size());

      FOUR_C_ASSERT_ALWAYS(equation_offset_[element_index + 1] - equation_offset_[element_index] ==
                               block_size_[element_index],
          "TreeNewtonLinearSolver local block for element {} has {} equations for {} unknowns.",
          element.global_element_id + 1,
          equation_offset_[element_index + 1] - equation_offset_[element_index],
          block_size_[element_index]);
    }

    workspace_matrix_.assign(static_cast<std::size_t>(matrix_entry_count), 0.0);
    workspace_rhs_constant_.assign(unknown_global_dof_ids_.size(), 0.0);
    workspace_rhs_inlet_pressure_.assign(unknown_global_dof_ids_.size(), 0.0);
    workspace_intercept_.assign(unknown_global_dof_ids_.size(), 0.0);
    workspace_slope_.assign(unknown_global_dof_ids_.size(), 0.0);
    child_pressure_slope_.assign(child_element_index_.size(), 0.0);
    child_pressure_intercept_.assign(child_element_index_.size(), 0.0);

    root_boundary_coefficient_ = {root_boundary_row_, root_boundary_local_dof_};
    equation_inlet_pressure_coefficients_.assign(unknown_global_dof_ids_.size(), {});
    matrix_coefficients_.assign(static_cast<std::size_t>(matrix_entry_count), {});
    child_pressure_parent_coefficients_.assign(child_element_index_.size(), {});
    child_pressure_child_coefficients_.assign(child_element_index_.size(), {});
    child_flow_coefficients_.assign(child_element_index_.size(), {});
    root_boundary_coefficient_value_ = 0.0;
    equation_inlet_pressure_coefficient_values_.assign(unknown_global_dof_ids_.size(), 0.0);
    matrix_coefficient_values_.assign(static_cast<std::size_t>(matrix_entry_count), 0.0);
    child_pressure_parent_coefficient_values_.assign(child_element_index_.size(), 0.0);
    child_pressure_child_coefficient_values_.assign(child_element_index_.size(), 0.0);
    child_flow_coefficient_values_.assign(child_element_index_.size(), 0.0);
    for (int element_index = 0; element_index < element_count; ++element_index)
    {
      const std::size_t element_index_size = static_cast<std::size_t>(element_index);
      const int unknown_begin = unknown_offset_[element_index_size];
      const int equation_begin = equation_offset_[element_index_size];
      const int matrix_begin = matrix_offset_[element_index_size];
      const int block_size = block_size_[element_index_size];
      for (int equation_index = 0; equation_index < block_size; ++equation_index)
      {
        const int local_row =
            equation_rows_[static_cast<std::size_t>(equation_begin + equation_index)];
        equation_inlet_pressure_coefficients_[static_cast<std::size_t>(
            unknown_begin + equation_index)] = {
            local_row, inlet_pressure_local_dof_[element_index_size]};
        for (int unknown_index = 0; unknown_index < block_size; ++unknown_index)
        {
          matrix_coefficients_[static_cast<std::size_t>(
              matrix_begin + equation_index * block_size + unknown_index)] = {local_row,
              unknown_local_dof_ids_[static_cast<std::size_t>(unknown_begin + unknown_index)]};
        }
      }

      const int child_begin = child_interface_offset_[element_index_size];
      for (int child_slot = 0; child_slot < child_interface_count_[element_index_size];
          ++child_slot)
      {
        const int child_interface_index = child_begin + child_slot;
        const std::size_t child_interface_index_size =
            static_cast<std::size_t>(child_interface_index);
        const int flow_row =
            equation_rows_[static_cast<std::size_t>(equation_begin + block_size - 1)];
        child_pressure_parent_coefficients_[child_interface_index_size] = {
            pressure_row_[child_interface_index_size],
            parent_outlet_pressure_local_dof_[child_interface_index_size]};
        child_pressure_child_coefficients_[child_interface_index_size] = {
            pressure_row_[child_interface_index_size],
            child_inlet_pressure_local_dof_[child_interface_index_size]};
        child_flow_coefficients_[child_interface_index_size] = {
            flow_row, child_inlet_flow_local_dof_[child_interface_index_size]};
      }
    }

    struct PendingDirectCoefficientEntry
    {
      int local_row = -1;
      int local_dof = -1;
      double* value = nullptr;
      const char* context = nullptr;
    };

    // Direct assembly is indexed by row first, then dof, so physics callbacks can update the SoA
    // coefficient storage without searching all coefficients used by the symbolic plan.
    std::vector<PendingDirectCoefficientEntry> pending_direct_coefficients;
    pending_direct_coefficients.reserve(
        1 + equation_inlet_pressure_coefficients_.size() + matrix_coefficients_.size() +
        child_pressure_parent_coefficients_.size() + child_pressure_child_coefficients_.size() +
        child_flow_coefficients_.size());
    int max_direct_coefficient_row = -1;
    const auto register_direct_coefficient =
        [&](const TreeCoefficientLocation& location, double& value, const char* context)
    {
      FOUR_C_ASSERT_ALWAYS(location.local_row >= 0 && location.local_dof >= 0,
          "TreeNewtonLinearSolver direct coefficient {} has invalid row {} dof {}.", context,
          location.local_row, location.local_dof);
      pending_direct_coefficients.push_back(PendingDirectCoefficientEntry{
          .local_row = location.local_row,
          .local_dof = location.local_dof,
          .value = &value,
          .context = context,
      });
      max_direct_coefficient_row = std::max(max_direct_coefficient_row, location.local_row);
    };
    register_direct_coefficient(
        root_boundary_coefficient_, root_boundary_coefficient_value_, "root inlet boundary");
    for (std::size_t i = 0; i < equation_inlet_pressure_coefficients_.size(); ++i)
    {
      register_direct_coefficient(equation_inlet_pressure_coefficients_[i],
          equation_inlet_pressure_coefficient_values_[i], "equation inlet-pressure coefficient");
    }
    for (std::size_t i = 0; i < matrix_coefficients_.size(); ++i)
    {
      register_direct_coefficient(
          matrix_coefficients_[i], matrix_coefficient_values_[i], "element matrix coefficient");
    }
    for (std::size_t i = 0; i < child_pressure_parent_coefficients_.size(); ++i)
    {
      register_direct_coefficient(child_pressure_parent_coefficients_[i],
          child_pressure_parent_coefficient_values_[i], "child pressure parent coefficient");
    }
    for (std::size_t i = 0; i < child_pressure_child_coefficients_.size(); ++i)
    {
      register_direct_coefficient(child_pressure_child_coefficients_[i],
          child_pressure_child_coefficient_values_[i], "child pressure child coefficient");
    }
    for (std::size_t i = 0; i < child_flow_coefficients_.size(); ++i)
    {
      register_direct_coefficient(
          child_flow_coefficients_[i], child_flow_coefficient_values_[i], "child flow coefficient");
    }

    direct_coefficient_row_offsets_.assign(
        static_cast<std::size_t>(std::max(max_direct_coefficient_row + 2, 1)), 0);
    for (const PendingDirectCoefficientEntry& entry : pending_direct_coefficients)
    {
      ++direct_coefficient_row_offsets_[static_cast<std::size_t>(entry.local_row + 1)];
    }
    for (std::size_t row = 1; row < direct_coefficient_row_offsets_.size(); ++row)
    {
      direct_coefficient_row_offsets_[row] += direct_coefficient_row_offsets_[row - 1];
    }

    direct_coefficient_entries_.assign(
        pending_direct_coefficients.size(), DirectCoefficientEntry{});
    std::vector<int> row_write_positions = direct_coefficient_row_offsets_;
    for (const PendingDirectCoefficientEntry& entry : pending_direct_coefficients)
    {
      const int insert_position = row_write_positions[static_cast<std::size_t>(entry.local_row)]++;
      for (int existing_position =
               direct_coefficient_row_offsets_[static_cast<std::size_t>(entry.local_row)];
          existing_position < insert_position; ++existing_position)
      {
        const DirectCoefficientEntry& existing_entry =
            direct_coefficient_entries_[static_cast<std::size_t>(existing_position)];
        FOUR_C_ASSERT_ALWAYS(existing_entry.local_dof != entry.local_dof,
            "TreeNewtonLinearSolver direct coefficient location row {} dof {} is registered more "
            "than once (existing {}, duplicate {}).",
            entry.local_row, entry.local_dof, existing_entry.context, entry.context);
      }
      direct_coefficient_entries_[static_cast<std::size_t>(insert_position)] =
          DirectCoefficientEntry{
              .local_dof = entry.local_dof,
              .value = entry.value,
              .context = entry.context,
          };
    }

    grouped_element_indices_.clear();
    grouped_unknown_begin_.clear();
    grouped_matrix_begin_.clear();
    grouped_equation_begin_.clear();
    grouped_child_begin_.clear();
    bottom_up_layer_groups_.clear();
    top_down_layer_groups_.clear();
    // Elements in the same traversal layer are independent. Grouping by block size and child count
    // preserves traversal order while sharing loop structure across elements with the same shape.
    const auto build_layer_groups = [&](const std::vector<std::vector<int>>& layers,
                                        std::vector<std::vector<ElementGroup>>& layer_groups)
    {
      layer_groups.clear();
      layer_groups.reserve(layers.size());
      for (const auto& layer : layers)
      {
        std::vector<ElementGroup> shape_keys;
        shape_keys.reserve(layer.size());
        for (const int element_index : layer)
        {
          const std::size_t element_index_size = static_cast<std::size_t>(element_index);
          const int element_block_size = block_size_[element_index_size];
          const int element_child_count = child_interface_count_[element_index_size];
          const auto same_shape = [&](const ElementGroup& group)
          {
            return group.block_size == element_block_size &&
                   group.child_count == element_child_count;
          };
          if (std::none_of(shape_keys.begin(), shape_keys.end(), same_shape))
          {
            ElementGroup shape_key;
            shape_key.block_size = element_block_size;
            shape_key.child_count = element_child_count;
            shape_keys.push_back(shape_key);
          }
        }

        std::sort(shape_keys.begin(), shape_keys.end(),
            [](const ElementGroup& a, const ElementGroup& b)
            {
              if (a.block_size != b.block_size)
              {
                return a.block_size < b.block_size;
              }
              return a.child_count < b.child_count;
            });

        std::vector<ElementGroup> groups;
        groups.reserve(shape_keys.size());
        for (const auto& shape_key : shape_keys)
        {
          ElementGroup group;
          group.begin = static_cast<int>(grouped_element_indices_.size());
          group.block_size = shape_key.block_size;
          group.child_count = shape_key.child_count;
          for (const int element_index : layer)
          {
            const std::size_t element_index_size = static_cast<std::size_t>(element_index);
            if (block_size_[element_index_size] == shape_key.block_size &&
                child_interface_count_[element_index_size] == shape_key.child_count)
            {
              grouped_element_indices_.push_back(element_index);
              grouped_unknown_begin_.push_back(unknown_offset_[element_index_size]);
              grouped_matrix_begin_.push_back(matrix_offset_[element_index_size]);
              grouped_equation_begin_.push_back(equation_offset_[element_index_size]);
              grouped_child_begin_.push_back(child_interface_offset_[element_index_size]);
            }
          }
          group.end = static_cast<int>(grouped_element_indices_.size());
          if (group.begin != group.end)
          {
            groups.push_back(group);
          }
        }
        layer_groups.push_back(groups);
      }
    };
    build_layer_groups(tree_metadata_.bottom_up_layers, bottom_up_layer_groups_);
    build_layer_groups(tree_metadata_.top_down_layers, top_down_layer_groups_);

    FOUR_C_ASSERT_ALWAYS(grouped_unknown_begin_.size() == grouped_element_indices_.size() &&
                             grouped_matrix_begin_.size() == grouped_element_indices_.size() &&
                             grouped_equation_begin_.size() == grouped_element_indices_.size() &&
                             grouped_child_begin_.size() == grouped_element_indices_.size(),
        "TreeNewtonLinearSolver grouped offset caches do not match grouped element indices.");

    const auto validate_grouped_traversal =
        [&](const std::vector<std::vector<ElementGroup>>& groups, const std::string& traversal_name)
    {
      std::vector<int> visit_count(static_cast<std::size_t>(element_count), 0);
      for (const auto& layer_groups : groups)
      {
        for (const auto& group : layer_groups)
        {
          FOUR_C_ASSERT_ALWAYS(
              group.begin >= 0 && group.end >= group.begin &&
                  static_cast<std::size_t>(group.end) <= grouped_element_indices_.size(),
              "TreeNewtonLinearSolver {} group has invalid range [{}, {}).", traversal_name,
              group.begin, group.end);
          for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
          {
            const std::size_t grouped_index_size = static_cast<std::size_t>(grouped_index);
            const int element_index = grouped_element_indices_[grouped_index_size];
            FOUR_C_ASSERT_ALWAYS(element_index >= 0 && element_index < element_count,
                "TreeNewtonLinearSolver {} group references invalid element index {}.",
                traversal_name, element_index);
            const std::size_t element_index_size = static_cast<std::size_t>(element_index);
            FOUR_C_ASSERT_ALWAYS(
                group.block_size == block_size_[element_index_size] &&
                    group.child_count == child_interface_count_[element_index_size],
                "TreeNewtonLinearSolver {} group shape does not match element {}.", traversal_name,
                global_element_id_[element_index_size] + 1);
            FOUR_C_ASSERT_ALWAYS(
                grouped_unknown_begin_[grouped_index_size] == unknown_offset_[element_index_size] &&
                    grouped_matrix_begin_[grouped_index_size] ==
                        matrix_offset_[element_index_size] &&
                    grouped_equation_begin_[grouped_index_size] ==
                        equation_offset_[element_index_size] &&
                    grouped_child_begin_[grouped_index_size] ==
                        child_interface_offset_[element_index_size],
                "TreeNewtonLinearSolver {} grouped offset cache does not match element {}.",
                traversal_name, global_element_id_[element_index_size] + 1);
            ++visit_count[element_index_size];
          }
        }
      }

      for (int element_index = 0; element_index < element_count; ++element_index)
      {
        FOUR_C_ASSERT_ALWAYS(visit_count[static_cast<std::size_t>(element_index)] == 1,
            "TreeNewtonLinearSolver {} grouped traversal visits element {} {} times.",
            traversal_name, global_element_id_[static_cast<std::size_t>(element_index)] + 1,
            visit_count[static_cast<std::size_t>(element_index)]);
      }
    };
    validate_grouped_traversal(bottom_up_layer_groups_, "bottom-up");
    validate_grouped_traversal(top_down_layer_groups_, "top-down");

    std::vector<int> correction_dof_visit_count(
        static_cast<std::size_t>(tree_metadata_.num_global_dofs), 0);
    for (const auto& element : tree_metadata_.elements)
    {
      for (const int global_dof_id : element.global_dof_ids)
      {
        FOUR_C_ASSERT_ALWAYS(global_dof_id >= 0 && global_dof_id < tree_metadata_.num_global_dofs,
            "TreeNewtonLinearSolver correction dof {} is outside [0, {}).", global_dof_id,
            tree_metadata_.num_global_dofs);
        ++correction_dof_visit_count[static_cast<std::size_t>(global_dof_id)];
      }
    }
    for (int global_dof_id = 0; global_dof_id < tree_metadata_.num_global_dofs; ++global_dof_id)
    {
      FOUR_C_ASSERT_ALWAYS(correction_dof_visit_count[static_cast<std::size_t>(global_dof_id)] == 1,
          "TreeNewtonLinearSolver top-down recovery writes correction dof {} {} times.",
          global_dof_id, correction_dof_visit_count[static_cast<std::size_t>(global_dof_id)]);
    }

    int max_top_down_group_size = 0;
    for (const auto& layer_groups : top_down_layer_groups_)
    {
      for (const auto& group : layer_groups)
      {
        max_top_down_group_size = std::max(max_top_down_group_size, group.end - group.begin);
      }
    }
    top_down_inlet_pressure_.assign(static_cast<std::size_t>(max_top_down_group_size), 0.0);
    top_down_unknown_values_.assign(static_cast<std::size_t>(max_top_down_group_size), 0.0);
    top_down_outlet_pressure_.assign(static_cast<std::size_t>(max_top_down_group_size), 0.0);
    top_down_child_pressure_.assign(static_cast<std::size_t>(max_top_down_group_size), 0.0);
  }

  void TreeNewtonLinearSolver::append_value(int local_row_id, int local_dof_id, double value)
  {
    set_direct_coefficient_value(local_row_id, local_dof_id, value, "append");
  }

  void TreeNewtonLinearSolver::replace_value(int local_row_id, int local_dof_id, double value)
  {
    set_direct_coefficient_value(local_row_id, local_dof_id, value, "replace");
  }

  void TreeNewtonLinearSolver::replace_values(std::span<const int> local_row_ids,
      std::span<const int> local_dof_ids, std::span<const double> values)
  {
    FOUR_C_ASSERT(
        local_row_ids.size() == local_dof_ids.size() && local_row_ids.size() == values.size(),
        "TreeNewtonLinearSolver direct coefficient batch replacement size mismatch: rows {}, dofs "
        "{}, values {}.",
        local_row_ids.size(), local_dof_ids.size(), values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
    {
      replace_value(local_row_ids[i], local_dof_ids[i], values[i]);
    }
  }

  void TreeNewtonLinearSolver::set_direct_coefficient_value(
      int local_row_id, int local_dof_id, double value, const char* operation)
  {
    const DirectCoefficientEntry* entry = nullptr;
    if (local_row_id >= 0 &&
        static_cast<std::size_t>(local_row_id) + 1 < direct_coefficient_row_offsets_.size())
    {
      const int begin = direct_coefficient_row_offsets_[static_cast<std::size_t>(local_row_id)];
      const int end = direct_coefficient_row_offsets_[static_cast<std::size_t>(local_row_id + 1)];
      for (int index = begin; index < end; ++index)
      {
        const DirectCoefficientEntry& candidate =
            direct_coefficient_entries_[static_cast<std::size_t>(index)];
        if (candidate.local_dof == local_dof_id)
        {
          entry = &candidate;
          break;
        }
      }
    }
    FOUR_C_ASSERT(entry != nullptr,
        "TreeNewtonLinearSolver direct coefficient {} for row {} dof {} does not match the "
        "serial tree symbolic plan.",
        operation, local_row_id, local_dof_id);
    FOUR_C_ASSERT(entry->value != nullptr,
        "TreeNewtonLinearSolver direct coefficient {} for row {} dof {} has no storage.", operation,
        local_row_id, local_dof_id);
    *entry->value = value;
  }

  void TreeNewtonLinearSolver::validate_solve_inputs(
      const Core::LinAlg::Vector<double>& residual, const Core::LinAlg::Vector<double>& delta) const
  {
    const int comm_size = Core::Communication::num_mpi_ranks(delta.get_comm());
    FOUR_C_ASSERT_ALWAYS(comm_size == 1,
        "TreeNewtonLinearSolver currently supports only serial reduced-lung solves.");
    FOUR_C_ASSERT_ALWAYS(residual.local_length() == tree_metadata_.num_global_equations,
        "TreeNewtonLinearSolver requires all residual rows to be locally available.");
    FOUR_C_ASSERT_ALWAYS(delta.local_length() == tree_metadata_.num_global_dofs,
        "TreeNewtonLinearSolver requires all correction dofs to be locally available.");
  }

  void TreeNewtonLinearSolver::initialize_correction_local_dof_ids(
      const Core::LinAlg::Vector<double>& delta)
  {
    if (!correction_local_dof_ids_initialized_)
    {
      const auto& correction_map = delta.get_map();
      const auto resolve_correction_local_dof = [&](int global_dof_id)
      {
        const int local_dof_id = correction_map.lid(global_dof_id);
        FOUR_C_ASSERT_ALWAYS(local_dof_id >= 0 && local_dof_id < delta.local_length(),
            "TreeNewtonLinearSolver correction dof {} is not locally available.", global_dof_id);
        return local_dof_id;
      };

      inlet_pressure_correction_local_dof_ids_.assign(tree_metadata_.elements.size(), -1);
      for (std::size_t element_index = 0; element_index < tree_metadata_.elements.size();
          ++element_index)
      {
        inlet_pressure_correction_local_dof_ids_[element_index] =
            resolve_correction_local_dof(tree_metadata_.elements[element_index].global_dof_ids[0]);
      }
      unknown_correction_local_dof_ids_.assign(unknown_global_dof_ids_.size(), -1);
      for (std::size_t unknown_index = 0; unknown_index < unknown_global_dof_ids_.size();
          ++unknown_index)
      {
        unknown_correction_local_dof_ids_[unknown_index] =
            resolve_correction_local_dof(unknown_global_dof_ids_[unknown_index]);
      }
      correction_local_dof_ids_initialized_ = true;
    }
  }

  double TreeNewtonLinearSolver::solve_root_inlet_pressure(
      const Core::LinAlg::Vector<double>& residual) const
  {
    const double root_boundary_coeff = required_matrix_value(
        root_boundary_coefficient_value_, pivot_tolerance_, "root inlet boundary");
    const std::size_t root_element_index_size =
        static_cast<std::size_t>(tree_metadata_.root_element_index);
    const double root_boundary_rhs = rhs_value(residual, root_boundary_row_);
    if (root_boundary_variable_ == BoundaryConditions::ConstrainedVariable::Pressure)
    {
      return root_boundary_rhs / root_boundary_coeff;
    }

    if (root_boundary_variable_ == BoundaryConditions::ConstrainedVariable::Flow)
    {
      const double root_subtree_slope = subtree_relation_g_[root_element_index_size];
      FOUR_C_ASSERT_ALWAYS(std::abs(root_subtree_slope) > pivot_tolerance_,
          "TreeNewtonLinearSolver root inlet flow boundary cannot determine the root inlet "
          "pressure because the condensed root flow relation has a near-zero pressure slope.");
      return (root_boundary_rhs / root_boundary_coeff -
                 subtree_relation_h_[root_element_index_size]) /
             root_subtree_slope;
    }

    FOUR_C_THROW("TreeNewtonLinearSolver found an unsupported root inlet boundary type.");
    return 0.0;
  }

  void TreeNewtonLinearSolver::solve_with_coefficients(
      const Core::LinAlg::Vector<double>& residual, Core::LinAlg::Vector<double>& delta)
  {
    const auto add_equation_row =
        [&](int element_index, int equation_index, int local_row, double rhs_shift)
    {
      const int unknown_begin = unknown_offset_[static_cast<std::size_t>(element_index)];
      const int matrix_begin = matrix_offset_[static_cast<std::size_t>(element_index)];
      const int block_size = block_size_[static_cast<std::size_t>(element_index)];
      const int matrix_row_offset = matrix_begin + equation_index * block_size;
      workspace_rhs_constant_[static_cast<std::size_t>(unknown_begin + equation_index)] =
          rhs_value(residual, local_row) - rhs_shift;
      workspace_rhs_inlet_pressure_[static_cast<std::size_t>(unknown_begin + equation_index)] =
          -equation_inlet_pressure_coefficient_values_[static_cast<std::size_t>(
              unknown_begin + equation_index)];

      for (int i = 0; i < block_size; ++i)
      {
        workspace_matrix_[static_cast<std::size_t>(matrix_row_offset + i)] =
            matrix_coefficient_values_[static_cast<std::size_t>(matrix_row_offset + i)];
      }
    };

    const auto assemble_group = [&](const ElementGroup& group)
    {
      for (int equation_index = 0; equation_index < group.block_size; ++equation_index)
      {
        for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
        {
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const std::size_t element_index_size = static_cast<std::size_t>(element_index);
          const int equation_begin = equation_offset_[element_index_size];
          const int local_row =
              equation_rows_[static_cast<std::size_t>(equation_begin + equation_index)];
          add_equation_row(element_index, equation_index, local_row, 0.0);
        }

        if (group.child_count == 0 || equation_index + 1 != group.block_size)
        {
          continue;
        }

        for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
        {
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const std::size_t element_index_size = static_cast<std::size_t>(element_index);
          const int unknown_begin = unknown_offset_[element_index_size];
          const int matrix_begin = matrix_offset_[element_index_size];
          const int matrix_row_offset = matrix_begin + equation_index * group.block_size;
          double rhs_shift = 0.0;
          const int child_begin = child_interface_offset_[element_index_size];
          for (int child_slot = 0; child_slot < group.child_count; ++child_slot)
          {
            const int child_interface_index = child_begin + child_slot;
            const std::size_t child_interface_index_size =
                static_cast<std::size_t>(child_interface_index);
            const std::size_t child_element_index =
                static_cast<std::size_t>(child_element_index_[child_interface_index_size]);

            const double pressure_parent_coeff = required_matrix_value(
                child_pressure_parent_coefficient_values_[child_interface_index_size],
                pivot_tolerance_, "pressure-continuity parent pressure");
            const double pressure_child_coeff = required_matrix_value(
                child_pressure_child_coefficient_values_[child_interface_index_size],
                pivot_tolerance_, "pressure-continuity child pressure");
            const double pressure_rhs =
                rhs_value(residual, pressure_row_[child_interface_index_size]);

            const double child_pressure_slope = -pressure_parent_coeff / pressure_child_coeff;
            const double child_pressure_intercept = pressure_rhs / pressure_child_coeff;
            child_pressure_slope_[child_interface_index_size] = child_pressure_slope;
            child_pressure_intercept_[child_interface_index_size] = child_pressure_intercept;

            const double child_flow_slope =
                subtree_relation_g_[child_element_index] * child_pressure_slope;
            const double child_flow_intercept =
                subtree_relation_g_[child_element_index] * child_pressure_intercept +
                subtree_relation_h_[child_element_index];

            const double flow_child_coeff =
                required_matrix_value(child_flow_coefficient_values_[child_interface_index_size],
                    pivot_tolerance_, "junction flow child-flow coefficient");

            const std::size_t parent_outlet_pressure_index = static_cast<std::size_t>(
                parent_outlet_pressure_unknown_index_[child_interface_index_size]);
            workspace_matrix_[static_cast<std::size_t>(matrix_row_offset) +
                              parent_outlet_pressure_index] += flow_child_coeff * child_flow_slope;
            rhs_shift += flow_child_coeff * child_flow_intercept;
          }

          workspace_rhs_constant_[static_cast<std::size_t>(unknown_begin + equation_index)] -=
              rhs_shift;
        }
      }
    };

    const auto assemble_scalar_element = [&](int element_index)
    {
      const std::size_t element_index_size = static_cast<std::size_t>(element_index);
      const int equation_begin = equation_offset_[element_index_size];
      const int block_size = block_size_[element_index_size];
      for (int equation_index = 0; equation_index < block_size; ++equation_index)
      {
        const int local_row =
            equation_rows_[static_cast<std::size_t>(equation_begin + equation_index)];
        add_equation_row(element_index, equation_index, local_row, 0.0);
      }

      const int child_count = child_interface_count_[element_index_size];
      if (child_count == 0)
      {
        return;
      }

      const int unknown_begin = unknown_offset_[element_index_size];
      const int matrix_begin = matrix_offset_[element_index_size];
      const int matrix_row_offset = matrix_begin + (block_size - 1) * block_size;
      double rhs_shift = 0.0;
      const int child_begin = child_interface_offset_[element_index_size];
      for (int child_slot = 0; child_slot < child_count; ++child_slot)
      {
        const int child_interface_index = child_begin + child_slot;
        const std::size_t child_interface_index_size =
            static_cast<std::size_t>(child_interface_index);
        const std::size_t child_element_index =
            static_cast<std::size_t>(child_element_index_[child_interface_index_size]);

        const double pressure_parent_coeff = required_matrix_value(
            child_pressure_parent_coefficient_values_[child_interface_index_size], pivot_tolerance_,
            "pressure-continuity parent pressure");
        const double pressure_child_coeff = required_matrix_value(
            child_pressure_child_coefficient_values_[child_interface_index_size], pivot_tolerance_,
            "pressure-continuity child pressure");
        const double pressure_rhs = rhs_value(residual, pressure_row_[child_interface_index_size]);

        const double child_pressure_slope = -pressure_parent_coeff / pressure_child_coeff;
        const double child_pressure_intercept = pressure_rhs / pressure_child_coeff;
        child_pressure_slope_[child_interface_index_size] = child_pressure_slope;
        child_pressure_intercept_[child_interface_index_size] = child_pressure_intercept;

        const double child_flow_slope =
            subtree_relation_g_[child_element_index] * child_pressure_slope;
        const double child_flow_intercept =
            subtree_relation_g_[child_element_index] * child_pressure_intercept +
            subtree_relation_h_[child_element_index];

        const double flow_child_coeff =
            required_matrix_value(child_flow_coefficient_values_[child_interface_index_size],
                pivot_tolerance_, "junction flow child-flow coefficient");

        const std::size_t parent_outlet_pressure_index = static_cast<std::size_t>(
            parent_outlet_pressure_unknown_index_[child_interface_index_size]);
        workspace_matrix_[static_cast<std::size_t>(matrix_row_offset) +
                          parent_outlet_pressure_index] += flow_child_coeff * child_flow_slope;
        rhs_shift += flow_child_coeff * child_flow_intercept;
      }

      workspace_rhs_constant_[static_cast<std::size_t>(unknown_begin + block_size - 1)] -=
          rhs_shift;
    };

    const auto solve_scalar_element = [&](int element_index)
    {
      const std::size_t element_index_size = static_cast<std::size_t>(element_index);
      const int unknown_begin = unknown_offset_[element_index_size];
      const int matrix_begin = matrix_offset_[element_index_size];
      const int block_size = block_size_[element_index_size];
      solve_dense_system(std::span<double>(workspace_matrix_.data() + matrix_begin,
                             static_cast<std::size_t>(block_size * block_size)),
          std::span<double>(
              workspace_rhs_constant_.data() + unknown_begin, static_cast<std::size_t>(block_size)),
          std::span<double>(workspace_rhs_inlet_pressure_.data() + unknown_begin,
              static_cast<std::size_t>(block_size)),
          std::span<double>(
              workspace_intercept_.data() + unknown_begin, static_cast<std::size_t>(block_size)),
          std::span<double>(
              workspace_slope_.data() + unknown_begin, static_cast<std::size_t>(block_size)),
          pivot_tolerance_, element_context_[element_index_size]);
    };

    const auto write_subtree_relation = [&](int element_index)
    {
      const std::size_t element_index_size = static_cast<std::size_t>(element_index);
      const int unknown_begin = unknown_offset_[element_index_size];
      subtree_relation_g_[element_index_size] = workspace_slope_[static_cast<std::size_t>(
          unknown_begin + inlet_flow_unknown_index_[element_index_size])];
      subtree_relation_h_[element_index_size] = workspace_intercept_[static_cast<std::size_t>(
          unknown_begin + inlet_flow_unknown_index_[element_index_size])];
    };

    const auto write_subtree_relation_group = [&](const ElementGroup& group)
    {
      for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
      {
        write_subtree_relation(grouped_element_indices_[static_cast<std::size_t>(grouped_index)]);
      }
    };

    const auto execute_bottom_up_pass = [&]()
    {
      // Bottom-up pass: assemble each element block after child relations are known, then
      // condense it to the affine inlet relation consumed by its parent.
      if (use_scalar_tree_solve_)
      {
        for (const auto& layer : tree_metadata_.bottom_up_layers)
        {
          for (const int element_index : layer)
          {
            assemble_scalar_element(element_index);
            solve_scalar_element(element_index);
            write_subtree_relation(element_index);
          }
        }
      }
      else
      {
        for (const auto& layer_groups : bottom_up_layer_groups_)
        {
          for (const auto& group : layer_groups)
          {
            assemble_group(group);
            for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
            {
              solve_scalar_element(
                  grouped_element_indices_[static_cast<std::size_t>(grouped_index)]);
            }
            write_subtree_relation_group(group);
          }
        }
      }
    };
    execute_bottom_up_pass();

    if (current_solve_stamp_ == std::numeric_limits<int>::max())
    {
      std::fill(inlet_pressure_stamp_.begin(), inlet_pressure_stamp_.end(), 0);
      current_solve_stamp_ = 0;
    }
    ++current_solve_stamp_;
    const int solve_stamp = current_solve_stamp_;

    const auto set_inlet_pressure = [&](int element_index, double value)
    {
      const std::size_t element_index_size = static_cast<std::size_t>(element_index);
      FOUR_C_ASSERT(inlet_pressure_stamp_[element_index_size] != solve_stamp,
          "TreeNewtonLinearSolver inlet-pressure correction for element {} was written more than "
          "once.",
          global_element_id_[element_index_size] + 1);
      inlet_pressure_by_element_[element_index_size] = value;
      inlet_pressure_stamp_[element_index_size] = solve_stamp;
    };

    set_inlet_pressure(tree_metadata_.root_element_index, solve_root_inlet_pressure(residual));

    double* const delta_values = delta.get_values();
    const auto set_delta_local_value = [&](int local_dof_id, double value)
    { delta_values[static_cast<std::size_t>(local_dof_id)] = value; };

    // Top-down pass: start from the root inlet-pressure correction, recover local unknowns, and
    // propagate child inlet-pressure corrections through stored pressure-continuity maps.
    const auto recover_top_down_element = [&](int element_index)
    {
      const std::size_t element_index_size = static_cast<std::size_t>(element_index);
      FOUR_C_ASSERT(inlet_pressure_stamp_[element_index_size] == solve_stamp,
          "TreeNewtonLinearSolver missing inlet-pressure correction for element {}.",
          global_element_id_[element_index_size] + 1);
      const double inlet_pressure = inlet_pressure_by_element_[element_index_size];
      set_delta_local_value(
          inlet_pressure_correction_local_dof_ids_[element_index_size], inlet_pressure);

      const int unknown_begin = unknown_offset_[element_index_size];
      const int element_block_size = block_size_[element_index_size];
      for (int unknown_index = 0; unknown_index < element_block_size; ++unknown_index)
      {
        const double value =
            workspace_slope_[static_cast<std::size_t>(unknown_begin + unknown_index)] *
                inlet_pressure +
            workspace_intercept_[static_cast<std::size_t>(unknown_begin + unknown_index)];
        set_delta_local_value(unknown_correction_local_dof_ids_[static_cast<std::size_t>(
                                  unknown_begin + unknown_index)],
            value);
      }

      const int child_count = child_interface_count_[element_index_size];
      if (child_count == 0)
      {
        return;
      }

      const int outlet_pressure_index = outlet_pressure_unknown_index_[element_index_size];
      const double outlet_pressure =
          workspace_slope_[static_cast<std::size_t>(unknown_begin + outlet_pressure_index)] *
              inlet_pressure +
          workspace_intercept_[static_cast<std::size_t>(unknown_begin + outlet_pressure_index)];
      const int child_begin = child_interface_offset_[element_index_size];
      for (int child_slot = 0; child_slot < child_count; ++child_slot)
      {
        const int child_interface_index = child_begin + child_slot;
        const std::size_t child_interface_index_size =
            static_cast<std::size_t>(child_interface_index);
        const double child_pressure =
            child_pressure_slope_[child_interface_index_size] * outlet_pressure +
            child_pressure_intercept_[child_interface_index_size];
        set_inlet_pressure(child_element_index_[child_interface_index_size], child_pressure);
      }
    };

    const auto recover_top_down_group = [&](const ElementGroup& group)
    {
      const int group_size = group.end - group.begin;
      for (int lane = 0; lane < group_size; ++lane)
      {
        const int grouped_index = group.begin + lane;
        const int element_index = grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
        const std::size_t element_index_size = static_cast<std::size_t>(element_index);
        FOUR_C_ASSERT(inlet_pressure_stamp_[element_index_size] == solve_stamp,
            "TreeNewtonLinearSolver missing inlet-pressure correction for element {}.",
            global_element_id_[element_index_size] + 1);
        top_down_inlet_pressure_[static_cast<std::size_t>(lane)] =
            inlet_pressure_by_element_[element_index_size];
      }

      for (int lane = 0; lane < group_size; ++lane)
      {
        const int grouped_index = group.begin + lane;
        const int element_index = grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
        const std::size_t element_index_size = static_cast<std::size_t>(element_index);
        set_delta_local_value(inlet_pressure_correction_local_dof_ids_[element_index_size],
            top_down_inlet_pressure_[static_cast<std::size_t>(lane)]);
      }

      for (int unknown_index = 0; unknown_index < group.block_size; ++unknown_index)
      {
        for (int lane = 0; lane < group_size; ++lane)
        {
          const int grouped_index = group.begin + lane;
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const int unknown_begin = unknown_offset_[static_cast<std::size_t>(element_index)];
          top_down_unknown_values_[static_cast<std::size_t>(lane)] =
              workspace_slope_[static_cast<std::size_t>(unknown_begin + unknown_index)] *
                  top_down_inlet_pressure_[static_cast<std::size_t>(lane)] +
              workspace_intercept_[static_cast<std::size_t>(unknown_begin + unknown_index)];
        }

        for (int lane = 0; lane < group_size; ++lane)
        {
          const int grouped_index = group.begin + lane;
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const int unknown_begin = unknown_offset_[static_cast<std::size_t>(element_index)];
          set_delta_local_value(unknown_correction_local_dof_ids_[static_cast<std::size_t>(
                                    unknown_begin + unknown_index)],
              top_down_unknown_values_[static_cast<std::size_t>(lane)]);
        }
      }

      if (group.child_count == 0)
      {
        return;
      }

      for (int lane = 0; lane < group_size; ++lane)
      {
        const int grouped_index = group.begin + lane;
        const int element_index = grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
        const std::size_t element_index_size = static_cast<std::size_t>(element_index);
        const int unknown_begin = unknown_offset_[element_index_size];
        const int outlet_pressure_index = outlet_pressure_unknown_index_[element_index_size];
        top_down_outlet_pressure_[static_cast<std::size_t>(lane)] =
            workspace_slope_[static_cast<std::size_t>(unknown_begin + outlet_pressure_index)] *
                top_down_inlet_pressure_[static_cast<std::size_t>(lane)] +
            workspace_intercept_[static_cast<std::size_t>(unknown_begin + outlet_pressure_index)];
      }

      for (int child_slot = 0; child_slot < group.child_count; ++child_slot)
      {
        for (int lane = 0; lane < group_size; ++lane)
        {
          const int grouped_index = group.begin + lane;
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const int child_interface_index =
              child_interface_offset_[static_cast<std::size_t>(element_index)] + child_slot;
          top_down_child_pressure_[static_cast<std::size_t>(lane)] =
              child_pressure_slope_[static_cast<std::size_t>(child_interface_index)] *
                  top_down_outlet_pressure_[static_cast<std::size_t>(lane)] +
              child_pressure_intercept_[static_cast<std::size_t>(child_interface_index)];
        }

        for (int lane = 0; lane < group_size; ++lane)
        {
          const int grouped_index = group.begin + lane;
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const int child_interface_index =
              child_interface_offset_[static_cast<std::size_t>(element_index)] + child_slot;
          set_inlet_pressure(child_element_index_[static_cast<std::size_t>(child_interface_index)],
              top_down_child_pressure_[static_cast<std::size_t>(lane)]);
        }
      }
    };

    const auto execute_top_down_recovery = [&]()
    {
      constexpr int top_down_scalar_group_threshold = 2;
      if (use_scalar_tree_solve_)
      {
        for (const auto& layer : tree_metadata_.top_down_layers)
        {
          for (const int element_index : layer)
          {
            recover_top_down_element(element_index);
          }
        }
      }
      else
      {
        for (const auto& layer_groups : top_down_layer_groups_)
        {
          for (const auto& group : layer_groups)
          {
            const int group_size = group.end - group.begin;
            const bool use_scalar_top_down_group = group_size <= top_down_scalar_group_threshold;
            if (use_scalar_top_down_group)
            {
              for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
              {
                recover_top_down_element(
                    grouped_element_indices_[static_cast<std::size_t>(grouped_index)]);
              }
            }
            else
            {
              recover_top_down_group(group);
            }
          }
        }
      }
    };
    execute_top_down_recovery();
  }

  void TreeNewtonLinearSolver::solve(
      const Core::LinAlg::Vector<double>& residual, Core::LinAlg::Vector<double>& delta)
  {
    validate_solve_inputs(residual, delta);
    initialize_correction_local_dof_ids(delta);

    solve_with_coefficients(residual, delta);
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE
