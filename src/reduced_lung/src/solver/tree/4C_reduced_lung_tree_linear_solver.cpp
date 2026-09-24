// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_reduced_lung_tree_linear_solver.hpp"

#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_reduced_lung_solver_profiles.hpp"
#include "4C_reduced_lung_tree_linearization.hpp"
#include "4C_utils_exceptions.hpp"

#include <mpi.h>
#include <Teuchos_TimeMonitor.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#if defined(__has_include)
#if __has_include(<experimental/simd>)
#include <experimental/simd>
#define FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD 1
#else
#define FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD 0
#endif
#else
#define FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD 0
#endif

FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  namespace
  {
    using Clock = std::chrono::steady_clock;

    namespace tree_solver_simd
    {
#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
      namespace stdx = std::experimental;

      using Double = stdx::native_simd<double>;
      using Mask = typename Double::mask_type;

      [[maybe_unused]] constexpr bool available = true;

      [[nodiscard, maybe_unused]] constexpr int width() { return static_cast<int>(Double::size()); }

      [[nodiscard, maybe_unused]] int full_chunk_end(int begin, int end)
      {
        const int count = end - begin;
        return begin + count / width() * width();
      }

      [[nodiscard, maybe_unused]] int padded_chunk_end(int begin, int end)
      {
        const int count = end - begin;
        return begin + (count + width() - 1) / width() * width();
      }

      [[nodiscard, maybe_unused]] bool has_full_chunk(int begin, int end)
      {
        return end - begin >= width();
      }

      template <typename Load>
      [[nodiscard, maybe_unused]] Double gather(int grouped_begin, Load&& load)
      {
        return Double([&](auto lane) { return load(grouped_begin + static_cast<int>(lane)); });
      }

      template <typename Load>
      [[nodiscard, maybe_unused]] Double gather_or(
          int grouped_begin, int valid_end, double pad_value, Load&& load)
      {
        return Double(
            [&](auto lane)
            {
              const int grouped_index = grouped_begin + static_cast<int>(lane);
              return grouped_index < valid_end ? load(grouped_index) : pad_value;
            });
      }

      template <typename Store>
      [[maybe_unused]] void scatter(const Double& values, int grouped_begin, Store&& store)
      {
        for (int lane = 0; lane < width(); ++lane)
        {
          store(grouped_begin + lane, values[static_cast<std::size_t>(lane)]);
        }
      }

      template <typename Store>
      [[maybe_unused]] void scatter_valid(
          const Double& values, int grouped_begin, int valid_end, Store&& store)
      {
        for (int lane = 0; lane < width(); ++lane)
        {
          const int grouped_index = grouped_begin + lane;
          if (grouped_index < valid_end)
          {
            store(grouped_index, values[static_cast<std::size_t>(lane)]);
          }
        }
      }

      template <typename Function>
      [[maybe_unused]] void for_each_valid_lane(int grouped_begin, int valid_end, Function&& fn)
      {
        for (int lane = 0; lane < width(); ++lane)
        {
          const int grouped_index = grouped_begin + lane;
          if (grouped_index < valid_end)
          {
            fn(grouped_index, lane);
          }
        }
      }
#else
      [[maybe_unused]] constexpr bool available = false;

      [[nodiscard, maybe_unused]] constexpr int width() { return 1; }

      [[nodiscard, maybe_unused]] constexpr int full_chunk_end(int begin, int /*end*/)
      {
        return begin;
      }

      [[nodiscard, maybe_unused]] constexpr int padded_chunk_end(int begin, int /*end*/)
      {
        return begin;
      }

      [[nodiscard, maybe_unused]] constexpr bool has_full_chunk(int /*begin*/, int /*end*/)
      {
        return false;
      }
#endif
    }  // namespace tree_solver_simd

    double elapsed_seconds(const Clock::time_point start)
    {
      return std::chrono::duration<double>(Clock::now() - start).count();
    }

    /**
     * Coefficient provider for the validation path using a completed sparse Jacobian.
     */
    class SparseTreeCoefficientProvider
    {
     public:
      explicit SparseTreeCoefficientProvider(
          const Core::LinAlg::SparseMatrix& jacobian, TreeNewtonLinearSolverProfile* profile)
          : jacobian_(jacobian), profile_(profile)
      {
      }

      [[nodiscard]] double value(int local_row, int local_col, double tolerance) const
      {
        const auto lookup_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
        FOUR_C_ASSERT_ALWAYS(local_row >= 0 && local_row < jacobian_.num_my_rows(),
            "TreeNewtonLinearSolver matrix row {} is not locally available.", local_row);
        FOUR_C_ASSERT_ALWAYS(local_col >= 0,
            "TreeNewtonLinearSolver matrix column is not locally available for row {}.", local_row);

        int n_entries = 0;
        double* values = nullptr;
        int* columns = nullptr;
        jacobian_.extract_my_row_view(local_row, n_entries, values, columns);

        for (int i = 0; i < n_entries; ++i)
        {
          if (columns[i] == local_col)
          {
            record_lookup(lookup_start);
            return values[i];
          }
        }

        (void)tolerance;
        record_lookup(lookup_start);
        return 0.0;
      }

      [[nodiscard]] double value(const TreeCoefficientLocation& location, double tolerance) const
      {
        return value(location.local_row, location.local_dof, tolerance);
      }

      [[nodiscard]] double value(
          const TreeCoefficientLocation& location, double direct_value, double tolerance) const
      {
        (void)direct_value;
        return value(location, tolerance);
      }

     private:
      void record_lookup(const Clock::time_point lookup_start) const
      {
        if (profile_ != nullptr)
        {
          profile_->coefficient_lookup_time += elapsed_seconds(lookup_start);
          ++profile_->coefficient_lookup_count;
        }
      }

      const Core::LinAlg::SparseMatrix& jacobian_;
      TreeNewtonLinearSolverProfile* profile_ = nullptr;
    };

    /**
     * Coefficient provider for row-oriented TreeLinearization storage.
     */
    class StructuredTreeCoefficientProvider
    {
     public:
      explicit StructuredTreeCoefficientProvider(
          const TreeLinearization& linearization, TreeNewtonLinearSolverProfile* profile)
          : linearization_(linearization), profile_(profile)
      {
      }

      [[nodiscard]] double value(int local_row, int local_col, double tolerance) const
      {
        (void)tolerance;
        const auto lookup_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
        const double coefficient = linearization_.value(local_row, local_col);
        record_lookup(lookup_start);
        return coefficient;
      }

      [[nodiscard]] double value(const TreeCoefficientLocation& location, double tolerance) const
      {
        (void)tolerance;
        const auto lookup_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
        FOUR_C_ASSERT_ALWAYS(
            location.local_row >= 0 && location.local_row < linearization_.num_rows(),
            "TreeNewtonLinearSolver structured coefficient row {} is outside [0, {}).",
            location.local_row, linearization_.num_rows());
        FOUR_C_ASSERT_ALWAYS(
            location.local_dof >= 0 && location.local_dof < linearization_.num_dofs(),
            "TreeNewtonLinearSolver structured coefficient dof {} is outside [0, {}).",
            location.local_dof, linearization_.num_dofs());
        if (location.structured_entry_index < 0)
        {
          record_lookup(lookup_start);
          return 0.0;
        }

        const auto& row = linearization_.entries(location.local_row);
        FOUR_C_ASSERT_ALWAYS(location.structured_entry_index < static_cast<int>(row.size()),
            "TreeNewtonLinearSolver structured coefficient entry index {} is outside row {} size "
            "{}.",
            location.structured_entry_index, location.local_row, row.size());
        const auto& coefficient = row[static_cast<std::size_t>(location.structured_entry_index)];
        FOUR_C_ASSERT_ALWAYS(coefficient.first == location.local_dof,
            "TreeNewtonLinearSolver structured coefficient location for row {} expected dof {} but "
            "found dof {}.",
            location.local_row, location.local_dof, coefficient.first);
        record_lookup(lookup_start);
        return coefficient.second;
      }

      [[nodiscard]] double value(
          const TreeCoefficientLocation& location, double direct_value, double tolerance) const
      {
        (void)location;
        (void)tolerance;
        return direct_value;
      }

     private:
      void record_lookup(const Clock::time_point lookup_start) const
      {
        if (profile_ != nullptr)
        {
          profile_->coefficient_lookup_time += elapsed_seconds(lookup_start);
          ++profile_->coefficient_lookup_count;
        }
      }

      const TreeLinearization& linearization_;
      TreeNewtonLinearSolverProfile* profile_ = nullptr;
    };

    /**
     * Coefficient provider for the fast path where physics modules write directly into solver SoA.
     */
    class DirectTreeCoefficientProvider
    {
     public:
      [[nodiscard]] double value(int local_row, int local_col, double tolerance) const
      {
        (void)tolerance;
        FOUR_C_THROW(
            "TreeNewtonLinearSolver direct coefficient provider cannot look up row {} dof {}.",
            local_row, local_col);
      }

      [[nodiscard]] double value(const TreeCoefficientLocation& location, double tolerance) const
      {
        (void)tolerance;
        FOUR_C_THROW(
            "TreeNewtonLinearSolver direct coefficient provider cannot look up row {} dof {}.",
            location.local_row, location.local_dof);
      }

      [[nodiscard]] double value(
          const TreeCoefficientLocation& location, double direct_value, double tolerance) const
      {
        (void)location;
        (void)tolerance;
        return direct_value;
      }
    };

    const TreeJunctionMetadata* find_junction_for_parent(
        const ReducedLungTreeMetadata& tree_metadata, int parent_element_index)
    {
      for (const auto& junction : tree_metadata.junctions)
      {
        if (junction.parent_element_index == parent_element_index)
        {
          return &junction;
        }
      }
      return nullptr;
    }

    std::vector<const TreeBoundaryConditionMetadata*> outlet_boundaries_for_element(
        const ReducedLungTreeMetadata& tree_metadata, int element_index)
    {
      std::vector<const TreeBoundaryConditionMetadata*> boundaries;
      for (const auto& boundary : tree_metadata.boundary_conditions)
      {
        if (boundary.element_index == element_index && boundary.side == TreeBoundarySide::Outlet)
        {
          boundaries.push_back(&boundary);
        }
      }
      return boundaries;
    }

    const TreeBoundaryConditionMetadata& root_inlet_boundary(
        const ReducedLungTreeMetadata& tree_metadata)
    {
      const TreeBoundaryConditionMetadata* root_boundary = nullptr;
      for (const auto& boundary : tree_metadata.boundary_conditions)
      {
        if (boundary.element_index == tree_metadata.root_element_index &&
            boundary.side == TreeBoundarySide::Inlet)
        {
          FOUR_C_ASSERT_ALWAYS(root_boundary == nullptr,
              "TreeNewtonLinearSolver found multiple root inlet boundary conditions.");
          root_boundary = &boundary;
        }
      }
      FOUR_C_ASSERT_ALWAYS(
          root_boundary != nullptr, "TreeNewtonLinearSolver requires a root inlet boundary.");
      return *root_boundary;
    }

    double rhs_value(const Core::LinAlg::Vector<double>& residual, int local_row)
    {
      FOUR_C_ASSERT_ALWAYS(local_row >= 0 && local_row < residual.local_length(),
          "TreeNewtonLinearSolver row {} is not locally available.", local_row);
      return -residual.local_values_as_span()[static_cast<std::size_t>(local_row)];
    }

    template <typename CoefficientProvider>
    double matrix_value(
        const CoefficientProvider& coefficients, int local_row, int local_col, double tolerance)
    {
      return coefficients.value(local_row, local_col, tolerance);
    }

    template <typename CoefficientProvider>
    double matrix_value(const CoefficientProvider& coefficients,
        const TreeCoefficientLocation& location, double tolerance)
    {
      return coefficients.value(location, tolerance);
    }

    template <typename CoefficientProvider>
    double matrix_value(const CoefficientProvider& coefficients,
        const TreeCoefficientLocation& location, double direct_value, double tolerance)
    {
      return coefficients.value(location, direct_value, tolerance);
    }

    template <typename CoefficientProvider>
    double required_matrix_value(const CoefficientProvider& coefficients, int local_row,
        int local_col, double tolerance, const std::string& context)
    {
      const double value = matrix_value(coefficients, local_row, local_col, tolerance);
      FOUR_C_ASSERT_ALWAYS(std::abs(value) > tolerance,
          "TreeNewtonLinearSolver missing or near-zero matrix coefficient for {}.", context);
      return value;
    }

    template <typename CoefficientProvider>
    double required_matrix_value(const CoefficientProvider& coefficients,
        const TreeCoefficientLocation& location, double tolerance, const std::string& context)
    {
      const double value = matrix_value(coefficients, location, tolerance);
      FOUR_C_ASSERT_ALWAYS(std::abs(value) > tolerance,
          "TreeNewtonLinearSolver missing or near-zero matrix coefficient for {}.", context);
      return value;
    }

    template <typename CoefficientProvider>
    double required_matrix_value(const CoefficientProvider& coefficients,
        const TreeCoefficientLocation& location, double direct_value, double tolerance,
        const std::string& context)
    {
      const double value = matrix_value(coefficients, location, direct_value, tolerance);
      FOUR_C_ASSERT_ALWAYS(std::abs(value) > tolerance,
          "TreeNewtonLinearSolver missing or near-zero matrix coefficient for {}.", context);
      return value;
    }

    /**
     * Solve one dense local block for two right-hand sides: constant term and inlet-pressure slope.
     */
    void solve_dense_system(double* matrix, double* rhs_a, double* rhs_b, double* solution_a,
        double* solution_b, int n, double pivot_tolerance, const std::string& context)
    {
      const auto matrix_entry = [matrix, n](int row, int col) -> double&
      { return matrix[static_cast<std::size_t>(row * n + col)]; };
      FOUR_C_ASSERT_ALWAYS(
          n > 0, "TreeNewtonLinearSolver dense system has invalid size for {}.", context);

      for (int pivot_col = 0; pivot_col < n; ++pivot_col)
      {
        int pivot_row = pivot_col;
        double pivot_abs = std::abs(matrix_entry(pivot_col, pivot_col));
        for (int row = pivot_col + 1; row < n; ++row)
        {
          const double candidate_abs = std::abs(matrix_entry(row, pivot_col));
          if (candidate_abs > pivot_abs)
          {
            pivot_abs = candidate_abs;
            pivot_row = row;
          }
        }

        FOUR_C_ASSERT_ALWAYS(pivot_abs > pivot_tolerance,
            "TreeNewtonLinearSolver found a singular or underconstrained local block for {}.",
            context);

        if (pivot_row != pivot_col)
        {
          for (int col = 0; col < n; ++col)
          {
            std::swap(matrix_entry(pivot_col, col), matrix_entry(pivot_row, col));
          }
          std::swap(rhs_a[pivot_col], rhs_a[pivot_row]);
          std::swap(rhs_b[pivot_col], rhs_b[pivot_row]);
        }

        const double pivot = matrix_entry(pivot_col, pivot_col);
        for (int row = pivot_col + 1; row < n; ++row)
        {
          const double factor = matrix_entry(row, pivot_col) / pivot;
          if (std::abs(factor) <= std::numeric_limits<double>::epsilon())
          {
            continue;
          }
          matrix_entry(row, pivot_col) = 0.0;
          for (int col = pivot_col + 1; col < n; ++col)
          {
            matrix_entry(row, col) -= factor * matrix_entry(pivot_col, col);
          }
          rhs_a[row] -= factor * rhs_a[pivot_col];
          rhs_b[row] -= factor * rhs_b[pivot_col];
        }
      }

      for (int row = n - 1; row >= 0; --row)
      {
        double value_a = rhs_a[row];
        double value_b = rhs_b[row];
        for (int col = row + 1; col < n; ++col)
        {
          const double entry = matrix_entry(row, col);
          value_a -= entry * solution_a[col];
          value_b -= entry * solution_b[col];
        }
        const double diagonal = matrix_entry(row, row);
        solution_a[row] = value_a / diagonal;
        solution_b[row] = value_b / diagonal;
      }
    }

    /**
     * Solve a batch of 2x2 local blocks, using SIMD when available and dense fallback if needed.
     */
    void solve_2x2_batch(int group_begin, int group_end,
        const std::vector<int>& grouped_element_indices, std::vector<double>& a00,
        std::vector<double>& a01, std::vector<double>& a10, std::vector<double>& a11,
        std::vector<double>& rhs_constant0, std::vector<double>& rhs_constant1,
        std::vector<double>& rhs_inlet_pressure0, std::vector<double>& rhs_inlet_pressure1,
        std::vector<double>& intercept0, std::vector<double>& intercept1,
        std::vector<double>& slope0, std::vector<double>& slope1, std::vector<int>& fallback_lanes,
        TreeNewtonLinearSolverProfile* profile, double pivot_tolerance,
        bool error_on_dense_fallback, const std::vector<std::string>& element_context)
    {
      const int group_size = group_end - group_begin;
      FOUR_C_ASSERT_ALWAYS(group_size >= 0, "TreeNewtonLinearSolver 2x2 batch has invalid range.");
      FOUR_C_ASSERT_ALWAYS(static_cast<int>(grouped_element_indices.size()) >= group_end,
          "TreeNewtonLinearSolver 2x2 grouped cache is too small.");
      FOUR_C_ASSERT_ALWAYS(static_cast<int>(a00.size()) >= group_size &&
                               static_cast<int>(a01.size()) >= group_size &&
                               static_cast<int>(a10.size()) >= group_size &&
                               static_cast<int>(a11.size()) >= group_size &&
                               static_cast<int>(rhs_constant0.size()) >= group_size &&
                               static_cast<int>(rhs_constant1.size()) >= group_size &&
                               static_cast<int>(rhs_inlet_pressure0.size()) >= group_size &&
                               static_cast<int>(rhs_inlet_pressure1.size()) >= group_size &&
                               static_cast<int>(intercept0.size()) >= group_size &&
                               static_cast<int>(intercept1.size()) >= group_size &&
                               static_cast<int>(slope0.size()) >= group_size &&
                               static_cast<int>(slope1.size()) >= group_size,
          "TreeNewtonLinearSolver 2x2 SoA batch workspace is too small.");
      FOUR_C_ASSERT_ALWAYS(static_cast<int>(fallback_lanes.size()) >= group_size,
          "TreeNewtonLinearSolver 2x2 batch workspace is too small.");

      int fallback_count = 0;
      const auto solve_direct_lane = [&](int lane)
      {
        const std::size_t lane_index = static_cast<std::size_t>(lane);
        const double determinant =
            a00[lane_index] * a11[lane_index] - a01[lane_index] * a10[lane_index];
        if (!std::isfinite(determinant) || std::abs(determinant) <= pivot_tolerance)
        {
          fallback_lanes[static_cast<std::size_t>(fallback_count)] = lane;
          ++fallback_count;
          return;
        }

        const double inverse_determinant = 1.0 / determinant;
        intercept0[lane_index] = (rhs_constant0[lane_index] * a11[lane_index] -
                                     a01[lane_index] * rhs_constant1[lane_index]) *
                                 inverse_determinant;
        intercept1[lane_index] = (a00[lane_index] * rhs_constant1[lane_index] -
                                     rhs_constant0[lane_index] * a10[lane_index]) *
                                 inverse_determinant;
        slope0[lane_index] = (rhs_inlet_pressure0[lane_index] * a11[lane_index] -
                                 a01[lane_index] * rhs_inlet_pressure1[lane_index]) *
                             inverse_determinant;
        slope1[lane_index] = (a00[lane_index] * rhs_inlet_pressure1[lane_index] -
                                 rhs_inlet_pressure0[lane_index] * a10[lane_index]) *
                             inverse_determinant;
      };

      int lane = 0;
#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
      const int padded_end = tree_solver_simd::padded_chunk_end(0, group_size);
      if (profile != nullptr && group_size > 0)
      {
        ++profile->simd_group_count;
        profile->simd_lane_count += static_cast<std::uint64_t>(group_size);
      }

      for (; lane < padded_end; lane += tree_solver_simd::width())
      {
        const int chunk_begin = lane;
        const tree_solver_simd::Double a00_values =
            tree_solver_simd::gather_or(chunk_begin, group_size, 1.0,
                [&](int lane_index) { return a00[static_cast<std::size_t>(lane_index)]; });
        const tree_solver_simd::Double a01_values =
            tree_solver_simd::gather_or(chunk_begin, group_size, 0.0,
                [&](int lane_index) { return a01[static_cast<std::size_t>(lane_index)]; });
        const tree_solver_simd::Double a10_values =
            tree_solver_simd::gather_or(chunk_begin, group_size, 0.0,
                [&](int lane_index) { return a10[static_cast<std::size_t>(lane_index)]; });
        const tree_solver_simd::Double a11_values =
            tree_solver_simd::gather_or(chunk_begin, group_size, 1.0,
                [&](int lane_index) { return a11[static_cast<std::size_t>(lane_index)]; });
        const tree_solver_simd::Double rhs_constant0_values =
            tree_solver_simd::gather_or(chunk_begin, group_size, 0.0, [&](int lane_index)
                { return rhs_constant0[static_cast<std::size_t>(lane_index)]; });
        const tree_solver_simd::Double rhs_constant1_values =
            tree_solver_simd::gather_or(chunk_begin, group_size, 0.0, [&](int lane_index)
                { return rhs_constant1[static_cast<std::size_t>(lane_index)]; });
        const tree_solver_simd::Double rhs_inlet_pressure0_values =
            tree_solver_simd::gather_or(chunk_begin, group_size, 0.0, [&](int lane_index)
                { return rhs_inlet_pressure0[static_cast<std::size_t>(lane_index)]; });
        const tree_solver_simd::Double rhs_inlet_pressure1_values =
            tree_solver_simd::gather_or(chunk_begin, group_size, 0.0, [&](int lane_index)
                { return rhs_inlet_pressure1[static_cast<std::size_t>(lane_index)]; });

        const tree_solver_simd::Double determinant =
            a00_values * a11_values - a01_values * a10_values;
        const auto valid_determinant =
            std::experimental::isfinite(determinant) &&
            std::experimental::abs(determinant) > tree_solver_simd::Double(pivot_tolerance);
        if (!std::experimental::all_of(valid_determinant))
        {
          tree_solver_simd::for_each_valid_lane(chunk_begin, group_size,
              [&](int lane_index, int /*chunk_lane*/)
              {
                fallback_lanes[static_cast<std::size_t>(fallback_count)] = lane_index;
                ++fallback_count;
              });
          continue;
        }

        const tree_solver_simd::Double inverse_determinant =
            tree_solver_simd::Double(1.0) / determinant;
        const tree_solver_simd::Double intercept0_values =
            (rhs_constant0_values * a11_values - a01_values * rhs_constant1_values) *
            inverse_determinant;
        const tree_solver_simd::Double intercept1_values =
            (a00_values * rhs_constant1_values - rhs_constant0_values * a10_values) *
            inverse_determinant;
        const tree_solver_simd::Double slope0_values =
            (rhs_inlet_pressure0_values * a11_values - a01_values * rhs_inlet_pressure1_values) *
            inverse_determinant;
        const tree_solver_simd::Double slope1_values =
            (a00_values * rhs_inlet_pressure1_values - rhs_inlet_pressure0_values * a10_values) *
            inverse_determinant;

        tree_solver_simd::scatter_valid(intercept0_values, chunk_begin, group_size,
            [&](int lane_index, double value)
            { intercept0[static_cast<std::size_t>(lane_index)] = value; });
        tree_solver_simd::scatter_valid(intercept1_values, chunk_begin, group_size,
            [&](int lane_index, double value)
            { intercept1[static_cast<std::size_t>(lane_index)] = value; });
        tree_solver_simd::scatter_valid(slope0_values, chunk_begin, group_size,
            [&](int lane_index, double value)
            { slope0[static_cast<std::size_t>(lane_index)] = value; });
        tree_solver_simd::scatter_valid(slope1_values, chunk_begin, group_size,
            [&](int lane_index, double value)
            { slope1[static_cast<std::size_t>(lane_index)] = value; });
      }
#endif

      if (profile != nullptr && lane < group_size)
      {
        profile->scalar_tail_lane_count += static_cast<std::uint64_t>(group_size - lane);
      }
      for (; lane < group_size; ++lane)
      {
        solve_direct_lane(lane);
      }

      if (profile != nullptr)
      {
        profile->dense_fallback_count += static_cast<std::uint64_t>(fallback_count);
      }
      FOUR_C_ASSERT_ALWAYS(!error_on_dense_fallback || fallback_count == 0,
          "TreeNewtonLinearSolver 2x2 batch requires dense fallback for {} lanes in group [{}, "
          "{}).",
          fallback_count, group_begin, group_end);
      for (int fallback_index = 0; fallback_index < fallback_count; ++fallback_index)
      {
        const int lane = fallback_lanes[static_cast<std::size_t>(fallback_index)];
        const int grouped_index = group_begin + lane;
        const int element_index = grouped_element_indices[static_cast<std::size_t>(grouped_index)];
        const std::size_t lane_index = static_cast<std::size_t>(lane);
        std::array<double, 4> local_matrix = {
            a00[lane_index], a01[lane_index], a10[lane_index], a11[lane_index]};
        std::array<double, 2> local_rhs_constant = {
            rhs_constant0[lane_index], rhs_constant1[lane_index]};
        std::array<double, 2> local_rhs_inlet_pressure = {
            rhs_inlet_pressure0[lane_index], rhs_inlet_pressure1[lane_index]};
        std::array<double, 2> local_intercept = {0.0, 0.0};
        std::array<double, 2> local_slope = {0.0, 0.0};
        solve_dense_system(local_matrix.data(), local_rhs_constant.data(),
            local_rhs_inlet_pressure.data(), local_intercept.data(), local_slope.data(), 2,
            pivot_tolerance, element_context[static_cast<std::size_t>(element_index)]);
        intercept0[lane_index] = local_intercept[0];
        intercept1[lane_index] = local_intercept[1];
        slope0[lane_index] = local_slope[0];
        slope1[lane_index] = local_slope[1];
      }
    }

    /**
     * Solve a batch of 3x3 local blocks, using SIMD when available and dense fallback if needed.
     */
    void solve_3x3_batch(int group_begin, int group_end,
        const std::vector<int>& grouped_element_indices, std::vector<double>& a00,
        std::vector<double>& a01, std::vector<double>& a02, std::vector<double>& a10,
        std::vector<double>& a11, std::vector<double>& a12, std::vector<double>& a20,
        std::vector<double>& a21, std::vector<double>& a22, std::vector<double>& rhs_constant0,
        std::vector<double>& rhs_constant1, std::vector<double>& rhs_constant2,
        std::vector<double>& rhs_inlet_pressure0, std::vector<double>& rhs_inlet_pressure1,
        std::vector<double>& rhs_inlet_pressure2, std::vector<double>& intercept0,
        std::vector<double>& intercept1, std::vector<double>& intercept2,
        std::vector<double>& slope0, std::vector<double>& slope1, std::vector<double>& slope2,
        std::vector<int>& fallback_lanes, TreeNewtonLinearSolverProfile* profile,
        double pivot_tolerance, bool error_on_dense_fallback,
        const std::vector<std::string>& element_context)
    {
      const int group_size = group_end - group_begin;
      FOUR_C_ASSERT_ALWAYS(group_size >= 0, "TreeNewtonLinearSolver 3x3 batch has invalid range.");
      FOUR_C_ASSERT_ALWAYS(static_cast<int>(grouped_element_indices.size()) >= group_end,
          "TreeNewtonLinearSolver 3x3 grouped cache is too small.");
      FOUR_C_ASSERT_ALWAYS(static_cast<int>(a00.size()) >= group_size &&
                               static_cast<int>(a01.size()) >= group_size &&
                               static_cast<int>(a02.size()) >= group_size &&
                               static_cast<int>(a10.size()) >= group_size &&
                               static_cast<int>(a11.size()) >= group_size &&
                               static_cast<int>(a12.size()) >= group_size &&
                               static_cast<int>(a20.size()) >= group_size &&
                               static_cast<int>(a21.size()) >= group_size &&
                               static_cast<int>(a22.size()) >= group_size &&
                               static_cast<int>(rhs_constant0.size()) >= group_size &&
                               static_cast<int>(rhs_constant1.size()) >= group_size &&
                               static_cast<int>(rhs_constant2.size()) >= group_size &&
                               static_cast<int>(rhs_inlet_pressure0.size()) >= group_size &&
                               static_cast<int>(rhs_inlet_pressure1.size()) >= group_size &&
                               static_cast<int>(rhs_inlet_pressure2.size()) >= group_size &&
                               static_cast<int>(intercept0.size()) >= group_size &&
                               static_cast<int>(intercept1.size()) >= group_size &&
                               static_cast<int>(intercept2.size()) >= group_size &&
                               static_cast<int>(slope0.size()) >= group_size &&
                               static_cast<int>(slope1.size()) >= group_size &&
                               static_cast<int>(slope2.size()) >= group_size &&
                               static_cast<int>(fallback_lanes.size()) >= group_size,
          "TreeNewtonLinearSolver 3x3 batch workspace is too small.");

      int fallback_count = 0;
      const auto record_fallback_lane = [&](int lane)
      { fallback_lanes[static_cast<std::size_t>(fallback_count++)] = lane; };
      const auto write_solution = [&](int lane, double intercept_value0, double intercept_value1,
                                      double intercept_value2, double slope_value0,
                                      double slope_value1, double slope_value2)
      {
        const std::size_t lane_index = static_cast<std::size_t>(lane);
        intercept0[lane_index] = intercept_value0;
        intercept1[lane_index] = intercept_value1;
        intercept2[lane_index] = intercept_value2;
        slope0[lane_index] = slope_value0;
        slope1[lane_index] = slope_value1;
        slope2[lane_index] = slope_value2;
      };
      const auto safe_pivot = [pivot_tolerance](double value)
      { return std::isfinite(value) && std::abs(value) > pivot_tolerance; };
      const auto solve_3x3_direct_lane = [&](int lane)
      {
        const std::size_t lane_index = static_cast<std::size_t>(lane);
        const double p0 = a00[lane_index];
        if (!safe_pivot(p0))
        {
          record_fallback_lane(lane);
          return;
        }

        const double l10 = a10[lane_index] / p0;
        const double l20 = a20[lane_index] / p0;
        const double u01 = a01[lane_index];
        const double u02 = a02[lane_index];
        const double p1 = a11[lane_index] - l10 * u01;
        const double u12 = a12[lane_index] - l10 * u02;
        const double u21 = a21[lane_index] - l20 * u01;
        const double u22 = a22[lane_index] - l20 * u02;
        if (!std::isfinite(l10) || !std::isfinite(l20) || !safe_pivot(p1) || !std::isfinite(u12) ||
            !std::isfinite(u21) || !std::isfinite(u22))
        {
          record_fallback_lane(lane);
          return;
        }

        const double l21 = u21 / p1;
        const double p2 = u22 - l21 * u12;
        if (!std::isfinite(l21) || !safe_pivot(p2))
        {
          record_fallback_lane(lane);
          return;
        }

        const auto solve_rhs = [&](double rhs0, double rhs1, double rhs2, double& value0,
                                   double& value1, double& value2)
        {
          const double y0 = rhs0;
          const double y1 = rhs1 - l10 * y0;
          const double y2 = rhs2 - l20 * y0 - l21 * y1;
          value2 = y2 / p2;
          value1 = (y1 - u12 * value2) / p1;
          value0 = (y0 - u01 * value1 - u02 * value2) / p0;
        };

        double intercept_value0 = 0.0;
        double intercept_value1 = 0.0;
        double intercept_value2 = 0.0;
        double slope_value0 = 0.0;
        double slope_value1 = 0.0;
        double slope_value2 = 0.0;
        solve_rhs(rhs_constant0[lane_index], rhs_constant1[lane_index], rhs_constant2[lane_index],
            intercept_value0, intercept_value1, intercept_value2);
        solve_rhs(rhs_inlet_pressure0[lane_index], rhs_inlet_pressure1[lane_index],
            rhs_inlet_pressure2[lane_index], slope_value0, slope_value1, slope_value2);
        if (!std::isfinite(intercept_value0) || !std::isfinite(intercept_value1) ||
            !std::isfinite(intercept_value2) || !std::isfinite(slope_value0) ||
            !std::isfinite(slope_value1) || !std::isfinite(slope_value2))
        {
          record_fallback_lane(lane);
          return;
        }

        write_solution(lane, intercept_value0, intercept_value1, intercept_value2, slope_value0,
            slope_value1, slope_value2);
      };

      int lane = 0;
#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
      const auto load_batch = [](const std::vector<double>& values, int lane)
      { return values[static_cast<std::size_t>(lane)]; };
      const auto simd_safe_pivot = [&](const tree_solver_simd::Double& values)
      {
        return std::experimental::isfinite(values) &&
               std::experimental::abs(values) > tree_solver_simd::Double(pivot_tolerance);
      };
      const auto simd_finite = [](const tree_solver_simd::Double& values)
      { return std::experimental::isfinite(values); };
      const auto scatter_batch = [](const tree_solver_simd::Double& values, int lane_begin,
                                     int valid_end, std::vector<double>& output)
      {
        tree_solver_simd::scatter_valid(values, lane_begin, valid_end,
            [&](int lane_index, double value)
            { output[static_cast<std::size_t>(lane_index)] = value; });
      };
      const int padded_end = tree_solver_simd::padded_chunk_end(0, group_size);
      if (profile != nullptr && group_size > 0)
      {
        ++profile->simd_group_count;
        profile->simd_lane_count += static_cast<std::uint64_t>(group_size);
      }
      for (; lane < padded_end; lane += tree_solver_simd::width())
      {
        const tree_solver_simd::Double p0 = tree_solver_simd::gather_or(
            lane, group_size, 1.0, [&](int lane_index) { return load_batch(a00, lane_index); });
        const tree_solver_simd::Double a01_values = tree_solver_simd::gather_or(
            lane, group_size, 0.0, [&](int lane_index) { return load_batch(a01, lane_index); });
        const tree_solver_simd::Double a02_values = tree_solver_simd::gather_or(
            lane, group_size, 0.0, [&](int lane_index) { return load_batch(a02, lane_index); });
        const tree_solver_simd::Double a10_values = tree_solver_simd::gather_or(
            lane, group_size, 0.0, [&](int lane_index) { return load_batch(a10, lane_index); });
        const tree_solver_simd::Double a11_values = tree_solver_simd::gather_or(
            lane, group_size, 1.0, [&](int lane_index) { return load_batch(a11, lane_index); });
        const tree_solver_simd::Double a12_values = tree_solver_simd::gather_or(
            lane, group_size, 0.0, [&](int lane_index) { return load_batch(a12, lane_index); });
        const tree_solver_simd::Double a20_values = tree_solver_simd::gather_or(
            lane, group_size, 0.0, [&](int lane_index) { return load_batch(a20, lane_index); });
        const tree_solver_simd::Double a21_values = tree_solver_simd::gather_or(
            lane, group_size, 0.0, [&](int lane_index) { return load_batch(a21, lane_index); });
        const tree_solver_simd::Double a22_values = tree_solver_simd::gather_or(
            lane, group_size, 1.0, [&](int lane_index) { return load_batch(a22, lane_index); });

        const tree_solver_simd::Double l10 = a10_values / p0;
        const tree_solver_simd::Double l20 = a20_values / p0;
        const tree_solver_simd::Double p1 = a11_values - l10 * a01_values;
        const tree_solver_simd::Double u12 = a12_values - l10 * a02_values;
        const tree_solver_simd::Double u21 = a21_values - l20 * a01_values;
        const tree_solver_simd::Double u22 = a22_values - l20 * a02_values;
        const tree_solver_simd::Double l21 = u21 / p1;
        const tree_solver_simd::Double p2 = u22 - l21 * u12;

        const auto valid_pivots = simd_safe_pivot(p0) && simd_finite(l10) && simd_finite(l20) &&
                                  simd_safe_pivot(p1) && simd_finite(u12) && simd_finite(u21) &&
                                  simd_finite(u22) && simd_finite(l21) && simd_safe_pivot(p2);

        const auto solve_rhs =
            [&](const tree_solver_simd::Double& rhs0, const tree_solver_simd::Double& rhs1,
                const tree_solver_simd::Double& rhs2, tree_solver_simd::Double& value0,
                tree_solver_simd::Double& value1, tree_solver_simd::Double& value2)
        {
          const tree_solver_simd::Double y0 = rhs0;
          const tree_solver_simd::Double y1 = rhs1 - l10 * y0;
          const tree_solver_simd::Double y2 = rhs2 - l20 * y0 - l21 * y1;
          value2 = y2 / p2;
          value1 = (y1 - u12 * value2) / p1;
          value0 = (y0 - a01_values * value1 - a02_values * value2) / p0;
        };

        const tree_solver_simd::Double rhs_constant0_values = tree_solver_simd::gather_or(lane,
            group_size, 0.0, [&](int lane_index) { return load_batch(rhs_constant0, lane_index); });
        const tree_solver_simd::Double rhs_constant1_values = tree_solver_simd::gather_or(lane,
            group_size, 0.0, [&](int lane_index) { return load_batch(rhs_constant1, lane_index); });
        const tree_solver_simd::Double rhs_constant2_values = tree_solver_simd::gather_or(lane,
            group_size, 0.0, [&](int lane_index) { return load_batch(rhs_constant2, lane_index); });
        const tree_solver_simd::Double rhs_inlet_pressure0_values =
            tree_solver_simd::gather_or(lane, group_size, 0.0,
                [&](int lane_index) { return load_batch(rhs_inlet_pressure0, lane_index); });
        const tree_solver_simd::Double rhs_inlet_pressure1_values =
            tree_solver_simd::gather_or(lane, group_size, 0.0,
                [&](int lane_index) { return load_batch(rhs_inlet_pressure1, lane_index); });
        const tree_solver_simd::Double rhs_inlet_pressure2_values =
            tree_solver_simd::gather_or(lane, group_size, 0.0,
                [&](int lane_index) { return load_batch(rhs_inlet_pressure2, lane_index); });

        tree_solver_simd::Double intercept_value0;
        tree_solver_simd::Double intercept_value1;
        tree_solver_simd::Double intercept_value2;
        tree_solver_simd::Double slope_value0;
        tree_solver_simd::Double slope_value1;
        tree_solver_simd::Double slope_value2;
        solve_rhs(rhs_constant0_values, rhs_constant1_values, rhs_constant2_values,
            intercept_value0, intercept_value1, intercept_value2);
        solve_rhs(rhs_inlet_pressure0_values, rhs_inlet_pressure1_values,
            rhs_inlet_pressure2_values, slope_value0, slope_value1, slope_value2);

        const auto valid_solution = valid_pivots && simd_finite(intercept_value0) &&
                                    simd_finite(intercept_value1) &&
                                    simd_finite(intercept_value2) && simd_finite(slope_value0) &&
                                    simd_finite(slope_value1) && simd_finite(slope_value2);
        if (!std::experimental::all_of(valid_solution))
        {
          tree_solver_simd::for_each_valid_lane(lane, group_size,
              [&](int lane_index, int /*chunk_lane*/) { record_fallback_lane(lane_index); });
          continue;
        }

        scatter_batch(intercept_value0, lane, group_size, intercept0);
        scatter_batch(intercept_value1, lane, group_size, intercept1);
        scatter_batch(intercept_value2, lane, group_size, intercept2);
        scatter_batch(slope_value0, lane, group_size, slope0);
        scatter_batch(slope_value1, lane, group_size, slope1);
        scatter_batch(slope_value2, lane, group_size, slope2);
      }
#endif

      if (profile != nullptr && lane < group_size)
      {
        profile->scalar_tail_lane_count += static_cast<std::uint64_t>(group_size - lane);
      }
      for (; lane < group_size; ++lane)
      {
        solve_3x3_direct_lane(lane);
      }

      if (profile != nullptr)
      {
        profile->dense_fallback_count += static_cast<std::uint64_t>(fallback_count);
      }
      FOUR_C_ASSERT_ALWAYS(!error_on_dense_fallback || fallback_count == 0,
          "TreeNewtonLinearSolver 3x3 batch requires dense fallback for {} lanes in group [{}, "
          "{}).",
          fallback_count, group_begin, group_end);
      for (int fallback_index = 0; fallback_index < fallback_count; ++fallback_index)
      {
        const int lane = fallback_lanes[static_cast<std::size_t>(fallback_index)];
        const int grouped_index = group_begin + lane;
        const int element_index = grouped_element_indices[static_cast<std::size_t>(grouped_index)];
        const std::size_t lane_index = static_cast<std::size_t>(lane);
        std::array<double, 9> local_matrix = {a00[lane_index], a01[lane_index], a02[lane_index],
            a10[lane_index], a11[lane_index], a12[lane_index], a20[lane_index], a21[lane_index],
            a22[lane_index]};
        std::array<double, 3> local_rhs_constant = {
            rhs_constant0[lane_index], rhs_constant1[lane_index], rhs_constant2[lane_index]};
        std::array<double, 3> local_rhs_inlet_pressure = {rhs_inlet_pressure0[lane_index],
            rhs_inlet_pressure1[lane_index], rhs_inlet_pressure2[lane_index]};
        std::array<double, 3> local_intercept = {0.0, 0.0, 0.0};
        std::array<double, 3> local_slope = {0.0, 0.0, 0.0};
        solve_dense_system(local_matrix.data(), local_rhs_constant.data(),
            local_rhs_inlet_pressure.data(), local_intercept.data(), local_slope.data(), 3,
            pivot_tolerance, element_context[static_cast<std::size_t>(element_index)]);
        write_solution(lane, local_intercept[0], local_intercept[1], local_intercept[2],
            local_slope[0], local_slope[1], local_slope[2]);
      }
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
      : tree_metadata_(context.tree_metadata),
        pivot_tolerance_(context.pivot_tolerance),
        coefficient_source_(context.coefficient_source),
        validation_jacobian_(context.validation_jacobian),
        profile_(context.profile),
        force_batch_tree_solve_(context.force_batch_tree_solve),
        error_on_dense_fallback_(context.error_on_dense_fallback)
  {
    FOUR_C_ASSERT_ALWAYS(pivot_tolerance_ > 0.0,
        "TreeNewtonLinearSolver requires a positive pivot tolerance, got {}.", pivot_tolerance_);
    FOUR_C_ASSERT_ALWAYS(
        coefficient_source_ != TreeNewtonLinearSolverCoefficientSource::SparseJacobian ||
            validation_jacobian_ != nullptr,
        "TreeNewtonLinearSolver sparse validation requires a Jacobian matrix.");
    FOUR_C_ASSERT_ALWAYS(
        coefficient_source_ != TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks ||
            validation_jacobian_ == nullptr,
        "TreeNewtonLinearSolver structured production solves must not receive a sparse Jacobian.");
    build_symbolic_plan();
  }

  void TreeNewtonLinearSolver::build_symbolic_plan()
  {
    const auto& root_boundary = root_inlet_boundary(tree_metadata_);
    root_boundary_variable_ = root_boundary.constrained_variable;
    root_boundary_row_ = root_boundary.local_equation_id;
    root_boundary_local_dof_ = root_boundary.local_dof_id;

    const int element_count = static_cast<int>(tree_metadata_.elements.size());
    constexpr int scalar_tree_element_threshold = 7;
    use_scalar_tree_solve_ =
        !force_batch_tree_solve_ && element_count <= scalar_tree_element_threshold;
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

    subtree_relation_G_.assign(static_cast<std::size_t>(element_count), 0.0);
    subtree_relation_h_.assign(static_cast<std::size_t>(element_count), 0.0);
    inlet_pressure_by_element_.assign(
        static_cast<std::size_t>(element_count), std::numeric_limits<double>::quiet_NaN());
    inlet_pressure_stamp_.assign(static_cast<std::size_t>(element_count), 0);
    current_solve_stamp_ = 0;
    if (profile_ != nullptr)
    {
      profile_->element_count = 0;
      profile_->total_local_block_dofs = 0;
      profile_->max_local_block_size = 0;
    }

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
        const auto outlet_boundaries =
            outlet_boundaries_for_element(tree_metadata_, element_index_int);
        FOUR_C_ASSERT_ALWAYS(outlet_boundaries.size() == 1u,
            "TreeNewtonLinearSolver requires exactly one outlet boundary for leaf element {}.",
            element.global_element_id + 1);
        equation_rows_.push_back(outlet_boundaries.front()->local_equation_id);
      }
      else
      {
        const TreeJunctionMetadata* junction =
            find_junction_for_parent(tree_metadata_, element_index_int);
        FOUR_C_ASSERT_ALWAYS(junction != nullptr,
            "TreeNewtonLinearSolver found no junction metadata for parent element {}.",
            element.global_element_id + 1);
        FOUR_C_ASSERT_ALWAYS(junction->child_count <= 2,
            "TreeNewtonLinearSolver supports at most two child interfaces per element {}.",
            element.global_element_id + 1);

        equation_rows_.push_back(junction->first_local_equation_id + junction->child_count);
        child_interface_count_[element_index] = junction->child_count;
        for (int child_slot = 0; child_slot < junction->child_count; ++child_slot)
        {
          const int child_element_index =
              junction->child_element_indices[static_cast<std::size_t>(child_slot)];
          const auto& child =
              tree_metadata_.elements[static_cast<std::size_t>(child_element_index)];
          child_element_index_.push_back(child_element_index);
          pressure_row_.push_back(junction->first_local_equation_id + child_slot);
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

      if (profile_ != nullptr)
      {
        ++profile_->element_count;
        profile_->total_local_block_dofs += block_size_[element_index];
        profile_->max_local_block_size =
            std::max(profile_->max_local_block_size, block_size_[element_index]);
      }
    }

    workspace_matrix_.assign(static_cast<std::size_t>(matrix_entry_count), 0.0);
    workspace_rhs_constant_.assign(unknown_global_dof_ids_.size(), 0.0);
    workspace_rhs_inlet_pressure_.assign(unknown_global_dof_ids_.size(), 0.0);
    workspace_intercept_.assign(unknown_global_dof_ids_.size(), 0.0);
    workspace_slope_.assign(unknown_global_dof_ids_.size(), 0.0);
    child_pressure_slope_.assign(child_element_index_.size(), 0.0);
    child_pressure_intercept_.assign(child_element_index_.size(), 0.0);

    root_boundary_coefficient_ = {root_boundary_row_, root_boundary_local_dof_, -1};
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
            local_row, inlet_pressure_local_dof_[element_index_size], -1};
        for (int unknown_index = 0; unknown_index < block_size; ++unknown_index)
        {
          matrix_coefficients_[static_cast<std::size_t>(
              matrix_begin + equation_index * block_size + unknown_index)] = {local_row,
              unknown_local_dof_ids_[static_cast<std::size_t>(unknown_begin + unknown_index)], -1};
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
            parent_outlet_pressure_local_dof_[child_interface_index_size], -1};
        child_pressure_child_coefficients_[child_interface_index_size] = {
            pressure_row_[child_interface_index_size],
            child_inlet_pressure_local_dof_[child_interface_index_size], -1};
        child_flow_coefficients_[child_interface_index_size] = {
            flow_row, child_inlet_flow_local_dof_[child_interface_index_size], -1};
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
    // lets the common 2x2 and 3x3 cases use batched/SIMD kernels while preserving traversal order.
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

    int max_2x2_group_size = 0;
    int max_3x3_group_size = 0;
    int max_top_down_group_size = 0;
    for (const auto& layer_groups : bottom_up_layer_groups_)
    {
      for (const auto& group : layer_groups)
      {
        if (group.block_size == 2)
        {
          max_2x2_group_size = std::max(max_2x2_group_size, group.end - group.begin);
        }
        else if (group.block_size == 3)
        {
          max_3x3_group_size = std::max(max_3x3_group_size, group.end - group.begin);
        }
      }
    }
    for (const auto& layer_groups : top_down_layer_groups_)
    {
      for (const auto& group : layer_groups)
      {
        max_top_down_group_size = std::max(max_top_down_group_size, group.end - group.begin);
      }
    }
    batch_2x2_fallback_lanes_.assign(static_cast<std::size_t>(max_2x2_group_size), 0);
    batch_2x2_a00_.assign(static_cast<std::size_t>(max_2x2_group_size), 0.0);
    batch_2x2_a01_.assign(static_cast<std::size_t>(max_2x2_group_size), 0.0);
    batch_2x2_a10_.assign(static_cast<std::size_t>(max_2x2_group_size), 0.0);
    batch_2x2_a11_.assign(static_cast<std::size_t>(max_2x2_group_size), 0.0);
    batch_2x2_rhs_constant0_.assign(static_cast<std::size_t>(max_2x2_group_size), 0.0);
    batch_2x2_rhs_constant1_.assign(static_cast<std::size_t>(max_2x2_group_size), 0.0);
    batch_2x2_rhs_inlet_pressure0_.assign(static_cast<std::size_t>(max_2x2_group_size), 0.0);
    batch_2x2_rhs_inlet_pressure1_.assign(static_cast<std::size_t>(max_2x2_group_size), 0.0);
    batch_2x2_intercept0_.assign(static_cast<std::size_t>(max_2x2_group_size), 0.0);
    batch_2x2_intercept1_.assign(static_cast<std::size_t>(max_2x2_group_size), 0.0);
    batch_2x2_slope0_.assign(static_cast<std::size_t>(max_2x2_group_size), 0.0);
    batch_2x2_slope1_.assign(static_cast<std::size_t>(max_2x2_group_size), 0.0);
    batch_3x3_a00_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_a01_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_a02_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_a10_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_a11_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_a12_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_a20_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_a21_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_a22_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_rhs_constant0_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_rhs_constant1_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_rhs_constant2_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_rhs_inlet_pressure0_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_rhs_inlet_pressure1_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_rhs_inlet_pressure2_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_intercept0_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_intercept1_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_intercept2_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_slope0_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_slope1_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_slope2_.assign(static_cast<std::size_t>(max_3x3_group_size), 0.0);
    batch_3x3_fallback_lanes_.assign(static_cast<std::size_t>(max_3x3_group_size), 0);
    top_down_inlet_pressure_.assign(static_cast<std::size_t>(max_top_down_group_size), 0.0);
    top_down_unknown_values_.assign(static_cast<std::size_t>(max_top_down_group_size), 0.0);
    top_down_outlet_pressure_.assign(static_cast<std::size_t>(max_top_down_group_size), 0.0);
    top_down_child_pressure_.assign(static_cast<std::size_t>(max_top_down_group_size), 0.0);
  }

  NewtonLinearizationType TreeNewtonLinearSolver::linearization_type() const
  {
    if (coefficient_source_ == TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks)
    {
      return NewtonLinearizationType::StructuredTreeBlocks;
    }
    return NewtonLinearizationType::SparseJacobian;
  }

  Core::LinAlg::SparseMatrix* TreeNewtonLinearSolver::sparse_jacobian_target()
  {
    return coefficient_source_ == TreeNewtonLinearSolverCoefficientSource::SparseJacobian
               ? validation_jacobian_
               : nullptr;
  }

  void TreeNewtonLinearSolver::resolve_structured_coefficient_locations(
      const TreeLinearization& tree_linearization)
  {
    const auto resolve = [&](TreeCoefficientLocation& location) -> double
    {
      FOUR_C_ASSERT_ALWAYS(
          location.local_row >= 0 && location.local_row < tree_linearization.num_rows(),
          "TreeNewtonLinearSolver coefficient row {} is outside [0, {}).", location.local_row,
          tree_linearization.num_rows());
      FOUR_C_ASSERT_ALWAYS(
          location.local_dof >= 0 && location.local_dof < tree_linearization.num_dofs(),
          "TreeNewtonLinearSolver coefficient dof {} is outside [0, {}).", location.local_dof,
          tree_linearization.num_dofs());

      const auto& row = tree_linearization.entries(location.local_row);
      if (location.structured_entry_index >= 0 &&
          location.structured_entry_index < static_cast<int>(row.size()))
      {
        const auto& coefficient = row[static_cast<std::size_t>(location.structured_entry_index)];
        if (coefficient.first == location.local_dof)
        {
          return coefficient.second;
        }
      }

      location.structured_entry_index = -1;
      for (int entry_index = 0; entry_index < static_cast<int>(row.size()); ++entry_index)
      {
        if (row[static_cast<std::size_t>(entry_index)].first == location.local_dof)
        {
          location.structured_entry_index = entry_index;
          return row[static_cast<std::size_t>(entry_index)].second;
        }
      }
      return 0.0;
    };

    root_boundary_coefficient_value_ = resolve(root_boundary_coefficient_);
    for (std::size_t i = 0; i < equation_inlet_pressure_coefficients_.size(); ++i)
    {
      equation_inlet_pressure_coefficient_values_[i] =
          resolve(equation_inlet_pressure_coefficients_[i]);
    }
    for (std::size_t i = 0; i < matrix_coefficients_.size(); ++i)
    {
      matrix_coefficient_values_[i] = resolve(matrix_coefficients_[i]);
    }
    for (std::size_t i = 0; i < child_pressure_parent_coefficients_.size(); ++i)
    {
      child_pressure_parent_coefficient_values_[i] =
          resolve(child_pressure_parent_coefficients_[i]);
    }
    for (std::size_t i = 0; i < child_pressure_child_coefficients_.size(); ++i)
    {
      child_pressure_child_coefficient_values_[i] = resolve(child_pressure_child_coefficients_[i]);
    }
    for (std::size_t i = 0; i < child_flow_coefficients_.size(); ++i)
    {
      child_flow_coefficient_values_[i] = resolve(child_flow_coefficients_[i]);
    }
  }

  void TreeNewtonLinearSolver::set_tree_linearization(const TreeLinearization& tree_linearization)
  {
    tree_linearization_ = &tree_linearization;
    if (coefficient_source_ == TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks)
    {
      resolve_structured_coefficient_locations(tree_linearization);
    }
  }

  TreeCoefficientAssemblyTarget* TreeNewtonLinearSolver::direct_tree_coefficient_target()
  {
    if (coefficient_source_ != TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks)
    {
      return nullptr;
    }
    tree_linearization_ = nullptr;
    return this;
  }

  std::vector<TreeStructuredCoefficientValue>
  TreeNewtonLinearSolver::structured_coefficient_values() const
  {
    std::vector<TreeStructuredCoefficientValue> values;
    values.reserve(1 + equation_inlet_pressure_coefficients_.size() + matrix_coefficients_.size() +
                   child_pressure_parent_coefficients_.size() +
                   child_pressure_child_coefficients_.size() + child_flow_coefficients_.size());
    const auto append_value =
        [&values](const TreeCoefficientLocation& location, double value, const char* context)
    {
      values.push_back(TreeStructuredCoefficientValue{.local_row = location.local_row,
          .local_dof = location.local_dof,
          .value = value,
          .context = context});
    };

    append_value(
        root_boundary_coefficient_, root_boundary_coefficient_value_, "root inlet boundary");
    for (std::size_t i = 0; i < equation_inlet_pressure_coefficients_.size(); ++i)
    {
      append_value(equation_inlet_pressure_coefficients_[i],
          equation_inlet_pressure_coefficient_values_[i], "equation inlet-pressure coefficient");
    }
    for (std::size_t i = 0; i < matrix_coefficients_.size(); ++i)
    {
      append_value(
          matrix_coefficients_[i], matrix_coefficient_values_[i], "element matrix coefficient");
    }
    for (std::size_t i = 0; i < child_pressure_parent_coefficients_.size(); ++i)
    {
      append_value(child_pressure_parent_coefficients_[i],
          child_pressure_parent_coefficient_values_[i], "child pressure parent coefficient");
    }
    for (std::size_t i = 0; i < child_pressure_child_coefficients_.size(); ++i)
    {
      append_value(child_pressure_child_coefficients_[i],
          child_pressure_child_coefficient_values_[i], "child pressure child coefficient");
    }
    for (std::size_t i = 0; i < child_flow_coefficients_.size(); ++i)
    {
      append_value(
          child_flow_coefficients_[i], child_flow_coefficient_values_[i], "child flow coefficient");
    }
    return values;
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
    FOUR_C_ASSERT_ALWAYS(
        local_row_ids.size() == local_dof_ids.size() && local_row_ids.size() == values.size(),
        "TreeNewtonLinearSolver direct coefficient batch replacement size mismatch: rows {}, dofs "
        "{}, values {}.",
        local_row_ids.size(), local_dof_ids.size(), values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
    {
      set_direct_coefficient_value(local_row_ids[i], local_dof_ids[i], values[i], "replace");
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
    FOUR_C_ASSERT_ALWAYS(entry != nullptr,
        "TreeNewtonLinearSolver direct coefficient {} for row {} dof {} does not match the "
        "serial tree symbolic plan.",
        operation, local_row_id, local_dof_id);
    FOUR_C_ASSERT_ALWAYS(entry->value != nullptr,
        "TreeNewtonLinearSolver direct coefficient {} for row {} dof {} has no storage.", operation,
        local_row_id, local_dof_id);
    *entry->value = value;
  }

  void TreeNewtonLinearSolver::solve(const Core::LinAlg::Vector<double>& residual,
      const Core::LinAlg::Vector<double>& x, const NewtonLinearSystemMetadata& metadata,
      Core::LinAlg::Vector<double>& delta)
  {
    TEUCHOS_FUNC_TIME_MONITOR("ReducedLung::NewtonTree:  2)   Solve");
    (void)x;
    (void)metadata;
    const auto solve_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};

    int comm_size = 1;
    MPI_Comm_size(delta.get_comm(), &comm_size);
    FOUR_C_ASSERT_ALWAYS(comm_size == 1,
        "TreeNewtonLinearSolver currently supports only serial reduced-lung solves.");
    if (coefficient_source_ == TreeNewtonLinearSolverCoefficientSource::SparseJacobian)
    {
      FOUR_C_ASSERT_ALWAYS(validation_jacobian_ != nullptr && validation_jacobian_->filled(),
          "TreeNewtonLinearSolver requires a completed sparse Jacobian before solving.");
    }
    else
    {
      if (tree_linearization_ != nullptr)
      {
        FOUR_C_ASSERT_ALWAYS(tree_linearization_->num_rows() == tree_metadata_.num_global_equations,
            "TreeNewtonLinearSolver structured linearization row count does not match metadata.");
        FOUR_C_ASSERT_ALWAYS(
            tree_linearization_->num_dofs() == tree_metadata_.num_locally_relevant_dofs,
            "TreeNewtonLinearSolver structured linearization dof count does not match metadata.");
      }
    }
    FOUR_C_ASSERT_ALWAYS(residual.local_length() == tree_metadata_.num_global_equations,
        "TreeNewtonLinearSolver requires all residual rows to be locally available.");
    FOUR_C_ASSERT_ALWAYS(delta.local_length() == tree_metadata_.num_global_dofs,
        "TreeNewtonLinearSolver requires all correction dofs to be locally available.");

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

    const auto solve_with_coefficients = [&](const auto& coefficients)
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
            -matrix_value(coefficients,
                equation_inlet_pressure_coefficients_[static_cast<std::size_t>(
                    unknown_begin + equation_index)],
                equation_inlet_pressure_coefficient_values_[static_cast<std::size_t>(
                    unknown_begin + equation_index)],
                pivot_tolerance_);

        for (int i = 0; i < block_size; ++i)
        {
          workspace_matrix_[static_cast<std::size_t>(matrix_row_offset + i)] = matrix_value(
              coefficients, matrix_coefficients_[static_cast<std::size_t>(matrix_row_offset + i)],
              matrix_coefficient_values_[static_cast<std::size_t>(matrix_row_offset + i)],
              pivot_tolerance_);
        }
      };

      const auto add_2x2_equation_rows = [&](int element_index)
      {
        const std::size_t element_index_size = static_cast<std::size_t>(element_index);
        const int unknown_begin = unknown_offset_[element_index_size];
        const int equation_begin = equation_offset_[element_index_size];
        const int matrix_begin = matrix_offset_[element_index_size];
        const int row0 = equation_rows_[static_cast<std::size_t>(equation_begin)];
        const int row1 = equation_rows_[static_cast<std::size_t>(equation_begin + 1)];

        workspace_rhs_constant_[static_cast<std::size_t>(unknown_begin)] =
            rhs_value(residual, row0);
        workspace_rhs_inlet_pressure_[static_cast<std::size_t>(unknown_begin)] = -matrix_value(
            coefficients,
            equation_inlet_pressure_coefficients_[static_cast<std::size_t>(unknown_begin)],
            equation_inlet_pressure_coefficient_values_[static_cast<std::size_t>(unknown_begin)],
            pivot_tolerance_);
        workspace_matrix_[static_cast<std::size_t>(matrix_begin)] = matrix_value(coefficients,
            matrix_coefficients_[static_cast<std::size_t>(matrix_begin)],
            matrix_coefficient_values_[static_cast<std::size_t>(matrix_begin)], pivot_tolerance_);
        workspace_matrix_[static_cast<std::size_t>(matrix_begin + 1)] = matrix_value(coefficients,
            matrix_coefficients_[static_cast<std::size_t>(matrix_begin + 1)],
            matrix_coefficient_values_[static_cast<std::size_t>(matrix_begin + 1)],
            pivot_tolerance_);

        workspace_rhs_constant_[static_cast<std::size_t>(unknown_begin + 1)] =
            rhs_value(residual, row1);
        workspace_rhs_inlet_pressure_[static_cast<std::size_t>(unknown_begin + 1)] =
            -matrix_value(coefficients,
                equation_inlet_pressure_coefficients_[static_cast<std::size_t>(unknown_begin + 1)],
                equation_inlet_pressure_coefficient_values_[static_cast<std::size_t>(
                    unknown_begin + 1)],
                pivot_tolerance_);
        workspace_matrix_[static_cast<std::size_t>(matrix_begin + 2)] = matrix_value(coefficients,
            matrix_coefficients_[static_cast<std::size_t>(matrix_begin + 2)],
            matrix_coefficient_values_[static_cast<std::size_t>(matrix_begin + 2)],
            pivot_tolerance_);
        workspace_matrix_[static_cast<std::size_t>(matrix_begin + 3)] = matrix_value(coefficients,
            matrix_coefficients_[static_cast<std::size_t>(matrix_begin + 3)],
            matrix_coefficient_values_[static_cast<std::size_t>(matrix_begin + 3)],
            pivot_tolerance_);
      };

      // A child subtree is represented by delta_q_in = G delta_p_in + h. Pressure continuity maps
      // the parent outlet-pressure unknown to the child's inlet pressure before adding its flow.
      const auto add_2x2_child_contribution =
          [&](int child_interface_index, int matrix_row_offset, double& rhs_shift)
      {
        const std::size_t child_interface_index_size =
            static_cast<std::size_t>(child_interface_index);
        const std::size_t child_element_index =
            static_cast<std::size_t>(child_element_index_[child_interface_index_size]);

        const double pressure_parent_coeff = required_matrix_value(coefficients,
            child_pressure_parent_coefficients_[child_interface_index_size],
            child_pressure_parent_coefficient_values_[child_interface_index_size], pivot_tolerance_,
            "pressure-continuity parent pressure");
        const double pressure_child_coeff = required_matrix_value(coefficients,
            child_pressure_child_coefficients_[child_interface_index_size],
            child_pressure_child_coefficient_values_[child_interface_index_size], pivot_tolerance_,
            "pressure-continuity child pressure");
        const double pressure_rhs = rhs_value(residual, pressure_row_[child_interface_index_size]);

        const double child_pressure_slope = -pressure_parent_coeff / pressure_child_coeff;
        const double child_pressure_intercept = pressure_rhs / pressure_child_coeff;
        child_pressure_slope_[child_interface_index_size] = child_pressure_slope;
        child_pressure_intercept_[child_interface_index_size] = child_pressure_intercept;

        const double child_flow_slope =
            subtree_relation_G_[child_element_index] * child_pressure_slope;
        const double child_flow_intercept =
            subtree_relation_G_[child_element_index] * child_pressure_intercept +
            subtree_relation_h_[child_element_index];

        const double flow_child_coeff = required_matrix_value(coefficients,
            child_flow_coefficients_[child_interface_index_size],
            child_flow_coefficient_values_[child_interface_index_size], pivot_tolerance_,
            "junction flow child-flow coefficient");

        const std::size_t parent_outlet_pressure_index = static_cast<std::size_t>(
            parent_outlet_pressure_unknown_index_[child_interface_index_size]);
        workspace_matrix_[static_cast<std::size_t>(matrix_row_offset) +
                          parent_outlet_pressure_index] += flow_child_coeff * child_flow_slope;
        rhs_shift += flow_child_coeff * child_flow_intercept;
      };

#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
      const auto require_structured_coefficients = [&](const tree_solver_simd::Double& values,
                                                       int chunk_begin, int valid_end,
                                                       const char* context)
      {
        tree_solver_simd::for_each_valid_lane(chunk_begin, valid_end,
            [&](int /*grouped_index*/, int lane)
            {
              FOUR_C_ASSERT_ALWAYS(
                  std::abs(values[static_cast<std::size_t>(lane)]) > pivot_tolerance_,
                  "TreeNewtonLinearSolver missing or near-zero matrix coefficient for {}.",
                  context);
            });
      };

      const auto assemble_2x2_equation_rows_structured_chunk =
          [&](int chunk_begin, int group_begin, int valid_end)
      {
        const auto load_rhs = [&](int grouped_index, int equation_offset)
        {
          const int equation_begin =
              grouped_equation_begin_[static_cast<std::size_t>(grouped_index)];
          return rhs_value(
              residual, equation_rows_[static_cast<std::size_t>(equation_begin + equation_offset)]);
        };
        const auto load_inlet_coefficient = [&](int grouped_index, int unknown_offset)
        {
          const int unknown_begin = grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
          return equation_inlet_pressure_coefficient_values_[static_cast<std::size_t>(
              unknown_begin + unknown_offset)];
        };
        const auto load_matrix_coefficient = [&](int grouped_index, int matrix_offset)
        {
          const int matrix_begin = grouped_matrix_begin_[static_cast<std::size_t>(grouped_index)];
          return matrix_coefficient_values_[static_cast<std::size_t>(matrix_begin + matrix_offset)];
        };
        const auto store_rhs_constant = [&](int grouped_index, int unknown_offset, double value)
        {
          const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group_begin);
          if (unknown_offset == 0)
          {
            batch_2x2_rhs_constant0_[lane_index] = value;
          }
          else
          {
            batch_2x2_rhs_constant1_[lane_index] = value;
          }
        };
        const auto store_rhs_inlet_pressure =
            [&](int grouped_index, int unknown_offset, double value)
        {
          const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group_begin);
          if (unknown_offset == 0)
          {
            batch_2x2_rhs_inlet_pressure0_[lane_index] = value;
          }
          else
          {
            batch_2x2_rhs_inlet_pressure1_[lane_index] = value;
          }
        };
        const auto store_matrix = [&](int grouped_index, int matrix_offset, double value)
        {
          const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group_begin);
          switch (matrix_offset)
          {
            case 0:
              batch_2x2_a00_[lane_index] = value;
              break;
            case 1:
              batch_2x2_a01_[lane_index] = value;
              break;
            case 2:
              batch_2x2_a10_[lane_index] = value;
              break;
            case 3:
              batch_2x2_a11_[lane_index] = value;
              break;
            default:
              FOUR_C_THROW(
                  "TreeNewtonLinearSolver 2x2 batch has invalid matrix offset {}.", matrix_offset);
          }
        };

        const tree_solver_simd::Double rhs0 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            0.0, [&](int grouped_index) { return load_rhs(grouped_index, 0); });
        const tree_solver_simd::Double rhs1 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            0.0, [&](int grouped_index) { return load_rhs(grouped_index, 1); });
        const tree_solver_simd::Double rhs_inlet0 =
            -tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                [&](int grouped_index) { return load_inlet_coefficient(grouped_index, 0); });
        const tree_solver_simd::Double rhs_inlet1 =
            -tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                [&](int grouped_index) { return load_inlet_coefficient(grouped_index, 1); });
        const tree_solver_simd::Double a00 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            1.0, [&](int grouped_index) { return load_matrix_coefficient(grouped_index, 0); });
        const tree_solver_simd::Double a01 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            0.0, [&](int grouped_index) { return load_matrix_coefficient(grouped_index, 1); });
        const tree_solver_simd::Double a10 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            0.0, [&](int grouped_index) { return load_matrix_coefficient(grouped_index, 2); });
        const tree_solver_simd::Double a11 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            1.0, [&](int grouped_index) { return load_matrix_coefficient(grouped_index, 3); });

        tree_solver_simd::scatter_valid(rhs0, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_rhs_constant(grouped_index, 0, value); });
        tree_solver_simd::scatter_valid(rhs1, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_rhs_constant(grouped_index, 1, value); });
        tree_solver_simd::scatter_valid(rhs_inlet0, chunk_begin, valid_end,
            [&](int grouped_index, double value)
            { store_rhs_inlet_pressure(grouped_index, 0, value); });
        tree_solver_simd::scatter_valid(rhs_inlet1, chunk_begin, valid_end,
            [&](int grouped_index, double value)
            { store_rhs_inlet_pressure(grouped_index, 1, value); });
        tree_solver_simd::scatter_valid(a00, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_matrix(grouped_index, 0, value); });
        tree_solver_simd::scatter_valid(a01, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_matrix(grouped_index, 1, value); });
        tree_solver_simd::scatter_valid(a10, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_matrix(grouped_index, 2, value); });
        tree_solver_simd::scatter_valid(a11, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_matrix(grouped_index, 3, value); });
      };

      const auto add_2x2_child_contribution_structured_chunk =
          [&](int chunk_begin, int group_begin, int valid_end, int child_slot,
              tree_solver_simd::Double& rhs_shift)
      {
        const auto child_interface_index = [&](int grouped_index)
        { return grouped_child_begin_[static_cast<std::size_t>(grouped_index)] + child_slot; };
        const auto load_child_value = [&](int grouped_index,
                                          const std::vector<double>& values) -> double
        { return values[static_cast<std::size_t>(child_interface_index(grouped_index))]; };
        const auto load_pressure_rhs = [&](int grouped_index)
        {
          const int child_index = child_interface_index(grouped_index);
          return rhs_value(residual, pressure_row_[static_cast<std::size_t>(child_index)]);
        };
        const auto load_subtree_G = [&](int grouped_index)
        {
          const int child_index = child_interface_index(grouped_index);
          const int element_index = child_element_index_[static_cast<std::size_t>(child_index)];
          return subtree_relation_G_[static_cast<std::size_t>(element_index)];
        };
        const auto load_subtree_h = [&](int grouped_index)
        {
          const int child_index = child_interface_index(grouped_index);
          const int element_index = child_element_index_[static_cast<std::size_t>(child_index)];
          return subtree_relation_h_[static_cast<std::size_t>(element_index)];
        };

        const tree_solver_simd::Double pressure_parent_coeff = tree_solver_simd::gather_or(
            chunk_begin, valid_end, 1.0, [&](int grouped_index)
            { return load_child_value(grouped_index, child_pressure_parent_coefficient_values_); });
        const tree_solver_simd::Double pressure_child_coeff = tree_solver_simd::gather_or(
            chunk_begin, valid_end, 1.0, [&](int grouped_index)
            { return load_child_value(grouped_index, child_pressure_child_coefficient_values_); });
        const tree_solver_simd::Double pressure_rhs = tree_solver_simd::gather_or(chunk_begin,
            valid_end, 0.0, [&](int grouped_index) { return load_pressure_rhs(grouped_index); });

        require_structured_coefficients(
            pressure_parent_coeff, chunk_begin, valid_end, "pressure-continuity parent pressure");
        require_structured_coefficients(
            pressure_child_coeff, chunk_begin, valid_end, "pressure-continuity child pressure");

        const tree_solver_simd::Double child_pressure_slope =
            -pressure_parent_coeff / pressure_child_coeff;
        const tree_solver_simd::Double child_pressure_intercept =
            pressure_rhs / pressure_child_coeff;

        tree_solver_simd::scatter_valid(child_pressure_slope, chunk_begin, valid_end,
            [&](int grouped_index, double value)
            {
              const std::size_t child_index =
                  static_cast<std::size_t>(child_interface_index(grouped_index));
              child_pressure_slope_[child_index] = value;
            });
        tree_solver_simd::scatter_valid(child_pressure_intercept, chunk_begin, valid_end,
            [&](int grouped_index, double value)
            {
              const std::size_t child_index =
                  static_cast<std::size_t>(child_interface_index(grouped_index));
              child_pressure_intercept_[child_index] = value;
            });

        const tree_solver_simd::Double subtree_G = tree_solver_simd::gather_or(chunk_begin,
            valid_end, 0.0, [&](int grouped_index) { return load_subtree_G(grouped_index); });
        const tree_solver_simd::Double subtree_h = tree_solver_simd::gather_or(chunk_begin,
            valid_end, 0.0, [&](int grouped_index) { return load_subtree_h(grouped_index); });
        const tree_solver_simd::Double child_flow_slope = subtree_G * child_pressure_slope;
        const tree_solver_simd::Double child_flow_intercept =
            subtree_G * child_pressure_intercept + subtree_h;

        const tree_solver_simd::Double flow_child_coeff =
            tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0, [&](int grouped_index)
                { return load_child_value(grouped_index, child_flow_coefficient_values_); });
        require_structured_coefficients(
            flow_child_coeff, chunk_begin, valid_end, "junction flow child-flow coefficient");

        const tree_solver_simd::Double matrix_update = flow_child_coeff * child_flow_slope;
        const tree_solver_simd::Double rhs_update = flow_child_coeff * child_flow_intercept;
        tree_solver_simd::scatter_valid(matrix_update, chunk_begin, valid_end,
            [&](int grouped_index, double value)
            {
              const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group_begin);
              const int child_index = child_interface_index(grouped_index);
              const int parent_outlet_pressure_index =
                  parent_outlet_pressure_unknown_index_[static_cast<std::size_t>(child_index)];
              if (parent_outlet_pressure_index == 0)
              {
                batch_2x2_a10_[lane_index] += value;
              }
              else
              {
                batch_2x2_a11_[lane_index] += value;
              }
            });
        rhs_shift += rhs_update;
      };

      const auto subtract_2x2_rhs_shift_structured_chunk =
          [&](int chunk_begin, int group_begin, int valid_end,
              const tree_solver_simd::Double& rhs_shift)
      {
        tree_solver_simd::scatter_valid(rhs_shift, chunk_begin, valid_end,
            [&](int grouped_index, double value)
            {
              const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group_begin);
              batch_2x2_rhs_constant1_[lane_index] -= value;
            });
      };

      const auto assemble_3x3_equation_rows_structured_chunk =
          [&](int chunk_begin, int group_begin, int valid_end)
      {
        const std::array<std::vector<double>*, 3> rhs_constant_outputs = {
            &batch_3x3_rhs_constant0_, &batch_3x3_rhs_constant1_, &batch_3x3_rhs_constant2_};
        const std::array<std::vector<double>*, 3> rhs_inlet_pressure_outputs = {
            &batch_3x3_rhs_inlet_pressure0_, &batch_3x3_rhs_inlet_pressure1_,
            &batch_3x3_rhs_inlet_pressure2_};
        const std::array<std::vector<double>*, 9> matrix_outputs = {&batch_3x3_a00_,
            &batch_3x3_a01_, &batch_3x3_a02_, &batch_3x3_a10_, &batch_3x3_a11_, &batch_3x3_a12_,
            &batch_3x3_a20_, &batch_3x3_a21_, &batch_3x3_a22_};
        const auto load_rhs = [&](int grouped_index, int equation_offset)
        {
          const int equation_begin =
              grouped_equation_begin_[static_cast<std::size_t>(grouped_index)];
          return rhs_value(
              residual, equation_rows_[static_cast<std::size_t>(equation_begin + equation_offset)]);
        };
        const auto load_inlet_coefficient = [&](int grouped_index, int unknown_offset)
        {
          const int unknown_begin = grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
          return equation_inlet_pressure_coefficient_values_[static_cast<std::size_t>(
              unknown_begin + unknown_offset)];
        };
        const auto load_matrix_coefficient = [&](int grouped_index, int matrix_offset)
        {
          const int matrix_begin = grouped_matrix_begin_[static_cast<std::size_t>(grouped_index)];
          return matrix_coefficient_values_[static_cast<std::size_t>(matrix_begin + matrix_offset)];
        };
        const auto store_rhs_constant = [&](int grouped_index, int unknown_offset, double value)
        {
          const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group_begin);
          (*rhs_constant_outputs[static_cast<std::size_t>(unknown_offset)])[lane_index] = value;
        };
        const auto store_rhs_inlet_pressure =
            [&](int grouped_index, int unknown_offset, double value)
        {
          const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group_begin);
          (*rhs_inlet_pressure_outputs[static_cast<std::size_t>(unknown_offset)])[lane_index] =
              value;
        };
        const auto store_matrix = [&](int grouped_index, int matrix_offset, double value)
        {
          const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group_begin);
          (*matrix_outputs[static_cast<std::size_t>(matrix_offset)])[lane_index] = value;
        };

        const tree_solver_simd::Double rhs0 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            0.0, [&](int grouped_index) { return load_rhs(grouped_index, 0); });
        const tree_solver_simd::Double rhs1 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            0.0, [&](int grouped_index) { return load_rhs(grouped_index, 1); });
        const tree_solver_simd::Double rhs2 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            0.0, [&](int grouped_index) { return load_rhs(grouped_index, 2); });
        const tree_solver_simd::Double rhs_inlet0 =
            -tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                [&](int grouped_index) { return load_inlet_coefficient(grouped_index, 0); });
        const tree_solver_simd::Double rhs_inlet1 =
            -tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                [&](int grouped_index) { return load_inlet_coefficient(grouped_index, 1); });
        const tree_solver_simd::Double rhs_inlet2 =
            -tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                [&](int grouped_index) { return load_inlet_coefficient(grouped_index, 2); });
        const tree_solver_simd::Double a00 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            1.0, [&](int grouped_index) { return load_matrix_coefficient(grouped_index, 0); });
        const tree_solver_simd::Double a01 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            0.0, [&](int grouped_index) { return load_matrix_coefficient(grouped_index, 1); });
        const tree_solver_simd::Double a02 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            0.0, [&](int grouped_index) { return load_matrix_coefficient(grouped_index, 2); });
        const tree_solver_simd::Double a10 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            0.0, [&](int grouped_index) { return load_matrix_coefficient(grouped_index, 3); });
        const tree_solver_simd::Double a11 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            1.0, [&](int grouped_index) { return load_matrix_coefficient(grouped_index, 4); });
        const tree_solver_simd::Double a12 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            0.0, [&](int grouped_index) { return load_matrix_coefficient(grouped_index, 5); });
        const tree_solver_simd::Double a20 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            0.0, [&](int grouped_index) { return load_matrix_coefficient(grouped_index, 6); });
        const tree_solver_simd::Double a21 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            0.0, [&](int grouped_index) { return load_matrix_coefficient(grouped_index, 7); });
        const tree_solver_simd::Double a22 = tree_solver_simd::gather_or(chunk_begin, valid_end,
            1.0, [&](int grouped_index) { return load_matrix_coefficient(grouped_index, 8); });

        tree_solver_simd::scatter_valid(rhs0, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_rhs_constant(grouped_index, 0, value); });
        tree_solver_simd::scatter_valid(rhs1, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_rhs_constant(grouped_index, 1, value); });
        tree_solver_simd::scatter_valid(rhs2, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_rhs_constant(grouped_index, 2, value); });
        tree_solver_simd::scatter_valid(rhs_inlet0, chunk_begin, valid_end,
            [&](int grouped_index, double value)
            { store_rhs_inlet_pressure(grouped_index, 0, value); });
        tree_solver_simd::scatter_valid(rhs_inlet1, chunk_begin, valid_end,
            [&](int grouped_index, double value)
            { store_rhs_inlet_pressure(grouped_index, 1, value); });
        tree_solver_simd::scatter_valid(rhs_inlet2, chunk_begin, valid_end,
            [&](int grouped_index, double value)
            { store_rhs_inlet_pressure(grouped_index, 2, value); });
        tree_solver_simd::scatter_valid(a00, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_matrix(grouped_index, 0, value); });
        tree_solver_simd::scatter_valid(a01, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_matrix(grouped_index, 1, value); });
        tree_solver_simd::scatter_valid(a02, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_matrix(grouped_index, 2, value); });
        tree_solver_simd::scatter_valid(a10, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_matrix(grouped_index, 3, value); });
        tree_solver_simd::scatter_valid(a11, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_matrix(grouped_index, 4, value); });
        tree_solver_simd::scatter_valid(a12, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_matrix(grouped_index, 5, value); });
        tree_solver_simd::scatter_valid(a20, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_matrix(grouped_index, 6, value); });
        tree_solver_simd::scatter_valid(a21, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_matrix(grouped_index, 7, value); });
        tree_solver_simd::scatter_valid(a22, chunk_begin, valid_end,
            [&](int grouped_index, double value) { store_matrix(grouped_index, 8, value); });
      };

      const auto add_3x3_child_contribution_structured_chunk =
          [&](int chunk_begin, int group_begin, int valid_end, int child_slot,
              tree_solver_simd::Double& rhs_shift)
      {
        const auto child_interface_index = [&](int grouped_index)
        { return grouped_child_begin_[static_cast<std::size_t>(grouped_index)] + child_slot; };
        const auto load_child_value = [&](int grouped_index,
                                          const std::vector<double>& values) -> double
        { return values[static_cast<std::size_t>(child_interface_index(grouped_index))]; };
        const auto load_pressure_rhs = [&](int grouped_index)
        {
          const int child_index = child_interface_index(grouped_index);
          return rhs_value(residual, pressure_row_[static_cast<std::size_t>(child_index)]);
        };
        const auto load_subtree_G = [&](int grouped_index)
        {
          const int child_index = child_interface_index(grouped_index);
          const int element_index = child_element_index_[static_cast<std::size_t>(child_index)];
          return subtree_relation_G_[static_cast<std::size_t>(element_index)];
        };
        const auto load_subtree_h = [&](int grouped_index)
        {
          const int child_index = child_interface_index(grouped_index);
          const int element_index = child_element_index_[static_cast<std::size_t>(child_index)];
          return subtree_relation_h_[static_cast<std::size_t>(element_index)];
        };

        const tree_solver_simd::Double pressure_parent_coeff = tree_solver_simd::gather_or(
            chunk_begin, valid_end, 1.0, [&](int grouped_index)
            { return load_child_value(grouped_index, child_pressure_parent_coefficient_values_); });
        const tree_solver_simd::Double pressure_child_coeff = tree_solver_simd::gather_or(
            chunk_begin, valid_end, 1.0, [&](int grouped_index)
            { return load_child_value(grouped_index, child_pressure_child_coefficient_values_); });
        const tree_solver_simd::Double pressure_rhs = tree_solver_simd::gather_or(chunk_begin,
            valid_end, 0.0, [&](int grouped_index) { return load_pressure_rhs(grouped_index); });

        require_structured_coefficients(
            pressure_parent_coeff, chunk_begin, valid_end, "pressure-continuity parent pressure");
        require_structured_coefficients(
            pressure_child_coeff, chunk_begin, valid_end, "pressure-continuity child pressure");

        const tree_solver_simd::Double child_pressure_slope =
            -pressure_parent_coeff / pressure_child_coeff;
        const tree_solver_simd::Double child_pressure_intercept =
            pressure_rhs / pressure_child_coeff;

        tree_solver_simd::scatter_valid(child_pressure_slope, chunk_begin, valid_end,
            [&](int grouped_index, double value)
            {
              const std::size_t child_index =
                  static_cast<std::size_t>(child_interface_index(grouped_index));
              child_pressure_slope_[child_index] = value;
            });
        tree_solver_simd::scatter_valid(child_pressure_intercept, chunk_begin, valid_end,
            [&](int grouped_index, double value)
            {
              const std::size_t child_index =
                  static_cast<std::size_t>(child_interface_index(grouped_index));
              child_pressure_intercept_[child_index] = value;
            });

        const tree_solver_simd::Double subtree_G = tree_solver_simd::gather_or(chunk_begin,
            valid_end, 0.0, [&](int grouped_index) { return load_subtree_G(grouped_index); });
        const tree_solver_simd::Double subtree_h = tree_solver_simd::gather_or(chunk_begin,
            valid_end, 0.0, [&](int grouped_index) { return load_subtree_h(grouped_index); });
        const tree_solver_simd::Double child_flow_slope = subtree_G * child_pressure_slope;
        const tree_solver_simd::Double child_flow_intercept =
            subtree_G * child_pressure_intercept + subtree_h;

        const tree_solver_simd::Double flow_child_coeff =
            tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0, [&](int grouped_index)
                { return load_child_value(grouped_index, child_flow_coefficient_values_); });
        require_structured_coefficients(
            flow_child_coeff, chunk_begin, valid_end, "junction flow child-flow coefficient");

        const tree_solver_simd::Double matrix_update = flow_child_coeff * child_flow_slope;
        const tree_solver_simd::Double rhs_update = flow_child_coeff * child_flow_intercept;
        tree_solver_simd::scatter_valid(matrix_update, chunk_begin, valid_end,
            [&](int grouped_index, double value)
            {
              const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group_begin);
              const int child_index = child_interface_index(grouped_index);
              const int parent_outlet_pressure_index =
                  parent_outlet_pressure_unknown_index_[static_cast<std::size_t>(child_index)];
              switch (parent_outlet_pressure_index)
              {
                case 0:
                  batch_3x3_a20_[lane_index] += value;
                  break;
                case 1:
                  batch_3x3_a21_[lane_index] += value;
                  break;
                case 2:
                  batch_3x3_a22_[lane_index] += value;
                  break;
                default:
                  FOUR_C_THROW(
                      "TreeNewtonLinearSolver 3x3 batch has invalid outlet-pressure index {}.",
                      parent_outlet_pressure_index);
              }
            });
        rhs_shift += rhs_update;
      };

      const auto subtract_3x3_rhs_shift_structured_chunk =
          [&](int chunk_begin, int group_begin, int valid_end,
              const tree_solver_simd::Double& rhs_shift)
      {
        tree_solver_simd::scatter_valid(rhs_shift, chunk_begin, valid_end,
            [&](int grouped_index, double value)
            {
              const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group_begin);
              batch_3x3_rhs_constant2_[lane_index] -= value;
            });
      };
#endif

      const auto assemble_2x2_leaf_group = [&](const ElementGroup& group)
      {
#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
        if (coefficient_source_ == TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks)
        {
          int grouped_index = group.begin;
          const int padded_end = tree_solver_simd::padded_chunk_end(group.begin, group.end);
          for (; grouped_index < padded_end; grouped_index += tree_solver_simd::width())
          {
            assemble_2x2_equation_rows_structured_chunk(grouped_index, group.begin, group.end);
          }
          return;
        }
#endif
        for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
        {
          add_2x2_equation_rows(grouped_element_indices_[static_cast<std::size_t>(grouped_index)]);
        }
      };

      const auto assemble_2x2_one_child_group = [&](const ElementGroup& group)
      {
#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
        if (coefficient_source_ == TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks)
        {
          int grouped_index = group.begin;
          const int padded_end = tree_solver_simd::padded_chunk_end(group.begin, group.end);
          for (; grouped_index < padded_end; grouped_index += tree_solver_simd::width())
          {
            assemble_2x2_equation_rows_structured_chunk(grouped_index, group.begin, group.end);
            tree_solver_simd::Double rhs_shift(0.0);
            add_2x2_child_contribution_structured_chunk(
                grouped_index, group.begin, group.end, 0, rhs_shift);
            subtract_2x2_rhs_shift_structured_chunk(
                grouped_index, group.begin, group.end, rhs_shift);
          }
          return;
        }
#endif
        for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
        {
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const std::size_t element_index_size = static_cast<std::size_t>(element_index);
          add_2x2_equation_rows(element_index);

          double rhs_shift = 0.0;
          add_2x2_child_contribution(child_interface_offset_[element_index_size],
              matrix_offset_[element_index_size] + 2, rhs_shift);
          workspace_rhs_constant_[static_cast<std::size_t>(
              unknown_offset_[element_index_size] + 1)] -= rhs_shift;
        }
      };

      const auto assemble_2x2_two_child_group = [&](const ElementGroup& group)
      {
#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
        if (coefficient_source_ == TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks)
        {
          int grouped_index = group.begin;
          const int padded_end = tree_solver_simd::padded_chunk_end(group.begin, group.end);
          for (; grouped_index < padded_end; grouped_index += tree_solver_simd::width())
          {
            assemble_2x2_equation_rows_structured_chunk(grouped_index, group.begin, group.end);
            tree_solver_simd::Double rhs_shift(0.0);
            add_2x2_child_contribution_structured_chunk(
                grouped_index, group.begin, group.end, 0, rhs_shift);
            add_2x2_child_contribution_structured_chunk(
                grouped_index, group.begin, group.end, 1, rhs_shift);
            subtract_2x2_rhs_shift_structured_chunk(
                grouped_index, group.begin, group.end, rhs_shift);
          }
          return;
        }
#endif
        for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
        {
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const std::size_t element_index_size = static_cast<std::size_t>(element_index);
          add_2x2_equation_rows(element_index);

          const int child_begin = child_interface_offset_[element_index_size];
          double rhs_shift = 0.0;
          add_2x2_child_contribution(
              child_begin, matrix_offset_[element_index_size] + 2, rhs_shift);
          add_2x2_child_contribution(
              child_begin + 1, matrix_offset_[element_index_size] + 2, rhs_shift);
          workspace_rhs_constant_[static_cast<std::size_t>(
              unknown_offset_[element_index_size] + 1)] -= rhs_shift;
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

              const double pressure_parent_coeff = required_matrix_value(coefficients,
                  child_pressure_parent_coefficients_[child_interface_index_size],
                  child_pressure_parent_coefficient_values_[child_interface_index_size],
                  pivot_tolerance_, "pressure-continuity parent pressure");
              const double pressure_child_coeff = required_matrix_value(coefficients,
                  child_pressure_child_coefficients_[child_interface_index_size],
                  child_pressure_child_coefficient_values_[child_interface_index_size],
                  pivot_tolerance_, "pressure-continuity child pressure");
              const double pressure_rhs =
                  rhs_value(residual, pressure_row_[child_interface_index_size]);

              const double child_pressure_slope = -pressure_parent_coeff / pressure_child_coeff;
              const double child_pressure_intercept = pressure_rhs / pressure_child_coeff;
              child_pressure_slope_[child_interface_index_size] = child_pressure_slope;
              child_pressure_intercept_[child_interface_index_size] = child_pressure_intercept;

              const double child_flow_slope =
                  subtree_relation_G_[child_element_index] * child_pressure_slope;
              const double child_flow_intercept =
                  subtree_relation_G_[child_element_index] * child_pressure_intercept +
                  subtree_relation_h_[child_element_index];

              const double flow_child_coeff = required_matrix_value(coefficients,
                  child_flow_coefficients_[child_interface_index_size],
                  child_flow_coefficient_values_[child_interface_index_size], pivot_tolerance_,
                  "junction flow child-flow coefficient");

              const std::size_t parent_outlet_pressure_index = static_cast<std::size_t>(
                  parent_outlet_pressure_unknown_index_[child_interface_index_size]);
              workspace_matrix_[static_cast<std::size_t>(matrix_row_offset) +
                                parent_outlet_pressure_index] +=
                  flow_child_coeff * child_flow_slope;
              rhs_shift += flow_child_coeff * child_flow_intercept;
            }

            workspace_rhs_constant_[static_cast<std::size_t>(unknown_begin + equation_index)] -=
                rhs_shift;
          }
        }
      };

      const auto assemble_3x3_leaf_group = [&](const ElementGroup& group)
      {
#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
        if (coefficient_source_ == TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks)
        {
          int grouped_index = group.begin;
          const int padded_end = tree_solver_simd::padded_chunk_end(group.begin, group.end);
          for (; grouped_index < padded_end; grouped_index += tree_solver_simd::width())
          {
            assemble_3x3_equation_rows_structured_chunk(grouped_index, group.begin, group.end);
          }
          return;
        }
#endif
        if (profile_ != nullptr)
        {
          ++profile_->scalar_group_count;
        }
        assemble_group(group);
      };

      const auto assemble_3x3_one_child_group = [&](const ElementGroup& group)
      {
#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
        if (coefficient_source_ == TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks)
        {
          int grouped_index = group.begin;
          const int padded_end = tree_solver_simd::padded_chunk_end(group.begin, group.end);
          for (; grouped_index < padded_end; grouped_index += tree_solver_simd::width())
          {
            assemble_3x3_equation_rows_structured_chunk(grouped_index, group.begin, group.end);
            tree_solver_simd::Double rhs_shift(0.0);
            add_3x3_child_contribution_structured_chunk(
                grouped_index, group.begin, group.end, 0, rhs_shift);
            subtract_3x3_rhs_shift_structured_chunk(
                grouped_index, group.begin, group.end, rhs_shift);
          }
          return;
        }
#endif
        if (profile_ != nullptr)
        {
          ++profile_->scalar_group_count;
        }
        assemble_group(group);
      };

      const auto assemble_3x3_two_child_group = [&](const ElementGroup& group)
      {
#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
        if (coefficient_source_ == TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks)
        {
          int grouped_index = group.begin;
          const int padded_end = tree_solver_simd::padded_chunk_end(group.begin, group.end);
          for (; grouped_index < padded_end; grouped_index += tree_solver_simd::width())
          {
            assemble_3x3_equation_rows_structured_chunk(grouped_index, group.begin, group.end);
            tree_solver_simd::Double rhs_shift(0.0);
            add_3x3_child_contribution_structured_chunk(
                grouped_index, group.begin, group.end, 0, rhs_shift);
            add_3x3_child_contribution_structured_chunk(
                grouped_index, group.begin, group.end, 1, rhs_shift);
            subtract_3x3_rhs_shift_structured_chunk(
                grouped_index, group.begin, group.end, rhs_shift);
          }
          return;
        }
#endif
        if (profile_ != nullptr)
        {
          ++profile_->scalar_group_count;
        }
        assemble_group(group);
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

          const double pressure_parent_coeff = required_matrix_value(coefficients,
              child_pressure_parent_coefficients_[child_interface_index_size],
              child_pressure_parent_coefficient_values_[child_interface_index_size],
              pivot_tolerance_, "pressure-continuity parent pressure");
          const double pressure_child_coeff = required_matrix_value(coefficients,
              child_pressure_child_coefficients_[child_interface_index_size],
              child_pressure_child_coefficient_values_[child_interface_index_size],
              pivot_tolerance_, "pressure-continuity child pressure");
          const double pressure_rhs =
              rhs_value(residual, pressure_row_[child_interface_index_size]);

          const double child_pressure_slope = -pressure_parent_coeff / pressure_child_coeff;
          const double child_pressure_intercept = pressure_rhs / pressure_child_coeff;
          child_pressure_slope_[child_interface_index_size] = child_pressure_slope;
          child_pressure_intercept_[child_interface_index_size] = child_pressure_intercept;

          const double child_flow_slope =
              subtree_relation_G_[child_element_index] * child_pressure_slope;
          const double child_flow_intercept =
              subtree_relation_G_[child_element_index] * child_pressure_intercept +
              subtree_relation_h_[child_element_index];

          const double flow_child_coeff = required_matrix_value(coefficients,
              child_flow_coefficients_[child_interface_index_size],
              child_flow_coefficient_values_[child_interface_index_size], pivot_tolerance_,
              "junction flow child-flow coefficient");

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
        const auto dense_solve_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
        solve_dense_system(workspace_matrix_.data() + matrix_begin,
            workspace_rhs_constant_.data() + unknown_begin,
            workspace_rhs_inlet_pressure_.data() + unknown_begin,
            workspace_intercept_.data() + unknown_begin, workspace_slope_.data() + unknown_begin,
            block_size, pivot_tolerance_, element_context_[element_index_size]);
        if (profile_ != nullptr)
        {
          profile_->dense_solve_time += elapsed_seconds(dense_solve_start);
          ++profile_->dense_solve_count;
        }
      };

      const auto write_subtree_relation = [&](int element_index)
      {
        const std::size_t element_index_size = static_cast<std::size_t>(element_index);
        const int unknown_begin = unknown_offset_[element_index_size];
        subtree_relation_G_[element_index_size] = workspace_slope_[static_cast<std::size_t>(
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

      const auto write_2x2_subtree_relation_group = [&](const ElementGroup& group)
      {
        FOUR_C_ASSERT_ALWAYS(group.block_size == 2,
            "TreeNewtonLinearSolver 2x2 subtree relation group has block size {}.",
            group.block_size);
        [[maybe_unused]] const auto load_relation_value = [&](int grouped_index, bool load_slope)
        {
          const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group.begin);
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const int flow_index = inlet_flow_unknown_index_[static_cast<std::size_t>(element_index)];
          switch (flow_index)
          {
            case 0:
              return load_slope ? batch_2x2_slope0_[lane_index] : batch_2x2_intercept0_[lane_index];
            case 1:
              return load_slope ? batch_2x2_slope1_[lane_index] : batch_2x2_intercept1_[lane_index];
            default:
              FOUR_C_THROW("TreeNewtonLinearSolver 2x2 subtree relation has invalid flow index {}.",
                  flow_index);
              return 0.0;
          }
        };
        [[maybe_unused]] const auto write_relation_lane = [&](int grouped_index)
        {
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const std::size_t element_index_size = static_cast<std::size_t>(element_index);
          subtree_relation_G_[element_index_size] = load_relation_value(grouped_index, true);
          subtree_relation_h_[element_index_size] = load_relation_value(grouped_index, false);
        };

#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
        const auto write_relation_chunk = [&](int chunk_begin, int valid_end)
        {
          const tree_solver_simd::Double slope = tree_solver_simd::gather_or(chunk_begin, valid_end,
              0.0, [&](int grouped_index) { return load_relation_value(grouped_index, true); });
          const tree_solver_simd::Double intercept =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index) { return load_relation_value(grouped_index, false); });
          tree_solver_simd::scatter_valid(slope, chunk_begin, valid_end,
              [&](int grouped_index, double value)
              {
                const int element_index =
                    grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
                subtree_relation_G_[static_cast<std::size_t>(element_index)] = value;
              });
          tree_solver_simd::scatter_valid(intercept, chunk_begin, valid_end,
              [&](int grouped_index, double value)
              {
                const int element_index =
                    grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
                subtree_relation_h_[static_cast<std::size_t>(element_index)] = value;
              });
        };

        int grouped_index = group.begin;
        const int padded_end = tree_solver_simd::padded_chunk_end(group.begin, group.end);
        for (; grouped_index < padded_end; grouped_index += tree_solver_simd::width())
        {
          write_relation_chunk(grouped_index, group.end);
        }
#else
        for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
        {
          write_relation_lane(grouped_index);
        }
#endif
      };

      const auto write_3x3_subtree_relation_group = [&](const ElementGroup& group)
      {
        FOUR_C_ASSERT_ALWAYS(group.block_size == 3,
            "TreeNewtonLinearSolver 3x3 subtree relation group has block size {}.",
            group.block_size);
        [[maybe_unused]] const auto load_relation_value = [&](int grouped_index, bool load_slope)
        {
          const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group.begin);
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const int flow_index = inlet_flow_unknown_index_[static_cast<std::size_t>(element_index)];
          switch (flow_index)
          {
            case 0:
              return load_slope ? batch_3x3_slope0_[lane_index] : batch_3x3_intercept0_[lane_index];
            case 1:
              return load_slope ? batch_3x3_slope1_[lane_index] : batch_3x3_intercept1_[lane_index];
            case 2:
              return load_slope ? batch_3x3_slope2_[lane_index] : batch_3x3_intercept2_[lane_index];
            default:
              FOUR_C_THROW("TreeNewtonLinearSolver 3x3 subtree relation has invalid flow index {}.",
                  flow_index);
              return 0.0;
          }
        };
        [[maybe_unused]] const auto write_relation_lane = [&](int grouped_index)
        {
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const std::size_t element_index_size = static_cast<std::size_t>(element_index);
          subtree_relation_G_[element_index_size] = load_relation_value(grouped_index, true);
          subtree_relation_h_[element_index_size] = load_relation_value(grouped_index, false);
        };

#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
        const auto write_relation_chunk = [&](int chunk_begin, int valid_end)
        {
          const tree_solver_simd::Double slope = tree_solver_simd::gather_or(chunk_begin, valid_end,
              0.0, [&](int grouped_index) { return load_relation_value(grouped_index, true); });
          const tree_solver_simd::Double intercept =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index) { return load_relation_value(grouped_index, false); });
          tree_solver_simd::scatter_valid(slope, chunk_begin, valid_end,
              [&](int grouped_index, double value)
              {
                const int element_index =
                    grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
                subtree_relation_G_[static_cast<std::size_t>(element_index)] = value;
              });
          tree_solver_simd::scatter_valid(intercept, chunk_begin, valid_end,
              [&](int grouped_index, double value)
              {
                const int element_index =
                    grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
                subtree_relation_h_[static_cast<std::size_t>(element_index)] = value;
              });
        };

        int grouped_index = group.begin;
        const int padded_end = tree_solver_simd::padded_chunk_end(group.begin, group.end);
        for (; grouped_index < padded_end; grouped_index += tree_solver_simd::width())
        {
          write_relation_chunk(grouped_index, group.end);
        }
#else
        for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
        {
          write_relation_lane(grouped_index);
        }
#endif
      };

      const auto pack_2x2_group_from_workspace = [&](const ElementGroup& group)
      {
        FOUR_C_ASSERT_ALWAYS(group.block_size == 2,
            "TreeNewtonLinearSolver can pack only 2x2 groups, got block size {}.",
            group.block_size);
        for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
        {
          const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group.begin);
          const int unknown_begin = grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
          const int matrix_begin = grouped_matrix_begin_[static_cast<std::size_t>(grouped_index)];
          batch_2x2_a00_[lane_index] = workspace_matrix_[static_cast<std::size_t>(matrix_begin)];
          batch_2x2_a01_[lane_index] =
              workspace_matrix_[static_cast<std::size_t>(matrix_begin + 1)];
          batch_2x2_a10_[lane_index] =
              workspace_matrix_[static_cast<std::size_t>(matrix_begin + 2)];
          batch_2x2_a11_[lane_index] =
              workspace_matrix_[static_cast<std::size_t>(matrix_begin + 3)];
          batch_2x2_rhs_constant0_[lane_index] =
              workspace_rhs_constant_[static_cast<std::size_t>(unknown_begin)];
          batch_2x2_rhs_constant1_[lane_index] =
              workspace_rhs_constant_[static_cast<std::size_t>(unknown_begin + 1)];
          batch_2x2_rhs_inlet_pressure0_[lane_index] =
              workspace_rhs_inlet_pressure_[static_cast<std::size_t>(unknown_begin)];
          batch_2x2_rhs_inlet_pressure1_[lane_index] =
              workspace_rhs_inlet_pressure_[static_cast<std::size_t>(unknown_begin + 1)];
        }
      };

      const auto pack_3x3_group_from_workspace = [&](const ElementGroup& group)
      {
        FOUR_C_ASSERT_ALWAYS(group.block_size == 3,
            "TreeNewtonLinearSolver can pack only 3x3 groups, got block size {}.",
            group.block_size);
        for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
        {
          const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group.begin);
          const int unknown_begin = grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
          const int matrix_begin = grouped_matrix_begin_[static_cast<std::size_t>(grouped_index)];
          batch_3x3_a00_[lane_index] = workspace_matrix_[static_cast<std::size_t>(matrix_begin)];
          batch_3x3_a01_[lane_index] =
              workspace_matrix_[static_cast<std::size_t>(matrix_begin + 1)];
          batch_3x3_a02_[lane_index] =
              workspace_matrix_[static_cast<std::size_t>(matrix_begin + 2)];
          batch_3x3_a10_[lane_index] =
              workspace_matrix_[static_cast<std::size_t>(matrix_begin + 3)];
          batch_3x3_a11_[lane_index] =
              workspace_matrix_[static_cast<std::size_t>(matrix_begin + 4)];
          batch_3x3_a12_[lane_index] =
              workspace_matrix_[static_cast<std::size_t>(matrix_begin + 5)];
          batch_3x3_a20_[lane_index] =
              workspace_matrix_[static_cast<std::size_t>(matrix_begin + 6)];
          batch_3x3_a21_[lane_index] =
              workspace_matrix_[static_cast<std::size_t>(matrix_begin + 7)];
          batch_3x3_a22_[lane_index] =
              workspace_matrix_[static_cast<std::size_t>(matrix_begin + 8)];
          batch_3x3_rhs_constant0_[lane_index] =
              workspace_rhs_constant_[static_cast<std::size_t>(unknown_begin)];
          batch_3x3_rhs_constant1_[lane_index] =
              workspace_rhs_constant_[static_cast<std::size_t>(unknown_begin + 1)];
          batch_3x3_rhs_constant2_[lane_index] =
              workspace_rhs_constant_[static_cast<std::size_t>(unknown_begin + 2)];
          batch_3x3_rhs_inlet_pressure0_[lane_index] =
              workspace_rhs_inlet_pressure_[static_cast<std::size_t>(unknown_begin)];
          batch_3x3_rhs_inlet_pressure1_[lane_index] =
              workspace_rhs_inlet_pressure_[static_cast<std::size_t>(unknown_begin + 1)];
          batch_3x3_rhs_inlet_pressure2_[lane_index] =
              workspace_rhs_inlet_pressure_[static_cast<std::size_t>(unknown_begin + 2)];
        }
      };

      const auto write_2x2_batch_solution_to_workspace = [&](const ElementGroup& group)
      {
        FOUR_C_ASSERT_ALWAYS(group.block_size == 2,
            "TreeNewtonLinearSolver can write only 2x2 batch solutions, got block size {}.",
            group.block_size);
        for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
        {
          const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group.begin);
          const int unknown_begin = grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
          workspace_intercept_[static_cast<std::size_t>(unknown_begin)] =
              batch_2x2_intercept0_[lane_index];
          workspace_intercept_[static_cast<std::size_t>(unknown_begin + 1)] =
              batch_2x2_intercept1_[lane_index];
          workspace_slope_[static_cast<std::size_t>(unknown_begin)] = batch_2x2_slope0_[lane_index];
          workspace_slope_[static_cast<std::size_t>(unknown_begin + 1)] =
              batch_2x2_slope1_[lane_index];
        }
      };

      const auto write_3x3_batch_solution_to_workspace = [&](const ElementGroup& group)
      {
        FOUR_C_ASSERT_ALWAYS(group.block_size == 3,
            "TreeNewtonLinearSolver can write only 3x3 batch solutions, got block size {}.",
            group.block_size);
        for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
        {
          const std::size_t lane_index = static_cast<std::size_t>(grouped_index - group.begin);
          const int unknown_begin = grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
          workspace_intercept_[static_cast<std::size_t>(unknown_begin)] =
              batch_3x3_intercept0_[lane_index];
          workspace_intercept_[static_cast<std::size_t>(unknown_begin + 1)] =
              batch_3x3_intercept1_[lane_index];
          workspace_intercept_[static_cast<std::size_t>(unknown_begin + 2)] =
              batch_3x3_intercept2_[lane_index];
          workspace_slope_[static_cast<std::size_t>(unknown_begin)] = batch_3x3_slope0_[lane_index];
          workspace_slope_[static_cast<std::size_t>(unknown_begin + 1)] =
              batch_3x3_slope1_[lane_index];
          workspace_slope_[static_cast<std::size_t>(unknown_begin + 2)] =
              batch_3x3_slope2_[lane_index];
        }
      };

      {
        const auto bottom_up_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
        // Bottom-up pass: assemble each element block after child relations are known, then
        // condense it to the affine inlet relation consumed by its parent.
        if (use_scalar_tree_solve_)
        {
          if (profile_ != nullptr)
          {
            ++profile_->scalar_group_count;
          }
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
              if (group.block_size == 2 && group.child_count == 0)
              {
                assemble_2x2_leaf_group(group);
              }
              else if (group.block_size == 2 && group.child_count == 1)
              {
                assemble_2x2_one_child_group(group);
              }
              else if (group.block_size == 2 && group.child_count == 2)
              {
                assemble_2x2_two_child_group(group);
              }
              else if (group.block_size == 3 && group.child_count == 0)
              {
                assemble_3x3_leaf_group(group);
              }
              else if (group.block_size == 3 && group.child_count == 1)
              {
                assemble_3x3_one_child_group(group);
              }
              else if (group.block_size == 3 && group.child_count == 2)
              {
                assemble_3x3_two_child_group(group);
              }
              else
              {
                if (profile_ != nullptr)
                {
                  ++profile_->scalar_group_count;
                }
                assemble_group(group);
              }

              if (group.block_size == 2)
              {
#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
                const bool assembled_2x2_direct_soa =
                    coefficient_source_ ==
                    TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks;
#else
                const bool assembled_2x2_direct_soa = false;
#endif
                if (!assembled_2x2_direct_soa)
                {
                  pack_2x2_group_from_workspace(group);
                }
                const auto dense_solve_start =
                    profile_ != nullptr ? Clock::now() : Clock::time_point{};
                solve_2x2_batch(group.begin, group.end, grouped_element_indices_, batch_2x2_a00_,
                    batch_2x2_a01_, batch_2x2_a10_, batch_2x2_a11_, batch_2x2_rhs_constant0_,
                    batch_2x2_rhs_constant1_, batch_2x2_rhs_inlet_pressure0_,
                    batch_2x2_rhs_inlet_pressure1_, batch_2x2_intercept0_, batch_2x2_intercept1_,
                    batch_2x2_slope0_, batch_2x2_slope1_, batch_2x2_fallback_lanes_, profile_,
                    pivot_tolerance_, error_on_dense_fallback_, element_context_);
                write_2x2_batch_solution_to_workspace(group);
                if (profile_ != nullptr)
                {
                  profile_->dense_solve_time += elapsed_seconds(dense_solve_start);
                  profile_->dense_solve_count +=
                      static_cast<std::uint64_t>(group.end - group.begin);
                }
              }
              else if (group.block_size == 3)
              {
#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
                const bool assembled_3x3_direct_soa =
                    coefficient_source_ ==
                        TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks &&
                    group.child_count >= 0 && group.child_count <= 2;
#else
                const bool assembled_3x3_direct_soa = false;
#endif
                if (!assembled_3x3_direct_soa)
                {
                  pack_3x3_group_from_workspace(group);
                }
                const auto dense_solve_start =
                    profile_ != nullptr ? Clock::now() : Clock::time_point{};
                solve_3x3_batch(group.begin, group.end, grouped_element_indices_, batch_3x3_a00_,
                    batch_3x3_a01_, batch_3x3_a02_, batch_3x3_a10_, batch_3x3_a11_, batch_3x3_a12_,
                    batch_3x3_a20_, batch_3x3_a21_, batch_3x3_a22_, batch_3x3_rhs_constant0_,
                    batch_3x3_rhs_constant1_, batch_3x3_rhs_constant2_,
                    batch_3x3_rhs_inlet_pressure0_, batch_3x3_rhs_inlet_pressure1_,
                    batch_3x3_rhs_inlet_pressure2_, batch_3x3_intercept0_, batch_3x3_intercept1_,
                    batch_3x3_intercept2_, batch_3x3_slope0_, batch_3x3_slope1_, batch_3x3_slope2_,
                    batch_3x3_fallback_lanes_, profile_, pivot_tolerance_, error_on_dense_fallback_,
                    element_context_);
                write_3x3_batch_solution_to_workspace(group);
                if (profile_ != nullptr)
                {
                  profile_->dense_solve_time += elapsed_seconds(dense_solve_start);
                  profile_->dense_solve_count +=
                      static_cast<std::uint64_t>(group.end - group.begin);
                }
              }
              else
              {
                if (profile_ != nullptr)
                {
                  ++profile_->scalar_group_count;
                  profile_->unsupported_block_fallback_count +=
                      static_cast<std::uint64_t>(group.end - group.begin);
                }
                for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
                {
                  solve_scalar_element(
                      grouped_element_indices_[static_cast<std::size_t>(grouped_index)]);
                }
              }

              if (group.block_size == 2)
              {
                write_2x2_subtree_relation_group(group);
              }
              else if (group.block_size == 3)
              {
                write_3x3_subtree_relation_group(group);
              }
              else
              {
                write_subtree_relation_group(group);
              }
            }
          }
        }
        if (profile_ != nullptr)
        {
          profile_->bottom_up_time += elapsed_seconds(bottom_up_start);
        }
      }

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
        FOUR_C_ASSERT_ALWAYS(inlet_pressure_stamp_[element_index_size] != solve_stamp,
            "TreeNewtonLinearSolver inlet-pressure correction for element {} was written more than "
            "once.",
            global_element_id_[element_index_size] + 1);
        inlet_pressure_by_element_[element_index_size] = value;
        inlet_pressure_stamp_[element_index_size] = solve_stamp;
      };

      {
        const double root_boundary_coeff =
            required_matrix_value(coefficients, root_boundary_coefficient_,
                root_boundary_coefficient_value_, pivot_tolerance_, "root inlet boundary");
        const std::size_t root_element_index_size =
            static_cast<std::size_t>(tree_metadata_.root_element_index);
        const double root_boundary_rhs = rhs_value(residual, root_boundary_row_);
        double root_inlet_pressure = 0.0;
        if (root_boundary_variable_ == BoundaryConditions::ConstrainedVariable::Pressure)
        {
          root_inlet_pressure = root_boundary_rhs / root_boundary_coeff;
        }
        else if (root_boundary_variable_ == BoundaryConditions::ConstrainedVariable::Flow)
        {
          const double root_subtree_slope = subtree_relation_G_[root_element_index_size];
          FOUR_C_ASSERT_ALWAYS(std::abs(root_subtree_slope) > pivot_tolerance_,
              "TreeNewtonLinearSolver root inlet flow boundary cannot determine the root inlet "
              "pressure because the condensed root flow relation has a near-zero pressure slope.");
          root_inlet_pressure = (root_boundary_rhs / root_boundary_coeff -
                                    subtree_relation_h_[root_element_index_size]) /
                                root_subtree_slope;
        }
        else
        {
          FOUR_C_THROW("TreeNewtonLinearSolver found an unsupported root inlet boundary type.");
        }
        set_inlet_pressure(tree_metadata_.root_element_index, root_inlet_pressure);
      }

      double* const delta_values = delta.get_values();
      const auto set_delta_local_value = [&](int local_dof_id, double value)
      { delta_values[static_cast<std::size_t>(local_dof_id)] = value; };

      // Top-down pass: start from the root inlet-pressure correction, recover local unknowns, and
      // propagate child inlet-pressure corrections through stored pressure-continuity maps.
      const auto recover_top_down_element = [&](int element_index)
      {
        const std::size_t element_index_size = static_cast<std::size_t>(element_index);
        FOUR_C_ASSERT_ALWAYS(inlet_pressure_stamp_[element_index_size] == solve_stamp,
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

      const auto recover_2x2_top_down_group = [&](const ElementGroup& group)
      {
        FOUR_C_ASSERT_ALWAYS(group.block_size == 2,
            "TreeNewtonLinearSolver 2x2 top-down group has block size {}.", group.block_size);
        FOUR_C_ASSERT_ALWAYS(group.child_count >= 0 && group.child_count <= 2,
            "TreeNewtonLinearSolver 2x2 top-down group has unsupported child count {}.",
            group.child_count);

        [[maybe_unused]] const auto recover_2x2_top_down_lane = [&](int grouped_index)
        {
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const std::size_t element_index_size = static_cast<std::size_t>(element_index);
          FOUR_C_ASSERT_ALWAYS(inlet_pressure_stamp_[element_index_size] == solve_stamp,
              "TreeNewtonLinearSolver missing inlet-pressure correction for element {}.",
              global_element_id_[element_index_size] + 1);

          const double inlet_pressure = inlet_pressure_by_element_[element_index_size];
          set_delta_local_value(
              inlet_pressure_correction_local_dof_ids_[element_index_size], inlet_pressure);

          const int unknown_begin = unknown_offset_[element_index_size];
          const std::size_t unknown0 = static_cast<std::size_t>(unknown_begin);
          const std::size_t unknown1 = static_cast<std::size_t>(unknown_begin + 1);
          const double value0 =
              workspace_slope_[unknown0] * inlet_pressure + workspace_intercept_[unknown0];
          const double value1 =
              workspace_slope_[unknown1] * inlet_pressure + workspace_intercept_[unknown1];
          set_delta_local_value(unknown_correction_local_dof_ids_[unknown0], value0);
          set_delta_local_value(unknown_correction_local_dof_ids_[unknown1], value1);

          if (group.child_count == 0)
          {
            return;
          }

          const int outlet_pressure_index = outlet_pressure_unknown_index_[element_index_size];
          const std::size_t outlet_pressure_unknown =
              static_cast<std::size_t>(unknown_begin + outlet_pressure_index);
          const double outlet_pressure =
              workspace_slope_[outlet_pressure_unknown] * inlet_pressure +
              workspace_intercept_[outlet_pressure_unknown];
          const int child_begin = child_interface_offset_[element_index_size];
          for (int child_slot = 0; child_slot < group.child_count; ++child_slot)
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

#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
        const auto recover_2x2_top_down_chunk = [&](int chunk_begin, int valid_end)
        {
          tree_solver_simd::for_each_valid_lane(chunk_begin, valid_end,
              [&](int grouped_index, int /*lane*/)
              {
                const int element_index =
                    grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
                const std::size_t element_index_size = static_cast<std::size_t>(element_index);
                FOUR_C_ASSERT_ALWAYS(inlet_pressure_stamp_[element_index_size] == solve_stamp,
                    "TreeNewtonLinearSolver missing inlet-pressure correction for element {}.",
                    global_element_id_[element_index_size] + 1);

                set_delta_local_value(inlet_pressure_correction_local_dof_ids_[element_index_size],
                    inlet_pressure_by_element_[element_index_size]);
              });

          const tree_solver_simd::Double inlet_pressure =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int element_index =
                        grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
                    return inlet_pressure_by_element_[static_cast<std::size_t>(element_index)];
                  });
          const tree_solver_simd::Double slope0 =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int unknown_begin =
                        grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                    return workspace_slope_[static_cast<std::size_t>(unknown_begin)];
                  });
          const tree_solver_simd::Double slope1 =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int unknown_begin =
                        grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                    return workspace_slope_[static_cast<std::size_t>(unknown_begin + 1)];
                  });
          const tree_solver_simd::Double intercept0 =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int unknown_begin =
                        grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                    return workspace_intercept_[static_cast<std::size_t>(unknown_begin)];
                  });
          const tree_solver_simd::Double intercept1 =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int unknown_begin =
                        grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                    return workspace_intercept_[static_cast<std::size_t>(unknown_begin + 1)];
                  });

          const tree_solver_simd::Double value0 = slope0 * inlet_pressure + intercept0;
          const tree_solver_simd::Double value1 = slope1 * inlet_pressure + intercept1;
          tree_solver_simd::scatter_valid(value0, chunk_begin, valid_end,
              [&](int grouped_index, double value)
              {
                const int unknown_begin =
                    grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                set_delta_local_value(
                    unknown_correction_local_dof_ids_[static_cast<std::size_t>(unknown_begin)],
                    value);
              });
          tree_solver_simd::scatter_valid(value1, chunk_begin, valid_end,
              [&](int grouped_index, double value)
              {
                const int unknown_begin =
                    grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                set_delta_local_value(
                    unknown_correction_local_dof_ids_[static_cast<std::size_t>(unknown_begin + 1)],
                    value);
              });

          if (group.child_count == 0)
          {
            return;
          }

          const tree_solver_simd::Double outlet_slope =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int element_index =
                        grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
                    const std::size_t element_index_size = static_cast<std::size_t>(element_index);
                    const int unknown_begin =
                        grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                    const int outlet_pressure_index =
                        outlet_pressure_unknown_index_[element_index_size];
                    return workspace_slope_[static_cast<std::size_t>(
                        unknown_begin + outlet_pressure_index)];
                  });
          const tree_solver_simd::Double outlet_intercept =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int element_index =
                        grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
                    const std::size_t element_index_size = static_cast<std::size_t>(element_index);
                    const int unknown_begin =
                        grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                    const int outlet_pressure_index =
                        outlet_pressure_unknown_index_[element_index_size];
                    return workspace_intercept_[static_cast<std::size_t>(
                        unknown_begin + outlet_pressure_index)];
                  });
          const tree_solver_simd::Double outlet_pressure =
              outlet_slope * inlet_pressure + outlet_intercept;

          for (int child_slot = 0; child_slot < group.child_count; ++child_slot)
          {
            const tree_solver_simd::Double child_pressure_slope = tree_solver_simd::gather_or(
                chunk_begin, valid_end, 0.0,
                [&](int grouped_index)
                {
                  const int child_interface_index =
                      grouped_child_begin_[static_cast<std::size_t>(grouped_index)] + child_slot;
                  return child_pressure_slope_[static_cast<std::size_t>(child_interface_index)];
                });
            const tree_solver_simd::Double child_pressure_intercept = tree_solver_simd::gather_or(
                chunk_begin, valid_end, 0.0,
                [&](int grouped_index)
                {
                  const int child_interface_index =
                      grouped_child_begin_[static_cast<std::size_t>(grouped_index)] + child_slot;
                  return child_pressure_intercept_[static_cast<std::size_t>(child_interface_index)];
                });
            const tree_solver_simd::Double child_pressure =
                child_pressure_slope * outlet_pressure + child_pressure_intercept;
            tree_solver_simd::scatter_valid(child_pressure, chunk_begin, valid_end,
                [&](int grouped_index, double value)
                {
                  const int child_interface_index =
                      grouped_child_begin_[static_cast<std::size_t>(grouped_index)] + child_slot;
                  set_inlet_pressure(
                      child_element_index_[static_cast<std::size_t>(child_interface_index)], value);
                });
          }
        };

        int grouped_index = group.begin;
        const int padded_end = tree_solver_simd::padded_chunk_end(group.begin, group.end);
        for (; grouped_index < padded_end; grouped_index += tree_solver_simd::width())
        {
          recover_2x2_top_down_chunk(grouped_index, group.end);
        }
#else
        for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
        {
          recover_2x2_top_down_lane(grouped_index);
        }
#endif
      };

      const auto recover_3x3_top_down_group = [&](const ElementGroup& group)
      {
        FOUR_C_ASSERT_ALWAYS(group.block_size == 3,
            "TreeNewtonLinearSolver 3x3 top-down group has block size {}.", group.block_size);
        FOUR_C_ASSERT_ALWAYS(group.child_count >= 0 && group.child_count <= 2,
            "TreeNewtonLinearSolver 3x3 top-down group has unsupported child count {}.",
            group.child_count);

        [[maybe_unused]] const auto recover_3x3_top_down_lane = [&](int grouped_index)
        {
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const std::size_t element_index_size = static_cast<std::size_t>(element_index);
          FOUR_C_ASSERT_ALWAYS(inlet_pressure_stamp_[element_index_size] == solve_stamp,
              "TreeNewtonLinearSolver missing inlet-pressure correction for element {}.",
              global_element_id_[element_index_size] + 1);

          const double inlet_pressure = inlet_pressure_by_element_[element_index_size];
          set_delta_local_value(
              inlet_pressure_correction_local_dof_ids_[element_index_size], inlet_pressure);

          const int unknown_begin = unknown_offset_[element_index_size];
          const std::size_t unknown0 = static_cast<std::size_t>(unknown_begin);
          const std::size_t unknown1 = static_cast<std::size_t>(unknown_begin + 1);
          const std::size_t unknown2 = static_cast<std::size_t>(unknown_begin + 2);
          const double value0 =
              workspace_slope_[unknown0] * inlet_pressure + workspace_intercept_[unknown0];
          const double value1 =
              workspace_slope_[unknown1] * inlet_pressure + workspace_intercept_[unknown1];
          const double value2 =
              workspace_slope_[unknown2] * inlet_pressure + workspace_intercept_[unknown2];
          set_delta_local_value(unknown_correction_local_dof_ids_[unknown0], value0);
          set_delta_local_value(unknown_correction_local_dof_ids_[unknown1], value1);
          set_delta_local_value(unknown_correction_local_dof_ids_[unknown2], value2);

          if (group.child_count == 0)
          {
            return;
          }

          const int outlet_pressure_index = outlet_pressure_unknown_index_[element_index_size];
          const std::size_t outlet_pressure_unknown =
              static_cast<std::size_t>(unknown_begin + outlet_pressure_index);
          const double outlet_pressure =
              workspace_slope_[outlet_pressure_unknown] * inlet_pressure +
              workspace_intercept_[outlet_pressure_unknown];
          const int child_begin = child_interface_offset_[element_index_size];
          for (int child_slot = 0; child_slot < group.child_count; ++child_slot)
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

#if FOUR_C_REDUCED_LUNG_HAS_EXPERIMENTAL_SIMD
        const auto recover_3x3_top_down_chunk = [&](int chunk_begin, int valid_end)
        {
          tree_solver_simd::for_each_valid_lane(chunk_begin, valid_end,
              [&](int grouped_index, int /*lane*/)
              {
                const int element_index =
                    grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
                const std::size_t element_index_size = static_cast<std::size_t>(element_index);
                FOUR_C_ASSERT_ALWAYS(inlet_pressure_stamp_[element_index_size] == solve_stamp,
                    "TreeNewtonLinearSolver missing inlet-pressure correction for element {}.",
                    global_element_id_[element_index_size] + 1);

                set_delta_local_value(inlet_pressure_correction_local_dof_ids_[element_index_size],
                    inlet_pressure_by_element_[element_index_size]);
              });

          const tree_solver_simd::Double inlet_pressure =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int element_index =
                        grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
                    return inlet_pressure_by_element_[static_cast<std::size_t>(element_index)];
                  });
          const tree_solver_simd::Double slope0 =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int unknown_begin =
                        grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                    return workspace_slope_[static_cast<std::size_t>(unknown_begin)];
                  });
          const tree_solver_simd::Double slope1 =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int unknown_begin =
                        grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                    return workspace_slope_[static_cast<std::size_t>(unknown_begin + 1)];
                  });
          const tree_solver_simd::Double slope2 =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int unknown_begin =
                        grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                    return workspace_slope_[static_cast<std::size_t>(unknown_begin + 2)];
                  });
          const tree_solver_simd::Double intercept0 =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int unknown_begin =
                        grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                    return workspace_intercept_[static_cast<std::size_t>(unknown_begin)];
                  });
          const tree_solver_simd::Double intercept1 =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int unknown_begin =
                        grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                    return workspace_intercept_[static_cast<std::size_t>(unknown_begin + 1)];
                  });
          const tree_solver_simd::Double intercept2 =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int unknown_begin =
                        grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                    return workspace_intercept_[static_cast<std::size_t>(unknown_begin + 2)];
                  });

          const tree_solver_simd::Double value0 = slope0 * inlet_pressure + intercept0;
          const tree_solver_simd::Double value1 = slope1 * inlet_pressure + intercept1;
          const tree_solver_simd::Double value2 = slope2 * inlet_pressure + intercept2;
          tree_solver_simd::scatter_valid(value0, chunk_begin, valid_end,
              [&](int grouped_index, double value)
              {
                const int unknown_begin =
                    grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                set_delta_local_value(
                    unknown_correction_local_dof_ids_[static_cast<std::size_t>(unknown_begin)],
                    value);
              });
          tree_solver_simd::scatter_valid(value1, chunk_begin, valid_end,
              [&](int grouped_index, double value)
              {
                const int unknown_begin =
                    grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                set_delta_local_value(
                    unknown_correction_local_dof_ids_[static_cast<std::size_t>(unknown_begin + 1)],
                    value);
              });
          tree_solver_simd::scatter_valid(value2, chunk_begin, valid_end,
              [&](int grouped_index, double value)
              {
                const int unknown_begin =
                    grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                set_delta_local_value(
                    unknown_correction_local_dof_ids_[static_cast<std::size_t>(unknown_begin + 2)],
                    value);
              });

          if (group.child_count == 0)
          {
            return;
          }

          const tree_solver_simd::Double outlet_slope =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int element_index =
                        grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
                    const std::size_t element_index_size = static_cast<std::size_t>(element_index);
                    const int unknown_begin =
                        grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                    const int outlet_pressure_index =
                        outlet_pressure_unknown_index_[element_index_size];
                    return workspace_slope_[static_cast<std::size_t>(
                        unknown_begin + outlet_pressure_index)];
                  });
          const tree_solver_simd::Double outlet_intercept =
              tree_solver_simd::gather_or(chunk_begin, valid_end, 0.0,
                  [&](int grouped_index)
                  {
                    const int element_index =
                        grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
                    const std::size_t element_index_size = static_cast<std::size_t>(element_index);
                    const int unknown_begin =
                        grouped_unknown_begin_[static_cast<std::size_t>(grouped_index)];
                    const int outlet_pressure_index =
                        outlet_pressure_unknown_index_[element_index_size];
                    return workspace_intercept_[static_cast<std::size_t>(
                        unknown_begin + outlet_pressure_index)];
                  });
          const tree_solver_simd::Double outlet_pressure =
              outlet_slope * inlet_pressure + outlet_intercept;

          for (int child_slot = 0; child_slot < group.child_count; ++child_slot)
          {
            const tree_solver_simd::Double child_pressure_slope = tree_solver_simd::gather_or(
                chunk_begin, valid_end, 0.0,
                [&](int grouped_index)
                {
                  const int child_interface_index =
                      grouped_child_begin_[static_cast<std::size_t>(grouped_index)] + child_slot;
                  return child_pressure_slope_[static_cast<std::size_t>(child_interface_index)];
                });
            const tree_solver_simd::Double child_pressure_intercept = tree_solver_simd::gather_or(
                chunk_begin, valid_end, 0.0,
                [&](int grouped_index)
                {
                  const int child_interface_index =
                      grouped_child_begin_[static_cast<std::size_t>(grouped_index)] + child_slot;
                  return child_pressure_intercept_[static_cast<std::size_t>(child_interface_index)];
                });
            const tree_solver_simd::Double child_pressure =
                child_pressure_slope * outlet_pressure + child_pressure_intercept;
            tree_solver_simd::scatter_valid(child_pressure, chunk_begin, valid_end,
                [&](int grouped_index, double value)
                {
                  const int child_interface_index =
                      grouped_child_begin_[static_cast<std::size_t>(grouped_index)] + child_slot;
                  set_inlet_pressure(
                      child_element_index_[static_cast<std::size_t>(child_interface_index)], value);
                });
          }
        };

        int grouped_index = group.begin;
        const int padded_end = tree_solver_simd::padded_chunk_end(group.begin, group.end);
        for (; grouped_index < padded_end; grouped_index += tree_solver_simd::width())
        {
          recover_3x3_top_down_chunk(grouped_index, group.end);
        }
#else
        for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
        {
          recover_3x3_top_down_lane(grouped_index);
        }
#endif
      };

      const auto recover_top_down_group = [&](const ElementGroup& group)
      {
        const int group_size = group.end - group.begin;
        FOUR_C_ASSERT_ALWAYS(group_size >= 0,
            "TreeNewtonLinearSolver top-down group has invalid range [{}, {}).", group.begin,
            group.end);
        FOUR_C_ASSERT_ALWAYS(static_cast<int>(top_down_inlet_pressure_.size()) >= group_size &&
                                 static_cast<int>(top_down_unknown_values_.size()) >= group_size &&
                                 static_cast<int>(top_down_outlet_pressure_.size()) >= group_size &&
                                 static_cast<int>(top_down_child_pressure_.size()) >= group_size,
            "TreeNewtonLinearSolver top-down batch workspace is too small.");

        for (int lane = 0; lane < group_size; ++lane)
        {
          const int grouped_index = group.begin + lane;
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const std::size_t element_index_size = static_cast<std::size_t>(element_index);
          FOUR_C_ASSERT_ALWAYS(inlet_pressure_stamp_[element_index_size] == solve_stamp,
              "TreeNewtonLinearSolver missing inlet-pressure correction for element {}.",
              global_element_id_[element_index_size] + 1);
          top_down_inlet_pressure_[static_cast<std::size_t>(lane)] =
              inlet_pressure_by_element_[element_index_size];
        }

        for (int lane = 0; lane < group_size; ++lane)
        {
          const int grouped_index = group.begin + lane;
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
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
          const int element_index =
              grouped_element_indices_[static_cast<std::size_t>(grouped_index)];
          const std::size_t element_index_size = static_cast<std::size_t>(element_index);
          FOUR_C_ASSERT_ALWAYS(child_interface_count_[element_index_size] > 0,
              "TreeNewtonLinearSolver found no children while recovering element {}.",
              global_element_id_[element_index_size] + 1);
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
            set_inlet_pressure(
                child_element_index_[static_cast<std::size_t>(child_interface_index)],
                top_down_child_pressure_[static_cast<std::size_t>(lane)]);
          }
        }
      };

      {
        const auto top_down_start = profile_ != nullptr ? Clock::now() : Clock::time_point{};
        constexpr int top_down_scalar_group_threshold = 2;
        if (use_scalar_tree_solve_)
        {
          if (profile_ != nullptr)
          {
            ++profile_->scalar_group_count;
          }
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
              FOUR_C_ASSERT_ALWAYS(group_size >= 0,
                  "TreeNewtonLinearSolver top-down group has invalid range [{}, {}).", group.begin,
                  group.end);
              const bool use_scalar_top_down_group =
                  !force_batch_tree_solve_ && group_size <= top_down_scalar_group_threshold;
              if (use_scalar_top_down_group)
              {
                if (profile_ != nullptr)
                {
                  ++profile_->scalar_group_count;
                }
                for (int grouped_index = group.begin; grouped_index < group.end; ++grouped_index)
                {
                  recover_top_down_element(
                      grouped_element_indices_[static_cast<std::size_t>(grouped_index)]);
                }
              }
              else
              {
                if (group.block_size == 2 && group.child_count <= 2)
                {
                  recover_2x2_top_down_group(group);
                }
                else if (group.block_size == 3 && group.child_count <= 2)
                {
                  recover_3x3_top_down_group(group);
                }
                else
                {
                  if (profile_ != nullptr)
                  {
                    ++profile_->scalar_group_count;
                  }
                  recover_top_down_group(group);
                }
              }
            }
          }
        }
        if (profile_ != nullptr)
        {
          profile_->top_down_time += elapsed_seconds(top_down_start);
        }
      }
      if (profile_ != nullptr)
      {
        profile_->total_solve_time += elapsed_seconds(solve_start);
        ++profile_->solve_count;
      }
    };

    if (coefficient_source_ == TreeNewtonLinearSolverCoefficientSource::StructuredTreeBlocks)
    {
      if (tree_linearization_ != nullptr)
      {
        const StructuredTreeCoefficientProvider structured_coefficients(
            *tree_linearization_, profile_);
        solve_with_coefficients(structured_coefficients);
      }
      else
      {
        const DirectTreeCoefficientProvider direct_coefficients;
        solve_with_coefficients(direct_coefficients);
      }
    }
    else
    {
      const SparseTreeCoefficientProvider sparse_coefficients(*validation_jacobian_, profile_);
      solve_with_coefficients(sparse_coefficients);
    }
  }
}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE
