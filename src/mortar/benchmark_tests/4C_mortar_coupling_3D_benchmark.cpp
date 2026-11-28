// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_mortar_coupling3d.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_input.hpp"
#include "4C_mortar_node.hpp"
#include "4C_utils_parameter_list.hpp"

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  using Point = std::array<double, 3>;
  using Polygon = std::vector<Point>;

  /**
   * @brief Owns a mortar element and the nodes referenced by it.
   *
   * Construction belongs to benchmark setup and is deliberately outside the timed regions.
   */
  struct OwnedMortarElement
  {
    OwnedMortarElement(const int element_id, const int first_node_id,
        const Core::FE::CellType shape, const Polygon& coordinates, const bool is_source)
    {
      node_ids.reserve(coordinates.size());
      nodes.reserve(coordinates.size());
      node_pointers.reserve(coordinates.size());

      for (std::size_t node = 0; node < coordinates.size(); ++node)
      {
        const int node_id = first_node_id + static_cast<int>(node);
        const std::vector<int> dofs{3 * node_id, 3 * node_id + 1, 3 * node_id + 2};

        node_ids.push_back(node_id);
        nodes.emplace_back(std::make_unique<Mortar::Node>(node_id,
            std::span<const double>{coordinates[node].data(), coordinates[node].size()}, 0, dofs,
            is_source));
        node_pointers.push_back(nodes.back().get());
      }

      element = std::make_unique<Mortar::Element>(
          element_id, 0, shape, static_cast<int>(coordinates.size()), node_ids.data(), is_source);
      element->build_nodal_pointers(node_pointers.data());
      element->initialize_data_container();
      element->mo_data().area() = element->compute_area();
    }

    OwnedMortarElement(const OwnedMortarElement&) = delete;
    OwnedMortarElement& operator=(const OwnedMortarElement&) = delete;

    std::vector<int> node_ids;
    std::vector<std::unique_ptr<Mortar::Node>> nodes;
    std::vector<Core::Nodes::Node*> node_pointers;
    std::unique_ptr<Mortar::Element> element;
  };

  Teuchos::ParameterList coupling_parameters(const Mortar::Triangulation triangulation)
  {
    Teuchos::ParameterList parameters;
    Core::Utils::add_enum_class_to_parameter_list<Mortar::ShapeFcn>(
        "LM_SHAPEFCN", Mortar::shape_standard, parameters);
    Core::Utils::add_enum_class_to_parameter_list<Mortar::LagMultQuad>(
        "LM_QUAD", Mortar::lagmult_undefined, parameters);
    Core::Utils::add_enum_class_to_parameter_list<Mortar::Triangulation>(
        "TRIANGULATION", triangulation, parameters);

    return parameters;
  }

  // We test different geometric cases of overlaps between triangles and quadrilaterals
  enum class GeometryCase
  {
    triangle_six_vertex_overlap,
    quadrilateral_four_vertex_overlap,
    quadrilateral_eight_vertex_overlap,
    disjoint_triangles,
    warped_quadrilaterals,
  };

  struct CouplingGeometry
  {
    Core::FE::CellType source_shape;
    Polygon source;
    Core::FE::CellType target_shape;
    Polygon target;
    bool expected_overlap;
  };

  // The following describes the different and fixed geometric properties of the overlap cases
  CouplingGeometry coupling_geometry(const GeometryCase geometry)
  {
    switch (geometry)
    {
      case GeometryCase::triangle_six_vertex_overlap:
        return {Core::FE::CellType::tri3, {{0.0, 0.0, 0.0}, {7.0, 0.0, 0.0}, {1.0, 6.0, 0.0}},
            Core::FE::CellType::tri3, {{1.0, 8.0, 0.0}, {3.5, 3.0, 0.0}, {0.0, -2.0, 0.0}}, true};

      case GeometryCase::quadrilateral_four_vertex_overlap:
        return {Core::FE::CellType::quad4,
            {{0.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {2.0, 2.0, 0.0}, {0.0, 1.5, 0.0}},
            Core::FE::CellType::quad4,
            {{-1.0, -1.0, 0.0}, {-1.0, 3.0, 0.0}, {4.0, 3.0, 0.0}, {4.0, -1.0, 0.0}}, true};

      case GeometryCase::quadrilateral_eight_vertex_overlap:
        return {Core::FE::CellType::quad4,
            {{-3.0, -1.0, 0.0}, {3.0, -1.0, 0.0}, {3.0, 1.0, 0.0}, {-3.0, 1.0, 0.0}},
            Core::FE::CellType::quad4,
            {{0.8, -3.6, 0.0}, {-3.2, -0.6, 0.0}, {-0.8, 2.6, 0.0}, {3.2, -0.4, 0.0}}, true};

      case GeometryCase::disjoint_triangles:
        return {Core::FE::CellType::tri3, {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}},
            Core::FE::CellType::tri3, {{3.0, 0.0, 0.0}, {3.0, 1.0, 0.0}, {4.0, 0.0, 0.0}}, false};

      case GeometryCase::warped_quadrilaterals:
        return {Core::FE::CellType::quad4,
            {{-1.0, -1.0, 0.0}, {1.0, -1.0, 0.3}, {1.0, 1.0, -0.2}, {-1.0, 1.0, 0.1}},
            Core::FE::CellType::quad4,
            {{-2.0, -2.0, -1.0}, {-2.0, 2.0, -1.0}, {2.0, 2.0, -1.0}, {2.0, -2.0, -1.0}}, true};
    }

    throw std::logic_error("Unknown mortar coupling benchmark geometry.");
  }

  /**
   * Measures construction of the auxiliary plane independently of the remaining pipeline.
   */
  static void mortar_coupling3d_auxiliary_plane(
      benchmark::State& state, const GeometryCase geometry_case)
  {
    const CouplingGeometry geometry = coupling_geometry(geometry_case);
    Core::FE::Discretization discretization{"mortar_coupling3d_benchmark", MPI_COMM_SELF, 3};
    OwnedMortarElement source(0, 0, geometry.source_shape, geometry.source, true);
    OwnedMortarElement target(1, 100, geometry.target_shape, geometry.target, false);
    auto parameters = coupling_parameters(Mortar::triangulation_center);
    Mortar::Coupling3d coupling(
        discretization, 3, false, parameters, *source.element, *target.element);

    for (auto _ : state)
    {
      bool successful = coupling.auxiliary_plane();
      benchmark::DoNotOptimize(successful);
      benchmark::DoNotOptimize(coupling.auxc()[0]);
      benchmark::DoNotOptimize(coupling.auxn()[0]);
      benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations());
  }

  /**
   * Measures projection of all source and target vertices after the auxiliary plane is known.
   */
  static void mortar_coupling3d_project_vertices(
      benchmark::State& state, const GeometryCase geometry_case)
  {
    const CouplingGeometry geometry = coupling_geometry(geometry_case);
    Core::FE::Discretization discretization{"mortar_coupling3d_benchmark", MPI_COMM_SELF, 3};
    OwnedMortarElement source(0, 0, geometry.source_shape, geometry.source, true);
    OwnedMortarElement target(1, 100, geometry.target_shape, geometry.target, false);
    auto parameters = coupling_parameters(Mortar::triangulation_center);
    Mortar::Coupling3d coupling(
        discretization, 3, false, parameters, *source.element, *target.element);

    if (!coupling.auxiliary_plane())
    {
      state.SkipWithError("Could not construct the mortar auxiliary plane.");
      return;
    }

    std::vector<Mortar::Vertex> projected_source_vertices;
    std::vector<Mortar::Vertex> projected_target_vertices;
    projected_source_vertices.reserve(geometry.source.size());
    projected_target_vertices.reserve(geometry.target.size());

    for (auto _ : state)
    {
      projected_source_vertices.clear();
      projected_target_vertices.clear();

      coupling.project_element_onto_auxiliary_plane(
          *source.element, projected_source_vertices, Mortar::Vertex::source);
      coupling.project_element_onto_auxiliary_plane(
          *target.element, projected_target_vertices, Mortar::Vertex::projtarget);

      benchmark::DoNotOptimize(projected_source_vertices.data());
      benchmark::DoNotOptimize(projected_source_vertices.size());
      benchmark::DoNotOptimize(projected_target_vertices.data());
      benchmark::DoNotOptimize(projected_target_vertices.size());
      benchmark::ClobberMemory();
    }

    state.counters["vertices"] =
        static_cast<double>(geometry.source.size() + geometry.target.size());
    state.SetItemsProcessed(
        state.iterations() * static_cast<int64_t>(geometry.source.size() + geometry.target.size()));
  }

  /**
   * Measures the geometric coupling pipeline.
   *
   * Each measured call covers auxiliary-plane construction, projection, clipping, overlap
   * rejection, linearization bookkeeping, and triangulation, but not Gauss integration of D and M.
   */
  static void mortar_coupling3d_evaluate_reused(benchmark::State& state,
      const GeometryCase geometry_case, const Mortar::Triangulation triangulation)
  {
    const CouplingGeometry geometry = coupling_geometry(geometry_case);
    Core::FE::Discretization discretization{"mortar_coupling3d_benchmark", MPI_COMM_SELF, 3};
    OwnedMortarElement source(0, 0, geometry.source_shape, geometry.source, true);
    OwnedMortarElement target(1, 100, geometry.target_shape, geometry.target, false);
    auto parameters = coupling_parameters(triangulation);
    Mortar::Coupling3d coupling(
        discretization, 3, false, parameters, *source.element, *target.element);

    if (coupling.evaluate_coupling() != geometry.expected_overlap)
    {
      state.SkipWithError("Unexpected mortar coupling result for benchmark geometry.");
      return;
    }

    for (auto _ : state)
    {
      bool successful = coupling.evaluate_coupling();
      benchmark::DoNotOptimize(successful);
      benchmark::DoNotOptimize(coupling.clip().data());
      benchmark::DoNotOptimize(coupling.clip().size());
      benchmark::DoNotOptimize(coupling.cells().data());
      benchmark::DoNotOptimize(coupling.cells().size());
      benchmark::ClobberMemory();
    }

    state.counters["clip_vertices"] = static_cast<double>(coupling.clip().size());
    state.counters["integration_cells"] = static_cast<double>(coupling.cells().size());
    state.SetItemsProcessed(state.iterations());
  }

  /**
   * Measures the geometric coupling pipeline by constructing a fresh Coupling3d object for every
   * iteration.
   *
   * Element and discretization setup remain outside the timed region. The benchmark therefore
   * includes Coupling3d construction, its first allocations, evaluation, and destruction.
   */
  static void mortar_coupling3d_construct_and_evaluate(benchmark::State& state,
      const GeometryCase geometry_case, const Mortar::Triangulation triangulation)
  {
    const CouplingGeometry geometry = coupling_geometry(geometry_case);
    Core::FE::Discretization discretization{"mortar_coupling3d_benchmark", MPI_COMM_SELF, 3};
    OwnedMortarElement source(0, 0, geometry.source_shape, geometry.source, true);
    OwnedMortarElement target(1, 100, geometry.target_shape, geometry.target, false);
    auto parameters = coupling_parameters(triangulation);

    {
      Mortar::Coupling3d validation_coupling(
          discretization, 3, false, parameters, *source.element, *target.element);
      if (validation_coupling.evaluate_coupling() != geometry.expected_overlap)
      {
        state.SkipWithError("Unexpected mortar coupling result for benchmark geometry.");
        return;
      }
    }

    for (auto _ : state)
    {
      Mortar::Coupling3d coupling(
          discretization, 3, false, parameters, *source.element, *target.element);
      bool successful = coupling.evaluate_coupling();

      benchmark::DoNotOptimize(successful);
      benchmark::DoNotOptimize(coupling.clip().data());
      benchmark::DoNotOptimize(coupling.clip().size());
      benchmark::DoNotOptimize(coupling.cells().data());
      benchmark::DoNotOptimize(coupling.cells().size());
      benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations());
  }

  /**
   * Measures the use case in which one source element couples to four target elements.
   *
   * The target quadrilaterals form a non-cocircular, nonmatching partition of the source square.
   */
  static void mortar_coupling3d_four_pair_partition(
      benchmark::State& state, const Mortar::Triangulation triangulation)
  {
    const Polygon source_coordinates = {
        {0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {2.0, 2.0, 0.0}, {0.0, 2.0, 0.0}};
    const Point center{0.8, 1.1, 0.0};
    const std::vector<Polygon> target_coordinates{
        {{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, center, {1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}, {0.0, 2.0, 0.0}, {1.0, 2.0, 0.0}, center},
        {center, {1.0, 2.0, 0.0}, {2.0, 2.0, 0.0}, {2.0, 1.0, 0.0}},
        {{1.0, 0.0, 0.0}, center, {2.0, 1.0, 0.0}, {2.0, 0.0, 0.0}},
    };

    Core::FE::Discretization discretization{"mortar_coupling3d_benchmark", MPI_COMM_SELF, 3};
    OwnedMortarElement source(0, 0, Core::FE::CellType::quad4, source_coordinates, true);

    std::vector<std::unique_ptr<OwnedMortarElement>> targets;
    std::vector<Teuchos::ParameterList> parameter_lists;
    std::vector<std::unique_ptr<Mortar::Coupling3d>> couplings;
    targets.reserve(target_coordinates.size());
    parameter_lists.reserve(target_coordinates.size());
    couplings.reserve(target_coordinates.size());

    for (std::size_t target = 0; target < target_coordinates.size(); ++target)
    {
      targets.emplace_back(std::make_unique<OwnedMortarElement>(static_cast<int>(target + 1),
          static_cast<int>(100 + 10 * target), Core::FE::CellType::quad4,
          target_coordinates[target], false));
      parameter_lists.emplace_back(coupling_parameters(triangulation));
      couplings.emplace_back(std::make_unique<Mortar::Coupling3d>(discretization, 3, false,
          parameter_lists.back(), *source.element, *targets.back()->element));
    }

    for (const auto& coupling : couplings)
    {
      if (!coupling->evaluate_coupling())
      {
        state.SkipWithError("A mortar partition pair was unexpectedly rejected.");
        return;
      }
    }

    for (auto _ : state)
    {
      std::size_t number_of_clip_vertices = 0;
      std::size_t number_of_cells = 0;

      for (const auto& coupling : couplings)
      {
        bool successful = coupling->evaluate_coupling();
        benchmark::DoNotOptimize(successful);
        number_of_clip_vertices += coupling->clip().size();
        number_of_cells += coupling->cells().size();
      }

      benchmark::DoNotOptimize(number_of_clip_vertices);
      benchmark::DoNotOptimize(number_of_cells);
      benchmark::ClobberMemory();
    }

    state.counters["pairs"] = static_cast<double>(couplings.size());
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(couplings.size()));
  }


  // Mortar auxiliary plane construction benchmark
  BENCHMARK_CAPTURE(mortar_coupling3d_auxiliary_plane, planar_triangle,
      GeometryCase::triangle_six_vertex_overlap);
  BENCHMARK_CAPTURE(
      mortar_coupling3d_auxiliary_plane, warped_quadrilateral, GeometryCase::warped_quadrilaterals);

  // Mortar projection benchmark
  BENCHMARK_CAPTURE(mortar_coupling3d_project_vertices, planar_quadrilaterals,
      GeometryCase::quadrilateral_eight_vertex_overlap);
  BENCHMARK_CAPTURE(mortar_coupling3d_project_vertices, warped_quadrilaterals,
      GeometryCase::warped_quadrilaterals);

  // Mortar (repeated) evaluation benchmark
  BENCHMARK_CAPTURE(mortar_coupling3d_evaluate_reused, tri3_tri3_clip6_center,
      GeometryCase::triangle_six_vertex_overlap, Mortar::triangulation_center);
  BENCHMARK_CAPTURE(mortar_coupling3d_evaluate_reused, tri3_tri3_clip6_delaunay,
      GeometryCase::triangle_six_vertex_overlap, Mortar::triangulation_delaunay);
  BENCHMARK_CAPTURE(mortar_coupling3d_evaluate_reused, quad4_quad4_clip4_center,
      GeometryCase::quadrilateral_four_vertex_overlap, Mortar::triangulation_center);
  BENCHMARK_CAPTURE(mortar_coupling3d_evaluate_reused, quad4_quad4_clip4_delaunay,
      GeometryCase::quadrilateral_four_vertex_overlap, Mortar::triangulation_delaunay);
  BENCHMARK_CAPTURE(mortar_coupling3d_evaluate_reused, quad4_quad4_clip8_center,
      GeometryCase::quadrilateral_eight_vertex_overlap, Mortar::triangulation_center);
  BENCHMARK_CAPTURE(mortar_coupling3d_evaluate_reused, quad4_quad4_clip8_delaunay,
      GeometryCase::quadrilateral_eight_vertex_overlap, Mortar::triangulation_delaunay);
  BENCHMARK_CAPTURE(mortar_coupling3d_evaluate_reused, disjoint_early_rejection,
      GeometryCase::disjoint_triangles, Mortar::triangulation_center);

  // Mortar coupling construction and evaluation benchmark
  BENCHMARK_CAPTURE(mortar_coupling3d_construct_and_evaluate, quad4_quad4_clip8_center,
      GeometryCase::quadrilateral_eight_vertex_overlap, Mortar::triangulation_center);
  BENCHMARK_CAPTURE(mortar_coupling3d_construct_and_evaluate, quad4_quad4_clip8_delaunay,
      GeometryCase::quadrilateral_eight_vertex_overlap, Mortar::triangulation_delaunay);

  BENCHMARK_CAPTURE(mortar_coupling3d_four_pair_partition, center, Mortar::triangulation_center);
  BENCHMARK_CAPTURE(
      mortar_coupling3d_four_pair_partition, delaunay, Mortar::triangulation_delaunay);

}  // namespace

FOUR_C_NAMESPACE_CLOSE
