// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_geometry_triangulation.hpp"

namespace
{
  using namespace FourC;

  using Pairedvector = Core::Gen::Pairedvector<int, double>;

  std::vector<Pairedvector> make_linearization(const int first_dof, const double factor)
  {
    std::vector<Pairedvector> linearization;
    linearization.reserve(3);

    for (int dim = 0; dim < 3; ++dim)
    {
      linearization.emplace_back(1);
      linearization.back()[first_dof + dim] = factor * static_cast<double>(dim + 1);
    }

    return linearization;
  }

  void expect_pairedvector_equal(const Pairedvector& actual, const Pairedvector& expected)
  {
    ASSERT_EQ(actual.size(), expected.size());

    auto actual_entry = actual.begin();
    auto expected_entry = expected.begin();

    for (; expected_entry != expected.end(); ++actual_entry, ++expected_entry)
    {
      EXPECT_EQ(actual_entry->first, expected_entry->first);
      EXPECT_DOUBLE_EQ(actual_entry->second, expected_entry->second);
    }
  }

  void expect_linearization_equal(
      const std::vector<Pairedvector>& actual, const std::vector<Pairedvector>& expected)
  {
    ASSERT_EQ(actual.size(), expected.size());

    for (std::size_t dim = 0; dim < actual.size(); ++dim)
      expect_pairedvector_equal(actual[dim], expected[dim]);
  }

  void expect_triangle_vertex(const Core::Geometry::Triangle& triangle,
      const std::size_t local_vertex, const std::array<double, 3>& expected_coordinates,
      const std::vector<Pairedvector>& expected_linearization)
  {
    for (std::size_t dim = 0; dim < 3; ++dim)
      EXPECT_DOUBLE_EQ(triangle.coordinates(dim, local_vertex), expected_coordinates[dim]);

    expect_linearization_equal(
        triangle.vertex_linearizations[local_vertex], expected_linearization);
  }

  // Test that a triangular polygon is returned unchanged as one triangle.
  TEST(LocalCenterTriangulation, TriangleIsReturnedUnchanged)
  {
    const std::vector<std::array<double, 3>> polygon = {
        {0.0, 0.0, 2.0},
        {3.0, 0.0, 2.0},
        {0.0, 3.0, 2.0},
    };

    const std::vector<std::vector<Pairedvector>> vertex_linearizations = {
        make_linearization(0, 1.0),
        make_linearization(3, 2.0),
        make_linearization(6, 3.0),
    };

    const auto triangles =
        Core::Geometry::local_center_triangulation(polygon, vertex_linearizations, {}, 0.0);

    ASSERT_TRUE(triangles.has_value());
    ASSERT_EQ(triangles->size(), 1);

    expect_triangle_vertex((*triangles)[0], 0, polygon[0], vertex_linearizations[0]);
    expect_triangle_vertex((*triangles)[0], 1, polygon[1], vertex_linearizations[1]);
    expect_triangle_vertex((*triangles)[0], 2, polygon[2], vertex_linearizations[2]);
  }

  // Test that a quadrilateral produces one triangle per polygon edge.
  TEST(LocalCenterTriangulation, QuadrilateralProducesTriangleFan)
  {
    const std::vector<std::array<double, 3>> polygon = {
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {2.0, 2.0, 0.0},
        {0.0, 2.0, 0.0},
    };

    const std::vector<std::vector<Pairedvector>> vertex_linearizations = {
        make_linearization(0, 1.0),
        make_linearization(3, 2.0),
        make_linearization(6, 3.0),
        make_linearization(9, 4.0),
    };

    const auto center_linearization = make_linearization(100, 10.0);
    const auto triangles = Core::Geometry::local_center_triangulation(
        polygon, vertex_linearizations, center_linearization, 0.0);

    ASSERT_TRUE(triangles.has_value());
    ASSERT_EQ(triangles->size(), polygon.size());

    const std::array<double, 3> expected_center{1.0, 1.0, 0.0};

    for (std::size_t triangle = 0; triangle < triangles->size(); ++triangle)
    {
      const std::size_t next_vertex = (triangle + 1) % polygon.size();

      expect_triangle_vertex((*triangles)[triangle], 0, expected_center, center_linearization);
      expect_triangle_vertex(
          (*triangles)[triangle], 1, polygon[triangle], vertex_linearizations[triangle]);
      expect_triangle_vertex(
          (*triangles)[triangle], 2, polygon[next_vertex], vertex_linearizations[next_vertex]);
    }
  }

  // Test that a polygon whose area does not exceed the tolerance is rejected.
  TEST(LocalCenterTriangulation, AreaToleranceRejectsPolygon)
  {
    const std::vector<std::array<double, 3>> polygon = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
    };

    const std::vector<std::vector<Pairedvector>> vertex_linearizations = {
        make_linearization(0, 1.0),
        make_linearization(3, 2.0),
        make_linearization(6, 3.0),
        make_linearization(9, 4.0),
    };

    const auto center_linearization = make_linearization(100, 10.0);

    EXPECT_FALSE(Core::Geometry::local_center_triangulation(
        polygon, vertex_linearizations, center_linearization, 1.0));

    EXPECT_TRUE(Core::Geometry::local_center_triangulation(
        polygon, vertex_linearizations, center_linearization, 0.5));
  }

  // Test that a polygon degenerating to a line cannot be triangulated.
  TEST(LocalCenterTriangulation, DegeneratePolygonIsRejected)
  {
    const std::vector<std::array<double, 3>> polygon = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {3.0, 0.0, 0.0},
    };

    const std::vector<std::vector<Pairedvector>> vertex_linearizations = {
        make_linearization(0, 1.0),
        make_linearization(3, 2.0),
        make_linearization(6, 3.0),
        make_linearization(9, 4.0),
    };

    const auto center_linearization = make_linearization(100, 10.0);

    EXPECT_FALSE(Core::Geometry::local_center_triangulation(
        polygon, vertex_linearizations, center_linearization, 0.0));
  }

  // Test that an irregular convex hexagon is decomposed into six triangles connecting
  // each polygon edge to the area-weighted center.
  TEST(LocalCenterTriangulation, IrregularHexagonProducesSixTriangles)
  {
    const std::vector<std::array<double, 3>> polygon = {
        {0.0, 0.0, 1.0},
        {3.0, 0.0, 1.0},
        {5.0, 1.0, 1.0},
        {4.0, 4.0, 1.0},
        {1.0, 5.0, 1.0},
        {-1.0, 2.0, 1.0},
    };

    std::vector<std::vector<Pairedvector>> vertex_linearizations;
    vertex_linearizations.reserve(polygon.size());

    for (std::size_t vertex = 0; vertex < polygon.size(); ++vertex)
    {
      vertex_linearizations.emplace_back(
          make_linearization(static_cast<int>(3 * vertex), static_cast<double>(vertex + 1)));
    }

    const auto center_linearization = make_linearization(100, 10.0);

    const auto triangles = Core::Geometry::local_center_triangulation(
        polygon, vertex_linearizations, center_linearization, 0.0);

    ASSERT_TRUE(triangles.has_value());
    ASSERT_EQ(triangles->size(), 6);

    const std::array<double, 3> expected_center = {124.0 / 63.0, 46.0 / 21.0, 1.0};

    for (std::size_t triangle = 0; triangle < triangles->size(); ++triangle)
    {
      const std::size_t next_vertex = (triangle + 1) % polygon.size();

      expect_triangle_vertex((*triangles)[triangle], 0, expected_center, center_linearization);
      expect_triangle_vertex(
          (*triangles)[triangle], 1, polygon[triangle], vertex_linearizations[triangle]);
      expect_triangle_vertex(
          (*triangles)[triangle], 2, polygon[next_vertex], vertex_linearizations[next_vertex]);
    }

    // The generated triangles must cover the complete polygon.
    const auto triangle_area = [](const Core::Geometry::Triangle& triangle)
    {
      std::array<double, 3> first_edge{};
      std::array<double, 3> second_edge{};

      for (std::size_t dim = 0; dim < 3; ++dim)
      {
        first_edge[dim] = triangle.coordinates(dim, 1) - triangle.coordinates(dim, 0);
        second_edge[dim] = triangle.coordinates(dim, 2) - triangle.coordinates(dim, 0);
      }

      const std::array<double, 3> cross_product = {
          first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
          first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
          first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
      };

      return 0.5 *
             std::sqrt(cross_product[0] * cross_product[0] + cross_product[1] * cross_product[1] +
                       cross_product[2] * cross_product[2]);
    };

    double triangulated_area = 0.0;
    for (const auto& triangle : *triangles) triangulated_area += triangle_area(triangle);

    EXPECT_NEAR(triangulated_area, 21.0, 1.0e-12);
  }
}  // namespace
