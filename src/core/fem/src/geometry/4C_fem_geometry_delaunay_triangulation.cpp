// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_geometry_delaunay_triangulation.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <string>

#ifdef FOUR_C_WITH_QHULL
extern "C"
{
#include <libqhull_r/qhull_ra.h>
}
#endif

FOUR_C_NAMESPACE_OPEN

namespace Core::Geometry
{
  namespace
  {
#ifdef FOUR_C_WITH_QHULL
    /**
     * @brief RAII wrapper for a reentrant Qhull context.
     *
     * Initializes the Qhull state on construction and releases all memory owned
     * by Qhull on destruction. The wrapper therefore also cleans up correctly
     * when an exception is thrown during triangulation or result extraction.
     */
    class QhullContext
    {
     public:
      explicit QhullContext(FILE* error_file) { qh_zero(&context_, error_file); }

      ~QhullContext()
      {
        if (!active_) return;

        qh_freeqhull(&context_, !qh_ALL);

        int current_long_memory = 0;
        int total_long_memory = 0;
        qh_memfreeshort(&context_, &current_long_memory, &total_long_memory);

        if (current_long_memory != 0 || total_long_memory != 0)
        {
          std::fprintf(stderr, "Qhull did not free %d bytes of long memory (%d pieces).\n",
              total_long_memory, current_long_memory);
        }
      }

      QhullContext(const QhullContext&) = delete;
      QhullContext& operator=(const QhullContext&) = delete;

      qhT* get() { return &context_; }

     private:
      qhT context_{};
      bool active_ = true;
    };
#endif

    template <std::size_t dim>
    std::vector<Simplex<dim>> delaunay_triangulation_impl(
        const std::span<const Point<dim>> input_points)
    {
      static_assert(dim == 2 || dim == 3);

      if (input_points.size() < dim + 1)
      {
        FOUR_C_THROW(
            "A {}D Delaunay triangulation requires at least {} points, "
            "but only {} were provided.",
            dim, dim + 1, input_points.size());
      }

      /*
       * Use normalized coordinates only for geometric predicates. Qhull
       * receives the original coordinates to preserve the behavior of the
       * former Cut implementation for numerically degenerate configurations.
       */
      const auto normalized_points = normalize_points<dim>(input_points);

      // The minimum full-dimensional point set forms one simplex.
      if (input_points.size() == dim + 1)
      {
        Simplex<dim> simplex{};
        std::iota(simplex.begin(), simplex.end(), 0);

        if (!canonicalize_simplex<dim>(simplex, normalized_points))
          FOUR_C_THROW("The provided {}D point set is not full-dimensional.", dim);

        return {simplex};
      }

#ifndef FOUR_C_WITH_QHULL
      FOUR_C_THROW("Delaunay triangulation requires a 4C build with Qhull.");
#else
      const int number_of_points = static_cast<int>(input_points.size());
      std::vector<coordT> coordinates(dim * input_points.size());

      for (std::size_t point_index = 0; point_index < input_points.size(); ++point_index)
        for (std::size_t d = 0; d < dim; ++d)
          coordinates[dim * point_index + d] = input_points[point_index][d];

      /*
       * First preserve the usual Cut topology. If this does not produce a
       * complete triangulation, use Qz to handle cocircular or cospherical
       * input. The coordinates are not joggled.
       */
      static constexpr std::array<std::string_view, 2> qhull_commands{
          "qhull d Qt QbB Qc Pp",
          "qhull d Qt QbB Qc Qz Pp",
      };

      for (const std::string_view command : qhull_commands)
      {
        QhullContext context(stderr);
        qhT* qh = context.get();
        std::string options(command);
        constexpr boolT qhull_owns_coordinates = false;

        const int exit_code = qh_new_qhull(qh, static_cast<int>(dim), number_of_points,
            coordinates.data(), qhull_owns_coordinates, options.data(), nullptr, stderr);

        if (exit_code != 0) continue;

        std::vector<Simplex<dim>> attempt_simplices;
        bool valid_attempt = true;
        facetT* facet = nullptr;

        FORALLfacets
        {
          // Delaunay cells correspond to lower facets of the lifted hull.
          if (facet->upperdelaunay || facet->vertices == nullptr) continue;

          if (qh_setsize(qh, facet->vertices) != static_cast<int>(dim + 1))
          {
            valid_attempt = false;
            break;
          }

          Simplex<dim> simplex{};
          std::size_t local_vertex = 0;
          vertexT* vertex = nullptr;

          for (void** vertex_pointer = &facet->vertices->e[0].p;
              (vertex = static_cast<vertexT*>(*vertex_pointer++));)
          {
            const int point_id = qh_pointid(qh, vertex->point);
            /*
             * Qz adds a point at infinity. It should occur only in upper
             * Delaunay facets, which were discarded above.
             */
            if (point_id < 0 || point_id >= number_of_points)
            {
              valid_attempt = false;
              break;
            }

            simplex[local_vertex++] = point_id;
          }

          if (!valid_attempt) break;

          /*
           * Canonicalize full-dimensional simplices. Degenerate simplices
           * remain sorted but have no orientation.
           *
           * They are intentionally retained because Cut may require their
           * connectivity. Cut::TetMesh::fix_broken_tets() later excludes
           * them from volume integration.
           */
          canonicalize_simplex<dim>(simplex, normalized_points);
          attempt_simplices.push_back(simplex);
        }

        if (!valid_attempt || attempt_simplices.empty()) continue;

        /*
         * Detect duplicates using a copy. Preserve Qhull's original facet
         * order in the returned triangulation because legacy Cut processing
         * can be sensitive to tetrahedron ordering in degenerate cases.
         */
        auto sorted_simplices = attempt_simplices;
        std::ranges::sort(sorted_simplices);

        if (std::ranges::adjacent_find(sorted_simplices) != sorted_simplices.end()) continue;

        std::vector<bool> point_is_used(input_points.size(), false);

        for (const auto& simplex : attempt_simplices)
          for (const int point_index : simplex) point_is_used[point_index] = true;

        if (!std::ranges::all_of(point_is_used, [](const bool used) { return used; })) continue;

        return attempt_simplices;
      }

      FOUR_C_THROW(
          "Qhull failed to compute a complete {}D Delaunay triangulation "
          "using the standard and point-at-infinity configurations.",
          dim);
#endif
    }
  }  // namespace

  std::vector<Simplex<2>> delaunay_triangulation(const std::span<const Point<2>> points)
  {
    return delaunay_triangulation_impl<2>(points);
  }

  std::vector<Simplex<3>> delaunay_triangulation(const std::span<const Point<3>> points)
  {
    return delaunay_triangulation_impl<3>(points);
  }

}  // namespace Core::Geometry

FOUR_C_NAMESPACE_CLOSE