// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_geometric_search_utils.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_geometric_search_bounding_volume.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_utils_fad.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::GeometricSearch
{
  void print_geometric_search_details(MPI_Comm comm, const GeometricSearchInfo info)
  {
    const int numproc = Core::Communication::num_mpi_ranks(comm);
    const int myrank = Core::Communication::my_mpi_rank(comm);

    std::vector<int> primitive_size(numproc, 0), my_primitive_size(numproc, 0);
    std::vector<int> predicate_size(numproc, 0), my_predicate_size(numproc, 0);
    std::vector<int> coupling_pair_size(numproc, 0), my_coupling_pair_size(numproc, 0);

    my_primitive_size[myrank] = info.primitive_size;
    my_predicate_size[myrank] = info.predicate_size;
    my_coupling_pair_size[myrank] = info.coupling_pair_size;

    primitive_size = Core::Communication::sum_all(my_primitive_size, comm);
    predicate_size = Core::Communication::sum_all(my_predicate_size, comm);
    coupling_pair_size = Core::Communication::sum_all(my_coupling_pair_size, comm);

    if (myrank == 0)
    {
      Core::IO::cout(Core::IO::verbose) << "\n   Collision search:" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "   +-----+------------+------------+--------------+" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "   | PID | primitives | predicates |  found pairs |" << Core::IO::endl;
      Core::IO::cout(Core::IO::verbose)
          << "   +-----+------------+------------+--------------+" << Core::IO::endl;

      for (int npid = 0; npid < numproc; ++npid)
      {
        Core::IO::cout(Core::IO::verbose)
            << "   | " << std::setw(3) << npid << " | " << std::setw(10) << primitive_size[npid]
            << " | " << std::setw(10) << predicate_size[npid] << " | " << std::setw(12)
            << coupling_pair_size[npid] << " | " << Core::IO::endl;
        Core::IO::cout(Core::IO::verbose)
            << "   +-----+------------+------------+--------------+" << Core::IO::endl;
      }
      Core::IO::cout(Core::IO::verbose) << Core::IO::endl;
    }
  }

  std::pair<std::vector<Core::LinAlg::Matrix<3, 1>>, std::vector<std::vector<int>>>
  get_k_dop_polyhedron_representation(const BoundingVolume boundingVolume)
  {
#ifndef FOUR_C_WITH_ARBORX
    FOUR_C_THROW(
        "Core::GeometricSearch::GetKDopPolyhedronRepresentation can only be used with ArborX."
        "To use it, enable ArborX during the configure process.");
#else
    // This value is used for comparison of point coordinates. ArborX only uses float, so this value
    // has to be rather high.
    const double eps = 1e-5;

    // Later on we need to create an AABB to check if a point intersects a kdop. This function is
    // used to get the AABB dimensions, which are extended from the point dimensions such that we
    // account for absolute and relative tolerances in the comparison
    const auto get_atol_rtol_dimension = [&](const auto& value, const auto& factor)
    {
      const double atol = eps;
      const double rtol = eps;
      return value + abs(value) * factor * rtol + factor * atol;
    };

    // The k-dop visualization is based on polygons defined which are defined by intersections
    // of the k-dop planes. This pre-computed array (see the Mathematica scrip in the scrips/
    // sub-directory) contains data that describes all possible intersection points between all
    // k-dop planes.
    //
    // The first array index is the k-dop direction.
    // The second array index specifies if the min (0) or max (1) value along a direction is to be
    // considered. The element addressed by these two indices is a vector of pairs. Each element of
    // the vector is a possible intersection plane to the described by the first two indices. This
    // is stored in a tuple where the first index is the direction id and the second index is the
    // min/max flag.
    const std::array<std::array<std::vector<std::pair<int, bool>>, 2>, kdop_directions>
        possible_intersection_partners = {{{{{{11, false}, {7, false}, {12, false}, {6, false},
                                                 {10, false}, {4, false}, {9, false}, {3, false}},
                                               {{12, true}, {7, true}, {11, true}, {3, true},
                                                   {9, true}, {4, true}, {10, true}, {6, true}}}},
            {{{{9, false}, {5, false}, {12, true}, {6, true}, {10, true}, {8, false}, {11, false},
                  {3, false}},
                {{12, false}, {5, true}, {9, true}, {3, true}, {11, true}, {8, true}, {10, false},
                    {6, false}}}},
            {{{{10, false}, {8, true}, {11, true}, {7, true}, {12, true}, {5, false}, {9, false},
                  {4, false}},
                {{11, false}, {8, false}, {10, true}, {4, true}, {9, true}, {5, true}, {12, false},
                    {7, false}}}},
            {{{{10, false}, {4, false}, {9, false}, {5, false}, {12, true}, {1, false}, {10, true},
                  {8, false}, {11, false}, {7, false}, {12, false}, {0, false}},
                {{12, false}, {5, true}, {9, true}, {4, true}, {10, true}, {0, true}, {12, true},
                    {7, true}, {11, true}, {8, true}, {10, false}, {1, true}}}},
            {{{{12, false}, {6, false}, {10, false}, {8, true}, {11, true}, {2, false}, {12, true},
                  {5, false}, {9, false}, {3, false}, {11, false}, {0, false}},
                {{11, false}, {8, false}, {10, true}, {6, true}, {12, true}, {0, true}, {11, true},
                    {3, true}, {9, true}, {5, true}, {12, false}, {2, true}}}},
            {{{{4, false}, {10, false}, {2, false}, {11, true}, {7, true}, {12, true}, {6, true},
                  {10, true}, {1, false}, {11, false}, {3, false}, {9, false}},
                {{7, false}, {11, false}, {2, true}, {10, true}, {4, true}, {9, true}, {3, true},
                    {11, true}, {1, true}, {10, false}, {6, false}, {12, false}}}},
            {{{{11, false}, {7, false}, {12, false}, {5, true}, {9, true}, {1, true}, {11, true},
                  {8, true}, {10, false}, {4, false}, {9, false}, {0, false}},
                {{9, false}, {5, false}, {12, true}, {7, true}, {11, true}, {0, true}, {9, true},
                    {4, true}, {10, true}, {8, false}, {11, false}, {1, false}}}},
            {{{{9, false}, {3, false}, {11, false}, {8, false}, {10, true}, {2, true}, {9, true},
                  {5, true}, {12, false}, {6, false}, {10, false}, {0, false}},
                {{10, false}, {8, true}, {11, true}, {3, true}, {9, true}, {0, true}, {10, true},
                    {6, true}, {12, true}, {5, false}, {9, false}, {2, false}}}},
            {{{{3, false}, {9, false}, {1, false}, {12, true}, {6, true}, {10, true}, {4, true},
                  {9, true}, {2, true}, {12, false}, {7, false}, {11, false}},
                {{6, false}, {12, false}, {1, true}, {9, true}, {3, true}, {11, true}, {7, true},
                    {12, true}, {2, false}, {9, false}, {4, false}, {10, false}}}},
            {{{{6, false}, {4, false}, {8, true}, {2, false}, {7, true}, {5, false}, {6, true},
                  {1, false}, {8, false}, {3, false}, {7, false}, {0, false}},
                {{7, false}, {2, true}, {8, false}, {4, true}, {6, true}, {0, true}, {7, true},
                    {3, true}, {8, true}, {1, true}, {6, false}, {5, true}}}},
            {{{{7, false}, {6, false}, {5, true}, {1, true}, {3, true}, {8, true}, {7, true},
                  {2, false}, {5, false}, {4, false}, {3, false}, {0, false}},
                {{3, false}, {1, false}, {5, false}, {6, true}, {7, true}, {0, true}, {3, true},
                    {4, true}, {5, true}, {2, true}, {7, false}, {8, false}}}},
            {{{{4, false}, {3, false}, {5, false}, {1, false}, {6, true}, {8, false}, {4, true},
                  {2, true}, {5, true}, {7, false}, {6, false}, {0, false}},
                {{6, false}, {1, true}, {5, true}, {3, true}, {4, true}, {0, true}, {6, true},
                    {7, true}, {5, false}, {2, false}, {4, false}, {8, true}}}},
            {{{{3, false}, {7, false}, {8, false}, {2, true}, {4, true}, {5, true}, {3, true},
                  {1, true}, {8, true}, {6, false}, {4, false}, {0, false}},
                {{4, false}, {2, false}, {8, true}, {7, true}, {3, true}, {0, true}, {4, true},
                    {6, true}, {8, false}, {1, false}, {3, false}, {5, false}}}}}};


    // The actual ArborX k-dop
    const auto& kdop = boundingVolume.bounding_volume_;

    // Vector with all intersection points
    std::vector<LinAlg::Matrix<3, 1>> all_points;

    // Vector with all polygons connecting the intersection points
    std::vector<std::vector<int>> polygon_ids;

    for (unsigned int i_direction = 0; i_direction < kdop_directions; i_direction++)
    {
      for (unsigned int i_min_max = 0; i_min_max < 2; i_min_max++)
      {
        const auto& my_possible_partners = possible_intersection_partners[i_direction][i_min_max];

        // Return a value along a k-dop direction depending on the min max flag
        const auto get_kdop_value = [&kdop](const unsigned int index, const bool min_max)
        {
          if (min_max)
            return kdop._max_values[index];
          else
            return kdop._min_values[index];
        };

        // Utility function to get indices including offsets that "exceed" the vector dimension
        const auto get_partner_index = [&](const unsigned int index, const unsigned int offset)
        { return (index + offset) % my_possible_partners.size(); };

        // Get the intersection point between this plane and two other planes. A pair is returned
        // where the first entry is a flag specifying if the point is inside or outside the k-Dop,
        // the second entry is the intersection point.
        const auto get_intersection_point =
            [&](const unsigned int local_index_1, const unsigned int local_index_2)
        {
          const auto index_1 = my_possible_partners[local_index_1].first;
          const auto min_max_1 = my_possible_partners[local_index_1].second;
          const auto index_2 = my_possible_partners[local_index_2].first;
          const auto min_max_2 = my_possible_partners[local_index_2].second;

          LinAlg::Matrix<3, 3> coefficient_matrix;
          for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
          {
            coefficient_matrix(0, i_dir) =
                ArborX::Details::GetKDOPDirections<kdop_directions>::directions()[i_direction]
                    ._coords[i_dir];
            coefficient_matrix(1, i_dir) =
                ArborX::Details::GetKDOPDirections<kdop_directions>::directions()[index_1]
                    ._coords[i_dir];
            coefficient_matrix(2, i_dir) =
                ArborX::Details::GetKDOPDirections<kdop_directions>::directions()[index_2]
                    ._coords[i_dir];
          }
          LinAlg::Matrix<3, 1> right_hand_side;
          right_hand_side(0) = get_kdop_value(i_direction, i_min_max);
          right_hand_side(1) = get_kdop_value(index_1, min_max_1);
          right_hand_side(2) = get_kdop_value(index_2, min_max_2);

          // Check if there is a solution
          LinAlg::Matrix<3, 1> intersection_point;
          const auto found_solution =
              LinAlg::solve_linear_system_do_not_throw_error_on_zero_determinant_scaled(
                  coefficient_matrix, right_hand_side, intersection_point, 1e-12);
          if (!found_solution)
            return std::make_pair(false, intersection_point);
          else
          {
            // The offset here is needed, since arborx only works with float and we work with
            // double, which can cause issues
            const auto inside_kdop = intersects(
                ArborX::Box{
                    {static_cast<float>(get_atol_rtol_dimension(intersection_point(0), -1.0)),
                        static_cast<float>(get_atol_rtol_dimension(intersection_point(1), -1.0)),
                        static_cast<float>(get_atol_rtol_dimension(intersection_point(2), -1.0))},
                    {static_cast<float>(get_atol_rtol_dimension(intersection_point(0), 1.0)),
                        static_cast<float>(get_atol_rtol_dimension(intersection_point(1), 1.0)),
                        static_cast<float>(get_atol_rtol_dimension(intersection_point(2), 1.0))}},
                kdop);
            return std::make_pair(inside_kdop, intersection_point);
          }
        };

        // Loop over cutting directions until we get valid point
        bool found_starting_point = false;
        unsigned int i_start = 0;
        for (unsigned int i_possible_partner = 0; i_possible_partner < my_possible_partners.size();
            i_possible_partner++)
        {
          for (unsigned int offset = 1; offset < my_possible_partners.size(); offset++)
          {
            const auto intersection_result = get_intersection_point(
                i_possible_partner, get_partner_index(i_possible_partner, offset));
            if (intersection_result.first)
            {
              i_start = i_possible_partner;
              found_starting_point = true;
              break;
            }
          }
          if (found_starting_point) break;
        }
        if (!found_starting_point)
          FOUR_C_THROW("A starting point for the polygon could not be found.");

        // Starting from the found point loop over the edges of the polygon
        unsigned int offset = 0;
        std::vector<LinAlg::Matrix<3, 1>> polygon_points;
        while (offset < my_possible_partners.size())
        {
          const auto i_partner = get_partner_index(i_start, offset);
          for (unsigned int inner_offset = 1; inner_offset < my_possible_partners.size();
              inner_offset++)
          {
            const auto intersection_result =
                get_intersection_point(i_partner, get_partner_index(i_partner, inner_offset));
            if (intersection_result.first)
            {
              if (polygon_points.size() != 0)
              {
                // Check if the new found point is the same as the last found point
                auto diff = polygon_points.back();
                diff -= intersection_result.second;
                if (diff.norm2() > eps) polygon_points.push_back(intersection_result.second);
              }
              else
              {
                polygon_points.push_back(intersection_result.second);
              }
              offset = offset + inner_offset;
              break;
            }
          }
        }

        if (polygon_points.size() == 0) FOUR_C_THROW("No polygon points where found.");

        // Check if the start and end point are the same, if so remove the double point
        auto diff = polygon_points.back();
        diff -= polygon_points[0];
        if (diff.norm2() < eps)
        {
          polygon_points.pop_back();
        }

        // For more than 2 found points we have an actual polygon. Add the points to the final point
        // vector and get the indices for the polygon connectivity
        if (polygon_points.size() > 2)
        {
          std::vector<int> my_point_ids;
          for (const auto& point : polygon_points)
          {
            bool found = false;
            for (unsigned int i_all_points = 0; i_all_points < all_points.size(); i_all_points++)
            {
              auto diff = all_points[i_all_points];
              diff -= point;
              if (diff.norm2() < eps)
              {
                my_point_ids.push_back(i_all_points);
                found = true;
                break;
              }
            }
            if (!found)
            {
              my_point_ids.push_back(all_points.size());
              all_points.push_back(point);
            }
          }

          polygon_ids.push_back(my_point_ids);
        }
      }
    }

    std::vector<int> tmp;
    return {all_points, polygon_ids};
#endif
  }
}  // namespace Core::GeometricSearch

FOUR_C_NAMESPACE_CLOSE
