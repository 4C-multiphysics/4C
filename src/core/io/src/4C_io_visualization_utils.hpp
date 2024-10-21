#ifndef FOUR_C_IO_VISUALIZATION_UTILS_HPP
#define FOUR_C_IO_VISUALIZATION_UTILS_HPP

#include "4C_config.hpp"

#include "4C_io_visualization_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  /**
   * @brief Add a polyhedron cell to a visualization data object
   *
   * @tparam n_dim Spatial dimension of the problem
   * @param visualization_data (in/out) Visualization data that the polyhedron is added to
   * @param point_coordinates (in) Points of the polyhedron
   * @param face_connectivity (in) Connectivity for each face of the polyhedron
   */
  template <unsigned int n_dim>
  void append_polyhedron_to_visualization_data(VisualizationData& visualization_data,
      const std::vector<Core::LinAlg::Matrix<n_dim, 1>>& point_coordinates,
      const std::vector<std::vector<int>>& face_connectivity);
}  // namespace Core::IO


FOUR_C_NAMESPACE_CLOSE

#endif
