// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_geometry_pair_line_to_volume_gauss_point_projection_cross_section.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beam3_triad_interpolation_local_rotation_vectors.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_to_3D_evaluation_data.hpp"
#include "4C_geometry_pair_utility_classes.hpp"

#include <math.h>

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
template <typename ScalarType, typename Line, typename Volume>
GEOMETRYPAIR::GeometryPairLineToVolumeGaussPointProjectionCrossSection<ScalarType, Line,
    Volume>::GeometryPairLineToVolumeGaussPointProjectionCrossSection(const Core::Elements::Element*
                                                                          element1,
    const Core::Elements::Element* element2,
    const std::shared_ptr<GEOMETRYPAIR::LineTo3DEvaluationData>& evaluation_data)
    : GeometryPairLineToVolume<ScalarType, Line, Volume>(element1, element2, evaluation_data)
{
  // Check if a projection tracking vector exists for this line element. If not a new one is
  // created.
  int line_element_id = this->element1()->id();
  std::map<int, std::vector<bool>>& projection_tracker =
      this->line_to_3d_evaluation_data_->get_gauss_point_projection_tracker();

  if (projection_tracker.find(line_element_id) == projection_tracker.end())
  {
    int n_gauss_points =
        this->line_to_3d_evaluation_data_->get_number_of_gauss_points() *
        this->line_to_3d_evaluation_data_->get_number_of_integration_points_circumference();
    std::vector<bool> new_tracking_vector;
    new_tracking_vector.resize(n_gauss_points, false);
    projection_tracker[line_element_id] = new_tracking_vector;
  }
}


/**
 *
 */
template <typename ScalarType, typename Line, typename Volume>
void GEOMETRYPAIR::GeometryPairLineToVolumeGaussPointProjectionCrossSection<ScalarType, Line,
    Volume>::pre_evaluate(const ElementData<Line, ScalarType>& element_data_line,
    const ElementData<Volume, ScalarType>& element_data_volume,
    std::vector<LineSegment<ScalarType>>& segments,
    const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>*
        line_triad_interpolation) const
{
  // Get the Gauss point projection tracker for this line element.
  std::vector<bool>& line_projection_tracker = get_line_projection_vector();

  // Gauss rule.
  Core::FE::IntegrationPoints1D gauss_points_axis =
      this->line_to_3d_evaluation_data_->get_gauss_points();
  unsigned int n_gauss_points_axis =
      this->line_to_3d_evaluation_data_->get_number_of_gauss_points();
  unsigned int n_integration_points_circ =
      this->line_to_3d_evaluation_data_->get_number_of_integration_points_circumference();

  // Initialize variables for the projection.
  ScalarType eta;
  double alpha;
  Core::LinAlg::Matrix<3, 1, ScalarType> r_line;
  Core::LinAlg::Matrix<3, 3, ScalarType> triad;
  Core::LinAlg::Matrix<3, 1, ScalarType> r_cross_section;
  Core::LinAlg::Matrix<3, 1, ScalarType> r_surface;
  Core::LinAlg::Matrix<3, 1, ScalarType> eta_cross_section(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<2, 1, ScalarType> eta_cross_section_2d;
  Core::LinAlg::Matrix<3, 1, ScalarType> xi_volume;
  ProjectionResult projection_result;
  segments.clear();
  bool one_projects = false;
  LineSegment<ScalarType> projection_point_segment;

  // Get the radius from the beam element.
  const double radius = (dynamic_cast<const Discret::Elements::Beam3Base*>(this->element1()))
                            ->get_circular_cross_section_radius_for_interactions();

  // Loop over Gauss points and check if they project to this volume.
  for (unsigned int index_gp_axis = 0; index_gp_axis < n_gauss_points_axis; index_gp_axis++)
  {
    // Parameter coordinate along the line.
    eta = gauss_points_axis.qxg[index_gp_axis][0];

    // Get the triad on the line.
    if (line_triad_interpolation != nullptr)
      line_triad_interpolation->get_interpolated_triad_at_xi(triad, eta);
    else
      GEOMETRYPAIR::evaluate_triad_at_plane_curve<Line>(eta, element_data_line, triad);

    // Get the position on the line.
    GEOMETRYPAIR::evaluate_position<Line>(eta, element_data_line, r_line);

    for (unsigned int index_gp_circ = 0; index_gp_circ < n_integration_points_circ; index_gp_circ++)
    {
      // Index of the current Gauss point in the tracking vector.
      unsigned int index_gp = index_gp_axis * n_integration_points_circ + index_gp_circ;

      // Only check points that do not already have a valid projection.
      if (line_projection_tracker[index_gp] == false)
      {
        // Coordinates in the cross section.
        alpha = 2.0 * M_PI / double(n_integration_points_circ) * index_gp_circ;
        eta_cross_section(0) = 0;
        eta_cross_section(1) = cos(alpha) * radius;
        eta_cross_section(2) = sin(alpha) * radius;

        // Get the point on the beams surface.
        r_cross_section.multiply(triad, eta_cross_section);
        r_surface = r_line;
        r_surface += r_cross_section;

        // Project point to the volume.
        this->project_point_to_other(r_surface, element_data_volume, xi_volume, projection_result);
        if (projection_result == ProjectionResult::projection_found_valid)
        {
          // Valid Gauss point was found, add to this segment and set tracking point to true.
          ProjectionPoint1DTo3D<ScalarType> new_point(eta, xi_volume,
              gauss_points_axis.qwgt[index_gp_axis] * 2.0 / double(n_integration_points_circ));
          for (unsigned int i_dim = 0; i_dim < 2; i_dim++)
            eta_cross_section_2d(i_dim) = eta_cross_section(i_dim + 1);
          new_point.set_eta_cross_section(eta_cross_section_2d);
          projection_point_segment.add_projection_point(new_point);
          line_projection_tracker[index_gp] = true;

          one_projects = true;
        }
      }
    }
  }

  if (one_projects)
  {
    // Clear the segment vector and add the found segment for the current line to volume pair.
    segments.clear();
    segments.push_back(projection_point_segment);
  }
}


/**
 *
 */
template <typename ScalarType, typename Line, typename Volume>
void GEOMETRYPAIR::GeometryPairLineToVolumeGaussPointProjectionCrossSection<ScalarType, Line,
    Volume>::evaluate(const ElementData<Line, ScalarType>& element_data_line,
    const ElementData<Volume, ScalarType>& element_data_volume,
    std::vector<LineSegment<ScalarType>>& segments) const
{
  // Only zero one segments are expected.
  if (segments.size() > 1)
    FOUR_C_THROW(
        "There should be zero or one segments for the Gauss point cylinder projection method. The "
        "actual value is {}!",
        segments.size());

  // Check if one point projected in pre_evaluate.
  if (segments.size() == 1 && segments[0].get_number_of_projection_points() > 0)
  {
    // Check if all points of this line projected.
    const std::vector<bool>& projection_vector = get_line_projection_vector();
    bool all_projected =
        std::all_of(projection_vector.begin(), projection_vector.end(), [](bool v) { return v; });
    if (!all_projected)
    {
      unsigned int valid_projection_points = 0;
      for (auto const& value : projection_vector)
        if (value) valid_projection_points += 1;
      FOUR_C_THROW(
          "The cross section projection currently only works if all points on a line project! Of "
          "the {} points, only {} projected.",
          projection_vector.size(), valid_projection_points);
    }
  }
}


/**
 *
 */
template <typename ScalarType, typename Line, typename Volume>
std::vector<bool>& GEOMETRYPAIR::GeometryPairLineToVolumeGaussPointProjectionCrossSection<
    ScalarType, Line, Volume>::get_line_projection_vector() const
{
  // Get the Gauss point projection tracker for this line element.
  int line_element_id = this->element1()->id();
  std::map<int, std::vector<bool>>& projection_tracker =
      this->line_to_3d_evaluation_data_->get_gauss_point_projection_tracker();
  return projection_tracker[line_element_id];
}


/**
 * Explicit template initialization of template class.
 */
template class GEOMETRYPAIR::GeometryPairLineToVolumeGaussPointProjectionCrossSection<double,
    GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex8>;
template class GEOMETRYPAIR::GeometryPairLineToVolumeGaussPointProjectionCrossSection<double,
    GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex20>;
template class GEOMETRYPAIR::GeometryPairLineToVolumeGaussPointProjectionCrossSection<double,
    GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex27>;
template class GEOMETRYPAIR::GeometryPairLineToVolumeGaussPointProjectionCrossSection<double,
    GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_tet4>;
template class GEOMETRYPAIR::GeometryPairLineToVolumeGaussPointProjectionCrossSection<double,
    GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_tet10>;

FOUR_C_NAMESPACE_CLOSE
