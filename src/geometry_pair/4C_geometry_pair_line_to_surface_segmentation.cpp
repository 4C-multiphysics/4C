// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_geometry_pair_line_to_surface_segmentation.hpp"

#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_projection.hpp"
#include "4C_geometry_pair_line_to_surface_evaluation_data.hpp"
#include "4C_geometry_pair_scalar_types.hpp"

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
template <typename ScalarType, typename Line, typename Surface>
GeometryPair::GeometryPairLineToSurfaceSegmentation<ScalarType, Line,
    Surface>::GeometryPairLineToSurfaceSegmentation(const Core::Elements::Element* element1,
    const Core::Elements::Element* element2,
    const std::shared_ptr<GeometryPair::LineToSurfaceEvaluationData>&
        line_to_surface_evaluation_data)
    : GeometryPairLineToSurface<ScalarType, Line, Surface>(
          element1, element2, line_to_surface_evaluation_data)
{
  // Check if a segment tracker exists for this line element. If not a new one is created.
  int line_element_id = this->element1()->id();
  std::map<int, std::set<LineSegment<double>>>& segment_tracker_map =
      this->line_to_surface_evaluation_data_->get_segment_tracker();

  if (segment_tracker_map.find(line_element_id) == segment_tracker_map.end())
  {
    std::set<LineSegment<double>> new_tracking_set;
    new_tracking_set.clear();
    segment_tracker_map[line_element_id] = new_tracking_set;
  }
}

/**
 *
 */
template <typename ScalarType, typename Line, typename Surface>
void GeometryPair::GeometryPairLineToSurfaceSegmentation<ScalarType, Line, Surface>::evaluate(
    const ElementData<Line, ScalarType>& element_data_line,
    const ElementData<Surface, ScalarType>& element_data_surface,
    std::vector<LineSegment<ScalarType>>& segments) const
{
  // Call the pre_evaluate method of the general Gauss point projection class.
  LineTo3DSegmentation<GeometryPairLineToSurfaceSegmentation<ScalarType, Line, Surface>>::evaluate(
      this, element_data_line, element_data_surface, segments);
}


/**
 * Explicit template initialization of template class.
 */
namespace GeometryPair
{
  template class GeometryPairLineToSurfaceSegmentation<double, t_line2, t_tri3>;
  template class GeometryPairLineToSurfaceSegmentation<double, t_line2, t_tri6>;
  template class GeometryPairLineToSurfaceSegmentation<double, t_line2, t_quad4>;
  template class GeometryPairLineToSurfaceSegmentation<double, t_line2, t_quad8>;
  template class GeometryPairLineToSurfaceSegmentation<double, t_line2, t_quad9>;
  template class GeometryPairLineToSurfaceSegmentation<double, t_line2, t_nurbs9>;

  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type_1st_order,
      t_line2, t_tri3>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type_1st_order,
      t_line2, t_tri6>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type_1st_order,
      t_line2, t_quad4>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type_1st_order,
      t_line2, t_quad8>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type_1st_order,
      t_line2, t_quad9>;
  template class GeometryPairLineToSurfaceSegmentation<
      line_to_surface_patch_scalar_type_fixed_size_1st_order<t_line2, t_nurbs9>, t_line2, t_nurbs9>;

  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type, t_line2,
      t_tri3>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type, t_line2,
      t_tri6>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type, t_line2,
      t_quad4>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type, t_line2,
      t_quad8>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type, t_line2,
      t_quad9>;
  template class GeometryPairLineToSurfaceSegmentation<
      line_to_surface_patch_scalar_type_fixed_size<t_line2, t_nurbs9>, t_line2, t_nurbs9>;

  template class GeometryPairLineToSurfaceSegmentation<double, t_hermite, t_tri3>;
  template class GeometryPairLineToSurfaceSegmentation<double, t_hermite, t_tri6>;
  template class GeometryPairLineToSurfaceSegmentation<double, t_hermite, t_quad4>;
  template class GeometryPairLineToSurfaceSegmentation<double, t_hermite, t_quad8>;
  template class GeometryPairLineToSurfaceSegmentation<double, t_hermite, t_quad9>;
  template class GeometryPairLineToSurfaceSegmentation<double, t_hermite, t_nurbs9>;

  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type_1st_order,
      t_hermite, t_tri3>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type_1st_order,
      t_hermite, t_tri6>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type_1st_order,
      t_hermite, t_quad4>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type_1st_order,
      t_hermite, t_quad8>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type_1st_order,
      t_hermite, t_quad9>;
  template class GeometryPairLineToSurfaceSegmentation<
      line_to_surface_patch_scalar_type_fixed_size_1st_order<t_hermite, t_nurbs9>, t_hermite,
      t_nurbs9>;

  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type, t_hermite,
      t_tri3>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type, t_hermite,
      t_tri6>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type, t_hermite,
      t_quad4>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type, t_hermite,
      t_quad8>;
  template class GeometryPairLineToSurfaceSegmentation<line_to_surface_patch_scalar_type, t_hermite,
      t_quad9>;
  template class GeometryPairLineToSurfaceSegmentation<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_nurbs9>, t_hermite, t_nurbs9>;
}  // namespace GeometryPair

FOUR_C_NAMESPACE_CLOSE
