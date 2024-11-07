// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_geometry_pair_line_to_surface_gauss_point_projection.hpp"

#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_projection.hpp"
#include "4C_geometry_pair_line_to_surface_evaluation_data.hpp"
#include "4C_geometry_pair_scalar_types.hpp"

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
template <typename ScalarType, typename Line, typename Surface>
GEOMETRYPAIR::GeometryPairLineToSurfaceGaussPointProjection<ScalarType, Line,
    Surface>::GeometryPairLineToSurfaceGaussPointProjection(const Core::Elements::Element* element1,
    const Core::Elements::Element* element2,
    const std::shared_ptr<GEOMETRYPAIR::LineToSurfaceEvaluationData>&
        line_to_surface_evaluation_data)
    : GeometryPairLineToSurface<ScalarType, Line, Surface>(
          element1, element2, line_to_surface_evaluation_data)
{
  // Check if a projection tracking vector exists for this line element. If not a new one is
  // created.
  int line_element_id = this->element1()->id();
  std::map<int, std::vector<bool>>& projection_tracker =
      this->line_to_surface_evaluation_data_->get_gauss_point_projection_tracker();

  if (projection_tracker.find(line_element_id) == projection_tracker.end())
  {
    int n_gauss_points = this->line_to_surface_evaluation_data_->get_number_of_gauss_points();
    std::vector<bool> new_tracking_vector;
    new_tracking_vector.resize(n_gauss_points, false);
    projection_tracker[line_element_id] = new_tracking_vector;
  }
}

/**
 *
 */
template <typename ScalarType, typename Line, typename Surface>
void GEOMETRYPAIR::GeometryPairLineToSurfaceGaussPointProjection<ScalarType, Line,
    Surface>::pre_evaluate(const ElementData<Line, ScalarType>& element_data_line,
    const ElementData<Surface, ScalarType>& element_data_surface,
    std::vector<LineSegment<ScalarType>>& segments) const
{
  // Call the pre_evaluate method of the general Gauss point projection class.
  LineTo3DGaussPointProjection<
      GeometryPairLineToSurfaceGaussPointProjection<ScalarType, Line, Surface>>::pre_evaluate(this,
      element_data_line, element_data_surface, segments);
}

/**
 *
 */
template <typename ScalarType, typename Line, typename Surface>
void GEOMETRYPAIR::GeometryPairLineToSurfaceGaussPointProjection<ScalarType, Line,
    Surface>::evaluate(const ElementData<Line, ScalarType>& element_data_line,
    const ElementData<Surface, ScalarType>& element_data_surface,
    std::vector<LineSegment<ScalarType>>& segments) const
{
  // Call the pre_evaluate method of the general Gauss point projection class.
  LineTo3DGaussPointProjection<
      GeometryPairLineToSurfaceGaussPointProjection<ScalarType, Line, Surface>>::evaluate(this,
      element_data_line, element_data_surface, segments);
}

/**
 *
 */
template <typename ScalarType, typename Line, typename Surface>
std::vector<bool>& GEOMETRYPAIR::GeometryPairLineToSurfaceGaussPointProjection<ScalarType, Line,
    Surface>::get_line_projection_vector() const
{
  // Get the Gauss point projection tracker for this line element.
  int line_element_id = this->element1()->id();
  std::map<int, std::vector<bool>>& projection_tracker =
      this->line_to_surface_evaluation_data_->get_gauss_point_projection_tracker();
  return projection_tracker[line_element_id];
}


/**
 * Explicit template initialization of template class.
 */
namespace GEOMETRYPAIR
{

  template class GeometryPairLineToSurfaceGaussPointProjection<double, t_line2, t_tri3>;
  template class GeometryPairLineToSurfaceGaussPointProjection<double, t_line2, t_tri6>;
  template class GeometryPairLineToSurfaceGaussPointProjection<double, t_line2, t_quad4>;
  template class GeometryPairLineToSurfaceGaussPointProjection<double, t_line2, t_quad8>;
  template class GeometryPairLineToSurfaceGaussPointProjection<double, t_line2, t_quad9>;
  template class GeometryPairLineToSurfaceGaussPointProjection<double, t_line2, t_nurbs9>;

  template class GeometryPairLineToSurfaceGaussPointProjection<
      line_to_surface_patch_scalar_type_1st_order, t_line2, t_tri3>;
  template class GeometryPairLineToSurfaceGaussPointProjection<
      line_to_surface_patch_scalar_type_1st_order, t_line2, t_tri6>;
  template class GeometryPairLineToSurfaceGaussPointProjection<
      line_to_surface_patch_scalar_type_1st_order, t_line2, t_quad4>;
  template class GeometryPairLineToSurfaceGaussPointProjection<
      line_to_surface_patch_scalar_type_1st_order, t_line2, t_quad8>;
  template class GeometryPairLineToSurfaceGaussPointProjection<
      line_to_surface_patch_scalar_type_1st_order, t_line2, t_quad9>;
  template class GeometryPairLineToSurfaceGaussPointProjection<
      line_to_surface_patch_scalar_type_fixed_size_1st_order<t_line2, t_nurbs9>, t_line2, t_nurbs9>;

  template class GeometryPairLineToSurfaceGaussPointProjection<line_to_surface_patch_scalar_type,
      t_line2, t_tri3>;
  template class GeometryPairLineToSurfaceGaussPointProjection<line_to_surface_patch_scalar_type,
      t_line2, t_tri6>;
  template class GeometryPairLineToSurfaceGaussPointProjection<line_to_surface_patch_scalar_type,
      t_line2, t_quad4>;
  template class GeometryPairLineToSurfaceGaussPointProjection<line_to_surface_patch_scalar_type,
      t_line2, t_quad8>;
  template class GeometryPairLineToSurfaceGaussPointProjection<line_to_surface_patch_scalar_type,
      t_line2, t_quad9>;
  template class GeometryPairLineToSurfaceGaussPointProjection<
      line_to_surface_patch_scalar_type_fixed_size<t_line2, t_nurbs9>, t_line2, t_nurbs9>;

  template class GeometryPairLineToSurfaceGaussPointProjection<double, t_hermite, t_tri3>;
  template class GeometryPairLineToSurfaceGaussPointProjection<double, t_hermite, t_tri6>;
  template class GeometryPairLineToSurfaceGaussPointProjection<double, t_hermite, t_quad4>;
  template class GeometryPairLineToSurfaceGaussPointProjection<double, t_hermite, t_quad8>;
  template class GeometryPairLineToSurfaceGaussPointProjection<double, t_hermite, t_quad9>;
  template class GeometryPairLineToSurfaceGaussPointProjection<double, t_hermite, t_nurbs9>;

  template class GeometryPairLineToSurfaceGaussPointProjection<
      line_to_surface_patch_scalar_type_1st_order, t_hermite, t_tri3>;
  template class GeometryPairLineToSurfaceGaussPointProjection<
      line_to_surface_patch_scalar_type_1st_order, t_hermite, t_tri6>;
  template class GeometryPairLineToSurfaceGaussPointProjection<
      line_to_surface_patch_scalar_type_1st_order, t_hermite, t_quad4>;
  template class GeometryPairLineToSurfaceGaussPointProjection<
      line_to_surface_patch_scalar_type_1st_order, t_hermite, t_quad8>;
  template class GeometryPairLineToSurfaceGaussPointProjection<
      line_to_surface_patch_scalar_type_1st_order, t_hermite, t_quad9>;
  template class GeometryPairLineToSurfaceGaussPointProjection<
      line_to_surface_patch_scalar_type_fixed_size_1st_order<t_hermite, t_nurbs9>, t_hermite,
      t_nurbs9>;

  template class GeometryPairLineToSurfaceGaussPointProjection<line_to_surface_patch_scalar_type,
      t_hermite, t_tri3>;
  template class GeometryPairLineToSurfaceGaussPointProjection<line_to_surface_patch_scalar_type,
      t_hermite, t_tri6>;
  template class GeometryPairLineToSurfaceGaussPointProjection<line_to_surface_patch_scalar_type,
      t_hermite, t_quad4>;
  template class GeometryPairLineToSurfaceGaussPointProjection<line_to_surface_patch_scalar_type,
      t_hermite, t_quad8>;
  template class GeometryPairLineToSurfaceGaussPointProjection<line_to_surface_patch_scalar_type,
      t_hermite, t_quad9>;
  template class GeometryPairLineToSurfaceGaussPointProjection<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_nurbs9>, t_hermite, t_nurbs9>;
}  // namespace GEOMETRYPAIR

FOUR_C_NAMESPACE_CLOSE
