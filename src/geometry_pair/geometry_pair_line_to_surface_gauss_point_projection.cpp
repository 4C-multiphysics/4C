/*----------------------------------------------------------------------*/
/*! \file

\brief Class for interaction of lines and surfaces.

\level 1
*/


#include "geometry_pair_line_to_surface_gauss_point_projection.H"

#include "geometry_pair_element_functions.H"
#include "geometry_pair_line_to_surface_evaluation_data.H"
#include "geometry_pair_line_projection.H"
#include "geometry_pair_scalar_types.H"


/**
 *
 */
template <typename scalar_type, typename line, typename surface>
void GEOMETRYPAIR::GeometryPairLineToSurfaceGaussPointProjection<scalar_type, line,
    surface>::Setup()
{
  // Call Setup on the base class.
  GeometryPairLineToSurface<scalar_type, line, surface>::Setup();

  // Check if a projection tracking vector exists for this line element. If not a new one is
  // created.
  int line_element_id = this->Element1()->Id();
  std::map<int, std::vector<bool>>& projection_tracker =
      this->line_to_surface_evaluation_data_->GetGaussPointProjectionTrackerMutable();

  if (projection_tracker.find(line_element_id) == projection_tracker.end())
  {
    int n_gauss_points = this->line_to_surface_evaluation_data_->GetNumberOfGaussPoints();
    std::vector<bool> new_tracking_vector;
    new_tracking_vector.resize(n_gauss_points, false);
    projection_tracker[line_element_id] = new_tracking_vector;
  }
}

/**
 *
 */
template <typename scalar_type, typename line, typename surface>
void GEOMETRYPAIR::GeometryPairLineToSurfaceGaussPointProjection<scalar_type, line,
    surface>::PreEvaluate(const CORE::LINALG::Matrix<line::n_dof_, 1, scalar_type>& q_line,
    const CORE::LINALG::Matrix<surface::n_dof_, 1, scalar_type>& q_surface,
    std::vector<LineSegment<scalar_type>>& segments,
    const CORE::LINALG::Matrix<3 * surface::n_nodes_, 1, scalar_type>* nodal_normals) const
{
  // Check if the element is initialized.
  this->CheckInitSetup();

  // Call the PreEvaluate method of the general Gauss point projection class.
  LineTo3DGaussPointProjection<
      GeometryPairLineToSurfaceGaussPointProjection<scalar_type, line, surface>>::PreEvaluate(this,
      q_line, q_surface, segments, nodal_normals);
}

/**
 *
 */
template <typename scalar_type, typename line, typename surface>
void GEOMETRYPAIR::GeometryPairLineToSurfaceGaussPointProjection<scalar_type, line,
    surface>::Evaluate(const CORE::LINALG::Matrix<line::n_dof_, 1, scalar_type>& q_line,
    const CORE::LINALG::Matrix<surface::n_dof_, 1, scalar_type>& q_surface,
    std::vector<LineSegment<scalar_type>>& segments,
    const CORE::LINALG::Matrix<3 * surface::n_nodes_, 1, scalar_type>* nodal_normals) const
{
  // Check if the element is initialized.
  this->CheckInitSetup();

  // Call the PreEvaluate method of the general Gauss point projection class.
  LineTo3DGaussPointProjection<
      GeometryPairLineToSurfaceGaussPointProjection<scalar_type, line, surface>>::Evaluate(this,
      q_line, q_surface, segments, nodal_normals);
}

/**
 *
 */
template <typename scalar_type, typename line, typename surface>
std::vector<bool>& GEOMETRYPAIR::GeometryPairLineToSurfaceGaussPointProjection<scalar_type, line,
    surface>::GetLineProjectionVectorMutable() const
{
  // Get the Gauss point projection tracker for this line element.
  int line_element_id = this->Element1()->Id();
  std::map<int, std::vector<bool>>& projection_tracker =
      this->line_to_surface_evaluation_data_->GetGaussPointProjectionTrackerMutable();
  return projection_tracker[line_element_id];
}


/**
 * Explicit template initialization of template class.
 */
namespace GEOMETRYPAIR
{
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
