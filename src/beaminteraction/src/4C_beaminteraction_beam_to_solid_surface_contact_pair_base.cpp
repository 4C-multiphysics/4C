// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_surface_contact_pair_base.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_contact_params.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_visualization_output_params.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_base.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_visualization.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_element_faces.hpp"
#include "4C_geometry_pair_factory.hpp"
#include "4C_geometry_pair_line_to_surface.hpp"
#include "4C_geometry_pair_scalar_types.hpp"
#include "4C_mat_shell_kl.hpp"
#include "4C_shell_kl_nurbs.hpp"


FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
BeamInteraction::BeamToSolidSurfaceContactPairBase<ScalarType, Beam,
    Surface>::BeamToSolidSurfaceContactPairBase()
    : base_class()
{
  // Empty constructor.
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Solid>
void BeamInteraction::BeamToSolidSurfaceContactPairBase<ScalarType, Beam, Solid>::reset_state(
    const std::vector<double>& beam_centerline_dofvec,
    const std::vector<double>& solid_nodal_dofvec)
{
  // Clean the segments, as they will be re-evaluated in each iteration.
  this->line_to_3D_segments_.clear();

  // Set the current position of the beam element.
  const int n_patch_dof = face_element_->get_patch_gid().size();
  for (unsigned int i = 0; i < Beam::n_dof_; i++)
    this->ele1pos_.element_position_(i) = Core::FADUtils::HigherOrderFadValue<ScalarType>::apply(
        Beam::n_dof_ + n_patch_dof, i, beam_centerline_dofvec[i]);
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
void BeamInteraction::BeamToSolidSurfaceContactPairBase<ScalarType, Beam, Surface>::pre_evaluate()
{
  // Call pre_evaluate on the geometry Pair.
  cast_geometry_pair()->pre_evaluate(
      this->ele1pos_, this->face_element_->get_face_element_data(), this->line_to_3D_segments_);
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
void BeamInteraction::BeamToSolidSurfaceContactPairBase<ScalarType, Beam,
    Surface>::get_pair_visualization(std::shared_ptr<BeamToSolidVisualizationOutputWriterBase>
                                         visualization_writer,
    Teuchos::ParameterList& visualization_params) const
{
  // Get visualization of base class.
  base_class::get_pair_visualization(visualization_writer, visualization_params);

  // Add segmentation and integration point data.
  std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization>
      visualization_segmentation =
          visualization_writer->get_visualization_writer("btss-contact-segmentation");
  if (visualization_segmentation != nullptr)
  {
    std::vector<GeometryPair::ProjectionPoint1DTo3D<ScalarType>> points;
    for (const auto& segment : this->line_to_3D_segments_)
      for (const auto& segmentation_point : {segment.get_start_point(), segment.get_end_point()})
        points.push_back(segmentation_point);
    add_visualization_integration_points(*visualization_segmentation, points, visualization_params);
  }

  std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization>
      visualization_integration_points =
          visualization_writer->get_visualization_writer("btss-contact-integration-points");
  if (visualization_integration_points != nullptr)
  {
    std::vector<GeometryPair::ProjectionPoint1DTo3D<ScalarType>> points;
    for (const auto& segment : this->line_to_3D_segments_)
      for (const auto& segmentation_point : (segment.get_projection_points()))
        points.push_back(segmentation_point);
    add_visualization_integration_points(
        *visualization_integration_points, points, visualization_params);
  }
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
void BeamInteraction::BeamToSolidSurfaceContactPairBase<ScalarType, Beam, Surface>::
    add_visualization_integration_points(
        BeamInteraction::BeamToSolidOutputWriterVisualization& visualization_writer,
        const std::vector<GeometryPair::ProjectionPoint1DTo3D<ScalarType>>& points,
        const Teuchos::ParameterList& visualization_params) const
{
  auto& visualization_data = visualization_writer.get_visualization_data();

  // Setup variables.
  Core::LinAlg::Matrix<3, 1, ScalarType> X_beam, u_beam;

  // Get beam cross-section diameter.
  auto beam_ptr = dynamic_cast<const Discret::Elements::Beam3Base*>(this->element1());
  const double beam_cross_section_radius =
      beam_ptr->get_circular_cross_section_radius_for_interactions();

  // Get the visualization vectors.
  std::vector<double>& point_coordinates = visualization_data.get_point_coordinates();
  std::vector<double>& displacement = visualization_data.get_point_data<double>("displacement");
  std::vector<double>& surface_normal_data =
      visualization_data.get_point_data<double>("surface_normal");
  std::vector<double>& gap_data = visualization_data.get_point_data<double>("gap");
  std::vector<double>& force_data = visualization_data.get_point_data<double>("force");

  const std::shared_ptr<const BeamToSolidSurfaceVisualizationOutputParams>& output_params_ptr =
      visualization_params.get<std::shared_ptr<const BeamToSolidSurfaceVisualizationOutputParams>>(
          "btss-output_params_ptr");
  const bool write_unique_ids = output_params_ptr->get_write_unique_ids_flag();
  std::vector<int>* pair_beam_id = nullptr;
  std::vector<int>* pair_solid_id = nullptr;
  if (write_unique_ids)
  {
    pair_beam_id = &(visualization_data.get_point_data<int>("uid_0_pair_beam_id"));
    pair_solid_id = &(visualization_data.get_point_data<int>("uid_1_pair_solid_id"));
  }

  for (const auto& point : points)
  {
    const auto [r_beam, r_surface, surface_normal, gap] =
        this->evaluate_contact_kinematics_at_projection_point(point, beam_cross_section_radius);
    GeometryPair::evaluate_position<Beam>(point.get_eta(), this->ele1posref_, X_beam);

    u_beam = r_beam;
    u_beam -= X_beam;
    const auto force = penalty_force(gap, *this->params()->beam_to_solid_surface_contact_params());

    for (unsigned int dim = 0; dim < 3; dim++)
    {
      point_coordinates.push_back(Core::FADUtils::cast_to_double(X_beam(dim)));
      displacement.push_back(Core::FADUtils::cast_to_double(u_beam(dim)));
      surface_normal_data.push_back(Core::FADUtils::cast_to_double(surface_normal(dim)));
      force_data.push_back(Core::FADUtils::cast_to_double(force * surface_normal(dim)));
    }
    gap_data.push_back(Core::FADUtils::cast_to_double(gap));

    if (write_unique_ids)
    {
      pair_beam_id->push_back(this->element1()->id());
      pair_solid_id->push_back(this->element2()->id());
    }
  }
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
void BeamInteraction::BeamToSolidSurfaceContactPairBase<ScalarType, Beam,
    Surface>::create_geometry_pair(const Core::Elements::Element* element1,
    const Core::Elements::Element* element2,
    const std::shared_ptr<GeometryPair::GeometryEvaluationDataBase>& geometry_evaluation_data_ptr)
{
  this->geometry_pair_ =
      GeometryPair::geometry_pair_line_to_surface_factory_fad<ScalarType, Beam, Surface>(
          element1, element2, geometry_evaluation_data_ptr);
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
void BeamInteraction::BeamToSolidSurfaceContactPairBase<ScalarType, Beam,
    Surface>::set_face_element(std::shared_ptr<GeometryPair::FaceElement>& face_element)
{
  face_element_ = std::dynamic_pointer_cast<GeometryPair::FaceElementTemplate<Surface, ScalarType>>(
      face_element);

  // Set the number of (centerline) degrees of freedom for the beam element in the face element
  face_element_->set_number_of_dof_other_element(
      Utils::get_number_of_element_centerline_dof(this->element1()));

  // If the solid surface is the surface of a 3D volume we set the face element here. Otherwise we
  // simply set the same element again.
  cast_geometry_pair()->set_element2(face_element_->get_element());
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
std::shared_ptr<GeometryPair::GeometryPairLineToSurface<ScalarType, Beam, Surface>>
BeamInteraction::BeamToSolidSurfaceContactPairBase<ScalarType, Beam, Surface>::cast_geometry_pair()
    const
{
  return std::dynamic_pointer_cast<
      GeometryPair::GeometryPairLineToSurface<ScalarType, Beam, Surface>>(this->geometry_pair_);
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
std::tuple<Core::LinAlg::Matrix<3, 1, ScalarType>, Core::LinAlg::Matrix<3, 1, ScalarType>,
    Core::LinAlg::Matrix<3, 1, ScalarType>, ScalarType>
BeamInteraction::BeamToSolidSurfaceContactPairBase<ScalarType, Beam, Surface>::
    evaluate_contact_kinematics_at_projection_point(
        const GeometryPair::ProjectionPoint1DTo3D<ScalarType>& projection_point,
        const double beam_cross_section_radius) const
{
  // Get the projection coordinates
  const auto& xi = projection_point.get_xi();
  const auto& eta = projection_point.get_eta();

  // Get the surface normal vector
  Core::LinAlg::Matrix<3, 1, ScalarType> surface_normal;
  GeometryPair::evaluate_surface_normal<Surface>(
      xi, this->face_element_->get_face_element_data(), surface_normal);

  // Evaluate the current position of beam and solid
  Core::LinAlg::Matrix<3, 1, ScalarType> r_beam;
  Core::LinAlg::Matrix<3, 1, ScalarType> r_surface;
  GeometryPair::evaluate_position<Beam>(eta, this->ele1pos_, r_beam);
  GeometryPair::evaluate_position<Surface>(
      xi, this->face_element_->get_face_element_data(), r_surface);

  // Evaluate the gap function
  Core::LinAlg::Matrix<3, 1, ScalarType> r_rel;
  r_rel = r_beam;
  r_rel -= r_surface;
  ScalarType gap = r_rel.dot(surface_normal);
  if constexpr (std::is_same<Surface, GeometryPair::t_nurbs9>::value)
  {
    const auto* kl_shell =
        dynamic_cast<const Discret::Elements::KirchhoffLoveShellNurbs*>(this->element2());
    if (kl_shell != nullptr)
    {
      // For shell elements we need to check which side of the shell the beam interacts with. And
      // account for the shell thickness.
      if (gap < 0)
      {
        // In this case we switch the normal direction, because the contact is happening on the
        // "negative" side of the face.
        surface_normal.scale(-1.0);
        gap *= -1.0;
      }
      const double shell_thickness =
          std::dynamic_pointer_cast<const Mat::KirchhoffLoveShell>(kl_shell->material())
              ->thickness();
      gap -= 0.5 * shell_thickness;
    }
  }
  gap -= beam_cross_section_radius;

  return {r_beam, r_surface, surface_normal, gap};
}

/**
 * Explicit template initialization of template class.
 */
namespace BeamInteraction
{
  using namespace GeometryPair;

  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type_1st_order,
      t_line2, t_tri3>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type_1st_order,
      t_line2, t_tri6>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type_1st_order,
      t_line2, t_quad4>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type_1st_order,
      t_line2, t_quad8>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type_1st_order,
      t_line2, t_quad9>;
  template class BeamToSolidSurfaceContactPairBase<
      line_to_surface_patch_scalar_type_fixed_size_1st_order<t_line2, t_nurbs9>, t_line2, t_nurbs9>;

  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type, t_line2,
      t_tri3>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type, t_line2,
      t_tri6>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type, t_line2,
      t_quad4>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type, t_line2,
      t_quad8>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type, t_line2,
      t_quad9>;
  template class BeamToSolidSurfaceContactPairBase<
      line_to_surface_patch_scalar_type_fixed_size<t_line2, t_nurbs9>, t_line2, t_nurbs9>;


  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type_1st_order,
      t_hermite, t_tri3>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type_1st_order,
      t_hermite, t_tri6>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type_1st_order,
      t_hermite, t_quad4>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type_1st_order,
      t_hermite, t_quad8>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type_1st_order,
      t_hermite, t_quad9>;
  template class BeamToSolidSurfaceContactPairBase<
      line_to_surface_patch_scalar_type_fixed_size_1st_order<t_hermite, t_nurbs9>, t_hermite,
      t_nurbs9>;

  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type, t_hermite,
      t_tri3>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type, t_hermite,
      t_tri6>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type, t_hermite,
      t_quad4>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type, t_hermite,
      t_quad8>;
  template class BeamToSolidSurfaceContactPairBase<line_to_surface_patch_scalar_type, t_hermite,
      t_quad9>;
  template class BeamToSolidSurfaceContactPairBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_nurbs9>, t_hermite, t_nurbs9>;

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE
