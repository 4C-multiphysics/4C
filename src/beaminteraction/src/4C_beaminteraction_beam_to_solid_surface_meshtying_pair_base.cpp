// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_surface_meshtying_pair_base.hpp"

#include "4C_beaminteraction_beam_to_solid_surface_meshtying_params.hpp"
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

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
BeamInteraction::BeamToSolidSurfaceMeshtyingPairBase<ScalarType, Beam,
    Surface>::BeamToSolidSurfaceMeshtyingPairBase()
    : base_class(), meshtying_is_evaluated_(false)
{
  // Empty constructor.
}

/**
 *
 */
template <typename ScalarTypeFad, typename Beam, typename Solid>
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairBase<ScalarTypeFad, Beam, Solid>::reset_state(
    const std::vector<double>& beam_centerline_dofvec,
    const std::vector<double>& solid_nodal_dofvec)
{
  // Beam element.
  const int n_patch_dof = face_element_->get_patch_gid().size();
  for (unsigned int i = 0; i < Beam::n_dof_; i++)
    this->ele1pos_.element_position_(i) = Core::FADUtils::HigherOrderFadValue<ScalarTypeFad>::apply(
        Beam::n_dof_ + n_patch_dof, i, beam_centerline_dofvec[i]);
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairBase<ScalarType, Beam, Surface>::pre_evaluate()
{
  // Call pre_evaluate on the geometry Pair.
  if (!meshtying_is_evaluated_)
  {
    cast_geometry_pair()->pre_evaluate(this->ele1posref_,
        this->face_element_->get_face_reference_element_data(), this->line_to_3D_segments_);
  }
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairBase<ScalarType, Beam,
    Surface>::get_pair_visualization(std::shared_ptr<BeamToSolidVisualizationOutputWriterBase>
                                         visualization_writer,
    Teuchos::ParameterList& visualization_params) const
{
  // Get visualization of base class.
  base_class::get_pair_visualization(visualization_writer, visualization_params);

  // Add segmentation and integration point data.
  std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization>
      visualization_segmentation =
          visualization_writer->get_visualization_writer("btss-coupling-segmentation");
  if (visualization_segmentation != nullptr)
  {
    std::vector<GEOMETRYPAIR::ProjectionPoint1DTo3D<double>> points;
    for (const auto& segment : this->line_to_3D_segments_)
      for (const auto& segmentation_point : {segment.get_start_point(), segment.get_end_point()})
        points.push_back(segmentation_point);
    add_visualization_integration_points(*visualization_segmentation, points, visualization_params);
  }

  std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization>
      visualization_integration_points =
          visualization_writer->get_visualization_writer("btss-coupling-integration-points");
  if (visualization_integration_points != nullptr)
  {
    std::vector<GEOMETRYPAIR::ProjectionPoint1DTo3D<double>> points;
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
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairBase<ScalarType, Beam, Surface>::
    add_visualization_integration_points(
        BeamInteraction::BeamToSolidOutputWriterVisualization& visualization_writer,
        const std::vector<GEOMETRYPAIR::ProjectionPoint1DTo3D<double>>& points,
        const Teuchos::ParameterList& visualization_params) const
{
  auto& visualization_data = visualization_writer.get_visualization_data();

  // Setup variables.
  Core::LinAlg::Matrix<3, 1, ScalarType> X_beam, u_beam, r_beam, r_solid, projection_dir;

  // Get the visualization vectors.
  std::vector<double>& point_coordinates = visualization_data.get_point_coordinates();
  std::vector<double>& displacement = visualization_data.get_point_data<double>("displacement");
  std::vector<double>& projection_direction =
      visualization_data.get_point_data<double>("projection_direction");

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
    GEOMETRYPAIR::evaluate_position<Beam>(point.get_eta(), this->ele1posref_, X_beam);
    GEOMETRYPAIR::evaluate_position<Beam>(point.get_eta(), this->ele1pos_, r_beam);
    u_beam = r_beam;
    u_beam -= X_beam;

    GEOMETRYPAIR::evaluate_position<Surface>(
        point.get_xi(), this->face_element_->get_face_element_data(), r_solid);
    projection_dir = r_solid;
    projection_dir -= r_beam;

    for (unsigned int dim = 0; dim < 3; dim++)
    {
      point_coordinates.push_back(Core::FADUtils::cast_to_double(X_beam(dim)));
      displacement.push_back(Core::FADUtils::cast_to_double(u_beam(dim)));
      projection_direction.push_back(Core::FADUtils::cast_to_double(projection_dir(dim)));
    }

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
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairBase<ScalarType, Beam,
    Surface>::create_geometry_pair(const Core::Elements::Element* element1,
    const Core::Elements::Element* element2,
    const std::shared_ptr<GEOMETRYPAIR::GeometryEvaluationDataBase>& geometry_evaluation_data_ptr)
{
  this->geometry_pair_ = GEOMETRYPAIR::geometry_pair_line_to_surface_factory<double, Beam, Surface>(
      element1, element2, geometry_evaluation_data_ptr);
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairBase<ScalarType, Beam,
    Surface>::set_face_element(std::shared_ptr<GEOMETRYPAIR::FaceElement>& face_element)
{
  face_element_ = std::dynamic_pointer_cast<GEOMETRYPAIR::FaceElementTemplate<Surface, ScalarType>>(
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
std::shared_ptr<GEOMETRYPAIR::GeometryPairLineToSurface<double, Beam, Surface>>
BeamInteraction::BeamToSolidSurfaceMeshtyingPairBase<ScalarType, Beam,
    Surface>::cast_geometry_pair() const
{
  return std::dynamic_pointer_cast<GEOMETRYPAIR::GeometryPairLineToSurface<double, Beam, Surface>>(
      this->geometry_pair_);
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
Core::LinAlg::Matrix<3, 1, ScalarType>
BeamInteraction::BeamToSolidSurfaceMeshtyingPairBase<ScalarType, Beam, Surface>::evaluate_coupling(
    const GEOMETRYPAIR::ProjectionPoint1DTo3D<double>& evaluation_point) const
{
  using namespace Inpar::BeamToSolid;

  Core::LinAlg::Matrix<3, 1, ScalarType> r_beam(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, ScalarType> r_surface(Core::LinAlg::Initialization::zero);

  const BeamToSolidSurfaceCoupling coupling_type =
      this->params()->beam_to_solid_surface_meshtying_params()->get_coupling_type();
  switch (coupling_type)
  {
    case BeamToSolidSurfaceCoupling::displacement:
    case BeamToSolidSurfaceCoupling::displacement_fad:
    {
      // In this case we have to subtract the reference position from the DOF vectors.
      auto beam_dof = this->ele1pos_;
      auto surface_dof = this->face_element_->get_face_element_data();

      for (unsigned int i_dof_beam = 0; i_dof_beam < Beam::n_dof_; i_dof_beam++)
        beam_dof.element_position_(i_dof_beam) -= this->ele1posref_.element_position_(i_dof_beam);
      for (unsigned int i_dof_surface = 0; i_dof_surface < Surface::n_dof_; i_dof_surface++)
        surface_dof.element_position_(i_dof_surface) -=
            this->face_element_->get_face_reference_element_data().element_position_(i_dof_surface);

      GEOMETRYPAIR::evaluate_position<Beam>(evaluation_point.get_eta(), beam_dof, r_beam);
      GEOMETRYPAIR::evaluate_position<Surface>(evaluation_point.get_xi(), surface_dof, r_surface);

      r_beam -= r_surface;
      return r_beam;
    }
    case BeamToSolidSurfaceCoupling::reference_configuration_forced_to_zero:
    case BeamToSolidSurfaceCoupling::reference_configuration_forced_to_zero_fad:
    {
      GEOMETRYPAIR::evaluate_position<Beam>(evaluation_point.get_eta(), this->ele1pos_, r_beam);
      GEOMETRYPAIR::evaluate_position<Surface>(
          evaluation_point.get_xi(), this->face_element_->get_face_element_data(), r_surface);

      r_beam -= r_surface;
      return r_beam;
    }
    case BeamToSolidSurfaceCoupling::consistent_fad:
    {
      GEOMETRYPAIR::evaluate_position<Beam>(evaluation_point.get_eta(), this->ele1pos_, r_beam);
      GEOMETRYPAIR::evaluate_surface_position<Surface>(
          evaluation_point.get_xi(), this->face_element_->get_face_element_data(), r_surface);

      r_beam -= r_surface;
      return r_beam;
    }
    default:
    {
      FOUR_C_THROW("Got unexpected coupling type.");
      return r_beam;
    }
  }
}

/**
 * Explicit template initialization of template class.
 */
namespace BeamInteraction
{
  using namespace GEOMETRYPAIR;

  template class BeamToSolidSurfaceMeshtyingPairBase<
      line_to_surface_scalar_type<t_hermite, t_quad4>, t_hermite, t_quad4>;
  template class BeamToSolidSurfaceMeshtyingPairBase<
      line_to_surface_scalar_type<t_hermite, t_quad8>, t_hermite, t_quad8>;
  template class BeamToSolidSurfaceMeshtyingPairBase<
      line_to_surface_scalar_type<t_hermite, t_quad9>, t_hermite, t_quad9>;
  template class BeamToSolidSurfaceMeshtyingPairBase<line_to_surface_scalar_type<t_hermite, t_tri3>,
      t_hermite, t_tri3>;
  template class BeamToSolidSurfaceMeshtyingPairBase<line_to_surface_scalar_type<t_hermite, t_tri6>,
      t_hermite, t_tri6>;
  template class BeamToSolidSurfaceMeshtyingPairBase<
      line_to_surface_scalar_type<t_hermite, t_nurbs9>, t_hermite, t_nurbs9>;

  template class BeamToSolidSurfaceMeshtyingPairBase<line_to_surface_patch_scalar_type, t_hermite,
      t_quad4>;
  template class BeamToSolidSurfaceMeshtyingPairBase<line_to_surface_patch_scalar_type, t_hermite,
      t_quad8>;
  template class BeamToSolidSurfaceMeshtyingPairBase<line_to_surface_patch_scalar_type, t_hermite,
      t_quad9>;
  template class BeamToSolidSurfaceMeshtyingPairBase<line_to_surface_patch_scalar_type, t_hermite,
      t_tri3>;
  template class BeamToSolidSurfaceMeshtyingPairBase<line_to_surface_patch_scalar_type, t_hermite,
      t_tri6>;
  template class BeamToSolidSurfaceMeshtyingPairBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_nurbs9>, t_hermite, t_nurbs9>;
  template class BeamToSolidSurfaceMeshtyingPairBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex8>, t_hermite, t_quad4>;
  template class BeamToSolidSurfaceMeshtyingPairBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex20>, t_hermite, t_quad8>;
  template class BeamToSolidSurfaceMeshtyingPairBase<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex27>, t_hermite, t_quad9>;
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE
