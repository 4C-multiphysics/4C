// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_surface_contact_pair.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_contact_params.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_visualization_output_params.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_base.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_visualization.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_element_faces.hpp"
#include "4C_geometry_pair_factory.hpp"
#include "4C_geometry_pair_line_to_surface.hpp"
#include "4C_geometry_pair_scalar_types.hpp"
#include "4C_linalg_fevector.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
BeamInteraction::BeamToSolidSurfaceContactPairGapVariation<ScalarType, Beam,
    Surface>::BeamToSolidSurfaceContactPairGapVariation()
    : base_class()
{
  // Empty constructor.
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
void BeamInteraction::BeamToSolidSurfaceContactPairGapVariation<ScalarType, Beam,
    Surface>::evaluate_and_assemble(const std::shared_ptr<const Core::FE::Discretization>& discret,
    const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
    const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector)
{
  // Call Evaluate on the geometry Pair.
  this->cast_geometry_pair()->evaluate(
      this->ele1pos_, this->face_element_->get_face_element_data(), this->line_to_3D_segments_);

  // If there are no intersection segments, no contact terms will be assembled.
  const unsigned int n_segments = this->line_to_3D_segments_.size();
  if (n_segments == 0) return;

  // Get beam cross-section diameter.
  auto beam_ptr = dynamic_cast<const Discret::Elements::Beam3Base*>(this->element1());
  const double beam_cross_section_radius =
      beam_ptr->get_circular_cross_section_radius_for_interactions();

  // Initialize variables for contact kinematics.
  Core::LinAlg::Matrix<3, 1, ScalarType> dr_beam_ref;
  Core::LinAlg::Matrix<1, Beam::n_nodes_ * Beam::n_val_, ScalarType> N_beam;
  Core::LinAlg::Matrix<1, Surface::n_nodes_ * Surface::n_val_, ScalarType> N_surface;
  Core::LinAlg::Matrix<Beam::n_dof_ + Surface::n_dof_, 1, ScalarType> gap_variation_times_normal;
  Core::LinAlg::Matrix<Beam::n_dof_ + Surface::n_dof_, 1, ScalarType> pair_force_vector;
  ScalarType segment_jacobian = 0.0;
  ScalarType beam_segmentation_factor = 0.0;

  // Integrate over segments.
  for (unsigned int i_segment = 0; i_segment < n_segments; i_segment++)
  {
    // Factor to account for the integration segment length.
    beam_segmentation_factor = 0.5 * this->line_to_3D_segments_[i_segment].get_segment_length();

    // Gauss point loop.
    const unsigned int n_gp = this->line_to_3D_segments_[i_segment].get_projection_points().size();
    for (unsigned int i_gp = 0; i_gp < n_gp; i_gp++)
    {
      // Get the current Gauss point.
      const auto& projected_gauss_point =
          this->line_to_3D_segments_[i_segment].get_projection_points()[i_gp];
      const auto& eta = projected_gauss_point.get_eta();
      const auto& xi = projected_gauss_point.get_xi();

      // Get the Jacobian in the reference configuration.
      GeometryPair::evaluate_position_derivative1<Beam>(eta, this->ele1posref_, dr_beam_ref);

      // Jacobian including the segment length.
      segment_jacobian = Core::FADUtils::vector_norm(dr_beam_ref) * beam_segmentation_factor;

      // Get the contact kinematics
      const auto [_1, _2, surface_normal, gap] =
          this->evaluate_contact_kinematics_at_projection_point(
              projected_gauss_point, beam_cross_section_radius);

      // Get the shape function matrices.
      GeometryPair::EvaluateShapeFunction<Beam>::evaluate(
          N_beam, eta, this->ele1pos_.shape_function_data_);
      GeometryPair::EvaluateShapeFunction<Surface>::evaluate(
          N_surface, xi, this->face_element_->get_face_element_data().shape_function_data_);

      // Calculate the variation of the gap function multiplied with the surface normal vector.
      for (unsigned int i_shape = 0; i_shape < N_beam.num_cols(); i_shape++)
      {
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        {
          gap_variation_times_normal(i_shape * 3 + i_dim) = N_beam(i_shape) * surface_normal(i_dim);
        }
      }
      for (unsigned int i_shape = 0; i_shape < N_surface.num_cols(); i_shape++)
      {
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        {
          gap_variation_times_normal(N_beam.num_cols() * 3 + i_shape * 3 + i_dim) =
              -1.0 * N_surface(i_shape) * surface_normal(i_dim);
        }
      }

      // Get the contact force.
      ScalarType force =
          penalty_force(gap, *this->params()->beam_to_solid_surface_contact_params());

      // Add the Gauss point contributions to the pair force vector.
      gap_variation_times_normal.scale(
          force * projected_gauss_point.get_gauss_weight() * segment_jacobian);
      pair_force_vector -= gap_variation_times_normal;
    }
  }

  // GIDs of the pair and the force vector acting on the pair.
  const auto pair_gid = get_beam_to_surface_pair_gid_combined<Beam>(
      *discret, *this->element1(), *this->face_element_);

  // If given, assemble force terms into the global vector.
  if (force_vector != nullptr)
  {
    std::vector<double> force_pair_double(pair_gid.size(), 0.0);
    for (unsigned int j_dof = 0; j_dof < pair_force_vector.num_rows(); j_dof++)
      force_pair_double[j_dof] = Core::FADUtils::cast_to_double(pair_force_vector(j_dof));
    force_vector->sum_into_global_values(
        pair_gid.size(), pair_gid.data(), force_pair_double.data());
  }

  // If given, assemble force terms into the global stiffness matrix.
  if (stiffness_matrix != nullptr)
    for (unsigned int i_dof = 0; i_dof < pair_force_vector.num_rows(); i_dof++)
      for (unsigned int j_dof = 0; j_dof < pair_gid.size(); j_dof++)
        stiffness_matrix->fe_assemble(
            Core::FADUtils::cast_to_double(pair_force_vector(i_dof).dx(j_dof)), pair_gid[i_dof],
            pair_gid[j_dof]);
}


/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
BeamInteraction::BeamToSolidSurfaceContactPairPotential<ScalarType, Beam,
    Surface>::BeamToSolidSurfaceContactPairPotential()
    : base_class()
{
  // Empty constructor.
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface>
void BeamInteraction::BeamToSolidSurfaceContactPairPotential<ScalarType, Beam,
    Surface>::evaluate_and_assemble(const std::shared_ptr<const Core::FE::Discretization>& discret,
    const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
    const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector)
{
  // Call Evaluate on the geometry Pair.
  this->cast_geometry_pair()->evaluate(
      this->ele1pos_, this->face_element_->get_face_element_data(), this->line_to_3D_segments_);

  // If there are no intersection segments, no contact terms will be assembled.
  const unsigned int n_segments = this->line_to_3D_segments_.size();
  if (n_segments == 0) return;

  // Get beam cross-section diameter.
  auto beam_ptr = dynamic_cast<const Discret::Elements::Beam3Base*>(this->element1());
  const double beam_cross_section_radius =
      beam_ptr->get_circular_cross_section_radius_for_interactions();

  // Initialize variables for contact kinematics.
  Core::LinAlg::Matrix<3, 1, ScalarType> dr_beam_ref;
  ScalarType potential = 0.0;
  ScalarType segment_jacobian = 0.0;
  ScalarType beam_segmentation_factor = 0.0;

  // Integrate over segments.
  for (unsigned int i_segment = 0; i_segment < n_segments; i_segment++)
  {
    // Factor to account for the integration segment length.
    beam_segmentation_factor = 0.5 * this->line_to_3D_segments_[i_segment].get_segment_length();

    // Gauss point loop.
    const unsigned int n_gp = this->line_to_3D_segments_[i_segment].get_projection_points().size();
    for (unsigned int i_gp = 0; i_gp < n_gp; i_gp++)
    {
      // Get the current Gauss point.
      const auto& projected_gauss_point =
          this->line_to_3D_segments_[i_segment].get_projection_points()[i_gp];
      const auto& eta = projected_gauss_point.get_eta();

      // Get the Jacobian in the reference configuration.
      GeometryPair::evaluate_position_derivative1<Beam>(eta, this->ele1posref_, dr_beam_ref);

      // Jacobian including the segment length.
      segment_jacobian = Core::FADUtils::vector_norm(dr_beam_ref) * beam_segmentation_factor;

      // Evaluate the contact kinematics
      const auto gap = std::get<3>(this->evaluate_contact_kinematics_at_projection_point(
          projected_gauss_point, beam_cross_section_radius));

      // Get the contact force.
      potential += projected_gauss_point.get_gauss_weight() * segment_jacobian *
                   penalty_potential(gap, *this->params()->beam_to_solid_surface_contact_params());
    }
  }

  // GIDs of the pair and the force vector acting on the pair.
  const auto pair_gid = get_beam_to_surface_pair_gid_combined<Beam>(
      *discret, *this->element1(), *this->face_element_);

  // If given, assemble force terms into the global vector.
  if (force_vector != nullptr)
  {
    std::vector<double> force_pair_double(pair_gid.size(), 0.0);
    for (unsigned int j_dof = 0; j_dof < pair_gid.size(); j_dof++)
      force_pair_double[j_dof] = Core::FADUtils::cast_to_double(potential.dx(j_dof));
    force_vector->sum_into_global_values(
        pair_gid.size(), pair_gid.data(), force_pair_double.data());
  }

  // If given, assemble force terms into the global stiffness matrix.
  if (stiffness_matrix != nullptr)
    for (unsigned int i_dof = 0; i_dof < pair_gid.size(); i_dof++)
      for (unsigned int j_dof = 0; j_dof < pair_gid.size(); j_dof++)
        stiffness_matrix->fe_assemble(Core::FADUtils::cast_to_double(potential.dx(i_dof).dx(j_dof)),
            pair_gid[i_dof], pair_gid[j_dof]);
}


/**
 * Explicit template initialization of template class.
 */
namespace BeamInteraction
{
  using namespace GeometryPair;

  template class BeamToSolidSurfaceContactPairGapVariation<
      line_to_surface_patch_scalar_type_1st_order, t_line2, t_tri3>;
  template class BeamToSolidSurfaceContactPairGapVariation<
      line_to_surface_patch_scalar_type_1st_order, t_line2, t_tri6>;
  template class BeamToSolidSurfaceContactPairGapVariation<
      line_to_surface_patch_scalar_type_1st_order, t_line2, t_quad4>;
  template class BeamToSolidSurfaceContactPairGapVariation<
      line_to_surface_patch_scalar_type_1st_order, t_line2, t_quad8>;
  template class BeamToSolidSurfaceContactPairGapVariation<
      line_to_surface_patch_scalar_type_1st_order, t_line2, t_quad9>;
  template class BeamToSolidSurfaceContactPairGapVariation<
      line_to_surface_patch_scalar_type_fixed_size_1st_order<t_line2, t_nurbs9>, t_line2, t_nurbs9>;

  template class BeamToSolidSurfaceContactPairPotential<line_to_surface_patch_scalar_type, t_line2,
      t_tri3>;
  template class BeamToSolidSurfaceContactPairPotential<line_to_surface_patch_scalar_type, t_line2,
      t_tri6>;
  template class BeamToSolidSurfaceContactPairPotential<line_to_surface_patch_scalar_type, t_line2,
      t_quad4>;
  template class BeamToSolidSurfaceContactPairPotential<line_to_surface_patch_scalar_type, t_line2,
      t_quad8>;
  template class BeamToSolidSurfaceContactPairPotential<line_to_surface_patch_scalar_type, t_line2,
      t_quad9>;
  template class BeamToSolidSurfaceContactPairPotential<
      line_to_surface_patch_scalar_type_fixed_size<t_line2, t_nurbs9>, t_line2, t_nurbs9>;

  template class BeamToSolidSurfaceContactPairGapVariation<
      line_to_surface_patch_scalar_type_1st_order, t_hermite, t_tri3>;
  template class BeamToSolidSurfaceContactPairGapVariation<
      line_to_surface_patch_scalar_type_1st_order, t_hermite, t_tri6>;
  template class BeamToSolidSurfaceContactPairGapVariation<
      line_to_surface_patch_scalar_type_1st_order, t_hermite, t_quad4>;
  template class BeamToSolidSurfaceContactPairGapVariation<
      line_to_surface_patch_scalar_type_1st_order, t_hermite, t_quad8>;
  template class BeamToSolidSurfaceContactPairGapVariation<
      line_to_surface_patch_scalar_type_1st_order, t_hermite, t_quad9>;
  template class BeamToSolidSurfaceContactPairGapVariation<
      line_to_surface_patch_scalar_type_fixed_size_1st_order<t_hermite, t_nurbs9>, t_hermite,
      t_nurbs9>;

  template class BeamToSolidSurfaceContactPairPotential<line_to_surface_patch_scalar_type,
      t_hermite, t_tri3>;
  template class BeamToSolidSurfaceContactPairPotential<line_to_surface_patch_scalar_type,
      t_hermite, t_tri6>;
  template class BeamToSolidSurfaceContactPairPotential<line_to_surface_patch_scalar_type,
      t_hermite, t_quad4>;
  template class BeamToSolidSurfaceContactPairPotential<line_to_surface_patch_scalar_type,
      t_hermite, t_quad8>;
  template class BeamToSolidSurfaceContactPairPotential<line_to_surface_patch_scalar_type,
      t_hermite, t_quad9>;
  template class BeamToSolidSurfaceContactPairPotential<
      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_nurbs9>, t_hermite, t_nurbs9>;

}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE
