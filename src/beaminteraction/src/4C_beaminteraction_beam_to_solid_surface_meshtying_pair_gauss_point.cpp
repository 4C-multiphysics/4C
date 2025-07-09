// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_surface_meshtying_pair_gauss_point.hpp"

#include "4C_beaminteraction_beam_to_solid_surface_meshtying_params.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_element_faces.hpp"
#include "4C_geometry_pair_factory.hpp"
#include "4C_geometry_pair_line_to_surface.hpp"
#include "4C_linalg_fevector.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename Beam, typename Surface>
BeamInteraction::BeamToSolidSurfaceMeshtyingPairGaussPoint<Beam,
    Surface>::BeamToSolidSurfaceMeshtyingPairGaussPoint()
    : base_class()
{
  // Empty constructor.
}

/**
 *
 */
template <typename Beam, typename Surface>
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairGaussPoint<Beam,
    Surface>::evaluate_and_assemble(const std::shared_ptr<const Core::FE::Discretization>& discret,
    const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
    const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector)
{
  // Call Evaluate on the geometry Pair. Only do this once for mesh tying.
  if (!this->meshtying_is_evaluated_)
  {
    this->cast_geometry_pair()->evaluate(this->ele1posref_,
        this->face_element_->get_face_reference_element_data(), this->line_to_3D_segments_);
    this->meshtying_is_evaluated_ = true;
  }

  // If there are no intersection segments, no coupling terms will be assembled.
  if (this->line_to_3D_segments_.size() == 0) return;

  // Initialize variables for position and force vectors.
  Core::LinAlg::Matrix<3, 1, double> dr_beam_ref;
  Core::LinAlg::Matrix<3, 1, scalar_type> coupling_vector;
  Core::LinAlg::Matrix<3, 1, scalar_type> force;
  Core::LinAlg::Matrix<Beam::n_dof_ + Surface::n_dof_, 1, scalar_type> force_pair(
      Core::LinAlg::Initialization::zero);

  // Initialize scalar variables.
  double segment_jacobian = 0.0;
  double beam_segmentation_factor = 0.0;
  double penalty_parameter =
      this->params()->beam_to_solid_surface_meshtying_params()->get_penalty_parameter();

  // Calculate the mesh tying forces.
  // Loop over segments.
  const unsigned int n_segments = this->line_to_3D_segments_.size();
  for (unsigned int i_segment = 0; i_segment < n_segments; i_segment++)
  {
    // Factor to account for the integration segment length.
    beam_segmentation_factor = 0.5 * this->line_to_3D_segments_[i_segment].get_segment_length();

    // Gauss point loop.
    const unsigned int n_gp = this->line_to_3D_segments_[i_segment].get_projection_points().size();
    for (unsigned int i_gp = 0; i_gp < n_gp; i_gp++)
    {
      // Get the current Gauss point.
      const GeometryPair::ProjectionPoint1DTo3D<double>& projected_gauss_point =
          this->line_to_3D_segments_[i_segment].get_projection_points()[i_gp];

      // Get the Jacobian in the reference configuration.
      GeometryPair::evaluate_position_derivative1<Beam>(
          projected_gauss_point.get_eta(), this->ele1posref_, dr_beam_ref);

      // Jacobian including the segment length.
      segment_jacobian = dr_beam_ref.norm2() * beam_segmentation_factor;

      // Calculate the force in this Gauss point. The sign of the force calculated here is the one
      // that acts on the beam.
      coupling_vector = this->evaluate_coupling(projected_gauss_point);
      force = coupling_vector;
      force.scale(penalty_parameter);

      // The force vector is in R3, we need to calculate the equivalent nodal forces on the element
      // dof. This is done with the virtual work equation $F \delta r = f \delta q$.
      for (unsigned int i_dof = 0; i_dof < Beam::n_dof_; i_dof++)
        for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
          force_pair(i_dof) += force(i_dir) * coupling_vector(i_dir).dx(i_dof) *
                               projected_gauss_point.get_gauss_weight() * segment_jacobian;
      for (unsigned int i_dof = 0; i_dof < Surface::n_dof_; i_dof++)
        for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
          force_pair(i_dof + Beam::n_dof_) +=
              force(i_dir) * coupling_vector(i_dir).dx(i_dof + Beam::n_dof_) *
              projected_gauss_point.get_gauss_weight() * segment_jacobian;
    }
  }

  // Get the pair GIDs.
  // Note we get the full patch GIDs here but we only use the ones for the beam and the surface, not
  // the ones for the averaged normals
  const auto pair_gid = get_beam_to_surface_pair_gid_combined<Beam>(
      *discret, *this->element1(), *this->face_element_);

  // If given, assemble force terms into the global vector.
  if (force_vector != nullptr)
  {
    const auto force_pair_double = Core::FADUtils::cast_to_double(force_pair);
    force_vector->sum_into_global_values(
        Beam::n_dof_ + Surface::n_dof_, pair_gid.data(), force_pair_double.data());
  }

  // If given, assemble force terms into the global stiffness matrix.
  if (stiffness_matrix != nullptr)
    for (unsigned int i_dof = 0; i_dof < Beam::n_dof_ + Surface::n_dof_; i_dof++)
      for (unsigned int j_dof = 0; j_dof < Beam::n_dof_ + Surface::n_dof_; j_dof++)
        stiffness_matrix->fe_assemble(Core::FADUtils::cast_to_double(force_pair(i_dof).dx(j_dof)),
            pair_gid[i_dof], pair_gid[j_dof]);
}


/**
 * Explicit template initialization of template class.
 */
namespace BeamInteraction
{
  using namespace GeometryPair;

  template class BeamToSolidSurfaceMeshtyingPairGaussPoint<t_hermite, t_tri3>;
  template class BeamToSolidSurfaceMeshtyingPairGaussPoint<t_hermite, t_tri6>;
  template class BeamToSolidSurfaceMeshtyingPairGaussPoint<t_hermite, t_quad4>;
  template class BeamToSolidSurfaceMeshtyingPairGaussPoint<t_hermite, t_quad8>;
  template class BeamToSolidSurfaceMeshtyingPairGaussPoint<t_hermite, t_quad9>;
  template class BeamToSolidSurfaceMeshtyingPairGaussPoint<t_hermite, t_nurbs9>;
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE
