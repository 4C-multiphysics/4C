// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_2d-3d_plane.hpp"

#include "4C_beam3_triad_interpolation_local_rotation_vectors.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_params.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_beaminteraction_geometry_pair_access_traits.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_to_3D_evaluation_data.hpp"
#include "4C_geometry_pair_line_to_volume_gauss_point_projection_cross_section.hpp"
#include "4C_geometry_pair_utility_classes.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename Beam, typename Solid>
void BeamInteraction::BeamToSolidVolumeMeshtyingPair2D3DPlane<Beam, Solid>::pre_evaluate()
{
  // Call pre_evaluate on the geometry Pair.
  if (!this->meshtying_is_evaluated_)
  {
    GEOMETRYPAIR::ElementData<Beam, double> beam_coupling_ref;
    GEOMETRYPAIR::ElementData<Solid, double> solid_coupling_ref;
    this->get_coupling_reference_position(beam_coupling_ref, solid_coupling_ref);
    this->cast_geometry_pair()->pre_evaluate(
        beam_coupling_ref, solid_coupling_ref, this->line_to_3D_segments_);
  }
}

/**
 *
 */
template <typename Beam, typename Solid>
bool BeamInteraction::BeamToSolidVolumeMeshtyingPair2D3DPlane<Beam, Solid>::evaluate(
    Core::LinAlg::SerialDenseVector* forcevec1, Core::LinAlg::SerialDenseVector* forcevec2,
    Core::LinAlg::SerialDenseMatrix* stiffmat11, Core::LinAlg::SerialDenseMatrix* stiffmat12,
    Core::LinAlg::SerialDenseMatrix* stiffmat21, Core::LinAlg::SerialDenseMatrix* stiffmat22)
{
  // Call Evaluate on the geometry Pair. Only do this once for meshtying.
  if (!this->meshtying_is_evaluated_)
  {
    GEOMETRYPAIR::ElementData<Beam, double> beam_coupling_ref;
    GEOMETRYPAIR::ElementData<Solid, double> solid_coupling_ref;
    this->get_coupling_reference_position(beam_coupling_ref, solid_coupling_ref);
    this->cast_geometry_pair()->evaluate(
        beam_coupling_ref, solid_coupling_ref, this->line_to_3D_segments_);
    this->meshtying_is_evaluated_ = true;
  }

  // If there are no segments, this pair has no contribution. Also there can be no more than one
  // segment.
  if (this->line_to_3D_segments_.size() == 0)
    return false;
  else if (this->line_to_3D_segments_.size() > 1)
    FOUR_C_THROW(
        "There can be a maximum of one segment for coupling pairs that couple on the beam "
        "surface!");

  // Get the vector with the projection points for this pair.
  const std::vector<GEOMETRYPAIR::ProjectionPoint1DTo3D<double>>& projection_points =
      this->line_to_3D_segments_[0].get_projection_points();

  // If there are no projection points, return no contact status.
  if (projection_points.size() == 0) return false;

  // Initialize variables for position and force vectors.
  Core::LinAlg::Matrix<3, 1, double> dr_beam_ref;
  Core::LinAlg::Matrix<3, 1, scalar_type> r_beam;
  Core::LinAlg::Matrix<3, 3, scalar_type> triad;
  Core::LinAlg::Matrix<3, 1, scalar_type> r_cross_section_ref;
  Core::LinAlg::Matrix<3, 1, scalar_type> r_cross_section;
  Core::LinAlg::Matrix<3, 1, scalar_type> r_solid;
  Core::LinAlg::Matrix<3, 1, scalar_type> force;
  Core::LinAlg::Matrix<Beam::n_dof_, 1, scalar_type> force_element_1(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Solid::n_dof_, 1, scalar_type> force_element_2(
      Core::LinAlg::Initialization::zero);

  // Initialize scalar variables.
  double beam_jacobian;
  double penalty_parameter =
      this->params()->beam_to_solid_volume_meshtying_params()->get_penalty_parameter();
  const double radius = (dynamic_cast<const Discret::Elements::Beam3Base*>(this->element1()))
                            ->get_circular_cross_section_radius_for_interactions();

  // Calculate the meshtying forces.
  // Loop over segments.
  for (unsigned int i_integration_point = 0; i_integration_point < projection_points.size();
      i_integration_point++)
  {
    // Get the current Gauss point.
    const GEOMETRYPAIR::ProjectionPoint1DTo3D<double>& projected_gauss_point =
        projection_points[i_integration_point];

    // Get the jacobian in the reference configuration.
    GEOMETRYPAIR::evaluate_position_derivative1<Beam>(
        projected_gauss_point.get_eta(), this->ele1posref_, dr_beam_ref);
    beam_jacobian = dr_beam_ref.norm2();

    // Get the current positions on beam and solid.
    GEOMETRYPAIR::evaluate_position<Beam>(projected_gauss_point.get_eta(), this->ele1pos_, r_beam);
    GEOMETRYPAIR::evaluate_triad_at_plane_curve<Beam>(
        projected_gauss_point.get_eta(), this->ele1pos_, triad);
    r_cross_section_ref(0) = 0.0;
    r_cross_section_ref(1) = projected_gauss_point.get_eta_cross_section()(0);
    r_cross_section_ref(2) = projected_gauss_point.get_eta_cross_section()(1);
    r_cross_section.multiply(triad, r_cross_section_ref);
    r_beam += r_cross_section;
    GEOMETRYPAIR::evaluate_position<Solid>(projected_gauss_point.get_xi(), this->ele2pos_, r_solid);

    // Calculate the force in this Gauss point. The sign of the force calculated here is the one
    // that acts on the beam.
    force = r_solid;
    force -= r_beam;
    force.scale(penalty_parameter);

    const double integration_factor =
        projected_gauss_point.get_gauss_weight() * beam_jacobian * radius * M_PI;

    // The force vector is in R3, we need to calculate the equivalent nodal forces on the element
    // dof. This is done with the virtual work equation $F \delta r = f \delta q$.
    for (unsigned int i_dof = 0; i_dof < Beam::n_dof_; i_dof++)
      for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
        force_element_1(i_dof) += force(i_dir) * r_beam(i_dir).dx(i_dof) * integration_factor;
    for (unsigned int i_dof = 0; i_dof < Solid::n_dof_; i_dof++)
      for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
        force_element_2(i_dof) -=
            force(i_dir) * r_solid(i_dir).dx(i_dof + Beam::n_dof_) * integration_factor;
  }


  // Fill in the entries for the local matrices and vectors.
  {
    // Resize and initialize the return variables.
    if (forcevec1 != nullptr) forcevec1->size(Beam::n_dof_);
    if (forcevec2 != nullptr) forcevec2->size(Solid::n_dof_);
    if (stiffmat11 != nullptr) stiffmat11->shape(Beam::n_dof_, Beam::n_dof_);
    if (stiffmat12 != nullptr) stiffmat12->shape(Beam::n_dof_, Solid::n_dof_);
    if (stiffmat21 != nullptr) stiffmat21->shape(Solid::n_dof_, Beam::n_dof_);
    if (stiffmat22 != nullptr) stiffmat22->shape(Solid::n_dof_, Solid::n_dof_);

    if (forcevec1 != nullptr && forcevec2 != nullptr)
    {
      // $f_1$
      for (unsigned int i_dof = 0; i_dof < Beam::n_dof_; i_dof++)
        (*forcevec1)(i_dof) = Core::FADUtils::cast_to_double(force_element_1(i_dof));
      // $f_2$
      for (unsigned int i_dof = 0; i_dof < Solid::n_dof_; i_dof++)
        (*forcevec2)(i_dof) = Core::FADUtils::cast_to_double(force_element_2(i_dof));
    }

    if (stiffmat11 != nullptr && stiffmat12 != nullptr && stiffmat21 != nullptr &&
        stiffmat22 != nullptr)
    {
      // $k_{11}$
      for (unsigned int i_dof_1 = 0; i_dof_1 < Beam::n_dof_; i_dof_1++)
        for (unsigned int i_dof_2 = 0; i_dof_2 < Beam::n_dof_; i_dof_2++)
          (*stiffmat11)(i_dof_1, i_dof_2) =
              -Core::FADUtils::cast_to_double(force_element_1(i_dof_1).dx(i_dof_2));

      // $k_{12}, k_{21}$
      for (unsigned int i_dof_1 = 0; i_dof_1 < Beam::n_dof_; i_dof_1++)
      {
        for (unsigned int i_dof_2 = 0; i_dof_2 < Solid::n_dof_; i_dof_2++)
        {
          (*stiffmat12)(i_dof_1, i_dof_2) =
              -Core::FADUtils::cast_to_double(force_element_1(i_dof_1).dx(Beam::n_dof_ + i_dof_2));
          (*stiffmat21)(i_dof_2, i_dof_1) =
              -Core::FADUtils::cast_to_double(force_element_2(i_dof_2).dx(i_dof_1));
        }
      }

      // $k_{22}$
      for (unsigned int i_dof_1 = 0; i_dof_1 < Solid::n_dof_; i_dof_1++)
        for (unsigned int i_dof_2 = 0; i_dof_2 < Solid::n_dof_; i_dof_2++)
          (*stiffmat22)(i_dof_1, i_dof_2) =
              -Core::FADUtils::cast_to_double(force_element_2(i_dof_1).dx(Beam::n_dof_ + i_dof_2));
    }
  }

  // Return true as there are mesh tying contributions.
  return true;
}

/**
 *
 */
template <typename Beam, typename Solid>
void BeamInteraction::BeamToSolidVolumeMeshtyingPair2D3DPlane<Beam, Solid>::get_triad_at_xi_double(
    const double xi, Core::LinAlg::Matrix<3, 3, double>& triad, const bool reference) const
{
  if (reference)
  {
    GEOMETRYPAIR::ElementData<Beam, double> beam_coupling_ref;
    GEOMETRYPAIR::ElementData<Solid, double> dummy;
    this->get_coupling_reference_position(beam_coupling_ref, dummy);
    GEOMETRYPAIR::evaluate_triad_at_plane_curve<Beam>(xi, beam_coupling_ref, triad);
  }
  else
  {
    GEOMETRYPAIR::evaluate_triad_at_plane_curve<Beam>(
        xi, GEOMETRYPAIR::ElementDataToDouble<Beam>::to_double(this->ele1pos_), triad);
  }
}


/**
 * Explicit template initialization of template class.
 */
namespace BeamInteraction
{
  using namespace GEOMETRYPAIR;

  template class BeamToSolidVolumeMeshtyingPair2D3DPlane<t_hermite, t_hex8>;
  template class BeamToSolidVolumeMeshtyingPair2D3DPlane<t_hermite, t_hex20>;
  template class BeamToSolidVolumeMeshtyingPair2D3DPlane<t_hermite, t_hex27>;
  template class BeamToSolidVolumeMeshtyingPair2D3DPlane<t_hermite, t_tet4>;
  template class BeamToSolidVolumeMeshtyingPair2D3DPlane<t_hermite, t_tet10>;
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE
