// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_2d-3d_full.hpp"

#include "4C_beam3_reissner.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_params.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_beaminteraction_geometry_pair_access_traits.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_to_3D_evaluation_data.hpp"
#include "4C_geometry_pair_line_to_volume_gauss_point_projection_cross_section.hpp"
#include "4C_geometry_pair_utility_classes.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename Beam, typename Solid>
void BeamInteraction::BeamToSolidVolumeMeshtyingPair2D3DFull<Beam, Solid>::pre_evaluate()
{
  // Call pre_evaluate on the geometry Pair.
  if (!this->meshtying_is_evaluated_)
  {
    this->cast_geometry_pair()->pre_evaluate(this->ele1posref_, this->ele2posref_,
        this->line_to_3D_segments_, &triad_interpolation_scheme_ref_);
  }
}

/**
 *
 */
template <typename Beam, typename Solid>
void BeamInteraction::BeamToSolidVolumeMeshtyingPair2D3DFull<Beam, Solid>::evaluate_and_assemble(
    const std::shared_ptr<const Core::FE::Discretization>& discret,
    const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
    const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector)
{
  // Call Evaluate on the geometry Pair. Only do this once for mesh tying.
  if (!this->meshtying_is_evaluated_)
  {
    this->cast_geometry_pair()->evaluate(
        this->ele1posref_, this->ele2posref_, this->line_to_3D_segments_);
    this->meshtying_is_evaluated_ = true;
  }

  // If there are no segments, this pair has no contribution. Also there can be no more than one
  // segment.
  if (this->line_to_3D_segments_.size() == 0)
    return;
  else if (this->line_to_3D_segments_.size() > 1)
    FOUR_C_THROW(
        "There can be a maximum of one segment for coupling pairs that couple on the beam "
        "surface!");

  // Check that the beam element is a Simo--Reissner beam.
  auto beam_ele = dynamic_cast<const Discret::Elements::Beam3r*>(this->element1());
  if (beam_ele == nullptr)
    FOUR_C_THROW("GetBeamTriadInterpolationScheme is only implemented for SR beams.");

  // Get the vector with the projection points for this pair.
  const std::vector<GeometryPair::ProjectionPoint1DTo3D<double>>& projection_points =
      this->line_to_3D_segments_[0].get_projection_points();

  // If there are no projection points, return no contact status.
  if (projection_points.size() == 0) return;

  // Set the FAD variables for the beam and solid DOFs.
  auto set_q_fad = [](const auto& q_original, auto& q_fad, unsigned int fad_offset = 0)
  {
    for (unsigned int i_dof = 0; i_dof < q_original.num_rows(); i_dof++)
      q_fad(i_dof) = Core::FADUtils::HigherOrderFadValue<scalar_type_pair>::apply(
          n_dof_fad_, fad_offset + i_dof, Core::FADUtils::cast_to_double(q_original(i_dof)));
  };
  auto q_beam =
      GeometryPair::InitializeElementData<Beam, scalar_type_pair>::initialize(this->element1());
  auto q_solid =
      GeometryPair::InitializeElementData<Solid, scalar_type_pair>::initialize(this->element2());
  set_q_fad(this->ele1pos_.element_position_, q_beam.element_position_);
  set_q_fad(this->ele2pos_.element_position_, q_solid.element_position_, Beam::n_dof_);

  // Shape function data for Lagrange functions for rotations
  auto q_rot =
      GeometryPair::InitializeElementData<GeometryPair::t_line3, scalar_type_pair>::initialize(
          nullptr);

  // Initialize pair wise vectors and matrices.
  Core::LinAlg::Matrix<n_dof_pair_, 1, double> force_pair(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<n_dof_pair_, n_dof_pair_, double> stiff_pair(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<n_dof_pair_, 1, scalar_type_pair> force_pair_local;
  Core::LinAlg::Matrix<n_dof_pair_, 3, double> d_force_d_psi;
  Core::LinAlg::Matrix<n_dof_pair_, n_dof_rot_, double> local_stiffness_rot;

  // Shape function matrices.
  Core::LinAlg::Matrix<3, Solid::n_dof_, scalar_type_pair> N;
  Core::LinAlg::Matrix<3, Beam::n_dof_, scalar_type_pair> H;
  Core::LinAlg::Matrix<3, n_dof_rot_, scalar_type_pair> L;
  std::vector<Core::LinAlg::Matrix<3, 3, double>> I_tilde_vector;
  Core::LinAlg::Matrix<3, n_dof_rot_, double> I_tilde;

  // Initialize vector and matrix variables for the Gauss integration.
  Core::LinAlg::Matrix<3, 1, double> dr_beam_ref;
  Core::LinAlg::Matrix<3, 1, scalar_type_pair> cross_section_vector_ref;
  Core::LinAlg::Matrix<3, 1, scalar_type_pair> cross_section_vector_current;
  Core::LinAlg::Matrix<3, 1, scalar_type_pair> pos_beam;
  Core::LinAlg::Matrix<3, 1, scalar_type_pair> pos_solid;
  Core::LinAlg::Matrix<4, 1, double> quaternion_double;
  Core::LinAlg::Matrix<3, 1, double> rotation_vector_double;
  Core::LinAlg::Matrix<4, 1, scalar_type_pair> quaternion_fad;
  Core::LinAlg::Matrix<3, 1, scalar_type_pair> rotation_vector_fad;
  Core::LinAlg::Matrix<3, 3, scalar_type_pair> triad_fad;
  Core::LinAlg::Matrix<3, 3, double> T_beam_double;
  Core::LinAlg::Matrix<3, n_dof_rot_, double> T_times_I_tilde;
  Core::LinAlg::Matrix<Solid::n_dof_, 1, scalar_type_pair> temp_solid_force;
  Core::LinAlg::Matrix<Beam::n_dof_, 1, scalar_type_pair> temp_beam_force;
  Core::LinAlg::Matrix<n_dof_rot_, 1, scalar_type_pair> temp_beam_force_rot;

  // Initialize scalar variables.
  double eta = 1e10;
  double eta_last_gauss_point = 1e10;
  double beam_jacobian = 0.0;
  const double penalty_parameter =
      this->params()->beam_to_solid_volume_meshtying_params()->get_penalty_parameter();
  const double radius = (dynamic_cast<const Discret::Elements::Beam3Base*>(this->element1()))
                            ->get_circular_cross_section_radius_for_interactions();

  // Calculate the mesh tying forces.
  // Loop over segments.
  for (unsigned int i_integration_point = 0; i_integration_point < projection_points.size();
      i_integration_point++)
  {
    // Get the current Gauss point.
    const GeometryPair::ProjectionPoint1DTo3D<double>& projected_gauss_point =
        projection_points[i_integration_point];
    eta = projected_gauss_point.get_eta();

    // Evaluate all beam specific terms. This only has to be done if the Gauss point position on the
    // beam has changed compared to the last Gauss point.
    if (std::abs(eta - eta_last_gauss_point) > 1e-10)
    {
      GeometryPair::evaluate_position_derivative1<Beam>(eta, this->ele1posref_, dr_beam_ref);
      beam_jacobian = dr_beam_ref.norm2();

      GeometryPair::evaluate_shape_function_matrix<Beam>(H, eta, q_beam.shape_function_data_);
      GeometryPair::evaluate_position<Beam>(eta, q_beam, pos_beam);

      GeometryPair::evaluate_shape_function_matrix<GeometryPair::t_line3>(
          L, eta, q_rot.shape_function_data_);

      triad_interpolation_scheme_.get_nodal_generalized_rotation_interpolation_matrices_at_xi(
          I_tilde_vector, eta);
      for (unsigned int i_node = 0; i_node < n_nodes_rot_; i_node++)
        for (unsigned int i_dim_0 = 0; i_dim_0 < 3; i_dim_0++)
          for (unsigned int i_dim_1 = 0; i_dim_1 < 3; i_dim_1++)
            I_tilde(i_dim_0, i_node * 3 + i_dim_1) = I_tilde_vector[i_node](i_dim_0, i_dim_1);

      // Get the rotation vector at this Gauss point.
      triad_interpolation_scheme_.get_interpolated_quaternion_at_xi(quaternion_double, eta);
      Core::LargeRotations::quaterniontoangle(quaternion_double, rotation_vector_double);
      T_beam_double = Core::LargeRotations::tmatrix(rotation_vector_double);
      set_q_fad(rotation_vector_double, rotation_vector_fad, Beam::n_dof_ + Solid::n_dof_);
      Core::LargeRotations::angletoquaternion(rotation_vector_fad, quaternion_fad);
      Core::LargeRotations::quaterniontotriad(quaternion_fad, triad_fad);
    }

    // Get the shape function matrices.
    GeometryPair::evaluate_shape_function_matrix<Solid>(
        N, projected_gauss_point.get_xi(), q_solid.shape_function_data_);
    GeometryPair::evaluate_position<Solid>(projected_gauss_point.get_xi(), q_solid, pos_solid);

    // Get the cross section vector.
    cross_section_vector_ref(0) = 0.0;
    cross_section_vector_ref(1) = projected_gauss_point.get_eta_cross_section()(0);
    cross_section_vector_ref(2) = projected_gauss_point.get_eta_cross_section()(1);
    cross_section_vector_current.multiply(triad_fad, cross_section_vector_ref);

    // Reset the local force vector for this Gauss point.
    force_pair_local.put_scalar(0.0);

    // Numerical integration factor for this Gauss point.
    const double integration_factor =
        projected_gauss_point.get_gauss_weight() * beam_jacobian * radius * M_PI;

    // The following calculations are based on Steinbrecher, Popp, Meier: "Consistent coupling of
    // positions and rotations for embedding 1D Cosserat beams into 3D solid volumes", eq. 97-98. Be
    // aware, that there is a typo in eq. 98 where the derivative is taken with respect to the
    // rotation angle and not the rotational DOFs.
    auto r_diff = pos_beam;
    r_diff += cross_section_vector_current;
    r_diff -= pos_solid;

    // Evaluate the force on the solid DOFs.
    temp_solid_force.multiply_tn(N, r_diff);
    temp_solid_force.scale(-1.0);
    for (unsigned int i_dof = 0; i_dof < Solid::n_dof_; i_dof++)
      force_pair_local(i_dof + Beam::n_dof_) += temp_solid_force(i_dof);

    // Evaluate the force on the positional beam DOFs.
    temp_beam_force.multiply_tn(H, r_diff);
    for (unsigned int i_dof = 0; i_dof < Beam::n_dof_; i_dof++)
      force_pair_local(i_dof) += temp_beam_force(i_dof);

    // Evaluate the force on the rotational beam DOFs.
    // In comparison to the mentioned paper, the relative cross section vector is also contained
    // here, but it cancels out in the cross product with itself.
    Core::LinAlg::Matrix<3, 1, scalar_type_pair> temp_beam_rot_cross;
    temp_beam_rot_cross.cross_product(cross_section_vector_current, r_diff);
    temp_beam_force_rot.multiply_tn(L, temp_beam_rot_cross);
    for (unsigned int i_dof = 0; i_dof < n_dof_rot_; i_dof++)
      force_pair_local(i_dof + Beam::n_dof_ + Solid::n_dof_) += temp_beam_force_rot(i_dof);

    // Add to pair force contributions.
    force_pair_local.scale(integration_factor * penalty_parameter);
    force_pair += Core::FADUtils::cast_to_double(force_pair_local);

    // The rotational stiffness contributions have to be handled separately due to the non-additive
    // nature of the rotational DOFs.
    for (unsigned int i_dof = 0; i_dof < n_dof_pair_; i_dof++)
      for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
        d_force_d_psi(i_dof, i_dir) =
            force_pair_local(i_dof).dx(Beam::n_dof_ + Solid::n_dof_ + i_dir);
    T_times_I_tilde.multiply(T_beam_double, I_tilde);
    local_stiffness_rot.multiply_nn(d_force_d_psi, T_times_I_tilde);

    // Add the full stiffness contribution from this Gauss point.
    for (unsigned int i_dof = 0; i_dof < n_dof_pair_; i_dof++)
    {
      for (unsigned int j_dof = 0; j_dof < n_dof_pair_; j_dof++)
      {
        if (j_dof < Beam::n_dof_ + Solid::n_dof_)
          stiff_pair(i_dof, j_dof) += force_pair_local(i_dof).dx(j_dof);
        else
          stiff_pair(i_dof, j_dof) +=
              local_stiffness_rot(i_dof, j_dof - Beam::n_dof_ - Solid::n_dof_);
      }
    }

    // Set the eta value for this Gauss point.
    eta_last_gauss_point = eta;
  }

  // Get the GIDs of this pair.
  Core::LinAlg::Matrix<n_dof_pair_, 1, int> gid_pair;

  // Beam centerline GIDs.
  Core::LinAlg::Matrix<Beam::n_dof_, 1, int> beam_centerline_gid;
  Utils::get_element_centerline_gid_indices(*discret, this->element1(), beam_centerline_gid);
  for (unsigned int i_dof_beam = 0; i_dof_beam < Beam::n_dof_; i_dof_beam++)
    gid_pair(i_dof_beam) = beam_centerline_gid(i_dof_beam);

  // Solid GIDs.
  std::vector<int> lm, lmowner, lmstride;
  this->element2()->location_vector(*discret, lm, lmowner, lmstride);
  for (unsigned int i = 0; i < Solid::n_dof_; i++) gid_pair(i + Beam::n_dof_) = lm[i];

  // Beam rot GIDs.
  const auto rot_gid = Utils::get_element_rot_gid_indices(*discret, this->element1());
  for (unsigned int i = 0; i < n_dof_rot_; i++)
    gid_pair(i + Beam::n_dof_ + Solid::n_dof_) = rot_gid[i];

  // If given, assemble force terms into the global force vector.
  if (force_vector != nullptr)
    force_vector->sum_into_global_values(gid_pair.num_rows(), gid_pair.data(), force_pair.data());

  // If given, assemble force terms into the global stiffness matrix.
  if (stiffness_matrix != nullptr)
    for (unsigned int i_dof = 0; i_dof < n_dof_pair_; i_dof++)
      for (unsigned int j_dof = 0; j_dof < n_dof_pair_; j_dof++)
        stiffness_matrix->fe_assemble(stiff_pair(i_dof, j_dof), gid_pair(i_dof), gid_pair(j_dof));
}

/**
 *
 */
template <typename Beam, typename Solid>
void BeamInteraction::BeamToSolidVolumeMeshtyingPair2D3DFull<Beam, Solid>::reset_rotation_state(
    const Core::FE::Discretization& discret,
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& ia_discolnp)
{
  get_beam_triad_interpolation_scheme(discret, *ia_discolnp, this->element1(),
      triad_interpolation_scheme_, this->triad_interpolation_scheme_ref_);
}

/**
 *
 */
template <typename Beam, typename Solid>
void BeamInteraction::BeamToSolidVolumeMeshtyingPair2D3DFull<Beam, Solid>::get_triad_at_xi_double(
    const double xi, Core::LinAlg::Matrix<3, 3, double>& triad, const bool reference) const
{
  if (reference)
  {
    this->triad_interpolation_scheme_ref_.get_interpolated_triad_at_xi(triad, xi);
  }
  else
  {
    this->triad_interpolation_scheme_.get_interpolated_triad_at_xi(triad, xi);
  }
}


/**
 * Explicit template initialization of template class.
 */
namespace BeamInteraction
{
  using namespace GeometryPair;

  template class BeamToSolidVolumeMeshtyingPair2D3DFull<t_hermite, t_hex8>;
  template class BeamToSolidVolumeMeshtyingPair2D3DFull<t_hermite, t_hex20>;
  template class BeamToSolidVolumeMeshtyingPair2D3DFull<t_hermite, t_hex27>;
  template class BeamToSolidVolumeMeshtyingPair2D3DFull<t_hermite, t_tet4>;
  template class BeamToSolidVolumeMeshtyingPair2D3DFull<t_hermite, t_tet10>;
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE
