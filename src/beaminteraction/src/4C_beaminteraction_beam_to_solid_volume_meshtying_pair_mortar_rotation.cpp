// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_mortar_rotation.hpp"

#include "4C_beam3_reissner.hpp"
#include "4C_beam3_triad_interpolation_local_rotation_vectors.hpp"
#include "4C_beaminteraction_beam_to_solid_mortar_manager.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_params.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_to_volume.hpp"
#include "4C_linalg_fevector.hpp"

FOUR_C_NAMESPACE_OPEN



/**
 *
 */
template <typename Beam, typename Solid, typename Mortar, typename MortarRot>
BeamInteraction::BeamToSolidVolumeMeshtyingPairMortarRotation<Beam, Solid, Mortar,
    MortarRot>::BeamToSolidVolumeMeshtyingPairMortarRotation()
    : BeamToSolidVolumeMeshtyingPairMortar<Beam, Solid, Mortar>()
{
  // Set the number of rotational mortar DOFs.
  this->n_mortar_rot_ = MortarRot::n_dof_;
}

/**
 *
 */
template <typename Beam, typename Solid, typename Mortar, typename MortarRot>
void BeamInteraction::BeamToSolidVolumeMeshtyingPairMortarRotation<Beam, Solid, Mortar,
    MortarRot>::evaluate_and_assemble_mortar_contributions(const Core::FE::Discretization& discret,
    const BeamToSolidMortarManager* mortar_manager,
    Core::LinAlg::SparseMatrix& global_constraint_lin_beam,
    Core::LinAlg::SparseMatrix& global_constraint_lin_solid,
    Core::LinAlg::SparseMatrix& global_force_beam_lin_lambda,
    Core::LinAlg::SparseMatrix& global_force_solid_lin_lambda,
    Core::LinAlg::FEVector<double>& global_constraint, Core::LinAlg::FEVector<double>& global_kappa,
    Core::LinAlg::SparseMatrix& global_kappa_lin_beam,
    Core::LinAlg::SparseMatrix& global_kappa_lin_solid,
    Core::LinAlg::FEVector<double>& global_lambda_active,
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector)
{
  // Call the base method.
  base_class::evaluate_and_assemble_mortar_contributions(discret, mortar_manager,
      global_constraint_lin_beam, global_constraint_lin_solid, global_force_beam_lin_lambda,
      global_force_solid_lin_lambda, global_constraint, global_kappa, global_kappa_lin_beam,
      global_kappa_lin_solid, global_lambda_active, displacement_vector);

  // If there are no intersection segments, return as no contact can occur.
  if (this->line_to_3D_segments_.size() == 0) return;

  // Get the beam triad interpolation schemes.
  LargeRotations::TriadInterpolationLocalRotationVectors<3, double> triad_interpolation_scheme;
  LargeRotations::TriadInterpolationLocalRotationVectors<3, double> ref_triad_interpolation_scheme;
  get_beam_triad_interpolation_scheme(discret, *displacement_vector, this->element1(),
      triad_interpolation_scheme, ref_triad_interpolation_scheme);

  // Set the FAD variables for the solid DOFs. For the terms calculated here we only need first
  // order derivatives.
  auto q_solid =
      GeometryPair::InitializeElementData<Solid, scalar_type_rot_1st>::initialize(this->element2());
  for (unsigned int i_solid = 0; i_solid < Solid::n_dof_; i_solid++)
    q_solid.element_position_(i_solid) =
        Core::FADUtils::HigherOrderFadValue<scalar_type_rot_1st>::apply(3 + Solid::n_dof_,
            3 + i_solid, Core::FADUtils::cast_to_double(this->ele2pos_.element_position_(i_solid)));

  // Initialize local matrices.
  Core::LinAlg::Matrix<MortarRot::n_dof_, 1, double> local_g(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<MortarRot::n_dof_, n_dof_rot_, double> local_G_B(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<MortarRot::n_dof_, Solid::n_dof_, double> local_G_S(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<n_dof_rot_, MortarRot::n_dof_, double> local_FB_L(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Solid::n_dof_, MortarRot::n_dof_, double> local_FS_L(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<MortarRot::n_dof_, 1, double> local_kappa(
      Core::LinAlg::Initialization::zero);

  const auto rot_coupling_type =
      this->params()->beam_to_solid_volume_meshtying_params()->get_rotational_coupling_type();
  if (rot_coupling_type == Inpar::BeamToSolid::BeamToSolidRotationCoupling::fix_triad_2d)
  {
    // In the case of "fix_triad_2d" we couple both, the ey and ez direction to the beam. Therefore,
    // we have to evaluate the coupling terms w.r.t both of those coupling types.
    evaluate_rotational_coupling_terms(
        Inpar::BeamToSolid::BeamToSolidRotationCoupling::deformation_gradient_y_2d, q_solid,
        triad_interpolation_scheme, ref_triad_interpolation_scheme, local_g, local_G_B, local_G_S,
        local_FB_L, local_FS_L, local_kappa);
    evaluate_rotational_coupling_terms(
        Inpar::BeamToSolid::BeamToSolidRotationCoupling::deformation_gradient_z_2d, q_solid,
        triad_interpolation_scheme, ref_triad_interpolation_scheme, local_g, local_G_B, local_G_S,
        local_FB_L, local_FS_L, local_kappa);
  }
  else
    evaluate_rotational_coupling_terms(rot_coupling_type, q_solid, triad_interpolation_scheme,
        ref_triad_interpolation_scheme, local_g, local_G_B, local_G_S, local_FB_L, local_FS_L,
        local_kappa);

  // Get the GIDs of the solid and beam.
  const auto rot_dof_gid = Utils::get_element_rot_gid_indices(discret, this->element1());
  std::vector<int> gid_solid, lmowner, lmstride;
  this->element2()->location_vector(discret, gid_solid, lmowner, lmstride);

  // Get the Lagrange multiplier GIDs.
  const auto& [_, lambda_gid_rot] = mortar_manager->location_vector(*this);

  // Assemble into the global vectors
  global_constraint.sum_into_global_values(
      lambda_gid_rot.size(), lambda_gid_rot.data(), local_g.data());
  global_kappa.sum_into_global_values(
      lambda_gid_rot.size(), lambda_gid_rot.data(), local_kappa.data());
  local_kappa.put_scalar(1.0);
  global_lambda_active.sum_into_global_values(
      lambda_gid_rot.size(), lambda_gid_rot.data(), local_kappa.data());

  // Assemble into global matrices.
  for (unsigned int i_dof_lambda = 0; i_dof_lambda < MortarRot::n_dof_; i_dof_lambda++)
  {
    for (unsigned int i_dof_rot = 0; i_dof_rot < n_dof_rot_; i_dof_rot++)
    {
      global_constraint_lin_beam.fe_assemble(
          local_G_B(i_dof_lambda, i_dof_rot), lambda_gid_rot[i_dof_lambda], rot_dof_gid[i_dof_rot]);
      global_force_beam_lin_lambda.fe_assemble(local_FB_L(i_dof_rot, i_dof_lambda),
          rot_dof_gid[i_dof_rot], lambda_gid_rot[i_dof_lambda]);
    }
    for (unsigned int i_dof_solid = 0; i_dof_solid < Solid::n_dof_; i_dof_solid++)
    {
      global_constraint_lin_solid.fe_assemble(local_G_S(i_dof_lambda, i_dof_solid),
          lambda_gid_rot[i_dof_lambda], gid_solid[i_dof_solid]);
      global_force_solid_lin_lambda.fe_assemble(local_FS_L(i_dof_solid, i_dof_lambda),
          gid_solid[i_dof_solid], lambda_gid_rot[i_dof_lambda]);
    }
  }
}

/**
 *
 */
template <typename Beam, typename Solid, typename Mortar, typename MortarRot>
void BeamInteraction::BeamToSolidVolumeMeshtyingPairMortarRotation<Beam, Solid, Mortar,
    MortarRot>::evaluate_rotational_coupling_terms(  //
    const Inpar::BeamToSolid::BeamToSolidRotationCoupling& rot_coupling_type,
    const GeometryPair::ElementData<Solid, scalar_type_rot_1st>& q_solid,
    const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&
        triad_interpolation_scheme,
    const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&
        ref_triad_interpolation_scheme,
    Core::LinAlg::Matrix<MortarRot::n_dof_, 1, double>& local_g,
    Core::LinAlg::Matrix<MortarRot::n_dof_, n_dof_rot_, double>& local_G_B,
    Core::LinAlg::Matrix<MortarRot::n_dof_, Solid::n_dof_, double>& local_G_S,
    Core::LinAlg::Matrix<n_dof_rot_, MortarRot::n_dof_, double>& local_FB_L,
    Core::LinAlg::Matrix<Solid::n_dof_, MortarRot::n_dof_, double>& local_FS_L,
    Core::LinAlg::Matrix<MortarRot::n_dof_, 1, double>& local_kappa) const
{
  // Initialize variables.
  Core::LinAlg::Matrix<3, 1, double> dr_beam_ref;
  Core::LinAlg::Matrix<4, 1, double> quaternion_beam_double;
  Core::LinAlg::Matrix<3, 1, double> psi_beam_double;
  Core::LinAlg::Matrix<3, 1, scalar_type_rot_1st> psi_beam;
  Core::LinAlg::Matrix<3, 1, scalar_type_rot_1st> psi_solid;
  Core::LinAlg::Matrix<3, 1, scalar_type_rot_1st> psi_rel;
  Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_beam;
  Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_beam_inv;
  Core::LinAlg::Matrix<4, 1, double> quaternion_beam_ref;
  Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_solid;
  Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_rel;
  Core::LinAlg::Matrix<3, 3, double> T_beam;
  Core::LinAlg::Matrix<3, 3, double> T_solid;
  Core::LinAlg::Matrix<3, 3, double> T_solid_inv;
  Core::LinAlg::Matrix<3, 3, double> T_rel;

  Core::LinAlg::Matrix<MortarRot::n_nodes_, 1, double> lambda_shape_functions;
  Core::LinAlg::Matrix<3, MortarRot::n_dof_, double> lambda_shape_functions_full(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::SerialDenseVector L_i(3);
  Core::LinAlg::Matrix<3, n_dof_rot_, double> L_full(Core::LinAlg::Initialization::zero);
  std::vector<Core::LinAlg::Matrix<3, 3, double>> I_beam_tilde;
  Core::LinAlg::Matrix<3, n_dof_rot_, double> I_beam_tilde_full;
  Core::LinAlg::Matrix<3, n_dof_rot_, double> T_beam_times_I_beam_tilde_full;
  Core::LinAlg::Matrix<3, MortarRot::n_dof_, double> T_rel_tr_times_lambda_shape;
  Core::LinAlg::Matrix<3, MortarRot::n_dof_, double> T_solid_mtr_times_T_rel_tr_times_lambda_shape;
  Core::LinAlg::Matrix<n_dof_rot_, MortarRot::n_dof_, double> d_fb_d_lambda_gp;
  Core::LinAlg::Matrix<Solid::n_dof_, MortarRot::n_dof_, double> d_fs_d_lambda_gp;
  Core::LinAlg::Matrix<MortarRot::n_dof_, 1, scalar_type_rot_1st> g_gp;
  Core::LinAlg::Matrix<3, Solid::n_dof_, double> d_psi_solid_d_q_solid;
  Core::LinAlg::Matrix<MortarRot::n_dof_, 3, double> d_g_d_psi_beam;
  Core::LinAlg::Matrix<MortarRot::n_dof_, n_dof_rot_, double> d_g_d_psi_beam_times_T_beam_I;
  Core::LinAlg::Matrix<MortarRot::n_dof_, Solid::n_dof_, double> d_g_d_q_solid;

  // Initialize scalar variables.
  double segment_jacobian = 0.0;
  double beam_segmentation_factor = 0.0;

  // Calculate the meshtying forces.
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

      // Get the jacobian in the reference configuration.
      GeometryPair::evaluate_position_derivative1<Beam>(
          projected_gauss_point.get_eta(), this->ele1posref_, dr_beam_ref);

      // Jacobian including the segment length.
      segment_jacobian = dr_beam_ref.norm2() * beam_segmentation_factor;

      // Calculate the rotation vector of this cross section.
      triad_interpolation_scheme.get_interpolated_quaternion_at_xi(
          quaternion_beam_double, projected_gauss_point.get_eta());
      Core::LargeRotations::quaterniontoangle(quaternion_beam_double, psi_beam_double);
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        psi_beam(i_dim) = Core::FADUtils::HigherOrderFadValue<scalar_type_rot_1st>::apply(
            3 + Solid::n_dof_, i_dim, psi_beam_double(i_dim));
      Core::LargeRotations::angletoquaternion(psi_beam, quaternion_beam);
      quaternion_beam_inv = Core::LargeRotations::inversequaternion(quaternion_beam);

      // Get the solid rotation vector.
      ref_triad_interpolation_scheme.get_interpolated_quaternion_at_xi(
          quaternion_beam_ref, projected_gauss_point.get_eta());
      get_solid_rotation_vector<Solid>(rot_coupling_type, projected_gauss_point.get_xi(),
          this->ele2posref_, q_solid, quaternion_beam_ref, psi_solid);
      Core::LargeRotations::angletoquaternion(psi_solid, quaternion_solid);

      // Calculate the relative rotation vector.
      Core::LargeRotations::quaternionproduct(
          quaternion_beam_inv, quaternion_solid, quaternion_rel);
      Core::LargeRotations::quaterniontoangle(quaternion_rel, psi_rel);

      // Calculate the transformation matrices.
      T_rel = Core::LargeRotations::tmatrix(Core::FADUtils::cast_to_double(psi_rel));
      T_beam = Core::LargeRotations::tmatrix(Core::FADUtils::cast_to_double(psi_beam));
      T_solid = Core::LargeRotations::tmatrix(Core::FADUtils::cast_to_double(psi_solid));
      T_solid_inv = T_solid;
      Core::LinAlg::inverse(T_solid_inv);

      // Evaluate shape functions.
      GeometryPair::EvaluateShapeFunction<MortarRot>::evaluate(
          lambda_shape_functions, projected_gauss_point.get_eta());
      for (unsigned int i_node = 0; i_node < MortarRot::n_nodes_; i_node++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          lambda_shape_functions_full(i_dim, 3 * i_node + i_dim) = lambda_shape_functions(i_node);

      Core::FE::shape_function_1d(L_i, projected_gauss_point.get_eta(), Core::FE::CellType::line3);
      for (unsigned int i_node = 0; i_node < 3; i_node++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          L_full(i_dim, 3 * i_node + i_dim) = L_i(i_node);

      triad_interpolation_scheme.get_nodal_generalized_rotation_interpolation_matrices_at_xi(
          I_beam_tilde, projected_gauss_point.get_eta());
      for (unsigned int i_node = 0; i_node < 3; i_node++)
        for (unsigned int i_dim_0 = 0; i_dim_0 < 3; i_dim_0++)
          for (unsigned int i_dim_1 = 0; i_dim_1 < 3; i_dim_1++)
            I_beam_tilde_full(i_dim_0, i_node * 3 + i_dim_1) =
                I_beam_tilde[i_node](i_dim_0, i_dim_1);

      // Solid angle derived w.r.t. the solid DOFs.
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        for (unsigned int i_solid = 0; i_solid < Solid::n_dof_; i_solid++)
          d_psi_solid_d_q_solid(i_dim, i_solid) = psi_solid(i_dim).dx(3 + i_solid);

      // Calculate the force terms derived w.r.t. the Lagrange multipliers.
      T_rel_tr_times_lambda_shape.multiply_tn(T_rel, lambda_shape_functions_full);
      d_fb_d_lambda_gp.multiply_tn(L_full, T_rel_tr_times_lambda_shape);
      d_fb_d_lambda_gp.scale(-1.0 * projected_gauss_point.get_gauss_weight() * segment_jacobian);

      T_solid_mtr_times_T_rel_tr_times_lambda_shape.multiply_tn(
          T_solid_inv, T_rel_tr_times_lambda_shape);
      d_fs_d_lambda_gp.multiply_tn(
          d_psi_solid_d_q_solid, T_solid_mtr_times_T_rel_tr_times_lambda_shape);
      d_fs_d_lambda_gp.scale(projected_gauss_point.get_gauss_weight() * segment_jacobian);

      // Constraint vector.
      g_gp.put_scalar(0.0);
      for (unsigned int i_row = 0; i_row < MortarRot::n_dof_; i_row++)
        for (unsigned int i_col = 0; i_col < 3; i_col++)
          g_gp(i_row) += lambda_shape_functions_full(i_col, i_row) * psi_rel(i_col);
      g_gp.scale(projected_gauss_point.get_gauss_weight() * segment_jacobian);

      // Derivatives of constraint vector.
      T_beam_times_I_beam_tilde_full.multiply(T_beam, I_beam_tilde_full);

      for (unsigned int i_lambda = 0; i_lambda < MortarRot::n_dof_; i_lambda++)
        for (unsigned int i_psi = 0; i_psi < 3; i_psi++)
          d_g_d_psi_beam(i_lambda, i_psi) = g_gp(i_lambda).dx(i_psi);
      d_g_d_psi_beam_times_T_beam_I.multiply(d_g_d_psi_beam, T_beam_times_I_beam_tilde_full);

      for (unsigned int i_lambda = 0; i_lambda < MortarRot::n_dof_; i_lambda++)
        for (unsigned int i_solid = 0; i_solid < Solid::n_dof_; i_solid++)
          d_g_d_q_solid(i_lambda, i_solid) = g_gp(i_lambda).dx(3 + i_solid);

      // Add to output matrices and vector.
      local_g += Core::FADUtils::cast_to_double(g_gp);
      local_G_B += d_g_d_psi_beam_times_T_beam_I;
      local_G_S += d_g_d_q_solid;
      local_FB_L += d_fb_d_lambda_gp;
      local_FS_L += d_fs_d_lambda_gp;

      // Calculate the scaling entries.
      for (unsigned int i_mortar_node = 0; i_mortar_node < MortarRot::n_nodes_; i_mortar_node++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          local_kappa(i_mortar_node * 3 + i_dim) += lambda_shape_functions(i_mortar_node) *
                                                    projected_gauss_point.get_gauss_weight() *
                                                    segment_jacobian;
    }
  }
}

/**
 *
 */
template <typename Beam, typename Solid, typename Mortar, typename MortarRot>
void BeamInteraction::BeamToSolidVolumeMeshtyingPairMortarRotation<Beam, Solid, Mortar,
    MortarRot>::evaluate_and_assemble(const Core::FE::Discretization& discret,
    const BeamToSolidMortarManager* mortar_manager,
    const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
    const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
    const Core::LinAlg::Vector<double>& global_lambda,
    const Core::LinAlg::Vector<double>& displacement_vector)
{
  // Call the base method.
  base_class::evaluate_and_assemble(
      discret, mortar_manager, force_vector, stiffness_matrix, global_lambda, displacement_vector);

  // If there are no intersection segments, return as no contact can occur.
  if (this->line_to_3D_segments_.size() == 0) return;

  // This pair only gives contributions to the stiffness matrix.
  if (stiffness_matrix == nullptr) return;

  // Get the beam triad interpolation schemes.
  LargeRotations::TriadInterpolationLocalRotationVectors<3, double> triad_interpolation_scheme;
  LargeRotations::TriadInterpolationLocalRotationVectors<3, double> ref_triad_interpolation_scheme;
  get_beam_triad_interpolation_scheme(discret, displacement_vector, this->element1(),
      triad_interpolation_scheme, ref_triad_interpolation_scheme);

  // Set the FAD variables for the solid DOFs. For the terms calculated here we only need first
  // order derivatives.
  auto q_solid =
      GeometryPair::InitializeElementData<Solid, scalar_type_rot_2nd>::initialize(this->element2());
  for (unsigned int i_solid = 0; i_solid < Solid::n_dof_; i_solid++)
    q_solid.element_position_(i_solid) =
        Core::FADUtils::HigherOrderFadValue<scalar_type_rot_2nd>::apply(3 + Solid::n_dof_,
            3 + i_solid, Core::FADUtils::cast_to_double(this->ele2pos_.element_position_(i_solid)));

  // Get the rotational Lagrange multipliers for this pair.
  const auto& [_, lambda_gid_rot] = mortar_manager->location_vector(*this);

  std::vector<double> lambda_rot_double = Core::FE::extract_values(global_lambda, lambda_gid_rot);
  Core::LinAlg::Matrix<MortarRot::n_dof_, 1, double> lambda_rot;
  for (unsigned int i_dof = 0; i_dof < MortarRot::n_dof_; i_dof++)
    lambda_rot(i_dof) = lambda_rot_double[i_dof];

  // Initialize local matrices.
  Core::LinAlg::Matrix<n_dof_rot_, n_dof_rot_, double> local_stiff_BB(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<n_dof_rot_, Solid::n_dof_, double> local_stiff_BS(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Solid::n_dof_, n_dof_rot_, double> local_stiff_SB(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Solid::n_dof_, Solid::n_dof_, double> local_stiff_SS(
      Core::LinAlg::Initialization::zero);

  const auto rot_coupling_type =
      this->params()->beam_to_solid_volume_meshtying_params()->get_rotational_coupling_type();
  if (rot_coupling_type == Inpar::BeamToSolid::BeamToSolidRotationCoupling::fix_triad_2d)
  {
    // In the case of "fix_triad_2d" we couple both, the ey and ez direction to the beam. Therefore,
    // we have to evaluate the coupling terms w.r.t both of those coupling types.
    evaluate_rotational_coupling_stiff_terms(
        Inpar::BeamToSolid::BeamToSolidRotationCoupling::deformation_gradient_y_2d, q_solid,
        lambda_rot, triad_interpolation_scheme, ref_triad_interpolation_scheme, local_stiff_BB,
        local_stiff_BS, local_stiff_SB, local_stiff_SS);
    evaluate_rotational_coupling_stiff_terms(
        Inpar::BeamToSolid::BeamToSolidRotationCoupling::deformation_gradient_z_2d, q_solid,
        lambda_rot, triad_interpolation_scheme, ref_triad_interpolation_scheme, local_stiff_BB,
        local_stiff_BS, local_stiff_SB, local_stiff_SS);
  }
  else
    evaluate_rotational_coupling_stiff_terms(rot_coupling_type, q_solid, lambda_rot,
        triad_interpolation_scheme, ref_triad_interpolation_scheme, local_stiff_BB, local_stiff_BS,
        local_stiff_SB, local_stiff_SS);

  // Get the GIDs of the solid and beam.
  const auto rot_dof_gid = Utils::get_element_rot_gid_indices(discret, this->element1());
  std::vector<int> gid_solid, lmowner, lmstride;
  this->element2()->location_vector(discret, gid_solid, lmowner, lmstride);

  // Assemble into global matrix.
  for (unsigned int i_dof_beam = 0; i_dof_beam < n_dof_rot_; i_dof_beam++)
  {
    for (unsigned int j_dof_beam = 0; j_dof_beam < n_dof_rot_; j_dof_beam++)
      stiffness_matrix->fe_assemble(
          local_stiff_BB(i_dof_beam, j_dof_beam), rot_dof_gid[i_dof_beam], rot_dof_gid[j_dof_beam]);
    for (unsigned int j_dof_solid = 0; j_dof_solid < Solid::n_dof_; j_dof_solid++)
      stiffness_matrix->fe_assemble(
          local_stiff_BS(i_dof_beam, j_dof_solid), rot_dof_gid[i_dof_beam], gid_solid[j_dof_solid]);
  }
  for (unsigned int i_dof_solid = 0; i_dof_solid < Solid::n_dof_; i_dof_solid++)
  {
    for (unsigned int j_dof_beam = 0; j_dof_beam < n_dof_rot_; j_dof_beam++)
      stiffness_matrix->fe_assemble(
          local_stiff_SB(i_dof_solid, j_dof_beam), gid_solid[i_dof_solid], rot_dof_gid[j_dof_beam]);
    for (unsigned int j_dof_solid = 0; j_dof_solid < Solid::n_dof_; j_dof_solid++)
      stiffness_matrix->fe_assemble(
          local_stiff_SS(i_dof_solid, j_dof_solid), gid_solid[i_dof_solid], gid_solid[j_dof_solid]);
  }
}

/**
 *
 */
template <typename Beam, typename Solid, typename Mortar, typename MortarRot>
void BeamInteraction::BeamToSolidVolumeMeshtyingPairMortarRotation<Beam, Solid, Mortar, MortarRot>::
    evaluate_rotational_coupling_stiff_terms(
        const Inpar::BeamToSolid::BeamToSolidRotationCoupling& rot_coupling_type,
        const GeometryPair::ElementData<Solid, scalar_type_rot_2nd>& q_solid,
        Core::LinAlg::Matrix<MortarRot::n_dof_, 1, double>& lambda_rot,
        const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&
            triad_interpolation_scheme,
        const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&
            ref_triad_interpolation_scheme,
        Core::LinAlg::Matrix<n_dof_rot_, n_dof_rot_, double>& local_stiff_BB,
        Core::LinAlg::Matrix<n_dof_rot_, Solid::n_dof_, double>& local_stiff_BS,
        Core::LinAlg::Matrix<Solid::n_dof_, n_dof_rot_, double>& local_stiff_SB,
        Core::LinAlg::Matrix<Solid::n_dof_, Solid::n_dof_, double>& local_stiff_SS) const
{
  // Initialize variables.
  Core::LinAlg::Matrix<3, 1, double> dr_beam_ref;
  Core::LinAlg::Matrix<4, 1, double> quaternion_beam_double;
  Core::LinAlg::Matrix<3, 1, double> psi_beam_double;
  Core::LinAlg::Matrix<3, 1, scalar_type_rot_1st> psi_beam;
  Core::LinAlg::Matrix<3, 1, scalar_type_rot_2nd> psi_solid;
  Core::LinAlg::Matrix<3, 1, scalar_type_rot_1st> psi_solid_val;
  Core::LinAlg::Matrix<3, 1, scalar_type_rot_1st> psi_rel;
  Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_beam;
  Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_beam_inv;
  Core::LinAlg::Matrix<4, 1, double> quaternion_beam_ref;
  Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_solid;
  Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_rel;
  Core::LinAlg::Matrix<3, 3, double> T_beam;
  Core::LinAlg::Matrix<3, 3, scalar_type_rot_1st> T_solid;
  Core::LinAlg::Matrix<3, 3, scalar_type_rot_1st> T_solid_inv;
  Core::LinAlg::Matrix<3, 3, scalar_type_rot_1st> T_rel;

  Core::LinAlg::Matrix<MortarRot::n_nodes_, 1, double> lambda_shape_functions;
  Core::LinAlg::Matrix<3, MortarRot::n_dof_, scalar_type_rot_1st> lambda_shape_functions_full(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::SerialDenseVector L_i(3);
  Core::LinAlg::Matrix<3, n_dof_rot_, scalar_type_rot_1st> L_full(
      Core::LinAlg::Initialization::zero);
  std::vector<Core::LinAlg::Matrix<3, 3, double>> I_beam_tilde;
  Core::LinAlg::Matrix<3, n_dof_rot_, double> I_beam_tilde_full;
  Core::LinAlg::Matrix<3, n_dof_rot_, double> T_beam_times_I_beam_tilde_full;
  Core::LinAlg::Matrix<3, MortarRot::n_dof_, scalar_type_rot_1st> T_rel_tr_times_lambda_shape;
  Core::LinAlg::Matrix<3, MortarRot::n_dof_, scalar_type_rot_1st>
      T_solid_mtr_times_T_rel_tr_times_lambda_shape;
  Core::LinAlg::Matrix<n_dof_rot_, MortarRot::n_dof_, scalar_type_rot_1st> d_fb_d_lambda_gp;
  Core::LinAlg::Matrix<Solid::n_dof_, MortarRot::n_dof_, scalar_type_rot_1st> d_fs_d_lambda_gp;
  Core::LinAlg::Matrix<3, Solid::n_dof_, scalar_type_rot_1st> d_psi_solid_d_q_solid;
  Core::LinAlg::Matrix<MortarRot::n_dof_, 3, double> d_g_d_psi_beam;
  Core::LinAlg::Matrix<MortarRot::n_dof_, Solid::n_dof_, double> d_g_d_q_solid;
  Core::LinAlg::Matrix<n_dof_rot_, 1, scalar_type_rot_1st> f_beam;
  Core::LinAlg::Matrix<Solid::n_dof_, 1, scalar_type_rot_1st> f_solid;
  Core::LinAlg::Matrix<n_dof_rot_, 3, double> d_f_beam_d_phi;
  Core::LinAlg::Matrix<Solid::n_dof_, 3, double> d_f_solid_d_phi;
  Core::LinAlg::Matrix<n_dof_rot_, n_dof_rot_, double>
      d_f_beam_d_phi_times_T_beam_times_I_beam_tilde_full;
  Core::LinAlg::Matrix<Solid::n_dof_, n_dof_rot_, double>
      d_f_solid_d_phi_times_T_beam_times_I_beam_tilde_full;

  // Initialize scalar variables.
  double segment_jacobian, beam_segmentation_factor;

  // Calculate the meshtying forces.
  // Loop over segments.
  const unsigned int n_segments = this->line_to_3D_segments_.size();
  for (unsigned int i_segment = 0; i_segment < n_segments; i_segment++)
  {
    // Factor to account for the integration segment length.
    beam_segmentation_factor = 0.5 * this->line_to_3D_segments_[i_segment].get_segment_length();

    // Gauss point loop.
    for (unsigned int i_gp = 0;
        i_gp < this->line_to_3D_segments_[i_segment].get_projection_points().size(); i_gp++)
    {
      // Get the current Gauss point.
      const GeometryPair::ProjectionPoint1DTo3D<double>& projected_gauss_point =
          this->line_to_3D_segments_[i_segment].get_projection_points()[i_gp];

      // Get the jacobian in the reference configuration.
      GeometryPair::evaluate_position_derivative1<Beam>(
          projected_gauss_point.get_eta(), this->ele1posref_, dr_beam_ref);

      // Jacobian including the segment length.
      segment_jacobian = dr_beam_ref.norm2() * beam_segmentation_factor;

      // Calculate the rotation vector of this cross section.
      triad_interpolation_scheme.get_interpolated_quaternion_at_xi(
          quaternion_beam_double, projected_gauss_point.get_eta());
      Core::LargeRotations::quaterniontoangle(quaternion_beam_double, psi_beam_double);
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        psi_beam(i_dim) = Core::FADUtils::HigherOrderFadValue<scalar_type_rot_1st>::apply(
            3 + Solid::n_dof_, i_dim, psi_beam_double(i_dim));
      Core::LargeRotations::angletoquaternion(psi_beam, quaternion_beam);
      quaternion_beam_inv = Core::LargeRotations::inversequaternion(quaternion_beam);

      // Get the solid rotation vector.
      ref_triad_interpolation_scheme.get_interpolated_quaternion_at_xi(
          quaternion_beam_ref, projected_gauss_point.get_eta());
      get_solid_rotation_vector<Solid>(rot_coupling_type, projected_gauss_point.get_xi(),
          this->ele2posref_, q_solid, quaternion_beam_ref, psi_solid);
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        psi_solid_val(i_dim) = psi_solid(i_dim).val();
      Core::LargeRotations::angletoquaternion(psi_solid_val, quaternion_solid);

      // Calculate the relative rotation vector.
      Core::LargeRotations::quaternionproduct(
          quaternion_beam_inv, quaternion_solid, quaternion_rel);
      Core::LargeRotations::quaterniontoangle(quaternion_rel, psi_rel);

      // Calculate the transformation matrices.
      T_rel = Core::LargeRotations::tmatrix(psi_rel);
      T_beam = Core::LargeRotations::tmatrix(Core::FADUtils::cast_to_double(psi_beam));
      T_solid = Core::LargeRotations::tmatrix(psi_solid_val);
      T_solid_inv = T_solid;
      Core::LinAlg::inverse(T_solid_inv);

      // Evaluate shape functions.
      GeometryPair::EvaluateShapeFunction<MortarRot>::evaluate(
          lambda_shape_functions, projected_gauss_point.get_eta());
      for (unsigned int i_node = 0; i_node < MortarRot::n_nodes_; i_node++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          lambda_shape_functions_full(i_dim, 3 * i_node + i_dim) = lambda_shape_functions(i_node);

      Core::FE::shape_function_1d(L_i, projected_gauss_point.get_eta(), Core::FE::CellType::line3);
      for (unsigned int i_node = 0; i_node < 3; i_node++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          L_full(i_dim, 3 * i_node + i_dim) = L_i(i_node);

      triad_interpolation_scheme.get_nodal_generalized_rotation_interpolation_matrices_at_xi(
          I_beam_tilde, projected_gauss_point.get_eta());
      for (unsigned int i_node = 0; i_node < 3; i_node++)
        for (unsigned int i_dim_0 = 0; i_dim_0 < 3; i_dim_0++)
          for (unsigned int i_dim_1 = 0; i_dim_1 < 3; i_dim_1++)
            I_beam_tilde_full(i_dim_0, i_node * 3 + i_dim_1) =
                I_beam_tilde[i_node](i_dim_0, i_dim_1);

      // Solid angle derived w.r.t. the solid DOFs.
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        for (unsigned int i_solid = 0; i_solid < Solid::n_dof_; i_solid++)
          d_psi_solid_d_q_solid(i_dim, i_solid) = psi_solid(i_dim).dx(3 + i_solid);

      // Calculate the force terms derived w.r.t. the Lagrange multipliers.
      T_rel_tr_times_lambda_shape.multiply_tn(T_rel, lambda_shape_functions_full);
      d_fb_d_lambda_gp.multiply_tn(L_full, T_rel_tr_times_lambda_shape);
      d_fb_d_lambda_gp.scale(-1.0 * projected_gauss_point.get_gauss_weight() * segment_jacobian);

      T_solid_mtr_times_T_rel_tr_times_lambda_shape.multiply_tn(
          T_solid_inv, T_rel_tr_times_lambda_shape);
      d_fs_d_lambda_gp.multiply_tn(
          d_psi_solid_d_q_solid, T_solid_mtr_times_T_rel_tr_times_lambda_shape);
      d_fs_d_lambda_gp.scale(projected_gauss_point.get_gauss_weight() * segment_jacobian);

      // Calculate the force vectors.
      f_beam.put_scalar(0.0);
      for (unsigned int i_row = 0; i_row < n_dof_rot_; i_row++)
        for (unsigned int i_col = 0; i_col < MortarRot::n_dof_; i_col++)
          f_beam(i_row) += d_fb_d_lambda_gp(i_row, i_col) * lambda_rot(i_col);
      f_solid.put_scalar(0.0);
      for (unsigned int i_row = 0; i_row < Solid::n_dof_; i_row++)
        for (unsigned int i_col = 0; i_col < MortarRot::n_dof_; i_col++)
          f_solid(i_row) += d_fs_d_lambda_gp(i_row, i_col) * lambda_rot(i_col);

      // Derivatives of the force vectors.
      for (unsigned int i_row = 0; i_row < n_dof_rot_; i_row++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          d_f_beam_d_phi(i_row, i_dim) = f_beam(i_row).dx(i_dim);
      for (unsigned int i_row = 0; i_row < Solid::n_dof_; i_row++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          d_f_solid_d_phi(i_row, i_dim) = f_solid(i_row).dx(i_dim);

      T_beam_times_I_beam_tilde_full.multiply(T_beam, I_beam_tilde_full);
      d_f_beam_d_phi_times_T_beam_times_I_beam_tilde_full.multiply(
          d_f_beam_d_phi, T_beam_times_I_beam_tilde_full);
      d_f_solid_d_phi_times_T_beam_times_I_beam_tilde_full.multiply(
          d_f_solid_d_phi, T_beam_times_I_beam_tilde_full);

      // Add to output matrices and vector.
      local_stiff_BB += d_f_beam_d_phi_times_T_beam_times_I_beam_tilde_full;
      for (unsigned int i_beam = 0; i_beam < n_dof_rot_; i_beam++)
        for (unsigned int j_solid = 0; j_solid < Solid::n_dof_; j_solid++)
          local_stiff_BS(i_beam, j_solid) += f_beam(i_beam).dx(3 + j_solid);
      local_stiff_SB += d_f_solid_d_phi_times_T_beam_times_I_beam_tilde_full;
      for (unsigned int i_solid = 0; i_solid < Solid::n_dof_; i_solid++)
        for (unsigned int j_solid = 0; j_solid < Solid::n_dof_; j_solid++)
          local_stiff_SS(i_solid, j_solid) += f_solid(i_solid).dx(3 + j_solid);
    }
  }
}


/**
 * Explicit template initialization of template class.
 */
namespace BeamInteraction
{
  using namespace GeometryPair;

#define initialize_template_beam_to_solid_volume_meshtying_pair_mortar_rotation(            \
    mortar, mortar_rot)                                                                     \
  template class BeamToSolidVolumeMeshtyingPairMortarRotation<t_hermite, t_hex8, mortar,    \
      mortar_rot>;                                                                          \
  template class BeamToSolidVolumeMeshtyingPairMortarRotation<t_hermite, t_hex20, mortar,   \
      mortar_rot>;                                                                          \
  template class BeamToSolidVolumeMeshtyingPairMortarRotation<t_hermite, t_hex27, mortar,   \
      mortar_rot>;                                                                          \
  template class BeamToSolidVolumeMeshtyingPairMortarRotation<t_hermite, t_tet4, mortar,    \
      mortar_rot>;                                                                          \
  template class BeamToSolidVolumeMeshtyingPairMortarRotation<t_hermite, t_tet10, mortar,   \
      mortar_rot>;                                                                          \
  template class BeamToSolidVolumeMeshtyingPairMortarRotation<t_hermite, t_nurbs27, mortar, \
      mortar_rot>;

  initialize_template_beam_to_solid_volume_meshtying_pair_mortar_rotation(t_line2, t_line2);
  initialize_template_beam_to_solid_volume_meshtying_pair_mortar_rotation(t_line2, t_line3);
  initialize_template_beam_to_solid_volume_meshtying_pair_mortar_rotation(t_line2, t_line4);

  initialize_template_beam_to_solid_volume_meshtying_pair_mortar_rotation(t_line3, t_line2);
  initialize_template_beam_to_solid_volume_meshtying_pair_mortar_rotation(t_line3, t_line3);
  initialize_template_beam_to_solid_volume_meshtying_pair_mortar_rotation(t_line3, t_line4);

  initialize_template_beam_to_solid_volume_meshtying_pair_mortar_rotation(t_line4, t_line2);
  initialize_template_beam_to_solid_volume_meshtying_pair_mortar_rotation(t_line4, t_line3);
  initialize_template_beam_to_solid_volume_meshtying_pair_mortar_rotation(t_line4, t_line4);
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE
