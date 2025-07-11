// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_gauss_point.hpp"

#include "4C_beam3_reissner.hpp"
#include "4C_beam3_triad_interpolation_local_rotation_vectors.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_params.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_to_volume.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename Beam, typename Solid>
BeamInteraction::BeamToSolidVolumeMeshtyingPairGaussPoint<Beam,
    Solid>::BeamToSolidVolumeMeshtyingPairGaussPoint()
    : base_class()
{
  // Empty constructor.
}


/**
 *
 */
template <typename Beam, typename Solid>
bool BeamInteraction::BeamToSolidVolumeMeshtyingPairGaussPoint<Beam, Solid>::evaluate(
    Core::LinAlg::SerialDenseVector* forcevec1, Core::LinAlg::SerialDenseVector* forcevec2,
    Core::LinAlg::SerialDenseMatrix* stiffmat11, Core::LinAlg::SerialDenseMatrix* stiffmat12,
    Core::LinAlg::SerialDenseMatrix* stiffmat21, Core::LinAlg::SerialDenseMatrix* stiffmat22)
{
  // Call Evaluate on the geometry Pair. Only do this once for meshtying.
  if (!this->meshtying_is_evaluated_)
  {
    GeometryPair::ElementData<Beam, double> beam_coupling_ref;
    GeometryPair::ElementData<Solid, double> solid_coupling_ref;
    this->get_coupling_reference_position(beam_coupling_ref, solid_coupling_ref);
    this->cast_geometry_pair()->evaluate(
        beam_coupling_ref, solid_coupling_ref, this->line_to_3D_segments_);
    this->meshtying_is_evaluated_ = true;
  }

  // If there are no intersection segments, return no contact status.
  if (this->line_to_3D_segments_.size() == 0) return false;

  // Initialize variables for position and force vectors.
  Core::LinAlg::Matrix<3, 1, double> dr_beam_ref;
  Core::LinAlg::Matrix<3, 1, scalar_type> r_beam;
  Core::LinAlg::Matrix<3, 1, scalar_type> r_solid;
  Core::LinAlg::Matrix<3, 1, scalar_type> force;
  Core::LinAlg::Matrix<Beam::n_dof_, 1, scalar_type> force_element_1(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Solid::n_dof_, 1, scalar_type> force_element_2(
      Core::LinAlg::Initialization::zero);

  // Initialize scalar variables.
  double segment_jacobian = 0.0;
  double beam_segmentation_factor = 0.0;
  double penalty_parameter =
      this->params()->beam_to_solid_volume_meshtying_params()->get_penalty_parameter();

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

      // Get the current positions on beam and solid.
      GeometryPair::evaluate_position<Beam>(
          projected_gauss_point.get_eta(), this->ele1pos_, r_beam);
      GeometryPair::evaluate_position<Solid>(
          projected_gauss_point.get_xi(), this->ele2pos_, r_solid);

      // Calculate the force in this Gauss point. The sign of the force calculated here is the one
      // that acts on the beam.
      force = r_solid;
      force -= r_beam;
      force.scale(penalty_parameter);

      // The force vector is in R3, we need to calculate the equivalent nodal forces on the element
      // dof. This is done with the virtual work equation $F \delta r = f \delta q$.
      for (unsigned int i_dof = 0; i_dof < Beam::n_dof_; i_dof++)
        for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
          force_element_1(i_dof) += force(i_dir) * r_beam(i_dir).dx(i_dof) *
                                    projected_gauss_point.get_gauss_weight() * segment_jacobian;
      for (unsigned int i_dof = 0; i_dof < Solid::n_dof_; i_dof++)
        for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
          force_element_2(i_dof) -= force(i_dir) * r_solid(i_dir).dx(i_dof + Beam::n_dof_) *
                                    projected_gauss_point.get_gauss_weight() * segment_jacobian;
    }
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

  // Return true as there are meshtying contributions.
  return true;
}


/**
 *
 */
template <typename Beam, typename Solid>
void BeamInteraction::BeamToSolidVolumeMeshtyingPairGaussPoint<Beam, Solid>::evaluate_and_assemble(
    const std::shared_ptr<const Core::FE::Discretization>& discret,
    const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
    const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector)
{
  // This function only gives contributions for rotational coupling.
  auto rot_coupling_type =
      this->params()->beam_to_solid_volume_meshtying_params()->get_rotational_coupling_type();
  if (rot_coupling_type == Inpar::BeamToSolid::BeamToSolidRotationCoupling::none) return;

  // Call Evaluate on the geometry Pair. Only do this once for meshtying.
  if (!this->meshtying_is_evaluated_)
  {
    GeometryPair::ElementData<Beam, double> beam_coupling_ref;
    GeometryPair::ElementData<Solid, double> solid_coupling_ref;
    this->get_coupling_reference_position(beam_coupling_ref, solid_coupling_ref);
    this->cast_geometry_pair()->evaluate(
        beam_coupling_ref, solid_coupling_ref, this->line_to_3D_segments_);
    this->meshtying_is_evaluated_ = true;
  }

  // If there are no intersection segments, return no contact status.
  if (this->line_to_3D_segments_.size() == 0) return;

  // Get the beam triad interpolation schemes.
  LargeRotations::TriadInterpolationLocalRotationVectors<3, double> triad_interpolation_scheme;
  LargeRotations::TriadInterpolationLocalRotationVectors<3, double> ref_triad_interpolation_scheme;
  get_beam_triad_interpolation_scheme(*discret, *displacement_vector, this->element1(),
      triad_interpolation_scheme, ref_triad_interpolation_scheme);

  // Set the FAD variables for the solid DOFs.
  auto q_solid =
      GeometryPair::InitializeElementData<Solid, scalar_type_rot_2nd>::initialize(this->element2());
  for (unsigned int i_solid = 0; i_solid < Solid::n_dof_; i_solid++)
    q_solid.element_position_(i_solid) =
        Core::FADUtils::HigherOrderFadValue<scalar_type_rot_2nd>::apply(3 + Solid::n_dof_,
            3 + i_solid, Core::FADUtils::cast_to_double(this->ele2pos_.element_position_(i_solid)));


  // Initialize local matrices.
  Core::LinAlg::Matrix<n_dof_pair_, 1, double> local_force(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<n_dof_pair_, n_dof_pair_, double> local_stiff(
      Core::LinAlg::Initialization::zero);


  if (rot_coupling_type == Inpar::BeamToSolid::BeamToSolidRotationCoupling::fix_triad_2d)
  {
    // In the case of "fix_triad_2d" we couple both, the ey and ez direction to the beam. Therefore,
    // we have to evaluate the coupling terms w.r.t both of those coupling types.
    evaluate_rotational_coupling_terms(
        Inpar::BeamToSolid::BeamToSolidRotationCoupling::deformation_gradient_y_2d, q_solid,
        triad_interpolation_scheme, ref_triad_interpolation_scheme, local_force, local_stiff);
    evaluate_rotational_coupling_terms(
        Inpar::BeamToSolid::BeamToSolidRotationCoupling::deformation_gradient_z_2d, q_solid,
        triad_interpolation_scheme, ref_triad_interpolation_scheme, local_force, local_stiff);
  }
  else
    evaluate_rotational_coupling_terms(rot_coupling_type, q_solid, triad_interpolation_scheme,
        ref_triad_interpolation_scheme, local_force, local_stiff);


  // Get the GIDs of this pair.
  // The first 9 entries in the vector will be the rotational DOFs of the beam, the other entries
  // are the solid DOFs.
  const auto rot_gid = Utils::get_element_rot_gid_indices(*discret, this->element1());
  std::vector<int> lm_solid, lmowner, lmstride;
  this->element2()->location_vector(*discret, lm_solid, lmowner, lmstride);
  Core::LinAlg::Matrix<n_dof_pair_, 1, int> gid_pair;
  for (unsigned int i = 0; i < n_dof_rot_; i++) gid_pair(i) = rot_gid[i];
  for (unsigned int i = 0; i < Solid::n_dof_; i++) gid_pair(i + n_dof_rot_) = lm_solid[i];


  // If given, assemble force terms into the global force vector.
  if (force_vector != nullptr)
    force_vector->sum_into_global_values(gid_pair.num_rows(), gid_pair.data(), local_force.data());

  // If given, assemble force terms into the global stiffness matrix.
  if (stiffness_matrix != nullptr)
    for (unsigned int i_dof = 0; i_dof < n_dof_pair_; i_dof++)
      for (unsigned int j_dof = 0; j_dof < n_dof_pair_; j_dof++)
        stiffness_matrix->fe_assemble(Core::FADUtils::cast_to_double(local_stiff(i_dof, j_dof)),
            gid_pair(i_dof), gid_pair(j_dof));
}

/**
 *
 */
template <typename Beam, typename Solid>
void BeamInteraction::BeamToSolidVolumeMeshtyingPairGaussPoint<Beam,
    Solid>::evaluate_rotational_coupling_terms(  //
    const Inpar::BeamToSolid::BeamToSolidRotationCoupling& rot_coupling_type,
    const GeometryPair::ElementData<Solid, scalar_type_rot_2nd>& q_solid,
    const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&
        triad_interpolation_scheme,
    const LargeRotations::TriadInterpolationLocalRotationVectors<3, double>&
        ref_triad_interpolation_scheme,
    Core::LinAlg::Matrix<n_dof_pair_, 1, double>& local_force,
    Core::LinAlg::Matrix<n_dof_pair_, n_dof_pair_, double>& local_stiff) const
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
  Core::LinAlg::Matrix<3, 1, scalar_type_rot_1st> potential_variation;
  Core::LinAlg::Matrix<n_dof_rot_, 1, scalar_type_rot_1st> fc_beam_gp;
  Core::LinAlg::Matrix<3, Solid::n_dof_, scalar_type_rot_1st> d_psi_solid_d_q_solid;
  Core::LinAlg::Matrix<3, 1, scalar_type_rot_1st> Tinv_solid_times_potential_variation;
  Core::LinAlg::Matrix<Solid::n_dof_, 1, scalar_type_rot_1st> fc_solid_gp;
  Core::LinAlg::SerialDenseVector L_i(3);
  Core::LinAlg::Matrix<n_dof_rot_, 3, double> d_fc_beam_d_psi_beam;
  Core::LinAlg::Matrix<Solid::n_dof_, 3, double> d_fc_solid_d_psi_beam;
  std::vector<Core::LinAlg::Matrix<3, 3, double>> I_beam_tilde;
  Core::LinAlg::Matrix<3, n_dof_rot_, double> I_beam_tilde_full;
  Core::LinAlg::Matrix<3, n_dof_rot_, double> T_beam_times_I_beam_tilde_full;
  Core::LinAlg::Matrix<n_dof_rot_, n_dof_rot_, double> stiff_beam_beam_gp;
  Core::LinAlg::Matrix<Solid::n_dof_, n_dof_rot_, double> stiff_solid_beam_gp;
  Core::LinAlg::Matrix<n_dof_rot_, Solid::n_dof_, double> stiff_beam_solid_gp;

  // Initialize scalar variables.
  double segment_jacobian = 0.0;
  double beam_segmentation_factor = 0.0;
  double rotational_penalty_parameter = this->params()
                                            ->beam_to_solid_volume_meshtying_params()
                                            ->get_rotational_coupling_penalty_parameter();

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
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        psi_solid_val(i_dim) = psi_solid(i_dim).val();
      Core::LargeRotations::angletoquaternion(psi_solid_val, quaternion_solid);

      // Calculate the relative rotation vector.
      Core::LargeRotations::quaternionproduct(
          quaternion_beam_inv, quaternion_solid, quaternion_rel);
      Core::LargeRotations::quaterniontoangle(quaternion_rel, psi_rel);

      // Calculate the transformation matrices.
      T_beam = Core::LargeRotations::tmatrix(Core::FADUtils::cast_to_double(psi_beam));
      T_solid = Core::LargeRotations::tmatrix(psi_solid_val);

      // Force terms.
      Core::FE::shape_function_1d(L_i, projected_gauss_point.get_eta(), Core::FE::CellType::line3);
      potential_variation = psi_rel;
      potential_variation.scale(rotational_penalty_parameter);
      for (unsigned int i_node = 0; i_node < 3; i_node++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          fc_beam_gp(3 * i_node + i_dim) = -1.0 * L_i(i_node) * potential_variation(i_dim) *
                                           projected_gauss_point.get_gauss_weight() *
                                           segment_jacobian;
      for (unsigned int i_dof = 0; i_dof < n_dof_rot_; i_dof++)
        local_force(i_dof) += Core::FADUtils::cast_to_double(fc_beam_gp(i_dof));

      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        for (unsigned int i_solid = 0; i_solid < Solid::n_dof_; i_solid++)
          d_psi_solid_d_q_solid(i_dim, i_solid) = psi_solid(i_dim).dx(3 + i_solid);
      T_solid_inv = T_solid;
      Core::LinAlg::inverse(T_solid_inv);
      Tinv_solid_times_potential_variation.multiply_tn(T_solid_inv, potential_variation);
      fc_solid_gp.multiply_tn(d_psi_solid_d_q_solid, Tinv_solid_times_potential_variation);
      fc_solid_gp.scale(projected_gauss_point.get_gauss_weight() * segment_jacobian);
      for (unsigned int i_dof = 0; i_dof < Solid::n_dof_; i_dof++)
        local_force(n_dof_rot_ + i_dof) += Core::FADUtils::cast_to_double(fc_solid_gp(i_dof));


      // Stiffness terms.
      for (unsigned int i_beam_dof = 0; i_beam_dof < n_dof_rot_; i_beam_dof++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          d_fc_beam_d_psi_beam(i_beam_dof, i_dim) = fc_beam_gp(i_beam_dof).dx(i_dim);
      for (unsigned int i_solid_dof = 0; i_solid_dof < Solid::n_dof_; i_solid_dof++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          d_fc_solid_d_psi_beam(i_solid_dof, i_dim) = fc_solid_gp(i_solid_dof).dx(i_dim);

      triad_interpolation_scheme.get_nodal_generalized_rotation_interpolation_matrices_at_xi(
          I_beam_tilde, projected_gauss_point.get_eta());
      for (unsigned int i_node = 0; i_node < 3; i_node++)
        for (unsigned int i_dim_0 = 0; i_dim_0 < 3; i_dim_0++)
          for (unsigned int i_dim_1 = 0; i_dim_1 < 3; i_dim_1++)
            I_beam_tilde_full(i_dim_0, i_node * 3 + i_dim_1) =
                I_beam_tilde[i_node](i_dim_0, i_dim_1);

      T_beam_times_I_beam_tilde_full.multiply(
          Core::FADUtils::cast_to_double(T_beam), I_beam_tilde_full);
      stiff_beam_beam_gp.multiply(d_fc_beam_d_psi_beam, T_beam_times_I_beam_tilde_full);
      for (unsigned int i_dof = 0; i_dof < n_dof_rot_; i_dof++)
        for (unsigned int j_dof = 0; j_dof < n_dof_rot_; j_dof++)
          local_stiff(i_dof, j_dof) += stiff_beam_beam_gp(i_dof, j_dof);

      stiff_solid_beam_gp.multiply(d_fc_solid_d_psi_beam, T_beam_times_I_beam_tilde_full);
      for (unsigned int i_dof = 0; i_dof < Solid::n_dof_; i_dof++)
        for (unsigned int j_dof = 0; j_dof < n_dof_rot_; j_dof++)
          local_stiff(i_dof + n_dof_rot_, j_dof) += stiff_solid_beam_gp(i_dof, j_dof);

      for (unsigned int i_dof = 0; i_dof < n_dof_rot_; i_dof++)
        for (unsigned int j_dof = 0; j_dof < Solid::n_dof_; j_dof++)
          local_stiff(i_dof, j_dof + n_dof_rot_) += fc_beam_gp(i_dof).dx(3 + j_dof);

      for (unsigned int i_dof = 0; i_dof < Solid::n_dof_; i_dof++)
        for (unsigned int j_dof = 0; j_dof < Solid::n_dof_; j_dof++)
          local_stiff(i_dof + n_dof_rot_, j_dof + n_dof_rot_) += fc_solid_gp(i_dof).dx(3 + j_dof);
    }
  }
}


/**
 * Explicit template initialization of template class.
 */
namespace BeamInteraction
{
  using namespace GeometryPair;

  template class BeamToSolidVolumeMeshtyingPairGaussPoint<t_hermite, t_hex8>;
  template class BeamToSolidVolumeMeshtyingPairGaussPoint<t_hermite, t_hex20>;
  template class BeamToSolidVolumeMeshtyingPairGaussPoint<t_hermite, t_hex27>;
  template class BeamToSolidVolumeMeshtyingPairGaussPoint<t_hermite, t_tet4>;
  template class BeamToSolidVolumeMeshtyingPairGaussPoint<t_hermite, t_tet10>;
  template class BeamToSolidVolumeMeshtyingPairGaussPoint<t_hermite, t_nurbs27>;
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE
