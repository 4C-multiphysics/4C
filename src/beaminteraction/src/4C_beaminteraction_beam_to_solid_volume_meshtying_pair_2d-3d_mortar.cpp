// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_2d-3d_mortar.hpp"

#include "4C_beam3_reissner.hpp"
#include "4C_beaminteraction_beam_to_solid_mortar_manager.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_base.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_visualization.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_params.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_visualization_output_params.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_beaminteraction_geometry_pair_access_traits.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_element_shape_functions.hpp"
#include "4C_geometry_pair_line_to_3D_evaluation_data.hpp"
#include "4C_geometry_pair_line_to_volume_gauss_point_projection_cross_section.hpp"
#include "4C_geometry_pair_utility_classes.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_utils_fad.hpp"

#include <math.h>

#include <cmath>
#include <memory>
#include <unordered_set>

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename Beam, typename Solid, typename Mortar>
void BeamInteraction::BeamToSolidVolumeMeshtyingPair2D3DMortar<Beam, Solid, Mortar>::pre_evaluate()
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
template <typename Beam, typename Solid, typename Mortar>
void BeamInteraction::BeamToSolidVolumeMeshtyingPair2D3DMortar<Beam, Solid,
    Mortar>::evaluate_and_assemble_mortar_contributions(const Core::FE::Discretization& discret,
    const BeamInteraction::BeamToSolidMortarManager* mortar_manager,
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

  // Check that the beam element is a Simo-Reissner beam.
  auto beam_ele = dynamic_cast<const Discret::Elements::Beam3r*>(this->element1());
  if (beam_ele == nullptr)
    FOUR_C_THROW("GetBeamTriadInterpolationScheme is only implemented for SR beams.");
  const double beam_cross_section_radius =
      beam_ele->get_circular_cross_section_radius_for_interactions();

  // Get the vector with the projection points for this pair.
  const std::vector<GeometryPair::ProjectionPoint1DTo3D<double>>& projection_points =
      this->line_to_3D_segments_[0].get_projection_points();

  // If there are no projection points, return no contact status.
  if (projection_points.size() == 0) return;

  // Reset the rotational coupling stiffness data
  for (unsigned int i = 0; i < n_dof_rot_; i++)
    for (unsigned int j = 0; j < Mortar::n_dof_; j++)
      for (unsigned int l = 0; l < n_dof_rot_; l++)
        lagrange_shape_times_skew_times_mortar_shape_lin_psi_times_t_times_itilde_[i][j][l] = 0.0;

  // Lambda to set the rotation vector FAD variables
  auto set_q_fad = [](const Core::LinAlg::Matrix<3, 1, double>& q_original,
                       Core::LinAlg::Matrix<3, 1, scalar_type_rotation_vector>& q_fad)
  {
    for (unsigned int i_dof = 0; i_dof < q_original.num_rows(); i_dof++)
      q_fad(i_dof) = Core::FADUtils::HigherOrderFadValue<scalar_type_rotation_vector>::apply(
          3, i_dof, Core::FADUtils::cast_to_double(q_original(i_dof)));
  };

  // Shape function matrices.
  Core::LinAlg::Matrix<3, Solid::n_dof_, double> N;
  Core::LinAlg::Matrix<3, Beam::n_dof_, double> H;
  Core::LinAlg::Matrix<3, n_dof_rot_, double> L;
  std::vector<Core::LinAlg::Matrix<3, 3, double>> I_tilde_vector;
  Core::LinAlg::Matrix<3, n_dof_rot_, double> I_tilde;
  Core::LinAlg::Matrix<3, Mortar::n_dof_, double> psi;
  Core::LinAlg::Matrix<1, Mortar::curve_discretization_::n_dof_, double> psi_curve;

  // Initialize vector and matrix variables for the Gauss integration.
  Core::LinAlg::Matrix<3, 1, double> dr_beam_ref;
  Core::LinAlg::Matrix<3, 1, double> cross_section_vector_ref;
  Core::LinAlg::Matrix<3, 1, scalar_type_rotation_vector> cross_section_vector_current;
  Core::LinAlg::Matrix<3, 1, double> pos_beam;
  Core::LinAlg::Matrix<3, 1, double> pos_solid;
  Core::LinAlg::Matrix<4, 1, double> quaternion_double;
  Core::LinAlg::Matrix<3, 1, double> rotation_vector_double;
  Core::LinAlg::Matrix<4, 1, scalar_type_rotation_vector> quaternion_fad;
  Core::LinAlg::Matrix<3, 1, scalar_type_rotation_vector> rotation_vector_fad;
  Core::LinAlg::Matrix<3, 3, scalar_type_rotation_vector> triad_fad;
  Core::LinAlg::Matrix<3, n_dof_rot_, double> T_times_I_tilde;

  // Initialize local matrices.
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> local_constraint(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Mortar::n_dof_, Beam::n_dof_, double> local_constraint_lin_beam_pos(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Mortar::n_dof_, n_dof_rot_, double> local_constraint_lin_beam_rot(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<n_dof_rot_, Mortar::n_dof_, double> force_beam_rot_lin_lambda(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Mortar::n_dof_, Solid::n_dof_, double> local_constraint_lin_solid(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> local_kappa(Core::LinAlg::Initialization::zero);

  // Initialize scalar variables.
  double eta_last_gauss_point = 1e10;
  double beam_jacobian = 0.0;

  // Calculate the mesh tying forces.
  // Loop over segments.
  for (const auto& projected_gauss_point : projection_points)
  {
    // Get the current Gauss point.
    const auto& eta = projected_gauss_point.get_eta();

    // Evaluate all beam specific terms. This only has to be done if the Gauss point position on the
    // beam has changed compared to the last Gauss point.
    if (std::abs(eta - eta_last_gauss_point) > 1e-10)
    {
      GeometryPair::evaluate_position_derivative1<Beam>(eta, this->ele1posref_, dr_beam_ref);
      beam_jacobian = dr_beam_ref.norm2();

      GeometryPair::evaluate_shape_function_matrix<Beam>(
          H, eta, this->ele1pos_.shape_function_data_);
      GeometryPair::evaluate_position<Beam>(eta, this->ele1pos_, pos_beam);

      GeometryPair::evaluate_shape_function_matrix<GeometryPair::t_line3>(L, eta);

      triad_interpolation_scheme_.get_nodal_generalized_rotation_interpolation_matrices_at_xi(
          I_tilde_vector, eta);
      for (unsigned int i_node = 0; i_node < n_nodes_rot_; i_node++)
        for (unsigned int i_dim_0 = 0; i_dim_0 < 3; i_dim_0++)
          for (unsigned int i_dim_1 = 0; i_dim_1 < 3; i_dim_1++)
            I_tilde(i_dim_0, i_node * 3 + i_dim_1) = I_tilde_vector[i_node](i_dim_0, i_dim_1);

      // Get the rotation vector at this Gauss point
      triad_interpolation_scheme_.get_interpolated_quaternion_at_xi(quaternion_double, eta);
      Core::LargeRotations::quaterniontoangle(quaternion_double, rotation_vector_double);
      set_q_fad(rotation_vector_double, rotation_vector_fad);
      Core::LargeRotations::angletoquaternion(rotation_vector_fad, quaternion_fad);
      Core::LargeRotations::quaterniontotriad(quaternion_fad, triad_fad);
      const auto T_beam = Core::LargeRotations::tmatrix(rotation_vector_double);
      T_times_I_tilde.multiply(T_beam, I_tilde);
    }

    // Get the cross section vector
    cross_section_vector_ref(0) = 0.0;
    cross_section_vector_ref(1) = projected_gauss_point.get_eta_cross_section()(0);
    cross_section_vector_ref(2) = projected_gauss_point.get_eta_cross_section()(1);
    cross_section_vector_current.multiply(triad_fad, cross_section_vector_ref);
    const double cross_section_angle_parameter_space =
        atan2(cross_section_vector_ref(2), cross_section_vector_ref(1));

    // Get the shape function matrices
    GeometryPair::evaluate_shape_function_matrix<Solid>(
        N, projected_gauss_point.get_xi(), this->ele2pos_.shape_function_data_);
    GeometryPair::evaluate_position<Solid>(
        projected_gauss_point.get_xi(), this->ele2pos_, pos_solid);
    Core::LinAlg::Matrix<2, 1, double> xi_mortar;
    xi_mortar(0) = eta;
    xi_mortar(1) = cross_section_angle_parameter_space;
    GeometryPair::evaluate_shape_function_matrix<Mortar>(psi, xi_mortar);

    // Numerical integration factor for this Gauss point
    const double integration_factor =
        projected_gauss_point.get_gauss_weight() * beam_jacobian * beam_cross_section_radius * M_PI;

    // Evaluate the mortar matrices
    Core::LinAlg::Matrix<Mortar::n_dof_, Solid::n_dof_, double> local_constraint_lin_solid_gp(
        Core::LinAlg::Initialization::zero);
    local_constraint_lin_solid_gp.multiply_tn(psi, N);
    local_constraint_lin_solid_gp.scale(-1.0 * integration_factor);
    local_constraint_lin_solid += local_constraint_lin_solid_gp;

    Core::LinAlg::Matrix<Mortar::n_dof_, Beam::n_dof_, double> local_constraint_lin_beam_pos_gp(
        Core::LinAlg::Initialization::zero);
    local_constraint_lin_beam_pos_gp.multiply_tn(psi, H);
    local_constraint_lin_beam_pos_gp.scale(integration_factor);
    local_constraint_lin_beam_pos += local_constraint_lin_beam_pos_gp;

    Core::LinAlg::Matrix<Mortar::n_dof_, 1, scalar_type_rotation_vector> psi_times_rcs(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<Mortar::n_dof_, 3, double> psi_times_rcs_lin_phi(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<Mortar::n_dof_, n_dof_rot_, double> local_constraint_lin_beam_rot_gp(
        Core::LinAlg::Initialization::zero);
    psi_times_rcs.multiply_tn(psi, cross_section_vector_current);
    for (unsigned int i_lambda = 0; i_lambda < Mortar::n_dof_; i_lambda++)
    {
      for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
      {
        psi_times_rcs_lin_phi(i_lambda, i_dir) = psi_times_rcs(i_lambda).dx(i_dir);
      }
    }
    local_constraint_lin_beam_rot_gp.multiply_nn(psi_times_rcs_lin_phi, T_times_I_tilde);
    local_constraint_lin_beam_rot_gp.scale(integration_factor);
    local_constraint_lin_beam_rot += local_constraint_lin_beam_rot_gp;

    Core::LinAlg::Matrix<n_dof_rot_, Mortar::n_dof_, scalar_type_rotation_vector>
        force_beam_rot_lin_lambda_gp(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 3, scalar_type_rotation_vector> skew_rcs(
        Core::LinAlg::Initialization::zero);
    Core::LargeRotations::computespin(skew_rcs, cross_section_vector_current);
    Core::LinAlg::Matrix<3, Mortar::n_dof_, scalar_type_rotation_vector> skew_rcs_times_psi(
        Core::LinAlg::Initialization::zero);
    skew_rcs_times_psi.multiply_nn(skew_rcs, psi);
    force_beam_rot_lin_lambda_gp.multiply_tn(L, skew_rcs_times_psi);
    for (unsigned int i = 0; i < n_dof_rot_; i++)
      for (unsigned int j = 0; j < Mortar::n_dof_; j++)
        for (unsigned int l = 0; l < n_dof_rot_; l++)
          for (unsigned int k = 0; k < 3; k++)
            lagrange_shape_times_skew_times_mortar_shape_lin_psi_times_t_times_itilde_[i][j][l] +=
                integration_factor * force_beam_rot_lin_lambda_gp(i, j).dx(k) *
                T_times_I_tilde(k, l);
    force_beam_rot_lin_lambda_gp.scale(integration_factor);
    force_beam_rot_lin_lambda += Core::FADUtils::cast_to_double(force_beam_rot_lin_lambda_gp);

    Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> local_constraint_gp(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, double> diff(Core::LinAlg::Initialization::zero);
    diff = pos_beam;
    diff += Core::FADUtils::cast_to_double(cross_section_vector_current);
    diff -= pos_solid;
    local_constraint_gp.multiply_tn(psi, diff);
    local_constraint_gp.scale(integration_factor);
    local_constraint += local_constraint_gp;

    // Fill in the local templated mortar scaling vector kappa
    GeometryPair::evaluate_shape_function_matrix<typename Mortar::curve_discretization_>(
        psi_curve, projected_gauss_point.get_eta());
    for (unsigned int i_node_mortar_centerline = 0; i_node_mortar_centerline < Mortar::n_nodes_;
        i_node_mortar_centerline++)
    {
      constexpr int n_dof_per_node = Mortar::n_val_ * Mortar::spatial_dim_;
      for (unsigned int i_dof = 0; i_dof < n_dof_per_node; i_dof++)
      {
        local_kappa(n_dof_per_node * i_node_mortar_centerline + i_dof) +=
            psi_curve(i_node_mortar_centerline) * integration_factor;
      }
    }

    // Set the eta value for this Gauss point.
    eta_last_gauss_point = eta;
  }

  // Get the GIDs of the solid and beam.
  Core::LinAlg::Matrix<Beam::n_dof_, 1, int> beam_centerline_gid;
  Utils::get_element_centerline_gid_indices(discret, this->element1(), beam_centerline_gid);
  const auto git_rot_beam = Utils::get_element_rot_gid_indices(discret, this->element1());
  std::vector<int> gid_solid, lmowner, lmstride;
  this->element2()->location_vector(discret, gid_solid, lmowner, lmstride);

  // Get the Lagrange multiplier GIDs.
  const auto& [lambda_gid, _] = mortar_manager->location_vector(*this);

  // Assemble into the global vectors
  global_constraint.sum_into_global_values(
      lambda_gid.size(), lambda_gid.data(), local_constraint.data());
  global_kappa.sum_into_global_values(lambda_gid.size(), lambda_gid.data(), local_kappa.data());
  local_kappa.put_scalar(1.0);
  global_lambda_active.sum_into_global_values(
      lambda_gid.size(), lambda_gid.data(), local_kappa.data());

  // Assemble into global matrices.
  for (unsigned int i_dof_lambda = 0; i_dof_lambda < Mortar::n_dof_; i_dof_lambda++)
  {
    for (unsigned int i_dof_beam_pos = 0; i_dof_beam_pos < Beam::n_dof_; i_dof_beam_pos++)
    {
      global_constraint_lin_beam.fe_assemble(
          local_constraint_lin_beam_pos(i_dof_lambda, i_dof_beam_pos), lambda_gid[i_dof_lambda],
          beam_centerline_gid(i_dof_beam_pos));
      global_force_beam_lin_lambda.fe_assemble(
          local_constraint_lin_beam_pos(i_dof_lambda, i_dof_beam_pos),
          beam_centerline_gid(i_dof_beam_pos), lambda_gid[i_dof_lambda]);
    }
    for (unsigned int i_dof_rot = 0; i_dof_rot < n_dof_rot_; i_dof_rot++)
    {
      global_constraint_lin_beam.fe_assemble(local_constraint_lin_beam_rot(i_dof_lambda, i_dof_rot),
          lambda_gid[i_dof_lambda], git_rot_beam[i_dof_rot]);
      global_force_beam_lin_lambda.fe_assemble(
          local_constraint_lin_beam_rot(i_dof_lambda, i_dof_rot), git_rot_beam[i_dof_rot],
          lambda_gid[i_dof_lambda]);
    }
    for (unsigned int i_dof_solid = 0; i_dof_solid < Solid::n_dof_; i_dof_solid++)
    {
      global_constraint_lin_solid.fe_assemble(local_constraint_lin_solid(i_dof_lambda, i_dof_solid),
          lambda_gid[i_dof_lambda], gid_solid[i_dof_solid]);
      global_force_solid_lin_lambda.fe_assemble(
          local_constraint_lin_solid(i_dof_lambda, i_dof_solid), gid_solid[i_dof_solid],
          lambda_gid[i_dof_lambda]);
    }
  }
}

/**
 *
 */
template <typename Beam, typename Solid, typename Mortar>
void BeamInteraction::BeamToSolidVolumeMeshtyingPair2D3DMortar<Beam, Solid,
    Mortar>::evaluate_and_assemble(const Core::FE::Discretization& discret,
    const BeamToSolidMortarManager* mortar_manager,
    const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
    const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
    const Core::LinAlg::Vector<double>& global_lambda,
    const Core::LinAlg::Vector<double>& displacement_vector)
{
  // At this point the pair is already evaluated in the current deformation state, so we don't have
  // to perform the projections or integration again, we can simply take the values previously
  // computed and multiply them with the Lagrange multipliers.

  // If there are no intersection segments, no contact terms will be assembled.
  const unsigned int n_segments = this->line_to_3D_segments_.size();
  if (n_segments == 0) return;

  // This pair only gives contributions to the stiffness matrix
  if (stiffness_matrix == nullptr) return;

  // Get the Lagrange multipliers DOF vector for this pair
  const auto& [lambda_gid_pos, _] = mortar_manager->location_vector(*this);
  std::vector<double> lambda_pos_vector = Core::FE::extract_values(global_lambda, lambda_gid_pos);
  const auto lambda_pos = Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>(lambda_pos_vector.data());

  // Multiply with the matrices evaluated in evaluate_and_assemble_mortar_contributions
  Core::LinAlg::Matrix<n_dof_rot_, n_dof_rot_, double> local_stiff(
      Core::LinAlg::Initialization::zero);
  for (unsigned int i = 0; i < n_dof_rot_; i++)
    for (unsigned int j = 0; j < Mortar::n_dof_; j++)
      for (unsigned int l = 0; l < n_dof_rot_; l++)
        local_stiff(i, l) +=
            lagrange_shape_times_skew_times_mortar_shape_lin_psi_times_t_times_itilde_[i][j][l] *
            lambda_pos(j);

  // Get the GIDs of the solid and beam.
  std::vector<int> gid_solid, lmowner, lmstride;
  this->element2()->location_vector(discret, gid_solid, lmowner, lmstride);
  const auto git_rot_beam = Utils::get_element_rot_gid_indices(discret, this->element1());

  // Assemble force terms into the global stiffness matrix.
  for (unsigned int i_dof = 0; i_dof < n_dof_rot_; i_dof++)
    for (unsigned int j_dof = 0; j_dof < n_dof_rot_; j_dof++)
      stiffness_matrix->fe_assemble(
          local_stiff(i_dof, j_dof), git_rot_beam[i_dof], git_rot_beam[j_dof]);
}

/**
 *
 */
template <typename Beam, typename Solid, typename Mortar>
void BeamInteraction::BeamToSolidVolumeMeshtyingPair2D3DMortar<Beam, Solid,
    Mortar>::reset_rotation_state(const Core::FE::Discretization& discret,
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& ia_discolnp)
{
  get_beam_triad_interpolation_scheme(discret, *ia_discolnp, this->element1(),
      triad_interpolation_scheme_, this->triad_interpolation_scheme_ref_);
}

/**
 *
 */
template <typename Beam, typename Solid, typename Mortar>
void BeamInteraction::BeamToSolidVolumeMeshtyingPair2D3DMortar<Beam, Solid,
    Mortar>::get_pair_visualization(std::shared_ptr<BeamToSolidVisualizationOutputWriterBase>
                                        visualization_writer,
    Teuchos::ParameterList& visualization_params) const
{
  // Get visualization of base method.
  base_class::get_pair_visualization(visualization_writer, visualization_params);

  std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization> visualization_continuous =
      visualization_writer->get_visualization_writer("btsv-mortar-continuous");

  const std::shared_ptr<const BeamToSolidVolumeMeshtyingVisualizationOutputParams>&
      output_params_ptr =
          visualization_params
              .get<std::shared_ptr<const BeamToSolidVolumeMeshtyingVisualizationOutputParams>>(
                  "btsv-output_params_ptr");
  const bool write_unique_ids = output_params_ptr->get_write_unique_ids_flag();

  if (visualization_continuous != nullptr)
  {
    // Check if data for this beam was already written.
    std::shared_ptr<std::unordered_set<int>> beam_tracker_2d_3d_continuous =
        visualization_params.get<std::shared_ptr<std::unordered_set<int>>>(
            "beam_tracker_2d_3d_continuous");

    auto it = beam_tracker_2d_3d_continuous->find(this->element1()->id());
    if (it == beam_tracker_2d_3d_continuous->end())
    {
      // Only do something if this beam element did not write any output yet.

      // Add this element Id to the tracker.
      beam_tracker_2d_3d_continuous->insert(this->element1()->id());

      // Setup variables.
      GeometryPair::ElementData<Mortar, double> element_data_lambda;
      Core::LinAlg::Matrix<3, 1, double> X;
      Core::LinAlg::Matrix<3, 1, double> r;
      Core::LinAlg::Matrix<3, 1, double> u;
      Core::LinAlg::Matrix<3, 3, double> triad;
      Core::LinAlg::Matrix<3, 3, double> triad_ref;
      Core::LinAlg::Matrix<3, 1, double> cross_section_vector_material;
      Core::LinAlg::Matrix<3, 1, double> cross_section_vector_ref;
      Core::LinAlg::Matrix<3, 1, double> cross_section_vector_current;
      Core::LinAlg::Matrix<3, 1, double> lambda_interpolated;

      // Get beam radius.
      const auto* beam_ele = dynamic_cast<const Discret::Elements::Beam3r*>(this->element1());
      if (beam_ele == nullptr)
        FOUR_C_THROW("GetBeamTriadInterpolationScheme is only implemented for SR beams.");
      const double beam_cross_section_radius =
          beam_ele->get_circular_cross_section_radius_for_interactions();

      // Get the mortar manager and the global lambda vector, those objects will be used to get the
      // discrete Lagrange multiplier values for this pair.
      std::shared_ptr<const BeamInteraction::BeamToSolidMortarManager> mortar_manager =
          visualization_params
              .get<std::shared_ptr<const BeamInteraction::BeamToSolidMortarManager>>(
                  "mortar_manager");
      std::shared_ptr<Core::LinAlg::Vector<double>> lambda =
          visualization_params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("lambda");

      // Get the lambda GIDs of this pair.
      const auto& [lambda_row_pos, _] = mortar_manager->location_vector(*this);

      std::vector<double> lambda_pair = Core::FE::extract_values(*lambda, lambda_row_pos);
      for (unsigned int i_dof = 0; i_dof < Mortar::n_dof_; i_dof++)
        element_data_lambda.element_position_(i_dof) = lambda_pair[i_dof];

      // Get parameters for visualization
      const unsigned int mortar_segments =
          visualization_params
              .get<std::shared_ptr<const BeamToSolidVolumeMeshtyingVisualizationOutputParams>>(
                  "btsv-output_params_ptr")
              ->get_mortar_lambda_continuous_segments();
      const unsigned int mortar_segments_circumference =
          visualization_params
              .get<std::shared_ptr<const BeamToSolidVolumeMeshtyingVisualizationOutputParams>>(
                  "btsv-output_params_ptr")
              ->get_mortar_lambda_continuous_segments_circumference();

      // Get visualization data vectors
      auto& visualization_data = visualization_continuous->get_visualization_data();
      std::vector<double>& point_coordinates = visualization_data.get_point_coordinates(
          (mortar_segments + 1) * 3 * this->line_to_3D_segments_.size());
      std::vector<double>& displacement = visualization_data.get_point_data<double>(
          "displacement", (mortar_segments + 1) * 3 * this->line_to_3D_segments_.size());
      std::vector<double>& lambda_vis = visualization_data.get_point_data<double>(
          "lambda", (mortar_segments + 1) * 3 * this->line_to_3D_segments_.size());
      std::vector<uint8_t>& cell_types = visualization_data.get_cell_types();
      std::vector<int32_t>& cell_offsets = visualization_data.get_cell_offsets();

      std::vector<int>* pair_point_uid_0 = nullptr;
      std::vector<int>* pair_point_uid_1 = nullptr;
      std::vector<int>* pair_cell_uid_0 = nullptr;
      std::vector<int>* pair_cell_uid_1 = nullptr;
      if (write_unique_ids)
      {
        pair_point_uid_0 = &(visualization_data.get_point_data<int>("uid_0_pair_beam_id"));
        pair_point_uid_1 = &(visualization_data.get_point_data<int>("uid_1_pair_solid_id"));
        pair_cell_uid_0 = &(visualization_data.get_cell_data<int>("uid_0_pair_beam_id"));
        pair_cell_uid_1 = &(visualization_data.get_cell_data<int>("uid_1_pair_solid_id"));
      }

      for (unsigned int i_curve_segment = 0; i_curve_segment < mortar_segments; i_curve_segment++)
      {
        for (unsigned int i_circumference_segment = 0;
            i_circumference_segment < mortar_segments_circumference; i_circumference_segment++)
        {
          for (const auto& [offset_axial, offset_circumference] :
              std::initializer_list<std::pair<int, int>>{
                  {0, 0}, {2, 0}, {2, 2}, {0, 2}, {1, 0}, {2, 1}, {1, 2}, {0, 1}, {1, 1}})
          {
            Core::LinAlg::Matrix<2, 1> xi;
            xi(0) =
                -1.0 + (2 * i_curve_segment + offset_axial) * 2.0 / (2.0 * (double)mortar_segments);
            xi(1) = (2.0 * i_circumference_segment + offset_circumference) * 2.0 * M_PI /
                    (2.0 * (double)mortar_segments_circumference);

            get_triad_at_xi_double(xi(0), triad_ref, true);
            get_triad_at_xi_double(xi(0), triad, false);

            cross_section_vector_material(0) = 0.0;
            cross_section_vector_material(1) = cos(xi(1));
            cross_section_vector_material(2) = sin(xi(1));
            cross_section_vector_material.scale(beam_cross_section_radius);

            cross_section_vector_ref.multiply(triad_ref, cross_section_vector_material);
            cross_section_vector_current.multiply(triad, cross_section_vector_material);

            GeometryPair::evaluate_position<Beam>(xi(0), this->ele1posref_, X);
            GeometryPair::evaluate_position<Beam>(xi(0), this->ele1pos_, r);

            X += cross_section_vector_ref;
            r += cross_section_vector_current;

            u = r;
            u -= X;

            GeometryPair::evaluate_position<Mortar>(xi, element_data_lambda, lambda_interpolated);

            // Add to output data.
            for (unsigned int dim = 0; dim < 3; dim++)
            {
              point_coordinates.push_back(X(dim));
              displacement.push_back(u(dim));
              lambda_vis.push_back(lambda_interpolated(dim));
            }
          }

          // Add the cell for this segment (bi-quadratic quad).
          cell_types.push_back(28);
          cell_offsets.push_back(point_coordinates.size() / 3);

          if (write_unique_ids)
          {
            // Since we create this output once for all solids that are interacting with this beam,
            // we don't need the solid element GID
            pair_cell_uid_0->push_back(this->element1()->id());
            pair_cell_uid_1->push_back(-1);
            for (unsigned int i_point = 0; i_point < 9; i_point++)
            {
              pair_point_uid_0->push_back(this->element1()->id());
              pair_point_uid_1->push_back(-1);
            }
          }
        }
      }
    }
  }
}

/**
 *
 */
template <typename Beam, typename Solid, typename Mortar>
void BeamInteraction::BeamToSolidVolumeMeshtyingPair2D3DMortar<Beam, Solid,
    Mortar>::get_triad_at_xi_double(const double xi, Core::LinAlg::Matrix<3, 3, double>& triad,
    const bool reference) const
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
 *
 */
std::shared_ptr<BeamInteraction::BeamContactPair>
BeamInteraction::create_beam_to_solid_volume_pair_mortar_cross_section(
    const Core::FE::CellType shape,
    const Inpar::BeamToSolid::BeamToSolidMortarShapefunctions mortar_shape_function,
    const int n_fourier_modes)
{
  switch (mortar_shape_function)
  {
    case Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::line2:
    {
      switch (n_fourier_modes)
      {
        case 0:
          return std::make_shared<BeamToSolidVolumeMeshtyingPair2D3DMortar<GeometryPair::t_hermite,
              GeometryPair::t_hex8, GeometryPair::t_line2_fourier_0>>();
        case 1:
          return std::make_shared<BeamToSolidVolumeMeshtyingPair2D3DMortar<GeometryPair::t_hermite,
              GeometryPair::t_hex8, GeometryPair::t_line2_fourier_1>>();
        case 2:
          return std::make_shared<BeamToSolidVolumeMeshtyingPair2D3DMortar<GeometryPair::t_hermite,
              GeometryPair::t_hex8, GeometryPair::t_line2_fourier_2>>();
        case 3:
          return std::make_shared<BeamToSolidVolumeMeshtyingPair2D3DMortar<GeometryPair::t_hermite,
              GeometryPair::t_hex8, GeometryPair::t_line2_fourier_3>>();
        default:
          FOUR_C_THROW("Got wrong number of fourier mortar modes {}.", n_fourier_modes);
      }
    }
    default:
      FOUR_C_THROW("Wrong mortar shape function.");
  }
}


FOUR_C_NAMESPACE_CLOSE
