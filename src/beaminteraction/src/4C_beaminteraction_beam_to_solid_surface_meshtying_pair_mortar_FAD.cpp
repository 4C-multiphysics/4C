// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_surface_meshtying_pair_mortar_FAD.hpp"

#include "4C_beam3_reissner.hpp"
#include "4C_beam3_triad_interpolation_local_rotation_vectors.hpp"
#include "4C_beaminteraction_beam_to_solid_mortar_manager.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_meshtying_params.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_element_faces.hpp"
#include "4C_geometry_pair_line_to_surface.hpp"
#include "4C_inpar_beam_to_solid.hpp"
#include "4C_inpar_geometry_pair.hpp"
#include "4C_linalg_fevector.hpp"

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface, typename Mortar>
BeamInteraction::BeamToSolidSurfaceMeshtyingPairMortarFAD<ScalarType, Beam, Surface,
    Mortar>::BeamToSolidSurfaceMeshtyingPairMortarFAD()
    : base_class()
{
  // Empty constructor.
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface, typename Mortar>
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairMortarFAD<ScalarType, Beam, Surface,
    Mortar>::evaluate_and_assemble(const Core::FE::Discretization& discret,
    const BeamToSolidMortarManager* mortar_manager,
    const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
    const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
    const Core::LinAlg::Vector<double>& global_lambda,
    const Core::LinAlg::Vector<double>& displacement_vector)
{
  // Call Evaluate on the geometry Pair. Only do this once for meshtying.
  if (!this->meshtying_is_evaluated_)
  {
    this->cast_geometry_pair()->evaluate(this->ele1posref_,
        this->face_element_->get_face_reference_element_data(), this->line_to_3D_segments_);
    this->meshtying_is_evaluated_ = true;
  }

  // If there are no intersection segments, return no contact status.
  if (this->line_to_3D_segments_.size() == 0) return;

  // Get the positional Lagrange multipliers for this pair.
  const auto& [lambda_gid_pos, _] = mortar_manager->location_vector(*this);
  std::vector<double> local_lambda_pos = Core::FE::extract_values(global_lambda, lambda_gid_pos);
  auto q_lambda = GeometryPair::InitializeElementData<Mortar, double>::initialize(nullptr);
  q_lambda.element_position_ =
      Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>(local_lambda_pos.data());

  // Initialize variables for local values.
  Core::LinAlg::Matrix<3, 1, ScalarType> coupling_vector(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, double> lambda(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, double> dr_beam_ref(Core::LinAlg::Initialization::zero);
  ScalarType potential = 0.0;

  // Initialize scalar variables.
  double segment_jacobian = 0.0;
  double beam_segmentation_factor = 0.0;

  // Loop over segments to evaluate the coupling potential.
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

      // Get the Gauss point contribution to the coupling potential.
      coupling_vector = this->evaluate_coupling(projected_gauss_point);
      GeometryPair::evaluate_position<Mortar>(projected_gauss_point.get_eta(), q_lambda, lambda);
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        potential += coupling_vector(i_dim) * lambda(i_dim) *
                     projected_gauss_point.get_gauss_weight() * segment_jacobian;
    }
  }

  // Get the pair GIDs.
  const auto pair_gid =
      get_beam_to_surface_pair_gid_combined<Beam>(discret, *this->element1(), *this->face_element_);

  // Add the terms to the global stiffness matrix.
  if (stiffness_matrix != nullptr)
    for (unsigned int i_dof = 0; i_dof < pair_gid.size(); i_dof++)
      for (unsigned int j_dof = 0; j_dof < pair_gid.size(); j_dof++)
        stiffness_matrix->fe_assemble(Core::FADUtils::cast_to_double(potential.dx(i_dof).dx(j_dof)),
            pair_gid[i_dof], pair_gid[j_dof]);
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface, typename Mortar>
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairMortarFAD<ScalarType, Beam, Surface,
    Mortar>::evaluate_and_assemble_mortar_contributions(const Core::FE::Discretization& discret,
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
  // Call Evaluate on the geometry Pair. Only do this once for meshtying.
  if (!this->meshtying_is_evaluated_)
  {
    this->cast_geometry_pair()->evaluate(this->ele1posref_,
        this->face_element_->get_face_reference_element_data(), this->line_to_3D_segments_);
    this->meshtying_is_evaluated_ = true;
  }

  // If there are no intersection segments, return no contact status.
  if (this->line_to_3D_segments_.size() == 0) return;

  // Initialize variables for local values.
  Core::LinAlg::Matrix<3, 1, ScalarType> coupling_vector(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, ScalarType> constraint_vector(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> local_kappa(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, double> dr_beam_ref(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, Mortar::n_nodes_ * Mortar::n_val_, double> N_mortar(
      Core::LinAlg::Initialization::zero);

  // Initialize scalar variables.
  double segment_jacobian = 0.0;
  double beam_segmentation_factor = 0.0;

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

      // Get the mortar shape functions.
      GeometryPair::EvaluateShapeFunction<Mortar>::evaluate(
          N_mortar, projected_gauss_point.get_eta());

      // Fill in the local mortar scaling vector kappa.
      for (unsigned int i_mortar_node = 0; i_mortar_node < Mortar::n_nodes_; i_mortar_node++)
        for (unsigned int i_mortar_val = 0; i_mortar_val < Mortar::n_val_; i_mortar_val++)
          for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
            local_kappa(i_mortar_node * Mortar::n_val_ * 3 + i_mortar_val * 3 + i_dim) +=
                N_mortar(i_mortar_node * Mortar::n_val_ + i_mortar_val) *
                projected_gauss_point.get_gauss_weight() * segment_jacobian;

      // Get the constraint vector. This is the coupling potentials variation w.r.t the discrete
      // Lagrange multiplier DOFs.
      coupling_vector = this->evaluate_coupling(projected_gauss_point);
      for (unsigned int i_mortar_node = 0; i_mortar_node < Mortar::n_nodes_; i_mortar_node++)
        for (unsigned int i_mortar_val = 0; i_mortar_val < Mortar::n_val_; i_mortar_val++)
          for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
            constraint_vector(i_mortar_node * Mortar::n_val_ * 3 + i_mortar_val * 3 + i_dim) +=
                N_mortar(i_mortar_node * Mortar::n_val_ + i_mortar_val) * coupling_vector(i_dim) *
                projected_gauss_point.get_gauss_weight() * segment_jacobian;
    }
  }

  // Get the pair GIDs.
  const auto [beam_centerline_gid, patch_gid] =
      get_beam_to_surface_pair_gid<Beam>(discret, *this->element1(), *this->face_element_);

  // Get the Lagrange multiplier GIDs.
  const auto& [lambda_gid_pos, _] = mortar_manager->location_vector(*this);

  // Assemble into the matrices related to beam DOFs.
  for (unsigned int i_lambda = 0; i_lambda < Mortar::n_dof_; i_lambda++)
    for (unsigned int i_beam = 0; i_beam < Beam::n_dof_; i_beam++)
    {
      const double val = Core::FADUtils::cast_to_double(constraint_vector(i_lambda).dx(i_beam));
      global_constraint_lin_beam.fe_assemble(
          val, lambda_gid_pos[i_lambda], beam_centerline_gid(i_beam));
      global_force_beam_lin_lambda.fe_assemble(
          val, beam_centerline_gid(i_beam), lambda_gid_pos[i_lambda]);
    }

  // Assemble into the matrices related to solid DOFs.
  for (unsigned int i_lambda = 0; i_lambda < Mortar::n_dof_; i_lambda++)
    for (unsigned int i_patch = 0; i_patch < patch_gid.size(); i_patch++)
    {
      const double val =
          Core::FADUtils::cast_to_double(constraint_vector(i_lambda).dx(Beam::n_dof_ + i_patch));
      global_constraint_lin_solid.fe_assemble(val, lambda_gid_pos[i_lambda], patch_gid[i_patch]);
      global_force_solid_lin_lambda.fe_assemble(val, patch_gid[i_patch], lambda_gid_pos[i_lambda]);
    }

  // Assemble into global coupling vector.
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> constraint_vector_double =
      Core::FADUtils::cast_to_double(constraint_vector);
  global_constraint.sum_into_global_values(
      lambda_gid_pos.size(), lambda_gid_pos.data(), constraint_vector_double.data());

  // Assemble into global kappa vector.
  global_kappa.sum_into_global_values(
      lambda_gid_pos.size(), lambda_gid_pos.data(), local_kappa.data());

  // Assemble into global lambda active vector.
  local_kappa.put_scalar(1.0);
  global_lambda_active.sum_into_global_values(
      lambda_gid_pos.size(), lambda_gid_pos.data(), local_kappa.data());
}


/**
 *
 */
template <typename Surface, typename ScalarTypeBasis>
void get_surface_basis(const Core::LinAlg::Matrix<3, 1, double>& xi,
    const GeometryPair::ElementData<Surface, ScalarTypeBasis>& q_solid,
    Core::LinAlg::Matrix<3, 3, ScalarTypeBasis>& surface_basis)
{
  // Calculate surface basis vectors in the surface plane.
  Core::LinAlg::Matrix<3, 2, ScalarTypeBasis> dr_surf(Core::LinAlg::Initialization::zero);
  GeometryPair::evaluate_position_derivative1<Surface>(xi, q_solid, dr_surf);

  Core::LinAlg::Matrix<3, 1, ScalarTypeBasis> dr_surf_0;
  Core::LinAlg::Matrix<3, 1, ScalarTypeBasis> dr_surf_1;
  for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
  {
    dr_surf_0(i_dir) = dr_surf(i_dir, 0);
    dr_surf_1(i_dir) = dr_surf(i_dir, 1);
  }

  // Calculate normal on the basis vectors.
  Core::LinAlg::Matrix<3, 1, ScalarTypeBasis> element_surface_normal;
  element_surface_normal.cross_product(dr_surf_0, dr_surf_1);
  element_surface_normal.scale(1.0 / Core::FADUtils::vector_norm(element_surface_normal));

  // Put the new basis vectors in a matrix.
  for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
  {
    surface_basis(i_dir, 0) = dr_surf(i_dir, 0);
    surface_basis(i_dir, 1) = dr_surf(i_dir, 1);
    surface_basis(i_dir, 2) = element_surface_normal(i_dir);
  }
}

/**
 *
 */
template <typename Surface, typename ScalarTypeRotVec>
void get_surface_rotation_vector_averaged(const Core::LinAlg::Matrix<3, 1, double>& xi,
    const GeometryPair::ElementData<Surface, double>& q_solid_ref,
    const GeometryPair::ElementData<Surface, ScalarTypeRotVec>& q_solid,
    const Core::LinAlg::Matrix<4, 1, double>& quaternion_beam_ref,
    Core::LinAlg::Matrix<3, 1, ScalarTypeRotVec>& psi_solid)
{
  // Get beam basis vectors in reference configuration.
  Core::LinAlg::Matrix<3, 3, double> triad_beam_ref(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::quaterniontotriad(quaternion_beam_ref, triad_beam_ref);

  // Calculate surface basis coordinate transformation matrix.
  Core::LinAlg::Matrix<3, 3, double> surface_basis_ref_inverse;
  get_surface_basis<Surface>(xi, q_solid_ref, surface_basis_ref_inverse);
  Core::LinAlg::inverse(surface_basis_ref_inverse);

  // Calculate the current surface basis vectors.
  Core::LinAlg::Matrix<3, 3, ScalarTypeRotVec> surface_basis_current;
  get_surface_basis<Surface>(xi, q_solid, surface_basis_current);

  // Calculate the in plane surface deformation gradient.
  Core::LinAlg::Matrix<3, 3, ScalarTypeRotVec> surface_basis_ref_inverse_scalar_type;
  for (unsigned int i_row = 0; i_row < 3; i_row++)
    for (unsigned int i_col = 0; i_col < 3; i_col++)
      surface_basis_ref_inverse_scalar_type(i_row, i_col) = surface_basis_ref_inverse(i_row, i_col);
  Core::LinAlg::Matrix<3, 3, ScalarTypeRotVec> surface_F;
  surface_F.multiply(surface_basis_current, surface_basis_ref_inverse_scalar_type);

  // Get the solid rotation vector from the deformation gradient via construction in the cross
  // section plane.
  BeamInteraction::get_solid_rotation_vector_deformation_gradient_3d_general_in_cross_section_plane(
      surface_F, triad_beam_ref, psi_solid);
}

/**
 *
 */
template <typename Surface, typename ScalarTypeRotVec>
void get_surface_rotation_vector_cross_section_director(
    const Core::LinAlg::Matrix<3, 1, double>& xi,
    const GeometryPair::ElementData<Surface, double>& q_solid_ref,
    const GeometryPair::ElementData<Surface, ScalarTypeRotVec>& q_solid,
    const Core::LinAlg::Matrix<4, 1, double>& quaternion_beam_ref,
    Core::LinAlg::Matrix<3, 1, ScalarTypeRotVec>& psi_solid)
{
  // Get beam basis vectors in reference configuration.
  Core::LinAlg::Matrix<3, 3, double> triad_beam_ref(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::quaterniontotriad(quaternion_beam_ref, triad_beam_ref);

  // Get the surface basis vectors in the reference configuration.
  Core::LinAlg::Matrix<3, 3, double> surface_basis_ref;
  get_surface_basis<Surface>(xi, q_solid_ref, surface_basis_ref);

  // Get the surface basis vectors in the current configuration.
  Core::LinAlg::Matrix<3, 3, ScalarTypeRotVec> surface_basis_current;
  get_surface_basis<Surface>(xi, q_solid, surface_basis_current);

  // Get the surface material director (the intersection between the beam cross-section and the
  // surface tangent plane).
  Core::LinAlg::Matrix<3, 1, double> surface_normal_ref;
  Core::LinAlg::Matrix<3, 1, double> beam_cross_section_normal_ref;
  for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
  {
    surface_normal_ref(i_dim) = surface_basis_ref(i_dim, 2);
    beam_cross_section_normal_ref(i_dim) = triad_beam_ref(i_dim, 0);
  }
  Core::LinAlg::Matrix<3, 1, double> surface_material_director_ref;
  surface_material_director_ref.cross_product(surface_normal_ref, beam_cross_section_normal_ref);
  surface_material_director_ref.scale(
      1.0 / Core::FADUtils::vector_norm(surface_material_director_ref));

  // Get the reference triad of the surface.
  Core::LinAlg::Matrix<3, 1, double> surface_material_director_perpendicular_ref;
  surface_material_director_perpendicular_ref.cross_product(
      surface_material_director_ref, surface_normal_ref);
  Core::LinAlg::Matrix<3, 3, double> surface_triad_ref;
  for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
  {
    surface_triad_ref(i_dim, 0) = surface_material_director_ref(i_dim);
    surface_triad_ref(i_dim, 1) = surface_normal_ref(i_dim);
    surface_triad_ref(i_dim, 2) = surface_material_director_perpendicular_ref(i_dim);
  }

  // Get the offset of the reference triad, so it matches the beam reference triad.
  Core::LinAlg::Matrix<3, 3, double> surface_triad_offset;
  surface_triad_offset.multiply_tn(surface_triad_ref, triad_beam_ref);

  // Calculate the in plane surface deformation gradient.
  Core::LinAlg::Matrix<3, 3, double> surface_basis_ref_inverse;
  surface_basis_ref_inverse = surface_basis_ref;
  Core::LinAlg::inverse(surface_basis_ref_inverse);
  Core::LinAlg::Matrix<3, 3, ScalarTypeRotVec> surface_basis_ref_inverse_scalar_type;
  for (unsigned int i_row = 0; i_row < 3; i_row++)
    for (unsigned int i_col = 0; i_col < 3; i_col++)
      surface_basis_ref_inverse_scalar_type(i_row, i_col) = surface_basis_ref_inverse(i_row, i_col);
  Core::LinAlg::Matrix<3, 3, ScalarTypeRotVec> surface_F;
  surface_F.multiply(surface_basis_current, surface_basis_ref_inverse_scalar_type);

  // Get the current material director.
  Core::LinAlg::Matrix<3, 1, ScalarTypeRotVec> surface_material_director_ref_fad;
  for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
    surface_material_director_ref_fad(i_dim) = surface_material_director_ref(i_dim);
  Core::LinAlg::Matrix<3, 1, ScalarTypeRotVec> surface_material_director_current;
  surface_material_director_current.multiply(surface_F, surface_material_director_ref_fad);
  surface_material_director_current.scale(
      1.0 / Core::FADUtils::vector_norm(surface_material_director_current));

  // Get the current triad of the surface.
  Core::LinAlg::Matrix<3, 1, ScalarTypeRotVec> surface_normal_current;
  for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
    surface_normal_current(i_dim) = surface_basis_current(i_dim, 2);
  Core::LinAlg::Matrix<3, 1, ScalarTypeRotVec> surface_material_director_perpendicular_current;
  surface_material_director_perpendicular_current.cross_product(
      surface_material_director_current, surface_normal_current);
  Core::LinAlg::Matrix<3, 3, ScalarTypeRotVec> surface_triad_current;
  for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
  {
    surface_triad_current(i_dim, 0) = surface_material_director_current(i_dim);
    surface_triad_current(i_dim, 1) = surface_normal_current(i_dim);
    surface_triad_current(i_dim, 2) = surface_material_director_perpendicular_current(i_dim);
  }

  // Add the offset to the surface triad.
  Core::LinAlg::Matrix<3, 3, ScalarTypeRotVec> surface_triad_offset_fad;
  for (unsigned int i_row = 0; i_row < 3; i_row++)
    for (unsigned int i_col = 0; i_col < 3; i_col++)
      surface_triad_offset_fad(i_row, i_col) = surface_triad_offset(i_row, i_col);
  Core::LinAlg::Matrix<3, 3, ScalarTypeRotVec> surface_triad_current_with_offset;
  surface_triad_current_with_offset.multiply(surface_triad_current, surface_triad_offset_fad);

  // Get the rotation angle.
  Core::LinAlg::Matrix<4, 1, ScalarTypeRotVec> rot_quat;
  Core::LargeRotations::triadtoquaternion(surface_triad_current_with_offset, rot_quat);
  Core::LargeRotations::quaterniontoangle(rot_quat, psi_solid);

#ifdef FOUR_C_ENABLE_ASSERTIONS
  Core::LinAlg::Matrix<3, 1, ScalarTypeRotVec> current_normal;
  for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
    current_normal(i_dim) = surface_basis_current(i_dim, 2);
  if (abs(surface_material_director_current.dot(current_normal)) > 1e-10)
    FOUR_C_THROW("The current material director has to lie within the surface tangent plane.");
#endif
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface, typename Mortar>
template <typename ScalarTypeRotVec>
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairMortarRotationFAD<ScalarType, Beam, Surface,
    Mortar>::get_surface_rotation_vector(const Core::LinAlg::Matrix<3, 1, double>& xi,
    const GeometryPair::ElementData<Surface, double>& q_solid_ref,
    const GeometryPair::ElementData<Surface, ScalarTypeRotVec>& q_solid,
    const Core::LinAlg::Matrix<4, 1, double>& quaternion_beam_ref,
    const Inpar::BeamToSolid::BeamToSolidSurfaceRotationCoupling surface_triad_type,
    Core::LinAlg::Matrix<3, 1, ScalarTypeRotVec>& psi_solid) const
{
  switch (surface_triad_type)
  {
    case Inpar::BeamToSolid::BeamToSolidSurfaceRotationCoupling::averaged:
      get_surface_rotation_vector_averaged<Surface>(
          xi, q_solid_ref, q_solid, quaternion_beam_ref, psi_solid);
      break;
    case Inpar::BeamToSolid::BeamToSolidSurfaceRotationCoupling::surface_cross_section_director:
      get_surface_rotation_vector_cross_section_director<Surface>(
          xi, q_solid_ref, q_solid, quaternion_beam_ref, psi_solid);
      break;
    default:
      FOUR_C_THROW("Please supply a suitable solid triad construction.");
      break;
  }
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface, typename Mortar>
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairMortarRotationFAD<ScalarType, Beam, Surface,
    Mortar>::evaluate_and_assemble(const Core::FE::Discretization& discret,
    const BeamToSolidMortarManager* mortar_manager,
    const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
    const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
    const Core::LinAlg::Vector<double>& global_lambda,
    const Core::LinAlg::Vector<double>& displacement_vector)
{
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

  // Set the FAD variables for the solid DOFs. For the terms calculated here we need second
  // order derivatives.
  GeometryPair::ElementData<Surface, scalar_type_rot_2nd> q_surface;
  q_surface.shape_function_data_ =
      this->face_element_->get_face_element_data().shape_function_data_;
  for (unsigned int i_surface = 0; i_surface < Surface::n_dof_; i_surface++)
    q_surface.element_position_(i_surface) =
        Core::FADUtils::HigherOrderFadValue<scalar_type_rot_2nd>::apply(3 + Surface::n_dof_,
            3 + i_surface,
            Core::FADUtils::cast_to_double(
                this->face_element_->get_face_element_data().element_position_(i_surface)));

  // Get the rotational Lagrange multipliers for this pair.
  const auto& [_, lambda_gid_rot] = mortar_manager->location_vector(*this);
  std::vector<double> lambda_rot_double = Core::FE::extract_values(global_lambda, lambda_gid_rot);
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> lambda_rot;
  for (unsigned int i_dof = 0; i_dof < Mortar::n_dof_; i_dof++)
    lambda_rot(i_dof) = lambda_rot_double[i_dof];

  // Get the type of surface triad construction.
  const auto surface_triad_type =
      this->params()->beam_to_solid_surface_meshtying_params()->get_surface_triad_construction();

  // Initialize local matrices.
  Core::LinAlg::Matrix<n_dof_rot_, n_dof_rot_, double> local_stiff_BB(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<n_dof_rot_, Surface::n_dof_, double> local_stiff_BS(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Surface::n_dof_, n_dof_rot_, double> local_stiff_SB(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Surface::n_dof_, Surface::n_dof_, double> local_stiff_SS(
      Core::LinAlg::Initialization::zero);

  // Evaluate the pair wise terms.
  {
    // Initialize variables.
    Core::LinAlg::Matrix<3, 1, double> dr_beam_ref;
    Core::LinAlg::Matrix<4, 1, double> quaternion_beam_double;
    Core::LinAlg::Matrix<3, 1, double> psi_beam_double;
    Core::LinAlg::Matrix<3, 1, scalar_type_rot_1st> psi_beam;
    Core::LinAlg::Matrix<3, 1, scalar_type_rot_2nd> psi_surface;
    Core::LinAlg::Matrix<3, 1, scalar_type_rot_1st> psi_surface_val;
    Core::LinAlg::Matrix<3, 1, scalar_type_rot_1st> psi_rel;
    Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_beam;
    Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_beam_inv;
    Core::LinAlg::Matrix<4, 1, double> quaternion_beam_ref;
    Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_surface;
    Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_rel;
    Core::LinAlg::Matrix<3, 3, double> T_beam;
    Core::LinAlg::Matrix<3, 3, scalar_type_rot_1st> T_surface;
    Core::LinAlg::Matrix<3, 3, scalar_type_rot_1st> T_surface_inv;
    Core::LinAlg::Matrix<3, 3, scalar_type_rot_1st> T_rel;

    Core::LinAlg::Matrix<Mortar::n_nodes_, 1, double> lambda_shape_functions;
    Core::LinAlg::Matrix<3, Mortar::n_dof_, scalar_type_rot_1st> lambda_shape_functions_full(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::SerialDenseVector L_i(3);
    Core::LinAlg::Matrix<3, n_dof_rot_, scalar_type_rot_1st> L_full(
        Core::LinAlg::Initialization::zero);
    std::vector<Core::LinAlg::Matrix<3, 3, double>> I_beam_tilde;
    Core::LinAlg::Matrix<3, n_dof_rot_, double> I_beam_tilde_full;
    Core::LinAlg::Matrix<3, n_dof_rot_, double> T_beam_times_I_beam_tilde_full;
    Core::LinAlg::Matrix<3, Mortar::n_dof_, scalar_type_rot_1st> T_rel_tr_times_lambda_shape;
    Core::LinAlg::Matrix<3, Mortar::n_dof_, scalar_type_rot_1st>
        T_surface_mtr_times_T_rel_tr_times_lambda_shape;
    Core::LinAlg::Matrix<n_dof_rot_, Mortar::n_dof_, scalar_type_rot_1st> d_fb_d_lambda_gp;
    Core::LinAlg::Matrix<Surface::n_dof_, Mortar::n_dof_, scalar_type_rot_1st> d_fs_d_lambda_gp;
    Core::LinAlg::Matrix<3, Surface::n_dof_, scalar_type_rot_1st> d_psi_surface_d_q_surface;
    Core::LinAlg::Matrix<Mortar::n_dof_, 3, double> d_g_d_psi_beam;
    Core::LinAlg::Matrix<Mortar::n_dof_, Surface::n_dof_, double> d_g_d_q_surface;
    Core::LinAlg::Matrix<n_dof_rot_, 1, scalar_type_rot_1st> f_beam;
    Core::LinAlg::Matrix<Surface::n_dof_, 1, scalar_type_rot_1st> f_surface;
    Core::LinAlg::Matrix<n_dof_rot_, 3, double> d_f_beam_d_phi;
    Core::LinAlg::Matrix<Surface::n_dof_, 3, double> d_f_surface_d_phi;
    Core::LinAlg::Matrix<n_dof_rot_, n_dof_rot_, double>
        d_f_beam_d_phi_times_T_beam_times_I_beam_tilde_full;
    Core::LinAlg::Matrix<Surface::n_dof_, n_dof_rot_, double>
        d_f_surface_d_phi_times_T_beam_times_I_beam_tilde_full;

    // Initialize scalar variables.
    double segment_jacobian = 0.0;
    double beam_segmentation_factor = 0.0;

    // Calculate the meshtying forces.
    // Loop over segments.
    for (unsigned int i_segment = 0; i_segment < this->line_to_3D_segments_.size(); i_segment++)
    {
      // Factor to account for a segment length not from -1 to 1.
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
              3 + Surface::n_dof_, i_dim, psi_beam_double(i_dim));
        Core::LargeRotations::angletoquaternion(psi_beam, quaternion_beam);
        quaternion_beam_inv = Core::LargeRotations::inversequaternion(quaternion_beam);

        // Get the surface rotation vector.
        ref_triad_interpolation_scheme.get_interpolated_quaternion_at_xi(
            quaternion_beam_ref, projected_gauss_point.get_eta());
        get_surface_rotation_vector(projected_gauss_point.get_xi(),
            this->face_element_->get_face_reference_element_data(), q_surface, quaternion_beam_ref,
            surface_triad_type, psi_surface);
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          psi_surface_val(i_dim) = psi_surface(i_dim).val();
        Core::LargeRotations::angletoquaternion(psi_surface_val, quaternion_surface);

        // Calculate the relative rotation vector.
        Core::LargeRotations::quaternionproduct(
            quaternion_beam_inv, quaternion_surface, quaternion_rel);
        Core::LargeRotations::quaterniontoangle(quaternion_rel, psi_rel);

        // Calculate the transformation matrices.
        T_rel = Core::LargeRotations::tmatrix(psi_rel);
        T_beam = Core::LargeRotations::tmatrix(Core::FADUtils::cast_to_double(psi_beam));
        T_surface = Core::LargeRotations::tmatrix(psi_surface_val);
        T_surface_inv = T_surface;
        Core::LinAlg::inverse(T_surface_inv);

        // Evaluate mortar shape functions.
        GeometryPair::EvaluateShapeFunction<Mortar>::evaluate(
            lambda_shape_functions, projected_gauss_point.get_eta());
        for (unsigned int i_node = 0; i_node < Mortar::n_nodes_; i_node++)
          for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
            lambda_shape_functions_full(i_dim, 3 * i_node + i_dim) = lambda_shape_functions(i_node);

        // Get the shape functions for the interpolation of the beam rotations. This is currently
        // only implemented for 2nd order Lagrange interpolation (Beam3rHerm2Line3).
        const unsigned int n_nodes_rot = 3;
        Core::FE::shape_function_1d(
            L_i, projected_gauss_point.get_eta(), Core::FE::CellType::line3);
        for (unsigned int i_node = 0; i_node < n_nodes_rot; i_node++)
          for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
            L_full(i_dim, 3 * i_node + i_dim) = L_i(i_node);

        triad_interpolation_scheme.get_nodal_generalized_rotation_interpolation_matrices_at_xi(
            I_beam_tilde, projected_gauss_point.get_eta());
        for (unsigned int i_node = 0; i_node < n_nodes_rot; i_node++)
          for (unsigned int i_dim_0 = 0; i_dim_0 < 3; i_dim_0++)
            for (unsigned int i_dim_1 = 0; i_dim_1 < 3; i_dim_1++)
              I_beam_tilde_full(i_dim_0, i_node * 3 + i_dim_1) =
                  I_beam_tilde[i_node](i_dim_0, i_dim_1);

        // Solid angle derived w.r.t. the surface DOFs.
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          for (unsigned int i_surface = 0; i_surface < Surface::n_dof_; i_surface++)
            d_psi_surface_d_q_surface(i_dim, i_surface) = psi_surface(i_dim).dx(3 + i_surface);

        // Calculate the force terms derived w.r.t. the Lagrange multipliers.
        T_rel_tr_times_lambda_shape.multiply_tn(T_rel, lambda_shape_functions_full);
        d_fb_d_lambda_gp.multiply_tn(L_full, T_rel_tr_times_lambda_shape);
        d_fb_d_lambda_gp.scale(-1.0 * projected_gauss_point.get_gauss_weight() * segment_jacobian);

        T_surface_mtr_times_T_rel_tr_times_lambda_shape.multiply_tn(
            T_surface_inv, T_rel_tr_times_lambda_shape);
        d_fs_d_lambda_gp.multiply_tn(
            d_psi_surface_d_q_surface, T_surface_mtr_times_T_rel_tr_times_lambda_shape);
        d_fs_d_lambda_gp.scale(projected_gauss_point.get_gauss_weight() * segment_jacobian);

        // Calculate the force vectors.
        f_beam.put_scalar(0.0);
        for (unsigned int i_row = 0; i_row < n_dof_rot_; i_row++)
          for (unsigned int i_col = 0; i_col < Mortar::n_dof_; i_col++)
            f_beam(i_row) += d_fb_d_lambda_gp(i_row, i_col) * lambda_rot(i_col);
        f_surface.put_scalar(0.0);
        for (unsigned int i_row = 0; i_row < Surface::n_dof_; i_row++)
          for (unsigned int i_col = 0; i_col < Mortar::n_dof_; i_col++)
            f_surface(i_row) += d_fs_d_lambda_gp(i_row, i_col) * lambda_rot(i_col);

        // Derivatives of the force vectors.
        for (unsigned int i_row = 0; i_row < n_dof_rot_; i_row++)
          for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
            d_f_beam_d_phi(i_row, i_dim) = f_beam(i_row).dx(i_dim);
        for (unsigned int i_row = 0; i_row < Surface::n_dof_; i_row++)
          for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
            d_f_surface_d_phi(i_row, i_dim) = f_surface(i_row).dx(i_dim);

        T_beam_times_I_beam_tilde_full.multiply(T_beam, I_beam_tilde_full);
        d_f_beam_d_phi_times_T_beam_times_I_beam_tilde_full.multiply(
            d_f_beam_d_phi, T_beam_times_I_beam_tilde_full);
        d_f_surface_d_phi_times_T_beam_times_I_beam_tilde_full.multiply(
            d_f_surface_d_phi, T_beam_times_I_beam_tilde_full);

        // Add to output matrices and vector.
        local_stiff_BB += d_f_beam_d_phi_times_T_beam_times_I_beam_tilde_full;
        for (unsigned int i_beam = 0; i_beam < n_dof_rot_; i_beam++)
          for (unsigned int j_surface = 0; j_surface < Surface::n_dof_; j_surface++)
            local_stiff_BS(i_beam, j_surface) += f_beam(i_beam).dx(3 + j_surface);
        local_stiff_SB += d_f_surface_d_phi_times_T_beam_times_I_beam_tilde_full;
        for (unsigned int i_surface = 0; i_surface < Surface::n_dof_; i_surface++)
          for (unsigned int j_surface = 0; j_surface < Surface::n_dof_; j_surface++)
            local_stiff_SS(i_surface, j_surface) += f_surface(i_surface).dx(3 + j_surface);
      }
    }
  }

  // Get the rotational GIDs of the surface and beam.
  std::vector<int> gid_surface;
  Core::LinAlg::Matrix<n_dof_rot_, 1, int> gid_rot;
  get_pair_rotational_gid(discret, gid_surface, gid_rot);

  // Assemble into global matrix.
  for (unsigned int i_dof_beam = 0; i_dof_beam < n_dof_rot_; i_dof_beam++)
  {
    for (unsigned int j_dof_beam = 0; j_dof_beam < n_dof_rot_; j_dof_beam++)
      stiffness_matrix->fe_assemble(
          local_stiff_BB(i_dof_beam, j_dof_beam), gid_rot(i_dof_beam), gid_rot(j_dof_beam));
    for (unsigned int j_dof_surface = 0; j_dof_surface < Surface::n_dof_; j_dof_surface++)
      stiffness_matrix->fe_assemble(local_stiff_BS(i_dof_beam, j_dof_surface), gid_rot(i_dof_beam),
          gid_surface[j_dof_surface]);
  }
  for (unsigned int i_dof_surface = 0; i_dof_surface < Surface::n_dof_; i_dof_surface++)
  {
    for (unsigned int j_dof_beam = 0; j_dof_beam < n_dof_rot_; j_dof_beam++)
      stiffness_matrix->fe_assemble(local_stiff_SB(i_dof_surface, j_dof_beam),
          gid_surface[i_dof_surface], gid_rot(j_dof_beam));
    for (unsigned int j_dof_surface = 0; j_dof_surface < Surface::n_dof_; j_dof_surface++)
      stiffness_matrix->fe_assemble(local_stiff_SS(i_dof_surface, j_dof_surface),
          gid_surface[i_dof_surface], gid_surface[j_dof_surface]);
  }
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface, typename Mortar>
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairMortarRotationFAD<ScalarType, Beam, Surface,
    Mortar>::evaluate_and_assemble_mortar_contributions(const Core::FE::Discretization& discret,
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

  // Set the FAD variables for the surface DOFs. For the terms calculated here we only need first
  // order derivatives.
  GeometryPair::ElementData<Surface, scalar_type_rot_1st> q_surface;
  q_surface.shape_function_data_ =
      this->face_element_->get_face_element_data().shape_function_data_;
  for (unsigned int i_surface = 0; i_surface < Surface::n_dof_; i_surface++)
    q_surface.element_position_(i_surface) =
        Core::FADUtils::HigherOrderFadValue<scalar_type_rot_1st>::apply(3 + Surface::n_dof_,
            3 + i_surface,
            Core::FADUtils::cast_to_double(
                this->face_element_->get_face_element_data().element_position_(i_surface)));

  // Initialize local matrices.
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> local_g(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Mortar::n_dof_, n_dof_rot_, double> local_GB(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Mortar::n_dof_, Surface::n_dof_, double> local_GS(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<n_dof_rot_, Mortar::n_dof_, double> local_FB(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Surface::n_dof_, Mortar::n_dof_, double> local_FS(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> local_kappa(Core::LinAlg::Initialization::zero);

  // Get the type of surface triad construction.
  const auto surface_triad_type =
      this->params()->beam_to_solid_surface_meshtying_params()->get_surface_triad_construction();

  // Evaluate the mortar terms for this pair.
  {
    // Initialize variables.
    Core::LinAlg::Matrix<3, 1, double> dr_beam_ref;
    Core::LinAlg::Matrix<4, 1, double> quaternion_beam_double;
    Core::LinAlg::Matrix<3, 1, double> psi_beam_double;
    Core::LinAlg::Matrix<3, 1, scalar_type_rot_1st> psi_beam;
    Core::LinAlg::Matrix<3, 1, scalar_type_rot_1st> psi_surface;
    Core::LinAlg::Matrix<3, 1, scalar_type_rot_1st> psi_rel;
    Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_beam;
    Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_beam_inv;
    Core::LinAlg::Matrix<4, 1, double> quaternion_beam_ref;
    Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_surface;
    Core::LinAlg::Matrix<4, 1, scalar_type_rot_1st> quaternion_rel;
    Core::LinAlg::Matrix<3, 3, double> T_beam;
    Core::LinAlg::Matrix<3, 3, double> T_surface;
    Core::LinAlg::Matrix<3, 3, double> T_surface_inv;
    Core::LinAlg::Matrix<3, 3, double> T_rel;

    Core::LinAlg::Matrix<Mortar::n_nodes_, 1, double> lambda_shape_functions;
    Core::LinAlg::Matrix<3, Mortar::n_dof_, double> lambda_shape_functions_full(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::SerialDenseVector L_i(3);
    Core::LinAlg::Matrix<3, n_dof_rot_, double> L_full(Core::LinAlg::Initialization::zero);
    std::vector<Core::LinAlg::Matrix<3, 3, double>> I_beam_tilde;
    Core::LinAlg::Matrix<3, n_dof_rot_, double> I_beam_tilde_full;
    Core::LinAlg::Matrix<3, n_dof_rot_, double> T_beam_times_I_beam_tilde_full;
    Core::LinAlg::Matrix<3, Mortar::n_dof_, double> T_rel_tr_times_lambda_shape;
    Core::LinAlg::Matrix<3, Mortar::n_dof_, double> T_surface_mtr_times_T_rel_tr_times_lambda_shape;
    Core::LinAlg::Matrix<n_dof_rot_, Mortar::n_dof_, double> d_fb_d_lambda_gp;
    Core::LinAlg::Matrix<Surface::n_dof_, Mortar::n_dof_, double> d_fs_d_lambda_gp;
    Core::LinAlg::Matrix<Mortar::n_dof_, 1, scalar_type_rot_1st> g_gp;
    Core::LinAlg::Matrix<3, Surface::n_dof_, double> d_psi_surface_d_q_surface;
    Core::LinAlg::Matrix<Mortar::n_dof_, 3, double> d_g_d_psi_beam;
    Core::LinAlg::Matrix<Mortar::n_dof_, n_dof_rot_, double> d_g_d_psi_beam_times_T_beam_I;
    Core::LinAlg::Matrix<Mortar::n_dof_, Surface::n_dof_, double> d_g_d_q_surface;

    // Initialize scalar variables.
    double segment_jacobian = 0.0;
    double beam_segmentation_factor = 0.0;

    // Calculate the meshtying forces.
    // Loop over segments.
    for (unsigned int i_segment = 0; i_segment < this->line_to_3D_segments_.size(); i_segment++)
    {
      // Factor to account for a segment length not from -1 to 1.
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
              3 + Surface::n_dof_, i_dim, psi_beam_double(i_dim));
        Core::LargeRotations::angletoquaternion(psi_beam, quaternion_beam);
        quaternion_beam_inv = Core::LargeRotations::inversequaternion(quaternion_beam);

        // Get the surface rotation vector.
        ref_triad_interpolation_scheme.get_interpolated_quaternion_at_xi(
            quaternion_beam_ref, projected_gauss_point.get_eta());
        get_surface_rotation_vector(projected_gauss_point.get_xi(),
            this->face_element_->get_face_reference_element_data(), q_surface, quaternion_beam_ref,
            surface_triad_type, psi_surface);
        Core::LargeRotations::angletoquaternion(psi_surface, quaternion_surface);

        // Calculate the relative rotation vector.
        Core::LargeRotations::quaternionproduct(
            quaternion_beam_inv, quaternion_surface, quaternion_rel);
        Core::LargeRotations::quaterniontoangle(quaternion_rel, psi_rel);

        // Calculate the transformation matrices.
        T_rel = Core::LargeRotations::tmatrix(Core::FADUtils::cast_to_double(psi_rel));
        T_beam = Core::LargeRotations::tmatrix(Core::FADUtils::cast_to_double(psi_beam));
        T_surface = Core::LargeRotations::tmatrix(Core::FADUtils::cast_to_double(psi_surface));
        T_surface_inv = T_surface;
        Core::LinAlg::inverse(T_surface_inv);

        // Evaluate mortar shape functions.
        GeometryPair::EvaluateShapeFunction<Mortar>::evaluate(
            lambda_shape_functions, projected_gauss_point.get_eta());
        for (unsigned int i_node = 0; i_node < Mortar::n_nodes_; i_node++)
          for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
            lambda_shape_functions_full(i_dim, 3 * i_node + i_dim) = lambda_shape_functions(i_node);

        // Get the shape functions for the interpolation of the beam rotations. This is currently
        // only implemented for 2nd order Lagrange interpolation (Beam3rHerm2Line3).
        const unsigned int n_nodes_rot = 3;
        Core::FE::shape_function_1d(
            L_i, projected_gauss_point.get_eta(), Core::FE::CellType::line3);
        for (unsigned int i_node = 0; i_node < n_nodes_rot; i_node++)
          for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
            L_full(i_dim, 3 * i_node + i_dim) = L_i(i_node);

        triad_interpolation_scheme.get_nodal_generalized_rotation_interpolation_matrices_at_xi(
            I_beam_tilde, projected_gauss_point.get_eta());
        for (unsigned int i_node = 0; i_node < n_nodes_rot; i_node++)
          for (unsigned int i_dim_0 = 0; i_dim_0 < 3; i_dim_0++)
            for (unsigned int i_dim_1 = 0; i_dim_1 < 3; i_dim_1++)
              I_beam_tilde_full(i_dim_0, i_node * 3 + i_dim_1) =
                  I_beam_tilde[i_node](i_dim_0, i_dim_1);

        // Solid angle derived w.r.t. the surface DOFs.
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          for (unsigned int i_surface = 0; i_surface < Surface::n_dof_; i_surface++)
            d_psi_surface_d_q_surface(i_dim, i_surface) = psi_surface(i_dim).dx(3 + i_surface);

        // Calculate the force terms derived w.r.t. the Lagrange multipliers.
        T_rel_tr_times_lambda_shape.multiply_tn(T_rel, lambda_shape_functions_full);
        d_fb_d_lambda_gp.multiply_tn(L_full, T_rel_tr_times_lambda_shape);
        d_fb_d_lambda_gp.scale(-1.0 * projected_gauss_point.get_gauss_weight() * segment_jacobian);

        T_surface_mtr_times_T_rel_tr_times_lambda_shape.multiply_tn(
            T_surface_inv, T_rel_tr_times_lambda_shape);
        d_fs_d_lambda_gp.multiply_tn(
            d_psi_surface_d_q_surface, T_surface_mtr_times_T_rel_tr_times_lambda_shape);
        d_fs_d_lambda_gp.scale(projected_gauss_point.get_gauss_weight() * segment_jacobian);

        // Constraint vector.
        g_gp.put_scalar(0.0);
        for (unsigned int i_row = 0; i_row < Mortar::n_dof_; i_row++)
          for (unsigned int i_col = 0; i_col < 3; i_col++)
            g_gp(i_row) += lambda_shape_functions_full(i_col, i_row) * psi_rel(i_col);
        g_gp.scale(projected_gauss_point.get_gauss_weight() * segment_jacobian);

        // Derivatives of constraint vector.
        T_beam_times_I_beam_tilde_full.multiply(T_beam, I_beam_tilde_full);

        for (unsigned int i_lambda = 0; i_lambda < Mortar::n_dof_; i_lambda++)
          for (unsigned int i_psi = 0; i_psi < 3; i_psi++)
            d_g_d_psi_beam(i_lambda, i_psi) = g_gp(i_lambda).dx(i_psi);
        d_g_d_psi_beam_times_T_beam_I.multiply(d_g_d_psi_beam, T_beam_times_I_beam_tilde_full);

        for (unsigned int i_lambda = 0; i_lambda < Mortar::n_dof_; i_lambda++)
          for (unsigned int i_surface = 0; i_surface < Surface::n_dof_; i_surface++)
            d_g_d_q_surface(i_lambda, i_surface) = g_gp(i_lambda).dx(3 + i_surface);

        // Add to output matrices and vector.
        local_g += Core::FADUtils::cast_to_double(g_gp);
        local_GB += d_g_d_psi_beam_times_T_beam_I;
        local_GS += d_g_d_q_surface;
        local_FB += d_fb_d_lambda_gp;
        local_FS += d_fs_d_lambda_gp;

        // Calculate the scaling entries.
        for (unsigned int i_mortar_node = 0; i_mortar_node < Mortar::n_nodes_; i_mortar_node++)
          for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
            local_kappa(i_mortar_node * 3 + i_dim) += lambda_shape_functions(i_mortar_node) *
                                                      projected_gauss_point.get_gauss_weight() *
                                                      segment_jacobian;
      }
    }
  }

  // Get the rotational GIDs of the surface and beam.
  std::vector<int> gid_surface;
  Core::LinAlg::Matrix<n_dof_rot_, 1, int> gid_rot;
  get_pair_rotational_gid(discret, gid_surface, gid_rot);

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
  for (unsigned int i_dof_lambda = 0; i_dof_lambda < Mortar::n_dof_; i_dof_lambda++)
  {
    for (unsigned int i_dof_rot = 0; i_dof_rot < n_dof_rot_; i_dof_rot++)
    {
      global_constraint_lin_beam.fe_assemble(
          local_GB(i_dof_lambda, i_dof_rot), lambda_gid_rot[i_dof_lambda], gid_rot(i_dof_rot));
      global_force_beam_lin_lambda.fe_assemble(
          local_FB(i_dof_rot, i_dof_lambda), gid_rot(i_dof_rot), lambda_gid_rot[i_dof_lambda]);
    }
    for (unsigned int i_dof_surface = 0; i_dof_surface < Surface::n_dof_; i_dof_surface++)
    {
      global_constraint_lin_solid.fe_assemble(local_GS(i_dof_lambda, i_dof_surface),
          lambda_gid_rot[i_dof_lambda], gid_surface[i_dof_surface]);
      global_force_solid_lin_lambda.fe_assemble(local_FS(i_dof_surface, i_dof_lambda),
          gid_surface[i_dof_surface], lambda_gid_rot[i_dof_lambda]);
    }
  }
}

/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface, typename Mortar>
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairMortarRotationFAD<ScalarType, Beam, Surface,
    Mortar>::get_pair_rotational_gid(const Core::FE::Discretization& discret,
    std::vector<int>& gid_surface, Core::LinAlg::Matrix<n_dof_rot_, 1, int>& gid_rot) const
{
  // Get the GIDs of the surface and beam.
  const auto gid_rot_vector = Utils::get_element_rot_gid_indices(discret, this->element1());
  for (unsigned int i = 0; i < n_dof_rot_; i++) gid_rot(i) = gid_rot_vector[i];
  std::vector<int> lmowner, lmstride;
  this->face_element_->get_element()->location_vector(discret, gid_surface, lmowner, lmstride);
}


/**
 *
 */
template <typename ScalarType, typename Beam, typename Surface, typename Mortar>
std::shared_ptr<BeamInteraction::BeamContactPair>
beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar_rotation(
    const bool rotational_coupling)
{
  using namespace BeamInteraction;
  using namespace GeometryPair;

  if (!rotational_coupling)
    return std::make_shared<
        BeamToSolidSurfaceMeshtyingPairMortarFAD<ScalarType, Beam, Surface, Mortar>>();
  else
    return std::make_shared<
        BeamToSolidSurfaceMeshtyingPairMortarRotationFAD<ScalarType, Beam, Surface, Mortar>>();
}

/**
 *
 */
template <typename Mortar>
std::shared_ptr<BeamInteraction::BeamContactPair>
beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar(
    const Core::FE::CellType surface_shape, const bool rotational_coupling)
{
  using namespace BeamInteraction;
  using namespace GeometryPair;

  switch (surface_shape)
  {
    case Core::FE::CellType::tri3:
      return beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar_rotation<
          line_to_surface_patch_scalar_type, t_hermite, t_tri3, Mortar>(rotational_coupling);
    case Core::FE::CellType::tri6:
      return beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar_rotation<
          line_to_surface_patch_scalar_type, t_hermite, t_tri6, Mortar>(rotational_coupling);
    case Core::FE::CellType::quad4:
      return beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar_rotation<
          line_to_surface_patch_scalar_type, t_hermite, t_quad4, Mortar>(rotational_coupling);
    case Core::FE::CellType::quad8:
      return beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar_rotation<
          line_to_surface_patch_scalar_type, t_hermite, t_quad8, Mortar>(rotational_coupling);
    case Core::FE::CellType::quad9:
      return beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar_rotation<
          line_to_surface_patch_scalar_type, t_hermite, t_quad9, Mortar>(rotational_coupling);
    case Core::FE::CellType::nurbs9:
      return beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar_rotation<
          line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_nurbs9>, t_hermite, t_nurbs9,
          Mortar>(rotational_coupling);
    default:
      FOUR_C_THROW("Wrong element type for surface element.");
      return nullptr;
  }
}

/**
 *
 */
template <typename Mortar>
std::shared_ptr<BeamInteraction::BeamContactPair>
beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar_x_volume(
    const Core::FE::CellType surface_shape, const bool rotational_coupling)
{
  using namespace BeamInteraction;
  using namespace GeometryPair;

  switch (surface_shape)
  {
    case Core::FE::CellType::quad4:
      return beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar_rotation<
          line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex8>, t_hermite, t_quad4,
          Mortar>(rotational_coupling);
    case Core::FE::CellType::quad8:
      return beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar_rotation<
          line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex20>, t_hermite, t_quad8,
          Mortar>(rotational_coupling);
    case Core::FE::CellType::quad9:
      return beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar_rotation<
          line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex27>, t_hermite, t_quad9,
          Mortar>(rotational_coupling);
    default:
      FOUR_C_THROW("Wrong element type for surface element.");
      return nullptr;
  }
}

/**
 *
 */
std::shared_ptr<BeamInteraction::BeamContactPair>
BeamInteraction::beam_to_solid_surface_meshtying_pair_mortar_fad_factory(
    const Core::FE::CellType surface_shape,
    const Inpar::BeamToSolid::BeamToSolidMortarShapefunctions mortar_shapefunction,
    const bool rotational_coupling,
    const Inpar::GeometryPair::SurfaceNormals surface_normal_strategy)
{
  using namespace GeometryPair;

  if (surface_normal_strategy == Inpar::GeometryPair::SurfaceNormals::standard)
  {
    switch (mortar_shapefunction)
    {
      case Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::line2:
      {
        return beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar<t_line2>(
            surface_shape, rotational_coupling);
      }
      case Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::line3:
      {
        return beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar<t_line3>(
            surface_shape, rotational_coupling);
      }
      case Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::line4:
      {
        return beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar<t_line4>(
            surface_shape, rotational_coupling);
      }
      default:
        FOUR_C_THROW("Wrong mortar shape function.");
    }
  }
  else if (surface_normal_strategy == Inpar::GeometryPair::SurfaceNormals::extended_volume)
  {
    switch (mortar_shapefunction)
    {
      case Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::line2:
        return beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar_x_volume<t_line2>(
            surface_shape, rotational_coupling);
      case Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::line3:
        return beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar_x_volume<t_line3>(
            surface_shape, rotational_coupling);
      case Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::line4:
        return beam_to_solid_surface_meshtying_pair_mortar_fad_factory_mortar_x_volume<t_line4>(
            surface_shape, rotational_coupling);
      default:
        FOUR_C_THROW("Wrong mortar shape function.");
    }
  }
  else
    FOUR_C_THROW("Surface normal strategy not recognized.");

  return nullptr;
}

FOUR_C_NAMESPACE_CLOSE
