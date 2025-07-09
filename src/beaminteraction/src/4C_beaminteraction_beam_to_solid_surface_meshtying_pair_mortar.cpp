// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_surface_meshtying_pair_mortar.hpp"

#include "4C_beaminteraction_beam_to_solid_surface_meshtying_params.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_element_faces.hpp"
#include "4C_geometry_pair_line_to_surface.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename Beam, typename Surface, typename Mortar>
BeamInteraction::BeamToSolidSurfaceMeshtyingPairMortar<Beam, Surface,
    Mortar>::BeamToSolidSurfaceMeshtyingPairMortar()
    : base_class()
{
  // Empty constructor.
}

/**
 *
 */
template <typename Beam, typename Surface, typename Mortar>
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairMortar<Beam, Surface,
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

  // Initialize variables for local mortar matrices.
  Core::LinAlg::Matrix<Mortar::n_dof_, Beam::n_dof_, double> local_D(
      Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Mortar::n_dof_, Surface::n_dof_, double> local_M(
      Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> local_kappa(
      Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> local_constraint(
      Core::LinAlg::Initialization::uninitialized);

  // Evaluate the local mortar contributions.
  evaluate_dm(local_D, local_M, local_kappa, local_constraint);

  // Assemble into global matrices.
  assemble_local_mortar_contributions<Beam, Surface, Mortar>(this, discret, mortar_manager,
      global_constraint_lin_beam, global_constraint_lin_solid, global_force_beam_lin_lambda,
      global_force_solid_lin_lambda, global_constraint, global_kappa, global_lambda_active, local_D,
      local_M, local_kappa, local_constraint);
}

/**
 *
 */
template <typename Beam, typename Surface, typename Mortar>
void BeamInteraction::BeamToSolidSurfaceMeshtyingPairMortar<Beam, Surface, Mortar>::evaluate_dm(
    Core::LinAlg::Matrix<Mortar::n_dof_, Beam::n_dof_, double>& local_D,
    Core::LinAlg::Matrix<Mortar::n_dof_, Surface::n_dof_, double>& local_M,
    Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_kappa,
    Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_constraint) const
{
  // Initialize the local mortar matrices.
  local_D.put_scalar(0.0);
  local_M.put_scalar(0.0);
  local_kappa.put_scalar(0.0);
  local_constraint.put_scalar(0.0);

  // Initialize variables for shape function values.
  Core::LinAlg::Matrix<1, Mortar::n_nodes_ * Mortar::n_val_, double> N_mortar(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, Beam::n_nodes_ * Beam::n_val_, double> N_beam(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, Surface::n_nodes_ * Surface::n_val_, double> N_surface(
      Core::LinAlg::Initialization::zero);

  // Initialize variable for beam position derivative.
  Core::LinAlg::Matrix<3, 1, double> dr_beam_ref(Core::LinAlg::Initialization::zero);

  // Initialize scalar variables.Clear
  double segment_jacobian = 0.0;
  double beam_segmentation_factor = 0.0;

  // Calculate the mortar matrices.
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

      // Get the shape function matrices.
      N_mortar.clear();
      N_beam.clear();
      N_surface.clear();
      GeometryPair::EvaluateShapeFunction<Mortar>::evaluate(
          N_mortar, projected_gauss_point.get_eta());
      GeometryPair::EvaluateShapeFunction<Beam>::evaluate(
          N_beam, projected_gauss_point.get_eta(), this->ele1pos_.shape_function_data_);
      GeometryPair::EvaluateShapeFunction<Surface>::evaluate(N_surface,
          projected_gauss_point.get_xi(),
          this->face_element_->get_face_element_data().shape_function_data_);

      // Fill in the local templated mortar matrix D.
      for (unsigned int i_mortar_node = 0; i_mortar_node < Mortar::n_nodes_; i_mortar_node++)
        for (unsigned int i_mortar_val = 0; i_mortar_val < Mortar::n_val_; i_mortar_val++)
          for (unsigned int i_beam_node = 0; i_beam_node < Beam::n_nodes_; i_beam_node++)
            for (unsigned int i_beam_val = 0; i_beam_val < Beam::n_val_; i_beam_val++)
              for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
                local_D(i_mortar_node * Mortar::n_val_ * 3 + i_mortar_val * 3 + i_dim,
                    i_beam_node * Beam::n_val_ * 3 + i_beam_val * 3 + i_dim) +=
                    N_mortar(i_mortar_node * Mortar::n_val_ + i_mortar_val) *
                    N_beam(i_beam_node * Beam::n_val_ + i_beam_val) *
                    projected_gauss_point.get_gauss_weight() * segment_jacobian;

      // Fill in the local templated mortar matrix M.
      for (unsigned int i_mortar_node = 0; i_mortar_node < Mortar::n_nodes_; i_mortar_node++)
        for (unsigned int i_mortar_val = 0; i_mortar_val < Mortar::n_val_; i_mortar_val++)
          for (unsigned int i_surface_node = 0; i_surface_node < Surface::n_nodes_;
              i_surface_node++)
            for (unsigned int i_surface_val = 0; i_surface_val < Surface::n_val_; i_surface_val++)
              for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
                local_M(i_mortar_node * Mortar::n_val_ * 3 + i_mortar_val * 3 + i_dim,
                    i_surface_node * Surface::n_val_ * 3 + i_surface_val * 3 + i_dim) +=
                    N_mortar(i_mortar_node * Mortar::n_val_ + i_mortar_val) *
                    N_surface(i_surface_node * Surface::n_val_ + i_surface_val) *
                    projected_gauss_point.get_gauss_weight() * segment_jacobian;

      // Fill in the local templated mortar scaling vector kappa.
      for (unsigned int i_mortar_node = 0; i_mortar_node < Mortar::n_nodes_; i_mortar_node++)
        for (unsigned int i_mortar_val = 0; i_mortar_val < Mortar::n_val_; i_mortar_val++)
          for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
            local_kappa(i_mortar_node * Mortar::n_val_ * 3 + i_mortar_val * 3 + i_dim) +=
                N_mortar(i_mortar_node * Mortar::n_val_ + i_mortar_val) *
                projected_gauss_point.get_gauss_weight() * segment_jacobian;
    }
  }

  // Add the local constraint contributions. For this we multiply the local mortar matrices with the
  // positions / displacements to get the actual constraint terms for this pair.
  Core::LinAlg::Matrix<Beam::n_dof_, 1, double> beam_coupling_dof(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Surface::n_dof_, 1, double> surface_coupling_dof(
      Core::LinAlg::Initialization::zero);
  switch (this->params()->beam_to_solid_surface_meshtying_params()->get_coupling_type())
  {
    case Inpar::BeamToSolid::BeamToSolidSurfaceCoupling::reference_configuration_forced_to_zero:
    {
      beam_coupling_dof = Core::FADUtils::cast_to_double(this->ele1pos_.element_position_);
      surface_coupling_dof = Core::FADUtils::cast_to_double(
          this->face_element_->get_face_element_data().element_position_);
      break;
    }
    case Inpar::BeamToSolid::BeamToSolidSurfaceCoupling::displacement:
    {
      beam_coupling_dof = Core::FADUtils::cast_to_double(this->ele1pos_.element_position_);
      beam_coupling_dof -= this->ele1posref_.element_position_;
      surface_coupling_dof = Core::FADUtils::cast_to_double(
          this->face_element_->get_face_element_data().element_position_);
      surface_coupling_dof -=
          this->face_element_->get_face_reference_element_data().element_position_;
      break;
    }
    default:
      FOUR_C_THROW("Wrong coupling type.");
  }
  for (unsigned int i_lambda = 0; i_lambda < Mortar::n_dof_; i_lambda++)
  {
    for (unsigned int i_beam = 0; i_beam < Beam::n_dof_; i_beam++)
      local_constraint(i_lambda) += local_D(i_lambda, i_beam) * beam_coupling_dof(i_beam);
    for (unsigned int i_surface = 0; i_surface < Surface::n_dof_; i_surface++)
      local_constraint(i_lambda) -= local_M(i_lambda, i_surface) * surface_coupling_dof(i_surface);
  }
}


/**
 *
 */
std::shared_ptr<BeamInteraction::BeamContactPair>
BeamInteraction::beam_to_solid_surface_meshtying_pair_mortar_factory(
    const Core::FE::CellType surface_shape,
    const Inpar::BeamToSolid::BeamToSolidMortarShapefunctions mortar_shapefunction)
{
  using namespace GeometryPair;

  switch (mortar_shapefunction)
  {
    case Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::line2:
    {
      switch (surface_shape)
      {
        case Core::FE::CellType::tri3:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_tri3, t_line2>>();
        case Core::FE::CellType::tri6:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_tri6, t_line2>>();
        case Core::FE::CellType::quad4:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_quad4, t_line2>>();
        case Core::FE::CellType::quad8:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_quad8, t_line2>>();
        case Core::FE::CellType::quad9:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_quad9, t_line2>>();
        case Core::FE::CellType::nurbs9:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_nurbs9, t_line2>>();
        default:
          FOUR_C_THROW("Wrong element type for surface element.");
      }
      break;
    }
    case Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::line3:
    {
      switch (surface_shape)
      {
        case Core::FE::CellType::tri3:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_tri3, t_line3>>();
        case Core::FE::CellType::tri6:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_tri6, t_line3>>();
        case Core::FE::CellType::quad4:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_quad4, t_line3>>();
        case Core::FE::CellType::quad8:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_quad8, t_line3>>();
        case Core::FE::CellType::quad9:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_quad9, t_line3>>();
        case Core::FE::CellType::nurbs9:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_nurbs9, t_line3>>();
        default:
          FOUR_C_THROW("Wrong element type for surface element.");
      }
      break;
    }
    case Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::line4:
    {
      switch (surface_shape)
      {
        case Core::FE::CellType::tri3:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_tri3, t_line4>>();
        case Core::FE::CellType::tri6:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_tri6, t_line4>>();
        case Core::FE::CellType::quad4:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_quad4, t_line4>>();
        case Core::FE::CellType::quad8:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_quad8, t_line4>>();
        case Core::FE::CellType::quad9:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_quad9, t_line4>>();
        case Core::FE::CellType::nurbs9:
          return std::make_shared<
              BeamToSolidSurfaceMeshtyingPairMortar<t_hermite, t_nurbs9, t_line4>>();
        default:
          FOUR_C_THROW("Wrong element type for surface element.");
      }
      break;
    }
    default:
      FOUR_C_THROW("Wrong mortar shape function.");
  }

  return nullptr;
}

FOUR_C_NAMESPACE_CLOSE
