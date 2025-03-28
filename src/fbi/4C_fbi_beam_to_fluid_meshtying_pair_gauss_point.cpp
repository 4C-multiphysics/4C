// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fbi_beam_to_fluid_meshtying_pair_gauss_point.hpp"

#include "4C_beaminteraction_contact_params.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_pair_base.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_params.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_to_volume.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
template <typename Beam, typename Fluid>
BeamInteraction::BeamToFluidMeshtyingPairGaussPoint<Beam,
    Fluid>::BeamToFluidMeshtyingPairGaussPoint()
    : BeamToFluidMeshtyingPairBase<Beam, Fluid>()
{
  // Empty constructor.
}

/*------------------------------------------------------------------------------------------------*/

template <typename Beam, typename Fluid>
bool BeamInteraction::BeamToFluidMeshtyingPairGaussPoint<Beam, Fluid>::evaluate(
    Core::LinAlg::SerialDenseVector* forcevec1, Core::LinAlg::SerialDenseVector* forcevec2,
    Core::LinAlg::SerialDenseMatrix* stiffmat11, Core::LinAlg::SerialDenseMatrix* stiffmat12,
    Core::LinAlg::SerialDenseMatrix* stiffmat21, Core::LinAlg::SerialDenseMatrix* stiffmat22)
{
  // Call Evaluate on the geometry Pair
  if (!this->meshtying_is_evaluated_)
  {
    this->cast_geometry_pair()->evaluate(
        this->ele1poscur_, this->ele2poscur_, this->line_to_3D_segments_);
    this->meshtying_is_evaluated_ = true;
  }

  // If there are no intersection segments, return no contact status.
  if (this->line_to_3D_segments_.size() == 0) return false;

  // Initialize variables for position and force vectors.
  Core::LinAlg::Matrix<3, 1, double> dr_beam_ref;
  Core::LinAlg::Matrix<3, 1, scalar_type> r_beam;
  Core::LinAlg::Matrix<3, 1, scalar_type> r_fluid;
  Core::LinAlg::Matrix<3, 1, scalar_type> v_beam;
  Core::LinAlg::Matrix<3, 1, scalar_type> v_fluid;
  Core::LinAlg::Matrix<3, 1, scalar_type> force;
  Core::LinAlg::Matrix<3, 1, scalar_type> force2;
  Core::LinAlg::Matrix<Beam::n_dof_, 1, scalar_type> force_element_1(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Fluid::n_dof_, 1, scalar_type> force_element_2(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Fluid::n_dof_, 1, scalar_type> force_element_f(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, Beam::n_nodes_ * Beam::n_val_, double> N_beam(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, Fluid::n_nodes_ * Fluid::n_val_, double> N_fluid(
      Core::LinAlg::Initialization::zero);

  // Resize and initialize the return variables.
  if (forcevec1 != nullptr) forcevec1->size(Beam::n_dof_);
  if (forcevec2 != nullptr) forcevec2->size(Fluid::n_dof_);
  if (stiffmat11 != nullptr) stiffmat11->shape(Beam::n_dof_, Beam::n_dof_);
  if (stiffmat12 != nullptr) stiffmat12->shape(Beam::n_dof_, Fluid::n_dof_);
  if (stiffmat21 != nullptr) stiffmat21->shape(Fluid::n_dof_, Beam::n_dof_);
  if (stiffmat22 != nullptr) stiffmat22->shape(Fluid::n_dof_, Fluid::n_dof_);

  // Initialize scalar variables.
  double segment_jacobian, beam_segmentation_factor;

  // Calculate the meshtying forces.
  // Loop over segments.
  for (unsigned int i_segment = 0; i_segment < this->line_to_3D_segments_.size(); i_segment++)
  {
    // Factor to account for the integration segment length.
    beam_segmentation_factor = 0.5 * this->line_to_3D_segments_[i_segment].get_segment_length();

    // Gauss point loop.
    for (unsigned int i_gp = 0;
        i_gp < this->line_to_3D_segments_[i_segment].get_projection_points().size(); i_gp++)
    {
      // Get the current Gauss point.
      const GEOMETRYPAIR::ProjectionPoint1DTo3D<double>& projected_gauss_point =
          this->line_to_3D_segments_[i_segment].get_projection_points()[i_gp];

      // Get the jacobian in the reference configuration.
      GEOMETRYPAIR::evaluate_position_derivative1<Beam>(
          projected_gauss_point.get_eta(), this->ele1posref_, dr_beam_ref);

      // Jacobian including the segment length.
      segment_jacobian = dr_beam_ref.norm2() * beam_segmentation_factor;

      // Get the current positions on beam and fluid.
      GEOMETRYPAIR::evaluate_position<Beam>(
          projected_gauss_point.get_eta(), this->ele1pos_, r_beam);
      GEOMETRYPAIR::evaluate_position<Fluid>(
          projected_gauss_point.get_xi(), this->ele2pos_, r_fluid);

      N_beam.clear();
      N_fluid.clear();

      // Evaluate the chapefunctions at the current gauss point
      GEOMETRYPAIR::EvaluateShapeFunction<Beam>::evaluate(
          N_beam, projected_gauss_point.get_eta(), this->ele1pos_.shape_function_data_);
      GEOMETRYPAIR::EvaluateShapeFunction<Fluid>::evaluate(N_fluid, projected_gauss_point.get_xi());

      // assemble fluid mass matrix
      if (stiffmat22 != nullptr)
      {
        for (unsigned int i_fluid_node1 = 0; i_fluid_node1 < Fluid::n_nodes_; i_fluid_node1++)
          for (unsigned int i_fluid_val1 = 0; i_fluid_val1 < Fluid::n_val_; i_fluid_val1++)
            for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
              for (unsigned int i_fluid_node2 = 0; i_fluid_node2 < Fluid::n_nodes_; i_fluid_node2++)
                for (unsigned int i_fluid_val2 = 0; i_fluid_val2 < Fluid::n_val_; i_fluid_val2++)
                  (*stiffmat22)(i_fluid_node1* Fluid::n_val_ * 3 + i_fluid_val1 * 3 + i_dir,
                      i_fluid_node2 * Fluid::n_val_ * 3 + i_fluid_val2 * 3 + i_dir) +=
                      N_fluid(i_fluid_node1) * N_fluid(i_fluid_node2) *
                      projected_gauss_point.get_gauss_weight() * segment_jacobian;
      }

      // assemble beam mass matrix
      if (stiffmat11 != nullptr)
      {
        for (unsigned int i_beam_node1 = 0; i_beam_node1 < Beam::n_nodes_; i_beam_node1++)
          for (unsigned int i_beam_val1 = 0; i_beam_val1 < Beam::n_val_; i_beam_val1++)
            for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
              for (unsigned int i_beam_node2 = 0; i_beam_node2 < Beam::n_nodes_; i_beam_node2++)
                for (unsigned int i_beam_val2 = 0; i_beam_val2 < Beam::n_val_; i_beam_val2++)
                  (*stiffmat11)(i_beam_node1* Beam::n_val_ * 3 + 3 * i_beam_val1 + i_dir,
                      i_beam_node2 * Beam::n_val_ * 3 + 3 * i_beam_val2 + i_dir) +=
                      N_beam(i_beam_node1 * Beam::n_val_ + i_beam_val1) *
                      N_beam(i_beam_node2 * Beam::n_val_ + i_beam_val2) *
                      projected_gauss_point.get_gauss_weight() * segment_jacobian;
      }

      // assemble fluid beam coupling matrix
      if (stiffmat21 != nullptr)
      {
        for (unsigned int i_fluid_node1 = 0; i_fluid_node1 < Fluid::n_nodes_; i_fluid_node1++)
          for (unsigned int i_fluid_val1 = 0; i_fluid_val1 < Fluid::n_val_; i_fluid_val1++)
            for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
              for (unsigned int i_beam_node2 = 0; i_beam_node2 < Beam::n_nodes_; i_beam_node2++)
                for (unsigned int i_beam_val2 = 0; i_beam_val2 < Beam::n_val_; i_beam_val2++)
                  (*stiffmat21)(i_fluid_node1* Fluid::n_val_ * 3 + i_fluid_val1 * 3 + i_dir,
                      i_beam_node2 * Beam::n_val_ * 3 + 3 * i_beam_val2 + i_dir) +=
                      N_fluid(i_fluid_node1) * N_beam(i_beam_node2 * Beam::n_val_ + i_beam_val2) *
                      projected_gauss_point.get_gauss_weight() * segment_jacobian;
      }

      // assemble fluid beam coupling matrix
      if (stiffmat12 != nullptr)
      {
        for (unsigned int i_beam_node1 = 0; i_beam_node1 < Beam::n_nodes_; i_beam_node1++)
          for (unsigned int i_beam_val1 = 0; i_beam_val1 < Beam::n_val_; i_beam_val1++)
            for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
              for (unsigned int i_fluid_node2 = 0; i_fluid_node2 < Fluid::n_nodes_; i_fluid_node2++)
                for (unsigned int i_fluid_val2 = 0; i_fluid_val2 < Fluid::n_val_; i_fluid_val2++)
                  (*stiffmat12)(i_beam_node1* Beam::n_val_ * 3 + 3 * i_beam_val1 + i_dir,
                      i_fluid_node2 * Fluid::n_val_ * 3 + 3 * i_fluid_val2 + i_dir) +=
                      N_fluid(i_fluid_node2) * N_beam(i_beam_node1 * Beam::n_val_ + i_beam_val1) *
                      projected_gauss_point.get_gauss_weight() * segment_jacobian;
      }
    }
  }

  // assemble structure force vector
  {
    if (forcevec1 != nullptr)
    {
      for (unsigned int i_dof1 = 0; i_dof1 < Beam::n_dof_; i_dof1++)
      {
        for (unsigned int i_dof2 = 0; i_dof2 < Beam::n_dof_; i_dof2++)
          (*forcevec1)(i_dof1) +=
              (*stiffmat11)(i_dof1, i_dof2) *
              Core::FADUtils::cast_to_double(this->ele1vel_.element_position_(i_dof2));
        for (unsigned int i_dof2 = 0; i_dof2 < Fluid::n_dof_; i_dof2++)
          (*forcevec1)(i_dof1) -=
              (*stiffmat12)(i_dof1, i_dof2) *
              Core::FADUtils::cast_to_double(this->ele2vel_.element_position_(i_dof2));
      }
    }

    // assemble fluid force vector
    if (forcevec2 != nullptr)
    {
      if (!std::dynamic_pointer_cast<FBI::BeamToFluidMeshtyingParams>(this->params())
              ->get_weak_dirichlet_flag())

      {
        for (unsigned int i_dof1 = 0; i_dof1 < Fluid::n_dof_; i_dof1++)
        {
          for (unsigned int i_dof2 = 0; i_dof2 < Fluid::n_dof_; i_dof2++)
            (*forcevec2)(i_dof1) +=
                (*stiffmat22)(i_dof1, i_dof2) *
                Core::FADUtils::cast_to_double(this->ele2vel_.element_position_(i_dof2));
          for (unsigned int i_dof2 = 0; i_dof2 < Beam::n_dof_; i_dof2++)
            (*forcevec2)(i_dof1) -=
                (*stiffmat21)(i_dof1, i_dof2) *
                Core::FADUtils::cast_to_double(this->ele1vel_.element_position_(i_dof2));
        }
      }
      else

        for (unsigned int i_dof1 = 0; i_dof1 < Fluid::n_dof_; i_dof1++)
        {
          for (unsigned int i_dof2 = 0; i_dof2 < Beam::n_dof_; i_dof2++)
            (*forcevec2)(i_dof1) -=
                (*stiffmat21)(i_dof1, i_dof2) *
                Core::FADUtils::cast_to_double(this->ele1vel_.element_position_(i_dof2));
        }
    }
  }

  // Return true as there are meshtying contributions.
  return true;
}

/**
 * Explicit template initialization of template class.
 */
// Hermite beam element, hex8 solid element.
template class BeamInteraction::BeamToFluidMeshtyingPairGaussPoint<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_hex8>;
// Hermite beam element, hex20 solid element.
template class BeamInteraction::BeamToFluidMeshtyingPairGaussPoint<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_hex20>;
// Hermite beam element, hex27 solid element.
template class BeamInteraction::BeamToFluidMeshtyingPairGaussPoint<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_hex27>;
// Hermite beam element, tet4 solid element.
template class BeamInteraction::BeamToFluidMeshtyingPairGaussPoint<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_tet4>;
// Hermite beam element, tet10 solid element.
template class BeamInteraction::BeamToFluidMeshtyingPairGaussPoint<GEOMETRYPAIR::t_hermite,
    GEOMETRYPAIR::t_tet10>;

FOUR_C_NAMESPACE_CLOSE
