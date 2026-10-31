// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_contact_beam_to_solid_volume_meshtying_pair_2d-3d_full.hpp"

#include "4C_beam3_reissner.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_contact_beam_to_solid_volume_meshtying_params.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_beaminteraction_geometry_pair_access_traits.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_to_volume_gauss_point_projection_cross_section.hpp"
#include "4C_geometry_pair_utility_classes.hpp"
#include "4C_linalg_fevector.hpp"

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
  this->evaluate_force_stiff_penalty_potential(
      discret, force_vector, stiffness_matrix, displacement_vector);
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
      this->triad_interpolation_scheme_, this->triad_interpolation_scheme_ref_);
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
