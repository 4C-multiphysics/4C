// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_edge_contact_pair.hpp"

#include "4C_beam3_reissner.hpp"
#include "4C_beam3_triad_interpolation_local_rotation_vectors.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_geometry_pair_access_traits.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename Beam, typename Edge>
BeamInteraction::BeamToSolidEdgeContactPair<Beam, Edge>::BeamToSolidEdgeContactPair(
    std::shared_ptr<BeamToSolidEdgeContactParameters> beam_to_solid_edge_parameters)
    : BeamContactPair(), beam_to_solid_edge_parameters_(beam_to_solid_edge_parameters)
{
  // Empty constructor.
}

/**
 *
 */
template <typename Beam, typename Edge>
void BeamInteraction::BeamToSolidEdgeContactPair<Beam, Edge>::setup()
{
  this->issetup_ = true;
}

/**
 *
 */
template <typename Beam, typename Edge>
void BeamInteraction::BeamToSolidEdgeContactPair<Beam, Edge>::evaluate_and_assemble(
    const std::shared_ptr<const Core::FE::Discretization>& discret,
    const std::shared_ptr<Core::LinAlg::FEVector<double>>& force_vector,
    const std::shared_ptr<Core::LinAlg::SparseMatrix>& stiffness_matrix,
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector)
{
}

/**
 *
 */
template <typename Beam, typename Edge>
void BeamInteraction::BeamToSolidEdgeContactPair<Beam, Edge>::print(std::ostream& out) const
{
  check_init_setup();

  // Print some general information: Element IDs and dofvecs.
  out << "\n------------------------------------------------------------------------";
  out << "\nInstance of BeamToSolidEdgeContactPair"
      << "\nBeam EleGID:  " << element1()->id() << "\nEdge EleGID: " << element2()->id();
  out << "------------------------------------------------------------------------\n";
}

/**
 *
 */
template <typename Beam, typename Edge>
void BeamInteraction::BeamToSolidEdgeContactPair<Beam,
    Edge>::print_summary_one_line_per_active_segment_pair(std::ostream& out) const
{
  check_init_setup();

  out << "Beam-to-solid edge contact pair, beam gid: " << element1()->id()
      << " solid gid: " << element2()->id();
}

/**
 * Explicit template initialization of template class.
 */
namespace BeamInteraction
{
  using namespace GeometryPair;

  template class BeamInteraction::BeamToSolidEdgeContactPair<t_hermite, t_line2>;
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE
