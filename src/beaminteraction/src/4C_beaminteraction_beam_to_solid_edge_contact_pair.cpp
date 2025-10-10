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
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_fad.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
template <typename Beam, typename Edge>
BeamInteraction::BeamToSolidEdgeContactPair<Beam, Edge>::BeamToSolidEdgeContactPair(
    std::shared_ptr<BeamToSolidEdgeContactParameters> beam_to_solid_edge_parameters,
    std::shared_ptr<const Core::Elements::Element> edge_element)
    : BeamContactPair(),
      beam_to_solid_edge_parameters_(beam_to_solid_edge_parameters),
      edge_element_(edge_element)
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
  constexpr int n_dof_fad = Beam::n_dof_ + Edge::n_dof_;

  // Get the current positions
  std::vector<double> beam_centerline_absolute_values(Beam::n_dof_, 0.0);
  BeamInteraction::Utils::extract_pos_dof_vec_absolute_values(
      *discret, element1(), *displacement_vector, beam_centerline_absolute_values);
  auto beam_pos = GeometryPair::InitializeElementData<Beam, scalar_type>::initialize(element1());
  for (unsigned int i_dof = 0; i_dof < Beam::n_dof_; i_dof++)
    beam_pos.element_position_(i_dof) = Core::FADUtils::HigherOrderFadValue<scalar_type>::apply(
        n_dof_fad, i_dof, beam_centerline_absolute_values[i_dof]);

  std::vector<double> edge_centerline_absolute_values(Edge::n_dof_, 0.0);
  BeamInteraction::Utils::extract_pos_dof_vec_absolute_values(
      *discret, edge_element_.get(), *displacement_vector, edge_centerline_absolute_values);
  auto edge_pos =
      GeometryPair::InitializeElementData<Edge, scalar_type>::initialize(edge_element_.get());
  for (unsigned int i_dof = 0; i_dof < Edge::n_dof_; i_dof++)
    edge_pos.element_position_(i_dof) = Core::FADUtils::HigherOrderFadValue<scalar_type>::apply(
        n_dof_fad, Beam::n_dof_ + i_dof, edge_centerline_absolute_values[i_dof]);

  // Closest point projection between the two curves
  scalar_type eta_beam = 0.0;
  scalar_type eta_edge = 0.0;

  // Evaluate the positions on the curves (maybe directly take them from the CPP)
  Core::LinAlg::Matrix<3, 1, scalar_type> r_beam;
  GeometryPair::evaluate_position(eta_beam, beam_pos, r_beam);
  Core::LinAlg::Matrix<3, 1, scalar_type> r_edge;
  GeometryPair::evaluate_position(eta_edge, edge_pos, r_edge);

  // Get beam cross-section diameter.
  const auto* beam_ptr = dynamic_cast<const Discret::Elements::Beam3Base*>(this->element1());
  const double beam_cross_section_radius =
      beam_ptr->get_circular_cross_section_radius_for_interactions();

  // Get the gap function.
  Core::LinAlg::Matrix<3, 1, scalar_type> diff = r_beam;
  diff -= r_edge;
  scalar_type gap = diff.norm2() - beam_cross_section_radius;

  // Get the contact force
  const auto contact_force = penalty_force(gap, beam_to_solid_edge_parameters_->penalty_law);

  // Get the contact force vector
  Core::LinAlg::Matrix<3, 1, scalar_type> contact_force_vector = diff;
  contact_force_vector.scale(1.0 / contact_force_vector.norm2());
  contact_force_vector.scale(contact_force);

  // Get the shape function matrices.
  Core::LinAlg::Matrix<n_dof_fad, 1, scalar_type> pair_force_vector;
  Core::LinAlg::Matrix<1, Beam::n_nodes_ * Beam::n_val_, scalar_type> N_beam;
  Core::LinAlg::Matrix<1, Edge::n_nodes_ * Edge::n_val_, scalar_type> N_edge;
  GeometryPair::EvaluateShapeFunction<Beam>::evaluate(
      N_beam, eta_beam, beam_pos.shape_function_data_);
  GeometryPair::EvaluateShapeFunction<Edge>::evaluate(
      N_edge, eta_edge, edge_pos.shape_function_data_);

  // Calculate pair force vector.
  for (unsigned int i_shape = 0; i_shape < N_beam.num_cols(); i_shape++)
    for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
      pair_force_vector(i_shape * 3 + i_dim) = N_beam(i_shape) * contact_force_vector(i_dim);
  for (unsigned int i_shape = 0; i_shape < N_edge.num_cols(); i_shape++)
    for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
      pair_force_vector(N_beam.num_cols() * 3 + i_shape * 3 + i_dim) =
          -1.0 * N_edge(i_shape) * contact_force_vector(i_dim);
  pair_force_vector.scale(-1.0);

  // Get pair GIDs.
  Core::LinAlg::Matrix<Beam::n_dof_, 1, int> beam_gid;
  Utils::get_element_centerline_gid_indices(*discret, element1(), beam_gid);
  Core::LinAlg::Matrix<Edge::n_dof_, 1, int> edge_gid;

  std::vector<int> lmrow;
  std::vector<int> lmrowowner;
  std::vector<int> lmstride;
  edge_element_->location_vector(*discret, lmrow, lmrowowner, lmstride);
  if (lmrow.size() != Edge::n_dof_)
    FOUR_C_THROW(
        "Dimension of Edge GID does not match, expected {}, got {}", Edge::n_dof_, lmrow.size());

  std::vector<int> pair_dof(n_dof_fad);
  for (unsigned int i_dof = 0; i_dof < Beam::n_dof_; i_dof++)
  {
    pair_dof[i_dof] = beam_gid(i_dof);
  }
  for (unsigned int i_dof = 0; i_dof < Edge::n_dof_; i_dof++)
  {
    pair_dof[i_dof + Beam::n_dof_] = lmrow[i_dof];
  }

  // If given, assemble force terms into the global vector.
  if (force_vector != nullptr)
  {
    std::vector<double> force_pair_double(pair_dof.size(), 0.0);
    for (unsigned int j_dof = 0; j_dof < pair_force_vector.num_rows(); j_dof++)
      force_pair_double[j_dof] = Core::FADUtils::cast_to_double(pair_force_vector(j_dof));
    force_vector->sum_into_global_values(
        pair_dof.size(), pair_dof.data(), force_pair_double.data());
  }

  // If given, assemble force terms into the global stiffness matrix.
  if (stiffness_matrix != nullptr)
  {
    for (unsigned int i_dof = 0; i_dof < pair_force_vector.num_rows(); i_dof++)
    {
      for (unsigned int j_dof = 0; j_dof < n_dof_fad; j_dof++)
      {
        stiffness_matrix->fe_assemble(
            Core::FADUtils::cast_to_double(pair_force_vector(i_dof).dx(j_dof)), pair_dof[i_dof],
            pair_dof[j_dof]);
      }
    }
  }
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
