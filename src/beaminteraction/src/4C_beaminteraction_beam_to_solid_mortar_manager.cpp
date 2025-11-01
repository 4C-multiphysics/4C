// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_mortar_manager.hpp"

#include "4C_beaminteraction_beam_to_solid_params_base.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_contact_params.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_meshtying_params.hpp"
#include "4C_beaminteraction_beam_to_solid_utils.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_params.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_beaminteraction_str_model_evaluator_datastate.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_geometry_pair.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
BeamInteraction::BeamToSolidMortarManager::BeamToSolidMortarManager(
    const std::shared_ptr<const Core::FE::Discretization>& discret,
    const std::shared_ptr<const BeamInteraction::BeamToSolidParamsBase>& params,
    int start_value_lambda_gid)
    : is_setup_(false),
      is_local_maps_build_(false),
      is_global_maps_build_(false),
      start_value_lambda_gid_(start_value_lambda_gid),
      n_lambda_node_translational_(0),
      n_lambda_element_translational_(0),
      n_lambda_node_rotational_(0),
      n_lambda_element_rotational_(0),
      discret_(discret),
      beam_to_solid_params_(params)
{
  // Get the dimension of the Lagrange multiplier field
  const bool is_contact = std::dynamic_pointer_cast<const BeamToSolidSurfaceContactParams>(
                              beam_to_solid_params_) != nullptr;
  const unsigned int n_dim = is_contact ? 1 : 3;

  // Get the number of Lagrange multiplier DOF on a beam node and on a beam element.
  const auto& [n_lambda_node_pos, n_lambda_element_pos] =
      mortar_shape_functions_to_number_of_lagrange_values(
          beam_to_solid_params_, beam_to_solid_params_->get_mortar_shape_function_type(), n_dim);
  n_lambda_node_ = n_lambda_node_pos;
  n_lambda_node_translational_ = n_lambda_node_pos;
  n_lambda_element_ = n_lambda_element_pos;
  n_lambda_element_translational_ = n_lambda_element_pos;

  if (beam_to_solid_params_->is_rotational_coupling())
  {
    // Get the mortar shape functions for rotational coupling
    auto mortar_shape_function_rotation = Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::none;

    const auto beam_to_volume_params =
        std::dynamic_pointer_cast<const BeamInteraction::BeamToSolidVolumeMeshtyingParams>(
            beam_to_solid_params_);
    const auto beam_to_surface_params =
        std::dynamic_pointer_cast<const BeamInteraction::BeamToSolidSurfaceMeshtyingParams>(
            beam_to_solid_params_);

    FOUR_C_ASSERT_ALWAYS(beam_to_volume_params || beam_to_surface_params,
        "The params object should be either of beam-to-solid volume or surface type.");
    if (beam_to_volume_params != nullptr)
    {
      mortar_shape_function_rotation =
          beam_to_volume_params->get_mortar_shape_function_rotation_type();
    }
    else
    {
      mortar_shape_function_rotation = beam_to_surface_params->get_mortar_shape_function_type();
    }

    // Get the number of Lagrange multiplier DOF for rotational coupling on a beam node and on a
    // beam element.
    const auto& [n_lambda_node_rot, n_lambda_element_rot] =
        mortar_shape_functions_to_number_of_lagrange_values(
            beam_to_solid_params_, mortar_shape_function_rotation, n_dim);
    n_lambda_node_ += n_lambda_node_rot;
    n_lambda_node_rotational_ = n_lambda_node_rot;
    n_lambda_element_ += n_lambda_element_rot;
    n_lambda_element_rotational_ = n_lambda_element_rot;
  }
}

/**
 *
 */
void BeamInteraction::BeamToSolidMortarManager::setup()
{
  // Get the global ids of all beam centerline nodes on this rank.
  std::vector<int> my_nodes_gid;
  for (int i_node = 0; i_node < discret_->node_row_map()->num_my_elements(); i_node++)
  {
    Core::Nodes::Node const& node = *(discret_->l_row_node(i_node));
    if (BeamInteraction::Utils::is_beam_centerline_node(node)) my_nodes_gid.push_back(node.id());
  }

  // Get the global ids of all beam elements on this rank.
  std::vector<int> my_elements_gid;
  for (int i_element = 0; i_element < discret_->element_row_map()->num_my_elements(); i_element++)
  {
    Core::Elements::Element const& element = *(discret_->l_row_element(i_element));
    if (BeamInteraction::Utils::is_beam_element(element)) my_elements_gid.push_back(element.id());
  }

  // Calculate the local number of centerline nodes, beam elements and Lagrange multiplier DOF.
  const unsigned int n_nodes = my_nodes_gid.size();
  const unsigned int n_element = my_elements_gid.size();
  const unsigned int n_lambda_dof = n_nodes * n_lambda_node_ + n_element * n_lambda_element_;


  // Tell all other processors how many lambda DOFs this processor has. This information is needed
  // to construct the lambda_dof_rowmap_.
  std::vector<int> lambda_dof_per_rank(Core::Communication::num_mpi_ranks(discret_->get_comm()), 0);
  int temp_my_n_lambda_dof = (int)n_lambda_dof;
  Core::Communication::gather_all(
      &temp_my_n_lambda_dof, lambda_dof_per_rank.data(), 1, discret_->get_comm());

  // Get the start GID for the lambda DOFs on this processor.
  int my_lambda_gid_start_value = start_value_lambda_gid_;
  for (int pid = 0; pid < Core::Communication::my_mpi_rank(discret_->get_comm()); pid++)
    my_lambda_gid_start_value += lambda_dof_per_rank[pid];

  // Fill in all GIDs of the lambda DOFs on this processor (for translations and rotations).
  std::vector<int> my_lambda_gid_translational;
  my_lambda_gid_translational.reserve(
      n_nodes * n_lambda_node_translational_ + n_element * n_lambda_element_translational_);
  std::vector<int> my_lambda_gid_rotational;
  my_lambda_gid_rotational.reserve(
      n_nodes * n_lambda_node_rotational_ + n_element * n_lambda_element_rotational_);
  for (unsigned int i_node = 0; i_node < n_nodes; i_node++)
  {
    for (unsigned int i_dof = 0; i_dof < n_lambda_node_translational_; i_dof++)
      my_lambda_gid_translational.push_back(
          my_lambda_gid_start_value + i_dof + i_node * n_lambda_node_);
    for (unsigned int i_dof = 0; i_dof < n_lambda_node_rotational_; i_dof++)
      my_lambda_gid_rotational.push_back(my_lambda_gid_start_value + i_dof +
                                         i_node * n_lambda_node_ + n_lambda_node_translational_);
  }
  my_lambda_gid_start_value += static_cast<int>(n_nodes * n_lambda_node_);
  for (unsigned int i_element = 0; i_element < n_element; i_element++)
  {
    for (unsigned int i_dof = 0; i_dof < n_lambda_element_translational_; i_dof++)
      my_lambda_gid_translational.push_back(
          my_lambda_gid_start_value + i_dof + i_element * n_lambda_element_);
    for (unsigned int i_dof = 0; i_dof < n_lambda_element_rotational_; i_dof++)
      my_lambda_gid_rotational.push_back(my_lambda_gid_start_value + i_dof +
                                         i_element * n_lambda_element_ +
                                         n_lambda_element_translational_);
  }

  // Rowmap for the additional GIDs used by the mortar contact discretization.
  lambda_dof_rowmap_translations_ =
      std::make_shared<Core::LinAlg::Map>(-1, my_lambda_gid_translational.size(),
          my_lambda_gid_translational.data(), 0, discret_->get_comm());
  lambda_dof_rowmap_rotations_ = std::make_shared<Core::LinAlg::Map>(-1,
      my_lambda_gid_rotational.size(), my_lambda_gid_rotational.data(), 0, discret_->get_comm());
  lambda_dof_rowmap_ =
      Core::LinAlg::merge_map(lambda_dof_rowmap_translations_, lambda_dof_rowmap_rotations_, false);
  // We need to be able to get the global ids for a Lagrange multiplier DOF from the global id
  // of a node or element. To do so, we 'abuse' the Core::LinAlg::MultiVector<double> as map between
  // the global node / element ids and the global Lagrange multiplier DOF ids.
  Core::LinAlg::Map node_gid_rowmap(-1, n_nodes, my_nodes_gid.data(), 0, discret_->get_comm());
  Core::LinAlg::Map element_gid_rowmap(
      -1, n_element, my_elements_gid.data(), 0, discret_->get_comm());

  // Map from global node / element ids to global lagrange multiplier ids. Only create the
  // multivector if it hase one or more columns.
  node_gid_to_lambda_gid_ = nullptr;
  element_gid_to_lambda_gid_ = nullptr;
  if (n_lambda_node_ > 0)
    node_gid_to_lambda_gid_ =
        std::make_shared<Core::LinAlg::MultiVector<double>>(node_gid_rowmap, n_lambda_node_, true);
  if (n_lambda_element_ > 0)
    element_gid_to_lambda_gid_ = std::make_shared<Core::LinAlg::MultiVector<double>>(
        element_gid_rowmap, n_lambda_element_, true);

  // Fill in the entries in the node / element global id to Lagrange multiplier global id vector.
  int lagrange_gid = -1;
  if (node_gid_to_lambda_gid_ != nullptr)
  {
    for (unsigned int i_node = 0; i_node < n_nodes; i_node++)
      for (unsigned int i_lambda = 0; i_lambda < n_lambda_node_; i_lambda++)
      {
        // Get the global Lagrange multiplier id for this node.
        lagrange_gid = lambda_dof_rowmap_->gid(i_node * n_lambda_node_ + i_lambda);

        // Set the global Lagrange multiplier id for this node.
        node_gid_to_lambda_gid_->replace_local_value(i_node, i_lambda, lagrange_gid);
      }
  }
  if (element_gid_to_lambda_gid_ != nullptr)
  {
    for (unsigned int i_element = 0; i_element < n_element; i_element++)
      for (unsigned int i_lambda = 0; i_lambda < n_lambda_element_; i_lambda++)
      {
        // Get the global Lagrange multiplier id for this element.
        lagrange_gid = lambda_dof_rowmap_->gid(
            n_nodes * n_lambda_node_ + i_element * n_lambda_element_ + i_lambda);

        // Set the global Lagrange multiplier id for this element.
        element_gid_to_lambda_gid_->replace_local_value(i_element, i_lambda, lagrange_gid);
      }
  }

  // Create the maps for beam and solid DOFs.
  set_global_maps();

  // Create the global coupling matrices.
  constraint_ = std::make_shared<Core::LinAlg::FEVector<double>>(*lambda_dof_rowmap_);
  constraint_lin_beam_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *lambda_dof_rowmap_, 30, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  constraint_lin_solid_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *lambda_dof_rowmap_, 100, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  force_beam_lin_lambda_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *beam_dof_rowmap_, 30, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  force_solid_lin_lambda_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *solid_dof_rowmap_, 100, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  kappa_ = std::make_shared<Core::LinAlg::FEVector<double>>(*lambda_dof_rowmap_);
  kappa_lin_beam_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *lambda_dof_rowmap_, 30, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  kappa_lin_solid_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *lambda_dof_rowmap_, 100, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  lambda_active_ = std::make_shared<Core::LinAlg::FEVector<double>>(*lambda_dof_rowmap_);

  // Set flag for successful setup.
  is_setup_ = true;
  is_local_maps_build_ = false;
}

/**
 *
 */
void BeamInteraction::BeamToSolidMortarManager::set_global_maps()
{
  // Loop over all nodes on this processor -> we assume all beam and solid DOFs are based on nodes.
  std::vector<int> beam_dofs;
  std::vector<int> solid_dofs;
  for (int i_node = 0; i_node < discret_->node_row_map()->num_my_elements(); i_node++)
  {
    const Core::Nodes::Node* node = discret_->l_row_node(i_node);
    if (BeamInteraction::Utils::is_beam_node(*node))
      discret_->dof(node, beam_dofs);
    else
      discret_->dof(node, solid_dofs);
  }

  auto make_unique_sorted = [](std::vector<int>& vector)
  {
    std::sort(vector.begin(), vector.end());
    vector.erase(std::unique(vector.begin(), vector.end()), vector.end());
  };

  make_unique_sorted(beam_dofs);
  make_unique_sorted(solid_dofs);

  // Create the beam and solid maps.
  beam_dof_rowmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, beam_dofs.size(), beam_dofs.data(), 0, discret_->get_comm());
  solid_dof_rowmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, solid_dofs.size(), solid_dofs.data(), 0, discret_->get_comm());

  // Reset the local maps.
  node_gid_to_lambda_gid_map_.clear();
  element_gid_to_lambda_gid_map_.clear();

  // Set flags for global maps.
  is_global_maps_build_ = true;
  is_local_maps_build_ = false;
}

/**
 *
 */
void BeamInteraction::BeamToSolidMortarManager::set_local_maps(
    const std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>& contact_pairs)
{
  check_setup();
  check_global_maps();

  contact_pairs_ = contact_pairs;

  // At this point the global multi vectors are filled up completely. To get the map for global
  // node element ids to the global lambda ids we need to be able to extract more than the local
  // values on this processor. Therefore we need a new map that contains all rows we want to
  // access in the global multi vector.
  std::vector<int> node_gid_needed;
  std::vector<int> element_gid_needed;

  // Loop over the pairs and get the global node and element indices needed on this rank.
  for (unsigned int i_pair = 0; i_pair < contact_pairs_.size(); i_pair++)
  {
    const std::shared_ptr<BeamInteraction::BeamContactPair>& pair = contact_pairs_[i_pair];

    // The first (beam) element should always be on the same processor as the pair.
    if (pair->element1()->owner() != Core::Communication::my_mpi_rank(discret_->get_comm()))
      FOUR_C_THROW(
          "The current implementation need the first element of a beam contact pair to be on the "
          "same processor as the pair!");

    // Get the global id of the nodes / elements that the pairs on this rank need.
    if (n_lambda_node_ > 0)
      // There are nodal lambda DOFs, add the gid for the nodes in this element to the vector.
      // The first two nodes are the centerline nodes.
      for (unsigned int i_node = 0; i_node < 2; i_node++)
        node_gid_needed.push_back(pair->element1()->nodes()[i_node]->id());

    if (n_lambda_element_ > 0)
      // There are element lambda DOFs, add the gid for this element to the vector.
      element_gid_needed.push_back(pair->element1()->id());
  }

  // Make the entries in the vectors unique.
  std::vector<int>::iterator it;
  std::sort(node_gid_needed.begin(), node_gid_needed.end());
  it = std::unique(node_gid_needed.begin(), node_gid_needed.end());
  node_gid_needed.resize(std::distance(node_gid_needed.begin(), it));
  std::sort(element_gid_needed.begin(), element_gid_needed.end());
  it = std::unique(element_gid_needed.begin(), element_gid_needed.end());
  element_gid_needed.resize(std::distance(element_gid_needed.begin(), it));

  // Create the maps for the extraction of the values.
  Core::LinAlg::Map node_gid_needed_rowmap(
      -1, node_gid_needed.size(), node_gid_needed.data(), 0, discret_->get_comm());
  Core::LinAlg::Map element_gid_needed_rowmap(
      -1, element_gid_needed.size(), element_gid_needed.data(), 0, discret_->get_comm());

  // Create the Multivectors that will be filled with all values needed on this rank.
  std::shared_ptr<Core::LinAlg::MultiVector<double>> node_gid_to_lambda_gid_copy = nullptr;
  std::shared_ptr<Core::LinAlg::MultiVector<double>> element_gid_to_lambda_gid_copy = nullptr;
  if (node_gid_to_lambda_gid_ != nullptr)
    node_gid_to_lambda_gid_copy = std::make_shared<Core::LinAlg::MultiVector<double>>(
        node_gid_needed_rowmap, n_lambda_node_, true);
  if (element_gid_to_lambda_gid_ != nullptr)
    element_gid_to_lambda_gid_copy = std::make_shared<Core::LinAlg::MultiVector<double>>(
        element_gid_needed_rowmap, n_lambda_element_, true);

  // Export values from the global multi vector to the ones needed on this rank.
  if (node_gid_to_lambda_gid_ != nullptr)
    Core::LinAlg::export_to(*node_gid_to_lambda_gid_, *node_gid_to_lambda_gid_copy);
  if (element_gid_to_lambda_gid_ != nullptr)
    Core::LinAlg::export_to(*element_gid_to_lambda_gid_, *element_gid_to_lambda_gid_copy);

  // Fill in the local maps.
  std::vector<int> lambda_gid_for_col_map;
  lambda_gid_for_col_map.clear();
  node_gid_to_lambda_gid_map_.clear();
  element_gid_to_lambda_gid_map_.clear();
  if (node_gid_to_lambda_gid_ != nullptr)
  {
    std::vector<int> temp_node(n_lambda_node_);
    for (int i_node = 0; i_node < node_gid_needed_rowmap.num_my_elements(); i_node++)
    {
      for (unsigned int i_temp = 0; i_temp < n_lambda_node_; i_temp++)
        temp_node[i_temp] =
            (int)(((*node_gid_to_lambda_gid_copy)(i_temp)).local_values_as_span()[i_node]);
      node_gid_to_lambda_gid_map_[node_gid_needed_rowmap.gid(i_node)] = temp_node;
      lambda_gid_for_col_map.insert(
          std::end(lambda_gid_for_col_map), std::begin(temp_node), std::end(temp_node));
    }
  }
  if (element_gid_to_lambda_gid_ != nullptr)
  {
    std::vector<int> temp_elements(n_lambda_element_);
    for (int i_element = 0; i_element < element_gid_needed_rowmap.num_my_elements(); i_element++)
    {
      for (unsigned int i_temp = 0; i_temp < n_lambda_element_; i_temp++)
        temp_elements[i_temp] =
            (int)(((*element_gid_to_lambda_gid_copy)(i_temp)).local_values_as_span()[i_element]);
      element_gid_to_lambda_gid_map_[element_gid_needed_rowmap.gid(i_element)] = temp_elements;
      lambda_gid_for_col_map.insert(
          std::end(lambda_gid_for_col_map), std::begin(temp_elements), std::end(temp_elements));
    }
  }

  // Create the global lambda col map.
  lambda_dof_colmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, lambda_gid_for_col_map.size(), lambda_gid_for_col_map.data(), 0, discret_->get_comm());

  // Set flags for local maps.
  is_local_maps_build_ = true;
}

/**
 *
 */
std::pair<std::vector<int>, std::vector<int>>
BeamInteraction::BeamToSolidMortarManager::location_vector(
    const BeamInteraction::BeamContactPair& contact_pair) const
{
  check_setup();
  check_local_maps();

  // Create the output vectors
  std::vector<int> lambda_pos_row;
  std::vector<int> lambda_rot_row;

  // Get the global DOFs ids of the nodal Lagrange multipliers.
  if (n_lambda_node_ > 0)
  {
    for (int i_node = 0; i_node < contact_pair.element1()->num_node(); i_node++)
    {
      const Core::Nodes::Node& node = *(contact_pair.element1()->nodes()[i_node]);
      if (BeamInteraction::Utils::is_beam_centerline_node(node))
      {
        // Get the global id of the node.
        int node_id = node.id();

        // Check if the id is in the map. If it is, add it to the output vector.
        auto search_key_in_map = node_gid_to_lambda_gid_map_.find(node_id);
        if (search_key_in_map == node_gid_to_lambda_gid_map_.end())
          FOUR_C_THROW("Global node id {} not in map!", node_id);
        const auto node_lambda_gid = search_key_in_map->second;
        for (unsigned int i_pos = 0; i_pos < n_lambda_node_translational_; i_pos++)
        {
          lambda_pos_row.push_back(node_lambda_gid[i_pos]);
        }
        for (unsigned int i_rot = 0; i_rot < n_lambda_node_rotational_; i_rot++)
        {
          lambda_rot_row.push_back(node_lambda_gid[n_lambda_node_translational_ + i_rot]);
        }
      }
    }
  }

  // Get the global DOFs ids of the element Lagrange multipliers.
  if (n_lambda_element_ > 0)
  {
    if (BeamInteraction::Utils::is_beam_element(*contact_pair.element1()))
    {
      // Get the global id of the element.
      int element_id = contact_pair.element1()->id();

      // Check if the id is in the map. If it is, add it to the output vector.
      auto search_key_in_map = element_gid_to_lambda_gid_map_.find(element_id);
      if (search_key_in_map == element_gid_to_lambda_gid_map_.end())
        FOUR_C_THROW("Global element id {} not in map!", element_id);
      const auto element_lambda_gid = search_key_in_map->second;
      for (unsigned int i_pos = 0; i_pos < n_lambda_element_translational_; i_pos++)
      {
        lambda_pos_row.push_back(element_lambda_gid[i_pos]);
      }
      for (unsigned int i_rot = 0; i_rot < n_lambda_element_rotational_; i_rot++)
      {
        lambda_rot_row.push_back(element_lambda_gid[n_lambda_element_translational_ + i_rot]);
      }
    }
  }

  return {lambda_pos_row, lambda_rot_row};
}

/**
 *
 */
void BeamInteraction::BeamToSolidMortarManager::evaluate_force_stiff_penalty_regularization(
    const std::shared_ptr<const Solid::ModelEvaluator::BeamInteractionDataState>& data_state,
    std::shared_ptr<Core::LinAlg::SparseMatrix> stiff,
    std::shared_ptr<Core::LinAlg::FEVector<double>> force)
{
  // Evaluate the global coupling terms
  evaluate_and_assemble_global_coupling_contributions(data_state->get_dis_col_np());

  // Add the penalty terms to the global force and stiffness matrix
  add_global_force_stiffness_penalty_contributions(data_state, stiff, force);
}

/**
 *
 */
void BeamInteraction::BeamToSolidMortarManager::evaluate_coupling_terms_lagrange(
    const std::shared_ptr<const Solid::ModelEvaluator::BeamInteractionDataState>& data_state)
{
  // Evaluate the global coupling terms
  evaluate_and_assemble_global_coupling_contributions(data_state->get_dis_col_np());

  // For now we need to set a link to the global Lagrange multiplier vector in the beam interaction
  // data state.
  global_lambda_ = data_state->get_lambda();
}

/**
 *
 */
bool BeamInteraction::BeamToSolidMortarManager::have_lagrange_dofs() const
{
  return beam_to_solid_params_ &&
         beam_to_solid_params_->get_constraint_enforcement() ==
             Inpar::BeamToSolid::BeamToSolidConstraintEnforcement::lagrange;
}

/**
 *
 */
std::shared_ptr<Core::LinAlg::Vector<double>>
BeamInteraction::BeamToSolidMortarManager::get_global_lambda() const
{
  if (have_lagrange_dofs())
  {
    auto global_lambda_current_rowmap =
        std::make_shared<Core::LinAlg::Vector<double>>(*lambda_dof_rowmap_);

    Core::LinAlg::export_to(*global_lambda_, *global_lambda_current_rowmap);
    return global_lambda_current_rowmap;
  }

  auto penalty_regularization = get_penalty_regularization(false);
  return std::get<0>(penalty_regularization);
}

/**
 *
 */
std::shared_ptr<Core::LinAlg::Vector<double>>
BeamInteraction::BeamToSolidMortarManager::get_global_lambda_col() const
{
  std::shared_ptr<Core::LinAlg::Vector<double>> lambda_col =
      std::make_shared<Core::LinAlg::Vector<double>>(*lambda_dof_colmap_);
  Core::LinAlg::export_to(*get_global_lambda(), *lambda_col);
  return lambda_col;
}

/**
 *
 */
double BeamInteraction::BeamToSolidMortarManager::get_energy() const
{
  // Since this value is also computed for the reference configuration, where the global mortar
  // matrices are not build yet we return 0 in this case.
  if (not constraint_lin_beam_->filled() or not constraint_lin_solid_->filled() or
      not force_beam_lin_lambda_->filled() or not force_solid_lin_lambda_->filled())
    return 0.0;

  // Calculate the penalty potential.
  std::shared_ptr<Core::LinAlg::Vector<double>> lambda = get_global_lambda();
  double dot_product = 0.0;
  constraint_->dot(*lambda, &dot_product);
  return 0.5 * dot_product;
}

/**
 *
 */
void BeamInteraction::BeamToSolidMortarManager::evaluate_and_assemble_global_coupling_contributions(
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement_vector)
{
  check_setup();
  check_global_maps();

  // Reset the global data structures.
  constraint_->put_scalar(0.);
  constraint_lin_beam_->put_scalar(0.);
  constraint_lin_solid_->put_scalar(0.);
  force_beam_lin_lambda_->put_scalar(0.);
  force_solid_lin_lambda_->put_scalar(0.);
  kappa_->put_scalar(0.);
  kappa_lin_beam_->put_scalar(0.);
  kappa_lin_solid_->put_scalar(0.);
  lambda_active_->put_scalar(0.);

  for (auto& elepairptr : contact_pairs_)
  {
    // Evaluate the mortar contributions of the pair and the pair assembles the terms into the
    // global matrices.
    elepairptr->evaluate_and_assemble_mortar_contributions(*discret_, this, *constraint_lin_beam_,
        *constraint_lin_solid_, *force_beam_lin_lambda_, *force_solid_lin_lambda_, *constraint_,
        *kappa_, *kappa_lin_beam_, *kappa_lin_solid_, *lambda_active_, displacement_vector);
  }

  // Complete the global mortar matrices.
  constraint_lin_beam_->complete(*beam_dof_rowmap_, *lambda_dof_rowmap_);
  constraint_lin_solid_->complete(*solid_dof_rowmap_, *lambda_dof_rowmap_);
  force_beam_lin_lambda_->complete(*lambda_dof_rowmap_, *beam_dof_rowmap_);
  force_solid_lin_lambda_->complete(*lambda_dof_rowmap_, *solid_dof_rowmap_);
  kappa_lin_beam_->complete(*beam_dof_rowmap_, *lambda_dof_rowmap_);
  kappa_lin_solid_->complete(*solid_dof_rowmap_, *lambda_dof_rowmap_);

  // Complete the global scaling vector.
  kappa_->complete();
  lambda_active_->complete();
  constraint_->complete();
}

/**
 *
 */
void BeamInteraction::BeamToSolidMortarManager::add_global_force_stiffness_penalty_contributions(
    const std::shared_ptr<const Solid::ModelEvaluator::BeamInteractionDataState>& data_state,
    std::shared_ptr<Core::LinAlg::SparseMatrix> stiff,
    std::shared_ptr<Core::LinAlg::FEVector<double>> force) const
{
  check_setup();
  check_global_maps();

  // Get the penalty regularization
  const bool is_stiff = stiff != nullptr;
  auto penalty_regularization = get_penalty_regularization(is_stiff);
  const auto lambda = std::get<0>(penalty_regularization);

  if (is_stiff)
  {
    // Penalty regularization linearized w.r.t. the constraint equations
    Core::LinAlg::SparseMatrix penalty_regularization_lin_constraint(
        *std::get<1>(penalty_regularization));
    penalty_regularization_lin_constraint.complete();
    auto regularized_constraint_lin_beam =
        Core::LinAlg::matrix_multiply(penalty_regularization_lin_constraint, false,
            *constraint_lin_beam_, false, false, false, true);
    auto regularized_constraint_lin_solid =
        Core::LinAlg::matrix_multiply(penalty_regularization_lin_constraint, false,
            *constraint_lin_solid_, false, false, false, true);

    // Penalty regularization linearized w.r.t. the scaling vector
    if (kappa_lin_beam_->NormInf() > 1e-12 && kappa_lin_solid_->NormInf() > 1e-12)
    {
      Core::LinAlg::SparseMatrix penalty_regularization_lin_kappa(
          *std::get<2>(penalty_regularization));
      penalty_regularization_lin_kappa.complete();
      const auto kappa_lin_beam_scaled = Core::LinAlg::matrix_multiply(
          penalty_regularization_lin_kappa, false, *kappa_lin_beam_, false, false, false, true);
      const auto kappa_lin_solid_scaled = Core::LinAlg::matrix_multiply(
          penalty_regularization_lin_kappa, false, *kappa_lin_solid_, false, false, false, true);
      regularized_constraint_lin_beam->add(*kappa_lin_beam_scaled, false, 1.0, 1.0);
      regularized_constraint_lin_solid->add(*kappa_lin_solid_scaled, false, 1.0, 1.0);
    }

    // Calculate the needed submatrices
    const auto force_beam_lin_lambda_times_constraint_lin_beam =
        Core::LinAlg::matrix_multiply(*force_beam_lin_lambda_, false,
            *regularized_constraint_lin_beam, false, false, false, true);
    const auto force_beam_lin_lambda_times_constraint_lin_solid =
        Core::LinAlg::matrix_multiply(*force_beam_lin_lambda_, false,
            *regularized_constraint_lin_solid, false, false, false, true);
    const auto force_solid_lin_lambda_times_constraint_lin_beam =
        Core::LinAlg::matrix_multiply(*force_solid_lin_lambda_, false,
            *regularized_constraint_lin_beam, false, false, false, true);
    const auto force_solid_lin_lambda_times_constraint_lin_solid =
        Core::LinAlg::matrix_multiply(*force_solid_lin_lambda_, false,
            *regularized_constraint_lin_solid, false, false, false, true);

    // Add contributions to the global stiffness matrix
    stiff->add(*force_beam_lin_lambda_times_constraint_lin_beam, false, 1.0, 1.0);
    stiff->add(*force_beam_lin_lambda_times_constraint_lin_solid, false, 1.0, 1.0);
    stiff->add(*force_solid_lin_lambda_times_constraint_lin_beam, false, 1.0, 1.0);
    stiff->add(*force_solid_lin_lambda_times_constraint_lin_solid, false, 1.0, 1.0);
  }

  if (force != nullptr)
  {
    // Factor for right hand side (forces). 1 corresponds to the mesh-tying forces being added to
    // the right hand side, -1 to the left hand side.
    const double rhs_factor = -1.0;

    // Multiply the lambda vector with FB_L and FS_L to get the forces on the beam and solid,
    // respectively.
    Core::LinAlg::Vector<double> beam_force(*beam_dof_rowmap_);
    Core::LinAlg::Vector<double> solid_force(*solid_dof_rowmap_);
    beam_force.put_scalar(0.);
    solid_force.put_scalar(0.);
    force_beam_lin_lambda_->multiply(false, *lambda, beam_force);
    force_solid_lin_lambda_->multiply(false, *lambda, solid_force);
    Core::LinAlg::Vector<double> global_temp(*discret_->dof_row_map());
    Core::LinAlg::export_to(beam_force, global_temp);
    Core::LinAlg::export_to(solid_force, global_temp);

    // Add force contributions to global vector.
    force->update(-1.0 * rhs_factor, global_temp, 1.0);
  }

  // Add the force and stiffness contributions that are assembled directly by the pairs.
  Core::LinAlg::Vector<double> lambda_col(*lambda_dof_colmap_);
  Core::LinAlg::export_to(*lambda, lambda_col);
  for (const auto& elepairptr : contact_pairs_)
    elepairptr->evaluate_and_assemble(
        *discret_, this, force, stiff, lambda_col, *data_state->get_dis_col_np());
}

/**
 *
 */
std::tuple<std::shared_ptr<Core::LinAlg::Vector<double>>,
    std::shared_ptr<Core::LinAlg::Vector<double>>, std::shared_ptr<Core::LinAlg::Vector<double>>>
BeamInteraction::BeamToSolidMortarManager::get_penalty_regularization(
    const bool compute_linearization) const
{
  check_setup();
  check_global_maps();

  // Get the inverted kappa matrix.
  std::shared_ptr<Core::LinAlg::Vector<double>> penalty_kappa_inv = penalty_invert_kappa();
  Core::LinAlg::SparseMatrix penalty_kappa_inv_mat(*penalty_kappa_inv);
  penalty_kappa_inv_mat.complete();

  // Multiply the inverted kappa matrix with the constraint equations.
  std::shared_ptr<Core::LinAlg::Vector<double>> lambda =
      std::make_shared<Core::LinAlg::Vector<double>>(*lambda_dof_rowmap_);

  penalty_kappa_inv_mat.multiply(false, Core::LinAlg::Vector<double>(*constraint_), *lambda);

  if (compute_linearization)
  {
    return {lambda, penalty_kappa_inv, nullptr};
  }
  else
  {
    return {lambda, nullptr, nullptr};
  }
}

/**
 *
 */
std::shared_ptr<Core::LinAlg::Vector<double>>
BeamInteraction::BeamToSolidMortarManager::penalty_invert_kappa() const
{
  // Create the inverse vector.
  std::shared_ptr<Core::LinAlg::Vector<double>> kappa_inv =
      std::make_shared<Core::LinAlg::Vector<double>>(*lambda_dof_rowmap_);

  // Get the penalty parameters.
  const double penalty_translation = beam_to_solid_params_->get_penalty_parameter();
  double penalty_rotation = 0.0;
  auto beam_to_volume_params =
      std::dynamic_pointer_cast<const BeamInteraction::BeamToSolidVolumeMeshtyingParams>(
          beam_to_solid_params_);
  auto beam_to_surface_params =
      std::dynamic_pointer_cast<const BeamInteraction::BeamToSolidSurfaceMeshtyingParams>(
          beam_to_solid_params_);
  if (beam_to_volume_params != nullptr)
    penalty_rotation = beam_to_volume_params->get_rotational_coupling_penalty_parameter();
  else if (beam_to_surface_params != nullptr)
    penalty_rotation = beam_to_surface_params->get_rotational_coupling_penalty_parameter();
  else if (lambda_dof_rowmap_rotations_->num_global_elements() > 0)
    FOUR_C_THROW(
        "Rotational penalty coupling only implemented for beam-to-volume and beam-to-surface "
        "case.");

  // Calculate the local inverse of kappa.
  double penalty = 0.0;
  double local_kappa_inv_value = 0.;
  for (int lid = 0; lid < lambda_dof_rowmap_->num_my_elements(); lid++)
  {
    if (lambda_active_->get_values()[lid] > 0.1)
    {
      const int gid = lambda_dof_rowmap_->gid(lid);
      if (lambda_dof_rowmap_translations_->lid(gid) != -1)
        penalty = penalty_translation;
      else if (lambda_dof_rowmap_rotations_->lid(gid) != -1)
        penalty = penalty_rotation;
      else
        FOUR_C_THROW("Could not find the GID {} in translation or rotation map", gid);

      local_kappa_inv_value = penalty / kappa_->get_values()[lid];
    }

    else
      // This LID is inactive.
      local_kappa_inv_value = 0.0;

    kappa_inv->replace_local_value(lid, local_kappa_inv_value);
  }

  return kappa_inv;
}

/**
 *
 */
void BeamInteraction::BeamToSolidMortarManager::assemble_force(
    const Solid::TimeInt::BaseDataGlobalState& gstate, Core::LinAlg::Vector<double>& f,
    const Solid::ModelEvaluator::BeamInteractionDataState& data_state) const
{
  if (!have_lagrange_dofs())
    FOUR_C_THROW("assemble_force only possible with Langrange multiplier constraint enforcement");
  auto constraint_rhs_map = Core::LinAlg::Vector<double>(f.get_map());
  Core::LinAlg::export_to(*constraint_, constraint_rhs_map);

  f.update(1., constraint_rhs_map, 1.);
}

/**
 *
 */
void BeamInteraction::BeamToSolidMortarManager::assemble_stiff(
    const Solid::TimeInt::BaseDataGlobalState& gstate, Core::LinAlg::SparseOperator& jac,
    const Solid::ModelEvaluator::BeamInteractionDataState& data_state) const
{
  if (!have_lagrange_dofs())
    FOUR_C_THROW("assemble_stiff only possible with Langrange multiplier constraint enforcement");
  std::shared_ptr<const Core::LinAlg::SparseOperator> jac_ptr(
      &jac, [](Core::LinAlg::SparseOperator*) {});
  std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase> jac_block_sparse_matrix_base =
      Core::LinAlg::cast_to_const_block_sparse_matrix_base_and_check_success(jac_ptr);
  auto block_lm_displ_row_map = jac_block_sparse_matrix_base->matrix(1, 0).row_map();
  auto block_displ_lm_row_map = jac_block_sparse_matrix_base->matrix(0, 1).row_map();

  const auto lagrange_formulation = beam_to_solid_params_->get_lagrange_formulation();
  if (lagrange_formulation == Inpar::BeamToSolid::BeamToSolidLagrangeFormulation::regularized)
  {
    // Set penalty entry
    const double penalty_translation = beam_to_solid_params_->get_penalty_parameter();
    auto kappa_vector = Core::LinAlg::Vector<double>(block_lm_displ_row_map);
    Core::LinAlg::export_to(*kappa_, kappa_vector);
    Core::LinAlg::SparseMatrix kappa_penalty_inv_mat(kappa_vector);
    kappa_penalty_inv_mat.scale(-1.0 / penalty_translation);
    kappa_penalty_inv_mat.complete();
    gstate.assign_model_block(jac, kappa_penalty_inv_mat, Inpar::Solid::model_beaminteraction,
        Solid::MatBlockType::lm_lm);
  }

  Core::LinAlg::SparseMatrix lm_displ =
      Core::LinAlg::SparseMatrix(*lambda_dof_rowmap_, 81, true, true);
  lm_displ.add(*constraint_lin_beam_, false, 1.0, 0.0);
  lm_displ.add(*constraint_lin_solid_, false, 1.0, 1.0);
  lm_displ.complete(*discret_->dof_row_map(), *lambda_dof_rowmap_);

  std::shared_ptr<Core::LinAlg::SparseMatrix> lm_displ_in_global_layout =
      Core::LinAlg::matrix_row_col_transform(
          lm_displ, block_lm_displ_row_map, jac_block_sparse_matrix_base->domain_map(0));
  gstate.assign_model_block(jac, *lm_displ_in_global_layout, Inpar::Solid::model_beaminteraction,
      Solid::MatBlockType::lm_displ);

  Core::LinAlg::SparseMatrix displ_lm =
      Core::LinAlg::SparseMatrix(*discret_->dof_row_map(), 81, true, true);
  displ_lm.add(*force_beam_lin_lambda_, false, 1.0, 0.0);
  displ_lm.add(*force_solid_lin_lambda_, false, 1.0, 1.0);
  displ_lm.complete(*lambda_dof_rowmap_, *discret_->dof_row_map());
  std::shared_ptr<Core::LinAlg::SparseMatrix> displ_lm_in_global_layout =
      Core::LinAlg::matrix_row_col_transform(
          displ_lm, block_displ_lm_row_map, jac_block_sparse_matrix_base->domain_map(1));
  gstate.assign_model_block(jac, *displ_lm_in_global_layout, Inpar::Solid::model_beaminteraction,
      Solid::MatBlockType::displ_lm);
}


FOUR_C_NAMESPACE_CLOSE
