// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fbi_beam_to_fluid_mortar_manager.hpp"

#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_params.hpp"
#include "4C_fbi_calc_utils.hpp"
#include "4C_fbi_input.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
BeamInteraction::BeamToFluidMortarManager::BeamToFluidMortarManager(
    std::shared_ptr<const Core::FE::Discretization> discretization1,
    std::shared_ptr<const Core::FE::Discretization> discretization2,
    std::shared_ptr<const FBI::BeamToFluidMeshtyingParams> params, int start_value_lambda_gid)
    : is_setup_(false),
      is_local_maps_build_(false),
      is_global_maps_build_(false),
      start_value_lambda_gid_(start_value_lambda_gid),
      discretization_structure_(discretization1),
      discretization_fluid_(discretization2),
      beam_contact_parameters_ptr_(params),
      lambda_dof_rowmap_(nullptr),
      lambda_dof_colmap_(nullptr),
      beam_dof_rowmap_(nullptr),
      fluid_dof_rowmap_(nullptr),
      node_gid_to_lambda_gid_(nullptr),
      element_gid_to_lambda_gid_(nullptr),
      global_d_(nullptr),
      global_m_(nullptr),
      global_kappa_(nullptr),
      global_active_lambda_(nullptr)
{
  // Get the number of Lagrange multiplier DOF on a beam node and on a beam element.
  switch (params->get_mortar_shape_function_type())
  {
    case FBI::BeamToFluidMeshtingMortarShapefunctions::line2:
    {
      n_lambda_node_ = 1 * 3;
      n_lambda_element_ = 0 * 3;
      break;
    }
    case FBI::BeamToFluidMeshtingMortarShapefunctions::line3:
    {
      n_lambda_node_ = 1 * 3;
      n_lambda_element_ = 1 * 3;
      break;
    }
    case FBI::BeamToFluidMeshtingMortarShapefunctions::line4:
    {
      n_lambda_node_ = 1 * 3;
      n_lambda_element_ = 2 * 3;
      break;
    }
    default:
      FOUR_C_THROW("Mortar shape function not implemented!");
  }
}

/**
 *
 */
void BeamInteraction::BeamToFluidMortarManager::setup()
{
  // Get the global ids of all beam centerline nodes on this rank.
  std::vector<int> my_nodes_gid;
  for (int i_node = 0; i_node < discretization_structure_->node_row_map()->num_my_elements();
      i_node++)
  {
    Core::Nodes::Node const& node = *(discretization_structure_->l_row_node(i_node));
    if (BeamInteraction::Utils::is_beam_centerline_node(node)) my_nodes_gid.push_back(node.id());
  }

  // Get the global ids of all beam elements on this rank.
  std::vector<int> my_elements_gid;
  for (int i_element = 0;
      i_element < discretization_structure_->element_row_map()->num_my_elements(); i_element++)
  {
    Core::Elements::Element const& element = *(discretization_structure_->l_row_element(i_element));
    if (BeamInteraction::Utils::is_beam_element(element)) my_elements_gid.push_back(element.id());
  }

  // Calculate the local number of centerline nodes, beam elements and Lagrange multiplier DOF.
  const unsigned int n_nodes = my_nodes_gid.size();
  const unsigned int n_element = my_elements_gid.size();
  const unsigned int n_lambda_dof = n_nodes * n_lambda_node_ + n_element * n_lambda_element_;


  // Tell all other processors how many lambda DOFs this processor has. This information is needed
  // to construct the lambda_dof_rowmap_.
  std::vector<int> lambda_dof_per_rank(
      Core::Communication::num_mpi_ranks(discretization_structure_->get_comm()), 0);
  int temp_my_n_lambda_dof = (int)n_lambda_dof;
  Core::Communication::gather_all(
      &temp_my_n_lambda_dof, lambda_dof_per_rank.data(), 1, discretization_structure_->get_comm());

  // Get the start GID for the lambda DOFs on this processor.
  int my_lambda_gid_start_value = start_value_lambda_gid_;
  for (int pid = 0; pid < Core::Communication::my_mpi_rank(discretization_structure_->get_comm());
      pid++)
    my_lambda_gid_start_value += lambda_dof_per_rank[pid];

  // Fill in all GIDs of the lambda DOFs on this processor.
  std::vector<int> my_lambda_gid(n_lambda_dof, 0);
  for (int my_lid = 0; my_lid < (int)n_lambda_dof; my_lid++)
    my_lambda_gid[my_lid] = my_lambda_gid_start_value + my_lid;

  // Rowmap for the additional GIDs used by the mortar contact discretization.
  lambda_dof_rowmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, my_lambda_gid.size(), my_lambda_gid.data(), 0, discretization_structure_->get_comm());


  // We need to be able to get the global ids for a Lagrange multiplier DOF from the global id
  // of a node or element. To do so, we 'abuse' the Core::LinAlg::MultiVector<double> as map between
  // the global node / element ids and the global Lagrange multiplier DOF ids.
  Core::LinAlg::Map node_gid_rowmap(
      -1, n_nodes, my_nodes_gid.data(), 0, discretization_structure_->get_comm());
  Core::LinAlg::Map element_gid_rowmap(
      -1, n_element, my_elements_gid.data(), 0, discretization_structure_->get_comm());

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

  // Create the global mortar matrices.
  global_d_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *lambda_dof_rowmap_, 30, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  global_m_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *lambda_dof_rowmap_, 100, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  global_kappa_ = std::make_shared<Core::LinAlg::FEVector<double>>(*lambda_dof_rowmap_);
  global_active_lambda_ = std::make_shared<Core::LinAlg::FEVector<double>>(*lambda_dof_rowmap_);

  // Create the maps for beam and solid DOFs.
  set_global_maps();

  // Set flag for successful setup.
  is_setup_ = true;
  is_local_maps_build_ = false;
}

/**
 *
 */
void BeamInteraction::BeamToFluidMortarManager::set_global_maps()
{
  // Loop over all nodes on this processor -> we assume all beam and fluid DOFs are based on nodes.
  std::vector<std::vector<int>> field_dofs(2);

  for (int i_node = 0; i_node < discretization_structure_->node_row_map()->num_my_elements();
      i_node++)
  {
    const Core::Nodes::Node* node = discretization_structure_->l_row_node(i_node);
    if (BeamInteraction::Utils::is_beam_node(*node))
      discretization_structure_->dof(node, field_dofs[0]);
    else
      FOUR_C_THROW("The given structure element is not a beam element!");
  }
  for (int i_node = 0; i_node < discretization_fluid_->node_row_map()->num_my_elements(); i_node++)
  {
    const Core::Nodes::Node* node = discretization_fluid_->l_row_node(i_node);
    discretization_fluid_->dof(node, field_dofs[1]);
  }

  // Create the beam and fluid maps.
  beam_dof_rowmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, field_dofs[0].size(), field_dofs[0].data(), 0, discretization_structure_->get_comm());
  fluid_dof_rowmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, field_dofs[1].size(), field_dofs[1].data(), 0, discretization_fluid_->get_comm());

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
void BeamInteraction::BeamToFluidMortarManager::set_local_maps(
    const std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>& contact_pairs)
{
  check_setup();
  check_global_maps();

  // At this point the global multi vectors are filled up completely. To get the map for global
  // node element ids to the global lambda ids we need to be able to extract more than the local
  // values on this processor. Therefore we need a new map that contains all rows we want to
  // access in the global multi vector.
  std::vector<int> node_gid_needed;
  std::vector<int> element_gid_needed;

  // Loop over the pairs and get the global node and element indices needed on this rank.
  for (unsigned int i_pair = 0; i_pair < contact_pairs.size(); i_pair++)
  {
    const std::shared_ptr<BeamInteraction::BeamContactPair>& pair = contact_pairs[i_pair];

    // The first (beam) element should always be on the same processor as the pair.
    if (pair->element1()->owner() !=
        Core::Communication::my_mpi_rank(discretization_structure_->get_comm()))
      FOUR_C_THROW(
          "The current implementation needs the first element of a beam contact pair to be on the "
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
      -1, node_gid_needed.size(), node_gid_needed.data(), 0, discretization_structure_->get_comm());
  Core::LinAlg::Map element_gid_needed_rowmap(-1, element_gid_needed.size(),
      element_gid_needed.data(), 0, discretization_structure_->get_comm());

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
        temp_node[i_temp] = (int)((*node_gid_to_lambda_gid_copy)(i_temp)[i_node]);
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
        temp_elements[i_temp] = (int)((*element_gid_to_lambda_gid_copy)(i_temp)[i_element]);
      element_gid_to_lambda_gid_map_[element_gid_needed_rowmap.gid(i_element)] = temp_elements;
      lambda_gid_for_col_map.insert(
          std::end(lambda_gid_for_col_map), std::begin(temp_elements), std::end(temp_elements));
    }
  }

  // Create the global lambda col map.
  lambda_dof_colmap_ = std::make_shared<Core::LinAlg::Map>(-1, lambda_gid_for_col_map.size(),
      lambda_gid_for_col_map.data(), 0, discretization_structure_->get_comm());

  // Set flags for local maps.
  is_local_maps_build_ = true;
}

/**
 *
 */
void BeamInteraction::BeamToFluidMortarManager::location_vector(
    const BeamInteraction::BeamContactPair& contact_pair, std::vector<int>& lambda_row) const
{
  check_setup();
  check_local_maps();

  // Clear the output vectors.
  lambda_row.clear();

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
        for (auto const& lambda_gid : search_key_in_map->second) lambda_row.push_back(lambda_gid);
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
      for (auto const& lambda_gid : search_key_in_map->second) lambda_row.push_back(lambda_gid);
    }
  }
}

/**
 *
 */
void BeamInteraction::BeamToFluidMortarManager::evaluate_global_dm(
    const std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>& contact_pairs)
{
  check_setup();
  check_global_maps();

  // Clear the old values of D, M and kappa.
  int linalg_error = 0;
  linalg_error = global_d_->put_scalar(0.);
  if (linalg_error != 0) FOUR_C_THROW("Error in PutScalar!");
  linalg_error = global_m_->put_scalar(0.);
  if (linalg_error != 0) FOUR_C_THROW("Error in PutScalar!");
  global_kappa_->put_scalar(0.);

  // Local mortar matrices that will be filled up by EvaluateDM.
  Core::LinAlg::SerialDenseMatrix local_D_centerlineDOFs;
  Core::LinAlg::SerialDenseMatrix local_M;
  Core::LinAlg::SerialDenseVector dummy_constraint_offset;
  Core::LinAlg::SerialDenseVector local_kappa;

  // For the D matrix we need to assemble the centerline DOF to the element dof. This is done
  // into this matrix.
  Core::LinAlg::SerialDenseMatrix local_D_elementDOFs;

  // Flag if pair has a active contribution.
  bool mortar_is_active = false;

  // Loop over elements and assemble the local D and M matrices into the global ones.
  for (auto& elepairptr : contact_pairs)
  {
    // Evaluate the mortar contributions on the pair, if there are some, assemble into the global
    // matrices.
    mortar_is_active = elepairptr->evaluate_dm(
        local_D_centerlineDOFs, local_M, local_kappa, dummy_constraint_offset);

    if (mortar_is_active)
    {
      // We got contributions from the pair. Now we have to assemble into the global matrices. We
      // use the FEAssembly here, since the contact pairs are not ghosted.

      // Assemble the centerline matrix calculated by EvaluateDM into the full element matrix.
      BeamInteraction::Utils::assemble_centerline_dof_col_matrix_into_element_col_matrix(
          *discretization_structure_, elepairptr->element1(), local_D_centerlineDOFs,
          local_D_elementDOFs);

      // Get the GIDs of the Lagrange multipliers.
      std::vector<int> lambda_row;
      location_vector(*elepairptr, lambda_row);

      // Get the GIDs of the beam DOF.
      std::vector<int> beam_row;
      std::vector<int> fluid_temp;
      std::vector<int> fluid_row;
      std::vector<int> dummy_1;
      std::vector<int> dummy_2;
      elepairptr->element1()->location_vector(
          *discretization_structure_, beam_row, dummy_1, dummy_2);
      elepairptr->element2()->location_vector(*discretization_fluid_, fluid_temp, dummy_1, dummy_2);
      for (unsigned int i = 0; i < fluid_temp.size(); i++)
      {
        if ((i + 1) % 4) fluid_row.push_back(fluid_temp[i]);
      }

      // Save check the matrix sizes.
      if (local_D_elementDOFs.numRows() != (int)lambda_row.size() &&
          local_D_elementDOFs.numCols() != (int)beam_row.size())
        FOUR_C_THROW("Size of local D matrix does not match the GID vectors!");
      if (local_M.numRows() != (int)lambda_row.size() && local_M.numCols() != (int)fluid_row.size())
        FOUR_C_THROW("Size of local M matrix does not match the GID vectors!");
      if (local_kappa.numRows() != (int)lambda_row.size() && local_kappa.numCols() != 1)
        FOUR_C_THROW("Size of local kappa vector does not match the GID vector!");

      // Assemble into the global matrices.
      global_d_->fe_assemble(local_D_elementDOFs, lambda_row, beam_row);
      global_m_->fe_assemble(local_M, lambda_row, fluid_row);
      global_kappa_->sum_into_global_values(
          local_kappa.numRows(), lambda_row.data(), local_kappa.values());

      // Set all entries in the local kappa vector to 1 and add them to the active vector.
      for (int i_local = 0; i_local < local_kappa.numRows(); i_local++) local_kappa(i_local) = 1.;
      global_active_lambda_->sum_into_global_values(
          local_kappa.numRows(), lambda_row.data(), local_kappa.values());
    }
  }

  // Complete the global mortar matrices.
  global_d_->complete(*beam_dof_rowmap_, *lambda_dof_rowmap_);
  global_m_->complete(*fluid_dof_rowmap_, *lambda_dof_rowmap_);

  // Complete the global scaling vector.
  global_kappa_->complete();
  global_active_lambda_->complete();
}

/**
 *
 */
void BeamInteraction::BeamToFluidMortarManager::add_global_force_stiffness_contributions(
    std::shared_ptr<Core::LinAlg::FEVector<double>> fluid_force,
    Core::LinAlg::FEVector<double>& beam_force, std::shared_ptr<Core::LinAlg::SparseMatrix> kbb,
    std::shared_ptr<Core::LinAlg::SparseMatrix> kbf,
    std::shared_ptr<Core::LinAlg::SparseMatrix> kff,
    std::shared_ptr<Core::LinAlg::SparseMatrix> kfb,
    std::shared_ptr<const Core::LinAlg::Vector<double>> beam_vel,
    std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_vel) const
{
  check_setup();
  check_global_maps();

  int linalg_error = 0;

  // Scale D and M with kappa^-1.
  std::shared_ptr<Core::LinAlg::Vector<double>> global_kappa_inv = invert_kappa();
  Core::LinAlg::SparseMatrix kappa_inv_mat(*global_kappa_inv);
  kappa_inv_mat.complete();
  std::shared_ptr<Core::LinAlg::SparseMatrix> global_D_scaled =
      Core::LinAlg::matrix_multiply(kappa_inv_mat, false, *global_d_, false, false, false, true);
  std::shared_ptr<Core::LinAlg::SparseMatrix> global_M_scaled =
      Core::LinAlg::matrix_multiply(kappa_inv_mat, false, *global_m_, false, false, false, true);

  // Calculate the needed submatrices.
  std::shared_ptr<Core::LinAlg::SparseMatrix> Dt_kappa_D =
      Core::LinAlg::matrix_multiply(*global_d_, true, *global_D_scaled, false, false, false, true);
  std::shared_ptr<Core::LinAlg::SparseMatrix> Dt_kappa_M =
      Core::LinAlg::matrix_multiply(*global_d_, true, *global_M_scaled, false, false, false, true);
  std::shared_ptr<Core::LinAlg::SparseMatrix> Mt_kappa_M =
      Core::LinAlg::matrix_multiply(*global_m_, true, *global_M_scaled, false, false, false, true);

  if (kff != nullptr) kff->add(*Mt_kappa_M, false, 1.0, 1.0);

  if (kfb != nullptr) kbf->add(*Dt_kappa_M, true, -1.0, 1.0);

  if (kbf != nullptr) kfb->add(*Dt_kappa_M, false, -1.0, 1.0);

  if (kbb != nullptr) kbb->add(*Dt_kappa_D, false, 1.0, 1.0);


  if (fluid_force != nullptr)
  {
    if (fluid_vel == nullptr || beam_vel == nullptr)
      FOUR_C_THROW("Force contributions can only be calculated with a given velocity pointer!");

    // Temporary vectors for matrix-vector multiplication and vector-vector additions.
    Core::LinAlg::Vector<double> fluid_temp(*fluid_dof_rowmap_);

    // Set the values in the global force vector to 0.
    fluid_force->put_scalar(0.);

    linalg_error = Dt_kappa_M->multiply(true, *beam_vel, fluid_temp);
    if (linalg_error != 0) FOUR_C_THROW("Error in Multiply!");
    fluid_force->update(-1.0, fluid_temp, 1.0);
  }

  Core::LinAlg::Vector<double> beam_temp(*beam_dof_rowmap_);
  beam_force.put_scalar(0.);
  // Get the force acting on the beam.
  linalg_error = Dt_kappa_D->multiply(false, *beam_vel, beam_temp);
  if (linalg_error != 0) FOUR_C_THROW("Error in Multiply!");
  beam_force.update(1.0, beam_temp, 1.0);
  linalg_error = Dt_kappa_M->multiply(false, *fluid_vel, beam_temp);
  if (linalg_error != 0) FOUR_C_THROW("Error in Multiply!");
  beam_force.update(-1.0, beam_temp, 1.0);
}


/**
 *
 */
std::shared_ptr<Core::LinAlg::Vector<double>>
BeamInteraction::BeamToFluidMortarManager::get_global_lambda(
    const Core::LinAlg::Vector<double>& vel) const
{
  check_setup();
  check_global_maps();

  // Get the velocity of the beams and the fluid.
  Core::LinAlg::Vector<double> beam_vel(*beam_dof_rowmap_);
  Core::LinAlg::Vector<double> fluid_vel(*fluid_dof_rowmap_);
  Core::LinAlg::export_to(vel, beam_vel);
  Core::LinAlg::export_to(vel, fluid_vel);

  // Set up lambda vector;
  std::shared_ptr<Core::LinAlg::Vector<double>> lambda =
      std::make_shared<Core::LinAlg::Vector<double>>(*lambda_dof_rowmap_);

  // Create a temporary vector and calculate lambda.
  Core::LinAlg::Vector<double> lambda_temp_1(*lambda_dof_rowmap_);
  Core::LinAlg::Vector<double> lambda_temp_2(*lambda_dof_rowmap_);
  int linalg_error = global_d_->multiply(false, beam_vel, lambda_temp_2);
  if (linalg_error != 0) FOUR_C_THROW("Error in Multiply!");
  linalg_error = lambda_temp_1.update(1.0, lambda_temp_2, 0.0);
  if (linalg_error != 0) FOUR_C_THROW("Error in Update!");
  linalg_error = global_m_->multiply(false, fluid_vel, lambda_temp_2);
  if (linalg_error != 0) FOUR_C_THROW("Error in Multiply!");
  linalg_error = lambda_temp_1.update(-1.0, lambda_temp_2, 1.0);
  if (linalg_error != 0) FOUR_C_THROW("Error in Multiply!");

  // Scale Lambda with kappa^-1.
  std::shared_ptr<Core::LinAlg::Vector<double>> global_kappa_inv = invert_kappa();
  Core::LinAlg::SparseMatrix kappa_inv_mat(*global_kappa_inv);
  kappa_inv_mat.complete();
  linalg_error = kappa_inv_mat.multiply(false, lambda_temp_1, *lambda);
  if (linalg_error != 0) FOUR_C_THROW("Error in Multiply!");

  return lambda;
}

/**
 *
 */
std::shared_ptr<Core::LinAlg::Vector<double>>
BeamInteraction::BeamToFluidMortarManager::get_global_lambda_col(
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel) const
{
  std::shared_ptr<Core::LinAlg::Vector<double>> lambda_col =
      std::make_shared<Core::LinAlg::Vector<double>>(*lambda_dof_colmap_);
  Core::LinAlg::export_to(*get_global_lambda(*vel), *lambda_col);
  return lambda_col;
}

/**
 *
 */
std::shared_ptr<Core::LinAlg::Vector<double>>
BeamInteraction::BeamToFluidMortarManager::invert_kappa() const
{
  // Create the inverse vector.
  std::shared_ptr<Core::LinAlg::Vector<double>> global_kappa_inv =
      std::make_shared<Core::LinAlg::Vector<double>>(*lambda_dof_rowmap_);

  // Calculate the local inverse of kappa.
  double local_kappa_inv_value = 0.;
  for (int lid = 0; lid < lambda_dof_rowmap_->num_my_elements(); lid++)
  {
    if (global_active_lambda_->get_values()[lid] > 0.1)
      local_kappa_inv_value = 1. / global_kappa_->get_values()[lid];
    else
      local_kappa_inv_value = 0.0;

    global_kappa_inv->replace_local_value(lid, local_kappa_inv_value);
  }

  return global_kappa_inv;
}

FOUR_C_NAMESPACE_CLOSE
