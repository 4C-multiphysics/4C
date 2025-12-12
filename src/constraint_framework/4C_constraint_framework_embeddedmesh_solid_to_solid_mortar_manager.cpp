// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_framework_embeddedmesh_solid_to_solid_mortar_manager.hpp"

#include "4C_constraint_framework_embeddedmesh_interaction_pair.hpp"
#include "4C_constraint_framework_embeddedmesh_params.hpp"
#include "4C_constraint_framework_embeddedmesh_solid_to_solid_utils.hpp"
#include "4C_cut_cutwizard.hpp"
#include "4C_io_visualization_manager.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
Constraints::EmbeddedMesh::SolidToSolidMortarManager::SolidToSolidMortarManager(
    std::shared_ptr<Core::FE::Discretization>& discret,
    const Core::LinAlg::Vector<double>& displacement_vector,
    Constraints::EmbeddedMesh::EmbeddedMeshParams& embedded_mesh_coupling_params,
    std::shared_ptr<Core::IO::VisualizationManager> visualization_manager,
    int start_value_lambda_gid)
    : discret_(discret),
      start_value_lambda_gid_(start_value_lambda_gid),
      embedded_mesh_coupling_params_(embedded_mesh_coupling_params),
      visualization_manager_(visualization_manager)
{
  // Initialize cutwizard instance and perform the cut
  std::shared_ptr<Cut::CutWizard> cutwizard = std::make_shared<Cut::CutWizard>(discret_);
  Constraints::EmbeddedMesh::prepare_and_perform_cut(
      cutwizard, discret_, embedded_mesh_coupling_params_);

  // Obtain the information of the background and its related interface elements
  std::vector<BackgroundInterfaceInfo> info_background_interface_elements =
      get_information_background_and_interface_elements(
          *cutwizard, *discret_, ids_cut_elements_col_, cut_elements_col_vector_);

  // Get the coupling pairs and cut elements
  get_coupling_pairs_and_background_elements(info_background_interface_elements, cutwizard,
      embedded_mesh_coupling_params_, *discret_, embedded_mesh_solid_pairs_);

  // Change integration rule of elements if they are cut
  Constraints::EmbeddedMesh::change_gauss_rule_of_cut_elements(
      cut_elements_col_vector_, *cutwizard);

  // Get the number of Lagrange multiplier DOF on a solid node and on a solid element
  unsigned int n_lambda_node_temp = 0;
  mortar_shape_functions_to_number_of_lagrange_values(
      embedded_mesh_coupling_params_.mortar_shape_function_, n_lambda_node_temp);

  n_lambda_node_ = n_lambda_node_temp;

  visualization_manager_->register_visualization_data("background_integration_points");
  visualization_manager_->register_visualization_data("interface_integration_points");
  visualization_manager_->register_visualization_data("cut_element_integration_points");
  visualization_manager_->register_visualization_data("lagrange_multipliers");

  auto& background_integration_points_visualization_data =
      visualization_manager_->get_visualization_data("background_integration_points");
  background_integration_points_visualization_data.register_point_data<double>("weights", 1);
  background_integration_points_visualization_data.register_point_data<int>("interface_id", 1);
  background_integration_points_visualization_data.register_point_data<int>("background_id", 1);
  background_integration_points_visualization_data.register_point_data<int>("boundary_cell_id", 1);

  auto& interface_integration_points_visualization_data =
      visualization_manager_->get_visualization_data("interface_integration_points");
  interface_integration_points_visualization_data.register_point_data<double>("weights", 1);
  interface_integration_points_visualization_data.register_point_data<int>("interface_id", 1);
  interface_integration_points_visualization_data.register_point_data<int>("background_id", 1);
  interface_integration_points_visualization_data.register_point_data<int>("boundary_cell_id", 1);

  auto& cut_element_integration_points_visualization_data =
      visualization_manager_->get_visualization_data("cut_element_integration_points");
  cut_element_integration_points_visualization_data.register_point_data<double>("weights", 1);
  cut_element_integration_points_visualization_data.register_point_data<int>(
      "integration_cell_id", 1);

  auto& lagrange_multipliers_visualization_data =
      visualization_manager_->get_visualization_data("lagrange_multipliers");
  lagrange_multipliers_visualization_data.register_point_data<double>("lambda", 3);

  // Setup the solid to solid mortar manager
  Core::LinAlg::Vector<double> disp_col_vec(*discret_->dof_col_map());
  Core::LinAlg::export_to(displacement_vector, disp_col_vec);
  setup(disp_col_vec);
}

void Constraints::EmbeddedMesh::SolidToSolidMortarManager::set_state(
    const Core::LinAlg::Vector<double>& displacement_vector)
{
  for (auto couplig_pair_iter : embedded_mesh_solid_pairs_)
    couplig_pair_iter->set_current_element_position(*discret_, displacement_vector);
}

void Constraints::EmbeddedMesh::SolidToSolidMortarManager::setup(
    const Core::LinAlg::Vector<double>& displacement_vector)
{
  // Get the global ids of all mesh nodes on this rank
  std::vector<int> my_nodes_gid;
  for (int i_node = 0; i_node < discret_->node_row_map()->num_my_elements(); i_node++)
  {
    Core::Nodes::Node const& node = *(discret_->l_row_node(i_node));
    if (Constraints::EmbeddedMesh::is_interface_node(*discret_, node))
      my_nodes_gid.push_back(node.id());
  }

  // Calculate the local number of interface nodes
  const unsigned int n_nodes = my_nodes_gid.size();
  const unsigned int n_lambda_dof = n_nodes * n_lambda_node_;

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

  // Fill in all GIDs of the lambda DOFs on this processor
  std::vector<int> my_lambda_gid;
  my_lambda_gid.reserve(n_nodes * n_lambda_node_);
  for (unsigned int i_node = 0; i_node < n_nodes; i_node++)
  {
    for (unsigned int i_dof = 0; i_dof < n_lambda_node_; i_dof++)
      my_lambda_gid.push_back(my_lambda_gid_start_value + i_dof + i_node * n_lambda_node_);
  }

  // Rowmap for the additional GIDs used by the mortar contact discretization.
  lambda_dof_rowmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, my_lambda_gid.size(), my_lambda_gid.data(), 0, discret_->get_comm());

  // We need to be able to get the global ids for a Lagrange multiplier DOF from the global id
  // of a node. To do so, we 'abuse' the Core::LinAlg::MultiVector<double> as map between the
  // global node ids and the global Lagrange multiplier DOF ids.
  Core::LinAlg::Map node_gid_rowmap(-1, n_nodes, my_nodes_gid.data(), 0, discret_->get_comm());

  // Map from global node ids to global lagrange multiplier ids. Only create the
  // multivector if it has one or more columns.
  node_gid_to_lambda_gid_ = nullptr;
  if (n_lambda_node_ > 0)
    node_gid_to_lambda_gid_ =
        std::make_shared<Core::LinAlg::MultiVector<double>>(node_gid_rowmap, n_lambda_node_, true);

  // Fill in the entries in the node global id to Lagrange multiplier global id vector.
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

  // Create the maps for boundary layer and background DOFs..
  set_global_maps();

  // Create the global coupling matrices.
  global_constraint_ = std::make_shared<Core::LinAlg::FEVector<double>>(*lambda_dof_rowmap_);
  global_g_bl_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *lambda_dof_rowmap_, 30, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  global_g_bg_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *lambda_dof_rowmap_, 100, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  global_fbl_l_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *boundary_layer_interface_dof_rowmap_, 30, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  global_fbg_l_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *background_dof_rowmap_, 100, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  global_kappa_ = std::make_shared<Core::LinAlg::FEVector<double>>(*lambda_dof_rowmap_);
  global_active_lambda_ = std::make_shared<Core::LinAlg::FEVector<double>>(*lambda_dof_rowmap_);

  // Set flag for successful setup.
  is_setup_ = true;
  is_local_maps_build_ = false;

  // Set the local maps
  set_local_maps(displacement_vector);
}

void Constraints::EmbeddedMesh::SolidToSolidMortarManager::set_global_maps()
{
  // Get the dofs of the background and interface elements
  std::vector<int> boundary_layer_interface_dofs;
  std::vector<int> background_dofs;
  for (int i_node = 0; i_node < discret_->node_row_map()->num_my_elements(); i_node++)
  {
    const Core::Nodes::Node* node = discret_->l_row_node(i_node);
    if (is_cut_node(*node))
      discret_->dof(node, background_dofs);
    else if (Constraints::EmbeddedMesh::is_interface_node(*discret_, *node))
      discret_->dof(node, boundary_layer_interface_dofs);
  }

  // Create the beam and solid maps.
  boundary_layer_interface_dof_rowmap_ =
      std::make_shared<Core::LinAlg::Map>(-1, boundary_layer_interface_dofs.size(),
          boundary_layer_interface_dofs.data(), 0, discret_->get_comm());
  background_dof_rowmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, background_dofs.size(), background_dofs.data(), 0, discret_->get_comm());

  // Reset the local maps.
  node_gid_to_lambda_gid_map_.clear();

  // Set flags for global maps.
  is_global_maps_build_ = true;
  is_local_maps_build_ = false;
}

void Constraints::EmbeddedMesh::SolidToSolidMortarManager::set_local_maps(
    const Core::LinAlg::Vector<double>& displacement_vector)
{
  check_setup();
  check_global_maps();

  // Set the displacement state of the coupling pairs
  set_state(displacement_vector);

  // At this point the global multi vectors are filled up completely. To get the map for global
  // node ids to the global lambda ids we need to be able to extract more than the local
  // values on this processor. Therefore, we need a new map that contains all rows we want to
  // access in the global multi vector.
  std::vector<int> node_gid_needed;

  // Loop over the pairs and get the global node and element indices needed on this rank.
  for (unsigned int i_pair = 0; i_pair < embedded_mesh_solid_pairs_.size(); i_pair++)
  {
    const std::shared_ptr<EmbeddedMesh::SolidInteractionPair>& pair =
        embedded_mesh_solid_pairs_[i_pair];

    // The second (background) element should always be on the same processor as the pair.
    if (pair->element_2().owner() != Core::Communication::my_mpi_rank(discret_->get_comm()))
      FOUR_C_THROW(
          "The current implementation needs the background element of "
          "a interface coupling pair to be on on the same "
          "processor as the pair! The background element id"
          " is {} trying to be processed in proc {}",
          pair->element_2().id(), Core::Communication::my_mpi_rank(discret_->get_comm()));

    // Get the global id of the nodes that the pairs on this rank need.
    if (n_lambda_node_ > 0)
      for (int i_node = 0; i_node < pair->element_1().num_node(); i_node++)
        node_gid_needed.push_back(pair->element_1().nodes()[i_node]->id());
  }

  // Make the entries in the vectors unique.
  std::vector<int>::iterator it;
  std::sort(node_gid_needed.begin(), node_gid_needed.end());
  it = std::unique(node_gid_needed.begin(), node_gid_needed.end());
  node_gid_needed.resize(std::distance(node_gid_needed.begin(), it));

  // Create the maps for the extraction of the values.
  Core::LinAlg::Map node_gid_needed_rowmap(
      -1, node_gid_needed.size(), node_gid_needed.data(), 0, discret_->get_comm());

  // Create the Multivectors that will be filled with all values needed on this rank.
  std::shared_ptr<Core::LinAlg::MultiVector<double>> node_gid_to_lambda_gid_copy = nullptr;
  if (node_gid_to_lambda_gid_ != nullptr)
    node_gid_to_lambda_gid_copy = std::make_shared<Core::LinAlg::MultiVector<double>>(
        node_gid_needed_rowmap, n_lambda_node_, true);

  // Export values from the global multi vector to the ones needed on this rank.
  if (node_gid_to_lambda_gid_ != nullptr)
    Core::LinAlg::export_to(*node_gid_to_lambda_gid_, *node_gid_to_lambda_gid_copy);

  // Fill in the local maps.
  std::vector<int> lambda_gid_for_col_map;
  lambda_gid_for_col_map.clear();
  node_gid_to_lambda_gid_map_.clear();
  if (node_gid_to_lambda_gid_ != nullptr)
  {
    std::vector<int> temp_node(n_lambda_node_);
    for (int i_node = 0; i_node < node_gid_needed_rowmap.num_my_elements(); i_node++)
    {
      for (unsigned int i_temp = 0; i_temp < n_lambda_node_; i_temp++)
        temp_node[i_temp] =
            (int)(node_gid_to_lambda_gid_copy->get_vector(i_temp).local_values_as_span()[i_node]);
      node_gid_to_lambda_gid_map_[node_gid_needed_rowmap.gid(i_node)] = temp_node;
      lambda_gid_for_col_map.insert(
          std::end(lambda_gid_for_col_map), std::begin(temp_node), std::end(temp_node));
    }
  }

  // Create the global lambda col map.
  lambda_dof_colmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, lambda_gid_for_col_map.size(), lambda_gid_for_col_map.data(), 0, discret_->get_comm());

  // Set flags for local maps.
  is_local_maps_build_ = true;
}

void Constraints::EmbeddedMesh::SolidToSolidMortarManager::location_vector(
    const Constraints::EmbeddedMesh::SolidInteractionPair* interaction_pair,
    std::vector<int>& lambda_row) const
{
  check_setup();
  check_local_maps();

  // Clear the output vectors.
  lambda_row.clear();

  // Get the global DOFs ids of the nodal Lagrange multipliers.
  if (n_lambda_node_ > 0)
  {
    for (int i_node = 0; i_node < interaction_pair->element_1().num_node(); i_node++)
    {
      const Core::Nodes::Node& node = *(interaction_pair->element_1().nodes()[i_node]);
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

/**
 *
 */
void Constraints::EmbeddedMesh::SolidToSolidMortarManager::evaluate_global_coupling_contributions(
    const Core::LinAlg::Vector<double>& displacement_vector)
{
  check_setup();
  check_global_maps();

  Core::LinAlg::Vector<double> disp_col_vec(*discret_->dof_col_map());
  Core::LinAlg::export_to(displacement_vector, disp_col_vec);
  set_state(disp_col_vec);

  // Reset the global data structures.
  global_constraint_->put_scalar(0.);
  global_g_bl_->put_scalar(0.);
  global_g_bg_->put_scalar(0.);
  global_fbl_l_->put_scalar(0.);
  global_fbg_l_->put_scalar(0.);
  global_kappa_->put_scalar(0.);
  global_active_lambda_->put_scalar(0.);

  for (auto& elepairptr : embedded_mesh_solid_pairs_)
  {
    elepairptr->evaluate_and_assemble_mortar_contributions(*discret_, this, *global_g_bl_,
        *global_g_bg_, *global_fbl_l_, *global_fbg_l_, *global_constraint_, *global_kappa_,
        *global_active_lambda_);
  }

  // Complete the global mortar matrices.
  global_g_bl_->complete(
      *boundary_layer_interface_dof_rowmap_, *lambda_dof_rowmap_);  // is the complete wrong?
  global_g_bg_->complete(*background_dof_rowmap_, *lambda_dof_rowmap_);
  global_fbl_l_->complete(*lambda_dof_rowmap_, *boundary_layer_interface_dof_rowmap_);
  global_fbg_l_->complete(*lambda_dof_rowmap_, *background_dof_rowmap_);

  // Complete the global scaling vector.
  global_kappa_->complete();
  global_active_lambda_->complete();
  global_constraint_->complete();
}

void Constraints::EmbeddedMesh::SolidToSolidMortarManager::
    add_global_force_stiffness_penalty_contributions(
        Solid::TimeInt::BaseDataGlobalState& data_state,
        std::shared_ptr<Core::LinAlg::SparseMatrix> stiff,
        std::shared_ptr<Core::LinAlg::Vector<double>> force) const
{
  check_setup();
  check_global_maps();

  if (stiff != nullptr)
  {
    // Scale the linearizations of the constraint equations.
    std::shared_ptr<Core::LinAlg::Vector<double>> global_penalty_kappa_inv = penalty_invert_kappa();
    Core::LinAlg::SparseMatrix penalty_kappa_inv_mat(*global_penalty_kappa_inv);
    penalty_kappa_inv_mat.complete();

    std::shared_ptr<Core::LinAlg::SparseMatrix> global_G_BL_scaled = Core::LinAlg::matrix_multiply(
        penalty_kappa_inv_mat, false, *global_g_bl_, false, false, false, true);
    std::shared_ptr<Core::LinAlg::SparseMatrix> global_G_BG_scaled = Core::LinAlg::matrix_multiply(
        penalty_kappa_inv_mat, false, *global_g_bg_, false, false, false, true);

    // Calculate the needed submatrices.
    std::shared_ptr<Core::LinAlg::SparseMatrix> FBL_L_times_G_BL = Core::LinAlg::matrix_multiply(
        *global_fbl_l_, false, *global_G_BL_scaled, false, false, false, true);
    std::shared_ptr<Core::LinAlg::SparseMatrix> FBL_L_times_G_BG = Core::LinAlg::matrix_multiply(
        *global_fbl_l_, false, *global_G_BG_scaled, false, false, false, true);
    std::shared_ptr<Core::LinAlg::SparseMatrix> FBG_L_times_G_BL = Core::LinAlg::matrix_multiply(
        *global_fbg_l_, false, *global_G_BL_scaled, false, false, false, true);
    std::shared_ptr<Core::LinAlg::SparseMatrix> FBG_L_times_G_BG = Core::LinAlg::matrix_multiply(
        *global_fbg_l_, false, *global_G_BG_scaled, false, false, false, true);

    // Add contributions to the global stiffness matrix.
    Core::LinAlg::matrix_add(*FBL_L_times_G_BL, false, 1.0, *stiff, 1.0);
    Core::LinAlg::matrix_add(*FBL_L_times_G_BG, false, 1.0, *stiff, 1.0);
    Core::LinAlg::matrix_add(*FBG_L_times_G_BL, false, 1.0, *stiff, 1.0);
    Core::LinAlg::matrix_add(*FBG_L_times_G_BG, false, 1.0, *stiff, 1.0);
  }

  if (force != nullptr)
  {
    // Factor for right hand side (forces). 1 corresponds to the meshtying forces being added to
    // the right hand side, -1 to the left hand side.
    const double rhs_factor = 1.0;

    // Get the penalty Lagrange multiplier vector.
    std::shared_ptr<Core::LinAlg::Vector<double>> lambda = get_global_lambda();

    // Multiply the lambda vector with FBL_L and FBG_L to get the forces on the boundary layer and
    // background, respectively.
    Core::LinAlg::Vector<double> boundary_layer_interface_force(
        *boundary_layer_interface_dof_rowmap_);
    Core::LinAlg::Vector<double> background_force(*background_dof_rowmap_);
    boundary_layer_interface_force.put_scalar(0.);
    background_force.put_scalar(0.);
    global_fbl_l_->multiply(false, *lambda, boundary_layer_interface_force);
    global_fbg_l_->multiply(false, *lambda, background_force);
    Core::LinAlg::Vector<double> global_temp(*discret_->dof_row_map());
    Core::LinAlg::export_to(boundary_layer_interface_force, global_temp);
    Core::LinAlg::export_to(background_force, global_temp);

    // Add force contributions to global vector.
    force->update(rhs_factor, global_temp, 1.0);
    ;
  }
}

/**
 *
 */
std::shared_ptr<Core::LinAlg::Vector<double>>
Constraints::EmbeddedMesh::SolidToSolidMortarManager::get_global_lambda() const
{
  check_setup();
  check_global_maps();

  // Get the inverted kappa matrix.
  std::shared_ptr<Core::LinAlg::Vector<double>> penalty_global_kappa_inv = penalty_invert_kappa();
  Core::LinAlg::SparseMatrix penalty_kappa_inv_mat(*penalty_global_kappa_inv);
  penalty_kappa_inv_mat.complete();

  // Multiply the inverted kappa matrix with the constraint equations and scale them with the
  // penalty parameter.
  std::shared_ptr<Core::LinAlg::Vector<double>> lambda =
      std::make_shared<Core::LinAlg::Vector<double>>(*lambda_dof_rowmap_);
  penalty_kappa_inv_mat.multiply(false, Core::LinAlg::Vector<double>(*global_constraint_), *lambda);

  return lambda;
}

/**
 *
 */
std::shared_ptr<Core::LinAlg::Vector<double>>
Constraints::EmbeddedMesh::SolidToSolidMortarManager::get_global_lambda_col() const
{
  std::shared_ptr<Core::LinAlg::Vector<double>> lambda_col =
      std::make_shared<Core::LinAlg::Vector<double>>(*lambda_dof_colmap_);
  Core::LinAlg::export_to(*get_global_lambda(), *lambda_col);
  return lambda_col;
}

std::shared_ptr<Core::LinAlg::Vector<double>>
Constraints::EmbeddedMesh::SolidToSolidMortarManager::penalty_invert_kappa() const
{
  // Create the inverse vector.
  std::shared_ptr<Core::LinAlg::Vector<double>> global_kappa_inv =
      std::make_shared<Core::LinAlg::Vector<double>>(*lambda_dof_rowmap_);

  // Get the penalty parameters.
  const double penalty_params = embedded_mesh_coupling_params_.constraint_penalty_parameter_;

  // Calculate the local inverse of kappa.
  double penalty = 0.0;
  double local_kappa_inv_value = 0.;
  for (int lid = 0; lid < lambda_dof_rowmap_->num_my_elements(); lid++)
  {
    if (global_active_lambda_->get_values()[lid] > 0.1)
    {
      const int gid = lambda_dof_rowmap_->gid(lid);
      if (lambda_dof_rowmap_->lid(gid) != -1)
        penalty = penalty_params;
      else
        FOUR_C_THROW("Could not find the GID {} in translation map", gid);

      local_kappa_inv_value = penalty / global_kappa_->get_values()[lid];
    }
    else
      // This LID is inactive.
      local_kappa_inv_value = 0.0;

    global_kappa_inv->replace_local_value(lid, local_kappa_inv_value);
  }

  return global_kappa_inv;
}

void Constraints::EmbeddedMesh::SolidToSolidMortarManager::write_output(
    double time, int timestep_number)
{
  // Clear the data of visualization manager
  visualization_manager_->clear_data();

  // Collect output of the physical positions of gauss points
  // where surface integrations will take place
  collect_output_integration_points();

  // Collect vtk output of lagrange multipliers
  collect_output_lagrange_multipliers();

  // Write output
  visualization_manager_->write_to_disk(time, timestep_number);
}

void Constraints::EmbeddedMesh::SolidToSolidMortarManager::collect_output_lagrange_multipliers()
{
  auto& lagrange_multipliers_visualization_data =
      visualization_manager_->get_visualization_data("lagrange_multipliers");

  std::shared_ptr<Core::LinAlg::Vector<double>> lambda = get_global_lambda_col();

  std::shared_ptr<std::unordered_set<int>> interface_tracker =
      std::make_shared<std::unordered_set<int>>();

  // Loop over pairs
  for (auto& elepairptr : embedded_mesh_solid_pairs_)
  {
    elepairptr->get_pair_visualization(
        lagrange_multipliers_visualization_data, lambda, this, interface_tracker);
  }
}

void Constraints::EmbeddedMesh::SolidToSolidMortarManager::collect_output_integration_points()
{
  auto& background_integration_points_visualization_data =
      visualization_manager_->get_visualization_data("background_integration_points");

  auto& interface_integration_points_visualization_data =
      visualization_manager_->get_visualization_data("interface_integration_points");

  auto& cut_element_integration_points_visualization_data =
      visualization_manager_->get_visualization_data("cut_element_integration_points");

  // Loop over pairs
  for (auto& elepairptr : embedded_mesh_solid_pairs_)
  {
    unsigned int n_segments = elepairptr->get_num_segments();
    for (size_t iter_segments = 0; iter_segments < n_segments; iter_segments++)
    {
      elepairptr->get_projected_gauss_rule_on_interface(
          background_integration_points_visualization_data,
          interface_integration_points_visualization_data);
    }

    elepairptr->get_projected_gauss_rule_in_cut_element(
        cut_element_integration_points_visualization_data);
  }
}

bool Constraints::EmbeddedMesh::SolidToSolidMortarManager::is_cut_node(
    Core::Nodes::Node const& node)
{
  bool is_cut_node = false;

  // Check if the node belongs to an element that is cut
  for (int num_ele = 0; num_ele < node.num_element(); num_ele++)
  {
    bool is_node_in_cut_ele =
        std::find(cut_elements_col_vector_.begin(), cut_elements_col_vector_.end(),
            node.adjacent_elements()[num_ele].user_element()) != cut_elements_col_vector_.end();
    if (is_node_in_cut_ele) is_cut_node = true;
  }

  return is_cut_node;
}

MPI_Comm Constraints::EmbeddedMesh::SolidToSolidMortarManager::get_my_comm()
{
  return discret_->get_comm();
}

double Constraints::EmbeddedMesh::SolidToSolidMortarManager::get_energy() const
{
  // Since this value is also computed for the reference configuration, where the global mortar
  // matrices are not build yet we return 0 in this case.
  if (not global_g_bl_->filled() or not global_g_bg_->filled() or not global_fbl_l_->filled() or
      not global_fbg_l_->filled())
    return 0.0;

  // Calculate the penalty potential.
  std::shared_ptr<Core::LinAlg::Vector<double>> lambda = get_global_lambda();
  double dot_product = 0.0;
  global_constraint_->dot(*lambda, &dot_product);
  return 0.5 * dot_product;
}

FOUR_C_NAMESPACE_CLOSE
