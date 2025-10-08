// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_framework_embeddedmesh_solid_to_solid_nitsche_manager.hpp"

#include "4C_constraint_framework_embeddedmesh_interaction_pair.hpp"
#include "4C_constraint_framework_embeddedmesh_params.hpp"
#include "4C_constraint_framework_embeddedmesh_solid_to_solid_utils.hpp"
#include "4C_cut_cutwizard.hpp"
#include "4C_io_visualization_manager.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
Constraints::EmbeddedMesh::SolidToSolidNitscheManager::SolidToSolidNitscheManager(
    std::shared_ptr<Core::FE::Discretization>& discret,
    const Core::LinAlg::Vector<double>& displacement_vector,
    Constraints::EmbeddedMesh::EmbeddedMeshParams& embedded_mesh_coupling_params,
    std::shared_ptr<Core::IO::VisualizationManager> visualization_manager)
    : discret_(discret),
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

  visualization_manager_->register_visualization_data("background_integration_points");
  visualization_manager_->register_visualization_data("interface_integration_points");
  visualization_manager_->register_visualization_data("cut_element_integration_points");

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

  // Setup the solid to solid Nitsche manager
  Core::LinAlg::Vector<double> disp_col_vec(*discret_->dof_col_map());
  Core::LinAlg::export_to(displacement_vector, disp_col_vec);
  setup(disp_col_vec);
}

void Constraints::EmbeddedMesh::SolidToSolidNitscheManager::setup(
    const Core::LinAlg::Vector<double>& displacement_vector)
{
  // Get the global ids of all mesh nodes on this rank
  std::vector<int> my_interface_nodes_gid;
  for (int i_node = 0; i_node < discret_->node_row_map()->num_my_elements(); i_node++)
  {
    Core::Nodes::Node const& node = *(discret_->l_row_node(i_node));
    if (Constraints::EmbeddedMesh::is_interface_node(*discret_, node))
      my_interface_nodes_gid.push_back(node.id());
  }

  // Create the maps for interface and background DOFs.
  set_global_maps();

  // Create the global coupling matrices.
  global_penalty_interface_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *interface_dof_rowmap_, 100, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  global_penalty_background_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *background_dof_rowmap_, 100, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  global_penalty_interface_background_ =
      std::make_shared<Core::LinAlg::SparseMatrix>(*interface_and_background_dof_rowmap_, 100, true,
          true, Core::LinAlg::SparseMatrix::FE_MATRIX);

  // Set flag for successful setup.
  is_setup_ = true;

  // Set current state
  set_state(displacement_vector);
}

void Constraints::EmbeddedMesh::SolidToSolidNitscheManager::set_state(
    const Core::LinAlg::Vector<double>& displacement_vector)
{
  for (auto couplig_pair_iter : embedded_mesh_solid_pairs_)
    couplig_pair_iter->set_current_element_position(*discret_, displacement_vector);
}

void Constraints::EmbeddedMesh::SolidToSolidNitscheManager::set_global_maps()
{
  // Get the dofs of the background and interface elements
  std::vector<int> interface_dofs;
  std::vector<int> background_dofs;
  std::vector<int> interface_and_background_dofs;
  for (int i_node = 0; i_node < discret_->node_row_map()->num_my_elements(); i_node++)
  {
    const Core::Nodes::Node* node = discret_->l_row_node(i_node);
    if (is_cut_node(*node))
      discret_->dof(node, background_dofs);
    else if (Constraints::EmbeddedMesh::is_interface_node(*discret_, *node))
      discret_->dof(node, interface_dofs);
  }

  interface_and_background_dofs.reserve(interface_dofs.size() + background_dofs.size());
  interface_and_background_dofs.insert(
      interface_and_background_dofs.end(), interface_dofs.begin(), interface_dofs.end());
  interface_and_background_dofs.insert(
      interface_and_background_dofs.end(), background_dofs.begin(), background_dofs.end());

  std::sort(interface_and_background_dofs.begin(), interface_and_background_dofs.end());

  // Create the interface and solid maps.
  interface_dof_rowmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, interface_dofs.size(), interface_dofs.data(), 0, discret_->get_comm());
  background_dof_rowmap_ = std::make_shared<Core::LinAlg::Map>(
      -1, background_dofs.size(), background_dofs.data(), 0, discret_->get_comm());
  interface_and_background_dof_rowmap_ =
      std::make_shared<Core::LinAlg::Map>(-1, interface_and_background_dofs.size(),
          interface_and_background_dofs.data(), 0, discret_->get_comm());

  // Set flags for global maps.
  is_global_maps_build_ = true;
}


/**
 * here I have to evaluate the contributions
 */
void Constraints::EmbeddedMesh::SolidToSolidNitscheManager::evaluate_global_coupling_contributions(
    const Core::LinAlg::Vector<double>& displacement_vector)
{
  check_setup();
  check_global_maps();

  Core::LinAlg::Vector<double> disp_col_vec(*discret_->dof_col_map());
  Core::LinAlg::export_to(displacement_vector, disp_col_vec);
  set_state(disp_col_vec);

  // Reset the global data structures.
  global_penalty_interface_->put_scalar(0.);
  global_penalty_background_->put_scalar(0.);
  global_penalty_interface_background_->put_scalar(0.);

  for (auto& elepairptr : embedded_mesh_solid_pairs_)
  {
    elepairptr->evaluate_and_assemble_nitsche_contributions(*discret_, this,
        *global_penalty_interface_, *global_penalty_background_,
        *global_penalty_interface_background_);
  }

  // scale_contributions_penalty_stiffness_matrices();

  // Complete the global mortar matrices.
  global_penalty_interface_->complete(*interface_dof_rowmap_, *interface_dof_rowmap_);
  global_penalty_background_->complete(*background_dof_rowmap_, *background_dof_rowmap_);
  global_penalty_interface_background_->complete(
      *interface_and_background_dof_rowmap_, *interface_and_background_dof_rowmap_);

  scale_contributions_penalty_stiffness_matrices();
}

void Constraints::EmbeddedMesh::SolidToSolidNitscheManager::
    scale_contributions_penalty_stiffness_matrices() const
{
  // Get the penalty parameters.
  const double penalty_params = embedded_mesh_coupling_params_.constraint_penalty_parameter_;

  global_penalty_interface_->scale(penalty_params);
  global_penalty_background_->scale(penalty_params);
  global_penalty_interface_background_->scale(-penalty_params);
}

/**
 * here is the assemble of the contributions
 */
void Constraints::EmbeddedMesh::SolidToSolidNitscheManager::
    add_global_force_stiffness_contributions(Solid::TimeInt::BaseDataGlobalState& data_state,
        std::shared_ptr<Core::LinAlg::SparseMatrix> stiff,
        std::shared_ptr<Core::LinAlg::Vector<double>> force) const
{
  check_setup();
  check_global_maps();

  int linalg_error = 0;

  if (stiff != nullptr)
  {
    // stiff->print(std::cout);
    // Add contributions to the global stiffness matrix.
    // global_penalty_interface_->print(std::cout);
    // global_penalty_background_->print(std::cout);
    // global_penalty_interface_background_->print(std::cout);

    stiff->add(*global_penalty_interface_, false, 1.0, 1.0);
    stiff->add(*global_penalty_background_, false, 1.0, 1.0);
    stiff->add(*global_penalty_interface_background_, false, 1.0, 1.0);
    stiff->add(*global_penalty_interface_background_, true, 1.0, 1.0);
    // stiff->print(std::cout);
  }

  if (force != nullptr)
  {
    // Factor for right hand side (forces). 1 corresponds to forces being added to
    // the right hand side, -1 to the left hand side.
    const double rhs_factor = 1.0;

    // Add penalty contributions of Nitsche method
    auto displacement = data_state.get_dis_np().get();

    // Initialize vectors to store penalty contributions to interface and background
    Core::LinAlg::Vector<double> temp_force_penalty_interface(*interface_dof_rowmap_);

    Core::LinAlg::Vector<double> temp_force_penalty_background(*background_dof_rowmap_);

    Core::LinAlg::Vector<double> temp_force_penalty_interface_contribution_inter_background(
        *interface_and_background_dof_rowmap_);
    Core::LinAlg::Vector<double> temp_force_penalty_background_contribution_inter_background(
        *interface_and_background_dof_rowmap_);

    temp_force_penalty_interface.put_scalar(0.);
    temp_force_penalty_background.put_scalar(0.);
    temp_force_penalty_interface_contribution_inter_background.put_scalar(0.);
    temp_force_penalty_background_contribution_inter_background.put_scalar(0.);

    // Multiply global penalty contributions with the displacement to obtain the penalty forces on
    // the interface and background
    linalg_error =
        global_penalty_interface_->multiply(false, *displacement, temp_force_penalty_interface);
    if (linalg_error != 0) FOUR_C_THROW("Error in Multiply!");
    // temp_force_penalty_interface.print(std::cout);

    linalg_error =
        global_penalty_background_->multiply(false, *displacement, temp_force_penalty_background);
    if (linalg_error != 0) FOUR_C_THROW("Error in Multiply!");
    // temp_force_penalty_background.print(std::cout);

    linalg_error = global_penalty_interface_background_->multiply(
        false, *displacement, temp_force_penalty_background_contribution_inter_background);
    if (linalg_error != 0) FOUR_C_THROW("Error in Multiply!");
    // temp_force_penalty_background_contribution_inter_background.print(std::cout);
    linalg_error = global_penalty_interface_background_->multiply(
        true, *displacement, temp_force_penalty_interface_contribution_inter_background);
    if (linalg_error != 0) FOUR_C_THROW("Error in Multiply!");
    // temp_force_penalty_interface_contribution_inter_background.print(std::cout);


    // Collect force contributions in a global temp
    Core::LinAlg::Vector<double> global_temp_1(*discret_->dof_row_map());
    Core::LinAlg::Vector<double> global_temp_2(*discret_->dof_row_map());
    Core::LinAlg::Vector<double> global_temp_3(*discret_->dof_row_map());
    Core::LinAlg::Vector<double> global_temp_4(*discret_->dof_row_map());

    if (linalg_error != 0) FOUR_C_THROW("Error in Update");
    Core::LinAlg::export_to(temp_force_penalty_interface, global_temp_1);
    Core::LinAlg::export_to(temp_force_penalty_background, global_temp_2);
    Core::LinAlg::export_to(
        temp_force_penalty_background_contribution_inter_background, global_temp_3);
    Core::LinAlg::export_to(
        temp_force_penalty_interface_contribution_inter_background, global_temp_4);

    // force->print(std::cout);
    // Add force contributions to global vector.
    linalg_error = force->update(rhs_factor, global_temp_1, 1.0);
    if (linalg_error != 0) FOUR_C_THROW("Error in Update");
    linalg_error = force->update(rhs_factor, global_temp_2, 1.0);
    if (linalg_error != 0) FOUR_C_THROW("Error in Update");
    linalg_error = force->update(rhs_factor, global_temp_3, 1.0);
    if (linalg_error != 0) FOUR_C_THROW("Error in Update");
    linalg_error = force->update(rhs_factor, global_temp_4, 1.0);
    if (linalg_error != 0) FOUR_C_THROW("Error in Update");
    // force->print(std::cout);
  }
}


bool Constraints::EmbeddedMesh::SolidToSolidNitscheManager::is_cut_node(
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

MPI_Comm Constraints::EmbeddedMesh::SolidToSolidNitscheManager::get_my_comm()
{
  return discret_->get_comm();
}


FOUR_C_NAMESPACE_CLOSE
