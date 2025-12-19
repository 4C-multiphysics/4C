// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_framework_embeddedmesh_solid_to_solid_coupling_manager.hpp"

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
Constraints::EmbeddedMesh::SolidToSolidCouplingManager::SolidToSolidCouplingManager(
    std::shared_ptr<Core::FE::Discretization>& discret,
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

  // Register default visualization data for the visualization manager
  visualization_manager_->register_visualization_data("background_integration_points");
  visualization_manager_->register_visualization_data("interface_integration_points");
  visualization_manager_->register_visualization_data("cut_element_integration_points");

  // Register default point data
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
}

void Constraints::EmbeddedMesh::SolidToSolidCouplingManager::set_state(
    const Core::LinAlg::Vector<double>& displacement_vector)
{
  for (auto couplig_pair_iter : embedded_mesh_solid_pairs_)
    couplig_pair_iter->set_current_element_position(*discret_, displacement_vector);
}



void Constraints::EmbeddedMesh::SolidToSolidCouplingManager::collect_output_integration_points()
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

bool Constraints::EmbeddedMesh::SolidToSolidCouplingManager::is_cut_node(
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

MPI_Comm Constraints::EmbeddedMesh::SolidToSolidCouplingManager::get_my_comm()
{
  return discret_->get_comm();
}


FOUR_C_NAMESPACE_CLOSE
