// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_linebased.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mat_cnst_1d_art.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_pair.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_segments.hpp"
#include "4C_porofluid_pressure_based_utils.hpp"
#include "4C_structure_new_input.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::
    PorofluidElastScatraArteryCouplingLineBasedAlgorithm(
        std::shared_ptr<Core::FE::Discretization> artery_dis,
        std::shared_ptr<Core::FE::Discretization> homogenized_dis,
        const Teuchos::ParameterList& coupling_params, const std::string& condition_name,
        const PoroPressureBased::PorofluidElastScatraArteryCouplingDeps& artery_coupling_deps)
    : PorofluidElastScatraArteryCouplingNonConformingAlgorithm(
          artery_dis, homogenized_dis, coupling_params, condition_name, artery_coupling_deps),
      max_num_segments_per_artery_element_(
          artery_coupling_deps.porofluid_pressure_based_dynamic_parameters
              ->sublist("artery_coupling")
              .get<int>("maximum_number_of_segments_per_artery_element"))
{
  // user info
  if (my_mpi_rank_ == 0)
  {
    std::cout << "<                                                  >" << '\n';
    print_coupling_method();
    std::cout << "<                                                  >" << '\n';
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>" << '\n';
    std::cout << "\n";
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::setup()
{
  // call base class
  PorofluidElastScatraArteryCouplingNonConformingAlgorithm::setup();

  // pre-evaluate the pairs
  pre_evaluate_coupling_pairs();

  // create the GID to segment boundaries map
  create_gid_to_segment_boundaries(
      segment_context(), coupled_ele_pairs_, gid_to_segment_boundaries_);

  // fill length of artery elements that are not changed by deformation of the underlying 2D/3D mesh
  // (basically protruding artery elements or segments)
  artery_segment_lengths_ = fill_unaffected_artery_length(
      segment_context(), coupled_ele_pairs_, gid_to_segment_boundaries_, gid_to_segment_length_);

  // fill unaffected integrated diameter (basically protruding artery elements or segments)
  if (homogenized_dis_->name() == "porofluid" && has_variable_diameter_)
    fill_unaffected_integrated_diameter();

  // calculate the blood vessel volume fraction (only porofluid needs to do this)
  if (homogenized_dis_->name() == "porofluid" &&
      coupling_params_.get<bool>("output_blood_vessel_volume_fraction"))
    calculate_blood_vessel_volume_fraction();

  // print summary of pairs
  if (homogenized_dis_->name() == "porofluid" &&
      coupling_params_.get<bool>("print_coupling_pairs_summary"))
    output_summary();

  is_setup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::setup_system(
    const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    const std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
    const std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_homogenized,
    const std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_artery,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_homogenized,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_artery,
    const std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_homogenized,
    const std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_artery)
{
  // copy vector
  const auto rhs_art_with_collapsed = std::make_shared<Core::LinAlg::Vector<double>>(*rhs_artery);
  const std::shared_ptr<Core::LinAlg::Map> dbcmap_art_with_collapsed =
      get_additional_dbc_for_collapsed_elements(*dbcmap_artery, *rhs_art_with_collapsed);

  // call base class
  PorofluidElastScatraArteryCouplingNonConformingAlgorithm::setup_system(*sysmat, rhs,
      *sysmat_homogenized, *sysmat_artery, rhs_homogenized, rhs_art_with_collapsed,
      *dbcmap_homogenized, *dbcmap_artery->cond_map(), *dbcmap_art_with_collapsed);
}

std::shared_ptr<Core::LinAlg::Map> PoroPressureBased::
    PorofluidElastScatraArteryCouplingLineBasedAlgorithm::get_additional_dbc_for_collapsed_elements(
        const Core::LinAlg::MapExtractor& dbcmap_artery,
        Core::LinAlg::Vector<double>& rhs_artery_with_collapsed) const
{
  // Zero flux is automatically assumed for nodes which are adjacent to a collapsed element
  // since the respective collapsed element is not evaluated. Nodes which only are adjacent to
  // collapsed elements are not evaluated at all, hence, leading to zero rows in the global
  // stiffness matrix and to singularity of the matrix. Here, we identify these nodes and set a
  // zero Dirichlet boundary condition on them. Note that this procedure is equivalent to deleting
  // elements from the simulation.

  const int artery_element_material = homogenized_dis_->name() == "scatra" ? 1 : 0;
  std::vector<int> dirichlet_dofs;

  const Core::LinAlg::Map* dof_row_map = artery_dis_->dof_row_map();

  for (auto node : artery_dis_->my_row_node_range())
  {
    bool all_elements_collapsed = true;
    for (auto ele : node.adjacent_elements())
    {
      const Core::Elements::Element* current_element = ele.user_element();
      const auto& artery_material = std::dynamic_pointer_cast<const Mat::Cnst1dArt>(
          current_element->material(artery_element_material));
      if (not artery_material->is_collapsed())
      {
        all_elements_collapsed = false;
        break;
      }
    }

    // all elements of this node are collapsed
    if (all_elements_collapsed)
    {
      // 1) insert all dofs of this node into Dirichlet dof vector
      std::vector<int> dofs = artery_dis_->dof(0, node);
      dirichlet_dofs.insert(dirichlet_dofs.end(), dofs.begin(), dofs.end());
      // 2) insert the negative value of all dofs of this node into the rhs, with the employed
      // incremental form as this will force the value to zero
      for (const auto& current_dof : dofs)
        rhs_artery_with_collapsed.replace_global_value(
            current_dof, -phinp_art_->get_values()[dof_row_map->lid(current_dof)]);
    }
  }

  // build map
  int num_dirichlet_values = static_cast<int>(dirichlet_dofs.size());
  const auto dirichlet_map = std::make_shared<Core::LinAlg::Map>(
      -1, num_dirichlet_values, dirichlet_dofs.data(), 0, artery_dis_->get_comm());

  // build vector of maps
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> condition_maps;
  condition_maps.push_back(dirichlet_map);
  condition_maps.push_back(dbcmap_artery.cond_map());

  // combined map
  std::shared_ptr<Core::LinAlg::Map> combined_map =
      Core::LinAlg::MultiMapExtractor::merge_maps(condition_maps);

  return combined_map;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::
    pre_evaluate_coupling_pairs()
{
  // pre-evaluate
  for (const auto& coupled_ele_pair : coupled_ele_pairs_) coupled_ele_pair->pre_evaluate(nullptr);

  // drop inactive / non-owned / duplicate pairs and resolve cross-processor duplicate segments
  filter_coupling_pairs_and_resolve_cross_proc_duplicates(segment_context(), coupled_ele_pairs_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::
    fill_unaffected_integrated_diameter() const
{
  Core::LinAlg::FEVector<double> unaffected_artery_diameters_row(
      *artery_dis_->element_row_map(), true);

  for (int i = 0; i < artery_dis_->element_row_map()->num_my_elements(); ++i)
  {
    const int artery_ele_gid = artery_dis_->element_row_map()->gid(i);
    Core::Elements::Element* current_element = artery_dis_->g_element(artery_ele_gid);

    // TODO: this will not work for higher order artery elements
    const double initial_length = get_max_nodal_distance(current_element, *artery_dis_);

    // first, add all contributions into unaffected_diams_artery_row-vector
    std::shared_ptr<Mat::Cnst1dArt> artery_material =
        std::dynamic_pointer_cast<Mat::Cnst1dArt>(current_element->material());
    if (artery_material == nullptr) FOUR_C_THROW("cast to artery material failed");
    const double length_x_diameter = initial_length * artery_material->diam();
    unaffected_artery_diameters_row.sum_into_global_values(1, &artery_ele_gid, &length_x_diameter);
  }
  // then, subtract the coupling pairs to detect protruding parts
  for (const auto& coupled_ele_pair : coupled_ele_pairs_)
  {
    // get the initial lengths
    double initial_segment_length =
        coupled_ele_pair->apply_mesh_movement(true, homogenized_dis_.get());
    initial_segment_length *= -1.0;

    const int artery_ele_gid = coupled_ele_pair->artery_ele_gid();
    const Core::Elements::Element* current_element = artery_dis_->g_element(artery_ele_gid);

    std::shared_ptr<Mat::Cnst1dArt> artery_material =
        std::dynamic_pointer_cast<Mat::Cnst1dArt>(current_element->material());
    if (artery_material == nullptr) FOUR_C_THROW("cast to artery material failed");
    const double length_x_diameter = initial_segment_length * artery_material->diam();
    unaffected_artery_diameters_row.sum_into_global_values(1, &artery_ele_gid, &length_x_diameter);
  }

  // global assembly and export
  unaffected_artery_diameters_row.complete();
  Core::LinAlg::export_to(Core::LinAlg::Vector<double>(unaffected_artery_diameters_row),
      *unaffected_integrated_artery_diameters_col_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::
    calculate_blood_vessel_volume_fraction()
{
  blood_vessel_volfrac_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*homogenized_dis_->element_row_map(), true);

  double total_blood_vessel_volume = 0.0;
  // evaluate all pairs
  for (const auto& coupled_ele_pair : coupled_ele_pairs_)
  {
    const int artery_ele_gid = coupled_ele_pair->artery_ele_gid();
    const int homogenized_ele_gid = coupled_ele_pair->homogenized_ele_gid();

    Core::Elements::Element* artery_element = artery_dis_->g_element(artery_ele_gid);

    std::shared_ptr<Mat::Cnst1dArt> artery_material =
        std::dynamic_pointer_cast<Mat::Cnst1dArt>(artery_element->material());
    if (artery_material == nullptr) FOUR_C_THROW("cast to artery material failed");

    // TODO: this will not work for higher order artery elements
    const double etaA = coupled_ele_pair->eta_start();
    const double etaB = coupled_ele_pair->eta_end();
    const double length = get_max_nodal_distance(artery_element, *artery_dis_);

    const double volume_homogenized = coupled_ele_pair->calculate_volume_homogenized_element();
    const double volume_artery = (etaB - etaA) / 2.0 * length * artery_material->diam() *
                                 artery_material->diam() * std::numbers::pi / 4.0;

    total_blood_vessel_volume += volume_artery;

    const double volfrac = volume_artery / volume_homogenized;

    // note: this works as the 2D/3D homogenized element of each pair is always owned by this proc
    blood_vessel_volfrac_->sum_into_global_values(1, &volfrac, &homogenized_ele_gid);
  }

  // user output
  double blood_vessel_volume_all_procs = 0.0;
  blood_vessel_volume_all_procs =
      Core::Communication::sum_all(total_blood_vessel_volume, get_comm());
  if (my_mpi_rank_ == 0)
  {
    std::cout << "\n<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>" << '\n';
    std::cout << "<    Calculating blood vessel volume fraction      >" << '\n';
    std::cout << "<    total volume blood:    " << std::setw(5) << blood_vessel_volume_all_procs
              << "                 >" << '\n';
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>" << '\n';
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::
    set_flag_variable_diameter()
{
  PorofluidElastScatraArteryCouplingNonConformingAlgorithm::set_flag_variable_diameter();

  // set up the required vectors
  if (has_variable_diameter_)
  {
    integrated_artery_diameters_row_ =
        std::make_shared<Core::LinAlg::FEVector<double>>(*artery_dis_->element_row_map(), true);
    unaffected_integrated_artery_diameters_col_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*artery_dis_->element_col_map(), true);
    integrated_artery_diameters_col_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*artery_dis_->element_col_map(), true);
    artery_elements_diameters_col_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*artery_dis_->element_col_map(), true);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::assemble(
    const int& ele1_gid, const int& ele2_gid, const double& integrated_diameter,
    std::vector<Core::LinAlg::SerialDenseVector> const& ele_rhs,
    std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const& ele_matrix,
    const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    const std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
  // call base class
  PorofluidElastScatraArteryCouplingNonConformingAlgorithm::assemble(
      ele1_gid, ele2_gid, integrated_diameter, ele_rhs, ele_matrix, sysmat, rhs);

  // also assemble the diameter if necessary
  if (homogenized_dis_->name() == "porofluid" && has_variable_diameter_)
    integrated_artery_diameters_row_->sum_into_global_values(1, &ele1_gid, &(integrated_diameter));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::
    set_artery_diameter_in_material()
{
  // assemble
  integrated_artery_diameters_row_->complete();

  // export to column format
  Core::LinAlg::export_to(Core::LinAlg::Vector<double>(*integrated_artery_diameters_row_),
      *integrated_artery_diameters_col_);

  // fill the vector collecting the element diameter
  fill_artery_ele_diam_col();

  // find the free-hanging elements which will be deleted
  std::vector<int> elements_to_be_deleted;
  if (delete_free_hanging_elements_) find_free_hanging_1d_elements(elements_to_be_deleted);

  // set the diameter in material
  for (int i = 0; i < artery_dis_->num_my_col_elements(); ++i)
  {
    // pointer to current element
    const Core::Elements::Element* current_element = artery_dis_->l_col_element(i);
    const int ele_gid = current_element->id();

    double diameter = artery_elements_diameters_col_->get_values()[i];

    // set to zero for free-hanging elements
    if (delete_free_hanging_elements_)
    {
      if (std::ranges::find(elements_to_be_deleted.begin(), elements_to_be_deleted.end(),
              ele_gid) != elements_to_be_deleted.end())
        diameter = 0.0;
    }

    // get the artery-material
    std::shared_ptr<Mat::Cnst1dArt> artery_material =
        std::dynamic_pointer_cast<Mat::Cnst1dArt>(current_element->material());
    if (artery_material == nullptr) FOUR_C_THROW("cast to artery material failed");

    // set to zero if collapsed
    if (diameter < artery_material->collapse_threshold())
    {
      // Collapse happens for the first time --> inform user
      if (artery_material->diam() >= artery_material->collapse_threshold() &&
          current_element->owner() == my_mpi_rank_)
        std::cout << ">>>>>> Artery element " << current_element->id() << " just collapsed <<<<<<"
                  << '\n';
      artery_material->set_diam(0.0);
    }
    else
      // otherwise set to calculated diameter
      artery_material->set_diam(diameter);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::
    reset_integrated_diameter_to_zero()
{
  integrated_artery_diameters_row_->put_scalar(0.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::
    fill_artery_ele_diam_col()
{
  // reset
  artery_elements_diameters_col_->put_scalar(0.0);
  // set the diameter in the vector
  for (int i = 0; i < artery_dis_->num_my_col_elements(); ++i)
  {
    // pointer to current element
    const Core::Elements::Element* current_element = artery_dis_->l_col_element(i);
    const int ele_gid = current_element->id();

    const std::vector<double> segment_lengths = get_ele_segment_length(ele_gid);
    const double current_ele_length =
        std::accumulate(segment_lengths.begin(), segment_lengths.end(), 0.0);
    // diam = int(diam)/length_element
    // also add the unaffected diameter --> diameter of artery elements which protrude
    const double diameter = (integrated_artery_diameters_col_->get_values()[i] +
                                unaffected_integrated_artery_diameters_col_->get_values()[i]) /
                            current_ele_length;

    artery_elements_diameters_col_->replace_local_value(i, diameter);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::
    find_free_hanging_1d_elements(std::vector<int>& elements_to_be_deleted)
{
  // user info
  if (my_mpi_rank_ == 0)
  {
    std::cout << "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<"
                 "<<<<<<<<<<<<<<<<<<<<<<<"
              << '\n';
    std::cout << ">>>>>>                               Find free-hanging 1D elements               "
                 "               <<<<<<"
              << '\n';
  }
  // get fully overlapping discretization
  std::shared_ptr<Core::FE::Discretization> artery_fully_overlapping_dis =
      create_fully_overlapping_artery_discretization(*artery_dis_, "conn_comp_dis", true);

  // vector to mark visited nodes
  std::shared_ptr<Core::LinAlg::Vector<int>> visited = std::make_shared<Core::LinAlg::Vector<int>>(
      *artery_fully_overlapping_dis->node_col_map(), true);

  // get the fully overlapping diameter vector
  std::shared_ptr<Core::LinAlg::Vector<double>> ele_artery_diameters_fully_overlapping =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *artery_fully_overlapping_dis->element_col_map(), true);
  Core::LinAlg::Vector<double> ele_artery_diameters_row(*artery_dis_->element_row_map(), true);
  Core::LinAlg::export_to(*artery_elements_diameters_col_, ele_artery_diameters_row);
  Core::LinAlg::export_to(ele_artery_diameters_row, *ele_artery_diameters_fully_overlapping);

  // vector of connected components of 1D graph
  std::vector<std::vector<int>> connected_components;
  int num_connected_components = 0;
  int num_connected_components_without_single_nodes = 0;

  // loop over fully-overlapping discretization
  for (int i = 0; i < artery_fully_overlapping_dis->num_my_col_nodes(); ++i)
  {
    // if not visited start a new connected component
    if (Core::Nodes::Node* current_node = artery_fully_overlapping_dis->l_col_node(i);
        visited->get_local_values()[current_node->lid()] == 0)
    {
      connected_components.push_back(std::vector<int>());
      // recursive call to depth-first search
      depth_first_search(current_node, visited, artery_fully_overlapping_dis,
          ele_artery_diameters_fully_overlapping, connected_components[num_connected_components]);
      // single nodes are not of interest as they are detected (and deleted) anyway
      if (connected_components[num_connected_components].size() > 1)
        num_connected_components_without_single_nodes++;

      num_connected_components++;
    }
  }

  // user info
  if (my_mpi_rank_ == 0 && num_connected_components_without_single_nodes > 1)
  {
    std::cout << "found " << num_connected_components_without_single_nodes
              << " connected components" << '\n';
  }

  const auto dirichlet_node_ids =
      Core::Conditions::find_conditioned_node_ids(*artery_fully_overlapping_dis, "Dirichlet",
          Core::Conditions::LookFor::locally_owned_and_ghosted);
  // loop over all connected components
  for (unsigned int i = 0; i < connected_components.size(); ++i)
  {
    // single nodes are not of interest as they are detected anyway
    if (const int connected_components_size = connected_components[i].size();
        connected_components_size > 1)
    {
      // user info
      if (my_mpi_rank_ == 0)
        std::cout << "connected_component with ID " << i
                  << " of size: " << connected_components_size << '\n';

      // check if any nodes of this connected component have a Dirichlet BC
      bool has_dirichlet = false;
      for (int j = 0; j < connected_components_size; ++j)
      {
        has_dirichlet = dirichlet_node_ids.contains(
            artery_fully_overlapping_dis->g_node((connected_components[i])[j])->id());
        if (has_dirichlet)
        {
          if (my_mpi_rank_ == 0)
            std::cout << "   ---> has at least one Dirichlet boundary condition" << '\n';
          break;
        }
      }

      // if no node of this connected component has a DBC or if it is smaller than the
      // user-specified threshold, all its elements are taken out
      if (!has_dirichlet or connected_components_size <
                                static_cast<int>(threshold_delete_free_hanging_elements_ *
                                                 artery_fully_overlapping_dis->num_global_nodes()))
      {
        // get the elements which have to be deleted
        for (int j = 0; j < connected_components_size; ++j)
        {
          Core::Nodes::Node* current_node =
              artery_fully_overlapping_dis->g_node((connected_components[i])[j]);
          for (auto ele : current_node->adjacent_elements())
            elements_to_be_deleted.push_back(ele.global_id());
        }
        // user info
        if (my_mpi_rank_ == 0)
        {
          if (!has_dirichlet)
          {
            std::cout
                << "   ---> has no Dirichlet boundary condition --> its elements will be taken out"
                << '\n';
          }
          if (has_dirichlet and
              connected_components_size <
                  static_cast<int>(threshold_delete_free_hanging_elements_ *
                                   artery_fully_overlapping_dis->num_global_nodes()))
          {
            std::cout << "   ---> smaller than threshold size of "
                      << static_cast<int>(threshold_delete_free_hanging_elements_ *
                                          artery_fully_overlapping_dis->num_global_nodes())
                      << " --> its elements will be taken out" << '\n';
          }
        }
      }
    }
  }

  // user info
  if (my_mpi_rank_ == 0)
  {
    std::cout << "\n>>>>>>                           End of Find free-hanging 1D elements          "
                 "                 <<<<<<"
              << '\n';
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<"
                 "<<<<<<<<<<<<<<<<<<<<<<<\n"
              << '\n';
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::depth_first_search(
    Core::Nodes::Node* current_node, std::shared_ptr<Core::LinAlg::Vector<int>> checked_nodes,
    std::shared_ptr<Core::FE::Discretization> artery_dis_fully_overlapping,
    std::shared_ptr<const Core::LinAlg::Vector<double>> artery_ele_diameters_fully_overlapping,
    std::vector<int>& current_connected_component)
{
  // mark this node visited and add it to this connected component
  const int lid = checked_nodes->get_map().lid(current_node->id());
  (*checked_nodes).get_local_values()[lid] = 1;
  current_connected_component.push_back(current_node->id());

  // check all adjacent elements (edges)
  for (auto ele : current_node->adjacent_elements())
  {
    // get diameter
    const double diameter = artery_ele_diameters_fully_overlapping->get_values()[ele.local_id()];

    // get the artery-material
    std::shared_ptr<Mat::Cnst1dArt> artery_material =
        std::dynamic_pointer_cast<Mat::Cnst1dArt>(ele.user_element()->material());
    if (artery_material == nullptr) FOUR_C_THROW("cast to artery material failed");

    // if the element is not collapsed, it is connected to this node,
    // and we continue with the depth-first search with all nodes of this element
    if (diameter >= artery_material->collapse_threshold())
    {
      for (auto node : ele.nodes())
      {
        if (checked_nodes->get_local_values()[node.local_id()] == 0)
          depth_first_search(node.user_node(), checked_nodes, artery_dis_fully_overlapping,
              artery_ele_diameters_fully_overlapping, current_connected_component);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::
    evaluate_additional_linearization_of_integrated_diameter()
{
  // linearizations
  std::vector<Core::LinAlg::SerialDenseMatrix> ele_matrix(2);

  // evaluate all pairs
  for (const auto& coupled_ele_pair : coupled_ele_pairs_)
  {
    // only needed if variable diameter is set for this pair
    if (coupled_ele_pair->variable_diameter_active())
    {
      // evaluate
      coupled_ele_pair->evaluate_additional_linearization_of_integrated_diameter(
          &(ele_matrix[0]), &(ele_matrix[1]));

      // and FE-Assemble
      const int ele1_gid = coupled_ele_pair->artery_ele_gid();
      const int ele2_gid = coupled_ele_pair->homogenized_ele_gid();
      const Core::Elements::Element* ele1 = artery_dis_->g_element(ele1_gid);
      const Core::Elements::Element* ele2 = homogenized_dis_->g_element(ele2_gid);
      // get element location vector and ownerships
      std::vector<int> lm_row_1;
      std::vector<int> lm_row_2;
      std::vector<int> lm_row_owner_1;
      std::vector<int> lm_row_owner_2;
      std::vector<int> lm_stride;

      ele1->location_vector(*artery_dis_, lm_row_1, lm_row_owner_1, lm_stride);
      ele2->location_vector(*homogenized_dis_, lm_row_2, lm_row_owner_2, lm_stride);

      coupling_matrix_->fe_assemble(ele_matrix[0], lm_row_1, lm_row_1);
      coupling_matrix_->fe_assemble(ele_matrix[1], lm_row_1, lm_row_2);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::apply_mesh_movement()
{
  // in a pure porofluid problem the length vectors are never created and not needed

  if (!artery_segment_lengths_)
  {
    FOUR_C_ASSERT_ALWAYS(pure_porofluid_problem_,
        "vector with artery segment lengths is never created although we are NOT in a pure "
        "porofluid problem type!");
  }


  apply_mesh_movement_for_pairs(segment_context(), coupled_ele_pairs_,
      *artery_segment_lengths_->unaffected, *artery_segment_lengths_->current);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::vector<double>
PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::get_ele_segment_length(
    const int artery_ele_gid)
{
  return get_ele_segment_lengths(segment_context(), gid_to_segment_length_, artery_ele_gid);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::ArterySegmentContext
PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::segment_context()
{
  return ArterySegmentContext{.artery_dis = artery_dis_.get(),
      .homogenized_dis = homogenized_dis_.get(),
      .my_mpi_rank = my_mpi_rank_,
      .max_num_segments_per_artery_element = max_num_segments_per_artery_element_,
      .pure_porofluid_problem = pure_porofluid_problem_,
      .evaluate_in_ref_config = evaluate_in_ref_config_};
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::
    print_coupling_method() const
{
  std::cout << "<   Line-based formulation                         >" << '\n';
  PorofluidElastScatraArteryCouplingNonConformingAlgorithm::print_coupling_method();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::output_summary() const
{
  if (my_mpi_rank_ == 0)
  {
    std::cout << "\nSummary of coupling pairs (segments):" << '\n';
    std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << '\n';
  }
  Core::Communication::barrier(get_comm());
  for (const auto& coupled_ele_pair : coupled_ele_pairs_)
  {
    std::cout << "Proc " << std::right << std::setw(2) << my_mpi_rank_ << ": Artery-ele "
              << std::right << std::setw(5) << coupled_ele_pair->artery_ele_gid() << ":   ["
              << std::left << std::setw(11) << coupled_ele_pair->eta_start() << "," << std::right
              << std::setw(11) << coupled_ele_pair->eta_end() << "] <---> continuous-ele "
              << std::right << std::setw(7) << coupled_ele_pair->homogenized_ele_gid() << '\n';
  }
  Core::Communication::barrier(get_comm());
  if (my_mpi_rank_ == 0) std::cout << "\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> PoroPressureBased::
    PorofluidElastScatraArteryCouplingLineBasedAlgorithm::blood_vessel_volume_fraction()
{
  return blood_vessel_volfrac_;
}

FOUR_C_NAMESPACE_CLOSE
