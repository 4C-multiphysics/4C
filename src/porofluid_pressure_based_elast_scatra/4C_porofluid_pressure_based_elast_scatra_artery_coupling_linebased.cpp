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
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mat_cnst_1d_art.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_pair.hpp"
#include "4C_porofluid_pressure_based_utils.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::
    PorofluidElastScatraArteryCouplingLineBasedAlgorithm(
        std::shared_ptr<Core::FE::Discretization> artery_dis,
        std::shared_ptr<Core::FE::Discretization> homogenized_dis,
        const Teuchos::ParameterList& coupling_params, const std::string& condition_name)
    : PorofluidElastScatraArteryCouplingNonConformingAlgorithm(
          artery_dis, homogenized_dis, coupling_params, condition_name),
      max_num_segments_per_artery_element_(Global::Problem::instance()
              ->porofluid_pressure_based_dynamic_params()
              .sublist("artery_coupling")
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

  // create the GID to segment vector
  create_gid_to_segment_vector();

  // fill length of artery elements that are not changed by deformation of the underlying 2D/3D mesh
  // (basically protruding artery elements or segments)
  fill_unaffected_artery_length();

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

  // delete the inactive and duplicated pairs
  std::vector<std::shared_ptr<PorofluidElastScatraArteryCouplingPairBase>> active_coupled_ele_pairs;
  for (auto& coupled_ele_pair : coupled_ele_pairs_)
  {
    const int homogenized_ele_gid = coupled_ele_pair->homogenized_ele_gid();
    const Core::Elements::Element* homogenized_ele =
        homogenized_dis_->g_element(homogenized_ele_gid);

    if (coupled_ele_pair->is_active() &&
        !is_duplicate_segment(active_coupled_ele_pairs, *coupled_ele_pair) &&
        homogenized_ele->owner() == my_mpi_rank_)
      active_coupled_ele_pairs.push_back(coupled_ele_pair);
  }

  // the following case takes care of the special case where the 1D element lies exactly in
  // between two 2D/3D-elements which are owned by different processors

  // fill the GID-to-segment vector
  std::map<int, std::vector<double>> gid_to_segment_length;
  fill_gid_to_segment_vector(active_coupled_ele_pairs, gid_to_segment_length);

  // dummy map to collect duplicates in form [ele2gid, eta_a, eta_b, ... ];
  std::map<int, std::vector<double>> duplicates;

  // loop over all artery elements
  for (int i = 0; i < artery_dis_->element_col_map()->num_my_elements(); ++i)
  {
    if (const int artery_ele_gid = artery_dis_->element_col_map()->gid(i);
        gid_to_segment_length[artery_ele_gid].size() > 0)  // check if element projects
    {
      // compare all segment with each other if it might be identical
      for (int iseg = 0; std::cmp_less(iseg, gid_to_segment_length[artery_ele_gid].size() / 2);
          iseg++)
      {
        const double eta_a = gid_to_segment_length[artery_ele_gid][2 * iseg];
        const double eta_b = gid_to_segment_length[artery_ele_gid][2 * iseg + 1];
        for (int jseg = iseg + 1;
            std::cmp_less(jseg, gid_to_segment_length[artery_ele_gid].size() / 2); jseg++)
        {
          const double eta_a_jseg = gid_to_segment_length[artery_ele_gid][2 * jseg];
          const double eta_b_jseg = gid_to_segment_length[artery_ele_gid][2 * jseg + 1];
          // identical segment found
          if (fabs(eta_a - eta_a_jseg) < 1.0e-9 && fabs(eta_b - eta_b_jseg) < 1.0e-9)
          {
            // we need this to get the GID of the second element
            int id = -1;
            if (is_identical_segment(active_coupled_ele_pairs, artery_ele_gid, eta_a, eta_b, id))
            {
              const int ele2_gid = active_coupled_ele_pairs[id]->homogenized_ele_gid();
              duplicates[artery_ele_gid].push_back((ele2_gid));
              duplicates[artery_ele_gid].push_back(eta_a);
              duplicates[artery_ele_gid].push_back(eta_b);
            }
          }
        }
      }
    }
  }

  // communicate the map to all procs
  std::vector<int> mpi_ranks(Core::Communication::num_mpi_ranks(get_comm()));
  for (int i = 0; i < Core::Communication::num_mpi_ranks(get_comm()); ++i) mpi_ranks[i] = i;
  Core::LinAlg::gather<double>(
      duplicates, duplicates, static_cast<int>(mpi_ranks.size()), mpi_ranks.data(), get_comm());

  // remove duplicate (the one where the 2D/3D element has the large ID)
  for (auto& duplicate : duplicates)
  {
    const int artery_ele_gid = duplicate.first;
    std::vector<double> current_duplicates = duplicate.second;
    // should always be a multiple of six because we should always find exactly two/four, etc.
    // duplicates
    if (current_duplicates.size() % 6 != 0)
      FOUR_C_THROW(
          "duplicate vector has size {}, should be multiple of six", current_duplicates.size());
    // compare the possible duplicates
    for (int idupl = 0; std::cmp_less(idupl, (current_duplicates.size() / 3)); idupl++)
    {
      const double eta_a = current_duplicates[3 * idupl + 1];
      const double eta_b = current_duplicates[3 * idupl + 2];
      for (int jdupl = idupl + 1; std::cmp_less(jdupl, (current_duplicates.size() / 3)); jdupl++)
      {
        const double eta_a_jdupl = current_duplicates[3 * jdupl + 1];
        const double eta_b_jdupl = current_duplicates[3 * jdupl + 2];
        // duplicate found
        if (fabs(eta_a - eta_a_jdupl) < 1.0e-9 && fabs(eta_b - eta_b_jdupl) < 1.0e-9)
        {
          const int ele_i = static_cast<int>(current_duplicates[3 * idupl]);
          const int ele_j = static_cast<int>(current_duplicates[3 * jdupl]);
          const int ele_to_be_erased = std::max(ele_i, ele_j);
          int id = -1;
          // delete the duplicate with the larger ele2_gid
          if (is_identical_segment(active_coupled_ele_pairs, artery_ele_gid, eta_a, eta_b, id))
          {
            if (active_coupled_ele_pairs[id]->homogenized_ele_gid() == ele_to_be_erased)
            {
              active_coupled_ele_pairs.erase(active_coupled_ele_pairs.begin() + id);
            }
          }
        }
      }
    }
  }

  // overwrite the coupling pairs
  coupled_ele_pairs_ = active_coupled_ele_pairs;

  // output
  int total_num_active_pairs = 0;
  int num_active_pairs = static_cast<int>(coupled_ele_pairs_.size());
  total_num_active_pairs = Core::Communication::sum_all(num_active_pairs, get_comm());
  if (my_mpi_rank_ == 0)
  {
    std::cout << "Only " << total_num_active_pairs
              << " Artery-to-PoroMultiphaseScatra coupling pairs (segments) are active" << '\n';
  }
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::
    fill_unaffected_artery_length()
{
  // no need to do this for a pure porofluid problem
  if (pure_porofluid_problem_)
  {
    for (int i = 0; i < artery_dis_->element_col_map()->num_my_elements(); ++i)
    {
      const int artery_ele_gid = artery_dis_->element_col_map()->gid(i);
      Core::Elements::Element* artery_element = artery_dis_->g_element(artery_ele_gid);

      // TODO: this will not work for higher order artery elements
      const double initial_length = get_max_nodal_distance(artery_element, *artery_dis_);
      const int num_segments = static_cast<int>(gid_to_segment_[artery_ele_gid].size() / 2);
      gid_to_segment_length_[artery_ele_gid].resize(num_segments);
      for (int iseg = 0; iseg < num_segments; iseg++)
      {
        const double etaA = gid_to_segment_[artery_ele_gid][2 * iseg];
        const double etaB = gid_to_segment_[artery_ele_gid][2 * iseg + 1];
        gid_to_segment_length_[artery_ele_gid][iseg] = initial_length * (etaB - etaA) / 2.0;

        // return also id -> index in coupled_ele_pairs_ of this segment
        // and set iseg as the segment id of the coupling pairs
        if (int id = -1; is_identical_segment(coupled_ele_pairs_, artery_ele_gid, etaA, etaB, id))
          coupled_ele_pairs_[id]->set_segment_id(iseg);
      }
    }

    return;
  }

  // The unaffected length is the length of 1D elements not changed by deformation,
  // basically if these elements protrude.
  // For each element, this length is computed as: ele_length - sum_segments seg_length.
  // If the above quantity is bigger than zero, a 1D element protrudes.

  // initialize the unaffected and current lengths
  unaffected_artery_segment_lengths_ =
      std::make_shared<Core::LinAlg::FEVector<double>>(*artery_dis_->dof_row_map(1), true);
  current_artery_segment_lengths_ =
      std::make_shared<Core::LinAlg::FEVector<double>>(*artery_dis_->dof_row_map(1));

  // set segment ID on coupling pairs and fill the unaffected artery length
  for (int iele = 0; iele < artery_dis_->element_col_map()->num_my_elements(); ++iele)
  {
    const int artery_ele_gid = artery_dis_->element_col_map()->gid(iele);
    Core::Elements::Element* current_element = artery_dis_->g_element(artery_ele_gid);

    // TODO: this will not work for higher order artery elements
    const double initial_length = get_max_nodal_distance(current_element, *artery_dis_);

    std::vector<double> segment_boundaries = gid_to_segment_[artery_ele_gid];
    for (unsigned int iseg = 0; iseg < segment_boundaries.size() / 2; iseg++)
    {
      // get EtaA and etaB and calculate initial length
      const double etaA = segment_boundaries[iseg * 2];
      const double etaB = segment_boundaries[iseg * 2 + 1];
      const double segment_length = initial_length * (etaB - etaA) / 2.0;

      // since we use an FE vector
      if (current_element->owner() == my_mpi_rank_)
      {
        // build the location array
        std::vector<int> segment_length_dofs = artery_dis_->dof(1, current_element);
        unaffected_artery_segment_lengths_->sum_into_global_values(
            1, &segment_length_dofs[iseg], &segment_length);
      }

      // return also id -> index in coupled_ele_pairs_ of this segment
      // and set iseg as the segment id of the coupling pairs
      if (int id = -1; is_identical_segment(coupled_ele_pairs_, artery_ele_gid, etaA, etaB, id))
        coupled_ele_pairs_[id]->set_segment_id(static_cast<int>(iseg));
    }
  }

  if (unaffected_artery_segment_lengths_->complete() != 0)
    FOUR_C_THROW("GlobalAssemble of unaffected_artery_segment_lengths_ failed.");

  // subtract the segment lengths only if we evaluate in current configuration
  if (!evaluate_in_ref_config_)
  {
    for (const auto& coupled_ele_pair : coupled_ele_pairs_)
    {
      // get the initial lengths
      double initial_segment_length = coupled_ele_pair->apply_mesh_movement(true, homogenized_dis_);
      initial_segment_length *= -1.0;

      const int artery_ele_gid = coupled_ele_pair->artery_ele_gid();
      const Core::Elements::Element* current_element = artery_dis_->g_element(artery_ele_gid);

      std::vector<int> segment_length_dofs = artery_dis_->dof(1, current_element);
      const int segment_id = coupled_ele_pair->get_segment_id();

      unaffected_artery_segment_lengths_->sum_into_global_values(
          1, &segment_length_dofs[segment_id], &(initial_segment_length));
    }
    if (unaffected_artery_segment_lengths_->complete() != 0)
      FOUR_C_THROW("GlobalAssemble of unaffected_seg_lengths_artery_ failed");
  }
  // the current length is simply the unaffected length
  else
  {
    current_artery_segment_lengths_->update(1.0, *unaffected_artery_segment_lengths_, 0.0);
  }
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
    double initial_segment_length = coupled_ele_pair->apply_mesh_movement(true, homogenized_dis_);
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
  if (unaffected_artery_diameters_row.complete() != 0)
    FOUR_C_THROW("GlobalAssemble of unaffected_artery_diameters_row failed");
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
    if (blood_vessel_volfrac_->sum_into_global_values(1, &volfrac, &homogenized_ele_gid))
      FOUR_C_THROW("SumIntoGlobalValues failed!");
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
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::
    create_gid_to_segment_vector()
{
  // fill the GID-to-segment vector
  fill_gid_to_segment_vector(coupled_ele_pairs_, gid_to_segment_);

  // sort and take care of special cases
  for (int i = 0; i < artery_dis_->element_col_map()->num_my_elements(); ++i)
  {
    if (const int artery_ele_gid = artery_dis_->element_col_map()->gid(i);
        gid_to_segment_[artery_ele_gid].size() > 0)  // check if element projects
    {
      std::ranges::sort(
          gid_to_segment_[artery_ele_gid].begin(), gid_to_segment_[artery_ele_gid].end());
      const int end = static_cast<int>(gid_to_segment_[artery_ele_gid].size());

      // the end of the element lies outside the domain
      if (const double value_at_end = gid_to_segment_[artery_ele_gid][end - 1];
          fabs(value_at_end - 1.0) > 1.0e-9)
      {
        gid_to_segment_[artery_ele_gid].push_back(value_at_end);
        gid_to_segment_[artery_ele_gid].push_back(1.0);
      }

      // the beginning of the element lies outside the domain
      if (const double value_at_beginning = gid_to_segment_[artery_ele_gid][0];
          fabs(value_at_beginning + 1.0) > 1.0e-9)
      {
        gid_to_segment_[artery_ele_gid].insert(
            gid_to_segment_[artery_ele_gid].begin(), value_at_beginning);
        gid_to_segment_[artery_ele_gid].insert(gid_to_segment_[artery_ele_gid].begin(), -1.0);
      }
    }
    // this element does not project
    else
    {
      gid_to_segment_[artery_ele_gid].push_back(-1.0);
      gid_to_segment_[artery_ele_gid].push_back(1.0);
    }
  }

  // safety checks
  for (int i = 0; i < artery_dis_->element_col_map()->num_my_elements(); ++i)
  {
    // 1) check if the artery element has more than MAXNUMSEGPERARTELE segments
    const int artery_ele_gid = artery_dis_->element_col_map()->gid(i);
    if (static_cast<int>(gid_to_segment_[artery_ele_gid].size()) >
        2 * max_num_segments_per_artery_element_)
    {
      FOUR_C_THROW(
          "Artery element {} has {} segments, which is more than the maximum allowed number of "
          "{} "
          "segments per artery element, increase MAXNUMSEGPERARTELE",
          artery_ele_gid, static_cast<int>(gid_to_segment_[artery_ele_gid].size() / 2),
          max_num_segments_per_artery_element_);
    }
    // 2) check if the segment has been overlooked
    for (int iseg = 0; iseg < static_cast<int>(gid_to_segment_[artery_ele_gid].size() / 2) - 1;
        iseg++)
    {
      if (fabs(gid_to_segment_[artery_ele_gid][2 * iseg + 1] -
               gid_to_segment_[artery_ele_gid][2 * iseg + 2]) > 1.0e-9)
      {
        std::cout << "Problem with segments of artery-element " << artery_ele_gid << ":" << '\n';
        for (int jseg = 0; std::cmp_less(jseg, gid_to_segment_[artery_ele_gid].size() / 2); jseg++)
        {
          std::cout << "[" << gid_to_segment_[artery_ele_gid][2 * jseg] << ", "
                    << gid_to_segment_[artery_ele_gid][2 * jseg + 1] << "]" << '\n';
        }
        FOUR_C_THROW(
            "artery element {} has probably not found all possible segments", artery_ele_gid);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::
    fill_gid_to_segment_vector(
        const std::vector<std::shared_ptr<PorofluidElastScatraArteryCouplingPairBase>>&
            coupled_ele_pairs,
        std::map<int, std::vector<double>>& gid_to_segment_length) const
{
  // fill the GID-to-segment vector
  for (const auto& coupled_ele_pair : coupled_ele_pairs)
  {
    const int artery_ele_gid = coupled_ele_pair->artery_ele_gid();
    const int homogenized_ele_gid = coupled_ele_pair->homogenized_ele_gid();

    const Core::Elements::Element* homogenized_ele =
        homogenized_dis_->g_element(homogenized_ele_gid);

    const double etaA = coupled_ele_pair->eta_start();
    const double etaB = coupled_ele_pair->eta_end();

    if (homogenized_ele->owner() == my_mpi_rank_)
    {
      gid_to_segment_length[artery_ele_gid].push_back(etaA);
      gid_to_segment_length[artery_ele_gid].push_back(etaB);
    }
    else
    {
      FOUR_C_THROW(
          "Something went wrong here, pair in coupling ele pairs where continuous-discretization "
          "element is not owned by this proc.");
    }
  }

  // communicate it to all procs
  std::vector<int> all_procs(Core::Communication::num_mpi_ranks(get_comm()));
  for (int i = 0; i < Core::Communication::num_mpi_ranks(get_comm()); ++i) all_procs[i] = i;
  Core::LinAlg::gather<double>(gid_to_segment_length, gid_to_segment_length,
      static_cast<int>(all_procs.size()), all_procs.data(), get_comm());
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
  if (integrated_artery_diameters_row_->complete() != 0)
    FOUR_C_THROW("GlobalAssemble of integrated_integrated_diams_artery_row_ failed");

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
std::vector<double>
PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::get_ele_segment_length(
    const int artery_ele_gid)
{
  if (pure_porofluid_problem_) return gid_to_segment_length_[artery_ele_gid];

  // safety checks
  if (!artery_dis_->has_state(1, "curr_seg_lengths"))
    FOUR_C_THROW("cannot get state curr_seg_lengths");

  // build the location array
  const Core::Elements::Element* current_element = artery_dis_->g_element(artery_ele_gid);
  const std::vector<int> segment_length_dof = artery_dis_->dof(1, current_element);

  const std::shared_ptr<const Core::LinAlg::Vector<double>> current_segment_lengths =
      artery_dis_->get_state(1, "curr_seg_lengths");

  std::vector<double> segment_lengths =
      Core::FE::extract_values(*current_segment_lengths, segment_length_dof);

  return segment_lengths;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::is_duplicate_segment(
    const std::vector<std::shared_ptr<PorofluidElastScatraArteryCouplingPairBase>>&
        coupled_ele_pairs,
    const PorofluidElastScatraArteryCouplingPairBase& possible_duplicate)
{
  // we have to sort out duplicate segments, these might occur if the artery element
  // lies exactly between two different 2D/3D-elements

  const double eta_a = possible_duplicate.eta_start();
  const double eta_b = possible_duplicate.eta_end();
  const int ele1_gid = possible_duplicate.artery_ele_gid();
  int ele_pair_id = -1;

  return is_identical_segment(coupled_ele_pairs, ele1_gid, eta_a, eta_b, ele_pair_id);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::is_identical_segment(
    const std::vector<std::shared_ptr<PorofluidElastScatraArteryCouplingPairBase>>&
        coupled_ele_pairs,
    const int& ele1_gid, const double& etaA, const double& etaB, int& ele_pair_id)
{
  for (unsigned i = 0; i < coupled_ele_pairs.size(); i++)
  {
    // first check if ele1-Gid is identical
    if (ele1_gid == coupled_ele_pairs[i]->artery_ele_gid())
    {
      // check if the integration segment is the same
      if (fabs(etaA - coupled_ele_pairs[i]->eta_start()) < 1.0e-9 &&
          fabs(etaB - coupled_ele_pairs[i]->eta_end()) < 1.0e-9)
      {
        if constexpr (projection_output) std::cout << "found duplicate integration segment" << '\n';
        ele_pair_id = static_cast<int>(i);
        return true;
      }
    }
  }

  ele_pair_id = -1;
  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingLineBasedAlgorithm::apply_mesh_movement()
{
  // no need to do this
  if (pure_porofluid_problem_) return;

  // only if we evaluate in current configuration
  if (!evaluate_in_ref_config_)
  {
    // safety
    if (!homogenized_dis_->has_state(1, "dispnp")) FOUR_C_THROW("cannot get displacement state");

    // update with unaffected length
    current_artery_segment_lengths_->update(1.0, *unaffected_artery_segment_lengths_, 0.0);

    // apply movement on pairs and fill gid-to-segment-length and current_seg_lengths_artery_
    for (const auto& coupled_ele_pair : coupled_ele_pairs_)
    {
      const double new_segment_length =
          coupled_ele_pair->apply_mesh_movement(false, homogenized_dis_);
      const int artery_ele_gid = coupled_ele_pair->artery_ele_gid();
      const int segment_id = coupled_ele_pair->get_segment_id();

      const Core::Elements::Element* artery_element = artery_dis_->g_element(artery_ele_gid);
      // build the location array
      std::vector<int> segment_length_dofs = artery_dis_->dof(1, artery_element);

      current_artery_segment_lengths_->sum_into_global_values(
          1, &segment_length_dofs[segment_id], &(new_segment_length));
    }

    if (current_artery_segment_lengths_->complete() != 0)
      FOUR_C_THROW("GlobalAssemble of current_seg_lengths_artery_ failed");
  }

  // set state on artery discretization
  artery_dis_->set_state(
      1, "curr_seg_lengths", Core::LinAlg::Vector<double>(*current_artery_segment_lengths_));
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