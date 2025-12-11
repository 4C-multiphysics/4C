// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_nonconforming.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mat_cnst_1d_art.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_pair.hpp"
#include "4C_porofluid_pressure_based_ele_parameter.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::
    PorofluidElastScatraArteryCouplingNonConformingAlgorithm(
        const std::shared_ptr<Core::FE::Discretization> artery_dis,
        const std::shared_ptr<Core::FE::Discretization> homogenized_dis,
        const Teuchos::ParameterList& coupling_params, const std::string& condition_name)
    : PorofluidElastScatraArteryCouplingBaseAlgorithm(artery_dis, homogenized_dis, coupling_params),
      coupling_params_(coupling_params),
      condition_name_(condition_name),
      porofluid_managers_initialized_(false),
      is_setup_(false),
      pure_porofluid_problem_(false),
      has_variable_diameter_(false),
      delete_free_hanging_elements_(Global::Problem::instance()
              ->porofluid_pressure_based_dynamic_params()
              .sublist("artery_coupling")
              .get<bool>("delete_free_hanging_elements")),
      threshold_delete_free_hanging_elements_(Global::Problem::instance()
              ->porofluid_pressure_based_dynamic_params()
              .sublist("artery_coupling")
              .get<double>("delete_small_components_fraction")),
      artery_coupling_method_(
          Teuchos::getIntegralValue<ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod>(
              coupling_params, "coupling_method")),
      timefacrhs_artery_(0.0),
      timefacrhs_homogenized_(0.0),
      penalty_parameter_(coupling_params_.get<double>("penalty_parameter"))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::init()
{
  // we do not have a moving mesh
  if (Global::Problem::instance()->get_problem_type() ==
      Core::ProblemType::porofluid_pressure_based)
  {
    evaluate_in_ref_config_ = true;
    pure_porofluid_problem_ = true;
  }

  // fill the vectors
  fill_function_and_scale_vectors();

  // initialize phinp for the homogenized discretization
  phinp_homogenized_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*homogenized_dis_->dof_row_map(), true);
  // initialize phin for the homogenized discretization
  phin_homogenized_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*homogenized_dis_->dof_row_map(), true);
  // initialize phinp for the artery discretization
  phinp_art_ = std::make_shared<Core::LinAlg::Vector<double>>(*artery_dis_->dof_row_map(), true);

  zeros_homogenized_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*homogenized_dis_->dof_row_map(), true);
  zeros_artery_ = std::make_shared<Core::LinAlg::Vector<double>>(*artery_dis_->dof_row_map(), true);

  // create empty mortar D and M matrices (27 adjacent nodes as 'good' guess)
  mortar_matrix_d_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(artery_dis_->dof_row_map()), 27, false, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  mortar_matrix_m_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *(artery_dis_->dof_row_map()), 27, false, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
  mortar_kappa_inv_ =
      std::make_shared<Core::LinAlg::FEVector<double>>(*artery_dis_->dof_row_map(), true);

  // full map of homogenized and artery dofs
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps;
  maps.push_back(std::make_shared<Core::LinAlg::Map>(*homogenized_dis_->dof_row_map()));
  maps.push_back(std::make_shared<Core::LinAlg::Map>(*artery_dis_->dof_row_map()));

  fullmap_ = Core::LinAlg::MultiMapExtractor::merge_maps(maps);
  /// dof row map of coupled problem split in (field) blocks
  global_extractor_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();
  global_extractor_->setup(*fullmap_, maps);

  coupling_matrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(
      *fullmap_, 81, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);

  coupling_rhs_vector_ = std::make_shared<Core::LinAlg::FEVector<double>>(*fullmap_);

  global_extractor_->check_for_valid_map_extractor();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::setup()
{
  // create the pairs
  if (artery_coupling_method_ ==
      ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::node_to_point)
  {
    get_coupling_nodes_from_input_node_to_point();
    if (coupling_nodes_for_node_to_point_.size() == 0)
      FOUR_C_THROW("No 1D Coupling Node Ids found for NTP Coupling");
    create_coupling_pairs_node_to_point();
  }
  else
  {
    create_coupling_pairs_line_surface_based();
  }

  // check if variable diameter is used
  if (homogenized_dis_->name() == "porofluid") set_flag_variable_diameter();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::
    get_coupling_nodes_from_input_node_to_point()
{
  FOUR_C_ASSERT(artery_coupling_method_ ==
                    ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::node_to_point,
      "This method should only be called for node-to-point coupling.");

  // get the node IDs of coupled 1D nodes from the input file
  std::vector<const Core::Conditions::Condition*> artery_coupling_ids;
  artery_dis_->get_condition(condition_name_, artery_coupling_ids);
  coupling_nodes_for_node_to_point_.resize(artery_coupling_ids.size());

  for (unsigned iter = 0; iter < artery_coupling_ids.size(); ++iter)
  {
    const std::vector<int>* ArteryNodeIds = (artery_coupling_ids[iter])->get_nodes();
    for (const auto coupling_id : *ArteryNodeIds)
    {
      coupling_nodes_for_node_to_point_[iter] = coupling_id;
      if (my_mpi_rank_ == 0)
      {
        std::cout << "Artery Coupling Node Id " << iter + 1
                  << " from Input = " << coupling_nodes_for_node_to_point_[iter] << "\n";
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::evaluate(
    const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    const std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
  if (!is_setup_) FOUR_C_THROW("setup() has not been called");

  if (!porofluid_managers_initialized_)
  {
    // set the right-hand side time factors (we assume constant time step size here)
    set_time_fac_rhs();
    for (const auto& coupled_ele_pair : coupled_ele_pairs_)
      coupled_ele_pair->setup_fluid_managers_and_materials(
          homogenized_dis_->name(), timefacrhs_artery_, timefacrhs_homogenized_);
    porofluid_managers_initialized_ = true;
  }

  // evaluate and assemble the pairs
  evaluate_coupling_pairs(sysmat, rhs);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::setup_system(
    Core::LinAlg::BlockSparseMatrixBase& sysmat, std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
    Core::LinAlg::SparseMatrix& sysmat_homogenized, Core::LinAlg::SparseMatrix& sysmat_artery,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_homogenized,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_artery,
    const Core::LinAlg::MapExtractor& dbcmap_homogenized, const Core::LinAlg::Map& dbcmap_artery,
    const Core::LinAlg::Map& dbcmap_artery_with_collapsed) const
{
  // add normal part to rhs
  rhs->update(1.0, *global_extractor_->insert_vector(*rhs_homogenized, 0), 1.0);
  rhs->update(1.0, *global_extractor_->insert_vector(*rhs_artery, 1), 1.0);

  // apply DBCs
  // 1) on vector
  Core::LinAlg::apply_dirichlet_to_system(
      *rhs, *zeros_homogenized_, *(dbcmap_homogenized.cond_map()));
  Core::LinAlg::apply_dirichlet_to_system(*rhs, *zeros_artery_, (dbcmap_artery));
  // 2) on OD-matrices
  sysmat.matrix(0, 1).complete(sysmat_artery.range_map(), sysmat_homogenized.range_map());
  sysmat.matrix(1, 0).complete(sysmat_homogenized.range_map(), sysmat_artery.range_map());
  sysmat.matrix(0, 1).apply_dirichlet(*(dbcmap_homogenized.cond_map()), false);
  sysmat.matrix(1, 0).apply_dirichlet((dbcmap_artery_with_collapsed), false);

  // 3) add the main-diagonal terms into the global sysmat
  Core::LinAlg::matrix_add(sysmat_homogenized, false, 1.0, sysmat.matrix(0, 0), 1.0);
  Core::LinAlg::matrix_add(sysmat_artery, false, 1.0, sysmat.matrix(1, 1), 1.0);
  sysmat.matrix(0, 0).complete();
  sysmat.matrix(1, 1).complete();
  // and apply DBC
  sysmat.matrix(0, 0).apply_dirichlet(*(dbcmap_homogenized.cond_map()), true);
  sysmat.matrix(1, 1).apply_dirichlet((dbcmap_artery_with_collapsed), true);
  // Assign view to 3D system matrix (such that it now includes also contributions from coupling)
  // this is necessary for the monolithic solution schemes
  sysmat_homogenized.assign(Core::LinAlg::DataAccess::Share, sysmat.matrix(0, 0));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::
    create_coupling_pairs_line_surface_based()
{
  const Teuchos::ParameterList& porofluid_coupling_params =
      Global::Problem::instance()->porofluid_pressure_based_dynamic_params().sublist(
          "artery_coupling");

  // loop over pairs found by search
  std::map<int, std::set<int>>::const_iterator nearby_ele_iter;
  int num_active_pairs = 0;
  for (nearby_ele_iter = nearby_ele_pairs_.begin(); nearby_ele_iter != nearby_ele_pairs_.end();
      ++nearby_ele_iter)
    num_active_pairs += nearby_ele_iter->second.size();

  coupled_ele_pairs_.resize(num_active_pairs);

  int coupled_ele_pair_idx = 0;
  for (nearby_ele_iter = nearby_ele_pairs_.begin(); nearby_ele_iter != nearby_ele_pairs_.end();
      ++nearby_ele_iter)
  {
    const int artery_ele_gid = nearby_ele_iter->first;
    std::vector<Core::Elements::Element const*> coupled_elements(2);
    coupled_elements[0] = artery_dis_->g_element(artery_ele_gid);

    std::set<int>::const_iterator second_ele_iter;
    for (second_ele_iter = nearby_ele_iter->second.begin();
        second_ele_iter != nearby_ele_iter->second.end(); ++second_ele_iter)
    {
      const int homogenized_ele_gid = *second_ele_iter;
      coupled_elements[1] = homogenized_dis_->g_element(homogenized_ele_gid);
      if (coupled_elements[1]->owner() == my_mpi_rank_)
      {
        // construct, init and setup coupling pairs
        const std::shared_ptr<PorofluidElastScatraArteryCouplingPairBase> current_pair =
            create_new_artery_coupling_pair(coupled_elements);
        current_pair->init(coupled_elements, coupling_params_, porofluid_coupling_params,
            coupled_dofs_homogenized_, coupled_dofs_artery_, scale_vector_, function_vector_,
            condition_name_, penalty_parameter_);

        // add to the list of current contact pairs
        coupled_ele_pairs_[coupled_ele_pair_idx] = current_pair;
        coupled_ele_pair_idx++;
      }
    }
  }
  coupled_ele_pairs_.resize(coupled_ele_pair_idx);

  // output
  int total_numactive_pairs = 0;
  num_active_pairs = static_cast<int>(coupled_ele_pairs_.size());
  total_numactive_pairs = Core::Communication::sum_all(num_active_pairs, get_comm());
  if (my_mpi_rank_ == 0)
  {
    std::cout << "\nFound " << total_numactive_pairs
              << " Artery-to-PoroMultiphaseScatra coupling pairs (segments)" << '\n';
  }

  nearby_ele_pairs_.clear();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::
    create_coupling_pairs_node_to_point()
{
  FOUR_C_ASSERT(artery_coupling_method_ ==
                    ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::node_to_point,
      "This method should only be called for node-to-point coupling.");

  const Teuchos::ParameterList& porofluid_coupling_params =
      Global::Problem::instance()->porofluid_pressure_based_dynamic_params().sublist(
          "artery_coupling");

  int num_active_pairs = std::accumulate(nearby_ele_pairs_.begin(), nearby_ele_pairs_.end(), 0,
      [](int a, auto b) { return a + (static_cast<int>(b.second.size())); });

  coupled_ele_pairs_.resize(num_active_pairs);

  // loop over pairs found by search
  int coupled_ele_pair_idx = 0;
  for (const auto& nearby_ele_iter : nearby_ele_pairs_)
  {
    // create vector of active coupling pairs
    std::vector<Core::Elements::Element const*> coupled_elements(2);
    // assign artery element
    coupled_elements[0] = artery_dis_->g_element(nearby_ele_iter.first);

    // get nodes of artery element
    const Core::Nodes::Node* const* nodes_artery = coupled_elements[0]->nodes();

    // loop over nodes of the artery element
    for (int i = 0; i < coupled_elements[0]->num_node(); i++)
    {
      // loop over prescribed couplings nodes from input
      for (unsigned int j = 0; j < coupling_nodes_for_node_to_point_.size(); j++)
      {
        // check if artery node is prescribed coupling node
        if (nodes_artery[i]->id() == coupling_nodes_for_node_to_point_[j])
        {
          // get coupling type (ARTERY or AIRWAY ?)
          std::vector<const Core::Conditions::Condition*> coupling_condition;
          artery_dis_->get_condition(condition_name_, coupling_condition);
          const auto coupling_element_type_ =
              (coupling_condition[j])->parameters().get<std::string>("COUPLING_TYPE");

          // recompute coupling dofs
          recompute_coupled_dofs_for_node_to_point_coupling(coupling_condition, j);

          // get penalty parameter
          const auto penalty = coupling_condition[j]->parameters().get<double>("PENALTY");

          // get eta (parameter coordinate of corresponding node)
          const int eta_ntp = (i == 0) ? -1 : 1;

          // loop over assigned 2D/3D elements
          for (const auto homogenized_ele_iter : nearby_ele_iter.second)
          {
            // assign 2D/3D element
            coupled_elements[1] = homogenized_dis_->g_element(homogenized_ele_iter);

            // only those pairs, where the 3D element is owned by this proc actually evaluated by
            // this proc
            if (coupled_elements[1]->owner() == my_mpi_rank_)
            {
              // construct, init and setup coupling pairs
              const std::shared_ptr<PorofluidElastScatraArteryCouplingPairBase> current_pair =
                  create_new_artery_coupling_pair(coupled_elements);
              current_pair->init(coupled_elements, coupling_params_, porofluid_coupling_params,
                  coupled_dofs_homogenized_, coupled_dofs_artery_, scale_vector_, function_vector_,
                  condition_name_, penalty, coupling_element_type_, eta_ntp);
              // add to the list of current contact pairs
              coupled_ele_pairs_[coupled_ele_pair_idx] = current_pair;
              coupled_ele_pair_idx++;
            }
          }
        }
      }
    }
  }
  coupled_ele_pairs_.resize(coupled_ele_pair_idx);

  // output
  int total_num_active_pairs = 0;
  num_active_pairs = static_cast<int>(coupled_ele_pairs_.size());
  total_num_active_pairs = Core::Communication::sum_all(num_active_pairs, get_comm());
  if (my_mpi_rank_ == 0)
  {
    std::cout << "\nFound " << total_num_active_pairs
              << " Artery-to-PoroMultiphaseScatra coupling pairs (segments)" << '\n';
  }

  nearby_ele_pairs_.clear();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::
    set_flag_variable_diameter()
{
  int has_variable_diameter = 0;
  // check all column elements if one of them uses the diameter law by function
  for (int i = 0; i < artery_dis_->num_my_col_elements(); ++i)
  {
    // pointer to current element
    const Core::Elements::Element* current_element = artery_dis_->l_col_element(i);

    // get the artery-material
    std::shared_ptr<Mat::Cnst1dArt> artery_material =
        std::dynamic_pointer_cast<Mat::Cnst1dArt>(current_element->material());
    if (artery_material == nullptr) FOUR_C_THROW("cast to artery material failed");

    if (artery_material->diameter_law() == Mat::PAR::ArteryDiameterLaw::diameterlaw_by_function)
    {
      has_variable_diameter = 1;
      break;
    }
  }

  // sum over all procs.
  int sum_has_variable_diameter = 0;
  sum_has_variable_diameter = Core::Communication::sum_all(has_variable_diameter, get_comm());
  // if one has a variable diameter, set the flag to true
  if (sum_has_variable_diameter > 0) has_variable_diameter_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::
    evaluate_coupling_pairs(const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        const std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
  // reset
  if (artery_coupling_method_ ==
      ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::mortar_penalty)
  {
    mortar_matrix_d_->zero();
    mortar_matrix_m_->zero();
    mortar_kappa_inv_->put_scalar(0.0);
  }

  coupling_matrix_->zero();
  coupling_rhs_vector_->put_scalar(0.0);

  // resulting discrete element force vectors of the two interacting elements
  std::vector<Core::LinAlg::SerialDenseVector> ele_rhs(2);

  // linearizations
  std::vector ele_matrix(2, std::vector<Core::LinAlg::SerialDenseMatrix>(2));

  // element mortar coupling matrices
  Core::LinAlg::SerialDenseMatrix D_ele;
  Core::LinAlg::SerialDenseMatrix M_ele;
  Core::LinAlg::SerialDenseVector Kappa_ele;

  // set states
  if (homogenized_dis_->name() == "porofluid")
  {
    homogenized_dis_->set_state("phinp_fluid", *phinp_homogenized_);
    homogenized_dis_->set_state("phin_fluid", *phin_homogenized_);
    artery_dis_->set_state("one_d_artery_pressure", *phinp_art_);
    if (not evaluate_in_ref_config_ && not homogenized_dis_->has_state(1, "velocity field"))
      FOUR_C_THROW(
          "evaluation in current configuration wanted but solid phase velocity not available!");
    if (has_variable_diameter_) reset_integrated_diameter_to_zero();
  }
  else if (homogenized_dis_->name() == "scatra")
  {
    homogenized_dis_->set_state("phinp", *phinp_homogenized_);
    artery_dis_->set_state("one_d_artery_phinp", *phinp_art_);
  }
  else
  {
    FOUR_C_THROW(
        "Only porofluid and scatra-discretizations are supported for linebased-coupling so far");
  }

  // evaluate all pairs
  for (const auto& coupled_ele_pair : coupled_ele_pairs_)
  {
    // reset state on pairs
    coupled_ele_pair->reset_state(homogenized_dis_, artery_dis_);

    // get the segment lengths
    const std::vector<double> segment_lengths =
        get_ele_segment_length(coupled_ele_pair->artery_ele_gid());

    // evaluate
    const double integrated_diameter = coupled_ele_pair->evaluate(&(ele_rhs[0]), &(ele_rhs[1]),
        &(ele_matrix[0][0]), &(ele_matrix[0][1]), &(ele_matrix[1][0]), &(ele_matrix[1][1]), &D_ele,
        &M_ele, &Kappa_ele, segment_lengths);

    // assemble
    assemble(coupled_ele_pair->artery_ele_gid(), coupled_ele_pair->homogenized_ele_gid(),
        integrated_diameter, ele_rhs, ele_matrix, sysmat, rhs);

    // in the case of MP, assemble D, M and Kappa
    if (artery_coupling_method_ ==
            ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::mortar_penalty and
        num_coupled_dofs_ > 0)
      assemble_mortar_matrices_and_vector(coupled_ele_pair->artery_ele_gid(),
          coupled_ele_pair->homogenized_ele_gid(), D_ele, M_ele, Kappa_ele);
  }

  // set artery diameter in material to be able to evaluate the 1D elements with variable diameter
  // and evaluate additional linearization of (integrated) element diameters
  if (homogenized_dis_->name() == "porofluid" && has_variable_diameter_)
  {
    set_artery_diameter_in_material();
    evaluate_additional_linearization_of_integrated_diameter();
  }

  coupling_rhs_vector_->complete();
  rhs->update(1.0, *coupling_rhs_vector_, 0.0);

  coupling_matrix_->complete();
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> artery_block =
      Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          *coupling_matrix_, *global_extractor_, *global_extractor_);

  artery_block->complete();
  Core::LinAlg::matrix_add(artery_block->matrix(1, 0), false, 1.0, sysmat->matrix(1, 0), 0.0);
  Core::LinAlg::matrix_add(artery_block->matrix(0, 1), false, 1.0, sysmat->matrix(0, 1), 0.0);
  Core::LinAlg::matrix_add(artery_block->matrix(0, 0), false, 1.0, sysmat->matrix(0, 0), 0.0);
  Core::LinAlg::matrix_add(artery_block->matrix(1, 1), false, 1.0, sysmat->matrix(1, 1), 0.0);

  // assemble D and M contributions into global force and stiffness
  if (artery_coupling_method_ ==
          ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::mortar_penalty and
      num_coupled_dofs_ > 0)
    sum_mortar_matrices_into_global_matrix(*sysmat, rhs);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::assemble(
    const int& ele1_gid, const int& ele2_gid, const double& integrated_diameter,
    std::vector<Core::LinAlg::SerialDenseVector> const& ele_rhs,
    std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const& ele_matrix,
    const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    const std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
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

  coupling_matrix_->fe_assemble(ele_matrix[0][0], lm_row_1, lm_row_1);
  coupling_matrix_->fe_assemble(ele_matrix[0][1], lm_row_1, lm_row_2);
  coupling_matrix_->fe_assemble(ele_matrix[1][0], lm_row_2, lm_row_1);
  coupling_matrix_->fe_assemble(ele_matrix[1][1], lm_row_2, lm_row_2);

  coupling_rhs_vector_->sum_into_global_values(
      ele_rhs[0].length(), lm_row_1.data(), ele_rhs[0].values());
  coupling_rhs_vector_->sum_into_global_values(
      ele_rhs[1].length(), lm_row_2.data(), ele_rhs[1].values());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::
    assemble_mortar_matrices_and_vector(const int& ele1_gid, const int& ele2_gid,
        const Core::LinAlg::SerialDenseMatrix& D_ele, const Core::LinAlg::SerialDenseMatrix& M_ele,
        const Core::LinAlg::SerialDenseVector& Kappa_ele) const
{
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

  mortar_matrix_d_->fe_assemble(D_ele, lm_row_1, lm_row_1);
  mortar_matrix_m_->fe_assemble(M_ele, lm_row_1, lm_row_2);
  mortar_kappa_inv_->sum_into_global_values(
      Kappa_ele.length(), lm_row_1.data(), Kappa_ele.values());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::
    sum_mortar_matrices_into_global_matrix(Core::LinAlg::BlockSparseMatrixBase& sysmat,
        const std::shared_ptr<Core::LinAlg::Vector<double>> rhs) const
{
  // invert kappa
  mortar_kappa_inv_->complete();

  // invert (pay attention to protruding elements)
  for (int i = 0; i < artery_dis_->dof_row_map()->num_my_elements(); ++i)
  {
    const int artery_dof_gid = artery_dis_->dof_row_map()->gid(i);
    const double kappa_value = (mortar_kappa_inv_
            ->get_ref_of_epetra_fevector())[0][mortar_kappa_inv_->get_map().lid(artery_dof_gid)];

    if (fabs(kappa_value) > 1.0e-10)
      mortar_kappa_inv_->replace_global_value(artery_dof_gid, 0, 1.0 / kappa_value);
    else
      mortar_kappa_inv_->replace_global_value(artery_dof_gid, 0, 0.0);
  }

  // complete
  mortar_matrix_d_->complete();
  mortar_matrix_m_->complete(*homogenized_dis_->dof_row_map(), *artery_dis_->dof_row_map());

  // get kappa matrix
  Core::LinAlg::SparseMatrix kappa_inverse(*new Core::LinAlg::Vector<double>(*mortar_kappa_inv_));
  kappa_inverse.complete();

  // kappa^{-1}*M
  const std::shared_ptr kappa_inv_M = Core::LinAlg::matrix_multiply(
      kappa_inverse, false, *mortar_matrix_m_, false, false, false, true);
  // kappa^{-1}*D
  const std::shared_ptr kappa_inv_D = Core::LinAlg::matrix_multiply(
      kappa_inverse, false, *mortar_matrix_d_, false, false, false, true);

  // D^T*kappa^{-1}*D
  const std::shared_ptr D_transpose_kappa_inv_D = Core::LinAlg::matrix_multiply(
      *mortar_matrix_d_, true, *kappa_inv_D, false, false, false, true);
  // D^T*kappa^{-1}*M
  const std::shared_ptr D_transpose_kappa_inv_M = Core::LinAlg::matrix_multiply(
      *mortar_matrix_d_, true, *kappa_inv_M, false, false, false, true);
  // M^T*kappa^{-1}*M
  const std::shared_ptr M_transpose_kappa_inv_M = Core::LinAlg::matrix_multiply(
      *mortar_matrix_m_, true, *kappa_inv_M, false, false, false, true);

  // add matrices
  Core::LinAlg::matrix_add(*M_transpose_kappa_inv_M, false,
      penalty_parameter_ * timefacrhs_homogenized_, sysmat.matrix(0, 0), 1.0);
  Core::LinAlg::matrix_add(*D_transpose_kappa_inv_D, false, penalty_parameter_ * timefacrhs_artery_,
      sysmat.matrix(1, 1), 1.0);
  Core::LinAlg::matrix_add(*D_transpose_kappa_inv_M, false,
      -penalty_parameter_ * timefacrhs_artery_, sysmat.matrix(1, 0), 1.0);
  Core::LinAlg::matrix_add(*D_transpose_kappa_inv_M, true,
      -penalty_parameter_ * timefacrhs_homogenized_, sysmat.matrix(0, 1), 1.0);

  // add vector
  const auto artery_contribution =
      std::make_shared<Core::LinAlg::Vector<double>>(*artery_dis_->dof_row_map());
  const auto homogenized_contribution =
      std::make_shared<Core::LinAlg::Vector<double>>(*homogenized_dis_->dof_row_map());

  // Note: all terms are negative since rhs
  // penalty parameter * D^T * kappa^{-1} * D * phi_np^artery
  D_transpose_kappa_inv_D->multiply(false, *phinp_art_, *artery_contribution);
  rhs->update(-penalty_parameter_ * timefacrhs_artery_,
      *global_extractor_->insert_vector(*artery_contribution, 1), 1.0);

  // -penalty parameter * D^T * kappa^{-1} * M * phi_np^homogenized
  D_transpose_kappa_inv_M->multiply(false, *phinp_homogenized_, *artery_contribution);
  rhs->update(penalty_parameter_ * timefacrhs_artery_,
      *global_extractor_->insert_vector(*artery_contribution, 1), 1.0);

  // penalty parameter * M^T * kappa^{-1} * M * phi_np^homogenized
  M_transpose_kappa_inv_M->multiply(false, *phinp_homogenized_, *homogenized_contribution);
  rhs->update(-penalty_parameter_ * timefacrhs_homogenized_,
      *global_extractor_->insert_vector(*homogenized_contribution, 0), 1.0);

  // -penalty parameter * M^T * kappa^{-1} * D * phi_np^artery
  // = -penalty parameter * (D^T * kappa^{-1} * M)^T * phi_np^artery
  D_transpose_kappa_inv_M->multiply(true, *phinp_art_, *homogenized_contribution);
  rhs->update(penalty_parameter_ * timefacrhs_homogenized_,
      *global_extractor_->insert_vector(*homogenized_contribution, 0), 1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<PoroPressureBased::PorofluidElastScatraArteryCouplingPairBase> PoroPressureBased::
    PorofluidElastScatraArteryCouplingNonConformingAlgorithm::create_new_artery_coupling_pair(
        std::vector<Core::Elements::Element const*> const& elements)
{
  const Core::FE::CellType dis_type_artery = elements[0]->shape();
  switch (dis_type_artery)
  {
    case Core::FE::CellType::line2:
    {
      const Core::FE::CellType dis_type_homogenized = elements[1]->shape();
      switch (dis_type_homogenized)
      {
        case Core::FE::CellType::quad4:
        {
          switch (Global::Problem::instance()->n_dim())
          {
            case 1:
              return std::make_shared<PorofluidElastScatraArteryCouplingPair<
                  Core::FE::CellType::line2, Core::FE::CellType::quad4, 1>>();
            case 2:
              return std::make_shared<PorofluidElastScatraArteryCouplingPair<
                  Core::FE::CellType::line2, Core::FE::CellType::quad4, 2>>();
            case 3:
              return std::make_shared<PorofluidElastScatraArteryCouplingPair<
                  Core::FE::CellType::line2, Core::FE::CellType::quad4, 3>>();
            default:
              FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
          }
        }
        case Core::FE::CellType::hex8:
        {
          switch (Global::Problem::instance()->n_dim())
          {
            case 1:
              return std::make_shared<PorofluidElastScatraArteryCouplingPair<
                  Core::FE::CellType::line2, Core::FE::CellType::hex8, 1>>();
            case 2:
              return std::make_shared<PorofluidElastScatraArteryCouplingPair<
                  Core::FE::CellType::line2, Core::FE::CellType::hex8, 2>>();
            case 3:
              return std::make_shared<PorofluidElastScatraArteryCouplingPair<
                  Core::FE::CellType::line2, Core::FE::CellType::hex8, 3>>();
            default:
              FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
          }
        }
        case Core::FE::CellType::tet4:
        {
          switch (Global::Problem::instance()->n_dim())
          {
            case 1:
              return std::make_shared<PorofluidElastScatraArteryCouplingPair<
                  Core::FE::CellType::line2, Core::FE::CellType::tet4, 1>>();
            case 2:
              return std::make_shared<PorofluidElastScatraArteryCouplingPair<
                  Core::FE::CellType::line2, Core::FE::CellType::tet4, 2>>();
            case 3:
              return std::make_shared<PorofluidElastScatraArteryCouplingPair<
                  Core::FE::CellType::line2, Core::FE::CellType::tet4, 3>>();
            default:
              FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
          }
        }
        case Core::FE::CellType::tet10:
        {
          switch (Global::Problem::instance()->n_dim())
          {
            case 1:
              return std::make_shared<PorofluidElastScatraArteryCouplingPair<
                  Core::FE::CellType::line2, Core::FE::CellType::tet10, 1>>();
            case 2:
              return std::make_shared<PorofluidElastScatraArteryCouplingPair<
                  Core::FE::CellType::line2, Core::FE::CellType::tet10, 2>>();
            case 3:
              return std::make_shared<PorofluidElastScatraArteryCouplingPair<
                  Core::FE::CellType::line2, Core::FE::CellType::tet10, 3>>();
            default:
              FOUR_C_THROW("Unsupported dimension {}.", Global::Problem::instance()->n_dim());
          }
        }
        default:
          FOUR_C_THROW(
              "Only quad4, hex8, tet4 and tet10 elements supported for homogenized elements.");
      }
    }
    default:
      FOUR_C_THROW("Only line2 elements supported for artery elements.");
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::
    setup_global_vector(const std::shared_ptr<Core::LinAlg::Vector<double>> global_vector,
        const std::shared_ptr<const Core::LinAlg::Vector<double>> homogenized_vector,
        const std::shared_ptr<const Core::LinAlg::Vector<double>> artery_vector)
{
  // zero out
  global_vector->put_scalar(0.0);
  // set up global vector
  global_extractor_->insert_vector(*homogenized_vector, 0, *global_vector);
  global_extractor_->insert_vector(*artery_vector, 1, *global_vector);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::
    extract_single_field_vectors(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> global_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& homogenized_vector,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& artery_vector)
{
  // process first field (homogenized)
  homogenized_vector = global_extractor_->extract_vector(*global_vector, 0);
  // process second field (artery)
  artery_vector = global_extractor_->extract_vector(*global_vector, 1);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::artery_dof_row_map()
    const
{
  return global_extractor_->map(1);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::dof_row_map() const
{
  return fullmap_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::
    set_solution_vectors(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_homogenized,
        const std::shared_ptr<const Core::LinAlg::Vector<double>> phin_homogenized,
        const std::shared_ptr<const Core::LinAlg::Vector<double>> phinp_artery)
{
  phinp_homogenized_ = phinp_homogenized;
  if (phin_homogenized != nullptr) phin_homogenized_ = phin_homogenized;
  phinp_art_ = phinp_artery;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> PoroPressureBased::
    PorofluidElastScatraArteryCouplingNonConformingAlgorithm::blood_vessel_volume_fraction()
{
  FOUR_C_THROW("blood_vessel_volume_fraction not implemented in for non-conforming coupling.");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::
    print_coupling_method() const
{
  std::string coupling_method;
  if (artery_coupling_method_ ==
      ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::mortar_penalty)
    coupling_method = "Mortar Penalty";
  else if (artery_coupling_method_ ==
           ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::gauss_point_to_segment)
    coupling_method = "Gauss-Point-To-Segment";
  else
    FOUR_C_THROW("unknown coupling method");

  std::cout << "<   Coupling-Method : " << std::left << std::setw(22) << coupling_method
            << "       >" << '\n';
  std::cout << "<   Penalty parameter : " << std::left << std::setw(6) << penalty_parameter_
            << "                     >" << '\n';
  if (evaluate_in_ref_config_)
    std::cout << "<   Moving arteries : No                           >" << '\n';
  else
    std::cout << "<   Moving arteries : Yes                          >" << '\n';
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::
    fill_function_and_scale_vectors()
{
  scale_vector_.resize(2);
  function_vector_.resize(2);

  // get the actual coupled DOFs
  // 1) 1D artery discretization
  const auto reaction_terms_params = coupling_params_.sublist("reaction_terms");
  int value;
  std::istringstream scale_artery_stream(
      Teuchos::getNumericStringParameter(reaction_terms_params, "artery_scaling"));
  while (scale_artery_stream >> value) scale_vector_[0].push_back(value);

  std::istringstream function_artery_stream(
      Teuchos::getNumericStringParameter(reaction_terms_params, "artery_function_ids"));
  while (function_artery_stream >> value) function_vector_[0].push_back(value);

  // 2) 2D, 3D homogenized field discretization
  std::istringstream scale_homogenized_stream(
      Teuchos::getNumericStringParameter(reaction_terms_params, "homogenized_scaling"));
  while (scale_homogenized_stream >> value) scale_vector_[1].push_back(value);

  std::istringstream function_homogenized_stream(
      Teuchos::getNumericStringParameter(reaction_terms_params, "homogenized_function_ids"));
  while (function_homogenized_stream >> value) function_vector_[1].push_back(value);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::set_time_fac_rhs()
{
  // set the right-hand side factor
  if (homogenized_dis_->name() == "porofluid")
  {
    const Discret::Elements::PoroFluidMultiPhaseEleParameter* ele_params =
        Discret::Elements::PoroFluidMultiPhaseEleParameter::instance("porofluid");
    timefacrhs_artery_ = 1.0;
    timefacrhs_homogenized_ = ele_params->time_fac_rhs();
  }
  else if (homogenized_dis_->name() == "scatra")
  {
    const Discret::Elements::ScaTraEleParameterTimInt* ele_params =
        Discret::Elements::ScaTraEleParameterTimInt::instance("scatra");
    timefacrhs_artery_ = ele_params->time_fac_rhs();
    timefacrhs_homogenized_ = ele_params->time_fac_rhs();
  }
  else
  {
    FOUR_C_THROW(
        "Only porofluid and scatra-discretizations are supported for non-conforming coupling.");
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PorofluidElastScatraArteryCouplingNonConformingAlgorithm::
    set_nearby_ele_pairs(const std::map<int, std::set<int>>* nearby_ele_pairs)
{
  nearby_ele_pairs_ = *nearby_ele_pairs;
}

FOUR_C_NAMESPACE_CLOSE