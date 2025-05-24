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
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mat_cnst_1d_art.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_defines.hpp"
#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_pair.hpp"
#include "4C_porofluid_pressure_based_utils.hpp"

#include <Epetra_FEVector.h>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::PoroMultiPhaseScaTraArtCouplLineBased(
    std::shared_ptr<Core::FE::Discretization> arterydis,
    std::shared_ptr<Core::FE::Discretization> contdis, const Teuchos::ParameterList& couplingparams,
    const std::string& condname, const std::string& artcoupleddofname,
    const std::string& contcoupleddofname)
    : PoroMultiPhaseScaTraArtCouplNonConforming(
          arterydis, contdis, couplingparams, condname, artcoupleddofname, contcoupleddofname),
      maxnumsegperartele_(Global::Problem::instance()
              ->poro_fluid_multi_phase_dynamic_params()
              .sublist("ARTERY COUPLING")
              .get<int>("MAXNUMSEGPERARTELE"))
{
  // user info
  if (myrank_ == 0)
  {
    std::cout << "<                                                  >" << std::endl;
    print_out_coupling_method();
    std::cout << "<                                                  >" << std::endl;
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
    std::cout << "\n";
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::setup()
{
  // call base class
  PoroPressureBased::PoroMultiPhaseScaTraArtCouplNonConforming::setup();

  // pre-evaluate the pairs
  pre_evaluate_coupling_pairs();

  // create the GID to segment vector
  create_gid_to_segment_vector();

  // fill length of artery elements that is not influenced if the underlying
  // 2D/3D mesh moves (basically protruding artery elements or segments)
  fill_unaffected_artery_length();

  // fill unaffected integrated diam (basically protruding artery elements or segments)
  if (contdis_->name() == "porofluid" && has_varying_diam_) fill_unaffected_integrated_diam();

  // calculate blood vessel volume fraction (only porofluid needs to do this)
  if (contdis_->name() == "porofluid" && couplingparams_.get<bool>("OUTPUT_BLOODVESSELVOLFRAC"))
    calculate_blood_vessel_volume_fraction();

  // print out summary of pairs
  if (contdis_->name() == "porofluid" && couplingparams_.get<bool>("PRINT_OUT_SUMMARY_PAIRS"))
    output_summary();

  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::setup_system(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_cont,
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_art,
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_cont,
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_art,
    std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_cont,
    std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_art)
{
  // copy vector
  std::shared_ptr<Core::LinAlg::Vector<double>> rhs_art_with_collapsed =
      std::make_shared<Core::LinAlg::Vector<double>>(*rhs_art);
  std::shared_ptr<Core::LinAlg::Map> dbcmap_art_with_collapsed =
      get_additional_dbc_for_collapsed_eles(*dbcmap_art, *rhs_art_with_collapsed);

  // call base class
  PoroPressureBased::PoroMultiPhaseScaTraArtCouplNonConforming::setup_system(*sysmat, rhs,
      *sysmat_cont, *sysmat_art, rhs_cont, rhs_art_with_collapsed, *dbcmap_cont,
      *dbcmap_art->cond_map(), *dbcmap_art_with_collapsed);
}

std::shared_ptr<Core::LinAlg::Map>
PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::get_additional_dbc_for_collapsed_eles(
    const Core::LinAlg::MapExtractor& dbcmap_art,
    Core::LinAlg::Vector<double>& rhs_art_with_collapsed)
{
  // Zero flux is automatically assumed for nodes which border a collapsed element
  // since the respective collapsed element is not evaluated
  // nodes which only border collapsed elements are not evaluated at all, hence, leading to zero
  // rows in global stiffness matrix and to singularity of this matrix
  // here we identify these nodes and set a zero dirichlet boundary condition on them
  // Note that this procedure is equivalent to taking collapsed elements out of the simulation
  // entirely

  int artelematerial = contdis_->name() == "scatra" ? 1 : 0;
  std::vector<int> mydirichdofs;

  const int numrownodes = arterydis_->num_my_row_nodes();
  const Core::LinAlg::Map* dofrowmap = arterydis_->dof_row_map();

  for (int inode = 0; inode < numrownodes; ++inode)
  {
    Core::Nodes::Node* actnode = arterydis_->l_row_node(inode);
    Core::Elements::Element** eles = actnode->elements();
    bool all_eles_collapsed = true;
    for (int iele = 0; iele < (actnode->num_element()); iele++)
    {
      Core::Elements::Element* actele = eles[iele];
      const auto& arterymat =
          std::dynamic_pointer_cast<const Mat::Cnst1dArt>(actele->material(artelematerial));
      if (not arterymat->is_collapsed())
      {
        all_eles_collapsed = false;
        break;
      }
    }

    // all elements of this node are collapsed
    if (all_eles_collapsed)
    {
      // 1) insert all dofs of this node into dirichlet dof vector
      std::vector<int> dofs = arterydis_->dof(0, actnode);
      mydirichdofs.insert(mydirichdofs.end(), dofs.begin(), dofs.end());
      // 2) insert the negative value of all dofs of this node into the rhs, with the employed
      // incremental form this will force the value to zero
      for (const auto& mydof : dofs)
        rhs_art_with_collapsed.replace_global_value(
            mydof, 0, -(*phinp_art_)[dofrowmap->lid(mydof)]);
    }
  }

  // build map
  int nummydirichvals = mydirichdofs.size();
  std::shared_ptr<Core::LinAlg::Map> dirichmap = std::make_shared<Core::LinAlg::Map>(
      -1, nummydirichvals, mydirichdofs.data(), 0, arterydis_->get_comm());

  // build vector of maps
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> condmaps;
  condmaps.push_back(dirichmap);
  condmaps.push_back(dbcmap_art.cond_map());

  // combined map
  std::shared_ptr<Core::LinAlg::Map> condmerged =
      Core::LinAlg::MultiMapExtractor::merge_maps(condmaps);

  return condmerged;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::pre_evaluate_coupling_pairs()
{
  // pre-evaluate
  for (unsigned i = 0; i < coupl_elepairs_.size(); i++) coupl_elepairs_[i]->pre_evaluate(nullptr);

  // delete the inactive and duplicate pairs
  std::vector<std::shared_ptr<PoroPressureBased::PoroMultiPhaseScatraArteryCouplingPairBase>>
      active_coupl_elepairs;
  for (unsigned i = 0; i < coupl_elepairs_.size(); i++)
  {
    const int contelegid = coupl_elepairs_[i]->ele2_gid();
    Core::Elements::Element* contele = contdis_->g_element(contelegid);

    if (coupl_elepairs_[i]->is_active() &&
        !is_duplicate_segment(active_coupl_elepairs, *coupl_elepairs_[i]) &&
        contele->owner() == myrank_)
      active_coupl_elepairs.push_back(coupl_elepairs_[i]);
  }

  // the following case takes care of the special occurrence where the 1D element lies exactly in
  // between two 2D/3D-elements which are owned by different processors

  // fill the GID-to-segment vector
  std::map<int, std::vector<double>> gid_to_seglength;
  fill_gid_to_segment_vector(active_coupl_elepairs, gid_to_seglength);

  // dummy map to collect duplicates in form [ele2gid, eta_a, eta_b, ... ];
  std::map<int, std::vector<double>> duplicates;

  // loop over all artery elements
  for (int i = 0; i < arterydis_->element_col_map()->num_my_elements(); ++i)
  {
    const int artelegid = arterydis_->element_col_map()->gid(i);
    if (gid_to_seglength[artelegid].size() > 0)  // check if element projects
    {
      // compare all segment with each other if it might be identical
      for (int iseg = 0; iseg < (int)(gid_to_seglength[artelegid].size() / 2); iseg++)
      {
        const double eta_a = gid_to_seglength[artelegid][2 * iseg];
        const double eta_b = gid_to_seglength[artelegid][2 * iseg + 1];
        for (int jseg = iseg + 1; jseg < (int)(gid_to_seglength[artelegid].size() / 2); jseg++)
        {
          const double eta_a_jseg = gid_to_seglength[artelegid][2 * jseg];
          const double eta_b_jseg = gid_to_seglength[artelegid][2 * jseg + 1];
          // identical segment found
          if (fabs(eta_a - eta_a_jseg) < XIETATOL && fabs(eta_b - eta_b_jseg) < XIETATOL)
          {
            // we need this to get the ele2gid
            int id = -1;
            if (is_identical_segment(active_coupl_elepairs, artelegid, eta_a, eta_b, id))
            {
              const int ele2gid = active_coupl_elepairs[id]->ele2_gid();
              duplicates[artelegid].push_back((double)(ele2gid));
              duplicates[artelegid].push_back(eta_a);
              duplicates[artelegid].push_back(eta_b);
            }
          }
        }
      }
    }
  }

  // communicate the dummy map to all procs.
  std::vector<int> allproc(Core::Communication::num_mpi_ranks(get_comm()));
  for (int i = 0; i < Core::Communication::num_mpi_ranks(get_comm()); ++i) allproc[i] = i;
  Core::LinAlg::gather<double>(
      duplicates, duplicates, (int)allproc.size(), allproc.data(), get_comm());

  // loop over duplicates and delete one duplicate (the one where the 2D/3D element has the larger
  // id)
  std::map<int, std::vector<double>>::iterator it;
  for (it = duplicates.begin(); it != duplicates.end(); it++)
  {
    const int artelegid = it->first;
    std::vector<double> myduplicates = it->second;
    // should always be a multiple of six because we should always find exactly two/four, etc.
    // duplicates
    if (myduplicates.size() % 6 != 0)
      FOUR_C_THROW("duplicate vector has size {}, should be multiple of six", myduplicates.size());
    // compare the possible duplicates
    for (int idupl = 0; idupl < (int)(myduplicates.size() / 3); idupl++)
    {
      const double eta_a = myduplicates[3 * idupl + 1];
      const double eta_b = myduplicates[3 * idupl + 2];
      for (int jdupl = idupl + 1; jdupl < (int)(myduplicates.size() / 3); jdupl++)
      {
        const double eta_a_jdupl = myduplicates[3 * jdupl + 1];
        const double eta_b_jdupl = myduplicates[3 * jdupl + 2];
        // duplicate found
        if (fabs(eta_a - eta_a_jdupl) < XIETATOL && fabs(eta_b - eta_b_jdupl) < XIETATOL)
        {
          const int ele_i = (int)myduplicates[3 * idupl];
          const int ele_j = (int)myduplicates[3 * jdupl];
          const int ele_to_be_erased = std::max(ele_i, ele_j);
          int id = -1;
          // delete the duplicate with the larger ele2gid
          if (is_identical_segment(active_coupl_elepairs, artelegid, eta_a, eta_b, id))
          {
            if (active_coupl_elepairs[id]->ele2_gid() == ele_to_be_erased)
            {
              active_coupl_elepairs.erase(active_coupl_elepairs.begin() + id);
            }
          }
        }
      }
    }
  }

  // overwrite the coupling pairs
  coupl_elepairs_ = active_coupl_elepairs;

  // output
  int total_numactive_pairs = 0;
  int numactive_pairs = static_cast<int>(coupl_elepairs_.size());
  Core::Communication::sum_all(&numactive_pairs, &total_numactive_pairs, 1, get_comm());
  if (myrank_ == 0)
  {
    std::cout << "Only " << total_numactive_pairs
              << " Artery-to-PoroMultiphaseScatra coupling pairs (segments) are active"
              << std::endl;
  }
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::fill_unaffected_artery_length()
{
  // no need to do this
  if (porofluidprob_)
  {
    for (int i = 0; i < arterydis_->element_col_map()->num_my_elements(); ++i)
    {
      const int artelegid = arterydis_->element_col_map()->gid(i);
      Core::Elements::Element* artele = arterydis_->g_element(artelegid);

      // TODO: this will not work for higher order artery eles
      const double initlength = PoroPressureBased::get_max_nodal_distance(artele, *arterydis_);
      const int numseg = (int)(gid_to_segment_[artelegid].size() / 2);
      gid_to_seglength_[artelegid].resize(numseg);
      for (int iseg = 0; iseg < numseg; iseg++)
      {
        const double etaA = gid_to_segment_[artelegid][2 * iseg];
        const double etaB = gid_to_segment_[artelegid][2 * iseg + 1];
        gid_to_seglength_[artelegid][iseg] = initlength * (etaB - etaA) / 2.0;

        int id = -1;
        // return also id -> index in coupl_elepairs_ of this segment
        // and set iseg as segment id of the coupling pairs
        if (is_identical_segment(coupl_elepairs_, artelegid, etaA, etaB, id))
          coupl_elepairs_[id]->set_segment_id(iseg);
      }
    }

    return;
  }

  // the unaffected length is the length of 1D elements not changed through deformation,
  // basically if these elements protrude.
  // for each element this length is computed as: ele_length - sum_segments seg_length
  // if the above quantity is bigger than zero, a 1D element protrudes

  // initialize the unaffected and current lengths
  unaffected_seg_lengths_artery_ =
      std::make_shared<Epetra_FEVector>(arterydis_->dof_row_map(1)->get_epetra_block_map(), true);
  current_seg_lengths_artery_ =
      std::make_shared<Epetra_FEVector>(arterydis_->dof_row_map(1)->get_epetra_block_map());

  // set segment ID on coupling pairs and fill the unaffected artery length
  for (int iele = 0; iele < arterydis_->element_col_map()->num_my_elements(); ++iele)
  {
    const int artelegid = arterydis_->element_col_map()->gid(iele);
    Core::Elements::Element* thisele = arterydis_->g_element(artelegid);

    // TODO: this will not work for higher order artery eles
    const double initlength = PoroPressureBased::get_max_nodal_distance(thisele, *arterydis_);

    std::vector<double> segmentboundaries = gid_to_segment_[artelegid];
    for (unsigned int iseg = 0; iseg < segmentboundaries.size() / 2; iseg++)
    {
      int id = -1;
      // get EtaA and etaB and calculate initial length
      const double etaA = segmentboundaries[iseg * 2];
      const double etaB = segmentboundaries[iseg * 2 + 1];
      const double seglength = initlength * (etaB - etaA) / 2.0;

      // since we use an FE-vector
      if (thisele->owner() == myrank_)
      {
        // build the location array
        std::vector<int> seglengthdofs = arterydis_->dof(1, thisele);
        unaffected_seg_lengths_artery_->SumIntoGlobalValues(1, &seglengthdofs[iseg], &seglength);
      }

      // return also id -> index in coupl_elepairs_ of this segment
      // and set iseg as segment id of the coupling pairs
      if (is_identical_segment(coupl_elepairs_, artelegid, etaA, etaB, id))
        coupl_elepairs_[id]->set_segment_id(iseg);
    }
  }

  if (unaffected_seg_lengths_artery_->GlobalAssemble(Add, false) != 0)
    FOUR_C_THROW("GlobalAssemble of unaffected_seg_lengths_artery_ failed");

  // subtract the segment lengths only if we evaluate in current configuration
  if (!evaluate_in_ref_config_)
  {
    for (unsigned i = 0; i < coupl_elepairs_.size(); i++)
    {
      // get the initial lengths
      double init_segment_length = coupl_elepairs_[i]->apply_mesh_movement(true, contdis_);
      init_segment_length *= -1.0;

      const int artelegid = coupl_elepairs_[i]->ele1_gid();
      Core::Elements::Element* artele = arterydis_->g_element(artelegid);

      std::vector<int> seglengthdofs = arterydis_->dof(1, artele);
      const int segid = coupl_elepairs_[i]->get_segment_id();

      unaffected_seg_lengths_artery_->SumIntoGlobalValues(
          1, &seglengthdofs[segid], &(init_segment_length));
    }
    if (unaffected_seg_lengths_artery_->GlobalAssemble(Add, false) != 0)
      FOUR_C_THROW("GlobalAssemble of unaffected_seg_lengths_artery_ failed");
  }
  // the current length is simply the unaffected length
  else
    current_seg_lengths_artery_->Update(1.0, *unaffected_seg_lengths_artery_, 0.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::fill_unaffected_integrated_diam()
{
  Epetra_FEVector unaffected_diams_artery_row(
      arterydis_->element_row_map()->get_epetra_block_map(), true);

  for (int i = 0; i < arterydis_->element_row_map()->num_my_elements(); ++i)
  {
    const int artelegid = arterydis_->element_row_map()->gid(i);
    Core::Elements::Element* artele = arterydis_->g_element(artelegid);

    // TODO: this will not work for higher order artery eles
    const double initlength = PoroPressureBased::get_max_nodal_distance(artele, *arterydis_);

    // first add all contributions int unaffected_diams_artery_row-vector
    std::shared_ptr<Mat::Cnst1dArt> arterymat =
        std::dynamic_pointer_cast<Mat::Cnst1dArt>(artele->material());
    if (arterymat == nullptr) FOUR_C_THROW("cast to artery material failed");
    const double length_diam = initlength * arterymat->diam();
    unaffected_diams_artery_row.SumIntoGlobalValues(1, &artelegid, &length_diam);
  }
  // then subtract the coupling pairs to detect protruding parts
  for (unsigned i = 0; i < coupl_elepairs_.size(); i++)
  {
    // get the initial lengths
    double init_segment_length = coupl_elepairs_[i]->apply_mesh_movement(true, contdis_);
    init_segment_length *= -1.0;

    const int artelegid = coupl_elepairs_[i]->ele1_gid();
    Core::Elements::Element* artele = arterydis_->g_element(artelegid);

    std::shared_ptr<Mat::Cnst1dArt> arterymat =
        std::dynamic_pointer_cast<Mat::Cnst1dArt>(artele->material());
    if (arterymat == nullptr) FOUR_C_THROW("cast to artery material failed");
    const double length_diam = init_segment_length * arterymat->diam();
    unaffected_diams_artery_row.SumIntoGlobalValues(1, &artelegid, &length_diam);
  }

  // global assembly and export
  if (unaffected_diams_artery_row.GlobalAssemble(Add, false) != 0)
    FOUR_C_THROW("GlobalAssemble of unaffected_seg_lengths_artery_ failed");
  Core::LinAlg::export_to(Core::LinAlg::Vector<double>(unaffected_diams_artery_row),
      *unaffected_integrated_diams_artery_col_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::
    calculate_blood_vessel_volume_fraction()
{
  bloodvesselvolfrac_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*contdis_->element_row_map(), true);

  double totalvolblood = 0.0;
  // evaluate all pairs
  for (unsigned i = 0; i < coupl_elepairs_.size(); i++)
  {
    const int artelegid = coupl_elepairs_[i]->ele1_gid();
    const int contelegid = coupl_elepairs_[i]->ele2_gid();

    Core::Elements::Element* artele = arterydis_->g_element(artelegid);

    std::shared_ptr<Mat::Cnst1dArt> arterymat =
        std::dynamic_pointer_cast<Mat::Cnst1dArt>(artele->material());
    if (arterymat == nullptr) FOUR_C_THROW("cast to artery material failed");

    // TODO: this will not work for higher order artery eles
    const double etaA = coupl_elepairs_[i]->etadata();
    const double etaB = coupl_elepairs_[i]->eta_b();
    const double length = PoroPressureBased::get_max_nodal_distance(artele, *arterydis_);

    const double vol_cont = coupl_elepairs_[i]->calculate_vol_2d_3d();
    const double vol_art =
        (etaB - etaA) / 2.0 * length * arterymat->diam() * arterymat->diam() * M_PI / 4.0;

    totalvolblood += vol_art;

    const double volfrac = vol_art / vol_cont;

    // note: this works since we 2D/3D continuous element of each pair is always owned by this proc.
    int err = bloodvesselvolfrac_->sum_into_global_values(1, &volfrac, &contelegid);
    if (err) FOUR_C_THROW("SumIntoGlobalValues failed!");
  }

  // user output
  double vol_sumall = 0.0;
  Core::Communication::sum_all(&totalvolblood, &vol_sumall, 1, get_comm());
  if (myrank_ == 0)
  {
    std::cout << "\n<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
    std::cout << "<    Calculating blood vessel volume fraction      >" << std::endl;
    std::cout << "<    total volume blood:    " << std::setw(5) << vol_sumall
              << "                 >" << std::endl;
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::set_varying_diam_flag()
{
  PoroMultiPhaseScaTraArtCouplNonConforming::set_varying_diam_flag();

  // set up the required vectors
  if (has_varying_diam_)
  {
    integrated_diams_artery_row_ = std::make_shared<Epetra_FEVector>(
        arterydis_->element_row_map()->get_epetra_block_map(), true);
    unaffected_integrated_diams_artery_col_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*arterydis_->element_col_map(), true);
    integrated_diams_artery_col_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*arterydis_->element_col_map(), true);
    ele_diams_artery_col_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*arterydis_->element_col_map(), true);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::create_gid_to_segment_vector()
{
  // fill the GID-to-segment vector
  fill_gid_to_segment_vector(coupl_elepairs_, gid_to_segment_);

  // sort and take care of special cases
  for (int i = 0; i < arterydis_->element_col_map()->num_my_elements(); ++i)
  {
    const int artelegid = arterydis_->element_col_map()->gid(i);
    if (gid_to_segment_[artelegid].size() > 0)  // check if element projects
    {
      // sort
      std::sort(gid_to_segment_[artelegid].begin(), gid_to_segment_[artelegid].end());
      const int end = gid_to_segment_[artelegid].size();
      const double valueAtEnd = gid_to_segment_[artelegid][end - 1];
      // end of element lies outside domain
      if (fabs(valueAtEnd - 1.0) > XIETATOL)
      {
        gid_to_segment_[artelegid].push_back(valueAtEnd);
        gid_to_segment_[artelegid].push_back(1.0);
      }
      const double valueAtBegin = gid_to_segment_[artelegid][0];
      // beginning of element lies outside domain
      if (fabs(valueAtBegin + 1.0) > XIETATOL)
      {
        gid_to_segment_[artelegid].insert(gid_to_segment_[artelegid].begin(), valueAtBegin);
        gid_to_segment_[artelegid].insert(gid_to_segment_[artelegid].begin(), -1.0);
      }
    }
    // this element does not project
    else
    {
      gid_to_segment_[artelegid].push_back(-1.0);
      gid_to_segment_[artelegid].push_back(1.0);
    }
  }

  // safety checks
  for (int i = 0; i < arterydis_->element_col_map()->num_my_elements(); ++i)
  {
    // 1) check if artery element has more than MAXNUMSEGPERARTELE segments
    const int artelegid = arterydis_->element_col_map()->gid(i);
    if ((int)gid_to_segment_[artelegid].size() > 2 * maxnumsegperartele_)
    {
      FOUR_C_THROW(
          "Artery element {} has {} segments, which is more than the maximum allowed number of {} "
          "segments per artery element, increase MAXNUMSEGPERARTELE",
          artelegid, (int)(gid_to_segment_[artelegid].size() / 2), maxnumsegperartele_);
    }
    // 2) check if segment has been overlooked
    for (int iseg = 0; iseg < (int)(gid_to_segment_[artelegid].size() / 2) - 1; iseg++)
    {
      if (fabs(gid_to_segment_[artelegid][2 * iseg + 1] -
               gid_to_segment_[artelegid][2 * iseg + 2]) > XIETATOL)
      {
        std::cout << "Problem with segments of artery-element " << artelegid << ":" << std::endl;
        for (int jseg = 0; jseg < (int)(gid_to_segment_[artelegid].size() / 2); jseg++)
        {
          std::cout << "[" << gid_to_segment_[artelegid][2 * jseg] << ", "
                    << gid_to_segment_[artelegid][2 * jseg + 1] << "]" << std::endl;
        }
        FOUR_C_THROW("artery element {} has probably not found all possible segments", artelegid);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::fill_gid_to_segment_vector(
    const std::vector<std::shared_ptr<
        PoroPressureBased::PoroMultiPhaseScatraArteryCouplingPairBase>>& coupl_elepairs,
    std::map<int, std::vector<double>>& gid_to_seglength)
{
  // fill the GID-to-segment vector
  for (unsigned i = 0; i < coupl_elepairs.size(); i++)
  {
    const int artelegid = coupl_elepairs[i]->ele1_gid();
    const int contelegid = coupl_elepairs[i]->ele2_gid();

    const Core::Elements::Element* contele = contdis_->g_element(contelegid);

    const double etaA = coupl_elepairs[i]->etadata();
    const double etaB = coupl_elepairs[i]->eta_b();

    if (contele->owner() == myrank_)
    {
      gid_to_seglength[artelegid].push_back(etaA);
      gid_to_seglength[artelegid].push_back(etaB);
    }
    else
      FOUR_C_THROW(
          "Something went wrong here, pair in coupling ele pairs where continuous-discretization "
          "element is not owned by this proc.");
  }

  // communicate it to all procs.
  std::vector<int> allproc(Core::Communication::num_mpi_ranks(get_comm()));
  for (int i = 0; i < Core::Communication::num_mpi_ranks(get_comm()); ++i) allproc[i] = i;
  Core::LinAlg::gather<double>(
      gid_to_seglength, gid_to_seglength, (int)allproc.size(), allproc.data(), get_comm());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::
    fe_assemble_ele_force_stiff_into_system_vector_matrix(const int& ele1gid, const int& ele2gid,
        const double& integrated_diam, std::vector<Core::LinAlg::SerialDenseVector> const& elevec,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const& elemat,
        std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
  // call base class
  PoroPressureBased::PoroMultiPhaseScaTraArtCouplNonConforming::
      fe_assemble_ele_force_stiff_into_system_vector_matrix(
          ele1gid, ele2gid, integrated_diam, elevec, elemat, sysmat, rhs);

  // also assemble the diameter if necessary
  if (contdis_->name() == "porofluid" && has_varying_diam_)
    integrated_diams_artery_row_->SumIntoGlobalValues(1, &ele1gid, &(integrated_diam));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::set_artery_diam_in_material()
{
  // assemble
  if (integrated_diams_artery_row_->GlobalAssemble(Add, false) != 0)
    FOUR_C_THROW("GlobalAssemble of integrated_integrated_diams_artery_row_ failed");

  // export to column format
  Core::LinAlg::export_to(
      Core::LinAlg::Vector<double>(*integrated_diams_artery_row_), *integrated_diams_artery_col_);

  // fill the vector collecting the element diameter
  fill_artery_ele_diam_col();

  // find the free-hanging elements which will be deleted
  std::vector<int> eles_to_be_deleted;
  if (delete_free_hanging_eles_) find_free_hanging_1d_elements(eles_to_be_deleted);

  // set the diameter in material
  for (int i = 0; i < arterydis_->num_my_col_elements(); ++i)
  {
    // pointer to current element
    Core::Elements::Element* actele = arterydis_->l_col_element(i);
    const int elegid = actele->id();

    double diam = (*ele_diams_artery_col_)[i];

    // set to zero for free-hanging elements
    if (delete_free_hanging_eles_)
    {
      if (std::find(eles_to_be_deleted.begin(), eles_to_be_deleted.end(), elegid) !=
          eles_to_be_deleted.end())
        diam = 0.0;
    }

    // get the artery-material
    std::shared_ptr<Mat::Cnst1dArt> arterymat =
        std::dynamic_pointer_cast<Mat::Cnst1dArt>(actele->material());
    if (arterymat == nullptr) FOUR_C_THROW("cast to artery material failed");

    // set to zero if collapsed
    if (diam < arterymat->collapse_threshold())
    {
      // Collapse happens for first time --> inform user
      if (arterymat->diam() >= arterymat->collapse_threshold() && actele->owner() == myrank_)
        std::cout << ">>>>>> Artery element " << actele->id() << " just collapsed <<<<<<"
                  << std::endl;
      arterymat->set_diam(0.0);
    }
    else  // otherwise set to calculated diameter
      arterymat->set_diam(diam);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::reset_integrated_diam_to_zero()
{
  integrated_diams_artery_row_->PutScalar(0.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::fill_artery_ele_diam_col()
{
  // reset
  ele_diams_artery_col_->put_scalar(0.0);
  // set the diameter in the vector
  for (int i = 0; i < arterydis_->num_my_col_elements(); ++i)
  {
    // pointer to current element
    Core::Elements::Element* actele = arterydis_->l_col_element(i);
    const int elegid = actele->id();

    const std::vector<double> seglengths = get_ele_segment_lengths(elegid);
    const double curr_ele_length = std::accumulate(seglengths.begin(), seglengths.end(), 0.0);
    // diam = int(diam)/length_element
    // also add the unaffected diameter --> diameter of artery elements which protrude
    const double diam =
        ((*integrated_diams_artery_col_)[i] + (*unaffected_integrated_diams_artery_col_)[i]) /
        curr_ele_length;

    ele_diams_artery_col_->replace_local_value(i, 0, diam);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::find_free_hanging_1d_elements(
    std::vector<int>& eles_to_be_deleted)
{
  // user info
  if (myrank_ == 0)
  {
    std::cout << "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<"
                 "<<<<<<<<<<<<<<<<<<<<<<<"
              << std::endl;
    std::cout << ">>>>>>                               Find free-hanging 1D elements               "
                 "               <<<<<<"
              << std::endl;
  }
  // get fully-overlapping discretization
  std::shared_ptr<Core::FE::Discretization> artconncompdis =
      PoroPressureBased::create_fully_overlapping_artery_discretization(
          *arterydis_, "conn_comp_dis", true);

  // vector to mark visited nodes
  std::shared_ptr<Core::LinAlg::Vector<int>> visited =
      std::make_shared<Core::LinAlg::Vector<int>>(*artconncompdis->node_col_map(), true);

  // get fully-overlapping diams vector
  std::shared_ptr<Core::LinAlg::Vector<double>> ele_diams_artery_full_overlap =
      std::make_shared<Core::LinAlg::Vector<double>>(*artconncompdis->element_col_map(), true);
  Core::LinAlg::Vector<double> ele_diams_artery_row(*arterydis_->element_row_map(), true);
  Core::LinAlg::export_to(*ele_diams_artery_col_, ele_diams_artery_row);
  Core::LinAlg::export_to(ele_diams_artery_row, *ele_diams_artery_full_overlap);

  // vector of connected components of 1D graph
  std::vector<std::vector<int>> connected_components;
  int num_conn_components = 0;
  int num_conn_components_wo_single_nodes = 0;

  // loop over fully-overlapping discretization
  for (int i = 0; i < artconncompdis->num_my_col_nodes(); ++i)
  {
    // pointer to current node
    Core::Nodes::Node* actnode = artconncompdis->l_col_node(i);
    // if not visited start a new connected component
    if ((*visited)[actnode->lid()] == 0)
    {
      connected_components.push_back(std::vector<int>());
      // recursive call to depth-first search
      depth_first_search_util(actnode, visited, artconncompdis, ele_diams_artery_full_overlap,
          connected_components[num_conn_components]);
      // single nodes are not of interest as they are detected (and taken out of simulation) anyways
      if (connected_components[num_conn_components].size() > 1)
        num_conn_components_wo_single_nodes++;

      num_conn_components++;
    }
  }

  // user info
  if (myrank_ == 0 && num_conn_components_wo_single_nodes > 1)
  {
    std::cout << "found " << num_conn_components_wo_single_nodes << " connected components"
              << std::endl;
  }

  const auto dirichlet_node_ids = Core::Conditions::find_conditioned_node_ids(
      *artconncompdis, "Dirichlet", Core::Conditions::LookFor::locally_owned_and_ghosted);
  // loop over all connected components
  for (unsigned int i = 0; i < connected_components.size(); ++i)
  {
    int conn_comp_size = connected_components[i].size();
    // single nodes are not of interest as they are detected anyways
    if (conn_comp_size > 1)
    {
      // user info
      if (myrank_ == 0)
        std::cout << "connected_component with ID " << i << " of size: " << conn_comp_size
                  << std::endl;

      // check if any of the nodes of this connected component has a Dirichlet BC
      bool has_dirichlet = false;
      for (int j = 0; j < conn_comp_size; ++j)
      {
        has_dirichlet =
            dirichlet_node_ids.contains(artconncompdis->g_node((connected_components[i])[j])->id());
        if (has_dirichlet)
        {
          if (myrank_ == 0)
            std::cout << "   ---> has at least one Dirichlet boundary condition" << std::endl;
          break;
        }
      }

      // if no node of this connected component has a DBC or if it is smaller than the
      // user-specified threshold, all its elements are taken out
      if (!has_dirichlet or conn_comp_size < (int)(delete_free_hanging_eles_threshold_ *
                                                   artconncompdis->num_global_nodes()))
      {
        // get the elements which have to be deleted
        for (int j = 0; j < conn_comp_size; ++j)
        {
          Core::Nodes::Node* mynode = artconncompdis->g_node((connected_components[i])[j]);
          Core::Elements::Element** myeles = mynode->elements();
          for (int i_element = 0; i_element < mynode->num_element(); i_element++)
            eles_to_be_deleted.push_back(myeles[i_element]->id());
        }
        // user info
        if (myrank_ == 0)
        {
          if (!has_dirichlet)
            std::cout
                << "   ---> has no Dirichlet boundary condition --> its elements will be taken out"
                << std::endl;
          if (has_dirichlet and conn_comp_size < (int)(delete_free_hanging_eles_threshold_ *
                                                       artconncompdis->num_global_nodes()))
            std::cout << "   ---> smaller than threshold size of "
                      << (int)(delete_free_hanging_eles_threshold_ *
                               artconncompdis->num_global_nodes())
                      << " --> its elements will be taken out" << std::endl;
        }
      }
    }
  }

  // user info
  if (myrank_ == 0)
  {
    std::cout << "\n>>>>>>                           End of Find free-hanging 1D elements          "
                 "                 <<<<<<"
              << std::endl;
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<"
                 "<<<<<<<<<<<<<<<<<<<<<<<\n"
              << std::endl;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::depth_first_search_util(
    Core::Nodes::Node* actnode, std::shared_ptr<Core::LinAlg::Vector<int>> visited,
    std::shared_ptr<Core::FE::Discretization> artconncompdis,
    std::shared_ptr<const Core::LinAlg::Vector<double>> ele_diams_artery_full_overlap,
    std::vector<int>& this_connected_comp)
{
  // mark this node visited and add it to this connected component
  const int lid = visited->get_map().lid(actnode->id());
  (*visited)[lid] = 1;
  this_connected_comp.push_back(actnode->id());

  // check all adjacent elements (edges)
  Core::Elements::Element** Elements = actnode->elements();
  for (int i_element = 0; i_element < actnode->num_element(); i_element++)
  {
    Core::Nodes::Node** Nodes = Elements[i_element]->nodes();

    // get diameter
    const double diam = (*ele_diams_artery_full_overlap)[Elements[i_element]->lid()];

    // get the artery-material
    std::shared_ptr<Mat::Cnst1dArt> arterymat =
        std::dynamic_pointer_cast<Mat::Cnst1dArt>(Elements[i_element]->material());
    if (arterymat == nullptr) FOUR_C_THROW("cast to artery material failed");

    // if the element is not collapsed it is connected to this node and we continue with the
    // depth-first search with all nodes of this element
    if (diam >= arterymat->collapse_threshold())
    {
      for (int i_node = 0; i_node < Elements[i_element]->num_node(); i_node++)
      {
        if ((*visited)[Nodes[i_node]->lid()] == 0)
          depth_first_search_util(Nodes[i_node], visited, artconncompdis,
              ele_diams_artery_full_overlap, this_connected_comp);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::
    evaluate_additional_linearizationof_integrated_diam()
{
  // linearizations
  std::vector<Core::LinAlg::SerialDenseMatrix> elestiff(2);

  // evaluate all pairs
  for (unsigned i = 0; i < coupl_elepairs_.size(); i++)
  {
    // only needed if varying diameter is set for this pair
    if (coupl_elepairs_[i]->diam_function_active())
    {
      // evaluate
      coupl_elepairs_[i]->evaluate_additional_linearizationof_integrated_diam(
          &(elestiff[0]), &(elestiff[1]));

      // and FE-Assemble
      const int ele1gid = coupl_elepairs_[i]->ele1_gid();
      const int ele2gid = coupl_elepairs_[i]->ele2_gid();
      const Core::Elements::Element* ele1 = arterydis_->g_element(ele1gid);
      const Core::Elements::Element* ele2 = contdis_->g_element(ele2gid);
      // get element location vector and ownerships
      std::vector<int> lmrow1;
      std::vector<int> lmrow2;
      std::vector<int> lmrowowner1;
      std::vector<int> lmrowowner2;
      std::vector<int> lmstride;

      ele1->location_vector(*arterydis_, lmrow1, lmrowowner1, lmstride);
      ele2->location_vector(*contdis_, lmrow2, lmrowowner2, lmstride);

      FEmat_->fe_assemble(elestiff[0], lmrow1, lmrow1);
      FEmat_->fe_assemble(elestiff[1], lmrow1, lmrow2);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::vector<double>
PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::get_ele_segment_lengths(
    const int artelegid)
{
  if (porofluidprob_) return gid_to_seglength_[artelegid];

  // safety checks
  if (!arterydis_->has_state(1, "curr_seg_lengths"))
    FOUR_C_THROW("cannot get state curr_seg_lengths");

  // build the location array
  Core::Elements::Element* artele = arterydis_->g_element(artelegid);
  std::vector<int> seglengthdofs = arterydis_->dof(1, artele);

  std::shared_ptr<const Core::LinAlg::Vector<double>> curr_seg_lengths =
      arterydis_->get_state(1, "curr_seg_lengths");

  std::vector<double> seglengths = Core::FE::extract_values(*curr_seg_lengths, seglengthdofs);

  return seglengths;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::is_duplicate_segment(
    const std::vector<std::shared_ptr<
        PoroPressureBased::PoroMultiPhaseScatraArteryCouplingPairBase>>& coupl_elepairs,
    PoroPressureBased::PoroMultiPhaseScatraArteryCouplingPairBase& possible_duplicate)
{
  // we have to sort out duplicate segments, these might occur if the artery element
  // lies exactly between two different 2D/3D-elements

  const double eta_a = possible_duplicate.etadata();
  const double eta_b = possible_duplicate.eta_b();
  const int ele1gid = possible_duplicate.ele1_gid();
  int elepairID = -1;

  return is_identical_segment(coupl_elepairs, ele1gid, eta_a, eta_b, elepairID);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::is_identical_segment(
    const std::vector<std::shared_ptr<
        PoroPressureBased::PoroMultiPhaseScatraArteryCouplingPairBase>>& coupl_elepairs,
    const int& ele1gid, const double& etaA, const double& etaB, int& elepairID)
{
  for (unsigned i = 0; i < coupl_elepairs.size(); i++)
  {
    // first check if ele1-Gid is identical
    if (ele1gid == coupl_elepairs[i]->ele1_gid())
      // check if integration segment is the same
      if (fabs(etaA - coupl_elepairs[i]->etadata()) < XIETATOL &&
          fabs(etaB - coupl_elepairs[i]->eta_b()) < XIETATOL)
      {
        if (PROJOUTPUT) std::cout << "found duplicate integration segment" << std::endl;
        elepairID = i;
        return true;
      }
  }

  elepairID = -1;
  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::apply_mesh_movement()
{
  // no need to do this
  if (porofluidprob_) return;

  // only if we evaluate in current configuration
  if (!evaluate_in_ref_config_)
  {
    // safety
    if (!contdis_->has_state(1, "dispnp")) FOUR_C_THROW("cannot get displacement state");

    // update with unaffected length
    current_seg_lengths_artery_->Update(1.0, *unaffected_seg_lengths_artery_, 0.0);

    // apply movement on pairs and fill gid-to-seglength and current_seg_lengths_artery_
    for (unsigned i = 0; i < coupl_elepairs_.size(); i++)
    {
      const double newsegmentlength = coupl_elepairs_[i]->apply_mesh_movement(false, contdis_);
      const int artelegid = coupl_elepairs_[i]->ele1_gid();
      const int segid = coupl_elepairs_[i]->get_segment_id();

      Core::Elements::Element* artele = arterydis_->g_element(artelegid);
      // build the location array
      std::vector<int> seglengthdofs = arterydis_->dof(1, artele);

      current_seg_lengths_artery_->SumIntoGlobalValues(
          1, &seglengthdofs[segid], &(newsegmentlength));
    }

    if (current_seg_lengths_artery_->GlobalAssemble(Add, false) != 0)
      FOUR_C_THROW("GlobalAssemble of current_seg_lengths_artery_ failed");
  }

  // set state on artery dis
  arterydis_->set_state(
      1, "curr_seg_lengths", Core::LinAlg::Vector<double>(*current_seg_lengths_artery_));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::print_out_coupling_method() const
{
  std::cout << "<   Line-based formulation                         >" << std::endl;
  PoroMultiPhaseScaTraArtCouplNonConforming::print_out_coupling_method();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::output_summary() const
{
  if (myrank_ == 0)
  {
    std::cout << "\nSummary of coupling pairs (segments):" << std::endl;
    std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
  }
  Core::Communication::barrier(get_comm());
  for (unsigned i = 0; i < coupl_elepairs_.size(); i++)
  {
    std::cout << "Proc " << std::right << std::setw(2) << myrank_ << ": Artery-ele " << std::right
              << std::setw(5) << coupl_elepairs_[i]->ele1_gid() << ":   [" << std::left
              << std::setw(11) << coupl_elepairs_[i]->etadata() << "," << std::right
              << std::setw(11) << coupl_elepairs_[i]->eta_b() << "] <---> continuous-ele "
              << std::right << std::setw(7) << coupl_elepairs_[i]->ele2_gid() << std::endl;
  }
  Core::Communication::barrier(get_comm());
  if (myrank_ == 0) std::cout << "\n";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
PoroPressureBased::PoroMultiPhaseScaTraArtCouplLineBased::blood_vessel_volume_fraction()
{
  return bloodvesselvolfrac_;
}

FOUR_C_NAMESPACE_CLOSE
