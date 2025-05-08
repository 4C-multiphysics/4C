// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_artery_coupling_nodebased.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_condition_selector.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::PoroMultiPhaseScaTraArtCouplNodeBased(
    std::shared_ptr<Core::FE::Discretization> arterydis,
    std::shared_ptr<Core::FE::Discretization> contdis,
    const Teuchos::ParameterList& meshtyingparams, const std::string& condname,
    const std::string& artcoupleddofname, const std::string& contcoupleddofname)
    : PoroMultiPhaseScaTraArtCouplBase(
          arterydis, contdis, meshtyingparams, condname, artcoupleddofname, contcoupleddofname),
      condname_(condname)
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

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::init()
{
  // ARTERY COUPLING CONDITIONS
  std::vector<std::vector<int>> condIDs;
  std::vector<int> artIDs;
  std::vector<int> contfieldIDs;
  condIDs.push_back(artIDs);
  condIDs.push_back(contfieldIDs);

  // check if conditions are defined on both discretizations --------------------------
  // 1) 1D artery discretization
  std::vector<Core::Conditions::Condition*> artCoupcond;
  arterydis_->get_condition(condname_, artCoupcond);

  for (auto& iter : artCoupcond)
  {
    int myID = iter->parameters().get<int>("COUPID");
    condIDs[0].push_back(myID);
  }

  // 2) 2D, 3D continuous field discretization
  std::vector<Core::Conditions::Condition*> contfieldCoupcond;
  contdis_->get_condition(condname_, contfieldCoupcond);

  for (auto& iter : contfieldCoupcond)
  {
    int myID = iter->parameters().get<int>("COUPID");
    condIDs[1].push_back(myID);
  }

  if (condIDs[0].size() != condIDs[1].size())
    FOUR_C_THROW("Artery coupling conditions need to be defined on both discretizations");

  // -----------------------------------------------------------------------------------------------------------------
  // create map extractors needed for artery condition coupling --> continuous field part
  contfieldex_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();
  setup_map_extractor(*contfieldex_, *contdis_, coupleddofs_cont_);
  check_dbc_on_coupled_dofs(*contdis_, contfieldex_->map(1));

  // -----------------------------------------------------------------------------------------------------------------
  // create map extractors needed for artery condition coupling --> artery part
  artex_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();
  setup_map_extractor(*artex_, *arterydis_, coupleddofs_art_);
  check_dbc_on_coupled_dofs(*arterydis_, artex_->map(1));

  // setup coupling adapter
  artcontfieldcoup_ = std::make_shared<Coupling::Adapter::Coupling>();
  artcontfieldcoup_->setup_condition_coupling(*contdis_, contfieldex_->map(1), *arterydis_,
      artex_->map(1), condname_, coupleddofs_cont_, coupleddofs_art_);

  // full map of continuous field and uncoupled dofs of artery
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps;
  maps.push_back(contfieldex_->full_map());
  maps.push_back(artex_->map(0));

  fullmap_ = Core::LinAlg::MultiMapExtractor::merge_maps(maps);
  /// dof row map of coupled problem split in (field) blocks
  globalex_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();
  globalex_->setup(*fullmap_, maps);

  // check global map extractor
  globalex_->check_for_valid_map_extractor();

  // needed for matrix transformations
  sbbtransform_ = std::make_shared<Coupling::Adapter::MatrixRowColTransform>();
  sbitransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  sibtransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::setup()
{
  // do nothing
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::setup_map_extractor(
    Core::LinAlg::MultiMapExtractor& mapextractor, Core::FE::Discretization& dis,
    const std::vector<int>& coupleddofs)
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> partialmaps_coupled;

  // build coupled maps for all coupled dofs
  for (int idof = 0; idof < num_coupled_dofs_; idof++)
  {
    Core::LinAlg::MultiMapExtractor dummy;
    Core::Conditions::setup_extractor(dis, dummy,
        {Core::Conditions::Selector(condname_, coupleddofs[idof], coupleddofs[idof] + 1)});

    partialmaps_coupled.push_back(dummy.map(1));
  }
  // fullmap coupled -> all coupled dofs
  std::shared_ptr<Core::LinAlg::Map> fullmap_coupled =
      Core::LinAlg::MultiMapExtractor::merge_maps(partialmaps_coupled);

  // fullmap uncoupled -> all uncoupled dofs
  Core::LinAlg::MapExtractor temp(*dis.dof_row_map(), fullmap_coupled, false);
  std::shared_ptr<Core::LinAlg::Map> fullmap_uncoupled =
      std::make_shared<Core::LinAlg::Map>(*temp.cond_map());

  // vector for setup of extractor
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> fullmap_vector;
  fullmap_vector.push_back(fullmap_uncoupled);
  fullmap_vector.push_back(fullmap_coupled);

  mapextractor.setup(*dis.dof_row_map(), fullmap_vector);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::setup_system(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_cont,
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_art,
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_cont,
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_art,
    std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_cont,
    std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmap_art)
{
  setup_rhs(rhs, rhs_cont, rhs_art);
  setup_matrix(sysmat, sysmat_cont, *sysmat_art);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::setup_rhs(
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs,
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_cont,
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_art)
{
  setup_vector(rhs, rhs_cont, rhs_art);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::setup_vector(
    std::shared_ptr<Core::LinAlg::Vector<double>> vec,
    std::shared_ptr<const Core::LinAlg::Vector<double>> vec_cont,
    std::shared_ptr<const Core::LinAlg::Vector<double>> vec_art)
{
  // zero out
  vec->put_scalar(0.0);

  // inner (uncoupled) DOFs of artery
  std::shared_ptr<Core::LinAlg::Vector<double>> vec2_uncoupled =
      artex_->extract_vector(*vec_art, 0);

  // boundary (coupled) DOFs of artery
  std::shared_ptr<Core::LinAlg::Vector<double>> vec2_coupled = artex_->extract_vector(*vec_art, 1);

  // transform boundary DOFs to continuous dis
  std::shared_ptr<Core::LinAlg::Vector<double>> temp =
      contfieldex_->insert_vector(*artcontfieldcoup_->slave_to_master(*vec2_coupled), 1);

  // add to continuous vec
  temp->update(1.0, *vec_cont, 1.0);

  // set up global vector
  globalex_->insert_vector(*temp, 0, *vec);
  globalex_->insert_vector(*vec2_uncoupled, 1, *vec);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::setup_matrix(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmat,
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_cont, Core::LinAlg::SparseMatrix& sysmat_art)
{
  // uncomplete
  sysmat_cont->un_complete();

  // artery
  // first split the matrix into 2x2 blocks (boundary vs. inner dofs)
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockartery =
      Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          sysmat_art, *(artex_), *(artex_));
  blockartery->complete();

  // inner artery dofs
  sysmat->assign(1, 1, Core::LinAlg::DataAccess::View, blockartery->matrix(0, 0));

  (*sibtransform_)(blockartery->full_row_map(), blockartery->full_col_map(),
      blockartery->matrix(0, 1), 1.0, Coupling::Adapter::CouplingSlaveConverter(*artcontfieldcoup_),
      sysmat->matrix(1, 0));

  (*sbitransform_)(blockartery->matrix(1, 0), 1.0,
      Coupling::Adapter::CouplingSlaveConverter(*artcontfieldcoup_), sysmat->matrix(0, 1));

  (*sbbtransform_)(blockartery->matrix(1, 1), 1.0,
      Coupling::Adapter::CouplingSlaveConverter(*artcontfieldcoup_),
      Coupling::Adapter::CouplingSlaveConverter(*artcontfieldcoup_), *sysmat_cont, true, true);

  // continuous field
  sysmat->assign(0, 0, Core::LinAlg::DataAccess::View, *sysmat_cont);
  // complete
  sysmat->complete();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::extract_single_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> globalvec,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& vec_cont,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& vec_art)
{
  // process second field (continuous)
  vec_cont = globalex_->extract_vector(*globalvec, 0);

  // process coupled (boundary) DOFs of the second field
  std::shared_ptr<Core::LinAlg::Vector<double>> boundary =
      contfieldex_->extract_vector(*vec_cont, 1);

  // process inner (uncoupled) and boundary (coupled) DOFs of artery
  std::shared_ptr<const Core::LinAlg::Vector<double>> artery_inner =
      globalex_->extract_vector(*globalvec, 1);
  std::shared_ptr<Core::LinAlg::Vector<double>> artery_boundary =
      artcontfieldcoup_->master_to_slave(*boundary);

  // build vector for artery
  // 1) inner DOFs
  std::shared_ptr<Core::LinAlg::Vector<double>> artery_temp =
      artex_->insert_vector(*artery_inner, 0);
  // 2) boundary DOFs
  artex_->insert_vector(*artery_boundary, 1, *artery_temp);

  vec_art = artery_temp;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::check_dbc_on_coupled_dofs(
    Core::FE::Discretization& dis, const std::shared_ptr<const Core::LinAlg::Map>& coupleddofmap)
{
  // object holds maps/subsets for DOFs subjected to Dirichlet BCs and otherwise
  std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps =
      std::make_shared<Core::LinAlg::MapExtractor>();
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros =
        Core::LinAlg::create_vector(*dis.dof_row_map(), true);
    Teuchos::ParameterList eleparams;
    // other parameters needed by the elements
    eleparams.set("total time", 0.0);
    eleparams.set<const Core::Utils::FunctionManager*>(
        "function_manager", &Global::Problem::instance()->function_manager());
    dis.evaluate_dirichlet(eleparams, zeros, nullptr, nullptr, nullptr, dbcmaps);
  }
  // intersect DBC maps and coupled dof map to check if coupling and DBC are applied on same dofs
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> dummy;
  dummy.push_back(dbcmaps->cond_map());
  dummy.push_back(coupleddofmap);
  std::shared_ptr<Core::LinAlg::Map> intersect_dbc_coupled =
      Core::LinAlg::MultiMapExtractor::intersect_maps(dummy);

  if (intersect_dbc_coupled->NumGlobalElements() > 0)
  {
    if (myrank_ == 0)
    {
      std::cout << "\n\n";
      std::cout << "You cannot define DBC and nodal coupling conditions on the same node\n"
                   "for discretization "
                << dis.name()
                << "\n"
                   "The problematic DOFs are:"
                << std::endl;
    }
    intersect_dbc_coupled->Print(std::cout);
    FOUR_C_THROW("Re-think your Input file definition");
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::check_initial_fields(
    std::shared_ptr<const Core::LinAlg::Vector<double>> vec_cont,
    std::shared_ptr<const Core::LinAlg::Vector<double>> vec_art)
{
  // boundary (coupled) DOFs of artery
  std::shared_ptr<Core::LinAlg::Vector<double>> vec2_coupled = artex_->extract_vector(*vec_art, 1);

  // transform boundary DOFs to continuous dis
  std::shared_ptr<Core::LinAlg::Vector<double>> temp =
      artcontfieldcoup_->slave_to_master(*vec2_coupled);

  // process coupled (boundary) DOFs of the second field
  std::shared_ptr<Core::LinAlg::Vector<double>> boundary =
      contfieldex_->extract_vector(*vec_cont, 1);

  // subtract artery DOF values from continuous DOF values
  boundary->update(-1.0, *temp, 1.0);

  // build L2 norm
  double diff(0.0);
  boundary->norm_2(&diff);

  if (diff > 1.0e-9)
  {
    FOUR_C_THROW("Your initial fields apparently are different with an L2 norm of {}", diff);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::artery_dof_row_map() const
{
  return artex_->map(0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::dof_row_map() const
{
  return fullmap_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::apply_mesh_movement()
{
  if (!evaluate_in_ref_config_)
    FOUR_C_THROW("Evaluation in current configuration not possible for node-based coupling");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::blood_vessel_volume_fraction()
{
  FOUR_C_THROW("Output of vessel volume fraction not possible for node-based coupling");

  return nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void PoroPressureBased::PoroMultiPhaseScaTraArtCouplNodeBased::print_out_coupling_method() const
{
  std::cout << "<   Coupling-Method : Nodebased                    >" << std::endl;
}

FOUR_C_NAMESPACE_CLOSE
