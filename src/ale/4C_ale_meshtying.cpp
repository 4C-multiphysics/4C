// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ale_meshtying.hpp"

#include "4C_ale_utils.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_coupling_adapter_mortar.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_krylov_projector.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mortar_interface.hpp"
#include "4C_mortar_node.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


ALE::Meshtying::Meshtying(std::shared_ptr<Core::FE::Discretization> dis,
    Core::LinAlg::Solver& solver, int msht, int nsd, const Utils::MapExtractor* surfacesplitter)
    : discret_(dis),
      solver_(solver),
      dofrowmap_(discret_->dof_row_map()),
      gsdofrowmap_(nullptr),
      gmdofrowmap_(nullptr),
      mergedmap_(nullptr),
      //  msht_(msht),
      surfacesplitter_(surfacesplitter),
      problemrowmap_(nullptr),
      gndofrowmap_(nullptr),
      gsmdofrowmap_(nullptr),
      valuesdc_(nullptr),
      dconmaster_(false),
      firstnonliniter_(false),
      //  nsd_(nsd),
      is_multifield_(false)
{
  // get the processor ID from the communicator
  myrank_ = Core::Communication::my_mpi_rank(discret_->get_comm());
}

/*-------------------------------------------------------*/
/*  Setup mesh-tying problem                 wirtz 01/16 */
/*-------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseOperator> ALE::Meshtying::setup(
    std::vector<int> coupleddof, std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp)
{
  // time measurement
  TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  1)   Setup Meshtying");
  //  if(coupleddof[nsd_]==0)
  //    pcoupled_=false;

  adapter_mortar(coupleddof);

  if (myrank_ == 0) compare_num_dof();

  dof_row_maps();

  // merge dofrowmap for slave and master discretization
  gsmdofrowmap_ = Core::LinAlg::merge_map(*gmdofrowmap_, *gsdofrowmap_, false);

  // dofrowmap for discretisation without slave and master dofrowmap
  gndofrowmap_ = Core::LinAlg::split_map(*dofrowmap_, *gsmdofrowmap_);

  // map for 2x2 (uncoupled dof's & master dof's)
  mergedmap_ = Core::LinAlg::merge_map(*gndofrowmap_, *gmdofrowmap_, false);

  // std::cout << "number of n dof   " << gndofrowmap_->NumGlobalElements() << std::endl;
  // std::cout << "number of m dof   " << gmdofrowmap_->NumGlobalElements() << std::endl;
  // std::cout << "number of s dof   " << gsdofrowmap_->NumGlobalElements() << std::endl;

  // generate map for blockmatrix
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> alemaps;
  alemaps.push_back(gndofrowmap_);
  alemaps.push_back(gmdofrowmap_);
  alemaps.push_back(gsdofrowmap_);

  Core::LinAlg::MultiMapExtractor extractor;

  extractor.setup(*dofrowmap_, alemaps);

  // check, if extractor maps are valid
  extractor.check_for_valid_map_extractor();

  // allocate 3x3 block sparse matrix with the interface split strategy
  // the interface split strategy speeds up the assembling process,
  // since the information, which nodes are part of the interface, is available
  // -------------------
  // | knn | knm | kns |
  // | kmn | kmm | kms |
  // | ksn | ksm | kss |
  // -------------------

  std::shared_ptr<Core::LinAlg::BlockSparseMatrix<ALE::Utils::InterfaceSplitStrategy>> mat;
  mat = std::make_shared<Core::LinAlg::BlockSparseMatrix<ALE::Utils::InterfaceSplitStrategy>>(
      extractor, extractor, 108, false, true);
  // nodes on the interface
  std::shared_ptr<std::set<int>> condelements =
      surfacesplitter_->conditioned_element_map(*discret_);

  mat->set_cond_elements(condelements);

  // Important: right way to do it (Tobias W.)
  // allocate 2x2 solution matrix with the default block matrix strategy in order to solve the
  // reduced system memory is not allocated(1), since the matrix gets a std::shared_ptr on the
  // respective blocks of the 3x3 block matrix
  // ---------------
  // | knn  | knm' |
  // | kmn' | kmm' |
  // ---------------

  Core::LinAlg::MapExtractor rowmapext(*mergedmap_, gmdofrowmap_, gndofrowmap_);
  Core::LinAlg::MapExtractor dommapext(*mergedmap_, gmdofrowmap_, gndofrowmap_);
  std::shared_ptr<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>
      matsolve = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          dommapext, rowmapext, 1, false, true);
  sysmatsolve_ = matsolve;

  return mat;
}

/*-------------------------------------------------------*/
/*  Use the split of the ale mesh tying for the sysmat   */
/*                                           wirtz 01/16 */
/*-------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseOperator> ALE::Meshtying::msht_split()
{
  // generate map for blockmatrix
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> alemaps;
  alemaps.push_back(gndofrowmap_);
  alemaps.push_back(gmdofrowmap_);
  alemaps.push_back(gsdofrowmap_);

  Core::LinAlg::MultiMapExtractor extractor;

  extractor.setup(*dofrowmap_, alemaps);

  // check, if extractor maps are valid
  extractor.check_for_valid_map_extractor();

  // allocate 3x3 block sparse matrix with the interface split strategy
  // the interface split strategy speeds up the assembling process,
  // since the information, which nodes are part of the interface, is available
  // -------------------
  // | knn | knm | kns |
  // | kmn | kmm | kms |
  // | ksn | ksm | kss |
  // -------------------

  std::shared_ptr<Core::LinAlg::BlockSparseMatrix<ALE::Utils::InterfaceSplitStrategy>> mat;
  mat = std::make_shared<Core::LinAlg::BlockSparseMatrix<ALE::Utils::InterfaceSplitStrategy>>(
      extractor, extractor, 108, false, true);
  // nodes on the interface
  std::shared_ptr<std::set<int>> condelements =
      surfacesplitter_->conditioned_element_map(*discret_);
  mat->set_cond_elements(condelements);

  return mat;
}

/*------------------------------------------------------------------------------*/
/*  Check if Dirichlet BC are defined on the master                 wirtz 01/16 */
/*------------------------------------------------------------------------------*/
void ALE::Meshtying::dirichlet_on_master(std::shared_ptr<const Core::LinAlg::Map> bmaps)
{
  // This method checks if Dirichlet or Dirichlet-like boundary conditions are defined
  // on the master side of the internal interface.
  // In this case, the slave side has to be handled in a special way
  // strategies:
  // (a)  Apply DC on both master and slave side of the internal interface (->disabled)
  //      -> over-constraint system, but nevertheless, result is correct and no solver issues
  // (b)  DC are projected from the master to the slave side during prepare_time_step
  //      (in project_master_to_slave_for_overlapping_bc()) (-> disabled)
  //      -> DC also influence slave nodes which are not part of the inflow
  //
  //      if(msht_ != ALE::no_meshtying)
  //        meshtying_->project_master_to_slave_for_overlapping_bc(dispnp_, dbcmaps_->cond_map());
  //
  // (c)  DC are included in the condensation process (-> actual strategy)

  std::vector<std::shared_ptr<const Core::LinAlg::Map>> intersectionmaps;
  intersectionmaps.push_back(bmaps);
  std::shared_ptr<const Core::LinAlg::Map> gmdofrowmap = gmdofrowmap_;
  intersectionmaps.push_back(gmdofrowmap);
  std::shared_ptr<Core::LinAlg::Map> intersectionmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(intersectionmaps);

  if (intersectionmap->num_global_elements() != 0)
  {
    dconmaster_ = true;
    if (myrank_ == 0)
    {
      std::cout
          << "Dirichlet or Dirichlet-like boundary condition defined on master side of the "
             "internal interface!\n "
          << "These conditions has to be also included at the slave side of the internal interface"
          << std::endl
          << std::endl;
    }
  }

  return;
}

/*---------------------------------------------------*/
/*  Prepare Meshtying system             wirtz 01/16 */
/*---------------------------------------------------*/
void ALE::Meshtying::prepare_meshtying_system(std::shared_ptr<Core::LinAlg::SparseOperator>& sysmat,
    std::shared_ptr<Core::LinAlg::Vector<double>>& residual,
    std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  2)   Condensation block matrix");

  condensation_operation_block_matrix(sysmat, residual, dispnp);
  return;
}

/*-------------------------------------------------------*/
/*  Split Vector                             wirtz 01/16 */
/*-------------------------------------------------------*/
void ALE::Meshtying::split_vector(Core::LinAlg::Vector<double>& vector,
    std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>& splitvector)
{
  TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  2.2)   - Split Vector");

  // we want to split f into 3 groups s.m,n
  std::shared_ptr<Core::LinAlg::Vector<double>> fs, fm, fn;

  // temporarily we need the group sm
  std::shared_ptr<Core::LinAlg::Vector<double>> fsm;

  /**********************************************************************/
  /* Split feff into 3 subvectors                                       */
  /**********************************************************************/

  // do the vector splitting smn -> sm+n
  Core::LinAlg::split_vector(*dofrowmap_, vector, gsmdofrowmap_, fsm, gndofrowmap_, fn);

  // we want to split fsm into 2 groups s,m
  fs = std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
  fm = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofrowmap_);

  // do the vector splitting sm -> s+m
  Core::LinAlg::split_vector(*gsmdofrowmap_, *fsm, gsdofrowmap_, fs, gmdofrowmap_, fm);

  // splitvector[ii]
  // fn [0]
  // fm [1]
  // fs [2]

  splitvector[0] = fn;
  splitvector[1] = fm;
  splitvector[2] = fs;

  return;
}

/*-------------------------------------------------------*/
/*-------------------------------------------------------*/
void ALE::Meshtying::split_vector_based_on3x3(
    Core::LinAlg::Vector<double>& orgvector, Core::LinAlg::Vector<double>& vectorbasedon2x2)
{
  // container for split residual vector
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitvector(3);

  split_vector(orgvector, splitvector);
  // build up the reduced residual
  Core::LinAlg::export_to(*(splitvector[0]), vectorbasedon2x2);
  Core::LinAlg::export_to(*(splitvector[1]), vectorbasedon2x2);

  return;
}



/*-------------------------------------------------------*/
/*  Set the flag for multifield problems     wirtz 01/16 */
/*                                                       */
/*-------------------------------------------------------*/
void ALE::Meshtying::is_multifield(
    const Core::LinAlg::MultiMapExtractor& interface,  ///< interface maps for split of ale matrix
    bool ismultifield                                  ///< flag for multifield problems
)
{
  multifield_interface_ = interface;
  is_multifield_ = ismultifield;

  return;
}

/*-------------------------------------------------------*/
/*  Use the split of the ale mesh tying for the sysmat   */
/*                                           wirtz 01/16 */
/*-------------------------------------------------------*/
void ALE::Meshtying::msht_split(std::shared_ptr<Core::LinAlg::SparseOperator>& sysmat)
{
  if (is_multifield_)
  {
    // generate map for blockmatrix
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> alemaps;
    alemaps.push_back(gndofrowmap_);
    alemaps.push_back(gmdofrowmap_);
    alemaps.push_back(gsdofrowmap_);

    Core::LinAlg::MultiMapExtractor extractor;

    extractor.setup(*dofrowmap_, alemaps);

    // check, if extractor maps are valid
    extractor.check_for_valid_map_extractor();

    // allocate 3x3 block sparse matrix with the interface split strategy
    // the interface split strategy speeds up the assembling process,
    // since the information, which nodes are part of the interface, is available
    // -------------------
    // | knn | knm | kns |
    // | kmn | kmm | kms |
    // | ksn | ksm | kss |
    // -------------------

    std::shared_ptr<Core::LinAlg::BlockSparseMatrix<ALE::Utils::InterfaceSplitStrategy>> mat;
    mat = std::make_shared<Core::LinAlg::BlockSparseMatrix<ALE::Utils::InterfaceSplitStrategy>>(
        extractor, extractor, 108, false, true);
    // nodes on the interface
    std::shared_ptr<std::set<int>> condelements =
        surfacesplitter_->conditioned_element_map(*discret_);
    mat->set_cond_elements(condelements);

    sysmat = mat;
  }
}

/*-------------------------------------------------------*/
/*  Use the split of the multifield problem for the      */
/*  sysmat                                   wirtz 01/16 */
/*-------------------------------------------------------*/
void ALE::Meshtying::multifield_split(std::shared_ptr<Core::LinAlg::SparseOperator>& sysmat)
{
  if (is_multifield_)
  {
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmatnew =
        std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat);

    Core::LinAlg::Vector<double> ones(sysmatnew->matrix(2, 2).row_map());
    ones.put_scalar(1.0);

    Core::LinAlg::SparseMatrix onesdiag(ones);
    onesdiag.complete();

    sysmatnew->matrix(0, 2).un_complete();
    sysmatnew->matrix(0, 2).zero();

    sysmatnew->matrix(1, 2).un_complete();
    sysmatnew->matrix(1, 2).zero();

    sysmatnew->matrix(2, 2).un_complete();
    sysmatnew->matrix(2, 2).zero();
    Core::LinAlg::matrix_add(onesdiag, false, 1.0, sysmatnew->matrix(2, 2), 1.0);

    sysmatnew->matrix(2, 0).un_complete();
    sysmatnew->matrix(2, 0).zero();

    sysmatnew->matrix(2, 1).un_complete();
    sysmatnew->matrix(2, 1).zero();

    sysmatnew->complete();

    std::shared_ptr<Core::LinAlg::SparseMatrix> mergedmatrix = sysmatnew->merge();

    Core::LinAlg::MapExtractor extractor(
        *multifield_interface_.full_map(), multifield_interface_.map(1));

    std::shared_ptr<Core::LinAlg::BlockSparseMatrix<ALE::Utils::InterfaceSplitStrategy>> mat =
        Core::LinAlg::split_matrix<ALE::Utils::InterfaceSplitStrategy>(
            *mergedmatrix, extractor, extractor);

    mat->complete();

    sysmat = mat;
  }
}

/*-------------------------------------------------------*/
/*  Call the constructor and the setup of the mortar     */
/*  coupling adapter                         wirtz 02/16 */
/*-------------------------------------------------------*/
void ALE::Meshtying::adapter_mortar(std::vector<int> coupleddof)
{
  adaptermeshtying_ = std::make_shared<Coupling::Adapter::CouplingMortar>(
      Global::Problem::instance()->n_dim(), Global::Problem::instance()->mortar_coupling_params(),
      Global::Problem::instance()->contact_dynamic_params(),
      Global::Problem::instance()->spatial_approximation_type());

  // Setup of meshtying adapter
  adaptermeshtying_->setup(discret_, discret_, nullptr, coupleddof, "Mortar", discret_->get_comm(),
      Global::Problem::instance()->function_manager(),
      Global::Problem::instance()->binning_strategy_params(),
      Global::Problem::instance()->discretization_map(),
      Global::Problem::instance()->output_control_file(),
      Global::Problem::instance()->spatial_approximation_type(), true);
}

/*-------------------------------------------------------*/
/*  Compare the size of the slave and master dof row map */
/*                                           wirtz 02/16 */
/*-------------------------------------------------------*/
void ALE::Meshtying::compare_num_dof()
{
  int numdofmaster = (adaptermeshtying_->master_dof_map())->num_global_elements();
  int numdofslave = (adaptermeshtying_->slave_dof_map())->num_global_elements();

  std::cout << std::endl << "number of master dof's:   " << numdofmaster << std::endl;
  std::cout << "number of slave dof's:   " << numdofslave << std::endl << std::endl;

  if (numdofmaster > numdofslave)
    std::cout << "The master side is discretized by more elements than the slave side" << std::endl;
  else
    std::cout << "The slave side is discretized by more elements than the master side" << std::endl;
}

/*-------------------------------------------------------*/
/*  Get function for the slave and master dof row map    */
/*                                           wirtz 02/16 */
/*-------------------------------------------------------*/
void ALE::Meshtying::dof_row_maps()
{
  // slave dof rowmap
  gsdofrowmap_ = adaptermeshtying_->slave_dof_map();

  // master dof rowmap
  gmdofrowmap_ = adaptermeshtying_->master_dof_map();
}

/*-------------------------------------------------------*/
/*  Get function for the P matrix            wirtz 02/16 */
/*                                                       */
/*-------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> ALE::Meshtying::get_mortar_matrix_p()
{
  return adaptermeshtying_->get_mortar_matrix_p();
}

/*-------------------------------------------------------*/
/*  Condensation operation block matrix      wirtz 01/16 */
/*                                                       */
/*-------------------------------------------------------*/
void ALE::Meshtying::condensation_operation_block_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator>& sysmat,
    std::shared_ptr<Core::LinAlg::Vector<double>>& residual,
    std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp)
{
  /**********************************************************************/
  /* Split residual into 3 subvectors                                   */
  /**********************************************************************/

  // container for split residual vector
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitres(3);
  split_vector(*residual, splitres);

  /**********************************************************************/
  /* Condensate blockmatrix                                             */
  /**********************************************************************/

  TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  2.1)   - Condensation Operation");

  // cast std::shared_ptr<Core::LinAlg::SparseOperator> to a
  // std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmatnew =
      std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat);

  /**********************************************************************/
  /* Build the final sysmat and residual                                */
  /**********************************************************************/

  // only the blocks nm, mn and mm are modified
  // the other blocks remain unchanged, since only the 2x2 block matrix system is solved
  // ---------------------        ------------------
  // | nn | nm | ns | 0  |        | nn  | nm' | ns  |
  // | mn | mm | ms | D  |   ->   | mn' | mm' | ms  |
  // | sn | sm | ss | -M |        | sn  | sm  | ss  |
  // |  0 | DT |-MT | 0  |        ------------------
  // ---------------------
  // solved system (2x2 matrix)
  // -------------
  // | nn  | nm' |
  // | mn' | mm' |
  // -------------

  // Dirichlet or Dirichlet-like condition on the master side of the internal interface:
  // First time step:
  // coupling condition: u_s - u_m = delta u_m^D
  // instead of          u_s - u_m = 0
  //
  // this has to be considered in the condensation and in update process

  std::shared_ptr<Core::LinAlg::Vector<double>> dcnm = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> dcmm = nullptr;
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitdcmaster(3);

  if (dconmaster_ == true and firstnonliniter_ == true)
  {
    dcnm = std::make_shared<Core::LinAlg::Vector<double>>(*gndofrowmap_, true);
    dcmm = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofrowmap_, true);

    split_vector(*valuesdc_, splitdcmaster);
  }

  // get transformation matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> P = get_mortar_matrix_p();

  /*--------------------------------------------------------------------*/
  // block nm
  /*--------------------------------------------------------------------*/
  // compute modification for block nm
  std::shared_ptr<Core::LinAlg::SparseMatrix> knm_mod =
      matrix_multiply(sysmatnew->matrix(0, 2), false, *P, false, false, false, true);

  // Add transformation matrix to nm
  sysmatnew->matrix(0, 1).un_complete();
  Core::LinAlg::matrix_add(*knm_mod, false, 1.0, sysmatnew->matrix(0, 1), 1.0);

  if (dconmaster_ == true and firstnonliniter_ == true)
    knm_mod->multiply(false, *(splitdcmaster[1]), *dcnm);

  /*--------------------------------------------------------------------*/
  // block mn
  /*--------------------------------------------------------------------*/
  // compute modification for block kmn
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmn_mod =
      matrix_multiply(*P, true, sysmatnew->matrix(2, 0), false, false, false, true);

  // Add transformation matrix to mn
  sysmatnew->matrix(1, 0).un_complete();
  Core::LinAlg::matrix_add(*kmn_mod, false, 1.0, sysmatnew->matrix(1, 0), 1.0);

  /*--------------------------------------------------------------------*/
  // block mm
  /*--------------------------------------------------------------------*/
  // compute modification for block kmm
  std::shared_ptr<Core::LinAlg::SparseMatrix> kss_mod =
      matrix_multiply(*P, true, sysmatnew->matrix(2, 2), false, false, false, true);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmm_mod =
      matrix_multiply(*kss_mod, false, *P, false, false, false, true);

  // Add transformation matrix to mm
  sysmatnew->matrix(1, 1).un_complete();
  Core::LinAlg::matrix_add(*kmm_mod, false, 1.0, sysmatnew->matrix(1, 1), 1.0);

  if (dconmaster_ == true and firstnonliniter_ == true)
    kmm_mod->multiply(false, *(splitdcmaster[1]), *dcmm);

  // complete matrix
  sysmatnew->complete();

  //*************************************************
  //  condensation operation for the residual
  //*************************************************

  // r_m: add P^T*r_s
  Core::LinAlg::Vector<double> fm_mod(*gmdofrowmap_, true);
  P->multiply(true, *(splitres[2]), fm_mod);

  // r_m: insert Dirichlet boundary conditions
  if (dconmaster_ == true and firstnonliniter_ == true) fm_mod.update(-1.0, *dcmm, 1.0);

  // export and add r_m subvector to residual
  Core::LinAlg::Vector<double> fm_modexp(*dofrowmap_);
  Core::LinAlg::export_to(fm_mod, fm_modexp);
  residual->update(1.0, fm_modexp, 1.0);

  if (dconmaster_ == true and firstnonliniter_ == true)
  {
    Core::LinAlg::Vector<double> fn_exp(*dofrowmap_, true);
    Core::LinAlg::export_to(*dcnm, fn_exp);
    residual->update(-1.0, fn_exp, 1.0);
  }

  // export r_s = zero to residual
  Core::LinAlg::Vector<double> fs_mod(*gsdofrowmap_, true);
  Core::LinAlg::export_to(fs_mod, *residual);

  return;
}

/*-------------------------------------------------------*/
/*  Compute and update Slave DOF's           wirtz 01/16 */
/*                                                       */
/*-------------------------------------------------------*/
void ALE::Meshtying::update_slave_dof(std::shared_ptr<Core::LinAlg::Vector<double>>& inc,
    std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  3.4)   - Update slave DOF");

  // get dof row map
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // split incremental and displacement vector
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitinc(3);
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitdisp(3);
  split_vector(*inc, splitinc);
  split_vector(*dispnp, splitdisp);

  // Dirichlet or Dirichlet-like condition on the master side of the internal interface:
  // First time step:
  // coupling condition: u_s - u_m = delta u_m^D
  // instead of          u_s - u_m = 0
  //
  // this has to be considered in the condensation and in update process

  // split vector containing Dirichlet boundary conditions, if any
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitdcmaster(3);
  if (dconmaster_ == true and firstnonliniter_ == true) split_vector(*valuesdc_, splitdcmaster);

  // get transformation matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> P = get_mortar_matrix_p();

  // define new incremental vector
  std::shared_ptr<Core::LinAlg::Vector<double>> incnew =
      std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // delta_vp^s: add P*delta_vp^m
  Core::LinAlg::Vector<double> fs_mod(*gsdofrowmap_, true);
  P->multiply(false, *(splitinc[1]), fs_mod);

  // delta_vp^s: subtract vp_i^s
  fs_mod.update(-1.0, *(splitdisp[2]), 1.0);

  // delta_vp^s: add P*vp_i^m
  Core::LinAlg::Vector<double> fs_mod_m(*gsdofrowmap_, true);
  P->multiply(false, *(splitdisp[1]), fs_mod_m);
  fs_mod.update(1.0, fs_mod_m, 1.0);

  // set Dirichlet boundary conditions, if any
  if (dconmaster_ == true and firstnonliniter_ == true)
  {
    Core::LinAlg::Vector<double> fsdc_mod(*gsdofrowmap_, true);
    P->multiply(false, *(splitdcmaster[1]), fsdc_mod);
    fs_mod.update(1.0, fsdc_mod, 1.0);
  }

  // export interior degrees of freedom
  Core::LinAlg::Vector<double> fnexp(*dofrowmap);
  Core::LinAlg::export_to(*(splitinc[0]), fnexp);
  incnew->update(1.0, fnexp, 1.0);

  // export master degrees of freedom
  Core::LinAlg::Vector<double> fmexp(*dofrowmap);
  Core::LinAlg::export_to(*(splitinc[1]), fmexp);
  incnew->update(1.0, fmexp, 1.0);

  // export slave degrees of freedom
  Core::LinAlg::Vector<double> fs_modexp(*dofrowmap);
  Core::LinAlg::export_to(fs_mod, fs_modexp);
  incnew->update(1.0, fs_modexp, 1.0);

  // set iteration counter for Dirichlet boundary conditions, if any
  if (dconmaster_ == true and firstnonliniter_ == true) firstnonliniter_ = false;

  // define incremental vector to new incremental vector
  inc = incnew;

  return;
}

/*-------------------------------------------------------*/
/*  solve mesh-tying system                  wirtz 01/16 */
/*                                                       */
/*-------------------------------------------------------*/
int ALE::Meshtying::solve_meshtying(Core::LinAlg::Solver& solver,
    std::shared_ptr<Core::LinAlg::SparseOperator> sysmat,
    std::shared_ptr<Core::LinAlg::Vector<double>>& disi,
    std::shared_ptr<Core::LinAlg::Vector<double>> residual,
    std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp)
{
  // time measurement
  TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  3)   Solve meshtying system");

  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmatnew =
      std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat);
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmatsolve =
      std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmatsolve_);

  std::shared_ptr<Core::LinAlg::Vector<double>> res = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> dis = nullptr;

  std::shared_ptr<Core::LinAlg::SparseMatrix> mergedmatrix = nullptr;

  res = std::make_shared<Core::LinAlg::Vector<double>>(*mergedmap_, true);
  dis = std::make_shared<Core::LinAlg::Vector<double>>(*mergedmap_, true);

  mergedmatrix = std::make_shared<Core::LinAlg::SparseMatrix>(*mergedmap_, 108, false, true);

  int errorcode = 0;

  {
    TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  3.1)   - Preparation");
    split_vector_based_on3x3(*residual, *res);

    // assign blocks to the solution matrix
    sysmatsolve->assign(0, 0, Core::LinAlg::DataAccess::Share, sysmatnew->matrix(0, 0));
    sysmatsolve->assign(0, 1, Core::LinAlg::DataAccess::Share, sysmatnew->matrix(0, 1));
    sysmatsolve->assign(1, 0, Core::LinAlg::DataAccess::Share, sysmatnew->matrix(1, 0));
    sysmatsolve->assign(1, 1, Core::LinAlg::DataAccess::Share, sysmatnew->matrix(1, 1));
    sysmatsolve->complete();

    mergedmatrix = sysmatsolve->merge();
  }

  {
    TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  3.2)   - Solve");

    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    errorcode = solver_.solve(mergedmatrix, dis, res, solver_params);

    Core::LinAlg::export_to(*dis, *disi);
    Core::LinAlg::export_to(*res, *residual);
    // compute and update slave dof's
    update_slave_dof(disi, dispnp);
  }
  return errorcode;
}

FOUR_C_NAMESPACE_CLOSE
