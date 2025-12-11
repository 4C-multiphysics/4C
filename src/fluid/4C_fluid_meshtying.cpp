// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_meshtying.hpp"

#include "4C_coupling_adapter_mortar.hpp"
#include "4C_fluid_utils.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_krylov_projector.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_mortar_interface.hpp"
#include "4C_mortar_node.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


FLD::Meshtying::Meshtying(std::shared_ptr<Core::FE::Discretization> dis,
    Core::LinAlg::Solver& solver, int msht, int nsd, const Utils::MapExtractor* surfacesplitter)
    : discret_(dis),
      solver_(solver),
      msht_(msht),
      myrank_(Core::Communication::my_mpi_rank(dis->get_comm())),
      surfacesplitter_(surfacesplitter),
      dofrowmap_(nullptr),
      problemrowmap_(nullptr),
      gndofrowmap_(nullptr),
      gsmdofrowmap_(nullptr),
      gsdofrowmap_(nullptr),
      gmdofrowmap_(nullptr),
      mergedmap_(nullptr),
      valuesdc_(nullptr),
      adaptermeshtying_(
          std::make_shared<Coupling::Adapter::CouplingMortar>(Global::Problem::instance()->n_dim(),
              Global::Problem::instance()->mortar_coupling_params(),
              Global::Problem::instance()->contact_dynamic_params(),
              Global::Problem::instance()->spatial_approximation_type())),
      pcoupled_(true),
      dconmaster_(false),
      firstnonliniter_(false),
      nsd_(nsd),
      multifield_condelements_(nullptr),
      multifield_condelements_shape_(nullptr),
      multifield_splitmatrix_(false),
      is_multifield_(false)
{
}

/*-------------------------------------------------------*/
/*  Setup mesh-tying problem                ehrl (04/11) */
/*-------------------------------------------------------*/

void FLD::Meshtying::setup_meshtying(const std::vector<int>& coupleddof, const bool pcoupled)
{
  // get pointer to dof row map
  dofrowmap_ = discret_->dof_row_map();

  // set whether pressure dof is coupled
  pcoupled_ = pcoupled;

  // time measurement
  TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  1)   Setup Meshtying");

  // Setup of meshtying adapter
  adaptermeshtying_->setup(discret_, discret_, nullptr, coupleddof, "Mortar", discret_->get_comm(),
      Global::Problem::instance()->function_manager(),
      Global::Problem::instance()->binning_strategy_params(),
      Global::Problem::instance()->discretization_map(),
      Global::Problem::instance()->output_control_file(),
      Global::Problem::instance()->spatial_approximation_type(), true);

  // 4 different systems to solve
  // a) Condensation with a block matrix (condensed_bmat)
  //    system is solved in a 2x2 (n,m) block matrix with the respective solvers

  // b) Condensation with a block matrix merged to a sparse matrix (condensed_bmat_merged)
  //    - condensation operation is done in the 2x2 (n,m) block matrix (no splitting operations)
  //      -> graph can be saved resulting in accelerated element assembly
  //         (ifdef: allocation of new matrix, more memory, slower element assembly,
  //                 no block matrix subtraction)
  //    - one's are assigned to the diagonal entries in the ss-block (as for dirichlet conditions)
  //    properties:
  //    - splitting operation is an additional, time consuming operation
  //    - resulting system matrix is easy to solve

  // c) Condensation in a sparse matrix (condensed_smat)
  //    - condensation operation is done in the original 3x3 (n,m,s) sparse matrix
  //      by splitting operations
  //      -> graph can be saved resulting in accelerated element assembly
  //         (ifdef: allocation of new matrix, more memory, slower element assembly,
  //                 no block matrix subtraction)
  //    - one's are assigned to the diagonal entries in the ss-block (as for dirichlet conditions)
  //    properties:
  //    - splitting operation is an additional, time consuming operation
  //    - resulting system matrix is easy to solve

  // these options were deleted (ehrl 19.06.2013)
  // since the implementation was only temporary and not well tested
  // c) Saddle point system sparse matrix (sps_coupled)
  // d) Saddle point system block matrix (sps_pc)

  // number of nodes master < number of nodes slave
  // -> better results, Krylov does not work ??

  if (myrank_ == 0)
  {
    int numdofmaster = (adaptermeshtying_->master_dof_map())->num_global_elements();
    int numdofslave = (adaptermeshtying_->slave_dof_map())->num_global_elements();

    std::cout << std::endl << "number of master dof's:   " << numdofmaster << std::endl;
    std::cout << "number of slave dof's:   " << numdofslave << std::endl << std::endl;

    if (numdofmaster > numdofslave)
    {
      std::cout << "The master side is discretized by more elements than the slave side"
                << std::endl;
    }
    else
      std::cout << "The slave side is discretized by more elements than the master side"
                << std::endl;
  }

  switch (msht_)
  {
    case Inpar::FLUID::condensed_bmat:
    case Inpar::FLUID::condensed_bmat_merged:
    {
      if (!pcoupled_)
      {
        FOUR_C_THROW(
            "The system cannot be solved in a block matrix!! \n"
            "The null space does not have the right length. Fix it or use option Smat");
      }

      // slave dof rowmap
      gsdofrowmap_ = adaptermeshtying_->slave_dof_map();

      // master dof rowmap
      gmdofrowmap_ = adaptermeshtying_->master_dof_map();

      // merge dofrowmap for slave and master discretization
      gsmdofrowmap_ = Core::LinAlg::merge_map(*gmdofrowmap_, *gsdofrowmap_, false);

      // dofrowmap for discretisation without slave and master dofrowmap
      gndofrowmap_ = Core::LinAlg::split_map(*dofrowmap_, *gsmdofrowmap_);

      // map for 2x2 (uncoupled dof's & master dof's)
      mergedmap_ = Core::LinAlg::merge_map(*gndofrowmap_, *gmdofrowmap_, false);

      // std::cout << "number of n dof   " << gndofrowmap_->NumGlobalElements() << std::endl;
      // std::cout << "number of m dof   " << gmdofrowmap_->NumGlobalElements() << std::endl;
      // std::cout << "number of s dof   " << gsdofrowmap_->NumGlobalElements() << std::endl;

      // Important: right way to do it (Tobias W.)
      // allocate 2x2 solution matrix with the default block matrix strategy in order to solve the
      // reduced system memory is not allocated(1), since the matrix gets a std::shared_ptr on the
      // respective blocks of the 3x3 block matrix
      // ---------------
      // | knn  | knm' |
      // | kmn' | kmm' |
      // ---------------

      Core::LinAlg::MapExtractor mapext(*mergedmap_, gmdofrowmap_, gndofrowmap_);
      std::shared_ptr<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>
          matsolve = std::make_shared<
              Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

              mapext, mapext, 1, false, true);
      sysmatsolve_ = matsolve;

      if (solver_.params().isSublist("MueLu Parameters") or
          solver_.params().isSublist("Teko Parameters"))
      {
        // fixing length of nullspace for block matrix (solver/preconditioner ML)
        if (msht_ == Inpar::FLUID::condensed_bmat_merged)
        {
          std::string inv = "BMatMerged";
          const Core::LinAlg::Map& oldmap = *(dofrowmap_);
          const Core::LinAlg::Map& newmap = *(mergedmap_);
          Core::LinearSolver::Parameters::fix_null_space(
              inv.data(), oldmap, newmap, solver_.params());
          std::cout << std::endl;
        }
        else if (msht_ == Inpar::FLUID::condensed_bmat)
        {
          // fixing length of Inverse1 nullspace (solver/preconditioner ML)
          {
            std::string inv = "Inverse1";
            const Core::LinAlg::Map& oldmap = *(dofrowmap_);
            const Core::LinAlg::Map& newmap = Core::LinAlg::Map(matsolve->matrix(0, 0).row_map());
            Core::LinearSolver::Parameters::fix_null_space(
                inv.data(), oldmap, newmap, solver_.params().sublist("Inverse1"));
            std::cout << std::endl;
          }
          // fixing length of Inverse2 nullspace (solver/preconditioner ML)
          {
            std::string inv = "Inverse2";
            const Core::LinAlg::Map& oldmap = *(dofrowmap_);
            const Core::LinAlg::Map& newmap = Core::LinAlg::Map(matsolve->matrix(1, 1).row_map());
            Core::LinearSolver::Parameters::fix_null_space(
                inv.data(), oldmap, newmap, solver_.params().sublist("Inverse2"));
            std::cout << std::endl;
          }
        }
      }
    }
    break;
    case Inpar::FLUID::condensed_smat:
    {
      // slave dof rowmap
      gsdofrowmap_ = adaptermeshtying_->slave_dof_map();

      // master dof rowmap
      gmdofrowmap_ = adaptermeshtying_->master_dof_map();

      // merge dofrowmap for slave and master discretization
      gsmdofrowmap_ = Core::LinAlg::merge_map(*gmdofrowmap_, *gsdofrowmap_, false);

      // dofrowmap for discretisation without slave and master dofrowmap
      gndofrowmap_ = Core::LinAlg::split_map(*dofrowmap_, *gsmdofrowmap_);

      if (myrank_ == 0)
      {
        std::cout << "Condensation operation takes place in the original sysmat -> graph is saved"
                  << std::endl;
        std::cout << "The sysmat is set to zero and all parts are added -> exact" << std::endl
                  << std::endl;
      }
    }
    break;
    default:
      FOUR_C_THROW("Choose a correct mesh-tying option");
      break;
  }
}

std::shared_ptr<Core::LinAlg::SparseOperator> FLD::Meshtying::init_system_matrix() const
{
  switch (msht_)
  {
    case Inpar::FLUID::condensed_bmat:
    case Inpar::FLUID::condensed_bmat_merged:
    {
      // generate map for blockmatrix
      std::vector<std::shared_ptr<const Core::LinAlg::Map>> fluidmaps;
      fluidmaps.push_back(gndofrowmap_);
      fluidmaps.push_back(gmdofrowmap_);
      fluidmaps.push_back(gsdofrowmap_);

      Core::LinAlg::MultiMapExtractor extractor;

      extractor.setup(*dofrowmap_, fluidmaps);

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

      std::shared_ptr<Core::LinAlg::BlockSparseMatrix<FLD::Utils::InterfaceSplitStrategy>> mat;
      mat = std::make_shared<Core::LinAlg::BlockSparseMatrix<FLD::Utils::InterfaceSplitStrategy>>(
          extractor, extractor, 108, false, true);
      // nodes on the interface
      std::shared_ptr<std::set<int>> condelements =
          surfacesplitter_->conditioned_element_map(*discret_);
      mat->set_cond_elements(condelements);

      return mat;
    }
    case Inpar::FLUID::condensed_smat:
    {
      return std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap_, 108, false, true);
    }
    default:
    {
      FOUR_C_THROW("Choose a correct mesh-tying option");
      break;
    }
  }
  return nullptr;
}

const Core::LinAlg::Map* FLD::Meshtying::get_merged_map()
{
  const Core::LinAlg::Map& newmap = *(mergedmap_);

  return &newmap;
}

/*-----------------------------------------------------*/
/*  Check if there are overlapping BCs    ehrl (08/13) */
/*-----------------------------------------------------*/
void FLD::Meshtying::check_overlapping_bc(Core::LinAlg::Map& map)
{
  bool overlap = false;

  // loop over all slave row nodes of the interface
  for (int j = 0; j < adaptermeshtying_->interface()->slave_row_nodes()->num_my_elements(); ++j)
  {
    int gid = adaptermeshtying_->interface()->slave_row_nodes()->gid(j);
    Core::Nodes::Node* node = adaptermeshtying_->interface()->discret().g_node(gid);
    if (!node) FOUR_C_THROW("ERROR: Cannot find node with gid %", gid);
    auto* mtnode = static_cast<Mortar::Node*>(node);

    // check if this node's dofs are in given map
    for (int k = 0; k < mtnode->num_dof(); ++k)
    {
      int currdof = mtnode->dofs()[k];
      int lid = map.lid(currdof);

      // found slave node intersecting with given map
      if (lid >= 0)
      {
        overlap = true;
        break;
      }
    }
  }

  // print warning message to screen
  if (overlap && myrank_ == 0)
  {
    if (myrank_ == 0)
    {
      FOUR_C_THROW(
          "Slave boundary and volume flow rate boundary conditions overlap!\n"
          "This leads to an over-constraint problem setup");
    }
  }
}

/*---------------------------------------------------*/
/*  Correct slave Dirichlet value       ehrl (12/12) */
/*---------------------------------------------------*/
void FLD::Meshtying::project_master_to_slave_for_overlapping_bc(
    Core::LinAlg::Vector<double>& velnp, std::shared_ptr<const Core::LinAlg::Map> bmaps)
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> intersectionmaps;
  intersectionmaps.push_back(bmaps);
  std::shared_ptr<const Core::LinAlg::Map> gmdofrowmap = gmdofrowmap_;
  intersectionmaps.push_back(gmdofrowmap);
  std::shared_ptr<Core::LinAlg::Map> intersectionmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(intersectionmaps);

  if (intersectionmap->num_global_elements() != 0)
  {
    if (myrank_ == 0)
    {
      std::cout << "number of intersecting elements:  " << intersectionmap->num_global_elements()
                << std::endl;
      std::cout << "Overlapping boundary conditions are projected from the master to the slave side"
                << std::endl
                << std::endl;
    }

    // All dofs of the master are projected to the slave nodes. Therefore, all values of slave nodes
    // are overwritten although it is not necessary for the nodes which do not intersect with the
    // overlapping BC. In the case of Dirichlet or Dirichlet-like BC (Wormersly) new values are
    // assigned to the master side of internal interface but not to the slave side.

    // std::cout << "BEFORE projection from master to slave side" << std::endl;
    // OutputVectorSplit(velnp);

    update_slave_dof(velnp, velnp);

    // std::cout << "AFTER projection from master to slave side" << std::endl;
    // OutputVectorSplit(velnp);
  }
}

/*------------------------------------------------------------------------------*/
/*  Check if Dirichlet BC are defined on the master                ehrl (08/13) */
/*------------------------------------------------------------------------------*/
void FLD::Meshtying::dirichlet_on_master(std::shared_ptr<const Core::LinAlg::Map> bmaps)
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
  //      if(msht_ != Inpar::FLUID::no_meshtying)
  //        meshtying_->project_master_to_slave_for_overlapping_bc(velnp_, dbcmaps_->cond_map());
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
}

/*------------------------------------------------------------------*/
/*  Include Dirichlet BC in condensation operation     ehrl (08/13) */
/*------------------------------------------------------------------*/
void FLD::Meshtying::include_dirichlet_in_condensation(
    Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& veln)
{
  if (dconmaster_)
  {
    valuesdc_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap_, true);
    valuesdc_->update(1.0, velnp, 1.0);
    valuesdc_->update(-1.0, veln, 1.0);

    firstnonliniter_ = true;
  }
}


/*---------------------------------------------------*/
/*  evaluation of matrix P with potential            */
/*  mesh relocation in ALE case             vg 01/14 */
/*---------------------------------------------------*/
void FLD::Meshtying::evaluate_with_mesh_relocation(
    std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp)
{
  // get ALE discretization
  std::shared_ptr<Core::FE::Discretization> aledis = Global::Problem::instance()->get_dis("ale");

  // call mortar evaluate routine including mesh correction
  adaptermeshtying_->evaluate_with_mesh_relocation(
      discret_, aledis, dispnp, discret_->get_comm(), true);
}

/*---------------------------------------------------*/
/*  Prepare Meshtying                    wirtz 02/16 */
/*---------------------------------------------------*/
void FLD::Meshtying::prepare_meshtying(std::shared_ptr<Core::LinAlg::SparseOperator>& sysmat,
    Core::LinAlg::Vector<double>& residual, Core::LinAlg::Vector<double>& velnp,
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>& shapederivatives)
{
  prepare_meshtying_system(sysmat, residual, velnp);
  multifield_split(sysmat);

  if (shapederivatives != nullptr)
  {
    condensation_operation_block_matrix_shape(*shapederivatives);
    multifield_split_shape(shapederivatives);
  }
}


/*---------------------------------------------------*/
/*  Prepare Meshtying system            ehrl (04/11) */
/*---------------------------------------------------*/
void FLD::Meshtying::prepare_meshtying_system(
    const std::shared_ptr<Core::LinAlg::SparseOperator>& sysmat,
    Core::LinAlg::Vector<double>& residual, Core::LinAlg::Vector<double>& velnp)
{
  switch (msht_)
  {
    case Inpar::FLUID::condensed_bmat:
    case Inpar::FLUID::condensed_bmat_merged:
      condensation_block_matrix(sysmat, residual, velnp);
      break;
    case Inpar::FLUID::condensed_smat:
      condensation_sparse_matrix(sysmat, residual, velnp);
      break;
    default:
      FOUR_C_THROW("Meshtying algorithm not recognized!");
      break;
  }
}


/*---------------------------------------------------*/
/*---------------------------------------------------*/
void FLD::Meshtying::apply_pt_to_residual(Core::LinAlg::SparseOperator& sysmat,
    Core::LinAlg::Vector<double>& residual, Core::LinAlg::KrylovProjector& projector)
{
  // define residual vector for case of block matrix
  std::shared_ptr<Core::LinAlg::Vector<double>> res =
      std::make_shared<Core::LinAlg::Vector<double>>(*mergedmap_, true);

  // split original residual vector
  split_vector_based_on3x3(residual, *res);

  // apply projector
  *res = projector.to_reduced(*res);

  // export residual back to original vector
  Core::LinAlg::export_to(*res, residual);
}


/*-------------------------------------------------------*/
/*  Krylov projection                      ehrl (04/11)  */
/*-------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::Meshtying::adapt_krylov_projector(
    std::shared_ptr<Core::LinAlg::Vector<double>> vec)
{
  if (pcoupled_)
  {
    // Remove slave nodes from vec
    Core::LinAlg::Vector<double> fm_slave(*gsdofrowmap_, true);
    // add fm subvector to feffnew
    Core::LinAlg::export_to(fm_slave, *vec);

    switch (msht_)
    {
      case Inpar::FLUID::condensed_bmat:
      case Inpar::FLUID::condensed_bmat_merged:
      {
        std::shared_ptr<Core::LinAlg::Vector<double>> vec_mesht =
            std::make_shared<Core::LinAlg::Vector<double>>(*mergedmap_, true);
        split_vector_based_on3x3(*vec, *vec_mesht);
        vec = vec_mesht;
      }
      break;
      case Inpar::FLUID::condensed_smat:
        break;
      default:
        FOUR_C_THROW("Krylov projection not supported for this meshtying option.");
        break;
    }
  }
  return vec;
}

/*-------------------------------------------------------*/
/*  solve mesh-tying system                ehrl (04/11)  */
/* (including ALE case   vg 01/14)                       */
/*-------------------------------------------------------*/
void FLD::Meshtying::solve_meshtying(Core::LinAlg::Solver& solver,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& sysmat,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& incvel,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& residual,
    Core::LinAlg::Vector<double>& velnp, const int itnum, Core::LinAlg::SolverParams& solver_params)
{
  solver_params.refactor = true;
  solver_params.reset = itnum == 1;

  // time measurement
  TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  3)   Solve meshtying system");

  switch (msht_)
  {
    case Inpar::FLUID::condensed_bmat:
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> res =
          std::make_shared<Core::LinAlg::Vector<double>>(*mergedmap_, true);
      std::shared_ptr<Core::LinAlg::Vector<double>> inc =
          std::make_shared<Core::LinAlg::Vector<double>>(*mergedmap_, true);

      std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmatnew =
          std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat);
      std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmatsolve =
          std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmatsolve_);

      {
        TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  3.1)   - Preparation");

        split_vector_based_on3x3(*residual, *res);
        // assign blocks to the solution matrix
        sysmatsolve->assign(0, 0, Core::LinAlg::DataAccess::Share, sysmatnew->matrix(0, 0));
        sysmatsolve->assign(0, 1, Core::LinAlg::DataAccess::Share, sysmatnew->matrix(0, 1));
        sysmatsolve->assign(1, 0, Core::LinAlg::DataAccess::Share, sysmatnew->matrix(1, 0));
        sysmatsolve->assign(1, 1, Core::LinAlg::DataAccess::Share, sysmatnew->matrix(1, 1));
        sysmatsolve->complete();
      }

      {
        TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  3.2)   - Solve");
        solver_.solve(sysmatsolve, inc, res, solver_params);
      }

      {
        TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  3.3)   - Update");

        // Export the computed increment to the global increment
        Core::LinAlg::export_to(*inc, *incvel);
        Core::LinAlg::export_to(*res, *residual);

        // compute and update slave dof's
        update_slave_dof(*incvel, velnp);
      }
    }
    break;
    case Inpar::FLUID::condensed_bmat_merged:
    {
      std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmatnew =
          std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat);
      std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> sysmatsolve =
          std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmatsolve_);

      std::shared_ptr<Core::LinAlg::Vector<double>> res = nullptr;
      std::shared_ptr<Core::LinAlg::Vector<double>> inc = nullptr;

      std::shared_ptr<Core::LinAlg::SparseMatrix> mergedmatrix = nullptr;

      res = std::make_shared<Core::LinAlg::Vector<double>>(*mergedmap_, true);
      inc = std::make_shared<Core::LinAlg::Vector<double>>(*mergedmap_, true);

      mergedmatrix = std::make_shared<Core::LinAlg::SparseMatrix>(*mergedmap_, 108, false, true);

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

        solver_.solve(mergedmatrix, inc, res, solver_params);

        Core::LinAlg::export_to(*inc, *incvel);
        Core::LinAlg::export_to(*res, *residual);
        // compute and update slave dof's
        update_slave_dof(*incvel, velnp);
      }
    }
    break;
    case Inpar::FLUID::condensed_smat:
    {
      {
        TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  3.3)   - Solve");
        solver_.solve(sysmat, incvel, residual, solver_params);
      }

      {
        TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  3.3)   - Update");
        // compute and update slave dof's
        update_slave_dof(*incvel, velnp);
      }
    }
    break;
    default:
      FOUR_C_THROW("");
      break;
  }
}


/*-------------------------------------------------------*/
/*  Condensation Sparse Matrix              ehrl (04/11) */
/* (including ALE case   vg 01/14)                       */
/*-------------------------------------------------------*/
void FLD::Meshtying::condensation_sparse_matrix(
    const std::shared_ptr<Core::LinAlg::SparseOperator>& sysmat,
    Core::LinAlg::Vector<double>& residual, Core::LinAlg::Vector<double>& velnp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  2)   Condensation sparse matrix");

  /**********************************************************************/
  /* Split sysmat and residual                                          */
  /**********************************************************************/

  // container for split matrix and vector
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> splitmatrix;
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitres(3);
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitvel(3);

  split_matrix(sysmat, splitmatrix);
  split_vector(residual, splitres);
  split_vector(velnp, splitvel);

  /**********************************************************************/
  /* Condensate sparse matrix                                           */
  /**********************************************************************/

  condensation_operation_sparse_matrix(*sysmat, residual, *splitmatrix, splitres, splitvel);
}


/*-------------------------------------------------------*/
/*  Condensation Block Matrix               ehrl (04/11) */
/* (including ALE case   vg 01/14)                       */
/*-------------------------------------------------------*/
void FLD::Meshtying::condensation_block_matrix(
    const std::shared_ptr<Core::LinAlg::SparseOperator>& sysmat,
    Core::LinAlg::Vector<double>& residual, Core::LinAlg::Vector<double>& velnp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  2)   Condensation block matrix");

  /**********************************************************************/
  /* Split residual into 3 subvectors                                   */
  /**********************************************************************/

  // container for split residual vector
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitres(3);
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitvel(3);
  split_vector(residual, splitres);
  split_vector(velnp, splitvel);

  /**********************************************************************/
  /* Condensate blockmatrix                                             */
  /**********************************************************************/

  condensation_operation_block_matrix(sysmat, residual, splitres, splitvel);
}


/*------------------------------------------------------------------------------------------------------------------------------*
 | split sparse global system matrix into 3x3 block sparse matrix associated with interior, master,
 and slave dofs   fang 08/15 |
 *------------------------------------------------------------------------------------------------------------------------------*/
void FLD::Meshtying::split_matrix(
    std::shared_ptr<Core::LinAlg::SparseOperator>
        matrix,  //!< original sparse global system matrix before split
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>&
        splitmatrix  //!< resulting block sparse matrix after split
)
{
  TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  2.1)   - Split Matrix");

  // initialize map extractor for matrix splitting
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> fluidmaps;
  fluidmaps.push_back(gndofrowmap_);
  fluidmaps.push_back(gmdofrowmap_);
  fluidmaps.push_back(gsdofrowmap_);
  Core::LinAlg::MultiMapExtractor extractor(*dofrowmap_, fluidmaps);
  extractor.check_for_valid_map_extractor();

  // perform matrix splitting
  // -------------------
  // | knn | knm | kns |
  // | kmn | kmm | kms |
  // | ksn | ksm | kss |
  // -------------------
  splitmatrix = Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
      *std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(matrix), extractor, extractor);

  // finalize resulting block sparse matrix
  splitmatrix->complete();
}

/*-------------------------------------------------------*/
/*  Split Vector                           ehrl (04/11)  */
/*-------------------------------------------------------*/
void FLD::Meshtying::split_vector(Core::LinAlg::Vector<double>& vector,
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
}


/*-------------------------------------------------------*/
/*-------------------------------------------------------*/
void FLD::Meshtying::split_vector_based_on3x3(
    Core::LinAlg::Vector<double>& orgvector, Core::LinAlg::Vector<double>& vectorbasedon2x2)
{
  // container for split residual vector
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitvector(3);

  split_vector(orgvector, splitvector);
  // build up the reduced residual
  Core::LinAlg::export_to(*(splitvector[0]), vectorbasedon2x2);
  Core::LinAlg::export_to(*(splitvector[1]), vectorbasedon2x2);
}


/*-------------------------------------------------------*/
/*  Condensation operation sparse matrix    ehrl (04/11) */
/* (including ALE case   vg 01/14)                       */
/*-------------------------------------------------------*/
void FLD::Meshtying::condensation_operation_sparse_matrix(
    Core::LinAlg::SparseOperator& sysmat,    ///> sysmat established by the element routine
    Core::LinAlg::Vector<double>& residual,  ///> residual established by the element routine
    Core::LinAlg::BlockSparseMatrixBase& splitmatrix,  ///> container with split original sysmat
    const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
        splitres,  ///> container with split original residual
    const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
        splitvel  ///> container with split velocity vector
)
{
  TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  2.3)   - Condensation Operation");

  /**********************************************************************/
  /* Build the final sysmat                                             */
  /**********************************************************************/

  // ---------------------        ------------------
  // | nn | nm | ns | 0  |        | nn  | nm' | 0  |
  // | mn | mm | ms | D  |   =    | mn' | mm' | 0  |
  // | sn | sm | ss | -M |        |  0  |  0  | 1  |
  // |  0 | DT |-MT | 0  |        ------------------
  // ---------------------
  // solved system
  // ------------------
  // | nn  | nm' | 0  |
  // | mn' | mm' | 0  |
  // |  0  |  0  | 1  |
  // ------------------

  // splitmatrix
  // -------------------
  // | knn | knm | kns |
  // | kmn | kmm | kms |
  // | ksn | ksm | kss |
  // -------------------

  // the sysmat is manipulated directly with out changing the graph
  // -> subtract blocks to get zeros in the slave blocks
  // -> graph is saved -> fast element assembly
  // -> less memory is needed since everything is done with the original system matrix
  // -> is it dangerous to subtract blocks to get zeros
  //

  // Dirichlet or Dirichlet-like condition on the master side of the internal interface:
  // First time step:
  // coupling condition: u_s - u_m = delta u_m^D
  // instead of          u_s - u_m = 0
  //
  // this has to be considered in the condensation and in update process
  std::shared_ptr<Core::LinAlg::Vector<double>> dcnm = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> dcmm = nullptr;
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitdcmaster(3);

  if (dconmaster_ and firstnonliniter_)
  {
    dcnm = std::make_shared<Core::LinAlg::Vector<double>>(*gndofrowmap_, true);
    dcmm = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofrowmap_, true);

    split_vector(*valuesdc_, splitdcmaster);
  }

  // get transformation matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> P = adaptermeshtying_->get_mortar_matrix_p();

  /**********************************************************************/
  /* Condensation operation for the sysmat                              */
  /**********************************************************************/

  // the sysmat is manipulated directly with out changing the graph
  // (subtract blocks to get zeros in the slave blocks)

  sysmat.un_complete();

  /*--------------------------------------------------------------------*/
  // Part nn
  /*--------------------------------------------------------------------*/
  sysmat.add(splitmatrix.matrix(0, 0), false, 1.0, 0.0);

  /*--------------------------------------------------------------------*/
  // Part nm
  /*--------------------------------------------------------------------*/
  // knm: add kns*P
  Core::LinAlg::SparseMatrix knm_mod(*gndofrowmap_, 100);
  Core::LinAlg::matrix_add(splitmatrix.matrix(0, 1), false, 1.0, knm_mod, 1.0);
  std::shared_ptr<Core::LinAlg::SparseMatrix> knm_add =
      matrix_multiply(splitmatrix.matrix(0, 2), false, *P, false, false, false, true);
  Core::LinAlg::matrix_add(*knm_add, false, 1.0, knm_mod, 1.0);
  knm_mod.complete(splitmatrix.matrix(0, 1).domain_map(), splitmatrix.matrix(0, 1).row_map());

  sysmat.add(knm_mod, false, 1.0, 1.0);

  if (dconmaster_ and firstnonliniter_) knm_add->multiply(false, *(splitdcmaster[1]), *dcnm);

  /*--------------------------------------------------------------------*/
  // Part mn
  /*--------------------------------------------------------------------*/
  // kmn: add P^T*ksn
  Core::LinAlg::SparseMatrix kmn_mod(*gmdofrowmap_, 100);
  Core::LinAlg::matrix_add(splitmatrix.matrix(1, 0), false, 1.0, kmn_mod, 1.0);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmn_add =
      matrix_multiply(*P, true, splitmatrix.matrix(2, 0), false, false, false, true);
  Core::LinAlg::matrix_add(*kmn_add, false, 1.0, kmn_mod, 1.0);
  kmn_mod.complete(splitmatrix.matrix(1, 0).domain_map(), splitmatrix.matrix(1, 0).row_map());

  sysmat.add(kmn_mod, false, 1.0, 1.0);

  /*--------------------------------------------------------------------*/
  // Part mm
  /*--------------------------------------------------------------------*/
  // kms: add P^T*kss, kmm: add kms*P + kmm
  Core::LinAlg::SparseMatrix kmm_mod(*gmdofrowmap_, 100);
  Core::LinAlg::matrix_add(splitmatrix.matrix(1, 1), false, 1.0, kmm_mod, 1.0);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kms =
      matrix_multiply(*P, true, splitmatrix.matrix(2, 2), false, false, false, true);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmm_add =
      matrix_multiply(*kms, false, *P, false, false, false, true);
  Core::LinAlg::matrix_add(*kmm_add, false, 1.0, kmm_mod, 1.0);
  kmm_mod.complete(splitmatrix.matrix(1, 1).domain_map(), splitmatrix.matrix(1, 1).row_map());

  sysmat.add(kmm_mod, false, 1.0, 1.0);

  if (dconmaster_ and firstnonliniter_) kmm_mod.multiply(false, *(splitdcmaster[1]), *dcmm);

  std::shared_ptr<Core::LinAlg::Vector<double>> ones =
      std::make_shared<Core::LinAlg::Vector<double>>(*gsdofrowmap_);
  std::shared_ptr<Core::LinAlg::SparseMatrix> onesdiag;
  ones->put_scalar(1.0);
  onesdiag = std::make_shared<Core::LinAlg::SparseMatrix>(*ones);
  onesdiag->complete();

  sysmat.add(*onesdiag, false, 1.0, 1.0);

  sysmat.complete();

  //*************************************************
  //  condensation operation for the residual
  //*************************************************
  // splitres[ii]
  // r_n [0]
  // r_m [1]
  // r_s [2]

  std::shared_ptr<Core::LinAlg::Vector<double>> resnew =
      std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap_, true);

  // r_m: add P^T*r_s
  Core::LinAlg::Vector<double> fm_mod(*gmdofrowmap_, true);
  P->multiply(true, *(splitres[2]), fm_mod);

  // r_m: add P^T*K_ss*vp_i^s
  Core::LinAlg::Vector<double> fm_mod_ss(*gmdofrowmap_, true);
  kms->multiply(false, *(splitvel[2]), fm_mod_ss);
  fm_mod.update(1.0, fm_mod_ss, 1.0);

  // r_m: subtract P^T*K_ss*P*vp_i^m
  Core::LinAlg::Vector<double> fm_mod_mm(*gmdofrowmap_, true);
  kmm_add->multiply(false, *(splitvel[1]), fm_mod_mm);
  fm_mod.update(-1.0, fm_mod_mm, 1.0);

  // r_m: insert Dirichlet boundary conditions
  if (dconmaster_ and firstnonliniter_) fm_mod.update(-1.0, *dcmm, 1.0);

  // export additions to r_m subvector to r_new
  Core::LinAlg::Vector<double> fm_modexp(*dofrowmap_);
  Core::LinAlg::export_to(fm_mod, fm_modexp);
  resnew->update(1.0, fm_modexp, 1.0);

  // export r_m subvector to r_new
  Core::LinAlg::Vector<double> fmexp(*dofrowmap_);
  Core::LinAlg::export_to(*(splitres[1]), fmexp);
  resnew->update(1.0, fmexp, 1.0);


  // r_n: add K_ns*vp_i^s
  Core::LinAlg::SparseMatrix knm(splitmatrix.matrix(0, 2));
  // Core::LinAlg::SparseMatrix& knm = splitmatrix->Matrix(0,2);
  Core::LinAlg::Vector<double> fn_mod(*gndofrowmap_, true);
  knm.multiply(false, *(splitvel[2]), fn_mod);

  // r_n: subtrac K_ns*P*vp_i^m
  Core::LinAlg::Vector<double> fn_mod_nm(*gndofrowmap_, true);
  knm_add->multiply(false, *(splitvel[1]), fn_mod_nm);
  fn_mod.update(-1.0, fn_mod_nm, 1.0);

  // export additions to r_n subvector to r_new
  Core::LinAlg::Vector<double> fn_modexp(*dofrowmap_);
  Core::LinAlg::export_to(fn_mod, fn_modexp);
  resnew->update(1.0, fn_modexp, 1.0);

  // export r_n subvector to r_new
  Core::LinAlg::Vector<double> fnexp(*dofrowmap_);
  Core::LinAlg::export_to(*(splitres[0]), fnexp);
  resnew->update(1.0, fnexp, 1.0);

  if (dconmaster_ and firstnonliniter_)
  {
    Core::LinAlg::Vector<double> fn_exp(*dofrowmap_, true);
    Core::LinAlg::export_to(*dcnm, fn_exp);
    resnew->update(-1.0, fn_exp, 1.0);
  }

  residual.update(1.0, *resnew, 0.0);
}

/*-------------------------------------------------------*/
/*  Condensation operation block matrix     ehrl (04/11) */
/* (including ALE case   vg 01/14)                       */
/*-------------------------------------------------------*/
void FLD::Meshtying::condensation_operation_block_matrix(
    const std::shared_ptr<Core::LinAlg::SparseOperator>& sysmat,
    Core::LinAlg::Vector<double>& residual,
    const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>& splitres,
    const std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>& splitvel)
{
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

  if (dconmaster_ and firstnonliniter_)
  {
    dcnm = std::make_shared<Core::LinAlg::Vector<double>>(*gndofrowmap_, true);
    dcmm = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofrowmap_, true);

    split_vector(*valuesdc_, splitdcmaster);
  }

  // get transformation matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> P = adaptermeshtying_->get_mortar_matrix_p();

  /*--------------------------------------------------------------------*/
  // block nm
  /*--------------------------------------------------------------------*/
  // compute modification for block nm
  std::shared_ptr<Core::LinAlg::SparseMatrix> knm_mod =
      matrix_multiply(sysmatnew->matrix(0, 2), false, *P, false, false, false, true);

  // Add transformation matrix to nm
  sysmatnew->matrix(0, 1).un_complete();
  Core::LinAlg::matrix_add(*knm_mod, false, 1.0, sysmatnew->matrix(0, 1), 1.0);

  if (dconmaster_ and firstnonliniter_) knm_mod->multiply(false, *(splitdcmaster[1]), *dcnm);

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

  if (dconmaster_ and firstnonliniter_) kmm_mod->multiply(false, *(splitdcmaster[1]), *dcmm);

  // complete matrix
  sysmatnew->complete();

  //*************************************************
  //  condensation operation for the residual
  //*************************************************
  // r_m: add P^T*r_s
  Core::LinAlg::Vector<double> fm_mod(*gmdofrowmap_, true);
  P->multiply(true, *(splitres[2]), fm_mod);

  // r_m: add P^T*K_ss*vp_i^s
  Core::LinAlg::Vector<double> fm_mod_ss(*gmdofrowmap_, true);
  kss_mod->multiply(false, *(splitvel[2]), fm_mod_ss);
  fm_mod.update(1.0, fm_mod_ss, 1.0);

  // r_m: subtract P^T*K_ss*P*vp_i^m
  Core::LinAlg::Vector<double> fm_mod_mm(*gmdofrowmap_, true);
  kmm_mod->multiply(false, *(splitvel[1]), fm_mod_mm);
  fm_mod.update(-1.0, fm_mod_mm, 1.0);

  // r_m: insert Dirichlet boundary conditions
  if (dconmaster_ and firstnonliniter_) fm_mod.update(-1.0, *dcmm, 1.0);

  // export and add r_m subvector to residual
  Core::LinAlg::Vector<double> fm_modexp(*dofrowmap_);
  Core::LinAlg::export_to(fm_mod, fm_modexp);
  residual.update(1.0, fm_modexp, 1.0);


  // r_n: add K_ns*vp_i^s
  Core::LinAlg::SparseMatrix knm(sysmatnew->matrix(0, 2));
  // Core::LinAlg::SparseMatrix& knm = sysmatnew->Matrix(0,2);
  Core::LinAlg::Vector<double> fn_mod(*gndofrowmap_, true);
  knm.multiply(false, *(splitvel[2]), fn_mod);

  // r_n: subtract K_ns*P*vp_i^m
  Core::LinAlg::Vector<double> fn_mod_nm(*gndofrowmap_, true);
  knm_mod->multiply(false, *(splitvel[1]), fn_mod_nm);
  fn_mod.update(-1.0, fn_mod_nm, 1.0);

  // export and add r_n subvector to residual
  Core::LinAlg::Vector<double> fn_modexp(*dofrowmap_);
  Core::LinAlg::export_to(fn_mod, fn_modexp);
  residual.update(1.0, fn_modexp, 1.0);

  if (dconmaster_ and firstnonliniter_)
  {
    Core::LinAlg::Vector<double> fn_exp(*dofrowmap_, true);
    Core::LinAlg::export_to(*dcnm, fn_exp);
    residual.update(-1.0, fn_exp, 1.0);
  }

  // export r_s = zero to residual
  Core::LinAlg::Vector<double> fs_mod(*gsdofrowmap_, true);
  Core::LinAlg::export_to(fs_mod, residual);
}

/*-------------------------------------------------------*/
/*  Compute and update Slave DOF's          ehrl (04/11) */
/* (including ALE case   vg 01/14)                       */
/*-------------------------------------------------------*/
void FLD::Meshtying::update_slave_dof(
    Core::LinAlg::Vector<double>& inc, Core::LinAlg::Vector<double>& velnp)
{
  TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  3.4)   - Update slave DOF");

  // get dof row map
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // split incremental and velocity-pressure vector
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitinc(3);
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitvel(3);
  split_vector(inc, splitinc);
  split_vector(velnp, splitvel);

  // Dirichlet or Dirichlet-like condition on the master side of the internal interface:
  // First time step:
  // coupling condition: u_s - u_m = delta u_m^D
  // instead of          u_s - u_m = 0
  //
  // this has to be considered in the condensation and in update process

  // split vector containing Dirichlet boundary conditions, if any
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitdcmaster(3);
  if (dconmaster_ and firstnonliniter_) split_vector(*valuesdc_, splitdcmaster);

  // get transformation matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> P = adaptermeshtying_->get_mortar_matrix_p();

  // define new incremental vector
  std::shared_ptr<Core::LinAlg::Vector<double>> incnew =
      std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // delta_vp^s: add P*delta_vp^m
  Core::LinAlg::Vector<double> fs_mod(*gsdofrowmap_, true);
  P->multiply(false, *(splitinc[1]), fs_mod);

  // delta_vp^s: subtract vp_i^s
  fs_mod.update(-1.0, *(splitvel[2]), 1.0);

  // delta_vp^s: add P*vp_i^m
  Core::LinAlg::Vector<double> fs_mod_m(*gsdofrowmap_, true);
  P->multiply(false, *(splitvel[1]), fs_mod_m);
  fs_mod.update(1.0, fs_mod_m, 1.0);

  // set Dirichlet boundary conditions, if any
  if (dconmaster_ and firstnonliniter_)
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
  if (dconmaster_ and firstnonliniter_) firstnonliniter_ = false;

  // define incremental vector to new incremental vector
  inc.update(1.0, *incnew, 0.0);
}

/*-------------------------------------------------------*/
/*  Output: vector                         ehrl (04/11)  */
/*-------------------------------------------------------*/
void FLD::Meshtying::output_vector_split(Core::LinAlg::Vector<double>& vector)
{
  std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> splitvector(3);
  split_vector(vector, splitvector);

  // std::cout << "vector " << std::endl << *vector << std::endl << std::endl;

  std::cout.precision(20);
  // std::cout << "Teil fn " << std::endl << *(splitvector[0]) << std::endl << std::endl;
  // std::cout << "Teil fm: " << std::endl << *(splitvector[1]->Print()) << std::endl << std::endl;
  // std::cout << "Teil fs: " << std::endl << *(splitvector[2]) << std::endl;
}

/*-------------------------------------------------------*/
/*  Output: Analyze matrix                 ehrl (11/11)  */
/*-------------------------------------------------------*/
void FLD::Meshtying::analyze_matrix(Core::LinAlg::SparseMatrix& sparsematrix)
{
  double localmatrixentries = 0.0;
  double parmatrixentries = 0.0;
  {
    // number of row elements
    const int numdofrows = sparsematrix.row_map().num_my_elements();

    for (int i = 0; i < numdofrows; ++i)
    {
      // max. number of non-zero values
      int maxnumentries = sparsematrix.max_num_entries();
      int numOfNonZeros = 0;
      std::vector<int> indices(maxnumentries, 0);
      std::vector<double> values(maxnumentries, 0.0);

      sparsematrix.extract_my_row_copy(
          i, maxnumentries, numOfNonZeros, values.data(), indices.data());

      for (int ii = 0; ii < numOfNonZeros; ii++)
      {
        localmatrixentries += values[ii];
      }
    }

    parmatrixentries = Core::Communication::sum_all(localmatrixentries, discret_->get_comm());
  }
  double normfrob = sparsematrix.norm_frobenius();
  double norminf = sparsematrix.NormInf();
  double normone = sparsematrix.norm_one();
  double matrixsize = sparsematrix.num_global_rows() * sparsematrix.num_global_cols();
  double nonzero = sparsematrix.num_global_nonzeros();

  if (myrank_ == 0)
  {
    {
      std::cout.precision(20);
      std::cout << std::endl;
      std::cout << "-------------- Analyze Matrix ----------------------" << std::endl;
      std::cout << "| global matrix size:          " << matrixsize << std::endl;
      std::cout << "| number of global non-zeros:  " << nonzero << std::endl;
      std::cout << "| Matrix norm (Frobenius):     " << normfrob << std::endl;
      std::cout << "| Matrix norm (Inf):           " << norminf << std::endl;
      std::cout << "| Matrix norm (One):           " << normone << std::endl;
      std::cout << "| sum of all matrix entries:   " << parmatrixentries << std::endl;
      std::cout << "----------------------------------------------------" << std::endl;
    }
  }
}  // end AnalyzeMatrix()

// -------------------------------------------------------------------
// check absolute velinc norm                           ehrl   11/2011
// -------------------------------------------------------------------
/*
void FLD::FluidImplicitTimeInt::PrintAbsoluteL2Norm(std::shared_ptr<Core::LinAlg::Vector<double>>&
vector)
{
  double incvelnorm_L2;
  double incprenorm_L2;

  std::shared_ptr<Core::LinAlg::Vector<double>> onlyvel =
velpressplitter_->extract_other_vector(vector); onlyvel->Norm2(&incvelnorm_L2);

  std::shared_ptr<Core::LinAlg::Vector<double>> onlypre =
velpressplitter_->extract_cond_vector(vector); onlypre->Norm2(&incprenorm_L2);

  printf("+------------+-------------------+--------------+\n");
  printf("| %10.14E   | %10.14E   |",
         incvelnorm_L2,incprenorm_L2);
  printf(")\n");
  printf("+------------+-------------------+--------------+\n");

  return;
}
  */

/*-------------------------------------------------------*/
/*  Set the flag for multifield problems     wirtz 01/16 */
/*                                                       */
/*-------------------------------------------------------*/
void FLD::Meshtying::is_multifield(
    std::shared_ptr<std::set<int>> condelements,        ///< conditioned elements of fluid
    const Core::LinAlg::MultiMapExtractor& domainmaps,  ///< domain maps for split of fluid matrix
    const Core::LinAlg::MultiMapExtractor& rangemaps,   ///< range maps for split of fluid matrix
    std::shared_ptr<std::set<int>> condelements_shape,  ///< conditioned elements
    const Core::LinAlg::MultiMapExtractor&
        domainmaps_shape,  ///< domain maps for split of shape deriv. matrix
    const Core::LinAlg::MultiMapExtractor&
        rangemaps_shape,  ///< domain maps for split of shape deriv. matrix
    bool splitmatrix,     ///< flag for split of matrices
    bool ismultifield     ///< flag for multifield problems
)
{
  multifield_condelements_ = condelements;
  multifield_domainmaps_ = domainmaps;
  multifield_rangemaps_ = rangemaps;
  multifield_condelements_shape_ = condelements_shape;
  multifield_domainmaps_shape_ = domainmaps_shape;
  multifield_rangemaps_shape_ = rangemaps_shape;
  multifield_splitmatrix_ = splitmatrix;
  is_multifield_ = ismultifield;
}

/*-------------------------------------------------------*/
/*  Use the split of the fluid mesh tying for the sysmat */
/*                                           wirtz 01/16 */
/*-------------------------------------------------------*/
void FLD::Meshtying::msht_split(std::shared_ptr<Core::LinAlg::SparseOperator>& sysmat,
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>& shapederivatives)
{
  if (is_multifield_)
  {
    // generate map for blockmatrix
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> fluidmaps;
    fluidmaps.push_back(gndofrowmap_);
    fluidmaps.push_back(gmdofrowmap_);
    fluidmaps.push_back(gsdofrowmap_);

    Core::LinAlg::MultiMapExtractor extractor;

    extractor.setup(*dofrowmap_, fluidmaps);

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

    std::shared_ptr<Core::LinAlg::BlockSparseMatrix<FLD::Utils::InterfaceSplitStrategy>> mat;
    mat = std::make_shared<Core::LinAlg::BlockSparseMatrix<FLD::Utils::InterfaceSplitStrategy>>(
        extractor, extractor, 108, false, true);
    // nodes on the interface
    std::shared_ptr<std::set<int>> condelements =
        surfacesplitter_->conditioned_element_map(*discret_);
    mat->set_cond_elements(condelements);

    sysmat = mat;

    if (shapederivatives != nullptr) msht_split_shape(shapederivatives);
  }
}

/*-------------------------------------------------------*/
/*  Use the split of the fluid mesh tying for the shape  */
/*  derivatives                              wirtz 01/16 */
/*-------------------------------------------------------*/
void FLD::Meshtying::msht_split_shape(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>& shapederivatives)
{
  // generate map for blockmatrix
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> fluidmaps;
  fluidmaps.push_back(gndofrowmap_);
  fluidmaps.push_back(gmdofrowmap_);
  fluidmaps.push_back(gsdofrowmap_);

  Core::LinAlg::MultiMapExtractor extractor;

  extractor.setup(*dofrowmap_, fluidmaps);

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

  std::shared_ptr<Core::LinAlg::BlockSparseMatrix<FLD::Utils::InterfaceSplitStrategy>> mat;
  mat = std::make_shared<Core::LinAlg::BlockSparseMatrix<FLD::Utils::InterfaceSplitStrategy>>(
      extractor, extractor, 108, false, true);
  // nodes on the interface
  std::shared_ptr<std::set<int>> condelements =
      surfacesplitter_->conditioned_element_map(*discret_);
  mat->set_cond_elements(condelements);

  shapederivatives = mat;
}

/*-------------------------------------------------------*/
/*  Use the split of the multifield problem for the      */
/*  sysmat                                   wirtz 01/16 */
/*-------------------------------------------------------*/
void FLD::Meshtying::multifield_split(std::shared_ptr<Core::LinAlg::SparseOperator>& sysmat)
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

    if (multifield_splitmatrix_)
    {
      Core::LinAlg::MapExtractor extractor(
          *multifield_domainmaps_.full_map(), multifield_domainmaps_.map(1));
      std::shared_ptr<Core::LinAlg::BlockSparseMatrix<FLD::Utils::InterfaceSplitStrategy>> mat =
          Core::LinAlg::split_matrix<FLD::Utils::InterfaceSplitStrategy>(
              *mergedmatrix, extractor, extractor);
      mat->set_cond_elements(multifield_condelements_);
      mat->complete();

      sysmat = mat;
    }
    else
    {
      sysmat = mergedmatrix;
    }
  }
}

/*-------------------------------------------------------*/
/*  Use the split of the multifield problem for the      */
/*  shape derivatives                        wirtz 01/16 */
/*-------------------------------------------------------*/
void FLD::Meshtying::multifield_split_shape(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>& shapederivatives)
{
  if (is_multifield_)
  {
    Core::LinAlg::Vector<double> ones(shapederivatives->matrix(2, 2).row_map());
    ones.put_scalar(1.0);
    Core::LinAlg::SparseMatrix onesdiag(ones);
    onesdiag.complete();

    shapederivatives->matrix(0, 2).un_complete();
    shapederivatives->matrix(0, 2).zero();

    shapederivatives->matrix(1, 2).un_complete();
    shapederivatives->matrix(1, 2).zero();

    shapederivatives->matrix(2, 2).un_complete();
    shapederivatives->matrix(2, 2).zero();
    Core::LinAlg::matrix_add(onesdiag, false, 1.0, shapederivatives->matrix(2, 2), 1.0);

    shapederivatives->matrix(2, 0).un_complete();
    shapederivatives->matrix(2, 0).zero();

    shapederivatives->matrix(2, 1).un_complete();
    shapederivatives->matrix(2, 1).zero();

    shapederivatives->complete();

    std::shared_ptr<Core::LinAlg::SparseMatrix> mergedshapederivatives = shapederivatives->merge();

    Core::LinAlg::MapExtractor extractor(
        *multifield_domainmaps_shape_.full_map(), multifield_domainmaps_shape_.map(1));
    std::shared_ptr<Core::LinAlg::BlockSparseMatrix<FLD::Utils::InterfaceSplitStrategy>>
        matshapederivatives = Core::LinAlg::split_matrix<FLD::Utils::InterfaceSplitStrategy>(
            *mergedshapederivatives, extractor, extractor);
    matshapederivatives->set_cond_elements(multifield_condelements_shape_);
    matshapederivatives->complete();
    shapederivatives = matshapederivatives;
  }
}

/*-------------------------------------------------------*/
/*  Prepare condensation of the shape derivatives        */
/*                                           wirtz 01/16 */
/*-------------------------------------------------------*/
void FLD::Meshtying::condensation_operation_block_matrix_shape(
    Core::LinAlg::BlockSparseMatrixBase& shapederivatives)
{
  TEUCHOS_FUNC_TIME_MONITOR("Meshtying:  2.1)   - Condensation Operation");

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

  if (dconmaster_ and firstnonliniter_)
  {
    dcnm = std::make_shared<Core::LinAlg::Vector<double>>(*gndofrowmap_, true);
    dcmm = std::make_shared<Core::LinAlg::Vector<double>>(*gmdofrowmap_, true);

    split_vector(*valuesdc_, splitdcmaster);
  }

  // get transformation matrix
  std::shared_ptr<Core::LinAlg::SparseMatrix> P = adaptermeshtying_->get_mortar_matrix_p();

  /*--------------------------------------------------------------------*/
  // block nm
  /*--------------------------------------------------------------------*/
  // compute modification for block nm
  std::shared_ptr<Core::LinAlg::SparseMatrix> knm_mod =
      matrix_multiply(shapederivatives.matrix(0, 2), false, *P, false, false, false, true);

  // Add transformation matrix to nm
  shapederivatives.matrix(0, 1).un_complete();
  Core::LinAlg::matrix_add(*knm_mod, false, 1.0, shapederivatives.matrix(0, 1), 1.0);

  if (dconmaster_ and firstnonliniter_) knm_mod->multiply(false, *(splitdcmaster[1]), *dcnm);  //???

  /*--------------------------------------------------------------------*/
  // block mn
  /*--------------------------------------------------------------------*/
  // compute modification for block kmn
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmn_mod =
      matrix_multiply(*P, true, shapederivatives.matrix(2, 0), false, false, false, true);

  // Add transformation matrix to mn
  shapederivatives.matrix(1, 0).un_complete();
  Core::LinAlg::matrix_add(*kmn_mod, false, 1.0, shapederivatives.matrix(1, 0), 1.0);

  /*--------------------------------------------------------------------*/
  // block mm
  /*--------------------------------------------------------------------*/
  // compute modification for block kmm
  std::shared_ptr<Core::LinAlg::SparseMatrix> kss_mod =
      matrix_multiply(*P, true, shapederivatives.matrix(2, 2), false, false, false, true);
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmm_mod =
      matrix_multiply(*kss_mod, false, *P, false, false, false, true);

  // Add transformation matrix to mm
  shapederivatives.matrix(1, 1).un_complete();
  Core::LinAlg::matrix_add(*kmm_mod, false, 1.0, shapederivatives.matrix(1, 1), 1.0);

  if (dconmaster_ and firstnonliniter_) kmm_mod->multiply(false, *(splitdcmaster[1]), *dcmm);

  // complete matrix
  shapederivatives.complete();
}

FOUR_C_NAMESPACE_CLOSE
