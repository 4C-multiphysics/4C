// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_monolithicstructuresplit.hpp"

#include "4C_adapter_ale_fsi.hpp"
#include "4C_adapter_fld_fluid_fsi.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_comm_utils.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fsi_nox_group.hpp"
#include "4C_fsi_statustest.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::MonolithicStructureSplit::MonolithicStructureSplit(
    MPI_Comm comm, const Teuchos::ParameterList& timeparams)
    : BlockMonolithic(comm, timeparams), lambda_(nullptr), lambdaold_(nullptr), energysum_(0.0)
{
  sggtransform_ = std::make_shared<Coupling::Adapter::MatrixRowColTransform>();
  sgitransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  sigtransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  aigtransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();

  fmiitransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  fmgitransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();

  fsaigtransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  fsmgitransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();

  fscoupfa_ = std::make_shared<Coupling::Adapter::Coupling>();

  // Recovery of Lagrange multiplier happens on structure field
  lambda_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->interface()->fsi_cond_map(), true);
  lambdaold_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->interface()->fsi_cond_map(), true);
  ddiinc_ = nullptr;
  soliprev_ = nullptr;
  ddginc_ = nullptr;
  duginc_ = nullptr;
  disgprev_ = nullptr;
  sgiprev_ = nullptr;
  sggprev_ = nullptr;

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check whether allocation was successful
  if (sggtransform_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'sggtransform_' failed.");
  }
  if (sgitransform_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'sgitransform_' failed.");
  }
  if (sigtransform_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'sigtransform_' failed.");
  }
  if (aigtransform_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'aigtransform_' failed.");
  }
  if (fmiitransform_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'fmiitransform_' failed.");
  }
  if (fmgitransform_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'fmgitransform_' failed.");
  }
  if (fsaigtransform_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'fsaigtransform_' failed.");
  }
  if (fsmgitransform_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'fsmgitransform_' failed.");
  }
  if (fscoupfa_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'fscoupfa_' failed.");
  }
  if (lambda_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'lambda_' failed.");
  }
  if (lambdaold_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'lambdaold_' failed.");
  }
#endif

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::check_dirichlet_boundary_conditions_on_interface()
{
  // ---------------------------------------------------------------------------
  // FSI specific check of Dirichlet boundary conditions
  // ---------------------------------------------------------------------------
  // Create intersection of slave DOFs that hold a Dirichlet boundary condition
  // and are located at the FSI interface
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> intersectionmaps;
  intersectionmaps.push_back(structure_field()->get_dbc_map_extractor()->cond_map());
  intersectionmaps.push_back(structure_field()->interface()->fsi_cond_map());
  std::shared_ptr<Core::LinAlg::Map> intersectionmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(intersectionmaps);

  // Check whether the intersection is empty
  if (intersectionmap->num_global_elements() != 0)
  {
    //    std::cout << "Slave interface nodes with Dirichlet boundary condition (input file
    //    numbering):" << std::endl; for (int i=0; i <
    //    (int)fluid_field()->discretization()->NumMyRowNodes(); i++)
    //    {
    //      // get all nodes and add them
    //      int gid = structure_field()->discretization()->NodeRowMap()->GID(i);
    //
    //      // do only nodes that I have in my discretization
    //      if (!structure_field()->discretization()->NodeColMap()->MyGID(gid)) continue;
    //      Core::Nodes::Node* node = structure_field()->discretization()->gNode(gid);
    //      if (!node) FOUR_C_THROW("Cannot find node with gid %",gid);
    //
    //      std::vector<int> nodedofs = structure_field()->discretization()->Dof(node);
    //
    //      for (int j=0; j < (int)nodedofs.size(); j++)
    //      {
    //        for (int k=0; k < (int)intersectionmap->NumGlobalElements(); k++)
    //        {
    //          if (nodedofs[j] == intersectionmap->GID(k))
    //          {
    //            std::cout << gid+1 << std::endl;
    //            k = (int)intersectionmap->GID(k);
    //            j = (int)nodedofs.size();
    //          }
    //        }
    //      }
    //    }

    // It is not allowed, that slave DOFs at the interface hold a Dirichlet
    // boundary condition. Thus --> Error message

    // We do not have to care whether ALE interface DOFs carry DBCs in the
    // input file since they do not occur in the monolithic system and, hence,
    // do not cause a conflict.

    std::stringstream errormsg;
    errormsg << "  "
                "+---------------------------------------------------------------------------------"
                "------------+"
             << std::endl
             << "  |                DIRICHLET BOUNDARY CONDITIONS ON SLAVE SIDE OF FSI INTERFACE   "
                "              |"
             << std::endl
             << "  "
                "+---------------------------------------------------------------------------------"
                "------------+"
             << std::endl
             << "  | NOTE: The slave side of the interface is not allowed to carry Dirichlet "
                "boundary conditions.|"
             << std::endl
             << "  |                                                                               "
                "              |"
             << std::endl
             << "  | This is a structure split scheme. Hence, master and slave field are chosen as "
                "follows:      |"
             << std::endl
             << "  |     MASTER  = FLUID                                                           "
                "              |"
             << std::endl
             << "  |     SLAVE   = STRUCTURE                                                       "
                "              |"
             << std::endl
             << "  |                                                                               "
                "              |"
             << std::endl
             << "  | Dirichlet boundary conditions were detected on slave interface degrees of "
                "freedom. Please   |"
             << std::endl
             << "  | remove Dirichlet boundary conditions from the slave side of the FSI "
                "interface.              |"
             << std::endl
             << "  | Only the master side of the FSI interface is allowed to carry Dirichlet "
                "boundary conditions.|"
             << std::endl
             << "  "
                "+---------------------------------------------------------------------------------"
                "------------+"
             << std::endl;

    FOUR_C_THROW("{}", errormsg.str());
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::setup_system()
{
  // check for proper dirichlet boundary condition application
  check_dirichlet_boundary_conditions_on_interface();

  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  set_default_parameters(fsidyn, nox_parameter_list());

  // call SetupSystem in base class
  FSI::Monolithic::setup_system();

  // create combined map
  create_combined_dof_row_map();

  // Use normal matrix for fluid equations but build (split) mesh movement
  // linearization (if requested in the input file)
  fluid_field()->use_block_matrix(false);

  // Do no longer use split structure matrix for new solid time integration
  if (use_old_structure_) structure_field()->use_block_matrix();

  // build ale system matrix in split system
  ale_field()->create_system_matrix(ale_field()->interface());

  aleresidual_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*ale_field()->interface()->other_map());

  // ---------------------------------------------------------------------------
  // Build the global Dirichlet map extractor
  setup_dbc_map_extractor();
  // ---------------------------------------------------------------------------

  create_system_matrix();

  // NOTE: if we restart from an part. fsi problem we still have to read lambda_. But since this
  // requires coupsf_ in order to map the nodal fluid forces on the structure nodes we have to do it
  // e.g. in here. But:
  // TODO: Move this to read_restart() when possible
  const int restart = Global::Problem::instance()->restart();
  if (restart)
  {
    const bool restartfrompartfsi = timeparams_.get<bool>("RESTART_FROM_PART_FSI");
    if (restartfrompartfsi)  // restart from part. fsi
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> lambdafullfluid =
          std::make_shared<Core::LinAlg::Vector<double>>(*fluid_field()->dof_row_map(), true);
      Core::IO::DiscretizationReader reader =
          Core::IO::DiscretizationReader(*fluid_field()->discretization(),
              Global::Problem::instance()->input_control_file(), restart);
      reader.read_vector(lambdafullfluid, "fsilambda");

      std::shared_ptr<Core::LinAlg::Vector<double>> lambdafluid =
          std::make_shared<Core::LinAlg::Vector<double>>(
              *fluid_field()->interface()->full_map(), true);
      lambdafluid = fluid_field()->interface()->extract_fsi_cond_vector(*lambdafullfluid);

      lambdaold_ = fluid_to_struct(lambdafluid);
      // Note: the above is normally enough. However, we can use the restart in order to
      // periodically repeat the fsi simulation (see AC-FS3I)
      lambda_ = fluid_to_struct(lambdafluid);
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::create_system_matrix()
{
  FSI::BlockMonolithic::create_system_matrix(systemmatrix_, true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::create_combined_dof_row_map()
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;
  vecSpaces.push_back(structure_field()->interface()->other_map());
  vecSpaces.push_back(fluid_field()->dof_row_map());
  vecSpaces.push_back(ale_field()->interface()->other_map());

  if (vecSpaces[0]->num_global_elements() == 0)
    FOUR_C_THROW("No inner structural equations. Splitting not possible. Panic.");

  set_dof_row_maps(vecSpaces);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::setup_dbc_map_extractor()
{
  // Dirichlet maps for structure and fluid do not intersect with interface map.
  // ALE Dirichlet map might intersect with interface map, but ALE interface DOFs
  // are not part of the final system of equations. Hence, we just need the
  // intersection of inner ALE DOFs with Dirichlet ALE DOFs.
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> aleintersectionmaps;
  aleintersectionmaps.push_back(ale_field()->get_dbc_map_extractor()->cond_map());
  aleintersectionmaps.push_back(ale_field()->interface()->other_map());
  std::shared_ptr<Core::LinAlg::Map> aleintersectionmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(aleintersectionmaps);

  // Merge Dirichlet maps of structure, fluid and ALE to global FSI Dirichlet map
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> dbcmaps;
  dbcmaps.push_back(structure_field()->get_dbc_map_extractor()->cond_map());
  dbcmaps.push_back(fluid_field()->get_dbc_map_extractor()->cond_map());
  dbcmaps.push_back(aleintersectionmap);
  std::shared_ptr<const Core::LinAlg::Map> dbcmap =
      Core::LinAlg::MultiMapExtractor::merge_maps(dbcmaps);

  // Finally, create the global FSI Dirichlet map extractor
  dbcmaps_ = std::make_shared<Core::LinAlg::MapExtractor>(*dof_row_map(), dbcmap, true);
  if (dbcmaps_ == nullptr)
  {
    FOUR_C_THROW("Creation of FSI Dirichlet map extractor failed.");
  }

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> FSI::MonolithicStructureSplit::system_matrix()
    const
{
  return systemmatrix_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::setup_rhs_residual(Core::LinAlg::Vector<double>& f)
{
  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  // some scaling factors for fluid
  const double fluidscale = fluid_field()->residual_scaling();

  // get single field residuals
  Core::LinAlg::Vector<double> sv(*structure_field()->rhs());
  const Core::LinAlg::Vector<double> fv(*fluid_field()->rhs());
  const Core::LinAlg::Vector<double> av(*ale_field()->rhs());

  // NOX treats rhs different in new solid time integration, hence sign inversion is necessary here
  if (!use_old_structure_) sv.scale(-1.0);

  // extract only inner DOFs from structure (=slave) and ALE field
  std::shared_ptr<const Core::LinAlg::Vector<double>> sov =
      structure_field()->interface()->extract_other_vector(sv);
  std::shared_ptr<const Core::LinAlg::Vector<double>> aov =
      ale_field()->interface()->extract_other_vector(av);

  // add structure interface residual to fluid interface residual considering temporal scaling
  std::shared_ptr<Core::LinAlg::Vector<double>> scv =
      structure_field()->interface()->extract_fsi_cond_vector(sv);
  std::shared_ptr<Core::LinAlg::Vector<double>> modfv =
      fluid_field()->interface()->insert_fsi_cond_vector(*struct_to_fluid(scv));
  modfv->update(1.0, fv, (1.0 - ftiparam) / ((1.0 - stiparam) * fluidscale));

  // put the single field residuals together
  FSI::Monolithic::combine_field_vectors(f, *sov, *modfv, *aov);

  // add additional ale residual
  extractor().add_vector(*aleresidual_, 2, f);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::setup_rhs_lambda(Core::LinAlg::Vector<double>& f)
{
  if (lambdaold_ != nullptr)
  {
    // get time integration parameters of structure and fluid time integrators
    // to enable consistent time integration among the fields
    const double stiparam = structure_field()->tim_int_param();
    const double ftiparam = fluid_field()->tim_int_param();

    // some scaling factors for fluid
    const double fluidscale = fluid_field()->residual_scaling();

    // project Lagrange multiplier field onto the master interface DOFs and consider temporal
    // scaling
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
        fluid_field()->interface()->insert_fsi_cond_vector(*struct_to_fluid(lambdaold_));
    lambdafull->scale((-ftiparam + (stiparam * (1.0 - ftiparam)) / (1.0 - stiparam)) / fluidscale);

    // add Lagrange multiplier
    extractor().add_vector(*lambdafull, 1, f);
  }

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::setup_rhs_firstiter(Core::LinAlg::Vector<double>& f)
{
  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  // some scaling factors for fluid
  const double scale = fluid_field()->residual_scaling();

  // old interface velocity of fluid field
  const std::shared_ptr<const Core::LinAlg::Vector<double>> fveln =
      fluid_field()->extract_interface_veln();

  // get fluid shape derivatives matrix
  const std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase> mmm =
      fluid_field()->shape_derivatives();

  // get structure matrix
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blocks = nullptr;
  if (use_old_structure_)
  {
    blocks = structure_field()->block_system_matrix();
  }
  else
  {
    const std::shared_ptr<const Core::LinAlg::SparseMatrix> sparse_matrix_solid =
        structure_field()->system_matrix();

    // convert sparse matrix to block sparse matrix
    // const std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase> blocks =
    blocks = Core::LinAlg::copy_sparse_to_block_sparse_matrix(
        *sparse_matrix_solid, *structure_field()->interface(), *structure_field()->interface());

    // Take care of Dirichlet boundary conditions
    blocks->apply_dirichlet(*structure_field()->get_dbc_map_extractor()->cond_map(), true);
    // blocks->apply_dirichlet(*(dbcmaps_->cond_map()), true);
  }

  // get ale matrix
  const std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase> blocka =
      ale_field()->block_system_matrix();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (blocks == nullptr) FOUR_C_THROW("Expected structure block matrix.");
  if (blocka == nullptr) FOUR_C_THROW("Expected ALE block matrix.");
#endif

  // extract submatrices
  const Core::LinAlg::SparseMatrix& sig = blocks->matrix(0, 1);  // S_{I\Gamma}
  const Core::LinAlg::SparseMatrix& sgg = blocks->matrix(1, 1);  // S_{\Gamma\Gamma}
  const Core::LinAlg::SparseMatrix& aig = blocka->matrix(0, 1);  // A_{I\Gamma}

  // some often re-used vectors
  std::shared_ptr<Core::LinAlg::Vector<double>> rhs =
      nullptr;  // right hand side of single set of DOFs

  // Different contributions/terms to the rhs are separated by the following comment line
  // ---------- inner structure DOFs
  /* The following terms are added to the inner structure DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  + S_{I \Gamma} * \Delta d_{\Gamma,p}
   *
   * (2)  - dt * S_{I \Gamma} * u^{n}_{\Gamma}
   *
   * Remarks on all terms:
   * +  tau: time scaling factor for interface time integration (tau =
   * 1/fluid_field()->TimeScaling())
   *
   */
  // ----------addressing term 1
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(sig.row_map(), true);
  sig.multiply(false, *ddgpred_, *rhs);

  extractor().add_vector(*rhs, 0, f);
  // ----------end of term 1

  // ----------addressing term 2
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(sig.row_map(), true);
  sig.multiply(false, *fluid_to_struct(fveln), *rhs);
  rhs->scale(-dt());

  extractor().add_vector(*rhs, 0, f);
  // ----------end of term 2
  // ----------end of inner structural DOFs

  // ---------- inner fluid DOFs
  /* The following terms are added to the inner fluid DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  - dt * F^{G}_{I\Gamma} * u^{n}_{\Gamma}
   *
   */
  // ----------addressing term 1
  if (mmm != nullptr)
  {
    const Core::LinAlg::SparseMatrix& fmig = mmm->matrix(0, 1);
    rhs = std::make_shared<Core::LinAlg::Vector<double>>(fmig.row_map(), true);
    fmig.multiply(false, *fveln, *rhs);
    rhs->scale(-dt());

    rhs = fluid_field()->interface()->insert_other_vector(*rhs);

    extractor().add_vector(*rhs, 1, f);
  }
  // ----------end of term 1
  // ---------- end of inner fluid DOFs

  // ---------- interface fluid DOFs
  /* The following terms are added to the interface fluid DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  - dt * F^{G}_{\Gamma\Gamma} * u^{n}_{\Gamma}
   *
   * (2)  - dt * (1-ftiparam)/(1-stiparam) * S_{\Gamma\Gamma} * u^{n}_{\Gamma}
   *
   * (3)  + (1-ftiparam)/(1-stiparam) * S_{\Gamma\Gamma} * \Delta d_{\Gamma,p}
   *
   * Remarks on all terms:
   * +  tau: time scaling factor for interface time integration (tau =
   * 1/fluid_field()->TimeScaling())
   *
   */
  // ----------addressing term 1
  if (mmm != nullptr)
  {
    const Core::LinAlg::SparseMatrix& fmgg = mmm->matrix(1, 1);
    rhs = std::make_shared<Core::LinAlg::Vector<double>>(fmgg.row_map(), true);
    fmgg.multiply(false, *fveln, *rhs);
    rhs->scale(-dt());

    rhs = fluid_field()->interface()->insert_fsi_cond_vector(*rhs);

    extractor().add_vector(*rhs, 1, f);
  }
  // ----------end of term 1

  // ----------addressing term 2
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(sgg.row_map(), true);
  sgg.multiply(false, *fluid_to_struct(fveln), *rhs);
  rhs->scale(-dt() * (1. - ftiparam) / ((1. - stiparam) * scale));

  rhs = struct_to_fluid(rhs);
  rhs = fluid_field()->interface()->insert_fsi_cond_vector(*rhs);

  extractor().add_vector(*rhs, 1, f);
  // ----------end of term 2

  // ----------addressing term 3
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(sgg.row_map(), true);
  sgg.multiply(false, *ddgpred_, *rhs);
  rhs->scale((1. - ftiparam) / ((1. - stiparam) * scale));

  rhs = struct_to_fluid(rhs);
  rhs = fluid_field()->interface()->insert_fsi_cond_vector(*rhs);

  extractor().add_vector(*rhs, 1, f);
  // ----------end of term 3
  // ---------- end of interface fluid DOFs

  // ---------- inner ale DOFs
  /* The following terms are added to the inner ale DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  - dt * A_{I\Gamma} * u^{n}_{\Gamma}
   *
   */
  // ----------addressing term 1
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(aig.row_map(), true);
  aig.multiply(false, *fluid_to_ale_interface(fveln), *rhs);
  rhs->scale(-dt());

  extractor().add_vector(*rhs, 2, f);
  // ----------end of term 1
  // ---------- end of inner ale DOFs

  // -----------------------------------------------------
  // Now, all contributions/terms to rhs in the first Newton iteration are added.

  // Reset quantities for previous iteration step since they still store values from the last time
  // step
  ddiinc_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->interface()->other_map(), true);
  soliprev_ = nullptr;
  ddginc_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->interface()->fsi_cond_map(), true);
  duginc_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *fluid_field()->interface()->fsi_cond_map(), true);
  disgprev_ = nullptr;
  sgicur_ = nullptr;
  sggcur_ = nullptr;

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::setup_system_matrix(Core::LinAlg::BlockSparseMatrixBase& mat)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicStructureSplit::setup_system_matrix");

  const Coupling::Adapter::Coupling& coupsf = structure_fluid_coupling();
  const Coupling::Adapter::Coupling& coupfa = fluid_ale_coupling();
  const Coupling::Adapter::Coupling& icoupfa = interface_fluid_ale_coupling();

  // get single field block matrices
  std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase> s = nullptr;
  if (use_old_structure_)
  {
    s = structure_field()->block_system_matrix();
  }
  else
  {
    // get structure matrix
    const std::shared_ptr<const Core::LinAlg::SparseMatrix> sparse_matrix_solid =
        structure_field()->system_matrix();
    // convert sparse matrix to block sparse matrix
    // const std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase> blocks =
    s = Core::LinAlg::copy_sparse_to_block_sparse_matrix(
        *sparse_matrix_solid, *structure_field()->interface(), *structure_field()->interface());
  }

  const std::shared_ptr<Core::LinAlg::SparseMatrix> f = fluid_field()->system_matrix();
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> a = ale_field()->block_system_matrix();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check whether allocation was successful
  if (s == nullptr)
  {
    FOUR_C_THROW("expect structure block matrix");
  }
  if (f == nullptr)
  {
    FOUR_C_THROW("expect fluid block matrix");
  }
  if (a == nullptr)
  {
    FOUR_C_THROW("expect ale block matrix");
  }
#endif

  // extract submatrices
  Core::LinAlg::SparseMatrix& aii = a->matrix(0, 0);
  Core::LinAlg::SparseMatrix& aig = a->matrix(0, 1);

  // scaling factors for fluid
  const double scale = fluid_field()->residual_scaling();
  const double timescale = fluid_field()->time_scaling();

  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  /*----------------------------------------------------------------------*/

  // build block matrix
  // The maps of the block matrix have to match the maps of the blocks we
  // insert here.

  // Incomplete fluid matrix to be able to deal with slightly defective
  // interface meshes.
  f->un_complete();

  mat.assign(0, 0, Core::LinAlg::DataAccess::Share, s->matrix(0, 0));

  (*sigtransform_)(s->full_row_map(), s->full_col_map(), s->matrix(0, 1), 1. / timescale,
      Coupling::Adapter::CouplingMasterConverter(coupsf), mat.matrix(0, 1));
  (*sggtransform_)(s->matrix(1, 1), (1.0 - ftiparam) / ((1.0 - stiparam) * scale * timescale),
      Coupling::Adapter::CouplingMasterConverter(coupsf),
      Coupling::Adapter::CouplingMasterConverter(coupsf), *f, true, true);

  std::shared_ptr<Core::LinAlg::SparseMatrix> lsgi =
      std::make_shared<Core::LinAlg::SparseMatrix>(f->row_map(), 81, false);
  (*sgitransform_)(s->matrix(1, 0), (1.0 - ftiparam) / ((1.0 - stiparam) * scale),
      Coupling::Adapter::CouplingMasterConverter(coupsf), *lsgi);

  lsgi->complete(s->matrix(1, 0).domain_map(), f->range_map());

  mat.assign(1, 0, Core::LinAlg::DataAccess::Share, *lsgi);

  (*aigtransform_)(a->full_row_map(), a->full_col_map(), aig, 1. / timescale,
      Coupling::Adapter::CouplingSlaveConverter(icoupfa), mat.matrix(2, 1));
  mat.assign(2, 2, Core::LinAlg::DataAccess::Share, aii);

  /*----------------------------------------------------------------------*/
  // add optional fluid linearization with respect to mesh motion block

  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mmm = fluid_field()->shape_derivatives();
  if (mmm != nullptr)
  {
    const Core::LinAlg::SparseMatrix& fmii = mmm->matrix(0, 0);
    const Core::LinAlg::SparseMatrix& fmig = mmm->matrix(0, 1);
    const Core::LinAlg::SparseMatrix& fmgi = mmm->matrix(1, 0);
    const Core::LinAlg::SparseMatrix& fmgg = mmm->matrix(1, 1);

    Core::LinAlg::matrix_add(fmgg, false, 1. / timescale, *f, 1.0);
    Core::LinAlg::matrix_add(fmig, false, 1. / timescale, *f, 1.0);

    Core::LinAlg::SparseMatrix lfmgi(f->row_map(), 81, false);
    (*fmgitransform_)(mmm->full_row_map(), mmm->full_col_map(), fmgi, 1.,
        Coupling::Adapter::CouplingMasterConverter(coupfa), lfmgi, false, false);

    (*fmiitransform_)(mmm->full_row_map(), mmm->full_col_map(), fmii, 1.,
        Coupling::Adapter::CouplingMasterConverter(coupfa), lfmgi, false, true);

    lfmgi.complete(aii.domain_map(), f->range_map());

    mat.assign(1, 2, Core::LinAlg::DataAccess::Share, lfmgi);
  }

  f->complete();

  // finally assign fluid block
  mat.assign(1, 1, Core::LinAlg::DataAccess::Share, *f);

  // done. make sure all blocks are filled.
  mat.complete();

  // Finally, take care of Dirichlet boundary conditions
  mat.apply_dirichlet(*(dbcmaps_->cond_map()), true);

  // store parts of structural matrix to know them in the next iteration as previous iteration
  // matrices
  sgiprev_ = sgicur_;
  sggprev_ = sggcur_;
  sgicur_ = std::make_shared<Core::LinAlg::SparseMatrix>(s->matrix(1, 0));
  sggcur_ = std::make_shared<Core::LinAlg::SparseMatrix>(s->matrix(1, 1));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::scale_system(
    Core::LinAlg::BlockSparseMatrixBase& mat, Core::LinAlg::Vector<double>& b)
{
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");
  const bool scaling_infnorm = fsimono.get<bool>("INFNORMSCALING");

  if (scaling_infnorm)
  {
    // TODO: The matrices are modified here. Do we have to change them back later on?

    // do scaling of structure rows
    Core::LinAlg::SparseMatrix& A_00 = mat.matrix(0, 0);
    srowsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_00.row_map(), false);
    scolsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_00.row_map(), false);
    A_00.inv_row_sums(*srowsum_);
    A_00.inv_col_sums(*scolsum_);
    A_00.left_scale(*srowsum_);
    A_00.right_scale(*scolsum_);
    mat.matrix(0, 1).left_scale(*srowsum_);
    mat.matrix(0, 2).left_scale(*srowsum_);
    mat.matrix(1, 0).right_scale(*scolsum_);
    mat.matrix(2, 0).right_scale(*scolsum_);

    // do scaling of ale rows
    Core::LinAlg::SparseMatrix& A_22 = mat.matrix(2, 2);
    arowsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_22.row_map(), false);
    acolsum_ = std::make_shared<Core::LinAlg::Vector<double>>(A_22.row_map(), false);
    A_22.inv_row_sums(*arowsum_);
    A_22.inv_col_sums(*acolsum_);
    A_22.left_scale(*arowsum_);
    A_22.right_scale(*acolsum_);
    mat.matrix(2, 0).left_scale(*arowsum_);
    mat.matrix(2, 1).left_scale(*arowsum_);
    mat.matrix(0, 2).right_scale(*acolsum_);
    mat.matrix(1, 2).right_scale(*acolsum_);

    // do scaling of structure and ale rhs vectors
    std::shared_ptr<Core::LinAlg::Vector<double>> sx = extractor().extract_vector(b, 0);
    std::shared_ptr<Core::LinAlg::Vector<double>> ax = extractor().extract_vector(b, 2);

    sx->multiply(1.0, *srowsum_, *sx, 0.0);
    ax->multiply(1.0, *arowsum_, *ax, 0.0);

    extractor().insert_vector(*sx, 0, b);
    extractor().insert_vector(*ax, 2, b);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::unscale_solution(Core::LinAlg::BlockSparseMatrixBase& mat,
    Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& b)
{
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");
  const bool scaling_infnorm = fsimono.get<bool>("INFNORMSCALING");

  if (scaling_infnorm)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> sy = extractor().extract_vector(x, 0);
    std::shared_ptr<Core::LinAlg::Vector<double>> ay = extractor().extract_vector(x, 2);

    sy->multiply(1.0, *scolsum_, *sy, 0.0);
    ay->multiply(1.0, *acolsum_, *ay, 0.0);

    extractor().insert_vector(*sy, 0, x);
    extractor().insert_vector(*ay, 2, x);

    std::shared_ptr<Core::LinAlg::Vector<double>> sx = extractor().extract_vector(b, 0);
    std::shared_ptr<Core::LinAlg::Vector<double>> ax = extractor().extract_vector(b, 2);

    sx->reciprocal_multiply(1.0, *srowsum_, *sx, 0.0);
    ax->reciprocal_multiply(1.0, *arowsum_, *ax, 0.0);

    extractor().insert_vector(*sx, 0, b);
    extractor().insert_vector(*ax, 2, b);

    Core::LinAlg::SparseMatrix& A_00 = mat.matrix(0, 0);
    srowsum_->reciprocal(*srowsum_);
    scolsum_->reciprocal(*scolsum_);
    A_00.left_scale(*srowsum_);
    A_00.right_scale(*scolsum_);
    mat.matrix(0, 1).left_scale(*srowsum_);
    mat.matrix(0, 2).left_scale(*srowsum_);
    mat.matrix(1, 0).right_scale(*scolsum_);
    mat.matrix(2, 0).right_scale(*scolsum_);

    Core::LinAlg::SparseMatrix& A_22 = mat.matrix(2, 2);
    arowsum_->reciprocal(*arowsum_);
    acolsum_->reciprocal(*acolsum_);
    A_22.left_scale(*arowsum_);
    A_22.right_scale(*acolsum_);
    mat.matrix(2, 0).left_scale(*arowsum_);
    mat.matrix(2, 1).left_scale(*arowsum_);
    mat.matrix(0, 2).right_scale(*acolsum_);
    mat.matrix(1, 2).right_scale(*acolsum_);
  }

  // very simple hack just to see the linear solution

  Core::LinAlg::Vector<double> r(b.get_map());
  mat.multiply(false, x, r);
  r.update(1., b, 1.);

  std::shared_ptr<Core::LinAlg::Vector<double>> sr = extractor().extract_vector(r, 0);
  std::shared_ptr<Core::LinAlg::Vector<double>> fr = extractor().extract_vector(r, 1);
  std::shared_ptr<Core::LinAlg::Vector<double>> ar = extractor().extract_vector(r, 2);

  // increment additional ale residual
  aleresidual_->update(-1., *ar, 0.);

  std::ios_base::fmtflags flags = utils()->out().flags();

  double n, ns, nf, na;
  r.norm_2(&n);
  sr->norm_2(&ns);
  fr->norm_2(&nf);
  ar->norm_2(&na);
  if (verbosity_ == Inpar::FSI::verbosity_full)
  {
    utils()->out() << std::scientific << "\nlinear solver quality:\n"
                   << "L_2-norms:\n"
                   << "   |r|=" << n << "   |rs|=" << ns << "   |rf|=" << nf << "   |ra|=" << na
                   << "\n";
  }
  r.norm_inf(&n);
  sr->norm_inf(&ns);
  fr->norm_inf(&nf);
  ar->norm_inf(&na);
  if (verbosity_ == Inpar::FSI::verbosity_full)
  {
    utils()->out() << "L_inf-norms:\n"
                   << "   |r|=" << n << "   |rs|=" << ns << "   |rf|=" << nf << "   |ra|=" << na
                   << "\n";
  }

  utils()->out().flags(flags);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<::NOX::StatusTest::Combo> FSI::MonolithicStructureSplit::create_status_test(
    Teuchos::ParameterList& nlParams, Teuchos::RCP<::NOX::Abstract::Group> grp)
{
  // --------------------------------------------------------------------
  // Setup the test framework
  // --------------------------------------------------------------------
  // Create the top-level test combo
  Teuchos::RCP<::NOX::StatusTest::Combo> combo =
      Teuchos::make_rcp<::NOX::StatusTest::Combo>(::NOX::StatusTest::Combo::OR);

  // Create test combo for convergence of residuals and iterative increments
  Teuchos::RCP<::NOX::StatusTest::Combo> converged =
      Teuchos::make_rcp<::NOX::StatusTest::Combo>(::NOX::StatusTest::Combo::AND);

  // Create some other plausibility tests
  Teuchos::RCP<::NOX::StatusTest::MaxIters> maxiters =
      Teuchos::make_rcp<::NOX::StatusTest::MaxIters>(nlParams.get<int>("Max Iterations"));
  Teuchos::RCP<::NOX::StatusTest::FiniteValue> fv =
      Teuchos::make_rcp<::NOX::StatusTest::FiniteValue>();

  // Add single tests to the top-level test combo
  combo->addStatusTest(fv);
  combo->addStatusTest(converged);
  combo->addStatusTest(maxiters);

  // Start filling the 'converged' combo here
  // require one solve
  converged->addStatusTest(Teuchos::make_rcp<NOX::FSI::MinIters>(1));


  // --------------------------------------------------------------------
  // setup tests for structural displacement field
  // --------------------------------------------------------------------
  // create ::NOX::StatusTest::Combo for structural displacement field
  Teuchos::RCP<::NOX::StatusTest::Combo> structcombo =
      Teuchos::make_rcp<::NOX::StatusTest::Combo>(::NOX::StatusTest::Combo::AND);

  // create Norm-objects for each norm that has to be tested
  Teuchos::RCP<NOX::FSI::PartialNormF> structureDisp_L2 = Teuchos::make_rcp<NOX::FSI::PartialNormF>(
      "DISPL residual", extractor(), 0, nlParams.get<double>("Tol dis res L2"),
      ::NOX::Abstract::Vector::TwoNorm, NOX::FSI::PartialNormF::Scaled);
  Teuchos::RCP<NOX::FSI::PartialNormF> structureDisp_inf =
      Teuchos::make_rcp<NOX::FSI::PartialNormF>("DISPL residual", extractor(), 0,
          nlParams.get<double>("Tol dis res Inf"), ::NOX::Abstract::Vector::MaxNorm,
          NOX::FSI::PartialNormF::Unscaled);
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> structureDispUpdate_L2 =
      Teuchos::make_rcp<NOX::FSI::PartialNormUpdate>("DISPL update", extractor(), 0,
          nlParams.get<double>("Tol dis inc L2"), ::NOX::Abstract::Vector::TwoNorm,
          NOX::FSI::PartialNormUpdate::Scaled);
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> structureDispUpdate_inf =
      Teuchos::make_rcp<NOX::FSI::PartialNormUpdate>("DISPL update", extractor(), 0,
          nlParams.get<double>("Tol dis inc Inf"), ::NOX::Abstract::Vector::MaxNorm,
          NOX::FSI::PartialNormUpdate::Unscaled);

  // tests needed to adapt relative tolerance of the linear solver
  add_status_test(structureDisp_L2);

  // add norm-tests to structural displacement ::NOX::StatusTest::Combo
  structcombo->addStatusTest(structureDisp_L2);
  structcombo->addStatusTest(structureDisp_inf);
  structcombo->addStatusTest(structureDispUpdate_L2);
  structcombo->addStatusTest(structureDispUpdate_inf);

  // add structural displacement test combo to top-level test combo
  converged->addStatusTest(structcombo);
  // ---------- end of structural displacement field tests

  // --------------------------------------------------------------------
  // setup tests for interface
  // --------------------------------------------------------------------
  // build mapextractor
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> interface;
  interface.push_back(fluid_field()->interface()->fsi_cond_map());
  interface.push_back(nullptr);
  Core::LinAlg::MultiMapExtractor interfaceextract(*dof_row_map(), interface);

  // create ::NOX::StatusTest::Combo for interface
  Teuchos::RCP<::NOX::StatusTest::Combo> interfacecombo =
      Teuchos::make_rcp<::NOX::StatusTest::Combo>(::NOX::StatusTest::Combo::AND);

  // create Norm-objects for each norm that has to be tested
  Teuchos::RCP<NOX::FSI::PartialNormF> interfaceTest_L2 = Teuchos::make_rcp<NOX::FSI::PartialNormF>(
      "GAMMA residual", interfaceextract, 0, nlParams.get<double>("Tol fsi res L2"),
      ::NOX::Abstract::Vector::TwoNorm, NOX::FSI::PartialNormF::Scaled);
  Teuchos::RCP<NOX::FSI::PartialNormF> interfaceTest_inf =
      Teuchos::make_rcp<NOX::FSI::PartialNormF>("GAMMA residual", interfaceextract, 0,
          nlParams.get<double>("Tol fsi res Inf"), ::NOX::Abstract::Vector::MaxNorm,
          NOX::FSI::PartialNormF::Unscaled);
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> interfaceTestUpdate_L2 =
      Teuchos::make_rcp<NOX::FSI::PartialNormUpdate>("GAMMA update", interfaceextract, 0,
          nlParams.get<double>("Tol fsi inc L2"), ::NOX::Abstract::Vector::TwoNorm,
          NOX::FSI::PartialNormUpdate::Scaled);
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> interfaceTestUpdate_inf =
      Teuchos::make_rcp<NOX::FSI::PartialNormUpdate>("GAMMA update", interfaceextract, 0,
          nlParams.get<double>("Tol fsi inc Inf"), ::NOX::Abstract::Vector::MaxNorm,
          NOX::FSI::PartialNormUpdate::Unscaled);

  // tests needed to adapt relative tolerance of the linear solver
  add_status_test(interfaceTest_L2);

  // add norm-tests to interface ::NOX::StatusTest::Combo
  interfacecombo->addStatusTest(interfaceTest_L2);
  interfacecombo->addStatusTest(interfaceTest_inf);
  interfacecombo->addStatusTest(interfaceTestUpdate_L2);
  interfacecombo->addStatusTest(interfaceTestUpdate_inf);

  // add interface test combo to top-level test combo
  converged->addStatusTest(interfacecombo);
  // ---------- end of interface tests

  // --------------------------------------------------------------------
  // setup tests for fluid velocity field
  // --------------------------------------------------------------------
  // build mapextractor
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> fluidvel;
  fluidvel.push_back(fluid_field()->inner_velocity_row_map());
  fluidvel.push_back(nullptr);
  Core::LinAlg::MultiMapExtractor fluidvelextract(*dof_row_map(), fluidvel);

  // create ::NOX::StatusTest::Combo for fluid velocity field
  Teuchos::RCP<::NOX::StatusTest::Combo> fluidvelcombo =
      Teuchos::make_rcp<::NOX::StatusTest::Combo>(::NOX::StatusTest::Combo::AND);

  // create Norm-objects for each norm that has to be tested
  Teuchos::RCP<NOX::FSI::PartialNormF> innerFluidVel_L2 = Teuchos::make_rcp<NOX::FSI::PartialNormF>(
      "VELOC residual", fluidvelextract, 0, nlParams.get<double>("Tol vel res L2"),
      ::NOX::Abstract::Vector::TwoNorm, NOX::FSI::PartialNormF::Scaled);
  Teuchos::RCP<NOX::FSI::PartialNormF> innerFluidVel_inf =
      Teuchos::make_rcp<NOX::FSI::PartialNormF>("VELOC residual", fluidvelextract, 0,
          nlParams.get<double>("Tol vel res Inf"), ::NOX::Abstract::Vector::MaxNorm,
          NOX::FSI::PartialNormF::Unscaled);
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> innerFluidVelUpdate_L2 =
      Teuchos::make_rcp<NOX::FSI::PartialNormUpdate>("VELOC update", fluidvelextract, 0,
          nlParams.get<double>("Tol vel inc L2"), ::NOX::Abstract::Vector::TwoNorm,
          NOX::FSI::PartialNormUpdate::Scaled);
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> innerFluidVelUpdate_inf =
      Teuchos::make_rcp<NOX::FSI::PartialNormUpdate>("VELOC update", fluidvelextract, 0,
          nlParams.get<double>("Tol vel inc Inf"), ::NOX::Abstract::Vector::MaxNorm,
          NOX::FSI::PartialNormUpdate::Unscaled);

  // tests needed to adapt relative tolerance of the linear solver
  add_status_test(innerFluidVel_L2);

  // add norm-tests to fluid velocity ::NOX::StatusTest::Combo
  fluidvelcombo->addStatusTest(innerFluidVel_L2);
  fluidvelcombo->addStatusTest(innerFluidVel_inf);
  fluidvelcombo->addStatusTest(innerFluidVelUpdate_L2);
  fluidvelcombo->addStatusTest(innerFluidVelUpdate_inf);

  // add fluid velocity test combo to top-level test combo
  converged->addStatusTest(fluidvelcombo);
  // ---------- end of fluid velocity field tests

  // --------------------------------------------------------------------
  // setup tests for fluid pressure field
  // --------------------------------------------------------------------
  // build mapextractor
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> fluidpress;
  fluidpress.push_back(fluid_field()->pressure_row_map());
  fluidpress.push_back(nullptr);
  Core::LinAlg::MultiMapExtractor fluidpressextract(*dof_row_map(), fluidpress);

  // create ::NOX::StatusTest::Combo for fluid pressure field
  Teuchos::RCP<::NOX::StatusTest::Combo> fluidpresscombo =
      Teuchos::make_rcp<::NOX::StatusTest::Combo>(::NOX::StatusTest::Combo::AND);

  // create Norm-objects for each norm that has to be tested
  Teuchos::RCP<NOX::FSI::PartialNormF> fluidPress_L2 = Teuchos::make_rcp<NOX::FSI::PartialNormF>(
      "PRESS residual", fluidpressextract, 0, nlParams.get<double>("Tol pre res L2"),
      ::NOX::Abstract::Vector::TwoNorm, NOX::FSI::PartialNormF::Scaled);
  Teuchos::RCP<NOX::FSI::PartialNormF> fluidPress_inf = Teuchos::make_rcp<NOX::FSI::PartialNormF>(
      "PRESS residual", fluidpressextract, 0, nlParams.get<double>("Tol pre res Inf"),
      ::NOX::Abstract::Vector::MaxNorm, NOX::FSI::PartialNormF::Unscaled);
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> fluidPressUpdate_L2 =
      Teuchos::make_rcp<NOX::FSI::PartialNormUpdate>("PRESS update", fluidpressextract, 0,
          nlParams.get<double>("Tol pre inc L2"), ::NOX::Abstract::Vector::TwoNorm,
          NOX::FSI::PartialNormUpdate::Scaled);
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> fluidPressUpdate_inf =
      Teuchos::make_rcp<NOX::FSI::PartialNormUpdate>("PRESS update", fluidpressextract, 0,
          nlParams.get<double>("Tol pre inc Inf"), ::NOX::Abstract::Vector::MaxNorm,
          NOX::FSI::PartialNormUpdate::Unscaled);

  // tests needed to adapt relative tolerance of the linear solver
  add_status_test(fluidPress_L2);

  // add norm-tests to fluid pressure ::NOX::StatusTest::Combo
  fluidpresscombo->addStatusTest(fluidPress_L2);
  fluidpresscombo->addStatusTest(fluidPress_inf);
  fluidpresscombo->addStatusTest(fluidPressUpdate_L2);
  fluidpresscombo->addStatusTest(fluidPressUpdate_inf);

  // add fluid pressure test combo to top-level test combo
  converged->addStatusTest(fluidpresscombo);
  // ---------- end of fluid pressure field tests

  // Finally, return the test combo
  return combo;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::extract_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& fx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& ax)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicStructureSplit::extract_field_vectors");

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (ddgpred_ == nullptr)
  {
    FOUR_C_THROW("Vector 'ddgpred_' has not been initialized properly.");
  }
#endif

  // ---------------------------------------------------------------------------
  // process fluid unknowns
  // ---------------------------------------------------------------------------
  // extract fluid solution increment from NOX increment
  fx = extractor().extract_vector(*x, 1);

  // ---------------------------------------------------------------------------
  // process ale unknowns
  // ---------------------------------------------------------------------------
  // extract inner ALE solution increment from NOX increment
  std::shared_ptr<const Core::LinAlg::Vector<double>> aox = extractor().extract_vector(*x, 2);

  // convert fluid interface velocities into ALE interface displacements
  std::shared_ptr<Core::LinAlg::Vector<double>> fcx =
      fluid_field()->interface()->extract_fsi_cond_vector(*fx);
  fluid_field()->velocity_to_displacement(fcx);
  std::shared_ptr<Core::LinAlg::Vector<double>> acx = fluid_to_ale_interface(fcx);

  // put inner and interface ALE solution increments together
  std::shared_ptr<Core::LinAlg::Vector<double>> a =
      ale_field()->interface()->insert_other_vector(*aox);
  ale_field()->interface()->insert_fsi_cond_vector(*acx, *a);
  ax = a;

  // ---------------------------------------------------------------------------
  // process structure unknowns
  // ---------------------------------------------------------------------------
  // extract inner structure solution increment from NOX increment
  std::shared_ptr<const Core::LinAlg::Vector<double>> sox = extractor().extract_vector(*x, 0);

  // convert ALE interface displacements to structure interface displacements
  std::shared_ptr<Core::LinAlg::Vector<double>> scx = ale_to_struct(acx);
  scx->update(-1.0, *ddgpred_, 1.0);

  // put inner and interface structure solution increments together
  std::shared_ptr<Core::LinAlg::Vector<double>> s =
      structure_field()->interface()->insert_other_vector(*sox);
  structure_field()->interface()->insert_fsi_cond_vector(*scx, *s);
  sx = s;

  // ---------------------------------------------------------------------------

  // Store field vectors to know them later on as previous quantities
  if (soliprev_ != nullptr)
    ddiinc_->update(1.0, *sox, -1.0, *soliprev_, 0.0);  // compute current iteration increment
  else
    ddiinc_ = std::make_shared<Core::LinAlg::Vector<double>>(*sox);  // first iteration increment

  soliprev_ = sox;  // store current step increment

  if (disgprev_ != nullptr)
    ddginc_->update(1.0, *scx, -1.0, *disgprev_, 0.0);  // compute current iteration increment
  else
    ddginc_ = std::make_shared<Core::LinAlg::Vector<double>>(*scx);  // first iteration increment

  disgprev_ = scx;  // store current step increment

  if (velgprev_ != nullptr)
    duginc_->update(1.0, *fcx, -1.0, *velgprev_, 0.0);  // compute current iteration increment
  else
    duginc_ = std::make_shared<Core::LinAlg::Vector<double>>(*fcx);  // first iteration increment

  velgprev_ = fcx;  // store current step increment
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::output()
{
  structure_field()->output();

  // output Lagrange multiplier
  output_lambda();

  fluid_field()->output();
  ale_field()->output();

  if (use_old_structure_)
  {
    if (structure_field()->get_constraint_manager()->have_monitor())
    {
      structure_field()->get_constraint_manager()->compute_monitor_values(
          *structure_field()->dispnp());
      if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        structure_field()->get_constraint_manager()->print_monitor_values();
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::output_lambda()
{
  /* 'lambda_' is only defined on the interface. So, insert 'lambda_' into
   * 'lambdafull' that is defined on the entire structure field. Then, write
   * output or restart data.
   */
  std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
      structure_field()->interface()->insert_fsi_cond_vector(*lambda_);
  const int uprestart = timeparams_.get<int>("RESTARTEVERY");
  const int upres = timeparams_.get<int>("RESULTSEVERY");
  if ((uprestart != 0 && fluid_field()->step() % uprestart == 0) or
      (upres != 0 and fluid_field()->step() % upres == 0))
    structure_field()->disc_writer()->write_vector("fsilambda", lambdafull);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::update()
{
  lambdaold_->update(1.0, *lambda_, 0.0);

  FSI::BlockMonolithic::update();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::read_restart(int step)
{
  const bool restartfrompartfsi = timeparams_.get<bool>("RESTART_FROM_PART_FSI");

  // read Lagrange multiplier
  if (not restartfrompartfsi)  // standard restart
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
        std::make_shared<Core::LinAlg::Vector<double>>(*structure_field()->dof_row_map(), true);
    Core::IO::DiscretizationReader reader =
        Core::IO::DiscretizationReader(*structure_field()->discretization(),
            Global::Problem::instance()->input_control_file(), step);
    reader.read_vector(lambdafull, "fsilambda");
    lambdaold_ = structure_field()->interface()->extract_fsi_cond_vector(*lambdafull);
  }

  structure_field()->read_restart(step);
  fluid_field()->read_restart(step);
  ale_field()->read_restart(step);

  set_time_step(fluid_field()->time(), fluid_field()->step());
}

/*----------------------------------------------------------------------*/
/* Recover the Lagrange multiplier at the interface   mayr.mt (03/2012) */
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::recover_lagrange_multiplier()
{
  // get time integration parameter of structural time integrator
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();

  // some scaling factors for fluid
  //  const double timescale = fluid_field()->TimeScaling();

  // some often re-used vectors
  std::shared_ptr<Core::LinAlg::Vector<double>> tmpvec =
      nullptr;  // stores intermediate result of terms (3)-(8)
  std::shared_ptr<Core::LinAlg::Vector<double>> auxvec = nullptr;     // just for convenience
  std::shared_ptr<Core::LinAlg::Vector<double>> auxauxvec = nullptr;  // just for convenience

  /* Recovery of Lagrange multiplier lambda^{n+1} is done by the following
   * condensation expression:
   *
   * lambda^{n+1} =
   *
   * (1)  - stiparam / (1.-stiparam) * lambda^{n}
   *
   * (2)  + 1. / (1.-stiparam) * tmpvec
   *
   * with tmpvec =
   *
   * (3)    r_{\Gamma}^{S,n+1}
   *
   * (4)  + S_{\Gamma I} * \Delta d_{I}^{S,n+1}
   *
   * (5)  + tau * S_{\Gamma\Gamma} * \Delta u_{\Gamma}^{F,n+1}
   *
   * (6)  + dt * S_{\Gamma\Gamma} * u_{\Gamma}^n]
   *
   * Remark on term (6):
   * Term (6) has to be considered only in the first Newton iteration.
   * Hence, it will usually not be computed since in general we need more
   * than one nonlinear iteration until convergence.
   *
   * Remarks on all terms:
   * +  Division by (1.0 - stiparam) will be done in the end
   *    since this is common to all terms
   * +  tau: time scaling factor for interface time integration (tau =
   * 1/fluid_field()->TimeScaling())
   * +  neglecting terms (4)-(6) should not alter the results significantly
   *    since at the end of the time step the solution increments tend to zero.
   *
   *                                                 Matthias Mayr (10/2012)
   */

  // ---------Addressing term (1)
  lambda_->update(-stiparam, *lambdaold_, 0.0);
  // ---------End of term (1)

  // ---------Addressing term (3)
  std::shared_ptr<Core::LinAlg::Vector<double>> structureresidual =
      structure_field()->interface()->extract_fsi_cond_vector(*structure_field()->rhs());
  if (use_old_structure_)
    structureresidual->scale(-1.0);  // invert sign to obtain residual, not rhs
  tmpvec = std::make_shared<Core::LinAlg::Vector<double>>(*structureresidual);
  // ---------End of term (3)

  /* You might want to comment out terms (4) to (6) since they tend to
   * introduce oscillations in the Lagrange multiplier field for certain
   * material properties of the structure.
   *                                                    Matthias Mayr 11/2012
  // ---------Addressing term (4)
  auxvec = Teuchos::rcp(new Core::LinAlg::Vector<double>(sgiprev_->RangeMap(),true));
  sgiprev_->Apply(*ddiinc_,*auxvec);
  tmpvec->Update(1.0,*auxvec,1.0);
  // ---------End of term (4)

  // ---------Addressing term (5)
  auxvec = Teuchos::rcp(new Core::LinAlg::Vector<double>(sggprev_->RangeMap(),true));
  sggprev_->Apply(*fluid_to_struct(duginc_),*auxvec);
  tmpvec->Update(1.0/timescale,*auxvec,1.0);
  // ---------End of term (5)

  //---------Addressing term (6)
  if (firstcall_)
  {
    auxvec = Teuchos::rcp(new Core::LinAlg::Vector<double>(sggprev_->RangeMap(),true));
    sggprev_->Apply(*fluid_to_struct(fluid_field()->extract_interface_veln()),*auxvec);
    tmpvec->Update(Dt(),*auxvec,1.0);
  }
  // ---------End of term (6)
   *
   */

  // ---------Addressing term (2)
  lambda_->update(1.0, *tmpvec, 1.0);
  // ---------End of term (2)

  // finally, divide by -(1.-stiparam) which is common to all terms
  lambda_->scale(1. / (1.0 - stiparam));

  // Finally, the Lagrange multiplier 'lambda_' is recovered here.
  // It represents nodal forces acting onto the structure.

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::calculate_interface_energy_increment()
{
  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  // interface traction weighted by time integration factors
  std::shared_ptr<Core::LinAlg::Vector<double>> tractionstructure =
      std::make_shared<Core::LinAlg::Vector<double>>(lambda_->get_map(), true);
  tractionstructure->update(stiparam - ftiparam, *lambdaold_, ftiparam - stiparam, *lambda_, 0.0);

  // displacement increment of this time step
  std::shared_ptr<Core::LinAlg::Vector<double>> deltad =
      std::make_shared<Core::LinAlg::Vector<double>>(*structure_field()->dof_row_map(), true);
  deltad->update(1.0, *structure_field()->dispnp(), -1.0, *structure_field()->dispn(), 0.0);

  // calculate the energy increment
  double energy = 0.0;
  tractionstructure->dot(
      *structure_field()->interface()->extract_fsi_cond_vector(*deltad), &energy);

  energysum_ += energy;

  write_interface_energy_file(energy, energysum_);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MonolithicStructureSplit::combine_field_vectors(Core::LinAlg::Vector<double>& v,
    std::shared_ptr<const Core::LinAlg::Vector<double>> sv,
    std::shared_ptr<const Core::LinAlg::Vector<double>> fv,
    std::shared_ptr<const Core::LinAlg::Vector<double>> av,
    const bool slave_vectors_contain_interface_dofs)
{
  if (slave_vectors_contain_interface_dofs)
  {
    // extract inner DOFs from slave vectors
    std::shared_ptr<Core::LinAlg::Vector<double>> sov =
        structure_field()->interface()->extract_other_vector(*sv);
    std::shared_ptr<Core::LinAlg::Vector<double>> aov =
        ale_field()->interface()->extract_other_vector(*av);

    // put them together
    FSI::Monolithic::combine_field_vectors(v, *sov, *fv, *aov);
  }
  else
    FSI::Monolithic::combine_field_vectors(v, *sv, *fv, *av);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double FSI::MonolithicStructureSplit::select_dt_error_based() const
{
  // get time step size suggestions based on some error norms
  const double dtfl = get_ada_fl_dt();               // based on all fluid DOFs
  const double dtflfsi = get_ada_fl_fsi_dt();        // based on fluid FSI DOFs
  const double dtstrinner = get_ada_str_inner_dt();  // based on inner structural DOFs

  double dt = MonolithicStructureSplit::dt();

  // select time step size based on error estimation
  if (is_ada_structure() and is_ada_fluid())
    dt = std::min(std::min(dtfl, dtflfsi), dtstrinner);
  else if (is_ada_structure() and (not is_ada_fluid()))
    dt = dtstrinner;
  else if ((not is_ada_structure()) and is_ada_fluid())
    dt = std::min(dtfl, dtflfsi);
  else
  {
    // no change in time step size based on structure or fluid field error estimation
  }

  return dt;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool FSI::MonolithicStructureSplit::set_accepted() const
{
  // get error norms
  const double flnorm = get_ada_flnorm();               // based on all fluid DOFs
  const double flfsinorm = get_ada_fl_fs_inorm();       // based on fluid FSI DOFs
  const double strinnernorm = get_ada_str_innernorm();  // based on inner structural DOFs

  bool accepted = std::max(flnorm, flfsinorm) < errtolfl_ && strinnernorm < errtolstr_;

  // in case error estimation in the fluid field is turned off:
  if (not is_ada_fluid()) accepted = strinnernorm < errtolstr_;

  // in case error estimation in the structure field is turned off:
  if (not is_ada_structure()) accepted = std::max(flnorm, flfsinorm) < errtolfl_;

  // no error based time adaptivity
  if ((not is_ada_structure()) and (not is_ada_fluid())) accepted = true;

  return accepted;
}

FOUR_C_NAMESPACE_CLOSE
