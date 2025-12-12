// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_mortarmonolithic_structuresplit.hpp"

#include "4C_adapter_ale_fsi.hpp"
#include "4C_adapter_fld_fluid_fsi.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_coupling_adapter_mortar.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fsi_statustest.hpp"
#include "4C_fsi_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::MortarMonolithicStructureSplit::MortarMonolithicStructureSplit(
    MPI_Comm comm, const Teuchos::ParameterList& timeparams)
    : BlockMonolithic(comm, timeparams),
      comm_(comm),
      lambda_(nullptr),
      lambdaold_(nullptr),
      energysum_(0.0)
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
    //    std::cout << "Slave interface nodes with Dirichlet boundary condition "
    //                 "(input file numbering):" << std::endl;
    //    for (int i=0; i < (int)fluid_field()->discretization()->NumMyRowNodes(); i++)
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
  // ---------------------------------------------------------------------------

  notsetup_ = true;

  coupsfm_ = std::make_shared<Coupling::Adapter::CouplingMortar>(
      Global::Problem::instance()->n_dim(), Global::Problem::instance()->mortar_coupling_params(),
      Global::Problem::instance()->contact_dynamic_params(),
      Global::Problem::instance()->spatial_approximation_type());
  fscoupfa_ = std::make_shared<Coupling::Adapter::Coupling>();

  aigtransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  fmiitransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  fmgitransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  fsaigtransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  fsmgitransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();

  set_lambda();
  ddiinc_ = nullptr;
  disiprev_ = nullptr;
  disgprev_ = nullptr;
  sgiprev_ = nullptr;
  sggprev_ = nullptr;

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (coupsfm_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'coupsfm_' failed.");
  }
  if (fscoupfa_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'fscoupfa_' failed.");
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

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::set_lambda()
{
  lambda_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->interface()->fsi_cond_map(), true);
  lambdaold_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->interface()->fsi_cond_map(), true);

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::setup_system()
{
  if (notsetup_)
  {
    const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
    const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");
    linearsolverstrategy_ =
        Teuchos::getIntegralValue<Inpar::FSI::LinearBlockSolver>(fsimono, "LINEARBLOCKSOLVER");

    aleproj_ = Teuchos::getIntegralValue<Inpar::FSI::SlideALEProj>(fsidyn, "SLIDEALEPROJ");

    set_default_parameters(fsidyn, nox_parameter_list());

    // we use non-matching meshes at the interface
    // mortar with: structure = slave, fluid = master

    const int ndim = Global::Problem::instance()->n_dim();

    // get coupling objects
    Coupling::Adapter::Coupling& icoupfa = interface_fluid_ale_coupling();

    /* structure to fluid
     * coupling condition at the fsi interface:
     * displacements (=number spatial dimensions) are coupled
     * e.g.: 3D: coupleddof = [1, 1, 1]
     */
    std::vector<int> coupleddof(ndim, 1);

    coupsfm_->setup(fluid_field()->discretization(), structure_field()->discretization(),
        ale_field()->write_access_discretization(), coupleddof, "FSICoupling", comm_,
        Global::Problem::instance()->function_manager(),
        Global::Problem::instance()->binning_strategy_params(),
        Global::Problem::instance()->discretization_map(),
        Global::Problem::instance()->output_control_file(),
        Global::Problem::instance()->spatial_approximation_type(), false);

    // fluid to ale at the interface
    icoupfa.setup_condition_coupling(*fluid_field()->discretization(),
        fluid_field()->interface()->fsi_cond_map(), *ale_field()->discretization(),
        ale_field()->interface()->fsi_cond_map(), "FSICoupling", ndim);

    // the fluid-ale coupling always matches
    const Core::LinAlg::Map* fluidnodemap = fluid_field()->discretization()->node_row_map();
    const Core::LinAlg::Map* alenodemap = ale_field()->discretization()->node_row_map();

    Coupling::Adapter::Coupling& coupfa = fluid_ale_coupling();

    coupfa.setup_coupling(*fluid_field()->discretization(), *ale_field()->discretization(),
        *fluidnodemap, *alenodemap, ndim);

    fluid_field()->set_mesh_map(coupfa.master_dof_map());

    // create combined map
    create_combined_dof_row_map();

    // Use normal matrix for fluid equations but build (split) mesh movement
    // linearization (if requested in the input file)
    fluid_field()->use_block_matrix(false);

    // Use split structure matrix
    structure_field()->use_block_matrix();

    // build ale system matrix in split system
    ale_field()->create_system_matrix(ale_field()->interface());

    aleresidual_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*ale_field()->interface()->other_map());

    // -------------------------------------------------------------------------
    // Build the global Dirichlet map extractor
    setup_dbc_map_extractor();
    // -------------------------------------------------------------------------

    create_system_matrix();

    // set up sliding ale if necessary
    if (aleproj_ != Inpar::FSI::ALEprojection_none)
    {
      // mesh_init possibly modifies reference configuration of slave side --> recompute element
      // volume in initialize_elements()
      structure_field()->discretization()->fill_complete({
          .assign_degrees_of_freedom = false,
          .init_elements = true,
          .do_boundary_conditions = true,
      });
      // set up sliding ale utils
      slideale_ = std::make_shared<FSI::Utils::SlideAleUtils>(structure_field()->discretization(),
          fluid_field()->discretization(), *coupsfm_, false, aleproj_);

      iprojdisp_ =
          std::make_shared<Core::LinAlg::Vector<double>>(*coupsfm_->master_dof_map(), true);
      iprojdispinc_ =
          std::make_shared<Core::LinAlg::Vector<double>>(*coupsfm_->master_dof_map(), true);
    }
    notsetup_ = false;
  }

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
      if (Core::Communication::my_mpi_rank(comm_) == 0)
        std::cout << "Warning: RESTART_FROM_PART_FSI for mortar fsi is not jet implemented. For "
                     "now lambda_ is simply assumed to be zero!"
                  << std::endl;

      //      std::shared_ptr<Core::LinAlg::Vector<double>> lambdafullfluid = Teuchos::rcp(new
      //      Core::LinAlg::Vector<double>(*fluid_field()->dof_row_map(),true));
      //      Core::IO::DiscretizationReader reader =
      //      Core::IO::DiscretizationReader(fluid_field()->discretization(),restart);
      //      reader.read_vector(lambdafullfluid, "fsilambda");
      //
      //      std::shared_ptr<Core::LinAlg::Vector<double>> lambdafluid = Teuchos::rcp(new
      //      Core::LinAlg::Vector<double>(*fluid_field()->Interface()->FullMap(),true));
      //
      //      lambdafluid = fluid_field()->Interface()->extract_fsi_cond_vector(*lambdafullfluid);
      //
      //      //mortar business still has to be done here..
      //
      //      lambdaold_ = fluid_to_struct(lambdafluid);
    }
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::create_combined_dof_row_map()
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

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::setup_dbc_map_extractor()
{
  /* Dirichlet maps for structure and fluid do not intersect with interface map.
   * ALE Dirichlet map might intersect with interface map, but ALE interface
   * DOFs are not part of the final system of equations. Hence, we just need the
   * intersection of inner ALE DOFs with Dirichlet ALE DOFs.
   */
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
  if (dbcmaps_ == nullptr) FOUR_C_THROW("Creation of FSI Dirichlet map extractor failed.");

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>
FSI::MortarMonolithicStructureSplit::system_matrix() const
{
  return systemmatrix_;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::setup_rhs_residual(Core::LinAlg::Vector<double>& f)
{
  /* get time integration parameters of structure and fluid time integrators
   * to enable consistent time integration among the fields
   */
  const double stiparam = structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  // some scaling factors for fluid
  const double fluidscale = fluid_field()->residual_scaling();

  // get the Mortar matrix M
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortarp = coupsfm_->get_mortar_matrix_p();

  // get single field residuals
  const Core::LinAlg::Vector<double> sv(*structure_field()->rhs());
  const Core::LinAlg::Vector<double> fv(*fluid_field()->rhs());
  const Core::LinAlg::Vector<double> av(*ale_field()->rhs());

  // extract only inner DOFs from structure (=slave) and ALE field
  std::shared_ptr<const Core::LinAlg::Vector<double>> sov =
      structure_field()->interface()->extract_other_vector(sv);
  std::shared_ptr<const Core::LinAlg::Vector<double>> aov =
      ale_field()->interface()->extract_other_vector(av);

  // add structure interface residual to fluid interface residual considering temporal scaling
  std::shared_ptr<const Core::LinAlg::Vector<double>> scv =
      structure_field()->interface()->extract_fsi_cond_vector(sv);
  std::shared_ptr<Core::LinAlg::Vector<double>> fcv =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *fluid_field()->interface()->fsi_cond_map(), true);
  mortarp->multiply(true, *scv, *fcv);
  std::shared_ptr<Core::LinAlg::Vector<double>> modfv =
      fluid_field()->interface()->insert_fsi_cond_vector(*fcv);
  modfv->update(1.0, fv, (1.0 - ftiparam) / ((1.0 - stiparam) * fluidscale));

  // put the single field residuals together
  FSI::Monolithic::combine_field_vectors(f, *sov, *modfv, *aov);

  // add additional ale residual
  extractor().add_vector(*aleresidual_, 2, f);

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::setup_rhs_lambda(Core::LinAlg::Vector<double>& f)
{
  if (lambdaold_ != nullptr)
  {
    // get time integration parameters of structure and fluid time integrators
    // to enable consistent time integration among the fields
    const double stiparam = structure_field()->tim_int_param();
    const double ftiparam = fluid_field()->tim_int_param();

    // some scaling factors for fluid
    const double fluidscale = fluid_field()->residual_scaling();

    // get the Mortar matrix M
    std::shared_ptr<const Core::LinAlg::SparseMatrix> mortarm = coupsfm_->get_mortar_matrix_m();

    // project Lagrange multiplier field onto the master interface DOFs and consider temporal
    // scaling
    Core::LinAlg::Vector<double> lambda(mortarm->domain_map(), true);
    mortarm->multiply(true, *lambdaold_, lambda);
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
        fluid_field()->interface()->insert_fsi_cond_vector(lambda);
    lambdafull->scale((-ftiparam + (stiparam * (1.0 - ftiparam)) / (1.0 - stiparam)) / fluidscale);

    // add Lagrange multiplier
    extractor().add_vector(*lambdafull, 1, f);
  }

  return;
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::setup_rhs_firstiter(Core::LinAlg::Vector<double>& f)
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

  // get the Mortar projection matrix P = D^{-1} * M
  const std::shared_ptr<const Core::LinAlg::SparseMatrix> mortarp = coupsfm_->get_mortar_matrix_p();

  // get fluid shape derivatives matrix
  const std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase> mmm =
      fluid_field()->shape_derivatives();

  // get structure matrix
  const std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase> blocks =
      structure_field()->block_system_matrix();

  // get ale matrix
  const std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase> blocka =
      ale_field()->block_system_matrix();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (mortarp == nullptr) FOUR_C_THROW("Expected Teuchos::rcp to mortar matrix P.");
  if (blocks == nullptr) FOUR_C_THROW("Expected Teuchos::rcp to structure block matrix.");
  if (blocka == nullptr) FOUR_C_THROW("Expected Teuchos::rcp to ALE block matrix.");
#endif

  // extract submatrices
  const Core::LinAlg::SparseMatrix& sig = blocks->matrix(0, 1);  // S_{I\Gamma}
  const Core::LinAlg::SparseMatrix& sgg = blocks->matrix(1, 1);  // S_{\Gamma\Gamma}
  const Core::LinAlg::SparseMatrix& aig = blocka->matrix(0, 1);  // A_{I\Gamma}

  // some often re-used vectors
  std::shared_ptr<Core::LinAlg::Vector<double>> rhs =
      nullptr;  // right hand side of single set of DOFs
  std::shared_ptr<Core::LinAlg::Vector<double>> auxvec = nullptr;  // just for convenience
  std::shared_ptr<Core::LinAlg::Vector<double>> tmpvec = nullptr;  // just for convenience

  /* Different contributions/terms to the rhs are separated by the following
   * comment line */
  // ---------- inner structure DOFs
  /* The following terms are added to the inner structure DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  - dt * S_{I \Gamma} * P * u^{n}_{\Gamma}
   *
   * (2)  + S_{I \Gamma} * \Delta d_{\Gamma,p}
   *
   * Remarks on all terms:
   * +  tau: time scaling factor for interface time integration
   *         (tau = 1/fluid_field()->TimeScaling())
   *
   */
  // ----------addressing term 1
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(sig.range_map(), true);
  auxvec = std::make_shared<Core::LinAlg::Vector<double>>(mortarp->range_map(), true);

  mortarp->Apply(*fveln, *auxvec);
  sig.Apply(*auxvec, *rhs);

  rhs->scale(-dt());

  extractor().add_vector(*rhs, 0, f);
  // ----------end of term 1

  // ----------addressing term 2
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(sig.range_map(), true);

  sig.Apply(*ddgpred_, *rhs);

  extractor().add_vector(*rhs, 0, f);
  // ----------end of term 2
  // ----------end of inner structure DOFs

  // ---------- inner fluid DOFs
  /* The following terms are added to the inner fluid DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  - dt * F^{G}_{I \Gamma} * u^{n}_{\Gamma}
   *
   */
  // ----------addressing term 1
  if (mmm != nullptr)
  {
    // extract F^{G}_{I \Gamma}
    const Core::LinAlg::SparseMatrix& fmig = mmm->matrix(0, 1);

    rhs = std::make_shared<Core::LinAlg::Vector<double>>(fmig.range_map(), true);

    fmig.Apply(*fveln, *rhs);

    rhs->scale(-dt());
    rhs = fluid_field()->interface()->insert_other_vector(*rhs);

    extractor().add_vector(*rhs, 1, f);
  }
  // ----------end of term 1
  // ----------end of inner fluid DOFs

  // ---------- interface fluid DOFs
  /* The following terms are added to the interface fluid DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  - dt * F^{G}_{\Gamma\Gamma} * u^{n}_{\Gamma}
   *
   * (2)  - (1-ftiparam) / (1-stiparam) * dt * P^{T} * S_{\Gamma\Gamma} * P * u^{n}_{\Gamma}
   *
   * (3)  + (1-ftiparam) / (1-stiparam) * P^{T} * S_{\Gamma\Gamma} * \Delta d_{\Gamma,p}
   *
   * Remarks on all terms:
   * +  tau: time scaling factor for interface time integration (tau =
   * 1/fluid_field()->TimeScaling())
   *
   */
  // ----------addressing term 1
  if (mmm != nullptr)
  {
    // extract F^{G}_{\Gamma\Gamma}
    const Core::LinAlg::SparseMatrix& fmgg = mmm->matrix(1, 1);

    rhs = std::make_shared<Core::LinAlg::Vector<double>>(fmgg.range_map(), true);

    fmgg.Apply(*fveln, *rhs);

    rhs->scale(-dt());
    rhs = fluid_field()->interface()->insert_fsi_cond_vector(*rhs);

    extractor().add_vector(*rhs, 1, f);
  }
  // ----------end of term 1

  // ----------addressing term 2
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(mortarp->domain_map(), true);
  auxvec = std::make_shared<Core::LinAlg::Vector<double>>(sgg.range_map(), true);
  tmpvec = std::make_shared<Core::LinAlg::Vector<double>>(mortarp->range_map(), true);

  mortarp->Apply(*fveln, *tmpvec);
  sgg.Apply(*tmpvec, *auxvec);
  mortarp->multiply(true, *auxvec, *rhs);

  rhs->scale(-(1. - ftiparam) / (1. - stiparam) * dt() / scale);
  rhs = fluid_field()->interface()->insert_fsi_cond_vector(*rhs);

  extractor().add_vector(*rhs, 1, f);
  // ----------end of term 2

  // ----------addressing term 3
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(mortarp->domain_map(), true);
  auxvec = std::make_shared<Core::LinAlg::Vector<double>>(sgg.range_map(), true);

  sgg.Apply(*ddgpred_, *auxvec);
  mortarp->multiply(true, *auxvec, *rhs);

  rhs->scale((1. - ftiparam) / (1. - stiparam) / scale);
  rhs = fluid_field()->interface()->insert_fsi_cond_vector(*rhs);

  extractor().add_vector(*rhs, 1, f);
  // ----------end of term 3
  // ----------end of interface fluid DOFs

  // ---------- inner ALE DOFs
  /* The following terms are added to the inner ALE DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  - dt * A_{I \Gamma} * u^{n}_{\Gamma}
   *
   */
  // ----------addressing term 1
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(aig.range_map(), true);

  aig.Apply(*fluid_to_ale_interface(fveln), *rhs);

  rhs->scale(-dt());

  extractor().add_vector(*rhs, 2, f);
  // ----------end of term 1
  // ----------end of inner ALE DOFs

  // only if relative movement between ale and structure is possible
  if (aleproj_ != Inpar::FSI::ALEprojection_none)
  {
    rhs = std::make_shared<Core::LinAlg::Vector<double>>(aig.row_map(), true);

    aig.Apply(*fluid_to_ale_interface(iprojdispinc_), *rhs);

    extractor().add_vector(*rhs, 2, f);
  }

  /* Reset quantities for previous iteration step since they still store values
   * from the last time step */
  ddiinc_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->interface()->other_map(), true);
  disiprev_ = nullptr;
  disgprev_ = nullptr;
  sgicur_ = nullptr;
  sggcur_ = nullptr;

  return;
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::setup_system_matrix(
    Core::LinAlg::BlockSparseMatrixBase& mat)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MortarMonolithicStructureSplit::setup_system_matrix");

  // get the Mortar projection matrix P = D^{-1} * M
  std::shared_ptr<Core::LinAlg::SparseMatrix> mortarp = coupsfm_->get_mortar_matrix_p();

  // get single field block matrices
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> s =
      structure_field()->block_system_matrix();
  const std::shared_ptr<Core::LinAlg::SparseMatrix> f = fluid_field()->system_matrix();
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> a = ale_field()->block_system_matrix();


#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check whether allocation was successful
  if (mortarp == nullptr) FOUR_C_THROW("Expected Teuchos::rcp to mortar matrix P.");
  if (s == nullptr)
  {
    FOUR_C_THROW("expect structure block matrix");
  }
  if (f == nullptr)
  {
    FOUR_C_THROW("expect fluid matrix");
  }
  if (a == nullptr)
  {
    FOUR_C_THROW("expect ale block matrix");
  }

  // some checks whether maps for matrix-matrix-multiplication do really match
  if (!s->matrix(0, 1).domain_map().point_same_as(mortarp->range_map()))
    FOUR_C_THROW("Maps do not match.");
  if (!s->matrix(1, 0).range_map().point_same_as(mortarp->range_map()))
    FOUR_C_THROW("Maps do not match.");
  if (!s->matrix(1, 1).domain_map().point_same_as(mortarp->range_map()))
    FOUR_C_THROW("Maps do not match.");
#endif

  // extract submatrices
  Core::LinAlg::SparseMatrix& aii = a->matrix(0, 0);
  const Core::LinAlg::SparseMatrix& aig = a->matrix(0, 1);

  // scaling factors for fluid
  const double scale = fluid_field()->residual_scaling();
  const double timescale = fluid_field()->time_scaling();

  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  // Incomplete fluid matrix to be able to deal with slightly defective
  // interface meshes.
  f->un_complete();

  // ---------------------------------------------------------------------------
  // BEGIN building the global 4x4 system matrix
  // ---------------------------------------------------------------------------
  // Contributions to blocks in system matrix are listed separately.
  // Block numbering in comments ranges from (1,1) to (4,4).

  mat.assign(0, 0, Core::LinAlg::DataAccess::Share, s->matrix(0, 0));

  // ----------Addressing contribution to block (1,3)
  std::shared_ptr<Core::LinAlg::SparseMatrix> sig =
      matrix_multiply(s->matrix(0, 1), false, *mortarp, false, false, false, true);
  Core::LinAlg::SparseMatrix lsig(sig->row_map(), 81, false);

  Core::LinAlg::matrix_add(*sig, false, 1. / timescale, lsig, 0.0);
  lsig.complete(f->domain_map(), sig->range_map());

  mat.assign(0, 1, Core::LinAlg::DataAccess::Share, lsig);

  // ----------Addressing contribution to block (3,1)
  std::shared_ptr<Core::LinAlg::SparseMatrix> sgi =
      matrix_multiply(*mortarp, true, s->matrix(1, 0), false, false, false, true);
  Core::LinAlg::SparseMatrix lsgi(f->row_map(), 81, false);

  Core::LinAlg::matrix_add(*sgi, false, (1. - ftiparam) / ((1. - stiparam) * scale), lsgi, 0.0);
  lsgi.complete(sgi->domain_map(), f->range_map());

  mat.assign(1, 0, Core::LinAlg::DataAccess::Share, lsgi);

  // ----------Addressing contribution to block (3,3)
  std::shared_ptr<Core::LinAlg::SparseMatrix> sgg =
      matrix_multiply(s->matrix(1, 1), false, *mortarp, false, false, false, true);
  sgg = matrix_multiply(*mortarp, true, *sgg, false, false, false, true);

  Core::LinAlg::matrix_add(
      *sgg, false, (1. - ftiparam) / ((1. - stiparam) * scale * timescale), *f, 1.0);
  mat.assign(1, 1, Core::LinAlg::DataAccess::Share, *f);

  (*aigtransform_)(a->full_row_map(), a->full_col_map(), aig, 1. / timescale,
      Coupling::Adapter::CouplingSlaveConverter(interface_fluid_ale_coupling()), mat.matrix(2, 1));
  mat.assign(2, 2, Core::LinAlg::DataAccess::Share, aii);

  /*--------------------------------------------------------------------------*/
  // add optional fluid linearization with respect to mesh motion block

  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mmm = fluid_field()->shape_derivatives();
  if (mmm != nullptr)
  {
    // extract submatrices
    const Core::LinAlg::SparseMatrix& fmii = mmm->matrix(0, 0);
    const Core::LinAlg::SparseMatrix& fmig = mmm->matrix(0, 1);
    const Core::LinAlg::SparseMatrix& fmgi = mmm->matrix(1, 0);
    const Core::LinAlg::SparseMatrix& fmgg = mmm->matrix(1, 1);

    // ----------Addressing contribution to block (3,3)
    Core::LinAlg::matrix_add(fmgg, false, 1. / timescale, mat.matrix(1, 1), 1.0);

    // ----------Addressing contribution to block (2,3)
    Core::LinAlg::matrix_add(fmig, false, 1. / timescale, mat.matrix(1, 1), 1.0);

    const Coupling::Adapter::Coupling& coupfa = fluid_ale_coupling();

    (*fmgitransform_)(mmm->full_row_map(), mmm->full_col_map(), fmgi, 1.,
        Coupling::Adapter::CouplingMasterConverter(coupfa), mat.matrix(1, 2), false, false);

    (*fmiitransform_)(mmm->full_row_map(), mmm->full_col_map(), fmii, 1.,
        Coupling::Adapter::CouplingMasterConverter(coupfa), mat.matrix(1, 2), false, true);
  }

  // done. make sure all blocks are filled.
  mat.complete();

  // Finally, take care of Dirichlet boundary conditions
  mat.apply_dirichlet(*(dbcmaps_->cond_map()), true);
  //
  // ---------------------------------------------------------------------------
  // END building the global system matrix
  // ---------------------------------------------------------------------------

  // store parts of structural matrix to know them in the next iteration as previous iteration
  // matrices
  sgiprev_ = sgicur_;
  sggprev_ = sggcur_;
  sgicur_ = std::make_shared<Core::LinAlg::SparseMatrix>(s->matrix(1, 0));
  sggcur_ = std::make_shared<Core::LinAlg::SparseMatrix>(s->matrix(1, 1));
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::update()
{
  lambdaold_->update(1.0, *lambda_, 0.0);

  // update history variables for sliding ale
  if (aleproj_ != Inpar::FSI::ALEprojection_none)
  {
    iprojdisp_ = std::make_shared<Core::LinAlg::Vector<double>>(*coupsfm_->master_dof_map(), true);
    std::shared_ptr<Core::LinAlg::Vector<double>> idispale = ale_to_fluid_interface(
        ale_field()->interface()->extract_fsi_cond_vector(*ale_field()->dispnp()));

    slideale_->remeshing(*structure_field(), *fluid_field()->discretization(), *idispale,
        *iprojdisp_, *coupsfm_, get_comm());

    iprojdispinc_->update(-1.0, *iprojdisp_, 1.0, *idispale, 0.0);

    slideale_->evaluate_mortar(
        *structure_field()->extract_interface_dispnp(), *iprojdisp_, *coupsfm_);
    slideale_->evaluate_fluid_mortar(idispale, iprojdisp_);

    std::shared_ptr<Core::LinAlg::Vector<double>> temp =
        std::make_shared<Core::LinAlg::Vector<double>>(*iprojdisp_);
    temp->replace_map(idispale->get_map());
    std::shared_ptr<Core::LinAlg::Vector<double>> acx = fluid_to_ale_interface(temp);
    ale_field()->apply_interface_displacements(acx);
    fluid_field()->apply_mesh_displacement(ale_to_fluid(ale_field()->dispnp()));

    std::shared_ptr<Core::LinAlg::Vector<double>> unew =
        slideale_->interpolate_fluid(*fluid_field()->extract_interface_velnp());
    fluid_field()->apply_interface_velocities(unew);
  }

  // call update()-routine in base class to handle the single fields
  FSI::BlockMonolithic::update();
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::scale_system(
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


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::unscale_solution(Core::LinAlg::BlockSparseMatrixBase& mat,
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
  mat.Apply(x, r);
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
  utils()->out() << std::scientific << "\nlinear solver quality:\n"
                 << "L_2-norms:\n"
                 << "   |r|=" << n << "   |rs|=" << ns << "   |rf|=" << nf << "   |ra|=" << na
                 << "\n";
  r.norm_inf(&n);
  sr->norm_inf(&ns);
  fr->norm_inf(&nf);
  ar->norm_inf(&na);
  utils()->out() << "L_inf-norms:\n"
                 << "   |r|=" << n << "   |rs|=" << ns << "   |rf|=" << nf << "   |ra|=" << na
                 << "\n";

  utils()->out().flags(flags);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::StatusTest::Combo> FSI::MortarMonolithicStructureSplit::create_status_test(
    Teuchos::ParameterList& nlParams, Teuchos::RCP<::NOX::Abstract::Group> grp)
{
  // ---------------------------------------------------------------------------
  // Setup the test framework
  // ---------------------------------------------------------------------------
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


  // ---------------------------------------------------------------------------
  // setup tests for structural displacement field
  // ---------------------------------------------------------------------------
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

  // ---------------------------------------------------------------------------
  // setup tests for interface
  // ---------------------------------------------------------------------------
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

  // ---------------------------------------------------------------------------
  // setup tests for fluid velocity field
  // ---------------------------------------------------------------------------
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

  // ---------------------------------------------------------------------------
  // setup tests for fluid pressure field
  // ---------------------------------------------------------------------------
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


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::extract_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& fx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& ax)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MortarMonolithicStructureSplit::extract_field_vectors");

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (ddgpred_ == nullptr) FOUR_C_THROW("Vector 'ddgpred_' has not been initialized properly.");
#endif

  // get the Mortar projection matrix P = D^{-1} * M
  const std::shared_ptr<const Core::LinAlg::SparseMatrix> mortarp = coupsfm_->get_mortar_matrix_p();

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
  std::shared_ptr<Core::LinAlg::Vector<double>> scx =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *structure_field()->interface()->fsi_cond_map());
  acx = ale_to_fluid_interface(acx);
  mortarp->Apply(*acx, *scx);
  scx->update(-1.0, *ddgpred_, 1.0);

  // put inner and interface structure solution increments together
  std::shared_ptr<Core::LinAlg::Vector<double>> s =
      structure_field()->interface()->insert_other_vector(*sox);
  structure_field()->interface()->insert_fsi_cond_vector(*scx, *s);
  sx = s;

  // ---------------------------------------------------------------------------

  // Store field vectors to know them later on as previous quantities
  if (disiprev_ != nullptr)
    ddiinc_->update(1.0, *sox, -1.0, *disiprev_, 0.0);  // compute current iteration increment
  else
    ddiinc_ = std::make_shared<Core::LinAlg::Vector<double>>(*sox);  // first iteration increment

  disiprev_ = sox;  // store current step increment

  if (velgprev_ != nullptr)
    duginc_->update(1.0, *fcx, -1.0, *velgprev_, 0.0);  // compute current iteration increment
  else
    duginc_ = std::make_shared<Core::LinAlg::Vector<double>>(*fcx);  // first iteration increment

  velgprev_ = fcx;  // store current step increment
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::output()
{
  structure_field()->output();

  // output Lagrange multiplier
  output_lambda();

  fluid_field()->output();

  if (aleproj_ != Inpar::FSI::ALEprojection_none)
  {
    int uprestart = timeparams_.get<int>("RESTARTEVERY");
    if (uprestart != 0 && fluid_field()->step() % uprestart == 0)
    {
      fluid_field()->disc_writer()->write_vector("slideALE", iprojdisp_);
      fluid_field()->disc_writer()->write_vector("slideALEincr", iprojdispinc_);
      slideale_->output_restart(*fluid_field()->disc_writer());
    }
  }
  ale_field()->output();

  if (structure_field()->get_constraint_manager()->have_monitor())
  {
    structure_field()->get_constraint_manager()->compute_monitor_values(
        *structure_field()->dispnp());
    if (Core::Communication::my_mpi_rank(comm_) == 0)
      structure_field()->get_constraint_manager()->print_monitor_values();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::output_lambda()
{
  /* 'lambda_' is only defined on the interface. So, insert 'lambda_' into

   * output or restart data.
   */
  std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
      structure_field()->interface()->insert_fsi_cond_vector(*lambda_);
  const int uprestart = timeparams_.get<int>("RESTARTEVERY");
  const int upres = timeparams_.get<int>("RESULTSEVERY");
  if ((uprestart != 0 and fluid_field()->step() % uprestart == 0) or
      (upres != 0 and fluid_field()->step() % upres == 0))
    structure_field()->disc_writer()->write_vector("fsilambda", lambdafull);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::read_restart(int step)
{
  const bool restartfrompartfsi = timeparams_.get<bool>("RESTART_FROM_PART_FSI");
  auto input_control_file = Global::Problem::instance()->input_control_file();

  // read Lagrange multiplier
  if (not restartfrompartfsi)  // standard restart
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
        std::make_shared<Core::LinAlg::Vector<double>>(*structure_field()->dof_row_map(), true);
    Core::IO::DiscretizationReader reader = Core::IO::DiscretizationReader(
        *structure_field()->discretization(), input_control_file, step);
    reader.read_vector(lambdafull, "fsilambda");
    lambdaold_ = structure_field()->interface()->extract_fsi_cond_vector(*lambdafull);
    // Note: the above is normally enough. However, we can use the restart in order to periodically
    // repeat the fsi simulation (see AC-FS3I)
    lambda_ = structure_field()->interface()->extract_fsi_cond_vector(*lambdafull);
  }

  structure_field()->read_restart(step);
  fluid_field()->read_restart(step);

  setup_system();

  if (aleproj_ != Inpar::FSI::ALEprojection_none)
  {
    Core::IO::DiscretizationReader reader =
        Core::IO::DiscretizationReader(*fluid_field()->discretization(), input_control_file, step);
    reader.read_vector(iprojdisp_, "slideALE");
    reader.read_vector(iprojdispinc_, "slideALEincr");
    slideale_->read_restart(reader);
  }

  ale_field()->read_restart(step);

  set_time_step(fluid_field()->time(), fluid_field()->step());

  if (aleproj_ != Inpar::FSI::ALEprojection_none)
    slideale_->evaluate_mortar(
        *structure_field()->extract_interface_dispn(), *iprojdisp_, *coupsfm_);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::recover_lagrange_multiplier()
{
  // get time integration parameter of structural time integrator
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();

  // some scaling factors for fluid
  //  const double timescale = fluid_field()->TimeScaling();

  // get the Mortar projection matrix P = D^{-1} * M
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortarp = coupsfm_->get_mortar_matrix_p();

  // get the inverted Mortar matrix D^{-1}
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortardinv = coupsfm_->get_mortar_matrix_dinv();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (mortarp == nullptr) FOUR_C_THROW("Expected Teuchos::rcp to mortar matrix P.");
  if (mortardinv == nullptr) FOUR_C_THROW("Expected Teuchos::rcp to mortar matrix D^{{-1}}.");
#endif

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
   * (2)  + 1. / (1.-stiparam) * D^{-T} * tmpvec
   *
   * with tmpvec =
   *
   * (3)    r_{\Gamma}^{S,n+1}
   *
   * (4)  + S_{\Gamma I} * \Delta d_{I}^{S,n+1}
   *
   * (5)  + tau * S_{\Gamma\Gamma} * P * \Delta u_{\Gamma}^{F,n+1}
   *
   * (6)  + dt * S_{\Gamma\Gamma} * P * u_{\Gamma}^n]
   *
   * Remark on term (6):
   * Term (6) has to be considered only in the first Newton iteration.
   * Hence, it will usually not be computed since in general we need more
   * than one nonlinear iteration until convergence.
   *
   * Remarks on all terms:
   * +  Division by -(1.0 - stiparam) will be done in the end
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
      std::make_shared<Core::LinAlg::Vector<double>>(
          *structure_field()->interface()->extract_fsi_cond_vector(*structure_field()->rhs()));
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
  auxvec = Teuchos::rcp(new Core::LinAlg::Vector<double>(mortarp->RangeMap(),true));
  mortarp->Apply(*duginc_,*auxvec);
  auxauxvec = Teuchos::rcp(new Core::LinAlg::Vector<double>(sggprev_->RangeMap(),true));
  sggprev_->Apply(*auxvec,*auxauxvec);
  tmpvec->Update(1.0/timescale,*auxauxvec,1.0);
  // ---------End of term (5)

  // ---------Addressing term (6)
  if (firstcall_)
  {
    auxvec = Teuchos::rcp(new Core::LinAlg::Vector<double>(mortarp->RangeMap(),true));
    mortarp->Apply(*fluid_field()->extract_interface_veln(),*auxvec);
    auxauxvec = Teuchos::rcp(new Core::LinAlg::Vector<double>(sggprev_->RangeMap(),true));
    sggprev_->Apply(*auxvec,*auxauxvec);
    tmpvec->Update(Dt(),*auxauxvec,1.0);
  }
  // ---------End of term (6)
   *
   */

  // ---------Addressing term (2)
  auxvec = std::make_shared<Core::LinAlg::Vector<double>>(mortardinv->domain_map(), true);
  mortardinv->multiply(true, *tmpvec, *auxvec);
  lambda_->update(1.0, *auxvec, 1.0);
  // ---------End of term (2)

  // finally, divide by -(1.-stiparam) which is common to all terms
  lambda_->scale(1. / (1.0 - stiparam));

  /* Finally, the Lagrange multiplier lambda_ is recovered here. It has the
   * unit [N/m^2]. Actual nodal forces are obtained by multiplication with
   * mortar matrices M or D later on.
   */

  //  check_kinematic_constraint();
  //  check_dynamic_equilibrium();

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::calculate_interface_energy_increment()
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

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::check_kinematic_constraint()
{
  // some scaling factors for fluid
  const double timescale = fluid_field()->time_scaling();

  // get the Mortar matrices D and M
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortard = coupsfm_->get_mortar_matrix_d();
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortarm = coupsfm_->get_mortar_matrix_m();

  // get interface displacements and velocities
  std::shared_ptr<Core::LinAlg::Vector<double>> disnp =
      structure_field()->extract_interface_dispnp();
  std::shared_ptr<Core::LinAlg::Vector<double>> disn = structure_field()->extract_interface_dispn();
  std::shared_ptr<Core::LinAlg::Vector<double>> velnp = fluid_field()->extract_interface_velnp();
  std::shared_ptr<Core::LinAlg::Vector<double>> veln = fluid_field()->extract_interface_veln();

  // prepare vectors for projected interface quantities
  Core::LinAlg::Vector<double> disnpproj(mortard->range_map(), true);
  Core::LinAlg::Vector<double> disnproj(mortard->range_map(), true);
  Core::LinAlg::Vector<double> velnpproj(mortarm->range_map(), true);
  Core::LinAlg::Vector<double> velnproj(mortarm->range_map(), true);

  // projection of interface displacements
  mortard->Apply(*disnp, disnpproj);
  mortard->Apply(*disn, disnproj);

  // projection of interface velocities
  mortarm->Apply(*velnp, velnpproj);
  mortarm->Apply(*veln, velnproj);

  // calculate violation of kinematic interface constraint
  Core::LinAlg::Vector<double> violation(disnpproj);
  violation.update(-1.0, disnproj, 1.0);
  violation.update(-1.0 / timescale, velnpproj, 1.0 / timescale, velnproj, 1.0);
  violation.update(-dt(), velnproj, 1.0);

  // calculate some norms
  double violationl2 = 0.0;
  double violationinf = 0.0;
  violation.norm_2(&violationl2);
  violation.norm_inf(&violationinf);

  // scale L2-Norm with length of vector
  violationl2 /= sqrt(violation.local_length());

  // output to screen
  std::ios_base::fmtflags flags = utils()->out().flags();

  utils()->out() << std::scientific << "\nViolation of kinematic interface constraint:\n"
                 << "L_2-norm: " << violationl2 << "        L_inf-norm: " << violationinf << "\n";
  utils()->out().flags(flags);

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::check_dynamic_equilibrium()
{
  // get the Mortar matrices D and M
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortard = coupsfm_->get_mortar_matrix_d();
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortarm = coupsfm_->get_mortar_matrix_m();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (mortarm == nullptr) FOUR_C_THROW("Expected Teuchos::rcp to mortar matrix M.");
  if (mortard == nullptr) FOUR_C_THROW("Expected Teuchos::rcp to mortar matrix D.");
#endif

  // auxiliary vectors
  Core::LinAlg::Vector<double> tractionmaster(mortarm->domain_map(), true);
  Core::LinAlg::Vector<double> tractionslave(mortard->domain_map(), true);

  // calculate forces on master and slave side
  mortarm->multiply(true, *lambda_, tractionmaster);
  mortard->multiply(true, *lambda_, tractionslave);

  // calculate violation of dynamic equilibrium
  Core::LinAlg::Vector<double> violation(tractionmaster);
  violation.update(-1.0, tractionslave, 1.0);

  // calculate some norms
  double violationl2 = 0.0;
  double violationinf = 0.0;
  violation.norm_2(&violationl2);
  violation.norm_inf(&violationinf);

  // scale L2-Norm with sqrt of length of interface vector
  violationl2 /= sqrt(structure_field()->interface()->fsi_cond_map()->num_global_elements());

  // output to screen
  std::ios_base::fmtflags flags = utils()->out().flags();

  utils()->out() << std::scientific << "\nViolation of dynamic interface equilibrium:\n"
                 << "L_2-norm: " << violationl2 << "        L_inf-norm: " << violationinf << "\n";
  utils()->out().flags(flags);

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::combine_field_vectors(Core::LinAlg::Vector<double>& v,
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

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
double FSI::MortarMonolithicStructureSplit::select_dt_error_based() const
{
  // get time step size suggestions based on some error norms
  const double dtfl = get_ada_fl_dt();               // based on all fluid DOFs
  const double dtflfsi = get_ada_fl_fsi_dt();        // based on fluid FSI DOFs
  const double dtstrinner = get_ada_str_inner_dt();  // based on inner structural DOFs

  double dt = MortarMonolithicStructureSplit::dt();

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

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
bool FSI::MortarMonolithicStructureSplit::set_accepted() const
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

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MortarMonolithicStructureSplit::create_system_matrix()
{
  FSI::BlockMonolithic::create_system_matrix(systemmatrix_, true);
}

FOUR_C_NAMESPACE_CLOSE
