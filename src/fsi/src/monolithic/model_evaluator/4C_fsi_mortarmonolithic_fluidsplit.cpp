// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_mortarmonolithic_fluidsplit.hpp"

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
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
FSI::MortarMonolithicFluidSplit::MortarMonolithicFluidSplit(
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
  intersectionmaps.push_back(fluid_field()->get_dbc_map_extractor()->cond_map());
  intersectionmaps.push_back(fluid_field()->interface()->fsi_cond_map());
  std::shared_ptr<Core::LinAlg::Map> intersectionmap =
      Core::LinAlg::MultiMapExtractor::intersect_maps(intersectionmaps);

  // Check whether the intersection is empty
  if (intersectionmap->num_global_elements() != 0)
  {
    //    std::cout << "Slave interface nodes with Dirichlet boundary condition "
    //              "(input file numbering):" << std::endl;
    //    for (int i=0; i < (int)fluid_field()->discretization()->NumMyRowNodes(); i++)
    //    {
    //      // get all nodes and add them
    //      int gid = fluid_field()->discretization()->NodeRowMap()->GID(i);
    //
    //      // do only nodes that I have in my discretization
    //      if (!fluid_field()->discretization()->NodeColMap()->MyGID(gid)) continue;
    //      Core::Nodes::Node* node = fluid_field()->discretization()->gNode(gid);
    //      if (!node) FOUR_C_THROW("Cannot find node with gid %",gid);
    //
    //      std::vector<int> nodedofs = fluid_field()->discretization()->Dof(node);
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
             << "  | This is a fluid split scheme. Hence, master and slave field are chosen as "
                "follows:          |"
             << std::endl
             << "  |     MASTER  = STRUCTURE                                                       "
                "              |"
             << std::endl
             << "  |     SLAVE   = FLUID                                                           "
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

  // Recovery of Lagrange multiplier happens on fluid field
  set_lambda();

  fmgiprev_ = nullptr;
  fmgicur_ = nullptr;
  fmggprev_ = nullptr;
  fmggcur_ = nullptr;
  fgiprev_ = nullptr;
  fgicur_ = nullptr;
  fggprev_ = nullptr;
  fggcur_ = nullptr;

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
void FSI::MortarMonolithicFluidSplit::set_lambda()
{
  lambda_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *fluid_field()->interface()->fsi_cond_map(), true);
  lambdaold_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *fluid_field()->interface()->fsi_cond_map(), true);

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplit::setup_system()
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
    // mortar with: structure = master, fluid = slave

    const int ndim = Global::Problem::instance()->n_dim();

    // get coupling objects
    Coupling::Adapter::Coupling& icoupfa = interface_fluid_ale_coupling();

    /* structure to fluid
     * coupling condition at the fsi interface:
     * displacements (=number spatial dimensions) are coupled
     * e.g.: 3D: coupleddof = [1, 1, 1]
     */
    std::vector<int> coupleddof(ndim, 1);

    coupsfm_->setup(structure_field()->discretization(), fluid_field()->discretization(),
        ale_field()->write_access_discretization(), coupleddof, "FSICoupling", comm_,
        Global::Problem::instance()->function_manager(),
        Global::Problem::instance()->binning_strategy_params(),
        Global::Problem::instance()->discretization_map(),
        Global::Problem::instance()->output_control_file(),
        Global::Problem::instance()->spatial_approximation_type(), true);

    // fluid to ale at the interface

    icoupfa.setup_condition_coupling(*fluid_field()->discretization(),
        fluid_field()->interface()->fsi_cond_map(), *ale_field()->discretization(),
        ale_field()->interface()->fsi_cond_map(), "FSICoupling", ndim);

    Coupling::Adapter::Coupling& coupfa = fluid_ale_coupling();

    // the fluid-ale coupling always matches
    const Core::LinAlg::Map* fluidnodemap = fluid_field()->discretization()->node_row_map();
    const Core::LinAlg::Map* alenodemap = ale_field()->discretization()->node_row_map();

    coupfa.setup_coupling(*fluid_field()->discretization(), *ale_field()->discretization(),
        *fluidnodemap, *alenodemap, ndim);

    fluid_field()->set_mesh_map(coupfa.master_dof_map());

    // create combined map
    create_combined_dof_row_map();

    /*------------------------------------------------------------------------*/
    // Switch fluid to interface split block matrix
    fluid_field()->use_block_matrix(true);

    // build ale system matrix in split system
    ale_field()->create_system_matrix(ale_field()->interface());

    aleresidual_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*ale_field()->interface()->other_map());

    // -------------------------------------------------------------------------
    // Build the global Dirichlet map extractor
    setup_dbc_map_extractor();
    // -------------------------------------------------------------------------

    create_system_matrix();

    if (aleproj_ != Inpar::FSI::ALEprojection_none)
    {
      // set up sliding ale utils
      slideale_ = std::make_shared<FSI::Utils::SlideAleUtils>(structure_field()->discretization(),
          fluid_field()->discretization(), *coupsfm_, true, aleproj_);

      iprojdispinc_ =
          std::make_shared<Core::LinAlg::Vector<double>>(*coupsfm_->slave_dof_map(), true);
      iprojdisp_ = std::make_shared<Core::LinAlg::Vector<double>>(*coupsfm_->slave_dof_map(), true);
    }
    notsetup_ = false;
  }

  // NOTE: if we restart from a part. fsi problem we still have to read lambda_. But since this
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

      // mortar business still has to be done here..
    }
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplit::create_combined_dof_row_map()
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;
  vecSpaces.push_back(structure_field()->dof_row_map());
  vecSpaces.push_back(fluid_field()->dof_row_map());
  vecSpaces.push_back(ale_field()->interface()->other_map());

  if (vecSpaces[1]->num_global_elements() == 0)
    FOUR_C_THROW("No inner fluid equations. Splitting not possible.");

  set_dof_row_maps(vecSpaces);

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplit::setup_dbc_map_extractor()
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
FSI::MortarMonolithicFluidSplit::system_matrix() const
{
  return systemmatrix_;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplit::setup_rhs_residual(Core::LinAlg::Vector<double>& f)
{
  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  // some scaling factors for fluid
  const double fluidscale = fluid_field()->residual_scaling();

  // get the Mortar matrix P
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortarp = coupsfm_->get_mortar_matrix_p();

  // get single field residuals
  Core::LinAlg::Vector<double> sv(*structure_field()->rhs());
  const Core::LinAlg::Vector<double> fv(*fluid_field()->rhs());
  const Core::LinAlg::Vector<double> av(*ale_field()->rhs());

  // NOX treats rhs different in new solid time integration, hence sign inversion is necessary here
  if (!use_old_structure_) sv.scale(-1.0);

  // extract only inner DOFs from fluid (=slave) and ALE field
  std::shared_ptr<Core::LinAlg::Vector<double>> fov =
      fluid_field()->interface()->extract_other_vector(fv);
  fov = fluid_field()->interface()->insert_other_vector(*fov);
  std::shared_ptr<const Core::LinAlg::Vector<double>> aov =
      ale_field()->interface()->extract_other_vector(av);

  /* add fluid interface residual to structure interface residual considering
   * temporal scaling
   */
  std::shared_ptr<Core::LinAlg::Vector<double>> fcv =
      fluid_field()->interface()->extract_fsi_cond_vector(fv);
  std::shared_ptr<Core::LinAlg::Vector<double>> scv =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *structure_field()->interface()->fsi_cond_map(), true);
  mortarp->multiply(true, *fcv, *scv);
  std::shared_ptr<Core::LinAlg::Vector<double>> modsv =
      structure_field()->interface()->insert_fsi_cond_vector(*scv);
  modsv->update(1.0, sv, (1.0 - stiparam) / (1.0 - ftiparam) * fluidscale);

  if (structure_field()->get_stc_algo() == Inpar::Solid::stc_currsym)
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> stcmat = structure_field()->get_stc_mat();
    stcmat->multiply(true, *modsv, *modsv);
  }

  // put the single field residuals together
  FSI::Monolithic::combine_field_vectors(f, *modsv, *fov, *aov);

  // add additional ale residual
  extractor().add_vector(*aleresidual_, 2, f);

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplit::setup_rhs_lambda(Core::LinAlg::Vector<double>& f)
{
  if (lambdaold_ != nullptr)
  {
    // get time integration parameters of structure and fluid time integrators
    // to enable consistent time integration among the fields
    const double stiparam = structure_field()->tim_int_param();
    const double ftiparam = fluid_field()->tim_int_param();

    // get the Mortar matrix M
    const std::shared_ptr<Core::LinAlg::SparseMatrix> mortarm = coupsfm_->get_mortar_matrix_m();

    /* project Lagrange multiplier field onto the master interface DOFs and
     * consider temporal scaling */
    Core::LinAlg::Vector<double> lambda(mortarm->domain_map(), true);
    mortarm->multiply(true, *lambdaold_, lambda);
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
        structure_field()->interface()->insert_fsi_cond_vector(lambda);
    lambdafull->scale(stiparam - (ftiparam * (1.0 - stiparam)) / (1.0 - ftiparam));

    // add Lagrange multiplier
    extractor().add_vector(*lambdafull, 0, f);
  }

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplit::setup_rhs_firstiter(Core::LinAlg::Vector<double>& f)
{
  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  // some scaling factors for fluid
  const double timescale = fluid_field()->time_scaling();
  const double scale = fluid_field()->residual_scaling();

  // old interface velocity of fluid field
  const std::shared_ptr<const Core::LinAlg::Vector<double>> fveln =
      fluid_field()->extract_interface_veln();

  // get the Mortar projection matrix P = D^{-1} * M
  const std::shared_ptr<const Core::LinAlg::SparseMatrix> mortarp = coupsfm_->get_mortar_matrix_p();

  // get fluid matrix
  const std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase> blockf =
      fluid_field()->block_system_matrix();

  // get fluid shape derivatives matrix
  const std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase> mmm =
      fluid_field()->shape_derivatives();

  // get ale matrix
  const std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase> blocka =
      ale_field()->block_system_matrix();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (mortarp == nullptr) FOUR_C_THROW("Expected Teuchos::rcp to mortar matrix P.");
  if (blockf == nullptr) FOUR_C_THROW("Expected Teuchos::rcp to fluid block matrix.");
  if (blocka == nullptr) FOUR_C_THROW("Expected Teuchos::rcp to ale block matrix.");
#endif

  // extract fluid and ale submatrices
  const Core::LinAlg::SparseMatrix& fig = blockf->matrix(0, 1);  // F_{I\Gamma}
  const Core::LinAlg::SparseMatrix& fgg = blockf->matrix(1, 1);  // F_{\Gamma\Gamma}
  const Core::LinAlg::SparseMatrix& aig = blocka->matrix(0, 1);  // A_{I\Gamma}

  // some often re-used vectors
  std::shared_ptr<Core::LinAlg::Vector<double>> rhs =
      nullptr;  // right hand side of single set of DOFs
  std::shared_ptr<Core::LinAlg::Vector<double>> auxvec = nullptr;  // just for convenience
  std::shared_ptr<Core::LinAlg::Vector<double>> tmpvec = nullptr;  // just for convenience

  /* Different contributions/terms to the rhs are separated by the following
   * comment line */
  // ---------- interface structure DOFs
  /* The following terms are added to the interface structure DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  + (1-stiparam)/(1-ftiparam) * dt / tau * P^{T} * F_{\Gamma\Gamma} * u^{n}_{\Gamma}
   *
   * (2)  - (1-stiparam)/(1-ftiparam) / tau * P^{T} * F_{\Gamma\Gamma} * P * \Delta d_{\Gamma,p}
   *
   * (3)  - (1-stiparam)/(1-ftiparam) * P^{T} * F^{G}_{\Gamma\Gamma} * P * \Delta d_{\Gamma,p}
   *
   * Remarks on all terms:
   * +  tau: time scaling factor for interface time integration (tau =
   * 1/fluid_field()->TimeScaling())
   *
   */
  // ----------addressing term 1
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(mortarp->domain_map(), true);
  auxvec = std::make_shared<Core::LinAlg::Vector<double>>(fgg.row_map(), true);

  fgg.multiply(false, *fveln, *auxvec);
  mortarp->multiply(true, *auxvec, *rhs);

  if (structure_field()->get_stc_algo() == Inpar::Solid::stc_currsym)
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> stcmat = structure_field()->get_stc_mat();
    stcmat->multiply(true, *rhs, *rhs);
  }

  rhs->scale(scale * (1. - stiparam) / (1. - ftiparam) * dt() * timescale);
  rhs = structure_field()->interface()->insert_fsi_cond_vector(*rhs);

  extractor().add_vector(*rhs, 0, f);
  // ----------end of term 1

  // ----------addressing term 2
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(mortarp->domain_map(), true);
  auxvec = std::make_shared<Core::LinAlg::Vector<double>>(fgg.range_map(), true);
  tmpvec = std::make_shared<Core::LinAlg::Vector<double>>(mortarp->range_map(), true);

  mortarp->multiply(false, *ddgpred_, *tmpvec);
  fgg.multiply(false, *tmpvec, *auxvec);
  mortarp->multiply(true, *auxvec, *rhs);

  rhs->scale(-scale * (1. - stiparam) / (1. - ftiparam) * timescale);
  rhs = structure_field()->interface()->insert_fsi_cond_vector(*rhs);

  extractor().add_vector(*rhs, 0, f);
  // ----------end of term 2

  // ----------addressing term 3
  if (mmm != nullptr)
  {
    // extract F^{G}_{\Gamma\Gamma}
    const Core::LinAlg::SparseMatrix& fmgg = mmm->matrix(1, 1);

    rhs = std::make_shared<Core::LinAlg::Vector<double>>(mortarp->domain_map(), true);
    auxvec = std::make_shared<Core::LinAlg::Vector<double>>(fmgg.range_map(), true);
    tmpvec = std::make_shared<Core::LinAlg::Vector<double>>(mortarp->range_map(), true);

    mortarp->multiply(false, *ddgpred_, *tmpvec);
    fmgg.multiply(false, *tmpvec, *auxvec);
    mortarp->multiply(true, *auxvec, *rhs);

    rhs->scale(-(1. - stiparam) / (1. - ftiparam));
    rhs = structure_field()->interface()->insert_fsi_cond_vector(*rhs);

    extractor().add_vector(*rhs, 0, f);
  }
  // ----------end of term 3
  // ----------end of interface structure DOFs

  // ---------- inner fluid DOFs
  /* The following terms are added to the inner fluid DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  + dt / tau * F_{I \Gamma} * u^{n}_{\Gamma}
   *
   * (2)  - 1 / tau * F_{I \Gamma} * P * \Delta d_{\Gamma,p}
   *
   * (3)  - F^{G}_{I \Gamma} * P * \Delta d_{\Gamma,p}
   *
   * Remarks on all terms:
   * +  tau: time scaling factor for interface time integration
   *         (tau = 1/fluid_field()->TimeScaling())
   *
   */
  // ----------addressing term 1
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(fig.row_map(), true);

  fig.multiply(false, *fveln, *rhs);

  rhs->scale(dt() * timescale);

  rhs = fluid_field()->interface()->insert_other_vector(*rhs);

  extractor().add_vector(*rhs, 1, f);
  // ----------end of term 1

  // ----------addressing term 2
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(fig.range_map(), true);
  auxvec = std::make_shared<Core::LinAlg::Vector<double>>(mortarp->range_map(), true);

  mortarp->multiply(false, *ddgpred_, *auxvec);
  fig.multiply(false, *auxvec, *rhs);

  rhs->scale(-timescale);

  rhs = fluid_field()->interface()->insert_other_vector(*rhs);

  extractor().add_vector(*rhs, 1, f);
  // ----------end of term 2

  // ----------addressing term 3
  if (mmm != nullptr)
  {
    // extract F^{G}_{I \Gamma}
    const Core::LinAlg::SparseMatrix& fmig = mmm->matrix(0, 1);

    rhs = std::make_shared<Core::LinAlg::Vector<double>>(fmig.range_map(), true);
    auxvec = std::make_shared<Core::LinAlg::Vector<double>>(mortarp->range_map(), true);

    mortarp->multiply(false, *ddgpred_, *auxvec);
    fmig.multiply(false, *auxvec, *rhs);

    rhs->scale(-1.);

    rhs = fluid_field()->interface()->insert_other_vector(*rhs);

    extractor().add_vector(*rhs, 1, f);
  }
  // ----------end of term 3
  // ----------end of inner fluid DOFs

  // ---------- inner ALE DOFs
  /* The following terms are added to the inner ALE DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  - A_{I \Gamma} * P * \Delta d_{\Gamma,p}
   *
   */
  // ----------addressing term 1
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(aig.range_map(), true);
  auxvec = std::make_shared<Core::LinAlg::Vector<double>>(mortarp->range_map(), true);

  mortarp->multiply(false, *ddgpred_, *auxvec);
  aig.multiply(false, *fluid_to_ale_interface(auxvec), *rhs);

  rhs->scale(-1.0);

  extractor().add_vector(*rhs, 2, f);
  // ----------end of term 1
  // ----------end of inner ALE DOFs

  // only if relative movement between ale and structure is possible
  if (aleproj_ != Inpar::FSI::ALEprojection_none)
  {
    // get block ale matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> a = ale_field()->block_system_matrix();
    if (a == nullptr)
    {
      FOUR_C_THROW("expect ale block matrix");
    }

    rhs = std::make_shared<Core::LinAlg::Vector<double>>(a->matrix(0, 1).row_map());

    a->matrix(0, 1).multiply(false, *fluid_to_ale_interface(iprojdispinc_), *rhs);

    extractor().add_vector(*rhs, 2, f);

    // get fluid shape derivative matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mmm = fluid_field()->shape_derivatives();
    if (mmm != nullptr)
    {
      // extract submatrices
      Core::LinAlg::SparseMatrix& fmig = mmm->matrix(0, 1);
      Core::LinAlg::SparseMatrix& fmgg = mmm->matrix(1, 1);

      rhs = std::make_shared<Core::LinAlg::Vector<double>>(fmgg.row_map());

      fmgg.multiply(false, *iprojdispinc_, *rhs);

      Core::LinAlg::Vector<double> tmprhs(mortarp->domain_map());
      mortarp->multiply(true, *rhs, tmprhs);

      rhs = structure_field()->interface()->insert_fsi_cond_vector(tmprhs);

      auto zeros = std::make_shared<Core::LinAlg::Vector<double>>(rhs->get_map(), true);
      Core::LinAlg::apply_dirichlet_to_system(
          *rhs, *zeros, *(structure_field()->get_dbc_map_extractor()->cond_map()));

      if (structure_field()->get_stc_algo() == Inpar::Solid::stc_currsym)
      {
        std::shared_ptr<Core::LinAlg::SparseMatrix> stcmat = structure_field()->get_stc_mat();
        stcmat->multiply(true, *rhs, *rhs);
      }

      rhs->scale(dt() * timescale * (1. - stiparam) / (1. - ftiparam));
      extractor().add_vector(*rhs, 0, f);

      rhs = std::make_shared<Core::LinAlg::Vector<double>>(fmig.row_map());

      fmig.multiply(false, *iprojdispinc_, *rhs);

      rhs = fluid_field()->interface()->insert_other_vector(*rhs);

      zeros = std::make_shared<Core::LinAlg::Vector<double>>(rhs->get_map(), true);
      Core::LinAlg::apply_dirichlet_to_system(
          *rhs, *zeros, *(structure_field()->get_dbc_map_extractor()->cond_map()));

      rhs->scale(-timescale * dt());

      extractor().add_vector(*rhs, 1, f);
    }
  }

  // Reset quantities of previous iteration step since they still store values from the last time
  // step
  ddginc_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *structure_field()->interface()->fsi_cond_map(), true);
  duiinc_ = std::make_shared<Core::LinAlg::Vector<double>>(
      *fluid_field()->interface()->other_map(), true);
  veliprev_ = nullptr;
  velgprev_ = nullptr;
  fgicur_ = nullptr;
  fggcur_ = nullptr;

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplit::setup_system_matrix(Core::LinAlg::BlockSparseMatrixBase& mat)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicOverlap::setup_system_matrix");

  // get the Mortar projection matrix P = D^{-1} * M
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortarp = coupsfm_->get_mortar_matrix_p();

  // get info about STC feature
  Inpar::Solid::StcScale stcalgo = structure_field()->get_stc_algo();
  std::shared_ptr<Core::LinAlg::SparseMatrix> stcmat = nullptr;
  if (stcalgo != Inpar::Solid::stc_inactive) stcmat = structure_field()->get_stc_mat();

  const Coupling::Adapter::Coupling& coupfa = fluid_ale_coupling();

  // get single field block matrices
  std::shared_ptr<Core::LinAlg::SparseMatrix> s =
      structure_field()->system_matrix();  // can't be 'const' --> is modified by STC
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> f =
      fluid_field()->block_system_matrix();
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
    FOUR_C_THROW("expect fluid block matrix");
  }
  if (a == nullptr)
  {
    FOUR_C_THROW("expect ale block matrix");
  }

  // some checks whether maps for matrix-matrix-multiplication do really match
  if (!f->matrix(0, 1).domain_map().point_same_as(mortarp->range_map()))
    FOUR_C_THROW("Maps do not match.");
  if (!f->matrix(1, 0).range_map().point_same_as(mortarp->range_map()))
    FOUR_C_THROW("Maps do not match.");
  if (!f->matrix(1, 1).domain_map().point_same_as(mortarp->range_map()))
    FOUR_C_THROW("Maps do not match.");
#endif

  // extract submatrices
  Core::LinAlg::SparseMatrix& aii = a->matrix(0, 0);  // A_{II}
  Core::LinAlg::SparseMatrix& aig = a->matrix(0, 1);  // A_{I\Gamma}
  Core::LinAlg::SparseMatrix& fii = f->matrix(0, 0);  // F_{II}

  // scaling factors for fluid
  const double scale = fluid_field()->residual_scaling();
  const double timescale = fluid_field()->time_scaling();

  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  // ---------------------------------------------------------------------------
  // BEGIN building the global 4x4 system matrix
  // ---------------------------------------------------------------------------
  // Contributions to blocks in system matrix are listed separately.
  // Block numbering in comments ranges from (1,1) to (4,4).

  // ---------Addressing contribution to block (2,2)
  std::shared_ptr<Core::LinAlg::SparseMatrix> fgg =
      matrix_multiply(f->matrix(1, 1), false, *mortarp, false, false, false, true);
  fgg = matrix_multiply(*mortarp, true, *fgg, false, false, false, true);

  Core::LinAlg::matrix_add(
      *fgg, false, scale * timescale * (1. - stiparam) / (1. - ftiparam), *s, 1.0);

  // ---------Addressing contribution to block (2,3)
  std::shared_ptr<Core::LinAlg::SparseMatrix> fgi =
      matrix_multiply(*mortarp, true, f->matrix(1, 0), false, false, false, true);
  std::shared_ptr<Core::LinAlg::SparseMatrix> lfgi =
      std::make_shared<Core::LinAlg::SparseMatrix>(s->row_map(), 81, false);

  Core::LinAlg::matrix_add(*fgi, false, scale, *lfgi, 0.0);
  lfgi->complete(fgi->domain_map(), s->range_map());

  if (stcalgo == Inpar::Solid::stc_currsym)
    lfgi = Core::LinAlg::matrix_multiply(*stcmat, true, *lfgi, false, true, true, true);

  mat.matrix(0, 1).un_complete();
  Core::LinAlg::matrix_add(*lfgi, false, (1. - stiparam) / (1. - ftiparam), mat.matrix(0, 1), 0.0);

  // ---------Addressing contribution to block (3,2)
  std::shared_ptr<Core::LinAlg::SparseMatrix> fig =
      matrix_multiply(f->matrix(0, 1), false, *mortarp, false, false, false, true);
  std::shared_ptr<Core::LinAlg::SparseMatrix> lfig =
      std::make_shared<Core::LinAlg::SparseMatrix>(fig->row_map(), 81, false);

  Core::LinAlg::matrix_add(*fig, false, timescale, *lfig, 0.0);
  lfig->complete(s->domain_map(), fig->range_map());

  if (stcalgo != Inpar::Solid::stc_inactive)
  {
    lfig = Core::LinAlg::matrix_multiply(*lfig, false, *stcmat, false, false, false, true);
  }

  mat.matrix(1, 0).un_complete();
  Core::LinAlg::matrix_add(*lfig, false, 1., mat.matrix(1, 0), 0.0);

  // ---------Addressing contribution to block (3,3)
  mat.matrix(1, 1).un_complete();
  Core::LinAlg::matrix_add(fii, false, 1., mat.matrix(1, 1), 0.0);
  std::shared_ptr<Core::LinAlg::SparseMatrix> eye =
      Core::LinAlg::create_identity_matrix(*fluid_field()->interface()->fsi_cond_map());
  Core::LinAlg::matrix_add(*eye, false, 1., mat.matrix(1, 1), 1.0);

  // ---------Addressing contribution to block (4,2)
  std::shared_ptr<Core::LinAlg::SparseMatrix> laig =
      std::make_shared<Core::LinAlg::SparseMatrix>(aii.row_map(), 81, false);
  (*aigtransform_)(a->full_row_map(), a->full_col_map(), aig, 1.,
      Coupling::Adapter::CouplingSlaveConverter(interface_fluid_ale_coupling()), *laig);

  laig->complete(f->matrix(1, 1).domain_map(), aii.range_map(), {.enforce_complete = true});
  std::shared_ptr<Core::LinAlg::SparseMatrix> llaig =
      matrix_multiply(*laig, false, *mortarp, false, false, false, true);
  laig = std::make_shared<Core::LinAlg::SparseMatrix>(llaig->row_map(), 81, false);

  Core::LinAlg::matrix_add(*llaig, false, 1.0, *laig, 0.0);
  laig->complete(s->domain_map(), llaig->range_map());

  if (stcalgo != Inpar::Solid::stc_inactive)
  {
    laig = Core::LinAlg::matrix_multiply(*laig, false, *stcmat, false, false, false, true);
  }

  mat.assign(2, 0, Core::LinAlg::DataAccess::Share, *laig);

  // ---------Addressing contribution to block (4,4)
  mat.assign(2, 2, Core::LinAlg::DataAccess::Share, aii);

  /*--------------------------------------------------------------------------*/
  // add optional fluid linearization with respect to mesh motion block
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mmm = fluid_field()->shape_derivatives();
  if (mmm != nullptr)
  {
    // extract submatrices
    Core::LinAlg::SparseMatrix& fmii = mmm->matrix(0, 0);
    Core::LinAlg::SparseMatrix& fmgi = mmm->matrix(1, 0);

    // reuse transform objects to add shape derivative matrices to structural blocks

    // ---------Addressing contribution to block (2,2)
    std::shared_ptr<Core::LinAlg::SparseMatrix> fmgg =
        matrix_multiply(mmm->matrix(1, 1), false, *mortarp, false, false, false, true);
    fmgg = matrix_multiply(*mortarp, true, *fmgg, false, false, false, true);

    Core::LinAlg::SparseMatrix lfmgg(fmgg->row_map(), 81, false);
    Core::LinAlg::matrix_add(*fmgg, false, 1.0, lfmgg, 0.0);
    lfmgg.complete(s->domain_map(), fmgg->range_map());

    Core::LinAlg::matrix_add(lfmgg, false, scale * (1. - stiparam) / (1. - ftiparam), *s, 1.0);

    // ---------Addressing contribution to block (3,2)
    std::shared_ptr<Core::LinAlg::SparseMatrix> fmig =
        matrix_multiply(mmm->matrix(0, 1), false, *mortarp, false, false, false, true);
    std::shared_ptr<Core::LinAlg::SparseMatrix> lfmig =
        std::make_shared<Core::LinAlg::SparseMatrix>(fmig->row_map(), 81, false);

    Core::LinAlg::matrix_add(*fmig, false, 1.0, *lfmig, 0.0);
    lfmig->complete(s->domain_map(), fmig->range_map());

    if (stcalgo != Inpar::Solid::stc_inactive)
    {
      lfmig = Core::LinAlg::matrix_multiply(*lfmig, false, *stcmat, false, false, false, true);
    }

    Core::LinAlg::matrix_add(*lfmig, false, 1.0, mat.matrix(1, 0), 1.0);

    // We cannot copy the pressure value. It is not used anyway. So no exact
    // match here.
    (*fmiitransform_)(mmm->full_row_map(), mmm->full_col_map(), fmii, 1.,
        Coupling::Adapter::CouplingMasterConverter(coupfa), mat.matrix(1, 2), false);

    std::shared_ptr<Core::LinAlg::SparseMatrix> lfmgi =
        std::make_shared<Core::LinAlg::SparseMatrix>(fmgi.row_map(), 81, false);
    (*fmiitransform_)(mmm->full_row_map(), mmm->full_col_map(), fmgi, 1.0,
        Coupling::Adapter::CouplingMasterConverter(coupfa), *lfmgi, false);

    // ---------Addressing contribution to block (2,4)
    lfmgi->complete(aii.domain_map(), mortarp->range_map(), {.enforce_complete = true});
    std::shared_ptr<Core::LinAlg::SparseMatrix> llfmgi =
        matrix_multiply(*mortarp, true, *lfmgi, false, false, false, true);
    lfmgi = std::make_shared<Core::LinAlg::SparseMatrix>(s->row_map(), 81, false);

    Core::LinAlg::matrix_add(*llfmgi, false, scale, *lfmgi, 0.0);
    lfmgi->complete(aii.domain_map(), s->range_map());

    if (stcalgo == Inpar::Solid::stc_currsym)
      lfmgi = Core::LinAlg::matrix_multiply(*stcmat, true, *lfmgi, false, true, true, false);
    lfmgi->scale((1. - stiparam) / (1. - ftiparam));
    mat.assign(0, 2, Core::LinAlg::DataAccess::Share, *lfmgi);
  }

  if (stcalgo != Inpar::Solid::stc_inactive)
  {
    s = Core::LinAlg::matrix_multiply(*s, false, *stcmat, false, true, true, true);

    if (stcalgo == Inpar::Solid::stc_currsym)
      s = Core::LinAlg::matrix_multiply(*stcmat, true, *s, false, true, true, false);
  }

  // finally assign structure matrix to block (0,0)
  mat.assign(0, 0, Core::LinAlg::DataAccess::Share, *s);

  // done. make sure all blocks are filled.
  mat.complete();

  // Finally, take care of Dirichlet boundary conditions
  mat.apply_dirichlet(*(dbcmaps_->cond_map()), true);
  //
  // ---------------------------------------------------------------------------
  // END building the global system matrix
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // NOX related stuff needed for recovery of Lagrange multiplier
  // ---------------------------------------------------------------------------
  // store parts of fluid matrix to know them in the next iteration as previous
  // iteration matrices
  fgiprev_ = fgicur_;
  fggprev_ = fggcur_;
  fgicur_ = std::make_shared<Core::LinAlg::SparseMatrix>(f->matrix(1, 0));
  fggcur_ = std::make_shared<Core::LinAlg::SparseMatrix>(f->matrix(1, 1));

  // store parts of fluid shape derivative matrix to know them in the next
  // iteration as previous iteration matrices
  fmgiprev_ = fmgicur_;
  fmggprev_ = fmggcur_;
  if (mmm != nullptr)
  {
    fmgicur_ = std::make_shared<Core::LinAlg::SparseMatrix>(mmm->matrix(1, 0));
    fmggcur_ = std::make_shared<Core::LinAlg::SparseMatrix>(mmm->matrix(1, 1));
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplit::scale_system(
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
void FSI::MortarMonolithicFluidSplit::unscale_solution(Core::LinAlg::BlockSparseMatrixBase& mat,
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

    // get info about STC feature and unscale solution if necessary
    Inpar::Solid::StcScale stcalgo = structure_field()->get_stc_algo();
    if (stcalgo != Inpar::Solid::stc_inactive)
    {
      structure_field()->get_stc_mat()->multiply(false, *sy, *sy);
    }

    extractor().insert_vector(*sy, 0, x);
    extractor().insert_vector(*ay, 2, x);

    std::shared_ptr<Core::LinAlg::Vector<double>> sx = extractor().extract_vector(b, 0);
    std::shared_ptr<Core::LinAlg::Vector<double>> ax = extractor().extract_vector(b, 2);

    sx->reciprocal_multiply(1.0, *srowsum_, *sx, 0.0);
    ax->reciprocal_multiply(1.0, *arowsum_, *ax, 0.0);

    // get info about STC feature
    if (stcalgo != Inpar::Solid::stc_inactive)
    {
      structure_field()->get_stc_mat()->multiply(false, *sx, *sx);
    }

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

  if (structure_field()->get_stc_algo() != Inpar::Solid::stc_inactive)
    structure_field()->system_matrix()->reset();
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::StatusTest::Combo> FSI::MortarMonolithicFluidSplit::create_status_test(
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
  interface.push_back(structure_field()->interface()->fsi_cond_map());
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
void FSI::MortarMonolithicFluidSplit::extract_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& fx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& ax)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MonolithicOverlap::extract_field_vectors");

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (ddgpred_ == nullptr) FOUR_C_THROW("Vector 'ddgpred_' has not been initialized properly.");
#endif

  // get the Mortar projection matrix P = D^{-1} * M
  const std::shared_ptr<const Core::LinAlg::SparseMatrix> mortarp = coupsfm_->get_mortar_matrix_p();

  // ---------------------------------------------------------------------------
  // process structure unknowns
  // ---------------------------------------------------------------------------
  // extract structure solution increment from NOX increment
  sx = extractor().extract_vector(*x, 0);

  // ---------------------------------------------------------------------------
  // process ale unknowns
  // ---------------------------------------------------------------------------
  // extract inner ALE solution increment from NOX increment
  std::shared_ptr<const Core::LinAlg::Vector<double>> aox = extractor().extract_vector(*x, 2);

  // convert structure solution increment to ALE solution increment at the interface
  std::shared_ptr<Core::LinAlg::Vector<double>> scx =
      structure_field()->interface()->extract_fsi_cond_vector(*sx);
  scx->update(1.0, *ddgpred_, 1.0);
  std::shared_ptr<Core::LinAlg::Vector<double>> acx =
      std::make_shared<Core::LinAlg::Vector<double>>(*fluid_field()->interface()->fsi_cond_map());
  mortarp->multiply(false, *scx, *acx);
  acx = fluid_to_ale_interface(acx);

  // put inner and interface ALE solution increments together
  std::shared_ptr<Core::LinAlg::Vector<double>> a =
      ale_field()->interface()->insert_other_vector(*aox);
  ale_field()->interface()->insert_fsi_cond_vector(*acx, *a);
  ax = a;

  // ---------------------------------------------------------------------------
  // process fluid unknowns
  // ---------------------------------------------------------------------------
  // extract inner fluid solution increment from NOX increment
  std::shared_ptr<const Core::LinAlg::Vector<double>> fox = extractor().extract_vector(*x, 1);
  fox = fluid_field()->interface()->extract_other_vector(*fox);

  // convert ALE solution increment to fluid solution increment at the interface
  std::shared_ptr<Core::LinAlg::Vector<double>> fcx = ale_to_fluid_interface(acx);
  fluid_field()->displacement_to_velocity(fcx);

  // put inner and interface fluid solution increments together
  std::shared_ptr<Core::LinAlg::Vector<double>> f =
      fluid_field()->interface()->insert_other_vector(*fox);
  fluid_field()->interface()->insert_fsi_cond_vector(*fcx, *f);
  fx = f;

  // ---------------------------------------------------------------------------

  // Store field vectors to know them later on as previous quantities
  // interface structure displacement increment
  if (disgprev_ != nullptr)
    ddginc_->update(1.0, *scx, -1.0, *disgprev_, 0.0);  // compute current iteration increment
  else
    ddginc_ = std::make_shared<Core::LinAlg::Vector<double>>(*scx);  // first iteration increment

  disgprev_ = scx;  // store current step increment
  // ------------------------------------

  // inner ale displacement increment
  if (aleiprev_ != nullptr)
    ddialeinc_->update(1.0, *aox, -1.0, *aleiprev_, 0.0);  // compute current iteration increment
  else
    ddialeinc_ = std::make_shared<Core::LinAlg::Vector<double>>(*aox);  // first iteration increment

  aleiprev_ = aox;  // store current step increment
  // ------------------------------------

  // inner fluid solution increment
  if (veliprev_ != nullptr)  // compute current iteration increment
    duiinc_->update(1.0, *fox, -1.0, *veliprev_, 0.0);
  else
    // first iteration increment
    duiinc_ = std::make_shared<Core::LinAlg::Vector<double>>(*fox);
  // store current step increment
  veliprev_ = fox;
  // ------------------------------------
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplit::update()
{
  // update lagrange multiplier field
  lambdaold_->update(1.0, *lambda_, 0.0);

  // update history variables for sliding ale
  if (aleproj_ != Inpar::FSI::ALEprojection_none)
  {
    iprojdisp_ = std::make_shared<Core::LinAlg::Vector<double>>(*coupsfm_->slave_dof_map(), true);
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
void FSI::MortarMonolithicFluidSplit::output()
{
  structure_field()->output();
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

  // output Lagrange multiplier
  output_lambda();

  ale_field()->output();

  if (use_old_structure_)
  {
    if (structure_field()->get_constraint_manager()->have_monitor())
    {
      structure_field()->get_constraint_manager()->compute_monitor_values(
          *structure_field()->dispnp());
      if (Core::Communication::my_mpi_rank(comm_) == 0)
        structure_field()->get_constraint_manager()->print_monitor_values();
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplit::output_lambda()
{
  /* 'lambda_' is only defined on the interface. So, insert 'lambda_' into
   * 'lambdafull' that is defined on the entire fluid field. Then, write
   * output or restart data.
   */
  std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
      fluid_field()->interface()->insert_fsi_cond_vector(*lambda_);
  const int uprestart = timeparams_.get<int>("RESTARTEVERY");
  const int upres = timeparams_.get<int>("RESULTSEVERY");
  if ((uprestart != 0 and fluid_field()->step() % uprestart == 0) or
      (upres != 0 and fluid_field()->step() % upres == 0))
    fluid_field()->disc_writer()->write_vector("fsilambda", lambdafull);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplit::read_restart(int step)
{
  structure_field()->read_restart(step);
  fluid_field()->read_restart(step);

  auto input_control_file = Global::Problem::instance()->input_control_file();

  // read Lagrange multiplier
  const bool restartfrompartfsi = timeparams_.get<bool>("RESTART_FROM_PART_FSI");
  if (not restartfrompartfsi)  // standard restart
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
        std::make_shared<Core::LinAlg::Vector<double>>(*fluid_field()->dof_row_map(), true);
    Core::IO::DiscretizationReader reader =
        Core::IO::DiscretizationReader(*fluid_field()->discretization(), input_control_file, step);
    reader.read_vector(lambdafull, "fsilambda");
    lambdaold_ = fluid_field()->interface()->extract_fsi_cond_vector(*lambdafull);
    // Note: the above is normally enough. However, we can use the restart in order to periodically
    // repeat the fsi simulation (see AC-FS3I)
    lambda_ = fluid_field()->interface()->extract_fsi_cond_vector(*lambdafull);
  }

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
void FSI::MortarMonolithicFluidSplit::prepare_time_step()
{
  precondreusecount_ = 0;

  increment_time_and_step();
  print_header();

  prepare_time_step_preconditioner();

  if (structure_field()->get_stc_algo() != Inpar::Solid::stc_inactive)
    structure_field()->system_matrix()->reset();

  prepare_time_step_fields();

  // Note: it's important to first prepare the single fields and than the fsi problem
  prepare_time_step_fsi();
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplit::recover_lagrange_multiplier()
{
  //  // store previous Lagrange multiplier for calculation of interface energy
  //  lambdaold_->Update(1.0, *lambda_, 0.0);

  // get time integration parameter of fluid time integrator
  // to enable consistent time integration among the fields
  const double ftiparam = fluid_field()->tim_int_param();

  // some scaling factors for fluid
  const double timescale = fluid_field()->time_scaling();
  const double scale = fluid_field()->residual_scaling();

  // get the Mortar projection matrix P = D^{-1} * M
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortarp = coupsfm_->get_mortar_matrix_p();

  // get the inverted Mortar matrix D^{-1}
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortardinv = coupsfm_->get_mortar_matrix_dinv();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (mortarp == nullptr) FOUR_C_THROW("Expected Teuchos::rcp to mortar matrix P.");
  if (mortardinv == nullptr) FOUR_C_THROW("Expected Teuchos::rcp to mortar matrix D^{{-1}}.");
#endif

  // get fluid shape derivative matrix
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> mmm =
      fluid_field()->shape_derivatives();

  // some often re-used vectors
  std::shared_ptr<Core::LinAlg::Vector<double>> tmpvec =
      nullptr;  // stores intermediate result of terms (3)-(8)
  std::shared_ptr<Core::LinAlg::Vector<double>> auxvec = nullptr;     // just for convenience
  std::shared_ptr<Core::LinAlg::Vector<double>> auxauxvec = nullptr;  // just for convenience

  /* Recovery of Lagrange multiplier \lambda_^{n+1} is done by the following
   * condensation expression:
   *
   * lambda_^{n+1} =
   *
   * (1)  - ftiparam / (1.-ftiparam) * lambda^{n}
   *
   * (2)  - 1. / (1.-ftiparam) * D^{-T} * tmpvec
   *
   * with tmpvec =
   *
   * (3)    r_{\Gamma}^{F,n+1}
   *
   * (4)  + 1 / tau * F_{\Gamma\Gamma} * P * \Delta d_{\Gamma}^{S,n+1}
   *
   * (5)  + F_{\Gamma\Gamma}^{G} * P * \Delta d_{\Gamma}^{S,n+1}
   *
   * (6)  + F_{\Gamma I} * \Delta u_{I}^{F,n+1}
   *
   * (7)  + F_{\Gamma I}^{G} * \Delta d_{I}^{G,n+1}
   *
   * (8)  - dt / tau * F_{\Gamma\Gamma} * u_{\Gamma}^n]
   *
   * Remark on term (8):
   * Term (8) has to be considered only in the first Newton iteration.
   * Hence, it will usually not be computed since in general we need more
   * than one nonlinear iteration until convergence.
   *
   * Remarks on all terms:
   * +  Division by (1.0 - ftiparam) will be done in the end
   *    since this is common to all terms
   * +  tau: time scaling factor for interface time integration (tau =
   * 1/fluid_field()->TimeScaling())
   * +  neglecting terms (4)-(8) should not alter the results significantly
   *    since at the end of the time step the solution increments tend to zero.
   *
   * Terms arising from field specific predictors have to be considered only in
   * the first Newton iteration. Since we usually need more than on iteration,
   * these terms are not implemented, yet.
   *
   *                                                 Matthias Mayr (10/2012)
   */

  // ---------Addressing term (1)
  lambda_->update(ftiparam, *lambdaold_, 0.0);
  // ---------End of term (1)

  // ---------Addressing term (3)
  std::shared_ptr<Core::LinAlg::Vector<double>> fluidresidual =
      fluid_field()->interface()->extract_fsi_cond_vector(*fluid_field()->rhs());
  fluidresidual->scale(-1.0);
  tmpvec = std::make_shared<Core::LinAlg::Vector<double>>(*fluidresidual);
  // ---------End of term (3)

  // ---------Addressing term (4)
  auxvec = std::make_shared<Core::LinAlg::Vector<double>>(mortarp->range_map(), true);
  mortarp->multiply(false, *ddginc_, *auxvec);
  auxauxvec = std::make_shared<Core::LinAlg::Vector<double>>(fggprev_->range_map(), true);
  fggprev_->multiply(false, *auxvec, *auxauxvec);
  tmpvec->update(timescale, *auxauxvec, 1.0);
  // ---------End of term (4)

  // ---------Addressing term (5)
  if (fmggprev_ != nullptr)
  {
    auxvec = std::make_shared<Core::LinAlg::Vector<double>>(mortarp->range_map(), true);
    mortarp->multiply(false, *ddginc_, *auxvec);
    fmggprev_->multiply(false, *auxvec, *auxauxvec);
    tmpvec->update(1.0, *auxauxvec, 1.0);
  }
  // ---------End of term (5)

  // ---------Addressing term (6)
  auxvec = std::make_shared<Core::LinAlg::Vector<double>>(fgiprev_->range_map(), true);
  fgiprev_->multiply(false, *duiinc_, *auxvec);
  tmpvec->update(1.0, *auxvec, 1.0);
  // ---------End of term (6)

  // ---------Addressing term (7)
  if (fmgiprev_ != nullptr)
  {
    /* For matrix-vector-product, the DomainMap() of the matrix and the Map()
     * of the vector have to match. DomaintMap() contains inner velocity DOFs
     * and all pressure DOFs. The inner ale displacement increment is converted
     * to the fluid map using AleToFluid(). This results in a map that contains
     * all velocity but no pressure DOFs.
     *
     * We have to circumvent some trouble with Core::LinAlg::Maps since we cannot
     * split an Core::LinAlg::Map into inner and interface DOFs.
     *
     * We create a map extractor 'velothermap' in order to extract the inner
     * velocity DOFs after calling AleToFluid(). Afterwards, a second map
     * extractor 'velotherpressuremapext' is used to append pressure DOFs filled
     * with zeros.
     *
     * Finally, maps match and matrix-vector-multiplication can be done.
     */

    // extract inner velocity DOFs after calling AleToFluid()
    std::shared_ptr<Core::LinAlg::Map> velothermap = Core::LinAlg::split_map(
        *fluid_field()->velocity_row_map(), *interface_fluid_ale_coupling().master_dof_map());
    Core::LinAlg::MapExtractor velothermapext =
        Core::LinAlg::MapExtractor(*fluid_field()->velocity_row_map(), velothermap, false);
    auxvec = std::make_shared<Core::LinAlg::Vector<double>>(*velothermap, true);
    velothermapext.extract_other_vector(
        *ale_to_fluid(ale_field()->interface()->insert_other_vector(*ddialeinc_)), *auxvec);

    // add pressure DOFs
    Core::LinAlg::MapExtractor velotherpressuremapext =
        Core::LinAlg::MapExtractor(fmgiprev_->domain_map(), velothermap);
    auxauxvec = std::make_shared<Core::LinAlg::Vector<double>>(fmgiprev_->domain_map(), true);
    velotherpressuremapext.insert_cond_vector(*auxvec, *auxauxvec);

    // prepare vector to store result of matrix-vector-product
    auxvec = std::make_shared<Core::LinAlg::Vector<double>>(fmgiprev_->range_map(), true);

    // Now, do the actual matrix-vector-product
    fmgiprev_->multiply(false, *auxauxvec, *auxvec);
    tmpvec->update(1.0, *auxvec, 1.0);
  }
  // ---------End of term (7)

  // ---------Addressing term (8)
  if (firstcall_)
  {
    auxvec = std::make_shared<Core::LinAlg::Vector<double>>(fggprev_->range_map(), true);
    fggprev_->multiply(false, *fluid_field()->extract_interface_veln(), *auxvec);
    tmpvec->update(dt() * timescale, *auxvec, 1.0);
  }
  // ---------End of term (8)

  // ---------Addressing term (2)
  auxvec = std::make_shared<Core::LinAlg::Vector<double>>(mortardinv->domain_map(), true);
  mortardinv->multiply(true, *tmpvec, *auxvec);
  lambda_->update(scale, *auxvec, 1.0);  // scale with residual_scaling() to get [N/m^2]
  // ---------End of term (2)

  // Finally, divide by (1.0-ftiparam) which is common to all terms
  lambda_->scale(-1.0 / (1.0 - ftiparam));

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
void FSI::MortarMonolithicFluidSplit::calculate_interface_energy_increment()
{
  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double stiparam = structure_field()->tim_int_param();
  const double ftiparam = fluid_field()->tim_int_param();

  // get the Mortar matrix M
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortarm = coupsfm_->get_mortar_matrix_m();

  // interface traction weighted by time integration factors
  Core::LinAlg::Vector<double> tractionfluid(lambda_->get_map(), true);
  std::shared_ptr<Core::LinAlg::Vector<double>> tractionstructure =
      std::make_shared<Core::LinAlg::Vector<double>>(
          *structure_field()->interface()->fsi_cond_map(), true);
  tractionfluid.update(stiparam - ftiparam, *lambdaold_, ftiparam - stiparam, *lambda_, 0.0);
  mortarm->multiply(true, tractionfluid, *tractionstructure);

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
void FSI::MortarMonolithicFluidSplit::check_kinematic_constraint()
{
  // some scaling factors for fluid
  const double timescale = fluid_field()->time_scaling();

  // get the Mortar matrices D and M
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortard = coupsfm_->get_mortar_matrix_d();
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortarm = coupsfm_->get_mortar_matrix_m();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (mortarm == nullptr) FOUR_C_THROW("Expected Teuchos::rcp to mortar matrix M.");
  if (mortard == nullptr) FOUR_C_THROW("Expected Teuchos::rcp to mortar matrix D.");
#endif

  // get interface displacements and velocities
  const std::shared_ptr<Core::LinAlg::Vector<double>> disnp =
      structure_field()->extract_interface_dispnp();
  const std::shared_ptr<Core::LinAlg::Vector<double>> disn =
      structure_field()->extract_interface_dispn();
  const std::shared_ptr<Core::LinAlg::Vector<double>> velnp =
      fluid_field()->extract_interface_velnp();
  const std::shared_ptr<Core::LinAlg::Vector<double>> veln =
      fluid_field()->extract_interface_veln();

  // prepare vectors for projected interface quantities
  Core::LinAlg::Vector<double> disnpproj(mortarm->range_map(), true);
  Core::LinAlg::Vector<double> disnproj(mortarm->range_map(), true);
  Core::LinAlg::Vector<double> velnpproj(mortard->range_map(), true);
  Core::LinAlg::Vector<double> velnproj(mortard->range_map(), true);

  // projection of interface displacements
  mortarm->multiply(false, *disnp, disnpproj);
  mortarm->multiply(false, *disn, disnproj);

  // projection of interface velocities
  mortard->multiply(false, *velnp, velnpproj);
  mortard->multiply(false, *veln, velnproj);

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
void FSI::MortarMonolithicFluidSplit::check_dynamic_equilibrium()
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

  // calculate tractions on master and slave side
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
  violationl2 /= sqrt(fluid_field()->interface()->fsi_cond_map()->num_global_elements());

  // output to screen
  std::ios_base::fmtflags flags = utils()->out().flags();

  utils()->out() << std::scientific << "\nViolation of dynamic interface equilibrium:\n"
                 << "L_2-norm: " << violationl2 << "        L_inf-norm: " << violationinf << "\n";
  utils()->out().flags(flags);

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplit::combine_field_vectors(Core::LinAlg::Vector<double>& v,
    std::shared_ptr<const Core::LinAlg::Vector<double>> sv,
    std::shared_ptr<const Core::LinAlg::Vector<double>> fv,
    std::shared_ptr<const Core::LinAlg::Vector<double>> av,
    const bool slave_vectors_contain_interface_dofs)
{
  if (slave_vectors_contain_interface_dofs)
  {
    // extract inner DOFs from slave vectors
    std::shared_ptr<Core::LinAlg::Vector<double>> fov =
        fluid_field()->interface()->extract_other_vector(*fv);
    fov = fluid_field()->interface()->insert_other_vector(*fov);
    std::shared_ptr<Core::LinAlg::Vector<double>> aov =
        ale_field()->interface()->extract_other_vector(*av);

    // put them together
    FSI::Monolithic::combine_field_vectors(v, *sv, *fov, *aov);
  }
  else
    FSI::Monolithic::combine_field_vectors(v, *sv, *fv, *av);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
double FSI::MortarMonolithicFluidSplit::select_dt_error_based() const
{
  // get time step size suggestions
  const double dtstr = get_ada_str_dt();           // based on all structure DOFs
  const double dtstrfsi = get_ada_str_fsi_dt();    // based on structure FSI DOFs
  const double dtflinner = get_ada_fl_inner_dt();  // based on inner fluid DOFs

  double dt = MortarMonolithicFluidSplit::dt();

  // select time step size based on error estimation
  if (is_ada_structure() and is_ada_fluid())
    dt = std::min(std::min(dtstr, dtstrfsi), dtflinner);
  else if (is_ada_structure() and (not is_ada_fluid()))
    dt = std::min(dtstr, dtstrfsi);
  else if ((not is_ada_structure()) and is_ada_fluid())
    dt = dtflinner;
  else
  {
    // no change in time step size based on structure or fluid field error estimation
  }

  return dt;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
bool FSI::MortarMonolithicFluidSplit::set_accepted() const
{
  // get error norms
  const double strnorm = get_ada_strnorm();            // based on all structure DOFs
  const double strfsinorm = get_ada_str_fs_inorm();    // based on structure FSI DOFs
  const double flinnernorm = get_ada_fl_inner_norm();  // based on inner fluid DOFs

  bool accepted = std::max(strnorm, strfsinorm) < errtolstr_ and flinnernorm < errtolfl_;

  // in case error estimation in the fluid field is turned off:
  if (not is_ada_fluid()) accepted = std::max(strnorm, strfsinorm) < errtolstr_;

  // in case error estimation in the structure field is turned off:
  if (not is_ada_structure()) accepted = flinnernorm < errtolfl_;

  // no error based time adaptivity
  if ((not is_ada_structure()) and (not is_ada_fluid())) accepted = true;

  return accepted;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplit::create_system_matrix()
{
  FSI::BlockMonolithic::create_system_matrix(systemmatrix_, false);
}

FOUR_C_NAMESPACE_CLOSE
