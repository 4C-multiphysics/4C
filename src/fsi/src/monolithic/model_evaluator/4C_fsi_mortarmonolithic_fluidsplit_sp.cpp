// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_mortarmonolithic_fluidsplit_sp.hpp"

#include "4C_adapter_ale_fsi.hpp"
#include "4C_adapter_fld_fluid_fsi.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_coupling_adapter_mortar.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fsi_statustest.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_io.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mortar_utils.hpp"
#include "4C_structure_aux.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
FSI::MortarMonolithicFluidSplitSaddlePoint::MortarMonolithicFluidSplitSaddlePoint(
    MPI_Comm comm, const Teuchos::ParameterList& timeparams)
    : BlockMonolithic(comm, timeparams), comm_(comm)
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
    //      std::cout << "Slave interface nodes with Dirichlet boundary condition "
    //                "(input file numbering):" << std::endl;
    //      for (int i=0; i < (int)fluid_field()->discretization()->NumMyRowNodes(); i++)
    //      {
    //        // get all nodes and add them
    //        int gid = fluid_field()->discretization()->NodeRowMap()->GID(i);

    //        // do only nodes that I have in my discretization
    //        if (!fluid_field()->discretization()->NodeColMap()->MyGID(gid)) continue;
    //        Core::Nodes::Node* node = fluid_field()->discretization()->gNode(gid);
    //        if (!node) FOUR_C_THROW("Cannot find node with gid %",gid);

    //        std::vector<int> nodedofs = fluid_field()->discretization()->Dof(node);

    //        for (int j=0; j < (int)nodedofs.size(); j++)
    //        {
    //          for (int k=0; k < (int)intersectionmap->NumGlobalElements(); k++)
    //          {
    //            if (nodedofs[j] == intersectionmap->GID(k))
    //            {
    //              std::cout << gid+1 << std::endl;
    //              k = (int)intersectionmap->GID(k);
    //              j = (int)nodedofs.size();
    //            }
    //          }
    //        }
    //      }

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

  coupling_solid_fluid_mortar_ = std::make_shared<Coupling::Adapter::CouplingMortar>(
      Global::Problem::instance()->n_dim(), Global::Problem::instance()->mortar_coupling_params(),
      Global::Problem::instance()->contact_dynamic_params(),
      Global::Problem::instance()->spatial_approximation_type());

  ale_inner_interf_transform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  fluid_mesh_inner_inner_transform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();

  create_lagrange_multiplier_dof_row_map();
  set_lag_mult();

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (coupling_solid_fluid_mortar_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'coupling_solid_fluid_mortar_' failed.");
  }
  if (ale_inner_interf_transform_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'ale_inner_interf_transform_' failed.");
  }
  if (fluid_mesh_inner_inner_transform_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'fluid_mesh_inner_inner_transform_' failed.");
  }
  if (lag_mult_ == nullptr)
  {
    FOUR_C_THROW("Allocation of 'lag_mult_' failed.");
  }
#endif
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::set_lag_mult()
{
  lag_mult_ = std::make_shared<Core::LinAlg::Vector<double>>(*lag_mult_dof_map_, true);
  lag_mult_old_ = std::make_shared<Core::LinAlg::Vector<double>>(*lag_mult_dof_map_, true);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::setup_system()
{
  if (notsetup_)
  {
    const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();

    set_default_parameters(fsidyn, nox_parameter_list());

    // we use non-matching meshes at the interface
    // mortar with: structure = master, fluid = slave

    const int ndim = Global::Problem::instance()->n_dim();

    // get coupling objects
    Coupling::Adapter::Coupling& interface_coup_fluid_ale = interface_fluid_ale_coupling();

    /* structure to fluid
     * coupling condition at the fsi interface:
     * displacements (=number spacial dimensions) are coupled
     * e.g.: 3D: coupleddof = [1, 1, 1]
     */
    std::vector<int> coupleddof(ndim, 1);

    coupling_solid_fluid_mortar_->setup(structure_field()->discretization(),
        fluid_field()->discretization(), ale_field()->write_access_discretization(), coupleddof,
        "FSICoupling", comm_, Global::Problem::instance()->function_manager(),
        Global::Problem::instance()->binning_strategy_params(),
        Global::Problem::instance()->discretization_map(),
        Global::Problem::instance()->output_control_file(),
        Global::Problem::instance()->spatial_approximation_type(), true);

    // fluid to ale at the interface
    interface_coup_fluid_ale.setup_condition_coupling(*fluid_field()->discretization(),
        fluid_field()->interface()->fsi_cond_map(), *ale_field()->discretization(),
        ale_field()->interface()->fsi_cond_map(), "FSICoupling", ndim);

    Coupling::Adapter::Coupling& coup_fluid_ale = fluid_ale_coupling();

    // the fluid-ale coupling always matches
    const Core::LinAlg::Map* fluidnodemap = fluid_field()->discretization()->node_row_map();
    const Core::LinAlg::Map* alenodemap = ale_field()->discretization()->node_row_map();

    coup_fluid_ale.setup_coupling(*fluid_field()->discretization(), *ale_field()->discretization(),
        *fluidnodemap, *alenodemap, ndim);

    fluid_field()->set_mesh_map(coup_fluid_ale.master_dof_map());

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
    // -------------------------------------------------------------------------#

    create_system_matrix();

    notsetup_ = false;
  }

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
void FSI::MortarMonolithicFluidSplitSaddlePoint::create_lagrange_multiplier_dof_row_map()
{
  const int num_glob_elem_fluid_interface =
      fluid_field()->interface()->fsi_cond_map()->num_global_elements();
  const int num_loc_elem_fluid_interface =
      fluid_field()->interface()->fsi_cond_map()->num_my_elements();
  const int max_gid_ale = ale_field()->dof_row_map()->max_all_gid();
  lag_mult_dof_map_ = std::make_shared<Core::LinAlg::Map>(
      num_glob_elem_fluid_interface, num_loc_elem_fluid_interface, max_gid_ale + 1, comm_);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::create_combined_dof_row_map()
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> vecSpaces;
  vecSpaces.push_back(structure_field()->dof_row_map());
  vecSpaces.push_back(fluid_field()->dof_row_map());
  vecSpaces.push_back(ale_field()->interface()->other_map());
  vecSpaces.push_back(lag_mult_dof_map_);

  if (vecSpaces[1]->num_global_elements() == 0)
    FOUR_C_THROW("No inner fluid equations. Splitting not possible.");

  set_dof_row_maps(vecSpaces);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::create_system_matrix()
{
  FSI::BlockMonolithic::create_system_matrix(systemmatrix_, false);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::StatusTest::Combo>
FSI::MortarMonolithicFluidSplitSaddlePoint::create_status_test(
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

  // ---------------------------------------------------------------------------
  // setup tests for Lagrange multiplier field
  // ---------------------------------------------------------------------------
  // create ::NOX::StatusTest::Combo for Lagrange multiplier field
  Teuchos::RCP<::NOX::StatusTest::Combo> lag_mult_combo =
      Teuchos::make_rcp<::NOX::StatusTest::Combo>(::NOX::StatusTest::Combo::AND);

  // create Norm-objects for each norm that has to be tested
  Teuchos::RCP<NOX::FSI::PartialNormF> lag_mult_L2 = Teuchos::make_rcp<NOX::FSI::PartialNormF>(
      "LAGMULT residual", extractor(), 3, nlParams.get<double>("Tol fsi res L2"),
      ::NOX::Abstract::Vector::TwoNorm, NOX::FSI::PartialNormF::Scaled);
  Teuchos::RCP<NOX::FSI::PartialNormF> lag_mult_inf = Teuchos::make_rcp<NOX::FSI::PartialNormF>(
      "LAGMULT residual", extractor(), 3, nlParams.get<double>("Tol fsi res Inf"),
      ::NOX::Abstract::Vector::MaxNorm, NOX::FSI::PartialNormF::Unscaled);
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> lag_mult_update_L2 =
      Teuchos::make_rcp<NOX::FSI::PartialNormUpdate>("LAGMULT update", extractor(), 3,
          nlParams.get<double>("Tol fsi inc L2"), ::NOX::Abstract::Vector::TwoNorm,
          NOX::FSI::PartialNormUpdate::Scaled);
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> lag_mult_update_inf =
      Teuchos::make_rcp<NOX::FSI::PartialNormUpdate>("LAGMULT update", extractor(), 3,
          nlParams.get<double>("Tol fsi inc Inf"), ::NOX::Abstract::Vector::MaxNorm,
          NOX::FSI::PartialNormUpdate::Unscaled);

  // tests needed to adapt relative tolerance of the linear solver
  add_status_test(lag_mult_L2);

  // add norm-tests to structural displacement ::NOX::StatusTest::Combo
  lag_mult_combo->addStatusTest(lag_mult_L2);
  lag_mult_combo->addStatusTest(lag_mult_inf);
  lag_mult_combo->addStatusTest(lag_mult_update_L2);
  lag_mult_combo->addStatusTest(lag_mult_update_inf);

  // add structural displacement test combo to top-level test combo
  converged->addStatusTest(lag_mult_combo);
  // ---------- end of Lagrange multiplier field tests

  // Finally, return the test combo
  return combo;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::setup_dbc_map_extractor()
{
  /* Dirichlet maps for structure and fluid do not intersect with interface map.
   * ALE Dirichlet map might intersect with interface map, but ALE interface
   * DOFs are not part of the final system of equations. Hence, we just need the
   * intersection of inner ALE DOFs with Dirichlet ALE DOFs.
   */
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> aleintersectionmaps;
  aleintersectionmaps.push_back(ale_field()->get_dbc_map_extractor()->cond_map());
  aleintersectionmaps.push_back(ale_field()->interface()->other_map());
  std::shared_ptr<const Core::LinAlg::Map> aleintersectionmap =
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
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>
FSI::MortarMonolithicFluidSplitSaddlePoint::system_matrix() const
{
  return systemmatrix_;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::initial_guess(
    std::shared_ptr<Core::LinAlg::Vector<double>> initial_guess)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MortarMonolithicFluidSplitSaddlePoint::initial_guess");

  std::shared_ptr<const Core::LinAlg::Vector<double>> lag_mult_initial_guess =
      std::make_shared<Core::LinAlg::Vector<double>>(*lag_mult_dof_map_, true);

  combine_field_vectors(*initial_guess, structure_field()->initial_guess(),
      fluid_field()->initial_guess(), ale_field()->initial_guess(), lag_mult_initial_guess, true);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::combine_field_vectors(
    Core::LinAlg::Vector<double>& f,
    std::shared_ptr<const Core::LinAlg::Vector<double>> solid_vector,
    std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_vector,
    std::shared_ptr<const Core::LinAlg::Vector<double>> ale_vector,
    std::shared_ptr<const Core::LinAlg::Vector<double>> lag_mult_vector, bool fullvectors)
{
  if (fullvectors)
  {
    // extract inner DOFs from slave vectors
    std::shared_ptr<const Core::LinAlg::Vector<double>> ale_other_vector =
        ale_field()->interface()->extract_other_vector(*ale_vector);

    extractor().add_vector(*solid_vector, 0, f);
    extractor().add_vector(*fluid_vector, 1, f);
    extractor().add_vector(*ale_other_vector, 2, f);
    extractor().add_vector(*lag_mult_vector, 3, f);
  }
  else
  {
    extractor().add_vector(*solid_vector, 0, f);
    extractor().add_vector(*fluid_vector, 1, f);
    extractor().add_vector(*ale_vector, 2, f);
    extractor().add_vector(*lag_mult_vector, 3, f);
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::setup_rhs_residual(Core::LinAlg::Vector<double>& f)
{
  // get single field residuals
  std::shared_ptr<Core::LinAlg::Vector<double>> solid_single_field_rhs_vector =
      std::make_shared<Core::LinAlg::Vector<double>>(*structure_field()->rhs());
  std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_single_field_rhs_vector =
      std::make_shared<Core::LinAlg::Vector<double>>(*fluid_field()->rhs());
  std::shared_ptr<const Core::LinAlg::Vector<double>> ale_single_field_rhs_vector =
      std::make_shared<Core::LinAlg::Vector<double>>(*ale_field()->rhs());
  std::shared_ptr<Core::LinAlg::Vector<double>> lag_mult_rhs_vector =
      std::make_shared<Core::LinAlg::Vector<double>>(*lag_mult_dof_map_, true);

  // NOX treats rhs different in new solid time integration, hence sign inversion is necessary here
  if (!use_old_structure_) solid_single_field_rhs_vector->scale(-1.0);

  // put the single field residuals together
  combine_field_vectors(f, solid_single_field_rhs_vector, fluid_single_field_rhs_vector,
      ale_single_field_rhs_vector, lag_mult_rhs_vector, true);

  // add additional ale residual to avoid incremental ale errors
  extractor().add_vector(*aleresidual_, 2, f);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::setup_rhs_lambda(Core::LinAlg::Vector<double>& f)
{
  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double solid_time_int_param = structure_field()->tim_int_param();
  const double fluid_time_int_param = fluid_field()->tim_int_param();
  const double fluid_res_scale = fluid_field()->residual_scaling();

  // get the mortar structure to fluid coupling matrix M
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_m =
      coupling_solid_fluid_mortar_->get_mortar_matrix_m();
  std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_m_transf =
      Core::LinAlg::matrix_row_transform_gids(*mortar_m, *lag_mult_dof_map_);

  // get the mortar fluid to structure coupling matrix D
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_d =
      coupling_solid_fluid_mortar_->get_mortar_matrix_d();
  std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_d_transf =
      Core::LinAlg::matrix_row_transform_gids(*mortar_d, *lag_mult_dof_map_);

  Core::LinAlg::Vector<double> lag_mult_step_increment(*lag_mult_dof_map_, true);
  lag_mult_step_increment.update(1.0, *lag_mult_, -1.0, *lag_mult_old_, 0.0);

  // helper variables
  Core::LinAlg::Vector<double> lag_mult_old_rhs_struct_interf(mortar_m_transf->domain_map(), true);
  Core::LinAlg::Vector<double> lag_mult_old_rhs_fluid_interf(mortar_d_transf->domain_map(), true);

  mortar_m_transf->multiply(true, *lag_mult_old_, lag_mult_old_rhs_struct_interf);
  mortar_d_transf->multiply(true, *lag_mult_old_, lag_mult_old_rhs_fluid_interf);

  std::shared_ptr<Core::LinAlg::Vector<double>> lag_mult_old_rhs_struct_interf_full =
      structure_field()->interface()->insert_fsi_cond_vector(lag_mult_old_rhs_struct_interf);
  std::shared_ptr<Core::LinAlg::Vector<double>> lag_mult_old_rhs_fluid_interf_full =
      fluid_field()->interface()->insert_fsi_cond_vector(lag_mult_old_rhs_fluid_interf);

  lag_mult_old_rhs_fluid_interf_full->scale(-1.0 / fluid_res_scale);

  // add lagrange multiplier
  extractor().add_vector(*lag_mult_old_rhs_struct_interf_full, 0, f);
  extractor().add_vector(*lag_mult_old_rhs_fluid_interf_full, 1, f);

  // helper variables
  Core::LinAlg::Vector<double> lag_mult_step_increment_rhs_struct_interf(
      mortar_m_transf->domain_map(), true);
  Core::LinAlg::Vector<double> lag_mult_step_increment_rhs_fluid_interf(
      mortar_d_transf->domain_map(), true);

  mortar_m_transf->multiply(
      true, lag_mult_step_increment, lag_mult_step_increment_rhs_struct_interf);
  mortar_d_transf->multiply(
      true, lag_mult_step_increment, lag_mult_step_increment_rhs_fluid_interf);

  std::shared_ptr<Core::LinAlg::Vector<double>> lag_mult_step_increment_rhs_struct_interf_full =
      structure_field()->interface()->insert_fsi_cond_vector(
          lag_mult_step_increment_rhs_struct_interf);
  std::shared_ptr<Core::LinAlg::Vector<double>> lag_mult_step_increment_rhs_fluid_interf_full =
      fluid_field()->interface()->insert_fsi_cond_vector(lag_mult_step_increment_rhs_fluid_interf);

  lag_mult_step_increment_rhs_struct_interf_full->scale(1.0 * (1. - solid_time_int_param));
  lag_mult_step_increment_rhs_fluid_interf_full->scale(
      -1.0 * (1. - fluid_time_int_param) / fluid_res_scale);

  // add lagrange multiplier
  extractor().add_vector(*lag_mult_step_increment_rhs_struct_interf_full, 0, f);
  extractor().add_vector(*lag_mult_step_increment_rhs_fluid_interf_full, 1, f);
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::setup_rhs_firstiter(
    Core::LinAlg::Vector<double>& f)
{
  // old interface velocity of fluid field
  const std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_veln =
      fluid_field()->extract_interface_veln();

  // get the mortar structure to fluid coupling matrix M
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_m =
      coupling_solid_fluid_mortar_->get_mortar_matrix_m();
  std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_m_transf =
      Core::LinAlg::matrix_row_transform_gids(*mortar_m, *lag_mult_dof_map_);

  // get the mortar fluid to structure coupling matrix D
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_d =
      coupling_solid_fluid_mortar_->get_mortar_matrix_d();
  std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_d_transf =
      Core::LinAlg::matrix_row_transform_gids(*mortar_d, *lag_mult_dof_map_);

  // get fluid shape derivatives matrix
  const std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase> fluid_shape_deriv =
      fluid_field()->shape_derivatives();

  // get ale matrix
  const std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase> aleblock =
      ale_field()->block_system_matrix();

  // extract ale submatrix
  const Core::LinAlg::SparseMatrix& ale_inner_interf = aleblock->matrix(0, 1);

  // right hand side of single set of DOFs
  std::shared_ptr<Core::LinAlg::Vector<double>> rhs = nullptr;

  /* Different contributions/terms to the rhs are separated by the following
   * comment line */
  // ---------- inner fluid DOFs
  /* The following term is added to the inner fluid DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  dt * F^{G}_{I \Gamma} * u^{n}_{\Gamma}
   *
   */
  // ----------addressing term 1
  if (fluid_shape_deriv != nullptr)
  {
    const Core::LinAlg::SparseMatrix& fluid_mesh_inner_interf = fluid_shape_deriv->matrix(0, 1);

    rhs = std::make_shared<Core::LinAlg::Vector<double>>(fluid_mesh_inner_interf.range_map(), true);

    fluid_mesh_inner_interf.multiply(false, *fluid_veln, *rhs);

    rhs->scale(dt());

    rhs = fluid_field()->interface()->insert_other_vector(*rhs);

    extractor().add_vector(*rhs, 1, f);
  }
  // ----------end of term 1
  // ----------end of inner fluid DOFs

  // ---------- interface fluid DOFs
  /* The following term is added to the interface fluid DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  dt * F^{G}_{\Gamma \Gamma} * u^{n}_{\Gamma}
   *
   */
  // ----------addressing term 1
  if (fluid_shape_deriv != nullptr)
  {
    const Core::LinAlg::SparseMatrix& fluid_mesh_interf_interf = fluid_shape_deriv->matrix(1, 1);
    rhs =
        std::make_shared<Core::LinAlg::Vector<double>>(fluid_mesh_interf_interf.range_map(), true);

    fluid_mesh_interf_interf.multiply(false, *fluid_veln, *rhs);
    rhs->scale(dt());
    rhs = fluid_field()->interface()->insert_fsi_cond_vector(*rhs);

    extractor().add_vector(*rhs, 1, f);
  }
  // ----------end of term 1
  // ----------end of interface fluid DOFs

  // ---------- inner ALE DOFs
  /* The following term is added to the inner ALE DOFs of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  dt * A_{I \Gamma} * u^{n}_{\Gamma}
   *
   */
  // ----------addressing term 1
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(ale_inner_interf.range_map(), true);
  ale_inner_interf.multiply(false, *fluid_veln, *rhs);
  rhs->scale(-1. * dt());

  extractor().add_vector(*rhs, 2, f);
  // ----------end of term 1
  // ----------end of inner ALE DOFs

  // ---------- lagrange multiplier
  /* The following term is added to the lagrange multiplier of right hand side:
   *
   * rhs_firstnewtonstep =
   *
   * (1)  + dt * D * u^{n}_{\Gamma}
   *
   * (2)  - M * \Delta d_{\Gamma,p}
   *
   */
  // ----------addressing term 1
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(*lag_mult_dof_map_, true);

  mortar_d_transf->multiply(false, *fluid_veln, *rhs);
  rhs->scale(dt());

  extractor().add_vector(*rhs, 3, f);
  // ----------end of term 1

  // ----------addressing term 2
  rhs = std::make_shared<Core::LinAlg::Vector<double>>(*lag_mult_dof_map_, true);

  mortar_m_transf->multiply(false, *ddgpred_, *rhs);
  rhs->scale(-1.);

  extractor().add_vector(*rhs, 3, f);
  // ----------end of term 2
  // ----------end of lagrange multiplier
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::setup_system_matrix(
    Core::LinAlg::BlockSparseMatrixBase& mat)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MortarMonolithicFluidSplitSaddlePoint::setup_system_matrix");

  // get the mortar structure to fluid coupling matrix M
  std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_m =
      coupling_solid_fluid_mortar_->get_mortar_matrix_m();

  // get the mortar fluid to structure coupling matrix D
  std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_d =
      coupling_solid_fluid_mortar_->get_mortar_matrix_d();

  // get time integration parameters of structure and fluid time integrators
  // to enable consistent time integration among the fields
  const double solid_time_int_param = structure_field()->tim_int_param();
  const double fluid_time_int_param = fluid_field()->tim_int_param();
  const double fluid_res_scale = fluid_field()->residual_scaling();

  // time scaling factor for fluid
  const double fluid_timescale = fluid_field()->time_scaling();

  // get fluid shape derivatives matrix
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> fluid_shape_deriv =
      fluid_field()->shape_derivatives();

  // get single field block matrices
  const std::shared_ptr<Core::LinAlg::SparseMatrix> solidblock = structure_field()->system_matrix();
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> fluidblock =
      fluid_field()->block_system_matrix();
  const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> aleblock =
      ale_field()->block_system_matrix();

  // extract submatrices
  const Core::LinAlg::SparseMatrix& fluid_inner_inner = fluidblock->matrix(0, 0);
  const Core::LinAlg::SparseMatrix& fluid_interf_inner = fluidblock->matrix(1, 0);
  const Core::LinAlg::SparseMatrix& fluid_inner_interf = fluidblock->matrix(0, 1);
  const Core::LinAlg::SparseMatrix& fluid_interf_interf = fluidblock->matrix(1, 1);
  const Core::LinAlg::SparseMatrix& ale_inner_inner = aleblock->matrix(0, 0);
  const Core::LinAlg::SparseMatrix& ale_inner_interf = aleblock->matrix(0, 1);

  // ---------------------------------------------------------------------------
  // BEGIN building the global 6x6 system matrix
  // ---------------------------------------------------------------------------
  // Contributions to blocks in system matrix are listed separately.
  // Block numbering in comments ranges from (1,1) to (6,6).

  // ---------Addressing contribution to blocks (1,1),(1,2),(2,1),(2,2)
  mat.assign(0, 0, Core::LinAlg::DataAccess::Share, *solidblock);

  // ---------Addressing contribution to blocks (3,3),(3,4),(4,3),(4,4)
  Core::LinAlg::SparseMatrix aux_fluidblock(fluidblock->full_row_map(), 108, false);
  Core::LinAlg::matrix_add(fluid_inner_inner, false, 1.0, aux_fluidblock, 0.0);
  Core::LinAlg::matrix_add(fluid_interf_inner, false, 1.0, aux_fluidblock, 1.0);
  Core::LinAlg::matrix_add(fluid_inner_interf, false, 1.0, aux_fluidblock, 1.0);
  Core::LinAlg::matrix_add(fluid_interf_interf, false, 1.0, aux_fluidblock, 1.0);
  aux_fluidblock.complete(
      fluidblock->full_domain_map(), fluidblock->full_range_map(), {.enforce_complete = true});

  // ---------Addressing contribution to block (5,4)
  std::shared_ptr<Core::LinAlg::SparseMatrix> aux_ale_inner_interf =
      std::make_shared<Core::LinAlg::SparseMatrix>(ale_inner_inner.row_map(), 81, false);
  (*ale_inner_interf_transform_)(aleblock->full_row_map(), aleblock->full_col_map(),
      ale_inner_interf, 1.,
      Coupling::Adapter::CouplingSlaveConverter(interface_fluid_ale_coupling()),
      *aux_ale_inner_interf);

  aux_ale_inner_interf->scale(1. / fluid_timescale);
  aux_ale_inner_interf->complete(
      fluidblock->domain_map(), aux_ale_inner_interf->range_map(), {.enforce_complete = true});

  mat.assign(2, 1, Core::LinAlg::DataAccess::Share, *aux_ale_inner_interf);

  // ---------Addressing contribution to block (5,5)
  mat.assign(2, 2, Core::LinAlg::DataAccess::Share, ale_inner_inner);

  // ---------Addressing contribution to block (6,2)
  std::shared_ptr<Core::LinAlg::SparseMatrix> aux_mortar_m =
      Core::LinAlg::matrix_row_transform_gids(*mortar_m, *lag_mult_dof_map_);
  aux_mortar_m->complete(solidblock->domain_map(), *lag_mult_dof_map_, {.enforce_complete = true});

  mat.assign(3, 0, Core::LinAlg::DataAccess::Share, *aux_mortar_m);

  // ---------Addressing contribution to block (2,6)
  aux_mortar_m = Core::LinAlg::matrix_row_transform_gids(*mortar_m, *lag_mult_dof_map_);
  Core::LinAlg::SparseMatrix aux_mortar_m_trans(solidblock->row_map(), 81, false);
  Core::LinAlg::matrix_add(
      *aux_mortar_m, true, -1.0 * (1.0 - solid_time_int_param), aux_mortar_m_trans, 0.0);
  aux_mortar_m_trans.complete(
      *lag_mult_dof_map_, solidblock->range_map(), {.enforce_complete = true});

  mat.assign(0, 3, Core::LinAlg::DataAccess::Share, aux_mortar_m_trans);

  // ---------Addressing contribution to block (6,4)
  std::shared_ptr<Core::LinAlg::SparseMatrix> aux_mortar_d =
      Core::LinAlg::matrix_row_transform_gids(*mortar_d, *lag_mult_dof_map_);

  aux_mortar_d->scale(-1.0 / fluid_timescale);
  aux_mortar_d->complete(
      fluidblock->full_domain_map(), *lag_mult_dof_map_, {.enforce_complete = true});

  mat.assign(3, 1, Core::LinAlg::DataAccess::Share, *aux_mortar_d);

  // ---------Addressing contribution to block (4,6)
  aux_mortar_d = Core::LinAlg::matrix_row_transform_gids(*mortar_d, *lag_mult_dof_map_);
  Core::LinAlg::SparseMatrix aux_mortar_d_trans(fluidblock->full_row_map(), 81, false);
  Core::LinAlg::matrix_add(*aux_mortar_d, true,
      1.0 * (1.0 - fluid_time_int_param) / fluid_res_scale, aux_mortar_d_trans, 0.0);
  aux_mortar_d_trans.complete(
      *lag_mult_dof_map_, fluidblock->full_range_map(), {.enforce_complete = true});

  mat.assign(1, 3, Core::LinAlg::DataAccess::Share, aux_mortar_d_trans);

  /*--------------------------------------------------------------------------*/
  // add optional fluid linearization with respect to mesh motion block
  if (fluid_shape_deriv != nullptr)
  {
    const Coupling::Adapter::Coupling& coup_fluid_ale = fluid_ale_coupling();

    // extract submatrices
    Core::LinAlg::SparseMatrix& fluid_mesh_inner_inner = fluid_shape_deriv->matrix(0, 0);
    Core::LinAlg::SparseMatrix& fluid_mesh_interf_inner = fluid_shape_deriv->matrix(1, 0);
    Core::LinAlg::SparseMatrix& fluid_mesh_inner_interf = fluid_shape_deriv->matrix(0, 1);
    Core::LinAlg::SparseMatrix& fluid_mesh_interf_interf = fluid_shape_deriv->matrix(1, 1);

    // Addressing contribution to block (3,4)
    Core::LinAlg::SparseMatrix aux_fluid_mesh_inner_interf(
        fluid_mesh_inner_interf.row_map(), 81, false);
    Core::LinAlg::matrix_add(fluid_mesh_inner_interf, false, 1.0, aux_fluid_mesh_inner_interf, 0.0);
    aux_fluid_mesh_inner_interf.complete(fluidblock->domain_map(),
        aux_fluid_mesh_inner_interf.range_map(), {.enforce_complete = true});
    Core::LinAlg::matrix_add(
        aux_fluid_mesh_inner_interf, false, 1. / fluid_timescale, aux_fluidblock, 1.0);

    // Addressing contribution to block (3,5)
    (*fluid_mesh_inner_inner_transform_)(fluid_shape_deriv->full_row_map(),
        fluid_shape_deriv->full_col_map(), fluid_mesh_inner_inner, 1.,
        Coupling::Adapter::CouplingSlaveConverter(coup_fluid_ale), mat.matrix(1, 2), false);

    // Addressing contribution to block (4,4)
    Core::LinAlg::SparseMatrix aux_fluid_mesh_interf_interf(
        fluid_mesh_interf_interf.row_map(), 81, false);
    Core::LinAlg::matrix_add(
        fluid_mesh_interf_interf, false, 1.0, aux_fluid_mesh_interf_interf, 0.0);
    aux_fluid_mesh_interf_interf.complete(fluidblock->domain_map(),
        aux_fluid_mesh_interf_interf.range_map(), {.enforce_complete = true});
    Core::LinAlg::matrix_add(
        aux_fluid_mesh_interf_interf, false, 1. / fluid_timescale, aux_fluidblock, 1.0);

    // Addressing contribution to block (4,5)
    (*fluid_mesh_inner_inner_transform_)(fluid_shape_deriv->full_row_map(),
        fluid_shape_deriv->full_col_map(), fluid_mesh_interf_inner, 1.,
        Coupling::Adapter::CouplingMasterConverter(coup_fluid_ale), mat.matrix(1, 2), false);
  }

  // finally assign fluid matrix to block (1,1)
  mat.assign(1, 1, Core::LinAlg::DataAccess::Share, aux_fluidblock);

  // done. make sure all blocks are filled.
  mat.complete();

  // Finally, take care of Dirichlet boundary conditions
  mat.apply_dirichlet(*(dbcmaps_->cond_map()), true);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::scale_system(
    Core::LinAlg::BlockSparseMatrixBase& mat, Core::LinAlg::Vector<double>& b)
{
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");
  const bool scaling_infnorm = fsimono.get<bool>("INFNORMSCALING");

  if (scaling_infnorm)
  {
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
    mat.matrix(0, 3).left_scale(*srowsum_);
    mat.matrix(1, 0).right_scale(*scolsum_);
    mat.matrix(2, 0).right_scale(*scolsum_);
    mat.matrix(3, 0).right_scale(*scolsum_);

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
    mat.matrix(2, 3).left_scale(*arowsum_);
    mat.matrix(0, 2).right_scale(*acolsum_);
    mat.matrix(1, 2).right_scale(*acolsum_);
    mat.matrix(3, 2).right_scale(*acolsum_);

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
void FSI::MortarMonolithicFluidSplitSaddlePoint::unscale_solution(
    Core::LinAlg::BlockSparseMatrixBase& mat, Core::LinAlg::Vector<double>& x,
    Core::LinAlg::Vector<double>& b)
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
    mat.matrix(0, 3).left_scale(*srowsum_);
    mat.matrix(1, 0).right_scale(*scolsum_);
    mat.matrix(2, 0).right_scale(*scolsum_);
    mat.matrix(3, 0).right_scale(*scolsum_);

    Core::LinAlg::SparseMatrix& A_22 = mat.matrix(2, 2);
    arowsum_->reciprocal(*arowsum_);
    acolsum_->reciprocal(*acolsum_);
    A_22.left_scale(*arowsum_);
    A_22.right_scale(*acolsum_);
    mat.matrix(2, 0).left_scale(*arowsum_);
    mat.matrix(2, 1).left_scale(*arowsum_);
    mat.matrix(2, 3).left_scale(*arowsum_);
    mat.matrix(0, 2).right_scale(*acolsum_);
    mat.matrix(1, 2).right_scale(*acolsum_);
    mat.matrix(3, 2).right_scale(*acolsum_);
  }

  Core::LinAlg::Vector<double> r(b.get_map());
  mat.multiply(false, x, r);
  r.update(1., b, 1.);

  std::shared_ptr<Core::LinAlg::Vector<double>> sr = extractor().extract_vector(r, 0);
  std::shared_ptr<Core::LinAlg::Vector<double>> fr = extractor().extract_vector(r, 1);
  std::shared_ptr<Core::LinAlg::Vector<double>> ar = extractor().extract_vector(r, 2);
  std::shared_ptr<Core::LinAlg::Vector<double>> lmr = extractor().extract_vector(r, 3);

  // increment additional ale residual
  aleresidual_->update(-1., *ar, 0.);

  std::ios_base::fmtflags flags = utils()->out().flags();

  double n, ns, nf, na, nlm;
  r.norm_2(&n);
  sr->norm_2(&ns);
  fr->norm_2(&nf);
  ar->norm_2(&na);
  lmr->norm_2(&nlm);
  utils()->out() << std::scientific << "\nlinear solver quality:\n"
                 << "L_2-norms:\n"
                 << "   |r|=" << n << "   |rs|=" << ns << "   |rf|=" << nf << "   |ra|=" << na
                 << "   |rlm|=" << nlm << "\n";
  r.norm_inf(&n);
  sr->norm_inf(&ns);
  fr->norm_inf(&nf);
  ar->norm_inf(&na);
  lmr->norm_inf(&nlm);
  utils()->out() << "L_inf-norms:\n"
                 << "   |r|=" << n << "   |rs|=" << ns << "   |rf|=" << nf << "   |ra|=" << na
                 << "   |rlm|=" << nlm << "\n";

  utils()->out().flags(flags);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::evaluate(
    std::shared_ptr<const Core::LinAlg::Vector<double>> step_increment)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MortarMonolithicFluidSplitSaddlePoint::Evaluate");

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check whether all fields have the same time step size
  check_if_dts_same();
#endif

  std::shared_ptr<const Core::LinAlg::Vector<double>> sx;
  std::shared_ptr<const Core::LinAlg::Vector<double>> fx;
  std::shared_ptr<const Core::LinAlg::Vector<double>> ax;
  std::shared_ptr<const Core::LinAlg::Vector<double>> lagx;

  if (step_increment != nullptr)
  {
    extract_field_vectors(step_increment, sx, fx, ax, lagx);
  }

  // Call all elements and assemble rhs and matrices
  // We only need the rhs here because NOX will ask for the rhs
  // only. But the Jacobian is stored internally and will be returned
  // later on without looking at x again!

  if (verbosity_ >= Inpar::FSI::verbosity_medium) utils()->out() << "\nEvaluate elements\n";

  {
    Teuchos::Time ts("structure", true);
    structure_field()->evaluate(sx);
    if (verbosity_ >= Inpar::FSI::verbosity_medium)
      utils()->out() << "structure           : " << ts.totalElapsedTime(true) << " sec\n";
  }

  {
    Teuchos::Time ta("ale", true);
    ale_field()->evaluate(ax);
    if (verbosity_ >= Inpar::FSI::verbosity_medium)
      utils()->out() << "ale                 : " << ta.totalElapsedTime(true) << " sec\n";
  }

  // transfer the current ale mesh positions to the fluid field
  std::shared_ptr<Core::LinAlg::Vector<double>> fluiddisp = ale_to_fluid(ale_field()->dispnp());
  fluid_field()->apply_mesh_displacement(fluiddisp);

  {
    Teuchos::Time tf("fluid", true);
    fluid_field()->evaluate(fx);
    if (verbosity_ >= Inpar::FSI::verbosity_medium)
      utils()->out() << "fluid                : " << tf.totalElapsedTime(true) << " sec\n";
  }

  {
    if (lagx != nullptr)
    {
      Teuchos::Time tlm("lag_mult", true);
      lag_mult_->update(1.0, *lag_mult_old_, 1.0, *lagx, 0.0);
      if (verbosity_ >= Inpar::FSI::verbosity_medium)
        utils()->out() << "Lagrange multiplier: " << tlm.totalElapsedTime(true) << " sec\n";
    }
  }

  if (verbosity_ >= Inpar::FSI::verbosity_medium) utils()->out() << "\n";
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::extract_field_vectors(
    std::shared_ptr<const Core::LinAlg::Vector<double>> x,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& fx,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& ax,
    std::shared_ptr<const Core::LinAlg::Vector<double>>& lagx)
{
  TEUCHOS_FUNC_TIME_MONITOR("FSI::MortarMonolithicFluidSplitSaddlePoint::extract_field_vectors");

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (ddgpred_ == nullptr) FOUR_C_THROW("Vector 'ddgpred_' has not been initialized properly.");
#endif

  // get the mortar structure to fluid coupling matrix M
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_m =
      coupling_solid_fluid_mortar_->get_mortar_matrix_m();
  std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_m_transf =
      Core::LinAlg::matrix_row_transform_gids(*mortar_m, *lag_mult_dof_map_);

  // get the mortar fluid to structure coupling matrix D
  const std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_d =
      coupling_solid_fluid_mortar_->get_mortar_matrix_d();
  std::shared_ptr<Core::LinAlg::SparseMatrix> mortar_d_transf =
      Core::LinAlg::matrix_row_transform_gids(*mortar_d, *lag_mult_dof_map_);

  // ---------------------------------------------------------------------------
  // process structure unknowns
  // ---------------------------------------------------------------------------
  sx = extractor().extract_vector(*x, 0);
  // extract structure solution increment from NOX increment

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

  lagx = extractor().extract_vector(*x, 3);

  ax = a;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::update()
{
  // save Lagrange multiplier for the next time step
  lag_mult_old_->update(1.0, *lag_mult_, 0.0);

  // call update()-routine in base class to handle the single fields
  FSI::BlockMonolithic::update();
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::output()
{
  structure_field()->output();
  fluid_field()->output();

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
void FSI::MortarMonolithicFluidSplitSaddlePoint::output_lambda()
{
  /* 'lag_mult_' is only defined on the interface. So, insert 'lag_mult_' into
   * 'lambdafull' that is defined on the entire fluid field. Then, we need to write
   * output or restart data.
   */
  Core::LinAlg::Vector<double> copy(*lag_mult_);
  copy.replace_map(*fluid_field()->interface()->fsi_cond_map());
  std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
      fluid_field()->interface()->insert_fsi_cond_vector(copy);
  const int uprestart = timeparams_.get<int>("RESTARTEVERY");
  const int upres = timeparams_.get<int>("RESULTSEVERY");
  if ((uprestart != 0 and fluid_field()->step() % uprestart == 0) or
      (upres != 0 and fluid_field()->step() % upres == 0))
    fluid_field()->disc_writer()->write_vector("fsilambda", lambdafull);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void FSI::MortarMonolithicFluidSplitSaddlePoint::read_restart(int step)
{
  structure_field()->read_restart(step);
  fluid_field()->read_restart(step);

  // read Lagrange multiplier into fluid map
  std::shared_ptr<Core::LinAlg::Vector<double>> lambdafull =
      std::make_shared<Core::LinAlg::Vector<double>>(*fluid_field()->dof_row_map(), true);
  Core::IO::DiscretizationReader reader = Core::IO::DiscretizationReader(
      *fluid_field()->discretization(), Global::Problem::instance()->input_control_file(), step);
  reader.read_vector(lambdafull, "fsilambda");
  auto lag_mult_old_on_fluid_map = fluid_field()->interface()->extract_fsi_cond_vector(*lambdafull);

  // Convert Lagrange multipliers to their actual map
  lag_mult_old_on_fluid_map->replace_map(*lag_mult_dof_map_);
  lag_mult_old_ = std::make_shared<Core::LinAlg::Vector<double>>(*lag_mult_old_on_fluid_map);

  // Note: the above is normally enough. However, we can use the restart in order to periodically
  // repeat the fsi simulation (see AC-FS3I)
  lag_mult_ = std::make_shared<Core::LinAlg::Vector<double>>(*lag_mult_old_on_fluid_map);

  setup_system();

  ale_field()->read_restart(step);

  set_time_step(fluid_field()->time(), fluid_field()->step());
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
double FSI::MortarMonolithicFluidSplitSaddlePoint::select_dt_error_based() const
{
  // get time step size suggestions
  const double dtstr = get_ada_str_dt();           // based on all structure DOFs
  const double dtstrfsi = get_ada_str_fsi_dt();    // based on structure FSI DOFs
  const double dtflinner = get_ada_fl_inner_dt();  // based on inner fluid DOFs

  double dt = MortarMonolithicFluidSplitSaddlePoint::dt();

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
bool FSI::MortarMonolithicFluidSplitSaddlePoint::set_accepted() const
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

FOUR_C_NAMESPACE_CLOSE
