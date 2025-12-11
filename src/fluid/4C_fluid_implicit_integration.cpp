// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_implicit_integration.hpp"

#include "4C_ale_ale3.hpp"
#include "4C_comm_utils.hpp"
#include "4C_fem_condition_locsys.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization_faces.hpp"
#include "4C_fem_discretization_hdg.hpp"
#include "4C_fem_discretization_nullspace.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fem_nurbs_discretization_initial_condition.hpp"
#include "4C_fluid_DbcHDG.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_intfaces_calc.hpp"
#include "4C_fluid_ele_parameter_intface.hpp"
#include "4C_fluid_ele_parameter_std.hpp"
#include "4C_fluid_impedancecondition.hpp"
#include "4C_fluid_meshtying.hpp"
#include "4C_fluid_result_test.hpp"
#include "4C_fluid_turbulence_boxfilter.hpp"
#include "4C_fluid_turbulence_dyn_smag.hpp"
#include "4C_fluid_turbulence_dyn_vreman.hpp"
#include "4C_fluid_turbulence_hit_forcing.hpp"
#include "4C_fluid_turbulence_hit_initial_field.hpp"
#include "4C_fluid_turbulence_statistic_manager.hpp"
#include "4C_fluid_turbulence_transfer_turb_inflow.hpp"
#include "4C_fluid_utils.hpp"
#include "4C_fluid_utils_infnormscaling.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fluid_xwall.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_xfem.hpp"  //for enums only
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_io_gmsh.hpp"
#include "4C_linalg_krylov_projector.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_function.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                     gammi 04/07|
 *----------------------------------------------------------------------*/
FLD::FluidImplicitTimeInt::FluidImplicitTimeInt(
    const std::shared_ptr<Core::FE::Discretization>& actdis,
    const std::shared_ptr<Core::LinAlg::Solver>& solver,
    const std::shared_ptr<Teuchos::ParameterList>& params,
    const std::shared_ptr<Core::IO::DiscretizationWriter>& output, bool alefluid /*= false*/
    )
    : TimInt(actdis, solver, params, output),
      // call constructor for "nontrivial" objects
      alefluid_(alefluid),
      writestresses_(params_->get<bool>("write stresses", false)),
      write_wall_shear_stresses_(params_->get<bool>("write wall shear stresses", false)),
      write_eledata_everystep_(params_->get<bool>("write element data in every step", false)),
      write_nodedata_first_step_(params_->get<bool>("write node data in first step")),
      dtele_(0.0),
      dtfilter_(0.0),
      dtsolve_(0.0),
      external_loads_(nullptr),
      forcing_(nullptr),
      forcing_interface_(nullptr),
      velpressplitter_(std::make_shared<Core::LinAlg::MapExtractor>()),
      surfacesplitter_(nullptr),
      inrelaxation_(false),
      xwall_(nullptr),
      msht_(Inpar::FLUID::no_meshtying),
      facediscret_(nullptr),
      fldgrdisp_(nullptr),
      locsysman_(nullptr),
      impedancebc_(nullptr),
      isimpedancebc_(false),
      off_proc_assembly_(params_->get<bool>("OFF_PROC_ASSEMBLY", false)),
      ndsale_((Global::Problem::instance()->spatial_approximation_type() ==
                  Core::FE::ShapeFunctionType::hdg) *
              2),
      massmat_(nullptr),
      logenergy_(nullptr),
      couplingcontributions_(nullptr)
{
}


/*----------------------------------------------------------------------*
 |  initialize algorithm                                rasthofer 04/14 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::init()
{
  // time measurement: initialization
  TEUCHOS_FUNC_TIME_MONITOR(" + initialization");

  // -------------------------------------------------------------------
  // get the basic parameters first
  // -------------------------------------------------------------------
  dtp_ = params_->get<double>("time step size");

  // parameter theta for time-integration schemes (required for all schemes)
  theta_ = params_->get<double>("theta");

  // cfl computation type and cfl number for adaptive time stepping
  cfl_estimator_ = Teuchos::getIntegralValue<Inpar::FLUID::AdaptiveTimeStepEstimator>(
      (params_->sublist("TIMEADAPTIVITY")), "ADAPTIVE_TIME_STEP_ESTIMATOR");
  cfl_ = params_->sublist("TIMEADAPTIVITY").get<double>("CFL_NUMBER", -1.0);
  if (cfl_estimator_ == Inpar::FLUID::cfl_number && cfl_ < 0.0)
    FOUR_C_THROW("specify cfl number for adaptive time step via cfl");

  // number of steps for starting algorithm, only for GenAlpha so far
  numstasteps_ = params_->get<int>("number of start steps");

  // parameter for linearization scheme (fixed-point-like or Newton)
  newton_ = Teuchos::getIntegralValue<Inpar::FLUID::LinearisationAction>(*params_, "Linearisation");

  predictor_ = params_->get<std::string>("predictor", "steady_state_predictor");

  // flag to skip calculation of residual after solution has converged
  inconsistent_ = params_->get<bool>("INCONSISTENT_RESIDUAL", false);
  if (inconsistent_ and myrank_ == 0)
  {
    std::cout << "Warning: residual will not be adapted to the final solution of the nonlinear "
                 "solution procedure!"
              << std::endl;
  }

  // form of convective term
  convform_ = params_->get<std::string>("form of convective term", "convective");



  // -------------------------------------------------------------------
  // flag for potential nonlinear boundary conditions
  // -------------------------------------------------------------------
  nonlinearbc_ = params_->get<bool>("Nonlinear boundary conditions", false);

  compute_null_space_if_necessary(*discret_, solver_->params(), true);

  // ensure that degrees of freedom in the discretization have been set
  if (!discret_->filled() || !discret_->have_dofs()) discret_->fill_complete();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  // -------------------------------------------------------------------
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization for a vector which only
  // contains the velocity dofs and for one vector which only contains
  // pressure degrees of freedom.
  // -------------------------------------------------------------------
  numdim_ = params_->get<int>("number of velocity degrees of freedom");

  if (velpressplitter_->num_maps() == 0)
    Core::LinAlg::create_map_extractor_from_discretization(*discret_, numdim_, *velpressplitter_);
  // if the pressure map is empty, the user obviously specified a wrong
  // number of space dimensions in the input file
  if (velpressplitter_->cond_map()->num_global_elements() < 1)
    FOUR_C_THROW("Pressure map empty. Wrong DIM value in input file?");

  // -------------------------------------------------------------------
  // create empty vectors
  // -------------------------------------------------------------------
  reset();

  // ---------------------------------------------------------------------
  // Set initial ALE mesh displacement and velocity
  // ---------------------------------------------------------------------
  if (alefluid_)
  {
    discret_->set_state(ndsale_, "dispnp", *dispnp_);
    discret_->set_state(ndsale_, "gridv", *gridv_);
  }

  // initialize nonlinear boundary conditions
  if (nonlinearbc_) init_nonlinear_bc();

  // ---------------------------------------------------------------------
  // Create LocSysManager, if needed (used for LocSys-Dirichlet BCs)
  // ---------------------------------------------------------------------
  {
    std::vector<const Core::Conditions::Condition*> locsysconditions;
    discret_->get_condition("Locsys", locsysconditions);
    if (locsysconditions.size())
    {
      // Initialize locsys manager
      locsysman_ = std::make_shared<Core::Conditions::LocsysManager>(
          *discret_, Global::Problem::instance()->n_dim());
      setup_locsys_dirichlet_bc(-1.0);
    }
  }

  // Vectors associated to boundary conditions
  // -----------------------------------------

  // a vector of zeros to be used to enforce zero dirichlet boundary conditions
  zeros_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // object holds maps/subsets for DOFs subjected to Dirichlet BCs and otherwise
  dbcmaps_ = std::make_shared<Core::LinAlg::MapExtractor>();
  {
    Teuchos::ParameterList eleparams;
    // other parameters needed by the elements
    eleparams.set("total time", time_);
    eleparams.set<const Core::Utils::FunctionManager*>(
        "function_manager", &Global::Problem::instance()->function_manager());

    apply_dirichlet_bc(eleparams, zeros_, nullptr, nullptr, true);
    // zeros_ has to be reset to zero here, since it has a different value after call of
    // apply_dirichlet_bc(...)
    zeros_->put_scalar(0.0);
  }

  // a vector containing the integrated traction in boundary normal direction for slip boundary
  // conditions (Unit: Newton [N])
  slip_bc_normal_tractions_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // manager for wall stress related things
  stressmanager_ =
      std::make_shared<FLD::Utils::StressManager>(discret_, dispnp_, alefluid_, numdim_);

  // -------------------------------------------------------------------
  // create empty system matrix --- stiffness and mass are assembled in
  // one system matrix!
  // -------------------------------------------------------------------

  // This is a first estimate for the number of non zeros in a row of
  // the matrix. Assuming a structured 3d-fluid mesh we have 27 adjacent
  // nodes with 4 dofs each. (27*4=108)
  // We do not need the exact number here, just for performance reasons
  // a 'good' estimate

  {
    // XWall: enrichment with spaldings law
    if (params_->sublist("WALL MODEL").get<bool>("X_WALL"))
    {
      if (Global::Problem::instance()->get_problem_type() == Core::ProblemType::fsi ||
          Global::Problem::instance()->get_problem_type() == Core::ProblemType::fluid_ale)
      {
        xwall_ = std::make_shared<XWallAleFSI>(
            discret_, numdim_, params_, dbcmaps_, stressmanager_, dispnp_, gridv_);
      }
      else
        xwall_ = std::make_shared<XWall>(discret_, numdim_, params_, dbcmaps_, stressmanager_);
    }
  }

  if (!params_->get<bool>("BLOCKMATRIX"))
  {
    if (Teuchos::getIntegralValue<Inpar::FLUID::MeshTying>(*params_, "MESHTYING") ==
        Inpar::FLUID::no_meshtying)
    {
      // initialize standard (stabilized) system matrix (construct its graph already)
      // off_proc_assembly_ requires an EpetraFECrs matrix
      if (off_proc_assembly_)
      {
        sysmat_ = std::make_shared<Core::LinAlg::SparseMatrix>(
            *dofrowmap, 108, false, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
      }
      else
      {
        sysmat_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap, 108, false, true);
      }
    }
    else if (Teuchos::getIntegralValue<Inpar::FLUID::MeshTying>(*params_, "MESHTYING") !=
             Inpar::FLUID::no_meshtying)
    {
      setup_meshtying();
      if (off_proc_assembly_)
        FOUR_C_THROW("Off processor assembly currently not available for this matrix type");
    }
  }
  else
  {
    std::shared_ptr<Core::LinAlg::BlockSparseMatrix<FLD::Utils::VelPressSplitStrategy>>
        blocksysmat =
            std::make_shared<Core::LinAlg::BlockSparseMatrix<FLD::Utils::VelPressSplitStrategy>>(
                *velpressplitter_, *velpressplitter_, 108, false, true);
    blocksysmat->set_numdim(numdim_);
    sysmat_ = blocksysmat;
    if (off_proc_assembly_)
      FOUR_C_THROW("Off processor assembly currently not available for this matrix type");
  }

  // the vector containing body and surface forces
  neumann_loads_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // Vectors used for solution process
  // ---------------------------------
  // rhs: standard (stabilized) residual vector (rhs for the incremental form)
  residual_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  trueresidual_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // Nonlinear iteration increment vector
  incvel_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // -------------------------------------------------------------------
  // initialize vectors and flags for turbulence approach
  // -------------------------------------------------------------------
  set_general_turbulence_parameters();

  // -------------------------------------------------------------------
  // initialize turbulence-statistics evaluation
  // -------------------------------------------------------------------
  statisticsmanager_ = std::make_shared<FLD::TurbulenceStatisticManager>(*this);
  // parameter for sampling/dumping period
  if (special_flow_ != "no")
    samstart_ = params_->sublist("TURBULENCE MODEL").get<int>("SAMPLING_START", 1);

  // set gas constant to 1.0 for incompressible flow
  gasconstant_ = 1.0;

  if (params_->get<bool>("INFNORMSCALING"))
  {
    fluid_infnormscaling_ = std::make_shared<FLD::Utils::FluidInfNormScaling>(*velpressplitter_);
  }

  // ------------------------------------------------------------------------------
  // Check, if features are used with the locsys manager that are not supported,
  // or better, not implemented yet.
  // ------------------------------------------------------------------------------
  if (locsysman_ != nullptr)
  {
    // TangVel predictor
    if (predictor_ == "TangVel")
    {
      FOUR_C_THROW(
          "No problem types involving TangVel predictors are supported for use with locsys "
          "conditions!");
    }

    // Meshtying
    if (msht_ != Inpar::FLUID::no_meshtying)
    {
      FOUR_C_THROW(
          "No problem types involving meshtying are supported for use with locsys conditions!");
    }

    // Additionally, locsys doesn't work yet with AVM3 and linear_relaxation_solve. Those
    // checks needed to be put in the corresponding functions, so they are not listed here.
  }

  // for the case of edge-oriented stabilization
  Teuchos::ParameterList* stabparams = &(params_->sublist("RESIDUAL-BASED STABILIZATION"));
  if (stabparams->get<Inpar::FLUID::StabType>("STABTYPE") == Inpar::FLUID::stabtype_edgebased)
  {
    create_faces_extension();
  }
  reconstructder_ = stabparams->get<bool>("Reconstruct_Sec_Der");

}  // FluidImplicitTimeInt::init()

/*----------------------------------------------------------------------*
 |  create internal faces for the case of EOS stab                      |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::create_faces_extension()
{
  // if the definition of internal faces would be included
  // in the standard discretization, these lines can be removed
  // and create_internal_faces_extension() can be called once
  // in the constructor of the fluid time integration
  // since we want to keep the standard discretization as clean as
  // possible, we create internal faces via an enhanced discretization
  // including the faces between elements
  facediscret_ = std::dynamic_pointer_cast<Core::FE::DiscretizationFaces>(discret_);
  facediscret_->create_internal_faces_extension(true);
}
/*----------------------------------------------------------------------*
 |  initialize algorithm for nonlinear BCs                   thon 09/14 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::init_nonlinear_bc()
{
  // initialize flow-rate and flow-volume vectors (fixed to length of four,
  // for the time being) in case of flow-dependent pressure boundary conditions,
  // including check of respective conditions.
  std::vector<const Core::Conditions::Condition*> flowdeppressureline;
  discret_->get_condition("LineFlowDepPressure", flowdeppressureline);
  std::vector<const Core::Conditions::Condition*> flowdeppressuresurf;
  discret_->get_condition("SurfaceFlowDepPressure", flowdeppressuresurf);
  std::vector<const Core::Conditions::Condition*> impedancecond;
  discret_->get_condition("ImpedanceCond", impedancecond);

  // check number of flow-rate and flow-volume boundary conditions
  if (flowdeppressureline.size() > 0 or flowdeppressuresurf.size() > 0)
  {
    // get the number of flow dependent line or surface conditions
    std::string fdpcondname;
    if (flowdeppressureline.size() != 0)
      fdpcondname = "LineFlowDepPressure";
    else if (flowdeppressuresurf.size() != 0)
      fdpcondname = "SurfaceFlowDepPressure";
    else
    {
      FOUR_C_THROW(
          "Line and surface flow-dependent pressure boundary conditions simultaneously "
          "prescribed!");
    }

    // get condition vector
    std::vector<const Core::Conditions::Condition*> fdpcond;
    discret_->get_condition(fdpcondname, fdpcond);

    // initialize vectors for flow rate and volume
    size_t numcond = (int)fdpcond.size();
    flowratenp_.resize(numcond, 0.0);
    flowratenpi_.resize(numcond, 0.0);
    flowraten_.resize(numcond, 0.0);
    flowratenm_.resize(numcond, 0.0);

    flowvolumenp_.resize(numcond, 0.0);
    flowvolumenpi_.resize(numcond, 0.0);
    flowvolumen_.resize(numcond, 0.0);
    flowvolumenm_.resize(numcond, 0.0);
  }

  // check number of impedance boundary conditions
  if (impedancecond.size() > 0)
  {
    if (alefluid_)
    {
      discret_->clear_state();
      discret_->set_state(ndsale_, "dispnp", *dispnp_);
    }

    impedancebc_ = std::make_shared<Utils::FluidImpedanceWrapper>(discret_);
    isimpedancebc_ = true;  // Set bool to true since there is an impedance BC

    // Test if also AVM3 is used
    fssgv_ = Teuchos::getIntegralValue<Inpar::FLUID::FineSubgridVisc>(
        params_->sublist("TURBULENCE MODEL"), "FSSUGRVISC");
    if (fssgv_ != Inpar::FLUID::no_fssgv)
    {
      FOUR_C_THROW(
          "The functionality of impedance BC together with AVM3 is not known. Take a look into "
          "function avm3_preparation()");
    }
  }
}


/*----------------------------------------------------------------------*
 | complete initialization                                              |
 |                                                                      |
 |  o is called at the end of the constructor of the time integrators   |
 |  o used for init functions that require the time integrators to exist|
 |                                                              bk 01/14|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::complete_general_init()
{
  // -------------------------------------------------------------------
  // initialize forcing for homogeneous isotropic turbulence
  // -------------------------------------------------------------------
  init_forcing();

  // Set general parameters:
  // the following two functions are overloaded (switched off) in TimIntPoro
  set_element_general_fluid_parameter();
  set_element_turbulence_parameters();

  // set special parameter for faces/edges when using edge-based fluid stabilizations
  if (params_->sublist("RESIDUAL-BASED STABILIZATION").get<Inpar::FLUID::StabType>("STABTYPE") ==
      Inpar::FLUID::StabType::stabtype_edgebased)
    set_face_general_fluid_parameter();

  // sysmat might be singular (if we have a purely Dirichlet constrained
  // problem, the pressure mode is defined only up to a constant)
  // in this case, we need a basis vector for the nullspace/kernel

  // initialize Krylov space projection
  init_krylov_space_projection();

  // Initialize WSS manager if smoothing via aggregation is desired
  if (not stressmanager_->is_init())
  {
    // necessary for the assembly
    set_element_time_parameter();

    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // necessary here, because some application time integrations add something to the residual
    // before the Neumann loads are added
    residual_->put_scalar(0.0);

    avm3_assemble_mat_and_rhs(eleparams);
    stressmanager_->init_aggr(sysmat_);
  }

  // ------------------------------------------------------------------------------
  // Pre-compute mass matrix in case the user wants output of kinetic energy
  // ------------------------------------------------------------------------------
  if (params_->get<bool>("COMPUTE_EKIN"))
  {
    // write energy-file
    {
      std::string fileiter = Global::Problem::instance()->output_control_file()->file_name();
      fileiter.append(".fluidenergy");
      logenergy_ = std::make_shared<std::ofstream>(fileiter.c_str());

      // write header of energy-file (if energy file is desired by user)
      if (myrank_ == 0 and (logenergy_))
      {
        (*logenergy_) << "# Kinetic energy in fluid field\n"
                      << "# num procs = "
                      << Core::Communication::num_mpi_ranks(discretization()->get_comm())
                      << std::endl
                      << std::right << std::setw(9) << "# step" << std::right << std::setw(16)
                      << "time" << std::right << std::setw(16) << "kinetic_energy" << std::endl;

        (*logenergy_) << "#" << std::endl;
      }
    }

    massmat_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dof_row_map(), 108, false, true);
    evaluate_mass_matrix();
  }
}

/*----------------------------------------------------------------------*
 | Start the time integration. Allows                                   |
 |                                                                      |
 |  o starting steps with different algorithms                          |
 |  o the "standard" time integration                                   |
 |                                                           gammi 04/07|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::integrate()
{
  print_stabilization_details();

  // TimeLoop() calls solve_stationary_problem() in stationary case
  time_loop();
}  // FluidImplicitTimeInt::Integrate

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 | contains the time loop                                    gammi 04/07|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void FLD::FluidImplicitTimeInt::time_loop()
{
  // time measurement: time loop
  TEUCHOS_FUNC_TIME_MONITOR(" + time loop");

  while (not_finished())
  {
    // -------------------------------------------------------------------
    //                       evaluate time step size if applicable
    // -------------------------------------------------------------------
    set_dt(evaluate_dt_via_cfl_if_applicable());

    // -------------------------------------------------------------------
    //                       prepare time step
    // -------------------------------------------------------------------
    prepare_time_step();
    // -------------------------------------------------------------------
    //                       output to screen
    // -------------------------------------------------------------------
    print_time_step_info();

    // -----------------------------------------------------------------
    // intermediate solution step for homogeneous isotropic turbulence
    // -----------------------------------------------------------------
    calc_intermediate_solution();

    // -----------------------------------------------------------------
    //                     solve nonlinear equation
    // -----------------------------------------------------------------
    solve();

    // -------------------------------------------------------------------
    //                         update solution
    //        current solution becomes old solution of next timestep
    // -------------------------------------------------------------------
    time_update();

    // -------------------------------------------------------------------
    //  lift'n'drag forces, statistics time sample and output of solution
    //  and statistics
    // -------------------------------------------------------------------
    statistics_and_output();

    // -------------------------------------------------------------------
    //                    stop criterion for timeloop
    // -------------------------------------------------------------------
  }

}  // FluidImplicitTimeInt::TimeLoop


void FLD::FluidImplicitTimeInt::setup_locsys_dirichlet_bc(double time)
{
  // Check how many locsys conditions exist
  std::vector<const Core::Conditions::Condition*> locsysconds_;
  discret_->get_condition("Locsys", locsysconds_);
  int numlocsys = (int)locsysconds_.size();

  if (numlocsys > 0)
  {
    // estimate the normals for every locsys condition
    std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> loc_sys_node_normals;
    loc_sys_node_normals.resize(numlocsys);

    for (int i = 0; i < numlocsys; ++i)
    {
      loc_sys_node_normals[i] =
          std::make_shared<Core::LinAlg::Vector<double>>(*discret_->dof_row_map(), true);

      Teuchos::ParameterList nodeNormalParams;

      // Set action for elements
      if (discret_->name() == "ale")
      {
        if (numdim_ == 2)
          FOUR_C_THROW("Locsys: Node Normal for type 'ale', only 3D case is implemented.");
        else
          nodeNormalParams.set<Discret::Elements::Ale3::ActionType>(
              "action", Discret::Elements::Ale3::boundary_calc_ale_node_normal);
      }
      else
      {
        nodeNormalParams.set<FLD::BoundaryAction>("action", FLD::boundary_calc_node_normal);
      }
      discret_->evaluate_condition(nodeNormalParams, loc_sys_node_normals[i], "Locsys", i);
    }
    locsysman_->update(time, loc_sys_node_normals, Global::Problem::instance()->function_manager());
  }
  else
    locsysman_->update(time, {}, Global::Problem::instance()->function_manager());

  discret_->clear_state();
}

/*----------------------------------------------------------------------*
 | setup the variables to do a new time step                 u.kue 06/07|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::prepare_time_step()
{
  // -------------------------------------------------------------------
  //              set time-dependent parameters
  // -------------------------------------------------------------------
  increment_time_and_step();

  // Sets theta_ to a specific value for bdf2 and calculates
  // a pseudo-theta for genalpha (the latter in case of startalgo_)
  set_theta();

  // -------------------------------------------------------------------
  // set part(s) of the rhs vector(s) belonging to the old timestep
  // (only meaningful for momentum part)
  //
  // stationary/af-generalized-alpha: hist_ = 0.0
  //
  // one-step-Theta:                  hist_ = veln_ + dt*(1-Theta)*accn_
  //
  // BDF2: for constant time step:    hist_ = 4/3*veln_ - 1/3*velnm_
  //
  // -------------------------------------------------------------------
  set_old_part_of_righthandside();

  // -------------------------------------------------------------------
  //                     do explicit predictor step
  // -------------------------------------------------------------------

  // no predictor in first time step
  if (step_ > 1)
  {
    if (predictor_ != "TangVel")
    {
      explicit_predictor();
    }
    else
    {
      predict_tang_vel_consist_acc();
    }
  }

  // -------------------------------------------------------------------
  //  Set time parameter for element call
  // -------------------------------------------------------------------
  set_element_time_parameter();

  // -------------------------------------------------------------------
  // Update local coordinate systems (which may be time dependent)
  // -------------------------------------------------------------------
  if (locsysman_ != nullptr)
  {
    discret_->clear_state();
    if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);

    setup_locsys_dirichlet_bc(time_);
  }


  // ----------------------------------------------------------------
  // Calculate new wall shear stress for xwall, if appropriate
  // ----------------------------------------------------------------
  if (xwall_ != nullptr)
  {
    // Transfer of boundary data if necessary
    turbulent_inflow_condition_->transfer(trueresidual_, trueresidual_, time_);
    xwall_->update_tau_w(step_, trueresidual_, 0, accn_, velnp_, veln_);
  }

  // -------------------------------------------------------------------
  //  evaluate Dirichlet and Neumann boundary conditions
  // -------------------------------------------------------------------
  set_dirichlet_neumann_bc();


  // --------------------------------------------------
  // adjust accnp according to Dirichlet values of velnp for GenAlpha
  //
  gen_alpha_update_acceleration();
  // ----------------------------------------------------------------
  // compute values at intermediate time steps for GenAlpha
  // ----------------------------------------------------------------
  gen_alpha_intermediate_values();

  // -------------------------------------------------------------------
  // meshtying: evaluation of matrix P with potential mesh relocation
  // in ALE case
  // -------------------------------------------------------------------
  if (msht_ != Inpar::FLUID::no_meshtying and alefluid_)
    meshtying_->evaluate_with_mesh_relocation(dispnp_);

  // -------------------------------------------------------------------
  //           preparation of AVM3-based scale separation
  // -------------------------------------------------------------------
  if (step_ == 1 and
      (fssgv_ != Inpar::FLUID::no_fssgv or scale_sep_ == FLUID::algebraic_multigrid_operator))
    avm3_preparation();
}

/*----------------------------------------------------------------------*
 | nonlinear solve, i.e., (multiple) corrector                 vg 02/09 |
 |   																	|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::solve()
{
  // -------------------------------------------------------------------
  // time measurement: nonlinear iteration
  // -------------------------------------------------------------------
  TEUCHOS_FUNC_TIME_MONITOR("   + corrector");

  dtsolve_ = 0.0;

  // -------------------------------------------------------------------
  // parameters and variables for nonlinear iteration
  // -------------------------------------------------------------------
  int itnum = 0;
  int itmax = 0;
  const double velrestol = params_->get<double>("velocity residual tolerance");
  const double velinctol = params_->get<double>("velocity increment tolerance");
  const double presrestol = params_->get<double>("pressure residual tolerance");
  const double presinctol = params_->get<double>("pressure increment tolerance");
  const double ittol = std::min(std::min(std::min(velrestol, presrestol), velinctol), presinctol);

  bool stopnonliniter = false;

  // -------------------------------------------------------------------
  // currently default for turbulent channel flow:
  // only one iteration before sampling
  // -------------------------------------------------------------------
  // REMARK:
  // commented reduced number of iterations out as it seems that more iterations
  // are necessary before sampling to obtain a converged result
  //  if (special_flow_ == "channel_flow_of_height_2" && step_<samstart_ )
  //       itmax = 1;
  //  else
  itmax = params_->get<int>("max nonlin iter steps");

  // -------------------------------------------------------------------
  // turn adaptive solver tolerance on/off
  // -------------------------------------------------------------------
  const bool isadapttol = params_->get<bool>("ADAPTCONV", true);
  const double adaptolbetter = params_->get<double>("ADAPTCONV_BETTER", 0.01);

  // -------------------------------------------------------------------
  // option for multifractal subgrid-scale modeling approach within
  // variable-density flow at low Mach number:
  // adaption of CsgsD to resolution dependent CsgsB
  // when near-wall limit is used
  // see also comment within function
  // -------------------------------------------------------------------
  if ((physicaltype_ == Inpar::FLUID::loma or statisticsmanager_->with_scatra()) and
      turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales)
    recompute_mean_csgs_b();

  // -------------------------------------------------------------------
  // prepare print out for (multiple) corrector
  // -------------------------------------------------------------------
  if (myrank_ == 0)
  {
    printf("+------------+-------------+-------------+-------------+-------------+\n");
    printf(
        "|- step/max -|-- vel-res --|-- pre-res --|-- vel-inc --|-- pre-inc "
        "--|\n");
    printf(
        "|-   norm   -|-- abs. L2 --|-- abs. L2 --|-- rel. L2 --|-- rel. L2 "
        "--|\n");
    printf("|-   tol    -| %10.3E  | %10.3E  | %10.3E  | %10.3E  |\n", velrestol, presrestol,
        velinctol, presinctol);
  }

  // -------------------------------------------------------------------
  // nonlinear iteration loop
  // -------------------------------------------------------------------
  while (!stopnonliniter)
  {
    itnum++;

    // necessary for adaptive quadrature
    if (xwall_ != nullptr) xwall_->set_iter(itnum);
    // -------------------------------------------------------------------
    // preparatives for solver
    // -------------------------------------------------------------------
    prepare_solve();

    // -------------------------------------------------------------------
    // solver:
    // - It is solved for velocity and pressure increments.
    // - Adaptive linear solver tolerance is used from second corrector
    //   step on.
    // - Time for solver is measured.
    // -------------------------------------------------------------------
    {
      TEUCHOS_FUNC_TIME_MONITOR("      + solver calls");

      const double tcpusolve = Teuchos::Time::wallTime();
      Core::LinAlg::SolverParams solver_params;
      if (isadapttol and itnum > 1)
      {
        double currresidual = std::max(vresnorm_, presnorm_);
        currresidual = std::max(currresidual, incvelnorm_L2_ / velnorm_L2_);
        currresidual = std::max(currresidual, incprenorm_L2_ / prenorm_L2_);
        solver_params.nonlin_tolerance = ittol;
        solver_params.nonlin_residual = currresidual;
        solver_params.lin_tol_better = adaptolbetter;
      }

      if (updateprojection_)
      {
        update_krylov_space_projection();
      }

      // if Krylov space projection is used, check whether constant pressure
      // is in nullspace of sysmat_
      // !!! only done for FEM since for NURBS- and meshfree-approximations,
      //     the integration error can already disturb matrix nullspace too
      //     much for sensitive problems
      //     xwall uses non-polynomial shape functions
      if (Global::Problem::instance()->spatial_approximation_type() ==
              Core::FE::ShapeFunctionType::polynomial &&
          xwall_ == nullptr)
        check_matrix_nullspace();

      if (msht_ == Inpar::FLUID::no_meshtying)
      {
        // scale system prior to solver call
        if (fluid_infnormscaling_ != nullptr)
          fluid_infnormscaling_->scale_system(sysmat_, *residual_);

        // solve the system
        solver_params.refactor = true;
        solver_params.reset = itnum == 1;
        solver_params.projector = projector_;
        solver_->solve(sysmat_, incvel_, residual_, solver_params);

        // unscale solution
        if (fluid_infnormscaling_ != nullptr)
          fluid_infnormscaling_->unscale_solution(sysmat_, *incvel_, *residual_);
      }
      else
        meshtying_->solve_meshtying(
            *solver_, sysmat_, incvel_, residual_, *velnp_, itnum, solver_params);

      solver_->reset_tolerance();

      dtsolve_ = Teuchos::Time::wallTime() - tcpusolve;
    }

    // -------------------------------------------------------------------
    // update within iteration
    // -------------------------------------------------------------------
    iter_update(incvel_);

    // -------------------------------------------------------------------
    // convergence check
    // -------------------------------------------------------------------
    stopnonliniter = convergence_check(itnum, itmax, velrestol, velinctol, presrestol, presinctol);

    // -------------------------------------------------------------------
    // Do the Ale update conditions update
    // -------------------------------------------------------------------
    ale_update("ALEUPDATECoupling");
  }

  // -------------------------------------------------------------------
  // recompute residual (i.e., residual belonging to the final solution)
  // -------------------------------------------------------------------
  if (not inconsistent_)
  {
    assemble_mat_and_rhs();

    if (locsysman_ != nullptr)
    {
      // Transform newly built residual to local coordinate system
      // in order to later properly erase the lines containing
      // Dirichlet conditions in function convergence_check()
      locsysman_->rotate_global_to_local(*residual_, false);
    }

    // prepare meshtying system
    if (msht_ != Inpar::FLUID::no_meshtying)
      meshtying_->prepare_meshtying_system(sysmat_, *residual_, *velnp_);

    // print to screen
    convergence_check(0, itmax, velrestol, velinctol, presrestol, presinctol);
  }
}  // FluidImplicitTimeInt::Solve

/*----------------------------------------------------------------------*
 | preparatives for solver                                     vg 09/11 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::prepare_solve()
{
  // call elements to calculate system matrix and rhs and assemble
  assemble_mat_and_rhs();

  // prepare meshtying system
  if (msht_ != Inpar::FLUID::no_meshtying)
    meshtying_->prepare_meshtying(sysmat_, *residual_, *velnp_, shapederivatives_);

  // update local coordinate systems for ALE fluid case
  // (which may be time and displacement dependent)
  if ((locsysman_ != nullptr) && (alefluid_))
  {
    discret_->clear_state();
    discret_->set_state(ndsale_, "dispnp", *dispnp_);

    setup_locsys_dirichlet_bc(time_);
  }

  // apply Dirichlet boundary conditions to system of equations
  apply_dirichlet_to_system();
}  // FluidImplicitTimeInt::PrepareSolve


/*----------------------------------------------------------------------*
 | call elements to calculate system matrix/rhs and assemble   vg 02/09 |
 | overloaded in TimIntRedModels                               bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::assemble_mat_and_rhs()
{  // forcing_->PutScalar(0.0);
  dtele_ = 0.0;
  dtfilter_ = 0.0;

  // time measurement: element
  TEUCHOS_FUNC_TIME_MONITOR("      + element calls");

  // get cpu time
  const double tcpu = Teuchos::Time::wallTime();

  sysmat_->zero();

  if (shapederivatives_ != nullptr) shapederivatives_->zero();

  // set old residual to zero and add Neumann loads
  residual_->update(1.0, *neumann_loads_, 0.0);

  // add external loads
  if (external_loads_ != nullptr)
  {
    residual_->update(1.0 / residual_scaling(), *external_loads_, 1.0);
  }

  // set external volume force (required, e.g., for forced homogeneous isotropic turbulence)
  if (Teuchos::getIntegralValue<Inpar::FLUID::ForcingType>(
          params_->sublist("TURBULENCE MODEL"), "FORCING_TYPE") == Inpar::FLUID::fixed_power_input)
  {
    // calculate required forcing
    forcing_interface_->calculate_forcing(step_);
    forcing_interface_->activate_forcing(true);
  }

  if (forcing_interface_ != nullptr) forcing_interface_->update_forcing(step_);

  //----------------------------------------------------------------------
  // evaluate elements
  //----------------------------------------------------------------------

  discret_->clear_state();

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  // set action type
  eleparams.set<FLD::Action>("action", FLD::calc_fluid_systemmat_and_residual);
  eleparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype_);

  // set parameters for turbulence models
  treat_turbulence_models(eleparams);

  // set thermodynamic pressures
  // set parameters for poro
  // set parameters for HDG
  set_custom_ele_params_assemble_mat_and_rhs(eleparams);

  //----------------------------------------------------------------------
  // set general vector values needed by elements
  //----------------------------------------------------------------------
  discret_->set_state("hist", *hist_);
  discret_->set_state("veln", *veln_);
  discret_->set_state("accam", *accam_);
  discret_->set_state("scaaf", *scaaf_);
  discret_->set_state("scaam", *scaam_);
  if (alefluid_)
  {
    discret_->set_state(ndsale_, "dispnp", *dispnp_);
    discret_->set_state(ndsale_, "gridv", *gridv_);
  }

  // set scheme-specific element parameters and vector values
  set_state_tim_int();

  if (forcing_ != nullptr)
  {
    eleparams.set("forcing", true);
    if (forcing_->get_map().same_as(*discret_->dof_row_map()))
      discret_->set_state("forcing", *forcing_);
    else
      discret_->set_state(1, "forcing", *forcing_);
  }

  //----------------------------------------------------------------------
  // AVM3-based solution approach if required
  //----------------------------------------------------------------------
  if (fssgv_ != Inpar::FLUID::no_fssgv) avm3_separation();

  //----------------------------------------------------------------------
  // multifractal subgrid-scale modeling
  //----------------------------------------------------------------------
  if (turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales)
  {
    this->apply_scale_separation_for_les();
    discret_->set_state("fsscaaf", *fsscaaf_);
  }

  //----------------------------------------------------------------------
  // Add further problem dependent vectors
  //----------------------------------------------------------------------
  add_problem_dependent_vectors();

  // call standard loop over elements
  evaluate_mat_and_rhs(eleparams);
  assemble_coupling_contributions();
  clear_state_assemble_mat_and_rhs();

  //----------------------------------------------------------------------
  // add potential edge-based stabilization terms
  //----------------------------------------------------------------------
  assemble_edge_based_matand_rhs();

  //----------------------------------------------------------------------
  // application of potential nonlinear boundary conditions to system
  //----------------------------------------------------------------------
  if (nonlinearbc_)
  {
    apply_nonlinear_boundary_conditions();
  }

  // scaling to get true residual vector
  trueresidual_->update(residual_scaling(), *residual_, 0.0);

  // finalize the complete matrix
  sysmat_->complete();

  if (shapederivatives_ != nullptr)
  {
    shapederivatives_->complete();
    // apply Dirichlet conditions to a non-diagonal matrix
    // (The Dirichlet rows will become all zero, no diagonal one.)
    shapederivatives_->apply_dirichlet(*(dbcmaps_->cond_map()), false);
  }

  // end time measurement for element
  dtele_ = Teuchos::Time::wallTime() - tcpu;
}  // FluidImplicitTimeInt::assemble_mat_and_rhs

/*----------------------------------------------------------------------*
 | Call evaluate routine on elements                           bk 06/15 |
 | only for assemble_mat_and_rhs                                           |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::evaluate_mat_and_rhs(Teuchos::ParameterList& eleparams)
{
  if (off_proc_assembly_)
  {
    if (shapederivatives_ != nullptr)
      FOUR_C_THROW("The shape derivative cannot be assembled off-proc currently");
    const Core::LinAlg::Map* dofcolmap = discret_->dof_col_map();
    std::shared_ptr<Core::LinAlg::Vector<double>> residual_col =
        std::make_shared<Core::LinAlg::Vector<double>>(*dofcolmap, true);
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat =
        std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(sysmat_);
    if (sysmat == nullptr) FOUR_C_THROW("expected Sparse Matrix");
    //------------------------------------------------------------
    Core::FE::AssembleStrategy strategy(0, 0, sysmat, nullptr, residual_col, nullptr, nullptr);

    Core::Elements::LocationArray la(1);

    //------------------------------------------------------------
    // call standard loop over elements

    // loop over row elements
    const int numrowele = discret_->num_my_row_elements();

    // REMARK: in this XFEM framework the whole evaluate routine uses only row elements
    // and assembles into EpetraFECrs matrix
    // this is 4C-unusual but more efficient in all XFEM applications
    for (int i = 0; i < numrowele; ++i)
    {
      Core::Elements::Element* actele = discret_->l_row_element(i);
      // std::shared_ptr<Core::Mat::Material> mat = actele->material();
      std::shared_ptr<Core::Mat::Material> mat = actele->material();
      if (mat->material_type() == Core::Materials::m_matlist)
        FOUR_C_THROW("No matlists allowed here!!");
      // get element location vector, dirichlet flags and ownerships
      actele->location_vector(*discret_, la);
      // get dimension of element matrices and vectors
      // Reshape element matrices and vectors and init to zero
      strategy.clear_element_storage(la[0].size(), la[0].size());
      {
        int err = actele->evaluate(eleparams, *discret_, la[0].lm_, strategy.elematrix1(),
            strategy.elematrix2(), strategy.elevector1(), strategy.elevector2(),
            strategy.elevector3());

        if (err)
          FOUR_C_THROW("Proc {}: Element {} returned err={}",
              Core::Communication::my_mpi_rank(discret_->get_comm()), actele->id(), err);
      }
      std::vector<int> myowner(la[0].lmowner_.size(),
          Core::Communication::my_mpi_rank(strategy.systemvector1()->get_comm()));
      {
        // calls the Assemble function for EpetraFECrs matrices including communication of non-row
        // entries
        sysmat->fe_assemble(strategy.elematrix1(), la[0].lm_, myowner, la[0].lm_);
      }
      // introduce an vector containing the rows for that values have to be communicated
      // REMARK: when assembling row elements also non-row rows have to be communicated

      // REMARK:: call Assemble without lmowner
      // to assemble the residual_col vector on only row elements also column nodes have to be
      // assembled do not exclude non-row nodes (modify the real owner to myowner) after assembly
      // the col vector it has to be exported to the row residual_ vector using the 'Add' flag to
      // get the right value for shared nodes
      Core::LinAlg::assemble(*strategy.systemvector1(), strategy.elevector1(), la[0].lm_, myowner);
    }
    //-------------------------------------------------------------------------------
    std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
        std::make_shared<Core::LinAlg::Vector<double>>(*discret_->dof_row_map(), true);

    Core::LinAlg::Export exporter(residual_col->get_map(), tmp->get_map());
    tmp->export_to(*residual_col, exporter, Core::LinAlg::CombineMode::add);
    residual_->update(1.0, *tmp, 1.0);
  }
  else
    discret_->evaluate(eleparams, sysmat_, shapederivatives_, residual_, nullptr, nullptr);
}

/*----------------------------------------------------------------------------*
 | Evaluate mass matrix                                       mayr.mt 05/2014 |
 *----------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::evaluate_mass_matrix()
{
  massmat_->zero();

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  // action for elements
  eleparams.set<FLD::Action>("action", FLD::calc_mass_matrix);

  if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);

  discret_->evaluate(eleparams, massmat_, nullptr, nullptr, nullptr, nullptr);
  discret_->clear_state();

  // finalize the complete matrix
  massmat_->complete();
}

/*----------------------------------------------------------------------*|
 | Set Eleparams for turbulence models                          bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::treat_turbulence_models(Teuchos::ParameterList& eleparams)
{
  //----------------------------------------------------------------------
  // apply filter for turbulence models (only if required)
  //----------------------------------------------------------------------
  if (turbmodel_ == Inpar::FLUID::dynamic_smagorinsky or turbmodel_ == Inpar::FLUID::dynamic_vreman)
  {
    // compute filtered velocity
    // time measurement
    const double tcpufilter = Teuchos::Time::wallTime();
    this->apply_scale_separation_for_les();
    dtfilter_ = Teuchos::Time::wallTime() - tcpufilter;
  }

  // parameters for turbulence model
  // TODO: rename list
  eleparams.sublist("TURBULENCE MODEL") = params_->sublist("TURBULENCE MODEL");

  if (turbmodel_ == Inpar::FLUID::dynamic_vreman)
  {
    double cv = 0.0;
    cv = params_->get<double>("C_vreman");
    eleparams.set<double>("C_vreman", cv);
  }

  // set xwall params
  if (xwall_ != nullptr) xwall_->set_x_wall_params(eleparams);
}

/*----------------------------------------------------------------------*
 | application of nonlinear boundary conditions to system, such as      |
 | 1) Impedance conditions                                              |
 | 2) Neumann inflow boundary conditions                                |
 | 3) flow-dependent pressure boundary conditions                       |
 | 4) weak Dirichlet boundary conditions                                |
 | 5) mixed/hybrid Dirichlet boundary conditions                        |
 | 6) Slip Supplemental Curved Boundary conditions                      |
 | 7) Navier-slip boundary conditions                                   |
 |                                                             vg 06/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::apply_nonlinear_boundary_conditions()
{
  //----------------------------------------------------------------------
  // 1) Impedance conditions
  //----------------------------------------------------------------------
  if (isimpedancebc_)
  {
    discret_->clear_state();
    discret_->set_state("velaf", *velnp_);

    if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);

    // update residual and sysmat with impedance boundary conditions
    impedancebc_->add_impedance_bc_to_residual_and_sysmat(dta_, time_, *residual_, *sysmat_);

    discret_->clear_state();
  }

  //----------------------------------------------------------------------
  // 2) Neumann inflow boundary conditions
  //----------------------------------------------------------------------
  // check whether there are Neumann inflow boundary conditions
  std::vector<const Core::Conditions::Condition*> neumanninflow;
  discret_->get_condition("FluidNeumannInflow", neumanninflow);

  if (neumanninflow.size() != 0)
  {
    // create parameter list
    Teuchos::ParameterList neuminparams;

    // set action for elements
    neuminparams.set<FLD::BoundaryAction>("action", FLD::calc_Neumann_inflow);

    // set thermodynamic pressure
    set_custom_ele_params_apply_nonlinear_boundary_conditions(neuminparams);

    // set required state vectors
    // (no contribution due to pressure or continuity equation for Neumann inflow
    // -> no difference between af_genalpha and np_genalpha)
    discret_->clear_state();
    discret_->set_state("scaaf", *scaaf_);
    set_state_tim_int();
    if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);

    // evaluate all Neumann inflow boundary conditions
    discret_->evaluate_condition(
        neuminparams, sysmat_, nullptr, residual_, nullptr, nullptr, "FluidNeumannInflow");

    // clear state
    discret_->clear_state();
  }

  //----------------------------------------------------------------------
  // 3) flow-dependent pressure boundary conditions
  //    (either based on (out)flow rate or on (out)flow volume (e.g.,
  //     for air cushion outside of boundary))
  //----------------------------------------------------------------------
  // check whether there are flow-dependent pressure boundary conditions
  std::vector<const Core::Conditions::Condition*> flowdeppressureline;
  discret_->get_condition("LineFlowDepPressure", flowdeppressureline);
  std::vector<const Core::Conditions::Condition*> flowdeppressuresurf;
  discret_->get_condition("SurfaceFlowDepPressure", flowdeppressuresurf);

  if (flowdeppressureline.size() != 0 or flowdeppressuresurf.size() != 0)
  {
    // get a vector layout from the discretization to construct matching
    // vectors and matrices local <-> global dof numbering
    const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

    // decide on whether it is a line or a surface condition and
    // set condition name accordingly
    std::string fdpcondname;
    if (flowdeppressureline.size() != 0)
      fdpcondname = "LineFlowDepPressure";
    else if (flowdeppressuresurf.size() != 0)
      fdpcondname = "SurfaceFlowDepPressure";
    else
    {
      FOUR_C_THROW(
          "Line and surface flow-dependent pressure boundary conditions simultaneously "
          "prescribed!");
    }

    // get condition vector
    std::vector<const Core::Conditions::Condition*> fdpcond;
    discret_->get_condition(fdpcondname, fdpcond);

    // define vectors for flow rate and volume for actual evaluation of boundary
    // conditions according to time-integration scheme and potential relaxation
    // within nonlinear iteration loop
    // (relaxation parameter 1.0, for the time being, that is, no relaxation)
    std::vector<double> flowraterel(fdpcond.size(), 0.0);
    std::vector<double> flowvolumerel(fdpcond.size(), 0.0);
    const double relaxpara = 1.0;

    double timefac = 1.0;
    timefac = set_time_fac();

    // assign ID to all conditions
    for (int fdpcondid = 0; fdpcondid < (int)fdpcond.size(); fdpcondid++)
    {
      // check for already existing ID and add ID
      const auto* fdpcondid_from_container =
          fdpcond[fdpcondid]->parameters().get_if<int>("ConditionID");
      if (fdpcondid_from_container)
      {
        if ((*fdpcondid_from_container) != fdpcondid)
          FOUR_C_THROW("Flow-dependent pressure condition {} has non-matching ID", fdpcondname);
      }
      else
      {
        // TODO this hacks the condition parameters
        const_cast<Core::Conditions::Condition*>(fdpcond[fdpcondid])
            ->parameters()
            .add("ConditionID", fdpcondid);
      }
    }

    // create or append to output file
    if (myrank_ == 0)
    {
      const std::string fname1 =
          Global::Problem::instance()->output_control_file()->file_name() + ".fdpressure";

      std::ofstream f1;

      // create file for output in first time step or append to existing file
      // in subsequent time steps
      if (step_ <= 1)
      {
        f1.open(fname1.c_str(), std::fstream::trunc);
        f1 << "#| Step | Time |";
        for (int fdpcondid = 0; fdpcondid < (int)fdpcond.size(); fdpcondid++)
        {
          f1 << " Flow rate " << fdpcondid << " | Flow volume " << fdpcondid << " | Mean pressure "
             << fdpcondid << " |";
        }
        f1 << "\n";
      }
      else
        f1.open(fname1.c_str(), std::fstream::ate | std::fstream::app);

      // write step number and time
      f1 << step_ << " " << time_ << " ";
    }

    //----------------------------------------------------------------------
    // compute
    // a) flow rate (current value and value used for evaluation of bc)
    // b) flow volume (current value and value used for evaluation of bc)
    // c) surface area,
    // d) pressure integral, and
    // e) mean pressure
    // for each flow-dependent pressure boundary condition
    //----------------------------------------------------------------------
    for (int fdpcondid = 0; fdpcondid < (int)fdpcond.size(); fdpcondid++)
    {
      // create parameter list
      Teuchos::ParameterList flowdeppressureparams;

      // initialization of values at first time step
      if (step_ <= 1)
      {
        flowraten_[fdpcondid] = 0.0;
        flowratenm_[fdpcondid] = 0.0;
        flowvolumen_[fdpcondid] = 0.0;
        flowvolumenm_[fdpcondid] = 0.0;
      }

      //--------------------------------------------------------------------
      // a) flow rate
      //--------------------------------------------------------------------
      // set action for elements
      flowdeppressureparams.set<FLD::BoundaryAction>("action", FLD::calc_flowrate);

      // create vector and initialize with zeros
      std::shared_ptr<Core::LinAlg::Vector<double>> flowrates =
          std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

      // set required state vectors
      discret_->clear_state();
      set_state_tim_int();
      if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);

      // evaluate flow rate
      discret_->evaluate_condition(flowdeppressureparams, flowrates, fdpcondname, fdpcondid);

      // sum up local flow rate on this processor
      double local_flowrate = 0.0;
      for (int i = 0; i < dofrowmap->num_my_elements(); i++)
      {
        local_flowrate += (flowrates->local_values_as_span()[i]);
      }

      // sum up global flow rate over all processors and set to global value
      double flowrate = 0.0;
      flowrate = Core::Communication::sum_all(local_flowrate, dofrowmap->get_comm());

      // set current flow rate
      flowratenp_[fdpcondid] = flowrate;

      // compute flow rate used for evaluation of boundary condition below
      flowraterel[fdpcondid] = (1.0 - timefac) * flowraten_[fdpcondid] +
                               timefac * ((1.0 - relaxpara) * flowratenpi_[fdpcondid] +
                                             relaxpara * flowratenp_[fdpcondid]);

      // clear state
      discret_->clear_state();

      //--------------------------------------------------------------------
      // b) flow volume
      //--------------------------------------------------------------------
      // compute current flow volume as integral of flow rate according to
      // trapezoidal rule
      flowvolumenp_[fdpcondid] =
          flowvolumen_[fdpcondid] + 0.5 * dta_ * (flowratenp_[fdpcondid] + flowraten_[fdpcondid]);

      // set current flow volume to zero if value smaller than zero,
      // meaning that no flow volume may be sucked in from outside
      if (flowvolumenp_[fdpcondid] < 0.0) flowvolumenp_[fdpcondid] = 0.0;

      // compute flow volume used for evaluation of boundary condition below
      flowvolumerel[fdpcondid] = (1.0 - timefac) * flowvolumen_[fdpcondid] +
                                 timefac * ((1.0 - relaxpara) * flowvolumenpi_[fdpcondid] +
                                               relaxpara * flowvolumenp_[fdpcondid]);

      //--------------------------------------------------------------------
      // c) surface area
      //--------------------------------------------------------------------
      // set action and parameter for elements
      flowdeppressureparams.set<FLD::BoundaryAction>("action", FLD::calc_area);
      flowdeppressureparams.set<double>("area", 0.0);

      // set required state vectors
      if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);

      // evaluate surface area
      discret_->evaluate_condition(flowdeppressureparams, fdpcondname, fdpcondid);

      // sum up local surface area on this processor
      double localarea = flowdeppressureparams.get<double>("area");

      // sum up global surface area over all processors
      double area = 0.0;
      area = Core::Communication::sum_all(localarea, discret_->get_comm());

      // clear state
      discret_->clear_state();

      //--------------------------------------------------------------------
      // d) pressure integral
      //--------------------------------------------------------------------
      // set action for elements
      flowdeppressureparams.set<FLD::BoundaryAction>("action", FLD::calc_pressure_bou_int);
      flowdeppressureparams.set<double>("pressure boundary integral", 0.0);

      // set required state vectors
      discret_->clear_state();
      set_state_tim_int();
      if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);

      // evaluate pressure integral
      discret_->evaluate_condition(flowdeppressureparams, fdpcondname, fdpcondid);

      // sum up local pressure integral on this processor
      double localpressint = flowdeppressureparams.get<double>("pressure boundary integral");

      // sum up global pressure integral over all processors
      double pressint = 0.0;
      pressint = Core::Communication::sum_all(localpressint, discret_->get_comm());

      // clear state
      discret_->clear_state();

      //--------------------------------------------------------------------
      // e) mean pressure
      //--------------------------------------------------------------------
      const double meanpressure = pressint / area;

      // append values to output file
      if (myrank_ == 0)
      {
        const std::string fname1 =
            Global::Problem::instance()->output_control_file()->file_name() + ".fdpressure";

        std::ofstream f1;
        f1.open(fname1.c_str(), std::fstream::ate | std::fstream::app);

        f1 << flowratenp_[fdpcondid] << " " << flowvolumenp_[fdpcondid] << " " << meanpressure
           << "   ";
      }
    }

    // append values to output file
    if (myrank_ == 0)
    {
      const std::string fname1 =
          Global::Problem::instance()->output_control_file()->file_name() + ".fdpressure";

      std::ofstream f1;
      f1.open(fname1.c_str(), std::fstream::ate | std::fstream::app);

      f1 << "\n";
      f1.flush();
      f1.close();
    }

    //----------------------------------------------------------------------
    // evaluate flow-dependent pressure boundary conditions
    // (proceeds in separate loop to enable, e.g., implementation of
    //  flow-rate sums of more than one condition in between)
    //----------------------------------------------------------------------
    for (int fdpcondid = 0; fdpcondid < (int)fdpcond.size(); fdpcondid++)
    {
      // create parameter list
      Teuchos::ParameterList flowdeppressureparams;

      // set action for elements
      flowdeppressureparams.set<FLD::BoundaryAction>("action", FLD::flow_dep_pressure_bc);

      // set thermodynamic pressure
      set_custom_ele_params_apply_nonlinear_boundary_conditions(flowdeppressureparams);

      // set required state vectors
      // (no contribution due to pressure or continuity equation for Neumann inflow
      // -> no difference between af_genalpha and np_genalpha)
      discret_->clear_state();
      discret_->set_state("scaaf", *scaaf_);
      set_state_tim_int();
      if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);

      // set values for elements
      const int fdp_cond_id = fdpcond[fdpcondid]->parameters().get<int>("ConditionID");
      flowdeppressureparams.set<double>("flow rate", flowraterel[fdp_cond_id]);
      flowdeppressureparams.set<double>("flow volume", flowvolumerel[fdp_cond_id]);

      // evaluate all flow-dependent pressure boundary conditions
      discret_->evaluate_condition(flowdeppressureparams, sysmat_, nullptr, residual_, nullptr,
          nullptr, fdpcondname, fdpcondid);

      // clear state
      discret_->clear_state();

      // update iteration values
      flowratenpi_[fdpcondid] = flowratenp_[fdpcondid];
      flowvolumenpi_[fdpcondid] = flowvolumenp_[fdpcondid];
    }
  }

  //----------------------------------------------------------------------
  // 4) weak Dirichlet boundary conditions
  //----------------------------------------------------------------------
  // check whether there are weak Dirichlet boundary conditions
  std::vector<const Core::Conditions::Condition*> weakdbcline;
  discret_->get_condition("LineWeakDirichlet", weakdbcline);
  std::vector<const Core::Conditions::Condition*> weakdbcsurf;
  discret_->get_condition("SurfaceWeakDirichlet", weakdbcsurf);

  if (weakdbcline.size() != 0 or weakdbcsurf.size() != 0)
  {
    // create parameter list
    Teuchos::ParameterList weakdbcparams;

    // set action for elements
    weakdbcparams.set<FLD::BoundaryAction>("action", FLD::enforce_weak_dbc);

    // set required state vectors
    set_state_tim_int();
    if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);

    // evaluate all line weak Dirichlet boundary conditions
    discret_->evaluate_condition(
        weakdbcparams, sysmat_, nullptr, residual_, nullptr, nullptr, "LineWeakDirichlet");

    // evaluate all surface weak Dirichlet boundary conditions
    discret_->evaluate_condition(
        weakdbcparams, sysmat_, nullptr, residual_, nullptr, nullptr, "SurfaceWeakDirichlet");

    // clear state
    discret_->clear_state();
  }

  //----------------------------------------------------------------------
  // 5) mixed/hybrid Dirichlet boundary conditions
  //----------------------------------------------------------------------
  // check whether there are mixed/hybrid Dirichlet boundary conditions
  std::vector<const Core::Conditions::Condition*> mhdbcline;
  discret_->get_condition("LineMixHybDirichlet", mhdbcline);
  std::vector<const Core::Conditions::Condition*> mhdbcsurf;
  discret_->get_condition("SurfaceMixHybDirichlet", mhdbcsurf);

  if (mhdbcline.size() != 0 or mhdbcsurf.size() != 0)
  {
    // create parameter list
    Teuchos::ParameterList mhdbcparams;

    // set action for elements
    mhdbcparams.set<FLD::BoundaryAction>("action", FLD::mixed_hybrid_dbc);

    // set required state vectors
    set_state_tim_int();
    if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);

    // evaluate all line mixed/hybrid Dirichlet boundary conditions
    discret_->evaluate_condition(
        mhdbcparams, sysmat_, nullptr, residual_, nullptr, nullptr, "LineMixHybDirichlet");

    // evaluate all surface mixed/hybrid Dirichlet boundary conditions
    discret_->evaluate_condition(
        mhdbcparams, sysmat_, nullptr, residual_, nullptr, nullptr, "SurfaceMixHybDirichlet");

    // clear state
    discret_->clear_state();
  }

  //------------------------------------------------------------------------
  // 6) Slip Supplemental Curved Boundary conditions            [hahn 07/14]
  //    (Boundary condition used for counteracting spurious velocities at
  //     curved boundaries with slip-conditions. For details see Behr M.,
  //     2003, "On the Application of Slip Boundary Condition on Curved
  //     Boundaries" and Coppola-Owen H. & Codina R., 2011, "A free surface
  //     finite element model for low Froude number mould filling problems
  //     on fixed meshes".)
  //------------------------------------------------------------------------

  // check whether there are Slip Supplemental Curved Boundary conditions
  std::vector<const Core::Conditions::Condition*> slipsuppline;
  discret_->get_condition("LineSlipSupp", slipsuppline);
  std::vector<const Core::Conditions::Condition*> slipsuppsurf;
  discret_->get_condition("SurfaceSlipSupp", slipsuppsurf);

  if (slipsuppline.size() != 0 or slipsuppsurf.size() != 0)
  {
    // get a vector layout from the discretization to construct matching
    // vectors and matrices local <-> global dof numbering
    const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

    // initialize global slip bc normal traction variable
    slip_bc_normal_tractions_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

    // decide on whether it is a line or a surface condition and set condition
    // name accordingly. Both types simultaneously is not supported
    if ((slipsuppline.size() != 0) && (slipsuppsurf.size() != 0))
    {
      FOUR_C_THROW(
          "Line and surface slip supplemental curved boundary conditions simultaneously "
          "prescribed!");
    }

    std::string sscbcondname;
    if (slipsuppline.size() != 0)
      sscbcondname = "LineSlipSupp";
    else if (slipsuppsurf.size() != 0)
      sscbcondname = "SurfaceSlipSupp";

    // get condition vector
    std::vector<const Core::Conditions::Condition*> sscbcond;
    discret_->get_condition(sscbcondname, sscbcond);

    // assign ID to all conditions
    for (int sscbcondid = 0; sscbcondid < (int)sscbcond.size(); sscbcondid++)
    {
      // check for already existing ID and add ID
      const int sscbcondid_from_container =
          sscbcond[sscbcondid]->parameters().get<int>("ConditionID");
      if (sscbcondid_from_container)
      {
        if (sscbcondid_from_container != sscbcondid)
          FOUR_C_THROW(
              "Slip Supplemental Curved Boundary condition {} has non-matching ID", sscbcondname);
      }
      else
      {
        // TODO this hacks the condition parameters
        const_cast<Core::Conditions::Condition*>(sscbcond[sscbcondid])
            ->parameters()
            .add("ConditionID", sscbcondid);
      }
    }

    //----------------------------------------------------------------------
    // evaluate slip supplemental curved boundary conditions
    //----------------------------------------------------------------------

    for (int sscbcondid = 0; sscbcondid < (int)sscbcond.size(); sscbcondid++)
    {
      // Evaluate condition
      // ******************************************************************
      // create parameter list
      Teuchos::ParameterList slipsuppparams;

      // set action for elements
      slipsuppparams.set<FLD::BoundaryAction>("action", FLD::slip_supp_bc);

      // set required state vectors
      set_state_tim_int();
      if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);
      // discret_->set_state("nodenormal",nodeNormal);

      // temporary variable holding the scaled residual contribution
      std::shared_ptr<Core::LinAlg::Vector<double>> slip_bc_normal_tractions_scaled;
      slip_bc_normal_tractions_scaled =
          std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

      // evaluate all slip supplemental curved boundary conditions
      discret_->evaluate_condition(slipsuppparams, sysmat_, nullptr,
          slip_bc_normal_tractions_scaled, nullptr, nullptr, sscbcondname, sscbcondid);

      // Update residual vector
      residual_->update(1.0, *slip_bc_normal_tractions_scaled, 1.0);

      // Add to tractions vector
      slip_bc_normal_tractions_->update(
          (-1) * residual_scaling(), *slip_bc_normal_tractions_scaled, 1.0);

      // clear state
      discret_->clear_state();
    }
  }

  //------------------------------------------------------------------------
  // 7) Navier-slip boundary conditions                         [hahn 03/14]
  //    At the boundary, apply a shear stress which is proportional to the
  //    tangential/bi-tangential velocity. In 4C, this is achieved by
  //    applying h = sigma*n = -beta*u under the condition that u*n=0 has
  //    been set as Dirichlet BC! For details on the Navier slip condition
  //    please refer to e.g. Behr M., 2003, "On the Application of Slip
  //    Boundary Condition on Curved Boundaries".
  //------------------------------------------------------------------------

  // check whether there are navier-slip boundary conditions
  std::vector<const Core::Conditions::Condition*> navierslipline;
  discret_->get_condition("LineNavierSlip", navierslipline);
  std::vector<const Core::Conditions::Condition*> navierslipsurf;
  discret_->get_condition("SurfNavierSlip", navierslipsurf);

  if (navierslipline.size() != 0 or navierslipsurf.size() != 0)
  {
    // decide on whether it is a line or a surface condition and set condition
    // name accordingly. Both types simultaneously is not supported
    if ((navierslipline.size() != 0) && (navierslipsurf.size() != 0))
      FOUR_C_THROW("Line and surface Navier slip boundary conditions simultaneously prescribed!");

    std::string nscondname;
    if (navierslipline.size() != 0)
      nscondname = "LineNavierSlip";
    else if (navierslipsurf.size() != 0)
      nscondname = "SurfNavierSlip";

    // get condition vector
    std::vector<const Core::Conditions::Condition*> nscond;
    discret_->get_condition(nscondname, nscond);

    // assign ID to all conditions
    for (int nscondid = 0; nscondid < (int)nscond.size(); nscondid++)
    {
      // check for already existing ID and add ID
      const int nscondid_from_container = nscond[nscondid]->parameters().get<int>("ConditionID");
      if (nscondid_from_container)
      {
        if (nscondid_from_container != nscondid)
          FOUR_C_THROW("Navier slip boundary condition {} has non-matching ID", nscondname);
      }
      else
      {
        // TODO this hacks the condition parameters
        const_cast<Core::Conditions::Condition*>(nscond[nscondid])
            ->parameters()
            .add("ConditionID", nscondid);
      }
    }

    //----------------------------------------------------------------------
    // evaluate navier slip boundary conditions
    //----------------------------------------------------------------------

    for (int nscondid = 0; nscondid < (int)nscond.size(); nscondid++)
    {
      // create parameter list
      Teuchos::ParameterList navierslipparams;

      // set action for elements
      navierslipparams.set<FLD::BoundaryAction>("action", FLD::navier_slip_bc);

      // set required state vectors
      set_state_tim_int();
      if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);

      // set slip coefficient
      const Core::Conditions::Condition* currnavierslip = nscond[nscondid];
      const double beta = currnavierslip->parameters().get<double>("SLIPCOEFFICIENT");
      navierslipparams.set<double>("beta", beta);

      // evaluate navier slip boundary condition
      discret_->evaluate_condition(
          navierslipparams, sysmat_, nullptr, residual_, nullptr, nullptr, nscondname, nscondid);

      // clear state
      discret_->clear_state();
    }
  }
}


/*----------------------------------------------------------------------*
 | add potential edge-based stabilization terms         rasthofer 06/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::assemble_edge_based_matand_rhs()
{
  // add edged-based stabilization, if selected
  if (Teuchos::getIntegralValue<Inpar::FLUID::StabType>(
          params_->sublist("RESIDUAL-BASED STABILIZATION"), "STABTYPE") ==
      Inpar::FLUID::StabType::stabtype_edgebased)
  {
    // set the only required state vectors
    set_state_tim_int();

    if (alefluid_)
    {
      discret_->set_state(ndsale_, "dispnp", *dispnp_);
      discret_->set_state(ndsale_, "gridv", *gridv_);
    }

    Teuchos::ParameterList params;
    if (params_->sublist("RESIDUAL-BASED STABILIZATION").isParameter("POROUS-FLOW STABILIZATION"))
      params.set<Inpar::XFEM::FaceType>("FaceType", Inpar::XFEM::face_type_porof);
    // set action for elements
    params.set<FLD::IntFaceAction>("action", FLD::EOS_and_GhostPenalty_stabilization);
    evaluate_fluid_edge_based(sysmat_, *residual_, params);

    discret_->clear_state();
  }
}


void FLD::FluidImplicitTimeInt::evaluate_fluid_edge_based(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1,
    Core::LinAlg::Vector<double>& systemvector1, Teuchos::ParameterList edgebasedparams)
{
  TEUCHOS_FUNC_TIME_MONITOR("FLD::FluidImplicitTimeInt::EvaluateEdgeBased");


  std::shared_ptr<Core::LinAlg::Vector<double>> residual_col =
      std::make_shared<Core::LinAlg::Vector<double>>(*(facediscret_->dof_col_map()), true);

  std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_linalg;
  if (systemmatrix1 != nullptr)
  {
    sysmat_linalg = std::make_shared<Core::LinAlg::SparseMatrix>(
        systemmatrix1->domain_map(), 256, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
  }
  else
    FOUR_C_THROW("sysmat is nullptr!");

  const int numrowintfaces = facediscret_->num_my_row_faces();

  for (int i = 0; i < numrowintfaces; ++i)
  {
    Core::Elements::Element* actface = facediscret_->l_row_face(i);

    {
      Discret::Elements::FluidIntFace* ele =
          dynamic_cast<Discret::Elements::FluidIntFace*>(actface);
      if (ele == nullptr) FOUR_C_THROW("expect FluidIntFace element");

      // get the parent fluid elements
      Core::Elements::Element* p_master = ele->parent_master_element();
      Core::Elements::Element* p_slave = ele->parent_slave_element();

      size_t p_master_numnode = p_master->num_node();
      size_t p_slave_numnode = p_slave->num_node();


      std::vector<int> nds_master;
      nds_master.reserve(p_master_numnode);

      std::vector<int> nds_slave;
      nds_slave.reserve(p_slave_numnode);

      {
        TEUCHOS_FUNC_TIME_MONITOR("XFEM::Edgestab EOS: create nds");

        for (size_t i = 0; i < p_master_numnode; i++) nds_master.push_back(0);

        for (size_t i = 0; i < p_slave_numnode; i++) nds_slave.push_back(0);
      }

      // call the internal faces stabilization routine for the current side/surface
      TEUCHOS_FUNC_TIME_MONITOR("XFEM::Edgestab EOS: assemble_edge_stab_ghost_penalty");



      // Set master ele to the Material for evaluation.
      std::shared_ptr<Core::Mat::Material> material = p_master->material();

#ifdef FOUR_C_ENABLE_ASSERTIONS
      // Set master ele to the Material for slave.
      std::shared_ptr<Core::Mat::Material> material_s = p_slave->material();

      // Test whether the materials for the parent and slave element are the same.
      if (material->material_type() != material_s->material_type())
        FOUR_C_THROW(" not the same material for master and slave parent element");
#endif

      // call the edge-based assemble and evaluate routine
      Discret::Elements::FluidIntFaceImplInterface::impl(ele)
          ->assemble_internal_faces_using_neighbor_data(ele, material, nds_master, nds_slave,
              Inpar::XFEM::face_type_std, edgebasedparams, *facediscret_, sysmat_linalg,
              residual_col);
    }
  }

  sysmat_linalg->complete();

  // if the fluid system matrix is of type BlockSparseMatrix, we cannot add
  // and have to split sysmat_linalg - therefore, we try to cast the fluid system matrix!
  // we need RTTI here - the type-IDs are compared and the dynamic cast is only performed,
  // if we really have an underlying BlockSparseMatrix; hopefully that saves some
  // runtime.. (kruse, 09/14)
  if (Core::Utils::get_dynamic_type(*systemmatrix1) ==
      Core::Utils::get_dynamic_type(*sysmat_linalg))
  {
    (systemmatrix1)->add(*sysmat_linalg, false, 1.0, 1.0);
  }
  else
  {
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_sysmat =
        std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(systemmatrix1);
    if (block_sysmat == nullptr)
      FOUR_C_THROW("Expected fluid system matrix as BlockSparseMatrix. Failed to cast to it.");
    std::shared_ptr<Core::LinAlg::SparseMatrix> f00, f01, f10, f11;
    std::shared_ptr<Core::LinAlg::Map> domainmap_00 =
        std::make_shared<Core::LinAlg::Map>(block_sysmat->domain_map(0));
    std::shared_ptr<Core::LinAlg::Map> domainmap_11 =
        std::make_shared<Core::LinAlg::Map>(block_sysmat->domain_map(1));

    // Split sparse system matrix into blocks according to the given maps
    Core::LinAlg::split_matrix2x2(
        sysmat_linalg, domainmap_00, domainmap_11, domainmap_00, domainmap_11, f00, f01, f10, f11);
    // add the blocks subsequently
    Core::LinAlg::matrix_add(*f00, false, 1.0, block_sysmat->matrix(0, 0), 1.0);
    Core::LinAlg::matrix_add(*f01, false, 1.0, block_sysmat->matrix(0, 1), 1.0);
    Core::LinAlg::matrix_add(*f10, false, 1.0, block_sysmat->matrix(1, 0), 1.0);
    Core::LinAlg::matrix_add(*f11, false, 1.0, block_sysmat->matrix(1, 1), 1.0);
  }

  //------------------------------------------------------------
  // need to export residual_col to systemvector1 (residual_)
  Core::LinAlg::Vector<double> res_tmp(systemvector1.get_map(), false);
  Core::LinAlg::Export exporter(residual_col->get_map(), res_tmp.get_map());
  res_tmp.export_to(*residual_col, exporter, Core::LinAlg::CombineMode::add);
  systemvector1.update(1.0, res_tmp, 1.0);
}

/*----------------------------------------------------------------------*
 | application of Dirichlet boundary conditions to system      vg 09/11 |
 | overloaded in TimIntRedModels                              bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::apply_dirichlet_to_system()
{
  // -------------------------------------------------------------------
  // apply Dirichlet boundary conditions to system of equations:
  // - Residual displacements are supposed to be zero for resp. dofs.
  // - Time for applying Dirichlet boundary conditions is measured.
  // -------------------------------------------------------------------
  incvel_->put_scalar(0.0);

  if (locsysman_ != nullptr)
  {
    TEUCHOS_FUNC_TIME_MONITOR("      + apply DBC");
    // Transform system matrix and rhs to local co-ordinate systems
    locsysman_->rotate_global_to_local(system_matrix(), *residual_);

    Core::LinAlg::apply_dirichlet_to_system(
        *Core::LinAlg::cast_to_sparse_matrix_and_check_success(sysmat_), *incvel_, *residual_,
        *locsysman_->trafo(), *zeros_, *(dbcmaps_->cond_map()));
  }
  else
  {
    Core::LinAlg::apply_dirichlet_to_system(
        *sysmat_, *incvel_, *residual_, *zeros_, *(dbcmaps_->cond_map()));
  }

}  // FluidImplicitTimeInt::apply_dirichlet_to_system


void FLD::FluidImplicitTimeInt::init_krylov_space_projection()
{
  // get condition "KrylovSpaceProjection" from discretization
  std::vector<const Core::Conditions::Condition*> KSPcond;
  discret_->get_condition("KrylovSpaceProjection", KSPcond);
  int numcond = KSPcond.size();
  int numfluid = 0;

  const Core::Conditions::Condition* kspcond = nullptr;
  // check if for fluid Krylov projection is required
  for (int icond = 0; icond < numcond; icond++)
  {
    const auto& name = KSPcond[icond]->parameters().get<std::string>("DIS");
    if (name == "fluid")
    {
      numfluid++;
      kspcond = KSPcond[icond];
    }
  }

  // initialize variables for Krylov projection if necessary
  if (numfluid == 1)
  {
    setup_krylov_space_projection(kspcond);
    if (myrank_ == 0) std::cout << "\nSetup of KrylovSpaceProjection in fluid field\n" << std::endl;
  }
  else if (numfluid == 0)
  {
    updateprojection_ = false;
    projector_ = nullptr;
  }
  else
    FOUR_C_THROW("Received more than one KrylovSpaceCondition for fluid field");
}

/*--------------------------------------------------------------------------*
 | setup Krylov projector including first fill                    nis Feb13 |
 *--------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::setup_krylov_space_projection(
    const Core::Conditions::Condition* kspcond)
{
  // confirm that mode flags are number of nodal dofs
  const int nummodes = kspcond->parameters().get<int>("NUMMODES");
  if (nummodes != (numdim_ + 1))
    FOUR_C_THROW("Expecting numdim_+1 modes in Krylov projection definition. Check input file!");

  // get vector of mode flags as given in input file
  const auto& modeflags = kspcond->parameters().get<std::vector<int>>("ONOFF");

  // confirm that only the pressure mode is selected for Krylov projection in input file
  for (int rr = 0; rr < numdim_; ++rr)
  {
    if ((modeflags[rr]) != 0)
    {
      FOUR_C_THROW("Expecting only an undetermined pressure. Check input file!");
    }
  }
  if ((modeflags[numdim_]) != 1)
    FOUR_C_THROW("Expecting an undetermined pressure. Check input file!");
  std::vector<int> activemodeids(1, numdim_);

  // allocate kspsplitter_
  kspsplitter_ = std::make_shared<FLD::Utils::KSPMapExtractor>();
  // create map of nodes involved in Krylov projection

  kspsplitter_->setup(*discret_);

  // get from input file definition how weights are to be computed
  const auto* weighttype = &kspcond->parameters().get<std::string>("WEIGHTVECDEF");

  // set flag for projection update true only if ALE and integral weights
  if (alefluid_ and (*weighttype == "integration")) updateprojection_ = true;

  auto map = discret_->dof_row_map();
  projector_ = std::make_shared<Core::LinAlg::KrylovProjector>(activemodeids, weighttype, map);

  // update the projector
  update_krylov_space_projection();

}  // FLD::FluidImplicitTimeInt::setup_krylov_space_projection


/*--------------------------------------------------------------------------*
 | update projection vectors w_ and c_ for Krylov projection      nis Feb13 |
 *--------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::update_krylov_space_projection()
{
  // get std::shared_ptr to kernel vector of projector
  std::shared_ptr<Core::LinAlg::MultiVector<double>> c = projector_->get_non_const_kernel();
  // scope to modify c
  {
    auto& c0 = c->get_vector(0);
    c0.put_scalar(0.0);
    // extract vector of pressure-dofs
    std::shared_ptr<Core::LinAlg::Vector<double>> presmode =
        velpressplitter_->extract_cond_vector(c0);

    const std::string* weighttype = projector_->weight_type();
    std::shared_ptr<Core::LinAlg::Vector<double>> w0_update = nullptr;
    // compute w_ as defined in input file
    if (*weighttype == "pointvalues")
    {
      /*
      // export to vector to normalize against
      // Note that in the case of definition pointvalue based,
      // the average pressure will vanish in a pointwise sense
      //
      //    +---+
      //     \
      //      +   p_i  = 0
      //     /
      //    +---+
      //
      // (everything is done below)
      */
    }
    else if (*weighttype == "integration")
    {
      // get std::shared_ptr to weight vector of projector
      std::shared_ptr<Core::LinAlg::MultiVector<double>> w = projector_->get_non_const_weights();
      auto& w0 = w->get_vector(0);
      w0.put_scalar(0.0);

      // create parameter list for condition evaluate and ...
      Teuchos::ParameterList mode_params;
      // ... set action for elements to integration of shape functions
      mode_params.set<FLD::Action>("action", FLD::integrate_shape);

      if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);

      if (xwall_ != nullptr) xwall_->set_x_wall_params(mode_params);

      /*
      // evaluate KrylovSpaceProjection condition in order to get
      // integrated nodal basis functions w_
      // Note that in the case of definition integration based,
      // the average pressure will vanish in an integral sense
      //
      //                    /              /                      /
      //   /    \          |              |  /          \        |  /    \
      //  | w_*p | = p_i * | N_i(x) dx =  | | N_i(x)*p_i | dx =  | | p(x) | dx = 0
      //   \    /          |              |  \          /        |  \    /
      //                   /              /                      /
      */

      // compute w_ by evaluating the integrals of all pressure basis functions
      discret_->evaluate_condition(mode_params, nullptr, nullptr,
          Core::Utils::shared_ptr_from_ref(w0), nullptr, nullptr, "KrylovSpaceProjection");

      discret_->clear_state();

      // adapt weight vector according to meshtying case
      if (msht_ != Inpar::FLUID::no_meshtying)
        w0_update = meshtying_->adapt_krylov_projector(Core::Utils::shared_ptr_from_ref(w0));
    }
    else
    {
      FOUR_C_THROW("unknown definition of weight vector w for restriction of Krylov space");
    }

    // construct c by setting all pressure values to 1.0 and export to c
    presmode->put_scalar(1.0);
    std::shared_ptr<Core::LinAlg::Vector<double>> tmpc =
        std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);
    Core::LinAlg::export_to(*presmode, *tmpc);
    std::shared_ptr<Core::LinAlg::Vector<double>> tmpkspc =
        kspsplitter_->extract_ksp_cond_vector(*tmpc);
    Core::LinAlg::export_to(*tmpkspc, c0);
    // adapt kernel vector according to meshtying case

    if (msht_ != Inpar::FLUID::no_meshtying)
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> c0_update;
      if (*weighttype != "integration")
        FOUR_C_THROW("Fluidmeshtying supports only an integration - like Krylov projector");
      c0_update = meshtying_->adapt_krylov_projector(Core::Utils::shared_ptr_from_ref(c0));
      if (msht_ == Inpar::FLUID::condensed_bmat || msht_ == Inpar::FLUID::condensed_bmat_merged)
      {
        projector_->set_cw(*c0_update, *w0_update, meshtying_->get_merged_map());
      }
      else
      {
        projector_->set_cw(*c0_update, *w0_update);
      }
    }
  }

  // fillcomplete the projector to compute (w^T c)^(-1)
  projector_->fill_complete();

}  // FluidImplicitTimeInt::update_krylov_space_projection


/*--------------------------------------------------------------------------*
 | check if constant pressure mode is in kernel of sysmat_     nissen Jan13 |
 *--------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::check_matrix_nullspace()
{
  // Note: this check is expensive and should only be used in the debug mode
  if (projector_ != nullptr)
  {
    std::shared_ptr<Core::LinAlg::MultiVector<double>> c = projector_->get_non_const_kernel();
    projector_->fill_complete();
    int nsdim = c->num_vectors();
    if (nsdim != 1) FOUR_C_THROW("Only one mode, namely the constant pressure mode, expected.");

    Core::LinAlg::Vector<double> result(c->get_map(), false);

    sysmat_->Apply(c->get_epetra_multi_vector(), result);

    double norm = 1e9;

    result.norm_2(&norm);

    if (norm > 1e-12)
    {
      std::cout << "#####################################################" << std::endl;
      std::cout << "Nullspace check for sysmat_ failed!                  " << std::endl;
      std::cout << "This might be caused by:                             " << std::endl;
      std::cout << " - you don't have pure Dirichlet boundary conditions " << std::endl;
      std::cout << "   or pbcs. pressure level is fixed. -> check datfile" << std::endl;
      std::cout << " - you don't integrate pressure dofs accurately      " << std::endl;
      std::cout << "   enough for sysmat_. constant pressure is not in   " << std::endl;
      std::cout << "   kernel of sysmat_. -> use more gauss points (often" << std::endl;
      std::cout << "   problem with nurbs)                               " << std::endl;
      std::cout << " - unlikely but not impossible: nullspace vector is  " << std::endl;
      std::cout << "   not the constant pressure mode (not totally clear " << std::endl;
      std::cout << "   for xfem, yet). In this case sysmat_ could be     " << std::endl;
      std::cout << "   correct. -> adapt nullspace vector                " << std::endl;
      std::cout << "#####################################################" << std::endl;
      FOUR_C_THROW("Nullspace check for sysmat_ failed, Ac returned {:12.5e}", norm);
    }
  }
}


/*----------------------------------------------------------------------*
 | update within iteration                                     vg 09/11 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::iter_update(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> increment)
{
  // store incremental vector to be available for convergence check
  // if incremental vector is received from outside for coupled problems
  incvel_->update(1.0, *increment, 0.0);

  // update velocity and pressure values by adding increments
  velnp_->update(1.0, *increment, 1.0);

  // -------------------------------------------------------------------
  // For af-generalized-alpha: update accelerations
  // Furthermore, calculate velocities, pressures, scalars and
  // accelerations at intermediate time steps n+alpha_F and n+alpha_M,
  // respectively, for next iteration.
  // This has to be done at the end of the iteration, since we might
  // need the velocities at n+alpha_F in a potential coupling
  // algorithm, for instance.
  // -------------------------------------------------------------------
  gen_alpha_update_acceleration();
  gen_alpha_intermediate_values();


}  // FluidImplicitTimeInt::IterUpdate

/*----------------------------------------------------------------------*
 | convergence check                                           vg 09/11 |
 *----------------------------------------------------------------------*/
bool FLD::FluidImplicitTimeInt::convergence_check(int itnum, int itmax, const double velrestol,
    const double velinctol, const double presrestol, const double presinctol)
{
  // -------------------------------------------------------------------
  // calculate and print out norms for convergence check
  // (blank residual DOFs which are on Dirichlet BC
  // We can do this because the values at the dirichlet positions
  // are not used anyway.
  // We could avoid this though, if velrowmap_ and prerowmap_ would
  // not include the dirichlet values as well. But it is expensive
  // to avoid that.)
  // -------------------------------------------------------------------
  dbcmaps_->insert_cond_vector(*dbcmaps_->extract_cond_vector(*zeros_), *residual_);

  // -------------------------------------------------------------------
  // take surface volumetric flow rate into account
  //    std::shared_ptr<Core::LinAlg::Vector<double>> temp_vec = Teuchos::rcp(new
  //    Core::LinAlg::Vector<double>(*vol_surf_flow_bcmaps_,true));
  //    vol_surf_flow_bc_->insert_cond_vector( *temp_vec , *residual_);
  // -------------------------------------------------------------------
  insert_volumetric_surface_flow_cond_vector(zeros_, residual_);

  // remove contributions of pressure mode that would not vanish due to the
  // projection
  // In meshtying with block matrix, the projector might have another length
  // compared to residual. Thus, the projector is applied differently in this case.
  if (projector_ != nullptr)
  {
    if (msht_ == Inpar::FLUID::condensed_bmat_merged or msht_ == Inpar::FLUID::condensed_bmat)
      meshtying_->apply_pt_to_residual(*sysmat_, *residual_, *projector_);
    else
      *residual_ = projector_->to_reduced(*residual_);
  }



  std::shared_ptr<Core::LinAlg::Vector<double>> onlyvel =
      velpressplitter_->extract_other_vector(*residual_);

  onlyvel->norm_2(&vresnorm_);

  velpressplitter_->extract_other_vector(*incvel_, *onlyvel);

  onlyvel->norm_2(&incvelnorm_L2_);

  velpressplitter_->extract_other_vector(*velnp_, *onlyvel);

  onlyvel->norm_2(&velnorm_L2_);

  std::shared_ptr<Core::LinAlg::Vector<double>> onlypre =
      velpressplitter_->extract_cond_vector(*residual_);
  onlypre->norm_2(&presnorm_);

  velpressplitter_->extract_cond_vector(*incvel_, *onlypre);
  onlypre->norm_2(&incprenorm_L2_);

  velpressplitter_->extract_cond_vector(*velnp_, *onlypre);
  onlypre->norm_2(&prenorm_L2_);

  // check for any INF's and NaN's
  if (std::isnan(vresnorm_) or std::isnan(incvelnorm_L2_) or std::isnan(velnorm_L2_) or
      std::isnan(presnorm_) or std::isnan(incprenorm_L2_) or std::isnan(prenorm_L2_))
    FOUR_C_THROW("At least one of the calculated vector norms is NaN.");

  if (std::isinf(vresnorm_) or std::isinf(incvelnorm_L2_) or std::isinf(velnorm_L2_) or
      std::isinf(presnorm_) or std::isinf(incprenorm_L2_) or std::isinf(prenorm_L2_))
    FOUR_C_THROW("At least one of the calculated vector norms is INF.");

  // care for the case that nothing really happens in velocity
  // or pressure field
  if (velnorm_L2_ < 1e-5) velnorm_L2_ = 1.0;
  if (prenorm_L2_ < 1e-5) prenorm_L2_ = 1.0;

  if (myrank_ == 0)
  {
    if (itnum > 0)
    {
      printf("|  %3d/%3d   | %10.3E  | %10.3E  | %10.3E  | %10.3E  |", itnum, itmax, vresnorm_,
          presnorm_, incvelnorm_L2_ / velnorm_L2_, incprenorm_L2_ / prenorm_L2_);
      printf(" (ts=%10.3E,te=%10.3E", dtsolve_, dtele_);
      if (turbmodel_ == Inpar::FLUID::dynamic_smagorinsky) printf(",tf=%10.3E", dtfilter_);
      printf(")\n");
    }
    else
    {
      printf("|   --/%3d   | %10.3E  | %10.3E  |      --     |      --     |", itmax, vresnorm_,
          presnorm_);
      printf(" (      --     ,te=%10.3E", dtele_);
      if (turbmodel_ == Inpar::FLUID::dynamic_smagorinsky) printf(",tf=%10.3E", dtfilter_);
      printf(")\n");
    }
  }

  // -------------------------------------------------------------------
  // check convergence and print out respective information:
  // - stop if convergence is achieved
  // - warn if itemax is reached without convergence, but proceed to
  //   next timestep
  // -------------------------------------------------------------------
  if (vresnorm_ <= velrestol and presnorm_ <= presrestol and
      incvelnorm_L2_ / velnorm_L2_ <= velinctol and incprenorm_L2_ / prenorm_L2_ <= presinctol)
  {
    if (myrank_ == 0 and (inconsistent_ or (not inconsistent_ and itnum == 0)))
    {
      printf("+------------+-------------+-------------+-------------+-------------+\n");
    }
    return true;
  }
  else
  {
    if (itnum == itmax or (not inconsistent_ and itnum == 0))
    {
      if (myrank_ == 0 and
          ((itnum == itmax and inconsistent_) or (not inconsistent_ and itnum == 0)))
      {
        printf("+---------------------------------------------------------------+\n");
        printf("|            >>>>>> not converged in itemax steps!              |\n");
        printf("+---------------------------------------------------------------+\n");
      }
      return true;
    }
  }

  return false;

}  // FluidImplicitTimeInt::convergence_check

/*----------------------------------------------------------------------*
 | Update of an Ale field based on the fluid state           hahn 08/14 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::ale_update(std::string condName)
{
  // Preparation: Check, if an Ale update needs to be done
  if (condName == "ALEUPDATECoupling")
  {
    if (not(alefluid_ and surfacesplitter_->au_cond_relevant())) return;
  }
  else
  {
    FOUR_C_THROW("AleUpdate: So far, only FREESURFCoupling and ALEUPDATECoupling are supported.");
  }

  // Sort Ale update conditions, such that line conditions overwrite surface
  // conditions overwrite volume conditions
  // **************************************************************************
  // Get (unsorted) Ale update conditions
  std::vector<const Core::Conditions::Condition*> unsortedConds;
  discret_->get_condition(condName, unsortedConds);

  // Sort Ale update conditions
  std::vector<const Core::Conditions::Condition*> conds;
  conds.clear();

  // - first volume conditions
  for (auto& unsortedCond : unsortedConds)
  {
    if (unsortedCond->g_type() == Core::Conditions::geometry_type_volume)
      conds.push_back(unsortedCond);
  }

  // - then surface conditions
  for (auto& unsortedCond : unsortedConds)
  {
    if (unsortedCond->g_type() == Core::Conditions::geometry_type_surface)
      conds.push_back(unsortedCond);
  }

  // - and finally line conditions
  for (auto& unsortedCond : unsortedConds)
  {
    if (unsortedCond->g_type() == Core::Conditions::geometry_type_line)
      conds.push_back(unsortedCond);
  }

  // Loop through all conditions and do the Ale update according to the coupling type
  // ********************************************************************************
  for (auto& cond : conds)
  {
    // Initialize some variables:
    // Select the i-th condition in the vector
    std::vector<const Core::Conditions::Condition*> selectedCond;
    selectedCond.clear();
    selectedCond.push_back(cond);

    // Get condition name
    std::string condName;
    if (selectedCond[0]->type() == Core::Conditions::ALEUPDATECoupling)
    {
      condName = "ALEUPDATECoupling";
    }

    // Get coupling type
    std::string coupling = (selectedCond[0]->parameters().get<std::string>("COUPLING"));

    // Get scaling value
    const double scalingValue = selectedCond[0]->parameters().get<double>("VAL");

    // Get function for node normal calculation
    const int nodeNormalFunct = selectedCond[0]->parameters().get<int>("NODENORMALFUNCT");

    // Get a vector layout from the discretization to construct matching
    // vectors and matrices
    //                 local <-> global dof numbering
    const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

    // Obtain the global IDs of the condition's nodes for the current processor
    auto gIdNodes = Core::Conditions::find_conditioned_node_ids(
        *discret_, selectedCond, Core::Conditions::LookFor::locally_owned);

    // Obtain fluid and ale state variables for nodes in the condition
    // **************************************************************************
    // Velocities at n+1 for nodes in the condition
    std::shared_ptr<Core::LinAlg::Vector<double>> velnp;

    // Grid velocities at n+1 for nodes in the condition
    std::shared_ptr<Core::LinAlg::Vector<double>> gridv;

    // Displacements at n for nodes in the condition
    std::shared_ptr<Core::LinAlg::Vector<double>> disp;

    // Create container for new displacements at n+1 for nodes in the condition
    std::shared_ptr<Core::LinAlg::Vector<double>> dispnp;

    // Set variables depending on condition
    if (condName == "ALEUPDATECoupling")
    {
      velnp = surfacesplitter_->extract_au_cond_vector(*velnp_);
      gridv = surfacesplitter_->extract_au_cond_vector(*gridv_);
      disp = surfacesplitter_->extract_au_cond_vector(*dispn_);
      dispnp = surfacesplitter_->extract_au_cond_vector(*dispnp_);
    }

    // Do the local lagrangian coupling
    // **************************************************************************
    if (coupling == "lagrange")
    {
      // Loop through all nodes in the condition
      for (int gIdNode : gIdNodes)
      {
        // Obtain local degree of freedom indices
        std::vector<int> dofsLocalInd;
        get_dofs_vector_local_indicesfor_node(gIdNode, *gridv, false, &dofsLocalInd);

        // Calculate new ale velocities
        for (int i = 0; i < numdim_; i++)
        {
          (*gridv).get_values()[dofsLocalInd[i]] = velnp->local_values_as_span()[dofsLocalInd[i]];
        }
      }
    }
    // For all other couplings, do the following common calculations
    // ************************************************************************
    else
    {
      // Calculate normalized node normals and tangents for current condition
      std::shared_ptr<Core::LinAlg::Vector<double>> nodeNormals;
      std::shared_ptr<Core::LinAlg::Vector<double>> nodeTangents;

      if (nodeNormalFunct == 0)
      {  // Obtain node normals from element (mass-consistent node normal)
        // Define corresponding parameter list
        Teuchos::ParameterList eleparams;
        eleparams.set<FLD::BoundaryAction>("action", FLD::boundary_calc_node_normal);

        // Initialize global node normals vector
        std::shared_ptr<Core::LinAlg::Vector<double>> globalNodeNormals =
            std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

        // Evaluate condition to calculate the node normals
        // Note: the normal vectors do not yet have length 1.0
        discret_->clear_state();
        discret_->set_state(ndsale_, "dispnp", *dispnp_);
        discret_->evaluate_condition(eleparams, globalNodeNormals, condName);
        discret_->clear_state();

        // Obtain node normals and initialize node tangents for current condition
        // (vector only contain the nodes in the condition).
        if (condName == "ALEUPDATECoupling")
        {
          nodeNormals = surfacesplitter_->extract_au_cond_vector(*globalNodeNormals);
          nodeTangents =
              std::make_shared<Core::LinAlg::Vector<double>>(*surfacesplitter_->au_cond_map());
        }
      }
      else
      {  // Obtain node normals from function
        if (condName == "ALEUPDATECoupling")
        {
          nodeNormals =
              std::make_shared<Core::LinAlg::Vector<double>>(*surfacesplitter_->au_cond_map());
          nodeTangents =
              std::make_shared<Core::LinAlg::Vector<double>>(*surfacesplitter_->au_cond_map());
        }

        // Loop through all nodes and obtain node normal from function
        for (int gIdNode : gIdNodes)
        {
          // Make sure, that the current processor shall calculate this node
          if (not((discret_->have_global_node(gIdNode)) &&
                  (discret_->g_node(gIdNode)->owner() == myrank_)))
            continue;

          // Obtain local degree of freedom indices
          std::vector<int> dofsLocalInd;
          get_dofs_vector_local_indicesfor_node(gIdNode, *nodeNormals, false, &dofsLocalInd);

          // Calculate current position for node
          Core::Nodes::Node* currNode = discret_->g_node(gIdNode);
          std::vector<double> currPos(numdim_);

          const auto& refPos = currNode->x();

          for (int i = 0; i < numdim_; ++i)
          {
            currPos[i] = refPos[i] + dispnp->local_values_as_span()[dofsLocalInd[i]];
          }

          // Calculate node normal components
          for (int i = 0; i < numdim_; i++)
          {
            (*nodeNormals).get_values()[dofsLocalInd[i]] =
                (Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(
                     nodeNormalFunct))
                    .evaluate(currPos, 0.0, i);
          }
        }
      }

      // Normalize node normal vectors and calculate node tangent vectors, which are
      // orthogonal to the normal vector and for 3D to e_y and for 2D to e_z!
      for (int gIdNode : gIdNodes)
      {
        // Make sure, that the current processor shall calculate this node
        if (not((discret_->have_global_node(gIdNode)) &&
                (discret_->g_node(gIdNode)->owner() == myrank_)))
          continue;

        // Obtain local degree of freedom indices
        std::vector<int> dofsLocalInd;
        get_dofs_vector_local_indicesfor_node(gIdNode, *nodeNormals, false, &dofsLocalInd);

        // Calculate length of node normal
        double lengthNodeNormal = 0.0;
        for (int i = 0; i < numdim_; i++)
          lengthNodeNormal += nodeNormals->local_values_as_span()[dofsLocalInd[i]] *
                              nodeNormals->local_values_as_span()[dofsLocalInd[i]];
        lengthNodeNormal = sqrt(lengthNodeNormal);

        // Normalize vector
        for (int i = 0; i < numdim_; i++)
        {
          (*nodeNormals).get_values()[dofsLocalInd[i]] =
              (1.0 / lengthNodeNormal) * nodeNormals->local_values_as_span()[dofsLocalInd[i]];
        }

        // Calculate normalized tangent vectors, which are orthogonal to
        // the normal vector and to e_y (3D) or to e_z (2D)! For 3D, in
        // case that the normal vector and e_y are parallel, the tangent
        // vector is constructed to  be orthogonal to the normal vector
        // and to e_z.
        if (numdim_ == 3)
        {
          double lengthNodeTangent = sqrt(nodeNormals->local_values_as_span()[dofsLocalInd[0]] *
                                              nodeNormals->local_values_as_span()[dofsLocalInd[0]] +
                                          nodeNormals->local_values_as_span()[dofsLocalInd[2]] *
                                              nodeNormals->local_values_as_span()[dofsLocalInd[2]]);
          if (lengthNodeTangent > 0.1)
          {  // Tangent vector orthogonal to normal and e_y
            (*nodeTangents).get_values()[dofsLocalInd[0]] =
                -nodeNormals->local_values_as_span()[dofsLocalInd[2]] / lengthNodeTangent;
            (*nodeTangents).get_values()[dofsLocalInd[1]] = 0.0;
            (*nodeTangents).get_values()[dofsLocalInd[2]] =
                nodeNormals->local_values_as_span()[dofsLocalInd[0]] / lengthNodeTangent;
          }
          else
          {  // Tangent vector orthogonal to normal and e_z
            lengthNodeTangent = sqrt(nodeNormals->local_values_as_span()[dofsLocalInd[0]] *
                                         nodeNormals->local_values_as_span()[dofsLocalInd[0]] +
                                     nodeNormals->local_values_as_span()[dofsLocalInd[1]] *
                                         nodeNormals->local_values_as_span()[dofsLocalInd[1]]);
            (*nodeTangents).get_values()[dofsLocalInd[0]] =
                -nodeNormals->local_values_as_span()[dofsLocalInd[1]] / lengthNodeTangent;
            (*nodeTangents).get_values()[dofsLocalInd[1]] =
                nodeNormals->local_values_as_span()[dofsLocalInd[0]] / lengthNodeTangent;
            (*nodeTangents).get_values()[dofsLocalInd[2]] = 0.0;
          }
        }
        else if (numdim_ == 2)
        {
          double lengthNodeTangent = sqrt(nodeNormals->local_values_as_span()[dofsLocalInd[0]] *
                                              nodeNormals->local_values_as_span()[dofsLocalInd[0]] +
                                          nodeNormals->local_values_as_span()[dofsLocalInd[1]] *
                                              nodeNormals->local_values_as_span()[dofsLocalInd[1]]);
          (*nodeTangents).get_values()[dofsLocalInd[0]] =
              nodeNormals->local_values_as_span()[dofsLocalInd[1]] / lengthNodeTangent;
          (*nodeTangents).get_values()[dofsLocalInd[1]] =
              -nodeNormals->local_values_as_span()[dofsLocalInd[0]] / lengthNodeTangent;
          (*nodeTangents).get_values()[dofsLocalInd[2]] = 0.0;
        }
        else
        {
          FOUR_C_THROW("Spatial dimension needs to be 2 or 3!");
        }
      }

      // Do the height function coupling
      // ************************************************************************
      if (coupling == "heightfunction")
      {
        // Loop through all nodes in the condition
        for (int gIdNode : gIdNodes)
        {
          // Obtain local degree of freedom indices
          std::vector<int> dofsLocalInd;
          get_dofs_vector_local_indicesfor_node(gIdNode, *nodeNormals, false, &dofsLocalInd);

          // Calculate dot product between velnp and nodeNormals
          double velnpDotNodeNormal = 0.0;
          for (int i = 0; i < numdim_; i++)
          {
            velnpDotNodeNormal += nodeNormals->local_values_as_span()[dofsLocalInd[i]] *
                                  velnp->local_values_as_span()[dofsLocalInd[i]];
          }

          // Calculate ale velocities
          for (int i = 0; i < numdim_; i++)
          {
            // Height function approach: last entry of u_G is
            // (velnp*nodeNormals / nodeNormals_z), the other entries are zero
            if (i == numdim_ - 1)
              (*gridv).get_values()[dofsLocalInd[i]] =
                  velnpDotNodeNormal / nodeNormals->local_values_as_span()[dofsLocalInd[i]];
            else
              (*gridv).get_values()[dofsLocalInd[i]] = 0.0;
          }
        }
        // Do the coupling based on a mean tangential velocity
        // ************************************************************************
      }
      else if ((coupling == "meantangentialvelocity") or
               (coupling == "meantangentialvelocityscaled"))
      {
        // Only implemented for 3D
        if (numdim_ != 3)
          FOUR_C_THROW("AleUpdate: meantangentialvelocity(scaled) only implemented for 3D.");

        // Determine the mean tangent velocity of the current condition's nodes
        double localSumVelnpDotNodeTangent = 0.0;
        int localNumOfCondNodes = 0;

        // Loop through all nodes in the condition
        for (int gIdNode : gIdNodes)
        {
          // Obtain local degree of freedom indices
          std::vector<int> dofsLocalInd;
          get_dofs_vector_local_indicesfor_node(gIdNode, *nodeNormals, false, &dofsLocalInd);

          // Calculate dot product between velnp and the node tangent vector
          double velnpDotNodeTangent = 0.0;
          for (int i = 0; i < numdim_; i++)
          {
            velnpDotNodeTangent += nodeTangents->local_values_as_span()[dofsLocalInd[i]] *
                                   velnp->local_values_as_span()[dofsLocalInd[i]];
          }

          localSumVelnpDotNodeTangent += velnpDotNodeTangent;
          localNumOfCondNodes += 1;
        }

        // Sum variables over all processors to obtain global value
        double globalSumVelnpDotNodeTangent = 0.0;
        globalSumVelnpDotNodeTangent =
            Core::Communication::sum_all(localSumVelnpDotNodeTangent, dofrowmap->get_comm());

        int globalNumOfCondNodes = 0;
        globalNumOfCondNodes =
            Core::Communication::sum_all(localNumOfCondNodes, dofrowmap->get_comm());

        // Finalize calculation of mean tangent velocity
        double lambda = 0.0;
        lambda = globalSumVelnpDotNodeTangent / globalNumOfCondNodes;

        // Loop through all nodes in the condition
        for (int gIdNode : gIdNodes)
        {
          // Obtain local degree of freedom indices
          std::vector<int> dofsLocalInd;
          get_dofs_vector_local_indicesfor_node(gIdNode, *nodeNormals, false, &dofsLocalInd);

          // Calculate dot product between velnp and nodeNormals
          double velnpDotNodeNormal = 0.0;
          for (int i = 0; i < numdim_; i++)
          {
            velnpDotNodeNormal += nodeNormals->local_values_as_span()[dofsLocalInd[i]] *
                                  velnp->local_values_as_span()[dofsLocalInd[i]];
          }

          // Setup matrix A and vector b for grid velocity calculation.
          // The following two equations are used:
          // 1) u_g * n = u_f * n
          // 2) u_g * t = lambda * scalingValue (* scalingFactor)
          // with u_g and u_f being the grid and fluid velocity, resp.,
          // n the normal and t the tangent vector.
          Core::LinAlg::Matrix<2, 3> A(Core::LinAlg::Initialization::zero);
          A(0, 0) = nodeNormals->local_values_as_span()[dofsLocalInd[0]];
          A(0, 1) = nodeNormals->local_values_as_span()[dofsLocalInd[1]];
          A(0, 2) = nodeNormals->local_values_as_span()[dofsLocalInd[2]];
          A(1, 0) = nodeTangents->local_values_as_span()[dofsLocalInd[0]];
          A(1, 1) = nodeTangents->local_values_as_span()[dofsLocalInd[1]];
          A(1, 2) = nodeTangents->local_values_as_span()[dofsLocalInd[2]];

          Core::LinAlg::Matrix<2, 1> b(Core::LinAlg::Initialization::zero);
          b(0, 0) = velnpDotNodeNormal;
          if (coupling == "meantangentialvelocity")
          {
            b(1, 0) = lambda * scalingValue;
          }
          else if (coupling == "meantangentialvelocityscaled")
          {
            double scalingFactor = sqrt(nodeNormals->local_values_as_span()[dofsLocalInd[0]] *
                                        nodeNormals->local_values_as_span()[dofsLocalInd[0]]);
            b(1, 0) = lambda * scalingValue * scalingFactor;
          }

          // Calculate pseudo inverse of A (always possible due to linear independent rows [n and
          // t])
          Core::LinAlg::Matrix<2, 2> matTimesMatTransposed(Core::LinAlg::Initialization::zero);
          matTimesMatTransposed.multiply_nt(A, A);

          Core::LinAlg::Matrix<2, 2> inverseOfMatTimesmatTransposed(
              Core::LinAlg::Initialization::zero);
          inverseOfMatTimesmatTransposed.invert(matTimesMatTransposed);

          Core::LinAlg::Matrix<3, 2> pInvA(Core::LinAlg::Initialization::zero);
          pInvA.multiply_tn(A, inverseOfMatTimesmatTransposed);

          // Solve for grid velocities
          Core::LinAlg::Matrix<3, 1> sol(Core::LinAlg::Initialization::zero);
          sol.multiply(pInvA, b);

          // Calculate ale velocities
          for (int i = 0; i < numdim_; i++)
          {
            (*gridv).get_values()[dofsLocalInd[i]] = sol(i, 0);
          }
        }
        // Do the coupling based on a spherical height function
        // ************************************************************************
      }
      else if (coupling == "sphereHeightFunction")
      {
        // Only implemented for 3D
        if (numdim_ != 3)
          FOUR_C_THROW("AleUpdate: sphericalHeightFunction only implemented for 3D.");

        // Loop through all nodes and determine grid velocity
        for (int gIdNode : gIdNodes)
        {
          // Obtain local degree of freedom indices
          std::vector<int> dofsLocalInd;
          get_dofs_vector_local_indicesfor_node(gIdNode, *nodeNormals, false, &dofsLocalInd);

          // Calculate current position for node and its vector length
          Core::Nodes::Node* currNode = discret_->g_node(gIdNode);
          std::vector<double> currPos(numdim_);

          const auto& refPos = currNode->x();

          double lengthCurrPos = 0.0;
          for (int i = 0; i < numdim_; ++i)
          {
            currPos[i] = refPos[i] + dispnp->local_values_as_span()[dofsLocalInd[i]];
            lengthCurrPos += currPos[i] * currPos[i];
          }
          lengthCurrPos = sqrt(lengthCurrPos);

          // Obtain angles phi and theta for spherical coordinate system representation
          double phi = atan2(currPos[1], currPos[0]);
          if (phi < 0) phi = phi + 2 * std::numbers::pi;
          const double theta = acos(currPos[2] / lengthCurrPos);

          // Precalculate some sin and cos
          const double sinTheta = sin(theta);
          const double cosTheta = cos(theta);
          const double sinPhi = sin(phi);
          const double cosPhi = cos(phi);

          // Calculate dot product between velnp and e_theta
          double velnpDotETheta =
              cosTheta * cosPhi * velnp->local_values_as_span()[dofsLocalInd[0]] +
              cosTheta * sinPhi * velnp->local_values_as_span()[dofsLocalInd[1]] +
              -sinTheta * velnp->local_values_as_span()[dofsLocalInd[2]];

          // Setup matrix A and vector b for grid velocity calculation.
          // The following three equations are used:
          // 1) u_g * e_r = 0
          // 2) u_g * e_phi = 0
          // 3) u_g * e_theta = u_f * e_theta
          // with u_g and u_f being the grid and fluid velocity, resp.
          // and e_r, e_phi and e_theta the basis vectors of a spherical
          // coordinate system, expressed in Cartesian coordinates.
          Core::LinAlg::Matrix<3, 3> A(Core::LinAlg::Initialization::zero);
          A(0, 0) = sinTheta * cosPhi;
          A(0, 1) = sinTheta * sinPhi;
          A(0, 2) = cosTheta;
          A(1, 0) = -sinPhi;
          A(1, 1) = cosPhi;
          A(1, 2) = 0.0;
          A(2, 0) = cosTheta * cosPhi;
          A(2, 1) = cosTheta * sinPhi;
          A(2, 2) = -sinTheta;

          Core::LinAlg::Matrix<3, 1> b(Core::LinAlg::Initialization::zero);
          b(0, 0) = 0.0;
          b(1, 0) = 0.0;
          b(2, 0) = velnpDotETheta;

          // Calculate inverse of A (always possible due to linear independent rows)
          Core::LinAlg::Matrix<3, 3> invA(Core::LinAlg::Initialization::zero);
          invA.invert(A);

          // Solve for grid velocities
          Core::LinAlg::Matrix<3, 1> sol(Core::LinAlg::Initialization::zero);
          sol.multiply(invA, b);

          // Calculate ale velocities
          for (int i = 0; i < numdim_; i++)
          {
            (*gridv).get_values()[dofsLocalInd[i]] = sol(i, 0);
          }
        }
      }
    }

    // Update Ale variables
    // ************************************************************************
    // Update the displacements at n+1
    dispnp->update(1.0, *disp, dta_, *gridv, 0.0);

    // Insert calculated displacements and velocities at n+1 into global Ale variables
    if (condName == "ALEUPDATECoupling")
    {
      surfacesplitter_->insert_au_cond_vector(*dispnp, *dispnp_);
      surfacesplitter_->insert_au_cond_vector(*gridv, *gridv_);
    }
  }
}

/*-------------------------------------------------------------------------------------------------*
 | For a given node, obtain local indices of dofs in a vector (like e.g. velnp)         hahn 08/14 |
 *-------------------------------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::get_dofs_vector_local_indicesfor_node(int nodeGid,
    Core::LinAlg::Vector<double>& vec, bool withPressure, std::vector<int>* dofsLocalInd)
{
  // Determine dimensions to be taken care of
  int dim = numdim_;
  if (withPressure) dim += 1;

  // Get local id for this node
  int nodeLid = (discret_->node_row_map())->lid(nodeGid);
  if (nodeLid == -1) FOUR_C_THROW("No LID for node!");

  // Get vector of global ids for this node's degrees of freedom
  std::vector<int> dofsGid;
  dofsGid.clear();
  discret_->dof(discret_->l_row_node(nodeLid), dofsGid);

  // Get local indices for dofs in vector (like e.g. velnp) for given node
  (*dofsLocalInd).clear();
  (*dofsLocalInd).resize(dim);

  for (int i = 0; i < dim; i++)
  {
    int dofGid = dofsGid[i];
    if (!vec.get_map().my_gid(dofGid))
      FOUR_C_THROW("Sparse vector does not have global row  {} or vectors don't match", dofGid);
    (*dofsLocalInd)[i] = vec.get_map().lid(dofGid);
  }
}

/*----------------------------------------------------------------------*
 | Assemble Mat and RHS and apply Dirichlet Conditions          bk 12/13|
 | Call routine from outside of fluid,                                  |
 | e.g. FSI, FPSI, Poro, ...                                            |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::evaluate(
    std::shared_ptr<const Core::LinAlg::Vector<double>> stepinc)
{
  // update solution by adding step increment to previous converged solution
  if (stepinc != nullptr)
  {
    // Add stepinc to veln_ for non-Dirichlet values.
    std::shared_ptr<Core::LinAlg::Vector<double>> aux =
        std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);
    aux->update(1.0, *veln_, 1.0, *stepinc, 0.0);

    // Set Dirichlet values
    dbcmaps_->insert_cond_vector(*dbcmaps_->extract_cond_vector(*velnp_), *aux);

    insert_volumetric_surface_flow_cond_vector(velnp_, aux);

    *velnp_ = *aux;
  }

  // --------------------------------------------------
  // the following steps have to be repeated after that the velocity has been updated
  // --------------------------------------------------

  // adjust accnp_ according to Dirichlet values of velnp_ for GenAlpha
  gen_alpha_update_acceleration();

  // compute values at intermediate time steps for GenAlpha
  // ----------------------------------------------------------------
  gen_alpha_intermediate_values();

  if (alefluid_)
  {
    // account for potentially moving Neumann boundaries
    Teuchos::ParameterList eleparams;
    discret_->set_state(ndsale_, "dispnp", *dispnp_);

    // evaluate Neumann conditions
    neumann_loads_->put_scalar(0.0);
    discret_->set_state("scaaf", *scaaf_);
    discret_->evaluate_neumann(eleparams, *neumann_loads_);
    discret_->clear_state();
  }

  if (msht_ != Inpar::FLUID::no_meshtying) meshtying_->msht_split(sysmat_, shapederivatives_);

  prepare_solve();
}

/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                                      |
 | One-step-Theta: (step>1)                                             |
 |                                                                      |
 |  accn_  = (velnp_-veln_) / (Theta * dt) - (1/Theta -1) * accn_"(n+1) |
 |                                                                      |
 |  velnm_ =veln_                                                       |
 |  veln_  =velnp_                                                      |
 |                                                                      |
 |  BDF2:           (step>1)                                            |
 |                                                                      |
 |               2*dt(n)+dt(n-1)              dt(n)+dt(n-1)             |
 |  accn_   = --------------------- velnp_ - --------------- veln_      |
 |            dt(n)*[dt(n)+dt(n-1)]           dt(n)*dt(n-1)             |
 |                                                                      |
 |                     dt(n)                                            |
 |           + ----------------------- velnm_                           |
 |             dt(n-1)*[dt(n)+dt(n-1)]                                  |
 |                                                                      |
 |  velnm_ =veln_                                                       |
 |  veln_  =velnp_                                                      |
 |                                                                      |
 |  BDF2 and  One-step-Theta: (step==1)                                 |
 |                                                                      |
 |  The given formulas are only valid from the second timestep. In the  |
 |  first step, the acceleration is calculated simply by                |
 |                                                                      |
 |  accn_  = (velnp_-veln_) / (dt)                                      |
 |                                                                      |
 |                                                           gammi 04/07|
 |  overloaded in TimIntRedModels                               bk 12/13|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::time_update()
{
  Teuchos::ParameterList* stabparams;
  stabparams = &(params_->sublist("RESIDUAL-BASED STABILIZATION"));

  if (Teuchos::getIntegralValue<Inpar::FLUID::StabType>(*stabparams, "STABTYPE") ==
      Inpar::FLUID::StabType::stabtype_residualbased)
  {
    if (Teuchos::getIntegralValue<Inpar::FLUID::SubscalesTD>(*stabparams, "TDS") ==
        Inpar::FLUID::SubscalesTD::subscales_time_dependent)
    {
      const double tcpu = Teuchos::Time::wallTime();

      if (myrank_ == 0)
      {
        std::cout << "time update for subscales";
      }

      // call elements to calculate system matrix and rhs and assemble
      // this is required for the time update of the subgrid scales and
      // makes sure that the current subgrid scales correspond to the
      // current residual
      assemble_mat_and_rhs();

      // create the parameters for the discretization
      Teuchos::ParameterList eleparams;
      // action for elements
      eleparams.set<FLD::Action>("action", FLD::calc_fluid_genalpha_update_for_subscales);

      // update time parameters
      set_gamma(eleparams);

      eleparams.set("dt", dta_);

      // call loop over elements to update subgrid scales
      discret_->evaluate(eleparams, nullptr, nullptr, nullptr, nullptr, nullptr);

      if (myrank_ == 0) std::cout << "(" << Teuchos::Time::wallTime() - tcpu << ")\n";
    }
  }

  // compute accelerations
  tim_int_calculate_acceleration();

  // acceleration of this step becomes most recent
  // acceleration of the last step
  accnm_->update(1.0, *accn_, 0.0);
  accn_->update(1.0, *accnp_, 0.0);

  // velocities/pressures of this step become most recent
  // velocities/pressures of the last step
  velnm_->update(1.0, *veln_, 0.0);
  veln_->update(1.0, *velnp_, 0.0);

  // displacement vectors for ALE
  if (alefluid_)
  {
    dispnm_->update(1.0, *dispn_, 0.0);
    dispn_->update(1.0, *dispnp_, 0.0);

    // gridvelocities of this step become most recent
    // gridvelocities of the last step
    gridvn_->update(1.0, *gridv_, 0.0);
  }

  // update stresses and wss
  time_update_stresses();

  // update flow-rate, flow-volume and impedance vectors in case of flow-dependent pressure boundary
  // conditions,
  if (nonlinearbc_) time_update_nonlinear_bc();

  // update external forces
  time_update_external_forces();

  // call time update of forcing routine
  if (forcing_interface_ != nullptr) forcing_interface_->time_update_forcing();

  // account for possible changes in time step size and update previous
  // time step size accordingly
  dtp_ = dta_;
  discret_->clear_state();
}  // FluidImplicitTimeInt::TimeUpdate


/*----------------------------------------------------------------------*
 | Update of stresses                                        thon 03/15 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::time_update_stresses()
{
  if (writestresses_) stressmanager_->get_stresses(*trueresidual_, dta_);

  if (write_wall_shear_stresses_) stressmanager_->get_wall_shear_stresses(*trueresidual_, dta_);
}

/*----------------------------------------------------------------------*
 | Update NonlinearBCs                                       thon 09/14 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::time_update_nonlinear_bc()
{
  std::vector<const Core::Conditions::Condition*> flowdeppressureline;
  discret_->get_condition("LineFlowDepPressure", flowdeppressureline);
  std::vector<const Core::Conditions::Condition*> flowdeppressuresurf;
  discret_->get_condition("SurfaceFlowDepPressure", flowdeppressuresurf);

  if (flowdeppressureline.size() != 0 or flowdeppressuresurf.size() != 0)
  {
    for (size_t i = 0; i < flowratenp_.size(); ++i)
    {
      flowratenm_[i] = flowraten_[i];
      flowraten_[i] = flowratenp_[i];

      flowvolumenm_[i] = flowvolumen_[i];
      flowvolumen_[i] = flowvolumenp_[i];
    }

    // close this time step also in output file
    if (myrank_ == 0)
    {
      const std::string fname1 =
          Global::Problem::instance()->output_control_file()->file_name() + ".fdpressure";
      std::ofstream f1;
      f1.open(fname1.c_str(), std::fstream::ate | std::fstream::app);

      f1 << "\n";
      f1.flush();
      f1.close();
    }
  }

  if (isimpedancebc_)
  {
    // do time update of impedance conditions
    impedancebc_->time_update_impedances(time_);
  }
}


/*----------------------------------------------------------------------*
 | Update of external forces                                ghamm 12/14 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::time_update_external_forces() {}


/*----------------------------------------------------------------------*
 | Calculate Acceleration                                               |
 | overloaded in TimIntPoro                                    bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::tim_int_calculate_acceleration()
{
  std::shared_ptr<Core::LinAlg::Vector<double>> onlyaccn = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> onlyaccnp = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> onlyvelnm = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> onlyveln = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> onlyvelnp = nullptr;

  if (physicaltype_ == Inpar::FLUID::artcomp)  // artcomp case
  {
    onlyaccn = accn_;
    onlyaccnp = accnp_;
    onlyvelnm = velnm_;
    onlyveln = veln_;
    onlyvelnp = velnp_;
  }
  else  // standard case
  {
    onlyaccn = velpressplitter_->extract_other_vector(*accn_);
    onlyaccnp = velpressplitter_->extract_other_vector(*accnp_);
    onlyvelnm = velpressplitter_->extract_other_vector(*velnm_);
    onlyveln = velpressplitter_->extract_other_vector(*veln_);
    onlyvelnp = velpressplitter_->extract_other_vector(*velnp_);
  }

  calculate_acceleration(onlyvelnp, onlyveln, onlyvelnm, onlyaccn, onlyaccnp);

  // copy back into global vector
  Core::LinAlg::export_to(*onlyaccnp, *accnp_);
}

/*----------------------------------------------------------------------*
 | calculate intermediate solution                       rasthofer 05/13|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::calc_intermediate_solution()
{
  if ((special_flow_ == "forced_homogeneous_isotropic_turbulence" or
          special_flow_ == "scatra_forced_homogeneous_isotropic_turbulence" or
          special_flow_ == "decaying_homogeneous_isotropic_turbulence") and
      Teuchos::getIntegralValue<Inpar::FLUID::ForcingType>(params_->sublist("TURBULENCE MODEL"),
          "FORCING_TYPE") == Inpar::FLUID::linear_compensation_from_intermediate_spectrum)
  {
    bool activate = true;
    if (special_flow_ == "decaying_homogeneous_isotropic_turbulence" and
        step_ > params_->sublist("TURBULENCE MODEL").get<int>("FORCING_TIME_STEPS", 0))
      activate = false;

    if (activate)
    {
      if (forcing_interface_ == nullptr) FOUR_C_THROW("Forcing expected!");

      if (myrank_ == 0)
      {
        std::cout << "+------------------------------------------------------------------------+\n";
        std::cout << "|     calculate intermediate solution\n";
        std::cout << "|" << std::endl;
      }

      // turn off forcing in Solve()
      forcing_interface_->activate_forcing(false);

      // temporary store velnp_ since it will be modified in Solve()
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
      std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
          std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
      tmp->update(1.0, *velnp_, 0.0);

      // compute intermediate solution without forcing
      forcing_->put_scalar(0.0);  // just to be sure
      solve();

      // calculate required forcing
      forcing_interface_->calculate_forcing(step_);

      // reset velnp_
      velnp_->update(1.0, *tmp, 0.0);

      // recompute intermediate values, since they have been likewise overwritten
      // --------------------------------------------------
      // adjust accnp according to Dirichlet values of velnp
      //
      //                                  n+1     n
      //                               vel   - vel
      //       n+1      n  gamma-1.0      (0)
      //    acc    = acc * --------- + ------------
      //       (0)           gamma      gamma * dt
      //
      gen_alpha_update_acceleration();

      // ----------------------------------------------------------------
      // compute values at intermediate time steps
      // ----------------------------------------------------------------
      gen_alpha_intermediate_values();

      forcing_interface_->activate_forcing(true);

      if (myrank_ == 0)
      {
        std::cout << "|\n";
        std::cout << "+------------------------------------------------------------------------+\n";
        std::cout << "+------------------------------------------------------------------------+\n";
        std::cout << "|" << std::endl;
      }
    }
    else
      // set force to zero
      forcing_->put_scalar(0.0);
  }
}


/*----------------------------------------------------------------------*
 | lift'n'drag forces, statistics time sample and output of solution    |
 | and statistics                                              vg 11/08 |
 -----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::statistics_and_output()
{
  // time measurement: output and statistics
  TEUCHOS_FUNC_TIME_MONITOR("      + output and statistics");

  // -------------------------------------------------------------------
  //   add calculated velocity to mean value calculation (statistics)
  // -------------------------------------------------------------------
  if (meshtying_ != nullptr) statisticsmanager_->get_current_velnp(velnp_);

  call_statistics_manager();

  // -------------------------------------------------------------------
  //                        compute flow rates
  // -------------------------------------------------------------------
  compute_flow_rates();

  // -------------------------------------------------------------------
  //                         output of solution
  // -------------------------------------------------------------------
  output();

  // -------------------------------------------------------------------
  //          dumping of turbulence statistics if required
  // -------------------------------------------------------------------
  statisticsmanager_->do_output(*output_, step_);

  // -------------------------------------------------------------------
  // evaluate error for test flows with analytical solutions
  // -------------------------------------------------------------------
  evaluate_error_compared_to_analytical_sol();

  // -------------------------------------------------------------------
  // evaluate divergence u
  // -------------------------------------------------------------------
  evaluate_div_u();
}  // FluidImplicitTimeInt::StatisticsAndOutput

/*----------------------------------------------------------------------*
 | statistics time sample, overloaded in TimIntLoma            bk 12/13 |
 -----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::call_statistics_manager()
{
  // -------------------------------------------------------------------
  //   add calculated velocity to mean value calculation (statistics)
  // -------------------------------------------------------------------
  statisticsmanager_->do_time_sample(step_, 0.0, 0.0, 0.0, 0.0, 0.0);
}

/*----------------------------------------------------------------------*
 | statistics time sample and output of statistics      rasthofer 06/11 |
 -----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::statistics_output()
{
  // -------------------------------------------------------------------
  //   add calculated velocity to mean value calculation (statistics)
  // -------------------------------------------------------------------
  call_statistics_manager();

  // -------------------------------------------------------------------
  //          dumping of turbulence statistics if required
  // -------------------------------------------------------------------
  statisticsmanager_->do_output(*output_, step_, true);

  if (params_->get<bool>("GMSH_OUTPUT")) output_to_gmsh(step_, time_, true);
}  // FluidImplicitTimeInt::StatisticsOutput

void FLD::FluidImplicitTimeInt::write_runtime_output()
{
  runtime_output_writer_->reset();

  if (runtime_output_params_.output_velocity_state())
  {
    std::vector<std::optional<std::string>> context(3, "velocity");
    context.emplace_back(std::nullopt);
    runtime_output_writer_->append_result_data_vector_with_context(
        *velnp_, Core::IO::OutputEntity::dof, context);
  }

  if (runtime_output_params_.output_pressure_state())
  {
    std::vector<std::optional<std::string>> context(3, std::nullopt);
    context.emplace_back("pressure");
    runtime_output_writer_->append_result_data_vector_with_context(
        *velnp_, Core::IO::OutputEntity::dof, context);
  }

  if (runtime_output_params_.output_acceleration_state())
  {
    std::vector<std::optional<std::string>> context(3, "acceleration");
    runtime_output_writer_->append_result_data_vector_with_context(
        *accnp_, Core::IO::OutputEntity::dof, context);
  }

  if (alefluid_)
  {
    if (runtime_output_params_.output_displacement_state())
    {
      std::vector<std::optional<std::string>> context(3, "displacement");
      runtime_output_writer_->append_result_data_vector_with_context(
          *dispnp_, Core::IO::OutputEntity::dof, context);
    }

    if (runtime_output_params_.output_grid_velocity_state())
    {
      std::vector<std::optional<std::string>> context(3, "grid-velocity");
      runtime_output_writer_->append_result_data_vector_with_context(
          *gridvn_, Core::IO::OutputEntity::dof, context);
    }
  }

  if (runtime_output_params_.output_element_owner())
    runtime_output_writer_->append_element_owner("element_owner");

  if (runtime_output_params_.output_element_gid())
    runtime_output_writer_->append_element_gid("element_gid");

  if (runtime_output_params_.output_node_gid()) runtime_output_writer_->append_node_gid("node_gid");

  // finalize everything and write all required files to filesystem
  runtime_output_writer_->write_to_disk(time_, step_);
}
/*----------------------------------------------------------------------*
 | output of solution vector to binio                        gammi 04/07|
 | overloaded in TimIntPoro                                             |
 | overloaded in TimIntRedModels                               bk 12/13|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::output()
{
  // output of solution
  if (upres_ > 0 and step_ % upres_ == 0)
  {
    if (runtime_output_writer_ != nullptr) write_runtime_output();
    // step number and time
    output_->new_step(step_, time_);

    // time step, especially necessary for adaptive dt
    output_->write_double("timestep", dta_);

    // velocity/pressure vector
    output_->write_vector("velnp", velnp_);

    // (hydrodynamic) pressure
    std::shared_ptr<Core::LinAlg::Vector<double>> pressure =
        velpressplitter_->extract_cond_vector(*velnp_);
    output_->write_vector("pressure", pressure);

    if (xwall_ != nullptr)
    {
      output_->write_vector("xwall_enrvelnp", xwall_->get_output_vector(*velnp_));
      output_->write_vector("xwall_tauw", xwall_->get_tauw_vector());
    }

    if (params_->get<bool>("GMSH_OUTPUT")) output_to_gmsh(step_, time_, false);

    if (alefluid_) output_->write_vector("dispnp", dispnp_);

    if (physicaltype_ == Inpar::FLUID::varying_density or
        physicaltype_ == Inpar::FLUID::boussinesq or physicaltype_ == Inpar::FLUID::tempdepwater)
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> scalar_field =
          velpressplitter_->extract_cond_vector(*scaaf_);
      output_->write_vector("scalar_field", scalar_field);
    }

    // only perform stress calculation when output is needed
    if (writestresses_)
    {
      output_->write_vector("traction", stressmanager_->get_pre_calc_stresses(*trueresidual_));
    }
    // only perform wall shear stress calculation when output is needed
    if (write_wall_shear_stresses_ && xwall_ == nullptr)
    {
      output_->write_vector(
          "wss", stressmanager_->get_pre_calc_wall_shear_stresses(*trueresidual_));
    }

    // biofilm growth
    if (fldgrdisp_ != nullptr)
    {
      output_->write_vector("fld_growth_displ", fldgrdisp_);
    }

    if (params_->get<bool>("COMPUTE_EKIN")) write_output_kinetic_energy();

    // write domain decomposition for visualization (only once!)
    output_->write_element_data(true);

    if (step_ <= 1 and write_nodedata_first_step_) output_->write_node_data(true);

    if (uprestart_ != 0 && step_ % uprestart_ == 0)  // add restart data
    {
      // acceleration vector at time n+1 and n, velocity/pressure vector at time n and n-1
      output_->write_vector("accnp", accnp_);
      output_->write_vector("accn", accn_);
      output_->write_vector("veln", veln_);
      output_->write_vector("velnm", velnm_);

      if (alefluid_)
      {
        output_->write_vector("dispn", dispn_);
        output_->write_vector("dispnm", dispnm_);
        output_->write_vector("gridvn", gridvn_);
      }

      if (xwall_ != nullptr)
        output_->write_vector("wss", stressmanager_->get_pre_calc_wall_shear_stresses(
                                         *xwall_->fix_dirichlet_inflow(*trueresidual_)));

      // flow rate, flow volume and impedance in case of flow-dependent pressure bc
      if (nonlinearbc_) output_nonlinear_bc();

      output_external_forces();

      // write mesh in each restart step --- the elements are required since
      // they contain history variables (the time dependent subscales)
      // But never do this for step 0 (visualization of initial field) since
      // it would lead to writing the mesh twice for step 0
      // (FluidBaseAlgorithm already wrote the mesh) -> HDF5 writer will claim!
      if ((step_ != 0) and (Teuchos::getIntegralValue<Inpar::FLUID::SubscalesTD>(
                                params_->sublist("RESIDUAL-BASED STABILIZATION"), "TDS") !=
                               Inpar::FLUID::SubscalesTD::subscales_quasistatic))
        output_->write_mesh(step_, time_);

      if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
        std::cout << "====== Restart for field '" << discret_->name() << "' written in step "
                  << step_ << std::endl;
    }
  }
  // write restart also when uprestart_ is not a integer multiple of upres_
  else if (uprestart_ > 0 && step_ % uprestart_ == 0)
  {
    // step number and time
    output_->new_step(step_, time_);

    // time step, especially necessary for adaptive dt
    output_->write_double("timestep", dta_);

    // velocity/pressure vector
    output_->write_vector("velnp", velnp_);

    // output_->write_vector("residual", trueresidual_);
    if (alefluid_)
    {
      output_->write_vector("dispnp", dispnp_);
      output_->write_vector("dispn", dispn_);
      output_->write_vector("dispnm", dispnm_);
      output_->write_vector("gridvn", gridvn_);
    }

    // write mesh in each restart step --- the elements are required since
    // they contain history variables (the time dependent subscales)
    // But never do this for step 0 (visualization of initial field) since
    // it would lead to writing the mesh twice for step 0
    // (FluidBaseAlgorithm already wrote the mesh) -> HDF5 writer will claim!
    if ((step_ != 0) and (Teuchos::getIntegralValue<Inpar::FLUID::SubscalesTD>(
                              params_->sublist("RESIDUAL-BASED STABILIZATION"), "TDS") !=
                             Inpar::FLUID::SubscalesTD::subscales_quasistatic))
      output_->write_mesh(step_, time_);

    // only perform stress calculation when output is needed
    if (writestresses_)
    {
      output_->write_vector("traction", stressmanager_->get_pre_calc_stresses(*trueresidual_));
    }
    // only perform wall shear stress calculation when output is needed
    if (write_wall_shear_stresses_ && xwall_ == nullptr)
    {
      output_->write_vector(
          "wss", stressmanager_->get_pre_calc_wall_shear_stresses(*trueresidual_));
    }
    // acceleration vector at time n+1 and n, velocity/pressure vector at time n and n-1
    output_->write_vector("accnp", accnp_);
    output_->write_vector("accn", accn_);
    output_->write_vector("veln", veln_);
    output_->write_vector("velnm", velnm_);

    if (xwall_ != nullptr)
    {
      output_->write_vector("xwall_tauw", xwall_->get_tauw_vector());
      output_->write_vector("wss", stressmanager_->get_pre_calc_wall_shear_stresses(
                                       *xwall_->fix_dirichlet_inflow(*trueresidual_)));
    }

    // flow rate, flow volume and impedance in case of flow-dependent pressure bc
    if (nonlinearbc_) output_nonlinear_bc();

    output_external_forces();
  }

  // -------------------------------------------------------------------
  // calculate and write lift'n'drag forces from the residual
  // -------------------------------------------------------------------
  lift_drag();
}  // FluidImplicitTimeInt::Output


//*----------------------------------------------------------------------*
// | output of solution vector for nonlinear BCs              thon  09/14|
// *---------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::output_nonlinear_bc()
{
  std::vector<const Core::Conditions::Condition*> flowdeppressureline;
  discret_->get_condition("LineFlowDepPressure", flowdeppressureline);
  std::vector<const Core::Conditions::Condition*> flowdeppressuresurf;
  discret_->get_condition("SurfaceFlowDepPressure", flowdeppressuresurf);

  if (flowdeppressureline.size() != 0 or flowdeppressuresurf.size() != 0)
  {
    for (size_t i = 0; i < flowratenp_.size(); ++i)
    {
      std::ostringstream ss;
      ss << "flowratenp" << i;
      output_->write_double(ss.str(), flowratenp_[i]);

      ss.str("");
      ss << "flowraten" << i;
      output_->write_double(ss.str(), flowraten_[i]);

      ss.str("");
      ss << "flowratenm" << i;
      output_->write_double(ss.str(), flowratenm_[i]);

      ss.str("");
      ss << "flowvolumenp" << i;
      output_->write_double(ss.str(), flowvolumenp_[i]);

      ss.str("");
      ss << "flowvolumen" << i;
      output_->write_double(ss.str(), flowvolumen_[i]);

      ss.str("");
      ss << "flowvolumenm" << i;
      output_->write_double(ss.str(), flowvolumenm_[i]);
    }
  }
  if (isimpedancebc_)
  {
    // write restart also when uprestart_ is not a integer multiple of upres_
    // also write impedance bc information if required
    // Note: this method acts only if there is an impedance BC
    impedancebc_->write_restart(*output_);
  }
}

void FLD::FluidImplicitTimeInt::output_to_gmsh(
    const int step, const double time, const bool inflow) const
{
  // turn on/off screen output for writing process of Gmsh postprocessing file
  const bool screen_out = true;

  // create Gmsh postprocessing file
  // 20 steps are kept
  std::string filename = "dummy";
  if (inflow)
  {
    filename = Core::IO::Gmsh::get_new_file_name_and_delete_old_files("solution_velpres_inflow",
        discret_->writer()->output().file_name(), step, 20, screen_out,
        Core::Communication::my_mpi_rank(discret_->get_comm()));
    // std::ofstream gmshfilecontent(filename.c_str());
  }
  else
  {
    filename = Core::IO::Gmsh::get_new_file_name_and_delete_old_files("solution_velpres",
        discret_->writer()->output().file_name(), step, 20, screen_out,
        Core::Communication::my_mpi_rank(discret_->get_comm()));
    // std::ofstream gmshfilecontent(filename.c_str());
  }
  std::ofstream gmshfilecontent(filename.c_str());

  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" "
                    << "velocity solution \" {" << std::endl;
    Core::IO::Gmsh::velocity_pressure_field_dof_based_to_gmsh(
        *discret_, velnp_, "velocity", gmshfilecontent);
    gmshfilecontent << "};" << std::endl;
  }
  {
    // add 'View' to Gmsh postprocessing file
    gmshfilecontent << "View \" "
                    << "pressure solution\" {" << std::endl;
    Core::IO::Gmsh::velocity_pressure_field_dof_based_to_gmsh(
        *discret_, velnp_, "pressure", gmshfilecontent);
    gmshfilecontent << "};" << std::endl;
  }

  gmshfilecontent.close();
  if (screen_out) std::cout << " done" << std::endl;
}


/*----------------------------------------------------------------------*
 | output of external forces for restart                     ghamm 12/14|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::output_external_forces()
{
  if (external_loads_ != nullptr)
  {
    output_->write_int("have_fexternal", external_loads_->global_length());
    output_->write_vector("fexternal", external_loads_);
  }
  else
  {
    output_->write_int("have_fexternal", -1);
  }
}


/*----------------------------------------------------------------------*
 |                                                             kue 04/07|
 -----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::read_restart(int step)
{
  Core::IO::DiscretizationReader reader(
      *discret_, Global::Problem::instance()->input_control_file(), step);
  time_ = reader.read_double("time");
  step_ = reader.read_int("step");

  // recover time step if adaptive time stepping is used
  if (cfl_ > 0.0)
  {
    dta_ = reader.read_double("timestep");
    dtp_ = dta_;
  }

  reader.read_vector(velnp_, "velnp");
  reader.read_vector(veln_, "veln");
  reader.read_vector(velnm_, "velnm");
  reader.read_vector(accnp_, "accnp");
  reader.read_vector(accn_, "accn");

  // set element time parameter after restart:
  // Here it is already needed by AVM3 and impedance boundary condition!!
  set_element_time_parameter();

  statisticsmanager_->read_restart(reader, step);

  if ((fssgv_ != Inpar::FLUID::no_fssgv) or (scale_sep_ == FLUID::algebraic_multigrid_operator))
  {
    avm3_preparation();
  }

  if (xwall_ != nullptr) xwall_->read_restart(reader);

  if (alefluid_)
  {
    reader.read_vector(dispnp_, "dispnp");
    reader.read_vector(dispn_, "dispn");
    reader.read_vector(dispnm_, "dispnm");
    reader.read_vector(gridvn_, "gridvn");
  }

  // flow rate and flow volume in case of flow-dependent pressure bc
  if (nonlinearbc_)
  {
    std::vector<const Core::Conditions::Condition*> flowdeppressureline;
    discret_->get_condition("LineFlowDepPressure", flowdeppressureline);
    std::vector<const Core::Conditions::Condition*> flowdeppressuresurf;
    discret_->get_condition("SurfaceFlowDepPressure", flowdeppressuresurf);

    if (flowdeppressureline.size() != 0 or flowdeppressuresurf.size() != 0)
    {
      for (size_t i = 0; i < flowratenp_.size(); ++i)
      {
        std::ostringstream ss;
        ss << "flowratenp" << i;
        flowratenp_[i] = reader.read_double(ss.str());

        ss.str("");
        ss << "flowraten" << i;
        flowraten_[i] = reader.read_double(ss.str());

        ss.str("");
        ss << "flowratenm" << i;
        flowratenm_[i] = reader.read_double(ss.str());

        ss.str("");
        ss << "flowvolumenp" << i;
        flowvolumenp_[i] = reader.read_double(ss.str());

        ss.str("");
        ss << "flowvolumen" << i;
        flowvolumen_[i] = reader.read_double(ss.str());

        ss.str("");
        ss << "flowvolumenm" << i;
        flowvolumenm_[i] = reader.read_double(ss.str());
      }
    }

    if (isimpedancebc_)
    {
      // also read impedance bc information if required
      // Note: this method acts only if there is an impedance BC
      impedancebc_->read_restart(reader);
    }
  }

  // check whether external forces were written
  const int have_fexternal = reader.read_int("have_fexternal");
  if (have_fexternal != -1)
  {
    external_loads_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*discret_->dof_row_map(), true);
    reader.read_vector(external_loads_, "fexternal");
    if (have_fexternal != external_loads_->global_length())
      FOUR_C_THROW("reading of external loads failed");
  }

  // read the previously written elements including the history data
  // only available+required for time-dependent subgrid scales!
  if (Teuchos::getIntegralValue<Inpar::FLUID::SubscalesTD>(
          params_->sublist("RESIDUAL-BASED STABILIZATION"), "TDS") !=
      Inpar::FLUID::SubscalesTD::subscales_quasistatic)
    reader.read_history_data(step_);

  // ensure that the overall dof numbering is identical to the one
  // that was used when the restart data was written. Especially
  // in case of multiphysics problems & periodic boundary conditions
  // it is better to check the consistency of the maps here:
  if (not(discret_->dof_row_map())->same_as(velnp_->get_map()))
    FOUR_C_THROW("Global dof numbering in maps does not match");
  if (not(discret_->dof_row_map())->same_as(veln_->get_map()))
    FOUR_C_THROW("Global dof numbering in maps does not match");
  if (not(discret_->dof_row_map())->same_as(accn_->get_map()))
    FOUR_C_THROW("Global dof numbering in maps does not match");
}


/*----------------------------------------------------------------------*
 |set restart values (turbulent inflow only)             rasthofer 06/11|
 -----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::set_restart(const int step, const double time,
    std::shared_ptr<const Core::LinAlg::Vector<double>> readvelnp,
    std::shared_ptr<const Core::LinAlg::Vector<double>> readveln,
    std::shared_ptr<const Core::LinAlg::Vector<double>> readvelnm,
    std::shared_ptr<const Core::LinAlg::Vector<double>> readaccnp,
    std::shared_ptr<const Core::LinAlg::Vector<double>> readaccn)
{
  time_ = time;
  step_ = step;

  velnp_->update(1.0, *readvelnp, 0.0);
  veln_->update(1.0, *readveln, 0.0);
  velnm_->update(1.0, *readvelnm, 0.0);
  accnp_->update(1.0, *readaccnp, 0.0);
  accn_->update(1.0, *readaccn, 0.0);

  if ((fssgv_ != Inpar::FLUID::no_fssgv) or (scale_sep_ == FLUID::algebraic_multigrid_operator))
  {
    set_element_time_parameter();
    avm3_preparation();
  }
}


/*----------------------------------------------------------------------*
 |                                                           chfoe 01/08|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::update_gridv()
{
  // get order of accuracy of grid velocity determination
  // from input file data
  const Teuchos::ParameterList& fluiddynparams =
      Global::Problem::instance()->fluid_dynamic_params();
  const auto gridvel = Teuchos::getIntegralValue<Inpar::FLUID::Gridvel>(fluiddynparams, "GRIDVEL");

  switch (gridvel)
  {
    case Inpar::FLUID::BE:
      /* get gridvelocity from BE time discretisation of mesh motion:
           -> cheap
           -> easy
           -> limits FSI algorithm to first order accuracy in time

                  x^n+1 - x^n
             uG = -----------
                    Delta t                        */
      gridv_->update(1 / dta_, *dispnp_, -1 / dta_, *dispn_, 0.0);
      break;
    case Inpar::FLUID::BDF2:
      /* get gridvelocity from BDF2 time discretisation of mesh motion:
           -> requires one more previous mesh position or displacement
           -> somewhat more complicated
           -> allows second order accuracy for the overall flow solution  */
      gridv_->update(1.5 / dta_, *dispnp_, -2.0 / dta_, *dispn_, 0.0);
      gridv_->update(0.5 / dta_, *dispnm_, 1.0);
      break;
    case Inpar::FLUID::OST:
    {
      /* get gridvelocity from OST time discretisation of mesh motion:
         -> needed to allow consistent linearization of FPSI problem  */
      const double theta = fluiddynparams.get<double>("THETA");
      gridv_->update(1 / (theta * dta_), *dispnp_, -1 / (theta * dta_), *dispn_, 0.0);
      gridv_->update(-((1.0 / theta) - 1.0), *gridvn_, 1.0);
    }
    break;
  }

  // Set proper grid velocities at the free-surface and for the ale update conditions
  std::shared_ptr<Core::LinAlg::Vector<double>> auveln =
      surfacesplitter_->extract_au_cond_vector(*veln_);
  surfacesplitter_->insert_au_cond_vector(*auveln, *gridv_);
}


/*----------------------------------------------------------------------*
 | prepare AVM3-based scale separation                         vg 10/08 |
 | overloaded in TimIntRedModels and TimIntLoma               bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::avm3_preparation()
{
  // AVM3 can't be used with locsys conditions cause it hasn't been implemented yet
  if (locsysman_ != nullptr)
  {
    FOUR_C_THROW("AVM3 can't be used with locsys conditions cause it hasn't been implemented yet!");
  }

  if (msht_ == Inpar::FLUID::condensed_bmat || msht_ == Inpar::FLUID::condensed_bmat_merged)
    FOUR_C_THROW(
        "Scale separation via aggregation is currently not implemented for block matrices");

  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("           + avm3");

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  // necessary here, because some application time integrations add something to the residual
  // before the Neumann loads are added
  residual_->put_scalar(0.0);

  // Maybe this needs to be inserted in case of impedanceBC + AVM3
  //  if (nonlinearbc_ && isimpedancebc_)
  //  {
  //    // add impedance Neumann loads
  //    impedancebc_->update_residual(residual_);
  //  }

  avm3_assemble_mat_and_rhs(eleparams);

  // get scale-separation matrix
  avm3_get_scale_separation_matrix();
}  // FluidImplicitTimeInt::avm3_preparation


/*----------------------------------------------------------------------*
 | prepare AVM3-based scale separation:                        vg 10/08 |
 | assemble mat and rhs                                                 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::avm3_assemble_mat_and_rhs(Teuchos::ParameterList& eleparams)
{
  // zero matrix
  sysmat_->zero();

  // add Neumann loads
  // has been set to zero before
  residual_->update(1.0, *neumann_loads_, 1.0);

  // set action type
  eleparams.set<FLD::Action>("action", FLD::calc_fluid_systemmat_and_residual);
  eleparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype_);

  // parameters for turbulence approach
  eleparams.sublist("TURBULENCE MODEL") = params_->sublist("TURBULENCE MODEL");

  // set xwall params
  if (xwall_ != nullptr) xwall_->set_x_wall_params(eleparams);

  // set general vector values needed by elements
  discret_->clear_state();
  discret_->set_state("hist", *hist_);
  discret_->set_state("veln", *veln_);
  discret_->set_state("accam", *accam_);
  // this vector contains only zeros unless SetIterScalarFields is called
  // as this function has not been called yet
  // we have to replace the zeros by ones
  // otherwise nans are occur
  scaaf_->put_scalar(1.0);
  discret_->set_state("scaaf", *scaaf_);
  scaam_->put_scalar(1.0);
  discret_->set_state("scaam", *scaam_);

  // set fine-scale vector
  // dummy vector initialized with zeros
  // Remark:
  // This is necessary because the fssgv_ flag
  // has already been set in SetParameters()
  // Therefore, the function evaluate() already
  // expects the state vector "fsvelaf" and "fsscaaf" for loma
  if (fssgv_ != Inpar::FLUID::no_fssgv or turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales)
  {
    discret_->set_state("fsvelaf", *fsvelaf_);
    if (turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales)
      discret_->set_state("fsscaaf", *fsscaaf_);
  }

  if (alefluid_)
  {
    discret_->set_state(ndsale_, "dispnp", *dispnp_);
    discret_->set_state(ndsale_, "gridv", *gridv_);
  }

  // set scheme-specific element parameters and vector values
  // set the only required state vectors
  set_state_tim_int();

  // set external volume force (required, e.g., for forced homogeneous isotropic turbulence)
  // dummy
  if (forcing_ != nullptr)
  {
    eleparams.set("forcing", true);
    discret_->set_state("forcing", *forcing_);
  }

  // element evaluation for getting system matrix
  // -> we merely need matrix "structure" below, not the actual contents
  discret_->evaluate(eleparams, sysmat_, nullptr, residual_, nullptr, nullptr);
  discret_->clear_state();
  // reset the vector modified above
  scaaf_->put_scalar(0.0);
  scaam_->put_scalar(0.0);

  // complete system matrix
  sysmat_->complete();

  // apply DBC to system matrix
  Core::LinAlg::apply_dirichlet_to_system(
      *sysmat_, *incvel_, *residual_, *zeros_, *(dbcmaps_->cond_map()));
}


/*----------------------------------------------------------------------*
| prepare AVM3-based scale separation:                                  |
| get scale separation matrix                                           |
*----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::avm3_get_scale_separation_matrix()
{
  Teuchos::ParameterList params;
  Core::LinearSolver::Parameters::compute_solver_parameters(*discret_, params);

  // get nullspace parameters
  auto nullspace = params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace");

  // get plain aggregation Ptent
  Core::LinAlg::SparseMatrix Ptent =
      Core::LinAlg::create_interpolation_matrix(*system_matrix(), *nullspace, params);

  // compute scale-separation matrix: S = I - Ptent*Ptent^T
  Sep_ = Core::LinAlg::matrix_multiply(Ptent, false, Ptent, true);
  Sep_->scale(-1.0);
  std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
      std::make_shared<Core::LinAlg::Vector<double>>(Sep_->row_map(), false);
  tmp->put_scalar(1.0);
  std::shared_ptr<Core::LinAlg::Vector<double>> diag =
      std::make_shared<Core::LinAlg::Vector<double>>(Sep_->row_map(), false);
  Sep_->extract_diagonal_copy(*diag);
  diag->update(1.0, *tmp, 1.0);
  // Hint: replace_diagonal_values doesn't do anything if nothing in graph before
  Sep_->replace_diagonal_values(*diag);

  // complete scale-separation matrix and check maps
  Sep_->complete(Sep_->domain_map(), Sep_->range_map());
  if (!Sep_->row_map().same_as(system_matrix()->row_map())) FOUR_C_THROW("rowmap not equal");
  if (!Sep_->range_map().same_as(system_matrix()->range_map())) FOUR_C_THROW("rangemap not equal");
  if (!Sep_->domain_map().same_as(system_matrix()->domain_map()))
    FOUR_C_THROW("domainmap not equal");
}

/*----------------------------------------------------------------------*
 | AVM3-based scale separation                                          |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::avm3_separation()
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("           + avm3");

  // get fine-scale part of velocity at time n+alpha_F or n+1
  sep_multiply();

  // set fine-scale vector
  discret_->set_state("fsvelaf", *fsvelaf_);
}  // FluidImplicitTimeInt::avm3_separation


/*----------------------------------------------------------------------*
 |  set initial flow field for test cases                               |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::set_initial_flow_field(
    const Inpar::FLUID::InitialField initfield, const int startfuncno)
{
  // initial field by (undisturbed) function (init==2)
  // or disturbed function (init==3)
  if (initfield == Inpar::FLUID::initfield_field_by_function or
      initfield == Inpar::FLUID::initfield_disturbed_field_from_function)
  {
    for (auto lnode : discret_->my_row_node_range())
    {
      // the set of degrees of freedom associated with the node
      const std::vector<int> nodedofset = discret_->dof(0, lnode);

      for (int index = 0; index < numdim_ + 1; ++index)
      {
        int gid = nodedofset[index];

        double initialval = Global::Problem::instance()
                                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(startfuncno)
                                .evaluate(lnode.x(), time_, index);

        velnp_->replace_global_value(gid, initialval);
      }
    }

    // for NURBS discretizations we have to solve a least squares problem,
    // with high accuracy! (do nothing for Lagrangian polynomials)
    Teuchos::ParameterList solver_params;
    if (!dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(discret_.get()))
    {
      const Teuchos::ParameterList& nurbs_params = Global::Problem::instance()->nurbs_params();

      const bool is_ls_dbc_needed = nurbs_params.get<bool>("DO_LS_DBC_PROJECTION");

      if (is_ls_dbc_needed)
      {
        const int ls_dbc_solver_num = nurbs_params.get<int>("SOLVER_LS_DBC_PROJECTION");

        if (ls_dbc_solver_num == (-1))
          FOUR_C_THROW(
              "No linear solver defined for the projection of least squares Dirichlet "
              "boundary conditions for the NURBS discretization. Please set "
              "SOLVER_LS_DBC_PROJECTION in NURBS to a valid number!");

        // Save solver parameters
        solver_params.sublist("ls_dbc_solver_params")
            .setParameters(Global::Problem::instance()->solver_params(ls_dbc_solver_num));
      }

      Core::FE::Nurbs::apply_nurbs_initial_condition(*discret_, solver_params,
          Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(
              startfuncno),
          velnp_);
    }

    // initialize veln_ as well. That's what we actually want to do here!
    veln_->update(1.0, *velnp_, 0.0);

    // add random perturbation of certain percentage to function
    if (initfield == Inpar::FLUID::initfield_disturbed_field_from_function)
    {
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

      // random noise is perc percent of the initial profile
      double perc = params_->sublist("TURBULENCE MODEL").get<double>("CHAN_AMPL_INIT_DIST", 0.1);

      // out to screen
      if (myrank_ == 0)
      {
        std::cout << "Disturbed initial profile:   max. " << perc * 100
                  << "% random perturbation\n";
        std::cout << "\n\n";
      }

      double bmvel = 0;
      double mybmvel = 0;
      double thisvel = 0;
      // loop all nodes on the processor
      for (int lnodeid = 0; lnodeid < discret_->num_my_row_nodes(); ++lnodeid)
      {
        // get the processor local node
        Core::Nodes::Node* lnode = discret_->l_row_node(lnodeid);
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->dof(lnode);

        for (int index = 0; index < numdim_; ++index)
        {
          int gid = nodedofset[index];
          int lid = dofrowmap->lid(gid);

          thisvel = velnp_->local_values_as_span()[lid];
          if (mybmvel * mybmvel < thisvel * thisvel) mybmvel = thisvel;
        }
      }

      // the noise is proportional to the bulk mean velocity of the
      // undisturbed initial field (=2/3*maximum velocity)
      mybmvel = (2.0 / 3.0) * mybmvel;
      bmvel = Core::Communication::max_all(mybmvel, discret_->get_comm());

      // loop all nodes on the processor
      for (int lnodeid = 0; lnodeid < discret_->num_my_row_nodes(); ++lnodeid)
      {
        // get the processor local node
        Core::Nodes::Node* lnode = discret_->l_row_node(lnodeid);
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->dof(lnode);

        // check whether we have a pbc condition on this node
        std::vector<const Core::Conditions::Condition*> mypbc =
            discret_->get_conditions_on_node("SurfacePeriodic", lnode);

        // check whether a periodic boundary condition is active on this node
        if (mypbc.size() > 0)
        {
          // yes, we have one

          // get the list of all its slavenodes
          auto master = (discret_->get_all_pbc_coupled_col_nodes())->find(lnode->id());

          // slavenodes are ignored
          if (master == (discret_->get_all_pbc_coupled_col_nodes())->end()) continue;
        }

        // add random noise on initial function field
        for (int index = 0; index < numdim_; ++index)
        {
          int gid = nodedofset[index];

          double randomnumber = Global::Problem::instance()->random()->uni();

          double noise = perc * bmvel * randomnumber;

          velnp_->sum_into_global_values(1, &noise, &gid);
          veln_->sum_into_global_values(1, &noise, &gid);
        }
      }
      // meshtying: this is necessary for the disturbed field. the interface does not work
      // otherwise due to the non-linear terms in the matrix.
      if (msht_ != Inpar::FLUID::no_meshtying)
      {
        meshtying_->update_slave_dof(*velnp_, *velnp_);
        meshtying_->update_slave_dof(*veln_, *veln_);
      }
    }
  }
  // special initial function: two counter-rotating vortices (2-D) and flame front
  // for flame-vortex interaction problem
  else if (initfield == Inpar::FLUID::initfield_flame_vortex_interaction)
  {
    const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

    int err = 0;

    // define vectors for velocity field, node coordinates and coordinates
    // of left and right vortex
    std::vector<double> u(numdim_);
    std::vector<double> xy(numdim_);
    std::vector<double> xy0_left(numdim_);
    std::vector<double> xy0_right(numdim_);

    // check whether present flow is indeed two-dimensional
    if (numdim_ != 2) FOUR_C_THROW("Counter-rotating vortices are a two-dimensional flow!");

    // set laminar burning velocity, vortex strength C (scaled by laminar
    // burning velocity and (squared) vortex radius R
    const double sl = 1.0;
    const double C = 70.0 * sl;
    const double R_squared = 16.0;

    // set density in unburnt and burnt phase and initialize actual density
    const double densu = 1.161;
    // -> for "pure fluid" computation: rhob = rhou = 1.161
    // const double densb = 1.161;
    const double densb = 0.157;
    double dens = 1.161;

    // initialize progress variable
    double pv = 0.0;

    // variables for evaluation of progress-variable profile
    // locations separating region 1 from region 2 and region 2 from region 3
    const double loc12 = 98.5;
    const double loc23 = 103.0;

    // define parameters for region 1 (exponential function for curve fitting)
    const double beta1 = 1.65;
    const double delta1 = 1.0;
    const double trans1 = 100.0;

    // define parameters for region 2 (linear function for curve fitting)
    const double abs2 = 0.0879;
    const double fac2 = 0.139309333;
    const double trans2 = 98.5;

    // define parameters for region 3 (exponential function for curve fitting)
    const double beta3 = 3.506209;
    const double delta3 = 4.28875;
    const double trans3 = 103.0;

    // set (scaled) vortex strength C, (squared) vortex radius R and define variables
    double r_squared_left;
    double r_squared_right;

    // set initial locations of vortices
    xy0_left[0] = 37.5;
    xy0_left[1] = 75.0;
    xy0_right[0] = 62.5;
    xy0_right[1] = 75.0;

    for (auto lnode : discret_->my_row_node_range())
    {
      // the set of degrees of freedom associated with the node
      std::vector<int> nodedofset = discret_->dof(0, lnode);

      // set node coordinates
      for (int dim = 0; dim < numdim_; dim++)
      {
        xy[dim] = lnode.x()[dim];
      }

      // compute preliminary values for both vortices
      r_squared_left = ((xy[0] - xy0_left[0]) * (xy[0] - xy0_left[0]) +
                           (xy[1] - xy0_left[1]) * (xy[1] - xy0_left[1])) /
                       R_squared;
      r_squared_right = ((xy[0] - xy0_right[0]) * (xy[0] - xy0_right[0]) +
                            (xy[1] - xy0_right[1]) * (xy[1] - xy0_right[1])) /
                        R_squared;

      // compute value of progress variable
      if (xy[1] < loc12 - 1e-10)
        pv = (1.0 - (1.0 / beta1)) * exp((xy[1] - trans1) / delta1);
      else if (xy[1] > loc23 + 1e-10)
        pv = 1.0 - (exp((1.0 - beta3) * (xy[1] - trans3) / delta3) / beta3);
      else
        pv = fac2 * (xy[1] - trans2) + abs2;

      // compute current density
      dens = densu + (densb - densu) * pv;

      // compute initial velocity components
      // including initial velocity distribution velocity in x2-direction
      u[0] = (C / R_squared) * (-(xy[1] - xy0_left[1]) * exp(-r_squared_left / 2.0) +
                                   (xy[1] - xy0_right[1]) * exp(-r_squared_right / 2.0));
      u[1] = (C / R_squared) * ((xy[0] - xy0_left[0]) * exp(-r_squared_left / 2.0) -
                                   (xy[0] - xy0_right[0]) * exp(-r_squared_right / 2.0)) +
             sl * densu / dens;

      // velocity profile due to flame without vortices:
      // u[1] = sl*densu/dens;

      // set initial velocity components
      for (int nveldof = 0; nveldof < numdim_; nveldof++)
      {
        const int gid = nodedofset[nveldof];
        int lid = dofrowmap->lid(gid);
        velnp_->replace_local_value(lid, u[nveldof]);
        veln_->replace_local_value(lid, u[nveldof]);
        velnm_->replace_local_value(lid, u[nveldof]);
      }
    }  // end loop nodes lnodeid

    if (err != 0) FOUR_C_THROW("dof not on proc");
  }
  // special initial function: Beltrami flow (3-D)
  else if (initfield == Inpar::FLUID::initfield_beltrami_flow)
  {
    const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

    const int npredof = numdim_;

    double p;
    std::vector<double> u(numdim_);
    std::vector<double> acc(numdim_);
    std::vector<double> xyz(numdim_);

    // check whether present flow is indeed three-dimensional
    if (numdim_ != 3) FOUR_C_THROW("Beltrami flow is a three-dimensional flow!");

    // set constants for analytical solution
    const double a = std::numbers::pi / 4.0;
    const double d = std::numbers::pi / 2.0;

    for (auto lnode : discret_->my_row_node_range())
    {
      // the set of degrees of freedom associated with the node
      std::vector<int> nodedofset = discret_->dof(0, lnode);

      // set node coordinates
      for (int dim = 0; dim < numdim_; dim++)
      {
        xyz[dim] = lnode.x()[dim];
      }

      // compute initial velocity components
      u[0] = -a * (exp(a * xyz[0]) * sin(a * xyz[1] + d * xyz[2]) +
                      exp(a * xyz[2]) * cos(a * xyz[0] + d * xyz[1]));
      u[1] = -a * (exp(a * xyz[1]) * sin(a * xyz[2] + d * xyz[0]) +
                      exp(a * xyz[0]) * cos(a * xyz[1] + d * xyz[2]));
      u[2] = -a * (exp(a * xyz[2]) * sin(a * xyz[0] + d * xyz[1]) +
                      exp(a * xyz[1]) * cos(a * xyz[2] + d * xyz[0]));

      // compute initial pressure
      int id = Global::Problem::instance()->materials()->first_id_by_type(Core::Materials::m_fluid);
      if (id == -1) FOUR_C_THROW("Newtonian fluid material could not be found");
      const Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance()->materials()->parameter_by_id(id);
      const auto* actmat = static_cast<const Mat::PAR::NewtonianFluid*>(mat);
      double dens = actmat->density_;
      double visc = actmat->viscosity_;

      p = -a * a / 2.0 * dens *
          (exp(2.0 * a * xyz[0]) + exp(2.0 * a * xyz[1]) + exp(2.0 * a * xyz[2]) +
              2.0 * sin(a * xyz[0] + d * xyz[1]) * cos(a * xyz[2] + d * xyz[0]) *
                  exp(a * (xyz[1] + xyz[2])) +
              2.0 * sin(a * xyz[1] + d * xyz[2]) * cos(a * xyz[0] + d * xyz[1]) *
                  exp(a * (xyz[2] + xyz[0])) +
              2.0 * sin(a * xyz[2] + d * xyz[0]) * cos(a * xyz[1] + d * xyz[2]) *
                  exp(a * (xyz[0] + xyz[1])));

      // Beltrami is always 3D
      acc[0] = u[0] * (-1.0 * d * d * visc / dens);
      acc[1] = u[1] * (-1.0 * d * d * visc / dens);
      acc[2] = u[2] * (-1.0 * d * d * visc / dens);

      // set initial velocity components
      for (int nveldof = 0; nveldof < numdim_; nveldof++)
      {
        const int gid = nodedofset[nveldof];
        int lid = dofrowmap->lid(gid);
        velnp_->replace_local_value(lid, u[nveldof]);
        veln_->replace_local_value(lid, u[nveldof]);
        velnm_->replace_local_value(lid, u[nveldof]);

        // set additionally the values for the time derivative to start with an exact acceleration
        // in case of OST (theta!=1.0) set initial acceleration components
        accnp_->replace_local_value(lid, acc[nveldof]);
        accn_->replace_local_value(lid, acc[nveldof]);
        accam_->replace_local_value(lid, acc[nveldof]);
      }

      // set initial pressure
      const int gid = nodedofset[npredof];
      int lid = dofrowmap->lid(gid);
      velnp_->replace_local_value(lid, p);
      veln_->replace_local_value(lid, p);
      velnm_->replace_local_value(lid, p);
    }  // end loop nodes lnodeid
  }

  else if (initfield == Inpar::FLUID::initfield_hit_comte_bellot_corrsin or
           initfield == Inpar::FLUID::initfield_forced_hit_simple_algebraic_spectrum or
           initfield == Inpar::FLUID::initfield_forced_hit_numeric_spectrum or
           initfield == Inpar::FLUID::initfield_passive_hit_const_input)
  {
    // initialize calculation of initial field based on fast Fourier transformation
    HomoIsoTurbInitialField HitInitialField(*this, initfield);
    // calculate initial field
    HitInitialField.calculate_initial_field();

    // get statistics of initial field
    call_statistics_manager();

    // initialize  forcing depending on initial field
    forcing_interface_->set_initial_spectrum(initfield);
  }
  else
  {
    FOUR_C_THROW(
        "Only initial fields such as a zero field, initial fields by (un-)disturbed functions, "
        "three special initial fields (counter-rotating vortices, Beltrami flow) "
        "as well as initial fields for homogeneous isotropic turbulence are available up to now!");
  }

}  // end SetInitialFlowField


/*----------------------------------------------------------------------*
 | set fields for scatra - fluid coupling, esp.                         |
 | set fields for low-Mach-number flow within iteration loop   vg 09/09 |
 | overloaded in TimIntLoma                                    bk 12/13 |
 | overloaded in TimIntTwoPhase                                mw 07/14 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::set_iter_scalar_fields(
    std::shared_ptr<const Core::LinAlg::Vector<double>> scalaraf,
    std::shared_ptr<const Core::LinAlg::Vector<double>> scalaram,
    std::shared_ptr<const Core::LinAlg::Vector<double>> scalardtam,
    std::shared_ptr<Core::FE::Discretization> scatradis, int dofset)
{
  // initializations
  double value(0.0);

  //--------------------------------------------------------------------------
  // Filling the scaaf-vector and scaam-vector at time n+alpha_F/n+1 and
  // n+alpha_M/n, respectively, with scalar at pressure dofs
  // Additionally, filling the scaam-vector at time n+alpha_M/n with
  // velocity at time n at velocity dofs for OST/BDF2
  // Filling the accam-vector at time n+alpha_M/n+1, respectively, with
  // scalar time derivative values at pressure dofs
  //--------------------------------------------------------------------------
  // get velocity values at time n in scaam-vector as copy from veln-vector
  scaam_->update(1.0, *veln_, 0.0);

  if (scatradis != nullptr)
  {
    // loop all nodes on the processor
    for (int lnodeid = 0; lnodeid < discret_->num_my_row_nodes(); lnodeid++)
    {
      // get the processor's local scatra node
      Core::Nodes::Node* lscatranode = scatradis->l_row_node(lnodeid);

      // find out the global dof id of the last(!) dof at the scatra node
      const int numscatradof = scatradis->num_dof(dofset, lscatranode);
      const int globalscatradofid = scatradis->dof(dofset, lscatranode, numscatradof - 1);
      const int localscatradofid = scalaraf->get_map().lid(globalscatradofid);
      if (localscatradofid < 0) FOUR_C_THROW("localdofid not found in map for given globaldofid");

      // get the processor's local fluid node
      Core::Nodes::Node* lnode = discret_->l_row_node(lnodeid);
      // get global and processor's local pressure dof id (using the map!)
      const int numdof = discret_->num_dof(0, lnode);
      const int globaldofid = discret_->dof(0, lnode, numdof - 1);
      const int localdofid = scaam_->get_map().lid(globaldofid);
      if (localdofid < 0) FOUR_C_THROW("localdofid not found in map for given globaldofid");

      // now copy the values
      value = scalaraf->local_values_as_span()[localscatradofid];
      scaaf_->replace_local_value(localdofid, value);

      value = scalaram->local_values_as_span()[localscatradofid];
      scaam_->replace_local_value(localdofid, value);

      if (scalardtam != nullptr)
      {
        value = scalardtam->local_values_as_span()[localscatradofid];
      }
      else
      {
        value = 0.0;  // for safety reasons: set zeros in accam_
      }
      accam_->replace_local_value(localdofid, value);
    }
  }
  else
  {
    // given vectors are already in dofrowmap layout of fluid and values can
    // be copied directly
    if (not scalaraf->get_map().same_as(scaaf_->get_map()) or
        not scalaram->get_map().same_as(scaam_->get_map()))
      FOUR_C_THROW("fluid dofrowmap layout expected");

    // loop all nodes on the processor
    for (int lnodeid = 0; lnodeid < discret_->num_my_row_nodes(); lnodeid++)
    {
      // get the processor's local fluid node
      Core::Nodes::Node* lnode = discret_->l_row_node(lnodeid);
      // get global and processor's local pressure dof id (using the map!)
      const int numdof = discret_->num_dof(0, lnode);
      const int globaldofid = discret_->dof(0, lnode, numdof - 1);
      const int localdofid = scaam_->get_map().lid(globaldofid);
      if (localdofid < 0) FOUR_C_THROW("localdofid not found in map for given globaldofid");

      // now copy the values
      value = scalaraf->local_values_as_span()[localdofid];
      scaaf_->replace_local_value(localdofid, value);

      value = scalaram->local_values_as_span()[localdofid];
      scaam_->replace_local_value(localdofid, value);

      if (scalardtam != nullptr)
      {
        value = scalardtam->local_values_as_span()[localdofid];
      }
      else
      {
        value = 0.0;  // for safety reasons: set zeros in accam_
      }
      accam_->replace_local_value(localdofid, value);
    }
  }

}  // FluidImplicitTimeInt::SetIterScalarFields


/*----------------------------------------------------------------------*
 | set fields for scatra - fluid coupling, esp.                         |
 | set scalar fields     vg 09/09 |
 | overloaded in TimIntLoma                                    bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::set_scalar_fields(
    std::shared_ptr<const Core::LinAlg::Vector<double>> scalarnp, const double thermpressnp,
    std::shared_ptr<const Core::LinAlg::Vector<double>> scatraresidual,
    std::shared_ptr<Core::FE::Discretization> scatradis, const int whichscalar)
{
  // initializations
  double value(0.0);
  std::vector<int> nodedofs;

  //--------------------------------------------------------------------------
  // Filling the scaaf-vector with scalar at time n+1 at pressure dofs
  //--------------------------------------------------------------------------
  // loop all nodes on the processor
  for (int lnodeid = 0; lnodeid < discret_->num_my_row_nodes(); lnodeid++)
  {
    // get the processor's local scatra node
    Core::Nodes::Node* lscatranode = scatradis->l_row_node(lnodeid);

    // find out the global dof id of the last(!) dof at the scatra node
    const int numscatradof = scatradis->num_dof(0, lscatranode);
    int globalscatradofid(-1);
    if (whichscalar == (-1))
    {
      // default: always take the LAST scatra dof at each node
      globalscatradofid = scatradis->dof(0, lscatranode, numscatradof - 1);
    }
    else
    {
      // respect the explicit wish of the user
      globalscatradofid = scatradis->dof(0, lscatranode, whichscalar);
    }
    const int localscatradofid = scalarnp->get_map().lid(globalscatradofid);
    if (localscatradofid < 0) FOUR_C_THROW("localdofid not found in map for given globaldofid");

    // get the processor's local fluid node
    Core::Nodes::Node* lnode = discret_->l_row_node(lnodeid);
    // get the global ids of degrees of freedom associated with this node
    nodedofs = discret_->dof(0, lnode);
    // get global and processor's local pressure dof id (using the map!)
    const int globaldofid = nodedofs[numdim_];
    const int localdofid = scaam_->get_map().lid(globaldofid);
    if (localdofid < 0) FOUR_C_THROW("localdofid not found in map for given globaldofid");

    value = scalarnp->local_values_as_span()[localscatradofid];
    scaaf_->replace_local_value(localdofid, value);

    //--------------------------------------------------------------------------
    // Filling the trueresidual vector with scatraresidual at pre-dofs
    //--------------------------------------------------------------------------
    if (scatraresidual != nullptr)
    {
      value = scatraresidual->local_values_as_span()[localscatradofid];
      trueresidual_->replace_local_value(localdofid, value);
    }
  }


}  // FluidImplicitTimeInt::SetScalarFields

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
FLD::FluidImplicitTimeInt::extract_velocity_part(
    std::shared_ptr<const Core::LinAlg::Vector<double>> velpres)
{
  return vel_pres_splitter()->extract_other_vector(*velpres);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
FLD::FluidImplicitTimeInt::extract_pressure_part(
    std::shared_ptr<const Core::LinAlg::Vector<double>> velpres)
{
  return vel_pres_splitter()->extract_cond_vector(*velpres);
}

/*----------------------------------------------------------------------*
 | evaluate error for test cases with analytical solutions   gammi 04/07|
 *----------------------------------------------------------------------*/
std::shared_ptr<std::vector<double>>
FLD::FluidImplicitTimeInt::evaluate_error_compared_to_analytical_sol()
{
  auto calcerr = Teuchos::getIntegralValue<Inpar::FLUID::CalcError>(*params_, "calculate error");

  switch (calcerr)
  {
    case Inpar::FLUID::no:
    {
      // do nothing --- no analytical solution available
      return nullptr;
      break;
    }
    case Inpar::FLUID::beltrami_flow:
    case Inpar::FLUID::channel2D:
    case Inpar::FLUID::gravitation:
    case Inpar::FLUID::shear_flow:
    case Inpar::FLUID::fsi_fluid_pusher:
    case Inpar::FLUID::byfunct:
    case Inpar::FLUID::channel_weakly_compressible:
    {
      // std::vector containing
      // [0]: relative L2 velocity error
      // [1]: relative L2 pressure error
      // [2]: relative H1 velocity error
      std::shared_ptr<std::vector<double>> relerror = std::make_shared<std::vector<double>>(3);

      // create the parameters for the discretization
      Teuchos::ParameterList eleparams;

      // action for elements
      eleparams.set<FLD::Action>("action", FLD::calc_fluid_error);
      eleparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype_);
      eleparams.set<Inpar::FLUID::CalcError>("calculate error", calcerr);

      const int errorfunctno = params_->get<int>("error function number", -1);
      eleparams.set<int>("error function number", errorfunctno);

      // set scheme-specific element parameters and vector values
      set_state_tim_int();

      if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);

      // get (squared) error values
      // 0: delta velocity for L2-error norm
      // 1: delta p for L2-error norm
      // 2: delta velocity for H1-error norm
      // 3: analytical velocity for L2 norm
      // 4: analytical p for L2 norm
      // 5: analytical velocity for H1 norm
      Core::LinAlg::SerialDenseVector errors(3 + 3);

      // call loop over elements (assemble nothing)
      discret_->evaluate_scalars(eleparams, errors);
      discret_->clear_state();

      (*relerror)[0] = sqrt((errors)[0]) / sqrt((errors)[3]);
      (*relerror)[1] = sqrt((errors)[1]) / sqrt((errors)[4]);

      if ((calcerr == Inpar::FLUID::beltrami_flow) or (calcerr == Inpar::FLUID::byfunct))
        (*relerror)[2] = sqrt((errors)[2]) / sqrt((errors)[5]);
      else
      {
        (*relerror)[2] = 0.0;
        if (myrank_ == 0)
        {
          std::cout << std::endl
                    << "Warning: H_1 velocity error norm for analytical solution Nr. "
                    << Teuchos::getIntegralValue<Inpar::FLUID::CalcError>(
                           *params_, "calculate error")
                    << " is not implemented yet!!" << std::endl;
        }
      }

      if (myrank_ == 0)
      {
        {
          std::cout.precision(8);
          std::cout << std::endl
                    << "---- error norm for analytical solution Nr. "
                    << Teuchos::getIntegralValue<Inpar::FLUID::CalcError>(
                           *params_, "calculate error")
                    << " ----------" << std::endl;
          std::cout << "| relative L_2 velocity error norm:     " << (*relerror)[0] << std::endl;
          std::cout << "| relative L_2 pressure error norm:     " << (*relerror)[1] << std::endl;
          if ((*relerror)[2] != 0.0)
            std::cout << "| relative H_1 velocity error norm:     " << (*relerror)[2] << std::endl;
          std::cout << "--------------------------------------------------------------------"
                    << std::endl
                    << std::endl;
          if ((*relerror)[2] != 0.0)
            std::cout << "H1 velocity scaling  " << sqrt((errors)[5]) << std::endl;
        }

        // print last error in a separate file

        // append error of the last time step to the error file
        if ((step_ == stepmax_) or (time_ == maxtime_))  // write results to file
        {
          std::ostringstream temp;
          const std::string simulation =
              Global::Problem::instance()->output_control_file()->file_name();
          const std::string fname = simulation + ".relerror";

          std::ofstream f;
          f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
          f << "#| " << simulation << "\n";
          f << "#| Step | Time | rel. L2-error velocity  |  rel. L2-error pressure  |  rel. "
               "H1-error velocity  |\n";
          f << step_ << " " << time_ << " " << (*relerror)[0] << " " << (*relerror)[1] << " "
            << (*relerror)[2] << "\n";
          f.flush();
          f.close();
        }


        std::ostringstream temp;
        const std::string simulation =
            Global::Problem::instance()->output_control_file()->file_name();
        const std::string fname = simulation + "_time.relerror";

        if (step_ == 1)
        {
          std::ofstream f;
          f.open(fname.c_str());
          f << "#| Step | Time | rel. L2-error velocity  |  rel. L2-error pressure  |  rel. "
               "H1-error velocity  |\n";
          f << std::setprecision(10) << step_ << " " << std::setw(1) << std::setprecision(5)
            << time_ << std::setw(1) << std::setprecision(6) << " " << (*relerror)[0]
            << std::setw(1) << std::setprecision(6) << " " << (*relerror)[1] << std::setprecision(6)
            << " " << (*relerror)[2] << "\n";

          f.flush();
          f.close();
        }
        else
        {
          std::ofstream f;
          f.open(fname.c_str(), std::fstream::ate | std::fstream::app);
          f << std::setprecision(10) << step_ << " " << std::setw(3) << std::setprecision(5)
            << time_ << std::setw(1) << std::setprecision(6) << " " << (*relerror)[0]
            << std::setw(1) << std::setprecision(6) << " " << (*relerror)[1] << std::setprecision(6)
            << " " << (*relerror)[2] << "\n";

          f.flush();
          f.close();
        }
      }
      return relerror;
    }
    break;
    default:
      FOUR_C_THROW("Cannot calculate error. Unknown type of analytical test problem");
      break;
  }
  return nullptr;
}  // end evaluate_error_compared_to_analytical_sol


/*----------------------------------------------------------------------*
 | evaluate divergence u                                      ehrl 12/12|
 *----------------------------------------------------------------------*/
std::shared_ptr<double> FLD::FluidImplicitTimeInt::evaluate_div_u()
{
  // Evaluate div u only at the last step
  // if ((step_==stepmax_) or (time_==maxtime_))// write results to file
  if (params_->get<bool>("COMPUTE_DIVU"))
  {
    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set<FLD::Action>("action", FLD::calc_div_u);

    if (xwall_ != nullptr) xwall_->set_x_wall_params(eleparams);

    // set vector values needed by elements
    // div u is always evaluated at time n+af (generalized alpha time integration schemes) and
    // at time  n+1 (one-step-theta)
    // set scheme-specific element parameters and vector values
    // continuity equation in np-genalpha is also evaluated at time n+1
    set_state_tim_int();

    const Core::LinAlg::Map* elementrowmap = discret_->element_row_map();
    Core::LinAlg::MultiVector<double> divu(*elementrowmap, 1, true);

    // optional: elementwise defined div u may be written to standard output file (not implemented
    // yet)
    discret_->evaluate_scalars(eleparams, divu);

    discret_->clear_state();

    double maxdivu = 0.0;
    std::shared_ptr<double> sumdivu = std::make_shared<double>(0.0);
    divu.norm_1(&(*sumdivu));
    divu.norm_inf(&maxdivu);

    if (myrank_ == 0)
    {
      std::cout << "---------------------------------------------------" << std::endl;
      std::cout << "| divergence-free condition:                      |" << std::endl;
      std::cout << "| Norm(inf) = " << maxdivu << " | Norm(1) = " << *sumdivu << "  |" << std::endl;
      std::cout << "---------------------------------------------------" << std::endl << std::endl;

      const std::string simulation =
          Global::Problem::instance()->output_control_file()->file_name();
      const std::string fname = simulation + ".divu";

      std::ofstream f;
      f.open(fname.c_str());
      if (step_ == 1)
      {
        f << "#| " << simulation << "\n";
        f << "#| Step | Time | max. div u | div u (Norm(1)) |\n";
        f << step_ << " " << time_ << " " << maxdivu << " " << *sumdivu << " "
          << "\n";
        f.flush();
        f.close();
      }
      else
      {
        f << step_ << " " << time_ << " " << maxdivu << " " << *sumdivu << " "
          << "\n";
        f.flush();
        f.close();
      }
    }
    return sumdivu;
  }
  else
    return nullptr;
}  // end EvaluateDivU

/*----------------------------------------------------------------------*
 | calculate adaptive time step with the CFL number             bk 08/14|
 *----------------------------------------------------------------------*/
double FLD::FluidImplicitTimeInt::evaluate_dt_via_cfl_if_applicable()
{
  int stependadaptivedt =
      params_->sublist("TIMEADAPTIVITY").get<int>("FREEZE_ADAPTIVE_DT_AT", 10000000);
  if (step_ + 1 == stependadaptivedt && myrank_ == 0)
  {
    std::cout << "\n    !!time step is kept constant from now on for sampling of turbulence "
                 "statistics!!\n"
              << std::endl;
  }
  if ((cfl_ > 0.0 && step_ + 1 < stependadaptivedt) ||
      (cfl_estimator_ == Inpar::FLUID::only_print_cfl_number))
  {
    // create the parameters for the discretization
    Teuchos::ParameterList eleparams;

    // action for elements
    eleparams.set<FLD::Action>("action", FLD::calc_dt_via_cfl);

    if (xwall_ != nullptr) xwall_->set_x_wall_params(eleparams);

    discret_->set_state("velnp", *velnp_);
    if (alefluid_)
    {
      discret_->set_state(ndsale_, "dispnp", *dispnp_);
      discret_->set_state(ndsale_, "gridv", *gridv_);
    }

    const Core::LinAlg::Map* elementrowmap = discret_->element_row_map();
    Core::LinAlg::Vector<double> h_u(*elementrowmap, true);

    // optional: elementwise defined h_u may be written to standard output file (not implemented
    // yet)
    discret_->evaluate_scalars(eleparams, h_u);

    discret_->clear_state();

    double min_h_u = 0.0;

    h_u.min_value(&min_h_u);

    if (cfl_estimator_ == Inpar::FLUID::only_print_cfl_number && myrank_ == 0)
    {
      if (min_h_u != 0.0) std::cout << "CFL number is: " << dta_ / min_h_u << std::endl;
      return -1.0;
    }

    // if the initial velocity field is zero and there are no non-zero Dirichlet-boundaries,
    // min_h_u is zero. In this case, we use the time step stated in the input file
    // and write this to the screen

    double inc = params_->sublist("TIMEADAPTIVITY").get<double>("ADAPTIVE_DT_INC", 0.8);

    if (min_h_u < 1.0e3)
    {
      if (step_ > 0)
        return dta_ + inc * (cfl_ * min_h_u - dtp_);
      else  // start of simulation
        return cfl_ * min_h_u;
    }
    else if (myrank_ == 0)
    {
      std::cout << "Calculated time step is zero due to zero velocity field: use time step stated "
                   "in input file for the first step!"
                << std::endl;
    }
  }

  return dta_;
}  // end EvaluateDtWithCFL


/*----------------------------------------------------------------------*
 | calculate lift and drag forces as well as angular moment: chfoe 11/07|
 | Lift and drag forces are based upon the right-hand side              |
 | true-residual entities of the corresponding nodes.                   |
 | The contribution of the end node of a line is entirely               |
 | added to a present L&D force.                                        |
 | For computing the angular moment, potential displacements            |
 | are taken into account when calculating the distance to              |
 | the center of rotation.                                              |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::lift_drag() const
{
  // initially check whether computation of lift and drag values is required
  if (params_->get<bool>("LIFTDRAG"))
  {
    // in this map, the results of the lift drag calculation are stored
    std::shared_ptr<std::map<int, std::vector<double>>> liftdragvals;

    // check whether there are slip supplemental curved boundary conditions
    std::vector<const Core::Conditions::Condition*> slipsuppline;
    discret_->get_condition("LineSlipSupp", slipsuppline);
    std::vector<const Core::Conditions::Condition*> slipsuppsurf;
    discret_->get_condition("SurfaceSlipSupp", slipsuppsurf);

    if (slipsuppline.size() != 0 or slipsuppsurf.size() != 0)
    {
      // Create temporary variable that holds a vector containing the real forces
      // acting on the node, i.e. taking into account the previously neglected
      // forces perpendicular to the boundary due to slip boundary condition(s)
      std::shared_ptr<Core::LinAlg::Vector<double>> forces;
      forces = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);

      forces->update(1.0, *trueresidual_, 1.0, *slip_bc_normal_tractions_, 0.0);

      FLD::Utils::lift_drag(discret_, *forces, dispnp_, numdim_, liftdragvals, alefluid_);
    }
    else
    {
      FLD::Utils::lift_drag(discret_, *trueresidual_, dispnp_, numdim_, liftdragvals, alefluid_);
    }

    if (liftdragvals != nullptr and Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
      FLD::Utils::write_lift_drag_to_file(time_, step_, *liftdragvals);
  }
}


/*----------------------------------------------------------------------*
 | compute flow rates through desired boundary parts        u.may 01/10 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::compute_flow_rates() const
{
  std::vector<const Core::Conditions::Condition*> flowratecond;
  std::string condstring;

  if (numdim_ == 2)
  {
    condstring = "LineFlowRate";
    discret_->get_condition("LineFlowRate", flowratecond);
    // if no flowrate condition is present we do not compute anything
    if ((int)flowratecond.size() == 0) return;
  }
  else if (numdim_ == 3)
  {
    condstring = "SurfFlowRate";
    discret_->get_condition("SurfFlowRate", flowratecond);
    // if no flowrate condition is present we do not compute anything
    if ((int)flowratecond.size() == 0) return;
  }
  else
    FOUR_C_THROW("flow rate computation is not implemented for the 1D case");

  if (alefluid_)
  {
    const std::map<int, double> flowrates = FLD::Utils::compute_flow_rates(
        *discret_, *velnp_, gridv_, dispnp_, condstring, physicaltype_);
    // const std::map<int,double> volume = FLD::Utils::compute_volume(*discret_,
    // velnp_,gridv_,dispnp_,physicaltype_);

    // write to file
    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
    {
      FLD::Utils::write_doubles_to_file(time_, step_, flowrates, "flowrate");
    }
  }
  else
  {
    const std::map<int, double> flowrates =
        FLD::Utils::compute_flow_rates(*discret_, *velnp_, condstring, physicaltype_);

    // write to file
    if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
      FLD::Utils::write_doubles_to_file(time_, step_, flowrates, "flowrate");
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::FluidImplicitTimeInt::integrate_interface_shape(
    std::string condname)
{
  Teuchos::ParameterList eleparams;
  // set action for elements
  eleparams.set<FLD::BoundaryAction>("action", FLD::integrate_Shapefunction);

  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // create vector (+ initialization with zeros)
  std::shared_ptr<Core::LinAlg::Vector<double>> integratedshapefunc =
      std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // call loop over elements
  discret_->clear_state();
  if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);
  discret_->evaluate_condition(eleparams, integratedshapefunc, condname);
  discret_->clear_state();

  return integratedshapefunc;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::use_block_matrix(std::shared_ptr<std::set<int>> condelements,
    const Core::LinAlg::MultiMapExtractor& domainmaps,
    const Core::LinAlg::MultiMapExtractor& rangemaps, bool splitmatrix)
{
  use_block_matrix(
      condelements, domainmaps, rangemaps, condelements, domainmaps, rangemaps, splitmatrix);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::use_block_matrix(std::shared_ptr<std::set<int>> condelements,
    const Core::LinAlg::MultiMapExtractor& domainmaps,
    const Core::LinAlg::MultiMapExtractor& rangemaps,
    std::shared_ptr<std::set<int>> condelements_shape,
    const Core::LinAlg::MultiMapExtractor& domainmaps_shape,
    const Core::LinAlg::MultiMapExtractor& rangemaps_shape, bool splitmatrix)
{
  if (msht_ != Inpar::FLUID::no_meshtying)
  {
    meshtying_->is_multifield(condelements, domainmaps, rangemaps, condelements_shape,
        domainmaps_shape, rangemaps_shape, splitmatrix, true);
  }
  else
  {
    std::shared_ptr<Core::LinAlg::BlockSparseMatrix<FLD::Utils::InterfaceSplitStrategy>> mat;

    if (splitmatrix)
    {
      if (off_proc_assembly_)
      {
        FOUR_C_THROW(
            "Off proc assembly does not work with Block Matrices currently. Use structure split if "
            "you do an FSI.");
      }

      // (re)allocate system matrix
      mat = std::make_shared<Core::LinAlg::BlockSparseMatrix<FLD::Utils::InterfaceSplitStrategy>>(
          domainmaps, rangemaps, 108, false, true);
      mat->set_cond_elements(condelements);
      sysmat_ = mat;

      if (nonlinearbc_)
      {
        if (isimpedancebc_)
        {
          impedancebc_->use_block_matrix(condelements, domainmaps, rangemaps, splitmatrix);
        }
      }
    }

    // if we never build the matrix nothing will be done
    if (params_->get<bool>("shape derivatives"))
    {
      // allocate special mesh moving matrix
      mat = std::make_shared<Core::LinAlg::BlockSparseMatrix<FLD::Utils::InterfaceSplitStrategy>>(
          domainmaps_shape, rangemaps_shape, 108, false, true);
      mat->set_cond_elements(condelements_shape);
      shapederivatives_ = mat;
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::linear_relaxation_solve(
    std::shared_ptr<Core::LinAlg::Vector<double>> relax)
{
  // linear_relaxation_solve can't be used with locsys conditions cause it hasn't been implemented
  // yet
  if (locsysman_ != nullptr)
  {
    FOUR_C_THROW(
        "linear_relaxation_solve can't be used with locsys conditions cause it hasn't been "
        "implemented yet!");
  }

  TEUCHOS_FUNC_TIME_MONITOR("FluidImplicitTimeInt::linear_relaxation_solve");

  //
  // Special linear solve used for steepest descent relaxation as well as
  // Jacobian-free Newton-Krylov on the FSI interface equations. The later one
  // presents a special challenge, as we have to solve the same linear system
  // repeatedly for different rhs. That is why we need the inrelaxation_ flag.
  //
  // Additionally we might want to include the mesh derivatives to get optimal
  // convergence in the Newton loop.
  //
  // This adds even more state to the fluid algorithm class, which is a bad
  // thing. And the explicit storage of the Dirichlet lines is
  // required. However, we do not need any special element code to perform the
  // steepest descent calculation. This is quite a benefit as the special code
  // in the old discretization was a real nightmare.
  //

  if (not inrelaxation_)
  {
    // setup relaxation matrices just once
    //
    // We use these matrices for several solves in Jacobian-free Newton-Krylov
    // solves of the FSI interface equations.

    const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();
    std::shared_ptr<Core::LinAlg::Vector<double>> griddisp =
        std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, false);

    // set the grid displacement independent of the trial value at the
    // interface
    griddisp->update(1., *dispnp_, -1., *dispn_, 0.);

    // dbcmaps_ has already been set up

    // zero out the stiffness matrix
    sysmat_->zero();

    // zero out residual, no neumann bc
    residual_->put_scalar(0.0);

    // Get matrix for mesh derivatives. This is not meant to be efficient.
    if (params_->get<bool>("shape derivatives"))
    {
      if (meshmatrix_ == nullptr)
      {
        meshmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*system_matrix());
      }
      else
      {
        meshmatrix_->zero();
      }
    }

    // general fluid and time parameter are set in prepare_time_step()
    Teuchos::ParameterList eleparams;

    // parameters for stabilization
    eleparams.sublist("TURBULENCE MODEL") = params_->sublist("TURBULENCE MODEL");

    // set thermodynamic pressures
    set_custom_ele_params_linear_relaxation_solve(eleparams);

    if (xwall_ != nullptr) xwall_->set_x_wall_params(eleparams);

    // set general vector values needed by elements
    discret_->clear_state();
    discret_->set_state("hist", *hist_);
    discret_->set_state("veln", *veln_);
    discret_->set_state("accam", *accam_);
    discret_->set_state("scaaf", *scaaf_);
    discret_->set_state("scaam", *scaam_);
    discret_->set_state(ndsale_, "dispnp", *griddisp);
    discret_->set_state(ndsale_, "gridv", *zeros_);

    eleparams.set<FLD::Action>("action", FLD::calc_fluid_systemmat_and_residual);
    eleparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype_);
    // set scheme-specific element parameters and vector values
    set_state_tim_int();

    // call loop over elements
    discret_->evaluate(eleparams, sysmat_, meshmatrix_, residual_, nullptr, nullptr);
    discret_->clear_state();

    // finalize the system matrix
    sysmat_->complete();

    if (meshmatrix_ != nullptr)
    {
      meshmatrix_->complete();
    }

    //--------- Apply dirichlet boundary conditions to system of equations
    //          residual displacements are supposed to be zero at
    //          boundary conditions
    dirichletlines_ = nullptr;
    dirichletlines_ = system_matrix()->extract_dirichlet_rows(*(dbcmaps_->cond_map()));
    sysmat_->apply_dirichlet(*(dbcmaps_->cond_map()));
  }

  // No, we do not want to have any rhs. There cannot be any.
  residual_->put_scalar(0.0);

  if (meshmatrix_ != nullptr)
  {
    // Calculate rhs due to mesh movement induced by the interface
    // displacement.
    meshmatrix_->Apply(*gridv_, *residual_);
    residual_->scale(-dta_);
  }

  //--------- Apply dirichlet boundary conditions to system of equations
  //          residual displacements are supposed to be zero at
  //          boundary conditions
  incvel_->put_scalar(0.0);

  Core::LinAlg::apply_dirichlet_to_system(*incvel_, *residual_, *relax, *(dbcmaps_->cond_map()));

  custom_solve(relax);
  //-------solve for residual displacements to correct incremental displacements
  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = !inrelaxation_;
  solver_params.reset = !inrelaxation_;
  solver_->solve(sysmat_, incvel_, residual_, solver_params);

  // and now we need the reaction forces

  if (dirichletlines_->Apply(*incvel_, *trueresidual_) != 0)
    FOUR_C_THROW("dirichletlines_->Apply() failed");

  if (meshmatrix_ != nullptr)
  {
    // Calculate rhs due to mesh movement induced by the interface
    // displacement.
    meshmatrix_->Apply(*gridv_, *residual_);
    trueresidual_->update(dta_, *residual_, 1.0);
  }

  trueresidual_->scale(-residual_scaling());

  if (not inrelaxation_) inrelaxation_ = true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::add_dirich_cond(
    const std::shared_ptr<const Core::LinAlg::Map> maptoadd)
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> condmaps;
  condmaps.push_back(maptoadd);
  condmaps.push_back(dbcmaps_->cond_map());
  std::shared_ptr<Core::LinAlg::Map> condmerged =
      Core::LinAlg::MultiMapExtractor::merge_maps(condmaps);
  *dbcmaps_ = Core::LinAlg::MapExtractor(*(discret_->dof_row_map()), condmerged);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::remove_dirich_cond(
    const std::shared_ptr<const Core::LinAlg::Map> maptoremove)
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> othermaps;
  othermaps.push_back(maptoremove);
  othermaps.push_back(dbcmaps_->other_map());
  std::shared_ptr<Core::LinAlg::Map> othermerged =
      Core::LinAlg::MultiMapExtractor::merge_maps(othermaps);
  *dbcmaps_ = Core::LinAlg::MapExtractor(*(discret_->dof_row_map()), othermerged, false);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> FLD::FluidImplicitTimeInt::dirichlet()
{
  if (dbcmaps_ == nullptr) FOUR_C_THROW("Dirichlet map has not been allocated");
  std::shared_ptr<Core::LinAlg::Vector<double>> dirichones =
      std::make_shared<Core::LinAlg::Vector<double>>(*(dbcmaps_->cond_map()), false);
  dirichones->put_scalar(1.0);
  std::shared_ptr<Core::LinAlg::Vector<double>> dirichtoggle =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);
  dbcmaps_->insert_cond_vector(*dirichones, *dirichtoggle);
  return dirichtoggle;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> FLD::FluidImplicitTimeInt::inv_dirichlet()
{
  if (dbcmaps_ == nullptr) FOUR_C_THROW("Dirichlet map has not been allocated");
  std::shared_ptr<Core::LinAlg::Vector<double>> dirichzeros =
      std::make_shared<Core::LinAlg::Vector<double>>(*(dbcmaps_->cond_map()), true);
  std::shared_ptr<Core::LinAlg::Vector<double>> invtoggle =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), false);
  invtoggle->put_scalar(1.0);
  dbcmaps_->insert_cond_vector(*dirichzeros, *invtoggle);
  return invtoggle;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> FLD::FluidImplicitTimeInt::velocity_row_map()
{
  return velpressplitter_->other_map();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> FLD::FluidImplicitTimeInt::pressure_row_map()
{
  return velpressplitter_->cond_map();
}


// -------------------------------------------------------------------
// set general fluid parameter (AE 01/2011)
// -------------------------------------------------------------------
void FLD::FluidImplicitTimeInt::set_element_general_fluid_parameter()
{
  Teuchos::ParameterList eleparams;

  // set general element parameters
  eleparams.set("form of convective term", convform_);
  eleparams.set<Inpar::FLUID::LinearisationAction>("Linearisation", newton_);
  eleparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype_);

  // parameter for stabilization
  eleparams.sublist("RESIDUAL-BASED STABILIZATION") =
      params_->sublist("RESIDUAL-BASED STABILIZATION");
  eleparams.sublist("EDGE-BASED STABILIZATION") = params_->sublist("EDGE-BASED STABILIZATION");

  // get function number of given Oseen advective field if necessary
  if (physicaltype_ == Inpar::FLUID::oseen)
    eleparams.set<int>("OSEENFIELDFUNCNO", params_->get<int>("OSEENFIELDFUNCNO"));

  Discret::Elements::FluidEleParameterStd::instance()->set_element_general_fluid_parameter(
      eleparams, Core::Communication::my_mpi_rank(discret_->get_comm()));
}

// -------------------------------------------------------------------
// set turbulence parameters for element level       rasthofer 11/2011
// -------------------------------------------------------------------
void FLD::FluidImplicitTimeInt::set_element_turbulence_parameters()
{
  Teuchos::ParameterList eleparams;

  eleparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype_);

  // set general parameters for turbulent flow
  eleparams.sublist("TURBULENCE MODEL") = params_->sublist("TURBULENCE MODEL");

  // set model-dependent parameters
  eleparams.sublist("SUBGRID VISCOSITY") = params_->sublist("SUBGRID VISCOSITY");
  eleparams.sublist("MULTIFRACTAL SUBGRID SCALES") =
      params_->sublist("MULTIFRACTAL SUBGRID SCALES");

  Discret::Elements::FluidEleParameterStd::instance()->set_element_turbulence_parameters(eleparams);
}

// -------------------------------------------------------------------
// set general face fluid parameter for face/edge-oriented fluid stabilizations (BS 06/2014)
// -------------------------------------------------------------------
void FLD::FluidImplicitTimeInt::set_face_general_fluid_parameter()
{
  Teuchos::ParameterList faceparams;

  // set general fluid face parameters are contained in the following two sublists
  faceparams.sublist("EDGE-BASED STABILIZATION") = params_->sublist("EDGE-BASED STABILIZATION");

  faceparams.set<Inpar::FLUID::StabType>(
      "STABTYPE", Teuchos::getIntegralValue<Inpar::FLUID::StabType>(
                      params_->sublist("RESIDUAL-BASED STABILIZATION"), "STABTYPE"));

  faceparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype_);

  // get function number of given Oseen advective field if necessary
  if (physicaltype_ == Inpar::FLUID::oseen)
    faceparams.set<int>("OSEENFIELDFUNCNO", params_->get<int>("OSEENFIELDFUNCNO"));

  Discret::Elements::FluidEleParameterIntFace* fldintfacepara =
      Discret::Elements::FluidEleParameterIntFace::instance();
  fldintfacepara->set_face_general_fluid_parameter(
      faceparams, Core::Communication::my_mpi_rank(discret_->get_comm()));
}

// -------------------------------------------------------------------
// set turbulence parameters
// -------------------------------------------------------------------
void FLD::FluidImplicitTimeInt::set_general_turbulence_parameters()
{
  turbmodel_ = Inpar::FLUID::no_model;

  std::string physmodel =
      params_->sublist("TURBULENCE MODEL").get<std::string>("PHYSICAL_MODEL", "no_model");

  statistics_outfilename_ =
      params_->sublist("TURBULENCE MODEL").get<std::string>("statistics outfile");

  // flag for special flow
  special_flow_ = params_->sublist("TURBULENCE MODEL").get<std::string>("CANONICAL_FLOW", "no");

  // scale-separation
  scale_sep_ = FLUID::no_scale_sep;

  // fine-scale subgrid viscosity?
  fssgv_ = params_->sublist("TURBULENCE MODEL").get<Inpar::FLUID::FineSubgridVisc>("FSSUGRVISC");

  // warning if classical (all-scale) turbulence model and fine-scale
  // subgrid-viscosity approach are intended to be used simultaneously
  if (fssgv_ != Inpar::FLUID::no_fssgv and
      (physmodel == "Smagorinsky" or physmodel == "Dynamic_Smagorinsky" or
          physmodel == "Smagorinsky_with_van_Driest_damping"))
  {
    FOUR_C_THROW(
        "No combination of classical all-scale subgrid-viscosity turbulence model and fine-scale "
        "subgrid-viscosity approach currently possible!");
  }

  if (params_->sublist("TURBULENCE MODEL")
          .get<std::string>("TURBULENCE_APPROACH", "DNS_OR_RESVMM_LES") == "CLASSICAL_LES")
  {
    if (physmodel == "Dynamic_Smagorinsky")
    {
      turbmodel_ = Inpar::FLUID::dynamic_smagorinsky;

      // get one instance of the dynamic Smagorinsky class
      DynSmag_ = std::make_shared<FLD::DynSmagFilter>(discret_, *params_);
    }
    else if (physmodel == "Smagorinsky")
      turbmodel_ = Inpar::FLUID::smagorinsky;
    else if (physmodel == "Smagorinsky_with_van_Driest_damping")
      turbmodel_ = Inpar::FLUID::smagorinsky_with_van_Driest_damping;
    else if (physmodel == "Multifractal_Subgrid_Scales")
    {
      turbmodel_ = Inpar::FLUID::multifractal_subgrid_scales;

      fsvelaf_ = std::make_shared<Core::LinAlg::Vector<double>>(*discret_->dof_row_map(), true);

      Teuchos::ParameterList* modelparams = &(params_->sublist("MULTIFRACTAL SUBGRID SCALES"));

      const std::string scale_sep = modelparams->get<std::string>("SCALE_SEPARATION");
      if (scale_sep == "box_filter")
      {
        scale_sep_ = FLUID::box_filter;

        // get one instance of the Boxfilter class
        Boxf_ = std::make_shared<FLD::Boxfilter>(discret_, *params_);

        if (fssgv_ != Inpar::FLUID::no_fssgv)
          FOUR_C_THROW("No fine-scale subgrid viscosity for this scale separation operator!");
      }
      else if (scale_sep == "algebraic_multigrid_operator")
      {
        scale_sep_ = FLUID::algebraic_multigrid_operator;
      }
      else
      {
        FOUR_C_THROW("Unknown filter type!");
      }

      // fine-scale scalar at time n+alpha_F/n+1 and n+alpha_M/n
      // (only required for low-Mach-number case)
      fsscaaf_ = std::make_shared<Core::LinAlg::Vector<double>>(*discret_->dof_row_map(), true);
    }
    else if (physmodel == "Vreman")
    {
      turbmodel_ = Inpar::FLUID::vreman;
    }
    else if (physmodel == "Dynamic_Vreman")
    {
      turbmodel_ = Inpar::FLUID::dynamic_vreman;
      Vrem_ = std::make_shared<FLD::Vreman>(discret_, *params_);
    }
    else if (physmodel == "no_model")
      FOUR_C_THROW("Turbulence model for LES expected!");
    else
      FOUR_C_THROW("Undefined turbulence model!");

    print_turbulence_model();
  }
  else
  {
    if (turbmodel_ != Inpar::FLUID::no_model)
      FOUR_C_THROW("Set TURBULENCE APPROACH to CLASSICAL LES to activate turbulence model!");
  }

  // -------------------------------------------------------------------
  // necessary only for the AVM3 approach:
  // fine-scale solution vector + respective output
  // -------------------------------------------------------------------
  if (fssgv_ != Inpar::FLUID::no_fssgv)
  {
    fsvelaf_ = std::make_shared<Core::LinAlg::Vector<double>>(*discret_->dof_row_map(), true);

    if (myrank_ == 0)
    {
      // Output
      std::cout << "FLUID: Fine-scale subgrid-viscosity approach based on AVM3: ";
      std::cout << &std::endl << &std::endl;
      std::cout << fssgv_;
      std::cout << " with Smagorinsky constant Cs= ";
      std::cout << params_->sublist("SUBGRID VISCOSITY").get<double>("C_SMAGORINSKY");
      std::cout << &std::endl << &std::endl << &std::endl;
    }
  }

  // -------------------------------------------------------------------
  // check whether we have a coupling to a turbulent inflow generating
  // computation and initialize the transfer if necessary
  // -------------------------------------------------------------------
  if (xwall_ == nullptr)
  {
    turbulent_inflow_condition_ =
        std::make_shared<TransferTurbulentInflowCondition>(discret_, dbcmaps_);
  }
  else
    turbulent_inflow_condition_ =
        std::make_shared<TransferTurbulentInflowConditionXW>(discret_, dbcmaps_);
}

/*----------------------------------------------------------------------*
 | update Newton step                                                   |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::update_newton(
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel)
{
  update_iter_incrementally(vel);
}


// -------------------------------------------------------------------
// provide access to turbulence statistics manager (gjb 06/2011)
// -------------------------------------------------------------------
std::shared_ptr<FLD::TurbulenceStatisticManager>
FLD::FluidImplicitTimeInt::turbulence_statistic_manager()
{
  return statisticsmanager_;
}


// -------------------------------------------------------------------
// provide access to box filter for dynamic Smagorinsk model     rasthofer/krank
// -------------------------------------------------------------------
std::shared_ptr<FLD::DynSmagFilter> FLD::FluidImplicitTimeInt::dyn_smag_filter()
{
  return DynSmag_;
}

// -------------------------------------------------------------------
// provide access to box filter for dynamic Vreman model         rasthofer/krank
// -------------------------------------------------------------------
std::shared_ptr<FLD::Vreman> FLD::FluidImplicitTimeInt::vreman() { return Vrem_; }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// Overloaded in TimIntPoro and TimIntRedModels bk 12/13
void FLD::FluidImplicitTimeInt::update_iter_incrementally(
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel)
{
  // set the new solution we just got
  if (vel != nullptr)
  {
    // Take Dirichlet values from velnp and add vel to veln for non-Dirichlet
    // values.
    std::shared_ptr<Core::LinAlg::Vector<double>> aux =
        std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map(0)), true);
    aux->update(1.0, *velnp_, 1.0, *vel, 0.0);
    //    dbcmaps_->insert_other_vector(dbcmaps_->extract_other_vector(*aux), velnp_);
    dbcmaps_->insert_cond_vector(*dbcmaps_->extract_cond_vector(*velnp_), *aux);

    *velnp_ = *aux;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::print_stabilization_details() const
{
  // output of stabilization details
  Teuchos::ParameterList* stabparams = &(params_->sublist("RESIDUAL-BASED STABILIZATION"));
  if (myrank_ == 0)
  {
    std::cout << "Stabilization type         : "
              << to_string(Teuchos::get<Inpar::FLUID::StabType>(*stabparams, "STABTYPE")) << "\n";
    std::cout << "                             "
              << "Evaluation Tau  = " << stabparams->get<std::string>("EVALUATION_TAU") << "\n";
    std::cout << "                             "
              << "Evaluation Mat  = " << stabparams->get<std::string>("EVALUATION_MAT") << "\n";
    std::cout << "\n";

    if (Teuchos::getIntegralValue<Inpar::FLUID::StabType>(*stabparams, "STABTYPE") ==
        Inpar::FLUID::stabtype_residualbased)
    {
      std::cout << "                             "
                << Teuchos::getIntegralValue<Inpar::FLUID::SubscalesTD>(*stabparams, "TDS") << "\n";
      std::cout << "\n";
      std::cout << "                             "
                << "Tau Type        = "
                << to_string(Teuchos::get<Inpar::FLUID::TauType>(*stabparams, "DEFINITION_TAU"))
                << "\n";

      if (Teuchos::getIntegralValue<Inpar::FLUID::SubscalesTD>(*stabparams, "TDS") ==
          Inpar::FLUID::SubscalesTD::subscales_quasistatic)
      {
        if (Teuchos::getIntegralValue<Inpar::FLUID::Transient>(*stabparams, "TRANSIENT") ==
            Inpar::FLUID::Transient::inertia_stab_keep)
        {
          FOUR_C_THROW(
              "The quasistatic version of the residual-based stabilization currently does not "
              "support the incorporation of the transient term.");
        }
      }

      std::cout << "                             "
                << "SUPG            = " << std::boolalpha << Teuchos::get<bool>(*stabparams, "SUPG")
                << "\n";
      std::cout << "                             "
                << "PSPG            = " << std::boolalpha << Teuchos::get<bool>(*stabparams, "PSPG")
                << "\n";
      std::cout << "                             "
                << "GRAD_DIV        = " << std::boolalpha
                << Teuchos::get<bool>(*stabparams, "GRAD_DIV") << "\n";
      std::cout << "                             "
                << "CROSS-STRESS    = "
                << to_string(Teuchos::get<Inpar::FLUID::CrossStress>(*stabparams, "CROSS-STRESS"))
                << "\n";
      std::cout << "                             "
                << "REYNOLDS-STRESS = "
                << to_string(
                       Teuchos::get<Inpar::FLUID::ReynoldsStress>(*stabparams, "REYNOLDS-STRESS"))
                << "\n";
      std::cout << "                             "
                << "VSTAB           = "
                << to_string(Teuchos::get<Inpar::FLUID::VStab>(*stabparams, "VSTAB")) << "\n";
      std::cout << "                             "
                << "RSTAB           = "
                << to_string(Teuchos::get<Inpar::FLUID::RStab>(*stabparams, "RSTAB")) << "\n";
      std::cout << "                             "
                << "TRANSIENT       = "
                << to_string(Teuchos::get<Inpar::FLUID::Transient>(*stabparams, "TRANSIENT"))
                << "\n";
      std::cout << "\n";
      std::cout << std::endl;
    }
    else if (Teuchos::getIntegralValue<Inpar::FLUID::StabType>(*stabparams, "STABTYPE") ==
             Inpar::FLUID::stabtype_edgebased)
    {
      Teuchos::ParameterList* stabparams_edgebased =
          &(params_->sublist("EDGE-BASED STABILIZATION"));

      std::cout << "\n\nEDGE-BASED (EOS) fluid stabilizations "
                << "\n";

      std::cout << "                    "
                << "EOS_PRES             = "
                << to_string(
                       Teuchos::get<Inpar::FLUID::EosPres>(*stabparams_edgebased, ("EOS_PRES")))
                << "\n";
      std::cout << "                    "
                << "EOS_CONV_STREAM      = "
                << to_string(Teuchos::get<Inpar::FLUID::EosConvStream>(
                       *stabparams_edgebased, "EOS_CONV_STREAM"))
                << "\n";
      std::cout << "                    "
                << "EOS_CONV_CROSS       = "
                << to_string(Teuchos::get<Inpar::FLUID::EosConvCross>(
                       *stabparams_edgebased, "EOS_CONV_CROSS"))
                << "\n";
      std::cout << "                    "
                << "EOS_DIV              = "
                << to_string(Teuchos::get<Inpar::FLUID::EosDiv>(*stabparams_edgebased, "EOS_DIV"))
                << "\n";
      std::cout << "                    "
                << "EOS_DEFINITION_TAU   = "
                << to_string(Teuchos::get<Inpar::FLUID::EosTauType>(
                       *stabparams_edgebased, "EOS_DEFINITION_TAU"))
                << "\n";
      std::cout << "                    "
                << "EOS_H_DEFINITION     = "
                << to_string(Teuchos::get<Inpar::FLUID::EosElementLength>(
                       *stabparams_edgebased, "EOS_H_DEFINITION"))
                << "\n";
      std::cout
          << "+---------------------------------------------------------------------------------+\n"
          << std::endl;
    }
  }
}

// -------------------------------------------------------------------
// print information about turbulence model         rasthofer 04/2011
// -------------------------------------------------------------------
void FLD::FluidImplicitTimeInt::print_turbulence_model()
{
  // a canonical flow with homogeneous directions would allow a
  // spatial averaging of data
  std::string homdir =
      params_->sublist("TURBULENCE MODEL").get<std::string>("HOMDIR", "not_specified");

  if (myrank_ == 0 and turbmodel_ != Inpar::FLUID::no_model)
  {
    std::cout << "Turbulence model        : ";
    std::cout
        << params_->sublist("TURBULENCE MODEL").get<std::string>("PHYSICAL_MODEL", "no_model");
    std::cout << &std::endl;

    if (turbmodel_ == Inpar::FLUID::smagorinsky)
    {
      std::cout << "                             ";
      std::cout << "with Smagorinsky constant Cs= ";
      std::cout << params_->sublist("SUBGRID VISCOSITY").get<double>("C_SMAGORINSKY") << "\n";
      std::cout << &std::endl;
    }
    else if (turbmodel_ == Inpar::FLUID::smagorinsky_with_van_Driest_damping)
    {
      if (special_flow_ != "channel_flow_of_height_2" || homdir != "xz")
      {
        FOUR_C_THROW(
            "The van Driest damping is only implemented for a channel flow with wall \nnormal "
            "direction y");
      }

      std::cout << "                             ";
      std::cout << "\n";
      std::cout << "- Smagorinsky constant:   Cs   = ";
      std::cout << params_->sublist("SUBGRID VISCOSITY").get<double>("C_SMAGORINSKY");
      std::cout << &std::endl;
      std::cout << "- viscous length      :   l_tau= ";
      std::cout << params_->sublist("SUBGRID VISCOSITY").get<double>("CHANNEL_L_TAU") << "\n";
      std::cout << &std::endl;
    }
    else if (turbmodel_ == Inpar::FLUID::dynamic_smagorinsky)
    {
      if (homdir == "not_specified")
      {
        std::cout << "      no homogeneous directions specified --- so we just use pointwise "
                     "clipping for Cs\n";
        std::cout << &std::endl;
      }
    }
    else if (turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales)
    {
      Teuchos::ParameterList* modelparams = &(params_->sublist("MULTIFRACTAL SUBGRID SCALES"));
      std::cout << "                             ";
      std::cout << "\n";
      std::cout << "- Csgs:              " << modelparams->get<double>("CSGS") << "\n";
      std::cout << "- Scale separation:  " << modelparams->get<std::string>("SCALE_SEPARATION")
                << "\n";
      if (modelparams->get<bool>("CALC_N"))
      {
        std::cout << "- Re_length:         " << modelparams->get<std::string>("REF_LENGTH") << "\n";
        std::cout << "- Re_vel:            " << modelparams->get<std::string>("REF_VELOCITY")
                  << "\n";
        std::cout << "- c_nu:              " << modelparams->get<double>("C_NU") << "\n";
      }
      else
        std::cout << "- N:                 " << modelparams->get<double>("N") << "\n";
      std::cout << "- near-wall limit:   " << modelparams->get<bool>("NEAR_WALL_LIMIT") << "\n";
      std::cout << "- beta:              " << modelparams->get<double>("BETA") << "\n";
      std::cout << "- evaluation B:      " << modelparams->get<std::string>("EVALUATION_B") << "\n";
      std::cout << "- conservative:      " << modelparams->get<std::string>("CONVFORM") << "\n";
      if (modelparams->get<bool>("SET_FINE_SCALE_VEL"))
        std::cout << "WARNING: fine-scale velocity is set for nightly tests!\n";
      std::cout << &std::endl;
    }
    else if (turbmodel_ == Inpar::FLUID::vreman)
    {
      std::cout << "                             ";
      std::cout << "\n";
      std::cout << "- Vreman model with constant coefficient\n";
      std::cout << "- Use filter width method:  "
                << to_string(Teuchos::get<Inpar::FLUID::VremanFiMethod>(
                       params_->sublist("SUBGRID VISCOSITY"), "FILTER_WIDTH"))
                << "\n";
      std::cout << &std::endl;
    }
    else if (turbmodel_ == Inpar::FLUID::dynamic_vreman)
    {
      std::cout << "                             ";
      std::cout << "\n";
      std::cout << "- Vreman model with dynamic calculation of coefficient\n";
      std::cout
          << "- Use filter width method:  Only cube root volume implemented for dynamic coefficient"
          << "\n";
      std::cout << &std::endl;
    }
  }
}


/*----------------------------------------------------------------------*
 | filtered quantities for classical LES models          rasthofer 02/11|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::apply_scale_separation_for_les()
{
  if (turbmodel_ == Inpar::FLUID::dynamic_smagorinsky)
  {
    // perform filtering and computation of Cs
    // compute averaged values for LijMij and MijMij
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dirichtoggle = dirichlet();
    DynSmag_->apply_filter_for_dynamic_computation_of_cs(
        evaluation_vel(), scaaf_, return_thermpressaf(), dirichtoggle);
  }
  else if (turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales)
  {
    switch (scale_sep_)
    {
      case FLUID::box_filter:
      {
        // perform filtering
        const std::shared_ptr<const Core::LinAlg::Vector<double>> dirichtoggle = dirichlet();
        // call only filtering
        Boxf_->apply_filter(evaluation_vel(), scaaf_, return_thermpressaf(), dirichtoggle);

        // get fine-scale velocity
        Boxf_->outputof_fine_scale_vel(*fsvelaf_);

        break;
      }
      case FLUID::algebraic_multigrid_operator:
      {
        // get fine-scale part of velocity at time n+alpha_F or n+1
        Sep_->multiply(false, *evaluation_vel(), *fsvelaf_);

        // set fine-scale velocity for parallel nightly tests
        // separation matrix depends on the number of proc here
        if (turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales and
            (params_->sublist("MULTIFRACTAL SUBGRID SCALES").get<bool>("SET_FINE_SCALE_VEL")))
          fsvelaf_->put_scalar(0.01);

        break;
      }
      default:
      {
        FOUR_C_THROW("Unknown filter type!");
        break;
      }
    }

    // set fine-scale vector
    discret_->set_state("fsvelaf", *fsvelaf_);
  }
  else if (turbmodel_ == Inpar::FLUID::dynamic_vreman)
  {
    // perform filtering
    const std::shared_ptr<const Core::LinAlg::Vector<double>> dirichtoggle = dirichlet();

    Vrem_->apply_filter_for_dynamic_computation_of_cv(
        evaluation_vel(), scaaf_, return_thermpressaf(), dirichtoggle);
  }
  else
    FOUR_C_THROW("Unknown turbulence model!");
}


//-------------------------------------------------------------------------
// calculate mean CsgsB to estimate CsgsD
// for multifractal subgrid-scale model                    rasthofer 08/12
//-------------------------------------------------------------------------
void FLD::FluidImplicitTimeInt::recompute_mean_csgs_b()
{
  // For loma, this function is required at the respective position to set up CsgsD of the scalar
  // field for including the subgrid-scale temperature in the physical properties and the
  // subgrid-scale terms arising in the continuity equation. This recomputation avoids transferring
  // the respective value from the scalar to the fluid field. The so computed value is also used for
  // calculating statistical data for MFS, although this is not the final value for gen-alpha that
  // is seen by the scalar field. However, note that vel_n ~ vel_np ~ vel_af for statistically
  // stationary flow. Hence, the expected error is marginal, but another computation is avoided.

  if (params_->sublist("MULTIFRACTAL SUBGRID SCALES").get<bool>("ADAPT_CSGS_PHI"))
  {
    // mean Cai
    double meanCai = 0.0;

    // variables required for calculation
    // local sums
    double local_sumCai = 0.0;
    double local_sumVol = 0.0;
    // global sums
    double global_sumCai = 0.0;
    double global_sumVol = 0.0;

    // define element matrices and vectors --- dummies
    Core::LinAlg::SerialDenseMatrix emat1;
    Core::LinAlg::SerialDenseMatrix emat2;
    Core::LinAlg::SerialDenseVector evec1;
    Core::LinAlg::SerialDenseVector evec2;
    Core::LinAlg::SerialDenseVector evec3;

    // generate a parameterlist for communication and control
    Teuchos::ParameterList myparams;
    // action for elements
    myparams.set<FLD::Action>("action", FLD::calc_mean_Cai);
    myparams.set<Inpar::FLUID::PhysicalType>("Physical Type", physicaltype_);

    // set state vector to pass distributed vector to the element
    // set velocity
    discret_->clear_state();
    set_state_tim_int();
    // set temperature
    discret_->set_state("scalar", *scaaf_);
    // set thermodynamic pressures
    set_custom_ele_params_apply_nonlinear_boundary_conditions(myparams);

    // loop all elements on this proc (excluding ghosted ones)
    for (int nele = 0; nele < discret_->num_my_row_elements(); ++nele)
    {
      // get the element
      Core::Elements::Element* ele = discret_->l_row_element(nele);

      // get element location vector, dirichlet flags and ownerships
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      ele->location_vector(*discret_, lm, lmowner, lmstride);

      // call the element evaluate method to integrate functions
      int err = ele->evaluate(myparams, *discret_, lm, emat1, emat2, evec1, evec2, evec2);
      if (err) FOUR_C_THROW("Proc {}: Element {} returned err={}", myrank_, ele->id(), err);

      // get contributions of this element and add it up
      local_sumCai += myparams.get<double>("Cai_int");
      local_sumVol += myparams.get<double>("ele_vol");
    }
    discret_->clear_state();

    // gather contributions of all procs
    global_sumCai = Core::Communication::sum_all(local_sumCai, discret_->get_comm());
    global_sumVol = Core::Communication::sum_all(local_sumVol, discret_->get_comm());

    // calculate mean Cai
    meanCai = global_sumCai / global_sumVol;

    // std::cout << "Proc:  " << myrank_ << "  local vol and Cai   "
    //<< local_sumVol << "   " << local_sumCai << "  global vol and Cai   "
    //<< global_sumVol << "   " << global_sumCai << "  mean   " << meanCai << std::endl;

    if (myrank_ == 0)
    {
      std::cout << "\n+----------------------------------------------------------------------------"
                   "----------------+"
                << std::endl;
      std::cout << "Multifractal subgrid scales: adaption of CsgsD from near-wall limit of CsgsB:  "
                << std::setprecision(8) << meanCai << std::endl;
      std::cout << "+------------------------------------------------------------------------------"
                   "--------------+\n"
                << std::endl;
    }

    // store value in element parameter list
    Discret::Elements::FluidEleParameterStd::instance()->set_csgs_phi(meanCai);
  }
}

// -------------------------------------------------------------------
// extrapolate from time mid-point to end-point         (mayr 12/2011)
// overloaded in TimIntGenAlpha                            bk 12/13
// -------------------------------------------------------------------
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::FluidImplicitTimeInt::extrapolate_end_point(
    std::shared_ptr<Core::LinAlg::Vector<double>> vecn,
    std::shared_ptr<Core::LinAlg::Vector<double>> vecm)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> vecnp =
      std::make_shared<Core::LinAlg::Vector<double>>(*vecm);

  return vecnp;
}


// -------------------------------------------------------------------
// apply external forces to the fluid                    ghamm 03/2013
// -------------------------------------------------------------------
void FLD::FluidImplicitTimeInt::apply_external_forces(
    std::shared_ptr<Core::LinAlg::MultiVector<double>> fext)
{
  if (external_loads_ == nullptr)
    external_loads_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*discret_->dof_row_map(), true);

  external_loads_->update(1.0, *fext, 0.0);
}


/*------------------------------------------------------------------------------------------------*
 | create field test
 *------------------------------------------------------------------------------------------------*/
std::shared_ptr<Core::Utils::ResultTest> FLD::FluidImplicitTimeInt::create_field_test()
{
  return std::make_shared<FLD::FluidResultTest>(*this);
}


/*------------------------------------------------------------------------------------------------*
 |
 *------------------------------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> FLD::FluidImplicitTimeInt::convective_vel()
{
  if (grid_vel() == nullptr)
    return velnp();  // no moving mesh present
  else
  {
    // make an intermediate copy of velnp
    std::shared_ptr<Core::LinAlg::Vector<double>> convel =
        std::make_shared<Core::LinAlg::Vector<double>>(*(velnp()));
    // now subtract the grid velocity
    convel->update(-1.0, *(grid_vel()), 1.0);

    return convel;
  }
}


/*------------------------------------------------------------------------------------------------*
 | Calculate an integrated divergence operator                                    (mayr.mt 04/12) |
 *------------------------------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FLD::FluidImplicitTimeInt::calc_div_op()
{
  // set action in order to calculate the integrated divergence operator
  Teuchos::ParameterList params;
  params.set<FLD::Action>("action", FLD::calc_divop);

  // integrated divergence operator B in vector form
  std::shared_ptr<Core::LinAlg::Vector<double>> divop =
      std::make_shared<Core::LinAlg::Vector<double>>(velnp_->get_map(), true);

  // copy row map of mesh displacement to column map (only if ALE is used)
  discret_->clear_state();
  if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);

  // construct the operator on element level as a column vector
  discret_->evaluate(params, nullptr, nullptr, divop, nullptr, nullptr);

  // clear column maps after the evaluate call
  discret_->clear_state();

  //  // blank DOFs which are on Dirichlet BC, since they may not be modified
  //  dbcmaps_->insert_cond_vector(dbcmaps_->extract_cond_vector(*zeros_), divop);

  return divop;
}


/*------------------------------------------------------------------------------------------------*
 |
 *------------------------------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::reset(int numsteps, int iter)
{
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // Vectors passed to the element
  // -----------------------------
  // velocity/pressure at time n+1, n and n-1
  velnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  veln_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  velnm_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // acceleration/(scalar time derivative) at time n+1 and n
  accnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  accn_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  accnm_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // velocity/pressure at time n+alpha_F
  velaf_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // velocity/pressure at time n+alpha_M
  velam_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // acceleration/(scalar time derivative) at time n+alpha_M/(n+alpha_M/n)
  accam_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // scalar at time n+alpha_F/n+1 and n+alpha_M/n
  // (only required for low-Mach-number case)
  scaaf_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);
  scaam_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  // history vector
  hist_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap, true);

  if (alefluid_)
  {
    const Core::LinAlg::Map* aledofrowmap = discret_->dof_row_map(ndsale_);

    if (!dispnp_) dispnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*aledofrowmap, true);
    if (!dispn_) dispn_ = std::make_shared<Core::LinAlg::Vector<double>>(*aledofrowmap, true);
    dispnm_ = std::make_shared<Core::LinAlg::Vector<double>>(*aledofrowmap, true);
    gridv_ = std::make_shared<Core::LinAlg::Vector<double>>(*aledofrowmap, true);
    gridvn_ = std::make_shared<Core::LinAlg::Vector<double>>(*aledofrowmap, true);
  }
}


/*------------------------------------------------------------------------------------------------*
 |
 *------------------------------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::predict_tang_vel_consist_acc()
{
  // message to screen
  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
  {
    std::cout << "fluid: doing TangVel predictor" << std::endl;
  }

  // total time required for evaluation of Dirichlet conditions
  Teuchos::ParameterList eleparams;
  eleparams.set("total time", time_);

  // initialize
  velnp_->update(1.0, *veln_, 0.0);
  accnp_->update(1.0, *accn_, 0.0);
  incvel_->put_scalar(0.0);

  // for solution increments on Dirichlet boundary
  std::shared_ptr<Core::LinAlg::Vector<double>> dbcinc =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);

  // copy last converged solution
  dbcinc->update(1.0, *veln_, 0.0);

  // get Dirichlet values at t_{n+1}
  // set vector values needed by elements
  discret_->clear_state();
  discret_->set_state("velnp", *velnp_);

  // predicted Dirichlet values
  // velnp_ then also holds prescribed new dirichlet values
  discret_->evaluate_dirichlet(eleparams, velnp_, nullptr, nullptr, nullptr);

  // subtract the displacements of the last converged step
  // DBC-DOFs hold increments of current step
  // free-DOFs hold zeros
  dbcinc->update(-1.0, *veln_, 1.0);

  // compute residual forces residual_ and stiffness sysmat_
  // at velnp_, etc which are unchanged
  evaluate(nullptr);

  // add linear reaction forces to residual
  // linear reactions
  std::shared_ptr<Core::LinAlg::Vector<double>> freact =
      std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);
  sysmat_->multiply(false, *dbcinc, *freact);

  // add linear reaction forces due to prescribed Dirichlet BCs
  residual_->update(1.0, *freact, 1.0);

  // extract reaction forces
  freact->update(1.0, *residual_, 0.0);
  dbcmaps_->insert_other_vector(*dbcmaps_->extract_other_vector(*zeros_), *freact);

  // blank residual at DOFs on Dirichlet BC
  dbcmaps_->insert_cond_vector(*dbcmaps_->extract_cond_vector(*zeros_), *residual_);

  // apply Dirichlet BCs to system of equations
  incvel_->put_scalar(0.0);
  sysmat_->complete();
  Core::LinAlg::apply_dirichlet_to_system(
      *sysmat_, *incvel_, *residual_, *zeros_, *(dbcmaps_->cond_map()));

  // solve for incvel_
  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  solver_->solve(sysmat_, incvel_, residual_, solver_params);

  // set Dirichlet increments in solution increments
  incvel_->update(1.0, *dbcinc, 1.0);

  // update end-point velocities and pressure
  update_iter_incrementally(incvel_);

  // keep pressure values from previous time step
  velpressplitter_->insert_cond_vector(*velpressplitter_->extract_cond_vector(*veln_), *velnp_);

  // Note: accelerations on Dirichlet DOFs are not set.

  // reset to zero
  incvel_->put_scalar(0.0);
}

/*----------------------------------------------------------------------*/
/* set fluid displacement vector due to biofilm growth          */
void FLD::FluidImplicitTimeInt::set_fld_gr_disp(
    std::shared_ptr<Core::LinAlg::Vector<double>> fluid_growth_disp)
{
  fldgrdisp_ = fluid_growth_disp;
}

// overloaded in TimIntRedModels bk 12/13
void FLD::FluidImplicitTimeInt::setup_meshtying()
{
  msht_ = Teuchos::getIntegralValue<Inpar::FLUID::MeshTying>(*params_, "MESHTYING");
  bool alldofcoupled = params_->get<bool>("ALLDOFCOUPLED");

  // meshtying: all dofs (velocity + pressure) are coupled
  //            -> vector of length velocity dofs (numdim_) + pressure dof initialized with ones
  // coupleddof [1, 1, 1, 1]
  std::vector<int> coupleddof(numdim_ + 1, 1);

  if (!alldofcoupled)
  {
    // meshtying: only velocity dofs are coupled
    // meshtying: all dofs (velocity + pressure) are coupled
    //            -> vector of length velocity dofs (numdim_) + pressure dof initialized with ones
    //            -> last entry (pressure) is set to zero -> pressure is not included into the
    //            coupling algorithm
    // coupleddof [1, 1, 1, 0]
    coupleddof[numdim_] = 0;
  }

  if (xwall_ != nullptr)
    for (int xdof = 0; xdof < 4; xdof++) coupleddof.push_back(0);

  meshtying_ = std::make_shared<Meshtying>(discret_, *solver_, msht_, numdim_, surfacesplitter_);
  meshtying_->setup_meshtying(coupleddof, alldofcoupled);
  sysmat_ = meshtying_->init_system_matrix();

  // Check if there are DC defined on the master side of the internal interface
  meshtying_->dirichlet_on_master(dbcmaps_->cond_map());

  if (predictor_ != "steady_state")
  {
    if (myrank_ == 0)
      FOUR_C_THROW("The meshtying framework does only support a steady-state predictor");
  }

  // meshtying_->OutputSetUp();
}

/*----------------------------------------------------------------------------*
 | Compute kinetic energy and write it to file                mayr.mt 05/2014 |
 *----------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::write_output_kinetic_energy()
{
  // take care of possibly changed element geometry
  if (alefluid_) evaluate_mass_matrix();

  // compute kinetic energy
  double energy = 0.0;
  Core::LinAlg::Vector<double> mtimesu(massmat_->OperatorRangeMap(), true);
  massmat_->Apply(*velnp_, mtimesu);
  velnp_->dot(mtimesu, &energy);
  energy *= 0.5;

  // write to file
  if (myrank_ == 0 and (logenergy_))
  {
    (*logenergy_) << std::right << std::setw(9) << step_ << std::right << std::setw(16) << time_
                  << std::right << std::setw(16) << energy << std::endl;
  }
}

/*----------------------------------------------------------------------------*
 | Set time step size                                         mayr.mt 09/2013 |
 *----------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::set_dt(const double dtnew) { dta_ = dtnew; }

/*----------------------------------------------------------------------------*
 | Set time and step                                          mayr.mt 09/2013 |
 *----------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::set_time_step(const double time, const int step)
{
  step_ = step;
  time_ = time;
}

/*---------------------------------------------------------------*/
/*---------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::set_dirichlet_neumann_bc()
{
  Teuchos::ParameterList eleparams;

  // total time required for Dirichlet conditions
  eleparams.set("total time", time_);
  eleparams.set<const Core::Utils::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());

  // predicted dirichlet values
  // velnp then also holds prescribed new dirichlet values
  apply_dirichlet_bc(eleparams, velnp_, nullptr, nullptr, false);

  // additionally evaluate problem-specific boundary conditions
  do_problem_specific_boundary_conditions();

  // By definition: Applying DC on the slave side of an internal interface is not allowed
  //                since it leads to an over-constraint system
  // Therefore, nodes belonging to the slave side of an internal interface have to be excluded from
  // the DC. However, a velocity value (projected from the Dirichlet condition on the master side)
  // has to be assigned to the DOF's on the slave side in order to evaluate the system matrix
  // completely

  // Preparation for including DC on the master side in the condensation process
  if (msht_ != Inpar::FLUID::no_meshtying)
    meshtying_->include_dirichlet_in_condensation(*velnp_, *veln_);

  discret_->clear_state();

  // Transfer of boundary data if necessary
  turbulent_inflow_condition_->transfer(veln_, velnp_, time_);

  // add problem-dependent parameters, e.g., thermodynamic pressure in case of loma
  set_custom_ele_params_apply_nonlinear_boundary_conditions(eleparams);

  if (alefluid_) discret_->set_state(ndsale_, "dispnp", *dispnp_);

  // evaluate Neumann conditions
  neumann_loads_->put_scalar(0.0);
  discret_->set_state("scaaf", *scaaf_);
  discret_->set_state("velaf", *velaf_);
  discret_->evaluate_neumann(eleparams, *neumann_loads_);
  discret_->clear_state();
}

/*---------------------------------------------------------------*/
/* Apply Dirichlet boundary conditions on provided state vectors */
void FLD::FluidImplicitTimeInt::apply_dirichlet_bc(Teuchos::ParameterList& params,
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvector,    //!< (may be nullptr)
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvectord,   //!< (may be nullptr)
    std::shared_ptr<Core::LinAlg::Vector<double>> systemvectordd,  //!< (may be nullptr)
    bool recreatemap  //!< recreate mapextractor/toggle-vector
)
{
  // In the case of local coordinate systems, we have to rotate forward ...
  // --------------------------------------------------------------------------------
  if (locsysman_ != nullptr)
  {
    if (systemvector != nullptr) locsysman_->rotate_global_to_local(*systemvector);
    if (systemvectord != nullptr) locsysman_->rotate_global_to_local(*systemvectord);
    if (systemvectordd != nullptr) locsysman_->rotate_global_to_local(*systemvectordd);
  }

  // Apply DBCs
  // --------------------------------------------------------------------------------
  discret_->clear_state();
  // If we have HDG discret
  if (dynamic_cast<const Core::FE::DiscretizationHDG*>(&(*discret_)) != nullptr)
  {
    auto dbc = std::shared_ptr<const Core::FE::Dbc>(new const FLD::Utils::DbcHdgFluid());
    (*dbc)(*discret_, params, systemvector, systemvectord, systemvectordd, nullptr,
        recreatemap ? dbcmaps_ : nullptr);
  }
  else if (recreatemap)
  {
    discret_->evaluate_dirichlet(
        params, systemvector, systemvectord, systemvectordd, nullptr, dbcmaps_);
  }
  else
  {
    discret_->evaluate_dirichlet(
        params, systemvector, systemvectord, systemvectordd, nullptr, nullptr);
  }
  discret_->clear_state();

  // In the case of local coordinate systems, we have to rotate back into global Cartesian frame
  // --------------------------------------------------------------------------------
  if (locsysman_ != nullptr)
  {
    if (systemvector != nullptr) locsysman_->rotate_local_to_global(*systemvector);
    if (systemvectord != nullptr) locsysman_->rotate_local_to_global(*systemvectord);
    if (systemvectordd != nullptr) locsysman_->rotate_local_to_global(*systemvectordd);
  }
}

/*----------------------------------------------------------------------*
 * Explicit predictor                                   rasthofer 12/13 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::explicit_predictor()
{
  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
  {
    printf("fluid: using explicit predictor %s", predictor_.c_str());
  }

  if (predictor_ == "steady_state")
  {
    // steady state predictor
    //
    //       n+1    n
    //      u    = u
    //       (0)
    //
    //  and
    //
    //       n+1    n
    //      p    = p
    //       (0)

    // this has already been done in TimeUpdate()
  }
  else if (predictor_ == "zero_acceleration")
  {
    // zero acceleration predictor
    //
    //       n+1    n                   n
    //      u    = u  + (1-gamma)*dt*acc
    //       (0)
    //
    //  and
    //
    //       n+1    n
    //      p    = p
    //       (0)
    //
    velnp_->update(1.0, *veln_, 0.0);

    // split between acceleration and pressure
    std::shared_ptr<Core::LinAlg::Vector<double>> inc =
        velpressplitter_->extract_other_vector(*accn_);
    inc->scale((1.0 - theta_) * dta_);

    velpressplitter_->add_other_vector(*inc, *velnp_);
  }
  else if (predictor_ == "constant_acceleration")
  {
    // constant acceleration predictor
    //
    //       n+1    n         n
    //      u    = u  + dt*acc
    //       (0)
    //
    //  and
    //
    //       n+1    n
    //      p    = p
    //       (0)
    //
    velnp_->update(1.0, *veln_, 0.0);

    std::shared_ptr<Core::LinAlg::Vector<double>> inc =
        velpressplitter_->extract_other_vector(*accn_);
    inc->scale(dta_);

    velpressplitter_->add_other_vector(*inc, *velnp_);
  }
  else if (predictor_ == "constant_increment")
  {
    // constant increment predictor
    //
    //       n+1      n    n-1
    //      u    = 2*u  - u
    //       (0)
    //
    //  and
    //
    //       n+1    n
    //      p    = p
    //       (0)
    //
    velnp_->update(1.0, *veln_, 0.0);

    std::shared_ptr<Core::LinAlg::Vector<double>> un =
        velpressplitter_->extract_other_vector(*veln_);
    std::shared_ptr<Core::LinAlg::Vector<double>> unm =
        velpressplitter_->extract_other_vector(*velnm_);
    unm->scale(-1.0);

    velpressplitter_->add_other_vector(*un, *velnp_);
    velpressplitter_->add_other_vector(*unm, *velnp_);
  }
  else if (predictor_ == "explicit_second_order_midpoint")
  {
    // the conventional explicit second order predictor (assuming constant dt)
    // also known as leapfrog integration
    /*
    //                        /          n    n-1 \
    //       n+1    n        |      n   u  - u     |
    //      u    = u  + dt * | 2*acc  - ---------  |
    //       (0)             |             dt      |
    //                        \                   /
    // respectively
    //
    //       n+1    n-1               n
    //      u    = u    + 2 * dt * acc
    //       (0)
    //
    //  and
    //
    //       n+1    n
    //      p    = p
    //       (0)
    */
    velnp_->update(1.0, *veln_, 0.0);

    // split between acceleration and pressure
    std::shared_ptr<Core::LinAlg::Vector<double>> unm =
        velpressplitter_->extract_other_vector(*velnm_);
    std::shared_ptr<Core::LinAlg::Vector<double>> an =
        velpressplitter_->extract_other_vector(*accn_);

    unm->update(2.0 * dta_, *an, 1.0);

    velpressplitter_->insert_other_vector(*unm, *velnp_);
  }
  else
    FOUR_C_THROW("Unknown fluid predictor {}", predictor_);

  if (Core::Communication::my_mpi_rank(discret_->get_comm()) == 0)
  {
    printf("\n");
  }
}

/*----------------------------------------------------------------------------*
 * Add vector to external loads being applied to rhs before solve  rauch 12/14 |
 *                                                                             |
 * external_loads_ may have been built before by method ApplyExternalForces()  |
 * Be careful here, because the external loads are not reset after a timestep!|
 *----------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::add_contribution_to_external_loads(
    const std::shared_ptr<const Core::LinAlg::Vector<double>> contributing_vector)
{
  /// important note:
  /// will be scaled with 1.0/residual_scaling() when applied in
  /// void FLD::FluidImplicitTimeInt::assemble_mat_and_rhs()
  if (external_loads_ == nullptr)
    external_loads_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*discret_->dof_row_map(), true);

  external_loads_->update(1.0, *contributing_vector, 1.0);
}

/*----------------------------------------------------------------------------*
 * Set external contributions to the system matrix as they appear in meshtying |
 * problems                                                                    |
 *----------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::set_coupling_contributions(
    std::shared_ptr<const Core::LinAlg::SparseOperator> contributing_matrix)
{
  // Setup the "storage" for the coupling matrix contributions in the first step
  if (couplingcontributions_ == nullptr)
  {
    /* The system matrix has a different structure in the meshtying case. To be able to simple add
  the additional contributions to the system matrix within every Newton step, this structure has to
  be the same. Make sure you hand in the correct derived type!
   */
    if (Teuchos::getIntegralValue<Inpar::FLUID::MeshTying>(*params_, "MESHTYING") ==
        Inpar::FLUID::no_meshtying)
    {
      if (std::dynamic_pointer_cast<const Core::LinAlg::SparseMatrix>(contributing_matrix) ==
          nullptr)
        FOUR_C_THROW(
            "In the none-meshtying case you need to hand in a Core::LinAlg::SparseMatrx for the "
            "behavior "
            "to be defined!");

      couplingcontributions_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *discret_->dof_row_map(), 30, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
    }
    else
    {
      couplingcontributions_ = meshtying_->init_system_matrix();
    }
  }
  // Note that we are passing the pointer to the coupling matrix here and do not copy the content!
  // So make sure the matrix contains the correct values at the moment of the assembly procedure!
  couplingcontributions_ = contributing_matrix;
}

/*----------------------------------------------------------------------------*
 * Assemble coupling contributions into the system matrix                     |
 *----------------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::assemble_coupling_contributions()
{
  if (couplingcontributions_ != nullptr)
  {
    // For now we assume to have a linear matrix, so we add the matrix itself to the system
    // matrix
    sysmat_->add(*couplingcontributions_, false, 1.0 / residual_scaling(), 1.0);

    // Add the matrix multiplied with the solution of the last time step to the rhs
    std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
        std::make_shared<Core::LinAlg::Vector<double>>(*discret_->dof_row_map(), true);
    int err = couplingcontributions_->multiply(false, *velnp_, *tmp);

    if (err != 0) FOUR_C_THROW(" Linalg Sparse Matrix Multiply threw error code {} ", err);

    residual_->update(-1.0 / residual_scaling(), *tmp, 1.0);
  }
}
/*----------------------------------------------------------------------*
 | Initialize forcing for HIT and periodic hill                  bk 04/15|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::init_forcing()
{
  // -------------------------------------------------------------------
  // initialize forcing for homogeneous isotropic turbulence
  // -------------------------------------------------------------------
  if (special_flow_ == "forced_homogeneous_isotropic_turbulence" or
      special_flow_ == "scatra_forced_homogeneous_isotropic_turbulence" or
      special_flow_ == "decaying_homogeneous_isotropic_turbulence" or
      special_flow_ == "periodic_hill")
  {
    forcing_ = std::make_shared<Core::LinAlg::Vector<double>>(*(discret_->dof_row_map()), true);

    if (special_flow_ == "forced_homogeneous_isotropic_turbulence" or
        special_flow_ == "scatra_forced_homogeneous_isotropic_turbulence" or
        special_flow_ == "decaying_homogeneous_isotropic_turbulence")
    {
      forcing_interface_ = std::make_shared<FLD::HomoIsoTurbForcing>(*this);
    }
    else if (special_flow_ == "periodic_hill")
      forcing_interface_ = std::make_shared<FLD::PeriodicHillForcing>(*this);
    else
      FOUR_C_THROW("forcing interface doesn't know this flow");
  }
}

/*----------------------------------------------------------------------*
 * Update slave dofs for multifield simulations with fluid mesh tying   |
 *                                                          wirtz 01/16 |
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::update_slave_dof(Core::LinAlg::Vector<double>& f)
{
  if (msht_ != Inpar::FLUID::no_meshtying)
  {
    meshtying_->update_slave_dof(f, *velnp_);
  }
}
/*----------------------------------------------------------------------*|
 *----------------------------------------------------------------------*/
void FLD::FluidImplicitTimeInt::reset_external_forces()
{
  if (external_loads_ == nullptr)
    external_loads_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*discret_->dof_row_map(), true);
  external_loads_->put_scalar(0);
}

FOUR_C_NAMESPACE_CLOSE
