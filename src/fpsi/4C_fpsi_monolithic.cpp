// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fpsi_monolithic.hpp"

#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_constraint_framework_embeddedmesh_solid_to_solid_mortar_manager.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fpsi_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_poroelast_monolithic.hpp"
#include "4C_structure_aux.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

FPSI::MonolithicBase::MonolithicBase(MPI_Comm comm, const Teuchos::ParameterList& fpsidynparams,
    const Teuchos::ParameterList& poroelastdynparams)
    : FpsiBase(comm, fpsidynparams)
{
  // Creation of the subproblems
  // As for general FPSI problems overlapping FSI/FPSI - interfaces can occur, the maps in the
  // different MapExtractors of the Fields are built overlapping! This is necessary to get the right
  // coupling objects, and be able to extract and insert whole FPSI vectors! For building the block
  // matrices of the different fields the following procedure happens to get non overlapping blocks
  // (as wanted):
  // 1. the maps of the FSI and FPSI block are overlapping
  // 2. during the assembling procedure all dofs which belong to the overlapping interface will be
  // assembled into the FSI block (is before the FPSI
  //    condition in the MapExtractors)
  // 3. as there are no entries in of the overlapping interface in the FPSI block, fill_complete()
  // removes them and the FPSI block has no overlapping
  //    entries!

  // --> It's clear that this is not really easily comprehensible, but as the alternatives to this
  // approach requires quite some modifications in 4C
  //     (e.g. extra interface for FSI and FPSI for every Field ...), therefore this version should
  //     be used at the moment.


  // create instance of poroelast subproblem
  poroelast_subproblem_ = std::make_shared<PoroElast::Monolithic>(comm, fpsidynparams, nullptr);
  // ask base algorithm for the fluid time integrator
  Global::Problem* problem = Global::Problem::instance();
  const Teuchos::ParameterList& fluiddynparams = problem->fluid_dynamic_params();
  std::shared_ptr<Adapter::FluidBaseAlgorithm> fluid =
      std::make_shared<Adapter::FluidBaseAlgorithm>(fpsidynparams, fluiddynparams, "fluid", true);
  fluid_subproblem_ = std::dynamic_pointer_cast<Adapter::FluidFPSI>(fluid->fluid_field());
  // ask base algorithm for the ale time integrator
  std::shared_ptr<Adapter::AleBaseAlgorithm> ale = std::make_shared<Adapter::AleBaseAlgorithm>(
      fpsidynparams, Global::Problem::instance()->get_dis("ale"));
  ale_ = std::dynamic_pointer_cast<Adapter::AleFpsiWrapper>(ale->ale_field());
  if (ale_ == nullptr) FOUR_C_THROW("cast from Adapter::Ale to Adapter::AleFpsiWrapper failed");

  coupfa_ = std::make_shared<Coupling::Adapter::Coupling>();

  coupsf_fsi_ = std::make_shared<Coupling::Adapter::Coupling>();
  coupsa_fsi_ = std::make_shared<Coupling::Adapter::Coupling>();
  coupfa_fsi_ = std::make_shared<Coupling::Adapter::Coupling>();
  icoupfa_fsi_ = std::make_shared<Coupling::Adapter::Coupling>();

  FPSI::InterfaceUtils* FPSI_UTILS = FPSI::InterfaceUtils::instance();

  Fluid_PoroFluid_InterfaceMap = FPSI_UTILS->get_fluid_poro_fluid_interface_map();
  PoroFluid_Fluid_InterfaceMap = FPSI_UTILS->get_poro_fluid_fluid_interface_map();

  // build a proxy of the fluid discretization for the ale field
  if (fluid_field()->discretization()->add_dof_set(
          ale_field()->write_access_discretization()->get_dof_set_proxy()) != 1)
  {
    FOUR_C_THROW("unexpected numbers of dofsets in fluid field");
  }
  fluid_field()->discretization()->fill_complete({
      .assign_degrees_of_freedom = true,
      .init_elements = false,
      .do_boundary_conditions = false,
  });
}  // MonolithicBase
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::MonolithicBase::read_restart(int step)
{
  poro_field()->read_restart(step);
  fluid_field()->read_restart(step);
  ale_field()->read_restart(step);

  set_time_step(fluid_field()->time(), fluid_field()->step());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::MonolithicBase::prepare_time_step()
{
  increment_time_and_step();
  print_header();

  poro_field()->prepare_time_step();
  poro_field()->setup_newton();
  fluid_field()->prepare_time_step();
  ale_field()->prepare_time_step();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::MonolithicBase::update()
{
  poro_field()->update();
  fluid_field()->update();
  ale_field()->update();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::MonolithicBase::prepare_output(bool force_prepare)
{
  poro_field()->prepare_output(force_prepare);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::MonolithicBase::output()
{
  poro_field()->output();
  fluid_field()->output();
  ale_field()->output();
}

/*----------------------------------------------------------------------*/
/*                          Coupling Methods                            */
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FPSI::MonolithicBase::fluid_to_ale(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const
{
  return coupfa_->master_to_slave(*iv);
}
std::shared_ptr<Core::LinAlg::Vector<double>> FPSI::MonolithicBase::ale_to_fluid(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const
{
  return coupfa_->slave_to_master(*iv);
}
/// Just in use for problems with FSI-interface ///
std::shared_ptr<Core::LinAlg::Vector<double>> FPSI::MonolithicBase::struct_to_fluid_fsi(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const
{
  return coupsf_fsi_->master_to_slave(*iv);
}
std::shared_ptr<Core::LinAlg::Vector<double>> FPSI::MonolithicBase::fluid_to_struct_fsi(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const
{
  return coupsf_fsi_->slave_to_master(*iv);
}
std::shared_ptr<Core::LinAlg::Vector<double>> FPSI::MonolithicBase::struct_to_ale_fsi(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const
{
  return coupsa_fsi_->master_to_slave(*iv);
}
std::shared_ptr<Core::LinAlg::Vector<double>> FPSI::MonolithicBase::ale_to_struct_fsi(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const
{
  return coupsa_fsi_->slave_to_master(*iv);
}
std::shared_ptr<Core::LinAlg::Vector<double>> FPSI::MonolithicBase::fluid_to_ale_fsi(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const
{
  return coupfa_fsi_->master_to_slave(*iv);
}
std::shared_ptr<Core::LinAlg::Vector<double>> FPSI::MonolithicBase::ale_to_fluid_fsi(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const
{
  return coupfa_fsi_->slave_to_master(*iv);
}
std::shared_ptr<Core::LinAlg::Vector<double>> FPSI::MonolithicBase::ale_to_fluid_interface_fsi(
    std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const
{
  return icoupfa_fsi_->slave_to_master(*iv);
}
/// ---------------------------------------------- ///

//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//<<<<<<<<<<<<<<<<<<<<<<  MonolithicBase -> Monolithic  >>>>>>>>>>>>>>>>>>>>>
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

FPSI::Monolithic::Monolithic(MPI_Comm comm, const Teuchos::ParameterList& fpsidynparams,
    const Teuchos::ParameterList& poroelastdynparams)
    : MonolithicBase(comm, fpsidynparams, poroelastdynparams),
      directsolve_(true),
      printscreen_(true),
      printiter_(true),
      timer_("FPSI Monolithic", true),
      isfirsttimestep_(true),
      islinesearch_(false),
      firstcall_(true)
{
  const Teuchos::ParameterList& sdynparams =
      Global::Problem::instance()->structural_dynamic_params();
  solveradapttol_ = (sdynparams.get<bool>("ADAPTCONV"));
  solveradaptolbetter_ = (sdynparams.get<double>("ADAPTCONV_BETTER"));

  // hydraulic conductivity (needed for coupling in case of probtype fps3i)
  // is overwritten in class fs3i
  conductivity_ = 0.0;

  // Check if FSI-Interface exists and set flag
  // Will be used to jump over all sections, which are just for FSI condensation procedure required!

  if (fluid_field()->interface()->map(FLD::Utils::MapExtractor::cond_fsi)->num_global_elements())
  {
    FSI_Interface_exists_ = true;
    if (Core::Communication::my_mpi_rank(comm) == 0)
      std::cout << "FPSI Calculation will be performed with FSI - Interface!" << std::endl;
  }
  else
  {
    FSI_Interface_exists_ = false;
    if (Core::Communication::my_mpi_rank(comm) == 0)
      std::cout << "FPSI Calculation will skip all FSI parts as there is no FSI - Interface!"
                << std::endl;
  }

  // Check for valid predictors
  if (Teuchos::getIntegralValue<Inpar::Solid::PredEnum>(sdynparams, "PREDICT") !=
      Inpar::Solid::PredEnum::pred_constdis)
    FOUR_C_THROW(
        "No Structural Predictor for FPSI implemented at the moment, choose <PREDICT = ConstDis> "
        "in you input file! \n --> Or feel free to add the missing terms coming from the "
        "predictors "
        "to 4C!");

  const Teuchos::ParameterList& fdynparams = Global::Problem::instance()->fluid_dynamic_params();
  if (fdynparams.get<std::string>("PREDICTOR") != "steady_state")
    FOUR_C_THROW(
        "No Fluid Predictor for FPSI implemented at the moment, choose <PREDICTOR = steady_state> "
        "in you input file! \n --> Or feel free to add the missing terms coming from the "
        "predictors "
        "to 4C!");

  active_FD_check_ = false;  // to avoid adding RHS of firstiter moreoften!
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Monolithic::setup_system()
{
  const int ndim = Global::Problem::instance()->n_dim();

  Coupling::Adapter::Coupling& coupfa = fluid_ale_coupling();

  const Core::LinAlg::Map* fluidnodemap = fluid_field()->discretization()->node_row_map();
  const Core::LinAlg::Map* alenodemap = ale_field()->discretization()->node_row_map();

  coupfa.setup_coupling(*fluid_field()->discretization(), *ale_field()->discretization(),
      *fluidnodemap, *alenodemap, ndim, false);
  fluid_field()->set_mesh_map(coupfa.master_dof_map());

  if (FSI_Interface_exists_) setup_system_fsi();

  // Setup the FPSI Coupling Adapter
  fpsi_coupl() = std::make_shared<FPSI::FpsiCoupling>(poroelast_subproblem_, fluid_subproblem_,
      ale_, Fluid_PoroFluid_InterfaceMap, PoroFluid_Fluid_InterfaceMap);
  fpsi_coupl()->set_conductivity(conductivity_);
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Monolithic::setup_system_fsi()
{
  // create local FPSI MapExtractors as the MapExtractors of the single Fields are nonoverlapping
  // and therefore not the whole FPSI Interface Map is available for FSI, FPSI Interface overlap!!!

  // right now we use matching meshes at the interface

  const int ndim = Global::Problem::instance()->n_dim();

  Coupling::Adapter::Coupling& coupsf_fsi = structure_fluid_coupling_fsi();
  Coupling::Adapter::Coupling& coupsa_fsi = structure_ale_coupling_fsi();
  Coupling::Adapter::Coupling& icoupfa_fsi = interface_fluid_ale_coupling_fsi();

  // structure to fluid
  coupsf_fsi.setup_condition_coupling(*poro_field()->structure_field()->discretization(),
      poro_field()->structure_field()->interface()->fsi_cond_map(),
      *fluid_field()->discretization(), fluid_field()->interface()->fsi_cond_map(), "FSICoupling",
      ndim);
  // structure to ale
  coupsa_fsi.setup_condition_coupling(*poro_field()->structure_field()->discretization(),
      poro_field()->structure_field()->interface()->fsi_cond_map(), *ale_field()->discretization(),
      ale_field()->interface()->fsi_cond_map(), "FSICoupling", ndim);

  // fluid to ale at the interface

  icoupfa_fsi.setup_condition_coupling(*fluid_field()->discretization(),
      fluid_field()->interface()->fsi_cond_map(), *ale_field()->discretization(),
      ale_field()->interface()->fsi_cond_map(), "FSICoupling", ndim);


  // In the following we assume that both couplings find the same dof
  // map at the structural side. This enables us to use just one
  // interface dof map for all fields and have just one transfer
  // operator from the interface map to the full field map.
  if (not coupsf_fsi.master_dof_map()->same_as(*coupsa_fsi.master_dof_map()))
    FOUR_C_THROW("fsi structure interface dof maps do not match");

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Monolithic::timeloop()
{
  prepare_timeloop();

  while (not_finished())  // while step < maxsteps and time < maxtime
  {
    prepare_time_step();
    setup_newton();
    time_step();
    constexpr bool force_prepare = false;
    prepare_output(force_prepare);
    update();
    output();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Monolithic::prepare_timeloop()
{
  // check if maps were destroyed before entering the timeloop
  extractor().check_for_valid_map_extractor();

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

void FPSI::Monolithic::time_step()
{
  //////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////                                                                ///////////////
  ///////////////                                 LOOP                           ///////////////
  ///////////////                                                                ///////////////
  //////////////////////////////////////////////////////////////////////////////////////////////
  while ((((not converged()) and (iter_ <= maximumiterations_)) or (iter_ <= minimumiterations_)) or
         islinesearch_ == true)
  {
    // start time measurement
    timer_.reset();
    Teuchos::Time timer("FPSI Time Step", true);
    evaluate(iterinc_);
    // create full monolithic FPSI right-hand-side vector
    // moved to evaluate()

    // create full monolithic FPSI tangent stiffness matrix and check if it is filled
    setup_system_matrix();

    if (not systemmatrix_->filled())
    {
      FOUR_C_THROW("Effective tangent matrix must be filled here !");
    }
    // (Newton-ready) residual with blanked Dirichlet DOFs (see adapter_timint!)
    // is done in prepare_system_for_newton_solve() within evaluate(iterinc_)
    linear_solve();
    // build norms
    build_convergence_norms();

    // print stuff
    if (islinesearch_ == false) print_newton_iter();

    // reset solver tolerance
    solver_->reset_tolerance();

    // increment equilibrium loop index
    if (islinesearch_ == false)
    {
      iter_ += 1;
      poro_field()->increment_poro_iter();
    }
  }  // end loop

  // correct iteration counter
  iter_ -= 1;

  // test whether max iterations was hit
  if ((converged()) and (Core::Communication::my_mpi_rank(get_comm()) == 0))
  {
    if (linesearch_counter > 0.5)
      std::cout << "            Evaluation of residual with scaled increment yields: " << normofrhs_
                << std::endl;
    islinesearch_ = false;
    linesearch_counter = 0.;
  }
  else if (iter_ >= maximumiterations_)
  {
    FOUR_C_THROW("Newton found no convergence in {} iterations", iter_);
  }

  poro_field()->recover_lagrange_multiplier_after_time_step();

  // recover Lagrange multiplier \lambda_{\Gamma} at the interface at the end of each time step
  // (i.e. condensed traction/forces onto the structure) needed for rhs in next time step
  if (FSI_Interface_exists_)
    recover_lagrange_multiplier();  // Lagrange multiplier of the FSI interface!
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

void FPSI::Monolithic::evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> stepinc)
{
  TEUCHOS_FUNC_TIME_MONITOR("FPSI::Monolithic::Evaluate");

  if (linesearch_ and islinesearch_ == false)
  {
    fluid_field()->discretization()->clear_state();
    linesearch_counter = 0.;
    fluid_field()->discretization()->set_state(0, "dispnp", *fluid_field()->dispnp());
    meshdispold_ = ale_to_fluid(ale_field()->dispnp());
    porointerfacedisplacementsold_ = fpsi_coupl()->i_porostruct_to_ale(
        *poro_field()->structure_field()->extract_interface_dispnp(true));
  }


  std::shared_ptr<const Core::LinAlg::Vector<double>> sx;
  std::shared_ptr<const Core::LinAlg::Vector<double>> fx;
  std::shared_ptr<const Core::LinAlg::Vector<double>> pfx;
  std::shared_ptr<const Core::LinAlg::Vector<double>> ax;

  if (stepinc != nullptr)
  {
    extract_field_vectors(stepinc, sx, pfx, fx, ax, (iter_ == 1 and !active_FD_check_));
  }
  else
  {
    FOUR_C_THROW("No existing increment vector !");
  }

  poro_field()->update_state_incrementally(sx, pfx);
  poro_field()->evaluate_fields(nullptr);
  poro_field()->setup_system_matrix();

  std::shared_ptr<Core::LinAlg::Vector<double>> porointerfacedisplacements_FPSI =
      fpsi_coupl()->i_porostruct_to_ale(
          *poro_field()->structure_field()->extract_interface_dispnp(true));
  ale_field()->apply_interface_displacements(porointerfacedisplacements_FPSI);

  if (FSI_Interface_exists_)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> porointerfacedisplacements_FSI =
        struct_to_ale_fsi(poro_field()->structure_field()->extract_interface_dispnp(false));
    ale_field()->apply_fsi_interface_displacements(porointerfacedisplacements_FSI);
  }

  ale_field()->write_access_dispnp()->update(
      1.0, *ax, 1.0);  // displacement increments on the interfaces are zero!!!
  ale_field()->evaluate(nullptr);

  std::shared_ptr<const Core::LinAlg::Vector<double>> aledisplacements =
      ale_to_fluid(ale_field()->dispnp());
  fluid_field()->apply_mesh_displacement(aledisplacements);

  fluid_field()->update_newton(fx);

  fluid_field()->evaluate(nullptr);

  // Evaluate FPSI Coupling Matrixes and RHS
  fpsi_coupl()->evaluate_coupling_matrixes_rhs();

  setup_rhs(iter_ == 1);

}  // Evaluate

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Monolithic::test_results(MPI_Comm comm)
{
  Global::Problem::instance()->add_field_test(poro_field()->structure_field()->create_field_test());
  Global::Problem::instance()->add_field_test(poro_field()->fluid_field()->create_field_test());
  Global::Problem::instance()->add_field_test(fluid_field()->create_field_test());
  Global::Problem::instance()->test_all(comm);
}

//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//<<<<<<<<<<<<<<              Methods concerning          >>>>>>>>>>>>>>>>>>>>
//<<<<<<<<<<<<<<                    solver                >>>>>>>>>>>>>>>>>>>>
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

void FPSI::Monolithic::setup_solver()
{
  const Teuchos::ParameterList& fpsidynamicparams =
      Global::Problem::instance()->fpsi_dynamic_params();

  const int linsolvernumber = fpsidynamicparams.get<int>("LINEAR_SOLVER");
  if (linsolvernumber == (-1))
    FOUR_C_THROW(
        "No linear solver defined for FPSI problem. Please set LINEAR_SOLVER in FPSI DYNAMIC to a "
        "valid number !");

  const Teuchos::ParameterList& solverparams =
      Global::Problem::instance()->solver_params(linsolvernumber);
  const auto solvertype =
      Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(solverparams, "SOLVER");

  directsolve_ = (solvertype == Core::LinearSolver::SolverType::umfpack or
                  solvertype == Core::LinearSolver::SolverType::superlu);

  if (directsolve_)
    solver_ = std::make_shared<Core::LinAlg::Solver>(solverparams, get_comm(),
        Global::Problem::instance()->solver_params_callback(),
        Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
            Global::Problem::instance()->io_params(), "VERBOSITY"));
  else
    // create a linear solver
    create_linear_solver();

  // Get the parameters for the Newton iteration
  maximumiterations_ = fpsidynamicparams.get<int>("ITEMAX");
  minimumiterations_ = fpsidynamicparams.get<int>("ITEMIN");
  normtypeinc_ =
      Teuchos::getIntegralValue<Inpar::FPSI::ConvergenceNorm>(fpsidynamicparams, "NORM_INC");
  normtypefres_ =
      Teuchos::getIntegralValue<Inpar::FPSI::ConvergenceNorm>(fpsidynamicparams, "NORM_RESF");
  combinedconvergence_ =
      Teuchos::getIntegralValue<Inpar::FPSI::BinaryOp>(fpsidynamicparams, "NORMCOMBI_RESFINC");

  {
    std::istringstream tolresstream(
        Teuchos::getNumericStringParameter(fpsidynamicparams, "RESTOL"));
    std::string word;
    while (tolresstream >> word)
    {
      toleranceresidualforceslist_.push_back(std::atof(word.c_str()));
    }
    toleranceresidualforces_ = toleranceresidualforceslist_[0];

    std::istringstream tolincstream(
        Teuchos::getNumericStringParameter(fpsidynamicparams, "INCTOL"));
    while (tolincstream >> word)
    {
      toleranceiterinclist_.push_back(std::atof(word.c_str()));
    }
    toleranceiterinc_ = toleranceiterinclist_[0];
  }

  Global::Problem* problem = Global::Problem::instance();
  const Teuchos::ParameterList& fpsidynparams = problem->fpsi_dynamic_params();
  linesearch_ = fpsidynparams.get<bool>("LineSearch");
  if (linesearch_)
    FOUR_C_THROW(
        "Parameter 'LineSearch' is set to 'Yes' in the FPSI Dynamic section in your input-file.  \n"
        "Though the framework for a line search algorithm is implemented in fpsi_monolithic.cpp, \n"
        "a proper routine to reset the participating single fields is still required. In Chuck's \n"
        "experimental 4C this was solved by performing an evaluate with the negative increment.\n"
        "However this has not yet been committed.\n");
  linesearch_counter = 0.;

  return;
}

/*----------------------------------------------------------------------*
 | create linear solver                                   vuong 08/15 |
 *----------------------------------------------------------------------*/
void FPSI::Monolithic::create_linear_solver()
{
  // get dynamic section
  const Teuchos::ParameterList& fpsidyn = Global::Problem::instance()->fpsi_dynamic_params();

  // get the linear solver number
  const int linsolvernumber = fpsidyn.get<int>("LINEAR_SOLVER");
  if (linsolvernumber == (-1))
    FOUR_C_THROW(
        "No linear solver defined for FPSI problem. Please set LINEAR_SOLVER in FPSI DYNAMIC to a "
        "valid number !");

  // get parameter list of structural dynamics
  const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
  // use solver blocks for structure
  // get the solver number used for structural solver
  const int slinsolvernumber = sdyn.get<int>("LINEAR_SOLVER");
  // check if the structural solver has a valid solver number
  if (slinsolvernumber == (-1))
    FOUR_C_THROW(
        "no linear solver defined for structural field. Please set LINEAR_SOLVER in STRUCTURAL "
        "DYNAMIC to a valid number!");

  // get parameter list of fluid dynamics
  const Teuchos::ParameterList& fdyn = Global::Problem::instance()->fluid_dynamic_params();
  // use solver blocks for fluid
  // get the solver number used for fluid solver
  const int flinsolvernumber = fdyn.get<int>("LINEAR_SOLVER");
  // check if the fluid solver has a valid solver number
  if (flinsolvernumber == (-1))
    FOUR_C_THROW(
        "no linear solver defined for fluid field. Please set LINEAR_SOLVER in FLUID DYNAMIC to a "
        "valid number!");

  // get parameter list of structural dynamics
  const Teuchos::ParameterList& aledyn = Global::Problem::instance()->ale_dynamic_params();
  // use solver blocks for structure
  // get the solver number used for structural solver
  const int alinsolvernumber = aledyn.get<int>("LINEAR_SOLVER");
  // check if the structural solver has a valid solver number
  if (alinsolvernumber == (-1))
    FOUR_C_THROW(
        "no linear solver defined for ALE field. Please set LINEAR_SOLVER in ALE DYNAMIC to a "
        "valid number!");

  // get solver parameter list of linear Poroelasticity solver
  const Teuchos::ParameterList& fpsisolverparams =
      Global::Problem::instance()->solver_params(linsolvernumber);

  const auto solvertype =
      Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(fpsisolverparams, "SOLVER");

  if (solvertype != Core::LinearSolver::SolverType::belos)
  {
    std::cout << "!!!!!!!!!!!!!!!!!!!!!! ATTENTION !!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << " Note: the BGS2x2 preconditioner now " << std::endl;
    std::cout << " uses the structural solver and fluid solver blocks" << std::endl;
    std::cout << " for building the internal inverses" << std::endl;
    std::cout << " Remove the old BGS PRECONDITIONER BLOCK entries " << std::endl;
    std::cout << " in the input files!" << std::endl;
    std::cout << "!!!!!!!!!!!!!!!!!!!!!! ATTENTION !!!!!!!!!!!!!!!!!!!!!" << std::endl;
    FOUR_C_THROW("Iterative solver expected");
  }
  const auto azprectype =
      Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(fpsisolverparams, "AZPREC");

  // plausibility check
  switch (azprectype)
  {
    case Core::LinearSolver::PreconditionerType::block_teko:
    {
      // no plausibility checks here
      // if you forget to declare an xml file you will get an error message anyway
    }
    break;
    default:
      FOUR_C_THROW("AMGnxn preconditioner expected");
      break;
  }

  solver_ = std::make_shared<Core::LinAlg::Solver>(fpsisolverparams, get_comm(),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));

  // use solver blocks for structure and fluid
  const Teuchos::ParameterList& ssolverparams =
      Global::Problem::instance()->solver_params(slinsolvernumber);
  const Teuchos::ParameterList& fsolverparams =
      Global::Problem::instance()->solver_params(flinsolvernumber);
  const Teuchos::ParameterList& asolverparams =
      Global::Problem::instance()->solver_params(alinsolvernumber);

  // for now, use same solver parameters for poro fluid and free fluid

  // poro/structure
  solver_->put_solver_params_to_sub_params("Inverse1", ssolverparams,
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"),
      get_comm());
  // poro fluid
  solver_->put_solver_params_to_sub_params("Inverse2", fsolverparams,
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"),
      get_comm());
  // fluid
  solver_->put_solver_params_to_sub_params("Inverse3", fsolverparams,
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"),
      get_comm());
  // ale
  solver_->put_solver_params_to_sub_params("Inverse4", asolverparams,
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"),
      get_comm());

  // prescribe rigid body modes
  Core::LinearSolver::Parameters::compute_solver_parameters(
      *poro_field()->structure_field()->discretization(), solver_->params().sublist("Inverse1"));
  Core::LinearSolver::Parameters::compute_solver_parameters(
      *poro_field()->fluid_field()->discretization(), solver_->params().sublist("Inverse2"));
  Core::LinearSolver::Parameters::compute_solver_parameters(
      *fluid_field()->discretization(), solver_->params().sublist("Inverse3"));
  Core::LinearSolver::Parameters::compute_solver_parameters(
      *ale_field()->write_access_discretization(), solver_->params().sublist("Inverse4"));

  // fixing length of Inverse1 nullspace (solver/preconditioner ML)
  {
    std::string inv = "Inverse1";
    const Core::LinAlg::Map& oldmap =
        *(Global::Problem::instance()->get_dis("structure")->dof_row_map());
    const Core::LinAlg::Map& newmap =
        Core::LinAlg::Map(systemmatrix_->matrix(structure_block_, structure_block_).row_map());
    Core::LinearSolver::Parameters::fix_null_space(
        inv.data(), oldmap, newmap, solver_->params().sublist("Inverse1"));
  }
  // fixing length of Inverse2 nullspace (solver/preconditioner ML)
  {
    std::string inv = "Inverse2";
    const Core::LinAlg::Map& oldmap = *(poro_field()->fluid_field()->dof_row_map());
    const Core::LinAlg::Map& newmap =
        Core::LinAlg::Map(systemmatrix_->matrix(porofluid_block_, porofluid_block_).row_map());
    Core::LinearSolver::Parameters::fix_null_space(
        inv.data(), oldmap, newmap, solver_->params().sublist("Inverse2"));
  }
  // fixing length of Inverse3 nullspace (solver/preconditioner ML)
  {
    std::string inv = "Inverse3";
    const Core::LinAlg::Map& oldmap = *(fluid_field()->dof_row_map());
    const Core::LinAlg::Map& newmap =
        Core::LinAlg::Map(systemmatrix_->matrix(fluid_block_, fluid_block_).row_map());
    Core::LinearSolver::Parameters::fix_null_space(
        inv.data(), oldmap, newmap, solver_->params().sublist("Inverse3"));
  }
  // fixing length of Inverse4 nullspace (solver/preconditioner ML)
  {
    std::string inv = "Inverse4";
    const Core::LinAlg::Map& oldmap = *(ale_field()->dof_row_map());
    const Core::LinAlg::Map& newmap =
        Core::LinAlg::Map(systemmatrix_->matrix(ale_i_block_, ale_i_block_).row_map());
    Core::LinearSolver::Parameters::fix_null_space(
        inv.data(), oldmap, newmap, solver_->params().sublist("Inverse4"));
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Monolithic::linear_solve()
{
  Core::LinAlg::SolverParams solver_params;
  if (solveradapttol_ and (iter_ > 1))
  {
    solver_params.nonlin_tolerance = normofrhs_;
    solver_params.nonlin_residual = toleranceresidualforces_;
    solver_params.lin_tol_better = solveradaptolbetter_;
  }
  Global::Problem* problem = Global::Problem::instance();
  const Teuchos::ParameterList& fpsidynparams = problem->fpsi_dynamic_params();
  if (fpsidynparams.get<bool>("FDCheck"))
  {
    fpsifd_check();
  }

  iterinc_->put_scalar(0.0);  // Useful? depends on solver and more
  poro_field()->clear_poro_iterinc();

  if (directsolve_)
  {
    std::shared_ptr<Core::LinAlg::SparseMatrix> sparse = systemmatrix_->merge();

    if (FSI_Interface_exists_)
    {
      // remove entries in condensed dofs from matrix and rhs...
      Core::LinAlg::apply_dirichlet_to_system(
          *sparse, *iterinc_, *rhs_, *zeros_, *fluid_field()->interface()->fsi_cond_map());
    }

    Core::LinAlg::apply_dirichlet_to_system(
        *sparse, *iterinc_, *rhs_, *zeros_, *combined_dbc_map());

    // line search
    if (linesearch_)
      line_search(*sparse);
    else
    {
      // standard solver call
      solver_params.refactor = true;
      solver_params.reset = iter_ == 1;
      solver_->solve(sparse, iterinc_, rhs_, solver_params);
    }
  }
  else
  {
    if (FSI_Interface_exists_)
    {
      // remove entries in condensed dofs from matrix and rhs...
      Core::LinAlg::apply_dirichlet_to_system(
          *systemmatrix_, *iterinc_, *rhs_, *zeros_, *fluid_field()->interface()->fsi_cond_map());
    }

    Core::LinAlg::apply_dirichlet_to_system(
        *systemmatrix_, *iterinc_, *rhs_, *zeros_, *combined_dbc_map());

    // standard solver call
    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    solver_params.reset = iter_ == 1;
    solver_->solve(systemmatrix_, iterinc_, rhs_, solver_params);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Monolithic::line_search(Core::LinAlg::SparseMatrix& sparse)
{
  // Note: the line search code seems to be experimental and is not
  // working properly (perhaps just a sign is wrong somewhere ...)
  // I would not recommend to use it without thorough testing
  // vuong 08/15

  if (iter_ > 1)
  {
    rhs_->norm_2(&normofrhs_);
    if (normofrhs_ - normofrhsold_ > 1e-13)
    {
      if (linesearch_counter > 0.5)
        std::cout << "            Evaluation of residual with bisected increment yields: "
                  << normofrhs_ << std::endl;

      islinesearch_ = true;
      iterinc_->update(pow(0.5, (linesearch_counter)), *iterincold_, 0.0);
      linesearch_counter = linesearch_counter + 1.0;
      std::cout << "linesearch_ : " << std::setprecision(1)
                << static_cast<int>(linesearch_counter + 0.5) << " iterinc_ multiplied by "
                << std::setprecision(4) << pow(0.5, linesearch_counter)
                << "   residual = " << normofrhs_ << " > " << normofrhsold_ << std::endl;

      // subtract the old interinc_ from all fields (undo the update)
      std::shared_ptr<Core::LinAlg::Vector<double>> sx;
      std::shared_ptr<Core::LinAlg::Vector<double>> pfx;
      std::shared_ptr<Core::LinAlg::Vector<double>> fx;
      std::shared_ptr<const Core::LinAlg::Vector<double>> constsx;
      std::shared_ptr<const Core::LinAlg::Vector<double>> constfpx;
      std::shared_ptr<const Core::LinAlg::Vector<double>> constfx;
      std::shared_ptr<const Core::LinAlg::Vector<double>> ax;

      sx = std::make_shared<Core::LinAlg::Vector<double>>(
          *poro_field()->structure_field()->dof_row_map(), true);
      pfx = std::make_shared<Core::LinAlg::Vector<double>>(
          *poro_field()->fluid_field()->dof_row_map(), true);
      fx = std::make_shared<Core::LinAlg::Vector<double>>(*fluid_field()->dof_row_map(), true);

      extract_field_vectors(iterinc_, constsx, constfpx, constfx, ax, iter_ == 1);
      iterinc_->norm_2(&normofiterinc_);
      std::cout << "            Norm of step back: " << normofiterinc_ << std::endl;
      // poro_field()  ->ResetNewton(sx);
      // fluid_field() ->ResetNewton(fx);
      sx->update(1.0, *constsx, 0.0);
      pfx->update(1.0, *constfpx, 0.0);
      fx->update(1.0, *constfx, 0.0);
      sx->scale(-1.0);
      pfx->scale(-1.0);
      fx->scale(-1.0);

      poro_field()->update_state_incrementally(sx, pfx);
      poro_field()->evaluate_fields(nullptr);
      poro_field()->setup_system_matrix();

      fluid_field()->update_newton(
          std::dynamic_pointer_cast<const Core::LinAlg::Vector<double>>(fx));
      // ale_field()   ->ResetNewton(ax);

      fluid_field()->apply_mesh_displacement(meshdispold_);
      ale_field()->apply_interface_displacements(porointerfacedisplacementsold_);


      // set iterinc_ to a fraction of the old iterinc_
      iterinc_->update(pow(0.5, linesearch_counter), *iterincold_, 0.0);
      iterinc_->norm_2(&normofiterinc_);
      std::cout << "            Norm of old increment: " << normofiterincold_
                << "  Norm of bisected increment: " << normofiterinc_ << std::endl;
    }
    else
    {
      islinesearch_ = false;
      if (linesearch_counter > 0.5)
        std::cout << "            Evaluation of residual with bisected increment yields: "
                  << normofrhs_ << std::endl;
      linesearch_counter = 0.0;
    }
  }

  // prepare linesearch_
  // copy the old iterinc_ before new solve
  if (linesearch_ and islinesearch_ == false)
  {
    rhsold_ = Core::LinAlg::create_vector(*dof_row_map(), true);
    rhsold_->update(1.0, *rhs_, 0.0);
    rhsold_->norm_2(&normofrhsold_);
    if (abs(normofrhs_ - normofrhsold_) > 1.e-12 and iter_ > 1)
      FOUR_C_THROW(" wrong copy of rhs_ ");
  }
  // end prepare linesearch_

  // standard solver call
  if (islinesearch_ == false)
  {
    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    solver_params.reset = iter_ == 1;
    solver_->solve(Core::Utils::shared_ptr_from_ref(sparse), iterinc_, rhs_, solver_params);
  }

  if (islinesearch_ == false)
  {
    // check whether iterinc_ points in right direction
    std::shared_ptr<Core::LinAlg::Vector<double>> tempvec =
        Core::LinAlg::create_vector(*dof_row_map(), true);
    sparse.multiply(true, *rhs_, *tempvec);
    double climb = 0.0;
    tempvec->dot(*iterinc_, &climb);
    climb = -climb;

    if (climb > 0.0)
    {
      std::cout << "########################################################################"
                << std::endl;
      std::cout << "##                                                                    ##"
                << std::endl;
      std::cout << "## WARNING: A*x-b=0 ; A^T*b*x > 0 ; increment vector multiplied by -1 ##"
                << std::endl;
      std::cout << "##                                                                    ##"
                << std::endl;
      std::cout << "##                       Value = " << std::setprecision(9) << climb << "    ##"
                << std::endl;
      std::cout << "##                                                                    ##"
                << std::endl;
      std::cout << "########################################################################"
                << std::endl;
      iterinc_->update(-1.0, *iterinc_, 0.0);
    }
  }

  if (linesearch_ and islinesearch_ == false)
  {
    iterincold_ = Core::LinAlg::create_vector(*dof_row_map(), true);
    iterincold_->update(1.0, *iterinc_, 0.0);
    iterincold_->norm_2(&normofiterincold_);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> FPSI::Monolithic::combined_dbc_map()
{
  const std::shared_ptr<const Core::LinAlg::Map> scondmap =
      poro_field()->structure_field()->get_dbc_map_extractor()->cond_map();
  const std::shared_ptr<const Core::LinAlg::Map> pfcondmap =
      poro_field()->fluid_field()->get_dbc_map_extractor()->cond_map();
  const std::shared_ptr<const Core::LinAlg::Map> fcondmap =
      fluid_field()->get_dbc_map_extractor()->cond_map();
  const std::shared_ptr<const Core::LinAlg::Map> acondmap =
      ale_field()->get_dbc_map_extractor()->cond_map();
  std::shared_ptr<Core::LinAlg::Map> tempmap = Core::LinAlg::merge_map(scondmap, pfcondmap, false);
  std::shared_ptr<Core::LinAlg::Map> condmap_0 = Core::LinAlg::merge_map(tempmap, fcondmap, false);
  std::shared_ptr<Core::LinAlg::Map> condmap = Core::LinAlg::merge_map(condmap_0, acondmap, false);

  return condmap;
}  // combined_dbc_map()

//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//<<<<<<<<<<<<<<                 Newton Loop              >>>>>>>>>>>>>>>>>>>>
//<<<<<<<<<<<<<<                   Methods                >>>>>>>>>>>>>>>>>>>>
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool FPSI::Monolithic::converged()
{
  // check for single norms
  bool convinc = false;
  bool convfres = false;

  // residual increments
  switch (normtypeinc_)
  {
    case Inpar::FPSI::absoluteconvergencenorm:
      convinc = normofiterinc_ < toleranceiterinc_;
      break;
    case Inpar::FPSI::absoluteconvergencenorm_sys_split:
      FOUR_C_THROW(
          "Check for convergence of primary variables with type <absolute sys split> not "
          "implemented yet!");
      break;
    case Inpar::FPSI::relativconvergencenorm_sys:
      // increment convergence is checked relative to the average dofs of the different fields
      std::cout << " |ps| " << norm1_ps_ << " |pfv| " << norm1_pfv_ << " |pfp| " << norm1_pfp_
                << " |fv| " << norm1_fv_ << " |fp| " << norm1_fp_ << " |a| " << norm1_a_
                << std::endl;
      convinc =
          ((normofiterincporostruct_ / norm1_ps_ * sqrtnps_ < toleranceiterinc_) and  // poro struct
              (normofiterincporofluidvelocity_ / norm1_pfv_ * sqrtnpfv_ <
                  toleranceiterinc_) and  // poro fluid velocity
              (normofiterincporofluidpressure_ / norm1_pfp_ * sqrtnpfp_ <
                  toleranceiterinc_) and  // poro fluid pressure
              (normofiterincale_ / norm1_a_ * sqrtna_ < toleranceiterinc_) and  // ale
              (normofiterincfluidvelocity_ / norm1_fv_ * sqrtnfv_ <
                  toleranceiterinc_) and  // fluid velocity
              (normofiterincfluidpressure_ / norm1_fp_ * sqrtnfp_ <
                  toleranceiterinc_)  // fluid pressure
          );
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of primary variables for any reason :-p !");
      break;
  }
  // residual forces
  switch (normtypefres_)
  {
    case Inpar::FPSI::absoluteconvergencenorm:
      convfres = normofrhs_ < toleranceresidualforces_;
      break;
    case Inpar::FPSI::absoluteconvergencenorm_sys_split:
      convfres = ((normrhsporostruct_ / sqrtnps_ < toleranceresidualforceslist_[2]) and
                  (normrhsfluidvelocity_ / sqrtnpfv_ < toleranceresidualforceslist_[0]) and
                  (normrhsporofluidpressure_ / sqrtnpfp_ < toleranceresidualforceslist_[1]) and
                  (normrhsale_ / sqrtna_ < toleranceresidualforceslist_[5]) and
                  (normrhsfluidvelocity_ / sqrtnfv_ < toleranceresidualforceslist_[3]) and
                  (normrhsfluidpressure_ / sqrtnfp_ < toleranceresidualforceslist_[4]));
      break;
    case Inpar::FPSI::relativconvergencenorm_sys:
      FOUR_C_THROW(
          "Check for convergence of residual forces with type <relative_sys> not implemented yet!");
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of residual forces for any reason :-P !");
      break;
  }

  // combine increments and forces
  bool converged = false;
  if (combinedconvergence_ == Inpar::FPSI::bop_and)
    converged = convinc and convfres;
  else if (combinedconvergence_ == Inpar::FPSI::bop_or)
    converged = convinc or convfres;
  else
    FOUR_C_THROW("Something went terribly wrong with binary operator!");

  // return things
  return converged;
}  // Converged()


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Monolithic::build_convergence_norms()
{
  rhs_->norm_2(&normofrhs_);
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_fluid;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_fluidvelocity;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_fluidpressure;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_porofluidvelocity;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_porofluidpressure;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_porointerface;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_fluidinterface;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_porofluid;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_porostruct;
  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_ale;


  rhs_porostruct = extractor().extract_vector(*rhs_, structure_block_);
  rhs_porofluid = extractor().extract_vector(*rhs_, porofluid_block_);
  rhs_porofluidvelocity = poro_field()->fluid_field()->extract_velocity_part(rhs_porofluid);
  rhs_porofluidpressure = poro_field()->fluid_field()->extract_pressure_part(rhs_porofluid);
  rhs_porointerface =
      fpsi_coupl()->poro_fluid_fpsi_vel_pres_extractor()->extract_cond_vector(*rhs_porofluid);

  rhs_fluid = extractor().extract_vector(*rhs_, fluid_block_);
  //  std::shared_ptr<const Core::LinAlg::Vector<double>> rhs_fullfluid = Teuchos::rcp(new
  //  Core::LinAlg::Vector<double>(*fluid_field()->dof_row_map())); std::shared_ptr<const
  //  Core::LinAlg::Vector<double>> rhs_fsi = Teuchos::rcp(new
  //  Core::LinAlg::Vector<double>(*fluid_field()->Interface()->Map(FLD::Utils::MapExtractor::cond_fsi),true));
  //  rhs_fullfluid = Core::LinAlg::MergeVector(rhs_fluid,rhs_fsi,false);

  rhs_fluidvelocity = fluid_field()->extract_velocity_part(rhs_fluid);
  rhs_fluidpressure = fluid_field()->extract_pressure_part(rhs_fluid);
  rhs_fluidinterface =
      fpsi_coupl()->fluid_fpsi_vel_pres_extractor()->extract_cond_vector(*rhs_fluid);

  rhs_ale =
      extractor().extract_vector(*rhs_, ale_i_block_);  // Extractor().extract_vector(*rhs_, 2);

  rhs_porostruct->norm_2(&normrhsporostruct_);
  rhs_fluid->norm_2(&normrhsfluid_);
  rhs_fluidvelocity->norm_2(&normrhsfluidvelocity_);
  rhs_fluidpressure->norm_2(&normrhsfluidpressure_);
  rhs_porofluidvelocity->norm_2(&normrhsporofluidvelocity_);
  rhs_porofluidpressure->norm_2(&normrhsporofluidpressure_);
  rhs_porointerface->norm_2(&normrhsporointerface_);
  rhs_fluidinterface->norm_2(&normrhsfluidinterface_);
  rhs_ale->norm_2(&normrhsale_);

  // get length of the porostructural, porofluid, fluid and ale vector
  sqrtnfv_ = rhs_fluidvelocity->global_length();  // correct length here
  sqrtnfp_ = rhs_fluidpressure->global_length();
  sqrtnpfv_ = rhs_porofluidvelocity->global_length();
  sqrtnpfp_ = rhs_porofluidpressure->global_length();
  sqrtnps_ = rhs_porostruct->global_length();
  sqrtna_ = rhs_ale->global_length();
  sqrtnall_ = sqrtnfv_ + sqrtnfp_ + sqrtnpfv_ + sqrtnpfp_ + sqrtnps_ + sqrtna_;

  sqrtnfv_ = sqrt(sqrtnfv_);
  sqrtnfp_ = sqrt(sqrtnfp_);
  sqrtnpfv_ = sqrt(sqrtnpfv_);
  sqrtnpfp_ = sqrt(sqrtnpfp_);
  sqrtnps_ = sqrt(sqrtnps_);
  sqrtna_ = sqrt(sqrtna_);
  sqrtnall_ = sqrt(sqrtnall_);

  if (islinesearch_ == false) iterinc_->norm_2(&normofiterinc_);

  std::shared_ptr<const Core::LinAlg::Vector<double>> iterincporostruct;
  std::shared_ptr<const Core::LinAlg::Vector<double>> iterincporofluid;
  std::shared_ptr<const Core::LinAlg::Vector<double>> iterincfluid;
  std::shared_ptr<const Core::LinAlg::Vector<double>> iterincfluidvelocity;
  std::shared_ptr<const Core::LinAlg::Vector<double>> iterincfluidpressure;
  std::shared_ptr<const Core::LinAlg::Vector<double>> iterincporofluidvelocity;
  std::shared_ptr<const Core::LinAlg::Vector<double>> iterincporofluidpressure;
  std::shared_ptr<const Core::LinAlg::Vector<double>> iterincale;
  std::shared_ptr<const Core::LinAlg::Vector<double>> iterincporointerface;
  std::shared_ptr<const Core::LinAlg::Vector<double>> iterincfluidinterface;

  iterincporostruct = extractor().extract_vector(*iterinc_, structure_block_);
  iterincporofluid = extractor().extract_vector(*iterinc_, porofluid_block_);
  iterincporofluidvelocity = poro_field()->fluid_field()->extract_velocity_part(iterincporofluid);
  iterincporofluidpressure = poro_field()->fluid_field()->extract_pressure_part(iterincporofluid);
  iterincporointerface =
      fpsi_coupl()->poro_fluid_fpsi_vel_pres_extractor()->extract_cond_vector(*iterincporofluid);

  iterincfluid = extractor().extract_vector(*iterinc_, fluid_block_);

  //  std::shared_ptr<const Core::LinAlg::Vector<double>> iterincfullfluid = Teuchos::rcp(new
  //  Core::LinAlg::Vector<double>(*fluid_field()->dof_row_map())); std::shared_ptr<const
  //  Core::LinAlg::Vector<double>> iterincfsi = Teuchos::rcp(new
  //  Core::LinAlg::Vector<double>(*fluid_field()->Interface()->Map(FLD::Utils::MapExtractor::cond_fsi),true));
  //  iterincfullfluid = Core::LinAlg::MergeVector(iterincfluid,iterincfsi,false);

  iterincfluidvelocity = fluid_field()->extract_velocity_part(iterincfluid);
  iterincfluidpressure = fluid_field()->extract_pressure_part(iterincfluid);
  iterincfluidinterface =
      fpsi_coupl()->fluid_fpsi_vel_pres_extractor()->extract_cond_vector(*iterincfluid);

  iterincale = extractor().extract_vector(*iterinc_, ale_i_block_);

  iterincporostruct->norm_2(&normofiterincporostruct_);
  iterincporofluid->norm_2(&normofiterincporofluid_);
  iterincporofluidvelocity->norm_2(&normofiterincporofluidvelocity_);
  iterincporofluidpressure->norm_2(&normofiterincporofluidpressure_);
  iterincporointerface->norm_2(&normofiterincporointerface_);

  iterincfluid->norm_2(&normofiterincfluid_);
  iterincfluidvelocity->norm_2(&normofiterincfluidvelocity_);
  iterincfluidpressure->norm_2(&normofiterincfluidpressure_);
  iterincfluidinterface->norm_2(&normofiterincfluidinterface_);

  iterincale->norm_2(&normofiterincale_);

  // Get Norm1 of dof values for each field

  std::shared_ptr<const Core::LinAlg::Vector<double>> porofluiddof;
  std::shared_ptr<const Core::LinAlg::Vector<double>> porofluidvelocity;
  std::shared_ptr<const Core::LinAlg::Vector<double>> porofluidpressure;
  std::shared_ptr<const Core::LinAlg::Vector<double>> porostructdisplacements;
  std::shared_ptr<const Core::LinAlg::Vector<double>> fluiddof;
  std::shared_ptr<const Core::LinAlg::Vector<double>> fluidvelocity;
  std::shared_ptr<const Core::LinAlg::Vector<double>> fluidpressure;
  std::shared_ptr<const Core::LinAlg::Vector<double>> aledisplacements;
  std::shared_ptr<const Core::LinAlg::Vector<double>> porointerface;
  std::shared_ptr<const Core::LinAlg::Vector<double>> fluidinterface;

  porofluiddof = poro_field()->fluid_field()->velnp();
  porofluidvelocity = poro_field()->fluid_field()->extract_velocity_part(porofluiddof);
  porofluidpressure = poro_field()->fluid_field()->extract_pressure_part(porofluiddof);
  porostructdisplacements = poro_field()->structure_field()->dispnp();
  fluiddof = fluid_field()->velnp();
  fluidvelocity = fluid_field()->extract_velocity_part(fluiddof);
  fluidpressure = fluid_field()->extract_pressure_part(fluiddof);
  aledisplacements = ale_field()->dispnp();


  porofluidvelocity->norm_1(&norm1_pfv_);
  porofluidpressure->norm_1(&norm1_pfp_);
  porostructdisplacements->norm_1(&norm1_ps_);
  fluidvelocity->norm_1(&norm1_fv_);
  fluidpressure->norm_1(&norm1_fp_);
  aledisplacements->norm_1(&norm1_a_);
  norm1_alldof_ = norm1_pfv_ + norm1_pfp_ + norm1_ps_ + norm1_fv_ + norm1_fp_ + norm1_a_;

  // add small number to avoid division by 0 --> division by 10^-10 results anyway in 'not
  // converged'
  norm1_alldof_ += 1e-10;
  norm1_pfv_ += 1e-10;
  norm1_pfp_ += 1e-10;
  norm1_ps_ += 1e-10;
  norm1_fv_ += 1e-10;
  norm1_fp_ += 1e-10;
  norm1_a_ += 1e-10;

  return;
}  // build_convergence_norms

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Monolithic::print_newton_iter()
{
  // print to standard out
  // replace myrank_ here general by Core::Communication::my_mpi_rank(Comm())
  if ((Core::Communication::my_mpi_rank(get_comm()) == 0) and printscreen_ and
      (step() % printscreen_ == 0) and printiter_)
  {
    if (iter_ == 1) print_newton_iter_header(stdout);
    print_newton_iter_text(stdout);
  }

}  // print_newton_iter()

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Monolithic::print_newton_iter_header(FILE* ofile)
{
  // open outstringstream
  std::ostringstream oss;

  // enter converged state etc
  oss << std::setw(8) << "numiter";

  // different style due relative or absolute error checking
  // displacement
  switch (normtypefres_)
  {
    case Inpar::FPSI::absoluteconvergencenorm:
      oss << std::setw(11) << "abs-res";
      break;
    case Inpar::FPSI::absoluteconvergencenorm_sys_split:
      oss << std::setw(11) << "abs-res_s";
      break;
    case Inpar::FPSI::relativconvergencenorm_sys:
      oss << std::setw(11) << "rel-res_s";
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  switch (normtypeinc_)
  {
    case Inpar::FPSI::absoluteconvergencenorm:
      oss << std::setw(11) << "abs-inc";
      break;
    case Inpar::FPSI::absoluteconvergencenorm_sys_split:
      oss << std::setw(11) << "abs-inc_s";
      break;
    case Inpar::FPSI::relativconvergencenorm_sys:
      oss << std::setw(11) << "rel-inc_s";
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  switch (normtypefres_)
  {
    case Inpar::FPSI::absoluteconvergencenorm:
    case Inpar::FPSI::absoluteconvergencenorm_sys_split:
    case Inpar::FPSI::relativconvergencenorm_sys:
      oss << std::setw(16) << "poro-s-res";
      // oss <<std::setw(15)<< "abs-f-res";
      oss << std::setw(15) << "poro-fvel-res";
      oss << std::setw(15) << "poro-fpres-res";
      oss << std::setw(15) << "fld-fvel-res";
      oss << std::setw(15) << "fld-fpres-res";
      // oss <<std::setw(15)<< "pinterface-res";
      // oss <<std::setw(15)<< "finterface-res";
      break;
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  switch (normtypeinc_)
  {
    case Inpar::FPSI::absoluteconvergencenorm:
    case Inpar::FPSI::absoluteconvergencenorm_sys_split:
      oss << std::setw(15) << "poro-s-inc";
      // oss <<std::setw(15)<< "abs-f-inc";
      oss << std::setw(16) << "poro-fvel-inc";
      oss << std::setw(16) << "poro-fpres-inc";
      oss << std::setw(15) << "fld-fvel-inc";
      oss << std::setw(15) << "fld-fpres-inc";
      oss << std::setw(11) << "ale-inc";
      oss << std::setw(14) << "poro-int-inc";
      oss << std::setw(14) << "fld-int-inc";
      break;
    case Inpar::FPSI::relativconvergencenorm_sys:
      oss << std::setw(15) << "poro-s-inc";
      // oss <<std::setw(15)<< "abs-f-inc";
      oss << std::setw(16) << "poro-fvel-inc";
      oss << std::setw(16) << "poro-fpres-inc";
      oss << std::setw(15) << "fld-fvel-inc";
      oss << std::setw(15) << "fld-fpres-inc";
      oss << std::setw(11) << "ale-inc";
      break;
    default:
      FOUR_C_THROW("Begin at the beginning and go on till you come to the end. Then stop.");
      break;
  }

  //  // add solution time
  //  oss << std::setw(14)<< "wct";

  // finish oss
  oss << std::ends;

  // print to screen (could be done differently...)
  if (ofile == nullptr) FOUR_C_THROW("no ofile available");
  fprintf(ofile, "%s\n", oss.str().c_str());

  // print it, now
  fflush(ofile);

  // nice to have met you
  return;
}  // print_newton_iter_header()


/*----------------------------------------------------------------------*
 | print Newton-Raphson iteration to screen                       |
 | originally by lw 12/07, tk 01/08                                     |
 *----------------------------------------------------------------------*/
void FPSI::Monolithic::print_newton_iter_text(FILE* ofile)
{
  // open outstringstream
  std::ostringstream oss;

  // enter  state etc
  oss << std::setw(4) << iter_;

  // different style due relative or absolute error checking
  // displacement
  switch (normtypefres_)
  {
    case Inpar::FPSI::absoluteconvergencenorm:
      oss << std::setw(15) << std::setprecision(5) << std::scientific << normofrhs_;
      break;
    case Inpar::FPSI::absoluteconvergencenorm_sys_split:
      oss << std::setw(15) << std::setprecision(5) << std::scientific << normofrhs_ / sqrtnall_;
      break;
    case Inpar::FPSI::relativconvergencenorm_sys:
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  switch (normtypeinc_)
  {
    case Inpar::FPSI::absoluteconvergencenorm:
      oss << std::setw(12) << std::setprecision(5) << std::scientific << normofiterinc_;
      break;
    case Inpar::FPSI::relativconvergencenorm_sys:
      oss << std::setw(12) << std::setprecision(5) << std::scientific
          << normofiterinc_ / norm1_alldof_ * sqrtnall_;
      break;
    case Inpar::FPSI::absoluteconvergencenorm_sys_split:
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  switch (normtypefres_)
  {
    case Inpar::FPSI::absoluteconvergencenorm:
      oss << std::setw(15) << std::setprecision(5) << std::scientific << normrhsporostruct_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific << normrhsporofluidvelocity_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific << normrhsporofluidpressure_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific << normrhsfluidvelocity_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific << normrhsfluidpressure_;
      // oss << std::setw(15) << std::setprecision(5) << std::scientific << normrhsporointerface_;
      // oss << std::setw(15) << std::setprecision(5) << std::scientific << normrhsfluidinterface_;
      break;
    case Inpar::FPSI::absoluteconvergencenorm_sys_split:
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << normrhsporostruct_ / sqrtnps_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << normrhsporofluidvelocity_ / sqrtnpfv_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << normrhsporofluidpressure_ / sqrtnpfp_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << normrhsfluidvelocity_ / sqrtnfv_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << normrhsfluidpressure_ / sqrtnfp_;
      // oss << std::setw(15) << std::setprecision(5) << std::scientific << normrhsporointerface_;
      // oss << std::setw(15) << std::setprecision(5) << std::scientific << normrhsfluidinterface_;
      break;
    case Inpar::FPSI::relativconvergencenorm_sys:
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  switch (normtypeinc_)
  {
    case Inpar::FPSI::absoluteconvergencenorm:
      oss << std::setw(15) << std::setprecision(5) << std::scientific << normofiterincporostruct_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << normofiterincporofluidvelocity_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << normofiterincporofluidpressure_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << normofiterincfluidvelocity_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << normofiterincfluidpressure_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific << normofiterincale_;
      oss << std::setw(13) << std::setprecision(5) << std::scientific
          << normofiterincfluidinterface_;
      oss << std::setw(14) << std::setprecision(5) << std::scientific
          << normofiterincporointerface_;
      break;
    case Inpar::FPSI::relativconvergencenorm_sys:
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << normofiterincporostruct_ / norm1_ps_ * sqrtnps_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << normofiterincporofluidvelocity_ / norm1_pfv_ * sqrtnpfv_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << normofiterincporofluidpressure_ / norm1_pfp_ * sqrtnpfp_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << normofiterincfluidvelocity_ / norm1_fv_ * sqrtnfv_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << normofiterincfluidpressure_ / norm1_fp_ * sqrtnfp_;
      oss << std::setw(15) << std::setprecision(5) << std::scientific
          << normofiterincale_ / norm1_a_ * sqrtna_;
      break;
    case Inpar::FPSI::absoluteconvergencenorm_sys_split:
    default:
      FOUR_C_THROW("You should not turn up here.");
      break;
  }

  // finish oss
  oss << std::ends;

  // print to screen (could be done differently...)
  if (ofile == nullptr) FOUR_C_THROW("no ofile available");
  fprintf(ofile, "%s\n", oss.str().c_str());

  // print it, now
  fflush(ofile);

  // nice to have met you
  return;

}  // PrintN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Monolithic::setup_newton()
{
  // initialise equilibrium loop and norms
  iter_ = 1;
  normofrhs_ = 0.0;
  normofiterinc_ = 0.0;
  normrhsfluid_ = 0.0;
  normofiterincfluid_ = 0.0;
  normrhsfluidvelocity_ = 0.0;
  normofiterincfluidvelocity_ = 0.0;
  normrhsporostruct_ = 0.0;
  normofiterincporostruct_ = 0.0;
  normofiterincporofluid_ = 0.0;
  normrhsfluidpressure_ = 0.0;
  normofiterincfluidpressure_ = 0.0;
  normrhsporofluidvelocity_ = 0.0;
  normofiterincporofluidvelocity_ = 0.0;
  normrhsporofluidpressure_ = 0.0;
  normofiterincporofluidpressure_ = 0.0;
  normrhsporointerface_ = 0.0;
  normofiterincporointerface_ = 0.0;
  normrhsfluidinterface_ = 0.0;
  normofiterincfluidinterface_ = 0.0;
  sqrtnall_ = 1;
  sqrtnfv_ = 1;
  sqrtnfp_ = 1;
  sqrtnpfv_ = 1;
  sqrtnpfp_ = 1;
  sqrtnps_ = 1;
  sqrtna_ = 1;
  norm1_alldof_ = 1.0;
  norm1_fv_ = 1.0;
  norm1_fp_ = 1.0;
  norm1_pfv_ = 1.0;
  norm1_pfp_ = 1.0;
  norm1_ps_ = 1.0;
  norm1_a_ = 1.0;


  // incremental solution vector with length of all dofs
  iterinc_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  iterinc_->put_scalar(0.0);

  // a zero vector of full length
  zeros_ = Core::LinAlg::create_vector(*dof_row_map(), true);
  zeros_->put_scalar(0.0);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Monolithic::fpsifd_check()
{
  // FD check is nice to check your linearisations, but be aware that we did not linearize following
  // terms:
  // gridvelocity in convective term, dispnp for the stabilization
  active_FD_check_ = true;
  // Set states
  poro_field()->fluid_field()->discretization()->clear_state();

  poro_field()->fluid_field()->discretization()->set_state(
      0, "dispnp", *poro_field()->fluid_field()->dispnp());
  poro_field()->fluid_field()->discretization()->set_state(
      0, "gridv", *poro_field()->fluid_field()->grid_vel());
  poro_field()->fluid_field()->discretization()->set_state(
      0, "dispn", *poro_field()->fluid_field()->dispn());
  poro_field()->fluid_field()->discretization()->set_state(
      0, "veln", *poro_field()->fluid_field()->veln());
  poro_field()->fluid_field()->discretization()->set_state(
      0, "velaf", *poro_field()->fluid_field()->velnp());
  poro_field()->fluid_field()->discretization()->set_state(
      0, "velnp", *poro_field()->fluid_field()->velnp());

  fluid_field()->discretization()->clear_state();

  fluid_field()->discretization()->set_state(0, "dispnp", *fluid_field()->dispnp());
  fluid_field()->discretization()->set_state(0, "gridv", *fluid_field()->grid_vel());
  fluid_field()->discretization()->set_state(0, "dispn", *fluid_field()->dispn());
  fluid_field()->discretization()->set_state(0, "veln", *fluid_field()->veln());
  fluid_field()->discretization()->set_state(0, "velaf", *fluid_field()->velnp());
  fluid_field()->discretization()->set_state(0, "velnp", *fluid_field()->velnp());

  // define and set toggle parameter delta
  const double delta = 1e-8;

  Global::Problem* problem = Global::Problem::instance();
  const Teuchos::ParameterList& fpsidynparams = problem->fpsi_dynamic_params();
  int columntocheck = fpsidynparams.get<int>("FDCheck_column");
  int rowtocheck = fpsidynparams.get<int>("FDCheck_row");
  //////////////////////////////////////////////////////////////7
  // matrices and vectors
  //////////////////////////////////////////////////////////////
  // build artificial iteration increment
  std::shared_ptr<Core::LinAlg::Vector<double>> iterinc =
      Core::LinAlg::create_vector(*dof_row_map(), true);
  const int dofs = iterinc->global_length();
  iterinc->put_scalar(0.0);
  iterinc->replace_global_value(0, delta);

  // build approximated FD stiffness matrix
  Core::LinAlg::SparseMatrix stiff_approx(*dof_row_map(), 81);

  // store old rhs
  Core::LinAlg::Vector<double> rhs_old(*dof_row_map(), true);
  rhs_old.update(1.0, *rhs_, 0.0);

  Core::LinAlg::Vector<double> rhs_copy(*dof_row_map(), true);

  std::shared_ptr<Core::LinAlg::SparseMatrix> sparse = systemmatrix_->merge();

  Core::LinAlg::SparseMatrix sparse_crs(*sparse, Core::LinAlg::DataAccess::Copy);


  std::cout << "\n****************** FPSI finite difference check ******************" << std::endl;

  int dof_poro_struct = (poro_field()->structure_field()->discretization()->num_global_nodes()) * 3;
  int dof_poro_fluid = (poro_field()->fluid_field()->discretization()->num_global_nodes()) * 4;
  int dof_fluid = (fluid_field()->discretization()->num_global_nodes()) * 4;
  int dof_ale = (ale_field()->discretization()->num_global_nodes()) * 3;

  std::cout << "poro structure field has " << dof_poro_struct << " DOFs" << std::endl;
  std::cout << "poro fluid field has     " << dof_poro_fluid << " DOFs" << std::endl;
  std::cout << "fluid field has          " << dof_fluid << " DOFs" << std::endl;
  std::cout << "ale field has            " << dof_ale << " DOFs" << std::endl;
  std::cout << "in total                 " << dofs << " DOFs" << std::endl;


  for (int i_loc = 0; i_loc < dofs; ++i_loc)  // loop over columns
  {
    int i = dof_row_map()->gid(i_loc);
    int im1 = dof_row_map()->gid(i_loc - 1);
    int ip1 = dof_row_map()->gid(i_loc + 1);
    std::cout << i << "... " << std::flush;
    if (combined_dbc_map()->my_gid(i) or fluid_field()->interface()->fsi_cond_map()->my_gid(i))
    {
      iterinc->replace_global_value(i, 0.0);
    }
    evaluate(iterinc);  // initial iterinc is varied at first dof (0-th element)
    setup_system_matrix();

    rhs_copy.update(1.0, *rhs_, 0.0);

    iterinc_->put_scalar(0.0);  // Useful? depends on solver and more
    poro_field()->clear_poro_iterinc();
    Core::LinAlg::apply_dirichlet_to_system(
        sparse_crs, *iterinc_, rhs_copy, *zeros_, *fluid_field()->interface()->fsi_cond_map());
    Core::LinAlg::apply_dirichlet_to_system(
        sparse_crs, *iterinc_, rhs_copy, *zeros_, *combined_dbc_map());

    rhs_copy.update(-1.0, rhs_old, 1.0);  // finite difference approximation of partial derivative
    rhs_copy.scale(-1.0 / delta);

    if (i == columntocheck)
    {
      std::cout << "iterinc:  " << std::endl;
      iterinc->print(std::cout);
      std::cout << "rhs_old:  " << std::endl;
      rhs_old.print(std::cout);
      std::cout << "rhs_copy: " << std::endl;
      rhs_copy.print(std::cout);
      FOUR_C_THROW("Stopped by FPSI - fd_check!");
    }

    int* index = &i;
    for (int j_loc = 0; j_loc < dofs; ++j_loc)  // loop over rows
    {
      int j = dof_row_map()->gid(j_loc);
      double value = rhs_copy.local_values_as_span()[j_loc];
      stiff_approx.insert_global_values(j, 1, &value, index);
    }  // j-loop (rows)

    if (not combined_dbc_map()->my_gid(i) and
        not fluid_field()->interface()->fsi_cond_map()->my_gid(i))
      iterinc->replace_global_value(i, -delta);

    iterinc->replace_global_value(im1, 0.0);

    if (i_loc != dofs - 1) iterinc->replace_global_value(ip1, delta);

  }  // i-loop (columns)
  evaluate(iterinc);
  setup_system_matrix();

  stiff_approx.complete();

  Core::LinAlg::SparseMatrix stiff_approx_sparse(stiff_approx);

  // calc error (subtraction of sparse_crs and stiff_approx_sparse)
  for (int i_loc = 0; i_loc < dofs; i_loc++)
  {
    int i = dof_row_map()->gid(i_loc);
    int length;
    int numentries = sparse_crs.num_global_entries(i);
    std::vector<double> values(numentries);
    std::vector<int> indices(numentries);
    sparse_crs.extract_global_row_copy(i, numentries, length, values.data(), indices.data());

    for (int k = 0; k < numentries; k++)
    {
      values[k] = -values[k];
    }

    stiff_approx_sparse.sum_into_global_values(i, numentries, values.data(), indices.data());
  }
  stiff_approx_sparse.complete();
  sparse_crs.complete();

  bool success = true;
  double error_max = 0.0;
  for (int i_loc = 0; i_loc < dofs; ++i_loc)
  {
    int i = dof_row_map()->gid(i_loc);
    if (not combined_dbc_map()->my_gid(i) and
        not fluid_field()->interface()->fsi_cond_map()->my_gid(
            i))  // only if there is no dirichlet value on dof and dof is not condensed
    {
      for (int j_loc = 0; j_loc < dofs; ++j_loc)
      {
        int j = dof_row_map()->gid(j_loc);
        if (not combined_dbc_map()->my_gid(j) and
            not fluid_field()->interface()->fsi_cond_map()->my_gid(j))
        {
          double stiff_approx_ij = 0.0;
          double sparse_ij = 0.0;
          double error_ij = 0.0;

          // get error_crs entry ij
          int errornumentries;
          int errorlength = stiff_approx_sparse.num_global_entries(i);
          std::vector<double> errorvalues(errorlength);
          std::vector<int> errorindices(errorlength);
          stiff_approx_sparse.extract_global_row_copy(
              i, errorlength, errornumentries, errorvalues.data(), errorindices.data());
          for (int k = 0; k < errorlength; ++k)
          {
            if (errorindices[k] == j)
            {
              error_ij = errorvalues[k];
              break;
            }
            else
              error_ij = 0.0;
          }

          // get sparse_ij entry ij
          int sparsenumentries;
          int sparselength = sparse_crs.num_global_entries(i);
          std::vector<double> sparsevalues(sparselength);
          std::vector<int> sparseindices(sparselength);
          sparse_crs.extract_global_row_copy(
              i, sparselength, sparsenumentries, sparsevalues.data(), sparseindices.data());
          for (int k = 0; k < sparselength; ++k)
          {
            if (sparseindices[k] == j)
            {
              sparse_ij = sparsevalues[k];
              break;
            }
            else
              sparse_ij = 0.0;
          }
          // get stiff_approx entry ijs
          int approxnumentries;
          int approxlength = stiff_approx.num_global_entries(i);
          std::vector<double> approxvalues(approxlength);
          std::vector<int> approxindices(approxlength);
          stiff_approx.extract_global_row_copy(
              i, approxlength, approxnumentries, approxvalues.data(), approxindices.data());
          for (int k = 0; k < approxlength; ++k)
          {
            if (approxindices[k] == j)
            {
              stiff_approx_ij = approxvalues[k];
              break;
            }
            else
              stiff_approx_ij = 0.0;
          }

          double error = 0.0;
          if (abs(stiff_approx_ij) > 1e-3)
            error = error_ij / stiff_approx_ij;
          else if (abs(sparse_ij) > 1e-3)
            error = error_ij / sparse_ij;

          if (i == rowtocheck and j == columntocheck)
          {
            std::cout << "K_approx value: " << stiff_approx_ij << std::endl;
            std::cout << "K value : " << sparse_ij << std::endl;
            std::cout << "error : " << error << std::endl;
            std::cout << "error_ij : " << error_ij << std::endl;
            std::cout << "i : " << i << std::endl;
            std::cout << "j : " << j << std::endl;
          }

          if (abs(error) > abs(error_max)) error_max = abs(error);

          if ((abs(error) > 1e-4))
          {
            if (abs(error_ij) > 1e-4)
            {
              std::cout << "finite difference check failed entry (" << i << "," << j
                        << ")! stiff: " << sparse_ij << ", approx: " << stiff_approx_ij
                        << " ,abs. error: " << error_ij << " , rel. error: " << error << std::endl;

              success = false;
            }
          }
        }
      }
    }
  }  // loop over dofs of success check

  if (success)
  {
    std::cout << "finite difference check successful, max. rel. error: " << error_max << std::endl;
    std::cout << "****************** finite difference check done ***************\n\n" << std::endl;
  }
  else
    std::cout << "FPSIFDCheck failed" << std::endl;
  // FOUR_C_THROW("FPSIFDCheck failed");
  poro_field()->fluid_field()->discretization()->clear_state();
  fluid_field()->discretization()->clear_state();

  active_FD_check_ = false;
  return;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Monolithic::extract_columnsfrom_sparse(Core::LinAlg::SparseMatrix& src,
    const Core::LinAlg::Map& colmap, Core::LinAlg::SparseMatrix& dst)
{
  dst.put_scalar(0.0);
  int rows = src.num_global_rows();
  for (int row = 0; row < rows; ++row)
  {
    int g_row = src.range_map().gid(row);
    int numentries;
    int length = src.num_global_entries(g_row);
    std::vector<double> values(length);
    std::vector<int> indices(length);
    src.extract_global_row_copy(g_row, length, numentries, values.data(), indices.data());
    for (int col = 0; col < length; ++col)
    {
      if (colmap.lid(indices[col]) != -1)
      {
        dst.insert_global_values(g_row, 1, &values[col], &indices[col]);
      }
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Monolithic::set_conductivity(double conduct)
{
  if (fpsi_coupl() != nullptr) fpsi_coupl()->set_conductivity(conduct);
  conductivity_ = conduct;  // remove me...
}

FOUR_C_NAMESPACE_CLOSE
