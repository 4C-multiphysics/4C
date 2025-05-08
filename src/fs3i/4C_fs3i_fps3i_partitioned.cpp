// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fs3i_fps3i_partitioned.hpp"

#include "4C_adapter_fld_poro.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_fem_condition_selector.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fpsi_monolithic_plain.hpp"
#include "4C_fpsi_utils.hpp"
#include "4C_fsi_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_poroelast_monolithic.hpp"
#include "4C_poroelast_scatra_utils.hpp"
#include "4C_poroelast_scatra_utils_clonestrategy.hpp"
#include "4C_scatra_algorithm.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_utils_clonestrategy.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  Constructor                                           hemmler 07/14 |
 *----------------------------------------------------------------------*/
FS3I::PartFPS3I::PartFPS3I(MPI_Comm comm) : FS3IBase(), comm_(comm)
{
  // keep empty
  return;
}


/*----------------------------------------------------------------------*
 |  Init                                                    rauch 09/16 |
 *----------------------------------------------------------------------*/
void FS3I::PartFPS3I::init()
{
  FS3I::FS3IBase::init();

  if (Core::Communication::my_mpi_rank(comm_) == 0)
  {
    // ##################       0.- Warning          //#########################
    std::cout << std::endl;
    std::cout << "##############################################################################"
              << std::endl;
    std::cout << "################################# WARNING!!! #################################"
              << std::endl;
    std::cout << "##############################################################################"
              << std::endl;
    std::cout << std::endl;
    std::cout << "This version of Fluid-porous-structure-scatra-scatra interaction (FPS3I) does NOT"
              << std::endl;
    std::cout << "account for the convective scalar transport at the fluid-poro interface!"
              << std::endl;
    std::cout << "The conservation of mass at the interface is only guaranteed for purely "
                 "diffusive transport"
              << std::endl;
    std::cout << std::endl;
    std::cout << "##############################################################################"
              << std::endl;
    std::cout << "################################# WARNING!!! #################################"
              << std::endl;
    std::cout << "##############################################################################"
              << std::endl;
    std::cout << std::endl;
  }
  // ##################       1.- Parameter reading          //#########################
  Global::Problem* problem = Global::Problem::instance();
  const Teuchos::ParameterList& fs3idyn = problem->f_s3_i_dynamic_params();
  const Teuchos::ParameterList& fpsidynparams = problem->fpsi_dynamic_params();
  const Teuchos::ParameterList& poroelastdynparams = problem->poroelast_dynamic_params();
  const Teuchos::ParameterList& scatradyn = problem->scalar_transport_dynamic_params();

  double dt_fpsi = fpsidynparams.get<double>("TIMESTEP");
  double dt_poroelast = poroelastdynparams.get<double>("TIMESTEP");
  if (dt_fpsi != dt_poroelast)
  {
    FOUR_C_THROW(
        "Please set \"TIMESTEP\" in \"POROELASTICITY DYNAMIC\" to the same value as in \"FPSI "
        "DYNAMIC\"!");
  }

  FPSI::InterfaceUtils* FPSI_UTILS = FPSI::InterfaceUtils::instance();

  // ##################    2.- Creation of Poroelastic + Fluid problem. (discretization called
  //  inside)     //##################
  std::shared_ptr<FPSI::FpsiBase> fpsi_algo = nullptr;

  fpsi_algo = FPSI_UTILS->setup_discretizations(comm_, fpsidynparams, poroelastdynparams);

  // only monolithic coupling of fpsi problem is supported!
  const auto coupling = Teuchos::getIntegralValue<FpsiCouplingType>(fpsidynparams, "COUPALGO");
  if (coupling == fpsi_monolithic_plain)
  {
    // Cast needed because functions such as poro_field() and fluid_field() are just a
    // member-functions of the derived class MonolithicPlain, but not of the base class FPSI_Base
    fpsi_ = std::dynamic_pointer_cast<FPSI::MonolithicPlain>(fpsi_algo);
  }
  else
  {
    FOUR_C_THROW(
        "Partitioned solution scheme not implemented for FPSI, yet. "
        "Make sure that the parameter COUPALGO is set to 'fpsi_monolithic_plain', "
        "and the parameter PARITIONED is set to 'monolithic'. ");
  }

  // ##################      3. discretization of Scatra problem       //##################
  problem->get_dis("scatra1")->fill_complete();
  problem->get_dis("scatra2")->fill_complete();

  //---------------------------------------------------------------------
  // access discretizations for poro (structure) and fluid as well as fluid-
  // and poro-based scalar transport and get material map for fluid
  // and scalar transport elements
  //---------------------------------------------------------------------
  std::shared_ptr<Core::FE::Discretization> fluiddis = problem->get_dis("fluid");
  std::shared_ptr<Core::FE::Discretization> structdis = problem->get_dis("structure");
  std::shared_ptr<Core::FE::Discretization> fluidscatradis = problem->get_dis("scatra1");
  std::shared_ptr<Core::FE::Discretization> structscatradis = problem->get_dis("scatra2");

  // determine type of scalar transport
  const auto impltype_fluid = Teuchos::getIntegralValue<Inpar::ScaTra::ImplType>(
      Global::Problem::instance()->f_s3_i_dynamic_params(), "FLUIDSCAL_SCATRATYPE");

  //---------------------------------------------------------------------
  // create discretization for fluid-based scalar transport from and
  // according to fluid discretization
  //---------------------------------------------------------------------
  if (fluiddis->num_global_nodes() == 0) FOUR_C_THROW("Fluid discretization is empty!");


  // std::map<std::pair<std::string,std::string>,std::map<int,int> > clonefieldmatmap =
  // problem->CloningMaterialMap(); if (clonefieldmatmap.size() < 2)
  //  FOUR_C_THROW("At least two material lists required for partitioned FS3I!");

  // create fluid-based scalar transport elements if fluid-based scalar
  // transport discretization is empty
  if (fluidscatradis->num_global_nodes() == 0)
  {
    // fill fluid-based scatra discretization by cloning fluid discretization
    Core::FE::clone_discretization<ScaTra::ScatraFluidCloneStrategy>(
        *fluiddis, *fluidscatradis, Global::Problem::instance()->cloning_material_map());
    fluidscatradis->fill_complete();

    // set implementation type of cloned scatra elements to advanced reactions
    for (int i = 0; i < fluidscatradis->num_my_col_elements(); ++i)
    {
      Discret::Elements::Transport* element =
          dynamic_cast<Discret::Elements::Transport*>(fluidscatradis->l_col_element(i));
      if (element == nullptr)
        FOUR_C_THROW("Invalid element type!");
      else
        element->set_impl_type(impltype_fluid);
    }
  }
  else
    FOUR_C_THROW("Fluid AND ScaTra discretization present. This is not supported.");

  //---------------------------------------------------------------------
  // create discretization for poro-based scalar transport from and
  // according to poro (structure) discretization
  //--------------------------------------------------------------------
  if (fluiddis->num_global_nodes() == 0) FOUR_C_THROW("Fluid discretization is empty!");

  if (!structscatradis->filled()) structscatradis->fill_complete();
  if (structscatradis->num_global_nodes() == 0)
  {
    // fill poro-based scatra discretization by cloning structure discretization
    Core::FE::clone_discretization<PoroElastScaTra::Utils::PoroScatraCloneStrategy>(
        *structdis, *structscatradis, Global::Problem::instance()->cloning_material_map());
  }
  else
    FOUR_C_THROW("Structure AND ScaTra discretization present. This is not supported.");

  // ##################      End of discretization       //##################

  //---------------------------------------------------------------------
  // create instances for fluid- and poro (structure)-based scalar transport
  // solver and arrange them in combined vector
  //---------------------------------------------------------------------
  // get the solver number used for structural ScalarTransport solver
  const int linsolver1number = fs3idyn.get<int>("LINEAR_SOLVER1");
  // get the solver number used for structural ScalarTransport solver
  const int linsolver2number = fs3idyn.get<int>("LINEAR_SOLVER2");

  // check if the linear solver has a valid solver number
  if (linsolver1number == (-1))
    FOUR_C_THROW(
        "no linear solver defined for fluid ScalarTransport solver. Please set LINEAR_SOLVER2 in "
        "FS3I DYNAMIC to a valid number!");
  if (linsolver2number == (-1))
    FOUR_C_THROW(
        "no linear solver defined for structural ScalarTransport solver. Please set LINEAR_SOLVER2 "
        "in FS3I DYNAMIC to a valid number!");
  fluidscatra_ = std::make_shared<Adapter::ScaTraBaseAlgorithm>(
      fs3idyn, scatradyn, problem->solver_params(linsolver1number), "scatra1", true);

  // now we can call init() on the scatra time integrator
  fluidscatra_->init();
  fluidscatra_->scatra_field()->set_number_of_dof_set_displacement(1);
  fluidscatra_->scatra_field()->set_number_of_dof_set_velocity(1);
  fluidscatra_->scatra_field()->set_number_of_dof_set_wall_shear_stress(1);
  fluidscatra_->scatra_field()->set_number_of_dof_set_pressure(1);

  structscatra_ = std::make_shared<Adapter::ScaTraBaseAlgorithm>(
      fs3idyn, scatradyn, problem->solver_params(linsolver2number), "scatra2", true);

  // only now we must call init() on the scatra time integrator.
  // all objects relying on the parallel distribution are
  // created and pointers are set.
  structscatra_->init();
  structscatra_->scatra_field()->set_number_of_dof_set_displacement(1);
  structscatra_->scatra_field()->set_number_of_dof_set_velocity(1);
  structscatra_->scatra_field()->set_number_of_dof_set_wall_shear_stress(2);
  structscatra_->scatra_field()->set_number_of_dof_set_pressure(2);

  scatravec_.push_back(fluidscatra_);
  scatravec_.push_back(structscatra_);

  //---------------------------------------------------------------------
  // check various input parameters
  //---------------------------------------------------------------------
  const Teuchos::ParameterList& structdyn = problem->structural_dynamic_params();
  const Teuchos::ParameterList& fluiddyn = problem->fluid_dynamic_params();
  // check consistency of time-integration schemes in input file
  // (including parameter theta itself in case of one-step-theta scheme)
  // and rule out unsupported versions of generalized-alpha time-integration
  // scheme (as well as other inappropriate schemes) for fluid subproblem
  auto scatratimealgo =
      Teuchos::getIntegralValue<Inpar::ScaTra::TimeIntegrationScheme>(scatradyn, "TIMEINTEGR");
  auto fluidtimealgo =
      Teuchos::getIntegralValue<Inpar::FLUID::TimeIntegrationScheme>(fluiddyn, "TIMEINTEGR");
  auto structtimealgo =
      Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(structdyn, "DYNAMICTYPE");

  if (fluidtimealgo == Inpar::FLUID::timeint_one_step_theta)
  {
    if (scatratimealgo != Inpar::ScaTra::timeint_one_step_theta or
        structtimealgo != Inpar::Solid::DynamicType::OneStepTheta)
      FOUR_C_THROW(
          "Partitioned FS3I computations should feature consistent time-integration schemes for "
          "the subproblems; in this case, a one-step-theta scheme is intended to be used for the "
          "fluid subproblem, and different schemes are intended to be used for the structure "
          "and/or scalar transport subproblems!");

    if (scatradyn.get<double>("THETA") != fluiddyn.get<double>("THETA") or
        scatradyn.get<double>("THETA") != structdyn.sublist("ONESTEPTHETA").get<double>("THETA"))
      FOUR_C_THROW(
          "Parameter(s) theta for one-step-theta time-integration scheme defined in one or more of "
          "the individual fields do(es) not match for partitioned FS3I computation.");
  }
  else if (fluidtimealgo == Inpar::FLUID::timeint_afgenalpha)
  {
    if (scatratimealgo != Inpar::ScaTra::timeint_gen_alpha or
        structtimealgo != Inpar::Solid::DynamicType::GenAlpha)
      FOUR_C_THROW(
          "Partitioned FS3I computations should feature consistent time-integration schemes for "
          "the subproblems; in this case, a (alpha_f-based) generalized-alpha scheme is intended "
          "to be used for the fluid subproblem, and different schemes are intended to be used for "
          "the structure and/or scalar transport subproblems!");
  }
  else if (fluidtimealgo == Inpar::FLUID::timeint_npgenalpha)
  {
    FOUR_C_THROW(
        "Partitioned FS3I computations do not support n+1-based generalized-alpha time-integration "
        "schemes for the fluid subproblem!");
  }
  else if (fluidtimealgo == Inpar::FLUID::timeint_bdf2 or
           fluidtimealgo == Inpar::FLUID::timeint_stationary)
  {
    FOUR_C_THROW(
        "Partitioned FS3I computations do not support stationary of BDF2 time-integration schemes "
        "for the fluid subproblem!");
  }

  // check that incremental formulation is used for scalar transport field,
  // according to structure and fluid field
  if (scatravec_[0]->scatra_field()->is_incremental() == false)
    FOUR_C_THROW("Incremental formulation required for partitioned FS3I computations!");


  return;
}


/*----------------------------------------------------------------------*
 |  Setup                                                   rauch 09/16 |
 *----------------------------------------------------------------------*/
void FS3I::PartFPS3I::setup()
{
  FS3I::FS3IBase::setup();

  // only now we must call setup() on the scatra base algo.
  // all objects relying on the parallel distribution are
  // created and pointers are set.
  // calls setup() on time integrator inside.
  fluidscatra_->setup();
  structscatra_->setup();

  //---------------------------------------------------------------------
  // check existence of scatra coupling conditions for both
  // discretizations and definition of the permeability coefficient
  //---------------------------------------------------------------------
  check_f_s3_i_inputs();

  // in case of FPS3I we have to handle the conductivity, too
  std::shared_ptr<Core::FE::Discretization> dis = scatravec_[0]->scatra_field()->discretization();
  std::vector<Core::Conditions::Condition*> coupcond;
  dis->get_condition("ScaTraCoupling", coupcond);
  double myconduct = coupcond[0]->parameters().get<double>(
      "CONDUCT");  // here we assume the conductivity to be the same in every BC

  // conductivity is not only needed in scatracoupling but also in FPSI coupling!
  if (myconduct == 0.0)
  {
    FOUR_C_THROW(
        "conductivity of 0.0 is not allowed!!! Should be set in \"DESIGN SCATRA COUPLING SURF "
        "CONDITIONS\"");
  }
  fpsi_->set_conductivity(myconduct);
}

/*----------------------------------------------------------------------*
 |  Restart                                               hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFPS3I::read_restart()
{
  // read restart information, set vectors and variables
  // (Note that dofmaps might have changed in a redistribution call!)
  Global::Problem* problem = Global::Problem::instance();
  const int restart = problem->restart();

  if (restart)
  {
    // restart of FPSI problem
    fpsi_->read_restart(restart);

    // restart of scatra problem
    for (unsigned i = 0; i < scatravec_.size(); ++i)
    {
      std::shared_ptr<Adapter::ScaTraBaseAlgorithm> currscatra = scatravec_[i];
      currscatra->scatra_field()->read_restart(restart);
    }

    time_ = fpsi_->fluid_field()->time();
    step_ = fpsi_->fluid_field()->step();
  }
}

/*----------------------------------------------------------------------*
 | redistribute the FPSI interface                           thon 11/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFPS3I::redistribute_interface()
{
  fpsi_->redistribute_interface();

  Global::Problem* problem = Global::Problem::instance();

  if (Core::Communication::num_mpi_ranks(comm_) >
      1)  // if we have more than one processor, we need to redistribute at the FPSI interface
  {
    FPSI::InterfaceUtils* FPSI_UTILS = FPSI::InterfaceUtils::instance();

    std::shared_ptr<std::map<int, int>> Fluid_PoroFluid_InterfaceMap =
        FPSI_UTILS->get_fluid_poro_fluid_interface_map();
    std::shared_ptr<std::map<int, int>> PoroFluid_Fluid_InterfaceMap =
        FPSI_UTILS->get_poro_fluid_fluid_interface_map();

    FPSI_UTILS->redistribute_interface(
        *problem->get_dis("scatra1"), "", *PoroFluid_Fluid_InterfaceMap);
    FPSI_UTILS->redistribute_interface(
        *problem->get_dis("scatra2"), "", *Fluid_PoroFluid_InterfaceMap);
  }

  std::shared_ptr<Core::FE::Discretization> structdis = problem->get_dis("structure");
  std::shared_ptr<Core::FE::Discretization> structscatradis = problem->get_dis("scatra2");

  // after redistributing the interface we have to fix the material pointers of the structure-scatra
  // discretisation
  PoroElast::Utils::set_material_pointers_matching_grid(*structdis, *structscatradis);
}

/*----------------------------------------------------------------------*
 |  System Setup                                          hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFPS3I::setup_system()
{
  // do the coupling setup and create the combined dofmap

  // Setup FPSI system
  fpsi_->setup_system();
  fpsi_->setup_solver();

  /*----------------------------------------------------------------------*/
  /*                  General set up for scalar fields                    */
  /*----------------------------------------------------------------------*/

  // create map extractors needed for scatra condition coupling
  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> currscatra = scatravec_[i];
    const int numscal = currscatra->scatra_field()->num_scal();
    std::shared_ptr<Core::FE::Discretization> currdis =
        currscatra->scatra_field()->discretization();
    std::shared_ptr<Core::LinAlg::MultiMapExtractor> mapex =
        std::make_shared<Core::LinAlg::MultiMapExtractor>();
    Core::Conditions::setup_extractor(
        *currdis, *mapex, {Core::Conditions::Selector("ScaTraCoupling", 0, numscal)});
    scatrafieldexvec_.push_back(mapex);
  }

  scatracoup_->setup_condition_coupling(*(scatravec_[0]->scatra_field()->discretization()),
      scatrafieldexvec_[0]->map(1), *(scatravec_[1]->scatra_field()->discretization()),
      scatrafieldexvec_[1]->map(1), "ScaTraCoupling",
      scatravec_[0]
          ->scatra_field()
          ->num_scal());  // we assume here that both discretisation have the same number of scalars

  // create map extractor for coupled scatra fields
  // the second field is always split
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps;

  // In the limiting case of an infinite permeability of the interface between
  // different scatra fields, the concentrations on both sides of the interface are
  // constrained to be equal. In this case, we keep the fluid scatra dofs at the
  // interface as unknowns in the overall system, whereas the poro (structure) scatra
  // dofs are condensed (cf. "structuresplit" in a monolithic FPSI
  // system). Otherwise, both concentrations are kept in the overall system
  // and the equality of fluxes is considered explicitly.
  if (infperm_)
  {
    maps.push_back(scatrafieldexvec_[0]->full_map());
    maps.push_back(scatrafieldexvec_[1]->map(0));
  }
  else
  {
    maps.push_back(scatrafieldexvec_[0]->full_map());
    maps.push_back(scatrafieldexvec_[1]->full_map());
  }
  std::shared_ptr<Core::LinAlg::Map> fullmap = Core::LinAlg::MultiMapExtractor::merge_maps(maps);
  scatraglobalex_->setup(*fullmap, maps);

  // create coupling vectors and matrices (only needed for finite surface permeabilities)
  if (not infperm_)
  {
    for (unsigned i = 0; i < scatravec_.size(); ++i)
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> scatracoupforce =
          std::make_shared<Core::LinAlg::Vector<double>>(*(scatraglobalex_->map(i)), true);
      scatracoupforce_.push_back(scatracoupforce);

      std::shared_ptr<Core::LinAlg::SparseMatrix> scatracoupmat =
          std::make_shared<Core::LinAlg::SparseMatrix>(*(scatraglobalex_->map(i)), 27, false, true);
      scatracoupmat_.push_back(scatracoupmat);

      const Core::LinAlg::Map* dofrowmap =
          scatravec_[i]->scatra_field()->discretization()->dof_row_map();
      std::shared_ptr<Core::LinAlg::Vector<double>> zeros =
          Core::LinAlg::create_vector(*dofrowmap, true);
      scatrazeros_.push_back(zeros);
    }
  }
  // create scatra block matrix
  scatrasystemmatrix_ =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *scatraglobalex_, *scatraglobalex_, 27, false, true);
  // create scatra rhs vector
  scatrarhs_ = std::make_shared<Core::LinAlg::Vector<double>>(*scatraglobalex_->full_map(), true);
  // create scatra increment vector
  scatraincrement_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*scatraglobalex_->full_map(), true);
  // check whether potential Dirichlet conditions at scatra interface are
  // defined for both discretizations
  check_interface_dirichlet_bc();
  // scatra solver
  std::shared_ptr<Core::FE::Discretization> firstscatradis =
      (scatravec_[0])->scatra_field()->discretization();

  const Teuchos::ParameterList& fs3idyn = Global::Problem::instance()->f_s3_i_dynamic_params();
  // get solver number used for fs3i
  const int linsolvernumber = fs3idyn.get<int>("COUPLED_LINEAR_SOLVER");
  // check if LOMA solvers has a valid number
  if (linsolvernumber == (-1))
    FOUR_C_THROW(
        "no linear solver defined for FS3I problems. Please set COUPLED_LINEAR_SOLVER in FS3I "
        "DYNAMIC to a valid number!");
  const Teuchos::ParameterList& coupledscatrasolvparams =
      Global::Problem::instance()->solver_params(linsolvernumber);

  const auto solvertype =
      Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(coupledscatrasolvparams, "SOLVER");

  if (solvertype != Core::LinearSolver::SolverType::belos)
    FOUR_C_THROW("Iterative solver expected");

  const auto azprectype = Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(
      coupledscatrasolvparams, "AZPREC");

  if (azprectype != Core::LinearSolver::PreconditionerType::block_teko)
    FOUR_C_THROW("Block Gauss-Seidel preconditioner expected");

  // use coupled scatra solver object
  scatrasolver_ = std::make_shared<Core::LinAlg::Solver>(coupledscatrasolvparams,
      firstscatradis->get_comm(), Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));
  // get the solver number used for structural ScalarTransport solver
  const int linsolver1number = fs3idyn.get<int>("LINEAR_SOLVER1");
  // get the solver number used for structural ScalarTransport solver
  const int linsolver2number = fs3idyn.get<int>("LINEAR_SOLVER2");

  // check if the linear solver has a valid solver number
  if (linsolver1number == (-1))
    FOUR_C_THROW(
        "no linear solver defined for fluid ScalarTransport solver. Please set LINEAR_SOLVER2 in "
        "FS3I DYNAMIC to a valid number!");
  if (linsolver2number == (-1))
    FOUR_C_THROW(
        "no linear solver defined for structural ScalarTransport solver. Please set LINEAR_SOLVER2 "
        "in FS3I DYNAMIC to a valid number!");
  scatrasolver_->put_solver_params_to_sub_params("Inverse1",
      Global::Problem::instance()->solver_params(linsolver1number),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));
  scatrasolver_->put_solver_params_to_sub_params("Inverse2",
      Global::Problem::instance()->solver_params(linsolver2number),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));

  if (azprectype == Core::LinearSolver::PreconditionerType::block_teko)
  {
    Core::LinearSolver::Parameters::compute_solver_parameters(
        *(scatravec_[0])->scatra_field()->discretization(),
        scatrasolver_->params().sublist("Inverse1"));
    Core::LinearSolver::Parameters::compute_solver_parameters(
        *(scatravec_[1])->scatra_field()->discretization(),
        scatrasolver_->params().sublist("Inverse2"));
  }
}


/*----------------------------------------------------------------------*
 |  Test results                                          hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFPS3I::test_results(MPI_Comm comm)
{
  Global::Problem::instance()->add_field_test(fpsi_->fluid_field()->create_field_test());

  fpsi_->poro_field()->structure_field()->create_field_test();
  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra = scatravec_[i];
    Global::Problem::instance()->add_field_test(scatra->create_scatra_field_test());
  }
  Global::Problem::instance()->test_all(comm);
}


/*----------------------------------------------------------------------*
 |  Transfer FPSI solution                                hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFPS3I::set_fpsi_solution()
{
  // we clear every state, including the states of the secondary dof sets
  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    scatravec_[i]->scatra_field()->discretization()->clear_state(true);
    // we have to manually clear this since this can not be saved directly in the
    // primary dof set (because it is cleared in between)
    scatravec_[i]->scatra_field()->clear_external_concentrations();
  }

  set_mesh_disp();
  set_velocity_fields();
  set_wall_shear_stresses();
  set_pressure_fields();
  set_membrane_concentration();
}

/*----------------------------------------------------------------------*
 |  Transfer scatra solution                              hemmler 07/14 |
 *----------------------------------------------------------------------*/
// only needed for two-way coupling; at the moment function is not used
void FS3I::PartFPS3I::set_struct_scatra_solution()
{
  fpsi_->poro_field()->structure_field()->discretization()->set_state(
      1, "scalarfield", *(scatravec_[1])->scatra_field()->phinp());
}


/*----------------------------------------------------------------------*
 |  Set displacements                                     hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFPS3I::set_mesh_disp()
{
  // fluid field
  scatravec_[0]->scatra_field()->apply_mesh_movement(*fpsi_->fluid_field()->dispnp());

  // Poro field
  scatravec_[1]->scatra_field()->apply_mesh_movement(
      *fpsi_->poro_field()->structure_field()->dispnp());
}


/*----------------------------------------------------------------------*
 |  Set velocities                                        hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFPS3I::set_velocity_fields()
{
  Global::Problem* problem = Global::Problem::instance();
  const Teuchos::ParameterList& scatradyn = problem->scalar_transport_dynamic_params();
  const auto cdvel =
      Teuchos::getIntegralValue<Inpar::ScaTra::VelocityField>(scatradyn, "VELOCITYFIELD");
  switch (cdvel)
  {
    case Inpar::ScaTra::velocity_zero:
    case Inpar::ScaTra::velocity_function:
    {
      for (auto scatra : scatravec_)
      {
        scatra->scatra_field()->set_velocity_field_from_function();
      }
      break;
    }
    case Inpar::ScaTra::velocity_Navier_Stokes:
    {
      std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> convel;
      std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> vel;
      extract_vel(convel, vel);

      for (unsigned i = 0; i < scatravec_.size(); ++i)
      {
        scatravec_[i]->scatra_field()->set_convective_velocity(*convel[i]);
        scatravec_[i]->scatra_field()->set_velocity_field(*vel[i]);
      }
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 |  Set wall shear stresses                               hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFPS3I::set_wall_shear_stresses()
{
  std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> wss;
  extract_wss(wss);

  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra = scatravec_[i];
    scatra->scatra_field()->set_wall_shear_stresses(*wss[i]);
  }
}

/*----------------------------------------------------------------------*
 |  Set presures                                          hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFPS3I::set_pressure_fields()
{
  std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> pressure;
  extract_pressure(pressure);

  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra = scatravec_[i];
    scatra->scatra_field()->set_pressure_field(*pressure[i]);
  }
}

/*----------------------------------------------------------------------*
 |  Evaluate scatra fields                                hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFPS3I::evaluate_scatra_fields()
{
  // membrane concentration at the interface needed for membrane equation of Kedem and Katchalsky.
  // NOTE: needs to be set here, since it depends on the scalar interface values on both
  // discretisations changing with each Newton iteration
  set_membrane_concentration();

  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    std::shared_ptr<Adapter::ScaTraBaseAlgorithm> scatra_adap = scatravec_[i];
    std::shared_ptr<ScaTra::ScaTraTimIntImpl> scatra = scatra_adap->scatra_field();

    scatra->prepare_linear_solve();

    // add contributions due to finite interface permeability
    if (!infperm_)
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> coupforce = scatracoupforce_[i];
      std::shared_ptr<Core::LinAlg::SparseMatrix> coupmat = scatracoupmat_[i];

      coupforce->put_scalar(0.0);
      coupmat->zero();

      // evaluate interface; second Kedem-Katchalsky equation for coupling of solute flux
      scatra->kedem_katchalsky(coupmat, coupforce);

      // apply Dirichlet boundary conditions to coupling matrix and vector
      const std::shared_ptr<const Core::LinAlg::Map> dbcmap = scatra->dirich_maps()->cond_map();
      coupmat->apply_dirichlet(*dbcmap, false);
      Core::LinAlg::apply_dirichlet_to_system(*coupforce, *scatrazeros_[i], *dbcmap);
    }
  }
}


/*----------------------------------------------------------------------*
 |  Extract velocities                                    hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFPS3I::extract_vel(
    std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>& convel,
    std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>& vel)
{
  // ############ Fluid Field ###############
  convel.push_back(fpsi_->fluid_field()->convective_vel());
  vel.push_back(fpsi_->fluid_field()->velnp());

  // ############ Poro Field ###############
  convel.push_back(fpsi_->poro_field()->fluid_field()->convective_vel());
  vel.push_back(fpsi_->poro_field()->fluid_field()->velnp());
}


/*----------------------------------------------------------------------*
 |  Extract wall shear stresses                           hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFPS3I::extract_wss(
    std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>& wss)
{
  // ############ Fluid Field ###############

  std::shared_ptr<Adapter::FluidFSI> fluid =
      std::dynamic_pointer_cast<Adapter::FluidFSI>(fpsi_->fluid_field());
  if (fluid == nullptr) FOUR_C_THROW("Dynamic cast to Adapter::FluidFSI failed!");

  std::shared_ptr<Core::LinAlg::Vector<double>> WallShearStress =
      fluid->calculate_wall_shear_stresses();  // CalcWallShearStress();
  wss.push_back(WallShearStress);

  // ############ Poro Field ###############

  // Hint: The Wall shear stresses in the fluid field at the Interface are equal to the ones of the
  // poro structure
  //      Therefore, we map the results of the wss of the fluid field to the dofs of the poro field
  //      without computing them explicitly in the poro field

  // extract FPSI-Interface from fluid field
  WallShearStress =
      fpsi_->fpsi_coupl()->fluid_fpsi_vel_pres_extractor()->extract_cond_vector(*WallShearStress);

  // replace global fluid interface dofs through porofluid interface dofs
  WallShearStress = fpsi_->fpsi_coupl()->i_fluid_to_porofluid(*WallShearStress);

  // insert porofluid interface entries into vector with full porofluid length
  std::shared_ptr<Core::LinAlg::Vector<double>> porofluid =
      Core::LinAlg::create_vector(*(fpsi_->poro_field()->fluid_field()->dof_row_map()), true);

  // Parameter int block of function InsertVector:
  fpsi_->fpsi_coupl()->poro_fluid_fpsi_vel_pres_extractor()->insert_vector(
      *WallShearStress, 1, *porofluid);

  wss.push_back(porofluid);
}

/*----------------------------------------------------------------------*
 |  Extract pressures                                     hemmler 07/14 |
 *----------------------------------------------------------------------*/
void FS3I::PartFPS3I::extract_pressure(
    std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>& pressure)
{
  // ############ Fluid Field ###############
  pressure.push_back(
      fpsi_->fluid_field()->velnp());  // we extract the velocities as well. We sort them out later.

  // ############ Poro Field ###############
  pressure.push_back(fpsi_->poro_field()
          ->fluid_field()
          ->velnp());  // we extract the velocities as well. We sort them out later.
}

FOUR_C_NAMESPACE_CLOSE
