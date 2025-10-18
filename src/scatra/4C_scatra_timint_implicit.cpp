// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_timint_implicit.hpp"

#include "4C_comm_utils_gid_vector.hpp"
#include "4C_contact_nitsche_strategy_ssi.hpp"
#include "4C_fem_condition_periodic.hpp"
#include "4C_fem_condition_selector.hpp"
#include "4C_fem_discretization_nullspace.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fem_nurbs_discretization_initial_condition.hpp"
#include "4C_fluid_turbulence_dyn_vreman.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_io_visualization_parameters.hpp"
#include "4C_linalg_krylov_projector.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_mat_elchmat.hpp"
#include "4C_mat_electrode.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_scatra.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_boundary_calc_elch_electrode_utils.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_ele_parameter_turbulence.hpp"
#include "4C_scatra_resulttest.hpp"
#include "4C_scatra_timint_heterogeneous_reaction_strategy.hpp"
#include "4C_scatra_timint_meshtying_strategy_artery.hpp"
#include "4C_scatra_timint_meshtying_strategy_fluid.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"
#include "4C_scatra_timint_meshtying_strategy_std.hpp"
#include "4C_scatra_turbulence_hit_initial_scalar_field.hpp"
#include "4C_scatra_turbulence_hit_scalar_forcing.hpp"
#include "4C_scatra_utils.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <unordered_set>
#include <utility>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
ScaTra::ScaTraTimIntImpl::ScaTraTimIntImpl(std::shared_ptr<Core::FE::Discretization> actdis,
    std::shared_ptr<Core::LinAlg::Solver> solver, std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Teuchos::ParameterList> extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output, const int probnum)
    : problem_(Global::Problem::instance(probnum)),
      probnum_(probnum),
      solver_(std::move(solver)),
      params_(params),
      extraparams_(extraparams),
      myrank_(Core::Communication::my_mpi_rank(actdis->get_comm())),
      splitter_(nullptr),
      strategy_(nullptr),
      additional_model_evaluator_(nullptr),
      isale_(extraparams->get<bool>("isale")),
      solvtype_(Teuchos::getIntegralValue<Inpar::ScaTra::SolverType>(*params, "SOLVERTYPE")),
      equilibrationmethod_(
          Teuchos::getIntegralValue<Core::LinAlg::EquilibrationMethod>(*params, "EQUILIBRATION")),
      matrixtype_(Teuchos::getIntegralValue<Core::LinAlg::MatrixType>(*params, "MATRIXTYPE")),
      incremental_(true),
      fssgd_(Teuchos::getIntegralValue<Inpar::ScaTra::FSSUGRDIFF>(*params, "FSSUGRDIFF")),
      turbmodel_(Inpar::FLUID::no_model),
      s2ikinetics_(actdis->has_condition("S2IKinetics")),
      s2imeshtying_(actdis->has_condition("S2IMeshtying")),
      arterycoupling_(
          problem_->poro_multi_phase_scatra_dynamic_params().get<bool>("artery_coupling_active") &&
          actdis->name() == "scatra"),
      heteroreaccoupling_(actdis->has_condition("ScatraHeteroReactionSlave")),
      macro_scale_(
          problem_->materials()->first_id_by_type(Core::Materials::m_scatra_multiscale) != -1 or
          problem_->materials()->first_id_by_type(Core::Materials::m_newman_multiscale) != -1),
      micro_scale_(probnum != 0),
      has_external_force_(params_->sublist("EXTERNAL FORCE").get<bool>("EXTERNAL_FORCE")),
      calcflux_domain_(
          Teuchos::getIntegralValue<Inpar::ScaTra::FluxType>(*params, "CALCFLUX_DOMAIN")),
      calcflux_domain_lumped_(params->get<bool>("CALCFLUX_DOMAIN_LUMPED")),
      calcflux_boundary_(
          Teuchos::getIntegralValue<Inpar::ScaTra::FluxType>(*params, "CALCFLUX_BOUNDARY")),
      calcflux_boundary_lumped_(params->get<bool>("CALCFLUX_BOUNDARY_LUMPED")),
      writefluxids_(std::make_shared<std::vector<int>>()),
      flux_domain_(nullptr),
      flux_boundary_(nullptr),
      flux_boundary_maps_(nullptr),
      sumnormfluxintegral_(nullptr),
      lastfluxoutputstep_(-1),
      output_element_material_id_(
          Global::Problem::instance()->io_params().get<bool>("ELEMENT_MAT_ID")),
      outputscalars_(
          Teuchos::getIntegralValue<Inpar::ScaTra::OutputScalarType>(*params, "OUTPUTSCALARS")),
      outputgmsh_(params->get<bool>("OUTPUT_GMSH")),
      output_state_matlab_(params->get<bool>("MATLAB_STATE_OUTPUT")),
      fdcheck_(Teuchos::getIntegralValue<Inpar::ScaTra::FdCheck>(*params, "FDCHECK")),
      fdcheckeps_(params->get<double>("FDCHECKEPS")),
      fdchecktol_(params->get<double>("FDCHECKTOL")),
      computeintegrals_(
          Teuchos::getIntegralValue<Inpar::ScaTra::ComputeIntegrals>(*params, "COMPUTEINTEGRALS")),
      calcerror_(Teuchos::getIntegralValue<Inpar::ScaTra::CalcError>(*params, "CALCERROR")),
      time_(0.0),
      maxtime_(params->get<double>("MAXTIME")),
      step_(0),
      stepmax_(params->get<int>("NUMSTEP")),
      dta_(params->get<double>("TIMESTEP")),
      dtele_(0.0),
      dtsolve_(0.0),
      iternum_(0),
      iternum_outer_(0),
      timealgo_(
          Teuchos::getIntegralValue<Inpar::ScaTra::TimeIntegrationScheme>(*params, "TIMEINTEGR")),
      nsd_(problem_->n_dim()),
      scalarhandler_(nullptr),
      outputscalarstrategy_(nullptr),
      outputdomainintegralstrategy_(nullptr),
      // Initialization of degrees of freedom variables
      phin_(nullptr),
      phinp_(nullptr),
      phinp_inc_(nullptr),
      phinp_inc_old_(nullptr),
      omega_(0, 0.),
      phidtn_(nullptr),
      phidtnp_(nullptr),
      hist_(nullptr),
      densafnp_(nullptr),
      velocity_field_type_(
          Teuchos::getIntegralValue<Inpar::ScaTra::VelocityField>(*params, "VELOCITYFIELD")),
      mean_conc_(nullptr),
      membrane_conc_(nullptr),
      phinp_micro_(nullptr),
      nds_disp_(-1),
      nds_growth_(-1),
      nds_micro_(-1),
      nds_pres_(-1),
      nds_scatra_(-1),
      nds_thermo_(-1),
      nds_two_tensor_quantity_(-1),
      nds_vel_(-1),
      nds_wss_(-1),
      densific_(0, 0.0),
      c0_(0, 0.0),
      macro_micro_rea_coeff_(0.0),
      discret_(actdis),
      output_(std::move(output)),
      convform_(Teuchos::getIntegralValue<Inpar::ScaTra::ConvForm>(*params, "CONVFORM")),
      sysmat_(nullptr),
      zeros_(nullptr),
      dbcmaps_(nullptr),
      neumann_loads_(nullptr),
      normals_(nullptr),
      residual_(nullptr),
      trueresidual_(nullptr),
      increment_(nullptr),
      msht_(Teuchos::getIntegralValue<Inpar::FLUID::MeshTying>(*params, "MESHTYING")),
      // Initialization of AVM3 variables
      sysmat_sd_(nullptr),
      Sep_(nullptr),
      Mnsv_(nullptr),
      // Initialization of turbulent flow variables
      DynSmag_(nullptr),
      Vrem_(nullptr),
      samstart_(0),
      samstop_(0),
      dumperiod_(0),
      turbinflow_(extraparams->sublist("TURBULENT INFLOW").get<bool>("TURBULENTINFLOW")),
      numinflowsteps_(extraparams->sublist("TURBULENT INFLOW").get<int>("NUMINFLOWSTEP")),
      special_flow_("initialization"),
      forcing_(nullptr),
      homisoturb_forcing_(nullptr),
      // Initialization of Krylov
      updateprojection_(false),
      projector_(nullptr),
      // Initialization of
      upres_(params->get<int>("RESULTSEVERY")),
      uprestart_(params->get<int>("RESTARTEVERY")),
      neumanninflow_(params->get<bool>("NEUMANNINFLOW")),
      convheatrans_(params->get<bool>("CONV_HEAT_TRANS")),
      phinp_macro_(0, 0.),
      q_(0.0),
      dq_dphi_(0, 0.),
      // Initialization of Biofilm specific stuff
      scfldgrdisp_(nullptr),
      scstrgrdisp_(nullptr),
      outintegrreac_(params->get<bool>("OUTINTEGRREAC")),
      skipinitder_(params->get<bool>("SKIPINITDER")),
      timestepadapted_(false),
      issetup_(false),
      isinit_(false)
{
  const int restart_step = problem_->restart();
  if (restart_step > 0)
  {
    Core::IO::DiscretizationReader reader(
        discret_, Global::Problem::instance()->input_control_file(), restart_step);

    time_ = reader.read_double("time");
  }

  visualization_writer_ = std::make_shared<Core::IO::DiscretizationVisualizationWriterMesh>(
      actdis, Core::IO::visualization_parameters_factory(
                  Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                  *Global::Problem::instance()->output_control_file(), time_));

  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting
  // this has to be done before all state vectors are initialized
}


/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::init()
{
  set_is_setup(false);

  // -------------------------------------------------------------------
  // safety check for spherical coordinates
  // -------------------------------------------------------------------
  if (params_->get<bool>("SPHERICALCOORDS") and nsd_ > 1)
    FOUR_C_THROW("Spherical coordinates only available for 1D problems!");

  // -------------------------------------------------------------------
  // connect degrees of freedom for periodic boundary conditions
  // -------------------------------------------------------------------
  // note: pbcs have to be correctly set up before extended ghosting is applied
  Core::Conditions::PeriodicBoundaryConditions pbc(discret_, false);
  if (pbc.has_pbc() and not isinit_)
  {
    pbc.update_dofs_for_periodic_boundary_conditions();
  }

  // -------------------------------------------------------------------
  // determine whether linear incremental or nonlinear solver
  // -------------------------------------------------------------------
  switch (solvtype_)
  {
    case Inpar::ScaTra::solvertype_nonlinear:
    case Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro:
    case Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken:
    case Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken_dofsplit:
    case Inpar::ScaTra::solvertype_nonlinear_multiscale_microtomacro:
    case Inpar::ScaTra::solvertype_linear_incremental:
    {
      incremental_ = true;
    }
    break;
    case Inpar::ScaTra::solvertype_linear_full:
    {
      incremental_ = false;
    }
    break;
    default:
      FOUR_C_THROW("Received illegal scatra solvertype enum.");
      break;
  }

  // -----------------------------------------------------------------------
  // determine number of degrees of freedom and transported scalars per node
  // -----------------------------------------------------------------------
  create_scalar_handler();

  // -------------------------------------------------------------------
  // check compatibility of boundary conditions
  // -------------------------------------------------------------------
  if (neumanninflow_ and convheatrans_)
  {
    FOUR_C_THROW(
        "Neumann inflow and convective heat transfer boundary conditions must not appear "
        "simultaneously for the same problem!");
  }

  // -----------------------------------------------------------------------------
  // initialize meshtying strategy (including standard strategy without meshtying)
  // -----------------------------------------------------------------------------
  // safety checks
  if (msht_ != Inpar::FLUID::no_meshtying and s2imeshtying_)
  {
    FOUR_C_THROW(
        "Fluid-fluid meshtying in combination with scatra-scatra interface mesh tying is not "
        "implemented yet!");
  }
  if (s2imeshtying_ and !incremental_)
  {
    FOUR_C_THROW(
        "Scatra-scatra interface mesh tying only working for incremental solve so far!\n"
        "Set the parameter SOLVERTYPE in SCALAR TRANSPORT DYNAMIC section to 'nonlinear' or "
        "'linear_incremental'!");
  }

  ScaTraUtils::check_consistency_of_s2_i_conditions(discretization());

  // create strategy
  create_meshtying_strategy();

  // initialize strategy
  strategy_->init_meshtying();

  // we have successfully initialized this class
  set_is_init(true);
}  // ScaTraTimIntImpl::init()


/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::setup()
{
  // we have to call init() first
  check_is_init();

  // compute Null Space
  compute_null_space_if_necessary();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices: local <-> global dof numbering
  // -------------------------------------------------------------------
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // initialize the scalar handler
  if (scalarhandler_ == nullptr)
    FOUR_C_THROW("Make sure you construct the scalarhandler_ in initialization.");
  else
    scalarhandler_->setup(this);

  // setup splitter (needed to solve initialization problems before setup_meshtying())
  setup_splitter();

  // setup the matrix block maps and the meshtying strategy
  setup_matrix_block_maps_and_meshtying();

  // -------------------------------------------------------------------
  // create empty system matrix (27 adjacent nodes as 'good' guess)
  // -------------------------------------------------------------------
  if (fssgd_ != Inpar::ScaTra::fssugrdiff_no and not incremental_)
  {
    // do not save the graph if fine-scale subgrid diffusivity is used in non-incremental case (very
    // special case)
    sysmat_ = std::make_shared<Core::LinAlg::SparseMatrix>(*(discret_->dof_row_map()), 27);
  }
  else
    sysmat_ = init_system_matrix();

  // for some special meshtying cases we need to override the information with the information from
  // the meshtying strategy, in such a case the meshtying strategy implements the method below
  if (strategy_->system_matrix_initialization_needed()) sysmat_ = strategy_->init_system_matrix();

  // -------------------------------------------------------------------
  // create vectors containing problem variables
  // -------------------------------------------------------------------
  // solutions at time n+1 and n
  phinp_ = Core::LinAlg::create_vector(*dofrowmap, true);
  phin_ = Core::LinAlg::create_vector(*dofrowmap, true);

  setup_context_vector();

  if (nds_micro() != -1)
    phinp_micro_ = Core::LinAlg::create_vector(*discret_->dof_row_map(nds_micro()));

  if (solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro or
      solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken or
      solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken_dofsplit or
      solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_microtomacro)
  {
    phinp_inc_ = Core::LinAlg::create_vector(*dofrowmap, true);
    if (solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken or
        solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken_dofsplit)
    {
      phinp_inc_old_ = Core::LinAlg::create_vector(*dofrowmap, true);
      if (solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken)
        omega_.resize(1, 1.);
      else
        omega_.resize(num_dof_per_node(), 1.);
    }
  }

  // temporal solution derivative at time n+1
  phidtnp_ = Core::LinAlg::create_vector(*dofrowmap, true);
  // temporal solution derivative at time n
  phidtn_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // history vector (a linear combination of phinm, phin (BDF)
  // or phin, phidtn (One-Step-Theta, Generalized-alpha))
  hist_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // -------------------------------------------------------------------
  // create vectors associated to boundary conditions
  // -------------------------------------------------------------------
  // a vector of zeros to be used to enforce zero dirichlet boundary conditions
  zeros_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // object holds maps/subsets for DOFs subjected to Dirichlet BCs and otherwise
  dbcmaps_ = std::make_shared<Core::LinAlg::MapExtractor>();
  {
    Teuchos::ParameterList eleparams;
    // other parameters needed by the elements
    eleparams.set("total time", time_);
    eleparams.set<const Core::Utils::FunctionManager*>(
        "function_manager", &Global::Problem::instance()->function_manager());
    const Core::ProblemType problem_type = Core::ProblemType::scatra;
    eleparams.set<const Core::ProblemType*>("problem_type", &problem_type);
    discret_->evaluate_dirichlet(eleparams, zeros_, nullptr, nullptr, nullptr, dbcmaps_);
    zeros_->put_scalar(0.0);  // just in case of change
  }

  // -------------------------------------------------------------------
  // create vectors associated to solution process
  // -------------------------------------------------------------------
  // the vector containing body and surface forces
  neumann_loads_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // the residual vector --- more or less the rhs
  residual_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // residual vector containing the normal boundary fluxes
  trueresidual_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // incremental solution vector
  increment_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // subgrid-diffusivity(-scaling) vector
  // (used either for AVM3 approach or temperature equation
  //  with all-scale subgrid-diffusivity model)
  if (fssgd_ != Inpar::ScaTra::fssugrdiff_no)
    subgrdiff_ = Core::LinAlg::create_vector(*dofrowmap, true);

  // -------------------------------------------------------------------
  // set parameters associated to potential statistical flux evaluations
  // -------------------------------------------------------------------
  // initialize vector for statistics (assume a maximum of 10 conditions)
  sumnormfluxintegral_ = std::make_shared<Core::LinAlg::SerialDenseVector>(10);

  if (calcflux_domain_ != Inpar::ScaTra::flux_none or
      calcflux_boundary_ != Inpar::ScaTra::flux_none)
  {
    // safety check
    if (not scalarhandler_->equal_num_dof())
    {
      FOUR_C_THROW(
          "Flux output only implement for equal number of DOFs per node within ScaTra "
          "discretization!");
    }

    // if the writefluxids vector has not been set yet
    if (writefluxids_->empty())
    {
      // write one by one of scalars (as flux output in the input file defined)
      // to the temporary variable word1
      int word1 = 0;
      std::istringstream mystream(Teuchos::getNumericStringParameter(*params_, "WRITEFLUX_IDS"));
      while (mystream >> word1)
        // get desired scalar id's for flux output
        writefluxids_->push_back(word1);
    }

    // default value (-1): flux is written for all dof's
    // scalar transport: numdofpernode_ = numscal_
    // elch:             numdofpernode_ = numscal_+1
    // -> current flux for potential only if div i is used to close the system otherwise zero
    if (writefluxids_->size() and
        (*writefluxids_)[0] == (-1))  // default is to perform flux output for ALL scalars
    {
      writefluxids_->resize(num_dof_per_node());
      for (int k = 0; k < num_dof_per_node(); ++k) (*writefluxids_)[k] = k + 1;
    }

    // flux_ vector is initialized when CalcFlux() is called

    // screen output
    if (myrank_ == 0)
    {
      Core::IO::cout << "Flux output is performed for " << writefluxids_->size() << " scalars: ";
      for (unsigned int i = 0; i < writefluxids_->size(); i++)
      {
        const int id = (*writefluxids_)[i];
        Core::IO::cout << id << " ";
        if ((id < 1) or (id > num_dof_per_node()))  // check validity of these numbers as well !
          FOUR_C_THROW("Received illegal scalar id for flux output: {}", id);
      }
      Core::IO::cout << Core::IO::endl;
    }

    // initialize map extractor associated with boundary segments for flux calculation
    if (calcflux_boundary_ != Inpar::ScaTra::flux_none)
    {
      std::vector<const Core::Conditions::Condition*> conditions;
      discret_->get_condition("ScaTraFluxCalc", conditions);
      // set up map extractor
      flux_boundary_maps_ = std::make_shared<Core::LinAlg::MultiMapExtractor>();
      std::vector<Core::Conditions::Selector> selectors;
      for (const auto& c : conditions)
        selectors.emplace_back(
            Core::Conditions::Selector(std::vector<const Core::Conditions::Condition*>{c}));
      Core::Conditions::setup_extractor(*discret_, *flux_boundary_maps_, selectors, true);
    }
  }

  // -------------------------------------------------------------------
  // preparations for turbulence models
  // -------------------------------------------------------------------
  const Core::LinAlg::Map* noderowmap = discret_->node_row_map();
  init_turbulence_model(dofrowmap, noderowmap);

  // -------------------------------------------------------------------
  // set initial field
  // -------------------------------------------------------------------
  set_initial_field(
      Teuchos::getIntegralValue<Inpar::ScaTra::InitialField>(*params_, "INITIALFIELD"),
      params_->get<int>("INITFUNCNO"));

  // -------------------------------------------------------------------
  // preparations for natural convection
  // -------------------------------------------------------------------
  if (params_->get<bool>("NATURAL_CONVECTION"))
  {
    // allocate global density vector and initialize
    densafnp_ = Core::LinAlg::create_vector(*discret_->dof_row_map(), true);
    densafnp_->put_scalar(1.);
  }

  // -------------------------------------------------------------------
  // preparations for total and mean values of transported scalars
  // -------------------------------------------------------------------
  if (outputscalars_ != Inpar::ScaTra::outputscalars_none)
  {
    // input check
    if (outputscalars_ == Inpar::ScaTra::outputscalars_entiredomain)
    {
      std::vector<const Core::Conditions::Condition*> conditions;
      // extract conditions for calculation of total and mean values of transported scalars
      discret_->get_condition("TotalAndMeanScalar", conditions);
      // input check
      if (conditions.size())
      {
        FOUR_C_THROW(
            "Found 'DESIGN TOTAL AND MEAN SCALAR' condition on ScaTra discretization, but "
            "'OUTPUTSCALAR' \n"
            "in 'SCALAR TRANSPORT DYNAMIC' is set to 'entire domain'. Either switch on the output "
            "of mean and total scalars\n"
            "on conditions or remove the 'DESIGN TOTAL AND MEAN SCALAR' condition from your input "
            "file!");
      }
    }

    // build helper class for total and mean scalar output depending on input parameter
    switch (outputscalars_)
    {
      case Inpar::ScaTra::outputscalars_entiredomain:
        outputscalarstrategy_ = std::make_shared<OutputScalarsStrategyDomain>(this);
        break;
      case Inpar::ScaTra::outputscalars_condition:
        outputscalarstrategy_ = std::make_shared<OutputScalarsStrategyCondition>(this);
        break;
      case Inpar::ScaTra::outputscalars_entiredomain_condition:
        outputscalarstrategy_ = std::make_shared<OutputScalarsStrategyDomainAndCondition>(this);
        break;
      default:
        FOUR_C_THROW("Unknown option for output of total and mean scalars!");
        break;
    }
  }
  else
  {
    // input check

    std::vector<const Core::Conditions::Condition*> conditions;
    // extract conditions for calculation of total and mean values of transported scalars
    discret_->get_condition("TotalAndMeanScalar", conditions);
    // input check
    if (conditions.size())
    {
      FOUR_C_THROW(
          "Found 'DESIGN TOTAL AND MEAN SCALAR' condition on ScaTra discretization, but "
          "'OUTPUTSCALAR' \n"
          "in 'SCALAR TRANSPORT DYNAMIC' is set to 'none'. Either switch on the output of mean and "
          "total scalars\n"
          "or remove the 'DESIGN TOTAL AND MEAN SCALAR' condition from your input file!");
    }
  }

  // -------------------------------------------------------------------
  // preparations for domain integrals
  // -------------------------------------------------------------------
  if (computeintegrals_ != Inpar::ScaTra::computeintegrals_none)
  {
    // initialize domain integral output strategy
    outputdomainintegralstrategy_ = std::make_shared<OutputDomainIntegralStrategy>(this);
  }
  else
  {
    // input check
    std::vector<const Core::Conditions::Condition*> conditions_boundary;
    // extract conditions for calculation of total and mean values of transported scalars
    discret_->get_condition("BoundaryIntegral", conditions_boundary);
    std::vector<const Core::Conditions::Condition*> conditions_domain;
    // extract conditions for calculation of total and mean values of transported scalars
    discret_->get_condition("DomainIntegral", conditions_domain);
    // input check
    if (conditions_boundary.size() > 0 || conditions_domain.size() > 0)
    {
      FOUR_C_THROW(
          "Found 'DESIGN DOMAIN INTEGRAL SURF CONDITIONS' or 'DESIGN DOMAIN INTEGRAL VOL "
          "CONDITIONS' condition on ScaTra discretization, but COMPUTEINTEGRALS\n"
          "in 'SCALAR TRANSPORT DYNAMIC' is set to 'none'. Either switch on the output of domain "
          "integrals "
          "or remove the 'DESIGN DOMAIN INTEGRAL * CONDITIONS' condition from your input file!");
    }
  }

  // -------------------------------------------------------------------
  // preparations for error evaluation
  // -------------------------------------------------------------------
  if (calcerror_ != Inpar::ScaTra::calcerror_no)
  {
    if (calcerror_ == Inpar::ScaTra::calcerror_bycondition)
    {
      std::vector<const Core::Conditions::Condition*> relerrorconditions;
      discret_->get_condition("ScatraRelError", relerrorconditions);
      const unsigned ncond = relerrorconditions.size();
      if (!ncond)
      {
        FOUR_C_THROW(
            "Calculation of relative error based on conditions desired, but no conditions "
            "specified!");
      }
      relerrors_ =
          std::make_shared<std::vector<double>>(2 * num_dof_per_node() * relerrorconditions.size());
    }
    else if (calcerror_ == Inpar::ScaTra::calcerror_AnalyticSeries)
      relerrors_ = std::make_shared<std::vector<double>>(2);  // TODO: Update two n species
    else
    {
      // It is important to make a distinction as HDG always have NumDofPerNode = 0
      // The vector is therefore sized to contain the errors of one scalar and its gradient
      if (Global::Problem::instance()->spatial_approximation_type() ==
          Core::FE::ShapeFunctionType::hdg)
        relerrors_ = std::make_shared<std::vector<double>>(2);  // TODO: update to n species
      else
        relerrors_ = std::make_shared<std::vector<double>>(2 * num_dof_per_node());
    }
  }

  // we have successfully set up this class
  set_is_setup(true);
}  // ScaTraTimIntImpl::setup()

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::setup_context_vector()
{
  if (scalarhandler_->equal_num_dof())
  {
    std::vector<std::string> context;
    context.reserve(num_dof_per_node());
    for (int i = 0; i < num_dof_per_node(); ++i)
    {
      phi_components_.emplace_back("phi_" + std::to_string(i + 1));
    }
  }
  else
  {
    std::vector<std::string> context;
    context.reserve(max_num_dof_per_node());
    for (int i = 0; i < max_num_dof_per_node(); ++i)
    {
      phi_components_.emplace_back("phi_" + std::to_string(i + 1));
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::setup_nat_conv()
{
  // calculate the initial mean concentration value
  if (num_scal() < 1) FOUR_C_THROW("Error since numscal = {}. Not allowed since < 1", num_scal());
  c0_.resize(num_scal());

  discret_->set_state("phinp", *phinp_);

  // set action for elements
  Teuchos::ParameterList eleparams;
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_total_and_mean_scalars, eleparams);
  eleparams.set("inverting", false);
  eleparams.set("calc_grad_phi", false);

  // evaluate integrals of concentrations and domain
  Core::LinAlg::SerialDenseVector scalars(num_scal() + 1);
  discret_->evaluate_scalars(eleparams, scalars);

  // calculate mean concentrations
  const double domint = (scalars)[num_scal()];
  if (std::abs(domint) < 1e-15) FOUR_C_THROW("Domain has zero volume!");
  for (int k = 0; k < num_scal(); ++k) c0_[k] = (scalars)[k] / domint;

  // initialization of the densification coefficient vector
  densific_.resize(num_scal());
  Core::Elements::Element* element = discret_->l_row_element(0);
  std::shared_ptr<Core::Mat::Material> mat = element->material();

  if (mat->material_type() == Core::Materials::m_matlist or
      mat->material_type() == Core::Materials::m_matlist_reactions)
  {
    std::shared_ptr<const Mat::MatList> actmat = std::static_pointer_cast<const Mat::MatList>(mat);

    for (int k = 0; k < num_scal(); ++k)
    {
      const int matid = actmat->mat_id(k);
      std::shared_ptr<const Core::Mat::Material> singlemat = actmat->material_by_id(matid);

      if (singlemat->material_type() == Core::Materials::m_scatra)
      {
        std::shared_ptr<const Mat::ScatraMat> actsinglemat =
            std::static_pointer_cast<const Mat::ScatraMat>(singlemat);

        densific_[k] = actsinglemat->densification();

        if (densific_[k] < 0.0) FOUR_C_THROW("received negative densification value");
      }
      else
        FOUR_C_THROW("Material type is not allowed!");
    }
  }

  // for a single species calculation
  else if (mat->material_type() == Core::Materials::m_scatra)
  {
    std::shared_ptr<const Mat::ScatraMat> actmat =
        std::static_pointer_cast<const Mat::ScatraMat>(mat);

    densific_[0] = actmat->densification();

    if (densific_[0] < 0.0) FOUR_C_THROW("received negative densification value");
    if (num_scal() > 1) FOUR_C_THROW("Single species calculation but numscal = {} > 1", num_scal());
  }
  else
    FOUR_C_THROW("Material type is not allowed!");
}  // ScaTraTimIntImpl::SetupNatConv


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::init_turbulence_model(
    const Core::LinAlg::Map* dofrowmap, const Core::LinAlg::Map* noderowmap)
{
  // get fluid turbulence sublist
  Teuchos::ParameterList* turbparams = &(extraparams_->sublist("TURBULENCE MODEL"));

  // parameters for statistical evaluation of normal fluxes
  samstart_ = turbparams->get<int>("SAMPLING_START");
  samstop_ = turbparams->get<int>("SAMPLING_STOP");
  dumperiod_ = turbparams->get<int>("DUMPING_PERIOD");
  if (dumperiod_ < 0) FOUR_C_THROW("dumperiod_ is negative!");

  // -------------------------------------------------------------------
  // necessary only for AVM3 approach:
  // initialize subgrid-diffusivity matrix + respective output
  // -------------------------------------------------------------------
  if (fssgd_ != Inpar::ScaTra::fssugrdiff_no and turbparams->get<bool>("TURBMODEL_LS"))
  {
    sysmat_sd_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap, 27);

    // Output
    if (myrank_ == 0)
    {
      std::cout << "SCATRA: Fine-scale subgrid-diffusivity approach based on AVM3: ";
      std::cout << fssgd_;
      std::cout << " with turbulent Prandtl number Prt= ";
      std::cout << extraparams_->sublist("SUBGRID VISCOSITY").get<double>("C_TURBPRANDTL");
      std::cout << &std::endl << &std::endl;
    }

    if (turbparams->get<std::string>("PHYSICAL_MODEL") != "Multifractal_Subgrid_Scales")
    {
      if (fssgd_ == Inpar::ScaTra::fssugrdiff_smagorinsky_small and
          turbparams->get<Inpar::FLUID::FineSubgridVisc>("FSSUGRVISC") !=
              Inpar::FLUID::FineSubgridVisc::smagorinsky_small)
        FOUR_C_THROW("Same subgrid-viscosity approach expected!");
      if (fssgd_ == Inpar::ScaTra::fssugrdiff_smagorinsky_all and
          turbparams->get<Inpar::FLUID::FineSubgridVisc>("FSSUGRVISC") !=
              Inpar::FLUID::FineSubgridVisc::smagorinsky_all)
        FOUR_C_THROW("Same subgrid-viscosity approach expected!");
    }
  }
  else
    fssgd_ = Inpar::ScaTra::fssugrdiff_no;  // in case of not "TURBMODEL_LS"

  // -------------------------------------------------------------------
  // get turbulence model and parameters
  // -------------------------------------------------------------------
  turbmodel_ = Inpar::FLUID::no_model;

  if (turbparams->get<bool>("TURBMODEL_LS"))
  {
    // set turbulence model
    if (turbparams->get<std::string>("PHYSICAL_MODEL") == "Smagorinsky")
    {
      turbmodel_ = Inpar::FLUID::smagorinsky;

      // Output
      if (turbmodel_ and myrank_ == 0)
      {
        std::cout << "All-scale subgrid-diffusivity model: ";
        std::cout << turbparams->get<std::string>("PHYSICAL_MODEL");
        std::cout << &std::endl << &std::endl;
      }
    }
    else if (turbparams->get<std::string>("PHYSICAL_MODEL") == "Dynamic_Smagorinsky")
    {
      turbmodel_ = Inpar::FLUID::dynamic_smagorinsky;
      // access to the dynamic Smagorinsky class will provided by the
      // scatra fluid couling algorithm
    }
    else if (turbparams->get<std::string>("PHYSICAL_MODEL") == "Multifractal_Subgrid_Scales")
    {
      turbmodel_ = Inpar::FLUID::multifractal_subgrid_scales;

      // initialize matrix used to build the scale separation operator
      sysmat_sd_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap, 27);

      Teuchos::ParameterList* mfsparams = &(extraparams_->sublist("MULTIFRACTAL SUBGRID SCALES"));
      if (mfsparams->get<std::string>("SCALE_SEPARATION") != "algebraic_multigrid_operator")
        FOUR_C_THROW("Only scale separation by plain algebraic multigrid available in scatra!");

      // Output
      if (turbmodel_ and myrank_ == 0)
      {
        std::cout << "Multifractal subgrid-scale model: ";
        std::cout << turbparams->get<std::string>("PHYSICAL_MODEL");
        std::cout << &std::endl << &std::endl;
      }
    }
    else if (turbparams->get<std::string>("PHYSICAL_MODEL") == "Dynamic_Vreman")
    {
      // equivalent to dynamic smagorinsky
      turbmodel_ = Inpar::FLUID::dynamic_vreman;
    }

    // warning No. 1: if classical (all-scale) turbulence model other than
    // Smagorinsky or multifractal subrgid-scale modeling
    // is intended to be used
    if (turbparams->get<std::string>("PHYSICAL_MODEL") != "Smagorinsky" and
        turbparams->get<std::string>("PHYSICAL_MODEL") != "Dynamic_Smagorinsky" and
        turbparams->get<std::string>("PHYSICAL_MODEL") != "Multifractal_Subgrid_Scales" and
        turbparams->get<std::string>("PHYSICAL_MODEL") != "Dynamic_Vreman" and
        turbparams->get<std::string>("PHYSICAL_MODEL") != "no_model")
    {
      FOUR_C_THROW(
          "No classical (all-scale) turbulence model other than constant-coefficient Smagorinsky "
          "model and multifractal subrgid-scale modeling currently possible!");
    }

    // warning No. 2: if classical (all-scale) turbulence model and fine-scale
    // subgrid-viscosity approach are intended to be used simultaneously
    if (turbmodel_ == Inpar::FLUID::smagorinsky and fssgd_ != Inpar::ScaTra::fssugrdiff_no)
    {
      FOUR_C_THROW(
          "No combination of classical turbulence model and fine-scale subgrid-diffusivity "
          "approach currently possible!");
    }
  }

  if (turbmodel_ != Inpar::FLUID::no_model and num_scal() > 1)
    FOUR_C_THROW("Turbulent passive scalar transport not supported for more than one scalar!");

  // -------------------------------------------------------------------
  // initialize forcing for homogeneous isotropic turbulence
  // -------------------------------------------------------------------
  // flag for special flow
  special_flow_ =
      extraparams_->sublist("TURBULENCE MODEL").get<std::string>("CANONICAL_FLOW", "no");
  if (special_flow_ == "scatra_forced_homogeneous_isotropic_turbulence")
  {
    if (extraparams_->sublist("TURBULENCE MODEL").get<std::string>("SCALAR_FORCING") == "isotropic")
    {
      forcing_ = Core::LinAlg::create_vector(*dofrowmap, true);
      forcing_->put_scalar(0.0);
    }
  }
}  // ScaTraTimIntImpl::InitTurbulenceModel()


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::prepare_krylov_projection()
{
  // sysmat might be singular (some modes are defined only up to a constant)
  // in this case, we need basis vectors for the nullspace/kernel

  // get condition "KrylovSpaceProjection" from discretization
  std::vector<const Core::Conditions::Condition*> KSPCond;
  discret_->get_condition("KrylovSpaceProjection", KSPCond);
  std::size_t numcond = KSPCond.size();
  int numscatra = 0;

  const Core::Conditions::Condition* kspcond = nullptr;
  // check if for scatra Krylov projection is required
  for (std::size_t icond = 0; icond < numcond; icond++)
  {
    const auto& name = KSPCond[icond]->parameters().get<std::string>("DIS");
    if (name == "scatra")
    {
      numscatra++;
      kspcond = KSPCond[icond];
    }
  }

  // initialize variables for Krylov projection if necessary
  if (numscatra == 1)
  {
    setup_krylov_space_projection(kspcond);
    if (myrank_ == 0)
      std::cout << "\nSetup of KrylovSpaceProjection in scatra field\n" << std::endl;
  }
  else if (numscatra == 0)
  {
    projector_ = nullptr;
  }
  else
    FOUR_C_THROW("Received more than one KrylovSpaceCondition for scatra field");
}

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_element_nodeset_parameters() const
{
  Teuchos::ParameterList eleparams;

  eleparams.set<int>("ndsdisp", nds_disp());
  eleparams.set<int>("ndsgrowth", nds_growth());
  eleparams.set<int>("ndspres", nds_pressure());
  eleparams.set<int>("ndsscatra", nds_scatra());
  eleparams.set<int>("ndsthermo", nds_thermo());
  eleparams.set<int>("ndsTwoTensorQuantity", nds_two_tensor_quantity());
  eleparams.set<int>("ndsvel", nds_vel());
  eleparams.set<int>("ndswss", nds_wall_shear_stress());

  Discret::Elements::ScaTraEleParameterStd::instance(discret_->name())
      ->set_nodeset_parameters(eleparams);
}

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_element_general_parameters(bool calcinitialtimederivative) const
{
  Teuchos::ParameterList eleparams;

  // set problem number
  eleparams.set<int>("probnum", probnum_);

  eleparams.set<Inpar::ScaTra::ConvForm>("convform", convform_);

  eleparams.set<bool>("isale", isale_);

  // set flag for writing the flux vector fields
  eleparams.set<Inpar::ScaTra::FluxType>("calcflux_domain", calcflux_domain_);

  //! set vector containing ids of scalars for which flux vectors are calculated
  eleparams.set<std::shared_ptr<std::vector<int>>>("writeflux_ids", writefluxids_);

  // parameters for stabilization
  eleparams.sublist("stabilization") = params_->sublist("STABILIZATION");
  if (calcinitialtimederivative)
  {
    // deactivate stabilization when calculating initial time derivative
    eleparams.sublist("stabilization")
        .set<Inpar::ScaTra::StabType>(
            "STABTYPE", Inpar::ScaTra::StabType::stabtype_no_stabilization);
    eleparams.sublist("stabilization")
        .set<Inpar::ScaTra::TauType>("DEFINITION_TAU", Inpar::ScaTra::TauType::tau_zero);
    // deactivate subgrid-scale velocity
    eleparams.sublist("stabilization").set<bool>("SUGRVEL", false);
    // deactivate subgrid diffusivity
    eleparams.sublist("stabilization").set<bool>("ASSUGRDIFF", false);
  }

  // parameters for finite difference check
  if (calcinitialtimederivative)
  {  // deactivate finite difference check when calculating initial time derivative
    eleparams.set<Inpar::ScaTra::FdCheck>("fdcheck", Inpar::ScaTra::fdcheck_none);
  }
  else
    eleparams.set<Inpar::ScaTra::FdCheck>("fdcheck", fdcheck_);

  eleparams.set<double>("fdcheckeps", fdcheckeps_);
  eleparams.set<double>("fdchecktol", fdchecktol_);

  // flag for spherical coordinates
  eleparams.set<bool>("sphericalcoords", params_->get<bool>("SPHERICALCOORDS"));

  // flag for truly partitioned multi-scale simulation
  eleparams.set<bool>("partitioned_multiscale",
      solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro or
          solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken or
          solvtype_ ==
              Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken_dofsplit or
          solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_microtomacro);

  // flag for external force
  eleparams.set<bool>("has_external_force", has_external_force_);

  // add parameters associated with meshtying strategy
  strategy_->set_element_general_parameters(eleparams);

  // additional problem-specific parameters for non-standard scalar transport problems
  // (electrochemistry etc.)
  set_element_specific_scatra_parameters(eleparams);

  Discret::Elements::ScaTraEleParameterStd::instance(discret_->name())->set_parameters(eleparams);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_element_turbulence_parameters(
    bool calcinitialtimederivative) const
{
  Teuchos::ParameterList eleparams;

  eleparams.sublist("TURBULENCE MODEL") = extraparams_->sublist("TURBULENCE MODEL");
  if (calcinitialtimederivative)
  {
    // deactivate turbulence model when calculating initial time
    // derivative
    Teuchos::setStringToIntegralParameter<Inpar::FLUID::TurbModelAction>("PHYSICAL_MODEL",
        "no_model",
        "Classical LES approaches require an additional model for\nthe turbulent viscosity.",
        Teuchos::tuple<std::string>("no_model"),
        Teuchos::tuple<std::string>("If classical LES is our turbulence approach, this is a "
                                    "contradiction and should cause a FOUR_C_THROW."),
        Teuchos::tuple<Inpar::FLUID::TurbModelAction>(Inpar::FLUID::no_model),
        &eleparams.sublist("TURBULENCE MODEL"));
  }

  // set model-dependent parameters
  eleparams.sublist("SUBGRID VISCOSITY") = extraparams_->sublist("SUBGRID VISCOSITY");

  // set parameters for multifractal subgrid-scale modeling
  eleparams.sublist("MULTIFRACTAL SUBGRID SCALES") =
      extraparams_->sublist("MULTIFRACTAL SUBGRID SCALES");

  eleparams.set<bool>("turbulent inflow", turbinflow_);

  if (calcinitialtimederivative)
  {
    eleparams.set<Inpar::ScaTra::FSSUGRDIFF>(
        "fs subgrid diffusivity", Inpar::ScaTra::fssugrdiff_no);
  }
  else
  {
    eleparams.set<Inpar::ScaTra::FSSUGRDIFF>("fs subgrid diffusivity", fssgd_);
  }

  Discret::Elements::ScaTraEleParameterTurbulence::instance(discret_->name())
      ->set_parameters(eleparams);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::prepare_time_loop()
{
  // provide information about initial field (do not do for restarts!)
  if (step_ == 0)
  {
    // write out initial state
    check_and_write_output_and_restart();

    // compute error for problems with analytical solution (initial field!)
    evaluate_error_compared_to_analytical_sol();

    // calculate mean concentration of micro discretization and set state to nds_micro_
    if (macro_scale_ and nds_micro() != -1) calc_mean_micro_concentration();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::prepare_time_step()
{
  // time measurement: prepare time step
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:    + prepare time step");

  // -------------------------------------------------------------------
  //                       initialization
  // -------------------------------------------------------------------
  if (step_ == 0) prepare_first_time_step();

  // -------------------------------------------------------------------
  //              set time dependent parameters
  // -------------------------------------------------------------------
  // adapt time step size if desired
  adapt_time_step_size();

  // tell micro scale about updated time step
  if (macro_scale_ and time_step_adapted()) set_time_stepping_to_micro_scale();

  // note the order of the following three functions is important
  increment_time_and_step();

  // -------------------------------------------------------------------
  // set part of the rhs vector belonging to the old timestep
  // -------------------------------------------------------------------
  set_old_part_of_righthandside();
  // TODO (Thon): We do not really want to call set_element_time_parameter() every time step.
  // But for now we just do it since "total time" has to be changed in the parameter class..
  set_element_time_parameter();

  // -------------------------------------------------------------------
  //         evaluate Dirichlet and Neumann boundary conditions
  // -------------------------------------------------------------------
  // TODO: Dirichlet auch im Fall von genalpha phinp
  // Neumann(n + alpha_f)
  apply_dirichlet_bc(time_, phinp_, nullptr);
  apply_neumann_bc(neumann_loads_);

  // By definition: Applying DC on the slave side of an internal interface is not allowed
  //                since it leads to an over-constraint system
  // Therefore, nodes belonging to the slave side of an internal interface have to be excluded from
  // the DC. However, a velocity value (projected from the Dirichlet condition on the master side)
  // has to be assigned to the DOF's on the slave side in order to evaluate the system matrix
  // completely

  // Preparation for including DC on the master side in the condensation process
  strategy_->include_dirichlet_in_condensation();

  // -------------------------------------------------------------------
  //     update velocity field if given by function (it might depend on time)
  // -------------------------------------------------------------------
  if (velocity_field_type_ == Inpar::ScaTra::velocity_function) set_velocity_field_from_function();

  // -------------------------------------------------------------------
  //     update external force given by function (it might depend on time)
  // -------------------------------------------------------------------
  if (has_external_force_) set_external_force();

  // -------------------------------------------------------------------
  //           preparation of AVM3-based scale separation
  // -------------------------------------------------------------------
  if ((step_ == 1 or (turbinflow_ and step_ == numinflowsteps_ + 1)) and
      (fssgd_ != Inpar::ScaTra::fssugrdiff_no or
          turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales))
    avm3_preparation();

  // -------------------------------------------------------------------
  // compute values at intermediate time steps (only for gen.-alpha)
  // -------------------------------------------------------------------
  compute_intermediate_values();

  // -------------------------------------------------------------------
  // prepare time step on micro scale if necessary
  // -------------------------------------------------------------------
  if (macro_scale_)
  {
    // create parameter list for macro-scale elements
    Teuchos::ParameterList eleparams;

    // set action
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
        "action", ScaTra::Action::micro_scale_prepare_time_step, eleparams);

    // add state vectors
    add_time_integration_specific_vectors();

    // loop over macro-scale elements
    discret_->evaluate(eleparams, nullptr, nullptr, nullptr, nullptr, nullptr);
  }
}

/*------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::prepare_first_time_step()
{
  if (not skipinitder_)
  {
    if (nds_vel() != -1)  // if some velocity field has been set
    {
      // TODO: Restructure enforcement of Dirichlet boundary conditions on phin_
      // A clean solution would incorporate apply_dirichlet_bc(...) into
      // calc_initial_time_derivative(). However, this would make a number of test cases fail. We
      // should have a closer look at this problem and fix it eventually.
      apply_dirichlet_bc(time_, phin_, nullptr);
      calc_initial_time_derivative();
    }

    // if initial velocity field has not been set here, the initial time derivative of phi will be
    // calculated wrongly for some time integration schemes
    else
      FOUR_C_THROW("Initial velocity field has not been set!");
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::prepare_linear_solve()
{
  // special preparations for multifractal subgrid-scale model
  if (turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales) recompute_mean_csgs_b();

  // call elements to calculate system matrix and rhs and assemble
  assemble_mat_and_rhs();

  // apply Dirichlet boundary conditions
  apply_dirichlet_to_system();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_velocity_field_from_function()
{
  // safety check
  if (nds_vel() >= discret_->num_dof_sets())
    FOUR_C_THROW("Too few dofsets on scatra discretization!");

  // initialize velocity vectors
  std::shared_ptr<Core::LinAlg::Vector<double>> convel =
      create_vector(*discret_->dof_row_map(nds_vel()), true);
  std::shared_ptr<Core::LinAlg::Vector<double>> vel =
      create_vector(*discret_->dof_row_map(nds_vel()), true);

  switch (velocity_field_type_)
  {
    case Inpar::ScaTra::velocity_zero:
    {
      // no action needed in case for zero velocity field
      break;
    }

    case Inpar::ScaTra::velocity_function:
    {
      int err(0);
      const int velfuncno = params_->get<int>("VELFUNCNO");

      // loop all nodes on the processor
      for (int lnodeid = 0; lnodeid < discret_->num_my_row_nodes(); lnodeid++)
      {
        // get the processor local node
        const Core::Nodes::Node* lnode = discret_->l_row_node(lnodeid);

        // get dofs associated with current node
        std::vector<int> nodedofs = discret_->dof(nds_vel(), lnode);

        for (int index = 0; index < nsd_; ++index)
        {
          double value =
              problem_->function_by_id<Core::Utils::FunctionOfSpaceTime>(velfuncno).evaluate(
                  lnode->x(), time_, index);

          // get global and local dof IDs
          const int gid = nodedofs[index];
          const int lid = convel->get_map().lid(gid);

          if (lid < 0) FOUR_C_THROW("Local ID not found in map for given global ID!");
          err = convel->replace_local_value(lid, value);
          if (err != 0) FOUR_C_THROW("error while inserting a value into convel");
          err = vel->replace_local_value(lid, value);
          if (err != 0) FOUR_C_THROW("error while inserting a value into vel");
        }
      }

      break;
    }

    default:
    {
      FOUR_C_THROW("Wrong SetVelocity() action for velocity field type {}!", velocity_field_type_);
      break;
    }
  }

  // provide scatra discretization with convective velocity
  discret_->set_state(nds_vel(), "convective velocity field", *convel);

  // provide scatra discretization with velocity
  discret_->set_state(nds_vel(), "velocity field", *vel);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_external_force() const
{
  const auto input_params_external_force = params_->sublist("EXTERNAL FORCE");
  const int external_force_function_id = input_params_external_force.get<int>("FORCE_FUNCTION_ID");
  const int intrinsic_mobility_function_id =
      input_params_external_force.get<int>("INTRINSIC_MOBILITY_FUNCTION_ID");

  if (nds_vel() >= discret_->num_dof_sets())
    FOUR_C_THROW("Too few dofsets on scatra discretization!");

  // vector for the external force
  auto external_force = Core::LinAlg::create_vector(*discret_->dof_row_map(nds_vel()), true);

  // vector for the intrinsic mobility
  auto intrinsic_mobility = Core::LinAlg::create_vector(*discret_->dof_row_map(nds_vel()), true);

  // vector for the velocity due to the external force:
  // force_velocity = intrinsic_mobility * external_force
  auto force_velocity = Core::LinAlg::create_vector(*discret_->dof_row_map(nds_vel()), true);

  // magnetic field
  auto magnetic_field = Core::LinAlg::create_vector(*discret_->dof_row_map(nds_vel()), true);

  for (int lnodeid = 0; lnodeid < discret_->num_my_row_nodes(); lnodeid++)
  {
    auto* const current_node = discret_->l_row_node(lnodeid);
    const auto nodedofs = discret_->dof(nds_vel(), current_node);

    std::vector<double> external_force_vector(3);
    problem_->function_by_id<Core::Utils::FunctionOfSpaceTime>(external_force_function_id)
        .evaluate_vector(current_node->x(), time_, external_force_vector);

    for (int spatial_dimension = 0; spatial_dimension < nsd_; ++spatial_dimension)
    {
      const double external_force_value = external_force_vector[spatial_dimension];
      const double intrinsic_mobility_value =
          problem_->function_by_id<Core::Utils::FunctionOfSpaceTime>(intrinsic_mobility_function_id)
              .evaluate(current_node->x(), time_, spatial_dimension);
      const double force_velocity_value = external_force_value * intrinsic_mobility_value;

      const int gid = nodedofs[spatial_dimension];
      const int lid = force_velocity->get_map().lid(gid);

      if (lid < 0) FOUR_C_THROW("Local ID not found in map for given global ID!");
      const int error_force_velocity =
          force_velocity->replace_local_value(lid, force_velocity_value);
      if (error_force_velocity != 0)
        FOUR_C_THROW("Error while inserting a force_velocity_value into force_velocity.");

      const int error_external_force =
          external_force->replace_local_value(lid, external_force_value);
      if (error_external_force != 0)
        FOUR_C_THROW("Error while inserting a external_force_value into external_force.");

      const int error_intrinsic_mobility =
          intrinsic_mobility->replace_local_value(lid, intrinsic_mobility_value);
      if (error_intrinsic_mobility != 0)
        FOUR_C_THROW("Error while inserting a intrinsic_mobility_value into intrinsic_mobility.");
    }
  }

  discret_->set_state(nds_vel(), "external_force", *external_force);
  discret_->set_state(nds_vel(), "intrinsic_mobility", *intrinsic_mobility);
  discret_->set_state(nds_vel(), "force_velocity", *force_velocity);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_wall_shear_stresses(
    const Core::LinAlg::Vector<double>& wall_shear_stress) const
{
  // We rely on the fact, that the nodal distribution of both fields is the same.
  // Although Scatra discretization was constructed as a clone of the fluid or
  // structure mesh, respectively, at the beginning, the nodal distribution may
  // have changed meanwhile (e.g., due to periodic boundary conditions applied only
  // to the fluid field)!
  // We have to be sure that everything is still matching.
  FOUR_C_ASSERT(
      wall_shear_stress.get_map().same_as(*discret_->dof_row_map(nds_wall_shear_stress())),
      "Maps are NOT identical. Emergency!");

  discret_->set_state(nds_wall_shear_stress(), "WallShearStress", wall_shear_stress);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_old_part_of_righthandside()
{
  // compute history values associated with meshtying strategy
  strategy_->set_old_part_of_rhs();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_pressure_field(
    const Core::LinAlg::Vector<double>& pressure) const
{
  // We rely on the fact, that the nodal distribution of both fields is the same.
  // Although Scatra discretization was constructed as a clone of the fluid or
  // structure mesh, respectively, at the beginning, the nodal distribution may
  // have changed meanwhile (e.g., due to periodic boundary conditions applied only
  // to the fluid field)!
  // We have to be sure that everything is still matching.
  FOUR_C_ASSERT(pressure.get_map().same_as(*discret_->dof_row_map(nds_pressure())),
      "Maps are NOT identical. Emergency!");

  discret_->set_state(nds_pressure(), "Pressure", pressure);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_membrane_concentration(
    std::shared_ptr<const Core::LinAlg::Vector<double>> MembraneConc)
{
  if (MembraneConc == nullptr) FOUR_C_THROW("MeanConc state is nullptr");

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // We rely on the fact, that the nodal distribution of both fields is the same.
  // Although Scatra discretization was constructed as a clone of the fluid or
  // structure mesh, respectively, at the beginning, the nodal distribution may
  // have changed meanwhile (e.g., due to periodic boundary conditions applied only
  // to the fluid field)!
  // We have to be sure that everything is still matching.
  if (not MembraneConc->get_map().same_as(*discret_->dof_row_map(0)))
    FOUR_C_THROW("Maps are NOT identical. Emergency!");
#endif

  // Note: we can not simply write this into the secondary discretisation here
  // since it is a variable of the primary dofset and is hence cleared
  // in between
  membrane_conc_ = MembraneConc;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_mean_concentration(
    std::shared_ptr<const Core::LinAlg::Vector<double>> MeanConc)
{
  if (MeanConc == nullptr) FOUR_C_THROW("MeanConc state is nullptr");

#ifdef FOUR_C_ENABLE_ASSERTIONS
  // We rely on the fact, that the nodal distribution of both fields is the same.
  // Although Scatra discretization was constructed as a clone of the fluid or
  // structure mesh, respectively, at the beginning, the nodal distribution may
  // have changed meanwhile (e.g., due to periodic boundary conditions applied only
  // to the fluid field)!
  // We have to be sure that everything is still matching.
  if (not MeanConc->get_map().same_as(*discret_->dof_row_map(0)))
    FOUR_C_THROW("Maps are NOT identical. Emergency!");
#endif

  // Note: we can not simply write this into the secondary discretisation here
  // since it is a variable of the primary dofset and is hence cleared
  // in between
  mean_conc_ = MeanConc;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_acceleration_field(
    const Core::LinAlg::Vector<double>& acceleration) const
{
  // time measurement
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA: set acceleration");

  FOUR_C_ASSERT(nds_vel() < discret_->num_dof_sets(), "Too few dof sets on scatra discretization!");

  // provide scatra discretization with acceleration field if required
  discret_->set_state(nds_vel(), "acceleration field", acceleration);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_convective_velocity(
    const Core::LinAlg::Vector<double>& convective_velocity) const
{
  // time measurement
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA: set convective velocity field");

  // checks
  FOUR_C_ASSERT(velocity_field_type_ == Inpar::ScaTra::velocity_Navier_Stokes,
      "Wrong set_velocity_field() called for velocity field type {}!", velocity_field_type_);
  FOUR_C_ASSERT(nds_vel() < discret_->num_dof_sets(), "Too few dof sets on scatra discretization!");

  // provide scatra discretization with convective velocity
  discret_->set_state(nds_vel(), "convective velocity field", convective_velocity);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_fine_scale_velocity(
    const Core::LinAlg::Vector<double>& fine_scale_velocity) const
{
  // time measurement
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA: set fine scale velocity field");

  // checks
  FOUR_C_ASSERT(velocity_field_type_ == Inpar::ScaTra::velocity_Navier_Stokes,
      "Wrong set_velocity_field() called for velocity field type {}!", velocity_field_type_);
  FOUR_C_ASSERT(nds_vel() < discret_->num_dof_sets(), "Too few dof sets on scatra discretization!");

  // provide scatra discretization with fine-scale convective velocity if required
  discret_->set_state(nds_vel(), "fine-scale velocity field", fine_scale_velocity);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool ScaTra::ScaTraTimIntImpl::fine_scale_velocity_field_required() const
{
  // for smagorinsky_all, the fine scale velocity is not necessary
  if (fssgd_ == Inpar::ScaTra::fssugrdiff_smagorinsky_all) return false;

  // in case the fine-scale subgrid-viscosity is turned off in the scalar field, the fine scale
  // velocity is not necessary
  if (turbmodel_ == Inpar::FLUID::no_model and fssgd_ == Inpar::ScaTra::fssugrdiff_no) return false;

  // do not communicate the fine scale velocity field before it was calculated, i.e., in the initial
  // step
  if (step_ < 1) return false;

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_velocity_field(const Core::LinAlg::Vector<double>& velocity)
{
  // time measurement
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA: set velocity field");

  // checks
  FOUR_C_ASSERT(velocity_field_type_ == Inpar::ScaTra::velocity_Navier_Stokes,
      "Wrong set_velocity_field() called for velocity field type {}!", velocity_field_type_);
  FOUR_C_ASSERT(nds_vel() < discret_->num_dof_sets(), "Too few dof sets on scatra discretization!");

  // provide scatra discretization with velocity
  discret_->set_state(nds_vel(), "velocity field", velocity);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::time_loop()
{
  // safety checks
  check_is_init();
  check_is_setup();

  // time measurement: time loop
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:  + time loop");

  // prepare time loop
  prepare_time_loop();

  while (not_finished())
  {
    // -------------------------------------------------------------------
    // prepare time step
    // -------------------------------------------------------------------
    prepare_time_step();

    // -------------------------------------------------------------------
    //                  solve nonlinear / linear equation
    // -------------------------------------------------------------------
    // store time before calling nonlinear solver
    double time = Teuchos::Time::wallTime();

    pre_solve();
    solve();
    post_solve();

    // determine time spent by nonlinear solver and take maximum over all processors via
    // communication
    double mydtnonlinsolve(Teuchos::Time::wallTime() - time), dtnonlinsolve(0.);
    dtnonlinsolve = Core::Communication::max_all(mydtnonlinsolve, discret_->get_comm());

    // output performance statistics associated with nonlinear solver into *.csv file if applicable
    if (params_->get<bool>("OUTPUTNONLINSOLVERSTATS"))
      output_nonlin_solver_stats(iternum_, dtnonlinsolve, step(), discret_->get_comm());

    // -------------------------------------------------------------------
    //                         update solution
    //        current solution becomes old solution of next timestep
    // -------------------------------------------------------------------
    update();

    // -------------------------------------------------------------------
    // evaluate error for problems with analytical solution
    // -------------------------------------------------------------------
    evaluate_error_compared_to_analytical_sol();

    // -------------------------------------------------------------------
    //                         output of solution
    // -------------------------------------------------------------------
    check_and_write_output_and_restart();

  }  // while
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::solve()
{
  // safety checks
  check_is_init();
  check_is_setup();

  // -----------------------------------------------------------------
  // intermediate solution step for homogeneous isotropic turbulence
  // -----------------------------------------------------------------
  if (solvtype_ == Inpar::ScaTra::solvertype_nonlinear) calc_intermediate_solution();

  // -----------------------------------------------------------------
  //                     solve (non-)linear equation
  // -----------------------------------------------------------------
  switch (solvtype_)
  {
    case Inpar::ScaTra::solvertype_linear_incremental:
    case Inpar::ScaTra::solvertype_linear_full:
    {
      linear_solve();
      break;
    }

    case Inpar::ScaTra::solvertype_nonlinear:
    {
      nonlinear_solve();
      break;
    }

    case Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro:
    case Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken:
    case Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken_dofsplit:
    case Inpar::ScaTra::solvertype_nonlinear_multiscale_microtomacro:
    {
      nonlinear_multi_scale_solve();
      break;
    }

    default:
    {
      FOUR_C_THROW("Unknown solver type!");
      break;
    }
  }
  // that's all
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::update()
{
  // update quantities associated with meshtying strategy
  strategy_->update();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::apply_mesh_movement(const Core::LinAlg::Vector<double>& dispnp) const
{
  //---------------------------------------------------------------------------
  // only required in ALE case
  //---------------------------------------------------------------------------
  if (isale_)
  {
    TEUCHOS_FUNC_TIME_MONITOR("SCATRA: apply mesh movement");

    // provide scatra discretization with displacement field
    discret_->set_state(nds_disp(), "dispnp", dispnp);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
inline void ScaTra::ScaTraTimIntImpl::print_time_step_info()
{
  if (myrank_ == 0)
  {
    std::cout << std::endl
              << "TIME: " << std::setw(11) << std::setprecision(4) << time_ << "/" << maxtime_
              << "  DT = " << dta_ << "  " << method_title() << std::setw(4) << "  STEP = " << step_
              << "/" << stepmax_ << std::endl;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> ScaTra::ScaTraTimIntImpl::system_matrix()
{
  return std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(sysmat_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> ScaTra::ScaTraTimIntImpl::block_system_matrix()
{
  return std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(sysmat_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::check_and_write_output_and_restart()
{
  // time measurement: output of solution
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:    + output of solution");

  // write result and potentially flux data
  if (is_result_step()) write_result();

  if (is_result_step()) write_runtime_output();

  // add restart data
  if (is_restart_step()) write_restart();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::write_result()
{
  // write output to Gmsh postprocessing files
  if (outputgmsh_) output_to_gmsh(step_, time_);

  // write mean values of scalar(s)
  output_total_and_mean_scalars();

  // write domain and boundary integrals, i.e., surface areas and volumes of specified nodesets
  output_domain_or_boundary_integrals("DomainIntegral");
  output_domain_or_boundary_integrals("BoundaryIntegral");

  // write integral values of reaction(s)
  output_integr_reac();

  // generate output associated with meshtying strategy
  strategy_->output();

  // generate output on micro scale if necessary
  if (macro_scale_)
  {
    // create parameter list for macro elements
    Teuchos::ParameterList eleparams;

    // set action
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
        "action", ScaTra::Action::micro_scale_output, eleparams);

    // loop over macro-scale elements
    discret_->evaluate(eleparams, nullptr, nullptr, nullptr, nullptr, nullptr);
  }

  if ((step_ != 0) and (output_state_matlab_))
  {
    std::ostringstream filename;
    filename << problem_->output_control_file()->file_name() << "-Result_Step" << step_ << ".m";
    Core::LinAlg::print_vector_in_matlab_format(filename.str(), *phinp_);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::collect_runtime_output_data()
{
  visualization_writer_->append_element_owner("Owner");

  visualization_writer_->append_result_data_vector_with_context(
      *phinp_, Core::IO::OutputEntity::dof, phi_components_);

  // convective velocity (written in case of coupled simulations since volmortar is now possible)
  if (velocity_field_type_ == Inpar::ScaTra::velocity_function or
      velocity_field_type_ == Inpar::ScaTra::velocity_Navier_Stokes)
  {
    auto convel = discret_->get_state(nds_vel(), "convective velocity field");
    if (convel == nullptr) FOUR_C_THROW("Cannot get state vector convective velocity");

    // convert dof-based vector into node-based multi-vector for postprocessing
    auto convel_multi = Core::IO::convert_dof_vector_to_node_based_multi_vector(
        *discret_, *convel, nds_vel(), nsd_);

    std::vector<std::optional<std::string>> context(nsd_, "convec_velocity");
    visualization_writer_->append_result_data_vector_with_context(
        *convel_multi, Core::IO::OutputEntity::node, context);
  }

  if (has_external_force_)
  {
    auto write_output_as_multivector = [this](const std::string& field_name)
    {
      const auto state_vector = discret_->get_state(nds_vel_, field_name);
      if (state_vector == nullptr) FOUR_C_THROW("Cannot get state vector {}", field_name);

      // convert dof-based linalg vector into node-based multi-vector for postprocessing
      auto multi_vector = Core::IO::convert_dof_vector_to_node_based_multi_vector(
          *discret_, *state_vector, nds_vel(), nsd_);

      const std::vector<std::optional<std::string>> context(nsd_, field_name);
      visualization_writer_->append_result_data_vector_with_context(
          *multi_vector, Core::IO::OutputEntity::node, context);
    };

    write_output_as_multivector("external_force");
    write_output_as_multivector("intrinsic_mobility");
    write_output_as_multivector("force_velocity");
  }

  // displacement field
  if (isale_)
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
        discret_->get_state(nds_disp(), "dispnp");
    if (dispnp == nullptr) FOUR_C_THROW("Cannot extract displacement field from discretization");

    // convert dof-based vector into node-based multi-vector for postprocessing
    auto dispnp_multi = Core::IO::convert_dof_vector_to_node_based_multi_vector(
        *discret_, *dispnp, nds_disp(), nsd_);

    std::vector<std::optional<std::string>> context(nsd_, "ale-displacement");
    visualization_writer_->append_result_data_vector_with_context(
        *dispnp_multi, Core::IO::OutputEntity::node, context);
  }

  if (output_element_material_id_) visualization_writer_->append_element_material_id();

  if (nds_micro() != -1)
  {
    // convert vector to multi vector
    auto micro_conc_multi = Core::LinAlg::MultiVector<double>(*discret_->node_row_map(), 1, true);

    for (int inode = 0; inode < discret_->num_my_row_nodes(); ++inode)
      (micro_conc_multi)(0).get_values()[inode] = (*phinp_micro_)[inode];

    visualization_writer_->append_result_data_vector_with_context(
        micro_conc_multi, Core::IO::OutputEntity::node, {"micro_conc"});
  }

  if (calcflux_domain_ != Inpar::ScaTra::flux_none or
      calcflux_boundary_ != Inpar::ScaTra::flux_none)
  {
    // for flux output of initial field (before first solve) do:
    // flux_domain_ and flux_boundary_ vectors are initialized when CalcFlux() is called
    if (step_ == 0 or (calcflux_domain_ != Inpar::ScaTra::flux_none and flux_domain_ == nullptr) or
        (calcflux_boundary_ != Inpar::ScaTra::flux_none and flux_boundary_ == nullptr))
      calc_flux(true);

    if (calcflux_domain_ != Inpar::ScaTra::flux_none)
      collect_output_flux_data(flux_domain_, "domain");
    if (calcflux_boundary_ != Inpar::ScaTra::flux_none)
      collect_output_flux_data(flux_boundary_, "boundary");
  }

  // biofilm growth
  if (scfldgrdisp_ != nullptr)
  {
    std::vector<std::optional<std::string>> context(
        scfldgrdisp_->num_vectors(), "scfld_growth_displ");
    visualization_writer_->append_result_data_vector_with_context(
        *scfldgrdisp_, Core::IO::OutputEntity::node, context);
  }

  // biofilm growth
  if (scstrgrdisp_ != nullptr)
  {
    std::vector<std::optional<std::string>> context(
        scstrgrdisp_->num_vectors(), "scstr_growth_displ");
    visualization_writer_->append_result_data_vector_with_context(
        *scstrgrdisp_, Core::IO::OutputEntity::node, context);
  }

  strategy_->collect_output_data();

  // generate output on micro scale if necessary
  if (macro_scale_)
  {
    // create parameter list for macro elements
    Teuchos::ParameterList eleparams;

    // set action
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
        "action", ScaTra::Action::collect_micro_scale_output, eleparams);

    // loop over macro-scale elements
    discret_->evaluate(eleparams, nullptr, nullptr, nullptr, nullptr, nullptr);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::write_runtime_output()
{
  visualization_writer_->reset();

  collect_runtime_output_data();

  visualization_writer_->write_to_disk(time_, step_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_initial_field(
    const Inpar::ScaTra::InitialField init, const int startfuncno)
{
  switch (init)
  {
    case Inpar::ScaTra::initfield_zero_field:
    {
      phin_->put_scalar(0.0);
      phinp_->put_scalar(0.0);
      break;
    }
    case Inpar::ScaTra::initfield_field_by_function:
    case Inpar::ScaTra::initfield_disturbed_field_by_function:
    {
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

      for (auto lnode : discret_->my_row_node_range())
      {
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->dof(0, lnode);

        int numdofs = static_cast<int>(nodedofset.size());
        for (int k = 0; k < numdofs; ++k)
        {
          const int dofgid = nodedofset[k];
          int doflid = dofrowmap->lid(dofgid);
          // evaluate component k of spatial function
          double initialval =
              problem_->function_by_id<Core::Utils::FunctionOfSpaceTime>(startfuncno)
                  .evaluate(lnode.x(), time_, k);
          int err = phin_->replace_local_value(doflid, initialval);
          if (err != 0) FOUR_C_THROW("dof not on proc");
        }
      }

      // for NURBS discretizations we have to solve a least squares problem,
      // with high accuracy! (do nothing for Lagrangian polynomials)
      const Teuchos::ParameterList& scatradyn = problem_->scalar_transport_dynamic_params();
      const int lstsolver = scatradyn.get<int>("LINEAR_SOLVER");

      auto* nurbsdis = dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(*discret_));
      if (nurbsdis != nullptr)
      {
        if (lstsolver == (-1))
        {
          FOUR_C_THROW(
              "no linear solver defined for least square NURBS problem. Please set LINEAR_SOLVER "
              "in SCALAR TRANSPORT DYNAMIC to a valid number! Note: this solver block is misused "
              "for the least square problem. Maybe one should add a separate parameter for this.");
        }

        Core::FE::Nurbs::apply_nurbs_initial_condition(*discret_,
            problem_->solver_params(lstsolver),
            problem_->function_by_id<Core::Utils::FunctionOfSpaceTime>(startfuncno), phin_);
      }

      // initialize also the solution vector. These values are a pretty good guess for the
      // solution after the first time step (much better than starting with a zero vector)
      phinp_->update(1.0, *phin_, 0.0);

      // add random perturbation for initial field of turbulent flows
      if (init == Inpar::ScaTra::initfield_disturbed_field_by_function)
      {
        int err = 0;

        // random noise is relative to difference of max-min values of initial profile
        double perc =
            extraparams_->sublist("TURBULENCE MODEL").get<double>("CHAN_AMPL_INIT_DIST", 0.1);

        // out to screen
        if (myrank_ == 0)
        {
          std::cout << "Disturbed initial scalar profile:   max. " << perc * 100
                    << "% random perturbation\n";
          std::cout << "\n\n";
        }

        // get overall max and min values and range between min and max
        double maxphi(0.0);
        double minphi(0.0);
        err = phinp_->max_value(&maxphi);
        if (err > 0) FOUR_C_THROW("Error during evaluation of maximum value.");
        err = phinp_->min_value(&minphi);
        if (err > 0) FOUR_C_THROW("Error during evaluation of minimum value.");
        double range = abs(maxphi - minphi);

        // disturb initial field for all degrees of freedom
        for (int k = 0; k < phinp_->local_length(); ++k)
        {
          double randomnumber = problem_->random()->uni();
          double noise = perc * range * randomnumber;
          err += phinp_->sum_into_local_value(k, noise);
          err += phin_->sum_into_local_value(k, noise);
          if (err != 0) FOUR_C_THROW("Error while disturbing initial field.");
        }
      }
      break;
    }
    case Inpar::ScaTra::initfield_field_by_condition:
    {
      // set initial field for ALL existing scatra fields in condition
      const std::string field = "ScaTra";

      // get initial field conditions
      std::vector<const Core::Conditions::Condition*> initfieldconditions;
      discret_->get_condition("Initfield", initfieldconditions);

      if (not initfieldconditions.size())
      {
        FOUR_C_THROW(
            "Tried to evaluate initial field by condition without a corresponding condition "
            "defined on the ScaTra discretization!");
      }
      if (scalarhandler_ == nullptr) FOUR_C_THROW("scalarhandler_ is null pointer!");

      std::set<int> numdofpernode;
      for (auto& initfieldcondition : initfieldconditions)
      {
        const int condmaxnumdofpernode =
            scalarhandler_->num_dof_per_node_in_condition(*initfieldcondition, *discret_);

        if (condmaxnumdofpernode != 0) numdofpernode.insert(condmaxnumdofpernode);
      }

      if (numdofpernode.empty()) FOUR_C_THROW("No DOFs defined on initial field condition!");

      const int maxnumdofpernode = *(numdofpernode.rbegin());

      std::vector<int> localdofs(maxnumdofpernode);
      for (int i = 0; i < maxnumdofpernode; i++)
      {
        localdofs[i] = i;
      }
      discret_->evaluate_initial_field(
          Global::Problem::instance()->function_manager(), field, *phin_, localdofs);

      // initialize also the solution vector. These values are a pretty good guess for the
      // solution after the first time step (much better than starting with a zero vector)
      phinp_->update(1.0, *phin_, 0.0);

      break;
    }
    // discontinuous 0-1 field for progress variable in 1-D
    case Inpar::ScaTra::initfield_discontprogvar_1D:
    {
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

      for (auto lnode : discret_->my_row_node_range())
      {
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->dof(0, lnode);

        // get coordinate
        const double x = lnode.x()[0];

        int numdofs = static_cast<int>(nodedofset.size());
        for (int k = 0; k < numdofs; ++k)
        {
          const int dofgid = nodedofset[k];
          int doflid = dofrowmap->lid(dofgid);

          double initialval = 0.0;
          if (x > -1e-10) initialval = 1.0;

          int err = 0;
          err += phin_->replace_local_value(doflid, initialval);
          // initialize also the solution vector. These values are a pretty good guess for the
          // solution after the first time step (much better than starting with a zero vector)
          err += phinp_->replace_local_value(doflid, initialval);
          if (err != 0) FOUR_C_THROW("dof not on proc");
        }
      }
      break;
    }
    // reconstructed initial profile for progress variable in x2-direction from
    // Lessani and Papalexandris (2006), also used in Moureau et al. (2007, 2009),
    // for two-dimensional flame-vortex interaction problem (x2=0-200)
    case Inpar::ScaTra::initfield_flame_vortex_interaction:
    {
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

      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

      // define variable
      double initialval = 0.0;

      for (auto lnode : discret_->my_row_node_range())
      {
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->dof(0, lnode);

        // get x2-coordinate
        const double x2 = lnode.x()[1];

        int numdofs = static_cast<int>(nodedofset.size());
        for (int k = 0; k < numdofs; ++k)
        {
          const int dofgid = nodedofset[k];
          int doflid = dofrowmap->lid(dofgid);

          if (x2 < loc12 - 1e-10)
            initialval = (1.0 - (1.0 / beta1)) * exp((x2 - trans1) / delta1);
          else if (x2 > loc23 + 1e-10)
            initialval = 1.0 - (exp((1.0 - beta3) * (x2 - trans3) / delta3) / beta3);
          else
            initialval = fac2 * (x2 - trans2) + abs2;

          int err = 0;
          err += phin_->replace_local_value(doflid, initialval);
          // initialize also the solution vector. These values are a pretty good guess for the
          // solution after the first time step (much better than starting with a zero vector)
          err += phinp_->replace_local_value(doflid, initialval);
          if (err != 0) FOUR_C_THROW("dof not on proc");
        }
      }
      break;
    }
    // initial mixture-fraction profile for Rayleigh-Taylor instability
    case Inpar::ScaTra::initfield_raytaymixfrac:
    {
      // define interface thickness, sinusoidal disturbance wave amplitude and pi
      const double delta = 0.002;
      const double alpha = 0.001;

      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

      for (auto lnode : discret_->my_row_node_range())
      {
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->dof(0, lnode);

        // get x1- and x2-coordinate
        const double x1 = lnode.x()[0];
        const double x2 = lnode.x()[1];

        // interface disturbance
        double x2_int = 0.0;
        x2_int -= std::cos(4 * std::numbers::pi * x1);
        x2_int -= std::cos(14 * std::numbers::pi * x1);
        x2_int -= std::cos(23 * std::numbers::pi * x1);
        x2_int -= std::cos(28 * std::numbers::pi * x1);
        x2_int -= std::cos(33 * std::numbers::pi * x1);
        x2_int -= std::cos(42 * std::numbers::pi * x1);
        x2_int -= std::cos(51 * std::numbers::pi * x1);
        x2_int -= std::cos(59 * std::numbers::pi * x1);
        x2_int *= alpha;

        const double value = (x2_int - x2) / (2.0 * delta);

        // values required for tanh-distribution
        const double vp = exp(value);
        const double vm = exp(-value);

        int numdofs = static_cast<int>(nodedofset.size());
        for (int k = 0; k < numdofs; ++k)
        {
          const int dofgid = nodedofset[k];
          int doflid = dofrowmap->lid(dofgid);

          // compute tanh-distribution
          double initialval = 0.0;
          initialval = 0.5 * (1.0 + (vp - vm) / (vp + vm));

          int err = 0;
          err += phin_->replace_local_value(doflid, initialval);
          // initialize also the solution vector. These values are a pretty good guess for the
          // solution after the first time step (much better than starting with a zero vector)
          err += phinp_->replace_local_value(doflid, initialval);
          if (err != 0) FOUR_C_THROW("dof not on proc");
        }
      }
      break;
    }
    // initial field for skew convection of L-shaped domain
    case Inpar::ScaTra::initfield_Lshapeddomain:
    {
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

      for (auto lnode : discret_->my_row_node_range())
      {
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->dof(0, lnode);

        // get x1- and x2-coordinate
        const double x1 = lnode.x()[0];
        const double x2 = lnode.x()[1];

        int numdofs = static_cast<int>(nodedofset.size());
        for (int k = 0; k < numdofs; ++k)
        {
          const int dofgid = nodedofset[k];
          int doflid = dofrowmap->lid(dofgid);

          // compute initial values 0.0 or 1.0 depending on geometrical location
          double initialval = 0.0;
          if ((x1 <= 0.25 and x2 <= 0.5) or (x1 <= 0.5 and x2 <= 0.25)) initialval = 1.0;

          int err = 0;
          err += phin_->replace_local_value(doflid, initialval);
          // initialize also the solution vector. These values are a pretty good
          // guess for the solution after the first time step (much better than
          // starting with a zero vector)
          err += phinp_->replace_local_value(doflid, initialval);
          if (err != 0) FOUR_C_THROW("dof not on proc");
        }
      }
      break;
    }
    case Inpar::ScaTra::initfield_facing_flame_fronts:
    {
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

      for (auto lnode : discret_->my_row_node_range())
      {
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->dof(0, lnode);

        // get x1- and x2-coordinate
        const double x1 = lnode.x()[0];

        int numdofs = static_cast<int>(nodedofset.size());
        for (int k = 0; k < numdofs; ++k)
        {
          const int dofgid = nodedofset[k];
          int doflid = dofrowmap->lid(dofgid);
          // evaluate component k of spatial function

          double initialval;
          if (x1 < 0.0)
            initialval = -(x1 + 0.75);
          else
            initialval = x1 - 0.75;

          int err = 0;
          err += phin_->replace_local_value(doflid, initialval);
          // initialize also the solution vector. These values are a pretty good guess for the
          // solution after the first time step (much better than starting with a zero vector)
          err += phinp_->replace_local_value(doflid, initialval);
          if (err != 0) FOUR_C_THROW("dof not on proc");
        }
      }
      break;
    }
    case Inpar::ScaTra::initfield_oracles_flame:
    {
      const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

      const double eps = 0.00152;

      for (auto lnode : discret_->my_row_node_range())
      {
        // the set of degrees of freedom associated with the node
        std::vector<int> nodedofset = discret_->dof(0, lnode);

        // get x2-coordinate
        const double x2 = lnode.x()[1];

        int numdofs = static_cast<int>(nodedofset.size());
        for (int k = 0; k < numdofs; ++k)
        {
          const int dofgid = nodedofset[k];
          int doflid = dofrowmap->lid(dofgid);
          // evaluate component k of spatial function

          double initval = 0.0;

          // initial plane implementation for periodic spanwise boundary
          if (x2 >= 0.0)
            initval = (x2 - 0.0354) - eps;
          else
            initval = (-0.0354 - x2) - eps;
          int err = 0;
          err += phin_->replace_local_value(doflid, initval);
          // initialize also the solution vector. These values are a pretty good guess for the
          // solution after the first time step (much better than starting with a zero vector)
          err += phinp_->replace_local_value(doflid, initval);
          if (err != 0) FOUR_C_THROW("dof not on proc");
        }
      }
      break;
    }
    case Inpar::ScaTra::initialfield_forced_hit_high_Sc:
    case Inpar::ScaTra::initialfield_forced_hit_low_Sc:
    {
      // initialize calculation of initial field based on fast Fourier transformation
      HomoIsoTurbInitialScalarField HitInitialScalarField(*this, init);
      // calculate initial field
      HitInitialScalarField.calculate_initial_field();

      break;
    }
    default:
      FOUR_C_THROW("Unknown option for initial field: {}", init);
      break;
  }  // switch(init)
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::update_iter(const Core::LinAlg::Vector<double>& inc)
{
  // store incremental vector to be available for convergence check
  // if incremental vector is received from outside for coupled problem
  increment_->update(1.0, inc, 0.0);

  // update scalar values by adding increments
  phinp_->update(1.0, inc, 1.0);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::setup_krylov_space_projection(
    const Core::Conditions::Condition* kspcond)
{
  // previously, scatra was able to define actual modes that formed a
  // nullspace. factors when assigned to scalars in the input file. it could
  // take several scatra Krylov conditions each forming one mode like:
  // 4.0*c_1 + 2.0*c_2 = const
  // since this was never used, not even in ELCH-problems, and for the sake of
  // consistency, now only a single scatra Krylov condition can be given, with
  // flags that triggers the scalars that are to be levelled by a projection
  // (like the pressure in a pure Dirichlet fluid problem).
  // furthermore, this is a step towards the ability to have projection on
  // more than one field.
  // to see the handling of different modes, the ability that is now lost, see
  // revision 17615.

  // confirm that mode flags are number of nodal dofs/scalars
  const int nummodes = kspcond->parameters().get<int>("NUMMODES");
  if (nummodes != num_dof_per_node())
  {
    FOUR_C_THROW(
        "Expecting as many mode flags as nodal dofs in Krylov projection definition. Check "
        "input file!");
  }

  // get vector of mode flags as given in input file
  const auto* modeflags = &kspcond->parameters().get<std::vector<int>>("ONOFF");

  // count actual active modes selected in input file
  std::vector<int> activemodeids;
  for (int rr = 0; rr < num_dof_per_node(); ++rr)
  {
    if (((*modeflags)[rr]) != 0)
    {
      activemodeids.push_back(rr);
    }
  }

  // get from input file definition how weights are to be computed
  const auto* weighttype = &kspcond->parameters().get<std::string>("WEIGHTVECDEF");

  // set flag for projection update true only if ALE and integral weights
  if (isale_ and (*weighttype == "integration")) updateprojection_ = true;

  // create the projector
  projector_ = std::make_shared<Core::LinAlg::KrylovProjector>(
      activemodeids, weighttype, discret_->dof_row_map());

  // update the projector
  update_krylov_space_projection();
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::update_krylov_space_projection()
{
  // loop over modes to create vectors within multi-vector
  // one could tailor a MapExtractor to extract necessary dofs, however this
  // seems to be an overkill, since normally only a single scalar is
  // projected. for only a second projected scalar it seems worthwhile. feel
  // free! :)

  // get std::shared_ptr to kernel vector of projector
  std::shared_ptr<Core::LinAlg::MultiVector<double>> c = projector_->get_non_const_kernel();
  c->put_scalar(0.0);

  const std::string* weighttype = projector_->weight_type();
  // compute w_ as defined in input file
  if (*weighttype == "pointvalues")
  {
    FOUR_C_THROW("option pointvalues not implemented");
  }
  else if (*weighttype == "integration")
  {
    // get std::shared_ptr to weight vector of projector
    std::shared_ptr<Core::LinAlg::MultiVector<double>> w = projector_->get_non_const_weights();
    w->put_scalar(0.0);

    // get number of modes and their ids
    int nummodes = projector_->nsdim();
    std::vector<int> modeids = projector_->modes();

    // initialize dofid vector to -1
    std::shared_ptr<Core::LinAlg::IntSerialDenseVector> dofids =
        std::make_shared<Core::LinAlg::IntSerialDenseVector>(num_dof_per_node());
    for (int rr = 0; rr < num_dof_per_node(); ++rr)
    {
      (*dofids)[rr] = -1;
    }

    Teuchos::ParameterList mode_params;

    // set parameters for elements that do not change over mode
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
        "action", ScaTra::Action::integrate_shape_functions, mode_params);

    // loop over all activemodes
    for (int imode = 0; imode < nummodes; ++imode)
    {
      // activate dof of current mode and add dofids to parameter list
      (*dofids)[modeids[imode]] = 1;
      mode_params.set("dofids", dofids);

      /*
      // evaluate KrylovSpaceProjection condition in order to get
      // integrated nodal basis functions w_
      // Note that in the case of definition integration based, the average
      // increment of the scalar quantity c will vanish in an integral sense
      //
      //                    /              /                      /
      //   /    \          |              |  /          \        |  /    \
      //  | w_*c | = c_i * | N_i(x) dx =  | | N_i(x)*c_i | dx =  | | c(x) | dx = 0
      //   \    /          |              |  \          /        |  \    /
      //                   /              /                      /
      */

      // get an std::shared_ptr of the current column Core::LinAlg::Vector<double> of the
      // MultiVector
      auto wi = std::make_shared<Core::LinAlg::Vector<double>>((*w)(imode));
      // compute integral of shape functions
      discret_->evaluate_condition(
          mode_params, nullptr, nullptr, wi, nullptr, nullptr, "KrylovSpaceProjection");
      (*w)(imode) = *wi;

      // deactivate dof of current mode
      (*dofids)[modeids[imode]] = -1;

      // set the current kernel basis vector - not very nice
      for (int inode = 0; inode < discret_->num_my_row_nodes(); inode++)
      {
        Core::Nodes::Node* node = discret_->l_row_node(inode);
        std::vector<int> gdof = discret_->dof(0, node);
        c->replace_global_value(gdof[modeids[imode]], imode, 1);
      }

    }  // loop over modes

    // adapt weight vector according to meshtying case
    if (msht_ != Inpar::FLUID::no_meshtying)
    {
      FOUR_C_THROW(
          "Since meshtying for scatra is not tested under Krylov projection FOUR_C_THROW is "
          "introduced. "
          "Remove at own responsibility.");
      // meshtying_->adapt_krylov_projector(w);
    }

  }  // endif integration
  else
  {
    FOUR_C_THROW("unknown definition of weight vector w for restriction of Krylov space");
  }

  // adapt kernel vector according to meshtying case
  if (msht_ != Inpar::FLUID::no_meshtying)
  {
    FOUR_C_THROW(
        "Since meshtying for scatra is not tested under Krylov projection FOUR_C_THROW is "
        "introduced. "
        "Remove at own responsibility.");
    // meshtying_->adapt_krylov_projector(c);
  }

  // fillcomplete the projector to compute (w^T c)^(-1)
  projector_->fill_complete();
}

/*----------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::create_meshtying_strategy()
{
  if (msht_ != Inpar::FLUID::no_meshtying)  // fluid meshtying
  {
    strategy_ = std::make_shared<MeshtyingStrategyFluid>(this);
  }
  else if (s2i_meshtying())  // scatra-scatra interface mesh tying
  {
    strategy_ = std::make_shared<MeshtyingStrategyS2I>(this, *params_);
  }
  else if (heteroreaccoupling_)  // scatra-scatra interface coupling
  {
    strategy_ = std::make_shared<HeterogeneousReactionStrategy>(this);
  }
  else if (arterycoupling_)
  {
    strategy_ = std::make_shared<MeshtyingStrategyArtery>(this);
  }
  else  // standard case without meshtying
  {
    strategy_ = std::make_shared<MeshtyingStrategyStd>(this);
  }
}  // ScaTraTimIntImpl::create_meshtying_strategy

/*----------------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::create_scalar_handler()
{
  scalarhandler_ = std::make_shared<ScalarHandler>();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::apply_dirichlet_to_system()
{
  // -------------------------------------------------------------------
  // Apply Dirichlet boundary conditions to system matrix
  // -------------------------------------------------------------------
  if (incremental_)
  {
    // blank residual DOFs which are on Dirichlet BC
    // We can do this because the values at the Dirichlet positions
    // are not used anyway.
    // We could avoid this though, if the dofrowmap would not include
    // the Dirichlet values as well. But it is expensive to avoid that.
    dbcmaps_->insert_cond_vector(*dbcmaps_->extract_cond_vector(*zeros_), *residual_);

    //--------- Apply Dirichlet boundary conditions to system of equations
    // residual values are supposed to be zero at Dirichlet boundaries
    increment_->put_scalar(0.0);

    {
      // time measurement: application of DBC to system
      TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + apply DBC to system");

      Core::LinAlg::apply_dirichlet_to_system(
          *sysmat_, *increment_, *residual_, *zeros_, *(dbcmaps_->cond_map()));
    }
  }
  else
  {
    // time measurement: application of DBC to system
    TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + apply DBC to system");

    Core::LinAlg::apply_dirichlet_to_system(
        *sysmat_, *phinp_, *residual_, *phinp_, *(dbcmaps_->cond_map()));
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::apply_dirichlet_bc(const double time,
    std::shared_ptr<Core::LinAlg::Vector<double>> phinp,
    std::shared_ptr<Core::LinAlg::Vector<double>> phidt)
{
  // time measurement: apply Dirichlet conditions
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:      + apply dirich cond.");

  // Todo: what happens in  the case of generalized alpha
  // needed parameters
  Teuchos::ParameterList p;
  p.set("total time", time);  // actual time t_{n+1}
  p.set<const Core::Utils::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());
  const Core::ProblemType problem_type = Core::ProblemType::scatra;
  p.set<const Core::ProblemType*>("problem_type", &problem_type);

  // predicted Dirichlet values
  // \c  phinp then also holds prescribed new Dirichlet values
  discret_->evaluate_dirichlet(p, phinp, phidt, nullptr, nullptr, dbcmaps_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::scaling_and_neumann()
{
  // scaling to get true residual vector for all time integration schemes
  // in incremental case: boundary flux values can be computed from trueresidual
  if (incremental_) trueresidual_->update(residual_scaling(), *residual_, 0.0);

  // add Neumann b.c. scaled with a factor due to time discretization
  add_neumann_to_residual();

  // add potential Neumann inflow or convective heat transfer boundary
  // conditions (simultaneous evaluation of both conditions not allowed!)
  if (neumanninflow_)
    compute_neumann_inflow(sysmat_, residual_);
  else if (convheatrans_)
    evaluate_convective_heat_transfer(sysmat_, residual_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::apply_neumann_bc(
    const std::shared_ptr<Core::LinAlg::Vector<double>>& neumann_loads)
{
  // prepare load vector
  neumann_loads->put_scalar(0.0);

  // create parameter list
  Teuchos::ParameterList condparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_Neumann, condparams);

  condparams.set<const Core::Utils::FunctionManager*>(
      "function_manager", &Global::Problem::instance()->function_manager());

  // specific parameters
  add_problem_specific_parameters_and_vectors(condparams);

  // set time for evaluation of point Neumann conditions as parameter depending on time integration
  // scheme line/surface/volume Neumann conditions use the time stored in the time parameter class
  set_time_for_neumann_evaluation(condparams);

  // evaluate Neumann boundary conditions at time t_{n+alpha_F} (generalized alpha) or time t_{n+1}
  // (otherwise)
  discret_->evaluate_neumann(condparams, *neumann_loads);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::evaluate_solution_depending_conditions(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
  // evaluate Robin type boundary condition
  evaluate_robin_boundary_conditions(systemmatrix, rhs);

  // evaluate meshtying
  // this needs to be done as final step for consistency
  strategy_->evaluate_meshtying();

  //----------------------------------------------------------------------
  // apply contact terms...
  // account for partitioning algorithm. The dofs in contact discretization must be frozen
  // before calling this function
  //----------------------------------------------------------------------
  if (contact_strategy_nitsche_ != nullptr)
  {
    const auto fint_scatra =
        contact_strategy_nitsche_->get_rhs_block_ptr(CONTACT::VecBlockType::scatra);
    if (residual_->update(1.0, *fint_scatra, 1.0)) FOUR_C_THROW("update failed");
  }

  // evaluate macro-micro coupling on micro scale in multi-scale scalar transport problems
  if (micro_scale_) evaluate_macro_micro_coupling();
  strategy_->evaluate_point_coupling();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int ScaTra::ScaTraTimIntImpl::get_max_dof_set_number() const
{
  return std::max({nds_disp_, nds_growth_, nds_micro_, nds_pres_, nds_scatra_, nds_thermo_,
      nds_two_tensor_quantity_, nds_vel_, nds_wss_});
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::evaluate_additional_solution_depending_models(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix,
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
  // evaluate solution depending additional models
  // this point is unequal nullptr only if a scatra
  // adapter has been constructed.
  if (additional_model_evaluator_ != nullptr)
    additional_model_evaluator_->evaluate_additional_solution_depending_models(systemmatrix, rhs);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::evaluate_robin_boundary_conditions(
    std::shared_ptr<Core::LinAlg::SparseOperator> matrix,
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs)
{
  // create parameter list
  Teuchos::ParameterList condparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_Robin, condparams);

  // add element parameters and set state vectors according to time-integration scheme
  add_time_integration_specific_vectors();

  // evaluate ElchBoundaryKinetics conditions at time t_{n+1} or t_{n+alpha_F}
  discret_->evaluate_condition(
      condparams, matrix, nullptr, rhs, nullptr, nullptr, "TransportRobin");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::assemble_mat_and_rhs()
{
  // safety check
  check_is_init();
  check_is_setup();

  // time measurement: element calls
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + element calls");

  // get cpu time
  const double tcpuele = Teuchos::Time::wallTime();

  // zero out matrix entries
  sysmat_->zero();

  // reset the residual vector
  residual_->put_scalar(0.0);

  // create parameter list for elements
  Teuchos::ParameterList eleparams;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_mat_and_rhs, eleparams);

  // DO THIS AT VERY FIRST!!!
  // compute reconstructed diffusive fluxes for better consistency
  const auto consistency = Teuchos::getIntegralValue<Inpar::ScaTra::Consistency>(
      params_->sublist("STABILIZATION"), "CONSISTENCY");
  if (consistency == Inpar::ScaTra::consistency_l2_projection_lumped)
  {
    // compute flux approximation and add it to the parameter list
    add_flux_approx_to_parameter_list(eleparams);
  }

  // prepare dynamic Smagorinsky model if required,
  // i.e. calculate turbulent Prandtl number
  if (timealgo_ != Inpar::ScaTra::timeint_stationary)
  {
    dynamic_computation_of_cs();
    dynamic_computation_of_cv();
  }
  // this parameter list is required here to get the element-based filtered constants
  eleparams.sublist("TURBULENCE MODEL") = extraparams_->sublist("TURBULENCE MODEL");

  // AVM3 separation for incremental solver: get fine-scale part of scalar
  if (incremental_ and step_ > 0 and
      (fssgd_ != Inpar::ScaTra::fssugrdiff_no or
          turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales))
    avm3_separation();

  // add state vectors according to time-integration scheme
  add_time_integration_specific_vectors();

  // set external volume force (required, e.g., for forced homogeneous isotropic turbulence)
  if (homisoturb_forcing_ != nullptr) homisoturb_forcing_->update_forcing(step_);

  if (forcing_ != nullptr) discret_->set_state("forcing", *forcing_);

  // add problem specific time-integration parameters
  add_problem_specific_parameters_and_vectors(eleparams);

  // call loop over elements (with or without subgrid-diffusivity(-scaling) vector)
  if (fssgd_ != Inpar::ScaTra::fssugrdiff_no)
    discret_->evaluate(eleparams, sysmat_, nullptr, residual_, subgrdiff_, nullptr);
  else
    discret_->evaluate(eleparams, sysmat_, residual_);

  //----------------------------------------------------------------------
  // apply weak Dirichlet boundary conditions
  //----------------------------------------------------------------------
  {
    Teuchos::ParameterList mhdbcparams;

    // set action for elements
    Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
        "action", ScaTra::BoundaryAction::calc_weak_Dirichlet, mhdbcparams);

    add_time_integration_specific_vectors();

    // evaluate all mixed hybrid Dirichlet boundary conditions
    discret_->evaluate_condition(
        mhdbcparams, sysmat_, nullptr, residual_, nullptr, nullptr, "LineWeakDirichlet");

    discret_->evaluate_condition(
        mhdbcparams, sysmat_, nullptr, residual_, nullptr, nullptr, "SurfaceWeakDirichlet");
  }

  // AVM3 scaling for non-incremental solver: scaling of normalized AVM3-based
  // fine-scale subgrid-diffusivity matrix by subgrid diffusivity
  if (not incremental_ and fssgd_ != Inpar::ScaTra::fssugrdiff_no) avm3_scaling(eleparams);

  // potential residual scaling and potential addition of Neumann terms
  scaling_and_neumann();  // TODO: do we have to call this function twice??

  // evaluate solution-depending additional models
  evaluate_additional_solution_depending_models(sysmat_, residual_);

  // evaluate solution-depending boundary and interface conditions
  evaluate_solution_depending_conditions(sysmat_, residual_);

  // finalize assembly of system matrix
  sysmat_->complete();

  // end time measurement for element and take average over all processors via communication
  double mydtele = Teuchos::Time::wallTime() - tcpuele;
  dtele_ = Core::Communication::max_all(mydtele, discret_->get_comm());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::linear_solve()
{
  // safety checks
  check_is_init();
  check_is_setup();

  // output to screen
  print_time_step_info();

  // clear state vectors
  discret_->clear_state();

  // preparations for solve
  prepare_linear_solve();

  // Solve system in incremental or non-incremental case
  if (incremental_)
  {
    TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + solver calls");

    // get cpu time
    const double tcpusolve = Teuchos::Time::wallTime();

    Core::LinAlg::SolverParams solver_params;

    strategy_->solve(solver_, sysmat_, increment_, residual_, phinp_, 1, solver_params);

    // end time measurement for solver
    dtsolve_ = Teuchos::Time::wallTime() - tcpusolve;

    //------------------------------------------------ update solution vector
    update_iter(*increment_);

    //--------------------------------------------- compute norm of increment
    double incnorm_L2(0.0);
    double scalnorm_L2(0.0);
    increment_->norm_2(&incnorm_L2);
    phinp_->norm_2(&scalnorm_L2);

    if (myrank_ == 0)
    {
      printf("+-------------------------------+-------------+\n");
      {
        if (scalnorm_L2 > 1e-10)
          printf("|  relative increment (L2 norm) | %10.3E  |", incnorm_L2 / scalnorm_L2);
        else  // prevent division by an almost zero value
          printf("|  absolute increment (L2 norm) | %10.3E  |\n", incnorm_L2);
      }
      printf(" (ts=%10.3E,te=%10.3E)\n", dtsolve_, dtele_);
      printf("+-------------------------------+-------------+\n");
    }
  }
  else
  {
    TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + solver calls");

    // get cpu time
    const double tcpusolve = Teuchos::Time::wallTime();

    Core::LinAlg::SolverParams solver_params;

    strategy_->solve(solver_, sysmat_, phinp_, residual_, phinp_, 1, solver_params);

    // end time measurement for solver
    dtsolve_ = Teuchos::Time::wallTime() - tcpusolve;

    if (myrank_ == 0) printf("Solvertype linear_full (ts=%10.3E,te=%10.3E)\n", dtsolve_, dtele_);
  }

  // -------------------------------------------------------------------
  // compute values at intermediate time steps (only for gen.-alpha)
  // -------------------------------------------------------------------
  compute_intermediate_values();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::nonlinear_solve()
{
  // safety checks
  check_is_init();
  check_is_setup();

  // time measurement: nonlinear iteration
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:   + nonlin. iteration/lin. solve");

  // out to screen
  print_time_step_info();

  // special preparations for multifractal subgrid-scale model
  if (turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales) recompute_mean_csgs_b();

  //------------------------------ turn adaptive solver tolerance on/off
  const double ittol = params_->sublist("NONLINEAR").get<double>("CONVTOL");
  const bool isadapttol = params_->sublist("NONLINEAR").get<bool>("ADAPTCONV");
  const double adaptolbetter = params_->sublist("NONLINEAR").get<double>("ADAPTCONV_BETTER");
  double actresidual(0.0);

  // prepare Newton-Raphson iteration
  iternum_ = 0;

  // perform explicit predictor step (-> better starting point for nonlinear solver)
  const bool explpredictor = (params_->sublist("NONLINEAR").get<bool>("EXPLPREDICT"));
  if (explpredictor)
  {
    // explicit predictor + recovery of DBC values
    auto phinp_dirich = dbcmaps_->extract_cond_vector(*phinp_);
    explicit_predictor();
    dbcmaps_->insert_cond_vector(*phinp_dirich, *phinp_);
  }

  // start Newton-Raphson iteration
  while (true)
  {
    // clear states
    discret_->clear_state();

    iternum_++;

    // call elements to calculate system matrix and rhs and assemble
    assemble_mat_and_rhs();

    // perform finite difference check on time integrator level
    if (fdcheck_ == Inpar::ScaTra::fdcheck_global) fd_check();

    // project residual such that only part orthogonal to nullspace is considered
    if (projector_ != nullptr) *residual_ = projector_->to_reduced(*residual_);

    // Apply Dirichlet boundary conditions to system of equations
    // residual values are supposed to be zero at Dirichlet boundaries
    {
      // time measurement: application of DBC to system
      TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + apply DBC to system");

      Core::LinAlg::apply_dirichlet_to_system(
          *sysmat_, *increment_, *residual_, *zeros_, *(dbcmaps_->cond_map()));
    }

    // abort nonlinear iteration if desired
    if (strategy_->abort_nonlin_iter(*this, actresidual)) break;

    // initialize increment vector
    increment_->put_scalar(0.0);

    {
      // get cpu time
      const double tcpusolve = Teuchos::Time::wallTime();

      // time measurement: call linear solver
      TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + call linear solver");

      // do adaptive linear solver tolerance (not in first solve)
      Core::LinAlg::SolverParams solver_params;
      if (isadapttol && iternum_ > 1)
      {
        solver_params.nonlin_tolerance = ittol;
        solver_params.nonlin_residual = actresidual;
        solver_params.lin_tol_better = adaptolbetter;
      }

      // reprepare Krylov projection only if ale and projection required
      if (updateprojection_) update_krylov_space_projection();

      solver_params.projector = projector_;

      strategy_->solve(solver_, sysmat_, increment_, residual_, phinp_, iternum_, solver_params);

      solver_->reset_tolerance();

      // end time measurement for solver and take average over all processors via communication
      double mydtsolve = Teuchos::Time::wallTime() - tcpusolve;
      dtsolve_ = Core::Communication::max_all(mydtsolve, discret_->get_comm());

      // output performance statistics associated with linear solver into text file if applicable
      if (params_->get<bool>("OUTPUTLINSOLVERSTATS"))
        output_lin_solver_stats(strategy_->solver(), dtsolve_, step(), iternum_,
            strategy_->dof_row_map().num_global_elements());
    }

    //------------------------------------------------ update solution vector
    phinp_->update(1.0, *increment_, 1.0);

    //-------- update values at intermediate time steps (only for gen.-alpha)
    compute_intermediate_values();

    // compute values at the interior of the elements (required for hdg)
    compute_interior_values();

    compute_time_derivative();
  }  // nonlinear iteration

  // calculate mean concentration of micro discretization and set state to nds_micro_
  if (macro_scale_ and nds_micro() != -1) calc_mean_micro_concentration();
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::nonlinear_multi_scale_solve()
{
  // reset number of outer iterations
  iternum_outer_ = 0;

  // initialize relaxed macro-scale state vector
  std::shared_ptr<Core::LinAlg::Vector<double>> phinp_relaxed =
      std::make_shared<Core::LinAlg::Vector<double>>(*phinp_);

  // begin outer iteration loop
  while (true)
  {
    // increment iteration number
    iternum_outer_++;

    // store current state vector on macro scale
    phinp_inc_->update(1., *phinp_relaxed, 0.);

    // solve micro scale first and macro scale second
    if (solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro or
        solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken or
        solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken_dofsplit)
    {
      // backup macro-scale state vector
      const std::shared_ptr<Core::LinAlg::Vector<double>> phinp = phinp_;

      // replace macro-scale state vector by relaxed macro-scale state vector as input for micro
      // scale
      phinp_ = phinp_relaxed;

      // solve micro-scale problems
      nonlinear_micro_scale_solve();

      // undo state vector replacement
      phinp_ = phinp;

      // solve macro-scale problem
      nonlinear_solve();
    }

    // solve macro scale first and micro scale second
    else if (solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_microtomacro)
    {
      // solve macro-scale problem
      nonlinear_solve();

      // solve micro-scale problems
      nonlinear_micro_scale_solve();
    }

    // compute increment of macro-scale state vector
    phinp_inc_->update(1., *phinp_, -1.);

    // convergence check
    if (strategy_->abort_outer_iter(*this)) break;

    if (solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken or
        solvtype_ == Inpar::ScaTra::solvertype_nonlinear_multiscale_macrotomicro_aitken_dofsplit)
    {
      // compute difference between current and previous increments of macro-scale state vector
      Core::LinAlg::Vector<double> phinp_inc_diff(*phinp_inc_);
      phinp_inc_diff.update(-1., *phinp_inc_old_, 1.);

      // perform Aitken relaxation
      perform_aitken_relaxation(*phinp_relaxed, phinp_inc_diff);

      // update increment of macro-scale state vector
      phinp_inc_old_->update(1., *phinp_inc_, 0.);
    }

    else
      // no relaxation
      phinp_relaxed = phinp_;
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::nonlinear_micro_scale_solve()
{
  // initialize parameter list for evaluation of macro-scale elements
  Teuchos::ParameterList eleparams;

  // set action for macro-scale elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::micro_scale_solve, eleparams);

  // clear state vectors
  discret_->clear_state();

  // set state vectors
  add_time_integration_specific_vectors();

  // evaluate macro-scale elements
  discret_->evaluate(eleparams, nullptr, nullptr, nullptr, nullptr, nullptr);
}

/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
std::string ScaTra::ScaTraTimIntImpl::map_tim_int_enum_to_string(
    const enum Inpar::ScaTra::TimeIntegrationScheme term)
{
  // length of return std::string is 14 due to usage in formatted screen output
  switch (term)
  {
    case Inpar::ScaTra::timeint_one_step_theta:
      return "One-Step-Theta";
      break;
    case Inpar::ScaTra::timeint_bdf2:
      return "     BDF2     ";
      break;
    case Inpar::ScaTra::timeint_stationary:
      return "  Stationary  ";
      break;
    case Inpar::ScaTra::timeint_gen_alpha:
      return "  Gen. Alpha  ";
      break;
    default:
      FOUR_C_THROW("Cannot cope with name enum {}", term);
      return "";
      break;
  }

  return "";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
inline void ScaTra::ScaTraTimIntImpl::increment_time_and_step()
{
  step_ += 1;
  time_ += dta_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::adapt_time_step_size()
{
  timestepadapted_ = false;

  // check flag for adaptive time stepping
  if (params_->get<bool>("ADAPTIVE_TIMESTEPPING"))
  {
    // initialize time step size with original value
    double dt(params_->get<double>("TIMESTEP"));

    // reduce time step size if necessary
    compute_time_step_size(dt);

    // adapt time step size if necessary
    if (dt < dta_ or dt > dta_)
    {
      // print information about adaptation of time step size to screen
      if (myrank_ == 0)
      {
        std::cout << std::scientific << std::setprecision(2) << std::endl;
        std::cout << "ADAPTIVE TIME STEPPING:" << std::endl;
        std::cout << "Time step size is " << (dt < dta_ ? "decreased" : "increased") << " from "
                  << dta_ << " to " << dt << "!" << std::endl;
      }

      // adapt time step size
      set_dt(dt);

      // time step was adapted
      timestepadapted_ = true;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::compute_time_step_size(double& dt)
{
  strategy_->compute_time_step_size(dt);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::add_dirich_cond(
    const std::shared_ptr<const Core::LinAlg::Map> maptoadd)
{
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> condmaps;
  condmaps.push_back(maptoadd);
  condmaps.push_back(dbcmaps_->cond_map());
  std::shared_ptr<Core::LinAlg::Map> condmerged =
      Core::LinAlg::MultiMapExtractor::merge_maps(condmaps);
  *dbcmaps_ = Core::LinAlg::MapExtractor(*(discret_->dof_row_map()), condmerged);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::add_time_integration_specific_vectors(bool forcedincrementalsolver)
{
  // add global state vectors associated with meshtying strategy
  strategy_->add_time_integration_specific_vectors();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::remove_dirich_cond(
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
std::shared_ptr<const Core::LinAlg::Map> ScaTra::ScaTraTimIntImpl::dof_row_map()
{
  return dof_row_map(0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> ScaTra::ScaTraTimIntImpl::dof_row_map(int nds)
{
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map(nds);
  return Core::Utils::shared_ptr_from_ref(*dofrowmap);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int ScaTra::ScaTraTimIntImpl::max_num_dof_per_node() const
{
  FOUR_C_ASSERT_ALWAYS(scalarhandler_ != nullptr, "scalar handler was not initialized!");
  return scalarhandler_->max_num_dof_per_node();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int ScaTra::ScaTraTimIntImpl::num_scal() const
{
  if (scalarhandler_ == nullptr) FOUR_C_THROW("scalar handler was not initialized!");
  return scalarhandler_->num_scal();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int ScaTra::ScaTraTimIntImpl::num_dof_per_node() const
{
  if (scalarhandler_ == nullptr) FOUR_C_THROW("scalar handler was not initialized!");
  return scalarhandler_->num_dof_per_node();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int ScaTra::ScaTraTimIntImpl::num_dof_per_node_in_condition(
    const Core::Conditions::Condition& condition) const
{
  if (scalarhandler_ == nullptr) FOUR_C_THROW("scalar handler was not initialized!");
  return scalarhandler_->num_dof_per_node_in_condition(condition, *discret_);
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
const std::map<const int, std::vector<double>>& ScaTra::ScaTraTimIntImpl::total_scalars() const
{
  if (outputscalarstrategy_ == nullptr) FOUR_C_THROW("output strategy was not initialized!");

  return outputscalarstrategy_->total_scalars();
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
const std::map<const int, std::vector<double>>& ScaTra::ScaTraTimIntImpl::mean_scalars() const
{
  if (outputscalarstrategy_ == nullptr) FOUR_C_THROW("output strategy was not initialized!");

  return outputscalarstrategy_->mean_scalars();
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
const std::vector<double>& ScaTra::ScaTraTimIntImpl::domain_integrals() const
{
  if (outputdomainintegralstrategy_ == nullptr)
    FOUR_C_THROW("output strategy for domain integration was not initialized!");

  return outputdomainintegralstrategy_->domain_integrals();
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
const std::vector<double>& ScaTra::ScaTraTimIntImpl::boundary_integrals() const
{
  if (outputdomainintegralstrategy_ == nullptr)
    FOUR_C_THROW("output strategy for domain integration was not initialized!");

  return outputdomainintegralstrategy_->boundary_integrals();
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::evaluate_macro_micro_coupling()
{
  // extract multi-scale coupling conditions
  std::vector<const Core::Conditions::Condition*> conditions;
  discret_->get_condition("ScatraMultiScaleCoupling", conditions);

  // loop over conditions
  for (auto& condition : conditions)
  {
    // extract nodal cloud
    const std::vector<int>* const nodeids = condition->get_nodes();
    if (nodeids == nullptr)
      FOUR_C_THROW("Multi-scale coupling condition does not have nodal cloud!");

    // loop over all nodes in nodal cloud
    for (int inode : *nodeids)
    {
      // process row nodes only
      if (discret_->node_row_map()->my_gid(inode))
      {
        // extract node
        Core::Nodes::Node* node = discret_->g_node(inode);
        if (node == nullptr)
          FOUR_C_THROW(
              "Cannot extract node with global ID {} from micro-scale discretization!", inode);

        // safety check
        if (node->num_element() != 1)
          FOUR_C_THROW("Number of 1D elements adjacent to the boundary node must be 1!");

        // compute domain integration factor
        constexpr double four_pi = 4.0 * std::numbers::pi;
        const double fac = params_->get<bool>("SPHERICALCOORDS")
                               ? *node->x().data() * *node->x().data() * four_pi
                               : 1.0;

        // extract degrees of freedom from node
        const std::vector<int> dofs = discret_->dof(0, node);

        // loop over all degrees of freedom
        for (int gid : dofs)
        {
          // extract global and local IDs of degree of freedom
          const int lid = discret_->dof_row_map()->lid(gid);
          if (lid < 0) FOUR_C_THROW("Cannot extract degree of freedom with global ID {}!", gid);

          // compute matrix and vector contributions according to kinetic model for current
          // macro-micro coupling condition
          const int kinetic_model =
              condition->parameters().get<Inpar::S2I::KineticModels>("KINETIC_MODEL");

          switch (kinetic_model)
          {
            case Inpar::S2I::kinetics_constperm:
            {
              // access real vector of constant permeabilities
              const std::vector<double>* permeabilities =
                  condition->parameters().get_if<std::vector<double>>("PERMEABILITIES");
              if (permeabilities == nullptr)
                FOUR_C_THROW("Cannot access vector of permeabilities for macro-micro coupling!");
              if (permeabilities->size() != (unsigned)num_scal())
                FOUR_C_THROW("Number of permeabilities does not match number of scalars!");

              // compute and store micro-scale coupling flux
              q_ = (*permeabilities)[0] * ((*phinp_)[lid] - phinp_macro_[0]);

              // compute and store derivative of micro-scale coupling flux w.r.t. macro-scale state
              // variable
              dq_dphi_[0] = -(*permeabilities)[0];

              // assemble contribution from macro-micro coupling into global residual vector
              (*residual_).get_values()[lid] -=
                  Discret::Elements::ScaTraEleParameterTimInt::instance(discret_->name())
                      ->time_fac_rhs() *
                  q_ * fac;

              // assemble contribution from macro-micro coupling into global system matrix
              sysmat_->assemble(
                  Discret::Elements::ScaTraEleParameterTimInt::instance(discret_->name())
                          ->time_fac() *
                      (*permeabilities)[0] * fac,
                  gid, gid);

              break;
            }

            case Inpar::S2I::kinetics_butlervolmer:
            case Inpar::S2I::kinetics_butlervolmerreduced:
            {
              // access material of electrode
              std::shared_ptr<const Mat::Electrode> matelectrode =
                  std::dynamic_pointer_cast<const Mat::Electrode>(
                      node->adjacent_elements()[0].user_element()->material());
              if (matelectrode == nullptr)
                FOUR_C_THROW("Invalid electrode material for multi-scale coupling!");

              // access input parameters associated with current condition
              const int nume = condition->parameters().get<int>("E-");
              if (nume != 1)
              {
                FOUR_C_THROW(
                    "Invalid number of electrons involved in charge transfer at "
                    "electrode-electrolyte interface!");
              }
              const std::vector<int>* stoichiometries =
                  condition->parameters().get_if<std::vector<int>>("STOICHIOMETRIES");
              if (stoichiometries == nullptr)
              {
                FOUR_C_THROW(
                    "Cannot access vector of stoichiometric coefficients for multi-scale "
                    "coupling!");
              }
              if (stoichiometries->size() != 1)
                FOUR_C_THROW(
                    "Number of stoichiometric coefficients does not match number of scalars!");
              if ((*stoichiometries)[0] != -1) FOUR_C_THROW("Invalid stoichiometric coefficient!");
              const double faraday =
                  Global::Problem::instance(0)->elch_control_params().get<double>(
                      "FARADAY_CONSTANT");
              const double gasconstant =
                  Global::Problem::instance(0)->elch_control_params().get<double>("GAS_CONSTANT");
              const double frt =
                  faraday /
                  (gasconstant * (Global::Problem::instance(0)->elch_control_params().get<double>(
                                     "TEMPERATURE")));
              const double alphaa =
                  condition->parameters().get<double>("ALPHA_A");  // anodic transfer coefficient
              const double alphac =
                  condition->parameters().get<double>("ALPHA_C");  // cathodic transfer coefficient
              const double kr = condition->parameters().get<double>(
                  "K_R");  // rate constant of charge transfer reaction
              if (kr < 0.) FOUR_C_THROW("Charge transfer constant k_r is negative!");

              // extract saturation value of intercalated lithium concentration from electrode
              // material
              const double cmax = matelectrode->c_max();
              if (cmax < 1.e-12)
                FOUR_C_THROW(
                    "Saturation value c_max of intercalated lithium concentration is too small!");

              // extract electrode-side and electrolyte-side concentration values at multi-scale
              // coupling point
              const double conc_ed = (*phinp_)[lid];
              const double conc_el = phinp_macro_[0];

              // evaluate overall integration factors
              const double timefacfac =
                  Discret::Elements::ScaTraEleParameterTimInt::instance(discret_->name())
                      ->time_fac() *
                  fac;
              const double timefacrhsfac =
                  Discret::Elements::ScaTraEleParameterTimInt::instance(discret_->name())
                      ->time_fac_rhs() *
                  fac;
              if (timefacfac < 0. or timefacrhsfac < 0.)
                FOUR_C_THROW("Integration factor is negative!");

              // no deformation available
              const double dummy_detF(1.0);

              // equilibrium electric potential difference and its derivative w.r.t. concentration
              // at electrode surface
              const double epd =
                  matelectrode->compute_open_circuit_potential(conc_ed, faraday, frt, dummy_detF);
              const double epdderiv =
                  matelectrode->compute_d_open_circuit_potential_d_concentration(
                      conc_ed, faraday, frt, dummy_detF);

              const double eta = phinp_macro_[2] - phinp_macro_[1] - epd;

              // Butler-Volmer exchange mass flux density
              const double j0 =
                  condition->parameters().get<Inpar::S2I::KineticModels>("KINETIC_MODEL") ==
                          Inpar::S2I::kinetics_butlervolmerreduced
                      ? kr
                      : kr * std::pow(conc_el, alphaa) * std::pow(cmax - conc_ed, alphaa) *
                            std::pow(conc_ed, alphac);

              // exponential Butler-Volmer terms
              const double expterm1 = std::exp(alphaa * frt * eta);
              const double expterm2 = std::exp(-alphac * frt * eta);
              const double expterm = expterm1 - expterm2;

              // core residual term associated with Butler-Volmer mass flux density
              q_ = j0 * expterm;

              const double dummyresistance(0.0);
              // define flux linearization terms
              double dj_dc_ed(0.0), dj_dc_el(0.0), dj_dpot_ed(0.0), dj_dpot_el(0.0);
              // calculate flux linearizations
              Discret::Elements::calculate_butler_volmer_elch_linearizations(kinetic_model, j0, frt,
                  epdderiv, alphaa, alphac, dummyresistance, expterm1, expterm2, kr, faraday,
                  conc_el, conc_ed, cmax, eta, dj_dc_ed, dj_dc_el, dj_dpot_ed, dj_dpot_el);

              dq_dphi_[0] = dj_dc_el;
              dq_dphi_[1] = dj_dpot_el;
              dq_dphi_[2] = dj_dpot_ed;

              // assemble contribution from macro-micro coupling into global residual vector
              (*residual_).get_values()[lid] -= timefacrhsfac * q_;

              // assemble contribution from macro-micro coupling into micro global system matrix
              sysmat_->assemble(timefacfac * dj_dc_ed, gid, gid);

              break;
            }
            case Inpar::S2I::kinetics_nointerfaceflux:
              break;

            default:
            {
              FOUR_C_THROW("Kinetic model for macro-micro coupling not yet implemented!");
              break;
            }
          }
        }
      }
    }
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::check_is_init() const
{
  if (not is_init()) FOUR_C_THROW("ScaTraTimIntImpl is not initialized. Call init() first.");
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::check_is_setup() const
{
  if (not is_setup()) FOUR_C_THROW("ScaTraTimIntImpl is not set up. Call setup() first.");
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::setup_matrix_block_maps()
{
  if (matrixtype_ == Core::LinAlg::MatrixType::block_condition or
      matrixtype_ == Core::LinAlg::MatrixType::block_condition_dof)
  {
    // extract domain partitioning conditions from discretization
    std::vector<const Core::Conditions::Condition*> partitioningconditions;
    discret_->get_condition("ScatraPartitioning", partitioningconditions);

    // safety check
    if (partitioningconditions.empty())
    {
      FOUR_C_THROW(
          "For block preconditioning based on domain partitioning, at least one associated "
          "condition needs to be specified in the input file!");
    }

    // build maps associated with blocks of global system matrix
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> dof_block_maps;
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> node_block_maps;
    build_block_maps(partitioningconditions, dof_block_maps, node_block_maps);

    // initialize a full map extractor associated with the degrees of freedom inside the blocks of
    // global system matrix
    dof_block_maps_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(
        *(discret_->dof_row_map()), dof_block_maps);
    // initialize a full map extractor associated with the nodes inside the blocks of global system
    // matrix
    node_block_maps_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(
        *discret_->node_row_map(), node_block_maps);

    // safety checks
    dof_block_maps_->check_for_valid_map_extractor();
    node_block_maps_->check_for_valid_map_extractor();
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::build_block_maps(
    const std::vector<const Core::Conditions::Condition*>& partitioning_conditions,
    std::vector<std::shared_ptr<const Core::LinAlg::Map>>& dof_block_maps,
    std::vector<std::shared_ptr<const Core::LinAlg::Map>>& node_block_maps) const
{
  if (matrixtype_ == Core::LinAlg::MatrixType::block_condition)
  {
    for (const auto& cond : partitioning_conditions)
    {
      // all dofs and nodes that form one block map
      std::vector<int> dof_gids;
      std::vector<int> node_gids;

      for (const int node_gid : *cond->get_nodes())
      {
        if (discret_->have_global_node(node_gid) and
            discret_->g_node(node_gid)->owner() ==
                Core::Communication::my_mpi_rank(discret_->get_comm()))
        {
          const std::vector<int> node_dofs = discret_->dof(0, discret_->g_node(node_gid));
          std::ranges::copy(node_dofs, std::inserter(dof_gids, dof_gids.end()));
          node_gids.push_back(node_gid);
        }
      }
#ifdef FOUR_C_ENABLE_ASSERTIONS
      std::unordered_set<int> dof_set(dof_gids.begin(), dof_gids.end());
      FOUR_C_ASSERT(dof_set.size() == dof_gids.size(), "The dofs are not unique");
      std::unordered_set<int> node_set(node_gids.begin(), node_gids.end());
      FOUR_C_ASSERT(node_set.size() == node_gids.size(), "The nodes are not unique");
#endif

      dof_block_maps.emplace_back(std::make_shared<Core::LinAlg::Map>(
          -1, static_cast<int>(dof_gids.size()), dof_gids.data(), 0, discret_->get_comm()));
      node_block_maps.emplace_back(std::make_shared<Core::LinAlg::Map>(
          -1, static_cast<int>(node_gids.size()), node_gids.data(), 0, discret_->get_comm()));
    }
  }
  else
  {
    FOUR_C_THROW("Invalid type of global system matrix!");
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::post_setup_matrix_block_maps() const
{
  // now build the null spaces
  build_block_null_spaces(*solver(), 0);

  // in case of an extended solver for scatra-scatra interface meshtying including interface growth
  // we need to equip it with the null space information generated above
  if (s2i_meshtying()) strategy_->equip_extended_solver_with_null_space_info();
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::build_block_null_spaces(
    const Core::LinAlg::Solver& solver, const int init_block_number) const
{
  // loop over blocks of global system matrix
  for (int iblock = init_block_number; iblock < dof_block_maps()->num_maps() + init_block_number;
      ++iblock)
  {
    // store number of current block starting from 1
    const int block_id = iblock + 1;

    Teuchos::ParameterList& block_smoother_parameters =
        solver.params().sublist("Inverse" + std::to_string(block_id));

    // Implementation for AMGnxn: needs to stay until dof split can be tackled differently
    if (solver.params().isSublist("AMGnxn Parameters"))
    {
      block_smoother_parameters.sublist("Belos Parameters");
      block_smoother_parameters.sublist("MueLu Parameters");

      // equip smoother for the current matrix block with null space associated with all degrees of
      // freedom on discretization
      Core::FE::compute_null_space_if_necessary(*discret_, block_smoother_parameters);
    }
    // Implementation for Teko
    else
    {
      if (solver.params().isSublist("MueLu Parameters"))
      {
        solver.params()
            .sublist("Inverse" + std::to_string(block_id))
            .set("MueLu Parameters", solver.params().sublist("MueLu Parameters"));
      }

      // add the null space map to extract relevant coordinates of the current block
      auto node_block_map =
          std::make_shared<Core::LinAlg::Map>(*node_block_maps()->map(iblock - init_block_number));
      auto dof_block_map =
          std::make_shared<Core::LinAlg::Map>(*dof_block_maps()->map(iblock - init_block_number));
      block_smoother_parameters.set<std::shared_ptr<Core::LinAlg::Map>>(
          "null space: node map", node_block_map);
      block_smoother_parameters.set<std::shared_ptr<Core::LinAlg::Map>>(
          "null space: dof map", dof_block_map);

      Core::LinearSolver::Parameters::compute_solver_parameters(
          *discret_, block_smoother_parameters);
    }

    // reduce full null space to match degrees of freedom associated with the current matrix block
    Core::LinearSolver::Parameters::fix_null_space("Block " + std::to_string(block_id),
        *discret_->dof_row_map(), *dof_block_maps()->map(iblock - init_block_number),
        block_smoother_parameters);
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::setup_matrix_block_maps_and_meshtying()
{
  switch (matrix_type())
  {
    // case Core::LinAlg::MatrixType::undefined:
    case Core::LinAlg::MatrixType::sparse:
    {
      // only set up the mesh tying in this case, as matrix has no block structure
      strategy_->setup_meshtying();

      break;
    }
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      // safety check
      const bool allowed_block_system_solvers = solver()->params().isSublist("AMGnxn Parameters") or
                                                solver()->params().isSublist("Teko Parameters") or
                                                solver()->params().isSublist("MueLu Parameters");
      FOUR_C_ASSERT_ALWAYS(allowed_block_system_solvers,
          "Global system matrix with block structure requires AMGnxn, MueLu or Teko block "
          "preconditioner!");

      // set up the matrix block maps
      setup_matrix_block_maps();

      // setup the meshtying
      strategy_->setup_meshtying();

      // do some post-setup matrix block map operations after the call to setup_meshtying, as they
      // rely on the fact that the interface maps have already been built
      post_setup_matrix_block_maps();

      break;
    }
    default:
    {
      FOUR_C_THROW("ScaTra Matrixtype {} not recognised", static_cast<int>(matrix_type()));
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseOperator> ScaTra::ScaTraTimIntImpl::init_system_matrix() const
{
  std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix(nullptr);

  switch (matrixtype_)
  {
    case Core::LinAlg::MatrixType::sparse:
    {
      // initialize system matrix
      systemmatrix =
          std::make_shared<Core::LinAlg::SparseMatrix>(*discret_->dof_row_map(), 27, false, true);
      break;
    }

    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      // initialize system matrix and associated strategy
      systemmatrix = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *dof_block_maps(), *dof_block_maps(), 81, false, true);

      break;
    }

    default:
    {
      FOUR_C_THROW(
          "Type of global system matrix for scatra-scatra interface coupling not recognized!");
    }
  }

  return systemmatrix;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::calc_mean_micro_concentration()
{
  phinp_micro_->put_scalar(0.0);

  if (nds_micro() < 0) FOUR_C_THROW("must set number of dofset for micro scale concentrations");

  discret_->set_state("phinp", *phinp_);

  Teuchos::ParameterList eleparams;

  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::calc_elch_elctrode_mean_concentration, eleparams);

  // evaluate nodal mean concentration of micro discretizations
  Core::FE::AssembleStrategy strategy(
      nds_micro(), nds_micro(), nullptr, nullptr, phinp_micro_, nullptr, nullptr);
  discret_->evaluate(eleparams, strategy);

  // copy states from first dof of MAT_Electrode
  for (int ele_lid = 0; ele_lid < discret_->element_row_map()->num_my_elements(); ++ele_lid)
  {
    const int ele_gid = discret_->element_row_map()->gid(ele_lid);
    auto* ele = discret_->g_element(ele_gid);

    if (ele->material()->material_type() != Core::Materials::m_electrode) continue;

    auto* nodes = ele->nodes();

    for (int node_lid = 0; node_lid < ele->num_node(); ++node_lid)
    {
      // micro and macro dofs at this node
      auto* node = nodes[node_lid];
      int dof_macro = discret_->dof(0, node)[0];
      int dof_micro = discret_->dof(nds_micro(), node)[0];

      const int dof_lid_micro = phinp_micro_->get_map().lid(dof_micro);
      const int dof_lid_macro = phinp_->get_map().lid(dof_macro);

      // only if owned by this proc
      if (dof_lid_micro != -1 and dof_lid_macro != -1)
      {
        const double macro_value = (*phinp_)[dof_lid_macro];
        // Sum, because afterwards it is divided by the number of adjacent nodes
        phinp_micro_->sum_into_local_value(dof_lid_micro, macro_value);
      }
    }
  }

  // divide nodal values by number of adjacent elements (due to assembly)
  const auto* node_row_map = discret_->node_row_map();
  for (int node_lid = 0; node_lid < node_row_map->num_my_elements(); ++node_lid)
  {
    const int node_gid = node_row_map->gid(node_lid);
    const auto* node = discret_->g_node(node_gid);
    std::vector<int> dofs = discret_->dof(nds_micro(), node);

    if (dofs.size() != 1) FOUR_C_THROW("Only one dof expected.");

    const int dof_gid = dofs[0];
    const int dof_lid = phinp_micro_->get_map().lid(dof_gid);

    // only if this dof is part of the phinp_micro_ vector/map
    if (dof_lid != -1)
    {
      const double old_value = (*phinp_micro_)[dof_lid];
      const int num_elements = node->num_element();
      const double new_value = old_value / static_cast<double>(num_elements);
      phinp_micro_->replace_local_value(dof_lid, new_value);
    }
  }

  // nodes with 3 dofs
  std::set<int> multiscale_nodes;
  // nodes with 2 dofs
  std::set<int> other_nodes;

  // loop over all element and search for nodes that are on elements with 2 dof on one side and 3
  // dofs at the other side
  for (int ele_lid = 0; ele_lid < discretization()->element_row_map()->num_my_elements(); ++ele_lid)
  {
    const int ele_gid = discretization()->element_row_map()->gid(ele_lid);
    auto* ele = discretization()->g_element(ele_gid);

    for (auto mat_id = 0; mat_id < ele->num_material(); ++mat_id)
    {
      auto ele_mat = ele->material(mat_id);
      auto material_type = ele_mat->material_type();

      if (material_type == Core::Materials::m_elchmat)
      {
        const auto* elchmat = static_cast<const Mat::ElchMat*>(ele_mat.get());

        const int num_dof_element = elchmat->num_dof();

        const Core::Nodes::Node* const* nodes = ele->nodes();
        for (int inode = 0; inode < ele->num_node(); ++inode)
        {
          if (num_dof_element == 3)
            Core::Communication::add_owned_node_gid(
                *discretization(), nodes[inode]->id(), multiscale_nodes);
          else if (num_dof_element == 2)
            Core::Communication::add_owned_node_gid(
                *discretization(), nodes[inode]->id(), other_nodes);
          else
            FOUR_C_THROW("Only 2 or 3 dofs per element supported");
        }
      }
    }
  }

  // find nodes that connect elements with 2 and 3 dofs ("hybrid nodes")
  std::vector<int> hybrid_nodes;
  for (int other_node : other_nodes)
  {
    for (int multiscale_node : multiscale_nodes)
    {
      if (other_node == multiscale_node)
      {
        hybrid_nodes.emplace_back(multiscale_node);
        break;
      }
    }
  }

  // get dofs from hybrid nodes
  std::vector<int> hybrid_dofs;
  for (int hybrid_node_gid : hybrid_nodes)
  {
    auto* hybrid_node = discretization()->g_node(hybrid_node_gid);
    auto dofs = discretization()->dof(2, hybrid_node);
    for (int dof : dofs) hybrid_dofs.emplace_back(dof);
  }

  // correct values on hybrid dofs (value on node with 2 dofs is artificially set to 0.0)
  for (int hybrid_dof : hybrid_dofs)
  {
    const int lid = phinp_micro_->get_map().lid(hybrid_dof);
    if (lid != -1)
    {
      const double value = (*phinp_micro_)[lid];
      const double corrected_value = 2.0 * value;
      phinp_micro_->replace_local_value(lid, corrected_value);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::add_problem_specific_parameters_and_vectors(
    Teuchos::ParameterList& params)
{
  if (micro_scale_) params.set<double>("rea_coeff", macro_micro_rea_coeff_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::set_time_stepping_to_micro_scale()
{
  Teuchos::ParameterList eleparams;

  Core::Utils::add_enum_class_to_parameter_list<ScaTra::Action>(
      "action", ScaTra::Action::micro_scale_set_time, eleparams);

  eleparams.set<double>("dt", dta_);
  eleparams.set<double>("time", time_);
  eleparams.set<int>("step", step_);

  // call standard loop over elements
  discret_->evaluate(eleparams, nullptr, nullptr, nullptr, nullptr, nullptr);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Utils::ResultTest> ScaTra::ScaTraTimIntImpl::create_scatra_field_test()
{
  return std::make_shared<ScaTra::ScaTraResultTest>(Core::Utils::shared_ptr_from_ref(*this));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::ScaTraTimIntImpl::test_results()
{
  Global::Problem::instance()->add_field_test(create_scatra_field_test());
  Global::Problem::instance()->test_all(discret_->get_comm());
}

FOUR_C_NAMESPACE_CLOSE
