// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_str_structure.hpp"

#include "4C_adapter_str_constr_merged.hpp"
#include "4C_adapter_str_fpsiwrapper.hpp"
#include "4C_adapter_str_fsi_timint_adaptive.hpp"
#include "4C_adapter_str_fsiwrapper_immersed.hpp"
#include "4C_adapter_str_redairway.hpp"
#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_adapter_str_timeloop.hpp"
#include "4C_adapter_str_timint_adaptive.hpp"
#include "4C_adapter_str_wrapper.hpp"
#include "4C_comm_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_inpar_poroelast.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_inpar_validparameters.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_structure_timada_create.hpp"
#include "4C_structure_timint_create.hpp"
#include "4C_structure_timint_impl.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Adapter::StructureBaseAlgorithm::StructureBaseAlgorithm(const Teuchos::ParameterList& prbdyn,
    const Teuchos::ParameterList& sdyn, std::shared_ptr<Core::FE::Discretization> actdis)
{
  create_structure(prbdyn, sdyn, actdis);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::StructureBaseAlgorithm::create_structure(const Teuchos::ParameterList& prbdyn,
    const Teuchos::ParameterList& sdyn, std::shared_ptr<Core::FE::Discretization> actdis)
{
  // major switch to different time integrators
  switch (Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(sdyn, "DYNAMICTYPE"))
  {
    case Inpar::Solid::dyna_statics:
    case Inpar::Solid::dyna_genalpha:
    case Inpar::Solid::dyna_onesteptheta:
    case Inpar::Solid::dyna_expleuler:
    case Inpar::Solid::dyna_centrdiff:
    case Inpar::Solid::dyna_ab2:
      create_tim_int(prbdyn, sdyn, actdis);  // <-- here is the show
      break;
    default:
      FOUR_C_THROW("unknown time integration scheme '%s'",
          Teuchos::getStringValue<Inpar::Solid::DynamicType>(sdyn, "DYNAMICTYPE").c_str());
      break;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::StructureBaseAlgorithm::create_tim_int(const Teuchos::ParameterList& prbdyn,
    const Teuchos::ParameterList& sdyn, std::shared_ptr<Core::FE::Discretization> actdis)
{
  // this is not exactly a one hundred meter race, but we need timing
  auto t =
      Teuchos::TimeMonitor::getNewTimer("Adapter::StructureTimIntBaseAlgorithm::SetupStructure");
  Teuchos::TimeMonitor monitor(*t);

  // get the problem instance
  Global::Problem* problem = Global::Problem::instance();
  // what's the current problem type?
  Core::ProblemType probtype = problem->get_problem_type();

  // get mortar information
  std::vector<Core::Conditions::Condition*> mtcond(0);
  std::vector<Core::Conditions::Condition*> ccond(0);
  actdis->get_condition("Mortar", mtcond);
  actdis->get_condition("Contact", ccond);
  bool onlymeshtying = false;
  bool onlycontact = false;
  bool meshtyingandcontact = false;
  if (mtcond.size() != 0 and ccond.size() != 0) meshtyingandcontact = true;
  if (mtcond.size() != 0 and ccond.size() == 0) onlymeshtying = true;
  if (mtcond.size() == 0 and ccond.size() != 0) onlycontact = true;

  // Problem-types involving changing mesh or redistribution of mesh
  // for load balancing (like contact) during the simulation needs an additional step.
  // This is because the discretization read from the input file
  // do not match with the discr. at the current time step.
  // Here we read the discretization at the current time step from restart files
  if (not actdis->filled() || not actdis->have_dofs()) actdis->fill_complete();

  // get input parameter lists and copy them, because a few parameters are overwritten
  // const Teuchos::ParameterList& probtype
  //  = problem->ProblemTypeParams();
  Teuchos::ParameterList ioflags(problem->io_params());
  Teuchos::ParameterList tap(sdyn.sublist("TIMEADAPTIVITY"));
  Teuchos::ParameterList snox(problem->structural_nox_params());

  // add extra parameters (a kind of work-around)
  std::shared_ptr<Teuchos::ParameterList> xparams = std::make_shared<Teuchos::ParameterList>();
  Teuchos::ParameterList& nox = xparams->sublist("NOX");
  nox = snox;

  // Check if for chosen Rayleigh damping the regarding parameters are given explicitly in the .dat
  // file
  if (Teuchos::getIntegralValue<Inpar::Solid::DampKind>(sdyn, "DAMPING") ==
      Inpar::Solid::damp_rayleigh)
  {
    if (sdyn.get<double>("K_DAMP") < 0.0)
    {
      FOUR_C_THROW("Rayleigh damping parameter K_DAMP not explicitly given.");
    }
    if (sdyn.get<double>("M_DAMP") < 0.0)
    {
      FOUR_C_THROW("Rayleigh damping parameter M_DAMP not explicitly given.");
    }
  }

  // create a solver
  std::shared_ptr<Core::LinAlg::Solver> solver = create_linear_solver(actdis, sdyn);

  // create contact/meshtying solver only if contact/meshtying problem.
  std::shared_ptr<Core::LinAlg::Solver> contactsolver = nullptr;

  if (onlymeshtying or onlycontact or meshtyingandcontact)
    contactsolver = create_contact_meshtying_solver(*actdis, sdyn);

  if (solver != nullptr && (solver->params().isSublist("Belos Parameters")) &&
      solver->params().isSublist("ML Parameters")  // TODO what about MueLu?
      && Teuchos::getIntegralValue<Inpar::Solid::StcScale>(sdyn, "STC_SCALING") !=
             Inpar::Solid::stc_none)
  {
    Teuchos::ParameterList& mllist = solver->params().sublist("ML Parameters");
    std::shared_ptr<std::vector<double>> ns =
        mllist.get<std::shared_ptr<std::vector<double>>>("nullspace");

    // prepare matrix for scaled thickness business of thin shell structures
    std::shared_ptr<Core::LinAlg::SparseMatrix> stcinv =
        std::make_shared<Core::LinAlg::SparseMatrix>(*actdis->dof_row_map(), 81, true, true);

    stcinv->zero();
    // create the parameters for the discretization
    Teuchos::ParameterList p;
    // action for elements
    const std::string action = "calc_stc_matrix_inverse";
    p.set("action", action);
    p.set<Inpar::Solid::StcScale>("stc_scaling", sdyn.get<Inpar::Solid::StcScale>("STC_SCALING"));
    p.set("stc_layer", 1);

    actdis->evaluate(p, stcinv, nullptr, nullptr, nullptr, nullptr);

    stcinv->complete();

    for (int lay = 2; lay <= sdyn.get<int>("STC_LAYER"); ++lay)
    {
      Teuchos::ParameterList pe;

      p.set("stc_layer", lay);

      std::shared_ptr<Core::LinAlg::SparseMatrix> tmpstcmat =
          std::make_shared<Core::LinAlg::SparseMatrix>(*actdis->dof_row_map(), 81, true, true);
      tmpstcmat->zero();

      actdis->evaluate(p, tmpstcmat, nullptr, nullptr, nullptr, nullptr);
      tmpstcmat->complete();

      stcinv = Core::LinAlg::matrix_multiply(*stcinv, false, *tmpstcmat, false, false, false, true);
    }

    Core::LinAlg::Vector<double> temp(*(actdis->dof_row_map()), false);

    const auto multiply_nullspace_vector = [&](std::size_t offset)
    {
      const int size = actdis->dof_row_map()->NumMyElements();
      Core::LinAlg::Vector<double> vec(*(actdis->dof_row_map()), false);
      std::copy(ns->data() + offset * size, ns->data() + (offset + 1) * size, vec.Values());

      stcinv->multiply(false, vec, temp);
      std::copy(temp.Values(), temp.Values() + size, ns->data() + offset * size);
    };


    // extract and modify the six nullspace vectors corresponding to the modes
    // trans x, trans y, trans z, rot x, rot y, rot z
    // Note: We assume 3d here!
    for (int i = 0; i < 6; ++i) multiply_nullspace_vector(i);
  }

  // Checks in case of multi-scale simulations
  {
    // make sure we IMR-like generalised-alpha requested for multi-scale
    // simulations
    std::shared_ptr<Mat::PAR::Bundle> materials = problem->materials();
    for (const auto& [_, par] : materials->map())
    {
      if (par->type() == Core::Materials::m_struct_multiscale)
      {
        if (Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(sdyn, "DYNAMICTYPE") !=
            Inpar::Solid::dyna_genalpha)
          FOUR_C_THROW("In multi-scale simulations, you have to use DYNAMICTYPE=GenAlpha");
        else if (Teuchos::getIntegralValue<Inpar::Solid::MidAverageEnum>(
                     sdyn.sublist("GENALPHA"), "GENAVG") != Inpar::Solid::midavg_trlike)
          FOUR_C_THROW(
              "In multi-scale simulations, you have to use DYNAMICTYPE=GenAlpha with "
              "GENAVG=TrLike");
        break;
      }
    }
  }

  // context for output and restart
  std::shared_ptr<Core::IO::DiscretizationWriter> output = actdis->writer();
  if (ioflags.get<bool>("OUTPUT_BIN"))
  {
    output->write_mesh(0, 0.0);
  }

  // create marching time integrator
  std::shared_ptr<Solid::TimInt> tmpstr =
      Solid::tim_int_create(prbdyn, ioflags, sdyn, *xparams, actdis, solver, contactsolver, output);
  // initialize the time integrator
  tmpstr->init(prbdyn, sdyn, *xparams, actdis, solver);

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  /* Overwrite certain parameters in STRUCTURAL DYNAMIC/TIMADAPTIVITY by those from
   * FSI DYNAMIC/TIMEADAPTIVITY
   *
   * In case, that the structure field is part of an FSI simulation with time step
   * size adaptivity based on structure field error estimation, we have to provide
   * the following algorithmic control parameters:
   *
   * - ADAPTSTEPMAX
   * - STEPSIZEMAX
   * - STEPSIZEMIN
   * - SIZERATIOMAX
   * - SIZERATIOMIN
   * - SIZERATIOSCALE
   *
   * They are specified by the FSI algorithm since they have to be the same for
   * the structure and fluid field. Hence, we overwrite the corresponding
   * parameters in the structural parameter list in order to avoid redundant
   * parameter specification in the input file.
   *
   * Note: This is really ugly, but currently the only way to avoid that the user
   * has to specify these parameters twice in the input file.
   *
   * ToDO: Find something nicer here!
   *
   * \author mayr.mt \date 12/2013
   */
  // ---------------------------------------------------------------------------
  if (probtype == Core::ProblemType::fsi or probtype == Core::ProblemType::fsi_redmodels)
  {
    const Teuchos::ParameterList& fsidyn = problem->fsi_dynamic_params();
    const Teuchos::ParameterList& fsiada = fsidyn.sublist("TIMEADAPTIVITY");
    if (fsiada.get<bool>("TIMEADAPTON"))
    {
      // overrule time step size adaptivity control parameters
      if (tap.get<Inpar::Solid::TimAdaKind>("KIND") != Inpar::Solid::timada_kind_none)
      {
        tap.set<int>("ADAPTSTEPMAX", fsiada.get<int>("ADAPTSTEPMAX"));
        tap.set<double>("STEPSIZEMAX", fsiada.get<double>("DTMAX"));
        tap.set<double>("STEPSIZEMIN", fsiada.get<double>("DTMIN"));
        tap.set<double>("SIZERATIOMAX", fsiada.get<double>("SIZERATIOMAX"));
        tap.set<double>("SIZERATIOMIN", fsiada.get<double>("SIZERATIOMIN"));
        tap.set<double>("SIZERATIOSCALE", fsiada.get<double>("SAFETYFACTOR"));

        if (Core::Communication::my_mpi_rank(actdis->get_comm()) == 0)
        {
          Core::IO::cout
              << "*** Due to FSI time step size adaptivity with structure based error estimation,\n"
                 "algorithmic control parameters in STRUCTURAL DYNAMIC/TIMEADAPTIVITY have been\n"
                 "overwritten by those from FSI DYNAMIC/TIMEADAPTIVITY."
              << Core::IO::endl
              << Core::IO::endl;
        }
      }
    }
  }
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------

  // create auxiliary time integrator, can be seen as a wrapper for tmpstr
  std::shared_ptr<Solid::TimAda> sta =
      Solid::tim_ada_create(ioflags, prbdyn, sdyn, *xparams, tap, tmpstr);

  if (sta != nullptr and tmpstr != nullptr)
  {
    switch (probtype)
    {
      case Core::ProblemType::structure:  // pure structural time adaptivity
      {
        structure_ = std::make_shared<StructureTimIntAda>(sta, tmpstr);
        break;
      }
      case Core::ProblemType::fsi:  // structure based time adaptivity within an FSI simulation
      case Core::ProblemType::fsi_redmodels:
      {
        if (Core::Communication::my_mpi_rank((actdis->get_comm())) == 0)
          Core::IO::cout << "Using StructureNOXCorrectionWrapper()..." << Core::IO::endl;

        std::shared_ptr<FSIStructureWrapper> fsiwrapperwithadaptivity =
            std::make_shared<StructureFSITimIntAda>(
                sta, std::make_shared<StructureNOXCorrectionWrapper>(tmpstr));
        // strTeuchos::rcp_dynamic_cast<StructureFSITimIntAda>(fsiwrapperwithadaptivity)->GetStrTimIntPtr();
        structure_ = fsiwrapperwithadaptivity;
        // structure_->GetStrTimIntPtr()-(prbdyn,sdyn,*xparams,actdis,solver);
        break;
      }
      default:
      {
        FOUR_C_THROW(
            "Adaptive time integration for the structure not implemented for desired problem "
            "type.");
        break;
      }
    }
  }
  else if (sta == nullptr and tmpstr != nullptr)
  {
    switch (probtype)
    {
      case Core::ProblemType::fsi:
      case Core::ProblemType::fsi_redmodels:
      case Core::ProblemType::gas_fsi:
      case Core::ProblemType::biofilm_fsi:
      case Core::ProblemType::thermo_fsi:
      {
        if (Core::Communication::my_mpi_rank((actdis->get_comm())) == 0)
          Core::IO::cout << "Using StructureNOXCorrectionWrapper()..." << Core::IO::endl;

        if (tmpstr->have_constraint())
        {
          structure_ = std::make_shared<StructureConstrMerged>(
              std::make_shared<StructureNOXCorrectionWrapper>(tmpstr));
        }
        else
        {
          structure_ = std::make_shared<FSIStructureWrapper>(
              std::make_shared<StructureNOXCorrectionWrapper>(tmpstr));
        }
      }
      break;
      case Core::ProblemType::immersed_fsi:
      {
        structure_ = std::make_shared<FSIStructureWrapperImmersed>(tmpstr);
      }
      break;
      case Core::ProblemType::ssi:
      case Core::ProblemType::ssti:
      {
        structure_ = std::make_shared<SSIStructureWrapper>(tmpstr);
      }
      break;
      case Core::ProblemType::redairways_tissue:
      {
        structure_ = std::make_shared<StructureRedAirway>(tmpstr);
      }
      break;
      case Core::ProblemType::poroelast:
      case Core::ProblemType::poroscatra:
      case Core::ProblemType::fpsi:
      case Core::ProblemType::fps3i:
      case Core::ProblemType::fpsi_xfem:
      case Core::ProblemType::fsi_xfem:
      {
        const Teuchos::ParameterList& porodyn = problem->poroelast_dynamic_params();
        const auto coupling = Teuchos::getIntegralValue<Inpar::PoroElast::SolutionSchemeOverFields>(
            porodyn, "COUPALGO");
        if (tmpstr->have_constraint())
        {
          if (coupling == Inpar::PoroElast::Monolithic_structuresplit or
              coupling == Inpar::PoroElast::Monolithic_fluidsplit or
              coupling == Inpar::PoroElast::Monolithic_nopenetrationsplit)
            structure_ = std::make_shared<FPSIStructureWrapper>(tmpstr);
          else
            structure_ = std::make_shared<StructureConstrMerged>(tmpstr);
        }
        else
        {
          structure_ = std::make_shared<FPSIStructureWrapper>(tmpstr);
        }
      }
      break;
      default:
      {
        /// wrap time loop for pure structure problems
        structure_ = (std::make_shared<StructureTimeLoop>(tmpstr));
      }
      break;
    }
  }
  else
  {
    FOUR_C_THROW("no proper time integration found");
  }
  // see you
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Solver> Adapter::StructureBaseAlgorithm::create_linear_solver(
    std::shared_ptr<Core::FE::Discretization>& actdis, const Teuchos::ParameterList& sdyn)
{
  std::shared_ptr<Core::LinAlg::Solver> solver = nullptr;

  // get the solver number used for structural problems
  const int linsolvernumber = sdyn.get<int>("LINEAR_SOLVER");
  // check if the structural solver has a valid solver number
  if (linsolvernumber == (-1))
    FOUR_C_THROW(
        "no linear solver defined for structural field. Please set LINEAR_SOLVER in STRUCTURAL "
        "DYNAMIC to a valid number!");

  solver = std::make_shared<Core::LinAlg::Solver>(
      Global::Problem::instance()->solver_params(linsolvernumber), actdis->get_comm(),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));

  actdis->compute_null_space_if_necessary(solver->params());

  return solver;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Solver>
Adapter::StructureBaseAlgorithm::create_contact_meshtying_solver(
    Core::FE::Discretization& actdis, const Teuchos::ParameterList& sdyn)
{
  std::shared_ptr<Core::LinAlg::Solver> solver = nullptr;

  // Get mortar information: contact or meshtying or both?
  bool onlymeshtying = false;
  bool onlycontact = false;
  bool meshtyingandcontact = false;
  {
    std::vector<Core::Conditions::Condition*> mtcond(0);
    std::vector<Core::Conditions::Condition*> ccond(0);
    actdis.get_condition("Mortar", mtcond);
    actdis.get_condition("Contact", ccond);
    if (mtcond.size() != 0 and ccond.size() != 0) meshtyingandcontact = true;
    if (mtcond.size() != 0 and ccond.size() == 0) onlymeshtying = true;
    if (mtcond.size() == 0 and ccond.size() != 0) onlycontact = true;
  }
  const Teuchos::ParameterList& mcparams = Global::Problem::instance()->contact_dynamic_params();

  // Get the solver number used for meshtying/contact problems
  const int linsolvernumber = mcparams.get<int>("LINEAR_SOLVER");
  // check if the meshtying/contact solver has a valid solver number
  if (linsolvernumber == (-1))
    FOUR_C_THROW(
        "No linear solver defined for meshtying/contact problem. Please set LINEAR_SOLVER in "
        "CONTACT DYNAMIC to a valid number!");

  // Distinguish the system type, i.e. condensed vs. saddle-point
  switch (Teuchos::getIntegralValue<Inpar::CONTACT::SystemType>(mcparams, "SYSTEM"))
  {
    case Inpar::CONTACT::system_saddlepoint:
    {
      /* Plausibility check
       *
       * Solver can be either a direct solver (UMFPACK, Superlu) or an iterative solver (Belos).
       */
      const auto sol = Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(
          Global::Problem::instance()->solver_params(linsolvernumber), "SOLVER");
      const auto prec = Teuchos::getIntegralValue<Core::LinearSolver::PreconditionerType>(
          Global::Problem::instance()->solver_params(linsolvernumber), "AZPREC");
      if (sol != Core::LinearSolver::SolverType::umfpack &&
          sol != Core::LinearSolver::SolverType::superlu)
      {
        // if an iterative solver is chosen we need a block preconditioner
        if (prec != Core::LinearSolver::PreconditionerType::multigrid_muelu &&
            prec != Core::LinearSolver::PreconditionerType::block_teko &&
            prec != Core::LinearSolver::PreconditionerType::ilu)
          FOUR_C_THROW(
              "You have chosen an iterative linear solver. For mortar meshtying/contact problems "
              "in saddle-point formulation, a block preconditioner is required. Choose an "
              "appropriate block preconditioner such as CheapSIMPLE or MueLu "
              "(if MueLu is available) in the SOLVER %i block in your input file.",
              linsolvernumber);
      }

      // build meshtying/contact solver
      solver = std::make_shared<Core::LinAlg::Solver>(
          Global::Problem::instance()->solver_params(linsolvernumber), actdis.get_comm(),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));

      // feed the solver object with additional information
      if (onlycontact or meshtyingandcontact)
        solver->params().set<bool>("CONTACT", true);
      else if (onlymeshtying)
        solver->params().set<bool>("MESHTYING", true);
      else
        FOUR_C_THROW(
            "Saddle-point formulations are only supported for solid CONTACT or MESHTYING problems. "
            "Problems like beamcontact or pure structure problem w/o contact do not support a "
            "saddle-point formulation.");

      auto soltype =
          Teuchos::getIntegralValue<Inpar::CONTACT::SolvingStrategy>(mcparams, "STRATEGY");
      if (soltype == Inpar::CONTACT::solution_lagmult)
      {
        // get the solver number used for structural problems
        const int linsolvernumber = sdyn.get<int>("LINEAR_SOLVER");
        // check if the structural solver has a valid solver number
        if (linsolvernumber == (-1))
          FOUR_C_THROW(
              "No linear solver defined for structural field. Please set LINEAR_SOLVER in "
              "STRUCTURAL DYNAMIC to a valid number!");

        // provide null space information
        if (prec == Core::LinearSolver::PreconditionerType::multigrid_muelu)
        {
          Core::LinearSolver::Parameters::compute_solver_parameters(
              actdis, solver->params().sublist("Inverse1").sublist("MueLu Parameters"));
          Core::LinearSolver::Parameters::compute_solver_parameters(
              actdis, solver->params().sublist("Inverse2").sublist("MueLu Parameters"));
        }
        else if (prec == Core::LinearSolver::PreconditionerType::block_teko)
        {
          Core::LinearSolver::Parameters::compute_solver_parameters(
              actdis, solver->params().sublist("Inverse1"));
        }
      }
    }
    break;
    default:
    {
      // build meshtying solver
      solver = std::make_shared<Core::LinAlg::Solver>(
          Global::Problem::instance()->solver_params(linsolvernumber), actdis.get_comm(),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));
      actdis.compute_null_space_if_necessary(solver->params());
    }
    break;
  }

  return solver;
}

FOUR_C_NAMESPACE_CLOSE
