// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_scatra_base_algorithm.hpp"

#include "4C_global_data.hpp"
#include "4C_inpar_ssi.hpp"
#include "4C_inpar_ssti.hpp"
#include "4C_inpar_sti.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_levelset_timint_ost.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_scatra_resulttest_hdg.hpp"
#include "4C_scatra_timint_bdf2.hpp"
#include "4C_scatra_timint_cardiac_monodomain_scheme.hpp"
#include "4C_scatra_timint_cardiac_monodomain_scheme_hdg.hpp"
#include "4C_scatra_timint_elch_scheme.hpp"
#include "4C_scatra_timint_genalpha.hpp"
#include "4C_scatra_timint_loma_genalpha.hpp"
#include "4C_scatra_timint_ost.hpp"
#include "4C_scatra_timint_poromulti.hpp"
#include "4C_scatra_timint_stat.hpp"
#include "4C_scatra_timint_stat_hdg.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Adapter::ScaTraBaseAlgorithm::ScaTraBaseAlgorithm(const Teuchos::ParameterList& prbdyn,
    const Teuchos::ParameterList& scatradyn, const Teuchos::ParameterList& solverparams,
    const std::string& disname, const bool isale)
    : scatra_(nullptr), issetup_(false), isinit_(false)
{
  // setup scalar transport algorithm (overriding some dynamic parameters
  // with values specified in given problem-dependent ParameterList prbdyn)

  // -------------------------------------------------------------------
  // what's the current problem type?
  // -------------------------------------------------------------------
  auto probtype = Global::Problem::instance()->get_problem_type();

  // -------------------------------------------------------------------
  // access the discretization
  // -------------------------------------------------------------------
  auto discret = Global::Problem::instance()->get_dis(disname);

  // -------------------------------------------------------------------
  // set degrees of freedom in the discretization
  // -------------------------------------------------------------------
  if (!discret->filled() or !discret->have_dofs()) discret->fill_complete();

  // -------------------------------------------------------------------
  // context for output and restart
  // -------------------------------------------------------------------
  auto output = discret->writer();
  if (discret->num_global_elements() == 0)
    FOUR_C_THROW("No elements in discretization {}", discret->name());
  output->write_mesh(0, 0.0);

  // -------------------------------------------------------------------
  // create a solver
  // -------------------------------------------------------------------
  // TODO: TAW use of solverparams???
  // change input parameter to solver number instead of parameter list?
  // -> no default parameter possible any more
  auto solver = std::make_shared<Core::LinAlg::Solver>(solverparams, discret->get_comm(),
      Global::Problem::instance()->solver_params_callback(),
      Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
          Global::Problem::instance()->io_params(), "VERBOSITY"));

  // -------------------------------------------------------------------
  // set parameters in list required for all schemes
  // -------------------------------------------------------------------
  // make a copy (inside an Teuchos::rcp) containing also all sublists
  auto scatratimeparams = std::make_shared<Teuchos::ParameterList>(scatradyn);
  if (scatratimeparams == nullptr) FOUR_C_THROW("Instantiation of Teuchos::ParameterList failed!");

  // -------------------------------------------------------------------
  // overrule certain parameters for coupled problems
  // -------------------------------------------------------------------
  // the default time step size
  scatratimeparams->set<double>("TIMESTEP", prbdyn.get<double>("TIMESTEP"));
  // maximum simulation time
  scatratimeparams->set<double>("MAXTIME", prbdyn.get<double>("MAXTIME"));
  // maximum number of timesteps
  scatratimeparams->set<int>("NUMSTEP", prbdyn.get<int>("NUMSTEP"));
  // restart
  scatratimeparams->set<int>("RESTARTEVERY", prbdyn.get<int>("RESTARTEVERY"));
  // solution output
  scatratimeparams->set<int>("RESULTSEVERY", prbdyn.get<int>("RESULTSEVERY"));

  // -------------------------------------------------------------------
  // overrule flags for solid-based scalar transport!
  // (assumed disname = "scatra2" for solid-based scalar transport)
  // -------------------------------------------------------------------
  if (probtype == Core::ProblemType::biofilm_fsi or probtype == Core::ProblemType::gas_fsi or
      probtype == Core::ProblemType::fps3i or probtype == Core::ProblemType::thermo_fsi)
  {
    // scatra1 (=fluid scalar) get's inputs from SCALAR TRANSPORT DYNAMIC/STABILIZATION, hence
    // nothing is to do here
    //    if (disname== "scatra1") //get's inputs from SCALAR TRANSPORT DYNAMIC/STABILIZATION

    if (disname == "scatra2")  // structure_scatra discretisation
    {
      // scatra2 (=structure scalar) get's inputs from FS3I DYNAMIC/STRUCTURE SCALAR STABILIZATION,
      // hence we have to replace it
      scatratimeparams->sublist("STABILIZATION") = prbdyn.sublist("STRUCTURE SCALAR STABILIZATION");
      scatratimeparams->set<Inpar::ScaTra::ConvForm>(
          "CONVFORM", prbdyn.get<Inpar::ScaTra::ConvForm>("STRUCTSCAL_CONVFORM"));

      auto initial_field =
          Teuchos::getIntegralValue<Inpar::ScaTra::InitialField>(prbdyn, "STRUCTSCAL_INITIALFIELD");

      scatratimeparams->set("INITIALFIELD", initial_field);
      // scatra2 get's in initial functions from FS3I DYNAMICS
      switch (initial_field)
      {
        case Inpar::ScaTra::initfield_zero_field:
          scatratimeparams->set<int>("INITFUNCNO", -1);
          break;
        case Inpar::ScaTra::initfield_field_by_function:
          scatratimeparams->set<int>("INITFUNCNO", prbdyn.get<int>("STRUCTSCAL_INITFUNCNO"));
          break;
        default:
          FOUR_C_THROW("Your STRUCTSCAL_INITIALFIELD type is not supported!");
          break;
      }

      // structure scatra does not require any Neumann inflow boundary conditions
      scatratimeparams->set<bool>("NEUMANNINFLOW", false);
    }
    else if (disname == "scatra1")  // fluid_scatra discretisation
    {
      // fluid scatra does not require any convective heat transfer boundary conditions
      scatratimeparams->set<bool>("CONV_HEAT_TRANS", false);
    }
  }

  // -------------------------------------------------------------------
  // list for extra parameters
  // (put here everything that is not available in scatradyn or its sublists)
  // -------------------------------------------------------------------
  auto extraparams = std::make_shared<Teuchos::ParameterList>();

  // ----------------Eulerian or ALE formulation of transport equation(s)
  extraparams->set<bool>("isale", isale);

  // ------------------------------------get also fluid turbulence sublist
  const auto& fdyn = Global::Problem::instance()->fluid_dynamic_params();
  extraparams->sublist("TURBULENCE MODEL") = fdyn.sublist("TURBULENCE MODEL");
  extraparams->sublist("SUBGRID VISCOSITY") = fdyn.sublist("SUBGRID VISCOSITY");
  extraparams->sublist("MULTIFRACTAL SUBGRID SCALES") = fdyn.sublist("MULTIFRACTAL SUBGRID SCALES");
  extraparams->sublist("TURBULENT INFLOW") = fdyn.sublist("TURBULENT INFLOW");

  // -------------------------------------------------------------------
  // algorithm construction depending on problem type and
  // time-integration (or stationary) scheme
  // -------------------------------------------------------------------
  auto timintscheme =
      Teuchos::getIntegralValue<Inpar::ScaTra::TimeIntegrationScheme>(scatradyn, "TIMEINTEGR");

  // low Mach number flow
  if (probtype == Core::ProblemType::loma or probtype == Core::ProblemType::thermo_fsi)
  {
    auto lomaparams = std::make_shared<Teuchos::ParameterList>(
        Global::Problem::instance()->loma_control_params());
    switch (timintscheme)
    {
      case Inpar::ScaTra::timeint_gen_alpha:
      {
        // create instance of time integration class (call the constructor)
        scatra_ = std::make_shared<ScaTra::TimIntLomaGenAlpha>(
            discret, solver, lomaparams, scatratimeparams, extraparams, output);
        break;
      }
      default:
        FOUR_C_THROW("Unknown time integration scheme for loMa!");
        break;
    }
  }

  // electrochemistry
  else if (probtype == Core::ProblemType::elch or
           ((probtype == Core::ProblemType::ssi and
                Teuchos::getIntegralValue<Inpar::SSI::ScaTraTimIntType>(
                    Global::Problem::instance()->ssi_control_params(), "SCATRATIMINTTYPE") ==
                    Inpar::SSI::ScaTraTimIntType::elch) or
               (disname == "scatra" and
                   ((probtype == Core::ProblemType::ssti and
                        Teuchos::getIntegralValue<Inpar::SSTI::ScaTraTimIntType>(
                            Global::Problem::instance()->ssti_control_params(),
                            "SCATRATIMINTTYPE") == Inpar::SSTI::ScaTraTimIntType::elch) or
                       (probtype == Core::ProblemType::sti and
                           Teuchos::getIntegralValue<Inpar::STI::ScaTraTimIntType>(
                               Global::Problem::instance()->sti_dynamic_params(),
                               "SCATRATIMINTTYPE") == Inpar::STI::ScaTraTimIntType::elch)))))
  {
    auto elchparams = std::make_shared<Teuchos::ParameterList>(
        Global::Problem::instance()->elch_control_params());

    switch (timintscheme)
    {
      case Inpar::ScaTra::timeint_one_step_theta:
      {
        if (elchparams->sublist("SCL").get<bool>("ADD_MICRO_MACRO_COUPLING"))
        {
          if (disname == "scatra")
          {
            scatra_ = std::make_shared<ScaTra::ScaTraTimIntElchSCLOST>(
                discret, solver, elchparams, scatratimeparams, extraparams, output);
          }
          else if (disname == "scatra_micro")
          {
            scatra_ = std::make_shared<ScaTra::ScaTraTimIntElchOST>(
                discret, solver, elchparams, scatratimeparams, extraparams, output);
          }
          else
            FOUR_C_THROW("not identified");
        }
        else
        {
          scatra_ = std::make_shared<ScaTra::ScaTraTimIntElchOST>(
              discret, solver, elchparams, scatratimeparams, extraparams, output);
        }

        break;
      }
      case Inpar::ScaTra::timeint_bdf2:
      {
        // create instance of time integration class (call the constructor)
        scatra_ = std::make_shared<ScaTra::ScaTraTimIntElchBDF2>(
            discret, solver, elchparams, scatratimeparams, extraparams, output);
        break;
      }
      case Inpar::ScaTra::timeint_gen_alpha:
      {
        // create instance of time integration class (call the constructor)
        scatra_ = std::make_shared<ScaTra::ScaTraTimIntElchGenAlpha>(
            discret, solver, elchparams, scatratimeparams, extraparams, output);
        break;
      }
      case Inpar::ScaTra::timeint_stationary:
      {
        // create instance of time integration class (call the constructor)
        scatra_ = std::make_shared<ScaTra::ScaTraTimIntElchStationary>(
            discret, solver, elchparams, scatratimeparams, extraparams, output);
        break;
      }
      default:
        FOUR_C_THROW("Unknown time integration scheme for electrochemistry!");
    }
  }

  // level-set
  else if (probtype == Core::ProblemType::level_set)
  {
    auto lsparams = std::make_shared<Teuchos::ParameterList>(prbdyn);

    switch (timintscheme)
    {
      case Inpar::ScaTra::timeint_one_step_theta:
      {
        // create an instance of time integration class
        scatra_ = std::make_shared<ScaTra::LevelSetTimIntOneStepTheta>(
            discret, solver, lsparams, scatratimeparams, extraparams, output);
        break;
      }
      default:
        FOUR_C_THROW("Unknown time-integration scheme for level-set problem");
    }
  }

  // cardiac monodomain
  else if (probtype == Core::ProblemType::cardiac_monodomain or
           (probtype == Core::ProblemType::ssi and
               Teuchos::getIntegralValue<Inpar::SSI::ScaTraTimIntType>(
                   Global::Problem::instance()->ssi_control_params(), "SCATRATIMINTTYPE") ==
                   Inpar::SSI::ScaTraTimIntType::cardiac_monodomain))
  {
    auto cmonoparams =
        std::make_shared<Teuchos::ParameterList>(Global::Problem::instance()->ep_control_params());

    // HDG implements all time stepping schemes within gen-alpha
    if (Global::Problem::instance()->spatial_approximation_type() ==
        Core::FE::ShapeFunctionType::hdg)
    {
      scatra_ = std::make_shared<ScaTra::TimIntCardiacMonodomainHDG>(
          discret, solver, cmonoparams, scatratimeparams, extraparams, output);
    }
    else
    {
      switch (timintscheme)
      {
        case Inpar::ScaTra::timeint_gen_alpha:
        {
          // create instance of time integration class (call the constructor)
          scatra_ = std::make_shared<ScaTra::TimIntCardiacMonodomainGenAlpha>(
              discret, solver, cmonoparams, scatratimeparams, extraparams, output);
          break;
        }
        case Inpar::ScaTra::timeint_one_step_theta:
        {
          // create instance of time integration class (call the constructor)
          scatra_ = std::make_shared<ScaTra::TimIntCardiacMonodomainOST>(
              discret, solver, cmonoparams, scatratimeparams, extraparams, output);
          break;
        }
        case Inpar::ScaTra::timeint_bdf2:
        {
          // create instance of time integration class (call the constructor)
          scatra_ = std::make_shared<ScaTra::TimIntCardiacMonodomainBDF2>(
              discret, solver, cmonoparams, scatratimeparams, extraparams, output);
          break;
        }
        default:
          FOUR_C_THROW("Unknown time integration scheme for cardiac monodomain problem!");
          break;
      }  // switch(timintscheme)
    }
  }
  else if (probtype == Core::ProblemType::porofluid_pressure_based_elast_scatra)
  {
    switch (timintscheme)
    {
      case Inpar::ScaTra::timeint_gen_alpha:
      {
        // create instance of time integration class (call the constructor)
        scatra_ = std::make_shared<ScaTra::ScaTraTimIntPoroMultiGenAlpha>(
            discret, solver, nullptr, scatratimeparams, extraparams, output);
        break;
      }
      case Inpar::ScaTra::timeint_one_step_theta:
      {
        // create instance of time integration class (call the constructor)
        scatra_ = std::make_shared<ScaTra::ScaTraTimIntPoroMultiOST>(
            discret, solver, nullptr, scatratimeparams, extraparams, output);
        break;
      }
      case Inpar::ScaTra::timeint_bdf2:
      {
        // create instance of time integration class (call the constructor)
        scatra_ = std::make_shared<ScaTra::ScaTraTimIntPoroMultiBDF2>(
            discret, solver, nullptr, scatratimeparams, extraparams, output);
        break;
      }
      case Inpar::ScaTra::timeint_stationary:
      {
        // create instance of time integration class (call the constructor)
        scatra_ = std::make_shared<ScaTra::ScaTraTimIntPoroMultiStationary>(
            discret, solver, nullptr, scatratimeparams, extraparams, output);
        break;
      }
      default:
        FOUR_C_THROW("Unknown time integration scheme for porous medium multiphase problem!");
        break;
    }  // switch(timintscheme)
  }
  // everything else
  else
  {
    // HDG implements all time stepping schemes within gen-alpha
    if (Global::Problem::instance()->spatial_approximation_type() ==
        Core::FE::ShapeFunctionType::hdg)
    {
      switch (timintscheme)
      {
        case Inpar::ScaTra::timeint_one_step_theta:
        case Inpar::ScaTra::timeint_bdf2:
        case Inpar::ScaTra::timeint_gen_alpha:
        {
          scatra_ = std::make_shared<ScaTra::TimIntHDG>(
              discret, solver, scatratimeparams, extraparams, output);
          break;
        }
        case Inpar::ScaTra::timeint_stationary:
        {
          scatra_ = std::make_shared<ScaTra::TimIntStationaryHDG>(
              discret, solver, scatratimeparams, extraparams, output);
          break;
        }
        default:
        {
          FOUR_C_THROW("Unknown time-integration scheme for HDG scalar transport problem");
          break;
        }
      }
    }
    else
    {
      switch (timintscheme)
      {
        case Inpar::ScaTra::timeint_stationary:
        {
          // create instance of time integration class (call the constructor)
          scatra_ = std::make_shared<ScaTra::TimIntStationary>(
              discret, solver, scatratimeparams, extraparams, output);
          break;
        }
        case Inpar::ScaTra::timeint_one_step_theta:
        {
          // create instance of time integration class (call the constructor)
          scatra_ = std::make_shared<ScaTra::TimIntOneStepTheta>(
              discret, solver, scatratimeparams, extraparams, output);
          break;
        }
        case Inpar::ScaTra::timeint_bdf2:
        {
          // create instance of time integration class (call the constructor)
          scatra_ = std::make_shared<ScaTra::TimIntBDF2>(
              discret, solver, scatratimeparams, extraparams, output);
          break;
        }
        case Inpar::ScaTra::timeint_gen_alpha:
        {
          // create instance of time integration class (call the constructor)
          scatra_ = std::make_shared<ScaTra::TimIntGenAlpha>(
              discret, solver, scatratimeparams, extraparams, output);
          break;
        }
        default:
          FOUR_C_THROW("Unknown time-integration scheme for scalar transport problem");
          break;
      }  // switch(timintscheme)
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::ScaTraBaseAlgorithm::init()
{
  set_is_setup(false);

  // initialize scatra time integrator
  scatra_->init();

  set_is_init(true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::ScaTraBaseAlgorithm::setup()
{
  check_is_init();

  // setup the time integrator
  scatra_->setup();

  set_is_setup(true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::Utils::ResultTest> Adapter::ScaTraBaseAlgorithm::create_scatra_field_test()
{
  return scatra_->create_scatra_field_test();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::ScaTraBaseAlgorithm::check_is_setup() const
{
  if (not is_setup()) FOUR_C_THROW("setup() was not called.");
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::ScaTraBaseAlgorithm::check_is_init() const
{
  if (not is_init()) FOUR_C_THROW("init(...) was not called.");
}

FOUR_C_NAMESPACE_CLOSE
