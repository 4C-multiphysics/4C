// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_fld_base_algorithm.hpp"

#include "4C_adapter_fld_fbi_wrapper.hpp"
#include "4C_adapter_fld_fluid_fluid_fsi.hpp"
#include "4C_adapter_fld_fluid_fpsi.hpp"
#include "4C_adapter_fld_fluid_fsi.hpp"
#include "4C_adapter_fld_fluid_fsi_msht.hpp"
#include "4C_adapter_fld_fluid_xfsi.hpp"
#include "4C_adapter_fld_poro.hpp"
#include "4C_elch_input.hpp"
#include "4C_fem_condition_periodic.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_nullspace.hpp"
#include "4C_fluid_implicit_integration.hpp"
#include "4C_fluid_timint_hdg.hpp"
#include "4C_fluid_timint_hdg_weak_comp.hpp"
#include "4C_fluid_timint_loma_bdf2.hpp"
#include "4C_fluid_timint_loma_genalpha.hpp"
#include "4C_fluid_timint_loma_ost.hpp"
#include "4C_fluid_timint_poro_genalpha.hpp"
#include "4C_fluid_timint_poro_ost.hpp"
#include "4C_fluid_timint_poro_stat.hpp"
#include "4C_fluid_timint_red_bdf2.hpp"
#include "4C_fluid_timint_red_genalpha.hpp"
#include "4C_fluid_timint_red_ost.hpp"
#include "4C_fluid_timint_red_stat.hpp"
#include "4C_fluid_timint_stat_hdg.hpp"
#include "4C_fluid_xfluid.hpp"
#include "4C_fluid_xfluid_fluid.hpp"
#include "4C_fsi_input.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_poroelast_input.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Adapter::FluidBaseAlgorithm::FluidBaseAlgorithm(Global::Problem& problem,
    const Teuchos::ParameterList& prbdyn, const Teuchos::ParameterList& fdyn,
    const std::string& disname, bool isale, bool init)
    : problem_(problem)
{
  setup_fluid(prbdyn, fdyn, disname, isale, init);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Adapter::FluidBaseAlgorithm::FluidBaseAlgorithm(Global::Problem& problem,
    const Teuchos::ParameterList& prbdyn, const std::shared_ptr<Core::FE::Discretization> discret)
    : problem_(problem)
{
  setup_inflow_fluid(prbdyn, discret);
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FluidBaseAlgorithm::setup_fluid(const Teuchos::ParameterList& prbdyn,
    const Teuchos::ParameterList& fdyn, const std::string& disname, bool isale, bool init)
{
  auto t = Teuchos::TimeMonitor::getNewTimer("Adapter::FluidBaseAlgorithm::setup_fluid");
  Teuchos::TimeMonitor monitor(*t);
  Global::Problem* problem = &problem_;

  // -------------------------------------------------------------------
  // what's the current problem type?
  // -------------------------------------------------------------------
  Core::ProblemType probtype = problem->get_problem_type();

  // -------------------------------------------------------------------
  // access the discretization
  // -------------------------------------------------------------------
  std::shared_ptr<Core::FE::Discretization> actdis = problem->get_dis(disname);

  // -------------------------------------------------------------------
  // connect degrees of freedom for periodic boundary conditions
  // -------------------------------------------------------------------
  if (probtype != Core::ProblemType::fsi)
  {
    Core::Conditions::PeriodicBoundaryConditions pbc(actdis);
    pbc.update_dofs_for_periodic_boundary_conditions();
  }

  // -------------------------------------------------------------------
  // set degrees of freedom in the discretization
  // -------------------------------------------------------------------
  if (!actdis->have_dofs())
  {
    if (probtype == Core::ProblemType::fsi_xfem or probtype == Core::ProblemType::fluid_xfem or
        (probtype == Core::ProblemType::fpsi_xfem and disname == "fluid"))
    {
      actdis->fill_complete(Core::FE::OptionsFillComplete::none());
    }
    else
    {
      actdis->fill_complete();
    }
  }

  // -------------------------------------------------------------------
  // context for output and restart
  // -------------------------------------------------------------------
  std::shared_ptr<Core::IO::DiscretizationWriter> output = actdis->writer();
  output->write_mesh(0, 0.0);

  // -------------------------------------------------------------------
  // create a solver
  // -------------------------------------------------------------------
  std::shared_ptr<Core::LinAlg::Solver> solver = nullptr;

  switch (Teuchos::getIntegralValue<Inpar::FLUID::MeshTying>(fdyn, "MESHTYING"))
  {
    case Inpar::FLUID::condensed_bmat:
    {
      // FIXME: The solver should not be taken from the contact dynamic section here,
      // but must be specified in the fluid dynamic section instead (popp 11/2012)

      const Teuchos::ParameterList& mshparams = problem->contact_dynamic_params();
      const int mshsolver = mshparams.get<int>(
          "LINEAR_SOLVER");  // meshtying solver (with block preconditioner, e.g. BGS 2x2)

      const auto solvertype = Teuchos::getIntegralValue<Core::LinearSolver::SolverType>(
          problem->solver_params(mshsolver), "SOLVER");

      // create solver objects
      solver = std::make_shared<Core::LinAlg::Solver>(problem->solver_params(mshsolver),
          actdis->get_comm(), problem->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(problem->io_params(), "VERBOSITY"));

      if (solvertype == Core::LinearSolver::SolverType::Belos)
      {
        const int fluidsolver = fdyn.get<int>("LINEAR_SOLVER");           // fluid solver
        const int fluidpressuresolver = fdyn.get<int>("SIMPLER_SOLVER");  // fluid pressure solver
        if (mshsolver == (-1))
          FOUR_C_THROW(
              "no linear solver defined for fluid meshtying problem. Please set LINEAR_SOLVER in "
              "CONTACT DYNAMIC to a valid number!");
        if (fluidsolver == (-1))
          FOUR_C_THROW(
              "no linear solver defined for fluid meshtying problem. Please set LINEAR_SOLVER in "
              "FLUID DYNAMIC to a valid number! This solver is used within block preconditioner "
              "(e.g. BGS2x2) as \"Inverse 1\".");
        if (fluidpressuresolver == (-1))
          FOUR_C_THROW(
              "no linear solver defined for fluid meshtying problem. Please set SIMPLER_SOLVER in "
              "FLUID DYNAMIC to a valid number! This solver is used within block preconditioner "
              "(e.g. BGS2x2) as \"Inverse 2\".");
      }
    }
    break;
    case Inpar::FLUID::condensed_smat:
    case Inpar::FLUID::condensed_bmat_merged:
    {
      // meshtying (no saddle point problem)
      const Teuchos::ParameterList& mshparams = problem->contact_dynamic_params();
      const int mshsolver = mshparams.get<int>(
          "LINEAR_SOLVER");  // meshtying solver (with block preconditioner, e.g. BGS 2x2)
      if (mshsolver == (-1))
        FOUR_C_THROW(
            "no linear solver defined for fluid meshtying problem. Please set LINEAR_SOLVER in "
            "CONTACT DYNAMIC to a valid number!");

      solver = std::make_shared<Core::LinAlg::Solver>(problem->solver_params(mshsolver),
          actdis->get_comm(), problem->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(problem->io_params(), "VERBOSITY"));
    }
    break;
    case Inpar::FLUID::no_meshtying:  // no meshtying -> use FLUID SOLVER
    default:
    {
      // default: create solver using the fluid solver params from FLUID SOLVER block

      // get the solver number used for linear fluid solver
      const int linsolvernumber = fdyn.get<int>("LINEAR_SOLVER");
      if (linsolvernumber == (-1))
        FOUR_C_THROW(
            "no linear solver defined for fluid problem. Please set LINEAR_SOLVER in FLUID DYNAMIC "
            "to a valid number!");
      solver = std::make_shared<Core::LinAlg::Solver>(problem->solver_params(linsolvernumber),
          actdis->get_comm(), problem->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(problem->io_params(), "VERBOSITY"));

      break;
    }
  }

  // compute null space information
  if (probtype != Core::ProblemType::fsi_xfem and probtype != Core::ProblemType::fpsi_xfem and
      probtype != Core::ProblemType::fluid_xfem and
      !(probtype == Core::ProblemType::fsi and
          problem->x_fluid_dynamic_params().sublist("GENERAL").get<bool>("XFLUIDFLUID")))
  {
    compute_null_space_if_necessary(*actdis, solver->params(), true);
  }

  // -------------------------------------------------------------------
  // set parameters in list
  // -------------------------------------------------------------------
  std::shared_ptr<Teuchos::ParameterList> fluidtimeparams =
      std::make_shared<Teuchos::ParameterList>();

  // physical type of fluid flow (incompressible, Boussinesq Approximation, varying density, loma,
  // temperature-dependent water, poro)
  fluidtimeparams->set<Inpar::FLUID::PhysicalType>("Physical Type",
      Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(fdyn, "PHYSICAL_TYPE"));
  // and  check correct setting
  if (probtype == Core::ProblemType::loma and
      (Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(fdyn, "PHYSICAL_TYPE") !=
              Inpar::FLUID::loma and
          Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(fdyn, "PHYSICAL_TYPE") !=
              Inpar::FLUID::tempdepwater))
    FOUR_C_THROW(
        "Input parameter PHYSICAL_TYPE in section FLUID DYNAMIC needs to be 'Loma' or "
        "'Temp_dep_water' for low-Mach-number flow!");
  if ((probtype == Core::ProblemType::thermo_fsi) and
      (Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(fdyn, "PHYSICAL_TYPE") !=
              Inpar::FLUID::loma and
          Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(fdyn, "PHYSICAL_TYPE") !=
              Inpar::FLUID::tempdepwater))
    FOUR_C_THROW(
        "Input parameter PHYSICAL_TYPE in section FLUID DYNAMIC needs to be 'Loma' or "
        "'Temp_dep_water' for Thermo-fluid-structure interaction!");
  if ((probtype == Core::ProblemType::poroelast or probtype == Core::ProblemType::poroscatra or
          probtype == Core::ProblemType::fpsi or probtype == Core::ProblemType::fps3i or
          probtype == Core::ProblemType::fpsi_xfem) and
      disname == "porofluid")
  {
    const Teuchos::ParameterList& pedyn = problem->poroelast_dynamic_params();
    fluidtimeparams->set<Inpar::FLUID::PhysicalType>("Physical Type",
        Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(pedyn, "PHYSICAL_TYPE"));
    if (fluidtimeparams->get<Inpar::FLUID::PhysicalType>("Physical Type") != Inpar::FLUID::poro and
        fluidtimeparams->get<Inpar::FLUID::PhysicalType>("Physical Type") != Inpar::FLUID::poro_p1)
      FOUR_C_THROW(
          "Input parameter PHYSICAL_TYPE in section POROELASTICITY DYNAMIC needs to be 'Poro' or "
          "'Poro_P1' for poro-elasticity!");

    fluidtimeparams->set<PoroElast::TransientEquationsOfPoroFluid>("Transient Terms Poro Fluid",
        Teuchos::getIntegralValue<PoroElast::TransientEquationsOfPoroFluid>(
            pedyn, "TRANSIENT_TERMS"));
  }

  // now, set general parameters required for all problems
  set_general_parameters(fluidtimeparams, prbdyn, fdyn);

  // and, finally, add problem specific parameters

  // for poro problems, use POROUS-FLOW STABILIZATION
  if ((probtype == Core::ProblemType::poroelast or probtype == Core::ProblemType::poroscatra or
          probtype == Core::ProblemType::fpsi or probtype == Core::ProblemType::fps3i or
          probtype == Core::ProblemType::fpsi_xfem) and
      disname == "porofluid")
  {
    fluidtimeparams->sublist("RESIDUAL-BASED STABILIZATION") =
        fdyn.sublist("POROUS-FLOW STABILIZATION");
    fluidtimeparams->sublist("RESIDUAL-BASED STABILIZATION")
        .set<bool>("POROUS-FLOW STABILIZATION", true);
  }

  // add some loma specific parameters
  // get also scatra stabilization sublist
  const Teuchos::ParameterList& lomadyn = problem->loma_control_params();
  fluidtimeparams->sublist("LOMA").set<bool>(
      "update material", lomadyn.get<bool>("SGS_MATERIAL_UPDATE"));

  // ----------------------------- sublist for general xfem-specific parameters
  if (probtype == Core::ProblemType::fluid_xfem or probtype == Core::ProblemType::fsi_xfem or
      (probtype == Core::ProblemType::fpsi_xfem and disname == "fluid") or
      (probtype == Core::ProblemType::fluid_ale and
          problem->x_fluid_dynamic_params().sublist("GENERAL").get<bool>("XFLUIDFLUID")) or
      (probtype == Core::ProblemType::fsi and
          problem->x_fluid_dynamic_params().sublist("GENERAL").get<bool>("XFLUIDFLUID")))
  {
    // get also scatra stabilization sublist
    const Teuchos::ParameterList& xdyn = problem->xfem_general_params();

    fluidtimeparams->sublist("XFEM") = xdyn;
    // ----------------------------- sublist for xfem-specific fluid parameters
    const Teuchos::ParameterList& xfdyn = problem->x_fluid_dynamic_params();

    fluidtimeparams->sublist("XFLUID DYNAMIC/GENERAL") = xfdyn.sublist("GENERAL");
    fluidtimeparams->sublist("XFLUID DYNAMIC/STABILIZATION") = xfdyn.sublist("STABILIZATION");
    fluidtimeparams->sublist("XFLUID DYNAMIC/XFPSI MONOLITHIC") = xfdyn.sublist("XFPSI MONOLITHIC");

    fluidtimeparams->sublist("XFLUID DYNAMIC/GENERAL")
        .set<XFEM::MonolithicXffsiApproach>("MONOLITHIC_XFFSI_APPROACH",
            xfdyn.sublist("GENERAL").get<XFEM::MonolithicXffsiApproach>(
                "MONOLITHIC_XFFSI_APPROACH"));
    fluidtimeparams->sublist("XFLUID DYNAMIC/GENERAL")
        .set<double>("XFLUIDFLUID_SEARCHRADIUS",
            xfdyn.sublist("GENERAL").get<double>("XFLUIDFLUID_SEARCHRADIUS"));
  }

  // -------------------------------------------------------------------
  // additional parameters and algorithm call depending on respective
  // time-integration (or stationary) scheme
  // -------------------------------------------------------------------
  Inpar::FLUID::TimeIntegrationScheme timeint =
      Teuchos::getIntegralValue<Inpar::FLUID::TimeIntegrationScheme>(fdyn, "TIMEINTEGR");

  // sanity checks and default flags
  if (probtype == Core::ProblemType::fsi or probtype == Core::ProblemType::gas_fsi or
      probtype == Core::ProblemType::biofilm_fsi or probtype == Core::ProblemType::thermo_fsi or
      probtype == Core::ProblemType::fsi_xfem or
      (probtype == Core::ProblemType::fpsi_xfem and disname == "fluid") or
      probtype == Core::ProblemType::fsi_redmodels)
  {
    // in case of FSI calculations we do not want a stationary fluid solver
    if (timeint == Inpar::FLUID::timeint_stationary)
      FOUR_C_THROW("Stationary fluid solver not allowed for FSI.");

    const Teuchos::ParameterList& fsidyn = problem->fsi_dynamic_params();
    const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");

    fluidtimeparams->set<bool>("interface second order", fsidyn.get<bool>("SECONDORDER"));
    fluidtimeparams->set<bool>("shape derivatives", fsimono.get<bool>("SHAPEDERIVATIVES"));
  }

  // sanity checks and default flags
  if (probtype == Core::ProblemType::fluid_xfem)
  {
    const Teuchos::ParameterList& fsidyn = problem->fsi_dynamic_params();
    fluidtimeparams->set<bool>("interface second order", fsidyn.get<bool>("SECONDORDER"));
  }

  // sanity checks and default flags
  if (probtype == Core::ProblemType::fsi_xfem or
      (probtype == Core::ProblemType::fpsi_xfem and disname == "fluid"))
  {
    const Teuchos::ParameterList& fsidyn = problem->fsi_dynamic_params();

    const FsiCoupling coupling = Teuchos::getIntegralValue<FsiCoupling>(fsidyn, "COUPALGO");

    if (coupling == fsi_iter_monolithicfluidsplit or coupling == fsi_iter_monolithicstructuresplit)
    {
      // there are a couple of restrictions in monolithic FSI
      FOUR_C_THROW(
          "for XFSI there is no monolithicfluidsplit or monolithicstructuresplit, use "
          "monolithicxfem or any partitioned algorithm instead");
    }
  }

  // sanity checks and default flags
  if (probtype == Core::ProblemType::fluid_xfem or probtype == Core::ProblemType::fsi_xfem or
      (probtype == Core::ProblemType::fpsi_xfem and disname == "fluid"))
  {
    const Teuchos::ParameterList& fsidyn = problem->fsi_dynamic_params();
    const FsiCoupling coupling = Teuchos::getIntegralValue<FsiCoupling>(fsidyn, "COUPALGO");
    fluidtimeparams->set("COUPALGO", coupling);
  }

  if (probtype == Core::ProblemType::elch)
  {
    const Teuchos::ParameterList& fsidyn = problem->fsi_dynamic_params();
    fluidtimeparams->set<bool>("interface second order", fsidyn.get<bool>("SECONDORDER"));
  }

  if (probtype == Core::ProblemType::poroelast or probtype == Core::ProblemType::poroscatra or
      (probtype == Core::ProblemType::fpsi and disname == "porofluid") or
      (probtype == Core::ProblemType::fps3i and disname == "porofluid") or
      (probtype == Core::ProblemType::fpsi_xfem and disname == "porofluid"))
  {
    const Teuchos::ParameterList& porodyn = problem->poroelast_dynamic_params();
    fluidtimeparams->set<bool>("poroelast", true);
    fluidtimeparams->set<bool>("interface second order", porodyn.get<bool>("SECONDORDER"));
    fluidtimeparams->set<bool>("shape derivatives", false);
    fluidtimeparams->set<bool>("conti partial integration", porodyn.get<bool>("CONTIPARTINT"));
    fluidtimeparams->set<bool>("convective term", porodyn.get<bool>("CONVECTIVE_TERM"));
  }
  else if ((probtype == Core::ProblemType::fpsi and disname == "fluid") or
           (probtype == Core::ProblemType::fps3i and disname == "fluid"))
  {
    if (timeint == Inpar::FLUID::timeint_stationary)
      FOUR_C_THROW("Stationary fluid solver not allowed for FPSI.");

    fluidtimeparams->set<bool>("interface second order", prbdyn.get<bool>("SECONDORDER"));
    fluidtimeparams->set<bool>("shape derivatives", prbdyn.get<bool>("SHAPEDERIVATIVES"));
  }

  // =================================================================================
  // Safety Check for usage of DESIGN SURF VOLUMETRIC FLOW CONDITIONS       AN 06/2014
  // =================================================================================
  if (actdis->has_condition("VolumetricSurfaceFlowCond"))
  {
    if (not(Core::ProblemType::fluid_redmodels == probtype or
            Core::ProblemType::fsi_redmodels == probtype))
    {
      FOUR_C_THROW(
          "ERROR: Given Volumetric Womersly infow condition only works with Problemtype "
          "Fluid_RedModels or Fluid_Structure_Interaction_RedModels. \n"
          " --> If you want to use this conditions change Problemtype to Fluid_RedModels or "
          "Fluid_Structure_Interaction_RedModels. \n"
          " --> If you don't want to use this condition comment the respective bcFluid section.");
    }
  }

  // -------------------------------------------------------------------
  // additional parameters and algorithm call depending on respective
  // time-integration (or stationary) scheme
  // -------------------------------------------------------------------
  if (timeint == Inpar::FLUID::timeint_stationary or
      timeint == Inpar::FLUID::timeint_one_step_theta or timeint == Inpar::FLUID::timeint_bdf2 or
      timeint == Inpar::FLUID::timeint_afgenalpha or timeint == Inpar::FLUID::timeint_npgenalpha)
  {
    // -----------------------------------------------------------------
    // set additional parameters in list for
    // one-step-theta/BDF2/af-generalized-alpha/stationary scheme
    // -----------------------------------------------------------------
    // type of time-integration (or stationary) scheme
    fluidtimeparams->set<Inpar::FLUID::TimeIntegrationScheme>("time int algo", timeint);
    // parameter theta for time-integration schemes
    fluidtimeparams->set<double>("theta", fdyn.get<double>("THETA"));
    // number of steps for potential start algorithm
    fluidtimeparams->set<int>("number of start steps", fdyn.get<int>("NUMSTASTEPS"));
    // parameter theta for potential start algorithm
    fluidtimeparams->set<double>("start theta", fdyn.get<double>("START_THETA"));
    // parameter for grid velocity interpolation
    fluidtimeparams->set<Inpar::FLUID::Gridvel>(
        "order gridvel", Teuchos::getIntegralValue<Inpar::FLUID::Gridvel>(fdyn, "GRIDVEL"));
    // handling of pressure and continuity discretization in new one step theta framework
    fluidtimeparams->set<Inpar::FLUID::OstContAndPress>("ost cont and press",
        Teuchos::getIntegralValue<Inpar::FLUID::OstContAndPress>(fdyn, "OST_CONT_PRESS"));
    // flag to switch on the new One Step Theta implementation
    bool ostnew = fdyn.get<bool>("NEW_OST");
    // if the time integration strategy is not even a one step theta strategy, it cannot be the
    // new one step theta strategy either. As it seems, so far there is no sanity check of the
    // input file
    if (timeint != Inpar::FLUID::timeint_one_step_theta and ostnew)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      FOUR_C_THROW(
          "You are not using the One Step Theta Integration Strategy in the Fluid solver,\n"
          "but you set the flag NEW_OST to use the new implementation of the One Step Theta "
          "Strategy. \n"
          "This is impossible. \n"
          "Please change your input file!\n");
#endif
      printf(
          "You are not using the One Step Theta Integration Strategy in the Fluid solver,\n"
          "but you set the flag NEW_OST to use the new implementation of the One Step Theta "
          "Strategy. \n"
          "This is impossible. \n"
          "Please change your input file! In this run, NEW_OST is set to false!\n");
      ostnew = false;
    }
    fluidtimeparams->set<bool>("ost new", ostnew);

    bool dirichletcond = true;
    if (probtype == Core::ProblemType::fsi or probtype == Core::ProblemType::gas_fsi or
        probtype == Core::ProblemType::biofilm_fsi or probtype == Core::ProblemType::thermo_fsi or
        probtype == Core::ProblemType::fsi_redmodels)
    {
      // FSI input parameters
      const Teuchos::ParameterList& fsidyn = problem->fsi_dynamic_params();
      const auto coupling = Teuchos::getIntegralValue<FsiCoupling>(fsidyn, "COUPALGO");
      if (coupling == fsi_iter_monolithicfluidsplit or
          coupling == fsi_iter_monolithicstructuresplit or
          coupling == fsi_iter_mortar_monolithicstructuresplit or
          coupling == fsi_iter_mortar_monolithicfluidsplit or
          coupling == fsi_iter_mortar_monolithicfluidsplit_saddlepoint or
          coupling == fsi_iter_fluidfluid_monolithicstructuresplit or
          coupling == fsi_iter_fluidfluid_monolithicfluidsplit or
          coupling == fsi_iter_fluidfluid_monolithicstructuresplit_nonox or
          coupling == fsi_iter_fluidfluid_monolithicfluidsplit_nonox or
          coupling == fsi_iter_sliding_monolithicfluidsplit or
          coupling == fsi_iter_sliding_monolithicstructuresplit)
      {
        dirichletcond = false;
      }
    }

    if (probtype == Core::ProblemType::poroelast or probtype == Core::ProblemType::poroscatra or
        probtype == Core::ProblemType::fpsi or probtype == Core::ProblemType::fps3i or
        (probtype == Core::ProblemType::fpsi_xfem and disname == "porofluid"))
      dirichletcond = false;

    //------------------------------------------------------------------
    // create all vectors and variables associated with the time
    // integration (call the constructor);
    // the only parameter from the list required here is the number of
    // velocity degrees of freedom

    switch (probtype)
    {
      case Core::ProblemType::fluid:
      case Core::ProblemType::scatra:
      {
        // HDG implements all time stepping schemes within gen-alpha
        if (problem->spatial_approximation_type() == Core::FE::ShapeFunctionType::hdg &&
            timeint != Inpar::FLUID::timeint_stationary &&
            Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(fdyn, "PHYSICAL_TYPE") !=
                Inpar::FLUID::weakly_compressible_dens_mom &&
            Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(fdyn, "PHYSICAL_TYPE") !=
                Inpar::FLUID::weakly_compressible_stokes_dens_mom)
          fluid_ = std::make_shared<FLD::TimIntHDG>(actdis, solver, fluidtimeparams, output, isale);
        else if (problem->spatial_approximation_type() == Core::FE::ShapeFunctionType::hdg &&
                 (Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(fdyn, "PHYSICAL_TYPE") ==
                         Inpar::FLUID::weakly_compressible_dens_mom ||
                     Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(fdyn, "PHYSICAL_TYPE") ==
                         Inpar::FLUID::weakly_compressible_stokes_dens_mom))
          fluid_ = std::make_shared<FLD::TimIntHDGWeakComp>(
              actdis, solver, fluidtimeparams, output, isale);
        else if (problem->spatial_approximation_type() == Core::FE::ShapeFunctionType::hdg &&
                 timeint == Inpar::FLUID::timeint_stationary)
          fluid_ = std::make_shared<FLD::TimIntStationaryHDG>(
              actdis, solver, fluidtimeparams, output, isale);
        else if (timeint == Inpar::FLUID::timeint_stationary)
          fluid_ = std::make_shared<FLD::TimIntStationary>(
              actdis, solver, fluidtimeparams, output, isale);
        else if (timeint == Inpar::FLUID::timeint_one_step_theta)
          fluid_ = std::make_shared<FLD::TimIntOneStepTheta>(
              actdis, solver, fluidtimeparams, output, isale);
        else if (timeint == Inpar::FLUID::timeint_bdf2)
          fluid_ =
              std::make_shared<FLD::TimIntBDF2>(actdis, solver, fluidtimeparams, output, isale);
        else if (timeint == Inpar::FLUID::timeint_afgenalpha or
                 timeint == Inpar::FLUID::timeint_npgenalpha)
          fluid_ =
              std::make_shared<FLD::TimIntGenAlpha>(actdis, solver, fluidtimeparams, output, isale);
        else
          FOUR_C_THROW("Unknown time integration for this fluid problem type\n");
      }
      break;
      case Core::ProblemType::fluid_redmodels:
      {
        if (timeint == Inpar::FLUID::timeint_stationary)
          fluid_ = std::make_shared<FLD::TimIntRedModelsStat>(
              actdis, solver, fluidtimeparams, output, isale);
        else if (timeint == Inpar::FLUID::timeint_one_step_theta)
          fluid_ = std::make_shared<FLD::TimIntRedModelsOst>(
              actdis, solver, fluidtimeparams, output, isale);
        else if (timeint == Inpar::FLUID::timeint_afgenalpha or
                 timeint == Inpar::FLUID::timeint_npgenalpha)
          fluid_ = std::make_shared<FLD::TimIntRedModelsGenAlpha>(
              actdis, solver, fluidtimeparams, output, isale);
        else if (timeint == Inpar::FLUID::timeint_bdf2)
          fluid_ = std::make_shared<FLD::TimIntRedModelsBDF2>(
              actdis, solver, fluidtimeparams, output, isale);
        else
          FOUR_C_THROW("Unknown time integration for this fluid problem type\n");
      }
      break;
      case Core::ProblemType::loma:
      {
        if (Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(fdyn, "PHYSICAL_TYPE") ==
            Inpar::FLUID::tempdepwater)
        {
          if (timeint == Inpar::FLUID::timeint_afgenalpha or
              timeint == Inpar::FLUID::timeint_npgenalpha)
            fluid_ = std::make_shared<FLD::TimIntGenAlpha>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_one_step_theta)
            fluid_ = std::make_shared<FLD::TimIntOneStepTheta>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_bdf2)
            fluid_ =
                std::make_shared<FLD::TimIntBDF2>(actdis, solver, fluidtimeparams, output, isale);
          else
            FOUR_C_THROW("Unknown time integration for this fluid problem type\n");
        }
        else
        {
          if (timeint == Inpar::FLUID::timeint_afgenalpha or
              timeint == Inpar::FLUID::timeint_npgenalpha)
            fluid_ = std::make_shared<FLD::TimIntLomaGenAlpha>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_one_step_theta)
            fluid_ = std::make_shared<FLD::TimIntLomaOst>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_bdf2)
            fluid_ = std::make_shared<FLD::TimIntLomaBDF2>(
                actdis, solver, fluidtimeparams, output, isale);
          else
            FOUR_C_THROW("Unknown time integration for this fluid problem type\n");
        }
      }
      break;
      case Core::ProblemType::fluid_xfem:
      {
        if (problem->x_fluid_dynamic_params().sublist("GENERAL").get<bool>("XFLUIDFLUID"))
        {
          // actdis is the embedded fluid discretization
          std::shared_ptr<Core::FE::Discretization> xfluiddis = problem->get_dis("xfluid");

          std::shared_ptr<FLD::FluidImplicitTimeInt> tmpfluid;
          if (timeint == Inpar::FLUID::timeint_stationary)
            tmpfluid = std::make_shared<FLD::TimIntStationary>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_one_step_theta)
            tmpfluid = std::make_shared<FLD::TimIntOneStepTheta>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_bdf2)
            tmpfluid =
                std::make_shared<FLD::TimIntBDF2>(actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_afgenalpha or
                   timeint == Inpar::FLUID::timeint_npgenalpha)
            tmpfluid = std::make_shared<FLD::TimIntGenAlpha>(
                actdis, solver, fluidtimeparams, output, isale);
          else
            FOUR_C_THROW("Unknown time integration for this fluid problem type\n");

          fluid_ = std::make_shared<FLD::XFluidFluid>(
              tmpfluid, xfluiddis, solver, fluidtimeparams, isale);
          break;
        }

        std::shared_ptr<Core::FE::Discretization> soliddis = problem->get_dis("structure");
        std::shared_ptr<Core::FE::Discretization> scatradis = nullptr;

        if (problem->does_exist_dis("scatra")) scatradis = problem->get_dis("scatra");

        std::shared_ptr<FLD::XFluid> tmpfluid = std::make_shared<FLD::XFluid>(
            actdis, soliddis, scatradis, solver, fluidtimeparams, output, isale);

        std::string condition_name = "";

        // TODO: actually in case of ale fluid with e.g. only level-set we do not want to use the
        // XFluidFSI class since not always
        // a boundary discretization is necessary.
        // however, the xfluid-class itself does not support the full ALE-functionality without the
        // FSI itself ALE-fluid with level-set/without mesh discretization not supported yet
        if (isale)  // in ale case
          fluid_ = std::make_shared<XFluidFSI>(
              tmpfluid, condition_name, solver, fluidtimeparams, output);
        else
          fluid_ = tmpfluid;
      }
      break;
      case Core::ProblemType::fsi_xfem:
      {
        std::string condition_name;

        // FSI input parameters
        const Teuchos::ParameterList& fsidyn = problem->fsi_dynamic_params();
        const auto coupling = Teuchos::getIntegralValue<FsiCoupling>(fsidyn, "COUPALGO");
        if (coupling == fsi_iter_xfem_monolithic)
        {
          condition_name = "XFEMSurfFSIMono";  // not used anymore!
        }
        else if (coupling == fsi_iter_stagg_fixed_rel_param or
                 coupling == fsi_iter_stagg_AITKEN_rel_param or
                 coupling == fsi_iter_stagg_steep_desc or
                 coupling == fsi_iter_stagg_CHEB_rel_param or
                 coupling == fsi_iter_stagg_AITKEN_rel_force or
                 coupling == fsi_iter_stagg_steep_desc_force or
                 coupling == fsi_iter_stagg_steep_desc_force or
                 coupling == fsi_iter_stagg_steep_desc_force)
        {
          condition_name = "XFEMSurfFSIPart";
        }
        else
          FOUR_C_THROW("non supported COUPALGO for FSI");

        std::shared_ptr<Core::FE::Discretization> soliddis = problem->get_dis("structure");
        std::shared_ptr<FLD::XFluid> tmpfluid;
        if (problem->x_fluid_dynamic_params().sublist("GENERAL").get<bool>("XFLUIDFLUID"))
        {
          FOUR_C_THROW(
              "XFLUIDFLUID with XFSI framework not supported via FLD::XFluidFluid but via "
              "FLD::XFluid");

          // actdis is the embedded fluid discretization
          std::shared_ptr<Core::FE::Discretization> xfluiddis = problem->get_dis("xfluid");

          std::shared_ptr<FLD::FluidImplicitTimeInt> tmpfluid_emb;
          if (timeint == Inpar::FLUID::timeint_stationary)
            tmpfluid_emb = std::make_shared<FLD::TimIntStationary>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_one_step_theta)
            tmpfluid_emb = std::make_shared<FLD::TimIntOneStepTheta>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_bdf2)
            tmpfluid_emb =
                std::make_shared<FLD::TimIntBDF2>(actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_afgenalpha or
                   timeint == Inpar::FLUID::timeint_npgenalpha)
            tmpfluid_emb = std::make_shared<FLD::TimIntGenAlpha>(
                actdis, solver, fluidtimeparams, output, isale);
          else
            FOUR_C_THROW("Unknown time integration for this fluid problem type\n");

          tmpfluid = std::make_shared<FLD::XFluidFluid>(
              tmpfluid_emb, xfluiddis, soliddis, solver, fluidtimeparams, isale);
        }
        else
        {
          std::shared_ptr<Core::FE::Discretization> scatradis = nullptr;

          if (problem->does_exist_dis("scatra")) scatradis = problem->get_dis("scatra");

          tmpfluid = std::make_shared<FLD::XFluid>(
              actdis, soliddis, scatradis, solver, fluidtimeparams, output, isale);
        }

        if (coupling == fsi_iter_xfem_monolithic)
          fluid_ = tmpfluid;
        else
          fluid_ = std::make_shared<XFluidFSI>(
              tmpfluid, condition_name, solver, fluidtimeparams, output);
      }
      break;
      case Core::ProblemType::fsi:
      case Core::ProblemType::gas_fsi:
      case Core::ProblemType::biofilm_fsi:
      case Core::ProblemType::fbi:
      case Core::ProblemType::fluid_ale:
      {  //
        std::shared_ptr<FLD::FluidImplicitTimeInt> tmpfluid;
        if (problem->spatial_approximation_type() == Core::FE::ShapeFunctionType::hdg &&
            (Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(fdyn, "PHYSICAL_TYPE") ==
                    Inpar::FLUID::weakly_compressible_dens_mom ||
                Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(fdyn, "PHYSICAL_TYPE") ==
                    Inpar::FLUID::weakly_compressible_stokes_dens_mom))
          tmpfluid = std::make_shared<FLD::TimIntHDGWeakComp>(
              actdis, solver, fluidtimeparams, output, isale);
        else if (timeint == Inpar::FLUID::timeint_stationary)
          tmpfluid = std::make_shared<FLD::TimIntStationary>(
              actdis, solver, fluidtimeparams, output, isale);
        else if (timeint == Inpar::FLUID::timeint_one_step_theta)
          tmpfluid = std::make_shared<FLD::TimIntOneStepTheta>(
              actdis, solver, fluidtimeparams, output, isale);
        else if (timeint == Inpar::FLUID::timeint_bdf2)
          tmpfluid =
              std::make_shared<FLD::TimIntBDF2>(actdis, solver, fluidtimeparams, output, isale);
        else if (timeint == Inpar::FLUID::timeint_afgenalpha or
                 timeint == Inpar::FLUID::timeint_npgenalpha)
          tmpfluid =
              std::make_shared<FLD::TimIntGenAlpha>(actdis, solver, fluidtimeparams, output, isale);
        else
          FOUR_C_THROW("Unknown time integration for this fluid problem type\n");

        const Teuchos::ParameterList& fsidyn = problem->fsi_dynamic_params();
        const auto coupling = Teuchos::getIntegralValue<FsiCoupling>(fsidyn, "COUPALGO");

        if (problem->x_fluid_dynamic_params().sublist("GENERAL").get<bool>("XFLUIDFLUID"))
        {
          fluidtimeparams->set<bool>("shape derivatives", false);
          // actdis is the embedded fluid discretization
          std::shared_ptr<Core::FE::Discretization> xfluiddis = problem->get_dis("xfluid");
          std::shared_ptr<FLD::XFluidFluid> xffluid = std::make_shared<FLD::XFluidFluid>(
              tmpfluid, xfluiddis, solver, fluidtimeparams, false, isale);
          fluid_ = std::make_shared<FluidFluidFSI>(
              *problem, xffluid, tmpfluid, solver, fluidtimeparams, isale, dirichletcond);
        }
        else if (coupling == fsi_iter_sliding_monolithicfluidsplit or
                 coupling == fsi_iter_sliding_monolithicstructuresplit)
          fluid_ = std::make_shared<FluidFSIMsht>(
              *problem, tmpfluid, actdis, solver, fluidtimeparams, output, isale, dirichletcond);
        else if (probtype == Core::ProblemType::fbi)
          fluid_ = std::make_shared<FluidFBI>(
              *problem, tmpfluid, actdis, solver, fluidtimeparams, output, isale, dirichletcond);
        else
          fluid_ = std::make_shared<FluidFSI>(
              *problem, tmpfluid, actdis, solver, fluidtimeparams, output, isale, dirichletcond);
      }
      break;
      case Core::ProblemType::thermo_fsi:
      {
        std::shared_ptr<FLD::FluidImplicitTimeInt> tmpfluid;
        if (Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(fdyn, "PHYSICAL_TYPE") ==
            Inpar::FLUID::tempdepwater)
        {
          if (timeint == Inpar::FLUID::timeint_afgenalpha or
              timeint == Inpar::FLUID::timeint_npgenalpha)
            tmpfluid = std::make_shared<FLD::TimIntGenAlpha>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_one_step_theta)
            tmpfluid = std::make_shared<FLD::TimIntOneStepTheta>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_bdf2)
            tmpfluid =
                std::make_shared<FLD::TimIntBDF2>(actdis, solver, fluidtimeparams, output, isale);
          else
            FOUR_C_THROW("Unknown time integration for this fluid problem type\n");
        }
        else
        {
          if (timeint == Inpar::FLUID::timeint_afgenalpha or
              timeint == Inpar::FLUID::timeint_npgenalpha)
            tmpfluid = std::make_shared<FLD::TimIntLomaGenAlpha>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_one_step_theta)
            tmpfluid = std::make_shared<FLD::TimIntLomaOst>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_bdf2)
            tmpfluid = std::make_shared<FLD::TimIntLomaBDF2>(
                actdis, solver, fluidtimeparams, output, isale);
          else
            FOUR_C_THROW("Unknown time integration for this fluid problem type\n");
        }

        const Teuchos::ParameterList& fsidyn = problem->fsi_dynamic_params();
        const auto coupling = Teuchos::getIntegralValue<FsiCoupling>(fsidyn, "COUPALGO");

        if (coupling == fsi_iter_sliding_monolithicfluidsplit or
            coupling == fsi_iter_sliding_monolithicstructuresplit)
          fluid_ = std::make_shared<FluidFSIMsht>(
              *problem, tmpfluid, actdis, solver, fluidtimeparams, output, isale, dirichletcond);
        else
          fluid_ = std::make_shared<FluidFSI>(
              *problem, tmpfluid, actdis, solver, fluidtimeparams, output, isale, dirichletcond);
      }
      break;
      case Core::ProblemType::fsi_redmodels:
      {  // give a warning
        if (Core::Communication::my_mpi_rank(actdis->get_comm()) == 0)
          std::cout << "\n Warning: FSI_RedModels is little tested. Keep testing! \n" << std::endl;

        // create the fluid time integration object
        std::shared_ptr<FLD::FluidImplicitTimeInt> tmpfluid;
        if (timeint == Inpar::FLUID::timeint_stationary)
          tmpfluid = std::make_shared<FLD::TimIntRedModelsStat>(
              actdis, solver, fluidtimeparams, output, isale);
        else if (timeint == Inpar::FLUID::timeint_one_step_theta)
          tmpfluid = std::make_shared<FLD::TimIntRedModelsOst>(
              actdis, solver, fluidtimeparams, output, isale);
        else if (timeint == Inpar::FLUID::timeint_afgenalpha or
                 timeint == Inpar::FLUID::timeint_npgenalpha)
          tmpfluid = std::make_shared<FLD::TimIntRedModelsGenAlpha>(
              actdis, solver, fluidtimeparams, output, isale);
        else if (timeint == Inpar::FLUID::timeint_bdf2)
          tmpfluid = std::make_shared<FLD::TimIntRedModelsBDF2>(
              actdis, solver, fluidtimeparams, output, isale);
        else
          FOUR_C_THROW("Unknown time integration for this fluid problem type\n");
        fluid_ = std::make_shared<FluidFSI>(
            *problem, tmpfluid, actdis, solver, fluidtimeparams, output, isale, dirichletcond);
      }
      break;
      case Core::ProblemType::poroelast:
      case Core::ProblemType::poroscatra:
      case Core::ProblemType::fpsi:
      case Core::ProblemType::fps3i:
      case Core::ProblemType::fpsi_xfem:
      {
        std::shared_ptr<FLD::FluidImplicitTimeInt> tmpfluid;
        if (disname == "porofluid")
        {
          if (timeint == Inpar::FLUID::timeint_stationary)
            tmpfluid = std::make_shared<FLD::TimIntPoroStat>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_one_step_theta)
            tmpfluid = std::make_shared<FLD::TimIntPoroOst>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_afgenalpha or
                   timeint == Inpar::FLUID::timeint_npgenalpha)
            tmpfluid = std::make_shared<FLD::TimIntPoroGenAlpha>(
                actdis, solver, fluidtimeparams, output, isale);
          else
            FOUR_C_THROW("Unknown time integration for this fluid problem type\n");
          fluid_ = std::make_shared<FluidPoro>(
              *problem, tmpfluid, actdis, solver, fluidtimeparams, output, isale, dirichletcond);
        }
        else if (disname == "fluid")
        {
          if (probtype == Core::ProblemType::fpsi or probtype == Core::ProblemType::fps3i)
          {
            if (timeint == Inpar::FLUID::timeint_stationary)
              tmpfluid = std::make_shared<FLD::TimIntStationary>(
                  actdis, solver, fluidtimeparams, output, isale);
            else if (timeint == Inpar::FLUID::timeint_one_step_theta)
              tmpfluid = std::make_shared<FLD::TimIntOneStepTheta>(
                  actdis, solver, fluidtimeparams, output, isale);
            else
              FOUR_C_THROW("Unknown time integration for this fluid problem type\n");
            fluid_ = std::make_shared<FluidFPSI>(
                *problem, tmpfluid, actdis, solver, fluidtimeparams, output, isale, dirichletcond);
          }
          else if (probtype == Core::ProblemType::fpsi_xfem)
          {
            std::shared_ptr<Core::FE::Discretization> soliddis = problem->get_dis("structure");
            std::shared_ptr<Core::FE::Discretization> scatradis = nullptr;

            if (problem->does_exist_dis("scatra")) scatradis = problem->get_dis("scatra");

            fluid_ = std::make_shared<FLD::XFluid>(
                actdis, soliddis, scatradis, solver, fluidtimeparams, output, isale);
          }
        }
      }
      break;
      case Core::ProblemType::elch:
      {
        // access the problem-specific parameter list
        const Teuchos::ParameterList& elchcontrol = problem->elch_control_params();
        // is ALE needed or not?
        const ElCh::ElchMovingBoundary withale =
            Teuchos::getIntegralValue<ElCh::ElchMovingBoundary>(elchcontrol, "MOVINGBOUNDARY");
        if (withale != ElCh::elch_mov_bndry_no)
        {
          std::shared_ptr<FLD::FluidImplicitTimeInt> tmpfluid;
          if (timeint == Inpar::FLUID::timeint_stationary)
            tmpfluid = std::make_shared<FLD::TimIntStationary>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_one_step_theta)
            tmpfluid = std::make_shared<FLD::TimIntOneStepTheta>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_bdf2)
            tmpfluid =
                std::make_shared<FLD::TimIntBDF2>(actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_afgenalpha or
                   timeint == Inpar::FLUID::timeint_npgenalpha)
            tmpfluid = std::make_shared<FLD::TimIntGenAlpha>(
                actdis, solver, fluidtimeparams, output, isale);
          else
            FOUR_C_THROW("Unknown time integration for this fluid problem type\n");
          fluid_ = std::make_shared<FluidFSI>(
              *problem, tmpfluid, actdis, solver, fluidtimeparams, output, isale, dirichletcond);
        }
        else
        {
          if (timeint == Inpar::FLUID::timeint_stationary)
            fluid_ = std::make_shared<FLD::TimIntStationary>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_one_step_theta)
            fluid_ = std::make_shared<FLD::TimIntOneStepTheta>(
                actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_bdf2)
            fluid_ =
                std::make_shared<FLD::TimIntBDF2>(actdis, solver, fluidtimeparams, output, isale);
          else if (timeint == Inpar::FLUID::timeint_afgenalpha or
                   timeint == Inpar::FLUID::timeint_npgenalpha)
            fluid_ = std::make_shared<FLD::TimIntGenAlpha>(
                actdis, solver, fluidtimeparams, output, isale);
          else
            FOUR_C_THROW("Unknown time integration for this fluid problem type\n");
        }
      }
      break;
      default:
      {
        FOUR_C_THROW("Undefined problem type.");
      }
      break;
    }  // end switch (probtype)
  }
  else
  {
    FOUR_C_THROW("Unknown time integration for fluid\n");
  }

  // initialize algorithm for specific time-integration scheme
  if (init)
  {
    fluid_->init();

    set_initial_flow_field(fdyn);
  }

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FluidBaseAlgorithm::set_initial_flow_field(const Teuchos::ParameterList& fdyn)
{
  // set initial field by given function
  // we do this here, since we have direct access to all necessary parameters
  Inpar::FLUID::InitialField initfield =
      Teuchos::getIntegralValue<Inpar::FLUID::InitialField>(fdyn, "INITIALFIELD");
  if (initfield != Inpar::FLUID::initfield_zero_field)
  {
    int startfuncno = fdyn.get<int>("STARTFUNCNO");
    if (initfield != Inpar::FLUID::initfield_field_by_function and
        initfield != Inpar::FLUID::initfield_disturbed_field_from_function)
    {
      startfuncno = -1;
    }
    fluid_->set_initial_flow_field(initfield, startfuncno);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FluidBaseAlgorithm::set_initial_inflow_field(const Teuchos::ParameterList& fdyn)
{
  // set initial field for inflow section by given function
  // we do this here, since we have direct access to all necessary parameters
  Inpar::FLUID::InitialField initfield = Teuchos::getIntegralValue<Inpar::FLUID::InitialField>(
      fdyn.sublist("TURBULENT INFLOW"), "INITIALINFLOWFIELD");
  if (initfield != Inpar::FLUID::initfield_zero_field)
  {
    int startfuncno = fdyn.sublist("TURBULENT INFLOW").get<int>("INFLOWFUNC");
    if (initfield != Inpar::FLUID::initfield_field_by_function and
        initfield != Inpar::FLUID::initfield_disturbed_field_from_function)
    {
      startfuncno = -1;
    }
    fluid_->set_initial_flow_field(initfield, startfuncno);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FluidBaseAlgorithm::setup_inflow_fluid(
    const Teuchos::ParameterList& prbdyn, const std::shared_ptr<Core::FE::Discretization> discret)
{
  auto t = Teuchos::TimeMonitor::getNewTimer("Adapter::FluidBaseAlgorithm::setup_fluid");
  Teuchos::TimeMonitor monitor(*t);
  Global::Problem* problem = &problem_;

  // -------------------------------------------------------------------
  // what's the current problem type?
  // -------------------------------------------------------------------
  Core::ProblemType probtype = problem->get_problem_type();

  // the inflow computation can only deal with standard fluid problems so far
  // extensions for xfluid, fsi problems have to be added if necessary
  // they should not pose any additional problem
  // meshtying or xfem related parameters are not supported, yet
  if (probtype != Core::ProblemType::fluid)
    FOUR_C_THROW("Only fluid problems supported! Read comment and add your problem type!");

  // -------------------------------------------------------------------
  // set degrees of freedom in the discretization
  // -------------------------------------------------------------------
  if (!discret->have_dofs())
  {
    FOUR_C_THROW("fill_complete shouldn't be necessary!");
  }

  // -------------------------------------------------------------------
  // context for output and restart
  // -------------------------------------------------------------------
  std::shared_ptr<Core::IO::DiscretizationWriter> output = discret->writer();

  // -------------------------------------------------------------------
  // set some pointers and variables
  // -------------------------------------------------------------------
  const Teuchos::ParameterList& fdyn = problem->fluid_dynamic_params();

  // -------------------------------------------------------------------
  // create a solver
  // -------------------------------------------------------------------
  // get the solver number used for linear fluid solver
  const int linsolvernumber = fdyn.get<int>("LINEAR_SOLVER");
  if (linsolvernumber == (-1))
    FOUR_C_THROW(
        "no linear solver defined for fluid problem. Please set LINEAR_SOLVER in FLUID DYNAMIC to "
        "a valid number!");
  std::shared_ptr<Core::LinAlg::Solver> solver =
      std::make_shared<Core::LinAlg::Solver>(problem->solver_params(linsolvernumber),
          discret->get_comm(), problem->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(problem->io_params(), "VERBOSITY"));

  compute_null_space_if_necessary(*discret, solver->params(), true);

  // -------------------------------------------------------------------
  // set parameters in list required for all schemes
  // -------------------------------------------------------------------
  std::shared_ptr<Teuchos::ParameterList> fluidtimeparams =
      std::make_shared<Teuchos::ParameterList>();

  // physical type of fluid flow (incompressible, Boussinesq Approximation, varying density, loma,
  // temperature-dependent water)
  fluidtimeparams->set<Inpar::FLUID::PhysicalType>("Physical Type",
      Teuchos::getIntegralValue<Inpar::FLUID::PhysicalType>(fdyn, "PHYSICAL_TYPE"));

  // now, set general parameters required for all problems
  set_general_parameters(fluidtimeparams, prbdyn, fdyn);

  // overwrite canonical flow parameters by inflow type
  fluidtimeparams->sublist("TURBULENCE MODEL")
      .set<std::string>(
          "CANONICAL_FLOW", fdyn.sublist("TURBULENT INFLOW").get<std::string>("CANONICAL_INFLOW"));
  fluidtimeparams->sublist("TURBULENCE MODEL")
      .set<std::string>(
          "HOMDIR", fdyn.sublist("TURBULENT INFLOW").get<std::string>("INFLOW_HOMDIR"));
  fluidtimeparams->sublist("TURBULENCE MODEL")
      .set<int>(
          "DUMPING_PERIOD", fdyn.sublist("TURBULENT INFLOW").get<int>("INFLOW_DUMPING_PERIOD"));
  fluidtimeparams->sublist("TURBULENCE MODEL")
      .set<int>(
          "SAMPLING_START", fdyn.sublist("TURBULENT INFLOW").get<int>("INFLOW_SAMPLING_START"));
  fluidtimeparams->sublist("TURBULENCE MODEL")
      .set<int>("SAMPLING_STOP", fdyn.sublist("TURBULENT INFLOW").get<int>("INFLOW_SAMPLING_STOP"));
  fluidtimeparams->sublist("TURBULENCE MODEL")
      .set<double>(
          "CHAN_AMPL_INIT_DIST", fdyn.sublist("TURBULENT INFLOW").get<double>("INFLOW_INIT_DIST"));

  // -------------------------------------------------------------------
  // additional parameters and algorithm call depending on respective
  // time-integration (or stationary) scheme
  // -------------------------------------------------------------------
  auto timeint = Teuchos::getIntegralValue<Inpar::FLUID::TimeIntegrationScheme>(fdyn, "TIMEINTEGR");

  // -------------------------------------------------------------------
  // additional parameters and algorithm call depending on respective
  // time-integration (or stationary) scheme
  // -------------------------------------------------------------------
  if (timeint == Inpar::FLUID::timeint_stationary or
      timeint == Inpar::FLUID::timeint_one_step_theta or timeint == Inpar::FLUID::timeint_bdf2 or
      timeint == Inpar::FLUID::timeint_afgenalpha or timeint == Inpar::FLUID::timeint_npgenalpha)
  {
    // -----------------------------------------------------------------
    // set additional parameters in list for
    // one-step-theta/BDF2/af-generalized-alpha/stationary scheme
    // -----------------------------------------------------------------
    // type of time-integration (or stationary) scheme
    fluidtimeparams->set<Inpar::FLUID::TimeIntegrationScheme>("time int algo", timeint);
    // parameter theta for time-integration schemes
    fluidtimeparams->set<double>("theta", fdyn.get<double>("THETA"));
    // number of steps for potential start algorithm
    fluidtimeparams->set<int>("number of start steps", fdyn.get<int>("NUMSTASTEPS"));
    // parameter theta for potential start algorithm
    fluidtimeparams->set<double>("start theta", fdyn.get<double>("START_THETA"));
    // parameter for grid velocity interpolation
    fluidtimeparams->set<Inpar::FLUID::Gridvel>(
        "order gridvel", Teuchos::getIntegralValue<Inpar::FLUID::Gridvel>(fdyn, "GRIDVEL"));
    // handling of pressure and continuity discretization in new one step theta framework
    fluidtimeparams->set<Inpar::FLUID::OstContAndPress>("ost cont and press",
        Teuchos::getIntegralValue<Inpar::FLUID::OstContAndPress>(fdyn, "OST_CONT_PRESS"));
    // flag to switch on the new One Step Theta implementation
    bool ostnew = fdyn.get<bool>("NEW_OST");
    // if the time integration strategy is not even a one step theta strategy, it cannot be the
    // new one step theta strategy either. As it seems, so far there is no sanity check of the
    // input file
    if (timeint != Inpar::FLUID::timeint_one_step_theta and ostnew)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      FOUR_C_THROW(
          "You are not using the One Step Theta Integration Strategy in the Fluid solver,\n"
          "but you set the flag NEW_OST to use the new implementation of the One Step Theta "
          "Strategy. \n"
          "This is impossible. \n"
          "Please change your input file!\n");
#endif
      printf(
          "You are not using the One Step Theta Integration Strategy in the Fluid solver,\n"
          "but you set the flag NEW_OST to use the new implementation of the One Step Theta "
          "Strategy. \n"
          "This is impossible. \n"
          "Please change your input file! In this run, NEW_OST is set to false!\n");
      ostnew = false;
    }
    fluidtimeparams->set<bool>("ost new", ostnew);

    //------------------------------------------------------------------
    // create all vectors and variables associated with the time
    // integration (call the constructor);
    // the only parameter from the list required here is the number of
    // velocity degrees of freedom
    //    fluid_ = Teuchos::rcp(new FLD::FluidImplicitTimeInt(discret, solver, fluidtimeparams,
    //    output, false));
    if (timeint == Inpar::FLUID::timeint_stationary)
      fluid_ =
          std::make_shared<FLD::TimIntStationary>(discret, solver, fluidtimeparams, output, false);
    else if (timeint == Inpar::FLUID::timeint_one_step_theta)
      fluid_ = std::make_shared<FLD::TimIntOneStepTheta>(
          discret, solver, fluidtimeparams, output, false);
    else if (timeint == Inpar::FLUID::timeint_bdf2)
      fluid_ = std::make_shared<FLD::TimIntBDF2>(discret, solver, fluidtimeparams, output, false);
    else if (timeint == Inpar::FLUID::timeint_afgenalpha or
             timeint == Inpar::FLUID::timeint_npgenalpha)
      fluid_ =
          std::make_shared<FLD::TimIntGenAlpha>(discret, solver, fluidtimeparams, output, false);
  }
  else
  {
    FOUR_C_THROW("Unknown time integration for fluid\n");
  }

  // initialize algorithm for specific time-integration scheme
  fluid_->init();

  set_initial_inflow_field(fdyn);

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FluidBaseAlgorithm::set_general_parameters(
    const std::shared_ptr<Teuchos::ParameterList> fluidtimeparams,
    const Teuchos::ParameterList& prbdyn, const Teuchos::ParameterList& fdyn)
{
  Global::Problem* problem = &problem_;
  fluidtimeparams->set<bool>("BLOCKMATRIX", fdyn.get<bool>("BLOCKMATRIX"));

  // -------------------------------------- number of degrees of freedom
  // number of degrees of freedom
  const int ndim = problem->n_dim();
  fluidtimeparams->set<int>("number of velocity degrees of freedom", ndim);

  // -------------------------------------------------- time integration
  // note: here, the values are taken out of the problem-dependent ParameterList prbdyn
  // (which also can be fluiddyn itself!)

  // the default time step size
  fluidtimeparams->set<double>("time step size", prbdyn.get<double>("TIMESTEP"));
  // maximum simulation time
  fluidtimeparams->set<double>("total time", prbdyn.get<double>("MAXTIME"));
  // maximum number of timesteps
  fluidtimeparams->set<int>("max number timesteps", prbdyn.get<int>("NUMSTEP"));
  // sublist for adaptive time stepping
  fluidtimeparams->sublist("TIMEADAPTIVITY") = fdyn.sublist("TIMEADAPTIVITY");

  // -------- additional parameters in list for generalized-alpha scheme
  // parameter alpha_M
  fluidtimeparams->set<double>("alpha_M", fdyn.get<double>("ALPHA_M"));
  // parameter alpha_F
  fluidtimeparams->set<double>("alpha_F", fdyn.get<double>("ALPHA_F"));
  // parameter gamma
  fluidtimeparams->set<double>("gamma", fdyn.get<double>("GAMMA"));

  // ---------------------------------------------- nonlinear iteration
  // type of predictor
  fluidtimeparams->set<std::string>("predictor", fdyn.get<std::string>("PREDICTOR"));
  // set linearisation scheme
  fluidtimeparams->set<Inpar::FLUID::LinearisationAction>("Linearisation",
      Teuchos::getIntegralValue<Inpar::FLUID::LinearisationAction>(fdyn, "NONLINITER"));
  // maximum number of nonlinear iteration steps
  fluidtimeparams->set<int>("max nonlin iter steps", fdyn.get<int>("ITEMAX"));
  // maximum number of nonlinear iteration steps for initial stationary solution
  fluidtimeparams->set<int>("max nonlin iter steps init stat sol", fdyn.get<int>("INITSTATITEMAX"));

  // parameter list containing the nonlinear solver tolerances
  const Teuchos::ParameterList& nonlinsolvertolerances =
      fdyn.sublist("NONLINEAR SOLVER TOLERANCES");

  // stop nonlinear iteration when the velocity residual is below this tolerance
  fluidtimeparams->set<double>(
      "velocity residual tolerance", nonlinsolvertolerances.get<double>("TOL_VEL_RES"));
  // stop nonlinear iteration when the pressure residual is below this tolerance
  fluidtimeparams->set<double>(
      "pressure residual tolerance", nonlinsolvertolerances.get<double>("TOL_PRES_RES"));
  // stop nonlinear iteration when the relative velocity increment is below this tolerance
  fluidtimeparams->set<double>(
      "velocity increment tolerance", nonlinsolvertolerances.get<double>("TOL_VEL_INC"));
  // stop nonlinear iteration when the relative pressure increment is below this tolerance
  fluidtimeparams->set<double>(
      "pressure increment tolerance", nonlinsolvertolerances.get<double>("TOL_PRES_INC"));

  // set convergence check
  fluidtimeparams->set<Inpar::FLUID::ItNorm>(
      "CONVCHECK", fdyn.get<Inpar::FLUID::ItNorm>("CONVCHECK"));
  // set recomputation of residual after solution has convergenced
  fluidtimeparams->set<bool>("INCONSISTENT_RESIDUAL", fdyn.get<bool>("INCONSISTENT_RESIDUAL"));
  // set solver for L2 projection of gradients for reconstruction of consistent residual
  fluidtimeparams->set<int>("VELGRAD_PROJ_SOLVER", fdyn.get<int>("VELGRAD_PROJ_SOLVER"));
  // set adaptive linear solver tolerance
  fluidtimeparams->set<bool>("ADAPTCONV", fdyn.get<bool>("ADAPTCONV"));
  fluidtimeparams->set<double>("ADAPTCONV_BETTER", fdyn.get<double>("ADAPTCONV_BETTER"));
  fluidtimeparams->set<bool>("INFNORMSCALING", (fdyn.get<bool>("INFNORMSCALING")));

  // ----------------------------------------------- restart and output
  const Teuchos::ParameterList& ioflags = problem->io_params();
  // restart
  fluidtimeparams->set<int>("write restart every", prbdyn.get<int>("RESTARTEVERY"));
  // solution output
  fluidtimeparams->set<int>("write solution every", prbdyn.get<int>("RESULTSEVERY"));
  // flag for writing stresses
  fluidtimeparams->set<bool>("write stresses", ioflags.get<bool>("FLUID_STRESS"));
  // flag for writing wall shear stress
  fluidtimeparams->set<bool>(
      "write wall shear stresses", ioflags.get<bool>("FLUID_WALL_SHEAR_STRESS"));
  // flag for writing element data in every step and not only once (i.e. at step == 0 or step ==
  // upres)
  fluidtimeparams->set<bool>(
      "write element data in every step", ioflags.get<bool>("FLUID_ELEDATA_EVERY_STEP"));
  // flag for writing node data in the first time step
  fluidtimeparams->set<bool>(
      "write node data in first step", ioflags.get<bool>("FLUID_NODEDATA_FIRST_STEP"));
  // flag for writing fluid field to gmsh
  if (not problem->io_params().get<bool>("OUTPUT_GMSH"))
  {
    fluidtimeparams->set<bool>("GMSH_OUTPUT", false);
    if (fdyn.get<bool>("GMSH_OUTPUT"))
    {
      std::cout << "WARNING! Conflicting GMSH parameter in IO and fluid sections. No GMSH output "
                   "is written!"
                << std::endl;
    }
  }
  else
    fluidtimeparams->set<bool>("GMSH_OUTPUT", fdyn.get<bool>("GMSH_OUTPUT"));
  // flag for computing divergence
  fluidtimeparams->set<bool>("COMPUTE_DIVU", fdyn.get<bool>("COMPUTE_DIVU"));
  // flag for computing kinetix energy
  fluidtimeparams->set<bool>("COMPUTE_EKIN", fdyn.get<bool>("COMPUTE_EKIN"));
  // flag for computing lift and drag values
  fluidtimeparams->set<bool>("LIFTDRAG", fdyn.get<bool>("LIFTDRAG"));

  // -------------------------------------------------- Oseen advection
  // set function number of given Oseen advective field
  fluidtimeparams->set<int>("OSEENFIELDFUNCNO", fdyn.get<int>("OSEENFIELDFUNCNO"));

  // ---------------------------------------------------- lift and drag
  fluidtimeparams->set<bool>("liftdrag", fdyn.get<bool>("LIFTDRAG"));

  // -----------evaluate error for test flows with analytical solutions
  auto initfield = Teuchos::getIntegralValue<Inpar::FLUID::InitialField>(fdyn, "INITIALFIELD");
  fluidtimeparams->set<int>("eval err for analyt sol", initfield);

  // ------------------------------------------ form of convective term
  fluidtimeparams->set<std::string>("form of convective term", fdyn.get<std::string>("CONVFORM"));

  // -------------------------- potential nonlinear boundary conditions
  fluidtimeparams->set("Nonlinear boundary conditions", fdyn.get<bool>("NONLINEARBC"));

  // ------------------------------------ potential reduced_D 3D coupling method
  fluidtimeparams->set("Strong 3D_redD coupling", fdyn.get<bool>("STRONG_REDD_3D_COUPLING_TYPE"));

  //--------------------------------------  mesh tying for fluid
  fluidtimeparams->set<Inpar::FLUID::MeshTying>(
      "MESHTYING", Teuchos::getIntegralValue<Inpar::FLUID::MeshTying>(fdyn, "MESHTYING"));

  fluidtimeparams->set<bool>("ALLDOFCOUPLED", fdyn.get<bool>("ALLDOFCOUPLED"));

  //--------------------------------------analytical error evaluation
  fluidtimeparams->set<Inpar::FLUID::CalcError>(
      "calculate error", Teuchos::getIntegralValue<Inpar::FLUID::CalcError>(fdyn, "CALCERROR"));
  fluidtimeparams->set<int>("error function number", fdyn.get<int>("CALCERRORFUNCNO"));

  // -----------------------sublist containing stabilization parameters
  fluidtimeparams->sublist("RESIDUAL-BASED STABILIZATION") =
      fdyn.sublist("RESIDUAL-BASED STABILIZATION");
  fluidtimeparams->sublist("EDGE-BASED STABILIZATION") = fdyn.sublist("EDGE-BASED STABILIZATION");

  // -----------------------------get also scatra stabilization sublist
  const Teuchos::ParameterList& scatradyn = problem->scalar_transport_dynamic_params();
  fluidtimeparams->sublist("SCATRA STABILIZATION") = scatradyn.sublist("STABILIZATION");

  // --------------------------sublist containing turbulence parameters
  {
    fluidtimeparams->sublist("TURBULENCE MODEL") = fdyn.sublist("TURBULENCE MODEL");
    fluidtimeparams->sublist("SUBGRID VISCOSITY") = fdyn.sublist("SUBGRID VISCOSITY");
    fluidtimeparams->sublist("MULTIFRACTAL SUBGRID SCALES") =
        fdyn.sublist("MULTIFRACTAL SUBGRID SCALES");
    fluidtimeparams->sublist("TURBULENT INFLOW") = fdyn.sublist("TURBULENT INFLOW");
    fluidtimeparams->sublist("WALL MODEL") = fdyn.sublist("WALL MODEL");

    fluidtimeparams->sublist("TURBULENCE MODEL")
        .set<std::string>("statistics outfile", problem->output_control_file()->file_name());
  }

  // ---------------------------parallel evaluation
  fluidtimeparams->set<bool>("OFF_PROC_ASSEMBLY", fdyn.get<bool>("OFF_PROC_ASSEMBLY"));

  return;
}

FOUR_C_NAMESPACE_CLOSE
