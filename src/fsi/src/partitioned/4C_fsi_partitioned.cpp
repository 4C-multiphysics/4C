// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_partitioned.hpp"

#include "4C_adapter_ale_fluid.hpp"
#include "4C_adapter_fld_fbi_movingboundary.hpp"
#include "4C_adapter_fld_fluid.hpp"
#include "4C_adapter_fld_fluid_xfem.hpp"
#include "4C_adapter_fld_fluid_xfsi.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_mortar.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fluid_xfluid.hpp"
#include "4C_fsi_nox_aitken.hpp"
#include "4C_fsi_nox_fixpoint.hpp"
#include "4C_fsi_nox_jacobian.hpp"
#include "4C_fsi_nox_linearsystem_gcr.hpp"
#include "4C_fsi_nox_mpe.hpp"
#include "4C_fsi_nox_sd.hpp"
#include "4C_fsi_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_io_control.hpp"
#include "4C_solver_nonlin_nox_group_base.hpp"
#include "4C_structure_aux.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <string>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::Partitioned::Partitioned(MPI_Comm comm)
    : Algorithm(comm),
      idispn_(nullptr),
      iveln_(nullptr),
      raw_graph_(nullptr),
      counter_(7),
      mfresitemax_(0),
      coupsfm_(nullptr),
      matchingnodes_(false)
{
  // empty constructor
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Partitioned::setup()
{
  // call setup of base class
  FSI::Algorithm::setup();

  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  set_default_parameters(fsidyn, noxparameterlist_);
  setup_coupling(fsidyn, get_comm());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Partitioned::setup_coupling(const Teuchos::ParameterList& fsidyn, MPI_Comm comm)
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "\n setup_coupling in FSI::Partitioned ..." << std::endl;

  Coupling::Adapter::Coupling& coupsf = structure_fluid_coupling();
  coupsfm_ = std::make_shared<Coupling::Adapter::CouplingMortar>(
      Global::Problem::instance()->n_dim(), Global::Problem::instance()->mortar_coupling_params(),
      Global::Problem::instance()->contact_dynamic_params(),
      Global::Problem::instance()->spatial_approximation_type());


  if (fsidyn.sublist("PARTITIONED SOLVER").get<std::string>("COUPMETHOD") == "conforming" and
      (Global::Problem::instance()->get_problem_type() != Core::ProblemType::fsi_xfem) and
      (Global::Problem::instance()->get_problem_type() != Core::ProblemType::fbi))
  {
    matchingnodes_ = true;
    const int ndim = Global::Problem::instance()->n_dim();
    coupsf.setup_condition_coupling(*structure_field()->discretization(),
        structure_field()->interface()->fsi_cond_map(), *mb_fluid_field()->discretization(),
        mb_fluid_field()->interface()->fsi_cond_map(), "FSICoupling", ndim);

    if (coupsf.master_dof_map()->num_global_elements() == 0)
      FOUR_C_THROW("No nodes in matching FSI interface. Empty FSI coupling condition?");
  }
  else if ((fsidyn.sublist("PARTITIONED SOLVER").get<std::string>("COUPMETHOD") == "conforming") and
           (Global::Problem::instance()->get_problem_type() == Core::ProblemType::fsi_xfem) and
           (Global::Problem::instance()->get_problem_type() != Core::ProblemType::fbi))
  {
    // matching between structure and boundary dis! non-matching between boundary dis and fluid is
    // handled bei XFluid itself
    matchingnodes_ = true;
    const int ndim = Global::Problem::instance()->n_dim();

    std::shared_ptr<Adapter::FluidXFEM> x_movingboundary =
        std::dynamic_pointer_cast<Adapter::FluidXFEM>(mb_fluid_field());
    coupsf.setup_condition_coupling(*structure_field()->discretization(),
        structure_field()->interface()->fsi_cond_map(),
        *x_movingboundary->boundary_discretization(),  // use the matching boundary discretization
        x_movingboundary->struct_interface()->fsi_cond_map(), "FSICoupling", ndim);

    if (coupsf.master_dof_map()->num_global_elements() == 0)
      FOUR_C_THROW("No nodes in matching FSI interface. Empty FSI coupling condition?");
  }
  else if ((Global::Problem::instance()->get_problem_type() == Core::ProblemType::fbi))
  {
    matchingnodes_ = true;
  }
  else if (fsidyn.sublist("PARTITIONED SOLVER").get<std::string>("COUPMETHOD") == "mortar" and
           (Global::Problem::instance()->get_problem_type() != Core::ProblemType::fsi_xfem))
  {
    // coupling condition at the fsi interface: displacements (=number of spatial dimensions) are
    // coupled e.g.: 3D: coupleddof = [1, 1, 1]
    std::vector<int> coupleddof(Global::Problem::instance()->n_dim(), 1);

    matchingnodes_ = false;
    coupsfm_->setup(structure_field()->discretization(), mb_fluid_field()->discretization(),
        (std::dynamic_pointer_cast<Adapter::FluidAle>(mb_fluid_field()))
            ->ale_field()
            ->write_access_discretization(),
        coupleddof, "FSICoupling", comm, Global::Problem::instance()->function_manager(),
        Global::Problem::instance()->binning_strategy_params(),
        Global::Problem::instance()->discretization_map(),
        Global::Problem::instance()->output_control_file(),
        Global::Problem::instance()->spatial_approximation_type(), true);
  }
  else
  {
    FOUR_C_THROW("You should not arrive here");
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Partitioned::set_default_parameters(
    const Teuchos::ParameterList& fsidyn, Teuchos::ParameterList& list)
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "\n set_default_parameters in FSI::Partitioned ..." << std::endl;

  // extract sublist with settings for partitioned solver
  const Teuchos::ParameterList& fsipart = fsidyn.sublist("PARTITIONED SOLVER");

  // Get the top level parameter list
  Teuchos::ParameterList& nlParams = list;

  nlParams.set("Nonlinear Solver", "Line Search Based");
  nlParams.set("Preconditioner", "None");
  nlParams.set("Norm abs F", fsipart.get<double>("CONVTOL"));
  nlParams.set("Max Iterations", fsipart.get<int>("ITEMAX"));

  // sublists

  Teuchos::ParameterList& dirParams = nlParams.sublist("Direction");
  Teuchos::ParameterList& lineSearchParams = nlParams.sublist("Line Search");

  //
  // Set parameters for NOX to chose the solver direction and line
  // search step.
  //

  switch (Teuchos::getIntegralValue<FsiCoupling>(fsidyn, "COUPALGO"))
  {
    case fsi_iter_stagg_fixed_rel_param:
    {
      // fixed-point solver with fixed relaxation parameter
      set_method("ITERATIVE STAGGERED SCHEME WITH FIXED RELAXATION PARAMETER");

      nlParams.set("Jacobian", "None");

      dirParams.set("Method", "User Defined");
      Teuchos::RCP<::NOX::Direction::UserDefinedFactory> fixpointfactory =
          Teuchos::make_rcp<NOX::FSI::FixPointFactory>();
      dirParams.set("User Defined Direction Factory", fixpointfactory);

      // Teuchos::ParameterList& lsParams = newtonParams.sublist("Linear Solver");
      // lsParams.set("Preconditioner","None");

      lineSearchParams.set("Method", "Full Step");
      lineSearchParams.sublist("Full Step").set("Full Step", fsipart.get<double>("RELAX"));
      break;
    }
    case fsi_iter_stagg_AITKEN_rel_param:
    {
      // fixed-point solver with Aitken relaxation parameter
      set_method("ITERATIVE STAGGERED SCHEME WITH RELAXATION PARAMETER VIA AITKEN ITERATION");

      nlParams.set("Jacobian", "None");

      dirParams.set("Method", "User Defined");
      Teuchos::RCP<::NOX::Direction::UserDefinedFactory> fixpointfactory =
          Teuchos::make_rcp<NOX::FSI::FixPointFactory>();
      dirParams.set("User Defined Direction Factory", fixpointfactory);

      Teuchos::RCP<::NOX::LineSearch::UserDefinedFactory> linesearchfactory =
          Teuchos::make_rcp<NOX::FSI::AitkenFactory>();
      lineSearchParams.set("Method", "User Defined");
      lineSearchParams.set("User Defined Line Search Factory", linesearchfactory);

      lineSearchParams.sublist("Aitken").set("max step size", fsipart.get<double>("MAXOMEGA"));
      lineSearchParams.sublist("Aitken").set("min step size", fsipart.get<double>("MINOMEGA"));
      break;
    }
    case fsi_iter_stagg_steep_desc:
    {
      // fixed-point solver with steepest descent relaxation parameter
      set_method(
          "ITERATIVE STAGGERED SCHEME WITH RELAXATION PARAMETER VIA STEEPEST DESCENT METHOD");

      nlParams.set("Jacobian", "None");

      dirParams.set("Method", "User Defined");
      Teuchos::RCP<::NOX::Direction::UserDefinedFactory> fixpointfactory =
          Teuchos::make_rcp<NOX::FSI::FixPointFactory>();
      dirParams.set("User Defined Direction Factory", fixpointfactory);

      Teuchos::RCP<::NOX::LineSearch::UserDefinedFactory> linesearchfactory =
          Teuchos::make_rcp<NOX::FSI::SDFactory>();
      lineSearchParams.set("Method", "User Defined");
      lineSearchParams.set("User Defined Line Search Factory", linesearchfactory);
      break;
    }
    case fsi_iter_stagg_NLCG:
    {
      // nonlinear CG solver (pretty much steepest descent with finite
      // difference Jacobian)
      set_method("ITERATIVE STAGGERED SCHEME WITH NONLINEAR CG SOLVER");

      nlParams.set("Jacobian", "None");
      dirParams.set("Method", "NonlinearCG");
      lineSearchParams.set("Method", "NonlinearCG");
      break;
    }
    case fsi_iter_stagg_MFNK_FD:
    {
      // matrix free Newton Krylov with finite difference Jacobian
      set_method("MATRIX FREE NEWTON KRYLOV SOLVER BASED ON FINITE DIFFERENCES");

      nlParams.set("Jacobian", "Matrix Free");

      Teuchos::ParameterList& mfParams = nlParams.sublist("Matrix Free");
      mfParams.set("lambda", 1.0e-4);
      mfParams.set("itemax", 1);
      mfParams.set("Kelley Perturbation", false);

      lineSearchParams.set("Method", "Full Step");
      lineSearchParams.sublist("Full Step").set("Full Step", 1.0);

      Teuchos::ParameterList& dirParams = nlParams.sublist("Direction");
      Teuchos::ParameterList& newtonParams = dirParams.sublist(dirParams.get("Method", "Newton"));
      Teuchos::ParameterList& lsParams = newtonParams.sublist("Linear Solver");

      lsParams.set("Tolerance", fsipart.get<double>("BASETOL"));

      break;
    }
    case fsi_iter_stagg_MFNK_FSI:
    {
      // matrix free Newton Krylov with FSI specific Jacobian
      set_method("MATRIX FREE NEWTON KRYLOV SOLVER BASED ON FSI SPECIFIC JACOBIAN APPROXIMATION");

      nlParams.set("Jacobian", "FSI Matrix Free");

      lineSearchParams.set("Method", "Full Step");
      lineSearchParams.sublist("Full Step").set("Full Step", 1.0);

      Teuchos::ParameterList& dirParams = nlParams.sublist("Direction");
      Teuchos::ParameterList& newtonParams = dirParams.sublist(dirParams.get("Method", "Newton"));
      Teuchos::ParameterList& lsParams = newtonParams.sublist("Linear Solver");

      lsParams.set("Tolerance", fsipart.get<double>("BASETOL"));

      break;
    }
    case fsi_iter_stagg_MPE:
    {
      // minimal polynomial extrapolation
      set_method("ITERATIVE STAGGERED SCHEME WITH MINIMAL POLYNOMIAL EXTRAPOLATION");

      nlParams.set("Jacobian", "None");
      dirParams.set("Method", "User Defined");

      Teuchos::RCP<::NOX::Direction::UserDefinedFactory> factory =
          Teuchos::make_rcp<NOX::FSI::MinimalPolynomialFactory>();
      dirParams.set("User Defined Direction Factory", factory);

      Teuchos::ParameterList& exParams = dirParams.sublist("Extrapolation");
      exParams.set("Tolerance", fsipart.get<double>("BASETOL"));
      exParams.set("omega", fsipart.get<double>("RELAX"));
      exParams.set("kmax", 25);
      exParams.set("Method", "MPE");

      // lsParams.set("Preconditioner","None");

      lineSearchParams.set("Method", "Full Step");
      lineSearchParams.sublist("Full Step").set("Full Step", 1.0);
      break;
    }
    default:
    {
      FOUR_C_THROW("Coupling method type {} unsupported",
          Teuchos::getIntegralValue<FsiCoupling>(fsidyn, "COUPALGO"));
    }
  }

  Teuchos::ParameterList& printParams = nlParams.sublist("Printing");
  printParams.set("MyPID", Core::Communication::my_mpi_rank(get_comm()));

  // set default output flag to no output
  // The field solver will output a lot, anyway.
  printParams.get("Output Information",
      ::NOX::Utils::Warning | ::NOX::Utils::OuterIteration | ::NOX::Utils::OuterIterationStatusTest
      // ::NOX::Utils::Parameters
  );

  Teuchos::ParameterList& solverOptions = nlParams.sublist("Solver Options");
  solverOptions.set<std::string>("Status Test Check Type", "Complete");
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Partitioned::timeloop(const Teuchos::RCP<::NOX::Epetra::Interface::Required>& interface)
{
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();

  // Get the top level parameter list
  Teuchos::ParameterList& nlParams = noxparameterlist_;

  // sublists

  Teuchos::ParameterList& dirParams = nlParams.sublist("Direction");
  Teuchos::ParameterList& newtonParams = dirParams.sublist(dirParams.get("Method", "Newton"));
  Teuchos::ParameterList& lsParams = newtonParams.sublist("Linear Solver");

  Teuchos::ParameterList& printParams = nlParams.sublist("Printing");

  // Create printing utilities
  utils_ = std::make_shared<::NOX::Utils>(printParams);

  // ==================================================================

  // log solver iterations

  std::shared_ptr<std::ofstream> log;
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    std::string s = Global::Problem::instance()->output_control_file()->file_name();
    s.append(".iteration");
    log = std::make_shared<std::ofstream>(s.c_str());
    (*log) << "# num procs      = " << Core::Communication::num_mpi_ranks(get_comm()) << "\n"
           << "# Method         = " << nlParams.sublist("Direction").get("Method", "Newton") << "\n"
           << "# Jacobian       = " << nlParams.get("Jacobian", "None") << "\n"
           << "# Preconditioner = " << nlParams.get("Preconditioner", "None") << "\n"
           << "# Line Search    = " << nlParams.sublist("Line Search").get("Method", "Aitken")
           << "\n"
           << "# Predictor      = '"
           << fsidyn.sublist("PARTITIONED SOLVER").get<std::string>("PREDICTOR") << "'\n"
           << "#\n"
           << "# step | time | time/step | #nliter  |R|  #liter  Residual  Jac  Prec  FD_Res  "
              "MF_Res  MF_Jac  User\n";
  }

  // get an idea of interface displacement
  extract_previous_interface_solution();

  Teuchos::Time timer("time step timer");

  // ==================================================================

  while (not_finished())
  {
    // Increment all field counters and predict field values whenever
    // appropriate.
    prepare_time_step();

    // reset all counters
    std::fill(counter_.begin(), counter_.end(), 0);
    lsParams.sublist("Output").set("Total Number of Linear Iterations", 0);
    linsolvcount_.resize(0);

    // start time measurement
    std::shared_ptr<Teuchos::TimeMonitor> timemonitor =
        std::make_shared<Teuchos::TimeMonitor>(timer, true);

    /*----------------- CSD - predictor for itnum==0 --------------------*/

    // Begin Nonlinear Solver ************************************

    // Get initial guess
    std::shared_ptr<Core::LinAlg::Vector<double>> soln = initial_guess();

    ::NOX::Epetra::Vector noxSoln(
        Teuchos::rcpFromRef(soln->get_ref_of_epetra_vector()), ::NOX::Epetra::Vector::CreateView);

    // Create the linear system
    Teuchos::RCP<NOX::Nln::LinearSystemBase> linSys =
        create_linear_system(nlParams, interface, noxSoln, *utils_);

    // Create the Group
    Teuchos::RCP<::NOX::Abstract::Group> grp =
        Teuchos::make_rcp<NOX::Nln::GroupBase>(printParams, interface, noxSoln, linSys);

    // Convergence Tests
    Teuchos::RCP<::NOX::StatusTest::Combo> combo = create_status_test(nlParams, grp);

    // Create the solver
    Teuchos::RCP<::NOX::Solver::Generic> solver = ::NOX::Solver::buildSolver(
        grp, combo, Teuchos::RCP<Teuchos::ParameterList>(&nlParams, false));

    // solve the whole thing
    ::NOX::StatusTest::StatusType status = solver->solve();

    if (status != ::NOX::StatusTest::Converged)
      FOUR_C_THROW("Nonlinear solver failed to converge!");

    // End Nonlinear Solver **************************************

    // Output the parameter list
    if (utils_->isPrintType(::NOX::Utils::Parameters))
      if (step() == 1 and Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        utils_->out() << std::endl
                      << "Final Parameters" << std::endl
                      << "****************" << std::endl;
        solver->getList().print(utils_->out());
        utils_->out() << std::endl;
      }

    // ==================================================================

    // stop time measurement
    timemonitor = nullptr;

    if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    {
      (*log) << step() << "\t" << time() << "\t" << timer.totalElapsedTime(true) << "\t"
             << nlParams.sublist("Output").get("Nonlinear Iterations", 0) << "\t"
             << nlParams.sublist("Output").get("2-Norm of Residual", 0.) << "\t"
             << lsParams.sublist("Output").get("Total Number of Linear Iterations", 0);
      for (std::vector<int>::size_type i = 0; i < counter_.size(); ++i)
      {
        (*log) << " " << counter_[i];
      }
      (*log) << std::endl;
      log->flush();
    }

    // ==================================================================

    // In case of sliding ALE interfaces, 'remesh' fluid field
    auto usedmethod = Teuchos::getIntegralValue<Inpar::FSI::PartitionedCouplingMethod>(
        fsidyn.sublist("PARTITIONED SOLVER"), "PARTITIONED");

    if (usedmethod == Inpar::FSI::DirichletNeumannSlideale)
    {
      remeshing();
    }

    // calculate stresses, strains, energies
    constexpr bool force_prepare = false;
    prepare_output(force_prepare);

    // prepare field variables for new time step
    update();

    // extract final displacement and velocity
    // since we did update, this is very easy to extract
    extract_previous_interface_solution();

    // write current solution
    output();
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<NOX::Nln::LinearSystemBase> FSI::Partitioned::create_linear_system(
    Teuchos::ParameterList& nlParams,
    const Teuchos::RCP<::NOX::Epetra::Interface::Required>& interface,
    ::NOX::Epetra::Vector& noxSoln, ::NOX::Utils& utils)
{
  Teuchos::ParameterList& printParams = nlParams.sublist("Printing");

  Teuchos::ParameterList& dirParams = nlParams.sublist("Direction");
  Teuchos::ParameterList& newtonParams = dirParams.sublist(dirParams.get("Method", "Aitken"));
  Teuchos::ParameterList& lsParams = newtonParams.sublist("Linear Solver");

  Teuchos::RCP<NOX::FSI::FSIMatrixFree> FSIMF;
  Teuchos::RCP<::NOX::Epetra::MatrixFree> MF;
  Teuchos::RCP<::NOX::Epetra::FiniteDifference> FD;
  Teuchos::RCP<::NOX::Epetra::FiniteDifferenceColoring> FDC;
  Teuchos::RCP<::NOX::Epetra::FiniteDifferenceColoring> FDC1;
  Teuchos::RCP<::NOX::Epetra::BroydenOperator> B;

  Teuchos::RCP<::NOX::Epetra::Interface::Jacobian> iJac;

  Teuchos::RCP<Epetra_Operator> J;
  Teuchos::RCP<Epetra_Operator> M;

  Teuchos::RCP<NOX::Nln::LinearSystemBase> linSys;

  // ==================================================================
  // decide on Jacobian and preconditioner
  // We might want to use no preconditioner at all. Some kind of
  // Jacobian has to be provided, otherwise the linear system uses
  // plain finite differences.

  const std::string jacobian = nlParams.get("Jacobian", "None");
  std::string preconditioner = nlParams.get("Preconditioner", "None");

  // Special FSI based matrix free method
  if (jacobian == "FSI Matrix Free")
  {
    // Teuchos::ParameterList& mfParams = nlParams.sublist("FSI Matrix Free");

    // MatrixFree seems to be the most interesting choice. This
    // version builds on our steepest descent relaxation
    // implementation to approximate the Jacobian times x.

    // This is the default method.

    FSIMF = Teuchos::make_rcp<NOX::FSI::FSIMatrixFree>(printParams, interface, noxSoln);
    iJac = FSIMF;
    J = FSIMF;
  }

  // Matrix Free Newton Krylov.
  else if (jacobian == "Matrix Free")
  {
    Teuchos::ParameterList& mfParams = nlParams.sublist("Matrix Free");
    double lambda = mfParams.get("lambda", 1.0e-4);
    mfresitemax_ = mfParams.get("itemax", -1);

    bool kelleyPerturbation = mfParams.get("Kelley Perturbation", false);

    // MatrixFree seems to be the most interesting choice. But you
    // must set a rather low tolerance for the linear solver.

    MF = Teuchos::make_rcp<::NOX::Epetra::MatrixFree>(
        printParams, interface, noxSoln, kelleyPerturbation);
    MF->setLambda(lambda);
    iJac = MF;
    J = MF;
  }

  // No Jacobian at all. Do a fix point iteration.
  else if (jacobian == "None")
  {
    preconditioner = "None";
  }

  // This is pretty much debug code. Or rather research code.
  else if (jacobian == "Dumb Finite Difference")
  {
    Teuchos::ParameterList& fdParams = nlParams.sublist("Finite Difference");
    // fdresitemax_ = fdParams.get("itemax", -1);
    double alpha = fdParams.get("alpha", 1.0e-4);
    double beta = fdParams.get("beta", 1.0e-6);
    std::string dt = fdParams.get("Difference Type", "Forward");
    ::NOX::Epetra::FiniteDifference::DifferenceType dtype =
        ::NOX::Epetra::FiniteDifference::Forward;
    if (dt == "Forward")
      dtype = ::NOX::Epetra::FiniteDifference::Forward;
    else if (dt == "Backward")
      dtype = ::NOX::Epetra::FiniteDifference::Backward;
    else if (dt == "Centered")
      dtype = ::NOX::Epetra::FiniteDifference::Centered;
    else
      FOUR_C_THROW("unsupported difference type '{}'", dt);

    FD = Teuchos::make_rcp<::NOX::Epetra::FiniteDifference>(printParams, interface, noxSoln,
        Teuchos::rcpFromRef(raw_graph_->get_epetra_crs_graph()), beta, alpha);
    FD->setDifferenceMethod(dtype);

    iJac = FD;
    J = FD;
  }
  else
  {
    FOUR_C_THROW("unsupported Jacobian '{}'", jacobian);
  }

  // ==================================================================

  // No preconditioning at all.
  if (preconditioner == "None")
  {
    if (!(iJac))
    {
      // if no Jacobian has been set this better be the fix point
      // method.
      FOUR_C_ASSERT(dirParams.get("Method", "Newton") == "User Defined",
          "No Jacobian for solver '{}'", dirParams.get("Method", "Newton"));

      linSys = Teuchos::null;
    }
    else
    {
      linSys = Teuchos::make_rcp<NOX::FSI::LinearSystemGCR>(
          printParams, lsParams, interface, iJac, J, noxSoln);
    }
  }
  else
  {
    FOUR_C_THROW("unsupported preconditioner '{}'", preconditioner);
  }

  return linSys;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<::NOX::StatusTest::Combo> FSI::Partitioned::create_status_test(
    Teuchos::ParameterList& nlParams, Teuchos::RCP<::NOX::Abstract::Group> grp)
{
  // Create the convergence tests
  Teuchos::RCP<::NOX::StatusTest::Combo> combo =
      Teuchos::make_rcp<::NOX::StatusTest::Combo>(::NOX::StatusTest::Combo::OR);
  Teuchos::RCP<::NOX::StatusTest::Combo> converged =
      Teuchos::make_rcp<::NOX::StatusTest::Combo>(::NOX::StatusTest::Combo::AND);

  Teuchos::RCP<::NOX::StatusTest::MaxIters> maxiters =
      Teuchos::make_rcp<::NOX::StatusTest::MaxIters>(nlParams.get("Max Iterations", 100));
  Teuchos::RCP<::NOX::StatusTest::FiniteValue> fv =
      Teuchos::make_rcp<::NOX::StatusTest::FiniteValue>();

  combo->addStatusTest(fv);
  combo->addStatusTest(converged);
  combo->addStatusTest(maxiters);

  // setup the real tests
  create_status_test(nlParams, grp, converged);

  return combo;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Partitioned::create_status_test(Teuchos::ParameterList& nlParams,
    Teuchos::RCP<::NOX::Abstract::Group> grp, Teuchos::RCP<::NOX::StatusTest::Combo> converged)
{
  Teuchos::RCP<::NOX::StatusTest::NormF> absresid =
      Teuchos::make_rcp<::NOX::StatusTest::NormF>(nlParams.get("Norm abs F", 1.0e-6));
  converged->addStatusTest(absresid);

  if (nlParams.isParameter("Norm Update"))
  {
    Teuchos::RCP<::NOX::StatusTest::NormUpdate> update =
        Teuchos::make_rcp<::NOX::StatusTest::NormUpdate>(nlParams.get("Norm Update", 1.0e-5));
    converged->addStatusTest(update);
  }

  if (nlParams.isParameter("Norm rel F"))
  {
    Teuchos::RCP<::NOX::StatusTest::NormF> relresid =
        Teuchos::make_rcp<::NOX::StatusTest::NormF>(*grp.get(), nlParams.get("Norm rel F", 1.0e-2));
    converged->addStatusTest(relresid);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::Partitioned::initial_guess()
{
  return structure_field()->predict_interface_dispnp();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::Partitioned::interface_disp()
{
  // extract displacements
  return structure_field()->extract_interface_dispnp();
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::Partitioned::interface_force()
{
  // extract forces
  return fluid_to_struct(mb_fluid_field()->extract_interface_forces());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool FSI::Partitioned::computeF(const Epetra_Vector& x, Epetra_Vector& F, const FillType fillFlag)
{
  const char* flags[] = {"Residual", "Jac", "Prec", "FD_Res", "MF_Res", "MF_Jac", "User", nullptr};

  Teuchos::Time timer("FSI_computeF", true);
  const double startTime = timer.wallTime();

  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    utils_->out() << "\n " << "FSI residual calculation" << ".\n";
    if (fillFlag != Residual) utils_->out() << " fillFlag = " << flags[fillFlag] << "\n";
  }

  // we count the number of times the residuum is build
  counter_[fillFlag] += 1;

  if (!x.Map().UniqueGIDs()) FOUR_C_THROW("source map not unique");

  const Core::LinAlg::Vector<double> x_new = Core::LinAlg::Vector<double>(x);
  Core::LinAlg::Vector<double> F_new = Core::LinAlg::Vector<double>(F);
  // Do the FSI step. The real work is in here.
  fsi_op(x_new, F_new, fillFlag);

  F = F_new;

  const double endTime = timer.wallTime();
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    utils_->out() << "\nTime for residual calculation: " << endTime - startTime << " secs\n\n";
  return true;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Partitioned::remeshing() {}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Partitioned::fsi_op(
    const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& F, const FillType fillFlag)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::Partitioned::fluid_op(
    std::shared_ptr<Core::LinAlg::Vector<double>> idisp, const FillType fillFlag)
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0 and
      utils_->isPrintType(::NOX::Utils::OuterIteration))
    utils_->out() << std::endl << "Fluid operator" << std::endl;
  return nullptr;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::Partitioned::struct_op(
    std::shared_ptr<Core::LinAlg::Vector<double>> iforce, const FillType fillFlag)
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0 and
      utils_->isPrintType(::NOX::Utils::OuterIteration))
    utils_->out() << std::endl << "Structural operator" << std::endl;
  return nullptr;
}


/*----------------------------------------------------------------------*
 | Calculate interface velocity based on given interface displacement   |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::Partitioned::interface_velocity(
    const Core::LinAlg::Vector<double>& idispnp) const
{
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  std::shared_ptr<Core::LinAlg::Vector<double>> ivel = nullptr;

  if (fsidyn.get<bool>("SECONDORDER"))
  {
    ivel = std::make_shared<Core::LinAlg::Vector<double>>(*iveln_);
    ivel->update(2. / dt(), idispnp, -2. / dt(), *idispn_, -1.);
  }
  else
  {
    ivel = std::make_shared<Core::LinAlg::Vector<double>>(*idispn_);
    ivel->update(1. / dt(), idispnp, -1. / dt());
  }
  return ivel;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::Partitioned::struct_to_fluid(
    std::shared_ptr<Core::LinAlg::Vector<double>> iv)
{
  const Coupling::Adapter::Coupling& coupsf = structure_fluid_coupling();
  if (matchingnodes_)
  {
    return coupsf.master_to_slave(*iv);
  }
  else
  {
    return coupsfm_->master_to_slave(*iv);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::Partitioned::fluid_to_struct(
    std::shared_ptr<Core::LinAlg::Vector<double>> iv)
{
  const Coupling::Adapter::Coupling& coupsf = structure_fluid_coupling();
  if (matchingnodes_)
  {
    return coupsf.slave_to_master(*iv);
  }
  else
  {
    // Translate consistent nodal forces to interface loads
    const std::shared_ptr<Core::LinAlg::Vector<double>> ishape =
        mb_fluid_field()->integrate_interface_shape();
    Core::LinAlg::Vector<double> iforce(iv->get_map());

    if (iforce.reciprocal_multiply(1.0, *ishape, *iv, 0.0))
      FOUR_C_THROW("ReciprocalMultiply failed");

    return coupsfm_->slave_to_master(iforce);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Coupling::Adapter::CouplingMortar& FSI::Partitioned::structure_fluid_coupling_mortar()
{
  return *coupsfm_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const Coupling::Adapter::CouplingMortar& FSI::Partitioned::structure_fluid_coupling_mortar() const
{
  return *coupsfm_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Partitioned::extract_previous_interface_solution()
{
  idispn_ = structure_field()->extract_interface_dispn();
  iveln_ = fluid_to_struct(mb_fluid_field()->extract_interface_veln());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Partitioned::output()
{
  // call base class version
  FSI::Algorithm::output();

  switch (Teuchos::getIntegralValue<FsiCoupling>(
      Global::Problem::instance()->fsi_dynamic_params(), "COUPALGO"))
  {
    case fsi_iter_stagg_AITKEN_rel_param:
    {
      auto linesearchfactory = noxparameterlist_.sublist("Line Search")
                                   .get<Teuchos::RCP<::NOX::LineSearch::UserDefinedFactory>>(
                                       "User Defined Line Search Factory");
      if (linesearchfactory == Teuchos::null)
        FOUR_C_THROW("Could not get UserDefinedFactory from noxparameterlist_");
      auto aitkenfactory = Teuchos::rcp_dynamic_cast<NOX::FSI::AitkenFactory>(linesearchfactory);

      // write aitken relaxation parameter
      mb_fluid_field()->fluid_field()->disc_writer()->write_double(
          "omega", aitkenfactory->get_aitken()->get_omega());

      break;
    }
    default:
      break;
  }  // switch-case to be extended for other solver variants if necessary
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Partitioned::read_restart(int step)
{
  // call base class version
  FSI::Algorithm::read_restart(step);

  switch (Teuchos::getIntegralValue<FsiCoupling>(
      Global::Problem::instance()->fsi_dynamic_params(), "COUPALGO"))
  {
    case fsi_iter_stagg_AITKEN_rel_param:
    {
      double omega = -1234.0;
      auto input_control_file = Global::Problem::instance()->input_control_file();

      if (std::dynamic_pointer_cast<Adapter::FBIFluidMB>(mb_fluid_field()) != nullptr)
      {
        Core::IO::DiscretizationReader reader(
            mb_fluid_field()->fluid_field()->discretization(), input_control_file, step);
        omega = reader.read_double("omega");
      }
      else if (std::dynamic_pointer_cast<Adapter::FluidAle>(mb_fluid_field()) != nullptr)
      {
        std::shared_ptr<Adapter::FluidAle> fluidale =
            std::dynamic_pointer_cast<Adapter::FluidAle>(mb_fluid_field());
        Core::IO::DiscretizationReader reader(std::const_pointer_cast<Core::FE::Discretization>(
                                                  fluidale->ale_field()->discretization()),
            input_control_file, step);
        omega = reader.read_double("omega");
      }
      else
        FOUR_C_THROW(
            "You want to restart a partitioned FSI scheme with AITKEN relaxation.\n"
            "This is only tested for standard ALE FSI.\n"
            "Check the implementation of FSI::Partitioned::read_restart.");

      noxparameterlist_.sublist("Line Search").sublist("Aitken").set<int>("restart", step);
      noxparameterlist_.sublist("Line Search")
          .sublist("Aitken")
          .set<double>("restart_omega", omega);

      break;
    }  // AITKEN case
    default:
      break;

  }  // switch-case to be extended for other solver variants if necessary
}

FOUR_C_NAMESPACE_CLOSE
