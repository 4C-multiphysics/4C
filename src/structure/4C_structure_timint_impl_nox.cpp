// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_pstream.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_structure_timint.hpp"
#include "4C_structure_timint_impl.hpp"
#include "4C_structure_timint_noxgroup.hpp"
#include "4C_structure_timint_noxlinsys.hpp"
#include "4C_utils_enum.hpp"

#include <Teuchos_RCPStdSharedPtrConversions.hpp>


FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Solid::TimIntImpl::nox_setup()
{
  // create
  noxparams_ = std::make_shared<Teuchos::ParameterList>();

  // solving
  Teuchos::ParameterList& newtonParams = (*noxparams_).sublist("Newton");
  newtonParams = *(nox_create_solver_parameters());
  //  Teuchos::ParameterList& lsParams = newtonParams.sublist("Linear Solver");
  //  Teuchos::ParameterList& dirParams = nlParams.sublist("Direction");
  // Teuchos::ParameterList& solverOptions = nlParams.sublist("Solver Options");

  // printing
  Teuchos::ParameterList& printParams = (*noxparams_).sublist("Printing");
  printParams = *(nox_create_print_parameters(false));

  // Create printing utilities
  noxutils_ = std::make_shared<::NOX::Utils>(printParams);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Solid::TimIntImpl::nox_setup(const Teuchos::ParameterList& noxparams)
{
  // copy the input list
  noxparams_ = std::make_shared<Teuchos::ParameterList>(noxparams);

  // Remove legacy parameters
  {
    Teuchos::ParameterList& solverOptionsList = noxparams_->sublist("Solver Options");

    // remove "Merit Function" parameter entry after building the user-defined function
    // (NOX does not accept this parameter)
    solverOptionsList.remove("Merit Function", false);

    // also remove *PrintEqualSign*
    solverOptionsList.remove("*PrintEqualSign*", false);
  }

  // adjust printing parameter list
  Teuchos::ParameterList& printParams = noxparams_->sublist("Printing");
  printParams.set("MyPID", myrank_);
  printParams.set("Output Precision", 3);
  printParams.set("Output Processor", 0);
  int outputinformationlevel = ::NOX::Utils::Error;  // ::NOX::Utils::Error==0
  if (printParams.get<bool>("Error")) outputinformationlevel += ::NOX::Utils::Error;
  if (printParams.get<bool>("Warning")) outputinformationlevel += ::NOX::Utils::Warning;
  if (printParams.get<bool>("Outer Iteration"))
    outputinformationlevel += ::NOX::Utils::OuterIteration;
  if (printParams.get<bool>("Inner Iteration"))
    outputinformationlevel += ::NOX::Utils::InnerIteration;
  if (printParams.get<bool>("Parameters")) outputinformationlevel += ::NOX::Utils::Parameters;
  if (printParams.get<bool>("Details")) outputinformationlevel += ::NOX::Utils::Details;
  if (printParams.get<bool>("Outer Iteration StatusTest"))
    outputinformationlevel += ::NOX::Utils::OuterIterationStatusTest;
  if (printParams.get<bool>("Linear Solver Details"))
    outputinformationlevel += ::NOX::Utils::LinearSolverDetails;
  if (printParams.get<bool>("Test Details")) outputinformationlevel += ::NOX::Utils::TestDetails;
  if (printParams.get<bool>("Debug")) outputinformationlevel += ::NOX::Utils::Debug;
  printParams.set("Output Information", outputinformationlevel);
  noxutils_ = std::make_shared<::NOX::Utils>(printParams);
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Teuchos::RCP<::NOX::StatusTest::Combo> Solid::TimIntImpl::nox_create_status_test(
    ::NOX::Abstract::Group& grp)
{
  // type of norm
  ::NOX::Epetra::Vector::NormType norm = ::NOX::Epetra::Vector::TwoNorm;
  ::NOX::StatusTest::NormF::ScaleType scalefres = ::NOX::StatusTest::NormF::Scaled;
  ::NOX::StatusTest::NormUpdate::ScaleType scaledisi = ::NOX::StatusTest::NormUpdate::Scaled;
  if (iternorm_ == Inpar::Solid::norm_l1)
  {
    norm = ::NOX::Epetra::Vector::OneNorm;
    scalefres = ::NOX::StatusTest::NormF::Unscaled;
    scaledisi = ::NOX::StatusTest::NormUpdate::Unscaled;
  }
  else if (iternorm_ == Inpar::Solid::norm_l2)
  {
    norm = ::NOX::Epetra::Vector::TwoNorm;
    scalefres = ::NOX::StatusTest::NormF::Unscaled;
    scaledisi = ::NOX::StatusTest::NormUpdate::Unscaled;
  }
  else if (iternorm_ == Inpar::Solid::norm_rms)
  {
    norm = ::NOX::Epetra::Vector::TwoNorm;
    scalefres = ::NOX::StatusTest::NormF::Scaled;
    scaledisi = ::NOX::StatusTest::NormUpdate::Scaled;
  }
  else if (iternorm_ == Inpar::Solid::norm_inf)
  {
    norm = ::NOX::Epetra::Vector::MaxNorm;
    scalefres = ::NOX::StatusTest::NormF::Unscaled;
    scaledisi = ::NOX::StatusTest::NormUpdate::Unscaled;
  }
  else
  {
    FOUR_C_THROW("Norm {} is not available", iternorm_);
  }

  // combined residual force and displacement test
  Teuchos::RCP<::NOX::StatusTest::Combo> combo2 = Teuchos::null;
  if (combdisifres_ == Inpar::Solid::bop_and)
  {
    combo2 = Teuchos::make_rcp<::NOX::StatusTest::Combo>(::NOX::StatusTest::Combo::AND);
  }
  else if (combdisifres_ == Inpar::Solid::bop_or)
  {
    combo2 = Teuchos::make_rcp<::NOX::StatusTest::Combo>(::NOX::StatusTest::Combo::OR);
  }
  else
  {
    FOUR_C_THROW("Cannot handle convergence check");
  }


  // convergence tests for force residual
  if (normtypefres_ == Inpar::Solid::convnorm_abs)
  {
    // absolute test
    Teuchos::RCP<::NOX::StatusTest::NormF> statusTestNormFres =
        Teuchos::make_rcp<::NOX::StatusTest::NormF>(tolfres_, norm, scalefres);
    combo2->addStatusTest(statusTestNormFres);
  }
  else if (normtypefres_ == Inpar::Solid::convnorm_rel)
  {
    // relative
    Teuchos::RCP<::NOX::StatusTest::NormF> statusTestNormFres =
        Teuchos::make_rcp<::NOX::StatusTest::NormF>(grp, tolfres_, norm, scalefres);
    combo2->addStatusTest(statusTestNormFres);
  }
  else if (normtypefres_ == Inpar::Solid::convnorm_mix)
  {
    // mixed
    Teuchos::RCP<::NOX::StatusTest::Combo> combo3 =
        Teuchos::make_rcp<::NOX::StatusTest::Combo>(::NOX::StatusTest::Combo::AND);
    combo2->addStatusTest(combo3);
    // absolute test
    Teuchos::RCP<::NOX::StatusTest::NormF> statusTestNormFresAbs =
        Teuchos::make_rcp<::NOX::StatusTest::NormF>(tolfres_, norm, scalefres);
    combo3->addStatusTest(statusTestNormFresAbs);
    // AND relative
    Teuchos::RCP<::NOX::StatusTest::NormF> statusTestNormFresRel =
        Teuchos::make_rcp<::NOX::StatusTest::NormF>(grp, tolfres_, norm, scalefres);
    combo3->addStatusTest(statusTestNormFresRel);
  }
  else
  {
    FOUR_C_THROW("Type of convergence control is not available");
  }

  // convergence tests for residual displacements
  if (normtypedisi_ == Inpar::Solid::convnorm_abs)
  {
    // absolute test
    Teuchos::RCP<::NOX::StatusTest::NormUpdate> statusTestNormDisi =
        Teuchos::make_rcp<::NOX::StatusTest::NormUpdate>(toldisi_, norm, scaledisi);
    combo2->addStatusTest(statusTestNormDisi);
  }
  else if (normtypedisi_ == Inpar::Solid::convnorm_rel)
  {
    // relative test
    // NOT AVAILABLE
    FOUR_C_THROW("Not available");
  }
  else if (normtypedisi_ == Inpar::Solid::convnorm_mix)
  {
    // mixed
    // absolute test
    Teuchos::RCP<::NOX::StatusTest::NormUpdate> statusTestNormDisi =
        Teuchos::make_rcp<::NOX::StatusTest::NormUpdate>(toldisi_, norm, scaledisi);
    combo2->addStatusTest(statusTestNormDisi);
    // relative test
    // NOT AVAILABLE
  }
  else
  {
    FOUR_C_THROW("Type of convergence control is not available");
  }


  // maximum iteration
  Teuchos::RCP<::NOX::StatusTest::MaxIters> statusTestMaxIters =
      Teuchos::make_rcp<::NOX::StatusTest::MaxIters>(itermax_);


  // the combined test object
  Teuchos::RCP<::NOX::StatusTest::Combo> combo =
      Teuchos::make_rcp<::NOX::StatusTest::Combo>(::NOX::StatusTest::Combo::OR);
  combo->addStatusTest(combo2);
  combo->addStatusTest(statusTestMaxIters);


  // hand over
  return combo;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Teuchos::ParameterList> Solid::TimIntImpl::nox_create_solver_parameters()
{
  // Create the list of solver parameters
  std::shared_ptr<Teuchos::ParameterList> solverParametersPtr =
      std::make_shared<Teuchos::ParameterList>();

  // Select the solver (this is the default)
  solverParametersPtr->set("Nonlinear Solver", "Line Search Based");

  // Create the line search parameters sublist
  Teuchos::ParameterList& lineSearchParameters = solverParametersPtr->sublist("Line Search");

  // Set the line search method
  lineSearchParameters.set("Method", "Full Step");

  // deliver it
  return solverParametersPtr;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Teuchos::ParameterList> Solid::TimIntImpl::nox_create_print_parameters(
    const bool verbose) const
{
  // Set the printing parameters in the "Printing" sublist
  std::shared_ptr<Teuchos::ParameterList> printParams = std::make_shared<Teuchos::ParameterList>();
  (*printParams).set("MyPID", myrank_);
  (*printParams).set("Output Precision", 6);
  (*printParams).set("Output Processor", 0);
  if (verbose)
  {
    (*printParams)
        .set("Output Information",
            ::NOX::Utils::OuterIteration + ::NOX::Utils::OuterIterationStatusTest +
                ::NOX::Utils::InnerIteration + ::NOX::Utils::LinearSolverDetails +
                ::NOX::Utils::Parameters + ::NOX::Utils::Details + ::NOX::Utils::Warning +
                ::NOX::Utils::Debug + ::NOX::Utils::TestDetails + ::NOX::Utils::Error);
  }
  else if (printiter_)
  {
    (*printParams)
        .set("Output Information", ::NOX::Utils::Error + ::NOX::Utils::OuterIterationStatusTest +
                                       ::NOX::Utils::TestDetails);
  }
  else
  {
    (*printParams).set("Output Information", ::NOX::Utils::Error + ::NOX::Utils::TestDetails);
  }

  // deliver liver
  return printParams;
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
bool Solid::TimIntImpl::computeF(const Epetra_Vector& x, Epetra_Vector& RHS,
    const ::NOX::Epetra::Interface::Required::FillType flag)
{
  /* determine residual displacements:
   * #x holds the current trial total displacements due to NOX: D_{n+1}^{<k+1>}
   * #disn_ holds the total displacement of last NOX iteration: D_{n+1}^{<k>}
   */
  disi_->update(1.0, x, -1.0, *disn_, 0.0);

  // update end-point displacements etc.
  //   brings #disn_ in sync with #x, so we are ready for next call here
  update_iter(0);

  // Evaluate the residual (without its linearization)
  evaluate_force_residual();

  // blank DBC stuff. HINT: a negative residual is returned
  prepare_system_for_newton_solve(false);

  // associate the RHS and scale back to positive residual as expected by NOX
  RHS.Update(-1.0, *fres_, 0.0);

  return true;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
bool Solid::TimIntImpl::computeJacobian(const Epetra_Vector& x, Epetra_Operator& Jac)
{
  // create parameter list
  Teuchos::ParameterList params;

  // make force residual and tangent, disi is needed for element-wise variables
  evaluate_force_stiff_residual(params);

  // blank DBC stuff. HINT: a negative residual is returned
  prepare_system_for_newton_solve(true);

  return true;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<::NOX::Epetra::LinearSystem> Solid::TimIntImpl::nox_create_linear_system(
    Teuchos::ParameterList& nlParams, ::NOX::Epetra::Vector& noxSoln, ::NOX::Utils& utils)
{
  std::shared_ptr<::NOX::Epetra::LinearSystem> linSys = nullptr;

  Teuchos::ParameterList& printParams = nlParams.sublist("Printing");
  Teuchos::ParameterList& dirParams = nlParams.sublist("Direction");
  Teuchos::ParameterList& newtonParams = dirParams.sublist("Newton");
  Teuchos::ParameterList& lsParams = newtonParams.sublist("Linear Solver");

  ::NOX::Epetra::Interface::Jacobian* iJac = this;
  const std::shared_ptr<Epetra_Operator> J = stiff_;

  linSys = std::make_shared<NOX::Solid::LinearSystem>(
      printParams, lsParams, Core::Utils::shared_ptr_from_ref(*iJac), J, noxSoln, solver_);

  return linSys;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
int Solid::TimIntImpl::nox_solve()
{
  // extract parameter lists
  Teuchos::ParameterList& nlParams = *noxparams_;
  Teuchos::ParameterList& printParams = nlParams.sublist("Printing");

  // create initial guess vector of predictor result
  ::NOX::Epetra::Vector noxSoln(
      Teuchos::rcpFromRef(disn_->get_ref_of_epetra_vector()), ::NOX::Epetra::Vector::CreateView);

  // Linear system
  Teuchos::RCP<::NOX::Epetra::LinearSystem> linSys =
      Teuchos::rcp(nox_create_linear_system(nlParams, noxSoln, *noxutils_));

  // Create group
  Teuchos::RCP<NOX::Solid::Group> grp = Teuchos::make_rcp<NOX::Solid::Group>(
      *this, printParams, Teuchos::rcpFromRef(*this), noxSoln, linSys);

  // Create status test
  noxstatustest_ = nox_create_status_test(*grp);

  // Create the solver
  auto solver = ::NOX::Solver::buildSolver(grp, noxstatustest_, Teuchos::rcpFromRef(*noxparams_));

  // Solve the nonlinear system
  ::NOX::StatusTest::StatusType status = solver->solve();

  noxstatustest_->print(std::cout);

  // error check
  return nox_error_check(status, *solver);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
int Solid::TimIntImpl::nox_error_check(
    ::NOX::StatusTest::StatusType status, ::NOX::Solver::Generic& solver)
{
  noxstatustest_->print(std::cout);

  // check if nonlinear solver converged
  if (status != ::NOX::StatusTest::Converged)
  {
    if (divcontype_ == Inpar::Solid::divcont_halve_step or
        divcontype_ == Inpar::Solid::divcont_repeat_step or
        divcontype_ == Inpar::Solid::divcont_repeat_simulation)
    {
      if (myrank_ == 0) noxutils_->out() << "Nonlinear solver failed to converge!" << std::endl;
      return 1;
    }
    else if (divcontype_ == Inpar::Solid::divcont_continue)
    {
      if (myrank_ == 0)
        noxutils_->out() << "Nonlinear solver failed to converge! continuing " << std::endl;
      return 0;
    }
    else
    {
      FOUR_C_THROW("Nonlinear solver failed to converge!");
      return 1;  // make compiler happy
    }
  }
  else  // everything is fine
  {
    // extract number of iteration steps
    iter_ = solver.getNumIterations();
    return 0;
  }
}

FOUR_C_NAMESPACE_CLOSE
