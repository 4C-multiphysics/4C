/*----------------------------------------------------------------------*/
/*!
\file strtimint_impl_nox.cpp
\brief Implicit time integration for spatial discretised
       structural dynamics

<pre>
Maintainer: Thomas Klöppel
            kloeppel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15257
</pre>
*/

/*----------------------------------------------------------------------*/
#ifdef CCADISCRET

/*----------------------------------------------------------------------*/
/* headers */
#include <sstream>

#include "strtimint.H"
#include "strtimint_impl.H"
#include "strtimint_noxgroup.H"
#include "strtimint_noxlinsys.H"
#include "../drt_inpar/drt_boolifyparameters.H"
#include "../linalg/linalg_blocksparsematrix.H"


/*----------------------------------------------------------------------*/
/* setup parameters for solution with NOX */
void STR::TimIntImpl::NoxSetup()
{
  // create
  noxparams_ = Teuchos::rcp(new Teuchos::ParameterList());

  // solving
  Teuchos::ParameterList& newtonParams = (*noxparams_).sublist("Newton");
  newtonParams = *(NoxCreateSolverParameters());
//  Teuchos::ParameterList& lsParams = newtonParams.sublist("Linear Solver");
//  Teuchos::ParameterList& dirParams = nlParams.sublist("Direction");
  //Teuchos::ParameterList& solverOptions = nlParams.sublist("Solver Options");

  // printing
  Teuchos::ParameterList& printParams = (*noxparams_).sublist("Printing");
  printParams = *(NoxCreatePrintParameters(false));

  // Create printing utilities
  noxutils_ = Teuchos::rcp(new NOX::Utils(printParams));
}

/*----------------------------------------------------------------------*/
/* setup parameters for solution with NOX */
void STR::TimIntImpl::NoxSetup(const Teuchos::ParameterList& noxparams)
{
  // copy the input list
  noxparams_ = Teuchos::rcp(new Teuchos::ParameterList(noxparams));
  // make all Yes/No integral values to Boolean
  DRT::INPUT::BoolifyValidInputParameters(*noxparams_);

  // adjust printing parameter list
  Teuchos::ParameterList& printParams = noxparams_->sublist("Printing");
  printParams.set("MyPID", myrank_);
  printParams.set("Output Precision", 3);
  printParams.set("Output Processor", 0);
  int outputinformationlevel = NOX::Utils::Error;  // NOX::Utils::Error==0
  if (printParams.get<bool>("Error"))
    outputinformationlevel += NOX::Utils::Error;
  if (printParams.get<bool>("Warning"))
    outputinformationlevel += NOX::Utils::Warning;
  if (printParams.get<bool>("Outer Iteration"))
    outputinformationlevel += NOX::Utils::OuterIteration;
  if (printParams.get<bool>("Inner Iteration"))
    outputinformationlevel += NOX::Utils::InnerIteration;
  if (printParams.get<bool>("Parameters"))
    outputinformationlevel += NOX::Utils::Parameters;
  if (printParams.get<bool>("Details"))
    outputinformationlevel += NOX::Utils::Details;
  if (printParams.get<bool>("Outer Iteration StatusTest"))
    outputinformationlevel += NOX::Utils::OuterIterationStatusTest;
  if (printParams.get<bool>("Linear Solver Details"))
    outputinformationlevel += NOX::Utils::LinearSolverDetails;
  if (printParams.get<bool>("Test Details"))
    outputinformationlevel += NOX::Utils::TestDetails;
  /*  // for LOCA
  if (printParams.get<bool>("Stepper Iteration"))
    outputinformationlevel += NOX::Utils::StepperIteration;
  if (printParams.get<bool>("Stepper Details"))
    outputinformationlevel += NOX::Utils::StepperDetails;
  if (printParams.get<bool>("Stepper Parameters"))
    outputinformationlevel += NOX::Utils::StepperParameters;
  */
  if (printParams.get<bool>("Debug"))
    outputinformationlevel += NOX::Utils::Debug;
  printParams.set("Output Information", outputinformationlevel);
  noxutils_ = Teuchos::rcp(new NOX::Utils(printParams));
}


/*----------------------------------------------------------------------*/
/* Create status test for non-linear solution with NOX */
Teuchos::RCP<NOX::StatusTest::Combo> STR::TimIntImpl::NoxCreateStatusTest
(
  Teuchos::RCP<NOX::Abstract::Group> grp
)
{
  // type of norm
  NOX::Epetra::Vector::NormType norm = NOX::Epetra::Vector::TwoNorm;
  NOX::StatusTest::NormF::ScaleType scalefres = NOX::StatusTest::NormF::Scaled;
  NOX::StatusTest::NormUpdate::ScaleType scaledisi = NOX::StatusTest::NormUpdate::Scaled;
  if (iternorm_ == INPAR::STR::norm_l1)
  {
    norm = NOX::Epetra::Vector::OneNorm;
    scalefres = NOX::StatusTest::NormF::Unscaled;
    scaledisi = NOX::StatusTest::NormUpdate::Unscaled;
  }
  else if (iternorm_ == INPAR::STR::norm_l2)
  {
    norm = NOX::Epetra::Vector::TwoNorm;
    scalefres = NOX::StatusTest::NormF::Unscaled;
    scaledisi = NOX::StatusTest::NormUpdate::Unscaled;
  }
  else if (iternorm_ == INPAR::STR::norm_rms)
  {
    norm = NOX::Epetra::Vector::TwoNorm;
    scalefres = NOX::StatusTest::NormF::Scaled;
    scaledisi = NOX::StatusTest::NormUpdate::Scaled;
  }
  else if (iternorm_ == INPAR::STR::norm_inf)
  {
    norm = NOX::Epetra::Vector::MaxNorm;
    scalefres = NOX::StatusTest::NormF::Unscaled;
    scaledisi = NOX::StatusTest::NormUpdate::Unscaled;
  }
  else
  {
    dserror("Norm %s is not available",
            INPAR::STR::VectorNormString(iternorm_).c_str());
  }

  // combined residual force and displacement test
  Teuchos::RCP<NOX::StatusTest::Combo> combo2 = Teuchos::null;
  if ( combdisifres_ == INPAR::STR::bop_and )
  {
    combo2 = Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::AND));
  }
  else if ( combdisifres_ == INPAR::STR::bop_or )
  {
    combo2 = Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
  }
  else
  {
    dserror("Cannot handle convergence check");
  }


  // convergence tests for force residual
  if ( normtypefres_ == INPAR::STR::convnorm_abs )
  {
    // absolute test
    Teuchos::RCP<NOX::StatusTest::NormF> statusTestNormFres
      = Teuchos::rcp(new NOX::StatusTest::NormF(tolfres_, norm, scalefres));
    combo2->addStatusTest(statusTestNormFres);
  }
  else if ( normtypefres_ == INPAR::STR::convnorm_rel )
  {
    // relative
    Teuchos::RCP<NOX::StatusTest::NormF> statusTestNormFres
      = Teuchos::rcp(new NOX::StatusTest::NormF(*grp, tolfres_, norm, scalefres));
    combo2->addStatusTest(statusTestNormFres);
  }
  else if ( normtypefres_ == INPAR::STR::convnorm_mix )
  {
    // mixed
    Teuchos::RCP<NOX::StatusTest::Combo> combo3
      = Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::AND));
    combo2->addStatusTest(combo3);
    // absolute test
    Teuchos::RCP<NOX::StatusTest::NormF> statusTestNormFresAbs
      = Teuchos::rcp(new NOX::StatusTest::NormF(tolfres_, norm, scalefres));
    combo3->addStatusTest(statusTestNormFresAbs);
    // AND relative
    Teuchos::RCP<NOX::StatusTest::NormF> statusTestNormFresRel
      = Teuchos::rcp(new NOX::StatusTest::NormF(*grp, tolfres_, norm, scalefres));
    combo3->addStatusTest(statusTestNormFresRel);
  }
  else
  {
    dserror("Type of convergence control is not available");
  }

  // convergence tests for residual displacements
  if ( normtypedisi_ == INPAR::STR::convnorm_abs )
  {
    // absolute test
    Teuchos::RCP<NOX::StatusTest::NormUpdate> statusTestNormDisi
      = Teuchos::rcp(new NOX::StatusTest::NormUpdate(toldisi_, norm, scaledisi));
    combo2->addStatusTest(statusTestNormDisi);
  }
  else if ( normtypedisi_ == INPAR::STR::convnorm_rel )
  {
    // relative test
    // NOT AVAILABLE
    dserror("Not available");
  }
  else if ( normtypedisi_ == INPAR::STR::convnorm_mix )
  {
    // mixed
    //Teuchos::RCP<NOX::StatusTest::Combo> combo3
    //  = Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::AND));
    //combo2->addStatusTest(combo3);
    // absolute test
    Teuchos::RCP<NOX::StatusTest::NormUpdate> statusTestNormDisi
      = Teuchos::rcp(new NOX::StatusTest::NormUpdate(toldisi_, norm, scaledisi));
    combo2->addStatusTest(statusTestNormDisi);
    // relative test
    // NOT AVAILABLE
  }
  else
  {
    dserror("Type of convergence control is not available");
  }


  // maximum iteration
  Teuchos::RCP<NOX::StatusTest::MaxIters> statusTestMaxIters
    = Teuchos::rcp(new NOX::StatusTest::MaxIters(itermax_));


  // the combined test object
  Teuchos::RCP<NOX::StatusTest::Combo> combo
    = Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
  combo->addStatusTest(combo2);
  combo->addStatusTest(statusTestMaxIters);


  // hand over
  return combo;
}

/*----------------------------------------------------------------------*/
/* Create solver parameters for  non-linear solution with NOX */
Teuchos::RCP<Teuchos::ParameterList> STR::TimIntImpl::NoxCreateSolverParameters()
{
  // Create the list of solver parameters
  Teuchos::RCP<Teuchos::ParameterList> solverParametersPtr
    = Teuchos::rcp(new Teuchos::ParameterList);

  // Select the solver (this is the default)
  solverParametersPtr->set("Nonlinear Solver", "Line Search Based");

  // Create the line search parameters sublist
  Teuchos::ParameterList& lineSearchParameters
    = solverParametersPtr->sublist("Line Search");

  // Set the line search method
  lineSearchParameters.set("Method", "Full Step");

  // deliver it
  return solverParametersPtr;
}

/*----------------------------------------------------------------------*/
/* Create printing parameters for non-linear solution with NOX */
Teuchos::RCP<Teuchos::ParameterList> STR::TimIntImpl::NoxCreatePrintParameters
(
  const bool verbose
) const
{
  // Set the printing parameters in the "Printing" sublist
  Teuchos::RCP<Teuchos::ParameterList> printParams
    = Teuchos::rcp(new Teuchos::ParameterList());
  (*printParams).set("MyPID", myrank_);
  (*printParams).set("Output Precision", 6);
  (*printParams).set("Output Processor", 0);
  if (verbose)
  {
    (*printParams).set("Output Information",
                       NOX::Utils::OuterIteration
                       + NOX::Utils::OuterIterationStatusTest
                       + NOX::Utils::InnerIteration
                       + NOX::Utils::LinearSolverDetails
                       + NOX::Utils::Parameters
                       + NOX::Utils::Details
                       + NOX::Utils::Warning
                       + NOX::Utils::Debug
                       + NOX::Utils::TestDetails
                       + NOX::Utils::Error);
  }
  else if (printiter_)
  {
    (*printParams).set("Output Information",
                       NOX::Utils::Error
                       + NOX::Utils::OuterIterationStatusTest
                       + NOX::Utils::TestDetails);
  }
  else
  {
    (*printParams).set("Output Information",
                       NOX::Utils::Error
                       + NOX::Utils::TestDetails);
  }

  // deliver liver
  return printParams;
}


/*----------------------------------------------------------------------*/
/* Compute the residual of discretised linear momentum */
bool STR::TimIntImpl::computeF
(
  const Epetra_Vector& x,
  Epetra_Vector& RHS,
  const NOX::Epetra::Interface::Required::FillType flag
)
{
  // determine residual displacements
  //   #x holds the current trial total displacements due to NOX: D_{n+1}^{<k+1>}
  //   #disn_ holds the total displacement of last NOX iteration: D_{n+1}^{<k>}
  disi_->Update(1.0, x, -1.0, *disn_, 0.0);

  // update end-point displacements etc.
  //   brings #disn_ in sync with #x, so we are ready for next call here
  UpdateIter(0);

  // make force residual and tangent, disi is needed for elementwise variables
  EvaluateForceStiffResidual();

  // blank DBC stuff etc.
  // HINT: a negative residual is returned
  PrepareSystemForNewtonSolve();

  // associate the RHS
  // scale back to positive residual as expected by NOX
  RHS.Update(-1.0, *fres_, 0.0);

  // deliver
  return true;
}

/*----------------------------------------------------------------------*/
/* Compute effective dynamic stiffness matrix */
bool STR::TimIntImpl::computeJacobian
(
  const Epetra_Vector& x,  //!< solution vector \f$x\f$ specified from NOX
  Epetra_Operator& Jac  //!< a reference to the Jacobian operator
)
{
  // deliver
  return true;
}

/*----------------------------------------------------------------------*/
/* preconditioner */
bool STR::TimIntImpl::computePreconditioner
(
  const Epetra_Vector &x,
  Epetra_Operator &M,
  Teuchos::ParameterList *precParams
)
{
  // deliver
  return true;
}

/*----------------------------------------------------------------------*/
/* Create linear system */
Teuchos::RCP<NOX::Epetra::LinearSystem> STR::TimIntImpl::NoxCreateLinearSystem
(
  Teuchos::ParameterList& nlParams,
  NOX::Epetra::Vector& noxSoln,
  Teuchos::RCP<NOX::Utils> utils
)
{
  Teuchos::RCP<NOX::Epetra::LinearSystem> linSys = Teuchos::null;

  Teuchos::ParameterList& printParams = nlParams.sublist("Printing");
  Teuchos::ParameterList& dirParams = nlParams.sublist("Direction");
  Teuchos::ParameterList& newtonParams = dirParams.sublist("Newton");
  Teuchos::ParameterList& lsParams = newtonParams.sublist("Linear Solver");

  NOX::Epetra::Interface::Jacobian* iJac = this;
  const Teuchos::RCP<Epetra_Operator> J = stiff_;

#if 0
  NOX::Epetra::Interface::Preconditioner* iPrec = this;
  const Teuchos::RCP<Epetra_Operator> M = stiff_;
  linSys = Teuchos::rcp(new NOX::Epetra::LinearSystemAztecOO(printParams,
                                                             lsParams,
                                                             Teuchos::rcp(iJac,false),
                                                             J,
                                                             Teuchos::rcp(iPrec,false),
                                                             M,
                                                             noxSoln));
#else
  linSys = Teuchos::rcp(new NOX::STR::LinearSystem(printParams,
                                                   lsParams,
                                                   Teuchos::rcp(iJac,false),
                                                   J,
                                                   noxSoln,
                                                   solver_));
#endif

  // just a half-empty tin of cat food
  return linSys;
}

/*----------------------------------------------------------------------*/
/* Do non-linear solve with NOX */
void STR::TimIntImpl::NoxSolve()
{
  // extract parameter lists
  Teuchos::ParameterList& nlParams = *noxparams_;
  Teuchos::ParameterList& printParams = nlParams.sublist("Printing");

  // create intial guess vector of predictor result
  NOX::Epetra::Vector noxSoln(disn_, NOX::Epetra::Vector::CreateView);

  // Linear system
  Teuchos::RCP<NOX::Epetra::LinearSystem> linSys
    = NoxCreateLinearSystem(nlParams, noxSoln, noxutils_);

  // Create group
  Teuchos::RCP<NOX::STR::Group> grp
    = Teuchos::rcp(new NOX::STR::Group(*this,
                                       printParams,
                                       Teuchos::rcp(this, false),
                                       noxSoln,
                                       linSys));

  // Create status test
  noxstatustest_ = NoxCreateStatusTest(grp);

  // Create the solver
  Teuchos::RCP<NOX::Solver::Generic> solver
    = NOX::Solver::buildSolver(grp, noxstatustest_, noxparams_);

  // Solve the nonlinear system
  NOX::StatusTest::StatusType status = solver->solve();
  noxstatustest_->print(std::cout);

  // bona nox : A divergent NOX solution
  if (status != NOX::StatusTest::Converged)
    if (myrank_ == 0)
      noxutils_->out() << "Nonlinear solver failed to converge!" << endl;

  // extract number of iteration steps
  iter_ = solver->getNumIterations();

  // return to sender
  return;
}

/*----------------------------------------------------------------------*/
#endif  // #ifdef CCADISCRET
