/*----------------------------------------------------------------------------*/
/*!
 \file nln_operator_linprec.cpp

 <pre>
 Maintainer: Matthias Mayr
             mayr@mhpc.mw.tum.de
             089 - 289-10362
 </pre>
 */

/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/* headers */

// Epetra
#include <Epetra_Comm.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Operator.h>
#include <Epetra_Vector.h>

// Ifpack
#include <Ifpack.h>

// standard
#include <iostream>
#include <vector>

// Teuchos
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

// baci
#include "linesearch_base.H"
#include "linesearch_factory.H"
#include "nln_operator_linprec.H"
#include "nln_problem.H"

#include "../drt_io/io_control.H"
#include "../drt_io/io_pstream.H"

#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_globalproblem.H"

#include "../linalg/linalg_solver.H"
#include "../linalg/linalg_sparsematrix.H"

#include "../solver/solver_preconditionertype.H"
#include "../solver/solver_ifpackpreconditioner.H"

/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
NLNSOL::NlnOperatorLinPrec::NlnOperatorLinPrec() :
    linprec_(Teuchos::null)
{
  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void NLNSOL::NlnOperatorLinPrec::Setup()
{
  // time measurements
  Teuchos::RCP<Teuchos::Time> time = Teuchos::TimeMonitor::getNewCounter(
      "NLNSOL::NlnOperatorLinPrec::Setup");
  Teuchos::TimeMonitor monitor(*time);

  // Make sure that Init() has been called
  if (not IsInit()) { dserror("Init() has not been called, yet."); }

  // ---------------------------------------------------------------------------
  // Get parameter list for linear preconditioner from input file
  // ---------------------------------------------------------------------------
  // get the solver number
  const int linsolvernumber = Params().get<int>(
      "Linear Preconditioner: Linear Solver");

  // check if the solver ID is valid
  if (linsolvernumber == (-1)) { dserror("No valid linear solver defined!"); }

  // linear solver parameter list
  const Teuchos::ParameterList& inputparams =
      DRT::Problem::Instance()->SolverParams(linsolvernumber);
  Teuchos::ParameterList solverparams =
      LINALG::Solver::TranslateSolverParameters(inputparams);

  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // Create the linear preconditioner based on user's choice
  // ---------------------------------------------------------------------------
  if (solverparams.isSublist("IFPACK Parameters"))
  {
    linprec_ = Teuchos::rcp(
        new LINALG::SOLVER::IFPACKPreconditioner(
            DRT::Problem::Instance()->ErrorFile()->Handle(),
            solverparams.sublist("IFPACK Parameters"),
            solverparams.sublist("Aztec Parameters")));
  }
  else
  {
    dserror("This type of linear preconditioner is not implemented, yet. "
        "Or maybe you just chose a wrong configuration in the input file.");
  }

  Teuchos::RCP<Epetra_MultiVector> tmp =
      Teuchos::rcp(new Epetra_MultiVector(NlnProblem()->DofRowMap(), true));

  // Setup with dummy vectors
  linprec_->Setup(true, &*(NlnProblem()->GetJacobianOperator()),
      &*tmp, &*tmp);

  // Setup() has been called
  SetIsSetup();

  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
int NLNSOL::NlnOperatorLinPrec::ApplyInverse(const Epetra_MultiVector& f,
    Epetra_MultiVector& x) const
{
  // time measurements
  Teuchos::RCP<Teuchos::Time> time = Teuchos::TimeMonitor::getNewCounter(
      "NLNSOL::NlnOperatorLinPrec::ApplyInverse");
  Teuchos::TimeMonitor monitor(*time);

  int err = 0;

  // Make sure that Init() and Setup() have been called
  if (not IsInit()) { dserror("Init() has not been called, yet."); }
  if (not IsSetup()) { dserror("Setup() has not been called, yet."); }

  Teuchos::RCP<Epetra_MultiVector> inc =
      Teuchos::rcp(new Epetra_MultiVector(x.Map(), true));

  // applying the linear preconditioner
  int errorcode = linprec_->ApplyInverse(f, *inc);

  /* Update solution by preconditioned increment. Use negative sign due to the
   * type of residual, that has been handed in.
   */
  err = x.Update(1.0, *inc, 1.0);
  if (err != 0) { dserror("Update failed."); }

  Teuchos::RCP<Epetra_MultiVector> fnew =
      Teuchos::rcp(new Epetra_MultiVector(f.Map(), true));
  NlnProblem()->ComputeF(x, *fnew);
  double fnorm2 = 1.0e+12;
  NlnProblem()->ConvergenceCheck(*fnew, fnorm2);
  PrintIterSummary(-1, fnorm2);

  return errorcode;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
const double NLNSOL::NlnOperatorLinPrec::ComputeStepLength(
    const Epetra_MultiVector& x, const Epetra_MultiVector& f,
    const Epetra_MultiVector& inc, double fnorm2) const
{
  dserror("There is no line search algorithm available for a linear "
      "preconditioner object.");

  // just for the compiler
  return 0.0;
}
