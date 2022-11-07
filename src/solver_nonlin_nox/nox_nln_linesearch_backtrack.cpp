/*-----------------------------------------------------------*/
/*! \file

\brief %NOX::NLN backtracking line search implementation.



\level 3

*/
/*-----------------------------------------------------------*/

#include "nox_nln_linesearch_backtrack.H"  // class definition
#include "nox_nln_statustest_normf.H"
#include "nox_nln_solver_linesearchbased.H"
#include "nox_nln_linesearch_prepostoperator.H"

#include "drt_dserror.H"

#include <NOX_Utils.H>
#include <NOX_GlobalData.H>
#include "nox_nln_group.H"

#include <Epetra_Vector.h>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::NLN::LineSearch::Backtrack::Backtrack(const Teuchos::RCP<NOX::GlobalData>& gd,
    const Teuchos::RCP<NOX::StatusTest::Generic> outerTests,
    const Teuchos::RCP<NOX::NLN::INNER::StatusTest::Generic> innerTests,
    Teuchos::ParameterList& params)
    : lsIters_(0),
      stepPtr_(NULL),
      defaultStep_(0.0),
      reductionFactor_(0.0),
      checkType_(NOX::StatusTest::Complete),
      status_(NOX::NLN::INNER::StatusTest::status_unevaluated),
      outerTestsPtr_(outerTests),
      innerTestsPtr_(innerTests)
{
  reset(gd, params);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool NOX::NLN::LineSearch::Backtrack::reset(
    const Teuchos::RCP<NOX::GlobalData>& gd, Teuchos::ParameterList& params)
{
  Teuchos::ParameterList& p = params.sublist("Backtrack");

  utils_ = gd->getUtils();
  meritFunctionPtr_ = gd->getMeritFunction();

  lsIters_ = 0;
  searchDirectionPtr_ = Teuchos::null;

  status_ = NOX::NLN::INNER::StatusTest::status_unevaluated;

  defaultStep_ = p.get("Default Step", 1.0);
  reductionFactor_ = p.get("Reduction Factor", 0.5);
  if ((reductionFactor_ <= 0.0) || (reductionFactor_ >= 1.0))
  {
    std::ostringstream msg;
    msg << "Invalid choice \"" << reductionFactor_ << "\" for \"Reduction Factor\"!\n"
        << "Value must be greater than zero and less than 1.0.";
    throwError("reset", msg.str());
  }

  checkType_ =
      Teuchos::getIntegralValue<NOX::StatusTest::CheckType>(params, "Inner Status Test Check Type");

  fp_except_.shall_be_caught_ = p.get("Allow Exceptions", false);

  prePostOperatorPtr_ = Teuchos::rcp(new PrePostOperator(params));

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::LineSearch::Backtrack::reset()
{
  lsIters_ = 0;
  searchDirectionPtr_ = Teuchos::null;

  status_ = NOX::NLN::INNER::StatusTest::status_unevaluated;

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool NOX::NLN::LineSearch::Backtrack::compute(NOX::Abstract::Group& grp, double& step,
    const NOX::Abstract::Vector& dir, const NOX::Solver::Generic& s)
{
  fp_except_.precompute();
  // -------------------------------------------------
  // (re)set important line search parameters
  // -------------------------------------------------
  reset();
  // get the old solution group
  const NOX::Abstract::Group& oldGrp = s.getPreviousSolutionGroup();
  // update the search direction pointer
  searchDirectionPtr_ = Teuchos::rcpFromRef(dir);
  // set the step pointer to the inserted step variable
  stepPtr_ = &step;
  // reset the step length
  step = defaultStep_;
  // initialize the inner status test
  // ----------------------------------------------------------------------
  // BE CAREFUL HERE:
  // During the copy operation in Solver::LineSearchBased::step()
  // the current grp loses the ownership of the sharedLinearSystem. If we
  // want to access the jacobian, we have to use the oldGrp
  // (target of the copy process), instead.                hiermeier 08/15
  // ----------------------------------------------------------------------
  const NOX::Epetra::Group* epetraOldGrpPtr = dynamic_cast<const NOX::Epetra::Group*>(&oldGrp);
  if (not epetraOldGrpPtr->isJacobian()) throwError("compute()", "Ownership changed unexpectedly!");

  /* Setup the inner status test */
  status_ = innerTestsPtr_->CheckStatus(*this, s, oldGrp, checkType_);

  // increase iteration counter after initialization
  ++lsIters_;

  // -------------------------------------------------
  // update the solution vector and get a trial point
  // -------------------------------------------------
  grp.computeX(oldGrp, dir, step);
  NOX::Abstract::Group::ReturnType rtype = NOX::Abstract::Group::Ok;
  bool failed = false;
  try
  {
    failed = false;
    rtype = grp.computeF();
    if (rtype != NOX::Abstract::Group::Ok) throwError("compute", "Unable to compute F!");

    /* Safe-guarding of the inner status test:
     * If the outer NormF test is converged for a full step length,
     * we don't have to reduce the step length any further.
     * This additional check becomes necessary, because of cancellation
     * errors and related numerical artifacts. */
    // check the outer status test for the full step length
    outerTestsPtr_->checkStatus(s, checkType_);

    const NOX::NLN::Solver::LineSearchBased& lsSolver =
        static_cast<const NOX::NLN::Solver::LineSearchBased&>(s);

    const NOX::StatusTest::StatusType ostatus = lsSolver.GetStatus<NOX::NLN::StatusTest::NormF>();

    /* Skip the inner status test, if the outer NormF test is
     * already converged! */
    if (ostatus == NOX::StatusTest::Converged)
    {
      fp_except_.enable();
      return true;
    }
  }
  // catch error of the computeF method
  catch (const char* e)
  {
    if (not fp_except_.shall_be_caught_) dserror("An exception occurred: %s", e);

    utils_->out(NOX::Utils::Warning) << "WARNING: Error caught = " << e << "\n";

    status_ = NOX::NLN::INNER::StatusTest::status_step_too_long;
    failed = true;
  }
  // clear the exception checks after the try/catch block
  fp_except_.clear();

  // -------------------------------------------------
  // print header if desired
  // -------------------------------------------------

  utils_->out(NOX::Utils::InnerIteration) << "\n"
                                          << NOX::Utils::fill(72, '=') << "\n"
                                          << "-- Backtrack Line Search -- \n";

  if (not failed)
  {
    status_ = innerTestsPtr_->CheckStatus(*this, s, grp, checkType_);
    PrintUpdate(utils_->out(NOX::Utils::InnerIteration));
  }
  // -------------------------------------------------
  // inner backtracking loop
  // -------------------------------------------------
  while (status_ == NOX::NLN::INNER::StatusTest::status_step_too_long)
  {
    // -------------------------------------------------
    // reduce step length
    // -------------------------------------------------
    prePostOperatorPtr_->runPreModifyStepLength(s, *this);
    step *= reductionFactor_;

    // -------------------------------------------------
    // - update the solution vector and get a trial point
    // - increase line search step counter
    // -------------------------------------------------
    grp.computeX(oldGrp, dir, step);
    ++lsIters_;

    try
    {
      rtype = grp.computeF();
      if (rtype != NOX::Abstract::Group::Ok) throwError("compute", "Unable to compute F!");
      status_ = innerTestsPtr_->CheckStatus(*this, s, grp, checkType_);
      PrintUpdate(utils_->out(NOX::Utils::InnerIteration));
    }
    // catch error of the computeF method
    catch (const char* e)
    {
      if (not fp_except_.shall_be_caught_) dserror("An exception occurred: %s", e);

      if (utils_->isPrintType(NOX::Utils::Warning))
        utils_->out() << "WARNING: Error caught = " << e << "\n";

      status_ = NOX::NLN::INNER::StatusTest::status_step_too_long;
    }

    // clear the exception checks after the try/catch block
    fp_except_.clear();
  }
  // -------------------------------------------------
  // print footer if desired
  // -------------------------------------------------
  utils_->out(NOX::Utils::InnerIteration) << NOX::Utils::fill(72, '=') << "\n";

  if (status_ == NOX::NLN::INNER::StatusTest::status_step_too_short)
    throwError("compute()",
        "The current step is too short and no "
        "restoration phase is implemented!");
  else if (status_ == NOX::NLN::INNER::StatusTest::status_no_descent_direction)
    throwError("compute()", "The given search direction is no descent direction!");

  fp_except_.enable();
  return (status_ == NOX::NLN::INNER::StatusTest::status_converged ? true : false);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const int& NOX::NLN::LineSearch::Backtrack::GetNumIterations() const { return lsIters_; }

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const NOX::MeritFunction::Generic& NOX::NLN::LineSearch::Backtrack::GetMeritFunction() const
{
  if (meritFunctionPtr_.is_null())
    throwError("GetMeritFunction", "The merit function pointer is not initialized!");

  return *meritFunctionPtr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const NOX::Abstract::Vector& NOX::NLN::LineSearch::Backtrack::GetSearchDirection() const
{
  if (searchDirectionPtr_.is_null())
    throwError("GetSearchDirection", "The search direction ptr is not initialized!");

  return *searchDirectionPtr_;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const double& NOX::NLN::LineSearch::Backtrack::GetStepLength() const
{
  if (stepPtr_ == NULL) throwError("GetStepLength", "Step pointer is NULL!");

  return *stepPtr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::LineSearch::Backtrack::SetStepLength(double step)
{
  if (stepPtr_ == NULL) throwError("GetMutableStepLength", "Step pointer is NULL!");

  *stepPtr_ = step;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::NLN::INNER::StatusTest::StatusType NOX::NLN::LineSearch::Backtrack::CheckInnerStatus(
    const NOX::Solver::Generic& solver, const NOX::Abstract::Group& grp,
    NOX::StatusTest::CheckType checkType) const
{
  return innerTestsPtr_->CheckStatus(*this, solver, grp, checkType);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::LineSearch::Backtrack::PrintUpdate(std::ostream& os) const
{
  // Print the status test parameters at each iteration if requested
  if (status_ == NOX::NLN::INNER::StatusTest::status_step_too_long)
  {
    os << NOX::Utils::fill(72, '-') << "\n";
    os << "-- Inner Status Test Results --\n";
    innerTestsPtr_->Print(os);
    os << NOX::Utils::fill(72, '-') << "\n";
  }
  // Print the final parameter values of the status test
  if (status_ != NOX::NLN::INNER::StatusTest::status_step_too_long)
  {
    os << NOX::Utils::fill(72, '-') << "\n";
    os << "-- Final Inner Status Test Results --\n";
    innerTestsPtr_->Print(os);
    os << NOX::Utils::fill(72, '-') << "\n";
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::LineSearch::Backtrack::throwError(
    const std::string& functionName, const std::string& errorMsg) const
{
  std::ostringstream msg;
  msg << "ERROR - NOX::NLN::LineSearch::Backtrack::" << functionName << " - " << errorMsg
      << std::endl;
  dserror(msg.str());
}
