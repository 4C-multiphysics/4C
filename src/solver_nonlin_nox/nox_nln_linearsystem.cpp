/*-----------------------------------------------------------*/
/*!
\file nox_nln_linearsystem.cpp

\maintainer Michael Hiermeier

\date Jul 2, 2015

\level 3

*/
/*-----------------------------------------------------------*/

#include "nox_nln_linearsystem.H"
#include "nox_nln_interface_jacobian.H"
#include "nox_nln_interface_required.H"
#include "nox_nln_linearsystem_prepostoperator.H"
#include "nox_nln_solver_ptc.H"

#include "../linalg/linalg_utils.H"
#include "../linalg/linalg_solver.H"
#include "../linalg/linalg_blocksparsematrix.H"

#include <Epetra_Vector.h>
#include <Epetra_LinearProblem.h>

#include <Teuchos_ParameterList.hpp>

#include <NOX_Epetra_Scaling.H>
#include <NOX_Epetra_Interface_Preconditioner.H>

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::NLN::LinearSystem::LinearSystem(
    Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::map<NOX::NLN::SolutionType,Teuchos::RCP<LINALG::Solver> >& solvers,
    const Teuchos::RCP<NOX::Epetra::Interface::Required>& iReq,
    const Teuchos::RCP<NOX::Epetra::Interface::Jacobian>& iJac,
    const Teuchos::RCP<LINALG::SparseOperator>& jacobian,
    const Teuchos::RCP<NOX::Epetra::Interface::Preconditioner>& iPrec,
    const Teuchos::RCP<LINALG::SparseOperator>& preconditioner,
    const NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<NOX::Epetra::Scaling> scalingObject)
    : linParams_(linearSolverParams),
      utils_(printParams),
      solvers_(solvers),
      reqInterfacePtr_(iReq),
      jacInterfacePtr_(iJac),
      jacType_(LinalgSparseOperator),
      jacPtr_(jacobian),
      precInterfacePtr_(iPrec),
      precType_(LinalgSparseOperator),
      precPtr_(preconditioner),
      precMatrixSource_(SeparateMatrix),
      scaling_(scalingObject),
      conditionNumberEstimate_(0.0),
      timer_(cloneVector.getEpetraVector().Comm()),
      timeCreatePreconditioner_(0.0),
      timeApplyJacbianInverse_(0.0),
      resNorm2_(0.0),
      prePostOperatorPtr_(Teuchos::null)
{
  // Jacobian operator is supplied
  jacType_ = getOperatorType(*jacPtr_);

  reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::NLN::LinearSystem::LinearSystem(
    Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::map<NOX::NLN::SolutionType,Teuchos::RCP<LINALG::Solver> >& solvers,
    const Teuchos::RCP<NOX::Epetra::Interface::Required>& iReq,
    const Teuchos::RCP<NOX::Epetra::Interface::Jacobian>& iJac,
    const Teuchos::RCP<LINALG::SparseOperator>& jacobian,
    const Teuchos::RCP<NOX::Epetra::Interface::Preconditioner>& iPrec,
    const Teuchos::RCP<LINALG::SparseOperator>& preconditioner,
    const NOX::Epetra::Vector& cloneVector)
    : linParams_(linearSolverParams),
      utils_(printParams),
      solvers_(solvers),
      reqInterfacePtr_(iReq),
      jacInterfacePtr_(iJac),
      jacType_(LinalgSparseOperator),
      jacPtr_(jacobian),
      precInterfacePtr_(iPrec),
      precType_(LinalgSparseOperator),
      precPtr_(preconditioner),
      precMatrixSource_(SeparateMatrix),
      scaling_(Teuchos::null),
      conditionNumberEstimate_(0.0),
      timer_(cloneVector.getEpetraVector().Comm()),
      timeCreatePreconditioner_(0.0),
      timeApplyJacbianInverse_(0.0),
      resNorm2_(0.0),
      prePostOperatorPtr_(Teuchos::null)
{
  // Jacobian operator is supplied
  jacType_ = getOperatorType(*jacPtr_);

  reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::NLN::LinearSystem::LinearSystem(
    Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::map<NOX::NLN::SolutionType,Teuchos::RCP<LINALG::Solver> >& solvers,
    const Teuchos::RCP<NOX::Epetra::Interface::Required>& iReq,
    const Teuchos::RCP<NOX::Epetra::Interface::Jacobian>& iJac,
    const Teuchos::RCP<LINALG::SparseOperator>& jacobian,
    const NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<NOX::Epetra::Scaling> scalingObject)
    : linParams_(linearSolverParams),
      utils_(printParams),
      solvers_(solvers),
      reqInterfacePtr_(iReq),
      jacInterfacePtr_(iJac),
      jacType_(LinalgSparseOperator),
      jacPtr_(jacobian),
      precInterfacePtr_(Teuchos::null),
      precType_(LinalgSparseOperator),
      precPtr_(Teuchos::null),
      precMatrixSource_(SeparateMatrix),
      scaling_(Teuchos::null),
      conditionNumberEstimate_(0.0),
      timer_(cloneVector.getEpetraVector().Comm()),
      timeCreatePreconditioner_(0.0),
      timeApplyJacbianInverse_(0.0),
      resNorm2_(0.0),
      prePostOperatorPtr_(Teuchos::null)
{
  // Jacobian operator is supplied
  jacType_ = getOperatorType(*jacPtr_);

  reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::NLN::LinearSystem::LinearSystem(
    Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams,
    const std::map<NOX::NLN::SolutionType,Teuchos::RCP<LINALG::Solver> >& solvers,
    const Teuchos::RCP<NOX::Epetra::Interface::Required>& iReq,
    const Teuchos::RCP<NOX::Epetra::Interface::Jacobian>& iJac,
    const Teuchos::RCP<LINALG::SparseOperator>& jacobian,
    const NOX::Epetra::Vector& cloneVector)
    : linParams_(linearSolverParams),
      utils_(printParams),
      solvers_(solvers),
      reqInterfacePtr_(iReq),
      jacInterfacePtr_(iJac),
      jacType_(LinalgSparseOperator),
      jacPtr_(jacobian),
      precInterfacePtr_(Teuchos::null),
      precType_(LinalgSparseOperator),
      precPtr_(Teuchos::null),
      precMatrixSource_(SeparateMatrix),
      scaling_(Teuchos::null),
      conditionNumberEstimate_(0.0),
      timer_(cloneVector.getEpetraVector().Comm()),
      timeCreatePreconditioner_(0.0),
      timeApplyJacbianInverse_(0.0),
      resNorm2_(0.0),
      prePostOperatorPtr_(Teuchos::null)
{
  // Jacobian operator is supplied
  jacType_ = getOperatorType(*jacPtr_);

  reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::LinearSystem::reset()
{
  zeroInitialGuess_ = linParams_.get("Zero Initial Guess", false);

  manualScaling_ = linParams_.get("Compute Scaling Manually", true);

  // Place linear solver details in the "Output" sublist of the
  // "Linear Solver" parameter list
  outputSolveDetails_ = linParams_.get("Output Solver Details", true);

  // set the pre/post-operator
  if (prePostOperatorPtr_.is_null())
    prePostOperatorPtr_ =
        Teuchos::rcp(new NOX::NLN::LinSystem::PrePostOperator(linParams_));
  else
    prePostOperatorPtr_->reset(linParams_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::NLN::LinearSystem::OperatorType NOX::NLN::LinearSystem::getOperatorType(const LINALG::SparseOperator& op) const
{
  const Epetra_Operator* testOperator = 0;

  // Is it a LINALG_BlockSparseMatrix
  testOperator = dynamic_cast<const LINALG::BlockSparseMatrix<LINALG::DefaultBlockMatrixStrategy>*>(&op);
  if (testOperator != 0)
    return LinalgBlockSparseMatrix;

  // Is it a LINALG_SparseMatrix?
  testOperator = dynamic_cast<const LINALG::SparseMatrix*>(&op);
  if (testOperator != 0)
    return LinalgSparseMatrix;

  // Is it a LINALG_SparseMatrixBase?
  testOperator = dynamic_cast<const LINALG::SparseMatrixBase*>(&op);
  if (testOperator != 0)
    return LinalgSparseMatrixBase;

  // Otherwise it must be a LINALG_SparseOperator
  return LinalgSparseOperator;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::NLN::LinearSystem::applyJacobian(
    const NOX::Epetra::Vector& input,
          NOX::Epetra::Vector& result) const
{
  jacPtr_->SetUseTranspose(false);
  int status = jacPtr_->Apply(input.getEpetraVector(),
                  result.getEpetraVector());
  return (status == 0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::NLN::LinearSystem::applyJacobianTranspose(
    const NOX::Epetra::Vector& input,
          NOX::Epetra::Vector& result) const
{
  // Apply the Jacobian
  jacPtr_->SetUseTranspose(true);
  int status = jacPtr_->Apply(input.getEpetraVector(),
                  result.getEpetraVector());
  jacPtr_->SetUseTranspose(false);

  return (status == 0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::NLN::LinearSystem::applyJacobianInverse(
    Teuchos::ParameterList& linearSolverParams,
    const NOX::Epetra::Vector& input,
    NOX::Epetra::Vector& result)
{
  /* Need non-const version of the input vector
   * Epetra_LinearProblem requires non-const versions so we can perform
   * scaling of the linear problem.
   * Same is valid for the prePostOperator. We want to have the
   * possibility to change the linear system. */
  NOX::Epetra::Vector& nonConstInput = const_cast<NOX::Epetra::Vector&>(input);

  prePostOperatorPtr_->runPreApplyJacobianInverse(nonConstInput,*jacPtr_,*this);

  double startTime = timer_.WallTime();

  // calculate the residual norm
  resNorm2_ = nonConstInput.norm(NOX::Abstract::Vector::TwoNorm);

  // Zero out the delta X of the linear problem if requested by user.
  if (zeroInitialGuess_)
    result.init(0.0);

  // Create Epetra linear problem object for the linear solve
  /* Note: We switch from LINALG_objects to pure Epetra_objects.
   * This is necessary for the linear solver.
   *     LINALG::SparseMatrix ---> Epetra_CrsMatrix */
  Epetra_LinearProblem linProblem(&(*jacPtr_->EpetraOperator()),
      &(result.getEpetraVector()),
      &(nonConstInput.getEpetraVector()));

  // ************* Begin linear system scaling *****************
  if ( !Teuchos::is_null(scaling_) )
  {
    if ( !manualScaling_ )
      scaling_->computeScaling(linProblem);

    scaling_->scaleLinearSystem(linProblem);

    if (utils_.isPrintType(NOX::Utils::Details))
      utils_.out() << *scaling_ << std::endl;
  }
  // ************* End linear system scaling *******************

  // get current linear solver from the std_map
  Teuchos::RCP<LINALG::Solver> currSolver;
  NOX::NLN::SolutionType solType = GetCurrentLinSolver(solvers_,currSolver);

  // set solver options if necessary
  SetSolverOptions(currSolver,solType);

  // solve
  int iter = linearSolverParams.get<int>("Number of Nonlinear Iterations",-10);
  if (iter==-10)
    throwError("applyJacobianInverse", "\"Number of Nonlinear Iterations\" was not specified");

  if (currSolver->NoxSolve(linProblem,true,iter==0))
      throwError("applyJacobianInverse", "linear solve failed");

  // ************* Begin linear system unscaling *************
  if ( !Teuchos::is_null(scaling_) )
    scaling_->unscaleLinearSystem(linProblem);
  // ************* End linear system unscaling ***************

  double endTime = timer_.WallTime();
  timeApplyJacbianInverse_ += (endTime - startTime);

  prePostOperatorPtr_->runPostApplyJacobianInverse(nonConstInput,*jacPtr_,*this);

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::NLN::LinearSystem::applyRightPreconditioning(bool useTranspose,
    Teuchos::ParameterList& params,
    const NOX::Epetra::Vector& input,
    NOX::Epetra::Vector& result) const
{
  if (&result != &input)
    result = input;
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::NLN::LinearSystem::computeJacobian(const NOX::Epetra::Vector& x)
{
  prePostOperatorPtr_->runPreComputeJacobian(*jacPtr_,x.getEpetraVector(),
      *this);

  bool success = jacInterfacePtr_->computeJacobian(x.getEpetraVector(),
                          *jacPtr_);

  prePostOperatorPtr_->runPostComputeJacobian(*jacPtr_,x.getEpetraVector(),
      *this);
  return success;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::NLN::LinearSystem::computeFandJacobian(
    const NOX::Epetra::Vector& x,
    NOX::Epetra::Vector& rhs)
{
  prePostOperatorPtr_->runPreComputeFandJacobian(rhs.getEpetraVector(),
      *jacPtr_,x.getEpetraVector(),*this);

  bool success = Teuchos::rcp_dynamic_cast<NOX::NLN::Interface::Jacobian>
      (jacInterfacePtr_)->computeFandJacobian(x.getEpetraVector(),
      rhs.getEpetraVector(),*jacPtr_);

  prePostOperatorPtr_->runPostComputeFandJacobian(rhs.getEpetraVector(),
      *jacPtr_,x.getEpetraVector(),*this);
  return success;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP< NOX::Epetra::Scaling> NOX::NLN::LinearSystem::getScaling()
{
  return scaling_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::LinearSystem::resetScaling(const Teuchos::RCP<NOX::Epetra::Scaling>& scalingObject)
{
  scaling_ = scalingObject;
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::LinearSystem::adjustPseudoTimeStep(
    double& delta,
    const double& stepSize,
    const NOX::Epetra::Vector& dir,
    const NOX::Epetra::Vector& rhs,
    const NOX::NLN::Solver::PseudoTransient& ptcsolver)
{
  const Epetra_Vector& scalingDiagOp = ptcsolver.getScalingDiagOperator();
  // ---------------------------------------------------------------------
  // first undo the modification of the jacobian
  // ---------------------------------------------------------------------
  Teuchos::RCP<Epetra_Vector> v =
      Teuchos::rcp(new Epetra_Vector(scalingDiagOp));
  v->Scale(ptcsolver.getInversePseudoTimeStep());
  Teuchos::RCP<LINALG::SparseMatrix> jac =
      Teuchos::rcp_dynamic_cast<LINALG::SparseMatrix>(jacPtr_);
  if (jac.is_null())
    throwError("adjustPseudoTimeStep()","Cast to LINALG::SparseMatrix failed!");
  // get the diagonal terms of the jacobian
  Teuchos::RCP<Epetra_Vector> diag = LINALG::CreateVector(jac->RowMap(),false);
  jac->ExtractDiagonalCopy(*diag);
  diag->Update(-1.0,*v,1.0);
  // Finally undo the changes
  jac->ReplaceDiagonalValues(*diag);

  // ---------------------------------------------------------------------
  // calculate the least squares approximated corrected pseudo time step
  // ---------------------------------------------------------------------
  /* evaluate the first vector:
   *    eta^{-1} F_{n-1} + (\nabla_{x} F_{n-1})^{T} d_{n-1}             */
  double stepSizeInv = 1.0/stepSize;
  Teuchos::RCP<Epetra_Vector> vec_1 = LINALG::CreateVector(jac->RowMap(),true);
  Teuchos::RCP<Epetra_Vector> vec_2 = Teuchos::rcp(new Epetra_Vector(rhs.getEpetraVector()));
  jac->Multiply(false,dir.getEpetraVector(),*vec_1);
  vec_2->Scale(stepSizeInv);
  vec_1->Update(1.0,*vec_2,1.0);
  /* evaluate the second vector:              d^{T} V                   */
  vec_2->Multiply(1.0,scalingDiagOp,dir.getEpetraVector(),0.0);

  // finally evaluate the scalar product
  double numerator = 0.0;
  double denominator = 0.0;
  vec_2->Dot(*vec_1,&numerator);
  vec_1->Dot(*vec_1,&denominator);

  // ----------------------------------------------------
  // show the error (L2-norm)
  // ----------------------------------------------------
  Teuchos::RCP<Epetra_Vector> vec_err = LINALG::CreateVector(jac->RowMap(),true);
  vec_err->Update(delta,*vec_1,1.0,*vec_2,0.0);
  double error_start = 0.0;
  vec_err->Norm2(&error_start);

  delta = - numerator/denominator;

  // ----------------------------------------------------
  // show the actual remaining error (L2-norm)
  // ----------------------------------------------------
  vec_err->Update(delta,*vec_1,1.0,*vec_2,0.0);
  double error_end = 0.0;
  vec_err->Norm2(&error_end);
  std::cout << "error: " << error_start << " --> " << error_end << std::endl;


}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::LinearSystem::throwError(
    const std::string& functionName,
    const std::string& errorMsg) const
{
  if (utils_.isPrintType(NOX::Utils::Error)) {
    utils_.out() << "NOX::NLN::LinearSystem::" << functionName
     << " - " << errorMsg << std::endl;
  }
  throw "NOX Error";
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const NOX::Epetra::Interface::Required> NOX::NLN::LinearSystem::getRequiredInterface() const
{
  return reqInterfacePtr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const NOX::Epetra::Interface::Jacobian> NOX::NLN::LinearSystem::getJacobianInterface() const
{
  return jacInterfacePtr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const NOX::Epetra::Interface::Preconditioner> NOX::NLN::LinearSystem::getPrecInterface() const
{
  return precInterfacePtr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Operator> NOX::NLN::LinearSystem::getJacobianOperator() const
{
  return jacPtr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Operator> NOX::NLN::LinearSystem::getJacobianOperator()
{
  return jacPtr_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const enum NOX::NLN::LinearSystem::OperatorType& NOX::NLN::LinearSystem::
    getJacobianOperatorType() const
{
  return jacType_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::LinearSystem::setJacobianOperatorForSolve(const Teuchos::RCP<const Epetra_Operator>& solveJacOp)
{
  const Teuchos::RCP<const LINALG::SparseOperator>& linalgSprOp =
      Teuchos::rcp_dynamic_cast<const LINALG::SparseOperator>(solveJacOp);
  if (linalgSprOp.is_null())
    throwError("setJacobianOperatorForSolve","dynamic_cast to LINALG_SparseOperator failed!");

  SetJacobianOperatorForSolve(linalgSprOp);
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::LinearSystem::SetJacobianOperatorForSolve(const Teuchos::RCP<const LINALG::SparseOperator>& solveJacOp)
{
  if (jacType_ != getOperatorType(*solveJacOp))
    throwError("SetJacobianOperatorForSolve","wrong operator type!");

  jacPtr_ = Teuchos::rcp_const_cast<LINALG::SparseOperator>(solveJacOp);
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::NLN::LinearSystem::createPreconditioner(const NOX::Epetra::Vector& x,
                                                  Teuchos::ParameterList& p,
                                                  bool recomputeGraph) const
{
  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::NLN::LinearSystem::destroyPreconditioner() const

{
  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::NLN::LinearSystem::recomputePreconditioner(const NOX::Epetra::Vector& x,
                                                     Teuchos::ParameterList& linearSolverParams) const
{
  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Epetra::LinearSystem::PreconditionerReusePolicyType
NOX::NLN::LinearSystem::getPreconditionerPolicy(bool advanceReuseCounter)
{
  return PRPT_REBUILD;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::NLN::LinearSystem::isPreconditionerConstructed() const
{
  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool NOX::NLN::LinearSystem::hasPreconditioner() const
{
  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Operator> NOX::NLN::LinearSystem::getGeneratedPrecOperator() const
{
  return Teuchos::null;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Operator> NOX::NLN::LinearSystem::getGeneratedPrecOperator()
{
  return Teuchos::null;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::LinearSystem::setPrecOperatorForSolve(const Teuchos::RCP<const Epetra_Operator>& solvePrecOp)
{
  throwError("setPrecOperatorForSolve", "no preconditioner supported");
  return;
}
