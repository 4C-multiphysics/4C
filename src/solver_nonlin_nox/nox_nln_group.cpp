/*-----------------------------------------------------------*/
/*!
\file nox_nln_group.cpp

\maintainer Michael Hiermeier

\date Jun 29, 2015

\level 3

*/
/*-----------------------------------------------------------*/

#include "nox_nln_group.H"
#include "nox_nln_interface_required.H"
#include "nox_nln_linearsystem.H"
#include "nox_nln_group_prepostoperator.H"
#include "nox_nln_solver_ptc.H"

#include <NOX_StatusTest_NormF.H>

#include "../linalg/linalg_utils.H"


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::NLN::Group::Group(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& grpOptionParams,
    const Teuchos::RCP<NOX::Epetra::Interface::Required>& i,
    const NOX::Epetra::Vector& x,
    const Teuchos::RCP<NOX::Epetra::LinearSystem>& linSys)
    : NOX::Epetra::Group(printParams,i,x,linSys),
      grpOptionParams_(grpOptionParams),
      prePostOperatorPtr_(Teuchos::rcp(new NOX::NLN::GROUP::PrePostOperator(grpOptionParams)))
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::NLN::Group::Group(const NOX::NLN::Group& source, CopyType type)
    : NOX::Epetra::Group(source,type),
      grpOptionParams_(source.grpOptionParams_),
      prePostOperatorPtr_(source.prePostOperatorPtr_)
{
  // no new variables, pointers or references
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<NOX::Abstract::Group> NOX::NLN::Group::clone(CopyType type) const
{
  Teuchos::RCP<NOX::Abstract::Group> newgrp =
    Teuchos::rcp(new NOX::NLN::Group(*this, type));
  return newgrp;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::Group::computeX(const NOX::Abstract::Group& grp,
    const NOX::Abstract::Vector& d,
    double step)
{
  // Cast to appropriate type, then call the "native" computeX
  const NOX::NLN::Group* nlngrp = dynamic_cast<const NOX::NLN::Group*> (&grp);
  if (nlngrp == NULL)
    throwError("computeX","dyn_cast to nox_nln_group failed!");
  const NOX::Epetra::Vector& epetrad =
    dynamic_cast<const NOX::Epetra::Vector&> (d);

  computeX(*nlngrp, epetrad, step);
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::Group::computeX(
    const NOX::NLN::Group& grp,
    const NOX::Epetra::Vector& d,
    double step)
{
  // -------------------------------------
  // Update the external variables
  // -------------------------------------
  // We have to remove the const state, because we want to change class variables
  // of the required interface.
  Teuchos::RCP<NOX::NLN::Interface::Required> iReq =
      Teuchos::rcp_const_cast<NOX::NLN::Interface::Required>(GetNlnReqInterfacePtr());
  iReq->SetPrimarySolution(grp.xVector,d,step);
  // -------------------------------------
  // Update the nox internal variables
  // -------------------------------------
  NOX::Epetra::Group::computeX(grp,d,step);

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Abstract::Group::ReturnType NOX::NLN::Group::setF(
    const Teuchos::RCP<const NOX::Epetra::Vector> Fptr)
{
  if (Fptr == Teuchos::null or Fptr->getEpetraVector().Map().NumGlobalElements()==0)
    return NOX::Abstract::Group::BadDependency;

  RHSVectorPtr =  Teuchos::rcp_const_cast<NOX::Epetra::Vector>(Fptr);
  isValidRHS = true;

  return NOX::Abstract::Group::Ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Abstract::Group::ReturnType NOX::NLN::Group::setJacobianOperator(
    const Teuchos::RCP<const Epetra_Operator> jacOperator)
{
  if (jacOperator==Teuchos::null or jacOperator->OperatorRangeMap().NumGlobalElements()==0)
    return NOX::Abstract::Group::BadDependency;

  sharedLinearSystem.getObject(this)->setJacobianOperatorForSolve(jacOperator);
  isValidJacobian = true;

  return NOX::Abstract::Group::Ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Abstract::Group::ReturnType NOX::NLN::Group::computeF()
{
  prePostOperatorPtr_->runPreComputeF(RHSVector.getEpetraVector(),*this);

  NOX::Abstract::Group::ReturnType ret = Epetra::Group::computeF();

  prePostOperatorPtr_->runPostComputeF(RHSVector.getEpetraVector(),*this);
  return ret;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Abstract::Group::ReturnType NOX::NLN::Group::computeFandJacobian()
{
  // initialize the return type
  NOX::Abstract::Group::ReturnType ret = NOX::Abstract::Group::Failed;

  // update right hand side vector
  if (!isF() and isJacobian())
  {
    ret = computeF();
  }
  // update right hand side vector and jacobian
  else if (!isJacobian())
  {
    isValidRHS = false;
    prePostOperatorPtr_->runPreComputeF(RHSVector.getEpetraVector(),*this);
    bool status = false;
    Teuchos::RCP<NOX::NLN::LinearSystem> nlnSharedLinearSystem =
        Teuchos::rcp_dynamic_cast<NOX::NLN::LinearSystem>(sharedLinearSystem.getObject(this));

    if (nlnSharedLinearSystem.is_null())
      throwError("computeFandJacobian","Dynamic cast of the shared linear system failed!");

    status = nlnSharedLinearSystem->computeFandJacobian(xVector,
                                                        RHSVector);
    if (!status)
      throwError("computeFandJacobian","evaluation failed!");

    isValidRHS = true;
    isValidJacobian = true;

    ret = NOX::Abstract::Group::Ok;
    prePostOperatorPtr_->runPostComputeF(RHSVector.getEpetraVector(),*this);
  }
  // nothing to do, because all quantities are up-to-date
  else
  {
    prePostOperatorPtr_->runPreComputeF(RHSVector.getEpetraVector(),*this);
    ret = NOX::Abstract::Group::Ok;
  }

  return ret;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const NOX::NLN::Interface::Required> NOX::NLN::Group::GetNlnReqInterfacePtr() const
{
  Teuchos::RCP<NOX::NLN::Interface::Required> userInterfaceNlnPtr =
      Teuchos::rcp_dynamic_cast<NOX::NLN::Interface::Required>(userInterfacePtr);

  if (userInterfaceNlnPtr.is_null())
    throwError("GetNlnReqInterfacePtr",
        "Dynamic cast of the userInterfacePtr to NOX::NLN::Interface::Required failed!");

  return userInterfaceNlnPtr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<const std::vector<double> > NOX::NLN::Group::GetRHSNorms(
    const std::vector<NOX::Abstract::Vector::NormType>& type,
    const std::vector<NOX::NLN::StatusTest::QuantityType>& chQ,
    Teuchos::RCP<const std::vector<NOX::StatusTest::NormF::ScaleType> > scale) const
{
  if (scale.is_null())
    scale = Teuchos::rcp(new std::vector<NOX::StatusTest::NormF::ScaleType>(chQ.size(),
        NOX::StatusTest::NormF::Unscaled));

  Teuchos::RCP<std::vector<double> > norms = Teuchos::rcp(new std::vector<double>(0));

  double rval = -1.0;
  for (std::size_t i=0;i<chQ.size();++i)
  {
    rval = GetNlnReqInterfacePtr()->GetPrimaryRHSNorms(chQ[i],type[i],
        (*scale)[i]==NOX::StatusTest::NormF::Scaled);
    if (rval>=0.0)
    {
      norms->push_back(rval);
    }
    else
    {
      std::ostringstream msg;
      msg << "The desired quantity"
          " for the \"NormF\" Status Test could not be found! (enum="
          << chQ[i] << " | " << NOX::NLN::StatusTest::QuantityType2String(chQ[i])
          << ")" << std::endl;
      throwError("GetRHSNorms",msg.str());
    }
  }

  return norms;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<std::vector<double> > NOX::NLN::Group::GetSolutionUpdateRMS(
    const std::vector<double>& aTol, const std::vector<double>& rTol,
    const std::vector<NOX::NLN::StatusTest::QuantityType>& chQ,
    const std::vector<bool>& disable_implicit_weighting) const
{
  Teuchos::RCP<std::vector<double> > rms = Teuchos::rcp(new std::vector<double>(0));

  double rval = -1.0;
  for (std::size_t i=0;i<chQ.size();++i)
  {
    rval = GetNlnReqInterfacePtr()->GetPrimarySolutionUpdateRMS(
        aTol[i],rTol[i],chQ[i],disable_implicit_weighting[i]);
    if (rval>=0.0)
    {
      rms->push_back(rval);
    }
    else
    {
      std::ostringstream msg;
      msg << "The desired quantity"
          " for the \"NormWRMS\" Status Test could not be found! (enum="
          << chQ[i] << " | " << NOX::NLN::StatusTest::QuantityType2String(chQ[i])
          << ")" << std::endl;
      throwError("GetSolutionUpdateRMS",msg.str());
    }
  }

  return rms;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<std::vector<double> > NOX::NLN::Group::GetSolutionUpdateNorms(
    const std::vector<NOX::Abstract::Vector::NormType>& type,
    const std::vector<StatusTest::QuantityType>& chQ,
    Teuchos::RCP<const std::vector<StatusTest::NormUpdate::ScaleType> > scale) const
{
  if (scale.is_null())
    scale = Teuchos::rcp(new std::vector<StatusTest::NormUpdate::ScaleType>(chQ.size(),
        StatusTest::NormUpdate::Unscaled));

  Teuchos::RCP<std::vector<double> > norms = Teuchos::rcp(new std::vector<double>(0));

  double rval = -1.0;
  for (std::size_t i=0;i<chQ.size();++i)
  {
    rval = GetNlnReqInterfacePtr()->GetPrimarySolutionUpdateNorms(chQ[i],type[i],
        (*scale)[i]==StatusTest::NormUpdate::Scaled);
    if (rval>=0.0)
    {
      norms->push_back(rval);
    }
    else
    {
      std::ostringstream msg;
      msg << "The desired quantity"
          " for the \"NormIncr\" Status Test could not be found! (enum="
          << chQ[i] << " | " << NOX::NLN::StatusTest::QuantityType2String(chQ[i])
          << ")" << std::endl;
      throwError("GetSolutionUpdateNorms",msg.str());
    }
  }

  return norms;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<std::vector<double> > NOX::NLN::Group::GetPreviousSolutionNorms(
    const std::vector<NOX::Abstract::Vector::NormType>& type,
    const std::vector<StatusTest::QuantityType>& chQ,
    Teuchos::RCP<const std::vector<StatusTest::NormUpdate::ScaleType> > scale) const
{
  if (scale.is_null())
    scale = Teuchos::rcp(new std::vector<StatusTest::NormUpdate::ScaleType>(chQ.size(),
        StatusTest::NormUpdate::Unscaled));

  Teuchos::RCP<std::vector<double> > norms = Teuchos::rcp(new std::vector<double>(0));

  double rval = -1.0;
  for (std::size_t i=0;i<chQ.size();++i)
  {
    rval = GetNlnReqInterfacePtr()->GetPreviousPrimarySolutionNorms(chQ[i],type[i],
        (*scale)[i]==StatusTest::NormUpdate::Scaled);
    if (rval>=0.0)
    {
      norms->push_back(rval);
    }
    else
    {
      std::ostringstream msg;
      msg << "The desired quantity"
          " for the \"NormUpdate\" Status Test could not be found! (enum="
          << chQ[i] << " | " << NOX::NLN::StatusTest::QuantityType2String(chQ[i])
          << ")" << std::endl;
      throwError("GetPreviousSolutionNorms",msg.str());
    }
  }

  return norms;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double NOX::NLN::Group::GetObjectiveModelValue(const std::string& name) const
{
  return GetNlnReqInterfacePtr()->GetObjectiveModelValue(name);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::Group::ResetPrePostOpertator()
{
  prePostOperatorPtr_->reset(grpOptionParams_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::Group::adjustPseudoTimeStep(double& delta,
    const double& stepSize,
    const NOX::Abstract::Vector& dir,
    const NOX::NLN::Solver::PseudoTransient& ptcsolver)
{
  const NOX::Epetra::Vector& dirEpetra =
      dynamic_cast<const NOX::Epetra::Vector&>(dir);
  adjustPseudoTimeStep(delta,stepSize,dirEpetra,ptcsolver);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::Group::adjustPseudoTimeStep(double& delta,
    const double& stepSize,
    const NOX::Epetra::Vector& dir,
    const NOX::NLN::Solver::PseudoTransient& ptcsolver)
{
  if (!isF() or !isJacobian())
    throwError("AdjustPseudoTimeStep",
        "F and/or the jacobian are not evaluated!");

  Teuchos::RCP<NOX::NLN::LinearSystem> nlnSharedLinearSystem =
      Teuchos::rcp_dynamic_cast<NOX::NLN::LinearSystem>(
      sharedLinearSystem.getObject(this));

  if (nlnSharedLinearSystem.is_null())
    throwError("AdjustPseudoTimeStep()",
        "Dynamic cast of the shared linear system failed!");

  nlnSharedLinearSystem->adjustPseudoTimeStep(delta,stepSize,dir,RHSVector,
      ptcsolver);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> NOX::NLN::Group::GetLumpedMassMatrixPtr() const
{
  return Teuchos::rcp_dynamic_cast<NOX::NLN::Interface::Required>(
      userInterfacePtr)->GetLumpedMassMatrixPtr();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::Group::throwError(
    const std::string& functionName,
    const std::string& errorMsg) const
{
  std::ostringstream msg;
  msg << "ERROR - NOX::NLN::Group::" << functionName
      << " - " << errorMsg << std::endl;

  dserror(msg.str());
}
