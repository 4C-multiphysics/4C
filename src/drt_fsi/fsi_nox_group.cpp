#ifdef CCADISCRET

#include "fsi_nox_group.H"


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::FSI::Group::Group(::FSI::Monolithic& mfsi,
                       Teuchos::ParameterList& printParams,
                       const Teuchos::RCP<NOX::Epetra::Interface::Required>& i,
                       const NOX::Epetra::Vector& x,
                       const Teuchos::RCP<NOX::Epetra::LinearSystem>& linSys)
  : NOX::Epetra::Group(printParams,i,x,linSys),
    mfsi_(mfsi)
{
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::FSI::Group::CaptureSystemState()
{
  // we know we already have the first linear system calculated

  mfsi_.SetupRHS(RHSVector.getEpetraVector(),true);
  mfsi_.SetupSystemMatrix();

  sharedLinearSystem.getObject(this);
  isValidJacobian = true;
  isValidRHS = true;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Abstract::Group::ReturnType NOX::FSI::Group::computeF()
{
  NOX::Abstract::Group::ReturnType ret = NOX::Epetra::Group::computeF();
  if (ret==NOX::Abstract::Group::Ok)
  {
    if (not isValidJacobian)
    {
      mfsi_.SetupSystemMatrix();
      sharedLinearSystem.getObject(this);
      isValidJacobian = true;
    }
  }
  return ret;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Abstract::Group::ReturnType NOX::FSI::Group::computeJacobian()
{
  NOX::Abstract::Group::ReturnType ret = NOX::Epetra::Group::computeJacobian();
  if (ret==NOX::Abstract::Group::Ok)
  {
    if (not isValidRHS)
    {
      mfsi_.SetupRHS(RHSVector.getEpetraVector());
      isValidRHS = true;
    }
  }
  return ret;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Abstract::Group::ReturnType NOX::FSI::Group::computeNewton(Teuchos::ParameterList& p)
{
  mfsi_.ScaleSystem(RHSVector.getEpetraVector());
  NOX::Abstract::Group::ReturnType status = NOX::Epetra::Group::computeNewton(p);
  mfsi_.UnscaleSolution(NewtonVector.getEpetraVector());
  return status;
}

#endif
