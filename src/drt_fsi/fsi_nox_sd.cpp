
#include "fsi_nox_sd.H"

#include <NOX_Epetra_Interface_Required.H>
#include <NOX_Common.H>
#include <NOX_Abstract_Vector.H>
#include <NOX_Abstract_Group.H>
#include <NOX_Epetra_Vector.H>
#include <NOX_Epetra_Group.H>
#include <NOX_Solver_Generic.H>
#include <Teuchos_ParameterList.hpp>
#include <NOX_Utils.H>
#include <NOX_GlobalData.H>


SDRelaxation::SDRelaxation(const Teuchos::RefCountPtr<NOX::Utils>& utils,
                           Teuchos::ParameterList& params)
  : utils_(utils)
{
}


SDRelaxation::~SDRelaxation()
{
}


bool SDRelaxation::reset(const Teuchos::RefCountPtr<NOX::GlobalData>& gd,
                         Teuchos::ParameterList& params)
{
  utils_ = gd->getUtils();
  //Teuchos::ParameterList& p = params.sublist("SDRelaxation");
  return true;
}


bool SDRelaxation::compute(NOX::Abstract::Group& newgrp,
                           double& step,
                           const NOX::Abstract::Vector& dir,
                           const NOX::Solver::Generic& s)
{
  if (utils_->isPrintType(NOX::Utils::InnerIteration))
  {
    utils_->out() << "\n" << NOX::Utils::fill(72) << "\n"
                 << "-- SDRelaxation Line Search -- \n";
  }

  const NOX::Abstract::Group& oldgrp = s.getPreviousSolutionGroup();
  NOX::Epetra::Group& egrp = dynamic_cast<NOX::Epetra::Group&>(newgrp);

  // Perform single-step linesearch

  // Note that the following could be wrapped with a while loop to allow
  // iterations to be attempted

  double numerator = oldgrp.getF().innerProduct(dir);
  double denominator = computeDirectionalDerivative(dir, *egrp.getRequiredInterface())
                       .innerProduct(dir);

  step = - numerator / denominator;
  newgrp.computeX(oldgrp, dir, step);
  newgrp.computeF();

  double checkOrthogonality = fabs( newgrp.getF().innerProduct(dir) );

  if (utils_->isPrintType(NOX::Utils::InnerIteration)) {
    utils_->out() << setw(3) << "1" << ":";
    utils_->out() << " step = " << utils_->sciformat(step);
    utils_->out() << " orth = " << utils_->sciformat(checkOrthogonality);
    utils_->out() << "\n" << NOX::Utils::fill(72) << "\n" << endl;
  }

  return true;
}


NOX::Abstract::Vector&
SDRelaxation::computeDirectionalDerivative(const NOX::Abstract::Vector& dir,
                                           NOX::Epetra::Interface::Required& interface)
{
  // Allocate space for vecPtr and grpPtr if necessary
  if (Teuchos::is_null(vecPtr_))
    vecPtr_ = dir.clone(NOX::ShapeCopy);

  const NOX::Epetra::Vector& edir = dynamic_cast<const NOX::Epetra::Vector&>(dir);
  NOX::Epetra::Vector& evec = dynamic_cast<NOX::Epetra::Vector&>(*vecPtr_);

  // we do not want the group to remember this solution
  // and we want to set our own flag
  // this tells computeF to do a SD relaxation calculation
  interface.computeF(edir.getEpetraVector(),
                     evec.getEpetraVector(),
                     NOX::Epetra::Interface::Required::User);

  return *vecPtr_;
}
