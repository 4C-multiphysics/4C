/*----------------------------------------------------------------------*/
/*!
\file fsi_statustest.cpp

\brief NOX Thyra group enhancement

<pre>
Maintainer: Ulrich Kuettler
            kuettler@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15238
</pre>
*/
/*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "fsi_statustest.H"
#include "fsi_nox_newton.H"
#include "../drt_lib/drt_dserror.H"
#include "fsi_dirichletneumann.H"

#include <NOX_Common.H>
#include <NOX_Abstract_Vector.H>
#include <NOX_Abstract_Group.H>
#include <NOX_Solver_Generic.H>
#include <NOX_Utils.H>

#include <NOX_Epetra_Vector.H>

#include "../linalg/linalg_utils.H"
#include "../drt_adapter/adapter_coupling.H"



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NOX::FSI::GenericNormF::GenericNormF(std::string name,
                                     double tolerance,
                                     NOX::Abstract::Vector::NormType normType,
                                     ScaleType stype)
  : status_(NOX::StatusTest::Unevaluated),
    normType_(normType),
    scaleType_(stype),
    toleranceType_(Absolute),
    specifiedTolerance_(tolerance),
    initialTolerance_(1.0),
    trueTolerance_(tolerance),
    normF_(0.0),
    name_(name)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double NOX::FSI::GenericNormF::computeNorm(const Epetra_Vector& v)
{
  int n = v.GlobalLength();
  double norm;
  int err;

  switch (normType_)
  {
  case NOX::Abstract::Vector::TwoNorm:
    err = v.Norm2(&norm);
    if (err!=0)
      dserror("norm failed");
    if (scaleType_ == Scaled)
      norm /= sqrt(1.0 * n);
    break;

  case NOX::Abstract::Vector::OneNorm:
    err = v.Norm1(&norm);
    if (err!=0)
      dserror("norm failed");
    if (scaleType_ == Scaled)
      norm /= n;
    break;

  case NOX::Abstract::Vector::MaxNorm:
    err = v.NormInf(&norm);
    if (err!=0)
      dserror("norm failed");
    if (scaleType_ == Scaled)
      norm /= n;
    break;

  default:
    dserror("norm type confusion");
    break;
  }

  return norm;
}


#if 0
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void NOX::FSI::GenericNormF::relativeSetup(NOX::Abstract::Group& initialGuess)
{
  NOX::Abstract::Group::ReturnType rtype;
  rtype = initialGuess.computeF();
  if (rtype != NOX::Abstract::Group::Ok)
  {
    utils.err() << "NOX::StatusTest::NormF::NormF - Unable to compute F"
		<< endl;
    throw "NOX Error";
  }

  initialTolerance_ = computeNorm(initialGuess);
  trueTolerance_ = specifiedTolerance_ / initialTolerance_;
}
#endif


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NOX::StatusTest::StatusType
NOX::FSI::GenericNormF::checkStatus(const NOX::Solver::Generic& problem,
                                    NOX::StatusTest::CheckType checkType)
{
  if (checkType == NOX::StatusTest::None)
  {
    normF_ = 0.0;
    status_ = NOX::StatusTest::Unevaluated;
  }
  else
  {
    normF_ = computeNorm( problem.getSolutionGroup() );
    if ((normF_ != -1) and (normF_ < trueTolerance_))
    {
      status_ = NOX::StatusTest::Converged;
    }
    else
    {
      status_ = NOX::StatusTest::Unconverged;
    }
  }

  return status_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NOX::StatusTest::StatusType NOX::FSI::GenericNormF::getStatus() const
{
  return status_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::ostream& NOX::FSI::GenericNormF::print(std::ostream& stream, int indent) const
{
  for (int j = 0; j < indent; j ++)
    stream << ' ';

  stream << status_
         << name_ << "-Norm = " << NOX::Utils::sciformat(normF_,3)
         << " < " << NOX::Utils::sciformat(trueTolerance_, 3)
         << "\n";

  for (int j = 0; j < indent; j ++)
    stream << ' ';

  stream << setw(13) << " (";

  if (scaleType_ == Scaled)
    stream << "Length-Scaled";
  else
    stream << "Unscaled";

  stream << " ";

  if (normType_ == NOX::Abstract::Vector::TwoNorm)
    stream << "Two-Norm";
  else if (normType_ == NOX::Abstract::Vector::OneNorm)
    stream << "One-Norm";
  else if (normType_ == NOX::Abstract::Vector::MaxNorm)
    stream << "Max-Norm";

  stream << ", ";

  if (toleranceType_ == Absolute)
    stream << "Absolute Tolerance";
  else
    stream << "Relative Tolerance";

  stream << ")\n";

  return stream;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double NOX::FSI::GenericNormF::getNormF() const
{
  return normF_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double NOX::FSI::GenericNormF::getTrueTolerance() const
{
  return trueTolerance_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double NOX::FSI::GenericNormF::getSpecifiedTolerance() const
{
  return specifiedTolerance_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double NOX::FSI::GenericNormF::getInitialTolerance() const
{
  return initialTolerance_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NOX::FSI::PartialNormF::PartialNormF(std::string name,
                                     const LINALG::MultiMapExtractor& extractor,
                                     int blocknum,
                                     double tolerance,
                                     NOX::Abstract::Vector::NormType normType,
                                     ScaleType stype)
  : AdaptiveNewtonNormF(name,tolerance,normType,stype),
    extractor_(extractor),
    blocknum_(blocknum)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double NOX::FSI::PartialNormF::computeNorm(const NOX::Abstract::Group& grp)
{
  if (!grp.isF())
    return -1.0;

  // extract the block epetra vector

  const NOX::Abstract::Vector& abstract_f = grp.getF();
  const NOX::Epetra::Vector& f = Teuchos::dyn_cast<const NOX::Epetra::Vector>(abstract_f);

  // extract the inner vector elements we are interested in

  Teuchos::RCP<Epetra_Vector> v = extractor_.ExtractVector(f.getEpetraVector(),blocknum_);

  double norm = FSI::GenericNormF::computeNorm(*v);

  if (Newton()!=Teuchos::null)
  {
    Newton()->Residual(norm,Tolerance());
  }

  return norm;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NOX::FSI::PartialSumNormF::PartialSumNormF(std::string name,
                                           const LINALG::MapExtractor& extractor1,
                                           double scale1,
                                           const LINALG::MapExtractor& extractor2,
                                           double scale2,
                                           Teuchos::RCP<ADAPTER::CouplingConverter> converter,
                                           double tolerance,
                                           ScaleType stype)
  : AdaptiveNewtonNormF(name,tolerance,NOX::Abstract::Vector::TwoNorm,stype),
    extractor1_(extractor1),
    extractor2_(extractor2),
    scale1_(scale1),
    scale2_(scale2),
    converter_(converter)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double NOX::FSI::PartialSumNormF::computeNorm(const NOX::Abstract::Group& grp)
{
  if (!grp.isF())
    return -1.0;

  // extract the block epetra vector

  const NOX::Abstract::Vector& abstract_f = grp.getF();
  const NOX::Epetra::Vector& f = Teuchos::dyn_cast<const NOX::Epetra::Vector>(abstract_f);

  // extract the inner vector elements we are interested in

  Teuchos::RCP<Epetra_Vector> v1 = extractor1_.ExtractCondVector(f.getEpetraVector());
  Teuchos::RCP<Epetra_Vector> v2 = extractor2_.ExtractCondVector(f.getEpetraVector());

  Teuchos::RCP<Epetra_Vector> v = converter_->SrcToDst(v2);
  v->Update(scale1_,*v1,scale2_);

  double norm = FSI::GenericNormF::computeNorm(*v);

  if (Newton()!=Teuchos::null)
  {
    Newton()->Residual(norm,Tolerance());
  }

  return norm;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NOX::FSI::GenericNormUpdate::GenericNormUpdate(std::string name,
                                               double tol,
                                               NOX::Abstract::Vector::NormType ntype,
                                               ScaleType stype)
  : status_(NOX::StatusTest::Unevaluated),
    normType_(ntype),
    scaleType_(stype),
    tolerance_(tol),
    normUpdate_(0.0),
    name_(name)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NOX::FSI::GenericNormUpdate::GenericNormUpdate(std::string name,
                                               double tol,
                                               ScaleType stype)
  : status_(NOX::StatusTest::Unevaluated),
    normType_(NOX::Abstract::Vector::TwoNorm),
    scaleType_(stype),
    tolerance_(tol),
    normUpdate_(0.0),
    name_(name)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NOX::FSI::GenericNormUpdate::~GenericNormUpdate()
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NOX::StatusTest::StatusType NOX::FSI::GenericNormUpdate::checkStatus(const NOX::Solver::Generic& problem,
                                                                     NOX::StatusTest::CheckType checkType)
{
  if (checkType == NOX::StatusTest::None)
  {
    status_ = NOX::StatusTest::Unevaluated;
    normUpdate_ = -1.0;
    return status_;
  }

  // On the first iteration, the old and current solution are the same so
  // we should return the test as unconverged until there is a valid
  // old solution (i.e. the number of iterations is greater than zero).
  int niters = problem.getNumIterations();
  if (niters == 0)
  {
    status_ = NOX::StatusTest::Unconverged;
    normUpdate_ = -1.0;
    return status_;
  }

  // Check that F exists!
  if (!problem.getSolutionGroup().isF())
  {
    status_ = NOX::StatusTest::Unconverged;
    normUpdate_ = -1.0;
    return status_;
  }

  const NOX::Abstract::Vector& oldSoln = problem.getPreviousSolutionGroup().getX();
  const NOX::Abstract::Vector& curSoln = problem.getSolutionGroup().getX();

  if (Teuchos::is_null(updateVectorPtr_))
    updateVectorPtr_ = curSoln.clone();

  updateVectorPtr_->update(1.0, curSoln, -1.0, oldSoln, 0.0);

  computeNorm(Teuchos::rcp_dynamic_cast<NOX::Epetra::Vector>(updateVectorPtr_)->getEpetraVector());

  status_ = (normUpdate_ < tolerance_) ? NOX::StatusTest::Converged : NOX::StatusTest::Unconverged;
  return status_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double NOX::FSI::GenericNormUpdate::computeNorm(const Epetra_Vector& v)
{
  //NOX::Epetra::Vector vec(Teuchos::rcp(&v,false),NOX::Epetra::Vector::CreateView);
  NOX::Epetra::Vector vec(v);
  int n = (scaleType_ == Scaled) ? vec.length() : 0;

  switch (normType_)
  {
  case NOX::Abstract::Vector::TwoNorm:
    normUpdate_ = vec.norm();
    if (scaleType_ == Scaled)
      normUpdate_ /= sqrt(1.0 * n);
    break;

  default:
    normUpdate_ = vec.norm(normType_);
    if (scaleType_ == Scaled)
      normUpdate_ /= n;
    break;
  }

  return normUpdate_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NOX::StatusTest::StatusType NOX::FSI::GenericNormUpdate::getStatus() const
{
  return status_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::ostream& NOX::FSI::GenericNormUpdate::print(std::ostream& stream, int indent) const
{
  for (int j = 0; j < indent; j ++)
    stream << ' ';
  stream << status_
         << "Absolute Update-Norm = "
         << NOX::Utils::sciformat(normUpdate_, 3)
	 << " < "
         << NOX::Utils::sciformat(tolerance_, 3)
         << endl;
  return stream;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double NOX::FSI::GenericNormUpdate::getNormUpdate() const
{
  return normUpdate_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double NOX::FSI::GenericNormUpdate::getTolerance() const
{
  return tolerance_;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NOX::FSI::PartialNormUpdate::PartialNormUpdate(std::string name,
                                               const LINALG::MultiMapExtractor& extractor,
                                               int blocknum,
                                               double tolerance,
                                               ScaleType stype)
  : GenericNormUpdate(name,tolerance,stype),
    extractor_(extractor),
    blocknum_(blocknum)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double NOX::FSI::PartialNormUpdate::computeNorm(const Epetra_Vector& v)
{
  return FSI::GenericNormUpdate::computeNorm(*extractor_.ExtractVector(v,blocknum_));
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NOX::FSI::MinIters::MinIters(int minIterations, const NOX::Utils* u)
  : miniters(minIterations),
    niters(0),
    status(NOX::StatusTest::Unevaluated)
{
  if (u != NULL)
    utils = *u;

  if (miniters < 1)
  {
    utils.err() << "NOX::StatusTest::MinIters - must choose a number greater than zero" << endl;
    throw "NOX Error";
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NOX::FSI::MinIters::~MinIters()
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NOX::StatusTest::StatusType NOX::FSI::MinIters::
checkStatus(const NOX::Solver::Generic& problem,
	    NOX::StatusTest::CheckType checkType)
{
  switch (checkType)
  {
  case NOX::StatusTest::Complete:
  case NOX::StatusTest::Minimal:
    niters = problem.getNumIterations();
    status = (niters < miniters) ? NOX::StatusTest::Unconverged : NOX::StatusTest::Converged;
    break;

  case NOX::StatusTest::None:
  default:
    niters = -1;
    status = NOX::StatusTest::Unevaluated;
    break;
  }

  return status;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
NOX::StatusTest::StatusType NOX::FSI::MinIters::getStatus() const
{
  return status;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ostream& NOX::FSI::MinIters::print(ostream& stream, int indent) const
{
 return stream;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int NOX::FSI::MinIters::getMinIters() const
{
  return miniters;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int NOX::FSI::MinIters::getNumIters() const
{
  return niters;
}


#endif
