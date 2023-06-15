/*-----------------------------------------------------------*/
/*! \file

\brief Check the active set for convergence. Only meaningful for
       inequality constrained problems.



\level 3

*/
/*-----------------------------------------------------------*/

#include "solver_nonlin_nox_statustest_activeset.H"
#include "solver_nonlin_nox_constraint_group.H"

#include <NOX_Solver_Generic.H>
#include <Epetra_Map.h>

#include "utils_exceptions.H"


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::NLN::StatusTest::ActiveSet::ActiveSet(
    const enum NOX::NLN::StatusTest::QuantityType& qtype, const int& max_cycle_size)
    : qtype_(qtype),
      status_(NOX::StatusTest::Unevaluated),
      max_cycle_size_(max_cycle_size),
      cycle_size_(0),
      activesetsize_(0)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::StatusTest::StatusType NOX::NLN::StatusTest::ActiveSet::checkStatus(
    const NOX::Solver::Generic& problem, NOX::StatusTest::CheckType checkType)
{
  // clear the cycling maps at the beginning of a new time step
  if (problem.getNumIterations() == 0) cycling_maps_.clear();

  if (checkType == NOX::StatusTest::None)
  {
    status_ = NOX::StatusTest::Unevaluated;
    activesetsize_ = 0;
    cycle_size_ = 0;
  }
  else
  {
    // get the abstract solution group from the non-linear solver
    const NOX::Abstract::Group& grp = problem.getSolutionGroup();

    // check if the right hand side was already updated
    if (!grp.isF())
      status_ = NOX::StatusTest::Unevaluated;
    else
    {
      // try to cast the nox group
      const NOX::NLN::CONSTRAINT::Group* cnlngrp =
          dynamic_cast<const NOX::NLN::CONSTRAINT::Group*>(&grp);
      if (cnlngrp == NULL) dserror("NOX::NLN::CONSTRAINT::Group cast failed");

      // do the actual active set check
      status_ = cnlngrp->GetActiveSetInfo(qtype_, activesetsize_);
      // check for cycling of the active set
      /* NOTE: This is just working, if you use Epetra_Map s to store your
       * active set informations! */
      if (max_cycle_size_ > 0)
      {
        // get the current active set
        Teuchos::RCP<const Epetra_Map> activeset = cnlngrp->GetCurrentActiveSetMap(qtype_);
        // add a new map a the beginning of the deque
        cycling_maps_.push_front(cnlngrp->GetOldActiveSetMap(qtype_));
        // remove the last entry of the deque, if the max_cycle_size_ is exceeded
        if (cycling_maps_.size() > static_cast<std::size_t>(max_cycle_size_))
          cycling_maps_.pop_back();

        // check for cycling, if the set is not converged
        cycle_size_ = 0;
        if (status_ != NOX::StatusTest::Converged)
        {
          std::deque<Teuchos::RCP<const Epetra_Map>>::const_iterator citer;
          int count = 1;
          // reset the detected cycle size
          for (citer = cycling_maps_.begin(); citer != cycling_maps_.end(); ++citer)
          {
            if ((activeset->NumGlobalElements() != 0) and (*citer)->SameAs(*activeset))
              cycle_size_ = count;
            ++count;
          }
        }
      }  // if (max_cycle_size_>0)
    }
  }

  return status_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::StatusTest::StatusType NOX::NLN::StatusTest::ActiveSet::getStatus() const { return status_; }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::ostream& NOX::NLN::StatusTest::ActiveSet::print(std::ostream& stream, int indent) const
{
  std::string indent_string;
  indent_string.assign(indent, ' ');

  stream << indent_string;
  stream << status_;
  stream << QuantityType2String(qtype_) << "-";
  stream << "Active-Set-Size = " << activesetsize_;
  stream << std::endl;
  // optional output
  if (cycle_size_ > 0)
    stream << indent_string << std::setw(13) << " "
           << "WARNING: "
              "The active set cycles between iteration (k) and (k-"
           << cycle_size_ << ")!" << std::endl;

  return stream;
}
