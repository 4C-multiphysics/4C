// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_statustest_activeset.hpp"

#include "4C_linalg_map.hpp"
#include "4C_solver_nonlin_nox_constraint_group.hpp"
#include "4C_utils_exceptions.hpp"

#include <NOX_Solver_Generic.H>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::StatusTest::ActiveSet::ActiveSet(
    const enum NOX::Nln::StatusTest::QuantityType& qtype, const int& max_cycle_size)
    : qtype_(qtype),
      status_(::NOX::StatusTest::Unevaluated),
      max_cycle_size_(max_cycle_size),
      cycle_size_(0),
      activesetsize_(0)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
::NOX::StatusTest::StatusType NOX::Nln::StatusTest::ActiveSet::checkStatus(
    const ::NOX::Solver::Generic& problem, ::NOX::StatusTest::CheckType checkType)
{
  // clear the cycling maps at the beginning of a new time step
  if (problem.getNumIterations() == 0) cycling_maps_.clear();

  if (checkType == ::NOX::StatusTest::None)
  {
    status_ = ::NOX::StatusTest::Unevaluated;
    activesetsize_ = 0;
    cycle_size_ = 0;
  }
  else
  {
    // get the abstract solution group from the non-linear solver
    const ::NOX::Abstract::Group& grp = problem.getSolutionGroup();

    // check if the right hand side was already updated
    if (!grp.isF())
      status_ = ::NOX::StatusTest::Unevaluated;
    else
    {
      // try to cast the nox group
      const NOX::Nln::CONSTRAINT::Group* cnlngrp =
          dynamic_cast<const NOX::Nln::CONSTRAINT::Group*>(&grp);
      if (cnlngrp == nullptr) FOUR_C_THROW("NOX::Nln::CONSTRAINT::Group cast failed");

      // do the actual active set check
      status_ = cnlngrp->get_active_set_info(qtype_, activesetsize_);
      // check for cycling of the active set
      /* NOTE: This is just working, if you use Core::LinAlg::Map s to store your
       * active set information! */
      if (max_cycle_size_ > 0)
      {
        // get the current active set
        Teuchos::RCP<const Core::LinAlg::Map> activeset =
            cnlngrp->get_current_active_set_map(qtype_);
        // add a new map a the beginning of the deque
        cycling_maps_.push_front(cnlngrp->get_old_active_set_map(qtype_));
        // remove the last entry of the deque, if the max_cycle_size_ is exceeded
        if (cycling_maps_.size() > static_cast<std::size_t>(max_cycle_size_))
          cycling_maps_.pop_back();

        // check for cycling, if the set is not converged
        cycle_size_ = 0;
        if (status_ != ::NOX::StatusTest::Converged)
        {
          int count = 1;
          // reset the detected cycle size
          for (auto citer = cycling_maps_.begin(); citer != cycling_maps_.end(); ++citer)
          {
            if ((activeset->num_global_elements() != 0) and (*citer)->same_as(*activeset))
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
::NOX::StatusTest::StatusType NOX::Nln::StatusTest::ActiveSet::getStatus() const { return status_; }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::ostream& NOX::Nln::StatusTest::ActiveSet::print(std::ostream& stream, int indent) const
{
  std::string indent_string;
  indent_string.assign(indent, ' ');

  stream << indent_string;
  stream << status_;
  stream << quantity_type_to_string(qtype_) << "-";
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

FOUR_C_NAMESPACE_CLOSE
