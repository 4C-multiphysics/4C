// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_solver_singlestep.hpp"  // class definition

#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_inner_statustest_generic.hpp"

// templated status tests
#include "4C_solver_nonlin_nox_statustest_activeset.hpp"
#include "4C_solver_nonlin_nox_statustest_normf.hpp"
#include "4C_solver_nonlin_nox_statustest_normupdate.hpp"
#include "4C_solver_nonlin_nox_statustest_normwrms.hpp"

#include <NOX_Abstract_Group.H>
#include <NOX_Solver_SolverUtils.H>
#include <NOX_Utils.H>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::Solver::SingleStep::SingleStep(const Teuchos::RCP<::NOX::Abstract::Group>& grp,
    const Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic>& innerTests,
    const Teuchos::RCP<Teuchos::ParameterList>& params)
    : ::NOX::Solver::SingleStep(grp, params)
{
  // call own init() after base init() was called.
  init(*innerTests);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Solver::SingleStep::init(NOX::Nln::Inner::StatusTest::Generic& innerTests)
{
  nIter = 0;
  status = ::NOX::StatusTest::Unconverged;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
::NOX::StatusTest::StatusType NOX::Nln::Solver::SingleStep::getStatus() const { return status; }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const ::NOX::Utils& NOX::Nln::Solver::SingleStep::get_utils() const { return *utilsPtr; }


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::Solver::SingleStep::printUpdate()
{
  if (utilsPtr->isPrintType(::NOX::Utils::OuterIteration))
  {
    utilsPtr->out() << "\n" << ::NOX::Utils::fill(72) << "\n";
    utilsPtr->out() << "-- The \"Explicit\" Solver Step -- \n";
    if (printNorms)
    {
      if (!solnPtr->isF()) solnPtr->computeF();
      const double normF = solnPtr->getF().norm();
      const double normDx = solnPtr->getNewtonPtr()->norm();
      utilsPtr->out() << "||F||=" << normF << ", ||dx||=" << normDx;
      if (computeRelativeNorm)
      {
        utilsPtr->out() << ", ||F|| / ||F_0||=" << normF / normF_0;
      }
    }
    if (status != ::NOX::StatusTest::Converged)
    {
      FOUR_C_THROW(
          "The SingleStep solver does not converge. This can't happen unless NaN values are "
          "encountered in the computation.");
    }
    utilsPtr->out() << "\n" << ::NOX::Utils::fill(72) << "\n" << std::endl;
  }
}

FOUR_C_NAMESPACE_CLOSE
