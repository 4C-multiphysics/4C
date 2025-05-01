// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solver_nonlin_nox_linearsystem_prepostoperator.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::LinSystem::PrePostOperator::PrePostOperator() : havePrePostOperator_(false)
{
  // Disallowed constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::LinSystem::PrePostOperator::PrePostOperator(const PrePostOperator& ppo)
    : havePrePostOperator_(false)
{
  // Disallowed copy constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::LinSystem::PrePostOperator& NOX::Nln::LinSystem::PrePostOperator::operator=(
    const PrePostOperator& ppo)
{
  // disallowed assignment operator
  return *this;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::LinSystem::PrePostOperator::PrePostOperator(Teuchos::ParameterList& linearSolverSubList)
    : havePrePostOperator_(false)
{
  reset(linearSolverSubList);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::LinSystem::PrePostOperator::reset(Teuchos::ParameterList& linearSolverSubList)
{
  havePrePostOperator_ = false;

  /* Check if a pre/post processor for the linear system is provided
   * by the user. */
  if (linearSolverSubList.isType<Teuchos::RCP<Map>>("User Defined Pre/Post Operator"))
  {
    prePostOperatorMapPtr_ =
        linearSolverSubList.get<Teuchos::RCP<Map>>("User Defined Pre/Post Operator");
    havePrePostOperator_ = true;
  }
}

// non-member function
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::LinSystem::PrePostOperator::Map& NOX::Nln::LinSystem::PrePostOp::get_map(
    Teuchos::ParameterList& p_linsolver)
{
  Teuchos::RCP<NOX::Nln::LinSystem::PrePostOperator::Map>& mapptr =
      p_linsolver.get<Teuchos::RCP<NOX::Nln::LinSystem::PrePostOperator::Map>>(
          "User Defined Pre/Post Operator",
          Teuchos::make_rcp<NOX::Nln::LinSystem::PrePostOperator::Map>());

  return *mapptr;
}

FOUR_C_NAMESPACE_CLOSE
