/*-----------------------------------------------------------*/
/*! \file



\level 3

*/
/*-----------------------------------------------------------*/

#include "baci_solver_nonlin_nox_linearsystem_prepostoperator.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::NLN::LinSystem::PrePostOperator::PrePostOperator() : havePrePostOperator_(false)
{
  // Disallowed constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::NLN::LinSystem::PrePostOperator::PrePostOperator(const PrePostOperator& ppo)
    : havePrePostOperator_(false)
{
  // Disallowed copy constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::NLN::LinSystem::PrePostOperator& NOX::NLN::LinSystem::PrePostOperator::operator=(
    const PrePostOperator& ppo)
{
  // disallowed assignment operator
  return *this;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::NLN::LinSystem::PrePostOperator::PrePostOperator(Teuchos::ParameterList& linearSolverSubList)
    : havePrePostOperator_(false)
{
  reset(linearSolverSubList);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::LinSystem::PrePostOperator::reset(Teuchos::ParameterList& linearSolverSubList)
{
  havePrePostOperator_ = false;

  /* Check if a pre/post processor for the linear system is provided
   * by the user. */
  if (linearSolverSubList.INVALID_TEMPLATE_QUALIFIER isType<Teuchos::RCP<Map>>(
          "User Defined Pre/Post Operator"))
  {
    prePostOperatorMapPtr_ = linearSolverSubList.INVALID_TEMPLATE_QUALIFIER get<Teuchos::RCP<Map>>(
        "User Defined Pre/Post Operator");
    havePrePostOperator_ = true;
  }
}

// non-member function
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::NLN::LinSystem::PrePostOperator::Map& NOX::NLN::LinSystem::PrePostOp::GetMap(
    Teuchos::ParameterList& p_linsolver)
{
  Teuchos::RCP<NOX::NLN::LinSystem::PrePostOperator::Map>& mapptr =
      p_linsolver.get<Teuchos::RCP<NOX::NLN::LinSystem::PrePostOperator::Map>>(
          "User Defined Pre/Post Operator",
          Teuchos::rcp(new NOX::NLN::LinSystem::PrePostOperator::Map()));

  return *mapptr;
}

FOUR_C_NAMESPACE_CLOSE
