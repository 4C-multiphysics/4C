/*-----------------------------------------------------------*/
/*!
\file str_nln_solver_generic.cpp

\maintainer Michael Hiermeier

\date Oct 9, 2015

\level 3

*/
/*-----------------------------------------------------------*/

#include "str_nln_solver_generic.H"
#include "str_timint_implicit.H"
#include "str_timint_base.H"
#include "str_timint_noxinterface.H"

#include <NOX_Abstract_Group.H>


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::NLN::SOLVER::Generic::Generic()
    : isinit_(false),
      issetup_(false),
      gstate_ptr_(Teuchos::null),
      sdyn_ptr_(Teuchos::null),
      noxinterface_ptr_(Teuchos::null),
      group_ptr_(Teuchos::null)
{
  // empty constructor
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::NLN::SOLVER::Generic::Init(
    const Teuchos::RCP<STR::TIMINT::BaseDataGlobalState> gstate,
    const Teuchos::RCP<STR::TIMINT::BaseDataSDyn> sdyn,
    const Teuchos::RCP<STR::TIMINT::NoxInterface> noxinterface)
{
  // We have to call Setup() after Init()
  issetup_ = false;

  // initialize internal variables
  gstate_ptr_ = gstate;
  sdyn_ptr_ = sdyn;
  noxinterface_ptr_ = noxinterface;

  isinit_ = true;

  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<NOX::Abstract::Group>& STR::NLN::SOLVER::Generic::GroupPtr()
{
  CheckInit();

  return group_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Abstract::Group& STR::NLN::SOLVER::Generic::Group()
{
  CheckInit();
  if (group_ptr_.is_null())
    dserror("The group pointer should be initialized beforehand!");
  return *group_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Abstract::Group& STR::NLN::SOLVER::Generic::SolutionGroup()
{
  CheckInitSetup();
  if (group_ptr_.is_null())
    dserror("The group pointer should be initialized beforehand!");

  return *group_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const NOX::Abstract::Group& STR::NLN::SOLVER::Generic::GetSolutionGroup() const
{
  CheckInitSetup();
  if (group_ptr_.is_null())
    dserror("The group pointer should be initialized beforehand!");

  return *group_ptr_;
}
