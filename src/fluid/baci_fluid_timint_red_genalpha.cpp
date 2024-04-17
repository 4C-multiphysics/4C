/*-----------------------------------------------------------*/
/*! \file

\brief Generalized-alpha time integration for reduced models


\level 2

*/
/*-----------------------------------------------------------*/

#include "baci_fluid_timint_red_genalpha.hpp"

#include "baci_io.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  Constructor (public)                                       bk 11/13 |
 *----------------------------------------------------------------------*/
FLD::TimIntRedModelsGenAlpha::TimIntRedModelsGenAlpha(
    const Teuchos::RCP<DRT::Discretization>& actdis,
    const Teuchos::RCP<CORE::LINALG::Solver>& solver,
    const Teuchos::RCP<Teuchos::ParameterList>& params,
    const Teuchos::RCP<IO::DiscretizationWriter>& output, bool alefluid /*= false*/)
    : FluidImplicitTimeInt(actdis, solver, params, output, alefluid),
      TimIntGenAlpha(actdis, solver, params, output, alefluid),
      TimIntRedModels(actdis, solver, params, output, alefluid)
{
  return;
}


/*----------------------------------------------------------------------*
 |  initialize algorithm                                rasthofer 04/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntRedModelsGenAlpha::Init()
{
  // call Init()-functions of base classes
  // note: this order is important
  TimIntGenAlpha::Init();
  TimIntRedModels::Init();

  return;
}


/*----------------------------------------------------------------------*
 |  read restart data                                   rasthofer 12/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntRedModelsGenAlpha::ReadRestart(int step)
{
  // call of base classes
  TimIntGenAlpha::ReadRestart(step);
  TimIntRedModels::ReadRestart(step);

  return;
}

FOUR_C_NAMESPACE_CLOSE
