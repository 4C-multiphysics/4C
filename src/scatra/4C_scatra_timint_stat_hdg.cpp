/*----------------------------------------------------------------------*/
/*! \file
\brief solution algorithm for stationary problems

\level 1
*/
/*----------------------------------------------------------------------*/
#include "4C_scatra_timint_stat_hdg.hpp"

#include "4C_io.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                               berardocco 05/20 |
 *----------------------------------------------------------------------*/
SCATRA::TimIntStationaryHDG::TimIntStationaryHDG(Teuchos::RCP<DRT::Discretization> actdis,
    Teuchos::RCP<CORE::LINALG::Solver> solver, Teuchos::RCP<Teuchos::ParameterList> params,
    Teuchos::RCP<Teuchos::ParameterList> extraparams, Teuchos::RCP<IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(actdis, solver, params, extraparams, output),
      TimIntHDG(actdis, solver, params, extraparams, output)
{
  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting
  // this has to be done before all state vectors are initialized
  return;
}


/*----------------------------------------------------------------------*
 |  initialize time integration                        berardocco 05/20 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntStationaryHDG::Init()
{
  // initialize base class
  TimIntHDG::Init();

  return;
}


/*----------------------------------------------------------------------*
 | set time parameter for element evaluation           berardocco 05/20 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntStationaryHDG::SetElementTimeParameter(bool forcedincrementalsolver) const
{
  Teuchos::ParameterList eleparams;

  CORE::UTILS::AddEnumClassToParameterList<SCATRA::Action>(
      "action", SCATRA::Action::set_time_parameter, eleparams);
  eleparams.set<bool>("using generalized-alpha time integration", false);
  eleparams.set<bool>("using stationary formulation", true);
  if (forcedincrementalsolver == false)
    eleparams.set<bool>("incremental solver", incremental_);
  else
    eleparams.set<bool>("incremental solver", true);

  // Force time step to be one for simplicity
  eleparams.set<double>("time-step length", dta_);
  eleparams.set<double>("total time", time_);
  eleparams.set<double>("time factor", 1.0);
  eleparams.set<double>("alpha_F", 1.0);

  // call standard loop over elements
  discret_->Evaluate(
      eleparams, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null, Teuchos::null);

  return;
}


/*----------------------------------------------------------------------*
 | set time for evaluation of Neumann boundary conditions               |
 |                                                     berardocco 05/20 |
 *----------------------------------------------------------------------*/
void SCATRA::TimIntStationaryHDG::SetTimeForNeumannEvaluation(Teuchos::ParameterList& params)
{
  params.set("total time", time_);
  return;
}
FOUR_C_NAMESPACE_CLOSE
