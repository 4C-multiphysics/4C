// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_timint_stat_hdg.hpp"

#include "4C_io.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                               berardocco 05/20 |
 *----------------------------------------------------------------------*/
ScaTra::TimIntStationaryHDG::TimIntStationaryHDG(std::shared_ptr<Core::FE::Discretization> actdis,
    std::shared_ptr<Core::LinAlg::Solver> solver, std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Teuchos::ParameterList> extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
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
void ScaTra::TimIntStationaryHDG::init()
{
  // initialize base class
  TimIntHDG::init();

  return;
}


/*----------------------------------------------------------------------*
 | set time parameter for element evaluation           berardocco 05/20 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntStationaryHDG::set_element_time_parameter(bool forcedincrementalsolver) const
{
  Teuchos::ParameterList eleparams;

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

  Discret::Elements::ScaTraEleParameterTimInt::instance(discret_->name())
      ->set_parameters(eleparams);
}


/*----------------------------------------------------------------------*
 | set time for evaluation of Neumann boundary conditions               |
 |                                                     berardocco 05/20 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntStationaryHDG::set_time_for_neumann_evaluation(Teuchos::ParameterList& params)
{
  params.set("total time", time_);
  return;
}
FOUR_C_NAMESPACE_CLOSE
