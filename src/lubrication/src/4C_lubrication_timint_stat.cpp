// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_lubrication_timint_stat.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_lubrication_ele_action.hpp"
#include "4C_lubrication_ele_parameter.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                    wirtz 11/15 |
 *----------------------------------------------------------------------*/
Lubrication::TimIntStationary::TimIntStationary(std::shared_ptr<Core::FE::Discretization> actdis,
    std::shared_ptr<Core::LinAlg::Solver> solver, std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Teuchos::ParameterList> extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : TimIntImpl(actdis, solver, params, extraparams, output)
{
  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting
  // this has to be done before all state vectors are initialized
}



/*----------------------------------------------------------------------*
 |  initialize time integration                             wirtz 11/15 |
 *----------------------------------------------------------------------*/
void Lubrication::TimIntStationary::init()
{
  // initialize base class
  TimIntImpl::init();

  // -------------------------------------------------------------------
  // set element parameters
  // -------------------------------------------------------------------
  // note: - this has to be done before element routines are called
  //       - order is important here: for safety checks in set_element_general_parameters(),
  //         we have to know the time-integration parameters
  set_element_time_parameter();
  set_element_general_parameters();
}


/*----------------------------------------------------------------------*
 | set time parameter for element evaluation (usual call)   wirtz 11/15 |
 *----------------------------------------------------------------------*/
void Lubrication::TimIntStationary::set_element_time_parameter() const
{
  Teuchos::ParameterList eleparams;
  eleparams.set<bool>("using generalized-alpha time integration", false);
  eleparams.set<bool>("using stationary formulation", true);
  eleparams.set<double>("time-step length", dta_);
  eleparams.set<double>("total time", time_);
  eleparams.set<double>("time factor", 1.0);
  eleparams.set<double>("alpha_F", 1.0);

  Discret::Elements::LubricationEleParameter::instance(discret_->name())
      ->set_time_parameters(eleparams);
}


/*----------------------------------------------------------------------*
 | set time for evaluation of Neumann boundary conditions   wirtz 11/15 |
 *----------------------------------------------------------------------*/
void Lubrication::TimIntStationary::set_time_for_neumann_evaluation(Teuchos::ParameterList& params)
{
  params.set("total time", time_);
}


/*----------------------------------------------------------------------*
 | add actual Neumann loads                                 wirtz 11/15 |
 *----------------------------------------------------------------------*/
void Lubrication::TimIntStationary::add_neumann_to_residual()
{
  residual_->update(1.0, *neumann_loads_, 1.0);
}


/*--------------------------------------------------------------------------*
 | add global state vectors specific for time-integration scheme            |
 |                                                              wirtz 11/15 |
 *--------------------------------------------------------------------------*/
void Lubrication::TimIntStationary::add_time_integration_specific_vectors(
    bool forcedincrementalsolver)
{
  discret_->set_state("prenp", *prenp_);
}


/*----------------------------------------------------------------------*
 | update of solution at end of time step                   wirtz 11/15 |
 *----------------------------------------------------------------------*/
void Lubrication::TimIntStationary::update(const int num)
{
  // for the stationary scheme there is nothing to do
}


/*----------------------------------------------------------------------*
 |                                                          wirtz 11/15 |
 -----------------------------------------------------------------------*/
void Lubrication::TimIntStationary::read_restart(int step)
{
  Core::IO::DiscretizationReader reader(
      discret_, Global::Problem::instance()->input_control_file(), step);
  time_ = reader.read_double("time");
  step_ = reader.read_int("step");

  if (myrank_ == 0)
    std::cout << "Reading Lubrication restart data (time=" << time_ << " ; step=" << step_ << ")"
              << '\n';

  // read state vectors that are needed for restart
  reader.read_vector(prenp_, "prenp");
}


/*----------------------------------------------------------------------*
 | incremental iteration update of state                    wirtz 01/16 |
 *----------------------------------------------------------------------*/
void Lubrication::TimIntStationary::update_iter_incrementally()
{
  //! new end-point temperatures
  //! T_{n+1}^{<k+1>} := T_{n+1}^{<k>} + IncT_{n+1}^{<k>}
  prenp_->update(1.0, *prei_, 1.0);
}  // update_iter_incrementally()

FOUR_C_NAMESPACE_CLOSE
