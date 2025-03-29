// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_timint_stat.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_timint_meshtying_strategy_base.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor (public)                                      gjb 08/08 |
 *----------------------------------------------------------------------*/
ScaTra::TimIntStationary::TimIntStationary(std::shared_ptr<Core::FE::Discretization> actdis,
    std::shared_ptr<Core::LinAlg::Solver> solver, std::shared_ptr<Teuchos::ParameterList> params,
    std::shared_ptr<Teuchos::ParameterList> extraparams,
    std::shared_ptr<Core::IO::DiscretizationWriter> output)
    : ScaTraTimIntImpl(actdis, solver, params, extraparams, output), fsphinp_(nullptr)
{
  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting
  // this has to be done before all state vectors are initialized
}


/*----------------------------------------------------------------------*
 |  initialize time integration                         rasthofer 09/13 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntStationary::init()
{
  // initialize base class
  ScaTraTimIntImpl::init();

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  // -------------------------------------------------------------------
  const Core::LinAlg::Map* dofrowmap = discret_->dof_row_map();

  // fine-scale vector
  if (fssgd_ != Inpar::ScaTra::fssugrdiff_no)
    fsphinp_ = Core::LinAlg::create_vector(*dofrowmap, true);
  if (turbmodel_ != Inpar::FLUID::no_model) FOUR_C_THROW("Turbulence is not stationary problem!");

  // -------------------------------------------------------------------
  // set element parameters
  // -------------------------------------------------------------------
  // note: - this has to be done before element routines are called
  //       - order is important here: for safety checks in set_element_general_parameters(),
  //         we have to know the time-integration parameters
  set_element_time_parameter();
  set_element_general_parameters();
  set_element_turbulence_parameters();

  // setup krylov
  prepare_krylov_projection();

  // safety check
  if (params_->get<bool>("NATURAL_CONVECTION"))
    FOUR_C_THROW("Natural convection for stationary time integration scheme is not implemented!");
}



/*----------------------------------------------------------------------*
 | set time parameter for element evaluation (usual call)    ehrl 11/13 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntStationary::set_element_time_parameter(bool forcedincrementalsolver) const
{
  Teuchos::ParameterList eleparams;

  eleparams.set<bool>("using generalized-alpha time integration", false);
  eleparams.set<bool>("using stationary formulation", true);
  if (!forcedincrementalsolver)
    eleparams.set<bool>("incremental solver", incremental_);
  else
    eleparams.set<bool>("incremental solver", true);

  eleparams.set<double>("time-step length", dta_);
  eleparams.set<double>("total time", time_);
  eleparams.set<double>("time factor", 1.0);
  eleparams.set<double>("alpha_F", 1.0);

  Discret::Elements::ScaTraEleParameterTimInt::instance(discret_->name())
      ->set_parameters(eleparams);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntStationary::setup()
{
  // setup base class
  ScaTraTimIntImpl::setup();

  set_element_nodeset_parameters();
}

/*----------------------------------------------------------------------*
 | set time for evaluation of Neumann boundary conditions      vg 12/08 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntStationary::set_time_for_neumann_evaluation(Teuchos::ParameterList& params)
{
  params.set("total time", time_);
}


/*----------------------------------------------------------------------*
 | set part of the residual vector belonging to the old timestep        |
 |                                                            gjb 08/08 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntStationary::set_old_part_of_righthandside()
{
  // call base class routine
  ScaTraTimIntImpl::set_old_part_of_righthandside();

  hist_->put_scalar(0.0);
}


/*----------------------------------------------------------------------*
 | add actual Neumann loads                                    vg 11/08 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntStationary::add_neumann_to_residual()
{
  residual_->update(1.0, *neumann_loads_, 1.0);
}


/*----------------------------------------------------------------------*
 | AVM3-based scale separation                                 vg 03/09 |
 *----------------------------------------------------------------------*/
void ScaTra::TimIntStationary::avm3_separation()
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:            + avm3");

  // AVM3 separation
  Sep_->multiply(false, *phinp_, *fsphinp_);

  // set fine-scale vector
  discret_->set_state("fsphinp", *fsphinp_);
}


/*--------------------------------------------------------------------------*
 | add global state vectors specific for time-integration scheme   vg 11/08 |
 *--------------------------------------------------------------------------*/
void ScaTra::TimIntStationary::add_time_integration_specific_vectors(bool forcedincrementalsolver)
{
  // call base class routine
  ScaTraTimIntImpl::add_time_integration_specific_vectors(forcedincrementalsolver);

  discret_->set_state("hist", *hist_);
  discret_->set_state("phinp", *phinp_);
}


/*----------------------------------------------------------------------*
 |                                                            gjb 09/08 |
 -----------------------------------------------------------------------*/
void ScaTra::TimIntStationary::read_restart(
    const int step, std::shared_ptr<Core::IO::InputControl> input)
{
  // call base class routine
  ScaTraTimIntImpl::read_restart(step, input);

  std::shared_ptr<Core::IO::DiscretizationReader> reader(nullptr);
  if (input == nullptr)
  {
    reader = std::make_shared<Core::IO::DiscretizationReader>(
        discret_, Global::Problem::instance()->input_control_file(), step);
  }
  else
    reader = std::make_shared<Core::IO::DiscretizationReader>(discret_, input, step);

  time_ = reader->read_double("time");
  step_ = reader->read_int("step");

  if (myrank_ == 0)
    std::cout << "Reading ScaTra restart data (time=" << time_ << " ; step=" << step_ << ")"
              << '\n';

  // read state vectors that are needed for restart
  reader->read_vector(phinp_, "phinp");

  read_restart_problem_specific(step, *reader);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntStationary::update()
{
  // call base class routine
  ScaTraTimIntImpl::update();

  // compute flux vector field for later output BEFORE time shift of results
  // is performed below !!
  if (calcflux_domain_ != Inpar::ScaTra::flux_none or
      calcflux_boundary_ != Inpar::ScaTra::flux_none)
  {
    if (is_result_step() or do_boundary_flux_statistics()) calc_flux(true);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::TimIntStationary::write_restart() const
{
  // call base class routine
  ScaTraTimIntImpl::write_restart();

  // This feature enables starting a time-dependent simulation from
  // a non-trivial steady-state solution that was calculated before.
  output_->write_vector("phin", phinp_);    // for OST and BDF2
  output_->write_vector("phinm", phinp_);   // for BDF2
  output_->write_vector("phidtn", zeros_);  // for OST
}

FOUR_C_NAMESPACE_CLOSE
