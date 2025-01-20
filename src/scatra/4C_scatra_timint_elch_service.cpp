// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_timint_elch_service.hpp"

#include "4C_io.hpp"
#include "4C_io_input_parameter_container.hpp"

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
ScaTra::CCCVCondition::CCCVCondition(const Core::Conditions::Condition& cccvcyclingcondition,
    const std::vector<Core::Conditions::Condition*>& cccvhalfcycleconditions,
    const bool adaptivetimestepping, const int num_dofs)
    : adaptivetimesteppingonoff_(static_cast<bool>(
          cccvcyclingcondition.parameters().get<int>("ADAPTIVE_TIME_STEPPING_INIT_RELAX"))),
      beginwithcharge_(
          static_cast<bool>(cccvcyclingcondition.parameters().get<int>("BEGIN_WITH_CHARGING"))),
      charging_(false),
      ihalfcycle_(-1),
      initrelaxtime_(cccvcyclingcondition.parameters().get<double>("INIT_RELAX_TIME")),
      min_time_steps_during_init_relax_(cccvcyclingcondition.parameters()
              .get<Core::IO::Noneable<int>>("MIN_TIME_STEPS_DURING_INIT_RELAX")
              .value_or(-1)),
      nhalfcycles_(cccvcyclingcondition.parameters().get<int>("NUMBER_OF_HALF_CYCLES")),
      num_add_adapt_timesteps_(cccvcyclingcondition.parameters()
              .get<Core::IO::Noneable<int>>("NUM_ADD_ADAPT_TIME_STEPS")
              .value_or(-1)),
      num_dofs_(num_dofs),
      phasechanged_(false),
      phaseinitialrelaxation_(false),
      steplastphasechange_(-1)
{
  // safety checks
  if (adaptivetimesteppingonoff_ and !adaptivetimestepping)
  {
    FOUR_C_THROW(
        "Must not activate adaptive time stepping for initial relaxation while adaptive time "
        "stepping in scatra is disabled.");
  }
  if (nhalfcycles_ < 1)
  {
    FOUR_C_THROW(
        "Less than one constant-current constant-voltage (CCCV) half-cycle specified in CCCV "
        "cell cycling boundary condition!");
  }

  // loop over all conditions and create half cycles
  for (const auto& condition : cccvhalfcycleconditions)
  {
    if (condition->parameters().get<int>("ConditionID") < 0)
    {
      FOUR_C_THROW(
          "Constant-current constant-voltage (CCCV) half-cycle boundary condition has invalid "
          "condition ID!");
    }

    if (condition->parameters().get<int>("ConditionID") ==
        cccvcyclingcondition.parameters()
            .get<Core::IO::Noneable<int>>("CONDITION_ID_FOR_CHARGE")
            .value_or(-1))
      halfcycle_charge_ =
          std::make_shared<ScaTra::CCCVHalfCycleCondition>(*condition, adaptivetimestepping);
    if (condition->parameters().get<int>("ConditionID") ==
        cccvcyclingcondition.parameters()
            .get<Core::IO::Noneable<int>>("CONDITION_ID_FOR_DISCHARGE")
            .value_or(-1))
      halfcycle_discharge_ =
          std::make_shared<ScaTra::CCCVHalfCycleCondition>(*condition, adaptivetimestepping);
  }
  if (halfcycle_charge_ == nullptr) FOUR_C_THROW("Invalid halfcycle for charge!");
  if (halfcycle_discharge_ == nullptr) FOUR_C_THROW("Invalid halfcycle for discharge!");

  // activate first half cycle depending on initial relaxation
  if (initrelaxtime_ < 0.0)
    set_first_cccv_half_cycle(1);
  else if (initrelaxtime_ > 0.0)
    phaseinitialrelaxation_ = true;
  else
    FOUR_C_THROW("Please choose an initial relaxation time larger than 0.0");
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::CCCVCondition::set_first_cccv_half_cycle(const int step)
{
  // only the current simulation is not restarted
  if (ihalfcycle_ == -1)
  {
    // if initial relaxation was before
    if (phaseinitialrelaxation_) set_phase_change_observer(step);

    // determine whether simulation starts with charge or discharge half-cycle
    charging_ = beginwithcharge_;

    // set number of first charge or discharge half-cycle
    ihalfcycle_ = 1;

    // set phase of half cycles to constant current
    halfcycle_charge_->reset_phase();
    halfcycle_discharge_->reset_phase();

    // initial relaxation is over (or has never been started)
    phaseinitialrelaxation_ = false;
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
Inpar::ElCh::CCCVHalfCyclePhase ScaTra::CCCVCondition::get_cccv_half_cycle_phase() const
{
  // find current phase. If not initial relaxation look up in active half cycle
  return (phaseinitialrelaxation_
              ? Inpar::ElCh::CCCVHalfCyclePhase::initital_relaxation
              : (charging_ ? halfcycle_charge_->get_cccv_half_cycle_phase()
                           : halfcycle_discharge_->get_cccv_half_cycle_phase()));
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
bool ScaTra::CCCVCondition::is_end_of_half_cycle_phase(
    const double cellvoltage, const double cellcrate, const double time) const
{
  bool phasefinished = false;

  // is the condition fulfilled to finish current phase?
  switch (get_cccv_half_cycle_phase())
  {
    case Inpar::ElCh::CCCVHalfCyclePhase::constant_current:
    {
      phasefinished = exceed_cell_voltage(cellvoltage);
      break;
    }
    case Inpar::ElCh::CCCVHalfCyclePhase::constant_voltage:
    {
      phasefinished = exceed_cell_c_rate(cellcrate);
      break;
    }
    case Inpar::ElCh::CCCVHalfCyclePhase::relaxation:
    {
      phasefinished = (time >= get_relax_end_time() - 1e-14);
      break;
    }
    default:
      FOUR_C_THROW("illegal CC-CV operation mode");
      break;
  }

  return phasefinished;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::CCCVCondition::next_phase(const int step, const double time, const bool print)
{
  // store, that phase is changed now.
  set_phase_change_observer(step);

  // print cycling information to improve readability of the output
  if (print)
  {
    std::cout << "\n CCCV cycling: ";
    if (charging_)
      std::cout << "Charging half-cycle: ";
    else
      std::cout << "Discharging half-cycle: ";
  }

  // update phase and check if this half cycle is over?
  if (charging_ ? halfcycle_charge_->is_end_of_half_cycle_next_phase(time, print)
                : halfcycle_discharge_->is_end_of_half_cycle_next_phase(time, print))
  {
    // if half cylce is over reset all phases to constant current, flip charge mode and increase
    // counter
    charging_ ? halfcycle_discharge_->reset_phase() : halfcycle_charge_->reset_phase();
    charging_ = !charging_;
    ihalfcycle_++;
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
bool ScaTra::CCCVCondition::exceed_cell_voltage(const double expected_cellvoltage) const
{
  return (
      (charging_ and expected_cellvoltage > halfcycle_charge_->get_cut_off_voltage() - 1e-14) or
      (!charging_ and expected_cellvoltage < halfcycle_discharge_->get_cut_off_voltage() + 1e-14));
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
bool ScaTra::CCCVCondition::exceed_cell_c_rate(const double expected_cellcrate) const
{
  return (
      (charging_ and expected_cellcrate < (halfcycle_charge_->get_cut_off_c_rate() + 1e-14)) or
      (!charging_ and expected_cellcrate < (halfcycle_discharge_->get_cut_off_c_rate() + 1e-14)));
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
int ScaTra::CCCVCondition::get_half_cycle_condition_id() const
{
  return charging_ ? halfcycle_charge_->get_condition_id()
                   : halfcycle_discharge_->get_condition_id();
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
bool ScaTra::CCCVCondition::exceed_max_steps_from_last_phase_change(const int step)
{
  const bool out = (phasechanged_ && (step >= (steplastphasechange_ + num_add_adapt_timesteps_)));
  if (out) phasechanged_ = false;

  return out;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
double ScaTra::CCCVCondition::get_relax_end_time() const
{
  return charging_ ? halfcycle_charge_->get_relax_end_time()
                   : halfcycle_discharge_->get_relax_end_time();
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
bool ScaTra::CCCVCondition::is_adaptive_time_stepping_phase() const
{
  return (phaseinitialrelaxation_
              ? static_cast<bool>(adaptivetimesteppingonoff_)
              : (charging_ ? halfcycle_charge_->is_adaptive_time_stepping_phase()
                           : halfcycle_discharge_->is_adaptive_time_stepping_phase()));
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::CCCVCondition::set_phase_change_observer(const int step)
{
  // store, that phase is changed now
  phasechanged_ = true;
  steplastphasechange_ = step;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::CCCVCondition::reset_phase_change_observer() { phasechanged_ = false; }

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::CCCVCondition::read_restart(Core::IO::DiscretizationReader& reader)
{
  // extract number of current charge or discharge half-cycle
  ihalfcycle_ = reader.read_int("ihalfcycle");

  // check whether number of current charge or discharge half-cycle is even, i.e., whether
  // current half-cycle is opposite to first one w.r.t. charge/discharge
  charging_ = ihalfcycle_ % 2 == 0 ? !beginwithcharge_ : beginwithcharge_;

  // phase was changed since last time step adaptivity
  phasechanged_ = static_cast<bool>(reader.read_int("phasechanged"));

  // we are in initial relaxation
  phaseinitialrelaxation_ = static_cast<bool>(reader.read_int("phaseinitialrelaxation"));

  // phase was lastly change at that step
  steplastphasechange_ = reader.read_int("steplastphasechange");

  // read restart in half cycles
  halfcycle_charge_->read_restart(reader);
  halfcycle_discharge_->read_restart(reader);
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
ScaTra::CCCVHalfCycleCondition::CCCVHalfCycleCondition(
    const Core::Conditions::Condition& cccvhalfcyclecondition, const bool adaptivetimestepping)
    : adaptivetimesteppingonoff_(cccvhalfcyclecondition.parameters().get<std::vector<int>>(
          "ADAPTIVE_TIME_STEPPING_PHASE_ON_OFF")),
      cutoffcrate_(cccvhalfcyclecondition.parameters().get<double>("CUT_OFF_C_RATE")),
      cutoffvoltage_(cccvhalfcyclecondition.parameters().get<double>("CUT_OFF_VOLTAGE")),
      halfcyclecondition_id_(cccvhalfcyclecondition.parameters().get<int>("ConditionID")),
      phase_cccv_(Inpar::ElCh::CCCVHalfCyclePhase::undefined),
      relaxendtime_(-1.0),
      relaxtime_(cccvhalfcyclecondition.parameters().get<double>("RELAX_TIME"))
{
  // safety check
  for (int i : adaptivetimesteppingonoff_)
  {
    if (i and !adaptivetimestepping)
    {
      FOUR_C_THROW(
          "Must not activate adaptive time stepping for half cycles while adaptive time "
          "stepping in scatra is disabled.");
    }
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
bool ScaTra::CCCVHalfCycleCondition::is_end_of_half_cycle_next_phase(
    const double time, const bool print)
{
  bool halfcycle_finished = false;

  // update phase and check if half cycle is over
  switch (phase_cccv_)
  {
    case Inpar::ElCh::CCCVHalfCyclePhase::constant_current:
    {
      phase_cccv_ = Inpar::ElCh::CCCVHalfCyclePhase::constant_voltage;
      if (print) std::cout << "CC-phase finished! \n";
      break;
    }
    case Inpar::ElCh::CCCVHalfCyclePhase::constant_voltage:
    {
      phase_cccv_ = Inpar::ElCh::CCCVHalfCyclePhase::relaxation;
      relaxendtime_ = time + relaxtime_;
      if (print) std::cout << "CV-phase finished! \n";
      break;
    }
    case Inpar::ElCh::CCCVHalfCyclePhase::relaxation:
    {
      halfcycle_finished = true;
      if (print) std::cout << "Relaxation-phase finished! \n";
      break;
    }
    default:
      FOUR_C_THROW("illegal CC-CV mode");
      break;
  }

  return halfcycle_finished;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::CCCVHalfCycleCondition::reset_phase()
{
  phase_cccv_ = Inpar::ElCh::CCCVHalfCyclePhase::constant_current;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
bool ScaTra::CCCVHalfCycleCondition::is_adaptive_time_stepping_phase() const
{
  bool adaptivetimestepping = false;

  // return, if adaptive time stepping is active for current phase
  switch (phase_cccv_)
  {
    case Inpar::ElCh::CCCVHalfCyclePhase::constant_current:
    {
      adaptivetimestepping = static_cast<bool>(adaptivetimesteppingonoff_[0]);
      break;
    }
    case Inpar::ElCh::CCCVHalfCyclePhase::constant_voltage:
    {
      adaptivetimestepping = static_cast<bool>(adaptivetimesteppingonoff_[1]);
      break;
    }
    case Inpar::ElCh::CCCVHalfCyclePhase::relaxation:
    {
      adaptivetimestepping = static_cast<bool>(adaptivetimesteppingonoff_[2]);
      break;
    }
    default:
      FOUR_C_THROW("illegal CC-CV mode");
      break;
  }

  return adaptivetimestepping;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void ScaTra::CCCVHalfCycleCondition::read_restart(Core::IO::DiscretizationReader& reader)
{
  // end time of current relaxation phase
  relaxendtime_ = reader.read_double("relaxendtime");

  // current phase in half cycle
  phase_cccv_ = static_cast<Inpar::ElCh::CCCVHalfCyclePhase>(reader.read_int("phase_cccv"));
}

FOUR_C_NAMESPACE_CLOSE
