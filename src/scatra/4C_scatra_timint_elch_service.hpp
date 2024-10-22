// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_TIMINT_ELCH_SERVICE_HPP
#define FOUR_C_SCATRA_TIMINT_ELCH_SERVICE_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_inpar_elch.hpp"
#include "4C_scatra_timint_implicit.hpp"

FOUR_C_NAMESPACE_OPEN

/*==========================================================================*/
// forward declarations
/*==========================================================================*/

namespace ScaTra
{
  /*!
   * \brief Control routine for constant-current, constant-voltage half cycle
   *
   * Holds all values for one constant-current, constant-voltage half cycle, including static
   * values from input file and dynamic values. This class controls the current phase within one
   * half cycle.
   */
  class CCCVHalfCycleCondition
  {
   public:
    //! constructor
    CCCVHalfCycleCondition(
        const Core::Conditions::Condition& cccvhalfcyclecondition, bool adaptivetimestepping);

    //! Get phase of half cycle
    Inpar::ElCh::CCCVHalfCyclePhase get_cccv_half_cycle_phase() const { return phase_cccv_; };

    //! Get ID of this half cycle condition
    int get_condition_id() const { return halfcyclecondition_id_; };

    //! get cut off c-rate during constant voltage
    double get_cut_off_c_rate() const { return cutoffcrate_; };

    //! get cut off voltage during constant current
    double get_cut_off_voltage() const { return cutoffvoltage_; };

    //! get end time of current relaxation phase
    double get_relax_end_time() const { return relaxendtime_; };

    //! does this phase have adaptive time stepping?
    bool is_adaptive_time_stepping_phase() const;

    //! is this half cycle completed after updating the phase?
    bool is_end_of_half_cycle_next_phase(double time, bool print);

    //! reset phase of this half cycle to constant current
    void reset_phase();

    //! read restart
    void read_restart(Core::IO::DiscretizationReader& reader);

   private:
    //! adaptive time stepping at end of phases?
    std::vector<int> adaptivetimesteppingonoff_;

    //! cut off c-rate during constant voltage
    const double cutoffcrate_;

    //! ut off voltage during constant current
    const double cutoffvoltage_;

    //! ID of this half cycle condition
    const int halfcyclecondition_id_;

    //! flag indicating whether cell is currently being operated in constant-current (CC),
    //! constant-voltage (CV), relaxation (RX), or initial relaxation mode
    Inpar::ElCh::CCCVHalfCyclePhase phase_cccv_;

    //! end time of current relaxation phase
    double relaxendtime_;

    //! duration of relaxation phase
    const double relaxtime_;
  };  // class CCCVHalfCycleCondition

  /*========================================================================*/
  /*========================================================================*/

  /*!
   * \brief Control routine for constant-current, constant-voltage condition
   *
   * Holds two half cycles (charge and discharge). Controls, which half cycle is activated. Serves
   * as interface between one half cycle and time integration.
   */
  class CCCVCondition
  {
   public:
    //! constructor
    CCCVCondition(const Core::Conditions::Condition& cccvcyclingcondition,
        const std::vector<Core::Conditions::Condition*>& cccvhalfcycleconditions,
        bool adaptivetimestepping, int num_dofs);

    //! true, when all half cylces are completed
    bool not_finished() const { return nhalfcycles_ >= ihalfcycle_; };

    //! phase of active half cycle
    Inpar::ElCh::CCCVHalfCyclePhase get_cccv_half_cycle_phase() const;

    //! ID of current half cycle
    int get_half_cycle_condition_id() const;

    double get_initial_relax_time() const { return initrelaxtime_; };

    //! number of current half cycle
    int get_num_current_half_cycle() const { return ihalfcycle_; };

    //! Step when phase was changed last time
    int get_step_last_phase_change() const { return steplastphasechange_; };

    //! get end time of current relaxation phase
    double get_relax_end_time() const;

    //! does this phase have adaptive time stepping?
    bool is_adaptive_time_stepping_phase() const;

    //! is this phase finished?
    bool is_end_of_half_cycle_phase(double cellvoltage, double cellcrate, double time) const;

    //! is cut off c rate exceeded?
    bool exceed_cell_c_rate(double expected_cellcrate) const;

    //! does cell voltage exceed bounds of current half cycle
    bool exceed_cell_voltage(double expected_cellvoltage) const;

    //! is this condition in initial relaxation?
    bool is_initial_relaxation(const double time, const double dt) const
    {
      return time <= initrelaxtime_ - dt;
    }

    //! was phase changed since last adaption of time step?
    bool is_phase_changed() const { return phasechanged_; };

    //! are we in initial relaxation phase?
    bool is_phase_initial_relaxation() const { return phaseinitialrelaxation_; };

    //! true if difference between @p step and the step when the phase was last changed is equal
    //! or more than  num_add_adapt_timesteps_
    bool exceed_max_steps_from_last_phase_change(int step);

    //! return minimum amount of time steps the initial relaxation time of cccv condition has to be
    //! discretized with
    int min_time_steps_during_init_relax() const { return min_time_steps_during_init_relax_; }

    //! go to next phase. If half cycle is finished switch to other half cycle and start with
    //! constant current
    void next_phase(int step, double time, bool print);

    //! number of dofs of this cccv condition
    int num_dofs() const { return num_dofs_; }

    //! read restart
    void read_restart(Core::IO::DiscretizationReader& reader);

    //! reset phasechanged_
    void reset_phase_change_observer();

    //! initialize first half cycle
    void set_first_cccv_half_cycle(int step);

    //! start counting steps from now on (phase was changed)
    void set_phase_change_observer(int step);

   private:
    //! adaptive time stepping for initial relaxation?
    const int adaptivetimesteppingonoff_;

    //! how to begin with CCCV condition (charge/discharge)
    const bool beginwithcharge_;

    //! flag indicating whether cell is currently being charged or discharged
    bool charging_;

    //! half cycle of charge
    Teuchos::RCP<ScaTra::CCCVHalfCycleCondition> halfcycle_charge_;

    //! half cycle of discharge
    Teuchos::RCP<ScaTra::CCCVHalfCycleCondition> halfcycle_discharge_;

    //! number of current charge or discharge half-cycle
    int ihalfcycle_;

    //! initial relaxation time of cccv condition
    const double initrelaxtime_;

    //! minimum amount of time steps the initial relaxation time of cccv condition has to be
    //! discretized with
    const int min_time_steps_during_init_relax_;

    //! total number of charge and discharge half-cycles
    const int nhalfcycles_;

    //! number of steps after phase was changed until reset time step to original value
    const int num_add_adapt_timesteps_;

    //! number of dofs of this cccv condition
    const int num_dofs_;

    //! for time step adaptivity: was phase changed since last time step adaptivity?
    bool phasechanged_;

    //! indicating, if currently in initial relaxation
    bool phaseinitialrelaxation_;

    //! step when phase lastly changed
    int steplastphasechange_;

  };  // class CCCVCondition

}  // namespace ScaTra
FOUR_C_NAMESPACE_CLOSE

#endif
