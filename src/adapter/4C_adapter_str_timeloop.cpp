// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_str_timeloop.hpp"

#include "4C_adapter_str_structure.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Adapter::StructureTimeLoop::StructureTimeLoop(std::shared_ptr<Structure> structure)
    : StructureWrapper(std::move(structure))
{
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::StructureTimeLoop::integrate()
{
  // target time #timen_ and step #stepn_ already set
  // time loop
  while (not_finished())
  {
    // call the predictor
    prepare_time_step();

    // integrate time step, i.e. do corrector steps
    // after this step we hold disn_, etc
    const Solid::StepStatus solve_status = solve();
    switch (perform_error_action(solve_status))
    {
      case Solid::StepAction::retry_step:
        continue;
      case Solid::StepAction::accept_step:
        break;  // do nothing
    }

    // calculate stresses, strains and energies
    // note: this has to be done before the update since otherwise a potential
    // material history is overwritten
    prepare_output(false);

    // update displacements, velocities, accelerations
    // after this call we will have disn_==dis_, etc
    // update time and step
    // update everything on the element level
    pre_update();
    update();
    post_update();

    // write output
    output();

    finalize_successful_step();
  }

  post_time_loop();
}

FOUR_C_NAMESPACE_CLOSE
