// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_adapter_str_timeloop.hpp"

#include "4C_global_data.hpp"
#include "4C_structure_new_input.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Adapter::StructureTimeLoop::StructureTimeLoop(
    Global::Problem& problem, std::shared_ptr<Structure> structure)
    : StructureWrapper(structure), problem_(problem)
{
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int Adapter::StructureTimeLoop::integrate()
{
  // error checking variables
  Solid::ConvergenceStatus convergencestatus = Solid::conv_success;

  // target time #timen_ and step #stepn_ already set
  // time loop
  while (not_finished() and
         (convergencestatus == Solid::conv_success or convergencestatus == Solid::conv_fail_repeat))
  {
    // call the predictor
    pre_predict();
    prepare_time_step();

    // integrate time step, i.e. do corrector steps
    // after this step we hold disn_, etc
    pre_solve();
    convergencestatus = solve();

    // if everything is fine
    if (convergencestatus == Solid::conv_success)
    {
      // calculate stresses, strains and energies
      // note: this has to be done before the update since otherwise a potential
      // material history is overwritten
      constexpr bool force_prepare = false;
      prepare_output(force_prepare);

      // update displacements, velocities, accelerations
      // after this call we will have disn_==dis_, etc
      // update time and step
      // update everything on the element level
      pre_update();
      update();
      post_update();

      // write output
      output();
      StructureWrapper::post_output();
      post_output();

      // print info about finished time step
      print_step();
    }
    // todo: remove this as soon as old structure time integration is gone
    else if (Teuchos::getIntegralValue<Solid::IntegrationStrategy>(
                 problem_.structural_dynamic_params(), "INT_STRATEGY") == Solid::int_old)
    {
      convergencestatus =
          perform_error_action(convergencestatus);  // something went wrong update error code
                                                    // according to chosen divcont action
    }
  }

  post_time_loop();

  // that's it say what went wrong
  return convergencestatus;
}

FOUR_C_NAMESPACE_CLOSE
