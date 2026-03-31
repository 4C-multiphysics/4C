// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_cardiac_monodomain.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
FOUR_C_NAMESPACE_OPEN

Core::IO::InputSpec Inpar::ElectroPhysiology::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;
  Core::IO::InputSpec spec = group("CARDIAC MONODOMAIN CONTROL",
      {

          // Parameters for reaction-diffusion systems (for example cardiac electrophysiology)
          parameter<int>("WRITEMAXINTSTATE",
              {.description = "number of maximal internal state variables to be postprocessed",
                  .default_value = 0}),
          parameter<int>("WRITEMAXIONICCURRENTS",
              {.description = "number of maximal ionic currents to be postprocessed",
                  .default_value = 0}),

          parameter<double>("ACTTHRES", {.description = "threshold for the potential for computing "
                                                        "and postprocessing activation time ",
                                            .default_value = 1.0})},
      {.required = false});
  return spec;
}


FOUR_C_NAMESPACE_CLOSE