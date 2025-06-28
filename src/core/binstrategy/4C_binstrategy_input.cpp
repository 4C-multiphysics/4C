// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_binstrategy_input.hpp"

#include "4C_binstrategy.hpp"
#include "4C_io_input_spec_builders.hpp"
FOUR_C_NAMESPACE_OPEN



void Core::Binstrategy::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using namespace Core::IO::InputSpecBuilders;

  list["BINNING STRATEGY"] = group("BINNING STRATEGY",
      {

          parameter<double>("BIN_SIZE_LOWER_BOUND",
              {.description =
                      "Lower bound for bin size. Exact bin size is computed via (Domain edge "
                      "length)/BIN_SIZE_LOWER_BOUND. This also determines the number of bins in "
                      "each spatial direction",
                  .default_value = -1.0}),

          parameter<std::string>("BIN_PER_DIR",
              {.description =
                      "Number of bins per direction (x, y, z) in particle simulations. Either "
                      "Define this value or  BIN_SIZE_LOWER_BOUND",
                  .default_value = "-1 -1 -1"}),


          parameter<std::string>("PERIODICONOFF",
              {.description = "Turn on/off periodic boundary conditions in each spatial direction",
                  .default_value = "0 0 0"}),

          parameter<std::string>("DOMAINBOUNDINGBOX",
              {.description = "Bounding box for computational domain using binning "
                              "strategy. Specify diagonal corner  points",
                  .default_value = "1e12 1e12 1e12 1e12 1e12 1e12"}),


          parameter<Core::Binstrategy::WriteBins>(
              "WRITEBINS", {.description = "Write none, row or column bins for visualization",
                               .default_value = Core::Binstrategy::WriteBins::none})},
      {.required = false});
}

FOUR_C_NAMESPACE_CLOSE