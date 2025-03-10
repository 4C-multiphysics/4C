// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_IO_runtime_output.hpp"

#include "4C_io_visualization_parameters.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Inpar
{
  namespace IORuntimeOutput
  {
    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
    {
      using Teuchos::tuple;
      using namespace Core::IO::InputSpecBuilders;

      // related sublist
      Core::Utils::SectionSpecs sublist_IO{"IO"};
      Core::Utils::SectionSpecs sublist_IO_VTK_structure{sublist_IO, "RUNTIME VTK OUTPUT"};


      // output interval regarding steps: write output every INTERVAL_STEPS steps
      sublist_IO_VTK_structure.specs.emplace_back(parameter<int>("INTERVAL_STEPS",
          {.description = "write visualization output at runtime every INTERVAL_STEPS steps",
              .default_value = -1}));
      sublist_IO_VTK_structure.specs.emplace_back(parameter<int>("STEP_OFFSET",
          {.description = "An offset added to the current step to shift the steps to be written.",
              .default_value = 0}));


      // data format for written numeric data
      Core::Utils::string_to_integral_parameter<Core::IO::OutputDataFormat>("OUTPUT_DATA_FORMAT",
          "binary", "data format for written numeric data", tuple<std::string>("binary", "ascii"),
          tuple<Core::IO::OutputDataFormat>(
              Core::IO::OutputDataFormat::binary, Core::IO::OutputDataFormat::ascii),
          sublist_IO_VTK_structure);

      // compression level of written output
      Core::Utils::string_to_integral_parameter<LibB64::CompressionLevel>("COMPRESSION_LEVEL",
          "best_speed", "Specify the compression level of written vtk output.",
          tuple<std::string>("best_compression", "best_speed", "no_compression"),
          tuple<LibB64::CompressionLevel>(LibB64::CompressionLevel::best_compression,
              LibB64::CompressionLevel::best_speed, LibB64::CompressionLevel::no_compression),
          sublist_IO_VTK_structure);

      // specify the maximum digits in the number of time steps that shall be written#
      sublist_IO_VTK_structure.specs.emplace_back(parameter<int>("TIMESTEP_RESERVE_DIGITS",
          {.description = "Specify the maximum digits in the number of time steps that shall be "
                          "written. This only affects the number of leading zeros in the output "
                          "file names.",
              .default_value = 5}));

      // whether to write output in every iteration of the nonlinear solver
      sublist_IO_VTK_structure.specs.emplace_back(parameter<bool>("EVERY_ITERATION",
          {.description = "write output in every iteration of the nonlinear solver",
              .default_value = false}));

      // virtual time increment that is added for each nonlinear output state
      sublist_IO_VTK_structure.specs.emplace_back(
          parameter<double>("EVERY_ITERATION_VIRTUAL_TIME_INCREMENT",
              {.description = "Specify the virtual time increment that is added for each nonlinear "
                              "output state",
                  .default_value = 1e-8}));

      // specify the maximum digits in the number of iterations that shall be written
      sublist_IO_VTK_structure.specs.emplace_back(parameter<int>("EVERY_ITERATION_RESERVE_DIGITS",
          {.description = "Specify the maximum digits in the number of iterations that shall be "
                          "written. This only affects the number of leading zeros in the output "
                          "file names.",
              .default_value = 4}));

      // specify the actual visualization writer
      Core::Utils::string_to_integral_parameter<Core::IO::OutputWriter>("OUTPUT_WRITER",
          "vtu_per_rank",
          "Specify which output writer shall be used to write the visualization data to disk",
          tuple<std::string>("vtu_per_rank"),
          tuple<Core::IO::OutputWriter>(Core::IO::OutputWriter::vtu_per_rank),
          sublist_IO_VTK_structure);

      sublist_IO_VTK_structure.move_into_collection(list);
    }


  }  // namespace IORuntimeOutput
}  // namespace Inpar

FOUR_C_NAMESPACE_CLOSE
