// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_visualization_parameters.hpp"

#include "4C_io_control.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
Core::IO::VisualizationParameters Core::IO::visualization_parameters_factory(
    const Teuchos::ParameterList& visualization_output_parameter_list,
    const Core::IO::OutputControl& output_control, const double restart_time)
{
  Core::IO::VisualizationParameters parameters;

  // Data format
  parameters.data_format_ = Teuchos::getIntegralValue<OutputDataFormat>(
      visualization_output_parameter_list, "OUTPUT_DATA_FORMAT");

  // Compression level
  parameters.compression_level_ = Teuchos::getIntegralValue<LibB64::CompressionLevel>(
      visualization_output_parameter_list, "COMPRESSION_LEVEL");

  // Number of digits to reserve for time step count
  parameters.digits_for_time_step_ =
      visualization_output_parameter_list.get<int>("TIMESTEP_RESERVE_DIGITS");

  parameters.directory_name_ = output_control.directory_name();

  // Parameters for output during the nonlinear solver
  parameters.every_iteration_virtual_time_increment_ =
      visualization_output_parameter_list.get<double>("EVERY_ITERATION_VIRTUAL_TIME_INCREMENT");
  parameters.digits_for_iteration_ =
      visualization_output_parameter_list.get<int>("EVERY_ITERATION_RESERVE_DIGITS");

  // This value can be overwritten from the physical field
  parameters.every_iteration_ = visualization_output_parameter_list.get<bool>("EVERY_ITERATION");

  parameters.file_name_prefix_ = output_control.file_name_only_prefix();

  parameters.restart_time_ = restart_time;

  parameters.restart_from_name_ = output_control.restart_name();

  // Type of output writer
  parameters.writer_ =
      Teuchos::getIntegralValue<OutputWriter>(visualization_output_parameter_list, "OUTPUT_WRITER");

  return parameters;
}

/*
 *
 */
[[nodiscard]] int Core::IO::get_total_digits_to_reserve_in_time_step(
    const VisualizationParameters& visualization_parameters)
{
  if (visualization_parameters.every_iteration_)
  {
    return visualization_parameters.digits_for_time_step_ +
           visualization_parameters.digits_for_iteration_;
  }
  else
  {
    return visualization_parameters.digits_for_time_step_;
  }
}

/*
 *
 */
[[nodiscard]] std::pair<double, int> Core::IO::get_time_and_time_step_index_for_output(
    const VisualizationParameters& visualization_parameters, const double time, const int step,
    const int iteration_number)
{
  if (iteration_number != 0 && !visualization_parameters.every_iteration_)
  {
    FOUR_C_THROW(
        "An iteration index was passed to get_time_and_time_step_index_for_output, but output for "
        "each iteration is deactivated, this is a contradiction");
  }

  if (visualization_parameters.every_iteration_)
  {
    return {
        time + visualization_parameters.every_iteration_virtual_time_increment_ * iteration_number,
        step * std::pow(10, visualization_parameters.digits_for_iteration_) + iteration_number};
  }
  else
  {
    return {time, step};
  }
}
FOUR_C_NAMESPACE_CLOSE
