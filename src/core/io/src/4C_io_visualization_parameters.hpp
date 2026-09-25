// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_VISUALIZATION_PARAMETERS_HPP
#define FOUR_C_IO_VISUALIZATION_PARAMETERS_HPP


#include "4C_config.hpp"

#include "4C_utils_parameter_list.fwd.hpp"

#include <cstdint>
#include <string>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace LibB64
{
  enum class CompressionLevel;
}

namespace Core::IO
{
  class OutputControl;

  /// data format for written numeric data
  enum class OutputDataFormat : std::uint8_t
  {
    ascii,
    binary
  };

  /// Specify the output writer that shall be used
  enum class OutputWriter : std::uint8_t
  {
    vtu_collective,  ///< Write one shared file per time step for all ranks via MPI-IO
    vtu_per_rank     ///< Write one file per time step per rank in the vtu format
  };

  /**
   * @brief This struct holds parameters for visualization output
   */
  struct VisualizationParameters
  {
    //! Enum containing the type of output data format, i.e., binary or ascii.
    OutputDataFormat data_format_;

    //! Level of compression used when writing output.
    LibB64::CompressionLevel compression_level_;

    //! Base output directory
    std::string directory_name_;

    //! Flag if output should be written for each nonlinear iteration
    bool every_iteration_;

    //! We need to add a small time increment for each iteration to avoid having "double" time steps
    double every_iteration_virtual_time_increment_;

    //! Only the prefix of the file name, without the directory
    std::string file_name_prefix_;

    //! Number of digits in the final "time_step_index" reserved for the nonlinear iteration
    int digits_for_iteration_;

    //! Number of digits in the final "time_step_index" reserved for the nonlinear iteration
    int digits_for_time_step_;

    //! Time the simulation is restarted from
    double restart_time_;

    //! In case of restart this prefix specifies the control file we read which might contain a path
    std::string restart_from_name_;

    //! Enum containing the output writer that shall be used
    OutputWriter writer_;
  };

  /**
   * @brief Create a container containing all visualization output parameters
   */
  [[nodiscard]] VisualizationParameters visualization_parameters_factory(
      const Teuchos::ParameterList& visualization_output_parameter_list,
      const Core::IO::OutputControl& output_control, double restart_time);

  /**
   * @brief Return the total number of digits to reserve in the time step numbering
   * @param visualization_parameters (in) Reference to a parameters container
   */
  [[nodiscard]] int get_total_digits_to_reserve_in_time_step(
      const VisualizationParameters& visualization_parameters);

  /**
   * @brief Get the time step value and index that will be stored in the output file
   *
   * Since we have the option to output states during the nonlinear solution process, we use the
   * following convention for time step values and indices:
   * - For standard output the 4C internal time step value and index are used
   * - For output during each iteration, we increment the 4C internal time step value with a
   *   small increment for each nonlinear iteration, and we "add" the iteration index to the time
   *   step index, e.g., the time step index 0000270005 corresponds to the 5th nonlinear iteration
   *   in the 28th step -> 0000280000 is the converged 28th step.
   *
   * @param visualization_parameters (in) Reference to a parameters container
   * @param time (in) 4C internal time
   * @param step (in) 4C internal time step index of the last converged state
   * @param iteration_number (in) Number of nonlinear iteration
   */
  [[nodiscard]] std::pair<double, int> get_time_and_time_step_index_for_output(
      const VisualizationParameters& visualization_parameters, double time, int step,
      int iteration_number = 0);
}  // namespace Core::IO


FOUR_C_NAMESPACE_CLOSE

#endif
