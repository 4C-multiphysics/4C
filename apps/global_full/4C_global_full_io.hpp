// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GLOBAL_FULL_IO_HPP
#define FOUR_C_GLOBAL_FULL_IO_HPP

#include "4C_config.hpp"

#include "4C_comm_utils.hpp"
#include "4C_io_command_line_helpers.hpp"
#include "4C_io_input_file.hpp"

FOUR_C_NAMESPACE_OPEN

/**
 * \brief Initializes the input file for reading.
 * \note Currently, this function is a wrapper around Global::set_up_input_file to keep the main
 * routine separate from the Global namespace. In the future, the input file will be set up based on
 * the activated modules.
 */
Core::IO::InputFile setup_input_file(MPI_Comm comm);

/**
 * \brief Emits general metadata to the YAML root reference.
 * \note Currently, this function is a wrapper around Global::emit_general_metadata to keep the main
 * routine separate from the Global namespace. In the future, this function will emit metadata based
 * on the activated modules.
 */
void emit_general_metadata(const Core::IO::YamlNodeRef& root_ref);

/**
 * \brief Sets up the Global::Problem instance and puts all the parameters from the input file
 * there.
 */
void setup_global_problem(Core::IO::InputFile& input_file, const CommandlineArguments& arguments,
    Core::Communication::Communicators& communicators);
/**
 * \brief Returns the wall time in seconds.
 */
double walltime_in_seconds();



/**
 * \brief Writes the Teuchos::TimeMonitor information to std::cout
 */
void write_timemonitor(MPI_Comm comm);
FOUR_C_NAMESPACE_CLOSE

#endif
