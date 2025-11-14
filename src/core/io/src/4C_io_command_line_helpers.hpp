// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_COMMAND_LINE_HELPERS_HPP
#define FOUR_C_IO_COMMAND_LINE_HELPERS_HPP

#include "4C_config.hpp"

#include "4C_comm_utils.hpp"

#include <filesystem>

FOUR_C_NAMESPACE_OPEN

/**
 * \brief Structure to hold command line arguments.
 */
struct CommandlineArguments
{
  bool help = false;
  int n_groups = 1;
  bool parameters = false;
  std::vector<int> group_layout = {};
  Core::Communication::NestedParallelismType nptype =
      Core::Communication::NestedParallelismType::no_nested_parallelism;
  int diffgroup = -1;
  int restart = 0;
  std::string restart_file_identifier = "";
  std::vector<int> restart_per_group = {};
  std::vector<std::string> restart_identifier_per_group = {};
  bool interactive = false;
  std::vector<std::pair<std::filesystem::path, std::string>> io_pairs;
  std::filesystem::path input_file_name = "";
  std::string output_file_identifier = "";
};

/**
 * \brief Build canonical input/output pairs from positional command line arguments.
 * Due to the legacy argument structure, we separate between the primary input and output (first two
 * positional arguments) and the rest (io_pairs). The latter are only required when using nested
 * parallelism with separate input files.
 * \param io_pairs Vector of strings from the command line representing input/output pairs.
 * \param primary_input The primary input file name (first positional argument).
 * \param primary_output The primary output file identifier (second positional argument).
 * \return A vector of pairs of input file paths and output file identifiers.
 */
std::vector<std::pair<std::filesystem::path, std::string>> build_io_pairs(
    std::vector<std::string> io_pairs, const std::filesystem::path& primary_input,
    const std::string& primary_output);

/**
 * \brief Validates cross-compatibility of command line options.
 * \param arguments The parsed command line arguments.
 */
void validate_argument_cross_compatibility(const CommandlineArguments& arguments);

/**
 * \brief Structure to hold legacy CLI option names.
 */
struct LegacyCliOptions
{
  std::vector<std::string> single_dash_legacy_names;
  std::vector<std::string> nodash_legacy_names;
};
/**
 * \brief Adapt legacy command line arguments.
 * This function converts legacy single-dash options (e.g. "-ngroup=2") into
 * their new form ("--ngroup=2") and combines legacy dashless positional
 * options (e.g. "restart=1") into comma-separated lists ("--restart=1,2").
 * \param args Input arguments (no program-name expected).
 * \param legacy_options Structure containing names of legacy options:
 *        - single_dash_legacy_names: Options that used a single dash and
 *          should be converted to the double-dash form (e.g. {"ngroup", "glayout"}).
 *        - nodash_legacy_names: Legacy dashless options that should be
 *          collected/combined (e.g. {"restart", "restartfrom"}).
 * \return Sanitized vector of arguments.
 */
std::vector<std::string> adapt_legacy_cli_arguments(
    const std::vector<std::string>& args, LegacyCliOptions& legacy_options);

/**
 * \brief Updates input/output identifiers based on group id and nested parallelism type.
 * \param arguments The command line arguments to update.
 * \param group The group id of the current process.
 */
void update_io_identifiers(CommandlineArguments& arguments, int group);

FOUR_C_NAMESPACE_CLOSE

#endif