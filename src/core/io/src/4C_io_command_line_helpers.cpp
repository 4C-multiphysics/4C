// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_io_command_line_helpers.hpp"


FOUR_C_NAMESPACE_OPEN


std::vector<std::pair<std::filesystem::path, std::string>> build_io_pairs(
    std::vector<std::string> io_pairs, const std::filesystem::path& primary_input,
    const std::string& primary_output)
{
  std::vector<std::pair<std::filesystem::path, std::string>> io_pairs_new;

  io_pairs_new.emplace_back(primary_input, primary_output);

  if (!io_pairs.empty())
  {
    if (io_pairs.size() % 2 != 0)
    {
      FOUR_C_THROW("Positional arguments must be provided as pairs: <input> <output>.\n");
    }
    for (size_t i = 0; i < io_pairs.size(); i += 2)
      io_pairs_new.emplace_back(std::filesystem::path(io_pairs[i]), io_pairs[i + 1]);
  }
  return io_pairs_new;
}

using NPT = Core::Communication::NestedParallelismType;
void validate_argument_cross_compatibility(const CommandlineArguments& arguments)
{
  if (!arguments.group_layout.empty())
  {
    const int layout_len = static_cast<int>(arguments.group_layout.size());
    if (arguments.n_groups != layout_len)
    {
      FOUR_C_THROW(
          "When --glayout is provided its number of entries must equal --ngroup.\n "
          "Example mpirun -np 4 ./4C --ngroup=2 --glayout=1,3 \n");
    }
  }

  if (arguments.n_groups > 1 && arguments.nptype == NPT::no_nested_parallelism)
  {
    FOUR_C_THROW("when --ngroup > 1, a nested parallelism type must be specified via --nptype.\n");
  }

  if (!arguments.parameters)
  {
    const size_t num_pairs = arguments.io_pairs.size();
    if (arguments.nptype == NPT::no_nested_parallelism ||
        arguments.nptype == NPT::every_group_read_input_file)
    {
      if (num_pairs != 1)
      {
        FOUR_C_THROW(
            "when using 'no_nested_parallelism' or 'everyGroupReadInputFile' the "
            "number of <input> <output> pairs must be exactly 1.\n");
      }
    }
    else if (arguments.nptype == NPT::separate_input_files ||
             arguments.nptype == NPT::nested_multiscale)
    {
      if (static_cast<int>(num_pairs) != arguments.n_groups)
      {
        FOUR_C_THROW(
            "when using 'separateInputFiles' or 'nestedMultiscale' the number of "
            "<input> <output> pairs must equal --ngroup {}.\n",
            arguments.n_groups);
      }
    }
  }

  if (arguments.nptype != NPT::separate_input_files &&
      (arguments.restart_per_group.size() > 1 || arguments.restart_identifier_per_group.size() > 1))
  {
    FOUR_C_THROW(
        "When using --nptype other than 'separateInputFiles', only one restart step and one "
        "restartfrom identifier must be given.");
  }

  for (size_t i = 0; i < arguments.restart_identifier_per_group.size(); ++i)
  {
    if (i >= arguments.restart_per_group.size())
    {
      FOUR_C_THROW("You need to specify a restart step when using restartfrom.");
    }
  }
}

std::vector<std::string> adapt_legacy_cli_arguments(
    const std::vector<std::string>& args, LegacyCliOptions& legacy_options)
{
  if (args.empty()) return {};

  std::vector<std::string> new_args;
  new_args.reserve(args.size());
  std::vector<std::vector<std::string>> pending_vals(legacy_options.nodash_legacy_names.size());

  auto warn = [](const std::string& name, const std::string& to)
  {
    std::cerr << "DEPRECATION WARNING: Legacy argument '" << name << "' has been converted to '"
              << to
              << "'. Please update your command line arguments. This legacy form will be removed "
                 "in a future release.\n";
  };

  auto combine_and_warn = [&warn](std::vector<std::string>& out_args, const std::string& name,
                              std::vector<std::string>& vals)
  {
    if (!vals.empty())
    {
      std::string combined = std::string("--") + name + "=";
      for (size_t i = 0; i < vals.size(); ++i)
      {
        if (i) combined += ",";
        combined += vals[i];
      }
      out_args.push_back(combined);
      warn(name, combined);
      vals.clear();
    }
  };

  for (size_t i = 0; i < args.size(); ++i)
  {
    const std::string& arg = args[i];

    bool handled = false;

    // Check nodash legacy names first (e.g., "restart=1") and collect values.
    for (size_t j = 0; j < legacy_options.nodash_legacy_names.size(); ++j)
    {
      const std::string& name = legacy_options.nodash_legacy_names[j];
      std::string prefix = name + "=";
      if (arg.rfind(prefix, 0) == 0)
      {
        pending_vals[j].push_back(arg.substr(prefix.size()));
        handled = true;
        break;
      }
    }
    if (handled) continue;

    // Convert known single-dash legacy options to double-dash (e.g., "-ngroup=2" -> "--ngroup=2").
    for (const auto& name : legacy_options.single_dash_legacy_names)
    {
      std::string prefix = std::string("-") + name + "=";
      if (arg.rfind(prefix, 0) == 0)
      {
        std::string new_arg = std::string("-") + arg;
        warn(arg, new_arg);
        new_args.push_back(new_arg);
        handled = true;
        break;
      }
    }
    if (handled) continue;

    // Already a long option with two dashes: keep as-is
    if (arg.rfind("--", 0) == 0)
    {
      new_args.push_back(arg);
      continue;
    }

    // Keep everything else unchanged (positional args, -p, -h, etc.)
    new_args.push_back(arg);
  }

  for (size_t j = 0; j < legacy_options.nodash_legacy_names.size(); ++j)
  {
    combine_and_warn(new_args, legacy_options.nodash_legacy_names[j], pending_vals[j]);
  }

  return new_args;
}

void update_io_identifiers(CommandlineArguments& arguments, int group)
{
  std::filesystem::path input_filename;
  std::string output_file_identifier;

  int restart_input_index = (arguments.nptype == NPT::separate_input_files) ? group : 0;

  arguments.restart =
      arguments.restart_per_group.empty() ? 0 : arguments.restart_per_group[restart_input_index];
  std::string restart_file_identifier =
      arguments.restart_identifier_per_group.empty()
          ? ""
          : arguments.restart_identifier_per_group[restart_input_index];

  switch (arguments.nptype)
  {
    case NPT::no_nested_parallelism:
      input_filename = arguments.io_pairs[0].first;
      output_file_identifier = arguments.io_pairs[0].second;
      if (restart_file_identifier == "")
      {
        restart_file_identifier = output_file_identifier;
      }
      break;
    case NPT::every_group_read_input_file:
    {
      input_filename = arguments.io_pairs[0].first;
      std::string output_file_identifier_temp = arguments.io_pairs[0].second;
      // check whether output_file_identifier includes a dash and in case separate the number at the
      // end
      size_t pos = output_file_identifier_temp.rfind('-');
      auto extract_number_and_identifier = [](const std::string& str, size_t pos)
      {
        std::string number_str = str.substr(pos + 1);
        std::string identifier = str.substr(0, pos);
        int number = 0;
        try
        {
          size_t idx = 0;
          number = std::stoi(number_str, &idx);
          if (idx != number_str.size())
          {
            FOUR_C_THROW("Invalid numeric value in output identifier: '{}'", number_str);
          }
        }
        catch (const std::exception& e)
        {
          FOUR_C_THROW(
              "Failed to parse number in output identifier '{}': {}", number_str, e.what());
        }
        return std::make_pair(identifier, number);
      };
      if (pos != std::string::npos)
      {
        auto [identifier, number] = extract_number_and_identifier(output_file_identifier_temp, pos);
        output_file_identifier = std::format("{}_group_{}_{}", identifier, group, number);
      }
      else
      {
        output_file_identifier = std::format("{}_group_{}", output_file_identifier_temp, group);
      }
      size_t pos_r = restart_file_identifier.rfind('-');
      if (restart_file_identifier == "")
      {
        restart_file_identifier = output_file_identifier;
      }
      else if (pos_r != std::string::npos)
      {
        auto [identifier, number] = extract_number_and_identifier(restart_file_identifier, pos_r);
        restart_file_identifier = std::format("{}_group_{}-{}", identifier, group, number);
      }
      else
      {
        restart_file_identifier = std::format("{}_group_{}", restart_file_identifier, group);
      }
      break;
    }
    case NPT::separate_input_files:
    case NPT::nested_multiscale:
      input_filename = arguments.io_pairs[group].first;
      output_file_identifier = arguments.io_pairs[group].second;
      if (restart_file_identifier == "")
      {
        restart_file_identifier = output_file_identifier;
      }
      break;
    default:
      FOUR_C_THROW("-nptype value {} is not valid.", static_cast<int>(arguments.nptype));
      break;
  }
  arguments.input_file_name = input_filename;
  arguments.output_file_identifier = output_file_identifier;
  arguments.restart_file_identifier = restart_file_identifier;
}

FOUR_C_NAMESPACE_CLOSE