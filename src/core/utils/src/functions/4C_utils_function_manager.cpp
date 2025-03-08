// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_utils_function_manager.hpp"

#include "4C_io_input_file.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_of_time.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  using InputParameters = std::vector<Core::IO::InputParameterContainer>;

  using TypeErasedFunctionCreator = std::function<std::any(const InputParameters&)>;

  template <typename T>
  using FunctionCreator = std::shared_ptr<T> (*)(const InputParameters&);

  /**
   * Utility function that takes a function object returning a std::shared_ptr<T> and erases its
   * return type via std::any. In addition, if the returned object would be nullptr, discard
   * it and return an empty std::any instead.
   */
  template <typename T>
  TypeErasedFunctionCreator wrap_function(FunctionCreator<T> fun)
  {
    return [fun](const InputParameters& linedefs) -> std::any
    {
      std::shared_ptr<T> created = fun(linedefs);
      if (created == nullptr)
        return {};
      else
        return created;
    };
  }


  std::any create_builtin_function(const std::vector<Core::IO::InputParameterContainer>& parameters)
  {
    // List all known TryCreate functions in a vector, so they can be called with a unified
    // syntax below. Also, erase their exact return type, since we can only store std::any.
    std::vector<TypeErasedFunctionCreator> try_create_function_vector{
        wrap_function(Core::Utils::try_create_symbolic_function_of_anything),
        wrap_function(Core::Utils::try_create_symbolic_function_of_space_time),
        wrap_function(Core::Utils::try_create_function_of_time)};

    for (const auto& try_create_function : try_create_function_vector)
    {
      auto maybe_function = try_create_function(parameters);
      if (maybe_function.has_value()) return maybe_function;
    }

    FOUR_C_THROW("Internal error: could not create a function that I should be able to create.");
  }

}  // namespace


void Core::Utils::add_valid_builtin_functions(Core::Utils::FunctionManager& function_manager)
{
  using namespace IO::InputSpecBuilders;

  auto time_info = all_of({
      parameter<int>("NUMPOINTS"),
      one_of({
          group("BYNUM",
              {
                  parameter<std::vector<double>>("TIMERANGE", {.size = 2}),
              },
              {.description = "Linearly distribute NUMPOINTS time points in the TIMERANGE."}),
          parameter<std::vector<double>>("TIMES", {.size = from_parameter<int>("NUMPOINTS")}),
      }),
  });

  auto spec = one_of({
      all_of({
          parameter<std::optional<int>>("COMPONENT"),
          parameter<std::string>("SYMBOLIC_FUNCTION_OF_SPACE_TIME"),
      }),

      parameter<std::string>("SYMBOLIC_FUNCTION_OF_TIME"),

      all_of({
          parameter<int>("VARIABLE"),
          parameter<std::string>("NAME"),
          one_of({
              all_of({
                  deprecated_selection<std::string>("TYPE", {"expression"}),
                  parameter<std::string>("DESCRIPTION"),
              }),
              all_of({
                  deprecated_selection<std::string>(
                      "TYPE", {"linearinterpolation", "fourierinterpolation"}),
                  time_info,
                  parameter<std::vector<double>>(
                      "VALUES", {.size = from_parameter<int>("NUMPOINTS")}),
              }),
              all_of({
                  deprecated_selection<std::string>("TYPE", {"multifunction"}),
                  time_info,
                  parameter<std::vector<std::string>>(
                      "DESCRIPTION", {.size = [](const IO::InputParameterContainer& container)
                                         { return container.get<int>("NUMPOINTS") - 1; }}),
              }),
          }),
          group("PERIODIC",
              {
                  parameter<double>("T1"),
                  parameter<double>("T2"),
              },
              {.required = false}),
      }),

      all_of({
          parameter<std::string>("VARFUNCTION"),
          parameter<std::optional<int>>("NUMCONSTANTS"),
          parameter<std::map<std::string, double>>("CONSTANTS",
              {.default_value = std::map<std::string, double>{},
                  .size = [](const IO::InputParameterContainer& container)
                  { return container.get<std::optional<int>>("NUMCONSTANTS").value_or(0); }}),
      }),
  });

  function_manager.add_function_definition(spec, create_builtin_function);
}


Core::IO::InputSpec Core::Utils::FunctionManager::valid_function_lines()
{
  std::vector<IO::InputSpec> specs;
  for (const auto& [spec, _] : attached_function_data_)
  {
    specs.emplace_back(spec);
  }
  return Core::IO::InputSpecBuilders::one_of(
      specs, Core::IO::InputSpecBuilders::store_index_as<int>("_internal_index"));
}


void Core::Utils::FunctionManager::add_function_definition(
    IO::InputSpec spec, FunctionFactory function_factory)
{
  attached_function_data_.emplace_back(std::move(spec), std::move(function_factory));
}


void Core::Utils::FunctionManager::read_input(Core::IO::InputFile& input)
{
  functions_.clear();

  // Read FUNCT sections starting from FUNCT1 until the first empty one is encountered.
  // This implies that the FUNCT sections must form a contiguous range in the input file.
  // Otherwise, the read fails later.
  for (int funct_suffix = 1;; ++funct_suffix)
  {
    const bool stop_parsing = std::invoke(
        [&]()
        {
          Core::IO::InputParameterContainer container;
          const std::string funct_section_name = "FUNCT" + std::to_string(funct_suffix);
          if (!input.has_section(funct_section_name)) return true;

          input.match_section(funct_section_name, container);

          const auto* list = container.get_if<Core::IO::InputParameterContainer::List>(
              "FUNCT" + std::to_string(funct_suffix));

          if (!list or list->empty()) return true;

          const auto& factory =
              attached_function_data_[list->front().get<int>("_internal_index")].second;
          functions_.emplace_back(factory(*list));
          return false;
        });

    // Stop reading as soon as the first FUNCT section in the input file is empty
    if (stop_parsing) break;
  }
}

FOUR_C_NAMESPACE_CLOSE
