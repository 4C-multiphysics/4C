// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_ALGORITHM_DEPENDENCIES_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_ALGORITHM_DEPENDENCIES_HPP

#include "4C_config.hpp"

#include "4C_io_pstream.hpp"
#include "4C_porofluid_pressure_based_algorithm_dependencies.hpp"

#include <Teuchos_ParameterList.hpp>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::IO
{
  class InputControl;
}  // namespace Core::IO

namespace Core::Utils
{
  class ResultTest;
}  // namespace Core::Utils

namespace Global
{
  class Problem;
}  // namespace Global

namespace PoroPressureBased
{
  struct PorofluidElastAlgorithmDeps
  {
    std::function<std::shared_ptr<Core::FE::Discretization>(const std::string&)>
        discretization_by_name;
    const std::map<std::pair<std::string, std::string>, std::map<int, int>>* cloning_material_map;
    const ::Teuchos::ParameterList* porofluid_pressure_based_dynamic_parameters;
    std::function<void(int)> validate_porofluid_material_id;
    std::function<const ::Teuchos::ParameterList&(int)> solver_params_by_id;
    Core::IO::Verbositylevel verbosity;
    std::shared_ptr<Core::IO::InputControl> input_control_file;
    std::function<void(std::shared_ptr<Core::Utils::ResultTest>)> add_field_test;
    PorofluidAlgorithmDeps porofluid_algorithm_deps;
  };

  PorofluidElastAlgorithmDeps make_elast_algorithm_deps_from_problem(Global::Problem& problem);
}  // namespace PoroPressureBased

FOUR_C_NAMESPACE_CLOSE

#endif
