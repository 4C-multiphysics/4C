// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ALGORITHM_DEPENDENCIES_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_ALGORITHM_DEPENDENCIES_HPP

#include "4C_config.hpp"

#include "4C_io_pstream.hpp"
#include "4C_porofluid_pressure_based_elast_algorithm_dependencies.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <functional>
#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{
  class FunctionManager;
  class FunctionOfAnything;
  class ResultTest;
}  // namespace Core::Utils

namespace Global
{
  class Problem;
}  // namespace Global

namespace PoroPressureBased
{
  struct PorofluidElastScatraArteryCouplingDeps
  {
    const Teuchos::ParameterList* porofluid_pressure_based_dynamic_parameters = nullptr;
    bool pure_porofluid_problem = false;
    int spatial_dimension = 0;
    const Core::Utils::FunctionManager* function_manager = nullptr;
    std::function<const Core::Utils::FunctionOfAnything&(int)> function_of_anything_by_id;
  };

  struct PorofluidElastScatraAlgorithmDeps
  {
    PorofluidElastAlgorithmDeps porofluid_elast_algorithm_deps;
    PorofluidElastScatraArteryCouplingDeps artery_coupling_deps;
    const Teuchos::ParameterList* poro_multi_phase_scatra_dynamic_parameters = nullptr;
    std::function<const Teuchos::ParameterList&(int)> solver_params_by_id;
    Core::IO::Verbositylevel verbosity = Core::IO::minimal;
    std::function<void(std::shared_ptr<Core::Utils::ResultTest>)> add_field_test;
  };

  PorofluidElastScatraArteryCouplingDeps make_artery_coupling_deps_from_problem(
      Global::Problem& problem);

  PorofluidElastScatraAlgorithmDeps make_elast_scatra_algorithm_deps_from_problem(
      Global::Problem& problem);
}  // namespace PoroPressureBased

FOUR_C_NAMESPACE_CLOSE

#endif
