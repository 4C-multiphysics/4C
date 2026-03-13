// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_scatra_algorithm_dependencies.hpp"

#include "4C_global_data.hpp"
#include "4C_global_legacy_module_problem_type.hpp"
#include "4C_utils_function.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

PoroPressureBased::PorofluidElastScatraArteryCouplingDeps
PoroPressureBased::make_artery_coupling_deps_from_problem(Global::Problem& problem)
{
  return PorofluidElastScatraArteryCouplingDeps{
      .porofluid_pressure_based_dynamic_parameters =
          &problem.porofluid_pressure_based_dynamic_params(),
      .pure_porofluid_problem =
          problem.get_problem_type() == Core::ProblemType::porofluid_pressure_based,
      .spatial_dimension = problem.n_dim(),
      .function_manager = &problem.function_manager(),
      .function_of_anything_by_id =
          [&problem](const int function_id) -> const Core::Utils::FunctionOfAnything&
      { return problem.function_by_id<Core::Utils::FunctionOfAnything>(function_id); },
  };
}

PoroPressureBased::PorofluidElastScatraAlgorithmDeps
PoroPressureBased::make_elast_scatra_algorithm_deps_from_problem(Global::Problem& problem)
{
  return PorofluidElastScatraAlgorithmDeps{
      .porofluid_elast_algorithm_deps = make_elast_algorithm_deps_from_problem(problem),
      .artery_coupling_deps = make_artery_coupling_deps_from_problem(problem),
      .poro_multi_phase_scatra_dynamic_parameters =
          &problem.poro_multi_phase_scatra_dynamic_params(),
      .solver_params_by_id = problem.solver_params_callback(),
      .verbosity =
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(problem.io_params(), "VERBOSITY"),
      .add_field_test = [&problem](std::shared_ptr<Core::Utils::ResultTest> result_test)
      { problem.add_field_test(result_test); },
  };
}

FOUR_C_NAMESPACE_CLOSE
