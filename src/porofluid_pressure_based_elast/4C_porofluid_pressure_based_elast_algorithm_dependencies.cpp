// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_elast_algorithm_dependencies.hpp"

#include "4C_global_data.hpp"
#include "4C_legacy_enum_definitions_materials.hpp"
#include "4C_mat_par_bundle.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

PoroPressureBased::PorofluidElastAlgorithmDeps
PoroPressureBased::make_elast_algorithm_deps_from_problem(Global::Problem& problem)
{
  return PorofluidElastAlgorithmDeps{
      .discretization_by_name = [&problem](const std::string& discretization_name)
      { return problem.get_dis(discretization_name); },
      .cloning_material_map = &problem.cloning_material_map(),
      .porofluid_pressure_based_dynamic_parameters =
          &problem.porofluid_pressure_based_dynamic_params(),
      .validate_porofluid_material_id =
          [&problem](int matid)
      {
        const Core::Materials::MaterialType mtype =
            problem.materials()->parameter_by_id(matid)->type();
        if ((mtype != Core::Materials::m_fluidporo_multiphase) and
            (mtype != Core::Materials::m_fluidporo_multiphase_reactions))
          FOUR_C_THROW(
              "Material with ID {} is not admissible for porofluid multiphase elements", matid);
      },
      .solver_params_by_id = problem.solver_params_callback(),
      .verbosity =
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(problem.io_params(), "VERBOSITY"),
      .input_control_file = problem.input_control_file(),
      .add_field_test = [&problem](std::shared_ptr<Core::Utils::ResultTest> result_test)
      { problem.add_field_test(result_test); },
      .porofluid_algorithm_deps = make_algorithm_deps_from_problem(problem),
  };
}

FOUR_C_NAMESPACE_CLOSE
