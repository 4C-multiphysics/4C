// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_reduced_lung_input.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_io_yaml.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

#include <string>

namespace
{
  using namespace FourC;

  //! A complete `reduced_dimensional_lung` section carrying one pressure boundary condition, so
  //! that a test can make a single value invalid and match the rest.
  std::string input_with_pressure_condition(const std::string& id, const std::string& function_id)
  {
    return R"(
reduced_dimensional_lung:
  air_properties:
    density: 1.176e-06
    dynamic_viscosity: 1.79105e-05
  dynamics:
    linear_solver: 1
    number_of_steps: 1
    restart_every: 1
    results_every: 1
    time_increment: 1
  geometry:
    file: "mesh.vtu"
  lung_tree:
    airways:
      element_blocks: [1]
      radius:
        constant: 1.0
      flow_model:
        resistance_type:
          constant: Linear
        include_inertia:
          constant: false
      wall_model_type:
        constant: Rigid
    terminal_units:
      element_blocks: [2]
      rheological_model:
        rheological_model_type:
          constant: KelvinVoigt
        kelvin_voigt:
          viscosity_kelvin_voigt_eta:
            constant: 0.0
      elasticity_model:
        elasticity_model_type:
          constant: Linear
        linear:
          elasticity_e:
            constant: 1.0
  boundary_conditions:
    pressure:
      - id: )" +
           id +
           R"(
        function_id: )" +
           function_id + "\n";
  }

  void match_input(const std::string& yaml_string)
  {
    Core::IO::InputParameterContainer container;
    ryml::Tree tree = Core::IO::init_yaml_tree_with_exceptions();
    ryml::NodeRef root = tree.rootref();
    ryml::parse_in_arena(ryml::to_csubstr(yaml_string), root);

    Core::IO::ConstYamlNodeRef node(root, "");
    ReducedLung::valid_parameters().match(node, container);
  }

  TEST(ReducedLungInputTest, AcceptsPositiveDefinitionAndFunctionIds)
  {
    EXPECT_NO_THROW(match_input(input_with_pressure_condition("1", "1")));
  }

  //! The id 0 marks nodes without a boundary condition and can therefore not be defined.
  TEST(ReducedLungInputTest, RejectsNonPositiveDefinitionId)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(match_input(input_with_pressure_condition("0", "1")),
        Core::Exception, "'id' does not pass validation");
  }

  TEST(ReducedLungInputTest, RejectsNonPositiveFunctionId)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(match_input(input_with_pressure_condition("1", "-1")),
        Core::Exception, "'function_id' does not pass validation");
  }
}  // namespace
