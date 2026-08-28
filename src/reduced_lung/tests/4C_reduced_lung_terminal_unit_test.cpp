// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_reduced_lung_terminal_unit.hpp"

#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_reduced_lung_helpers.hpp"
#include "4C_reduced_lung_terminal_unit_elasticity.hpp"
#include "4C_reduced_lung_terminal_unit_model_registry.hpp"
#include "4C_reduced_lung_terminal_unit_recruitment.hpp"
#include "4C_reduced_lung_terminal_unit_rheology.hpp"
#include "4C_reduced_lung_test_utils_test.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <unordered_map>
#include <vector>

namespace
{
  namespace Core = FourC::Core;
  using namespace FourC::ReducedLung;
  using namespace FourC::Core::LinAlg;
  using namespace FourC::ReducedLung::TestUtils;
  using namespace FourC::ReducedLung::TerminalUnits;

  using RheologicalModelType =
      ReducedLungParameters::LungTree::TerminalUnits::RheologicalModel::RheologicalModelType;
  using ElasticityModelType =
      ReducedLungParameters::LungTree::TerminalUnits::ElasticityModel::ElasticityModelType;
  using PressureLawType =
      ReducedLungParameters::LungTree::TerminalUnits::RecruitmentModel::PressureLawType;
  // The same enum under its registry-facing name, where "recruitment" is not otherwise in scope.
  using ModelRegistry::RecruitmentModelType;
  using TimeLawType = ReducedLungParameters::LungTree::TerminalUnits::RecruitmentModel::TimeLawType;
  using HysteresisPath =
      ReducedLungParameters::LungTree::TerminalUnits::RecruitmentModel::HysteresisPath;
  using TerminalUnits::Recruitment::LinearPressureRecruitment;
  using TerminalUnits::Recruitment::NoRecruitment;

  struct TerminalUnitModelCase
  {
    const char* name;
    RheologicalModelType rheological_model_type;
    ElasticityModelType elasticity_model_type;
    std::array<double, 3> kelvin_voigt_eta;
    std::array<double, 3> maxwell_e_m;
    std::array<double, 3> maxwell_eta_m;
    std::array<double, 3> linear_elasticity_e;
    std::array<double, 3> ogden_kappa;
    std::array<double, 3> ogden_beta;
    RecruitmentModelType recruitment_model_type = RecruitmentModelType::None;
    TimeLawType time_law_type = TimeLawType::None;
    std::array<double, 3> dofs = {1.0, 1.0, 1.0};
    std::array<double, 3> recruitment_initial_v0 = {1.2, 1.3, 1.4};
    std::array<double, 3> recruitment_tau = {0.5, 0.6, 0.7};
    bool check_completed_jacobian_update = false;
  };

  Core::IO::InputField<double> make_elementwise_double_field(const std::array<double, 3>& values)
  {
    return Core::IO::InputField<double>(
        std::unordered_map<int, double>{{1, values[0]}, {2, values[1]}, {3, values[2]}});
  }

  struct RecruitmentFixtureData
  {
    TerminalUnitData data;
    TerminalUnits::Recruitment::RecruitmentModel recruitment_model;
  };

  RecruitmentFixtureData make_single_recruitment_data(const PressureLawType pressure_law_type,
      const TimeLawType time_law_type = TimeLawType::None,
      const HysteresisPath active_path = HysteresisPath::Opening)
  {
    RecruitmentFixtureData fixture;
    auto& data = fixture.data;
    data.global_element_id = {0};
    data.local_element_id = {0};
    data.local_row_id = {0};
    data.lid_p1 = {0};
    data.lid_p2 = {1};
    data.lid_q = {2};
    data.volume_v = {1.2};

    if (pressure_law_type == PressureLawType::None)
    {
      fixture.recruitment_model = NoRecruitment{.v0 = {4.0}};
      return fixture;
    }

    LinearPressureRecruitment recruitment_model;
    recruitment_model.pressure_law.active_path_n = {active_path};
    recruitment_model.pressure_law.v0_min = {1.0};
    recruitment_model.pressure_law.v0_max = {2.0};
    recruitment_model.pressure_law.p_closing_min = {-0.2};
    recruitment_model.pressure_law.p_opening_min = {0.0};
    recruitment_model.pressure_law.delta_p_minmax = {1.0};
    recruitment_model.pressure_law.epsilon_v0_switch = {0.05};
    recruitment_model.time_law.type = {time_law_type};
    recruitment_model.time_law.tau = {0.5};
    recruitment_model.v0_n = {1.2};
    recruitment_model.v0_target = {1.2};
    fixture.recruitment_model = std::move(recruitment_model);
    return fixture;
  }

  /**
   * Access the recruiting alternative of a fixture's model block.
   */
  LinearPressureRecruitment& linear_pressure(
      TerminalUnits::Recruitment::RecruitmentModel& recruitment_model)
  {
    return std::get<LinearPressureRecruitment>(recruitment_model);
  }

  Vector<double> make_recruitment_dofs(const double p1, const double p2, const double q = 0.0)
  {
    Core::LinAlg::Map map(-1, 3, 0, MPI_COMM_WORLD);
    Vector<double> dofs(map, true);
    dofs.replace_local_values(
        3, std::array<double, 3>{p1, p2, q}.data(), std::array<int, 3>{0, 1, 2}.data());
    return dofs;
  }

  ReducedLungParameters make_terminal_unit_parameters(const TerminalUnitModelCase& model_case)
  {
    ReducedLungParameters params{};

    params.lung_tree.terminal_units.rheological_model.rheological_model_type =
        Core::IO::InputField<RheologicalModelType>(model_case.rheological_model_type);
    params.lung_tree.terminal_units.rheological_model.kelvin_voigt.viscosity_kelvin_voigt_eta =
        make_elementwise_double_field(model_case.kelvin_voigt_eta);
    params.lung_tree.terminal_units.rheological_model.four_element_maxwell
        .viscosity_kelvin_voigt_eta = make_elementwise_double_field(model_case.kelvin_voigt_eta);
    params.lung_tree.terminal_units.rheological_model.four_element_maxwell.elasticity_maxwell_e_m =
        make_elementwise_double_field(model_case.maxwell_e_m);
    params.lung_tree.terminal_units.rheological_model.four_element_maxwell.viscosity_maxwell_eta_m =
        make_elementwise_double_field(model_case.maxwell_eta_m);

    params.lung_tree.terminal_units.elasticity_model.elasticity_model_type =
        Core::IO::InputField<ElasticityModelType>(model_case.elasticity_model_type);
    params.lung_tree.terminal_units.elasticity_model.linear.elasticity_e =
        make_elementwise_double_field(model_case.linear_elasticity_e);
    params.lung_tree.terminal_units.elasticity_model.ogden.ogden_parameter_kappa =
        make_elementwise_double_field(model_case.ogden_kappa);
    params.lung_tree.terminal_units.elasticity_model.ogden.ogden_parameter_beta =
        make_elementwise_double_field(model_case.ogden_beta);

    params.lung_tree.terminal_units.recruitment_model.pressure_law_type =
        Core::IO::InputField<PressureLawType>(model_case.recruitment_model_type);
    params.lung_tree.terminal_units.recruitment_model.time_law_type =
        Core::IO::InputField<TimeLawType>(model_case.time_law_type);
    params.lung_tree.terminal_units.recruitment_model.linear_pressure.v0_min =
        Core::IO::InputField<double>(1.0);
    params.lung_tree.terminal_units.recruitment_model.linear_pressure.v0_max =
        Core::IO::InputField<double>(2.0);
    params.lung_tree.terminal_units.recruitment_model.linear_pressure.p_closing_min =
        Core::IO::InputField<double>(-0.2);
    params.lung_tree.terminal_units.recruitment_model.linear_pressure.p_opening_min =
        Core::IO::InputField<double>(0.0);
    params.lung_tree.terminal_units.recruitment_model.linear_pressure.delta_p_minmax =
        Core::IO::InputField<double>(1.0);
    params.lung_tree.terminal_units.recruitment_model.linear_pressure.epsilon_v0_switch =
        Core::IO::InputField<double>(0.05);
    params.lung_tree.terminal_units.recruitment_model.linear_pressure.initial_v0 =
        make_elementwise_double_field(model_case.recruitment_initial_v0);
    params.lung_tree.terminal_units.recruitment_model.linear_pressure.initial_path =
        Core::IO::InputField<HysteresisPath>(HysteresisPath::Opening);
    params.lung_tree.terminal_units.recruitment_model.exponential_relaxation.tau =
        make_elementwise_double_field(model_case.recruitment_tau);

    return params;
  }

  class TerminalUnitRegistryAndJacobianTest : public testing::TestWithParam<TerminalUnitModelCase>
  {
  };

  TEST(TerminalUnitRecruitmentTests, ReferenceVolumeContextNoneUsesGeometryVolume)
  {
    auto fixture = make_single_recruitment_data(PressureLawType::None);
    auto dofs = make_recruitment_dofs(0.4, 0.0);

    const auto context = TerminalUnits::Recruitment::evaluate_recruitment_context(
        fixture.data, fixture.recruitment_model, dofs, 0, 0.1);

    EXPECT_DOUBLE_EQ(context.v0_eff, 4.0);
  }

  TEST(TerminalUnitRecruitmentTests, EndOfStepUpdatesLinearPressureStateAndTarget)
  {
    auto fixture = make_single_recruitment_data(PressureLawType::LinearPressure);
    auto dofs = make_recruitment_dofs(0.4, 0.0);

    TerminalUnits::Recruitment::update_recruitment_state(
        fixture.data, fixture.recruitment_model, dofs, 0.1);

    EXPECT_DOUBLE_EQ(linear_pressure(fixture.recruitment_model).v0_target[0], 1.4);
    EXPECT_DOUBLE_EQ(linear_pressure(fixture.recruitment_model).v0_n[0], 1.4);
  }

  TEST(TerminalUnitRecruitmentTests, EndOfStepUpdatesExponentialTimeLawStateAndTarget)
  {
    auto fixture = make_single_recruitment_data(
        PressureLawType::LinearPressure, TimeLawType::ExponentialRelaxation);
    auto dofs = make_recruitment_dofs(0.4, 0.0);
    constexpr double dt = 0.1;
    const double relaxation =
        std::exp(-dt / linear_pressure(fixture.recruitment_model).time_law.tau[0]);

    TerminalUnits::Recruitment::update_recruitment_state(
        fixture.data, fixture.recruitment_model, dofs, dt);

    EXPECT_DOUBLE_EQ(linear_pressure(fixture.recruitment_model).v0_target[0], 1.4);
    EXPECT_NEAR(
        linear_pressure(fixture.recruitment_model).v0_n[0], 1.4 + (1.2 - 1.4) * relaxation, 1e-14);
  }

  TEST(TerminalUnitRecruitmentTests, PathSwitchingOnlyHappensAtEndOfStep)
  {
    auto fixture = make_single_recruitment_data(PressureLawType::LinearPressure);
    auto dofs = make_recruitment_dofs(1.2, 0.0);

    (void)TerminalUnits::Recruitment::evaluate_recruitment_context(
        fixture.data, fixture.recruitment_model, dofs, 0, 0.1);
    EXPECT_EQ(linear_pressure(fixture.recruitment_model).pressure_law.active_path_n[0],
        HysteresisPath::Opening);

    TerminalUnits::Recruitment::update_recruitment_state(
        fixture.data, fixture.recruitment_model, dofs, 0.1);
    EXPECT_EQ(linear_pressure(fixture.recruitment_model).pressure_law.active_path_n[0],
        HysteresisPath::Closing);
  }

  // The hysteresis loop only closes if the closing branch hands back to the opening one, so a
  // fully derecruited element must reopen rather than stay on the closing branch for good.
  TEST(TerminalUnitRecruitmentTests, FullDerecruitmentSwitchesBackToTheOpeningPath)
  {
    auto fixture = make_single_recruitment_data(
        PressureLawType::LinearPressure, TimeLawType::None, HysteresisPath::Closing);
    // Below p_closing_min the closing branch derecruits the element all the way to v0_min.
    auto dofs = make_recruitment_dofs(-0.3, 0.0);

    TerminalUnits::Recruitment::update_recruitment_state(
        fixture.data, fixture.recruitment_model, dofs, 0.1);

    EXPECT_DOUBLE_EQ(linear_pressure(fixture.recruitment_model).v0_n[0], 1.0);
    EXPECT_EQ(linear_pressure(fixture.recruitment_model).pressure_law.active_path_n[0],
        HysteresisPath::Opening);
  }

  TEST(TerminalUnitRecruitmentTests, InitializationUsesGeometryForNoneAndInitialV0ForRecruitable)
  {
    const TerminalUnitModelCase base_case{.name = "init",
        .rheological_model_type = RheologicalModelType::KelvinVoigt,
        .elasticity_model_type = ElasticityModelType::Linear,
        .kelvin_voigt_eta = {0.0, 0.0, 0.0},
        .maxwell_e_m = {1.0, 1.0, 1.0},
        .maxwell_eta_m = {1.0, 1.0, 1.0},
        .linear_elasticity_e = {1.0, 1.0, 1.0},
        .ogden_kappa = {1.0, 1.0, 1.0},
        .ogden_beta = {2.0, 2.0, 2.0}};

    auto none_params = make_terminal_unit_parameters(base_case);
    TerminalUnitContainer none_units;
    TerminalUnits::ModelRegistry::add_terminal_unit_with_model_selection(none_units, 0, 0, 1.0,
        none_params, RheologicalModelType::KelvinVoigt, ElasticityModelType::Linear,
        RecruitmentModelType::None);
    EXPECT_NEAR(none_units.models.front().data.volume_v[0], 4.0 / 3.0 * std::numbers::pi, 1e-14);

    auto recruitment_case = base_case;
    recruitment_case.recruitment_model_type = RecruitmentModelType::LinearPressure;
    recruitment_case.recruitment_initial_v0 = {1.2, 1.3, 1.4};
    auto recruitment_params = make_terminal_unit_parameters(recruitment_case);
    TerminalUnitContainer recruitment_units;
    TerminalUnits::ModelRegistry::add_terminal_unit_with_model_selection(recruitment_units, 0, 0,
        1.0, recruitment_params, RheologicalModelType::KelvinVoigt, ElasticityModelType::Linear,
        RecruitmentModelType::LinearPressure);
    EXPECT_DOUBLE_EQ(recruitment_units.models.front().data.volume_v[0], 1.2);
  }

  // Recruitment is part of the model block key, so a tree that mixes recruiting and
  // non-recruiting terminal units splits into blocks instead of padding one with unused data.
  TEST(TerminalUnitRecruitmentTests, MixedTreeSplitsIntoBlocksByRecruitmentLaw)
  {
    TerminalUnitModelCase model_case{.name = "mixed",
        .rheological_model_type = RheologicalModelType::KelvinVoigt,
        .elasticity_model_type = ElasticityModelType::Linear,
        .kelvin_voigt_eta = {0.0, 0.0, 0.0},
        .maxwell_e_m = {1.0, 1.0, 1.0},
        .maxwell_eta_m = {1.0, 1.0, 1.0},
        .linear_elasticity_e = {1.0, 1.0, 1.0},
        .ogden_kappa = {1.0, 1.0, 1.0},
        .ogden_beta = {2.0, 2.0, 2.0}};
    auto params = make_terminal_unit_parameters(model_case);
    params.lung_tree.terminal_units.recruitment_model.pressure_law_type =
        Core::IO::InputField<PressureLawType>(
            std::unordered_map<int, PressureLawType>{{1, PressureLawType::None},
                {2, PressureLawType::LinearPressure}, {3, PressureLawType::None}});

    TerminalUnitContainer terminal_units;
    for (int global_element_id = 0; global_element_id < 3; ++global_element_id)
    {
      const auto recruitment_model_type =
          params.lung_tree.terminal_units.recruitment_model.pressure_law_type.at(
              global_element_id, "pressure_law_type");
      TerminalUnits::ModelRegistry::add_terminal_unit_with_model_selection(terminal_units,
          global_element_id, global_element_id, 1.0, params, RheologicalModelType::KelvinVoigt,
          ElasticityModelType::Linear, recruitment_model_type);
    }

    ASSERT_EQ(terminal_units.models.size(), 2u);

    const auto& rigid_block = terminal_units.models.front();
    EXPECT_TRUE(std::holds_alternative<NoRecruitment>(rigid_block.recruitment_model));
    EXPECT_EQ(rigid_block.data.number_of_elements(), 2u);
    EXPECT_EQ(rigid_block.data.global_element_id, (std::vector<int>{0, 2}));

    auto& recruiting_block = terminal_units.models.back();
    EXPECT_TRUE(
        std::holds_alternative<LinearPressureRecruitment>(recruiting_block.recruitment_model));
    EXPECT_EQ(recruiting_block.data.number_of_elements(), 1u);
    EXPECT_EQ(linear_pressure(recruiting_block.recruitment_model).v0_n.size(), 1u);
  }

  TEST(TerminalUnitRecruitmentTests, InvalidRecruitmentParametersThrow)
  {
    const TerminalUnitModelCase model_case{.name = "invalid_recruitment",
        .rheological_model_type = RheologicalModelType::KelvinVoigt,
        .elasticity_model_type = ElasticityModelType::Linear,
        .kelvin_voigt_eta = {0.0, 0.0, 0.0},
        .maxwell_e_m = {1.0, 1.0, 1.0},
        .maxwell_eta_m = {1.0, 1.0, 1.0},
        .linear_elasticity_e = {1.0, 1.0, 1.0},
        .ogden_kappa = {1.0, 1.0, 1.0},
        .ogden_beta = {2.0, 2.0, 2.0},
        .recruitment_model_type = RecruitmentModelType::LinearPressure,
        .time_law_type = TimeLawType::ExponentialRelaxation};

    const auto expect_invalid = [&](auto mutate)
    {
      auto params = make_terminal_unit_parameters(model_case);
      mutate(params);
      TerminalUnitContainer units;
      const auto recruitment_model_type =
          params.lung_tree.terminal_units.recruitment_model.pressure_law_type.at(
              0, "pressure_law_type");
      EXPECT_THROW(TerminalUnits::ModelRegistry::add_terminal_unit_with_model_selection(units, 0, 0,
                       1.0, params, RheologicalModelType::KelvinVoigt, ElasticityModelType::Linear,
                       recruitment_model_type),
          Core::Exception);
    };

    expect_invalid(
        [](auto& params)
        {
          params.lung_tree.terminal_units.recruitment_model.linear_pressure.v0_min =
              Core::IO::InputField<double>(0.0);
        });
    expect_invalid(
        [](auto& params)
        {
          params.lung_tree.terminal_units.recruitment_model.linear_pressure.v0_max =
              Core::IO::InputField<double>(0.5);
        });
    expect_invalid(
        [](auto& params)
        {
          params.lung_tree.terminal_units.recruitment_model.linear_pressure.initial_v0 =
              Core::IO::InputField<double>(3.0);
        });
    expect_invalid(
        [](auto& params)
        {
          params.lung_tree.terminal_units.recruitment_model.linear_pressure.delta_p_minmax =
              Core::IO::InputField<double>(0.0);
        });
    expect_invalid(
        [](auto& params)
        {
          params.lung_tree.terminal_units.recruitment_model.linear_pressure.p_opening_min =
              Core::IO::InputField<double>(-0.3);
        });
    expect_invalid(
        [](auto& params)
        {
          params.lung_tree.terminal_units.recruitment_model.linear_pressure.epsilon_v0_switch =
              Core::IO::InputField<double>(0.5);
        });
    expect_invalid(
        [](auto& params)
        {
          params.lung_tree.terminal_units.recruitment_model.exponential_relaxation.tau =
              Core::IO::InputField<double>(0.0);
        });

    expect_invalid(
        [](auto& params)
        {
          params.lung_tree.terminal_units.recruitment_model.pressure_law_type =
              Core::IO::InputField<PressureLawType>(PressureLawType::None);
        });
  }

  TEST(TerminalUnitRecruitmentTests, OutputWrittenOnlyByRecruitingBlocks)
  {
    Core::LinAlg::Map output_map(-1, 1, 0, MPI_COMM_WORLD);

    // A block without a recruitment law owns no reference-volume state and contributes no field;
    // its elements keep the collector's not-applicable marker, like every other inapplicable row.
    auto none_fixture = make_single_recruitment_data(PressureLawType::None);
    RuntimeOutputCollector none_collector(output_map);
    TerminalUnits::Recruitment::make_output_evaluator(none_fixture.recruitment_model)(
        none_fixture.data, none_collector, ReducedLungParameters::OutputVerbosity::high);
    EXPECT_TRUE(none_collector.vectors.empty());

    auto recruitable_fixture = make_single_recruitment_data(PressureLawType::LinearPressure);
    const auto output_evaluator =
        TerminalUnits::Recruitment::make_output_evaluator(recruitable_fixture.recruitment_model);

    RuntimeOutputCollector collector_before_update(output_map);
    output_evaluator(recruitable_fixture.data, collector_before_update,
        ReducedLungParameters::OutputVerbosity::high);
    EXPECT_DOUBLE_EQ(collector_before_update.vectors.at("v_0").local_values_as_span()[0], 1.2);
    EXPECT_DOUBLE_EQ(
        collector_before_update.vectors.at("v0_target").local_values_as_span()[0], 1.2);

    auto dofs = make_recruitment_dofs(0.4, 0.0);
    TerminalUnits::Recruitment::update_recruitment_state(
        recruitable_fixture.data, recruitable_fixture.recruitment_model, dofs, 0.1);
    RuntimeOutputCollector collector_after_update(output_map);
    output_evaluator(recruitable_fixture.data, collector_after_update,
        ReducedLungParameters::OutputVerbosity::high);
    EXPECT_DOUBLE_EQ(collector_after_update.vectors.at("v_0").local_values_as_span()[0], 1.4);
    EXPECT_DOUBLE_EQ(collector_after_update.vectors.at("v0_target").local_values_as_span()[0], 1.4);

    // Below high verbosity nothing is emitted at all.
    RuntimeOutputCollector medium_collector(output_map);
    output_evaluator(
        recruitable_fixture.data, medium_collector, ReducedLungParameters::OutputVerbosity::medium);
    EXPECT_TRUE(medium_collector.vectors.empty());
  }

  // Tests model registration + analytic Jacobian by comparing against FD residual derivatives.
  TEST_P(TerminalUnitRegistryAndJacobianTest, JacobianVsFiniteDifference)
  {
    const auto& model_case = GetParam();
    SCOPED_TRACE(model_case.name);

    const auto params = make_terminal_unit_parameters(model_case);
    TerminalUnitContainer terminal_units;

    constexpr std::array element_lengths{1.0, 2.0, 3.0};
    for (int global_element_id = 0; global_element_id < 3; ++global_element_id)
    {
      TerminalUnits::ModelRegistry::add_terminal_unit_with_model_selection(terminal_units,
          global_element_id, global_element_id, element_lengths[global_element_id], params,
          model_case.rheological_model_type, model_case.elasticity_model_type,
          model_case.recruitment_model_type);
    }
    ASSERT_EQ(terminal_units.models.size(), 1u);

    auto& model = terminal_units.models.front();

    if (model_case.rheological_model_type == RheologicalModelType::KelvinVoigt)
    {
      EXPECT_TRUE(std::holds_alternative<KelvinVoigt>(model.rheological_model));
    }
    else
    {
      EXPECT_TRUE(std::holds_alternative<FourElementMaxwell>(model.rheological_model));
    }

    if (model_case.elasticity_model_type == ElasticityModelType::Linear)
    {
      EXPECT_TRUE(std::holds_alternative<LinearElasticity>(model.elasticity_model));
    }
    else
    {
      EXPECT_TRUE(std::holds_alternative<OgdenHyperelasticity>(model.elasticity_model));
    }

    Airways::AirwayContainer airways;
    const std::map<int, int> global_dof_per_ele = {{0, 3}, {1, 3}, {2, 3}};
    std::map<int, int> first_global_dof_of_ele = {{0, 0}, {1, 3}, {2, 6}};

    assign_global_dof_ids_to_models(first_global_dof_of_ele, airways, terminal_units);

    int n_local_equations = 0;
    TerminalUnits::assign_local_equation_ids(terminal_units, n_local_equations);

    const auto dof_map = create_domain_map(MPI_COMM_WORLD, airways, terminal_units);
    const auto row_map = create_row_map(MPI_COMM_WORLD, airways, terminal_units, {}, {}, {});
    const auto col_map = create_column_map(MPI_COMM_WORLD, airways, terminal_units,
        global_dof_per_ele, first_global_dof_of_ele, {}, {}, {});
    TerminalUnits::assign_local_dof_ids(col_map, terminal_units);

    // Artificial dof vector
    Vector<double> dofs(dof_map, true);
    Vector<double> locally_relevant_dofs(col_map, true);
    const std::array<double, 9> dof_values{model_case.dofs[0], model_case.dofs[1],
        model_case.dofs[2], model_case.dofs[0], model_case.dofs[1], model_case.dofs[2],
        model_case.dofs[0], model_case.dofs[1], model_case.dofs[2]};
    dofs.replace_local_values(
        9, dof_values.data(), std::array<int, 9>{0, 1, 2, 3, 4, 5, 6, 7, 8}.data());
    export_to(dofs, locally_relevant_dofs);

    double dt = 1e-1;         // Dummy time step size
    const double eps = 1e-6;  // Perturbation parameter for the FD approximation

    TerminalUnits::create_evaluators(terminal_units);

    SparseMatrix jac(row_map, col_map, 3);
    // Assembly consumes the cached reference volume, so the model state must be synchronized with
    // the dof vector first -- the solver does this via the state updaters on every dof change.
    model.internal_state_updater(model.data, locally_relevant_dofs, dt);
    model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);

    check_jacobian_column_against_fd(
        model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
    check_jacobian_column_against_fd(
        model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
    check_jacobian_column_against_fd(
        model.data.lid_q, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);

    if (model_case.check_completed_jacobian_update)
    {
      jac.complete();  // Sparsity pattern already filled the first time
      model.jacobian_evaluator(model.data, jac, locally_relevant_dofs, dt);
      check_jacobian_column_against_fd(
          model.data.lid_p1, 0, model, jac, locally_relevant_dofs, dt, eps, row_map);
      check_jacobian_column_against_fd(
          model.data.lid_p2, 1, model, jac, locally_relevant_dofs, dt, eps, row_map);
      check_jacobian_column_against_fd(
          model.data.lid_q, 2, model, jac, locally_relevant_dofs, dt, eps, row_map);
    }
  }

  TEST(TerminalUnitRegistryTest, ThrowsOnUnknownRheologyType)
  {
    const auto model_case = TerminalUnitModelCase{
        .name = "invalid_rheology",
        .rheological_model_type = RheologicalModelType::KelvinVoigt,
        .elasticity_model_type = ElasticityModelType::Linear,
        .kelvin_voigt_eta = {0.0, 1.0, 100.0},
        .maxwell_e_m = {10.0, 0.0, 20.0},
        .maxwell_eta_m = {2.5, 10.0, 0.0},
        .linear_elasticity_e = {1.0, 1.0, 0.0},
        .ogden_kappa = {1.0, 1.0, 1.0},
        .ogden_beta = {5.0, -0.4, -8.0},
    };
    const auto params = make_terminal_unit_parameters(model_case);

    TerminalUnitContainer terminal_units;
    EXPECT_THROW(TerminalUnits::ModelRegistry::add_terminal_unit_with_model_selection(
                     terminal_units, 0, 0, 1.0, params, static_cast<RheologicalModelType>(-1),
                     ElasticityModelType::Linear, RecruitmentModelType::None),
        Core::Exception);
  }

  INSTANTIATE_TEST_SUITE_P(TerminalUnitModelPairs, TerminalUnitRegistryAndJacobianTest,
      testing::Values(TerminalUnitModelCase{.name = "KelvinVoigt_Linear",
                          .rheological_model_type = RheologicalModelType::KelvinVoigt,
                          .elasticity_model_type = ElasticityModelType::Linear,
                          .kelvin_voigt_eta = {0.0, 1.0, 100.0},
                          .maxwell_e_m = {10.0, 0.0, 20.0},
                          .maxwell_eta_m = {2.5, 10.0, 0.0},
                          .linear_elasticity_e = {1.0, 1.0, 0.0},
                          .ogden_kappa = {1.0, 1.0, 1.0},
                          .ogden_beta = {5.0, -0.4, -8.0}},
          TerminalUnitModelCase{.name = "KelvinVoigt_Ogden",
              .rheological_model_type = RheologicalModelType::KelvinVoigt,
              .elasticity_model_type = ElasticityModelType::Ogden,
              .kelvin_voigt_eta = {0.0, 1.0, 100.0},
              .maxwell_e_m = {10.0, 0.0, 20.0},
              .maxwell_eta_m = {2.5, 10.0, 0.0},
              .linear_elasticity_e = {1.0, 1.0, 0.0},
              .ogden_kappa = {1.0, 1.0, 1.0},
              .ogden_beta = {5.0, -0.4, -8.0}},
          TerminalUnitModelCase{.name = "FourElementMaxwell_Linear",
              .rheological_model_type = RheologicalModelType::FourElementMaxwell,
              .elasticity_model_type = ElasticityModelType::Linear,
              .kelvin_voigt_eta = {0.0, 1.0, 100.0},
              .maxwell_e_m = {10.0, 0.0, 20.0},
              .maxwell_eta_m = {2.5, 10.0, 0.0},
              .linear_elasticity_e = {1.0, 1.0, 0.0},
              .ogden_kappa = {1.0, 1.0, 1.0},
              .ogden_beta = {5.0, -0.4, -8.0},
              .check_completed_jacobian_update = true},
          TerminalUnitModelCase{.name = "FourElementMaxwell_Ogden",
              .rheological_model_type = RheologicalModelType::FourElementMaxwell,
              .elasticity_model_type = ElasticityModelType::Ogden,
              .kelvin_voigt_eta = {10.5, 1.0, 100.0},
              .maxwell_e_m = {10.0, 0.0, 20.0},
              .maxwell_eta_m = {2.5, 10.0, 0.0},
              .linear_elasticity_e = {1.0, 1.0, 0.0},
              .ogden_kappa = {0.0, 1.0, 1.0},
              .ogden_beta = {1.0, 6.4, -3.0}},
          TerminalUnitModelCase{.name = "KelvinVoigt_Linear_LinearPressure",
              .rheological_model_type = RheologicalModelType::KelvinVoigt,
              .elasticity_model_type = ElasticityModelType::Linear,
              .kelvin_voigt_eta = {0.5, 1.0, 1.5},
              .maxwell_e_m = {10.0, 10.0, 10.0},
              .maxwell_eta_m = {2.5, 2.5, 2.5},
              .linear_elasticity_e = {1.0, 1.2, 1.4},
              .ogden_kappa = {1.0, 1.0, 1.0},
              .ogden_beta = {2.0, 2.0, 2.0},
              .recruitment_model_type = RecruitmentModelType::LinearPressure,
              .dofs = {0.55, 0.15, 0.2}},
          TerminalUnitModelCase{.name = "KelvinVoigt_Ogden_LinearPressure",
              .rheological_model_type = RheologicalModelType::KelvinVoigt,
              .elasticity_model_type = ElasticityModelType::Ogden,
              .kelvin_voigt_eta = {0.5, 1.0, 1.5},
              .maxwell_e_m = {10.0, 10.0, 10.0},
              .maxwell_eta_m = {2.5, 2.5, 2.5},
              .linear_elasticity_e = {1.0, 1.2, 1.4},
              .ogden_kappa = {1.1, 1.2, 1.3},
              .ogden_beta = {2.0, 2.5, 3.0},
              .recruitment_model_type = RecruitmentModelType::LinearPressure,
              .dofs = {0.55, 0.15, 0.2}},
          TerminalUnitModelCase{.name = "KelvinVoigt_Linear_LinearPressure_Exponential",
              .rheological_model_type = RheologicalModelType::KelvinVoigt,
              .elasticity_model_type = ElasticityModelType::Linear,
              .kelvin_voigt_eta = {0.5, 1.0, 1.5},
              .maxwell_e_m = {10.0, 10.0, 10.0},
              .maxwell_eta_m = {2.5, 2.5, 2.5},
              .linear_elasticity_e = {1.0, 1.2, 1.4},
              .ogden_kappa = {1.0, 1.0, 1.0},
              .ogden_beta = {2.0, 2.0, 2.0},
              .recruitment_model_type = RecruitmentModelType::LinearPressure,
              .time_law_type = TimeLawType::ExponentialRelaxation,
              .dofs = {0.55, 0.15, 0.2}},
          TerminalUnitModelCase{.name = "FourElementMaxwell_Linear_LinearPressure",
              .rheological_model_type = RheologicalModelType::FourElementMaxwell,
              .elasticity_model_type = ElasticityModelType::Linear,
              .kelvin_voigt_eta = {0.5, 1.0, 1.5},
              .maxwell_e_m = {2.0, 2.5, 3.0},
              .maxwell_eta_m = {1.5, 2.0, 2.5},
              .linear_elasticity_e = {1.0, 1.2, 1.4},
              .ogden_kappa = {1.0, 1.0, 1.0},
              .ogden_beta = {2.0, 2.0, 2.0},
              .recruitment_model_type = RecruitmentModelType::LinearPressure,
              .dofs = {0.55, 0.15, 0.2},
              .check_completed_jacobian_update = true}),
      [](const testing::TestParamInfo<TerminalUnitModelCase>& info) { return info.param.name; });
}  // namespace
