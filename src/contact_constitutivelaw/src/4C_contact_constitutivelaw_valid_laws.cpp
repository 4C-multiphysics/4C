// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_constitutivelaw_valid_laws.hpp"

#include "4C_contact_constitutivelaw_bundle.hpp"
#include "4C_contact_constitutivelaw_contactconstitutivelaw_parameter.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_spec_builders.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::InputSpec CONTACT::CONSTITUTIVELAW::valid_contact_constitutive_laws()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> specs;
  std::vector<CONTACT::CONSTITUTIVELAW::ConstitutiveLawType> group_index_to_type;

  /*----------------------------------------------------------------------*/
  // broken rational function
  {
    specs.emplace_back(group("CoConstLaw_brokenrational",
        {
            parameter<double>("A", {.description = "scaling factor"}),
            parameter<double>("B", {.description = "asymptote"}),
            parameter<double>("C", {.description = "y intercept"}),
            parameter<double>(
                "Offset", {.description = "offset for contact to start", .default_value = 0.0}),
        },
        {
            .description = "Broken rational law",
        }));
    group_index_to_type.push_back(
        CONTACT::CONSTITUTIVELAW::ConstitutiveLawType::colaw_brokenrational);
  }

  // Python surrogate law function
  {
    specs.emplace_back(group("CoConstLaw_python_surrogate",
        {parameter<std::filesystem::path>("Python_Filename",
             {.description = "Path to the Python surrogate model file / script"}),
            parameter<double>(
                "Offset", {.description = "offset for contact to start", .default_value = 0.0})},
        {
            .description = "Python surrogate model",
        }));
    group_index_to_type.push_back(
        CONTACT::CONSTITUTIVELAW::ConstitutiveLawType::colaw_python_surrogate);
  }

  // power law function
  {
    specs.emplace_back(group("CoConstLaw_power",
        {
            parameter<double>("A", {.description = "scaling factor"}),
            parameter<double>("B", {.description = "power coefficient"}),
            parameter<double>(
                "Offset", {.description = "offset for contact to start", .default_value = 0.0}),
        },
        {
            .description = "Power law",
        }));
    group_index_to_type.push_back(CONTACT::CONSTITUTIVELAW::ConstitutiveLawType::colaw_power);
  }

  // cubic function
  {
    specs.emplace_back(group("CoConstLaw_cubic",
        {
            parameter<double>("A", {.description = "A"}),
            parameter<double>("B", {.description = "B"}),
            parameter<double>("C", {.description = "C"}),
            parameter<double>("D", {.description = "D"}),
            parameter<double>(
                "Offset", {.description = "offset for contact to start", .default_value = 0.0}),
        },
        {
            .description = "Cubic function",
        }));
    group_index_to_type.push_back(CONTACT::CONSTITUTIVELAW::ConstitutiveLawType::colaw_cubic);
  }

  // linear function
  {
    specs.emplace_back(group("CoConstLaw_linear",
        {
            parameter<double>("A", {.description = "slope"}),
            parameter<double>("B", {.description = "y intercept"}),
            parameter<double>(
                "Offset", {.description = "offset for contact to start", .default_value = 0.0}),
        },
        {
            .description = "Linear function",
        }));
    group_index_to_type.push_back(CONTACT::CONSTITUTIVELAW::ConstitutiveLawType::colaw_linear);
  }

  // mirco function
  {
    specs.emplace_back(group("CoConstLaw_mirco",
        {
            parameter<int>("FirstMatID", {.description = "First material ID"}),
            parameter<int>("SecondMatID", {.description = "Second material ID"}),
            parameter<double>(
                "LateralLength", {.description = "length of lateral side of the BEM patch"}),
            parameter<int>("Resolution", {.description = "resolution of the surface"}),
            parameter<bool>("PressureGreenFunFlag",
                {.description = "Use pressure-based Green function instead of a point-force-based",
                    .default_value = true}),
            parameter<int>("InitialTopologyStdDeviationFunct",
                {.description = "Function id for Initial Standard deviation for the "
                                "random-midpoint generator",
                    .default_value = 100}),
            parameter<int>(
                "HurstExponentFunct", {.description = "Function for Hurst exponent of the surface",
                                          .default_value = 100}),
            parameter<bool>("RandomTopologyFlag",
                {.description = "Use random midpoint generator flag", .default_value = true}),
            parameter<bool>(
                "RandomSeedFlag", {.description = "Random seed flag", .default_value = false}),
            parameter<int>("RandomGeneratorSeed",
                {.description = "Use random seed to reproduce results", .default_value = 95}),
            parameter<double>("Tolerance",
                {.description = "Tolerance for the convergence of force", .default_value = 0.01}),
            parameter<int>("MaxIteration",
                {.description = "Maximum iteration of NNLS", .default_value = 1000}),
            parameter<bool>("WarmStartingFlag",
                {.description = "Warm-starting flag, solution accelerator", .default_value = true}),
            parameter<double>(
                "Offset", {.description = "offset for contact to start", .default_value = 0.0}),
            parameter<double>("FiniteDifferenceFraction",
                {.description = "Fraction of perturbation difference compared to the actual gap",
                    .default_value = 0.001}),
            parameter<double>("ActiveGapTolerance",
                {.description = "Minimum gap to consider a node as active", .default_value = 1e-6}),
            parameter<std::string>("TopologyFilePath",
                {.description = "Path to file with micro-topology data", .default_value = ""}),
        },
        {
            .description = "Mirco function",
        }));
    group_index_to_type.push_back(CONTACT::CONSTITUTIVELAW::ConstitutiveLawType::colaw_mirco);
  }

  auto valid_law = all_of({
      parameter<int>("LAW"),
      one_of(specs, store_index_as("LAW_TYPE", group_index_to_type)),
  });

  return valid_law;
}

void CONTACT::CONSTITUTIVELAW::create_contact_constitutive_law_from_input(
    const Core::IO::InputParameterContainer& container)
{
  const auto id = container.get<int>("LAW");

  Global::Problem::instance()->contact_constitutive_laws()->insert(id, container);
}

FOUR_C_NAMESPACE_CLOSE
