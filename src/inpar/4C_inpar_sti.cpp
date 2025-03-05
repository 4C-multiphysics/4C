// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_sti.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*------------------------------------------------------------------------*
 | set valid parameters for scatra-thermo interaction          fang 10/16 |
 *------------------------------------------------------------------------*/
void Inpar::STI::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs stidyn{"STI DYNAMIC"};

  // type of scalar transport time integration
  Core::Utils::string_to_integral_parameter<ScaTraTimIntType>("SCATRATIMINTTYPE", "Standard",
      "scalar transport time integration type is needed to instantiate correct scalar transport "
      "time integration scheme for scatra-thermo interaction problems",
      tuple<std::string>("Standard", "Elch"),
      tuple<ScaTraTimIntType>(ScaTraTimIntType::standard, ScaTraTimIntType::elch), stidyn);

  // type of coupling between scatra and thermo fields
  Core::Utils::string_to_integral_parameter<CouplingType>("COUPLINGTYPE", "Undefined",
      "type of coupling between scatra and thermo fields",
      tuple<std::string>("Undefined", "Monolithic", "OneWay_ScatraToThermo",
          "OneWay_ThermoToScatra", "TwoWay_ScatraToThermo", "TwoWay_ScatraToThermo_Aitken",
          "TwoWay_ScatraToThermo_Aitken_Dofsplit", "TwoWay_ThermoToScatra",
          "TwoWay_ThermoToScatra_Aitken"),
      tuple<CouplingType>(CouplingType::undefined, CouplingType::monolithic,
          CouplingType::oneway_scatratothermo, CouplingType::oneway_thermotoscatra,
          CouplingType::twoway_scatratothermo, CouplingType::twoway_scatratothermo_aitken,
          CouplingType::twoway_scatratothermo_aitken_dofsplit, CouplingType::twoway_thermotoscatra,
          CouplingType::twoway_thermotoscatra_aitken),
      stidyn);

  // specification of initial temperature field
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::InitialField>("THERMO_INITIALFIELD",
      "zero_field", "initial temperature field for scatra-thermo interaction problems",
      tuple<std::string>("zero_field", "field_by_function", "field_by_condition"),
      tuple<Inpar::ScaTra::InitialField>(Inpar::ScaTra::initfield_zero_field,
          Inpar::ScaTra::initfield_field_by_function, Inpar::ScaTra::initfield_field_by_condition),
      stidyn);

  // function number for initial temperature field
  Core::Utils::int_parameter("THERMO_INITFUNCNO", -1,
      "function number for initial temperature field for scatra-thermo interaction problems",
      stidyn);

  // ID of linear solver for temperature field
  Core::Utils::int_parameter(
      "THERMO_LINEAR_SOLVER", -1, "ID of linear solver for temperature field", stidyn);

  // flag for double condensation of linear equations associated with temperature field
  stidyn.specs.emplace_back(parameter<bool>("THERMO_CONDENSATION",
      {.description =
              "flag for double condensation of linear equations associated with temperature field",
          .default_value = false}));

  stidyn.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  // valid parameters for monolithic scatra-thermo interaction
  Core::Utils::SectionSpecs stidyn_monolithic{stidyn, "MONOLITHIC"};

  // ID of linear solver for global system of equations
  Core::Utils::int_parameter(
      "LINEAR_SOLVER", -1, "ID of linear solver for global system of equations", stidyn_monolithic);

  // type of global system matrix in global system of equations
  Core::Utils::string_to_integral_parameter<Core::LinAlg::MatrixType>("MATRIXTYPE", "block",
      "type of global system matrix in global system of equations",
      tuple<std::string>("block", "sparse"),
      tuple<Core::LinAlg::MatrixType>(
          Core::LinAlg::MatrixType::block_condition, Core::LinAlg::MatrixType::sparse),
      stidyn_monolithic);

  stidyn_monolithic.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  // valid parameters for partitioned scatra-thermo interaction
  Core::Utils::SectionSpecs stidyn_partitioned{stidyn, "PARTITIONED"};

  // relaxation parameter
  stidyn_partitioned.specs.emplace_back(
      parameter<double>("OMEGA", {.description = "relaxation parameter", .default_value = 1.}));

  // maximum value of Aitken relaxation parameter
  stidyn_partitioned.specs.emplace_back(parameter<double>("OMEGAMAX",
      {.description = "maximum value of Aitken relaxation parameter (0.0 = no constraint)",
          .default_value = 0.}));

  stidyn_partitioned.move_into_collection(list);

  return;
}


/*------------------------------------------------------------------------*
 | set valid conditions for scatra-thermo interaction          fang 10/16 |
 *------------------------------------------------------------------------*/
void Inpar::STI::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  return;
}

FOUR_C_NAMESPACE_CLOSE
