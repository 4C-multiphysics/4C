// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_plasticity.hpp"

#include "4C_inpar_structure.hpp"
#include "4C_inpar_tsi.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::Plasticity::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  /*----------------------------------------------------------------------*/
  /* parameters for semi-smooth Newton plasticity algorithm */
  Core::Utils::SectionSpecs iplast{"SEMI-SMOOTH PLASTICITY"};

  iplast.specs.emplace_back(parameter<double>("SEMI_SMOOTH_CPL",
      {.description = "Weighting factor cpl for semi-smooth PDASS", .default_value = 1.0}));
  iplast.specs.emplace_back(parameter<double>("STABILIZATION_S",
      {.description = "Stabilization factor s for semi-smooth PDASS", .default_value = 1.0}));

  // solver convergence test parameters for semi-smooth plasticity formulation
  Core::Utils::string_to_integral_parameter<Inpar::Solid::BinaryOp>("NORMCOMBI_RESFPLASTCONSTR",
      "And", "binary operator to combine plasticity constraints and residual force values",
      tuple<std::string>("And", "Or"),
      tuple<Inpar::Solid::BinaryOp>(Inpar::Solid::bop_and, Inpar::Solid::bop_or), iplast);

  Core::Utils::string_to_integral_parameter<Inpar::Solid::BinaryOp>("NORMCOMBI_DISPPLASTINCR",
      "And",
      "binary operator to combine displacement increments and plastic flow (Delta Lp) increment "
      "values",
      tuple<std::string>("And", "Or"),
      tuple<Inpar::Solid::BinaryOp>(Inpar::Solid::bop_and, Inpar::Solid::bop_or), iplast);

  iplast.specs.emplace_back(parameter<double>("TOLPLASTCONSTR",
      {.description = "tolerance in the plastic constraint norm for the newton iteration",
          .default_value = 1.0E-8}));
  iplast.specs.emplace_back(parameter<double>("TOLDELTALP",
      {.description = "tolerance in the plastic flow (Delta Lp) norm for the Newton iteration",
          .default_value = 1.0E-8}));

  Core::Utils::string_to_integral_parameter<Inpar::Solid::BinaryOp>("NORMCOMBI_EASRES", "And",
      "binary operator to combine EAS-residual and residual force values",
      tuple<std::string>("And", "Or"),
      tuple<Inpar::Solid::BinaryOp>(Inpar::Solid::bop_and, Inpar::Solid::bop_or), iplast);

  Core::Utils::string_to_integral_parameter<Inpar::Solid::BinaryOp>("NORMCOMBI_EASINCR", "And",
      "binary operator to combine displacement increments and EAS increment values",
      tuple<std::string>("And", "Or"),
      tuple<Inpar::Solid::BinaryOp>(Inpar::Solid::bop_and, Inpar::Solid::bop_or), iplast);

  iplast.specs.emplace_back(parameter<double>(
      "TOLEASRES", {.description = "tolerance in the EAS residual norm for the newton iteration",
                       .default_value = 1.0E-8}));
  iplast.specs.emplace_back(parameter<double>(
      "TOLEASINCR", {.description = "tolerance in the EAS increment norm for the Newton iteration",
                        .default_value = 1.0E-8}));

  Core::Utils::string_to_integral_parameter<Inpar::TSI::DissipationMode>("DISSIPATION_MODE",
      "pl_multiplier", "method to calculate the plastic dissipation",
      tuple<std::string>("pl_multiplier", "pl_flow", "Taylor_Quinney"),
      tuple<Inpar::TSI::DissipationMode>(
          Inpar::TSI::pl_multiplier, Inpar::TSI::pl_flow, Inpar::TSI::Taylor_Quinney),
      iplast);

  iplast.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
