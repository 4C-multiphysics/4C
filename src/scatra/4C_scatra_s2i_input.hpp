// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_S2I_INPUT_HPP
#define FOUR_C_SCATRA_S2I_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Conditions
{
  class ConditionDefinition;
}
namespace S2I
{
  //! type of interface side
  enum InterfaceSides
  {
    side_undefined,
    side_source,
    side_master
  };

  //! type of mesh coupling
  enum CouplingType
  {
    coupling_undefined,
    coupling_matching_nodes,
    coupling_mortar_standard,
    coupling_mortar_saddlepoint_petrov,
    coupling_mortar_saddlepoint_bubnov,
    coupling_mortar_condensed_petrov,
    coupling_mortar_condensed_bubnov,
    coupling_nts_standard
  };

  //! type of interface layer growth evaluation
  enum GrowthEvaluation
  {
    growth_evaluation_none,
    growth_evaluation_monolithic,
    growth_evaluation_semi_implicit
  };

  //! models for interface layer growth kinetics
  enum GrowthKineticModels
  {
    growth_kinetics_butlervolmer
  };

  //! models for interface kinetics
  enum KineticModels
  {
    kinetics_constperm,
    kinetics_linearperm,
    kinetics_butlervolmer,
    kinetics_butlervolmerlinearized,
    kinetics_butlervolmerpeltier,
    kinetics_butlervolmerreduced,
    kinetics_butlervolmerreducedcapacitance,
    kinetics_butlervolmerreducedlinearized,
    kinetics_butlervolmerresistance,
    kinetics_butlervolmerreducedresistance,
    kinetics_butlervolmerreducedthermoresistance,
    kinetics_constantinterfaceresistance,
    kinetics_nointerfaceflux
  };

  //! actions for mortar cell evaluation
  enum EvaluationActions
  {
    evaluate_condition,
    evaluate_condition_nts,
    evaluate_condition_od,
    evaluate_mortar_matrices,
    evaluate_nodal_area_fractions
  };

  //! regularization types for plating reaction
  enum RegularizationType
  {
    regularization_undefined,
    regularization_none,
    regularization_hein,
    regularization_polynomial,
    regularization_trigonometrical
  };

  //! valid parameters for scatra-scatra interface coupling
  std::vector<Core::IO::InputSpec> valid_parameters();

  //! set valid conditions for scatra-scatra interface coupling
  void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);
}  // namespace S2I

FOUR_C_NAMESPACE_CLOSE

#endif
