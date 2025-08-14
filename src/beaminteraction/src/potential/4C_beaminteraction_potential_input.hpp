// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_POTENTIAL_INPUT_HPP
#define FOUR_C_BEAMINTERACTION_POTENTIAL_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"
#include "4C_io_visualization_parameters.hpp"

#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::Conditions
{
  class ConditionDefinition;
}  // namespace Core::Conditions

namespace BeamInteraction::Potential
{
  enum class Type
  {
    surface,
    volume,
  };

  enum class Strategy
  {
    double_length_specific_large_separations,  ///< Section-Section Interaction Potential(SSIP)
                                               ///< approach
    double_length_specific_small_separations,  ///< Section-Section Interaction Potential(SSIP)
                                               ///< approach
    single_length_specific_small_separations,  ///< Section-Beam Interaction Potential(SBIP)
                                               ///< approach
    single_length_specific_small_separations_simple,  ///< Section-Beam Interaction Potential(SBIP)
                                                      ///< approach
  };

  enum class MasterSlaveChoice
  {
    smaller_eleGID_is_slave,
    higher_eleGID_is_slave,
  };


  enum class RegularizationType
  {
    linear,
    constant,
    none
  };

  struct BeamPotentialRegularizationParameters
  {
    RegularizationType type{RegularizationType::none};
    double separation{};
  };

  struct BeamPotentialVisualizationParameters
  {
    Core::IO::VisualizationParameters
        visualization_parameters;  ///< global visualization parameters
    std::optional<int> output_interval{};
    bool write_every_iteration{};
    bool write_forces{};
    bool write_moments{};
    bool write_forces_moments_per_pair{};
    bool write_uids{};
  };

  /// Beam potential parameters
  struct BeamPotentialParameters
  {
    std::vector<double> potential_law_exponents{};
    std::vector<double> potential_law_prefactors{};
    Type type{};
    Strategy strategy{};
    std::optional<double> cutoff_radius{};
    BeamPotentialRegularizationParameters regularization{};
    int n_integration_segments{};
    int n_gauss_points{};
    bool automatic_differentiation{};
    MasterSlaveChoice choice_master_slave{};
    std::optional<double> potential_reduction_length{};
    BeamPotentialVisualizationParameters runtime_output_params{};

    // data container for prior element lengths for potential reduction strategy
    // first entry is left prior length and second entry is right prior length
    // this is stored in the beam potential params for easy access during evaluation
    std::unordered_map<int, std::pair<double, double>> ele_gid_prior_length_map;
  };


  void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list);

  void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

  void initialize_validate_beam_potential_params(
      BeamPotentialParameters& beam_potential_params, const double restart_time);

}  // namespace BeamInteraction::Potential

FOUR_C_NAMESPACE_CLOSE

#endif
