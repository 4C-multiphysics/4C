// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_INPUT_HPP
#define FOUR_C_REDUCED_LUNG_INPUT_HPP

#include "4C_config.hpp"

#include "4C_io_input_field.hpp"
#include "4C_io_input_spec.hpp"


FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  struct ReducedLungParameters
  {
    struct Dynamics
    {
      double time_increment;
      int number_of_steps;
      int restart_every = -1;
      int results_every = -1;
      int linear_solver;
    } dynamics;
    struct LungTree
    {
      /**
       * Enum to distinguish between airway and terminal unit elements in the reduced
       * lung implementation.
       */
      enum class ElementType : std::uint8_t
      {
        Airway,
        TerminalUnit,
      };

      Core::IO::InputField<ElementType> element_type;
      Core::IO::InputField<int> generation;

      struct Airways
      {
        Core::IO::InputField<double> radius;
        struct FlowModel
        {
          /**
           * Enum to distinguish between different resistance models in the reduced lung
           * implementation.
           */
          enum class ResistanceType : std::uint8_t
          {
            Linear,
            NonLinear
          };
          Core::IO::InputField<ResistanceType> resistance_type;

          struct ResistanceModel
          {
            struct NonLinear
            {
              Core::IO::InputField<double> turbulence_factor_gamma;
            } non_linear;
          } resistance_model;

          Core::IO::InputField<bool> include_inertia;
        } flow_model;

        /**
         * Enum to distinguish between different airway wall models.
         */
        enum class WallModelType : std::uint8_t
        {
          Rigid,
          KelvinVoigt
        };

        Core::IO::InputField<WallModelType> wall_model_type;
        struct WallModel
        {
          struct KelvinVoigt
          {
            struct Elasticity
            {
              Core::IO::InputField<double> wall_poisson_ratio;
              Core::IO::InputField<double> wall_elasticity;
              Core::IO::InputField<double> wall_thickness;
            } elasticity;
            struct Viscosity
            {
              Core::IO::InputField<double> viscous_time_constant;
              Core::IO::InputField<double> viscous_phase_shift;
            } viscosity;
          } kelvin_voigt;
        } wall_model;
      } airways;

      struct TerminalUnits
      {
        struct RheologicalModel
        {
          /**
           * Enum to distinguish between different rheological models for the terminal units in
           * the reduced lung implementation.
           */
          enum class RheologicalModelType : std::uint8_t
          {
            KelvinVoigt,
            FourElementMaxwell,
          };

          Core::IO::InputField<RheologicalModelType> rheological_model_type;

          struct KelvinVoigt
          {
            Core::IO::InputField<double> viscosity_kelvin_voigt_eta;
          } kelvin_voigt;

          struct FourElementMaxwell
          {
            Core::IO::InputField<double> viscosity_kelvin_voigt_eta;
            Core::IO::InputField<double> viscosity_maxwell_eta_m;
            Core::IO::InputField<double> elasticity_maxwell_e_m;
          } four_element_maxwell;
        } rheological_model;

        struct ElasticityModel
        {
          /**
           * Enum to distinguish between different elasticity models for the terminal units in the
           * reduced lung implementation.
           */
          enum class ElasticityModelType : std::uint8_t
          {
            Linear,
            Ogden
          };

          Core::IO::InputField<ElasticityModelType> elasticity_model_type;

          struct Linear
          {
            Core::IO::InputField<double> elasticity_e;
          } linear;

          struct Ogden
          {
            Core::IO::InputField<double> ogden_parameter_kappa;
            Core::IO::InputField<double> ogden_parameter_beta;
          } ogden;
        } elasticity_model;
      } terminal_units;
    } lung_tree;
    struct AirProperties
    {
      double density;
      double dynamic_viscosity;
    } air_properties;
  };
  /// reduced airways parameters
  Core::IO::InputSpec valid_parameters();

}  // namespace ReducedLung

FOUR_C_NAMESPACE_CLOSE

#endif