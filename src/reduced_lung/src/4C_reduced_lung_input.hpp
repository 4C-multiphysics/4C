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

#include <cstdint>
#include <filesystem>
#include <vector>


FOUR_C_NAMESPACE_OPEN

namespace ReducedLung
{
  struct ReducedLungParameters
  {
    enum class OutputVerbosity : std::uint8_t
    {
      //  write only core fields (p_1, p_2, q_in, q_out)
      minimal,
      //  include minimal + advanced model outputs (currently e.g. area, volume where available).
      medium,
      // include medium + model-internal diagnostic quantities (e.g. flow_k_turb, elastic_pressure,
      // maxwell_pressure).
      high
    };

    struct Dynamics
    {
      double time_increment;
      int number_of_steps;
      int restart_every = -1;
      int results_every = -1;
      int linear_solver;
      int max_nonlinear_iterations;
      double nonlinear_residual_tolerance;
      double nonlinear_increment_tolerance;
      OutputVerbosity output_verbosity = OutputVerbosity::minimal;
    } dynamics;
    /**
     * The geometry of the lung tree is read from a VTU mesh file. The mesh provides the
     * nodes and the line2 cells forming the tree, and it is also the source of all input fields
     * that are specified as `from_mesh` in the input file.
     */
    struct Geometry
    {
      std::filesystem::path file;
    } geometry;

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

      struct Airways
      {
        std::vector<int> element_blocks;
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
        std::vector<int> element_blocks;

        struct RecruitmentModel
        {
          enum class PressureLawType : std::uint8_t
          {
            None,
            LinearPressure,
          };

          enum class TimeLawType : std::uint8_t
          {
            None,
            ExponentialRelaxation,
          };

          enum class HysteresisPath : std::uint8_t
          {
            Opening,
            Closing,
          };

          enum class ReferenceVolumeLinearization : std::uint8_t
          {
            Frozen,
            Coupled,
          };

          Core::IO::InputField<PressureLawType> pressure_law_type{PressureLawType::None};
          Core::IO::InputField<TimeLawType> time_law_type{TimeLawType::None};
          Core::IO::InputField<ReferenceVolumeLinearization> reference_volume_linearization{
              ReferenceVolumeLinearization::Frozen};

          struct LinearPressure
          {
            Core::IO::InputField<double> v0_min;
            Core::IO::InputField<double> v0_max;
            Core::IO::InputField<double> p_closing_min;
            Core::IO::InputField<double> p_opening_min;
            Core::IO::InputField<double> delta_p_minmax;
            Core::IO::InputField<double> epsilon_v0_switch;
            Core::IO::InputField<double> initial_v0;
            Core::IO::InputField<HysteresisPath> initial_path;
          } linear_pressure;

          struct ExponentialRelaxation
          {
            Core::IO::InputField<double> tau;
          } exponential_relaxation;
        } recruitment_model;

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
    /**
     * Boundary conditions are attached to the nodes of the mesh via its `bc_id` point data
     * array: a node with `bc_id == 0` is unconstrained, a node with `bc_id == N` carries the
     * condition defined below under `id: N`.
     */
    struct BoundaryConditions
    {
      /**
       * One reusable boundary condition whose value follows a function of time. The variable it
       * constrains is not part of the definition: it follows from the list the definition sits in.
       * Any number of nodes may carry the same id.
       */
      struct FromFunctionDefinition
      {
        int id;
        int function_id;
      };

      /**
       * One reusable boundary condition prescribing the pleural pressure of a boundary node from
       * the total volume of all terminal units. Any number of nodes may carry the same id.
       */
      struct VolumeDependentPleuralPressureDefinition
      {
        /**
         * How the pleural pressure is tied to the terminal unit volume.
         */
        enum class Coupling : std::uint8_t
        {
          //! Evaluate from the total volume of the last converged timestep.
          Frozen,
        };

        /**
         * Pleural pressure over the normalized volume
         * xi = (V - residual_volume) / (total_lung_capacity - residual_volume) as
         * p_pl(xi) = pressure_offset + linear_coefficient * xi
         *          + exponential_coefficient * (exp(exponential_rate * xi) - 1).
         */
        struct NormalizedLinearExponential
        {
          double pressure_offset;
          double linear_coefficient;
          double exponential_coefficient;
          double exponential_rate;
        };

        int id;
        Coupling coupling;
        double residual_volume;
        double total_lung_capacity;
        NormalizedLinearExponential normalized_linear_exponential;
      };

      //! Definitions constraining the pressure dof, prescribing it by a function of time.
      std::vector<FromFunctionDefinition> pressure;
      //! Definitions constraining the flow dof, prescribing it by a function of time.
      std::vector<FromFunctionDefinition> flow;
      //! Definitions constraining the pressure dof, prescribing it from the terminal unit volume.
      std::vector<VolumeDependentPleuralPressureDefinition> volume_dependent_pleural_pressure;
    } boundary_conditions;
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
