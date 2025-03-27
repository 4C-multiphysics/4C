// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_potential_params.hpp"

#include "4C_beamcontact_input.hpp"
#include "4C_beaminteraction_potential_input.hpp"
#include "4C_global_data.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


namespace BeamInteraction
{

  void initialize_validate_beam_potential_params(BeamPotentialParams& params, double restart_time)
  {
    // Teuchos parameter list for beam potential-based interactions
    const Teuchos::ParameterList& beam_potential_params_list =
        Global::Problem::instance()->beam_potential_params();


    // potential law parameters
    // TODO remove parsers once parameters are directly read as vectors
    {
      std::string potential_law_exponents_in =
          Teuchos::getNumericStringParameter(beam_potential_params_list, "POT_LAW_EXPONENT");

      Core::IO::ValueParser potential_law_exponents_parser(potential_law_exponents_in,
          {.user_scope_message = "While reading potential law exponents: "});

      while (!potential_law_exponents_parser.at_end())
      {
        params.potential_law_exponents.push_back(potential_law_exponents_parser.read<double>());
      }
    }
    {
      std::string potential_law_prefactors_in =
          Teuchos::getNumericStringParameter(beam_potential_params_list, "POT_LAW_PREFACTOR");

      Core::IO::ValueParser potential_law_prefactors_parser(potential_law_prefactors_in,
          {.user_scope_message = "While reading potential law prefactors: "});

      while (!potential_law_prefactors_parser.at_end())
      {
        params.potential_law_prefactors.push_back(potential_law_prefactors_parser.read<double>());
      }
    }

    if (params.potential_law_prefactors.size() != params.potential_law_exponents.size())
      FOUR_C_THROW(
          "Number of potential law prefactors does not match number of potential law exponents."
          " Check your input file!");

    for (double pot_law_exponent : params.potential_law_exponents)
      if (pot_law_exponent <= 0)
        FOUR_C_THROW(
            "Only positive values are allowed for potential law exponent."
            " Check your input file");


    // potential strategy
    params.strategy =
        Teuchos::getIntegralValue<BeamPotential::Strategy>(beam_potential_params_list, "STRATEGY");

    if (params.strategy == BeamPotential::Strategy::vague)
      FOUR_C_THROW(
          "You must specify a strategy to be used to evaluate beam interaction potential!");

    // potential type
    params.potential_type =
        Teuchos::getIntegralValue<BeamPotential::Type>(beam_potential_params_list, "TYPE");

    if (params.potential_type == BeamPotential::Type::vague)
      FOUR_C_THROW("You must specify the type of the specified beam interaction potential!");

    if (params.potential_type == BeamPotential::Type::surface and
        params.strategy != BeamPotential::Strategy::double_length_specific_large_separations)
    {
      FOUR_C_THROW("Surface interaction is not implemented for this strategy yet!");
    }

    // cutoff radius
    params.cutoff_radius = beam_potential_params_list.get<std::optional<double>>("CUTOFF_RADIUS");

    if (params.cutoff_radius.has_value() and params.cutoff_radius.value() <= 0.0)
      FOUR_C_THROW("Invalid cutoff radius! Must be positive value or null to deactivate.");

    // regularization
    params.regularization_type = Teuchos::getIntegralValue<BeamPotential::RegularizationType>(
        beam_potential_params_list, "REGULARIZATION_TYPE");

    if ((params.regularization_type != BeamPotential::RegularizationType::none and
            params.strategy == BeamPotential::Strategy::double_length_specific_large_separations) or
        (params.regularization_type == BeamPotential::RegularizationType::constant and
            params.strategy == BeamPotential::Strategy::single_length_specific_small_separations))
    {
      FOUR_C_THROW(
          "This kind of regularization of the force law is not implemented for this strategy yet!");
    }

    params.regularization_separation =
        beam_potential_params_list.get<double>("REGULARIZATION_SEPARATION");

    if (params.regularization_type != BeamPotential::RegularizationType::none and
        params.regularization_separation <= 0.0)
    {
      FOUR_C_THROW(
          "Invalid regularization separation! Must be a positive value since force law "
          "is not defined for separations <= 0!");
    }

    // potential reduction strategy
    params.potential_reduction_length =
        beam_potential_params_list.get<std::optional<double>>("POTENTIAL_REDUCTION_LENGTH");

    if (params.potential_reduction_length.has_value() and
        params.potential_reduction_length.value() <= 0.0)
      FOUR_C_THROW(
          "Invalid potential reduction length! Must be positive value or none to deactivate.");

    // integration parameters
    params.n_integration_segments = beam_potential_params_list.get<int>("N_INTEGRATION_SEGMENTS");

    if (params.n_integration_segments <= 0)
      FOUR_C_THROW("Invalid number of integration segments per element!");

    params.n_gauss_points = beam_potential_params_list.get<int>("N_GAUSS_POINTS");

    if (params.n_gauss_points <= 0)
      FOUR_C_THROW("Invalid number of Gauss points per integration segment!");

    // automatic differentiation
    params.use_fad = beam_potential_params_list.get<bool>("AUTOMATIC_DIFFERENTIATION");

    // master/slave choice
    params.choice_master_slave = Teuchos::getIntegralValue<BeamPotential::MasterSlaveChoice>(
        beam_potential_params_list, "CHOICE_MASTER_SLAVE");

    if (params.choice_master_slave == BeamPotential::MasterSlaveChoice::vague)
    {
      FOUR_C_THROW("Invalid choice of master and slave!");
    }

    // check for vtk output which is to be handled by an own writer object
    params.write_visualization_output = beam_potential_params_list.sublist("RUNTIME VTK OUTPUT")
                                            .get<bool>("VTK_OUTPUT_BEAM_POTENTIAL");

    // create and initialize parameter container object for runtime output
    if (params.write_visualization_output)
    {
      params.runtime_output_params = {
          .visualization_parameters = Core::IO::visualization_parameters_factory(
              Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
              *Global::Problem::instance()->output_control_file(), restart_time)};

      initialize_validate_beam_potential_runtime_output_params(
          params.runtime_output_params, beam_potential_params_list.sublist("RUNTIME VTK OUTPUT"));
    }
  }

  void initialize_validate_beam_potential_runtime_output_params(
      BeamPotentialRuntimeOutputParams& params, Teuchos::ParameterList const& sublist)
  {
    params.output_interval = sublist.get<int>("INTERVAL_STEPS");

    params.write_all_iterations = sublist.get<bool>("EVERY_ITERATION");

    params.write_forces = sublist.get<bool>("FORCES");

    params.write_moments = sublist.get<bool>("MOMENTS");

    params.write_forces_moments_per_pair = sublist.get<bool>("WRITE_FORCE_MOMENT_PER_ELEMENTPAIR");

    params.write_uids = sublist.get<bool>("WRITE_UIDS");

    // overwrite global option to write visualization output of every iteration
    params.visualization_parameters.every_iteration_ = params.write_all_iterations;
  }
}  // namespace BeamInteraction
FOUR_C_NAMESPACE_CLOSE
