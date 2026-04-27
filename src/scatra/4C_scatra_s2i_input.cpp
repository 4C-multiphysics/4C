// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_s2i_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
FOUR_C_NAMESPACE_OPEN

/*------------------------------------------------------------------------*
 | valid parameters for scatra-scatra interface coupling   fang 01/16 |
 *------------------------------------------------------------------------*/
std::vector<Core::IO::InputSpec> S2I::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> specs;
  specs.push_back(group("SCALAR TRANSPORT DYNAMIC/S2I COUPLING",
      {

          // type of mortar meshtying
          deprecated_selection<CouplingType>("COUPLINGTYPE",
              {
                  {"Undefined", coupling_undefined},
                  {"MatchingNodes", coupling_matching_nodes},
                  {"StandardMortar", coupling_mortar_standard},
                  {"SaddlePointMortar_Petrov", coupling_mortar_saddlepoint_petrov},
                  {"SaddlePointMortar_Bubnov", coupling_mortar_saddlepoint_bubnov},
                  {"CondensedMortar_Petrov", coupling_mortar_condensed_petrov},
                  {"CondensedMortar_Bubnov", coupling_mortar_condensed_bubnov},
                  {"StandardNodeToSegment", coupling_nts_standard},
              },
              {.description = "type of mortar meshtying", .default_value = coupling_undefined}),

          // flag for interface side underlying Lagrange multiplier definition
          deprecated_selection<InterfaceSides>("LMSIDE",
              {
                  {"slave", side_source},
                  {"master", side_master},
              },
              {.description = "flag for interface side underlying Lagrange multiplier definition",
                  .default_value = side_source}),

          // flag for evaluation of interface linearizations and residuals on slave side only
          parameter<bool>(
              "SLAVEONLY", {.description = "flag for evaluation of interface linearizations and "
                                           "residuals on slave side only",
                               .default_value = false}),

          // node-to-segment projection tolerance
          parameter<double>("NTSPROJTOL",
              {.description = "node-to-segment projection tolerance", .default_value = 0.0}),

          // flag for evaluation of scatra-scatra interface coupling involving interface layer
          // growth
          deprecated_selection<GrowthEvaluation>("INTLAYERGROWTH_EVALUATION",
              {
                  {"none", growth_evaluation_none},
                  {"monolithic", growth_evaluation_monolithic},
                  {"semi-implicit", growth_evaluation_semi_implicit},
              },
              {.description =
                      "flag for evaluation of scatra-scatra interface coupling involving interface "
                      "layer growth",
                  .default_value = growth_evaluation_none}),

          // local Newton-Raphson convergence tolerance for scatra-scatra interface coupling
          // involving
          // interface layer growth
          parameter<double>("INTLAYERGROWTH_CONVTOL",
              {.description =
                      "local Newton-Raphson convergence tolerance for scatra-scatra interface "
                      "coupling involving interface layer growth",
                  .default_value = 1.e-12}),

          // maximum number of local Newton-Raphson iterations for scatra-scatra interface coupling
          // involving interface layer growth
          parameter<int>("INTLAYERGROWTH_ITEMAX",
              {.description = "maximum number of local Newton-Raphson iterations for scatra-scatra "
                              "interface coupling involving interface layer growth",
                  .default_value = 5}),

          // ID of linear solver for monolithic scatra-scatra interface coupling involving interface
          // layer
          // growth
          parameter<int>("INTLAYERGROWTH_LINEAR_SOLVER",
              {.description = "ID of linear solver for monolithic scatra-scatra interface coupling "
                              "involving interface layer growth",
                  .default_value = -1}),
          // modified time step size for scatra-scatra interface coupling involving interface layer
          // growth
          parameter<double>("INTLAYERGROWTH_TIMESTEP",
              {.description =
                      "modified time step size for scatra-scatra interface coupling involving "
                      "interface layer growth",
                  .default_value = -1.}),

          parameter<bool>("MESHTYING_CONDITIONS_INDEPENDENT_SETUP",
              {.description = "mesh tying for different conditions should be setup independently",
                  .default_value = false}),

          parameter<bool>("OUTPUT_INTERFACE_FLUX",
              {.description = "evaluate integral of coupling flux on slave side "
                              "for each s2i condition and write it to csv file",
                  .default_value = false})},
      {.required = false}));
  return specs;
}


/*------------------------------------------------------------------------*
 | set valid conditions for scatra-scatra interface coupling   fang 01/16 |
 *------------------------------------------------------------------------*/
void S2I::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // scatra-scatra interface mesh tying condition
  {
    // definition of scatra-scatra interface mesh tying line condition
    Core::Conditions::ConditionDefinition s2imeshtyingline("DESIGN S2I MESHTYING LINE CONDITIONS",
        "S2IMeshtying", "Scatra-scatra line interface mesh tying", Core::Conditions::S2IMeshtying,
        true, Core::Conditions::geometry_type_line);

    // definition of scatra-scatra interface mesh tying surface condition
    Core::Conditions::ConditionDefinition s2imeshtyingsurf("DESIGN S2I MESHTYING SURF CONDITIONS",
        "S2IMeshtying", "Scatra-scatra surface interface mesh tying",
        Core::Conditions::S2IMeshtying, true, Core::Conditions::geometry_type_surface);

    const auto make_s2imeshtying = [&condlist](Core::Conditions::ConditionDefinition& cond)
    {
      cond.add_component(parameter<int>("ConditionID"));
      cond.add_component(deprecated_selection<S2I::InterfaceSides>("INTERFACE_SIDE",
          {{"Undefined", S2I::side_undefined}, {"Slave", S2I::side_source},
              {"Master", S2I::side_master}},
          {.description = "interface side"}));
      cond.add_component(parameter<int>("S2I_KINETICS_ID"));

      condlist.push_back(cond);
    };

    make_s2imeshtying(s2imeshtyingline);
    make_s2imeshtying(s2imeshtyingsurf);
  }

  /*--------------------------------------------------------------------*/
  // scatra-scatra interface kinetics condition
  {
    // definition of scatra-scatra interface kinetics point condition
    Core::Conditions::ConditionDefinition s2ikineticspoint("DESIGN S2I KINETICS POINT CONDITIONS",
        "S2IKinetics", "Scatra-scatra line interface kinetics", Core::Conditions::S2IKinetics, true,
        Core::Conditions::geometry_type_point);

    // definition of scatra-scatra interface kinetics line condition
    Core::Conditions::ConditionDefinition s2ikineticsline("DESIGN S2I KINETICS LINE CONDITIONS",
        "S2IKinetics", "Scatra-scatra line interface kinetics", Core::Conditions::S2IKinetics, true,
        Core::Conditions::geometry_type_line);

    // definition of scatra-scatra interface kinetics surface condition
    Core::Conditions::ConditionDefinition s2ikineticssurf("DESIGN S2I KINETICS SURF CONDITIONS",
        "S2IKinetics", "Scatra-scatra surface interface kinetics", Core::Conditions::S2IKinetics,
        true, Core::Conditions::geometry_type_surface);

    // Macro-micro coupling condition for micro scale in multi-scale scalar transport problems
    Core::Conditions::ConditionDefinition multiscalecouplingpoint(
        "DESIGN SCATRA MULTI-SCALE COUPLING POINT CONDITIONS", "ScatraMultiScaleCoupling",
        "Scalar transport multi-scale coupling condition",
        Core::Conditions::ScatraMultiScaleCoupling, false, Core::Conditions::geometry_type_point);

    std::vector<Core::IO::InputSpec> kinetic_model_choices;
    {
      {
        // constant and linear permeability
        auto constlinperm = all_of({
            deprecated_selection<S2I::KineticModels>("KINETIC_MODEL",
                {
                    {"ConstantPermeability", S2I::kinetics_constperm},
                    {"LinearPermeability", S2I::kinetics_linearperm},
                }),
            parameter<int>("NUMSCAL"),
            parameter<std::vector<double>>(
                "PERMEABILITIES", {.size = from_parameter<int>("NUMSCAL")}),
            parameter<bool>("IS_PSEUDO_CONTACT"),
        });
        kinetic_model_choices.emplace_back(std::move(constlinperm));
      }

      {
        auto butler_volmer = all_of({
            deprecated_selection<S2I::KineticModels>("KINETIC_MODEL",
                {
                    {"Butler-Volmer", S2I::kinetics_butlervolmer},
                    {"Butler-Volmer_Linearized", S2I::kinetics_butlervolmerlinearized},
                    {"Butler-VolmerReduced", S2I::kinetics_butlervolmerreduced},
                    {"Butler-VolmerReduced_Linearized",
                        S2I::kinetics_butlervolmerreducedlinearized},
                }),
            parameter<int>("NUMSCAL"),
            parameter<std::vector<int>>(
                "STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
            parameter<int>("E-"),
            parameter<double>("K_R"),
            parameter<double>("ALPHA_A"),
            parameter<double>("ALPHA_C"),
            parameter<bool>("IS_PSEUDO_CONTACT"),
        });
        kinetic_model_choices.emplace_back(std::move(butler_volmer));
      }

      {
        auto butler_volmer_peltier = all_of({
            deprecated_selection<S2I::KineticModels>("KINETIC_MODEL",
                {
                    {"Butler-Volmer-Peltier", S2I::kinetics_butlervolmerpeltier},
                }),
            parameter<int>("NUMSCAL"),
            parameter<std::vector<int>>(
                "STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
            parameter<int>("E-"),
            parameter<double>("K_R"),
            parameter<double>("ALPHA_A"),
            parameter<double>("ALPHA_C"),
            parameter<bool>("IS_PSEUDO_CONTACT"),
            parameter<double>("PELTIER"),
        });

        kinetic_model_choices.emplace_back(std::move(butler_volmer_peltier));
      }

      {
        auto butler_volmer_reduced_capacitance = all_of({
            deprecated_selection<S2I::KineticModels>("KINETIC_MODEL",
                {
                    {"Butler-VolmerReduced_Capacitance",
                        S2I::kinetics_butlervolmerreducedcapacitance},
                }),
            parameter<int>("NUMSCAL"),
            parameter<std::vector<int>>(
                "STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
            parameter<int>("E-"),
            parameter<double>("K_R"),
            parameter<double>("CAPACITANCE"),
            parameter<double>("ALPHA_A"),
            parameter<double>("ALPHA_C"),
            parameter<bool>("IS_PSEUDO_CONTACT"),
        });

        kinetic_model_choices.emplace_back(std::move(butler_volmer_reduced_capacitance));
      }

      {
        auto butler_volmer_resistance = all_of({
            deprecated_selection<S2I::KineticModels>("KINETIC_MODEL",
                {
                    {"Butler-Volmer_Resistance", S2I::kinetics_butlervolmerresistance},
                }),
            parameter<int>("NUMSCAL"),
            parameter<std::vector<int>>(
                "STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
            parameter<int>("E-"),
            parameter<double>("K_R"),
            parameter<double>("ALPHA_A"),
            parameter<double>("ALPHA_C"),
            parameter<bool>("IS_PSEUDO_CONTACT"),
            parameter<double>("RESISTANCE"),
            parameter<double>("CONVTOL_IMPLBUTLERVOLMER"),
            parameter<int>("ITEMAX_IMPLBUTLERVOLMER"),
        });
        kinetic_model_choices.emplace_back(std::move(butler_volmer_resistance));
      }

      {
        auto butler_volmer_reduced_with_resistance = all_of({
            deprecated_selection<S2I::KineticModels>("KINETIC_MODEL",
                {
                    {"Butler-VolmerReduced_Resistance",
                        S2I::kinetics_butlervolmerreducedresistance},
                }),
            parameter<int>("NUMSCAL"),
            parameter<std::vector<int>>(
                "STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
            parameter<int>("E-"),
            parameter<double>("K_R"),
            parameter<double>("ALPHA_A"),
            parameter<double>("ALPHA_C"),
            parameter<bool>("IS_PSEUDO_CONTACT"),
            parameter<double>("RESISTANCE"),
            parameter<double>("CONVTOL_IMPLBUTLERVOLMER"),
            parameter<int>("ITEMAX_IMPLBUTLERVOLMER"),
        });

        kinetic_model_choices.emplace_back(std::move(butler_volmer_reduced_with_resistance));
      }

      {
        auto butler_volmer_reduced_thermo = all_of({
            deprecated_selection<S2I::KineticModels>("KINETIC_MODEL",
                {
                    {"Butler-VolmerReduced_ThermoResistance",
                        S2I::kinetics_butlervolmerreducedthermoresistance},
                }),
            parameter<int>("NUMSCAL"),
            parameter<std::vector<int>>(
                "STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
            parameter<int>("E-"),
            parameter<double>("K_R"),
            parameter<double>("ALPHA_A"),
            parameter<double>("ALPHA_C"),
            parameter<bool>("IS_PSEUDO_CONTACT"),
            parameter<double>("THERMOPERM"),
            parameter<double>("MOLAR_HEAT_CAPACITY"),
        });

        kinetic_model_choices.emplace_back(std::move(butler_volmer_reduced_thermo));
      }

      {
        auto constant_interface_resistance = all_of({
            deprecated_selection<S2I::KineticModels>("KINETIC_MODEL",
                {
                    {"ConstantInterfaceResistance", S2I::kinetics_constantinterfaceresistance},
                }),
            parameter<std::vector<int>>("ONOFF", {.size = 2}),
            parameter<double>("RESISTANCE"),
            parameter<int>("E-"),
            parameter<bool>("IS_PSEUDO_CONTACT"),
        });

        kinetic_model_choices.emplace_back(std::move(constant_interface_resistance));
      }

      {
        // no interface flux
        auto noflux = deprecated_selection<KineticModels>(
            "KINETIC_MODEL", {
                                 {"NoInterfaceFlux", S2I::kinetics_nointerfaceflux},
                             });

        kinetic_model_choices.emplace_back(std::move(noflux));
      }

      multiscalecouplingpoint.add_component(one_of(kinetic_model_choices));
    }

    auto interface_side_options = one_of({
        deprecated_selection<InterfaceSides>(
            "INTERFACE_SIDE", {{"Master", side_master}, {"Undefined", side_undefined}}),
        all_of({
            deprecated_selection<InterfaceSides>("INTERFACE_SIDE", {{"Slave", side_source}}),
            one_of(kinetic_model_choices),
        }),
    });

    const auto make_s2ikinetics = [&condlist, &interface_side_options](
                                      Core::Conditions::ConditionDefinition& cond)
    {
      cond.add_component(parameter<int>("ConditionID"));
      cond.add_component(interface_side_options);

      condlist.push_back(cond);
    };

    make_s2ikinetics(s2ikineticspoint);
    make_s2ikinetics(s2ikineticsline);
    make_s2ikinetics(s2ikineticssurf);


    condlist.emplace_back(multiscalecouplingpoint);
  }



  /*--------------------------------------------------------------------*/
  // scatra-scatra interface coupling involving interface layer growth
  {
    // definition of scatra-scatra interface coupling line condition involving interface layer
    // growth
    Core::Conditions::ConditionDefinition s2igrowthline(
        "DESIGN S2I KINETICS GROWTH LINE CONDITIONS", "S2IKineticsGrowth",
        "Scatra-scatra line interface layer growth kinetics", Core::Conditions::S2IKineticsGrowth,
        true, Core::Conditions::geometry_type_line);

    // definition of scatra-scatra interface coupling surface condition involving interface layer
    // growth
    Core::Conditions::ConditionDefinition s2igrowthsurf(
        "DESIGN S2I KINETICS GROWTH SURF CONDITIONS", "S2IKineticsGrowth",
        "Scatra-scatra surface interface layer growth kinetics",
        Core::Conditions::S2IKineticsGrowth, true, Core::Conditions::geometry_type_surface);

    auto butler_volmer = all_of({
        parameter<int>("NUMSCAL"),
        parameter<std::vector<int>>("STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
        parameter<int>("E-"),
        parameter<double>("K_R"),
        parameter<double>("ALPHA_A"),
        parameter<double>("ALPHA_C"),
        parameter<double>("MOLMASS"),
        parameter<double>("DENSITY"),
        parameter<double>("CONDUCTIVITY"),
        deprecated_selection<RegularizationType>("REGTYPE",
            {
                {"none", S2I::regularization_none},
                {"polynomial", S2I::regularization_polynomial},
                {"Hein", S2I::regularization_hein},
                {"trigonometrical", S2I::regularization_trigonometrical},
            }),
        parameter<double>("REGPAR"),
        parameter<double>("INITTHICKNESS"),
    });

    const auto make_s2igrowth = [&condlist, &butler_volmer](
                                    Core::Conditions::ConditionDefinition& cond)
    {
      cond.add_component(parameter<int>("ConditionID"));
      cond.add_component(deprecated_selection<GrowthKineticModels>(
          "KINETIC_MODEL", {{"Butler-Volmer", growth_kinetics_butlervolmer}}));
      cond.add_component(butler_volmer);

      condlist.emplace_back(cond);
    };

    make_s2igrowth(s2igrowthline);
    make_s2igrowth(s2igrowthsurf);
  }

  /*--------------------------------------------------------------------*/
  // scatra-scatra interface with micro-macro coupling for space-charge layers
  {
    Core::Conditions::ConditionDefinition s2isclcond("DESIGN S2I SCL COUPLING SURF CONDITIONS",
        "S2ISCLCoupling", "Scatra-scatra surface with SCL micro-macro coupling between",
        Core::Conditions::S2ISCLCoupling, true, Core::Conditions::geometry_type_surface);

    s2isclcond.add_component(deprecated_selection<InterfaceSides>("INTERFACE_SIDE",
        {{"Undefined", S2I::side_undefined}, {"Slave", S2I::side_source},
            {"Master", S2I::side_master}},
        {.description = "interface side"}));

    condlist.emplace_back(s2isclcond);
  }
}

FOUR_C_NAMESPACE_CLOSE