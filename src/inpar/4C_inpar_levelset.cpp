// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_levelset.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


void Inpar::LevelSet::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs levelsetcontrol{"LEVEL-SET CONTROL"};

  levelsetcontrol.specs.emplace_back(parameter<int>(
      "NUMSTEP", {.description = "Total number of time steps", .default_value = 24}));
  levelsetcontrol.specs.emplace_back(
      parameter<double>("TIMESTEP", {.description = "Time increment dt", .default_value = 0.1}));
  levelsetcontrol.specs.emplace_back(parameter<double>(
      "MAXTIME", {.description = "Total simulation time", .default_value = 1000.0}));
  levelsetcontrol.specs.emplace_back(parameter<int>(
      "RESULTSEVERY", {.description = "Increment for writing solution", .default_value = 1}));
  levelsetcontrol.specs.emplace_back(parameter<int>(
      "RESTARTEVERY", {.description = "Increment for writing restart", .default_value = 1}));

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::CalcErrorLevelSet>("CALCERROR", "No",
      "compute error compared to analytical solution", tuple<std::string>("No", "InitialField"),
      tuple<Inpar::ScaTra::CalcErrorLevelSet>(
          Inpar::ScaTra::calcerror_no_ls, Inpar::ScaTra::calcerror_initial_field),
      levelsetcontrol);

  levelsetcontrol.specs.emplace_back(parameter<bool>("EXTRACT_INTERFACE_VEL",
      {.description = "replace computed velocity at nodes of given distance of interface by "
                      "approximated interface velocity",
          .default_value = false}));
  levelsetcontrol.specs.emplace_back(parameter<int>("NUM_CONVEL_LAYERS",
      {.description =
              "number of layers around the interface which keep their computed convective velocity",
          .default_value = -1}));

  levelsetcontrol.move_into_collection(list);


  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs ls_reinit{levelsetcontrol, "REINITIALIZATION"};

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::ReInitialAction>("REINITIALIZATION",
      "None", "Type of reinitialization strategy for level set function",
      tuple<std::string>("None", "Signed_Distance_Function", "Sussman", "EllipticEq"),
      tuple<Inpar::ScaTra::ReInitialAction>(Inpar::ScaTra::reinitaction_none,
          Inpar::ScaTra::reinitaction_signeddistancefunction, Inpar::ScaTra::reinitaction_sussman,
          Inpar::ScaTra::reinitaction_ellipticeq),
      ls_reinit);

  ls_reinit.specs.emplace_back(parameter<bool>("REINIT_INITIAL",
      {.description = "Has level set field to be reinitialized before first time step?",
          .default_value = false}));
  ls_reinit.specs.emplace_back(parameter<int>(
      "REINITINTERVAL", {.description = "reinitialization interval", .default_value = 1}));

  // parameters for signed distance reinitialization
  ls_reinit.specs.emplace_back(parameter<bool>("REINITBAND",
      {.description = "reinitialization only within a band around the interface, or entire domain?",
          .default_value = false}));
  ls_reinit.specs.emplace_back(parameter<double>(
      "REINITBANDWIDTH", {.description = "level-set value defining band width for reinitialization",
                             .default_value = 1.0}));

  // parameters for reinitialization equation
  ls_reinit.specs.emplace_back(parameter<int>("NUMSTEPSREINIT",
      {.description = "(maximal) number of pseudo-time steps", .default_value = 1}));
  ls_reinit.specs.emplace_back(parameter<double>(
      "TIMESTEPREINIT", {.description = "pseudo-time step length (usually a * characteristic "
                                        "element length of discretization with a>0)",
                            .default_value = 1.0}));
  ls_reinit.specs.emplace_back(parameter<double>(
      "THETAREINIT", {.description = "theta for time discretization of reinitialization equation",
                         .default_value = 1.0}));
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::StabType>("STABTYPEREINIT", "SUPG",
      "Type of stabilization (if any). No stabilization is only reasonable for low-Peclet-number.",
      tuple<std::string>("no_stabilization", "SUPG", "GLS", "USFEM"),
      tuple<Inpar::ScaTra::StabType>(Inpar::ScaTra::stabtype_no_stabilization,
          Inpar::ScaTra::stabtype_SUPG, Inpar::ScaTra::stabtype_GLS, Inpar::ScaTra::stabtype_USFEM),
      ls_reinit);
  // this parameter selects the tau definition applied
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::TauType>("DEFINITION_TAU_REINIT",
      "Taylor_Hughes_Zarins", "Definition of tau",
      tuple<std::string>("Taylor_Hughes_Zarins", "Taylor_Hughes_Zarins_wo_dt", "Franca_Valentin",
          "Franca_Valentin_wo_dt", "Shakib_Hughes_Codina", "Shakib_Hughes_Codina_wo_dt", "Codina",
          "Codina_wo_dt", "Exact_1D", "Zero"),
      tuple<Inpar::ScaTra::TauType>(Inpar::ScaTra::tau_taylor_hughes_zarins,
          Inpar::ScaTra::tau_taylor_hughes_zarins_wo_dt, Inpar::ScaTra::tau_franca_valentin,
          Inpar::ScaTra::tau_franca_valentin_wo_dt, Inpar::ScaTra::tau_shakib_hughes_codina,
          Inpar::ScaTra::tau_shakib_hughes_codina_wo_dt, Inpar::ScaTra::tau_codina,
          Inpar::ScaTra::tau_codina_wo_dt, Inpar::ScaTra::tau_exact_1d, Inpar::ScaTra::tau_zero),
      ls_reinit);
  // this parameter governs whether all-scale subgrid diffusivity is included
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::ArtDiff>("ARTDIFFREINIT", "no",
      "potential incorporation of all-scale subgrid diffusivity (a.k.a. discontinuity-capturing) "
      "term",
      tuple<std::string>("no", "isotropic", "crosswind"),
      tuple<Inpar::ScaTra::ArtDiff>(Inpar::ScaTra::artdiff_none, Inpar::ScaTra::artdiff_isotropic,
          Inpar::ScaTra::artdiff_crosswind),
      ls_reinit);
  // this parameter selects the all-scale subgrid-diffusivity definition applied
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::AssgdType>("DEFINITION_ARTDIFFREINIT",
      "artificial_linear", "Definition of (all-scale) subgrid diffusivity",
      tuple<std::string>("artificial_linear", "artificial_linear_reinit",
          "Hughes_etal_86_nonlinear", "Tezduyar_Park_86_nonlinear",
          "Tezduyar_Park_86_nonlinear_wo_phizero", "doCarmo_Galeao_91_nonlinear",
          "Almeida_Silva_97_nonlinear", "YZbeta_nonlinear", "Codina_nonlinear"),
      tuple<Inpar::ScaTra::AssgdType>(Inpar::ScaTra::assgd_artificial,
          Inpar::ScaTra::assgd_lin_reinit, Inpar::ScaTra::assgd_hughes,
          Inpar::ScaTra::assgd_tezduyar, Inpar::ScaTra::assgd_tezduyar_wo_phizero,
          Inpar::ScaTra::assgd_docarmo, Inpar::ScaTra::assgd_almeida, Inpar::ScaTra::assgd_yzbeta,
          Inpar::ScaTra::assgd_codina),
      ls_reinit);

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::SmoothedSignType>("SMOOTHED_SIGN_TYPE",
      "SussmanSmerekaOsher1994", "sign function for reinitialization equation",
      tuple<std::string>("NonSmoothed",
          "SussmanFatemi1999",  // smeared-out Heaviside function
          "SussmanSmerekaOsher1994", "PengEtAl1999"),
      tuple<Inpar::ScaTra::SmoothedSignType>(Inpar::ScaTra::signtype_nonsmoothed,
          Inpar::ScaTra::signtype_SussmanFatemi1999,
          Inpar::ScaTra::signtype_SussmanSmerekaOsher1994, Inpar::ScaTra::signtype_PengEtAl1999),
      ls_reinit);
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::CharEleLengthReinit>(
      "CHARELELENGTHREINIT", "root_of_volume", "characteristic element length for sign function",
      tuple<std::string>("root_of_volume", "streamlength"),
      tuple<Inpar::ScaTra::CharEleLengthReinit>(
          Inpar::ScaTra::root_of_volume_reinit, Inpar::ScaTra::streamlength_reinit),
      ls_reinit);
  ls_reinit.specs.emplace_back(parameter<double>("INTERFACE_THICKNESS",
      {.description = "factor for interface thickness (multiplied by element length)",
          .default_value = 1.0}));
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::VelReinit>("VELREINIT",
      "integration_point_based",
      "evaluate velocity at integration point or compute node-based velocity",
      tuple<std::string>("integration_point_based", "node_based"),
      tuple<Inpar::ScaTra::VelReinit>(
          Inpar::ScaTra::vel_reinit_integration_point_based, Inpar::ScaTra::vel_reinit_node_based),
      ls_reinit);
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::LinReinit>("LINEARIZATIONREINIT",
      "newton", "linearization scheme for nonlinear convective term of reinitialization equation",
      tuple<std::string>("newton", "fixed_point"),
      tuple<Inpar::ScaTra::LinReinit>(Inpar::ScaTra::newton, Inpar::ScaTra::fixed_point),
      ls_reinit);
  ls_reinit.specs.emplace_back(parameter<bool>(
      "CORRECTOR_STEP", {.description = "correction of interface position via volume constraint "
                                        "according to Sussman & Fatemi",
                            .default_value = true}));
  ls_reinit.specs.emplace_back(parameter<double>(
      "CONVTOL_REINIT", {.description = "tolerance for convergence check according to Sussman et "
                                        "al. 1994 (turned off negative)",
                            .default_value = -1.0}));

  ls_reinit.specs.emplace_back(parameter<bool>("REINITVOLCORRECTION",
      {.description = "volume correction after reinitialization", .default_value = false}));

  ls_reinit.specs.emplace_back(parameter<double>("PENALTY_PARA",
      {.description = "penalty parameter for elliptic reinitialization", .default_value = -1.0}));

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::LSDim>("DIMENSION", "3D",
      "number of space dimensions for handling of quasi-2D problems with 3D elements",
      tuple<std::string>("3D", "2Dx", "2Dy", "2Dz"),
      tuple<Inpar::ScaTra::LSDim>(Inpar::ScaTra::ls_3D, Inpar::ScaTra::ls_2Dx,
          Inpar::ScaTra::ls_2Dy, Inpar::ScaTra::ls_2Dz),
      ls_reinit);

  ls_reinit.specs.emplace_back(parameter<bool>(
      "PROJECTION", {.description = "use L2-projection for grad phi and related quantities",
                        .default_value = true}));
  ls_reinit.specs.emplace_back(parameter<double>("PROJECTION_DIFF",
      {.description = "use diffusive term for L2-projection", .default_value = 0.0}));
  ls_reinit.specs.emplace_back(parameter<bool>("LUMPING",
      {.description = "use lumped mass matrix for L2-projection", .default_value = false}));

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::DiffFunc>("DIFF_FUNC", "hyperbolic",
      "function for diffusivity",
      tuple<std::string>("hyperbolic", "hyperbolic_smoothed_positive", "hyperbolic_clipped_05",
          "hyperbolic_clipped_1"),
      tuple<Inpar::ScaTra::DiffFunc>(Inpar::ScaTra::hyperbolic,
          Inpar::ScaTra::hyperbolic_smoothed_positive, Inpar::ScaTra::hyperbolic_clipped_05,
          Inpar::ScaTra::hyperbolic_clipped_1),
      ls_reinit);

  ls_reinit.move_into_collection(list);
}



void Inpar::LevelSet::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  /*--------------------------------------------------------------------*/
  // Taylor Galerkin outflow Boundaries for level set transport equation

  Core::Conditions::ConditionDefinition surfOutflowTaylorGalerkin(
      "TAYLOR GALERKIN OUTFLOW SURF CONDITIONS", "TaylorGalerkinOutflow",
      "Surface Taylor Galerkin Outflow", Core::Conditions::TaylorGalerkinOutflow, true,
      Core::Conditions::geometry_type_surface);

  condlist.push_back(surfOutflowTaylorGalerkin);

  /*--------------------------------------------------------------------*/

  Core::Conditions::ConditionDefinition surfneumanninflowTaylorGalerkin(
      "TAYLOR GALERKIN NEUMANN INFLOW SURF CONDITIONS", "TaylorGalerkinNeumannInflow",
      "Surface Taylor Galerkin Neumann Inflow", Core::Conditions::TaylorGalerkinNeumannInflow, true,
      Core::Conditions::geometry_type_surface);

  condlist.push_back(surfneumanninflowTaylorGalerkin);


  /*--------------------------------------------------------------------*/
  // Characteristic Galerkin Boundaries for LevelSet-reinitialization

  Core::Conditions::ConditionDefinition surfreinitializationtaylorgalerkin(
      "REINITIALIZATION TAYLOR GALERKIN SURF CONDITIONS", "ReinitializationTaylorGalerkin",
      "Surface reinitialization Taylor Galerkin", Core::Conditions::ReinitializationTaylorGalerkin,
      true, Core::Conditions::geometry_type_surface);

  condlist.push_back(surfreinitializationtaylorgalerkin);

  /*--------------------------------------------------------------------*/
  // level-set condition for contact points

  Core::Conditions::ConditionDefinition linelscontact("DESIGN LINE LEVEL SET CONTACT CONDITION",
      "LsContact", "level-set condition for contact points", Core::Conditions::LsContact, false,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition pointlscontact("DESIGN POINT LEVEL SET CONTACT CONDITION",
      "LsContact", "level-set condition for contact points", Core::Conditions::LsContact, false,
      Core::Conditions::geometry_type_point);

  condlist.push_back(linelscontact);
  condlist.push_back(pointlscontact);
}

FOUR_C_NAMESPACE_CLOSE
