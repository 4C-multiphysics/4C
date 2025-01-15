// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_elch.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_io_linecomponent.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Inpar::ElCh::set_valid_parameters(Teuchos::ParameterList& list)
{
  using namespace Input;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& elchcontrol =
      list.sublist("ELCH CONTROL", false, "control parameters for electrochemistry problems\n");

  Core::Utils::int_parameter("MOVBOUNDARYITEMAX", 10,
      "Maximum number of outer iterations in electrode shape change computations", &elchcontrol);
  Core::Utils::double_parameter("MOVBOUNDARYCONVTOL", 1e-6,
      "Convergence check tolerance for outer loop in electrode shape change computations",
      &elchcontrol);
  Core::Utils::double_parameter(
      "TEMPERATURE", 298.0, "Constant temperature (Kelvin)", &elchcontrol);
  Core::Utils::int_parameter("TEMPERATURE_FROM_FUNCT", -1,
      "Homogeneous temperature within electrochemistry field that can be time dependent according "
      "to function definition",
      &elchcontrol);
  Core::Utils::double_parameter("FARADAY_CONSTANT", 9.64853399e4,
      "Faraday constant (in unit system as chosen in input file)", &elchcontrol);
  Core::Utils::double_parameter("GAS_CONSTANT", 8.314472,
      "(universal) gas constant (in unit system as chosen in input file)", &elchcontrol);
  // parameter for possible types of ELCH algorithms for deforming meshes
  setStringToIntegralParameter<Inpar::ElCh::ElchMovingBoundary>("MOVINGBOUNDARY", "No",
      "ELCH algorithm for deforming meshes",
      tuple<std::string>("No", "pseudo-transient", "fully-transient"),
      tuple<std::string>("no moving boundary algorithm",
          "pseudo-transient moving boundary algorithm",
          "full moving boundary algorithm including fluid solve"),
      tuple<Inpar::ElCh::ElchMovingBoundary>(
          elch_mov_bndry_no, elch_mov_bndry_pseudo_transient, elch_mov_bndry_fully_transient),
      &elchcontrol);
  Core::Utils::double_parameter(
      "MOLARVOLUME", 0.0, "Molar volume for electrode shape change computations", &elchcontrol);
  Core::Utils::double_parameter("MOVBOUNDARYTHETA", 0.0,
      "One-step-theta factor in electrode shape change computations", &elchcontrol);
  Core::Utils::bool_parameter("GALVANOSTATIC", "No", "flag for galvanostatic mode", &elchcontrol);
  setStringToIntegralParameter<Inpar::ElCh::ApproxElectResist>("GSTAT_APPROX_ELECT_RESIST",
      "relation_pot_cur", "relation of potential and current flow",
      tuple<std::string>("relation_pot_cur", "effective_length_with_initial_cond",
          "effective_length_with_integrated_cond"),
      tuple<Inpar::ElCh::ApproxElectResist>(approxelctresist_relpotcur,
          approxelctresist_effleninitcond, approxelctresist_efflenintegcond),
      &elchcontrol);
  Core::Utils::int_parameter(
      "GSTATCONDID_CATHODE", 0, "condition id of electrode kinetics for cathode", &elchcontrol);
  Core::Utils::int_parameter(
      "GSTATCONDID_ANODE", 1, "condition id of electrode kinetics for anode", &elchcontrol);
  Core::Utils::double_parameter(
      "GSTATCONVTOL", 1.e-5, "Convergence check tolerance for galvanostatic mode", &elchcontrol);
  Core::Utils::double_parameter("GSTATCURTOL", 1.e-15, "Current Tolerance", &elchcontrol);
  Core::Utils::int_parameter(
      "GSTATFUNCTNO", -1, "function number defining the imposed current curve", &elchcontrol);
  Core::Utils::int_parameter(
      "GSTATITEMAX", 10, "maximum number of iterations for galvanostatic mode", &elchcontrol);
  Core::Utils::double_parameter(
      "GSTAT_LENGTH_CURRENTPATH", 0.0, "average length of the current path", &elchcontrol);

  setStringToIntegralParameter<Inpar::ElCh::EquPot>("EQUPOT", "Undefined",
      "type of closing equation for electric potential",
      tuple<std::string>(
          "Undefined", "ENC", "ENC_PDE", "ENC_PDE_ELIM", "Poisson", "Laplace", "divi"),
      tuple<Inpar::ElCh::EquPot>(equpot_undefined, equpot_enc, equpot_enc_pde, equpot_enc_pde_elim,
          equpot_poisson, equpot_laplace, equpot_divi),
      &elchcontrol);
  Core::Utils::bool_parameter(
      "DIFFCOND_FORMULATION", "No", "Activation of diffusion-conduction formulation", &elchcontrol);
  Core::Utils::bool_parameter("INITPOTCALC", "No",
      "Automatically calculate initial field for electric potential", &elchcontrol);
  Core::Utils::bool_parameter("ONLYPOTENTIAL", "no",
      "Coupling of general ion transport equation with Laplace equation", &elchcontrol);
  Core::Utils::bool_parameter("COUPLE_BOUNDARY_FLUXES", "Yes",
      "Coupling of lithium-ion flux density and electric current density at Dirichlet and Neumann "
      "boundaries",
      &elchcontrol);
  Core::Utils::double_parameter(
      "CYCLING_TIMESTEP", -1., "modified time step size for CCCV cell cycling", &elchcontrol);
  Core::Utils::bool_parameter("ELECTRODE_INFO_EVERY_STEP", "No",
      "the cell voltage, SOC, and C-Rate will be written to the csv file every step, even if "
      "RESULTSEVERY is not 1",
      &elchcontrol);

  /*----------------------------------------------------------------------*/
  // attention: this list is a sublist of elchcontrol
  Teuchos::ParameterList& elchdiffcondcontrol = elchcontrol.sublist(
      "DIFFCOND", false, "control parameters for electrochemical diffusion conduction problems\n");

  Core::Utils::bool_parameter(
      "CURRENT_SOLUTION_VAR", "No", "Current as a solution variable", &elchdiffcondcontrol);
  Core::Utils::bool_parameter("MAT_DIFFCOND_DIFFBASED", "Yes",
      "Coupling terms of chemical diffusion for current equation are based on t and kappa",
      &elchdiffcondcontrol);

  /// dilute solution theory (diffusion potential in current equation):
  ///    A          B
  ///   |--|  |----------|
  ///   z_1 + (z_2 - z_1) t_1
  /// ------------------------ (RT/F kappa (1+f+-) 1/c_k grad c_k)
  ///      z_1 z_2
  ///     |________|
  ///         C
  //
  // default: concentrated solution theory according to Newman
  Core::Utils::double_parameter("MAT_NEWMAN_CONST_A", 2.0,
      "Constant A for the Newman model(term for the concentration overpotential)",
      &elchdiffcondcontrol);
  Core::Utils::double_parameter("MAT_NEWMAN_CONST_B", -2.0,
      "Constant B for the Newman model(term for the concentration overpotential)",
      &elchdiffcondcontrol);
  Core::Utils::double_parameter("MAT_NEWMAN_CONST_C", -1.0,
      "Constant C for the Newman model(term for the concentration overpotential)",
      &elchdiffcondcontrol);
  Core::Utils::double_parameter(
      "PERMITTIVITY_VACUUM", 8.8541878128e-12, "Vacuum permittivity", &elchdiffcondcontrol);

  /*----------------------------------------------------------------------*/
  // sublist for space-charge layers
  auto& sclcontrol = elchcontrol.sublist(
      "SCL", false, "control parameters for coupled problems with space-charge layer formation\n");

  Core::Utils::bool_parameter(
      "ADD_MICRO_MACRO_COUPLING", "No", "flag for micro macro coupling with scls", &sclcontrol);
  Core::Utils::bool_parameter("COUPLING_OUTPUT", "No",
      "write coupled node gids and node coordinates to csv file", &sclcontrol);
  Core::Utils::bool_parameter(
      "INITPOTCALC", "No", "calculate initial potential field?", &sclcontrol);
  Core::Utils::int_parameter("SOLVER", -1, "solver for coupled SCL problem", &sclcontrol);
  setStringToIntegralParameter<Core::LinAlg::MatrixType>("MATRIXTYPE", "undefined",
      "type of global system matrix in global system of equations",
      tuple<std::string>("undefined", "block", "sparse"),
      tuple<Core::LinAlg::MatrixType>(Core::LinAlg::MatrixType::undefined,
          Core::LinAlg::MatrixType::block_field, Core::LinAlg::MatrixType::sparse),
      &sclcontrol);
  Core::Utils::int_parameter("ADAPT_TIME_STEP", -1,
      "time step when time step size should be updated to 'ADAPTED_TIME_STEP_SIZE'.", &sclcontrol);
  Core::Utils::double_parameter("ADAPTED_TIME_STEP_SIZE", -1.0, "new time step size.", &sclcontrol);

  setStringToIntegralParameter<ScaTra::InitialField>("INITIALFIELD", "zero_field",
      "Initial Field for scalar transport problem",
      tuple<std::string>("zero_field", "field_by_function", "field_by_condition"),
      tuple<ScaTra::InitialField>(ScaTra::initfield_zero_field, ScaTra::initfield_field_by_function,
          ScaTra::initfield_field_by_condition),
      &sclcontrol);

  Core::Utils::int_parameter(
      "INITFUNCNO", -1, "function number for scalar transport initial field", &sclcontrol);
}


void Inpar::ElCh::set_valid_conditions(
    std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>& condlist)
{
  using namespace Input;

  /*--------------------------------------------------------------------*/
  // electrode state of charge

  // definition of electrode state of charge surface and volume conditions
  auto electrodesocline = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN ELECTRODE STATE OF CHARGE LINE CONDITIONS", "ElectrodeSOC",
      "electrode state of charge line condition", Core::Conditions::ElectrodeSOC, true,
      Core::Conditions::geometry_type_line);

  auto electrodesocsurf = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN ELECTRODE STATE OF CHARGE SURF CONDITIONS", "ElectrodeSOC",
      "electrode state of charge surface condition", Core::Conditions::ElectrodeSOC, true,
      Core::Conditions::geometry_type_surface);
  auto electrodesocvol = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN ELECTRODE STATE OF CHARGE VOL CONDITIONS", "ElectrodeSOC",
      "electrode state of charge volume condition", Core::Conditions::ElectrodeSOC, true,
      Core::Conditions::geometry_type_volume);

  for (const auto& cond : {electrodesocline, electrodesocsurf, electrodesocvol})
  {
    // insert input file line components into condition definitions
    cond->add_component(std::make_shared<Input::SeparatorComponent>("ID"));
    cond->add_component(std::make_shared<Input::IntComponent>("ConditionID"));
    add_named_real(cond, "C_0%");
    add_named_real(cond, "C_100%");
    add_named_real(cond, "ONE_HOUR");

    // insert condition definitions into global list of valid condition definitions
    condlist.emplace_back(cond);
  }


  /*--------------------------------------------------------------------*/
  // cell voltage

  // definition of cell voltage point, line, and surface conditions
  auto cellvoltagepoint = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN CELL VOLTAGE POINT CONDITIONS", "CellVoltagePoint", "cell voltage point condition",
      Core::Conditions::CellVoltage, false, Core::Conditions::geometry_type_point);

  auto cellvoltageline = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN CELL VOLTAGE LINE CONDITIONS", "CellVoltage", "cell voltage line condition",
      Core::Conditions::CellVoltage, true, Core::Conditions::geometry_type_line);

  auto cellvoltagesurf = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN CELL VOLTAGE SURF CONDITIONS", "CellVoltage", "cell voltage surface condition",
      Core::Conditions::CellVoltage, true, Core::Conditions::geometry_type_surface);

  for (const auto& cond : {cellvoltagepoint, cellvoltageline, cellvoltagesurf})
  {
    // insert input file line components into condition definitions
    cond->add_component(std::make_shared<Input::SeparatorComponent>("ID"));
    cond->add_component(std::make_shared<Input::IntComponent>("ConditionID"));

    // insert condition definitions into global list of valid condition definitions
    condlist.emplace_back(cond);
  }


  /*--------------------------------------------------------------------*/
  // electrode kinetics as boundary condition on electrolyte
  {
    std::map<int, std::pair<std::string, std::vector<std::shared_ptr<Input::LineComponent>>>>
        reaction_model_choices;

    // Butler-Volmer
    std::vector<std::shared_ptr<Input::LineComponent>> butlervolmer;
    butlervolmer.emplace_back(std::make_shared<Input::SeparatorComponent>("ALPHA_A"));
    butlervolmer.emplace_back(std::make_shared<Input::RealComponent>("ALPHA_A"));
    butlervolmer.emplace_back(std::make_shared<Input::SeparatorComponent>("ALPHA_C"));
    butlervolmer.emplace_back(std::make_shared<Input::RealComponent>("ALPHA_C"));
    butlervolmer.emplace_back(std::make_shared<Input::SeparatorComponent>("I0"));
    butlervolmer.emplace_back(std::make_shared<Input::RealComponent>("I0"));
    butlervolmer.emplace_back(std::make_shared<Input::SeparatorComponent>("GAMMA"));
    butlervolmer.emplace_back(std::make_shared<Input::RealComponent>("GAMMA"));
    butlervolmer.emplace_back(std::make_shared<Input::SeparatorComponent>("REFCON"));
    butlervolmer.emplace_back(std::make_shared<Input::RealComponent>("REFCON"));
    butlervolmer.emplace_back(std::make_shared<Input::SeparatorComponent>("DL_SPEC_CAP"));
    butlervolmer.emplace_back(std::make_shared<Input::RealComponent>("DL_SPEC_CAP"));
    reaction_model_choices.emplace(
        butler_volmer, std::make_pair("Butler-Volmer", std::move(butlervolmer)));

    // Butler-Volmer Yang
    // parameter are identical to Butler-Volmer
    std::vector<std::shared_ptr<Input::LineComponent>> butlervolmeryang;
    butlervolmeryang.emplace_back(std::make_shared<Input::SeparatorComponent>("ALPHA_A"));
    butlervolmeryang.emplace_back(std::make_shared<Input::RealComponent>("ALPHA_A"));
    butlervolmeryang.emplace_back(std::make_shared<Input::SeparatorComponent>("ALPHA_C"));
    butlervolmeryang.emplace_back(std::make_shared<Input::RealComponent>("ALPHA_C"));
    butlervolmeryang.emplace_back(std::make_shared<Input::SeparatorComponent>("I0"));
    butlervolmeryang.emplace_back(std::make_shared<Input::RealComponent>("I0"));
    butlervolmeryang.emplace_back(std::make_shared<Input::SeparatorComponent>("GAMMA"));
    butlervolmeryang.emplace_back(std::make_shared<Input::RealComponent>("GAMMA"));
    butlervolmeryang.emplace_back(std::make_shared<Input::SeparatorComponent>("REFCON"));
    butlervolmeryang.emplace_back(std::make_shared<Input::RealComponent>("REFCON"));
    butlervolmeryang.emplace_back(std::make_shared<Input::SeparatorComponent>("DL_SPEC_CAP"));
    butlervolmeryang.emplace_back(std::make_shared<Input::RealComponent>("DL_SPEC_CAP"));
    reaction_model_choices.emplace(Inpar::ElCh::butler_volmer_yang1997,
        std::make_pair("Butler-Volmer-Yang1997", butlervolmeryang));

    // Tafel kinetics
    std::vector<std::shared_ptr<Input::LineComponent>> tafel;
    tafel.emplace_back(std::make_shared<Input::SeparatorComponent>("ALPHA"));
    tafel.emplace_back(std::make_shared<Input::RealComponent>("ALPHA"));
    tafel.emplace_back(std::make_shared<Input::SeparatorComponent>("I0"));
    tafel.emplace_back(std::make_shared<Input::RealComponent>("I0"));
    tafel.emplace_back(std::make_shared<Input::SeparatorComponent>("GAMMA"));
    tafel.emplace_back(std::make_shared<Input::RealComponent>("GAMMA"));
    tafel.emplace_back(std::make_shared<Input::SeparatorComponent>("REFCON"));
    tafel.emplace_back(std::make_shared<Input::RealComponent>("REFCON"));
    tafel.emplace_back(std::make_shared<Input::SeparatorComponent>("DL_SPEC_CAP"));
    tafel.emplace_back(std::make_shared<Input::RealComponent>("DL_SPEC_CAP"));
    reaction_model_choices.emplace(Inpar::ElCh::tafel, std::make_pair("Tafel", tafel));

    // linear kinetics
    std::vector<std::shared_ptr<Input::LineComponent>> linear;
    linear.emplace_back(std::make_shared<Input::SeparatorComponent>("ALPHA"));
    linear.emplace_back(std::make_shared<Input::RealComponent>("ALPHA"));
    linear.emplace_back(std::make_shared<Input::SeparatorComponent>("I0"));
    linear.emplace_back(std::make_shared<Input::RealComponent>("I0"));
    linear.emplace_back(std::make_shared<Input::SeparatorComponent>("GAMMA"));
    linear.emplace_back(std::make_shared<Input::RealComponent>("GAMMA"));
    linear.emplace_back(std::make_shared<Input::SeparatorComponent>("REFCON"));
    linear.emplace_back(std::make_shared<Input::RealComponent>("REFCON"));
    linear.emplace_back(std::make_shared<Input::SeparatorComponent>("DL_SPEC_CAP"));
    linear.emplace_back(std::make_shared<Input::RealComponent>("DL_SPEC_CAP"));
    reaction_model_choices.emplace(Inpar::ElCh::linear, std::make_pair("linear", linear));

    // Butler-Volmer-Newman: "Newman (book), 2004, p. 213, eq. 8.26"
    //                       "Wittmann (Bachelor thesis), 2011, p. 15, eq. 2.30"
    std::vector<std::shared_ptr<Input::LineComponent>> bvnewman;
    bvnewman.emplace_back(std::make_shared<Input::SeparatorComponent>("K_A"));
    bvnewman.emplace_back(std::make_shared<Input::RealComponent>("k_a"));
    bvnewman.emplace_back(std::make_shared<Input::SeparatorComponent>("K_C"));
    bvnewman.emplace_back(std::make_shared<Input::RealComponent>("k_c"));
    bvnewman.emplace_back(std::make_shared<Input::SeparatorComponent>("BETA"));
    bvnewman.emplace_back(std::make_shared<Input::RealComponent>("BETA"));
    bvnewman.emplace_back(std::make_shared<Input::SeparatorComponent>("DL_SPEC_CAP"));
    bvnewman.emplace_back(std::make_shared<Input::RealComponent>("DL_SPEC_CAP"));
    reaction_model_choices.emplace(
        Inpar::ElCh::butler_volmer_newman, std::make_pair("Butler-Volmer-Newman", bvnewman));

    // Butler-Volmer-Newman: "Bard (book), 2001, p. 99, eq. 3.4.10"
    //                       "Wittmann (Bachelor thesis), 2011, p. 16, eq. 2.32"
    std::vector<std::shared_ptr<Input::LineComponent>> bvbard;
    bvbard.emplace_back(std::make_shared<Input::SeparatorComponent>("E0"));
    bvbard.emplace_back(std::make_shared<Input::RealComponent>("e0"));
    bvbard.emplace_back(std::make_shared<Input::SeparatorComponent>("K0"));
    bvbard.emplace_back(std::make_shared<Input::RealComponent>("k0"));
    bvbard.emplace_back(std::make_shared<Input::SeparatorComponent>("BETA"));
    bvbard.emplace_back(std::make_shared<Input::RealComponent>("BETA"));
    bvbard.emplace_back(std::make_shared<Input::SeparatorComponent>("C_C0"));
    bvbard.emplace_back(std::make_shared<Input::RealComponent>("C_C0"));
    bvbard.emplace_back(std::make_shared<Input::SeparatorComponent>("C_A0"));
    bvbard.emplace_back(std::make_shared<Input::RealComponent>("C_A0"));
    bvbard.emplace_back(std::make_shared<Input::SeparatorComponent>("DL_SPEC_CAP"));
    bvbard.emplace_back(std::make_shared<Input::RealComponent>("DL_SPEC_CAP"));
    reaction_model_choices.emplace(
        Inpar::ElCh::butler_volmer_bard, std::make_pair("Butler-Volmer-Bard", bvbard));

    // Nernst equation:
    std::vector<std::shared_ptr<Input::LineComponent>> nernst;
    nernst.emplace_back(std::make_shared<Input::SeparatorComponent>("E0"));
    nernst.emplace_back(std::make_shared<Input::RealComponent>("e0"));
    nernst.emplace_back(std::make_shared<Input::SeparatorComponent>("C0"));
    nernst.emplace_back(std::make_shared<Input::RealComponent>("c0"));
    nernst.emplace_back(std::make_shared<Input::SeparatorComponent>("DL_SPEC_CAP"));
    nernst.emplace_back(std::make_shared<Input::RealComponent>("DL_SPEC_CAP"));
    reaction_model_choices.emplace(Inpar::ElCh::nernst, std::make_pair("Nernst", nernst));

    auto electrodeboundarykineticspoint = std::make_shared<Core::Conditions::ConditionDefinition>(
        "ELECTRODE BOUNDARY KINETICS POINT CONDITIONS", "ElchBoundaryKineticsPoint",
        "point electrode boundary kinetics", Core::Conditions::ElchBoundaryKinetics, false,
        Core::Conditions::geometry_type_point);

    auto electrodeboundarykineticsline = std::make_shared<Core::Conditions::ConditionDefinition>(
        "ELECTRODE BOUNDARY KINETICS LINE CONDITIONS", "ElchBoundaryKinetics",
        "line electrode boundary kinetics", Core::Conditions::ElchBoundaryKinetics, true,
        Core::Conditions::geometry_type_line);

    auto electrodeboundarykineticssurf = std::make_shared<Core::Conditions::ConditionDefinition>(
        "ELECTRODE BOUNDARY KINETICS SURF CONDITIONS", "ElchBoundaryKinetics",
        "surface electrode boundary kinetics", Core::Conditions::ElchBoundaryKinetics, true,
        Core::Conditions::geometry_type_surface);

    for (const auto& cond : {electrodeboundarykineticspoint, electrodeboundarykineticsline,
             electrodeboundarykineticssurf})
    {
      cond->add_component(std::make_shared<Input::SeparatorComponent>("ID"));
      cond->add_component(std::make_shared<Input::IntComponent>("ConditionID"));
      add_named_real(cond, "POT");
      add_named_int(cond, "FUNCT", "", 0, false);
      add_named_int(cond, "NUMSCAL");
      add_named_int_vector(cond, "STOICH", "", "NUMSCAL");
      add_named_int(cond, "E-");
      add_named_real(cond, "EPSILON",
          "porosity of electrode boundary, set to -1 if equal to porosity of electrolyte domain");
      add_named_int(cond, "ZERO_CUR");
      cond->add_component(std::make_shared<Input::SwitchComponent>(
          "KINETIC_MODEL", butler_volmer, reaction_model_choices));
      condlist.emplace_back(cond);
    }
  }

  /*--------------------------------------------------------------------*/
  // electrode kinetics as domain condition within electrolyte
  {
    // definition of line, surface, and volume conditions for electrode domain kinetics
    auto electrodedomainkineticsline = std::make_shared<Core::Conditions::ConditionDefinition>(
        "ELECTRODE DOMAIN KINETICS LINE CONDITIONS", "ElchDomainKinetics",
        "line electrode domain kinetics", Core::Conditions::ElchDomainKinetics, true,
        Core::Conditions::geometry_type_line);

    auto electrodedomainkineticssurf = std::make_shared<Core::Conditions::ConditionDefinition>(
        "ELECTRODE DOMAIN KINETICS SURF CONDITIONS", "ElchDomainKinetics",
        "surface electrode domain kinetics", Core::Conditions::ElchDomainKinetics, true,
        Core::Conditions::geometry_type_surface);

    auto electrodedomainkineticsvol = std::make_shared<Core::Conditions::ConditionDefinition>(
        "ELECTRODE DOMAIN KINETICS VOL CONDITIONS", "ElchDomainKinetics",
        "volume electrode domain kinetics", Core::Conditions::ElchDomainKinetics, true,
        Core::Conditions::geometry_type_volume);

    // equip condition definition with input file line components
    std::vector<std::shared_ptr<Input::LineComponent>> electrodedomainkineticscomponents;

    {
      electrodedomainkineticscomponents.emplace_back(
          std::make_shared<Input::SeparatorComponent>("ID"));
      electrodedomainkineticscomponents.emplace_back(
          std::make_shared<Input::IntComponent>("ConditionID"));
      electrodedomainkineticscomponents.emplace_back(
          std::make_shared<Input::SeparatorComponent>("POT"));
      electrodedomainkineticscomponents.emplace_back(std::make_shared<Input::RealComponent>("POT"));
      electrodedomainkineticscomponents.emplace_back(
          std::make_shared<Input::SeparatorComponent>("FUNCT"));
      electrodedomainkineticscomponents.emplace_back(
          std::make_shared<Input::IntComponent>("FUNCT", IntComponentData{0, false}));
      electrodedomainkineticscomponents.emplace_back(
          std::make_shared<Input::SeparatorComponent>("NUMSCAL"));

      electrodedomainkineticscomponents.emplace_back(
          std::make_shared<Input::IntComponent>("NUMSCAL"));
      electrodedomainkineticscomponents.emplace_back(
          std::make_shared<Input::SeparatorComponent>("STOICH"));
      electrodedomainkineticscomponents.emplace_back(
          std::make_shared<Input::IntVectorComponent>("STOICH", Input::LengthFromInt("NUMSCAL")));

      electrodedomainkineticscomponents.emplace_back(
          std::make_shared<Input::SeparatorComponent>("E-"));
      electrodedomainkineticscomponents.emplace_back(std::make_shared<Input::IntComponent>("E-"));
      electrodedomainkineticscomponents.emplace_back(
          std::make_shared<Input::SeparatorComponent>("ZERO_CUR"));
      electrodedomainkineticscomponents.emplace_back(
          std::make_shared<Input::IntComponent>("ZERO_CUR"));


      {
        // Butler-Volmer
        std::vector<std::shared_ptr<Input::LineComponent>> butlervolmer;
        butlervolmer.emplace_back(std::make_shared<Input::SeparatorComponent>(
            "A_S"));  // ratio of electrode-electrolyte interface area to total two-phase volume
        butlervolmer.emplace_back(std::make_shared<Input::RealComponent>("A_S"));
        butlervolmer.emplace_back(std::make_shared<Input::SeparatorComponent>("ALPHA_A"));
        butlervolmer.emplace_back(std::make_shared<Input::RealComponent>("ALPHA_A"));
        butlervolmer.emplace_back(std::make_shared<Input::SeparatorComponent>("ALPHA_C"));
        butlervolmer.emplace_back(std::make_shared<Input::RealComponent>("ALPHA_C"));
        butlervolmer.emplace_back(std::make_shared<Input::SeparatorComponent>("I0"));
        butlervolmer.emplace_back(std::make_shared<Input::RealComponent>("I0"));
        butlervolmer.emplace_back(std::make_shared<Input::SeparatorComponent>("GAMMA"));
        butlervolmer.emplace_back(std::make_shared<Input::RealComponent>("GAMMA"));
        butlervolmer.emplace_back(std::make_shared<Input::SeparatorComponent>("REFCON"));
        butlervolmer.emplace_back(std::make_shared<Input::RealComponent>("REFCON"));
        butlervolmer.emplace_back(std::make_shared<Input::SeparatorComponent>("DL_SPEC_CAP"));
        butlervolmer.emplace_back(std::make_shared<Input::RealComponent>("DL_SPEC_CAP"));
        butlervolmer.emplace_back(std::make_shared<Input::SeparatorComponent>("END"));

        electrodedomainkineticscomponents.emplace_back(std::shared_ptr<LineComponent>(
            new Input::SwitchComponent("KINETIC_MODEL", butler_volmer,
                {{butler_volmer, std::make_pair("Butler-Volmer", std::move(butlervolmer))}})));
      }
    }

    // insert input file line components into condition definitions
    for (auto& electrodedomainkineticscomponent : electrodedomainkineticscomponents)
    {
      electrodedomainkineticsline->add_component(electrodedomainkineticscomponent);
      electrodedomainkineticssurf->add_component(electrodedomainkineticscomponent);
      electrodedomainkineticsvol->add_component(electrodedomainkineticscomponent);
    }

    // insert condition definitions into global list of valid condition definitions
    condlist.emplace_back(electrodedomainkineticsline);
    condlist.emplace_back(electrodedomainkineticssurf);
    condlist.emplace_back(electrodedomainkineticsvol);
  }

  /*--------------------------------------------------------------------*/
  // boundary condition for constant-current constant-voltage (CCCV) cell cycling

  // definition of point, line and surface conditions for CCCV cell cycling
  auto cccvcyclingpoint = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN CCCV CELL CYCLING POINT CONDITIONS", "CCCVCycling",
      "line boundary condition for constant-current constant-voltage (CCCV) cell cycling",
      Core::Conditions::CCCVCycling, true, Core::Conditions::geometry_type_point);

  auto cccvcyclingline = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN CCCV CELL CYCLING LINE CONDITIONS", "CCCVCycling",
      "line boundary condition for constant-current constant-voltage (CCCV) cell cycling",
      Core::Conditions::CCCVCycling, true, Core::Conditions::geometry_type_line);

  auto cccvcyclingsurf = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN CCCV CELL CYCLING SURF CONDITIONS", "CCCVCycling",
      "surface boundary condition for constant-current constant-voltage (CCCV) cell cycling",
      Core::Conditions::CCCVCycling, true, Core::Conditions::geometry_type_surface);

  for (const auto& cond : {cccvcyclingpoint, cccvcyclingline, cccvcyclingsurf})
  {
    // insert input file line components into condition definitions
    {
      add_named_int(cond, "NUMBER_OF_HALF_CYCLES");
      add_named_int(
          cond, "BEGIN_WITH_CHARGING");  // Boolean parameter represented by integer parameter
      add_named_int(cond, "CONDITION_ID_FOR_CHARGE", "", 0, false);
      add_named_int(cond, "CONDITION_ID_FOR_DISCHARGE", "", 0, false);
      add_named_real(cond, "INIT_RELAX_TIME");
      add_named_int(cond, "ADAPTIVE_TIME_STEPPING_INIT_RELAX");
      add_named_int(cond, "NUM_ADD_ADAPT_TIME_STEPS", "", 0, false);
      add_named_int(cond, "MIN_TIME_STEPS_DURING_INIT_RELAX", "", 0, false);

      // insert condition definitions into global list of valid condition definitions
      condlist.emplace_back(cond);
    }
  }

  /*--------------------------------------------------------------------*/
  // boundary condition for constant-current constant-voltage (CCCV) half-cycle

  // definition of point, line and surface conditions for CCCV half-cycle
  auto cccvhalfcyclepoint = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN CCCV HALF-CYCLE POINT CONDITIONS", "CCCVHalfCycle",
      "line boundary condition for constant-current constant-voltage (CCCV) half-cycle",
      Core::Conditions::CCCVHalfCycle, true, Core::Conditions::geometry_type_point);

  auto cccvhalfcycleline = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN CCCV HALF-CYCLE LINE CONDITIONS", "CCCVHalfCycle",
      "line boundary condition for constant-current constant-voltage (CCCV) half-cycle",
      Core::Conditions::CCCVHalfCycle, true, Core::Conditions::geometry_type_line);

  auto cccvhalfcyclesurf = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN CCCV HALF-CYCLE SURF CONDITIONS", "CCCVHalfCycle",
      "surface boundary condition for constant-current constant-voltage (CCCV) half-cycle",
      Core::Conditions::CCCVHalfCycle, true, Core::Conditions::geometry_type_surface);

  for (const auto& cond : {cccvhalfcyclepoint, cccvhalfcycleline, cccvhalfcyclesurf})
  {
    // insert input file line components into condition definitions
    cond->add_component(std::make_shared<Input::SeparatorComponent>("ID"));
    cond->add_component(std::make_shared<Input::IntComponent>("ConditionID"));
    add_named_real(cond, "CURRENT");
    add_named_real(cond, "CUT_OFF_VOLTAGE");
    add_named_real(cond, "CUT_OFF_C_RATE");
    add_named_real(cond, "RELAX_TIME");
    // switch adaptive time stepping on for different phases of half cycle: 1st: end of constant
    // current, 2nd: end of constant voltage, 3rd: end of relaxation
    add_named_int_vector(cond, "ADAPTIVE_TIME_STEPPING_PHASE_ON_OFF", "", 3);

    // insert condition definitions into global list of valid condition definitions
    condlist.emplace_back(cond);
  }
}

FOUR_C_NAMESPACE_CLOSE
