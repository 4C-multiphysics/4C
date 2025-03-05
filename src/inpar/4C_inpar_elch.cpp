// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_elch.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Inpar::ElCh::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs elchcontrol{"ELCH CONTROL"};

  Core::Utils::int_parameter("MOVBOUNDARYITEMAX", 10,
      "Maximum number of outer iterations in electrode shape change computations", elchcontrol);
  elchcontrol.specs.emplace_back(parameter<double>("MOVBOUNDARYCONVTOL",
      {.description =
              "Convergence check tolerance for outer loop in electrode shape change computations",
          .default_value = 1e-6}));
  elchcontrol.specs.emplace_back(parameter<double>(
      "TEMPERATURE", {.description = "Constant temperature (Kelvin)", .default_value = 298.0}));
  Core::Utils::int_parameter("TEMPERATURE_FROM_FUNCT", -1,
      "Homogeneous temperature within electrochemistry field that can be time dependent according "
      "to function definition",
      elchcontrol);
  elchcontrol.specs.emplace_back(parameter<double>("FARADAY_CONSTANT",
      {.description = "Faraday constant (in unit system as chosen in input file)",
          .default_value = 9.64853399e4}));
  elchcontrol.specs.emplace_back(parameter<double>("GAS_CONSTANT",
      {.description = "(universal) gas constant (in unit system as chosen in input file)",
          .default_value = 8.314472}));
  // parameter for possible types of ELCH algorithms for deforming meshes
  Core::Utils::string_to_integral_parameter<Inpar::ElCh::ElchMovingBoundary>("MOVINGBOUNDARY", "No",
      "ELCH algorithm for deforming meshes",
      tuple<std::string>("No", "pseudo-transient", "fully-transient"),
      tuple<Inpar::ElCh::ElchMovingBoundary>(
          elch_mov_bndry_no, elch_mov_bndry_pseudo_transient, elch_mov_bndry_fully_transient),
      elchcontrol);
  elchcontrol.specs.emplace_back(parameter<double>(
      "MOLARVOLUME", {.description = "Molar volume for electrode shape change computations",
                         .default_value = 0.0}));
  elchcontrol.specs.emplace_back(parameter<double>("MOVBOUNDARYTHETA",
      {.description = "One-step-theta factor in electrode shape change computations",
          .default_value = 0.0}));
  elchcontrol.specs.emplace_back(parameter<bool>(
      "GALVANOSTATIC", {.description = "flag for galvanostatic mode", .default_value = false}));
  Core::Utils::string_to_integral_parameter<Inpar::ElCh::ApproxElectResist>(
      "GSTAT_APPROX_ELECT_RESIST", "relation_pot_cur", "relation of potential and current flow",
      tuple<std::string>("relation_pot_cur", "effective_length_with_initial_cond",
          "effective_length_with_integrated_cond"),
      tuple<Inpar::ElCh::ApproxElectResist>(approxelctresist_relpotcur,
          approxelctresist_effleninitcond, approxelctresist_efflenintegcond),
      elchcontrol);
  Core::Utils::int_parameter(
      "GSTATCONDID_CATHODE", 0, "condition id of electrode kinetics for cathode", elchcontrol);
  Core::Utils::int_parameter(
      "GSTATCONDID_ANODE", 1, "condition id of electrode kinetics for anode", elchcontrol);
  elchcontrol.specs.emplace_back(parameter<double>(
      "GSTATCONVTOL", {.description = "Convergence check tolerance for galvanostatic mode",
                          .default_value = 1.e-5}));
  elchcontrol.specs.emplace_back(parameter<double>(
      "GSTATCURTOL", {.description = "Current Tolerance", .default_value = 1.e-15}));
  Core::Utils::int_parameter(
      "GSTATFUNCTNO", -1, "function number defining the imposed current curve", elchcontrol);
  Core::Utils::int_parameter(
      "GSTATITEMAX", 10, "maximum number of iterations for galvanostatic mode", elchcontrol);
  elchcontrol.specs.emplace_back(parameter<double>("GSTAT_LENGTH_CURRENTPATH",
      {.description = "average length of the current path", .default_value = 0.0}));

  Core::Utils::string_to_integral_parameter<Inpar::ElCh::EquPot>("EQUPOT", "Undefined",
      "type of closing equation for electric potential",
      tuple<std::string>(
          "Undefined", "ENC", "ENC_PDE", "ENC_PDE_ELIM", "Poisson", "Laplace", "divi"),
      tuple<Inpar::ElCh::EquPot>(equpot_undefined, equpot_enc, equpot_enc_pde, equpot_enc_pde_elim,
          equpot_poisson, equpot_laplace, equpot_divi),
      elchcontrol);
  elchcontrol.specs.emplace_back(parameter<bool>("DIFFCOND_FORMULATION",
      {.description = "Activation of diffusion-conduction formulation", .default_value = false}));
  elchcontrol.specs.emplace_back(parameter<bool>(
      "INITPOTCALC", {.description = "Automatically calculate initial field for electric potential",
                         .default_value = false}));
  elchcontrol.specs.emplace_back(parameter<bool>("ONLYPOTENTIAL",
      {.description = "Coupling of general ion transport equation with Laplace equation",
          .default_value = false}));
  elchcontrol.specs.emplace_back(parameter<bool>("COUPLE_BOUNDARY_FLUXES",
      {.description = "Coupling of lithium-ion flux density and electric current density at "
                      "Dirichlet and Neumann boundaries",
          .default_value = true}));
  elchcontrol.specs.emplace_back(parameter<double>("CYCLING_TIMESTEP",
      {.description = "modified time step size for CCCV cell cycling", .default_value = -1.}));
  elchcontrol.specs.emplace_back(parameter<bool>("ELECTRODE_INFO_EVERY_STEP",
      {.description = "the cell voltage, SOC, and C-Rate will be written to the csv file every "
                      "step, even if RESULTSEVERY is not 1",
          .default_value = false}));

  elchcontrol.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  // attention: this list is a sublist of elchcontrol
  Core::Utils::SectionSpecs elchdiffcondcontrol{elchcontrol, "DIFFCOND"};

  elchdiffcondcontrol.specs.emplace_back(parameter<bool>("CURRENT_SOLUTION_VAR",
      {.description = "Current as a solution variable", .default_value = false}));
  elchdiffcondcontrol.specs.emplace_back(parameter<bool>("MAT_DIFFCOND_DIFFBASED",
      {.description =
              "Coupling terms of chemical diffusion for current equation are based on t and kappa",
          .default_value = true}));

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
  elchdiffcondcontrol.specs.emplace_back(parameter<double>("MAT_NEWMAN_CONST_A",
      {.description = "Constant A for the Newman model(term for the concentration overpotential)",
          .default_value = 2.0}));
  elchdiffcondcontrol.specs.emplace_back(parameter<double>("MAT_NEWMAN_CONST_B",
      {.description = "Constant B for the Newman model(term for the concentration overpotential)",
          .default_value = -2.0}));
  elchdiffcondcontrol.specs.emplace_back(parameter<double>("MAT_NEWMAN_CONST_C",
      {.description = "Constant C for the Newman model(term for the concentration overpotential)",
          .default_value = -1.0}));
  elchdiffcondcontrol.specs.emplace_back(parameter<double>("PERMITTIVITY_VACUUM",
      {.description = "Vacuum permittivity", .default_value = 8.8541878128e-12}));

  elchdiffcondcontrol.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  // sublist for space-charge layers
  Core::Utils::SectionSpecs sclcontrol{elchcontrol, "SCL"};

  sclcontrol.specs.emplace_back(parameter<bool>("ADD_MICRO_MACRO_COUPLING",
      {.description = "flag for micro macro coupling with scls", .default_value = false}));
  sclcontrol.specs.emplace_back(parameter<bool>(
      "COUPLING_OUTPUT", {.description = "write coupled node gids and node coordinates to csv file",
                             .default_value = false}));
  sclcontrol.specs.emplace_back(parameter<bool>("INITPOTCALC",
      {.description = "calculate initial potential field?", .default_value = false}));
  Core::Utils::int_parameter("SOLVER", -1, "solver for coupled SCL problem", sclcontrol);
  Core::Utils::string_to_integral_parameter<Core::LinAlg::MatrixType>("MATRIXTYPE", "undefined",
      "type of global system matrix in global system of equations",
      tuple<std::string>("undefined", "block", "sparse"),
      tuple<Core::LinAlg::MatrixType>(Core::LinAlg::MatrixType::undefined,
          Core::LinAlg::MatrixType::block_field, Core::LinAlg::MatrixType::sparse),
      sclcontrol);
  Core::Utils::int_parameter("ADAPT_TIME_STEP", -1,
      "time step when time step size should be updated to 'ADAPTED_TIME_STEP_SIZE'.", sclcontrol);
  sclcontrol.specs.emplace_back(parameter<double>(
      "ADAPTED_TIME_STEP_SIZE", {.description = "new time step size.", .default_value = -1.0}));

  Core::Utils::string_to_integral_parameter<ScaTra::InitialField>("INITIALFIELD", "zero_field",
      "Initial Field for scalar transport problem",
      tuple<std::string>("zero_field", "field_by_function", "field_by_condition"),
      tuple<ScaTra::InitialField>(ScaTra::initfield_zero_field, ScaTra::initfield_field_by_function,
          ScaTra::initfield_field_by_condition),
      sclcontrol);

  Core::Utils::int_parameter(
      "INITFUNCNO", -1, "function number for scalar transport initial field", sclcontrol);

  sclcontrol.move_into_collection(list);
}


void Inpar::ElCh::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // electrode state of charge
  {
    // definition of electrode state of charge surface and volume conditions
    Core::Conditions::ConditionDefinition electrodesocline(
        "DESIGN ELECTRODE STATE OF CHARGE LINE CONDITIONS", "ElectrodeSOC",
        "electrode state of charge line condition", Core::Conditions::ElectrodeSOC, true,
        Core::Conditions::geometry_type_line);

    Core::Conditions::ConditionDefinition electrodesocsurf(
        "DESIGN ELECTRODE STATE OF CHARGE SURF CONDITIONS", "ElectrodeSOC",
        "electrode state of charge surface condition", Core::Conditions::ElectrodeSOC, true,
        Core::Conditions::geometry_type_surface);
    Core::Conditions::ConditionDefinition electrodesocvol(
        "DESIGN ELECTRODE STATE OF CHARGE VOL CONDITIONS", "ElectrodeSOC",
        "electrode state of charge volume condition", Core::Conditions::ElectrodeSOC, true,
        Core::Conditions::geometry_type_volume);

    const auto make_electrodesoc = [&condlist](Core::Conditions::ConditionDefinition& cond)
    {
      // insert input file line components into condition definitions
      cond.add_component(parameter<int>("ConditionID"));
      cond.add_component(parameter<double>("C_0%"));
      cond.add_component(parameter<double>("C_100%"));
      cond.add_component(parameter<double>("ONE_HOUR"));

      // insert condition definitions into global list of valid condition definitions
      condlist.emplace_back(cond);
    };

    make_electrodesoc(electrodesocline);
    make_electrodesoc(electrodesocsurf);
    make_electrodesoc(electrodesocvol);
  }

  /*--------------------------------------------------------------------*/
  // cell voltage

  {
    // definition of cell voltage point, line, and surface conditions
    Core::Conditions::ConditionDefinition cellvoltagepoint("DESIGN CELL VOLTAGE POINT CONDITIONS",
        "CellVoltagePoint", "cell voltage point condition", Core::Conditions::CellVoltage, false,
        Core::Conditions::geometry_type_point);

    Core::Conditions::ConditionDefinition cellvoltageline("DESIGN CELL VOLTAGE LINE CONDITIONS",
        "CellVoltage", "cell voltage line condition", Core::Conditions::CellVoltage, true,
        Core::Conditions::geometry_type_line);

    Core::Conditions::ConditionDefinition cellvoltagesurf("DESIGN CELL VOLTAGE SURF CONDITIONS",
        "CellVoltage", "cell voltage surface condition", Core::Conditions::CellVoltage, true,
        Core::Conditions::geometry_type_surface);

    const auto make_cellvoltage = [&condlist](Core::Conditions::ConditionDefinition& cond)
    {
      // insert input file line components into condition definitions
      cond.add_component(parameter<int>("ConditionID"));

      // insert condition definitions into global list of valid condition definitions
      condlist.emplace_back(cond);
    };

    make_cellvoltage(cellvoltagepoint);
    make_cellvoltage(cellvoltageline);
    make_cellvoltage(cellvoltagesurf);
  }


  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // electrode kinetics as boundary condition on electrolyte
  {
    auto reaction_model_choices = one_of({
        all_of({
            selection<int>("KINETIC_MODEL",
                {
                    {"Butler-Volmer", Inpar::ElCh::ElectrodeKinetics::butler_volmer},
                    {"Butler-Volmer-Yang1997",
                        Inpar::ElCh::ElectrodeKinetics::butler_volmer_yang1997},
                }),
            parameter<double>("ALPHA_A"),
            parameter<double>("ALPHA_C"),
            parameter<double>("I0"),
            parameter<double>("GAMMA"),
            parameter<double>("REFCON"),
            parameter<double>("DL_SPEC_CAP"),
        }),
        all_of({
            selection<int>("KINETIC_MODEL", {{"Tafel", Inpar::ElCh::ElectrodeKinetics::tafel}}),
            parameter<double>("ALPHA"),
            parameter<double>("I0"),
            parameter<double>("GAMMA"),
            parameter<double>("REFCON"),
            parameter<double>("DL_SPEC_CAP"),
        }),
        all_of({
            selection<int>("KINETIC_MODEL", {{"linear", Inpar::ElCh::ElectrodeKinetics::linear}}),
            parameter<double>("ALPHA"),
            parameter<double>("I0"),
            parameter<double>("GAMMA"),
            parameter<double>("REFCON"),
            parameter<double>("DL_SPEC_CAP"),
        }),
        all_of({
            selection<int>("KINETIC_MODEL",
                {{"Butler-Volmer-Newman", Inpar::ElCh::ElectrodeKinetics::butler_volmer_newman}}),
            parameter<double>("K_A"),
            parameter<double>("K_C"),
            parameter<double>("BETA"),
            parameter<double>("DL_SPEC_CAP"),
        }),
        all_of({
            selection<int>("KINETIC_MODEL",
                {{"Butler-Volmer-Bard", Inpar::ElCh::ElectrodeKinetics::butler_volmer_bard}}),
            parameter<double>("E0"),
            parameter<double>("K0"),
            parameter<double>("BETA"),
            parameter<double>("C_C0"),
            parameter<double>("C_A0"),
            parameter<double>("DL_SPEC_CAP"),
        }),
        all_of({
            selection<int>("KINETIC_MODEL", {{"Nernst", Inpar::ElCh::ElectrodeKinetics::nernst}}),
            parameter<double>("E0"),
            parameter<double>("C0"),
            parameter<double>("DL_SPEC_CAP"),
        }),
    });


    Core::Conditions::ConditionDefinition electrodeboundarykineticspoint(
        "ELECTRODE BOUNDARY KINETICS POINT CONDITIONS", "ElchBoundaryKineticsPoint",
        "point electrode boundary kinetics", Core::Conditions::ElchBoundaryKinetics, false,
        Core::Conditions::geometry_type_point);

    Core::Conditions::ConditionDefinition electrodeboundarykineticsline(
        "ELECTRODE BOUNDARY KINETICS LINE CONDITIONS", "ElchBoundaryKinetics",
        "line electrode boundary kinetics", Core::Conditions::ElchBoundaryKinetics, true,
        Core::Conditions::geometry_type_line);

    Core::Conditions::ConditionDefinition electrodeboundarykineticssurf(
        "ELECTRODE BOUNDARY KINETICS SURF CONDITIONS", "ElchBoundaryKinetics",
        "surface electrode boundary kinetics", Core::Conditions::ElchBoundaryKinetics, true,
        Core::Conditions::geometry_type_surface);

    const auto make_electrodeboundarykinetics = [&condlist, &reaction_model_choices](
                                                    Core::Conditions::ConditionDefinition& cond)
    {
      cond.add_component(parameter<int>("ConditionID"));
      cond.add_component(parameter<double>("POT"));
      cond.add_component(parameter<std::optional<int>>("FUNCT", {.description = ""}));
      cond.add_component(parameter<int>("NUMSCAL"));
      cond.add_component(parameter<std::vector<int>>(
          "STOICH", {.description = "", .size = from_parameter<int>("NUMSCAL")}));
      cond.add_component(parameter<int>("E-"));
      cond.add_component(parameter<double>(
          "EPSILON", {.description = "porosity of electrode boundary, set to -1 if "
                                     "equal to porosity of electrolyte domain"}));
      cond.add_component(parameter<int>("ZERO_CUR"));
      cond.add_component(reaction_model_choices);
      condlist.emplace_back(cond);
    };

    make_electrodeboundarykinetics(electrodeboundarykineticspoint);
    make_electrodeboundarykinetics(electrodeboundarykineticsline);
    make_electrodeboundarykinetics(electrodeboundarykineticssurf);
  }

  /*--------------------------------------------------------------------*/
  // electrode kinetics as domain condition within electrolyte
  {
    // definition of line, surface, and volume conditions for electrode domain kinetics
    Core::Conditions::ConditionDefinition electrodedomainkineticsline(
        "ELECTRODE DOMAIN KINETICS LINE CONDITIONS", "ElchDomainKinetics",
        "line electrode domain kinetics", Core::Conditions::ElchDomainKinetics, true,
        Core::Conditions::geometry_type_line);

    Core::Conditions::ConditionDefinition electrodedomainkineticssurf(
        "ELECTRODE DOMAIN KINETICS SURF CONDITIONS", "ElchDomainKinetics",
        "surface electrode domain kinetics", Core::Conditions::ElchDomainKinetics, true,
        Core::Conditions::geometry_type_surface);

    Core::Conditions::ConditionDefinition electrodedomainkineticsvol(
        "ELECTRODE DOMAIN KINETICS VOL CONDITIONS", "ElchDomainKinetics",
        "volume electrode domain kinetics", Core::Conditions::ElchDomainKinetics, true,
        Core::Conditions::geometry_type_volume);

    // equip condition definition with input file line components
    auto electrodedomainkineticscomponents = all_of({
        parameter<int>("ConditionID"),
        parameter<double>("POT"),
        parameter<std::optional<int>>("FUNCT"),
        parameter<int>("NUMSCAL"),
        parameter<std::vector<int>>("STOICH", {.size = from_parameter<int>("NUMSCAL")}),
        parameter<int>("E-"),
        parameter<int>("ZERO_CUR"),
        selection<int>("KINETIC_MODEL",
            {
                {"Butler-Volmer", Inpar::ElCh::ElectrodeKinetics::butler_volmer},
            }),
        parameter<double>("A_S"),
        parameter<double>("ALPHA_A"),
        parameter<double>("ALPHA_C"),
        parameter<double>("I0"),
        parameter<double>("GAMMA"),
        parameter<double>("REFCON"),
        parameter<double>("DL_SPEC_CAP"),
    });

    {
      electrodedomainkineticsline.add_component(electrodedomainkineticscomponents);
      electrodedomainkineticssurf.add_component(electrodedomainkineticscomponents);
      electrodedomainkineticsvol.add_component(electrodedomainkineticscomponents);
    }

    // insert condition definitions into global list of valid condition definitions
    condlist.emplace_back(electrodedomainkineticsline);
    condlist.emplace_back(electrodedomainkineticssurf);
    condlist.emplace_back(electrodedomainkineticsvol);
  }

  /*--------------------------------------------------------------------*/
  // boundary condition for constant-current constant-voltage (CCCV) cell cycling

  // definition of point, line and surface conditions for CCCV cell cycling
  Core::Conditions::ConditionDefinition cccvcyclingpoint(
      "DESIGN CCCV CELL CYCLING POINT CONDITIONS", "CCCVCycling",
      "line boundary condition for constant-current constant-voltage (CCCV) cell cycling",
      Core::Conditions::CCCVCycling, true, Core::Conditions::geometry_type_point);

  Core::Conditions::ConditionDefinition cccvcyclingline("DESIGN CCCV CELL CYCLING LINE CONDITIONS",
      "CCCVCycling",
      "line boundary condition for constant-current constant-voltage (CCCV) cell cycling",
      Core::Conditions::CCCVCycling, true, Core::Conditions::geometry_type_line);

  Core::Conditions::ConditionDefinition cccvcyclingsurf("DESIGN CCCV CELL CYCLING SURF CONDITIONS",
      "CCCVCycling",
      "surface boundary condition for constant-current constant-voltage (CCCV) cell cycling",
      Core::Conditions::CCCVCycling, true, Core::Conditions::geometry_type_surface);

  const auto make_cccvcycling = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    // insert input file line components into condition definitions
    {
      cond.add_component(parameter<int>("NUMBER_OF_HALF_CYCLES"));
      cond.add_component(parameter<int>(
          "BEGIN_WITH_CHARGING"));  // Boolean parameter represented by integer parameter
      cond.add_component(
          parameter<std::optional<int>>("CONDITION_ID_FOR_CHARGE", {.description = ""}));
      cond.add_component(
          parameter<std::optional<int>>("CONDITION_ID_FOR_DISCHARGE", {.description = ""}));
      cond.add_component(parameter<double>("INIT_RELAX_TIME"));
      cond.add_component(parameter<int>("ADAPTIVE_TIME_STEPPING_INIT_RELAX"));
      cond.add_component(
          parameter<std::optional<int>>("NUM_ADD_ADAPT_TIME_STEPS", {.description = ""}));
      cond.add_component(
          parameter<std::optional<int>>("MIN_TIME_STEPS_DURING_INIT_RELAX", {.description = ""}));

      // insert condition definitions into global list of valid condition definitions
      condlist.emplace_back(cond);
    }
  };

  make_cccvcycling(cccvcyclingpoint);
  make_cccvcycling(cccvcyclingline);
  make_cccvcycling(cccvcyclingsurf);

  /*--------------------------------------------------------------------*/
  // boundary condition for constant-current constant-voltage (CCCV) half-cycle

  // definition of point, line and surface conditions for CCCV half-cycle
  Core::Conditions::ConditionDefinition cccvhalfcyclepoint(
      "DESIGN CCCV HALF-CYCLE POINT CONDITIONS", "CCCVHalfCycle",
      "line boundary condition for constant-current constant-voltage (CCCV) half-cycle",
      Core::Conditions::CCCVHalfCycle, true, Core::Conditions::geometry_type_point);

  Core::Conditions::ConditionDefinition cccvhalfcycleline("DESIGN CCCV HALF-CYCLE LINE CONDITIONS",
      "CCCVHalfCycle",
      "line boundary condition for constant-current constant-voltage (CCCV) half-cycle",
      Core::Conditions::CCCVHalfCycle, true, Core::Conditions::geometry_type_line);

  Core::Conditions::ConditionDefinition cccvhalfcyclesurf("DESIGN CCCV HALF-CYCLE SURF CONDITIONS",
      "CCCVHalfCycle",
      "surface boundary condition for constant-current constant-voltage (CCCV) half-cycle",
      Core::Conditions::CCCVHalfCycle, true, Core::Conditions::geometry_type_surface);

  const auto make_cccvhalfcycle = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    // insert input file line components into condition definitions
    cond.add_component(parameter<int>("ConditionID"));
    cond.add_component(parameter<double>("CURRENT"));
    cond.add_component(parameter<double>("CUT_OFF_VOLTAGE"));
    cond.add_component(parameter<double>("CUT_OFF_C_RATE"));
    cond.add_component(parameter<double>("RELAX_TIME"));
    // switch adaptive time stepping on for different phases of half cycle: 1st: end of constant
    // current, 2nd: end of constant voltage, 3rd: end of relaxation
    cond.add_component(parameter<std::vector<int>>(
        "ADAPTIVE_TIME_STEPPING_PHASE_ON_OFF", {.description = "", .size = 3}));

    // insert condition definitions into global list of valid condition definitions
    condlist.emplace_back(cond);
  };

  make_cccvhalfcycle(cccvhalfcyclepoint);
  make_cccvhalfcycle(cccvhalfcycleline);
  make_cccvhalfcycle(cccvhalfcyclesurf);
}

FOUR_C_NAMESPACE_CLOSE
