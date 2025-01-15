// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_ssi.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_s2i.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_io_linecomponent.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

void Inpar::SSI::set_valid_parameters(Teuchos::ParameterList& list)
{
  using namespace Input;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& ssidyn =
      list.sublist("SSI CONTROL", false, "Control parameters for scatra structure interaction");

  // Output type
  Core::Utils::double_parameter(
      "RESTARTEVERYTIME", 0, "write restart possibility every RESTARTEVERY steps", &ssidyn);
  Core::Utils::int_parameter(
      "RESTARTEVERY", 1, "write restart possibility every RESTARTEVERY steps", &ssidyn);
  // Time loop control
  Core::Utils::int_parameter("NUMSTEP", 200, "maximum number of Timesteps", &ssidyn);
  Core::Utils::double_parameter("MAXTIME", 1000.0, "total simulation time", &ssidyn);
  Core::Utils::double_parameter("TIMESTEP", -1, "time step size dt", &ssidyn);
  Core::Utils::bool_parameter(
      "DIFFTIMESTEPSIZE", "No", "use different step size for scatra and solid", &ssidyn);
  Core::Utils::double_parameter("RESULTSEVERYTIME", 0, "increment for writing solution", &ssidyn);
  Core::Utils::int_parameter("RESULTSEVERY", 1, "increment for writing solution", &ssidyn);
  Core::Utils::int_parameter("ITEMAX", 10, "maximum number of iterations over fields", &ssidyn);
  Core::Utils::bool_parameter("SCATRA_FROM_RESTART_FILE", "No",
      "read scatra result from restart files (use option 'restartfromfile' during execution of "
      "4C)",
      &ssidyn);
  Core::Utils::string_parameter(
      "SCATRA_FILENAME", "nil", "Control-file name for reading scatra results in SSI", &ssidyn);

  // Type of coupling strategy between the two fields
  setStringToIntegralParameter<FieldCoupling>("FIELDCOUPLING", "volume_matching",
      "Type of coupling strategy between fields",
      tuple<std::string>("volume_matching", "volume_nonmatching", "boundary_nonmatching",
          "volumeboundary_matching"),
      tuple<FieldCoupling>(FieldCoupling::volume_match, FieldCoupling::volume_nonmatch,
          FieldCoupling::boundary_nonmatch, FieldCoupling::volumeboundary_match),
      &ssidyn);

  // Coupling strategy for SSI solvers
  setStringToIntegralParameter<SolutionSchemeOverFields>("COUPALGO", "ssi_IterStagg",
      "Coupling strategies for SSI solvers",
      tuple<std::string>("ssi_OneWay_ScatraToSolid", "ssi_OneWay_SolidToScatra",
          //                                "ssi_SequStagg_ScatraToSolid",
          //                                "ssi_SequStagg_SolidToScatra",
          "ssi_IterStagg", "ssi_IterStaggFixedRel_ScatraToSolid",
          "ssi_IterStaggFixedRel_SolidToScatra", "ssi_IterStaggAitken_ScatraToSolid",
          "ssi_IterStaggAitken_SolidToScatra", "ssi_Monolithic"),
      tuple<SolutionSchemeOverFields>(SolutionSchemeOverFields::ssi_OneWay_ScatraToSolid,
          SolutionSchemeOverFields::ssi_OneWay_SolidToScatra,
          //                                ssi_SequStagg_ScatraToSolid,
          //                                ssi_SequStagg_SolidToScatra,
          SolutionSchemeOverFields::ssi_IterStagg,
          SolutionSchemeOverFields::ssi_IterStaggFixedRel_ScatraToSolid,
          SolutionSchemeOverFields::ssi_IterStaggFixedRel_SolidToScatra,
          SolutionSchemeOverFields::ssi_IterStaggAitken_ScatraToSolid,
          SolutionSchemeOverFields::ssi_IterStaggAitken_SolidToScatra,
          SolutionSchemeOverFields::ssi_Monolithic),
      &ssidyn);

  // type of scalar transport time integration
  setStringToIntegralParameter<ScaTraTimIntType>("SCATRATIMINTTYPE", "Standard",
      "scalar transport time integration type is needed to instantiate correct scalar transport "
      "time integration scheme for ssi problems",
      tuple<std::string>("Standard", "Cardiac_Monodomain", "Elch"),
      tuple<ScaTraTimIntType>(
          ScaTraTimIntType::standard, ScaTraTimIntType::cardiac_monodomain, ScaTraTimIntType::elch),
      &ssidyn);

  // Restart from Structure problem instead of SSI
  Core::Utils::bool_parameter("RESTART_FROM_STRUCTURE", "no",
      "restart from structure problem (e.g. from prestress calculations) instead of ssi", &ssidyn);

  // Adaptive time stepping
  Core::Utils::bool_parameter(
      "ADAPTIVE_TIMESTEPPING", "no", "flag for adaptive time stepping", &ssidyn);

  // do redistribution by binning of solid mechanics discretization (scatra dis is cloned from solid
  // dis for volume_matching and volumeboundary_matching)
  Core::Utils::bool_parameter("REDISTRIBUTE_SOLID", "No",
      "redistribution by binning of solid mechanics discretization", &ssidyn);

  /*----------------------------------------------------------------------*/
  /* parameters for partitioned SSI */
  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& ssidynpart = ssidyn.sublist("PARTITIONED", false,
      "Partitioned Structure Scalar Interaction\n"
      "Control section for partitioned SSI");

  // Solver parameter for relaxation of iterative staggered partitioned SSI
  Core::Utils::double_parameter(
      "MAXOMEGA", 10.0, "largest omega allowed for Aitken relaxation", &ssidynpart);
  Core::Utils::double_parameter(
      "MINOMEGA", 0.1, "smallest omega allowed for Aitken relaxation", &ssidynpart);
  Core::Utils::double_parameter("STARTOMEGA", 1.0, "fixed relaxation parameter", &ssidynpart);

  // convergence tolerance of outer iteration loop
  Core::Utils::double_parameter("CONVTOL", 1e-6,
      "tolerance for convergence check of outer iteration within partitioned SSI", &ssidynpart);

  /*----------------------------------------------------------------------*/
  /* parameters for monolithic SSI */
  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& ssidynmono = ssidyn.sublist("MONOLITHIC", false,
      "Monolithic Structure Scalar Interaction\n"
      "Control section for monolithic SSI");

  // convergence tolerances of Newton-Raphson iteration loop
  Core::Utils::double_parameter("ABSTOLRES", 1.e-14,
      "absolute tolerance for deciding if global residual of nonlinear problem is already zero",
      &ssidynmono);
  Core::Utils::double_parameter("CONVTOL", 1.e-6,
      "tolerance for convergence check of Newton-Raphson iteration within monolithic SSI",
      &ssidynmono);

  // ID of linear solver for global system of equations
  Core::Utils::int_parameter(
      "LINEAR_SOLVER", -1, "ID of linear solver for global system of equations", &ssidynmono);

  // type of global system matrix in global system of equations
  setStringToIntegralParameter<Core::LinAlg::MatrixType>("MATRIXTYPE", "undefined",
      "type of global system matrix in global system of equations",
      tuple<std::string>("undefined", "block", "sparse"),
      tuple<Core::LinAlg::MatrixType>(Core::LinAlg::MatrixType::undefined,
          Core::LinAlg::MatrixType::block_field, Core::LinAlg::MatrixType::sparse),
      &ssidynmono);

  setStringToIntegralParameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION", "none",
      "flag for equilibration of global system of equations",
      tuple<std::string>("none", "rows_full", "rows_maindiag", "columns_full", "columns_maindiag",
          "rowsandcolumns_full", "rowsandcolumns_maindiag", "local"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_full,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_full,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_full,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag,
          Core::LinAlg::EquilibrationMethod::local),
      &ssidynmono);

  setStringToIntegralParameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION_STRUCTURE", "none",
      "flag for equilibration of structural equations",
      tuple<std::string>(
          "none", "rows_maindiag", "columns_maindiag", "rowsandcolumns_maindiag", "symmetry"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag,
          Core::LinAlg::EquilibrationMethod::symmetry),
      &ssidynmono);

  setStringToIntegralParameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION_SCATRA", "none",
      "flag for equilibration of scatra equations",
      tuple<std::string>(
          "none", "rows_maindiag", "columns_maindiag", "rowsandcolumns_maindiag", "symmetry"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag,
          Core::LinAlg::EquilibrationMethod::symmetry),
      &ssidynmono);

  Core::Utils::bool_parameter("PRINT_MAT_RHS_MAP_MATLAB", "no",
      "print system matrix, rhs vector, and full map to matlab readable file after solution of "
      "time step",
      &ssidynmono);

  Core::Utils::double_parameter("RELAX_LIN_SOLVER_TOLERANCE", 1.0,
      "relax the tolerance of the linear solver in case it is an iterative solver by scaling the "
      "convergence tolerance with factor RELAX_LIN_SOLVER_TOLERANCE",
      &ssidynmono);

  Core::Utils::int_parameter("RELAX_LIN_SOLVER_STEP", -1,
      "relax the tolerance of the linear solver within the first RELAX_LIN_SOLVER_STEP steps",
      &ssidynmono);

  /*----------------------------------------------------------------------*/
  /* parameters for SSI with manifold */
  /*----------------------------------------------------------------------*/

  Teuchos::ParameterList& ssidynmanifold = ssidyn.sublist("MANIFOLD", false,
      "Monolithic Structure Scalar Interaction with additional scalar transport on manifold");

  Core::Utils::bool_parameter(
      "ADD_MANIFOLD", "no", "activate additional manifold?", &ssidynmanifold);

  Core::Utils::bool_parameter("MESHTYING_MANIFOLD", "no",
      "activate meshtying between all manifold fields in case they intersect?", &ssidynmanifold);

  setStringToIntegralParameter<Inpar::ScaTra::InitialField>("INITIALFIELD", "zero_field",
      "Initial field for scalar transport on manifold",
      tuple<std::string>("zero_field", "field_by_function", "field_by_condition"),
      tuple<Inpar::ScaTra::InitialField>(Inpar::ScaTra::initfield_zero_field,
          Inpar::ScaTra::initfield_field_by_function, Inpar::ScaTra::initfield_field_by_condition),
      &ssidynmanifold);

  Core::Utils::int_parameter("INITFUNCNO", -1,
      "function number for scalar transport on manifold initial field", &ssidynmanifold);

  Core::Utils::int_parameter(
      "LINEAR_SOLVER", -1, "linear solver for scalar transport on manifold", &ssidynmanifold);

  Core::Utils::bool_parameter("OUTPUT_INFLOW", "no",
      "write output of inflow of scatra manifold - scatra coupling into scatra manifold to csv "
      "file",
      &ssidynmanifold);

  /*----------------------------------------------------------------------*/
  /* parameters for SSI with elch */
  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& ssidynelch = ssidyn.sublist(
      "ELCH", false, "Monolithic Structure Scalar Interaction with Elch as SCATRATIMINTTYPE");
  Core::Utils::bool_parameter("INITPOTCALC", "No",
      "Automatically calculate initial field for electric potential", &ssidynelch);
}

/*--------------------------------------------------------------------
--------------------------------------------------------------------*/
void Inpar::SSI::set_valid_conditions(
    std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>& condlist)
{
  using namespace Input;


  /*--------------------------------------------------------------------*/
  auto linessiplain = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI COUPLING LINE CONDITIONS", "SSICoupling", "SSI Coupling",
      Core::Conditions::SSICoupling, true, Core::Conditions::geometry_type_line);
  auto surfssiplain = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI COUPLING SURF CONDITIONS", "SSICoupling", "SSI Coupling",
      Core::Conditions::SSICoupling, true, Core::Conditions::geometry_type_surface);
  auto volssiplain = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI COUPLING VOL CONDITIONS", "SSICoupling", "SSI Coupling",
      Core::Conditions::SSICoupling, true, Core::Conditions::geometry_type_volume);

  // insert input file line components into condition definitions
  for (const auto& cond : {linessiplain, surfssiplain, volssiplain})
  {
    add_named_int(cond, "coupling_id");
    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  //! set solid dofset on scatra discretization
  auto linessi = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI COUPLING SOLIDTOSCATRA LINE CONDITIONS", "SSICouplingSolidToScatra",
      "SSI Coupling SolidToScatra", Core::Conditions::SSICouplingSolidToScatra, true,
      Core::Conditions::geometry_type_line);
  auto surfssi = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI COUPLING SOLIDTOSCATRA SURF CONDITIONS", "SSICouplingSolidToScatra",
      "SSI Coupling SolidToScatra", Core::Conditions::SSICouplingSolidToScatra, true,
      Core::Conditions::geometry_type_surface);
  auto volssi = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI COUPLING SOLIDTOSCATRA VOL CONDITIONS", "SSICouplingSolidToScatra",
      "SSI Coupling SolidToScatra", Core::Conditions::SSICouplingSolidToScatra, true,
      Core::Conditions::geometry_type_volume);

  // insert input file line components into condition definitions
  for (const auto& cond : {linessi, surfssi, volssi})
  {
    add_named_int(cond, "coupling_id");
    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  //! set scatra dofset on solid discretization
  auto linessi2 = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI COUPLING SCATRATOSOLID LINE CONDITIONS", "SSICouplingScatraToSolid",
      "SSI Coupling ScatraToSolid", Core::Conditions::SSICouplingScatraToSolid, true,
      Core::Conditions::geometry_type_line);
  auto surfssi2 = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI COUPLING SCATRATOSOLID SURF CONDITIONS", "SSICouplingScatraToSolid",
      "SSI Coupling ScatraToSolid", Core::Conditions::SSICouplingScatraToSolid, true,
      Core::Conditions::geometry_type_surface);
  auto volssi2 = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI COUPLING SCATRATOSOLID VOL CONDITIONS", "SSICouplingScatraToSolid",
      "SSI Coupling ScatraToSolid", Core::Conditions::SSICouplingScatraToSolid, true,
      Core::Conditions::geometry_type_volume);

  // insert input file line components into condition definitions
  for (const auto& cond : {linessi2, surfssi2, volssi2})
  {
    add_named_int(cond, "coupling_id");
    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // set ScaTra-Structure interaction interface meshtying condition
  auto pointssiinterfacemeshtying = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI INTERFACE MESHTYING POINT CONDITIONS", "ssi_interface_meshtying",
      "SSI Interface Meshtying", Core::Conditions::ssi_interface_meshtying, true,
      Core::Conditions::geometry_type_point);
  auto linessiinterfacemeshtying = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI INTERFACE MESHTYING LINE CONDITIONS", "ssi_interface_meshtying",
      "SSI Interface Meshtying", Core::Conditions::ssi_interface_meshtying, true,
      Core::Conditions::geometry_type_line);
  auto surfssiinterfacemeshtying = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI INTERFACE MESHTYING SURF CONDITIONS", "ssi_interface_meshtying",
      "SSI Interface Meshtying", Core::Conditions::ssi_interface_meshtying, true,
      Core::Conditions::geometry_type_surface);

  // equip condition definitions with input file line components
  //
  // REMARK: it would be cleaner to also set a reference to the structural meshtying condition here
  // and not only to the S2ICoupling condition. Of course, then also the structural meshtying should
  // be used which could/should be the long-term goal. However, to date, a simple structural
  // meshtying version for matching node is implemented within the SSI framework and therefore no
  // reference is necessary.

  // insert input file line components into condition definitions
  for (const auto& cond :
      {pointssiinterfacemeshtying, linessiinterfacemeshtying, surfssiinterfacemeshtying})
  {
    add_named_int(cond, "ConditionID");
    add_named_selection_component(cond, "INTERFACE_SIDE", "interface_side", "Undefined",
        Teuchos::tuple<std::string>("Undefined", "Slave", "Master"),
        Teuchos::tuple<int>(
            Inpar::S2I::side_undefined, Inpar::S2I::side_slave, Inpar::S2I::side_master));
    add_named_int(cond, "S2I_KINETICS_ID");

    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // condition, where additional scatra field on manifold is created
  auto ssisurfacemanifold = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI MANIFOLD SURF CONDITIONS", "SSISurfaceManifold", "scalar transport on manifold",
      Core::Conditions::SSISurfaceManifold, true, Core::Conditions::geometry_type_surface);

  add_named_int(ssisurfacemanifold, "ConditionID");
  add_named_selection_component(ssisurfacemanifold, "ImplType", "implementation type", "Undefined",
      Teuchos::tuple<std::string>("Undefined", "Standard", "ElchElectrode", "ElchDiffCond"),
      Teuchos::tuple<int>(Inpar::ScaTra::impltype_undefined, Inpar::ScaTra::impltype_std,
          Inpar::ScaTra::impltype_elch_electrode, Inpar::ScaTra::impltype_elch_diffcond));
  add_named_real(ssisurfacemanifold, "thickness");

  condlist.emplace_back(ssisurfacemanifold);

  /*--------------------------------------------------------------------*/
  // initial field by condition for scatra on manifold
  auto surfmanifoldinitfields = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SURF SCATRA MANIFOLD INITIAL FIELD CONDITIONS", "ScaTraManifoldInitfield",
      "Surface ScaTra Manifold Initfield", Core::Conditions::SurfaceInitfield, false,
      Core::Conditions::geometry_type_surface);

  add_named_selection_component(surfmanifoldinitfields, "FIELD", "init field", "ScaTra",
      Teuchos::tuple<std::string>("ScaTra"), Teuchos::tuple<std::string>("ScaTra"));
  add_named_int(surfmanifoldinitfields, "FUNCT");

  condlist.emplace_back(surfmanifoldinitfields);

  /*--------------------------------------------------------------------*/
  // kinetics condition for flux scatra <-> scatra on manifold
  auto surfmanifoldkinetics = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI MANIFOLD KINETICS SURF CONDITIONS", "SSISurfaceManifoldKinetics",
      "kinetics model for coupling scatra <-> scatra on manifold",
      Core::Conditions::SSISurfaceManifoldKinetics, true, Core::Conditions::geometry_type_surface);

  {
    add_named_int(surfmanifoldkinetics, "ConditionID");
    add_named_int(surfmanifoldkinetics, "ManifoldConditionID");

    std::map<int, std::pair<std::string, std::vector<std::shared_ptr<Input::LineComponent>>>>
        kinetic_model_choices;
    {
      {
        std::vector<std::shared_ptr<Input::LineComponent>> constantinterfaceresistance;
        constantinterfaceresistance.emplace_back(
            std::make_shared<Input::SeparatorComponent>("ONOFF"));
        constantinterfaceresistance.emplace_back(
            std::make_shared<Input::IntVectorComponent>("ONOFF", 2));

        constantinterfaceresistance.emplace_back(
            std::make_shared<Input::SeparatorComponent>("RESISTANCE"));
        constantinterfaceresistance.emplace_back(
            std::make_shared<Input::RealComponent>("RESISTANCE"));
        constantinterfaceresistance.emplace_back(new Input::SeparatorComponent("E-"));
        constantinterfaceresistance.emplace_back(new Input::IntComponent("E-"));

        kinetic_model_choices.emplace(Inpar::S2I::kinetics_constantinterfaceresistance,
            std::make_pair("ConstantInterfaceResistance", constantinterfaceresistance));
      }

      {
        // Butler-Volmer-reduced
        std::vector<std::shared_ptr<Input::LineComponent>> butlervolmerreduced;
        // total number of existing scalars
        butlervolmerreduced.emplace_back(std::make_shared<Input::SeparatorComponent>("NUMSCAL"));
        butlervolmerreduced.emplace_back(std::make_shared<Input::IntComponent>("NUMSCAL"));
        butlervolmerreduced.emplace_back(
            std::make_shared<Input::SeparatorComponent>("STOICHIOMETRIES"));
        butlervolmerreduced.emplace_back(std::make_shared<Input::IntVectorComponent>(
            "STOICHIOMETRIES", Input::LengthFromInt("NUMSCAL")));

        butlervolmerreduced.emplace_back(std::make_shared<Input::SeparatorComponent>("E-"));
        butlervolmerreduced.emplace_back(std::make_shared<Input::IntComponent>("E-"));
        butlervolmerreduced.emplace_back(std::make_shared<Input::SeparatorComponent>("K_R"));
        butlervolmerreduced.emplace_back(std::make_shared<Input::RealComponent>("K_R"));
        butlervolmerreduced.emplace_back(std::make_shared<Input::SeparatorComponent>("ALPHA_A"));
        butlervolmerreduced.emplace_back(std::make_shared<Input::RealComponent>("ALPHA_A"));
        butlervolmerreduced.emplace_back(std::make_shared<Input::SeparatorComponent>("ALPHA_C"));
        butlervolmerreduced.emplace_back(std::make_shared<Input::RealComponent>("ALPHA_C"));

        kinetic_model_choices.emplace(Inpar::S2I::kinetics_butlervolmerreduced,
            std::make_pair("Butler-VolmerReduced", butlervolmerreduced));
      }

      {
        std::vector<std::shared_ptr<Input::LineComponent>> noflux;

        kinetic_model_choices.emplace(
            Inpar::S2I::kinetics_nointerfaceflux, std::make_pair("NoInterfaceFlux", noflux));
      }
    }

    surfmanifoldkinetics->add_component(
        std::make_shared<Input::SeparatorComponent>("KINETIC_MODEL"));
    surfmanifoldkinetics->add_component(std::make_shared<Input::SwitchComponent>(
        "KINETIC_MODEL", Inpar::S2I::kinetics_constantinterfaceresistance, kinetic_model_choices));
  }

  condlist.emplace_back(surfmanifoldkinetics);

  /*--------------------------------------------------------------------*/
  // Dirichlet conditions for scatra on manifold
  auto pointmanifolddirichlet = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN POINT MANIFOLD DIRICH CONDITIONS", "ManifoldDirichlet", "Point Dirichlet",
      Core::Conditions::PointDirichlet, false, Core::Conditions::geometry_type_point);
  auto linemanifolddirichlet = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN LINE MANIFOLD DIRICH CONDITIONS", "ManifoldDirichlet", "Line Dirichlet",
      Core::Conditions::LineDirichlet, false, Core::Conditions::geometry_type_line);
  auto surfmanifolddirichlet = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SURF MANIFOLD DIRICH CONDITIONS", "ManifoldDirichlet", "Surface Dirichlet",
      Core::Conditions::SurfaceDirichlet, false, Core::Conditions::geometry_type_surface);

  const auto add_dirichlet_manifold_components =
      [](std::shared_ptr<Core::Conditions::ConditionDefinition> definition)
  {
    add_named_int(definition, "NUMDOF");
    add_named_int_vector(definition, "ONOFF", "", "NUMDOF");
    add_named_real_vector(definition, "VAL", "", "NUMDOF");
    add_named_int_vector(definition, "FUNCT", "", "NUMDOF", 0, false);
  };

  {
    add_dirichlet_manifold_components(pointmanifolddirichlet);
    add_dirichlet_manifold_components(linemanifolddirichlet);
    add_dirichlet_manifold_components(surfmanifolddirichlet);
  }

  condlist.push_back(pointmanifolddirichlet);
  condlist.push_back(linemanifolddirichlet);
  condlist.push_back(surfmanifolddirichlet);

  /*--------------------------------------------------------------------*/
  // set ScaTra-Structure Interaction interface contact condition
  auto linessiinterfacecontact = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI INTERFACE CONTACT LINE CONDITIONS", "SSIInterfaceContact",
      "SSI Interface Contact", Core::Conditions::SSIInterfaceContact, true,
      Core::Conditions::geometry_type_line);
  auto surfssiinterfacecontact = std::make_shared<Core::Conditions::ConditionDefinition>(
      "DESIGN SSI INTERFACE CONTACT SURF CONDITIONS", "SSIInterfaceContact",
      "SSI Interface Contact", Core::Conditions::SSIInterfaceContact, true,
      Core::Conditions::geometry_type_surface);

  // insert input file line components into condition definitions
  for (const auto& cond : {linessiinterfacecontact, surfssiinterfacecontact})
  {
    add_named_int(cond, "ConditionID");
    add_named_selection_component(cond, "INTERFACE_SIDE", "interface_side", "Undefined",
        Teuchos::tuple<std::string>("Undefined", "Slave", "Master"),
        Teuchos::tuple<int>(
            Inpar::S2I::side_undefined, Inpar::S2I::side_slave, Inpar::S2I::side_master));
    add_named_int(cond, "S2I_KINETICS_ID");
    add_named_int(cond, "CONTACT_CONDITION_ID");

    condlist.push_back(cond);
  }
}

FOUR_C_NAMESPACE_CLOSE
