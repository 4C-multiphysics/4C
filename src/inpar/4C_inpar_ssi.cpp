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
#include "4C_io_input_spec_builders.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_linalg_sparseoperator.hpp"
FOUR_C_NAMESPACE_OPEN

void Inpar::SSI::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using namespace Core::IO::InputSpecBuilders;

  list["SSI CONTROL"] = group("SSI CONTROL",
      {

          // Output type
          parameter<double>("RESTARTEVERYTIME",
              {.description = "write restart possibility every RESTARTEVERY steps",
                  .default_value = 0.0}),
          parameter<int>(
              "RESTARTEVERY", {.description = "write restart possibility every RESTARTEVERY steps",
                                  .default_value = 1}),
          // Time loop control
          parameter<int>(
              "NUMSTEP", {.description = "maximum number of Timesteps", .default_value = 200}),
          parameter<double>(
              "MAXTIME", {.description = "total simulation time", .default_value = 1000.0}),

          parameter<double>(
              "TIMESTEP", {.description = "time step size dt", .default_value = -1.0}),
          parameter<bool>(
              "DIFFTIMESTEPSIZE", {.description = "use different step size for scatra and solid",
                                      .default_value = false}),
          parameter<double>("RESULTSEVERYTIME",
              {.description = "increment for writing solution", .default_value = 0.0}),
          parameter<int>("RESULTSEVERY",
              {.description = "increment for writing solution", .default_value = 1}),
          parameter<int>("ITEMAX",
              {.description = "maximum number of iterations over fields", .default_value = 10}),
          parameter<bool>("SCATRA_FROM_RESTART_FILE",
              {.description =
                      "read scatra result from restart files (use option 'restartfromfile' during "
                      "execution of 4C)",
                  .default_value = false}),
          parameter<std::string>("SCATRA_FILENAME",
              {.description = "Control-file name for reading scatra results in SSI",
                  .default_value = "nil"}),

          // Type of coupling strategy between the two fields
          deprecated_selection<FieldCoupling>("FIELDCOUPLING",
              {
                  {"volume_matching", FieldCoupling::volume_match},
                  {"volume_nonmatching", FieldCoupling::volume_nonmatch},
                  {"boundary_nonmatching", FieldCoupling::boundary_nonmatch},
                  {"volumeboundary_matching", FieldCoupling::volumeboundary_match},
              },
              {.description = "Type of coupling strategy between fields",
                  .default_value = FieldCoupling::volume_match}),

          // Coupling strategy for SSI solvers
          parameter<SolutionSchemeOverFields>(
              "COUPALGO", {.description = "Coupling strategies for SSI solvers",
                              .default_value = SolutionSchemeOverFields::ssi_IterStagg}),

          // type of scalar transport time integration
          deprecated_selection<ScaTraTimIntType>("SCATRATIMINTTYPE",
              {
                  {"Standard", ScaTraTimIntType::standard},
                  {"Cardiac_Monodomain", ScaTraTimIntType::cardiac_monodomain},
                  {"Elch", ScaTraTimIntType::elch},
              },
              {.description =
                      "scalar transport time integration type is needed to instantiate correct "
                      "scalar transport time integration scheme for ssi problems",
                  .default_value = ScaTraTimIntType::standard}),

          // Restart from Structure problem instead of SSI
          parameter<bool>(
              "RESTART_FROM_STRUCTURE", {.description = "restart from structure problem (e.g. from "
                                                        "prestress calculations) instead of ssi",
                                            .default_value = false}),

          // Adaptive time stepping
          parameter<bool>("ADAPTIVE_TIMESTEPPING",
              {.description = "flag for adaptive time stepping", .default_value = false}),

          // do redistribution by binning of solid mechanics discretization (scatra dis is cloned
          // from
          // solid
          // dis for volume_matching and volumeboundary_matching)
          parameter<bool>("REDISTRIBUTE_SOLID",
              {.description = "redistribution by binning of solid mechanics discretization",
                  .default_value = false})},
      {.required =
              false}); /*----------------------------------------------------------------------*/
  /* parameters for partitioned SSI */
  /*----------------------------------------------------------------------*/
  list["SSI CONTROL/PARTITIONED"] = group("SSI CONTROL/PARTITIONED",
      {

          // Solver parameter for relaxation of iterative staggered partitioned SSI
          parameter<double>(
              "MAXOMEGA", {.description = "largest omega allowed for Aitken relaxation",
                              .default_value = 10.0}),
          parameter<double>(
              "MINOMEGA", {.description = "smallest omega allowed for Aitken relaxation",
                              .default_value = 0.1}),
          parameter<double>(
              "STARTOMEGA", {.description = "fixed relaxation parameter", .default_value = 1.0}),

          // convergence tolerance of outer iteration loop
          parameter<double>("CONVTOL",
              {.description =
                      "tolerance for convergence check of outer iteration within partitioned SSI",
                  .default_value = 1e-6})},
      {.required = false});

  /*----------------------------------------------------------------------*/
  /* parameters for monolithic SSI */
  /*----------------------------------------------------------------------*/
  list["SSI CONTROL/MONOLITHIC"] = group("SSI CONTROL/MONOLITHIC",
      {

          // convergence tolerances of Newton-Raphson iteration loop
          parameter<double>(
              "ABSTOLRES", {.description = "absolute tolerance for deciding if global residual of "
                                           "nonlinear problem is already zero",
                               .default_value = 1.e-14}),
          parameter<double>(
              "CONVTOL", {.description = "tolerance for convergence check of "
                                         "Newton-Raphson iteration within monolithic SSI",
                             .default_value = 1.e-6}),

          // ID of linear solver for global system of equations
          parameter<int>(
              "LINEAR_SOLVER", {.description = "ID of linear solver for global system of equations",
                                   .default_value = -1}),

          // type of global system matrix in global system of equations
          deprecated_selection<Core::LinAlg::MatrixType>("MATRIXTYPE",
              {
                  {"undefined", Core::LinAlg::MatrixType::undefined},
                  {"block", Core::LinAlg::MatrixType::block_field},
                  {"sparse", Core::LinAlg::MatrixType::sparse},
              },
              {.description = "type of global system matrix in global system of equations",
                  .default_value = Core::LinAlg::MatrixType::undefined}),

          parameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION",
              {.description = "flag for equilibration of global system of equations",
                  .default_value = Core::LinAlg::EquilibrationMethod::none}),

          parameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION_STRUCTURE",
              {.description = "flag for equilibration of structural equations",
                  .default_value = Core::LinAlg::EquilibrationMethod::none}),

          parameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION_SCATRA",
              {.description = "flag for equilibration of scatra equations",
                  .default_value = Core::LinAlg::EquilibrationMethod::none}),

          parameter<bool>("PRINT_MAT_RHS_MAP_MATLAB",
              {.description =
                      "print system matrix, rhs vector, and full map to matlab readable file after "
                      "solution of time step",
                  .default_value = false}),

          parameter<double>("RELAX_LIN_SOLVER_TOLERANCE",
              {.description =
                      "relax the tolerance of the linear solver in case it is an iterative solver "
                      "by scaling the convergence tolerance with factor RELAX_LIN_SOLVER_TOLERANCE",
                  .default_value = 1.0}),

          parameter<int>("RELAX_LIN_SOLVER_STEP",
              {.description = "relax the tolerance of the linear solver within "
                              "the first RELAX_LIN_SOLVER_STEP steps",
                  .default_value = -1})},
      {.required = false});

  /*----------------------------------------------------------------------*/
  /* parameters for SSI with manifold */
  /*----------------------------------------------------------------------*/

  list["SSI CONTROL/MANIFOLD"] = group("SSI CONTROL/MANIFOLD",
      {

          parameter<bool>("ADD_MANIFOLD",
              {.description = "activate additional manifold?", .default_value = false}),

          parameter<bool>("MESHTYING_MANIFOLD",
              {.description =
                      "activate meshtying between all manifold fields in case they intersect?",
                  .default_value = false}),


          deprecated_selection<Inpar::ScaTra::InitialField>("INITIALFIELD",
              {
                  {"zero_field", Inpar::ScaTra::initfield_zero_field},
                  {"field_by_function", Inpar::ScaTra::initfield_field_by_function},
                  {"field_by_condition", Inpar::ScaTra::initfield_field_by_condition},
              },
              {.description = "Initial field for scalar transport on manifold",
                  .default_value = Inpar::ScaTra::initfield_zero_field}),

          parameter<int>("INITFUNCNO",
              {.description = "function number for scalar transport on manifold initial field",
                  .default_value = -1}),
          parameter<int>(
              "LINEAR_SOLVER", {.description = "linear solver for scalar transport on manifold",
                                   .default_value = -1}),

          parameter<bool>(
              "OUTPUT_INFLOW", {.description = "write output of inflow of scatra manifold - scatra "
                                               "coupling into scatra manifold to csv file",
                                   .default_value = false})},
      {.required = false});

  /*----------------------------------------------------------------------*/
  /* parameters for SSI with elch */
  /*----------------------------------------------------------------------*/
  list["SSI CONTROL/ELCH"] = group("SSI CONTROL/ELCH",
      {

          parameter<bool>("INITPOTCALC",
              {.description = "Automatically calculate initial field for electric potential",
                  .default_value = false})},
      {.required = false});
}

/*--------------------------------------------------------------------
--------------------------------------------------------------------*/
void Inpar::SSI::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;


  /*--------------------------------------------------------------------*/
  Core::Conditions::ConditionDefinition linessiplain("DESIGN SSI COUPLING LINE CONDITIONS",
      "SSICoupling", "SSI Coupling", Core::Conditions::SSICoupling, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfssiplain("DESIGN SSI COUPLING SURF CONDITIONS",
      "SSICoupling", "SSI Coupling", Core::Conditions::SSICoupling, true,
      Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition volssiplain("DESIGN SSI COUPLING VOL CONDITIONS",
      "SSICoupling", "SSI Coupling", Core::Conditions::SSICoupling, true,
      Core::Conditions::geometry_type_volume);

  // insert input file line components into condition definitions
  const auto make_ssiplain = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(parameter<int>("coupling_id"));
    condlist.push_back(cond);
  };

  make_ssiplain(linessiplain);
  make_ssiplain(surfssiplain);

  /*--------------------------------------------------------------------*/
  //! set solid dofset on scatra discretization
  Core::Conditions::ConditionDefinition linessi("DESIGN SSI COUPLING SOLIDTOSCATRA LINE CONDITIONS",
      "SSICouplingSolidToScatra", "SSI Coupling SolidToScatra",
      Core::Conditions::SSICouplingSolidToScatra, true, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfssi("DESIGN SSI COUPLING SOLIDTOSCATRA SURF CONDITIONS",
      "SSICouplingSolidToScatra", "SSI Coupling SolidToScatra",
      Core::Conditions::SSICouplingSolidToScatra, true, Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition volssi("DESIGN SSI COUPLING SOLIDTOSCATRA VOL CONDITIONS",
      "SSICouplingSolidToScatra", "SSI Coupling SolidToScatra",
      Core::Conditions::SSICouplingSolidToScatra, true, Core::Conditions::geometry_type_volume);

  // insert input file line components into condition definitions
  const auto make_ssi = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(parameter<int>("coupling_id"));
    condlist.push_back(cond);
  };

  make_ssi(linessi);
  make_ssi(surfssi);
  make_ssi(volssi);

  /*--------------------------------------------------------------------*/
  //! set scatra dofset on solid discretization
  Core::Conditions::ConditionDefinition linessi2(
      "DESIGN SSI COUPLING SCATRATOSOLID LINE CONDITIONS", "SSICouplingScatraToSolid",
      "SSI Coupling ScatraToSolid", Core::Conditions::SSICouplingScatraToSolid, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfssi2(
      "DESIGN SSI COUPLING SCATRATOSOLID SURF CONDITIONS", "SSICouplingScatraToSolid",
      "SSI Coupling ScatraToSolid", Core::Conditions::SSICouplingScatraToSolid, true,
      Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition volssi2("DESIGN SSI COUPLING SCATRATOSOLID VOL CONDITIONS",
      "SSICouplingScatraToSolid", "SSI Coupling ScatraToSolid",
      Core::Conditions::SSICouplingScatraToSolid, true, Core::Conditions::geometry_type_volume);

  // insert input file line components into condition definitions
  const auto make_ssi2 = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(parameter<int>("coupling_id"));
    condlist.push_back(cond);
  };

  make_ssi2(linessi2);
  make_ssi2(surfssi2);
  make_ssi2(volssi2);


  /*--------------------------------------------------------------------*/
  // set ScaTra-Structure interaction interface meshtying condition
  Core::Conditions::ConditionDefinition pointssiinterfacemeshtying(
      "DESIGN SSI INTERFACE MESHTYING POINT CONDITIONS", "ssi_interface_meshtying",
      "SSI Interface Meshtying", Core::Conditions::ssi_interface_meshtying, true,
      Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition linessiinterfacemeshtying(
      "DESIGN SSI INTERFACE MESHTYING LINE CONDITIONS", "ssi_interface_meshtying",
      "SSI Interface Meshtying", Core::Conditions::ssi_interface_meshtying, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfssiinterfacemeshtying(
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
  const auto make_ssiinterfacemeshtying = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(parameter<int>("ConditionID"));
    cond.add_component(deprecated_selection<S2I::InterfaceSides>("INTERFACE_SIDE",
        {{"Undefined", Inpar::S2I::side_undefined}, {"Slave", Inpar::S2I::side_slave},
            {"Master", Inpar::S2I::side_master}},
        {.description = "interface_side"}));
    cond.add_component(parameter<int>("S2I_KINETICS_ID"));

    condlist.push_back(cond);
  };

  make_ssiinterfacemeshtying(pointssiinterfacemeshtying);
  make_ssiinterfacemeshtying(linessiinterfacemeshtying);
  make_ssiinterfacemeshtying(surfssiinterfacemeshtying);

  /*--------------------------------------------------------------------*/
  // condition, where additional scatra field on manifold is created
  Core::Conditions::ConditionDefinition ssisurfacemanifold("DESIGN SSI MANIFOLD SURF CONDITIONS",
      "SSISurfaceManifold", "scalar transport on manifold", Core::Conditions::SSISurfaceManifold,
      true, Core::Conditions::geometry_type_surface);

  ssisurfacemanifold.add_component(parameter<int>("ConditionID"));
  ssisurfacemanifold.add_component(deprecated_selection<Inpar::ScaTra::ImplType>("ImplType",
      {{"Undefined", Inpar::ScaTra::impltype_undefined}, {"Standard", Inpar::ScaTra::impltype_std},
          {"ElchElectrode", Inpar::ScaTra::impltype_elch_electrode},
          {"ElchDiffCond", Inpar::ScaTra::impltype_elch_diffcond}},
      {.description = "implementation type"}));
  ssisurfacemanifold.add_component(parameter<double>("thickness"));

  condlist.emplace_back(ssisurfacemanifold);

  /*--------------------------------------------------------------------*/
  // initial field by condition for scatra on manifold
  Core::Conditions::ConditionDefinition surfmanifoldinitfields(
      "DESIGN SURF SCATRA MANIFOLD INITIAL FIELD CONDITIONS", "ScaTraManifoldInitfield",
      "Surface ScaTra Manifold Initfield", Core::Conditions::SurfaceInitfield, false,
      Core::Conditions::geometry_type_surface);

  surfmanifoldinitfields.add_component(
      deprecated_selection<std::string>("FIELD", {"ScaTra"}, {.description = "init field"}));
  surfmanifoldinitfields.add_component(parameter<int>("FUNCT"));

  condlist.emplace_back(surfmanifoldinitfields);

  /*--------------------------------------------------------------------*/
  // kinetics condition for flux scatra <-> scatra on manifold
  Core::Conditions::ConditionDefinition surfmanifoldkinetics(
      "DESIGN SSI MANIFOLD KINETICS SURF CONDITIONS", "SSISurfaceManifoldKinetics",
      "kinetics model for coupling scatra <-> scatra on manifold",
      Core::Conditions::SSISurfaceManifoldKinetics, true, Core::Conditions::geometry_type_surface);

  {
    surfmanifoldkinetics.add_component(parameter<int>("ConditionID"));
    surfmanifoldkinetics.add_component(parameter<int>("ManifoldConditionID"));

    using namespace Core::IO::InputSpecBuilders;

    surfmanifoldkinetics.add_component(one_of({
        all_of({
            deprecated_selection<Inpar::S2I::KineticModels>(
                "KINETIC_MODEL", {{"ConstantInterfaceResistance",
                                     Inpar::S2I::kinetics_constantinterfaceresistance}}),
            parameter<std::vector<int>>("ONOFF", {.size = 2}),
            parameter<double>("RESISTANCE"),
            parameter<int>("E-"),
        }),
        all_of({
            deprecated_selection<Inpar::S2I::KineticModels>("KINETIC_MODEL",
                {{"Butler-VolmerReduced", Inpar::S2I::kinetics_butlervolmerreduced}}),
            parameter<int>("NUMSCAL"),
            parameter<std::vector<int>>(
                "STOICHIOMETRIES", {.size = from_parameter<int>("NUMSCAL")}),
            parameter<int>("E-"),
            parameter<double>("K_R"),
            parameter<double>("ALPHA_A"),
            parameter<double>("ALPHA_C"),
        }),
        deprecated_selection<S2I::KineticModels>(
            "KINETIC_MODEL", {{"NoInterfaceFlux", Inpar::S2I::kinetics_nointerfaceflux}}),
    }));
  }

  condlist.emplace_back(surfmanifoldkinetics);

  /*--------------------------------------------------------------------*/
  // Dirichlet conditions for scatra on manifold
  Core::Conditions::ConditionDefinition pointmanifolddirichlet(
      "DESIGN POINT MANIFOLD DIRICH CONDITIONS", "ManifoldDirichlet", "Point Dirichlet",
      Core::Conditions::PointDirichlet, false, Core::Conditions::geometry_type_point);
  Core::Conditions::ConditionDefinition linemanifolddirichlet(
      "DESIGN LINE MANIFOLD DIRICH CONDITIONS", "ManifoldDirichlet", "Line Dirichlet",
      Core::Conditions::LineDirichlet, false, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfmanifolddirichlet(
      "DESIGN SURF MANIFOLD DIRICH CONDITIONS", "ManifoldDirichlet", "Surface Dirichlet",
      Core::Conditions::SurfaceDirichlet, false, Core::Conditions::geometry_type_surface);

  const auto add_dirichlet_manifold_components =
      [](Core::Conditions::ConditionDefinition& definition)
  {
    using namespace Core::IO::InputSpecBuilders::Validators;
    definition.add_component(parameter<int>("NUMDOF"));
    definition.add_component(parameter<std::vector<int>>(
        "ONOFF", {.description = "", .size = from_parameter<int>("NUMDOF")}));
    definition.add_component(parameter<std::vector<double>>(
        "VAL", {.description = "", .size = from_parameter<int>("NUMDOF")}));
    definition.add_component(parameter<std::vector<std::optional<int>>>(
        "FUNCT", {
                     .description = "",
                     .validator = all_elements(null_or(positive<int>())),
                     .size = from_parameter<int>("NUMDOF"),
                 }));
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
  Core::Conditions::ConditionDefinition linessiinterfacecontact(
      "DESIGN SSI INTERFACE CONTACT LINE CONDITIONS", "SSIInterfaceContact",
      "SSI Interface Contact", Core::Conditions::SSIInterfaceContact, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfssiinterfacecontact(
      "DESIGN SSI INTERFACE CONTACT SURF CONDITIONS", "SSIInterfaceContact",
      "SSI Interface Contact", Core::Conditions::SSIInterfaceContact, true,
      Core::Conditions::geometry_type_surface);

  // insert input file line components into condition definitions
  const auto make_ssiinterfacecontact = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(parameter<int>("ConditionID"));
    cond.add_component(deprecated_selection<S2I::InterfaceSides>("INTERFACE_SIDE",
        {{"Undefined", Inpar::S2I::side_undefined}, {"Slave", Inpar::S2I::side_slave},
            {"Master", Inpar::S2I::side_master}},
        {.description = "interface_side"}));
    cond.add_component(parameter<int>("S2I_KINETICS_ID"));
    cond.add_component(parameter<int>("CONTACT_CONDITION_ID"));

    condlist.push_back(cond);
  };

  make_ssiinterfacecontact(linessiinterfacecontact);
  make_ssiinterfacecontact(surfssiinterfacecontact);
}

FOUR_C_NAMESPACE_CLOSE