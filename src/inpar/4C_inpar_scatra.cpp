// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_scatra.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_inpar_bio.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

void Inpar::ScaTra::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using namespace Core::IO::InputSpecBuilders;

  list["SCALAR TRANSPORT DYNAMIC"] = group("SCALAR TRANSPORT DYNAMIC",
      {

          deprecated_selection<Inpar::ScaTra::SolverType>("SOLVERTYPE",
              {
                  {"linear_full", solvertype_linear_full},
                  {"linear_incremental", solvertype_linear_incremental},
                  {"nonlinear", solvertype_nonlinear},
                  {"nonlinear_multiscale_macrotomicro",
                      solvertype_nonlinear_multiscale_macrotomicro},
                  {"nonlinear_multiscale_macrotomicro_aitken",
                      solvertype_nonlinear_multiscale_macrotomicro_aitken},
                  {"nonlinear_multiscale_macrotomicro_aitken_dofsplit",
                      solvertype_nonlinear_multiscale_macrotomicro_aitken_dofsplit},
                  {"nonlinear_multiscale_microtomacro",
                      solvertype_nonlinear_multiscale_microtomacro},
              },
              {.description = "type of scalar transport solver",
                  .default_value = solvertype_linear_full}),


          deprecated_selection<Inpar::ScaTra::TimeIntegrationScheme>("TIMEINTEGR",
              {
                  {"Stationary", timeint_stationary},
                  {"One_Step_Theta", timeint_one_step_theta},
                  {"BDF2", timeint_bdf2},
                  {"Gen_Alpha", timeint_gen_alpha},
              },
              {.description = "Time Integration Scheme", .default_value = timeint_one_step_theta}),

          parameter<double>(
              "MAXTIME", {.description = "Total simulation time", .default_value = 1000.0}),
          parameter<int>(
              "NUMSTEP", {.description = "Total number of time steps", .default_value = 20}),

          parameter<double>("TIMESTEP", {.description = "Time increment dt", .default_value = 0.1}),
          parameter<double>("THETA",
              {.description = "One-step-theta time integration factor", .default_value = 0.5}),
          parameter<double>("ALPHA_M",
              {.description = "Generalized-alpha time integration factor", .default_value = 0.5}),
          parameter<double>("ALPHA_F",
              {.description = "Generalized-alpha time integration factor", .default_value = 0.5}),
          parameter<double>("GAMMA",
              {.description = "Generalized-alpha time integration factor", .default_value = 0.5}),
          parameter<int>("RESULTSEVERY",
              {.description = "Increment for writing solution", .default_value = 1}),
          parameter<int>(
              "RESTARTEVERY", {.description = "Increment for writing restart", .default_value = 1}),
          parameter<int>("MATID",
              {.description = "Material ID for automatic mesh generation", .default_value = -1}),

          deprecated_selection<Inpar::ScaTra::VelocityField>("VELOCITYFIELD",
              {
                  {"zero", velocity_zero},
                  {"function", velocity_function},
                  {"Navier_Stokes", velocity_Navier_Stokes},
              },
              {.description = "type of velocity field used for scalar transport problems",
                  .default_value = velocity_zero}),

          parameter<int>(
              "VELFUNCNO", {.description = "function number for scalar transport velocity field",
                               .default_value = -1}),



          deprecated_selection<Inpar::ScaTra::InitialField>("INITIALFIELD",
              {
                  {"zero_field", initfield_zero_field},
                  {"field_by_function", initfield_field_by_function},
                  {"field_by_condition", initfield_field_by_condition},
                  {"disturbed_field_by_function", initfield_disturbed_field_by_function},
                  {"1D_DISCONTPV", initfield_discontprogvar_1D},
                  {"FLAME_VORTEX_INTERACTION", initfield_flame_vortex_interaction},
                  {"RAYTAYMIXFRAC", initfield_raytaymixfrac},
                  {"L_shaped_domain", initfield_Lshapeddomain},
                  {"facing_flame_fronts", initfield_facing_flame_fronts},
                  {"oracles_flame", initfield_oracles_flame},
                  {"high_forced_hit", initialfield_forced_hit_high_Sc},
                  {"low_forced_hit", initialfield_forced_hit_low_Sc},
                  {"algebraic_field_dependence", initialfield_algebraic_field_dependence},
              },
              {.description = "Initial Field for transport problem",
                  .default_value = initfield_zero_field}),

          parameter<int>(
              "INITFUNCNO", {.description = "function number for scalar transport initial field",
                                .default_value = -1}),

          parameter<bool>("SPHERICALCOORDS",
              {.description = "use of spherical coordinates", .default_value = false}),

          deprecated_selection<Inpar::ScaTra::CalcError>("CALCERROR",
              {
                  {"No", calcerror_no},
                  {"Kwok_Wu", calcerror_Kwok_Wu},
                  {"ConcentricCylinders", calcerror_cylinder},
                  {"Electroneutrality", calcerror_electroneutrality},
                  {"error_by_function", calcerror_byfunction},
                  {"error_by_condition", calcerror_bycondition},
                  {"SphereDiffusion", calcerror_spherediffusion},
                  {"AnalyticSeries", calcerror_AnalyticSeries},
              },
              {.description = "compute error compared to analytical solution",
                  .default_value = calcerror_no}),

          parameter<int>("CALCERRORNO",
              {.description = "function number for scalar transport error computation",
                  .default_value = -1}),

          deprecated_selection<Inpar::ScaTra::FluxType>("CALCFLUX_DOMAIN",
              {
                  {"No", flux_none},
                  {"total", flux_total},
                  {"diffusive", flux_diffusive},
              },
              {.description = "output of diffusive/total flux vectors inside domain",
                  .default_value = flux_none}),

          parameter<bool>("CALCFLUX_DOMAIN_LUMPED",
              {.description =
                      "perform approximate domain flux calculation involving matrix lumping",
                  .default_value = true}),

          deprecated_selection<Inpar::ScaTra::FluxType>("CALCFLUX_BOUNDARY",
              {
                  {"No", flux_none},
                  {"total", flux_total},
                  {"diffusive", flux_diffusive},
                  {"convective", flux_convective},
              },
              {.description = "output of convective/diffusive/total flux vectors on boundary",
                  .default_value = flux_none}),

          parameter<bool>("CALCFLUX_BOUNDARY_LUMPED",
              {.description =
                      "perform approximate boundary flux calculation involving matrix lumping",
                  .default_value = true}),

          parameter<std::string>("WRITEFLUX_IDS",
              {.description = "Write diffusive/total flux vector fields for these scalar "
                              "fields only (starting with 1)",
                  .default_value = "-1"}),


          deprecated_selection<Inpar::ScaTra::OutputScalarType>("OUTPUTSCALARS",
              {
                  {"none", outputscalars_none},
                  {"entire_domain", outputscalars_entiredomain},
                  {"by_condition", outputscalars_condition},
                  {"entire_domain_and_by_condition", outputscalars_entiredomain_condition},
              },
              {.description = "Output of total and mean values for transported scalars",
                  .default_value = outputscalars_none}),
          parameter<bool>("OUTPUTSCALARSMEANGRAD",
              {.description = "Output of mean gradient of scalars", .default_value = false}),
          parameter<bool>("OUTINTEGRREAC",
              {.description = "Output of integral reaction values", .default_value = false}),
          parameter<bool>(
              "OUTPUT_GMSH", {.description = "Do you want to write Gmsh postprocessing files?",
                                 .default_value = false}),

          parameter<bool>("MATLAB_STATE_OUTPUT",
              {.description = "Do you want to write the state solution to Matlab file?",
                  .default_value = false}),

          deprecated_selection<Inpar::ScaTra::ConvForm>("CONVFORM",
              {
                  {"convective", convform_convective},
                  {"conservative", convform_conservative},
              },
              {.description = "form of convective term", .default_value = convform_convective}),

          parameter<bool>("NEUMANNINFLOW",
              {.description = "Flag to (de)activate potential Neumann inflow term(s)",
                  .default_value = false}),

          parameter<bool>("CONV_HEAT_TRANS",
              {.description =
                      "Flag to (de)activate potential convective heat transfer boundary conditions",
                  .default_value = false}),

          parameter<bool>(
              "SKIPINITDER", {.description = "Flag to skip computation of initial time derivative",
                                 .default_value = false}),

          deprecated_selection<Inpar::ScaTra::FSSUGRDIFF>("FSSUGRDIFF",
              {
                  {"No", fssugrdiff_no},
                  {"artificial", fssugrdiff_artificial},
                  {"Smagorinsky_all", fssugrdiff_smagorinsky_all},
                  {"Smagorinsky_small", fssugrdiff_smagorinsky_small},
              },
              {.description = "fine-scale subgrid diffusivity", .default_value = fssugrdiff_no}),

          deprecated_selection<Inpar::FLUID::MeshTying>("MESHTYING",
              {
                  {"no", Inpar::FLUID::no_meshtying},
                  {"Condensed_Smat", Inpar::FLUID::condensed_smat},
                  {"Condensed_Bmat", Inpar::FLUID::condensed_bmat},
                  {"Condensed_Bmat_merged", Inpar::FLUID::condensed_bmat_merged},
              },
              {.description = "Flag to (de)activate mesh tying algorithm",
                  .default_value = Inpar::FLUID::no_meshtying}),

          // Type of coupling strategy between the two fields
          deprecated_selection<Inpar::ScaTra::FieldCoupling>("FIELDCOUPLING",
              {
                  {"matching", coupling_match},
                  {"volmortar", coupling_volmortar},
              },
              {.description = "Type of coupling strategy between fields",
                  .default_value = coupling_match}),

          // linear solver id used for scalar transport/elch problems
          parameter<int>("LINEAR_SOLVER",
              {.description = "number of linear solver used for scalar transport/elch...",
                  .default_value = -1}),
          // linear solver id used for l2 projection problems (e.g. gradient projections)
          parameter<int>("L2_PROJ_LINEAR_SOLVER",
              {.description = "number of linear solver used for l2-projection sub-problems",
                  .default_value = -1}),

          // flag for equilibration of global system of equations
          parameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION",
              {.description = "flag for equilibration of global system of equations",
                  .default_value = Core::LinAlg::EquilibrationMethod::none}),

          // type of global system matrix in global system of equations
          parameter<Core::LinAlg::MatrixType>("MATRIXTYPE",
              {.description = "type of global system matrix in global system of equations",
                  .default_value = Core::LinAlg::MatrixType::sparse}),

          // flag for natural convection effects
          parameter<bool>("NATURAL_CONVECTION",
              {.description = "Include natural convection effects", .default_value = false}),

          // parameters for finite difference check
          deprecated_selection<Inpar::ScaTra::FdCheck>("FDCHECK",
              {
                  {"none", fdcheck_none},
                  {"global", fdcheck_global},
                  {"global_extended", fdcheck_global_extended},
                  {"local", fdcheck_local},
              },
              {.description = "flag for finite difference check: none, local, or global",
                  .default_value = fdcheck_none}),
          parameter<double>("FDCHECKEPS",
              {.description = "dof perturbation magnitude for finite difference check (1.e-6 "
                              "seems to work very well, whereas smaller values don't)",
                  .default_value = 1.e-6}),
          parameter<double>(
              "FDCHECKTOL", {.description = "relative tolerance for finite difference check",
                                .default_value = 1.e-6}),

          // parameter for optional computation of domain and boundary integrals, i.e., of surface
          // areas and volumes associated with specified nodesets
          deprecated_selection<Inpar::ScaTra::ComputeIntegrals>("COMPUTEINTEGRALS",
              {
                  {"none", computeintegrals_none},
                  {"initial", computeintegrals_initial},
                  {"repeated", computeintegrals_repeated},
              },
              {.description = "flag for optional computation of domain integrals",
                  .default_value = computeintegrals_none}),

          // parameter for using p-adpativity and semi-implicit evaluation of the reaction term (at
          // the moment only used for HDG and cardiac monodomain problems)
          parameter<bool>("PADAPTIVITY",
              {.description = "Flag to (de)activate p-adativity", .default_value = false}),
          parameter<double>("PADAPTERRORTOL",
              {.description =
                      "The error tolerance to calculate the variation of the elemental degree",
                  .default_value = 1e-6}),
          parameter<double>("PADAPTERRORBASE",
              {.description =
                      "The error tolerance base to calculate the variation of the elemental degree",
                  .default_value = 1.66}),
          parameter<int>("PADAPTDEGREEMAX",
              {.description = "The max. degree of the shape functions", .default_value = 4}),
          parameter<bool>("SEMIIMPLICIT",
              {.description = "Flag to (de)activate semi-implicit calculation of the reaction term",
                  .default_value = false}),

          // flag for output of performance statistics associated with linear solver into *.csv file
          parameter<bool>(
              "OUTPUTLINSOLVERSTATS", {.description = "flag for output of performance statistics "
                                                      "associated with linear solver into csv file",
                                          .default_value = false}),

          // flag for output of performance statistics associated with nonlinear solver into *.csv
          // file
          parameter<bool>("OUTPUTNONLINSOLVERSTATS",
              {.description = "flag for output of performance statistics "
                              "associated with nonlinear solver into csv file",
                  .default_value = false}),

          // flag for point-based null space calculation
          parameter<bool>(
              "NULLSPACE_POINTBASED", {.description = "flag for point-based null space calculation",
                                          .default_value = false}),

          // flag for adaptive time stepping
          parameter<bool>("ADAPTIVE_TIMESTEPPING",
              {.description = "flag for adaptive time stepping", .default_value = false})},
      {.defaultable =
              true}); /*----------------------------------------------------------------------*/
  list["SCALAR TRANSPORT DYNAMIC/NONLINEAR"] = group("SCALAR TRANSPORT DYNAMIC/NONLINEAR",
      {

          parameter<int>(
              "ITEMAX", {.description = "max. number of nonlin. iterations", .default_value = 10}),
          parameter<double>(
              "CONVTOL", {.description = "Tolerance for convergence check", .default_value = 1e-6}),
          parameter<int>("ITEMAX_OUTER",
              {.description = "Maximum number of outer iterations in partitioned coupling "
                              "schemes (natural convection, multi-scale simulations etc.)",
                  .default_value = 10}),
          parameter<double>("CONVTOL_OUTER",
              {.description =
                      "Convergence check tolerance for outer loop in partitioned coupling schemes "
                      "(natural convection, multi-scale simulations etc.)",
                  .default_value = 1e-6}),
          parameter<bool>("EXPLPREDICT",
              {.description = "do an explicit predictor step before starting nonlinear iteration",
                  .default_value = false}),
          parameter<double>(
              "ABSTOLRES", {.description = "Absolute tolerance for deciding if residual "
                                           "of nonlinear problem is already zero",
                               .default_value = 1e-14}),

          // convergence criteria adaptivity
          parameter<bool>("ADAPTCONV", {.description = "Switch on adaptive control of linear "
                                                       "solver tolerance for nonlinear solution",
                                           .default_value = false}),
          parameter<double>("ADAPTCONV_BETTER",
              {.description =
                      "The linear solver shall be this much better than the current nonlinear "
                      "residual in the nonlinear convergence limit",
                  .default_value = 0.1})},
      {.defaultable = true});

  /*----------------------------------------------------------------------*/
  list["SCALAR TRANSPORT DYNAMIC/STABILIZATION"] = group("SCALAR TRANSPORT DYNAMIC/STABILIZATION",
      {all_specs_for_scatra_stabilization()}, {.defaultable = true});

  // ----------------------------------------------------------------------
  // artery mesh tying
  list["SCALAR TRANSPORT DYNAMIC/ARTERY COUPLING"] = group(
      "SCALAR TRANSPORT DYNAMIC/ARTERY COUPLING",
      {

          deprecated_selection<Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod>(
              "ARTERY_COUPLING_METHOD",
              {
                  {"None", Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::none},
                  {"Nodal", Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::nodal},
                  {"GPTS", Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::gpts},
                  {"MP", Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::mp},
                  {"NTP", Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::ntp},
              },
              {.description = "Coupling method for artery coupling.",
                  .default_value =
                      Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod::none}),

          // penalty parameter
          parameter<double>("PENALTY", {.description = "Penalty parameter for line-based coupling",
                                           .default_value = 1000.0}),

          // coupled artery dofs for mesh tying
          parameter<std::string>("COUPLEDDOFS_ARTSCATRA",
              {.description = "coupled artery dofs for mesh tying", .default_value = "-1.0"}),

          // coupled porofluid dofs for mesh tying
          parameter<std::string>("COUPLEDDOFS_SCATRA",
              {.description = "coupled porofluid dofs for mesh tying", .default_value = "-1.0"}),

          // functions for coupling (arteryscatra part)
          parameter<std::string>("REACFUNCT_ART",
              {.description = "functions for coupling (arteryscatra part)", .default_value = "-1"}),

          // scale for coupling (arteryscatra part)
          parameter<std::string>("SCALEREAC_ART",
              {.description = "scale for coupling (arteryscatra part)", .default_value = "0"}),

          // functions for coupling (scatra part)
          parameter<std::string>("REACFUNCT_CONT",
              {.description = "functions for coupling (scatra part)", .default_value = "-1"}),

          // scale for coupling (scatra part)
          parameter<std::string>("SCALEREAC_CONT",
              {.description = "scale for coupling (scatra part)", .default_value = "0"})},
      {.defaultable = true});

  // ----------------------------------------------------------------------
  list["SCALAR TRANSPORT DYNAMIC/EXTERNAL FORCE"] = group("SCALAR TRANSPORT DYNAMIC/EXTERNAL FORCE",
      {

          // Flag for external force
          parameter<bool>("EXTERNAL_FORCE",
              {.description = "Flag to activate external force", .default_value = false}),

          // Function ID for external force
          parameter<int>("FORCE_FUNCTION_ID",
              {.description = "Function ID for external force", .default_value = -1}),

          // Function ID for mobility of the scalar
          parameter<int>("INTRINSIC_MOBILITY_FUNCTION_ID",
              {.description = "Function ID for intrinsic mobility", .default_value = -1})},
      {.defaultable = true});
}



void Inpar::ScaTra::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // Boundary flux evaluation condition for scalar transport
  Core::Conditions::ConditionDefinition linebndryfluxeval("SCATRA FLUX CALC LINE CONDITIONS",
      "ScaTraFluxCalc", "Scalar Transport Boundary Flux Calculation",
      Core::Conditions::ScaTraFluxCalc, true, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfbndryfluxeval("SCATRA FLUX CALC SURF CONDITIONS",
      "ScaTraFluxCalc", "Scalar Transport Boundary Flux Calculation",
      Core::Conditions::ScaTraFluxCalc, true, Core::Conditions::geometry_type_surface);
  condlist.emplace_back(linebndryfluxeval);
  condlist.emplace_back(surfbndryfluxeval);

  /*--------------------------------------------------------------------*/
  // conditions for calculation of total and mean values of transported scalars
  Core::Conditions::ConditionDefinition totalandmeanscalarline(
      "DESIGN TOTAL AND MEAN SCALAR LINE CONDITIONS", "TotalAndMeanScalar",
      "calculation of total and mean values of transported scalars",
      Core::Conditions::TotalAndMeanScalar, true, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition totalandmeanscalarsurf(
      "DESIGN TOTAL AND MEAN SCALAR SURF CONDITIONS", "TotalAndMeanScalar",
      "calculation of total and mean values of transported scalars",
      Core::Conditions::TotalAndMeanScalar, true, Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition totalandmeanscalarvol(
      "DESIGN TOTAL AND MEAN SCALAR VOL CONDITIONS", "TotalAndMeanScalar",
      "calculation of total and mean values of transported scalars",
      Core::Conditions::TotalAndMeanScalar, true, Core::Conditions::geometry_type_volume);

  const auto make_totalandmeanscalar = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    // insert input file line components into condition definitions
    cond.add_component(parameter<int>("ConditionID"));

    // insert condition definitions into global list of valid condition definitions
    condlist.emplace_back(cond);
  };

  make_totalandmeanscalar(totalandmeanscalarline);
  make_totalandmeanscalar(totalandmeanscalarsurf);
  make_totalandmeanscalar(totalandmeanscalarvol);

  /*--------------------------------------------------------------------*/
  // conditions for calculation of relative error with reference to analytical solution
  Core::Conditions::ConditionDefinition relerrorline("DESIGN SCATRA RELATIVE ERROR LINE CONDITIONS",
      "ScatraRelError", "calculation of relative error with reference to analytical solution",
      Core::Conditions::ScatraRelError, true, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition relerrorsurf("DESIGN SCATRA RELATIVE ERROR SURF CONDITIONS",
      "ScatraRelError", "calculation of relative error with reference to analytical solution",
      Core::Conditions::ScatraRelError, true, Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition relerrorvol("DESIGN SCATRA RELATIVE ERROR VOL CONDITIONS",
      "ScatraRelError", "calculation of relative error with reference to analytical solution",
      Core::Conditions::ScatraRelError, true, Core::Conditions::geometry_type_volume);

  const auto make_relerror = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    // insert input file line components into condition definitions
    cond.add_component(parameter<int>("ConditionID"));
    cond.add_component(parameter<int>("Function"));

    // insert condition definitions into global list of valid condition definitions
    condlist.emplace_back(cond);
  };

  make_relerror(relerrorline);
  make_relerror(relerrorsurf);
  make_relerror(relerrorvol);

  /*--------------------------------------------------------------------*/
  // Coupling of different scalar transport fields

  Core::Conditions::ConditionDefinition surfscatracoup("DESIGN SCATRA COUPLING SURF CONDITIONS",
      "ScaTraCoupling", "ScaTra Coupling", Core::Conditions::ScaTraCoupling, true,
      Core::Conditions::geometry_type_surface);

  surfscatracoup.add_component(parameter<int>("NUMSCAL"));
  surfscatracoup.add_component(parameter<std::vector<int>>(
      "ONOFF", {.description = "", .size = from_parameter<int>("NUMSCAL")}));
  surfscatracoup.add_component(parameter<int>("COUPID"));
  surfscatracoup.add_component(parameter<double>("PERMCOEF"));
  surfscatracoup.add_component(parameter<double>("CONDUCT"));
  surfscatracoup.add_component(parameter<double>("FILTR"));
  surfscatracoup.add_component(
      parameter<bool>("WSSON", {.description = "flag if wall shear stress coupling is on"}));
  surfscatracoup.add_component(
      parameter<std::vector<double>>("WSSCOEFFS", {.description = "", .size = 2}));

  condlist.emplace_back(surfscatracoup);

  /*--------------------------------------------------------------------*/
  // Robin boundary condition for scalar transport problems
  // line
  Core::Conditions::ConditionDefinition scatrarobinline("DESIGN TRANSPORT ROBIN LINE CONDITIONS",
      "TransportRobin", "Scalar Transport Robin Boundary Condition",
      Core::Conditions::TransportRobin, true, Core::Conditions::geometry_type_line);
  // surface
  Core::Conditions::ConditionDefinition scatrarobinsurf("DESIGN TRANSPORT ROBIN SURF CONDITIONS",
      "TransportRobin", "Scalar Transport Robin Boundary Condition",
      Core::Conditions::TransportRobin, true, Core::Conditions::geometry_type_surface);

  const auto make_scatrarobin = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(parameter<int>("NUMSCAL"));
    cond.add_component(parameter<std::vector<int>>(
        "ONOFF", {.description = "", .size = from_parameter<int>("NUMSCAL")}));
    cond.add_component(parameter<double>("PREFACTOR"));
    cond.add_component(parameter<double>("REFVALUE"));

    condlist.emplace_back(cond);
  };

  make_scatrarobin(scatrarobinline);
  make_scatrarobin(scatrarobinsurf);

  /*--------------------------------------------------------------------*/
  // Neumann inflow for SCATRA

  Core::Conditions::ConditionDefinition linetransportneumanninflow(
      "TRANSPORT NEUMANN INFLOW LINE CONDITIONS", "TransportNeumannInflow",
      "Line Transport Neumann Inflow", Core::Conditions::TransportNeumannInflow, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surftransportneumanninflow(
      "TRANSPORT NEUMANN INFLOW SURF CONDITIONS", "TransportNeumannInflow",
      "Surface Transport Neumann Inflow", Core::Conditions::TransportNeumannInflow, true,
      Core::Conditions::geometry_type_surface);

  condlist.emplace_back(linetransportneumanninflow);
  condlist.emplace_back(surftransportneumanninflow);

  /*--------------------------------------------------------------------*/
  // Scatra convective heat transfer (Newton's law of heat transfer)
  Core::Conditions::ConditionDefinition linetransportthermoconvect(
      "TRANSPORT THERMO CONVECTION LINE CONDITIONS", "TransportThermoConvections",
      "Line Transport Thermo Convections", Core::Conditions::TransportThermoConvections, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surftransportthermoconvect(
      "TRANSPORT THERMO CONVECTION SURF CONDITIONS", "TransportThermoConvections",
      "Surface Transport Thermo Convections", Core::Conditions::TransportThermoConvections, true,
      Core::Conditions::geometry_type_surface);

  const auto make_transportthermoconvect = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    // decide here if approximation is sufficient
    // --> Tempn (old temperature T_n)
    // or if the exact solution is needed
    // --> Tempnp (current temperature solution T_n+1) with linearisation
    cond.add_component(deprecated_selection<std::string>(
        "temperature_state", {"Tempnp", "Tempn"}, {.description = "temperature state"}));
    cond.add_component(parameter<double>("coeff", {.description = "heat transfer coefficient h"}));
    cond.add_component(
        parameter<double>("surtemp", {.description = "surrounding (fluid) temperature T_oo"}));
    cond.add_component(parameter<std::optional<int>>("surtempfunct",
        {.description =
                "time curve to increase the surrounding (fluid) temperature T_oo in time"}));
    cond.add_component(parameter<std::optional<int>>("funct",
        {.description =
                "time curve to increase the complete boundary condition, i.e., the heat flux"}));

    condlist.emplace_back(cond);
  };

  make_transportthermoconvect(linetransportthermoconvect);
  make_transportthermoconvect(surftransportthermoconvect);

  /*--------------------------------------------------------------------*/
  // conditions for calculation of calculation of heterogeneous reactions
  Core::Conditions::ConditionDefinition scatraheteroreactionmasterline(
      "DESIGN SCATRA HETEROGENEOUS REACTION LINE CONDITIONS / MASTER", "ScatraHeteroReactionMaster",
      "calculation of heterogeneous reactions", Core::Conditions::ScatraHeteroReactionCondMaster,
      true, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition scatraheteroreactionmastersurf(
      "DESIGN SCATRA HETEROGENEOUS REACTION SURF CONDITIONS / MASTER", "ScatraHeteroReactionMaster",
      "calculation of heterogeneous reactions", Core::Conditions::ScatraHeteroReactionCondMaster,
      true, Core::Conditions::geometry_type_surface);

  // insert condition definitions into global list of valid condition definitions
  condlist.emplace_back(scatraheteroreactionmasterline);
  condlist.emplace_back(scatraheteroreactionmastersurf);

  /*--------------------------------------------------------------------*/
  // conditions for calculation of calculation of heterogeneous reactions
  Core::Conditions::ConditionDefinition scatraheteroreactionslaveline(
      "DESIGN SCATRA HETEROGENEOUS REACTION LINE CONDITIONS / SLAVE", "ScatraHeteroReactionSlave",
      "calculation of heterogeneous reactions", Core::Conditions::ScatraHeteroReactionCondSlave,
      true, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition scatraheteroreactionslavesurf(
      "DESIGN SCATRA HETEROGENEOUS REACTION SURF CONDITIONS / SLAVE", "ScatraHeteroReactionSlave",
      "calculation of heterogeneous reactions", Core::Conditions::ScatraHeteroReactionCondSlave,
      true, Core::Conditions::geometry_type_surface);

  // insert condition definitions into global list of valid condition definitions
  condlist.emplace_back(scatraheteroreactionslaveline);
  condlist.emplace_back(scatraheteroreactionslavesurf);


  /*--------------------------------------------------------------------*/
  // scatra domain partitioning for block preconditioning of global system matrix
  // please note: this is currently only used in combination with scatra-scatra interface coupling
  // however the complete scatra matrix is subdivided into blocks which is not related to the
  // interface coupling at all
  {
    // partitioning of 2D domain into 2D subdomains
    Core::Conditions::ConditionDefinition scatrasurfpartitioning(
        "DESIGN SCATRA SURF CONDITIONS / PARTITIONING", "ScatraPartitioning",
        "Domain partitioning of scatra field", Core::Conditions::ScatraPartitioning, false,
        Core::Conditions::geometry_type_surface);

    // partitioning of 3D domain into 3D subdomains
    Core::Conditions::ConditionDefinition scatravolpartitioning(
        "DESIGN SCATRA VOL CONDITIONS / PARTITIONING", "ScatraPartitioning",
        "Domain partitioning of scatra field", Core::Conditions::ScatraPartitioning, false,
        Core::Conditions::geometry_type_volume);

    // insert condition definitions into global list of valid condition definitions
    condlist.emplace_back(scatrasurfpartitioning);
    condlist.emplace_back(scatravolpartitioning);
  }
}

std::string Inpar::ScaTra::impltype_to_string(ImplType impltype)
{
  switch (impltype)
  {
    case Inpar::ScaTra::impltype_undefined:
      return "Undefined";
    case Inpar::ScaTra::impltype_std:
      return "Std";
    case Inpar::ScaTra::impltype_loma:
      return "Loma";
    case Inpar::ScaTra::impltype_elch_NP:
      return "ElchNP";
    case Inpar::ScaTra::impltype_elch_electrode:
      return "ElchElectrode";
    case Inpar::ScaTra::impltype_elch_electrode_growth:
      return "ElchElectrodeGrowth";
    case Inpar::ScaTra::impltype_elch_electrode_thermo:
      return "ElchElectrodeThermo";
    case Inpar::ScaTra::impltype_elch_diffcond:
      return "ElchDiffCond";
    case Inpar::ScaTra::impltype_elch_diffcond_multiscale:
      return "ElchDiffCondMultiScale";
    case Inpar::ScaTra::impltype_elch_diffcond_thermo:
      return "ElchDiffCondThermo";
    case Inpar::ScaTra::impltype_elch_scl:
      return "ElchScl";
    case Inpar::ScaTra::impltype_thermo_elch_electrode:
      return "ThermoElchElectrode";
    case Inpar::ScaTra::impltype_thermo_elch_diffcond:
      return "ThermoElchDiffCond";
    case Inpar::ScaTra::impltype_lsreinit:
      return "LsReinit";
    case Inpar::ScaTra::impltype_levelset:
      return "Ls";
    case Inpar::ScaTra::impltype_poro:
      return "Poro";
    case Inpar::ScaTra::impltype_advreac:
      return "Advanced_Reaction";
    case Inpar::ScaTra::impltype_multipororeac:
      return "PoroMultiReac";
    case Inpar::ScaTra::impltype_pororeac:
      return "PoroReac";
    case Inpar::ScaTra::impltype_pororeacECM:
      return "PoroReacECM";
    case Inpar::ScaTra::impltype_aniso:
      return "Aniso";
    case Inpar::ScaTra::impltype_cardiac_monodomain:
      return "CardMono";
    case Inpar::ScaTra::impltype_chemo:
      return "Chemotaxis";
    case Inpar::ScaTra::impltype_chemoreac:
      return "Chemo_Reac";
    case Inpar::ScaTra::impltype_std_hdg:
      return "Hdg";
    case Inpar::ScaTra::impltype_cardiac_monodomain_hdg:
      return "HdgCardMono";
    case Inpar::ScaTra::impltype_one_d_artery:
      return "OneDArtery";
    case Inpar::ScaTra::impltype_no_physics:
      return "NoPhysics";
  }

  FOUR_C_THROW("Unknown implementation type given: {}", impltype);
}

Core::IO::InputSpec Inpar::ScaTra::all_specs_for_scatra_stabilization()
{
  using namespace Core::IO::InputSpecBuilders;
  return all_of({// this parameter governs type of stabilization
      deprecated_selection<Inpar::ScaTra::StabType>("STABTYPE",
          {
              {"no_stabilization", stabtype_no_stabilization},
              {"SUPG", stabtype_SUPG},
              {"GLS", stabtype_GLS},
              {"USFEM", stabtype_USFEM},
              {"centered", stabtype_hdg_centered},
              {"upwind", stabtype_hdg_upwind},
          },
          {.description = "Type of stabilization (if any). No stabilization is only reasonable for "
                          "low-Peclet-number.",
              .default_value = stabtype_SUPG}),

      // this parameter governs whether subgrid-scale velocity is included
      parameter<bool>(
          "SUGRVEL", {.description = "potential incorporation of subgrid-scale velocity",
                         .default_value = false}),

      // this parameter governs whether all-scale subgrid diffusivity is included
      parameter<bool>(
          "ASSUGRDIFF", {.description = "potential incorporation of all-scale subgrid diffusivity "
                                        "(a.k.a. discontinuity-capturing) term",
                            .default_value = false}),

      // this parameter selects the tau definition applied
      deprecated_selection<Inpar::ScaTra::TauType>("DEFINITION_TAU",
          {
              {"Taylor_Hughes_Zarins", tau_taylor_hughes_zarins},
              {"Taylor_Hughes_Zarins_wo_dt", tau_taylor_hughes_zarins_wo_dt},
              {"Franca_Valentin", tau_franca_valentin},
              {"Franca_Valentin_wo_dt", tau_franca_valentin_wo_dt},
              {"Shakib_Hughes_Codina", tau_shakib_hughes_codina},
              {"Shakib_Hughes_Codina_wo_dt", tau_shakib_hughes_codina_wo_dt},
              {"Codina", tau_codina},
              {"Codina_wo_dt", tau_codina_wo_dt},
              {"Franca_Madureira_Valentin", tau_franca_madureira_valentin},
              {"Franca_Madureira_Valentin_wo_dt", tau_franca_madureira_valentin_wo_dt},
              {"Exact_1D", tau_exact_1d},
              {"Zero", tau_zero},
              {"Numerical_Value", tau_numerical_value},
          },
          {.description = "Definition of tau", .default_value = tau_franca_valentin}),

      // this parameter selects the characteristic element length for tau for all
      // stabilization parameter definitions requiring such a length
      parameter<Inpar::ScaTra::CharEleLength>("CHARELELENGTH",
          {.description = "Characteristic element length for tau", .default_value = streamlength}),

      // this parameter selects the all-scale subgrid-diffusivity definition applied
      deprecated_selection<Inpar::ScaTra::AssgdType>("DEFINITION_ASSGD",
          {
              {"artificial_linear", assgd_artificial},
              {"artificial_linear_reinit", assgd_lin_reinit},
              {"Hughes_etal_86_nonlinear", assgd_hughes},
              {"Tezduyar_Park_86_nonlinear", assgd_tezduyar},
              {"Tezduyar_Park_86_nonlinear_wo_phizero", assgd_tezduyar_wo_phizero},
              {"doCarmo_Galeao_91_nonlinear", assgd_docarmo},
              {"Almeida_Silva_97_nonlinear", assgd_almeida},
              {"YZbeta_nonlinear", assgd_yzbeta},
              {"Codina_nonlinear", assgd_codina},
          },
          {.description = "Definition of (all-scale) subgrid diffusivity",
              .default_value = assgd_artificial}),

      // this parameter selects the location where tau is evaluated
      deprecated_selection<Inpar::ScaTra::EvalTau>("EVALUATION_TAU",
          {
              {"element_center", evaltau_element_center},
              {"integration_point", evaltau_integration_point},
          },
          {.description = "Location where tau is evaluated",
              .default_value = evaltau_element_center}),

      // this parameter selects the location where the material law is evaluated
      // (does not fit here very well, but parameter transfer is easier)
      deprecated_selection<Inpar::ScaTra::EvalMat>("EVALUATION_MAT",
          {
              {"element_center", evalmat_element_center},
              {"integration_point", evalmat_integration_point},
          },
          {.description = "Location where material law is evaluated",
              .default_value = evalmat_element_center}),

      // this parameter selects methods for improving consistency of stabilization terms
      deprecated_selection<Inpar::ScaTra::Consistency>("CONSISTENCY",
          {
              {"no", consistency_no},
              {"L2_projection_lumped", consistency_l2_projection_lumped},
          },
          {.description = "improvement of consistency for stabilization",
              .default_value = consistency_no}),

      // this parameter defines the numerical value, if stabilization with numerical values is
      // used
      parameter<double>("TAU_VALUE",
          {.description = "Numerical value for tau for stabilization", .default_value = 0.0})});
}


FOUR_C_NAMESPACE_CLOSE