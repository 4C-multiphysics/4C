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
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.hpp"

#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

void Inpar::ScaTra::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs scatradyn{"SCALAR TRANSPORT DYNAMIC"};

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::SolverType>("SOLVERTYPE", "linear_full",
      "type of scalar transport solver",
      tuple<std::string>("linear_full", "linear_incremental", "nonlinear",
          "nonlinear_multiscale_macrotomicro", "nonlinear_multiscale_macrotomicro_aitken",
          "nonlinear_multiscale_macrotomicro_aitken_dofsplit", "nonlinear_multiscale_microtomacro"),
      tuple<Inpar::ScaTra::SolverType>(solvertype_linear_full, solvertype_linear_incremental,
          solvertype_nonlinear, solvertype_nonlinear_multiscale_macrotomicro,
          solvertype_nonlinear_multiscale_macrotomicro_aitken,
          solvertype_nonlinear_multiscale_macrotomicro_aitken_dofsplit,
          solvertype_nonlinear_multiscale_microtomacro),
      scatradyn);

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::TimeIntegrationScheme>("TIMEINTEGR",
      "One_Step_Theta", "Time Integration Scheme",
      tuple<std::string>("Stationary", "One_Step_Theta", "BDF2", "Gen_Alpha"),
      tuple<Inpar::ScaTra::TimeIntegrationScheme>(
          timeint_stationary, timeint_one_step_theta, timeint_bdf2, timeint_gen_alpha),
      scatradyn);

  scatradyn.specs.emplace_back(parameter<double>(
      "MAXTIME", {.description = "Total simulation time", .default_value = 1000.0}));
  Core::Utils::int_parameter("NUMSTEP", 20, "Total number of time steps", scatradyn);
  scatradyn.specs.emplace_back(
      parameter<double>("TIMESTEP", {.description = "Time increment dt", .default_value = 0.1}));
  scatradyn.specs.emplace_back(parameter<double>(
      "THETA", {.description = "One-step-theta time integration factor", .default_value = 0.5}));
  scatradyn.specs.emplace_back(parameter<double>("ALPHA_M",
      {.description = "Generalized-alpha time integration factor", .default_value = 0.5}));
  scatradyn.specs.emplace_back(parameter<double>("ALPHA_F",
      {.description = "Generalized-alpha time integration factor", .default_value = 0.5}));
  scatradyn.specs.emplace_back(parameter<double>(
      "GAMMA", {.description = "Generalized-alpha time integration factor", .default_value = 0.5}));
  Core::Utils::int_parameter("RESULTSEVERY", 1, "Increment for writing solution", scatradyn);
  Core::Utils::int_parameter("RESTARTEVERY", 1, "Increment for writing restart", scatradyn);
  Core::Utils::int_parameter("MATID", -1, "Material ID for automatic mesh generation", scatradyn);

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::VelocityField>("VELOCITYFIELD", "zero",
      "type of velocity field used for scalar transport problems",
      tuple<std::string>("zero", "function", "Navier_Stokes"),
      tuple<Inpar::ScaTra::VelocityField>(velocity_zero, velocity_function, velocity_Navier_Stokes),
      scatradyn);

  Core::Utils::int_parameter(
      "VELFUNCNO", -1, "function number for scalar transport velocity field", scatradyn);

  {
    // a standard Teuchos::tuple can have at maximum 10 entries! We have to circumvent this here.
    Teuchos::Tuple<std::string, 13> name;
    Teuchos::Tuple<Inpar::ScaTra::InitialField, 13> label;
    name[0] = "zero_field";
    label[0] = initfield_zero_field;
    name[1] = "field_by_function";
    label[1] = initfield_field_by_function;
    name[2] = "field_by_condition";
    label[2] = initfield_field_by_condition;
    name[3] = "disturbed_field_by_function";
    label[3] = initfield_disturbed_field_by_function;
    name[4] = "1D_DISCONTPV";
    label[4] = initfield_discontprogvar_1D;
    name[5] = "FLAME_VORTEX_INTERACTION";
    label[5] = initfield_flame_vortex_interaction;
    name[6] = "RAYTAYMIXFRAC";
    label[6] = initfield_raytaymixfrac;
    name[7] = "L_shaped_domain";
    label[7] = initfield_Lshapeddomain;
    name[8] = "facing_flame_fronts";
    label[8] = initfield_facing_flame_fronts;
    name[9] = "oracles_flame";
    label[9] = initfield_oracles_flame;
    name[10] = "high_forced_hit";
    label[10] = initialfield_forced_hit_high_Sc;
    name[11] = "low_forced_hit";
    label[11] = initialfield_forced_hit_low_Sc;
    name[12] = "algebraic_field_dependence";
    label[12] = initialfield_algebraic_field_dependence;

    Core::Utils::string_to_integral_parameter<Inpar::ScaTra::InitialField>("INITIALFIELD",
        "zero_field", "Initial Field for scalar transport problem", name, label, scatradyn);
  }

  Core::Utils::int_parameter(
      "INITFUNCNO", -1, "function number for scalar transport initial field", scatradyn);

  scatradyn.specs.emplace_back(parameter<bool>(
      "SPHERICALCOORDS", {.description = "use of spherical coordinates", .default_value = false}));

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::CalcError>("CALCERROR", "No",
      "compute error compared to analytical solution",
      tuple<std::string>("No", "Kwok_Wu", "ConcentricCylinders", "Electroneutrality",
          "error_by_function", "error_by_condition", "SphereDiffusion", "AnalyticSeries"),
      tuple<Inpar::ScaTra::CalcError>(calcerror_no, calcerror_Kwok_Wu, calcerror_cylinder,
          calcerror_electroneutrality, calcerror_byfunction, calcerror_bycondition,
          calcerror_spherediffusion, calcerror_AnalyticSeries),
      scatradyn);

  Core::Utils::int_parameter(
      "CALCERRORNO", -1, "function number for scalar transport error computation", scatradyn);

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::FluxType>("CALCFLUX_DOMAIN", "No",
      "output of diffusive/total flux vectors inside domain",
      tuple<std::string>("No", "total", "diffusive"),
      tuple<Inpar::ScaTra::FluxType>(flux_none, flux_total, flux_diffusive), scatradyn);

  scatradyn.specs.emplace_back(parameter<bool>("CALCFLUX_DOMAIN_LUMPED",
      {.description = "perform approximate domain flux calculation involving matrix lumping",
          .default_value = true}));

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::FluxType>("CALCFLUX_BOUNDARY", "No",
      "output of convective/diffusive/total flux vectors on boundary",
      tuple<std::string>("No", "total", "diffusive", "convective"),
      tuple<Inpar::ScaTra::FluxType>(flux_none, flux_total, flux_diffusive, flux_convective),
      scatradyn);

  scatradyn.specs.emplace_back(parameter<bool>("CALCFLUX_BOUNDARY_LUMPED",
      {.description = "perform approximate boundary flux calculation involving matrix lumping",
          .default_value = true}));

  scatradyn.specs.emplace_back(parameter<std::string>(
      "WRITEFLUX_IDS", {.description = "Write diffusive/total flux vector fields for these scalar "
                                       "fields only (starting with 1)",
                           .default_value = "-1"}));

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::OutputScalarType>("OUTPUTSCALARS",
      "none", "Output of total and mean values for transported scalars",
      tuple<std::string>("none", "entire_domain", "by_condition", "entire_domain_and_by_condition"),
      tuple<Inpar::ScaTra::OutputScalarType>(outputscalars_none, outputscalars_entiredomain,
          outputscalars_condition, outputscalars_entiredomain_condition),
      scatradyn);
  scatradyn.specs.emplace_back(parameter<bool>("OUTPUTSCALARSMEANGRAD",
      {.description = "Output of mean gradient of scalars", .default_value = false}));
  scatradyn.specs.emplace_back(parameter<bool>("OUTINTEGRREAC",
      {.description = "Output of integral reaction values", .default_value = false}));
  scatradyn.specs.emplace_back(parameter<bool>("OUTPUT_GMSH",
      {.description = "Do you want to write Gmsh postprocessing files?", .default_value = false}));

  scatradyn.specs.emplace_back(parameter<bool>("MATLAB_STATE_OUTPUT",
      {.description = "Do you want to write the state solution to Matlab file?",
          .default_value = false}));

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::ConvForm>("CONVFORM", "convective",
      "form of convective term", tuple<std::string>("convective", "conservative"),
      tuple<Inpar::ScaTra::ConvForm>(convform_convective, convform_conservative), scatradyn);

  scatradyn.specs.emplace_back(parameter<bool>(
      "NEUMANNINFLOW", {.description = "Flag to (de)activate potential Neumann inflow term(s)",
                           .default_value = false}));

  scatradyn.specs.emplace_back(parameter<bool>("CONV_HEAT_TRANS",
      {.description = "Flag to (de)activate potential convective heat transfer boundary conditions",
          .default_value = false}));

  scatradyn.specs.emplace_back(parameter<bool>(
      "SKIPINITDER", {.description = "Flag to skip computation of initial time derivative",
                         .default_value = false}));

  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::FSSUGRDIFF>("FSSUGRDIFF", "No",
      "fine-scale subgrid diffusivity",
      tuple<std::string>("No", "artificial", "Smagorinsky_all", "Smagorinsky_small"),
      tuple<Inpar::ScaTra::FSSUGRDIFF>(fssugrdiff_no, fssugrdiff_artificial,
          fssugrdiff_smagorinsky_all, fssugrdiff_smagorinsky_small),
      scatradyn);

  // flag for output of performance statistics associated with nonlinear solver into *.csv file
  scatradyn.specs.emplace_back(parameter<bool>("ELECTROMAGNETICDIFFUSION",
      {.description = "flag to activate electromagnetic diffusion problems",
          .default_value = false}));

  // Current density source function for EMD problems
  Core::Utils::int_parameter("EMDSOURCE", -1, "Current density source", scatradyn);

  Core::Utils::string_to_integral_parameter<Inpar::FLUID::MeshTying>("MESHTYING", "no",
      "Flag to (de)activate mesh tying algorithm",
      tuple<std::string>("no", "Condensed_Smat", "Condensed_Bmat", "Condensed_Bmat_merged"),
      tuple<Inpar::FLUID::MeshTying>(Inpar::FLUID::no_meshtying, Inpar::FLUID::condensed_smat,
          Inpar::FLUID::condensed_bmat, Inpar::FLUID::condensed_bmat_merged),
      scatradyn);

  // Type of coupling strategy between the two fields
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::FieldCoupling>("FIELDCOUPLING",
      "matching", "Type of coupling strategy between fields",
      tuple<std::string>("matching", "volmortar"),
      tuple<Inpar::ScaTra::FieldCoupling>(coupling_match, coupling_volmortar), scatradyn);

  // linear solver id used for scalar transport/elch problems
  Core::Utils::int_parameter(
      "LINEAR_SOLVER", -1, "number of linear solver used for scalar transport/elch...", scatradyn);
  // linear solver id used for l2 projection problems (e.g. gradient projections)
  Core::Utils::int_parameter("L2_PROJ_LINEAR_SOLVER", -1,
      "number of linear solver used for l2-projection sub-problems", scatradyn);

  // flag for equilibration of global system of equations
  Core::Utils::string_to_integral_parameter<Core::LinAlg::EquilibrationMethod>("EQUILIBRATION",
      "none", "flag for equilibration of global system of equations",
      tuple<std::string>("none", "rows_full", "rows_maindiag", "columns_full", "columns_maindiag",
          "rowsandcolumns_full", "rowsandcolumns_maindiag"),
      tuple<Core::LinAlg::EquilibrationMethod>(Core::LinAlg::EquilibrationMethod::none,
          Core::LinAlg::EquilibrationMethod::rows_full,
          Core::LinAlg::EquilibrationMethod::rows_maindiag,
          Core::LinAlg::EquilibrationMethod::columns_full,
          Core::LinAlg::EquilibrationMethod::columns_maindiag,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_full,
          Core::LinAlg::EquilibrationMethod::rowsandcolumns_maindiag),
      scatradyn);

  // type of global system matrix in global system of equations
  Core::Utils::string_to_integral_parameter<Core::LinAlg::MatrixType>("MATRIXTYPE", "sparse",
      "type of global system matrix in global system of equations",
      tuple<std::string>("sparse", "block_condition", "block_condition_dof"),
      tuple<Core::LinAlg::MatrixType>(Core::LinAlg::MatrixType::sparse,
          Core::LinAlg::MatrixType::block_condition, Core::LinAlg::MatrixType::block_condition_dof),
      scatradyn);

  // flag for natural convection effects
  scatradyn.specs.emplace_back(parameter<bool>("NATURAL_CONVECTION",
      {.description = "Include natural convection effects", .default_value = false}));

  // parameters for finite difference check
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::FdCheck>("FDCHECK", "none",
      "flag for finite difference check: none, local, or global",
      tuple<std::string>("none",
          "global",           // perform finite difference check on time integrator level
          "global_extended",  // perform finite difference check on time integrator level for
                              // extended system matrix (e.g., involving Lagrange multipliers or
                              // interface layer thicknesses)
          "local"             // perform finite difference check on element level
          ),
      tuple<Inpar::ScaTra::FdCheck>(
          fdcheck_none, fdcheck_global, fdcheck_global_extended, fdcheck_local),
      scatradyn);
  scatradyn.specs.emplace_back(parameter<double>(
      "FDCHECKEPS", {.description = "dof perturbation magnitude for finite difference check (1.e-6 "
                                    "seems to work very well, whereas smaller values don't)",
                        .default_value = 1.e-6}));
  scatradyn.specs.emplace_back(parameter<double>("FDCHECKTOL",
      {.description = "relative tolerance for finite difference check", .default_value = 1.e-6}));

  // parameter for optional computation of domain and boundary integrals, i.e., of surface areas and
  // volumes associated with specified nodesets
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::ComputeIntegrals>("COMPUTEINTEGRALS",
      "none", "flag for optional computation of domain integrals",
      tuple<std::string>("none", "initial", "repeated"),
      tuple<Inpar::ScaTra::ComputeIntegrals>(
          computeintegrals_none, computeintegrals_initial, computeintegrals_repeated),
      scatradyn);

  // parameter for using p-adpativity and semi-implicit evaluation of the reaction term (at the
  // moment only used for HDG and cardiac monodomain problems)
  scatradyn.specs.emplace_back(parameter<bool>(
      "PADAPTIVITY", {.description = "Flag to (de)activate p-adativity", .default_value = false}));
  scatradyn.specs.emplace_back(parameter<double>("PADAPTERRORTOL",
      {.description = "The error tolerance to calculate the variation of the elemental degree",
          .default_value = 1e-6}));
  scatradyn.specs.emplace_back(parameter<double>("PADAPTERRORBASE",
      {.description = "The error tolerance base to calculate the variation of the elemental degree",
          .default_value = 1.66}));
  Core::Utils::int_parameter(
      "PADAPTDEGREEMAX", 4, "The max. degree of the shape functions", scatradyn);
  scatradyn.specs.emplace_back(parameter<bool>("SEMIIMPLICIT",
      {.description = "Flag to (de)activate semi-implicit calculation of the reaction term",
          .default_value = false}));

  // flag for output of performance statistics associated with linear solver into *.csv file
  scatradyn.specs.emplace_back(parameter<bool>(
      "OUTPUTLINSOLVERSTATS", {.description = "flag for output of performance statistics "
                                              "associated with linear solver into csv file",
                                  .default_value = false}));

  // flag for output of performance statistics associated with nonlinear solver into *.csv file
  scatradyn.specs.emplace_back(parameter<bool>(
      "OUTPUTNONLINSOLVERSTATS", {.description = "flag for output of performance statistics "
                                                 "associated with nonlinear solver into csv file",
                                     .default_value = false}));

  // flag for point-based null space calculation
  scatradyn.specs.emplace_back(parameter<bool>("NULLSPACE_POINTBASED",
      {.description = "flag for point-based null space calculation", .default_value = false}));

  // flag for adaptive time stepping
  scatradyn.specs.emplace_back(parameter<bool>("ADAPTIVE_TIMESTEPPING",
      {.description = "flag for adaptive time stepping", .default_value = false}));

  scatradyn.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs scatra_nonlin{scatradyn, "NONLINEAR"};

  Core::Utils::int_parameter("ITEMAX", 10, "max. number of nonlin. iterations", scatra_nonlin);
  scatra_nonlin.specs.emplace_back(parameter<double>(
      "CONVTOL", {.description = "Tolerance for convergence check", .default_value = 1e-6}));
  Core::Utils::int_parameter("ITEMAX_OUTER", 10,
      "Maximum number of outer iterations in partitioned coupling schemes (natural convection, "
      "multi-scale simulations etc.)",
      scatra_nonlin);
  scatra_nonlin.specs.emplace_back(parameter<double>("CONVTOL_OUTER",
      {.description = "Convergence check tolerance for outer loop in partitioned coupling schemes "
                      "(natural convection, multi-scale simulations etc.)",
          .default_value = 1e-6}));
  scatra_nonlin.specs.emplace_back(parameter<bool>("EXPLPREDICT",
      {.description = "do an explicit predictor step before starting nonlinear iteration",
          .default_value = false}));
  scatra_nonlin.specs.emplace_back(parameter<double>("ABSTOLRES",
      {.description =
              "Absolute tolerance for deciding if residual of nonlinear problem is already zero",
          .default_value = 1e-14}));

  // convergence criteria adaptivity
  scatra_nonlin.specs.emplace_back(parameter<bool>("ADAPTCONV",
      {.description =
              "Switch on adaptive control of linear solver tolerance for nonlinear solution",
          .default_value = false}));
  scatra_nonlin.specs.emplace_back(parameter<double>("ADAPTCONV_BETTER",
      {.description = "The linear solver shall be this much better than the current nonlinear "
                      "residual in the nonlinear convergence limit",
          .default_value = 0.1}));

  scatra_nonlin.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs scatradyn_stab{scatradyn, "STABILIZATION"};

  // this parameter governs type of stabilization
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::StabType>("STABTYPE", "SUPG",
      "Type of stabilization (if any). No stabilization is only reasonable for low-Peclet-number.",
      tuple<std::string>("no_stabilization", "SUPG", "GLS", "USFEM", "centered", "upwind"),
      tuple<Inpar::ScaTra::StabType>(stabtype_no_stabilization, stabtype_SUPG, stabtype_GLS,
          stabtype_USFEM, stabtype_hdg_centered, stabtype_hdg_upwind),
      scatradyn_stab);

  // this parameter governs whether subgrid-scale velocity is included
  scatradyn_stab.specs.emplace_back(parameter<bool>(
      "SUGRVEL", {.description = "potential incorporation of subgrid-scale velocity",
                     .default_value = false}));

  // this parameter governs whether all-scale subgrid diffusivity is included
  scatradyn_stab.specs.emplace_back(parameter<bool>(
      "ASSUGRDIFF", {.description = "potential incorporation of all-scale subgrid diffusivity "
                                    "(a.k.a. discontinuity-capturing) term",
                        .default_value = false}));

  // this parameter selects the tau definition applied
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::TauType>("DEFINITION_TAU",
      "Franca_Valentin", "Definition of tau",
      tuple<std::string>("Taylor_Hughes_Zarins", "Taylor_Hughes_Zarins_wo_dt", "Franca_Valentin",
          "Franca_Valentin_wo_dt", "Shakib_Hughes_Codina", "Shakib_Hughes_Codina_wo_dt", "Codina",
          "Codina_wo_dt", "Franca_Madureira_Valentin", "Franca_Madureira_Valentin_wo_dt",
          "Exact_1D", "Zero", "Numerical_Value"),
      tuple<Inpar::ScaTra::TauType>(tau_taylor_hughes_zarins, tau_taylor_hughes_zarins_wo_dt,
          tau_franca_valentin, tau_franca_valentin_wo_dt, tau_shakib_hughes_codina,
          tau_shakib_hughes_codina_wo_dt, tau_codina, tau_codina_wo_dt,
          tau_franca_madureira_valentin, tau_franca_madureira_valentin_wo_dt, tau_exact_1d,
          tau_zero, tau_numerical_value),
      scatradyn_stab);

  // this parameter selects the characteristic element length for tau for all
  // stabilization parameter definitions requiring such a length
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::CharEleLength>("CHARELELENGTH",
      "streamlength", "Characteristic element length for tau",
      tuple<std::string>("streamlength", "volume_equivalent_diameter", "root_of_volume"),
      tuple<Inpar::ScaTra::CharEleLength>(streamlength, volume_equivalent_diameter, root_of_volume),
      scatradyn_stab);

  // this parameter selects the all-scale subgrid-diffusivity definition applied
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::AssgdType>("DEFINITION_ASSGD",
      "artificial_linear", "Definition of (all-scale) subgrid diffusivity",
      tuple<std::string>("artificial_linear", "artificial_linear_reinit",
          "Hughes_etal_86_nonlinear", "Tezduyar_Park_86_nonlinear",
          "Tezduyar_Park_86_nonlinear_wo_phizero", "doCarmo_Galeao_91_nonlinear",
          "Almeida_Silva_97_nonlinear", "YZbeta_nonlinear", "Codina_nonlinear"),
      tuple<Inpar::ScaTra::AssgdType>(assgd_artificial, assgd_lin_reinit, assgd_hughes,
          assgd_tezduyar, assgd_tezduyar_wo_phizero, assgd_docarmo, assgd_almeida, assgd_yzbeta,
          assgd_codina),
      scatradyn_stab);

  // this parameter selects the location where tau is evaluated
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::EvalTau>("EVALUATION_TAU",
      "element_center", "Location where tau is evaluated",
      tuple<std::string>("element_center", "integration_point"),
      tuple<Inpar::ScaTra::EvalTau>(evaltau_element_center, evaltau_integration_point),
      scatradyn_stab);

  // this parameter selects the location where the material law is evaluated
  // (does not fit here very well, but parameter transfer is easier)
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::EvalMat>("EVALUATION_MAT",
      "element_center", "Location where material law is evaluated",
      tuple<std::string>("element_center", "integration_point"),
      tuple<Inpar::ScaTra::EvalMat>(evalmat_element_center, evalmat_integration_point),
      scatradyn_stab);

  // this parameter selects methods for improving consistency of stabilization terms
  Core::Utils::string_to_integral_parameter<Inpar::ScaTra::Consistency>("CONSISTENCY", "no",
      "improvement of consistency for stabilization",
      tuple<std::string>("no", "L2_projection_lumped"),
      tuple<Inpar::ScaTra::Consistency>(consistency_no, consistency_l2_projection_lumped),
      scatradyn_stab);

  // this parameter defines the numerical value, if stabilization with numerical values is used
  scatradyn_stab.specs.emplace_back(parameter<double>("TAU_VALUE",
      {.description = "Numerical value for tau for stabilization", .default_value = 0.0}));

  scatradyn_stab.move_into_collection(list);

  // ----------------------------------------------------------------------
  // artery mesh tying
  Core::Utils::SectionSpecs scatradyn_art{scatradyn, "ARTERY COUPLING"};

  Core::Utils::string_to_integral_parameter<
      Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod>("ARTERY_COUPLING_METHOD",
      "None", "Coupling method for artery coupling.",
      tuple<std::string>("None", "Nodal", "GPTS", "MP", "NTP"),
      tuple<Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod>(
          Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::none,   // none
          Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::nodal,  // Nodal Coupling
          Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::
              gpts,  // Gauss-Point-To-Segment
          Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::mp,  // Mortar Penalty
          Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod::ntp  // 1D node-to-point
                                                                               // in 2D/3D
          ),
      scatradyn_art);

  // penalty parameter
  scatradyn_art.specs.emplace_back(parameter<double>("PENALTY",
      {.description = "Penalty parameter for line-based coupling", .default_value = 1000.0}));

  // coupled artery dofs for mesh tying
  scatradyn_art.specs.emplace_back(parameter<std::string>("COUPLEDDOFS_ARTSCATRA",
      {.description = "coupled artery dofs for mesh tying", .default_value = "-1.0"}));

  // coupled porofluid dofs for mesh tying
  scatradyn_art.specs.emplace_back(parameter<std::string>("COUPLEDDOFS_SCATRA",
      {.description = "coupled porofluid dofs for mesh tying", .default_value = "-1.0"}));

  // functions for coupling (arteryscatra part)
  scatradyn_art.specs.emplace_back(parameter<std::string>("REACFUNCT_ART",
      {.description = "functions for coupling (arteryscatra part)", .default_value = "-1"}));

  // scale for coupling (arteryscatra part)
  scatradyn_art.specs.emplace_back(parameter<std::string>("SCALEREAC_ART",
      {.description = "scale for coupling (arteryscatra part)", .default_value = "0"}));

  // functions for coupling (scatra part)
  scatradyn_art.specs.emplace_back(parameter<std::string>("REACFUNCT_CONT",
      {.description = "functions for coupling (scatra part)", .default_value = "-1"}));

  // scale for coupling (scatra part)
  scatradyn_art.specs.emplace_back(parameter<std::string>(
      "SCALEREAC_CONT", {.description = "scale for coupling (scatra part)", .default_value = "0"}));

  scatradyn_art.move_into_collection(list);

  // ----------------------------------------------------------------------
  Core::Utils::SectionSpecs scatradyn_external_force{scatradyn, "EXTERNAL FORCE"};

  // Flag for external force
  scatradyn_external_force.specs.emplace_back(parameter<bool>("EXTERNAL_FORCE",
      {.description = "Flag to activate external force", .default_value = false}));

  // Function ID for external force
  Core::Utils::int_parameter(
      "FORCE_FUNCTION_ID", -1, "Function ID for external force", scatradyn_external_force);

  // Function ID for mobility of the scalar
  Core::Utils::int_parameter("INTRINSIC_MOBILITY_FUNCTION_ID", -1,
      "Function ID for intrinsic mobility", scatradyn_external_force);

  scatradyn_external_force.move_into_collection(list);
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
    cond.add_component(selection<std::string>(
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

  FOUR_C_THROW("Unknown implementation type given: %d", impltype);
}

FOUR_C_NAMESPACE_CLOSE
