// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_fluid.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_geometry_type.hpp"
#include "4C_io_linecomponent.hpp"
#include "4C_legacy_enum_definitions_conditions.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::FLUID::set_valid_parameters(Teuchos::ParameterList& list)
{
  using namespace Input;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& fdyn = list.sublist("FLUID DYNAMIC", false, "");

  // physical type of fluid flow (incompressible, varying density, loma, Boussinesq approximation,
  // temperature-dependent water)
  setStringToIntegralParameter<Inpar::FLUID::PhysicalType>("PHYSICAL_TYPE", "Incompressible",
      "Physical Type",
      tuple<std::string>("Incompressible", "Weakly_compressible", "Weakly_compressible_stokes",
          "Weakly_compressible_dens_mom", "Weakly_compressible_stokes_dens_mom",
          "Artificial_compressibility", "Varying_density", "Loma", "Temp_dep_water", "Boussinesq",
          "Stokes", "Oseen"),
      tuple<Inpar::FLUID::PhysicalType>(incompressible, weakly_compressible,
          weakly_compressible_stokes, weakly_compressible_dens_mom,
          weakly_compressible_stokes_dens_mom, artcomp, varying_density, loma, tempdepwater,
          boussinesq, stokes, oseen),
      &fdyn);

  // number of linear solver used for fluid problem
  Core::Utils::int_parameter(
      "LINEAR_SOLVER", -1, "number of linear solver used for fluid dynamics", &fdyn);

  // number of linear solver used for fluid problem (former fluid pressure solver for SIMPLER
  // preconditioning with fluid)
  Core::Utils::int_parameter("SIMPLER_SOLVER", -1,
      "number of linear solver used for fluid dynamics (ONLY NECESSARY FOR BlockGaussSeidel solver "
      "block within fluid mehstying case any more!!!!)",
      &fdyn);

  // Flag to define the way of calculating stresses and wss
  setStringToIntegralParameter<Inpar::FLUID::WSSType>("WSS_TYPE", "Standard",
      "which type of stresses and wss", tuple<std::string>("Standard", "Aggregation", "Mean"),
      tuple<std::string>(
          "calculate 'normal' wss", "calculate aggregated wss", "calculate mean wss"),
      tuple<Inpar::FLUID::WSSType>(wss_standard, wss_aggregation, wss_mean), &fdyn);

  // Set ML-solver number for smoothing of residual-based calculated wallshearstress via plain
  // aggregation.
  Core::Utils::int_parameter("WSS_ML_AGR_SOLVER", -1,
      "Set ML-solver number for smoothing of residual-based calculated wallshearstress via plain "
      "aggregation.",
      &fdyn);

  setStringToIntegralParameter<Inpar::FLUID::TimeIntegrationScheme>("TIMEINTEGR", "One_Step_Theta",
      "Time Integration Scheme",
      tuple<std::string>("Stationary", "Np_Gen_Alpha", "Af_Gen_Alpha", "One_Step_Theta", "BDF2"),
      tuple<Inpar::FLUID::TimeIntegrationScheme>(timeint_stationary, timeint_npgenalpha,
          timeint_afgenalpha, timeint_one_step_theta, timeint_bdf2),
      &fdyn);

  setStringToIntegralParameter<Inpar::FLUID::OstContAndPress>("OST_CONT_PRESS",
      "Cont_normal_Press_normal",
      "One step theta option for time discretization of continuity eq. and pressure",
      tuple<std::string>(
          "Cont_normal_Press_normal", "Cont_impl_Press_normal", "Cont_impl_Press_impl"),
      tuple<Inpar::FLUID::OstContAndPress>(
          Cont_normal_Press_normal, Cont_impl_Press_normal, Cont_impl_Press_impl),
      &fdyn);

  setStringToIntegralParameter<Core::IO::GeometryType>("GEOMETRY", "full",
      "How the geometry is specified", tuple<std::string>("full", "box", "file"),
      tuple<Core::IO::GeometryType>(
          Core::IO::geometry_full, Core::IO::geometry_box, Core::IO::geometry_file),
      &fdyn);

  setStringToIntegralParameter<Inpar::FLUID::LinearisationAction>("NONLINITER", "fixed_point_like",
      "Nonlinear iteration scheme", tuple<std::string>("fixed_point_like", "Newton"),
      tuple<Inpar::FLUID::LinearisationAction>(fixed_point_like, Newton), &fdyn);

  std::vector<std::string> predictor_valid_input = {"steady_state", "zero_acceleration",
      "constant_acceleration", "constant_increment", "explicit_second_order_midpoint", "TangVel"};
  Core::Utils::string_parameter("PREDICTOR", "steady_state",
      "Predictor for first guess in nonlinear iteration", &fdyn, predictor_valid_input);


  setStringToIntegralParameter<Inpar::FLUID::ItNorm>("CONVCHECK", "L_2_norm",
      "norm for convergence check", tuple<std::string>("L_2_norm"),
      tuple<std::string>("compute L2 errors of increments (relative) and residuals (absolute)"),
      tuple<Inpar::FLUID::ItNorm>(fncc_L2), &fdyn);

  Core::Utils::bool_parameter("INCONSISTENT_RESIDUAL", "No",
      "do not evaluate residual after solution has converged (->faster)", &fdyn);

  {
    // a standard Teuchos::tuple can have at maximum 10 entries! We have to circumvent this here.
    Teuchos::Tuple<std::string, 11> name;
    Teuchos::Tuple<Inpar::FLUID::InitialField, 11> label;
    name[0] = "zero_field";
    label[0] = initfield_zero_field;
    name[1] = "field_by_function";
    label[1] = initfield_field_by_function;
    name[2] = "disturbed_field_from_function";
    label[2] = initfield_disturbed_field_from_function;
    name[3] = "FLAME_VORTEX_INTERACTION";
    label[3] = initfield_flame_vortex_interaction;
    name[4] = "BELTRAMI-FLOW";
    label[4] = initfield_beltrami_flow;
    name[5] = "KIM-MOIN-FLOW";
    label[5] = initfield_kim_moin_flow;
    name[6] = "hit_comte_bellot_corrsin_initial_field";
    label[6] = initfield_hit_comte_bellot_corrsin;
    name[7] = "forced_hit_simple_algebraic_spectrum";
    label[7] = initfield_forced_hit_simple_algebraic_spectrum;
    name[8] = "forced_hit_numeric_spectrum";
    label[8] = initfield_forced_hit_numeric_spectrum;
    name[9] = "forced_hit_passive";
    label[9] = initfield_passive_hit_const_input;
    name[10] = "channel_weakly_compressible";
    label[10] = initfield_channel_weakly_compressible;

    setStringToIntegralParameter<Inpar::FLUID::InitialField>(
        "INITIALFIELD", "zero_field", "Initial field for fluid problem", name, label, &fdyn);
  }

  Core::Utils::int_parameter(
      "OSEENFIELDFUNCNO", -1, "function number of Oseen advective field", &fdyn);

  Core::Utils::bool_parameter(
      "LIFTDRAG", "No", "Calculate lift and drag forces along specified boundary", &fdyn);

  std::vector<std::string> convform_valid_input = {"convective", "conservative"};
  Core::Utils::string_parameter(
      "CONVFORM", "convective", "form of convective term", &fdyn, convform_valid_input);

  std::vector<std::string> nonlinearbc_valid_input = {"no", "yes"};
  Core::Utils::string_parameter("NONLINEARBC", "no",
      "Flag to activate check for potential nonlinear boundary conditions", &fdyn,
      nonlinearbc_valid_input);

  setStringToIntegralParameter<Inpar::FLUID::MeshTying>("MESHTYING", "no",
      "Flag to (de)activate mesh tying algorithm",
      tuple<std::string>("no", "Condensed_Smat", "Condensed_Bmat", "Condensed_Bmat_merged"),
      tuple<Inpar::FLUID::MeshTying>(
          no_meshtying, condensed_smat, condensed_bmat, condensed_bmat_merged),
      &fdyn);

  setStringToIntegralParameter<Inpar::FLUID::Gridvel>("GRIDVEL", "BE",
      "scheme for determination of gridvelocity from displacements",
      tuple<std::string>("BE", "BDF2", "OST"), tuple<Inpar::FLUID::Gridvel>(BE, BDF2, OST), &fdyn);

  Core::Utils::bool_parameter(
      "ALLDOFCOUPLED", "Yes", "all dof (incl. pressure) are coupled", &fdyn);

  {
    Teuchos::Tuple<std::string, 16> name;
    Teuchos::Tuple<Inpar::FLUID::CalcError, 16> label;

    name[0] = "no";
    label[0] = no_error_calculation;
    name[1] = "beltrami_flow";
    label[1] = beltrami_flow;
    name[2] = "channel2D";
    label[2] = channel2D;
    name[3] = "gravitation";
    label[3] = gravitation;
    name[4] = "shear_flow";
    label[4] = shear_flow;
    name[5] = "byfunct";
    label[5] = byfunct;
    name[6] = "beltrami_stat_stokes";
    label[6] = beltrami_stat_stokes;
    name[7] = "beltrami_stat_navier_stokes";
    label[7] = beltrami_stat_navier_stokes;
    name[8] = "beltrami_instat_stokes";
    label[8] = beltrami_instat_stokes;
    name[9] = "beltrami_instat_navier_stokes";
    label[9] = beltrami_instat_navier_stokes;
    name[10] = "kimmoin_stat_stokes";
    label[10] = kimmoin_stat_stokes;
    name[11] = "kimmoin_stat_navier_stokes";
    label[11] = kimmoin_stat_navier_stokes;
    name[12] = "kimmoin_instat_stokes";
    label[12] = kimmoin_instat_stokes;
    name[13] = "kimmoin_instat_navier_stokes";
    label[13] = kimmoin_instat_navier_stokes;
    name[14] = "fsi_fluid_pusher";
    label[14] = fsi_fluid_pusher;
    name[15] = "channel_weakly_compressible";
    label[15] = channel_weakly_compressible;

    setStringToIntegralParameter<Inpar::FLUID::CalcError>(
        "CALCERROR", "no", "Flag to (de)activate error calculations", name, label, &fdyn);
  }
  Core::Utils::int_parameter("CALCERRORFUNCNO", -1, "Function for Error Calculation", &fdyn);

  Core::Utils::int_parameter("CORRTERMFUNCNO", -1,
      "Function for calculation of the correction term for the weakly compressible problem", &fdyn);

  Core::Utils::int_parameter("BODYFORCEFUNCNO", -1,
      "Function for calculation of the body force for the weakly compressible problem", &fdyn);

  Core::Utils::double_parameter("STAB_DEN_REF", 0.0,
      "Reference stabilization parameter for the density for the HDG weakly compressible "
      "formulation",
      &fdyn);

  Core::Utils::double_parameter("STAB_MOM_REF", 0.0,
      "Reference stabilization parameter for the momentum for the HDG weakly compressible "
      "formulation",
      &fdyn);

  Core::Utils::int_parameter("VARVISCFUNCNO", -1,
      "Function for calculation of a variable viscosity for the weakly compressible problem",
      &fdyn);

  {
    Teuchos::Tuple<std::string, 2> name;
    Teuchos::Tuple<Inpar::FLUID::PressAvgBc, 2> label;

    name[0] = "no";
    label[0] = no_pressure_average_bc;
    name[1] = "yes";
    label[1] = yes_pressure_average_bc;

    setStringToIntegralParameter<Inpar::FLUID::PressAvgBc>("PRESSAVGBC", "no",
        "Flag to (de)activate imposition of boundary condition for the considered element average "
        "pressure",
        name, label, &fdyn);
  }

  Core::Utils::double_parameter("REFMACH", 1.0, "Reference Mach number", &fdyn);

  Core::Utils::bool_parameter("BLOCKMATRIX", "No",
      "Indicates if system matrix should be assembled into a sparse block matrix type.", &fdyn);

  Core::Utils::bool_parameter("ADAPTCONV", "No",
      "Switch on adaptive control of linear solver tolerance for nonlinear solution", &fdyn);
  Core::Utils::double_parameter("ADAPTCONV_BETTER", 0.1,
      "The linear solver shall be this much better than the current nonlinear residual in the "
      "nonlinear convergence limit",
      &fdyn);

  Core::Utils::bool_parameter(
      "INFNORMSCALING", "no", "Scale blocks of matrix with row infnorm?", &fdyn);

  Core::Utils::bool_parameter("GMSH_OUTPUT", "No", "write output to gmsh files", &fdyn);
  Core::Utils::bool_parameter(
      "COMPUTE_DIVU", "No", "Compute divergence of velocity field at the element center", &fdyn);
  Core::Utils::bool_parameter("COMPUTE_EKIN", "No",
      "Compute kinetic energy at the end of each time step and write it to file.", &fdyn);
  Core::Utils::bool_parameter("NEW_OST", "No",
      "Solve the Navier-Stokes equation with the new One Step Theta algorithm",
      &fdyn);  // TODO: To be removed.
  Core::Utils::int_parameter("RESULTSEVERY", 1, "Increment for writing solution", &fdyn);
  Core::Utils::int_parameter("RESTARTEVERY", 20, "Increment for writing restart", &fdyn);
  Core::Utils::int_parameter("NUMSTEP", 1, "Total number of Timesteps", &fdyn);
  Core::Utils::int_parameter("STEADYSTEP", -1, "steady state check every step", &fdyn);
  Core::Utils::int_parameter("NUMSTASTEPS", 0, "Number of Steps for Starting Scheme", &fdyn);
  Core::Utils::int_parameter("STARTFUNCNO", -1, "Function for Initial Starting Field", &fdyn);
  Core::Utils::int_parameter("ITEMAX", 10, "max. number of nonlin. iterations", &fdyn);
  Core::Utils::int_parameter("INITSTATITEMAX", 5,
      "max number of nonlinear iterations for initial stationary solution", &fdyn);
  Core::Utils::double_parameter("TIMESTEP", 0.01, "Time increment dt", &fdyn);
  Core::Utils::double_parameter("MAXTIME", 1000.0, "Total simulation time", &fdyn);
  Core::Utils::double_parameter("ALPHA_M", 1.0, "Time integration factor", &fdyn);
  Core::Utils::double_parameter("ALPHA_F", 1.0, "Time integration factor", &fdyn);
  Core::Utils::double_parameter("GAMMA", 1.0, "Time integration factor", &fdyn);
  Core::Utils::double_parameter("THETA", 0.66, "Time integration factor", &fdyn);

  Core::Utils::double_parameter(
      "START_THETA", 1.0, "Time integration factor for starting scheme", &fdyn);

  std::vector<std::string> strong_redd_3d_coupling_valid_input = {"no", "yes"};
  Core::Utils::string_parameter("STRONG_REDD_3D_COUPLING_TYPE", "no",
      "Flag to (de)activate potential Strong 3D redD coupling", &fdyn,
      strong_redd_3d_coupling_valid_input);

  Core::Utils::int_parameter(
      "VELGRAD_PROJ_SOLVER", -1, "Number of linear solver used for L2 projection", &fdyn);

  setStringToIntegralParameter<Inpar::FLUID::GradientReconstructionMethod>("VELGRAD_PROJ_METHOD",
      "none", "Flag to (de)activate gradient reconstruction.",
      tuple<std::string>("none", "superconvergent_patch_recovery", "L2_projection"),
      tuple<std::string>("no gradient reconstruction",
          "gradient reconstruction via superconvergent patch recovery",
          "gracient reconstruction via l2-projection"),
      tuple<Inpar::FLUID::GradientReconstructionMethod>(
          gradreco_none,  // no convective streamline edge-based stabilization
          gradreco_spr,   // convective streamline edge-based stabilization on the entire domain
          gradreco_l2     // pressure edge-based stabilization as ghost penalty around cut elements
          ),
      &fdyn);

  Core::Utils::bool_parameter("OFF_PROC_ASSEMBLY", "No",
      "Do not evaluate ghosted elements but communicate them --> faster if element call is "
      "expensive",
      &fdyn);
  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& fdyn_nln = fdyn.sublist("NONLINEAR SOLVER TOLERANCES", false, "");

  Core::Utils::double_parameter(
      "TOL_VEL_RES", 1e-6, "Tolerance for convergence check of velocity residual", &fdyn_nln);

  Core::Utils::double_parameter(
      "TOL_VEL_INC", 1e-6, "Tolerance for convergence check of velocity increment", &fdyn_nln);

  Core::Utils::double_parameter(
      "TOL_PRES_RES", 1e-6, "Tolerance for convergence check of pressure residual", &fdyn_nln);

  Core::Utils::double_parameter(
      "TOL_PRES_INC", 1e-6, "Tolerance for convergence check of pressure increment", &fdyn_nln);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& fdyn_stab = fdyn.sublist("RESIDUAL-BASED STABILIZATION", false, "");
  // this parameter defines various stabilized methods
  setStringToIntegralParameter<Inpar::FLUID::StabType>("STABTYPE", "residual_based",
      "Apply (un)stabilized fluid formulation",
      tuple<std::string>("no_stabilization", "residual_based", "edge_based", "pressure_projection"),
      tuple<std::string>("Do not use any stabilization -> inf-sup stable elements required!",
          "Use a residual-based stabilization or, more generally, a stabilization \nbased on the "
          "concept of the residual-based variational multiscale method...\nExpecting additional "
          "input",
          "Use an edge-based stabilization, especially for XFEM",
          "Element/cell based polynomial pressure projection, see Dohrmann/Bochev 2004, IJNMF"),
      tuple<Inpar::FLUID::StabType>(
          stabtype_nostab, stabtype_residualbased, stabtype_edgebased, stabtype_pressureprojection),
      &fdyn_stab);

  Core::Utils::bool_parameter("INCONSISTENT", "No",
      "residual based without second derivatives (i.e. only consistent for tau->0, but faster)",
      &fdyn_stab);

  Core::Utils::bool_parameter("Reconstruct_Sec_Der", "No",
      "residual computed with a reconstruction of the second derivatives via projection or "
      "superconvergent patch recovery",
      &fdyn_stab);

  // the following parameters are necessary only if a residual based stabilized method is applied
  setStringToIntegralParameter<SubscalesTD>("TDS", "quasistatic",
      "Flag to allow time dependency of subscales for residual-based stabilization.",
      tuple<std::string>("quasistatic", "time_dependent"),
      tuple<std::string>("Use a quasi-static residual-based stabilization (standard case)",
          "Residual-based stabilization including time evolution equations for subscales"),
      tuple<SubscalesTD>(subscales_quasistatic, subscales_time_dependent), &fdyn_stab);

  setStringToIntegralParameter<Transient>("TRANSIENT", "no_transient",
      "Specify how to treat the transient term.",
      tuple<std::string>("no_transient", "yes_transient", "transient_complete"),
      tuple<std::string>(
          "Do not use transient term (currently only opportunity for quasistatic stabilization)",
          "Use transient term (recommended for time dependent subscales)",
          "Use transient term including a linearisation of 1/tau"),
      tuple<Transient>(inertia_stab_drop, inertia_stab_keep, inertia_stab_keep_complete),
      &fdyn_stab);

  Core::Utils::bool_parameter(
      "PSPG", "Yes", "Flag to (de)activate PSPG stabilization.", &fdyn_stab);
  Core::Utils::bool_parameter(
      "SUPG", "Yes", "Flag to (de)activate SUPG stabilization.", &fdyn_stab);
  Core::Utils::bool_parameter("GRAD_DIV", "Yes", "Flag to (de)activate grad-div term.", &fdyn_stab);

  setStringToIntegralParameter<VStab>("VSTAB", "no_vstab",
      "Flag to (de)activate viscous term in residual-based stabilization.",
      tuple<std::string>(
          "no_vstab", "vstab_gls", "vstab_gls_rhs", "vstab_usfem", "vstab_usfem_rhs"),
      tuple<std::string>("No viscous term in stabilization", "Viscous stabilization of GLS type",
          "Viscous stabilization of GLS type, included only on the right hand side",
          "Viscous stabilization of USFEM type",
          "Viscous stabilization of USFEM type, included only on the right hand side"),
      tuple<VStab>(viscous_stab_none, viscous_stab_gls, viscous_stab_gls_only_rhs,
          viscous_stab_usfem, viscous_stab_usfem_only_rhs),
      &fdyn_stab);

  setStringToIntegralParameter<RStab>("RSTAB", "no_rstab",
      "Flag to (de)activate reactive term in residual-based stabilization.",
      tuple<std::string>("no_rstab", "rstab_gls", "rstab_usfem"),
      tuple<std::string>("no reactive term in stabilization", "reactive stabilization of GLS type",
          "reactive stabilization of USFEM type"),
      tuple<RStab>(reactive_stab_none, reactive_stab_gls, reactive_stab_usfem), &fdyn_stab);

  setStringToIntegralParameter<CrossStress>("CROSS-STRESS", "no_cross",
      "Flag to (de)activate cross-stress term -> residual-based VMM.",
      tuple<std::string>("no_cross", "yes_cross", "cross_rhs"
          //"cross_complete"
          ),
      tuple<std::string>("No cross-stress term",
          "Include the cross-stress term with a linearization of the convective part",
          "Include cross-stress term, but only explicitly on right hand side"
          //""
          ),
      tuple<CrossStress>(cross_stress_stab_none, cross_stress_stab, cross_stress_stab_only_rhs),
      &fdyn_stab);

  setStringToIntegralParameter<ReynoldsStress>("REYNOLDS-STRESS", "no_reynolds",
      "Flag to (de)activate Reynolds-stress term -> residual-based VMM.",
      tuple<std::string>("no_reynolds", "yes_reynolds", "reynolds_rhs"
          //"reynolds_complete"
          ),
      tuple<std::string>(
          "No Reynolds-stress term", "Include Reynolds-stress term with linearisation",
          "Include Reynolds-stress term explicitly on right hand side"
          //""
          ),
      tuple<ReynoldsStress>(
          reynolds_stress_stab_none, reynolds_stress_stab, reynolds_stress_stab_only_rhs),
      &fdyn_stab);

  {
    // this parameter selects the tau definition applied
    // a standard Teuchos::tuple can have at maximum 10 entries! We have to circumvent this here.
    Teuchos::Tuple<std::string, 16> name;
    Teuchos::Tuple<TauType, 16> label;
    name[0] = "Taylor_Hughes_Zarins";
    label[0] = tau_taylor_hughes_zarins;
    name[1] = "Taylor_Hughes_Zarins_wo_dt";
    label[1] = tau_taylor_hughes_zarins_wo_dt;
    name[2] = "Taylor_Hughes_Zarins_Whiting_Jansen";
    label[2] = tau_taylor_hughes_zarins_whiting_jansen;
    name[3] = "Taylor_Hughes_Zarins_Whiting_Jansen_wo_dt";
    label[3] = tau_taylor_hughes_zarins_whiting_jansen_wo_dt;
    name[4] = "Taylor_Hughes_Zarins_scaled";
    label[4] = tau_taylor_hughes_zarins_scaled;
    name[5] = "Taylor_Hughes_Zarins_scaled_wo_dt";
    label[5] = tau_taylor_hughes_zarins_scaled_wo_dt;
    name[6] = "Franca_Barrenechea_Valentin_Frey_Wall";
    label[6] = tau_franca_barrenechea_valentin_frey_wall;
    name[7] = "Franca_Barrenechea_Valentin_Frey_Wall_wo_dt";
    label[7] = tau_franca_barrenechea_valentin_frey_wall_wo_dt;
    name[8] = "Shakib_Hughes_Codina";
    label[8] = tau_shakib_hughes_codina;
    name[9] = "Shakib_Hughes_Codina_wo_dt";
    label[9] = tau_shakib_hughes_codina_wo_dt;
    name[10] = "Codina";
    label[10] = tau_codina;
    name[11] = "Codina_wo_dt";
    label[11] = tau_codina_wo_dt;
    name[12] = "Codina_convscaled";
    label[12] = tau_codina_convscaled;
    name[13] = "Franca_Madureira_Valentin_Badia_Codina";
    label[13] = tau_franca_madureira_valentin_badia_codina;
    name[14] = "Franca_Madureira_Valentin_Badia_Codina_wo_dt";
    label[14] = tau_franca_madureira_valentin_badia_codina_wo_dt;
    name[15] = "Hughes_Franca_Balestra_wo_dt";
    label[15] = tau_hughes_franca_balestra_wo_dt;

    setStringToIntegralParameter<TauType>("DEFINITION_TAU", "Franca_Barrenechea_Valentin_Frey_Wall",
        "Definition of tau_M and Tau_C", name, label, &fdyn_stab);
  }

  // this parameter selects the characteristic element length for tau_Mu for all
  // stabilization parameter definitions requiring such a length
  setStringToIntegralParameter<CharEleLengthU>("CHARELELENGTH_U", "streamlength",
      "Characteristic element length for tau_Mu",
      tuple<std::string>("streamlength", "volume_equivalent_diameter", "root_of_volume"),
      tuple<CharEleLengthU>(streamlength_u, volume_equivalent_diameter_u, root_of_volume_u),
      &fdyn_stab);

  // this parameter selects the characteristic element length for tau_Mp and tau_C for
  // all stabilization parameter definitions requiring such a length
  setStringToIntegralParameter<CharEleLengthPC>("CHARELELENGTH_PC", "volume_equivalent_diameter",
      "Characteristic element length for tau_Mp/tau_C",
      tuple<std::string>("streamlength", "volume_equivalent_diameter", "root_of_volume"),
      tuple<CharEleLengthPC>(streamlength_pc, volume_equivalent_diameter_pc, root_of_volume_pc),
      &fdyn_stab);

  // this parameter selects the location where tau is evaluated

  std::vector<std::string> evaluation_tau_valid_input = {"element_center", "integration_point"};
  Core::Utils::string_parameter("EVALUATION_TAU", "element_center",
      "Location where tau is evaluated", &fdyn_stab, evaluation_tau_valid_input);

  // this parameter selects the location where the material law is evaluated
  // (does not fit here very well, but parameter transfer is easier)

  std::vector<std::string> evaluation_mat_valid_input = {"element_center", "integration_point"};
  Core::Utils::string_parameter("EVALUATION_MAT", "element_center",
      "Location where material law is evaluated", &fdyn_stab, evaluation_mat_valid_input);

  // these parameters active additional terms in loma continuity equation
  // which might be identified as SUPG-/cross- and Reynolds-stress term
  Core::Utils::bool_parameter("LOMA_CONTI_SUPG", "No",
      "Flag to (de)activate SUPG stabilization in loma continuity equation.", &fdyn_stab);

  setStringToIntegralParameter<CrossStress>("LOMA_CONTI_CROSS_STRESS", "no_cross",
      "Flag to (de)activate cross-stress term loma continuity equation-> residual-based VMM.",
      tuple<std::string>("no_cross", "yes_cross", "cross_rhs"
          //"cross_complete"
          ),
      tuple<std::string>("No cross-stress term",
          "Include the cross-stress term with a linearization of the convective part",
          "Include cross-stress term, but only explicitly on right hand side"
          //""
          ),
      tuple<CrossStress>(cross_stress_stab_none, cross_stress_stab, cross_stress_stab_only_rhs),
      &fdyn_stab);

  setStringToIntegralParameter<ReynoldsStress>("LOMA_CONTI_REYNOLDS_STRESS", "no_reynolds",
      "Flag to (de)activate Reynolds-stress term loma continuity equation-> residual-based VMM.",
      tuple<std::string>("no_reynolds", "yes_reynolds", "reynolds_rhs"),
      tuple<std::string>(
          "No Reynolds-stress term", "Include Reynolds-stress term with linearisation",
          "Include Reynolds-stress term explicitly on right hand side"
          //""
          ),
      tuple<ReynoldsStress>(
          reynolds_stress_stab_none, reynolds_stress_stab, reynolds_stress_stab_only_rhs),
      &fdyn_stab);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& fdyn_edge_based_stab =
      fdyn.sublist("EDGE-BASED STABILIZATION", false, "");

  //! Flag to (de)activate edge-based (EOS) pressure stabilization
  setStringToIntegralParameter<EosPres>("EOS_PRES", "none",
      "Flag to (de)activate pressure edge-based stabilization.",
      tuple<std::string>("none", "std_eos", "xfem_gp"),
      tuple<std::string>("do not use pressure edge-based stabilization",
          "use pressure edge-based stabilization as standard edge-based stabilization on the "
          "entire domain",
          "use pressure edge-based stabilization as xfem ghost-penalty stabilization just around "
          "cut elements"),
      tuple<EosPres>(EOS_PRES_none,  // no pressure edge-based stabilization
          EOS_PRES_std_eos,          // pressure edge-based stabilization on the entire domain
          EOS_PRES_xfem_gp  // pressure edge-based stabilization as ghost penalty around cut
                            // elements
          ),
      &fdyn_edge_based_stab);

  //! Flag to (de)activate edge-based (EOS) convective streamline stabilization
  setStringToIntegralParameter<EosConvStream>("EOS_CONV_STREAM", "none",
      "Flag to (de)activate convective streamline edge-based stabilization.",
      tuple<std::string>("none", "std_eos", "xfem_gp"),
      tuple<std::string>("do not use convective streamline edge-based stabilization",
          "use convective streamline edge-based stabilization as standard edge-based stabilization "
          "on the entire domain",
          "use convective streamline edge-based stabilization as xfem ghost-penalty stabilization "
          "just around cut elements"),
      tuple<EosConvStream>(
          EOS_CONV_STREAM_none,     // no convective streamline edge-based stabilization
          EOS_CONV_STREAM_std_eos,  // convective streamline edge-based stabilization on the entire
                                    // domain
          EOS_CONV_STREAM_xfem_gp   // pressure edge-based stabilization as ghost penalty around cut
                                    // elements
          ),
      &fdyn_edge_based_stab);

  //! Flag to (de)activate edge-based (EOS) convective crosswind stabilization
  setStringToIntegralParameter<EosConvCross>("EOS_CONV_CROSS", "none",
      "Flag to (de)activate convective crosswind edge-based stabilization.",
      tuple<std::string>("none", "std_eos", "xfem_gp"),
      tuple<std::string>("do not use convective crosswind edge-based stabilization",
          "use convective crosswind edge-based stabilization as standard edge-based stabilization "
          "on the entire domain",
          "use convective crosswind edge-based stabilization as xfem ghost-penalty stabilization "
          "just around cut elements"),
      tuple<EosConvCross>(EOS_CONV_CROSS_none,  // no convective crosswind edge-based stabilization
          EOS_CONV_CROSS_std_eos,  // convective crosswind edge-based stabilization on the entire
                                   // domain
          EOS_CONV_CROSS_xfem_gp   // convective crosswind edge-based stabilization as ghost penalty
                                   // around cut elements
          ),
      &fdyn_edge_based_stab);

  //! Flag to (de)activate edge-based (EOS) divergence stabilization
  setStringToIntegralParameter<EosDiv>("EOS_DIV", "none",
      "Flag to (de)activate divergence edge-based stabilization.",
      tuple<std::string>(
          "none", "vel_jump_std_eos", "vel_jump_xfem_gp", "div_jump_std_eos", "div_jump_xfem_gp"),
      tuple<std::string>("do not use divergence edge-based stabilization",
          "divergence edge-based stabilization based on velocity jump on the entire domain",
          "divergence edge-based stabilization based on divergence jump just around cut elements",
          "divergence edge-based stabilization based on velocity jump on the entire domain",
          "divergence edge-based stabilization based on divergence jump just around cut elements"),
      tuple<EosDiv>(EOS_DIV_none,    // no convective edge-based stabilization
          EOS_DIV_vel_jump_std_eos,  // streamline convective edge-based stabilization
          EOS_DIV_vel_jump_xfem_gp,  // streamline convective edge-based stabilization
          EOS_DIV_div_jump_std_eos,  // crosswind convective edge-based stabilization
          EOS_DIV_div_jump_xfem_gp   // crosswind convective edge-based stabilization
          ),
      &fdyn_edge_based_stab);

  //! special least-squares condition for pseudo 2D examples where pressure level is determined via
  //! Krylov-projection
  Core::Utils::bool_parameter("PRES_KRYLOV_2Dz", "No",
      "residual based without second derivatives (i.e. only consistent for tau->0, but faster)",
      &fdyn_edge_based_stab);

  //! this parameter selects the definition of Edge-based stabilization parameter
  setStringToIntegralParameter<EosTauType>("EOS_DEFINITION_TAU", "Burman_Hansbo_DAngelo_Zunino",
      "Definition of stabilization parameter for edge-based stabilization",
      tuple<std::string>("Burman_Fernandez_Hansbo", "Burman_Fernandez_Hansbo_wo_dt",
          "Braack_Burman_John_Lube", "Braack_Burman_John_Lube_wo_divjump",
          "Franca_Barrenechea_Valentin_Wall", "Burman_Fernandez", "Burman_Hansbo_DAngelo_Zunino",
          "Burman_Hansbo_DAngelo_Zunino_wo_dt", "Schott_Massing_Burman_DAngelo_Zunino",
          "Schott_Massing_Burman_DAngelo_Zunino_wo_dt", "Burman",
          "Taylor_Hughes_Zarins_Whiting_Jansen_Codina_scaling", "tau_not_defined"),
      tuple<std::string>("definition of burman_fernandez_hansbo",
          "definition of burman_fernandez_hansbo for stationary problems",
          "definition of braack_burman_john_lube",
          "definition of braack_burman_john_lube without explicit inclusion of divergence jump",
          "definition of tau_franca_barrenechea_valentin_wall",
          "definition of EOS_tau_burman_fernandez",
          "definition of EOS_tau_burman_hansbo_dangelo_zunino",
          "definition of EOS_tau_burman_hansbo_dangelo_zunino for stationary problems",
          "definition of EOS_tau_schott_massing_burman_dangelo_zunino",
          "definition of EOS_tau_schott_massing_burman_dangelo_zunino for stationary problems",
          "definition of EOS_tau_burman",
          "definition of EOS_tau related to residual-based stabilization", "no chosen definition"),
      tuple<EosTauType>(Inpar::FLUID::EOS_tau_burman_fernandez_hansbo,
          Inpar::FLUID::EOS_tau_burman_fernandez_hansbo_wo_dt,
          Inpar::FLUID::EOS_tau_braack_burman_john_lube,
          Inpar::FLUID::EOS_tau_braack_burman_john_lube_wo_divjump,
          Inpar::FLUID::EOS_tau_franca_barrenechea_valentin_wall,
          Inpar::FLUID::EOS_tau_burman_fernandez,
          Inpar::FLUID::EOS_tau_burman_hansbo_dangelo_zunino,
          Inpar::FLUID::EOS_tau_burman_hansbo_dangelo_zunino_wo_dt,
          Inpar::FLUID::EOS_tau_schott_massing_burman_dangelo_zunino,
          Inpar::FLUID::EOS_tau_schott_massing_burman_dangelo_zunino_wo_dt,
          Inpar::FLUID::EOS_tau_burman,
          Inpar::FLUID::EOS_tau_Taylor_Hughes_Zarins_Whiting_Jansen_Codina_scaling,
          Inpar::FLUID::EOS_tau_not_defined),
      &fdyn_edge_based_stab);

  //! this parameter selects how the element length of Edge-based stabilization is defined
  setStringToIntegralParameter<EosElementLength>("EOS_H_DEFINITION",
      "EOS_he_max_diameter_to_opp_surf",
      "Definition of element length for edge-based stabilization",
      tuple<std::string>("EOS_he_max_diameter_to_opp_surf", "EOS_he_max_dist_to_opp_surf",
          "EOS_he_surf_with_max_diameter", "EOS_hk_max_diameter", "EOS_he_surf_diameter",
          "EOS_he_vol_eq_diameter"),
      tuple<std::string>("take the maximal (nsd-1)D diameter of faces that connect the internal "
                         "face to its opposite faces",
          "take the maximal 1D distance along 1D edge to opposite surface for both parent elements",
          "take the maximal (nsd-1)D face diameter of all faces for both parent elements",
          "maximal nD diameter of the neighboring elements",
          "maximal (n-1)D diameter of the internal face/edge",
          "take the maximal volume equivalent diameter of adjacent elements"),
      tuple<EosElementLength>(EOS_he_max_diameter_to_opp_surf, EOS_he_max_dist_to_opp_surf,
          EOS_he_surf_with_max_diameter, EOS_hk_max_diameter, EOS_he_surf_diameter,
          EOS_he_vol_eq_diameter),
      &fdyn_edge_based_stab);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& fdyn_porostab = fdyn.sublist("POROUS-FLOW STABILIZATION", false, "");

  Core::Utils::bool_parameter(
      "STAB_BIOT", "No", "Flag to (de)activate BIOT stabilization.", &fdyn_porostab);
  Core::Utils::double_parameter("STAB_BIOT_SCALING", 1.0,
      "Scaling factor for stabilization parameter for biot stabilization of porous flow.",
      &fdyn_porostab);

  // this parameter defines various stabilized methods
  setStringToIntegralParameter<Inpar::FLUID::StabType>("STABTYPE", "residual_based",
      "Apply (un)stabilized fluid formulation",
      tuple<std::string>("no_stabilization", "residual_based", "edge_based"),
      tuple<std::string>("Do not use any stabilization -> inf-sup stable elements required!",
          "Use a residual-based stabilization or, more generally, a stabilization \nbased on the "
          "concept of the residual-based variational multiscale method...\nExpecting additional "
          "input",
          "Use an edge-based stabilization, especially for XFEM"),
      tuple<Inpar::FLUID::StabType>(stabtype_nostab, stabtype_residualbased, stabtype_edgebased),
      &fdyn_porostab);

  Core::Utils::bool_parameter("INCONSISTENT", "No",
      "residual based without second derivatives (i.e. only consistent for tau->0, but faster)",
      &fdyn_porostab);

  Core::Utils::bool_parameter("Reconstruct_Sec_Der", "No",
      "residual computed with a reconstruction of the second derivatives via projection or "
      "superconvergent patch recovery",
      &fdyn_porostab);

  // the following parameters are necessary only if a residual based stabilized method is applied
  setStringToIntegralParameter<SubscalesTD>("TDS", "quasistatic",
      "Flag to allow time dependency of subscales for residual-based stabilization.",
      tuple<std::string>("quasistatic", "time_dependent"),
      tuple<std::string>("Use a quasi-static residual-based stabilization (standard case)",
          "Residual-based stabilization including time evolution equations for subscales"),
      tuple<SubscalesTD>(subscales_quasistatic, subscales_time_dependent), &fdyn_porostab);

  setStringToIntegralParameter<Transient>("TRANSIENT", "no_transient",
      "Specify how to treat the transient term.",
      tuple<std::string>("no_transient", "yes_transient", "transient_complete"),
      tuple<std::string>(
          "Do not use transient term (currently only opportunity for quasistatic stabilization)",
          "Use transient term (recommended for time dependent subscales)",
          "Use transient term including a linearisation of 1/tau"),
      tuple<Transient>(inertia_stab_drop, inertia_stab_keep, inertia_stab_keep_complete),
      &fdyn_porostab);

  Core::Utils::bool_parameter(
      "PSPG", "Yes", "Flag to (de)activate PSPG stabilization.", &fdyn_porostab);
  Core::Utils::bool_parameter(
      "SUPG", "Yes", "Flag to (de)activate SUPG stabilization.", &fdyn_porostab);
  Core::Utils::bool_parameter(
      "GRAD_DIV", "Yes", "Flag to (de)activate grad-div term.", &fdyn_porostab);

  setStringToIntegralParameter<VStab>("VSTAB", "no_vstab",
      "Flag to (de)activate viscous term in residual-based stabilization.",
      tuple<std::string>(
          "no_vstab", "vstab_gls", "vstab_gls_rhs", "vstab_usfem", "vstab_usfem_rhs"),
      tuple<std::string>("No viscous term in stabilization", "Viscous stabilization of GLS type",
          "Viscous stabilization of GLS type, included only on the right hand side",
          "Viscous stabilization of USFEM type",
          "Viscous stabilization of USFEM type, included only on the right hand side"),
      tuple<VStab>(viscous_stab_none, viscous_stab_gls, viscous_stab_gls_only_rhs,
          viscous_stab_usfem, viscous_stab_usfem_only_rhs),
      &fdyn_porostab);

  setStringToIntegralParameter<RStab>("RSTAB", "no_rstab",
      "Flag to (de)activate reactive term in residual-based stabilization.",
      tuple<std::string>("no_rstab", "rstab_gls", "rstab_usfem"),
      tuple<std::string>("no reactive term in stabilization", "reactive stabilization of GLS type",
          "reactive stabilization of USFEM type"),
      tuple<RStab>(reactive_stab_none, reactive_stab_gls, reactive_stab_usfem), &fdyn_porostab);

  setStringToIntegralParameter<CrossStress>("CROSS-STRESS", "no_cross",
      "Flag to (de)activate cross-stress term -> residual-based VMM.",
      tuple<std::string>("no_cross", "yes_cross", "cross_rhs"
          //"cross_complete"
          ),
      tuple<std::string>("No cross-stress term",
          "Include the cross-stress term with a linearization of the convective part",
          "Include cross-stress term, but only explicitly on right hand side"
          //""
          ),
      tuple<CrossStress>(cross_stress_stab_none, cross_stress_stab, cross_stress_stab_only_rhs),
      &fdyn_porostab);

  setStringToIntegralParameter<ReynoldsStress>("REYNOLDS-STRESS", "no_reynolds",
      "Flag to (de)activate Reynolds-stress term -> residual-based VMM.",
      tuple<std::string>("no_reynolds", "yes_reynolds", "reynolds_rhs"
          //"reynolds_complete"
          ),
      tuple<std::string>(
          "No Reynolds-stress term", "Include Reynolds-stress term with linearisation",
          "Include Reynolds-stress term explicitly on right hand side"
          //""
          ),
      tuple<ReynoldsStress>(
          reynolds_stress_stab_none, reynolds_stress_stab, reynolds_stress_stab_only_rhs),
      &fdyn_porostab);

  // this parameter selects the tau definition applied
  setStringToIntegralParameter<TauType>("DEFINITION_TAU", "Franca_Barrenechea_Valentin_Frey_Wall",
      "Definition of tau_M and Tau_C",
      tuple<std::string>("Taylor_Hughes_Zarins", "Taylor_Hughes_Zarins_wo_dt",
          "Taylor_Hughes_Zarins_Whiting_Jansen", "Taylor_Hughes_Zarins_Whiting_Jansen_wo_dt",
          "Taylor_Hughes_Zarins_scaled", "Taylor_Hughes_Zarins_scaled_wo_dt",
          "Franca_Barrenechea_Valentin_Frey_Wall", "Franca_Barrenechea_Valentin_Frey_Wall_wo_dt",
          "Shakib_Hughes_Codina", "Shakib_Hughes_Codina_wo_dt", "Codina", "Codina_wo_dt",
          "Franca_Madureira_Valentin_Badia_Codina", "Franca_Madureira_Valentin_Badia_Codina_wo_dt"),
      tuple<TauType>(tau_taylor_hughes_zarins, tau_taylor_hughes_zarins_wo_dt,
          tau_taylor_hughes_zarins_whiting_jansen, tau_taylor_hughes_zarins_whiting_jansen_wo_dt,
          tau_taylor_hughes_zarins_scaled, tau_taylor_hughes_zarins_scaled_wo_dt,
          tau_franca_barrenechea_valentin_frey_wall,
          tau_franca_barrenechea_valentin_frey_wall_wo_dt, tau_shakib_hughes_codina,
          tau_shakib_hughes_codina_wo_dt, tau_codina, tau_codina_wo_dt,
          tau_franca_madureira_valentin_badia_codina,
          tau_franca_madureira_valentin_badia_codina_wo_dt),
      &fdyn_porostab);

  // this parameter selects the characteristic element length for tau_Mu for all
  // stabilization parameter definitions requiring such a length
  setStringToIntegralParameter<CharEleLengthU>("CHARELELENGTH_U", "streamlength",
      "Characteristic element length for tau_Mu",
      tuple<std::string>("streamlength", "volume_equivalent_diameter", "root_of_volume"),
      tuple<CharEleLengthU>(streamlength_u, volume_equivalent_diameter_u, root_of_volume_u),
      &fdyn_porostab);

  // this parameter selects the characteristic element length for tau_Mp and tau_C for
  // all stabilization parameter definitions requiring such a length
  setStringToIntegralParameter<CharEleLengthPC>("CHARELELENGTH_PC", "volume_equivalent_diameter",
      "Characteristic element length for tau_Mp/tau_C",
      tuple<std::string>("streamlength", "volume_equivalent_diameter", "root_of_volume"),
      tuple<CharEleLengthPC>(streamlength_pc, volume_equivalent_diameter_pc, root_of_volume_pc),
      &fdyn_porostab);

  // this parameter selects the location where tau is evaluated
  evaluation_tau_valid_input = {"element_center", "integration_point"};
  Core::Utils::string_parameter("EVALUATION_TAU", "element_center",
      "Location where tau is evaluated", &fdyn_porostab, evaluation_tau_valid_input);

  // this parameter selects the location where the material law is evaluated
  // (does not fit here very well, but parameter transfer is easier)
  evaluation_mat_valid_input = {"element_center", "integration_point"};
  Core::Utils::string_parameter("EVALUATION_MAT", "element_center",
      "Location where material law is evaluated", &fdyn_porostab, evaluation_mat_valid_input);


  // these parameters active additional terms in loma continuity equation
  // which might be identified as SUPG-/cross- and Reynolds-stress term
  Core::Utils::bool_parameter("LOMA_CONTI_SUPG", "No",
      "Flag to (de)activate SUPG stabilization in loma continuity equation.", &fdyn_porostab);

  setStringToIntegralParameter<CrossStress>("LOMA_CONTI_CROSS_STRESS", "no_cross",
      "Flag to (de)activate cross-stress term loma continuity equation-> residual-based VMM.",
      tuple<std::string>("no_cross", "yes_cross", "cross_rhs"
          //"cross_complete"
          ),
      tuple<std::string>("No cross-stress term",
          "Include the cross-stress term with a linearization of the convective part",
          "Include cross-stress term, but only explicitly on right hand side"
          //""
          ),
      tuple<CrossStress>(cross_stress_stab_none, cross_stress_stab, cross_stress_stab_only_rhs),
      &fdyn_porostab);

  setStringToIntegralParameter<ReynoldsStress>("LOMA_CONTI_REYNOLDS_STRESS", "no_reynolds",
      "Flag to (de)activate Reynolds-stress term loma continuity equation-> residual-based VMM.",
      tuple<std::string>("no_reynolds", "yes_reynolds", "reynolds_rhs"),
      tuple<std::string>(
          "No Reynolds-stress term", "Include Reynolds-stress term with linearisation",
          "Include Reynolds-stress term explicitly on right hand side"
          //""
          ),
      tuple<ReynoldsStress>(
          reynolds_stress_stab_none, reynolds_stress_stab, reynolds_stress_stab_only_rhs),
      &fdyn_porostab);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& fdyn_turbu = fdyn.sublist("TURBULENCE MODEL", false, "");

  //----------------------------------------------------------------------
  // modeling strategies
  //----------------------------------------------------------------------

  std::vector<std::string> turbulence_approach_valid_input = {"DNS_OR_RESVMM_LES", "CLASSICAL_LES"};
  std::string turbulence_approach_doc_string =
      "Try to solve flow as an underresolved DNS. Mind that your stabilisation already acts as a "
      "kind of turbulence model! Perform a classical Large Eddy Simulation adding addititional "
      "turbulent viscosity. This may be based on various physical models.)";

  Core::Utils::string_parameter("TURBULENCE_APPROACH", "DNS_OR_RESVMM_LES",
      turbulence_approach_doc_string, &fdyn_turbu, turbulence_approach_valid_input);

  std::vector<std::string> physical_model_valid_input = {"no_model", "Smagorinsky",
      "Smagorinsky_with_van_Driest_damping", "Dynamic_Smagorinsky", "Multifractal_Subgrid_Scales",
      "Vreman", "Dynamic_Vreman"};
  std::string physical_model_doc_string =
      "If classical LES is our turbulence approach, this is a contradiction and should cause a "
      "FOUR_C_THROW. Classical constant coefficient Smagorinsky model. Be careful if you have a "
      "wall bounded flow domain! Use an exponential damping function for the turbulent viscosity "
      "close to the wall. This is only implemented for a channel geometry of height 2 in y "
      "direction. The viscous lengthscale l_tau is required as additional input. The solution is "
      "filtered and by comparison of the filtered velocity field with the real solution, the "
      "Smagorinsky constant is estimated in each step --- mind that this procedure includes an "
      "averaging in the xz plane, hence this implementation will only work for a channel flow. "
      "Multifractal Subgrid-Scale Modeling based on the work of burton. Vremans constant model. "
      "Dynamic Vreman model according to You and Moin (2007)";
  Core::Utils::string_parameter("PHYSICAL_MODEL", "no_model", physical_model_doc_string,
      &fdyn_turbu, physical_model_valid_input);

  setStringToIntegralParameter<Inpar::FLUID::FineSubgridVisc>("FSSUGRVISC", "No",
      "fine-scale subgrid viscosity",
      tuple<std::string>("No", "Smagorinsky_all", "Smagorinsky_small"),
      tuple<Inpar::FLUID::FineSubgridVisc>(no_fssgv, smagorinsky_all, smagorinsky_small),
      &fdyn_turbu);

  //----------------------------------------------------------------------
  // turbulence specific output and statistics
  //----------------------------------------------------------------------

  Core::Utils::int_parameter(
      "SAMPLING_START", 10000000, "Time step after when sampling shall be started", &fdyn_turbu);
  Core::Utils::int_parameter(
      "SAMPLING_STOP", 1, "Time step when sampling shall be stopped", &fdyn_turbu);
  Core::Utils::int_parameter("DUMPING_PERIOD", 1,
      "Period of time steps after which statistical data shall be dumped", &fdyn_turbu);
  Core::Utils::bool_parameter("SUBGRID_DISSIPATION", "No",
      "Flag to (de)activate estimation of subgrid-scale dissipation (only for seclected flows).",
      &fdyn_turbu);
  Core::Utils::bool_parameter(
      "OUTMEAN", "No", "Flag to (de)activate averaged paraview output", &fdyn_turbu);
  Core::Utils::bool_parameter("TURBMODEL_LS", "Yes",
      "Flag to (de)activate turbulence model in level-set equation", &fdyn_turbu);

  //----------------------------------------------------------------------
  // turbulent flow problem and general characteristics
  //----------------------------------------------------------------------

  {
    std::vector<std::string> canonical_flow_valid_input = {"no", "time_averaging",
        "channel_flow_of_height_2", "lid_driven_cavity", "backward_facing_step", "square_cylinder",
        "square_cylinder_nurbs", "rotating_circular_cylinder_nurbs",
        "rotating_circular_cylinder_nurbs_scatra", "loma_channel_flow_of_height_2",
        "loma_lid_driven_cavity", "loma_backward_facing_step", "combust_oracles",
        "bubbly_channel_flow", "scatra_channel_flow_of_height_2",
        "decaying_homogeneous_isotropic_turbulence", "forced_homogeneous_isotropic_turbulence",
        "scatra_forced_homogeneous_isotropic_turbulence", "taylor_green_vortex", "periodic_hill",
        "blood_fda_flow", "backward_facing_step2"};

    std::string canonical_flow_doc =
        ""
        "Sampling is different for different canonical flows - so specify what kind of flow you've "
        "got \n\n"
        "no: The flow is not further specified, so spatial averaging and hence the standard "
        "sampling procedure is not possible\n"
        "time_averaging: The flow is not further specified, but time averaging of velocity and "
        "pressure field is performed\n"
        "channel_flow_of_height_2: For this flow, all statistical data could be averaged in the "
        "homogeneous planes - it is essentially a statistically one dimensional flow.\n"
        "lid_driven_cavity: For this flow, all statistical data are evaluated on the center lines "
        "of the xy-midplane, averaged only over time.\n"
        "backward_facing_step: For this flow, statistical data are evaluated on various lines, "
        "averaged over time and z.\n"
        "square_cylinder: For this flow, statistical data are evaluated on various lines of the "
        "xy-midplane, averaged only over time.\n"
        "square_cylinder_nurbs: For this flow, statistical data are evaluated on various lines of "
        "the xy-midplane, averaged over time and eventually in one home.direction.\n"
        "rotating_circular_cylinder_nurbs: For this flow, statistical data is computed in "
        "concentric surfaces and averaged. in time and in one home. direction\n"
        "rotating_circular_cylinder_nurbs_scatra: For this flow with mass transport, statistical "
        "data is computed in concentric surfaces and averaged. in time and in one home. direction\n"
        "loma_channel_flow_of_height_2: For this low-Mach-number flow, all statistical data could "
        "be averaged in the homogeneous planes - it is essentially a statistically one dimensional "
        "flow.\n"
        "loma_lid_driven_cavity: For this low-Mach-number flow, all statistical data are evaluated "
        "on the center lines of the xy-midplane, averaged only over time.\n"
        "loma_backward_facing_step: For this low-Mach-number flow, statistical data are evaluated "
        "on various lines, averaged over time and z.\n"
        "combust_oracles: ORACLES test rig for turbulent premixed combustion.\n"
        "bubbly_channel_flow: Turbulent two-phase flow: bubbly channel flow, statistical data are "
        "averaged in homogeneous planes and over time.\n"
        "scatra_channel_flow_of_height_2: For this flow, all statistical data could be averaged in "
        "the homogeneous planes - it is essentially a statistically one dimensional flow.\n"
        "decaying_homogeneous_isotropic_turbulence: For this flow, all statistical data could be "
        "averaged in the in all homogeneous directions  - it is essentially a statistically zero "
        "dimensional flow.\n"
        "forced_homogeneous_isotropic_turbulence: For this flow, all statistical data could be "
        "averaged in the in all homogeneous directions  - it is essentially a statistically zero "
        "dimensional flow.\n"
        "scatra_forced_homogeneous_isotropic_turbulence: For this flow, all statistical data could "
        "be averaged in the in all homogeneous directions  - it is essentially a statistically "
        "zero dimensional flow.\n"
        "taylor_green_vortex: For this flow, dissipation rate could be averaged in the in all "
        "homogeneous directions  - it is essentially a statistically zero dimensional flow.\n"
        "periodic_hill: For this flow, statistical data is evaluated on various lines, averaged "
        "over time and z.\n"
        "blood_fda_flow: For this flow, statistical data is evaluated on various planes.\n"
        "backward_facing_step2: For this flow, statistical data is evaluated on various planes.\n";

    Core::Utils::string_parameter(
        "CANONICAL_FLOW", "no", canonical_flow_doc, &fdyn_turbu, canonical_flow_valid_input);
  }

  std::vector<std::string> homdir_valid_input = {
      "not_specified", "x", "y", "z", "xy", "xz", "yz", "xyz"};
  std::string homdir_doc =
      "Specify the homogeneous direction(s) of a flow.\n"
      "not_specified: no homogeneous directions available, averaging is restricted to time "
      "averaging\n"
      "x: average along x-direction\n"
      "y: average along y-direction\n"
      "z: average along z-direction\n"
      "xy: Wall normal direction is z, average in x and y direction\n"
      "xz: Wall normal direction is y, average in x and z direction\n"
      "yz: Wall normal direction is x, average in y and z direction\n"
      "xyz: Averaging in all directions\n";
  Core::Utils::string_parameter(
      "HOMDIR", "not_specified", homdir_doc, &fdyn_turbu, homdir_valid_input);

  //---------------------------------------
  // further problem-specific parameters

  // CHANNEL FLOW
  //--------------

  Core::Utils::double_parameter("CHAN_AMPL_INIT_DIST", 0.1,
      "Max. amplitude of the random disturbance in percent of the initial value in mean flow "
      "direction.",
      &fdyn_turbu);

  setStringToIntegralParameter<ForcingType>("FORCING_TYPE",
      "linear_compensation_from_intermediate_spectrum", "forcing strategy",
      tuple<std::string>("linear_compensation_from_intermediate_spectrum", "fixed_power_input"),
      tuple<ForcingType>(linear_compensation_from_intermediate_spectrum, fixed_power_input),
      &fdyn_turbu);

  Core::Utils::int_parameter(
      "CHA_NUMSUBDIVISIONS", 5, "Number of homogeneous sampling planes in element", &fdyn_turbu);

  // HIT
  //--------------

  Core::Utils::int_parameter("FORCING_TIME_STEPS", 0,
      "Number of time steps during which forcing is applied. Decaying homogeneous isotropic "
      "turbulence only.",
      &fdyn_turbu);

  Core::Utils::double_parameter("THRESHOLD_WAVENUMBER", 0.0,
      "Forcing is only applied to wave numbers lower or equal than the given threshold wave "
      "number.",
      &fdyn_turbu);

  Core::Utils::double_parameter("POWER_INPUT", 0.0, "power of forcing", &fdyn_turbu);

  std::vector<std::string> scalar_forcing_valid_input = {"no", "isotropic", "mean_scalar_gradient"};
  std::string scalar_forcing_doc =
      "no: Do not force the scalar field\n"
      "isotropic: Force scalar field isotropically such as the fluid field.\n"
      "mean_scalar_gradient: Force scalar field by imposed mean-scalar gradient.\n";
  Core::Utils::string_parameter(
      "SCALAR_FORCING", "no", scalar_forcing_doc, &fdyn_turbu, scalar_forcing_valid_input);

  Core::Utils::double_parameter("MEAN_SCALAR_GRADIENT", 0.0,
      "Value of imposed mean-scalar gradient to force scalar field.", &fdyn_turbu);

  // filtering with xfem
  //--------------

  Core::Utils::bool_parameter("EXCLUDE_XFEM", "No",
      "Flag to (de)activate XFEM dofs in calculation of fine-scale velocity.", &fdyn_turbu);

  /*----------------------------------------------------------------------*/
  // sublist with additional input parameters for Smagorinsky model
  Teuchos::ParameterList& fdyn_turbsgv = fdyn.sublist("SUBGRID VISCOSITY", false, "");

  Core::Utils::double_parameter("C_SMAGORINSKY", 0.0,
      "Constant for the Smagorinsky model. Something between 0.1 to 0.24. Vreman constant if the "
      "constant vreman model is applied (something between 0.07 and 0.01).",
      &fdyn_turbsgv);
  Core::Utils::double_parameter("C_YOSHIZAWA", -1.0,
      "Constant for the compressible Smagorinsky model: isotropic part of subgrid-stress tensor. "
      "About 0.09 or 0.0066. Ci will not be squared!",
      &fdyn_turbsgv);
  Core::Utils::bool_parameter("C_SMAGORINSKY_AVERAGED", "No",
      "Flag to (de)activate averaged Smagorinksy constant", &fdyn_turbsgv);
  Core::Utils::bool_parameter(
      "C_INCLUDE_CI", "No", "Flag to (de)inclusion of Yoshizawa model", &fdyn_turbsgv);
  // remark: following Moin et al. 1991, the extension of the dynamic Smagorinsky model to
  // compressibel flow
  //        also contains a model for the isotropic part of the subgrid-stress tensor according to
  //        Yoshizawa 1989 although used in literature for turbulent variable-density flow at low
  //        Mach number, this additional term seems to destabilize the simulation when the flow is
  //        only weakly compressible therefore C_INCLUDE_CI allows to exclude this term if
  //        C_SMAGORINSKY_AVERAGED == true
  //           if C_INCLUDE_CI==true and C_YOSHIZAWA>=0.0 then the given value C_YOSHIZAWA is used
  //           if C_INCLUDE_CI==true and C_YOSHIZAWA<0.0 then C_YOSHIZAWA is determined dynamically
  //        else all values are taken from input

  Core::Utils::double_parameter("CHANNEL_L_TAU", 0.0,
      "Used for normalisation of the wall normal distance in the Van \nDriest Damping function. "
      "May be taken from the output of \nthe apply_mesh_stretching.pl preprocessing script.",
      &fdyn_turbsgv);

  Core::Utils::double_parameter("C_TURBPRANDTL", 1.0,
      "(Constant) turbulent Prandtl number for the Smagorinsky model in scalar transport.",
      &fdyn_turbsgv);

  setStringToIntegralParameter<VremanFiMethod>("FILTER_WIDTH", "CubeRootVol",
      "The Vreman model requires a filter width.",
      tuple<std::string>("CubeRootVol", "Direction_dependent", "Minimum_length"),
      tuple<VremanFiMethod>(cuberootvol, dir_dep, min_len), &fdyn_turbsgv);


  /*----------------------------------------------------------------------*/
  // sublist with additional input parameters for Smagorinsky model
  Teuchos::ParameterList& fdyn_wallmodel = fdyn.sublist("WALL MODEL", false, "");

  Core::Utils::bool_parameter("X_WALL", "No", "Flag to switch on the xwall model", &fdyn_wallmodel);


  std::vector<std::string> tauw_type_valid_input = {"constant", "between_steps"};
  std::string tauw_type_doc =
      "constant: Use the constant wall shear stress given in the input file for the whole "
      "simulation.\n"
      "between_steps: Calculate wall shear stress in between time steps.\n";
  Core::Utils::string_parameter(
      "Tauw_Type", "constant", tauw_type_doc, &fdyn_wallmodel, tauw_type_valid_input);

  std::vector<std::string> tauw_calc_type_valid_input = {
      "residual", "gradient", "gradient_to_residual"};
  std::string tauw_calc_type_doc =
      "residual: Residual (force) divided by area.\n"
      "gradient: Gradient via shape functions and nodal values.\n"
      "gradient_to_residual: First gradient, then residual.\n";
  Core::Utils::string_parameter("Tauw_Calc_Type", "residual", tauw_calc_type_doc, &fdyn_wallmodel,
      tauw_calc_type_valid_input);


  Core::Utils::int_parameter(
      "Switch_Step", -1, "Switch from gradient to residual based tauw.", &fdyn_wallmodel);

  std::vector<std::string> projection_valid_input = {
      "No", "onlyl2projection", "l2projectionwithcontinuityconstraint"};
  std::string projection_doc =
      "Flag to switch projection of the enriched dofs after updating tauw, alternatively with or "
      "without continuity constraint.";
  Core::Utils::string_parameter(
      "Projection", "No", projection_doc, &fdyn_wallmodel, projection_valid_input);

  Core::Utils::double_parameter("C_Tauw", 1.0,
      "Constant wall shear stress for Spalding's law, if applicable", &fdyn_wallmodel);

  Core::Utils::double_parameter("Min_Tauw", 2.0e-9,
      "Minimum wall shear stress preventing system to become singular", &fdyn_wallmodel);

  Core::Utils::double_parameter(
      "Inc_Tauw", 1.0, "Increment of Tauw of full step, between 0.0 and 1.0", &fdyn_wallmodel);

  std::vector<std::string> blending_type_valid_input = {"none", "ramp_function"};
  std::string blending_type_doc =
      "Methods for blending the enrichment space.\n"
      "none: No ramp function, does not converge!\n"
      "ramp_function: Enrichment is multiplied with linear ramp function resulting in zero "
      "enrichment at the interface.\n";
  Core::Utils::string_parameter(
      "Blending_Type", "none", blending_type_doc, &fdyn_wallmodel, blending_type_valid_input);


  Core::Utils::int_parameter(
      "GP_Wall_Normal", 3, "Gauss points in wall normal direction", &fdyn_wallmodel);
  Core::Utils::int_parameter("GP_Wall_Normal_Off_Wall", 3,
      "Gauss points in wall normal direction, off-wall elements", &fdyn_wallmodel);
  Core::Utils::int_parameter(
      "GP_Wall_Parallel", 3, "Gauss points in wall parallel direction", &fdyn_wallmodel);

  Core::Utils::bool_parameter("Treat_Tauw_on_Dirichlet_Inflow", "No",
      "Flag to treat residual on Dirichlet inflow nodes for calculation of wall shear stress",
      &fdyn_wallmodel);

  Core::Utils::int_parameter(
      "PROJECTION_SOLVER", -1, "Set solver number for l2-projection.", &fdyn_wallmodel);

  /*----------------------------------------------------------------------*/
  // sublist with additional input parameters for multifractal subgrid-scales
  Teuchos::ParameterList& fdyn_turbmfs = fdyn.sublist("MULTIFRACTAL SUBGRID SCALES", false, "");

  Core::Utils::double_parameter(
      "CSGS", 0.0, "Modelparameter of multifractal subgrid-scales.", &fdyn_turbmfs);

  std::vector<std::string> scale_separation_valid_input = {
      "no_scale_sep", "box_filter", "algebraic_multigrid_operator"};
  std::string scale_separation_doc =
      "Specify the filter type for scale separation in LES.\n"
      "no_scale_sep: no scale separation.\n"
      "box_filter: classical box filter.\n"
      "algebraic_multigrid_operator: scale separation by algebraic multigrid operator.\n";
  Core::Utils::string_parameter("SCALE_SEPARATION", "no_scale_sep", scale_separation_doc,
      &fdyn_turbmfs, scale_separation_valid_input);


  Core::Utils::int_parameter("ML_SOLVER", -1,
      "Set solver number for scale separation via level set transfer operators from plain "
      "aggregation.",
      &fdyn_turbmfs);

  Core::Utils::bool_parameter("CALC_N", "No",
      "Flag to (de)activate calculation of N from the Reynolds number.", &fdyn_turbmfs);

  Core::Utils::double_parameter("N", 1.0, "Set grid to viscous scale ratio.", &fdyn_turbmfs);

  std::vector<std::string> ref_length_valid_input = {
      "cube_edge", "sphere_diameter", "streamlength", "gradient_based", "metric_tensor"};
  std::string ref_length_doc =
      "Specify the reference length for Re-dependent N.\n"
      "cube_edge: edge length of volume equivalent cube.\n"
      "sphere_diameter: diameter of volume equivalent sphere.\n"
      "streamlength: streamlength taken from stabilization.\n"
      "gradient_based: gradient based length taken from stabilization.\n"
      "metric_tensor: metric tensor taken from stabilization.\n";
  Core::Utils::string_parameter(
      "REF_LENGTH", "cube_edge", ref_length_doc, &fdyn_turbmfs, ref_length_valid_input);

  std::vector<std::string> ref_velocity_valid_input = {"strainrate", "resolved", "fine_scale"};
  std::string ref_velocity_doc =
      "Specify the reference velocity for Re-dependent N.\n"
      "strainrate: norm of strain rate.\n"
      "resolved: resolved velocity.\n"
      "fine_scale: fine-scale velocity.\n";
  Core::Utils::string_parameter(
      "REF_VELOCITY", "strainrate", ref_velocity_doc, &fdyn_turbmfs, ref_velocity_valid_input);


  Core::Utils::double_parameter("C_NU", 1.0,
      "Proportionality constant between Re and ratio viscous scale to element length.",
      &fdyn_turbmfs);

  Core::Utils::bool_parameter(
      "NEAR_WALL_LIMIT", "No", "Flag to (de)activate near-wall limit.", &fdyn_turbmfs);

  std::vector<std::string> evaluation_b_valid_input = {"element_center", "integration_point"};
  std::string evaluation_b_doc =
      "Location where B is evaluated\n"
      "element_center: evaluate B at element center.\n"
      "integration_point: evaluate B at integration point.\n";
  Core::Utils::string_parameter(
      "EVALUATION_B", "element_center", evaluation_b_doc, &fdyn_turbmfs, evaluation_b_valid_input);


  Core::Utils::double_parameter(
      "BETA", 0.0, "Cross- and Reynolds-stress terms only on right-hand-side.", &fdyn_turbmfs);

  convform_valid_input = {"convective", "conservative"};
  std::string convform_doc =
      "form of convective term\n"
      "convective: Use the convective form.\n"
      "conservative: Use the conservative form.\n";
  Core::Utils::string_parameter(
      "CONVFORM", "convective", convform_doc, &fdyn_turbmfs, convform_valid_input);

  Core::Utils::double_parameter("CSGS_PHI", 0.0,
      "Modelparameter of multifractal subgrid-scales for scalar transport.", &fdyn_turbmfs);

  Core::Utils::bool_parameter(
      "ADAPT_CSGS_PHI", "No", "Flag to (de)activate adaption of CsgsD to CsgsB.", &fdyn_turbmfs);

  Core::Utils::bool_parameter("NEAR_WALL_LIMIT_CSGS_PHI", "No",
      "Flag to (de)activate near-wall limit for scalar field.", &fdyn_turbmfs);

  Core::Utils::bool_parameter("CONSISTENT_FLUID_RESIDUAL", "No",
      "Flag to (de)activate the consistency term for residual-based stabilization.", &fdyn_turbmfs);

  Core::Utils::double_parameter("C_DIFF", 1.0,
      "Proportionality constant between Re*Pr and ratio dissipative scale to element length. "
      "Usually equal cnu.",
      &fdyn_turbmfs);

  Core::Utils::bool_parameter("SET_FINE_SCALE_VEL", "No",
      "Flag to set fine-scale velocity for parallel nightly tests.", &fdyn_turbmfs);

  // activate cross- and Reynolds-stress terms in loma continuity equation
  Core::Utils::bool_parameter("LOMA_CONTI", "No",
      "Flag to (de)activate cross- and Reynolds-stress terms in loma continuity equation.",
      &fdyn_turbmfs);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& fdyn_turbinf = fdyn.sublist("TURBULENT INFLOW", false, "");

  Core::Utils::bool_parameter("TURBULENTINFLOW", "No",
      "Flag to (de)activate potential separate turbulent inflow section", &fdyn_turbinf);

  setStringToIntegralParameter<InitialField>("INITIALINFLOWFIELD", "zero_field",
      "Initial field for inflow section",
      tuple<std::string>("zero_field", "field_by_function", "disturbed_field_from_function"),
      tuple<InitialField>(initfield_zero_field, initfield_field_by_function,
          initfield_disturbed_field_from_function),
      &fdyn_turbinf);

  Core::Utils::int_parameter(
      "INFLOWFUNC", -1, "Function number for initial flow field in inflow section", &fdyn_turbinf);

  Core::Utils::double_parameter("INFLOW_INIT_DIST", 0.1,
      "Max. amplitude of the random disturbance in percent of the initial value in mean flow "
      "direction.",
      &fdyn_turbinf);

  Core::Utils::int_parameter("NUMINFLOWSTEP", 1,
      "Total number of time steps for development of turbulent flow", &fdyn_turbinf);

  std::vector<std::string> canonical_inflow_valid_input = {"no", "time_averaging",
      "channel_flow_of_height_2", "loma_channel_flow_of_height_2",
      "scatra_channel_flow_of_height_2"};
  std::string canonical_inflow_doc =
      "Sampling is different for different canonical flows \n--- so specify what kind of flow "
      "you've got\n"
      "no: The flow is not further specified, so spatial averaging \nand hence the standard "
      "sampling procedure is not possible.\n"
      "time_averaging: The flow is not further specified, but time averaging of velocity and "
      "pressure field is performed.\n"
      "channel_flow_of_height_2: For this flow, all statistical data could be averaged in \nthe "
      "homogeneous planes --- it is essentially a statistically one dimensional flow.\n"
      "loma_channel_flow_of_height_2: For this low-Mach-number flow, all statistical data could be "
      "averaged in \nthe homogeneous planes --- it is essentially a statistically one dimensional "
      "flow.\n"
      "scatra_channel_flow_of_height_2: For this flow, all statistical data could be averaged in "
      "\nthe homogeneous planes --- it is essentially a statistically one dimensional flow.\n";
  Core::Utils::string_parameter(
      "CANONICAL_INFLOW", "no", canonical_inflow_doc, &fdyn_turbinf, canonical_inflow_valid_input);


  Core::Utils::double_parameter("INFLOW_CHA_SIDE", 0.0,
      "Most right side of inflow channel. Necessary to define sampling domain.", &fdyn_turbinf);

  std::vector<std::string> inflow_homdir_valid_input = {
      "not_specified", "x", "y", "z", "xy", "xz", "yz"};
  std::string inflow_homdir_doc =
      "Specify the homogeneous direction(s) of a flow\n"
      "not_specified: no homogeneous directions available, averaging is restricted to time "
      "averaging.\n"
      "x: average along x-direction.\n"
      "y: average along y-direction.\n"
      "z: average along z-direction.\n"
      "xy: Wall normal direction is z, average in x and y direction.\n"
      "xz: Wall normal direction is y, average in x and z direction (standard case).\n"
      "yz: Wall normal direction is x, average in y and z direction.\n";
  Core::Utils::string_parameter("INFLOW_HOMDIR", "not_specified", inflow_homdir_doc, &fdyn_turbinf,
      inflow_homdir_valid_input);

  Core::Utils::int_parameter("INFLOW_SAMPLING_START", 10000000,
      "Time step after when sampling shall be started", &fdyn_turbinf);
  Core::Utils::int_parameter(
      "INFLOW_SAMPLING_STOP", 1, "Time step when sampling shall be stopped", &fdyn_turbinf);
  Core::Utils::int_parameter("INFLOW_DUMPING_PERIOD", 1,
      "Period of time steps after which statistical data shall be dumped", &fdyn_turbinf);

  /*----------------------------------------------------------------------*/
  // sublist with additional input parameters for time adaptivity in fluid/ coupled problems
  Teuchos::ParameterList& fdyn_timintada = fdyn.sublist("TIMEADAPTIVITY", false, "");
  setStringToIntegralParameter<AdaptiveTimeStepEstimator>("ADAPTIVE_TIME_STEP_ESTIMATOR", "none",
      "Method used to determine adaptive time step size.",
      tuple<std::string>("none", "cfl_number", "only_print_cfl_number"),
      tuple<std::string>(
          "constant time step", "evaluated via CFL number", "CFL number evaluated and printed"
          //""
          ),
      tuple<AdaptiveTimeStepEstimator>(const_dt, cfl_number, only_print_cfl_number),
      &fdyn_timintada);

  Core::Utils::double_parameter(
      "CFL_NUMBER", -1.0, "CFL number for adaptive time step", &fdyn_timintada);
  Core::Utils::int_parameter("FREEZE_ADAPTIVE_DT_AT", 1000000,
      "keep time step constant after this step, otherwise turbulence statistics sampling is not "
      "consistent",
      &fdyn_timintada);
  Core::Utils::double_parameter(
      "ADAPTIVE_DT_INC", 0.8, "Increment of whole step for adaptive dt via CFL", &fdyn_timintada);
}



void Inpar::LowMach::set_valid_parameters(Teuchos::ParameterList& list)
{
  using namespace Input;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& lomacontrol =
      list.sublist("LOMA CONTROL", false, "control parameters for low-Mach-number flow problems\n");

  Core::Utils::bool_parameter("MONOLITHIC", "no", "monolithic solver", &lomacontrol);
  Core::Utils::int_parameter("NUMSTEP", 24, "Total number of time steps", &lomacontrol);
  Core::Utils::double_parameter("TIMESTEP", 0.1, "Time increment dt", &lomacontrol);
  Core::Utils::double_parameter("MAXTIME", 1000.0, "Total simulation time", &lomacontrol);
  Core::Utils::int_parameter("ITEMAX", 10, "Maximum number of outer iterations", &lomacontrol);
  Core::Utils::int_parameter("ITEMAX_BEFORE_SAMPLING", 1,
      "Maximum number of outer iterations before sampling (for turbulent flows only)",
      &lomacontrol);
  Core::Utils::double_parameter("CONVTOL", 1e-6, "Tolerance for convergence check", &lomacontrol);
  Core::Utils::int_parameter("RESULTSEVERY", 1, "Increment for writing solution", &lomacontrol);
  Core::Utils::int_parameter("RESTARTEVERY", 1, "Increment for writing restart", &lomacontrol);

  std::vector<std::string> constthermpress_valid_input = {"No_energy", "No_mass", "Yes"};
  Core::Utils::string_parameter("CONSTHERMPRESS", "Yes",
      "treatment of thermodynamic pressure in time", &lomacontrol, constthermpress_valid_input);

  Core::Utils::bool_parameter("SGS_MATERIAL_UPDATE", "no",
      "update material by adding subgrid-scale scalar field", &lomacontrol);

  // number of linear solver used for LOMA solver
  Core::Utils::int_parameter(
      "LINEAR_SOLVER", -1, "number of linear solver used for LOMA problem", &lomacontrol);
}


void Inpar::FLUID::set_valid_conditions(
    std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>& condlist)
{
  using namespace Input;

  /*--------------------------------------------------------------------*/
  // transfer boundary condition for turbulent inflow

  std::shared_ptr<Core::Conditions::ConditionDefinition> tbc_turb_inflow =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF TURBULENT INFLOW TRANSFER", "TransferTurbulentInflow",
          "TransferTurbulentInflow", Core::Conditions::TransferTurbulentInflow, true,
          Core::Conditions::geometry_type_surface);

  add_named_int(tbc_turb_inflow, "ID", "", 0, false);
  add_named_selection_component(tbc_turb_inflow, "toggle", "toggle", "master",
      Teuchos::tuple<std::string>("master", "slave"),
      Teuchos::tuple<std::string>("master", "slave"));
  add_named_selection_component(tbc_turb_inflow, "DIRECTION", "transfer direction", "x",
      Teuchos::tuple<std::string>("x", "y", "z"), Teuchos::tuple<int>(0, 1, 2));
  add_named_int(tbc_turb_inflow, "curve", "curve id", 0, false);

  condlist.push_back(tbc_turb_inflow);

  /*--------------------------------------------------------------------*/
  // separate domain for turbulent inflow generation

  std::shared_ptr<Core::Conditions::ConditionDefinition> turbulentinflowgeneration =
      std::make_shared<Core::Conditions::ConditionDefinition>("FLUID TURBULENT INFLOW VOLUME",
          "TurbulentInflowSection", "TurbulentInflowSection",
          Core::Conditions::TurbulentInflowSection, true, Core::Conditions::geometry_type_volume);

  condlist.push_back(turbulentinflowgeneration);


  /*--------------------------------------------------------------------*/
  // flow-dependent pressure conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> surfflowdeppressure =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURFACE FLOW-DEPENDENT PRESSURE CONDITIONS", "SurfaceFlowDepPressure",
          "SurfaceFlowDepPressure", Core::Conditions::SurfaceFlowDepPressure, true,
          Core::Conditions::geometry_type_surface);

  // flow-dependent pressure conditions can be imposed either based on
  // (out)flow rate or (out)flow volume (e.g., for air-cushion condition)
  add_named_selection_component(surfflowdeppressure, "TYPE_OF_FLOW_DEPENDENCE",
      "type of flow dependence", "flow_rate",
      Teuchos::tuple<std::string>("flow_rate", "flow_volume", "fixed_pressure"),
      Teuchos::tuple<std::string>("flow_rate", "flow_volume", "fixed_pressure"));
  add_named_real(surfflowdeppressure, "ConstCoeff",
      "constant coefficient for (linear) flow-rate-based condition");
  add_named_real(
      surfflowdeppressure, "LinCoeff", "linear coefficient for (linear) flow-rate-based condition");
  add_named_real(
      surfflowdeppressure, "InitialVolume", "initial (air-cushion) volume outside of boundary");
  add_named_real(
      surfflowdeppressure, "ReferencePressure", " reference pressure outside of boundary");
  add_named_real(surfflowdeppressure, "AdiabaticExponent", "adiabatic exponent");
  add_named_int(surfflowdeppressure, "curve", "curve id", 0, false);

  condlist.emplace_back(surfflowdeppressure);


  /*--------------------------------------------------------------------*/
  // Slip Supplemental Curved Boundary conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> lineslipsupp =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE SLIP SUPPLEMENTAL CURVED BOUNDARY CONDITIONS", "LineSlipSupp",
          "LineSlipSupp", Core::Conditions::LineSlipSupp, true,
          Core::Conditions::geometry_type_line);

  std::shared_ptr<Core::Conditions::ConditionDefinition> surfslipsupp =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURFACE SLIP SUPPLEMENTAL CURVED BOUNDARY CONDITIONS", "SurfaceSlipSupp",
          "SurfaceSlipSupp", Core::Conditions::SurfaceSlipSupp, true,
          Core::Conditions::geometry_type_surface);

  for (const auto& cond : {lineslipsupp, surfslipsupp})
  {
    add_named_real(cond, "USEUPDATEDNODEPOS");
    condlist.emplace_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // Navier-slip boundary conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> linenavierslip =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE NAVIER-SLIP BOUNDARY CONDITIONS", "LineNavierSlip", "LineNavierSlip",
          Core::Conditions::LineNavierSlip, true, Core::Conditions::geometry_type_line);

  std::shared_ptr<Core::Conditions::ConditionDefinition> surfnavierslip =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF NAVIER-SLIP BOUNDARY CONDITIONS", "SurfNavierSlip", "SurfNavierSlip",
          Core::Conditions::SurfNavierSlip, true, Core::Conditions::geometry_type_surface);

  for (const auto& cond : {linenavierslip, surfnavierslip})
  {
    add_named_real(cond, "SLIPCOEFFICIENT");
    condlist.emplace_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // consistent outflow bcs for conservative element formulations

  std::shared_ptr<Core::Conditions::ConditionDefinition> surfconsistentoutflowconsistency =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURFACE CONSERVATIVE OUTFLOW CONSISTENCY",
          "SurfaceConservativeOutflowConsistency", "SurfaceConservativeOutflowConsistency",
          Core::Conditions::SurfaceConservativeOutflowConsistency, true,
          Core::Conditions::geometry_type_surface);

  condlist.push_back(surfconsistentoutflowconsistency);

  /*--------------------------------------------------------------------*/
  // Neumann condition for fluid that can handle inflow/backflow

  std::shared_ptr<Core::Conditions::ConditionDefinition> linefluidneumanninflow =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "FLUID NEUMANN INFLOW LINE CONDITIONS", "FluidNeumannInflow", "Line Fluid Neumann Inflow",
          Core::Conditions::FluidNeumannInflow, true, Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surffluidneumanninflow =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "FLUID NEUMANN INFLOW SURF CONDITIONS", "FluidNeumannInflow",
          "Surface Fluid Neumann Inflow", Core::Conditions::FluidNeumannInflow, true,
          Core::Conditions::geometry_type_surface);

  condlist.push_back(linefluidneumanninflow);
  condlist.push_back(surffluidneumanninflow);

  /*--------------------------------------------------------------------*/
  // mixed/hybrid Dirichlet conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> linemixhybDirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE MIXED/HYBRID DIRICHLET CONDITIONS", "LineMixHybDirichlet",
          "LineMixHybDirichlet", Core::Conditions::LineMixHybDirichlet, true,
          Core::Conditions::geometry_type_line);


  std::shared_ptr<Core::Conditions::ConditionDefinition> surfmixhybDirichlet =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURFACE MIXED/HYBRID DIRICHLET CONDITIONS", "SurfaceMixHybDirichlet",
          "SurfaceMixHybDirichlet", Core::Conditions::SurfaceMixHybDirichlet, true,
          Core::Conditions::geometry_type_surface);

  // we attach all the components of this condition to this condition
  for (const auto& cond : {linemixhybDirichlet, surfmixhybDirichlet})
  {
    // we provide a vector of 3 values for velocities
    add_named_real_vector(cond, "val", "velocity", 3);

    // and optional spatial functions
    add_named_int_vector(cond, "funct", "spatial function", 3, 0, true);

    // characteristic velocity
    add_named_real(cond, "u_C");

    // the penalty parameter could be computed dynamically (using Spaldings
    // law of the wall) or using a fixed value (1)
    add_named_selection_component(cond, "PENTYPE", "how penalty parameter is computed", "constant",
        Teuchos::tuple<std::string>("constant", "Spalding"),
        Teuchos::tuple<std::string>("constant", "Spalding"));

    // scaling factor for penalty parameter tauB
    add_named_real(cond, "hB_divided_by");

    // if Spaldings law is used, this defines the way how the traction at y is computed from utau
    add_named_selection_component(cond, "utau_computation",
        "how traction at y is computed from utau", "at_wall",
        Teuchos::tuple<std::string>("at_wall", "viscous_tangent"),
        Teuchos::tuple<std::string>("at_wall", "viscous_tangent"));

    // we append it to the list of all conditions
    condlist.push_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // fluid stress

  std::shared_ptr<Core::Conditions::ConditionDefinition> linefluidstress =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN FLUID STRESS CALC LINE CONDITIONS", "FluidStressCalc",
          "Line Fluid Stress Calculation", Core::Conditions::FluidStressCalc, true,
          Core::Conditions::geometry_type_line);
  std::shared_ptr<Core::Conditions::ConditionDefinition> surffluidstress =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN FLUID STRESS CALC SURF CONDITIONS", "FluidStressCalc",
          "Surf Fluid Stress Calculation", Core::Conditions::FluidStressCalc, true,
          Core::Conditions::geometry_type_surface);

  condlist.push_back(linefluidstress);
  condlist.push_back(surffluidstress);

  /*--------------------------------------------------------------------*/
  // lift & drag
  std::shared_ptr<Core::Conditions::ConditionDefinition> surfliftdrag =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN FLUID SURF LIFT&DRAG",
          "LIFTDRAG", "Surface LIFTDRAG", Core::Conditions::SurfLIFTDRAG, true,
          Core::Conditions::geometry_type_surface);

  add_named_int(surfliftdrag, "label");
  add_named_real_vector(surfliftdrag, "CENTER", "", 3);
  add_named_real_vector(surfliftdrag, "AXIS", "", 3, 0.0, true);

  condlist.push_back(surfliftdrag);

  /*--------------------------------------------------------------------*/
  // flow rate through line

  std::shared_ptr<Core::Conditions::ConditionDefinition> lineflowrate =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN FLOW RATE LINE CONDITIONS",
          "LineFlowRate", "Line Flow Rate", Core::Conditions::FlowRateThroughLine_2D, true,
          Core::Conditions::geometry_type_line);

  add_named_int(lineflowrate, "ConditionID");

  condlist.push_back(lineflowrate);

  /*--------------------------------------------------------------------*/
  // flow rate through surface

  std::shared_ptr<Core::Conditions::ConditionDefinition> surfflowrate =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN FLOW RATE SURF CONDITIONS",
          "SurfFlowRate", "Surface Flow Rate", Core::Conditions::FlowRateThroughSurface_3D, true,
          Core::Conditions::geometry_type_surface);

  add_named_int(surfflowrate, "ConditionID");

  condlist.push_back(surfflowrate);

  /*--------------------------------------------------------------------*/
  // Volumetric surface flow profile condition
  std::shared_ptr<Core::Conditions::ConditionDefinition> volumetric_surface_flow_cond =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF VOLUMETRIC FLOW CONDITIONS", "VolumetricSurfaceFlowCond",
          "volumetric surface flow condition", Core::Conditions::VolumetricSurfaceFlowCond, true,
          Core::Conditions::geometry_type_surface);

  add_named_int(volumetric_surface_flow_cond, "ConditionID");

  Input::add_named_selection_component(volumetric_surface_flow_cond, "ConditionType",
      "condition type", "POLYNOMIAL", Teuchos::tuple<std::string>("POLYNOMIAL", "WOMERSLEY"),
      Teuchos::tuple<std::string>("POLYNOMIAL", "WOMERSLEY"), true);

  Input::add_named_selection_component(volumetric_surface_flow_cond, "prebiased", "prebiased",
      "NOTPREBIASED", Teuchos::tuple<std::string>("NOTPREBIASED", "PREBIASED", "FORCED"),
      Teuchos::tuple<std::string>("NOTPREBIASED", "PREBIASED", "FORCED"), true);

  Input::add_named_selection_component(volumetric_surface_flow_cond, "FlowType", "flow type",
      "InFlow", Teuchos::tuple<std::string>("InFlow", "OutFlow"),
      Teuchos::tuple<std::string>("InFlow", "OutFlow"), true);
  Input::add_named_selection_component(volumetric_surface_flow_cond, "CorrectionFlag",
      "correction flag", "WithOutCorrection",
      Teuchos::tuple<std::string>("WithOutCorrection", "WithCorrection"),
      Teuchos::tuple<std::string>("WithOutCorrection", "WithCorrection"), true);

  Input::add_named_real(volumetric_surface_flow_cond, "Period");
  Input::add_named_int(volumetric_surface_flow_cond, "Order");
  Input::add_named_int(volumetric_surface_flow_cond, "Harmonics");
  Input::add_named_real(volumetric_surface_flow_cond, "Val");
  Input::add_named_int(volumetric_surface_flow_cond, "Funct");

  Input::add_named_selection_component(volumetric_surface_flow_cond, "NORMAL", "normal",
      "SelfEvaluateNormal",
      Teuchos::tuple<std::string>("SelfEvaluateNormal", "UsePrescribedNormal"),
      Teuchos::tuple<std::string>("SelfEvaluateNormal", "UsePrescribedNormal"), true);
  Input::add_named_real(volumetric_surface_flow_cond, "n1");
  Input::add_named_real(volumetric_surface_flow_cond, "n2");
  Input::add_named_real(volumetric_surface_flow_cond, "n3");

  Input::add_named_selection_component(volumetric_surface_flow_cond, "CenterOfMass",
      "center of mass", "SelfEvaluateCenterOfMass",
      Teuchos::tuple<std::string>("SelfEvaluateCenterOfMass", "UsePrescribedCenterOfMass"),
      Teuchos::tuple<std::string>("SelfEvaluateCenterOfMass", "UsePrescribedCenterOfMass"), true);
  Input::add_named_real(volumetric_surface_flow_cond, "c1");
  Input::add_named_real(volumetric_surface_flow_cond, "c2");
  Input::add_named_real(volumetric_surface_flow_cond, "c3");

  condlist.push_back(volumetric_surface_flow_cond);



  /*--------------------------------------------------------------------*/
  // Volumetric flow border nodes condition

  std::shared_ptr<Core::Conditions::ConditionDefinition> volumetric_border_nodes_cond =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE VOLUMETRIC FLOW BORDER NODES", "VolumetricFlowBorderNodesCond",
          "volumetric flow border nodes condition", Core::Conditions::VolumetricFlowBorderNodes,
          true, Core::Conditions::geometry_type_line);

  add_named_int(volumetric_border_nodes_cond, "ConditionID");

  condlist.push_back(volumetric_border_nodes_cond);

  /*--------------------------------------------------------------------*/
  // Volumetric surface total traction corrector
  std::shared_ptr<Core::Conditions::ConditionDefinition> total_traction_correction_cond =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURF TOTAL TRACTION CORRECTION CONDITIONS", "TotalTractionCorrectionCond",
          "total traction correction condition", Core::Conditions::TotalTractionCorrectionCond,
          true, Core::Conditions::geometry_type_surface);

  Input::add_named_int(total_traction_correction_cond, "ConditionID");
  Input::add_named_selection_component(total_traction_correction_cond, "ConditionType",
      "condition type", "POLYNOMIAL", Teuchos::tuple<std::string>("POLYNOMIAL", "WOMERSLEY"),
      Teuchos::tuple<std::string>("POLYNOMIAL", "WOMERSLEY"), true);

  Input::add_named_selection_component(total_traction_correction_cond, "prebiased", "prebiased",
      "NOTPREBIASED", Teuchos::tuple<std::string>("NOTPREBIASED", "PREBIASED", "FORCED"),
      Teuchos::tuple<std::string>("NOTPREBIASED", "PREBIASED", "FORCED"), true);

  Input::add_named_selection_component(total_traction_correction_cond, "FlowType", "flow type",
      "InFlow", Teuchos::tuple<std::string>("InFlow", "OutFlow"),
      Teuchos::tuple<std::string>("InFlow", "OutFlow"), true);
  Input::add_named_selection_component(total_traction_correction_cond, "CorrectionFlag",
      "correction flag", "WithOutCorrection",
      Teuchos::tuple<std::string>("WithOutCorrection", "WithCorrection"),
      Teuchos::tuple<std::string>("WithOutCorrection", "WithCorrection"), true);

  Input::add_named_real(total_traction_correction_cond, "Period");
  Input::add_named_int(total_traction_correction_cond, "Order");
  Input::add_named_int(total_traction_correction_cond, "Harmonics");
  Input::add_named_real(total_traction_correction_cond, "Val");
  Input::add_named_int(total_traction_correction_cond, "Funct");

  Input::add_named_selection_component(total_traction_correction_cond, "NORMAL", "normal",
      "SelfEvaluateNormal",
      Teuchos::tuple<std::string>("SelfEvaluateNormal", "UsePrescribedNormal"),
      Teuchos::tuple<std::string>("SelfEvaluateNormal", "UsePrescribedNormal"), true);
  Input::add_named_real(total_traction_correction_cond, "n1");
  Input::add_named_real(total_traction_correction_cond, "n2");
  Input::add_named_real(total_traction_correction_cond, "n3");

  Input::add_named_selection_component(total_traction_correction_cond, "CenterOfMass",
      "center of mass", "SelfEvaluateCenterOfMass",
      Teuchos::tuple<std::string>("SelfEvaluateCenterOfMass", "UsePrescribedCenterOfMass"),
      Teuchos::tuple<std::string>("SelfEvaluateCenterOfMass", "UsePrescribedCenterOfMass"), true);
  Input::add_named_real(total_traction_correction_cond, "c1");
  Input::add_named_real(total_traction_correction_cond, "c2");
  Input::add_named_real(total_traction_correction_cond, "c3");

  condlist.push_back(total_traction_correction_cond);

  /*--------------------------------------------------------------------*/
  // Volumetric flow traction correction border nodes condition

  std::shared_ptr<Core::Conditions::ConditionDefinition> traction_corrector_border_nodes_cond =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE TOTAL TRACTION CORRECTION BORDER NODES",
          "TotalTractionCorrectionBorderNodesCond",
          "total traction correction border nodes condition",
          Core::Conditions::TotalTractionCorrectionBorderNodes, true,
          Core::Conditions::geometry_type_line);

  add_named_int(traction_corrector_border_nodes_cond, "ConditionID");

  condlist.push_back(traction_corrector_border_nodes_cond);


  /*--------------------------------------------------------------------*/
  // no penetration for darcy flow in porous media

  std::shared_ptr<Core::Conditions::ConditionDefinition> nopenetration_surf =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURFACE NORMAL NO PENETRATION CONDITION", "no_penetration", "No Penetration",
          Core::Conditions::no_penetration, true, Core::Conditions::geometry_type_surface);

  condlist.push_back(nopenetration_surf);

  /*--------------------------------------------------------------------*/
  // no penetration for darcy flow in porous media

  std::shared_ptr<Core::Conditions::ConditionDefinition> nopenetration_line =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE NORMAL NO PENETRATION CONDITION", "no_penetration", "No Penetration",
          Core::Conditions::no_penetration, true, Core::Conditions::geometry_type_line);

  condlist.push_back(nopenetration_line);

  /*--------------------------------------------------------------------*/
  // condition for evaluation of coupling terms in porous media

  std::shared_ptr<Core::Conditions::ConditionDefinition> porocoupling_vol =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN VOLUME POROCOUPLING CONDITION", "PoroCoupling", "Poro Coupling",
          Core::Conditions::PoroCoupling, true, Core::Conditions::geometry_type_volume);

  condlist.push_back(porocoupling_vol);

  /*--------------------------------------------------------------------*/
  // condition for evaluation of coupling terms in porous media

  std::shared_ptr<Core::Conditions::ConditionDefinition> porocoupling_surf =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURFACE POROCOUPLING CONDITION", "PoroCoupling", "Poro Coupling",
          Core::Conditions::PoroCoupling, true, Core::Conditions::geometry_type_surface);

  condlist.push_back(porocoupling_surf);

  /*--------------------------------------------------------------------*/
  // condition for evaluation of boundary terms in porous media problems

  std::shared_ptr<Core::Conditions::ConditionDefinition> poropartint_surf =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURFACE PORO PARTIAL INTEGRATION", "PoroPartInt", "Poro Partial Integration",
          Core::Conditions::PoroPartInt, true, Core::Conditions::geometry_type_surface);

  condlist.push_back(poropartint_surf);

  /*--------------------------------------------------------------------*/
  // condition for evaluation of boundary terms in porous media problems

  std::shared_ptr<Core::Conditions::ConditionDefinition> poropartint_line =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE PORO PARTIAL INTEGRATION", "PoroPartInt", "Poro Partial Integration",
          Core::Conditions::PoroPartInt, true, Core::Conditions::geometry_type_line);

  condlist.push_back(poropartint_line);

  /*--------------------------------------------------------------------*/
  // condition for evaluation of boundary terms in porous media problems

  std::shared_ptr<Core::Conditions::ConditionDefinition> poropresint_surf =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN SURFACE PORO PRESSURE INTEGRATION", "PoroPresInt", "Poro Pressure Integration",
          Core::Conditions::PoroPresInt, true, Core::Conditions::geometry_type_surface);

  condlist.push_back(poropresint_surf);

  /*--------------------------------------------------------------------*/
  // condition for evaluation of boundary terms in porous media problems

  std::shared_ptr<Core::Conditions::ConditionDefinition> poropresint_line =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN LINE PORO PRESSURE INTEGRATION", "PoroPresInt", "Poro Pressure Integration",
          Core::Conditions::PoroPresInt, true, Core::Conditions::geometry_type_line);

  condlist.push_back(poropresint_line);
}

FOUR_C_NAMESPACE_CLOSE
