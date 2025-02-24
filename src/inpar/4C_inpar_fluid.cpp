// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_fluid.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_geometry_type.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_legacy_enum_definitions_conditions.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::FLUID::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;

  Core::Utils::SectionSpecs fdyn{"FLUID DYNAMIC"};

  // physical type of fluid flow (incompressible, varying density, loma, Boussinesq approximation,
  // temperature-dependent water)
  Core::Utils::string_to_integral_parameter<Inpar::FLUID::PhysicalType>("PHYSICAL_TYPE",
      "Incompressible", "Physical Type",
      tuple<std::string>("Incompressible", "Weakly_compressible", "Weakly_compressible_stokes",
          "Weakly_compressible_dens_mom", "Weakly_compressible_stokes_dens_mom",
          "Artificial_compressibility", "Varying_density", "Loma", "Temp_dep_water", "Boussinesq",
          "Stokes", "Oseen"),
      tuple<Inpar::FLUID::PhysicalType>(incompressible, weakly_compressible,
          weakly_compressible_stokes, weakly_compressible_dens_mom,
          weakly_compressible_stokes_dens_mom, artcomp, varying_density, loma, tempdepwater,
          boussinesq, stokes, oseen),
      fdyn);

  // number of linear solver used for fluid problem
  Core::Utils::int_parameter(
      "LINEAR_SOLVER", -1, "number of linear solver used for fluid dynamics", fdyn);

  // number of linear solver used for fluid problem (former fluid pressure solver for SIMPLER
  // preconditioning with fluid)
  Core::Utils::int_parameter("SIMPLER_SOLVER", -1,
      "number of linear solver used for fluid dynamics (ONLY NECESSARY FOR BlockGaussSeidel solver "
      "block within fluid mehstying case any more!!!!)",
      fdyn);

  // Flag to define the way of calculating stresses and wss
  Core::Utils::string_to_integral_parameter<Inpar::FLUID::WSSType>("WSS_TYPE", "Standard",
      "which type of stresses and wall shear stress",
      tuple<std::string>("Standard", "Aggregation", "Mean"),
      tuple<Inpar::FLUID::WSSType>(wss_standard, wss_aggregation, wss_mean), fdyn);

  // Set ML-solver number for smoothing of residual-based calculated wallshearstress via plain
  // aggregation.
  Core::Utils::int_parameter("WSS_ML_AGR_SOLVER", -1,
      "Set ML-solver number for smoothing of residual-based calculated wallshearstress via plain "
      "aggregation.",
      fdyn);

  Core::Utils::string_to_integral_parameter<Inpar::FLUID::TimeIntegrationScheme>("TIMEINTEGR",
      "One_Step_Theta", "Time Integration Scheme",
      tuple<std::string>("Stationary", "Np_Gen_Alpha", "Af_Gen_Alpha", "One_Step_Theta", "BDF2"),
      tuple<Inpar::FLUID::TimeIntegrationScheme>(timeint_stationary, timeint_npgenalpha,
          timeint_afgenalpha, timeint_one_step_theta, timeint_bdf2),
      fdyn);

  Core::Utils::string_to_integral_parameter<Inpar::FLUID::OstContAndPress>("OST_CONT_PRESS",
      "Cont_normal_Press_normal",
      "One step theta option for time discretization of continuity eq. and pressure",
      tuple<std::string>(
          "Cont_normal_Press_normal", "Cont_impl_Press_normal", "Cont_impl_Press_impl"),
      tuple<Inpar::FLUID::OstContAndPress>(
          Cont_normal_Press_normal, Cont_impl_Press_normal, Cont_impl_Press_impl),
      fdyn);

  Core::Utils::string_to_integral_parameter<Core::IO::GeometryType>("GEOMETRY", "full",
      "How the geometry is specified", tuple<std::string>("full", "box", "file"),
      tuple<Core::IO::GeometryType>(
          Core::IO::geometry_full, Core::IO::geometry_box, Core::IO::geometry_file),
      fdyn);

  Core::Utils::string_to_integral_parameter<Inpar::FLUID::LinearisationAction>("NONLINITER",
      "fixed_point_like", "Nonlinear iteration scheme",
      tuple<std::string>("fixed_point_like", "Newton"),
      tuple<Inpar::FLUID::LinearisationAction>(fixed_point_like, Newton), fdyn);

  std::vector<std::string> predictor_valid_input = {"steady_state", "zero_acceleration",
      "constant_acceleration", "constant_increment", "explicit_second_order_midpoint", "TangVel"};
  Core::Utils::string_parameter("PREDICTOR", "steady_state",
      "Predictor for first guess in nonlinear iteration", fdyn, predictor_valid_input);


  Core::Utils::string_to_integral_parameter<Inpar::FLUID::ItNorm>("CONVCHECK", "L_2_norm",
      "Norm for convergence check (relative increments and absolute residuals)",
      tuple<std::string>("L_2_norm"), tuple<Inpar::FLUID::ItNorm>(fncc_L2), fdyn);

  Core::Utils::bool_parameter("INCONSISTENT_RESIDUAL", false,
      "do not evaluate residual after solution has converged (->faster)", fdyn);

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

    Core::Utils::string_to_integral_parameter<Inpar::FLUID::InitialField>(
        "INITIALFIELD", "zero_field", "Initial field for fluid problem", name, label, fdyn);
  }

  Core::Utils::int_parameter(
      "OSEENFIELDFUNCNO", -1, "function number of Oseen advective field", fdyn);

  Core::Utils::bool_parameter(
      "LIFTDRAG", false, "Calculate lift and drag forces along specified boundary", fdyn);

  std::vector<std::string> convform_valid_input = {"convective", "conservative"};
  Core::Utils::string_parameter(
      "CONVFORM", "convective", "form of convective term", fdyn, convform_valid_input);

  Core::Utils::bool_parameter("NONLINEARBC", false,
      "Flag to activate check for potential nonlinear boundary conditions", fdyn);

  Core::Utils::string_to_integral_parameter<Inpar::FLUID::MeshTying>("MESHTYING", "no",
      "Flag to (de)activate mesh tying algorithm",
      tuple<std::string>("no", "Condensed_Smat", "Condensed_Bmat", "Condensed_Bmat_merged"),
      tuple<Inpar::FLUID::MeshTying>(
          no_meshtying, condensed_smat, condensed_bmat, condensed_bmat_merged),
      fdyn);

  Core::Utils::string_to_integral_parameter<Inpar::FLUID::Gridvel>("GRIDVEL", "BE",
      "scheme for determination of gridvelocity from displacements",
      tuple<std::string>("BE", "BDF2", "OST"), tuple<Inpar::FLUID::Gridvel>(BE, BDF2, OST), fdyn);

  Core::Utils::bool_parameter("ALLDOFCOUPLED", true, "all dof (incl. pressure) are coupled", fdyn);

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

    Core::Utils::string_to_integral_parameter<Inpar::FLUID::CalcError>(
        "CALCERROR", "no", "Flag to (de)activate error calculations", name, label, fdyn);
  }
  Core::Utils::int_parameter("CALCERRORFUNCNO", -1, "Function for Error Calculation", fdyn);

  Core::Utils::int_parameter("CORRTERMFUNCNO", -1,
      "Function for calculation of the correction term for the weakly compressible problem", fdyn);

  Core::Utils::int_parameter("BODYFORCEFUNCNO", -1,
      "Function for calculation of the body force for the weakly compressible problem", fdyn);

  Core::Utils::double_parameter("STAB_DEN_REF", 0.0,
      "Reference stabilization parameter for the density for the HDG weakly compressible "
      "formulation",
      fdyn);

  Core::Utils::double_parameter("STAB_MOM_REF", 0.0,
      "Reference stabilization parameter for the momentum for the HDG weakly compressible "
      "formulation",
      fdyn);

  Core::Utils::int_parameter("VARVISCFUNCNO", -1,
      "Function for calculation of a variable viscosity for the weakly compressible problem", fdyn);

  Core::Utils::bool_parameter("PRESSAVGBC", false,
      "Flag to (de)activate imposition of boundary condition for the considered element average "
      "pressure",
      fdyn);

  Core::Utils::double_parameter("REFMACH", 1.0, "Reference Mach number", fdyn);

  Core::Utils::bool_parameter("BLOCKMATRIX", false,
      "Indicates if system matrix should be assembled into a sparse block matrix type.", fdyn);

  Core::Utils::bool_parameter("ADAPTCONV", false,
      "Switch on adaptive control of linear solver tolerance for nonlinear solution", fdyn);
  Core::Utils::double_parameter("ADAPTCONV_BETTER", 0.1,
      "The linear solver shall be this much better than the current nonlinear residual in the "
      "nonlinear convergence limit",
      fdyn);

  Core::Utils::bool_parameter(
      "INFNORMSCALING", false, "Scale blocks of matrix with row infnorm?", fdyn);

  Core::Utils::bool_parameter("GMSH_OUTPUT", false, "write output to gmsh files", fdyn);
  Core::Utils::bool_parameter(
      "COMPUTE_DIVU", false, "Compute divergence of velocity field at the element center", fdyn);
  Core::Utils::bool_parameter("COMPUTE_EKIN", false,
      "Compute kinetic energy at the end of each time step and write it to file.", fdyn);
  Core::Utils::bool_parameter("NEW_OST", false,
      "Solve the Navier-Stokes equation with the new One Step Theta algorithm",
      fdyn);  // TODO: To be removed.
  Core::Utils::int_parameter("RESULTSEVERY", 1, "Increment for writing solution", fdyn);
  Core::Utils::int_parameter("RESTARTEVERY", 20, "Increment for writing restart", fdyn);
  Core::Utils::int_parameter("NUMSTEP", 1, "Total number of Timesteps", fdyn);
  Core::Utils::int_parameter("STEADYSTEP", -1, "steady state check every step", fdyn);
  Core::Utils::int_parameter("NUMSTASTEPS", 0, "Number of Steps for Starting Scheme", fdyn);
  Core::Utils::int_parameter("STARTFUNCNO", -1, "Function for Initial Starting Field", fdyn);
  Core::Utils::int_parameter("ITEMAX", 10, "max. number of nonlin. iterations", fdyn);
  Core::Utils::int_parameter("INITSTATITEMAX", 5,
      "max number of nonlinear iterations for initial stationary solution", fdyn);
  Core::Utils::double_parameter("TIMESTEP", 0.01, "Time increment dt", fdyn);
  Core::Utils::double_parameter("MAXTIME", 1000.0, "Total simulation time", fdyn);
  Core::Utils::double_parameter("ALPHA_M", 1.0, "Time integration factor", fdyn);
  Core::Utils::double_parameter("ALPHA_F", 1.0, "Time integration factor", fdyn);
  Core::Utils::double_parameter("GAMMA", 1.0, "Time integration factor", fdyn);
  Core::Utils::double_parameter("THETA", 0.66, "Time integration factor", fdyn);

  Core::Utils::double_parameter(
      "START_THETA", 1.0, "Time integration factor for starting scheme", fdyn);

  Core::Utils::bool_parameter("STRONG_REDD_3D_COUPLING_TYPE", false,
      "Flag to (de)activate potential Strong 3D redD coupling", fdyn);

  Core::Utils::int_parameter(
      "VELGRAD_PROJ_SOLVER", -1, "Number of linear solver used for L2 projection", fdyn);

  Core::Utils::string_to_integral_parameter<Inpar::FLUID::GradientReconstructionMethod>(
      "VELGRAD_PROJ_METHOD", "none", "Flag to (de)activate gradient reconstruction.",
      tuple<std::string>("none", "superconvergent_patch_recovery", "L2_projection"),
      tuple<Inpar::FLUID::GradientReconstructionMethod>(
          gradreco_none,  // no convective streamline edge-based stabilization
          gradreco_spr,   // convective streamline edge-based stabilization on the entire domain
          gradreco_l2     // pressure edge-based stabilization as ghost penalty around cut elements
          ),
      fdyn);

  Core::Utils::bool_parameter("OFF_PROC_ASSEMBLY", false,
      "Do not evaluate ghosted elements but communicate them --> faster if element call is "
      "expensive",
      fdyn);

  fdyn.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs fdyn_nln{fdyn, "NONLINEAR SOLVER TOLERANCES"};

  Core::Utils::double_parameter(
      "TOL_VEL_RES", 1e-6, "Tolerance for convergence check of velocity residual", fdyn_nln);

  Core::Utils::double_parameter(
      "TOL_VEL_INC", 1e-6, "Tolerance for convergence check of velocity increment", fdyn_nln);

  Core::Utils::double_parameter(
      "TOL_PRES_RES", 1e-6, "Tolerance for convergence check of pressure residual", fdyn_nln);

  Core::Utils::double_parameter(
      "TOL_PRES_INC", 1e-6, "Tolerance for convergence check of pressure increment", fdyn_nln);

  fdyn_nln.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs fdyn_stab{fdyn, "RESIDUAL-BASED STABILIZATION"};
  // this parameter defines various stabilized methods
  Core::Utils::string_to_integral_parameter<Inpar::FLUID::StabType>("STABTYPE", "residual_based",
      "Apply (un)stabilized fluid formulation. No stabilization is only possible for inf-sup "
      "stable elements."
      "Use a residual-based stabilization or, more generally, a stabilization \nbased on the "
      "concept of the residual-based variational multiscale method...\nExpecting additional "
      "input. "
      "Use an edge-based stabilization, especially for XFEM. "
      "Alternative: Element/cell based polynomial pressure projection, see Dohrmann/Bochev 2004, "
      "IJNMF",
      tuple<std::string>("no_stabilization", "residual_based", "edge_based", "pressure_projection"),
      tuple<Inpar::FLUID::StabType>(
          stabtype_nostab, stabtype_residualbased, stabtype_edgebased, stabtype_pressureprojection),
      fdyn_stab);

  Core::Utils::bool_parameter("INCONSISTENT", false,
      "residual based without second derivatives (i.e. only consistent for tau->0, but faster)",
      fdyn_stab);

  Core::Utils::bool_parameter("Reconstruct_Sec_Der", false,
      "residual computed with a reconstruction of the second derivatives via projection or "
      "superconvergent patch recovery",
      fdyn_stab);

  // the following parameters are necessary only if a residual based stabilized method is applied
  Core::Utils::string_to_integral_parameter<SubscalesTD>("TDS", "quasistatic",
      "Flag to allow time dependency of subscales for residual-based stabilization.",
      tuple<std::string>("quasistatic", "time_dependent"),
      tuple<SubscalesTD>(subscales_quasistatic, subscales_time_dependent), fdyn_stab);

  Core::Utils::string_to_integral_parameter<Transient>("TRANSIENT", "no_transient",
      "Specify how to treat the transient term. "
      "Use transient term (recommended for time dependent subscales) or "
      "use transient term including a linearisation of 1/tau",
      tuple<std::string>("no_transient", "yes_transient", "transient_complete"),
      tuple<Transient>(inertia_stab_drop, inertia_stab_keep, inertia_stab_keep_complete),
      fdyn_stab);

  Core::Utils::bool_parameter("PSPG", true, "Flag to (de)activate PSPG stabilization.", fdyn_stab);
  Core::Utils::bool_parameter("SUPG", true, "Flag to (de)activate SUPG stabilization.", fdyn_stab);
  Core::Utils::bool_parameter("GRAD_DIV", true, "Flag to (de)activate grad-div term.", fdyn_stab);

  Core::Utils::string_to_integral_parameter<VStab>("VSTAB", "no_vstab",
      "Flag to (de)activate viscous term in residual-based stabilization. Options: "
      "No viscous term in stabilization, or, "
      "Viscous stabilization of GLS type, or, "
      "Viscous stabilization of GLS type, included only on the right hand side, or, "
      "Viscous stabilization of USFEM type, or, "
      "Viscous stabilization of USFEM type, included only on the right hand side",
      tuple<std::string>(
          "no_vstab", "vstab_gls", "vstab_gls_rhs", "vstab_usfem", "vstab_usfem_rhs"),
      tuple<VStab>(viscous_stab_none, viscous_stab_gls, viscous_stab_gls_only_rhs,
          viscous_stab_usfem, viscous_stab_usfem_only_rhs),
      fdyn_stab);

  Core::Utils::string_to_integral_parameter<RStab>("RSTAB", "no_rstab",
      "Flag to (de)activate reactive term in residual-based stabilization.",
      tuple<std::string>("no_rstab", "rstab_gls", "rstab_usfem"),
      tuple<RStab>(reactive_stab_none, reactive_stab_gls, reactive_stab_usfem), fdyn_stab);

  Core::Utils::string_to_integral_parameter<CrossStress>("CROSS-STRESS", "no_cross",
      "Flag to (de)activate cross-stress term -> residual-based VMM. Options:"
      "No cross-stress term, or,"
      "Include the cross-stress term with a linearization of the convective part, or, "
      "Include cross-stress term, but only explicitly on right hand side",
      tuple<std::string>("no_cross", "yes_cross", "cross_rhs"),
      tuple<CrossStress>(cross_stress_stab_none, cross_stress_stab, cross_stress_stab_only_rhs),
      fdyn_stab);

  Core::Utils::string_to_integral_parameter<ReynoldsStress>("REYNOLDS-STRESS", "no_reynolds",
      "Flag to (de)activate Reynolds-stress term -> residual-based VMM.",
      tuple<std::string>("no_reynolds", "yes_reynolds", "reynolds_rhs"),
      tuple<ReynoldsStress>(
          reynolds_stress_stab_none, reynolds_stress_stab, reynolds_stress_stab_only_rhs),
      fdyn_stab);

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

    Core::Utils::string_to_integral_parameter<TauType>("DEFINITION_TAU",
        "Franca_Barrenechea_Valentin_Frey_Wall", "Definition of tau_M and Tau_C", name, label,
        fdyn_stab);
  }

  // this parameter selects the characteristic element length for tau_Mu for all
  // stabilization parameter definitions requiring such a length
  Core::Utils::string_to_integral_parameter<CharEleLengthU>("CHARELELENGTH_U", "streamlength",
      "Characteristic element length for tau_Mu",
      tuple<std::string>("streamlength", "volume_equivalent_diameter", "root_of_volume"),
      tuple<CharEleLengthU>(streamlength_u, volume_equivalent_diameter_u, root_of_volume_u),
      fdyn_stab);

  // this parameter selects the characteristic element length for tau_Mp and tau_C for
  // all stabilization parameter definitions requiring such a length
  Core::Utils::string_to_integral_parameter<CharEleLengthPC>("CHARELELENGTH_PC",
      "volume_equivalent_diameter", "Characteristic element length for tau_Mp/tau_C",
      tuple<std::string>("streamlength", "volume_equivalent_diameter", "root_of_volume"),
      tuple<CharEleLengthPC>(streamlength_pc, volume_equivalent_diameter_pc, root_of_volume_pc),
      fdyn_stab);

  // this parameter selects the location where tau is evaluated

  std::vector<std::string> evaluation_tau_valid_input = {"element_center", "integration_point"};
  Core::Utils::string_parameter("EVALUATION_TAU", "element_center",
      "Location where tau is evaluated", fdyn_stab, evaluation_tau_valid_input);

  // this parameter selects the location where the material law is evaluated
  // (does not fit here very well, but parameter transfer is easier)

  std::vector<std::string> evaluation_mat_valid_input = {"element_center", "integration_point"};
  Core::Utils::string_parameter("EVALUATION_MAT", "element_center",
      "Location where material law is evaluated", fdyn_stab, evaluation_mat_valid_input);

  // these parameters active additional terms in loma continuity equation
  // which might be identified as SUPG-/cross- and Reynolds-stress term
  Core::Utils::bool_parameter("LOMA_CONTI_SUPG", false,
      "Flag to (de)activate SUPG stabilization in loma continuity equation.", fdyn_stab);

  Core::Utils::string_to_integral_parameter<CrossStress>("LOMA_CONTI_CROSS_STRESS", "no_cross",
      "Flag to (de)activate cross-stress term loma continuity equation-> residual-based VMM.",
      tuple<std::string>("no_cross", "yes_cross", "cross_rhs"),
      tuple<CrossStress>(cross_stress_stab_none, cross_stress_stab, cross_stress_stab_only_rhs),
      fdyn_stab);

  Core::Utils::string_to_integral_parameter<ReynoldsStress>("LOMA_CONTI_REYNOLDS_STRESS",
      "no_reynolds",
      "Flag to (de)activate Reynolds-stress term loma continuity equation-> residual-based VMM.",
      tuple<std::string>("no_reynolds", "yes_reynolds", "reynolds_rhs"),
      tuple<ReynoldsStress>(
          reynolds_stress_stab_none, reynolds_stress_stab, reynolds_stress_stab_only_rhs),
      fdyn_stab);

  fdyn_stab.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs fdyn_edge_based_stab{fdyn, "EDGE-BASED STABILIZATION"};

  //! Flag to (de)activate edge-based (EOS) pressure stabilization
  Core::Utils::string_to_integral_parameter<EosPres>("EOS_PRES", "none",
      "Flag to (de)activate pressure edge-based stabilization. Options: "
      "do not use pressure edge-based stabilization, or, "
      "use pressure edge-based stabilization as standard edge-based stabilization on the "
      "entire domain, or, "
      "use pressure edge-based stabilization as xfem ghost-penalty stabilization just around "
      "cut elements",
      tuple<std::string>("none", "std_eos", "xfem_gp"),
      tuple<EosPres>(EOS_PRES_none,  // no pressure edge-based stabilization
          EOS_PRES_std_eos,          // pressure edge-based stabilization on the entire domain
          EOS_PRES_xfem_gp  // pressure edge-based stabilization as ghost penalty around cut
                            // elements
          ),
      fdyn_edge_based_stab);

  //! Flag to (de)activate edge-based (EOS) convective streamline stabilization
  Core::Utils::string_to_integral_parameter<EosConvStream>("EOS_CONV_STREAM", "none",
      "Flag to (de)activate convective streamline edge-based stabilization. Options: "
      "do not use convective streamline edge-based stabilization, or, "
      "use convective streamline edge-based stabilization as standard edge-based stabilization "
      "on the entire domain, or, "
      "use convective streamline edge-based stabilization as xfem ghost-penalty stabilization "
      "just around cut elements",
      tuple<std::string>("none", "std_eos", "xfem_gp"),
      tuple<EosConvStream>(
          EOS_CONV_STREAM_none,     // no convective streamline edge-based stabilization
          EOS_CONV_STREAM_std_eos,  // convective streamline edge-based stabilization on the entire
                                    // domain
          EOS_CONV_STREAM_xfem_gp   // pressure edge-based stabilization as ghost penalty around cut
                                    // elements
          ),
      fdyn_edge_based_stab);

  //! Flag to (de)activate edge-based (EOS) convective crosswind stabilization
  Core::Utils::string_to_integral_parameter<EosConvCross>("EOS_CONV_CROSS", "none",
      "Flag to (de)activate convective crosswind edge-based stabilization. Options:"
      "do not use convective crosswind edge-based stabilization, or, "
      "use convective crosswind edge-based stabilization as standard edge-based stabilization "
      "on the entire domain, or, "
      "use convective crosswind edge-based stabilization as xfem ghost-penalty stabilization "
      "just around cut elements",
      tuple<std::string>("none", "std_eos", "xfem_gp"),
      tuple<EosConvCross>(EOS_CONV_CROSS_none,  // no convective crosswind edge-based stabilization
          EOS_CONV_CROSS_std_eos,  // convective crosswind edge-based stabilization on the entire
                                   // domain
          EOS_CONV_CROSS_xfem_gp   // convective crosswind edge-based stabilization as ghost penalty
                                   // around cut elements
          ),
      fdyn_edge_based_stab);

  //! Flag to (de)activate edge-based (EOS) divergence stabilization
  Core::Utils::string_to_integral_parameter<EosDiv>("EOS_DIV", "none",
      "Flag to (de)activate divergence edge-based stabilization. Options: "
      "do not use divergence edge-based stabilization, or, "
      "divergence edge-based stabilization based on velocity jump on the entire domain, or, "
      "divergence edge-based stabilization based on divergence jump just around cut elements, or, "
      "divergence edge-based stabilization based on velocity jump on the entire domain, or, "
      "divergence edge-based stabilization based on divergence jump just around cut elements",
      tuple<std::string>(
          "none", "vel_jump_std_eos", "vel_jump_xfem_gp", "div_jump_std_eos", "div_jump_xfem_gp"),
      tuple<EosDiv>(EOS_DIV_none,    // no convective edge-based stabilization
          EOS_DIV_vel_jump_std_eos,  // streamline convective edge-based stabilization
          EOS_DIV_vel_jump_xfem_gp,  // streamline convective edge-based stabilization
          EOS_DIV_div_jump_std_eos,  // crosswind convective edge-based stabilization
          EOS_DIV_div_jump_xfem_gp   // crosswind convective edge-based stabilization
          ),
      fdyn_edge_based_stab);

  //! special least-squares condition for pseudo 2D examples where pressure level is determined via
  //! Krylov-projection
  Core::Utils::bool_parameter("PRES_KRYLOV_2Dz", false,
      "residual based without second derivatives (i.e. only consistent for tau->0, but faster)",
      fdyn_edge_based_stab);

  //! this parameter selects the definition of Edge-based stabilization parameter
  Core::Utils::string_to_integral_parameter<EosTauType>("EOS_DEFINITION_TAU",
      "Burman_Hansbo_DAngelo_Zunino",
      "Definition of stabilization parameter for edge-based stabilization",
      tuple<std::string>("Burman_Fernandez_Hansbo", "Burman_Fernandez_Hansbo_wo_dt",
          "Braack_Burman_John_Lube", "Braack_Burman_John_Lube_wo_divjump",
          "Franca_Barrenechea_Valentin_Wall", "Burman_Fernandez", "Burman_Hansbo_DAngelo_Zunino",
          "Burman_Hansbo_DAngelo_Zunino_wo_dt", "Schott_Massing_Burman_DAngelo_Zunino",
          "Schott_Massing_Burman_DAngelo_Zunino_wo_dt", "Burman",
          "Taylor_Hughes_Zarins_Whiting_Jansen_Codina_scaling", "tau_not_defined"),
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
      fdyn_edge_based_stab);

  //! this parameter selects how the element length of Edge-based stabilization is defined
  Core::Utils::string_to_integral_parameter<EosElementLength>("EOS_H_DEFINITION",
      "EOS_he_max_diameter_to_opp_surf",
      "Definition of element length for edge-based stabilization. Options:"
      "take the maximal (nsd-1)D diameter of faces that connect the internal "
      "face to its opposite faces, or, "
      "take the maximal 1D distance along 1D edge to opposite surface for both parent elements, "
      "or, "
      "take the maximal (nsd-1)D face diameter of all faces for both parent elements, or, "
      "maximal nD diameter of the neighboring elements, or, "
      "maximal (n-1)D diameter of the internal face/edge, or, "
      "take the maximal volume equivalent diameter of adjacent elements",
      tuple<std::string>("EOS_he_max_diameter_to_opp_surf", "EOS_he_max_dist_to_opp_surf",
          "EOS_he_surf_with_max_diameter", "EOS_hk_max_diameter", "EOS_he_surf_diameter",
          "EOS_he_vol_eq_diameter"),
      tuple<EosElementLength>(EOS_he_max_diameter_to_opp_surf, EOS_he_max_dist_to_opp_surf,
          EOS_he_surf_with_max_diameter, EOS_hk_max_diameter, EOS_he_surf_diameter,
          EOS_he_vol_eq_diameter),
      fdyn_edge_based_stab);

  fdyn_edge_based_stab.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs fdyn_porostab{fdyn, "POROUS-FLOW STABILIZATION"};

  Core::Utils::bool_parameter(
      "STAB_BIOT", false, "Flag to (de)activate BIOT stabilization.", fdyn_porostab);
  Core::Utils::double_parameter("STAB_BIOT_SCALING", 1.0,
      "Scaling factor for stabilization parameter for biot stabilization of porous flow.",
      fdyn_porostab);

  // this parameter defines various stabilized methods
  Core::Utils::string_to_integral_parameter<Inpar::FLUID::StabType>("STABTYPE", "residual_based",
      "Apply (un)stabilized fluid formulation. No stabilization is only possible for inf-sup "
      "stable elements. "
      "Use a residual-based stabilization or, more generally, a stabilization \nbased on the "
      "concept of the residual-based variational multiscale method...\nExpecting additional "
      "input"
      "Use an edge-based stabilization, especially for XFEM",
      tuple<std::string>("no_stabilization", "residual_based", "edge_based"),
      tuple<Inpar::FLUID::StabType>(stabtype_nostab, stabtype_residualbased, stabtype_edgebased),
      fdyn_porostab);

  Core::Utils::bool_parameter("INCONSISTENT", false,
      "residual based without second derivatives (i.e. only consistent for tau->0, but faster)",
      fdyn_porostab);

  Core::Utils::bool_parameter("Reconstruct_Sec_Der", false,
      "residual computed with a reconstruction of the second derivatives via projection or "
      "superconvergent patch recovery",
      fdyn_porostab);

  // the following parameters are necessary only if a residual based stabilized method is applied
  Core::Utils::string_to_integral_parameter<SubscalesTD>("TDS", "quasistatic",
      "Flag to allow time dependency of subscales for residual-based stabilization.",
      tuple<std::string>("quasistatic", "time_dependent"),
      tuple<SubscalesTD>(subscales_quasistatic, subscales_time_dependent), fdyn_porostab);

  Core::Utils::string_to_integral_parameter<Transient>("TRANSIENT", "no_transient",
      "Specify how to treat the transient term.",
      tuple<std::string>("no_transient", "yes_transient", "transient_complete"),
      tuple<Transient>(inertia_stab_drop, inertia_stab_keep, inertia_stab_keep_complete),
      fdyn_porostab);

  Core::Utils::bool_parameter(
      "PSPG", true, "Flag to (de)activate PSPG stabilization.", fdyn_porostab);
  Core::Utils::bool_parameter(
      "SUPG", true, "Flag to (de)activate SUPG stabilization.", fdyn_porostab);
  Core::Utils::bool_parameter(
      "GRAD_DIV", true, "Flag to (de)activate grad-div term.", fdyn_porostab);

  Core::Utils::string_to_integral_parameter<VStab>("VSTAB", "no_vstab",
      "Flag to (de)activate viscous term in residual-based stabilization.",
      tuple<std::string>(
          "no_vstab", "vstab_gls", "vstab_gls_rhs", "vstab_usfem", "vstab_usfem_rhs"),
      tuple<VStab>(viscous_stab_none, viscous_stab_gls, viscous_stab_gls_only_rhs,
          viscous_stab_usfem, viscous_stab_usfem_only_rhs),
      fdyn_porostab);

  Core::Utils::string_to_integral_parameter<RStab>("RSTAB", "no_rstab",
      "Flag to (de)activate reactive term in residual-based stabilization.",
      tuple<std::string>("no_rstab", "rstab_gls", "rstab_usfem"),
      tuple<RStab>(reactive_stab_none, reactive_stab_gls, reactive_stab_usfem), fdyn_porostab);

  Core::Utils::string_to_integral_parameter<CrossStress>("CROSS-STRESS", "no_cross",
      "Flag to (de)activate cross-stress term -> residual-based VMM.",
      tuple<std::string>("no_cross", "yes_cross", "cross_rhs"),
      tuple<CrossStress>(cross_stress_stab_none, cross_stress_stab, cross_stress_stab_only_rhs),
      fdyn_porostab);

  Core::Utils::string_to_integral_parameter<ReynoldsStress>("REYNOLDS-STRESS", "no_reynolds",
      "Flag to (de)activate Reynolds-stress term -> residual-based VMM.",
      tuple<std::string>("no_reynolds", "yes_reynolds", "reynolds_rhs"),
      tuple<ReynoldsStress>(
          reynolds_stress_stab_none, reynolds_stress_stab, reynolds_stress_stab_only_rhs),
      fdyn_porostab);

  // this parameter selects the tau definition applied
  Core::Utils::string_to_integral_parameter<TauType>("DEFINITION_TAU",
      "Franca_Barrenechea_Valentin_Frey_Wall", "Definition of tau_M and Tau_C",
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
      fdyn_porostab);

  // this parameter selects the characteristic element length for tau_Mu for all
  // stabilization parameter definitions requiring such a length
  Core::Utils::string_to_integral_parameter<CharEleLengthU>("CHARELELENGTH_U", "streamlength",
      "Characteristic element length for tau_Mu",
      tuple<std::string>("streamlength", "volume_equivalent_diameter", "root_of_volume"),
      tuple<CharEleLengthU>(streamlength_u, volume_equivalent_diameter_u, root_of_volume_u),
      fdyn_porostab);

  // this parameter selects the characteristic element length for tau_Mp and tau_C for
  // all stabilization parameter definitions requiring such a length
  Core::Utils::string_to_integral_parameter<CharEleLengthPC>("CHARELELENGTH_PC",
      "volume_equivalent_diameter", "Characteristic element length for tau_Mp/tau_C",
      tuple<std::string>("streamlength", "volume_equivalent_diameter", "root_of_volume"),
      tuple<CharEleLengthPC>(streamlength_pc, volume_equivalent_diameter_pc, root_of_volume_pc),
      fdyn_porostab);

  // this parameter selects the location where tau is evaluated
  evaluation_tau_valid_input = {"element_center", "integration_point"};
  Core::Utils::string_parameter("EVALUATION_TAU", "element_center",
      "Location where tau is evaluated", fdyn_porostab, evaluation_tau_valid_input);

  // this parameter selects the location where the material law is evaluated
  // (does not fit here very well, but parameter transfer is easier)
  evaluation_mat_valid_input = {"element_center", "integration_point"};
  Core::Utils::string_parameter("EVALUATION_MAT", "element_center",
      "Location where material law is evaluated", fdyn_porostab, evaluation_mat_valid_input);


  // these parameters active additional terms in loma continuity equation
  // which might be identified as SUPG-/cross- and Reynolds-stress term
  Core::Utils::bool_parameter("LOMA_CONTI_SUPG", false,
      "Flag to (de)activate SUPG stabilization in loma continuity equation.", fdyn_porostab);

  Core::Utils::string_to_integral_parameter<CrossStress>("LOMA_CONTI_CROSS_STRESS", "no_cross",
      "Flag to (de)activate cross-stress term loma continuity equation-> residual-based VMM.",
      tuple<std::string>("no_cross", "yes_cross", "cross_rhs"),
      tuple<CrossStress>(cross_stress_stab_none, cross_stress_stab, cross_stress_stab_only_rhs),
      fdyn_porostab);

  Core::Utils::string_to_integral_parameter<ReynoldsStress>("LOMA_CONTI_REYNOLDS_STRESS",
      "no_reynolds",
      "Flag to (de)activate Reynolds-stress term loma continuity equation-> residual-based VMM.",
      tuple<std::string>("no_reynolds", "yes_reynolds", "reynolds_rhs"),
      tuple<ReynoldsStress>(
          reynolds_stress_stab_none, reynolds_stress_stab, reynolds_stress_stab_only_rhs),
      fdyn_porostab);

  fdyn_porostab.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs fdyn_turbu{fdyn, "TURBULENCE MODEL"};

  //----------------------------------------------------------------------
  // modeling strategies
  //----------------------------------------------------------------------

  std::vector<std::string> turbulence_approach_valid_input = {"DNS_OR_RESVMM_LES", "CLASSICAL_LES"};
  std::string turbulence_approach_doc_string =
      "Try to solve flow as an underresolved DNS. Mind that your stabilisation already acts as a "
      "kind of turbulence model! Perform a classical Large Eddy Simulation adding addititional "
      "turbulent viscosity. This may be based on various physical models.)";

  Core::Utils::string_parameter("TURBULENCE_APPROACH", "DNS_OR_RESVMM_LES",
      turbulence_approach_doc_string, fdyn_turbu, turbulence_approach_valid_input);

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
  Core::Utils::string_parameter("PHYSICAL_MODEL", "no_model", physical_model_doc_string, fdyn_turbu,
      physical_model_valid_input);

  Core::Utils::string_to_integral_parameter<Inpar::FLUID::FineSubgridVisc>("FSSUGRVISC", "No",
      "fine-scale subgrid viscosity",
      tuple<std::string>("No", "Smagorinsky_all", "Smagorinsky_small"),
      tuple<Inpar::FLUID::FineSubgridVisc>(no_fssgv, smagorinsky_all, smagorinsky_small),
      fdyn_turbu);

  //----------------------------------------------------------------------
  // turbulence specific output and statistics
  //----------------------------------------------------------------------

  Core::Utils::int_parameter(
      "SAMPLING_START", 10000000, "Time step after when sampling shall be started", fdyn_turbu);
  Core::Utils::int_parameter(
      "SAMPLING_STOP", 1, "Time step when sampling shall be stopped", fdyn_turbu);
  Core::Utils::int_parameter("DUMPING_PERIOD", 1,
      "Period of time steps after which statistical data shall be dumped", fdyn_turbu);
  Core::Utils::bool_parameter("SUBGRID_DISSIPATION", false,
      "Flag to (de)activate estimation of subgrid-scale dissipation (only for seclected flows).",
      fdyn_turbu);
  Core::Utils::bool_parameter(
      "OUTMEAN", false, "Flag to (de)activate averaged paraview output", fdyn_turbu);
  Core::Utils::bool_parameter("TURBMODEL_LS", true,
      "Flag to (de)activate turbulence model in level-set equation", fdyn_turbu);

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
        "CANONICAL_FLOW", "no", canonical_flow_doc, fdyn_turbu, canonical_flow_valid_input);
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
      "HOMDIR", "not_specified", homdir_doc, fdyn_turbu, homdir_valid_input);

  //---------------------------------------
  // further problem-specific parameters

  // CHANNEL FLOW
  //--------------

  Core::Utils::double_parameter("CHAN_AMPL_INIT_DIST", 0.1,
      "Max. amplitude of the random disturbance in percent of the initial value in mean flow "
      "direction.",
      fdyn_turbu);

  Core::Utils::string_to_integral_parameter<ForcingType>("FORCING_TYPE",
      "linear_compensation_from_intermediate_spectrum", "forcing strategy",
      tuple<std::string>("linear_compensation_from_intermediate_spectrum", "fixed_power_input"),
      tuple<ForcingType>(linear_compensation_from_intermediate_spectrum, fixed_power_input),
      fdyn_turbu);

  Core::Utils::int_parameter(
      "CHA_NUMSUBDIVISIONS", 5, "Number of homogeneous sampling planes in element", fdyn_turbu);

  // HIT
  //--------------

  Core::Utils::int_parameter("FORCING_TIME_STEPS", 0,
      "Number of time steps during which forcing is applied. Decaying homogeneous isotropic "
      "turbulence only.",
      fdyn_turbu);

  Core::Utils::double_parameter("THRESHOLD_WAVENUMBER", 0.0,
      "Forcing is only applied to wave numbers lower or equal than the given threshold wave "
      "number.",
      fdyn_turbu);

  Core::Utils::double_parameter("POWER_INPUT", 0.0, "power of forcing", fdyn_turbu);

  std::vector<std::string> scalar_forcing_valid_input = {"no", "isotropic", "mean_scalar_gradient"};
  std::string scalar_forcing_doc =
      "no: Do not force the scalar field\n"
      "isotropic: Force scalar field isotropically such as the fluid field.\n"
      "mean_scalar_gradient: Force scalar field by imposed mean-scalar gradient.\n";
  Core::Utils::string_parameter(
      "SCALAR_FORCING", "no", scalar_forcing_doc, fdyn_turbu, scalar_forcing_valid_input);

  Core::Utils::double_parameter("MEAN_SCALAR_GRADIENT", 0.0,
      "Value of imposed mean-scalar gradient to force scalar field.", fdyn_turbu);

  // filtering with xfem
  //--------------

  Core::Utils::bool_parameter("EXCLUDE_XFEM", false,
      "Flag to (de)activate XFEM dofs in calculation of fine-scale velocity.", fdyn_turbu);

  fdyn_turbu.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  // sublist with additional input parameters for Smagorinsky model
  Core::Utils::SectionSpecs fdyn_turbsgv{fdyn, "SUBGRID VISCOSITY"};

  Core::Utils::double_parameter("C_SMAGORINSKY", 0.0,
      "Constant for the Smagorinsky model. Something between 0.1 to 0.24. Vreman constant if the "
      "constant vreman model is applied (something between 0.07 and 0.01).",
      fdyn_turbsgv);
  Core::Utils::double_parameter("C_YOSHIZAWA", -1.0,
      "Constant for the compressible Smagorinsky model: isotropic part of subgrid-stress tensor. "
      "About 0.09 or 0.0066. Ci will not be squared!",
      fdyn_turbsgv);
  Core::Utils::bool_parameter("C_SMAGORINSKY_AVERAGED", false,
      "Flag to (de)activate averaged Smagorinksy constant", fdyn_turbsgv);
  Core::Utils::bool_parameter(
      "C_INCLUDE_CI", false, "Flag to (de)inclusion of Yoshizawa model", fdyn_turbsgv);
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
      fdyn_turbsgv);

  Core::Utils::double_parameter("C_TURBPRANDTL", 1.0,
      "(Constant) turbulent Prandtl number for the Smagorinsky model in scalar transport.",
      fdyn_turbsgv);

  Core::Utils::string_to_integral_parameter<VremanFiMethod>("FILTER_WIDTH", "CubeRootVol",
      "The Vreman model requires a filter width.",
      tuple<std::string>("CubeRootVol", "Direction_dependent", "Minimum_length"),
      tuple<VremanFiMethod>(cuberootvol, dir_dep, min_len), fdyn_turbsgv);

  fdyn_turbsgv.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  // sublist with additional input parameters for Smagorinsky model
  Core::Utils::SectionSpecs fdyn_wallmodel{fdyn, "WALL MODEL"};

  Core::Utils::bool_parameter("X_WALL", false, "Flag to switch on the xwall model", fdyn_wallmodel);


  std::vector<std::string> tauw_type_valid_input = {"constant", "between_steps"};
  std::string tauw_type_doc =
      "constant: Use the constant wall shear stress given in the input file for the whole "
      "simulation.\n"
      "between_steps: Calculate wall shear stress in between time steps.\n";
  Core::Utils::string_parameter(
      "Tauw_Type", "constant", tauw_type_doc, fdyn_wallmodel, tauw_type_valid_input);

  std::vector<std::string> tauw_calc_type_valid_input = {
      "residual", "gradient", "gradient_to_residual"};
  std::string tauw_calc_type_doc =
      "residual: Residual (force) divided by area.\n"
      "gradient: Gradient via shape functions and nodal values.\n"
      "gradient_to_residual: First gradient, then residual.\n";
  Core::Utils::string_parameter(
      "Tauw_Calc_Type", "residual", tauw_calc_type_doc, fdyn_wallmodel, tauw_calc_type_valid_input);


  Core::Utils::int_parameter(
      "Switch_Step", -1, "Switch from gradient to residual based tauw.", fdyn_wallmodel);

  std::vector<std::string> projection_valid_input = {
      "No", "onlyl2projection", "l2projectionwithcontinuityconstraint"};
  std::string projection_doc =
      "Flag to switch projection of the enriched dofs after updating tauw, alternatively with or "
      "without continuity constraint.";
  Core::Utils::string_parameter(
      "Projection", "No", projection_doc, fdyn_wallmodel, projection_valid_input);

  Core::Utils::double_parameter("C_Tauw", 1.0,
      "Constant wall shear stress for Spalding's law, if applicable", fdyn_wallmodel);

  Core::Utils::double_parameter("Min_Tauw", 2.0e-9,
      "Minimum wall shear stress preventing system to become singular", fdyn_wallmodel);

  Core::Utils::double_parameter(
      "Inc_Tauw", 1.0, "Increment of Tauw of full step, between 0.0 and 1.0", fdyn_wallmodel);

  std::vector<std::string> blending_type_valid_input = {"none", "ramp_function"};
  std::string blending_type_doc =
      "Methods for blending the enrichment space.\n"
      "none: No ramp function, does not converge!\n"
      "ramp_function: Enrichment is multiplied with linear ramp function resulting in zero "
      "enrichment at the interface.\n";
  Core::Utils::string_parameter(
      "Blending_Type", "none", blending_type_doc, fdyn_wallmodel, blending_type_valid_input);


  Core::Utils::int_parameter(
      "GP_Wall_Normal", 3, "Gauss points in wall normal direction", fdyn_wallmodel);
  Core::Utils::int_parameter("GP_Wall_Normal_Off_Wall", 3,
      "Gauss points in wall normal direction, off-wall elements", fdyn_wallmodel);
  Core::Utils::int_parameter(
      "GP_Wall_Parallel", 3, "Gauss points in wall parallel direction", fdyn_wallmodel);

  Core::Utils::bool_parameter("Treat_Tauw_on_Dirichlet_Inflow", false,
      "Flag to treat residual on Dirichlet inflow nodes for calculation of wall shear stress",
      fdyn_wallmodel);

  Core::Utils::int_parameter(
      "PROJECTION_SOLVER", -1, "Set solver number for l2-projection.", fdyn_wallmodel);

  fdyn_wallmodel.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  // sublist with additional input parameters for multifractal subgrid-scales
  Core::Utils::SectionSpecs fdyn_turbmfs{fdyn, "MULTIFRACTAL SUBGRID SCALES"};

  Core::Utils::double_parameter(
      "CSGS", 0.0, "Modelparameter of multifractal subgrid-scales.", fdyn_turbmfs);

  std::vector<std::string> scale_separation_valid_input = {
      "no_scale_sep", "box_filter", "algebraic_multigrid_operator"};
  std::string scale_separation_doc =
      "Specify the filter type for scale separation in LES.\n"
      "no_scale_sep: no scale separation.\n"
      "box_filter: classical box filter.\n"
      "algebraic_multigrid_operator: scale separation by algebraic multigrid operator.\n";
  Core::Utils::string_parameter("SCALE_SEPARATION", "no_scale_sep", scale_separation_doc,
      fdyn_turbmfs, scale_separation_valid_input);


  Core::Utils::int_parameter("ML_SOLVER", -1,
      "Set solver number for scale separation via level set transfer operators from plain "
      "aggregation.",
      fdyn_turbmfs);

  Core::Utils::bool_parameter("CALC_N", false,
      "Flag to (de)activate calculation of N from the Reynolds number.", fdyn_turbmfs);

  Core::Utils::double_parameter("N", 1.0, "Set grid to viscous scale ratio.", fdyn_turbmfs);

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
      "REF_LENGTH", "cube_edge", ref_length_doc, fdyn_turbmfs, ref_length_valid_input);

  std::vector<std::string> ref_velocity_valid_input = {"strainrate", "resolved", "fine_scale"};
  std::string ref_velocity_doc =
      "Specify the reference velocity for Re-dependent N.\n"
      "strainrate: norm of strain rate.\n"
      "resolved: resolved velocity.\n"
      "fine_scale: fine-scale velocity.\n";
  Core::Utils::string_parameter(
      "REF_VELOCITY", "strainrate", ref_velocity_doc, fdyn_turbmfs, ref_velocity_valid_input);


  Core::Utils::double_parameter("C_NU", 1.0,
      "Proportionality constant between Re and ratio viscous scale to element length.",
      fdyn_turbmfs);

  Core::Utils::bool_parameter(
      "NEAR_WALL_LIMIT", false, "Flag to (de)activate near-wall limit.", fdyn_turbmfs);

  std::vector<std::string> evaluation_b_valid_input = {"element_center", "integration_point"};
  std::string evaluation_b_doc =
      "Location where B is evaluated\n"
      "element_center: evaluate B at element center.\n"
      "integration_point: evaluate B at integration point.\n";
  Core::Utils::string_parameter(
      "EVALUATION_B", "element_center", evaluation_b_doc, fdyn_turbmfs, evaluation_b_valid_input);


  Core::Utils::double_parameter(
      "BETA", 0.0, "Cross- and Reynolds-stress terms only on right-hand-side.", fdyn_turbmfs);

  convform_valid_input = {"convective", "conservative"};
  std::string convform_doc =
      "form of convective term\n"
      "convective: Use the convective form.\n"
      "conservative: Use the conservative form.\n";
  Core::Utils::string_parameter(
      "CONVFORM", "convective", convform_doc, fdyn_turbmfs, convform_valid_input);

  Core::Utils::double_parameter("CSGS_PHI", 0.0,
      "Modelparameter of multifractal subgrid-scales for scalar transport.", fdyn_turbmfs);

  Core::Utils::bool_parameter(
      "ADAPT_CSGS_PHI", false, "Flag to (de)activate adaption of CsgsD to CsgsB.", fdyn_turbmfs);

  Core::Utils::bool_parameter("NEAR_WALL_LIMIT_CSGS_PHI", false,
      "Flag to (de)activate near-wall limit for scalar field.", fdyn_turbmfs);

  Core::Utils::bool_parameter("CONSISTENT_FLUID_RESIDUAL", false,
      "Flag to (de)activate the consistency term for residual-based stabilization.", fdyn_turbmfs);

  Core::Utils::double_parameter("C_DIFF", 1.0,
      "Proportionality constant between Re*Pr and ratio dissipative scale to element length. "
      "Usually equal cnu.",
      fdyn_turbmfs);

  Core::Utils::bool_parameter("SET_FINE_SCALE_VEL", false,
      "Flag to set fine-scale velocity for parallel nightly tests.", fdyn_turbmfs);

  // activate cross- and Reynolds-stress terms in loma continuity equation
  Core::Utils::bool_parameter("LOMA_CONTI", false,
      "Flag to (de)activate cross- and Reynolds-stress terms in loma continuity equation.",
      fdyn_turbmfs);

  fdyn_turbmfs.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs fdyn_turbinf{fdyn, "TURBULENT INFLOW"};

  Core::Utils::bool_parameter("TURBULENTINFLOW", false,
      "Flag to (de)activate potential separate turbulent inflow section", fdyn_turbinf);

  Core::Utils::string_to_integral_parameter<InitialField>("INITIALINFLOWFIELD", "zero_field",
      "Initial field for inflow section",
      tuple<std::string>("zero_field", "field_by_function", "disturbed_field_from_function"),
      tuple<InitialField>(initfield_zero_field, initfield_field_by_function,
          initfield_disturbed_field_from_function),
      fdyn_turbinf);

  Core::Utils::int_parameter(
      "INFLOWFUNC", -1, "Function number for initial flow field in inflow section", fdyn_turbinf);

  Core::Utils::double_parameter("INFLOW_INIT_DIST", 0.1,
      "Max. amplitude of the random disturbance in percent of the initial value in mean flow "
      "direction.",
      fdyn_turbinf);

  Core::Utils::int_parameter("NUMINFLOWSTEP", 1,
      "Total number of time steps for development of turbulent flow", fdyn_turbinf);

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
      "CANONICAL_INFLOW", "no", canonical_inflow_doc, fdyn_turbinf, canonical_inflow_valid_input);


  Core::Utils::double_parameter("INFLOW_CHA_SIDE", 0.0,
      "Most right side of inflow channel. Necessary to define sampling domain.", fdyn_turbinf);

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
  Core::Utils::string_parameter(
      "INFLOW_HOMDIR", "not_specified", inflow_homdir_doc, fdyn_turbinf, inflow_homdir_valid_input);

  Core::Utils::int_parameter("INFLOW_SAMPLING_START", 10000000,
      "Time step after when sampling shall be started", fdyn_turbinf);
  Core::Utils::int_parameter(
      "INFLOW_SAMPLING_STOP", 1, "Time step when sampling shall be stopped", fdyn_turbinf);
  Core::Utils::int_parameter("INFLOW_DUMPING_PERIOD", 1,
      "Period of time steps after which statistical data shall be dumped", fdyn_turbinf);

  fdyn_turbinf.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  // sublist with additional input parameters for time adaptivity in fluid/ coupled problems
  Core::Utils::SectionSpecs fdyn_timintada{fdyn, "TIMEADAPTIVITY"};
  Core::Utils::string_to_integral_parameter<AdaptiveTimeStepEstimator>(
      "ADAPTIVE_TIME_STEP_ESTIMATOR", "none", "Method used to determine adaptive time step size.",
      tuple<std::string>("none", "cfl_number", "only_print_cfl_number"),
      tuple<AdaptiveTimeStepEstimator>(const_dt, cfl_number, only_print_cfl_number),
      fdyn_timintada);

  Core::Utils::double_parameter(
      "CFL_NUMBER", -1.0, "CFL number for adaptive time step", fdyn_timintada);
  Core::Utils::int_parameter("FREEZE_ADAPTIVE_DT_AT", 1000000,
      "keep time step constant after this step, otherwise turbulence statistics sampling is not "
      "consistent",
      fdyn_timintada);
  Core::Utils::double_parameter(
      "ADAPTIVE_DT_INC", 0.8, "Increment of whole step for adaptive dt via CFL", fdyn_timintada);

  fdyn_timintada.move_into_collection(list);
}



void Inpar::LowMach::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;

  Core::Utils::SectionSpecs lomacontrol{"LOMA CONTROL"};

  Core::Utils::bool_parameter("MONOLITHIC", false, "monolithic solver", lomacontrol);
  Core::Utils::int_parameter("NUMSTEP", 24, "Total number of time steps", lomacontrol);
  Core::Utils::double_parameter("TIMESTEP", 0.1, "Time increment dt", lomacontrol);
  Core::Utils::double_parameter("MAXTIME", 1000.0, "Total simulation time", lomacontrol);
  Core::Utils::int_parameter("ITEMAX", 10, "Maximum number of outer iterations", lomacontrol);
  Core::Utils::int_parameter("ITEMAX_BEFORE_SAMPLING", 1,
      "Maximum number of outer iterations before sampling (for turbulent flows only)", lomacontrol);
  Core::Utils::double_parameter("CONVTOL", 1e-6, "Tolerance for convergence check", lomacontrol);
  Core::Utils::int_parameter("RESULTSEVERY", 1, "Increment for writing solution", lomacontrol);
  Core::Utils::int_parameter("RESTARTEVERY", 1, "Increment for writing restart", lomacontrol);

  std::vector<std::string> constthermpress_valid_input = {"No_energy", "No_mass", "Yes"};
  Core::Utils::string_parameter("CONSTHERMPRESS", "Yes",
      "treatment of thermodynamic pressure in time", lomacontrol, constthermpress_valid_input);

  Core::Utils::bool_parameter("SGS_MATERIAL_UPDATE", false,
      "update material by adding subgrid-scale scalar field", lomacontrol);

  // number of linear solver used for LOMA solver
  Core::Utils::int_parameter(
      "LINEAR_SOLVER", -1, "number of linear solver used for LOMA problem", lomacontrol);

  lomacontrol.move_into_collection(list);
}


void Inpar::FLUID::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // transfer boundary condition for turbulent inflow

  Core::Conditions::ConditionDefinition tbc_turb_inflow("DESIGN SURF TURBULENT INFLOW TRANSFER",
      "TransferTurbulentInflow", "TransferTurbulentInflow",
      Core::Conditions::TransferTurbulentInflow, true, Core::Conditions::geometry_type_surface);

  tbc_turb_inflow.add_component(parameter<int>("ID", {.description = ""}));
  tbc_turb_inflow.add_component(
      selection<std::string>("toggle", {"master", "slave"}, {.description = "toggle"}));
  tbc_turb_inflow.add_component(selection<int>("DIRECTION", {{"x", 0}, {"y", 1}, {"z", 2}},
      {.description = "transfer direction", .default_value = 0}));
  tbc_turb_inflow.add_component(parameter<Noneable<int>>(
      "curve", {.description = "curve id", .default_value = Core::IO::none<int>}));

  condlist.push_back(tbc_turb_inflow);

  /*--------------------------------------------------------------------*/
  // separate domain for turbulent inflow generation

  Core::Conditions::ConditionDefinition turbulentinflowgeneration("FLUID TURBULENT INFLOW VOLUME",
      "TurbulentInflowSection", "TurbulentInflowSection", Core::Conditions::TurbulentInflowSection,
      true, Core::Conditions::geometry_type_volume);

  condlist.push_back(turbulentinflowgeneration);


  /*--------------------------------------------------------------------*/
  // flow-dependent pressure conditions

  Core::Conditions::ConditionDefinition surfflowdeppressure(
      "DESIGN SURFACE FLOW-DEPENDENT PRESSURE CONDITIONS", "SurfaceFlowDepPressure",
      "SurfaceFlowDepPressure", Core::Conditions::SurfaceFlowDepPressure, true,
      Core::Conditions::geometry_type_surface);

  // flow-dependent pressure conditions can be imposed either based on
  // (out)flow rate or (out)flow volume (e.g., for air-cushion condition)
  surfflowdeppressure.add_component(selection<std::string>("TYPE_OF_FLOW_DEPENDENCE",
      {"flow_rate", "flow_volume", "fixed_pressure"}, {.description = "type of flow dependence"}));
  surfflowdeppressure.add_component(parameter<double>("ConstCoeff",
      {.description = "constant coefficient for (linear) flow-rate-based condition"}));
  surfflowdeppressure.add_component(parameter<double>(
      "LinCoeff", {.description = "linear coefficient for (linear) flow-rate-based condition"}));
  surfflowdeppressure.add_component(parameter<double>(
      "InitialVolume", {.description = "initial (air-cushion) volume outside of boundary"}));
  surfflowdeppressure.add_component(parameter<double>(
      "ReferencePressure", {.description = " reference pressure outside of boundary"}));
  surfflowdeppressure.add_component(
      parameter<double>("AdiabaticExponent", {.description = "adiabatic exponent"}));
  surfflowdeppressure.add_component(parameter<Noneable<int>>(
      "curve", {.description = "curve id", .default_value = Core::IO::none<int>}));

  condlist.emplace_back(surfflowdeppressure);


  /*--------------------------------------------------------------------*/
  // Slip Supplemental Curved Boundary conditions

  Core::Conditions::ConditionDefinition lineslipsupp(
      "DESIGN LINE SLIP SUPPLEMENTAL CURVED BOUNDARY CONDITIONS", "LineSlipSupp", "LineSlipSupp",
      Core::Conditions::LineSlipSupp, true, Core::Conditions::geometry_type_line);

  Core::Conditions::ConditionDefinition surfslipsupp(
      "DESIGN SURFACE SLIP SUPPLEMENTAL CURVED BOUNDARY CONDITIONS", "SurfaceSlipSupp",
      "SurfaceSlipSupp", Core::Conditions::SurfaceSlipSupp, true,
      Core::Conditions::geometry_type_surface);

  const auto make_slip_supp = [&](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(parameter<double>("USEUPDATEDNODEPOS"));
    condlist.emplace_back(cond);
  };

  make_slip_supp(lineslipsupp);
  make_slip_supp(surfslipsupp);

  /*--------------------------------------------------------------------*/
  // Navier-slip boundary conditions

  Core::Conditions::ConditionDefinition linenavierslip(
      "DESIGN LINE NAVIER-SLIP BOUNDARY CONDITIONS", "LineNavierSlip", "LineNavierSlip",
      Core::Conditions::LineNavierSlip, true, Core::Conditions::geometry_type_line);

  Core::Conditions::ConditionDefinition surfnavierslip(
      "DESIGN SURF NAVIER-SLIP BOUNDARY CONDITIONS", "SurfNavierSlip", "SurfNavierSlip",
      Core::Conditions::SurfNavierSlip, true, Core::Conditions::geometry_type_surface);

  const auto make_navierslip = [&](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(parameter<double>("SLIPCOEFFICIENT"));
    condlist.emplace_back(cond);
  };

  make_navierslip(linenavierslip);
  make_navierslip(surfnavierslip);

  /*--------------------------------------------------------------------*/
  // consistent outflow bcs for conservative element formulations

  Core::Conditions::ConditionDefinition surfconsistentoutflowconsistency(
      "DESIGN SURFACE CONSERVATIVE OUTFLOW CONSISTENCY", "SurfaceConservativeOutflowConsistency",
      "SurfaceConservativeOutflowConsistency",
      Core::Conditions::SurfaceConservativeOutflowConsistency, true,
      Core::Conditions::geometry_type_surface);

  condlist.push_back(surfconsistentoutflowconsistency);

  /*--------------------------------------------------------------------*/
  // Neumann condition for fluid that can handle inflow/backflow

  Core::Conditions::ConditionDefinition linefluidneumanninflow(
      "FLUID NEUMANN INFLOW LINE CONDITIONS", "FluidNeumannInflow", "Line Fluid Neumann Inflow",
      Core::Conditions::FluidNeumannInflow, true, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surffluidneumanninflow(
      "FLUID NEUMANN INFLOW SURF CONDITIONS", "FluidNeumannInflow", "Surface Fluid Neumann Inflow",
      Core::Conditions::FluidNeumannInflow, true, Core::Conditions::geometry_type_surface);

  condlist.push_back(linefluidneumanninflow);
  condlist.push_back(surffluidneumanninflow);

  /*--------------------------------------------------------------------*/
  // mixed/hybrid Dirichlet conditions

  Core::Conditions::ConditionDefinition linemixhybDirichlet(
      "DESIGN LINE MIXED/HYBRID DIRICHLET CONDITIONS", "LineMixHybDirichlet", "LineMixHybDirichlet",
      Core::Conditions::LineMixHybDirichlet, true, Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surfmixhybDirichlet(
      "DESIGN SURFACE MIXED/HYBRID DIRICHLET CONDITIONS", "SurfaceMixHybDirichlet",
      "SurfaceMixHybDirichlet", Core::Conditions::SurfaceMixHybDirichlet, true,
      Core::Conditions::geometry_type_surface);

  // we attach all the components of this condition to this condition
  const auto make_mixhybDirichlet = [&](Core::Conditions::ConditionDefinition& cond)
  {
    // we provide a vector of 3 values for velocities
    cond.add_component(
        parameter<std::vector<double>>("val", {.description = "velocity", .size = 3}));

    // and optional spatial functions
    cond.add_component(parameter<std::vector<Noneable<int>>>(
        "funct", {.description = "spatial function",
                     .default_value = std::vector(3, Core::IO::none<int>),
                     .size = 3}));

    // characteristic velocity
    cond.add_component(parameter<double>("u_C"));

    // the penalty parameter could be computed dynamically (using Spaldings
    // law of the wall) or using a fixed value (1)
    cond.add_component(selection<std::string>(
        "PENTYPE", {"constant", "Spalding"}, {.description = "how penalty parameter is computed"}));

    // scaling factor for penalty parameter tauB
    cond.add_component(parameter<double>("hB_divided_by"));

    // if Spaldings law is used, this defines the way how the traction at y is computed from utau
    cond.add_component(selection<std::string>("utau_computation", {"at_wall", "viscous_tangent"},
        {.description = "how traction at y is computed from utau"}));

    // we append it to the list of all conditions
    condlist.push_back(cond);
  };

  make_mixhybDirichlet(linemixhybDirichlet);
  make_mixhybDirichlet(surfmixhybDirichlet);

  /*--------------------------------------------------------------------*/
  // fluid stress

  Core::Conditions::ConditionDefinition linefluidstress("DESIGN FLUID STRESS CALC LINE CONDITIONS",
      "FluidStressCalc", "Line Fluid Stress Calculation", Core::Conditions::FluidStressCalc, true,
      Core::Conditions::geometry_type_line);
  Core::Conditions::ConditionDefinition surffluidstress("DESIGN FLUID STRESS CALC SURF CONDITIONS",
      "FluidStressCalc", "Surf Fluid Stress Calculation", Core::Conditions::FluidStressCalc, true,
      Core::Conditions::geometry_type_surface);

  condlist.push_back(linefluidstress);
  condlist.push_back(surffluidstress);

  /*--------------------------------------------------------------------*/
  // lift & drag
  Core::Conditions::ConditionDefinition surfliftdrag("DESIGN FLUID SURF LIFT&DRAG", "LIFTDRAG",
      "Surface LIFTDRAG", Core::Conditions::SurfLIFTDRAG, true,
      Core::Conditions::geometry_type_surface);

  surfliftdrag.add_component(parameter<int>("label"));
  surfliftdrag.add_component(
      parameter<std::vector<double>>("CENTER", {.description = "", .size = 3}));
  surfliftdrag.add_component(parameter<std::vector<double>>(
      "AXIS", {.description = "", .default_value = std::vector<double>{0.0, 0.0, 0.0}, .size = 3}));

  condlist.push_back(surfliftdrag);

  /*--------------------------------------------------------------------*/
  // flow rate through line

  Core::Conditions::ConditionDefinition lineflowrate("DESIGN FLOW RATE LINE CONDITIONS",
      "LineFlowRate", "Line Flow Rate", Core::Conditions::FlowRateThroughLine_2D, true,
      Core::Conditions::geometry_type_line);

  lineflowrate.add_component(parameter<int>("ConditionID"));

  condlist.push_back(lineflowrate);

  /*--------------------------------------------------------------------*/
  // flow rate through surface

  Core::Conditions::ConditionDefinition surfflowrate("DESIGN FLOW RATE SURF CONDITIONS",
      "SurfFlowRate", "Surface Flow Rate", Core::Conditions::FlowRateThroughSurface_3D, true,
      Core::Conditions::geometry_type_surface);

  surfflowrate.add_component(parameter<int>("ConditionID"));

  condlist.push_back(surfflowrate);

  /*--------------------------------------------------------------------*/
  // Volumetric surface flow profile condition
  Core::Conditions::ConditionDefinition volumetric_surface_flow_cond(
      "DESIGN SURF VOLUMETRIC FLOW CONDITIONS", "VolumetricSurfaceFlowCond",
      "volumetric surface flow condition", Core::Conditions::VolumetricSurfaceFlowCond, true,
      Core::Conditions::geometry_type_surface);

  volumetric_surface_flow_cond.add_component(parameter<int>("ConditionID"));

  volumetric_surface_flow_cond.add_component(
      selection<std::string>("CONDITIONTYPE", {"polynomial", "womersley"},
          {.description = "condition type", .default_value = "polynomial"}));

  volumetric_surface_flow_cond.add_component(
      selection<std::string>("PREBIAS", {"not_prebiased", "prebiased", "forced"},
          {.description = "prebias type", .default_value = "not_prebiased"}));

  volumetric_surface_flow_cond.add_component(selection<std::string>(
      "FLOWTYPE", {"inflow", "outflow"}, {.description = "flow type", .default_value = "inflow"}));
  volumetric_surface_flow_cond.add_component(
      parameter<bool>("CORRECTPROFILE", {.description = "correction flag", .default_value = false}));

  volumetric_surface_flow_cond.add_component(parameter<double>("PERIOD"));
  volumetric_surface_flow_cond.add_component(parameter<int>("ORDER"));
  volumetric_surface_flow_cond.add_component(parameter<int>("HARMONICS"));
  volumetric_surface_flow_cond.add_component(parameter<double>("VAL"));
  volumetric_surface_flow_cond.add_component(parameter<int>("FUNCT"));

  volumetric_surface_flow_cond.add_component(
      one_of({all_of({
                  selection<std::string>(
                      "NORMAL", {"self_evaluate"}, {.description = "type of normal evaluation"}),
              }),
          all_of({
              selection<std::string>(
                  "NORMAL", {"use_prescribed"}, {.description = "type of normal evaluation"}),
              parameter<std::vector<double>>("n", {.size = 3}),
          })}));

  volumetric_surface_flow_cond.add_component(
      one_of({all_of({
                  selection<std::string>("CENTEROFMASS", {"self_evaluate"},
                      {.description = "type of center of mass evaluation"}),
              }),
          all_of({
              selection<std::string>("CENTEROFMASS", {"use_prescribed"},
                  {.description = "type of center of mass evaluation"}),
              parameter<std::vector<double>>("c", {.size = 3}),
          })}));

  condlist.push_back(volumetric_surface_flow_cond);


  /*--------------------------------------------------------------------*/
  // Volumetric flow border nodes condition

  Core::Conditions::ConditionDefinition volumetric_border_nodes_cond(
      "DESIGN LINE VOLUMETRIC FLOW BORDER NODES", "VolumetricFlowBorderNodesCond",
      "volumetric flow border nodes condition", Core::Conditions::VolumetricFlowBorderNodes, true,
      Core::Conditions::geometry_type_line);

  volumetric_border_nodes_cond.add_component(parameter<int>("ConditionID"));

  condlist.push_back(volumetric_border_nodes_cond);

  /*--------------------------------------------------------------------*/
  // Volumetric surface total traction corrector
  Core::Conditions::ConditionDefinition total_traction_correction_cond(
      "DESIGN SURF TOTAL TRACTION CORRECTION CONDITIONS", "TotalTractionCorrectionCond",
      "total traction correction condition", Core::Conditions::TotalTractionCorrectionCond, true,
      Core::Conditions::geometry_type_surface);

  total_traction_correction_cond.add_component(parameter<int>("ConditionID"));
  total_traction_correction_cond.add_component(
      selection<std::string>("CONDITIONTYPE", {"polynomial", "womersley"},
          {.description = "condition type", .default_value = "polynomial"}));

  total_traction_correction_cond.add_component(
      selection<std::string>("PREBIAS", {"not_prebiased", "prebiased", "forced"},
          {.description = "prebias type", .default_value = "not_prebiased"}));

  total_traction_correction_cond.add_component(selection<std::string>(
      "FLOWTYPE", {"inflow", "outflow"}, {.description = "flow type", .default_value = "inflow"}));
  total_traction_correction_cond.add_component(
      parameter<bool>("CORRECTPROFILE", {.description = "correction flag", .default_value = false}));

  total_traction_correction_cond.add_component(parameter<double>("PERIOD"));
  total_traction_correction_cond.add_component(parameter<int>("ORDER"));
  total_traction_correction_cond.add_component(parameter<int>("HARMONICS"));
  total_traction_correction_cond.add_component(parameter<double>("VAL"));
  total_traction_correction_cond.add_component(parameter<int>("FUNCT"));

  total_traction_correction_cond.add_component(
      one_of({all_of({
                  selection<std::string>(
                      "NORMAL", {"self_evaluate"}, {.description = "type of normal evaluation"}),
              }),
          all_of({
              selection<std::string>(
                  "NORMAL", {"use_prescribed"}, {.description = "type of normal evaluation"}),
              parameter<std::vector<double>>("NVECTOR", {.size = 3}),
          })}));

  total_traction_correction_cond.add_component(
      one_of({all_of({
                  selection<std::string>("CENTEROFMASS", {"self_evaluate"},
                      {.description = "type of center of mass evaluation"}),
              }),
          all_of({
              selection<std::string>("CENTEROFMASS", {"use_prescribed"},
                  {.description = "type of center of mass evaluation"}),
              parameter<std::vector<double>>("CVECTOR", {.size = 3}),
          })}));

  condlist.push_back(total_traction_correction_cond);

  /*--------------------------------------------------------------------*/
  // Volumetric flow traction correction border nodes condition

  Core::Conditions::ConditionDefinition traction_corrector_border_nodes_cond(
      "DESIGN LINE TOTAL TRACTION CORRECTION BORDER NODES",
      "TotalTractionCorrectionBorderNodesCond", "total traction correction border nodes condition",
      Core::Conditions::TotalTractionCorrectionBorderNodes, true,
      Core::Conditions::geometry_type_line);

  traction_corrector_border_nodes_cond.add_component(parameter<int>("ConditionID"));

  condlist.push_back(traction_corrector_border_nodes_cond);


  /*--------------------------------------------------------------------*/
  // no penetration for darcy flow in porous media

  Core::Conditions::ConditionDefinition nopenetration_surf(
      "DESIGN SURFACE NORMAL NO PENETRATION CONDITION", "no_penetration", "No Penetration",
      Core::Conditions::no_penetration, true, Core::Conditions::geometry_type_surface);

  condlist.push_back(nopenetration_surf);

  /*--------------------------------------------------------------------*/
  // no penetration for darcy flow in porous media

  Core::Conditions::ConditionDefinition nopenetration_line(
      "DESIGN LINE NORMAL NO PENETRATION CONDITION", "no_penetration", "No Penetration",
      Core::Conditions::no_penetration, true, Core::Conditions::geometry_type_line);

  condlist.push_back(nopenetration_line);

  /*--------------------------------------------------------------------*/
  // condition for evaluation of coupling terms in porous media

  Core::Conditions::ConditionDefinition porocoupling_vol("DESIGN VOLUME POROCOUPLING CONDITION",
      "PoroCoupling", "Poro Coupling", Core::Conditions::PoroCoupling, true,
      Core::Conditions::geometry_type_volume);

  condlist.push_back(porocoupling_vol);

  /*--------------------------------------------------------------------*/
  // condition for evaluation of coupling terms in porous media

  Core::Conditions::ConditionDefinition porocoupling_surf("DESIGN SURFACE POROCOUPLING CONDITION",
      "PoroCoupling", "Poro Coupling", Core::Conditions::PoroCoupling, true,
      Core::Conditions::geometry_type_surface);

  condlist.push_back(porocoupling_surf);

  /*--------------------------------------------------------------------*/
  // condition for evaluation of boundary terms in porous media problems

  Core::Conditions::ConditionDefinition poropartint_surf("DESIGN SURFACE PORO PARTIAL INTEGRATION",
      "PoroPartInt", "Poro Partial Integration", Core::Conditions::PoroPartInt, true,
      Core::Conditions::geometry_type_surface);

  condlist.push_back(poropartint_surf);

  /*--------------------------------------------------------------------*/
  // condition for evaluation of boundary terms in porous media problems

  Core::Conditions::ConditionDefinition poropartint_line("DESIGN LINE PORO PARTIAL INTEGRATION",
      "PoroPartInt", "Poro Partial Integration", Core::Conditions::PoroPartInt, true,
      Core::Conditions::geometry_type_line);

  condlist.push_back(poropartint_line);

  /*--------------------------------------------------------------------*/
  // condition for evaluation of boundary terms in porous media problems

  Core::Conditions::ConditionDefinition poropresint_surf("DESIGN SURFACE PORO PRESSURE INTEGRATION",
      "PoroPresInt", "Poro Pressure Integration", Core::Conditions::PoroPresInt, true,
      Core::Conditions::geometry_type_surface);

  condlist.push_back(poropresint_surf);

  /*--------------------------------------------------------------------*/
  // condition for evaluation of boundary terms in porous media problems

  Core::Conditions::ConditionDefinition poropresint_line("DESIGN LINE PORO PRESSURE INTEGRATION",
      "PoroPresInt", "Poro Pressure Integration", Core::Conditions::PoroPresInt, true,
      Core::Conditions::geometry_type_line);

  condlist.push_back(poropresint_line);
}

std::string Inpar::FLUID::to_string(Inpar::FLUID::StabType stabtype)
{
  switch (stabtype)
  {
    case stabtype_nostab:
      return "no_stabilization";
    case stabtype_residualbased:
      return "residual_based";
    case stabtype_edgebased:
      return "edge_based";
    case stabtype_pressureprojection:
      return "pressure_projection";
    default:
      return "unknown";
  }
}

std::string Inpar::FLUID::to_string(Inpar::FLUID::TauType tau)
{
  switch (tau)
  {
    case tau_taylor_hughes_zarins:
      return "Taylor_Hughes_Zarins";
    case tau_taylor_hughes_zarins_wo_dt:
      return "Taylor_Hughes_Zarins_wo_dt";
    case tau_taylor_hughes_zarins_whiting_jansen:
      return "Taylor_Hughes_Zarins_Whiting_Jansen";
    case tau_taylor_hughes_zarins_whiting_jansen_wo_dt:
      return "Taylor_Hughes_Zarins_Whiting_Jansen_wo_dt";
    case tau_taylor_hughes_zarins_scaled:
      return "Taylor_Hughes_Zarins_scaled";
    case tau_taylor_hughes_zarins_scaled_wo_dt:
      return "Taylor_Hughes_Zarins_scaled_wo_dt";
    case tau_franca_barrenechea_valentin_frey_wall:
      return "Franca_Barrenechea_Valentin_Frey_Wall";
    case tau_franca_barrenechea_valentin_frey_wall_wo_dt:
      return "Franca_Barrenechea_Valentin_Frey_Wall_wo_dt";
    case tau_shakib_hughes_codina:
      return "Shakib_Hughes_Codina";
    case tau_shakib_hughes_codina_wo_dt:
      return "Shakib_Hughes_Codina_wo_dt";
    case tau_codina:
      return "Codina";
    case tau_codina_wo_dt:
      return "Codina_wo_dt";
    case tau_codina_convscaled:
      return "Codina_convscaled";
    case tau_franca_madureira_valentin_badia_codina:
      return "Franca_Madureira_Valentin_Badia_Codina";
    case tau_franca_madureira_valentin_badia_codina_wo_dt:
      return "Franca_Madureira_Valentin_Badia_Codina_wo_dt";
    case tau_hughes_franca_balestra_wo_dt:
      return "Hughes_Franca_Balestra_wo_dt";
    case tau_not_defined:
      return "not_defined";
  }
  return "not_defined";
}

std::string Inpar::FLUID::to_string(Inpar::FLUID::CrossStress crossstress)
{
  switch (crossstress)
  {
    case cross_stress_stab_none:
      return "no_cross";
    case cross_stress_stab:
      return "yes_cross";
    case cross_stress_stab_only_rhs:
      return "cross_rhs";
    default:
      return "unknown";
  }
}

std::string Inpar::FLUID::to_string(Inpar::FLUID::VStab vstab)
{
  switch (vstab)
  {
    case viscous_stab_none:
      return "no_viscous";
    case viscous_stab_gls:
      return "yes_viscous";
    case viscous_stab_gls_only_rhs:
      return "viscous_rhs";
    case viscous_stab_usfem:
      return "usfem_viscous";
    case viscous_stab_usfem_only_rhs:
      return "usfem_viscous_rhs";
    default:
      return "unknown";
  }
}

std::string Inpar::FLUID::to_string(Inpar::FLUID::RStab rstab)
{
  switch (rstab)
  {
    case reactive_stab_none:
      return "no_reactive";
    case reactive_stab_gls:
      return "yes_reactive";
    case reactive_stab_usfem:
      return "usfem_reactive";
    default:
      return "unknown";
  }
}

std::string Inpar::FLUID::to_string(Inpar::FLUID::ReynoldsStress reynoldsstress)
{
  switch (reynoldsstress)
  {
    case reynolds_stress_stab_none:
      return "no_reynolds";
    case reynolds_stress_stab:
      return "yes_reynolds";
    case reynolds_stress_stab_only_rhs:
      return "reynolds_rhs";
    default:
      return "unknown";
  }
}

std::string Inpar::FLUID::to_string(Inpar::FLUID::Transient transient)
{
  switch (transient)
  {
    case inertia_stab_drop:
      return "no_transient";
    case inertia_stab_keep:
      return "yes_transient";
    case inertia_stab_keep_complete:
      return "transient_complete";
    default:
      return "unknown";
  }
}

std::string Inpar::FLUID::to_string(Inpar::FLUID::EosPres eospres)
{
  switch (eospres)
  {
    case EOS_PRES_none:
      return "none";
    case EOS_PRES_std_eos:
      return "std_eos";
    case EOS_PRES_xfem_gp:
      return "xfem_gp";
    default:
      return "unknown";
  }
}

std::string Inpar::FLUID::to_string(Inpar::FLUID::EosConvStream eosconvstream)
{
  switch (eosconvstream)
  {
    case EOS_CONV_STREAM_none:
      return "none";
    case EOS_CONV_STREAM_std_eos:
      return "std_eos";
    case EOS_CONV_STREAM_xfem_gp:
      return "xfem_gp";
    default:
      return "unknown";
  }
}

std::string Inpar::FLUID::to_string(Inpar::FLUID::EosConvCross eosconvcross)
{
  switch (eosconvcross)
  {
    case EOS_CONV_CROSS_none:
      return "none";
    case EOS_CONV_CROSS_std_eos:
      return "std_eos";
    case EOS_CONV_CROSS_xfem_gp:
      return "xfem_gp";
    default:
      return "unknown";
  }
}

std::string Inpar::FLUID::to_string(Inpar::FLUID::EosDiv eosdiv)
{
  switch (eosdiv)
  {
    case EOS_DIV_none:
      return "none";
    case EOS_DIV_vel_jump_std_eos:
      return "vel_jump_std_eos";
    case EOS_DIV_vel_jump_xfem_gp:
      return "vel_jump_xfem_gp";
    case EOS_DIV_div_jump_std_eos:
      return "div_jump_std_eos";
    case EOS_DIV_div_jump_xfem_gp:
      return "div_jump_xfem_gp";
    default:
      return "unknown";
  }
}

std::string Inpar::FLUID::to_string(Inpar::FLUID::EosTauType eostautype)
{
  switch (eostautype)
  {
    case EOS_tau_burman:
      return "Burman";
    case EOS_tau_burman_fernandez_hansbo:
      return "Burman_Fernandez_Hansbo";
    case EOS_tau_burman_fernandez_hansbo_wo_dt:
      return "Burman_Fernandez_Hansbo_wo_dt";
    case EOS_tau_braack_burman_john_lube:
      return "Braack_Burman_John_Lube";
    case EOS_tau_braack_burman_john_lube_wo_divjump:
      return "Braack_Burman_John_Lube_wo_divjump";
    case EOS_tau_franca_barrenechea_valentin_wall:
      return "Franca_Barrenechea_Valentin_Wall";
    case EOS_tau_burman_fernandez:
      return "Burman_Fernandez";
    case EOS_tau_burman_hansbo_dangelo_zunino:
      return "Burman_Hansbo_DAngelo_Zunino";
    case EOS_tau_burman_hansbo_dangelo_zunino_wo_dt:
      return "Burman_Hansbo_DAngelo_Zunino_wo_dt";
    case EOS_tau_schott_massing_burman_dangelo_zunino:
      return "Schott_Massing_Burman_DAngelo_Zunino";
    case EOS_tau_schott_massing_burman_dangelo_zunino_wo_dt:
      return "Schott_Massing_Burman_DAngelo_Zunino_wo_dt";
    case EOS_tau_Taylor_Hughes_Zarins_Whiting_Jansen_Codina_scaling:
      return "Taylor_Hughes_Zarins_Whiting_Jansen_Codina_scaling";
    case EOS_tau_poroelast_fluid:
      return "Poroelast_Fluid";
    case EOS_tau_not_defined:
      return "not_defined";
    default:
      return "unknown";
  }
}

std::string Inpar::FLUID::to_string(Inpar::FLUID::EosElementLength eoselementlength)
{
  switch (eoselementlength)
  {
    case EOS_he_max_diameter_to_opp_surf:
      return "EOS_he_max_diameter_to_opp_surf";
    case EOS_he_max_dist_to_opp_surf:
      return "EOS_he_max_dist_to_opp_surf";
    case EOS_he_surf_with_max_diameter:
      return "EOS_he_surf_with_max_diameter";
    case EOS_hk_max_diameter:
      return "EOS_hk_max_diameter";
    case EOS_he_surf_diameter:
      return "EOS_he_surf_diameter";
    case EOS_he_vol_eq_diameter:
      return "EOS_he_vol_eq_diameter";
    default:
      return "unknown";
  }
}

std::string Inpar::FLUID::to_string(Inpar::FLUID::VremanFiMethod vremanfimethod)
{
  switch (vremanfimethod)
  {
    case cuberootvol:
      return "CubeRootVol";
    case dir_dep:
      return "Direction_dependent";
    case min_len:
      return "Minimum_length";
    default:
      return "unknown";
  }
}

FOUR_C_NAMESPACE_CLOSE
