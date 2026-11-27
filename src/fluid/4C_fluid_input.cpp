// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_legacy_enum_definitions_conditions.hpp"
FOUR_C_NAMESPACE_OPEN



std::vector<Core::IO::InputSpec> FLUID::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;

  std::vector<Core::IO::InputSpec> specs;

  specs.push_back(group("FLUID DYNAMIC",
      {

          // physical type of fluid flow (incompressible, varying density, loma, Boussinesq
          // approximation,
          // temperature-dependent water)
          deprecated_selection<FLUID::PhysicalType>("PHYSICAL_TYPE",
              {
                  {"Incompressible", incompressible},
                  {"Weakly_compressible", weakly_compressible},
                  {"Weakly_compressible_stokes", weakly_compressible_stokes},
                  {"Weakly_compressible_dens_mom", weakly_compressible_dens_mom},
                  {"Weakly_compressible_stokes_dens_mom", weakly_compressible_stokes_dens_mom},
                  {"Artificial_compressibility", artcomp},
                  {"Varying_density", varying_density},
                  {"Loma", loma},
                  {"Temp_dep_water", tempdepwater},
                  {"Boussinesq", boussinesq},
                  {"Stokes", stokes},
                  {"Oseen", oseen},
              },
              {.description = "Physical Type", .default_value = incompressible}),

          // number of linear solver used for fluid problem
          parameter<int>(
              "LINEAR_SOLVER", {.description = "number of linear solver used for fluid dynamics",
                                   .default_value = -1}),

          // number of linear solver used for fluid problem (former fluid pressure solver for
          // SIMPLER
          // preconditioning with fluid)
          parameter<int>("SIMPLER_SOLVER",
              {.description = "number of linear solver used for fluid dynamic (ONLY "
                              "NECESSARY FOR BlockGaussSeidel solver block within fluid "
                              "mehstying case any more!!!!)",
                  .default_value = -1}),

          // Flag to define the way of calculating stresses and wss
          deprecated_selection<FLUID::WSSType>("WSS_TYPE",
              {
                  {"Standard", wss_standard},
                  {"Aggregation", wss_aggregation},
                  {"Mean", wss_mean},
              },
              {.description = "which type of stresses and wall shear stress",
                  .default_value = wss_standard}),

          // Set ML-solver number for smoothing of residual-based calculated wallshearstress via
          // plain
          // aggregation.
          parameter<int>("WSS_ML_AGR_SOLVER",
              {.description = "Set ML-solver number for smoothing of residual-based calculated "
                              "wallshearstress via plain aggregation.",
                  .default_value = -1}),

          deprecated_selection<FLUID::TimeIntegrationScheme>("TIMEINTEGR",
              {
                  {"Stationary", timeint_stationary},
                  {"Np_Gen_Alpha", timeint_npgenalpha},
                  {"Af_Gen_Alpha", timeint_afgenalpha},
                  {"One_Step_Theta", timeint_one_step_theta},
                  {"BDF2", timeint_bdf2},
              },
              {.description = "Time Integration Scheme", .default_value = timeint_one_step_theta}),

          parameter<FLUID::OstContAndPress>(
              "OST_CONT_PRESS", {.description = "One step theta option for time discretization of "
                                                "continuity eq. and pressure",
                                    .default_value = Cont_normal_Press_normal}),

          parameter<FLUID::LinearisationAction>("NONLINITER",
              {.description = "Nonlinear iteration scheme", .default_value = fixed_point_like}),

          deprecated_selection<std::string>("PREDICTOR",
              {"steady_state", "zero_acceleration", "constant_acceleration", "constant_increment",
                  "explicit_second_order_midpoint", "TangVel"},
              {.description = "Predictor for first guess in nonlinear iteration",
                  .default_value = "steady_state"}),


          deprecated_selection<FLUID::ItNorm>("CONVCHECK",
              {
                  {"L_2_norm", fncc_L2},
              },
              {.description =
                      "Norm for convergence check (relative increments and absolute residuals)",
                  .default_value = fncc_L2}),

          parameter<bool>("INCONSISTENT_RESIDUAL",
              {.description = "do not evaluate residual after solution has converged (->faster)",
                  .default_value = false}),



          deprecated_selection<FLUID::InitialField>("INITIALFIELD",
              {
                  {"zero_field", initfield_zero_field},
                  {"field_by_function", initfield_field_by_function},
                  {"disturbed_field_from_function", initfield_disturbed_field_from_function},
                  {"FLAME_VORTEX_INTERACTION", initfield_flame_vortex_interaction},
                  {"BELTRAMI-FLOW", initfield_beltrami_flow},
                  {"KIM-MOIN-FLOW", initfield_kim_moin_flow},
                  {"hit_comte_bellot_corrsin_initial_field", initfield_hit_comte_bellot_corrsin},
                  {"forced_hit_simple_algebraic_spectrum",
                      initfield_forced_hit_simple_algebraic_spectrum},
                  {"forced_hit_numeric_spectrum", initfield_forced_hit_numeric_spectrum},
                  {"forced_hit_passive", initfield_passive_hit_const_input},
                  {"channel_weakly_compressible", initfield_channel_weakly_compressible},
              },
              {.description = "Initial Field for transport problem",
                  .default_value = initfield_zero_field}),


          parameter<int>("OSEENFIELDFUNCNO",
              {.description = "function number of Oseen advective field", .default_value = -1}),

          parameter<bool>(
              "LIFTDRAG", {.description = "Calculate lift and drag forces along specified boundary",
                              .default_value = false}),


          deprecated_selection<std::string>("CONVFORM", {"convective", "conservative"},
              {.description = "form of convective term", .default_value = "convective"}),

          parameter<bool>("NONLINEARBC",
              {.description = "Flag to activate check for potential nonlinear boundary conditions",
                  .default_value = false}),

          deprecated_selection<FLUID::MeshTying>("MESHTYING",
              {
                  {"no", no_meshtying},
                  {"Condensed_Smat", condensed_smat},
                  {"Condensed_Bmat", condensed_bmat},
                  {"Condensed_Bmat_merged", condensed_bmat_merged},
              },
              {.description = "Flag to (de)activate mesh tying algorithm",
                  .default_value = no_meshtying}),

          parameter<FLUID::Gridvel>("GRIDVEL",
              {.description = "scheme for determination of gridvelocity from displacements",
                  .default_value = BE}),

          parameter<bool>("ALLDOFCOUPLED",
              {.description = "all dof (incl. pressure) are coupled", .default_value = true}),

          parameter<CalcError>("CALCERROR",
              {.description = "Flag to (de)activate error calculations", .default_value = no}),
          parameter<int>("CALCERRORFUNCNO",
              {.description = "Function for Error Calculation", .default_value = -1}),

          parameter<int>(
              "CORRTERMFUNCNO", {.description = "Function for calculation of the correction "
                                                "term for the weakly compressible problem",
                                    .default_value = -1}),

          parameter<int>(
              "BODYFORCEFUNCNO", {.description = "Function for calculation of the body force for "
                                                 "the weakly compressible problem",
                                     .default_value = -1}),

          parameter<double>("STAB_DEN_REF",
              {.description = "Reference stabilization parameter for the density for the "
                              "HDG weakly compressible formulation",
                  .default_value = 0.0}),

          parameter<double>("STAB_MOM_REF",
              {.description = "Reference stabilization parameter for the momentum for the "
                              "HDG weakly compressible formulation",
                  .default_value = 0.0}),

          parameter<int>("VARVISCFUNCNO",
              {.description = "Function for calculation of a variable viscosity for the "
                              "weakly compressible problem",
                  .default_value = -1}),

          parameter<bool>("PRESSAVGBC",
              {.description = "Flag to (de)activate imposition of boundary condition for the "
                              "considered element average pressure",
                  .default_value = false}),


          parameter<double>(
              "REFMACH", {.description = "Reference Mach number", .default_value = 1.0}),

          parameter<bool>(
              "BLOCKMATRIX", {.description = "Indicates if system matrix should be assembled into "
                                             "a sparse block matrix type.",
                                 .default_value = false}),

          parameter<bool>("ADAPTCONV", {.description = "Switch on adaptive control of linear "
                                                       "solver tolerance for nonlinear solution",
                                           .default_value = false}),
          parameter<double>("ADAPTCONV_BETTER",
              {.description =
                      "The linear solver shall be this much better than the current nonlinear "
                      "residual in the nonlinear convergence limit",
                  .default_value = 0.1}),

          parameter<bool>("INFNORMSCALING",
              {.description = "Scale blocks of matrix with row infnorm?", .default_value = false}),

          parameter<bool>(
              "GMSH_OUTPUT", {.description = "write output to gmsh files", .default_value = false}),
          parameter<bool>("COMPUTE_DIVU",
              {.description = "Compute divergence of velocity field at the element center",
                  .default_value = false}),
          parameter<bool>("COMPUTE_EKIN",
              {.description =
                      "Compute kinetic energy at the end of each time step and write it to file.",
                  .default_value = false}),
          parameter<bool>("NEW_OST",
              {.description =
                      "Solve the Navier-Stokes equation with the new One Step Theta algorithm",
                  .default_value = false}),  // TODO: To be removed.
          parameter<int>("RESULTSEVERY",
              {.description = "Increment for writing solution", .default_value = 1}),
          parameter<int>("RESTARTEVERY",
              {.description = "Increment for writing restart", .default_value = 20}),

          parameter<int>(
              "NUMSTEP", {.description = "Total number of Timesteps", .default_value = 1}),
          parameter<int>(
              "STEADYSTEP", {.description = "steady state check every step", .default_value = -1}),
          parameter<int>("NUMSTASTEPS",
              {.description = "Number of Steps for Starting Scheme", .default_value = 0}),
          parameter<int>("STARTFUNCNO",
              {.description = "Function for Initial Starting Field", .default_value = -1}),
          parameter<int>(
              "ITEMAX", {.description = "max. number of nonlin. iterations", .default_value = 10}),
          parameter<int>("INITSTATITEMAX",
              {.description = "max number of nonlinear iterations for initial stationary solution",
                  .default_value = 5}),

          parameter<double>(
              "TIMESTEP", {.description = "Time increment dt", .default_value = 0.01}),
          parameter<double>(
              "MAXTIME", {.description = "Total simulation time", .default_value = 1000.0}),
          parameter<double>(
              "ALPHA_M", {.description = "Time integration factor", .default_value = 1.0}),
          parameter<double>(
              "ALPHA_F", {.description = "Time integration factor", .default_value = 1.0}),

          parameter<double>(
              "GAMMA", {.description = "Time integration factor", .default_value = 1.0}),
          parameter<double>(
              "THETA", {.description = "Time integration factor", .default_value = 0.66}),

          parameter<double>("START_THETA",
              {.description = "Time integration factor for starting scheme", .default_value = 1.0}),

          parameter<bool>("STRONG_REDD_3D_COUPLING_TYPE",
              {.description = "Flag to (de)activate potential Strong 3D redD coupling",
                  .default_value = false}),

          parameter<int>("VELGRAD_PROJ_SOLVER",
              {.description = "Number of linear solver used for L2 projection",
                  .default_value = -1}),


          deprecated_selection<FLUID::GradientReconstructionMethod>("VELGRAD_PROJ_METHOD",
              {
                  {"none", gradreco_none},
                  {"superconvergent_patch_recovery", gradreco_spr},
                  {"L2_projection", gradreco_l2},
              },
              {.description = "Flag to (de)activate gradient reconstruction.",
                  .default_value = gradreco_none}),

          parameter<bool>("OFF_PROC_ASSEMBLY",
              {.description = "Do not evaluate ghosted elements but communicate them "
                              "--> faster if element call is expensive",
                  .default_value = false})},
      {.required = false}));
  /*----------------------------------------------------------------------*/
  specs.push_back(group("FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
      {

          parameter<double>(
              "TOL_VEL_RES", {.description = "Tolerance for convergence check of velocity residual",
                                 .default_value = 1e-6}),

          parameter<double>("TOL_VEL_INC",
              {.description = "Tolerance for convergence check of velocity increment",
                  .default_value = 1e-6}),

          parameter<double>("TOL_PRES_RES",
              {.description = "Tolerance for convergence check of pressure residual",
                  .default_value = 1e-6}),

          parameter<double>("TOL_PRES_INC",
              {.description = "Tolerance for convergence check of pressure increment",
                  .default_value = 1e-6})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  specs.push_back(group("FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION",
      {

          // this parameter defines various stabilized methods
          deprecated_selection<FLUID::StabType>("STABTYPE",
              {
                  {"no_stabilization", stabtype_nostab},
                  {"residual_based", stabtype_residualbased},
                  {"edge_based", stabtype_edgebased},
                  {"pressure_projection", stabtype_pressureprojection},
              },
              {.description =
                      "Apply (un)stabilized fluid formulation. No stabilization is only possible "
                      "for inf-sup stable elements.Use a residual-based stabilization or, more "
                      "generally, a stabilization \nbased on the concept of the residual-based "
                      "variational multiscale method...\nExpecting additional input. Use an "
                      "edge-based stabilization, especially for XFEM. Alternative: Element/cell "
                      "based polynomial pressure projection, see Dohrmann/Bochev 2004, IJNMF",
                  .default_value = stabtype_residualbased}),

          parameter<bool>("INCONSISTENT",
              {.description = "residual based without second derivatives (i.e. only "
                              "consistent for tau->0, but faster)",
                  .default_value = false}),

          parameter<bool>("Reconstruct_Sec_Der",
              {.description =
                      "residual computed with a reconstruction of the second derivatives via "
                      "projection or superconvergent patch recovery",
                  .default_value = false}),

          // the following parameters are necessary only if a residual based stabilized method is
          // applied
          deprecated_selection<SubscalesTD>("TDS",
              {
                  {"quasistatic", subscales_quasistatic},
                  {"time_dependent", subscales_time_dependent},
              },
              {.description = "Flag to allow time dependency of subscales for residual-based "
                              "stabilization.",
                  .default_value = subscales_quasistatic}),

          deprecated_selection<Transient>("TRANSIENT",
              {
                  {"no_transient", inertia_stab_drop},
                  {"yes_transient", inertia_stab_keep},
                  {"transient_complete", inertia_stab_keep_complete},
              },
              {.description = "Specify how to treat the transient term. Use transient term "
                              "(recommended for "
                              "time "
                              "dependent subscales) or use transient term including a "
                              "linearisation of 1/tau",
                  .default_value = inertia_stab_drop}),

          parameter<bool>("PSPG",
              {.description = "Flag to (de)activate PSPG stabilization.", .default_value = true}),
          parameter<bool>("SUPG",
              {.description = "Flag to (de)activate SUPG stabilization.", .default_value = true}),
          parameter<bool>("GRAD_DIV",
              {.description = "Flag to (de)activate grad-div term.", .default_value = true}),

          deprecated_selection<VStab>("VSTAB",
              {
                  {"no_vstab", viscous_stab_none},
                  {"vstab_gls", viscous_stab_gls},
                  {"vstab_gls_rhs", viscous_stab_gls_only_rhs},
                  {"vstab_usfem", viscous_stab_usfem},
                  {"vstab_usfem_rhs", viscous_stab_usfem_only_rhs},
              },
              {.description =
                      "Flag to (de)activate viscous term in residual-based stabilization. Options: "
                      "No viscous term in stabilization, or, Viscous stabilization of GLS type, "
                      "or, Viscous stabilization of GLS type, included only on the right hand "
                      "side, or, Viscous stabilization of USFEM type, or, Viscous stabilization of "
                      "USFEM type, included only on the right hand side",
                  .default_value = viscous_stab_none}),

          deprecated_selection<RStab>("RSTAB",
              {
                  {"no_rstab", reactive_stab_none},
                  {"rstab_gls", reactive_stab_gls},
                  {"rstab_usfem", reactive_stab_usfem},
              },
              {.description = "Flag to (de)activate reactive term in residual-based stabilization.",
                  .default_value = reactive_stab_none}),

          deprecated_selection<CrossStress>("CROSS-STRESS",
              {
                  {"no_cross", cross_stress_stab_none},
                  {"yes_cross", cross_stress_stab},
                  {"cross_rhs", cross_stress_stab_only_rhs},
              },
              {.description =
                      "Flag to (de)activate cross-stress term -> residual-based VMM. Options:No "
                      "cross-stress term, or,Include the cross-stress term with a linearization of "
                      "the convective part, or, Include cross-stress term, but only explicitly on "
                      "right hand side",
                  .default_value = cross_stress_stab_none}),

          deprecated_selection<ReynoldsStress>("REYNOLDS-STRESS",
              {
                  {"no_reynolds", reynolds_stress_stab_none},
                  {"yes_reynolds", reynolds_stress_stab},
                  {"reynolds_rhs", reynolds_stress_stab_only_rhs},
              },
              {.description = "Flag to (de)activate Reynolds-stress term -> residual-based VMM.",
                  .default_value = reynolds_stress_stab_none}),



          deprecated_selection<TauType>("DEFINITION_TAU",
              {
                  {"Taylor_Hughes_Zarins", tau_taylor_hughes_zarins},
                  {"Taylor_Hughes_Zarins_wo_dt", tau_taylor_hughes_zarins_wo_dt},
                  {"Taylor_Hughes_Zarins_Whiting_Jansen", tau_taylor_hughes_zarins_whiting_jansen},
                  {"Taylor_Hughes_Zarins_Whiting_Jansen_wo_dt",
                      tau_taylor_hughes_zarins_whiting_jansen_wo_dt},
                  {"Taylor_Hughes_Zarins_scaled", tau_taylor_hughes_zarins_scaled},
                  {"Taylor_Hughes_Zarins_scaled_wo_dt", tau_taylor_hughes_zarins_scaled_wo_dt},
                  {"Franca_Barrenechea_Valentin_Frey_Wall",
                      tau_franca_barrenechea_valentin_frey_wall},
                  {"Franca_Barrenechea_Valentin_Frey_Wall_wo_dt",
                      tau_franca_barrenechea_valentin_frey_wall_wo_dt},
                  {"Shakib_Hughes_Codina", tau_shakib_hughes_codina},
                  {"Shakib_Hughes_Codina_wo_dt", tau_shakib_hughes_codina_wo_dt},
                  {"Codina", tau_codina},
                  {"Codina_wo_dt", tau_codina_wo_dt},
                  {"Codina_convscaled", tau_codina_convscaled},
                  {"Franca_Madureira_Valentin_Badia_Codina",
                      tau_franca_madureira_valentin_badia_codina},
                  {"Franca_Madureira_Valentin_Badia_Codina_wo_dt",
                      tau_franca_madureira_valentin_badia_codina_wo_dt},
                  {"Hughes_Franca_Balestra_wo_dt", tau_hughes_franca_balestra_wo_dt},
              },
              {.description = "Definition of tau",
                  .default_value = tau_franca_barrenechea_valentin_frey_wall}),


          // this parameter selects the characteristic element length for tau_Mu for all
          // stabilization parameter definitions requiring such a length
          deprecated_selection<CharEleLengthU>("CHARELELENGTH_U",
              {
                  {"streamlength", streamlength_u},
                  {"volume_equivalent_diameter", volume_equivalent_diameter_u},
                  {"root_of_volume", root_of_volume_u},
              },
              {.description = "Characteristic element length for tau_Mu",
                  .default_value = streamlength_u}),

          // this parameter selects the characteristic element length for tau_Mp and tau_C for
          // all stabilization parameter definitions requiring such a length
          deprecated_selection<CharEleLengthPC>("CHARELELENGTH_PC",
              {
                  {"streamlength", streamlength_pc},
                  {"volume_equivalent_diameter", volume_equivalent_diameter_pc},
                  {"root_of_volume", root_of_volume_pc},
              },
              {.description = "Characteristic element length for tau_Mp/tau_C",
                  .default_value = volume_equivalent_diameter_pc}),

          // this parameter selects the location where tau is evaluated
          deprecated_selection<std::string>("EVALUATION_TAU",
              {"element_center", "integration_point"},
              {.description = "Location where tau is evaluated",
                  .default_value = "element_center"}),

          // this parameter selects the location where the material law is evaluated
          // (does not fit here very well, but parameter transfer is easier)
          deprecated_selection<std::string>("EVALUATION_MAT",
              {"element_center", "integration_point"},
              {.description = "Location where material law is evaluated",
                  .default_value = "element_center"}),

          // these parameters active additional terms in loma continuity equation
          // which might be identified as SUPG-/cross- and Reynolds-stress term
          parameter<bool>("LOMA_CONTI_SUPG",
              {.description =
                      "Flag to (de)activate SUPG stabilization in loma continuity equation.",
                  .default_value = false}),

          deprecated_selection<CrossStress>("LOMA_CONTI_CROSS_STRESS",
              {
                  {"no_cross", cross_stress_stab_none},
                  {"yes_cross", cross_stress_stab},
                  {"cross_rhs", cross_stress_stab_only_rhs},
              },
              {.description = "Flag to (de)activate cross-stress term loma continuity equation-> "
                              "residual-based VMM.",
                  .default_value = cross_stress_stab_none}),

          deprecated_selection<ReynoldsStress>("LOMA_CONTI_REYNOLDS_STRESS",
              {
                  {"no_reynolds", reynolds_stress_stab_none},
                  {"yes_reynolds", reynolds_stress_stab},
                  {"reynolds_rhs", reynolds_stress_stab_only_rhs},
              },
              {.description =
                      "Flag to (de)activate Reynolds-stress term loma continuity equation-> "
                      "residual-based VMM.",
                  .default_value = reynolds_stress_stab_none})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  specs.push_back(group("FLUID DYNAMIC/EDGE-BASED STABILIZATION",
      {

          //! Flag to (de)activate edge-based (EOS) pressure stabilization
          deprecated_selection<EosPres>("EOS_PRES",
              {
                  {"none", EOS_PRES_none},
                  {"std_eos", EOS_PRES_std_eos},
                  {"xfem_gp", EOS_PRES_xfem_gp},
              },
              {.description =
                      "Flag to (de)activate pressure edge-based stabilization. Options: do not use "
                      "pressure edge-based stabilization, or, use pressure edge-based "
                      "stabilization as "
                      "standard edge-based stabilization on the entire domain, or, use pressure "
                      "edge-based "
                      "stabilization as xfem ghost-penalty stabilization just around cut elements",
                  .default_value = EOS_PRES_none}),

          //! Flag to (de)activate edge-based (EOS) convective streamline stabilization
          deprecated_selection<EosConvStream>("EOS_CONV_STREAM",
              {
                  {"none", EOS_CONV_STREAM_none},
                  {"std_eos", EOS_CONV_STREAM_std_eos},
                  {"xfem_gp", EOS_CONV_STREAM_xfem_gp},
              },
              {.description =
                      "Flag to (de)activate convective streamline edge-based stabilization. "
                      "Options: "
                      "do "
                      "not use convective streamline edge-based stabilization, or, use convective "
                      "streamline edge-based stabilization as standard edge-based stabilization on "
                      "the "
                      "entire domain, or, use convective streamline edge-based stabilization as "
                      "xfem "
                      "ghost-penalty stabilization just around cut elements",
                  .default_value = EOS_CONV_STREAM_none}),

          //! Flag to (de)activate edge-based (EOS) convective crosswind stabilization
          deprecated_selection<EosConvCross>("EOS_CONV_CROSS",
              {
                  {"none", EOS_CONV_CROSS_none},
                  {"std_eos", EOS_CONV_CROSS_std_eos},
                  {"xfem_gp", EOS_CONV_CROSS_xfem_gp},
              },
              {.description =
                      "Flag to (de)activate convective crosswind edge-based stabilization. "
                      "Options:do not use convective crosswind edge-based stabilization, or, use "
                      "convective crosswind edge-based stabilization as standard edge-based "
                      "stabilization on the entire domain, or, use convective crosswind edge-based "
                      "stabilization as xfem ghost-penalty stabilization just around cut elements",
                  .default_value = EOS_CONV_CROSS_none}),

          //! Flag to (de)activate edge-based (EOS) divergence stabilization
          deprecated_selection<EosDiv>("EOS_DIV",
              {
                  {"none", EOS_DIV_none},
                  {"vel_jump_std_eos", EOS_DIV_vel_jump_std_eos},
                  {"vel_jump_xfem_gp", EOS_DIV_vel_jump_xfem_gp},
                  {"div_jump_std_eos", EOS_DIV_div_jump_std_eos},
                  {"div_jump_xfem_gp", EOS_DIV_div_jump_xfem_gp},
              },
              {.description =
                      "Flag to (de)activate divergence edge-based stabilization. Options: do not "
                      "use "
                      "divergence edge-based stabilization, or, divergence edge-based "
                      "stabilization "
                      "based "
                      "on velocity jump on the entire domain, or, divergence edge-based "
                      "stabilization "
                      "based on divergence jump just around cut elements, or, divergence "
                      "edge-based "
                      "stabilization based on velocity jump on the entire domain, or, divergence "
                      "edge-based stabilization based on divergence jump just around cut elements",
                  .default_value = EOS_DIV_none}),

          //! special least-squares condition for pseudo 2D examples where pressure level is
          //! determined
          //! via
          //! Krylov-projection
          parameter<bool>("PRES_KRYLOV_2Dz",
              {.description = "residual based without second derivatives (i.e. only "
                              "consistent for tau->0, but faster)",
                  .default_value = false}),

          //! this parameter selects the definition of Edge-based stabilization parameter
          deprecated_selection<EosTauType>("EOS_DEFINITION_TAU",
              {
                  {"Burman_Fernandez_Hansbo", FLUID::EOS_tau_burman_fernandez_hansbo},
                  {"Burman_Fernandez_Hansbo_wo_dt", FLUID::EOS_tau_burman_fernandez_hansbo_wo_dt},
                  {"Braack_Burman_John_Lube", FLUID::EOS_tau_braack_burman_john_lube},
                  {"Braack_Burman_John_Lube_wo_divjump",
                      FLUID::EOS_tau_braack_burman_john_lube_wo_divjump},
                  {"Franca_Barrenechea_Valentin_Wall",
                      FLUID::EOS_tau_franca_barrenechea_valentin_wall},
                  {"Burman_Fernandez", FLUID::EOS_tau_burman_fernandez},
                  {"Burman_Hansbo_DAngelo_Zunino", FLUID::EOS_tau_burman_hansbo_dangelo_zunino},
                  {"Burman_Hansbo_DAngelo_Zunino_wo_dt",
                      FLUID::EOS_tau_burman_hansbo_dangelo_zunino_wo_dt},
                  {"Schott_Massing_Burman_DAngelo_Zunino",
                      FLUID::EOS_tau_schott_massing_burman_dangelo_zunino},
                  {"Schott_Massing_Burman_DAngelo_Zunino_wo_dt",
                      FLUID::EOS_tau_schott_massing_burman_dangelo_zunino_wo_dt},
                  {"Burman", FLUID::EOS_tau_burman},
                  {"Taylor_Hughes_Zarins_Whiting_Jansen_Codina_scaling",
                      FLUID::EOS_tau_Taylor_Hughes_Zarins_Whiting_Jansen_Codina_scaling},
                  {"tau_not_defined", FLUID::EOS_tau_not_defined},
              },
              {.description = "Definition of stabilization parameter for edge-based stabilization",
                  .default_value = FLUID::EOS_tau_burman_hansbo_dangelo_zunino}),

          //! this parameter selects how the element length of Edge-based stabilization is defined
          parameter<EosElementLength>("EOS_H_DEFINITION",
              {.description = "Definition of element length for edge-based "
                              "stabilization. Options:take the maximal "
                              "(nsd-1)D diameter of faces that connect the "
                              "internal face to its opposite faces, "
                              "or, take the maximal 1D distance along 1D edge to "
                              "opposite surface for both parent "
                              "elements, or, take the maximal (nsd-1)D face "
                              "diameter of all faces for both parent "
                              "elements, or, maximal nD diameter of the "
                              "neighboring elements, or, maximal (n-1)D "
                              "diameter of the internal face/edge, or, take the "
                              "maximal volume equivalent diameter "
                              "of adjacent elements",
                  .default_value = EOS_he_max_diameter_to_opp_surf})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  specs.push_back(group("FLUID DYNAMIC/POROUS-FLOW STABILIZATION",
      {

          parameter<bool>("STAB_BIOT",
              {.description = "Flag to (de)activate BIOT stabilization.", .default_value = false}),
          parameter<double>(
              "STAB_BIOT_SCALING", {.description = "Scaling factor for stabilization parameter for "
                                                   "biot stabilization of porous flow.",
                                       .default_value = 1.0}),

          // this parameter defines various stabilized methods
          deprecated_selection<FLUID::StabType>("STABTYPE",
              {
                  {"no_stabilization", stabtype_nostab},
                  {"residual_based", stabtype_residualbased},
                  {"edge_based", stabtype_edgebased},
              },
              {.description =
                      "Apply (un)stabilized fluid formulation. No stabilization is only possible "
                      "for inf-sup stable elements. Use a residual-based stabilization or, more "
                      "generally, a stabilization \nbased on the concept of the residual-based "
                      "variational multiscale method...\nExpecting additional inputUse an "
                      "edge-based stabilization, especially for XFEM",
                  .default_value = stabtype_residualbased}),

          parameter<bool>("INCONSISTENT",
              {.description = "residual based without second derivatives (i.e. only "
                              "consistent for tau->0, but faster)",
                  .default_value = false}),

          parameter<bool>("Reconstruct_Sec_Der",
              {.description =
                      "residual computed with a reconstruction of the second derivatives via "
                      "projection or superconvergent patch recovery",
                  .default_value = false}),

          // the following parameters are necessary only if a residual based stabilized method is
          // applied
          deprecated_selection<SubscalesTD>("TDS",
              {
                  {"quasistatic", subscales_quasistatic},
                  {"time_dependent", subscales_time_dependent},
              },
              {.description = "Flag to allow time dependency of subscales for residual-based "
                              "stabilization.",
                  .default_value = subscales_quasistatic}),

          deprecated_selection<Transient>("TRANSIENT",
              {
                  {"no_transient", inertia_stab_drop},
                  {"yes_transient", inertia_stab_keep},
                  {"transient_complete", inertia_stab_keep_complete},
              },
              {.description = "Specify how to treat the transient term.",
                  .default_value = inertia_stab_drop}),

          parameter<bool>("PSPG",
              {.description = "Flag to (de)activate PSPG stabilization.", .default_value = true}),
          parameter<bool>("SUPG",
              {.description = "Flag to (de)activate SUPG stabilization.", .default_value = true}),
          parameter<bool>("GRAD_DIV",
              {.description = "Flag to (de)activate grad-div term.", .default_value = true}),

          deprecated_selection<VStab>("VSTAB",
              {
                  {"no_vstab", viscous_stab_none},
                  {"vstab_gls", viscous_stab_gls},
                  {"vstab_gls_rhs", viscous_stab_gls_only_rhs},
                  {"vstab_usfem", viscous_stab_usfem},
                  {"vstab_usfem_rhs", viscous_stab_usfem_only_rhs},
              },
              {.description = "Flag to (de)activate viscous term in residual-based stabilization.",
                  .default_value = viscous_stab_none}),

          deprecated_selection<RStab>("RSTAB",
              {
                  {"no_rstab", reactive_stab_none},
                  {"rstab_gls", reactive_stab_gls},
                  {"rstab_usfem", reactive_stab_usfem},
              },
              {.description = "Flag to (de)activate reactive term in residual-based stabilization.",
                  .default_value = reactive_stab_none}),

          deprecated_selection<CrossStress>("CROSS-STRESS",
              {
                  {"no_cross", cross_stress_stab_none},
                  {"yes_cross", cross_stress_stab},
                  {"cross_rhs", cross_stress_stab_only_rhs},
              },
              {.description = "Flag to (de)activate cross-stress term -> residual-based VMM.",
                  .default_value = cross_stress_stab_none}),

          deprecated_selection<ReynoldsStress>("REYNOLDS-STRESS",
              {
                  {"no_reynolds", reynolds_stress_stab_none},
                  {"yes_reynolds", reynolds_stress_stab},
                  {"reynolds_rhs", reynolds_stress_stab_only_rhs},
              },
              {.description = "Flag to (de)activate Reynolds-stress term -> residual-based VMM.",
                  .default_value = reynolds_stress_stab_none}),

          // this parameter selects the tau definition applied
          deprecated_selection<TauType>("DEFINITION_TAU",
              {
                  {"Taylor_Hughes_Zarins", tau_taylor_hughes_zarins},
                  {"Taylor_Hughes_Zarins_wo_dt", tau_taylor_hughes_zarins_wo_dt},
                  {"Taylor_Hughes_Zarins_Whiting_Jansen", tau_taylor_hughes_zarins_whiting_jansen},
                  {"Taylor_Hughes_Zarins_Whiting_Jansen_wo_dt",
                      tau_taylor_hughes_zarins_whiting_jansen_wo_dt},
                  {"Taylor_Hughes_Zarins_scaled", tau_taylor_hughes_zarins_scaled},
                  {"Taylor_Hughes_Zarins_scaled_wo_dt", tau_taylor_hughes_zarins_scaled_wo_dt},
                  {"Franca_Barrenechea_Valentin_Frey_Wall",
                      tau_franca_barrenechea_valentin_frey_wall},
                  {"Franca_Barrenechea_Valentin_Frey_Wall_wo_dt",
                      tau_franca_barrenechea_valentin_frey_wall_wo_dt},
                  {"Shakib_Hughes_Codina", tau_shakib_hughes_codina},
                  {"Shakib_Hughes_Codina_wo_dt", tau_shakib_hughes_codina_wo_dt},
                  {"Codina", tau_codina},
                  {"Codina_wo_dt", tau_codina_wo_dt},
                  {"Franca_Madureira_Valentin_Badia_Codina",
                      tau_franca_madureira_valentin_badia_codina},
                  {"Franca_Madureira_Valentin_Badia_Codina_wo_dt",
                      tau_franca_madureira_valentin_badia_codina_wo_dt},
              },
              {.description = "Definition of tau_M and Tau_C",
                  .default_value = tau_franca_barrenechea_valentin_frey_wall}),

          // this parameter selects the characteristic element length for tau_Mu for all
          // stabilization parameter definitions requiring such a length
          deprecated_selection<CharEleLengthU>("CHARELELENGTH_U",
              {
                  {"streamlength", streamlength_u},
                  {"volume_equivalent_diameter", volume_equivalent_diameter_u},
                  {"root_of_volume", root_of_volume_u},
              },
              {.description = "Characteristic element length for tau_Mu",
                  .default_value = streamlength_u}),

          // this parameter selects the characteristic element length for tau_Mp and tau_C for
          // all stabilization parameter definitions requiring such a length
          deprecated_selection<CharEleLengthPC>("CHARELELENGTH_PC",
              {
                  {"streamlength", streamlength_pc},
                  {"volume_equivalent_diameter", volume_equivalent_diameter_pc},
                  {"root_of_volume", root_of_volume_pc},
              },
              {.description = "Characteristic element length for tau_Mp/tau_C",
                  .default_value = volume_equivalent_diameter_pc}),

          // this parameter selects the location where tau is evaluated
          deprecated_selection<std::string>("EVALUATION_TAU",
              {"element_center", "integration_point"},
              {.description = "Location where tau is evaluated",
                  .default_value = "element_center"}),

          // this parameter selects the location where the material law is evaluated
          // (does not fit here very well, but parameter transfer is easier)
          deprecated_selection<std::string>("EVALUATION_MAT",
              {"element_center", "integration_point"},
              {.description = "Location where material law is evaluated",
                  .default_value = "element_center"}),


          // these parameters active additional terms in loma continuity equation
          // which might be identified as SUPG-/cross- and Reynolds-stress term
          parameter<bool>("LOMA_CONTI_SUPG",
              {.description =
                      "Flag to (de)activate SUPG stabilization in loma continuity equation.",
                  .default_value = false}),

          deprecated_selection<CrossStress>("LOMA_CONTI_CROSS_STRESS",
              {
                  {"no_cross", cross_stress_stab_none},
                  {"yes_cross", cross_stress_stab},
                  {"cross_rhs", cross_stress_stab_only_rhs},
              },
              {.description = "Flag to (de)activate cross-stress term loma continuity equation-> "
                              "residual-based VMM.",
                  .default_value = cross_stress_stab_none}),


          deprecated_selection<ReynoldsStress>("LOMA_CONTI_REYNOLDS_STRESS",
              {
                  {"no_reynolds", reynolds_stress_stab_none},
                  {"yes_reynolds", reynolds_stress_stab},
                  {"reynolds_rhs", reynolds_stress_stab_only_rhs},
              },
              {.description =
                      "Flag to (de)activate Reynolds-stress term loma continuity equation-> "
                      "residual-based VMM.",
                  .default_value = reynolds_stress_stab_none})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  specs.push_back(group("FLUID DYNAMIC/TURBULENCE MODEL",
      {

          //----------------------------------------------------------------------
          // modeling strategies
          //----------------------------------------------------------------------
          deprecated_selection<std::string>("TURBULENCE_APPROACH",
              {"DNS_OR_RESVMM_LES", "CLASSICAL_LES"},
              {.description =
                      "Try to solve flow as an underresolved DNS. Mind that your stabilisation "
                      "already acts as a "
                      "kind of turbulence model! Perform a classical Large Eddy Simulation adding "
                      "addititional "
                      "turbulent viscosity. This may be based on various physical models.)",
                  .default_value = "DNS_OR_RESVMM_LES"}),

          deprecated_selection<std::string>("PHYSICAL_MODEL",
              {"no_model", "Smagorinsky", "Smagorinsky_with_van_Driest_damping",
                  "Dynamic_Smagorinsky", "Multifractal_Subgrid_Scales", "Vreman", "Dynamic_Vreman"},
              {.description =
                      "If classical LES is our turbulence approach, this is a contradiction and "
                      "should "
                      "cause a "
                      "FOUR_C_THROW. Classical constant coefficient Smagorinsky model. Be careful "
                      "if "
                      "you "
                      "have a "
                      "wall bounded flow domain! Use an exponential damping function for the "
                      "turbulent "
                      "viscosity "
                      "close to the wall. This is only implemented for a channel geometry of "
                      "height 2 "
                      "in y "
                      "direction. The viscous lengthscale l_tau is required as additional input. "
                      "The "
                      "solution is "
                      "filtered and by comparison of the filtered velocity field with the real "
                      "solution, "
                      "the "
                      "Smagorinsky constant is estimated in each step --- mind that this procedure "
                      "includes an "
                      "averaging in the xz plane, hence this implementation will only work for a "
                      "channel "
                      "flow. "
                      "Multifractal Subgrid-Scale Modeling based on the work of burton. Vremans "
                      "constant "
                      "model. "
                      "Dynamic Vreman model according to You and Moin (2007)",
                  .default_value = "no_model"}),

          deprecated_selection<FLUID::FineSubgridVisc>("FSSUGRVISC",
              {
                  {"No", no_fssgv},
                  {"Smagorinsky_all", smagorinsky_all},
                  {"Smagorinsky_small", smagorinsky_small},
              },
              {.description = "fine-scale subgrid viscosity", .default_value = no_fssgv}),

          //----------------------------------------------------------------------
          // turbulence specific output and statistics
          //----------------------------------------------------------------------
          parameter<int>(
              "SAMPLING_START", {.description = "Time step after when sampling shall be started",
                                    .default_value = 10000000}),
          parameter<int>("SAMPLING_STOP",
              {.description = "Time step when sampling shall be stopped", .default_value = 1}),
          parameter<int>("DUMPING_PERIOD",
              {.description = "Period of time steps after which statistical data shall be dumped",
                  .default_value = 1}),
          parameter<bool>("SUBGRID_DISSIPATION",
              {.description = "Flag to (de)activate estimation of subgrid-scale "
                              "dissipation (only for seclected flows).",
                  .default_value = false}),
          parameter<bool>(
              "OUTMEAN", {.description = "Flag to (de)activate averaged paraview output",
                             .default_value = false}),
          parameter<bool>("TURBMODEL_LS",
              {.description = "Flag to (de)activate turbulence model in level-set equation",
                  .default_value = true}),

          //----------------------------------------------------------------------
          // turbulent flow problem and general characteristics
          //----------------------------------------------------------------------
          deprecated_selection<std::string>("CANONICAL_FLOW",
              {
                  "no",
                  "time_averaging",
                  "channel_flow_of_height_2",
                  "square_cylinder_nurbs",
                  "rotating_circular_cylinder_nurbs",
                  "rotating_circular_cylinder_nurbs_scatra",
                  "loma_channel_flow_of_height_2",
                  "combust_oracles",
                  "bubbly_channel_flow",
                  "scatra_channel_flow_of_height_2",
                  "decaying_homogeneous_isotropic_turbulence",
                  "forced_homogeneous_isotropic_turbulence",
                  "scatra_forced_homogeneous_isotropic_turbulence",
                  "periodic_hill",
              },
              {.description =
                      ""
                      "Sampling is different for different canonical flows - so specify what kind "
                      "of "
                      "flow you've "
                      "got \n\n"
                      "no: The flow is not further specified, so spatial averaging and hence the "
                      "standard "
                      "sampling procedure is not possible\n"
                      "time_averaging: The flow is not further specified, but time averaging of "
                      "velocity "
                      "and "
                      "pressure field is performed\n"
                      "channel_flow_of_height_2: For this flow, all statistical data could be "
                      "averaged "
                      "in the "
                      "homogeneous planes - it is essentially a statistically one dimensional "
                      "flow.\n"
                      "square_cylinder_nurbs: For this flow, statistical data are evaluated on "
                      "various "
                      "lines of "
                      "the xy-midplane, averaged over time and eventually in one home.direction.\n"
                      "rotating_circular_cylinder_nurbs: For this flow, statistical data is "
                      "computed in "
                      "concentric surfaces and averaged. in time and in one home. direction\n"
                      "rotating_circular_cylinder_nurbs_scatra: For this flow with mass transport, "
                      "statistical "
                      "data is computed in concentric surfaces and averaged. in time and in one "
                      "home. "
                      "direction\n"
                      "loma_channel_flow_of_height_2: For this low-Mach-number flow, all "
                      "statistical "
                      "data could "
                      "be averaged in the homogeneous planes - it is essentially a statistically "
                      "one "
                      "dimensional "
                      "flow.\n"
                      "combust_oracles: ORACLES test rig for turbulent premixed combustion.\n"
                      "bubbly_channel_flow: Turbulent two-phase flow: bubbly channel flow, "
                      "statistical "
                      "data are "
                      "averaged in homogeneous planes and over time.\n"
                      "scatra_channel_flow_of_height_2: For this flow, all statistical data could "
                      "be "
                      "averaged in "
                      "the homogeneous planes - it is essentially a statistically one dimensional "
                      "flow.\n"
                      "decaying_homogeneous_isotropic_turbulence: For this flow, all statistical "
                      "data "
                      "could be "
                      "averaged in the in all homogeneous directions  - it is essentially a "
                      "statistically zero "
                      "dimensional flow.\n"
                      "forced_homogeneous_isotropic_turbulence: For this flow, all statistical "
                      "data "
                      "could be "
                      "averaged in the in all homogeneous directions  - it is essentially a "
                      "statistically zero "
                      "dimensional flow.\n"
                      "scatra_forced_homogeneous_isotropic_turbulence: For this flow, all "
                      "statistical "
                      "data could "
                      "be averaged in the in all homogeneous directions  - it is essentially a "
                      "statistically "
                      "zero dimensional flow.\n"
                      "periodic_hill: For this flow, statistical data is evaluated on various "
                      "lines, "
                      "averaged "
                      "over time and z.\n"
                      "blood_fda_flow: For this flow, statistical data is evaluated on various "
                      "planes.\n",
                  .default_value = "no"}),

          deprecated_selection<std::string>("HOMDIR",
              {"not_specified", "x", "y", "z", "xy", "xz", "yz", "xyz"},
              {.description = "Specify the homogeneous direction(s) of a flow.\n"
                              "not_specified: no homogeneous directions available, averaging is "
                              "restricted to time "
                              "averaging\n"
                              "x: average along x-direction\n"
                              "y: average along y-direction\n"
                              "z: average along z-direction\n"
                              "xy: Wall normal direction is z, average in x and y direction\n"
                              "xz: Wall normal direction is y, average in x and z direction\n"
                              "yz: Wall normal direction is x, average in y and z direction\n"
                              "xyz: Averaging in all directions\n",
                  .default_value = "not_specified"}),

          //---------------------------------------
          // further problem-specific parameters

          // CHANNEL FLOW
          //--------------
          parameter<double>("CHAN_AMPL_INIT_DIST",
              {.description = "Max. amplitude of the random disturbance in percent "
                              "of the initial value in mean flow direction.",
                  .default_value = 0.1}),

          parameter<ForcingType>(
              "FORCING_TYPE", {.description = "forcing strategy",
                                  .default_value = linear_compensation_from_intermediate_spectrum}),

          parameter<int>("CHA_NUMSUBDIVISIONS",
              {.description = "Number of homogeneous sampling planes in element",
                  .default_value = 5}),

          // HIT
          //--------------
          parameter<int>("FORCING_TIME_STEPS",
              {.description = "Number of time steps during which forcing is applied",
                  .default_value = 0}),

          parameter<double>("THRESHOLD_WAVENUMBER",
              {.description = "Forcing is only applied to wave numbers lower or "
                              "equal than the given threshold wave number.",
                  .default_value = 0.0}),


          parameter<double>(
              "POWER_INPUT", {.description = "power of forcing", .default_value = 0.0}),

          deprecated_selection<std::string>("SCALAR_FORCING",
              {"no", "isotropic", "mean_scalar_gradient"},
              {.description =
                      "no: Do not force the scalar field\n"
                      "isotropic: Force scalar field isotropically such as the fluid field.\n"
                      "mean_scalar_gradient: Force scalar field by imposed mean-scalar gradient.\n",
                  .default_value = "no"}),

          parameter<double>("MEAN_SCALAR_GRADIENT",
              {.description = "Value of imposed mean-scalar gradient to force scalar field.",
                  .default_value = 0.0}),

          // filtering with xfem
          //--------------
          parameter<bool>("EXCLUDE_XFEM",
              {.description =
                      "Flag to (de)activate XFEM dofs in calculation of fine-scale velocity.",
                  .default_value = false})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  // sublist with additional input parameters for Smagorinsky model
  specs.push_back(group("FLUID DYNAMIC/SUBGRID VISCOSITY",
      {

          parameter<double>("C_SMAGORINSKY",
              {.description =
                      "Constant for the Smagorinsky model. Something between 0.1 to 0.24. Vreman "
                      "constant "
                      "if the constant vreman model is applied (something between 0.07 and 0.01).",
                  .default_value = 0.0}),
          parameter<double>("C_YOSHIZAWA",
              {.description =
                      "Constant for the compressible Smagorinsky model: isotropic part of "
                      "subgrid-stress tensor. About 0.09 or 0.0066. Ci will not be squared!",
                  .default_value = -1.0}),
          parameter<bool>("C_SMAGORINSKY_AVERAGED",
              {.description = "Flag to (de)activate averaged Smagorinksy constant",
                  .default_value = false}),
          parameter<bool>("C_INCLUDE_CI",
              {.description = "Flag to (de)inclusion of Yoshizawa model", .default_value = false}),
          // remark: following Moin et al. 1991, the extension of the dynamic Smagorinsky model to
          // compressibel flow
          //        also contains a model for the isotropic part of the subgrid-stress tensor
          //        according to Yoshizawa 1989 although used in literature for turbulent
          //        variable-density flow at low Mach number, this additional term seems to
          //        destabilize the simulation when the flow is only weakly compressible therefore
          //        C_INCLUDE_CI allows to exclude this term if C_SMAGORINSKY_AVERAGED == true
          //           if C_INCLUDE_CI==true and C_YOSHIZAWA>=0.0 then the given value C_YOSHIZAWA
          //           is used if C_INCLUDE_CI==true and C_YOSHIZAWA<0.0 then C_YOSHIZAWA is
          //           determined dynamically
          //        else all values are taken from input
          parameter<double>("CHANNEL_L_TAU",
              {.description =
                      "Used for normalisation of the wall normal distance in the Van \nDriest "
                      "Damping function. May be taken from the output of \nthe "
                      "apply_mesh_stretching.pl preprocessing script.",
                  .default_value = 0.0}),

          parameter<double>(
              "C_TURBPRANDTL", {.description = "(Constant) turbulent Prandtl number for the "
                                               "Smagorinsky model in scalar transport.",
                                   .default_value = 1.0}),

          deprecated_selection<VremanFiMethod>("FILTER_WIDTH",
              {
                  {"CubeRootVol", cuberootvol},
                  {"Direction_dependent", dir_dep},
                  {"Minimum_length", min_len},
              },
              {.description = "The Vreman model requires a filter width.",
                  .default_value = cuberootvol})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  // sublist with additional input parameters for Smagorinsky model
  specs.push_back(group("FLUID DYNAMIC/WALL MODEL",
      {

          parameter<bool>("X_WALL",
              {.description = "Flag to switch on the xwall model", .default_value = false}),


          deprecated_selection<std::string>("Tauw_Type", {"constant", "between_steps"},
              {.description =
                      "constant: Use the constant wall shear stress given in the input file "
                      "for the whole "
                      "simulation.\n"
                      "between_steps: Calculate wall shear stress in between time steps.\n",
                  .default_value = "constant"}),

          deprecated_selection<std::string>("Tauw_Calc_Type",
              {"residual", "gradient", "gradient_to_residual"},
              {.description = "residual: Residual (force) divided by area.\n"
                              "gradient: Gradient via shape functions and nodal values.\n"
                              "gradient_to_residual: First gradient, then residual.\n",
                  .default_value = "residual"}),


          parameter<int>("Switch_Step",
              {.description = "Switch from gradient to residual based tauw.", .default_value = -1}),

          deprecated_selection<std::string>("Projection",
              {"No", "only_l2_projection", "l2_projection_with_continuity_constraint"},
              {.description = "Flag to switch projection of the enriched dofs after updating tauw, "
                              "alternatively with or "
                              "without continuity constraint.",
                  .default_value = "No"}),

          parameter<double>("C_Tauw",
              {.description = "Constant wall shear stress for Spalding's law, if applicable",
                  .default_value = 1.0}),

          parameter<double>("Min_Tauw",
              {.description = "Minimum wall shear stress preventing system to become singular",
                  .default_value = 2.0e-9}),

          parameter<double>(
              "Inc_Tauw", {.description = "Increment of Tauw of full step, between 0.0 and 1.0",
                              .default_value = 1.0}),

          deprecated_selection<std::string>("Blending_Type", {"none", "ramp_function"},
              {.description = "Methods for blending the enrichment space.\n"
                              "none: No ramp function, does not converge!\n"
                              "ramp_function: Enrichment is multiplied with linear ramp function "
                              "resulting in zero "
                              "enrichment at the interface.\n",
                  .default_value = "none"}),


          parameter<int>("GP_Wall_Normal",
              {.description = "Gauss points in wall normal direction", .default_value = 3}),
          parameter<int>("GP_Wall_Normal_Off_Wall",
              {.description = "Gauss points in wall normal direction, off-wall elements",
                  .default_value = 3}),
          parameter<int>("GP_Wall_Parallel",
              {.description = "Gauss points in wall parallel direction", .default_value = 3}),

          parameter<bool>("Treat_Tauw_on_Dirichlet_Inflow",
              {.description =
                      "Flag to treat residual on Dirichlet inflow nodes for calculation of wall "
                      "shear stress",
                  .default_value = false}),

          parameter<int>("PROJECTION_SOLVER",
              {.description = "Set solver number for l2-projection.", .default_value = -1})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  // sublist with additional input parameters for multifractal subgrid-scales
  specs.push_back(group("FLUID DYNAMIC/MULTIFRACTAL SUBGRID SCALES",
      {

          parameter<double>(
              "CSGS", {.description = "Modelparameter of multifractal subgrid-scales.",
                          .default_value = 0.0}),

          deprecated_selection<std::string>("SCALE_SEPARATION",
              {"no_scale_sep", "box_filter", "algebraic_multigrid_operator"},
              {.description =
                      "Specify the filter type for scale separation in LES.\n"
                      "no_scale_sep: no scale separation.\n"
                      "box_filter: classical box filter.\n"
                      "algebraic_multigrid_operator: scale separation by algebraic multigrid "
                      "operator.\n",
                  .default_value = "no_scale_sep"}),


          parameter<int>("ML_SOLVER", {.description = "Set solver number for scale separation via "
                                                      "level set transfer operators from "
                                                      "plain aggregation.",
                                          .default_value = -1}),

          parameter<bool>("CALC_N",
              {.description = "Flag to (de)activate calculation of N from the Reynolds number.",
                  .default_value = false}),

          parameter<double>(
              "N", {.description = "Set grid to viscous scale ratio.", .default_value = 1.0}),

          deprecated_selection<std::string>("REF_LENGTH",
              {"cube_edge", "sphere_diameter", "streamlength", "gradient_based", "metric_tensor"},
              {.description = "Specify the reference length for Re-dependent N.\n"
                              "cube_edge: edge length of volume equivalent cube.\n"
                              "sphere_diameter: diameter of volume equivalent sphere.\n"
                              "streamlength: streamlength taken from stabilization.\n"
                              "gradient_based: gradient based length taken from stabilization.\n"
                              "metric_tensor: metric tensor taken from stabilization.\n",
                  .default_value = "cube_edge"}),


          deprecated_selection<std::string>("REF_VELOCITY",
              {"strainrate", "resolved", "fine_scale"},
              {.description = "Specify the reference velocity for Re-dependent N.\n"
                              "strainrate: norm of strain rate.\n"
                              "resolved: resolved velocity.\n"
                              "fine_scale: fine-scale velocity.\n",
                  .default_value = "strainrate"}),


          parameter<double>("C_NU", {.description = "Proportionality constant between Re and ratio "
                                                    "viscous scale to element length.",
                                        .default_value = 1.0}),

          parameter<bool>("NEAR_WALL_LIMIT",
              {.description = "Flag to (de)activate near-wall limit.", .default_value = false}),


          deprecated_selection<std::string>("EVALUATION_B", {"element_center", "integration_point"},
              {.description = "Location where B is evaluated\n"
                              "element_center: evaluate B at element center.\n"
                              "integration_point: evaluate B at integration point.\n",
                  .default_value = "element_center"}),


          parameter<double>(
              "BETA", {.description = "Cross- and Reynolds-stress terms only on right-hand-side.",
                          .default_value = 0.0}),


          deprecated_selection<std::string>("CONVFORM", {"convective", "conservative"},
              {.description = "form of convective term\n"
                              "convective: Use the convective form.\n"
                              "conservative: Use the conservative form.\n",
                  .default_value = "convective"}),

          parameter<double>("CSGS_PHI",
              {.description = "Modelparameter of multifractal subgrid-scales for scalar transport.",
                  .default_value = 0.0}),

          parameter<bool>(
              "ADAPT_CSGS_PHI", {.description = "Flag to (de)activate adaption of CsgsD to CsgsB.",
                                    .default_value = false}),

          parameter<bool>("NEAR_WALL_LIMIT_CSGS_PHI",
              {.description = "Flag to (de)activate near-wall limit for scalar field.",
                  .default_value = false}),

          parameter<bool>("CONSISTENT_FLUID_RESIDUAL",
              {.description =
                      "Flag to (de)activate the consistency term for residual-based stabilization.",
                  .default_value = false}),

          parameter<double>("C_DIFF",
              {.description = "Proportionality constant between Re*Pr and ratio dissipative "
                              "scale to element length. Usually equal cnu.",
                  .default_value = 1.0}),

          parameter<bool>("SET_FINE_SCALE_VEL",
              {.description = "Flag to set fine-scale velocity for parallel nightly tests.",
                  .default_value = false}),

          // activate cross- and Reynolds-stress terms in loma continuity equation
          parameter<bool>("LOMA_CONTI",
              {.description = "Flag to (de)activate cross- and Reynolds-stress terms in "
                              "loma continuity equation.",
                  .default_value = false})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  specs.push_back(group("FLUID DYNAMIC/TURBULENT INFLOW",
      {

          parameter<bool>("TURBULENTINFLOW",
              {.description = "Flag to (de)activate potential separate turbulent inflow section",
                  .default_value = false}),

          deprecated_selection<InitialField>("INITIALINFLOWFIELD",
              {
                  {"zero_field", initfield_zero_field},
                  {"field_by_function", initfield_field_by_function},
                  {"disturbed_field_from_function", initfield_disturbed_field_from_function},
              },
              {.description = "Initial field for inflow section",
                  .default_value = initfield_zero_field}),

          parameter<int>("INFLOWFUNC",
              {.description = "Function number for initial flow field in inflow section",
                  .default_value = -1}),

          parameter<double>("INFLOW_INIT_DIST",
              {.description = "Max. amplitude of the random disturbance in percent of "
                              "the initial value in mean flow direction.",
                  .default_value = 0.1}),

          parameter<int>("NUMINFLOWSTEP",
              {.description = "Total number of time steps for development of turbulent flow",
                  .default_value = 1}),

          deprecated_selection<std::string>("CANONICAL_INFLOW",
              {"no", "time_averaging", "channel_flow_of_height_2", "loma_channel_flow_of_height_2",
                  "scatra_channel_flow_of_height_2"},
              {.description =
                      "Sampling is different for different canonical flows \n--- so specify what "
                      "kind "
                      "of "
                      "flow "
                      "you've got\n"
                      "no: The flow is not further specified, so spatial averaging \nand hence the "
                      "standard "
                      "sampling procedure is not possible.\n"
                      "time_averaging: The flow is not further specified, but time averaging of "
                      "velocity "
                      "and "
                      "pressure field is performed.\n"
                      "channel_flow_of_height_2: For this flow, all statistical data could be "
                      "averaged "
                      "in "
                      "\nthe "
                      "homogeneous planes --- it is essentially a statistically one dimensional "
                      "flow.\n"
                      "loma_channel_flow_of_height_2: For this low-Mach-number flow, all "
                      "statistical "
                      "data "
                      "could be "
                      "averaged in \nthe homogeneous planes --- it is essentially a statistically "
                      "one "
                      "dimensional "
                      "flow.\n"
                      "scatra_channel_flow_of_height_2: For this flow, all statistical data could "
                      "be "
                      "averaged in "
                      "\nthe homogeneous planes --- it is essentially a statistically one "
                      "dimensional "
                      "flow.\n",
                  .default_value = "no"}),


          parameter<double>("INFLOW_CHA_SIDE",
              {.description =
                      "Most right side of inflow channel. Necessary to define sampling domain.",
                  .default_value = 0.0}),

          deprecated_selection<std::string>("INFLOW_HOMDIR",
              {"not_specified", "x", "y", "z", "xy", "xz", "yz"},
              {.description = "Specify the homogeneous direction(s) of a flow\n"
                              "not_specified: no homogeneous directions available, averaging is "
                              "restricted to "
                              "time "
                              "averaging.\n"
                              "x: average along x-direction.\n"
                              "y: average along y-direction.\n"
                              "z: average along z-direction.\n"
                              "xy: Wall normal direction is z, average in x and y direction.\n"
                              "xz: Wall normal direction is y, average in x and z direction "
                              "(standard case).\n"
                              "yz: Wall normal direction is x, average in y and z direction.\n",
                  .default_value = "not_specified"}),

          parameter<int>("INFLOW_SAMPLING_START",
              {.description = "Time step after when sampling shall be started",
                  .default_value = 10000000}),
          parameter<int>("INFLOW_SAMPLING_STOP",
              {.description = "Time step when sampling shall be stopped", .default_value = 1}),
          parameter<int>("INFLOW_DUMPING_PERIOD",
              {.description = "Period of time steps after which statistical data shall be dumped",
                  .default_value = 1})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  // sublist with additional input parameters for time adaptivity in fluid/ coupled problems
  specs.push_back(group("FLUID DYNAMIC/TIMEADAPTIVITY",
      {

          deprecated_selection<AdaptiveTimeStepEstimator>("ADAPTIVE_TIME_STEP_ESTIMATOR",
              {
                  {"none", const_dt},
                  {"cfl_number", cfl_number},
                  {"only_print_cfl_number", only_print_cfl_number},
              },
              {.description = "Method used to determine adaptive time step size.",
                  .default_value = const_dt}),

          parameter<double>("CFL_NUMBER",
              {.description = "CFL number for adaptive time step", .default_value = -1.0}),
          parameter<int>("FREEZE_ADAPTIVE_DT_AT",
              {.description =
                      "keep time step constant after this step, otherwise turbulence statistics "
                      "sampling is not consistent",
                  .default_value = 1000000}),
          parameter<double>(
              "ADAPTIVE_DT_INC", {.description = "Increment of whole step for adaptive dt via CFL",
                                     .default_value = 0.8})},
      {.required = false}));

  /*----------------------------------------------------------------------*/
  // sublist with runtime output parameters
  specs.push_back(group("IO/RUNTIME VTK OUTPUT/FLUID",
      {
          // whether to write output for fluid
          parameter<bool>(
              "OUTPUT_FLUID", {.description = "write fluid output", .default_value = false}),

          // whether to write velocity state
          parameter<bool>(
              "VELOCITY", {.description = "write velocity output", .default_value = false}),

          // whether to write pressure state
          parameter<bool>(
              "PRESSURE", {.description = "write pressure output", .default_value = false}),

          // whether to write acceleration state
          parameter<bool>(
              "ACCELERATION", {.description = "write acceleration output", .default_value = false}),

          // whether to write displacement state
          parameter<bool>(
              "DISPLACEMENT", {.description = "write displacement output", .default_value = false}),

          // whether to write displacement state
          parameter<bool>("GRIDVELOCITY",
              {.description = "write grid velocity output", .default_value = false}),

          // whether to write element owner
          parameter<bool>(
              "ELEMENT_OWNER", {.description = "write element owner", .default_value = false}),

          // whether to write element GIDs
          parameter<bool>("ELEMENT_GID",
              {.description = "write 4C internal element GIDs", .default_value = false}),

          // whether to write node GIDs
          parameter<bool>(
              "NODE_GID", {.description = "write 4C internal node GIDs", .default_value = false}),
      },
      {.required = false}));

  return specs;
}


void FLUID::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*--------------------------------------------------------------------*/
  // transfer boundary condition for turbulent inflow

  Core::Conditions::ConditionDefinition tbc_turb_inflow("DESIGN SURF TURBULENT INFLOW TRANSFER",
      "TransferTurbulentInflow", "TransferTurbulentInflow",
      Core::Conditions::TransferTurbulentInflow, true, Core::Conditions::geometry_type_surface);

  tbc_turb_inflow.add_component(parameter<int>("ID", {.description = ""}));
  tbc_turb_inflow.add_component(
      deprecated_selection<std::string>("toggle", {"master", "slave"}, {.description = "toggle"}));
  tbc_turb_inflow.add_component(parameter<TurbInflowDirection>(
      "DIRECTION", {.description = "transfer direction", .default_value = TurbInflowDirection::x}));
  tbc_turb_inflow.add_component(
      parameter<std::optional<int>>("curve", {.description = "curve id"}));

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
  surfflowdeppressure.add_component(deprecated_selection<std::string>("TYPE_OF_FLOW_DEPENDENCE",
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
  surfflowdeppressure.add_component(
      parameter<std::optional<int>>("curve", {.description = "curve id"}));

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
    cond.add_component(parameter<std::vector<std::optional<int>>>(
        "funct", {.description = "spatial function",
                     .default_value = std::vector(3, std::optional<int>{}),
                     .size = 3}));

    // characteristic velocity
    cond.add_component(parameter<double>("u_C"));

    // the penalty parameter could be computed dynamically (using Spaldings
    // law of the wall) or using a fixed value (1)
    cond.add_component(deprecated_selection<std::string>(
        "PENTYPE", {"constant", "Spalding"}, {.description = "how penalty parameter is computed"}));

    // scaling factor for penalty parameter tauB
    cond.add_component(parameter<double>("hB_divided_by"));

    // if Spaldings law is used, this defines the way how the traction at y is computed from utau
    cond.add_component(
        deprecated_selection<std::string>("utau_computation", {"at_wall", "viscous_tangent"},
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
      deprecated_selection<std::string>("ConditionType", {"POLYNOMIAL", "WOMERSLEY"},
          {.description = "condition type", .default_value = "POLYNOMIAL"}));

  volumetric_surface_flow_cond.add_component(
      deprecated_selection<std::string>("prebiased", {"NOTPREBIASED", "PREBIASED", "FORCED"},
          {.description = "prebiased", .default_value = "NOTPREBIASED"}));

  volumetric_surface_flow_cond.add_component(deprecated_selection<std::string>(
      "FlowType", {"InFlow", "OutFlow"}, {.description = "flow type", .default_value = "InFlow"}));
  volumetric_surface_flow_cond.add_component(
      deprecated_selection<std::string>("CorrectionFlag", {"WithOutCorrection", "WithCorrection"},
          {.description = "correction flag", .default_value = "WithOutCorrection"}));

  volumetric_surface_flow_cond.add_component(parameter<double>("Period"));
  volumetric_surface_flow_cond.add_component(parameter<int>("Order"));
  volumetric_surface_flow_cond.add_component(parameter<int>("Harmonics"));
  volumetric_surface_flow_cond.add_component(parameter<double>("Val"));
  volumetric_surface_flow_cond.add_component(parameter<int>("Funct"));

  volumetric_surface_flow_cond.add_component(
      deprecated_selection<std::string>("NORMAL", {"SelfEvaluateNormal", "UsePrescribedNormal"},
          {.description = "normal", .default_value = "SelfEvaluateNormal"}));
  volumetric_surface_flow_cond.add_component(parameter<double>("n1"));
  volumetric_surface_flow_cond.add_component(parameter<double>("n2"));
  volumetric_surface_flow_cond.add_component(parameter<double>("n3"));

  volumetric_surface_flow_cond.add_component(deprecated_selection<std::string>("CenterOfMass",
      {"SelfEvaluateCenterOfMass", "UsePrescribedCenterOfMass"},
      {.description = "center of mass", .default_value = "SelfEvaluateCenterOfMass"}));
  volumetric_surface_flow_cond.add_component(parameter<double>("c1"));
  volumetric_surface_flow_cond.add_component(parameter<double>("c2"));
  volumetric_surface_flow_cond.add_component(parameter<double>("c3"));

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
      deprecated_selection<std::string>("ConditionType", {"POLYNOMIAL", "WOMERSLEY"},
          {.description = "condition type", .default_value = "POLYNOMIAL"}));

  total_traction_correction_cond.add_component(
      deprecated_selection<std::string>("prebiased", {"NOTPREBIASED", "PREBIASED", "FORCED"},
          {.description = "prebiased", .default_value = "NOTPREBIASED"}));

  total_traction_correction_cond.add_component(deprecated_selection<std::string>(
      "FlowType", {"InFlow", "OutFlow"}, {.description = "flow type", .default_value = "InFlow"}));
  total_traction_correction_cond.add_component(
      deprecated_selection<std::string>("CorrectionFlag", {"WithOutCorrection", "WithCorrection"},
          {.description = "correction flag", .default_value = "WithOutCorrection"}));

  total_traction_correction_cond.add_component(parameter<double>("Period"));
  total_traction_correction_cond.add_component(parameter<int>("Order"));
  total_traction_correction_cond.add_component(parameter<int>("Harmonics"));
  total_traction_correction_cond.add_component(parameter<double>("Val"));
  total_traction_correction_cond.add_component(parameter<int>("Funct"));

  total_traction_correction_cond.add_component(
      deprecated_selection<std::string>("NORMAL", {"SelfEvaluateNormal", "UsePrescribedNormal"},
          {.description = "normal", .default_value = "SelfEvaluateNormal"}));
  total_traction_correction_cond.add_component(parameter<double>("n1"));
  total_traction_correction_cond.add_component(parameter<double>("n2"));
  total_traction_correction_cond.add_component(parameter<double>("n3"));

  total_traction_correction_cond.add_component(deprecated_selection<std::string>("CenterOfMass",
      {"SelfEvaluateCenterOfMass", "UsePrescribedCenterOfMass"},
      {.description = "center of mass", .default_value = "SelfEvaluateCenterOfMass"}));
  total_traction_correction_cond.add_component(parameter<double>("c1"));
  total_traction_correction_cond.add_component(parameter<double>("c2"));
  total_traction_correction_cond.add_component(parameter<double>("c3"));

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

  /*--------------------------------------------------------------------*/
  // Impedance condition

  Core::Conditions::ConditionDefinition impedancebc("DESIGN SURF IMPEDANCE CONDITIONS",
      "ImpedanceCond", "Impedance boundary condition", Core::Conditions::ImpedanceCond, true,
      Core::Conditions::geometry_type_surface);

  impedancebc.add_component(parameter<int>("ConditionID"));
  impedancebc.add_component(
      deprecated_selection<std::string>("TYPE", {"windkessel", "resistive", "pressure_by_funct"},
          {.description = "type", .default_value = "windkessel"}));
  impedancebc.add_component(parameter<double>("R1"));
  impedancebc.add_component(parameter<double>("R2"));
  impedancebc.add_component(parameter<double>("C"));
  impedancebc.add_component(parameter<double>("TIMEPERIOD"));
  impedancebc.add_component(parameter<int>("FUNCT"));

  condlist.push_back(impedancebc);
}

std::string FLUID::to_string(FLUID::StabType stabtype)
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

std::string FLUID::to_string(FLUID::TauType tau)
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

std::string FLUID::to_string(FLUID::CrossStress crossstress)
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

std::string FLUID::to_string(FLUID::VStab vstab)
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

std::string FLUID::to_string(FLUID::RStab rstab)
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

std::string FLUID::to_string(FLUID::ReynoldsStress reynoldsstress)
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

std::string FLUID::to_string(FLUID::Transient transient)
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

std::string FLUID::to_string(FLUID::EosPres eospres)
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

std::string FLUID::to_string(FLUID::EosConvStream eosconvstream)
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

std::string FLUID::to_string(FLUID::EosConvCross eosconvcross)
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

std::string FLUID::to_string(FLUID::EosDiv eosdiv)
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

std::string FLUID::to_string(FLUID::EosTauType eostautype)
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

std::string FLUID::to_string(FLUID::EosElementLength eoselementlength)
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

std::string FLUID::to_string(FLUID::VremanFiMethod vremanfimethod)
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