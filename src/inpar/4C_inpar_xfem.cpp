// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_xfem.hpp"

#include "4C_cut_enum.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_input_types.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::XFEM::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;

  Core::Utils::SectionSpecs xfem_general{"XFEM GENERAL"};

  // OUTPUT options
  Core::Utils::bool_parameter("GMSH_DEBUG_OUT", false,
      "Do you want to write extended Gmsh output for each timestep?", xfem_general);
  Core::Utils::bool_parameter("GMSH_DEBUG_OUT_SCREEN", false,
      "Do you want to be informed, if Gmsh output is written?", xfem_general);
  Core::Utils::bool_parameter("GMSH_SOL_OUT", false,
      "Do you want to write extended Gmsh output for each timestep?", xfem_general);
  Core::Utils::bool_parameter("GMSH_TIMINT_OUT", false,
      "Do you want to write extended Gmsh output for each timestep?", xfem_general);
  Core::Utils::bool_parameter("GMSH_EOS_OUT", false,
      "Do you want to write extended Gmsh output for each timestep?", xfem_general);
  Core::Utils::bool_parameter("GMSH_DISCRET_OUT", false,
      "Do you want to write extended Gmsh output for each timestep?", xfem_general);
  Core::Utils::bool_parameter("GMSH_CUT_OUT", false,
      "Do you want to write extended Gmsh output for each timestep?", xfem_general);
  Core::Utils::bool_parameter(
      "PRINT_OUTPUT", false, "Is the output of the cut process desired?", xfem_general);

  Core::Utils::int_parameter(
      "MAX_NUM_DOFSETS", 3, "Maximum number of volumecells in the XFEM element", xfem_general);

  Core::Utils::string_to_integral_parameter<Cut::NodalDofSetStrategy>("NODAL_DOFSET_STRATEGY",
      "full", "Strategy used for the nodal dofset management per node",
      tuple<std::string>(
          "OneDofset_PerNodeAndPosition", "ConnectGhostDofsets_PerNodeAndPosition", "full"),
      tuple<Cut::NodalDofSetStrategy>(Cut::NDS_Strategy_OneDofset_PerNodeAndPosition,
          Cut::NDS_Strategy_ConnectGhostDofsets_PerNodeAndPosition, Cut::NDS_Strategy_full),
      xfem_general);

  // Integration options
  Core::Utils::string_to_integral_parameter<Cut::VCellGaussPts>("VOLUME_GAUSS_POINTS_BY",
      "Tessellation", "Method for finding Gauss Points for the cut volumes",
      tuple<std::string>("Tessellation", "MomentFitting", "DirectDivergence"),
      tuple<Cut::VCellGaussPts>(Cut::VCellGaussPts_Tessellation, Cut::VCellGaussPts_MomentFitting,
          Cut::VCellGaussPts_DirectDivergence),
      xfem_general);

  Core::Utils::string_to_integral_parameter<Cut::BCellGaussPts>("BOUNDARY_GAUSS_POINTS_BY",
      "Tessellation", "Method for finding Gauss Points for the boundary cells",
      tuple<std::string>("Tessellation", "MomentFitting", "DirectDivergence"),
      tuple<Cut::BCellGaussPts>(Cut::BCellGaussPts_Tessellation, Cut::BCellGaussPts_MomentFitting,
          Cut::BCellGaussPts_DirectDivergence),
      xfem_general);

  xfem_general.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs xfluid_dyn{"XFLUID DYNAMIC"};

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs xfluid_general{xfluid_dyn, "GENERAL"};

  // Do we use more than one fluid discretization?
  Core::Utils::bool_parameter("XFLUIDFLUID", false, "Use an embedded fluid patch.", xfluid_general);

  // How many monolithic steps we keep the fluidfluid-boundary fixed
  Core::Utils::int_parameter(
      "RELAXING_ALE_EVERY", 1, "Relaxing Ale after how many monolithic steps", xfluid_general);

  Core::Utils::bool_parameter("RELAXING_ALE", true,
      "switch on/off for relaxing Ale in monolithic fluid-fluid-fsi", xfluid_general);

  Core::Utils::double_parameter(
      "XFLUIDFLUID_SEARCHRADIUS", 1.0, "Radius of the search tree", xfluid_general);

  // xfluidfluid-fsi-monolithic approach
  Core::Utils::string_to_integral_parameter<MonolithicXffsiApproach>("MONOLITHIC_XFFSI_APPROACH",
      "xffsi_fixedALE_partitioned", "The monolithic approach for xfluidfluid-fsi",
      tuple<std::string>(
          "xffsi_full_newton", "xffsi_fixedALE_interpolation", "xffsi_fixedALE_partitioned"),
      tuple<MonolithicXffsiApproach>(
          Inpar::XFEM::XFFSI_Full_Newton,             // xffsi with no fixed xfem-coupling
          Inpar::XFEM::XFFSI_FixedALE_Interpolation,  // xffsi with fixed xfem-coupling in every
                                                      // newtonstep and interpolations for
                                                      // embedded-dis afterwards
          Inpar::XFEM::XFFSI_FixedALE_Partitioned     // xffsi with fixed xfem-coupling in every
                                                      // newtonstep and solving fluid-field again
          ),
      xfluid_general);

  // xfluidfluid time integration approach
  Core::Utils::string_to_integral_parameter<XFluidFluidTimeInt>("XFLUIDFLUID_TIMEINT",
      "Xff_TimeInt_FullProj", "The xfluidfluid-timeintegration approach",
      tuple<std::string>("Xff_TimeInt_FullProj", "Xff_TimeInt_ProjIfMoved",
          "Xff_TimeInt_KeepGhostValues", "Xff_TimeInt_IncompProj"),
      tuple<XFluidFluidTimeInt>(Inpar::XFEM::Xff_TimeInt_FullProj,  // always project nodes from
                                                                    // embedded to background nodes
          Inpar::XFEM::Xff_TimeInt_ProjIfMoved,  // project nodes just if the status of background
                                                 // nodes changed
          Inpar::XFEM::Xff_TimeInt_KeepGhostValues,  // always keep the ghost values of the
                                                     // background discretization
          Inpar::XFEM::Xff_TimeInt_IncompProj  // after projecting nodes do a incompressibility
                                               // projection
          ),
      xfluid_general);

  Core::Utils::string_to_integral_parameter<XFluidTimeIntScheme>("XFLUID_TIMEINT",
      "STD=COPY/SL_and_GHOST=COPY/GP", "The xfluid time integration approach",
      tuple<std::string>("STD=COPY_and_GHOST=COPY/GP", "STD=COPY/SL_and_GHOST=COPY/GP",
          "STD=SL(boundary-zone)_and_GHOST=GP", "STD=COPY/PROJ_and_GHOST=COPY/PROJ/GP"),
      tuple<XFluidTimeIntScheme>(
          Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_AND_GHOST_by_Copy_or_GP,  // STD= only copy,
                                                                              // GHOST= copy or
                                                                              // ghost penalty
                                                                              // reconstruction
          Inpar::XFEM::
              Xf_TimeIntScheme_STD_by_Copy_or_SL_AND_GHOST_by_Copy_or_GP,  // STD= copy or SL,
                                                                           // GHOST= copy or ghost
                                                                           // penalty reconstruction
          Inpar::XFEM::Xf_TimeIntScheme_STD_by_SL_cut_zone_AND_GHOST_by_GP,  // STD= only SL on
                                                                             // whole boundary zone,
                                                                             // GHOST= ghost penalty
                                                                             // reconstruction
          Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_Proj_AND_GHOST_by_Proj_or_Copy_or_GP),
      xfluid_general);

  Core::Utils::bool_parameter("ALE_XFluid", false, "XFluid is Ale Fluid?", xfluid_general);

  // for new OST-implementation: which interface terms to be evaluated for previous time step
  Core::Utils::string_to_integral_parameter<InterfaceTermsPreviousState>(
      "INTERFACE_TERMS_PREVIOUS_STATE", "PreviousState_only_consistency",
      "how to treat interface terms from previous time step (new OST)",
      tuple<std::string>("PreviousState_only_consistency", "PreviousState_full"),
      tuple<InterfaceTermsPreviousState>(
          Inpar::XFEM::PreviousState_only_consistency,  /// evaluate only consistency terms
                                                        /// for previous time step
          Inpar::XFEM::PreviousState_full  /// evaluate consistency, adjoint consistency and penalty
                                           /// terms or previous time step
          ),
      xfluid_general);

  Core::Utils::bool_parameter("XFLUID_TIMEINT_CHECK_INTERFACETIPS", true,
      "Xfluid TimeIntegration Special Check if node has changed the side!", xfluid_general);

  Core::Utils::bool_parameter("XFLUID_TIMEINT_CHECK_SLIDINGONSURFACE", true,
      "Xfluid TimeIntegration Special Check if node is sliding on surface!", xfluid_general);

  xfluid_general.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs xfluid_stab{xfluid_dyn, "STABILIZATION"};

  // Boundary-Coupling options
  Core::Utils::string_to_integral_parameter<CouplingMethod>("COUPLING_METHOD", "Nitsche",
      "method how to enforce embedded boundary/coupling conditions at the interface",
      tuple<std::string>("Hybrid_LM_Cauchy_stress", "Hybrid_LM_viscous_stress", "Nitsche"),
      tuple<CouplingMethod>(
          Inpar::XFEM::Hybrid_LM_Cauchy_stress,   // Cauchy stress-based mixed/hybrid formulation
          Inpar::XFEM::Hybrid_LM_viscous_stress,  // viscous stress-based mixed/hybrid formulation
          Inpar::XFEM::Nitsche                    // Nitsche's formulation
          ),
      xfluid_stab);

  Core::Utils::string_to_integral_parameter<HybridLmL2Proj>("HYBRID_LM_L2_PROJ", "part_ele_proj",
      "perform the L2 projection between stress fields on whole element or on fluid part?",
      tuple<std::string>("full_ele_proj", "part_ele_proj"),
      tuple<HybridLmL2Proj>(
          Inpar::XFEM::Hybrid_LM_L2_Proj_full,  // L2 stress projection on whole fluid element
          Inpar::XFEM::Hybrid_LM_L2_Proj_part   // L2 stress projection on partial fluid element
                                                // volume
          ),
      xfluid_stab);

  Core::Utils::string_to_integral_parameter<AdjointScaling>("VISC_ADJOINT_SYMMETRY", "yes",
      "viscous and adjoint viscous interface terms with matching sign?",
      tuple<std::string>("yes", "no", "sym", "skew", "none"),
      tuple<AdjointScaling>(Inpar::XFEM::adj_sym, Inpar::XFEM::adj_skew, Inpar::XFEM::adj_sym,
          Inpar::XFEM::adj_skew, Inpar::XFEM::adj_none),
      xfluid_stab);

  // viscous and convective Nitsche/MSH stabilization parameter
  Core::Utils::double_parameter(
      "NIT_STAB_FAC", 35.0, " ( stabilization parameter for Nitsche's penalty term", xfluid_stab);
  Core::Utils::double_parameter("NIT_STAB_FAC_TANG", 35.0,
      " ( stabilization parameter for Nitsche's penalty tangential term", xfluid_stab);

  Core::Utils::string_to_integral_parameter<ViscStabTraceEstimate>("VISC_STAB_TRACE_ESTIMATE",
      "CT_div_by_hk", "how to estimate the scaling from the trace inequality in Nitsche's method",
      tuple<std::string>("CT_div_by_hk", "eigenvalue"),
      tuple<ViscStabTraceEstimate>(
          Inpar::XFEM::ViscStab_TraceEstimate_CT_div_by_hk,  // estimate the trace inequality by a
                                                             // trace-constant CT and hk: CT/hk
          Inpar::XFEM::ViscStab_TraceEstimate_eigenvalue     // estimate the trace inequality by
                                                          // solving a local element-wise eigenvalue
                                                          // problem
          ),
      xfluid_stab);

  Core::Utils::string_to_integral_parameter<TraceEstimateEigenvalueUpdate>(
      "UPDATE_EIGENVALUE_TRACE_ESTIMATE", "every_iter",
      "how often should the local eigenvalue problem be updated",
      tuple<std::string>("every_iter", "every_timestep", "once"),
      tuple<TraceEstimateEigenvalueUpdate>(Inpar::XFEM::Eigenvalue_update_every_iter,
          Inpar::XFEM::Eigenvalue_update_every_timestep, Inpar::XFEM::Eigenvalue_update_once),
      xfluid_stab);

  Core::Utils::string_to_integral_parameter<ViscStabHk>("VISC_STAB_HK",
      "ele_vol_div_by_max_ele_surf",
      "how to define the characteristic element length in cut elements",
      tuple<std::string>("vol_equivalent", "cut_vol_div_by_cut_surf", "ele_vol_div_by_cut_surf",
          "ele_vol_div_by_ele_surf", "ele_vol_div_by_max_ele_surf"),
      tuple<ViscStabHk>(
          Inpar::XFEM::ViscStab_hk_vol_equivalent,           /// volume equivalent element diameter
          Inpar::XFEM::ViscStab_hk_cut_vol_div_by_cut_surf,  /// physical partial/cut volume divided
                                                             /// by physical partial/cut surface
                                                             /// measure ( used to estimate the
                                                             /// cut-dependent inverse estimate on
                                                             /// cut elements, not useful for sliver
                                                             /// and/or dotted cut situations)
          Inpar::XFEM::ViscStab_hk_ele_vol_div_by_cut_surf,  /// full element volume divided by
                                                             /// physical partial/cut surface
                                                             /// measure ( used to estimate the
                                                             /// cut-dependent inverse estimate on
                                                             /// cut elements, however, avoids
                                                             /// problems with sliver cuts, not
                                                             /// useful for dotted cuts)
          Inpar::XFEM::ViscStab_hk_ele_vol_div_by_ele_surf,  /// full element volume divided by
                                                             /// surface measure ( used for uncut
                                                             /// situations, standard weak Dirichlet
                                                             /// boundary/coupling conditions)
          Inpar::XFEM::
              ViscStab_hk_ele_vol_div_by_max_ele_surf  /// default: full element volume divided by
                                                       /// maximal element surface measure ( used to
                                                       /// estimate the trace inequality for
                                                       /// stretched elements in combination with
                                                       /// ghost-penalties)
          ),
      xfluid_stab);


  Core::Utils::string_to_integral_parameter<ConvStabScaling>("CONV_STAB_SCALING", "none",
      "scaling factor for viscous interface stabilization (Nitsche, MSH)",
      tuple<std::string>("inflow", "abs_inflow", "none"),
      tuple<ConvStabScaling>(Inpar::XFEM::ConvStabScaling_inflow,  // scaling with max(0,-u*n)
          Inpar::XFEM::ConvStabScaling_abs_inflow,                 // scaling with |u*n|
          Inpar::XFEM::ConvStabScaling_none                        // no convective stabilization
          ),
      xfluid_stab);

  Core::Utils::string_to_integral_parameter<XffConvStabScaling>("XFF_CONV_STAB_SCALING", "none",
      "scaling factor for convective interface stabilization of fluid-fluid Coupling",
      tuple<std::string>("inflow", "averaged", "none"),
      tuple<XffConvStabScaling>(
          Inpar::XFEM::XFF_ConvStabScaling_upwinding,      // one-sided inflow stabilization
          Inpar::XFEM::XFF_ConvStabScaling_only_averaged,  // averaged inflow stabilization
          Inpar::XFEM::XFF_ConvStabScaling_none            // no convective stabilization
          ),
      xfluid_stab);

  Core::Utils::string_to_integral_parameter<MassConservationCombination>("MASS_CONSERVATION_COMBO",
      "max", "choose the maximum from viscous and convective contributions or just sum both up",
      tuple<std::string>("max", "sum"),
      tuple<MassConservationCombination>(
          Inpar::XFEM::MassConservationCombination_max,  /// use the maximum contribution
          Inpar::XFEM::MassConservationCombination_sum  /// sum viscous and convective contributions
          ),
      xfluid_stab);

  Core::Utils::string_to_integral_parameter<MassConservationScaling>("MASS_CONSERVATION_SCALING",
      "only_visc",
      "apply additional scaling of penalty term to enforce mass conservation for "
      "convection-dominated flow",
      tuple<std::string>("full", "only_visc"),
      tuple<MassConservationScaling>(
          Inpar::XFEM::MassConservationScaling_full,  /// apply mass-conserving convective scaling
                                                      /// additionally
          Inpar::XFEM::MassConservationScaling_only_visc  /// use only the viscous scaling
          ),
      xfluid_stab);

  Core::Utils::bool_parameter("GHOST_PENALTY_STAB", false,
      "switch on/off ghost penalty interface stabilization", xfluid_stab);

  Core::Utils::bool_parameter("GHOST_PENALTY_TRANSIENT_STAB", false,
      "switch on/off ghost penalty transient interface stabilization", xfluid_stab);

  Core::Utils::bool_parameter("GHOST_PENALTY_2nd_STAB", false,
      "switch on/off ghost penalty interface stabilization for 2nd order derivatives", xfluid_stab);
  Core::Utils::bool_parameter("GHOST_PENALTY_2nd_STAB_NORMAL", false,
      "switch between ghost penalty interface stabilization for 2nd order derivatives in normal or "
      "all spatial directions",
      xfluid_stab);


  Core::Utils::double_parameter("GHOST_PENALTY_FAC", 0.1,
      "define stabilization parameter ghost penalty interface stabilization", xfluid_stab);

  Core::Utils::double_parameter("GHOST_PENALTY_TRANSIENT_FAC", 0.001,
      "define stabilization parameter ghost penalty transient interface stabilization",
      xfluid_stab);

  Core::Utils::double_parameter("GHOST_PENALTY_2nd_FAC", 0.05,
      "define stabilization parameter ghost penalty 2nd order viscous interface stabilization",
      xfluid_stab);
  Core::Utils::double_parameter("GHOST_PENALTY_PRESSURE_2nd_FAC", 0.05,
      "define stabilization parameter ghost penalty 2nd order pressure interface stabilization",
      xfluid_stab);


  Core::Utils::bool_parameter("XFF_EOS_PRES_EMB_LAYER", false,
      "switch on/off edge-based pressure stabilization on interface-contributing elements of the "
      "embedded fluid",
      xfluid_stab);

  Core::Utils::bool_parameter("IS_PSEUDO_2D", false,
      "modify viscous interface stabilization due to the vanishing polynomial in third dimension "
      "when using strong Dirichlet conditions to block polynomials in one spatial dimension",
      xfluid_stab);

  Core::Utils::bool_parameter("GHOST_PENALTY_ADD_INNER_FACES", false,
      "Apply ghost penalty stabilization also for inner faces if this is possible due to the "
      "dofsets",
      xfluid_stab);

  xfluid_stab.move_into_collection(list);

  /*----------------------------------------------------------------------*/
  Core::Utils::SectionSpecs xfsi_monolithic{xfluid_dyn, "XFPSI MONOLITHIC"};

  Core::Utils::int_parameter(
      "ITEMIN", 1, "How many iterations are performed minimal", xfsi_monolithic);
  Core::Utils::int_parameter(
      "ITEMAX_OUTER", 5, "How many outer iterations are performed maximal", xfsi_monolithic);
  Core::Utils::bool_parameter("ND_NEWTON_DAMPING", false,
      "Activate Newton damping based on residual and increment", xfsi_monolithic);
  Core::Utils::double_parameter("ND_MAX_DISP_ITERINC", -1.0,
      "Maximal displacement increment to apply full newton --> otherwise damp newton",
      xfsi_monolithic);
  Core::Utils::double_parameter("ND_MAX_VEL_ITERINC", -1.0,
      "Maximal fluid velocity increment to apply full newton --> otherwise damp newton",
      xfsi_monolithic);
  Core::Utils::double_parameter("ND_MAX_PRES_ITERINC", -1.0,
      "Maximal fluid pressure increment to apply full newton --> otherwise damp newton",
      xfsi_monolithic);
  Core::Utils::double_parameter("ND_MAX_PVEL_ITERINC", -1.0,
      "Maximal porofluid velocity increment to apply full newton --> otherwise damp newton",
      xfsi_monolithic);
  Core::Utils::double_parameter("ND_MAX_PPRES_ITERINC", -1.0,
      "Maximal porofluid pressure increment to apply full newton --> otherwise damp newton",
      xfsi_monolithic);
  Core::Utils::double_parameter("CUT_EVALUATE_MINTOL", 0.0,
      "Minimal value of the maximal structural displacement for which the CUT is evaluate in this "
      "iteration!",
      xfsi_monolithic);
  Core::Utils::int_parameter("CUT_EVALUATE_MINITER", 0,
      "Minimal number of nonlinear iterations, before the CUT is potentially not evaluated",
      xfsi_monolithic);
  Core::Utils::bool_parameter("EXTRAPOLATE_TO_ZERO", false,
      "the extrapolation of the fluid stress in the contact zone is relaxed to zero after a "
      "certain distance",
      xfsi_monolithic);
  Core::Utils::double_parameter("POROCONTACTFPSI_HFRACTION", 1.0,
      "factor of element size, when transition between FPSI and PSCI is started!", xfsi_monolithic);
  Core::Utils::double_parameter("POROCONTACTFPSI_FULLPCFRACTION", 0.0,
      "ration of gap/(POROCONTACTFPSI_HFRACTION*h) when full PSCI is started!", xfsi_monolithic);
  Core::Utils::bool_parameter("USE_PORO_PRESSURE", true,
      "the extrapolation of the fluid stress in the contact zone is relaxed to zero after a "
      "certtain distance",
      xfsi_monolithic);

  xfsi_monolithic.move_into_collection(list);
}


void Inpar::XFEM::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO;
  using namespace Core::IO::InputSpecBuilders;

  auto dirichletbundcomponents = all_of({
      parameter<int>("NUMDOF"),
      parameter<std::vector<int>>("ONOFF", {.size = from_parameter<int>("NUMDOF")}),
      parameter<std::vector<double>>("VAL", {.size = from_parameter<int>("NUMDOF")}),
      parameter<std::vector<std::optional<int>>>("FUNCT", {.size = from_parameter<int>("NUMDOF")}),
      selection<std::string>("TAG", {"none", "monitor_reaction"}, {.default_value = "none"}),
  });

  auto neumanncomponents = all_of({
      parameter<int>("NUMDOF"),
      parameter<std::vector<int>>("ONOFF", {.size = from_parameter<int>("NUMDOF")}),
      parameter<std::vector<double>>("VAL", {.size = from_parameter<int>("NUMDOF")}),
      parameter<std::vector<std::optional<int>>>("FUNCT", {.size = from_parameter<int>("NUMDOF")}),
      selection<std::string>("TYPE",
          {"Live", "Dead", "pseudo_orthopressure", "orthopressure", "PressureGrad"},
          {.default_value = "Live"}),
  });

  Core::Conditions::ConditionDefinition movingfluid("DESIGN FLUID MESH VOL CONDITIONS", "FluidMesh",
      "Fluid Mesh", Core::Conditions::FluidMesh, true, Core::Conditions::geometry_type_volume);
  Core::Conditions::ConditionDefinition fluidfluidcoupling(
      "DESIGN FLUID FLUID COUPLING SURF CONDITIONS", "FluidFluidCoupling", "FLUID FLUID Coupling",
      Core::Conditions::FluidFluidCoupling, true, Core::Conditions::geometry_type_surface);
  Core::Conditions::ConditionDefinition ALEfluidcoupling(
      "DESIGN ALE FLUID COUPLING SURF CONDITIONS", "ALEFluidCoupling", "ALE FLUID Coupling",
      Core::Conditions::ALEFluidCoupling, true, Core::Conditions::geometry_type_surface);

  const auto make_fluid_cond = [&condlist](Core::Conditions::ConditionDefinition& cond)
  {
    cond.add_component(parameter<int>("COUPLINGID"));

    condlist.emplace_back(cond);
  };

  make_fluid_cond(movingfluid);
  make_fluid_cond(fluidfluidcoupling);
  make_fluid_cond(ALEfluidcoupling);

  /*--------------------------------------------------------------------*/
  // XFEM coupling conditions

  //*----------------*/
  // Displacement surface condition for XFEM WDBC and Neumann boundary conditions

  Core::Conditions::ConditionDefinition xfem_surf_displacement(
      "DESIGN XFEM DISPLACEMENT SURF CONDITIONS", "XFEMSurfDisplacement", "XFEM Surf Displacement",
      Core::Conditions::XFEM_Surf_Displacement, true, Core::Conditions::geometry_type_surface);

  xfem_surf_displacement.add_component(parameter<int>("COUPLINGID"));
  xfem_surf_displacement.add_component(selection<std::string>("EVALTYPE",
      {"zero", "funct", "implementation"}, {.description = "", .default_value = "funct"}));

  xfem_surf_displacement.add_component(dirichletbundcomponents);

  condlist.push_back(xfem_surf_displacement);

  //*----------------*/
  // Levelset field condition components

  auto levelsetfield_components = all_of({
      parameter<int>("COUPLINGID"),
      parameter<int>("LEVELSETFIELDNO"),
      selection<std::string>("BOOLEANTYPE",
          {"none", "cut", "union", "difference", "sym_difference"},
          {.description = "define which boolean operator is used for combining this level-set "
                          "field with the previous one with smaller coupling id"}),
      parameter<bool>(
          "COMPLEMENTARY", {.description = "define which complementary operator is applied "
                                           "after combining the level-set field with a boolean "
                                           "operator with the previous one"}),
  });

  //*----------------*/
  // Levelset based Weak Dirichlet conditions

  Core::Conditions::ConditionDefinition xfem_levelset_wdbc(
      "DESIGN XFEM LEVELSET WEAK DIRICHLET VOL CONDITIONS", "XFEMLevelsetWeakDirichlet",
      "XFEM Levelset Weak Dirichlet", Core::Conditions::XFEM_Levelset_Weak_Dirichlet, true,
      Core::Conditions::geometry_type_volume);

  xfem_levelset_wdbc.add_component(levelsetfield_components);

  xfem_levelset_wdbc.add_component(dirichletbundcomponents);

  // optional: allow for random noise, set percentage used in uniform random distribution
  xfem_levelset_wdbc.add_component(parameter<double>("RANDNOISE",
      {.description = "set percentage of random noise used in uniform random distribution",
          .default_value = 0.0}));

  condlist.push_back(xfem_levelset_wdbc);

  //*----------------*/
  // Levelset based Neumann conditions

  Core::Conditions::ConditionDefinition xfem_levelset_neumann(
      "DESIGN XFEM LEVELSET NEUMANN VOL CONDITIONS", "XFEMLevelsetNeumann", "XFEM Levelset Neumann",
      Core::Conditions::XFEM_Levelset_Neumann, true, Core::Conditions::geometry_type_volume);

  xfem_levelset_neumann.add_component(levelsetfield_components);

  xfem_levelset_neumann.add_component(neumanncomponents);

  // define if we use inflow stabilization on the xfem neumann surf condition
  xfem_levelset_neumann.add_component(
      parameter<bool>("INFLOW_STAB", {.description = "", .default_value = false}));
  condlist.push_back(xfem_levelset_neumann);

  //*----------------*/
  // Levelset based Navier Slip conditions

  Core::Conditions::ConditionDefinition xfem_levelset_navier_slip(
      "DESIGN XFEM LEVELSET NAVIER SLIP VOL CONDITIONS", "XFEMLevelsetNavierSlip",
      "XFEM Levelset Navier Slip", Core::Conditions::XFEM_Levelset_Navier_Slip, true,
      Core::Conditions::geometry_type_volume);

  xfem_levelset_navier_slip.add_component(levelsetfield_components);

  xfem_levelset_navier_slip.add_component(selection<int>("SURFACE_PROJECTION",
      {{"proj_normal", Inpar::XFEM::Proj_normal}, {"proj_smoothed", Inpar::XFEM::Proj_smoothed},
          {"proj_normal_smoothed_comb", Inpar::XFEM::Proj_normal_smoothed_comb},
          {"proj_normal_phi", Inpar::XFEM::Proj_normal_phi}},
      {.description = "", .default_value = Inpar::XFEM::Proj_normal}));
  xfem_levelset_navier_slip.add_component(
      parameter<int>("L2_PROJECTION_SOLVER", {.description = ""}));
  xfem_levelset_navier_slip.add_component(
      parameter<std::optional<int>>("ROBIN_DIRICHLET_ID", {.description = ""}));
  xfem_levelset_navier_slip.add_component(
      parameter<std::optional<int>>("ROBIN_NEUMANN_ID", {.description = ""}));
  xfem_levelset_navier_slip.add_component(parameter<double>("SLIPCOEFFICIENT"));
  xfem_levelset_navier_slip.add_component(
      parameter<std::optional<int>>("FUNCT", {.description = "slip function id"}));
  xfem_levelset_navier_slip.add_component(
      parameter<bool>("FORCE_ONLY_TANG_VEL", {.description = "", .default_value = false}));

  condlist.push_back(xfem_levelset_navier_slip);

  // Add condition XFEM DIRICHLET/NEUMANN?

  Core::Conditions::ConditionDefinition xfem_navier_slip_robin_dirch(
      "DESIGN XFEM ROBIN DIRICHLET VOL CONDITIONS", "XFEMRobinDirichletVol",
      "XFEM Robin Dirichlet Volume", Core::Conditions::XFEM_Robin_Dirichlet_Volume, true,
      Core::Conditions::geometry_type_volume);

  xfem_navier_slip_robin_dirch.add_component(
      parameter<std::optional<int>>("ROBIN_ID", {.description = "robin id"}));

  xfem_navier_slip_robin_dirch.add_component(dirichletbundcomponents);

  condlist.push_back(xfem_navier_slip_robin_dirch);

  Core::Conditions::ConditionDefinition xfem_navier_slip_robin_neumann(
      "DESIGN XFEM ROBIN NEUMANN VOL CONDITIONS", "XFEMRobinNeumannVol",
      "XFEM Robin Neumann Volume", Core::Conditions::XFEM_Robin_Neumann_Volume, true,
      Core::Conditions::geometry_type_volume);

  xfem_navier_slip_robin_neumann.add_component(
      parameter<std::optional<int>>("ROBIN_ID", {.description = "robin id"}));

  xfem_navier_slip_robin_neumann.add_component(neumanncomponents);

  condlist.push_back(xfem_navier_slip_robin_neumann);


  //*----------------*/
  // Levelset based Twophase conditions

  Core::Conditions::ConditionDefinition xfem_levelset_twophase(
      "DESIGN XFEM LEVELSET TWOPHASE VOL CONDITIONS", "XFEMLevelsetTwophase",
      "XFEM Levelset Twophase", Core::Conditions::XFEM_Levelset_Twophase, true,
      Core::Conditions::geometry_type_volume);

  xfem_levelset_twophase.add_component(levelsetfield_components);

  condlist.push_back(xfem_levelset_twophase);

  //*----------------*/
  // Surface Fluid-Fluid coupling conditions

  Core::Conditions::ConditionDefinition xfem_surf_fluidfluid(
      "DESIGN XFEM FLUIDFLUID SURF CONDITIONS", "XFEMSurfFluidFluid", "XFEM Surf FluidFluid",
      Core::Conditions::XFEM_Surf_FluidFluid, true, Core::Conditions::geometry_type_surface);

  xfem_surf_fluidfluid.add_component(parameter<int>("COUPLINGID"));
  xfem_surf_fluidfluid.add_component(selection<int>("COUPSTRATEGY",
      {{"xfluid", Inpar::XFEM::Xfluid_Sided}, {"embedded", Inpar::XFEM::Embedded_Sided},
          {"mean", Inpar::XFEM::Mean}},
      {.description = "coupling strategy"}));

  condlist.push_back(xfem_surf_fluidfluid);

  //*----------------*/
  // Surface partitioned XFSI boundary conditions

  Core::Conditions::ConditionDefinition xfem_surf_fsi_part(
      "DESIGN XFEM FSI PARTITIONED SURF CONDITIONS", "XFEMSurfFSIPart", "XFEM Surf FSI Part",
      Core::Conditions::XFEM_Surf_FSIPart, true, Core::Conditions::geometry_type_surface);

  xfem_surf_fsi_part.add_component(parameter<int>("COUPLINGID"));

  // COUPSTRATEGY IS FLUID SIDED
  xfem_surf_fsi_part.add_component(selection<int>("INTLAW",
      {{"noslip", Inpar::XFEM::noslip}, {"noslip_splitpen", Inpar::XFEM::noslip_splitpen},
          {"slip", Inpar::XFEM::slip}, {"navslip", Inpar::XFEM::navierslip}},
      {.description = "", .default_value = Inpar::XFEM::noslip}));
  xfem_surf_fsi_part.add_component(
      parameter<double>("SLIPCOEFFICIENT", {.description = "", .default_value = 0.0}));
  xfem_surf_fsi_part.add_component(
      parameter<std::optional<int>>("FUNCT", {.description = "slip function id"}));

  condlist.push_back(xfem_surf_fsi_part);

  //*----------------*/
  // Surface monolithic XFSI coupling conditions

  Core::Conditions::ConditionDefinition xfem_surf_fsi_mono(
      "DESIGN XFEM FSI MONOLITHIC SURF CONDITIONS", "XFEMSurfFSIMono", "XFEM Surf FSI Mono",
      Core::Conditions::XFEM_Surf_FSIMono, true, Core::Conditions::geometry_type_surface);

  xfem_surf_fsi_mono.add_component(parameter<int>("COUPLINGID"));
  xfem_surf_fsi_mono.add_component(selection<int>("COUPSTRATEGY",
      {{"xfluid", Inpar::XFEM::Xfluid_Sided}, {"solid", Inpar::XFEM::Embedded_Sided},
          {"mean", Inpar::XFEM::Mean}, {"harmonic", Inpar::XFEM::Harmonic}},
      {.description = "", .default_value = Inpar::XFEM::Xfluid_Sided}));
  xfem_surf_fsi_mono.add_component(selection<int>("INTLAW",
      {{"noslip", Inpar::XFEM::noslip}, {"noslip_splitpen", Inpar::XFEM::noslip_splitpen},
          {"slip", Inpar::XFEM::slip}, {"navslip", Inpar::XFEM::navierslip},
          {"navslip_contact", Inpar::XFEM::navierslip_contact}},
      {.description = "", .default_value = Inpar::XFEM::noslip}));
  xfem_surf_fsi_mono.add_component(
      parameter<double>("SLIPCOEFFICIENT", {.description = "", .default_value = 0.0}));
  xfem_surf_fsi_mono.add_component(
      parameter<std::optional<int>>("FUNCT", {.description = "slip function id"}));

  condlist.push_back(xfem_surf_fsi_mono);

  //*----------------*/
  // Surface monolithic XFPI coupling conditions

  Core::Conditions::ConditionDefinition xfem_surf_fpi_mono(
      "DESIGN XFEM FPI MONOLITHIC SURF CONDITIONS", "XFEMSurfFPIMono", "XFEM Surf FPI Mono",
      Core::Conditions::XFEM_Surf_FPIMono, true, Core::Conditions::geometry_type_surface);

  xfem_surf_fpi_mono.add_component(parameter<int>("COUPLINGID"));
  xfem_surf_fpi_mono.add_component(
      parameter<double>("BJ_COEFF", {.description = "", .default_value = 0.}));
  xfem_surf_fpi_mono.add_component(selection<std::string>(
      "Variant", {"BJ", "BJS"}, {.description = "variant", .default_value = "BJ"}));
  xfem_surf_fpi_mono.add_component(selection<std::string>(
      "Method", {"NIT", "SUB"}, {.description = "method", .default_value = "NIT"}));
  xfem_surf_fpi_mono.add_component(
      parameter<bool>("Contact", {.description = "contact", .default_value = false}));

  condlist.push_back(xfem_surf_fpi_mono);


  //*----------------*/
  // Surface Weak Dirichlet conditions

  Core::Conditions::ConditionDefinition xfem_surf_wdbc("DESIGN XFEM WEAK DIRICHLET SURF CONDITIONS",
      "XFEMSurfWeakDirichlet", "XFEM Surf Weak Dirichlet",
      Core::Conditions::XFEM_Surf_Weak_Dirichlet, true, Core::Conditions::geometry_type_surface);

  xfem_surf_wdbc.add_component(parameter<int>("COUPLINGID"));
  xfem_surf_wdbc.add_component(selection<std::string>("EVALTYPE",
      {"zero", "funct_interpolated", "funct_gausspoint", "displacement_1storder_wo_initfunct",
          "displacement_2ndorder_wo_initfunct", "displacement_1storder_with_initfunct",
          "displacement_2ndorder_with_initfunct"},
      {.description = "", .default_value = "funct_interpolated"}));

  xfem_surf_wdbc.add_component(dirichletbundcomponents);

  // optional: allow for random noise, set percentage used in uniform random distribution
  xfem_surf_wdbc.add_component(parameter<double>("RANDNOISE",
      {.description = "set percentage of random noise used in uniform random distribution",
          .default_value = 0.0}));

  condlist.push_back(xfem_surf_wdbc);


  //*----------------*/
  // Surface Neumann conditions

  Core::Conditions::ConditionDefinition xfem_surf_neumann("DESIGN XFEM NEUMANN SURF CONDITIONS",
      "XFEMSurfNeumann", "XFEM Surf Neumann", Core::Conditions::XFEM_Surf_Neumann, true,
      Core::Conditions::geometry_type_surface);

  xfem_surf_neumann.add_component(parameter<int>("COUPLINGID"));

  xfem_surf_neumann.add_component(neumanncomponents);

  // define if we use inflow stabilization on the xfem neumann surf condition
  xfem_surf_neumann.add_component(parameter<bool>(
      "INFLOW_STAB", {.description = "toggle inflow stabilization", .default_value = false}));

  condlist.push_back(xfem_surf_neumann);

  //*----------------*/
  // Surface Navier Slip conditions

  Core::Conditions::ConditionDefinition xfem_surf_navier_slip(
      "DESIGN XFEM NAVIER SLIP SURF CONDITIONS", "XFEMSurfNavierSlip", "XFEM Surf Navier Slip",
      Core::Conditions::XFEM_Surf_Navier_Slip, true, Core::Conditions::geometry_type_surface);

  xfem_surf_navier_slip.add_component(parameter<int>("COUPLINGID"));
  xfem_surf_navier_slip.add_component(selection<std::string>("EVALTYPE",
      {"zero", "funct_interpolated", "funct_gausspoint", "displacement_1storder_wo_initfunct",
          "displacement_2ndorder_wo_initfunct", "displacement_1storder_with_initfunct",
          "displacement_2ndorder_with_initfunct"},
      {.description = "", .default_value = "funct_interpolated"}));
  xfem_surf_navier_slip.add_component(
      parameter<std::optional<int>>("ROBIN_DIRICHLET_ID", {.description = ""}));
  xfem_surf_navier_slip.add_component(
      parameter<std::optional<int>>("ROBIN_NEUMANN_ID", {.description = ""}));
  xfem_surf_navier_slip.add_component(parameter<double>("SLIPCOEFFICIENT"));
  xfem_surf_navier_slip.add_component(
      parameter<std::optional<int>>("FUNCT", {.description = "slip function id"}));
  xfem_surf_navier_slip.add_component(
      parameter<bool>("FORCE_ONLY_TANG_VEL", {.description = "", .default_value = false}));

  condlist.push_back(xfem_surf_navier_slip);

  Core::Conditions::ConditionDefinition xfem_navier_slip_robin_dirch_surf(
      "DESIGN XFEM ROBIN DIRICHLET SURF CONDITIONS", "XFEMRobinDirichletSurf",
      "XFEM Robin Dirichlet Volume", Core::Conditions::XFEM_Robin_Dirichlet_Surf, true,
      Core::Conditions::geometry_type_surface);

  // this implementation should be reviewed at some point as it requires these conditions
  //  to have a couplingID. In theory this should not be necessary.
  xfem_navier_slip_robin_dirch_surf.add_component(parameter<int>("COUPLINGID"));
  xfem_navier_slip_robin_dirch_surf.add_component(
      parameter<std::optional<int>>("ROBIN_ID", {.description = "robin id"}));

  // Likely, not necessary. But needed for the current structure.
  xfem_navier_slip_robin_dirch_surf.add_component(selection<std::string>("EVALTYPE",
      {"zero", "funct_interpolated", "funct_gausspoint", "displacement_1storder_wo_initfunct",
          "displacement_2ndorder_wo_initfunct", "displacement_1storder_with_initfunct",
          "displacement_2ndorder_with_initfunct"},
      {.description = "", .default_value = "funct_interpolated"}));

  xfem_navier_slip_robin_dirch_surf.add_component(dirichletbundcomponents);

  condlist.push_back(xfem_navier_slip_robin_dirch_surf);

  Core::Conditions::ConditionDefinition xfem_navier_slip_robin_neumann_surf(
      "DESIGN XFEM ROBIN NEUMANN SURF CONDITIONS", "XFEMRobinNeumannSurf",
      "XFEM Robin Neumann Volume", Core::Conditions::XFEM_Robin_Neumann_Surf, true,
      Core::Conditions::geometry_type_surface);

  // this implementation should be reviewed at some point as it requires these conditions
  //  to have a couplingID. In theory this should not be necessary.
  xfem_navier_slip_robin_neumann_surf.add_component(parameter<int>("COUPLINGID"));
  xfem_navier_slip_robin_neumann_surf.add_component(
      parameter<std::optional<int>>("ROBIN_ID", {.description = "robin id"}));

  xfem_navier_slip_robin_neumann_surf.add_component(neumanncomponents);

  condlist.push_back(xfem_navier_slip_robin_neumann_surf);

  //*----------------*/
  // Solid to solid embedded mesh coupling conditions
  Core::Conditions::ConditionDefinition solid_surf_coupling(
      "DESIGN EMBEDDED MESH SOLID SURF COUPLING CONDITIONS", "EmbeddedMeshSolidSurfCoupling",
      "Embedded Mesh Solid Surface Coupling", Core::Conditions::Embedded_Mesh_Solid_Surf_Coupling,
      true, Core::Conditions::geometry_type_surface);

  solid_surf_coupling.add_component(parameter<int>("COUPLINGID"));

  condlist.push_back(solid_surf_coupling);

  // Solid to solid embedded mesh volume background mesh condition
  Core::Conditions::ConditionDefinition solid_vol_background_coupling(
      "DESIGN EMBEDDED SOLID VOL BACKGROUND CONDITIONS", "EmbeddedMeshSolidVolBackground",
      "Embedded Mesh Solid Volume Background",
      Core::Conditions::Embedded_Mesh_Solid_Volume_Background, true,
      Core::Conditions::geometry_type_volume);

  solid_vol_background_coupling.add_component(parameter<int>("COUPLINGID"));
  condlist.push_back(solid_vol_background_coupling);
}

FOUR_C_NAMESPACE_CLOSE
