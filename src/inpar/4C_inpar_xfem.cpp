// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_xfem.hpp"

#include "4C_cut_enum.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_io_linecomponent.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::XFEM::set_valid_parameters(Teuchos::ParameterList& list)
{
  using namespace Input;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& xfem_general = list.sublist("XFEM GENERAL", false, "");

  // OUTPUT options
  Core::Utils::bool_parameter("GMSH_DEBUG_OUT", "No",
      "Do you want to write extended Gmsh output for each timestep?", &xfem_general);
  Core::Utils::bool_parameter("GMSH_DEBUG_OUT_SCREEN", "No",
      "Do you want to be informed, if Gmsh output is written?", &xfem_general);
  Core::Utils::bool_parameter("GMSH_SOL_OUT", "No",
      "Do you want to write extended Gmsh output for each timestep?", &xfem_general);
  Core::Utils::bool_parameter("GMSH_TIMINT_OUT", "No",
      "Do you want to write extended Gmsh output for each timestep?", &xfem_general);
  Core::Utils::bool_parameter("GMSH_EOS_OUT", "No",
      "Do you want to write extended Gmsh output for each timestep?", &xfem_general);
  Core::Utils::bool_parameter("GMSH_DISCRET_OUT", "No",
      "Do you want to write extended Gmsh output for each timestep?", &xfem_general);
  Core::Utils::bool_parameter("GMSH_CUT_OUT", "No",
      "Do you want to write extended Gmsh output for each timestep?", &xfem_general);
  Core::Utils::bool_parameter(
      "PRINT_OUTPUT", "No", "Is the output of the cut process desired?", &xfem_general);

  Core::Utils::int_parameter(
      "MAX_NUM_DOFSETS", 3, "Maximum number of volumecells in the XFEM element", &xfem_general);

  setStringToIntegralParameter<Cut::NodalDofSetStrategy>("NODAL_DOFSET_STRATEGY", "full",
      "Strategy used for the nodal dofset management per node",
      tuple<std::string>(
          "OneDofset_PerNodeAndPosition", "ConnectGhostDofsets_PerNodeAndPosition", "full"),
      tuple<Cut::NodalDofSetStrategy>(Cut::NDS_Strategy_OneDofset_PerNodeAndPosition,
          Cut::NDS_Strategy_ConnectGhostDofsets_PerNodeAndPosition, Cut::NDS_Strategy_full),
      &xfem_general);

  // Integration options
  setStringToIntegralParameter<Cut::VCellGaussPts>("VOLUME_GAUSS_POINTS_BY", "Tessellation",
      "Method for finding Gauss Points for the cut volumes",
      tuple<std::string>("Tessellation", "MomentFitting", "DirectDivergence"),
      tuple<Cut::VCellGaussPts>(Cut::VCellGaussPts_Tessellation, Cut::VCellGaussPts_MomentFitting,
          Cut::VCellGaussPts_DirectDivergence),
      &xfem_general);

  setStringToIntegralParameter<Cut::BCellGaussPts>("BOUNDARY_GAUSS_POINTS_BY", "Tessellation",
      "Method for finding Gauss Points for the boundary cells",
      tuple<std::string>("Tessellation", "MomentFitting", "DirectDivergence"),
      tuple<Cut::BCellGaussPts>(Cut::BCellGaussPts_Tessellation, Cut::BCellGaussPts_MomentFitting,
          Cut::BCellGaussPts_DirectDivergence),
      &xfem_general);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& xfluid_dyn = list.sublist("XFLUID DYNAMIC", false, "");

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& xfluid_general = xfluid_dyn.sublist("GENERAL", false, "");

  // Do we use more than one fluid discretization?
  Core::Utils::bool_parameter("XFLUIDFLUID", "no", "Use an embedded fluid patch.", &xfluid_general);

  // How many monolithic steps we keep the fluidfluid-boundary fixed
  Core::Utils::int_parameter(
      "RELAXING_ALE_EVERY", 1, "Relaxing Ale after how many monolithic steps", &xfluid_general);

  Core::Utils::bool_parameter("RELAXING_ALE", "yes",
      "switch on/off for relaxing Ale in monolithic fluid-fluid-fsi", &xfluid_general);

  Core::Utils::double_parameter(
      "XFLUIDFLUID_SEARCHRADIUS", 1.0, "Radius of the search tree", &xfluid_general);

  // xfluidfluid-fsi-monolithic approach
  setStringToIntegralParameter<MonolithicXffsiApproach>("MONOLITHIC_XFFSI_APPROACH",
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
      &xfluid_general);

  // xfluidfluid time integration approach
  setStringToIntegralParameter<XFluidFluidTimeInt>("XFLUIDFLUID_TIMEINT", "Xff_TimeInt_FullProj",
      "The xfluidfluid-timeintegration approach",
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
      &xfluid_general);

  setStringToIntegralParameter<XFluidTimeIntScheme>("XFLUID_TIMEINT",
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
      &xfluid_general);

  Core::Utils::bool_parameter("ALE_XFluid", "no", "XFluid is Ale Fluid?", &xfluid_general);

  // for new OST-implementation: which interface terms to be evaluated for previous time step
  setStringToIntegralParameter<InterfaceTermsPreviousState>("INTERFACE_TERMS_PREVIOUS_STATE",
      "PreviousState_only_consistency",
      "how to treat interface terms from previous time step (new OST)",
      tuple<std::string>("PreviousState_only_consistency", "PreviousState_full"),
      tuple<InterfaceTermsPreviousState>(
          Inpar::XFEM::PreviousState_only_consistency,  /// evaluate only consistency terms
                                                        /// for previous time step
          Inpar::XFEM::PreviousState_full  /// evaluate consistency, adjoint consistency and penalty
                                           /// terms or previous time step
          ),
      &xfluid_general);

  Core::Utils::bool_parameter("XFLUID_TIMEINT_CHECK_INTERFACETIPS", "Yes",
      "Xfluid TimeIntegration Special Check if node has changed the side!", &xfluid_general);

  Core::Utils::bool_parameter("XFLUID_TIMEINT_CHECK_SLIDINGONSURFACE", "Yes",
      "Xfluid TimeIntegration Special Check if node is sliding on surface!", &xfluid_general);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& xfluid_stab = xfluid_dyn.sublist("STABILIZATION", false, "");

  // Boundary-Coupling options
  setStringToIntegralParameter<CouplingMethod>("COUPLING_METHOD", "Nitsche",
      "method how to enforce embedded boundary/coupling conditions at the interface",
      tuple<std::string>("Hybrid_LM_Cauchy_stress", "Hybrid_LM_viscous_stress", "Nitsche"),
      tuple<CouplingMethod>(
          Inpar::XFEM::Hybrid_LM_Cauchy_stress,   // Cauchy stress-based mixed/hybrid formulation
          Inpar::XFEM::Hybrid_LM_viscous_stress,  // viscous stress-based mixed/hybrid formulation
          Inpar::XFEM::Nitsche                    // Nitsche's formulation
          ),
      &xfluid_stab);

  setStringToIntegralParameter<HybridLmL2Proj>("HYBRID_LM_L2_PROJ", "part_ele_proj",
      "perform the L2 projection between stress fields on whole element or on fluid part?",
      tuple<std::string>("full_ele_proj", "part_ele_proj"),
      tuple<HybridLmL2Proj>(
          Inpar::XFEM::Hybrid_LM_L2_Proj_full,  // L2 stress projection on whole fluid element
          Inpar::XFEM::Hybrid_LM_L2_Proj_part   // L2 stress projection on partial fluid element
                                                // volume
          ),
      &xfluid_stab);

  setStringToIntegralParameter<AdjointScaling>("VISC_ADJOINT_SYMMETRY", "yes",
      "viscous and adjoint viscous interface terms with matching sign?",
      tuple<std::string>("yes", "no", "sym", "skew", "none"),
      tuple<AdjointScaling>(Inpar::XFEM::adj_sym, Inpar::XFEM::adj_skew, Inpar::XFEM::adj_sym,
          Inpar::XFEM::adj_skew, Inpar::XFEM::adj_none),
      &xfluid_stab);

  // viscous and convective Nitsche/MSH stabilization parameter
  Core::Utils::double_parameter(
      "NIT_STAB_FAC", 35.0, " ( stabilization parameter for Nitsche's penalty term", &xfluid_stab);
  Core::Utils::double_parameter("NIT_STAB_FAC_TANG", 35.0,
      " ( stabilization parameter for Nitsche's penalty tangential term", &xfluid_stab);

  setStringToIntegralParameter<ViscStabTraceEstimate>("VISC_STAB_TRACE_ESTIMATE", "CT_div_by_hk",
      "how to estimate the scaling from the trace inequality in Nitsche's method",
      tuple<std::string>("CT_div_by_hk", "eigenvalue"),
      tuple<ViscStabTraceEstimate>(
          Inpar::XFEM::ViscStab_TraceEstimate_CT_div_by_hk,  // estimate the trace inequality by a
                                                             // trace-constant CT and hk: CT/hk
          Inpar::XFEM::ViscStab_TraceEstimate_eigenvalue     // estimate the trace inequality by
                                                          // solving a local element-wise eigenvalue
                                                          // problem
          ),
      &xfluid_stab);

  setStringToIntegralParameter<TraceEstimateEigenvalueUpdate>("UPDATE_EIGENVALUE_TRACE_ESTIMATE",
      "every_iter", "how often should the local eigenvalue problem be updated",
      tuple<std::string>("every_iter", "every_timestep", "once"),
      tuple<TraceEstimateEigenvalueUpdate>(Inpar::XFEM::Eigenvalue_update_every_iter,
          Inpar::XFEM::Eigenvalue_update_every_timestep, Inpar::XFEM::Eigenvalue_update_once),
      &xfluid_stab);

  setStringToIntegralParameter<ViscStabHk>("VISC_STAB_HK", "ele_vol_div_by_max_ele_surf",
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
      &xfluid_stab);


  setStringToIntegralParameter<ConvStabScaling>("CONV_STAB_SCALING", "none",
      "scaling factor for viscous interface stabilization (Nitsche, MSH)",
      tuple<std::string>("inflow", "abs_inflow", "none"),
      tuple<ConvStabScaling>(Inpar::XFEM::ConvStabScaling_inflow,  // scaling with max(0,-u*n)
          Inpar::XFEM::ConvStabScaling_abs_inflow,                 // scaling with |u*n|
          Inpar::XFEM::ConvStabScaling_none                        // no convective stabilization
          ),
      &xfluid_stab);

  setStringToIntegralParameter<XffConvStabScaling>("XFF_CONV_STAB_SCALING", "none",
      "scaling factor for convective interface stabilization of fluid-fluid Coupling",
      tuple<std::string>("inflow", "averaged", "none"),
      tuple<XffConvStabScaling>(
          Inpar::XFEM::XFF_ConvStabScaling_upwinding,      // one-sided inflow stabilization
          Inpar::XFEM::XFF_ConvStabScaling_only_averaged,  // averaged inflow stabilization
          Inpar::XFEM::XFF_ConvStabScaling_none            // no convective stabilization
          ),
      &xfluid_stab);

  setStringToIntegralParameter<MassConservationCombination>("MASS_CONSERVATION_COMBO", "max",
      "choose the maximum from viscous and convective contributions or just sum both up",
      tuple<std::string>("max", "sum"),
      tuple<MassConservationCombination>(
          Inpar::XFEM::MassConservationCombination_max,  /// use the maximum contribution
          Inpar::XFEM::MassConservationCombination_sum  /// sum viscous and convective contributions
          ),
      &xfluid_stab);

  setStringToIntegralParameter<MassConservationScaling>("MASS_CONSERVATION_SCALING", "only_visc",
      "apply additional scaling of penalty term to enforce mass conservation for "
      "convection-dominated flow",
      tuple<std::string>("full", "only_visc"),
      tuple<MassConservationScaling>(
          Inpar::XFEM::MassConservationScaling_full,  /// apply mass-conserving convective scaling
                                                      /// additionally
          Inpar::XFEM::MassConservationScaling_only_visc  /// use only the viscous scaling
          ),
      &xfluid_stab);

  Core::Utils::bool_parameter("GHOST_PENALTY_STAB", "no",
      "switch on/off ghost penalty interface stabilization", &xfluid_stab);

  Core::Utils::bool_parameter("GHOST_PENALTY_TRANSIENT_STAB", "no",
      "switch on/off ghost penalty transient interface stabilization", &xfluid_stab);

  Core::Utils::bool_parameter("GHOST_PENALTY_2nd_STAB", "no",
      "switch on/off ghost penalty interface stabilization for 2nd order derivatives",
      &xfluid_stab);
  Core::Utils::bool_parameter("GHOST_PENALTY_2nd_STAB_NORMAL", "no",
      "switch between ghost penalty interface stabilization for 2nd order derivatives in normal or "
      "all spatial directions",
      &xfluid_stab);


  Core::Utils::double_parameter("GHOST_PENALTY_FAC", 0.1,
      "define stabilization parameter ghost penalty interface stabilization", &xfluid_stab);

  Core::Utils::double_parameter("GHOST_PENALTY_TRANSIENT_FAC", 0.001,
      "define stabilization parameter ghost penalty transient interface stabilization",
      &xfluid_stab);

  Core::Utils::double_parameter("GHOST_PENALTY_2nd_FAC", 0.05,
      "define stabilization parameter ghost penalty 2nd order viscous interface stabilization",
      &xfluid_stab);
  Core::Utils::double_parameter("GHOST_PENALTY_PRESSURE_2nd_FAC", 0.05,
      "define stabilization parameter ghost penalty 2nd order pressure interface stabilization",
      &xfluid_stab);


  Core::Utils::bool_parameter("XFF_EOS_PRES_EMB_LAYER", "no",
      "switch on/off edge-based pressure stabilization on interface-contributing elements of the "
      "embedded fluid",
      &xfluid_stab);

  Core::Utils::bool_parameter("IS_PSEUDO_2D", "no",
      "modify viscous interface stabilization due to the vanishing polynomial in third dimension "
      "when using strong Dirichlet conditions to block polynomials in one spatial dimension",
      &xfluid_stab);

  Core::Utils::bool_parameter("GHOST_PENALTY_ADD_INNER_FACES", "no",
      "Apply ghost penalty stabilization also for inner faces if this is possible due to the "
      "dofsets",
      &xfluid_stab);

  /*----------------------------------------------------------------------*/
  Teuchos::ParameterList& xfsi_monolithic = xfluid_dyn.sublist("XFPSI MONOLITHIC", false, "");

  Core::Utils::int_parameter(
      "ITEMIN", 1, "How many iterations are performed minimal", &xfsi_monolithic);
  Core::Utils::int_parameter(
      "ITEMAX_OUTER", 5, "How many outer iterations are performed maximal", &xfsi_monolithic);
  Core::Utils::bool_parameter("ND_NEWTON_DAMPING", "no",
      "Activate Newton damping based on residual and increment", &xfsi_monolithic);
  Core::Utils::double_parameter("ND_MAX_DISP_ITERINC", -1.0,
      "Maximal displacement increment to apply full newton --> otherwise damp newton",
      &xfsi_monolithic);
  Core::Utils::double_parameter("ND_MAX_VEL_ITERINC", -1.0,
      "Maximal fluid velocity increment to apply full newton --> otherwise damp newton",
      &xfsi_monolithic);
  Core::Utils::double_parameter("ND_MAX_PRES_ITERINC", -1.0,
      "Maximal fluid pressure increment to apply full newton --> otherwise damp newton",
      &xfsi_monolithic);
  Core::Utils::double_parameter("ND_MAX_PVEL_ITERINC", -1.0,
      "Maximal porofluid velocity increment to apply full newton --> otherwise damp newton",
      &xfsi_monolithic);
  Core::Utils::double_parameter("ND_MAX_PPRES_ITERINC", -1.0,
      "Maximal porofluid pressure increment to apply full newton --> otherwise damp newton",
      &xfsi_monolithic);
  Core::Utils::double_parameter("CUT_EVALUATE_MINTOL", 0.0,
      "Minimal value of the maximal structural displacement for which the CUT is evaluate in this "
      "iteration!",
      &xfsi_monolithic);
  Core::Utils::int_parameter("CUT_EVALUATE_MINITER", 0,
      "Minimal number of nonlinear iterations, before the CUT is potentially not evaluated",
      &xfsi_monolithic);
  Core::Utils::bool_parameter("EXTRAPOLATE_TO_ZERO", "no",
      "the extrapolation of the fluid stress in the contact zone is relaxed to zero after a "
      "certain distance",
      &xfsi_monolithic);
  Core::Utils::double_parameter("POROCONTACTFPSI_HFRACTION", 1.0,
      "factor of element size, when transition between FPSI and PSCI is started!",
      &xfsi_monolithic);
  Core::Utils::double_parameter("POROCONTACTFPSI_FULLPCFRACTION", 0.0,
      "ration of gap/(POROCONTACTFPSI_HFRACTION*h) when full PSCI is started!", &xfsi_monolithic);
  Core::Utils::bool_parameter("USE_PORO_PRESSURE", "yes",
      "the extrapolation of the fluid stress in the contact zone is relaxed to zero after a "
      "certtain distance",
      &xfsi_monolithic);
}


void Inpar::XFEM::set_valid_conditions(
    std::vector<std::shared_ptr<Core::Conditions::ConditionDefinition>>& condlist)
{
  using namespace Input;

  // define standard Dirichlet condition components
  std::vector<std::shared_ptr<LineComponent>> dirichletbundcomponents;

  dirichletbundcomponents.emplace_back(std::make_shared<SeparatorComponent>("NUMDOF"));
  dirichletbundcomponents.emplace_back(std::make_shared<IntComponent>("NUMDOF"));

  dirichletbundcomponents.emplace_back(std::make_shared<SeparatorComponent>("ONOFF"));
  dirichletbundcomponents.emplace_back(
      std::make_shared<IntVectorComponent>("ONOFF", LengthFromInt("NUMDOF")));
  dirichletbundcomponents.emplace_back(std::make_shared<SeparatorComponent>("VAL"));
  dirichletbundcomponents.emplace_back(
      std::make_shared<RealVectorComponent>("VAL", LengthFromInt("NUMDOF")));
  dirichletbundcomponents.emplace_back(std::make_shared<SeparatorComponent>("FUNCT"));
  dirichletbundcomponents.emplace_back(std::shared_ptr<LineComponent>(
      new IntVectorComponent("FUNCT", LengthFromInt("NUMDOF"), {0, true, false})));

  // optional
  dirichletbundcomponents.emplace_back(std::make_shared<SeparatorComponent>("TAG", "", true));
  dirichletbundcomponents.emplace_back(std::make_shared<SelectionComponent>("TAG", "none",
      Teuchos::tuple<std::string>("none", "monitor_reaction"),
      Teuchos::tuple<std::string>("none", "monitor_reaction"), true));

  // define standard Neumann condition components
  std::vector<std::shared_ptr<LineComponent>> neumanncomponents;

  neumanncomponents.emplace_back(std::make_shared<SeparatorComponent>("NUMDOF"));
  neumanncomponents.emplace_back(std::make_shared<IntComponent>("NUMDOF"));

  neumanncomponents.emplace_back(std::make_shared<SeparatorComponent>("ONOFF"));
  neumanncomponents.emplace_back(
      std::make_shared<IntVectorComponent>("ONOFF", LengthFromInt("NUMDOF")));
  neumanncomponents.emplace_back(std::make_shared<SeparatorComponent>("VAL"));
  neumanncomponents.emplace_back(
      std::make_shared<RealVectorComponent>("VAL", LengthFromInt("NUMDOF")));
  neumanncomponents.emplace_back(std::make_shared<SeparatorComponent>("FUNCT"));
  neumanncomponents.emplace_back(std::shared_ptr<LineComponent>(
      new IntVectorComponent("FUNCT", LengthFromInt("NUMDOF"), {0, true, false})));

  // optional
  neumanncomponents.emplace_back(std::make_shared<SeparatorComponent>("TYPE", "", true));
  neumanncomponents.emplace_back(std::make_shared<SelectionComponent>("TYPE", "Live",
      Teuchos::tuple<std::string>(
          "Live", "Dead", "pseudo_orthopressure", "orthopressure", "PressureGrad"),
      Teuchos::tuple<std::string>("neum_live", "neum_dead", "neum_pseudo_orthopressure",
          "neum_orthopressure", "neum_pgrad"),
      true));

  std::shared_ptr<Core::Conditions::ConditionDefinition> movingfluid =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN FLUID MESH VOL CONDITIONS",
          "FluidMesh", "Fluid Mesh", Core::Conditions::FluidMesh, true,
          Core::Conditions::geometry_type_volume);
  std::shared_ptr<Core::Conditions::ConditionDefinition> fluidfluidcoupling =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN FLUID FLUID COUPLING SURF CONDITIONS", "FluidFluidCoupling",
          "FLUID FLUID Coupling", Core::Conditions::FluidFluidCoupling, true,
          Core::Conditions::geometry_type_surface);
  std::shared_ptr<Core::Conditions::ConditionDefinition> ALEfluidcoupling =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN ALE FLUID COUPLING SURF CONDITIONS", "ALEFluidCoupling", "ALE FLUID Coupling",
          Core::Conditions::ALEFluidCoupling, true, Core::Conditions::geometry_type_surface);

  for (const auto& cond : {movingfluid, fluidfluidcoupling, ALEfluidcoupling})
  {
    add_named_int(cond, "COUPLINGID");

    condlist.emplace_back(cond);
  }

  /*--------------------------------------------------------------------*/
  // XFEM coupling conditions

  //*----------------*/
  // Displacement surface condition for XFEM WDBC and Neumann boundary conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_surf_displacement =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN XFEM DISPLACEMENT SURF CONDITIONS", "XFEMSurfDisplacement",
          "XFEM Surf Displacement", Core::Conditions::XFEM_Surf_Displacement, true,
          Core::Conditions::geometry_type_surface);

  add_named_int(xfem_surf_displacement, "COUPLINGID");
  add_named_selection_component(xfem_surf_displacement, "EVALTYPE", "", "funct",
      Teuchos::tuple<std::string>("zero", "funct", "implementation"),
      Teuchos::tuple<std::string>("zero", "funct", "implementation"), true);

  for (unsigned i = 0; i < dirichletbundcomponents.size(); ++i)
  {
    xfem_surf_displacement->add_component(dirichletbundcomponents[i]);
  }

  condlist.push_back(xfem_surf_displacement);

  //*----------------*/
  // Levelset field condition components

  std::vector<std::shared_ptr<Input::LineComponent>> levelsetfield_components;

  levelsetfield_components.push_back(std::make_shared<Input::SeparatorComponent>("COUPLINGID"));
  levelsetfield_components.push_back(std::make_shared<Input::IntComponent>("COUPLINGID"));

  levelsetfield_components.push_back(
      std::make_shared<Input::SeparatorComponent>("LEVELSETFIELDNO"));
  levelsetfield_components.push_back(std::make_shared<Input::IntComponent>("LEVELSETFIELDNO"));

  // define which boolean operator is used for combining this level-set field with the previous one
  // with smaller coupling id
  levelsetfield_components.push_back(std::make_shared<Input::SeparatorComponent>("BOOLEANTYPE"));
  levelsetfield_components.push_back(std::make_shared<Input::SelectionComponent>("BOOLEANTYPE",
      "none", Teuchos::tuple<std::string>("none", "cut", "union", "difference", "sym_difference"),
      Teuchos::tuple<std::string>("none", "cut", "union", "difference", "sym_difference"), false));

  // define which complementary operator is applied after combining the level-set field with a
  // boolean operator with the previous one
  levelsetfield_components.push_back(std::make_shared<Input::SeparatorComponent>("COMPLEMENTARY"));
  levelsetfield_components.push_back(std::make_shared<Input::BoolComponent>("COMPLEMENTARY"));


  //*----------------*/
  // Levelset based Weak Dirichlet conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_levelset_wdbc =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN XFEM LEVELSET WEAK DIRICHLET VOL CONDITIONS", "XFEMLevelsetWeakDirichlet",
          "XFEM Levelset Weak Dirichlet", Core::Conditions::XFEM_Levelset_Weak_Dirichlet, true,
          Core::Conditions::geometry_type_volume);

  for (unsigned i = 0; i < levelsetfield_components.size(); ++i)
  {
    xfem_levelset_wdbc->add_component(levelsetfield_components[i]);
  }

  for (unsigned i = 0; i < dirichletbundcomponents.size(); ++i)
  {
    xfem_levelset_wdbc->add_component(dirichletbundcomponents[i]);
  }

  // optional: allow for random noise, set percentage used in uniform random distribution
  add_named_real(xfem_levelset_wdbc, "RANDNOISE",
      "set percentage of random noise used in uniform random distribution", 0.0, true);

  condlist.push_back(xfem_levelset_wdbc);

  //*----------------*/
  // Levelset based Neumann conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_levelset_neumann =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN XFEM LEVELSET NEUMANN VOL CONDITIONS", "XFEMLevelsetNeumann",
          "XFEM Levelset Neumann", Core::Conditions::XFEM_Levelset_Neumann, true,
          Core::Conditions::geometry_type_volume);

  for (unsigned i = 0; i < levelsetfield_components.size(); ++i)
  {
    xfem_levelset_neumann->add_component(levelsetfield_components[i]);
  }

  for (unsigned i = 0; i < neumanncomponents.size(); ++i)
  {
    xfem_levelset_neumann->add_component(neumanncomponents[i]);
  }

  // define if we use inflow stabilization on the xfem neumann surf condition
  add_named_bool(xfem_levelset_neumann, "INFLOW_STAB", "", false, true);
  condlist.push_back(xfem_levelset_neumann);

  //*----------------*/
  // Levelset based Navier Slip conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_levelset_navier_slip =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN XFEM LEVELSET NAVIER SLIP VOL CONDITIONS", "XFEMLevelsetNavierSlip",
          "XFEM Levelset Navier Slip", Core::Conditions::XFEM_Levelset_Navier_Slip, true,
          Core::Conditions::geometry_type_volume);

  for (unsigned i = 0; i < levelsetfield_components.size(); ++i)
  {
    xfem_levelset_navier_slip->add_component(levelsetfield_components[i]);
  }

  add_named_selection_component(xfem_levelset_navier_slip, "SURFACE_PROJECTION", "", "proj_normal",
      Teuchos::tuple<std::string>(
          "proj_normal", "proj_smoothed", "proj_normal_smoothed_comb", "proj_normal_phi"),
      Teuchos::tuple<int>(Inpar::XFEM::Proj_normal, Inpar::XFEM::Proj_smoothed,
          Inpar::XFEM::Proj_normal_smoothed_comb, Inpar::XFEM::Proj_normal_phi),
      true);
  add_named_int(xfem_levelset_navier_slip, "L2_PROJECTION_SOLVER", "", 0, false, false);
  add_named_int(xfem_levelset_navier_slip, "ROBIN_DIRICHLET_ID", "", 0, false, true);
  add_named_int(xfem_levelset_navier_slip, "ROBIN_NEUMANN_ID", "", 0, false, true);
  add_named_real(xfem_levelset_navier_slip, "SLIPCOEFFICIENT");
  add_named_int(xfem_levelset_navier_slip, "FUNCT", "slip function id", 0, true, false);
  add_named_bool(xfem_levelset_navier_slip, "FORCE_ONLY_TANG_VEL", "", false, true);

  condlist.push_back(xfem_levelset_navier_slip);

  // Add condition XFEM DIRICHLET/NEUMANN?

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_navier_slip_robin_dirch =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN XFEM ROBIN DIRICHLET VOL CONDITIONS", "XFEMRobinDirichletVol",
          "XFEM Robin Dirichlet Volume", Core::Conditions::XFEM_Robin_Dirichlet_Volume, true,
          Core::Conditions::geometry_type_volume);

  add_named_int(xfem_navier_slip_robin_dirch, "ROBIN_ID", "robin id", 0, false, true);

  for (const auto& dirichletbundcomponent : dirichletbundcomponents)
  {
    xfem_navier_slip_robin_dirch->add_component(dirichletbundcomponent);
  }

  condlist.push_back(xfem_navier_slip_robin_dirch);

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_navier_slip_robin_neumann =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN XFEM ROBIN NEUMANN VOL CONDITIONS", "XFEMRobinNeumannVol",
          "XFEM Robin Neumann Volume", Core::Conditions::XFEM_Robin_Neumann_Volume, true,
          Core::Conditions::geometry_type_volume);

  add_named_int(xfem_navier_slip_robin_neumann, "ROBIN_ID", "robin id", 0, false, true);

  for (const auto& neumanncomponent : neumanncomponents)
  {
    xfem_navier_slip_robin_neumann->add_component(neumanncomponent);
  }

  condlist.push_back(xfem_navier_slip_robin_neumann);


  //*----------------*/
  // Levelset based Twophase conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_levelset_twophase =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN XFEM LEVELSET TWOPHASE VOL CONDITIONS", "XFEMLevelsetTwophase",
          "XFEM Levelset Twophase", Core::Conditions::XFEM_Levelset_Twophase, true,
          Core::Conditions::geometry_type_volume);

  for (unsigned i = 0; i < levelsetfield_components.size(); ++i)
  {
    xfem_levelset_twophase->add_component(levelsetfield_components[i]);
  }

  condlist.push_back(xfem_levelset_twophase);

  //*----------------*/
  // Surface Fluid-Fluid coupling conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_surf_fluidfluid =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN XFEM FLUIDFLUID SURF CONDITIONS", "XFEMSurfFluidFluid", "XFEM Surf FluidFluid",
          Core::Conditions::XFEM_Surf_FluidFluid, true, Core::Conditions::geometry_type_surface);

  add_named_int(xfem_surf_fluidfluid, "COUPLINGID");
  add_named_selection_component(xfem_surf_fluidfluid, "COUPSTRATEGY", "coupling strategy", "xfluid",
      Teuchos::tuple<std::string>("xfluid", "embedded", "mean"),
      Teuchos::tuple<int>(
          Inpar::XFEM::Xfluid_Sided, Inpar::XFEM::Embedded_Sided, Inpar::XFEM::Mean));

  condlist.push_back(xfem_surf_fluidfluid);

  //*----------------*/
  // Surface partitioned XFSI boundary conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_surf_fsi_part =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN XFEM FSI PARTITIONED SURF CONDITIONS", "XFEMSurfFSIPart", "XFEM Surf FSI Part",
          Core::Conditions::XFEM_Surf_FSIPart, true, Core::Conditions::geometry_type_surface);

  add_named_int(xfem_surf_fsi_part, "COUPLINGID");

  // COUPSTRATEGY IS FLUID SIDED
  add_named_selection_component(xfem_surf_fsi_part, "INTLAW", "", "noslip",
      Teuchos::tuple<std::string>("noslip", "noslip_splitpen", "slip", "navslip"),
      Teuchos::tuple<int>(Inpar::XFEM::noslip, Inpar::XFEM::noslip_splitpen, Inpar::XFEM::slip,
          Inpar::XFEM::navierslip),
      true);
  add_named_real(xfem_surf_fsi_part, "SLIPCOEFFICIENT", "", 0.0, true);
  add_named_int(xfem_surf_fsi_part, "SLIP_FUNCT", "slip function id", 0, true, false);

  condlist.push_back(xfem_surf_fsi_part);

  //*----------------*/
  // Surface monolithic XFSI coupling conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_surf_fsi_mono =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN XFEM FSI MONOLITHIC SURF CONDITIONS", "XFEMSurfFSIMono", "XFEM Surf FSI Mono",
          Core::Conditions::XFEM_Surf_FSIMono, true, Core::Conditions::geometry_type_surface);

  add_named_int(xfem_surf_fsi_mono, "COUPLINGID");
  add_named_selection_component(xfem_surf_fsi_mono, "COUPSTRATEGY", "", "xfluid",
      Teuchos::tuple<std::string>("xfluid", "solid", "mean", "harmonic"),
      Teuchos::tuple<int>(Inpar::XFEM::Xfluid_Sided, Inpar::XFEM::Embedded_Sided, Inpar::XFEM::Mean,
          Inpar::XFEM::Harmonic),
      true);
  add_named_selection_component(xfem_surf_fsi_mono, "INTLAW", "", "noslip",
      Teuchos::tuple<std::string>(
          "noslip", "noslip_splitpen", "slip", "navslip", "navslip_contact"),
      Teuchos::tuple<int>(Inpar::XFEM::noslip, Inpar::XFEM::noslip_splitpen, Inpar::XFEM::slip,
          Inpar::XFEM::navierslip, Inpar::XFEM::navierslip_contact),
      true);
  add_named_real(xfem_surf_fsi_mono, "SLIPCOEFFICIENT", "", 0.0, true);
  add_named_int(xfem_surf_fsi_mono, "SLIP_FUNCT", "slip function id", 0, true, false);

  condlist.push_back(xfem_surf_fsi_mono);

  //*----------------*/
  // Surface monolithic XFPI coupling conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_surf_fpi_mono =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN XFEM FPI MONOLITHIC SURF CONDITIONS", "XFEMSurfFPIMono", "XFEM Surf FPI Mono",
          Core::Conditions::XFEM_Surf_FPIMono, true, Core::Conditions::geometry_type_surface);

  add_named_int(xfem_surf_fpi_mono, "COUPLINGID");
  add_named_real(xfem_surf_fpi_mono, "BJ_COEFF", "", 0, true);
  add_named_selection_component(xfem_surf_fpi_mono, "Variant", "variant", "BJ",
      Teuchos::tuple<std::string>("BJ", "BJS"), Teuchos::tuple<std::string>("BJ", "BJS"), true);
  add_named_selection_component(xfem_surf_fpi_mono, "Method", "method", "NIT",
      Teuchos::tuple<std::string>("NIT", "SUB"), Teuchos::tuple<std::string>("NIT", "SUB"), true);
  add_named_bool(xfem_surf_fpi_mono, "Contact", "contact", false, true);

  condlist.push_back(xfem_surf_fpi_mono);


  //*----------------*/
  // Surface Weak Dirichlet conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_surf_wdbc =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN XFEM WEAK DIRICHLET SURF CONDITIONS", "XFEMSurfWeakDirichlet",
          "XFEM Surf Weak Dirichlet", Core::Conditions::XFEM_Surf_Weak_Dirichlet, true,
          Core::Conditions::geometry_type_surface);

  add_named_int(xfem_surf_wdbc, "COUPLINGID");
  add_named_selection_component(xfem_surf_wdbc, "EVALTYPE", "", "funct_interpolated",
      Teuchos::tuple<std::string>("zero", "funct_interpolated", "funct_gausspoint",
          "displacement_1storder_wo_initfunct", "displacement_2ndorder_wo_initfunct",
          "displacement_1storder_with_initfunct", "displacement_2ndorder_with_initfunct"),
      Teuchos::tuple<std::string>("zero", "funct_interpolated", "funct_gausspoint",
          "displacement_1storder_wo_initfunct", "displacement_2ndorder_wo_initfunct",
          "displacement_1storder_with_initfunct", "displacement_2ndorder_with_initfunct"),
      true);

  for (unsigned i = 0; i < dirichletbundcomponents.size(); ++i)
  {
    xfem_surf_wdbc->add_component(dirichletbundcomponents[i]);
  }

  // optional: allow for random noise, set percentage used in uniform random distribution
  add_named_real(xfem_surf_wdbc, "RANDNOISE",
      "set percentage of random noise used in uniform random distribution", 0.0, true);

  condlist.push_back(xfem_surf_wdbc);


  //*----------------*/
  // Surface Neumann conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_surf_neumann =
      std::make_shared<Core::Conditions::ConditionDefinition>("DESIGN XFEM NEUMANN SURF CONDITIONS",
          "XFEMSurfNeumann", "XFEM Surf Neumann", Core::Conditions::XFEM_Surf_Neumann, true,
          Core::Conditions::geometry_type_surface);

  add_named_int(xfem_surf_neumann, "COUPLINGID");

  for (unsigned i = 0; i < neumanncomponents.size(); ++i)
  {
    xfem_surf_neumann->add_component(neumanncomponents[i]);
  }

  // define if we use inflow stabilization on the xfem neumann surf condition
  add_named_bool(xfem_surf_neumann, "INFLOW_STAB", "toggle inflow stabilization", false, true);

  condlist.push_back(xfem_surf_neumann);

  //*----------------*/
  // Surface Navier Slip conditions

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_surf_navier_slip =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN XFEM NAVIER SLIP SURF CONDITIONS", "XFEMSurfNavierSlip", "XFEM Surf Navier Slip",
          Core::Conditions::XFEM_Surf_Navier_Slip, true, Core::Conditions::geometry_type_surface);

  add_named_int(xfem_surf_navier_slip, "COUPLINGID");
  add_named_selection_component(xfem_surf_navier_slip, "EVALTYPE", "", "funct_interpolated",
      Teuchos::tuple<std::string>("zero", "funct_interpolated", "funct_gausspoint",
          "displacement_1storder_wo_initfunct", "displacement_2ndorder_wo_initfunct",
          "displacement_1storder_with_initfunct", "displacement_2ndorder_with_initfunct"),
      Teuchos::tuple<std::string>("zero", "funct_interpolated", "funct_gausspoint",
          "displacement_1storder_wo_initfunct", "displacement_2ndorder_wo_initfunct",
          "displacement_1storder_with_initfunct", "displacement_2ndorder_with_initfunct"),
      true);
  add_named_int(xfem_surf_navier_slip, "ROBIN_DIRICHLET_ID", "", 0, false, true);
  add_named_int(xfem_surf_navier_slip, "ROBIN_NEUMANN_ID", "", 0, false, true);
  add_named_real(xfem_surf_navier_slip, "SLIPCOEFFICIENT");
  add_named_int(xfem_surf_navier_slip, "FUNCT", "slip function id", 0, true, false);
  add_named_bool(xfem_surf_navier_slip, "FORCE_ONLY_TANG_VEL", "", false, true);

  condlist.push_back(xfem_surf_navier_slip);

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_navier_slip_robin_dirch_surf =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN XFEM ROBIN DIRICHLET SURF CONDITIONS", "XFEMRobinDirichletSurf",
          "XFEM Robin Dirichlet Volume", Core::Conditions::XFEM_Robin_Dirichlet_Surf, true,
          Core::Conditions::geometry_type_surface);

  // this implementation should be reviewed at some point as it requires these conditions
  //  to have a couplingID. In theory this should not be necessary.
  add_named_int(xfem_navier_slip_robin_dirch_surf, "COUPLINGID");
  add_named_int(xfem_navier_slip_robin_dirch_surf, "ROBIN_ID", "robin id", 0, false, true);

  // Likely, not necessary. But needed for the current structure.
  add_named_selection_component(xfem_navier_slip_robin_dirch_surf, "EVALTYPE", "",
      "funct_interpolated",
      Teuchos::tuple<std::string>("zero", "funct_interpolated", "funct_gausspoint",
          "displacement_1storder_wo_initfunct", "displacement_2ndorder_wo_initfunct",
          "displacement_1storder_with_initfunct", "displacement_2ndorder_with_initfunct"),
      Teuchos::tuple<std::string>("zero", "funct_interpolated", "funct_gausspoint",
          "displacement_1storder_wo_initfunct", "displacement_2ndorder_wo_initfunct",
          "displacement_1storder_with_initfunct", "displacement_2ndorder_with_initfunct"),
      true);

  for (unsigned i = 0; i < dirichletbundcomponents.size(); ++i)
  {
    xfem_navier_slip_robin_dirch_surf->add_component(dirichletbundcomponents[i]);
  }

  condlist.push_back(xfem_navier_slip_robin_dirch_surf);

  std::shared_ptr<Core::Conditions::ConditionDefinition> xfem_navier_slip_robin_neumann_surf =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN XFEM ROBIN NEUMANN SURF CONDITIONS", "XFEMRobinNeumannSurf",
          "XFEM Robin Neumann Volume", Core::Conditions::XFEM_Robin_Neumann_Surf, true,
          Core::Conditions::geometry_type_surface);

  // this implementation should be reviewed at some point as it requires these conditions
  //  to have a couplingID. In theory this should not be necessary.
  add_named_int(xfem_navier_slip_robin_neumann_surf, "COUPLINGID");
  add_named_int(xfem_navier_slip_robin_neumann_surf, "ROBIN_ID", "robin id", 0, false, true);

  for (unsigned i = 0; i < neumanncomponents.size(); ++i)
  {
    xfem_navier_slip_robin_neumann_surf->add_component(neumanncomponents[i]);
  }

  condlist.push_back(xfem_navier_slip_robin_neumann_surf);

  //*----------------*/
  // Solid to solid embedded mesh coupling conditions
  std::shared_ptr<Core::Conditions::ConditionDefinition> solid_surf_coupling =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN EMBEDDED MESH SOLID SURF COUPLING CONDITIONS", "EmbeddedMeshSolidSurfCoupling",
          "Embedded Mesh Solid Surface Coupling",
          Core::Conditions::Embedded_Mesh_Solid_Surf_Coupling, true,
          Core::Conditions::geometry_type_surface);

  add_named_int(solid_surf_coupling, "COUPLINGID");

  condlist.push_back(solid_surf_coupling);

  // Solid to solid embedded mesh volume background mesh condition
  std::shared_ptr<Core::Conditions::ConditionDefinition> solid_vol_background_coupling =
      std::make_shared<Core::Conditions::ConditionDefinition>(
          "DESIGN EMBEDDED SOLID VOL BACKGROUND CONDITIONS", "EmbeddedMeshSolidVolBackground",
          "Embedded Mesh Solid Volume Background",
          Core::Conditions::Embedded_Mesh_Solid_Volume_Background, true,
          Core::Conditions::geometry_type_volume);

  add_named_int(solid_vol_background_coupling, "COUPLINGID");
  condlist.push_back(solid_vol_background_coupling);
}

FOUR_C_NAMESPACE_CLOSE
