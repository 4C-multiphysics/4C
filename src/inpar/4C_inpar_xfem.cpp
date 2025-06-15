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
FOUR_C_NAMESPACE_OPEN



void Inpar::XFEM::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using namespace Core::IO::InputSpecBuilders;

  list["XFEM GENERAL"] = group("XFEM GENERAL",
      {

          // OUTPUT options
          parameter<bool>("GMSH_DEBUG_OUT",
              {.description = "Do you want to write extended Gmsh output for each timestep?",
                  .default_value = false}),
          parameter<bool>("GMSH_DEBUG_OUT_SCREEN",
              {.description = "Do you want to be informed, if Gmsh output is written?",
                  .default_value = false}),
          parameter<bool>("GMSH_SOL_OUT",
              {.description = "Do you want to write extended Gmsh output for each timestep?",
                  .default_value = false}),
          parameter<bool>("GMSH_TIMINT_OUT",
              {.description = "Do you want to write extended Gmsh output for each timestep?",
                  .default_value = false}),
          parameter<bool>("GMSH_EOS_OUT",
              {.description = "Do you want to write extended Gmsh output for each timestep?",
                  .default_value = false}),
          parameter<bool>("GMSH_DISCRET_OUT",
              {.description = "Do you want to write extended Gmsh output for each timestep?",
                  .default_value = false}),
          parameter<bool>("GMSH_CUT_OUT",
              {.description = "Do you want to write extended Gmsh output for each timestep?",
                  .default_value = false}),
          parameter<bool>("PRINT_OUTPUT",
              {.description = "Is the output of the cut process desired?", .default_value = false}),

          parameter<int>("MAX_NUM_DOFSETS",
              {.description = "Maximum number of volumecells in the XFEM element",
                  .default_value = 3}),


          deprecated_selection<Cut::NodalDofSetStrategy>("NODAL_DOFSET_STRATEGY",
              {
                  {"OneDofset_PerNodeAndPosition", Cut::NDS_Strategy_OneDofset_PerNodeAndPosition},
                  {"ConnectGhostDofsets_PerNodeAndPosition",
                      Cut::NDS_Strategy_ConnectGhostDofsets_PerNodeAndPosition},
                  {"full", Cut::NDS_Strategy_full},
              },
              {.description = "Strategy used for the nodal dofset management per node",
                  .default_value = Cut::NDS_Strategy_full}),

          // Integration options
          deprecated_selection<Cut::VCellGaussPts>("VOLUME_GAUSS_POINTS_BY",
              {
                  {"Tessellation", Cut::VCellGaussPts_Tessellation},
                  {"MomentFitting", Cut::VCellGaussPts_MomentFitting},
                  {"DirectDivergence", Cut::VCellGaussPts_DirectDivergence},
              },
              {.description = "Method for finding Gauss Points for the cut volumes",
                  .default_value = Cut::VCellGaussPts_Tessellation}),


          deprecated_selection<Cut::BCellGaussPts>("BOUNDARY_GAUSS_POINTS_BY",
              {
                  {"Tessellation", Cut::BCellGaussPts_Tessellation},
                  {"MomentFitting", Cut::BCellGaussPts_MomentFitting},
                  {"DirectDivergence", Cut::BCellGaussPts_DirectDivergence},
              },
              {.description = "Method for finding Gauss Points for the boundary cells",
                  .default_value = Cut::BCellGaussPts_Tessellation})},
      {.defaultable =
              true}); /*----------------------------------------------------------------------*/
  list["XFLUID DYNAMIC/GENERAL"] = group("XFLUID DYNAMIC/GENERAL",
      {

          // Do we use more than one fluid discretization?
          parameter<bool>("XFLUIDFLUID",
              {.description = "Use an embedded fluid patch.", .default_value = false}),

          // How many monolithic steps we keep the fluidfluid-boundary fixed
          parameter<int>("RELAXING_ALE_EVERY",
              {.description = "Relaxing Ale after how many monolithic steps", .default_value = 1}),

          parameter<bool>("RELAXING_ALE",
              {.description = "switch on/off for relaxing Ale in monolithic fluid-fluid-fsi",
                  .default_value = true}),

          parameter<double>("XFLUIDFLUID_SEARCHRADIUS",
              {.description = "Radius of the search tree", .default_value = 1.0}),

          // xfluidfluid-fsi-monolithic approach
          deprecated_selection<MonolithicXffsiApproach>("MONOLITHIC_XFFSI_APPROACH",
              {
                  {"xffsi_full_newton", Inpar::XFEM::XFFSI_Full_Newton},
                  {"xffsi_fixedALE_interpolation", Inpar::XFEM::XFFSI_FixedALE_Interpolation},
                  {"xffsi_fixedALE_partitioned", Inpar::XFEM::XFFSI_FixedALE_Partitioned},
              },
              {.description = "The monolithic approach for xfluidfluid-fsi",
                  .default_value = Inpar::XFEM::XFFSI_FixedALE_Partitioned}),

          // xfluidfluid time integration approach
          parameter<XFluidFluidTimeInt>(
              "XFLUIDFLUID_TIMEINT", {.description = "The xfluidfluid-timeintegration approach",
                                         .default_value = Inpar::XFEM::Xff_TimeInt_FullProj}),

          deprecated_selection<XFluidTimeIntScheme>("XFLUID_TIMEINT",
              {
                  {"STD=COPY_and_GHOST=COPY/GP",
                      Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_AND_GHOST_by_Copy_or_GP},
                  {"STD=COPY/SL_and_GHOST=COPY/GP",
                      Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_SL_AND_GHOST_by_Copy_or_GP},
                  {"STD=SL(boundary-zone)_and_GHOST=GP",
                      Inpar::XFEM::Xf_TimeIntScheme_STD_by_SL_cut_zone_AND_GHOST_by_GP},
                  {"STD=COPY/PROJ_and_GHOST=COPY/PROJ/GP",
                      Inpar::XFEM::
                          Xf_TimeIntScheme_STD_by_Copy_or_Proj_AND_GHOST_by_Proj_or_Copy_or_GP},
              },
              {.description = "The xfluid time integration approach",
                  .default_value =
                      Inpar::XFEM::Xf_TimeIntScheme_STD_by_Copy_or_SL_AND_GHOST_by_Copy_or_GP}),

          parameter<bool>(
              "ALE_XFluid", {.description = "XFluid is Ale Fluid?", .default_value = false}),

          // for new OST-implementation: which interface terms to be evaluated for previous time
          // step
          parameter<InterfaceTermsPreviousState>("INTERFACE_TERMS_PREVIOUS_STATE",
              {.description = "how to treat interface terms from previous time step (new OST)",
                  .default_value = Inpar::XFEM::PreviousState_only_consistency}),

          parameter<bool>("XFLUID_TIMEINT_CHECK_INTERFACETIPS",
              {.description = "Xfluid TimeIntegration Special Check if node has changed the side!",
                  .default_value = true}),

          parameter<bool>("XFLUID_TIMEINT_CHECK_SLIDINGONSURFACE",
              {.description = "Xfluid TimeIntegration Special Check if node is sliding on surface!",
                  .default_value = true})},
      {.defaultable = true});

  /*----------------------------------------------------------------------*/
  list["XFLUID DYNAMIC/STABILIZATION"] = group("XFLUID DYNAMIC/STABILIZATION",
      {

          // Boundary-Coupling options
          parameter<CouplingMethod>("COUPLING_METHOD",
              {.description = "method how to enforce embedded boundary/coupling conditions at the "
                              "interface",
                  .default_value = Inpar::XFEM::Nitsche}),

          deprecated_selection<HybridLmL2Proj>("HYBRID_LM_L2_PROJ",
              {
                  {"full_ele_proj", Inpar::XFEM::Hybrid_LM_L2_Proj_full},
                  {"part_ele_proj", Inpar::XFEM::Hybrid_LM_L2_Proj_part},
              },
              {.description =
                      "perform the L2 projection between stress fields on whole element or on "
                      "fluid part?",
                  .default_value = Inpar::XFEM::Hybrid_LM_L2_Proj_part}),

          deprecated_selection<AdjointScaling>("VISC_ADJOINT_SYMMETRY",
              {
                  {"yes", Inpar::XFEM::adj_sym},
                  {"no", Inpar::XFEM::adj_skew},
                  {"sym", Inpar::XFEM::adj_sym},
                  {"skew", Inpar::XFEM::adj_skew},
                  {"none", Inpar::XFEM::adj_none},
              },
              {.description = "viscous and adjoint viscous interface terms with matching sign?",
                  .default_value = Inpar::XFEM::adj_sym}),

          // viscous and convective Nitsche/MSH stabilization parameter
          parameter<double>("NIT_STAB_FAC",
              {.description = " ( stabilization parameter for Nitsche's penalty term",
                  .default_value = 35.0}),
          parameter<double>("NIT_STAB_FAC_TANG",
              {.description = " ( stabilization parameter for Nitsche's penalty tangential term",
                  .default_value = 35.0}),

          deprecated_selection<ViscStabTraceEstimate>("VISC_STAB_TRACE_ESTIMATE",
              {
                  {"CT_div_by_hk", Inpar::XFEM::ViscStab_TraceEstimate_CT_div_by_hk},
                  {"eigenvalue", Inpar::XFEM::ViscStab_TraceEstimate_eigenvalue},
              },
              {.description =
                      "how to estimate the scaling from the trace inequality in Nitsche's method",
                  .default_value = Inpar::XFEM::ViscStab_TraceEstimate_CT_div_by_hk}),


          deprecated_selection<TraceEstimateEigenvalueUpdate>("UPDATE_EIGENVALUE_TRACE_ESTIMATE",
              {
                  {"every_iter", Inpar::XFEM::Eigenvalue_update_every_iter},
                  {"every_timestep", Inpar::XFEM::Eigenvalue_update_every_timestep},
                  {"once", Inpar::XFEM::Eigenvalue_update_once},
              },
              {.description = "how often should the local eigenvalue problem be updated",
                  .default_value = Inpar::XFEM::Eigenvalue_update_every_iter}),

          deprecated_selection<ViscStabHk>("VISC_STAB_HK",
              {
                  {"vol_equivalent", Inpar::XFEM::ViscStab_hk_vol_equivalent},
                  {"cut_vol_div_by_cut_surf", Inpar::XFEM::ViscStab_hk_cut_vol_div_by_cut_surf},
                  {"ele_vol_div_by_cut_surf", Inpar::XFEM::ViscStab_hk_ele_vol_div_by_cut_surf},
                  {"ele_vol_div_by_ele_surf", Inpar::XFEM::ViscStab_hk_ele_vol_div_by_ele_surf},
                  {"ele_vol_div_by_max_ele_surf",
                      Inpar::XFEM::ViscStab_hk_ele_vol_div_by_max_ele_surf},
              },
              {.description = "how to define the characteristic element length in cut elements",
                  .default_value = Inpar::XFEM::ViscStab_hk_ele_vol_div_by_max_ele_surf}),


          deprecated_selection<ConvStabScaling>("CONV_STAB_SCALING",
              {
                  {"inflow", Inpar::XFEM::ConvStabScaling_inflow},
                  {"abs_inflow", Inpar::XFEM::ConvStabScaling_abs_inflow},
                  {"none", Inpar::XFEM::ConvStabScaling_none},
              },
              {.description = "scaling factor for viscous interface stabilization (Nitsche, MSH)",
                  .default_value = Inpar::XFEM::ConvStabScaling_none}),

          deprecated_selection<XffConvStabScaling>("XFF_CONV_STAB_SCALING",
              {
                  {"inflow", Inpar::XFEM::XFF_ConvStabScaling_upwinding},
                  {"averaged", Inpar::XFEM::XFF_ConvStabScaling_only_averaged},
                  {"none", Inpar::XFEM::XFF_ConvStabScaling_none},
              },
              {.description = "scaling factor for convective interface stabilization of "
                              "fluid-fluid Coupling",
                  .default_value = Inpar::XFEM::XFF_ConvStabScaling_none}),

          deprecated_selection<MassConservationCombination>("MASS_CONSERVATION_COMBO",
              {
                  {"max", Inpar::XFEM::MassConservationCombination_max},
                  {"sum", Inpar::XFEM::MassConservationCombination_sum},
              },
              {.description =
                      "choose the maximum from viscous and convective contributions or just "
                      "sum both up",
                  .default_value = Inpar::XFEM::MassConservationCombination_max}),


          deprecated_selection<MassConservationScaling>("MASS_CONSERVATION_SCALING",
              {
                  {"full", Inpar::XFEM::MassConservationScaling_full},
                  {"only_visc", Inpar::XFEM::MassConservationScaling_only_visc},
              },
              {.description =
                      "apply additional scaling of penalty term to enforce mass conservation "
                      "for convection-dominated flow",
                  .default_value = Inpar::XFEM::MassConservationScaling_only_visc}),

          parameter<bool>("GHOST_PENALTY_STAB",
              {.description = "switch on/off ghost penalty interface stabilization",
                  .default_value = false}),

          parameter<bool>("GHOST_PENALTY_TRANSIENT_STAB",
              {.description = "switch on/off ghost penalty transient interface stabilization",
                  .default_value = false}),

          parameter<bool>(
              "GHOST_PENALTY_2nd_STAB", {.description = "switch on/off ghost penalty interface "
                                                        "stabilization for 2nd order derivatives",
                                            .default_value = false}),
          parameter<bool>("GHOST_PENALTY_2nd_STAB_NORMAL",
              {.description = "switch between ghost penalty interface stabilization for 2nd order "
                              "derivatives in normal or all spatial directions",
                  .default_value = false}),


          parameter<double>("GHOST_PENALTY_FAC",
              {.description =
                      "define stabilization parameter ghost penalty interface stabilization",
                  .default_value = 0.1}),

          parameter<double>("GHOST_PENALTY_TRANSIENT_FAC",
              {.description = "define stabilization parameter ghost penalty transient interface "
                              "stabilization",
                  .default_value = 0.001}),

          parameter<double>("GHOST_PENALTY_2nd_FAC",
              {.description = "define stabilization parameter ghost penalty 2nd "
                              "order viscous interface stabilization",
                  .default_value = 0.05}),
          parameter<double>("GHOST_PENALTY_PRESSURE_2nd_FAC",
              {.description =
                      "define stabilization parameter ghost penalty 2nd order pressure interface "
                      "stabilization",
                  .default_value = 0.05}),


          parameter<bool>("XFF_EOS_PRES_EMB_LAYER",
              {.description =
                      "switch on/off edge-based pressure stabilization on interface-contributing "
                      "elements of the embedded fluid",
                  .default_value = false}),

          parameter<bool>("IS_PSEUDO_2D",
              {.description = "modify viscous interface stabilization due to the vanishing "
                              "polynomial in third dimension when using strong Dirichlet "
                              "conditions to block polynomials in one spatial dimension",
                  .default_value = false}),

          parameter<bool>("GHOST_PENALTY_ADD_INNER_FACES",
              {.description =
                      "Apply ghost penalty stabilization also for inner faces if this is possible "
                      "due to the dofsets",
                  .default_value = false})},
      {.defaultable = true});

  /*----------------------------------------------------------------------*/
  list["XFLUID DYNAMIC/XFPSI MONOLITHIC"] = group("XFLUID DYNAMIC/XFPSI MONOLITHIC",
      {

          parameter<int>("ITEMIN",
              {.description = "How many iterations are performed minimal", .default_value = 1}),
          parameter<int>(
              "ITEMAX_OUTER", {.description = "How many outer iterations are performed maximal",
                                  .default_value = 5}),
          parameter<bool>("ND_NEWTON_DAMPING",
              {.description = "Activate Newton damping based on residual and increment",
                  .default_value = false}),
          parameter<double>(
              "ND_MAX_DISP_ITERINC", {.description = "Maximal displacement increment to apply full "
                                                     "newton --> otherwise damp newton",
                                         .default_value = -1.0}),
          parameter<double>(
              "ND_MAX_VEL_ITERINC", {.description = "Maximal fluid velocity increment to apply "
                                                    "full newton --> otherwise damp newton",
                                        .default_value = -1.0}),
          parameter<double>(
              "ND_MAX_PRES_ITERINC", {.description = "Maximal fluid pressure increment to apply "
                                                     "full newton --> otherwise damp newton",
                                         .default_value = -1.0}),
          parameter<double>("ND_MAX_PVEL_ITERINC",
              {.description = "Maximal porofluid velocity increment to apply "
                              "full newton --> otherwise damp newton",
                  .default_value = -1.0}),
          parameter<double>("ND_MAX_PPRES_ITERINC",
              {.description = "Maximal porofluid pressure increment to apply "
                              "full newton --> otherwise damp newton",
                  .default_value = -1.0}),
          parameter<double>("CUT_EVALUATE_MINTOL",
              {.description = "Minimal value of the maximal structural displacement "
                              "for which the CUT is evaluate in this iteration!",
                  .default_value = 0.0}),
          parameter<bool>("EXTRAPOLATE_TO_ZERO",
              {.description = "the extrapolation of the fluid stress in the contact "
                              "zone is relaxed to zero after a certain distance",
                  .default_value = false}),
          parameter<int>("CUT_EVALUATE_MINITER",
              {.description = "Minimal number of nonlinear iterations, before "
                              "the CUT is potentially not evaluated",
                  .default_value = 0}),
          parameter<double>("POROCONTACTFPSI_HFRACTION",
              {.description =
                      "factor of element size, when transition between FPSI and PSCI is started!",
                  .default_value = 1.0}),
          parameter<double>("POROCONTACTFPSI_FULLPCFRACTION",
              {.description =
                      "ration of gap/(POROCONTACTFPSI_HFRACTION*h) when full PSCI is started!",
                  .default_value = 0.0}),
          parameter<bool>("USE_PORO_PRESSURE",
              {.description = "the extrapolation of the fluid stress in the contact "
                              "zone is relaxed to zero after a certtain distance",
                  .default_value = true})},
      {.defaultable = true});
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
      deprecated_selection<std::string>(
          "TAG", {"none", "monitor_reaction"}, {.default_value = "none"}),
  });

  auto neumanncomponents = all_of({
      parameter<int>("NUMDOF"),
      parameter<std::vector<int>>("ONOFF", {.size = from_parameter<int>("NUMDOF")}),
      parameter<std::vector<double>>("VAL", {.size = from_parameter<int>("NUMDOF")}),
      parameter<std::vector<std::optional<int>>>("FUNCT", {.size = from_parameter<int>("NUMDOF")}),
      deprecated_selection<std::string>("TYPE",
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
  xfem_surf_displacement.add_component(deprecated_selection<std::string>("EVALTYPE",
      {"zero", "funct", "implementation"}, {.description = "", .default_value = "funct"}));

  xfem_surf_displacement.add_component(dirichletbundcomponents);

  condlist.push_back(xfem_surf_displacement);

  //*----------------*/
  // Levelset field condition components

  auto levelsetfield_components = all_of({
      parameter<int>("COUPLINGID"),
      parameter<int>("LEVELSETFIELDNO"),
      deprecated_selection<std::string>("BOOLEANTYPE",
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

  xfem_levelset_navier_slip.add_component(deprecated_selection<ProjToSurface>("SURFACE_PROJECTION",
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
      parameter<int>("FUNCT", {.description = "slip function id", .default_value = 0}));
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
  xfem_surf_fluidfluid.add_component(deprecated_selection<AveragingStrategy>("COUPSTRATEGY",
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
  xfem_surf_fsi_part.add_component(deprecated_selection<InterfaceLaw>("INTLAW",
      {{"noslip", Inpar::XFEM::noslip}, {"noslip_splitpen", Inpar::XFEM::noslip_splitpen},
          {"slip", Inpar::XFEM::slip}, {"navslip", Inpar::XFEM::navierslip}},
      {.description = "", .default_value = Inpar::XFEM::noslip}));
  xfem_surf_fsi_part.add_component(
      parameter<double>("SLIPCOEFFICIENT", {.description = "", .default_value = 0.0}));
  xfem_surf_fsi_part.add_component(
      parameter<int>("SLIP_FUNCT", {.description = "slip function id", .default_value = 0}));

  condlist.push_back(xfem_surf_fsi_part);

  //*----------------*/
  // Surface monolithic XFSI coupling conditions

  Core::Conditions::ConditionDefinition xfem_surf_fsi_mono(
      "DESIGN XFEM FSI MONOLITHIC SURF CONDITIONS", "XFEMSurfFSIMono", "XFEM Surf FSI Mono",
      Core::Conditions::XFEM_Surf_FSIMono, true, Core::Conditions::geometry_type_surface);

  xfem_surf_fsi_mono.add_component(parameter<int>("COUPLINGID"));
  xfem_surf_fsi_mono.add_component(deprecated_selection<AveragingStrategy>("COUPSTRATEGY",
      {{"xfluid", Inpar::XFEM::Xfluid_Sided}, {"solid", Inpar::XFEM::Embedded_Sided},
          {"mean", Inpar::XFEM::Mean}, {"harmonic", Inpar::XFEM::Harmonic}},
      {.description = "", .default_value = Inpar::XFEM::Xfluid_Sided}));
  xfem_surf_fsi_mono.add_component(deprecated_selection<InterfaceLaw>("INTLAW",
      {{"noslip", Inpar::XFEM::noslip}, {"noslip_splitpen", Inpar::XFEM::noslip_splitpen},
          {"slip", Inpar::XFEM::slip}, {"navslip", Inpar::XFEM::navierslip},
          {"navslip_contact", Inpar::XFEM::navierslip_contact}},
      {.description = "", .default_value = Inpar::XFEM::noslip}));
  xfem_surf_fsi_mono.add_component(
      parameter<double>("SLIPCOEFFICIENT", {.description = "", .default_value = 0.0}));
  xfem_surf_fsi_mono.add_component(
      parameter<int>("SLIP_FUNCT", {.description = "slip function id", .default_value = 0}));

  condlist.push_back(xfem_surf_fsi_mono);

  //*----------------*/
  // Surface monolithic XFPI coupling conditions

  Core::Conditions::ConditionDefinition xfem_surf_fpi_mono(
      "DESIGN XFEM FPI MONOLITHIC SURF CONDITIONS", "XFEMSurfFPIMono", "XFEM Surf FPI Mono",
      Core::Conditions::XFEM_Surf_FPIMono, true, Core::Conditions::geometry_type_surface);

  xfem_surf_fpi_mono.add_component(parameter<int>("COUPLINGID"));
  xfem_surf_fpi_mono.add_component(
      parameter<double>("BJ_COEFF", {.description = "", .default_value = 0.}));
  xfem_surf_fpi_mono.add_component(deprecated_selection<std::string>(
      "Variant", {"BJ", "BJS"}, {.description = "variant", .default_value = "BJ"}));
  xfem_surf_fpi_mono.add_component(deprecated_selection<std::string>(
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
  xfem_surf_wdbc.add_component(deprecated_selection<std::string>("EVALTYPE",
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
  xfem_surf_navier_slip.add_component(deprecated_selection<std::string>("EVALTYPE",
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
      parameter<int>("FUNCT", {.description = "slip function id", .default_value = 0}));
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
  xfem_navier_slip_robin_dirch_surf.add_component(deprecated_selection<std::string>("EVALTYPE",
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