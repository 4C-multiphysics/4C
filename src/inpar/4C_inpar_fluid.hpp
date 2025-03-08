// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_INPAR_FLUID_HPP
#define FOUR_C_INPAR_FLUID_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::Conditions
{
  class ConditionDefinition;
}
/*----------------------------------------------------------------------*/
namespace Inpar
{
  namespace FLUID
  {
    //! physical type of the fluid flow (incompressible, weakly compressible, weakly compressible
    //! stokes, Boussinesq approximation, varying density, loma temperature-dependent water)
    enum PhysicalType
    {
      physicaltype_undefined,
      incompressible,
      weakly_compressible,
      weakly_compressible_stokes,
      weakly_compressible_dens_mom,
      weakly_compressible_stokes_dens_mom,
      artcomp,
      boussinesq,
      varying_density,
      loma,
      tempdepwater,
      poro,
      poro_p1,
      stokes,
      oseen
    };

    //! flag to switch the stabilization type
    enum StabType
    {
      stabtype_nostab,
      stabtype_residualbased,
      stabtype_edgebased,
      stabtype_pressureprojection
    };

    [[nodiscard]] std::string to_string(StabType stabtype);

    //! flag to select the type of viscosity stabilization
    enum VStab
    {
      viscous_stab_none,
      viscous_stab_gls,
      viscous_stab_gls_only_rhs,
      viscous_stab_usfem,
      viscous_stab_usfem_only_rhs
    };

    [[nodiscard]] std::string to_string(VStab vstab);


    //! flag to select the type of reactive stabilization
    enum RStab
    {
      reactive_stab_none,
      reactive_stab_gls,
      reactive_stab_usfem
    };

    [[nodiscard]] std::string to_string(RStab rstab);

    //! flag to select the type of cross stress stabilization
    enum CrossStress
    {
      cross_stress_stab_none,
      cross_stress_stab,
      cross_stress_stab_only_rhs
    };

    std::string to_string(CrossStress crossstress);

    //! flag to select the type of Reynolds stress stabilization
    enum ReynoldsStress
    {
      reynolds_stress_stab_none,
      reynolds_stress_stab,
      reynolds_stress_stab_only_rhs
    };

    std::string to_string(ReynoldsStress reynoldsstress);

    //! flag to select time-dependent subgrid-scales
    enum SubscalesTD
    {
      subscales_quasistatic,
      subscales_time_dependent
    };

    enum Transient
    {
      inertia_stab_drop,
      inertia_stab_keep,
      inertia_stab_keep_complete
    };

    [[nodiscard]] std::string to_string(Transient transient);

    /// tau type for residual based fluid stabilizations
    enum TauType
    {
      tau_taylor_hughes_zarins,
      tau_taylor_hughes_zarins_wo_dt,
      tau_taylor_hughes_zarins_whiting_jansen,
      tau_taylor_hughes_zarins_whiting_jansen_wo_dt,
      tau_taylor_hughes_zarins_scaled,
      tau_taylor_hughes_zarins_scaled_wo_dt,
      tau_franca_barrenechea_valentin_frey_wall,
      tau_franca_barrenechea_valentin_frey_wall_wo_dt,
      tau_shakib_hughes_codina,
      tau_shakib_hughes_codina_wo_dt,
      tau_codina,
      tau_codina_wo_dt,
      tau_codina_convscaled,
      tau_franca_madureira_valentin_badia_codina,
      tau_franca_madureira_valentin_badia_codina_wo_dt,
      tau_hughes_franca_balestra_wo_dt,
      tau_not_defined
    };

    std::string to_string(TauType tau);

    /// characteristic element length for tau_Mu
    enum CharEleLengthU
    {
      streamlength_u,
      volume_equivalent_diameter_u,
      root_of_volume_u
    };

    /// characteristic element length for tau_Mp and tau_C
    enum CharEleLengthPC
    {
      streamlength_pc,
      volume_equivalent_diameter_pc,
      root_of_volume_pc
    };

    //! flag to select the type of pressure edge-based (EOS) stabilization
    enum EosPres
    {
      EOS_PRES_none,
      EOS_PRES_std_eos,
      EOS_PRES_xfem_gp
    };

    [[nodiscard]] std::string to_string(EosPres eospres);

    //! flag to select the type of convective streamline edge-based (EOS) stabilization
    enum EosConvStream
    {
      EOS_CONV_STREAM_none,
      EOS_CONV_STREAM_std_eos,
      EOS_CONV_STREAM_xfem_gp
    };

    [[nodiscard]] std::string to_string(EosConvStream eosconvstream);

    //! flag to select the type of convective crosswind edge-based (EOS) stabilization
    enum EosConvCross
    {
      EOS_CONV_CROSS_none,
      EOS_CONV_CROSS_std_eos,
      EOS_CONV_CROSS_xfem_gp
    };

    [[nodiscard]] std::string to_string(EosConvCross eosconvcross);

    //! flag to select the type of divergence EOS stabilization
    enum EosDiv
    {
      EOS_DIV_none,
      EOS_DIV_vel_jump_std_eos,
      EOS_DIV_vel_jump_xfem_gp,
      EOS_DIV_div_jump_std_eos,
      EOS_DIV_div_jump_xfem_gp
    };

    [[nodiscard]] std::string to_string(EosDiv eosdiv);

    /// tau type for edge-oriented / continuous interior penalty stabilization
    enum EosTauType
    {
      EOS_tau_burman,
      EOS_tau_burman_fernandez_hansbo,
      EOS_tau_burman_fernandez_hansbo_wo_dt,
      EOS_tau_braack_burman_john_lube,
      EOS_tau_braack_burman_john_lube_wo_divjump,
      EOS_tau_franca_barrenechea_valentin_wall,
      EOS_tau_burman_fernandez,
      EOS_tau_burman_hansbo_dangelo_zunino,
      EOS_tau_burman_hansbo_dangelo_zunino_wo_dt,
      EOS_tau_schott_massing_burman_dangelo_zunino,
      EOS_tau_schott_massing_burman_dangelo_zunino_wo_dt,
      EOS_tau_Taylor_Hughes_Zarins_Whiting_Jansen_Codina_scaling,
      EOS_tau_poroelast_fluid,
      EOS_tau_not_defined
    };

    [[nodiscard]] std::string to_string(EosTauType eostautype);

    /// Element length for edge-oriented / continuous interior penalty stabilization
    enum EosElementLength
    {
      EOS_he_max_diameter_to_opp_surf,
      EOS_he_max_dist_to_opp_surf,
      EOS_he_surf_with_max_diameter,
      EOS_hk_max_diameter,   // maximal nD diameter of the neighboring elements
      EOS_he_surf_diameter,  // maximal (n-1)D diameter of the internal face/edge
      EOS_he_vol_eq_diameter
    };

    [[nodiscard]] std::string to_string(EosElementLength eoselementlength);

    /// EdgeBased (EOS) and Ghost Penalty matrix pattern
    enum EosGpPattern
    {
      EOS_GP_Pattern_uvwp,
      EOS_GP_Pattern_up,
      EOS_GP_Pattern_full
    };

    /// time integration scheme
    enum TimeIntegrationScheme
    {
      timeint_stationary,
      timeint_one_step_theta,
      timeint_npgenalpha,
      timeint_afgenalpha,
      timeint_bdf2,
    };

    /// One step theta variations
    enum OstContAndPress
    {
      Cont_normal_Press_normal,
      Cont_impl_Press_normal,
      Cont_impl_Press_impl
    };

    /// initial and Oseen advective field
    enum InitialField
    {
      initfield_zero_field,
      initfield_field_by_function,
      initfield_disturbed_field_from_function,
      initfield_flame_vortex_interaction,
      initfield_beltrami_flow,
      initfield_kim_moin_flow,
      initfield_hit_comte_bellot_corrsin,
      initfield_forced_hit_simple_algebraic_spectrum,
      initfield_forced_hit_numeric_spectrum,
      initfield_passive_hit_const_input,
      initfield_channel_weakly_compressible
    };

    /// initial field
    enum CalcError
    {
      no_error_calculation,
      beltrami_flow,
      channel2D,
      gravitation,
      shear_flow,
      byfunct,
      beltrami_stat_stokes,
      beltrami_stat_navier_stokes,
      beltrami_instat_stokes,
      beltrami_instat_navier_stokes,
      kimmoin_stat_stokes,
      kimmoin_stat_navier_stokes,
      kimmoin_instat_stokes,
      kimmoin_instat_navier_stokes,
      fsi_fluid_pusher,  ///< pseudo 1D FSI fluid pusher
      channel_weakly_compressible,
    };

    /// meshtying algorithm
    enum MeshTying
    {
      no_meshtying,
      condensed_smat,
      condensed_bmat,
      condensed_bmat_merged
    };

    /// scheme for gridvel
    enum Gridvel
    {
      BE,    // first order
      BDF2,  // second order
      OST    // one-step-theta
    };

    //! physical turbulence models
    enum TurbModelAction
    {
      no_model,
      smagorinsky,
      smagorinsky_with_van_Driest_damping,
      dynamic_smagorinsky,
      avm3,
      multifractal_subgrid_scales,
      vreman,
      dynamic_vreman
    };

    //! Direction of inflow for turbulence
    enum class TurbInflowDirection
    {
      x,
      y,
      z,
    };

    /// Define forcing for scalar field
    enum ScalarForcing
    {
      scalarforcing_no,
      scalarforcing_isotropic,
      scalarforcing_mean_scalar_gradient
    };

    //! options for fine-scale subgrid viscosity
    enum FineSubgridVisc
    {
      no_fssgv,
      smagorinsky_all,
      smagorinsky_small
    };

    enum VremanFiMethod
    {
      cuberootvol,
      dir_dep,
      min_len
    };

    [[nodiscard]] std::string to_string(VremanFiMethod vremanfimethod);

    //! options for forcing of homogeneous isotropic turbulence
    enum ForcingType
    {
      linear_compensation_from_intermediate_spectrum,
      fixed_power_input
    };

    //! linearisation actions recognized by fluid3 (genalpha implementation)
    enum LinearisationAction
    {
      fixed_point_like,
      Newton
    };

    //!  norm for convergence check of nonlinear iteration
    enum ItNorm
    {
      fncc_L1,        // converg. check with L1 norm
      fncc_L2,        // converg. check with L2 norm
      fncc_L2_wo_res, /* converg. check with L2 norm, no computation
                       *   of residual norm if itmax is reached        */
      fncc_Linf       // converg. check with L-inf. norm
    };

    enum XWallTauwType
    {
      constant,
      between_steps
    };

    enum WSSType  //! wss calculation type
    {
      wss_standard,     //! calculate 'normal' wss
      wss_aggregation,  //! calculate aggregated wss
      wss_mean          //! calculate mean wss
    };

    enum XWallTauwCalcType
    {
      residual,
      gradient,
      gradient_to_residual
    };

    enum XWallBlendingType
    {
      none,
      ramp_function
    };

    enum AdaptiveTimeStepEstimator
    {
      const_dt,
      cfl_number,
      only_print_cfl_number
    };

    //! reconstruction type of gradients (e.g. velocity gradient)
    enum GradientReconstructionMethod
    {
      gradreco_none,
      gradreco_spr,
      gradreco_l2
    };

    /// set the fluid parameters
    void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list);

    /// set fluid-specific conditions
    void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

  }  // namespace FLUID

  namespace LowMach
  {
    /// set the low mach number parameters
    void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list);
  }  // namespace LowMach

}  // namespace Inpar

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
