// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_INPAR_SCATRA_HPP
#define FOUR_C_INPAR_SCATRA_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"
#include "4C_io_input_spec_builders.hpp"

#include <map>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Conditions
{
  class ConditionDefinition;
}

/*----------------------------------------------------------------------*/
namespace Inpar
{
  namespace ScaTra
  {
    /// time integration schemes for scalar transport problems
    enum TimeIntegrationScheme
    {
      timeint_stationary,
      timeint_one_step_theta,
      timeint_bdf2,
      timeint_gen_alpha
    };

    /// type of solution procedures for scalar transport problems
    enum SolverType
    {
      solvertype_linear_full,
      solvertype_linear_incremental,
      solvertype_nonlinear,
      solvertype_nonlinear_multiscale_macrotomicro,
      solvertype_nonlinear_multiscale_macrotomicro_aitken,
      solvertype_nonlinear_multiscale_macrotomicro_aitken_dofsplit,
      solvertype_nonlinear_multiscale_microtomacro
    };

    /// type of convective velocity field
    enum VelocityField
    {
      velocity_zero,
      velocity_function,
      velocity_Navier_Stokes
    };

    /// initial field for scalar transport problem
    enum InitialField
    {
      initfield_zero_field,
      initfield_field_by_function,
      initfield_field_by_condition,
      initfield_disturbed_field_by_function,
      initfield_discontprogvar_1D,
      initfield_flame_vortex_interaction,
      initfield_raytaymixfrac,
      initfield_Lshapeddomain,
      initfield_facing_flame_fronts,
      initfield_oracles_flame,
      initialfield_forced_hit_high_Sc,
      initialfield_forced_hit_low_Sc,
      initialfield_algebraic_field_dependence
    };

    /// form of convective term
    enum ConvForm
    {
      convform_convective,
      convform_conservative
    };

    /// compute error compared to analytical solution
    enum CalcError
    {
      calcerror_no,
      calcerror_Kwok_Wu,
      calcerror_cylinder,
      calcerror_electroneutrality,
      calcerror_byfunction,
      calcerror_bycondition,
      calcerror_spherediffusion,
      calcerror_AnalyticSeries
    };

    /// possible types of stabilization
    enum StabType
    {
      stabtype_no_stabilization,
      stabtype_SUPG,
      stabtype_GLS,
      stabtype_USFEM,
      stabtype_hdg_centered,
      stabtype_hdg_upwind
    };

    /// possible options for the stabilization parameter
    enum TauType
    {
      tau_taylor_hughes_zarins,
      tau_taylor_hughes_zarins_wo_dt,
      tau_franca_valentin,
      tau_franca_valentin_wo_dt,
      tau_shakib_hughes_codina,
      tau_shakib_hughes_codina_wo_dt,
      tau_codina,
      tau_codina_wo_dt,
      tau_franca_madureira_valentin,
      tau_franca_madureira_valentin_wo_dt,
      tau_exact_1d,
      tau_zero,
      tau_numerical_value
    };

    /// possible options for characteristic element length
    enum CharEleLength
    {
      streamlength,
      volume_equivalent_diameter,
      root_of_volume
    };

    /// possible options for all-scale subgrid diffusivity
    enum AssgdType
    {
      assgd_artificial,
      assgd_hughes,
      assgd_tezduyar,
      assgd_tezduyar_wo_phizero,
      assgd_docarmo,
      assgd_almeida,
      assgd_lin_reinit,
      assgd_yzbeta,
      assgd_codina,
    };

    /// possible options for fine-scale subgrid diffusivity
    enum FSSUGRDIFF
    {
      fssugrdiff_no,
      fssugrdiff_artificial,
      fssugrdiff_smagorinsky_all,
      fssugrdiff_smagorinsky_small
    };

    /// parameters for flux calculation of scalar transport problems
    enum FluxType
    {
      flux_none,
      flux_convective,
      flux_diffusive,
      flux_total
    };

    // this parameter selects the location where tau is evaluated
    enum EvalTau
    {
      evaltau_element_center,
      evaltau_integration_point
    };

    // this parameter selects the location where the material is evaluated
    enum EvalMat
    {
      evalmat_element_center,
      evalmat_integration_point
    };

    //! element implementation type
    enum ImplType
    {
      impltype_undefined,
      impltype_std,
      impltype_loma,
      impltype_elch_NP,
      impltype_elch_electrode,
      impltype_elch_electrode_growth,
      impltype_elch_electrode_thermo,
      impltype_elch_diffcond,
      impltype_elch_diffcond_multiscale,
      impltype_elch_diffcond_thermo,
      impltype_elch_scl,
      impltype_thermo_elch_electrode,
      impltype_thermo_elch_diffcond,
      impltype_lsreinit,
      impltype_levelset,
      impltype_poro,
      impltype_advreac,
      impltype_refconcreac,
      impltype_multipororeac,
      impltype_pororeac,
      impltype_pororeacECM,
      impltype_aniso,
      impltype_cardiac_monodomain,
      impltype_chemo,
      impltype_chemoreac,
      impltype_std_hdg,
      impltype_cardiac_monodomain_hdg,
      impltype_one_d_artery,
      impltype_no_physics
    };

    /// type of method for improving consistency of stabilized methods
    enum Consistency
    {
      consistency_no,
      consistency_l2_projection_lumped
    };

    /// type of finite difference check
    enum FdCheck
    {
      fdcheck_none,
      fdcheck_global,
      fdcheck_global_extended,
      fdcheck_local
    };

    /// possible options for output of means of scalars
    enum OutputScalarType
    {
      outputscalars_none,
      outputscalars_entiredomain,
      outputscalars_condition,
      outputscalars_entiredomain_condition
    };

    /// possible options for computation of domain and boundary integrals, i.e., surface areas and
    /// volumes
    enum ComputeIntegrals
    {
      computeintegrals_none,
      computeintegrals_initial,
      computeintegrals_repeated
    };

    /// Type of coupling strategy between the two fields of the SSI problems
    enum FieldCoupling
    {
      coupling_match,
      coupling_volmortar
    };

    /// set the scatra parameters
    void set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list);

    /// set additional scatra conditions
    void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

    std::string impltype_to_string(ImplType impltype);
  }  // namespace ScaTra
}  // namespace Inpar

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
