// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_ELE_ACTION_HPP
#define FOUR_C_FLUID_ELE_ACTION_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace FLD
{
  /*--------------------------------------------------------------------------
   | enum that provides all possible fluid actions
   *--------------------------------------------------------------------------*/
  enum Action
  {
    calc_dissipation,
    calc_div_u,
    calc_divop,
    calc_dt_via_cfl,
    calc_fluid_box_filter,
    calc_fluid_error,
    calc_fluid_genalpha_sysmat_and_residual,
    calc_fluid_genalpha_update_for_subscales,
    calc_fluid_systemmat_and_residual,
    calc_loma_mono_odblock,
    calc_loma_statistics,
    calc_mass_flow_periodic_hill,
    calc_mass_matrix,
    calc_mean_Cai,
    calc_model_params_mfsubgr_scales,
    calc_node_normal,
    calc_poroscatra_mono_odblock,
    calc_porousflow_fluid_coupling,
    calc_pressure_average,
    calc_smagorinsky_const,
    calc_turbscatra_statistics,
    calc_turbulence_statistics,
    calc_velgrad_ele_center,
    calc_volume,
    calc_vreman_const,
    integrate_shape,
    interpolate_hdg_for_hit,
    interpolate_hdg_to_node,
    none,
    presgradient_projection,
    project_fluid_field,
    project_hdg_force_on_dof_vec_for_hit,
    project_hdg_initial_field_for_hit,
    tauw_via_gradient,
    update_local_solution,
    velgradient_projection,
    xwall_calc_mk,
    xwall_l2_projection,
  };  // enum Action

  /*--------------------------------------------------------------------------
   | enum that provides all possible fluid actions on a boundary
   *--------------------------------------------------------------------------*/
  enum BoundaryAction
  {
    Outletimpedance,
    boundary_calc_node_normal,
    boundary_none,
    calc_Neumann_inflow,
    calc_area,
    calc_flowrate,
    calc_node_curvature,
    calc_pressure_bou_int,
    calc_surface_tension,
    center_of_mass_calc,
    dQdu,
    enforce_weak_dbc,
    estimate_Nitsche_trace_maxeigenvalue_,
    flow_dep_pressure_bc,
    flowratederiv,
    fpsi_coupling,
    integrate_Shapefunction,
    mixed_hybrid_dbc,
    navier_slip_bc,
    no_penetration,
    no_penetrationIDs,
    poro_boundary,
    poro_prescoupl,
    poro_splitnopenetration,
    poro_splitnopenetration_OD,
    poro_splitnopenetration_ODdisp,
    poro_splitnopenetration_ODpres,
    slip_supp_bc,
    traction_Uv_integral_component,
    traction_velocity_component,
  };  // enum BoundaryAction

  /*--------------------------------------------------------------------------
   | enum that provides all possible fluid actions on a element interfaces
   *--------------------------------------------------------------------------*/
  enum IntFaceAction
  {
    ifa_none,
    EOS_and_GhostPenalty_stabilization
  };  // enum IntFaceAction

}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
