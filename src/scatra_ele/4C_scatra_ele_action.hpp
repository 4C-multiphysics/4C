// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_ACTION_HPP
#define FOUR_C_SCATRA_ELE_ACTION_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace ScaTra
{
  /*--------------------------------------------------------------------------*/
  /*!
   * \brief enum that provides all possible scatra actions
   *///                                                             nis Mar12
  /*--------------------------------------------------------------------------*/
  enum class Action
  {
    // domain action
    calc_dissipation,  ///< calculate dissipation introduced by stabilization and turbulence models
    calc_domain_and_bodyforce,              ///< calc_domain_and_bodyforce,
    calc_domain_integral,                   ///< n/a
    calc_elch_boundary_kinetics_point,      ///< evaluate electrode kinetics point conditions
    calc_elch_conductivity,                 ///< calc_elch_conductivity,
    calc_elch_domain_kinetics,              ///< calculate electrode kinetics in domain
    calc_elch_electrode_soc_and_c_rate,     ///< calculate electrode state of charge and C rate
    calc_elch_elctrode_mean_concentration,  ///< calculate mean concentration of micro
                                            ///< discretization
    calc_error,                             ///< calc_error,
    calc_error_reinit,                      ///< calc_error_reinit,
    calc_flux_domain,                       ///< calc_condif_flux,
    calc_heteroreac_mat_and_rhs,   ///< calculate RHS and linearizations of heterogeneous reaction
                                   ///< terms
    calc_immersed_element_source,  ///< calculate sink/source term due to segregation from an
                                   ///< immersed discretizaton
    calc_initial_time_deriv,       ///< calc_initial_time_deriv,
    calc_integr_reaction,          ///< calculate integral reaction
    calc_mass_center_smoothingfunct,  ///< calculate mass center of minus domain, using the
                                      ///< smoothing function. (Interface thickness required)
    calc_mass_matrix,                 ///< calculate mass matrix
    calc_mat,                         ///< calc_condif_systemmat
    calc_mat_and_rhs,                 ///< calc_condif_systemmat_and_residual,
    calc_mat_and_rhs_lsreinit_correction_step,  ///< level-set action: setup of system for
                                                ///< correction step
    calc_mat_initial,                           ///< calc initial mat for hdg
    calc_mean_Cai,                              ///< calc_mean_Cai (multifractal subgrid-scales)
    calc_mean_scalar_time_derivatives,  ///< calculate mean time derivatives of transported scalars
    calc_node_based_reinit_velocity,    ///< level-set action: compute node-based velocity field for
                                        ///< reinitialization equation
    calc_padaptivity,                   ///< calc error and set p-adativity on element
    calc_rhs,                           ///< calc_condif_residual
    calc_scatra_box_filter,             ///< calc_scatra_box_filter,

    calc_scatra_mono_odblock_fluid,  ///< calculate off diagonal matrix block for coupling with
                                     ///< fluid
    calc_scatra_mono_odblock_mesh,   ///< calculate off diagonal matrix block for coupling with mesh
                                     ///< (shape derivatives)
    calc_scatra_mono_odblock_scatrathermo,  ///< calculate off-diagonal matrix block (derivatives of
                                            ///< scatra residuals w.r.t. thermo dofs) for
                                            ///< scatra-thermo interaction
    calc_scatra_mono_odblock_thermoscatra,  ///< calculate off-diagonal matrix block (derivatives of
                                            ///< thermo residuals w.r.t. scatra dofs) for
                                            ///< scatra-thermo interaction
    calc_subgrid_diffusivity_matrix,        ///< calc_subgrid_diffusivity_matrix,
    calc_total_and_mean_scalars,     ///< calculate total and mean values of transported scalars
    calc_turbulent_prandtl_number,   ///< calc_turbulent_prandtl_number
    calc_vreman_scatra,              ///< calc_vreman_scatra,
    check_scatra_element_parameter,  ///< set parameter defined in every single element
    evaluate_field_in_point,         ///< evaluates field in given point (in reference coordinates)
    get_material_internal_state,  ///< get material internal state (for postprocessing for example),
    get_material_ionic_currents,  ///< get material ionic currents (for postprocessing for example),
    get_material_parameters,      ///< get_material_parameters,
    integrate_shape_functions,    ///< integrate_shape_functions,
    interpolate_hdg_to_node,      ///< interpolate hdg to node,
    micro_scale_initialize,       ///< initialize micro scale in multi-scale simulations
    micro_scale_output,           ///< output micro-scale quantities in multi-scale simulations
    collect_micro_scale_output,   ///< output micro-scale quantities in multi-scale simulations
    micro_scale_prepare_time_step,  ///< prepare time step on micro scale in multi-scale simulations
    micro_scale_read_restart,       ///< read restart on micro scale in multi-scale simulations
    micro_scale_solve,              ///< solve micro scale in multi-scale simulations
    micro_scale_update,             ///< update micro scale in multi-scale simulations
    micro_scale_set_time,           ///< set time stepping data in micro scale
    project_dirich_field,           ///< project dirichlet field for hdg
    project_field,                  ///< project field
    project_material_field,         ///< project material field for hdg
    project_neumann_field,          ///< project neumann field for hdg
    set_initial_field,              ///< set initial field for hdg
    set_material_internal_state,    ///< set material internal state (for restart for example),
    time_update_material,           ///< time update for materials
    transform_real_to_reference_point,  ///< gets a point in real coordinates and returns reference
                                        ///< coordinates
    update_interior_variables           ///< update interior variables for hdg
  };


  /*--------------------------------------------------------------------------*/
  /*!
   * \brief enum that provides all possible scatra actions on a boundary
   *///                                                             nis Mar12
  /*--------------------------------------------------------------------------*/
  enum class BoundaryAction
  {
    calc_Neumann,
    add_convective_mass_flux,
    calc_boundary_integral,
    calc_nodal_size,  ///< evaluate int N dv
    calc_convective_heat_transfer,
    calc_elch_boundary_kinetics,
    calc_elch_cell_voltage,
    calc_elch_linearize_nernst,
    calc_elch_minmax_overpotential,
    calc_fps3i_surface_permeability,  ///< Kedem-Katchalsky equations --> FPS3I problem
    calc_fs3i_surface_permeability,
    calc_loma_therm_press,
    calc_mass_matrix,
    calc_Neumann_inflow,
    calc_normal_vectors,
    calc_Robin,                       ///< Robin boundary condition
    calc_s2icoupling,                 ///< evaluate scatra-scatra interface coupling conditions
    calc_s2icoupling_flux,            ///< integral of all positive fluxes on s2i interface
    calc_s2icoupling_capacitance,     ///< evaluate capacitance part of scatra-scatra interface
                                      ///< coupling conditions
    calc_s2icoupling_capacitance_od,  ///< evaluate derivatives of scatra-scatra interface
                                      ///< coupling capacitance conditions w.r.t. secondary dofs
    calc_s2icoupling_growthgrowth,    ///< evaluate global growth-growth matrix block for
                                      ///< scatra-scatra interface coupling involving interface
                                      ///< layer growth
    calc_s2icoupling_growthscatra,    ///< evaluate global growth-scatra matrix block for
                                      ///< scatra-scatra interface coupling involving interface
                                      ///< layer growth
    calc_s2icoupling_od,              ///< evaluate derivatives of scatra-scatra interface coupling
                                      ///< conditions w.r.t. secondary dofs
    calc_s2icoupling_scatragrowth,    ///< evaluate global scatra-growth matrix block for
                                      ///< scatra-scatra interface coupling involving interface
                                      ///< layer growth
    calc_weak_Dirichlet,
    integrate_shape_functions
  };

  /*--------------------------------------------------------------------------*/
  /*!
   * \brief for coupled, monolithic problems: linearization w.r.t. other primary variable
   */
  /*--------------------------------------------------------------------------*/
  enum class DifferentiationType
  {
    none,
    disp,
    elch,
    temp
  };
}  // namespace ScaTra
FOUR_C_NAMESPACE_CLOSE

#endif
