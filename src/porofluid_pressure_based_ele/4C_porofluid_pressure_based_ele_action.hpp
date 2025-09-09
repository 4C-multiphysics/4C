// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELE_ACTION_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELE_ACTION_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

namespace PoroPressureBased
{
  /*--------------------------------------------------------------------------*/
  /*!
   * \brief enum that provides all possible porofluidmultiphase actions
   *///                                                            vuong 08/16
  /*--------------------------------------------------------------------------*/
  enum Action
  {
    // domain action
    calc_domain_integrals,        // calculate domain integrals
    calc_error,                   // calc_error
    calc_fluid_scatra_coupl_mat,  // calculate off-diagonal fluid-structure coupling matrix
    calc_fluid_struct_coupl_mat,  // calculate off-diagonal fluid-structure coupling matrix
    calc_initial_time_deriv,      // calculate initial time derivative
    calc_mat_and_rhs,             // calc system matrix and residual,
    calc_phase_velocities,        // calculate phase velocities
    calc_porosity,                // calculate porosity
    calc_pres_and_sat,            // calculate pressure and saturation
    calc_solidpressure,           // calculate solid pressure
    calc_determinant_of_deformationgradient,  // calculate determinant of derformation gradient
    calc_volfrac_blood_lung,      // calculate volume fraction of additional pore space with
    calc_valid_dofs,              // calculate the valid volume fraction pressure and species dofs
    get_access_from_artcoupling,  // get access from artery-coupling to evaluate variables
    get_access_from_scatra,       // get access from scatra-framework to evaluate variables
    recon_flux_at_nodes,          // reconstruct flux at nodes
  };  // enum Action

  /*--------------------------------------------------------------------------*/
  /*!
   * \brief enum that provides all possible porofluidmultiphase actions on a boundary
   *///                                                            vuong 08/16
  /*--------------------------------------------------------------------------*/
  enum BoundaryAction
  {
    bd_calc_Neumann,  // evaluate neumann loads
  };  // enum POROFLUIDMULTIPHASE::BoundaryAction
}  // namespace PoroPressureBased

FOUR_C_NAMESPACE_CLOSE

#endif
