// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LEGACY_ENUM_DEFINITIONS_ELEMENT_ACTIONS_HPP
#define FOUR_C_LEGACY_ENUM_DEFINITIONS_ELEMENT_ACTIONS_HPP

#include "4C_config.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

namespace Core::Elements
{
  enum ActionType
  {
    none,
    struct_calc_linstiff,
    struct_calc_nlnstiff,       //!< evaluate the tangential stiffness matrix and the internal force
                                //!< vector
    struct_calc_internalforce,  //!< evaluate only the internal forces (no need for the stiffness
                                //!< terms)
    struct_calc_internalinertiaforce,  //!< evaluate only the internal and inertia forces
    struct_calc_linstiffmass,
    struct_calc_nlnstiffmass,   //!< evaluate the dynamic state: internal forces vector, stiffness
                                //!< and the default/nln mass matrix
    struct_calc_nlnstifflmass,  //!< evaluate the dynamic state: internal forces vector, stiffness
                                //!< and the lumped mass matrix
    struct_calc_nlnstiff_gemm,  //!< internal force, stiffness and mass for GEMM
    struct_calc_recover,        //!< recover elementwise condensed internal variables
    struct_calc_predict,        //!< predict elementwise condensed internal variables
    struct_calc_stress,
    struct_calc_thickness,
    struct_calc_eleload,
    struct_calc_fsiload,
    struct_calc_update_istep,
    struct_calc_reset_istep,  //!< reset elementwise internal variables, during iteration to last
                              //!< converged state
    struct_calc_energy,       //!< compute internal energy
    struct_postprocess_thickness,         //!< postprocess thickness of membrane finite elements
    struct_init_gauss_point_data_output,  //!< initialize quantities for output of gauss point
                                          //!< data
    struct_gauss_point_data_output,       //!< collect material data for vtk runtime output
    struct_update_prestress,

    struct_calc_ptcstiff,   //!< calculate artificial stiffness due to PTC solution strategy
    struct_calc_stifftemp,  //!< TSI specific: mechanical-thermal stiffness
    struct_calc_global_gpstresses_map,  //!< basically calc_struct_stress but with assembly of
                                        //!< global gpstresses map
    struct_calc_brownianforce,        //!< thermal (i.e. stochastic) and damping forces according to
                                      //!< Brownian dynamics
    struct_calc_brownianstiff,        //!< thermal (i.e. stochastic) and damping forces and stiffnes
                                      //!< according to Brownian dynamics
    struct_poro_calc_fluidcoupling,   //!< calculate stiffness matrix related to fluid coupling
                                      //!< within porous medium problem
    struct_poro_calc_scatracoupling,  //!< calculate stiffness matrix related to scatra coupling
                                      //!< within porous medium problem
    struct_poro_calc_prescoupling,    //!< calculate stiffness matrix related to pressure coupling
                                      //!< within porous medium problem
    struct_calc_addjacPTC,            //!< calculate element based PTC contributions
    struct_create_backup,        //!< create a backup state of the internally store state quantities
                                 //!< (e.g. EAS, material history, etc.)
    struct_recover_from_backup,  //!< recover from previously stored backup state
    calc_struct_stiffscalar,     //!< calculate coupling term k_dS for monolithic SSI
    struct_calc_analytical_error  //!< compute L2 error in comparison to analytical solution
  };

  static inline enum ActionType string_to_action_type(const std::string& action)
  {
    if (action == "none")
      return none;
    else if (action == "calc_struct_linstiff")
      return struct_calc_linstiff;
    else if (action == "calc_struct_nlnstiff")
      return struct_calc_nlnstiff;
    else if (action == "calc_struct_internalforce")
      return struct_calc_internalforce;
    else if (action == "calc_struct_linstiffmass")
      return struct_calc_linstiffmass;
    else if (action == "calc_struct_nlnstiffmass")
      return struct_calc_nlnstiffmass;
    else if (action == "calc_struct_nlnstifflmass")
      return struct_calc_nlnstifflmass;
    else if (action == "struct_calc_analytical_error")
      return struct_calc_analytical_error;
    else if (action == "calc_struct_stress")
      return struct_calc_stress;
    else if (action == "calc_struct_eleload")
      return struct_calc_eleload;
    else if (action == "calc_struct_fsiload")
      return struct_calc_fsiload;
    else if (action == "calc_struct_update_istep")
      return struct_calc_update_istep;
    else if (action == "calc_struct_reset_istep")
      return struct_calc_reset_istep;
    else if (action == "calc_struct_energy")
      return struct_calc_energy;
    else if (action == "struct_init_gauss_point_data_output")
      return struct_init_gauss_point_data_output;
    else if (action == "struct_gauss_point_data_output")
      return struct_gauss_point_data_output;
    else if (action == "calc_struct_prestress_update")
      return struct_update_prestress;
    else if (action == "calc_global_gpstresses_map")
      return struct_calc_global_gpstresses_map;
    else if (action == "calc_struct_predict")
      return struct_calc_predict;
    else if (action == "struct_poro_calc_fluidcoupling")
      return struct_poro_calc_fluidcoupling;
    else if (action == "struct_poro_calc_scatracoupling")
      return struct_poro_calc_scatracoupling;
    else if (action == "calc_struct_stiffscalar")
      return calc_struct_stiffscalar;
    else
      return none;
  }

  //! Map action type enum to std::string
  static inline std::string action_type_to_string(const enum ActionType& type)
  {
    switch (type)
    {
      case none:
        return "none";
      case struct_calc_linstiff:
        return "struct_calc_linstiff";
      case struct_calc_nlnstiff:
        return "struct_calc_nlnstiff";
      case struct_calc_internalforce:
        return "struct_calc_internalforce";
      case struct_calc_internalinertiaforce:
        return "struct_calc_internalinertiaforce";
      case struct_calc_linstiffmass:
        return "struct_calc_linstiffmass";
      case struct_calc_nlnstiffmass:
        return "struct_calc_nlnstiffmass";
      case struct_calc_nlnstifflmass:
        return "struct_calc_nlnstifflmass";
      case struct_calc_analytical_error:
        return "struct_calc_analytical_error";
      case struct_calc_predict:
        return "struct_calc_predict";
      case struct_calc_recover:
        return "struct_calc_recover";
      case struct_calc_stress:
        return "struct_calc_stress";
      case struct_calc_thickness:
        return "struct_calc_thickness";
      case struct_calc_eleload:
        return "struct_calc_eleload";
      case struct_calc_fsiload:
        return "struct_calc_fsiload";
      case struct_calc_update_istep:
        return "struct_calc_update_istep";
      case struct_calc_reset_istep:
        return "struct_calc_reset_istep";
      case struct_calc_energy:
        return "struct_calc_energy";
      case struct_postprocess_thickness:
        return "struct_postprocess_thickness";
      case struct_init_gauss_point_data_output:
        return "struct_init_gauss_point_data_output";
      case struct_gauss_point_data_output:
        return "struct_gauss_point_data_output";
      case struct_update_prestress:
        return "struct_update_prestress";
      case struct_calc_ptcstiff:
        return "struct_calc_ptcstiff";
      case struct_calc_stifftemp:
        return "struct_calc_stifftemp";
      case struct_calc_global_gpstresses_map:
        return "struct_calc_global_gpstresses_map";
      case struct_calc_brownianforce:
        return "struct_calc_brownianforce";
      case struct_calc_brownianstiff:
        return "struct_calc_brownianstiff";
      case struct_create_backup:
        return "struct_create_backup";
      case struct_recover_from_backup:
        return "struct_recover_from_backup";
      case struct_poro_calc_fluidcoupling:
        return "struct_poro_calc_fluidcoupling";
      case struct_poro_calc_scatracoupling:
        return "struct_poro_calc_scatracoupling";
      case calc_struct_stiffscalar:
        return "calc_struct_stiffscalar";
      default:
        return "unknown";
    }
  };

}  // namespace Core::Elements

FOUR_C_NAMESPACE_CLOSE

#endif
