// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_INPAR_MATERIAL_HPP
#define FOUR_C_INPAR_MATERIAL_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <map>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/



namespace Inpar
{
  namespace Mat
  {
    /*----------------------------------------------------------------------*
    | Robinson's visco-plastic material                        bborn 03/07 |
    | material parameters                                                  |
    | [1] Butler, Aboudi and Pindera: "Role of the material constitutive   |
    |     model in simulating the reusable launch vehicle thrust cell      |
    |     liner response", J Aerospace Engrg, 18(1), 2005.                 |
    | [2] Arya: "Analytical and finite element solutions of some problems  |
    |     using a vsicoplastic model", Comput & Struct, 33(4), 1989.       |
    | [3] Arya: "Viscoplastic analysis of an experimental cylindrical      |
    |     thrust chamber liner", AIAA J, 30(3), 1992.                      |
    *----------------------------------------------------------------------*/
    enum RobinsonType
    {
      vp_robinson_kind_vague = 0,       ///< unset
      vp_robinson_kind_arya,            ///< Arya, 1989 [2]
      vp_robinson_kind_arya_crmosteel,  ///< Arya, 1992 [3]
      vp_robinson_kind_arya_narloyz,    ///< Arya, 1992 [3]
      vp_robinson_kind_butler           ///< Butler et al, 2005 [1]
    };  // RobinsonType

    //! valid types for prescription of time-/space-dependent muscle activation
    enum ActivationType
    {
      tensor,                  ///< globally-defined tensor given via a vector
      function_of_space_time,  ///< analytical activation prescription via a symbolic function of
                               ///< space and time
      map  ///< discrete elementwise-defined activation prescription via an input pattern file
    };
  }  // namespace Mat
}  // namespace Inpar

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif