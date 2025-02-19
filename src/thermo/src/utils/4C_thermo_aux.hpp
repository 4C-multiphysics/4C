// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_THERMO_AUX_HPP
#define FOUR_C_THERMO_AUX_HPP


/*----------------------------------------------------------------------*
 | headers                                                  bborn 08/09 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_thermo_input.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | belongs to thermal dynamics namespace                    bborn 08/09 |
 *----------------------------------------------------------------------*/
namespace Thermo
{
  /*====================================================================*/
  namespace Aux
  {
    //! Determine norm of force residual
    double calculate_vector_norm(const enum Thermo::VectorNorm norm,  //!< norm to use
        Core::LinAlg::Vector<double>& vect                            //!< the vector of interest
    );

  }  // namespace Aux

}  // namespace Thermo

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
