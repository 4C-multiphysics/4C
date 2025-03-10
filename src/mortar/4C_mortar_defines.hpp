// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MORTAR_DEFINES_HPP
#define FOUR_C_MORTAR_DEFINES_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

/************************************************************************/
/* Mortar algorithm parameters                                          */
/************************************************************************/

// MORTAR INTEGRATION
#define MORTARINTTOL 0.0 /* tolerance for assembling gp-values*/

// MORTAR PROJECTION (2D/3D)
#define MORTARMAXITER 10      /* max. no. iterations for local Newton */
#define MORTARCONVTOL 1.0e-12 /* convergence tolerance for local Newton */

// MORTAR PROJECTION AND INTEGRATION (2D)
#define MORTARPROJTOL 0.05   /* projection tolerance for overlap */
#define MORTARPROJLIM 1.0e-8 /* exact projection limit (no tolerance!) */

// MORTAR PROJECTION AND INTEGRATION (3D)
#define MORTARCLIPTOL 1.0e-8 /* tolerance for polygon clipping */
#define MORTARINTLIM 1.0e-12 /* min(area-%) cell/slave for integration */

FOUR_C_NAMESPACE_CLOSE

#endif
