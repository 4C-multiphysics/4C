// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_particle_base.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | constructor                                                               |
 *---------------------------------------------------------------------------*/
Mat::PAR::ParticleMaterialBase::ParticleMaterialBase(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      initRadius_(matdata.parameters.get<double>("INITRADIUS")),
      initDensity_(matdata.parameters.get<double>("INITDENSITY"))
{
  // empty constructor
}

FOUR_C_NAMESPACE_CLOSE
