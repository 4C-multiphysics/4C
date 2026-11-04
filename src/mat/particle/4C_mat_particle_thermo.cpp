// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_particle_thermo.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | constructor                                                               |
 *---------------------------------------------------------------------------*/
Mat::PAR::ParticleMaterialThermo::ParticleMaterialThermo(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      initTemperature_(matdata.parameters.get<double>("INITTEMPERATURE")),
      thermalCapacity_(matdata.parameters.get<double>("THERMALCAPACITY")),
      invThermalCapacity_((thermalCapacity_ > 0.0) ? (1.0 / thermalCapacity_) : 0.0),
      thermalConductivity_(matdata.parameters.get<double>("THERMALCONDUCTIVITY")),
      thermalAbsorptivity_(matdata.parameters.get<double>("THERMALABSORPTIVITY"))
{
  // empty constructor
}

FOUR_C_NAMESPACE_CLOSE
