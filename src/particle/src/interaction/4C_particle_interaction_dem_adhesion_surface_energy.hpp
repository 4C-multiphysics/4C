// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_DEM_ADHESION_SURFACE_ENERGY_HPP
#define FOUR_C_PARTICLE_INTERACTION_DEM_ADHESION_SURFACE_ENERGY_HPP

#include "4C_config.hpp"

#include "4C_particle_input.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Particle
{

  /*!
   * Parameters necessary to calculate the surface energy for the adhesive forces in DEM.
   */
  struct DEMAdhesionSurfaceEnergyDistributionParams
  {
    SurfaceEnergyDistribution type;  //!< the type of distribution
    double standard_deviation;       //!< standard deviation of the distribution if available
    double cutoff_factor;  //!< multiplicative factor to limit the tails of the distribution, e.g.,
                           //!< max deviation = cutoff_factor * standard_deviation
  };

  /*!
   * Verifies that the parameters are valid and throws an exception otherwise.
   */
  void verify_params_adhesion_surface_energy_distribution(
      const DEMAdhesionSurfaceEnergyDistributionParams& params);

  /*!
   * Calculates the surface energy based on the given Particle::SurfaceEnergyDistribution type in
   * params and the given mean_surface_energy value.
   */
  double dem_adhesion_surface_energy(
      const DEMAdhesionSurfaceEnergyDistributionParams& params, double mean_surface_energy);

}  // namespace Particle

FOUR_C_NAMESPACE_CLOSE

#endif
