// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_dem_adhesion_surface_energy.hpp"

#include "4C_global_data.hpp"
#include "4C_particle_input.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>

FOUR_C_NAMESPACE_OPEN

namespace
{

  double surface_energy_from_normal_distribution(
      const Particle::DEMAdhesionSurfaceEnergyDistributionParams& params,
      double mean_surface_energy)
  {
    Global::Problem::instance()->random()->set_mean_stddev(
        mean_surface_energy, params.standard_deviation);
    double surface_energy = Global::Problem::instance()->random()->normal();

    // Adjust surface energy to allowed bounds
    const double adhesion_surface_energy_min =
        std::min(0.0, surface_energy - params.cutoff_factor * params.standard_deviation);
    const double adhesion_surface_energy_max =
        surface_energy + params.cutoff_factor * params.standard_deviation;

    return std::clamp(surface_energy, adhesion_surface_energy_min, adhesion_surface_energy_max);
  }
}  // namespace

void Particle::verify_params_adhesion_surface_energy_distribution(
    const DEMAdhesionSurfaceEnergyDistributionParams& params)
{
  if (params.type == SurfaceEnergyDistribution::Normal)
  {
    if (params.standard_deviation < 0.0)
      FOUR_C_THROW("negative standard deviation for adhesion surface energy distribution!");
    if (params.cutoff_factor < 0.0)
      FOUR_C_THROW("negative cutoff factor of adhesion surface energy!");
  }
}

double Particle::dem_adhesion_surface_energy(
    const DEMAdhesionSurfaceEnergyDistributionParams& params, double mean_surface_energy)
{
  switch (params.type)
  {
    case SurfaceEnergyDistribution::Constant:
    {
      return mean_surface_energy;
    }
    case SurfaceEnergyDistribution::Normal:
    {
      return surface_energy_from_normal_distribution(params, mean_surface_energy);
    }
    default:
    {
      FOUR_C_THROW("unknown adhesion surface energy distribution type!");
      break;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
