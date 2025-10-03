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

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::DEMAdhesionSurfaceEnergyBase::DEMAdhesionSurfaceEnergyBase(
    const Teuchos::ParameterList& params)
    : params_dem_(params)
{
  // empty constructor
}

void ParticleInteraction::DEMAdhesionSurfaceEnergyBase::init()
{
  // nothing to do
}

void ParticleInteraction::DEMAdhesionSurfaceEnergyBase::setup()
{
  // safety check
  if (not(params_dem_.get<double>("ADHESION_SURFACE_ENERGY") > 0.0))
    FOUR_C_THROW("non-positive adhesion surface energy!");
}

ParticleInteraction::DEMAdhesionSurfaceEnergyConstant::DEMAdhesionSurfaceEnergyConstant(
    const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMAdhesionSurfaceEnergyBase(params)
{
  // empty constructor
}

ParticleInteraction::DEMAdhesionSurfaceEnergyDistributionBase::
    DEMAdhesionSurfaceEnergyDistributionBase(const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMAdhesionSurfaceEnergyBase(params),
      variance_(params_dem_.get<double>("ADHESION_SURFACE_ENERGY_DISTRIBUTION_VAR")),
      cutofffactor_(params_dem_.get<double>("ADHESION_SURFACE_ENERGY_DISTRIBUTION_CUTOFF_FACTOR"))
{
  // empty constructor
}

void ParticleInteraction::DEMAdhesionSurfaceEnergyDistributionBase::setup()
{
  // call base class setup
  DEMAdhesionSurfaceEnergyBase::setup();

  // safety checks
  if (variance_ < 0.0) FOUR_C_THROW("negative variance for adhesion surface energy distribution!");
  if (cutofffactor_ < 0.0) FOUR_C_THROW("negative cutoff factor of adhesion surface energy!");
}

void ParticleInteraction::DEMAdhesionSurfaceEnergyDistributionBase::
    adjust_surface_energy_to_allowed_bounds(
        const double& mean_surface_energy, double& surface_energy) const
{
  const double adhesion_surface_energy_min =
      std::min(0.0, mean_surface_energy - cutofffactor_ * variance_);
  const double adhesion_surface_energy_max = mean_surface_energy + cutofffactor_ * variance_;

  if (surface_energy > adhesion_surface_energy_max)
    surface_energy = adhesion_surface_energy_max;
  else if (surface_energy < adhesion_surface_energy_min)
    surface_energy = adhesion_surface_energy_min;
}

ParticleInteraction::DEMAdhesionSurfaceEnergyDistributionNormal::
    DEMAdhesionSurfaceEnergyDistributionNormal(const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMAdhesionSurfaceEnergyDistributionBase(params)
{
  // empty constructor
}

void ParticleInteraction::DEMAdhesionSurfaceEnergyDistributionNormal::adhesion_surface_energy(
    const double& mean_surface_energy, double& surface_energy) const
{
  // initialize random number generator
  Global::Problem::instance()->random()->set_mean_stddev(mean_surface_energy, variance_);

  // set normal distributed random value for surface energy
  surface_energy = Global::Problem::instance()->random()->normal();

  // adjust surface energy to allowed bounds
  adjust_surface_energy_to_allowed_bounds(mean_surface_energy, surface_energy);
}

ParticleInteraction::DEMAdhesionSurfaceEnergyDistributionLogNormal::
    DEMAdhesionSurfaceEnergyDistributionLogNormal(const Teuchos::ParameterList& params)
    : ParticleInteraction::DEMAdhesionSurfaceEnergyDistributionBase(params)
{
  // empty constructor
}

void ParticleInteraction::DEMAdhesionSurfaceEnergyDistributionLogNormal::adhesion_surface_energy(
    const double& mean_surface_energy, double& surface_energy) const
{
  // initialize random number generator
  Global::Problem::instance()->random()->set_mean_stddev(std::log(mean_surface_energy), variance_);

  // set log-normal distributed random value for surface energy
  surface_energy = std::exp(Global::Problem::instance()->random()->normal());

  // adjust surface energy to allowed bounds
  adjust_surface_energy_to_allowed_bounds(mean_surface_energy, surface_energy);
}

FOUR_C_NAMESPACE_CLOSE
