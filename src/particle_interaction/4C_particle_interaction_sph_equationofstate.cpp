/*---------------------------------------------------------------------------*/
/*! \file
\brief equation of state handler for smoothed particle hydrodynamics (SPH) interactions
\level 3
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_particle_interaction_sph_equationofstate.hpp"

#include "4C_particle_interaction_utils.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::SPHEquationOfStateBase::SPHEquationOfStateBase()
{
  // empty constructor
}

void ParticleInteraction::SPHEquationOfStateBase::init()
{
  // nothing to do
}

void ParticleInteraction::SPHEquationOfStateBase::setup()
{
  // nothing to do
}

ParticleInteraction::SPHEquationOfStateGenTait::SPHEquationOfStateGenTait(
    const double& speedofsound, const double& refdensfac, const double& exponent)
    : ParticleInteraction::SPHEquationOfStateBase(),
      speedofsound_(speedofsound),
      refdensfac_(refdensfac),
      exponent_(exponent)
{
  // empty constructor
}

double ParticleInteraction::SPHEquationOfStateGenTait::density_to_pressure(
    const double& density, const double& density0) const
{
  if (exponent_ == 1)
    return Utils::pow<2>(speedofsound_) * (density - refdensfac_ * density0);
  else
  {
    double initPressure = Utils::pow<2>(speedofsound_) * density0 / exponent_;
    return initPressure * (std::pow((density / density0), exponent_) - refdensfac_);
  }
}

double ParticleInteraction::SPHEquationOfStateGenTait::pressure_to_density(
    const double& pressure, const double& density0) const
{
  if (exponent_ == 1)
    return pressure / Utils::pow<2>(speedofsound_) + refdensfac_ * density0;
  else
  {
    double initPressure = Utils::pow<2>(speedofsound_) * density0 / exponent_;
    return density0 * std::pow(((pressure / initPressure) + refdensfac_), (1.0 / exponent_));
  }
}

double ParticleInteraction::SPHEquationOfStateGenTait::density_to_energy(
    const double& density, const double& mass, const double& density0) const
{
  // thermodynamic energy E with p=-dE/dV, T=dE/dS (see Espanol2003, Eq.(5))
  // Attention: currently only the pressure-dependent contribution of the thermodynamic energy is
  // implemented! Thus, it is only valid for isentrop problems, i.e. dE/dS=0! Remark: integration of
  // the pressure law with the relation V=mass/density and integration constant from initial
  // condition E(V=mass/initDensity)
  if (exponent_ == 1)
    return -Utils::pow<2>(speedofsound_) * mass *
           (std::log(Utils::pow<2>(mass) / (density0 * density)) -
               refdensfac_ * (1 + (density0 / density)));
  else
  {
    double initPressure = Utils::pow<2>(speedofsound_) * density0 / exponent_;
    return -initPressure *
           ((1.0 / (1 - exponent_)) *
                   (mass / (std::pow(density0, exponent_) * std::pow(density, (1 - exponent_))) +
                       mass / density0) -
               refdensfac_ * (mass / density0 + mass / density));
  }
}

ParticleInteraction::SPHEquationOfStateIdealGas::SPHEquationOfStateIdealGas(
    const double& speedofsound)
    : ParticleInteraction::SPHEquationOfStateBase(), speedofsound_(speedofsound)
{
  // empty constructor
}

double ParticleInteraction::SPHEquationOfStateIdealGas::density_to_pressure(
    const double& density, const double& density0) const
{
  return Utils::pow<2>(speedofsound_) * density;
}

double ParticleInteraction::SPHEquationOfStateIdealGas::pressure_to_density(
    const double& pressure, const double& density0) const
{
  return pressure / Utils::pow<2>(speedofsound_);
}

double ParticleInteraction::SPHEquationOfStateIdealGas::density_to_energy(
    const double& density, const double& mass, const double& density0) const
{
  // thermodynamic energy E with p=-dE/dV, T=dE/dS (see Espanol2003, Eq.(5))
  // Attention: currently only the pressure-dependent contribution of the thermodynamic energy is
  // implemented! Thus, it is only valid for isentrop problems, i.e. dE/dS=0! Remark: integration of
  // the pressure law with the relation V=mass/density and integration constant from initial
  // condition E(V=mass/initDensity)
  return -Utils::pow<2>(speedofsound_) * mass *
         std::log(Utils::pow<2>(mass) / (density0 * density));
}

FOUR_C_NAMESPACE_CLOSE
