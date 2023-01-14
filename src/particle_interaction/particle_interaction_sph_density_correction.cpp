/*---------------------------------------------------------------------------*/
/*! \file
\brief density correction handler in smoothed particle hydrodynamics (SPH)
\level 3
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "particle_interaction_sph_density_correction.H"

#include "inpar_particle.H"

#include "lib_dserror.H"

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEINTERACTION::SPHDensityCorrectionBase::SPHDensityCorrectionBase()
{
  // empty constructor
}

void PARTICLEINTERACTION::SPHDensityCorrectionBase::Init()
{
  // nothing to do
}

void PARTICLEINTERACTION::SPHDensityCorrectionBase::Setup()
{
  // nothing to do
}

void PARTICLEINTERACTION::SPHDensityCorrectionBase::CorrectedDensityInterior(
    const double* denssum, double* dens) const
{
  dens[0] = denssum[0];
}

PARTICLEINTERACTION::SPHDensityCorrectionInterior::SPHDensityCorrectionInterior()
    : PARTICLEINTERACTION::SPHDensityCorrectionBase()
{
  // empty constructor
}

bool PARTICLEINTERACTION::SPHDensityCorrectionInterior::ComputeDensityBC() const { return false; }

void PARTICLEINTERACTION::SPHDensityCorrectionInterior::CorrectedDensityFreeSurface(
    const double* denssum, const double* colorfield, const double* dens_bc, double* dens) const
{
  // density of free surface particles is not corrected
}

PARTICLEINTERACTION::SPHDensityCorrectionNormalized::SPHDensityCorrectionNormalized()
    : PARTICLEINTERACTION::SPHDensityCorrectionBase()
{
  // empty constructor
}

bool PARTICLEINTERACTION::SPHDensityCorrectionNormalized::ComputeDensityBC() const { return false; }

void PARTICLEINTERACTION::SPHDensityCorrectionNormalized::CorrectedDensityFreeSurface(
    const double* denssum, const double* colorfield, const double* dens_bc, double* dens) const
{
  dens[0] = denssum[0] / colorfield[0];
}

PARTICLEINTERACTION::SPHDensityCorrectionRandles::SPHDensityCorrectionRandles()
    : PARTICLEINTERACTION::SPHDensityCorrectionBase()
{
  // empty constructor
}

bool PARTICLEINTERACTION::SPHDensityCorrectionRandles::ComputeDensityBC() const { return true; }

void PARTICLEINTERACTION::SPHDensityCorrectionRandles::CorrectedDensityFreeSurface(
    const double* denssum, const double* colorfield, const double* dens_bc, double* dens) const
{
  dens[0] = denssum[0] + dens_bc[0] * (1.0 - colorfield[0]);
}
