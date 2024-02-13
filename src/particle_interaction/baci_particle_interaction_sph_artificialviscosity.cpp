/*---------------------------------------------------------------------------*/
/*! \file
\brief artificial viscosity handler for smoothed particle hydrodynamics (SPH) interactions
\level 3
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "baci_particle_interaction_sph_artificialviscosity.hpp"

#include "baci_particle_interaction_utils.hpp"
#include "baci_utils_exceptions.hpp"

#include <cmath>

BACI_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEINTERACTION::SPHArtificialViscosity::SPHArtificialViscosity()
{
  // empty constructor
}

void PARTICLEINTERACTION::SPHArtificialViscosity::Init()
{
  // nothing to do
}

void PARTICLEINTERACTION::SPHArtificialViscosity::Setup()
{
  // nothing to do
}

void PARTICLEINTERACTION::SPHArtificialViscosity::ArtificialViscosity(const double* vel_i,
    const double* vel_j, const double* mass_i, const double* mass_j, const double& artvisc_i,
    const double& artvisc_j, const double& dWdrij, const double& dWdrji, const double& dens_ij,
    const double& h_ij, const double& c_ij, const double& abs_rij, const double* e_ij,
    double* acc_i, double* acc_j) const
{
  double vel_ij[3];
  UTILS::VecSet(vel_ij, vel_i);
  UTILS::VecSub(vel_ij, vel_j);

  // avoid division by zero for close particles
  const double epsilon = 0.01;

  const double fac = h_ij * c_ij * UTILS::VecDot(vel_ij, e_ij) * abs_rij /
                     (dens_ij * (UTILS::Pow<2>(abs_rij) + epsilon * UTILS::Pow<2>(h_ij)));

  if (acc_i) UTILS::VecAddScale(acc_i, (artvisc_i * mass_j[0] * dWdrij * fac), e_ij);
  if (acc_j) UTILS::VecAddScale(acc_j, (-artvisc_j * mass_i[0] * dWdrji * fac), e_ij);
}

BACI_NAMESPACE_CLOSE
