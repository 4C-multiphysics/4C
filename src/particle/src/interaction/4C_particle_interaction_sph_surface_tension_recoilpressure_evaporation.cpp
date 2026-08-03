// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_surface_tension_recoilpressure_evaporation.hpp"

#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_utils.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::SPHRecoilPressureEvaporation::SPHRecoilPressureEvaporation(
    const Teuchos::ParameterList& params)
    : params_sph_(params),
      evaporatingphase_(Particle::Phase1),
      recoilboilingtemp_(params_sph_.get<double>("VAPOR_RECOIL_BOILINGTEMPERATURE")),
      recoil_pfac_(params_sph_.get<double>("VAPOR_RECOIL_PFAC")),
      recoil_tfac_(params_sph_.get<double>("VAPOR_RECOIL_TFAC"))
{
  // empty constructor
}

void Particle::SPHRecoilPressureEvaporation::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();
}

void Particle::SPHRecoilPressureEvaporation::compute_recoil_pressure_contribution() const
{
  // get container of owned particles of evaporating phase
  Particle::ParticleContainer* container_i =
      particlecontainerbundle_->get_specific_container(evaporatingphase_, Particle::Owned);

  // iterate over particles in container
  for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
  {
    const double* dens_i = container_i->get_ptr_to_state(Particle::Density, particle_i);
    const double* temp_i = container_i->get_ptr_to_state(Particle::Temperature, particle_i);
    const double* cfg_i = container_i->get_ptr_to_state(Particle::ColorfieldGradient, particle_i);
    const double* ifn_i = container_i->get_ptr_to_state(Particle::InterfaceNormal, particle_i);
    double* acc_i = container_i->get_ptr_to_state_writable(Particle::Acceleration, particle_i);

    // evaluation only for non-zero interface normal
    if (not(ParticleUtils::vec_norm_two(ifn_i) > 0.0)) continue;

    // recoil pressure contribution only for temperature above boiling temperature
    if (not(temp_i[0] > recoilboilingtemp_)) continue;

    // compute evaporation induced recoil pressure
    const double recoil_press_i =
        recoil_pfac_ * std::exp(-recoil_tfac_ * (1.0 / temp_i[0] - 1.0 / recoilboilingtemp_));

    // add contribution to acceleration
    ParticleUtils::vec_add_scale(acc_i, -recoil_press_i / dens_i[0], cfg_i);
  }
}

FOUR_C_NAMESPACE_CLOSE
