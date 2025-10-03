// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_heatloss_evaporation.hpp"

#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_material_handler.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::SPHHeatLossEvaporation::SPHHeatLossEvaporation(
    const Teuchos::ParameterList& params)
    : params_sph_(params),
      evaporatingphase_(PARTICLEENGINE::Phase1),
      recoilboilingtemp_(params_sph_.get<double>("VAPOR_RECOIL_BOILINGTEMPERATURE")),
      recoil_pfac_(params_sph_.get<double>("VAPOR_RECOIL_PFAC")),
      recoil_tfac_(params_sph_.get<double>("VAPOR_RECOIL_TFAC")),
      latentheat_(params_sph_.get<double>("VAPOR_HEATLOSS_LATENTHEAT")),
      enthalpyreftemp_(params_sph_.get<double>("VAPOR_HEATLOSS_ENTHALPY_REFTEMP")),
      heatloss_pfac_(params_sph_.get<double>("VAPOR_HEATLOSS_PFAC")),
      heatloss_tfac_(params_sph_.get<double>("VAPOR_HEATLOSS_TFAC"))
{
  // empty constructor
}

void ParticleInteraction::SPHHeatLossEvaporation::init()
{
  // safety check
  if (Teuchos::getIntegralValue<PARTICLE::SurfaceTensionFormulation>(
          params_sph_, "SURFACETENSIONFORMULATION") == PARTICLE::NoSurfaceTension)
    FOUR_C_THROW("surface tension evaluation needed for evaporation induced heat loss!");
}

void ParticleInteraction::SPHHeatLossEvaporation::setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set particle material handler
  particlematerial_ = particlematerial;

  // determine size of vectors indexed by particle types
  const int typevectorsize = *(--particlecontainerbundle_->get_particle_types().end()) + 1;

  // allocate memory to hold particle types
  thermomaterial_.resize(typevectorsize);

  // iterate over particle types
  for (const auto& type_i : particlecontainerbundle_->get_particle_types())
    thermomaterial_[type_i] = dynamic_cast<const Mat::PAR::ParticleMaterialThermo*>(
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i));
}

void ParticleInteraction::SPHHeatLossEvaporation::evaluate_evaporation_induced_heat_loss() const
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "ParticleInteraction::SPHHeatLossEvaporation::evaluate_evaporation_induced_heat_loss");

  // get container of owned particles of evaporating phase
  PARTICLEENGINE::ParticleContainer* container_i =
      particlecontainerbundle_->get_specific_container(evaporatingphase_, PARTICLEENGINE::Owned);

  const Mat::PAR::ParticleMaterialThermo* thermomaterial_i = thermomaterial_[evaporatingphase_];

  // iterate over particles in container
  for (int particle_i = 0; particle_i < container_i->particles_stored(); ++particle_i)
  {
    const double* dens_i = container_i->get_ptr_to_state(PARTICLEENGINE::Density, particle_i);
    const double* temp_i = container_i->get_ptr_to_state(PARTICLEENGINE::Temperature, particle_i);
    const double* cfg_i =
        container_i->get_ptr_to_state(PARTICLEENGINE::ColorfieldGradient, particle_i);
    const double* ifn_i =
        container_i->get_ptr_to_state(PARTICLEENGINE::InterfaceNormal, particle_i);
    double* tempdot_i = container_i->get_ptr_to_state(PARTICLEENGINE::TemperatureDot, particle_i);

    // evaluation only for non-zero interface normal
    if (not(Utils::vec_norm_two(ifn_i) > 0.0)) continue;

    // heat loss contribution only for temperature above boiling temperature
    if (not(temp_i[0] > recoilboilingtemp_)) continue;

    // compute evaporation induced recoil pressure
    const double recoil_press_i =
        recoil_pfac_ * std::exp(-recoil_tfac_ * (1.0 / temp_i[0] - 1.0 / recoilboilingtemp_));

    // compute vapor mass flow
    const double m_dot_i = heatloss_pfac_ * recoil_press_i * std::sqrt(heatloss_tfac_ / temp_i[0]);

    // evaluate specific enthalpy
    const double specificenthalpy_i =
        thermomaterial_i->thermalCapacity_ * (temp_i[0] - enthalpyreftemp_);

    // add contribution of heat loss
    tempdot_i[0] -= Utils::vec_norm_two(cfg_i) * m_dot_i * (latentheat_ + specificenthalpy_i) *
                    thermomaterial_i->invThermalCapacity_ / dens_i[0];
  }
}

FOUR_C_NAMESPACE_CLOSE
