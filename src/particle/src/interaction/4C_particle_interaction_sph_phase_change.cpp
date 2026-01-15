// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_phase_change.hpp"

#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_engine_object.hpp"
#include "4C_particle_interaction_material_handler.hpp"
#include "4C_particle_interaction_sph_equationofstate.hpp"
#include "4C_particle_interaction_sph_equationofstate_bundle.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::SPHPhaseChangeBase::SPHPhaseChangeBase(const Teuchos::ParameterList& params)
    : params_sph_(params),
      belowphase_(Particle::Phase1),
      abovephase_(Particle::Phase2),
      transitionstate_(Particle::Density),
      transitionvalue_(0.0),
      hysteresisgap_(0.0)
{
  initialize_parameters();
}

void Particle::SPHPhaseChangeBase::initialize_parameters()
{
  // read from input file
  std::string word;
  std::istringstream phasechangedefinition(
      Teuchos::getNumericStringParameter(params_sph_, "PHASECHANGEDEFINITION"));

  // get phase below transition value
  if (phasechangedefinition >> word)
    belowphase_ = Particle::enum_from_type_name(word);
  else
    FOUR_C_THROW("expecting particle type for phase below transition value!");

  // get phase above transition value
  if (phasechangedefinition >> word)
    abovephase_ = Particle::enum_from_type_name(word);
  else
    FOUR_C_THROW("expecting particle type for phase above transition value!");

  // safety check
  if (belowphase_ == abovephase_)
    FOUR_C_THROW("equal particle types for phase below and above transition value!");

  // get transition state of phase change
  if (phasechangedefinition >> word)
    transitionstate_ = Particle::enum_from_state_name(word);
  else
    FOUR_C_THROW("expecting particle state of phase change!");

  // safety check
  if (Particle::enum_to_state_dim(transitionstate_) != 1)
    FOUR_C_THROW("expecting scalar particle state for phase change!");

  // get transition value of phase change
  if (phasechangedefinition >> word)
  {
    try
    {
      transitionvalue_ = std::stod(word);
    }
    catch (...)
    {
      FOUR_C_THROW(
          "invalid argument: expecting a double value for transition value of phase change!");
    };
  }
  else
    FOUR_C_THROW("expecting transition value of phase change!");

  // probe for optional hysteresis gap at transition value
  if (phasechangedefinition >> word)
  {
    if (not(word == "hysteresis")) FOUR_C_THROW("expecting optional keyword 'hysteresis'!");

    // get hysteresis gap at transition value
    if (phasechangedefinition >> word)
    {
      try
      {
        hysteresisgap_ = std::stod(word);
      }
      catch (...)
      {
        FOUR_C_THROW("invalid argument: expecting a double value for hysteresis gap!");
      };
    }
    else
      FOUR_C_THROW("expecting hysteresis gap at transition value!");
  }
}

void Particle::SPHPhaseChangeBase::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::MaterialHandler> particlematerial,
    const std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set particle material handler
  particlematerial_ = particlematerial;

  // set equation of state handler
  equationofstatebundle_ = equationofstatebundle;

  // safety check
  for (const auto& type_i : {belowphase_, abovephase_})
    if (not particlecontainerbundle_->get_particle_types().contains(type_i))
      FOUR_C_THROW("no particle container for particle type '{}' found!",
          Particle::enum_to_type_name(type_i));
}

void Particle::SPHPhaseChangeBase::evaluate_phase_change_from_below_to_above_phase(
    std::vector<Particle::ParticleTypeToType>& particlesfromphasetophase,
    std::vector<std::set<int>>& particlestoremove,
    std::vector<std::vector<std::pair<int, Particle::ParticleObjShrdPtr>>>& particlestoinsert) const
{
  // set source and target type of particles
  Particle::TypeEnum type_source = belowphase_;
  Particle::TypeEnum type_target = abovephase_;

  // check for boundary or rigid particles
  bool isboundaryrigid_source =
      (type_source == Particle::BoundaryPhase or type_source == Particle::RigidPhase);
  bool isboundaryrigid_target =
      (type_target == Particle::BoundaryPhase or type_target == Particle::RigidPhase);

  // get container of owned particles of source particle type
  Particle::ParticleContainer* container =
      particlecontainerbundle_->get_specific_container(type_source, Particle::Owned);

  // get number of particles stored in container
  int particlestored = container->particles_stored();

  // no owned particles of current particle type
  if (particlestored <= 0) return;

  // get pointer to particle state
  const double* state = container->get_ptr_to_state(transitionstate_, 0);

  // get material for particle types
  const Mat::PAR::ParticleMaterialBase* material_source =
      particlematerial_->get_ptr_to_particle_mat_parameter(type_source);
  const Mat::PAR::ParticleMaterialBase* material_target =
      particlematerial_->get_ptr_to_particle_mat_parameter(type_target);

  // get equation of state of target particle type
  const Particle::SPHEquationOfStateBase* equationofstate_target;
  if (not isboundaryrigid_target)
    equationofstate_target =
        equationofstatebundle_->get_ptr_to_specific_equation_of_state(type_target);

  // iterate over owned particles of current type
  for (int index = 0; index < particlestored; ++index)
  {
    // evaluate phase change condition of current particle
    if (state[index] > (transitionvalue_ + 0.5 * hysteresisgap_))
    {
      int globalid(0);
      Particle::ParticleStates particlestates;
      container->get_particle(index, globalid, particlestates);

      // add density and pressure state for boundary or rigid particles
      if (isboundaryrigid_source and (not isboundaryrigid_target))
      {
        particlestates[Particle::Density].assign(1, material_source->initDensity_);

        const double press = equationofstate_target->density_to_pressure(
            material_source->initDensity_, material_target->initDensity_);

        particlestates[Particle::Pressure].assign(1, press);
      }

      // clear velocity and acceleration state of boundary or rigid particles
      if (isboundaryrigid_target and (not isboundaryrigid_source))
      {
        particlestates[Particle::Velocity].assign(3, 0.0);
        particlestates[Particle::Acceleration].assign(3, 0.0);
      }

      Particle::ParticleObjShrdPtr particleobject =
          std::make_shared<Particle::ParticleObject>(type_target, globalid, particlestates);

      // append particle to be insert
      particlestoinsert[type_target].push_back(std::make_pair(-1, particleobject));

      // store index of particle to be removed from containers
      particlestoremove[type_source].insert(index);

      // append source and target type together with global id of particle
      particlesfromphasetophase.push_back(std::make_tuple(type_source, type_target, globalid));
    }
  }
}

void Particle::SPHPhaseChangeBase::evaluate_phase_change_from_above_to_below_phase(
    std::vector<Particle::ParticleTypeToType>& particlesfromphasetophase,
    std::vector<std::set<int>>& particlestoremove,
    std::vector<std::vector<std::pair<int, Particle::ParticleObjShrdPtr>>>& particlestoinsert) const
{
  // set source and target type of particles
  Particle::TypeEnum type_source = abovephase_;
  Particle::TypeEnum type_target = belowphase_;

  // check for boundary or rigid particles
  bool isboundaryrigid_source =
      (type_source == Particle::BoundaryPhase or type_source == Particle::RigidPhase);
  bool isboundaryrigid_target =
      (type_target == Particle::BoundaryPhase or type_target == Particle::RigidPhase);

  // get container of owned particles of source particle type
  Particle::ParticleContainer* container =
      particlecontainerbundle_->get_specific_container(type_source, Particle::Owned);

  // get number of particles stored in container
  int particlestored = container->particles_stored();

  // no owned particles of current particle type
  if (particlestored <= 0) return;

  // get pointer to particle state
  const double* state = container->get_ptr_to_state(transitionstate_, 0);

  // get material for particle types
  const Mat::PAR::ParticleMaterialBase* material_source =
      particlematerial_->get_ptr_to_particle_mat_parameter(type_source);
  const Mat::PAR::ParticleMaterialBase* material_target =
      particlematerial_->get_ptr_to_particle_mat_parameter(type_target);

  // get equation of state of target particle type
  const Particle::SPHEquationOfStateBase* equationofstate_target;
  if (not isboundaryrigid_target)
    equationofstate_target =
        equationofstatebundle_->get_ptr_to_specific_equation_of_state(type_target);

  // iterate over owned particles of current type
  for (int index = 0; index < particlestored; ++index)
  {
    // evaluate phase change condition of current particle
    if (state[index] < (transitionvalue_ - 0.5 * hysteresisgap_))
    {
      int globalid(0);
      Particle::ParticleStates particlestates;
      container->get_particle(index, globalid, particlestates);

      // add density and pressure state for boundary or rigid particles
      if (isboundaryrigid_source and (not isboundaryrigid_target))
      {
        particlestates[Particle::Density].assign(1, material_source->initDensity_);

        const double press = equationofstate_target->density_to_pressure(
            material_source->initDensity_, material_target->initDensity_);

        particlestates[Particle::Pressure].assign(1, press);
      }

      // clear velocity and acceleration state of boundary or rigid particles
      if (isboundaryrigid_target and (not isboundaryrigid_source))
      {
        particlestates[Particle::Velocity].assign(3, 0.0);
        particlestates[Particle::Acceleration].assign(3, 0.0);
      }

      Particle::ParticleObjShrdPtr particleobject =
          std::make_shared<Particle::ParticleObject>(type_target, globalid, particlestates);

      // append particle to be insert
      particlestoinsert[type_target].push_back(std::make_pair(-1, particleobject));

      // store index of particle to be removed from containers
      particlestoremove[type_source].insert(index);

      // append source and target type together with global id of particle
      particlesfromphasetophase.push_back(std::make_tuple(type_source, type_target, globalid));
    }
  }
}

Particle::SPHPhaseChangeOneWayScalarBelowToAbove::SPHPhaseChangeOneWayScalarBelowToAbove(
    const Teuchos::ParameterList& params)
    : SPHPhaseChangeBase::SPHPhaseChangeBase(params)
{
  // empty constructor
}

void Particle::SPHPhaseChangeOneWayScalarBelowToAbove::evaluate_phase_change(
    std::vector<Particle::ParticleTypeToType>& particlesfromphasetophase) const
{
  // determine size of vectors indexed by particle types
  const int typevectorsize = *(--particlecontainerbundle_->get_particle_types().end()) + 1;

  std::vector<std::set<int>> particlestoremove(typevectorsize);
  std::vector<std::vector<std::pair<int, Particle::ParticleObjShrdPtr>>> particlestoinsert(
      typevectorsize);

  // evaluate phase change from below to above phase
  evaluate_phase_change_from_below_to_above_phase(
      particlesfromphasetophase, particlestoremove, particlestoinsert);

  // hand over particles to be removed
  particleengineinterface_->hand_over_particles_to_be_removed(particlestoremove);

  // hand over particles to be inserted
  particleengineinterface_->hand_over_particles_to_be_inserted(particlestoinsert);
}

Particle::SPHPhaseChangeOneWayScalarAboveToBelow::SPHPhaseChangeOneWayScalarAboveToBelow(
    const Teuchos::ParameterList& params)
    : SPHPhaseChangeBase::SPHPhaseChangeBase(params)
{
  // empty constructor
}

void Particle::SPHPhaseChangeOneWayScalarAboveToBelow::evaluate_phase_change(
    std::vector<Particle::ParticleTypeToType>& particlesfromphasetophase) const
{
  // determine size of vectors indexed by particle types
  const int typevectorsize = *(--particlecontainerbundle_->get_particle_types().end()) + 1;

  std::vector<std::set<int>> particlestoremove(typevectorsize);
  std::vector<std::vector<std::pair<int, Particle::ParticleObjShrdPtr>>> particlestoinsert(
      typevectorsize);

  // evaluate phase change from above to below phase
  evaluate_phase_change_from_above_to_below_phase(
      particlesfromphasetophase, particlestoremove, particlestoinsert);

  // hand over particles to be removed
  particleengineinterface_->hand_over_particles_to_be_removed(particlestoremove);

  // hand over particles to be inserted
  particleengineinterface_->hand_over_particles_to_be_inserted(particlestoinsert);
}

Particle::SPHPhaseChangeTwoWayScalar::SPHPhaseChangeTwoWayScalar(
    const Teuchos::ParameterList& params)
    : SPHPhaseChangeBase::SPHPhaseChangeBase(params)
{
  // empty constructor
}

void Particle::SPHPhaseChangeTwoWayScalar::evaluate_phase_change(
    std::vector<Particle::ParticleTypeToType>& particlesfromphasetophase) const
{
  // determine size of vectors indexed by particle types
  const int typevectorsize = *(--particlecontainerbundle_->get_particle_types().end()) + 1;

  std::vector<std::set<int>> particlestoremove(typevectorsize);
  std::vector<std::vector<std::pair<int, Particle::ParticleObjShrdPtr>>> particlestoinsert(
      typevectorsize);

  // evaluate phase change from below to above phase
  evaluate_phase_change_from_below_to_above_phase(
      particlesfromphasetophase, particlestoremove, particlestoinsert);

  // evaluate phase change from above to below phase
  evaluate_phase_change_from_above_to_below_phase(
      particlesfromphasetophase, particlestoremove, particlestoinsert);

  // hand over particles to be removed
  particleengineinterface_->hand_over_particles_to_be_removed(particlestoremove);

  // hand over particles to be inserted
  particleengineinterface_->hand_over_particles_to_be_inserted(particlestoinsert);
}

FOUR_C_NAMESPACE_CLOSE
