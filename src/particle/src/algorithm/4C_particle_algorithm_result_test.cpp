// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_algorithm_result_test.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_container_bundle.hpp"
#include "4C_particle_engine_interface.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::ParticleResultTest::ParticleResultTest() : Core::Utils::ResultTest("PARTICLE")
{
  // empty constructor
}

void Particle::ParticleResultTest::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;
}

void Particle::ParticleResultTest::test_special(
    const Core::IO::InputParameterContainer& result_container, int& nerr, int& test_count)
{
  // extract global particle id
  int globalid = result_container.get<int>("ID");

  // get local index in specific particle container
  Particle::LocalIndexTupleShrdPtr localindextuple =
      particleengineinterface_->get_local_index_in_specific_container(globalid);

  // particle with global id found on this processor
  if (localindextuple)
  {
    // access values of local index tuple
    Particle::TypeEnum particleType;
    Particle::StatusEnum particleStatus;
    int index;
    std::tie(particleType, particleStatus, index) = *localindextuple;

    // consider only owned particle
    if (particleStatus == Particle::Owned)
    {
      // get particle container bundle
      Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
          particleengineinterface_->get_particle_container_bundle();

      // get container of owned particles of current particle type
      Particle::ParticleContainer* container =
          particlecontainerbundle->get_specific_container(particleType, Particle::Owned);

      // get result
      std::string quantity = result_container.get<std::string>("QUANTITY");

      // init actual result
      double actresult = 0.0;

      // component of result
      int dim = 0;

      // declare enum of particle state
      Particle::StateEnum particleState;

      // position
      if (quantity == "posx" or quantity == "posy" or quantity == "posz")
      {
        // get enum of particle state
        particleState = Particle::Position;

        // get component of result
        if (quantity == "posx")
          dim = 0;
        else if (quantity == "posy")
          dim = 1;
        else if (quantity == "posz")
          dim = 2;
      }
      // velocity
      else if (quantity == "velx" or quantity == "vely" or quantity == "velz")
      {
        // get enum of particle state
        particleState = Particle::Velocity;

        // get component of result
        if (quantity == "velx")
          dim = 0;
        else if (quantity == "vely")
          dim = 1;
        else if (quantity == "velz")
          dim = 2;
      }
      // acceleration
      else if (quantity == "accx" or quantity == "accy" or quantity == "accz")
      {
        // get enum of particle state
        particleState = Particle::Acceleration;

        // get component of result
        if (quantity == "accx")
          dim = 0;
        else if (quantity == "accy")
          dim = 1;
        else if (quantity == "accz")
          dim = 2;
      }
      // angular velocity
      else if (quantity == "angvelx" or quantity == "angvely" or quantity == "angvelz")
      {
        // get enum of particle state
        particleState = Particle::AngularVelocity;

        // get component of result
        if (quantity == "angvelx")
          dim = 0;
        else if (quantity == "angvely")
          dim = 1;
        else if (quantity == "angvelz")
          dim = 2;
      }
      // radius
      else if (quantity == "radius")
      {
        // get enum of particle state
        particleState = Particle::Radius;

        // get component of result
        dim = 0;
      }
      // density
      else if (quantity == "density")
      {
        // get enum of particle state
        particleState = Particle::Density;

        // get component of result
        dim = 0;
      }
      // pressure
      else if (quantity == "pressure")
      {
        // get enum of particle state
        particleState = Particle::Pressure;

        // get component of result
        dim = 0;
      }
      // temperature
      else if (quantity == "temperature")
      {
        // get enum of particle state
        particleState = Particle::Temperature;

        // get component of result
        dim = 0;
      }
      // temperature gradient
      else if (quantity == "tempgradx" or quantity == "tempgrady" or quantity == "tempgradz")
      {
        // get enum of particle state
        particleState = Particle::temperature_gradient;

        // get component of result
        if (quantity == "tempgradx")
          dim = 0;
        else if (quantity == "tempgrady")
          dim = 1;
        else if (quantity == "tempgradz")
          dim = 2;
      }
      // phi
      else if (quantity == "pd_damage_phi")
      {
        // get enum of particle state
        particleState = Particle::PDDamageVariable;

        // get component of result
        dim = 0;
      }
      else
        FOUR_C_THROW("result check failed with unknown quantity '{}'!", quantity);

      // container contains current particle state
      if (not container->have_stored_state(particleState))
        FOUR_C_THROW(
            "state '{}' not found in container!", Particle::enum_to_state_name(particleState));

      // get pointer to particle state
      const double* state = container->get_ptr_to_state(particleState, 0);

      // get particle state dimension
      int statedim = container->get_state_dim(particleState);

      // get actual result
      actresult = state[statedim * index + dim];

      // compare values
      const int err = compare_values(actresult, "SPECIAL", result_container);
      nerr += err;
      test_count++;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
