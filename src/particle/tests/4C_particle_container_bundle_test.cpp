// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_particle_engine_container_bundle.hpp"


namespace
{
  using namespace FourC;

  class ParticleContainerBundleTest : public ::testing::Test
  {
   protected:
    std::unique_ptr<Particle::ParticleContainerBundle> particlecontainerbundle_;

    int statesvectorsize_;

    ParticleContainerBundleTest()
    {
      // create particle container bundle
      particlecontainerbundle_ = std::make_unique<Particle::ParticleContainerBundle>();

      // init two phases with different particle states
      std::map<Particle::Type, std::set<Particle::State>> particlestatestotypes;
      std::set<Particle::State> stateEnumSet = {
          Particle::State::Position, Particle::State::Mass, Particle::State::Radius};
      particlestatestotypes.insert(std::make_pair(Particle::Type::Phase1, stateEnumSet));
      particlestatestotypes.insert(std::make_pair(Particle::Type::Phase2, stateEnumSet));

      // setup particle container bundle
      particlecontainerbundle_->setup(particlestatestotypes);

      const auto GetMaximumStoredStateEnumSetValue = [&stateEnumSet]()
      { return static_cast<int>(*(--stateEnumSet.end())); };
      statesvectorsize_ = static_cast<int>(GetMaximumStoredStateEnumSetValue()) + 1;

      // init some particles
      int index(0);
      int globalid(0);

      Particle::ParticleStates particle;
      particle.assign(statesvectorsize_, std::vector<double>{});

      // owned particles for phase 1
      {
        Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
            Particle::Type::Phase1, Particle::Status::Owned);

        // first particle
        globalid = 1;
        particle = create_test_particle({1.20, 0.70, 2.10}, {0.1}, {0.12});
        container->add_particle(index, globalid, particle);

        // second particle
        globalid = 2;
        particle = create_test_particle({-1.05, 12.6, -8.54}, {0.5}, {12.34});
        container->add_particle(index, globalid, particle);

        // third particle
        globalid = 3;
        particle = create_test_particle({-5.02, 2.26, -7.4}, {0.2}, {2.9});
        container->add_particle(index, globalid, particle);
      }

      // ghosted particles for phase 1
      {
        Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
            Particle::Type::Phase1, Particle::Status::Ghosted);

        // first particle
        globalid = 4;
        particle = create_test_particle({2.20, -0.52, 1.10}, {0.8}, {3.12});
        container->add_particle(index, globalid, particle);

        // second particle
        globalid = 5;
        particle = create_test_particle({-16.08, 1.46, -3.54}, {1.4}, {1.4});
        container->add_particle(index, globalid, particle);
      }

      // owned particles for phase 2
      {
        Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
            Particle::Type::Phase2, Particle::Status::Owned);

        // first particle
        globalid = 6;
        particle = create_test_particle({0.24, -1.71, -2.15}, {1.91}, {2.2});
        container->add_particle(index, globalid, particle);

        // second particle
        globalid = 7;
        particle = create_test_particle({-1.15, 2.6, 7.24}, {0.4}, {1.2});
        container->add_particle(index, globalid, particle);

        // third particle
        globalid = 8;
        particle = create_test_particle({5.12, 4.26, -3.4}, {1.1}, {0.2});
        container->add_particle(index, globalid, particle);
      }
    }

    Particle::ParticleStates create_test_particle(
        std::vector<double> pos, std::vector<double> mass, std::vector<double> rad)
    {
      Particle::ParticleStates particle;
      particle.assign(statesvectorsize_, std::vector<double>{});

      particle[static_cast<int>(Particle::State::Position)] = pos;
      particle[static_cast<int>(Particle::State::Mass)] = mass;
      particle[static_cast<int>(Particle::State::Radius)] = rad;

      return particle;
    }

    // note: the public functions setup() and get_specific_container() of class
    // ParticleContainerBundle are called in the constructor and thus implicitly tested by all
    // following unittests
  };

  void compare_particle_states(
      Particle::ParticleStates& particle_reference, Particle::ParticleStates& particle)
  {
    ASSERT_EQ(particle_reference.size(), particle.size());

    for (std::size_t i = 0; i < particle.size(); ++i)
    {
      std::vector<double>& state_reference = particle_reference[i];
      std::vector<double>& state = particle[i];

      ASSERT_EQ(state_reference.size(), state.size());

      for (std::size_t j = 0; j < state_reference.size(); ++j)
        EXPECT_NEAR(state_reference[j], state[j], 1e-14)
            << "state '" << Particle::enum_to_state_name(static_cast<Particle::ParticleState>(i))
            << "' j = " << j;
    }
  }

  TEST_F(ParticleContainerBundleTest, scale_state_specific_container)
  {
    particlecontainerbundle_->scale_state_specific_container(
        2.0, Particle::State::Radius, Particle::Type::Phase1);

    Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
        Particle::Type::Phase1, Particle::Status::Owned);

    ASSERT_EQ(container->particles_stored(), 3);

    int globalid(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});
    Particle::ParticleStates particle_reference;
    particle_reference.assign(statesvectorsize_, std::vector<double>{});

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference = create_test_particle({1.20, 0.70, 2.10}, {0.1}, {0.24});
      }
      else if (index == 1)
      {
        particle_reference = create_test_particle({-1.05, 12.6, -8.54}, {0.5}, {24.68});
      }
      else if (index == 2)
      {
        particle_reference = create_test_particle({-5.02, 2.26, -7.4}, {0.2}, {5.8});
      }

      container->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }
  }

  TEST_F(ParticleContainerBundleTest, update_state_specific_container)
  {
    particlecontainerbundle_->update_state_specific_container(
        2.0, Particle::State::Radius, 1.0, Particle::State::Mass, Particle::Type::Phase1);

    Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
        Particle::Type::Phase1, Particle::Status::Owned);

    ASSERT_EQ(container->particles_stored(), 3);

    int globalid(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});
    Particle::ParticleStates particle_reference;
    particle_reference.assign(statesvectorsize_, std::vector<double>{});

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference = create_test_particle({1.20, 0.70, 2.10}, {0.1}, {0.34});
      }
      else if (index == 1)
      {
        particle_reference = create_test_particle({-1.05, 12.6, -8.54}, {0.5}, {25.18});
      }
      else if (index == 2)
      {
        particle_reference = create_test_particle({-5.02, 2.26, -7.4}, {0.2}, {6.0});
      }

      container->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }
  }

  TEST_F(ParticleContainerBundleTest, set_state_specific_container)
  {
    std::vector<double> mass{1.1};

    particlecontainerbundle_->set_state_specific_container(
        mass, Particle::State::Mass, Particle::Type::Phase2);

    Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
        Particle::Type::Phase2, Particle::Status::Owned);

    ASSERT_EQ(container->particles_stored(), 3);

    int globalid(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});
    Particle::ParticleStates particle_reference;
    particle_reference.assign(statesvectorsize_, std::vector<double>{});

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference = create_test_particle({0.24, -1.71, -2.15}, mass, {2.2});
      }
      else if (index == 1)
      {
        particle_reference = create_test_particle({-1.15, 2.6, 7.24}, mass, {1.2});
      }
      else if (index == 2)
      {
        particle_reference = create_test_particle({5.12, 4.26, -3.4}, mass, {0.2});
      }

      container->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }
  }

  TEST_F(ParticleContainerBundleTest, clear_state_specific_container)
  {
    std::vector<double> mass{0.0};

    particlecontainerbundle_->clear_state_specific_container(
        Particle::State::Mass, Particle::Type::Phase2);

    Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
        Particle::Type::Phase2, Particle::Status::Owned);

    ASSERT_EQ(container->particles_stored(), 3);

    int globalid(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});
    Particle::ParticleStates particle_reference;
    particle_reference.assign(statesvectorsize_, std::vector<double>{});

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference = create_test_particle({0.24, -1.71, -2.15}, mass, {2.2});
      }
      else if (index == 1)
      {
        particle_reference = create_test_particle({-1.15, 2.6, 7.24}, mass, {1.2});
      }
      else if (index == 2)
      {
        particle_reference = create_test_particle({5.12, 4.26, -3.4}, mass, {0.2});
      }

      container->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }
  }

  TEST_F(ParticleContainerBundleTest, scale_state_all_containers)
  {
    particlecontainerbundle_->scale_state_all_containers(2.0, Particle::State::Mass);

    Particle::ParticleContainer* container = nullptr;
    int globalid(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});
    Particle::ParticleStates particle_reference;
    particle_reference.assign(statesvectorsize_, std::vector<double>{});

    container = particlecontainerbundle_->get_specific_container(
        Particle::Type::Phase1, Particle::Status::Owned);

    ASSERT_EQ(container->particles_stored(), 3);

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Phase1, Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference = create_test_particle({1.20, 0.70, 2.10}, {0.2}, {0.12});
      }
      else if (index == 1)
      {
        particle_reference = create_test_particle({-1.05, 12.6, -8.54}, {1.0}, {12.34});
      }
      else if (index == 2)
      {
        particle_reference = create_test_particle({-5.02, 2.26, -7.4}, {0.4}, {2.9});
      }

      container->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }

    container = particlecontainerbundle_->get_specific_container(
        Particle::Type::Phase2, Particle::Status::Owned);

    ASSERT_EQ(container->particles_stored(), 3);

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Phase2, Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference = create_test_particle({0.24, -1.71, -2.15}, {3.82}, {2.2});
      }
      else if (index == 1)
      {
        particle_reference = create_test_particle({-1.15, 2.6, 7.24}, {0.8}, {1.2});
      }
      else if (index == 2)
      {
        particle_reference = create_test_particle({5.12, 4.26, -3.4}, {2.2}, {0.2});
      }

      container->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }
  }

  TEST_F(ParticleContainerBundleTest, update_state_all_containers)
  {
    particlecontainerbundle_->update_state_all_containers(
        2.0, Particle::State::Mass, 1.0, Particle::State::Radius);

    Particle::ParticleContainer* container = nullptr;
    int globalid(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});
    Particle::ParticleStates particle_reference;
    particle_reference.assign(statesvectorsize_, std::vector<double>{});

    container = particlecontainerbundle_->get_specific_container(
        Particle::Type::Phase1, Particle::Status::Owned);

    ASSERT_EQ(container->particles_stored(), 3);

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Phase1, Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference = create_test_particle({1.20, 0.70, 2.10}, {0.32}, {0.12});
      }
      else if (index == 1)
      {
        particle_reference = create_test_particle({-1.05, 12.6, -8.54}, {13.34}, {12.34});
      }
      else if (index == 2)
      {
        particle_reference = create_test_particle({-5.02, 2.26, -7.4}, {3.3}, {2.9});
      }

      container->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }

    container = particlecontainerbundle_->get_specific_container(
        Particle::Type::Phase2, Particle::Status::Owned);

    ASSERT_EQ(container->particles_stored(), 3);

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Phase2, Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference = create_test_particle({0.24, -1.71, -2.15}, {6.02}, {2.2});
      }
      else if (index == 1)
      {
        particle_reference = create_test_particle({-1.15, 2.6, 7.24}, {2.0}, {1.2});
      }
      else if (index == 2)
      {
        particle_reference = create_test_particle({5.12, 4.26, -3.4}, {2.4}, {0.2});
      }

      container->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }
  }

  TEST_F(ParticleContainerBundleTest, set_state_all_containers)
  {
    std::vector<double> mass{1.1};

    particlecontainerbundle_->set_state_all_containers(mass, Particle::State::Mass);

    Particle::ParticleContainer* container = nullptr;
    int globalid(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});
    Particle::ParticleStates particle_reference;
    particle_reference.assign(statesvectorsize_, std::vector<double>{});

    container = particlecontainerbundle_->get_specific_container(
        Particle::Type::Phase1, Particle::Status::Owned);

    ASSERT_EQ(container->particles_stored(), 3);

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Phase1, Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference = create_test_particle({1.20, 0.70, 2.10}, mass, {0.12});
      }
      else if (index == 1)
      {
        particle_reference = create_test_particle({-1.05, 12.6, -8.54}, mass, {12.34});
      }
      else if (index == 2)
      {
        particle_reference = create_test_particle({-5.02, 2.26, -7.4}, mass, {2.9});
      }

      container->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }

    container = particlecontainerbundle_->get_specific_container(
        Particle::Type::Phase2, Particle::Status::Owned);

    ASSERT_EQ(container->particles_stored(), 3);

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Phase2, Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference = create_test_particle({0.24, -1.71, -2.15}, mass, {2.2});
      }
      else if (index == 1)
      {
        particle_reference = create_test_particle({-1.15, 2.6, 7.24}, mass, {1.2});
      }
      else if (index == 2)
      {
        particle_reference = create_test_particle({5.12, 4.26, -3.4}, mass, {0.2});
      }

      container->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }
  }

  TEST_F(ParticleContainerBundleTest, clear_state_all_containers)
  {
    std::vector<double> mass{0.0};

    particlecontainerbundle_->clear_state_all_containers(Particle::State::Mass);

    Particle::ParticleContainer* container = nullptr;
    int globalid(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});
    Particle::ParticleStates particle_reference;
    particle_reference.assign(statesvectorsize_, std::vector<double>{});

    container = particlecontainerbundle_->get_specific_container(
        Particle::Type::Phase1, Particle::Status::Owned);

    ASSERT_EQ(container->particles_stored(), 3);

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Phase1, Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference = create_test_particle({1.20, 0.70, 2.10}, mass, {0.12});
      }
      else if (index == 1)
      {
        particle_reference = create_test_particle({-1.05, 12.6, -8.54}, mass, {12.34});
      }
      else if (index == 2)
      {
        particle_reference = create_test_particle({-5.02, 2.26, -7.4}, mass, {2.9});
      }

      container->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }

    container = particlecontainerbundle_->get_specific_container(
        Particle::Type::Phase2, Particle::Status::Owned);

    ASSERT_EQ(container->particles_stored(), 3);

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Phase2, Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference = create_test_particle({0.24, -1.71, -2.15}, mass, {2.2});
      }
      else if (index == 1)
      {
        particle_reference = create_test_particle({-1.15, 2.6, 7.24}, mass, {1.2});
      }
      else if (index == 2)
      {
        particle_reference = create_test_particle({5.12, 4.26, -3.4}, mass, {0.2});
      }

      container->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }
  }

  TEST_F(ParticleContainerBundleTest, check_and_decrease_size_all_containers_of_specific_status)
  {
    Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
        Particle::Type::Phase1, Particle::Status::Owned);

    ASSERT_EQ(container->particles_stored(), 3);
    ASSERT_EQ(container->container_size(), 4);

    container->remove_particle(0);
    container->remove_particle(0);

    particlecontainerbundle_->check_and_decrease_size_all_containers_of_specific_status(
        Particle::Status::Owned);

    EXPECT_EQ(container->particles_stored(), 1);
    EXPECT_EQ(container->container_size(), 2);
  }

  TEST_F(ParticleContainerBundleTest, clear_all_containers_of_specific_status)
  {
    particlecontainerbundle_->clear_all_containers_of_specific_status(Particle::Status::Ghosted);

    Particle::ParticleContainer* container = particlecontainerbundle_->get_specific_container(
        Particle::Type::Phase1, Particle::Status::Ghosted);

    EXPECT_EQ(container->particles_stored(), 0);
  }

  TEST_F(ParticleContainerBundleTest, get_vector_of_particle_objects_of_all_containers)
  {
    std::vector<Particle::ParticleObjShrdPtr> particlesstored;

    particlecontainerbundle_->get_vector_of_particle_objects_of_all_containers(particlesstored);

    EXPECT_EQ(particlesstored.size(), 6);
  }
}  // namespace
