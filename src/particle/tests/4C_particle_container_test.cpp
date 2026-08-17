// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_particle_engine_container.hpp"
#include "4C_unittest_utils_assertions_test.hpp"


namespace
{
  using namespace FourC;

  class ParticleContainerTest : public ::testing::Test
  {
   protected:
    std::unique_ptr<Particle::ParticleContainer> container_;

    int statesvectorsize_;

    ParticleContainerTest()
    {
      const int size = 7;
      std::set<Particle::State> stateEnumSet = {
          Particle::State::Position, Particle::State::Velocity, Particle::State::Mass};

      // create, init and setup container
      container_ = std::make_unique<Particle::ParticleContainer>();
      container_->setup(size, stateEnumSet);

      const int maximum_stored_state_enum_set_value{static_cast<int>(*(--stateEnumSet.end()))};
      statesvectorsize_ = maximum_stored_state_enum_set_value + 1;

      // init some particles
      int index(0);
      int globalid(0);

      Particle::ParticleStates particle;
      particle.assign(statesvectorsize_, std::vector<double>{});

      // first particle
      {
        globalid = 1;
        particle = create_test_particle({1.20, 0.70, 2.10}, {0.23, 1.76, 3.89}, {0.12});
        container_->add_particle(index, globalid, particle);
      }

      // second particle
      {
        globalid = 2;
        particle = create_test_particle({-1.05, 12.6, -8.54}, {0.25, -21.5, 1.0}, {12.34});
        container_->add_particle(index, globalid, particle);
      }

      // third particle
      {
        globalid = 3;
        particle = create_test_particle({61.0, -2.63, 0.11}, {-7.35, -5.98, 1.11}, {0.5});
        container_->add_particle(index, globalid, particle);
      }
    }

    Particle::ParticleStates create_test_particle(
        std::vector<double> pos, std::vector<double> vel, std::vector<double> mass)
    {
      Particle::ParticleStates particle;
      particle.assign(statesvectorsize_, std::vector<double>{});

      particle[static_cast<int>(Particle::State::Position)] = pos;
      particle[static_cast<int>(Particle::State::Velocity)] = vel;
      particle[static_cast<int>(Particle::State::Mass)] = mass;

      return particle;
    }

    // note: the public functions setup() and AddParticle() of class ParticleContainer are
    // called in the constructor and thus implicitly tested by all following unittests
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
            << "state '" << Particle::enum_to_state_name(static_cast<Particle::State>(i))
            << "' j = " << j;
    }
  }

  TEST_F(ParticleContainerTest, increase_container_size)
  {
    container_->increase_container_size();
    EXPECT_EQ(container_->particles_stored(), 3);
    EXPECT_EQ(container_->container_size(), 14);
  }

  TEST_F(ParticleContainerTest, decrease_container_size)
  {
    container_->decrease_container_size();
    EXPECT_EQ(container_->particles_stored(), 3);
    EXPECT_EQ(container_->container_size(), 3);
  }

  TEST_F(ParticleContainerTest, check_and_decrease_container_size)
  {
    container_->check_and_decrease_container_size();
    EXPECT_EQ(container_->particles_stored(), 3);
    EXPECT_EQ(container_->container_size(), 3);
  }

  TEST_F(ParticleContainerTest, ClearContainer)
  {
    container_->clear_container();
    EXPECT_EQ(container_->particles_stored(), 0);
  }

  TEST_F(ParticleContainerTest, AddParticle)
  {
    // init a particle
    int globalid(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});

    globalid = 4;
    particle = create_test_particle({-1.23, 1.70, 9.10}, {6.23, 2.3, 6.9}, {5.12});

    int index(0);
    container_->add_particle(index, globalid, particle);
    EXPECT_EQ(index, 3);
    EXPECT_EQ(container_->particles_stored(), 4);
  }

  TEST_F(ParticleContainerTest, ReplaceParticle)
  {
    // init a particle
    int globalid(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});
    Particle::ParticleStates particle_reference;
    particle_reference.assign(statesvectorsize_, std::vector<double>{});

    particle_reference = create_test_particle({-1.23, 1.70, 9.10}, {6.23, 2.3, 6.9}, {5.12});

    int index = 0;

    // replace only states and leave global id untouched
    container_->replace_particle(index, -1, particle_reference);
    EXPECT_EQ(container_->particles_stored(), 3);
    container_->get_particle(index, globalid, particle);
    EXPECT_EQ(globalid, 1);
    compare_particle_states(particle_reference, particle);

    // also replace global id
    container_->replace_particle(index, 4, particle_reference);
    EXPECT_EQ(container_->particles_stored(), 3);
    container_->get_particle(index, globalid, particle);
    EXPECT_EQ(globalid, 4);
    compare_particle_states(particle_reference, particle);
  }

  TEST_F(ParticleContainerTest, GetParticle)
  {
    int globalid(0);
    int globalid_reference(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});
    Particle::ParticleStates particle_reference;
    particle_reference.assign(statesvectorsize_, std::vector<double>{});

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Particle " + std::to_string(index));
      if (index == 0)
      {
        globalid_reference = 1;
        particle_reference = create_test_particle({1.20, 0.70, 2.10}, {0.23, 1.76, 3.89}, {0.12});
      }
      else if (index == 1)
      {
        globalid_reference = 2;
        particle_reference =
            create_test_particle({-1.05, 12.6, -8.54}, {0.25, -21.5, 1.0}, {12.34});
      }
      else if (index == 2)
      {
        globalid_reference = 3;
        particle_reference = create_test_particle({61.0, -2.63, 0.11}, {-7.35, -5.98, 1.11}, {0.5});
      }

      container_->get_particle(index, globalid, particle);
      EXPECT_EQ(globalid_reference, globalid);
      compare_particle_states(particle_reference, particle);
    }
  }

  TEST_F(ParticleContainerTest, RemoveParticle)
  {
    int globalid(0);
    int globalid_reference(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});
    Particle::ParticleStates particle_reference;
    particle_reference.assign(statesvectorsize_, std::vector<double>{});

    container_->remove_particle(0);
    EXPECT_EQ(container_->particles_stored(), 2);

    for (int index = 0; index < 2; ++index)
    {
      SCOPED_TRACE("Particle " + std::to_string(index));
      if (index == 0)
      {
        globalid_reference = 3;
        particle_reference = create_test_particle({61.0, -2.63, 0.11}, {-7.35, -5.98, 1.11}, {0.5});
      }
      else if (index == 1)
      {
        globalid_reference = 2;
        particle_reference =
            create_test_particle({-1.05, 12.6, -8.54}, {0.25, -21.5, 1.0}, {12.34});
      }

      container_->get_particle(index, globalid, particle);
      EXPECT_EQ(globalid_reference, globalid);
      compare_particle_states(particle_reference, particle);
    }
  }

  TEST_F(ParticleContainerTest, GetStateDim)
  {
    EXPECT_EQ(container_->get_state_dim(Particle::State::Position), 3);
    EXPECT_EQ(container_->get_state_dim(Particle::State::Velocity), 3);
    EXPECT_EQ(container_->get_state_dim(Particle::State::Mass), 1);
  }

  TEST_F(ParticleContainerTest, GetPtrToState)
  {
    std::array<double, 3> pos = {0.0};
    std::array<double, 3> vel = {0.0};
    std::array<double, 1> mass = {0.0};

    for (int index = 0; index < 3; ++index)
    {
      if (index == 0)
      {
        pos[0] = 1.20;
        pos[1] = 0.70;
        pos[2] = 2.10;
        vel[0] = 0.23;
        vel[1] = 1.76;
        vel[2] = 3.89;
        mass[0] = 0.12;
      }
      else if (index == 1)
      {
        pos[0] = -1.05;
        pos[1] = 12.6;
        pos[2] = -8.54;
        vel[0] = 0.25;
        vel[1] = -21.5;
        vel[2] = 1.0;
        mass[0] = 12.34;
      }
      else if (index == 2)
      {
        pos[0] = 61.0;
        pos[1] = -2.63;
        pos[2] = 0.11;
        vel[0] = -7.35;
        vel[1] = -5.98;
        vel[2] = 1.11;
        mass[0] = 0.5;
      }

      const double* currpos = container_->get_ptr_to_state(Particle::State::Position, index);
      FOUR_C_EXPECT_ITERABLE_NEAR(currpos, pos.begin(), 3, 1e-14);

      const double* currvel = container_->get_ptr_to_state(Particle::State::Velocity, index);
      FOUR_C_EXPECT_ITERABLE_NEAR(currvel, vel.begin(), 3, 1e-14);

      const double* currmass = container_->get_ptr_to_state(Particle::State::Mass, index);
      EXPECT_NEAR(currmass[0], mass[0], 1e-14);

      {
        double* newmass = container_->get_ptr_to_state_writable(Particle::State::Mass, index);
        newmass[0] = newmass[0] * 3.14;
      }
      {
        const double* currmass = container_->get_ptr_to_state(Particle::State::Mass, index);
        EXPECT_NEAR(currmass[0], mass[0] * 3.14, 1e-14);
      }
    }
  }

  TEST_F(ParticleContainerTest, TryGetPtrToState)
  {
    std::array<double, 3> pos = {0.0};
    std::array<double, 3> vel = {0.0};
    std::array<double, 1> mass = {0.0};

    for (int index = 0; index < 3; ++index)
    {
      if (index == 0)
      {
        pos[0] = 1.20;
        pos[1] = 0.70;
        pos[2] = 2.10;
        vel[0] = 0.23;
        vel[1] = 1.76;
        vel[2] = 3.89;
        mass[0] = 0.12;
      }
      else if (index == 1)
      {
        pos[0] = -1.05;
        pos[1] = 12.6;
        pos[2] = -8.54;
        vel[0] = 0.25;
        vel[1] = -21.5;
        vel[2] = 1.0;
        mass[0] = 12.34;
      }
      else if (index == 2)
      {
        pos[0] = 61.0;
        pos[1] = -2.63;
        pos[2] = 0.11;
        vel[0] = -7.35;
        vel[1] = -5.98;
        vel[2] = 1.11;
        mass[0] = 0.5;
      }

      const double* currpos = container_->try_get_ptr_to_state(Particle::State::Position, index);
      FOUR_C_EXPECT_ITERABLE_NEAR(currpos, pos.begin(), 3, 1.0e-14);

      const double* currvel = container_->try_get_ptr_to_state(Particle::State::Velocity, index);
      FOUR_C_EXPECT_ITERABLE_NEAR(currvel, vel.begin(), 3, 1.0e-14);

      const double* currmass = container_->try_get_ptr_to_state(Particle::State::Mass, index);
      EXPECT_NEAR(currmass[0], mass[0], 1e-14);

      {
        double* newmass = container_->try_get_ptr_to_state_writable(Particle::State::Mass, index);
        newmass[0] = newmass[0] * 3.14;
      }
      {
        const double* currmass = container_->try_get_ptr_to_state(Particle::State::Mass, index);
        EXPECT_NEAR(currmass[0], mass[0] * 3.14, 1e-14);
      }
    }
  }

  TEST_F(ParticleContainerTest, TryGetPtrToStateNotStored)
  {
    const double* acc = container_->try_get_ptr_to_state(Particle::State::Acceleration, 0);
    EXPECT_EQ(acc, nullptr);

    double* angacc =
        container_->try_get_ptr_to_state_writable(Particle::State::AngularAcceleration, 0);
    EXPECT_EQ(angacc, nullptr);
  }

  TEST_F(ParticleContainerTest, GetPtrToGlobalID)
  {
    int globalid_reference(0);

    for (int index = 0; index < 3; ++index)
    {
      if (index == 0)
        globalid_reference = 1;
      else if (index == 1)
        globalid_reference = 2;
      else if (index == 2)
        globalid_reference = 3;

      int* globalid = container_->get_ptr_to_global_id(index);
      EXPECT_EQ(globalid[0], globalid_reference);
    }
  }

  TEST_F(ParticleContainerTest, ScaleState)
  {
    int globalid(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});
    Particle::ParticleStates particle_reference;
    particle_reference.assign(statesvectorsize_, std::vector<double>{});

    container_->scale_state(1.5, Particle::State::Position);
    container_->scale_state(3.25, Particle::State::Velocity, Particle::Space::Device);
    container_->scale_state(0.95, Particle::State::Mass);

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference =
            create_test_particle({1.8, 1.05, 3.15}, {0.7475, 5.72, 12.6425}, {0.114});
      }
      else if (index == 1)
      {
        particle_reference =
            create_test_particle({-1.575, 18.9, -12.81}, {0.8125, -69.875, 3.25}, {11.723});
      }
      else if (index == 2)
      {
        particle_reference =
            create_test_particle({91.5, -3.945, 0.165}, {-23.8875, -19.435, 3.6075}, {0.475});
      }

      container_->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }

    container_->scale_state(0.5, Particle::State::Mass, Particle::Space::Device);

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference =
            create_test_particle({1.8, 1.05, 3.15}, {0.7475, 5.72, 12.6425}, {0.057});
      }
      else if (index == 1)
      {
        particle_reference =
            create_test_particle({-1.575, 18.9, -12.81}, {0.8125, -69.875, 3.25}, {5.8615});
      }
      else if (index == 2)
      {
        particle_reference =
            create_test_particle({91.5, -3.945, 0.165}, {-23.8875, -19.435, 3.6075}, {0.2375});
      }

      container_->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }
  }

  TEST_F(ParticleContainerTest, UpdateState)
  {
    int globalid(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});
    Particle::ParticleStates particle_reference;
    particle_reference.assign(statesvectorsize_, std::vector<double>{});

    container_->update_state(1.0, Particle::State::Position, 0.5, Particle::State::Velocity);

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference = create_test_particle({1.315, 1.58, 4.045}, {0.23, 1.76, 3.89}, {0.12});
      }
      else if (index == 1)
      {
        particle_reference =
            create_test_particle({-0.925, 1.85, -8.04}, {0.25, -21.5, 1.0}, {12.34});
      }
      else if (index == 2)
      {
        particle_reference =
            create_test_particle({57.325, -5.62, 0.665}, {-7.35, -5.98, 1.11}, {0.5});
      }

      container_->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }

    container_->update_state(
        1.0, Particle::State::Position, 1.0, Particle::State::Velocity, Particle::Space::Device);

    for (int index = 0; index < 3; ++index)
    {
      SCOPED_TRACE("Particle " + std::to_string(index));
      if (index == 0)
      {
        particle_reference = create_test_particle({1.545, 3.34, 7.935}, {0.23, 1.76, 3.89}, {0.12});
      }
      else if (index == 1)
      {
        particle_reference =
            create_test_particle({-0.675, -19.65, -7.04}, {0.25, -21.5, 1.0}, {12.34});
      }
      else if (index == 2)
      {
        particle_reference =
            create_test_particle({49.975, -11.6, 1.775}, {-7.35, -5.98, 1.11}, {0.5});
      }

      container_->get_particle(index, globalid, particle);

      compare_particle_states(particle_reference, particle);
    }
  }

  TEST_F(ParticleContainerTest, SetState)
  {
    int globalid(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});
    Particle::ParticleStates particle_reference;
    particle_reference.assign(statesvectorsize_, std::vector<double>{});

    std::vector<double> pos(3);
    std::vector<double> vel(3);
    std::vector<double> mass(1);

    pos[0] = 3.15;
    pos[1] = -1.45;
    pos[2] = 9.5;
    vel[0] = -21.30;
    vel[1] = -4.33;
    vel[2] = 0.933;
    mass[0] = 1.234;

    particle_reference = create_test_particle(pos, vel, mass);

    container_->set_state(pos, Particle::State::Position);
    container_->set_state(vel, Particle::State::Velocity);
    container_->set_state(mass, Particle::State::Mass, Particle::Space::Device);

    for (int index = 0; index < 3; ++index)
    {
      container_->get_particle(index, globalid, particle);
      compare_particle_states(particle_reference, particle);
    }
  }

  TEST_F(ParticleContainerTest, ClearState)
  {
    int globalid(0);

    Particle::ParticleStates particle;
    particle.assign(statesvectorsize_, std::vector<double>{});
    Particle::ParticleStates particle_reference;
    particle_reference.assign(statesvectorsize_, std::vector<double>{});

    particle_reference = create_test_particle({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0});

    container_->clear_state(Particle::State::Position);
    container_->clear_state(Particle::State::Velocity);
    container_->clear_state(Particle::State::Mass, Particle::Space::Device);

    for (int index = 0; index < 3; ++index)
    {
      container_->get_particle(index, globalid, particle);
      compare_particle_states(particle_reference, particle);
    }
  }

  TEST_F(ParticleContainerTest, GetMinValueOfState)
  {
    EXPECT_NEAR(container_->get_min_value_of_state(Particle::State::Position), -8.54, 1e-14);
    EXPECT_NEAR(container_->get_min_value_of_state(Particle::State::Mass, Particle::Space::Device),
        0.12, 1e-14);
  }

  TEST_F(ParticleContainerTest, GetMaxValueOfState)
  {
    EXPECT_NEAR(container_->get_max_value_of_state(Particle::State::Mass), 12.34, 1e-14);
    EXPECT_NEAR(
        container_->get_max_value_of_state(Particle::State::Position, Particle::Space::Device),
        61.0, 1e-14);
  }

  TEST_F(ParticleContainerTest, GetStoredStates)
  {
    const std::set<Particle::State>& particleStates = container_->get_stored_states();
    EXPECT_EQ(particleStates.size(), 3);
    EXPECT_TRUE(particleStates.find(Particle::State::Position) != particleStates.end());
    EXPECT_TRUE(particleStates.find(Particle::State::Velocity) != particleStates.end());
    EXPECT_TRUE(particleStates.find(Particle::State::Mass) != particleStates.end());
  }

  TEST_F(ParticleContainerTest, HaveStoredState)
  {
    EXPECT_TRUE(container_->have_stored_state(Particle::State::Position));
    EXPECT_TRUE(container_->have_stored_state(Particle::State::Velocity));
    EXPECT_TRUE(container_->have_stored_state(Particle::State::Mass));

    EXPECT_FALSE(container_->have_stored_state(Particle::State::Acceleration));
    EXPECT_FALSE(container_->have_stored_state(Particle::State::Density));
  }

  TEST_F(ParticleContainerTest, ContainerSize) { EXPECT_EQ(container_->container_size(), 7); }

  TEST_F(ParticleContainerTest, ParticlesStored) { EXPECT_EQ(container_->particles_stored(), 3); }
}  // namespace
