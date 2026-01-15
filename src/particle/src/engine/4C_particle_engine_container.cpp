// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_engine_container.hpp"

#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::ParticleContainer::ParticleContainer()
    : containersize_(0), particlestored_(0), statesvectorsize_(0), globalids_(0, -1)
{
  // empty constructor
}

void Particle::ParticleContainer::setup(int containersize, const std::set<ParticleState>& stateset)
{
  // set size of particle container (at least one)
  containersize_ = (containersize > 0) ? containersize : 1;

  // set of stored particle states
  storedstates_ = stateset;

  // determine necessary size of vector for states
  statesvectorsize_ = *(--storedstates_.end()) + 1;

  // allocate memory for global ids
  globalids_.resize(containersize_, -1);

  // allocate memory to hold particle states and dimension
  states_.resize(statesvectorsize_);
  statedim_.resize(statesvectorsize_);

  // iterate over states to be stored in container
  for (const auto& state : storedstates_)
  {
    // set particle state dimension for current state
    statedim_[state] = enum_to_state_dim(state);

    // allocate memory for current state in particle container
    states_[state].resize(containersize_ * statedim_[state]);
  }
}

void Particle::ParticleContainer::increase_container_size()
{
  // size of container is doubled
  containersize_ *= 2;

  // resize vector of global ids
  globalids_.resize(containersize_);

  // iterate over states stored in container
  for (const auto& state : storedstates_)
  {
    // resize vector of current state
    states_[state].resize(containersize_ * statedim_[state]);
  }
}

void Particle::ParticleContainer::decrease_container_size()
{
  // size of container is halved
  int newsize = static_cast<int>(0.5 * containersize_);

  // set size of particle container (at least one)
  containersize_ = (newsize > 0) ? newsize : 1;

#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (particlestored_ > containersize_)
    FOUR_C_THROW(
        "decreasing size of container not possible: particles stored {} > new container size {}!",
        particlestored_, containersize_);
#endif

  // resize vector of global ids
  globalids_.resize(containersize_);

  // iterate over states stored in container
  for (const auto& state : storedstates_)
  {
    // resize vector of current state
    states_[state].resize(containersize_ * statedim_[state]);
  }
}

void Particle::ParticleContainer::add_particle(
    int& index, int globalid, const ParticleStates& states)
{
  // increase size of container
  if (particlestored_ == containersize_) increase_container_size();

  // store global id
  globalids_[particlestored_] = globalid;

  // iterate over states stored in container
  for (const auto& state : storedstates_)
  {
    // state not handed over
    if (states.size() <= state or states[state].empty())
    {
      // initialize to zero
      for (int dim = 0; dim < statedim_[state]; ++dim)
        (states_[state])[particlestored_ * statedim_[state] + dim] = 0.0;
    }
    // state handed over
    else
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (static_cast<int>(states[state].size()) != statedim_[state])
        FOUR_C_THROW("can not add particle: dimensions of state '{}' do not match!",
            enum_to_state_name(state));
#endif

      // store state in container
      for (int dim = 0; dim < statedim_[state]; ++dim)
        (states_[state])[particlestored_ * statedim_[state] + dim] = states[state][dim];
    }
  }

  // set index of added particle
  index = particlestored_;

  // increase counter of stored particles
  particlestored_++;
}

void Particle::ParticleContainer::replace_particle(
    int index, int globalid, const ParticleStates& states)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (index < 0 or index > (particlestored_ - 1))
    FOUR_C_THROW("can not replace particle as index {} out of bounds!", index);
#endif

  // replace global id in container
  if (globalid >= 0) globalids_[index] = globalid;

  // iterate over states stored in container
  for (const auto& state : storedstates_)
  {
    // state not handed over
    if (states.size() <= state or states[state].empty())
    {
      // leave state untouched
    }
    // state handed over
    else
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (static_cast<int>(states[state].size()) != statedim_[state])
        FOUR_C_THROW("can not replace particle: dimensions of state '{}' do not match!",
            enum_to_state_name(state));
#endif

      // replace state in container
      for (int dim = 0; dim < statedim_[state]; ++dim)
        (states_[state])[index * statedim_[state] + dim] = states[state][dim];
    }
  }
}

void Particle::ParticleContainer::get_particle(
    int index, int& globalid, ParticleStates& states) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (index < 0 or index > (particlestored_ - 1))
    FOUR_C_THROW("can not return particle as index {} out of bounds!", index);
#endif

  // get global id from container
  globalid = globalids_[index];

  // allocate memory to hold particle states
  states.assign(statesvectorsize_, std::vector<double>(0));

  // iterate over states stored in container
  for (const auto& state : storedstates_)
  {
    // get pointer to particle state
    const double* state_ptr = &((states_[state])[index * statedim_[state]]);

    // fill particle state
    states[state].assign(state_ptr, state_ptr + statedim_[state]);
  }
}

void Particle::ParticleContainer::remove_particle(int index)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (index < 0 or index > (particlestored_ - 1))
    FOUR_C_THROW("can not remove particle as index {} out of bounds!", index);
#endif

  // decrease counter of stored particles
  --particlestored_;

  // overwrite global id in container
  globalids_[index] = globalids_[particlestored_];

  // iterate over states stored in container
  for (const auto& state : storedstates_)
  {
    // overwrite state in container
    for (int dim = 0; dim < statedim_[state]; ++dim)
      (states_[state])[index * statedim_[state] + dim] =
          (states_[state])[particlestored_ * statedim_[state] + dim];
  }
}

double Particle::ParticleContainer::get_min_value_of_state(ParticleState state) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (not storedstates_.contains(state))
    FOUR_C_THROW("particle state '{}' not stored in container!", enum_to_state_name(state));
#endif

  if (particlestored_ <= 0) return 0.0;

  double min = (states_[state])[0];

  for (int i = 0; i < (particlestored_ * statedim_[state]); ++i)
    min = std::min(min, states_[state][i]);

  return min;
}

double Particle::ParticleContainer::get_max_value_of_state(ParticleState state) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (not storedstates_.contains(state))
    FOUR_C_THROW("particle state '{}' not stored in container!", enum_to_state_name(state));
#endif

  if (particlestored_ <= 0) return 0.0;

  double max = (states_[state])[0];

  for (int i = 0; i < (particlestored_ * statedim_[state]); ++i)
    max = std::max(max, states_[state][i]);

  return max;
}

FOUR_C_NAMESPACE_CLOSE
