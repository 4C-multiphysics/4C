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

void Particle::ParticleContainer::setup(
    int containersize, const std::set<Particle::State>& stateset)
{
  // set size of particle container (at least one)
  containersize_ = (containersize > 0) ? containersize : 1;

  // set of stored particle states
  storedstates_ = stateset;

  // determine necessary size of vector for states
  statesvectorsize_ = static_cast<int>(*(--storedstates_.end())) + 1;

  // allocate memory for global ids
  globalids_.resize(containersize_, -1);

  // allocate memory to hold particle states and dimension
  states_.resize(statesvectorsize_);
  statedim_.resize(statesvectorsize_);

  // iterate over states to be stored in container
  for (const auto& state : storedstates_)
  {
    const int state_idx = static_cast<int>(state);

    // set particle state dimension for current state
    statedim_[state_idx] = enum_to_state_dim(state);

    // allocate memory for current state in particle container
    states_[state_idx].resize(containersize_ * statedim_[state_idx]);
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
    const int state_idx = static_cast<int>(state);

    // resize vector of current state
    states_[state_idx].resize(containersize_ * statedim_[state_idx]);
  }
}

void Particle::ParticleContainer::decrease_container_size()
{
  // size of container is halved
  int newsize = static_cast<int>(0.5 * containersize_);

  // set size of particle container (at least one)
  containersize_ = (newsize > 0) ? newsize : 1;

  FOUR_C_ASSERT(particlestored_ <= containersize_,
      "decreasing size of container not possible: particles stored {} > new container size {}!",
      particlestored_, containersize_);

  // resize vector of global ids
  globalids_.resize(containersize_);

  // iterate over states stored in container
  for (const auto& state : storedstates_)
  {
    const int state_idx = static_cast<int>(state);

    // resize vector of current state
    states_[state_idx].resize(containersize_ * statedim_[state_idx]);
  }
}

void Particle::ParticleContainer::add_particle(
    int& index, int globalid, const ParticleStates& states)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check states in container
  for (const auto& state : storedstates_)
  {
    const int state_idx = static_cast<int>(state);

    if (state_idx < static_cast<int>(states.size()) and not states[state_idx].empty() and
        static_cast<int>(states[state_idx].size()) != statedim_[state_idx])
      FOUR_C_THROW("can not add particle: dimensions of state '{}' do not match!",
          enum_to_state_name(state));
  }
#endif

  // increase size of container
  if (particlestored_ == containersize_) increase_container_size();

  // store local index before incrementing
  index = particlestored_;

  // increase counter of stored particles
  particlestored_++;

  // store global id
  globalids_[index] = globalid;

  // iterate over states stored in container
  for (const auto& state : storedstates_)
  {
    const int state_idx = static_cast<int>(state);

    // get pointer to particle state
    double* state_ptr = get_ptr_to_state_writable(state, index);

    // state not handed over
    if (static_cast<int>(states.size()) <= state_idx or states[state_idx].empty())
    {
      // initialize to zero
      for (int dim = 0; dim < statedim_[state_idx]; ++dim) state_ptr[dim] = 0.0;
    }
    // state handed over
    else
    {
      // store state in container
      for (int dim = 0; dim < statedim_[state_idx]; ++dim) state_ptr[dim] = states[state_idx][dim];
    }
  }
}

void Particle::ParticleContainer::replace_particle(
    int index, int globalid, const ParticleStates& states)
{
  FOUR_C_ASSERT(index >= 0 and index < particlestored_,
      "can not replace particle as index {} out of bounds!", index);

  // replace global id in container
  if (globalid >= 0) globalids_[index] = globalid;

  // iterate over states stored in container
  for (const auto& state : storedstates_)
  {
    const int state_idx = static_cast<int>(state);

    // state not handed over
    if (static_cast<int>(states.size()) <= state_idx or states[state_idx].empty())
    {
      // leave state untouched
    }
    // state handed over
    else
    {
      FOUR_C_ASSERT(static_cast<int>(states[state_idx].size()) == statedim_[state_idx],
          "can not replace particle: dimensions of state '{}' do not match!",
          enum_to_state_name(state));

      // get pointer to particle state
      double* state_ptr = get_ptr_to_state_writable(state, index);

      // replace state in container
      for (int dim = 0; dim < statedim_[state_idx]; ++dim) state_ptr[dim] = states[state_idx][dim];
    }
  }
}

void Particle::ParticleContainer::get_particle(
    int index, int& globalid, ParticleStates& states) const
{
  FOUR_C_ASSERT(index >= 0 and index < particlestored_,
      "can not return particle as index {} out of bounds!", index);

  // get global id from container
  globalid = globalids_[index];

  // allocate memory to hold particle states
  states.assign(statesvectorsize_, std::vector<double>(0));

  // iterate over states stored in container
  for (const auto& state : storedstates_)
  {
    // get pointer to particle state
    const double* state_ptr = get_ptr_to_state(state, index);

    // fill particle state
    const int state_idx = static_cast<int>(state);
    states[state_idx].assign(state_ptr, state_ptr + statedim_[state_idx]);
  }
}

void Particle::ParticleContainer::remove_particle(int index)
{
  FOUR_C_ASSERT(index >= 0 and index < particlestored_,
      "can not remove particle as index {} out of bounds!", index);

  // index of last particle
  auto last_index = particlestored_ - 1;

  if (index == last_index)
  {
    --particlestored_;
    return;
  }

  // overwrite global id in container
  globalids_[index] = globalids_[last_index];

  // iterate over states stored in container
  for (const auto& state : storedstates_)
  {
    // get pointers to particle state
    // note - this is the same underlying array, so using same access pattern
    double* state_ptr_index = get_ptr_to_state_writable(state, index);
    double* state_ptr_last = get_ptr_to_state_writable(state, last_index);

    for (int dim = 0; dim < statedim_[static_cast<int>(state)]; ++dim)
      state_ptr_index[dim] = state_ptr_last[dim];
  }

  // decrease counter of stored particles
  --particlestored_;
}

const double* Particle::ParticleContainer::get_ptr_to_state(Particle::State state, int index) const
{
  FOUR_C_ASSERT(storedstates_.contains(state), "particle state '{}' not stored in container!",
      enum_to_state_name(state));

  FOUR_C_ASSERT(index >= 0 and index < particlestored_,
      "can not return pointer to state of particle as index {} out of bounds!", index);

  const int state_idx = static_cast<int>(state);

  return &((states_[state_idx])[index * statedim_[state_idx]]);
};

double* Particle::ParticleContainer::get_ptr_to_state_writable(Particle::State state, int index)
{
  return const_cast<double*>(
      const_cast<const ParticleContainer&>(*this).get_ptr_to_state(state, index));
};

double Particle::ParticleContainer::get_min_value_of_state(Particle::State state) const
{
  FOUR_C_ASSERT(storedstates_.contains(state), "particle state '{}' not stored in container!",
      enum_to_state_name(state));

  if (particlestored_ <= 0) return 0.0;

  const double* state_ptr = get_ptr_to_state(state);
  double min = state_ptr[0];

  for (int i = 1; i < (particlestored_ * statedim_[static_cast<int>(state)]); ++i)
    min = std::min(min, state_ptr[i]);

  return min;
}

double Particle::ParticleContainer::get_max_value_of_state(Particle::State state) const
{
  FOUR_C_ASSERT(storedstates_.contains(state), "particle state '{}' not stored in container!",
      enum_to_state_name(state));

  if (particlestored_ <= 0) return 0.0;

  const double* state_ptr = get_ptr_to_state(state);
  double max = state_ptr[0];

  for (int i = 1; i < (particlestored_ * statedim_[static_cast<int>(state)]); ++i)
    max = std::max(max, state_ptr[i]);

  return max;
}

FOUR_C_NAMESPACE_CLOSE
