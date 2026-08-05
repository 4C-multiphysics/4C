// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_engine_container.hpp"

#include "4C_utils_exceptions.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
struct Particle::ParticleContainer::StatesImpl
{
  //! particle states in Kokkos Views indexed by particle state enum
  //! note: `states_->host_` should only be used before the dual state is initialized.
  //!        All code should use `states_->dual_` if valid otherwise use `states_->host_`.
  std::vector<Kokkos::View<double*, Kokkos::HostSpace>> host_;
  std::vector<Kokkos::DualView<double*>> dual_;
  std::vector<bool> is_dual_valid_;
};

Particle::ParticleContainer::ParticleContainer()
    : containersize_(0),
      particlestored_(0),
      statesvectorsize_(0),
      globalids_(0, -1),
      states_(std::make_unique<StatesImpl>())
{
  // empty constructor
}

Particle::ParticleContainer::~ParticleContainer()
{
  // empty (default) destructor
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
  states_->host_.resize(statesvectorsize_);
  states_->dual_.resize(statesvectorsize_);
  states_->is_dual_valid_.resize(statesvectorsize_);
  statedim_.resize(statesvectorsize_);

  // iterate over states to be stored in container
  for (const auto& state : storedstates_)
  {
    // set particle state dimension for current state
    statedim_[state] = enum_to_state_dim(state);

    // flag dual view as not currently valid
    states_->is_dual_valid_[state] = false;

    // allocate memory for current state in particle container
    Kokkos::resize(states_->host_[state], containersize_ * statedim_[state]);
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
    // resize container of current state
    if (states_->is_dual_valid_[state])
    {
      states_->dual_[state].resize(containersize_ * statedim_[state]);
      states_->host_[state] = states_->dual_[state].view_host();
    }
    else
    {
      Kokkos::resize(states_->host_[state], containersize_ * statedim_[state]);
    }
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
    // resize container of current state
    if (states_->is_dual_valid_[state])
    {
      states_->dual_[state].resize(containersize_ * statedim_[state]);
      states_->host_[state] = states_->dual_[state].view_host();
    }
    else
    {
      Kokkos::resize(states_->host_[state], containersize_ * statedim_[state]);
    }
  }
}

void Particle::ParticleContainer::add_particle(
    int& index, int globalid, const ParticleStates& states)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  // check states in container
  for (const auto& state : storedstates_)
  {
    if (state < static_cast<int>(states.size()) and not states[state].empty() and
        static_cast<int>(states[state].size()) != statedim_[state])
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
    // get pointer to particle state
    double* state_ptr = get_ptr_to_state_writable(state, index);

    // state not handed over
    if (states.size() <= state or states[state].empty())
    {
      // initialize to zero
      for (int dim = 0; dim < statedim_[state]; ++dim) state_ptr[dim] = 0.0;
    }
    // state handed over
    else
    {
      // store state in container
      for (int dim = 0; dim < statedim_[state]; ++dim) state_ptr[dim] = states[state][dim];
    }
  }
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

      // get pointer to particle state
      double* state_ptr = get_ptr_to_state_writable(state, index);

      // replace state in container
      for (int dim = 0; dim < statedim_[state]; ++dim) state_ptr[dim] = states[state][dim];
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
    const double* state_ptr = get_ptr_to_state(state, index);

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
    double* state_ptr_index = get_ptr_to_state_writable(state, index);
    double* state_ptr_last = get_ptr_to_state_writable(state, last_index);

    // overwrite state in container
    for (int dim = 0; dim < statedim_[state]; ++dim) state_ptr_index[dim] = state_ptr_last[dim];
  }

  // decrease counter of stored particles
  --particlestored_;
}

const double* Particle::ParticleContainer::get_ptr_to_state(
    ParticleState state, int index, ParticleSpace space) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (not storedstates_.contains(state))
    FOUR_C_THROW("particle state '{}' not stored in container!", enum_to_state_name(state));

  if (index < 0 or index > (particlestored_ - 1))
    FOUR_C_THROW("can not return pointer to state of particle as index {} out of bounds!", index);
#endif

  if (space == Host)
  {
    if (!states_->is_dual_valid_[state])
      return &(states_->host_[state].data()[index * statedim_[state]]);
    states_->dual_[state].sync_host();
    return &(states_->dual_[state].view_host().data()[index * statedim_[state]]);
  }
  else if (space == Device)
  {
    if (!states_->is_dual_valid_[state]) init_state_dual(state);
    states_->dual_[state].sync_device();
    return &(states_->dual_[state].view_device().data()[index * statedim_[state]]);
  }
  else
  {
    FOUR_C_THROW("unknown space requested");
  }
}

double* Particle::ParticleContainer::get_ptr_to_state_writable(
    ParticleState state, int index, ParticleSpace space)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (not storedstates_.contains(state))
    FOUR_C_THROW("particle state '{}' not stored in container!", enum_to_state_name(state));

  if (index < 0 or index > (particlestored_ - 1))
    FOUR_C_THROW("can not return pointer to state of particle as index {} out of bounds!", index);
#endif

  if (space == Host)
  {
    if (!states_->is_dual_valid_[state])
      return &(states_->host_[state].data()[index * statedim_[state]]);
    states_->dual_[state].sync_host();
    states_->dual_[state].modify_host();
    return &(states_->dual_[state].view_host().data()[index * statedim_[state]]);
  }
  else if (space == Device)
  {
    if (!states_->is_dual_valid_[state]) init_state_dual(state);
    states_->dual_[state].sync_device();
    states_->dual_[state].modify_device();
    return &(states_->dual_[state].view_device().data()[index * statedim_[state]]);
  }
  else
  {
    FOUR_C_THROW("unknown space requested");
  }
}

double Particle::ParticleContainer::get_min_value_of_state(ParticleState state) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (not storedstates_.contains(state))
    FOUR_C_THROW("particle state '{}' not stored in container!", enum_to_state_name(state));
#endif

  if (particlestored_ <= 0) return 0.0;

  const double* state_ptr = get_ptr_to_state(state, 0);
  double min = state_ptr[0];

  for (int i = 1; i < (particlestored_ * statedim_[state]); ++i) min = std::min(min, state_ptr[i]);

  return min;
}

double Particle::ParticleContainer::get_max_value_of_state(ParticleState state) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (not storedstates_.contains(state))
    FOUR_C_THROW("particle state '{}' not stored in container!", enum_to_state_name(state));
#endif

  if (particlestored_ <= 0) return 0.0;

  const double* state_ptr = get_ptr_to_state(state, 0);
  double max = state_ptr[0];

  for (int i = 1; i < (particlestored_ * statedim_[state]); ++i) max = std::max(max, state_ptr[i]);

  return max;
}

void Particle::ParticleContainer::init_state_dual(ParticleState state) const
{
  if (states_->is_dual_valid_[state]) return;
  Kokkos::View<double*> device_view = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultExecutionSpace::memory_space(), states_->host_[state]);
  states_->dual_[state] = Kokkos::DualView<double*>(device_view, states_->host_[state]);
  states_->is_dual_valid_[state] = true;
}

FOUR_C_NAMESPACE_CLOSE
