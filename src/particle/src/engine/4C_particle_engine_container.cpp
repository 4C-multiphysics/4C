// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_engine_container.hpp"

#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
struct Particle::ParticleContainer::StatesImpl
{
  // particle states in Kokkos Views indexed by particle state enum
  // note: `states_->host_` should only be used before the dual state is initialized.
  //        All code should use `states_->dual_` if valid otherwise use `states_->host_`.
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

Particle::ParticleContainer::~ParticleContainer() = default;

void Particle::ParticleContainer::setup(int containersize, const std::set<ParticleState>& stateset)
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
  states_->host_.resize(statesvectorsize_);
  states_->dual_.resize(statesvectorsize_);
  states_->is_dual_valid_.resize(statesvectorsize_);
  statedim_.resize(statesvectorsize_);

  // iterate over states to be stored in container
  for (const auto& state : storedstates_)
  {
    const int state_idx = static_cast<int>(state);

    // set particle state dimension for current state
    statedim_[state_idx] = enum_to_state_dim(state);

    // flag dual view as not currently valid
    states_->is_dual_valid_[state_idx] = false;

    // allocate memory for current state in particle container
    states_->host_[state_idx] = Kokkos::View<double*, Kokkos::HostSpace>(
        std::string{EnumTools::enum_name(state)}, containersize_ * statedim_[state_idx]);
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

    // resize container of current state
    if (states_->is_dual_valid_[state_idx])
    {
      states_->dual_[state_idx].resize(containersize_ * statedim_[state_idx]);
      states_->host_[state_idx] = states_->dual_[state_idx].view<Kokkos::HostSpace>();
    }
    else
    {
      Kokkos::resize(states_->host_[state_idx], containersize_ * statedim_[state_idx]);
    }
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

    // resize container of current state
    if (states_->is_dual_valid_[state_idx])
    {
      states_->dual_[state_idx].resize(containersize_ * statedim_[state_idx]);
      states_->host_[state_idx] = states_->dual_[state_idx].view<Kokkos::HostSpace>();
    }
    else
    {
      Kokkos::resize(states_->host_[state_idx], containersize_ * statedim_[state_idx]);
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
    const int state_idx = static_cast<int>(state);

    if (state_idx < static_cast<int>(states.size()) and not states[state_idx].empty() and
        static_cast<int>(states[state_idx].size()) != statedim_[state_idx])
      FOUR_C_THROW("can not add particle: dimensions of state '{}' do not match!",
          EnumTools::enum_name(state));
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
          EnumTools::enum_name(state));

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

const double* Particle::ParticleContainer::get_ptr_to_state(
    ParticleState state, int index, ParticleSpace space) const
{
  FOUR_C_ASSERT(storedstates_.contains(state), "particle state '{}' not stored in container!",
      EnumTools::enum_name(state));

  FOUR_C_ASSERT(index >= 0 and index < particlestored_,
      "can not return pointer to state of particle as index {} out of bounds!", index);

  const int state_idx = static_cast<int>(state);

  if (space == ParticleSpace::Host)
  {
    if (!states_->is_dual_valid_[state_idx])
      return &(states_->host_[state_idx].data()[index * statedim_[state_idx]]);
    states_->dual_[state_idx].sync<Kokkos::HostSpace>();
    return &(
        states_->dual_[state_idx].view<Kokkos::HostSpace>().data()[index * statedim_[state_idx]]);
  }
  else if (space == ParticleSpace::Device)
  {
    if (!states_->is_dual_valid_[state_idx]) init_state_dual(state);
    states_->dual_[state_idx].sync<Kokkos::DefaultExecutionSpace>();
    return &(states_->dual_[state_idx]
            .view<Kokkos::DefaultExecutionSpace>()
            .data()[index * statedim_[state_idx]]);
  }
  else
  {
    FOUR_C_THROW("unknown memory space requested");
  }
}

double* Particle::ParticleContainer::get_ptr_to_state_writable(
    ParticleState state, int index, ParticleSpace space)
{
  auto ptr = const_cast<double*>(
      const_cast<const ParticleContainer&>(*this).get_ptr_to_state(state, index, space));

  const int state_idx = static_cast<int>(state);

  if (states_->is_dual_valid_[state_idx])
  {
    if (space == ParticleSpace::Host)
      states_->dual_[state_idx].modify<Kokkos::HostSpace>();
    else if (space == ParticleSpace::Device)
      states_->dual_[state_idx].modify<Kokkos::DefaultExecutionSpace>();
  }
  return ptr;
}

template <class ExecutionSpace>
void scale_kernel(int size, double fac, double* ptr)
{
  Kokkos::parallel_for(
      "scale state", Kokkos::RangePolicy<ExecutionSpace>(0, size),
      KOKKOS_LAMBDA(const int i) { ptr[i] *= fac; });
}

void Particle::ParticleContainer::scale_state(
    double fac, ParticleState state, std::optional<ParticleSpace> space_option)
{
  FOUR_C_ASSERT(storedstates_.contains(state), "particle state '{}' not stored in container!",
      EnumTools::enum_name(state));

  if (particlestored_ <= 0) return;

  ParticleSpace space =
      space_option.value_or(is_sync_device(state) ? ParticleSpace::Device : ParticleSpace::Host);
  double* state_ptr = get_ptr_to_state_writable(state, 0, space);
  const int size = particlestored_ * statedim_[static_cast<int>(state)];

  if (space == ParticleSpace::Device)
    scale_kernel<Kokkos::DefaultExecutionSpace>(size, fac, state_ptr);
  else if (space == ParticleSpace::Host)
    scale_kernel<Kokkos::Serial>(size, fac, state_ptr);
}

template <class ExecutionSpace>
void update_kernel(int size, double facA, double* ptrA, double facB, const double* ptrB)
{
  Kokkos::parallel_for(
      "update state", Kokkos::RangePolicy<ExecutionSpace>(0, size),
      KOKKOS_LAMBDA(const int i) { ptrA[i] = facA * ptrA[i] + facB * ptrB[i]; });
}

void Particle::ParticleContainer::update_state(double facA, ParticleState stateA, double facB,
    ParticleState stateB, std::optional<ParticleSpace> space_option)
{
  FOUR_C_ASSERT(stateA != stateB,
      "adding scaled particle state '{}' to itself is not allowed. Use "
      "scale_state instead!",
      EnumTools::enum_name(stateA));

  FOUR_C_ASSERT(storedstates_.contains(stateA), "particle state '{}' not stored in container!",
      EnumTools::enum_name(stateA));

  FOUR_C_ASSERT(storedstates_.contains(stateB), "particle state '{}' not stored in container!",
      EnumTools::enum_name(stateB));

  FOUR_C_ASSERT(statedim_[static_cast<int>(stateA)] == statedim_[static_cast<int>(stateB)],
      "dimensions of states do not match!");

  if (particlestored_ <= 0) return;

  ParticleSpace space = space_option.value_or(is_sync_device(stateA) and is_sync_device(stateB)
                                                  ? ParticleSpace::Device
                                                  : ParticleSpace::Host);
  const double* state_b_ptr = get_ptr_to_state(stateB, 0, space);
  double* state_a_ptr = get_ptr_to_state_writable(stateA, 0, space);
  const int size = particlestored_ * statedim_[static_cast<int>(stateA)];

  if (space == ParticleSpace::Device)
    update_kernel<Kokkos::DefaultExecutionSpace>(size, facA, state_a_ptr, facB, state_b_ptr);
  else if (space == ParticleSpace::Host)
    update_kernel<Kokkos::Serial>(size, facA, state_a_ptr, facB, state_b_ptr);
}

template <class ExecutionSpace>
void set_kernel(int size, int dim, double* val, double* ptr)
{
  Kokkos::parallel_for(
      "set state", Kokkos::RangePolicy<ExecutionSpace>(0, size), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < dim; ++j) ptr[i * dim + j] = val[j];
      });
}

void Particle::ParticleContainer::set_state(
    std::vector<double> val, ParticleState state, std::optional<ParticleSpace> space_option)
{
  FOUR_C_ASSERT(storedstates_.contains(state), "particle state '{}' not stored in container!",
      EnumTools::enum_name(state));

  FOUR_C_ASSERT(statedim_[static_cast<int>(state)] == static_cast<int>(val.size()),
      "dimensions of states do not match!");

  if (particlestored_ <= 0) return;

  ParticleSpace space =
      space_option.value_or(is_sync_device(state) ? ParticleSpace::Device : ParticleSpace::Host);
  double* state_ptr = get_ptr_to_state_writable(state, 0, space);
  const int dim = statedim_[static_cast<int>(state)];

  if (space == ParticleSpace::Device)
  {
    Kokkos::View<double*> val_view("values to set", val.size());

    Kokkos::deep_copy(
        val_view, Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                      val.data(), val.size()));
    set_kernel<Kokkos::DefaultExecutionSpace>(particlestored_, dim, val_view.data(), state_ptr);
  }
  else if (space == ParticleSpace::Host)
  {
    set_kernel<Kokkos::Serial>(particlestored_, dim, val.data(), state_ptr);
  }
}

template <class ExecutionSpace>
void clear_kernel(int size, double* ptr)
{
  Kokkos::parallel_for(
      "clear state", Kokkos::RangePolicy<ExecutionSpace>(0, size),
      KOKKOS_LAMBDA(const int i) { ptr[i] = 0.0; });
}

void Particle::ParticleContainer::clear_state(
    ParticleState state, std::optional<ParticleSpace> space_option)
{
  FOUR_C_ASSERT(storedstates_.contains(state), "particle state '{}' not stored in container!",
      EnumTools::enum_name(state));

  if (particlestored_ <= 0) return;

  ParticleSpace space =
      space_option.value_or(is_sync_device(state) ? ParticleSpace::Device : ParticleSpace::Host);
  double* state_ptr = get_ptr_to_state_writable(state, 0, space);
  const int size = particlestored_ * statedim_[static_cast<int>(state)];

  if (space == ParticleSpace::Device)
    clear_kernel<Kokkos::DefaultExecutionSpace>(size, state_ptr);
  else if (space == ParticleSpace::Host)
    clear_kernel<Kokkos::Serial>(size, state_ptr);
}

double Particle::ParticleContainer::get_min_value_of_state(ParticleState state) const
{
  FOUR_C_ASSERT(storedstates_.contains(state), "particle state '{}' not stored in container!",
      EnumTools::enum_name(state));

  if (particlestored_ <= 0) return 0.0;

  const double* state_ptr = get_ptr_to_state(state, 0);
  double min = state_ptr[0];

  for (int i = 1; i < (particlestored_ * statedim_[static_cast<int>(state)]); ++i)
    min = std::min(min, state_ptr[i]);

  return min;
}

double Particle::ParticleContainer::get_max_value_of_state(ParticleState state) const
{
  FOUR_C_ASSERT(storedstates_.contains(state), "particle state '{}' not stored in container!",
      EnumTools::enum_name(state));

  if (particlestored_ <= 0) return 0.0;

  const double* state_ptr = get_ptr_to_state(state, 0);
  double max = state_ptr[0];

  for (int i = 1; i < (particlestored_ * statedim_[static_cast<int>(state)]); ++i)
    max = std::max(max, state_ptr[i]);

  return max;
}

void Particle::ParticleContainer::init_state_dual(ParticleState state) const
{
  const int state_idx = static_cast<int>(state);

  if (states_->is_dual_valid_[state_idx]) return;
  Kokkos::View<double*> device_view = Kokkos::create_mirror_view_and_copy(
      Kokkos::DefaultExecutionSpace::memory_space(), states_->host_[state_idx]);
  states_->dual_[state_idx] = Kokkos::DualView<double*>(device_view, states_->host_[state_idx]);
  states_->is_dual_valid_[state_idx] = true;
}

bool Particle::ParticleContainer::is_sync_host(ParticleState state) const
{
  const int state_idx = static_cast<int>(state);

  if (!states_->is_dual_valid_[state_idx]) return true;
  return states_->dual_[state_idx].need_sync<Kokkos::DefaultExecutionSpace>() ||
         !states_->dual_[state_idx].need_sync<Kokkos::HostSpace>();
}

bool Particle::ParticleContainer::is_sync_device(ParticleState state) const
{
  const int state_idx = static_cast<int>(state);

  if (!states_->is_dual_valid_[state_idx]) return false;
  return states_->dual_[state_idx].need_sync<Kokkos::HostSpace>() ||
         !states_->dual_[state_idx].need_sync<Kokkos::DefaultExecutionSpace>();
}

FOUR_C_NAMESPACE_CLOSE
