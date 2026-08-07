// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ENGINE_CONTAINER_HPP
#define FOUR_C_PARTICLE_ENGINE_CONTAINER_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_typedefs.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief smart particle container class
   *
   * A particle container class that allows for dynamic adding and removing of particles while
   * avoiding both expensive memory reallocations and memory fragmentation. Fast access to particle
   * states is provided.
   *
   */
  class ParticleContainer final
  {
   public:
    //! constructor
    explicit ParticleContainer();

    //! destructor
    ~ParticleContainer();

    /*!
     * \brief setup particle container
     *
     *
     * \param[in] containersize size of particle container
     * \param[in] stateset      set of particle states to be stored
     */
    void setup(int containersize, const std::set<ParticleState>& stateset);

    //! \name manipulate container size
    //! @{

    /*!
     * \brief increase the container size
     *
     * The size of the particle container is doubled to reduce memory (re-)allocation costs.
     *
     */
    void increase_container_size();

    /*!
     * \brief decrease the container size
     *
     * The size of the particle container is halved.
     *
     */
    void decrease_container_size();

    /*!
     * \brief check and decrease the container size
     *
     */
    inline void check_and_decrease_container_size()
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (particlestored_ > containersize_)
        FOUR_C_THROW(
            "checking size of container not possible: particles stored {} > new container size {}!",
            particlestored_, containersize_);
#endif

      if (particlestored_ < 0.45 * containersize_) decrease_container_size();
    };

    //! @}

    //! \name manage particles stored in container
    //! @{

    /*!
     * \brief clear particle container
     *
     * Clear the particle container by resetting the number of particles stored in the container.
     * The container size remains untouched. The global id and the particle states are not cleared.
     *
     */
    inline void clear_container() { particlestored_ = 0; };

    /*!
     * \brief add particle to particle container and get index
     *
     * Add a particle to the particle container and return the index of the particle in the
     * container. The size of the particle container is increased if necessary. A state that is not
     * handed over is initialized to zero.
     *
     *
     * \param[out] index    index of particle in container
     * \param[in]  globalid global id of particle
     * \param[in]  states   states of particle
     */
    void add_particle(int& index, int globalid, const ParticleStates& states);

    /*!
     * \brief replace particle in particle container at given index
     *
     * Replace a particle at the given index in the particle container, meaning the global id and
     * the particle states are overwritten. A negative global id is ignored. A state that is not
     * handed over is untouched.
     *
     *
     * \param[in] index    index of particle in container
     * \param[in] globalid global id of particle
     * \param[in] states   states of particle
     */
    void replace_particle(int index, int globalid, const ParticleStates& states);

    /*!
     * \brief get particle at index from particle container
     *
     * Get a particle at the given index from the particle container.
     *
     *
     * \param[in]  index    index of particle in container
     * \param[out] globalid global id of particle
     * \param[out] states   states of particle
     */
    void get_particle(int index, int& globalid, ParticleStates& states) const;

    /*!
     * \brief remove particle from particle container
     *
     * Remove a particle at the given index from the particle container. The particle is swapped
     * with the particle at the end of the container and the number of particles stored in the
     * container is decreased.
     *
     *
     * \param[in] index index of particle in container
     */
    void remove_particle(int index);

    //! @}

    /*!
     * \brief get particle state dimension
     *
     *
     * \param[in] state particle state
     *
     * \return dimension of particle state
     */
    inline int get_state_dim(ParticleState state)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (not storedstates_.contains(state))
        FOUR_C_THROW("particle state '{}' not stored in container!", enum_to_state_name(state));
#endif

      return statedim_[static_cast<int>(state)];
    };

    //! \name access global id and particle states
    //! @{

    /*!
     * \brief get read-only pointer to state of a particle at index
     *
     * This is the default method to be used to get a pointer with read-only access to the state of
     * a particle at a certain index.
     *
     * \note Throws an error in the debug version in case the requested state is not stored in the
     *       particle container.
     *
     *
     * \param[in] state particle state
     * \param[in] index index of particle in container
     * \param[in] space memory space to access
     *
     * \return pointer with read-only access to particle state
     */
    const double* get_ptr_to_state(
        ParticleState state, int index, ParticleSpace space = ParticleSpace::Host) const;

    /*!
     * \brief conditionally get read-only pointer to state of a particle at index
     *
     * This method to get a pointer with read-only access to the state of a particle at a certain
     * index is used in cases when a state may not be stored in the particle container.
     * Conditionally, a pointer is returned in case the state is stored in the particle container,
     * otherwise, a nullptr is returned.
     *
     * \note The returned pointer may not be used to access memory without checking for a nullptr.
     *
     *
     * \param[in] state particle state
     * \param[in] index index of particle in container
     * \param[in] space memory space to access
     *
     * \return pointer with read-only access to particle state or nullptr
     */
    inline const double* try_get_ptr_to_state(
        ParticleState state, int index, ParticleSpace space = ParticleSpace::Host) const
    {
      if (storedstates_.contains(state)) return get_ptr_to_state(state, index, space);

      return nullptr;
    };

    /*!
     * \brief get writable pointer to state of a particle at index
     *
     * This is the default method to be used to get a pointer with writable access to the state of
     * a particle at a certain index.
     *
     * \note Throws an error in the debug version in case the requested state is not stored in the
     *       particle container.
     *
     *
     * \param[in] state particle state
     * \param[in] index index of particle in container
     * \param[in] space memory space to access
     *
     * \return pointer with writable access to particle state
     */
    double* get_ptr_to_state_writable(
        ParticleState state, int index, ParticleSpace space = ParticleSpace::Host);

    /*!
     * \brief conditionally get writable pointer to state of a particle at index
     *
     * This method to get a pointer with writable access to the state of a particle at a certain
     * index is used in cases when a state may not be stored in the particle container.
     * Conditionally, a pointer is returned in case the state is stored in the particle container,
     * otherwise, a nullptr is returned.
     *
     * \note The returned pointer may not be used to access memory without checking for a nullptr.
     *
     *
     * \param[in] state particle state
     * \param[in] index index of particle in container
     * \param[in] space memory space to access
     *
     * \return pointer with writable access to particle state or nullptr
     */
    inline double* try_get_ptr_to_state_writable(
        ParticleState state, int index, ParticleSpace space = ParticleSpace::Host)
    {
      if (storedstates_.contains(state)) return get_ptr_to_state_writable(state, index, space);

      return nullptr;
    };

    /*!
     * \brief get pointer to global id of a particle at index
     *
     *
     * \param[in] index index of particle in container
     *
     * \return pointer to global id
     */
    inline int* get_ptr_to_global_id(int index)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (index < 0 or index > (particlestored_ - 1))
        FOUR_C_THROW(
            "can not return pointer to global id of particle as index {} out of bounds!", index);
#endif

      return &(globalids_[index]);
    };

    //! @}

    //! \name manipulate particle states
    //! @{

    /*!
     * \brief scale state of particles
     *
     *
     * \param[in] fac   scale factor
     * \param[in] state particle state
     */
    inline void scale_state(double fac, ParticleState state)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (not storedstates_.contains(state))
        FOUR_C_THROW("particle state '{}' not stored in container!", enum_to_state_name(state));
#endif

      if (particlestored_ <= 0) return;

      double* state_ptr = get_ptr_to_state_writable(state, 0);

      for (int i = 0; i < (particlestored_ * statedim_[static_cast<int>(state)]); ++i)
        state_ptr[i] *= fac;
    };

    /*!
     * \brief add scaled states to first state of particles
     *
     *
     * \param[in] facA   first scale factor
     * \param[in] stateA first particle state
     * \param[in] facB   second scale factor
     * \param[in] stateB second particle state
     */
    inline void update_state(double facA, ParticleState stateA, double facB, ParticleState stateB)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (stateA == stateB)
        FOUR_C_THROW("adding scaled particle state '{}' to itself!", enum_to_state_name(stateA));

      if (not storedstates_.contains(stateA))
        FOUR_C_THROW("particle state '{}' not stored in container!", enum_to_state_name(stateA));

      if (not storedstates_.contains(stateB))
        FOUR_C_THROW("particle state '{}' not stored in container!", enum_to_state_name(stateB));

      if (statedim_[static_cast<int>(stateA)] != statedim_[static_cast<int>(stateB)])
        FOUR_C_THROW("dimensions of states do not match!");
#endif

      if (particlestored_ <= 0) return;

      const double* state_b_ptr = get_ptr_to_state(stateB, 0);
      double* state_a_ptr = get_ptr_to_state_writable(stateA, 0);

      for (int i = 0; i < (particlestored_ * statedim_[static_cast<int>(stateA)]); ++i)
        state_a_ptr[i] = facA * state_a_ptr[i] + facB * state_b_ptr[i];
    };

    /*!
     * \brief set given state to all particles
     *
     *
     * \param[in] val   particle state
     * \param[in] state particle state
     */
    inline void set_state(std::vector<double> val, ParticleState state)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (not storedstates_.contains(state))
        FOUR_C_THROW("particle state '{}' not stored in container!", enum_to_state_name(state));

      if (statedim_[static_cast<int>(state)] != static_cast<int>(val.size()))
        FOUR_C_THROW("dimensions of states do not match!");
#endif

      if (particlestored_ <= 0) return;

      double* state_ptr = get_ptr_to_state_writable(state, 0);

      for (int i = 0; i < particlestored_; ++i)
        for (int dim = 0; dim < statedim_[static_cast<int>(state)]; ++dim)
          state_ptr[i * statedim_[static_cast<int>(state)] + dim] = val[dim];
    };

    /*!
     * \brief clear state of all particles
     *
     *
     * \param[in] state particle state
     */
    inline void clear_state(ParticleState state)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (not storedstates_.contains(state))
        FOUR_C_THROW("particle state '{}' not stored in container!", enum_to_state_name(state));
#endif

      if (particlestored_ <= 0) return;

      double* state_ptr = get_ptr_to_state_writable(state, 0);

      for (int i = 0; i < (particlestored_ * statedim_[static_cast<int>(state)]); ++i)
        state_ptr[i] = 0.0;
    };

    //! @}

    /*!
     * \brief get stored particle states
     *
     *
     * \return stored particle states
     */
    inline const std::set<ParticleState>& get_stored_states() const { return storedstates_; };

    /*!
     * \brief get flag indicating stored state
     *
     * Get a flag that is indicating if a state is stored in the particle container.
     *
     *
     * \param[in] state particle state
     *
     * \return flag indicating stored state
     */
    inline bool have_stored_state(ParticleState state) const
    {
      return storedstates_.contains(state);
    };

    /*!
     * \brief get size of particle container
     *
     *
     * \return size of particle container
     */
    inline int container_size() const { return containersize_; };

    /*!
     * \brief get number of particles stored in container
     *
     *
     * \return number of particles stored in container
     */
    inline int particles_stored() const { return particlestored_; };

    /*!
     * \brief get minimum stored value of state in container
     *
     *
     * \param[in] state particle state
     *
     * \return minimum stored value of state in container
     */
    double get_min_value_of_state(ParticleState state) const;

    /*!
     * \brief get maximum stored value of state in container
     *
     *
     * \param[in] state particle state
     *
     * \return maximum stored value of state in container
     */
    double get_max_value_of_state(ParticleState state) const;

   private:
    //! size of particles container
    int containersize_;

    //! number of particles stored in container
    int particlestored_;

    //! set of stored particle states
    std::set<ParticleState> storedstates_;

    //! size of vector for states
    int statesvectorsize_;

    //! global ids of stored particles
    std::vector<int> globalids_;

    //! particle states private data
    struct StatesImpl;
    std::unique_ptr<StatesImpl> states_;

    //! particle state dimension indexed by particle state enum
    std::vector<int> statedim_;

    /*!
     * \brief initialize DualView for on-device computation
     *
     * \note This method is labeled as const because it does not change the _logical_ state of the
     * ParticleContainer, only the representation of the data.
     *
     *
     * \param[in] state particle state
     *
     * \return none
     */
    void init_state_dual(ParticleState state) const;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
