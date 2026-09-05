// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ENGINE_CONTAINER_BUNDLE_HPP
#define FOUR_C_PARTICLE_ENGINE_CONTAINER_BUNDLE_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_typedefs.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class ParticleObject;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief handler managing bundle of particle containers
   *
   * A handler managing the access to the bundle of particle containers. For each particle type a
   * container for owned particles and a container for ghosted particles is initialized.
   *
   */

  class ParticleContainerBundle final
  {
   public:
    //! constructor
    explicit ParticleContainerBundle();

    /*!
     * \brief setup particle container bundle
     *
     *
     * \param[in] particlestatestotypes particle types and corresponding states
     */
    void setup(const std::map<Particle::Type, std::set<Particle::State>>& particlestatestotypes);

    /*!
     * \brief get particle types of stored containers
     *
     *
     * \return reference to particle types of stored containers
     */
    inline const std::set<Particle::Type>& get_particle_types() const { return storedtypes_; };

    /*!
     * \brief get specific particle container
     *
     *
     * \param[in] type   particle type
     * \param[in] status particle status
     *
     * @return pointer to particle container
     */
    inline ParticleContainer* get_specific_container(
        Particle::Type type, Particle::Status status) const
    {
      FOUR_C_ASSERT(storedtypes_.contains(type), "container for particle type '{}' not stored!",
          enum_to_type_name(type));

      return (containers_[static_cast<int>(type)])[static_cast<int>(status)].get();
    };

    /*!
     * \brief conditionally get read-only pointer to state of a particle at index
     *
     * This array is indexed by particle type and status. If a particular combination of type,
     * status, and state is not found in the bundle, then the pointer is a nullptr.
     *
     * \note The returned pointers may not be used to access memory without checking for a nullptr.
     *
     *
     * \param[in] state         particle state
     * \param[in] types_option  particle types, optional
     * \param[in] status_option particle status, optional
     *
     * \return reference to array with pointers with read-only access to particle state
     */
    ConstParticleContainerBundleStatePtrs& try_get_ptrs_to_state(Particle::State state,
        std::optional<std::set<Particle::Type>> types_option = std::nullopt,
        std::optional<ParticleStatus> status_optional = std::nullopt) const;

    /*!
     * \brief conditionally get read-only pointer to state of a particle at index
     *
     * This array is indexed by particle type and status. If a particular combination of type,
     * status, and state is not found in the bundle, then the pointer is a nullptr.
     *
     * \note The returned pointers may not be used to access memory without checking for a nullptr.
     *
     *
     * \param[in] state  particle state
     * \param[in] status particle status
     *
     * \return reference to array with pointers with read-only access to particle state
     */
    inline ConstParticleContainerBundleStatePtrs& try_get_ptrs_to_state(
        Particle::State state, ParticleStatus status) const
    {
      return try_get_ptrs_to_state(state, std::nullopt, status);
    };

    /*!
     * \brief conditionally get writable pointer to state of a particle at index
     *
     * This array is indexed by particle type and status. If a particular combination of type,
     * status, and state is not found in the bundle, then the pointer is a nullptr.
     *
     * \note The returned pointers may not be used to access memory without checking for a nullptr.
     *
     * \note The returned pointer may not be used to access memory without checking for a nullptr.
     *
     *
     * \param[in] state         particle state
     * \param[in] types_option  particle types, optional
     * \param[in] status_option particle status, optional
     *
     * \return reference to array with pointers with writable access to particle states
     */
    ParticleContainerBundleStatePtrs& try_get_ptrs_to_state_writable(Particle::State state,
        std::optional<std::set<Particle::Type>> types_option = std::nullopt,
        std::optional<ParticleStatus> status_optional = std::nullopt);

    /*!
     * \brief conditionally get writable pointer to state of a particle at index
     *
     * This array is indexed by particle type and status. If a particular combination of type,
     * status, and state is not found in the bundle, then the pointer is a nullptr.
     *
     * \note The returned pointers may not be used to access memory without checking for a nullptr.
     *
     * \note The returned pointer may not be used to access memory without checking for a nullptr.
     *
     *
     * \param[in] state  particle state
     * \param[in] status particle status
     *
     * \return reference to array with pointers with writable access to particle states
     */
    inline ParticleContainerBundleStatePtrs& try_get_ptrs_to_state_writable(
        Particle::State state, ParticleStatus status)
    {
      return try_get_ptrs_to_state_writable(state, std::nullopt, status);
    };

    //! @}

    //! \name manipulate particle states of owned particles of specific type
    //! @{

    /*!
     * \brief scale state of particles in container of owned particles of specific type
     *
     *
     * \param[in] fac       scale factor
     * \param[in] state particle state
     * \param[in] type  particle type
     */
    inline void scale_state_specific_container(
        double fac, Particle::State state, Particle::Type type) const
    {
      FOUR_C_ASSERT(storedtypes_.contains(type), "container for particle type '{}' not stored!",
          enum_to_type_name(type));

      ((containers_[static_cast<int>(type)])[static_cast<int>(Status::Owned)])
          ->scale_state(fac, state);
    };

    /*!
     * \brief add scaled states to first state of particles in container of owned particles of
     *        specific type
     *
     *
     * \param[in] facA   first scale factor
     * \param[in] stateA first particle state
     * \param[in] facB   second scale factor
     * \param[in] stateB second particle state
     * \param[in] type   particle type
     */
    inline void update_state_specific_container(double facA, Particle::State stateA, double facB,
        Particle::State stateB, Particle::Type type) const
    {
      FOUR_C_ASSERT(storedtypes_.contains(type), "container for particle type '{}' not stored!",
          enum_to_type_name(type));

      ((containers_[static_cast<int>(type)])[static_cast<int>(Status::Owned)])
          ->update_state(facA, stateA, facB, stateB);
    };

    /*!
     * \brief set given state to all particles in container of owned particles of specific type
     *
     *
     * \param[in] val   particle state value
     * \param[in] state particle state
     * \param[in] type  particle type
     */
    inline void set_state_specific_container(
        std::vector<double> val, Particle::State state, Particle::Type type) const
    {
      FOUR_C_ASSERT(storedtypes_.contains(type), "container for particle type '{}' not stored!",
          enum_to_type_name(type));

      ((containers_[static_cast<int>(type)])[static_cast<int>(Status::Owned)])
          ->set_state(val, state);
    };

    /*!
     * \brief clear state of all particles in container of owned particles of specific type
     *
     *
     * \param[in] state particle state
     * \param[in] type  particle type
     */
    inline void clear_state_specific_container(Particle::State state, Particle::Type type) const
    {
      FOUR_C_ASSERT(storedtypes_.contains(type), "container for particle type '{}' not stored!",
          enum_to_type_name(type));

      ((containers_[static_cast<int>(type)])[static_cast<int>(Status::Owned)])->clear_state(state);
    };

    //! @}

    //! \name manipulate particle states of owned particles of all types
    //! @{

    /*!
     * \brief scale state of particles in container of owned particles of all types
     *
     *
     * \param[in] fac   scale factor
     * \param[in] state particle state
     */
    inline void scale_state_all_containers(double fac, Particle::State state) const
    {
      for (const auto& type : storedtypes_)
        ((containers_[static_cast<int>(type)])[static_cast<int>(Status::Owned)])
            ->scale_state(fac, state);
    };

    /*!
     * \brief add scaled states to first state of particles in container of owned particles of all
     *        types
     *
     *
     * \param[in] facA   first scale factor
     * \param[in] stateA first particle state
     * \param[in] facB   second scale factor
     * \param[in] stateB second particle state
     */
    inline void update_state_all_containers(
        double facA, Particle::State stateA, double facB, Particle::State stateB) const
    {
      for (const auto& type : storedtypes_)
        ((containers_[static_cast<int>(type)])[static_cast<int>(Status::Owned)])
            ->update_state(facA, stateA, facB, stateB);
    };

    /*!
     * \brief set given state to all particles in container of owned particles of all types
     *
     *
     * \param[in] val   particle state value
     * \param[in] state particle state
     */
    inline void set_state_all_containers(std::vector<double> val, Particle::State state) const
    {
      for (const auto& type : storedtypes_)
        ((containers_[static_cast<int>(type)])[static_cast<int>(Status::Owned)])
            ->set_state(val, state);
    };

    /*!
     * \brief clear state of all particles in container of owned particles of all types
     *
     *
     * \param[in] state particle state
     */
    inline void clear_state_all_containers(Particle::State state) const
    {
      for (const auto& type : storedtypes_)
        ((containers_[static_cast<int>(type)])[static_cast<int>(Status::Owned)])
            ->clear_state(state);
    };

    //! @}

    //! \name manipulate particle container of specific status
    //! @{

    /*!
     * \brief check and decrease the size of all containers of specific status
     *
     *
     * \param[in] status particle status
     */
    inline void check_and_decrease_size_all_containers_of_specific_status(
        Particle::Status status) const
    {
      for (const auto& type : storedtypes_)
        ((containers_[static_cast<int>(type)])[static_cast<int>(status)])
            ->check_and_decrease_container_size();
    }

    /*!
     * \brief clear all containers of specific status
     *
     *
     * \param[in] status particle status
     */
    inline void clear_all_containers_of_specific_status(Particle::Status status) const
    {
      for (const auto& type : storedtypes_)
        ((containers_[static_cast<int>(type)])[static_cast<int>(status)])->clear_container();
    };

    //! @}

    //! \name get particle objects of all particle containers
    //! @{

    /*!
     * \brief get packed particle objects of all containers
     *
     *
     * \param[out] particlebuffer buffer of packed particle objects of all containers
     */
    void get_packed_particle_objects_of_all_containers(std::vector<char>& particlebuffer) const;

    /*!
     * \brief get particle objects of all containers
     *
     *
     * \param[out] particlesstored particle objects of all containers
     */
    void get_vector_of_particle_objects_of_all_containers(
        std::vector<ParticleObjShrdPtr>& particlesstored) const;

    //! @}

   private:
    //! set of particle types of stored containers
    std::set<Particle::Type> storedtypes_;

    //! collection of particle containers indexed by particle type enum and particle status enum
    TypeStatusContainers containers_;

    //! arrays to hold pointers to particle states, indexed by type and status
    mutable std::array<ConstParticleContainerBundleStatePtrs,
        static_cast<int>(ParticleState::OpenBoundaryId) + 1>
        conststates_;
    mutable std::array<ParticleContainerBundleStatePtrs,
        static_cast<int>(ParticleState::OpenBoundaryId) + 1>
        states_;
  };

  /**
   *  \brief index into particle container bundle pointers
   */
  inline const double* bundle_state_ptrs_index(ConstParticleContainerBundleStatePtrs& ptrs,
      ParticleType type, ParticleStatus status, const int index, const int statedim = 1)
  {
    FOUR_C_ASSERT(ptrs[static_cast<int>(type)][static_cast<int>(status)] != nullptr,
        "Dereferencing null state pointer");

    return &ptrs[static_cast<int>(type)][static_cast<int>(status)][statedim * index];
  };

  /**
   *  \brief index into particle container bundle pointers
   */
  inline const double* bundle_state_ptrs_index(ConstParticleContainerBundleStatePtrs& ptrs,
      const double* fallback, ParticleType type, ParticleStatus status, const int index,
      const int statedim = 1)
  {
    return ptrs[static_cast<int>(type)][static_cast<int>(status)]
               ? &ptrs[static_cast<int>(type)][static_cast<int>(status)][statedim * index]
               : fallback;
  };

  /**
   *  \brief index into particle container bundle pointers
   */
  inline double* bundle_state_ptrs_index(ParticleContainerBundleStatePtrs& ptrs, ParticleType type,
      ParticleStatus status, const int index, const int statedim = 1)
  {
    FOUR_C_ASSERT(ptrs[static_cast<int>(type)][static_cast<int>(status)] != nullptr,
        "Dereferencing null state pointer");

    return &ptrs[static_cast<int>(type)][static_cast<int>(status)][statedim * index];
  };

  /**
   *  \brief index into particle container bundle pointers
   */
  inline double* bundle_state_ptrs_index(ParticleContainerBundleStatePtrs& ptrs, double* fallback,
      ParticleType type, ParticleStatus status, const int index, const int statedim = 1)
  {
    return ptrs[static_cast<int>(type)][static_cast<int>(status)]
               ? &ptrs[static_cast<int>(type)][static_cast<int>(status)][statedim * index]
               : fallback;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
