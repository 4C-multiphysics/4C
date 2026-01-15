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
    void setup(const std::map<ParticleType, std::set<ParticleState>>& particlestatestotypes);

    /*!
     * \brief get particle types of stored containers
     *
     *
     * \return reference to particle types of stored containers
     */
    inline const std::set<ParticleType>& get_particle_types() const { return storedtypes_; };

    /*!
     * \brief get specific particle container
     *
     *
     * \param[in] type   particle type
     * \param[in] status particle status
     *
     * @return pointer to particle container
     */
    inline ParticleContainer* get_specific_container(ParticleType type, ParticleStatus status) const
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (not storedtypes_.contains(type))
        FOUR_C_THROW("container for particle type '{}' not stored!", enum_to_type_name(type));
#endif

      return (containers_[type])[status].get();
    };

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
        double fac, ParticleState state, ParticleType type) const
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (not storedtypes_.contains(type))
        FOUR_C_THROW("container for particle type '{}' not stored!", enum_to_type_name(type));
#endif

      ((containers_[type])[Owned])->scale_state(fac, state);
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
    inline void update_state_specific_container(double facA, ParticleState stateA, double facB,
        ParticleState stateB, ParticleType type) const
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (not storedtypes_.contains(type))
        FOUR_C_THROW("container for particle type '{}' not stored!", enum_to_type_name(type));
#endif

      ((containers_[type])[Owned])->update_state(facA, stateA, facB, stateB);
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
        std::vector<double> val, ParticleState state, ParticleType type) const
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (not storedtypes_.contains(type))
        FOUR_C_THROW("container for particle type '{}' not stored!", enum_to_type_name(type));
#endif

      ((containers_[type])[Owned])->set_state(val, state);
    };

    /*!
     * \brief clear state of all particles in container of owned particles of specific type
     *
     *
     * \param[in] state particle state
     * \param[in] type  particle type
     */
    inline void clear_state_specific_container(ParticleState state, ParticleType type) const
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (not storedtypes_.contains(type))
        FOUR_C_THROW("container for particle type '{}' not stored!", enum_to_type_name(type));
#endif

      ((containers_[type])[Owned])->clear_state(state);
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
    inline void scale_state_all_containers(double fac, ParticleState state) const
    {
      for (const auto& type : storedtypes_) ((containers_[type])[Owned])->scale_state(fac, state);
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
        double facA, ParticleState stateA, double facB, ParticleState stateB) const
    {
      for (const auto& type : storedtypes_)
        ((containers_[type])[Owned])->update_state(facA, stateA, facB, stateB);
    };

    /*!
     * \brief set given state to all particles in container of owned particles of all types
     *
     *
     * \param[in] val   particle state value
     * \param[in] state particle state
     */
    inline void set_state_all_containers(std::vector<double> val, ParticleState state) const
    {
      for (const auto& type : storedtypes_) ((containers_[type])[Owned])->set_state(val, state);
    };

    /*!
     * \brief clear state of all particles in container of owned particles of all types
     *
     *
     * \param[in] state particle state
     */
    inline void clear_state_all_containers(ParticleState state) const
    {
      for (const auto& type : storedtypes_) ((containers_[type])[Owned])->clear_state(state);
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
        ParticleStatus status) const
    {
      for (const auto& type : storedtypes_)
        ((containers_[type])[status])->check_and_decrease_container_size();
    }

    /*!
     * \brief clear all containers of specific status
     *
     *
     * \param[in] status particle status
     */
    inline void clear_all_containers_of_specific_status(ParticleStatus status) const
    {
      for (const auto& type : storedtypes_) ((containers_[type])[status])->clear_container();
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
    std::set<ParticleType> storedtypes_;

    //! collection of particle containers indexed by particle type enum and particle status enum
    TypeStatusContainers containers_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
