// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ENGINE_INTERFACE_HPP
#define FOUR_C_PARTICLE_ENGINE_INTERFACE_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_particle_engine_container_bundle.hpp"

#include <Teuchos_RCPStdSharedPtrConversions.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Core::IO
{
  class DiscretizationWriter;
}

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief interface to provide restricted access to particle engine
   *
   * The particle algorithm holds an instance of the particle engine, thus having full access and
   * control over it. This abstract interface class to the particle engine provides restricted
   * access to be used in all other classes.
   *
   * \note Methods in this class are documented briefly. Refer to the full documentation of the
   *       particle engine class!
   *
   */
  class ParticleEngineInterface
  {
   public:
    //! virtual destructor
    virtual ~ParticleEngineInterface() = default;

    /*!
     * \brief free unique global ids
     *
     *
     * \param[in] freeuniquegids free unique global ids
     */
    virtual void free_unique_global_ids(std::vector<int>& freeuniquegids) = 0;

    /*!
     * \brief get unique global ids for all particles
     *
     *
     * \param[in] particlestogetuniquegids particles to get unique global ids
     */
    virtual void get_unique_global_ids_for_all_particles(
        std::vector<ParticleObjShrdPtr>& particlestogetuniquegids) = 0;

    /*!
     * \brief refresh specific states of particles of specific types
     *
     *
     * \param[in] particlestatestotypes particle types and corresponding particle states to be
     *                                  refreshed
     */
    virtual void refresh_particles_of_specific_states_and_types(
        const StatesOfTypesToRefresh& particlestatestotypes) const = 0;

    /*!
     * \brief hand over particles to be removed
     *
     *
     * \param[in] particlestoremove particles to be removed from containers on this processor
     */
    virtual void hand_over_particles_to_be_removed(
        std::vector<std::set<int>>& particlestoremove) = 0;

    /*!
     * \brief hand over particles to be inserted
     *
     *
     * \param[in] particlestoinsert particles to be inserted into containers on this processor
     */
    virtual void hand_over_particles_to_be_inserted(
        std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestoinsert) = 0;

    /*!
     * \brief get particle container bundle
     *
     *
     * \return particle container bundle
     */
    virtual ParticleContainerBundleShrdPtr get_particle_container_bundle() const = 0;

    /*!
     * \brief get reference to potential particle neighbors
     *
     *
     * \return potential particle neighbor pairs
     */
    virtual const PotentialParticleNeighbors& get_potential_particle_neighbors() const = 0;

    /*!
     * \brief get reference to particles being communicated to target processors
     *
     *
     * \return particles being communicated to target processors
     */
    virtual const std::vector<std::vector<int>>& get_communicated_particle_targets() const = 0;

    /*!
     * \brief get local index in specific particle container
     *
     *
     * \param[in] globalid global id of particle
     *
     * \return local index tuple of particle
     */
    virtual LocalIndexTupleShrdPtr get_local_index_in_specific_container(int globalid) const = 0;

    /*!
     * \brief get bin discretization writer
     *
     *
     * \return bin discretization writer
     */
    virtual std::shared_ptr<Core::IO::DiscretizationWriter> get_bin_discretization_writer()
        const = 0;

    /*!
     * \brief relate all particles to all processors
     *
     *
     * \param[out] particlestoproc relate global id of particles to global id of processor
     */
    virtual void relate_all_particles_to_all_procs(std::vector<int>& particlestoproc) const = 0;

    /*!
     * \brief get particles within radius
     *
     *
     * \param[in]  position             position of search point
     * \param[in]  radius               search radius around search point
     * \param[out] neighboringparticles particles within search radius
     */
    virtual void get_particles_within_radius(const double* position, const double radius,
        std::vector<LocalIndexTuple>& neighboringparticles) const = 0;

    //! \name get information regarding underlying bin discretization
    //! @{

    /*!
     * \brief get bin size
     *
     *
     * \return pointer to bin size
     */
    virtual std::array<double, 3> bin_size() const = 0;

    /*!
     * \brief get minimum relevant bin size
     *
     *
     * \return minimum relevant bin size
     */
    virtual double min_bin_size() const = 0;

    /*!
     * \brief get flag indicating periodic boundary conditions
     *
     *
     * \return flag indicating periodic boundary conditions
     */
    virtual bool have_periodic_boundary_conditions() const = 0;

    /*!
     * \brief get flag indicating periodic boundary conditions in spatial direction
     *
     *
     * \param[in] dim spatial direction
     *
     * \return flag indicating periodic boundary conditions in spatial direction
     */
    virtual bool have_periodic_boundary_conditions_in_spatial_direction(const int dim) const = 0;

    /*!
     * \brief get length of binning domain in a spatial direction
     *
     *
     * \param[in] dim spatial direction
     *
     * \return length of binning domain in a spatial direction
     */
    virtual double length_of_binning_domain_in_a_spatial_direction(const int dim) const = 0;

    /*!
     * \brief get bounding box dimensions
     *
     *
     * \return bounding box dimensions
     */
    virtual Core::LinAlg::Matrix<3, 2> const& domain_bounding_box_corner_positions() const = 0;

    //! @}

    /*!
     * \brief get distance between particles considering periodic boundaries
     *
     *
     * \param[in]  pos_i pointer to position of particle i
     * \param[in]  pos_j pointer to position of particle j
     * \param[out] r_ji  vector from particle i to j
     */
    virtual void distance_between_particles(
        const double* pos_i, const double* pos_j, double* r_ji) const = 0;

    /*!
     * \brief get number of particles on this processors
     *
     *
     * \return number of particles on this processors
     */
    virtual int get_number_of_particles() const = 0;

    /*!
     * \brief get number of particles on this processor of specific type
     *
     *
     * \param[in] type particle type
     *
     * \return number of particles on this processors
     */
    virtual int get_number_of_particles_of_specific_type(const ParticleType type) const = 0;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
