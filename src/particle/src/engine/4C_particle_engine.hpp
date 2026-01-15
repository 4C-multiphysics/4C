// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ENGINE_HPP
#define FOUR_C_PARTICLE_ENGINE_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_linalg_map.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class ParticleContainer;
  class ParticleContainerBundle;
  class ParticleObject;
  class ParticleRuntimeVtpWriter;
  class UniqueGlobalIdHandler;
}  // namespace Particle

namespace Core::Binstrategy
{
  class BinningStrategy;
}

namespace Core::IO
{
  class DiscretizationWriter;
  class DiscretizationReader;
}  // namespace Core::IO

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief particle engine to control particle problem
   *
   * The particle engine is responsible for the parallel distribution of particles to all
   * processors and keeps track of the dynamic change in position of all particles, including
   * communication of particles to other processors if required. In addition potential particle
   * neighbor pair relations are build.
   *
   */
  class ParticleEngine final : public ParticleEngineInterface
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] comm   communicator
     * \param[in] params particle simulation parameter list
     */
    explicit ParticleEngine(MPI_Comm comm, const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~ParticleEngine() override;

    /*!
     * \brief setup particle engine
     *
     *
     * \param[in] particlestatestotypes particle types and corresponding particle states
     */
    void setup(const std::map<ParticleType, std::set<ParticleState>>& particlestatestotypes);

    /*!
     * \brief write restart of particle engine
     *
     *
     * \param[in] step restart step
     * \param[in] time restart time
     */
    void write_restart(const int step, const double time) const;

    /*!
     * \brief read restart of particle engine
     *
     *
     * \param[in]  reader          discretization reader
     * \param[out] particlestoread particle objects read in from restart
     */
    void read_restart(const std::shared_ptr<Core::IO::DiscretizationReader> reader,
        std::vector<ParticleObjShrdPtr>& particlestoread) const;

    /*!
     * \brief write particle runtime output
     *
     *
     * \param[in] step output step
     * \param[in] time output time
     */
    void write_particle_runtime_output(const int step, const double time) const;

    /*!
     * \brief free unique global ids
     *
     * All unique global ids handed in are freed in order to be reassigned.
     *
     *
     * \param[in] freeuniquegids free unique global ids
     */
    void free_unique_global_ids(std::vector<int>& freeuniquegids) override;

    /*!
     * \brief get unique global ids for all particles
     *
     * All particles handed in do not have a unique global id yet. In this method a unique global id
     * is assigned to each particle.
     *
     *
     * \param[in] particlestogetuniquegids particles to get unique global ids
     */
    void get_unique_global_ids_for_all_particles(
        std::vector<ParticleObjShrdPtr>& particlestogetuniquegids) override;

    /*!
     * \brief check number of unique global ids
     *
     * Check the number of unique global ids to ensure that all freed global ids are reused. This
     * means the sum of all particles and of all reusable global ids should equal the number of
     * global ids.
     *
     */
    void check_number_of_unique_global_ids();

    /*!
     * \brief erase particles outside bounding box
     *
     * All particles handed in are checked if they are located inside the bounding box. Particles
     * that are outside the bounding box are erased.
     *
     *
     * \param[in] particlestocheck particles to be checked if located in bounding box
     */
    void erase_particles_outside_bounding_box(std::vector<ParticleObjShrdPtr>& particlestocheck);

    /*!
     * \brief distribute particles to owning processor
     *
     * All particle objects handed in are checked for their owning processor and send to that owning
     * processor. This method is primarily used for initialization or after redistribution of the
     * underlying bins.
     *
     *
     * \param[in] particlestodistribute particles to be distributed to their owning processor
     */
    void distribute_particles(std::vector<ParticleObjShrdPtr>& particlestodistribute);

    /*!
     * \brief transfer particles to new bins and processors
     *
     * Check all owned particle for their current position. If a particle leaves the spatial domain
     * of its currently owning processor it either left the computational domain and is removed or
     * it entered the spatial domain of a neighboring processor and is transferred to this new
     * owning processor.
     *
     */
    void transfer_particles();

    /*!
     * \brief ghost particles on other processors
     *
     */
    void ghost_particles();

    /*!
     * \brief refresh particles being ghosted on other processors
     *
     * Make use of the direct ghosting targets map and send particles that are ghosted on other
     * processors directly to those processors together with the information of the local index in
     * the respective ghosted particle container.
     *
     */
    void refresh_particles() const;

    /*!
     * \brief refresh specific states of particles of specific types
     *
     * Communicate the information of specific states of particles of specific types from owned
     * particles to processors ghosting that respective particles.
     *
     *
     * \param[in] particlestatestotypes particle types and corresponding particle states to be
     *                                  refreshed
     */
    void refresh_particles_of_specific_states_and_types(
        const StatesOfTypesToRefresh& particlestatestotypes) const override;

    /*!
     * \brief dynamic load balancing
     *
     * Dynamically balance the work load over all processors based on the total number of particles
     * on each processor. This results in a new distribution of bins and requires to rebuild all
     * dependent maps and sets.
     *
     */
    void dynamic_load_balancing();

    /*!
     * \brief hand over particles to be removed
     *
     * Hand over owned particles of any type to be removed in the respective container on this
     * processor.
     *
     * \note In case particles are removed without directly reinserting the particles, e.g., during
     *       phase change in a different type, the global ids must be handed to the particle unique
     *       global id handler beforehand in order to be reused.
     *
     *
     * \param[in] particlestoremove particles to be removed from containers on this processor
     */
    void hand_over_particles_to_be_removed(std::vector<std::set<int>>& particlestoremove) override;

    /*!
     * \brief hand over particles to be inserted
     *
     * Hand over owned particles of any type to be inserted in the respective container on this
     * processor.
     *
     * \note All particle added need a global id assigned.
     *
     *
     * \param[in] particlestoinsert particles to be inserted into containers on this processor
     */
    void hand_over_particles_to_be_inserted(
        std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestoinsert) override;

    /*!
     * \brief build particle to particle neighbors
     *
     * Build potential particle to particle neighbor pairs storing each pair only once.
     *
     */
    void build_particle_to_particle_neighbors();

    /*!
     * \brief build global id to local index map
     *
     * Relate the global id of all particles to the local index as stored in the particle container.
     *
     */
    void build_global_id_to_local_index_map();

    /*!
     * \brief check for valid particle connectivity
     *
     */
    bool have_valid_particle_connectivity() const;

    /*!
     * \brief check for valid particle neighbors
     *
     */
    bool have_valid_particle_neighbors() const;

    /*!
     * \brief get binning strategy
     *
     *
     * \return binning strategy
     */
    std::shared_ptr<Core::Binstrategy::BinningStrategy> get_binning_strategy() const
    {
      return binstrategy_;
    };

    /*!
     * \brief get bin row map
     *
     *
     * \return bin row map
     */
    std::shared_ptr<Core::LinAlg::Map> get_bin_row_map() const { return binrowmap_; };

    /*!
     * \brief get bin column map
     *
     *
     * \return bin column map
     */
    std::shared_ptr<Core::LinAlg::Map> get_bin_col_map() const { return bincolmap_; };

    ParticleContainerBundleShrdPtr get_particle_container_bundle() const override
    {
      return particlecontainerbundle_;
    };

    /*!
     * \brief get reference to relation of (owned and ghosted) particles to bins
     *
     *
     * \return relation of (owned and ghosted) particles to bins
     */
    const ParticlesToBins& get_particles_to_bins() const;

    const PotentialParticleNeighbors& get_potential_particle_neighbors() const override;

    const std::vector<std::vector<int>>& get_communicated_particle_targets() const override
    {
      return communicatedparticletargets_;
    };

    LocalIndexTupleShrdPtr get_local_index_in_specific_container(int globalid) const override;

    std::shared_ptr<Core::IO::DiscretizationWriter> get_bin_discretization_writer() const override;

    /*!
     * \brief relate all particles to all processors
     *
     * Relate the global id of all particles to the global id of the owning processor stored in a
     * vector indexed by the particle global id. In case a particle for a certain global id does not
     * exist the value is set to -1.
     *
     * \note This method is computational expensive and should be used only for initialization.
     *
     *
     * \param[out] particlestoproc relate global id of particles to global id of processor
     */
    void relate_all_particles_to_all_procs(std::vector<int>& particlestoproc) const override;

    /*!
     * \brief get particles within radius
     *
     * Get all (owned and ghosted) particles within a search radius around a search point.
     *
     * \note In case the search point is located in a ghosted bin on this processor, particles
     *       within the search radius but not owned or ghosted by this processor are not found.
     *
     *
     * \param[in]  position             position of search point
     * \param[in]  radius               search radius around search point
     * \param[out] neighboringparticles particles within search radius
     */
    void get_particles_within_radius(const double* position, const double radius,
        std::vector<LocalIndexTuple>& neighboringparticles) const override;

    std::array<double, 3> bin_size() const override;

    double min_bin_size() const override { return minbinsize_; };

    bool have_periodic_boundary_conditions() const override;

    bool have_periodic_boundary_conditions_in_spatial_direction(const int dim) const override;

    double length_of_binning_domain_in_a_spatial_direction(const int dim) const override;

    Core::LinAlg::Matrix<3, 2> const& domain_bounding_box_corner_positions() const override;

    /*!
     * \brief get distance between particles considering periodic boundaries
     *
     * Compute the distance vector between two given particles. In the case of periodic boundary
     * conditions it is checked if the distance over the periodic boundary is closer than inside the
     * domain.
     *
     *
     * \param[in]  pos_i pointer to position of particle i
     * \param[in]  pos_j pointer to position of particle j
     * \param[out] r_ji  vector from particle i to j
     */
    void distance_between_particles(
        const double* pos_i, const double* pos_j, double* r_ji) const override;

    /*!
     * \brief create binning discretization reader
     *
     *
     * \param[in] restartstep restart step
     *
     * \return discretization reader
     */
    std::shared_ptr<Core::IO::DiscretizationReader> bin_dis_reader(int restartstep) const;

    int get_number_of_particles() const override;

    int get_number_of_particles_of_specific_type(const ParticleType type) const override;

    /*!
     * \brief write binning discretization output
     *
     * \note The binning discretization output is solely meant as a debug feature as this method is
     * computational expensive!
     *
     *
     * \param[in] step output step
     * \param[in] time output time
     */
    void write_bin_dis_output(const int step, const double time) const;

   private:
    //! \name init and setup methods
    //! @{

    /*!
     * \brief init binning strategy
     *
     */
    void init_binning_strategy();

    /*!
     * \brief setup binning strategy
     *
     */
    void setup_binning_strategy();

    /*!
     * \brief setup ghosting of bins
     *
     */
    void setup_bin_ghosting();

    /*!
     * \brief setup particle container bundle
     *
     *
     * \param[in] particlestatestotypes particle types and corresponding particle states
     */
    void setup_particle_container_bundle(
        const std::map<ParticleType, std::set<ParticleState>>& particlestatestotypes) const;

    /*!
     * \brief setup data storage
     *
     *
     * \param[in] particlestatestotypes particle types and corresponding particle states
     */
    void setup_data_storage(
        const std::map<ParticleType, std::set<ParticleState>>& particlestatestotypes);

    /*!
     * \brief init particle runtime vtp writer
     *
     */
    void init_particle_vtp_writer();

    /*!
     * \brief setup particle runtime vtp writer
     *
     */
    void setup_particle_vtp_writer() const;

    /*!
     * \brief setup particle type weights for dynamic load balancing
     *
     */
    void setup_type_weights();

    //! @}

    //! \name bin distribution and ghosting
    //! @{

    /*!
     * \brief determine bin distribution dependent maps/sets
     *
     * Determine the following maps respectively sets that depend on the parallel distribution of
     * the bins to the processors:
     * - set of owned bins at an open boundary or a periodic boundary
     * - set of owned bins touched by other processors
     * - map of non-owned neighboring bins related to global id of neighboring owning processor
     *
     */
    void determine_bin_dis_dependent_maps_and_sets();

    /*!
     * \brief determine ghosting dependent maps/sets for communication
     *
     * Determine the following maps respectively sets that depend on the ghosting of the bins:
     * - set of bins being ghosted on this processor
     * - map of owned bins related to global ids of processors ghosting that bins
     *
     */
    void determine_ghosting_dependent_maps_and_sets();

    /*!
     * \brief relate half neighboring bins to owned bins
     *
     * In order to evaluate potential particle neighbor pairs all neighboring bin pairs need to be
     * processed. It is sufficient to find each particle neighbor pair once and evaluate the
     * contribution of both particles onto each other, thus it also is sufficient to relate only
     * half of the surrounding neighboring bins to an owned bin following a specific stencil. In
     * addition, all ghosted bins and also the bin itself are added to that list in order to not
     * miss any potential particle pairs.
     *
     */
    void relate_half_neighboring_bins_to_owned_bins();

    //! @}

    /*!
     * \brief check particles for periodic boundaries/leaving domain
     *
     * Check the current position of owned particles in the proximity of the boundary of the
     * computational domain. In case a particle left the computational domain it is removed from the
     * containers. In case a particle travels over a periodic boundary it is shifted by the periodic
     * length to be re-injected at the opposite site.
     *
     *
     * \param[out] particlestoremove particles to be removed from containers
     */
    void check_particles_at_boundaries(std::vector<std::set<int>>& particlestoremove) const;

    /*!
     * \brief determine particles that need to be distributed
     *
     * Find the owning processors of all particle objects handed in.
     *
     *
     * \param[in]  particlestodistribute particles to be distributed to their owning processor
     * \param[out] particlestosend       particles to be send to other processors
     * \param[out] particlestokeep       particles to be kept at this processor
     */
    void determine_particles_to_be_distributed(
        std::vector<ParticleObjShrdPtr>& particlestodistribute,
        std::vector<std::vector<ParticleObjShrdPtr>>& particlestosend,
        std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestokeep);

    /*!
     * \brief determine particles that need to be transferred
     *
     * Check the current position of owned particles close to the spatial domain of neighboring
     * processors. In case a particle left the spatial domain of its current owning processor it
     * needs to be transferred to its new owning processor. This requires that particles travel not
     * more than one bin layer as each processor only knows the owning processor of the first layer
     * of bins surrounding its own spatial domain.
     *
     *
     * \param[out] particlestoremove particles to be removed from containers on this processor
     * \param[out] particlestosend   particles to be send to other processors
     */
    void determine_particles_to_be_transferred(std::vector<std::set<int>>& particlestoremove,
        std::vector<std::vector<ParticleObjShrdPtr>>& particlestosend);

    /*!
     * \brief determine particles that need to be ghosted
     *
     * Determine all particles that need to be ghosted on other processors based on the information
     * of owned bins, that are ghosted by other processors.
     *
     *
     * \param[out] particlestosend particles to be send to other processors
     */
    void determine_particles_to_be_ghosted(
        std::vector<std::vector<ParticleObjShrdPtr>>& particlestosend) const;

    /*!
     * \brief determine particles that need to be refreshed
     *
     * Determine particles that need to be refreshed on other processors based on the direct
     * ghosting targets map.
     *
     *
     * \param[out] particlestosend particles to be send to other processors
     */
    void determine_particles_to_be_refreshed(
        std::vector<std::vector<ParticleObjShrdPtr>>& particlestosend) const;

    /*!
     * \brief determine specific states of particles of specific type that need to be refreshed
     *
     *
     * \param[in]  particlestatestotypes particle types and corresponding particle states to be
     *                                   refreshed
     * \param[out] particlestosend       particles to be send to other processors
     */
    void determine_specific_states_of_particles_of_specific_types_to_be_refreshed(
        const StatesOfTypesToRefresh& particlestatestotypes,
        std::vector<std::vector<ParticleObjShrdPtr>>& particlestosend) const;

    /*!
     * \brief communicate particles
     *
     * This method communicates particles to be send to other processors and receives particles from
     * other processors.
     *
     *
     * \param[in]  particlestosend    particles to be send to other processors
     * \param[out] particlestoreceive particles to be received on this processor
     */
    void communicate_particles(std::vector<std::vector<ParticleObjShrdPtr>>& particlestosend,
        std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestoreceive) const;

    /*!
     * \brief communicate and build map for direct ghosting
     *
     * Communicate the information at which local index in the particle container a particle at the
     * processor ghosting that particle is inserted to the owning processors.
     *
     *
     * \param[in] directghosting direct ghosting information
     */
    void communicate_direct_ghosting_map(
        std::map<int, std::map<ParticleType, std::map<int, std::pair<int, int>>>>& directghosting);

    /*!
     * \brief insert owned particles received from other processors
     *
     *
     * \param[in] particlestoinsert particles to be inserted into containers on this processor
     */
    void insert_owned_particles(
        std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestoinsert);

    /*!
     * \brief insert ghosted particles received from other processors
     *
     *
     * \param[in]  particlestoinsert particles to be inserted into containers on this processor
     * \param[out] directghosting    direct ghosting information
     */
    void insert_ghosted_particles(
        std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestoinsert,
        std::map<int, std::map<ParticleType, std::map<int, std::pair<int, int>>>>& directghosting);

    /*!
     * \brief insert refreshed particles received from other processors
     *
     *
     * \param[in] particlestoinsert particles to be inserted into containers on this processor
     */
    void insert_refreshed_particles(
        std::vector<std::vector<std::pair<int, ParticleObjShrdPtr>>>& particlestoinsert) const;

    /*!
     * \brief remove particles from containers
     *
     *
     * \param[in] particlestoremove particles to be removed from containers on this processor
     */
    void remove_particles_from_containers(std::vector<std::set<int>>& particlestoremove);

    /*!
     * \brief store particle positions after transfer of particles
     *
     */
    void store_positions_after_particle_transfer();

    /*!
     * \brief relate owned particles to bins
     *
     */
    void relate_owned_particles_to_bins();

    /*!
     * \brief determine minimum relevant bin size
     *
     */
    void determine_min_relevant_bin_size();

    /*!
     * \brief determine bin weights needed for repartitioning
     *
     */
    void determine_bin_weights();

    /*!
     * \brief invalidate particle safety flags
     *
     */
    void invalidate_particle_safety_flags();

    //! communicator
    MPI_Comm comm_;

    //! processor id
    const int myrank_;

    //! particle simulation parameter list
    const Teuchos::ParameterList& params_;

    //! binning strategy
    std::shared_ptr<Core::Binstrategy::BinningStrategy> binstrategy_;

    //! distribution of row bins
    std::shared_ptr<Core::LinAlg::Map> binrowmap_;

    //! distribution of column bins
    std::shared_ptr<Core::LinAlg::Map> bincolmap_;

    //! minimum relevant bin size
    double minbinsize_;

    //! size of vectors indexed by particle types
    int typevectorsize_;

    //! vector of bin center coordinates
    std::shared_ptr<Core::LinAlg::MultiVector<double>> bincenters_;

    //! vector of bin weights
    std::shared_ptr<Core::LinAlg::MultiVector<double>> binweights_;

    //! vector of particle type weights for dynamic load balancing
    std::vector<double> typeweights_;

    //! particle container bundle
    ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! particle unique global identifier handler
    std::unique_ptr<UniqueGlobalIdHandler> particleuniqueglobalidhandler_;

    //! particle runtime vtp writer
    std::unique_ptr<ParticleRuntimeVtpWriter> particlevtpwriter_;

    //! relate (owned and ghosted) particles to bins
    ParticlesToBins particlestobins_;

    //! relate potential particle neighbors of all types and statuses
    PotentialParticleNeighbors potentialparticleneighbors_;

    //! owned particles being communicated (transfered/distributed) to target processors
    std::vector<std::vector<int>> communicatedparticletargets_;

    //! maps particle global ids to local index in specific particle container
    std::unordered_map<int, LocalIndexTupleShrdPtr> globalidtolocalindex_;

    //! relate local index of owned particles to other processors local index of ghosted particles
    std::vector<std::map<int, std::vector<std::pair<int, int>>>> directghostingtargets_;

    //! relate half surrounding neighboring bins (including owned bin itself) to owned bins
    std::vector<std::set<int>> halfneighboringbinstobins_;

    //! flag denoting valid relation of owned particles to bins
    bool validownedparticles_;

    //! flag denoting valid relation of ghosted particles to bins
    bool validghostedparticles_;

    //! flag denoting valid relation of particle neighbors
    bool validparticleneighbors_;

    //! flag denoting validity of map relating particle global ids to local index
    bool validglobalidtolocalindex_;

    //! flag denoting validity of direct ghosting
    bool validdirectghosting_;

    //! flag denoting valid relation of half surrounding neighboring bins to owned bins
    bool validhalfneighboringbins_;

    //! owned bins at an open boundary or a periodic boundary
    std::set<int> boundarybins_;

    //! owned bins touched by other processors
    std::set<int> touchedbins_;

    //! maps bins of surrounding first layer to owning processors
    std::map<int, int> firstlayerbinsownedby_;

    //! bins being ghosted on this processor
    std::set<int> ghostedbins_;

    //! maps bins on this processor to processors ghosting that bins
    std::map<int, std::set<int>> thisbinsghostedby_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
