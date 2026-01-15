// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_WALL_HPP
#define FOUR_C_PARTICLE_WALL_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_linalg_map.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_wall_interface.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class ParticleEngineInterface;
  class WallDataState;
  class WallDiscretizationRuntimeVtuWriter;
}  // namespace Particle

namespace Core::Binstrategy
{
  class BinningStrategy;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief particle wall handler for particle problem
   *
   * The particle wall handler is responsible for the parallel distribution of all wall elements to
   * all processors following the bin distribution as handed in from the particle engine. In
   * addition potential particle wall neighbor pair relations are build.
   *
   */
  class WallHandlerBase : public WallHandlerInterface
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] comm   communicator
     * \param[in] params particle simulation parameter list
     */
    explicit WallHandlerBase(MPI_Comm comm, const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~WallHandlerBase() override;

    /*!
     * \brief init wall handler
     *
     *
     * \param[in] binstrategy binning strategy
     */
    virtual void init(const std::shared_ptr<Core::Binstrategy::BinningStrategy> binstrategy);

    /*!
     * \brief setup wall handler
     *
     *
     * \param[in] particlestatestotypes particle types and corresponding particle states
     * \param[in] restart_time          restart time of the simulation
     */
    virtual void setup(
        const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        double restart_time);

    /*!
     * \brief write restart of wall handler
     *
     *
     * \param[in] step restart step
     * \param[in] time restart time
     */
    virtual void write_restart(const int step, const double time) const;

    /*!
     * \brief read restart of wall handler
     *
     *
     * \param[in] restartstep restart step
     */
    virtual void read_restart(const int restartstep);

    /*!
     * \brief insert wall handler dependent states of all particle types
     *
     *
     * \param[out] particlestatestotypes map of particle types and corresponding states
     */
    virtual void insert_particle_states_of_particle_types(
        std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes)
        const final;

    /*!
     * \brief write wall runtime output
     *
     *
     * \param[in] step output step
     * \param[in] time output time
     */
    virtual void write_wall_runtime_output(const int step, const double time) const final;

    /*!
     * \brief update bin row and column map
     *
     *
     * \param[in] binrowmap bin row map
     * \param[in] bincolmap bin column map
     */
    virtual void update_bin_row_and_col_map(const std::shared_ptr<Core::LinAlg::Map> binrowmap,
        const std::shared_ptr<Core::LinAlg::Map> bincolmap) final;

    /*!
     * \brief check that wall nodes are located in bounding box
     *
     */
    virtual void check_wall_nodes_located_in_bounding_box() const final;

    /*!
     * \brief get maximum wall position increment since last transfer
     *
     *
     * \param[out] allprocmaxpositionincrement maximum wall position increment
     */
    virtual void get_max_wall_position_increment(double& allprocmaxpositionincrement) const final;

    //! distribute wall elements and nodes
    virtual void distribute_wall_elements_and_nodes() = 0;

    //! transfer wall elements and nodes
    virtual void transfer_wall_elements_and_nodes() = 0;

    /*!
     * \brief relate bins to column wall elements
     *
     */
    virtual void relate_bins_to_col_wall_eles() final;

    /*!
     * \brief build particle to wall neighbors
     *
     * Build potential particle to wall neighbor pairs.
     *
     *
     * \param[in] particlestobins relation of (owned and ghosted) particles to bins
     */
    virtual void build_particle_to_wall_neighbors(
        const Particle::ParticlesToBins& particlestobins) final;

    /*!
     * \brief check for valid wall neighbors
     *
     */
    virtual bool have_valid_wall_neighbors() const final { return validwallneighbors_; };

    std::shared_ptr<const Core::FE::Discretization> get_wall_discretization() const final
    {
      return walldiscretization_;
    };

    std::shared_ptr<Particle::WallDataState> get_wall_data_state() const final
    {
      return walldatastate_;
    }

    const Particle::PotentialWallNeighbors& get_potential_wall_neighbors() const final;

    /*!
     * \brief determine nodal positions of column wall element
     *
     *
     * \param ele[in]             column wall element
     * \param colelenodalpos[out] current nodal position
     */
    void determine_col_wall_ele_nodal_pos(Core::Elements::Element* ele,
        std::map<int, Core::LinAlg::Matrix<3, 1>>& colelenodalpos) const final;

   private:
    //! \name init and setup methods
    //! @{

    //! init wall discretization
    virtual void init_wall_discretization() = 0;

    //! setup wall discretization
    virtual void setup_wall_discretization() const = 0;

    /*!
     * \brief init wall data state container
     *
     */
    virtual void init_wall_data_state() final;

    /*!
     * \brief create wall discretization runtime vtu writer
     *
     *
     * \note this has to be called during setup() as it requires the restart to be read already
     *
     * \param[in] restart_time   restart time of the simulation
     */
    virtual void create_wall_discretization_runtime_vtu_writer(double restart_time) final;

    //! @}

    //! relate bins to column wall elements
    Particle::BinsToColWallEles binstocolwalleles_;

    //! relate potential neighboring column wall elements to particles of all types
    Particle::PotentialWallNeighbors potentialwallneighbors_;

    //! wall discretization runtime vtu writer
    std::unique_ptr<Particle::WallDiscretizationRuntimeVtuWriter>
        walldiscretizationruntimevtuwriter_;

   protected:
    //! create wall discretization
    virtual void create_wall_discretization() final;

    //! communicator
    MPI_Comm comm_;

    //! processor id
    const int myrank_;

    //! particle simulation parameter list
    const Teuchos::ParameterList& params_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! binning strategy
    std::shared_ptr<Core::Binstrategy::BinningStrategy> binstrategy_;

    //! distribution of row bins
    std::shared_ptr<Core::LinAlg::Map> binrowmap_;

    //! distribution of col bins
    std::shared_ptr<Core::LinAlg::Map> bincolmap_;

    //! wall discretization
    std::shared_ptr<Core::FE::Discretization> walldiscretization_;

    //! wall data state container
    std::shared_ptr<Particle::WallDataState> walldatastate_;

    //! flag denoting valid relation of bins to column wall elements
    bool validwallelements_;

    //! flag denoting valid relation of wall neighbors
    bool validwallneighbors_;
  };

  /*!
   * \brief particle wall handler with wall from condition on discretization
   *
   * Particle wall handler with wall discretization generated from a surface condition on a
   * structure discretization.
   *
   */
  class WallHandlerDiscretCondition final : public WallHandlerBase
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] comm   communicator
     * \param[in] params particle simulation parameter list
     */
    explicit WallHandlerDiscretCondition(MPI_Comm comm, const Teuchos::ParameterList& params);

    /*!
     * \brief distribute wall elements and nodes
     *
     */
    void distribute_wall_elements_and_nodes() override;

    /*!
     * \brief transfer wall elements and nodes
     *
     */
    void transfer_wall_elements_and_nodes() override;

   private:
    /*!
     * \brief extend wall element ghosting
     *
     *
     * \param[in] bintorowelemap bin to row wall element distribution
     */
    void extend_wall_element_ghosting(std::map<int, std::set<int>>& bintorowelemap);

    /*!
     * \brief init wall discretization
     *
     */
    void init_wall_discretization() override;

    /*!
     * \brief setup wall discretization
     *
     */
    void setup_wall_discretization() const override;
  };

  /*!
   * \brief particle wall handler with wall from bounding box
   *
   * Particle wall handler with wall discretization generated from bounding box of binning strategy.
   *
   */
  class WallHandlerBoundingBox final : public WallHandlerBase
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] comm   communicator
     * \param[in] params particle simulation parameter list
     */
    explicit WallHandlerBoundingBox(MPI_Comm comm, const Teuchos::ParameterList& params);

    /*!
     * \brief distribute wall elements and nodes
     *
     */
    void distribute_wall_elements_and_nodes() override;

    /*!
     * \brief transfer wall elements and nodes
     *
     */
    void transfer_wall_elements_and_nodes() override;

   private:
    /*!
     * \brief init wall discretization
     *
     */
    void init_wall_discretization() override;

    /*!
     * \brief setup wall discretization
     *
     */
    void setup_wall_discretization() const override;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
