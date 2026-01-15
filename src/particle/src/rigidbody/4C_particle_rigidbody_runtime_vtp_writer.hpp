// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_RIGIDBODY_RUNTIME_VTP_WRITER_HPP
#define FOUR_C_PARTICLE_RIGIDBODY_RUNTIME_VTP_WRITER_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_io_visualization_manager.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Core::IO
{
  class DiscretizationReader;
}  // namespace Core::IO

namespace Particle
{
  class RigidBodyDataState;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief rigid body runtime vtp writer class
   *
   * A class that writes visualization output for rigid bodies in vtk/vtp format at runtime.
   *
   */
  class RigidBodyRuntimeVtpWriter final
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] comm communicator
     */
    explicit RigidBodyRuntimeVtpWriter(MPI_Comm comm);

    /*!
     * \brief init rigid body runtime vtp writer
     *
     *
     * \param[in] rigidbodydatastate rigid body data state container
     */
    void init(const std::shared_ptr<Particle::RigidBodyDataState> rigidbodydatastate);

    /*!
     * \brief read restart of runtime vtp writer
     *
     *
     * \param[in] reader discretization reader
     */
    void read_restart(const std::shared_ptr<Core::IO::DiscretizationReader> reader);

    /*!
     * \brief set positions and states of rigid bodies
     *
     * Set positions and states of rigid bodies owned by this processor.
     *
     *
     * \param[in] ownedrigidbodies owned rigid bodies by this processor
     */
    void set_rigid_body_positions_and_states(const std::vector<int>& ownedrigidbodies);

    /*!
     * \brief Write the visualization files to disk
     */
    void write_to_disk(const double time, const unsigned int timestep_number);

   private:
    //! communicator
    MPI_Comm comm_;

    //! setup time of runtime vtp writer
    double setuptime_;

    //! rigid body data state container
    std::shared_ptr<Particle::RigidBodyDataState> rigidbodydatastate_;

    //! visualization manager
    std::shared_ptr<Core::IO::VisualizationManager> visualization_manager_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
