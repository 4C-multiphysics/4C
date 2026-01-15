// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ENGINE_RUNTIME_VTP_WRITER_HPP
#define FOUR_C_PARTICLE_ENGINE_RUNTIME_VTP_WRITER_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_io_visualization_manager.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_container_bundle.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Core::IO
{
  class DiscretizationReader;
}  // namespace Core::IO

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief particle runtime vtp writer class
   *
   * A class that writes visualization output for particles in vtk/vtu format at runtime. For each
   * particle type (and particle status) a RuntimeVtuWriter is initialized that writes the output to
   * a separate name.
   *
   */
  class ParticleRuntimeVtpWriter final
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] comm communicator
     */
    explicit ParticleRuntimeVtpWriter(MPI_Comm comm);

    /*!
     * \brief init particle runtime vtp writer
     *
     *
     * \param[in] particlecontainerbundle particle container bundle
     */
    void init(const ParticleContainerBundleShrdPtr particlecontainerbundle);

    /*!
     * \brief setup particle runtime vtp writer
     *
     *
     * \param[in] write_ghosted_particles flag for output of ghosted particles
     */
    void setup(bool write_ghosted_particles);

    /*!
     * \brief read restart of runtime vtp writer
     *
     *
     * \param[in] reader discretization reader
     */
    void read_restart(const std::shared_ptr<Core::IO::DiscretizationReader> reader);

    /*!
     * \brief set positions and states of particles
     *
     */
    void set_particle_positions_and_states();

    /*!
     * \brief Write visualization files to disk
     */
    void write_to_disk(const double visualization_time, const int visualization_step);

   private:
    //! communicator
    MPI_Comm comm_;

    //! particle container bundle
    ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! collection of vtu writer objects indexed by particle type enum and particle status enum
    std::vector<std::vector<std::shared_ptr<Core::IO::VisualizationManager>>>
        runtime_visualization_managers_;

    //! setup time of runtime vtp writer
    double setuptime_;

    /*!
     * \brief black list of particle state enums
     *
     * All particle states on the black list are ignored when writing output. The states on the
     * black list are hard coded and typically contain states not relevant for output.
     *
     * \note Reconsider the concept of a black list with hard coded particle state enums. Think
     *       about defining all particle states for the output via a list in the input file.
     *
     */
    std::set<ParticleState> blackliststates_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
