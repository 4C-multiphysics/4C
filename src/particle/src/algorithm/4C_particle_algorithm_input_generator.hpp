// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ALGORITHM_INPUT_GENERATOR_HPP
#define FOUR_C_PARTICLE_ALGORITHM_INPUT_GENERATOR_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_typedefs.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <mpi.h>

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
   * \brief particle input generator
   *
   */
  class InputGenerator
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] comm   communicator
     * \param[in] params particle simulation parameter list
     */
    explicit InputGenerator(MPI_Comm comm, const Teuchos::ParameterList& params);

    /*!
     * \brief generate particles
     *
     * Generate initial particles in addition to particles read in from input file. Add all
     * generated particles to the vector using the function add_generated_particle().
     *
     * \note Either generate particles only on one processor or be sure that particles are not
     *       generated twice (or even more) on different processors.
     *
     * \note There is no need to set a global id at this stage. The unique global id handler sets
     *       the global id of all particles read in from the input file and generated herein later
     *       on.
     *
     * \note Think about reserving (not resizing!) the vector storing the generated particles in
     *       advance if you have a rough estimate of the total number of particles being generated.
     *       Keep in mind that particles read in from the input files are also stored in that
     *       vector.
     *
     *
     * \param[out] particlesgenerated particle objects generated
     */
    void generate_particles(std::vector<Particle::ParticleObjShrdPtr>& particlesgenerated) const;

   protected:
    /*!
     * \brief add generated particle
     *
     *
     * \param[in]  position           position of particle
     * \param[in]  particletype       particle type enum
     * \param[out] particlesgenerated particle objects generated
     */
    void add_generated_particle(const std::vector<double>& position,
        const Particle::TypeEnum particletype,
        std::vector<Particle::ParticleObjShrdPtr>& particlesgenerated) const;

    //! processor id
    const int myrank_;

    //! particle simulation parameter list
    const Teuchos::ParameterList& params_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
