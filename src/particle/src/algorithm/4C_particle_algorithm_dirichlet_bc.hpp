// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ALGORITHM_DIRICHLET_BC_HPP
#define FOUR_C_PARTICLE_ALGORITHM_DIRICHLET_BC_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_typedefs.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class ParticleEngineInterface;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief dirichlet boundary condition handler for particle simulations
   *
   */
  class DirichletBoundaryConditionHandler
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] params particle simulation parameter list
     */
    explicit DirichletBoundaryConditionHandler(const Teuchos::ParameterList& params);

    /*!
     * \brief setup dirichlet boundary condition handler
     *
     *
     * \param[in] particleengineinterface interface to particle engine
     */
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface);

    /*!
     * \brief get reference to set of particle types subjected to dirichlet boundary conditions
     *
     *
     * \return set of particle types subjected to dirichlet boundary conditions
     */
    const std::set<Particle::TypeEnum>& get_particle_types_subjected_to_dirichlet_bc_set() const
    {
      return typessubjectedtodirichletbc_;
    };

    /*!
     * \brief insert dirichlet boundary condition dependent states of all particle types
     *
     *
     * \param[out] particlestatestotypes map of particle types and corresponding states
     */
    void insert_particle_states_of_particle_types(
        std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const;

    /*!
     * \brief set particle reference position
     *
     */
    void set_particle_reference_position() const;

    /*!
     * \brief evaluate dirichlet boundary condition
     *
     *
     * \param[in] evaltime evaluation time
     * \param[in] evalpos  flag to indicate evaluation of position
     * \param[in] evalvel  flag to indicate evaluation of velocity
     * \param[in] evalacc  flag to indicate evaluation of acceleration
     */
    void evaluate_dirichlet_boundary_condition(
        const double& evaltime, const bool evalpos, const bool evalvel, const bool evalacc) const;

   protected:
    //! particle simulation parameter list
    const Teuchos::ParameterList& params_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! relating particle types to function ids of dirichlet boundary conditions
    std::map<Particle::TypeEnum, int> dirichletbctypetofunctid_;

    //! set of particle types subjected to dirichlet boundary conditions
    std::set<Particle::TypeEnum> typessubjectedtodirichletbc_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
