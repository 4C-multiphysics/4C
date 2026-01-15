// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ALGORITHM_TIMINT_HPP
#define FOUR_C_PARTICLE_ALGORITHM_TIMINT_HPP

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
  class DirichletBoundaryConditionHandler;
  class ParticleEngineInterface;
  class RigidBodyHandlerInterface;
  class TemperatureBoundaryConditionHandler;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief time integration base
   *
   */
  class TimInt
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] params particle simulation parameter list
     */
    explicit TimInt(const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    virtual ~TimInt();

    /*!
     * \brief time integration scheme specific initialization routine
     *
     */
    virtual void setup(
        const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::RigidBodyHandlerInterface> particlerigidbodyinterface);

    /*!
     * \brief insert integration dependent states of all particle types
     *
     *
     * \param[out] particlestatestotypes map of particle types and corresponding states
     */
    void insert_particle_states_of_particle_types(
        std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const;

    /*!
     * \brief time integration scheme specific initialization routine
     *
     */
    virtual void set_initial_states();

    /*!
     * \brief set current time
     *
     *
     * \param[in] currenttime current time
     */
    virtual void set_current_time(const double currenttime) final;

    /*!
     * \brief time integration scheme specific pre-interaction routine
     *
     */
    virtual void pre_interaction_routine() = 0;

    /*!
     * \brief time integration scheme specific post-interaction routine
     *
     */
    virtual void post_interaction_routine() = 0;

   private:
    /*!
     * \brief init dirichlet boundary condition handler
     *
     */
    void init_dirichlet_boundary_condition();

    /*!
     * \brief init temperature boundary condition handler
     *
     */
    void init_temperature_boundary_condition();

    /*!
     * \brief add initial random noise to particle position
     *
     */
    void add_initial_random_noise_to_position();

   protected:
    //! particle simulation parameter list
    const Teuchos::ParameterList& params_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! interface to rigid body handler
    std::shared_ptr<Particle::RigidBodyHandlerInterface> particlerigidbodyinterface_;

    //! dirichlet boundary condition handler
    std::unique_ptr<Particle::DirichletBoundaryConditionHandler> dirichletboundarycondition_;

    //! temperature boundary condition handler
    std::unique_ptr<Particle::TemperatureBoundaryConditionHandler> temperatureboundarycondition_;

    //! set of particle types to integrate in time
    std::set<Particle::TypeEnum> typestointegrate_;

    //! current time
    double time_;

    //! time step size
    double dt_;
  };

  /*!
   * \brief semi-implicit Euler time integration scheme
   *
   * semi-implicit Euler time integration scheme of first order accuracy (also denoted as
   * semi-explicit Euler or symplectic Euler scheme)
   *
   * \$f v_{n+1} = v_{n} + dt * a_{n}   \$f with \$f a_{n} = a( r_{n-1}, v_{n-1} ) \$f
   * \$f x_{n+1} = x_{n} + dt * v_{n+1} \$f
   *
   */
  class TimIntSemiImplicitEuler : public TimInt
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] params particle simulation parameter list
     */
    TimIntSemiImplicitEuler(const Teuchos::ParameterList& params);

    /*!
     * \brief time integration scheme specific initialization routine
     *
     */
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::RigidBodyHandlerInterface> particlerigidbodyinterface)
        override;

    /*!
     * \brief time integration scheme specific pre-interaction routine
     *
     */
    void pre_interaction_routine() override;

    /*!
     * \brief time integration scheme specific post-interaction routine
     *
     */
    void post_interaction_routine() override;
  };

  /*!
   * \brief explicit velocity Verlet time integration scheme
   *
   * explicit velocity Verlet time integration scheme of second order accuracy (also denoted as
   * leapfrog scheme in kick-drift-kick form)
   *
   * \$f v_{n+1/2} = v_{n}     + dt/2 * a_{n}     \$f with \$f a_{n}   = a( r_{n}, v_{n-1/2} )   \$f
   * \$f x_{n+1}   = x_{n}     + dt   * v_{n+1/2} \$f
   * \$f v_{n+1}   = v_{n+1/2} + dt/2 * a_{n+1}   \$f with \$f a_{n+1} = a( r_{n+1}, v_{n+1/2} ) \$f
   *
   */
  class TimIntVelocityVerlet : public TimInt
  {
   public:
    /*!
     * \brief constructor
     *
     *
     * \param[in] params particle simulation parameter list
     */
    TimIntVelocityVerlet(const Teuchos::ParameterList& params);

    /*!
     * \brief time integration scheme specific initialization routine
     *
     */
    void set_initial_states() override;

    /*!
     * \brief time integration scheme specific pre-interaction routine
     *
     */
    void pre_interaction_routine() override;

    /*!
     * \brief time integration scheme specific post-interaction routine
     *
     */
    void post_interaction_routine() override;

   private:
    //! half time step size
    double dthalf_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
