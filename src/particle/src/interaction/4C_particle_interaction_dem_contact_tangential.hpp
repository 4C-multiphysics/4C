// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_DEM_CONTACT_TANGENTIAL_HPP
#define FOUR_C_PARTICLE_INTERACTION_DEM_CONTACT_TANGENTIAL_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class DEMContactTangentialBase
  {
   public:
    //! constructor
    explicit DEMContactTangentialBase(const Teuchos::ParameterList& params);

    //! virtual destructor
    virtual ~DEMContactTangentialBase() = default;

    //! init tangential contact handler
    virtual void init();

    //! setup tangential contact handler
    virtual void setup(const double& k_normal);

    //! set current step size
    virtual void set_current_step_size(const double currentstepsize) final;

    //! calculate tangential contact force
    virtual void tangential_contact_force(double* gap_tangential, bool& stick_tangential,
        const double* normal, const double* v_rel_tangential, const double& m_eff,
        const double& mu_tangential, const double& normalcontactforce,
        double* tangentialcontactforce) const = 0;

    //! evaluate tangential potential energy
    virtual void tangential_potential_energy(
        const double* gap_tangential, double& tangentialpotentialenergy) const = 0;

   protected:
    //! discrete element method parameter list
    const Teuchos::ParameterList& params_dem_;

    //! timestep size
    double dt_;
  };

  class DEMContactTangentialLinearSpringDamp : public DEMContactTangentialBase
  {
   public:
    //! constructor
    explicit DEMContactTangentialLinearSpringDamp(const Teuchos::ParameterList& params);

    //! init tangential contact handler
    void init() override;

    //! setup tangential contact handler
    void setup(const double& k_normal) override;

    //! calculate tangential contact force
    void tangential_contact_force(double* gap_tangential, bool& stick_tangential,
        const double* normal, const double* v_rel_tangential, const double& m_eff,
        const double& mu_tangential, const double& normalcontactforce,
        double* tangentialcontactforce) const override;

    //! evaluate tangential potential energy
    void tangential_potential_energy(
        const double* gap_tangential, double& tangentialpotentialenergy) const override;

   private:
    //! coefficient of restitution
    const double e_;

    //! particle Poisson ratio
    const double nue_;

    //! tangential contact stiffness
    double k_tangential_;

    //! tangential contact damping factor
    double d_tangential_fac_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
