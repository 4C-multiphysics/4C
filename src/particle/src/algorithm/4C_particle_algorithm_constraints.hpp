// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ALGORITHM_CONSTRAINTS_HPP
#define FOUR_C_PARTICLE_ALGORITHM_CONSTRAINTS_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_typedefs.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN


/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*!
   * \brief constraints handler for particle simulations
   *
   */
  class ConstraintsHandler
  {
   public:
    virtual ~ConstraintsHandler() {};

    virtual void apply(Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
        const std::set<Particle::TypeEnum>& types_to_integrate, const double time) const = 0;
  };

  class ConstraintsProjection2D : public ConstraintsHandler
  {
   public:
    void apply(Particle::ParticleContainerBundleShrdPtr particle_container_bundle,
        const std::set<Particle::TypeEnum>& types_to_integrate, const double time) const override;
  };

  std::unique_ptr<Particle::ConstraintsHandler> create_constraints(
      const Teuchos::ParameterList& params);

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
