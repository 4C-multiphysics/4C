// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_RIGID_PARTICLE_CONTACT_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_RIGID_PARTICLE_CONTACT_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_input.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class InteractionWriter;
  class ParticleContainerBundle;
  class ParticleEngineInterface;
  class SPHNeighborPairs;
  class WallHandlerInterface;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class SPHRigidParticleContactBase
  {
   public:
    //! constructor
    explicit SPHRigidParticleContactBase(const Teuchos::ParameterList& params);

    //! virtual destructor
    virtual ~SPHRigidParticleContactBase() = default;

    //! setup rigid particle contact handler
    virtual void setup(
        const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface,
        const std::shared_ptr<Particle::InteractionWriter> particleinteractionwriter,
        const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs);

    //! add rigid particle contact contribution to force field
    virtual void add_force_contribution() = 0;

   private:
    //! setup particle interaction writer
    void setup_particle_interaction_writer();

   protected:
    //! smoothed particle hydrodynamics specific parameter list
    const Teuchos::ParameterList& params_sph_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! interface to particle wall handler
    std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface_;

    //! particle interaction writer
    std::shared_ptr<Particle::InteractionWriter> particleinteractionwriter_;

    //! neighbor pair handler
    std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs_;

    //! write particle-wall interaction output
    const bool writeparticlewallinteraction_;

    //! set of boundary particle types
    std::set<Particle::TypeEnum> boundarytypes_;
  };

  class SPHRigidParticleContactElastic : public SPHRigidParticleContactBase
  {
   public:
    //! constructor
    explicit SPHRigidParticleContactElastic(const Teuchos::ParameterList& params);

    //! setup rigid particle contact handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface,
        const std::shared_ptr<Particle::InteractionWriter> particleinteractionwriter,
        const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs) override;

    //! add rigid particle contact contribution to force field
    void add_force_contribution() override;

   private:
    //! elastic contact (particle contribution)
    void elastic_contact_particle_contribution();

    //! elastic contact (particle-wall contribution)
    void elastic_contact_particle_wall_contribution();

    //! contact stiffness
    const double stiff_;

    //! contact damping parameter
    const double damp_;
  };

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
