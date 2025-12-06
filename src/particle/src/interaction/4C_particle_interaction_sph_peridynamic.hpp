// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_PERIDYNAMIC_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_PERIDYNAMIC_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_io_pstream.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_interaction_sph.hpp"

#include <list>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class ParticleEngineInterface;
  class ParticleContainerBundle;
  class MaterialHandler;
}  // namespace Particle

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class SPHPeridynamic final
  {
   public:
    //! constructor
    explicit SPHPeridynamic(const Teuchos::ParameterList& params);

    //! init peridynamic handler
    void init(const std::shared_ptr<Particle::DEMNeighborPairs> neighborpairs_solid_type);

    //! setup peridynamic handler
    void setup(const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<Particle::MaterialHandler> particlematerial);

    //! insert peridynamic evaluation dependent states
    void insert_particle_states_of_particle_types(
        std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const;

    //! setup peridynamic bondlist
    void setup_peridynamic_bondlist();

    //! compute peridynamic forces
    void compute_peridynamic_forces() const;

    //! check valid peridynamic bond
    bool check_valid_peridynamic_bond_entry(
        const int localid, const int globalid, Particle::ParticleContainer* container) const;

    //! damage evaluation of peridynamic body
    void damage_evaluation();

   private:
    //! peridynamics specific parameter list
    const Teuchos::ParameterList& params_pd_;

    //! interface to particle engine
    std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    Particle::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! particle material handler
    std::shared_ptr<Particle::MaterialHandler> particlematerial_;

    //! set of peridynamic types
    std::set<Particle::TypeEnum> pdtypes_;

    //! bond list for PD bodies
    std::shared_ptr<
        std::list<std::pair<Particle::LocalGlobalIndexTuple, Particle::LocalGlobalIndexTuple>>>
        bondlist_;
  };

}  // namespace Particle
FOUR_C_NAMESPACE_CLOSE

#endif
