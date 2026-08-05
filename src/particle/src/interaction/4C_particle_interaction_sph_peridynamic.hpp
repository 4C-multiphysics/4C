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

#include <array>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  class MaterialHandler;
  class ParticleContainerBundle;
  class ParticleEngineInterface;
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
    void init(const std::shared_ptr<PDNeighborPairs> neighborpairs_pd);

    //! setup peridynamic handler
    void setup(const std::shared_ptr<ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<MaterialHandler> particlematerial);

    //! insert peridynamic evaluation dependent states
    void insert_particle_states_of_particle_types(
        std::map<Type, std::set<Particle::State>>& particlestatestotypes) const;

    //! setup peridynamic bond list
    void init_peridynamic_bondlist();

    //! compute peridynamic forces
    void add_acceleration_contribution() const;

    //! check valid peridynamic bond
    bool check_valid_peridynamic_bond_entry(
        const int localid, const int globalid, ParticleContainer* container) const;

    //! damage evaluation of peridynamic body
    void damage_evaluation();

   private:
    //! calculate peridynamic volume correction factor
    double calculate_volume_correction_factor(double xi) const;

    //! calculate force for rigid phase
    void compute_interaction_forces() const;

    //! calculate acceleration for rigid phase
    void compute_acceleration() const;

    //! interface to particle engine
    std::shared_ptr<ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! particle material handler
    std::shared_ptr<MaterialHandler> particlematerial_;

    //! neighbor pair handler for PD and for DEM like interaction
    std::shared_ptr<PDNeighborPairs> neighborpairs_pd_;

    //! bond list for PD bodies
    std::shared_ptr<std::vector<std::pair<LocalGlobalIndexTuple, LocalGlobalIndexTuple>>> bondlist_;

    //! peridynamic interaction horizon
    const double horizon_pd_;

    //! peridynamic grid spacing
    const double dx_pd_;

    //! contact stiffness
    const double stiff_;

    //! contact damping parameter
    const double damp_;

    //! peridynamic dimension
    const PeridynamicDimension peridynamic_dimension_;

    //! pre-crack line segments: each is {x1, y1, z1, x2, y2, z2}
    std::vector<std::array<double, 6>> pre_crack_lines_;

    //! pre-crack parallelogram patches: each is {p0x,p0y,p0z, p1x,p1y,p1z, p2x,p2y,p2z}
    std::vector<std::array<double, 9>> pre_crack_planes_;

    //! check if the bond from pos_i to pos_j crosses any pre-crack line segment or plane
    bool bond_crosses_pre_crack(const double* pos_i, const double* pos_j) const;
  };

}  // namespace Particle
FOUR_C_NAMESPACE_CLOSE

#endif
