// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_dem_neighbor_pairs.hpp"

#include "4C_fem_geometry_element_coordtrafo.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_fem_geometry_searchtree_service.hpp"
#include "4C_mat_particle_wall_dem.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_particle_wall_interface.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::DEMNeighborPairs::DEMNeighborPairs()
{
  // empty constructor
}

void ParticleInteraction::DEMNeighborPairs::init()
{
  // nothing to do
}

void ParticleInteraction::DEMNeighborPairs::setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set interface to particle wall handler
  particlewallinterface_ = particlewallinterface;
}

void ParticleInteraction::DEMNeighborPairs::evaluate_neighbor_pairs()
{
  // evaluate particle pairs
  evaluate_particle_pairs();

  // evaluate particle-wall pairs
  if (particlewallinterface_) evaluate_particle_wall_pairs();
}

void ParticleInteraction::DEMNeighborPairs::evaluate_neighbor_pairs_adhesion(
    const double& adhesion_distance)
{
  // evaluate adhesion particle pairs
  evaluate_particle_pairs_adhesion(adhesion_distance);

  // evaluate adhesion particle-wall pairs
  if (particlewallinterface_) evaluate_particle_wall_pairs_adhesion(adhesion_distance);
}

void ParticleInteraction::DEMNeighborPairs::evaluate_particle_pairs()
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::DEMNeighborPairs::evaluate_particle_pairs");

  // clear particle pair data
  particlepairdata_.clear();

  // iterate over potential particle neighbors
  for (const auto& potentialneighbors :
      particleengineinterface_->get_potential_particle_neighbors())
  {
    // access values of local index tuples of particle i and j
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = potentialneighbors.first;

    PARTICLEENGINE::TypeEnum type_j;
    PARTICLEENGINE::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = potentialneighbors.second;

    // get corresponding particle containers
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    PARTICLEENGINE::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // get pointer to particle states
    const double* pos_i = container_i->get_ptr_to_state(PARTICLEENGINE::Position, particle_i);
    const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);
    const double* mass_i = container_i->get_ptr_to_state(PARTICLEENGINE::Mass, particle_i);

    const double* pos_j = container_j->get_ptr_to_state(PARTICLEENGINE::Position, particle_j);
    const double* rad_j = container_j->get_ptr_to_state(PARTICLEENGINE::Radius, particle_j);
    const double* mass_j = container_j->get_ptr_to_state(PARTICLEENGINE::Mass, particle_j);

    // vector from particle i to j
    double r_ji[3];

    // distance between particles considering periodic boundaries
    particleengineinterface_->distance_between_particles(pos_i, pos_j, r_ji);

    // absolute distance between particles
    const double absdist = Utils::vec_norm_two(r_ji);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (absdist < (1.0e-10 * rad_i[0]) or absdist < (1.0e-10 * rad_j[0]))
      FOUR_C_THROW("absolute distance {} between particles close to zero!", absdist);
#endif

    // gap between particles
    const double gap = absdist - rad_i[0] - rad_j[0];

    // neighboring particles within interaction distance
    if (gap < 0.0)
    {
      // initialize particle pair
      particlepairdata_.push_back(DEMParticlePair());

      // get reference to current particle pair
      DEMParticlePair& particlepair = particlepairdata_.back();

      // set local index tuple of particles i and j
      particlepair.tuple_i_ = potentialneighbors.first;
      particlepair.tuple_j_ = potentialneighbors.second;

      // set gap between particles
      particlepair.gap_ = gap;

      // versor from particle i to j
      Utils::vec_set_scale(particlepair.e_ji_, (1.0 / absdist), r_ji);

      // set effective mass of particles i and j
      particlepair.m_eff_ = mass_i[0] * mass_j[0] / (mass_i[0] + mass_j[0]);
    }
  }
}

void ParticleInteraction::DEMNeighborPairs::evaluate_particle_wall_pairs()
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::DEMNeighborPairs::evaluate_particle_wall_pairs");

  // clear particle-wall pair data
  particlewallpairdata_.clear();

  // relate particles to index of particle-wall pairs (considering object type of contact point)
  std::unordered_map<int, std::vector<std::pair<Core::Geo::ObjectType, int>>>
      particletoindexofparticlewallpairs;

  // index of particle-wall pairs
  int particlewallpairindex = 0;

  // iterate over potential wall neighbors
  for (const auto& potentialneighbors : particlewallinterface_->get_potential_wall_neighbors())
  {
    // access values of local index tuple of particle i
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = potentialneighbors.first;

    // get corresponding particle container
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    // get global id of particle i
    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);

    // get pointer to particle states
    const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);

    // get position of particle i
    const Core::LinAlg::Matrix<3, 1> pos_i(
        container_i->get_ptr_to_state(PARTICLEENGINE::Position, particle_i));

    // get pointer to column wall element
    Core::Elements::Element* ele = potentialneighbors.second;

    // determine nodal positions of column wall element
    std::map<int, Core::LinAlg::Matrix<3, 1>> colelenodalpos;
    particlewallinterface_->determine_col_wall_ele_nodal_pos(ele, colelenodalpos);

    // get coordinates of closest point on current column wall element to particle
    Core::LinAlg::Matrix<3, 1> closestpos;
    Core::Geo::ObjectType objecttype =
        Core::Geo::nearest_3d_object_on_element(ele, colelenodalpos, pos_i, closestpos);

    // vector from particle i to wall contact point j
    double r_ji[3];
    for (int i = 0; i < 3; i++) r_ji[i] = closestpos(i) - pos_i(i);

    // absolute distance between particle and wall contact point
    const double absdist = Utils::vec_norm_two(r_ji);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (absdist < (1.0e-10 * rad_i[0]))
      FOUR_C_THROW("absolute distance {} between particle and wall close to zero!", absdist);
#endif

    // gap between particle and wall contact point
    const double gap = absdist - rad_i[0];

    // neighboring particle and wall element within interaction distance
    if (gap < 0.0)
    {
      // initialize particle-wall pair
      particlewallpairdata_.push_back(DEMParticleWallPair());

      // get reference to current particle-wall pair
      DEMParticleWallPair& particlewallpair = particlewallpairdata_[particlewallpairindex];

      // store index of particle-wall pair
      particletoindexofparticlewallpairs[globalid_i[0]].push_back(
          std::make_pair(objecttype, particlewallpairindex));

      // increase index
      ++particlewallpairindex;

      // set local index tuple of particle i
      particlewallpair.tuple_i_ = potentialneighbors.first;

      // set pointer to column wall element
      particlewallpair.ele_ = potentialneighbors.second;

      // set gap between particle and wall contact point
      particlewallpair.gap_ = gap;

      // versor from particle i to wall contact point j
      Utils::vec_set_scale(particlewallpair.e_ji_, (1.0 / absdist), r_ji);

      // get coordinates of wall contact point in element parameter space
      Core::LinAlg::Matrix<2, 1> elecoords(Core::LinAlg::Initialization::zero);
      const Core::LinAlg::SerialDenseMatrix xyze(
          Core::Geo::get_current_nodal_positions(ele, colelenodalpos));
      Core::Geo::current_to_surface_element_coordinates(ele->shape(), xyze, closestpos, elecoords);

      // set parameter space coordinates of wall contact point
      particlewallpair.elecoords_[0] = elecoords(0, 0);
      particlewallpair.elecoords_[1] = elecoords(1, 0);
    }
  }

  // set of particle-wall pairs to remove
  std::set<int> particlewallpairstoremove;

  // iterate over particles with neighboring wall contact points
  for (auto& particleIt : particletoindexofparticlewallpairs)
  {
    // get reference to index of particle-wall pairs for current particle
    std::vector<std::pair<Core::Geo::ObjectType, int>>& indexofparticlewallpairs =
        particleIt.second;

    // only one particle-wall pair for current particle
    if (indexofparticlewallpairs.size() == 1) continue;

    // get local index tuple of current particle
    PARTICLEENGINE::LocalIndexTuple tuple_i =
        particlewallpairdata_[indexofparticlewallpairs[0].second].tuple_i_;

    // access values of local index tuple of particle i
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = tuple_i;

    // get corresponding particle container
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    // get pointer to particle states
    const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);

    // define tolerance dependent on the particle radius
    const double adaptedtol = 1.0e-7 * rad_i[0];

    // iterate over particle-wall pairs (master)
    for (std::pair<Core::Geo::ObjectType, int>& master : indexofparticlewallpairs)
    {
      // get reference to particle-wall pair (master)
      DEMParticleWallPair& masterpair = particlewallpairdata_[master.second];

      // intersection radius of particle with column wall element in wall contact point
      const double intersectionradius =
          std::sqrt(Utils::pow<2>(rad_i[0]) - Utils::pow<2>(rad_i[0] + masterpair.gap_));

      // check with other particle-wall pairs (slave)
      for (std::pair<Core::Geo::ObjectType, int>& slave : indexofparticlewallpairs)
      {
        // no-self checking
        if (master.second == slave.second) continue;

        // get reference to particle-wall pair (slave)
        DEMParticleWallPair& slavepair = particlewallpairdata_[slave.second];

        // vector between detected wall contact points
        double dist[3];
        Utils::vec_set_scale(dist, (rad_i[0] + masterpair.gap_), masterpair.e_ji_);
        Utils::vec_add_scale(dist, -(rad_i[0] + slavepair.gap_), slavepair.e_ji_);

        // absolute distance between wall contact points
        const double absdist = Utils::vec_norm_two(dist);

        bool removeslavepair = false;

        // check for coincident contact points of same type (e.g. on line between two surfaces)
        if (master.first == slave.first)
        {
          // contact point already detected (e.g. on line between two surfaces)
          if (absdist <= adaptedtol)
          {
            if (master.second < slave.second) removeslavepair = true;
          }
        }
        // check for line/node contact points within penetration volume of a surface contact point
        else if (master.first == Core::Geo::SURFACE_OBJECT)
        {
          if (absdist <= intersectionradius) removeslavepair = true;
        }
        // check for node contact points within penetration volume of a line contact point
        else if (master.first == Core::Geo::LINE_OBJECT and slave.first == Core::Geo::NODE_OBJECT)
        {
          if (absdist <= intersectionradius) removeslavepair = true;
        }

        if (removeslavepair)
        {
          // mark particle-wall pair (slave) to be removed
          particlewallpairstoremove.insert(slave.second);
          // add global id of slave wall element for interaction history
          masterpair.histeles_.insert(slavepair.ele_->id());
        }
      }
    }
  }

  // erase particle-wall pairs to be removed
  {
    int numparticlewallpairs = particlewallpairdata_.size();

    std::set<int>::reverse_iterator rit;
    for (rit = particlewallpairstoremove.rbegin(); rit != particlewallpairstoremove.rend(); ++rit)
      particlewallpairdata_[*rit] = particlewallpairdata_[--numparticlewallpairs];

    particlewallpairdata_.resize(numparticlewallpairs);
  }
}

void ParticleInteraction::DEMNeighborPairs::evaluate_particle_pairs_adhesion(
    const double& adhesion_distance)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "ParticleInteraction::DEMNeighborPairs::evaluate_particle_pairs_adhesion");

  // clear adhesion particle pair data
  particlepairadhesiondata_.clear();

  // iterate over potential particle neighbors
  for (const auto& potentialneighbors :
      particleengineinterface_->get_potential_particle_neighbors())
  {
    // access values of local index tuples of particle i and j
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = potentialneighbors.first;

    PARTICLEENGINE::TypeEnum type_j;
    PARTICLEENGINE::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = potentialneighbors.second;

    // get corresponding particle containers
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    PARTICLEENGINE::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // get pointer to particle states
    const double* pos_i = container_i->get_ptr_to_state(PARTICLEENGINE::Position, particle_i);
    const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);
    const double* mass_i = container_i->get_ptr_to_state(PARTICLEENGINE::Mass, particle_i);

    const double* pos_j = container_j->get_ptr_to_state(PARTICLEENGINE::Position, particle_j);
    const double* rad_j = container_j->get_ptr_to_state(PARTICLEENGINE::Radius, particle_j);
    const double* mass_j = container_j->get_ptr_to_state(PARTICLEENGINE::Mass, particle_j);

    // vector from particle i to j
    double r_ji[3];

    // distance between particles considering periodic boundaries
    particleengineinterface_->distance_between_particles(pos_i, pos_j, r_ji);

    // absolute distance between particles
    const double absdist = Utils::vec_norm_two(r_ji);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (absdist < (1.0e-10 * rad_i[0]) or absdist < (1.0e-10 * rad_j[0]))
      FOUR_C_THROW("absolute distance {} between particles close to zero!", absdist);
#endif

    // gap between particles
    const double gap = absdist - rad_i[0] - rad_j[0];

    // neighboring particles within adhesion distance
    if (gap < adhesion_distance)
    {
      // initialize particle pair
      particlepairadhesiondata_.push_back(DEMParticlePair());

      // get reference to current particle pair
      DEMParticlePair& particlepair = particlepairadhesiondata_.back();

      // set local index tuple of particles i and j
      particlepair.tuple_i_ = potentialneighbors.first;
      particlepair.tuple_j_ = potentialneighbors.second;

      // set gap between particles
      particlepair.gap_ = gap;

      // versor from particle i to j
      Utils::vec_set_scale(particlepair.e_ji_, (1.0 / absdist), r_ji);

      // set effective mass of particles i and j
      particlepair.m_eff_ = mass_i[0] * mass_j[0] / (mass_i[0] + mass_j[0]);
    }
  }
}

void ParticleInteraction::DEMNeighborPairs::evaluate_particle_wall_pairs_adhesion(
    const double& adhesion_distance)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "ParticleInteraction::DEMNeighborPairs::evaluate_particle_wall_pairs_adhesion");

  // clear particle-wall pair data
  particlewallpairadhesiondata_.clear();

  // relate particles to index of particle-wall pairs (considering object type of contact point)
  std::unordered_map<int, std::vector<std::pair<Core::Geo::ObjectType, int>>>
      particletoindexofparticlewallpairs;

  // index of particle-wall pairs
  int particlewallpairindex = 0;

  // iterate over potential wall neighbors
  for (const auto& potentialneighbors : particlewallinterface_->get_potential_wall_neighbors())
  {
    // access values of local index tuple of particle i
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = potentialneighbors.first;

    // get corresponding particle container
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    // get global id of particle i
    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);

    // get pointer to particle states
    const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);

    // get position of particle i
    const Core::LinAlg::Matrix<3, 1> pos_i(
        container_i->get_ptr_to_state(PARTICLEENGINE::Position, particle_i));

    // get pointer to column wall element
    Core::Elements::Element* ele = potentialneighbors.second;

    // adhesion surface energy
    double surface_energy = 0.0;

    // get material parameters of wall element
    {
      // cast material to particle wall material
      const std::shared_ptr<const Mat::ParticleWallMaterialDEM>& particlewallmaterial =
          std::dynamic_pointer_cast<const Mat::ParticleWallMaterialDEM>(ele->material());
      if (particlewallmaterial == nullptr)
        FOUR_C_THROW("cast to Mat::ParticleWallMaterialDEM failed!");

      // get adhesion surface energy
      surface_energy = particlewallmaterial->adhesion_surface_energy();
    }

    // no evaluation of adhesion contribution
    if (not(surface_energy > 0.0)) continue;

    // determine nodal positions of column wall element
    std::map<int, Core::LinAlg::Matrix<3, 1>> colelenodalpos;
    particlewallinterface_->determine_col_wall_ele_nodal_pos(ele, colelenodalpos);

    // get coordinates of closest point on current column wall element to particle
    Core::LinAlg::Matrix<3, 1> closestpos;
    Core::Geo::ObjectType objecttype =
        Core::Geo::nearest_3d_object_on_element(ele, colelenodalpos, pos_i, closestpos);

    // vector from particle i to wall contact point j
    double r_ji[3];
    for (int i = 0; i < 3; i++) r_ji[i] = closestpos(i) - pos_i(i);

    // absolute distance between particle and wall contact point
    const double absdist = Utils::vec_norm_two(r_ji);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (absdist < (1.0e-10 * rad_i[0]))
      FOUR_C_THROW("absolute distance {} between particle and wall close to zero!", absdist);
#endif

    // gap between particle and wall contact point
    const double gap = absdist - rad_i[0];

    // neighboring particle and wall element within adhesion distance
    if (gap < adhesion_distance)
    {
      // initialize particle-wall pair
      particlewallpairadhesiondata_.push_back(DEMParticleWallPair());

      // get reference to current particle-wall pair
      DEMParticleWallPair& particlewallpair = particlewallpairadhesiondata_[particlewallpairindex];

      // store index of particle-wall pair
      particletoindexofparticlewallpairs[globalid_i[0]].push_back(
          std::make_pair(objecttype, particlewallpairindex));

      // increase index
      ++particlewallpairindex;

      // set local index tuple of particle i
      particlewallpair.tuple_i_ = potentialneighbors.first;

      // set pointer to column wall element
      particlewallpair.ele_ = potentialneighbors.second;

      // set gap between particle and wall contact point
      particlewallpair.gap_ = gap;

      // versor from particle i to wall contact point j
      Utils::vec_set_scale(particlewallpair.e_ji_, (1.0 / absdist), r_ji);

      // get coordinates of wall contact point in element parameter space
      Core::LinAlg::Matrix<2, 1> elecoords(Core::LinAlg::Initialization::zero);
      const Core::LinAlg::SerialDenseMatrix xyze(
          Core::Geo::get_current_nodal_positions(ele, colelenodalpos));
      Core::Geo::current_to_surface_element_coordinates(ele->shape(), xyze, closestpos, elecoords);

      // set parameter space coordinates of wall contact point
      particlewallpair.elecoords_[0] = elecoords(0, 0);
      particlewallpair.elecoords_[1] = elecoords(1, 0);
    }
  }

  // set of particle-wall pairs to remove
  std::set<int> particlewallpairstoremove;

  // iterate over particles with neighboring wall contact points
  for (auto& particleIt : particletoindexofparticlewallpairs)
  {
    // get reference to index of particle-wall pairs for current particle
    std::vector<std::pair<Core::Geo::ObjectType, int>>& indexofparticlewallpairs =
        particleIt.second;

    // only one particle-wall pair for current particle
    if (indexofparticlewallpairs.size() == 1) continue;

    // get local index tuple of current particle
    PARTICLEENGINE::LocalIndexTuple tuple_i =
        particlewallpairadhesiondata_[indexofparticlewallpairs[0].second].tuple_i_;

    // access values of local index tuple of particle i
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = tuple_i;

    // get corresponding particle container
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    // get pointer to particle states
    const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);

    // define tolerance dependent on the particle radius
    const double adaptedtol = 1.0e-7 * rad_i[0];

    // iterate over particle-wall pairs (master)
    for (std::pair<Core::Geo::ObjectType, int>& master : indexofparticlewallpairs)
    {
      // get reference to particle-wall pair (master)
      DEMParticleWallPair& masterpair = particlewallpairadhesiondata_[master.second];

      // intersection radius of particle with column wall element in wall contact point
      const double intersectionradius = std::sqrt(
          Utils::pow<2>(rad_i[0] + adhesion_distance) - Utils::pow<2>(rad_i[0] + masterpair.gap_));

      // check with other particle-wall pairs (slave)
      for (std::pair<Core::Geo::ObjectType, int>& slave : indexofparticlewallpairs)
      {
        // no-self checking
        if (master.second == slave.second) continue;

        // get reference to particle-wall pair (slave)
        DEMParticleWallPair& slavepair = particlewallpairadhesiondata_[slave.second];

        // vector between detected wall contact points
        double dist[3];
        Utils::vec_set_scale(dist, (rad_i[0] + masterpair.gap_), masterpair.e_ji_);
        Utils::vec_add_scale(dist, -(rad_i[0] + slavepair.gap_), slavepair.e_ji_);

        // absolute distance between wall contact points
        const double absdist = Utils::vec_norm_two(dist);

        bool removeslavepair = false;

        // check for coincident contact points of same type (e.g. on line between two surfaces)
        if (master.first == slave.first)
        {
          // contact point already detected (e.g. on line between two surfaces)
          if (absdist <= adaptedtol)
          {
            if (master.second < slave.second) removeslavepair = true;
          }
        }
        // check for line/node contact points within penetration volume of a surface contact point
        else if (master.first == Core::Geo::SURFACE_OBJECT)
        {
          if (absdist <= intersectionradius) removeslavepair = true;
        }
        // check for node contact points within penetration volume of a line contact point
        else if (master.first == Core::Geo::LINE_OBJECT and slave.first == Core::Geo::NODE_OBJECT)
        {
          if (absdist <= intersectionradius) removeslavepair = true;
        }

        if (removeslavepair)
        {
          // mark particle-wall pair (slave) to be removed
          particlewallpairstoremove.insert(slave.second);
          // add global id of slave wall element for interaction history
          masterpair.histeles_.insert(slavepair.ele_->id());
        }
      }
    }
  }

  // erase particle-wall pairs to be removed
  {
    int numparticlewallpairs = particlewallpairadhesiondata_.size();

    std::set<int>::reverse_iterator rit;
    for (rit = particlewallpairstoremove.rbegin(); rit != particlewallpairstoremove.rend(); ++rit)
      particlewallpairadhesiondata_[*rit] = particlewallpairadhesiondata_[--numparticlewallpairs];

    particlewallpairadhesiondata_.resize(numparticlewallpairs);
  }
}

FOUR_C_NAMESPACE_CLOSE
