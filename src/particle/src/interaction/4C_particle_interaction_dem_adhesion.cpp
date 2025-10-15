// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_dem_adhesion.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_io_visualization_manager.hpp"
#include "4C_mat_particle_wall_dem.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_dem_adhesion_law.hpp"
#include "4C_particle_interaction_dem_adhesion_surface_energy.hpp"
#include "4C_particle_interaction_dem_history_pairs.hpp"
#include "4C_particle_interaction_dem_neighbor_pairs.hpp"
#include "4C_particle_interaction_runtime_writer.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_particle_wall_datastate.hpp"
#include "4C_particle_wall_interface.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::DEMAdhesion::DEMAdhesion(const Teuchos::ParameterList& params)
    : params_dem_(params),
      adhesion_distance_(params_dem_.get<double>("ADHESION_DISTANCE")),
      writeparticlewallinteraction_(params_dem_.get<bool>("WRITE_PARTICLE_WALL_INTERACTION"))
{
  // empty constructor
}

ParticleInteraction::DEMAdhesion::~DEMAdhesion() = default;

void ParticleInteraction::DEMAdhesion::init()
{
  // init adhesion law handler
  init_adhesion_law_handler();

  // init adhesion surface energy handler
  init_adhesion_surface_energy_handler();
}

void ParticleInteraction::DEMAdhesion::setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface,
    const std::shared_ptr<ParticleInteraction::InteractionWriter> particleinteractionwriter,
    const std::shared_ptr<ParticleInteraction::DEMNeighborPairs> neighborpairs,
    const std::shared_ptr<ParticleInteraction::DEMHistoryPairs> historypairs,
    const double& k_normal)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set interface to particle wall handler
  particlewallinterface_ = particlewallinterface;

  // set particle interaction writer
  particleinteractionwriter_ = particleinteractionwriter;

  // setup particle interaction writer
  setup_particle_interaction_writer();

  // set neighbor pair handler
  neighborpairs_ = neighborpairs;

  // set history pair handler
  historypairs_ = historypairs;

  // setup adhesion law handler
  adhesionlaw_->setup(k_normal);

  // setup adhesion surface energy handler
  adhesionsurfaceenergy_->setup();

  // safety check
  if (adhesion_distance_ < 0.0) FOUR_C_THROW("negative adhesion distance!");
}

void ParticleInteraction::DEMAdhesion::add_force_contribution()
{
  // evaluate particle adhesion contribution
  evaluate_particle_adhesion();

  // evaluate particle-wall adhesion contribution
  if (particlewallinterface_) evaluate_particle_wall_adhesion();
}

void ParticleInteraction::DEMAdhesion::init_adhesion_law_handler()
{
  // get type of adhesion law
  auto adhesionlaw = Teuchos::getIntegralValue<PARTICLE::AdhesionLaw>(params_dem_, "ADHESIONLAW");

  // create adhesion law handler
  switch (adhesionlaw)
  {
    case PARTICLE::AdhesionVdWDMT:
    {
      adhesionlaw_ = std::unique_ptr<ParticleInteraction::DEMAdhesionLawVdWDMT>(
          new ParticleInteraction::DEMAdhesionLawVdWDMT(params_dem_));
      break;
    }
    case PARTICLE::AdhesionRegDMT:
    {
      adhesionlaw_ = std::unique_ptr<ParticleInteraction::DEMAdhesionLawRegDMT>(
          new ParticleInteraction::DEMAdhesionLawRegDMT(params_dem_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown adhesion law type!");
      break;
    }
  }

  // init adhesion law handler
  adhesionlaw_->init();
}

void ParticleInteraction::DEMAdhesion::init_adhesion_surface_energy_handler()
{
  // get type of adhesion surface energy distribution
  auto surfaceenergydistributiontype =
      Teuchos::getIntegralValue<PARTICLE::SurfaceEnergyDistribution>(
          params_dem_, "ADHESION_SURFACE_ENERGY_DISTRIBUTION");

  // create adhesion surface energy handler
  switch (surfaceenergydistributiontype)
  {
    case PARTICLE::ConstantSurfaceEnergy:
    {
      adhesionsurfaceenergy_ =
          std::unique_ptr<ParticleInteraction::DEMAdhesionSurfaceEnergyConstant>(
              new ParticleInteraction::DEMAdhesionSurfaceEnergyConstant(params_dem_));
      break;
    }
    case PARTICLE::NormalSurfaceEnergyDistribution:
    {
      adhesionsurfaceenergy_ =
          std::unique_ptr<ParticleInteraction::DEMAdhesionSurfaceEnergyDistributionNormal>(
              new ParticleInteraction::DEMAdhesionSurfaceEnergyDistributionNormal(params_dem_));
      break;
    }
    case PARTICLE::LogNormalSurfaceEnergyDistribution:
    {
      adhesionsurfaceenergy_ =
          std::unique_ptr<ParticleInteraction::DEMAdhesionSurfaceEnergyDistributionLogNormal>(
              new ParticleInteraction::DEMAdhesionSurfaceEnergyDistributionLogNormal(params_dem_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown adhesion surface energy distribution type!");
      break;
    }
  }

  // init adhesion surface energy handler
  adhesionsurfaceenergy_->init();
}

void ParticleInteraction::DEMAdhesion::setup_particle_interaction_writer()
{
  // register specific runtime output writer
  if (writeparticlewallinteraction_)
    particleinteractionwriter_->register_specific_runtime_output_writer("particle-wall-adhesion");
}

void ParticleInteraction::DEMAdhesion::evaluate_particle_adhesion()
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::DEMAdhesion::evaluate_particle_adhesion");

  // get reference to particle adhesion history pair data
  DEMHistoryPairAdhesionData& adhesionhistorydata =
      historypairs_->get_ref_to_particle_adhesion_history_data();

  // adhesion surface energy
  const double surface_energy = params_dem_.get<double>("ADHESION_SURFACE_ENERGY");

  // iterate over particle pairs
  for (const auto& particlepair : neighborpairs_->get_ref_to_particle_pair_adhesion_data())
  {
    // access values of local index tuples of particle i and j
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    PARTICLEENGINE::TypeEnum type_j;
    PARTICLEENGINE::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // get corresponding particle containers
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    PARTICLEENGINE::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // get global ids of particle
    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);
    const int* globalid_j = container_j->get_ptr_to_global_id(particle_j);

    // get pointer to particle states
    const double* vel_i = container_i->get_ptr_to_state(PARTICLEENGINE::Velocity, particle_i);
    const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);
    double* force_i = container_i->get_ptr_to_state(PARTICLEENGINE::Force, particle_i);

    const double* vel_j = container_j->get_ptr_to_state(PARTICLEENGINE::Velocity, particle_j);
    const double* rad_j = container_j->get_ptr_to_state(PARTICLEENGINE::Radius, particle_j);
    double* force_j = container_j->get_ptr_to_state(PARTICLEENGINE::Force, particle_j);

    // relative velocity in contact point c between particle i and j (neglecting angular velocity)
    double vel_rel[3];
    Utils::vec_set(vel_rel, vel_i);
    Utils::vec_sub(vel_rel, vel_j);

    // magnitude of relative velocity in normal direction
    const double vel_rel_normal = Utils::vec_dot(vel_rel, particlepair.e_ji_);

    // calculate effective radius
    const double r_eff = (rad_i[0] * rad_j[0]) / (rad_i[0] + rad_j[0]);

    // get reference to touched adhesion history
    TouchedDEMHistoryPairAdhesion& touchedadhesionhistory_ij =
        adhesionhistorydata[globalid_i[0]][globalid_j[0]];

    // mark adhesion history as touched
    touchedadhesionhistory_ij.first = true;

    // get reference to adhesion history
    DEMHistoryPairAdhesion& adhesionhistory_ij = touchedadhesionhistory_ij.second;

    // calculate adhesion surface energy
    if (not(adhesionhistory_ij.surface_energy_ > 0.0))
      adhesionsurfaceenergy_->adhesion_surface_energy(
          surface_energy, adhesionhistory_ij.surface_energy_);

    // calculate adhesion force
    adhesionlaw_->adhesion_force(particlepair.gap_, adhesionhistory_ij.surface_energy_, r_eff,
        vel_rel_normal, particlepair.m_eff_, adhesionhistory_ij.adhesion_force_);

    // copy history from interaction pair ij to ji
    if (status_j == PARTICLEENGINE::Owned)
    {
      // get reference to touched adhesion history
      TouchedDEMHistoryPairAdhesion& touchedadhesionhistory_ji =
          adhesionhistorydata[globalid_j[0]][globalid_i[0]];

      // mark adhesion history as touched
      touchedadhesionhistory_ji.first = true;

      // get reference to adhesion history
      DEMHistoryPairAdhesion& adhesionhistory_ji = touchedadhesionhistory_ji.second;

      // set adhesion surface energy and adhesion force
      adhesionhistory_ji.surface_energy_ = adhesionhistory_ij.surface_energy_;
      adhesionhistory_ji.adhesion_force_ = adhesionhistory_ij.adhesion_force_;
    }

    // add adhesion force contribution
    Utils::vec_add_scale(force_i, adhesionhistory_ij.adhesion_force_, particlepair.e_ji_);
    if (status_j == PARTICLEENGINE::Owned)
      Utils::vec_add_scale(force_j, -adhesionhistory_ij.adhesion_force_, particlepair.e_ji_);
  }
}

void ParticleInteraction::DEMAdhesion::evaluate_particle_wall_adhesion()
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::DEMAdhesion::evaluate_particle_wall_adhesion");

  // get wall data state container
  std::shared_ptr<PARTICLEWALL::WallDataState> walldatastate =
      particlewallinterface_->get_wall_data_state();

  // get reference to particle-wall pair data
  const DEMParticleWallPairData& particlewallpairdata =
      neighborpairs_->get_ref_to_particle_wall_pair_adhesion_data();

  // get reference to particle-wall adhesion history pair data
  DEMHistoryPairAdhesionData& adhesionhistorydata =
      historypairs_->get_ref_to_particle_wall_adhesion_history_data();

  // write interaction output
  const bool writeinteractionoutput =
      particleinteractionwriter_->get_current_write_result_flag() and writeparticlewallinteraction_;

  // init storage for interaction output
  std::vector<double> attackpoints;
  std::vector<double> adhesionforces;
  std::vector<double> normaldirection;
  std::vector<double> surfaceenergy;

  // prepare storage for interaction output
  if (writeinteractionoutput)
  {
    const int numparticlewallpairs = particlewallpairdata.size();

    attackpoints.reserve(3 * numparticlewallpairs);
    adhesionforces.reserve(3 * numparticlewallpairs);
    normaldirection.reserve(3 * numparticlewallpairs);
    surfaceenergy.reserve(numparticlewallpairs);
  }

  // iterate over particle-wall pairs
  for (const auto& particlewallpair : particlewallpairdata)
  {
    // access values of local index tuple of particle i
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlewallpair.tuple_i_;

    // get corresponding particle container
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    // get global id of particle
    const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);

    // get pointer to particle states
    const double* pos_i = container_i->get_ptr_to_state(PARTICLEENGINE::Position, particle_i);
    const double* vel_i = container_i->get_ptr_to_state(PARTICLEENGINE::Velocity, particle_i);
    const double* rad_i = container_i->get_ptr_to_state(PARTICLEENGINE::Radius, particle_i);
    const double* mass_i = container_i->get_ptr_to_state(PARTICLEENGINE::Mass, particle_i);
    double* force_i = container_i->get_ptr_to_state(PARTICLEENGINE::Force, particle_i);

    // get pointer to column wall element
    Core::Elements::Element* ele = particlewallpair.ele_;

    // number of nodes of wall element
    const int numnodes = ele->num_node();

    // shape functions and location vector of wall element
    Core::LinAlg::SerialDenseVector funct(numnodes);
    std::vector<int> lmele;

    if (walldatastate->get_vel_col() != nullptr or walldatastate->get_force_col() != nullptr)
    {
      // evaluate shape functions of element at wall contact point
      Core::FE::shape_function_2d(
          funct, particlewallpair.elecoords_[0], particlewallpair.elecoords_[1], ele->shape());

      // get location vector of wall element
      lmele.reserve(numnodes * 3);
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      ele->location_vector(
          *particlewallinterface_->get_wall_discretization(), lmele, lmowner, lmstride);
    }

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

    // velocity of wall contact point j
    double vel_j[3] = {0.0, 0.0, 0.0};

    if (walldatastate->get_vel_col() != nullptr)
    {
      // get nodal velocities
      std::vector<double> nodal_vel =
          Core::FE::extract_values(*walldatastate->get_vel_col(), lmele);

      // determine velocity of wall contact point j
      for (int node = 0; node < numnodes; ++node)
        for (int dim = 0; dim < 3; ++dim) vel_j[dim] += funct[node] * nodal_vel[node * 3 + dim];
    }

    // relative velocity in wall contact point j (neglecting angular velocity)
    double vel_rel[3];
    Utils::vec_set(vel_rel, vel_i);
    Utils::vec_sub(vel_rel, vel_j);

    // magnitude of relative velocity in normal direction
    const double vel_rel_normal = Utils::vec_dot(vel_rel, particlewallpair.e_ji_);

    // get reference to touched adhesion history
    TouchedDEMHistoryPairAdhesion& touchedadhesionhistory_ij =
        adhesionhistorydata[globalid_i[0]][ele->id()];

    // mark adhesion history as touched
    touchedadhesionhistory_ij.first = true;

    // get reference to adhesion history
    DEMHistoryPairAdhesion& adhesionhistory_ij = touchedadhesionhistory_ij.second;

    // calculate adhesion surface energy
    if (not(adhesionhistory_ij.surface_energy_ > 0.0))
      adhesionsurfaceenergy_->adhesion_surface_energy(
          surface_energy, adhesionhistory_ij.surface_energy_);

    // calculate adhesion force
    adhesionlaw_->adhesion_force(particlewallpair.gap_, adhesionhistory_ij.surface_energy_,
        rad_i[0], vel_rel_normal, mass_i[0], adhesionhistory_ij.adhesion_force_);

    // add adhesion force contribution
    Utils::vec_add_scale(force_i, adhesionhistory_ij.adhesion_force_, particlewallpair.e_ji_);

    // copy history to relevant wall elements in penetration volume
    for (int histele : particlewallpair.histeles_)
      adhesionhistorydata[globalid_i[0]][histele] = touchedadhesionhistory_ij;

    // calculation of wall adhesion force
    double walladhesionforce[3] = {0.0, 0.0, 0.0};
    if (writeinteractionoutput or walldatastate->get_force_col() != nullptr)
    {
      Utils::vec_set_scale(
          walladhesionforce, -adhesionhistory_ij.adhesion_force_, particlewallpair.e_ji_);
    }

    // write interaction output
    if (writeinteractionoutput)
    {
      // compute vector from particle i to wall contact point j
      double r_ji[3];
      Utils::vec_set_scale(r_ji, (rad_i[0] + particlewallpair.gap_), particlewallpair.e_ji_);

      // calculate wall contact point
      double wallcontactpoint[3];
      Utils::vec_set(wallcontactpoint, pos_i);
      Utils::vec_add(wallcontactpoint, r_ji);

      // set wall attack point and states
      for (int dim = 0; dim < 3; ++dim) attackpoints.push_back(wallcontactpoint[dim]);
      for (int dim = 0; dim < 3; ++dim) adhesionforces.push_back(walladhesionforce[dim]);
      for (int dim = 0; dim < 3; ++dim) normaldirection.push_back(-particlewallpair.e_ji_[dim]);
      surfaceenergy.push_back(adhesionhistory_ij.surface_energy_);
    }

    // assemble adhesion force acting on wall element
    if (walldatastate->get_force_col() != nullptr)
    {
      // determine nodal forces
      std::vector<double> nodal_force(numnodes * 3);
      for (int node = 0; node < numnodes; ++node)
        for (int dim = 0; dim < 3; ++dim)
          nodal_force[node * 3 + dim] = funct[node] * walladhesionforce[dim];

      // assemble nodal forces
      walldatastate->get_force_col()->sum_into_global_values(
          numnodes * 3, nodal_force.data(), lmele.data());
    }
  }

  if (writeinteractionoutput)
  {
    // get specific runtime output writer
    Core::IO::VisualizationManager* visualization_manager =
        particleinteractionwriter_->get_specific_runtime_output_writer("particle-wall-adhesion");
    auto& visualization_data = visualization_manager->get_visualization_data();

    // set wall attack points
    visualization_data.get_point_coordinates() = attackpoints;

    // append states
    visualization_data.set_point_data_vector<double>("adhesion force", adhesionforces, 3);
    visualization_data.set_point_data_vector<double>("normal direction", normaldirection, 3);
    visualization_data.set_point_data_vector<double>("surface energy", surfaceenergy, 1);
  }
}

FOUR_C_NAMESPACE_CLOSE
