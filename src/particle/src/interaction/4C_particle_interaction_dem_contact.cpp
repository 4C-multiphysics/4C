// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_dem_contact.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_io_pstream.hpp"
#include "4C_io_visualization_manager.hpp"
#include "4C_mat_particle_wall_dem.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_dem_contact_normal.hpp"
#include "4C_particle_interaction_dem_contact_rolling.hpp"
#include "4C_particle_interaction_dem_contact_tangential.hpp"
#include "4C_particle_interaction_dem_history_pairs.hpp"
#include "4C_particle_interaction_dem_neighbor_pairs.hpp"
#include "4C_particle_interaction_material_handler.hpp"
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
ParticleInteraction::DEMContact::DEMContact(const Teuchos::ParameterList& params)
    : params_dem_(params),
      dt_(0.0),
      tension_cutoff_(params_dem_.get<bool>("TENSION_CUTOFF")),
      writeparticlewallinteraction_(params_dem_.get<bool>("WRITE_PARTICLE_WALL_INTERACTION"))
{
  // empty constructor
}

ParticleInteraction::DEMContact::~DEMContact() = default;

void ParticleInteraction::DEMContact::init()
{
  // init normal contact handler
  init_normal_contact_handler();

  // init tangential contact handler
  init_tangential_contact_handler();

  // init rolling contact handler
  init_rolling_contact_handler();
}

void ParticleInteraction::DEMContact::setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface,
    const std::shared_ptr<ParticleInteraction::MaterialHandler> particlematerial,
    const std::shared_ptr<ParticleInteraction::InteractionWriter> particleinteractionwriter,
    const std::shared_ptr<ParticleInteraction::DEMNeighborPairs> neighborpairs,
    const std::shared_ptr<ParticleInteraction::DEMHistoryPairs> historypairs)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set interface to particle wall handler
  particlewallinterface_ = particlewallinterface;

  // set particle material handler
  particlematerial_ = particlematerial;

  // set particle interaction writer
  particleinteractionwriter_ = particleinteractionwriter;

  // setup particle interaction writer
  setup_particle_interaction_writer();

  // set neighbor pair handler
  neighborpairs_ = neighborpairs;

  // set history pair handler
  historypairs_ = historypairs;

  // get maximum density of all materials
  const double maxdensity = get_max_density_of_all_materials();

  // setup normal contact handler
  contactnormal_->setup(maxdensity);

  // setup tangential contact handler
  if (contacttangential_) contacttangential_->setup(contactnormal_->get_normal_contact_stiffness());

  // setup rolling contact handler
  if (contactrolling_) contactrolling_->setup(contactnormal_->get_normal_contact_stiffness());

  // safety check
  if (contacttangential_)
  {
    // get type of normal contact law
    auto normalcontacttype =
        Teuchos::getIntegralValue<PARTICLE::NormalContact>(params_dem_, "NORMALCONTACTLAW");

    if (normalcontacttype != PARTICLE::NormalLinSpring and
        normalcontacttype != PARTICLE::NormalLinSpringDamp)
      FOUR_C_THROW("tangential contact law only valid with linear normal contact law!");
  }
}

void ParticleInteraction::DEMContact::set_current_step_size(const double currentstepsize)
{
  dt_ = currentstepsize;

  // set current step size
  if (contacttangential_) contacttangential_->set_current_step_size(currentstepsize);

  // set current step size
  if (contactrolling_) contactrolling_->set_current_step_size(currentstepsize);
}

void ParticleInteraction::DEMContact::insert_particle_states_of_particle_types(
    std::map<PARTICLEENGINE::TypeEnum, std::set<PARTICLEENGINE::StateEnum>>& particlestatestotypes)
    const
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    // set of particle states for current particle type
    std::set<PARTICLEENGINE::StateEnum>& particlestates = typeIt.second;

    // states for tangential and rolling contact evaluation scheme
    if (contacttangential_ or contactrolling_)
      particlestates.insert({PARTICLEENGINE::Moment, PARTICLEENGINE::AngularVelocity,
          PARTICLEENGINE::AngularAcceleration});
  }
}

double ParticleInteraction::DEMContact::get_normal_contact_stiffness() const
{
  return contactnormal_->get_normal_contact_stiffness();
}

void ParticleInteraction::DEMContact::check_critical_time_step() const
{
  if (particleengineinterface_->get_number_of_particles() == 0) return;
  // init value of minimum mass
  double minmass = std::numeric_limits<double>::max();

  // iterate over particle types
  for (const auto& type_i : particlecontainerbundle_->get_particle_types())
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container =
        particlecontainerbundle_->get_specific_container(type_i, PARTICLEENGINE::Owned);

    // get number of particles stored in container
    const int particlestored = container->particles_stored();

    // no owned particles of current particle type
    if (particlestored <= 0) continue;

    // get minimum stored value of state
    double currminmass = container->get_min_value_of_state(PARTICLEENGINE::Mass);

    // update value of minimum mass
    minmass = std::min(minmass, currminmass);
  }

  // get critical normal contact stiffness
  const double k_normal_crit = contactnormal_->get_critical_normal_contact_stiffness();

  // critical time step size based on particle-particle contact
  const double safety = 0.75;
  const double factor = contacttangential_ ? 0.22 : 0.34;
  const double dt_crit = safety * factor * std::sqrt(minmass / k_normal_crit);

  // checks time step
  if (dt_ > dt_crit)
    Core::IO::cout << "Warning: time step " << dt_ << " larger than critical time step " << dt_crit
                   << "!" << Core::IO::endl;
}

void ParticleInteraction::DEMContact::add_force_and_moment_contribution()
{
  // evaluate particle contact contribution
  evaluate_particle_contact();

  // evaluate particle-wall contact contribution
  if (particlewallinterface_) evaluate_particle_wall_contact();
}

void ParticleInteraction::DEMContact::evaluate_elastic_potential_energy(
    double& elasticpotentialenergy) const
{
  // evaluate particle elastic potential energy contribution
  evaluate_particle_elastic_potential_energy(elasticpotentialenergy);

  // evaluate particle-wall elastic potential energy contribution
  if (particlewallinterface_)
    evaluate_particle_wall_elastic_potential_energy(elasticpotentialenergy);
}

void ParticleInteraction::DEMContact::init_normal_contact_handler()
{
  // get type of normal contact law
  auto normalcontacttype =
      Teuchos::getIntegralValue<PARTICLE::NormalContact>(params_dem_, "NORMALCONTACTLAW");

  // create normal contact handler
  switch (normalcontacttype)
  {
    case PARTICLE::NormalLinSpring:
    {
      contactnormal_ = std::unique_ptr<ParticleInteraction::DEMContactNormalLinearSpring>(
          new ParticleInteraction::DEMContactNormalLinearSpring(params_dem_));
      break;
    }
    case PARTICLE::NormalLinSpringDamp:
    {
      contactnormal_ = std::unique_ptr<ParticleInteraction::DEMContactNormalLinearSpringDamp>(
          new ParticleInteraction::DEMContactNormalLinearSpringDamp(params_dem_));
      break;
    }
    case PARTICLE::NormalHertz:
    {
      contactnormal_ = std::unique_ptr<ParticleInteraction::DEMContactNormalHertz>(
          new ParticleInteraction::DEMContactNormalHertz(params_dem_));
      break;
    }
    case PARTICLE::NormalLeeHerrmann:
    {
      contactnormal_ = std::unique_ptr<ParticleInteraction::DEMContactNormalLeeHerrmann>(
          new ParticleInteraction::DEMContactNormalLeeHerrmann(params_dem_));
      break;
    }
    case PARTICLE::NormalKuwabaraKono:
    {
      contactnormal_ = std::unique_ptr<ParticleInteraction::DEMContactNormalKuwabaraKono>(
          new ParticleInteraction::DEMContactNormalKuwabaraKono(params_dem_));
      break;
    }
    case PARTICLE::NormalTsuji:
    {
      contactnormal_ = std::unique_ptr<ParticleInteraction::DEMContactNormalTsuji>(
          new ParticleInteraction::DEMContactNormalTsuji(params_dem_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown normal contact law type!");
      break;
    }
  }

  // init normal contact handler
  contactnormal_->init();
}

void ParticleInteraction::DEMContact::init_tangential_contact_handler()
{
  // get type of tangential contact law
  auto tangentialcontacttype =
      Teuchos::getIntegralValue<PARTICLE::TangentialContact>(params_dem_, "TANGENTIALCONTACTLAW");

  // create tangential contact handler
  switch (tangentialcontacttype)
  {
    case PARTICLE::NoTangentialContact:
    {
      contacttangential_ = std::unique_ptr<ParticleInteraction::DEMContactTangentialBase>(nullptr);
      break;
    }
    case PARTICLE::TangentialLinSpringDamp:
    {
      contacttangential_ =
          std::unique_ptr<ParticleInteraction::DEMContactTangentialLinearSpringDamp>(
              new ParticleInteraction::DEMContactTangentialLinearSpringDamp(params_dem_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown tangential contact law type!");
      break;
    }
  }

  // init tangential contact handler
  if (contacttangential_) contacttangential_->init();
}

void ParticleInteraction::DEMContact::init_rolling_contact_handler()
{
  // get type of rolling contact law
  auto rollingcontacttype =
      Teuchos::getIntegralValue<PARTICLE::RollingContact>(params_dem_, "ROLLINGCONTACTLAW");

  // create rolling contact handler
  switch (rollingcontacttype)
  {
    case PARTICLE::NoRollingContact:
    {
      contactrolling_ = std::unique_ptr<ParticleInteraction::DEMContactRollingBase>(nullptr);
      break;
    }
    case PARTICLE::RollingViscous:
    {
      contactrolling_ = std::unique_ptr<ParticleInteraction::DEMContactRollingViscous>(
          new ParticleInteraction::DEMContactRollingViscous(params_dem_));
      break;
    }
    case PARTICLE::RollingCoulomb:
    {
      contactrolling_ = std::unique_ptr<ParticleInteraction::DEMContactRollingCoulomb>(
          new ParticleInteraction::DEMContactRollingCoulomb(params_dem_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown rolling contact law type!");
      break;
    }
  }

  // init rolling contact handler
  if (contactrolling_) contactrolling_->init();
}

void ParticleInteraction::DEMContact::setup_particle_interaction_writer()
{
  // register specific runtime output writer
  if (writeparticlewallinteraction_)
    particleinteractionwriter_->register_specific_runtime_output_writer("particle-wall-contact");
}

double ParticleInteraction::DEMContact::get_max_density_of_all_materials() const
{
  // init value of maximum density
  double maxdensity(0.0);

  // iterate over particle types
  for (const auto& type_i : particlecontainerbundle_->get_particle_types())
  {
    // get material for current particle type
    const Mat::PAR::ParticleMaterialBase* material =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

    // compare to current maximum
    maxdensity = std::max(maxdensity, material->initDensity_);
  }

  return maxdensity;
}

void ParticleInteraction::DEMContact::evaluate_particle_contact()
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::DEMContact::evaluate_particle_contact");

  // get reference to particle tangential history pair data
  DEMHistoryPairTangentialData& tangentialhistorydata =
      historypairs_->get_ref_to_particle_tangential_history_data();

  // get reference to particle rolling history pair data
  DEMHistoryPairRollingData& rollinghistorydata =
      historypairs_->get_ref_to_particle_rolling_history_data();

  // tangential contact friction coefficient
  const double mu_tangential =
      contacttangential_ ? params_dem_.get<double>("FRICT_COEFF_TANG") : 0.0;

  // rolling contact friction coefficient
  const double mu_rolling = contactrolling_ ? params_dem_.get<double>("FRICT_COEFF_ROLL") : 0.0;

  // iterate over particle pairs
  for (const auto& particlepair : neighborpairs_->get_ref_to_particle_pair_data())
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

    const double* angvel_i = nullptr;
    double* moment_i = nullptr;
    if (contacttangential_ or contactrolling_)
    {
      angvel_i = container_i->get_ptr_to_state(PARTICLEENGINE::AngularVelocity, particle_i);
      moment_i = container_i->get_ptr_to_state(PARTICLEENGINE::Moment, particle_i);
    }

    const double* vel_j = container_j->get_ptr_to_state(PARTICLEENGINE::Velocity, particle_j);
    const double* rad_j = container_j->get_ptr_to_state(PARTICLEENGINE::Radius, particle_j);
    double* force_j = container_j->get_ptr_to_state(PARTICLEENGINE::Force, particle_j);

    const double* angvel_j = nullptr;
    double* moment_j = nullptr;
    if (contacttangential_ or contactrolling_)
    {
      angvel_j = container_j->get_ptr_to_state(PARTICLEENGINE::AngularVelocity, particle_j);
      moment_j = container_j->get_ptr_to_state(PARTICLEENGINE::Moment, particle_j);
    }

    // compute vectors from particle i and j to contact point c
    double r_ci[3], r_cj[3];
    if (contacttangential_)
    {
      Utils::vec_set_scale(r_ci, (rad_i[0] + 0.5 * particlepair.gap_), particlepair.e_ji_);
      Utils::vec_set_scale(r_cj, -(rad_j[0] + 0.5 * particlepair.gap_), particlepair.e_ji_);
    }
    else
    {
      std::fill(r_ci, r_ci + 3, 0.);
      std::fill(r_cj, r_cj + 3, 0.);
    }

    // relative velocity in contact point c between particle i and j
    double vel_rel[3];
    Utils::vec_set(vel_rel, vel_i);
    Utils::vec_sub(vel_rel, vel_j);
    if (contacttangential_)
    {
      Utils::vec_add_cross(vel_rel, angvel_i, r_ci);
      Utils::vec_add_cross(vel_rel, r_cj, angvel_j);
    }

    // magnitude of relative velocity in normal direction
    const double vel_rel_normal = Utils::vec_dot(vel_rel, particlepair.e_ji_);

    // calculate normal contact force
    double normalcontactforce(0.0);
    contactnormal_->normal_contact_force(
        particlepair.gap_, rad_i, rad_j, vel_rel_normal, particlepair.m_eff_, normalcontactforce);

    // evaluate tension cutoff of normal contact force
    if (tension_cutoff_) normalcontactforce = std::min(normalcontactforce, 0.0);

    // add normal contact force contribution
    Utils::vec_add_scale(force_i, normalcontactforce, particlepair.e_ji_);
    if (status_j == PARTICLEENGINE::Owned)
      Utils::vec_add_scale(force_j, -normalcontactforce, particlepair.e_ji_);

    // calculation of tangential contact force
    if (contacttangential_)
    {
      // get reference to touched tangential history
      TouchedDEMHistoryPairTangential& touchedtangentialhistory_ij =
          tangentialhistorydata[globalid_i[0]][globalid_j[0]];

      // mark tangential history as touched
      touchedtangentialhistory_ij.first = true;

      // get reference to tangential history
      DEMHistoryPairTangential& tangentialhistory_ij = touchedtangentialhistory_ij.second;

      // relative velocity in tangential direction
      double vel_rel_tangential[3];
      Utils::vec_set(vel_rel_tangential, vel_rel);
      Utils::vec_add_scale(vel_rel_tangential, -vel_rel_normal, particlepair.e_ji_);

      // calculate tangential contact force
      double tangentialcontactforce[3];
      contacttangential_->tangential_contact_force(tangentialhistory_ij.gap_t_,
          tangentialhistory_ij.stick_, particlepair.e_ji_, vel_rel_tangential, particlepair.m_eff_,
          mu_tangential, normalcontactforce, tangentialcontactforce);

      // copy history from interaction pair ij to ji
      if (status_j == PARTICLEENGINE::Owned)
      {
        // get reference to touched tangential history
        TouchedDEMHistoryPairTangential& touchedtangentialhistory_ji =
            tangentialhistorydata[globalid_j[0]][globalid_i[0]];

        // mark tangential history as touched
        touchedtangentialhistory_ji.first = true;

        // get reference to tangential history
        DEMHistoryPairTangential& tangentialhistory_ji = touchedtangentialhistory_ji.second;

        // set tangential gap and tangential stick flag
        Utils::vec_set_scale(tangentialhistory_ji.gap_t_, -1.0, tangentialhistory_ij.gap_t_);
        tangentialhistory_ji.stick_ = tangentialhistory_ij.stick_;
      }

      // add tangential contact force contribution
      Utils::vec_add(force_i, tangentialcontactforce);
      if (status_j == PARTICLEENGINE::Owned) Utils::vec_sub(force_j, tangentialcontactforce);

      // add tangential contact moment contribution
      Utils::vec_add_cross(moment_i, r_ci, tangentialcontactforce);
      if (status_j == PARTICLEENGINE::Owned)
        Utils::vec_add_cross(moment_j, tangentialcontactforce, r_cj);
    }

    // calculation of rolling contact moment
    if (contactrolling_)
    {
      // get reference to touched rolling history
      TouchedDEMHistoryPairRolling& touchedrollinghistory_ij =
          rollinghistorydata[globalid_i[0]][globalid_j[0]];

      // mark rolling history as touched
      touchedrollinghistory_ij.first = true;

      // get reference to rolling history
      DEMHistoryPairRolling& rollinghistory_ij = touchedrollinghistory_ij.second;

      // calculate effective radius
      double r_eff;
      contactrolling_->effective_radius_particle(rad_i, rad_j, particlepair.gap_, r_eff);

      // calculate relative rolling velocity
      double vel_rel_rolling[3];
      contactrolling_->relative_rolling_velocity(
          r_eff, particlepair.e_ji_, angvel_i, angvel_j, vel_rel_rolling);

      // calculate rolling contact moment
      double rollingcontactmoment[3];
      contactrolling_->rolling_contact_moment(rollinghistory_ij.gap_r_, rollinghistory_ij.stick_,
          particlepair.e_ji_, vel_rel_rolling, particlepair.m_eff_, r_eff, mu_rolling,
          normalcontactforce, rollingcontactmoment);

      // copy history from interaction pair ij to ji
      if (status_j == PARTICLEENGINE::Owned)
      {
        // get reference to touched rolling history
        TouchedDEMHistoryPairRolling& touchedrollinghistory_ji =
            rollinghistorydata[globalid_j[0]][globalid_i[0]];

        // mark rolling history as touched
        touchedrollinghistory_ji.first = true;

        // get reference to rolling history
        DEMHistoryPairRolling& rollinghistory_ji = touchedrollinghistory_ji.second;

        // set rolling gap and rolling stick flag
        Utils::vec_set_scale(rollinghistory_ji.gap_r_, -1.0, rollinghistory_ij.gap_r_);
        rollinghistory_ji.stick_ = rollinghistory_ij.stick_;
      }

      // add rolling contact moment contribution
      Utils::vec_add(moment_i, rollingcontactmoment);
      if (status_j == PARTICLEENGINE::Owned) Utils::vec_sub(moment_j, rollingcontactmoment);
    }
  }
}

void ParticleInteraction::DEMContact::evaluate_particle_wall_contact()
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::DEMContact::evaluate_particle_wall_contact");

  // get wall data state container
  std::shared_ptr<PARTICLEWALL::WallDataState> walldatastate =
      particlewallinterface_->get_wall_data_state();

  // get reference to particle-wall pair data
  const DEMParticleWallPairData& particlewallpairdata =
      neighborpairs_->get_ref_to_particle_wall_pair_data();

  // get reference to particle-wall tangential history pair data
  DEMHistoryPairTangentialData& tangentialhistorydata =
      historypairs_->get_ref_to_particle_wall_tangential_history_data();

  // get reference to particle-wall rolling history pair data
  DEMHistoryPairRollingData& rollinghistorydata =
      historypairs_->get_ref_to_particle_wall_rolling_history_data();

  // write interaction output
  const bool writeinteractionoutput =
      particleinteractionwriter_->get_current_write_result_flag() and writeparticlewallinteraction_;

  // init storage for interaction output
  std::vector<double> attackpoints;
  std::vector<double> contactforces;
  std::vector<double> normaldirection;

  // prepare storage for interaction output
  if (writeinteractionoutput)
  {
    const int numparticlewallpairs = particlewallpairdata.size();

    attackpoints.reserve(3 * numparticlewallpairs);
    contactforces.reserve(3 * numparticlewallpairs);
    normaldirection.reserve(3 * numparticlewallpairs);
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

    const double* angvel_i = nullptr;
    double* moment_i = nullptr;
    if (contacttangential_ or contactrolling_)
    {
      angvel_i = container_i->get_ptr_to_state(PARTICLEENGINE::AngularVelocity, particle_i);
      moment_i = container_i->get_ptr_to_state(PARTICLEENGINE::Moment, particle_i);
    }

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

    // tangential and rolling contact friction coefficient
    double mu_tangential = 0.0;
    double mu_rolling = 0.0;

    // get material parameters of wall element
    if (contacttangential_ or contactrolling_)
    {
      // cast material to particle wall material
      const std::shared_ptr<const Mat::ParticleWallMaterialDEM>& particlewallmaterial =
          std::dynamic_pointer_cast<const Mat::ParticleWallMaterialDEM>(ele->material());
      if (particlewallmaterial == nullptr)
        FOUR_C_THROW("cast to Mat::ParticleWallMaterialDEM failed!");

      // get tangential contact friction coefficient
      if (contacttangential_) mu_tangential = particlewallmaterial->mu_tangential();

      // get rolling contact friction coefficient
      if (contactrolling_) mu_rolling = particlewallmaterial->mu_rolling();
    }

    // declare zero radius for wall contact point j
    const double rad_j = 0.0;

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

    // compute vector from particle i to wall contact point j
    double r_ji[3];
    Utils::vec_set_scale(r_ji, (rad_i[0] + particlewallpair.gap_), particlewallpair.e_ji_);

    // relative velocity in wall contact point j
    double vel_rel[3];
    Utils::vec_set(vel_rel, vel_i);
    Utils::vec_sub(vel_rel, vel_j);
    if (contacttangential_) Utils::vec_add_cross(vel_rel, angvel_i, r_ji);

    // magnitude of relative velocity in normal direction
    const double vel_rel_normal = Utils::vec_dot(vel_rel, particlewallpair.e_ji_);

    // calculate normal contact force
    double normalcontactforce(0.0);
    contactnormal_->normal_contact_force(
        particlewallpair.gap_, rad_i, &rad_j, vel_rel_normal, mass_i[0], normalcontactforce);

    // evaluate tension cutoff of normal contact force
    if (tension_cutoff_) normalcontactforce = std::min(normalcontactforce, 0.0);

    // add normal contact force contribution
    Utils::vec_add_scale(force_i, normalcontactforce, particlewallpair.e_ji_);

    // calculation of tangential contact force
    double tangentialcontactforce[3] = {0.0, 0.0, 0.0};
    if (contacttangential_ and mu_tangential > 0.0)
    {
      // get reference to touched tangential history
      TouchedDEMHistoryPairTangential& touchedtangentialhistory_ij =
          tangentialhistorydata[globalid_i[0]][ele->id()];

      // mark tangential history as touched
      touchedtangentialhistory_ij.first = true;

      // get reference to tangential history
      DEMHistoryPairTangential& tangentialhistory_ij = touchedtangentialhistory_ij.second;

      // relative velocity in tangential direction
      double vel_rel_tangential[3];
      Utils::vec_set(vel_rel_tangential, vel_rel);
      Utils::vec_add_scale(vel_rel_tangential, -vel_rel_normal, particlewallpair.e_ji_);

      // calculate tangential contact force
      contacttangential_->tangential_contact_force(tangentialhistory_ij.gap_t_,
          tangentialhistory_ij.stick_, particlewallpair.e_ji_, vel_rel_tangential, mass_i[0],
          mu_tangential, normalcontactforce, tangentialcontactforce);

      // add tangential contact force contribution
      Utils::vec_add(force_i, tangentialcontactforce);

      // add tangential contact moment contribution
      Utils::vec_add_cross(moment_i, r_ji, tangentialcontactforce);

      // copy history to relevant wall elements in penetration volume
      for (int histele : particlewallpair.histeles_)
        tangentialhistorydata[globalid_i[0]][histele] = touchedtangentialhistory_ij;
    }

    // calculation of rolling contact moment
    double rollingcontactmoment[3] = {0.0, 0.0, 0.0};
    if (contactrolling_ and mu_rolling > 0.0)
    {
      // get reference to touched rolling history
      TouchedDEMHistoryPairRolling& touchedrollinghistory_ij =
          rollinghistorydata[globalid_i[0]][ele->id()];

      // mark rolling history as touched
      touchedrollinghistory_ij.first = true;

      // get reference to rolling history
      DEMHistoryPairRolling& rollinghistory_ij = touchedrollinghistory_ij.second;

      // calculate effective radius
      double r_eff;
      contactrolling_->effective_radius_particle(rad_i, nullptr, particlewallpair.gap_, r_eff);

      // calculate relative rolling velocity
      double vel_rel_rolling[3];
      contactrolling_->relative_rolling_velocity(
          r_eff, particlewallpair.e_ji_, angvel_i, nullptr, vel_rel_rolling);

      // calculate rolling contact moment
      contactrolling_->rolling_contact_moment(rollinghistory_ij.gap_r_, rollinghistory_ij.stick_,
          particlewallpair.e_ji_, vel_rel_rolling, mass_i[0], r_eff, mu_rolling, normalcontactforce,
          rollingcontactmoment);

      // add rolling contact moment contribution
      Utils::vec_add(moment_i, rollingcontactmoment);

      // copy history to relevant wall elements in penetration volume
      for (int histele : particlewallpair.histeles_)
        rollinghistorydata[globalid_i[0]][histele] = touchedrollinghistory_ij;
    }

    // calculation of wall contact force
    double wallcontactforce[3] = {0.0, 0.0, 0.0};
    if (writeinteractionoutput or walldatastate->get_force_col() != nullptr)
    {
      Utils::vec_set_scale(wallcontactforce, -normalcontactforce, particlewallpair.e_ji_);
      Utils::vec_sub(wallcontactforce, tangentialcontactforce);
    }

    // write interaction output
    if (writeinteractionoutput)
    {
      // calculate wall contact point
      double wallcontactpoint[3];
      Utils::vec_set(wallcontactpoint, pos_i);
      Utils::vec_add(wallcontactpoint, r_ji);

      // set wall attack point and states
      for (int dim = 0; dim < 3; ++dim) attackpoints.push_back(wallcontactpoint[dim]);
      for (int dim = 0; dim < 3; ++dim) contactforces.push_back(wallcontactforce[dim]);
      for (int dim = 0; dim < 3; ++dim) normaldirection.push_back(-particlewallpair.e_ji_[dim]);
    }

    // assemble contact force acting on wall element
    if (walldatastate->get_force_col() != nullptr)
    {
      // determine nodal forces
      std::vector<double> nodal_force(numnodes * 3);
      for (int node = 0; node < numnodes; ++node)
        for (int dim = 0; dim < 3; ++dim)
          nodal_force[node * 3 + dim] = funct[node] * wallcontactforce[dim];

      // assemble nodal forces
      walldatastate->get_force_col()->sum_into_global_values(
          numnodes * 3, nodal_force.data(), lmele.data());
    }
  }

  if (writeinteractionoutput)
  {
    // get specific runtime output writer
    Core::IO::VisualizationManager* visualization_manager =
        particleinteractionwriter_->get_specific_runtime_output_writer("particle-wall-contact");
    auto& visualization_data = visualization_manager->get_visualization_data();

    // set wall attack points
    visualization_data.get_point_coordinates() = attackpoints;

    // append states
    visualization_data.set_point_data_vector<double>("contact force", contactforces, 3);
    visualization_data.set_point_data_vector<double>("normal direction", normaldirection, 3);
  }
}

void ParticleInteraction::DEMContact::evaluate_particle_elastic_potential_energy(
    double& elasticpotentialenergy) const
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "ParticleInteraction::DEMContact::evaluate_particle_elastic_potential_energy");

  // get reference to particle tangential history pair data
  DEMHistoryPairTangentialData& tangentialhistorydata =
      historypairs_->get_ref_to_particle_tangential_history_data();

  // get reference to particle rolling history pair data
  DEMHistoryPairRollingData& rollinghistorydata =
      historypairs_->get_ref_to_particle_rolling_history_data();

  // iterate over particle pairs
  for (const auto& particlepair : neighborpairs_->get_ref_to_particle_pair_data())
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

    // calculate normal potential energy
    double normalpotentialenergy(0.0);
    contactnormal_->normal_potential_energy(particlepair.gap_, normalpotentialenergy);

    // add normal potential energy contribution
    elasticpotentialenergy += 0.5 * normalpotentialenergy;
    if (status_j == PARTICLEENGINE::Owned) elasticpotentialenergy += 0.5 * normalpotentialenergy;

    // calculation of tangential potential energy
    if (contacttangential_)
    {
      // get reference to touched tangential history
      TouchedDEMHistoryPairTangential& touchedtangentialhistory_ij =
          tangentialhistorydata[globalid_i[0]][globalid_j[0]];

      // get reference to tangential history
      DEMHistoryPairTangential& tangentialhistory_ij = touchedtangentialhistory_ij.second;

      // calculate tangential potential energy
      double tangentialpotentialenergy(0.0);
      contacttangential_->tangential_potential_energy(
          tangentialhistory_ij.gap_t_, tangentialpotentialenergy);

      // add tangential potential energy contribution
      elasticpotentialenergy += 0.5 * tangentialpotentialenergy;
      if (status_j == PARTICLEENGINE::Owned)
        elasticpotentialenergy += 0.5 * tangentialpotentialenergy;
    }

    // calculation of rolling potential energy
    if (contactrolling_)
    {
      // get reference to touched rolling history
      TouchedDEMHistoryPairRolling& touchedrollinghistory_ij =
          rollinghistorydata[globalid_i[0]][globalid_j[0]];

      // get reference to rolling history
      DEMHistoryPairRolling& rollinghistory_ij = touchedrollinghistory_ij.second;

      // calculate rolling potential energy
      double rollingpotentialenergy(0.0);
      contactrolling_->rolling_potential_energy(rollinghistory_ij.gap_r_, rollingpotentialenergy);

      // add rolling potential energy contribution
      elasticpotentialenergy += 0.5 * rollingpotentialenergy;
      if (status_j == PARTICLEENGINE::Owned) elasticpotentialenergy += 0.5 * rollingpotentialenergy;
    }
  }
}

void ParticleInteraction::DEMContact::evaluate_particle_wall_elastic_potential_energy(
    double& elasticpotentialenergy) const
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "ParticleInteraction::DEMContact::evaluate_particle_wall_elastic_potential_energy");

  // get reference to particle-wall tangential history pair data
  DEMHistoryPairTangentialData& tangentialhistorydata =
      historypairs_->get_ref_to_particle_wall_tangential_history_data();

  // get reference to particle-wall rolling history pair data
  DEMHistoryPairRollingData& rollinghistorydata =
      historypairs_->get_ref_to_particle_wall_rolling_history_data();

  // iterate over particle-wall pairs
  for (const auto& particlewallpair : neighborpairs_->get_ref_to_particle_wall_pair_data())
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

    // get pointer to column wall element
    Core::Elements::Element* ele = particlewallpair.ele_;

    // calculate normal potential energy
    double normalpotentialenergy(0.0);
    contactnormal_->normal_potential_energy(particlewallpair.gap_, normalpotentialenergy);

    // add normal potential energy contribution
    elasticpotentialenergy += normalpotentialenergy;

    // calculation of tangential potential energy
    if (contacttangential_)
    {
      // get reference to touched tangential history
      TouchedDEMHistoryPairTangential& touchedtangentialhistory_ij =
          tangentialhistorydata[globalid_i[0]][ele->id()];

      // get reference to tangential history
      DEMHistoryPairTangential& tangentialhistory_ij = touchedtangentialhistory_ij.second;

      // calculate tangential potential energy
      double tangentialpotentialenergy(0.0);
      contacttangential_->tangential_potential_energy(
          tangentialhistory_ij.gap_t_, tangentialpotentialenergy);

      // add tangential potential energy contribution
      elasticpotentialenergy += tangentialpotentialenergy;
    }

    // calculation of rolling potential energy
    if (contactrolling_)
    {
      // get reference to touched rolling history
      TouchedDEMHistoryPairRolling& touchedrollinghistory_ij =
          rollinghistorydata[globalid_i[0]][ele->id()];

      // get reference to rolling history
      DEMHistoryPairRolling& rollinghistory_ij = touchedrollinghistory_ij.second;

      // calculate rolling potential energy
      double rollingpotentialenergy(0.0);
      contactrolling_->rolling_potential_energy(rollinghistory_ij.gap_r_, rollingpotentialenergy);

      // add rolling potential energy contribution
      elasticpotentialenergy += rollingpotentialenergy;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
