// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_peridynamic.hpp"

#include "4C_global_data.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_interaction_dem_neighbor_pairs.hpp"
#include "4C_particle_interaction_material_handler.hpp"
#include "4C_particle_interaction_pd_neighbor_pairs.hpp"
#include "4C_particle_interaction_sph.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <numbers>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::SPHPeridynamic::SPHPeridynamic(const Teuchos::ParameterList& particle_params)
    : bondlist_(std::make_shared<std::vector<
              std::pair<Particle::LocalGlobalIndexTuple, Particle::LocalGlobalIndexTuple>>>()),
      horizon_pd_(particle_params.sublist("PD").get<double>("INTERACTION_HORIZON")),
      dx_pd_(particle_params.sublist("PD").get<double>("PERIDYNAMIC_GRID_SPACING")),
      stiff_(particle_params.sublist("PD").get<double>("NORMAL_STIFF")),
      damp_(particle_params.sublist("PD").get<double>("NORMAL_DAMP")),
      peridynamic_dimension_(Teuchos::getIntegralValue<Particle::PeridynamicDimension>(
          particle_params.sublist("PD"), "PD_DIMENSION"))
{
  FOUR_C_ASSERT_ALWAYS(not particle_params.get<bool>("RIGID_BODY_MOTION"),
      "Peridynamic interaction is not available in combination with rigid body motion!");

  auto normalcontacttype = Teuchos::getIntegralValue<Particle::NormalContact>(
      particle_params.sublist("PD"), "NORMALCONTACTLAW");

  if (normalcontacttype == Particle::NormalLinSpring and damp_ != 0.0)
    FOUR_C_THROW(
        "NORMAL_DAMP in PARTICLE DYNAMIC/PD section needs to be 0.0 if NORMALCONTACTLAW is set "
        "to NormalLinearSpring!");

  FOUR_C_ASSERT_ALWAYS(
      horizon_pd_ > 0.0, "Peridynamic INTERACTION_HORIZON must be greater than zero!");

  Teuchos::ParameterList binning_params = Global::Problem::instance()->binning_strategy_params();
  const double minimum_bin_size = binning_params.get<double>("BIN_SIZE_LOWER_BOUND");
  FOUR_C_ASSERT_ALWAYS(horizon_pd_ <= minimum_bin_size,
      "Peridynamic INTERACTION_HORIZON must be smaller than BIN_SIZE_LOWER_BOUND!");

  // checks for the dimensionality of the problem
  {
    const auto constraint_type = particle_params.sublist("INITIAL AND BOUNDARY CONDITIONS")
                                     .get<Particle::Constraint>("CONSTRAINT");

    if (peridynamic_dimension_ == Particle::PD_2DPlaneStress ||
        peridynamic_dimension_ == Particle::PD_2DPlaneStrain)
    {
      FOUR_C_ASSERT_ALWAYS(constraint_type == Particle::Projection2D,
          "Plane stress or plane strain for peridynamic requested. CONSTRAINT must be set to "
          "Projection2D!");
    }

    if (constraint_type == Particle::Projection2D)
    {
      FOUR_C_ASSERT_ALWAYS(peridynamic_dimension_ != Particle::PD_3D,
          "Projection2D CONSTRAINT is active. Choose 2D_PlaneStress or 2D_PlaneStrain as "
          "PD_DIMENSION!");
    }
  }
}

void Particle::SPHPeridynamic::init(
    const std::shared_ptr<Particle::PDNeighborPairs> neighborpairs_pd)
{
  // set bond list to pd neighbor pairs here because read_restart requires already bond list
  neighborpairs_pd->set_bond_list(bondlist_);

  neighborpairs_pd_ = neighborpairs_pd;
}

void Particle::SPHPeridynamic::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::MaterialHandler> particlematerial)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set particle material handler
  particlematerial_ = particlematerial;
}

void Particle::SPHPeridynamic::insert_particle_states_of_particle_types(
    std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    if (typeIt.first == Particle::PDPhase)
    {
      // set of particle states for current particle type
      std::set<Particle::StateEnum>& particlestates = typeIt.second;

      // set temperature state
      particlestates.insert({Particle::Force, Particle::PDBodyId, Particle::ReferencePosition,
          Particle::Young, Particle::CriticalStretch, Particle::InitialConnectedBonds,
          Particle::CurrentConnectedBonds, Particle::PDDamageVariable});
    }
  }
}

void Particle::SPHPeridynamic::init_peridynamic_bondlist()
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  // get material for peridynamic phase
  const Mat::PAR::ParticleMaterialBase* material =
      particlematerial_->get_ptr_to_particle_mat_parameter(Particle::PDPhase);

  // (initial) radius of current phase
  const double initradius = material->initRadius_;
#endif

  // important: bin size must be large enough to cover at least peridynamic horizon
  const Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();
  // iterate over potential particle neighbors
  for (const auto& potentialneighbors :
      particleengineinterface_->get_potential_particle_neighbors())
  {
    // access values of local index tuples of particle i and j
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = potentialneighbors.first;

    Particle::TypeEnum type_j;
    Particle::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = potentialneighbors.second;

    // only peridynamic phase particles can undergo peridynamic interaction
    if (type_i != Particle::PDPhase || type_j != Particle::PDPhase) continue;

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle->get_specific_container(type_i, status_i);
    Particle::ParticleContainer* container_j =
        particlecontainerbundle->get_specific_container(type_j, status_j);

    // get pointer to particle states
    const double* pos_i = container_i->get_ptr_to_state(Particle::Position, particle_i);
    const double* pdbodyid_i = container_i->get_ptr_to_state(Particle::PDBodyId, particle_i);
    double* initialconnectedbonds_i =
        container_i->get_ptr_to_state(Particle::InitialConnectedBonds, particle_i);

    const double* pos_j = container_j->get_ptr_to_state(Particle::Position, particle_j);
    const double* pdbodyid_j = container_j->get_ptr_to_state(Particle::PDBodyId, particle_j);
    double* initialconnectedbonds_j =
        container_j->get_ptr_to_state(Particle::InitialConnectedBonds, particle_j);

    // vector from particle i to j
    double r_ji[3];
    // distance between particles considering periodic boundaries
    particleengineinterface_->distance_between_particles(pos_i, pos_j, r_ji);
    // absolute distance between particles
    const double absdist = ParticleUtils::vec_norm_two(r_ji);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (absdist < (1.0e-10 * initradius) or absdist < (1.0e-10 * initradius))
      FOUR_C_THROW("absolute distance %f between particles close to zero!", absdist);
#endif

    // neighboring particles within interaction distance
    if (absdist <= horizon_pd_)
    {
      // initialize particle pair
      const int id_i = std::round(pdbodyid_i[0]);
      const int id_j = std::round(pdbodyid_j[0]);
      if (id_j == id_i)
      {
        // get global id of particle i
        const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);
        const int* globalid_j = container_j->get_ptr_to_global_id(particle_j);

        // creat the content of the bond between particles i & j
        Particle::LocalGlobalIndexTuple tuple_i =
            std::make_tuple(type_i, status_i, particle_i, globalid_i[0]);
        Particle::LocalGlobalIndexTuple tuple_j =
            std::make_tuple(type_j, status_j, particle_j, globalid_j[0]);

        bondlist_->push_back(std::make_pair(tuple_i, tuple_j));

        //  increase bond counter for particles i and j
        initialconnectedbonds_i[0] += 1.0;
        initialconnectedbonds_j[0] += 1.0;
      }
    }
  }

  Core::IO::cout << "Number of initialized peridynamic bonds on this proc: " << bondlist_->size()
                 << Core::IO::endl;

  // initialize the current number of connected bonds with initial number of connected bonds
  Particle::ParticleContainer* container =
      particlecontainerbundle->get_specific_container(Particle::PDPhase, Particle::Owned);
  container->update_state(
      0.0, Particle::CurrentConnectedBonds, 1.0, Particle::InitialConnectedBonds);
}

void Particle::SPHPeridynamic::add_acceleration_contribution() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPeridynamic::add_acceleration_contribution");

  // calculate peridynamic and short range DEM interaction forces
  compute_interaction_forces();

  // calculate acceleration from forces
  compute_acceleration();

  // clear force of peridynamic phase particles
  Particle::ParticleContainer* container =
      particleengineinterface_->get_particle_container_bundle()->get_specific_container(
          Particle::PDPhase, Particle::Owned);
  container->clear_state(Particle::Force);
}

void Particle::SPHPeridynamic::compute_interaction_forces() const
{
  const Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();

  size_t iter = 0;
  while (iter < bondlist_->size())
  {
    const auto& particlepair = (*bondlist_)[iter];

    // access values of local index tuples of particle i and j
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i, globalid_i;
    std::tie(type_i, status_i, particle_i, globalid_i) = particlepair.first;

    Particle::TypeEnum type_j;
    Particle::StatusEnum status_j;
    int particle_j, globalid_j;
    std::tie(type_j, status_j, particle_j, globalid_j) = particlepair.second;

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle->get_specific_container(type_i, status_i);

    Particle::ParticleContainer* container_j =
        particlecontainerbundle->get_specific_container(type_j, status_j);

    //  get pointer to particle states
    const double* ref_pos_i =
        container_i->get_ptr_to_state(Particle::ReferencePosition, particle_i);

    const double* pos_i = container_i->get_ptr_to_state(Particle::Position, particle_i);
    const double* young_i = container_i->get_ptr_to_state(Particle::Young, particle_i);
    double* force_i = container_i->cond_get_ptr_to_state(Particle::Force, particle_i);
    const double* critical_stretch_i =
        container_i->get_ptr_to_state(Particle::CriticalStretch, particle_i);

    const double* ref_pos_j =
        container_j->get_ptr_to_state(Particle::ReferencePosition, particle_j);
    const double* pos_j = container_j->get_ptr_to_state(Particle::Position, particle_j);
    const double* young_j = container_j->get_ptr_to_state(Particle::Young, particle_j);
    double* force_j = container_j->get_ptr_to_state(Particle::Force, particle_j);

    const double* critical_stretch_j =
        container_j->get_ptr_to_state(Particle::CriticalStretch, particle_j);
    // calculate the bond between two particles
    double xi[3];
    ParticleUtils::vec_set(xi, ref_pos_j);
    ParticleUtils::vec_sub(xi, ref_pos_i);

    // calculate the relative position of the pair
    double xi_eta[3];
    ParticleUtils::vec_set(xi_eta, pos_j);
    ParticleUtils::vec_sub(xi_eta, pos_i);

    // calculate the required norms of the pair
    const double xi_norm = ParticleUtils::vec_norm_two(xi);
    const double xi_eta_norm = ParticleUtils::vec_norm_two(xi_eta);

    // calculate the bond stretch
    double stretch = (xi_eta_norm - xi_norm) / xi_norm;

    // check the stretch
    const double critical_stretch = 0.5 * (critical_stretch_i[0] + critical_stretch_j[0]);

    // if critical stretch is not reached
    if (stretch < critical_stretch)
    {
      double m[3];
      ParticleUtils::vec_set_scale(m, 1.0 / xi_eta_norm, xi_eta);

      // calculate the bond force of the pair
      double fac;
      switch (peridynamic_dimension_)
      {
        case Particle::PD_3D:
          fac = (12.0 * (young_i[0] + young_j[0]) * 0.5) /
                (std::numbers::pi * horizon_pd_ * horizon_pd_ * horizon_pd_ * horizon_pd_) *
                stretch * calculate_volume_correction_factor(xi_norm) * dx_pd_ * dx_pd_ * dx_pd_ *
                dx_pd_ * dx_pd_ * dx_pd_;
          break;
        case Particle::PD_2DPlaneStress:
          fac = (9.0 * (young_i[0] + young_j[0]) * 0.5) /
                (std::numbers::pi * horizon_pd_ * horizon_pd_ * horizon_pd_) * stretch *
                calculate_volume_correction_factor(xi_norm) * dx_pd_ * dx_pd_ * dx_pd_ * dx_pd_;
          break;
        case Particle::PD_2DPlaneStrain:
          fac = (48.0 * (young_i[0] + young_j[0]) * 0.5) /
                (std::numbers::pi * 5.0 * horizon_pd_ * horizon_pd_ * horizon_pd_) * stretch *
                calculate_volume_correction_factor(xi_norm) * dx_pd_ * dx_pd_ * dx_pd_ * dx_pd_;
          break;
        default:
          FOUR_C_THROW("Unknown PD_DIMENSION given: {}!", peridynamic_dimension_);
      }

      // add bond force contribution
      ParticleUtils::vec_add_scale(force_i, fac, m);
      if (status_j == Particle::Owned) ParticleUtils::vec_add_scale(force_j, -fac, m);

      ++iter;
    }
    else
    {
      double* currentconnectedbonds_i =
          container_i->get_ptr_to_state(Particle::CurrentConnectedBonds, particle_i);
      double* currentconnectedbonds_j =
          container_j->get_ptr_to_state(Particle::CurrentConnectedBonds, particle_j);

      currentconnectedbonds_i[0] -= 1.0;
      currentconnectedbonds_j[0] -= 1.0;

      (*bondlist_)[iter] = std::move(bondlist_->back());
      bondlist_->pop_back();
    }
  }

  const PDParticlePairData& pd_neighbor_pairs = neighborpairs_pd_->get_ref_to_particle_pair_data();
  Core::IO::cout << "Number of pd_neighbor_pairs in peridynamic evaluation on this proc: "
                 << pd_neighbor_pairs.size() << Core::IO::endl;
  for (const auto& particlepair : pd_neighbor_pairs)
  {
    // access values of local index tuples of particle i and j
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    Particle::TypeEnum type_j;
    Particle::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle->get_specific_container(type_i, status_i);

    Particle::ParticleContainer* container_j =
        particlecontainerbundle->get_specific_container(type_j, status_j);

    // get pointer to particle states
    const double* vel_i = container_i->get_ptr_to_state(Particle::Velocity, particle_i);
    double* force_i = container_i->cond_get_ptr_to_state(Particle::Force, particle_i);

    // get pointer to particle states
    const double* vel_j = container_j->get_ptr_to_state(Particle::Velocity, particle_j);

    double* force_j = nullptr;
    if (status_j == Particle::Owned)
      force_j = container_j->cond_get_ptr_to_state(Particle::Force, particle_j);

    // compute normal gap and rate of normal gap
    const double gap = particlepair.gap_;
    const double gapdot = -1.0 * (ParticleUtils::vec_dot(vel_i, particlepair.e_ji_) -
                                     ParticleUtils::vec_dot(vel_j, particlepair.e_ji_));

    // magnitude of peridynamic particle contact force
    const double fac = std::min(0.0, (stiff_ * gap + damp_ * gapdot));

    // add contributions
    if (force_i) ParticleUtils::vec_add_scale(force_i, fac, particlepair.e_ji_);
    if (force_j) ParticleUtils::vec_add_scale(force_j, -fac, particlepair.e_ji_);
  }
}

void Particle::SPHPeridynamic::compute_acceleration() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPeridynamic::compute_acceleration");

  // get container of owned particles of current particle type
  Particle::ParticleContainer* container =
      particlecontainerbundle_->get_specific_container(Particle::PDPhase, Particle::Owned);

  // get number of particles stored in container
  const int particlestored = container->particles_stored();

  // no owned particles of current particle type
  if (particlestored <= 0) return;

  // get particle state dimension
  const int statedim = container->get_state_dim(Particle::Acceleration);

  // get pointer to particle states
  const double* radius = container->get_ptr_to_state(Particle::Radius, 0);
  const double* mass = container->get_ptr_to_state(Particle::Mass, 0);
  const double* force = container->get_ptr_to_state(Particle::Force, 0);
  const double* moment = container->cond_get_ptr_to_state(Particle::Moment, 0);
  double* acc = container->get_ptr_to_state(Particle::Acceleration, 0);
  double* angacc = container->cond_get_ptr_to_state(Particle::AngularAcceleration, 0);

  // compute acceleration
  for (int i = 0; i < particlestored; ++i)
  {
    ParticleUtils::vec_add_scale(&acc[statedim * i], (1.0 / mass[i]), &force[statedim * i]);
  }

  // compute angular acceleration
  if (angacc and moment)
  {
    for (int i = 0; i < particlestored; ++i)
      ParticleUtils::vec_add_scale(&angacc[statedim * i],
          (5.0 / (2.0 * mass[i] * ParticleUtils::pow<2>(radius[i]))), &moment[statedim * i]);
  }
}

void Particle::SPHPeridynamic::damage_evaluation()
{
  // get particle container bundle
  Particle::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->get_particle_container_bundle();
  // get container of owned particles of peridynamic phase
  Particle::ParticleContainer* container =
      particlecontainerbundle->get_specific_container(Particle::PDPhase, Particle::Owned);

  // loop over particles in container
  for (int particle_i = 0; particle_i < container->particles_stored(); ++particle_i)
  {
    const double* initialconnectedbonds_i =
        container->get_ptr_to_state(Particle::InitialConnectedBonds, particle_i);

    const double* currentconnectedbonds_i =
        container->get_ptr_to_state(Particle::CurrentConnectedBonds, particle_i);

    double* pddamagevariable_i =
        container->get_ptr_to_state(Particle::PDDamageVariable, particle_i);

    pddamagevariable_i[0] = 1.0 - currentconnectedbonds_i[0] / initialconnectedbonds_i[0];
  }
}

// the beta correction volume function in peridynamic
double Particle::SPHPeridynamic::calculate_volume_correction_factor(const double xi) const
{
  if (xi <= horizon_pd_ - dx_pd_ * 0.5)
  {
    return 1.0;
  }
  else if (xi <= horizon_pd_ + dx_pd_ * 0.5)
  {
    return (horizon_pd_ + dx_pd_ * 0.5 - xi) / (dx_pd_);
  }
  else
  {
    return 0.0;
  }
}

FOUR_C_NAMESPACE_CLOSE
