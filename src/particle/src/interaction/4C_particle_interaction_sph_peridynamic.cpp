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
#include "4C_utils_std23_unreachable.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <numbers>

FOUR_C_NAMESPACE_OPEN

// forward declarations of file-local geometry helpers
namespace
{
  bool intervals_overlap(double min_a, double max_a, double min_b, double max_b);
  bool bond_aabb_overlaps_segment(
      const double* pos_i, const double* pos_j, const std::array<double, 6>& segment);
  bool bond_aabb_overlaps_patch(
      const double* pos_i, const double* pos_j, const std::array<double, 9>& patch);
}  // namespace

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

  // A mismatch produces physically inconsistent forces
  const double dx_sph = particle_params.sublist("SPH").get<double>("INITIALPARTICLESPACING");
  FOUR_C_ASSERT_ALWAYS(abs(dx_pd_ - dx_sph) <= 1.0e-10 * dx_pd_,
      "PERIDYNAMIC_GRID_SPACING ({:.10f}) must equal INITIALPARTICLESPACING ({:.10f}). ", dx_pd_,
      dx_sph);

  auto normalcontacttype = Teuchos::getIntegralValue<Particle::NormalContact>(
      particle_params.sublist("PD"), "NORMALCONTACTLAW");

  if (normalcontacttype == Particle::NormalLinSpring and damp_ != 0.0)
    FOUR_C_THROW(
        "NORMAL_DAMP in PARTICLE DYNAMIC/PD section needs to be 0.0 if NORMALCONTACTLAW is set "
        "to NormalLinearSpring!");

  FOUR_C_ASSERT_ALWAYS(
      horizon_pd_ > 0.0, "Peridynamic INTERACTION_HORIZON must be greater than zero!");

  // to ensure that slightly stretched bonds are still found properly in the neighbor search, the
  // interaction horizon needs to be smaller than the minimum bin size of the binning strategy
  // The factor 1.1 is physics motivated as usual critical stretch values are (way) smaller
  Teuchos::ParameterList binning_params = Global::Problem::instance()->binning_strategy_params();
  const double minimum_bin_size = binning_params.get<double>("BIN_SIZE_LOWER_BOUND");
  FOUR_C_ASSERT_ALWAYS(horizon_pd_ * 1.1 < minimum_bin_size,
      "Peridynamic INTERACTION_HORIZON * 1.1(safety factor to account for stretch) must be smaller "
      "than BIN_SIZE_LOWER_BOUND!");

  // parse pre-crack line segments (3D): each entry has START and END (3 doubles each)
  {
    const Teuchos::ParameterList& pd_params = particle_params.sublist("PD");
    if (pd_params.isParameter("PRE_CRACK_LINES"))
    {
      const auto& lines = pd_params.get<std::vector<Teuchos::ParameterList>>("PRE_CRACK_LINES");
      for (const auto& entry : lines)
      {
        const auto& s = entry.get<std::vector<double>>("START");
        const auto& e = entry.get<std::vector<double>>("END");
        pre_crack_lines_.push_back({s[0], s[1], s[2], e[0], e[1], e[2]});
      }
      Core::IO::cout << "Number of pre-crack line segments: " << pre_crack_lines_.size()
                     << Core::IO::endl;
    }
  }

  // parse pre-crack parallelogram patches (3D): each entry has P0, P1, P2 (3 doubles each)
  {
    const Teuchos::ParameterList& pd_params = particle_params.sublist("PD");
    if (pd_params.isParameter("PRE_CRACK_PLANES"))
    {
      const auto& planes = pd_params.get<std::vector<Teuchos::ParameterList>>("PRE_CRACK_PLANES");
      for (const auto& entry : planes)
      {
        const auto& p0 = entry.get<std::vector<double>>("P0");
        const auto& p1 = entry.get<std::vector<double>>("P1");
        const auto& p2 = entry.get<std::vector<double>>("P2");
        pre_crack_planes_.push_back(
            {p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]});
      }
      Core::IO::cout << "Number of pre-crack planes: " << pre_crack_planes_.size()
                     << Core::IO::endl;
    }
  }

  // checks for the dimensionality of the problem
  {
    const auto constraint_type = particle_params.sublist("INITIAL AND BOUNDARY CONDITIONS")
                                     .get<Particle::Constraint>("CONSTRAINT");

    if (peridynamic_dimension_ == PeridynamicDimension::Peridynamic_2DPlaneStress ||
        peridynamic_dimension_ == PeridynamicDimension::Peridynamic_2DPlaneStrain)
    {
      FOUR_C_ASSERT_ALWAYS(constraint_type == Particle::Projection2D,
          "Plane stress or plane strain for peridynamic requested. CONSTRAINT must be set to "
          "Projection2D!");
    }

    if (constraint_type == Particle::Projection2D)
    {
      FOUR_C_ASSERT_ALWAYS(peridynamic_dimension_ != PeridynamicDimension::Peridynamic_3D,
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
    double* currentconnectedbonds_i =
        container_i->get_ptr_to_state(Particle::CurrentConnectedBonds, particle_i);

    const double* pos_j = container_j->get_ptr_to_state(Particle::Position, particle_j);
    const double* pdbodyid_j = container_j->get_ptr_to_state(Particle::PDBodyId, particle_j);
    double* initialconnectedbonds_j =
        container_j->get_ptr_to_state(Particle::InitialConnectedBonds, particle_j);
    double* currentconnectedbonds_j =
        container_j->get_ptr_to_state(Particle::CurrentConnectedBonds, particle_j);

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
        // always count towards initial bonds (including pre-cracked)
        initialconnectedbonds_i[0] += 1.0;
        initialconnectedbonds_j[0] += 1.0;

        // bonds crossing a pre-crack are initially broken
        if (bond_crosses_pre_crack(pos_i, pos_j)) continue;

        // count active (non-pre-cracked) bonds as current
        currentconnectedbonds_i[0] += 1.0;
        currentconnectedbonds_j[0] += 1.0;

        // get global id of particle i
        const int* globalid_i = container_i->get_ptr_to_global_id(particle_i);
        const int* globalid_j = container_j->get_ptr_to_global_id(particle_j);

        // creat the content of the bond between particles i & j
        Particle::LocalGlobalIndexTuple tuple_i =
            std::make_tuple(type_i, status_i, particle_i, globalid_i[0]);
        Particle::LocalGlobalIndexTuple tuple_j =
            std::make_tuple(type_j, status_j, particle_j, globalid_j[0]);

        bondlist_->push_back(std::make_pair(tuple_i, tuple_j));
      }
    }
  }

  Core::IO::cout << "Number of initialized peridynamic bonds on this proc: " << bondlist_->size()
                 << Core::IO::endl;
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
        case PeridynamicDimension::Peridynamic_3D:
          fac = (12.0 * (young_i[0] + young_j[0]) * 0.5) /
                (std::numbers::pi * horizon_pd_ * horizon_pd_ * horizon_pd_ * horizon_pd_) *
                stretch * calculate_volume_correction_factor(xi_norm) * dx_pd_ * dx_pd_ * dx_pd_ *
                dx_pd_ * dx_pd_ * dx_pd_;
          break;
        case PeridynamicDimension::Peridynamic_2DPlaneStress:
          fac = (9.0 * (young_i[0] + young_j[0]) * 0.5) /
                (std::numbers::pi * horizon_pd_ * horizon_pd_ * horizon_pd_) * stretch *
                calculate_volume_correction_factor(xi_norm) * dx_pd_ * dx_pd_ * dx_pd_ * dx_pd_;
          break;
        case PeridynamicDimension::Peridynamic_2DPlaneStrain:
          fac = (48.0 * (young_i[0] + young_j[0]) * 0.5) /
                (std::numbers::pi * 5.0 * horizon_pd_ * horizon_pd_ * horizon_pd_) * stretch *
                calculate_volume_correction_factor(xi_norm) * dx_pd_ * dx_pd_ * dx_pd_ * dx_pd_;
          break;
        default:
          std23::unreachable();
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
  if (!pd_neighbor_pairs.empty())
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

bool Particle::SPHPeridynamic::bond_crosses_pre_crack(
    const double* pos_i, const double* pos_j) const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHPeridynamic::bond_crosses_pre_crack");

  // --- line segments (3D) ---
  // Two segments in 3D are generally skew; a bond crosses a crack segment if the
  // minimum distance between the two segments is below a tolerance and both
  // parameters t (along bond) and u (along crack segment) lie in [0, 1].
  // Closest-approach parametric form:
  //   bond:   P(t) = pos_i + t*ab,  t in [0,1]
  //   crack:  Q(u) = c    + u*cd,   u in [0,1]
  //   w = pos_i - c;  A = dot(ab,ab);  B = dot(ab,cd);  C = dot(cd,cd)
  //   D = dot(ab,w);  E = dot(cd,w);   denom = A*C - B*B
  //   t = (B*E - C*D)/denom;  u = (A*E - B*D)/denom
  //   distance = |P(t) - Q(u)|  (must be zero for a true 3D intersection)

  for (const auto& crack_segment : pre_crack_lines_)
  {
    // fast AABB rejection
    if (!bond_aabb_overlaps_segment(pos_i, pos_j, crack_segment)) continue;

    // bond vector ab = pos_j - pos_i
    double ab[3] = {pos_j[0] - pos_i[0], pos_j[1] - pos_i[1], pos_j[2] - pos_i[2]};
    // crack segment vector cd = d - c
    double cd[3] = {crack_segment[3] - crack_segment[0], crack_segment[4] - crack_segment[1],
        crack_segment[5] - crack_segment[2]};
    // offset vector w = pos_i - c
    double w[3] = {
        pos_i[0] - crack_segment[0], pos_i[1] - crack_segment[1], pos_i[2] - crack_segment[2]};

    const double A = ParticleUtils::vec_dot(ab, ab);
    const double B = ParticleUtils::vec_dot(ab, cd);
    const double C = ParticleUtils::vec_dot(cd, cd);
    const double D = ParticleUtils::vec_dot(ab, w);
    const double E = ParticleUtils::vec_dot(cd, w);

    const double denom = A * C - B * B;
    if (std::abs(denom) < 1.0e-14) continue;  // segments parallel

    const double t = (B * E - C * D) / denom;
    const double u = (A * E - B * D) / denom;

    if (t < 0.0 || t > 1.0 || u < 0.0 || u > 1.0) continue;

    // distance vector between closest points: g = w + t*ab - u*cd
    double g[3] = {
        w[0] + t * ab[0] - u * cd[0], w[1] + t * ab[1] - u * cd[1], w[2] + t * ab[2] - u * cd[2]};

    // bond crosses crack if the two lines touch
    if (ParticleUtils::vec_dot(g, g) <= 1.0e-20) return true;
  }

  // --- parallelogram patches (3D) ---
  // Patch corners: p0, p1, p2; edges e1 = p1-p0, e2 = p2-p0; normal n = e1 x e2.
  // 1. Find bond-plane intersection: pos_i + t*(pos_j - pos_i), t in [0,1].
  // 2. Express intersection point q in local coords: q - p0 = s*e1 + r*e2, s,r in [0,1].
  for (const auto& patch : pre_crack_planes_)
  {
    // fast AABB rejection
    if (!bond_aabb_overlaps_patch(pos_i, pos_j, patch)) continue;

    const double p0[3] = {patch[0], patch[1], patch[2]};
    // edge vectors
    double e1[3] = {patch[3] - p0[0], patch[4] - p0[1], patch[5] - p0[2]};
    double e2[3] = {patch[6] - p0[0], patch[7] - p0[1], patch[8] - p0[2]};

    // normal n = e1 x e2
    double n[3];
    ParticleUtils::vec_set_cross(n, e1, e2);

    // bond direction d = pos_j - pos_i
    double d[3] = {pos_j[0] - pos_i[0], pos_j[1] - pos_i[1], pos_j[2] - pos_i[2]};

    const double denom = ParticleUtils::vec_dot(d, n);
    if (std::abs(denom) < 1.0e-14) continue;  // bond parallel to patch plane

    // offset from p0 to pos_i
    double p0_to_i[3] = {p0[0] - pos_i[0], p0[1] - pos_i[1], p0[2] - pos_i[2]};

    // parameter t along bond where it crosses the plane
    const double t = ParticleUtils::vec_dot(p0_to_i, n) / denom;
    if (t < 0.0 || t > 1.0) continue;  // intersection outside bond

    // intersection point q relative to p0: q = pos_i + t*d - p0
    double q[3] = {
        pos_i[0] + t * d[0] - p0[0], pos_i[1] + t * d[1] - p0[1], pos_i[2] + t * d[2] - p0[2]};

    // solve q = s*e1 + r*e2 via Gram matrix (works for non-orthogonal edges)
    const double e1e1 = ParticleUtils::vec_dot(e1, e1);
    const double e2e2 = ParticleUtils::vec_dot(e2, e2);
    const double e1e2 = ParticleUtils::vec_dot(e1, e2);
    const double qe1 = ParticleUtils::vec_dot(q, e1);
    const double qe2 = ParticleUtils::vec_dot(q, e2);

    const double det = e1e1 * e2e2 - e1e2 * e1e2;
    if (std::abs(det) < 1.0e-14) continue;  // degenerate patch

    const double s = (qe1 * e2e2 - qe2 * e1e2) / det;
    const double r = (qe2 * e1e1 - qe1 * e1e2) / det;

    if (s >= 0.0 && s <= 1.0 && r >= 0.0 && r <= 1.0) return true;
  }

  return false;
}

// -----------------------------------------------------------------------
// File-local geometry helper implementations
// -----------------------------------------------------------------------
namespace
{
  /*!
   * \brief Check whether two 1D intervals [min_a, max_a] and [min_b, max_b] overlap.
   *
   * Used as a per-dimension AABB rejection test before the full geometric intersection
   * calculation in bond_crosses_pre_crack().
   */
  bool intervals_overlap(
      const double min_a, const double max_a, const double min_b, const double max_b)
  {
    return min_a <= max_b && max_a >= min_b;
  }

  /*!
   * \brief Check whether the AABB of bond [pos_i, pos_j] overlaps the AABB of a crack
   * line segment stored as {c0x,c0y,c0z, c1x,c1y,c1z} in \p segment.
   */
  bool bond_aabb_overlaps_segment(
      const double* pos_i, const double* pos_j, const std::array<double, 6>& segment)
  {
    for (int dim = 0; dim < 3; ++dim)
    {
      if (!intervals_overlap(std::min(pos_i[dim], pos_j[dim]), std::max(pos_i[dim], pos_j[dim]),
              std::min(segment[dim], segment[dim + 3]), std::max(segment[dim], segment[dim + 3])))
        return false;
    }
    return true;
  }

  /*!
   * \brief Check whether the AABB of bond [pos_i, pos_j] overlaps the AABB of a crack
   * patch stored as {p0x,p0y,p0z, p1x,p1y,p1z, p2x,p2y,p2z} in \p patch.
   */
  bool bond_aabb_overlaps_patch(
      const double* pos_i, const double* pos_j, const std::array<double, 9>& patch)
  {
    for (int dim = 0; dim < 3; ++dim)
    {
      const double patch_min = std::min({patch[dim], patch[dim + 3], patch[dim + 6]});
      const double patch_max = std::max({patch[dim], patch[dim + 3], patch[dim + 6]});
      if (!intervals_overlap(std::min(pos_i[dim], pos_j[dim]), std::max(pos_i[dim], pos_j[dim]),
              patch_min, patch_max))
        return false;
    }
    return true;
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE
