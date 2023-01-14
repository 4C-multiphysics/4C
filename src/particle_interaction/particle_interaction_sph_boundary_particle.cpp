/*---------------------------------------------------------------------------*/
/*! \file
\brief boundary particle handler for smoothed particle hydrodynamics (SPH) interactions
\level 3
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "particle_interaction_sph_boundary_particle.H"

#include "particle_interaction_sph_neighbor_pairs.H"

#include "particle_interaction_utils.H"

#include "particle_engine_interface.H"
#include "particle_engine_container.H"

#include "lib_dserror.H"

#include <Teuchos_TimeMonitor.hpp>

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEINTERACTION::SPHBoundaryParticleBase::SPHBoundaryParticleBase(
    const Teuchos::ParameterList& params)
    : params_sph_(params)
{
  // empty constructor
}

void PARTICLEINTERACTION::SPHBoundaryParticleBase::Init()
{
  // init with potential fluid particle types
  fluidtypes_ = {PARTICLEENGINE::Phase1, PARTICLEENGINE::Phase2, PARTICLEENGINE::DirichletPhase,
      PARTICLEENGINE::NeumannPhase};

  // init with potential boundary particle types
  boundarytypes_ = {PARTICLEENGINE::BoundaryPhase, PARTICLEENGINE::RigidPhase};
}

void PARTICLEINTERACTION::SPHBoundaryParticleBase::Setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<PARTICLEINTERACTION::SPHNeighborPairs> neighborpairs)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->GetParticleContainerBundle();

  // set neighbor pair handler
  neighborpairs_ = neighborpairs;

  // update with actual fluid particle types
  const auto fluidtypes = fluidtypes_;
  for (const auto& type_i : fluidtypes)
    if (not particlecontainerbundle_->GetParticleTypes().count(type_i)) fluidtypes_.erase(type_i);

  // update with actual boundary particle types
  const auto boundarytypes = boundarytypes_;
  for (const auto& type_i : boundarytypes)
    if (not particlecontainerbundle_->GetParticleTypes().count(type_i))
      boundarytypes_.erase(type_i);

  // safety check
  if (boundarytypes_.empty())
    dserror("no boundary or rigid particles defined but a boundary particle formulation is set!");
}

PARTICLEINTERACTION::SPHBoundaryParticleAdami::SPHBoundaryParticleAdami(
    const Teuchos::ParameterList& params)
    : PARTICLEINTERACTION::SPHBoundaryParticleBase(params)
{
  // empty constructor
}

void PARTICLEINTERACTION::SPHBoundaryParticleAdami::Setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<PARTICLEINTERACTION::SPHNeighborPairs> neighborpairs)
{
  // call base class setup
  SPHBoundaryParticleBase::Setup(particleengineinterface, neighborpairs);

  // setup modified states of ghosted boundary particles to refresh
  {
    std::vector<PARTICLEENGINE::StateEnum> states{
        PARTICLEENGINE::BoundaryPressure, PARTICLEENGINE::BoundaryVelocity};

    for (const auto& type_i : boundarytypes_)
      boundarystatestorefresh_.push_back(std::make_pair(type_i, states));
  }

  // determine size of vectors indexed by particle types
  const int typevectorsize = *(--boundarytypes_.end()) + 1;

  // allocate memory to hold contributions of neighboring particles
  sumj_Wij_.resize(typevectorsize);
  sumj_press_j_Wij_.resize(typevectorsize);
  sumj_dens_j_r_ij_Wij_.resize(typevectorsize);
  sumj_vel_j_Wij_.resize(typevectorsize);
}

void PARTICLEINTERACTION::SPHBoundaryParticleAdami::InitBoundaryParticleStates(
    std::vector<double>& gravity)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "PARTICLEINTERACTION::SPHBoundaryParticleAdami::InitBoundaryParticleStates");

  // iterate over boundary particle types
  for (const auto& type_i : boundarytypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->GetSpecificContainer(type_i, PARTICLEENGINE::Owned);

    // get number of particles stored in container
    const int particlestored = container_i->ParticlesStored();

    // allocate memory
    sumj_Wij_[type_i].assign(particlestored, 0.0);
    sumj_press_j_Wij_[type_i].assign(particlestored, 0.0);
    sumj_dens_j_r_ij_Wij_[type_i].assign(particlestored, std::vector<double>(3, 0.0));
    sumj_vel_j_Wij_[type_i].assign(particlestored, std::vector<double>(3, 0.0));
  }

  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->GetRelevantParticlePairIndicesForDisjointCombination(
      boundarytypes_, fluidtypes_, relindices);

  // iterate over relevant particle pairs
  for (const int particlepairindex : relindices)
  {
    const SPHParticlePair& particlepair =
        neighborpairs_->GetRefToParticlePairData()[particlepairindex];

    // access values of local index tuples of particle i and j
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    PARTICLEENGINE::TypeEnum type_j;
    PARTICLEENGINE::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // evaluate contribution of neighboring fluid particle j
    if (boundarytypes_.count(type_i))
    {
      // get container of owned particles
      PARTICLEENGINE::ParticleContainer* container_j =
          particlecontainerbundle_->GetSpecificContainer(type_j, status_j);

      // get pointer to particle states
      const double* vel_j = container_j->GetPtrToState(PARTICLEENGINE::Velocity, particle_j);
      const double* dens_j = container_j->GetPtrToState(PARTICLEENGINE::Density, particle_j);
      const double* press_j = container_j->GetPtrToState(PARTICLEENGINE::Pressure, particle_j);

      // sum contribution of neighboring particle j
      sumj_Wij_[type_i][particle_i] += particlepair.Wij_;
      sumj_press_j_Wij_[type_i][particle_i] += press_j[0] * particlepair.Wij_;

      const double fac = dens_j[0] * particlepair.absdist_ * particlepair.Wij_;
      UTILS::VecAddScale(&sumj_dens_j_r_ij_Wij_[type_i][particle_i][0], fac, particlepair.e_ij_);

      UTILS::VecAddScale(&sumj_vel_j_Wij_[type_i][particle_i][0], particlepair.Wij_, vel_j);
    }

    // evaluate contribution of neighboring fluid particle i
    if (boundarytypes_.count(type_j) and status_j == PARTICLEENGINE::Owned)
    {
      // get container of owned particles
      PARTICLEENGINE::ParticleContainer* container_i =
          particlecontainerbundle_->GetSpecificContainer(type_i, status_i);

      // get pointer to particle states
      const double* vel_i = container_i->GetPtrToState(PARTICLEENGINE::Velocity, particle_i);
      const double* dens_i = container_i->GetPtrToState(PARTICLEENGINE::Density, particle_i);
      const double* press_i = container_i->GetPtrToState(PARTICLEENGINE::Pressure, particle_i);

      // sum contribution of neighboring particle i
      sumj_Wij_[type_j][particle_j] += particlepair.Wji_;
      sumj_press_j_Wij_[type_j][particle_j] += press_i[0] * particlepair.Wji_;

      const double fac = -dens_i[0] * particlepair.absdist_ * particlepair.Wji_;
      UTILS::VecAddScale(&sumj_dens_j_r_ij_Wij_[type_j][particle_j][0], fac, particlepair.e_ij_);

      UTILS::VecAddScale(&sumj_vel_j_Wij_[type_j][particle_j][0], particlepair.Wji_, vel_i);
    }
  }

  // iterate over boundary particle types
  for (const auto& type_i : boundarytypes_)
  {
    // get container of owned particles
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->GetSpecificContainer(type_i, PARTICLEENGINE::Owned);

    // clear modified boundary particle states
    container_i->ClearState(PARTICLEENGINE::BoundaryPressure);
    container_i->ClearState(PARTICLEENGINE::BoundaryVelocity);

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->ParticlesStored(); ++particle_i)
    {
      // set modified boundary particle states
      if (sumj_Wij_[type_i][particle_i] > 0.0)
      {
        // get pointer to particle states
        const double* vel_i = container_i->GetPtrToState(PARTICLEENGINE::Velocity, particle_i);
        const double* acc_i = container_i->GetPtrToState(PARTICLEENGINE::Acceleration, particle_i);
        double* boundarypress_i =
            container_i->GetPtrToState(PARTICLEENGINE::BoundaryPressure, particle_i);
        double* boundaryvel_i =
            container_i->GetPtrToState(PARTICLEENGINE::BoundaryVelocity, particle_i);

        // get relative acceleration of boundary particle
        double relacc[3];
        UTILS::VecSet(relacc, &gravity[0]);
        UTILS::VecSub(relacc, acc_i);

        const double inv_sumj_Wij = 1.0 / sumj_Wij_[type_i][particle_i];

        // set modified boundary pressure
        boundarypress_i[0] =
            (sumj_press_j_Wij_[type_i][particle_i] +
                UTILS::VecDot(relacc, &sumj_dens_j_r_ij_Wij_[type_i][particle_i][0])) *
            inv_sumj_Wij;

        // set modified boundary velocity
        UTILS::VecSetScale(boundaryvel_i, 2.0, vel_i);
        UTILS::VecAddScale(boundaryvel_i, -inv_sumj_Wij, &sumj_vel_j_Wij_[type_i][particle_i][0]);
      }
    }
  }

  // refresh modified states of ghosted boundary particles
  particleengineinterface_->RefreshParticlesOfSpecificStatesAndTypes(boundarystatestorefresh_);
}
