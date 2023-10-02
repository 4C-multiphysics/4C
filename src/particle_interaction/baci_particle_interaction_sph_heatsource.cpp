/*---------------------------------------------------------------------------*/
/*! \file
\brief heat source handler for smoothed particle hydrodynamics (SPH) interactions
\level 3
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "baci_particle_interaction_sph_heatsource.H"

#include "baci_lib_function.H"
#include "baci_lib_globalproblem.H"
#include "baci_particle_engine_container.H"
#include "baci_particle_engine_interface.H"
#include "baci_particle_interaction_material_handler.H"
#include "baci_particle_interaction_sph_neighbor_pairs.H"
#include "baci_particle_interaction_utils.H"
#include "baci_utils_exceptions.H"

#include <Teuchos_TimeMonitor.hpp>

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEINTERACTION::SPHHeatSourceBase::SPHHeatSourceBase(const Teuchos::ParameterList& params)
    : params_sph_(params), heatsourcefctnumber_(params.get<int>("HEATSOURCE_FUNCT"))
{
  // empty constructor
}

void PARTICLEINTERACTION::SPHHeatSourceBase::Init()
{
  // nothing to do
}

void PARTICLEINTERACTION::SPHHeatSourceBase::Setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<PARTICLEINTERACTION::MaterialHandler> particlematerial,
    const std::shared_ptr<PARTICLEINTERACTION::SPHNeighborPairs> neighborpairs)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->GetParticleContainerBundle();

  // set particle material handler
  particlematerial_ = particlematerial;

  // set neighbor pair handler
  neighborpairs_ = neighborpairs;

  // determine size of vectors indexed by particle types
  const int typevectorsize = *(--particlecontainerbundle_->GetParticleTypes().end()) + 1;

  // allocate memory to hold particle types
  thermomaterial_.resize(typevectorsize);

  // iterate over particle types
  for (const auto& type_i : particlecontainerbundle_->GetParticleTypes())
    thermomaterial_[type_i] = dynamic_cast<const MAT::PAR::ParticleMaterialThermo*>(
        particlematerial_->GetPtrToParticleMatParameter(type_i));

  // set of potential absorbing particle types
  std::set<PARTICLEENGINE::TypeEnum> potentialabsorbingtypes = {
      PARTICLEENGINE::Phase1, PARTICLEENGINE::Phase2, PARTICLEENGINE::RigidPhase};

  // iterate over particle types
  for (const auto& type_i : particlecontainerbundle_->GetParticleTypes())
  {
    // determine absorbing particle types
    if (thermomaterial_[type_i]->thermalAbsorptivity_ > 0.0)
    {
      // safety check
      if (not potentialabsorbingtypes.count(type_i))
        dserror("thermal absorptivity for particles of type '%s' not possible!",
            PARTICLEENGINE::EnumToTypeName(type_i).c_str());

      absorbingtypes_.insert(type_i);
    }
    // determine non-absorbing particle types
    else if (potentialabsorbingtypes.count(type_i))
    {
      nonabsorbingtypes_.insert(type_i);
    }
  }
}

PARTICLEINTERACTION::SPHHeatSourceVolume::SPHHeatSourceVolume(const Teuchos::ParameterList& params)
    : PARTICLEINTERACTION::SPHHeatSourceBase(params)
{
  // empty constructor
}

void PARTICLEINTERACTION::SPHHeatSourceVolume::EvaluateHeatSource(const double& evaltime) const
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLEINTERACTION::SPHHeatSourceVolume::EvaluateHeatSource");

  // init vector containing evaluated function
  std::vector<double> funct(1);

  // get reference to function
  const auto& function = DRT::Problem::Instance()->FunctionById<DRT::UTILS::FunctionOfSpaceTime>(
      heatsourcefctnumber_ - 1);

  // safety check
  if (function.NumberComponents() != 1)
    dserror("dimension of function defining heat source is not one!");

  // iterate over absorbing particle types
  for (const auto& type_i : absorbingtypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->GetSpecificContainer(type_i, PARTICLEENGINE::Owned);

    // get material for current particle type
    const MAT::PAR::ParticleMaterialBase* basematerial_i =
        particlematerial_->GetPtrToParticleMatParameter(type_i);

    const MAT::PAR::ParticleMaterialThermo* thermomaterial_i = thermomaterial_[type_i];

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->ParticlesStored(); ++particle_i)
    {
      // get pointer to particle states
      const double* dens_i = (container_i->HaveStoredState(PARTICLEENGINE::Density))
                                 ? container_i->GetPtrToState(PARTICLEENGINE::Density, particle_i)
                                 : &(basematerial_i->initDensity_);

      const double* pos_i = container_i->GetPtrToState(PARTICLEENGINE::Position, particle_i);
      double* tempdot_i = container_i->GetPtrToState(PARTICLEENGINE::TemperatureDot, particle_i);

      // evaluate function defining heat source
      funct = function.EvaluateTimeDerivative(pos_i, evaltime, 0, 0);

      // add contribution of heat source
      tempdot_i[0] += thermomaterial_i->thermalAbsorptivity_ * funct[0] *
                      thermomaterial_i->invThermalCapacity_ / dens_i[0];
    }
  }
}

PARTICLEINTERACTION::SPHHeatSourceSurface::SPHHeatSourceSurface(
    const Teuchos::ParameterList& params)
    : PARTICLEINTERACTION::SPHHeatSourceBase(params), eval_direction_(false)
{
  // empty constructor
}

void PARTICLEINTERACTION::SPHHeatSourceSurface::Init()
{
  // call base class init
  SPHHeatSourceBase::Init();

  // init heat source direction vector
  double value;
  std::istringstream directionstream(
      Teuchos::getNumericStringParameter(params_sph_, "HEATSOURCE_DIRECTION"));

  while (directionstream >> value) direction_.push_back(value);

  // safety check
  if (static_cast<int>(direction_.size()) != 3)
    dserror("dimension (dim = %d) of heat source direction vector is wrong!",
        static_cast<int>(direction_.size()));

  // normalize heat source direction vector
  const double direction_norm = UTILS::VecNormTwo(direction_.data());
  if (direction_norm > 0.0)
  {
    eval_direction_ = true;
    UTILS::VecSetScale(direction_.data(), 1.0 / direction_norm, direction_.data());
  }
}

void PARTICLEINTERACTION::SPHHeatSourceSurface::EvaluateHeatSource(const double& evaltime) const
{
  TEUCHOS_FUNC_TIME_MONITOR("PARTICLEINTERACTION::SPHHeatSourceSurface::EvaluateHeatSource");

  // determine size of vectors indexed by particle types
  const int typevectorsize = *(--absorbingtypes_.end()) + 1;

  // colorfield gradient of absorbing interface particles
  std::vector<std::vector<std::vector<double>>> cfg_i(typevectorsize);

  // iterate over absorbing particle types
  for (const auto& type_i : absorbingtypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->GetSpecificContainer(type_i, PARTICLEENGINE::Owned);

    // get number of particles stored in container
    const int particlestored = container_i->ParticlesStored();

    // allocate memory
    cfg_i[type_i].assign(particlestored, std::vector<double>(3, 0.0));
  }

  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->GetRelevantParticlePairIndicesForDisjointCombination(
      absorbingtypes_, nonabsorbingtypes_, relindices);

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

    // get corresponding particle containers
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->GetSpecificContainer(type_i, status_i);

    PARTICLEENGINE::ParticleContainer* container_j =
        particlecontainerbundle_->GetSpecificContainer(type_j, status_j);

    // get material for particle types
    const MAT::PAR::ParticleMaterialBase* material_i =
        particlematerial_->GetPtrToParticleMatParameter(type_i);

    const MAT::PAR::ParticleMaterialBase* material_j =
        particlematerial_->GetPtrToParticleMatParameter(type_j);

    // get pointer to particle states
    const double* mass_i = container_i->GetPtrToState(PARTICLEENGINE::Mass, particle_i);

    const double* dens_i = container_i->HaveStoredState(PARTICLEENGINE::Density)
                               ? container_i->GetPtrToState(PARTICLEENGINE::Density, particle_i)
                               : &(material_i->initDensity_);

    const double* mass_j = container_j->GetPtrToState(PARTICLEENGINE::Mass, particle_j);

    const double* dens_j = container_j->HaveStoredState(PARTICLEENGINE::Density)
                               ? container_j->GetPtrToState(PARTICLEENGINE::Density, particle_j)
                               : &(material_j->initDensity_);

    // (current) volume of particle i and j
    const double V_i = mass_i[0] / dens_i[0];
    const double V_j = mass_j[0] / dens_j[0];

    const double fac = (UTILS::Pow<2>(V_i) + UTILS::Pow<2>(V_j)) / (dens_i[0] + dens_j[0]);

    // evaluate contribution of neighboring particle j
    if (absorbingtypes_.count(type_i))
    {
      // sum contribution of neighboring particle j
      UTILS::VecAddScale(cfg_i[type_i][particle_i].data(),
          dens_i[0] / V_i * fac * particlepair.dWdrij_, particlepair.e_ij_);
    }

    // evaluate contribution of neighboring particle i
    if (absorbingtypes_.count(type_j) and status_j == PARTICLEENGINE::Owned)
    {
      // sum contribution of neighboring particle i
      UTILS::VecAddScale(cfg_i[type_j][particle_j].data(),
          -dens_j[0] / V_j * fac * particlepair.dWdrji_, particlepair.e_ij_);
    }
  }

  // init vector containing evaluated function
  std::vector<double> funct(1);

  // get reference to function
  const auto& function = DRT::Problem::Instance()->FunctionById<DRT::UTILS::FunctionOfSpaceTime>(
      heatsourcefctnumber_ - 1);

  // safety check
  if (function.NumberComponents() != 1)
    dserror("dimension of function defining heat source is not one!");

  // iterate over absorbing particle types
  for (const auto& type_i : absorbingtypes_)
  {
    // get container of owned particles of current particle type
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->GetSpecificContainer(type_i, PARTICLEENGINE::Owned);

    // get material for current particle type
    const MAT::PAR::ParticleMaterialBase* basematerial_i =
        particlematerial_->GetPtrToParticleMatParameter(type_i);

    const MAT::PAR::ParticleMaterialThermo* thermomaterial_i = thermomaterial_[type_i];

    // iterate over particles in container
    for (int particle_i = 0; particle_i < container_i->ParticlesStored(); ++particle_i)
    {
      // norm of colorfield gradient of absorbing interface particles
      const double f_i = UTILS::VecNormTwo(cfg_i[type_i][particle_i].data());

      // no heat source contribution to current particle
      if (not(f_i > 0.0)) continue;

      // projection of colorfield gradient with heat source direction
      const double f_i_proj =
          eval_direction_ ? -UTILS::VecDot(direction_.data(), cfg_i[type_i][particle_i].data())
                          : f_i;

      // heat source contribution only for surface opposing heat source
      if (f_i_proj < 0.0) continue;

      // get pointer to particle states
      const double* dens_i = (container_i->HaveStoredState(PARTICLEENGINE::Density))
                                 ? container_i->GetPtrToState(PARTICLEENGINE::Density, particle_i)
                                 : &(basematerial_i->initDensity_);

      const double* pos_i = container_i->GetPtrToState(PARTICLEENGINE::Position, particle_i);
      double* tempdot_i = container_i->GetPtrToState(PARTICLEENGINE::TemperatureDot, particle_i);

      // evaluate function defining heat source
      funct = function.EvaluateTimeDerivative(pos_i, evaltime, 0, 0);

      // add contribution of heat source
      tempdot_i[0] += f_i_proj * thermomaterial_i->thermalAbsorptivity_ * funct[0] *
                      thermomaterial_i->invThermalCapacity_ / dens_i[0];
    }
  }
}
