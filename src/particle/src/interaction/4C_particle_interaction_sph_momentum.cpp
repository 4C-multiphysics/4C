// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph_momentum.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_io_visualization_manager.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_material_handler.hpp"
#include "4C_particle_interaction_runtime_writer.hpp"
#include "4C_particle_interaction_sph_artificialviscosity.hpp"
#include "4C_particle_interaction_sph_equationofstate.hpp"
#include "4C_particle_interaction_sph_equationofstate_bundle.hpp"
#include "4C_particle_interaction_sph_kernel.hpp"
#include "4C_particle_interaction_sph_momentum_formulation.hpp"
#include "4C_particle_interaction_sph_neighbor_pairs.hpp"
#include "4C_particle_interaction_sph_virtual_wall_particle.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_particle_wall_datastate.hpp"
#include "4C_particle_wall_interface.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
Particle::SPHMomentum::SPHMomentum(const Teuchos::ParameterList& params)
    : params_sph_(params),
      boundaryparticleinteraction_(Teuchos::getIntegralValue<Particle::BoundaryParticleInteraction>(
          params_sph_, "BOUNDARYPARTICLEINTERACTION")),
      transportvelocityformulation_(
          Teuchos::getIntegralValue<Particle::TransportVelocityFormulation>(
              params_sph_, "TRANSPORTVELOCITYFORMULATION")),
      writeparticlewallinteraction_(params_sph_.get<bool>("WRITE_PARTICLE_WALL_INTERACTION")),
      allfluidtypes_(
          {Particle::Phase1, Particle::Phase2, Particle::DirichletPhase, Particle::NeumannPhase}),
      intfluidtypes_({Particle::Phase1, Particle::Phase2, Particle::NeumannPhase}),
      purefluidtypes_({Particle::Phase1, Particle::Phase2}),
      boundarytypes_({Particle::BoundaryPhase, Particle::RigidPhase})
{
  init_momentum_formulation_handler();

  artificialviscosity_ = std::make_unique<Particle::SPHArtificialViscosity>();
}

Particle::SPHMomentum::~SPHMomentum() = default;

void Particle::SPHMomentum::setup(
    const std::shared_ptr<Particle::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<Particle::WallHandlerInterface> particlewallinterface,
    const std::shared_ptr<Particle::SPHKernelBase> kernel,
    const std::shared_ptr<Particle::MaterialHandler> particlematerial,
    const std::shared_ptr<Particle::InteractionWriter> particleinteractionwriter,
    const std::shared_ptr<Particle::SPHEquationOfStateBundle> equationofstatebundle,
    const std::shared_ptr<Particle::SPHNeighborPairs> neighborpairs,
    const std::shared_ptr<Particle::SPHVirtualWallParticle> virtualwallparticle)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->get_particle_container_bundle();

  // set interface to particle wall handler
  particlewallinterface_ = particlewallinterface;

  // set kernel handler
  kernel_ = kernel;

  // set particle material handler
  particlematerial_ = particlematerial;

  // set particle interaction writer
  particleinteractionwriter_ = particleinteractionwriter;

  // setup particle interaction writer
  setup_particle_interaction_writer();

  // set equation of state handler
  equationofstatebundle_ = equationofstatebundle;

  // set neighbor pair handler
  neighborpairs_ = neighborpairs;

  // set virtual wall particle handler
  virtualwallparticle_ = virtualwallparticle;

  // update with actual fluid particle types
  const auto allfluidtypes = allfluidtypes_;
  for (const auto& type_i : allfluidtypes)
    if (not particlecontainerbundle_->get_particle_types().contains(type_i))
      allfluidtypes_.erase(type_i);

  const auto intfluidtypes = intfluidtypes_;
  for (const auto& type_i : intfluidtypes)
    if (not particlecontainerbundle_->get_particle_types().contains(type_i))
      intfluidtypes_.erase(type_i);

  const auto purefluidtypes = purefluidtypes_;
  for (const auto& type_i : purefluidtypes)
    if (not particlecontainerbundle_->get_particle_types().contains(type_i))
      purefluidtypes_.erase(type_i);

  // update with actual boundary particle types
  const auto boundarytypes = boundarytypes_;
  for (const auto& type_i : boundarytypes)
    if (not particlecontainerbundle_->get_particle_types().contains(type_i))
      boundarytypes_.erase(type_i);

  // determine size of vectors indexed by particle types
  const int typevectorsize = *(--particlecontainerbundle_->get_particle_types().end()) + 1;

  // allocate memory to hold particle types
  fluidmaterial_.resize(typevectorsize);

  // iterate over all fluid particle types
  for (const auto& type_i : allfluidtypes_)
  {
    fluidmaterial_[type_i] = dynamic_cast<const Mat::PAR::ParticleMaterialSPHFluid*>(
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i));
  }
}

void Particle::SPHMomentum::insert_particle_states_of_particle_types(
    std::map<Particle::TypeEnum, std::set<Particle::StateEnum>>& particlestatestotypes) const
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    // get type of particles
    Particle::TypeEnum type_i = typeIt.first;

    // set of particle states for current particle type
    std::set<Particle::StateEnum>& particlestates = typeIt.second;

    // current particle type is not a pure fluid particle type
    if (not purefluidtypes_.contains(type_i)) continue;

    // additional states for transport velocity formulation
    if (transportvelocityformulation_ !=
        Particle::TransportVelocityFormulation::NoTransportVelocity)
      particlestates.insert({Particle::ModifiedVelocity, Particle::ModifiedAcceleration});
  }
}

void Particle::SPHMomentum::add_acceleration_contribution() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHMomentum::add_acceleration_contribution");

  // momentum equation (particle contribution)
  momentum_equation_particle_contribution();

  // momentum equation (particle-boundary contribution)
  momentum_equation_particle_boundary_contribution();

  // momentum equation (particle-wall contribution)
  if (virtualwallparticle_) momentum_equation_particle_wall_contribution();
}

void Particle::SPHMomentum::init_momentum_formulation_handler()
{
  // get type of smoothed particle hydrodynamics momentum formulation
  auto momentumformulationtype = Teuchos::getIntegralValue<Particle::MomentumFormulationType>(
      params_sph_, "MOMENTUMFORMULATION");

  // create momentum formulation handler
  switch (momentumformulationtype)
  {
    case Particle::AdamiMomentumFormulation:
    {
      momentumformulation_ = std::unique_ptr<Particle::SPHMomentumFormulationAdami>(
          new Particle::SPHMomentumFormulationAdami());
      break;
    }
    case Particle::MonaghanMomentumFormulation:
    {
      momentumformulation_ = std::unique_ptr<Particle::SPHMomentumFormulationMonaghan>(
          new Particle::SPHMomentumFormulationMonaghan());
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown acceleration formulation type!");
      break;
    }
  }
}

void Particle::SPHMomentum::setup_particle_interaction_writer()
{
  // register specific runtime output writer
  if (writeparticlewallinteraction_)
    particleinteractionwriter_->register_specific_runtime_output_writer("particle-wall-momentum");
}

void Particle::SPHMomentum::momentum_equation_particle_contribution() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHMomentum::momentum_equation_particle_contribution");

  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_pair_indices_for_equal_combination(
      allfluidtypes_, relindices);

  // iterate over relevant particle pairs
  for (const int particlepairindex : relindices)
  {
    const SPHParticlePair& particlepair =
        neighborpairs_->get_ref_to_particle_pair_data()[particlepairindex];

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
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    Particle::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // get material for particle types
    const Mat::PAR::ParticleMaterialSPHFluid* material_i = fluidmaterial_[type_i];
    const Mat::PAR::ParticleMaterialSPHFluid* material_j = fluidmaterial_[type_j];

    // get pointer to particle states
    const double* rad_i = container_i->get_ptr_to_state(Particle::Radius, particle_i);
    const double* mass_i = container_i->get_ptr_to_state(Particle::Mass, particle_i);
    const double* dens_i = container_i->get_ptr_to_state(Particle::Density, particle_i);
    const double* press_i = container_i->get_ptr_to_state(Particle::Pressure, particle_i);
    const double* vel_i = container_i->get_ptr_to_state(Particle::Velocity, particle_i);

    double* acc_i = nullptr;
    if (intfluidtypes_.contains(type_i))
      acc_i = container_i->get_ptr_to_state(Particle::Acceleration, particle_i);

    const double* mod_vel_i =
        container_i->cond_get_ptr_to_state(Particle::ModifiedVelocity, particle_i);
    double* mod_acc_i =
        container_i->cond_get_ptr_to_state(Particle::ModifiedAcceleration, particle_i);

    // get pointer to particle states
    const double* rad_j = container_j->get_ptr_to_state(Particle::Radius, particle_j);
    const double* mass_j = container_j->get_ptr_to_state(Particle::Mass, particle_j);
    const double* dens_j = container_j->get_ptr_to_state(Particle::Density, particle_j);
    const double* press_j = container_j->get_ptr_to_state(Particle::Pressure, particle_j);
    const double* vel_j = container_j->get_ptr_to_state(Particle::Velocity, particle_j);

    double* acc_j = nullptr;
    if (intfluidtypes_.contains(type_j) and status_j == Particle::Owned)
      acc_j = container_j->get_ptr_to_state(Particle::Acceleration, particle_j);

    const double* mod_vel_j =
        container_j->cond_get_ptr_to_state(Particle::ModifiedVelocity, particle_j);

    double* mod_acc_j = nullptr;
    if (status_j == Particle::Owned)
      mod_acc_j = container_j->cond_get_ptr_to_state(Particle::ModifiedAcceleration, particle_j);

    // evaluate specific coefficient
    double speccoeff_ij(0.0);
    double speccoeff_ji(0.0);
    momentumformulation_->specific_coefficient(dens_i, dens_j, mass_i, mass_j, particlepair.dWdrij_,
        particlepair.dWdrji_, &speccoeff_ij, &speccoeff_ji);

    // evaluate pressure gradient
    momentumformulation_->pressure_gradient(dens_i, dens_j, press_i, press_j, speccoeff_ij,
        speccoeff_ji, particlepair.e_ij_, acc_i, acc_j);

    // evaluate shear forces
    {
      // get factor from kernel space dimension
      int kernelfac = 0;
      kernel_->kernel_space_dimension(kernelfac);
      kernelfac += 2;

      // evaluate shear forces
      momentumformulation_->shear_forces(dens_i, dens_j, vel_i, vel_j, kernelfac,
          material_i->dynamicViscosity_, material_j->dynamicViscosity_, material_i->bulkViscosity_,
          material_j->bulkViscosity_, particlepair.absdist_, speccoeff_ij, speccoeff_ji,
          particlepair.e_ij_, acc_i, acc_j);
    }

    // apply transport velocity formulation
    if (transportvelocityformulation_ ==
        Particle::TransportVelocityFormulation::StandardTransportVelocity)
    {
      // evaluate background pressure (standard formulation)
      momentumformulation_->standard_background_pressure(dens_i, dens_j,
          material_i->backgroundPressure_, material_j->backgroundPressure_, speccoeff_ij,
          speccoeff_ji, particlepair.e_ij_, mod_acc_i, mod_acc_j);

      // evaluate convection of momentum with relative velocity
      momentumformulation_->modified_velocity_contribution(dens_i, dens_j, vel_i, vel_j, mod_vel_i,
          mod_vel_j, speccoeff_ij, speccoeff_ji, particlepair.e_ij_, acc_i, acc_j);
    }
    else if (transportvelocityformulation_ ==
             Particle::TransportVelocityFormulation::GeneralizedTransportVelocity)
    {
      // modified first derivative of kernel
      const double mod_dWdrij =
          (mod_acc_i) ? kernel_->d_wdrij(particlepair.absdist_, kernel_->smoothing_length(rad_i[0]))
                      : 0.0;
      const double mod_dWdrji =
          (mod_acc_j) ? kernel_->d_wdrij(particlepair.absdist_, kernel_->smoothing_length(rad_j[0]))
                      : 0.0;

      // modified background pressure
      const double mod_bg_press_i =
          (mod_acc_i) ? std::min(std::abs(10.0 * press_i[0]), material_i->backgroundPressure_)
                      : 0.0;
      const double mod_bg_press_j =
          (mod_acc_j) ? std::min(std::abs(10.0 * press_j[0]), material_j->backgroundPressure_)
                      : 0.0;

      // evaluate background pressure (generalized formulation)
      momentumformulation_->generalized_background_pressure(dens_i, dens_j, mass_i, mass_j,
          mod_bg_press_i, mod_bg_press_j, mod_dWdrij, mod_dWdrji, particlepair.e_ij_, mod_acc_i,
          mod_acc_j);

      // evaluate convection of momentum with relative velocity
      momentumformulation_->modified_velocity_contribution(dens_i, dens_j, vel_i, vel_j, mod_vel_i,
          mod_vel_j, speccoeff_ij, speccoeff_ji, particlepair.e_ij_, acc_i, acc_j);
    }

    // evaluate artificial viscosity
    if (material_i->artificialViscosity_ > 0.0 or material_j->artificialViscosity_ > 0.0)
    {
      // particle averaged smoothing length
      const double h_ij =
          0.5 * (kernel_->smoothing_length(rad_i[0]) + kernel_->smoothing_length(rad_j[0]));

      // get speed of sound
      const double c_i = material_i->speed_of_sound();
      const double c_j = (type_i == type_j) ? c_i : material_j->speed_of_sound();

      // particle averaged speed of sound
      const double c_ij = 0.5 * (c_i + c_j);

      // particle averaged density
      const double dens_ij = 0.5 * (dens_i[0] + dens_j[0]);

      // evaluate artificial viscosity
      artificialviscosity_->artificial_viscosity(vel_i, vel_j, mass_i, mass_j,
          material_i->artificialViscosity_, material_j->artificialViscosity_, particlepair.dWdrij_,
          particlepair.dWdrji_, dens_ij, h_ij, c_ij, particlepair.absdist_, particlepair.e_ij_,
          acc_i, acc_j);
    }
  }
}

void Particle::SPHMomentum::momentum_equation_particle_boundary_contribution() const
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Particle::SPHMomentum::momentum_equation_particle_boundary_contribution");

  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_pair_indices_for_disjoint_combination(
      intfluidtypes_, boundarytypes_, relindices);

  // iterate over relevant particle pairs
  for (const int particlepairindex : relindices)
  {
    const SPHParticlePair& particlepair =
        neighborpairs_->get_ref_to_particle_pair_data()[particlepairindex];

    // access values of local index tuples of particle i and j
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    Particle::TypeEnum type_j;
    Particle::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // swap fluid particle and boundary particle
    const bool swapparticles = boundarytypes_.contains(type_i);
    if (swapparticles)
    {
      std::tie(type_i, status_i, particle_i) = particlepair.tuple_j_;
      std::tie(type_j, status_j, particle_j) = particlepair.tuple_i_;
    }

    // absolute distance between particles
    const double absdist = particlepair.absdist_;

    // versor from particle j to i
    double e_ij[3];
    ParticleUtils::vec_set(e_ij, particlepair.e_ij_);
    if (swapparticles) ParticleUtils::vec_scale(e_ij, -1.0);

    // first derivative of kernel
    const double dWdrij = (swapparticles) ? particlepair.dWdrji_ : particlepair.dWdrij_;

    // get corresponding particle containers
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    Particle::ParticleContainer* container_j =
        particlecontainerbundle_->get_specific_container(type_j, status_j);

    // get material for particle types
    const Mat::PAR::ParticleMaterialSPHFluid* material_i = fluidmaterial_[type_i];

    // get equation of state for particle types
    const Particle::SPHEquationOfStateBase* equationofstate_i =
        equationofstatebundle_->get_ptr_to_specific_equation_of_state(type_i);

    // get pointer to particle states
    const double* rad_i = container_i->get_ptr_to_state(Particle::Radius, particle_i);
    const double* mass_i = container_i->get_ptr_to_state(Particle::Mass, particle_i);
    const double* dens_i = container_i->get_ptr_to_state(Particle::Density, particle_i);
    const double* press_i = container_i->get_ptr_to_state(Particle::Pressure, particle_i);
    const double* vel_i = container_i->get_ptr_to_state(Particle::Velocity, particle_i);

    double* acc_i = nullptr;
    if (status_i == Particle::Owned)
      acc_i = container_i->get_ptr_to_state(Particle::Acceleration, particle_i);

    const double* mod_vel_i =
        container_i->cond_get_ptr_to_state(Particle::ModifiedVelocity, particle_i);

    double* mod_acc_i = nullptr;
    if (status_i == Particle::Owned)
      mod_acc_i = container_i->cond_get_ptr_to_state(Particle::ModifiedAcceleration, particle_i);

    // get pointer to boundary particle states
    const double* mass_j = container_i->get_ptr_to_state(Particle::Mass, particle_i);
    const double* press_j = container_j->get_ptr_to_state(Particle::BoundaryPressure, particle_j);
    const double* vel_j = container_j->get_ptr_to_state(Particle::BoundaryVelocity, particle_j);

    double temp_dens(0.0);
    temp_dens = equationofstate_i->pressure_to_density(press_j[0], material_i->initDensity_);
    const double* dens_j = &temp_dens;

    double* force_j = nullptr;
    if (status_j == Particle::Owned)
      force_j = container_j->cond_get_ptr_to_state(Particle::Force, particle_j);

    // contribution from neighboring boundary particle j
    double acc_ij[3] = {0.0, 0.0, 0.0};
    double mod_acc_ij[3] = {0.0, 0.0, 0.0};

    // evaluate specific coefficient
    double speccoeff_ij(0.0);
    momentumformulation_->specific_coefficient(
        dens_i, dens_j, mass_i, mass_j, dWdrij, 0.0, &speccoeff_ij, nullptr);

    // evaluate pressure gradient
    momentumformulation_->pressure_gradient(
        dens_i, dens_j, press_i, press_j, speccoeff_ij, 0.0, e_ij, acc_ij, nullptr);

    // evaluate shear forces
    if (boundaryparticleinteraction_ == Particle::NoSlipBoundaryParticle)
    {
      // get factor from kernel space dimension
      int kernelfac = 0;
      kernel_->kernel_space_dimension(kernelfac);
      kernelfac += 2;

      // evaluate shear forces
      momentumformulation_->shear_forces(dens_i, dens_j, vel_i, vel_j, kernelfac,
          material_i->dynamicViscosity_, material_i->dynamicViscosity_, material_i->bulkViscosity_,
          material_i->bulkViscosity_, absdist, speccoeff_ij, 0.0, e_ij, acc_ij, nullptr);
    }

    // apply transport velocity formulation
    if (transportvelocityformulation_ ==
        Particle::TransportVelocityFormulation::StandardTransportVelocity)
    {
      // evaluate background pressure (standard formulation)
      momentumformulation_->standard_background_pressure(dens_i, dens_j,
          material_i->backgroundPressure_, 0.0, speccoeff_ij, 0.0, e_ij, mod_acc_ij, nullptr);

      // evaluate convection of momentum with relative velocity
      momentumformulation_->modified_velocity_contribution(dens_i, dens_j, vel_i, vel_j, mod_vel_i,
          nullptr, speccoeff_ij, 0.0, e_ij, acc_ij, nullptr);
    }
    else if (transportvelocityformulation_ ==
             Particle::TransportVelocityFormulation::GeneralizedTransportVelocity)
    {
      // modified first derivative of kernel
      const double mod_dWdrij = kernel_->d_wdrij(absdist, kernel_->smoothing_length(rad_i[0]));

      // modified background pressure
      const double mod_bg_press_i =
          std::min(std::abs(10.0 * press_i[0]), material_i->backgroundPressure_);

      // evaluate background pressure (generalized formulation)
      momentumformulation_->generalized_background_pressure(dens_i, dens_j, mass_i, mass_j,
          mod_bg_press_i, 0.0, mod_dWdrij, 0.0, e_ij, mod_acc_ij, nullptr);

      // evaluate convection of momentum with relative velocity
      momentumformulation_->modified_velocity_contribution(dens_i, dens_j, vel_i, vel_j, mod_vel_i,
          nullptr, speccoeff_ij, 0.0, e_ij, acc_ij, nullptr);
    }

    // evaluate artificial viscosity
    if (boundaryparticleinteraction_ == Particle::NoSlipBoundaryParticle and
        material_i->artificialViscosity_ > 0.0)
    {
      // get smoothing length
      const double h_i = kernel_->smoothing_length(rad_i[0]);

      // get speed of sound
      const double c_i = material_i->speed_of_sound();

      // particle averaged density
      const double dens_ij = 0.5 * (dens_i[0] + dens_j[0]);

      // evaluate artificial viscosity
      artificialviscosity_->artificial_viscosity(vel_i, vel_j, mass_i, mass_j,
          material_i->artificialViscosity_, 0.0, dWdrij, 0.0, dens_ij, h_i, c_i, absdist, e_ij,
          acc_ij, nullptr);
    }

    // add contribution from neighboring boundary particle j
    if (acc_i) ParticleUtils::vec_add(acc_i, acc_ij);
    if (mod_acc_i) ParticleUtils::vec_add(mod_acc_i, mod_acc_ij);

    // add contribution to neighboring boundary particle j
    if (force_j) ParticleUtils::vec_add_scale(force_j, -mass_i[0], acc_ij);
  }
}

void Particle::SPHMomentum::momentum_equation_particle_wall_contribution() const
{
  TEUCHOS_FUNC_TIME_MONITOR("Particle::SPHMomentum::momentum_equation_particle_wall_contribution");

  // get wall data state container
  std::shared_ptr<Particle::WallDataState> walldatastate =
      particlewallinterface_->get_wall_data_state();

  // get reference to particle-wall pair data
  const SPHParticleWallPairData& particlewallpairdata =
      neighborpairs_->get_ref_to_particle_wall_pair_data();

  // get number of particle-wall pairs
  const int numparticlewallpairs = particlewallpairdata.size();

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
    attackpoints.reserve(3 * numparticlewallpairs);
    contactforces.reserve(3 * numparticlewallpairs);
    normaldirection.reserve(3 * numparticlewallpairs);
  }

  // get reference to weighted fluid particle pressure
  const std::vector<double>& weightedpressure = virtualwallparticle_->get_weighted_pressure();

  // get reference to weighted fluid particle pressure gradient
  const std::vector<std::vector<double>>& weightedpressuregradient =
      virtualwallparticle_->get_weighted_pressure_gradient();

  // get reference to weighted fluid particle distance vector
  const std::vector<std::vector<double>>& weighteddistancevector =
      virtualwallparticle_->get_weighted_distance_vector();

  // get reference to weighted fluid particle velocity
  const std::vector<std::vector<double>>& weightedvelocity =
      virtualwallparticle_->get_weighted_velocity();

  // get relevant particle wall pair indices for specific particle types
  std::vector<int> relindices;
  neighborpairs_->get_relevant_particle_wall_pair_indices(intfluidtypes_, relindices);

  // iterate over relevant particle-wall pairs
  for (const int particlewallpairindex : relindices)
  {
    const SPHParticleWallPair& particlewallpair = particlewallpairdata[particlewallpairindex];

    // access values of local index tuple of particle i
    Particle::TypeEnum type_i;
    Particle::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlewallpair.tuple_i_;

    // get corresponding particle container
    Particle::ParticleContainer* container_i =
        particlecontainerbundle_->get_specific_container(type_i, status_i);

    // get material for particle types
    const Mat::PAR::ParticleMaterialSPHFluid* material_i = fluidmaterial_[type_i];

    // get equation of state for particle types
    const Particle::SPHEquationOfStateBase* equationofstate_i =
        equationofstatebundle_->get_ptr_to_specific_equation_of_state(type_i);

    // get pointer to particle states
    const double* pos_i = container_i->get_ptr_to_state(Particle::Position, particle_i);
    const double* rad_i = container_i->get_ptr_to_state(Particle::Radius, particle_i);
    const double* mass_i = container_i->get_ptr_to_state(Particle::Mass, particle_i);
    const double* dens_i = container_i->get_ptr_to_state(Particle::Density, particle_i);
    const double* press_i = container_i->get_ptr_to_state(Particle::Pressure, particle_i);
    const double* vel_i = container_i->get_ptr_to_state(Particle::Velocity, particle_i);
    double* acc_i = container_i->get_ptr_to_state(Particle::Acceleration, particle_i);

    const double* mod_vel_i =
        container_i->cond_get_ptr_to_state(Particle::ModifiedVelocity, particle_i);
    double* mod_acc_i =
        container_i->cond_get_ptr_to_state(Particle::ModifiedAcceleration, particle_i);

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

    // sum contribution from neighboring virtual particle k
    double sumk_acc_ik[3] = {0.0, 0.0, 0.0};
    double sumk_mod_acc_ik[3] = {0.0, 0.0, 0.0};

    // compute vector from wall contact point j to particle i
    double r_ij[3];
    ParticleUtils::vec_set_scale(r_ij, particlewallpair.absdist_, particlewallpair.e_ij_);

    // vector from weighted fluid particle positions l to wall contact point j
    double r_jl_weighted[3];
    ParticleUtils::vec_set(r_jl_weighted, weighteddistancevector[particlewallpairindex].data());

    // inverse normal distance from weighted fluid particle positions l to wall contact point j
    const double inv_norm_dist_jl_weighted =
        1.0 / ParticleUtils::vec_dot(r_jl_weighted, particlewallpair.e_ij_);

    // unit surface tangent vectors in wall contact point j
    double t_j_1[3];
    double t_j_2[3];
    ParticleUtils::unit_surface_tangents(particlewallpair.e_ij_, t_j_1, t_j_2);

    // iterate over virtual particles
    for (const std::vector<double>& virtualparticle :
        virtualwallparticle_->get_relative_positions_of_virtual_particles())
    {
      // vector from virtual particle k to wall contact point j
      double r_jk[3];
      ParticleUtils::vec_set_scale(r_jk, virtualparticle[0], particlewallpair.e_ij_);
      ParticleUtils::vec_add_scale(r_jk, virtualparticle[1], t_j_1);
      ParticleUtils::vec_add_scale(r_jk, virtualparticle[2], t_j_2);

      // vector from virtual particle k to particle i
      double r_ik[3];
      ParticleUtils::vec_set(r_ik, r_ij);
      ParticleUtils::vec_add(r_ik, r_jk);

      // absolute distance between virtual particle k and particle i
      const double absdist = ParticleUtils::vec_norm_two(r_ik);

      // virtual particle within interaction distance
      if (absdist < rad_i[0])
      {
        // vector from weighted fluid particle positions l to virtual particle k
        double r_kl_weighted[3];
        ParticleUtils::vec_set(r_kl_weighted, r_jl_weighted);
        ParticleUtils::vec_sub(r_kl_weighted, r_jk);

        // get pointer to virtual particle states
        const double* mass_k = container_i->get_ptr_to_state(Particle::Mass, particle_i);

        const double temp_press_k = weightedpressure[particlewallpairindex] +
                                    ParticleUtils::vec_dot(r_kl_weighted,
                                        weightedpressuregradient[particlewallpairindex].data());
        const double* press_k = &temp_press_k;

        const double temp_dens_k =
            equationofstate_i->pressure_to_density(press_k[0], material_i->initDensity_);
        const double* dens_k = &temp_dens_k;

        double temp_vel_k[3];
        double fac = -virtualparticle[0] * inv_norm_dist_jl_weighted;
        ParticleUtils::vec_set_scale(temp_vel_k, 1 + fac, vel_j);
        ParticleUtils::vec_add_scale(
            temp_vel_k, -fac, weightedvelocity[particlewallpairindex].data());
        const double* vel_k = temp_vel_k;

        // versor from virtual particle k to particle i
        double e_ik[3];
        ParticleUtils::vec_set_scale(e_ik, 1.0 / absdist, r_ik);

        // evaluate first derivative of kernel
        const double dWdrik = kernel_->d_wdrij(absdist, rad_i[0]);

        // evaluate specific coefficient
        double speccoeff_ik(0.0);
        momentumformulation_->specific_coefficient(
            dens_i, dens_k, mass_i, mass_k, dWdrik, 0.0, &speccoeff_ik, nullptr);

        // evaluate pressure gradient
        momentumformulation_->pressure_gradient(
            dens_i, dens_k, press_i, press_k, speccoeff_ik, 0.0, e_ik, sumk_acc_ik, nullptr);

        // evaluate shear forces
        if (boundaryparticleinteraction_ == Particle::NoSlipBoundaryParticle)
        {
          // get factor from kernel space dimension
          int kernelfac = 0;
          kernel_->kernel_space_dimension(kernelfac);
          kernelfac += 2;

          // evaluate shear forces
          momentumformulation_->shear_forces(dens_i, dens_k, vel_i, vel_k, kernelfac,
              material_i->dynamicViscosity_, material_i->dynamicViscosity_,
              material_i->bulkViscosity_, material_i->bulkViscosity_, absdist, speccoeff_ik, 0.0,
              e_ik, sumk_acc_ik, nullptr);
        }

        // apply transport velocity formulation
        if (transportvelocityformulation_ ==
            Particle::TransportVelocityFormulation::StandardTransportVelocity)
        {
          // evaluate background pressure (standard formulation)
          momentumformulation_->standard_background_pressure(dens_i, dens_k,
              material_i->backgroundPressure_, 0.0, speccoeff_ik, 0.0, e_ik, sumk_mod_acc_ik,
              nullptr);

          // evaluate convection of momentum with relative velocity
          momentumformulation_->modified_velocity_contribution(dens_i, dens_k, vel_i, vel_k,
              mod_vel_i, nullptr, speccoeff_ik, 0.0, e_ik, sumk_acc_ik, nullptr);
        }
        else if (transportvelocityformulation_ ==
                 Particle::TransportVelocityFormulation::GeneralizedTransportVelocity)
        {
          // modified first derivative of kernel
          const double mod_dWdrij = kernel_->d_wdrij(absdist, kernel_->smoothing_length(rad_i[0]));

          // modified background pressure
          const double mod_bg_press_i =
              std::min(std::abs(10.0 * press_i[0]), material_i->backgroundPressure_);

          // evaluate background pressure (generalized formulation)
          momentumformulation_->generalized_background_pressure(dens_i, dens_k, mass_i, mass_k,
              mod_bg_press_i, 0.0, mod_dWdrij, 0.0, e_ik, sumk_mod_acc_ik, nullptr);

          // evaluate convection of momentum with relative velocity
          momentumformulation_->modified_velocity_contribution(dens_i, dens_k, vel_i, vel_k,
              mod_vel_i, nullptr, speccoeff_ik, 0.0, e_ik, sumk_acc_ik, nullptr);
        }

        // evaluate artificial viscosity
        if (boundaryparticleinteraction_ == Particle::NoSlipBoundaryParticle and
            material_i->artificialViscosity_ > 0.0)
        {
          // get smoothing length
          const double h_i = kernel_->smoothing_length(rad_i[0]);

          // get speed of sound
          const double c_i = material_i->speed_of_sound();

          // particle averaged density
          const double dens_ij = 0.5 * (dens_i[0] + dens_k[0]);

          // evaluate artificial viscosity
          artificialviscosity_->artificial_viscosity(vel_i, vel_k, mass_i, mass_k,
              material_i->artificialViscosity_, 0.0, dWdrik, 0.0, dens_ij, h_i, c_i, absdist, e_ik,
              sumk_acc_ik, nullptr);
        }
      }
    }

    // add contribution from neighboring virtual particle k
    ParticleUtils::vec_add(acc_i, sumk_acc_ik);
    if (mod_acc_i) ParticleUtils::vec_add(mod_acc_i, sumk_mod_acc_ik);

    // calculation of wall contact force
    double wallcontactforce[3] = {0.0, 0.0, 0.0};
    if (writeinteractionoutput or walldatastate->get_force_col() != nullptr)
      ParticleUtils::vec_set_scale(wallcontactforce, -mass_i[0], sumk_acc_ik);

    // write interaction output
    if (writeinteractionoutput)
    {
      // calculate wall contact point
      double wallcontactpoint[3];
      ParticleUtils::vec_set(wallcontactpoint, pos_i);
      ParticleUtils::vec_sub(wallcontactpoint, r_ij);

      // set wall attack point and states
      for (int dim = 0; dim < 3; ++dim) attackpoints.push_back(wallcontactpoint[dim]);
      for (int dim = 0; dim < 3; ++dim) contactforces.push_back(wallcontactforce[dim]);
      for (int dim = 0; dim < 3; ++dim) normaldirection.push_back(particlewallpair.e_ij_[dim]);
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
        particleinteractionwriter_->get_specific_runtime_output_writer("particle-wall-momentum");
    auto& visualization_data = visualization_manager->get_visualization_data();

    // set wall attack points
    visualization_data.get_point_coordinates() = attackpoints;

    // append states
    visualization_data.set_point_data_vector<double>("contact force", contactforces, 3);
    visualization_data.set_point_data_vector<double>("normal direction", normaldirection, 3);
  }
}

FOUR_C_NAMESPACE_CLOSE
