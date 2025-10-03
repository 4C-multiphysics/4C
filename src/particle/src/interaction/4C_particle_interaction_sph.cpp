// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_interaction_sph.hpp"

#include "4C_io_pstream.hpp"
#include "4C_particle_engine_container.hpp"
#include "4C_particle_engine_interface.hpp"
#include "4C_particle_interaction_material_handler.hpp"
#include "4C_particle_interaction_sph_boundary_particle.hpp"
#include "4C_particle_interaction_sph_density.hpp"
#include "4C_particle_interaction_sph_equationofstate.hpp"
#include "4C_particle_interaction_sph_equationofstate_bundle.hpp"
#include "4C_particle_interaction_sph_kernel.hpp"
#include "4C_particle_interaction_sph_momentum.hpp"
#include "4C_particle_interaction_sph_neighbor_pairs.hpp"
#include "4C_particle_interaction_sph_open_boundary.hpp"
#include "4C_particle_interaction_sph_phase_change.hpp"
#include "4C_particle_interaction_sph_pressure.hpp"
#include "4C_particle_interaction_sph_rigid_particle_contact.hpp"
#include "4C_particle_interaction_sph_surface_tension.hpp"
#include "4C_particle_interaction_sph_temperature.hpp"
#include "4C_particle_interaction_sph_virtual_wall_particle.hpp"
#include "4C_particle_interaction_utils.hpp"
#include "4C_particle_wall_interface.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
ParticleInteraction::ParticleInteractionSPH::ParticleInteractionSPH(
    MPI_Comm comm, const Teuchos::ParameterList& params)
    : ParticleInteraction::ParticleInteractionBase(comm, params), params_sph_(params.sublist("SPH"))
{
  // empty constructor
}

ParticleInteraction::ParticleInteractionSPH::~ParticleInteractionSPH() = default;

void ParticleInteraction::ParticleInteractionSPH::init()
{
  // call base class init
  ParticleInteractionBase::init();

  // init kernel handler
  init_kernel_handler();

  // init equation of state bundle
  init_equation_of_state_bundle();

  // init neighbor pair handler
  init_neighbor_pair_handler();

  // init density handler
  init_density_handler();

  // init pressure handler
  init_pressure_handler();

  // init temperature handler
  init_temperature_handler();

  // init momentum handler
  init_momentum_handler();

  // init surface tension handler
  init_surface_tension_handler();

  // init boundary particle handler
  init_boundary_particle_handler();

  // init dirichlet open boundary handler
  init_dirichlet_open_boundary_handler();

  // init neumann open boundary handler
  init_neumann_open_boundary_handler();

  // init virtual wall particle handler
  init_virtual_wall_particle_handler();

  // init phase change handler
  init_phase_change_handler();

  // init rigid particle contact handler
  init_rigid_particle_contact_handler();

  // safety check
  if (surfacetension_ and virtualwallparticle_)
    FOUR_C_THROW("surface tension formulation with wall interaction not implemented!");
}

void ParticleInteraction::ParticleInteractionSPH::setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface)
{
  // call base class setup
  ParticleInteractionBase::setup(particleengineinterface, particlewallinterface);

  // setup kernel handler
  kernel_->setup();

  // setup equation of state bundle
  equationofstatebundle_->setup();

  // setup neighbor pair handler
  neighborpairs_->setup(particleengineinterface, particlewallinterface, kernel_);

  // setup density handler
  density_->setup(particleengineinterface, particlewallinterface, kernel_, particlematerial_,
      equationofstatebundle_, neighborpairs_, virtualwallparticle_);

  // setup pressure handler
  pressure_->setup(particleengineinterface, particlematerial_, equationofstatebundle_);

  // setup temperature handler
  if (temperature_) temperature_->setup(particleengineinterface, particlematerial_, neighborpairs_);

  // setup momentum handler
  momentum_->setup(particleengineinterface, particlewallinterface, kernel_, particlematerial_,
      particleinteractionwriter_, equationofstatebundle_, neighborpairs_, virtualwallparticle_);

  // setup surface tension handler
  if (surfacetension_)
    surfacetension_->setup(particleengineinterface, kernel_, particlematerial_,
        equationofstatebundle_, neighborpairs_);

  // setup boundary particle handler
  if (boundaryparticle_) boundaryparticle_->setup(particleengineinterface, neighborpairs_);

  // setup dirichlet open boundary handler
  if (dirichletopenboundary_)
    dirichletopenboundary_->setup(particleengineinterface, kernel_, particlematerial_,
        equationofstatebundle_, neighborpairs_);

  // setup neumann open boundary handler
  if (neumannopenboundary_)
    neumannopenboundary_->setup(particleengineinterface, kernel_, particlematerial_,
        equationofstatebundle_, neighborpairs_);

  // setup virtual wall particle handler
  if (virtualwallparticle_)
    virtualwallparticle_->setup(
        particleengineinterface, particlewallinterface, kernel_, neighborpairs_);

  // setup phase change handler
  if (phasechange_)
    phasechange_->setup(particleengineinterface, particlematerial_, equationofstatebundle_);

  // setup rigid particle contact handler
  if (rigidparticlecontact_)
    rigidparticlecontact_->setup(
        particleengineinterface, particlewallinterface, particleinteractionwriter_, neighborpairs_);

  // short screen output
  if ((dirichletopenboundary_ or neumannopenboundary_) and
      particleengineinterface_->have_periodic_boundary_conditions())
  {
    if (myrank_ == 0)
      Core::IO::cout << "Warning: periodic boundary and open boundary conditions applied!"
                     << Core::IO::endl;
  }
}

void ParticleInteraction::ParticleInteractionSPH::write_restart() const
{
  // call base class function
  ParticleInteractionBase::write_restart();
}

void ParticleInteraction::ParticleInteractionSPH::read_restart(
    const std::shared_ptr<Core::IO::DiscretizationReader> reader)
{
  // call base class function
  ParticleInteractionBase::read_restart(reader);
}

void ParticleInteraction::ParticleInteractionSPH::insert_particle_states_of_particle_types(
    std::map<PARTICLEENGINE::TypeEnum, std::set<PARTICLEENGINE::StateEnum>>& particlestatestotypes)
{
  // iterate over particle types
  for (auto& typeIt : particlestatestotypes)
  {
    // get type of particles
    PARTICLEENGINE::TypeEnum type = typeIt.first;

    // set of particle states for current particle type
    std::set<PARTICLEENGINE::StateEnum>& particlestates = typeIt.second;

    if (type == PARTICLEENGINE::BoundaryPhase or type == PARTICLEENGINE::RigidPhase)
    {
      // insert states of boundary and rigid particles
      particlestates.insert({PARTICLEENGINE::Mass, PARTICLEENGINE::Radius,
          PARTICLEENGINE::BoundaryPressure, PARTICLEENGINE::BoundaryVelocity});
    }
    else if (type == PARTICLEENGINE::DirichletPhase or type == PARTICLEENGINE::NeumannPhase)
    {
      // insert states of open boundary particles
      particlestates.insert({PARTICLEENGINE::Mass, PARTICLEENGINE::Radius, PARTICLEENGINE::Density,
          PARTICLEENGINE::Pressure});
    }
    else
    {
      // insert states of regular phase particles
      particlestates.insert({PARTICLEENGINE::Mass, PARTICLEENGINE::Radius, PARTICLEENGINE::Density,
          PARTICLEENGINE::Pressure});
    }
  }

  // states for density evaluation scheme
  density_->insert_particle_states_of_particle_types(particlestatestotypes);

  // states for temperature evaluation scheme
  if (temperature_) temperature_->insert_particle_states_of_particle_types(particlestatestotypes);

  // insert momentum evaluation dependent states
  momentum_->insert_particle_states_of_particle_types(particlestatestotypes);

  // additional states for surface tension formulation
  if (surfacetension_)
    surfacetension_->insert_particle_states_of_particle_types(particlestatestotypes);
}

void ParticleInteraction::ParticleInteractionSPH::set_initial_states()
{
  // get kernel space dimension
  int kernelspacedim = 0;
  kernel_->kernel_space_dimension(kernelspacedim);

  // get initial particle spacing
  const double initialparticlespacing = params_sph_.get<double>("INITIALPARTICLESPACING");

  // safety check
  if (not(initialparticlespacing > 0.0)) FOUR_C_THROW("negative initial particle spacing!");

  // compute initial particle volume
  const double initialparticlevolume = std::pow(initialparticlespacing, kernelspacedim);

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

    // get material for current particle type
    const Mat::PAR::ParticleMaterialBase* material =
        particlematerial_->get_ptr_to_particle_mat_parameter(type_i);

    // initial density of current phase
    std::vector<double> initdensity(1);
    initdensity[0] = material->initDensity_;

    // (initial) mass of current phase
    std::vector<double> initmass(1);
    initmass[0] = initdensity[0] * initialparticlevolume;

    // (initial) radius of current phase
    std::vector<double> initradius(1);
    initradius[0] = material->initRadius_;

    // set initial density for respective particles of current type
    if (container->have_stored_state(PARTICLEENGINE::Density))
      container->set_state(initdensity, PARTICLEENGINE::Density);

    // set initial mass and radius for all particles of current type
    container->set_state(initmass, PARTICLEENGINE::Mass);
    container->set_state(initradius, PARTICLEENGINE::Radius);

    // evaluate initial inertia for respective particles of current type
    if (container->have_stored_state(PARTICLEENGINE::Inertia))
    {
      // (initial) inertia of current phase
      std::vector<double> initinertia(1);

      if (kernelspacedim == 2)
      {
        // effective particle radius considering initial particle volume in disk shape
        const double effectiveradius = std::sqrt(std::numbers::inv_pi * initialparticlevolume);

        // inertia for disk shape
        initinertia[0] = 0.5 * initmass[0] * Utils::pow<2>(effectiveradius);
      }
      else if (kernelspacedim == 3)
      {
        // effective particle radius considering initial particle volume in spherical shape
        const double effectiveradius =
            std::pow(0.75 * std::numbers::inv_pi * initialparticlevolume, 1.0 / 3.0);

        // inertia for spherical shape
        initinertia[0] = 0.4 * initmass[0] * Utils::pow<2>(effectiveradius);
      }
      else
      {
        FOUR_C_THROW("inertia for particles only in two and three dimensional evaluation given!");
      }

      // set initial inertia for respective particles of current type
      container->set_state(initinertia, PARTICLEENGINE::Inertia);
    }

    // initial states for temperature evaluation
    if (temperature_)
    {
      // get material for current particle type
      const Mat::PAR::ParticleMaterialThermo* material =
          dynamic_cast<const Mat::PAR::ParticleMaterialThermo*>(
              particlematerial_->get_ptr_to_particle_mat_parameter(type_i));

      // initial temperature of current phase
      std::vector<double> inittemperature(1);
      inittemperature[0] = material->initTemperature_;

      // set initial temperature for all particles of current type
      container->set_state(inittemperature, PARTICLEENGINE::Temperature);
    }
  }
}

void ParticleInteraction::ParticleInteractionSPH::pre_evaluate_time_step()
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::ParticleInteractionSPH::pre_evaluate_time_step");

  // prescribe open boundary states
  if (dirichletopenboundary_) dirichletopenboundary_->prescribe_open_boundary_states(time_);

  // prescribe open boundary states
  if (neumannopenboundary_) neumannopenboundary_->prescribe_open_boundary_states(time_);
}

void ParticleInteraction::ParticleInteractionSPH::evaluate_interactions()
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::ParticleInteractionSPH::evaluate_interactions");

  // evaluate particle neighbor pairs
  neighborpairs_->evaluate_neighbor_pairs();

  // init relative positions of virtual particles
  if (virtualwallparticle_)
    virtualwallparticle_->init_relative_positions_of_virtual_particles(max_interaction_distance());

  // compute density field
  density_->compute_density();

  // compute pressure using equation of state and density
  pressure_->compute_pressure();

  // compute interface quantities
  if (surfacetension_) surfacetension_->compute_interface_quantities();

  // compute temperature field
  if (temperature_) temperature_->compute_temperature();

  // interpolate open boundary states
  if (dirichletopenboundary_) dirichletopenboundary_->interpolate_open_boundary_states();

  // interpolate open boundary states
  if (neumannopenboundary_) neumannopenboundary_->interpolate_open_boundary_states();

  // init boundary particle states
  if (boundaryparticle_) boundaryparticle_->init_boundary_particle_states(gravity_);

  // init states at wall contact points
  if (virtualwallparticle_) virtualwallparticle_->init_states_at_wall_contact_points(gravity_);

  // add momentum contribution to acceleration field
  momentum_->add_acceleration_contribution();

  // add surface tension contribution to acceleration field
  if (surfacetension_) surfacetension_->add_acceleration_contribution();

  // add rigid particle contact contribution to force field
  if (rigidparticlecontact_) rigidparticlecontact_->add_force_contribution();
}

void ParticleInteraction::ParticleInteractionSPH::post_evaluate_time_step(
    std::vector<PARTICLEENGINE::ParticleTypeToType>& particlesfromphasetophase)
{
  TEUCHOS_FUNC_TIME_MONITOR("ParticleInteraction::ParticleInteractionSPH::post_evaluate_time_step");

  // check open boundary phase change
  if (dirichletopenboundary_)
    dirichletopenboundary_->check_open_boundary_phase_change(max_interaction_distance());

  // check open boundary phase change
  if (neumannopenboundary_)
    neumannopenboundary_->check_open_boundary_phase_change(max_interaction_distance());

  // evaluate phase change
  if (phasechange_) phasechange_->evaluate_phase_change(particlesfromphasetophase);
}

double ParticleInteraction::ParticleInteractionSPH::max_interaction_distance() const
{
  return max_particle_radius();
}

void ParticleInteraction::ParticleInteractionSPH::distribute_interaction_history() const
{
  // nothing to do
}

void ParticleInteraction::ParticleInteractionSPH::communicate_interaction_history() const
{
  // nothing to do
}

void ParticleInteraction::ParticleInteractionSPH::set_current_time(const double currenttime)
{
  // call base class method
  ParticleInteractionBase::set_current_time(currenttime);

  // set current time
  if (temperature_) temperature_->set_current_time(currenttime);

  // set current time
  if (surfacetension_) surfacetension_->set_current_time(currenttime);
}

void ParticleInteraction::ParticleInteractionSPH::set_current_step_size(
    const double currentstepsize)
{
  // call base class method
  ParticleInteractionBase::set_current_step_size(currentstepsize);

  // set current step size
  density_->set_current_step_size(currentstepsize);

  // set current step size
  if (temperature_) temperature_->set_current_step_size(currentstepsize);
}

void ParticleInteraction::ParticleInteractionSPH::init_kernel_handler()
{
  // get type of smoothed particle hydrodynamics kernel
  auto kerneltype = Teuchos::getIntegralValue<PARTICLE::KernelType>(params_sph_, "KERNEL");

  // create kernel handler
  switch (kerneltype)
  {
    case PARTICLE::CubicSpline:
    {
      kernel_ = std::make_shared<ParticleInteraction::SPHKernelCubicSpline>(params_sph_);
      break;
    }
    case PARTICLE::QuinticSpline:
    {
      kernel_ = std::make_shared<ParticleInteraction::SPHKernelQuinticSpline>(params_sph_);
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown kernel type!");
      break;
    }
  }

  // init kernel handler
  kernel_->init();
}

void ParticleInteraction::ParticleInteractionSPH::init_equation_of_state_bundle()
{
  // create equation of state bundle
  equationofstatebundle_ =
      std::make_shared<ParticleInteraction::SPHEquationOfStateBundle>(params_sph_);

  // init equation of state bundle
  equationofstatebundle_->init(*particlematerial_);
}

void ParticleInteraction::ParticleInteractionSPH::init_neighbor_pair_handler()
{
  // create neighbor pair handler
  neighborpairs_ = std::make_shared<ParticleInteraction::SPHNeighborPairs>();

  // init neighbor pair handler
  neighborpairs_->init();
}

void ParticleInteraction::ParticleInteractionSPH::init_density_handler()
{
  // get type of smoothed particle hydrodynamics density evaluation scheme
  auto densityevaluationscheme = Teuchos::getIntegralValue<PARTICLE::DensityEvaluationScheme>(
      params_sph_, "DENSITYEVALUATION");

  // create density handler
  switch (densityevaluationscheme)
  {
    case PARTICLE::DensitySummation:
    {
      density_ = std::unique_ptr<ParticleInteraction::SPHDensitySummation>(
          new ParticleInteraction::SPHDensitySummation(params_sph_));
      break;
    }
    case PARTICLE::DensityIntegration:
    {
      density_ = std::unique_ptr<ParticleInteraction::SPHDensityIntegration>(
          new ParticleInteraction::SPHDensityIntegration(params_sph_));
      break;
    }
    case PARTICLE::DensityPredictCorrect:
    {
      density_ = std::unique_ptr<ParticleInteraction::SPHDensityPredictCorrect>(
          new ParticleInteraction::SPHDensityPredictCorrect(params_sph_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown density evaluation scheme type!");
      break;
    }
  }

  // init density handler
  density_->init();

  // safety check
  if (densityevaluationscheme != PARTICLE::DensityPredictCorrect and
      Teuchos::getIntegralValue<PARTICLE::DensityCorrectionScheme>(
          params_sph_, "DENSITYCORRECTION") != PARTICLE::NoCorrection)
    FOUR_C_THROW(
        "the density correction scheme set is not valid with the current density evaluation "
        "scheme!");
}

void ParticleInteraction::ParticleInteractionSPH::init_pressure_handler()
{
  // create pressure handler
  pressure_ =
      std::unique_ptr<ParticleInteraction::SPHPressure>(new ParticleInteraction::SPHPressure());

  // init pressure handler
  pressure_->init();
}

void ParticleInteraction::ParticleInteractionSPH::init_temperature_handler()
{
  // get type of smoothed particle hydrodynamics temperature evaluation scheme
  auto temperatureevaluationscheme =
      Teuchos::getIntegralValue<PARTICLE::TemperatureEvaluationScheme>(
          params_sph_, "TEMPERATUREEVALUATION");

  // create temperature handler
  switch (temperatureevaluationscheme)
  {
    case PARTICLE::NoTemperatureEvaluation:
    {
      temperature_ = std::unique_ptr<ParticleInteraction::SPHTemperature>(nullptr);
      break;
    }
    case PARTICLE::TemperatureIntegration:
    {
      temperature_ = std::unique_ptr<ParticleInteraction::SPHTemperature>(
          new ParticleInteraction::SPHTemperature(params_sph_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown temperature evaluation scheme!");
      break;
    }
  }

  // init temperature handler
  if (temperature_) temperature_->init();
}

void ParticleInteraction::ParticleInteractionSPH::init_momentum_handler()
{
  // create momentum handler
  momentum_ = std::unique_ptr<ParticleInteraction::SPHMomentum>(
      new ParticleInteraction::SPHMomentum(params_sph_));

  // init momentum handler
  momentum_->init();
}

void ParticleInteraction::ParticleInteractionSPH::init_surface_tension_handler()
{
  // get type of smoothed particle hydrodynamics surface tension formulation
  auto surfacetensionformulation = Teuchos::getIntegralValue<PARTICLE::SurfaceTensionFormulation>(
      params_sph_, "SURFACETENSIONFORMULATION");

  // create surface tension handler
  switch (surfacetensionformulation)
  {
    case PARTICLE::NoSurfaceTension:
    {
      surfacetension_ = std::unique_ptr<ParticleInteraction::SPHSurfaceTension>(nullptr);
      break;
    }
    case PARTICLE::ContinuumSurfaceForce:
    {
      surfacetension_ = std::unique_ptr<ParticleInteraction::SPHSurfaceTension>(
          new ParticleInteraction::SPHSurfaceTension(params_sph_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown surface tension formulation type!");
      break;
    }
  }

  // init surface tension handler
  if (surfacetension_) surfacetension_->init();
}

void ParticleInteraction::ParticleInteractionSPH::init_boundary_particle_handler()
{
  // get type of boundary particle formulation
  auto boundaryparticleformulation =
      Teuchos::getIntegralValue<PARTICLE::BoundaryParticleFormulationType>(
          params_sph_, "BOUNDARYPARTICLEFORMULATION");

  // create boundary particle handler
  switch (boundaryparticleformulation)
  {
    case PARTICLE::NoBoundaryFormulation:
    {
      boundaryparticle_ = std::unique_ptr<ParticleInteraction::SPHBoundaryParticleBase>(nullptr);
      break;
    }
    case PARTICLE::AdamiBoundaryFormulation:
    {
      boundaryparticle_ = std::unique_ptr<ParticleInteraction::SPHBoundaryParticleAdami>(
          new ParticleInteraction::SPHBoundaryParticleAdami(params_sph_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown boundary particle formulation type!");
      break;
    }
  }

  // init boundary particle handler
  if (boundaryparticle_) boundaryparticle_->init();
}

void ParticleInteraction::ParticleInteractionSPH::init_dirichlet_open_boundary_handler()
{
  // get type of dirichlet open boundary
  auto dirichletopenboundarytype = Teuchos::getIntegralValue<PARTICLE::DirichletOpenBoundaryType>(
      params_sph_, "DIRICHLETBOUNDARYTYPE");

  // create open boundary handler
  switch (dirichletopenboundarytype)
  {
    case PARTICLE::NoDirichletOpenBoundary:
    {
      dirichletopenboundary_ = std::unique_ptr<ParticleInteraction::SPHOpenBoundaryBase>(nullptr);
      break;
    }
    case PARTICLE::DirichletNormalToPlane:
    {
      dirichletopenboundary_ = std::unique_ptr<ParticleInteraction::SPHOpenBoundaryDirichlet>(
          new ParticleInteraction::SPHOpenBoundaryDirichlet(params_sph_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown dirichlet open boundary type!");
      break;
    }
  }

  // init open boundary handler
  if (dirichletopenboundary_) dirichletopenboundary_->init();
}

void ParticleInteraction::ParticleInteractionSPH::init_neumann_open_boundary_handler()
{
  // get type of neumann open boundary
  auto neumannopenboundarytype = Teuchos::getIntegralValue<PARTICLE::NeumannOpenBoundaryType>(
      params_sph_, "NEUMANNBOUNDARYTYPE");

  // create open boundary handler
  switch (neumannopenboundarytype)
  {
    case PARTICLE::NoNeumannOpenBoundary:
    {
      neumannopenboundary_ = std::unique_ptr<ParticleInteraction::SPHOpenBoundaryBase>(nullptr);
      break;
    }
    case PARTICLE::NeumannNormalToPlane:
    {
      neumannopenboundary_ = std::unique_ptr<ParticleInteraction::SPHOpenBoundaryNeumann>(
          new ParticleInteraction::SPHOpenBoundaryNeumann(params_sph_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown neumann open boundary type!");
      break;
    }
  }

  // init open boundary handler
  if (neumannopenboundary_) neumannopenboundary_->init();
}

void ParticleInteraction::ParticleInteractionSPH::init_virtual_wall_particle_handler()
{
  // get type of wall formulation
  auto wallformulation =
      Teuchos::getIntegralValue<PARTICLE::WallFormulationType>(params_sph_, "WALLFORMULATION");

  // create virtual wall particle handler
  switch (wallformulation)
  {
    case PARTICLE::NoWallFormulation:
    {
      virtualwallparticle_ = std::shared_ptr<ParticleInteraction::SPHVirtualWallParticle>(nullptr);
      break;
    }
    case PARTICLE::VirtualParticleWallFormulation:
    {
      virtualwallparticle_ =
          std::make_shared<ParticleInteraction::SPHVirtualWallParticle>(params_sph_);
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown wall formulation type!");
      break;
    }
  }

  // init virtual wall particle handler
  if (virtualwallparticle_) virtualwallparticle_->init();
}

void ParticleInteraction::ParticleInteractionSPH::init_phase_change_handler()
{
  // get type of phase change
  auto phasechangetype =
      Teuchos::getIntegralValue<PARTICLE::PhaseChangeType>(params_sph_, "PHASECHANGETYPE");

  // create phase change handler
  switch (phasechangetype)
  {
    case PARTICLE::NoPhaseChange:
    {
      phasechange_ = std::unique_ptr<ParticleInteraction::SPHPhaseChangeBase>(nullptr);
      break;
    }
    case PARTICLE::OneWayScalarBelowToAbovePhaseChange:
    {
      phasechange_ = std::unique_ptr<ParticleInteraction::SPHPhaseChangeOneWayScalarBelowToAbove>(
          new ParticleInteraction::SPHPhaseChangeOneWayScalarBelowToAbove(params_sph_));
      break;
    }
    case PARTICLE::OneWayScalarAboveToBelowPhaseChange:
    {
      phasechange_ = std::unique_ptr<ParticleInteraction::SPHPhaseChangeOneWayScalarAboveToBelow>(
          new ParticleInteraction::SPHPhaseChangeOneWayScalarAboveToBelow(params_sph_));
      break;
    }
    case PARTICLE::TwoWayScalarPhaseChange:
    {
      phasechange_ = std::unique_ptr<ParticleInteraction::SPHPhaseChangeTwoWayScalar>(
          new ParticleInteraction::SPHPhaseChangeTwoWayScalar(params_sph_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown boundary particle formulation type!");
      break;
    }
  }

  // init phase change handler
  if (phasechange_) phasechange_->init();
}

void ParticleInteraction::ParticleInteractionSPH::init_rigid_particle_contact_handler()
{
  // get type of rigid particle contact
  auto rigidparticlecontacttype = Teuchos::getIntegralValue<PARTICLE::RigidParticleContactType>(
      params_sph_, "RIGIDPARTICLECONTACTTYPE");

  // create rigid particle contact handler
  switch (rigidparticlecontacttype)
  {
    case PARTICLE::NoRigidParticleContact:
    {
      rigidparticlecontact_ =
          std::unique_ptr<ParticleInteraction::SPHRigidParticleContactBase>(nullptr);
      break;
    }
    case PARTICLE::ElasticRigidParticleContact:
    {
      rigidparticlecontact_ = std::unique_ptr<ParticleInteraction::SPHRigidParticleContactElastic>(
          new ParticleInteraction::SPHRigidParticleContactElastic(params_sph_));
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown rigid particle contact type!");
      break;
    }
  }

  // init rigid particle contact handler
  if (rigidparticlecontact_) rigidparticlecontact_->init();
}

FOUR_C_NAMESPACE_CLOSE
