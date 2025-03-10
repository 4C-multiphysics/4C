// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_particle.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | set the particle parameters                                               |
 *---------------------------------------------------------------------------*/
void Inpar::PARTICLE::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  /*-------------------------------------------------------------------------*
   | general control parameters for particle simulations                     |
   *-------------------------------------------------------------------------*/
  Core::Utils::SectionSpecs particledyn{"PARTICLE DYNAMIC"};

  // type of particle time integration
  Core::Utils::string_to_integral_parameter<DynamicType>("DYNAMICTYPE", "VelocityVerlet",
      "type of particle time integration",
      tuple<std::string>("SemiImplicitEuler", "VelocityVerlet"),
      tuple<DynamicType>(
          Inpar::PARTICLE::dyna_semiimpliciteuler, Inpar::PARTICLE::dyna_velocityverlet),
      particledyn);

  // type of particle interaction
  Core::Utils::string_to_integral_parameter<InteractionType>("INTERACTION", "None",
      "type of particle interaction", tuple<std::string>("None", "SPH", "DEM"),
      tuple<InteractionType>(Inpar::PARTICLE::interaction_none, Inpar::PARTICLE::interaction_sph,
          Inpar::PARTICLE::interaction_dem),
      particledyn);

  // output type
  particledyn.specs.emplace_back(parameter<int>(
      "RESULTSEVERY", {.description = "write particle runtime output every RESULTSEVERY steps",
                          .default_value = 1}));
  particledyn.specs.emplace_back(parameter<int>("RESTARTEVERY",
      {.description = "write restart possibility every RESTARTEVERY steps", .default_value = 1}));

  // write ghosted particles
  particledyn.specs.emplace_back(parameter<bool>("WRITE_GHOSTED_PARTICLES",
      {.description = "write ghosted particles (debug feature)", .default_value = false}));

  // time loop control
  particledyn.specs.emplace_back(
      parameter<double>("TIMESTEP", {.description = "time step size", .default_value = 0.01}));
  particledyn.specs.emplace_back(
      parameter<int>("NUMSTEP", {.description = "maximum number of steps", .default_value = 100}));
  particledyn.specs.emplace_back(
      parameter<double>("MAXTIME", {.description = "maximum time", .default_value = 1.0}));

  // gravity acceleration control
  particledyn.specs.emplace_back(parameter<std::string>("GRAVITY_ACCELERATION",
      {.description = "acceleration due to gravity", .default_value = "0.0 0.0 0.0"}));
  particledyn.specs.emplace_back(parameter<int>("GRAVITY_RAMP_FUNCT",
      {.description = "number of function governing gravity ramp", .default_value = -1}));

  // viscous damping factor
  particledyn.specs.emplace_back(parameter<double>("VISCOUS_DAMPING",
      {.description = "apply viscous damping force to determine static equilibrium solutions",
          .default_value = -1.0}));

  // transfer particles to new bins every time step
  particledyn.specs.emplace_back(parameter<bool>("TRANSFER_EVERY",
      {.description = "transfer particles to new bins every time step", .default_value = false}));

  // considered particle phases with dynamic load balance weighting factor
  particledyn.specs.emplace_back(parameter<std::string>("PHASE_TO_DYNLOADBALFAC",
      {.description = "considered particle phases with dynamic load balance weighting factor",
          .default_value = "none"}));

  // relate particle phase to material id
  particledyn.specs.emplace_back(parameter<std::string>("PHASE_TO_MATERIAL_ID",
      {.description = "relate particle phase to material id", .default_value = "none"}));

  // amplitude of noise added to initial position for each spatial direction
  particledyn.specs.emplace_back(parameter<std::string>("INITIAL_POSITION_AMPLITUDE",
      {.description = "amplitude of noise added to initial position for each spatial direction",
          .default_value = "0.0 0.0 0.0"}));

  // type of particle wall source
  Core::Utils::string_to_integral_parameter<ParticleWallSource>("PARTICLE_WALL_SOURCE",
      "NoParticleWall", "type of particle wall source",
      tuple<std::string>("NoParticleWall", "DiscretCondition", "BoundingBox"),
      tuple<ParticleWallSource>(Inpar::PARTICLE::NoParticleWall, Inpar::PARTICLE::DiscretCondition,
          Inpar::PARTICLE::BoundingBox),
      particledyn);

  // material id for particle wall from bounding box source
  particledyn.specs.emplace_back(parameter<int>(
      "PARTICLE_WALL_MAT", {.description = "material id for particle wall from bounding box source",
                               .default_value = -1}));

  // flags defining considered states of particle wall
  particledyn.specs.emplace_back(parameter<bool>("PARTICLE_WALL_MOVING",
      {.description = "consider a moving particle wall", .default_value = false}));
  particledyn.specs.emplace_back(parameter<bool>("PARTICLE_WALL_LOADED",
      {.description = "consider loading on particle wall", .default_value = false}));

  // consider rigid body motion
  particledyn.specs.emplace_back(parameter<bool>(
      "RIGID_BODY_MOTION", {.description = "consider rigid body motion", .default_value = false}));

  particledyn.specs.emplace_back(parameter<double>("RIGID_BODY_PHASECHANGE_RADIUS",
      {.description = "search radius for neighboring rigid bodies in case of phase change",
          .default_value = -1.0}));

  particledyn.move_into_collection(list);

  /*-------------------------------------------------------------------------*
   | control parameters for initial/boundary conditions                      |
   *-------------------------------------------------------------------------*/
  Core::Utils::SectionSpecs particledynconditions{particledyn, "INITIAL AND BOUNDARY CONDITIONS"};

  // initial temperature field of particle phase given by function
  particledynconditions.specs.emplace_back(parameter<std::string>("INITIAL_TEMP_FIELD",
      {.description =
              "Refer to the function ID describing the initial temperature field of particle phase",
          .default_value = "none"}));

  // initial velocity field of particle phase given by function
  particledynconditions.specs.emplace_back(parameter<std::string>("INITIAL_VELOCITY_FIELD",
      {.description =
              "Refer to the function ID describing the initial velocity field of particle phase",
          .default_value = "none"}));

  // initial angular velocity field of particle phase given by function
  particledynconditions.specs.emplace_back(parameter<std::string>("INITIAL_ANGULAR_VELOCITY_FIELD",
      {.description = "Refer to the function ID describing the initial angular velocity field of "
                      "rigid body  phase/DEM particle",
          .default_value = "none"}));


  // initial acceleration field of particle phase given by function
  particledynconditions.specs.emplace_back(parameter<std::string>(
      "INITIAL_ACCELERATION_FIELD", {.description = "Refer to the function ID describing the "
                                                    "initial acceleration field of particle phase",
                                        .default_value = "none"}));

  // initial angular acceleration field of particle phase given by function
  particledynconditions.specs.emplace_back(
      parameter<std::string>("INITIAL_ANGULAR_ACCELERATION_FIELD",
          {.description = "Refer to the function ID describing the initial angular acceleration "
                          "field of rigid body  phase/DEM particle",
              .default_value = "none"}));


  // dirichlet boundary condition of particle phase given by function
  particledynconditions.specs.emplace_back(parameter<std::string>("DIRICHLET_BOUNDARY_CONDITION",
      {.description = "Refer to the function ID describing the dirichlet boundary condition of "
                      "particle phase",
          .default_value = "none"}));

  // temperature boundary condition of particle phase given by function
  particledynconditions.specs.emplace_back(parameter<std::string>("TEMPERATURE_BOUNDARY_CONDITION",
      {.description = "Refer to the function ID describing the temperature boundary condition of "
                      "particle phase",
          .default_value = "none"}));

  particledynconditions.move_into_collection(list);

  /*-------------------------------------------------------------------------*
   | smoothed particle hydrodynamics (SPH) specific control parameters       |
   *-------------------------------------------------------------------------*/
  Core::Utils::SectionSpecs particledynsph{particledyn, "SPH"};

  // write particle-wall interaction output
  particledynsph.specs.emplace_back(parameter<bool>("WRITE_PARTICLE_WALL_INTERACTION",
      {.description = "write particle-wall interaction output", .default_value = false}));

  // type of smoothed particle hydrodynamics kernel
  Core::Utils::string_to_integral_parameter<KernelType>("KERNEL", "CubicSpline",
      "type of smoothed particle hydrodynamics kernel",
      tuple<std::string>("CubicSpline", "QuinticSpline"),
      tuple<KernelType>(Inpar::PARTICLE::CubicSpline, Inpar::PARTICLE::QuinticSpline),
      particledynsph);

  // kernel space dimension number
  Core::Utils::string_to_integral_parameter<KernelSpaceDimension>("KERNEL_SPACE_DIM", "Kernel3D",
      "kernel space dimension number", tuple<std::string>("Kernel1D", "Kernel2D", "Kernel3D"),
      tuple<KernelSpaceDimension>(
          Inpar::PARTICLE::Kernel1D, Inpar::PARTICLE::Kernel2D, Inpar::PARTICLE::Kernel3D),
      particledynsph);

  particledynsph.specs.emplace_back(parameter<double>("INITIALPARTICLESPACING",
      {.description = "initial spacing of particles", .default_value = 0.0}));

  // type of smoothed particle hydrodynamics equation of state
  Core::Utils::string_to_integral_parameter<EquationOfStateType>("EQUATIONOFSTATE", "GenTait",
      "type of smoothed particle hydrodynamics equation of state",
      tuple<std::string>("GenTait", "IdealGas"),
      tuple<EquationOfStateType>(Inpar::PARTICLE::GenTait, Inpar::PARTICLE::IdealGas),
      particledynsph);

  // type of smoothed particle hydrodynamics momentum formulation
  Core::Utils::string_to_integral_parameter<MomentumFormulationType>("MOMENTUMFORMULATION",
      "AdamiMomentumFormulation", "type of smoothed particle hydrodynamics momentum formulation",
      tuple<std::string>("AdamiMomentumFormulation", "MonaghanMomentumFormulation"),
      tuple<MomentumFormulationType>(
          Inpar::PARTICLE::AdamiMomentumFormulation, Inpar::PARTICLE::MonaghanMomentumFormulation),
      particledynsph);

  // type of density evaluation scheme
  Core::Utils::string_to_integral_parameter<DensityEvaluationScheme>("DENSITYEVALUATION",
      "DensitySummation", "type of density evaluation scheme",
      tuple<std::string>("DensitySummation", "DensityIntegration", "DensityPredictCorrect"),
      tuple<DensityEvaluationScheme>(Inpar::PARTICLE::DensitySummation,
          Inpar::PARTICLE::DensityIntegration, Inpar::PARTICLE::DensityPredictCorrect),
      particledynsph);

  // type of density correction scheme
  Core::Utils::string_to_integral_parameter<DensityCorrectionScheme>("DENSITYCORRECTION",
      "NoCorrection", "type of density correction scheme",
      tuple<std::string>(
          "NoCorrection", "InteriorCorrection", "NormalizedCorrection", "RandlesCorrection"),
      tuple<DensityCorrectionScheme>(Inpar::PARTICLE::NoCorrection,
          Inpar::PARTICLE::InteriorCorrection, Inpar::PARTICLE::NormalizedCorrection,
          Inpar::PARTICLE::RandlesCorrection),
      particledynsph);

  // type of boundary particle formulation
  Core::Utils::string_to_integral_parameter<BoundaryParticleFormulationType>(
      "BOUNDARYPARTICLEFORMULATION", "NoBoundaryFormulation",
      "type of boundary particle formulation",
      tuple<std::string>("NoBoundaryFormulation", "AdamiBoundaryFormulation"),
      tuple<BoundaryParticleFormulationType>(
          Inpar::PARTICLE::NoBoundaryFormulation, Inpar::PARTICLE::AdamiBoundaryFormulation),
      particledynsph);

  // type of boundary particle interaction
  Core::Utils::string_to_integral_parameter<BoundaryParticleInteraction>(
      "BOUNDARYPARTICLEINTERACTION", "NoSlipBoundaryParticle",
      "type of boundary particle interaction",
      tuple<std::string>("NoSlipBoundaryParticle", "FreeSlipBoundaryParticle"),
      tuple<BoundaryParticleInteraction>(
          Inpar::PARTICLE::NoSlipBoundaryParticle, Inpar::PARTICLE::FreeSlipBoundaryParticle),
      particledynsph);

  // type of wall formulation
  Core::Utils::string_to_integral_parameter<WallFormulationType>("WALLFORMULATION",
      "NoWallFormulation", "type of wall formulation",
      tuple<std::string>("NoWallFormulation", "VirtualParticleWallFormulation"),
      tuple<WallFormulationType>(
          Inpar::PARTICLE::NoWallFormulation, Inpar::PARTICLE::VirtualParticleWallFormulation),
      particledynsph);

  // type of transport velocity formulation
  Core::Utils::string_to_integral_parameter<TransportVelocityFormulation>(
      "TRANSPORTVELOCITYFORMULATION", "NoTransportVelocity",
      "type of transport velocity formulation",
      tuple<std::string>(
          "NoTransportVelocity", "StandardTransportVelocity", "GeneralizedTransportVelocity"),
      tuple<TransportVelocityFormulation>(Inpar::PARTICLE::NoTransportVelocity,
          Inpar::PARTICLE::StandardTransportVelocity,
          Inpar::PARTICLE::GeneralizedTransportVelocity),
      particledynsph);

  // type of temperature evaluation scheme
  Core::Utils::string_to_integral_parameter<TemperatureEvaluationScheme>("TEMPERATUREEVALUATION",
      "NoTemperatureEvaluation", "type of temperature evaluation scheme",
      tuple<std::string>("NoTemperatureEvaluation", "TemperatureIntegration"),
      tuple<TemperatureEvaluationScheme>(
          Inpar::PARTICLE::NoTemperatureEvaluation, Inpar::PARTICLE::TemperatureIntegration),
      particledynsph);

  particledynsph.specs.emplace_back(parameter<bool>("TEMPERATUREGRADIENT",
      {.description = "evaluate temperature gradient", .default_value = false}));

  // type of heat source
  Core::Utils::string_to_integral_parameter<HeatSourceType>("HEATSOURCETYPE", "NoHeatSource",
      "type of heat source",
      tuple<std::string>("NoHeatSource", "VolumeHeatSource", "SurfaceHeatSource"),
      tuple<HeatSourceType>(Inpar::PARTICLE::NoHeatSource, Inpar::PARTICLE::VolumeHeatSource,
          Inpar::PARTICLE::SurfaceHeatSource),
      particledynsph);

  particledynsph.specs.emplace_back(parameter<int>("HEATSOURCE_FUNCT",
      {.description = "number of function governing heat source", .default_value = -1}));

  particledynsph.specs.emplace_back(parameter<std::string>("HEATSOURCE_DIRECTION",
      {.description = "direction of surface heat source", .default_value = "0.0 0.0 0.0"}));

  // evaporation induced heat loss
  particledynsph.specs.emplace_back(parameter<bool>("VAPOR_HEATLOSS",
      {.description = "evaluate evaporation induced heat loss", .default_value = false}));
  particledynsph.specs.emplace_back(parameter<double>("VAPOR_HEATLOSS_LATENTHEAT",
      {.description = "latent heat in heat loss formula", .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("VAPOR_HEATLOSS_ENTHALPY_REFTEMP",
      {.description = "enthalpy reference temperature in heat loss formula",
          .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("VAPOR_HEATLOSS_PFAC",
      {.description = "pressure factor in heat loss formula", .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("VAPOR_HEATLOSS_TFAC",
      {.description = "temperature factor in heat loss formula", .default_value = 0.0}));

  // evaporation induced recoil pressure
  particledynsph.specs.emplace_back(parameter<bool>("VAPOR_RECOIL",
      {.description = "evaluate evaporation induced recoil pressure", .default_value = false}));
  particledynsph.specs.emplace_back(parameter<double>("VAPOR_RECOIL_BOILINGTEMPERATURE",
      {.description = "boiling temperature in recoil pressure formula", .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("VAPOR_RECOIL_PFAC",
      {.description = "pressure factor in recoil pressure formula", .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("VAPOR_RECOIL_TFAC",
      {.description = "temperature factor in recoil pressure formula", .default_value = 0.0}));

  // type of surface tension formulation
  Core::Utils::string_to_integral_parameter<SurfaceTensionFormulation>("SURFACETENSIONFORMULATION",
      "NoSurfaceTension", "type of surface tension formulation",
      tuple<std::string>("NoSurfaceTension", "ContinuumSurfaceForce"),
      tuple<SurfaceTensionFormulation>(
          Inpar::PARTICLE::NoSurfaceTension, Inpar::PARTICLE::ContinuumSurfaceForce),
      particledynsph);

  particledynsph.specs.emplace_back(parameter<int>("SURFACETENSION_RAMP_FUNCT",
      {.description = "number of function governing surface tension ramp", .default_value = -1}));

  particledynsph.specs.emplace_back(parameter<double>("SURFACETENSIONCOEFFICIENT",
      {.description = "constant part of surface tension coefficient", .default_value = -1.0}));
  particledynsph.specs.emplace_back(parameter<double>("SURFACETENSIONMINIMUM",
      {.description = "minimum surface tension coefficient in case of temperature dependence",
          .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("SURFACETENSIONTEMPFAC",
      {.description = "factor of dependence of surface tension coefficient on temperature",
          .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("SURFACETENSIONREFTEMP",
      {.description = "reference temperature for surface tension coefficient",
          .default_value = 0.0}));

  // wetting
  particledynsph.specs.emplace_back(parameter<double>(
      "STATICCONTACTANGLE", {.description = "static contact angle in degree with wetting effects",
                                .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("TRIPLEPOINTNORMAL_CORR_CF_LOW",
      {.description = "triple point normal correction wall color field low",
          .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("TRIPLEPOINTNORMAL_CORR_CF_UP",
      {.description = "triple point normal correction wall color field up", .default_value = 0.0}));

  // interface viscosity
  particledynsph.specs.emplace_back(parameter<bool>("INTERFACE_VISCOSITY",
      {.description = "evaluate interface viscosity", .default_value = false}));
  particledynsph.specs.emplace_back(parameter<double>("INTERFACE_VISCOSITY_LIQUIDGAS",
      {.description = "artificial viscosity on liquid-gas interface", .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("INTERFACE_VISCOSITY_SOLIDLIQUID",
      {.description = "artificial viscosity on solid-liquid interface", .default_value = 0.0}));

  // barrier force
  particledynsph.specs.emplace_back(parameter<bool>(
      "BARRIER_FORCE", {.description = "evaluate barrier force", .default_value = false}));
  particledynsph.specs.emplace_back(parameter<double>(
      "BARRIER_FORCE_DISTANCE", {.description = "barrier force distance", .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("BARRIER_FORCE_TEMPSCALE",
      {.description = "barrier force temperature scaling", .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("BARRIER_FORCE_STIFF_HEAVY",
      {.description = "barrier force stiffness of heavy phase", .default_value = -1.0}));
  particledynsph.specs.emplace_back(parameter<double>("BARRIER_FORCE_DAMP_HEAVY",
      {.description = "barrier force damping parameter of heavy phase", .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("BARRIER_FORCE_STIFF_GAS",
      {.description = "barrier force stiffness of gas phase", .default_value = -1.0}));
  particledynsph.specs.emplace_back(parameter<double>("BARRIER_FORCE_DAMP_GAS",
      {.description = "barrier force damping parameter of gas phase", .default_value = 0.0}));

  // linear transition in surface tension evaluation
  particledynsph.specs.emplace_back(parameter<double>("TRANS_REF_TEMPERATURE",
      {.description = "transition reference temperature", .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("TRANS_DT_SURFACETENSION",
      {.description = "transition temperature difference for surface tension evaluation",
          .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("TRANS_DT_MARANGONI",
      {.description = "transition temperature difference for marangoni evaluation",
          .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("TRANS_DT_CURVATURE",
      {.description = "transition temperature difference for curvature evaluation",
          .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("TRANS_DT_WETTING",
      {.description = "transition temperature difference for wetting evaluation",
          .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("TRANS_DT_INTVISC",
      {.description = "transition temperature difference for interface viscosity evaluation",
          .default_value = 0.0}));
  particledynsph.specs.emplace_back(parameter<double>("TRANS_DT_BARRIER",
      {.description = "transition temperature difference for barrier force evaluation",
          .default_value = 0.0}));

  // type of dirichlet open boundary
  Core::Utils::string_to_integral_parameter<DirichletOpenBoundaryType>("DIRICHLETBOUNDARYTYPE",
      "NoDirichletOpenBoundary", "type of dirichlet open boundary",
      tuple<std::string>("NoDirichletOpenBoundary", "DirichletNormalToPlane"),
      tuple<DirichletOpenBoundaryType>(
          Inpar::PARTICLE::NoDirichletOpenBoundary, Inpar::PARTICLE::DirichletNormalToPlane),
      particledynsph);

  particledynsph.specs.emplace_back(parameter<int>("DIRICHLET_FUNCT",
      {.description = "number of function governing velocity condition on dirichlet open boundary",
          .default_value = -1}));

  particledynsph.specs.emplace_back(parameter<std::string>("DIRICHLET_OUTWARD_NORMAL",
      {.description = "direction of outward normal on dirichlet open boundary",
          .default_value = "0.0 0.0 0.0"}));
  particledynsph.specs.emplace_back(parameter<std::string>("DIRICHLET_PLANE_POINT",
      {.description = "point on dirichlet open boundary plane", .default_value = "0.0 0.0 0.0"}));

  // type of neumann open boundary
  Core::Utils::string_to_integral_parameter<NeumannOpenBoundaryType>("NEUMANNBOUNDARYTYPE",
      "NoNeumannOpenBoundary", "type of neumann open boundary",
      tuple<std::string>("NoNeumannOpenBoundary", "NeumannNormalToPlane"),
      tuple<NeumannOpenBoundaryType>(
          Inpar::PARTICLE::NoNeumannOpenBoundary, Inpar::PARTICLE::NeumannNormalToPlane),
      particledynsph);

  particledynsph.specs.emplace_back(parameter<int>("NEUMANN_FUNCT",
      {.description = "number of function governing pressure condition on neumann open boundary",
          .default_value = -1}));

  particledynsph.specs.emplace_back(parameter<std::string>("NEUMANN_OUTWARD_NORMAL",
      {.description = "direction of outward normal on neumann open boundary",
          .default_value = "0.0 0.0 0.0"}));
  particledynsph.specs.emplace_back(parameter<std::string>("NEUMANN_PLANE_POINT",
      {.description = "point on neumann open boundary plane", .default_value = "0.0 0.0 0.0"}));

  // type of phase change
  Core::Utils::string_to_integral_parameter<PhaseChangeType>("PHASECHANGETYPE", "NoPhaseChange",
      "type of phase change",
      tuple<std::string>("NoPhaseChange", "OneWayScalarBelowToAbovePhaseChange",
          "OneWayScalarAboveToBelowPhaseChange", "TwoWayScalarPhaseChange"),
      tuple<PhaseChangeType>(Inpar::PARTICLE::NoPhaseChange,
          Inpar::PARTICLE::OneWayScalarBelowToAbovePhaseChange,
          Inpar::PARTICLE::OneWayScalarAboveToBelowPhaseChange,
          Inpar::PARTICLE::TwoWayScalarPhaseChange),
      particledynsph);

  // definition of phase change
  particledynsph.specs.emplace_back(parameter<std::string>("PHASECHANGEDEFINITION",
      {.description = "phase change definition", .default_value = "none"}));

  // type of rigid particle contact
  Core::Utils::string_to_integral_parameter<RigidParticleContactType>("RIGIDPARTICLECONTACTTYPE",
      "NoRigidParticleContact", "type of rigid particle contact",
      tuple<std::string>("NoRigidParticleContact", "ElasticRigidParticleContact"),
      tuple<RigidParticleContactType>(
          Inpar::PARTICLE::NoRigidParticleContact, Inpar::PARTICLE::ElasticRigidParticleContact),
      particledynsph);

  particledynsph.specs.emplace_back(parameter<double>("RIGIDPARTICLECONTACTSTIFF",
      {.description = "rigid particle contact stiffness", .default_value = -1.0}));
  particledynsph.specs.emplace_back(parameter<double>("RIGIDPARTICLECONTACTDAMP",
      {.description = "rigid particle contact damping parameter", .default_value = 0.0}));

  particledynsph.move_into_collection(list);

  /*-------------------------------------------------------------------------*
   | discrete element method (DEM) specific control parameters               |
   *-------------------------------------------------------------------------*/
  Core::Utils::SectionSpecs particledyndem{particledyn, "DEM"};

  // write particle energy output
  particledyndem.specs.emplace_back(parameter<bool>("WRITE_PARTICLE_ENERGY",
      {.description = "write particle energy output", .default_value = false}));

  // write particle-wall interaction output
  particledyndem.specs.emplace_back(parameter<bool>("WRITE_PARTICLE_WALL_INTERACTION",
      {.description = "write particle-wall interaction output", .default_value = false}));

  // type of normal contact law
  Core::Utils::string_to_integral_parameter<NormalContact>("NORMALCONTACTLAW", "NormalLinearSpring",
      "normal contact law for particles",
      tuple<std::string>("NormalLinearSpring", "NormalLinearSpringDamp", "NormalHertz",
          "NormalLeeHerrmann", "NormalKuwabaraKono", "NormalTsuji"),
      tuple<NormalContact>(Inpar::PARTICLE::NormalLinSpring, Inpar::PARTICLE::NormalLinSpringDamp,
          Inpar::PARTICLE::NormalHertz, Inpar::PARTICLE::NormalLeeHerrmann,
          Inpar::PARTICLE::NormalKuwabaraKono, Inpar::PARTICLE::NormalTsuji),
      particledyndem);

  // type of tangential contact law
  Core::Utils::string_to_integral_parameter<TangentialContact>("TANGENTIALCONTACTLAW",
      "NoTangentialContact", "tangential contact law for particles",
      tuple<std::string>("NoTangentialContact", "TangentialLinSpringDamp"),
      tuple<TangentialContact>(
          Inpar::PARTICLE::NoTangentialContact, Inpar::PARTICLE::TangentialLinSpringDamp),
      particledyndem);

  // type of rolling contact law
  Core::Utils::string_to_integral_parameter<RollingContact>("ROLLINGCONTACTLAW", "NoRollingContact",
      "rolling contact law for particles",
      tuple<std::string>("NoRollingContact", "RollingViscous", "RollingCoulomb"),
      tuple<RollingContact>(Inpar::PARTICLE::NoRollingContact, Inpar::PARTICLE::RollingViscous,
          Inpar::PARTICLE::RollingCoulomb),
      particledyndem);

  // type of normal adhesion law
  Core::Utils::string_to_integral_parameter<AdhesionLaw>("ADHESIONLAW", "NoAdhesion",
      "type of adhesion law for particles",
      tuple<std::string>("NoAdhesion", "AdhesionVdWDMT", "AdhesionRegDMT"),
      tuple<AdhesionLaw>(Inpar::PARTICLE::NoAdhesion, Inpar::PARTICLE::AdhesionVdWDMT,
          Inpar::PARTICLE::AdhesionRegDMT),
      particledyndem);

  // type of (random) surface energy distribution
  Core::Utils::string_to_integral_parameter<SurfaceEnergyDistribution>(
      "ADHESION_SURFACE_ENERGY_DISTRIBUTION", "ConstantSurfaceEnergy",
      "type of (random) surface energy distribution",
      tuple<std::string>("ConstantSurfaceEnergy", "NormalSurfaceEnergyDistribution",
          "LogNormalSurfaceEnergyDistribution"),
      tuple<SurfaceEnergyDistribution>(Inpar::PARTICLE::ConstantSurfaceEnergy,
          Inpar::PARTICLE::NormalSurfaceEnergyDistribution,
          Inpar::PARTICLE::LogNormalSurfaceEnergyDistribution),
      particledyndem);

  particledyndem.specs.emplace_back(parameter<double>(
      "MIN_RADIUS", {.description = "minimum allowed particle radius", .default_value = 0.0}));
  particledyndem.specs.emplace_back(parameter<double>(
      "MAX_RADIUS", {.description = "maximum allowed particle radius", .default_value = 0.0}));
  particledyndem.specs.emplace_back(parameter<double>("MAX_VELOCITY",
      {.description = "maximum expected particle velocity", .default_value = -1.0}));

  // type of initial particle radius assignment
  Core::Utils::string_to_integral_parameter<InitialRadiusAssignment>("INITIAL_RADIUS",
      "RadiusFromParticleMaterial", "type of initial particle radius assignment",
      tuple<std::string>("RadiusFromParticleMaterial", "RadiusFromParticleInput",
          "NormalRadiusDistribution", "LogNormalRadiusDistribution"),
      tuple<InitialRadiusAssignment>(Inpar::PARTICLE::RadiusFromParticleMaterial,
          Inpar::PARTICLE::RadiusFromParticleInput, Inpar::PARTICLE::NormalRadiusDistribution,
          Inpar::PARTICLE::LogNormalRadiusDistribution),
      particledyndem);

  particledyndem.specs.emplace_back(parameter<double>("RADIUSDISTRIBUTION_SIGMA",
      {.description = "sigma of random particle radius distribution", .default_value = -1.0}));

  particledyndem.specs.emplace_back(parameter<double>("REL_PENETRATION",
      {.description = "maximum allowed relative penetration", .default_value = -1.0}));
  particledyndem.specs.emplace_back(parameter<double>(
      "NORMAL_STIFF", {.description = "normal contact stiffness", .default_value = -1.0}));
  particledyndem.specs.emplace_back(parameter<double>(
      "NORMAL_DAMP", {.description = "normal contact damping parameter", .default_value = -1.0}));
  particledyndem.specs.emplace_back(parameter<double>(
      "COEFF_RESTITUTION", {.description = "coefficient of restitution", .default_value = -1.0}));
  particledyndem.specs.emplace_back(parameter<double>(
      "DAMP_REG_FAC", {.description = "linearly regularized damping normal force in the interval "
                                      "$|g| < (\\text{DAMP_REG_FAC} \\cdot r_{\\min})$",
                          .default_value = -1.0}));
  particledyndem.specs.emplace_back(parameter<bool>("TENSION_CUTOFF",
      {.description = "evaluate tension cutoff of normal contact force", .default_value = true}));

  particledyndem.specs.emplace_back(
      parameter<double>("POISSON_RATIO", {.description = "poisson ratio", .default_value = -1.0}));
  particledyndem.specs.emplace_back(parameter<double>(
      "YOUNG_MODULUS", {.description = "young's modulus", .default_value = -1.0}));

  particledyndem.specs.emplace_back(parameter<double>("FRICT_COEFF_TANG",
      {.description = "friction coefficient for tangential contact", .default_value = -1.0}));
  particledyndem.specs.emplace_back(parameter<double>("FRICT_COEFF_ROLL",
      {.description = "friction coefficient for rolling contact", .default_value = -1.0}));

  particledyndem.specs.emplace_back(parameter<double>("ADHESION_DISTANCE",
      {.description = "adhesion distance between interacting surfaces", .default_value = -1.0}));

  particledyndem.specs.emplace_back(parameter<double>("ADHESION_MAX_CONTACT_PRESSURE",
      {.description = "adhesion maximum contact pressure", .default_value = 0.0}));
  particledyndem.specs.emplace_back(parameter<double>("ADHESION_MAX_CONTACT_FORCE",
      {.description = "adhesion maximum contact force", .default_value = 0.0}));
  particledyndem.specs.emplace_back(parameter<bool>("ADHESION_USE_MAX_CONTACT_FORCE",
      {.description = "use maximum contact force instead of maximum contact pressure",
          .default_value = false}));

  particledyndem.specs.emplace_back(parameter<bool>("ADHESION_VDW_CURVE_SHIFT",
      {.description = "shifts van-der-Waals-curve to g = 0", .default_value = false}));

  particledyndem.specs.emplace_back(parameter<double>("ADHESION_HAMAKER",
      {.description = "hamaker constant of van-der-Waals interaction", .default_value = -1.0}));

  particledyndem.specs.emplace_back(parameter<double>("ADHESION_SURFACE_ENERGY",
      {.description = "adhesion surface energy for the calculation of the pull-out force",
          .default_value = -1.0}));
  particledyndem.specs.emplace_back(parameter<double>("ADHESION_SURFACE_ENERGY_DISTRIBUTION_VAR",
      {.description = "variance of adhesion surface energy distribution", .default_value = -1.0}));
  particledyndem.specs.emplace_back(
      parameter<double>("ADHESION_SURFACE_ENERGY_DISTRIBUTION_CUTOFF_FACTOR",
          {.description = "adhesion surface energy distribution limited by multiple of variance",
              .default_value = -1.0}));
  particledyndem.specs.emplace_back(parameter<double>("ADHESION_SURFACE_ENERGY_FACTOR",
      {.description = "factor to calculate minimum adhesion surface energy",
          .default_value = 1.0}));

  particledyndem.move_into_collection(list);
}

/*---------------------------------------------------------------------------*
 | set the particle conditions                                               |
 *---------------------------------------------------------------------------*/
void Inpar::PARTICLE::set_valid_conditions(
    std::vector<Core::Conditions::ConditionDefinition>& condlist)
{
  using namespace Core::IO::InputSpecBuilders;

  /*-------------------------------------------------------------------------*
   | particle wall condition                                                 |
   *-------------------------------------------------------------------------*/
  Core::Conditions::ConditionDefinition surfpartwall("DESIGN SURFACE PARTICLE WALL", "ParticleWall",
      "Wall for particle interaction with (optional) material definition",
      Core::Conditions::ParticleWall, true, Core::Conditions::geometry_type_surface);

  surfpartwall.add_component(parameter<int>("MAT"));

  condlist.push_back(surfpartwall);
}

FOUR_C_NAMESPACE_CLOSE
