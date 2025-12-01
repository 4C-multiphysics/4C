// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_input.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_io_input_spec_builders.hpp"
FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | set the particle parameters                                               |
 *---------------------------------------------------------------------------*/
std::vector<Core::IO::InputSpec> Particle::valid_parameters()
{
  using namespace Core::IO::InputSpecBuilders;
  using namespace Core::IO::InputSpecBuilders::Validators;

  /*-------------------------------------------------------------------------*
   | general control parameters for particle simulations                     |
   *-------------------------------------------------------------------------*/

  std::vector<Core::IO::InputSpec> specs;
  specs.push_back(group("PARTICLE DYNAMIC",
      {

          // type of particle time integration
          deprecated_selection<DynamicType>("DYNAMICTYPE",
              {
                  {"SemiImplicitEuler", Particle::dyna_semiimpliciteuler},
                  {"VelocityVerlet", Particle::dyna_velocityverlet},
              },
              {.description = "type of particle time integration",
                  .default_value = Particle::dyna_velocityverlet}),

          // type of particle interaction
          deprecated_selection<InteractionType>("INTERACTION",
              {
                  {"None", Particle::interaction_none},
                  {"SPH", Particle::interaction_sph},
                  {"DEM", Particle::interaction_dem},
              },
              {.description = "type of particle interaction",
                  .default_value = Particle::interaction_none}),

          // output type
          parameter<int>("RESULTSEVERY",
              {.description = "write particle runtime output every RESULTSEVERY steps",
                  .default_value = 1}),
          parameter<int>(
              "RESTARTEVERY", {.description = "write restart possibility every RESTARTEVERY steps",
                                  .default_value = 1}),

          // write ghosted particles
          parameter<bool>("WRITE_GHOSTED_PARTICLES",
              {.description = "write ghosted particles (debug feature)", .default_value = false}),

          // time loop control
          parameter<double>("TIMESTEP", {.description = "time step size", .default_value = 0.01}),

          parameter<int>(
              "NUMSTEP", {.description = "maximum number of steps", .default_value = 100}),

          parameter<double>("MAXTIME", {.description = "maximum time", .default_value = 1.0}),

          // gravity acceleration control
          parameter<std::string>("GRAVITY_ACCELERATION",
              {.description = "acceleration due to gravity", .default_value = "0.0 0.0 0.0"}),
          parameter<int>("GRAVITY_RAMP_FUNCT",
              {.description = "number of function governing gravity ramp", .default_value = -1}),

          // viscous damping factor
          parameter<double>("VISCOUS_DAMPING",
              {.description =
                      "apply viscous damping force to determine static equilibrium solutions",
                  .default_value = -1.0}),

          // transfer particles to new bins every time step
          parameter<bool>(
              "TRANSFER_EVERY", {.description = "transfer particles to new bins every time step",
                                    .default_value = false}),

          // considered particle phases with dynamic load balance weighting factor
          parameter<std::string>("PHASE_TO_DYNLOADBALFAC",
              {.description =
                      "considered particle phases with dynamic load balance weighting factor",
                  .default_value = "none"}),

          // relate particle phase to material id
          parameter<std::string>("PHASE_TO_MATERIAL_ID",
              {.description = "relate particle phase to material id", .default_value = "none"}),

          // amplitude of noise added to initial position for each spatial direction
          parameter<std::string>("INITIAL_POSITION_AMPLITUDE",
              {.description =
                      "amplitude of noise added to initial position for each spatial direction",
                  .default_value = "0.0 0.0 0.0"}),

          // type of particle wall source
          parameter<ParticleWallSource>(
              "PARTICLE_WALL_SOURCE", {.description = "type of particle wall source",
                                          .default_value = Particle::NoParticleWall}),

          // material id for particle wall from bounding box source
          parameter<int>("PARTICLE_WALL_MAT",
              {.description = "material id for particle wall from bounding box source",
                  .default_value = -1}),

          // flags defining considered states of particle wall
          parameter<bool>("PARTICLE_WALL_MOVING",
              {.description = "consider a moving particle wall", .default_value = false}),
          parameter<bool>("PARTICLE_WALL_LOADED",
              {.description = "consider loading on particle wall", .default_value = false}),

          // consider rigid body motion
          parameter<bool>("RIGID_BODY_MOTION",
              {.description = "consider rigid body motion", .default_value = false}),

          parameter<double>("RIGID_BODY_PHASECHANGE_RADIUS",
              {.description = "search radius for neighboring rigid bodies in case of phase change",
                  .default_value = -1.0}),
          parameter<bool>("PD_BODY_INTERACTION",
              {.description = "consider peridynamic body interaction", .default_value = false})},
      {.required = false}));
  /*-------------------------------------------------------------------------*
 | control parameters for initial/boundary conditions |
 *-------------------------------------------------------------------------*/
  specs.push_back(group("PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS",
      {

          // initial temperature field of particle phase given by function
          parameter<std::string>("INITIAL_TEMP_FIELD",
              {.description = "Refer to the function ID describing the initial "
                              "temperature field of particle phase",
                  .default_value = "none"}),

          // initial velocity field of particle phase given by function
          parameter<std::string>(
              "INITIAL_VELOCITY_FIELD", {.description = "Refer to the function ID describing the "
                                                        "initial velocity field of particle phase",
                                            .default_value = "none"}),

          // initial angular velocity field of particle phase given by function
          parameter<std::string>("INITIAL_ANGULAR_VELOCITY_FIELD",
              {.description =
                      "Refer to the function ID describing the initial angular velocity field of "
                      "rigid body  phase/DEM particle",
                  .default_value = "none"}),


          // initial acceleration field of particle phase given by function
          parameter<std::string>("INITIAL_ACCELERATION_FIELD",
              {.description = "Refer to the function ID describing the "
                              "initial acceleration field of particle phase",
                  .default_value = "none"}),

          // initial angular acceleration field of particle phase given by function
          parameter<std::string>("INITIAL_ANGULAR_ACCELERATION_FIELD",
              {.description =
                      "Refer to the function ID describing the initial angular acceleration "
                      "field of rigid body  phase/DEM particle",
                  .default_value = "none"}),


          // dirichlet boundary condition of particle phase given by function
          parameter<std::string>("DIRICHLET_BOUNDARY_CONDITION",
              {.description =
                      "Refer to the function ID describing the dirichlet boundary condition of "
                      "particle phase",
                  .default_value = "none"}),

          // temperature boundary condition of particle phase given by function
          parameter<std::string>("TEMPERATURE_BOUNDARY_CONDITION",
              {.description =
                      "Refer to the function ID describing the temperature boundary condition of "
                      "particle phase",
                  .default_value = "none"})},
      {.required = false}));

  /*-------------------------------------------------------------------------*
   | smoothed particle hydrodynamics (SPH) specific control parameters       |
   *-------------------------------------------------------------------------*/
  specs.push_back(group("PARTICLE DYNAMIC/SPH",
      {

          // write particle-wall interaction output
          parameter<bool>("WRITE_PARTICLE_WALL_INTERACTION",
              {.description = "write particle-wall interaction output", .default_value = false}),

          // type of smoothed particle hydrodynamics kernel
          parameter<KernelType>(
              "KERNEL", {.description = "type of smoothed particle hydrodynamics kernel",
                            .default_value = Particle::CubicSpline}),

          // kernel space dimension number
          parameter<KernelSpaceDimension>(
              "KERNEL_SPACE_DIM", {.description = "kernel space dimension number",
                                      .default_value = Particle::Kernel3D}),

          parameter<double>("INITIALPARTICLESPACING",
              {.description = "initial spacing of particles", .default_value = 0.0}),

          // type of smoothed particle hydrodynamics equation of state
          parameter<EquationOfStateType>("EQUATIONOFSTATE",
              {.description = "type of smoothed particle hydrodynamics equation of state",
                  .default_value = Particle::GenTait}),

          // type of smoothed particle hydrodynamics momentum formulation
          parameter<MomentumFormulationType>("MOMENTUMFORMULATION",
              {.description = "type of smoothed particle hydrodynamics momentum formulation",
                  .default_value = Particle::AdamiMomentumFormulation}),

          // type of density evaluation scheme
          parameter<DensityEvaluationScheme>(
              "DENSITYEVALUATION", {.description = "type of density evaluation scheme",
                                       .default_value = Particle::DensitySummation}),

          // type of density correction scheme
          parameter<DensityCorrectionScheme>(
              "DENSITYCORRECTION", {.description = "type of density correction scheme",
                                       .default_value = Particle::NoCorrection}),

          // type of boundary particle formulation
          parameter<BoundaryParticleFormulationType>("BOUNDARYPARTICLEFORMULATION",
              {.description = "type of boundary particle formulation",
                  .default_value = Particle::NoBoundaryFormulation}),

          // type of boundary particle interaction
          parameter<BoundaryParticleInteraction>("BOUNDARYPARTICLEINTERACTION",
              {.description = "type of boundary particle interaction",
                  .default_value = Particle::NoSlipBoundaryParticle}),

          // type of wall formulation
          parameter<WallFormulationType>(
              "WALLFORMULATION", {.description = "type of wall formulation",
                                     .default_value = Particle::NoWallFormulation}),

          // type of transport velocity formulation
          parameter<TransportVelocityFormulation>("TRANSPORTVELOCITYFORMULATION",
              {.description = "type of transport velocity formulation",
                  .default_value = Particle::NoTransportVelocity}),

          // type of temperature evaluation scheme
          parameter<TemperatureEvaluationScheme>(
              "TEMPERATUREEVALUATION", {.description = "type of temperature evaluation scheme",
                                           .default_value = Particle::NoTemperatureEvaluation}),

          parameter<bool>("TEMPERATUREGRADIENT",
              {.description = "evaluate temperature gradient", .default_value = false}),

          // type of heat source
          parameter<HeatSourceType>("HEATSOURCETYPE",
              {.description = "type of heat source", .default_value = Particle::NoHeatSource}),

          parameter<int>("HEATSOURCE_FUNCT",
              {.description = "number of function governing heat source", .default_value = -1}),

          parameter<std::string>("HEATSOURCE_DIRECTION",
              {.description = "direction of surface heat source", .default_value = "0.0 0.0 0.0"}),

          // evaporation induced heat loss
          parameter<bool>("VAPOR_HEATLOSS",
              {.description = "evaluate evaporation induced heat loss", .default_value = false}),
          parameter<double>("VAPOR_HEATLOSS_LATENTHEAT",
              {.description = "latent heat in heat loss formula", .default_value = 0.0}),
          parameter<double>("VAPOR_HEATLOSS_ENTHALPY_REFTEMP",
              {.description = "enthalpy reference temperature in heat loss formula",
                  .default_value = 0.0}),
          parameter<double>("VAPOR_HEATLOSS_PFAC",
              {.description = "pressure factor in heat loss formula", .default_value = 0.0}),
          parameter<double>("VAPOR_HEATLOSS_TFAC",
              {.description = "temperature factor in heat loss formula", .default_value = 0.0}),

          // evaporation induced recoil pressure
          parameter<bool>(
              "VAPOR_RECOIL", {.description = "evaluate evaporation induced recoil pressure",
                                  .default_value = false}),
          parameter<double>("VAPOR_RECOIL_BOILINGTEMPERATURE",
              {.description = "boiling temperature in recoil pressure formula",
                  .default_value = 0.0}),
          parameter<double>("VAPOR_RECOIL_PFAC",
              {.description = "pressure factor in recoil pressure formula", .default_value = 0.0}),
          parameter<double>(
              "VAPOR_RECOIL_TFAC", {.description = "temperature factor in recoil pressure formula",
                                       .default_value = 0.0}),

          // type of surface tension formulation
          parameter<SurfaceTensionFormulation>(
              "SURFACETENSIONFORMULATION", {.description = "type of surface tension formulation",
                                               .default_value = Particle::NoSurfaceTension}),

          parameter<int>("SURFACETENSION_RAMP_FUNCT",
              {.description = "number of function governing surface tension ramp",
                  .default_value = -1}),

          parameter<double>("SURFACETENSIONCOEFFICIENT",
              {.description = "constant part of surface tension coefficient",
                  .default_value = -1.0}),
          parameter<double>("SURFACETENSIONMINIMUM",
              {.description =
                      "minimum surface tension coefficient in case of temperature dependence",
                  .default_value = 0.0}),
          parameter<double>("SURFACETENSIONTEMPFAC",
              {.description = "factor of dependence of surface tension coefficient on temperature",
                  .default_value = 0.0}),
          parameter<double>("SURFACETENSIONREFTEMP",
              {.description = "reference temperature for surface tension coefficient",
                  .default_value = 0.0}),

          // wetting
          parameter<double>("STATICCONTACTANGLE",
              {.description = "static contact angle in degree with wetting effects",
                  .default_value = 0.0}),
          parameter<double>("TRIPLEPOINTNORMAL_CORR_CF_LOW",
              {.description = "triple point normal correction wall color field low",
                  .default_value = 0.0}),
          parameter<double>("TRIPLEPOINTNORMAL_CORR_CF_UP",
              {.description = "triple point normal correction wall color field up",
                  .default_value = 0.0}),

          // interface viscosity
          parameter<bool>("INTERFACE_VISCOSITY",
              {.description = "evaluate interface viscosity", .default_value = false}),
          parameter<double>("INTERFACE_VISCOSITY_LIQUIDGAS",
              {.description = "artificial viscosity on liquid-gas interface",
                  .default_value = 0.0}),
          parameter<double>("INTERFACE_VISCOSITY_SOLIDLIQUID",
              {.description = "artificial viscosity on solid-liquid interface",
                  .default_value = 0.0}),

          // barrier force
          parameter<bool>(
              "BARRIER_FORCE", {.description = "evaluate barrier force", .default_value = false}),
          parameter<double>("BARRIER_FORCE_DISTANCE",
              {.description = "barrier force distance", .default_value = 0.0}),
          parameter<double>("BARRIER_FORCE_TEMPSCALE",
              {.description = "barrier force temperature scaling", .default_value = 0.0}),
          parameter<double>("BARRIER_FORCE_STIFF_HEAVY",
              {.description = "barrier force stiffness of heavy phase", .default_value = -1.0}),
          parameter<double>("BARRIER_FORCE_DAMP_HEAVY",
              {.description = "barrier force damping parameter of heavy phase",
                  .default_value = 0.0}),
          parameter<double>("BARRIER_FORCE_STIFF_GAS",
              {.description = "barrier force stiffness of gas phase", .default_value = -1.0}),
          parameter<double>("BARRIER_FORCE_DAMP_GAS",
              {.description = "barrier force damping parameter of gas phase",
                  .default_value = 0.0}),

          // linear transition in surface tension evaluation
          parameter<double>("TRANS_REF_TEMPERATURE",
              {.description = "transition reference temperature", .default_value = 0.0}),
          parameter<double>("TRANS_DT_SURFACETENSION",
              {.description = "transition temperature difference for surface tension evaluation",
                  .default_value = 0.0}),
          parameter<double>("TRANS_DT_MARANGONI",
              {.description = "transition temperature difference for marangoni evaluation",
                  .default_value = 0.0}),
          parameter<double>("TRANS_DT_CURVATURE",
              {.description = "transition temperature difference for curvature evaluation",
                  .default_value = 0.0}),
          parameter<double>("TRANS_DT_WETTING",
              {.description = "transition temperature difference for wetting evaluation",
                  .default_value = 0.0}),
          parameter<double>("TRANS_DT_INTVISC",
              {.description =
                      "transition temperature difference for interface viscosity evaluation",
                  .default_value = 0.0}),
          parameter<double>("TRANS_DT_BARRIER",
              {.description = "transition temperature difference for barrier force evaluation",
                  .default_value = 0.0}),

          // type of dirichlet open boundary
          parameter<DirichletOpenBoundaryType>(
              "DIRICHLETBOUNDARYTYPE", {.description = "type of dirichlet open boundary",
                                           .default_value = Particle::NoDirichletOpenBoundary}),

          parameter<int>("DIRICHLET_FUNCT", {.description = "number of function governing velocity "
                                                            "condition on dirichlet open boundary",
                                                .default_value = -1}),

          parameter<std::string>("DIRICHLET_OUTWARD_NORMAL",
              {.description = "direction of outward normal on dirichlet open boundary",
                  .default_value = "0.0 0.0 0.0"}),
          parameter<std::string>(
              "DIRICHLET_PLANE_POINT", {.description = "point on dirichlet open boundary plane",
                                           .default_value = "0.0 0.0 0.0"}),

          // type of neumann open boundary
          parameter<NeumannOpenBoundaryType>(
              "NEUMANNBOUNDARYTYPE", {.description = "type of neumann open boundary",
                                         .default_value = Particle::NoNeumannOpenBoundary}),

          parameter<int>("NEUMANN_FUNCT",
              {.description =
                      "number of function governing pressure condition on neumann open boundary",
                  .default_value = -1}),

          parameter<std::string>("NEUMANN_OUTWARD_NORMAL",
              {.description = "direction of outward normal on neumann open boundary",
                  .default_value = "0.0 0.0 0.0"}),
          parameter<std::string>(
              "NEUMANN_PLANE_POINT", {.description = "point on neumann open boundary plane",
                                         .default_value = "0.0 0.0 0.0"}),

          // type of phase change
          parameter<PhaseChangeType>("PHASECHANGETYPE",
              {.description = "type of phase change", .default_value = Particle::NoPhaseChange}),

          // definition of phase change
          parameter<std::string>("PHASECHANGEDEFINITION",
              {.description = "phase change definition", .default_value = "none"}),

          // type of rigid particle contact
          parameter<RigidParticleContactType>(
              "RIGIDPARTICLECONTACTTYPE", {.description = "type of rigid particle contact",
                                              .default_value = Particle::NoRigidParticleContact}),

          parameter<double>("RIGIDPARTICLECONTACTSTIFF",
              {.description = "rigid particle contact stiffness", .default_value = -1.0}),
          parameter<double>("RIGIDPARTICLECONTACTDAMP",
              {.description = "rigid particle contact damping parameter", .default_value = 0.0})},
      {.required = false}));

  /*-------------------------------------------------------------------------*
   | discrete element method (DEM) specific control parameters               |
   *-------------------------------------------------------------------------*/
  specs.push_back(group("PARTICLE DYNAMIC/DEM",
      {

          // write particle energy output
          parameter<bool>("WRITE_PARTICLE_ENERGY",
              {.description = "write particle energy output", .default_value = false}),

          // write particle-wall interaction output
          parameter<bool>("WRITE_PARTICLE_WALL_INTERACTION",
              {.description = "write particle-wall interaction output", .default_value = false}),

          // type of normal contact law
          deprecated_selection<NormalContact>("NORMALCONTACTLAW",
              {
                  {"NormalLinearSpring", Particle::NormalLinSpring},
                  {"NormalLinearSpringDamp", Particle::NormalLinSpringDamp},
                  {"NormalHertz", Particle::NormalHertz},
                  {"NormalLeeHerrmann", Particle::NormalLeeHerrmann},
                  {"NormalKuwabaraKono", Particle::NormalKuwabaraKono},
                  {"NormalTsuji", Particle::NormalTsuji},
              },
              {.description = "normal contact law for particles",
                  .default_value = Particle::NormalLinSpring}),

          // type of tangential contact law
          parameter<TangentialContact>(
              "TANGENTIALCONTACTLAW", {.description = "tangential contact law for particles",
                                          .default_value = Particle::NoTangentialContact}),

          // type of rolling contact law
          parameter<RollingContact>(
              "ROLLINGCONTACTLAW", {.description = "rolling contact law for particles",
                                       .default_value = Particle::NoRollingContact}),

          // type of normal adhesion law
          parameter<AdhesionLaw>(
              "ADHESIONLAW", {.description = "type of adhesion law for particles",
                                 .default_value = Particle::NoAdhesion}),

          // type of (random) surface energy distribution
          parameter<SurfaceEnergyDistribution>("ADHESION_SURFACE_ENERGY_DISTRIBUTION",
              {.description = "type of (random) surface energy distribution",
                  .default_value = SurfaceEnergyDistribution::Constant}),

          parameter<double>("MIN_RADIUS",
              {.description = "minimum allowed particle radius", .default_value = 0.0}),
          parameter<double>("MAX_RADIUS",
              {.description = "maximum allowed particle radius", .default_value = 0.0}),
          parameter<double>("MAX_VELOCITY",
              {.description = "maximum expected particle velocity", .default_value = -1.0}),

          // type of initial particle radius assignment
          parameter<InitialRadiusAssignment>(
              "INITIAL_RADIUS", {.description = "type of initial particle radius assignment",
                                    .default_value = Particle::RadiusFromParticleMaterial}),

          parameter<std::optional<double>>("RADIUSDISTRIBUTION_SIGMA",
              {
                  .description = "Standard deviation sigma of random particle radius distribution",
                  .validator = null_or(positive<double>()),
              }),

          parameter<double>("REL_PENETRATION",
              {.description = "maximum allowed relative penetration", .default_value = -1.0}),
          parameter<double>(
              "NORMAL_STIFF", {.description = "normal contact stiffness", .default_value = -1.0}),
          parameter<double>("NORMAL_DAMP",
              {.description = "normal contact damping parameter", .default_value = -1.0}),
          parameter<double>("COEFF_RESTITUTION",
              {.description = "coefficient of restitution", .default_value = -1.0}),
          parameter<double>("DAMP_REG_FAC",
              {.description = "linearly regularized damping normal force in the interval "
                              "$|g| < (\\text{DAMP_REG_FAC} \\cdot r_{\\min})$",
                  .default_value = -1.0}),
          parameter<bool>(
              "TENSION_CUTOFF", {.description = "evaluate tension cutoff of normal contact force",
                                    .default_value = true}),


          parameter<double>(
              "POISSON_RATIO", {.description = "poisson ratio", .default_value = -1.0}),
          parameter<double>(
              "YOUNG_MODULUS", {.description = "young's modulus", .default_value = -1.0}),

          parameter<double>(
              "FRICT_COEFF_TANG", {.description = "friction coefficient for tangential contact",
                                      .default_value = -1.0}),
          parameter<double>("FRICT_COEFF_ROLL",
              {.description = "friction coefficient for rolling contact", .default_value = -1.0}),

          parameter<double>(
              "ADHESION_DISTANCE", {.description = "adhesion distance between interacting surfaces",
                                       .default_value = -1.0}),

          parameter<double>("ADHESION_MAX_CONTACT_PRESSURE",
              {.description = "adhesion maximum contact pressure", .default_value = 0.0}),
          parameter<double>("ADHESION_MAX_CONTACT_FORCE",
              {.description = "adhesion maximum contact force", .default_value = 0.0}),
          parameter<bool>("ADHESION_USE_MAX_CONTACT_FORCE",
              {.description = "use maximum contact force instead of maximum contact pressure",
                  .default_value = false}),

          parameter<bool>("ADHESION_VDW_CURVE_SHIFT",
              {.description = "shifts van-der-Waals-curve to g = 0", .default_value = false}),

          parameter<double>(
              "ADHESION_HAMAKER", {.description = "hamaker constant of van-der-Waals interaction",
                                      .default_value = -1.0}),

          parameter<double>("ADHESION_SURFACE_ENERGY",
              {.description = "adhesion surface energy for the calculation of the pull-out force",
                  .default_value = -1.0}),
          parameter<double>("ADHESION_SURFACE_ENERGY_DISTRIBUTION_STDDEV",
              {.description = "standard deviation of adhesion surface energy distribution",
                  .default_value = -1.0}),

          parameter<double>("ADHESION_SURFACE_ENERGY_DISTRIBUTION_CUTOFF_FACTOR",
              {.description = "adhesion surface energy distribution limited by multiple of "
                              "standard deviation",
                  .default_value = -1.0}),
          parameter<double>("ADHESION_SURFACE_ENERGY_FACTOR",
              {.description = "factor to calculate minimum adhesion surface energy",
                  .default_value = 1.0})},
      {.required = false}));

  /*-------------------------------------------------------------------------*
   | peridynamics (PD) specific control parameters                           |
   *-------------------------------------------------------------------------*/
  specs.push_back(group("PARTICLE DYNAMIC/PD",
      {

          // compact support in peridynamics
          parameter<double>(
              "INTERACTION_HORIZON", {.description = "peridynamic horizon", .default_value = 0.0}),

          // the spatial grid spacing of the peridynamic grid
          parameter<double>("PERIDYNAMIC_GRID_SPACING",
              {.description = "peridynamic grid spacing", .default_value = 0.0}),

          // type of normal contact law
          deprecated_selection<NormalContact>("NORMALCONTACTLAW",
              {
                  {"NormalLinearSpring", Particle::NormalLinSpring},
                  {"NormalLinearSpringDamp", Particle::NormalLinSpringDamp},
              },
              {.description = "normal contact law for particles between peridynamic bodies",
                  .default_value = Particle::NormalLinSpring}),

          parameter<double>(
              "NORMAL_STIFF", {.description = "normal contact stiffness", .default_value = 0.0}),
          parameter<double>("NORMAL_DAMP",
              {.description = "normal contact damping parameter", .default_value = 0.0})},
      {.required = false}));
  return specs;
}

/*---------------------------------------------------------------------------*
 | set the particle conditions                                               |
 *---------------------------------------------------------------------------*/
void Particle::set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist)
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
