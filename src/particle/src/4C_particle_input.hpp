// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INPUT_HPP
#define FOUR_C_PARTICLE_INPUT_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Conditions
{
  class ConditionDefinition;
}
/*---------------------------------------------------------------------------*
 | input parameters for particle problems                                    |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  /*---------------------------------------------------------------------------*
   | general control parameters for particle simulations                       |
   *---------------------------------------------------------------------------*/

  //! type of particle time integration
  enum DynamicType
  {
    dyna_semiimpliciteuler,  //! semi implicit euler scheme (explicit)
    dyna_velocityverlet      //! velocity verlet scheme (explicit)
  };

  //! type of particle interaction
  enum InteractionType
  {
    interaction_none,  //! no particle interaction
    interaction_sph,   //! smoothed particle hydrodynamics
    interaction_dem    //! discrete element method
  };

  //! data format for written numeric data via vtp
  enum OutputDataFormat
  {
    binary,
    ascii
  };

  //! type of particle wall source
  enum ParticleWallSource
  {
    NoParticleWall,    //! no particle wall
    DiscretCondition,  //! particle wall from discretization condition
    BoundingBox        //! particle wall from bounding box
  };

  /*---------------------------------------------------------------------------*
   | smoothed particle hydrodynamics (SPH) specific control parameters         |
   *---------------------------------------------------------------------------*/

  //! type of smoothed particle hydrodynamics kernel
  enum KernelType
  {
    CubicSpline,
    QuinticSpline
  };

  //! kernel space dimension number
  enum KernelSpaceDimension
  {
    Kernel1D,
    Kernel2D,
    Kernel3D
  };

  //! type of smoothed particle hydrodynamics equation of state
  enum EquationOfStateType
  {
    GenTait,
    IdealGas
  };

  //! type of smoothed particle hydrodynamics momentum formulation
  enum MomentumFormulationType
  {
    AdamiMomentumFormulation,
    MonaghanMomentumFormulation
  };

  //! type of density evaluation scheme
  enum DensityEvaluationScheme
  {
    DensitySummation,
    DensityIntegration,
    DensityPredictCorrect
  };

  //! type of density correction scheme
  enum DensityCorrectionScheme
  {
    NoCorrection,
    InteriorCorrection,
    NormalizedCorrection,
    RandlesCorrection
  };

  //! type of constraint projection
  enum Constraint
  {
    NoConstraint,
    Projection1D,
    Projection2D
  };

  //! type of boundary particle formulation
  enum BoundaryParticleFormulationType
  {
    NoBoundaryFormulation,
    AdamiBoundaryFormulation
  };

  //! type of boundary particle interaction
  enum BoundaryParticleInteraction
  {
    NoSlipBoundaryParticle,
    FreeSlipBoundaryParticle
  };

  //! type of wall formulation
  enum WallFormulationType
  {
    NoWallFormulation,
    VirtualParticleWallFormulation
  };

  //! type of transport velocity formulation
  enum TransportVelocityFormulation
  {
    NoTransportVelocity,
    StandardTransportVelocity,
    GeneralizedTransportVelocity
  };

  //! type of temperature evaluation scheme
  enum TemperatureEvaluationScheme
  {
    NoTemperatureEvaluation,
    TemperatureIntegration
  };

  //! type of heat source
  enum HeatSourceType
  {
    NoHeatSource,
    VolumeHeatSource,
    SurfaceHeatSource
  };

  //! type of surface tension formulation
  enum SurfaceTensionFormulation
  {
    NoSurfaceTension,
    ContinuumSurfaceForce
  };

  //! type of dirichlet open boundary
  enum DirichletOpenBoundaryType
  {
    NoDirichletOpenBoundary,
    DirichletNormalToPlane
  };

  //! type of neumann open boundary
  enum NeumannOpenBoundaryType
  {
    NoNeumannOpenBoundary,
    NeumannNormalToPlane
  };

  //! type of phase change
  enum PhaseChangeType
  {
    NoPhaseChange,
    OneWayScalarBelowToAbovePhaseChange,
    OneWayScalarAboveToBelowPhaseChange,
    TwoWayScalarPhaseChange
  };

  //! type of rigid particle contact
  enum RigidParticleContactType
  {
    NoRigidParticleContact,
    ElasticRigidParticleContact
  };

  /*---------------------------------------------------------------------------*
   | discrete element method (DEM) specific control parameters                 |
   *---------------------------------------------------------------------------*/

  //! type of normal contact law
  enum NormalContact
  {
    NormalLinSpring,
    NormalLinSpringDamp,
    NormalHertz,
    NormalLeeHerrmann,
    NormalKuwabaraKono,
    NormalTsuji
  };

  //! type of tangential contact law
  enum TangentialContact
  {
    NoTangentialContact,
    TangentialLinSpringDamp
  };

  //! type of rolling contact law
  enum RollingContact
  {
    NoRollingContact,
    RollingViscous,
    RollingCoulomb
  };

  //! type of adhesion law
  enum AdhesionLaw
  {
    NoAdhesion,
    AdhesionVdWDMT,
    AdhesionRegDMT
  };

  //! type of (random) surface energy distribution
  enum class SurfaceEnergyDistribution
  {
    Constant,
    Normal
  };

  //! type of initial particle radius assignment
  enum InitialRadiusAssignment
  {
    RadiusFromParticleMaterial,
    RadiusFromParticleInput,
    NormalRadiusDistribution,
    LogNormalRadiusDistribution
  };

  //! particle parameters
  std::vector<Core::IO::InputSpec> valid_parameters();

  //! set the particle conditions
  void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
