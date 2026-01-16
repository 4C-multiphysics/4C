// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ENGINE_ENUMS_HPP
#define FOUR_C_PARTICLE_ENGINE_ENUMS_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | enum definitions                                                          |
 *---------------------------------------------------------------------------*/
namespace Particle
{
  //! \name definition of particle states
  //! @{

  /*!
   * \brief enums of particle states
   *
   */
  enum ParticleState
  {
    /*---------------------------------------------------------------------------*/
    // default particle states
    Position,              //!< position
    Velocity,              //!< velocity
    Acceleration,          //!< acceleration
    LastTransferPosition,  //!< position after last particle transfer
    /*---------------------------------------------------------------------------*/
    // particle states for advanced time integration schemes
    ModifiedVelocity,      //!< modified velocity
    ModifiedAcceleration,  //!< modified acceleration
    /*---------------------------------------------------------------------------*/
    // particle states for boundary conditions
    ReferencePosition,  //!< reference position
    /*---------------------------------------------------------------------------*/
    // common particle states
    Radius,       //!< radius
    Mass,         //!< mass
    Density,      //!< density
    Pressure,     //!< pressure
    Temperature,  //!< temperature
    /*---------------------------------------------------------------------------*/
    // particle states for rigid body evaluation
    RigidBodyColor,             //!< rigid body color
    RelativePosition,           //!< position relative to rigid body position in reference frame
    RelativePositionBodyFrame,  //!< position relative to rigid body position in body frame
    Inertia,                    //!< mass moment of inertia
    /*---------------------------------------------------------------------------*/
    // particle states for SPH interaction
    DensitySum,            //!< density from summation of weighted masses
    DensityDot,            //!< first time derivative of density
    TemperatureDot,        //!< first time derivative of temperature
    BoundaryPressure,      //!< boundary pressure
    BoundaryVelocity,      //!< boundary velocity
    Colorfield,            //!< colorfield
    ColorfieldGradient,    //!< colorfield gradient
    InterfaceNormal,       //!< interface normal
    Curvature,             //!< curvature
    WallColorfield,        //!< wall colorfield
    WallInterfaceNormal,   //!< wall interface normal
    temperature_gradient,  //!< temperature gradient
    /*---------------------------------------------------------------------------*/
    // particle states for DEM interaction
    AngularVelocity,      //!< angular velocity
    AngularAcceleration,  //!< angular acceleration
    Force,                //!< force
    Moment,               //!< moment
    /*---------------------------------------------------------------------------*/
    // particle states for two way coupled partitioned PASI
    LastIterPosition,              //!< position after last converged iteration
    LastIterVelocity,              //!< velocity after last converged iteration
    LastIterAcceleration,          //!< acceleration after last converged iteration
    LastIterAngularVelocity,       //!< angular velocity after last converged iteration
    LastIterAngularAcceleration,   //!< angular acceleration after last converged iteration
    LastIterModifiedAcceleration,  //!< modified acceleration after last converged iteration
    LastIterDensity,               //!< density after last converged iteration
    LastIterTemperature,           //!< temperature after last converged iteration
    /*---------------------------------------------------------------------------*/
    // particle states for peridynamics body evaluation
    PDBodyId,               //!< peridynamics body id
    CriticalStretch,        //!< critical stretch
    Young,                  //!< young's modulus
    InitialConnectedBonds,  //!< initial active bonds
    CurrentConnectedBonds,  //!< remained active bonds
    PDDamageVariable        //!< damage level at pd node
    /*---------------------------------------------------------------------------*/
  };

  /*!
   * \brief convert particle state enum to dimension
   *
   * \note This method should be used only for initialization.
   *
   *
   * \param[in] state particle state
   *
   * \return particle state dimension
   */
  int enum_to_state_dim(const ParticleState& state);

  /*!
   * \brief convert particle state enum to name
   *
   *
   * \param[in] state particle state
   *
   * \return particle state name
   */
  std::string enum_to_state_name(const ParticleState& state);

  /*!
   * \brief convert particle state name to enum
   *
   * \note Only relevant particle states, i.e., currently needed for initialization are processed.
   *       Extend any additional states if needed!
   *
   * \warning This method is computational expensive due to comparison of strings and should only be
   *          used for initialization.
   *
   *
   * \param[in] name particle state name
   *
   * \return particle state
   */
  ParticleState enum_from_state_name(const std::string& name);

  //! @}

  //! \name definition of particle types
  //! @{

  /*!
   * \brief enums of particle type
   *
   * Enum for respective particle type used to relate and distinguish particles to different phases.
   *
   */
  enum ParticleType
  {
    /*---------------------------------------------------------------------------*/
    // default particle types
    Phase1,  //!< particle type for particles of first phase
    Phase2,  //!< particle type for particles of second phase
    /*---------------------------------------------------------------------------*/
    // particle types for SPH interaction
    BoundaryPhase,   //!< particle type for boundary phase particles
    RigidPhase,      //!< particle type for rigid phase particles
    DirichletPhase,  //!< particle type for dirichlet phase particles (open boundary)
    NeumannPhase,    //!< particle type for neumann phase particles (open boundary)
    PDPhase          //!< particle tpe for peridynamic phase particles
    /*---------------------------------------------------------------------------*/
  };

  /*!
   * \brief convert particle type enum to name
   *
   *
   * \param[in] type particle type
   *
   * \return particle type name
   */
  std::string enum_to_type_name(const ParticleType& type);

  /*!
   * \brief convert particle type name to enum
   *
   * \warning This method is computational expensive due to comparison of strings and should only be
   *          used for initialization.
   *
   *
   * \param[in] name particle type name
   *
   * \return particle type
   */
  ParticleType enum_from_type_name(const std::string& name);

  const std::vector<std::string>& get_particle_type_names();

  //! @}

  //! \name definition of particle status
  //! @{

  /*!
   * \brief enums of particle status
   *
   * Enum for respective particle status used to distinguish owned and ghosted particles on each
   * processor.
   *
   */
  enum ParticleStatus
  {
    Owned,   //!< particle status for particles being owned on processors
    Ghosted  //!< particle status for particles being ghosted on processors
  };

  /*!
   * \brief convert particle status enum to name
   *
   *
   * \param[in] status particle status
   *
   * \return particle status name
   */
  std::string enum_to_status_name(const ParticleStatus& status);

  //! @}

}  // namespace Particle

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
