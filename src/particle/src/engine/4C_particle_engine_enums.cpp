// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_engine_enums.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
int Particle::enum_to_state_dim(const ParticleState& state)
{
  int dim = 0;

  switch (state)
  {
    // scalar states
    case ParticleState::Radius:
    case ParticleState::Mass:
    case ParticleState::Density:
    case ParticleState::Pressure:
    case ParticleState::Temperature:
    case ParticleState::RigidBodyColor:
    case ParticleState::Inertia:
    case ParticleState::DensitySum:
    case ParticleState::DensityDot:
    case ParticleState::TemperatureDot:
    case ParticleState::BoundaryPressure:
    case ParticleState::Colorfield:
    case ParticleState::Curvature:
    case ParticleState::WallColorfield:
    case ParticleState::LastIterDensity:
    case ParticleState::LastIterTemperature:
    case ParticleState::Young:
    case ParticleState::CriticalStretch:
    case ParticleState::PDBodyId:
    case ParticleState::InitialConnectedBonds:
    case ParticleState::CurrentConnectedBonds:
    case ParticleState::PDDamageVariable:
    case ParticleState::DirichletFunctionId:
    case ParticleState::OpenBoundaryId:
      dim = 1;
      break;

    // vectorial states
    case ParticleState::Position:
    case ParticleState::Velocity:
    case ParticleState::Acceleration:
    case ParticleState::LastTransferPosition:
    case ParticleState::ModifiedVelocity:
    case ParticleState::ModifiedAcceleration:
    case ParticleState::ReferencePosition:
    case ParticleState::RelativePosition:
    case ParticleState::RelativePositionBodyFrame:
    case ParticleState::BoundaryVelocity:
    case ParticleState::ColorfieldGradient:
    case ParticleState::InterfaceNormal:
    case ParticleState::WallInterfaceNormal:
    case ParticleState::TemperatureGradient:
    case ParticleState::AngularVelocity:
    case ParticleState::AngularAcceleration:
    case ParticleState::Force:
    case ParticleState::Moment:
    case ParticleState::LastIterPosition:
    case ParticleState::LastIterVelocity:
    case ParticleState::LastIterAcceleration:
    case ParticleState::LastIterAngularVelocity:
    case ParticleState::LastIterAngularAcceleration:
    case ParticleState::LastIterModifiedAcceleration:
      dim = 3;
      break;

    default:
      FOUR_C_THROW("particle state unknown!");
  }

  return dim;
}

std::string Particle::enum_to_state_name(const ParticleState& state)
{
  std::string name;

  switch (state)
  {
    case ParticleState::Radius:
      name = "radius";
      break;
    case ParticleState::Mass:
      name = "mass";
      break;
    case ParticleState::Density:
      name = "density";
      break;
    case ParticleState::DensitySum:
      name = "density sum";
      break;
    case ParticleState::DensityDot:
      name = "density dot";
      break;
    case ParticleState::Pressure:
      name = "pressure";
      break;
    case ParticleState::Temperature:
      name = "temperature";
      break;
    case ParticleState::RigidBodyColor:
      name = "rigid body color";
      break;
    case ParticleState::Inertia:
      name = "inertia";
      break;
    case ParticleState::TemperatureDot:
      name = "temperature dot";
      break;
    case ParticleState::Position:
      name = "position";
      break;
    case ParticleState::Velocity:
      name = "velocity";
      break;
    case ParticleState::Acceleration:
      name = "acceleration";
      break;
    case ParticleState::AngularVelocity:
      name = "angular velocity";
      break;
    case ParticleState::AngularAcceleration:
      name = "angular acceleration";
      break;
    case ParticleState::Force:
      name = "force";
      break;
    case ParticleState::Moment:
      name = "moment";
      break;
    case ParticleState::LastTransferPosition:
      name = "position last transfer";
      break;
    case ParticleState::ReferencePosition:
      name = "reference position";
      break;
    case ParticleState::RelativePosition:
      name = "relative position";
      break;
    case ParticleState::RelativePositionBodyFrame:
      name = "body frame relative position";
      break;
    case ParticleState::ModifiedVelocity:
      name = "modified velocity";
      break;
    case ParticleState::ModifiedAcceleration:
      name = "modified acceleration";
      break;
    case ParticleState::BoundaryPressure:
      name = "boundary pressure";
      break;
    case ParticleState::BoundaryVelocity:
      name = "boundary velocity";
      break;
    case ParticleState::Colorfield:
      name = "colorfield";
      break;
    case ParticleState::ColorfieldGradient:
      name = "colorfield gradient";
      break;
    case ParticleState::InterfaceNormal:
      name = "interface normal";
      break;
    case ParticleState::Curvature:
      name = "curvature";
      break;
    case ParticleState::WallColorfield:
      name = "wall colorfield";
      break;
    case ParticleState::WallInterfaceNormal:
      name = "wall interface normal";
      break;
    case ParticleState::TemperatureGradient:
      name = "temperature gradient";
      break;
    case ParticleState::LastIterPosition:
      name = "position last iteration";
      break;
    case ParticleState::LastIterVelocity:
      name = "velocity last iteration";
      break;
    case ParticleState::LastIterAcceleration:
      name = "acceleration last iteration";
      break;
    case ParticleState::LastIterAngularVelocity:
      name = "angular velocity last iteration";
      break;
    case ParticleState::LastIterAngularAcceleration:
      name = "angular acceleration last iteration";
      break;
    case ParticleState::LastIterModifiedAcceleration:
      name = "modified acceleration last iteration";
      break;
    case ParticleState::LastIterDensity:
      name = "density last iteration";
      break;
    case ParticleState::LastIterTemperature:
      name = "temperature last iteration";
      break;
    case ParticleState::PDBodyId:
      name = "peridynamics body id";
      break;
    case ParticleState::CriticalStretch:
      name = "critical stretch";
      break;
    case ParticleState::Young:
      name = "Youngs modulus";
      break;
    case ParticleState::InitialConnectedBonds:
      name = "initial active bonds";
      break;
    case ParticleState::CurrentConnectedBonds:
      name = "remained active bonds";
      break;
    case ParticleState::PDDamageVariable:
      name = "pd_damage_phi";
      break;
    case ParticleState::DirichletFunctionId:
      name = "dirichlet_function_id";
      break;
    case ParticleState::OpenBoundaryId:
      name = "open_boundary_id";
      break;
    default:
      FOUR_C_THROW("particle state unknown!");
  }

  return name;
}

Particle::ParticleState Particle::enum_from_state_name(const std::string& name)
{
  ParticleState state;

  if (name == "density")
    state = ParticleState::Density;
  else if (name == "pressure")
    state = ParticleState::Pressure;
  else if (name == "temperature")
    state = ParticleState::Temperature;
  else
    FOUR_C_THROW("particle state '{}' unknown!", name);

  return state;
}

static std::vector<std::string> particle_type_names = {
    "phase1", "phase2", "boundaryphase", "rigidphase", "dirichletphase", "neumannphase", "pdphase"};

std::string Particle::enum_to_type_name(const ParticleType& type)
{
  const int type_idx = static_cast<int>(type);

  FOUR_C_ASSERT(type_idx >= 0 and type_idx < static_cast<int>(particle_type_names.size()),
      "particle type out of range!");
  return particle_type_names[type_idx];
}

Particle::ParticleType Particle::enum_from_type_name(const std::string& name)
{
  auto it = std::ranges::find(particle_type_names, name);
  FOUR_C_ASSERT(it != particle_type_names.end(), "particle type '{}' unknown!", name);
  return static_cast<ParticleType>(std::distance(particle_type_names.begin(), it));
}

const std::vector<std::string>& Particle::get_particle_type_names() { return particle_type_names; }

std::string Particle::enum_to_status_name(const ParticleStatus& status)
{
  std::string name;

  switch (status)
  {
    case ParticleStatus::Owned:
      name = "owned";
      break;
    case ParticleStatus::Ghosted:
      name = "ghosted";
      break;
    default:
      FOUR_C_THROW("particle status unknown!");
  }

  return name;
}

std::string Particle::enum_to_space_name(const ParticleSpace& space)
{
  std::string name;

  switch (space)
  {
    case ParticleSpace::Host:
      name = "host";
      break;
    case ParticleSpace::Device:
      name = "device";
      break;
    default:
      FOUR_C_THROW("particle memory space unknown!");
  }

  return name;
}

FOUR_C_NAMESPACE_CLOSE
