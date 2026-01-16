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
    case Radius:
    case Mass:
    case Density:
    case Pressure:
    case Temperature:
    case RigidBodyColor:
    case Inertia:
    case DensitySum:
    case DensityDot:
    case TemperatureDot:
    case BoundaryPressure:
    case Colorfield:
    case Curvature:
    case WallColorfield:
    case LastIterDensity:
    case LastIterTemperature:
    case Young:
    case CriticalStretch:
    case PDBodyId:
    case InitialConnectedBonds:
    case CurrentConnectedBonds:
    case PDDamageVariable:
      dim = 1;
      break;

    // vectorial states
    case Position:
    case Velocity:
    case Acceleration:
    case LastTransferPosition:
    case ModifiedVelocity:
    case ModifiedAcceleration:
    case ReferencePosition:
    case RelativePosition:
    case RelativePositionBodyFrame:
    case BoundaryVelocity:
    case ColorfieldGradient:
    case InterfaceNormal:
    case WallInterfaceNormal:
    case temperature_gradient:
    case AngularVelocity:
    case AngularAcceleration:
    case Force:
    case Moment:
    case LastIterPosition:
    case LastIterVelocity:
    case LastIterAcceleration:
    case LastIterAngularVelocity:
    case LastIterAngularAcceleration:
    case LastIterModifiedAcceleration:
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
    case Radius:
      name = "radius";
      break;
    case Mass:
      name = "mass";
      break;
    case Density:
      name = "density";
      break;
    case DensitySum:
      name = "density sum";
      break;
    case DensityDot:
      name = "density dot";
      break;
    case Pressure:
      name = "pressure";
      break;
    case Temperature:
      name = "temperature";
      break;
    case RigidBodyColor:
      name = "rigid body color";
      break;
    case Inertia:
      name = "inertia";
      break;
    case TemperatureDot:
      name = "temperature dot";
      break;
    case Position:
      name = "position";
      break;
    case Velocity:
      name = "velocity";
      break;
    case Acceleration:
      name = "acceleration";
      break;
    case AngularVelocity:
      name = "angular velocity";
      break;
    case AngularAcceleration:
      name = "angular acceleration";
      break;
    case Force:
      name = "force";
      break;
    case Moment:
      name = "moment";
      break;
    case LastTransferPosition:
      name = "position last transfer";
      break;
    case ReferencePosition:
      name = "reference position";
      break;
    case RelativePosition:
      name = "relative position";
      break;
    case RelativePositionBodyFrame:
      name = "body frame relative position";
      break;
    case ModifiedVelocity:
      name = "modified velocity";
      break;
    case ModifiedAcceleration:
      name = "modified acceleration";
      break;
    case BoundaryPressure:
      name = "boundary pressure";
      break;
    case BoundaryVelocity:
      name = "boundary velocity";
      break;
    case Colorfield:
      name = "colorfield";
      break;
    case ColorfieldGradient:
      name = "colorfield gradient";
      break;
    case InterfaceNormal:
      name = "interface normal";
      break;
    case Curvature:
      name = "curvature";
      break;
    case WallColorfield:
      name = "wall colorfield";
      break;
    case WallInterfaceNormal:
      name = "wall interface normal";
      break;
    case temperature_gradient:
      name = "temperature gradient";
      break;
    case LastIterPosition:
      name = "position last iteration";
      break;
    case LastIterVelocity:
      name = "velocity last iteration";
      break;
    case LastIterAcceleration:
      name = "acceleration last iteration";
      break;
    case LastIterAngularVelocity:
      name = "angular velocity last iteration";
      break;
    case LastIterAngularAcceleration:
      name = "angular acceleration last iteration";
      break;
    case LastIterModifiedAcceleration:
      name = "modified acceleration last iteration";
      break;
    case LastIterDensity:
      name = "density last iteration";
      break;
    case LastIterTemperature:
      name = "temperature last iteration";
      break;
    case PDBodyId:
      name = "peridynamics body id";
      break;
    case CriticalStretch:
      name = "critical stretch";
      break;
    case Young:
      name = "Youngs modulus";
      break;
    case InitialConnectedBonds:
      name = "initial active bonds";
      break;
    case CurrentConnectedBonds:
      name = "remained active bonds";
      break;
    case PDDamageVariable:
      name = "pd_damage_phi";
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
    state = Density;
  else if (name == "pressure")
    state = Pressure;
  else if (name == "temperature")
    state = Temperature;
  else
    FOUR_C_THROW("particle state '{}' unknown!", name);

  return state;
}

static std::vector<std::string> particle_type_names = {
    "phase1", "phase2", "boundaryphase", "rigidphase", "dirichletphase", "neumannphase", "pdphase"};

std::string Particle::enum_to_type_name(const ParticleType& type)
{
  FOUR_C_ASSERT(type >= 0 and type < static_cast<int>(particle_type_names.size()),
      "particle type out of range!");
  return particle_type_names[type];
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
    case Owned:
      name = "owned";
      break;
    case Ghosted:
      name = "ghosted";
      break;
    default:
      FOUR_C_THROW("particle status unknown!");
  }

  return name;
}

FOUR_C_NAMESPACE_CLOSE
