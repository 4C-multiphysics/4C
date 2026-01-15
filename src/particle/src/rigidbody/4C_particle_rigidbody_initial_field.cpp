// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_rigidbody_initial_field.hpp"

#include "4C_global_data.hpp"
#include "4C_particle_algorithm_utils.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_rigidbody_datastate.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  std::map<Particle::StateEnum, std::map<Particle::TypeEnum, int>>
  extract_particle_types_to_function_ids(const Teuchos::ParameterList& params);

  std::vector<std::vector<double>>& get_rigid_body_state(
      Particle::StateEnum particleState, Particle::RigidBodyDataState& rigidbodydatastates);
}  // namespace

void Particle::set_initial_fields(const Teuchos::ParameterList& params,
    const std::vector<int>& ownedrigidbodies, Particle::RigidBodyDataState& rigidbodydatastates)
{
  // relating particle types to function ids
  std::map<Particle::StateEnum, std::map<Particle::TypeEnum, int>> statetotypetofunctidmap =
      extract_particle_types_to_function_ids(params);

  for (auto& stateIt : statetotypetofunctidmap)
  {
    if (not stateIt.second.contains(Particle::RigidPhase)) continue;

    // state vector
    Particle::StateEnum particleState = stateIt.first;

    // get pointer to rigid body state
    std::vector<std::vector<double>>& state =
        get_rigid_body_state(particleState, rigidbodydatastates);

    // get id of function
    const int functid = stateIt.second[Particle::RigidPhase];

    // get reference to function
    const auto& function =
        Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfSpaceTime>(functid);

    // get rigid body state dimension
    const int statedim = 3;

    // safety check
    if (static_cast<std::size_t>(statedim) != function.number_components())
    {
      FOUR_C_THROW(
          "dimensions of function defining initial field and of state of rigid bodies '{}' not "
          "matching!",
          Particle::enum_to_state_name(particleState).c_str());
    }

    // iterate over owned rigid bodies
    for (const int rigidbody_k : ownedrigidbodies)
    {
      // get pointer to rigid body states
      const double* pos_k = rigidbodydatastates.get_ref_position()[rigidbody_k].data();

      // evaluate function to set initial field
      for (int dim = 0; dim < statedim; ++dim)
      {
        state[rigidbody_k][dim] = function.evaluate(std::span(pos_k, 3), 0.0, dim);
      }
    }
  }
}

namespace
{
  std::map<Particle::StateEnum, std::map<Particle::TypeEnum, int>>
  extract_particle_types_to_function_ids(const Teuchos::ParameterList& params)
  {
    std::map<Particle::StateEnum, std::map<Particle::TypeEnum, int>> statetotypetofunctidmap;

    // get control parameters for initial/boundary conditions
    const Teuchos::ParameterList& params_conditions =
        params.sublist("INITIAL AND BOUNDARY CONDITIONS");

    // relate particle state to input name
    std::map<std::string, Particle::StateEnum> initialfieldtostateenum = {
        std::make_pair("INITIAL_VELOCITY_FIELD", Particle::Velocity),
        std::make_pair("INITIAL_ANGULAR_VELOCITY_FIELD", Particle::AngularVelocity),
        std::make_pair("INITIAL_ACCELERATION_FIELD", Particle::Acceleration),
        std::make_pair("INITIAL_ANGULAR_ACCELERATION_FIELD", Particle::AngularAcceleration)};

    // iterate over particle states
    for (const auto& stateIt : initialfieldtostateenum)
    {
      // get reference to sub-map
      std::map<Particle::TypeEnum, int>& currentstatetypetofunctidmap =
          statetotypetofunctidmap[stateIt.second];

      // read parameters relating particle types to values
      Particle::ParticleUtils::read_params_types_related_to_values(
          params_conditions, stateIt.first, currentstatetypetofunctidmap);
    }

#ifdef FOUR_C_ENABLE_ASSERTIONS
    // safety check
    for (const auto& iter : statetotypetofunctidmap)
    {
      if (iter.first == Particle::Temperature and not iter.second.empty())
        FOUR_C_THROW("initial temperature cannot be specified for rigid bodies '{}' !",
            Particle::enum_to_state_name(iter.first));
    }
#endif

    return statetotypetofunctidmap;
  }

  std::vector<std::vector<double>>& get_rigid_body_state(
      Particle::StateEnum particleState, Particle::RigidBodyDataState& rigidbodydatastates)
  {
    switch (particleState)
    {
      case Particle::Velocity:
        return rigidbodydatastates.get_ref_velocity();
      case Particle::AngularVelocity:
        return rigidbodydatastates.get_ref_angular_velocity();
      case Particle::Acceleration:
        return rigidbodydatastates.get_ref_acceleration();
      case Particle::AngularAcceleration:
        return rigidbodydatastates.get_ref_angular_acceleration();

      default:
        FOUR_C_THROW("unsupported state vector '{}' for initialization of rigid body!",
            Particle::enum_to_state_name(particleState));
    }
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE
