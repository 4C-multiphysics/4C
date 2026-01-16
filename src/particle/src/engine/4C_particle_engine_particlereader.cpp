// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_engine_particlereader.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_io_input_file.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_pstream.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_object.hpp"
#include "4C_particle_engine_typedefs.hpp"

#include <sys/stat.h>
#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

void Particle::read_particles(Core::IO::InputFile& input, const std::string& section_name,
    std::vector<Particle::ParticleObjShrdPtr>& particles)
{
  const int myrank = Core::Communication::my_mpi_rank(input.get_comm());
  if (myrank > 0) return;

  Teuchos::Time time("", true);

  bool any_particles_read = false;
  for (const auto& particle_line : input.in_section_rank_0_only(section_name))
  {
    if (!any_particles_read) Core::IO::cout << "Read and create particles\n" << Core::IO::flush;
    any_particles_read = true;

    {
      Particle::TypeEnum particletype;
      Particle::ParticleStates particlestates;

      Core::IO::ValueParser parser{particle_line.get_as_dat_style_string(),
          {.user_scope_message = "While reading particle data: "}};
      parser.consume("TYPE");
      auto type = parser.read<std::string>();
      parser.consume("POS");
      auto pos = parser.read<std::array<double, 3>>();

      // get enum of particle type
      particletype = Particle::enum_from_type_name(type);

      // allocate memory to hold particle position state
      particlestates.resize(Particle::Position + 1);

      // set position state
      particlestates[Particle::Position] = std::vector<double>(pos.begin(), pos.end());

      // optional particle states
      {
        std::string statelabel;
        Particle::StateEnum particlestate;
        std::vector<double> state;

        while (!parser.at_end())
        {
          auto next = parser.peek();
          // optional particle radius
          if (next == "RAD")
          {
            particlestate = Particle::Radius;
            parser.consume("RAD");

            if (auto val = parser.read<std::optional<double>>())
            {
              state.resize(1);
              state[0] = *val;
            }
            else
            {
              continue;
            }
          }
          // optional rigid body color
          else if (next == "RIGIDCOLOR")
          {
            particlestate = Particle::RigidBodyColor;
            parser.consume("RIGIDCOLOR");

            if (auto val = parser.read<std::optional<double>>())
            {
              state.resize(1);
              state[0] = *val;
            }
            else
            {
              continue;
            }
          }
          // optional pd body id
          else if (next == "PDBODYID")
          {
            particlestate = Particle::PDBodyId;
            parser.consume("PDBODYID");

            if (auto val = parser.read<std::optional<double>>())
            {
              state.resize(1);
              state[0] = *val;
            }
            else
            {
              continue;
            }
          }
          else
            FOUR_C_THROW("Optional particle state with label '{}' unknown!", statelabel);

          // allocate memory to hold optional particle state
          if (static_cast<int>(particlestates.size()) < (particlestate + 1))
            particlestates.resize(particlestate + 1);

          // set optional particle state
          particlestates[particlestate] = state;
        }
      }

      // construct and store read in particle object
      particles.emplace_back(
          std::make_shared<Particle::ParticleObject>(particletype, -1, particlestates));
    }
  }

  if (any_particles_read)
    printf("in............................................. %10.5e secs\n",
        time.totalElapsedTime(true));
}


Core::IO::InputSpec Particle::create_particle_spec()
{
  using namespace Core::IO::InputSpecBuilders;

  return all_of({deprecated_selection<std::string>("TYPE", get_particle_type_names()),
      parameter<std::vector<double>>("POS", {.size = 3}), parameter<std::optional<double>>("RAD"),
      parameter<std::optional<double>>("RIGIDCOLOR"),
      parameter<std::optional<double>>("PDBODYID")});
}


FOUR_C_NAMESPACE_CLOSE
