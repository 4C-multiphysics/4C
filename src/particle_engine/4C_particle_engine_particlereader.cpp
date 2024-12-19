// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_engine_particlereader.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_io_input_file.hpp"
#include "4C_io_pstream.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_object.hpp"
#include "4C_particle_engine_typedefs.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

void PARTICLEENGINE::read_particles(Core::IO::InputFile& input, const std::string& section_name,
    std::vector<PARTICLEENGINE::ParticleObjShrdPtr>& particles)
{
  const int myrank = Core::Communication::my_mpi_rank(input.get_comm());
  if (myrank > 0) return;

  Teuchos::Time time("", true);

  bool any_particles_read = false;
  for (const auto& particle_line : input.lines_in_section(section_name))
  {
    if (!any_particles_read) Core::IO::cout << "Read and create particles\n" << Core::IO::flush;
    any_particles_read = true;

    double t1 = time.totalElapsedTime(true);
    {
      PARTICLEENGINE::TypeEnum particletype;
      PARTICLEENGINE::ParticleStates particlestates;

      Core::IO::ValueParser parser{
          particle_line, {.user_scope_message = "While reading particle data: "}};
      parser.consume("TYPE");
      auto type = parser.read<std::string>();
      parser.consume("POS");
      auto pos = parser.read<std::array<double, 3>>();

      // get enum of particle type
      particletype = PARTICLEENGINE::enum_from_type_name(type);

      // allocate memory to hold particle position state
      particlestates.resize(PARTICLEENGINE::Position + 1);

      // set position state
      particlestates[PARTICLEENGINE::Position] = std::vector<double>(pos.begin(), pos.end());

      // optional particle states
      {
        std::string statelabel;
        PARTICLEENGINE::StateEnum particlestate;
        std::vector<double> state;

        std::istringstream linestream(std::string(parser.get_unparsed_remainder()));

        while (linestream >> statelabel)
        {
          // optional particle radius
          if (statelabel == "RAD")
          {
            particlestate = PARTICLEENGINE::Radius;
            state.resize(1);
            linestream >> state[0];
          }
          // optional rigid body color
          else if (statelabel == "RIGIDCOLOR")
          {
            particlestate = PARTICLEENGINE::RigidBodyColor;
            state.resize(1);
            linestream >> state[0];
          }
          else
            FOUR_C_THROW("optional particle state with label '%s' unknown!", statelabel.c_str());

          if (not linestream)
            FOUR_C_THROW("expecting values of state '%s' if label '%s' is set!",
                PARTICLEENGINE::enum_to_state_name(particlestate).c_str(), statelabel.c_str());

          // allocate memory to hold optional particle state
          if (static_cast<int>(particlestates.size()) < (particlestate + 1))
            particlestates.resize(particlestate + 1);

          // set optional particle state
          particlestates[particlestate] = state;
        }
      }

      // construct and store read in particle object
      particles.emplace_back(
          std::make_shared<PARTICLEENGINE::ParticleObject>(particletype, -1, particlestates));
    }

    double t2 = time.totalElapsedTime(true);
    if (!myrank)
    {
      printf("reading %10.5e secs\n", t2 - t1);
      fflush(stdout);
    }
  }

  if (any_particles_read)
    printf("in............................................. %10.5e secs\n",
        time.totalElapsedTime(true));
}


FOUR_C_NAMESPACE_CLOSE
