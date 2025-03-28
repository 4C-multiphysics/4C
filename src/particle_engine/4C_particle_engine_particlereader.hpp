// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_ENGINE_PARTICLEREADER_HPP
#define FOUR_C_PARTICLE_ENGINE_PARTICLEREADER_HPP

#include "4C_config.hpp"

#include "4C_io_input_spec.hpp"
#include "4C_particle_engine_typedefs.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class InputFile;
}

namespace PARTICLEENGINE
{
  /**
   * Read particles from an input file. The particles are read from the section
   * with name @p section_name.
   */
  void read_particles(Core::IO::InputFile& input, const std::string& section_name,
      std::vector<PARTICLEENGINE::ParticleObjShrdPtr>& particles);

  /**
   * The InputSpec for a single particle line.
   */
  [[nodiscard]] Core::IO::InputSpec create_particle_spec();

}  // namespace PARTICLEENGINE

FOUR_C_NAMESPACE_CLOSE

#endif
