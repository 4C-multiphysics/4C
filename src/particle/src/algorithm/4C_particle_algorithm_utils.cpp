// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_particle_algorithm_utils.hpp"

#include "4C_utils_exceptions.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
template <typename Valtype>
void Particle::ParticleUtils::read_params_types_related_to_values(
    const Teuchos::ParameterList& params, const std::string& name,
    std::map<Particle::Type, Valtype>& typetovalmap)
{
  // read from input file
  std::vector<std::string> typetoval;
  std::string word;
  std::istringstream typetovalstream(Teuchos::getNumericStringParameter(params, name));
  while (typetovalstream >> word) typetoval.push_back(word);

  // default case
  if (typetoval.size() == 1 and typetoval[0] == "none") return;

  // safety check
  if (static_cast<int>(typetoval.size()) % 2)
    FOUR_C_THROW("input of '{}' (size = {}) relating particle type to value is odd!", name,
        typetoval.size());

  std::string typestring;
  std::string valstring;
  for (int i = 0; i < static_cast<int>(typetoval.size() / 2); ++i)
  {
    typestring = typetoval[2 * i];
    valstring = typetoval[2 * i + 1];

    // get enum of particle types
    Particle::Type particleType = Particle::enum_from_type_name(typestring);

    // get numeric value
    Valtype val;
    try
    {
      // standard conversion (double to valtype) via assignment
      val = std::stod(valstring);
    }
    catch (const std::invalid_argument& e)
    {
      FOUR_C_THROW(
          "wrong format of numeric value provided following the name of the particle type!");
    }

    // insert into map
    auto iterator = typetovalmap.insert(std::make_pair(particleType, val));

    // safety check
    if (not iterator.second)
      FOUR_C_THROW("failed inserting numeric value into map since key '{}' is already existing!",
          Particle::enum_to_type_name(particleType));
  }
}

/*---------------------------------------------------------------------------*
 | template instantiations                                                   |
 *---------------------------------------------------------------------------*/
template void Particle::ParticleUtils::read_params_types_related_to_values<int>(
    const Teuchos::ParameterList& params, const std::string& name,
    std::map<Particle::Type, int>& typetovalmap);

template void Particle::ParticleUtils::read_params_types_related_to_values<double>(
    const Teuchos::ParameterList& params, const std::string& name,
    std::map<Particle::Type, double>& typetovalmap);

FOUR_C_NAMESPACE_CLOSE
