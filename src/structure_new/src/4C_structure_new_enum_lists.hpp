// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_ENUM_LISTS_HPP
#define FOUR_C_STRUCTURE_NEW_ENUM_LISTS_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

namespace Solid
{
  //! Supported types of energy contributions
  enum EnergyType : int
  {
    internal_energy,                 ///< internal, i.e. strain energy
    kinetic_energy,                  ///< kinetic energy
    beam_contact_penalty_potential,  ///< penalty potential for beam-to-? contact
    beam_interaction_potential,      ///< interaction potential for beam-to-? molecular interactions
    beam_to_beam_link_internal_energy,    ///< internal energy of beam to beam links
    beam_to_beam_link_kinetic_energy,     ///< kinetic energy of beam to beam links
    beam_to_sphere_link_internal_energy,  ///< internal energy of beam to sphere links
    beam_to_sphere_link_kinetic_energy,   ///< kinetic energy of beam to sphere links
    embedded_mesh_penalty_potential       ///< internal energy of penalty potential in embedded-mesh
                                          ///< mortar method
  };

  //! Map energy type to std::string
  inline std::string energy_type_to_string(const enum EnergyType type)
  {
    switch (type)
    {
      case internal_energy:
        return "internal_energy";
      case kinetic_energy:
        return "kinetic_energy";
      case beam_contact_penalty_potential:
        return "beam_contact_penalty_potential";
      case beam_interaction_potential:
        return "beam_interaction_potential";
      case beam_to_beam_link_internal_energy:
        return "beam_to_beam_link_internal_energy";
      case beam_to_beam_link_kinetic_energy:
        return "beam_to_beam_link_kinetic_energy";
      case beam_to_sphere_link_internal_energy:
        return "beam_to_sphere_link_internal_energy";
      case beam_to_sphere_link_kinetic_energy:
        return "beam_to_sphere_link_kinetic_energy";
      case embedded_mesh_penalty_potential:
        return "embedded_mesh_penalty_potential";
      default:
        return "unknown_type_of_energy";
    }
  };

  //! Map std::string to energy type
  inline EnergyType string_to_energy_type(const std::string type)
  {
    if (type == "internal_energy")
      return internal_energy;
    else if (type == "kinetic_energy")
      return kinetic_energy;
    else if (type == "beam_contact_penalty_potential")
      return beam_contact_penalty_potential;
    else if (type == "beam_interaction_potential")
      return beam_interaction_potential;
    else if (type == "beam_to_beam_link_internal_energy")
      return beam_to_beam_link_internal_energy;
    else if (type == "beam_to_beam_link_kinetic_energy")
      return beam_to_beam_link_kinetic_energy;
    else if (type == "beam_to_sphere_link_internal_energy")
      return beam_to_sphere_link_internal_energy;
    else if (type == "beam_to_sphere_link_kinetic_energy")
      return beam_to_sphere_link_kinetic_energy;
    else if (type == "embedded_mesh_penalty_potential")
      return embedded_mesh_penalty_potential;
    else
      FOUR_C_THROW("Unknown type of energy {}", type);
  };


  //! for coupled, monolithic problems: linearization w.r.t. other primary variable
  enum class DifferentiationType : int
  {
    none,
    elch,
    temp
  };

  enum class MatBlockType
  {
    displ_displ,  ///< Kdd block (structural block)
    displ_lm,     ///< Kdz block (of the corresponding model evaluator)
    lm_displ,     ///< Kzd block (of the corresponding model evaluator)
    lm_lm,        ///< Kzz block (of the corresponding model evaluator)
  };

  inline std::string mat_block_type_to_string(const enum MatBlockType type)
  {
    switch (type)
    {
      case MatBlockType::displ_displ:
        return "block_displ_displ";
      case MatBlockType::displ_lm:
        return "block_displ_lm";
      case MatBlockType::lm_displ:
        return "block_lm_displ";
      case MatBlockType::lm_lm:
        return "block_lm_lm";
      default:
        return "unknown matrix block type";
    }
  }
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
