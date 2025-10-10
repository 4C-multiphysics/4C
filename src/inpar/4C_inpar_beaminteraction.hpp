// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_INPAR_BEAMINTERACTION_HPP
#define FOUR_C_INPAR_BEAMINTERACTION_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_integration.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_utils_exceptions.hpp"

#include <vector>


FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::Conditions
{
  class ConditionDefinition;
}
namespace Inpar
{
  namespace BeamInteraction
  {
    enum RepartitionStrategy
    {
      repstr_adaptive,  ///< only do repartitioning in case physically necessary
      repstr_everydt    ///< do repartitioning every time step
    };

    enum class SearchStrategy
    {
      bruteforce_with_binning,   ///< coupling pair search based on binning
      bounding_volume_hierarchy  ///< coupling pair search based on a bounding volume hierarchy
    };

    /// type of the used submodel for beaminteraction
    enum SubModelType
    {
      submodel_crosslinking,    ///< evaluate the structural model
      submodel_beamcontact,     ///< evaluate the contact model
      submodel_potential,       ///< evaluate the model for potential-based interactions
      submodel_spherebeamlink,  ///< evaluate model for cell filament interactions
      submodel_vague            ///< undefined model type
    };

    /// type of employed solving strategy for contact
    /// (this enum represents the input file parameter STRATEGY)
    enum Strategy
    {
      bstr_none,    ///< no beam contact
      bstr_penalty  ///< penalty method
    };

    /// type of linker
    enum JointType
    {
      beam3r_line2_rigid,  ///< rigid joint
      beam3r_line2_pin,    ///< pin joint
      truss                ///< truss
    };

    /// type of filament
    enum FilamentType
    {
      filetype_arbitrary,  ///< no special type
      filetype_actin,      ///< actin type
      filetype_collagen,   ///< collagen type
      filetype_none        ///< filament which does not bind
    };

    /// type of crosslinker
    enum CrosslinkerType
    {
      linkertype_arbitrary,  ///< binds to all filament
      linkertype_actin,      ///< only binds to actin filaments
      linkertype_collagen,   ///< only binds to collagen filaments
      linkertype_integrin    ///< sphere to beam linker
    };

    /**
     * \brief Types of beam interaction conditions
     */
    enum class BeamInteractionConditions
    {
      //! Default value.
      none,
      //! Beam-to-beam contact.
      beam_to_beam_contact,
      //! Beam-to-solid volume mesh tying.
      beam_to_solid_volume_meshtying,
      //! Beam-to-solid surface mesh tying.
      beam_to_solid_surface_meshtying,
      //! beam-to-beam penalty point coupling.
      beam_to_beam_point_coupling,
      //! Beam-to-solid surface contact.
      beam_to_solid_surface_contact,
      //! Beam-to-solid edge contact.
      beam_to_solid_edge_contact
    };

    //! Map type std::string to enum
    inline enum JointType string_to_joint_type(const std::string& name)
    {
      JointType type = beam3r_line2_rigid;
      if (name == "beam3rline2rigid")
        type = beam3r_line2_rigid;
      else if (name == "beam3rline2pin")
        type = beam3r_line2_pin;
      else if (name == "truss")
        type = truss;
      else
        FOUR_C_THROW("invalid filament type std::string ");

      return type;
    };

    //! Map type std::string to enum
    inline enum FilamentType string_to_filament_type(const std::string& name)
    {
      FilamentType type = filetype_arbitrary;
      if (name == "arbitrary")
        type = filetype_arbitrary;
      else if (name == "actin")
        type = filetype_actin;
      else if (name == "collagen")
        type = filetype_collagen;
      else if (name == "none")
        type = filetype_none;
      else
        FOUR_C_THROW("invalid filament type std::string ");

      return type;
    };

    //! Map type std::string to enum
    inline enum CrosslinkerType string_to_crosslinker_type(const std::string& name)
    {
      CrosslinkerType type = linkertype_arbitrary;
      if (name == "arbitrary")
        type = linkertype_arbitrary;
      else if (name == "actin")
        type = linkertype_actin;
      else if (name == "collagen")
        type = linkertype_collagen;
      else if (name == "integrin")
        type = linkertype_integrin;
      else
      {
        FOUR_C_THROW(
            "invalid crosslinker type {}. Possible values are arbitrary, actin, collagen and "
            "integrin.",
            name.c_str());
      }

      return type;
    };

    //! Map action type enum to std::string
    static inline std::string crosslinker_type_to_string(const enum CrosslinkerType type)
    {
      switch (type)
      {
        case linkertype_arbitrary:
          return "arbitrary";
        case linkertype_actin:
          return "actin";
        case linkertype_collagen:
          return "collagen";
        case linkertype_integrin:
          return "integrin";
        default:
          return "unknown";
      }
      return "";
    };

    /**
     * \brief Get all available beam interaction conditions, excluding the default value.
     */
    void beam_interaction_conditions_get_all(
        std::vector<Inpar::BeamInteraction::BeamInteractionConditions>& interactions);

    /// beam interaction parameters
    std::vector<Core::IO::InputSpec> valid_parameters();

    /// set beam interaction specific conditions
    void set_valid_conditions(std::vector<Core::Conditions::ConditionDefinition>& condlist);

  }  // namespace BeamInteraction

}  // namespace Inpar

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
