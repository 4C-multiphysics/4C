// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fbi_beam_to_fluid_meshtying_pair_factory.hpp"

#include "4C_fbi_beam_to_fluid_meshtying_pair_gauss_point.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_pair_mortar.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_params.hpp"
#include "4C_fbi_input.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
std::shared_ptr<BeamInteraction::BeamContactPair> FBI::PairFactory::create_pair(
    std::vector<Core::Elements::Element const*> const& ele_ptrs,
    FBI::BeamToFluidMeshtyingParams& params_ptr)
{
  // Cast the fluid element.
  Discret::Elements::Fluid const* fluidele =
      dynamic_cast<Discret::Elements::Fluid const*>(ele_ptrs[1]);
  Core::FE::CellType shape = fluidele->shape();

  // Get the meshtying discretization method.
  FBI::BeamToFluidDiscretization meshtying_discretization = params_ptr.get_contact_discretization();

  // Check which contact discretization is wanted.
  if (meshtying_discretization == FBI::BeamToFluidDiscretization::gauss_point_to_segment)
  {
    switch (shape)
    {
      case Core::FE::CellType::hex8:
        return std::shared_ptr<BeamInteraction::BeamContactPair>(
            new BeamInteraction::BeamToFluidMeshtyingPairGaussPoint<GEOMETRYPAIR::t_hermite,
                GEOMETRYPAIR::t_hex8>());
      case Core::FE::CellType::hex20:
        return std::shared_ptr<BeamInteraction::BeamContactPair>(
            new BeamInteraction::BeamToFluidMeshtyingPairGaussPoint<GEOMETRYPAIR::t_hermite,
                GEOMETRYPAIR::t_hex20>());
      case Core::FE::CellType::hex27:
        return std::shared_ptr<BeamInteraction::BeamContactPair>(
            new BeamInteraction::BeamToFluidMeshtyingPairGaussPoint<GEOMETRYPAIR::t_hermite,
                GEOMETRYPAIR::t_hex27>());
      case Core::FE::CellType::tet4:
        return std::shared_ptr<BeamInteraction::BeamContactPair>(
            new BeamInteraction::BeamToFluidMeshtyingPairGaussPoint<GEOMETRYPAIR::t_hermite,
                GEOMETRYPAIR::t_tet4>());
      case Core::FE::CellType::tet10:
        return std::shared_ptr<BeamInteraction::BeamContactPair>(
            new BeamInteraction::BeamToFluidMeshtyingPairGaussPoint<GEOMETRYPAIR::t_hermite,
                GEOMETRYPAIR::t_tet10>());
      default:
        FOUR_C_THROW("Wrong element type for fluid element.");
    }
  }
  else if (meshtying_discretization == FBI::BeamToFluidDiscretization::mortar)
  {
    FBI::BeamToFluidMeshtingMortarShapefunctions mortar_shape_function =
        params_ptr.get_mortar_shape_function_type();

    switch (mortar_shape_function)
    {
      case FBI::BeamToFluidMeshtingMortarShapefunctions::line2:
      {
        switch (shape)
        {
          case Core::FE::CellType::hex8:
            return std::make_shared<BeamInteraction::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex8, GEOMETRYPAIR::t_line2>>();
          case Core::FE::CellType::hex20:
            return std::make_shared<BeamInteraction::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex20, GEOMETRYPAIR::t_line2>>();
          case Core::FE::CellType::hex27:
            return std::make_shared<BeamInteraction::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex27, GEOMETRYPAIR::t_line2>>();
          case Core::FE::CellType::tet4:
            return std::make_shared<BeamInteraction::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_tet4, GEOMETRYPAIR::t_line2>>();
          case Core::FE::CellType::tet10:
            return std::make_shared<BeamInteraction::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_tet10, GEOMETRYPAIR::t_line2>>();
          default:
            FOUR_C_THROW("Wrong element type for solid element.");
        }
        break;
      }
      case FBI::BeamToFluidMeshtingMortarShapefunctions::line3:
      {
        switch (shape)
        {
          case Core::FE::CellType::hex8:
            return std::make_shared<BeamInteraction::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex8, GEOMETRYPAIR::t_line3>>();
          case Core::FE::CellType::hex20:
            return std::make_shared<BeamInteraction::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex20, GEOMETRYPAIR::t_line3>>();
          case Core::FE::CellType::hex27:
            return std::make_shared<BeamInteraction::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex27, GEOMETRYPAIR::t_line3>>();
          case Core::FE::CellType::tet4:
            return std::make_shared<BeamInteraction::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_tet4, GEOMETRYPAIR::t_line3>>();
          case Core::FE::CellType::tet10:
            return std::make_shared<BeamInteraction::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_tet10, GEOMETRYPAIR::t_line3>>();
          default:
            FOUR_C_THROW("Wrong element type for solid element.");
        }
        break;
      }
      case FBI::BeamToFluidMeshtingMortarShapefunctions::line4:
      {
        switch (shape)
        {
          case Core::FE::CellType::hex8:
            return std::make_shared<BeamInteraction::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex8, GEOMETRYPAIR::t_line4>>();
          case Core::FE::CellType::hex20:
            return std::make_shared<BeamInteraction::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex20, GEOMETRYPAIR::t_line4>>();
          case Core::FE::CellType::hex27:
            return std::make_shared<BeamInteraction::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex27, GEOMETRYPAIR::t_line4>>();
          case Core::FE::CellType::tet4:
            return std::make_shared<BeamInteraction::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_tet4, GEOMETRYPAIR::t_line4>>();
          case Core::FE::CellType::tet10:
            return std::make_shared<BeamInteraction::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_tet10, GEOMETRYPAIR::t_line4>>();
          default:
            FOUR_C_THROW("Wrong element type for solid element.");
        }
        break;
      }
      default:
        FOUR_C_THROW("Wrong mortar shape function.");
    }
  }
  else
    FOUR_C_THROW("discretization type not yet implemented!\n");

  // Default return value.
  return nullptr;
}

FOUR_C_NAMESPACE_CLOSE
