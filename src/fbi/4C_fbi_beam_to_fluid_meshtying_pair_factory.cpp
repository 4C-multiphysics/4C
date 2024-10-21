#include "4C_fbi_beam_to_fluid_meshtying_pair_factory.hpp"

#include "4C_fbi_beam_to_fluid_meshtying_pair_gauss_point.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_pair_mortar.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_params.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_inpar_fbi.hpp"

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
Teuchos::RCP<BEAMINTERACTION::BeamContactPair> FBI::PairFactory::create_pair(
    std::vector<Core::Elements::Element const*> const& ele_ptrs,
    FBI::BeamToFluidMeshtyingParams& params_ptr)
{
  // Cast the fluid element.
  Discret::ELEMENTS::Fluid const* fluidele =
      dynamic_cast<Discret::ELEMENTS::Fluid const*>(ele_ptrs[1]);
  Core::FE::CellType shape = fluidele->shape();

  // Get the meshtying discretization method.
  Inpar::FBI::BeamToFluidDiscretization meshtying_discretization =
      params_ptr.get_contact_discretization();

  // Check which contact discretization is wanted.
  if (meshtying_discretization == Inpar::FBI::BeamToFluidDiscretization::gauss_point_to_segment)
  {
    switch (shape)
    {
      case Core::FE::CellType::hex8:
        return Teuchos::RCP(
            new BEAMINTERACTION::BeamToFluidMeshtyingPairGaussPoint<GEOMETRYPAIR::t_hermite,
                GEOMETRYPAIR::t_hex8>());
      case Core::FE::CellType::hex20:
        return Teuchos::RCP(
            new BEAMINTERACTION::BeamToFluidMeshtyingPairGaussPoint<GEOMETRYPAIR::t_hermite,
                GEOMETRYPAIR::t_hex20>());
      case Core::FE::CellType::hex27:
        return Teuchos::RCP(
            new BEAMINTERACTION::BeamToFluidMeshtyingPairGaussPoint<GEOMETRYPAIR::t_hermite,
                GEOMETRYPAIR::t_hex27>());
      case Core::FE::CellType::tet4:
        return Teuchos::RCP(
            new BEAMINTERACTION::BeamToFluidMeshtyingPairGaussPoint<GEOMETRYPAIR::t_hermite,
                GEOMETRYPAIR::t_tet4>());
      case Core::FE::CellType::tet10:
        return Teuchos::RCP(
            new BEAMINTERACTION::BeamToFluidMeshtyingPairGaussPoint<GEOMETRYPAIR::t_hermite,
                GEOMETRYPAIR::t_tet10>());
      default:
        FOUR_C_THROW("Wrong element type for fluid element.");
    }
  }
  else if (meshtying_discretization == Inpar::FBI::BeamToFluidDiscretization::mortar)
  {
    Inpar::FBI::BeamToFluidMeshtingMortarShapefunctions mortar_shape_function =
        params_ptr.get_mortar_shape_function_type();

    switch (mortar_shape_function)
    {
      case Inpar::FBI::BeamToFluidMeshtingMortarShapefunctions::line2:
      {
        switch (shape)
        {
          case Core::FE::CellType::hex8:
            return Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex8, GEOMETRYPAIR::t_line2>>();
          case Core::FE::CellType::hex20:
            return Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex20, GEOMETRYPAIR::t_line2>>();
          case Core::FE::CellType::hex27:
            return Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex27, GEOMETRYPAIR::t_line2>>();
          case Core::FE::CellType::tet4:
            return Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_tet4, GEOMETRYPAIR::t_line2>>();
          case Core::FE::CellType::tet10:
            return Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_tet10, GEOMETRYPAIR::t_line2>>();
          default:
            FOUR_C_THROW("Wrong element type for solid element.");
        }
        break;
      }
      case Inpar::FBI::BeamToFluidMeshtingMortarShapefunctions::line3:
      {
        switch (shape)
        {
          case Core::FE::CellType::hex8:
            return Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex8, GEOMETRYPAIR::t_line3>>();
          case Core::FE::CellType::hex20:
            return Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex20, GEOMETRYPAIR::t_line3>>();
          case Core::FE::CellType::hex27:
            return Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex27, GEOMETRYPAIR::t_line3>>();
          case Core::FE::CellType::tet4:
            return Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_tet4, GEOMETRYPAIR::t_line3>>();
          case Core::FE::CellType::tet10:
            return Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_tet10, GEOMETRYPAIR::t_line3>>();
          default:
            FOUR_C_THROW("Wrong element type for solid element.");
        }
        break;
      }
      case Inpar::FBI::BeamToFluidMeshtingMortarShapefunctions::line4:
      {
        switch (shape)
        {
          case Core::FE::CellType::hex8:
            return Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex8, GEOMETRYPAIR::t_line4>>();
          case Core::FE::CellType::hex20:
            return Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex20, GEOMETRYPAIR::t_line4>>();
          case Core::FE::CellType::hex27:
            return Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_hex27, GEOMETRYPAIR::t_line4>>();
          case Core::FE::CellType::tet4:
            return Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMeshtyingPairMortar<
                GEOMETRYPAIR::t_hermite, GEOMETRYPAIR::t_tet4, GEOMETRYPAIR::t_line4>>();
          case Core::FE::CellType::tet10:
            return Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMeshtyingPairMortar<
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
  return Teuchos::null;
}

FOUR_C_NAMESPACE_CLOSE
