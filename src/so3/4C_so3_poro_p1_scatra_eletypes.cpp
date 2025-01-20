// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_so3_poro_p1_scatra_eletypes.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_so3_poro_p1_scatra.hpp"

FOUR_C_NAMESPACE_OPEN

using namespace Core::IO::InputSpecBuilders;

/*----------------------------------------------------------------------*
 |  HEX 8 Element                                         schmidt 09/17 |
 *----------------------------------------------------------------------*/
Discret::Elements::SoHex8PoroP1ScatraType Discret::Elements::SoHex8PoroP1ScatraType::instance_;

Discret::Elements::SoHex8PoroP1ScatraType& Discret::Elements::SoHex8PoroP1ScatraType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::SoHex8PoroP1ScatraType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::So3PoroP1Scatra<Discret::Elements::SoHex8, Core::FE::CellType::hex8>* object =
      new Discret::Elements::So3PoroP1Scatra<Discret::Elements::SoHex8, Core::FE::CellType::hex8>(
          -1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoHex8PoroP1ScatraType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
        Discret::Elements::So3PoroP1Scatra<Discret::Elements::SoHex8, Core::FE::CellType::hex8>>(

        id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoHex8PoroP1ScatraType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
      Discret::Elements::So3PoroP1Scatra<Discret::Elements::SoHex8, Core::FE::CellType::hex8>>(

      id, owner);
  return ele;
}

void Discret::Elements::SoHex8PoroP1ScatraType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_hex8poro;
  SoHex8PoroP1Type::setup_element_definition(definitions_hex8poro);

  auto& defs_hex8 = definitions_hex8poro["SOLIDH8POROP1"];

  auto& defs = definitions[get_element_type_string()];

  defs["HEX8"] = anonymous_group({
      defs_hex8["HEX8"],
      entry<std::string>("TYPE"),
  });
}

/*----------------------------------------------------------------------*
 |  TET 4 Element                                         schmidt 09/17 |
 *----------------------------------------------------------------------*/
Discret::Elements::SoTet4PoroP1ScatraType Discret::Elements::SoTet4PoroP1ScatraType::instance_;

Discret::Elements::SoTet4PoroP1ScatraType& Discret::Elements::SoTet4PoroP1ScatraType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::SoTet4PoroP1ScatraType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::So3PoroP1Scatra<Discret::Elements::SoTet4, Core::FE::CellType::tet4>* object =
      new Discret::Elements::So3PoroP1Scatra<Discret::Elements::SoTet4, Core::FE::CellType::tet4>(
          -1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoTet4PoroP1ScatraType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
        Discret::Elements::So3PoroP1Scatra<Discret::Elements::SoTet4, Core::FE::CellType::tet4>>(

        id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoTet4PoroP1ScatraType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
      Discret::Elements::So3PoroP1Scatra<Discret::Elements::SoTet4, Core::FE::CellType::tet4>>(

      id, owner);
  return ele;
}

void Discret::Elements::SoTet4PoroP1ScatraType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_tet4;
  SoTet4PoroP1Type::setup_element_definition(definitions_tet4);

  auto& defs_tet4 = definitions_tet4["SOLIDT4POROP1"];

  auto& defs = definitions[get_element_type_string()];

  defs["TET4"] = anonymous_group({
      defs_tet4["TET4"],
      entry<std::string>("TYPE"),
  });
}

FOUR_C_NAMESPACE_CLOSE
