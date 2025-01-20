// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_w1_poro_p1_scatra_eletypes.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_w1_poro_p1_scatra.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  QUAD 4 Element                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/

Discret::Elements::WallQuad4PoroP1ScatraType
    Discret::Elements::WallQuad4PoroP1ScatraType::instance_;

Discret::Elements::WallQuad4PoroP1ScatraType&
Discret::Elements::WallQuad4PoroP1ScatraType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::WallQuad4PoroP1ScatraType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::quad4>* object =
      new Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::quad4>(-1, -1);
  object->unpack(buffer);
  return object;
}

/*----------------------------------------------------------------------*
 |                                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::WallQuad4PoroP1ScatraType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "WALLQ4POROP1SCATRA")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::quad4>>(
            id, owner);
    return ele;
  }
  return nullptr;
}

/*----------------------------------------------------------------------*
 |                                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::WallQuad4PoroP1ScatraType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::quad4>>(id, owner);
  return ele;
}

/*----------------------------------------------------------------------*
 |                                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/
void Discret::Elements::WallQuad4PoroP1ScatraType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_wallporo;
  WallQuad4PoroP1Type::setup_element_definition(definitions_wallporo);

  auto& defs_wallporo = definitions_wallporo["WALLQ4POROP1"];

  auto& defs = definitions["WALLQ4POROP1SCATRA"];

  using namespace Core::IO::InputSpecBuilders;

  defs["QUAD4"] = anonymous_group({
      defs_wallporo["QUAD4"],
      entry<std::string>("TYPE"),
  });
}

/*----------------------------------------------------------------------*
 |  QUAD 9 Element                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/

Discret::Elements::WallQuad9PoroP1ScatraType
    Discret::Elements::WallQuad9PoroP1ScatraType::instance_;

Discret::Elements::WallQuad9PoroP1ScatraType&
Discret::Elements::WallQuad9PoroP1ScatraType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::WallQuad9PoroP1ScatraType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::quad9>* object =
      new Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::quad9>(-1, -1);
  object->unpack(buffer);
  return object;
}

/*----------------------------------------------------------------------*
 |                                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::WallQuad9PoroP1ScatraType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "WALLQ9POROP1SCATRA")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::quad9>>(
            id, owner);
    return ele;
  }
  return nullptr;
}

/*----------------------------------------------------------------------*
 |                                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::WallQuad9PoroP1ScatraType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::quad9>>(id, owner);
  return ele;
}

/*----------------------------------------------------------------------*
 |                                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/
void Discret::Elements::WallQuad9PoroP1ScatraType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_wallporo;
  WallQuad9PoroP1Type::setup_element_definition(definitions_wallporo);

  auto& defs_wallporo = definitions_wallporo["WALLQ9POROP1"];

  auto& defs = definitions["WALLQ9POROP1SCATRA"];

  using namespace Core::IO::InputSpecBuilders;

  defs["QUAD9"] = anonymous_group({
      defs_wallporo["QUAD9"],
      entry<std::string>("TYPE"),
  });
}

/*----------------------------------------------------------------------*
 |  TRI 3 Element                                         schmidt 09/17 |
 *----------------------------------------------------------------------*/

Discret::Elements::WallTri3PoroP1ScatraType Discret::Elements::WallTri3PoroP1ScatraType::instance_;

Discret::Elements::WallTri3PoroP1ScatraType& Discret::Elements::WallTri3PoroP1ScatraType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::WallTri3PoroP1ScatraType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::tri3>* object =
      new Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::tri3>(-1, -1);
  object->unpack(buffer);
  return object;
}

/*----------------------------------------------------------------------*
 |                                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::WallTri3PoroP1ScatraType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "WALLT3POROP1SCATRA")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::tri3>>(id, owner);
    return ele;
  }
  return nullptr;
}

/*----------------------------------------------------------------------*
 |                                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::WallTri3PoroP1ScatraType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Wall1PoroP1Scatra<Core::FE::CellType::tri3>>(id, owner);
  return ele;
}

/*----------------------------------------------------------------------*
 |                                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/
void Discret::Elements::WallTri3PoroP1ScatraType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_wallporo;
  WallTri3PoroP1Type::setup_element_definition(definitions_wallporo);

  auto& defs_wallporo = definitions_wallporo["WALLT3POROP1"];

  auto& defs = definitions["WALLT3POROP1SCATRA"];

  using namespace Core::IO::InputSpecBuilders;

  defs["TRI3"] = anonymous_group({
      defs_wallporo["TRI3"],
      entry<std::string>("TYPE"),
  });
}

FOUR_C_NAMESPACE_CLOSE
