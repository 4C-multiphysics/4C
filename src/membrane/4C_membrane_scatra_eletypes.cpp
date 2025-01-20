// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_membrane_scatra_eletypes.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_membrane_scatra.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  TRI 3 Element                                                       |
 *----------------------------------------------------------------------*/
Discret::Elements::MembraneScatraTri3Type Discret::Elements::MembraneScatraTri3Type::instance_;

Discret::Elements::MembraneScatraTri3Type& Discret::Elements::MembraneScatraTri3Type::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::MembraneScatraTri3Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::MembraneScatra<Core::FE::CellType::tri3>* object =
      new Discret::Elements::MembraneScatra<Core::FE::CellType::tri3>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneScatraTri3Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "MEMBRANESCATRA3" && eledistype == "TRI3")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::MembraneScatra<Core::FE::CellType::tri3>>(id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneScatraTri3Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::MembraneScatra<Core::FE::CellType::tri3>>(id, owner);
  return ele;
}

void Discret::Elements::MembraneScatraTri3Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_membrane;
  MembraneTri3Type::setup_element_definition(definitions_membrane);

  auto& defs_membrane = definitions_membrane["MEMBRANE3"];
  auto& defs = definitions["MEMBRANESCATRA3"];

  using namespace Core::IO::InputSpecBuilders;

  defs["TRI3"] = anonymous_group({
      defs_membrane["TRI3"],
      entry<std::string>("TYPE"),
  });
}

/*----------------------------------------------------------------------*
 |  TRI 6 Element                                                       |
 *----------------------------------------------------------------------*/
Discret::Elements::MembraneScatraTri6Type Discret::Elements::MembraneScatraTri6Type::instance_;

Discret::Elements::MembraneScatraTri6Type& Discret::Elements::MembraneScatraTri6Type::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::MembraneScatraTri6Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::MembraneScatra<Core::FE::CellType::tri6>* object =
      new Discret::Elements::MembraneScatra<Core::FE::CellType::tri6>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneScatraTri6Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "MEMBRANESCATRA6" && eledistype == "TRI6")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::MembraneScatra<Core::FE::CellType::tri6>>(id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneScatraTri6Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::MembraneScatra<Core::FE::CellType::tri6>>(id, owner);
  return ele;
}

void Discret::Elements::MembraneScatraTri6Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_membrane;
  MembraneTri6Type::setup_element_definition(definitions_membrane);

  auto& defs_membrane = definitions_membrane["MEMBRANE6"];

  auto& defs = definitions["MEMBRANESCATRA6"];

  using namespace Core::IO::InputSpecBuilders;

  defs["TRI6"] = anonymous_group({
      defs_membrane["TRI6"],
      entry<std::string>("TYPE"),
  });
}

/*----------------------------------------------------------------------*
 |  QUAD 4 Element                                                      |
 *----------------------------------------------------------------------*/
Discret::Elements::MembraneScatraQuad4Type Discret::Elements::MembraneScatraQuad4Type::instance_;

Discret::Elements::MembraneScatraQuad4Type& Discret::Elements::MembraneScatraQuad4Type::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::MembraneScatraQuad4Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::MembraneScatra<Core::FE::CellType::quad4>* object =
      new Discret::Elements::MembraneScatra<Core::FE::CellType::quad4>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneScatraQuad4Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "MEMBRANESCATRA4" && eledistype == "QUAD4")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::MembraneScatra<Core::FE::CellType::quad4>>(id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneScatraQuad4Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::MembraneScatra<Core::FE::CellType::quad4>>(id, owner);
  return ele;
}

void Discret::Elements::MembraneScatraQuad4Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_membrane;
  MembraneQuad4Type::setup_element_definition(definitions_membrane);

  auto& defs_membrane = definitions_membrane["MEMBRANE4"];

  auto& defs = definitions["MEMBRANESCATRA4"];

  using namespace Core::IO::InputSpecBuilders;

  defs["QUAD4"] = anonymous_group({
      defs_membrane["QUAD4"],
      entry<std::string>("TYPE"),
  });
}

/*----------------------------------------------------------------------*
 |  QUAD 9 Element                                                      |
 *----------------------------------------------------------------------*/
Discret::Elements::MembraneScatraQuad9Type Discret::Elements::MembraneScatraQuad9Type::instance_;

Discret::Elements::MembraneScatraQuad9Type& Discret::Elements::MembraneScatraQuad9Type::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::MembraneScatraQuad9Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::MembraneScatra<Core::FE::CellType::quad9>* object =
      new Discret::Elements::MembraneScatra<Core::FE::CellType::quad9>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneScatraQuad9Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "MEMBRANESCATRA9" && eledistype == "QUAD9")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::MembraneScatra<Core::FE::CellType::quad9>>(id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneScatraQuad9Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::MembraneScatra<Core::FE::CellType::quad9>>(id, owner);
  return ele;
}

void Discret::Elements::MembraneScatraQuad9Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_membrane;
  MembraneQuad9Type::setup_element_definition(definitions_membrane);

  auto& defs_membrane = definitions_membrane["MEMBRANE9"];

  auto& defs = definitions["MEMBRANESCATRA9"];

  using namespace Core::IO::InputSpecBuilders;

  defs["QUAD9"] = anonymous_group({
      defs_membrane["QUAD9"],
      entry<std::string>("TYPE"),
  });
}

FOUR_C_NAMESPACE_CLOSE
