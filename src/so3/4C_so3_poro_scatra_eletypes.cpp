// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_so3_poro_scatra_eletypes.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_so3_poro_scatra.hpp"

FOUR_C_NAMESPACE_OPEN

using namespace Core::IO::InputSpecBuilders;

/*----------------------------------------------------------------------*
 |  HEX 8 Element                                         schmidt 09/17 |
 *----------------------------------------------------------------------*/
Discret::Elements::SoHex8PoroScatraType Discret::Elements::SoHex8PoroScatraType::instance_;

Discret::Elements::SoHex8PoroScatraType& Discret::Elements::SoHex8PoroScatraType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::SoHex8PoroScatraType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::So3PoroScatra<Discret::Elements::SoHex8, Core::FE::CellType::hex8>* object =
      new Discret::Elements::So3PoroScatra<Discret::Elements::SoHex8, Core::FE::CellType::hex8>(
          -1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoHex8PoroScatraType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
        Discret::Elements::So3PoroScatra<Discret::Elements::SoHex8, Core::FE::CellType::hex8>>(

        id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoHex8PoroScatraType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
      Discret::Elements::So3PoroScatra<Discret::Elements::SoHex8, Core::FE::CellType::hex8>>(

      id, owner);
  return ele;
}

void Discret::Elements::SoHex8PoroScatraType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_hex8;
  SoHex8PoroType::setup_element_definition(definitions_hex8);

  auto& defs_hex8 = definitions_hex8["SOLIDH8PORO"];

  auto& defs = definitions[get_element_type_string()];


  defs["HEX8"] = anonymous_group({
      defs_hex8["HEX8"],
      entry<std::string>("TYPE"),
  });
}

/*----------------------------------------------------------------------*
 |  TET 4 Element                                         schmidt 09/17 |
 *----------------------------------------------------------------------*/


Discret::Elements::SoTet4PoroScatraType Discret::Elements::SoTet4PoroScatraType::instance_;

Discret::Elements::SoTet4PoroScatraType& Discret::Elements::SoTet4PoroScatraType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::SoTet4PoroScatraType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::So3PoroScatra<Discret::Elements::SoTet4, Core::FE::CellType::tet4>* object =
      new Discret::Elements::So3PoroScatra<Discret::Elements::SoTet4, Core::FE::CellType::tet4>(
          -1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoTet4PoroScatraType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
        Discret::Elements::So3PoroScatra<Discret::Elements::SoTet4, Core::FE::CellType::tet4>>(

        id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoTet4PoroScatraType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
      Discret::Elements::So3PoroScatra<Discret::Elements::SoTet4, Core::FE::CellType::tet4>>(

      id, owner);
  return ele;
}

void Discret::Elements::SoTet4PoroScatraType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_tet4;
  SoTet4PoroType::setup_element_definition(definitions_tet4);

  auto& defs_tet4 = definitions_tet4["SOLIDT4PORO"];

  auto& defs = definitions[get_element_type_string()];

  defs["TET4"] = anonymous_group({
      defs_tet4["TET4"],
      entry<std::string>("TYPE"),
  });
}

/*----------------------------------------------------------------------*
 |  HEX 27 Element                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/


Discret::Elements::SoHex27PoroScatraType Discret::Elements::SoHex27PoroScatraType::instance_;

Discret::Elements::SoHex27PoroScatraType& Discret::Elements::SoHex27PoroScatraType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::SoHex27PoroScatraType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::So3PoroScatra<Discret::Elements::SoHex27, Core::FE::CellType::hex27>* object =
      new Discret::Elements::So3PoroScatra<Discret::Elements::SoHex27, Core::FE::CellType::hex27>(
          -1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoHex27PoroScatraType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
        Discret::Elements::So3PoroScatra<Discret::Elements::SoHex27, Core::FE::CellType::hex27>>(

        id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoHex27PoroScatraType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
      Discret::Elements::So3PoroScatra<Discret::Elements::SoHex27, Core::FE::CellType::hex27>>(

      id, owner);
  return ele;
}

void Discret::Elements::SoHex27PoroScatraType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_hex27;
  SoHex27PoroType::setup_element_definition(definitions_hex27);

  auto& defs_hex27 = definitions_hex27["SOLIDH27PORO"];

  auto& defs = definitions[get_element_type_string()];

  defs["HEX27"] = anonymous_group({
      defs_hex27["HEX27"],
      entry<std::string>("TYPE"),
  });
}


/*----------------------------------------------------------------------*
 |  TET 10 Element                                        schmidt 09/17 |
 *----------------------------------------------------------------------*/


Discret::Elements::SoTet10PoroScatraType Discret::Elements::SoTet10PoroScatraType::instance_;

Discret::Elements::SoTet10PoroScatraType& Discret::Elements::SoTet10PoroScatraType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::SoTet10PoroScatraType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::So3PoroScatra<Discret::Elements::SoTet10, Core::FE::CellType::tet10>* object =
      new Discret::Elements::So3PoroScatra<Discret::Elements::SoTet10, Core::FE::CellType::tet10>(
          -1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoTet10PoroScatraType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
        Discret::Elements::So3PoroScatra<Discret::Elements::SoTet10, Core::FE::CellType::tet10>>(

        id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoTet10PoroScatraType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
      Discret::Elements::So3PoroScatra<Discret::Elements::SoTet10, Core::FE::CellType::tet10>>(

      id, owner);
  return ele;
}

void Discret::Elements::SoTet10PoroScatraType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_tet10;
  SoTet10PoroType::setup_element_definition(definitions_tet10);

  auto& defs_tet10 = definitions_tet10["SOLIDT10PORO"];

  auto& defs = definitions[get_element_type_string()];

  defs["TET10"] = anonymous_group({
      defs_tet10["TET10"],
      entry<std::string>("TYPE"),
  });
}

/*----------------------------------------------------------------------*
 |  NURBS 27 Element                                      schmidt 09/17 |
 *----------------------------------------------------------------------*/


Discret::Elements::SoNurbs27PoroScatraType Discret::Elements::SoNurbs27PoroScatraType::instance_;

Discret::Elements::SoNurbs27PoroScatraType& Discret::Elements::SoNurbs27PoroScatraType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::SoNurbs27PoroScatraType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::So3PoroScatra<Discret::Elements::Nurbs::SoNurbs27,
      Core::FE::CellType::nurbs27>* object =
      new Discret::Elements::So3PoroScatra<Discret::Elements::Nurbs::SoNurbs27,
          Core::FE::CellType::nurbs27>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoNurbs27PoroScatraType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::So3PoroScatra<Discret::Elements::Nurbs::SoNurbs27,
            Core::FE::CellType::nurbs27>>(id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoNurbs27PoroScatraType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::So3PoroScatra<Discret::Elements::Nurbs::SoNurbs27,
          Core::FE::CellType::nurbs27>>(id, owner);
  return ele;
}

void Discret::Elements::SoNurbs27PoroScatraType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_nurbs27;
  SoNurbs27PoroType::setup_element_definition(definitions_nurbs27);

  auto& defs_nurbs27 = definitions_nurbs27["SONURBS27PORO"];

  auto& defs = definitions[get_element_type_string()];

  defs["NURBS27"] = anonymous_group({
      defs_nurbs27["NURBS27"],
      entry<std::string>("TYPE"),
  });
}

FOUR_C_NAMESPACE_CLOSE
