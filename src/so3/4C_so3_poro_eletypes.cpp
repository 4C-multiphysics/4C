// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_so3_poro_eletypes.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_so3_poro.hpp"

FOUR_C_NAMESPACE_OPEN

using namespace Core::IO::InputSpecBuilders;

/*----------------------------------------------------------------------*
 |  HEX 8 Element                                       |
 *----------------------------------------------------------------------*/
Discret::Elements::SoHex8PoroType Discret::Elements::SoHex8PoroType::instance_;

Discret::Elements::SoHex8PoroType& Discret::Elements::SoHex8PoroType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::SoHex8PoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object =
      new Discret::Elements::So3Poro<Discret::Elements::SoHex8, Core::FE::CellType::hex8>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoHex8PoroType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
        Discret::Elements::So3Poro<Discret::Elements::SoHex8, Core::FE::CellType::hex8>>(

        id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoHex8PoroType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
      Discret::Elements::So3Poro<Discret::Elements::SoHex8, Core::FE::CellType::hex8>>(

      id, owner);
  return ele;
}

void Discret::Elements::SoHex8PoroType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_hex8;
  SoHex8Type::setup_element_definition(definitions_hex8);

  auto& defs_hex8 = definitions_hex8["SOLIDH8_DEPRECATED"];

  auto& defs = definitions[get_element_type_string()];

  defs["HEX8"] = anonymous_group({
      defs_hex8["HEX8"],
      entry<std::vector<double>>("POROANISODIR1", {.required = false, .size = 3}),
      entry<std::vector<double>>("POROANISODIR2", {.required = false, .size = 3}),
      entry<std::vector<double>>("POROANISODIR3", {.required = false, .size = 3}),
      entry<std::vector<double>>("POROANISONODALCOEFFS1", {.required = false, .size = 8}),
      entry<std::vector<double>>("POROANISONODALCOEFFS2", {.required = false, .size = 8}),
      entry<std::vector<double>>("POROANISONODALCOEFFS3", {.required = false, .size = 8}),
  });
}

int Discret::Elements::SoHex8PoroType::initialize(Core::FE::Discretization& dis)
{
  SoHex8Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<
        Discret::Elements::So3Poro<Discret::Elements::SoHex8, Core::FE::CellType::hex8>*>(
        dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_hex8_poro* failed");
    actele->init_element();
  }
  return 0;
}


/*----------------------------------------------------------------------*
 |  TET 4 Element                                       |
 *----------------------------------------------------------------------*/
Discret::Elements::SoTet4PoroType Discret::Elements::SoTet4PoroType::instance_;

Discret::Elements::SoTet4PoroType& Discret::Elements::SoTet4PoroType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::SoTet4PoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object =
      new Discret::Elements::So3Poro<Discret::Elements::SoTet4, Core::FE::CellType::tet4>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoTet4PoroType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
        Discret::Elements::So3Poro<Discret::Elements::SoTet4, Core::FE::CellType::tet4>>(

        id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoTet4PoroType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
      Discret::Elements::So3Poro<Discret::Elements::SoTet4, Core::FE::CellType::tet4>>(

      id, owner);
  return ele;
}

void Discret::Elements::SoTet4PoroType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_tet4;
  SoTet4Type::setup_element_definition(definitions_tet4);

  auto& defs_tet4 = definitions_tet4["SOLIDT4_DEPRECATED"];

  auto& defs = definitions[get_element_type_string()];

  defs["TET4"] = anonymous_group({
      defs_tet4["TET4"],
      entry<std::vector<double>>("POROANISODIR1", {.required = false, .size = 3}),
      entry<std::vector<double>>("POROANISODIR2", {.required = false, .size = 3}),
      entry<std::vector<double>>("POROANISODIR3", {.required = false, .size = 3}),
      entry<std::vector<double>>("POROANISONODALCOEFFS1", {.required = false, .size = 4}),
      entry<std::vector<double>>("POROANISONODALCOEFFS2", {.required = false, .size = 4}),
      entry<std::vector<double>>("POROANISONODALCOEFFS3", {.required = false, .size = 4}),
  });
}

int Discret::Elements::SoTet4PoroType::initialize(Core::FE::Discretization& dis)
{
  SoTet4Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<
        Discret::Elements::So3Poro<Discret::Elements::SoTet4, Core::FE::CellType::tet4>*>(
        dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_tet4_poro* failed");
    actele->So3Poro<Discret::Elements::SoTet4, Core::FE::CellType::tet4>::init_element();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  HEX 27 Element                                       |
 *----------------------------------------------------------------------*/
Discret::Elements::SoHex27PoroType Discret::Elements::SoHex27PoroType::instance_;

Discret::Elements::SoHex27PoroType& Discret::Elements::SoHex27PoroType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::SoHex27PoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object =
      new Discret::Elements::So3Poro<Discret::Elements::SoHex27, Core::FE::CellType::hex27>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoHex27PoroType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
        Discret::Elements::So3Poro<Discret::Elements::SoHex27, Core::FE::CellType::hex27>>(

        id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoHex27PoroType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
      Discret::Elements::So3Poro<Discret::Elements::SoHex27, Core::FE::CellType::hex27>>(

      id, owner);
  return ele;
}

void Discret::Elements::SoHex27PoroType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_hex27;
  SoHex27Type::setup_element_definition(definitions_hex27);

  auto& defs_hex27 = definitions_hex27["SOLIDH27_DEPRECATED"];

  auto& defs = definitions[get_element_type_string()];

  defs["HEX27"] = anonymous_group({
      defs_hex27["HEX27"],
      entry<std::vector<double>>("POROANISODIR1", {.required = false, .size = 3}),
      entry<std::vector<double>>("POROANISODIR2", {.required = false, .size = 3}),
      entry<std::vector<double>>("POROANISODIR3", {.required = false, .size = 3}),
  });
}

int Discret::Elements::SoHex27PoroType::initialize(Core::FE::Discretization& dis)
{
  SoHex27Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<
        Discret::Elements::So3Poro<Discret::Elements::SoHex27, Core::FE::CellType::hex27>*>(
        dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_hex27_poro* failed");
    actele->So3Poro<Discret::Elements::SoHex27, Core::FE::CellType::hex27>::init_element();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  TET 10 Element                                       |
 *----------------------------------------------------------------------*/
Discret::Elements::SoTet10PoroType Discret::Elements::SoTet10PoroType::instance_;

Discret::Elements::SoTet10PoroType& Discret::Elements::SoTet10PoroType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::SoTet10PoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object =
      new Discret::Elements::So3Poro<Discret::Elements::SoTet10, Core::FE::CellType::tet10>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoTet10PoroType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
        Discret::Elements::So3Poro<Discret::Elements::SoTet10, Core::FE::CellType::tet10>>(

        id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoTet10PoroType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
      Discret::Elements::So3Poro<Discret::Elements::SoTet10, Core::FE::CellType::tet10>>(

      id, owner);
  return ele;
}

void Discret::Elements::SoTet10PoroType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_tet10;
  SoTet10Type::setup_element_definition(definitions_tet10);

  auto& defs_tet10 = definitions_tet10["SOLIDT10_DEPRECATED"];

  auto& defs = definitions[get_element_type_string()];

  defs["TET10"] = anonymous_group({
      defs_tet10["TET10"],
      entry<std::vector<double>>("POROANISODIR1", {.required = false, .size = 3}),
      entry<std::vector<double>>("POROANISODIR2", {.required = false, .size = 3}),
      entry<std::vector<double>>("POROANISODIR3", {.required = false, .size = 3}),
  });
}

int Discret::Elements::SoTet10PoroType::initialize(Core::FE::Discretization& dis)
{
  SoTet10Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<
        Discret::Elements::So3Poro<Discret::Elements::SoTet10, Core::FE::CellType::tet10>*>(
        dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_tet10_poro* failed");
    actele->So3Poro<Discret::Elements::SoTet10, Core::FE::CellType::tet10>::init_element();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  NURBS 27 Element                                       |
 *----------------------------------------------------------------------*/
Discret::Elements::SoNurbs27PoroType Discret::Elements::SoNurbs27PoroType::instance_;

Discret::Elements::SoNurbs27PoroType& Discret::Elements::SoNurbs27PoroType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::SoNurbs27PoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::So3Poro<Discret::Elements::Nurbs::SoNurbs27,
      Core::FE::CellType::nurbs27>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoNurbs27PoroType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::So3Poro<Discret::Elements::Nurbs::SoNurbs27,
            Core::FE::CellType::nurbs27>>(id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoNurbs27PoroType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
      Discret::Elements::So3Poro<Discret::Elements::Nurbs::SoNurbs27, Core::FE::CellType::nurbs27>>(
      id, owner);
  return ele;
}

void Discret::Elements::SoNurbs27PoroType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_nurbs27;
  Nurbs::SoNurbs27Type::setup_element_definition(definitions_nurbs27);

  auto& defs_nurbs27 = definitions_nurbs27["SONURBS27_DEPRECATED"];

  auto& defs = definitions[get_element_type_string()];

  defs["NURBS27"] = anonymous_group({
      defs_nurbs27["NURBS27"],
      entry<std::vector<double>>("POROANISODIR1", {.required = false, .size = 3}),
      entry<std::vector<double>>("POROANISODIR2", {.required = false, .size = 3}),
      entry<std::vector<double>>("POROANISODIR3", {.required = false, .size = 3}),
  });
}

int Discret::Elements::SoNurbs27PoroType::initialize(Core::FE::Discretization& dis)
{
  Nurbs::SoNurbs27Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<Discret::Elements::So3Poro<Discret::Elements::Nurbs::SoNurbs27,
        Core::FE::CellType::nurbs27>*>(dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_nurbs27_poro* failed");
    actele
        ->So3Poro<Discret::Elements::Nurbs::SoNurbs27, Core::FE::CellType::nurbs27>::init_element();
  }
  return 0;
}

FOUR_C_NAMESPACE_CLOSE
