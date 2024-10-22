// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_so3_poro_eletypes.hpp"

#include "4C_io_linedefinition.hpp"
#include "4C_so3_poro.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  HEX 8 Element                                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoHex8PoroType Discret::ELEMENTS::SoHex8PoroType::instance_;

Discret::ELEMENTS::SoHex8PoroType& Discret::ELEMENTS::SoHex8PoroType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::ELEMENTS::SoHex8PoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object =
      new Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoHex8, Core::FE::CellType::hex8>(-1, -1);
  object->unpack(buffer);
  return object;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoHex8PoroType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
        Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoHex8, Core::FE::CellType::hex8>>(

        id, owner);
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoHex8PoroType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
      Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoHex8, Core::FE::CellType::hex8>>(

      id, owner);
  return ele;
}

void Discret::ELEMENTS::SoHex8PoroType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_hex8;
  SoHex8Type::setup_element_definition(definitions_hex8);

  std::map<std::string, Input::LineDefinition>& defs_hex8 = definitions_hex8["SOLIDH8_DEPRECATED"];

  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["HEX8"] = Input::LineDefinition::Builder(defs_hex8["HEX8"])
                     .add_optional_named_double_vector("POROANISODIR1", 3)
                     .add_optional_named_double_vector("POROANISODIR2", 3)
                     .add_optional_named_double_vector("POROANISODIR3", 3)
                     .add_optional_named_double_vector("POROANISONODALCOEFFS1", 8)
                     .add_optional_named_double_vector("POROANISONODALCOEFFS2", 8)
                     .add_optional_named_double_vector("POROANISONODALCOEFFS3", 8)
                     .build();
}

int Discret::ELEMENTS::SoHex8PoroType::initialize(Core::FE::Discretization& dis)
{
  SoHex8Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<
        Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoHex8, Core::FE::CellType::hex8>*>(
        dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_hex8_poro* failed");
    actele->init_element();
  }
  return 0;
}


/*----------------------------------------------------------------------*
 |  TET 4 Element                                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoTet4PoroType Discret::ELEMENTS::SoTet4PoroType::instance_;

Discret::ELEMENTS::SoTet4PoroType& Discret::ELEMENTS::SoTet4PoroType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::ELEMENTS::SoTet4PoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object =
      new Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoTet4, Core::FE::CellType::tet4>(-1, -1);
  object->unpack(buffer);
  return object;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoTet4PoroType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
        Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoTet4, Core::FE::CellType::tet4>>(

        id, owner);
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoTet4PoroType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
      Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoTet4, Core::FE::CellType::tet4>>(

      id, owner);
  return ele;
}

void Discret::ELEMENTS::SoTet4PoroType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_tet4;
  SoTet4Type::setup_element_definition(definitions_tet4);

  std::map<std::string, Input::LineDefinition>& defs_tet4 = definitions_tet4["SOLIDT4_DEPRECATED"];

  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["TET4"] = Input::LineDefinition::Builder(defs_tet4["TET4"])
                     .add_optional_named_double_vector("POROANISODIR1", 3)
                     .add_optional_named_double_vector("POROANISODIR2", 3)
                     .add_optional_named_double_vector("POROANISODIR3", 3)
                     .add_optional_named_double_vector("POROANISONODALCOEFFS1", 4)
                     .add_optional_named_double_vector("POROANISONODALCOEFFS2", 4)
                     .add_optional_named_double_vector("POROANISONODALCOEFFS3", 4)
                     .build();
}

int Discret::ELEMENTS::SoTet4PoroType::initialize(Core::FE::Discretization& dis)
{
  SoTet4Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<
        Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoTet4, Core::FE::CellType::tet4>*>(
        dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_tet4_poro* failed");
    actele->So3Poro<Discret::ELEMENTS::SoTet4, Core::FE::CellType::tet4>::init_element();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  HEX 27 Element                                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoHex27PoroType Discret::ELEMENTS::SoHex27PoroType::instance_;

Discret::ELEMENTS::SoHex27PoroType& Discret::ELEMENTS::SoHex27PoroType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::ELEMENTS::SoHex27PoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object =
      new Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoHex27, Core::FE::CellType::hex27>(-1, -1);
  object->unpack(buffer);
  return object;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoHex27PoroType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
        Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoHex27, Core::FE::CellType::hex27>>(

        id, owner);
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoHex27PoroType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
      Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoHex27, Core::FE::CellType::hex27>>(

      id, owner);
  return ele;
}

void Discret::ELEMENTS::SoHex27PoroType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_hex27;
  SoHex27Type::setup_element_definition(definitions_hex27);

  std::map<std::string, Input::LineDefinition>& defs_hex27 =
      definitions_hex27["SOLIDH27_DEPRECATED"];

  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["HEX27"] = Input::LineDefinition::Builder(defs_hex27["HEX27"])
                      .add_optional_named_double_vector("POROANISODIR1", 3)
                      .add_optional_named_double_vector("POROANISODIR2", 3)
                      .add_optional_named_double_vector("POROANISODIR3", 3)
                      .build();
}

int Discret::ELEMENTS::SoHex27PoroType::initialize(Core::FE::Discretization& dis)
{
  SoHex27Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<
        Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoHex27, Core::FE::CellType::hex27>*>(
        dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_hex27_poro* failed");
    actele->So3Poro<Discret::ELEMENTS::SoHex27, Core::FE::CellType::hex27>::init_element();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  TET 10 Element                                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoTet10PoroType Discret::ELEMENTS::SoTet10PoroType::instance_;

Discret::ELEMENTS::SoTet10PoroType& Discret::ELEMENTS::SoTet10PoroType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::ELEMENTS::SoTet10PoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object =
      new Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoTet10, Core::FE::CellType::tet10>(-1, -1);
  object->unpack(buffer);
  return object;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoTet10PoroType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
        Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoTet10, Core::FE::CellType::tet10>>(

        id, owner);
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoTet10PoroType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
      Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoTet10, Core::FE::CellType::tet10>>(

      id, owner);
  return ele;
}

void Discret::ELEMENTS::SoTet10PoroType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_tet10;
  SoTet10Type::setup_element_definition(definitions_tet10);

  std::map<std::string, Input::LineDefinition>& defs_tet10 =
      definitions_tet10["SOLIDT10_DEPRECATED"];

  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["TET10"] = Input::LineDefinition::Builder(defs_tet10["TET10"])
                      .add_optional_named_double_vector("POROANISODIR1", 3)
                      .add_optional_named_double_vector("POROANISODIR2", 3)
                      .add_optional_named_double_vector("POROANISODIR3", 3)
                      .build();
}

int Discret::ELEMENTS::SoTet10PoroType::initialize(Core::FE::Discretization& dis)
{
  SoTet10Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<
        Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::SoTet10, Core::FE::CellType::tet10>*>(
        dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_tet10_poro* failed");
    actele->So3Poro<Discret::ELEMENTS::SoTet10, Core::FE::CellType::tet10>::init_element();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  NURBS 27 Element                                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::SoNurbs27PoroType Discret::ELEMENTS::SoNurbs27PoroType::instance_;

Discret::ELEMENTS::SoNurbs27PoroType& Discret::ELEMENTS::SoNurbs27PoroType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::ELEMENTS::SoNurbs27PoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::Nurbs::SoNurbs27,
      Core::FE::CellType::nurbs27>(-1, -1);
  object->unpack(buffer);
  return object;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoNurbs27PoroType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    Teuchos::RCP<Core::Elements::Element> ele =
        Teuchos::make_rcp<Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::Nurbs::SoNurbs27,
            Core::FE::CellType::nurbs27>>(id, owner);
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::SoNurbs27PoroType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
      Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::Nurbs::SoNurbs27, Core::FE::CellType::nurbs27>>(
      id, owner);
  return ele;
}

void Discret::ELEMENTS::SoNurbs27PoroType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_nurbs27;
  Nurbs::SoNurbs27Type::setup_element_definition(definitions_nurbs27);

  std::map<std::string, Input::LineDefinition>& defs_nurbs27 = definitions_nurbs27["SONURBS27"];

  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["NURBS27"] = Input::LineDefinition::Builder(defs_nurbs27["NURBS27"])
                        .add_optional_named_double_vector("POROANISODIR1", 3)
                        .add_optional_named_double_vector("POROANISODIR2", 3)
                        .add_optional_named_double_vector("POROANISODIR3", 3)
                        .build();
}

int Discret::ELEMENTS::SoNurbs27PoroType::initialize(Core::FE::Discretization& dis)
{
  Nurbs::SoNurbs27Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<Discret::ELEMENTS::So3Poro<Discret::ELEMENTS::Nurbs::SoNurbs27,
        Core::FE::CellType::nurbs27>*>(dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_nurbs27_poro* failed");
    actele
        ->So3Poro<Discret::ELEMENTS::Nurbs::SoNurbs27, Core::FE::CellType::nurbs27>::init_element();
  }
  return 0;
}

FOUR_C_NAMESPACE_CLOSE
