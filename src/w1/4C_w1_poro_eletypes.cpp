/*----------------------------------------------------------------------------*/
/*! \file
\brief Element types of the 2D solid-poro element.

\level 2


*/
/*---------------------------------------------------------------------------*/

#include "4C_w1_poro_eletypes.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_w1_poro.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  QUAD 4 Element                                       |
 *----------------------------------------------------------------------*/

Discret::ELEMENTS::WallQuad4PoroType Discret::ELEMENTS::WallQuad4PoroType::instance_;

Discret::ELEMENTS::WallQuad4PoroType& Discret::ELEMENTS::WallQuad4PoroType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::ELEMENTS::WallQuad4PoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::quad4>(-1, -1);
  object->unpack(buffer);
  return object;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::WallQuad4PoroType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "WALLQ4PORO")
  {
    Teuchos::RCP<Core::Elements::Element> ele =
        Teuchos::RCP(new Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::quad4>(id, owner));
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::WallQuad4PoroType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele =
      Teuchos::RCP(new Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::quad4>(id, owner));
  return ele;
}

void Discret::ELEMENTS::WallQuad4PoroType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_wall;
  Wall1Type::setup_element_definition(definitions_wall);

  std::map<std::string, Input::LineDefinition>& defs_wall = definitions_wall["WALL"];

  std::map<std::string, Input::LineDefinition>& defs = definitions["WALLQ4PORO"];

  defs["QUAD4"] = Input::LineDefinition::Builder(defs_wall["QUAD4"])
                      .add_optional_named_double_vector("POROANISODIR1", 2)
                      .add_optional_named_double_vector("POROANISODIR2", 2)
                      .add_optional_named_double_vector("POROANISONODALCOEFFS1", 4)
                      .add_optional_named_double_vector("POROANISONODALCOEFFS2", 4)
                      .build();
}

int Discret::ELEMENTS::WallQuad4PoroType::initialize(Core::FE::Discretization& dis)
{
  Discret::ELEMENTS::Wall1Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::quad4>*>(
        dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to Wall1_Poro* failed");
    actele->init_element();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  QUAD 9 Element                                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::WallQuad9PoroType Discret::ELEMENTS::WallQuad9PoroType::instance_;

Discret::ELEMENTS::WallQuad9PoroType& Discret::ELEMENTS::WallQuad9PoroType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::ELEMENTS::WallQuad9PoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::quad9>(-1, -1);
  object->unpack(buffer);
  return object;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::WallQuad9PoroType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "WALLQ9PORO")
  {
    Teuchos::RCP<Core::Elements::Element> ele =
        Teuchos::RCP(new Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::quad9>(id, owner));
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::WallQuad9PoroType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele =
      Teuchos::RCP(new Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::quad9>(id, owner));
  return ele;
}

void Discret::ELEMENTS::WallQuad9PoroType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_wall;
  Wall1Type::setup_element_definition(definitions_wall);

  std::map<std::string, Input::LineDefinition>& defs_wall = definitions_wall["WALL"];

  std::map<std::string, Input::LineDefinition>& defs = definitions["WALLQ9PORO"];

  defs["QUAD9"] = Input::LineDefinition::Builder(defs_wall["QUAD9"])
                      .add_optional_named_double_vector("POROANISODIR1", 2)
                      .add_optional_named_double_vector("POROANISODIR2", 2)
                      .build();
}

int Discret::ELEMENTS::WallQuad9PoroType::initialize(Core::FE::Discretization& dis)
{
  Discret::ELEMENTS::Wall1Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::quad9>*>(
        dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to Wall1_Poro* failed");
    actele->init_element();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  NURBS 4 Element                                       |
 *----------------------------------------------------------------------*/

Discret::ELEMENTS::WallNurbs4PoroType Discret::ELEMENTS::WallNurbs4PoroType::instance_;

Discret::ELEMENTS::WallNurbs4PoroType& Discret::ELEMENTS::WallNurbs4PoroType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::ELEMENTS::WallNurbs4PoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::nurbs4>(-1, -1);
  object->unpack(buffer);
  return object;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::WallNurbs4PoroType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "WALLN4PORO")
  {
    Teuchos::RCP<Core::Elements::Element> ele =
        Teuchos::RCP(new Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::nurbs4>(id, owner));
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::WallNurbs4PoroType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele =
      Teuchos::RCP(new Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::nurbs4>(id, owner));
  return ele;
}

void Discret::ELEMENTS::WallNurbs4PoroType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_wall;
  Wall1Type::setup_element_definition(definitions_wall);

  std::map<std::string, Input::LineDefinition>& defs_wall = definitions_wall["WALL"];

  std::map<std::string, Input::LineDefinition>& defs = definitions["WALLN4PORO"];

  defs["NURBS4"] = Input::LineDefinition::Builder(defs_wall["NURBS4"])
                       .add_optional_named_double_vector("POROANISODIR1", 2)
                       .add_optional_named_double_vector("POROANISODIR2", 2)
                       .build();
}

int Discret::ELEMENTS::WallNurbs4PoroType::initialize(Core::FE::Discretization& dis)
{
  Discret::ELEMENTS::Wall1Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::nurbs4>*>(
        dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to Wall1_Poro* failed");
    actele->init_element();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  NURBS 9 Element                                       |
 *----------------------------------------------------------------------*/

Discret::ELEMENTS::WallNurbs9PoroType Discret::ELEMENTS::WallNurbs9PoroType::instance_;

Discret::ELEMENTS::WallNurbs9PoroType& Discret::ELEMENTS::WallNurbs9PoroType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::ELEMENTS::WallNurbs9PoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::nurbs9>(-1, -1);
  object->unpack(buffer);
  return object;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::WallNurbs9PoroType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "WALLN9PORO")
  {
    Teuchos::RCP<Core::Elements::Element> ele =
        Teuchos::RCP(new Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::nurbs9>(id, owner));
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::WallNurbs9PoroType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele =
      Teuchos::RCP(new Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::nurbs9>(id, owner));
  return ele;
}

void Discret::ELEMENTS::WallNurbs9PoroType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_wall;
  Wall1Type::setup_element_definition(definitions_wall);

  std::map<std::string, Input::LineDefinition>& defs_wall = definitions_wall["WALL"];

  std::map<std::string, Input::LineDefinition>& defs = definitions["WALLN9PORO"];

  defs["NURBS9"] = Input::LineDefinition::Builder(defs_wall["NURBS9"])
                       .add_optional_named_double_vector("POROANISODIR1", 2)
                       .add_optional_named_double_vector("POROANISODIR2", 2)
                       .build();
}

int Discret::ELEMENTS::WallNurbs9PoroType::initialize(Core::FE::Discretization& dis)
{
  Discret::ELEMENTS::Wall1Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::nurbs9>*>(
        dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to Wall1_Poro* failed");
    actele->init_element();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  TRI 3 Element                                       |
 *----------------------------------------------------------------------*/

Discret::ELEMENTS::WallTri3PoroType Discret::ELEMENTS::WallTri3PoroType::instance_;

Discret::ELEMENTS::WallTri3PoroType& Discret::ELEMENTS::WallTri3PoroType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::ELEMENTS::WallTri3PoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::tri3>(-1, -1);
  object->unpack(buffer);
  return object;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::WallTri3PoroType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "WALLT3PORO")
  {
    Teuchos::RCP<Core::Elements::Element> ele =
        Teuchos::RCP(new Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::tri3>(id, owner));
    return ele;
  }
  return Teuchos::null;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::WallTri3PoroType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele =
      Teuchos::RCP(new Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::tri3>(id, owner));
  return ele;
}

void Discret::ELEMENTS::WallTri3PoroType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_wall;
  Wall1Type::setup_element_definition(definitions_wall);

  std::map<std::string, Input::LineDefinition>& defs_wall = definitions_wall["WALL"];

  std::map<std::string, Input::LineDefinition>& defs = definitions["WALLT3PORO"];

  defs["TRI3"] = Input::LineDefinition::Builder(defs_wall["TRI3"])
                     .add_optional_named_double_vector("POROANISODIR1", 2)
                     .add_optional_named_double_vector("POROANISODIR2", 2)
                     .add_optional_named_double_vector("POROANISONODALCOEFFS1", 3)
                     .add_optional_named_double_vector("POROANISONODALCOEFFS2", 3)
                     .build();
}

int Discret::ELEMENTS::WallTri3PoroType::initialize(Core::FE::Discretization& dis)
{
  Discret::ELEMENTS::Wall1Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele =
        dynamic_cast<Discret::ELEMENTS::Wall1Poro<Core::FE::CellType::tri3>*>(dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to Wall1_Poro* failed");
    actele->init_element();
  }
  return 0;
}

FOUR_C_NAMESPACE_CLOSE
