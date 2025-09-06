// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_w1_poro_p1_eletypes.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fluid_ele_nullspace.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_w1_poro_p1.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  QUAD 4 Element                                                      |
 *----------------------------------------------------------------------*/

Discret::Elements::WallQuad4PoroP1Type Discret::Elements::WallQuad4PoroP1Type::instance_;

Discret::Elements::WallQuad4PoroP1Type& Discret::Elements::WallQuad4PoroP1Type::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::WallQuad4PoroP1Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::Wall1PoroP1<Core::FE::CellType::quad4>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::WallQuad4PoroP1Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "WALLQ4POROP1")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Wall1PoroP1<Core::FE::CellType::quad4>>(id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::WallQuad4PoroP1Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Wall1PoroP1<Core::FE::CellType::quad4>>(id, owner);
  return ele;
}

void Discret::Elements::WallQuad4PoroP1Type::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>> definitions_wallporo;
  WallQuad4PoroType::setup_element_definition(definitions_wallporo);

  auto& defs_wallporo = definitions_wallporo["WALLQ4PORO"];

  auto& defs = definitions["WALLQ4POROP1"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::quad4] = defs_wallporo[Core::FE::CellType::quad4];
}

void Discret::Elements::WallQuad4PoroP1Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 3;
  dimns = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::WallQuad4PoroP1Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return FLD::compute_fluid_null_space(node, numdof, dimnsp);
}

int Discret::Elements::WallQuad4PoroP1Type::initialize(Core::FE::Discretization& dis)
{
  Discret::Elements::Wall1Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    Discret::Elements::Wall1PoroP1<Core::FE::CellType::quad4>* actele =
        dynamic_cast<Discret::Elements::Wall1PoroP1<Core::FE::CellType::quad4>*>(
            dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to Wall1_PoroP1* failed");
    actele->init_element();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  QUAD 9 Element                                                      |
 *----------------------------------------------------------------------*/

Discret::Elements::WallQuad9PoroP1Type Discret::Elements::WallQuad9PoroP1Type::instance_;

Discret::Elements::WallQuad9PoroP1Type& Discret::Elements::WallQuad9PoroP1Type::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::WallQuad9PoroP1Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::Wall1PoroP1<Core::FE::CellType::quad9>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::WallQuad9PoroP1Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "WALLQ9POROP1")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Wall1PoroP1<Core::FE::CellType::quad9>>(id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::WallQuad9PoroP1Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Wall1PoroP1<Core::FE::CellType::quad9>>(id, owner);
  return ele;
}

void Discret::Elements::WallQuad9PoroP1Type::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>> definitions_wallporo;
  WallQuad9PoroType::setup_element_definition(definitions_wallporo);

  auto& defs_wallporo = definitions_wallporo["WALLQ9PORO"];

  auto& defs = definitions["WALLQ9POROP1"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::quad9] = defs_wallporo[Core::FE::CellType::quad9];
}

void Discret::Elements::WallQuad9PoroP1Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 3;
  dimns = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::WallQuad9PoroP1Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return FLD::compute_fluid_null_space(node, numdof, dimnsp);
}

int Discret::Elements::WallQuad9PoroP1Type::initialize(Core::FE::Discretization& dis)
{
  Discret::Elements::Wall1Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    Discret::Elements::Wall1PoroP1<Core::FE::CellType::quad9>* actele =
        dynamic_cast<Discret::Elements::Wall1PoroP1<Core::FE::CellType::quad9>*>(
            dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to Wall1_PoroP1* failed");
    actele->init_element();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  TRI 3 Element                                                       |
 *----------------------------------------------------------------------*/

Discret::Elements::WallTri3PoroP1Type Discret::Elements::WallTri3PoroP1Type::instance_;

Discret::Elements::WallTri3PoroP1Type& Discret::Elements::WallTri3PoroP1Type::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::WallTri3PoroP1Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Wall1PoroP1<Core::FE::CellType::tri3>* object =
      new Discret::Elements::Wall1PoroP1<Core::FE::CellType::tri3>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::WallTri3PoroP1Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "WALLT3POROP1")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Wall1PoroP1<Core::FE::CellType::tri3>>(id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::WallTri3PoroP1Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Wall1PoroP1<Core::FE::CellType::tri3>>(id, owner);
  return ele;
}

void Discret::Elements::WallTri3PoroP1Type::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>> definitions_wallporo;
  WallTri3PoroType::setup_element_definition(definitions_wallporo);

  auto& defs_wallporo = definitions_wallporo["WALLT3PORO"];

  auto& defs = definitions["WALLT3POROP1"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::tri3] = defs_wallporo[Core::FE::CellType::tri3];
}

void Discret::Elements::WallTri3PoroP1Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 3;
  dimns = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::WallTri3PoroP1Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return FLD::compute_fluid_null_space(node, numdof, dimnsp);
}

int Discret::Elements::WallTri3PoroP1Type::initialize(Core::FE::Discretization& dis)
{
  Discret::Elements::Wall1Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    Discret::Elements::Wall1PoroP1<Core::FE::CellType::tri3>* actele =
        dynamic_cast<Discret::Elements::Wall1PoroP1<Core::FE::CellType::tri3>*>(
            dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to Wall1_PoroP1* failed");
    actele->init_element();
  }
  return 0;
}

FOUR_C_NAMESPACE_CLOSE
