// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_so3_poro_p1_eletypes.hpp"

#include "4C_fluid_ele_nullspace.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_so3_poro_p1.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  HEX 8 Element                                                       |
 *----------------------------------------------------------------------*/
Discret::Elements::SoHex8PoroP1Type Discret::Elements::SoHex8PoroP1Type::instance_;

Discret::Elements::SoHex8PoroP1Type& Discret::Elements::SoHex8PoroP1Type::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::SoHex8PoroP1Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object =
      new Discret::Elements::So3PoroP1<Discret::Elements::SoHex8, Core::FE::CellType::hex8>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoHex8PoroP1Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
        Discret::Elements::So3PoroP1<Discret::Elements::SoHex8, Core::FE::CellType::hex8>>(

        id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoHex8PoroP1Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
      Discret::Elements::So3PoroP1<Discret::Elements::SoHex8, Core::FE::CellType::hex8>>(

      id, owner);
  return ele;
}

void Discret::Elements::SoHex8PoroP1Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_hex8poro;
  SoHex8PoroType::setup_element_definition(definitions_hex8poro);

  auto& defs_hex8 = definitions_hex8poro["SOLIDH8PORO"];

  auto& defs = definitions[get_element_type_string()];

  defs["HEX8"] = defs_hex8["HEX8"];
}

void Discret::Elements::SoHex8PoroP1Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 4;
  dimns = 4;
  nv = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::SoHex8PoroP1Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return FLD::compute_fluid_null_space(node, numdof, dimnsp);
}

int Discret::Elements::SoHex8PoroP1Type::initialize(Core::FE::Discretization& dis)
{
  SoHex8Type::initialize(dis);
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<
        Discret::Elements::So3PoroP1<Discret::Elements::SoHex8, Core::FE::CellType::hex8>*>(
        dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So3_Poro_P1* failed");
    actele->So3PoroP1<Discret::Elements::SoHex8, Core::FE::CellType::hex8>::init_element();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  TET 4 Element                                                       |
 *----------------------------------------------------------------------*/

Discret::Elements::SoTet4PoroP1Type Discret::Elements::SoTet4PoroP1Type::instance_;

Discret::Elements::SoTet4PoroP1Type& Discret::Elements::SoTet4PoroP1Type::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::SoTet4PoroP1Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object =
      new Discret::Elements::So3PoroP1<Discret::Elements::SoTet4, Core::FE::CellType::tet4>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoTet4PoroP1Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
        Discret::Elements::So3PoroP1<Discret::Elements::SoTet4, Core::FE::CellType::tet4>>(

        id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::SoTet4PoroP1Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele = std::make_shared<
      Discret::Elements::So3PoroP1<Discret::Elements::SoTet4, Core::FE::CellType::tet4>>(

      id, owner);
  return ele;
}

void Discret::Elements::SoTet4PoroP1Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_tet4;
  SoTet4PoroType::setup_element_definition(definitions_tet4);

  auto& defs_tet4 = definitions_tet4["SOLIDT4PORO"];

  auto& defs = definitions[get_element_type_string()];

  defs["TET4"] = defs_tet4["TET4"];
}

int Discret::Elements::SoTet4PoroP1Type::initialize(Core::FE::Discretization& dis)
{
  SoTet4PoroType::initialize(dis);
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

void Discret::Elements::SoTet4PoroP1Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 4;
  dimns = 4;
  nv = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::SoTet4PoroP1Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return FLD::compute_fluid_null_space(node, numdof, dimnsp);
}

FOUR_C_NAMESPACE_CLOSE
