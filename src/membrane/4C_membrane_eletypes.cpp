// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_membrane_eletypes.hpp"

#include "4C_fem_general_node.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_membrane.hpp"
#include "4C_solid_3D_ele_nullspace.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  TRI 3 Element                                          fbraeu 06/16 |
 *----------------------------------------------------------------------*/
Discret::Elements::MembraneTri3Type Discret::Elements::MembraneTri3Type::instance_;

Discret::Elements::MembraneTri3Type& Discret::Elements::MembraneTri3Type::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::MembraneTri3Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Membrane<Core::FE::CellType::tri3>* object =
      new Discret::Elements::Membrane<Core::FE::CellType::tri3>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneTri3Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "MEMBRANE3" && eledistype == "TRI3")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Membrane<Core::FE::CellType::tri3>>(id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneTri3Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Membrane<Core::FE::CellType::tri3>>(id, owner);
  return ele;
}

void Discret::Elements::MembraneTri3Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 3;  // number of degrees of freedom per node
  dimns = 3;  // number of nullspace vectors
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::MembraneTri3Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_null_space<3>(node.x(), x0);
}

void Discret::Elements::MembraneTri3Type::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["MEMBRANE3"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::tri3] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("KINEM"),
      parameter<double>("THICK"),
      parameter<std::string>("STRESS_STRAIN"),
      parameter<std::optional<std::vector<double>>>("RAD", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("AXI", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("CIR", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER2", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER3", {.size = 3}),
  });
}

/*----------------------------------------------------------------------*
 |  TRI 6 Element                                          fbraeu 06/16 |
 *----------------------------------------------------------------------*/
Discret::Elements::MembraneTri6Type Discret::Elements::MembraneTri6Type::instance_;

Discret::Elements::MembraneTri6Type& Discret::Elements::MembraneTri6Type::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::MembraneTri6Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Membrane<Core::FE::CellType::tri6>* object =
      new Discret::Elements::Membrane<Core::FE::CellType::tri6>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneTri6Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "MEMBRANE6" && eledistype == "TRI6")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Membrane<Core::FE::CellType::tri6>>(id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneTri6Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Membrane<Core::FE::CellType::tri6>>(id, owner);
  return ele;
}

void Discret::Elements::MembraneTri6Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 3;  // number of degrees of freedom per node
  dimns = 3;  // number of nullspace vectors
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::MembraneTri6Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_null_space<2>(node.x(), x0);
}

void Discret::Elements::MembraneTri6Type::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["MEMBRANE6"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::tri6] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("KINEM"),
      parameter<double>("THICK"),
      parameter<std::string>("STRESS_STRAIN"),
      parameter<std::optional<std::vector<double>>>("RAD", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("AXI", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("CIR", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER2", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER3", {.size = 3}),
  });
}

/*----------------------------------------------------------------------*
 |  QUAD 4 Element                                         fbraeu 06/16 |
 *----------------------------------------------------------------------*/
Discret::Elements::MembraneQuad4Type Discret::Elements::MembraneQuad4Type::instance_;

Discret::Elements::MembraneQuad4Type& Discret::Elements::MembraneQuad4Type::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::MembraneQuad4Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Membrane<Core::FE::CellType::quad4>* object =
      new Discret::Elements::Membrane<Core::FE::CellType::quad4>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneQuad4Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "MEMBRANE4" && eledistype == "QUAD4")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Membrane<Core::FE::CellType::quad4>>(id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneQuad4Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Membrane<Core::FE::CellType::quad4>>(id, owner);
  return ele;
}

void Discret::Elements::MembraneQuad4Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 3;  // number of degrees of freedom per node
  dimns = 3;  // number of nullspace vectors
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::MembraneQuad4Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_null_space<2>(node.x(), x0);
}

void Discret::Elements::MembraneQuad4Type::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["MEMBRANE4"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::quad4] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("KINEM"),
      parameter<double>("THICK"),
      parameter<std::string>("STRESS_STRAIN"),
      parameter<std::optional<std::vector<double>>>("RAD", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("AXI", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("CIR", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER2", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER3", {.size = 3}),
  });
}

/*----------------------------------------------------------------------*
 |  QUAD 9 Element                                         fbraeu 06/16 |
 *----------------------------------------------------------------------*/
Discret::Elements::MembraneQuad9Type Discret::Elements::MembraneQuad9Type::instance_;

Discret::Elements::MembraneQuad9Type& Discret::Elements::MembraneQuad9Type::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::MembraneQuad9Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Membrane<Core::FE::CellType::quad9>* object =
      new Discret::Elements::Membrane<Core::FE::CellType::quad9>(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneQuad9Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "MEMBRANE9" && eledistype == "QUAD9")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Membrane<Core::FE::CellType::quad9>>(id, owner);
    return ele;
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::MembraneQuad9Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Membrane<Core::FE::CellType::quad9>>(id, owner);
  return ele;
}

void Discret::Elements::MembraneQuad9Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 3;  // number of degrees of freedom per node
  dimns = 3;  // number of nullspace vectors
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::MembraneQuad9Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_null_space<2>(node.x(), x0);
}

void Discret::Elements::MembraneQuad9Type::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["MEMBRANE9"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::quad9] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("KINEM"),
      parameter<double>("THICK"),
      parameter<std::string>("STRESS_STRAIN"),
      parameter<std::optional<std::vector<double>>>("RAD", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("AXI", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("CIR", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER1", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER2", {.size = 3}),
      parameter<std::optional<std::vector<double>>>("FIBER3", {.size = 3}),
  });
}

FOUR_C_NAMESPACE_CLOSE
