// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ale_ale2.hpp"

#include "4C_ale_ale2_nurbs.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_solid_3D_ele_nullspace.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::Elements::Ale2Type Discret::Elements::Ale2Type::instance_;

Discret::Elements::Ale2Type& Discret::Elements::Ale2Type::instance() { return instance_; }

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::Ale2Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Ale2* object = new Discret::Elements::Ale2(-1, -1);
  object->unpack(buffer);
  return object;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::Ale2Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele;

  if (eletype == "ALE2")
  {
    if (eledistype != "NURBS4" and eledistype != "NURBS9")
    {
      ele = std::make_shared<Discret::Elements::Ale2>(id, owner);
    }
  }

  return ele;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::Ale2Type::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::Ale2>(id, owner);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale2Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 2;
  dimns = 3;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::LinAlg::SerialDenseMatrix Discret::Elements::Ale2Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_null_space<2>(node.x(), x0);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale2Type::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["ALE2"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::quad4] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::quad8] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::quad9] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::tri3] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::tri6] = all_of({
      parameter<int>("MAT"),
  });
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::Ale2LineType::create(
    const int id, const int owner)
{
  // return Teuchos::rcp( new Ale2Line( id, owner ) );
  return nullptr;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Discret::Elements::Ale2::Ale2(int id, int owner) : Core::Elements::Element(id, owner) {}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Discret::Elements::Ale2::Ale2(const Discret::Elements::Ale2& old) : Core::Elements::Element(old) {}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Ale2::clone() const
{
  Discret::Elements::Ale2* newelement = new Discret::Elements::Ale2(*this);
  return newelement;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::Ale2::shape() const
{
  switch (num_node())
  {
    case 3:
      return Core::FE::CellType::tri3;
    case 4:
      return Core::FE::CellType::quad4;
    case 6:
      return Core::FE::CellType::tri6;
    case 8:
      return Core::FE::CellType::quad8;
    case 9:
      return Core::FE::CellType::quad9;
    default:
      FOUR_C_THROW("unexpected number of nodes {}", num_node());
      break;
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Element::pack(data);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale2::print(std::ostream& os) const
{
  os << "Ale2 ";
  Element::print(os);
  std::cout << std::endl;
  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Ale2::lines()
{
  return Core::Communication::element_boundary_factory<Ale2Line, Ale2>(
      Core::Communication::buildLines, *this);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Ale2::surfaces()
{
  return {Core::Utils::shared_ptr_from_ref(*this)};
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::FE::GaussRule2D Discret::Elements::Ale2::get_optimal_gaussrule(
    const Core::FE::CellType& distype)
{
  Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::undefined;
  switch (distype)
  {
    case Core::FE::CellType::quad4:
    case Core::FE::CellType::nurbs4:
      rule = Core::FE::GaussRule2D::quad_4point;
      break;
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
    case Core::FE::CellType::nurbs9:
      rule = Core::FE::GaussRule2D::quad_9point;
      break;
    case Core::FE::CellType::tri3:
      rule = Core::FE::GaussRule2D::tri_3point;
      break;
    case Core::FE::CellType::tri6:
      rule = Core::FE::GaussRule2D::tri_6point;
      break;
    default:
      FOUR_C_THROW("unknown number of nodes for gaussrule initialization");
      break;
  }
  return rule;
}

FOUR_C_NAMESPACE_CLOSE
