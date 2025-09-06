// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ale_ale3.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_solid_3D_ele_nullspace.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::Elements::Ale3Type Discret::Elements::Ale3Type::instance_;

Discret::Elements::Ale3Type& Discret::Elements::Ale3Type::instance() { return instance_; }

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::Ale3Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Ale3* object = new Discret::Elements::Ale3(-1, -1);
  object->unpack(buffer);
  return object;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::Ale3Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele;

  if (eletype == "ALE3")
  {
    if (eledistype != "NURBS27")
    {
      ele = std::make_shared<Discret::Elements::Ale3>(id, owner);
    }
  }

  return ele;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::Ale3Type::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::Ale3>(id, owner);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale3Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 3;
  dimns = 6;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::LinAlg::SerialDenseMatrix Discret::Elements::Ale3Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_null_space<3>(node.x(), x0);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale3Type::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["ALE3"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::hex8] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::hex20] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::hex27] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::tet4] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::tet10] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::wedge6] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::wedge15] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::pyramid5] = all_of({
      parameter<int>("MAT"),
  });
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::Ale3SurfaceType::create(
    const int id, const int owner)
{
  // return Teuchos::rcp( new Ale3Surface( id, owner ) );
  return nullptr;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Discret::Elements::Ale3::Ale3(int id, int owner) : Core::Elements::Element(id, owner) {}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Discret::Elements::Ale3::Ale3(const Discret::Elements::Ale3& old) : Core::Elements::Element(old)
{
  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Ale3::clone() const
{
  Discret::Elements::Ale3* newelement = new Discret::Elements::Ale3(*this);
  return newelement;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::Ale3::shape() const
{
  switch (num_node())
  {
    case 4:
      return Core::FE::CellType::tet4;
    case 5:
      return Core::FE::CellType::pyramid5;
    case 6:
      return Core::FE::CellType::wedge6;
    case 8:
      return Core::FE::CellType::hex8;
    case 10:
      return Core::FE::CellType::tet10;
    case 20:
      return Core::FE::CellType::hex20;
    case 27:
      return Core::FE::CellType::hex27;
    default:
      FOUR_C_THROW("unexpected number of nodes {}", num_node());
      break;
  }
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale3::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Element::pack(data);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale3::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Ale3::print(std::ostream& os) const
{
  os << "Ale3 ";
  Element::print(os);
  std::cout << std::endl;
  // cout << data_;
  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Ale3::surfaces()
{
  return Core::Communication::element_boundary_factory<Ale3Surface, Ale3>(
      Core::Communication::buildSurfaces, *this);
}

FOUR_C_NAMESPACE_CLOSE
