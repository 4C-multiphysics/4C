// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_bele_vele3.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::Elements::Vele3Type Discret::Elements::Vele3Type::instance_;

Discret::Elements::Vele3Type& Discret::Elements::Vele3Type::instance() { return instance_; }

Core::Communication::ParObject* Discret::Elements::Vele3Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Vele3* object = new Discret::Elements::Vele3(-1, -1);
  object->unpack(buffer);
  return object;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Vele3Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "VELE3")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Vele3>(id, owner);
    return ele;
  }
  return nullptr;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Vele3Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Vele3>(id, owner);
  return ele;
}


void Discret::Elements::Vele3Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::Vele3Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  Core::LinAlg::SerialDenseMatrix nullspace;
  FOUR_C_THROW("method ComputeNullSpace not implemented for element type vele3!");
  return nullspace;
}

void Discret::Elements::Vele3Type::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["VELE3"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::hex8] = all_of({});
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::Vele3SurfaceType::create(
    const int id, const int owner)
{
  // return Teuchos::rcp( new Vele3Surface( id, owner ) );
  return nullptr;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Vele3LineType::create(
    const int id, const int owner)
{
  // return Teuchos::rcp( new Vele3Line( id, owner ) );
  return nullptr;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::Vele3::Vele3(int id, int owner) : Core::Elements::Element(id, owner) { return; }


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::Vele3::Vele3(const Discret::Elements::Vele3& old) : Core::Elements::Element(old)
{
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Vele3::clone() const
{
  Discret::Elements::Vele3* newelement = new Discret::Elements::Vele3(*this);
  return newelement;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::Vele3::shape() const
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
    case 15:
      return Core::FE::CellType::wedge15;
    case 20:
      return Core::FE::CellType::hex20;
    case 27:
      return Core::FE::CellType::hex27;
    default:
      FOUR_C_THROW("unexpected number of nodes {}", num_node());
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Vele3::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Element::pack(data);

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Vele3::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);



  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Vele3::print(std::ostream& os) const
{
  os << "Vele3 " << Core::FE::cell_type_to_string(shape());
  Element::print(os);
  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                               gjb 05/08|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Vele3::lines()
{
  return Core::Communication::element_boundary_factory<Vele3Line, Vele3>(
      Core::Communication::buildLines, *this);
}


/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                            gjb 05/08|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Vele3::surfaces()
{
  return Core::Communication::element_boundary_factory<Vele3Surface, Vele3>(
      Core::Communication::buildSurfaces, *this);
}



/*----------------------------------------------------------------------*
 |  get optimal gauss rule (public)                          u.may 05/09|
 *----------------------------------------------------------------------*/
Core::FE::GaussRule3D Discret::Elements::Vele3::get_optimal_gaussrule(
    const Core::FE::CellType& distype) const
{
  Core::FE::GaussRule3D rule = Core::FE::GaussRule3D::undefined;
  switch (distype)
  {
    case Core::FE::CellType::hex8:
      rule = Core::FE::GaussRule3D::hex_8point;
      break;
    case Core::FE::CellType::hex20:
    case Core::FE::CellType::hex27:
      rule = Core::FE::GaussRule3D::hex_27point;
      break;
    case Core::FE::CellType::tet4:
      rule = Core::FE::GaussRule3D::tet_4point;
      break;
    case Core::FE::CellType::tet10:
      rule = Core::FE::GaussRule3D::tet_10point;
      break;
    default:
      FOUR_C_THROW("unknown number of nodes for gaussrule initialization");
  }
  return rule;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Discret::Elements::Vele3::read_element(const std::string& eletype, const std::string& distype,
    const Core::IO::InputParameterContainer& container)
{
  return true;
}

FOUR_C_NAMESPACE_CLOSE
