// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_bele_bele3.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_so3_nullspace.hpp"
#include "4C_utils_exceptions.hpp"

#include <sstream>

FOUR_C_NAMESPACE_OPEN


Discret::Elements::Bele3Type Discret::Elements::Bele3Type::instance_;


Discret::Elements::Bele3Type& Discret::Elements::Bele3Type::instance() { return instance_; }


Core::Communication::ParObject* Discret::Elements::Bele3Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Bele3* object = new Discret::Elements::Bele3(-1, -1);
  object->unpack(buffer);
  return object;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Bele3Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  // Search for "BELE3". If found, search for "_"
  // the number after "_" is numdof: so BELE3_4 is a BELE3 element
  // with numdof=4
  std::size_t pos = eletype.rfind("BELE3");
  if (pos != std::string::npos)
  {
    if (eletype.substr(pos + 5, 1) == "_")
    {
      std::istringstream is(eletype.substr(pos + 6, 1));

      int numdof = -1;
      is >> numdof;
      std::shared_ptr<Discret::Elements::Bele3> ele =
          std::make_shared<Discret::Elements::Bele3>(id, owner);
      ele->set_num_dof_per_node(numdof);
      return ele;
    }
    else
    {
      FOUR_C_THROW("ERROR: Found BELE3 element without specified number of dofs!");
    }
  }

  return nullptr;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Bele3Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Bele3>(id, owner);
  return ele;
}


void Discret::Elements::Bele3Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 3;
  dimns = 6;
  nv = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::Bele3Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_3d_null_space(node, x0);
}

void Discret::Elements::Bele3Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  auto& defs3 = definitions["BELE3_3"];

  using namespace Core::IO::InputSpecBuilders;

  defs3["TRI3"] = anonymous_group({
      entry<std::vector<int>>("TRI3", {.size = 3}),
      entry<int>("MAT", {.required = false}),
  });

  defs3["TRI6"] = anonymous_group({
      entry<std::vector<int>>("TRI6", {.size = 6}),
      entry<int>("MAT", {.required = false}),
  });

  defs3["QUAD4"] = anonymous_group({
      entry<std::vector<int>>("QUAD4", {.size = 4}),
      entry<int>("MAT", {.required = false}),
  });

  defs3["QUAD8"] = anonymous_group({
      entry<std::vector<int>>("QUAD8", {.size = 8}),
      entry<int>("MAT", {.required = false}),
  });

  defs3["QUAD9"] = anonymous_group({
      entry<std::vector<int>>("QUAD9", {.size = 9}),
      entry<int>("MAT", {.required = false}),
  });
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Bele3LineType::create(
    const int id, const int owner)
{
  // return Teuchos::rcp( new Bele3Line( id, owner ) );
  return nullptr;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::Bele3::Bele3(int id, int owner)
    : Core::Elements::Element(id, owner), numdofpernode_(-1)
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::Bele3::Bele3(const Discret::Elements::Bele3& old)
    : Core::Elements::Element(old), numdofpernode_(old.numdofpernode_)
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Bele3::clone() const
{
  Discret::Elements::Bele3* newelement = new Discret::Elements::Bele3(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::Bele3::shape() const
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
      FOUR_C_THROW("unexpected number of nodes %d", num_node());
      break;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Bele3::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Element::pack(data);
  // numdofpernode_
  add_to_pack(data, numdofpernode_);

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Bele3::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);
  // numdofpernode_
  extract_from_pack(buffer, numdofpernode_);
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Bele3::print(std::ostream& os) const
{
  os << "Bele3_" << numdofpernode_ << " " << Core::FE::cell_type_to_string(shape());
  Element::print(os);
  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                               gjb 05/08|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Bele3::lines()
{
  return Core::Communication::element_boundary_factory<Bele3Line, Bele3>(
      Core::Communication::buildLines, *this);
}


/*----------------------------------------------------------------------*
 |  get vector of Surfaces (length 1) (public)               gammi 04/07|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Bele3::surfaces()
{
  return {Core::Utils::shared_ptr_from_ref(*this)};
}


Core::FE::GaussRule2D Discret::Elements::Bele3::get_optimal_gaussrule() const
{
  Core::FE::GaussRule2D rule = Core::FE::GaussRule2D::undefined;
  switch (shape())
  {
    case Core::FE::CellType::quad4:
      rule = Core::FE::GaussRule2D::quad_4point;
      break;
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
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

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Discret::Elements::Bele3::read_element(const std::string& eletype, const std::string& distype,
    const Core::IO::InputParameterContainer& container)
{
  // check if material is defined
  int material = container.get_or("MAT", -1);
  if (material != -1)
  {
    set_material(0, Mat::factory(material));
  }
  return true;
}

FOUR_C_NAMESPACE_CLOSE
