// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_so3_sh8.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_so3_hex8.hpp"
#include "4C_so3_nullspace.hpp"
#include "4C_so3_utils.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


Discret::Elements::SoSh8Type Discret::Elements::SoSh8Type::instance_;

Discret::Elements::SoSh8Type& Discret::Elements::SoSh8Type::instance() { return instance_; }

Core::Communication::ParObject* Discret::Elements::SoSh8Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::SoSh8(-1, -1);
  object->unpack(buffer);
  return object;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::SoSh8Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == get_element_type_string())
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::SoSh8>(id, owner);
    return ele;
  }
  return nullptr;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::SoSh8Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::SoSh8>(id, owner);
  return ele;
}


void Discret::Elements::SoSh8Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = 3;
  dimns = 6;
  nv = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::SoSh8Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_3d_null_space(node, x0);
}

void Discret::Elements::SoSh8Type::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions[get_element_type_string()];

  using namespace Core::IO::InputSpecBuilders;

  defs["HEX8"] = anonymous_group({
      entry<std::vector<int>>("HEX8", {.size = 8}),
      entry<int>("MAT"),
      entry<std::string>("KINEM"),
      entry<std::string>("EAS"),
      entry<std::string>("ANS"),
      entry<std::string>("THICKDIR"),
      entry<std::vector<double>>("RAD", {.required = false, .size = 3}),
      entry<std::vector<double>>("AXI", {.required = false, .size = 3}),
      entry<std::vector<double>>("CIR", {.required = false, .size = 3}),
      entry<std::vector<double>>("FIBER1", {.required = false, .size = 3}),
      entry<std::vector<double>>("FIBER2", {.required = false, .size = 3}),
      entry<std::vector<double>>("FIBER3", {.required = false, .size = 3}),
      entry<double>("STRENGTH", {.required = false}),
      entry<double>("GROWTHTRIG", {.required = false}),
  });
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                              maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::SoSh8::SoSh8(int id, int owner)
    : Discret::Elements::SoHex8(id, owner),
      thickdir_(undefined),
      anstype_(ansnone),
      nodes_rearranged_(false),
      thickvec_(3, 0.0)
{
  std::shared_ptr<const Teuchos::ParameterList> params =
      Global::Problem::instance()->get_parameter_list();
  if (params != nullptr)
  {
    Discret::Elements::Utils::throw_error_fd_material_tangent(
        Global::Problem::instance()->structural_dynamic_params(), get_element_type_string());
  }

  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                         maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::SoSh8::SoSh8(const Discret::Elements::SoSh8& old)
    : Discret::Elements::SoHex8(old),
      thickdir_(old.thickdir_),
      anstype_(old.anstype_),
      nodes_rearranged_(old.nodes_rearranged_),
      thickvec_(old.thickvec_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Solid3 and return pointer to it (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::SoSh8::clone() const
{
  auto* newelement = new Discret::Elements::SoSh8(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoSh8::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class So_hex8 Element
  Discret::Elements::SoHex8::pack(data);
  // thickdir
  add_to_pack(data, thickdir_);
  add_to_pack(data, thickvec_);
  add_to_pack(data, anstype_);
  add_to_pack(data, nodes_rearranged_);

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoSh8::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class So_hex8 Element
  Discret::Elements::SoHex8::unpack(buffer);
  // thickdir
  extract_from_pack(buffer, thickdir_);
  extract_from_pack(buffer, thickvec_);
  extract_from_pack(buffer, anstype_);
  extract_from_pack(buffer, nodes_rearranged_);


  return;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                                maf 04/07|
 *----------------------------------------------------------------------*/
void Discret::Elements::SoSh8::print(std::ostream& os) const
{
  os << "So_sh8 ";
  Element::print(os);
  std::cout << std::endl;
  return;
}

FOUR_C_NAMESPACE_CLOSE
