// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_torsion3.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_solid_3D_ele_nullspace.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

Discret::Elements::Torsion3Type Discret::Elements::Torsion3Type::instance_;

Discret::Elements::Torsion3Type& Discret::Elements::Torsion3Type::instance() { return instance_; }

Core::Communication::ParObject* Discret::Elements::Torsion3Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Torsion3* object = new Discret::Elements::Torsion3(-1, -1);
  object->unpack(buffer);
  return object;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Torsion3Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "TORSION3")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Torsion3>(id, owner);
    return ele;
  }
  return nullptr;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Torsion3Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Torsion3>(id, owner);
  return ele;
}


void Discret::Elements::Torsion3Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 3;
  dimns = 6;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::Torsion3Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_null_space<3>(node.x(), x0);
}

void Discret::Elements::Torsion3Type::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["TORSION3"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::line3] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("BENDINGPOTENTIAL"),
  });
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            cyron 02/10|
 *----------------------------------------------------------------------*/
Discret::Elements::Torsion3::Torsion3(int id, int owner) : Core::Elements::Element(id, owner)
{
  return;
}
/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       cyron 02/10|
 *----------------------------------------------------------------------*/
Discret::Elements::Torsion3::Torsion3(const Discret::Elements::Torsion3& old)
    : Core::Elements::Element(old)
{
}

/*----------------------------------------------------------------------*
 | Deep copy this instance of Torsion3 and return pointer to it (public)|
 |                                                           cyron 02/10|
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Torsion3::clone() const
{
  Discret::Elements::Torsion3* newelement = new Discret::Elements::Torsion3(*this);
  return newelement;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                              cyron 02/10|
 *----------------------------------------------------------------------*/
void Discret::Elements::Torsion3::print(std::ostream& os) const { return; }


/*----------------------------------------------------------------------*
 |(public)                                                   cyron 02/10|
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::Torsion3::shape() const { return Core::FE::CellType::line3; }


/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                           cyron 02/10|
 *----------------------------------------------------------------------*/
void Discret::Elements::Torsion3::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  Element::pack(data);
  add_to_pack(data, bendingpotential_);
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                           cyron 02/10|
 *----------------------------------------------------------------------*/
void Discret::Elements::Torsion3::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  Element::unpack(buffer);
  extract_from_pack(buffer, bendingpotential_);
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                             cyron 02/10|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Torsion3::lines()
{
  return {Core::Utils::shared_ptr_from_ref(*this)};
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Torsion3::set_params_interface_ptr(const Teuchos::ParameterList& p)
{
  if (p.isParameter("interface"))
    interface_ptr_ = std::dynamic_pointer_cast<Solid::Elements::ParamsInterface>(
        p.get<std::shared_ptr<Core::Elements::ParamsInterface>>("interface"));
  else
    interface_ptr_ = nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::ParamsInterface> Discret::Elements::Torsion3::params_interface_ptr()
{
  return interface_ptr_;
}

FOUR_C_NAMESPACE_CLOSE
