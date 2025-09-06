// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_element2.hpp"

#include "4C_comm_pack_helpers.hpp"

FOUR_C_NAMESPACE_OPEN


Discret::Elements::ConstraintElement2Type Discret::Elements::ConstraintElement2Type::instance_;


Discret::Elements::ConstraintElement2Type& Discret::Elements::ConstraintElement2Type::instance()
{
  return instance_;
}


Core::Communication::ParObject* Discret::Elements::ConstraintElement2Type::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::ConstraintElement2* object = new Discret::Elements::ConstraintElement2(-1, -1);
  object->unpack(buffer);
  return object;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::ConstraintElement2Type::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "CONSTRELE2")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::ConstraintElement2>(id, owner);
    return ele;
  }
  return nullptr;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::ConstraintElement2Type::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::ConstraintElement2>(id, owner);
  return ele;
}


void Discret::Elements::ConstraintElement2Type::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::ConstraintElement2Type::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  Core::LinAlg::SerialDenseMatrix nullspace;
  FOUR_C_THROW("method ComputeNullSpace not implemented!");
  return nullspace;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::ConstraintElement2::ConstraintElement2(int id, int owner)
    : Core::Elements::Element(id, owner)
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::ConstraintElement2::ConstraintElement2(
    const Discret::Elements::ConstraintElement2& old)
    : Core::Elements::Element(old)
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::ConstraintElement2::clone() const
{
  Discret::Elements::ConstraintElement2* newelement =
      new Discret::Elements::ConstraintElement2(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::ConstraintElement2::pack(Core::Communication::PackBuffer& data) const
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
void Discret::Elements::ConstraintElement2::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);


  return;
}

FOUR_C_NAMESPACE_CLOSE
