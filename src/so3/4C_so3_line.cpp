// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_so3_line.hpp"

#include "4C_linalg_serialdensematrix.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


Discret::ELEMENTS::StructuralLineType Discret::ELEMENTS::StructuralLineType::instance_;

Discret::ELEMENTS::StructuralLineType& Discret::ELEMENTS::StructuralLineType::instance()
{
  return instance_;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::StructuralLineType::create(
    const int id, const int owner)
{
  // return Teuchos::rcp( new StructuralLine( id, owner ) );
  return Teuchos::null;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                              gee 04/08|
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::StructuralLine::StructuralLine(int id, int owner, int nnode, const int* nodeids,
    Core::Nodes::Node** nodes, Core::Elements::Element* parent, const int lline)
    : Core::Elements::FaceElement(id, owner)
{
  set_node_ids(nnode, nodeids);
  build_nodal_pointers(nodes);
  set_parent_master_element(parent, lline);
  // type of gaussian integration
  switch (shape())
  {
    case Core::FE::CellType::line2:
      gaussrule_ = Core::FE::GaussRule1D::line_2point;
      break;
    case Core::FE::CellType::line3:
      gaussrule_ = Core::FE::GaussRule1D::line_3point;
      break;
    default:
      FOUR_C_THROW("shape type unknown!\n");
  }
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                         gee 04/08|
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::StructuralLine::StructuralLine(const Discret::ELEMENTS::StructuralLine& old)
    : Core::Elements::FaceElement(old), gaussrule_(old.gaussrule_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it               gee 04/08|
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::ELEMENTS::StructuralLine::clone() const
{
  auto* newelement = new Discret::ELEMENTS::StructuralLine(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             gee 04/08|
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::ELEMENTS::StructuralLine::shape() const
{
  switch (num_node())
  {
    case 2:
      return Core::FE::CellType::line2;
    case 3:
      return Core::FE::CellType::line3;
    default:
      FOUR_C_THROW("unexpected number of nodes %d", num_node());
  }
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  gee 04/08|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::StructuralLine::pack(Core::Communication::PackBuffer& data) const
{
  FOUR_C_THROW("StructuralLine element does not support communication");
  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                gee 04/08|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::StructuralLine::unpack(Core::Communication::UnpackBuffer& buffer)
{
  FOUR_C_THROW("StructuralLine element does not support communication");
  return;
}


/*----------------------------------------------------------------------*
 |  print this element (public)                               gee 04/08|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::StructuralLine::print(std::ostream& os) const
{
  os << "StructuralLine ";
  Element::print(os);
  return;
}

FOUR_C_NAMESPACE_CLOSE
