// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ale_ale3_nurbs.hpp"

#include "4C_fem_general_node.hpp"
#include "4C_solid_3D_ele_nullspace.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::Elements::Nurbs::Ale3NurbsType Discret::Elements::Nurbs::Ale3NurbsType::instance_;

Discret::Elements::Nurbs::Ale3NurbsType& Discret::Elements::Nurbs::Ale3NurbsType::instance()
{
  return instance_;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::Nurbs::Ale3NurbsType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Nurbs::Ale3Nurbs* object = new Discret::Elements::Nurbs::Ale3Nurbs(-1, -1);
  object->unpack(buffer);
  return object;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::Nurbs::Ale3NurbsType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "ALE3")
  {
    if (eledistype == "NURBS8" || eledistype == "NURBS27")
    {
      return std::make_shared<Discret::Elements::Nurbs::Ale3Nurbs>(id, owner);
    }
  }
  return nullptr;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::Nurbs::Ale3NurbsType::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::Nurbs::Ale3Nurbs>(id, owner);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Nurbs::Ale3NurbsType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 3;
  dimns = 6;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::LinAlg::SerialDenseMatrix Discret::Elements::Nurbs::Ale3NurbsType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_null_space<3>(node.x(), x0);
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Discret::Elements::Nurbs::Ale3Nurbs::Ale3Nurbs(int id, int owner)
    : Discret::Elements::Ale3::Ale3(id, owner)
{
  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Discret::Elements::Nurbs::Ale3Nurbs::Ale3Nurbs(const Discret::Elements::Nurbs::Ale3Nurbs& old)
    : Discret::Elements::Ale3::Ale3(old)
{
  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Discret::Elements::Nurbs::Ale3Nurbs::print(std::ostream& os) const
{
  os << "Ale3Nurbs ";
  Element::print(os);
  return;
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::Nurbs::Ale3Nurbs::shape() const
{
  switch (num_node())
  {
    case 8:
      return Core::FE::CellType::nurbs8;
    case 27:
      return Core::FE::CellType::nurbs27;
    default:
      FOUR_C_THROW("unexpected number of nodes {}", num_node());
      break;
  }
}

FOUR_C_NAMESPACE_CLOSE
