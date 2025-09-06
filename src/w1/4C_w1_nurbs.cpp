// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_w1_nurbs.hpp"

#include "4C_comm_utils_factory.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_solid_3D_ele_nullspace.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::Elements::Nurbs::Wall1NurbsType Discret::Elements::Nurbs::Wall1NurbsType::instance_;

Discret::Elements::Nurbs::Wall1NurbsType& Discret::Elements::Nurbs::Wall1NurbsType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::Nurbs::Wall1NurbsType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Nurbs::Wall1Nurbs* object = new Discret::Elements::Nurbs::Wall1Nurbs(-1, -1);
  object->unpack(buffer);
  return object;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Nurbs::Wall1NurbsType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "WALLNURBS")
  {
    if (eledistype == "NURBS4" || eledistype == "NURBS9")
    {
      return std::make_shared<Discret::Elements::Nurbs::Wall1Nurbs>(id, owner);
    }
  }
  return nullptr;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::Nurbs::Wall1NurbsType::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::Nurbs::Wall1Nurbs>(id, owner);
}

void Discret::Elements::Nurbs::Wall1NurbsType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 2;
  dimns = 3;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::Nurbs::Wall1NurbsType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return compute_solid_null_space<2>(node.x(), x0);
}

void Discret::Elements::Nurbs::Wall1NurbsType::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["WALLNURBS"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::nurbs4] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("KINEM"),
      parameter<std::string>("EAS"),
      parameter<double>("THICK"),
      parameter<std::string>("STRESS_STRAIN"),
      parameter<std::vector<int>>("GP", {.size = 2}),
  });

  defs[Core::FE::CellType::nurbs9] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("KINEM"),
      parameter<std::string>("EAS"),
      parameter<double>("THICK"),
      parameter<std::string>("STRESS_STRAIN"),
      parameter<std::vector<int>>("GP", {.size = 2}),
  });
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            gammi 02/09|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::Nurbs::Wall1Nurbs::Wall1Nurbs(int id, int owner)
    : Discret::Elements::Wall1::Wall1(id, owner)
{
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       gammi 02/09|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::Nurbs::Wall1Nurbs::Wall1Nurbs(const Discret::Elements::Nurbs::Wall1Nurbs& old)
    : Discret::Elements::Wall1::Wall1(old)
{
}



/*----------------------------------------------------------------------*
 |  Deep copy this instance of Wall1 and return pointer to it (public) |
 |                                                          gammi 05/09|
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Nurbs::Wall1Nurbs::clone() const
{
  Discret::Elements::Nurbs::Wall1Nurbs* newelement =
      new Discret::Elements::Nurbs::Wall1Nurbs(*this);
  return newelement;
}


/*----------------------------------------------------------------------*
 |  print this element (public)                              gammi 02/09|
 *----------------------------------------------------------------------*/
void Discret::Elements::Nurbs::Wall1Nurbs::print(std::ostream& os) const
{
  os << "Wall1Nurbs ";
  Element::print(os);
  return;
}


/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                          gammi 02/09 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::Nurbs::Wall1Nurbs::shape() const
{
  switch (num_node())
  {
    case 4:
      return Core::FE::CellType::nurbs4;
    case 9:
      return Core::FE::CellType::nurbs9;
    default:
      FOUR_C_THROW("unexpected number of nodes {}", num_node());
  }
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                             gammi 05/09|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Nurbs::Wall1Nurbs::lines()
{
  return Core::Communication::element_boundary_factory<Wall1Line, Wall1>(
      Core::Communication::buildLines, *this);
}

/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                          gammi 05/09|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::Nurbs::Wall1Nurbs::surfaces()
{
  return {Core::Utils::shared_ptr_from_ref(*this)};
}

FOUR_C_NAMESPACE_CLOSE
