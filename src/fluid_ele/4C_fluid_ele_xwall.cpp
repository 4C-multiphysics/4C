// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele_xwall.hpp"

#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fluid_ele_nullspace.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_spec_builders.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::Elements::FluidXWallType Discret::Elements::FluidXWallType::instance_;

Discret::Elements::FluidXWallType& Discret::Elements::FluidXWallType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::FluidXWallType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::FluidXWall* object = new Discret::Elements::FluidXWall(-1, -1);
  object->unpack(buffer);
  return object;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::FluidXWallType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "FLUIDXW")
  {
    return std::make_shared<Discret::Elements::FluidXWall>(id, owner);
  }
  return nullptr;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::FluidXWallType::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::FluidXWall>(id, owner);
}

void Discret::Elements::FluidXWallType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  // this is necessary here! Otherwise it would not be consistent with the non-enriched nodes
  // since we are assuming that all elements are equal during nullspace computation
  numdf = 4;
  dimns = numdf;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::FluidXWallType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return FLD::compute_fluid_null_space(node, numdof, dimnsp);
}

void Discret::Elements::FluidXWallType::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defsxwall = definitions["FLUIDXW"];

  using namespace Core::IO::InputSpecBuilders;

  defsxwall[Core::FE::CellType::hex8] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("NA"),
  });
  defsxwall[Core::FE::CellType::tet4] = all_of({
      parameter<int>("MAT"),
      parameter<std::string>("NA"),
  });
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            gammi 02/08|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::FluidXWall::FluidXWall(int id, int owner) : Fluid(id, owner) { return; }

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       gammi 02/08|
 *----------------------------------------------------------------------*/
Discret::Elements::FluidXWall::FluidXWall(const Discret::Elements::FluidXWall& old) : Fluid(old)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Fluid and return pointer to it (public) |
 |                                                          gammi 02/08 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::FluidXWall::clone() const
{
  Discret::Elements::FluidXWall* newelement = new Discret::Elements::FluidXWall(*this);
  return newelement;
}



/*----------------------------------------------------------------------*
 |  get vector of lines              (public)                           |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::FluidXWall::lines()
{
  return Core::Communication::get_element_lines<FluidXWallBoundary, FluidXWall>(*this);
}


/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                                     |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::FluidXWall::surfaces()
{
  return Core::Communication::get_element_surfaces<FluidXWallBoundary, FluidXWall>(*this);
}

/*----------------------------------------------------------------------*
 |  print this element (public)                                         |
 *----------------------------------------------------------------------*/
void Discret::Elements::FluidXWall::print(std::ostream& os) const
{
  os << "FluidXWall ";
  Element::print(os);
  return;
}

FOUR_C_NAMESPACE_CLOSE
