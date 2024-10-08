/*-----------------------------------------------------------*/
/*! \file

\brief Implementation of enrichment-wall fluid elements.
In addition to that, it contains the interface between element call
and Gauss point loop (depending on the fluid implementation)
as well as some additional service routines (for the evaluation
of errors, turbulence statistics etc.)


\level 2

*/
/*-----------------------------------------------------------*/


#include "4C_fluid_ele_xwall.hpp"

#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fluid_ele_nullspace.hpp"
#include "4C_global_data.hpp"
#include "4C_io_linedefinition.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::ELEMENTS::FluidXWallType Discret::ELEMENTS::FluidXWallType::instance_;

Discret::ELEMENTS::FluidXWallType& Discret::ELEMENTS::FluidXWallType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::ELEMENTS::FluidXWallType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::ELEMENTS::FluidXWall* object = new Discret::ELEMENTS::FluidXWall(-1, -1);
  object->unpack(buffer);
  return object;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::FluidXWallType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "FLUIDXW")
  {
    return Teuchos::make_rcp<Discret::ELEMENTS::FluidXWall>(id, owner);
  }
  return Teuchos::null;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::FluidXWallType::create(
    const int id, const int owner)
{
  return Teuchos::make_rcp<Discret::ELEMENTS::FluidXWall>(id, owner);
}

void Discret::ELEMENTS::FluidXWallType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  // this is necessary here! Otherwise it would not be consistent with the non-enriched nodes
  // since we are assuming that all elements are equal during nullspace computation
  numdf = 4;
  dimns = numdf;
  nv = numdf - 1;
  np = 1;
}

Core::LinAlg::SerialDenseMatrix Discret::ELEMENTS::FluidXWallType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return FLD::compute_fluid_null_space(node, numdof, dimnsp);
}

void Discret::ELEMENTS::FluidXWallType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, Input::LineDefinition>& defsxwall = definitions["FLUIDXW"];

  defsxwall["HEX8"] = Input::LineDefinition::Builder()
                          .add_int_vector("HEX8", 8)
                          .add_named_int("MAT")
                          .add_named_string("NA")
                          .build();
  defsxwall["TET4"] = Input::LineDefinition::Builder()
                          .add_int_vector("TET4", 4)
                          .add_named_int("MAT")
                          .add_named_string("NA")
                          .build();
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            gammi 02/08|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::FluidXWall::FluidXWall(int id, int owner) : Fluid(id, owner) { return; }

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       gammi 02/08|
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::FluidXWall::FluidXWall(const Discret::ELEMENTS::FluidXWall& old) : Fluid(old)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Fluid and return pointer to it (public) |
 |                                                          gammi 02/08 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::ELEMENTS::FluidXWall::clone() const
{
  Discret::ELEMENTS::FluidXWall* newelement = new Discret::ELEMENTS::FluidXWall(*this);
  return newelement;
}



/*----------------------------------------------------------------------*
 |  get vector of lines              (public)                           |
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::FluidXWall::lines()
{
  return Core::Communication::get_element_lines<FluidXWallBoundary, FluidXWall>(*this);
}


/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                                     |
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::FluidXWall::surfaces()
{
  return Core::Communication::get_element_surfaces<FluidXWallBoundary, FluidXWall>(*this);
}

/*----------------------------------------------------------------------*
 |  print this element (public)                                         |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::FluidXWall::print(std::ostream& os) const
{
  os << "FluidXWall ";
  Element::print(os);
  return;
}

FOUR_C_NAMESPACE_CLOSE
