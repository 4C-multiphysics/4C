// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_lubrication_ele.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_fluid_ele_nullspace.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_material_factory.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::Elements::LubricationType Discret::Elements::LubricationType::instance_;

Discret::Elements::LubricationType& Discret::Elements::LubricationType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::Elements::LubricationType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Lubrication* object = new Discret::Elements::Lubrication(-1, -1);
  object->unpack(buffer);
  return object;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::LubricationType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "LUBRICATION")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Lubrication>(id, owner);
    return ele;
  }
  return nullptr;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::LubricationType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Lubrication>(id, owner);
  return ele;
}


void Discret::Elements::LubricationType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = dwele->num_dof_per_node(*(dwele->nodes()[0]));
  dimns = numdf;
}

Core::LinAlg::SerialDenseMatrix Discret::Elements::LubricationType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  return FLD::compute_fluid_null_space(node, numdof, dimnsp);
}

void Discret::Elements::LubricationType::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["LUBRICATION"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::quad4] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::quad8] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::quad9] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::tri3] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::tri6] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::line2] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::line3] = all_of({
      parameter<int>("MAT"),
  });
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/


Discret::Elements::LubricationBoundaryType Discret::Elements::LubricationBoundaryType::instance_;

Discret::Elements::LubricationBoundaryType& Discret::Elements::LubricationBoundaryType::instance()
{
  return instance_;
}

std::shared_ptr<Core::Elements::Element> Discret::Elements::LubricationBoundaryType::create(
    const int id, const int owner)
{
  // return Teuchos::rcp( new LubricationBoundary( id, owner ) );
  return nullptr;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                           wirtz 10/15 |
 *----------------------------------------------------------------------*/
Discret::Elements::Lubrication::Lubrication(int id, int owner)
    : Core::Elements::Element(id, owner), distype_(Core::FE::CellType::dis_none)
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                      wirtz 10/15 |
 *----------------------------------------------------------------------*/
Discret::Elements::Lubrication::Lubrication(const Discret::Elements::Lubrication& old)
    : Core::Elements::Element(old), distype_(old.distype_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Lubrication and return pointer to it        |
 |                                                 (public) wirtz 10/15 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Lubrication::clone() const
{
  Discret::Elements::Lubrication* newelement = new Discret::Elements::Lubrication(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Return the shape of a Lubrication element                     (public) |
 |                                                          wirtz 10/15 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::Lubrication::shape() const { return distype_; }

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                          wirtz 10/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::Lubrication::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // add base class Element
  Element::pack(data);

  // add internal data
  add_to_pack(data, distype_);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                          wirtz 10/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::Lubrication::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);

  // extract internal data
  extract_from_pack(buffer, distype_);



  return;
}

/*----------------------------------------------------------------------*
 |  Return number of lines of this element (public)         wirtz 10/15 |
 *----------------------------------------------------------------------*/
int Discret::Elements::Lubrication::num_line() const
{
  return Core::FE::get_number_of_element_lines(distype_);
}


/*----------------------------------------------------------------------*
 |  Return number of surfaces of this element (public)      wirtz 10/15 |
 *----------------------------------------------------------------------*/
int Discret::Elements::Lubrication::num_surface() const
{
  return Core::FE::get_number_of_element_surfaces(distype_);
}


/*----------------------------------------------------------------------*
 | Return number of volumes of this element (public)        wirtz 10/15 |
 *----------------------------------------------------------------------*/
int Discret::Elements::Lubrication::num_volume() const
{
  return Core::FE::get_number_of_element_volumes(distype_);
}



/*----------------------------------------------------------------------*
 |  print this element (public)                             wirtz 10/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::Lubrication::print(std::ostream& os) const
{
  os << "Lubrication element";
  Element::print(os);
  std::cout << std::endl;
  std::cout << "DiscretizationType:  " << Core::FE::cell_type_to_string(distype_) << std::endl;

  return;
}


/*----------------------------------------------------------------------*
 |  get vector of lines            (public)                 wirtz 10/15 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Lubrication::lines()
{
  return Core::Communication::get_element_lines<LubricationBoundary, Lubrication>(*this);
}


/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                         wirtz 10/15 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Lubrication::surfaces()
{
  return Core::Communication::get_element_surfaces<LubricationBoundary, Lubrication>(*this);
}

/*----------------------------------------------------------------------*
 | read element input                                       wirtz 10/15 |
 *----------------------------------------------------------------------*/
bool Discret::Elements::Lubrication::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container)
{
  // read number of material model
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));

  // set discretization type
  set_dis_type(Core::FE::string_to_cell_type(distype));

  return true;
}


//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================


/*----------------------------------------------------------------------*
 |  ctor (public)                                           wirtz 10/15 |
 *----------------------------------------------------------------------*/
Discret::Elements::LubricationBoundary::LubricationBoundary(int id, int owner, int nnode,
    const int* nodeids, Core::Nodes::Node** nodes, Discret::Elements::Lubrication* parent,
    const int lsurface)
    : Core::Elements::FaceElement(id, owner)
{
  set_node_ids(nnode, nodeids);
  build_nodal_pointers(nodes);
  set_parent_master_element(parent, lsurface);
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                      wirtz 10/15 |
 *----------------------------------------------------------------------*/
Discret::Elements::LubricationBoundary::LubricationBoundary(
    const Discret::Elements::LubricationBoundary& old)
    : Core::Elements::FaceElement(old)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it   (public) wirtz 10/15 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::LubricationBoundary::clone() const
{
  Discret::Elements::LubricationBoundary* newelement =
      new Discret::Elements::LubricationBoundary(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Return shape of this element                   (public) wirtz 10/15 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::LubricationBoundary::shape() const
{
  return Core::FE::get_shape_of_boundary_element(num_node(), parent_element()->shape());
}

/*----------------------------------------------------------------------*
 |  Pack data (public)                                      wirtz 10/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::LubricationBoundary::pack(Core::Communication::PackBuffer& data) const
{
  FOUR_C_THROW("This LubricationBoundary element does not support communication");

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data (public)                                    wirtz 10/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::LubricationBoundary::unpack(Core::Communication::UnpackBuffer& buffer)
{
  FOUR_C_THROW("This LubricationBoundary element does not support communication");
  return;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                             wirtz 10/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::LubricationBoundary::print(std::ostream& os) const
{
  os << "LubricationBoundary element";
  Element::print(os);
  std::cout << std::endl;
  std::cout << "DiscretizationType:  " << Core::FE::cell_type_to_string(shape()) << std::endl;
  std::cout << std::endl;
  return;
}

/*----------------------------------------------------------------------*
 | Return number of lines of boundary element (public)      wirtz 10/15 |
 *----------------------------------------------------------------------*/
int Discret::Elements::LubricationBoundary::num_line() const
{
  return Core::FE::get_number_of_element_lines(shape());
}

/*----------------------------------------------------------------------*
 |  Return number of surfaces of boundary element (public)  wirtz 10/15 |
 *----------------------------------------------------------------------*/
int Discret::Elements::LubricationBoundary::num_surface() const
{
  return Core::FE::get_number_of_element_surfaces(shape());
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                            wirtz 10/15 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::LubricationBoundary::lines()
{
  FOUR_C_THROW("Lines of LubricationBoundary not implemented");
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                            wirtz 10/15 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::LubricationBoundary::surfaces()
{
  FOUR_C_THROW("Surfaces of LubricationBoundary not implemented");
}

FOUR_C_NAMESPACE_CLOSE
