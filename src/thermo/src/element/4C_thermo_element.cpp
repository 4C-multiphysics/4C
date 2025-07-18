// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_thermo_element.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_thermostvenantkirchhoff.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Thermo::ElementType Thermo::ElementType::instance_;

Thermo::ElementType& Thermo::ElementType::instance() { return instance_; }

/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 09/09 |
 | is called in ElementRegisterType                                     |
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Thermo::ElementType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Thermo::Element* object = new Thermo::Element(-1, -1);
  object->unpack(buffer);
  return object;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 09/09 |
 | is called from ParObjectFactory                                      |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Thermo::ElementType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "THERMO")
  {
    std::shared_ptr<Core::Elements::Element> ele = std::make_shared<Thermo::Element>(id, owner);
    return ele;
  }
  return nullptr;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 09/09 |
 | virtual method of ElementType                                        |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Thermo::ElementType::create(const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele = std::make_shared<Thermo::Element>(id, owner);
  return ele;
}  // Create()


/*----------------------------------------------------------------------*
 |                                                           dano 08/12 |
 *----------------------------------------------------------------------*/
void Thermo::ElementType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = dwele->num_dof_per_node(*(dwele->nodes()[0]));
  dimns = numdf;
  nv = numdf;
}  // nodal_block_information()


/*----------------------------------------------------------------------*
 | ctor (public)                                             dano 08/12 |
 *----------------------------------------------------------------------*/
Core::LinAlg::SerialDenseMatrix Thermo::ElementType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  Core::LinAlg::SerialDenseMatrix nullspace;
  FOUR_C_THROW("method ComputeNullSpace not implemented!");
  return nullspace;
}


/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 09/09 |
 | is called from ParObjectFactory                                      |
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Thermo::FaceElementType::create(
    const int id, const int owner)
{
  // return Teuchos::rcp(new FaceElement(id,owner));
  return nullptr;
}  // Create()


/*----------------------------------------------------------------------*
 | setup element                                             dano 09/09 |
 *----------------------------------------------------------------------*/
void Thermo::ElementType::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["THERMO"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::hex8] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::hex20] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::hex27] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::tet4] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::tet10] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::wedge6] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::wedge15] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::pyramid5] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::nurbs27] = all_of({
      parameter<int>("MAT"),
  });

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

  defs[Core::FE::CellType::nurbs4] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::nurbs9] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::line2] = all_of({
      parameter<int>("MAT"),
  });

  defs[Core::FE::CellType::line3] = all_of({
      parameter<int>("MAT"),
  });
}  // setup_element_definition()


Thermo::FaceElementType Thermo::FaceElementType::instance_;

Thermo::FaceElementType& Thermo::FaceElementType::instance() { return instance_; }

/*----------------------------------------------------------------------*
 | ctor (public)                                             dano 09/09 |
 *----------------------------------------------------------------------*/
Thermo::Element::Element(int id, int owner)
    : Core::Elements::Element(id, owner), distype_(Core::FE::CellType::dis_none)
{
  // default: geometrically linear, also including purely thermal problem
  kintype_ = Inpar::Solid::KinemType::linear;
  return;
}  // ctor


/*----------------------------------------------------------------------*
 | copy-ctor (public)                                        dano 09/09 |
 *----------------------------------------------------------------------*/
Thermo::Element::Element(const Element& old)
    : Core::Elements::Element(old), kintype_(old.kintype_), distype_(old.distype_)
{
}  // copy-ctor


/*----------------------------------------------------------------------*
 | deep copy this instance of Thermo and return              dano 09/09 |
 | pointer to it (public)                                               |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Thermo::Element::clone() const
{
  Thermo::Element* newelement = new Thermo::Element(*this);
  return newelement;
}  // clone()


/*----------------------------------------------------------------------*
 | return the shape of a Thermo element (public)             dano 09/09 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Thermo::Element::shape() const { return distype_; }  // Shape()


/*----------------------------------------------------------------------*
 | pack data (public)                                        dano 09/09 |
 *----------------------------------------------------------------------*/
void Thermo::Element::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Core::Elements::Element::pack(data);
  // kintype
  add_to_pack(data, kintype_);
  // distype
  add_to_pack(data, distype_);

  return;
}  // pack()


/*----------------------------------------------------------------------*
 | unpack data (public)                                      dano 09/09 |
 *----------------------------------------------------------------------*/
void Thermo::Element::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Core::Elements::Element::unpack(buffer);
  // kintype_
  extract_from_pack(buffer, kintype_);
  // distype
  extract_from_pack(buffer, distype_);


  return;
}  // unpack()



/*----------------------------------------------------------------------*
 | print this element (public)                               dano 09/09 |
 *----------------------------------------------------------------------*/
void Thermo::Element::print(std::ostream& os) const
{
  os << "Thermo element";
  Core::Elements::Element::print(os);
  std::cout << std::endl;
  std::cout << "DiscretizationType:  " << Core::FE::cell_type_to_string(distype_) << std::endl;
  std::cout << std::endl;
  std::cout << "Number DOF per Node: " << numdofpernode_ << std::endl;
  std::cout << std::endl;
  return;
}  // print()


/*----------------------------------------------------------------------*
 | get vector of lines (public)                              dano 09/09 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Thermo::Element::lines()
{
  return Core::Communication::get_element_lines<FaceElement, Element>(*this);
}  // Lines()


/*----------------------------------------------------------------------*
 | get vector of surfaces (public)                           dano 09/09 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Thermo::Element::surfaces()
{
  return Core::Communication::get_element_surfaces<FaceElement, Element>(*this);
}  // Surfaces()

/*----------------------------------------------------------------------*
 | return visualization data (public)                        dano 09/09 |
 *----------------------------------------------------------------------*/
bool Thermo::Element::vis_data(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  return false;
}  // vis_data()

/*----------------------------------------------------------------------------*
 | END Thermo::Element
 *----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | ctor (public)                                             dano 09/09 |
 *----------------------------------------------------------------------*/
Thermo::FaceElement::FaceElement(int id, int owner, int nnode, const int* nodeids,
    Core::Nodes::Node** nodes, Element* parent, const int lsurface)
    : Core::Elements::FaceElement(id, owner)
{
  set_node_ids(nnode, nodeids);
  build_nodal_pointers(nodes);
  set_parent_master_element(parent, lsurface);
  return;
}  // ctor


/*----------------------------------------------------------------------*
 | copy-ctor (public)                                        dano 09/09 |
 *----------------------------------------------------------------------*/
Thermo::FaceElement::FaceElement(const FaceElement& old) : Core::Elements::FaceElement(old)
{
  return;
}  // copy-ctor


/*----------------------------------------------------------------------*
 | deep copy this instance return pointer to it (public)     dano 09/09 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Thermo::FaceElement::clone() const
{
  FaceElement* newelement = new FaceElement(*this);
  return newelement;
}  // clone()


/*----------------------------------------------------------------------*
 | return shape of this element (public)                     dano 09/09 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Thermo::FaceElement::shape() const
{
  switch (num_node())
  {
    case 2:
      return Core::FE::CellType::line2;
    case 3:
      if ((parent_element()->shape() == Core::FE::CellType::quad8) or
          (parent_element()->shape() == Core::FE::CellType::quad9))
        return Core::FE::CellType::line3;
      else
        return Core::FE::CellType::tri3;
    case 4:
      return Core::FE::CellType::quad4;
    case 6:
      return Core::FE::CellType::tri6;
    case 8:
      return Core::FE::CellType::quad8;
    case 9:
      if (parent_element()->shape() == Core::FE::CellType::hex27)
        return Core::FE::CellType::quad9;
      else if (parent_element()->shape() == Core::FE::CellType::nurbs27)
        return Core::FE::CellType::nurbs9;
      else
      {
        FOUR_C_THROW(
            "Your parent discretization type is {}. Ccurrently only hex27 and nurbs27 are "
            "implemented.",
            Core::FE::cell_type_to_string(parent_element()->shape()).c_str());
      }
      break;
    default:
      FOUR_C_THROW("unexpected number of nodes {}", num_node());
  }
}  // Shape()


/*----------------------------------------------------------------------*
 | pack data (public)                                        dano 09/09 |
 *----------------------------------------------------------------------*/
void Thermo::FaceElement::pack(std::vector<char>& data) const
{
  FOUR_C_THROW("This FaceElement element does not support communication");

  return;
}  // pack()


/*----------------------------------------------------------------------*
 | unpack data (public)                                      dano 09/09 |
 *----------------------------------------------------------------------*/
void Thermo::FaceElement::unpack(Core::Communication::UnpackBuffer& buffer)
{
  FOUR_C_THROW("This FaceElement element does not support communication");
  return;
}  // unpack()



/*----------------------------------------------------------------------*
 | print this element (public)                               dano 09/09 |
 *----------------------------------------------------------------------*/
void Thermo::FaceElement::print(std::ostream& os) const
{
  os << "FaceElement ";
  Element::print(os);
  return;
}  // print()


/*----------------------------------------------------------------------*
 | get vector of lines (public)                              dano 09/09 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Thermo::FaceElement::lines()
{
  FOUR_C_THROW("Lines of FaceElement not implemented");
}  // Lines()


/*----------------------------------------------------------------------*
 | get vector of lines (public)                              dano 09/09 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Thermo::FaceElement::surfaces()
{
  FOUR_C_THROW("Surfaces of FaceElement not implemented");
}  // Surfaces()

FOUR_C_NAMESPACE_CLOSE
