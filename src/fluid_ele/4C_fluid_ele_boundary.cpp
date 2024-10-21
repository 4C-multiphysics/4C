#include "4C_comm_pack_helpers.hpp"
#include "4C_fluid_ele.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::ELEMENTS::FluidBoundaryType Discret::ELEMENTS::FluidBoundaryType::instance_;

Discret::ELEMENTS::FluidBoundaryType& Discret::ELEMENTS::FluidBoundaryType::instance()
{
  return instance_;
}

Core::Communication::ParObject* Discret::ELEMENTS::FluidBoundaryType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::ELEMENTS::FluidBoundary* object = new Discret::ELEMENTS::FluidBoundary(-1, -1);
  object->unpack(buffer);
  return object;
}

Teuchos::RCP<Core::Elements::Element> Discret::ELEMENTS::FluidBoundaryType::create(
    const int id, const int owner)
{
  return Teuchos::null;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 01/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::FluidBoundary::FluidBoundary(int id, int owner, int nnode, const int* nodeids,
    Core::Nodes::Node** nodes, Discret::ELEMENTS::Fluid* parent, const int lsurface)
    : Core::Elements::FaceElement(id, owner),
      distype_(Core::FE::CellType::dis_none),
      numdofpernode_(-1)
{
  set_parent_master_element(parent, lsurface);
  set_node_ids(nnode, nodeids);
  build_nodal_pointers(nodes);
  distype_ = Core::FE::get_shape_of_boundary_element(num_node(), parent_master_element()->shape());

  numdofpernode_ = parent_master_element()->num_dof_per_node(*FluidBoundary::nodes()[0]);
  // Safety check if all nodes have the same number of dofs!
  for (int nlid = 1; nlid < num_node(); ++nlid)
  {
    if (numdofpernode_ != parent_master_element()->num_dof_per_node(*FluidBoundary::nodes()[nlid]))
      FOUR_C_THROW(
          "You need different NumDofPerNode for each node on this fluid boundary? (%d != %d)",
          numdofpernode_, parent_master_element()->num_dof_per_node(*FluidBoundary::nodes()[nlid]));
  }
  return;
}

/*------------------------------------------------------------------------*
 |  ctor (private) - used by FluidBoundaryType                  ager 12/16|
 *-----------------------------------------------------------------------*/
Discret::ELEMENTS::FluidBoundary::FluidBoundary(int id, int owner)
    : Core::Elements::FaceElement(id, owner),
      distype_(Core::FE::CellType::dis_none),
      numdofpernode_(-1)
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       mwgee 01/07|
 *----------------------------------------------------------------------*/
Discret::ELEMENTS::FluidBoundary::FluidBoundary(const Discret::ELEMENTS::FluidBoundary& old)
    : Core::Elements::FaceElement(old), distype_(old.distype_), numdofpernode_(old.numdofpernode_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it               (public) |
 |                                                            gee 01/07 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::ELEMENTS::FluidBoundary::clone() const
{
  Discret::ELEMENTS::FluidBoundary* newelement = new Discret::ELEMENTS::FluidBoundary(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                           ager 12/16 |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::FluidBoundary::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  FaceElement::pack(data);
  // Discretisation type
  add_to_pack(data, distype_);
  // add numdofpernode_
  add_to_pack(data, numdofpernode_);
  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                           ager 12/16 |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::FluidBoundary::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  std::vector<char> basedata(0);
  extract_from_pack(buffer, basedata);
  Core::Communication::UnpackBuffer base_buffer(basedata);
  FaceElement::unpack(base_buffer);
  // distype
  distype_ = static_cast<Core::FE::CellType>(extract_int(buffer));
  // numdofpernode_
  numdofpernode_ = extract_int(buffer);

  FOUR_C_THROW_UNLESS(buffer.at_end(), "Buffer not fully consumed.");
  return;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                              mwgee 01/07|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::FluidBoundary::print(std::ostream& os) const
{
  os << "FluidBoundary ";
  Element::print(os);
  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                             gammi 04/07|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::FluidBoundary::lines()
{
  FOUR_C_THROW("Lines of FluidBoundary not implemented");
}

/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                          ager 12/16 |
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<Core::Elements::Element>> Discret::ELEMENTS::FluidBoundary::surfaces()
{
  return {Teuchos::rcpFromRef(*this)};
}

FOUR_C_NAMESPACE_CLOSE
