// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_elemag_ele.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_elemag_ele_boundary_calc.hpp"
#include "4C_elemag_ele_intfaces_calc.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_faces.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_so3_nullspace.hpp"

FOUR_C_NAMESPACE_OPEN


Discret::Elements::ElemagType Discret::Elements::ElemagType::instance_;
Discret::Elements::ElemagBoundaryType Discret::Elements::ElemagBoundaryType::instance_;
Discret::Elements::ElemagIntFaceType Discret::Elements::ElemagIntFaceType::instance_;

Discret::Elements::ElemagType& Discret::Elements::ElemagType::instance() { return instance_; }

Discret::Elements::ElemagBoundaryType& Discret::Elements::ElemagBoundaryType::instance()
{
  return instance_;
}

Discret::Elements::ElemagIntFaceType& Discret::Elements::ElemagIntFaceType::instance()
{
  return instance_;
}


Core::Communication::ParObject* Discret::Elements::ElemagType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::Elemag* object = new Discret::Elements::Elemag(-1, -1);
  object->unpack(buffer);
  return object;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::ElemagType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "ELECTROMAGNETIC")
  {
    return std::make_shared<Discret::Elements::Elemag>(id, owner);
  }
  return nullptr;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::ElemagType::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::Elemag>(id, owner);
}


void Discret::Elements::ElemagType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  nv = Core::FE::get_dimension(dwele->shape()) - 1;
  dimns = nv;
  numdf = dimns;
  return;
}


Core::LinAlg::SerialDenseMatrix Discret::Elements::ElemagType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  Core::LinAlg::SerialDenseMatrix nullspace;
  FOUR_C_THROW("method ComputeNullSpace not implemented right now!");
  return nullspace;
}


void Discret::Elements::ElemagType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["ELECTROMAGNETIC"];

  using namespace Core::IO::InputSpecBuilders;

  // 3D elements
  defs["HEX8"] = anonymous_group({
      entry<std::vector<int>>("HEX8", {.size = 8}),
      entry<int>("MAT"),
      entry<int>("DEG"),
      entry<int>("SPC"),
  });

  defs["TET4"] = anonymous_group({
      entry<std::vector<int>>("TET4", {.size = 4}),
      entry<int>("MAT"),
      entry<int>("DEG"),
      entry<int>("SPC"),
  });

  // 2D elements
  defs["QUAD4"] = anonymous_group({
      entry<std::vector<int>>("QUAD4", {.size = 4}),
      entry<int>("MAT"),
      entry<int>("DEG"),
      entry<int>("SPC"),
  });

  defs["QUAD9"] = anonymous_group({
      entry<std::vector<int>>("QUAD9", {.size = 9}),
      entry<int>("MAT"),
      entry<int>("DEG"),
      entry<int>("SPC"),
  });

  defs["TRI3"] = anonymous_group({
      entry<std::vector<int>>("TRI3", {.size = 3}),
      entry<int>("MAT"),
      entry<int>("DEG"),
      entry<int>("SPC"),
  });
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                       berardocco 02/18|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::Elemag::Elemag(int id, int owner)
    : Core::Elements::Element(id, owner), degree_(1), completepol_(true)
{
  distype_ = Core::FE::CellType::dis_none;
}


/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                  berardocco 02/18|
 *----------------------------------------------------------------------*/
Discret::Elements::Elemag::Elemag(const Discret::Elements::Elemag& old)
    : Core::Elements::Element(old),
      distype_(old.distype_),
      degree_(old.degree_),
      completepol_(old.completepol_)
{
}


/*----------------------------------------------------------------------*
 |  Deep copy this instance of Elemag and return pointer to it (public) |
 |                                                      berardocco 02/18|
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Elemag::clone() const
{
  Discret::Elements::Elemag* newelement = new Discret::Elements::Elemag(*this);
  return newelement;
}


/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                      berardocco 02/18|
 *----------------------------------------------------------------------*/
void Discret::Elements::Elemag::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // add base class Element
  Element::pack(data);

  // Discretisation type
  add_to_pack(data, distype_);
  add_to_pack(data, degree_);
  add_to_pack(data, completepol_);
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                      berardocco 02/18|
 *----------------------------------------------------------------------*/
void Discret::Elements::Elemag::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);

  // distype
  extract_from_pack(buffer, distype_);
  extract_from_pack(buffer, degree_);
  extract_from_pack(buffer, completepol_);
}



/*----------------------------------------------------------------------*
 |  print this element (public)                         berardocco 02/18|
 *----------------------------------------------------------------------*/
void Discret::Elements::Elemag::print(std::ostream& os) const
{
  os << "Elemag ";
  Element::print(os);
}


bool Discret::Elements::Elemag::read_element(const std::string& eletype, const std::string& distype,
    const Core::IO::InputParameterContainer& container)
{
  // read number of material model
  int material_id = container.get<int>("MAT");
  set_material(0, Mat::factory(material_id));
  degree_ = container.get<int>("DEG");

  completepol_ = container.get<int>("SPC");

  // set discretization type (setOptimalgaussrule is pushed into element
  // routine)
  set_dis_type(Core::FE::string_to_cell_type(distype));

  return true;
}


/*----------------------------------------------------------------------*
 |  get vector of lines              (public)           berardocco 02/18|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Elemag::lines()
{
  return Core::Communication::get_element_lines<ElemagBoundary, Elemag>(*this);
}


/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                     berardocco 02/18|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Elemag::surfaces()
{
  return Core::Communication::get_element_surfaces<ElemagBoundary, Elemag>(*this);
}

/*----------------------------------------------------------------------*
 |  get face element (public)                           berardocco 02/18|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::Elemag::create_face_element(
    Core::Elements::Element* parent_slave, int nnode, const int* nodeids, Core::Nodes::Node** nodes,
    const int lsurface_master, const int lsurface_slave, const std::vector<int>& localtrafomap)
{
  // dynamic cast for slave parent element
  Discret::Elements::Elemag* slave_pele = dynamic_cast<Discret::Elements::Elemag*>(parent_slave);

  // insert both parent elements
  return Core::Communication::element_int_face_factory<ElemagIntFace, Elemag>(-1, -1, nnode,
      nodeids, nodes, this, slave_pele, lsurface_master, lsurface_slave, localtrafomap);
}

//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================



//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================

std::shared_ptr<Core::Elements::Element> Discret::Elements::ElemagBoundaryType::create(
    const int id, const int owner)
{
  return nullptr;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                      berardocco 02/18 |
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::ElemagBoundary::ElemagBoundary(int id, int owner, int nnode, const int* nodeids,
    Core::Nodes::Node** nodes, Discret::Elements::Elemag* parent, const int lsurface)
    : Core::Elements::FaceElement(id, owner)
{
  set_parent_master_element(parent, lsurface);
  set_node_ids(nnode, nodeids);
  build_nodal_pointers(nodes);
  return;
}


/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                 berardocco 02/18 |
 *----------------------------------------------------------------------*/
Discret::Elements::ElemagBoundary::ElemagBoundary(const Discret::Elements::ElemagBoundary& old)
    : Core::Elements::FaceElement(old)
{
  return;
}


/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it               (public) |
 |                                                     berardocco 02/18 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::ElemagBoundary::clone() const
{
  Discret::Elements::ElemagBoundary* newelement = new Discret::Elements::ElemagBoundary(*this);
  return newelement;
}


/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                      berardocco 02/18|
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::ElemagBoundary::shape() const
{
  return Core::FE::get_shape_of_boundary_element(num_node(), parent_master_element()->shape());
}


/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                      berardocco 02/18|
 *----------------------------------------------------------------------*/
void Discret::Elements::ElemagBoundary::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Element::pack(data);

  // Discretisation type
  // add_to_pack(data,distype_);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                      berardocco 02/18|
 *----------------------------------------------------------------------*/
void Discret::Elements::ElemagBoundary::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);



  return;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                        berardocco 02/18 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ElemagBoundary::print(std::ostream& os) const
{
  os << "ElemagBoundary ";
  Element::print(os);
  return;
}


/*----------------------------------------------------------------------*
 |  get vector of lines (public)                       berardocco 02/18 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::ElemagBoundary::lines()
{
  FOUR_C_THROW("Lines of ElemagBoundary not implemented");
}


/*----------------------------------------------------------------------*
 |  get vector of lines (public)                       berardocco 02/18 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::ElemagBoundary::surfaces()
{
  FOUR_C_THROW("Surfaces of ElemagBoundary not implemented");
}


/*----------------------------------------------------------------------*
 |  evaluate the element (public)                      berardocco 02/18 |
 *----------------------------------------------------------------------*/
int Discret::Elements::ElemagBoundary::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  Discret::Elements::ElemagBoundaryImplInterface::impl(this)->evaluate(
      this, params, discretization, lm, elemat1, elemat2, elevec1, elevec2, elevec3);
  return 0;
}


/*-----------------------------------------------------------------------*
 |  Integrate a surface/line Neumann boundary condition berardocco 02/18 |
 *-----------------------------------------------------------------------*/
int Discret::Elements::ElemagBoundary::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  FOUR_C_THROW("dummy function called");
  return 0;
}

/*------------------------------------------------------------------------*
 |  Get degrees of freedom used by this element (public) berardocco 02/18 |
 *------------------------------------------------------------------------*/
void Discret::Elements::ElemagBoundary::location_vector(const Core::FE::Discretization& dis,
    Core::Elements::LocationArray& la, bool doDirichlet, const std::string& condstring,
    Teuchos::ParameterList& params) const
{
  // we have to do it this way, just as for weak Dirichlet conditions
  parent_master_element()->location_vector(dis, la, false);
  return;
}

//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================



//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================

std::shared_ptr<Core::Elements::Element> Discret::Elements::ElemagIntFaceType::create(
    const int id, const int owner)
{
  return nullptr;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                       berardocco 02/18|
 *----------------------------------------------------------------------*/
Discret::Elements::ElemagIntFace::ElemagIntFace(int id,  // element id
    int owner,                  // owner (= owner of parent element with smallest gid)
    int nnode,                  // number of nodes
    const int* nodeids,         // node ids
    Core::Nodes::Node** nodes,  // nodes of surface
    Discret::Elements::Elemag* parent_master,  // master parent element
    Discret::Elements::Elemag* parent_slave,   // slave parent element
    const int lsurface_master,  // local surface index with respect to master parent element
    const int lsurface_slave,   // local surface index with respect to slave parent element
    const std::vector<int>
        localtrafomap  // get the transformation map between the local coordinate systems of the
                       // face w.r.t the master parent element's face's coordinate system and the
                       // slave element's face's coordinate system
    )
    : Core::Elements::FaceElement(id, owner)
{
  set_parent_master_element(parent_master, lsurface_master);
  set_parent_slave_element(parent_slave, lsurface_slave);

  if (parent_slave != nullptr)
    degree_ = std::max(parent_master->degree(), parent_slave->degree());
  else
    degree_ = parent_master->degree();

  set_local_trafo_map(localtrafomap);

  set_node_ids(nnode, nodeids);
  build_nodal_pointers(nodes);
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                  berardocco 02/18|
 *----------------------------------------------------------------------*/
Discret::Elements::ElemagIntFace::ElemagIntFace(const Discret::Elements::ElemagIntFace& old)
    : Core::Elements::FaceElement(old), degree_(old.degree_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it               (public) |
 |                                                      berardocco 02/18|
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::ElemagIntFace::clone() const
{
  Discret::Elements::ElemagIntFace* newelement = new Discret::Elements::ElemagIntFace(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                     berardocco 02/18 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::ElemagIntFace::shape() const
{
  // could be called for master parent or slave parent element, doesn't matter
  return Core::FE::get_shape_of_boundary_element(num_node(), parent_master_element()->shape());
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                     berardocco 02/18 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ElemagIntFace::pack(Core::Communication::PackBuffer& data) const
{
  FOUR_C_THROW("this ElemagIntFace element does not support communication");
  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                     berardocco 02/18 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ElemagIntFace::unpack(Core::Communication::UnpackBuffer& buffer)
{
  FOUR_C_THROW("this ElemagIntFace element does not support communication");
  return;
}



/*----------------------------------------------------------------------*
 |  create the patch location vector (public)          berardocco 02/18 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ElemagIntFace::patch_location_vector(
    Core::FE::Discretization& discretization,  // discretization
    std::vector<int>& nds_master,              // nodal dofset w.r.t master parent element
    std::vector<int>& nds_slave,               // nodal dofset w.r.t slave parent element
    std::vector<int>& patchlm,                 // local map for gdof ids for patch of elements
    std::vector<int>& master_lm,               // local map for gdof ids for master element
    std::vector<int>& slave_lm,                // local map for gdof ids for slave element
    std::vector<int>& face_lm,                 // local map for gdof ids for face element
    std::vector<int>& lm_masterToPatch,        // local map between lm_master and lm_patch
    std::vector<int>& lm_slaveToPatch,         // local map between lm_slave and lm_patch
    std::vector<int>& lm_faceToPatch,          // local map between lm_face and lm_patch
    std::vector<int>& lm_masterNodeToPatch,    // local map between master nodes and nodes in patch
    std::vector<int>& lm_slaveNodeToPatch      // local map between slave nodes and nodes in patch
)
{
  // create one patch location vector containing all dofs of master, slave and
  // *this ElemagIntFace element only once (no duplicates)

  //-----------------------------------------------------------------------
  const int m_numnode = parent_master_element()->num_node();
  Core::Nodes::Node** m_nodes = parent_master_element()->nodes();

  if (m_numnode != static_cast<int>(nds_master.size()))
  {
    FOUR_C_THROW("wrong number of nodes for master element");
  }

  //-----------------------------------------------------------------------
  const int s_numnode = parent_slave_element()->num_node();
  Core::Nodes::Node** s_nodes = parent_slave_element()->nodes();

  if (s_numnode != static_cast<int>(nds_slave.size()))
  {
    FOUR_C_THROW("wrong number of nodes for slave element");
  }

  //-----------------------------------------------------------------------
  const int f_numnode = num_node();
  Core::Nodes::Node** f_nodes = nodes();

  //-----------------------------------------------------------------------
  // create the patch local map and additional local maps between elements lm and patch lm

  patchlm.clear();

  master_lm.clear();
  slave_lm.clear();
  face_lm.clear();

  lm_masterToPatch.clear();
  lm_slaveToPatch.clear();
  lm_faceToPatch.clear();

  // maps between master/slave nodes and nodes in patch
  lm_masterNodeToPatch.clear();
  lm_slaveNodeToPatch.clear();

  // for each master node, the offset for node's dofs in master_lm
  std::map<int, int> m_node_lm_offset;


  // ---------------------------------------------------
  int dofset = 0;  // assume dofset 0

  int patchnode_count = 0;

  // fill patch lm with master's nodes
  for (int k = 0; k < m_numnode; ++k)
  {
    Core::Nodes::Node* node = m_nodes[k];
    std::vector<int> dof = discretization.dof(dofset, node);

    // get maximum of numdof per node with the help of master and/or slave element (returns 4 in 3D
    // case, does not return dofset's numnode)
    const int size = discretization.num_dof(dofset, node);
    const int offset = size * nds_master[k];

    FOUR_C_ASSERT(
        dof.size() >= static_cast<unsigned>(offset + size), "illegal physical dofs offset");

    // insert a pair of node-Id and current length of master_lm ( to get the start offset for node's
    // dofs)
    m_node_lm_offset.insert(std::pair<int, int>(node->id(), master_lm.size()));

    for (int j = 0; j < size; ++j)
    {
      int actdof = dof[offset + j];

      // current last index will be the index for next push_back operation
      lm_masterToPatch.push_back((patchlm.size()));

      patchlm.push_back(actdof);
      master_lm.push_back(actdof);
    }

    lm_masterNodeToPatch.push_back(patchnode_count);

    patchnode_count++;
  }

  // ---------------------------------------------------
  // fill patch lm with missing slave's nodes and extract slave's lm from patch_lm

  for (int k = 0; k < s_numnode; ++k)
  {
    Core::Nodes::Node* node = s_nodes[k];

    // slave node already contained?
    std::map<int, int>::iterator m_offset;
    m_offset = m_node_lm_offset.find(node->id());

    if (m_offset == m_node_lm_offset.end())  // node not included yet
    {
      std::vector<int> dof = discretization.dof(dofset, node);

      // get maximum of numdof per node with the help of master and/or slave element (returns 4 in
      // 3D case, does not return dofset's numnode)
      const int size = discretization.num_dof(dofset, node);
      const int offset = size * nds_slave[k];

      FOUR_C_ASSERT(
          dof.size() >= static_cast<unsigned>(offset + size), "illegal physical dofs offset");
      for (int j = 0; j < size; ++j)
      {
        int actdof = dof[offset + j];

        lm_slaveToPatch.push_back(patchlm.size());

        patchlm.push_back(actdof);
        slave_lm.push_back(actdof);
      }

      lm_slaveNodeToPatch.push_back(patchnode_count);

      patchnode_count++;
    }
    else  // node is also a master's node
    {
      const int size = discretization.num_dof(dofset, node);

      int offset = m_offset->second;

      for (int j = 0; j < size; ++j)
      {
        int actdof = master_lm[offset + j];

        slave_lm.push_back(actdof);

        // copy from lm_masterToPatch
        lm_slaveToPatch.push_back(lm_masterToPatch[offset + j]);
      }

      if (offset % size != 0)
        FOUR_C_THROW("there was at least one node with not %d dofs per node", size);
      int patchnode_index = offset / size;

      lm_slaveNodeToPatch.push_back(patchnode_index);
      // no patchnode_count++; (node already contained)
    }
  }

  // ---------------------------------------------------
  // extract face's lm from patch_lm
  for (int k = 0; k < f_numnode; ++k)
  {
    Core::Nodes::Node* node = f_nodes[k];

    // face node must be contained
    std::map<int, int>::iterator m_offset;
    m_offset = m_node_lm_offset.find(node->id());

    if (m_offset != m_node_lm_offset.end())  // node not included yet
    {
      const int size = discretization.num_dof(dofset, node);

      int offset = m_offset->second;

      for (int j = 0; j < size; ++j)
      {
        int actdof = master_lm[offset + j];

        face_lm.push_back(actdof);

        // copy from lm_masterToPatch
        lm_faceToPatch.push_back(lm_masterToPatch[offset + j]);
      }
    }
    else
      FOUR_C_THROW("face's nodes not contained in masternodes_offset map");
  }

  return;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                        berardocco 02/18 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ElemagIntFace::print(std::ostream& os) const
{
  os << "ElemagIntFace ";
  Element::print(os);
  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                       berardocco 02/18 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::ElemagIntFace::lines()
{
  FOUR_C_THROW("Lines of ElemagIntFace not implemented");
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                       berardocco 02/18 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::ElemagIntFace::surfaces()
{
  FOUR_C_THROW("Surfaces of ElemagIntFace not implemented");
}

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                      berardocco 02/18 |
 *----------------------------------------------------------------------*/
int Discret::Elements::ElemagIntFace::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // REMARK: this line ensures that the static Discret::Elements::ElemagIntFaceImplInterface::Impl
  // is created
  //         this line avoids linker errors
  // Discret::Elements::ElemagIntFaceImplInterface::Impl(this);

  FOUR_C_THROW("not available");

  return 0;
}


/*------------------------------------------------------------------------*
 |  Integrate a surface/line Neumann boundary condition  berardocco 02/18 |
 *------------------------------------------------------------------------*/
int Discret::Elements::ElemagIntFace::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  FOUR_C_THROW("not available");

  return 0;
}

FOUR_C_NAMESPACE_CLOSE
