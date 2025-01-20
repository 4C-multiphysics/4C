// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_elemag_diff_ele.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_elemag_ele_boundary_calc.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_faces.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_io_input_spec_builders.hpp"

FOUR_C_NAMESPACE_OPEN

Discret::Elements::ElemagDiffType Discret::Elements::ElemagDiffType::instance_;
Discret::Elements::ElemagDiffBoundaryType Discret::Elements::ElemagDiffBoundaryType::instance_;
Discret::Elements::ElemagDiffIntFaceType Discret::Elements::ElemagDiffIntFaceType::instance_;

Discret::Elements::ElemagDiffType& Discret::Elements::ElemagDiffType::instance()
{
  return instance_;
}

Discret::Elements::ElemagDiffBoundaryType& Discret::Elements::ElemagDiffBoundaryType::instance()
{
  return instance_;
}

Discret::Elements::ElemagDiffIntFaceType& Discret::Elements::ElemagDiffIntFaceType::instance()
{
  return instance_;
}

/*----------------------------------------------------------------------*
 |                                                      berardocco 03/19|
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::ElemagDiffType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::ElemagDiff* object = new Discret::Elements::ElemagDiff(-1, -1);
  object->unpack(buffer);
  return object;
}

/*----------------------------------------------------------------------*
 |                                                      berardocco 03/19|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::ElemagDiffType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "ELECTROMAGNETICDIFF")
  {
    return std::make_shared<Discret::Elements::ElemagDiff>(id, owner);
  }
  return nullptr;
}

/*----------------------------------------------------------------------*
 |                                                      berardocco 03/19|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::ElemagDiffType::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::ElemagDiff>(id, owner);
}

void Discret::Elements::ElemagDiffType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = Core::FE::get_dimension(dwele->shape()) - 1;  // 2;  // Bad Luca! Hard coding is not nice!
  dimns = numdf;
  nv = numdf;
  np = 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SerialDenseMatrix Discret::Elements::ElemagDiffType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  Core::LinAlg::SerialDenseMatrix nullspace;
  FOUR_C_THROW("method ComputeNullSpace not implemented right now!");
  return nullspace;
}

/*----------------------------------------------------------------------*
 |                                                      berardocco 03/19|
 *----------------------------------------------------------------------*/
void Discret::Elements::ElemagDiffType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["ELECTROMAGNETICDIFF"];

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
 |  ctor (public)                                       berardocco 03/19|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::ElemagDiff::ElemagDiff(int id, int owner) : Elemag(id, owner)
{
  distype_ = Core::FE::CellType::dis_none;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                    berardocco 03/19|
 *----------------------------------------------------------------------*/
Discret::Elements::ElemagDiff::ElemagDiff(const Discret::Elements::ElemagDiff& old) : Elemag(old) {}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Elemag and return pointer to it (public)   |
 |                                                        berardocco 03/19|
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::ElemagDiff::clone() const
{
  Discret::Elements::ElemagDiff* newelement = new Discret::Elements::ElemagDiff(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  print this element (public)                         berardocco 03/19|
 *----------------------------------------------------------------------*/
void Discret::Elements::ElemagDiff::print(std::ostream& os) const
{
  os << "ElemagDiff ";
  Element::print(os);
  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines              (public)           berardocco 03/19|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::ElemagDiff::lines()
{
  return Core::Communication::get_element_lines<ElemagDiffBoundary, ElemagDiff>(*this);
}


/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                     berardocco 03/19|
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::ElemagDiff::surfaces()
{
  return Core::Communication::get_element_surfaces<ElemagDiffBoundary, ElemagDiff>(*this);
}


/*----------------------------------------------------------------------*
 |  get face element (public)                           berardocco 03/19|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::ElemagDiff::create_face_element(
    Core::Elements::Element* parent_slave,  //!< parent slave fluid3 element
    int nnode,                              //!< number of surface nodes
    const int* nodeids,                     //!< node ids of surface element
    Core::Nodes::Node** nodes,              //!< nodes of surface element
    const int lsurface_master,              //!< local surface number w.r.t master parent element
    const int lsurface_slave,               //!< local surface number w.r.t slave parent element
    const std::vector<int>& localtrafomap   //! local trafo map
)
{
  // dynamic cast for slave parent element
  Discret::Elements::ElemagDiff* slave_pele =
      dynamic_cast<Discret::Elements::ElemagDiff*>(parent_slave);

  // insert both parent elements
  return Core::Communication::element_int_face_factory<ElemagDiffIntFace, ElemagDiff>(
      -1,               //!< internal face element id
      -1,               //!< owner of internal face element
      nnode,            //!< number of surface nodes
      nodeids,          //!< node ids of surface element
      nodes,            //!< nodes of surface element
      this,             //!< master parent element
      slave_pele,       //!< slave parent element
      lsurface_master,  //!< local surface number w.r.t master parent element
      lsurface_slave,   //!< local surface number w.r.t slave parent element
      localtrafomap     //!< local trafo map
  );
}


//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================



//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================

std::shared_ptr<Core::Elements::Element> Discret::Elements::ElemagDiffBoundaryType::create(
    const int id, const int owner)
{
  return nullptr;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                      berardocco 03/19 |
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
Discret::Elements::ElemagDiffBoundary::ElemagDiffBoundary(int id, int owner, int nnode,
    const int* nodeids, Core::Nodes::Node** nodes, Discret::Elements::ElemagDiff* parent,
    const int lsurface)
    : ElemagBoundary(id, owner, nnode, nodeids, nodes, parent, lsurface)
{
  //  set_parent_master_element(parent,lsurface);
  //  SetNodeIds(nnode,nodeids);
  //  BuildNodalPointers(nodes);
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                 berardocco 03/19 |
 *----------------------------------------------------------------------*/
Discret::Elements::ElemagDiffBoundary::ElemagDiffBoundary(
    const Discret::Elements::ElemagDiffBoundary& old)
    : ElemagBoundary(old)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it               (public) |
 |                                                     berardocco 03/19 |
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::ElemagDiffBoundary::clone() const
{
  Discret::Elements::ElemagDiffBoundary* newelement =
      new Discret::Elements::ElemagDiffBoundary(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                      berardocco 03/19|
 *----------------------------------------------------------------------*/
void Discret::Elements::ElemagDiffBoundary::pack(Core::Communication::PackBuffer& data) const
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
 |                                                      berardocco 03/19|
 *----------------------------------------------------------------------*/
void Discret::Elements::ElemagDiffBoundary::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Element::unpack(buffer);



  return;
}

/*----------------------------------------------------------------------*
 |  print this element (public)                        berardocco 03/19 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ElemagDiffBoundary::print(std::ostream& os) const
{
  os << "ElemagDiffBoundary ";
  Element::print(os);
  return;
}

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                      berardocco 03/19 |
 *----------------------------------------------------------------------*/
int Discret::Elements::ElemagDiffBoundary::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  Discret::Elements::ElemagBoundaryImplInterface::impl(this)->evaluate(
      this, params, discretization, lm, elemat1, elemat2, elevec1, elevec2, elevec3);
  return 0;
}

/*----------------------------------------------------------------------*
 |  Get degrees of freedom used by this element (public) berardocco 03/19 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ElemagDiffBoundary::location_vector(const Core::FE::Discretization& dis,
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

std::shared_ptr<Core::Elements::Element> Discret::Elements::ElemagDiffIntFaceType::create(
    const int id, const int owner)
{
  return nullptr;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                       berardocco 03/19|
 *----------------------------------------------------------------------*/
Discret::Elements::ElemagDiffIntFace::ElemagDiffIntFace(int id,  ///< element id
    int owner,                  ///< owner (= owner of parent element with smallest gid)
    int nnode,                  ///< number of nodes
    const int* nodeids,         ///< node ids
    Core::Nodes::Node** nodes,  ///< nodes of surface
    Discret::Elements::ElemagDiff* parent_master,  ///< master parent element
    Discret::Elements::ElemagDiff* parent_slave,   ///< slave parent element
    const int lsurface_master,  ///< local surface index with respect to master parent element
    const int lsurface_slave,   ///< local surface index with respect to slave parent element
    const std::vector<int>
        localtrafomap  ///< get the transformation map between the local coordinate systems of the
                       ///< face w.r.t the master parent element's face's coordinate system and the
                       ///< slave element's face's coordinate system
    )
    : ElemagIntFace(id, owner, nnode, nodeids, nodes, parent_master, parent_slave, lsurface_master,
          lsurface_slave, localtrafomap)
{
  set_parent_master_element(parent_master, lsurface_master);
  set_parent_slave_element(parent_slave, lsurface_slave);
  set_local_trafo_map(localtrafomap);
  set_node_ids(nnode, nodeids);
  build_nodal_pointers(nodes);
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                  berardocco 03/19|
 *----------------------------------------------------------------------*/
Discret::Elements::ElemagDiffIntFace::ElemagDiffIntFace(
    const Discret::Elements::ElemagDiffIntFace& old)
    : ElemagIntFace(old)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it               (public) |
 |                                                      berardocco 03/19|
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::ElemagDiffIntFace::clone() const
{
  Discret::Elements::ElemagDiffIntFace* newelement =
      new Discret::Elements::ElemagDiffIntFace(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  create the patch location vector (public)          berardocco 03/19 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ElemagDiffIntFace::patch_location_vector(
    Core::FE::Discretization& discretization,  ///< discretization
    std::vector<int>& nds_master,              ///< nodal dofset w.r.t master parent element
    std::vector<int>& nds_slave,               ///< nodal dofset w.r.t slave parent element
    std::vector<int>& patchlm,                 ///< local map for gdof ids for patch of elements
    std::vector<int>& master_lm,               ///< local map for gdof ids for master element
    std::vector<int>& slave_lm,                ///< local map for gdof ids for slave element
    std::vector<int>& face_lm,                 ///< local map for gdof ids for face element
    std::vector<int>& lm_masterToPatch,        ///< local map between lm_master and lm_patch
    std::vector<int>& lm_slaveToPatch,         ///< local map between lm_slave and lm_patch
    std::vector<int>& lm_faceToPatch,          ///< local map between lm_face and lm_patch
    std::vector<int>& lm_masterNodeToPatch,  ///< local map between master nodes and nodes in patch
    std::vector<int>& lm_slaveNodeToPatch    ///< local map between slave nodes and nodes in patch
)
{
  // create one patch location vector containing all dofs of master, slave and
  // *this ElemagDiffIntFace element only once (no duplicates)

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
 |  print this element (public)                        berardocco 03/19 |
 *----------------------------------------------------------------------*/
void Discret::Elements::ElemagDiffIntFace::print(std::ostream& os) const
{
  os << "ElemagDiffIntFace ";
  Element::print(os);
  return;
}

FOUR_C_NAMESPACE_CLOSE
