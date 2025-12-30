// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_fluid_ele_intfaces_calc.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN


Discret::Elements::FluidIntFaceType Discret::Elements::FluidIntFaceType::instance_;

Discret::Elements::FluidIntFaceType& Discret::Elements::FluidIntFaceType::instance()
{
  return instance_;
}


std::shared_ptr<Core::Elements::Element> Discret::Elements::FluidIntFaceType::create(
    const int id, const int owner)
{
  return nullptr;
}



/*----------------------------------------------------------------------*
 |  ctor (public)                                           schott 03/12|
 *----------------------------------------------------------------------*/
Discret::Elements::FluidIntFace::FluidIntFace(int id,  ///< element id
    int owner,                  ///< owner (= owner of parent element with smallest gid)
    int nnode,                  ///< number of nodes
    const int* nodeids,         ///< node ids
    Core::Nodes::Node** nodes,  ///< nodes of surface
    Discret::Elements::Fluid* parent_master,  ///< master parent element
    Discret::Elements::Fluid* parent_slave,   ///< slave parent element
    const int lsurface_master,  ///< local surface index with respect to master parent element
    const int lsurface_slave,   ///< local surface index with respect to slave parent element
    const std::vector<int>
        localtrafomap  ///< get the transformation map between the local coordinate systems of the
                       ///< face w.r.t the master parent element's face's coordinate system and the
                       ///< slave element's face's coordinate system
    )
    : Core::Elements::FaceElement(id, owner)
{
  set_parent_master_element(parent_master, lsurface_master);
  set_parent_slave_element(parent_slave, lsurface_slave);
  set_local_trafo_map(localtrafomap);
  set_node_ids(nnode, nodeids);
  build_nodal_pointers(nodes);
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                      schott 03/12|
 *----------------------------------------------------------------------*/
Discret::Elements::FluidIntFace::FluidIntFace(const Discret::Elements::FluidIntFace& old)
    : Core::Elements::FaceElement(old)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it               (public) |
 |                                                          schott 03/12|
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::FluidIntFace::clone() const
{
  Discret::Elements::FluidIntFace* newelement = new Discret::Elements::FluidIntFace(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                         schott 03/12 |
 *----------------------------------------------------------------------*/
Core::FE::CellType Discret::Elements::FluidIntFace::shape() const
{
  // could be called for master parent or slave parent element, doesn't matter
  return Core::FE::get_shape_of_boundary_element(num_node(), parent_master_element()->shape());
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                         schott 03/12 |
 *----------------------------------------------------------------------*/
void Discret::Elements::FluidIntFace::pack(Core::Communication::PackBuffer& data) const
{
  FOUR_C_THROW("this FluidIntFace element does not support communication");
  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                         schott 03/12 |
 *----------------------------------------------------------------------*/
void Discret::Elements::FluidIntFace::unpack(Core::Communication::UnpackBuffer& buffer)
{
  FOUR_C_THROW("this FluidIntFace element does not support communication");
  return;
}



/*----------------------------------------------------------------------*
 |  create the patch location vector (public)              schott 06/14 |
 *----------------------------------------------------------------------*/
void Discret::Elements::FluidIntFace::patch_location_vector(
    Core::FE::Discretization& discretization,  ///< discretization
    std::vector<int>& nds_master,              ///< nodal dofset w.r.t master parent element
    std::vector<int>& nds_slave,               ///< nodal dofset w.r.t slave parent element
    std::vector<int>& patchlm,                 ///< local map for gdof ids for patch of elements
    std::vector<int>& lm_masterToPatch,        ///< local map between lm_master and lm_patch
    std::vector<int>& lm_slaveToPatch,         ///< local map between lm_slave and lm_patch
    std::vector<int>& lm_faceToPatch,          ///< local map between lm_face and lm_patch
    std::vector<int>& lm_masterNodeToPatch,  ///< local map between master nodes and nodes in patch
    std::vector<int>& lm_slaveNodeToPatch,   ///< local map between slave nodes and nodes in patch
    std::shared_ptr<const std::map<int, int>>
        pbcconnectivity  ///< connectivity between slave and PBC's master nodes
)
{
  // create one patch location vector containing all dofs of master, slave and
  // *this FluidIntFace element only once (no duplicates)
  TEUCHOS_FUNC_TIME_MONITOR("XFEM::Edgestab EOS: PatchLocationVector");


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



  // for each master node, the offset for node's dofs in master_lm
  std::map<int, int> m_node_lm_offset;

  //----------------------------------------------------
  int curr_patch_lm_size = 0;  // patch_lm.size() (equal to master_lm.size() during the fill of
                               // patch data with master data)

  //----------------------------------------------------
  // check for PBC nodes
  bool has_PBC = (pbcconnectivity != nullptr);


  // ---------------------------------------------------
  const int dofset = 0;  // assume dofset 0

  int patchnode_count = 0;

  // fill patch lm with master's nodes
  for (int k = 0; k < m_numnode; ++k)
  {
    Core::Nodes::Node* node = m_nodes[k];
    std::vector<int> dof;
    discretization.dof(dof, node, dofset, nds_master[k]);

    const int size = dof.size();

    int nid = node->id();  // node id of node and id of master node if it is a PBC node

    if (has_PBC)  // set the id of the master node if the node is a PBC node
    {
      const auto slave_it = pbcconnectivity->find(
          nid);  // find the slave node id, is there a corresponding pbc master node?

      if (slave_it != pbcconnectivity->end()) nid = slave_it->second;
    }

    // insert a pair of node-Id and current length of master_lm ( to get the start offset for node's
    // dofs)
    m_node_lm_offset.insert(std::pair<int, int>(nid, curr_patch_lm_size));

    for (int j = 0; j < size; ++j)
    {
      lm_masterToPatch.push_back(curr_patch_lm_size);

      int actdof = dof[j];
      patchlm.push_back(actdof);
      curr_patch_lm_size++;
    }

    lm_masterNodeToPatch.push_back(patchnode_count);

    patchnode_count++;
  }



  // ---------------------------------------------------
  // fill patch lm with missing slave's nodes and extract slave's lm from patch_lm

  for (int k = 0; k < s_numnode; ++k)
  {
    Core::Nodes::Node* node = s_nodes[k];

    int nid = node->id();  // node id of node and id of master node if it is a PBC node

    if (has_PBC)  // set the id of the master node if the node is a PBC node
    {
      const auto slave_it = pbcconnectivity->find(
          nid);  // find the slave node id, is there a corresponding pbc master node?

      if (slave_it != pbcconnectivity->end()) nid = slave_it->second;
    }

    // slave node already contained?
    std::map<int, int>::iterator m_offset;
    m_offset = m_node_lm_offset.find(nid);

    if (m_offset == m_node_lm_offset.end())  // node not included yet
    {
      std::vector<int> dof;  // = discretization.Dof(dofset,node);
      discretization.dof(dof, node, dofset,
          nds_slave[k]);  // in case of pbcs, the right dofs are stored also for the slave node

      const int size = dof.size();

      for (int j = 0; j < size; ++j)
      {
        lm_slaveToPatch.push_back(curr_patch_lm_size);

        int actdof = dof[j];
        patchlm.push_back(actdof);
        curr_patch_lm_size++;
      }

      lm_slaveNodeToPatch.push_back(patchnode_count);

      patchnode_count++;
    }
    else  // node is also a master's node
    {
      // get maximum of numdof per node with the help of master and/or slave element (returns 4 in
      // 3D case, does not return dofset's numdof)
      const int size = num_dof_per_node(*node);

      int offset = m_offset->second;

      for (int j = 0; j < size; ++j)
      {
        // copy from lm_masterToPatch
        lm_slaveToPatch.push_back(lm_masterToPatch[offset + j]);
      }

      if (offset % size != 0)
        FOUR_C_THROW("there was at least one node with not {} dofs per node", size);
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

    int nid = node->id();  // node id of node and id of master node if it is a PBC node

    if (has_PBC)  // set the id of the master node if the node is a PBC node
    {
      const auto slave_it = pbcconnectivity->find(
          nid);  // find the slave node id, is there a corresponding pbc master node?

      if (slave_it != pbcconnectivity->end()) nid = slave_it->second;
    }

    // face node must be contained
    std::map<int, int>::iterator m_offset;
    m_offset = m_node_lm_offset.find(nid);

    if (m_offset != m_node_lm_offset.end())  // node not included yet
    {
      // get maximum of numdof per node with the help of master and/or slave element (returns 4 in
      // 3D case, does not return dofset's numdof)
      const int size = num_dof_per_node(*node);

      int offset = m_offset->second;

      for (int j = 0; j < size; ++j)
      {
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
 |  create the patch location vector (public)              schott 03/12 |
 *----------------------------------------------------------------------*/
void Discret::Elements::FluidIntFace::patch_location_vector(
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
    std::vector<int>& lm_slaveNodeToPatch,   ///< local map between slave nodes and nodes in patch
    std::shared_ptr<const std::map<int, int>>
        pbcconnectivity  ///< connectivity between slave and PBC's master nodes
)
{
  // create one patch location vector containing all dofs of master, slave and
  // *this FluidIntFace element only once (no duplicates)
  TEUCHOS_FUNC_TIME_MONITOR("XFEM::Edgestab EOS: PatchLocationVector");

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

  //----------------------------------------------------
  int curr_patch_lm_size = 0;   // patch_lm.size()
  int curr_master_lm_size = 0;  // master_lm.size()

  //----------------------------------------------------
  // check for PBC nodes
  bool has_PBC = (pbcconnectivity != nullptr);

  // ---------------------------------------------------
  const int dofset = 0;  // assume dofset 0

  int patchnode_count = 0;


  // fill patch lm with master's nodes
  for (int k = 0; k < m_numnode; ++k)
  {
    Core::Nodes::Node* node = m_nodes[k];
    std::vector<int> dof;
    discretization.dof(dof, node, dofset, nds_master[k]);

    const int size = dof.size();

    int nid = node->id();  // node id of node and id of master node if it is a PBC node

    if (has_PBC)  // set the id of the master node if the node is a PBC node
    {
      const auto slave_it = pbcconnectivity->find(
          nid);  // find the slave node id, is there a corresponding pbc master node?

      if (slave_it != pbcconnectivity->end()) nid = slave_it->second;
    }

    // insert a pair of node-Id and current length of master_lm ( to get the start offset for node's
    // dofs)

    m_node_lm_offset.insert(std::pair<int, int>(nid, curr_master_lm_size));

    for (int j = 0; j < size; ++j)
    {
      int actdof = dof[j];

      // current last index will be the index for next push_back operation
      lm_masterToPatch.push_back(curr_patch_lm_size);

      patchlm.push_back(actdof);
      curr_patch_lm_size++;

      master_lm.push_back(actdof);
      curr_master_lm_size++;
    }

    lm_masterNodeToPatch.push_back(patchnode_count);

    patchnode_count++;
  }


  // ---------------------------------------------------
  // fill patch lm with missing slave's nodes and extract slave's lm from patch_lm

  for (int k = 0; k < s_numnode; ++k)
  {
    Core::Nodes::Node* node = s_nodes[k];

    int nid = node->id();  // node id of node and id of master node if it is a PBC node

    if (has_PBC)  // set the id of the master node if the node is a PBC node
    {
      const auto slave_it = pbcconnectivity->find(
          nid);  // find the slave node id, is there a corresponding pbc master node?

      if (slave_it != pbcconnectivity->end()) nid = slave_it->second;
    }

    // slave node already contained?
    std::map<int, int>::iterator m_offset;
    m_offset = m_node_lm_offset.find(nid);

    if (m_offset == m_node_lm_offset.end())  // node not included yet
    {
      std::vector<int> dof;  // = discretization.Dof(dofset,node);
      discretization.dof(dof, node, dofset, nds_slave[k]);

      const int size = dof.size();

      for (int j = 0; j < size; ++j)
      {
        int actdof = dof[j];

        lm_slaveToPatch.push_back(curr_patch_lm_size);

        patchlm.push_back(actdof);
        curr_patch_lm_size++;

        slave_lm.push_back(actdof);
      }

      lm_slaveNodeToPatch.push_back(patchnode_count);

      patchnode_count++;
    }
    else  // node is also a master's node
    {
      // get maximum of numdof per node with the help of master and/or slave element (returns 4 in
      // 3D case, does not return dofset's numdof)
      const int size = num_dof_per_node(*node);

      int offset = m_offset->second;

      for (int j = 0; j < size; ++j)
      {
        int actdof = master_lm[offset + j];

        slave_lm.push_back(actdof);

        // copy from lm_masterToPatch
        lm_slaveToPatch.push_back(lm_masterToPatch[offset + j]);
      }

      if (offset % size != 0)
        FOUR_C_THROW("there was at least one node with not {} dofs per node", size);
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

    int nid = node->id();  // node id of node and id of master node if it is a PBC node

    if (has_PBC)  // set the id of the master node if the node is a PBC node
    {
      const auto slave_it = pbcconnectivity->find(
          nid);  // find the slave node id, is there a corresponding pbc master node?

      if (slave_it != pbcconnectivity->end()) nid = slave_it->second;
    }

    // face node must be contained
    std::map<int, int>::iterator m_offset;
    m_offset = m_node_lm_offset.find(nid);

    if (m_offset != m_node_lm_offset.end())  // node not included yet
    {
      // get maximum of numdof per node with the help of master and/or slave element (returns 4 in
      // 3D case, does not return dofset's numdof)
      const int size = num_dof_per_node(*node);

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
 |  print this element (public)                            schott 03/12 |
 *----------------------------------------------------------------------*/
void Discret::Elements::FluidIntFace::print(std::ostream& os) const
{
  os << "FluidIntFace ";
  Element::print(os);
  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                           schott 03/12 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::FluidIntFace::lines()
{
  FOUR_C_THROW("Lines of FluidIntFace not implemented");
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                           schott 03/12 |
 *----------------------------------------------------------------------*/
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::FluidIntFace::surfaces()
{
  FOUR_C_THROW("Surfaces of FluidIntFace not implemented");
}

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                          schott 03/12 |
 *----------------------------------------------------------------------*/
int Discret::Elements::FluidIntFace::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // REMARK: this line ensures that the static Discret::Elements::FluidIntFaceImplInterface::Impl is
  // created
  //         this line avoids linker errors
  Discret::Elements::FluidIntFaceImplInterface::impl(this);

  FOUR_C_THROW("not available");

  return 0;
}


/*----------------------------------------------------------------------*
 |  Integrate a surface/line Neumann boundary condition    schott 03/12 |
 *----------------------------------------------------------------------*/
int Discret::Elements::FluidIntFace::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  FOUR_C_THROW("not available");

  return 0;
}

FOUR_C_NAMESPACE_CLOSE
