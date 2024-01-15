/*----------------------------------------------------------------------*/
/*! \file

\brief Implementation of electromagnetic diffusion elements

<pre>
\level 2

</pre>
*/
/*----------------------------------------------------------------------*/

#include "baci_elemag_diff_ele.H"

#include "baci_comm_utils_factory.H"
#include "baci_elemag_ele_boundary_calc.H"
#include "baci_io_linedefinition.H"
#include "baci_lib_discret.H"
#include "baci_lib_discret_faces.H"

BACI_NAMESPACE_OPEN

DRT::ELEMENTS::ElemagDiffType DRT::ELEMENTS::ElemagDiffType::instance_;
DRT::ELEMENTS::ElemagDiffBoundaryType DRT::ELEMENTS::ElemagDiffBoundaryType::instance_;
DRT::ELEMENTS::ElemagDiffIntFaceType DRT::ELEMENTS::ElemagDiffIntFaceType::instance_;

DRT::ELEMENTS::ElemagDiffType& DRT::ELEMENTS::ElemagDiffType::Instance() { return instance_; }

DRT::ELEMENTS::ElemagDiffBoundaryType& DRT::ELEMENTS::ElemagDiffBoundaryType::Instance()
{
  return instance_;
}

DRT::ELEMENTS::ElemagDiffIntFaceType& DRT::ELEMENTS::ElemagDiffIntFaceType::Instance()
{
  return instance_;
}

/*----------------------------------------------------------------------*
 |                                                      berardocco 03/19|
 *----------------------------------------------------------------------*/
CORE::COMM::ParObject* DRT::ELEMENTS::ElemagDiffType::Create(const std::vector<char>& data)
{
  DRT::ELEMENTS::ElemagDiff* object = new DRT::ELEMENTS::ElemagDiff(-1, -1);
  object->Unpack(data);
  return object;
}

/*----------------------------------------------------------------------*
 |                                                      berardocco 03/19|
 *----------------------------------------------------------------------*/
Teuchos::RCP<DRT::Element> DRT::ELEMENTS::ElemagDiffType::Create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "ELECTROMAGNETICDIFF")
  {
    return Teuchos::rcp(new DRT::ELEMENTS::ElemagDiff(id, owner));
  }
  return Teuchos::null;
}

/*----------------------------------------------------------------------*
 |                                                      berardocco 03/19|
 *----------------------------------------------------------------------*/
Teuchos::RCP<DRT::Element> DRT::ELEMENTS::ElemagDiffType::Create(const int id, const int owner)
{
  return Teuchos::rcp(new DRT::ELEMENTS::ElemagDiff(id, owner));
}

void DRT::ELEMENTS::ElemagDiffType::NodalBlockInformation(
    Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  numdf = CORE::FE::getDimension(dwele->Shape()) - 1;  // 2;  // Bad Luca! Hard coding is not nice!
  dimns = numdf;
  nv = numdf;
  np = 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CORE::LINALG::SerialDenseMatrix DRT::ELEMENTS::ElemagDiffType::ComputeNullSpace(
    DRT::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  CORE::LINALG::SerialDenseMatrix nullspace;
  dserror("method ComputeNullSpace not implemented right now!");
  return nullspace;
}

/*----------------------------------------------------------------------*
 |                                                      berardocco 03/19|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::ElemagDiffType::SetupElementDefinition(
    std::map<std::string, std::map<std::string, INPUT::LineDefinition>>& definitions)
{
  std::map<std::string, INPUT::LineDefinition>& defs = definitions["ELECTROMAGNETICDIFF"];

  // 3D elements
  defs["HEX8"] = INPUT::LineDefinition::Builder()
                     .AddIntVector("HEX8", 8)
                     .AddNamedInt("MAT")
                     .AddNamedInt("DEG")
                     .AddNamedInt("SPC")
                     .Build();

  defs["TET4"] = INPUT::LineDefinition::Builder()
                     .AddIntVector("TET4", 4)
                     .AddNamedInt("MAT")
                     .AddNamedInt("DEG")
                     .AddNamedInt("SPC")
                     .Build();

  // 2D elements
  defs["QUAD4"] = INPUT::LineDefinition::Builder()
                      .AddIntVector("QUAD4", 4)
                      .AddNamedInt("MAT")
                      .AddNamedInt("DEG")
                      .AddNamedInt("SPC")
                      .Build();

  defs["QUAD9"] = INPUT::LineDefinition::Builder()
                      .AddIntVector("QUAD9", 9)
                      .AddNamedInt("MAT")
                      .AddNamedInt("DEG")
                      .AddNamedInt("SPC")
                      .Build();

  defs["TRI3"] = INPUT::LineDefinition::Builder()
                     .AddIntVector("TRI3", 3)
                     .AddNamedInt("MAT")
                     .AddNamedInt("DEG")
                     .AddNamedInt("SPC")
                     .Build();
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                       berardocco 03/19|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::ElemagDiff::ElemagDiff(int id, int owner) : Elemag(id, owner)
{
  distype_ = CORE::FE::CellType::dis_none;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                    berardocco 03/19|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::ElemagDiff::ElemagDiff(const DRT::ELEMENTS::ElemagDiff& old) : Elemag(old) {}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Elemag and return pointer to it (public)   |
 |                                                        berardocco 03/19|
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::ElemagDiff::Clone() const
{
  DRT::ELEMENTS::ElemagDiff* newelement = new DRT::ELEMENTS::ElemagDiff(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  print this element (public)                         berardocco 03/19|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::ElemagDiff::Print(std::ostream& os) const
{
  os << "ElemagDiff ";
  Element::Print(os);
  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines              (public)           berardocco 03/19|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<DRT::Element>> DRT::ELEMENTS::ElemagDiff::Lines()
{
  return CORE::COMM::GetElementLines<ElemagDiffBoundary, ElemagDiff>(*this);
}


/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                     berardocco 03/19|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<DRT::Element>> DRT::ELEMENTS::ElemagDiff::Surfaces()
{
  return CORE::COMM::GetElementSurfaces<ElemagDiffBoundary, ElemagDiff>(*this);
}


/*----------------------------------------------------------------------*
 |  get face element (public)                           berardocco 03/19|
 *----------------------------------------------------------------------*/
Teuchos::RCP<DRT::Element> DRT::ELEMENTS::ElemagDiff::CreateFaceElement(
    DRT::Element* parent_slave,            //!< parent slave fluid3 element
    int nnode,                             //!< number of surface nodes
    const int* nodeids,                    //!< node ids of surface element
    DRT::Node** nodes,                     //!< nodes of surface element
    const int lsurface_master,             //!< local surface number w.r.t master parent element
    const int lsurface_slave,              //!< local surface number w.r.t slave parent element
    const std::vector<int>& localtrafomap  //! local trafo map
)
{
  // dynamic cast for slave parent element
  DRT::ELEMENTS::ElemagDiff* slave_pele = dynamic_cast<DRT::ELEMENTS::ElemagDiff*>(parent_slave);

  // insert both parent elements
  return CORE::COMM::ElementIntFaceFactory<ElemagDiffIntFace, ElemagDiff>(
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

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::ElemagDiffBoundaryType::Create(
    const int id, const int owner)
{
  return Teuchos::null;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                      berardocco 03/19 |
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::ElemagDiffBoundary::ElemagDiffBoundary(int id, int owner, int nnode,
    const int* nodeids, DRT::Node** nodes, DRT::ELEMENTS::ElemagDiff* parent, const int lsurface)
    : ElemagBoundary(id, owner, nnode, nodeids, nodes, parent, lsurface)
{
  //  SetParentMasterElement(parent,lsurface);
  //  SetNodeIds(nnode,nodeids);
  //  BuildNodalPointers(nodes);
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                 berardocco 03/19 |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::ElemagDiffBoundary::ElemagDiffBoundary(const DRT::ELEMENTS::ElemagDiffBoundary& old)
    : ElemagBoundary(old)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it               (public) |
 |                                                     berardocco 03/19 |
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::ElemagDiffBoundary::Clone() const
{
  DRT::ELEMENTS::ElemagDiffBoundary* newelement = new DRT::ELEMENTS::ElemagDiffBoundary(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                      berardocco 03/19|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::ElemagDiffBoundary::Pack(CORE::COMM::PackBuffer& data) const
{
  CORE::COMM::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);
  // add base class Element
  Element::Pack(data);

  // Discretisation type
  // AddtoPack(data,distype_);

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                      berardocco 03/19|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::ElemagDiffBoundary::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;

  CORE::COMM::ExtractAndAssertId(position, data, UniqueParObjectId());

  // extract base class Element
  std::vector<char> basedata(0);
  ExtractfromPack(position, data, basedata);
  Element::Unpack(basedata);

  // distype
  // distype_ = static_cast<CORE::FE::CellType>( ExtractInt(position,data) );

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d", (int)data.size(), position);

  return;
}

/*----------------------------------------------------------------------*
 |  print this element (public)                        berardocco 03/19 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::ElemagDiffBoundary::Print(std::ostream& os) const
{
  os << "ElemagDiffBoundary ";
  Element::Print(os);
  return;
}

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                      berardocco 03/19 |
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::ElemagDiffBoundary::Evaluate(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, std::vector<int>& lm,
    CORE::LINALG::SerialDenseMatrix& elemat1, CORE::LINALG::SerialDenseMatrix& elemat2,
    CORE::LINALG::SerialDenseVector& elevec1, CORE::LINALG::SerialDenseVector& elevec2,
    CORE::LINALG::SerialDenseVector& elevec3)
{
  DRT::ELEMENTS::ElemagBoundaryImplInterface::Impl(this)->Evaluate(
      this, params, discretization, lm, elemat1, elemat2, elevec1, elevec2, elevec3);
  return 0;
}

/*----------------------------------------------------------------------*
 |  Get degrees of freedom used by this element (public) berardocco 03/19 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::ElemagDiffBoundary::LocationVector(const Discretization& dis, LocationArray& la,
    bool doDirichlet, const std::string& condstring, Teuchos::ParameterList& params) const
{
  // we have to do it this way, just as for weak Dirichlet conditions
  ParentMasterElement()->LocationVector(dis, la, false);
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

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::ElemagDiffIntFaceType::Create(
    const int id, const int owner)
{
  return Teuchos::null;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                       berardocco 03/19|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::ElemagDiffIntFace::ElemagDiffIntFace(int id,  ///< element id
    int owner,           ///< owner (= owner of parent element with smallest gid)
    int nnode,           ///< number of nodes
    const int* nodeids,  ///< node ids
    DRT::Node** nodes,   ///< nodes of surface
    DRT::ELEMENTS::ElemagDiff* parent_master,  ///< master parent element
    DRT::ELEMENTS::ElemagDiff* parent_slave,   ///< slave parent element
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
  SetParentMasterElement(parent_master, lsurface_master);
  SetParentSlaveElement(parent_slave, lsurface_slave);
  SetLocalTrafoMap(localtrafomap);
  SetNodeIds(nnode, nodeids);
  BuildNodalPointers(nodes);
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                  berardocco 03/19|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::ElemagDiffIntFace::ElemagDiffIntFace(const DRT::ELEMENTS::ElemagDiffIntFace& old)
    : ElemagIntFace(old)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it               (public) |
 |                                                      berardocco 03/19|
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::ElemagDiffIntFace::Clone() const
{
  DRT::ELEMENTS::ElemagDiffIntFace* newelement = new DRT::ELEMENTS::ElemagDiffIntFace(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  create the patch location vector (public)          berardocco 03/19 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::ElemagDiffIntFace::PatchLocationVector(
    DRT::Discretization& discretization,     ///< discretization
    std::vector<int>& nds_master,            ///< nodal dofset w.r.t master parent element
    std::vector<int>& nds_slave,             ///< nodal dofset w.r.t slave parent element
    std::vector<int>& patchlm,               ///< local map for gdof ids for patch of elements
    std::vector<int>& master_lm,             ///< local map for gdof ids for master element
    std::vector<int>& slave_lm,              ///< local map for gdof ids for slave element
    std::vector<int>& face_lm,               ///< local map for gdof ids for face element
    std::vector<int>& lm_masterToPatch,      ///< local map between lm_master and lm_patch
    std::vector<int>& lm_slaveToPatch,       ///< local map between lm_slave and lm_patch
    std::vector<int>& lm_faceToPatch,        ///< local map between lm_face and lm_patch
    std::vector<int>& lm_masterNodeToPatch,  ///< local map between master nodes and nodes in patch
    std::vector<int>& lm_slaveNodeToPatch    ///< local map between slave nodes and nodes in patch
)
{
  // create one patch location vector containing all dofs of master, slave and
  // *this ElemagDiffIntFace element only once (no duplicates)

  //-----------------------------------------------------------------------
  const int m_numnode = ParentMasterElement()->NumNode();
  DRT::Node** m_nodes = ParentMasterElement()->Nodes();

  if (m_numnode != static_cast<int>(nds_master.size()))
  {
    throw CORE::Exception("wrong number of nodes for master element");
  }

  //-----------------------------------------------------------------------
  const int s_numnode = ParentSlaveElement()->NumNode();
  DRT::Node** s_nodes = ParentSlaveElement()->Nodes();

  if (s_numnode != static_cast<int>(nds_slave.size()))
  {
    throw CORE::Exception("wrong number of nodes for slave element");
  }

  //-----------------------------------------------------------------------
  const int f_numnode = NumNode();
  DRT::Node** f_nodes = Nodes();

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
    DRT::Node* node = m_nodes[k];
    std::vector<int> dof = discretization.Dof(dofset, node);

    // get maximum of numdof per node with the help of master and/or slave element (returns 4 in 3D
    // case, does not return dofset's numnode)
    const int size = discretization.NumDof(dofset, node);
    const int offset = size * nds_master[k];

    dsassert(dof.size() >= static_cast<unsigned>(offset + size), "illegal physical dofs offset");

    // insert a pair of node-Id and current length of master_lm ( to get the start offset for node's
    // dofs)
    m_node_lm_offset.insert(std::pair<int, int>(node->Id(), master_lm.size()));

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
    DRT::Node* node = s_nodes[k];

    // slave node already contained?
    std::map<int, int>::iterator m_offset;
    m_offset = m_node_lm_offset.find(node->Id());

    if (m_offset == m_node_lm_offset.end())  // node not included yet
    {
      std::vector<int> dof = discretization.Dof(dofset, node);

      // get maximum of numdof per node with the help of master and/or slave element (returns 4 in
      // 3D case, does not return dofset's numnode)
      const int size = discretization.NumDof(dofset, node);
      const int offset = size * nds_slave[k];

      dsassert(dof.size() >= static_cast<unsigned>(offset + size), "illegal physical dofs offset");
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
      const int size = discretization.NumDof(dofset, node);

      int offset = m_offset->second;

      for (int j = 0; j < size; ++j)
      {
        int actdof = master_lm[offset + j];

        slave_lm.push_back(actdof);

        // copy from lm_masterToPatch
        lm_slaveToPatch.push_back(lm_masterToPatch[offset + j]);
      }

      if (offset % size != 0)
        dserror("there was at least one node with not %d dofs per node", size);
      int patchnode_index = offset / size;

      lm_slaveNodeToPatch.push_back(patchnode_index);
      // no patchnode_count++; (node already contained)
    }
  }

  // ---------------------------------------------------
  // extract face's lm from patch_lm
  for (int k = 0; k < f_numnode; ++k)
  {
    DRT::Node* node = f_nodes[k];

    // face node must be contained
    std::map<int, int>::iterator m_offset;
    m_offset = m_node_lm_offset.find(node->Id());

    if (m_offset != m_node_lm_offset.end())  // node not included yet
    {
      const int size = discretization.NumDof(dofset, node);

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
      throw CORE::Exception("face's nodes not contained in masternodes_offset map");
  }

  return;
}


/*----------------------------------------------------------------------*
 |  print this element (public)                        berardocco 03/19 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::ElemagDiffIntFace::Print(std::ostream& os) const
{
  os << "ElemagDiffIntFace ";
  Element::Print(os);
  return;
}

BACI_NAMESPACE_CLOSE
