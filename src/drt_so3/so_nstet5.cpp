/*!----------------------------------------------------------------------**##
\file so_nstet5.cpp

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include <Teuchos_TimeMonitor.hpp>
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_linedefinition.H"

#include "so_nstet5.H"
#include "so_surface.H"
#include "so_line.H"

DRT::ELEMENTS::NStet5Type DRT::ELEMENTS::NStet5Type::instance_;


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
DRT::ParObject* DRT::ELEMENTS::NStet5Type::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::NStet5* object = new DRT::ELEMENTS::NStet5(-1,-1);
  object->Unpack(data);
  return object;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
Teuchos::RCP<DRT::Element> DRT::ELEMENTS::NStet5Type::Create( const string eletype,
                                                            const string eledistype,
                                                            const int id,
                                                            const int owner )
{
  if ( eletype=="NSTET5" )
  {
    RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::NStet5(id,owner));
    return ele;
  }
  return Teuchos::null;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
Teuchos::RCP<DRT::Element> DRT::ELEMENTS::NStet5Type::Create( const int id, const int owner )
{
  RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::NStet5(id,owner));
  return ele;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void DRT::ELEMENTS::NStet5Type::NodalBlockInformation( DRT::Element * dwele, int & numdf, int & dimns, int & nv, int & np )
{
  numdf = 3;
  dimns = 6;
  nv = 3;
  np = 0;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void DRT::ELEMENTS::NStet5Type::ComputeNullSpace( DRT::Discretization & dis, std::vector<double> & ns, const double * x0, int numdf, int dimns )
{
  DRT::UTILS::ComputeStructure3DNullSpace( dis, ns, x0, numdf, dimns );
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void DRT::ELEMENTS::NStet5Type::SetupElementDefinition( std::map<std::string,std::map<std::string,DRT::INPUT::LineDefinition> > & definitions )
{
  std::map<std::string,DRT::INPUT::LineDefinition>& defs = definitions["NSTET5"];

  defs["TET4"].AddIntVector("TET4",4).AddNamedInt("MAT");
}


/*-----------------------------------------------------------------------
 |  ctor (public)                                              gee 03/12|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::NStet5::NStet5(int id, int owner) :
DRT::Element(id,owner),
material_(0),
V_(-1.0)
{
  sublm_[0] = 0;
  sublm_[1] = 1;
  sublm_[2] = 2;
  sublm_[3] = 4;
  sublm_[4] = 1;
  sublm_[5] = 3;
  sublm_[6] = 2;
  sublm_[7] = 4;
  sublm_[8] = 0;
  sublm_[9] = 3;
  sublm_[10] = 1;
  sublm_[11] = 4;
  sublm_[12] = 0;
  sublm_[13] = 2;
  sublm_[14] = 3;
  sublm_[15] = 4;
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                         gee 03/128|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::NStet5::NStet5(const DRT::ELEMENTS::NStet5& old) :
DRT::Element(old),
material_(old.material_),
V_(old.V_),
nxyz_(old.nxyz_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  dtor (public)                                              gee 03/12|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::NStet5::~NStet5()
{
  return;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                             gee 03/12|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::NStet5::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class Element
  Element::Pack(data);
  // material_
  AddtoPack(data,material_);
  // stresstype_
  AddtoPack(data,stresstype_);
  // V_
  AddtoPack(data,V_);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                             gee 03/12|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::NStet5::Unpack(const vector<char>& data)
{
  vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // extract base class Element
  vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  Element::Unpack(basedata);
  // material_
  ExtractfromPack(position,data,material_);
  // stresstype_
  stresstype_ = static_cast<StressType>( ExtractInt(position,data) );
  // V_
  ExtractfromPack(position,data,V_);

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}


/*----------------------------------------------------------------------*
 |  extrapolation of quantities at the GPs to the nodes      lw 03/08   |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::NStet5::so_nstet5_expol(LINALG::Matrix<1,6>& stresses,
                                        LINALG::Matrix<4,6>& nodalstresses)
{
  LINALG::Matrix<4,1> expol;
  expol(0,0)=1.0;
  expol(1,0)=1.0;
  expol(2,0)=1.0;
  expol(3,0)=1.0;
  nodalstresses.Multiply(expol,stresses);
  return;
}


/*----------------------------------------------------------------------*
 |  print this element (public)                                gee 03/12|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::NStet5::Print(ostream& os) const
{
  os << "NStet5 ";
  Element::Print(os);
  return;
}


  /*====================================================================*/
  /* 4-node tetrahedra node topology*/
  /*--------------------------------------------------------------------*/
  /* parameter coordinates (ksi1, ksi2, ksi3, ksi4) of nodes
   * of a common tetrahedron [-1,1]x[-1,1]x[-1,1]
   *  4-node hexahedron: node 0,1,...,3
   *
   * -----------------------
   *- this is the numbering used in GiD & EXODUS!!
   *      3-
   *      |\ ---
   *      |  \    ---
   *      |    \      ---
   *      |      \        -2
   *      |        \       /\
   *      |          \   /   \
   *      |            X      \
   *      |          /   \     \
   *      |        /       \    \
   *      |      /           \   \
   *      |    /               \  \
   *      |  /                   \ \
   *      |/                       \\
   *      0--------------------------1
   */
  /*====================================================================*/

/*----------------------------------------------------------------------*
 |  get vector of volumes (length 1) (public)                  gee 03/12|
 *----------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::NStet5::Volumes()
{
  dserror("volume not impl. yet");
  vector<RCP<Element> > volumes(1);
  volumes[0]= rcp(this, false);
  return volumes;
}


 /*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                             gee 03/12|
 *----------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::NStet5::Surfaces()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:
  return DRT::UTILS::ElementBoundaryFactory<StructuralSurface,DRT::Element>(DRT::UTILS::buildSurfaces,this);
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                               gee 03/12|
 *----------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::NStet5::Lines()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:
  return DRT::UTILS::ElementBoundaryFactory<StructuralLine,DRT::Element>(DRT::UTILS::buildLines,this);
}


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


/*----------------------------------------------------------------------*
 |                                                             gee 03/12|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::NStet5Type::InitElementsandMaps(
                           map<int,DRT::ELEMENTS::NStet5*>& elecids,
                           map<int,DRT::Node*>&             noderids,
                           const int                        myrank,
                           const int                        numproc,
                           DRT::Discretization&             dis)
{
  const int numele = dis.NumMyColElements();

  for (int i=0; i<numele; ++i)
  {
    if (dis.lColElement(i)->ElementType() != *this) continue;
    DRT::ELEMENTS::NStet5* actele =
                    dynamic_cast<DRT::ELEMENTS::NStet5*>(dis.lColElement(i));
    if (!actele) dserror("cast to NStet5* failed");

    // init the element
    actele->InitElement();

    // register element in list of column nstet elements
    elecids[actele->Id()] = actele;

    // compute a map of all row nodes adjacent to a NStet5 element
    for (int j=0; j<actele->NumNode(); ++j)
    {
      DRT::Node* node = actele->Nodes()[j];
      if (myrank == node->Owner())
        noderids[node->Id()] = node;
    }
  } // i

  return;
}


/*----------------------------------------------------------------------*
 |                                                             gee 03/12|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::NStet5Type::InitAdjacency(
                     map<int,DRT::ELEMENTS::NStet5*>&          elecids,
                     map<int,DRT::Node*>&                     noderids,
                     map<int,vector<DRT::ELEMENTS::NStet5*> >& adjele,
                     map<int,map<int,DRT::Node*> >&           adjnode,
                     map<int,vector<int> >&                   adjlm,
                     DRT::Discretization&                     dis)
{
  std::map<int,DRT::Node*>::iterator node;
  for (node=noderids.begin(); node != noderids.end(); ++node)
  {
    DRT::Node* nodeL  = node->second;
    const int nodeidL = nodeL->Id();

    // list of adjacent elements
    vector<DRT::ELEMENTS::NStet5*> myadjele(0);
    for (int j=0; j<nodeL->NumElement(); ++j)
    {
      const int eleid = node->second->Elements()[j]->Id();
      std::map<int,DRT::ELEMENTS::NStet5*>::iterator ele = elecids_.find(eleid);
      if (ele==elecids_.end()) continue;
      myadjele.push_back(ele->second);
    }
    adjele[nodeidL] = myadjele;

    // patch of all nodes adjacent to adjacent elements
    map<int,DRT::Node*> nodepatch;
    for (unsigned j=0; j<myadjele.size(); ++j)
    {
      DRT::Node** nodes = myadjele[j]->Nodes();
      for (int k=0; k<myadjele[j]->NumNode(); ++k)
        nodepatch[nodes[k]->Id()] = nodes[k];
    }
    adjnode_[nodeidL] = nodepatch;

    // lm array
    const int ndofperpatch = ((int)nodepatch.size() + 
                              (int)myadjele.size()) * 3;

    // location and ownership vector of nodal patch
    vector<int> lm(ndofperpatch);
    std::map<int,DRT::Node*>::iterator pnode;
    int count=0;
    // add dofs of nodes
    for (pnode=nodepatch.begin(); pnode != nodepatch.end(); ++pnode)
    {
      const vector<int>& dofs = dis.Dof(pnode->second);
      for (unsigned j=0; j<dofs.size(); ++j)
        lm[count++]        = dofs[j];
    }
    
#if 0
    printf("node %d nodal dofs: ",nodeidL);
    for (int i=0; i<count; ++i) printf("%d ",lm[i]);
    printf("\n");
    int start = count;
#endif
    
    // add dofs of center nodes from elements. These appear as element dofs
    for (unsigned j=0; j<myadjele.size(); ++j)
    {
      const vector<int>& dofs = dis.Dof(myadjele[j]);
      for (unsigned j=0; j<dofs.size(); ++j)
        lm[count++]        = dofs[j];
    }
    
#if 0
    printf("node %d ele   dofs: ",nodeidL);
    for (int i=start; i<count; ++i) printf("%d ",lm[i]);
    printf("\n\n");
#endif    
    
    adjlm[nodeidL] = lm;

    // for each adjele, find out which subelements I participate in
    map<int,vector<int> > masterele;
    for (unsigned j=0; j<myadjele.size(); ++j)
    {
      DRT::ELEMENTS::NStet5* ele = myadjele[j];
      bool foundit = false;
      for (int i=0; i<ele->NumNode(); ++i)
      {
        if (ele->Nodes()[i]->Id() == nodeL->Id())
        {
          // found the center node on the element
          // local to the element, its node i
          foundit = true;
          // determine subelements node i is attached to
          // its attached to definitely 3 out of 4 subelements
          vector<int> subele;
          for (int k=0; k<4; ++k)
          {
            const int* sublm = ele->SubLM(k); // subelement k
            for (int l=0; l<3; ++l) // the first 3 nodes of the subelement
              if (sublm[l] == i)
              {
                subele.push_back(k);
                break;
              }
          }
          if ((int)subele.size()!=3) dserror("Node not attached to exactly 3 subelements");
#if 0
          printf("node %d ele %d subele.size %d :",nodeidL,ele->Id(),(int)subele.size());
          for (int l=0; l<(int)subele.size(); ++l) printf("%d ",subele[l]);
          printf("\n");
#endif

          masterele[ele->Id()] = subele;
          
          // no longer need to look at this element
          break; 
        } 
      }
      if (!foundit) dserror("Weired, this adjele seems not attached to me");
    } // for (unsigned j=0; j<myadjele.size(); ++j)
    if (masterele.size() != myadjele.size()) dserror("subelement connectivity wrong");

    adjsubele_[nodeidL] = masterele;

#if 0
    printf("\n");
#endif
  } // for (node=noderids.begin(); node != noderids.end(); ++node)
  return;
}






/*----------------------------------------------------------------------*
 |  init the element (public)                                  gee 03/12|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::NStet5Type::Initialize(DRT::Discretization& dis)
{
  TEUCHOS_FUNC_TIME_MONITOR("DRT::ELEMENTS::NStet5Type::Initialize");

  const int myrank = dis.Comm().MyPID();
  const int numproc = dis.Comm().NumProc();

  //----------------------------------------------------------------------
  // init elements, make maps of column elements and row nodes
  InitElementsandMaps(elecids_,noderids_,myrank,numproc,dis);

  //----------------------------------------------------------------------
  // compute adjacency for each row node
  // make patch of adjacent elements
  // make patch of adjacent nodes (including center node itself)
  // make lm for nodal patch
  InitAdjacency(elecids_,noderids_,adjele_,adjnode_,adjlm_,dis);


  return 0;
}



#endif  // #ifdef CCADISCRET
