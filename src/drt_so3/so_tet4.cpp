/*!----------------------------------------------------------------------**##
\file so_tet4.cpp
\brief

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/

#include "so_tet4.H"
#include "so_surface.H"
#include "so_line.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_utils_factory.H"
#include "../drt_lib/drt_utils_nullspace.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_mat/so3_material.H"
#include "../drt_lib/drt_linedefinition.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_fem_general/drt_utils_fem_shapefunctions.H"

#include <Teuchos_StandardParameterEntryValidators.hpp>

// inverse design object
#include "inversedesign.H"
#include "prestress.H"


DRT::ELEMENTS::So_tet4Type DRT::ELEMENTS::So_tet4Type::instance_;

DRT::ELEMENTS::So_tet4Type& DRT::ELEMENTS::So_tet4Type::Instance()
{
  return instance_;
}

//------------------------------------------------------------------------
DRT::ParObject* DRT::ELEMENTS::So_tet4Type::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::So_tet4* object = new DRT::ELEMENTS::So_tet4(-1,-1);
  object->Unpack(data);
  return object;
}


//------------------------------------------------------------------------
Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_tet4Type::Create( const std::string eletype,
                                                            const std::string eledistype,
                                                            const int id,
                                                            const int owner )
{
  if ( eletype=="SOLIDT4" )
  {
    Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(new DRT::ELEMENTS::So_tet4(id,owner));
    return ele;
  }
  return Teuchos::null;
}


//------------------------------------------------------------------------
Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_tet4Type::Create( const int id, const int owner )
{
  Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(new DRT::ELEMENTS::So_tet4(id,owner));
  return ele;
}


//------------------------------------------------------------------------
void DRT::ELEMENTS::So_tet4Type::NodalBlockInformation( DRT::Element * dwele, int & numdf, int & dimns, int & nv, int & np )
{
  numdf = 3;
  dimns = 6;
  nv = 3;
}

//------------------------------------------------------------------------
void DRT::ELEMENTS::So_tet4Type::ComputeNullSpace( DRT::Discretization & dis, std::vector<double> & ns, const double * x0, int numdf, int dimns )
{
  DRT::UTILS::ComputeStructure3DNullSpace( dis, ns, x0, numdf, dimns );
}

//------------------------------------------------------------------------
void DRT::ELEMENTS::So_tet4Type::SetupElementDefinition( std::map<std::string,std::map<std::string,DRT::INPUT::LineDefinition> > & definitions )
{
  std::map<std::string,DRT::INPUT::LineDefinition>& defs = definitions["SOLIDT4"];

  defs["TET4"]
    .AddIntVector("TET4",4)
    .AddNamedInt("MAT")
    .AddNamedString("KINEM")
    .AddOptionalNamedDoubleVector("RAD",3)
    .AddOptionalNamedDoubleVector("AXI",3)
    .AddOptionalNamedDoubleVector("CIR",3)
    .AddOptionalNamedDoubleVector("FIBER1",3)
    .AddOptionalNamedDoubleVector("FIBER2",3)
    .AddOptionalNamedDoubleVector("FIBER3",3)
    .AddOptionalNamedDouble("HU")
    ;
}

/*----------------------------------------------------------------------***
 |  ctor (public)                                              maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_tet4::So_tet4(int id, int owner) :
DRT::Element(id,owner),
//material_(0),
//data_(),
V_(-1.0),
pstype_(INPAR::STR::prestress_none),
pstime_(0.0),
time_(0.0)
{

  Teuchos::RCP<const Teuchos::ParameterList> params = DRT::Problem::Instance()->getParameterList();
  if (params!=Teuchos::null)
  {
    const Teuchos::ParameterList& sdyn = DRT::Problem::Instance()->StructuralDynamicParams();
    pstype_ = DRT::INPUT::IntegralValue<INPAR::STR::PreStress>(sdyn,"PRESTRESS");
    pstime_ = sdyn.get<double>("PRESTRESSTIME");
  }

  if (pstype_==INPAR::STR::prestress_mulf)
    prestress_ = Teuchos::rcp(new DRT::ELEMENTS::PreStress(NUMNOD_SOTET4,NUMGPT_SOTET4,true));

  if (pstype_==INPAR::STR::prestress_id)
    invdesign_ = Teuchos::rcp(new DRT::ELEMENTS::InvDesign(NUMNOD_SOTET4,NUMGPT_SOTET4,true));

  return;
}

/*----------------------------------------------------------------------***
 |  copy-ctor (public)                                         maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_tet4::So_tet4(const DRT::ELEMENTS::So_tet4& old) :
DRT::Element(old),
//material_(old.material_),
//data_(old.data_),
V_(old.V_),
pstype_(old.pstype_),
pstime_(old.pstime_),
time_(old.time_)
{

  if (pstype_==INPAR::STR::prestress_mulf)
    prestress_ = Teuchos::rcp(new DRT::ELEMENTS::PreStress(*(old.prestress_)));

  if (pstype_==INPAR::STR::prestress_id)
    invdesign_ = Teuchos::rcp(new DRT::ELEMENTS::InvDesign(*(old.invdesign_)));

  return;
}

/*----------------------------------------------------------------------***
 |  Deep copy this instance of Solid3 and return pointer to it (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::So_tet4::Clone() const
{
  DRT::ELEMENTS::So_tet4* newelement = new DRT::ELEMENTS::So_tet4(*this);
  return newelement;
}

/*----------------------------------------------------------------------***
 |                                                             (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
DRT::Element::DiscretizationType DRT::ELEMENTS::So_tet4::Shape() const
{
  return tet4;
}

/*----------------------------------------------------------------------***
 |  Pack data                                                  (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_tet4::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class Element
  Element::Pack(data);
  // ngp_
  //AddtoPack(data,ngp_,3*sizeof(int));
  // material_
  //AddtoPack(data,material_);
  // kintype_
  AddtoPack(data,kintype_);

  //vector<char> tmp(0);
  //data_.Pack(tmp);
  //AddtoPack(data,tmp);

  // V_
  AddtoPack(data,V_);

  // prestress_
  AddtoPack(data,pstype_);
  AddtoPack(data,pstime_);
  AddtoPack(data,time_);
  if (pstype_==INPAR::STR::prestress_mulf)
  {
    DRT::ParObject::AddtoPack(data,*prestress_);
  }

  // invdesign_
  if (pstype_==INPAR::STR::prestress_id)
  {
    DRT::ParObject::AddtoPack(data,*invdesign_);
  }

  return;
}


/*----------------------------------------------------------------------***
 |  Unpack data                                                (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_tet4::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // extract base class Element
  std::vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  Element::Unpack(basedata);
  // ngp_
  //ExtractfromPack(position,data,ngp_,3*sizeof(int));
  // material_
  //ExtractfromPack(position,data,material_);
  // kintype_
  kintype_ = static_cast<INPAR::STR::KinemType>( ExtractInt(position,data) );
  // data_
  //vector<char> tmp(0);
  //ExtractfromPack(position,data,tmp);
  //data_.Unpack(tmp);
  // V_
  ExtractfromPack(position,data,V_);

  // prestress_
  pstype_ = static_cast<INPAR::STR::PreStress>( ExtractInt(position,data) );
  ExtractfromPack(position,data,pstime_);
  ExtractfromPack(position,data,time_);
  if (pstype_==INPAR::STR::prestress_mulf)
  {
    std::vector<char> tmpprestress(0);
    ExtractfromPack(position,data,tmpprestress);
    if (prestress_ == Teuchos::null)
      prestress_ = Teuchos::rcp(new DRT::ELEMENTS::PreStress(NUMNOD_SOTET4,NUMGPT_SOTET4,true));
    prestress_->Unpack(tmpprestress);
  }

  // invdesign_
  if (pstype_==INPAR::STR::prestress_id)
  {
    std::vector<char> tmpinvdesign(0);
    ExtractfromPack(position,data,tmpinvdesign);
    if (invdesign_ == Teuchos::null)
      invdesign_ = Teuchos::rcp(new DRT::ELEMENTS::InvDesign(NUMNOD_SOTET4,NUMGPT_SOTET4,true));
    invdesign_->Unpack(tmpinvdesign);
  }

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}


/*----------------------------------------------------------------------*
 |  extrapolation of quantities at the GPs to the nodes      lw 03/08   |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_tet4::so_tet4_expol
(
    LINALG::Matrix<NUMGPT_SOTET4,MAT::NUM_STRESS_3D>& stresses,
    Epetra_MultiVector& expolstresses
)
{
  static LINALG::Matrix<NUMNOD_SOTET4, NUMGPT_SOTET4> expol;
  static bool isfilled;

  if (isfilled==false)
  {
    expol(0,0)=1.0;
    expol(1,0)=1.0;
    expol(2,0)=1.0;
    expol(3,0)=1.0;

    isfilled=true;
  }

  LINALG::Matrix<NUMNOD_SOTET4,MAT::NUM_STRESS_3D> nodalstresses;
  nodalstresses.Multiply(expol,stresses);

  // "assembly" of extrapolated nodal stresses
  for (int i=0;i<NUMNOD_SOTET4;++i)
  {
    int gid = NodeIds()[i];
    if (expolstresses.Map().MyGID(NodeIds()[i])) // rownode
    {
      int lid = expolstresses.Map().LID(gid);
      int myadjele = Nodes()[i]->NumElement();
      for (int j=0;j<6;j++)
        (*(expolstresses(j)))[lid] += nodalstresses(i,j)/myadjele;
    }
  }
  return;
}




/*----------------------------------------------------------------------***
 |  dtor (public)                                              maf 04/07|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_tet4::~So_tet4()
{
  return;
}


/*----------------------------------------------------------------------***
 |  print this element (public)                                maf 04/07|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_tet4::Print(std::ostream& os) const
{
  os << "So_tet4 ";
  Element::Print(os);
  std::cout << std::endl;
  //std::cout << data_;
  return;
}

  /*====================================================================*/
  /* 4-node tetrahedra node topology*/
  /*--------------------------------------------------------------------*/
  /* parameter coordinates (ksi1, ksi2, ksi3) of nodes
   * of a common tetrahedron [0,1]x[0,1]x[0,1]
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

/*----------------------------------------------------------------------***
 |  get vector of volumes (length 1) (public)                  maf 04/07|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<DRT::Element> > DRT::ELEMENTS::So_tet4::Volumes()
{
  std::vector<Teuchos::RCP<Element> > volumes(1);
  volumes[0]= Teuchos::rcp(this, false);
  return volumes;
}


 /*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                             maf 04/07|
 |  surface normals always point outward                                 |
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<DRT::Element> > DRT::ELEMENTS::So_tet4::Surfaces()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:
  return DRT::UTILS::ElementBoundaryFactory<StructuralSurface,DRT::Element>(DRT::UTILS::buildSurfaces,this);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
std::vector<double> DRT::ELEMENTS::So_tet4::ElementCenterRefeCoords()
{
  // update element geometry
  DRT::Node** nodes = Nodes();
  LINALG::Matrix<NUMNOD_SOTET4,NUMDIM_SOTET4> xrefe;  // material coord. of element
  for (int i=0; i<NUMNOD_SOTET4; ++i){
    const double* x = nodes[i]->X();
    xrefe(i,0) = x[0];
    xrefe(i,1) = x[1];
    xrefe(i,2) = x[2];
  }
  const DRT::Element::DiscretizationType distype = Shape();
  LINALG::Matrix<NUMNOD_SOTET4,1> funct;
  // Centroid of a tet with (0,1)(0,1)(0,1) is (0.25, 0.25, 0.25)
  DRT::UTILS::shape_function_3D(funct, 0.25, 0.25, 0.25, distype);
  LINALG::Matrix<1,NUMDIM_SOTET4> midpoint;
  //midpoint.Multiply('T','N',1.0,funct,xrefe,0.0);
  midpoint.MultiplyTN(funct, xrefe);
  std::vector<double> centercoords(3);
  centercoords[0] = midpoint(0,0);
  centercoords[1] = midpoint(0,1);
  centercoords[2] = midpoint(0,2);
  return centercoords;
}

/*----------------------------------------------------------------------***++
 |  get vector of lines (public)                               maf 04/07|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<DRT::Element> > DRT::ELEMENTS::So_tet4::Lines()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:
  return DRT::UTILS::ElementBoundaryFactory<StructuralLine,DRT::Element>(DRT::UTILS::buildLines,this);
}

/*----------------------------------------------------------------------*
 |  Return names of visualization data (public)                 st 01/10|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_tet4::VisNames(std::map<std::string,int>& names)
{
  Teuchos::RCP<MAT::So3Material> so3mat = Teuchos::rcp_dynamic_cast<MAT::So3Material>(Material());
  so3mat->VisNames(names);

  return;
}

/*----------------------------------------------------------------------*
 |  Return visualization data (public)                          st 01/10|
 *----------------------------------------------------------------------*/
bool DRT::ELEMENTS::So_tet4::VisData(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (DRT::Element::VisData(name,data))
    return true;

  Teuchos::RCP<MAT::So3Material> so3mat = Teuchos::rcp_dynamic_cast<MAT::So3Material>(Material());
  return so3mat->VisData(name, data, NUMGPT_SOTET4,this->Id());
}

