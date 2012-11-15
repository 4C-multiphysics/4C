/*!----------------------------------------------------------------------
\file combust3.cpp
\brief

<pre>
Maintainer: Florian Henke
            henke@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15265
</pre>

*----------------------------------------------------------------------*/

#include "combust3.H"
#include "combust_interface.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_utils_factory.H"
#include "../drt_lib/drt_utils_nullspace.H"
#include "../drt_lib/drt_linedefinition.H"
#include "../drt_xfem/dof_management_element.H"
#include "../linalg/linalg_serialdensevector.H"


DRT::ELEMENTS::Combust3Type DRT::ELEMENTS::Combust3Type::instance_;


DRT::ParObject* DRT::ELEMENTS::Combust3Type::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::Combust3* object = new DRT::ELEMENTS::Combust3(-1,-1);
  object->Unpack(data);
  return object;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Combust3Type::Create( const string eletype,
                                                            const string eledistype,
                                                            const int id,
                                                            const int owner )
{
  if ( eletype=="COMBUST3" )
  {
    Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(new DRT::ELEMENTS::Combust3(id,owner));
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Combust3Type::Create( const int id, const int owner )
{
  Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(new DRT::ELEMENTS::Combust3(id,owner));
  return ele;
}


void DRT::ELEMENTS::Combust3Type::NodalBlockInformation( Element * dwele, int & numdf, int & dimns, int & nv, int & np )
{
  numdf = 4;
  dimns = 4;
  nv = 3;
  np = 1;
}


void DRT::ELEMENTS::Combust3Type::ComputeNullSpace( DRT::Discretization & dis, std::vector<double> & ns, const double * x0, int numdf, int dimns )
{
  DRT::UTILS::ComputeXFluidDNullSpace( dis, ns, x0, numdf, dimns );
}

void DRT::ELEMENTS::Combust3Type::SetupElementDefinition( std::map<std::string,std::map<std::string,DRT::INPUT::LineDefinition> > & definitions )
{
  std::map<std::string,DRT::INPUT::LineDefinition>& defs = definitions["COMBUST3"];

  defs["HEX8"]
    .AddIntVector("HEX8",8)
    .AddNamedInt("MAT")
    ;

  defs["HEX20"]
    .AddIntVector("HEX20",20)
    .AddNamedInt("MAT")
    ;

  defs["HEX27"]
    .AddIntVector("HEX27",27)
    .AddNamedInt("MAT")
    ;

  defs["TET4"]
    .AddIntVector("TET4",4)
    .AddNamedInt("MAT")
    ;

  defs["TET10"]
    .AddIntVector("TET10",10)
    .AddNamedInt("MAT")
    ;

  defs["WEDGE6"]
    .AddIntVector("WEDGE6",6)
    .AddNamedInt("MAT")
    ;

  defs["WEDGE15"]
    .AddIntVector("WEDGE15",15)
    .AddNamedInt("MAT")
    ;

  defs["PYRAMID5"]
    .AddIntVector("PYRAMID5",5)
    .AddNamedInt("MAT")
    ;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Combust3SurfaceType::Create( const int id, const int owner )
{
  //return Teuchos::rcp( new Combust3Surface( id, owner ) );
  return Teuchos::null;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Combust3LineType::Create( const int id, const int owner )
{
  //return Teuchos::rcp( new Combust3Line( id, owner ) );
  return Teuchos::null;
}


/*----------------------------------------------------------------------*/
// map to convert strings to actions (stabilization)
/*----------------------------------------------------------------------*/
std::map<string,DRT::ELEMENTS::Combust3::StabilisationAction> DRT::ELEMENTS::Combust3::stabstrtoact_;

/*----------------------------------------------------------------------*
 |  ctor (public)                                            gammi 02/08|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Combust3::Combust3(int id, int owner) :
DRT::Element(id,owner),
eleDofManager_(Teuchos::null),
standard_mode_(false),
bisected_(false),
trisected_(false),
touched_(false)
{
    return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       gammi 02/08|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Combust3::Combust3(const DRT::ELEMENTS::Combust3& old) :
DRT::Element(old),
eleDofManager_(old.eleDofManager_),
standard_mode_(old.standard_mode_),
bisected_(old.bisected_),
trisected_(old.trisected_),
touched_(old.touched_)
{
    return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Combust3 and return pointer to it (public)|
 |                                                          gammi 02/08 |
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::Combust3::Clone() const
{
  DRT::ELEMENTS::Combust3* newelement = new DRT::ELEMENTS::Combust3(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                          u.kue 03/07 |
 *----------------------------------------------------------------------*/
DRT::Element::DiscretizationType DRT::ELEMENTS::Combust3::Shape() const
{
  switch (NumNode())
  {
  case  4: return tet4;
  case  5: return pyramid5;
  case  6: return wedge6;
  case  8: return hex8;
  case 10: return tet10;
  case 15: return wedge15;
  case 20: return hex20;
  case 27: return hex27;
  default:
    dserror("unexpected number of nodes %d", NumNode());
  }
  return dis_none;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                          gammi 02/08 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Combust3::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class Element
  Element::Pack(data);

  AddtoPack(data,standard_mode_);
  AddtoPack(data,bisected_);
  AddtoPack(data,trisected_);
  AddtoPack(data,touched_);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                          gammi 02/08 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Combust3::Unpack(const std::vector<char>& data)
{
  vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  dsassert(type == UniqueParObjectId(), "wrong instance type data");
  // extract base class Element
  vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  Element::Unpack(basedata);

  standard_mode_ = ExtractInt(position,data);
  bisected_ = ExtractInt(position,data);
  trisected_ = ExtractInt(position,data);
  touched_ = ExtractInt(position,data);

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}


int DRT::ELEMENTS::Combust3::NumDofPerNode(const DRT::Node& node) const
{
  if (standard_mode_)
  {
    return 4;
  }
  else
  {
    if (eleDofManager_ != Teuchos::null)
    {
      return eleDofManager_->NumDofPerNode(node.Id());
    }
    else
    {
      dserror("no element dof information available!");
      return 0;
    }
  }
}



int DRT::ELEMENTS::Combust3::NumDofPerElement() const
{
  if (standard_mode_)
  {
    return 0;
  }
  else
  {
    if (eleDofManager_ != Teuchos::null)
    {
      return eleDofManager_->NumElemDof();
    }
    else
    {
      dserror("no element dof information available!");
      return 0;
    }
  }
}



/*----------------------------------------------------------------------*
 |  dtor (public)                                            gammi 02/08|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Combust3::~Combust3()
{
  return;
}


/*----------------------------------------------------------------------*
 |  print this element (public)                              gammi 02/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Combust3::Print(ostream& os) const
{
  os << "Combust3 ";
  if (standard_mode_)
    os << "(outputmode=true)";
  Element::Print(os);
  cout << endl;
  return;
}


/*----------------------------------------------------------------------*
 |  get vector of lines              (public)                  gjb 03/07|
 *----------------------------------------------------------------------*/
vector<Teuchos::RCP<DRT::Element> > DRT::ELEMENTS::Combust3::Lines()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:
  return DRT::UTILS::ElementBoundaryFactory<Combust3Line,Combust3>(DRT::UTILS::buildLines,this);
}


/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                            gjb 05/08|
 *----------------------------------------------------------------------*/
vector<Teuchos::RCP<DRT::Element> > DRT::ELEMENTS::Combust3::Surfaces()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:
  return DRT::UTILS::ElementBoundaryFactory<Combust3Surface,Combust3>(DRT::UTILS::buildSurfaces,this);
}


/*----------------------------------------------------------------------*
 |  get vector of volumes (length 1) (public)                g.bau 03/07|
 *----------------------------------------------------------------------*/
vector<Teuchos::RCP<DRT::Element> > DRT::ELEMENTS::Combust3::Volumes()
{
  vector<RCP<Element> > volumes(1);
  volumes[0]= Teuchos::rcp(this, false);
  return volumes;
}


/*----------------------------------------------------------------------*
 | constructor                                              henke 04/10 |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Combust3::MyState::MyState(
    const DRT::Discretization&                  discretization,
    const std::vector<int>&                     lm,
    const bool                                  instationary,
    const bool                                  genalpha,
    const bool                                  gradphi,
    const DRT::ELEMENTS::Combust3*              ele,
    const Epetra_Vector*                        phinp,
    const Epetra_MultiVector*                   gradphinp,
    const Epetra_Vector*                        curvature
    ) :
      instationary_(instationary),
      genalpha_(genalpha),
      gradphi_(gradphi)
{
  if (!genalpha_)
    DRT::UTILS::ExtractMyValues(*discretization.GetState("velnp"),velnp_,lm);
  if (instationary_)
  {
    DRT::UTILS::ExtractMyValues(*discretization.GetState("veln") ,veln_ ,lm);
    DRT::UTILS::ExtractMyValues(*discretization.GetState("velnm"),velnm_,lm);
    DRT::UTILS::ExtractMyValues(*discretization.GetState("accn") ,accn_ ,lm);
    if(genalpha_)
    {
      DRT::UTILS::ExtractMyValues(*discretization.GetState("velaf") ,velaf_ ,lm);
      DRT::UTILS::ExtractMyValues(*discretization.GetState("accam") ,accam_ ,lm);
    }
  }

#ifdef DEBUG
  // check if this element is the first element on this processor
  // remark:
  // The SameAs-operation requires MPI communication between processors. Therefore it can only
  // be performed once (at the beginning) on each processor. Otherwise some processors would
  // wait to receive MPI information, but would never get it, because some processores are
  // already done with their element loop. This will cause a mean parallel bug!   henke 11.08.09
  if(ele->Id() == discretization.lRowElement(0)->Id())
  {
    // get map of this vector
    const Epetra_BlockMap& phimap = phinp->Map();
    // check, whether this map is still identical with the current node map in the discretization
    if (not phimap.SameAs(*discretization.NodeColMap())) dserror("node column map has changed!");
  }
#endif

  // extract local (element level) G-function values from global vector
  DRT::UTILS::ExtractMyNodeBasedValues(ele, phinp_,*phinp);

  if(gradphi_)
  {
    // extract local (element level) G-function values from global vector
    // only if element is intersected; only adjacent nodal values are calculated
#ifndef COMBUST_NORMAL_ENRICHMENT
    if(ele->Bisected() or ele->Touched() )
    {
#endif
      // remark: - for the normal enrichment strategy all enriched elements are needed here
      //         - the intersected elements are not enough since normal shape functions are also
      //           needed for partially enriched elements
      //         - for simplicity the phi gradient is fetched for every element
      if (gradphinp == NULL) dserror("no gradient of phi has been computed!");
      DRT::UTILS::ExtractMyNodeBasedValues(ele, gradphinp_,*gradphinp);
      if (curvature == NULL) dserror("no curvature has been computed!");
      DRT::UTILS::ExtractMyNodeBasedValues(ele, curv_,*curvature);
#ifndef COMBUST_NORMAL_ENRICHMENT
    }
#endif
  }
}


/*----------------------------------------------------------------------*
 | constructor                                              henke 04/10 |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Combust3::MyStateSurface::MyStateSurface(
    const DRT::Discretization&             discretization,
    const std::vector<int>&                lm,
    const bool                             instationary,
    const bool                             genalpha,
    const bool                             gradphi,
    const DRT::ELEMENTS::Combust3*         ele,
    const Epetra_Vector*                   phinp
    ) :
      instationary_(instationary),
      genalpha_(genalpha)
{
  if (!genalpha_)
    DRT::UTILS::ExtractMyValues(*discretization.GetState("velnp"),velnp_,lm);
  if (instationary_)
  {
    if(genalpha_)
    {
      DRT::UTILS::ExtractMyValues(*discretization.GetState("velaf") ,velaf_ ,lm);
    }
  }

#ifdef DEBUG
  // check if this element is the first element on this processor
  // remark:
  // The SameAs-operation requires MPI communication between processors. Therefore it can only
  // be performed once (at the beginning) on each processor. Otherwise some processors would
  // wait to receive MPI information, but would never get it, because some processores are
  // already done with their element loop. This will cause a mean parallel bug!   henke 11.08.09
  if(ele->Id() == discretization.lRowElement(0)->Id())
  {
    // get map of this vector
    const Epetra_BlockMap& phimap = phinp->Map();
    // check, whether this map is still identical with the current node map in the discretization
    if (not phimap.SameAs(*discretization.NodeColMap())) dserror("node column map has changed!");
  }
#endif

  // extract local (element level) G-function values from global vector
  DRT::UTILS::ExtractMyNodeBasedValues(ele, phinp_, *phinp);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Combust3::DLMInfo::DLMInfo(const int nd, const int na)
{
  oldKaainv_ = Teuchos::rcp(new LINALG::SerialDenseMatrix(na,na,true));
  oldKad_ = Teuchos::rcp(new LINALG::SerialDenseMatrix(na,nd,true));
  oldfa_ = Teuchos::rcp(new LINALG::SerialDenseVector(na,true));
  stressdofs_ = Teuchos::rcp(new LINALG::SerialDenseVector(na,true));

  return;
}


