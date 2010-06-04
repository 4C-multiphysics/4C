/*!----------------------------------------------------------------------
\file xfluid3.cpp
\brief

<pre>
Maintainer: Axel Gerstenberger
            gerstenberger@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15236
</pre>

*----------------------------------------------------------------------*/
#ifdef D_FLUID3
#ifdef CCADISCRET

#include "xdiff3.H"
#include "../drt_lib/drt_utils.H"

using namespace DRT::UTILS;

DRT::ELEMENTS::XDiff3Type DRT::ELEMENTS::XDiff3Type::instance_;


DRT::ParObject* DRT::ELEMENTS::XDiff3Type::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::XDiff3* object = new DRT::ELEMENTS::XDiff3(-1,-1);
  object->Unpack(data);
  return object;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::XDiff3Type::Create( const string eletype,
                                                            const string eledistype,
                                                            const int id,
                                                            const int owner )
{
  if ( eletype=="XDIFF3" )
  {
    Teuchos::RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::XDiff3(id,owner));
    return ele;
  }
  return Teuchos::null;
}


DRT::ELEMENTS::XDiff3RegisterType DRT::ELEMENTS::XDiff3RegisterType::instance_;


DRT::ParObject* DRT::ELEMENTS::XDiff3RegisterType::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::XDiff3Register* object =
    new DRT::ELEMENTS::XDiff3Register(DRT::Element::element_xdiff3);
  object->Unpack(data);
  return object;
}


/*----------------------------------------------------------------------*/
// map to convert strings to actions (stabilization)
/*----------------------------------------------------------------------*/
map<string,DRT::ELEMENTS::XDiff3::StabilisationAction> DRT::ELEMENTS::XDiff3::stabstrtoact_;

/*----------------------------------------------------------------------*
 |  ctor (public)                                            gammi 02/08|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::XDiff3::XDiff3(int id, int owner) :
DRT::Element(id,element_xdiff3,owner),
eleDofManager_(Teuchos::null),
eleDofManager_uncondensed_(Teuchos::null),
output_mode_(false)
{
    return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       gammi 02/08|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::XDiff3::XDiff3(const DRT::ELEMENTS::XDiff3& old) :
DRT::Element(old),
eleDofManager_(old.eleDofManager_),
eleDofManager_uncondensed_(old.eleDofManager_uncondensed_),
output_mode_(old.output_mode_)
{
    return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of XDiff3 and return pointer to it (public)|
 |                                                          gammi 02/08 |
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::XDiff3::Clone() const
{
  return new DRT::ELEMENTS::XDiff3(*this);
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                          u.kue 03/07 |
 *----------------------------------------------------------------------*/
DRT::Element::DiscretizationType DRT::ELEMENTS::XDiff3::Shape() const
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
  case  0: return dis_none;
  default:
    dserror("unexpected number of nodes %d", NumNode());
  }
  return dis_none;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                          gammi 02/08 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::XDiff3::Pack(std::vector<char>& data) const
{
  data.resize(0);

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class Element
  vector<char> basedata(0);
  Element::Pack(basedata);
  AddtoPack(data,basedata);

  AddtoPack(data,output_mode_);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                          gammi 02/08 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::XDiff3::Unpack(const std::vector<char>& data)
{
  int position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  dsassert(type == UniqueParObjectId(), "wrong instance type data");
  // extract base class Element
  vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  Element::Unpack(basedata);

  ExtractfromPack(position,data,output_mode_);

  if (position != (int)data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}


/*----------------------------------------------------------------------*
 |  dtor (public)                                            gammi 02/08|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::XDiff3::~XDiff3()
{
  return;
}


/*----------------------------------------------------------------------*
 |  print this element (public)                              gammi 02/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::XDiff3::Print(ostream& os) const
{
  os << "XDiff3 ";
  if (output_mode_)
    os << "(outputmode=true)";
  Element::Print(os);
  cout << endl;
  return;
}


/*----------------------------------------------------------------------*
 |  allocate and return Fluid3Register (public)              mwgee 02/08|
 *----------------------------------------------------------------------*/
RCP<DRT::ElementRegister> DRT::ELEMENTS::XDiff3::ElementRegister() const
{
  return rcp(new DRT::ELEMENTS::XDiff3Register(Type()));
}


/*----------------------------------------------------------------------*
 |  get vector of lines              (public)                  gjb 03/07|
 *----------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::XDiff3::Lines()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:
  return DRT::UTILS::ElementBoundaryFactory<XDiff3Line,XDiff3>(DRT::UTILS::buildLines,this);
}


/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                            gjb 05/08|
 *----------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::XDiff3::Surfaces()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:
  return DRT::UTILS::ElementBoundaryFactory<XDiff3Surface,XDiff3>(DRT::UTILS::buildSurfaces,this);
}


/*----------------------------------------------------------------------*
 |  get vector of volumes (length 1) (public)                g.bau 03/07|
 *----------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::XDiff3::Volumes()
{
  vector<RCP<Element> > volumes(1);
  volumes[0]= rcp(this, false);
  return volumes;
}


/*----------------------------------------------------------------------*
 |  constructor
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::XDiff3::MyState::MyState(
    const DRT::Discretization&      discret,
    const std::vector<int>&         lm,
    const bool                      instat
    ) :
      instationary(instat)
{
  DRT::UTILS::ExtractMyValues(*discret.GetState("velnp"),velnp,lm);
  if (instat)
  {
    DRT::UTILS::ExtractMyValues(*discret.GetState("veln") ,veln ,lm);
    DRT::UTILS::ExtractMyValues(*discret.GetState("velnm"),velnm,lm);
    DRT::UTILS::ExtractMyValues(*discret.GetState("accn") ,accn ,lm);
  }
}



//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================

/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 12/06|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::XDiff3Register::XDiff3Register(DRT::Element::ElementType etype) :
ElementRegister(etype)
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       mwgee 12/06|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::XDiff3Register::XDiff3Register(
                               const DRT::ELEMENTS::XDiff3Register& old) :
ElementRegister(old)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it               (public) |
 |                                                            gee 12/06 |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::XDiff3Register* DRT::ELEMENTS::XDiff3Register::Clone() const
{
  return new DRT::ELEMENTS::XDiff3Register(*this);
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            gee 02/07 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::XDiff3Register::Pack(std::vector<char>& data) const
{
  data.resize(0);

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class ElementRegister
  vector<char> basedata(0);
  ElementRegister::Pack(basedata);
  AddtoPack(data,basedata);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            gee 02/07 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::XDiff3Register::Unpack(const std::vector<char>& data)
{
  int position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // base class ElementRegister
  vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  ElementRegister::Unpack(basedata);

  if (position != (int)data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}


/*----------------------------------------------------------------------*
 |  dtor (public)                                            mwgee 12/06|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::XDiff3Register::~XDiff3Register()
{
  return;
}

/*----------------------------------------------------------------------*
 |  print (public)                                           mwgee 12/06|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::XDiff3Register::Print(ostream& os) const
{
  os << "XDiff3Register ";
  ElementRegister::Print(os);
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::XDiff3::DLMInfo::DLMInfo(const int nd, const int na)
: oldKaainv_(LINALG::SerialDenseMatrix(na,na,true)),
  oldKad_(LINALG::SerialDenseMatrix(na,nd,true)),
  oldfa_(LINALG::SerialDenseVector(na,true)),
  stressdofs_(LINALG::SerialDenseVector(na,true))
{
  return;
}



#endif  // #ifdef CCADISCRET
#endif  // #ifdef D_FLUID3
