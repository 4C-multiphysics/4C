/*!----------------------------------------------------------------------
\file wall1_nurbs.cpp

<pre>
Maintainer: Peter Gamnitzer
            gamnitzer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15235
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "wall1_nurbs.H"

DRT::ELEMENTS::NURBS::Wall1NurbsType DRT::ELEMENTS::NURBS::Wall1NurbsType::instance_;


DRT::ParObject* DRT::ELEMENTS::NURBS::Wall1NurbsType::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::NURBS::Wall1Nurbs* object = new DRT::ELEMENTS::NURBS::Wall1Nurbs(-1,-1);
  object->Unpack(data);
  return object;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::NURBS::Wall1NurbsType::Create( const string eletype,
                                                                         const string eledistype,
                                                                         const int id,
                                                                         const int owner )
{
  if ( eletype=="WALL" )
  {
    if(eledistype=="NURBS4" || eledistype=="NURBS9")
    {
      return rcp(new DRT::ELEMENTS::NURBS::Wall1Nurbs(id,owner));
    }
  }
  return Teuchos::null;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::NURBS::Wall1NurbsType::Create( const int id, const int owner )
{
  return rcp(new DRT::ELEMENTS::NURBS::Wall1Nurbs(id,owner));
}

void DRT::ELEMENTS::NURBS::Wall1NurbsType::NodalBlockInformation( DRT::Element * dwele, int & numdf, int & dimns, int & nv, int & np )
{
  numdf = 2;
  dimns = 3;
  nv = 2;
}

void DRT::ELEMENTS::NURBS::Wall1NurbsType::ComputeNullSpace( DRT::Discretization & dis, std::vector<double> & ns, const double * x0, int numdf, int dimns )
{
  DRT::UTILS::ComputeStructure2DNullSpace( dis, ns, x0, numdf, dimns );
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            gammi 02/09|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::NURBS::Wall1Nurbs::Wall1Nurbs(int id, int owner) :
DRT::ELEMENTS::Wall1::Wall1(id,owner)
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       gammi 02/09|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::NURBS::Wall1Nurbs::Wall1Nurbs
(const DRT::ELEMENTS::NURBS::Wall1Nurbs& old) :
DRT::ELEMENTS::Wall1::Wall1(old)
{
  return;
}


/*----------------------------------------------------------------------*
 |  dtor (public)                                            gammi 02/09|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::NURBS::Wall1Nurbs::~Wall1Nurbs()
{
  return;
}


/*----------------------------------------------------------------------*
 |  Deep copy this instance of Wall1 and return pointer to it (public) |
 |                                                          gammi 05/09|
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::NURBS::Wall1Nurbs::Clone() const
{
  DRT::ELEMENTS::NURBS::Wall1Nurbs* newelement = new DRT::ELEMENTS::NURBS::Wall1Nurbs(*this);
  return newelement;
}


/*----------------------------------------------------------------------*
 |  print this element (public)                              gammi 02/09|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::NURBS::Wall1Nurbs::Print(ostream& os) const
{
  os << "Wall1Nurbs ";
  Element::Print(os);
  return;
}


/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                          gammi 02/09 |
 *----------------------------------------------------------------------*/
DRT::Element::DiscretizationType DRT::ELEMENTS::NURBS::Wall1Nurbs::Shape() const
{
  switch (NumNode())
  {
  case  4: return nurbs4;
  case  9: return nurbs9;
  default:
    dserror("unexpected number of nodes %d", NumNode());
  }

  return dis_none;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                             gammi 05/09|
 *----------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::NURBS::Wall1Nurbs::Lines()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:
  return DRT::UTILS::ElementBoundaryFactory<Wall1Line,Wall1>(DRT::UTILS::buildLines,this);
}

/*----------------------------------------------------------------------*
 |  get vector of surfaces (public)                          gammi 05/09|
 *----------------------------------------------------------------------*/
vector<RCP<DRT::Element> >  DRT::ELEMENTS::NURBS::Wall1Nurbs::Surfaces()
{
  vector<RCP<Element> > surfaces(1);
  surfaces[0]= rcp(this, false);
  return surfaces;
}


#endif // CCADISCRET
