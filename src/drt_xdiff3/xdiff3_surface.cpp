/*!----------------------------------------------------------------------
\file xfluid3_surface.cpp
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


DRT::ELEMENTS::XDiff3SurfaceType DRT::ELEMENTS::XDiff3SurfaceType::instance_;


/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 01/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::XDiff3Surface::XDiff3Surface(
    int id,
    int owner,
    const int nnode,
    const int* nodeids,
    DRT::Node** nodes,
    DRT::ELEMENTS::XDiff3* parent,
    const int lsurface) :
DRT::Element(id,owner),
parent_(parent),
lsurface_(lsurface)
{
  SetNodeIds(nnode, nodeids);
  BuildNodalPointers(nodes);
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       mwgee 01/07|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::XDiff3Surface::XDiff3Surface(const DRT::ELEMENTS::XDiff3Surface& old) :
DRT::Element(old),
parent_(old.parent_),
lsurface_(old.lsurface_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it               (public) |
 |                                                            gee 01/07 |
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::XDiff3Surface::Clone() const
{
  DRT::ELEMENTS::XDiff3Surface* newelement = new DRT::ELEMENTS::XDiff3Surface(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                          u.kue 03/07 |
 *----------------------------------------------------------------------*/
DRT::Element::DiscretizationType DRT::ELEMENTS::XDiff3Surface::Shape() const
{
  switch (NumNode())
  {
  case 3: return tri3;
  case 4: return quad4;
  case 6: return tri6;
  case 8: return quad8;
  case 9: return quad9;
  default:
    dserror("unexpected number of nodes %d", NumNode());
  }
  return dis_none;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            gee 02/07 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::XDiff3Surface::Pack(std::vector<char>& data) const
{
  data.resize(0);
  dserror("this XDiff3Surface element does not support communication");

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            gee 02/07 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::XDiff3Surface::Unpack(const std::vector<char>& data)
{
  dserror("this XDiff3Surface element does not support communication");
  return;
}

/*----------------------------------------------------------------------*
 |  dtor (public)                                            mwgee 01/07|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::XDiff3Surface::~XDiff3Surface()
{
  return;
}


/*----------------------------------------------------------------------*
 |  print this element (public)                              mwgee 01/07|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::XDiff3Surface::Print(ostream& os) const
{
  os << "XDiff3Surface ";
  Element::Print(os);
  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                             gammi 04/07|
 *----------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::XDiff3Surface::Lines()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:
  return DRT::UTILS::ElementBoundaryFactory<XDiff3Line,XDiff3Surface>(DRT::UTILS::buildLines,this);
}


#endif  // #ifdef CCADISCRET
#endif // #ifdef D_FLUID3
