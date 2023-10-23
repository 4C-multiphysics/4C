/*----------------------------------------------------------------------------*/
/*! \file
\brief Line definitions for wall1 element.

\level 1


*/
/*---------------------------------------------------------------------------*/

#include "baci_lib_discret.H"
#include "baci_linalg_utils_sparse_algebra_math.H"
#include "baci_utils_exceptions.H"
#include "baci_w1.H"


DRT::ELEMENTS::Wall1LineType DRT::ELEMENTS::Wall1LineType::instance_;

DRT::ELEMENTS::Wall1LineType& DRT::ELEMENTS::Wall1LineType::Instance() { return instance_; }

/*----------------------------------------------------------------------*
 |  ctor (public)                                            mgit 03/07|
  *----------------------------------------------------------------------*/
DRT::ELEMENTS::Wall1Line::Wall1Line(int id, int owner, int nnode, const int* nodeids,
    DRT::Node** nodes, DRT::ELEMENTS::Wall1* parent, const int lline)
    : DRT::FaceElement(id, owner)
{
  SetNodeIds(nnode, nodeids);
  BuildNodalPointers(nodes);
  SetParentMasterElement(parent, lline);
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       mgit 03/07|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Wall1Line::Wall1Line(const DRT::ELEMENTS::Wall1Line& old) : DRT::FaceElement(old)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance return pointer to it               (public) |
 |                                                            mgit 03/07 |
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::Wall1Line::Clone() const
{
  DRT::ELEMENTS::Wall1Line* newelement = new DRT::ELEMENTS::Wall1Line(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                          farah 02/14 |
 *----------------------------------------------------------------------*/
DRT::Element::DiscretizationType DRT::ELEMENTS::Wall1Line::Shape() const
{
  DRT::Element::DiscretizationType distype_line = DRT::Element::DiscretizationType::dis_none;

  switch (ParentMasterElement()->Shape())
  {
    case DRT::Element::DiscretizationType::tri3:
    {
      distype_line = DRT::Element::DiscretizationType::line2;
      break;
    }
    case DRT::Element::DiscretizationType::tri6:
    {
      distype_line = DRT::Element::DiscretizationType::line3;
      break;
    }
    case DRT::Element::DiscretizationType::quad4:
    {
      distype_line = DRT::Element::DiscretizationType::line2;
      break;
    }
    case DRT::Element::DiscretizationType::quad8:
    {
      distype_line = DRT::Element::DiscretizationType::line3;
      break;
    }
    case DRT::Element::DiscretizationType::quad9:
    {
      distype_line = DRT::Element::DiscretizationType::line3;
      break;
    }
    case DRT::Element::DiscretizationType::nurbs4:
    {
      distype_line = DRT::Element::DiscretizationType::nurbs2;
      break;
    }
    case DRT::Element::DiscretizationType::nurbs9:
    {
      distype_line = DRT::Element::DiscretizationType::nurbs3;
      break;
    }
    default:
      dserror("DRT::ELEMENTS::Wall1Line::Wall1Line: Unknown parent shape!");
  }

  return distype_line;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            mgit 03/07 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Wall1Line::Pack(DRT::PackBuffer& data) const
{
  dserror("this Wall1Line element does not support communication");

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            mgit 03/07 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Wall1Line::Unpack(const std::vector<char>& data)
{
  dserror("this line element does not support communication");
  return;
}



/*----------------------------------------------------------------------*
 |  print this element (public)                              mgit 03/07|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Wall1Line::Print(std::ostream& os) const
{
  os << "Wall1Line ";
  Element::Print(os);
  return;
}
