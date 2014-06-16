/*!----------------------------------------------------------------------
\file bele3.cpp
\brief

<pre>
Maintainer: Raffaela Kruse
            kruse@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15249
</pre>

*----------------------------------------------------------------------*/

#include "bele3.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_utils_factory.H"
#include <sstream>


DRT::ELEMENTS::Bele3Type DRT::ELEMENTS::Bele3Type::instance_;


DRT::ParObject* DRT::ELEMENTS::Bele3Type::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::Bele3* object = new DRT::ELEMENTS::Bele3(-1,-1);
  object->Unpack(data);
  return object;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Bele3Type::Create( const std::string eletype,
                                                             const std::string eledistype,
                                                             const int id,
                                                             const int owner )
{
  std::size_t pos = eletype.rfind("_");
  if (pos!=std::string::npos)
  {
    if(eletype.substr(0, pos) == "BELE3")
    {
      std::istringstream is(eletype.substr(pos+1));

      int numdof = -1;
      is >> numdof;
      Teuchos::RCP<DRT::ELEMENTS::Bele3> ele = Teuchos::rcp(new DRT::ELEMENTS::Bele3(id,owner));
      ele->SetNumDofPerNode(numdof);
      return ele;
    }
  }

  return Teuchos::null;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Bele3Type::Create( const int id, const int owner )
{
  Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(new DRT::ELEMENTS::Bele3(id,owner));
  return ele;
}


void DRT::ELEMENTS::Bele3Type::NodalBlockInformation( DRT::Element * dwele, int & numdf, int & dimns, int & nv, int & np )
{
}

void DRT::ELEMENTS::Bele3Type::ComputeNullSpace( DRT::Discretization & dis, std::vector<double> & ns, const double * x0, int numdf, int dimns )
{
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Bele3LineType::Create( const int id, const int owner )
{
  //return Teuchos::rcp( new Bele3Line( id, owner ) );
  return Teuchos::null;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Bele3::Bele3(int id, int owner) :
DRT::Element(id,owner),
numdofpernode_(-1)
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Bele3::Bele3(const DRT::ELEMENTS::Bele3& old) :
DRT::Element(old),
numdofpernode_(old.numdofpernode_)
{
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::Bele3::Clone() const
{
  DRT::ELEMENTS::Bele3* newelement = new DRT::ELEMENTS::Bele3(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::Element::DiscretizationType DRT::ELEMENTS::Bele3::Shape() const
{
  switch (NumNode())
  {
  case  3: return tri3;
  case  4: return quad4;
  case  6: return tri6;
  case  8: return quad8;
  case  9: return quad9;
  default:
    dserror("unexpected number of nodes %d", NumNode());
    break;
  }
  return dis_none;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Bele3::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class Element
  Element::Pack(data);
  // numdofpernode_
  AddtoPack(data,numdofpernode_);

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Bele3::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  dsassert(type == UniqueParObjectId(), "wrong instance type data");
  // extract base class Element
  std::vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  Element::Unpack(basedata);
  // numdofpernode_
  numdofpernode_ = ExtractInt(position,data);

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",data.size(),position);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Bele3::~Bele3()
{
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Bele3::Print(std::ostream& os) const
{
  os << "Bele3_" << numdofpernode_ << " " << DRT::DistypeToString(Shape());
  Element::Print(os);
  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                               gjb 05/08|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<DRT::Element> > DRT::ELEMENTS::Bele3::Lines()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:
  return DRT::UTILS::ElementBoundaryFactory<Bele3Line,Bele3>(DRT::UTILS::buildLines,this);
}


/*----------------------------------------------------------------------*
 |  get vector of Surfaces (length 1) (public)               gammi 04/07|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<DRT::Element> > DRT::ELEMENTS::Bele3::Surfaces()
{
  std::vector<Teuchos::RCP<DRT::Element> > surfaces(1);
  surfaces[0]=Teuchos::rcp(this,false);
  return surfaces;
}


DRT::UTILS::GaussRule2D DRT::ELEMENTS::Bele3::getOptimalGaussrule(const DRT::Element::DiscretizationType& distype) const
{
  DRT::UTILS::GaussRule2D rule = DRT::UTILS::intrule2D_undefined;
    switch (distype)
    {
    case DRT::Element::quad4:
        rule = DRT::UTILS::intrule_quad_4point;
        break;
    case DRT::Element::quad8: case DRT::Element::quad9:
        rule = DRT::UTILS::intrule_quad_9point;
        break;
    case DRT::Element::tri3:
        rule = DRT::UTILS::intrule_tri_3point;
        break;
    case DRT::Element::tri6:
        rule = DRT::UTILS::intrule_tri_6point;
        break;
    default:
        dserror("unknown number of nodes for gaussrule initialization");
        break;
  }
  return rule;
}


