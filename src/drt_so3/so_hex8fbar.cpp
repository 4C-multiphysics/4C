/*!----------------------------------------------------------------------
\file so_hex8fbar.cpp
\brief

<pre>
Maintainer: Alexander Popp
            popp@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15264
</pre>

*----------------------------------------------------------------------*/
#ifdef D_SOLID3
#ifdef CCADISCRET

#include "so_hex8fbar.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_linedefinition.H"

DRT::ELEMENTS::So_hex8fbarType DRT::ELEMENTS::So_hex8fbarType::instance_;


DRT::ParObject* DRT::ELEMENTS::So_hex8fbarType::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::So_hex8fbar* object = new DRT::ELEMENTS::So_hex8fbar(-1,-1);
  object->Unpack(data);
  return object;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_hex8fbarType::Create( const string eletype,
                                                            const string eledistype,
                                                            const int id,
                                                            const int owner )
{
  if ( eletype=="SOLIDH8FBAR" )
  {
    Teuchos::RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::So_hex8fbar(id,owner));
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_hex8fbarType::Create( const int id, const int owner )
{
  Teuchos::RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::So_hex8fbar(id,owner));
  return ele;
}


void DRT::ELEMENTS::So_hex8fbarType::NodalBlockInformation( DRT::Element * dwele, int & numdf, int & dimns, int & nv, int & np )
{
  numdf = 3;
  dimns = 6;
  nv = 3;
  np = 0;
}

void DRT::ELEMENTS::So_hex8fbarType::ComputeNullSpace( DRT::Discretization & dis, std::vector<double> & ns, const double * x0, int numdf, int dimns )
{
  DRT::UTILS::ComputeStructure3DNullSpace( dis, ns, x0, numdf, dimns );
}

void DRT::ELEMENTS::So_hex8fbarType::SetupElementDefinition( std::map<std::string,std::map<std::string,DRT::INPUT::LineDefinition> > & definitions )
{
  std::map<std::string,DRT::INPUT::LineDefinition>& defs = definitions["SOLIDH8FBAR"];

  defs["HEX8"]
    .AddIntVector("HEX8",8)
    .AddNamedInt("MAT")
    .AddNamedString("KINTYP")
    ;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 07/10|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_hex8fbar::So_hex8fbar(int id, int owner) :
DRT::ELEMENTS::So_hex8(id,owner)
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                        popp 07/10|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_hex8fbar::So_hex8fbar(const DRT::ELEMENTS::So_hex8fbar& old) :
DRT::ELEMENTS::So_hex8(old)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Solid3 and return pointer to it (public) |
 |                                                            popp 07/10|
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::So_hex8fbar::Clone() const
{
  DRT::ELEMENTS::So_hex8fbar* newelement = new DRT::ELEMENTS::So_hex8fbar(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            popp 07/10|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_hex8fbar::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class So_hex8 Element
  DRT::ELEMENTS::So_hex8::Pack(data);

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            popp 07/10|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_hex8fbar::Unpack(const vector<char>& data)
{
  vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // extract base class So_hex8 Element
  vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  DRT::ELEMENTS::So_hex8::Unpack(basedata);

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}

/*----------------------------------------------------------------------*
 |  dtor (public)                                             popp 07/10|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_hex8fbar::~So_hex8fbar()
{
  return;
}

/*----------------------------------------------------------------------*
 |  print this element (public)                               popp 07/10|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_hex8fbar::Print(ostream& os) const
{
  os << "So_hex8fbar ";
  Element::Print(os);
  cout << endl;
  cout << data_;
  return;
}

#endif  // #ifdef CCADISCRET
#endif  // #ifdef D_SOLID3
