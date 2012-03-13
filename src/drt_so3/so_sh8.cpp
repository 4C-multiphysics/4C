/*!----------------------------------------------------------------------
\file so_sh8.cpp
\brief

<pre>
Maintainer: Moritz Frenzel
            frenzel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15240
</pre>

*----------------------------------------------------------------------*/
#ifdef D_SOLID3
#ifdef CCADISCRET

#include "so_sh8.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_linedefinition.H"
#include "so_hex8.H"


DRT::ELEMENTS::So_sh8Type DRT::ELEMENTS::So_sh8Type::instance_;


DRT::ParObject* DRT::ELEMENTS::So_sh8Type::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::So_sh8* object = new DRT::ELEMENTS::So_sh8(-1,-1);
  object->Unpack(data);
  return object;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_sh8Type::Create( const string eletype,
                                                            const string eledistype,
                                                            const int id,
                                                            const int owner )
{
  if ( eletype=="SOLIDSH8" )
  {
    Teuchos::RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::So_sh8(id,owner));
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::So_sh8Type::Create( const int id, const int owner )
{
  Teuchos::RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::So_sh8(id,owner));
  return ele;
}


void DRT::ELEMENTS::So_sh8Type::NodalBlockInformation( DRT::Element * dwele, int & numdf, int & dimns, int & nv, int & np )
{
  numdf = 3;
  dimns = 6;
  nv = 3;
}

void DRT::ELEMENTS::So_sh8Type::ComputeNullSpace( DRT::Discretization & dis, std::vector<double> & ns, const double * x0, int numdf, int dimns )
{
  DRT::UTILS::ComputeStructure3DNullSpace( dis, ns, x0, numdf, dimns );
}

void DRT::ELEMENTS::So_sh8Type::SetupElementDefinition( std::map<std::string,std::map<std::string,DRT::INPUT::LineDefinition> > & definitions )
{
  std::map<std::string,DRT::INPUT::LineDefinition>& defs = definitions["SOLIDSH8"];

  defs["HEX8"]
    .AddIntVector("HEX8",8)
    .AddNamedInt("MAT")
    .AddNamedString("KINEM")
    .AddNamedString("EAS")
    .AddNamedString("ANS")
    .AddNamedString("THICKDIR")
    .AddOptionalNamedDoubleVector("RAD",3)
    .AddOptionalNamedDoubleVector("AXI",3)
    .AddOptionalNamedDoubleVector("CIR",3)
    .AddOptionalNamedDouble("STRENGTH")
    ;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                              maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_sh8::So_sh8(int id, int owner) :
DRT::ELEMENTS::So_hex8(id,owner)
{
  thickdir_ = globx;
  nodes_rearranged_ = false;
  thickvec_.resize(3, 0.0);

  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                         maf 04/07|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_sh8::So_sh8(const DRT::ELEMENTS::So_sh8& old) :
DRT::ELEMENTS::So_hex8(old)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of Solid3 and return pointer to it (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::So_sh8::Clone() const
{
  DRT::ELEMENTS::So_sh8* newelement = new DRT::ELEMENTS::So_sh8(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_sh8::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class So_hex8 Element
  DRT::ELEMENTS::So_hex8::Pack(data);
  // thickdir
  AddtoPack(data,thickdir_);
  AddtoPack(data,thickvec_);
  AddtoPack(data,anstype_);
  AddtoPack(data,nodes_rearranged_);

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            maf 04/07 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_sh8::Unpack(const vector<char>& data)
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
  // thickdir
  thickdir_ = static_cast<ThicknessDirection>( ExtractInt(position,data) );
  ExtractfromPack(position,data,thickvec_);
  anstype_ = static_cast<ANSType>( ExtractInt(position,data) );
  nodes_rearranged_ = ExtractInt(position,data);

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}

/*----------------------------------------------------------------------*
 |  dtor (public)                                              maf 04/07|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::So_sh8::~So_sh8()
{
  return;
}


/*----------------------------------------------------------------------*
 |  print this element (public)                                maf 04/07|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_sh8::Print(ostream& os) const
{
  os << "So_sh8 ";
  Element::Print(os);
  cout << endl;
  cout << data_;
  return;
}


#endif  // #ifdef CCADISCRET
#endif  // #ifdef D_SOLID3
