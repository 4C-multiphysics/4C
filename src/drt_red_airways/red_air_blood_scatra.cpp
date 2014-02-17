/*!----------------------------------------------------------------------
\file red_inter_acinar_dep.cpp
\brief

<pre>
Maintainer: Mahmoud Ismail
            ismail@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15268
            </pre>

*----------------------------------------------------------------------*/

#include "red_airway.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_linedefinition.H"

using namespace DRT::UTILS;

DRT::ELEMENTS::RedAirBloodScatraType DRT::ELEMENTS::RedAirBloodScatraType::instance_;


DRT::ParObject* DRT::ELEMENTS::RedAirBloodScatraType::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::RedAirBloodScatra* object = new DRT::ELEMENTS::RedAirBloodScatra(-1,-1);
  object->Unpack(data);
  return object;
}


Teuchos::RCP<DRT::Element>
DRT::ELEMENTS::RedAirBloodScatraType::Create( const std::string eletype,
                                              const std::string eledistype,
                                              const int id,
                                              const int owner )
{
  if ( eletype=="RED_AIR_BLOOD_SCATRA" )
  {
    Teuchos::RCP<DRT::Element> ele =  Teuchos::rcp(new DRT::ELEMENTS::RedAirBloodScatra(id,owner));
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::RedAirBloodScatraType::Create( const int id, const int owner )
{
  Teuchos::RCP<DRT::Element> ele =  Teuchos::rcp(new DRT::ELEMENTS::RedAirBloodScatra(id,owner));
  return ele;
}


void DRT::ELEMENTS::RedAirBloodScatraType::SetupElementDefinition( std::map<std::string,std::map<std::string,DRT::INPUT::LineDefinition> > & definitions )
{
  std::map<std::string,DRT::INPUT::LineDefinition>& defs = definitions["RED_AIR_BLOOD_SCATRA"];

  defs["LINE2"]
    .AddIntVector("LINE2",2)
    .AddNamedDouble("DiffusionCoefficient")
    .AddNamedDouble("WallThickness")
    .AddNamedDouble("PercentageOfDiffusionArea");
    ;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                           ismail 05/13|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::RedAirBloodScatra::RedAirBloodScatra(int id, int owner) :
DRT::Element(id,owner),
data_()
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                      ismail 05/13|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::RedAirBloodScatra::RedAirBloodScatra(const DRT::ELEMENTS::RedAirBloodScatra& old) :
DRT::Element(old),
data_(old.data_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of RedAirBloodScatra and return pointer             |
 |  to it                                                      (public) |
 |                                                         ismail 05/13 |
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::RedAirBloodScatra::Clone() const
{
  DRT::ELEMENTS::RedAirBloodScatra* newelement = new DRT::ELEMENTS::RedAirBloodScatra(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                         ismail 05/13 |
 *----------------------------------------------------------------------*/
DRT::Element::DiscretizationType DRT::ELEMENTS::RedAirBloodScatra::Shape() const
{
  switch (NumNode())
  {
  case  2: return line2;
  case  3: return line3;
  default:
    dserror("unexpected number of nodes %d", NumNode());
  }
  return dis_none;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                         ismail 05/13 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::RedAirBloodScatra::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);

  // add base class Element
  Element::Pack(data);


  std::map<std::string,double>::const_iterator it;

  AddtoPack(data,(int)(elemParams_.size()));
  for (it = elemParams_.begin(); it!= elemParams_.end(); it++)
  {
    AddtoPack(data,it->first);
    AddtoPack(data,it->second);
  }

  AddtoPack(data,generation_);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                         ismail 05/13 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::RedAirBloodScatra::Unpack(const std::vector<char>& data)
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

  std::map<std::string,double> it;
  int n = 0;

  ExtractfromPack(position,data,n);

  for (int i = 0; i<n; i++)
  {
    std::string name;
    double val;
    ExtractfromPack(position,data,name);
    ExtractfromPack(position,data,val);
    elemParams_[name] = val;
  }

  // extract generation
  ExtractfromPack(position,data,generation_);

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);

  return;
}


/*----------------------------------------------------------------------*
 |  dtor (public)                                           ismail 05/13|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::RedAirBloodScatra::~RedAirBloodScatra()
{
  return;
}


/*----------------------------------------------------------------------*
 |  print this element (public)                             ismail 05/13|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::RedAirBloodScatra::Print(std::ostream& os) const
{
  os << "RedAirBloodScatra ";
  Element::Print(os);

  return;
}

/*----------------------------------------------------------------------*
 |  Return names of visualization data                     ismail 05/13 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::RedAirBloodScatra::VisNames(std::map<std::string,int>& names)
{
  // Put the owner of this element into the file (use base class method for this)
  DRT::Element::VisNames(names);

#if 0
  // see whether we have additional data for visualization in our container
  std::ostringstream temp;
  temp << 1;

  // in flow of volumetric flow profile
  string name = "flow_in";
  names.insert(std::pair<string,int>(name,1));

  // out flow of volumetric flow profile
  name = "flow_out";
  names.insert(std::pair<string,int>(name,1));
#endif

  return;
}

/*----------------------------------------------------------------------*
 |  Return visualization data (public)                     ismail 02/10 |
 *----------------------------------------------------------------------*/
bool DRT::ELEMENTS::RedAirBloodScatra::VisData(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if(DRT::Element::VisData(name,data))
    return true;

  return false;
}



/*----------------------------------------------------------------------*
 |  Get element parameters (public)                        ismail 04/10 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::RedAirBloodScatra::getParams(std::string name, double & var)
{

  std::map<std::string,double>::iterator it;
  it = elemParams_.find(name);
  if (it == elemParams_.end())
  {
    dserror ("[%s] is not found with in the element variables",name.c_str());
    exit(1);
  }
  var = elemParams_[name];

}

/*----------------------------------------------------------------------*
 |  Get element parameters (public)                        ismail 03/11 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::RedAirBloodScatra::getParams(std::string name, int & var)
{

  if (name == "Generation")
  {
    var = generation_;
  }
  else
  {
    dserror ("[%s] is not found with in the element INT variables",name.c_str());
    exit(1);
  }

}

/*----------------------------------------------------------------------*
 |  get vector of lines              (public)              ismail  02/13|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<DRT::Element> > DRT::ELEMENTS::RedAirBloodScatra::Lines()
{
  // do NOT store line or surface elements inside the parent element
  // after their creation.
  // Reason: if a Redistribute() is performed on the discretization,
  // stored node ids and node pointers owned by these boundary elements might
  // have become illegal and you will get a nice segmentation fault ;-)

  // so we have to allocate new line elements:

  if (NumLine()>1) // 1D boundary element and 2D/3D parent element
  {
    dserror("RED_AIRWAY element must have one and only one line");
    exit(1);
  }
  else if (NumLine()==1) // 1D boundary element and 1D parent element -> body load (calculated in evaluate)
  {
    // 1D (we return the element itself)
    std::vector<Teuchos::RCP<Element> > lines(1);
    lines[0]= Teuchos::rcp(this, false);
    return lines;
  }
  else
  {
    dserror("Lines() does not exist for points ");
    return DRT::Element::Surfaces();
  }
}
