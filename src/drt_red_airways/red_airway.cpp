
/*!----------------------------------------------------------------------
\file red_airway.cpp
\brief

<pre>
Maintainer: Mahmoud Ismail
            ismail@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15268
            </pre>

*----------------------------------------------------------------------*/
#ifdef D_RED_AIRWAYS
#ifdef CCADISCRET

#include "red_airway.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_linedefinition.H"

using namespace DRT::UTILS;

DRT::ELEMENTS::RedAirwayType DRT::ELEMENTS::RedAirwayType::instance_;


DRT::ParObject* DRT::ELEMENTS::RedAirwayType::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::RedAirway* object = new DRT::ELEMENTS::RedAirway(-1,-1);
  object->Unpack(data);
  return object;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::RedAirwayType::Create( const string eletype,
                                                            const string eledistype,
                                                            const int id,
                                                            const int owner )
{
  if ( eletype=="RED_AIRWAY" )
  {
    Teuchos::RCP<DRT::Element> ele =  rcp(new DRT::ELEMENTS::RedAirway(id,owner));
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::RedAirwayType::Create( const int id, const int owner )
{
  Teuchos::RCP<DRT::Element> ele =  rcp(new DRT::ELEMENTS::RedAirway(id,owner));
  return ele;
}


void DRT::ELEMENTS::RedAirwayType::SetupElementDefinition( std::map<std::string,std::map<std::string,DRT::INPUT::LineDefinition> > & definitions )
{
  std::map<std::string,DRT::INPUT::LineDefinition>& defs = definitions["RED_AIRWAY"];

  defs["LINE2"]
    .AddIntVector("LINE2",2)
    .AddNamedInt("MAT")
    .AddNamedString("TYPE")
    .AddNamedDouble("WallCompliance")
    .AddNamedDouble("AirCompliance")
    .AddNamedDouble("WallThickness")
    .AddNamedDouble("Area")
    ;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                           ismail 01/10|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::RedAirway::RedAirway(int id, int owner) :
DRT::Element(id,owner),
data_()
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                      ismail 01/10|
 |  id             (in)  this element's global id                       |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::RedAirway::RedAirway(const DRT::ELEMENTS::RedAirway& old) :
DRT::Element(old),
data_(old.data_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  Deep copy this instance of RedAirway and return pointer             |
 |  to it                                                      (public) |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::RedAirway::Clone() const
{
  DRT::ELEMENTS::RedAirway* newelement = new DRT::ELEMENTS::RedAirway(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
DRT::Element::DiscretizationType DRT::ELEMENTS::RedAirway::Shape() const
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
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::RedAirway::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);

  // add base class Element
  Element::Pack(data);

  AddtoPack(data,elemType_);

  map<std::string,double>::const_iterator it;
  AddtoPack(data,(int)(elemVars_.size()));
  for (it = elemVars_.begin(); it!= elemVars_.end(); it++)
  {
    AddtoPack(data,it->first);
    AddtoPack(data,it->second);
  }

  AddtoPack(data,(int)(elemParams_.size()));
  for (it = elemParams_.begin(); it!= elemParams_.end(); it++)
  {
    AddtoPack(data,it->first);
    AddtoPack(data,it->second);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                         ismail 01/10 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::RedAirway::Unpack(const vector<char>& data)
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

  ExtractfromPack(position,data,elemType_);

  map<std::string,double> it;
  int n = 0;
  ExtractfromPack(position,data,n);

  for (int i = 0; i<n; i++)
  {
    std::string name;
    double val;
    ExtractfromPack(position,data,name);
    ExtractfromPack(position,data,val);
    elemVars_[name] = val;
  }

  ExtractfromPack(position,data,n);

  for (int i = 0; i<n; i++)
  {
    std::string name;
    double val;
    ExtractfromPack(position,data,name);
    ExtractfromPack(position,data,val);
    elemParams_[name] = val;
  }

  //  cout<<"Var size: "<<elemVars_.size();
  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);

  return;
}


/*----------------------------------------------------------------------*
 |  dtor (public)                                           ismail 01/10|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::RedAirway::~RedAirway()
{
  return;
}


/*----------------------------------------------------------------------*
 |  print this element (public)                             ismail 01/10|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::RedAirway::Print(ostream& os) const
{
  os << "RedAirway ";
  Element::Print(os);

  return;
}

/*----------------------------------------------------------------------*
 |  Return names of visualization data                     ismail 01/10 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::RedAirway::VisNames(map<string,int>& names)
{
  // Put the owner of this element into the file (use base class method for this)
  DRT::Element::VisNames(names);

#if 0
  // see whether we have additional data for visualization in our container
  ostringstream temp;
  temp << 1;

  // in flow of volumetric flow profile
  string name = "flow_in";
  names.insert(pair<string,int>(name,1));

  // out flow of volumetric flow profile
  name = "flow_out";
  names.insert(pair<string,int>(name,1));
#endif

  return;
}

/*----------------------------------------------------------------------*
 |  Return visualization data (public)                     ismail 02/10 |
 *----------------------------------------------------------------------*/
bool DRT::ELEMENTS::RedAirway::VisData(const string& name, vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if(DRT::Element::VisData(name,data))
    return true;

#if 0
  if ( (name == "flow_in") )
  {
    if ((int)data.size()!=1) dserror("size mismatch");
    const double value = elemVars_[name];
    data[0] = value;
    return true;
  }
  else if ( (name == "flow_out") )
  {
    if ((int)data.size()!=1) dserror("size mismatch");
    const double value = elemVars_[name];
    data[0] = value;
    return true;
  }
  else
  {
    return false;
  }
#endif
}

/*----------------------------------------------------------------------*
 |  Get element variable  (public)                         ismail 04/10 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::RedAirway::getVars(std::string name, double & var)
{

  map<std::string,double>::iterator it;
  it = elemVars_.find(name);
  if (it == elemVars_.end())
  {
    dserror ("[%s] is not found with in the element variables",name.c_str());
    exit(1);
  }
  var = elemVars_[name];

}

/*----------------------------------------------------------------------*
 |  Set element variable  (public)                         ismail 04/10 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::RedAirway::setVars(std::string name, double var)
{

  map<std::string,double>::iterator it;
  it = elemVars_.find(name);
  if (it == elemVars_.end())
  {
    dserror ("[%s] is not found with in the element variables",name.c_str());
    exit(1);
  }
  elemVars_[name] = var;

}


/*----------------------------------------------------------------------*
 |  Get element parameters (public)                        ismail 04/10 |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::RedAirway::getParams(std::string name, double & var)
{

  map<std::string,double>::iterator it;
  it = elemParams_.find(name);
  if (it == elemParams_.end())
  {
    dserror ("[%s] is not found with in the element variables",name.c_str());
    exit(1);
  }
  var = elemParams_[name];

}

#endif  // #ifdef CCADISCRET
#endif  // #ifdef D_RED_AIRWAYS
