//-----------------------------------------------------------------------
/*!
\file ale2.cpp

<pre>

</pre>
*/
//-----------------------------------------------------------------------
#ifdef D_ALE
#ifdef CCADISCRET

#include "ale2.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_dserror.H"

using namespace DRT::UTILS;


DRT::ELEMENTS::Ale2::Ale2(int id, int owner)
  : DRT::Element(id,element_ale2,owner),
    data_()
{
  lines_.resize(0);
  lines_.resize(0);
}


DRT::ELEMENTS::Ale2::Ale2(const DRT::ELEMENTS::Ale2& old)
  : DRT::Element(old),
    data_(old.data_),
    lines_(old.lines_),
    lineptrs_(old.lineptrs_)
{
  return;
}


DRT::Element* DRT::ELEMENTS::Ale2::Clone() const
{
  DRT::ELEMENTS::Ale2* newelement = new DRT::ELEMENTS::Ale2(*this);
  return newelement;
}


DRT::Element::DiscretizationType DRT::ELEMENTS::Ale2::Shape() const
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
  }
  return dis_none;
}


void DRT::ELEMENTS::Ale2::Pack(vector<char>& data) const
{
  data.resize(0);

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class Element
  vector<char> basedata(0);
  Element::Pack(basedata);
  AddtoPack(data,basedata);
  // data_
  vector<char> tmp(0);
  data_.Pack(tmp);
  AddtoPack(data,tmp);
}


void DRT::ELEMENTS::Ale2::Unpack(const vector<char>& data)
{
  int position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // extract base class Element
  vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  Element::Unpack(basedata);
  // data_
  vector<char> tmp(0);
  ExtractfromPack(position,data,tmp);
  data_.Unpack(tmp);

  if (position != (int)data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
}


DRT::ELEMENTS::Ale2::~Ale2()
{
}


void DRT::ELEMENTS::Ale2::Print(ostream& os) const
{
  os << "Ale2 ";
  Element::Print(os);
  cout << endl;
  cout << data_;
  return;
}


RefCountPtr<DRT::ElementRegister> DRT::ELEMENTS::Ale2::ElementRegister() const
{
  return rcp(new DRT::ELEMENTS::Ale2Register(Type()));
}


//
// get vector of lines
//
DRT::Element** DRT::ELEMENTS::Ale2::Lines()
{
  // once constructed do not reconstruct again
  // make sure they exist
  if ((int)lines_.size()    == NumLine() &&
      (int)lineptrs_.size() == NumLine() &&
      dynamic_cast<DRT::ELEMENTS::Ale2Line*>(lineptrs_[0]) )
    return (DRT::Element**)(&(lineptrs_[0]));
  
  // so we have to allocate new line elements
  DRT::UTILS::ElementBoundaryFactory<Ale2Line,Ale2>(false,lines_,lineptrs_,this);

  return (DRT::Element**)(&(lineptrs_[0]));
}


DRT::Element** DRT::ELEMENTS::Ale2::Surfaces()
{
  surface_.resize(1);
  surface_[0] = this; //points to Ale2 element itself
  return &surface_[0];
}


GaussRule2D DRT::ELEMENTS::Ale2::getOptimalGaussrule(const DiscretizationType& distype)
{
    GaussRule2D rule = intrule2D_undefined;
    switch (distype)
    {
    case quad4:
        rule = intrule_quad_4point;
        break;
    case quad8: case quad9:
        rule = intrule_quad_9point;
        break;
    case tri3:
        rule = intrule_tri_3point;
        break;
    case tri6:
        rule = intrule_tri_6point;
        break;
    default: 
        dserror("unknown number of nodes for gaussrule initialization");
  }
  return rule;
}

//=======================================================================
//=======================================================================
//=======================================================================
//=======================================================================


DRT::ELEMENTS::Ale2Register::Ale2Register(DRT::Element::ElementType etype)
  : ElementRegister(etype)
{
}


DRT::ELEMENTS::Ale2Register::Ale2Register(const DRT::ELEMENTS::Ale2Register& old)
  : ElementRegister(old)
{
}


DRT::ELEMENTS::Ale2Register* DRT::ELEMENTS::Ale2Register::Clone() const
{
  return new DRT::ELEMENTS::Ale2Register(*this);
}


void DRT::ELEMENTS::Ale2Register::Pack(vector<char>& data) const
{
  data.resize(0);

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class ElementRegister
  vector<char> basedata(0);
  ElementRegister::Pack(basedata);
  AddtoPack(data,basedata);
}


void DRT::ELEMENTS::Ale2Register::Unpack(const vector<char>& data)
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
}


DRT::ELEMENTS::Ale2Register::~Ale2Register()
{
}


void DRT::ELEMENTS::Ale2Register::Print(ostream& os) const
{
  os << "Ale2Register ";
  ElementRegister::Print(os);
}


#endif
#endif
