/*!----------------------------------------------------------------------
\file drt_elementregister.cpp
\brief

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "drt_elementregister.H"
#include "drt_dserror.H"


/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 11/06|
 *----------------------------------------------------------------------*/
DRT::ElementRegister::ElementRegister(DRT::Element::ElementType etype) :
ParObject(),
etype_(etype)
{
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       mwgee 11/06|
 *----------------------------------------------------------------------*/
DRT::ElementRegister::ElementRegister(const DRT::ElementRegister& old) :
ParObject(old),
etype_(old.etype_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  dtor (public)                                            mwgee 11/06|
 *----------------------------------------------------------------------*/
DRT::ElementRegister::~ElementRegister()
{
  return;
}


/*----------------------------------------------------------------------*
 |  clone (public)                                           mwgee 02/07|
 *----------------------------------------------------------------------*/
DRT::ElementRegister* DRT::ElementRegister::Clone() const
{
  DRT::ElementRegister* tmp = new DRT::ElementRegister(*this);
  return tmp;
}

/*----------------------------------------------------------------------*
 |  << operator                                              mwgee 11/06|
 *----------------------------------------------------------------------*/
ostream& operator << (ostream& os, const DRT::ElementRegister& eler)
{
  eler.Print(os); 
  return os;
}


/*----------------------------------------------------------------------*
 |  print element (public)                                   mwgee 11/06|
 *----------------------------------------------------------------------*/
void DRT::ElementRegister::Print(ostream& os) const
{
  os << "ElementRegister to element with type ";
  switch(Type())
  {
    case DRT::Element::element_shell8line:
      os << "Shell8Line ";
    break;
    case DRT::Element::element_shell8:
      os << "Shell8 ";
    break;
    case DRT::Element::element_none:
    default:
      dserror("Unknown type of element");
    break;
  }
  return;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            gee 02/07 |
 *----------------------------------------------------------------------*/
void DRT::ElementRegister::Pack(vector<char>& data) const
{
  data.resize(0);
  
  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add type of element
  AddtoPack(data,etype_);
  
  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            gee 02/07 |
 *----------------------------------------------------------------------*/
void DRT::ElementRegister::Unpack(const vector<char>& data)
{
  int position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  ExtractfromPack(position,data,etype_);
  
  if (position != (int)data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
} 

/*----------------------------------------------------------------------*
 |  Init the elements of a discretization                      (public) |
 |                                                            gee 02/07 |
 *----------------------------------------------------------------------*/
int DRT::ElementRegister::Initialize(DRT::Discretization& dis)
{
  // This is a base class dummy that does nothing.
  // It does not even print a message because it might become
  // heavily used by elements that do not need an initialize call
  return 0;
} 

#endif  // #ifdef CCADISCRET
