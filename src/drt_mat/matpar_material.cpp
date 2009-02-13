/*======================================================================*/
/*!
\file drt_condition.cpp

<pre>
-------------------------------------------------------------------------
                 BACI finite element library subsystem
            Copyright (2008) Technical University of Munich
              
Under terms of contract T004.008.000 there is a non-exclusive license for use
of this work by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library is proprietary software. It must not be published, distributed, 
copied or altered in any form or any media without written permission
of the copyright holder. It may be used under terms and conditions of the
above mentioned license by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library may solemnly used in conjunction with the BACI contact library
for purposes described in the above mentioned contract.

This library contains and makes use of software copyrighted by Sandia Corporation
and distributed under LGPL licence. Licensing does not apply to this or any
other third party software used here.

Questions? Contact Dr. Michael W. Gee (gee@lnm.mw.tum.de) 
                   or
                   Prof. Dr. Wolfgang A. Wall (wall@lnm.mw.tum.de)

http://www.lnm.mw.tum.de                   

-------------------------------------------------------------------------
</pre>

\brief A condition of any kind

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>
*/

/*----------------------------------------------------------------------*/
/* macros */
#ifdef CCADISCRET

/*----------------------------------------------------------------------*/
/* headers */
#include "matpar_material.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PAR::Material::Material(
  const int id, 
  const INPAR::MAT::MaterialType type,
  const std::string name
  )
: Container(),
  id_(id),
  type_(type),
  name_(name),
  comm_(Teuchos::null),
  params_(NULL)
{
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PAR::Material::Material() 
: Container(),
  id_(-1),
  type_(INPAR::MAT::m_none),
  name_(""),
  comm_(Teuchos::null),
  params_(NULL)
{
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PAR::Material::Material(
  const MAT::PAR::Material& old
  )
: Container(old),
  id_(old.id_),
  type_(old.type_),
  comm_(old.comm_),
  params_(old.params_)
{
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PAR::Material::~Material()
{
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ostream& operator << (std::ostream& os, const MAT::PAR::Material& cond)
{
  cond.Print(os);
  return os;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::PAR::Material::Print(std::ostream& os) const
{
  os << "MAT " << Id()
     << " " << Name()
     << " :: ";

  DRT::Container::Print(os);

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::PAR::Material::Pack(std::vector<char>& data) const
{
  data.resize(0);

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class container
  std::vector<char> basedata;
  DRT::Container::Pack(basedata);
  AddtoPack(data,basedata);
  // id_
  AddtoPack(data,id_);
  // type_
  AddtoPack<INPAR::MAT::MaterialType>(data,type_);
  // name_
  AddtoPack(data,name_);

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::PAR::Material::Unpack(const std::vector<char>& data)
{
  int position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // extract base class Container
  std::vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  DRT::Container::Unpack(basedata);
  // id_
  ExtractfromPack(position,data,id_);
  // type_
  ExtractfromPack<INPAR::MAT::MaterialType>(position,data,type_);
  // name_
  ExtractfromPack(position,data,name_);

  if (position != (int)data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}


/*----------------------------------------------------------------------*/
#endif  // #ifdef CCADISCRET
