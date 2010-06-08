/*----------------------------------------------------------------------*/
/*!
\file cnst_1d_art.H

<pre>
Maintainer: Mahmoud Ismail
            ismail@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15268
</pre>
*/
/*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include <vector>
#include "cnst_1d_art.H"



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PAR::Cnst_1d_art::Cnst_1d_art(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: Parameter(matdata),
  viscosity_(matdata->GetDouble("VISCOSITY")),
  density_(matdata->GetDouble("DENS")),
  young_(matdata->GetDouble("YOUNG")),
  nue_(matdata->GetDouble("NUE")),
  diam_(matdata->GetDouble("DIAM")),
  th_(matdata->GetDouble("TH")),
  pext1_(matdata->GetDouble("PEXT1")),
  pext2_(matdata->GetDouble("PEXT2"))
{
}


MAT::Cnst_1d_artType MAT::Cnst_1d_artType::instance_;


DRT::ParObject* MAT::Cnst_1d_artType::Create( const std::vector<char> & data )
{
  MAT::Cnst_1d_art* cnst_art = new MAT::Cnst_1d_art();
  cnst_art->Unpack(data);
  return cnst_art;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::Cnst_1d_art::Cnst_1d_art()
  : params_(NULL)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::Cnst_1d_art::Cnst_1d_art(MAT::PAR::Cnst_1d_art* params)
  : params_(params)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::Cnst_1d_art::Pack(vector<char>& data) const
{
  data.resize(0);

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);

  // matid
  int matid = -1;
  if (params_ != NULL) matid = params_->Id();  // in case we are in post-process mode
  AddtoPack(data,matid);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::Cnst_1d_art::Unpack(const vector<char>& data)
{
  vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");

  // matid
  int matid;
  ExtractfromPack(position,data,matid);
  // in post-process mode we do not have any instance of DRT::Problem
  if (DRT::Problem::NumInstances() > 0)
  {
    const int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();
    MAT::PAR::Parameter* mat = DRT::Problem::Instance(probinst)->Materials()->ParameterById(matid);
    if (mat->Type() == MaterialType())
      params_ = static_cast<MAT::PAR::Cnst_1d_art*>(mat);
    else
      dserror("Type of parameter material %d does not fit to calling type %d", mat->Type(), MaterialType());
  }
  else
  {
    params_ = NULL;
  }

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",data.size(),position);
}

#endif
