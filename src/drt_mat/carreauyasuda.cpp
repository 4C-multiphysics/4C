/*----------------------------------------------------------------------*/
/*!
\file carreauyasuda.cpp

<pre>
Maintainer: Ursula Mayer
			mayer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15257
</pre>
*/
/*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include <vector>

#include "carreauyasuda.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_mat/matpar_bundle.H"

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PAR::CarreauYasuda::CarreauYasuda(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: Parameter(matdata),
  nu_0_(matdata->GetDouble("NU_0")),
  nu_inf_(matdata->GetDouble("NU_INF")),
  lambda_(matdata->GetDouble("LAMBDA")),
  a_param_(matdata->GetDouble("APARAM")),
  b_param_(matdata->GetDouble("BPARAM")),
  density_(matdata->GetDouble("DENSITY"))
{
}

Teuchos::RCP<MAT::Material> MAT::PAR::CarreauYasuda::CreateMaterial()
{
  return Teuchos::rcp(new MAT::CarreauYasuda(this));
}


MAT::CarreauYasudaType MAT::CarreauYasudaType::instance_;


DRT::ParObject* MAT::CarreauYasudaType::Create( const std::vector<char> & data )
{
  MAT::CarreauYasuda* carYas = new MAT::CarreauYasuda();
  carYas->Unpack(data);
  return carYas;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::CarreauYasuda::CarreauYasuda()
  : params_(NULL)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::CarreauYasuda::CarreauYasuda(MAT::PAR::CarreauYasuda* params)
  : params_(params)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::CarreauYasuda::Pack(DRT::PackBuffer& data) const
{
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
void MAT::CarreauYasuda::Unpack(const vector<char>& data)
{
  vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");

  // matid and recover params_
  int matid;
  ExtractfromPack(position,data,matid);
  if (DRT::Problem::Instance()->Materials() != Teuchos::null)
  {
    const int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();
    MAT::PAR::Parameter* mat = DRT::Problem::Instance(probinst)->Materials()->ParameterById(matid);
    if (mat->Type() == MaterialType())
      params_ = static_cast<MAT::PAR::CarreauYasuda*>(mat);
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
