/*!----------------------------------------------------------------------
\file acoustic.cpp
\brief contains a density and a speed of sound to characterize all
       necessary properties for isotropic sound propagation in fluid
       like material
       example input line:
       MAT 1
<pre>
Maintainer: Svenja Schoeder
</pre>
*/
/*----------------------------------------------------------------------*/

#include <vector>
#include <Epetra_SerialDenseMatrix.h>
#include <Epetra_SerialDenseVector.h>
#include "acoustic.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_mat/matpar_bundle.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PAR::AcousticMat::AcousticMat(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: Parameter(matdata),
  c_(matdata->GetDouble("C")),
  density_(matdata->GetDouble("DENSITY"))
{
}


Teuchos::RCP<MAT::Material> MAT::PAR::AcousticMat::CreateMaterial()
{
  return Teuchos::rcp(new MAT::AcousticMat(this));
}

MAT::AcousticMatType MAT::AcousticMatType::instance_;

DRT::ParObject* MAT::AcousticMatType::Create( const std::vector<char> & data )
{
  MAT::AcousticMat* soundprop = new MAT::AcousticMat();
  soundprop->Unpack(data);
  return soundprop;
}

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
MAT::AcousticMat::AcousticMat()
  : params_(NULL)
{
}

/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
MAT::AcousticMat::AcousticMat(MAT::PAR::AcousticMat* params)
  : params_(params)
{
}


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void MAT::AcousticMat::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);

  // matid
  int matid = -1;
  if (params_ != NULL) matid = params_->Id();  // in case we are in post-process mode
  AddtoPack(data,matid);
}


/*----------------------------------------------------------------------*
 |                                                                      |
 *----------------------------------------------------------------------*/
void MAT::AcousticMat::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");

  // matid and recover params_
  int matid;
  ExtractfromPack(position,data,matid);
  params_ = NULL;
  if (DRT::Problem::Instance()->Materials() != Teuchos::null)
    if (DRT::Problem::Instance()->Materials()->Num() != 0)
    {
      const int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();
      MAT::PAR::Parameter* mat = DRT::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
        params_ = static_cast<MAT::PAR::AcousticMat*>(mat);
      else
        dserror("Type of parameter material %d does not fit to calling type %d", mat->Type(), MaterialType());
    }

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",data.size(),position);
}

