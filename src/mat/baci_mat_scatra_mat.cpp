/*----------------------------------------------------------------------*/
/*! \file
\brief scalar transport material

\level 1

*/
/*----------------------------------------------------------------------*/


#include <vector>
#include "baci_mat_scatra_mat.H"
#include "baci_lib_globalproblem.H"
#include "baci_mat_par_bundle.H"
#include "baci_comm_utils.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PAR::ScatraMat::ScatraMat(Teuchos::RCP<MAT::PAR::Material> matdata) : Parameter(matdata)
{
  // extract relevant communicator
  const Epetra_Comm& comm = DRT::Problem::Instance()->Materials()->GetReadFromProblem() == 0
                                ? *DRT::Problem::Instance()->GetCommunicators()->LocalComm()
                                : *DRT::Problem::Instance()->GetCommunicators()->SubComm();

  Epetra_Map dummy_map(1, 1, 0, comm);
  for (int i = first; i <= last; i++)
  {
    matparams_.push_back(Teuchos::rcp(new Epetra_Vector(dummy_map, true)));
  }
  matparams_.at(diff)->PutScalar(matdata->GetDouble("DIFFUSIVITY"));
  matparams_.at(reac)->PutScalar(matdata->GetDouble("REACOEFF"));
  matparams_.at(densific)->PutScalar(matdata->GetDouble("DENSIFICATION"));

  return;
}


Teuchos::RCP<MAT::Material> MAT::PAR::ScatraMat::CreateMaterial()
{
  return Teuchos::rcp(new MAT::ScatraMat(this));
}


MAT::ScatraMatType MAT::ScatraMatType::instance_;


DRT::ParObject* MAT::ScatraMatType::Create(const std::vector<char>& data)
{
  MAT::ScatraMat* scatra_mat = new MAT::ScatraMat();
  scatra_mat->Unpack(data);
  return scatra_mat;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::ScatraMat::ScatraMat() : params_(NULL) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::ScatraMat::ScatraMat(MAT::PAR::ScatraMat* params) : params_(params) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ScatraMat::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);

  // matid
  int matid = -1;
  if (params_ != NULL) matid = params_->Id();  // in case we are in post-process mode
  AddtoPack(data, matid);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::ScatraMat::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position, data, type);
  if (type != UniqueParObjectId())
    dserror(
        "wrong instance type data. type = %d, UniqueParObjectId()=%d", type, UniqueParObjectId());

  // matid and recover params_
  int matid;
  ExtractfromPack(position, data, matid);
  params_ = NULL;
  if (DRT::Problem::Instance()->Materials() != Teuchos::null)
    if (DRT::Problem::Instance()->Materials()->Num() != 0)
    {
      const int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();
      MAT::PAR::Parameter* mat =
          DRT::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
        params_ = static_cast<MAT::PAR::ScatraMat*>(mat);
      else
        dserror("Type of parameter material %d does not fit to calling type %d", mat->Type(),
            MaterialType());
    }

  if (position != data.size()) dserror("Mismatch in size of data %d <-> %d", data.size(), position);
}
