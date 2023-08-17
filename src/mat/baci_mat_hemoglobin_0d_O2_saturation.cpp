/*----------------------------------------------------------------------*/
/*! \file
\brief Gives relevant quantities of O2 saturation of blood (hemoglobin), used for scatra in reduced
dimensional airway elements framework (transport in elements and between air and blood)


\level 3
*/
/*----------------------------------------------------------------------*/


#include "baci_mat_hemoglobin_0d_O2_saturation.H"

#include "baci_lib_globalproblem.H"
#include "baci_mat_par_bundle.H"

#include <vector>



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PAR::Hemoglobin_0d_O2_saturation::Hemoglobin_0d_O2_saturation(
    Teuchos::RCP<MAT::PAR::Material> matdata)
    : Parameter(matdata),
      per_volume_blood_(matdata->GetDouble("PerVolumeBlood")),
      o2_sat_per_vol_blood_(matdata->GetDouble("O2SaturationPerVolBlood")),
      p_half_(matdata->GetDouble("PressureHalf")),
      power_(matdata->GetDouble("Power")),
      nO2_per_VO2_(matdata->GetDouble("NumberOfO2PerVO2"))
{
}

Teuchos::RCP<MAT::Material> MAT::PAR::Hemoglobin_0d_O2_saturation::CreateMaterial()
{
  return Teuchos::rcp(new MAT::Hemoglobin_0d_O2_saturation(this));
}


MAT::Hemoglobin_0d_O2_saturationType MAT::Hemoglobin_0d_O2_saturationType::instance_;


DRT::ParObject* MAT::Hemoglobin_0d_O2_saturationType::Create(const std::vector<char>& data)
{
  MAT::Hemoglobin_0d_O2_saturation* hem_0d_O2_sat = new MAT::Hemoglobin_0d_O2_saturation();
  hem_0d_O2_sat->Unpack(data);
  return hem_0d_O2_sat;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::Hemoglobin_0d_O2_saturation::Hemoglobin_0d_O2_saturation() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::Hemoglobin_0d_O2_saturation::Hemoglobin_0d_O2_saturation(
    MAT::PAR::Hemoglobin_0d_O2_saturation* params)
    : params_(params)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::Hemoglobin_0d_O2_saturation::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->Id();  // in case we are in post-process mode
  AddtoPack(data, matid);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::Hemoglobin_0d_O2_saturation::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position, data, type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");

  // matid
  int matid;
  ExtractfromPack(position, data, matid);
  params_ = nullptr;
  if (DRT::Problem::Instance()->Materials() != Teuchos::null)
    if (DRT::Problem::Instance()->Materials()->Num() != 0)
    {
      const int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();
      MAT::PAR::Parameter* mat =
          DRT::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
        params_ = static_cast<MAT::PAR::Hemoglobin_0d_O2_saturation*>(mat);
      else
        dserror("Type of parameter material %d does not fit to calling type %d", mat->Type(),
            MaterialType());
    }

  if (position != data.size()) dserror("Mismatch in size of data %d <-> %d", data.size(), position);
}
