/*----------------------------------------------------------------------*/
/*! \file
\brief material according to mixture-fraction approach

\level 2

*----------------------------------------------------------------------*/


#include "4C_mat_mixfrac.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::PAR::MixFrac::MixFrac(Teuchos::RCP<MAT::PAR::Material> matdata)
    : Parameter(matdata),
      kinvisc_(*matdata->Get<double>("KINVISC")),
      kindiff_(*matdata->Get<double>("KINDIFF")),
      eosfaca_(*matdata->Get<double>("EOSFACA")),
      eosfacb_(*matdata->Get<double>("EOSFACB"))
{
}

Teuchos::RCP<MAT::Material> MAT::PAR::MixFrac::CreateMaterial()
{
  return Teuchos::rcp(new MAT::MixFrac(this));
}


MAT::MixFracType MAT::MixFracType::instance_;


CORE::COMM::ParObject* MAT::MixFracType::Create(const std::vector<char>& data)
{
  MAT::MixFrac* mixfrac = new MAT::MixFrac();
  mixfrac->Unpack(data);
  return mixfrac;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::MixFrac::MixFrac() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MAT::MixFrac::MixFrac(MAT::PAR::MixFrac* params) : params_(params) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MAT::MixFrac::Pack(CORE::COMM::PackBuffer& data) const
{
  CORE::COMM::PackBuffer::SizeMarker sm(data);
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
void MAT::MixFrac::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;

  CORE::COMM::ExtractAndAssertId(position, data, UniqueParObjectId());

  // matid and recover params_
  int matid;
  ExtractfromPack(position, data, matid);
  params_ = nullptr;
  if (GLOBAL::Problem::Instance()->Materials() != Teuchos::null)
    if (GLOBAL::Problem::Instance()->Materials()->Num() != 0)
    {
      const int probinst = GLOBAL::Problem::Instance()->Materials()->GetReadFromProblem();
      MAT::PAR::Parameter* mat =
          GLOBAL::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
        params_ = static_cast<MAT::PAR::MixFrac*>(mat);
      else
        FOUR_C_THROW("Type of parameter material %d does not fit to calling type %d", mat->Type(),
            MaterialType());
    }

  if (position != data.size())
    FOUR_C_THROW("Mismatch in size of data %d <-> %d", data.size(), position);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double MAT::MixFrac::ComputeViscosity(const double mixfrac) const
{
  const double visc = KinVisc() / (EosFacA() * mixfrac + EosFacB());

  return visc;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double MAT::MixFrac::ComputeDiffusivity(const double mixfrac) const
{
  const double diffus = KinDiff() / (EosFacA() * mixfrac + EosFacB());

  return diffus;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double MAT::MixFrac::ComputeDensity(const double mixfrac) const
{
  const double density = 1.0 / (EosFacA() * mixfrac + EosFacB());

  return density;
}

FOUR_C_NAMESPACE_CLOSE
