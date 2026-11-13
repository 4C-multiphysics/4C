// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_scatra_chemotaxis.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::ScatraChemotaxisMat::ScatraChemotaxisMat(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      numscal_(matdata.parameters.get<int>("NUMSCAL")),
      pair_(matdata.parameters.get<std::vector<int>>("PAIR")),
      chemocoeff_(matdata.parameters.get<double>("CHEMOCOEFF"))
{
  // Some checks for more safety
  if (numscal_ != (int)pair_.size())
    FOUR_C_THROW("number of materials {} does not fit to size of material vector {}", numscal_,
        pair_.size());

  // is there exactly one '1' (i.e. attractant) and at least one '-1' (i.e. chemotractant)?
  int numpos = 0;
  int numneg = 0;
  for (int i = 0; i < numscal_; i++)
  {
    if (pair_.at(i) > 1e-10)
      numpos++;
    else if (pair_.at(i) < -1e-10)
      numneg++;
  }
  if (numpos != 1 or numneg != 1)
    FOUR_C_THROW(
        "Each PAIR vector must contain exactly one '-1' (i.e. chemotractant) and exactly one '1' "
        "(i.e. attractant)!");

  return;
}


std::shared_ptr<Core::Mat::Material> Mat::PAR::ScatraChemotaxisMat::create_material()
{
  return std::make_shared<Mat::ScatraChemotaxisMat>(this);
}


Mat::ScatraChemotaxisMatType Mat::ScatraChemotaxisMatType::instance_;


Core::Communication::ParObject* Mat::ScatraChemotaxisMatType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ScatraChemotaxisMat* scatra_chemotaxis_mat = new Mat::ScatraChemotaxisMat();
  scatra_chemotaxis_mat->unpack(buffer);
  return scatra_chemotaxis_mat;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ScatraChemotaxisMat::ScatraChemotaxisMat() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ScatraChemotaxisMat::ScatraChemotaxisMat(Mat::PAR::ScatraChemotaxisMat* params)
    : params_(params)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScatraChemotaxisMat::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScatraChemotaxisMat::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::ScatraChemotaxisMat*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
}

FOUR_C_NAMESPACE_CLOSE
