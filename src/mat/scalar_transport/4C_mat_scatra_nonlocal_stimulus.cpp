// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_scatra_nonlocal_stimulus.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"


FOUR_C_NAMESPACE_OPEN

Mat::PAR::ScatraNonlocalStimulusMat::ScatraNonlocalStimulusMat(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      characteristic_length_sq_(matdata.parameters.get<double>("CHAR_LENGTH_SQ")),
      structure_material_id_(matdata.parameters.get<int>("STRUCTURE_MAT_ID"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::ScatraNonlocalStimulusMat::create_material()
{
  return std::make_shared<Mat::ScatraNonlocalStimulusMat>(this);
}

Mat::ScatraNonlocalStimulusMatType Mat::ScatraNonlocalStimulusMatType::instance_;

Core::Communication::ParObject* Mat::ScatraNonlocalStimulusMatType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ScatraNonlocalStimulusMat* mat = new Mat::ScatraNonlocalStimulusMat();
  mat->unpack(buffer);
  return mat;
}

/*----------------------------------------------------------------------*/
Mat::ScatraNonlocalStimulusMat::ScatraNonlocalStimulusMat() : params_(nullptr) {}

Mat::ScatraNonlocalStimulusMat::ScatraNonlocalStimulusMat(
    Mat::PAR::ScatraNonlocalStimulusMat* params)
    : params_(params)
{
}

/*----------------------------------------------------------------------*/
void Mat::ScatraNonlocalStimulusMat::pack(Core::Communication::PackBuffer& data) const
{
  int type = unique_par_object_id();
  add_to_pack(data, type);

  int matid = -1;
  if (params_ != nullptr) matid = params_->id();
  add_to_pack(data, matid);
}

/*----------------------------------------------------------------------*/
void Mat::ScatraNonlocalStimulusMat::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

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
        params_ = static_cast<Mat::PAR::ScatraNonlocalStimulusMat*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
}

FOUR_C_NAMESPACE_CLOSE
