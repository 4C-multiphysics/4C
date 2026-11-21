// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_scatra_poro_ecm.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::ScatraMatPoroECM::ScatraMatPoroECM(const Core::Mat::PAR::Parameter::Data& matdata)
    : ScatraReactionMat(matdata), reacscale_(matdata.parameters.get<double>("REACSCALE"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::ScatraMatPoroECM::create_material()
{
  return std::make_shared<Mat::ScatraMatPoroECM>(this);
}


Mat::ScatraMatPoroECMType Mat::ScatraMatPoroECMType::instance_;

Core::Communication::ParObject* Mat::ScatraMatPoroECMType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ScatraMatPoroECM* scatra_mat = new Mat::ScatraMatPoroECM();
  scatra_mat->unpack(buffer);
  return scatra_mat;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ScatraMatPoroECM::ScatraMatPoroECM() : params_(nullptr), reaccoeff_(0.0) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ScatraMatPoroECM::ScatraMatPoroECM(Mat::PAR::ScatraMatPoroECM* params)
    : ScatraReactionMat(params), params_(params), reaccoeff_(0.0)
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScatraMatPoroECM::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  // reaccoeff_
  add_to_pack(data, reaccoeff_);

  // add base class material
  ScatraReactionMat::pack(data);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScatraMatPoroECM::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid
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
        params_ = static_cast<Mat::PAR::ScatraMatPoroECM*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  // reaccoeff_
  extract_from_pack(buffer, reaccoeff_);

  // extract base class material
  ScatraReactionMat::unpack(buffer);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScatraMatPoroECM::compute_reac_coeff(double chempot)
{
  reaccoeff_ = params_->reaccoeff_ * exp(params_->reacscale_ * chempot);
  return;
}

FOUR_C_NAMESPACE_CLOSE
