// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_soret.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor                                               fang 06/15 |
 *----------------------------------------------------------------------*/
Mat::PAR::Soret::Soret(const Core::Mat::PAR::Parameter::Data& matdata)
    : Fourier(matdata), soretcoefficient_(matdata.parameters.get<double>("SORET"))
{
  return;
}


/*------------------------------------------------------------------*
 | create instance of Soret material                     fang 06/15 |
 *------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::Soret::create_material()
{
  return std::make_shared<Mat::Soret>(this);
}

Mat::SoretType Mat::SoretType::instance_;

Core::Communication::ParObject* Mat::SoretType::create(Core::Communication::UnpackBuffer& buffer)
{
  Mat::Soret* soret = new Mat::Soret();
  soret->unpack(buffer);
  return soret;
}


/*------------------------------------------------------------------*
 | construct empty Soret material                        fang 06/15 |
 *------------------------------------------------------------------*/
Mat::Soret::Soret() : params_(nullptr) { return; }


/*-------------------------------------------------------------------------*
 | construct Soret material with specific material parameters   fang 06/15 |
 *-------------------------------------------------------------------------*/
Mat::Soret::Soret(Mat::PAR::Soret* params) : Fourier(params), params_(params) { return; }


/*----------------------------------------------------------------------*
 | pack material for communication purposes                  fang 06/15 |
 *----------------------------------------------------------------------*/
void Mat::Soret::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  // pack base class material
  Fourier::pack(data);
}


/*----------------------------------------------------------------------*
 | unpack data from a char vector                            fang 06/15 |
 *----------------------------------------------------------------------*/
void Mat::Soret::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::Soret*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not match calling type {}!", mat->type(),
            material_type());
    }

  // extract base class material
  Fourier::unpack(buffer);
}

FOUR_C_NAMESPACE_CLOSE
