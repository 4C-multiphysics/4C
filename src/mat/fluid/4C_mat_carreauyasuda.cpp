// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_carreauyasuda.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::CarreauYasuda::CarreauYasuda(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      nu_0_(matdata.parameters.get<double>("NU_0")),
      nu_inf_(matdata.parameters.get<double>("NU_INF")),
      lambda_(matdata.parameters.get<double>("LAMBDA")),
      a_param_(matdata.parameters.get<double>("APARAM")),
      b_param_(matdata.parameters.get<double>("BPARAM")),
      density_(matdata.parameters.get<double>("DENSITY"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::CarreauYasuda::create_material()
{
  return std::make_shared<Mat::CarreauYasuda>(this);
}


Mat::CarreauYasudaType Mat::CarreauYasudaType::instance_;


Core::Communication::ParObject* Mat::CarreauYasudaType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::CarreauYasuda* carYas = new Mat::CarreauYasuda();
  carYas->unpack(buffer);
  return carYas;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::CarreauYasuda::CarreauYasuda() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::CarreauYasuda::CarreauYasuda(Mat::PAR::CarreauYasuda* params) : params_(params) {}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::CarreauYasuda::pack(Core::Communication::PackBuffer& data) const
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
void Mat::CarreauYasuda::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::CarreauYasuda*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
}

FOUR_C_NAMESPACE_CLOSE
