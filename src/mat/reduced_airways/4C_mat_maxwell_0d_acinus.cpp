// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_maxwell_0d_acinus.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::Maxwell0dAcinus::Maxwell0dAcinus(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      stiffness1_(matdata.parameters.get<double>("Stiffness1")),
      stiffness2_(matdata.parameters.get<double>("Stiffness2")),
      viscosity1_(matdata.parameters.get<double>("Viscosity1")),
      viscosity2_(matdata.parameters.get<double>("Viscosity2"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::Maxwell0dAcinus::create_material()
{
  return std::make_shared<Mat::Maxwell0dAcinus>(this);
}


Mat::Maxwell0dAcinusType Mat::Maxwell0dAcinusType::instance_;


Core::Communication::ParObject* Mat::Maxwell0dAcinusType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::Maxwell0dAcinus* mxwll_0d_acin = new Mat::Maxwell0dAcinus();
  mxwll_0d_acin->unpack(buffer);
  return mxwll_0d_acin;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Maxwell0dAcinus::Maxwell0dAcinus() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Maxwell0dAcinus::Maxwell0dAcinus(Mat::PAR::Maxwell0dAcinus* params) : params_(params) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinus::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}


/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
void Mat::Maxwell0dAcinus::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::Maxwell0dAcinus*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Maxwell0dAcinus::get_params(std::string parametername)
{
  FOUR_C_THROW("GetParams not implemented yet for this material!");
  return 0;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Maxwell0dAcinus::set_params(std::string parametername, double new_value)
{
  FOUR_C_THROW("SetParams not implemented yet for this material!");
}

FOUR_C_NAMESPACE_CLOSE
