// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_newman_multiscale.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::NewmanMultiScale::NewmanMultiScale(const Core::Mat::PAR::Parameter::Data& matdata)
    : Newman(matdata),
      ScatraMicroMacroCoupling(matdata),
      electronic_cond_(matdata.parameters.get<double>("ELECTRONIC_COND")),
      conc_dep_scale_func_num_(matdata.parameters.get<int>("ELECTRONIC_COND_CONC_SCALE_FUNC_NUM"))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::NewmanMultiScale::create_material()
{
  return std::make_shared<Mat::NewmanMultiScale>(this);
}


Mat::NewmanMultiScaleType Mat::NewmanMultiScaleType::instance_;

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::NewmanMultiScaleType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* const NewmanMultiScale = new Mat::NewmanMultiScale();
  NewmanMultiScale->unpack(buffer);
  return NewmanMultiScale;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::NewmanMultiScale::NewmanMultiScale(Mat::PAR::NewmanMultiScale* params)
    : Newman(params), params_(params)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::NewmanMultiScale::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  const int type = unique_par_object_id();
  add_to_pack(data, type);

  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  // pack base class material
  Newman::pack(data);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::NewmanMultiScale::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::NewmanMultiScale*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not match calling type {}!", mat->type(),
            material_type());
    }
  }

  // extract base class material
  Newman::unpack(buffer);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::NewmanMultiScale::electronic_cond(const int gp) const
{
  const int func_num = params_->conc_dep_scale_func_num();
  if (func_num > 0)
  {
    return Global::Problem::instance()
               ->function_by_id<Core::Utils::FunctionOfAnything>(func_num)
               .evaluate({{"c", evaluate_mean_concentration(gp)}}, {}, 0) *
           params_->electronic_cond();
  }
  else
  {
    return params_->electronic_cond();
  }
}

FOUR_C_NAMESPACE_CLOSE
