// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_scatra_growth_remodel.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"


FOUR_C_NAMESPACE_OPEN

Mat::PAR::ScatraGrowthRemodelMat::ScatraGrowthRemodelMat(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      diffusivity_(matdata.parameters.get<double>("DIFFUSIVITY")),
      structure_material_id_(matdata.parameters.get<int>("STRUCTURE_MAT_ID")),
      scalar_quantity_(matdata.parameters.get<Mat::PAR::ScatraGrowthRemodelMat::ScalarQuantity>(
          "SCALAR_QUANTITY"))

{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::ScatraGrowthRemodelMat::create_material()
{
  return std::make_shared<Mat::ScatraGrowthRemodelMat>(this);
}

Mat::ScatraGrowthRemodelMatType Mat::ScatraGrowthRemodelMatType::instance_;

Core::Communication::ParObject* Mat::ScatraGrowthRemodelMatType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ScatraGrowthRemodelMat* scatra_gr_mat = new Mat::ScatraGrowthRemodelMat();
  scatra_gr_mat->unpack(buffer);
  return scatra_gr_mat;
}

/*----------------------------------------------------------------------*/
Mat::ScatraGrowthRemodelMat::ScatraGrowthRemodelMat() : params_(nullptr) {}

Mat::ScatraGrowthRemodelMat::ScatraGrowthRemodelMat(Mat::PAR::ScatraGrowthRemodelMat* params)
    : params_(params)
{
}

/*----------------------------------------------------------------------*/
void Mat::ScatraGrowthRemodelMat::pack(Core::Communication::PackBuffer& data) const
{
  int type = unique_par_object_id();
  add_to_pack(data, type);

  int matid = -1;
  if (params_ != nullptr) matid = params_->id();
  add_to_pack(data, matid);
}

/*----------------------------------------------------------------------*/
void Mat::ScatraGrowthRemodelMat::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::ScatraGrowthRemodelMat*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
}

FOUR_C_NAMESPACE_CLOSE