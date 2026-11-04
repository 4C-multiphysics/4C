// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_sutherland.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::Sutherland::Sutherland(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      refvisc_(matdata.parameters.get<double>("REFVISC")),
      reftemp_(matdata.parameters.get<double>("REFTEMP")),
      suthtemp_(matdata.parameters.get<double>("SUTHTEMP")),
      shc_(matdata.parameters.get<double>("SHC")),
      pranum_(matdata.parameters.get<double>("PRANUM")),
      thermpress_(matdata.parameters.get<double>("THERMPRESS")),
      gasconst_(matdata.parameters.get<double>("GASCON"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::Sutherland::create_material()
{
  return std::make_shared<Mat::Sutherland>(this);
}


Mat::SutherlandType Mat::SutherlandType::instance_;


Core::Communication::ParObject* Mat::SutherlandType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::Sutherland* sutherland = new Mat::Sutherland();
  sutherland->unpack(buffer);
  return sutherland;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Sutherland::Sutherland() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::Sutherland::Sutherland(Mat::PAR::Sutherland* params) : params_(params) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::Sutherland::pack(Core::Communication::PackBuffer& data) const
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
void Mat::Sutherland::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::Sutherland*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Sutherland::compute_viscosity(const double temp) const
{
  // previous implementation using "pow"-function appears to be extremely
  // time-consuming sometimes, at least on the computing cluster
  // const double visc =
  // std::pow((temp/RefTemp()),1.5)*((RefTemp()+SuthTemp())/(temp+SuthTemp()))*RefVisc();
  const double visc = sqrt((temp / ref_temp()) * (temp / ref_temp()) * (temp / ref_temp())) *
                      ((ref_temp() + suth_temp()) / (temp + suth_temp())) * ref_visc();

  return visc;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Sutherland::compute_diffusivity(const double temp) const
{
  // previous implementation using "pow"-function appears to be extremely
  // time-consuming sometimes, at least on the computing cluster
  // const double diffus =
  // std::pow((temp/RefTemp()),1.5)*((RefTemp()+SuthTemp())/(temp+SuthTemp()))*RefVisc()/PraNum();
  const double diffus = sqrt((temp / ref_temp()) * (temp / ref_temp()) * (temp / ref_temp())) *
                        ((ref_temp() + suth_temp()) / (temp + suth_temp())) * ref_visc() /
                        pra_num();

  return diffus;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::Sutherland::compute_density(const double temp, const double thermpress) const
{
  const double density = thermpress / (gas_const() * temp);

  return density;
}

FOUR_C_NAMESPACE_CLOSE
