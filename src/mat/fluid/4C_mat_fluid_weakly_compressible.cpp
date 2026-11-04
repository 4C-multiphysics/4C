// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_fluid_weakly_compressible.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::WeaklyCompressibleFluid::WeaklyCompressibleFluid(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      viscosity_(matdata.parameters.get<double>("VISCOSITY")),
      refdensity_(matdata.parameters.get<double>("REFDENSITY")),
      refpressure_(matdata.parameters.get<double>("REFPRESSURE")),
      comprcoeff_(matdata.parameters.get<double>("COMPRCOEFF"))
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::WeaklyCompressibleFluid::create_material()
{
  return std::make_shared<Mat::WeaklyCompressibleFluid>(this);
}


Mat::WeaklyCompressibleFluidType Mat::WeaklyCompressibleFluidType::instance_;


Core::Communication::ParObject* Mat::WeaklyCompressibleFluidType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::WeaklyCompressibleFluid* fluid = new Mat::WeaklyCompressibleFluid();
  fluid->unpack(buffer);
  return fluid;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::WeaklyCompressibleFluid::WeaklyCompressibleFluid() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::WeaklyCompressibleFluid::WeaklyCompressibleFluid(Mat::PAR::WeaklyCompressibleFluid* params)
    : params_(params)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::WeaklyCompressibleFluid::pack(Core::Communication::PackBuffer& data) const
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
void Mat::WeaklyCompressibleFluid::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::WeaklyCompressibleFluid*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::WeaklyCompressibleFluid::compute_density(const double press) const
{
  const double density = ref_density() + compr_coeff() * (press - ref_pressure());

  return density;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double Mat::WeaklyCompressibleFluid::compute_pressure(const double dens) const
{
  const double pressure = ref_pressure() + 1.0 / compr_coeff() * (dens - ref_density());

  return pressure;
}

FOUR_C_NAMESPACE_CLOSE
