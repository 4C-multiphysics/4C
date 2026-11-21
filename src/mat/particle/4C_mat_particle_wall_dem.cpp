// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_particle_wall_dem.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | define static class member                                                |
 *---------------------------------------------------------------------------*/
Mat::ParticleWallMaterialDEMType Mat::ParticleWallMaterialDEMType::instance_;

/*---------------------------------------------------------------------------*
 | constructor                                                               |
 *---------------------------------------------------------------------------*/
Mat::PAR::ParticleWallMaterialDEM::ParticleWallMaterialDEM(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      frictionTang_(matdata.parameters.get<double>("FRICT_COEFF_TANG")),
      frictionRoll_(matdata.parameters.get<double>("FRICT_COEFF_ROLL")),
      adhesionSurfaceEnergy_(matdata.parameters.get<double>("ADHESION_SURFACE_ENERGY"))
{
  // empty constructor
}

/*---------------------------------------------------------------------------*
 | create material instance of matching type with parameters                 |
 *---------------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::ParticleWallMaterialDEM::create_material()
{
  return std::make_shared<Mat::ParticleWallMaterialDEM>(this);
}

/*---------------------------------------------------------------------------*
 *---------------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::ParticleWallMaterialDEMType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ParticleWallMaterialDEM* particlewallmatdem = new Mat::ParticleWallMaterialDEM();
  particlewallmatdem->unpack(buffer);
  return particlewallmatdem;
}

/*---------------------------------------------------------------------------*
 | constructor (empty material object)                                       |
 *---------------------------------------------------------------------------*/
Mat::ParticleWallMaterialDEM::ParticleWallMaterialDEM() : params_(nullptr)
{
  // empty constructor
}

/*---------------------------------------------------------------------------*
 | constructor (with given material parameters)                              |
 *---------------------------------------------------------------------------*/
Mat::ParticleWallMaterialDEM::ParticleWallMaterialDEM(Mat::PAR::ParticleWallMaterialDEM* params)
    : params_(params)
{
  // empty constructor
}

/*---------------------------------------------------------------------------*
 | pack                                                                      |
 *---------------------------------------------------------------------------*/
void Mat::ParticleWallMaterialDEM::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}

/*---------------------------------------------------------------------------*
 | unpack                                                                    |
 *---------------------------------------------------------------------------*/
void Mat::ParticleWallMaterialDEM::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::ParticleWallMaterialDEM*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
}

FOUR_C_NAMESPACE_CLOSE
