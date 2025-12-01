// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_particle_pd.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | define static class member                                                |
 *---------------------------------------------------------------------------*/
Mat::ParticleMaterialPDType Mat::ParticleMaterialPDType::instance_;

/*---------------------------------------------------------------------------*
 | constructor                                                               |
 *---------------------------------------------------------------------------*/
Mat::PAR::ParticleMaterialPD::ParticleMaterialPD(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      ParticleMaterialBase(matdata),
      young_(matdata.parameters.get<double>("YOUNG")),
      critical_stretch_(matdata.parameters.get<double>("CRITICAL_STRETCH"))
{
  // empty constructor
}

/*---------------------------------------------------------------------------*
 | create material instance of matching type with parameters                 |
 *---------------------------------------------------------------------------*/
std::shared_ptr<Core::Mat::Material> Mat::PAR::ParticleMaterialPD::create_material()
{
  return std::make_shared<Mat::ParticleMaterialPD>(this);
}

/*---------------------------------------------------------------------------*
 *---------------------------------------------------------------------------*/
Core::Communication::ParObject* Mat::ParticleMaterialPDType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ParticleMaterialPD* particlematpd = new Mat::ParticleMaterialPD();
  particlematpd->unpack(buffer);
  return particlematpd;
}

/*---------------------------------------------------------------------------*
 | constructor (empty material object)                                       |
 *---------------------------------------------------------------------------*/
Mat::ParticleMaterialPD::ParticleMaterialPD() : params_(nullptr)
{
  // empty constructor
}

/*---------------------------------------------------------------------------*
 | constructor (with given material parameters)                              |
 *---------------------------------------------------------------------------*/
Mat::ParticleMaterialPD::ParticleMaterialPD(Mat::PAR::ParticleMaterialPD* params) : params_(params)
{
  // empty constructor
}

/*---------------------------------------------------------------------------*
 | pack                                                                      |
 *---------------------------------------------------------------------------*/
void Mat::ParticleMaterialPD::pack(Core::Communication::PackBuffer& data) const
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
void Mat::ParticleMaterialPD::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      // note: dynamic_cast needed due diamond inheritance structure
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = dynamic_cast<Mat::PAR::ParticleMaterialPD*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
}

FOUR_C_NAMESPACE_CLOSE
