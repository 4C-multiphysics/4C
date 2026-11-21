// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_micromaterial.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


// Be careful when adding new member functions of MicroMaterial that
// relate to MicroMaterialGP (which is NOT in the filter). See also
// comments in micromaterial_evaluate.cpp which is separated from
// this file for the very reason.


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::MicroMaterial::MicroMaterial(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      microfile_((matdata.parameters.get<std::string>("MICROFILE"))),
      microdisnum_(matdata.parameters.get<int>("MICRODIS_NUM")),
      initvol_(matdata.parameters.get<double>("INITVOL"))
{
  runtime_output_option_ =
      matdata.parameters.get<Mat::PAR::MicroMaterial::RuntimeOutputOption>("RUNTIMEOUTPUT_GP");
}


std::shared_ptr<Core::Mat::Material> Mat::PAR::MicroMaterial::create_material()
{
  return std::make_shared<Mat::MicroMaterial>(this);
}


Mat::MicroMaterialType Mat::MicroMaterialType::instance_;


Core::Communication::ParObject* Mat::MicroMaterialType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::MicroMaterial* micro = new Mat::MicroMaterial();
  micro->unpack(buffer);
  return micro;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::MicroMaterial::MicroMaterial(Mat::PAR::MicroMaterial* params) : params_(params) {}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::MicroMaterial::pack(Core::Communication::PackBuffer& data) const
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
void Mat::MicroMaterial::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::MicroMaterial*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
}

FOUR_C_NAMESPACE_CLOSE
