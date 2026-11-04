// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_scatra.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::ScatraMat::ScatraMat(const Core::Mat::PAR::Parameter::Data& matdata) : Parameter(matdata)
{
  // extract relevant communicator
  MPI_Comm comm = Global::Problem::instance()->materials()->get_read_from_problem() == 0
                      ? Global::Problem::instance()->get_communicators().local_comm()
                      : Global::Problem::instance()->get_communicators().sub_comm();

  Core::LinAlg::Map dummy_map(1, 1, 0, comm);
  for (int i = first; i <= last; i++)
  {
    matparams_.push_back(std::make_shared<Core::LinAlg::Vector<double>>(dummy_map, true));
  }
  matparams_.at(diff)->put_scalar(matdata.parameters.get<double>("DIFFUSIVITY"));
  matparams_.at(reac)->put_scalar(matdata.parameters.get<double>("REACOEFF"));
  matparams_.at(densific)->put_scalar(matdata.parameters.get<double>("DENSIFICATION"));
  matparams_.at(reacts_to_external_force)
      ->put_scalar(matdata.parameters.get<bool>("REACTS_TO_EXTERNAL_FORCE"));
}


std::shared_ptr<Core::Mat::Material> Mat::PAR::ScatraMat::create_material()
{
  return std::make_shared<Mat::ScatraMat>(this);
}


Mat::ScatraMatType Mat::ScatraMatType::instance_;


Core::Communication::ParObject* Mat::ScatraMatType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ScatraMat* scatra_mat = new Mat::ScatraMat();
  scatra_mat->unpack(buffer);
  return scatra_mat;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ScatraMat::ScatraMat() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ScatraMat::ScatraMat(Mat::PAR::ScatraMat* params) : params_(params) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ScatraMat::pack(Core::Communication::PackBuffer& data) const
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
void Mat::ScatraMat::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::ScatraMat*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
}

FOUR_C_NAMESPACE_CLOSE
