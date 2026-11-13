// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elchmat.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::ElchMat::ElchMat(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      numdof_((matdata.parameters.get<int>("NUMDOF"))),
      numscal_((matdata.parameters.get<int>("NUMSCAL"))),
      numphase_(matdata.parameters.get<int>("NUMPHASE")),
      phaseids_(matdata.parameters.get<std::vector<int>>("PHASEIDS")),
      local_(matdata.parameters.get<bool>("LOCAL"))
{
  if (numphase_ != (int)phaseids_.size())
    FOUR_C_THROW(
        "number of phases {} does not fit to size of phase vector {}", numphase_, phaseids_.size());

  if (not local_)
  {
    // make sure the referenced materials in material list have quick access parameters
    std::vector<int>::const_iterator n;
    // phase
    for (n = phaseids_.begin(); n != phaseids_.end(); ++n)
    {
      const int phaseid = *n;
      std::shared_ptr<Core::Mat::Material> mat = Mat::factory(phaseid);
      mat_.insert(std::pair<int, std::shared_ptr<Core::Mat::Material>>(phaseid, mat));
    }
  }
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::ElchMat::create_material()
{
  return std::make_shared<Mat::ElchMat>(this);
}


Mat::ElchMatType Mat::ElchMatType::instance_;


Core::Communication::ParObject* Mat::ElchMatType::create(Core::Communication::UnpackBuffer& buffer)
{
  Mat::ElchMat* elchmat = new Mat::ElchMat();
  elchmat->unpack(buffer);
  return elchmat;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ElchMat::ElchMat() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ElchMat::ElchMat(Mat::PAR::ElchMat* params) : params_(params)
{
  // setup of material map
  if (params_->local_)
  {
    setup_mat_map();
  }
  // else: material rcps live inside Mat::PAR::MatList
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElchMat::setup_mat_map()
{
  // safety first
  mat_.clear();
  if (not mat_.empty()) FOUR_C_THROW("What's going wrong here?");

  // make sure the referenced materials in material list have quick access parameters

  // here's the recursive creation of materials
  std::vector<int>::const_iterator n;
  for (n = params_->phase_ids().begin(); n != params_->phase_ids().end(); ++n)
  {
    const int phaseid = *n;
    std::shared_ptr<Core::Mat::Material> mat = Mat::factory(phaseid);
    if (mat == nullptr) FOUR_C_THROW("Failed to allocate this material");
    mat_.insert(std::pair<int, std::shared_ptr<Core::Mat::Material>>(phaseid, mat));
  }
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElchMat::clear()
{
  params_ = nullptr;
  mat_.clear();
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElchMat::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  Core::Communication::PotentiallyUnusedBufferScope mat_scope{data};
  if (params_ != nullptr and params_->local_)
  {
    // loop map of associated local materials
    std::vector<int>::const_iterator n;
    for (n = params_->phase_ids().begin(); n != params_->phase_ids().end(); n++)
      (mat_.find(*n))->second->pack(data);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElchMat::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // make sure we have a pristine material
  clear();



  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid(-1);
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::ElchMat*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  Core::Communication::PotentiallyUnusedBufferScope mat_scope{buffer};
  if (params_ != nullptr && params_->local_)  // params_ are not accessible in postprocessing mode
  {
    std::vector<int>::const_iterator n;
    for (n = params_->phase_ids().begin(); n != params_->phase_ids().end(); n++)
    {
      const int actphaseid = *n;
      std::shared_ptr<Core::Mat::Material> mat = Mat::factory(actphaseid);
      if (mat == nullptr) FOUR_C_THROW("Failed to allocate this material");
      mat_.insert(std::pair<int, std::shared_ptr<Core::Mat::Material>>(actphaseid, mat));
    }

    if (params_->local_)
    {
      // loop map of associated local materials
      for (n = params_->phase_ids().begin(); n != params_->phase_ids().end(); n++)
      {
        (mat_.find(*n))->second->unpack(buffer);
      }
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
