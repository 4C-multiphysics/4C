// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elchphase.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::ElchPhase::ElchPhase(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      epsilon_(matdata.parameters.get<double>("EPSILON")),
      tortuosity_(matdata.parameters.get<double>("TORTUOSITY")),
      nummat_(matdata.parameters.get<int>("NUMMAT")),
      matids_(matdata.parameters.get<std::vector<int>>("MATIDS")),
      local_(matdata.parameters.get<bool>("LOCAL"))
{
  if (nummat_ != (int)matids_.size())
    FOUR_C_THROW(
        "number of phases {} does not fit to size of phase vector {}", nummat_, matids_.size());

  if (not local_)
  {
    // make sure the referenced materials in material list have quick access parameters
    std::vector<int>::const_iterator n;
    // phase
    for (n = matids_.begin(); n != matids_.end(); ++n)
    {
      const int matid = *n;
      std::shared_ptr<Core::Mat::Material> mat = Mat::factory(matid);
      mat_.insert(std::pair<int, std::shared_ptr<Core::Mat::Material>>(matid, mat));
    }
  }
}


std::shared_ptr<Core::Mat::Material> Mat::PAR::ElchPhase::create_material()
{
  return std::make_shared<Mat::ElchPhase>(this);
}

Mat::ElchPhaseType Mat::ElchPhaseType::instance_;


Core::Communication::ParObject* Mat::ElchPhaseType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::ElchPhase* elchphase = new Mat::ElchPhase();
  elchphase->unpack(buffer);
  return elchphase;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ElchPhase::ElchPhase() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::ElchPhase::ElchPhase(Mat::PAR::ElchPhase* params) : params_(params)
{
  // setup of material map
  if (params_->local_)
  {
    setup_mat_map();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElchPhase::setup_mat_map()
{
  // safety first
  mat_.clear();
  if (not mat_.empty()) FOUR_C_THROW("What's going wrong here?");

  // make sure the referenced materials in material list have quick access parameters

  // here's the recursive creation of materials
  std::vector<int>::const_iterator n;
  for (n = params_->mat_ids().begin(); n != params_->mat_ids().end(); ++n)
  {
    const int matid = *n;
    std::shared_ptr<Core::Mat::Material> mat = Mat::factory(matid);
    if (mat == nullptr) FOUR_C_THROW("Failed to allocate this material");
    mat_.insert(std::pair<int, std::shared_ptr<Core::Mat::Material>>(matid, mat));
  }
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElchPhase::clear()
{
  params_ = nullptr;
  mat_.clear();
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElchPhase::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  if (params_->local_)
  {
    // loop map of associated local materials
    if (params_ != nullptr)
    {
      // std::map<int, std::shared_ptr<Core::Mat::Material> >::const_iterator m;
      std::vector<int>::const_iterator n;
      for (n = params_->mat_ids().begin(); n != params_->mat_ids().end(); n++)
      {
        (mat_.find(*n))->second->pack(data);
      }
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::ElchPhase::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::ElchPhase*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  if (params_ != nullptr)  // params_ are not accessible in postprocessing mode
  {
    std::vector<int>::const_iterator n;
    for (n = params_->mat_ids().begin(); n != params_->mat_ids().end(); n++)
    {
      const int actmatid = *n;
      std::shared_ptr<Core::Mat::Material> mat = Mat::factory(actmatid);
      if (mat == nullptr) FOUR_C_THROW("Failed to allocate this material");
      mat_.insert(std::pair<int, std::shared_ptr<Core::Mat::Material>>(actmatid, mat));
    }

    if (params_->local_)
    {
      // loop map of associated local materials
      for (n = params_->mat_ids().begin(); n != params_->mat_ids().end(); n++)
      {
        (mat_.find(*n))->second->unpack(buffer);
      }
    }
  }  // if (params_ != nullptr)
}

FOUR_C_NAMESPACE_CLOSE
