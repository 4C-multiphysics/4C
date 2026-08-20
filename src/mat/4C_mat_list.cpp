// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_list.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::PAR::MatList::MatList(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      nummat_(static_cast<int>(matdata.parameters.get<std::vector<int>>("MATIDS").size())),
      matids_(matdata.parameters.get<std::vector<int>>("MATIDS")),
      local_((matdata.parameters.get<bool>("LOCAL")))
{
  if (not local_)
  {
    // make sure the referenced materials in material list have quick access parameters
    std::vector<int>::const_iterator m;
    for (m = matids_.begin(); m != matids_.end(); ++m)
    {
      const int matid = *m;
      std::shared_ptr<Core::Mat::Material> mat = Mat::factory(matid);
      mat_.insert(std::pair<int, std::shared_ptr<Core::Mat::Material>>(matid, mat));
    }
  }
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::MatList::create_material()
{
  return std::make_shared<Mat::MatList>(this);
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::MatList::material_by_id(const int id) const
{
  if (not local_)
  {
    std::map<int, std::shared_ptr<Core::Mat::Material>>::const_iterator m = mat_.find(id);

    if (m == mat_.end())
    {
      FOUR_C_THROW("Material {} could not be found", id);
      return nullptr;
    }
    else
      return m->second;
  }
  else
    FOUR_C_THROW("This is not allowed");

  return nullptr;
}


Mat::MatListType Mat::MatListType::instance_;


Core::Communication::ParObject* Mat::MatListType::create(Core::Communication::UnpackBuffer& buffer)
{
  Mat::MatList* matlist = new Mat::MatList();
  matlist->unpack(buffer);
  return matlist;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::MatList::MatList() : params_(nullptr) {}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Mat::MatList::MatList(Mat::PAR::MatList* params) : params_(params)
{
  // setup of material map
  if (params_->local_)
  {
    setup_mat_map();
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::MatList::setup_mat_map()
{
  // safety first
  mat_.clear();
  if (not mat_.empty()) FOUR_C_THROW("What's going wrong here?");

  // make sure the referenced materials in material list have quick access parameters

  // here's the recursive creation of materials
  std::vector<int>::const_iterator m;
  for (m = params_->mat_ids()->begin(); m != params_->mat_ids()->end(); ++m)
  {
    const int matid = *m;
    std::shared_ptr<Core::Mat::Material> mat = Mat::factory(matid);
    if (mat == nullptr) FOUR_C_THROW("Failed to allocate this material");
    mat_.insert(std::pair<int, std::shared_ptr<Core::Mat::Material>>(matid, mat));
  }
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::MatList::clear()
{
  params_ = nullptr;
  mat_.clear();
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::MatList::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  if (params_ != nullptr)
    if (params_->local_)
    {
      // loop map of associated local materials
      if (params_ != nullptr)
      {
        // std::map<int, std::shared_ptr<Core::Mat::Material> >::const_iterator m;
        std::vector<int>::const_iterator m;
        for (m = params_->mat_ids()->begin(); m != params_->mat_ids()->end(); m++)
        {
          (mat_.find(*m))->second->pack(data);
        }
      }
    }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mat::MatList::unpack(Core::Communication::UnpackBuffer& buffer)
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
        params_ = static_cast<Mat::PAR::MatList*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  if (params_ != nullptr)  // params_ are not accessible in postprocessing mode
  {
    // make sure the referenced materials in material list have quick access parameters
    std::vector<int>::const_iterator m;
    for (m = params_->mat_ids()->begin(); m != params_->mat_ids()->end(); m++)
    {
      const int actmatid = *m;
      std::shared_ptr<Core::Mat::Material> mat = Mat::factory(actmatid);
      if (mat == nullptr) FOUR_C_THROW("Failed to allocate this material");
      mat_.insert(std::pair<int, std::shared_ptr<Core::Mat::Material>>(actmatid, mat));
    }

    if (params_->local_)
    {
      // loop map of associated local materials
      for (m = params_->mat_ids()->begin(); m != params_->mat_ids()->end(); m++)
      {
        (mat_.find(*m))->second->unpack(buffer);
      }
    }
    // in the postprocessing mode, we do not unpack everything we have packed
    // -> position check cannot be done in this case

  }  // if (params_ != nullptr)
}

/*----------------------------------------------------------------------*
 | material ID by Index                                      thon 11/14 |
 *----------------------------------------------------------------------*/
int Mat::MatList::mat_id(const unsigned index) const
{
  if ((int)index < params_->nummat_)
    return params_->matids_.at(index);
  else
  {
    FOUR_C_THROW("Index too large");
    return -1;
  }
}

/*----------------------------------------------------------------------*
 | provide access to material by its ID                      thon 11/14 |
 *----------------------------------------------------------------------*/
///
std::shared_ptr<Core::Mat::Material> Mat::MatList::material_by_id(const int id) const
{
  if (params_->local_)
  {
    std::map<int, std::shared_ptr<Core::Mat::Material>>::const_iterator m =
        material_map_read()->find(id);
    if (m == mat_.end())
    {
      FOUR_C_THROW("Material {} could not be found", id);
      return nullptr;
    }
    else
      return m->second;
  }
  else  // material is global (stored in material parameters)
    return params_->material_by_id(id);
}

FOUR_C_NAMESPACE_CLOSE
