// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_list_chemotaxis.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | standard constructor                                      thon 06/15 |
 *----------------------------------------------------------------------*/
Mat::PAR::MatListChemotaxis::MatListChemotaxis(const Core::Mat::PAR::Parameter::Data& matdata)
    : MatList(matdata),
      numpair_((matdata.parameters.get<int>("NUMPAIR"))),
      pairids_((matdata.parameters.get<std::vector<int>>("PAIRIDS")))
{
  // check if sizes fit
  if (numpair_ != (int)pairids_.size())
    FOUR_C_THROW("number of materials {} does not fit to size of material vector {}", nummat_,
        pairids_.size());

  if (numpair_ < 1)
    FOUR_C_THROW(
        "If you don't have chemotactic pairs, use MAT_matlist instead of MAT_matlist_chemotaxis!");

  if (not local_)
  {
    // make sure the referenced materials in material list have quick access parameters
    std::vector<int>::const_iterator m;
    for (m = pairids_.begin(); m != pairids_.end(); ++m)
    {
      const int pairid = *m;
      std::shared_ptr<Core::Mat::Material> mat = Mat::factory(pairid);
      material_map_write()->insert(
          std::pair<int, std::shared_ptr<Core::Mat::Material>>(pairid, mat));
    }
  }
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::MatListChemotaxis::create_material()
{
  return std::make_shared<Mat::MatListChemotaxis>(this);
}


Mat::MatListChemotaxisType Mat::MatListChemotaxisType::instance_;


Core::Communication::ParObject* Mat::MatListChemotaxisType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::MatListChemotaxis* MatListChemotaxis = new Mat::MatListChemotaxis();
  MatListChemotaxis->unpack(buffer);
  return MatListChemotaxis;
}


/*----------------------------------------------------------------------*
 | construct empty material object                           thon 06/15 |
 *----------------------------------------------------------------------*/
Mat::MatListChemotaxis::MatListChemotaxis() : MatList(), paramschemo_(nullptr) {}


/*----------------------------------------------------------------------*
 | construct the material object given material parameter     thon 06/15 |
 *----------------------------------------------------------------------*/
Mat::MatListChemotaxis::MatListChemotaxis(Mat::PAR::MatListChemotaxis* params)
    : MatList(params), paramschemo_(params)
{
  // setup of material map
  if (paramschemo_->local_)
  {
    setup_mat_map();
  }
}


/*----------------------------------------------------------------------*
 | setup of material map                                     thon 06/15 |
 *----------------------------------------------------------------------*/
void Mat::MatListChemotaxis::setup_mat_map()
{
  // We just have to add the chemotactic materials, since the rest is already done in
  // Mat::MatList::setup_mat_map() called from the MatList constructor

  // here's the recursive creation of materials
  std::vector<int>::const_iterator m;
  for (m = paramschemo_->pair_ids()->begin(); m != paramschemo_->pair_ids()->end(); ++m)
  {
    const int pairid = *m;
    std::shared_ptr<Core::Mat::Material> mat = Mat::factory(pairid);
    if (mat == nullptr) FOUR_C_THROW("Failed to allocate this material");
    material_map_write()->insert(std::pair<int, std::shared_ptr<Core::Mat::Material>>(pairid, mat));
  }
  return;
}


/*----------------------------------------------------------------------*
 | reset everything                                          thon 06/15 |
 *----------------------------------------------------------------------*/
void Mat::MatListChemotaxis::clear()
{
  paramschemo_ = nullptr;
  return;
}


/*----------------------------------------------------------------------*
 | Unpack data from a char vector into this class            thon 06/15 |
 *----------------------------------------------------------------------*/
void Mat::MatListChemotaxis::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (paramschemo_ != nullptr) matid = paramschemo_->id();  // in case we are in post-process mode

  add_to_pack(data, matid);

  // Pack base class material
  Mat::MatList::pack(data);
}


/*----------------------------------------------------------------------*
 | Unpack data from a char vector into this class            thon 06/15 |
 *----------------------------------------------------------------------*/
void Mat::MatListChemotaxis::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // make sure we have a pristine material
  clear();



  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover paramsreac_
  int matid(-1);
  extract_from_pack(buffer, matid);
  paramschemo_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
      {
        // Note: We need to do a dynamic_cast here since Chemotaxis, Reaction, and Chemo-reaction
        // are in a diamond inheritance structure
        paramschemo_ = dynamic_cast<Mat::PAR::MatListChemotaxis*>(mat);
      }
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  // extract base class material
  Mat::MatList::unpack(buffer);

  // in the postprocessing mode, we do not unpack everything we have packed
  // -> position check cannot be done in this case
}


/*----------------------------------------------------------------------*
 | reaction ID by Index                                      thon 06/15 |
 *----------------------------------------------------------------------*/
int Mat::MatListChemotaxis::pair_id(const unsigned index) const
{
  if ((int)index < paramschemo_->numpair_)
    return paramschemo_->pairids_.at(index);
  else
  {
    FOUR_C_THROW("Index too large");
    return -1;
  }
}

FOUR_C_NAMESPACE_CLOSE
