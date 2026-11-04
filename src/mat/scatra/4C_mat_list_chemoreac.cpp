// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_list_chemoreac.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | standard constructor                                     thon 06/15 |
 *----------------------------------------------------------------------*/
Mat::PAR::MatListChemoReac::MatListChemoReac(const Core::Mat::PAR::Parameter::Data& matdata)
    : MatList(matdata), MatListReactions(matdata), MatListChemotaxis(matdata)
{
}


std::shared_ptr<Core::Mat::Material> Mat::PAR::MatListChemoReac::create_material()
{
  return std::make_shared<Mat::MatListChemoReac>(this);
}


Mat::MatListChemoReacType Mat::MatListChemoReacType::instance_;


Core::Communication::ParObject* Mat::MatListChemoReacType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::MatListChemoReac* MatListChemoReac = new Mat::MatListChemoReac();
  MatListChemoReac->unpack(buffer);
  return MatListChemoReac;
}


/*----------------------------------------------------------------------*
 | construct empty material object                           thon 06/15 |
 *----------------------------------------------------------------------*/
Mat::MatListChemoReac::MatListChemoReac()
    : MatList(), MatListChemotaxis(), MatListReactions(), paramsreachemo_(nullptr)
{
}


/*----------------------------------------------------------------------*
 | construct the material object given material parameter     thon 06/15 |
 *----------------------------------------------------------------------*/
Mat::MatListChemoReac::MatListChemoReac(Mat::PAR::MatListChemoReac* params)
    : MatList(params), MatListChemotaxis(params), MatListReactions(params), paramsreachemo_(params)
{
  // setup of material map
  if (paramsreachemo_->local_)
  {
    setup_mat_map();
  }
}


/*----------------------------------------------------------------------*
 | setup of material map                                     thon 06/15 |
 *----------------------------------------------------------------------*/
void Mat::MatListChemoReac::setup_mat_map()
{
  // We just have to add the chemotactic/reactive materials, since the rest is already done in
  // Mat::MatList::setup_mat_map() called from the MatList constructor

  // here's the recursive creation of materials
  Mat::MatListReactions::setup_mat_map();
  Mat::MatListChemotaxis::setup_mat_map();

  return;
}


/*----------------------------------------------------------------------*
 | reset everything                                          thon 06/15 |
 *----------------------------------------------------------------------*/
void Mat::MatListChemoReac::clear()
{
  paramsreachemo_ = nullptr;
  return;
}


/*----------------------------------------------------------------------*
 | Unpack data from a char vector into this class            thon 06/15 |
 *----------------------------------------------------------------------*/
void Mat::MatListChemoReac::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (paramsreachemo_ != nullptr)
    matid = paramsreachemo_->id();  // in case we are in post-process mode

  add_to_pack(data, matid);

  // Pack base class material
  Mat::MatListReactions::pack(data);
  Mat::MatListChemotaxis::pack(data);
}


/*----------------------------------------------------------------------*
 | Unpack data from a char vector into this class            thon 06/15 |
 *----------------------------------------------------------------------*/
void Mat::MatListChemoReac::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // make sure we have a pristine material
  clear();



  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover paramsreac_
  int matid(-1);
  extract_from_pack(buffer, matid);
  paramsreachemo_ = nullptr;
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
        paramsreachemo_ = dynamic_cast<Mat::PAR::MatListChemoReac*>(mat);
      }
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  // extract base class material
  Mat::MatListReactions::unpack(buffer);
  Mat::MatListChemotaxis::unpack(buffer);
}

FOUR_C_NAMESPACE_CLOSE
