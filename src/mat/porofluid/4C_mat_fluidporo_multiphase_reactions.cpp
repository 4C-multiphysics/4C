// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_fluidporo_multiphase_reactions.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_fluidporo_multiphase_singlereaction.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | rstandard constructor                                     vuong 08/16 |
 *----------------------------------------------------------------------*/
Mat::PAR::FluidPoroMultiPhaseReactions::FluidPoroMultiPhaseReactions(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : FluidPoroMultiPhase(matdata),
      numreac_((matdata.parameters.get<int>("NUMREAC"))),
      reacids_((matdata.parameters.get<std::vector<int>>("REACIDS")))
{
  // check if sizes fit
  if (numreac_ != (int)reacids_.size())
    FOUR_C_THROW("number of materials {} does not fit to size of material vector {}", nummat_,
        reacids_.size());

  if (numreac_ < 1)
    FOUR_C_THROW("if you don't have reactions, use MAT_matlist instead of MAT_matlist_reactions!");

  if (not local_)
  {
    // make sure the referenced materials in material list have quick access parameters
    std::vector<int>::const_iterator m;
    for (m = reacids_.begin(); m != reacids_.end(); ++m)
    {
      const int reacid = *m;
      std::shared_ptr<Core::Mat::Material> mat = Mat::factory(reacid);

      // safety check and cast
      if (mat->material_type() != Core::Materials::m_fluidporo_singlereaction)
        FOUR_C_THROW("only MAT_FluidPoroSingleReaction material valid");
      Mat::FluidPoroSingleReaction singlereacmat = static_cast<Mat::FluidPoroSingleReaction&>(*mat);
      if (singlereacmat.total_num_dof() != this->nummat_)
        FOUR_C_THROW(
            "TOTALNUMDOF in MAT_FluidPoroSingleReaction does not correspond to NUMMAT in "
            "MAT_FluidPoroMultiPhaseReactions");

      material_map_write()->insert(
          std::pair<int, std::shared_ptr<Core::Mat::Material>>(reacid, mat));
    }
  }
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::FluidPoroMultiPhaseReactions::create_material()
{
  return std::make_shared<Mat::FluidPoroMultiPhaseReactions>(this);
}


Mat::FluidPoroMultiPhaseReactionsType Mat::FluidPoroMultiPhaseReactionsType::instance_;


Core::Communication::ParObject* Mat::FluidPoroMultiPhaseReactionsType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::FluidPoroMultiPhaseReactions* FluidPoroMultiPhaseReactions =
      new Mat::FluidPoroMultiPhaseReactions();
  FluidPoroMultiPhaseReactions->unpack(buffer);
  return FluidPoroMultiPhaseReactions;
}


/*----------------------------------------------------------------------*
 | construct empty material object                           vuong 08/16 |
 *----------------------------------------------------------------------*/
Mat::FluidPoroMultiPhaseReactions::FluidPoroMultiPhaseReactions()
    : FluidPoroMultiPhase(), paramsreac_(nullptr)
{
}

/*----------------------------------------------------------------------*
 | construct the material object given material parameter     vuong 08/16 |
 *----------------------------------------------------------------------*/
Mat::FluidPoroMultiPhaseReactions::FluidPoroMultiPhaseReactions(
    Mat::PAR::FluidPoroMultiPhaseReactions* params)
    : FluidPoroMultiPhase(params), paramsreac_(params)
{
  // setup of material map
  if (paramsreac_->local_)
  {
    setup_mat_map();
  }
}

/*----------------------------------------------------------------------*
 | setup of material map                                     vuong 08/16 |
 *----------------------------------------------------------------------*/
void Mat::FluidPoroMultiPhaseReactions::setup_mat_map()
{
  // We just have to add the reaction materials, since the rest is already done in
  // Mat::MatList::setup_mat_map() called from the MatList constructor

  // here's the recursive creation of materials
  std::vector<int>::const_iterator m;
  for (m = paramsreac_->reac_ids()->begin(); m != paramsreac_->reac_ids()->end(); ++m)
  {
    const int reacid = *m;
    std::shared_ptr<Core::Mat::Material> mat = Mat::factory(reacid);
    if (mat == nullptr) FOUR_C_THROW("Failed to allocate this material");
    material_map_write()->insert(std::pair<int, std::shared_ptr<Core::Mat::Material>>(reacid, mat));
  }
  return;
}

/*----------------------------------------------------------------------*
 | reset everything                                          vuong 08/16 |
 *----------------------------------------------------------------------*/
void Mat::FluidPoroMultiPhaseReactions::clear()
{
  paramsreac_ = nullptr;
  return;
}

/*----------------------------------------------------------------------*
 | Unpack data from a char vector into this class            vuong 08/16 |
 *----------------------------------------------------------------------*/
void Mat::FluidPoroMultiPhaseReactions::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (paramsreac_ != nullptr) matid = paramsreac_->id();  // in case we are in post-process mode

  add_to_pack(data, matid);

  // Pack base class material
  Mat::FluidPoroMultiPhase::pack(data);
}

/*----------------------------------------------------------------------*
 | Unpack data from a char vector into this class            vuong 08/16 |
 *----------------------------------------------------------------------*/
void Mat::FluidPoroMultiPhaseReactions::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // make sure we have a pristine material
  clear();



  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover paramsreac_
  int matid(-1);
  extract_from_pack(buffer, matid);
  paramsreac_ = nullptr;
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
        paramsreac_ = dynamic_cast<Mat::PAR::FluidPoroMultiPhaseReactions*>(mat);
      }
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  // extract base class material
  Mat::FluidPoroMultiPhase::unpack(buffer);

  // in the postprocessing mode, we do not unpack everything we have packed
  // -> position check cannot be done in this case
}

/*----------------------------------------------------------------------*
 | reaction ID by Index                                      vuong 08/16 |
 *----------------------------------------------------------------------*/
int Mat::FluidPoroMultiPhaseReactions::reac_id(const unsigned index) const
{
  if ((int)index < paramsreac_->numreac_)
    return paramsreac_->reacids_.at(index);
  else
  {
    FOUR_C_THROW("Index too large");
    return -1;
  }
}

FOUR_C_NAMESPACE_CLOSE
