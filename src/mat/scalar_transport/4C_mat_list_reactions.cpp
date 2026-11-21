// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_list_reactions.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_scatra_reaction.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | standard constructor                                     thon 11/14 |
 *----------------------------------------------------------------------*/
Mat::PAR::MatListReactions::MatListReactions(const Core::Mat::PAR::Parameter::Data& matdata)
    : MatList(matdata),
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
      material_map_write()->insert(
          std::pair<int, std::shared_ptr<Core::Mat::Material>>(reacid, mat));
    }
  }
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::MatListReactions::create_material()
{
  return std::make_shared<Mat::MatListReactions>(this);
}


Mat::MatListReactionsType Mat::MatListReactionsType::instance_;


Core::Communication::ParObject* Mat::MatListReactionsType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mat::MatListReactions* MatListReactions = new Mat::MatListReactions();
  MatListReactions->unpack(buffer);
  return MatListReactions;
}


/*----------------------------------------------------------------------*
 | construct empty material object                           thon 11/14 |
 *----------------------------------------------------------------------*/
Mat::MatListReactions::MatListReactions() : MatList(), paramsreac_(nullptr) {}

/*----------------------------------------------------------------------*
 | construct the material object given material parameter    thon 11/14 |
 *----------------------------------------------------------------------*/
Mat::MatListReactions::MatListReactions(Mat::PAR::MatListReactions* params)
    : MatList(params), paramsreac_(params)
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
void Mat::MatListReactions::initialize()
{
  if (paramsreac_ != nullptr)
  {
    std::vector<int>::const_iterator m;
    for (m = paramsreac_->reac_ids()->begin(); m != paramsreac_->reac_ids()->end(); ++m)
    {
      const int reacid = *m;
      std::shared_ptr<Core::Mat::Material> mat = material_by_id(reacid);
      if (mat == nullptr) FOUR_C_THROW("Failed to allocate this material");
      std::shared_ptr<Mat::ScatraReactionMat> reacmat =
          std::dynamic_pointer_cast<Mat::ScatraReactionMat>(mat);
      reacmat->initialize();
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 | setup of material map                                     thon 11/14 |
 *----------------------------------------------------------------------*/
void Mat::MatListReactions::setup_mat_map()
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
 | reset everything                                          thon 11/14 |
 *----------------------------------------------------------------------*/
void Mat::MatListReactions::clear()
{
  paramsreac_ = nullptr;
  return;
}

/*----------------------------------------------------------------------*
 | Unpack data from a char vector into this class            thon 11/14 |
 *----------------------------------------------------------------------*/
void Mat::MatListReactions::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (paramsreac_ != nullptr) matid = paramsreac_->id();  // in case we are in post-process mode

  add_to_pack(data, matid);

  // Pack base class material
  Mat::MatList::pack(data);

  if (paramsreac_ != nullptr)
  {
    if (paramsreac_->local_)
    {
      std::vector<int>::const_iterator m;
      for (m = paramsreac_->reac_ids()->begin(); m != paramsreac_->reac_ids()->end(); m++)
      {
        (material_map_read()->find(*m))->second->pack(data);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 | Unpack data from a char vector into this class            thon 11/14 |
 *----------------------------------------------------------------------*/
void Mat::MatListReactions::unpack(Core::Communication::UnpackBuffer& buffer)
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
        paramsreac_ = dynamic_cast<Mat::PAR::MatListReactions*>(mat);
      }
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }

  // extract base class material
  Mat::MatList::unpack(buffer);

  if (paramsreac_ != nullptr)  // paramsreac_ are not accessible in postprocessing mode
  {
    // make sure the referenced materials in material list have quick access parameters
    std::vector<int>::const_iterator m;
    for (m = paramsreac_->reac_ids()->begin(); m != paramsreac_->reac_ids()->end(); m++)
    {
      const int actmatid = *m;
      std::shared_ptr<Core::Mat::Material> mat = Mat::factory(actmatid);
      if (mat == nullptr) FOUR_C_THROW("Failed to allocate this material");
      material_map_write()->insert(
          std::pair<int, std::shared_ptr<Core::Mat::Material>>(actmatid, mat));
    }

    if (paramsreac_->local_)
    {
      // loop map of associated local materials
      for (m = paramsreac_->reac_ids()->begin(); m != paramsreac_->reac_ids()->end(); m++)
      {
        (material_map_write()->find(*m))->second->unpack(buffer);
      }
    }
    // in the postprocessing mode, we do not unpack everything we have packed
    // -> position check cannot be done in this case
  }
}

/*----------------------------------------------------------------------*
 | reaction ID by Index                                      thon 11/14 |
 *----------------------------------------------------------------------*/
int Mat::MatListReactions::reac_id(const unsigned index) const
{
  if ((int)index < paramsreac_->numreac_)
    return paramsreac_->reacids_.at(index);
  else
  {
    FOUR_C_THROW("Index too large");
    return -1;
  }
}

/*----------------------------------------------------------------------*
 | calculate advanced reaction terms                         thon 08/16 |
 *----------------------------------------------------------------------*/
double Mat::MatListReactions::calc_rea_body_force_term(const int k,
    const std::vector<double>& phinp, const double* gpcoord, const double time,
    const double scale) const
{
  // set time and space coordinates
  std::vector<std::pair<std::string, double>> constants;
  constants.emplace_back("t", time);
  constants.emplace_back("x", gpcoord[0]);
  constants.emplace_back("y", gpcoord[1]);
  constants.emplace_back("z", gpcoord[2]);

  double bodyforcetermK = 0.0;

  for (int condnum = 0; condnum < num_reac(); condnum++)
  {
    const int reacid = reac_id(condnum);
    const std::shared_ptr<const Mat::ScatraReactionMat> reacmat =
        std::static_pointer_cast<const Mat::ScatraReactionMat>(material_by_id(reacid));

    bodyforcetermK += reacmat->calc_rea_body_force_term(k, phinp, constants, scale);
  }

  return bodyforcetermK;
}

/*----------------------------------------------------------------------*
 | calculate advanced reaction term derivatives              thon 08/16 |
 *----------------------------------------------------------------------*/
void Mat::MatListReactions::calc_rea_body_force_deriv_matrix(const int k,
    std::vector<double>& derivs, const std::vector<double>& phinp, const double* gpcoord,
    const double time, const double scale) const
{
  std::vector<std::pair<std::string, double>> constants;
  constants.emplace_back("t", time);
  constants.emplace_back("x", gpcoord[0]);
  constants.emplace_back("y", gpcoord[1]);
  constants.emplace_back("z", gpcoord[2]);

  for (int condnum = 0; condnum < num_reac(); condnum++)
  {
    const int reacid = reac_id(condnum);
    const std::shared_ptr<const ScatraReactionMat> reacmat =
        std::static_pointer_cast<const ScatraReactionMat>(material_by_id(reacid));

    reacmat->calc_rea_body_force_deriv_matrix(k, derivs, phinp, constants, scale);
  }
}

/*----------------------------------------------------------------------*
 | calculate advanced reaction terms                         thon 08/16 |
 *----------------------------------------------------------------------*/
double Mat::MatListReactions::calc_rea_body_force_term(const int k,
    const std::vector<double>& phinp, const std::vector<std::pair<std::string, double>>& constants,
    const double* gpcoord, const double time, const double scale) const
{
  std::vector constants_mod(constants);
  constants_mod.emplace_back("t", time);
  constants_mod.emplace_back("x", gpcoord[0]);
  constants_mod.emplace_back("y", gpcoord[1]);
  constants_mod.emplace_back("z", gpcoord[2]);

  double bodyforcetermK = 0.0;

  for (int condnum = 0; condnum < num_reac(); condnum++)
  {
    const int reacid = reac_id(condnum);
    const std::shared_ptr<const ScatraReactionMat> reacmat =
        std::static_pointer_cast<const ScatraReactionMat>(material_by_id(reacid));

    bodyforcetermK += reacmat->calc_rea_body_force_term(k, phinp, constants_mod, scale);
  }

  return bodyforcetermK;
}

/*----------------------------------------------------------------------*
 | calculate advanced reaction term derivatives              thon 08/16 |
 *----------------------------------------------------------------------*/
void Mat::MatListReactions::calc_rea_body_force_deriv_matrix(const int k,
    std::vector<double>& derivs, const std::vector<double>& phinp,
    const std::vector<std::pair<std::string, double>>& constants, const double* gpcoord,
    const double time, const double scale) const
{
  std::vector constants_mod(constants);
  constants_mod.emplace_back("t", time);
  constants_mod.emplace_back("x", gpcoord[0]);
  constants_mod.emplace_back("y", gpcoord[1]);
  constants_mod.emplace_back("z", gpcoord[2]);

  for (int condnum = 0; condnum < num_reac(); condnum++)
  {
    const int reacid = reac_id(condnum);
    const std::shared_ptr<const ScatraReactionMat> reacmat =
        std::static_pointer_cast<const ScatraReactionMat>(material_by_id(reacid));

    reacmat->calc_rea_body_force_deriv_matrix(k, derivs, phinp, constants_mod, scale);
  }
}

/*----------------------------------------------------------------------*
 | add additional variables to reaction                kremheller 07/17 |
 *----------------------------------------------------------------------*/
void Mat::MatListReactions::add_additional_variables(
    const int k, const std::vector<std::pair<std::string, double>>& variables) const
{
  for (int condnum = 0; condnum < num_reac(); condnum++)
  {
    const int reacid = reac_id(condnum);
    const std::shared_ptr<const Mat::ScatraReactionMat> reacmat =
        std::static_pointer_cast<const Mat::ScatraReactionMat>(material_by_id(reacid));

    reacmat->add_additional_variables(k, variables);
  }
}

/*--------------------------------------------------------------------------------*
 |  calculating advanced reaction term derivatives after additional variables     |
 |  (e.g. for monolithic coupling)                               kremheller 07/17 |
 *--------------------------------------------------------------------------------*/
void Mat::MatListReactions::calc_rea_body_force_deriv_matrix_add_variables(const int k,
    std::vector<double>& derivs, const std::vector<double>& phinp,
    const std::vector<std::pair<std::string, double>>& variables,
    const std::vector<std::pair<std::string, double>>& constants, const double* gpcoord,
    const double time, const double scale) const
{
  std::vector constants_mod(constants);

  // add scalar values as constants
  for (unsigned iscal = 0; iscal < phinp.size(); iscal++)
  {
    constants_mod.emplace_back("phi" + std::to_string(iscal + 1), phinp[iscal]);
  }
  constants_mod.emplace_back("t", time);
  constants_mod.emplace_back("x", gpcoord[0]);
  constants_mod.emplace_back("y", gpcoord[1]);
  constants_mod.emplace_back("z", gpcoord[2]);

  for (int condnum = 0; condnum < num_reac(); condnum++)
  {
    const int reacid = reac_id(condnum);
    const std::shared_ptr<const ScatraReactionMat> reacmat =
        std::static_pointer_cast<const ScatraReactionMat>(material_by_id(reacid));

    reacmat->calc_rea_body_force_deriv_matrix_add_variables(
        k, derivs, variables, constants_mod, scale);
  }
}

FOUR_C_NAMESPACE_CLOSE
