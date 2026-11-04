// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_scatra_micro_macro_coupling.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_scatra_multiscale_gp.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::PAR::ScatraMicroMacroCoupling::ScatraMicroMacroCoupling(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : microfile_((matdata.parameters.get<std::string>("MICROFILE"))),
      microdisnum_(matdata.parameters.get<int>("MICRODIS_NUM")),
      A_s_(matdata.parameters.get<double>("A_s"))
{
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::ScatraMicroMacroCoupling::ScatraMicroMacroCoupling() : matgp_() {}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMicroMacroCoupling::initialize(const int ele_id, const int gp_id, const bool is_ale)
{
  // safety check
  if (gp_id < 0) FOUR_C_THROW("Invalid macro-scale Gauss point ID!");

  // initialize multi-scale scalar transport material
  if (matgp_.find(gp_id) == matgp_.end())
  {
    // instantiate and initialize multi-scale scalar transport submaterial at macro-scale Gauss
    // point
    matgp_[gp_id] = std::make_shared<ScatraMultiScaleGP>(ele_id, gp_id, micro_dis_num(), is_ale);
    matgp_[gp_id]->init();
  }
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMicroMacroCoupling::prepare_time_step(
    const int gp_id, const std::vector<double>& phinp_macro) const
{
  // safety check
  if (gp_id < 0) FOUR_C_THROW("Invalid macro-scale Gauss point ID!");

  // prepare time step on micro scale
  matgp_.at(gp_id)->prepare_time_step(phinp_macro);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMicroMacroCoupling::collect_output_data(const int gp_id) const
{
  // safety check
  FOUR_C_ASSERT_ALWAYS(gp_id >= 0, "Invalid macro-scale Gauss point ID: {}!", gp_id);

  // create output on micro scale
  matgp_.at(gp_id)->collect_and_write_output_data();
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMicroMacroCoupling::evaluate(const int gp_id,
    const std::vector<double>& phinp_macro, double& q_micro, std::vector<double>& dq_dphi_micro,
    const double detF, const bool solve) const
{
  // safety check
  if (gp_id < 0) FOUR_C_THROW("Invalid macro-scale Gauss point ID!");

  // evaluate multi-scale scalar transport sub material at macro-scale Gauss point
  matgp_.at(gp_id)->evaluate(phinp_macro, q_micro, dq_dphi_micro, detF, solve);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::ScatraMicroMacroCoupling::evaluate_mean_concentration(const int gp_id) const
{
  // safety check
  if (gp_id < 0) FOUR_C_THROW("Invalid macro-scale Gauss point ID!");

  // evaluate mean concentration on micro scale
  return matgp_.at(gp_id)->evaluate_mean_concentration();
}


/*-------------------------------------------------------------------------*
 *-------------------------------------------------------------------------*/
double Mat::ScatraMicroMacroCoupling::evaluate_mean_concentration_time_derivative(
    const int gp_id) const
{
  // safety check
  if (gp_id < 0) FOUR_C_THROW("Invalid macro-scale Gauss point ID!");

  // evaluate mean concentration time derivative on micro scale
  return matgp_.at(gp_id)->evaluate_mean_concentration_time_derivative();
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMicroMacroCoupling::update(const int gp_id) const
{
  // safety check
  if (gp_id < 0) FOUR_C_THROW("Invalid macro-scale Gauss point ID!");

  // update multi-scale scalar transport submaterial at macro-scale Gauss point
  matgp_.at(gp_id)->update();
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMicroMacroCoupling::output(const int gp_id) const
{
  // safety check
  if (gp_id < 0) FOUR_C_THROW("Invalid macro-scale Gauss point ID!");

  // create output on micro scale
  matgp_.at(gp_id)->output();
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMicroMacroCoupling::read_restart(const int gp_id) const
{
  // safety check
  if (gp_id < 0) FOUR_C_THROW("Invalid macro-scale Gauss point ID!");

  // read restart on micro scale
  matgp_.at(gp_id)->read_restart();
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::ScatraMicroMacroCoupling::set_time_stepping(
    const int gp_id, const double dt, const double time, const int step)
{
  matgp_.at(gp_id)->set_time_stepping(dt, time, step);
}

FOUR_C_NAMESPACE_CLOSE
