// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_predict_constdisvelaccpress.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_structure_new_impl_generic.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"
#include "4C_structure_new_predict_factory.hpp"
#include "4C_structure_new_timint_base.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::Predict::ConstDisVelAccPress::ConstDisVelAccPress()
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Predict::ConstDisVelAccPress::setup()
{
  check_init();

  // fallback predictor
  tangdis_ptr_ = Solid::Predict::build_predictor(Solid::PredictorType::tangdis);
  tangdis_ptr_->init(
      get_type(), impl_int_ptr(), dbc_ptr(), global_state_ptr(), io_data_ptr(), nox_params_ptr());
  tangdis_ptr_->setup();

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Predict::ConstDisVelAccPress::compute(::NOX::Abstract::Group& grp)
{
  check_init_setup();

  std::shared_ptr<Core::LinAlg::Vector<double>>& disnp_ptr = global_state().get_dis_np();
  std::shared_ptr<Core::LinAlg::Vector<double>>& velnp_ptr = global_state().get_vel_np();
  std::shared_ptr<Core::LinAlg::Vector<double>>& accnp_ptr = global_state().get_acc_np();

  bool ok = true;
  switch (get_type())
  {
    case Solid::PredictorType::constdis:
    case Solid::PredictorType::constdispres:
    {
      impl_int().predict_const_dis_consist_vel_acc(*disnp_ptr, *velnp_ptr, *accnp_ptr);
      break;
    }
    case Solid::PredictorType::constvel:
    {
      ok = impl_int().predict_const_vel_consist_acc(*disnp_ptr, *velnp_ptr, *accnp_ptr);
      break;
    }
    case Solid::PredictorType::constacc:
    {
      ok = impl_int().predict_const_acc(*disnp_ptr, *velnp_ptr, *accnp_ptr);
      break;
    }
    case Solid::PredictorType::constdisvelacc:
    case Solid::PredictorType::constdisvelaccpres:
    {
      disnp_ptr->update(1.0, *global_state().get_dis_n(), 0.0);
      velnp_ptr->update(1.0, *global_state().get_vel_n(), 0.0);
      accnp_ptr->update(1.0, *global_state().get_acc_n(), 0.0);
      break;
    }
    default:
    {
      FOUR_C_THROW("Unsupported ConstDisVelAccPress predictor!");
      break;
    }
  }
  impl_int().model_eval().predict();

  // If the const predictors failed e.g. due to too little history information,
  // we use the tangdis predictor as fallback predictor.
  if (not ok) tangdis_ptr_->compute(grp);
}

FOUR_C_NAMESPACE_CLOSE
