// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_impl_statics.hpp"

#include "4C_io.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"
#include "4C_structure_new_model_evaluator_structure.hpp"
#include "4C_structure_new_predict_generic.hpp"
#include "4C_structure_new_timint_implicit.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::IMPLICIT::Statics::Statics()
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Statics::setup()
{
  check_init();

  // Call the setup() of the abstract base class first.
  Generic::setup();

  // check for valid parameter combinations:
  if (eval_data().get_damping_type() != Solid::damp_none)
    FOUR_C_THROW("ERROR: Damping not provided for statics time integration!");

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Statics::post_setup()
{
  check_init_setup();
  // no acceleration to equilibriate

  model_eval().post_setup();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Statics::set_state(const Core::LinAlg::Vector<double>& x)
{
  check_init_setup();
  if (is_predictor_state()) return;

  std::shared_ptr<Core::LinAlg::Vector<double>> disnp_ptr = global_state().extract_displ_entries(x);
  global_state().get_dis_np()->scale(1.0, *disnp_ptr);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::IMPLICIT::Statics::apply_force(
    const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& f)
{
  check_init_setup();
  reset_eval_params();
  return model_eval().apply_force(x, f, 1.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::IMPLICIT::Statics::apply_stiff(
    const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac)
{
  check_init_setup();
  reset_eval_params();
  bool ok = model_eval().apply_stiff(x, jac, 1.0);
  jac.complete();
  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::IMPLICIT::Statics::apply_force_stiff(const Core::LinAlg::Vector<double>& x,
    Core::LinAlg::Vector<double>& f, Core::LinAlg::SparseOperator& jac)
{
  check_init_setup();
  reset_eval_params();
  bool ok = model_eval().apply_force_stiff(x, f, jac, 1.0);
  jac.complete();
  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::IMPLICIT::Statics::assemble_force(Core::LinAlg::Vector<double>& f,
    const std::vector<Solid::ModelType>* without_these_models) const
{
  check_init_setup();
  return model_eval().assemble_force(1.0, f, without_these_models);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Statics::write_restart(
    Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  check_init_setup();

  // create empty dynamic forces
  auto finertialn =
      std::make_shared<Core::LinAlg::Vector<double>>(*global_state().dof_row_map_view(), true);
  auto fviscon =
      std::make_shared<Core::LinAlg::Vector<double>>(*global_state().dof_row_map_view(), true);

  // write dynamic forces, so that it can be used later on for restart dynamics analysis
  iowriter.write_vector("finert", finertialn);
  iowriter.write_vector("fvisco", fviscon);

  model_eval().write_restart(iowriter, forced_writerestart);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Statics::read_restart(Core::IO::DiscretizationReader& ioreader)
{
  check_init_setup();
  model_eval().read_restart(ioreader);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::IMPLICIT::Statics::calc_ref_norm_force(
    const enum ::NOX::Abstract::Vector::NormType& type) const
{
  check_init_setup();

  const std::shared_ptr<Core::LinAlg::Vector<double>> fintnp =
      std::const_pointer_cast<Core::LinAlg::Vector<double>>(global_state().get_fint_np());
  const std::shared_ptr<Core::LinAlg::Vector<double>> fextnp =
      std::const_pointer_cast<Core::LinAlg::Vector<double>>(global_state().get_fext_np());
  const std::shared_ptr<Core::LinAlg::Vector<double>> freactnp =
      std::const_pointer_cast<Core::LinAlg::Vector<double>>(global_state().get_freact_np());

  // switch from Core::LinAlg::Vector<double> to NOX::Nln::Vector (view but read-only)
  const NOX::Nln::Vector fintnp_nox_ptr(fintnp, NOX::Nln::Vector::MemoryType::View);
  const NOX::Nln::Vector fextnp_nox_ptr(fextnp, NOX::Nln::Vector::MemoryType::View);
  const NOX::Nln::Vector freactnp_nox_ptr(freactnp, NOX::Nln::Vector::MemoryType::View);

  // norm of the internal forces
  double fintnorm = fintnp_nox_ptr.norm(type);

  // norm of the external forces
  double fextnorm = fextnp_nox_ptr.norm(type);

  // norm of reaction forces
  double freactnorm = freactnp_nox_ptr.norm(type);

  // return characteristic norm
  return std::max(fintnorm, std::max(fextnorm, freactnorm));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::IMPLICIT::Statics::get_int_param() const { return 0.0; }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Statics::pre_update()
{
  check_init_setup();
  const Solid::TimeInt::Implicit* impl_ptr =
      dynamic_cast<const Solid::TimeInt::Implicit*>(&tim_int());
  if (impl_ptr == nullptr) return;

  // get the time step size
  const double dt = global_state().get_delta_time()[0];

  const Solid::PredictorType& pred_type = impl_ptr->predictor().get_type();
  std::shared_ptr<Core::LinAlg::Vector<double>>& accnp_ptr = global_state().get_acc_np();
  std::shared_ptr<Core::LinAlg::Vector<double>>& velnp_ptr = global_state().get_vel_np();

  switch (pred_type)
  {
    // case: constant acceleration
    case Solid::PredictorType::constacc:
    {
      // read-only access
      std::shared_ptr<const Core::LinAlg::Vector<double>> veln_ptr = global_state().get_vel_n();
      // update the pseudo acceleration (statics!)
      accnp_ptr->update(1.0 / dt, *velnp_ptr, -1.0 / dt, *veln_ptr, 0.0);

      [[fallthrough]];
    }
    // case: constant acceleration OR constant velocity
    case Solid::PredictorType::constvel:
    {
      // read-only access
      std::shared_ptr<const Core::LinAlg::Vector<double>> disn_ptr = global_state().get_dis_n();
      std::shared_ptr<const Core::LinAlg::Vector<double>> disnp_ptr = global_state().get_dis_np();
      // update the pseudo velocity (statics!)
      velnp_ptr->update(1.0 / dt, *disnp_ptr, -1.0 / dt, *disn_ptr, 0.0);
      // ATTENTION: Break for both cases!
      break;
    }
    default:
      /* do nothing */
      break;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Statics::update_step_state()
{
  check_init_setup();
  // update model specific variables
  model_eval().update_step_state(0.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Statics::update_step_element()
{
  check_init_setup();
  model_eval().update_step_element();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Statics::predict_const_dis_consist_vel_acc(
    Core::LinAlg::Vector<double>& disnp, Core::LinAlg::Vector<double>& velnp,
    Core::LinAlg::Vector<double>& accnp) const
{
  check_init_setup();
  // constant predictor : displacement in domain
  disnp.update(1.0, *global_state().get_dis_n(), 0.0);
  // new end-point velocities, these stay zero in static calculation
  velnp.put_scalar(0.0);
  // new end-point accelerations, these stay zero in static calculation
  accnp.put_scalar(0.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::IMPLICIT::Statics::predict_const_vel_consist_acc(Core::LinAlg::Vector<double>& disnp,
    Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const
{
  check_init_setup();
  // If there is not enough history information, return a fail status.
  if (global_state().get_step_n() == 0) return false;

  // Displacement increment over last time step
  std::shared_ptr<Core::LinAlg::Vector<double>> disp_inc =
      std::make_shared<Core::LinAlg::Vector<double>>(*global_state().dof_row_map_view(), true);
  disp_inc->update(global_state().get_delta_time()[0], *global_state().get_vel_n(), 0.);
  // apply the dbc on the auxiliary vector
  tim_int().get_dbc().apply_dirichlet_to_vector(*disp_inc);
  // update the solution variables
  disnp.update(1.0, *global_state().get_dis_n(), 0.0);
  disnp.update(1.0, *disp_inc, 1.0);
  velnp.update(1.0, *global_state().get_vel_n(), 0.0);
  accnp.update(1.0, *global_state().get_acc_n(), 0.0);

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::IMPLICIT::Statics::predict_const_acc(Core::LinAlg::Vector<double>& disnp,
    Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const
{
  check_init_setup();
  // If there is not enough history information try a different predictor with
  // less requirements.
  if (global_state().get_step_n() < 2) return predict_const_vel_consist_acc(disnp, velnp, accnp);

  // Displacement increment over last time step
  std::shared_ptr<Core::LinAlg::Vector<double>> disp_inc =
      std::make_shared<Core::LinAlg::Vector<double>>(*global_state().dof_row_map_view(), true);
  const double& dt = global_state().get_delta_time()[0];
  disp_inc->update(dt, *global_state().get_vel_n(), 0.);
  disp_inc->update(0.5 * dt * dt, *global_state().get_acc_n(), 1.0);
  // apply the dbc on the auxiliary vector
  tim_int().get_dbc().apply_dirichlet_to_vector(*disp_inc);
  // update the solution variables
  disnp.update(1.0, *global_state().get_dis_n(), 0.0);
  disnp.update(1., *disp_inc, 1.);
  velnp.update(1.0, *global_state().get_vel_n(), 0.0);
  accnp.update(1.0, *global_state().get_acc_n(), 0.0);

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Statics::reset_eval_params()
{
  // call base class
  Solid::IMPLICIT::Generic::reset_eval_params();
}

FOUR_C_NAMESPACE_CLOSE
