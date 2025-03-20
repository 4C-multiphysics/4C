// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_impl_ost.hpp"

#include "4C_io.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_basedatasdyn.hpp"
#include "4C_structure_new_utils.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::IMPLICIT::OneStepTheta::OneStepTheta()
    : theta_(-1.0),
      fvisconp_ptr_(nullptr),
      fviscon_ptr_(nullptr),
      const_vel_acc_update_ptr_(nullptr),
      finertian_ptr_(nullptr),
      finertianp_ptr_(nullptr)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::OneStepTheta::setup()
{
  check_init();
  // Call the setup() of the abstract base class first.
  Generic::setup();

  // ---------------------------------------------------------------------------
  // setup time integration parameters
  // ---------------------------------------------------------------------------
  // get a copy of the input parameters
  theta_ = get_theta();

  // sanity checks and some screen output
  if (global_state().get_my_rank() == 0)
  {
    if ((theta_ <= 0.0) or (theta_ > 1.0))
      FOUR_C_THROW("theta out of range (0.0,1.0]");
    else
      std::cout << "   theta = " << theta_ << std::endl;
  }

  // ---------------------------------------------------------------------------
  // setup mid-point vectors
  // ---------------------------------------------------------------------------
  const_vel_acc_update_ptr_ = std::make_shared<Core::LinAlg::MultiVector<double>>(
      *global_state().dof_row_map_view(), 2, true);

  // ---------------------------------------------------------------------------
  // setup pointers to the force vectors of the global state data container
  // ---------------------------------------------------------------------------
  finertian_ptr_ = global_state().get_finertial_n();
  finertianp_ptr_ = global_state().get_finertial_np();

  fviscon_ptr_ = global_state().get_fvisco_n();
  fvisconp_ptr_ = global_state().get_fvisco_np();

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::OneStepTheta::post_setup()
{
  check_init_setup();

  if (sdyn().get_mass_lin_type() != Inpar::Solid::MassLin::ml_rotations and
      !sdyn().neglect_inertia())
  {
    /* we can use this method for all elements with additive DoFs,
     * but it won't work like this for non-additive rotation vector DoFs */
    compute_mass_matrix_and_init_acc();
  }
  else
  {
    /* If we are restarting the simulation, we get the acceleration state from the
     * restart file. So we are already done at this point. */
    if (tim_int().is_restarting()) return;

    // so far, we are restricted to vanishing initial accelerations
    std::shared_ptr<Core::LinAlg::Vector<double>> accnp_ptr = global_state().get_acc_np();
    accnp_ptr->put_scalar(0.0);

    // sanity check whether assumption is fulfilled
    /* ToDo tolerance value is experience and based on following consideration:
     * epsilon = O(1e-15) scaled with EA = O(1e8) yields residual contributions in
     * initial, stress free state of order 1e-8 */
    if (not current_state_is_equilibrium(1.0e-6) and global_state().get_my_rank() == 0)
      std::cout << "\nSERIOUS WARNING: Initially non vanishing acceleration states "
                   "in case of ml_rotation = true,\ni.e. an initial state where the system "
                   "is not equilibrated, cannot yet be computed correctly.\nThis means your "
                   "results in the beginning are not physically correct\n"
                << std::endl;

    // call update routines to copy states from t_{n+1} to t_{n}
    // note that the time step is not incremented
    pre_update();
    update_step_state();
    update_step_element();
    post_update();
  }

  model_eval().post_setup();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::IMPLICIT::OneStepTheta::get_theta() const
{
  if (is_init() and is_setup()) return theta_;

  const Solid::TimeInt::OneStepThetaDataSDyn& onesteptheta_sdyn =
      dynamic_cast<const Solid::TimeInt::OneStepThetaDataSDyn&>(tim_int().get_data_sdyn());

  return onesteptheta_sdyn.get_theta();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::OneStepTheta::set_state(const Core::LinAlg::Vector<double>& x)
{
  check_init_setup();

  if (is_predictor_state()) return;

  update_constant_state_contributions();

  const double& dt = (*global_state().get_delta_time())[0];
  // ---------------------------------------------------------------------------
  // new end-point displacements
  // ---------------------------------------------------------------------------
  std::shared_ptr<Core::LinAlg::Vector<double>> disnp_ptr = global_state().extract_displ_entries(x);
  global_state().get_dis_np()->scale(1.0, *disnp_ptr);

  // ---------------------------------------------------------------------------
  // new end-point velocities
  // ---------------------------------------------------------------------------
  global_state().get_vel_np()->update(
      1.0, (*const_vel_acc_update_ptr_)(0), 1.0 / (theta_ * dt), *disnp_ptr, 0.0);

  // ---------------------------------------------------------------------------
  // new end-point accelerations
  // ---------------------------------------------------------------------------
  global_state().get_acc_np()->update(
      1.0, (*const_vel_acc_update_ptr_)(1), 1.0 / (theta_ * theta_ * dt * dt), *disnp_ptr, 0.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::OneStepTheta::update_constant_state_contributions()
{
  const double& dt = (*global_state().get_delta_time())[0];

  // ---------------------------------------------------------------------------
  // velocity
  // ---------------------------------------------------------------------------
  (*const_vel_acc_update_ptr_)(0).scale(-(1.0 - theta_) / theta_, *global_state().get_vel_n());
  (*const_vel_acc_update_ptr_)(0).update(-1.0 / (theta_ * dt), *global_state().get_dis_n(), 1.0);

  // ---------------------------------------------------------------------------
  // acceleration
  // ---------------------------------------------------------------------------
  (*const_vel_acc_update_ptr_)(1).scale(-(1.0 - theta_) / theta_, *global_state().get_acc_n());
  (*const_vel_acc_update_ptr_)(1).update(
      -1.0 / (theta_ * theta_ * dt), *global_state().get_vel_n(), 1.0);
  (*const_vel_acc_update_ptr_)(1).update(
      -1.0 / (theta_ * theta_ * dt * dt), *global_state().get_dis_n(), 1.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::IMPLICIT::OneStepTheta::apply_force(
    const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& f)
{
  check_init_setup();

  // ---------------------------------------------------------------------------
  // evaluate the different model types (static case) at t_{n+1}^{i}
  // ---------------------------------------------------------------------------
  // set the time step dependent parameters for the element evaluation
  reset_eval_params();
  return model_eval().apply_force(x, f, theta_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::IMPLICIT::OneStepTheta::apply_stiff(
    const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac)
{
  check_init_setup();

  // ---------------------------------------------------------------------------
  // evaluate the different model types (static case) at t_{n+1}^{i}
  // ---------------------------------------------------------------------------
  // set the time step dependent parameters for the element evaluation
  reset_eval_params();
  const bool ok = model_eval().apply_stiff(x, jac, theta_);

  if (not ok) return ok;

  jac.complete();

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::IMPLICIT::OneStepTheta::apply_force_stiff(const Core::LinAlg::Vector<double>& x,
    Core::LinAlg::Vector<double>& f, Core::LinAlg::SparseOperator& jac)
{
  check_init_setup();
  // ---------------------------------------------------------------------------
  // evaluate the different model types (static case) at t_{n+1}^{i}
  // ---------------------------------------------------------------------------
  // set the time step dependent parameters for the element evaluation
  reset_eval_params();
  const bool ok = model_eval().apply_force_stiff(x, f, jac, theta_);

  if (not ok) return ok;

  jac.complete();

  return ok;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::IMPLICIT::OneStepTheta::assemble_force(Core::LinAlg::Vector<double>& f,
    const std::vector<Inpar::Solid::ModelType>* without_these_models) const
{
  return model_eval().assemble_force(theta_, f, without_these_models);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::OneStepTheta::add_visco_mass_contributions(
    Core::LinAlg::Vector<double>& f) const
{
  // the following is only done for rayleigh damping as for material damping viscous forces are
  // already added at element level and else would be added twice
  if (tim_int().get_data_sdyn().get_damping_type() == Inpar::Solid::damp_rayleigh)
  {
    // viscous damping forces at t_{n}
    Core::LinAlg::assemble_my_vector(1.0, f, 1.0 - theta_, *fviscon_ptr_);
    // viscous damping forces at t_{n+1}
    Core::LinAlg::assemble_my_vector(1.0, f, theta_, *fvisconp_ptr_);
  }

  // inertial forces at t_{n}
  Core::LinAlg::assemble_my_vector(1.0, f, (1.0 - theta_), *finertian_ptr_);
  // inertial forces at t_{n+1}
  Core::LinAlg::assemble_my_vector(1.0, f, (theta_), *finertianp_ptr_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::OneStepTheta::add_visco_mass_contributions(
    Core::LinAlg::SparseOperator& jac) const
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> stiff_ptr = global_state().extract_displ_block(jac);
  const double& dt = (*global_state().get_delta_time())[0];
  // add inertial contributions and scale the structural stiffness block
  stiff_ptr->add(*global_state().get_mass_matrix(), false, 1.0 / (theta_ * dt * dt), 1.0);
  // add damping contributions
  if (tim_int().get_data_sdyn().get_damping_type() != Inpar::Solid::damp_none)
    stiff_ptr->add(*global_state().get_damp_matrix(), false, 1.0 / dt, 1.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::OneStepTheta::write_restart(
    Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  check_init_setup();
  // write dynamic forces
  iowriter.write_vector("finert", finertian_ptr_);
  iowriter.write_vector("fvisco", fviscon_ptr_);

  model_eval().write_restart(iowriter, forced_writerestart);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::OneStepTheta::read_restart(Core::IO::DiscretizationReader& ioreader)
{
  check_init_setup();
  ioreader.read_vector(finertian_ptr_, "finert");
  ioreader.read_vector(fviscon_ptr_, "fvisco");

  model_eval().read_restart(ioreader);
  update_constant_state_contributions();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::IMPLICIT::OneStepTheta::calc_ref_norm_force(
    const enum ::NOX::Abstract::Vector::NormType& type) const
{
  FOUR_C_THROW("Not yet implemented! (see the Statics integration for an example)");
  return -1.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::IMPLICIT::OneStepTheta::get_int_param() const { return (1.0 - get_theta()); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::OneStepTheta::update_step_state()
{
  check_init_setup();
  // ---------------------------------------------------------------------------
  // dynamic effects
  // ---------------------------------------------------------------------------
  // new at t_{n+1} -> t_n
  //    finertial_{n} := finertial_{n+1}
  finertian_ptr_->scale(1.0, *global_state().get_finertial_np());
  // new at t_{n+1} -> t_n
  //    fviscous_{n} := fviscous_{n+1}
  fviscon_ptr_->scale(1.0, *fvisconp_ptr_);

  // ---------------------------------------------------------------------------
  // update model specific variables
  // ---------------------------------------------------------------------------
  model_eval().update_step_state(1 - theta_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::OneStepTheta::update_step_element()
{
  check_init_setup();
  model_eval().update_step_element();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::OneStepTheta::post_update() { update_constant_state_contributions(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::OneStepTheta::predict_const_dis_consist_vel_acc(
    Core::LinAlg::Vector<double>& disnp, Core::LinAlg::Vector<double>& velnp,
    Core::LinAlg::Vector<double>& accnp) const
{
  check_init_setup();
  std::shared_ptr<const Core::LinAlg::Vector<double>> disn = global_state().get_dis_n();
  std::shared_ptr<const Core::LinAlg::Vector<double>> veln = global_state().get_vel_n();
  std::shared_ptr<const Core::LinAlg::Vector<double>> accn = global_state().get_acc_n();
  const double& dt = (*global_state().get_delta_time())[0];

  // constant predictor: displacement in domain
  disnp.scale(1.0, *disn);

  // consistent velocities
  /* Since disnp and disn are equal we can skip the current
   * update part and have to consider only the old state at t_{n}.
   *           disnp-disn = 0.0                                 */
  velnp.scale(-(1.0 - theta_) / theta_, *veln);

  // consistent accelerations
  /* Since disnp and disn are equal we can skip the current
   * update part and have to consider only the old state at t_{n}.
   *           disnp-disn = 0.0                                 */
  accnp.update(-1.0 / (theta_ * theta_ * dt), *veln, -(1.0 - theta_) / theta_, *accn, 0.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::IMPLICIT::OneStepTheta::predict_const_vel_consist_acc(
    Core::LinAlg::Vector<double>& disnp, Core::LinAlg::Vector<double>& velnp,
    Core::LinAlg::Vector<double>& accnp) const
{
  check_init_setup();

  std::shared_ptr<const Core::LinAlg::Vector<double>> disn = global_state().get_dis_n();
  std::shared_ptr<const Core::LinAlg::Vector<double>> veln = global_state().get_vel_n();
  std::shared_ptr<const Core::LinAlg::Vector<double>> accn = global_state().get_acc_n();
  const double& dt = (*global_state().get_delta_time())[0];

  /* extrapolated displacements based upon constant velocities
   * d_{n+1} = d_{n} + dt * v_{n} */
  disnp.update(1.0, *disn, dt, *veln, 0.0);

  // consistent velocities
  velnp.update(1.0, disnp, -1.0, *disn, 0.0);
  velnp.update(-(1.0 - theta_) / theta_, *veln, 1.0 / (theta_ * dt));

  // consistent accelerations
  accnp.update(1.0, disnp, -1.0, *disn, 0.0);
  accnp.update(-1.0 / (theta_ * theta_ * dt), *veln, -(1.0 - theta_) / theta_, *accn,
      1.0 / (theta_ * theta_ * dt * dt));

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::IMPLICIT::OneStepTheta::predict_const_acc(Core::LinAlg::Vector<double>& disnp,
    Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const
{
  check_init_setup();

  std::shared_ptr<const Core::LinAlg::Vector<double>> disn = global_state().get_dis_n();
  std::shared_ptr<const Core::LinAlg::Vector<double>> veln = global_state().get_vel_n();
  std::shared_ptr<const Core::LinAlg::Vector<double>> accn = global_state().get_acc_n();
  const double& dt = (*global_state().get_delta_time())[0];

  /* extrapolated displacements based upon constant accelerations
   * d_{n+1} = d_{n} + dt * v_{n} + dt^2 / 2 * a_{n} */
  disnp.update(1.0, *disn, dt, *veln, 0.0);
  disnp.update(0.5 * dt * dt, *accn, 1.0);

  // consistent velocities
  velnp.update(1.0, disnp, -1.0, *disn, 0.0);
  velnp.update(-(1.0 - theta_) / theta_, *veln, 1.0 / (theta_ * dt));

  // consistent accelerations
  accnp.update(1.0, disnp, -1.0, *disn, 0.0);
  accnp.update(-1.0 / (theta_ * theta_ * dt), *veln, -(1.0 - theta_) / theta_, *accn,
      1.0 / (theta_ * theta_ * dt * dt));

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::OneStepTheta::reset_eval_params()
{
  // call base class
  Solid::IMPLICIT::Generic::reset_eval_params();

  // set the time step dependent parameters for the element evaluation
  const double& dt = (*global_state().get_delta_time())[0];
  double timeintfac_dis = theta_ * theta_ * dt * dt;
  double timeintfac_vel = theta_ * dt;

  eval_data().set_tim_int_factor_disp(timeintfac_dis);
  eval_data().set_tim_int_factor_vel(timeintfac_vel);
}

FOUR_C_NAMESPACE_CLOSE
