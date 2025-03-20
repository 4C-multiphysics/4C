// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_impl_genalpha_liegroup.hpp"

#include "4C_io.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_utils.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::IMPLICIT::GenAlphaLieGroup::GenAlphaLieGroup() : accn_mod_(nullptr)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::GenAlphaLieGroup::setup()
{
  check_init();

  // ---------------------------------------------------------------------------
  // setup additional state vectors of modified acceleration
  // ---------------------------------------------------------------------------
  accn_mod_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*global_state().dof_row_map_view(), true);

  // Call the setup() of the parent GenAlpha class
  GenAlpha::setup();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::GenAlphaLieGroup::post_setup()
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
void Solid::IMPLICIT::GenAlphaLieGroup::set_state(const Core::LinAlg::Vector<double>& x)
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

  /* ToDo in case we want to handle rotation vector DoFs correctly on time
   *      integrator level, the update procedure needs to be adapted here;
   *      use global_state().ExportAdditiveEntries() and ExportRotvecEntries() for
   *      this */

  // ---------------------------------------------------------------------------
  // new end-point velocities
  // ---------------------------------------------------------------------------
  global_state().get_vel_np()->update(
      1.0, (*const_vel_acc_update_ptr_)(0), gamma_ / (beta_ * dt), *disnp_ptr, 0.0);

  // ---------------------------------------------------------------------------
  // new end-point accelerations
  // ---------------------------------------------------------------------------
  global_state().get_acc_np()->update(1.0, (*const_vel_acc_update_ptr_)(1),
      (1.0 - alpham_) / (beta_ * dt * dt * (1.0 - alphaf_)), *disnp_ptr, 0.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::GenAlphaLieGroup::write_restart(
    Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  check_init_setup();
  // write modified acceleration vector
  iowriter.write_vector("accn_mod", accn_mod_);

  GenAlpha::write_restart(iowriter, forced_writerestart);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::GenAlphaLieGroup::read_restart(Core::IO::DiscretizationReader& ioreader)
{
  check_init_setup();
  ioreader.read_vector(accn_mod_, "accn_mod");

  GenAlpha::read_restart(ioreader);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::GenAlphaLieGroup::update_step_state()
{
  check_init_setup();

  // new at t_{n+1} -> t_n
  //    acc_mod_{n} := -alpha_m/(1-alpha_m) * acc_mod_{n}
  accn_mod_->scale(-alpham_ / (1.0 - alpham_));
  accn_mod_->update(alphaf_ / (1.0 - alpham_), *global_state().get_acc_n(), 1.0);
  accn_mod_->update((1.0 - alphaf_) / (1.0 - alpham_), *global_state().get_acc_np(), 1.0);

  // ---------------------------------------------------------------------------
  // update model specific variables
  // ---------------------------------------------------------------------------
  model_eval().update_step_state(0.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::GenAlphaLieGroup::update_constant_state_contributions()
{
  const double& dt = (*global_state().get_delta_time())[0];

  /* ToDo in case we want to handle rotation vector DoFs correctly on time
   *      integrator level, the update procedure needs to be adapted here;
   *      use global_state().ExportAdditiveEntries() and ExportRotvecEntries() for
   *      this */

  // ---------------------------------------------------------------------------
  // velocity
  // ---------------------------------------------------------------------------
  (*const_vel_acc_update_ptr_)(0).scale((1.0 - gamma_ / (2.0 * beta_)) * dt, *accn_mod_);
  (*const_vel_acc_update_ptr_)(0).update(1.0 - gamma_ / beta_, *global_state().get_vel_n(), 1.0);
  (*const_vel_acc_update_ptr_)(0).update(-gamma_ / (beta_ * dt), *global_state().get_dis_n(), 1.0);

  // ---------------------------------------------------------------------------
  // acceleration
  // ---------------------------------------------------------------------------
  (*const_vel_acc_update_ptr_)(1).scale(alphaf_ / (alphaf_ - 1.0), *global_state().get_acc_n());
  (*const_vel_acc_update_ptr_)(1).update(
      alpham_ / (1.0 - alphaf_) - (1.0 - alpham_) * (0.5 - beta_) / (beta_ * (1.0 - alphaf_)),
      *accn_mod_, 1.0);
  (*const_vel_acc_update_ptr_)(1).update(
      -(1.0 - alpham_) / (beta_ * dt * (1.0 - alphaf_)), *global_state().get_vel_n(), 1.0);
  (*const_vel_acc_update_ptr_)(1).update(
      -(1.0 - alpham_) / (beta_ * dt * dt * (1.0 - alphaf_)), *global_state().get_dis_n(), 1.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::IMPLICIT::GenAlphaLieGroup::get_int_param() const
{
  // access the alphaf value even if the time integrator has not yet been setup
  return 0.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::GenAlphaLieGroup::add_visco_mass_contributions(
    Core::LinAlg::Vector<double>& f) const
{
  // the following is only done for rayleigh damping as for material damping viscous forces are
  // already added at element level and else would be added twice
  if (tim_int().get_data_sdyn().get_damping_type() == Inpar::Solid::damp_rayleigh)
  {
    // viscous damping forces at t_{n+1}
    Core::LinAlg::assemble_my_vector(1.0, f, 1.0, *fvisconp_ptr_);
  }

  // inertia forces at t_{n+1}
  Core::LinAlg::assemble_my_vector(1.0, f, 1.0, *finertianp_ptr_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::GenAlphaLieGroup::add_visco_mass_contributions(
    Core::LinAlg::SparseOperator& jac) const
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> stiff_ptr = global_state().extract_displ_block(jac);
  const double& dt = (*global_state().get_delta_time())[0];
  // add inertial contributions to structural stiffness block
  stiff_ptr->add(*global_state().get_mass_matrix(), false,
      (1.0 - alpham_) / (beta_ * dt * dt * (1.0 - alphaf_)), 1.0);
  // add damping contributions
  if (tim_int().get_data_sdyn().get_damping_type() != Inpar::Solid::damp_none)
    stiff_ptr->add(*global_state().get_damp_matrix(), false, gamma_ / (beta_ * dt), 1.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::GenAlphaLieGroup::predict_const_dis_consist_vel_acc(
    Core::LinAlg::Vector<double>& disnp, Core::LinAlg::Vector<double>& velnp,
    Core::LinAlg::Vector<double>& accnp) const
{
  check_init_setup();
  std::shared_ptr<const Core::LinAlg::Vector<double>> disn = global_state().get_dis_n();
  std::shared_ptr<const Core::LinAlg::Vector<double>> veln = global_state().get_vel_n();
  std::shared_ptr<const Core::LinAlg::Vector<double>> accn = global_state().get_acc_n();
  const double& dt = (*global_state().get_delta_time())[0];

  // constant predictor: displacement in domain
  disnp.update(1.0, *disn, 0.0);

  // consistent velocities following Newmark formulas
  velnp.update(1.0, disnp, -1.0, *disn, 0.0);
  velnp.update((beta_ - gamma_) / beta_, *veln, (2.0 * beta_ - gamma_) * dt / (2.0 * beta_),
      *accn_mod_, gamma_ / (beta_ * dt));

  // consistent accelerations following Newmark formulas
  accnp.update(1.0, disnp, -1.0, *disn, 0.0);
  accnp.update(-(1.0 - alpham_) / (beta_ * dt * (1 - alphaf_)), *veln, -alphaf_ / (1.0 - alphaf_),
      *accn, (1.0 - alpham_) / (beta_ * dt * dt * (1.0 - alphaf_)));
  accnp.update(
      alpham_ / (1.0 - alphaf_) - (1.0 - alpham_) * (0.5 - beta_) / (beta_ * (1.0 - alphaf_)),
      *accn_mod_, 1.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::IMPLICIT::GenAlphaLieGroup::predict_const_vel_consist_acc(
    Core::LinAlg::Vector<double>& disnp, Core::LinAlg::Vector<double>& velnp,
    Core::LinAlg::Vector<double>& accnp) const
{
  check_init_setup();

  FOUR_C_THROW(
      "Predictor ConstVelConsistAcc is not supported in Lie group GenAlpha so"
      " far! Use ConstDisConsistVelAcc!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::IMPLICIT::GenAlphaLieGroup::predict_const_acc(Core::LinAlg::Vector<double>& disnp,
    Core::LinAlg::Vector<double>& velnp, Core::LinAlg::Vector<double>& accnp) const
{
  check_init_setup();

  FOUR_C_THROW(
      "Predictor ConstAcc is not supported in Lie group GenAlpha so far! Use "
      "ConstDisConsistVelAcc!");

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::GenAlphaLieGroup::reset_eval_params()
{
  // set the time step dependent parameters for the element evaluation
  GenAlpha::reset_eval_params();

  /* in case we have non-additive rotation (pseudo-)vector DOFs, we need to pass
   * the GenAlpha parameters to the beam elements via beam parameter interface */
  if (tim_int().get_data_sdyn().get_mass_lin_type() == Inpar::Solid::MassLin::ml_rotations)
  {
    eval_data().get_beam_data().set_beta(beta_);
    eval_data().get_beam_data().set_gamma(gamma_);
    eval_data().get_beam_data().set_alphaf(alphaf_);
    eval_data().get_beam_data().set_alpham(alpham_);
  }
}

FOUR_C_NAMESPACE_CLOSE
