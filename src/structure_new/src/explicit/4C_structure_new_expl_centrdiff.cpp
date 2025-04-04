// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_expl_centrdiff.hpp"

#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_structure_new_model_evaluator_generic.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::EXPLICIT::CentrDiff::CentrDiff()
    : fvisconp_ptr_(nullptr),
      fviscon_ptr_(nullptr),
      finertianp_ptr_(nullptr),
      finertian_ptr_(nullptr)
{
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::CentrDiff::setup()
{
  check_init();

  // Call the setup() of the abstract base class first.
  Generic::setup();

  // ---------------------------------------------------------------------------
  // setup pointers to the force vectors of the global state data container
  // ---------------------------------------------------------------------------
  finertian_ptr_ = global_state().get_finertial_n();
  finertianp_ptr_ = global_state().get_finertial_np();

  fviscon_ptr_ = global_state().get_fvisco_n();
  fvisconp_ptr_ = global_state().get_fvisco_np();

  // -------------------------------------------------------------------
  // set initial displacement
  // -------------------------------------------------------------------
  set_initial_displacement(
      tim_int().get_data_sdyn().get_initial_disp(), tim_int().get_data_sdyn().start_func_no());

  // Has to be set before the post_setup() routine is called!
  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::CentrDiff::post_setup()
{
  check_init_setup();
  compute_mass_matrix_and_init_acc();

  model_eval().post_setup();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::CentrDiff::set_state(const Core::LinAlg::Vector<double>& x)
{
  check_init_setup();

  const double dt = (*global_state().get_delta_time())[0];
  const double dthalf = dt / 2.0;

  // ---------------------------------------------------------------------------
  // new end-point acceleration
  // ---------------------------------------------------------------------------
  std::shared_ptr<Core::LinAlg::Vector<double>> accnp_ptr = global_state().extract_displ_entries(x);
  global_state().get_acc_np()->scale(1.0, *accnp_ptr);

  // ---------------------------------------------------------------------------
  // new half-point velocities
  // ---------------------------------------------------------------------------
  global_state().get_vel_np()->update(1.0, *global_state().get_vel_n(), 0.0);
  global_state().get_vel_np()->update(dthalf, *global_state().get_acc_n(), 1.0);

  // ---------------------------------------------------------------------------
  // new end-point displacements
  // ---------------------------------------------------------------------------
  global_state().get_dis_np()->update(1.0, *global_state().get_dis_n(), 0.0);
  global_state().get_dis_np()->update(dt, *global_state().get_vel_np(), 1.0);

  // ---------------------------------------------------------------------------
  // update the elemental state
  // ---------------------------------------------------------------------------
  model_eval().update_residual();
  model_eval().run_recover();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::CentrDiff::add_visco_mass_contributions(Core::LinAlg::Vector<double>& f) const
{
  // do not add damping forces for material damping to residual as already done on element level
  if (tim_int().get_data_sdyn().get_damping_type() == Inpar::Solid::damp_material) return;

  // viscous damping forces at t_{n+1}
  Core::LinAlg::assemble_my_vector(1.0, f, 1.0, *fvisconp_ptr_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::CentrDiff::add_visco_mass_contributions(
    Core::LinAlg::SparseOperator& jac) const
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> stiff_ptr = global_state().extract_displ_block(jac);
  // set mass matrix
  stiff_ptr->add(*global_state().get_mass_matrix(), false, 1.0, 0.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::CentrDiff::write_restart(
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
void Solid::EXPLICIT::CentrDiff::read_restart(Core::IO::DiscretizationReader& ioreader)
{
  check_init_setup();
  ioreader.read_vector(finertian_ptr_, "finert");
  ioreader.read_vector(fviscon_ptr_, "fvisco");

  model_eval().read_restart(ioreader);
  update_constant_state_contributions();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::CentrDiff::update_step_state()
{
  check_init_setup();

  const double dt = (*global_state().get_delta_time())[0];
  const double dthalf = dt / 2.0;

  // ---------------------------------------------------------------------------
  // dynamic effects
  // ---------------------------------------------------------------------------
  // new at t_{n+1} -> t_n
  //    finertial_{n} := finertial_{n+1}
  finertian_ptr_->scale(1.0, *finertianp_ptr_);
  // new at t_{n+1} -> t_n
  //    fviscous_{n} := fviscous_{n+1}
  fviscon_ptr_->scale(1.0, *fvisconp_ptr_);

  // recompute the velocity to account for new acceleration
  global_state().get_vel_np()->update(1.0, *global_state().get_vel_n(), 0.0);
  global_state().get_vel_np()->update(dthalf, *global_state().get_acc_n(), 1.0);
  global_state().get_vel_np()->update(dthalf, *global_state().get_acc_np(), 1.0);

  // ---------------------------------------------------------------------------
  // update model specific variables
  // ---------------------------------------------------------------------------
  model_eval().update_step_state(0.0);
}

FOUR_C_NAMESPACE_CLOSE
