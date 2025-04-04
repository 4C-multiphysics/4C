// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_expl_abx.hpp"

#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <int t_order>
Solid::EXPLICIT::AdamsBashforthX<t_order>::AdamsBashforthX()
    : fvisconp_ptr_(nullptr),
      fviscon_ptr_(nullptr),
      finertianp_ptr_(nullptr),
      finertian_ptr_(nullptr),
      compute_phase_(0)
{
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <int t_order>
void Solid::EXPLICIT::AdamsBashforthX<t_order>::setup()
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

  // ---------------------------------------------------------------------------
  // resizing of multi-step quantities
  // ---------------------------------------------------------------------------
  constexpr int nhist = t_order - 1;
  global_state().get_multi_time()->resize(-nhist, 0, true);
  global_state().get_delta_time()->resize(-nhist, 0, true);
  global_state().get_multi_dis()->resize(-nhist, 0, global_state().dof_row_map_view(), true);
  global_state().get_multi_vel()->resize(-nhist, 0, global_state().dof_row_map_view(), true);
  global_state().get_multi_acc()->resize(-nhist, 0, global_state().dof_row_map_view(), true);

  // here we initialized the dt of previous steps in the database, since a resize is performed
  const double dt = (*global_state().get_delta_time())[0];
  for (int i = 0; i < nhist; ++i) global_state().get_delta_time()->update_steps(dt);

  // -------------------------------------------------------------------
  // set initial displacement
  // -------------------------------------------------------------------
  set_initial_displacement(
      tim_int().get_data_sdyn().get_initial_disp(), tim_int().get_data_sdyn().start_func_no());

  // Has to be set before the post_setup() routine is called!
  issetup_ = true;

  // Set the compute phase to compute the initial value
  compute_phase_ = 0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <int t_order>
void Solid::EXPLICIT::AdamsBashforthX<t_order>::post_setup()
{
  check_init_setup();
  compute_mass_matrix_and_init_acc();

  model_eval().post_setup();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <int t_order>
void Solid::EXPLICIT::AdamsBashforthX<t_order>::set_state(const Core::LinAlg::Vector<double>& x)
{
  check_init_setup();

  // ---------------------------------------------------------------------------
  // new end-point acceleration
  // ---------------------------------------------------------------------------
  std::shared_ptr<Core::LinAlg::Vector<double>> accnp_ptr = global_state().extract_displ_entries(x);
  global_state().get_acc_np()->scale(1.0, *accnp_ptr);
  if (compute_phase_ < t_order)
  {
    const double dt = (*global_state().get_delta_time())[0];

    // ---------------------------------------------------------------------------
    // new end-point velocities
    // ---------------------------------------------------------------------------
    global_state().get_vel_np()->update(1.0, *global_state().get_vel_n(), 0.0);
    global_state().get_vel_np()->update(dt, *global_state().get_acc_n(), 1.0);

    // ---------------------------------------------------------------------------
    // new end-point displacements
    // ---------------------------------------------------------------------------
    global_state().get_dis_np()->update(1.0, *global_state().get_dis_n(), 0.0);
    global_state().get_dis_np()->update(dt, *global_state().get_vel_np(), 1.0);
  }
  else
  {
    constexpr int nhist = t_order - 1;

    const double dt = (*global_state().get_delta_time())[0];

    // At present, a variable step size for high order Adams-Bashforth is not supported due to a
    // good reference is not yet been found. The time coefficient shall be adapted for a variable
    // step size approach.
    double test = 0.0, dti = dt;
    for (int i = 0; i < nhist; ++i)
    {
      const double dti1 = (*global_state().get_delta_time())[-i - 1];
      test += std::abs(dti - dti1);
      dti = dti1;
    }

    if (test > 1.0e-13)
      FOUR_C_THROW("High Order AdamsBashforth does not currently support the variable step size.");

    // ---------------------------------------------------------------------------
    // new end-point velocities
    // ---------------------------------------------------------------------------
    global_state().get_vel_np()->update(1.0, (*(global_state().get_multi_vel()))[0], 0.0);
    for (int i = 0; i < t_order; ++i)
    {
      double c = AdamsBashforthHelper<t_order>::exc[i];
      global_state().get_vel_np()->update(c * dt, (*(global_state().get_multi_acc()))[-i], 1.0);
    }

    // ---------------------------------------------------------------------------
    // new end-point displacements
    // ---------------------------------------------------------------------------
    global_state().get_dis_np()->update(1.0, (*(global_state().get_multi_dis()))[0], 0.0);
    for (int i = 0; i < t_order; ++i)
    {
      double c = AdamsBashforthHelper<t_order>::exc[i];
      global_state().get_dis_np()->update(c * dt, (*(global_state().get_multi_vel()))[-i], 1.0);
    }
  }

  // ---------------------------------------------------------------------------
  // update the elemental state
  // ---------------------------------------------------------------------------
  model_eval().update_residual();
  model_eval().run_recover();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <int t_order>
void Solid::EXPLICIT::AdamsBashforthX<t_order>::add_visco_mass_contributions(
    Core::LinAlg::Vector<double>& f) const
{
  // do not add damping forces for material damping to residual as already done on element level
  if (tim_int().get_data_sdyn().get_damping_type() == Inpar::Solid::damp_material) return;

  // viscous damping forces at t_{n+1}
  Core::LinAlg::assemble_my_vector(1.0, f, 1.0, *fvisconp_ptr_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <int t_order>
void Solid::EXPLICIT::AdamsBashforthX<t_order>::add_visco_mass_contributions(
    Core::LinAlg::SparseOperator& jac) const
{
  std::shared_ptr<Core::LinAlg::SparseMatrix> stiff_ptr = global_state().extract_displ_block(jac);
  // set mass matrix
  stiff_ptr->add(*global_state().get_mass_matrix(), false, 1.0, 0.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <int t_order>
void Solid::EXPLICIT::AdamsBashforthX<t_order>::write_restart(
    Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  check_init_setup();
  // write dynamic forces
  iowriter.write_vector("finert", finertian_ptr_);
  iowriter.write_vector("fvisco", fviscon_ptr_);

  // write compute phase
  iowriter.write_int("compute_phase", compute_phase_);

  // write velocities and accelerations
  if (compute_phase_ >= t_order)
  {
    for (int i = 0; i < t_order; ++i)
    {
      std::stringstream velname;
      velname << "histvel_" << i;
      std::shared_ptr<const Core::LinAlg::Vector<double>> vel_ptr_ =
          Core::Utils::shared_ptr_from_ref<const Core::LinAlg::Vector<double>>(
              (*(global_state().get_multi_vel()))[-i]);
      iowriter.write_vector(velname.str(), vel_ptr_);

      std::stringstream accname;
      accname << "histacc_" << i;
      std::shared_ptr<const Core::LinAlg::Vector<double>> acc_ptr_ =
          Core::Utils::shared_ptr_from_ref<const Core::LinAlg::Vector<double>>(
              (*(global_state().get_multi_acc()))[-i]);
      iowriter.write_vector(accname.str(), acc_ptr_);
    }
  }

  model_eval().write_restart(iowriter, forced_writerestart);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <int t_order>
void Solid::EXPLICIT::AdamsBashforthX<t_order>::read_restart(
    Core::IO::DiscretizationReader& ioreader)
{
  check_init_setup();
  // read dynamic forces
  ioreader.read_vector(finertian_ptr_, "finert");
  ioreader.read_vector(fviscon_ptr_, "fvisco");

  // read compute phase
  if (ioreader.has_int("compute_phase"))
  {
    compute_phase_ = ioreader.read_int("compute_phase");
  }
  else
  {
    compute_phase_ = 0;
  }

  // read velocities and accelerations
  if (compute_phase_ >= t_order)
  {
    for (int i = t_order - 1; i >= 0; --i)
    {
      std::stringstream velname;
      velname << "histvel_" << i;
      std::shared_ptr<Core::LinAlg::Vector<double>> vel_ptr =
          std::make_shared<Core::LinAlg::Vector<double>>(*global_state().get_vel_n());
      ioreader.read_vector(vel_ptr, velname.str());
      global_state().get_multi_vel()->update_steps(*vel_ptr);

      std::stringstream accname;
      accname << "histacc_" << i;
      std::shared_ptr<Core::LinAlg::Vector<double>> acc_ptr =
          std::make_shared<Core::LinAlg::Vector<double>>(*global_state().get_acc_n());
      ioreader.read_vector(acc_ptr, accname.str());
      global_state().get_multi_acc()->update_steps(*acc_ptr);
    }
  }

  model_eval().read_restart(ioreader);
  update_constant_state_contributions();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <int t_order>
void Solid::EXPLICIT::AdamsBashforthX<t_order>::update_step_state()
{
  check_init_setup();

  // ---------------------------------------------------------------------------
  // dynamic effects
  // ---------------------------------------------------------------------------
  // new at t_{n+1} -> t_n
  //    finertial_{n} := finertial_{n+1}
  finertian_ptr_->scale(1.0, *finertianp_ptr_);
  // new at t_{n+1} -> t_n
  //    fviscous_{n} := fviscous_{n+1}
  fviscon_ptr_->scale(1.0, *fvisconp_ptr_);

  // ---------------------------------------------------------------------------
  // update model specific variables
  // ---------------------------------------------------------------------------
  model_eval().update_step_state(0.0);

  // update the compute phase step flag
  if (compute_phase_ < t_order) ++compute_phase_;
}

/*----------------------------------------------------------------------------*
 | Template instantiation for supported order                                 |
 *----------------------------------------------------------------------------*/
template class Solid::EXPLICIT::AdamsBashforthX<2>;
// template class Solid::EXPLICIT::AdamsBashforthX<3>;
template class Solid::EXPLICIT::AdamsBashforthX<4>;

FOUR_C_NAMESPACE_CLOSE
