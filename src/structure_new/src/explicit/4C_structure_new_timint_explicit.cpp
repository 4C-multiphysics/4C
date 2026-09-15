// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_timint_explicit.hpp"

#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_linearsystem.hpp"
#include "4C_structure_new_nln_solver_factory.hpp"
#include "4C_structure_new_timint_noxinterface.hpp"
#include "4C_utils_exceptions.hpp"

#include <NOX_Abstract_Group.H>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::TimeInt::Explicit::Explicit() : Solid::TimeInt::Base()
{
  // empty
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Explicit::setup()
{
  // safety check
  check_init();
  Solid::TimeInt::Base::setup();
  // ---------------------------------------------------------------------------
  // cast the base class integrator
  // ---------------------------------------------------------------------------
  explint_ptr_ = std::dynamic_pointer_cast<Solid::EXPLICIT::Generic>(integrator_ptr());
  // ---------------------------------------------------------------------------
  // build NOX interface
  // ---------------------------------------------------------------------------
  std::shared_ptr<Solid::TimeInt::NoxInterface> noxinterface_ptr =
      std::make_shared<Solid::TimeInt::NoxInterface>();
  noxinterface_ptr->init(
      data_global_state_ptr(), explint_ptr_, dbc_ptr(), Core::Utils::shared_ptr_from_ref(*this));
  noxinterface_ptr->setup();
  // ---------------------------------------------------------------------------
  // build non-linear solver
  // ---------------------------------------------------------------------------
  Solid::NonlinSolTech nlnSolverType = data_sdyn().get_nln_solver_type();
  if (nlnSolverType != Solid::soltech_singlestep)
  {
    std::cout << "WARNING!!!Nonlinear solver for explicit dynamics is given (in the input file) as "
              << nlnSolverType << ". This is not compatible. singlestep solver will be selected."
              << std::endl;
    nlnSolverType = Solid::soltech_singlestep;
  }
  nlnsolver_ptr_ = Solid::Nln::SOLVER::build_nln_solver(nlnSolverType, data_global_state_ptr(),
      data_s_dyn_ptr(), noxinterface_ptr, explint_ptr_, Core::Utils::shared_ptr_from_ref(*this));

  // set setup flag
  issetup_ = true;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Explicit::prepare_time_step()
{
  check_init_setup();

  // ToDo prepare contact for new time step
  // PrepareStepContact();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Explicit::update_state_incrementally(
    std::shared_ptr<const Core::LinAlg::Vector<double>> disiterinc)
{
  check_init_setup();
  FOUR_C_THROW(
      "All monolithically coupled problems work with implicit time "
      "integration schemes. Thus, calling evaluate() in an explicit scheme "
      "is not possible.");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Explicit::determine_stress_strain() { expl_int().determine_stress_strain(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Explicit::evaluate(
    std::shared_ptr<const Core::LinAlg::Vector<double>> disiterinc)
{
  check_init_setup();
  FOUR_C_THROW(
      "All monolithically coupled problems work with implicit time "
      "integration schemes. Thus, calling evaluate() in an explicit scheme "
      "is not possible.");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Explicit::evaluate()
{
  check_init_setup();
  throw_if_state_not_in_sync_with_nox_group();
  ::NOX::Abstract::Group& grp = nln_solver().get_solution_group();

  auto* grp_ptr = dynamic_cast<NOX::Nln::Group*>(&grp);
  if (grp_ptr == nullptr) FOUR_C_THROW("Dynamic cast failed!");

  // you definitely have to evaluate here. You might be called from a coupled
  // problem and the group might not be aware, that a different state than
  // the internally stored displacements may have changed.
  // This is a hack to get NOX to set IsValid to false.
  grp_ptr->setX(grp_ptr->getX());

  // compute the rhs vector and the stiffness matrix
  const auto status = grp_ptr->compute_f_and_jacobian();
  FOUR_C_ASSERT_ALWAYS(
      status == ::NOX::Abstract::Group::ReturnType::Ok, "Failed to compute residual and Jacobian");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Explicit::set_state(const std::shared_ptr<Core::LinAlg::Vector<double>>& x)
{
  FOUR_C_THROW(
      "All coupled problems work with implicit time "
      "integration schemes. Thus, calling set_state() in an explicit scheme "
      "is not considered, yet.");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Explicit::reset_step()
{
  // calling the base reset
  Solid::TimeInt::Base::reset_step();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::StepStatus Solid::TimeInt::Explicit::solve()
{
  check_init_setup();
  integrate_step();
  return Solid::StepStatus::no_errors;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Explicit::prepare_partition_step()
{
  // do nothing for explicit time integrators
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Explicit::update(double endtime)
{
  check_init_setup();
  FOUR_C_THROW("Not implemented. No time adaptivity available for explicit time integration.");
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Explicit::print_step()
{
  check_init_setup();

  if (data_global_state().get_my_rank() != 0 or group_id() != 0) return;

  const int stepmax = data_sdyn().get_step_max();
  const int stepn = data_global_state().get_step_n();
  const double timen = data_global_state().get_time_n();
  const double dt = data_global_state().get_delta_time()[0];
  const double wct = data_global_state().get_timer()->totalElapsedTime(true);

  // open outstd::stringstream
  std::ostringstream oss;

  /* Output of the following quantities
   * time   : total simulated time
   * dt     : used time step
   * wct    : wall clock time */
  oss << "Finalised step " << std::setw(1) << stepn;
  oss << " / " << std::setw(1) << stepmax;
  oss << " | time " << std::setw(9) << std::setprecision(3) << std::scientific << timen;
  oss << " | dt " << std::setw(9) << std::setprecision(3) << std::scientific << dt;
  oss << " | wct " << std::setw(8) << std::setprecision(2) << std::scientific << wct;
  oss << "\n--------------------------------------------------------------------------------\n";

  // print to ofile (could be done differently...)
  fprintf(stdout, "%s\n", oss.str().c_str());

  // print it, now
  fflush(stdout);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::StcScale Solid::TimeInt::Explicit::get_stc_algo()
{
  check_init_setup();
  FOUR_C_THROW("get_stc_algo() has not been tested for explicit time integration.");
  return Solid::stc_inactive;
};


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Solid::TimeInt::Explicit::get_stc_mat()
{
  check_init_setup();
  FOUR_C_THROW("get_stc_mat() has not been tested for explicit time integration.");
  return nullptr;
};


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Explicit::integrate()
{
  FOUR_C_THROW(
      "The function is unused since the Adapter::StructureTimeLoop "
      "wrapper gives you all the flexibility you need.");
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Explicit::integrate_step()
{
  check_init_setup();
  throw_if_state_not_in_sync_with_nox_group();
  // reset the non-linear solver
  nln_solver().reset();

  // reset x vector of the group
  auto& group = nln_solver().get_solution_group();
  group.computeX(group, group.getX(), -1.0);

  // solve the non-linear problem
  const auto status = nln_solver().solve();
  FOUR_C_ASSERT_ALWAYS(
      status == Solid::StepStatus::no_errors, "Explicit structural nonlinear solve failed.");
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> Solid::TimeInt::Explicit::initial_guess()
{
  FOUR_C_THROW("initial_guess() is not available for explicit time integration");
  return nullptr;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>> Solid::TimeInt::Explicit::get_f() const
{
  FOUR_C_THROW("RHS() is not available for explicit time integration");
  return nullptr;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> Solid::TimeInt::Explicit::freact()
{
  check_init_setup();
  FOUR_C_THROW("Not implemented!");
  return nullptr;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::SparseMatrix> Solid::TimeInt::Explicit::system_matrix()
{
  FOUR_C_THROW("system_matrix() is not available for explicit time integration");
  return nullptr;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> Solid::TimeInt::Explicit::block_system_matrix()
{
  FOUR_C_THROW("block_system_matrix() is not available for explicit time integration");
  return nullptr;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::TimeInt::Explicit::use_block_matrix(
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> domainmaps,
    std::shared_ptr<const Core::LinAlg::MultiMapExtractor> rangemaps)
{
  FOUR_C_THROW("use_block_matrix() is not available for explicit time integration");
}
///@}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::DynamicType Solid::TimeInt::Explicit::method_name() const
{
  return explint_ptr_->method_name();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::TimeInt::Explicit::method_steps() const { return explint_ptr_->method_steps(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::TimeInt::Explicit::method_order_of_accuracy_dis() const
{
  return explint_ptr_->method_order_of_accuracy_dis();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::TimeInt::Explicit::method_order_of_accuracy_vel() const
{
  return explint_ptr_->method_order_of_accuracy_vel();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::TimeInt::Explicit::method_lin_err_coeff_dis() const
{
  return explint_ptr_->method_lin_err_coeff_dis();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::TimeInt::Explicit::method_lin_err_coeff_vel() const
{
  return explint_ptr_->method_lin_err_coeff_vel();
}

FOUR_C_NAMESPACE_CLOSE
