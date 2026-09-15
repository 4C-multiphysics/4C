// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_str_model_evaluator_partitioned.hpp"

#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_impl_generic.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"
#include "4C_structure_new_nln_solver_generic.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"
#include "4C_structure_new_timint_implicit.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Solid::ModelEvaluator::PartitionedFSI::PartitionedFSI()
    : interface_force_np_ptr_(nullptr), is_relaxationsolve_(false)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::PartitionedFSI::setup()
{
  // fsi interface force at t_{n+1}
  interface_force_np_ptr_ =
      std::make_shared<Core::LinAlg::Vector<double>>(*global_state().dof_row_map(), true);

  // set flag
  issetup_ = true;

  return;
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::PartitionedFSI::setup_multi_map_extractor()
{
  integrator().model_eval().setup_multi_map_extractor();
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map>
Solid::ModelEvaluator::PartitionedFSI::get_block_dof_row_map_ptr() const
{
  check_init_setup();
  return global_state().dof_row_map();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::PartitionedFSI::get_current_solution_ptr() const
{
  check_init();
  return global_state().get_dis_np();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::PartitionedFSI::get_last_time_step_solution_ptr() const
{
  check_init();
  return global_state().get_dis_n();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Solid::ModelEvaluator::PartitionedFSI::assemble_force(
    Core::LinAlg::Vector<double>& f, const double& timefac_np) const
{
  Core::LinAlg::assemble_my_vector(1.0, f, -timefac_np, *interface_force_np_ptr_);
  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::PartitionedFSI::update_step_state(const double& timefac_n)
{
  if (not is_relaxationsolve_)  // standard case
  {
    // add the old time factor scaled contributions to the residual
    std::shared_ptr<Core::LinAlg::Vector<double>>& fstructold_ptr =
        global_state().get_fstructure_old();
    fstructold_ptr->update(-timefac_n, *interface_force_np_ptr_, 1.0);
  }
  else
  {
    // do nothing
  }
  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Vector<double>>
Solid::ModelEvaluator::PartitionedFSI::solve_relaxation_linear(
    std::shared_ptr<Adapter::Structure> structure)
{
  // print to screen
  if (Core::Communication::my_mpi_rank(global_state().dof_row_map()->get_comm()) == 0)
    std::cout << "\n DO SRUCTURAL RELAXATION SOLVE ..." << std::endl;

  // cast adapter structure to implicit time integrator
  std::shared_ptr<Solid::TimeInt::Implicit> ti_impl =
      std::dynamic_pointer_cast<Solid::TimeInt::Implicit>(structure);

  // get the nonlinear solver pointer
  Solid::Nln::SOLVER::Generic& nlnsolver =
      const_cast<Solid::Nln::SOLVER::Generic&>(*(ti_impl->get_nln_solver_ptr()));

  // get the solution group
  ::NOX::Abstract::Group& grp = nlnsolver.get_solution_group();
  NOX::Nln::Group* grp_ptr = dynamic_cast<NOX::Nln::Group*>(&grp);
  if (grp_ptr == nullptr) FOUR_C_THROW("Dynamic cast failed!");

  // get nox parameter
  Teuchos::ParameterList& noxparams = ti_impl->data_sdyn().get_nox_params();

  // create new state vector
  std::shared_ptr<NOX::Nln::Vector> x_ptr =
      global_state().create_global_vector(TimeInt::BaseDataGlobalState::VecInitType::last_time_step,
          ti_impl->impl_int_ptr()->model_eval_ptr());
  // Set the solution vector in the nox group. This will reset all isValid
  // flags.
  grp.setX(*x_ptr);

  // ---------------------------------------------------------------------------
  // Compute F and jacobian
  // ---------------------------------------------------------------------------
  const auto jacobian_status = grp_ptr->computeJacobian();
  FOUR_C_ASSERT_ALWAYS(jacobian_status == ::NOX::Abstract::Group::ReturnType::Ok,
      "Failed to compute Jacobian for partitioned FSI structure model evaluator.");

  // overwrite F with boundary force
  interface_force_np_ptr_->scale(-(ti_impl->tim_int_param()));
  ti_impl->dbc_ptr()->apply_dirichlet_to_rhs(*interface_force_np_ptr_);
  Teuchos::RCP<NOX::Nln::Vector> nox_force =
      Teuchos::make_rcp<NOX::Nln::Vector>(interface_force_np_ptr_);
  const auto residual_status = grp_ptr->set_f(nox_force);
  FOUR_C_ASSERT_ALWAYS(residual_status == ::NOX::Abstract::Group::ReturnType::Ok,
      "Failed to set residual for partitioned FSI structure model evaluator.");

  // ---------------------------------------------------------------------------
  // Check if we are using a Newton direction
  // ---------------------------------------------------------------------------
  const std::string dir_str(NOX::Nln::Aux::get_direction_method_list_name(noxparams));
  if (dir_str != "Newton")
    FOUR_C_THROW(
        "The RelaxationSolve is currently only working for the direction-"
        "method \"Newton\".");

  // ---------------------------------------------------------------------------
  // (re)set the linear solver parameters
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList& p =
      noxparams.sublist("Direction").sublist("Newton").sublist("Linear Solver");
  p.set<int>("Number of Nonlinear Iterations", 0);
  p.set<int>("Current Time Step", global_state().get_step_np());
  p.set<double>("Wanted Tolerance", 1.0e-6);  //!< dummy

  // ---------------------------------------------------------------------------
  // solve the linear system of equations and update the current state
  // ---------------------------------------------------------------------------
  // compute the Newton direction
  const auto newton_status = grp_ptr->computeNewton(p);
  FOUR_C_ASSERT_ALWAYS(newton_status == ::NOX::Abstract::Group::ReturnType::Ok,
      "Failed to compute Newton direction for partitioned FSI structure model evaluator.");

  // get the increment from the previous solution step
  const auto& increment = dynamic_cast<const NOX::Nln::Vector&>(grp_ptr->getNewton());

  // return the increment
  return std::make_shared<Core::LinAlg::Vector<double>>(increment.get_linalg_vector());
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::TimeInt::BaseDataIO& Solid::ModelEvaluator::PartitionedFSI::get_in_output() const
{
  check_init();
  return global_in_output();
}

FOUR_C_NAMESPACE_CLOSE
