// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_expl_generic.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_group_prepostoperator.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"
#include "4C_structure_new_timint_basedatasdyn.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::Generic::setup()
{
  check_init();

  // call base class first
  Solid::Integrator::setup();

  // ---------------------------------------------------------------------------
  // set the new pre/post operator for the nox nln group in the parameter list
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList& p_grp_opt = sdyn().get_nox_params().sublist("Group Options");

  // create the new generic pre/post operator
  Teuchos::RCP<NOX::Nln::Abstract::PrePostOperator> prepost_generic_ptr =
      Teuchos::make_rcp<NOX::Nln::PrePostOp::EXPLICIT::Generic>(*this);

  // Get the current map. If there is no map, return a new empty one. (reference)
  NOX::Nln::GROUP::PrePostOperator::Map& prepostgroup_map =
      NOX::Nln::GROUP::PrePostOp::get_map(p_grp_opt);

  // insert/replace the old pointer in the map
  prepostgroup_map[NOX::Nln::GROUP::prepost_impl_generic] = prepost_generic_ptr;

  // ---------------------------------------------------------------------------
  // set the new pre/post operator for the nox nln solver in the parameter list
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList& p_sol_opt = sdyn().get_nox_params().sublist("Solver Options");

  NOX::Nln::Aux::add_to_pre_post_op_vector(p_sol_opt, prepost_generic_ptr);

  // No issetup_ = true, since the setup() functions of the derived classes
  // have to be called and finished first!
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::Generic::apply_force(
    const Core::LinAlg::Vector<double>& x, Core::LinAlg::Vector<double>& f)
{
  check_init_setup();

  // ---------------------------------------------------------------------------
  // evaluate the different model types (static case) at t_{n+1}^{i}
  // ---------------------------------------------------------------------------
  // set the time step dependent parameters for the element evaluation
  reset_eval_params();
  model_eval().apply_force(x, f, 1.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::Generic::apply_stiff(
    const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac)
{
  check_init_setup();

  // ---------------------------------------------------------------------------
  // evaluate the different model types (static case) at t_{n+1}^{i}
  // ---------------------------------------------------------------------------
  // set the time step dependent parameters for the element evaluation
  reset_eval_params();
  model_eval().apply_stiff(x, jac, 1.0);
  jac.complete();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::Generic::apply_force_stiff(const Core::LinAlg::Vector<double>& x,
    Core::LinAlg::Vector<double>& f, Core::LinAlg::SparseOperator& jac)
{
  check_init_setup();
  // ---------------------------------------------------------------------------
  // evaluate the different model types (static case) at t_{n+1}^{i}
  // ---------------------------------------------------------------------------
  // set the time step dependent parameters for the element evaluation
  reset_eval_params();
  model_eval().apply_force_stiff(x, f, jac, 1.0);
  jac.complete();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::EXPLICIT::Generic::calc_ref_norm_force(
    const enum ::NOX::Abstract::Vector::NormType& type) const
{
  FOUR_C_THROW("{} is not yet implemented", __FUNCTION__);
  return 0.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::Generic::compute_jacobian_contributions_from_element_level_for_ptc(
    std::shared_ptr<Core::LinAlg::SparseMatrix>& scalingMatrixOpPtr)
{
  FOUR_C_THROW("{} is not yet implemented", __FUNCTION__);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::Generic::assemble_force(Core::LinAlg::Vector<double>& f,
    const std::vector<Solid::ModelType>* without_these_models) const
{
  FOUR_C_THROW("{} is not yet implemented", __FUNCTION__);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::Generic::remove_condensed_contributions_from_rhs(
    Core::LinAlg::Vector<double>& rhs) const
{
  model_eval().remove_condensed_contributions_from_rhs(rhs);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::Generic::update_step_element()
{
  check_init_setup();
  model_eval().update_step_element();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::Generic::update_constant_state_contributions() {}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::Generic::post_update() { update_constant_state_contributions(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::EXPLICIT::Generic::get_default_step_length() const
{
  // default: return a step length of 1.0
  return 1.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::EXPLICIT::Generic::reset_eval_params()
{
  // set the time step dependent parameters for the element evaluation
  eval_data().set_total_time(global_state().get_time_np());
  eval_data().set_delta_time(global_state().get_delta_time()[0]);
  eval_data().set_function_manager(Global::Problem::instance()->function_manager());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::PrePostOp::EXPLICIT::Generic::runPreIterate(const ::NOX::Solver::Generic& nlnSolver)
{
  // For explicit integration this action simply does nothing.
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::PrePostOp::EXPLICIT::Generic::runPreSolve(const ::NOX::Solver::Generic& nlnSolver)
{
  // For explicit integration this action simply does nothing.
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::PrePostOp::EXPLICIT::Generic::run_pre_compute_x(const NOX::Nln::Group& input_grp,
    const Core::LinAlg::Vector<double>& dir, const double& step, const NOX::Nln::Group& curr_grp)
{
  // For explicit integration this action simply does nothing.
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::PrePostOp::EXPLICIT::Generic::run_post_compute_x(const NOX::Nln::Group& input_grp,
    const Core::LinAlg::Vector<double>& dir, const double& step, const NOX::Nln::Group& curr_grp)
{
  // For explicit integration this action simply does nothing.
}

FOUR_C_NAMESPACE_CLOSE
