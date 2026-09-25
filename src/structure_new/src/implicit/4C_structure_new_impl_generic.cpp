// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_impl_generic.hpp"

#include "4C_global_data.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_group_prepostoperator.hpp"
#include "4C_solver_nonlin_nox_solver_linesearchbased.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"
#include "4C_structure_new_timint_implicit.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::IMPLICIT::Generic::Generic() : ispredictor_state_(false)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Generic::setup()
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
      Teuchos::make_rcp<NOX::Nln::PrePostOp::IMPLICIT::Generic>(*this);

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
void Solid::IMPLICIT::Generic::set_is_predictor_state(const bool ispredictor_state)
{
  ispredictor_state_ = ispredictor_state;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::IMPLICIT::Generic::is_predictor_state() const { return ispredictor_state_; }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::ParameterList& Solid::IMPLICIT::Generic::get_nox_params()
{
  return sdyn().get_nox_params();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Solid::TimeInt::BaseDataSDyn& Solid::IMPLICIT::Generic::get_data_sdyn() const
{
  return tim_int().get_data_sdyn();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::IMPLICIT::Generic::get_default_step_length() const
{
  const Teuchos::ParameterList& p_nox = tim_int().get_data_sdyn().get_nox_params();
  const std::string nln_solver = p_nox.get<std::string>("Nonlinear Solver");
  // The pseudo transient implementation holds also a line search object!
  if (nln_solver == "Line Search Based" or nln_solver == "Pseudo Transient")
  {
    const Teuchos::ParameterList& p_ls = p_nox.sublist("Line Search");
    const std::string method = p_ls.get<std::string>("Method");
    const Teuchos::ParameterList& p_method = p_ls.sublist(method);
    if (p_method.isParameter("Default Step")) return p_method.get<double>("Default Step");
  }
  // default: return a step length of 1.0
  return 1.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Generic::reset_eval_params()
{
  // set the time step dependent parameters for the element evaluation
  eval_data().set_total_time(global_state().get_time_np());
  eval_data().set_delta_time(global_state().get_delta_time()[0]);
  eval_data().set_function_manager(Global::Problem::instance()->function_manager());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Generic::print_jacobian_in_matlab_format(
    const NOX::Nln::Group& curr_grp) const
{
  const Solid::TimeInt::Implicit& timint_impl =
      dynamic_cast<const Solid::TimeInt::Implicit&>(tim_int());

  timint_impl.print_jacobian_in_matlab_format(curr_grp);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Generic::apply_correction_system(const NOX::Nln::CorrectionType type,
    const std::vector<Solid::ModelType>& constraint_models, const Core::LinAlg::Vector<double>& x,
    Core::LinAlg::Vector<double>& f, Core::LinAlg::SparseOperator& jac)
{
  check_init_setup();

  reset_eval_params();

  switch (type)
  {
    case NOX::Nln::CorrectionType::soc_full:
    {
      // Do a standard full step.
      /* Note that there is a difference, since we tagged this evaluation by
       * setting it to a non-default step. */
      apply_force_stiff(x, f, jac);
      break;
    }
    case NOX::Nln::CorrectionType::soc_cheap:
    {
      model_eval().apply_cheap_soc_rhs(type, constraint_models, x, f, 1.0);
      if (not jac.filled()) FOUR_C_THROW("The jacobian is supposed to be filled at this point!");

      break;
    }
    default:
    {
      FOUR_C_THROW(
          "No action defined for the given second order correction type: "
          "\"{}\"",
          NOX::Nln::correction_type_to_string(type).c_str());
    }
  }

  if (not jac.filled()) jac.complete();
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::PrePostOp::IMPLICIT::Generic::run_pre_compute_x(const NOX::Nln::Group& input_grp,
    const Core::LinAlg::Vector<double>& dir, const double& step, const NOX::Nln::Group& curr_grp)
{
  // set the evaluation parameters
  const auto& xold = dynamic_cast<const NOX::Nln::Vector&>(input_grp.getX()).get_linalg_vector();
  auto& dir_mutable = const_cast<Core::LinAlg::Vector<double>&>(dir);

  const bool isdefaultstep = (step == default_step_);
  impl_.model_eval().run_pre_compute_x(xold, dir_mutable, step, curr_grp, isdefaultstep);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::PrePostOp::IMPLICIT::Generic::run_post_compute_x(const NOX::Nln::Group& input_grp,
    const Core::LinAlg::Vector<double>& dir, const double& step, const NOX::Nln::Group& curr_grp)
{
  // set the evaluation parameters
  const auto& xold = dynamic_cast<const NOX::Nln::Vector&>(input_grp.getX()).get_linalg_vector();
  const auto& xnew = dynamic_cast<const NOX::Nln::Vector&>(curr_grp.getX()).get_linalg_vector();

  bool isdefaultstep = (step == default_step_);
  impl_.model_eval().run_post_compute_x(xold, dir, step, xnew, isdefaultstep);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::PrePostOp::IMPLICIT::Generic::runPostIterate(const ::NOX::Solver::Generic& solver)
{
  double step = 0.0;
  const bool isdefaultstep = get_step(step, solver);

  impl_.model_eval().run_post_iterate(solver, step, isdefaultstep);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::PrePostOp::IMPLICIT::Generic::runPreSolve(const ::NOX::Solver::Generic& solver)
{
  double step = 0.0;
  const bool isdefaultstep = get_step(step, solver);

  impl_.model_eval().run_pre_solve(solver, step, isdefaultstep);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::PrePostOp::IMPLICIT::Generic::run_pre_apply_jacobian_inverse(
    const ::NOX::Abstract::Vector& rhs, ::NOX::Abstract::Vector& result,
    const ::NOX::Abstract::Vector& xold, const NOX::Nln::Group& grp)
{
  const auto& nox_rhs = dynamic_cast<const NOX::Nln::Vector&>(rhs);
  auto& nox_result = dynamic_cast<NOX::Nln::Vector&>(result);
  const auto& nox_xold = dynamic_cast<const NOX::Nln::Vector&>(xold);

  impl_.model_eval().run_pre_apply_jacobian_inverse(nox_rhs.get_linalg_vector(),
      nox_result.get_linalg_vector(), nox_xold.get_linalg_vector(), grp);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::PrePostOp::IMPLICIT::Generic::run_post_apply_jacobian_inverse(
    const ::NOX::Abstract::Vector& rhs, ::NOX::Abstract::Vector& result,
    const ::NOX::Abstract::Vector& xold, const NOX::Nln::Group& grp)
{
  const auto& nox_rhs = dynamic_cast<const NOX::Nln::Vector&>(rhs);
  auto& nox_result = dynamic_cast<NOX::Nln::Vector&>(result);
  const auto& nox_xold = dynamic_cast<const NOX::Nln::Vector&>(xold);

  impl_.model_eval().run_post_apply_jacobian_inverse(nox_rhs.get_linalg_vector(),
      nox_result.get_linalg_vector(), nox_xold.get_linalg_vector(), grp);

  impl_.print_jacobian_in_matlab_format(grp);
}



/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool NOX::Nln::PrePostOp::IMPLICIT::Generic::get_step(
    double& step, const ::NOX::Solver::Generic& solver) const
{
  // try to cast the given solver object
  const NOX::Nln::Solver::LineSearchBased* ls_solver =
      dynamic_cast<const NOX::Nln::Solver::LineSearchBased*>(&solver);

  bool isdefaultstep = false;

  if (not ls_solver)
  {
    step = default_step_;
    isdefaultstep = true;
  }
  else
  {
    step = ls_solver->getStepSize();
    isdefaultstep = (step == default_step_);
  }

  return isdefaultstep;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Generic::compute_jacobian_contributions_from_element_level_for_ptc(
    std::shared_ptr<Core::LinAlg::SparseMatrix>& scalingMatrixOpPtr)
{
  model_eval().compute_jacobian_contributions_from_element_level_for_ptc(scalingMatrixOpPtr);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::IMPLICIT::Generic::remove_condensed_contributions_from_rhs(
    Core::LinAlg::Vector<double>& rhs) const
{
  model_eval().remove_condensed_contributions_from_rhs(rhs);
}

FOUR_C_NAMESPACE_CLOSE
