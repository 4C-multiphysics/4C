// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_predict_tangdis.hpp"

#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_group_prepostoperator.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_impl_generic.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <NOX_Epetra_Vector.H>
#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::Predict::TangDis::TangDis() : dbc_incr_ptr_(nullptr), apply_linear_reaction_forces_(false)
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Predict::TangDis::setup()
{
  check_init();
  // ---------------------------------------------------------------------------
  // set the new pre/post operator for the nox nln group in the parameter list
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList& p_grp_opt = nox_params().sublist("Group Options");
  // Get the current map. If there is no map, return a new empty one. (reference)
  NOX::Nln::GROUP::PrePostOperator::Map& prepostgroup_map =
      NOX::Nln::GROUP::PrePostOp::get_map(p_grp_opt);
  // create the new tangdis pre/post operator
  Teuchos::RCP<NOX::Nln::Abstract::PrePostOperator> preposttangdis_ptr =
      Teuchos::make_rcp<NOX::Nln::GROUP::PrePostOp::TangDis>(Teuchos::rcpFromRef(*this));
  // insert/replace the old pointer in the map
  prepostgroup_map[NOX::Nln::GROUP::prepost_tangdis] = preposttangdis_ptr;

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Predict::TangDis::compute(::NOX::Abstract::Group& grp)
{
  check_init_setup();
  NOX::Nln::Group* grp_ptr = dynamic_cast<NOX::Nln::Group*>(&grp);
  FOUR_C_ASSERT(grp_ptr != nullptr, "Dynamic cast failed!");
  grp_ptr->reset_pre_post_operator(nox_params().sublist("Group Options"));

  impl_int().eval_data().set_predictor_type(Inpar::Solid::pred_tangdis);

  // ---------------------------------------------------------------------------
  // calculate the dbc increment on the dirichlet boundary
  // ---------------------------------------------------------------------------
  dbc_incr_ptr_ = dbc().get_dirichlet_increment();

  // ---------------------------------------------------------------------------
  // We create at this point a new solution vector and initialize it
  // with the values of the last converged time step.
  // ---------------------------------------------------------------------------
  std::shared_ptr<::NOX::Epetra::Vector> x_ptr = global_state().create_global_vector(
      TimeInt::BaseDataGlobalState::VecInitType::last_time_step, impl_int().model_eval_ptr());
  // Set the solution vector in the nox group. This will reset all isValid
  // flags.
  grp.setX(*x_ptr);

  // ---------------------------------------------------------------------------
  // Compute F and jacobian and apply the linear reaction forces due to changing
  // Dirichlet boundary conditions.
  // ---------------------------------------------------------------------------
  apply_linear_reaction_forces_ = true;
  grp_ptr->compute_f_and_jacobian();
  apply_linear_reaction_forces_ = false;

  // ---------------------------------------------------------------------------
  // Check if we are using a Newton direction
  // ---------------------------------------------------------------------------
  std::string dir_str = nox_params().sublist("Direction").get<std::string>("Method");
  if (dir_str == "User Defined")
    dir_str = nox_params().sublist("Direction").get<std::string>("User Defined Method");
  if (dir_str != "Newton" and dir_str != "Modified Newton")
    FOUR_C_THROW(
        "The TangDis predictor is currently only working for the direction-"
        "methods \"Newton\" and \"Modified Newton\".");

  // ---------------------------------------------------------------------------
  // (re)set the linear solver parameters
  // ---------------------------------------------------------------------------
  Teuchos::ParameterList& p =
      nox_params().sublist("Direction").sublist("Newton").sublist("Linear Solver");
  p.set<int>("Number of Nonlinear Iterations", 0);
  p.set<int>("Current Time Step", global_state().get_step_np());
  // ToDo Get the actual tolerance value
  p.set<double>("Wanted Tolerance", 1.0e-6);

  // ---------------------------------------------------------------------------
  // solve the linear system of equations and update the current state
  // ---------------------------------------------------------------------------
  // compute the Newton direction
  grp_ptr->computeNewton(p);
  // reset isValid flags
  grp_ptr->computeX(*grp_ptr, grp_ptr->getNewton(), 1.0);
  // add the DBC values to the current state vector
  std::shared_ptr<Core::LinAlg::Vector<double>> dbc_incr_exp_ptr =
      std::make_shared<Core::LinAlg::Vector<double>>(global_state().global_problem_map(), true);
  Core::LinAlg::export_to(*dbc_incr_ptr_, *dbc_incr_exp_ptr);
  grp_ptr->computeX(*grp_ptr, dbc_incr_exp_ptr->get_ref_of_epetra_vector(), 1.0);
  // Reset the state variables
  const ::NOX::Epetra::Vector& x_eptra =
      dynamic_cast<const ::NOX::Epetra::Vector&>(grp_ptr->getX());
  // set the consistent state in the models (e.g. structure and contact models)
  impl_int().reset_model_states(Core::LinAlg::Vector<double>(x_eptra.getEpetraVector()));

  // For safety purposes, we set the dbc_incr vector to zero
  dbc_incr_ptr_->put_scalar(0.0);

  impl_int().model_eval().predict(get_type());

  impl_int().eval_data().set_predictor_type(Inpar::Solid::pred_vague);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Core::LinAlg::Vector<double>& Solid::Predict::TangDis::get_dbc_incr() const
{
  FOUR_C_ASSERT(dbc_incr_ptr_, "The dbc increment is not initialized!");
  return *dbc_incr_ptr_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const bool& Solid::Predict::TangDis::is_apply_linear_reaction_forces() const
{
  return apply_linear_reaction_forces_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Solid::Predict::TangDis::pre_apply_force_external(Core::LinAlg::Vector<double>& fextnp) const
{
  check_init_setup();

  if (get_type() != Inpar::Solid::pred_tangdis_constfext) return false;

  if (apply_linear_reaction_forces_)
  {
    fextnp.scale(1.0, *global_state().get_fext_n());
    return true;
  }
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
NOX::Nln::GROUP::PrePostOp::TangDis::TangDis(
    const Teuchos::RCP<const Solid::Predict::TangDis>& tang_predict_ptr)
    : tang_predict_ptr_(tang_predict_ptr)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::GROUP::PrePostOp::TangDis::run_post_compute_f(
    Core::LinAlg::Vector<double>& F, const NOX::Nln::Group& grp)
{
  // If we do not want to apply linear reaction forces due to changing Dirichlet
  // boundary conditions, we just return.
  if (not tang_predict_ptr_->is_apply_linear_reaction_forces()) return;

  // get the new dirichlet boundary increment
  const Core::LinAlg::Vector<double>& dbc_incr = tang_predict_ptr_->get_dbc_incr();

  double dbc_incr_nrm2 = 0.0;
  dbc_incr.norm_2(&dbc_incr_nrm2);

  // If there are only Neumann loads, do a direct return.
  if (dbc_incr_nrm2 == 0.0) return;

  /* Alternatively, it's also possible to get a const pointer on the jacobian
   * by calling grp.get_linear_system()->get_jacobian_operator()... */
  std::shared_ptr<const Core::LinAlg::SparseMatrix> stiff_ptr =
      tang_predict_ptr_->global_state().get_jacobian_displ_block();

  // check if the jacobian is filled
  if (not stiff_ptr->filled()) FOUR_C_THROW("The jacobian is not yet filled!");

  Core::LinAlg::Vector<double> freact_ptr(*tang_predict_ptr_->global_state().dof_row_map_view());
  if (stiff_ptr->multiply(false, dbc_incr, freact_ptr)) FOUR_C_THROW("Multiply failed!");

  // finally add the linear reaction forces to the current rhs
  Core::LinAlg::assemble_my_vector(1.0, F, 1.0, freact_ptr);
}

FOUR_C_NAMESPACE_CLOSE
