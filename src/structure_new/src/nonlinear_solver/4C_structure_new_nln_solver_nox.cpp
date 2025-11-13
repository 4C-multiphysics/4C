// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_nln_solver_nox.hpp"  // class header

#include "4C_linear_solver_method_linalg.hpp"
#include "4C_solver_nonlin_nox_constraint_interface_required.hpp"
#include "4C_solver_nonlin_nox_globaldata.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_linearsystem.hpp"
#include "4C_solver_nonlin_nox_problem.hpp"
#include "4C_solver_nonlin_nox_scaling.hpp"
#include "4C_solver_nonlin_nox_solver_factory.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_structure_new_timint_noxinterface.hpp"
#include "4C_structure_new_utils.hpp"

#include <NOX_Abstract_Group.H>
#include <NOX_Solver_Generic.H>
#include <Teuchos_RCPStdSharedPtrConversions.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::Nln::SOLVER::Nox::Nox(const Teuchos::ParameterList& default_params,
    const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate,
    const std::shared_ptr<Solid::TimeInt::BaseDataSDyn>& sdyn,
    const std::shared_ptr<Solid::TimeInt::NoxInterface>& noxinterface,
    const std::shared_ptr<Solid::Integrator>& integr,
    const std::shared_ptr<const Solid::TimeInt::Base>& timint)
    : Generic(gstate, sdyn, noxinterface, integr, timint), default_params_(default_params)
{
  /* Set NOX::Nln::Interface::RequiredBase
   * This interface is necessary for the evaluation of basic things
   * which are evaluated outside of the non-linear solver, but
   * are always necessary. A simple example is the right-hand-side
   * F. (see computeF) */
  const auto ireq = nox_interface_ptr();

  /* Set ::NOX::Epetra::Interface::Jacobian
   * This interface is necessary for the evaluation of the jacobian
   * and everything, which is directly related to the jacobian.
   * This interface is optional. You can think of Finite-Differences
   * as one way to circumvent the evaluation of the jacobian.
   * Nevertheless, we always set this interface ptr in the structural
   * case. */
  const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian> ijac =
      Teuchos::rcpFromRef(*nox_interface_ptr());

  // vector of currently present solution types
  std::vector<NOX::Nln::SolutionType> soltypes;
  // map of linear solvers, the key is the solution type
  NOX::Nln::LinearSystem::SolverMap linsolvers;
  /* convert the Inpar::Solid::ModelType to a NOX::Nln::SolType
   * and fill the linear solver map. */
  Solid::Nln::convert_model_type2_sol_type(
      soltypes, linsolvers, data_sdyn().get_model_types(), data_sdyn().get_lin_solvers());

  // define and initialize the optimization type
  const NOX::Nln::OptimizationProblemType opttype = Solid::Nln::optimization_type(soltypes);

  // map of constraint interfaces, the key is the solution type
  NOX::Nln::CONSTRAINT::ReqInterfaceMap iconstr;
  // set constraint interfaces
  Solid::Nln::create_constraint_interfaces(iconstr, integrator(), soltypes);

  // preconditioner map for constraint problems
  NOX::Nln::CONSTRAINT::PrecInterfaceMap iconstr_prec;
  Solid::Nln::create_constraint_preconditioner(iconstr_prec, integrator(), soltypes);

  // create object to scale linear system
  std::shared_ptr<NOX::Nln::Scaling> iscale = nullptr;
  Solid::Nln::create_scaling(iscale, data_sdyn(), data_global_state());

  // handle NOX settings
  auto& nox_params = data_sdyn().get_nox_params();

  // set parameters
  nox_params.setParameters(default_params_);

  // build the global data container for the nox_nln_solver
  nlnglobaldata_ = Teuchos::make_rcp<NOX::Nln::GlobalData>(data_global_state().get_comm(),
      nox_params, linsolvers, ireq, ijac, opttype, iconstr, iconstr_prec, iscale);

  // -------------------------------------------------------------------------
  // Create NOX control class: NoxProblem()
  // -------------------------------------------------------------------------
  Teuchos::RCP<NOX::Nln::Vector> soln = Teuchos::rcp(data_global_state().create_global_vector());
  Teuchos::RCP<Core::LinAlg::SparseOperator> jac =
      Teuchos::rcp(data_global_state().create_jacobian());
  problem_ = Teuchos::make_rcp<NOX::Nln::Problem>(nlnglobaldata_, soln, jac);

  // -------------------------------------------------------------------------
  // Create NOX linear system to provide access to Jacobian etc.
  // -------------------------------------------------------------------------
  linsys_ = problem_->create_linear_system();

  // -------------------------------------------------------------------------
  // Create NOX group
  // -------------------------------------------------------------------------
  /* use NOX::Nln::Group to enable access to time integration
   * use NOX::Nln::Constraint::Group to enable access to the constraint data*/
  group_ptr() = problem_->create_group(linsys_);

  // -------------------------------------------------------------------------
  // Create NOX status test
  // -------------------------------------------------------------------------
  // get the stopping criteria from the nox parameter list
  problem_->create_status_tests(ostatus_, istatus_);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Nln::SOLVER::Nox::reset()
{
  // do a hard reset at the beginning to be on the safe side
  nlnsolver_ = Teuchos::null;

  // -------------------------------------------------------------------------
  // reset the parameter list
  // -------------------------------------------------------------------------
  reset_params();

  // -------------------------------------------------------------------------
  // Create NOX non-linear solver
  // -------------------------------------------------------------------------
  nlnsolver_ = NOX::Nln::Solver::build_solver(
      Teuchos::rcpFromRef(*group_ptr()), ostatus_, istatus_, *nlnglobaldata_);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Nln::SOLVER::Nox::reset_params()
{
  if (default_params_.isSublist("NOX"))
  {
    nlnglobaldata_->get_nln_parameter_list().setParametersNotAlreadySet(default_params_);
    return;
  }

  const Teuchos::ParameterList& pdir =
      nlnglobaldata_->get_nln_parameter_list().sublist("Direction", true);
  std::string method = pdir.get<std::string>("Method");

  if (method == "User Defined") method = pdir.get<std::string>("User Defined Method");

  if (method == "Newton" or method == "Modified Newton")
  {
    // get the linear solver sub-sub-sub-list
    Teuchos::ParameterList& lsparams = nlnglobaldata_->get_nln_parameter_list()
                                           .sublist("Direction", true)
                                           .sublist("Newton", true)
                                           .sublist("Linear Solver", true);

    // get current time step and update the parameter list entry
    lsparams.set<int>("Current Time Step", data_global_state().get_step_np());
  }
  else if (method == "Single Step")
  {
    // get the linear solver sub-sub-list
    Teuchos::ParameterList& lsparams = nlnglobaldata_->get_nln_parameter_list()
                                           .sublist("Single Step Solver", true)
                                           .sublist("Linear Solver", true);

    // get current time step and update the parameter list entry
    lsparams.set<int>("Current Time Step", data_global_state().get_step_np());
  }
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Inpar::Solid::ConvergenceStatus Solid::Nln::SOLVER::Nox::solve()
{
#if !(FOUR_C_TRILINOS_INTERNAL_VERSION_GE(2025, 4))
  const auto solver_type =
      nlnglobaldata_->get_nln_parameter_list().get<std::string>("Nonlinear Solver");

  if (solver_type == "Single Step")
  {
    auto& nln_group = dynamic_cast<NOX::Nln::Group&>(*group_ptr());

    nln_group.set_is_valid_newton(true);  // to circumvent the check in ::NOX::Solver::SingleStep
    nln_group.set_is_valid_rhs(false);    // force to compute the RHS
    nln_group.reset_x();                  // to initialize the solution vector to zero
  }
#endif

  // solve the non-linear step
  ::NOX::StatusTest::StatusType finalstatus = nlnsolver_->solve();

  // Check if we do something special if the non-linear solver fails,
  // otherwise an error is thrown.
  if (data_sdyn().get_divergence_action() == Inpar::Solid::divcont_stop)
    problem_->check_final_status(finalstatus);

  return convert_final_status(finalstatus);
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Inpar::Solid::ConvergenceStatus Solid::Nln::SOLVER::Nox::convert_final_status(
    const ::NOX::StatusTest::StatusType& finalstatus) const
{
  Inpar::Solid::ConvergenceStatus convstatus = Inpar::Solid::conv_success;

  switch (finalstatus)
  {
    case ::NOX::StatusTest::Unevaluated:
      convstatus = Inpar::Solid::conv_ele_fail;
      break;
    case ::NOX::StatusTest::Unconverged:
    case ::NOX::StatusTest::Failed:
      convstatus = Inpar::Solid::conv_nonlin_fail;
      break;
    case ::NOX::StatusTest::Converged:
      convstatus = Inpar::Solid::conv_success;
      break;
    default:
      FOUR_C_THROW(
          "Conversion of the ::NOX::StatusTest::StatusType to "
          "a Inpar::Solid::ConvergenceStatus is not possible!");
      break;
  }

  return convstatus;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::Nln::SOLVER::Nox::get_num_nln_iterations() const
{
  if (nlnsolver_) return nlnsolver_->getNumIterations();
  return 0;
}

FOUR_C_NAMESPACE_CLOSE
