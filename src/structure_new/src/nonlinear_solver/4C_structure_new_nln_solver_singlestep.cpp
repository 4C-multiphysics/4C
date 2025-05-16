// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_nln_solver_singlestep.hpp"

#include "4C_solver_nonlin_nox_globaldata.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_structure_new_integrator.hpp"
#include "4C_structure_new_nln_solver_utils.hpp"
#include "4C_structure_new_timint_base.hpp"

#include <NOX_Solver_Generic.H>
#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Nln::SOLVER::SingleStep::setup()
{
  check_init();

  // setup the nox parameter list for a full Newton solution method
  set_single_step_params();

  // Call the setup() function of the base class
  // Note, that the issetup_ flag is also updated during this call.
  Nox::setup();

  FOUR_C_ASSERT(is_setup(), "issetup_ should be \"true\" at this point!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Nln::SOLVER::SingleStep::set_single_step_params()
{
  check_init();

  // get the nox parameter list and set the necessary parameters for a
  // full Newton solution procedure
  Teuchos::ParameterList& p = data_sdyn().get_nox_params();
  set_single_step_params(p);

  // ---------------------------------------------------------------------------
  // STATUS TEST
  // ---------------------------------------------------------------------------
  /* This is only necessary for the special case, that you use no xml-file for
   * the definition of your convergence tests, but you use the input file instead.
   */
  if (not is_xml_status_test_file(data_sdyn().get_nox_params().sublist("Status Test")))
  {
    std::set<enum NOX::Nln::StatusTest::QuantityType> qtypes;
    create_quantity_types(qtypes, data_sdyn());
    set_status_test_params(
        data_sdyn().get_nox_params().sublist("Status Test"), data_sdyn(), qtypes);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Nln::SOLVER::SingleStep::set_single_step_params(Teuchos::ParameterList& p)
{
  // ---------------------------------------------------------------------------
  // Set-up the single step method
  // ---------------------------------------------------------------------------
  // Non-linear solver
  p.set("Nonlinear Solver", "Single Step");

  // Set the printing parameters in the "Printing" sublist
  Teuchos::ParameterList& printParams = p.sublist("Printing");
  printParams.set("Output Precision", 3);
  printParams.set("Output Processor", 0);
  printParams.set("Output Information",
      ::NOX::Utils::OuterIteration + ::NOX::Utils::OuterIterationStatusTest +
          ::NOX::Utils::InnerIteration + ::NOX::Utils::Details + ::NOX::Utils::Warning);

  // Set NOX solving parameters: "Update Jacobian" is essential to compute the Newton direction,
  // i.e. x=A^-1 b. For details behind how it's handled, see NOX_Solver_SingleStep.C
  Teuchos::ParameterList& sssParams = p.sublist("Single Step Solver");
  sssParams.set("Update Jacobian", true);
  sssParams.set("Print Norms", true);
  sssParams.set("Compute Relative Norm", true);

  // Sublist for linear solver for the single step
  Teuchos::ParameterList& lsParams = sssParams.sublist("Linear Solver");
  lsParams.set("Adaptive Control", false);
  lsParams.set("Number of Nonlinear Iterations", 1);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Nln::SOLVER::SingleStep::reset_params()
{
  set_single_step_params(nlnglobaldata_->get_nln_parameter_list());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
enum Inpar::Solid::ConvergenceStatus Solid::Nln::SOLVER::SingleStep::solve()
{
  check_init_setup();

#if !(FOUR_C_TRILINOS_INTERNAL_VERSION_GE(2025, 4))
  auto& nln_group = dynamic_cast<NOX::Nln::Group&>(*group_ptr());
#endif

  const auto& x_epetra = dynamic_cast<const ::NOX::Epetra::Vector&>(group_ptr()->getX());

#if !(FOUR_C_TRILINOS_INTERNAL_VERSION_GE(2025, 4))
  nln_group.set_is_valid_newton(true);  // to circumvent the check in ::NOX::Solver::SingleStep
  nln_group.set_is_valid_rhs(false);    // force to compute the RHS
  nln_group.reset_x();                  // to initialize the solution vector to zero
#else
  // This nullifies the solution vector in the group
  group_ptr()->computeX(*group_ptr(), group_ptr()->getX(), -1.0);
#endif

  //// do one non-linear step using solve
  ::NOX::StatusTest::StatusType stepstatus = nlnsolver_->solve();

  integrator().set_state(Core::LinAlg::Vector<double>(x_epetra.getEpetraVector()));

  return convert_final_status(stepstatus);
}

FOUR_C_NAMESPACE_CLOSE
