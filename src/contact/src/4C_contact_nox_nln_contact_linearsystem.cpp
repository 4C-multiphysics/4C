// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_nox_nln_contact_linearsystem.hpp"  // base class

#include "4C_contact_abstract_strategy.hpp"
#include "4C_contact_input.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mortar_strategy_base.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian.hpp"
#include "4C_solver_nonlin_nox_interface_required.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::CONTACT::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams, const SolverMap& solvers,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& J,
    const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& M, const NOX::Nln::Vector& cloneVector,
    const std::shared_ptr<NOX::Nln::Scaling> scalingObject)
    : NOX::Nln::LinearSystem(
          printParams, linearSolverParams, solvers, iReq, iJac, J, M, cloneVector, scalingObject),
      i_constr_(iConstr),
      i_constr_prec_(iConstrPrec),
      p_lin_prob_(*this)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::CONTACT::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams, const SolverMap& solvers,
    const std::shared_ptr<NOX::Nln::Interface::RequiredBase> iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const NOX::Nln::CONSTRAINT::ReqInterfaceMap& iConstr,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& J,
    const NOX::Nln::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
    const Teuchos::RCP<Core::LinAlg::SparseOperator>& M, const NOX::Nln::Vector& cloneVector)
    : NOX::Nln::LinearSystem(
          printParams, linearSolverParams, solvers, iReq, iJac, J, M, cloneVector),
      i_constr_(iConstr),
      i_constr_prec_(iConstrPrec),
      p_lin_prob_(*this)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::LinAlg::SolverParams NOX::Nln::CONTACT::LinearSystem::set_solver_options(
    Teuchos::ParameterList& p, Teuchos::RCP<Core::LinAlg::Solver>& solverPtr,
    const NOX::Nln::SolutionType& solverType)
{
  Core::LinAlg::SolverParams solver_params;

  bool isAdaptiveControl = p.get<bool>("Adaptive Control");
  double adaptiveControlObjective = p.get<double>("Adaptive Control Objective");
  // This value is specified in the underlying time integrator
  // (i.e. RunPreNoxNlnSolve())
  int step = p.get<int>("Current Time Step");
  // This value is specified in the PrePostOperator object of
  // the non-linear solver (i.e. runPreIterate())
  int nlnIter = p.get<int>("Number of Nonlinear Iterations");

  if (isAdaptiveControl)
  {
    // dynamic cast of the required/rhs interface
    const auto iNlnReq = std::dynamic_pointer_cast<NOX::Nln::Interface::Required>(reqInterfacePtr_);
    FOUR_C_ASSERT(iNlnReq,
        "NOX::Nln::CONTACT::LinearSystem::set_solver_options(): required interface cast "
        "failed");

    double worst = iNlnReq->calc_ref_norm_force();
    // This value has to be specified in the PrePostOperator object of
    // the non-linear solver (i.e. runPreSolve())
    double wanted = p.get<double>("Wanted Tolerance");
    solver_params.nonlin_tolerance = wanted;
    solver_params.nonlin_residual = worst;
    solver_params.lin_tol_better = adaptiveControlObjective;
  }

  // nothing more to do for a pure structural solver
  if (solverType == NOX::Nln::sol_structure) return solver_params;

  // update information about active slave dofs
  // ---------------------------------------------------------------------
  // feed solver/preconditioner with additional information about the
  // contact/meshtying problem
  // ---------------------------------------------------------------------
  {
    // TODO: maps for merged meshtying and contact problem !!!
    // feed Belos based solvers with contact information
    if (solverPtr->params().isSublist("Belos Parameters"))
    {
      if (i_constr_prec_.size() > 1)
        FOUR_C_THROW(
            "Currently only one constraint preconditioner interface can be handled! \n "
            "Needs to be extended!");

      Teuchos::ParameterList& mueluParams = solverPtr->params().sublist("Belos Parameters");

      // vector entries:
      // (0) masterDofMap
      // (1) slaveDofMap
      // (2) innerDofMap
      // (3) activeDofMap
      std::vector<Teuchos::RCP<Core::LinAlg::Map>> prec_maps(4, Teuchos::null);
      i_constr_prec_.begin()->second->fill_maps_for_preconditioner(prec_maps);
      mueluParams.set<Teuchos::RCP<Epetra_Map>>(
          "contact masterDofMap", Teuchos::rcpFromRef(prec_maps[0]->get_epetra_map()));
      mueluParams.set<Teuchos::RCP<Epetra_Map>>(
          "contact slaveDofMap", Teuchos::rcpFromRef(prec_maps[1]->get_epetra_map()));
      mueluParams.set<Teuchos::RCP<Epetra_Map>>(
          "contact innerDofMap", Teuchos::rcpFromRef(prec_maps[2]->get_epetra_map()));
      mueluParams.set<Teuchos::RCP<Epetra_Map>>(
          "contact activeDofMap", Teuchos::rcpFromRef(prec_maps[3]->get_epetra_map()));
      // contact or contact/meshtying
      if (i_constr_prec_.begin()->first == NOX::Nln::sol_contact)
        mueluParams.set<std::string>("Core::ProblemType", "contact");
      // only meshtying
      else if (i_constr_prec_.begin()->first == NOX::Nln::sol_meshtying)
        mueluParams.set<std::string>("Core::ProblemType", "meshtying");
      else
        FOUR_C_THROW("Currently we support only a pure meshtying OR a pure contact problem!");

      mueluParams.set<int>("time step", step);
      // increase counter by one (historical reasons)
      mueluParams.set<int>("iter", nlnIter + 1);
    }
  }  // end: feed solver with contact/meshtying information

  return solver_params;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::LinearProblem NOX::Nln::CONTACT::LinearSystem::set_linear_problem_for_solve(
    Core::LinAlg::SparseOperator& jac, Core::LinAlg::Vector<double>& lhs,
    Core::LinAlg::Vector<double>& rhs) const
{
  switch (jacType_)
  {
    case NOX::Nln::LinSystem::LinalgSparseMatrix:
      return NOX::Nln::LinearSystem::set_linear_problem_for_solve(jac, lhs, rhs);

    case NOX::Nln::LinSystem::LinalgBlockSparseMatrix:
    {
      p_lin_prob_.extract_active_blocks(jac, lhs, rhs);
      return NOX::Nln::LinearSystem::set_linear_problem_for_solve(
          *p_lin_prob_.p_jac_, *p_lin_prob_.p_lhs_, *p_lin_prob_.p_rhs_);
    }
    default:
    {
      FOUR_C_THROW("Unsupported matrix type! Type = {}",
          NOX::Nln::LinSystem::operator_type_to_string(jacType_));
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::CONTACT::LinearSystem::complete_solution_after_solve(
    const NOX::Nln::LinearProblem& linProblem, Core::LinAlg::Vector<double>& lhs) const
{
  p_lin_prob_.insert_into_global_lhs(lhs);
  p_lin_prob_.reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::Nln::SolutionType NOX::Nln::CONTACT::LinearSystem::get_active_lin_solver(
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers,
    Teuchos::RCP<Core::LinAlg::Solver>& currSolver)
{
  // ---------------------------------------------------------------------
  // Solving a saddle point system
  // (1) Standard / Dual Lagrange multipliers -> SaddlePoint
  // (2) Direct Augmented Lagrange strategy
  // ---------------------------------------------------------------------
  NOX::Nln::CONSTRAINT::PrecInterfaceMap::const_iterator cit;
  bool issaddlepoint = false;
  for (cit = i_constr_prec_.begin(); cit != i_constr_prec_.end(); ++cit)
  {
    if (cit->second->is_saddle_point_system())
    {
      issaddlepoint = true;
      break;
    }
  }
  // ---------------------------------------------------------------------
  // Solving a purely displacement based system
  // (1) Dual (not Standard) Lagrange multipliers -> Condensed
  // (2) Penalty and Uzawa Augmented Lagrange strategies
  // ---------------------------------------------------------------------
  bool iscondensed = false;
  for (cit = i_constr_prec_.begin(); cit != i_constr_prec_.end(); ++cit)
  {
    if (cit->second->is_condensed_system())
    {
      iscondensed = true;
      break;
    }
  }

  if (issaddlepoint or iscondensed)
  {
    currSolver = get_linear_contact_solver(solvers);
    return NOX::Nln::sol_contact;
  }
  // ----------------------------------------------------------------------
  // check if contact contributions are present,
  // if not we make a standard solver call to speed things up
  // ----------------------------------------------------------------------
  if (!issaddlepoint and !iscondensed)
  {
    currSolver = solvers.at(NOX::Nln::sol_structure);
    return NOX::Nln::sol_structure;
  }

  // default return
  currSolver = get_linear_contact_solver(solvers);
  return NOX::Nln::sol_contact;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::Solver> NOX::Nln::CONTACT::LinearSystem::get_linear_contact_solver(
    const std::map<NOX::Nln::SolutionType, Teuchos::RCP<Core::LinAlg::Solver>>& solvers) const
{
  auto cs_iter = solvers.find(NOX::Nln::sol_contact);
  if (cs_iter != solvers.end() and not cs_iter->second.is_null()) return cs_iter->second;

  /* If no general linear solver is provided we ask the currently active
   * strategy for a meaningful linear solver. This is a small work around
   * to enable different linear solvers for changing contact solving strategies.
   *                                                       06/18 -- hiermeier */
  Core::LinAlg::Solver* linsolver = i_constr_prec_.begin()->second->get_linear_solver();

  if (not linsolver)
    FOUR_C_THROW(
        "Neither the general solver map, nor the constraint preconditioner "
        "knows the correct linear solver. Please check your input file.");

  return Teuchos::rcpFromRef(*linsolver);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::CONTACT::LinearSystem::LinearSubProblem::insert_into_global_lhs(
    Core::LinAlg::Vector<double>& glhs) const
{
  if (p_lhs_.is_null() or p_lhs_.get() == &glhs) return;

  Core::LinAlg::assemble_my_vector(0.0, glhs, 1.0, *p_lhs_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::CONTACT::LinearSystem::LinearSubProblem::set_original_system(
    Core::LinAlg::SparseOperator& mat, Core::LinAlg::Vector<double>& lhs,
    Core::LinAlg::Vector<double>& rhs)
{
  p_jac_ = Teuchos::rcpFromRef(mat);
  p_rhs_ = Teuchos::rcpFromRef(rhs);
  p_lhs_ = Teuchos::rcpFromRef(lhs);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::CONTACT::LinearSystem::LinearSubProblem::extract_active_blocks(
    Core::LinAlg::SparseOperator& mat, Core::LinAlg::Vector<double>& lhs,
    Core::LinAlg::Vector<double>& rhs)
{
  Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>& block_mat =
      dynamic_cast<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>&>(mat);

  const int numrows = block_mat.rows();
  const int numcols = block_mat.cols();

  if (numrows != numcols)
    FOUR_C_THROW("The number of row blocks must be equal the number of column blocks.");

  std::vector<std::vector<bool>> isempty(numrows, std::vector<bool>(numcols, false));

  std::vector<bool> isdiagonal(numrows, false);

  for (int r = 0; r < numrows; ++r)
  {
    for (int c = 0; c < numcols; ++c)
    {
      if (block_mat(r, c).global_max_num_entries() == 0 or block_mat(r, c).norm_frobenius() == 0.0)
        isempty[r][c] = true;
    }

    if (block_mat(r, r).num_global_diagonals() == block_mat(r, r).num_global_nonzeros() &&
        !isempty[r][r])
      isdiagonal[r] = true;
  }

  // find row/col index with empty off-diagonal blocks and
  // a diagonal matrix in the diagonal block. These rows/cols
  // can be identified by a skip_row_col value of one. These
  // sub-systems can be solved directly.
  std::vector<unsigned> skip_row_col_index;
  skip_row_col_index.reserve(isempty.size());

  std::vector<unsigned> keep_row_col_index;
  keep_row_col_index.reserve(isempty.size());

  for (unsigned i = 0; i < isempty.size(); ++i)
  {
    int skip_row_col = 0;
    for (unsigned j = 0; j < isempty.size(); ++j)
      skip_row_col += (!isempty[i][j]) + (!isempty[j][i]);
    skip_row_col -= isdiagonal[i];

    switch (skip_row_col)
    {
      case 1:
      {
        skip_row_col_index.push_back(i);
        break;
      }
      default:
      {
        keep_row_col_index.push_back(i);
        break;
      }
    }
  }

  // solve sub-systems if possible
  for (std::vector<unsigned>::const_iterator it = skip_row_col_index.begin();
      it != skip_row_col_index.end(); ++it)
    linsys_.apply_diagonal_inverse(block_mat(*it, *it), lhs, rhs);

  // build remaining active linear problem
  const unsigned num_skip_row_col = skip_row_col_index.size();

  // nothing to skip, use the given linear problem
  switch (num_skip_row_col)
  {
    case 0:
    {
      set_original_system(mat, lhs, rhs);
      return;
    }
    default:
      break;
  }

  const unsigned rc_id = keep_row_col_index[0];
  p_jac_ = Teuchos::rcpFromRef(block_mat(rc_id, rc_id));

  Teuchos::RCP<Core::LinAlg::SparseMatrix> active_sparse_mat =
      Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(p_jac_, true);

  p_rhs_ =
      Teuchos::rcp(Core::LinAlg::extract_my_vector(rhs, active_sparse_mat->range_map()).release());
  p_lhs_ =
      Teuchos::rcp(Core::LinAlg::extract_my_vector(lhs, active_sparse_mat->domain_map()).release());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::Nln::CONTACT::LinearSystem::apply_diagonal_inverse(Core::LinAlg::SparseMatrix& mat,
    Core::LinAlg::Vector<double>& lhs, const Core::LinAlg::Vector<double>& rhs) const
{
  if (mat.num_global_diagonals() != mat.num_global_nonzeros())
    FOUR_C_THROW("The given matrix seems to be no diagonal matrix!");

  Core::LinAlg::Vector<double> lhs_block(mat.domain_map(), true);
  Core::LinAlg::extract_my_vector(lhs, lhs_block);

  Core::LinAlg::Vector<double> rhs_block(mat.range_map(), true);
  Core::LinAlg::extract_my_vector(rhs, rhs_block);

  Core::LinAlg::Vector<double> diag_mat(mat.range_map(), true);
  mat.extract_diagonal_copy(diag_mat);

  lhs_block.reciprocal_multiply(1.0, diag_mat, rhs_block, 0.0);

  Core::LinAlg::assemble_my_vector(0.0, lhs, 1.0, lhs_block);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::Nln::CONTACT::LinearSystem::throw_error(
    const std::string& functionName, const std::string& errorMsg) const
{
  if (utils_.isPrintType(::NOX::Utils::Error))
  {
    utils_.out() << "NOX::CONTACT::LinearSystem::" << functionName << " - " << errorMsg
                 << std::endl;
  }
  throw "NOX Error";
}

FOUR_C_NAMESPACE_CLOSE
