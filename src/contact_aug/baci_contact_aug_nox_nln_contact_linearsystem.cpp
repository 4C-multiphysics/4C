/*----------------------------------------------------------------------*/
/*! \file
\brief Derived class which manages the special requirements to the linear
       solver for contact problems.

\level 3

*/
/*----------------------------------------------------------------------*/
#include "baci_contact_aug_nox_nln_contact_linearsystem.H"  // base class

#include "baci_contact_abstract_strategy.H"
#include "baci_inpar_contact.H"
#include "baci_linalg_blocksparsematrix.H"
#include "baci_linalg_utils_sparse_algebra_assemble.H"
#include "baci_linalg_utils_sparse_algebra_manipulation.H"
#include "baci_linear_solver_method_linalg.H"
#include "baci_mortar_strategy_base.H"
#include "baci_solver_nonlin_nox_aux.H"
#include "baci_solver_nonlin_nox_interface_jacobian.H"
#include "baci_solver_nonlin_nox_interface_required.H"

BACI_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::NLN::CONTACT::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams, const SolverMap& solvers,
    const Teuchos::RCP<::NOX::Epetra::Interface::Required>& iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const NOX::NLN::CONSTRAINT::ReqInterfaceMap& iConstr,
    const Teuchos::RCP<CORE::LINALG::SparseOperator>& J,
    const Teuchos::RCP<::NOX::Epetra::Interface::Preconditioner>& iPrec,
    const NOX::NLN::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
    const Teuchos::RCP<CORE::LINALG::SparseOperator>& M, const ::NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<::NOX::Epetra::Scaling> scalingObject)
    : NOX::NLN::LinearSystem(printParams, linearSolverParams, solvers, iReq, iJac, J, iPrec, M,
          cloneVector, scalingObject),
      iConstr_(iConstr),
      iConstrPrec_(iConstrPrec),
      p_lin_prob_(*this)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::NLN::CONTACT::LinearSystem::LinearSystem(Teuchos::ParameterList& printParams,
    Teuchos::ParameterList& linearSolverParams, const SolverMap& solvers,
    const Teuchos::RCP<::NOX::Epetra::Interface::Required>& iReq,
    const Teuchos::RCP<::NOX::Epetra::Interface::Jacobian>& iJac,
    const NOX::NLN::CONSTRAINT::ReqInterfaceMap& iConstr,
    const Teuchos::RCP<CORE::LINALG::SparseOperator>& J,
    const Teuchos::RCP<::NOX::Epetra::Interface::Preconditioner>& iPrec,
    const NOX::NLN::CONSTRAINT::PrecInterfaceMap& iConstrPrec,
    const Teuchos::RCP<CORE::LINALG::SparseOperator>& M, const ::NOX::Epetra::Vector& cloneVector)
    : NOX::NLN::LinearSystem(
          printParams, linearSolverParams, solvers, iReq, iJac, J, iPrec, M, cloneVector),
      iConstr_(iConstr),
      iConstrPrec_(iConstrPrec),
      p_lin_prob_(*this)
{
  // empty
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CORE::LINALG::SolverParams NOX::NLN::CONTACT::LinearSystem::SetSolverOptions(
    Teuchos::ParameterList& p, Teuchos::RCP<CORE::LINALG::Solver>& solverPtr,
    const NOX::NLN::SolutionType& solverType)
{
  CORE::LINALG::SolverParams solver_params;

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
    Teuchos::RCP<NOX::NLN::Interface::Required> iNlnReq =
        Teuchos::rcp_dynamic_cast<NOX::NLN::Interface::Required>(reqInterfacePtr_);
    if (iNlnReq.is_null()) throwError("setSolverOptions", "required interface cast failed");

    double worst = iNlnReq->CalcRefNormForce();
    // This value has to be specified in the PrePostOperator object of
    // the non-linear solver (i.e. runPreSolve())
    double wanted = p.get<double>("Wanted Tolerance");
    solver_params.nonlin_tolerance = wanted;
    solver_params.nonlin_residual = worst;
    solver_params.lin_tol_better = adaptiveControlObjective;
  }

  // nothing more to do for a pure structural solver
  if (solverType == NOX::NLN::sol_structure) return solver_params;

  // update information about active slave dofs
  // ---------------------------------------------------------------------
  // feed solver/preconditioner with additional information about the
  // contact/meshtying problem
  // ---------------------------------------------------------------------
  {
    // TODO: maps for merged meshtying and contact problem !!!
    // feed Belos based solvers with contact information
    if (solverPtr->Params().isSublist("Belos Parameters"))
    {
      if (iConstrPrec_.size() > 1)
        dserror(
            "Currently only one constraint preconditioner interface can be handled! \n "
            "Needs to be extended!");

      Teuchos::ParameterList& mueluParams = solverPtr->Params().sublist("Belos Parameters");

      // vector entries:
      // (0) masterDofMap
      // (1) slaveDofMap
      // (2) innerDofMap
      // (3) activeDofMap
      std::vector<Teuchos::RCP<Epetra_Map>> prec_maps(4, Teuchos::null);
      iConstrPrec_.begin()->second->FillMapsForPreconditioner(prec_maps);
      Teuchos::ParameterList& linSystemProps = mueluParams.sublist("Linear System properties");
      linSystemProps.set<Teuchos::RCP<Epetra_Map>>("contact masterDofMap", prec_maps[0]);
      linSystemProps.set<Teuchos::RCP<Epetra_Map>>("contact slaveDofMap", prec_maps[1]);
      linSystemProps.set<Teuchos::RCP<Epetra_Map>>("contact innerDofMap", prec_maps[2]);
      linSystemProps.set<Teuchos::RCP<Epetra_Map>>("contact activeDofMap", prec_maps[3]);
      // contact or contact/meshtying
      if (iConstrPrec_.begin()->first == NOX::NLN::sol_contact)
        linSystemProps.set<std::string>("GLOBAL::ProblemType", "contact");
      // only meshtying
      else if (iConstrPrec_.begin()->first == NOX::NLN::sol_meshtying)
        linSystemProps.set<std::string>("GLOBAL::ProblemType", "meshtying");
      else
        dserror("Currently we support only a pure meshtying OR a pure contact problem!");

      linSystemProps.set<int>("time step", step);
      // increase counter by one (historical reasons)
      linSystemProps.set<int>("iter", nlnIter + 1);
    }
  }  // end: feed solver with contact/meshtying information

  return solver_params;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::CONTACT::LinearSystem::SetLinearProblemForSolve(Epetra_LinearProblem& linear_problem,
    CORE::LINALG::SparseOperator& jac, Epetra_Vector& lhs, Epetra_Vector& rhs) const
{
  switch (jacType_)
  {
    case NOX::NLN::LinSystem::LinalgSparseMatrix:
      NOX::NLN::LinearSystem::SetLinearProblemForSolve(linear_problem, jac, lhs, rhs);

      break;
    case NOX::NLN::LinSystem::LinalgBlockSparseMatrix:
    {
      p_lin_prob_.ExtractActiveBlocks(jac, lhs, rhs);
      NOX::NLN::LinearSystem::SetLinearProblemForSolve(
          linear_problem, *p_lin_prob_.p_jac_, *p_lin_prob_.p_lhs_, *p_lin_prob_.p_rhs_);

      break;
    }
    default:
    {
      dserror("Unsupported matrix type! Type = %s",
          NOX::NLN::LinSystem::OperatorType2String(jacType_).c_str());

      exit(EXIT_FAILURE);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::CONTACT::LinearSystem::CompleteSolutionAfterSolve(
    const Epetra_LinearProblem& linProblem, Epetra_Vector& lhs) const
{
  p_lin_prob_.InsertIntoGlobalLhs(lhs);
  p_lin_prob_.Reset();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
NOX::NLN::SolutionType NOX::NLN::CONTACT::LinearSystem::GetActiveLinSolver(
    const std::map<NOX::NLN::SolutionType, Teuchos::RCP<CORE::LINALG::Solver>>& solvers,
    Teuchos::RCP<CORE::LINALG::Solver>& currSolver)
{
  // ---------------------------------------------------------------------
  // Solving a saddle point system
  // (1) Standard / Dual Lagrange multipliers -> SaddlePoint
  // (2) Direct Augmented Lagrange strategy
  // ---------------------------------------------------------------------
  NOX::NLN::CONSTRAINT::PrecInterfaceMap::const_iterator cit;
  bool issaddlepoint = false;
  for (cit = iConstrPrec_.begin(); cit != iConstrPrec_.end(); ++cit)
  {
    if (cit->second->IsSaddlePointSystem())
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
  for (cit = iConstrPrec_.begin(); cit != iConstrPrec_.end(); ++cit)
  {
    if (cit->second->IsCondensedSystem())
    {
      iscondensed = true;
      break;
    }
  }

  if (issaddlepoint or iscondensed)
  {
    currSolver = GetLinearContactSolver(solvers);
    return NOX::NLN::sol_contact;
  }
  // ----------------------------------------------------------------------
  // check if contact contributions are present,
  // if not we make a standard solver call to speed things up
  // ----------------------------------------------------------------------
  if (!issaddlepoint and !iscondensed)
  {
    currSolver = solvers.at(NOX::NLN::sol_structure);
    return NOX::NLN::sol_structure;
  }

  // default return
  currSolver = GetLinearContactSolver(solvers);
  return NOX::NLN::sol_contact;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<CORE::LINALG::Solver> NOX::NLN::CONTACT::LinearSystem::GetLinearContactSolver(
    const std::map<NOX::NLN::SolutionType, Teuchos::RCP<CORE::LINALG::Solver>>& solvers) const
{
  auto cs_iter = solvers.find(NOX::NLN::sol_contact);
  if (cs_iter != solvers.end() and not cs_iter->second.is_null()) return cs_iter->second;

  /* If no general linear solver is provided we ask the currently active
   * strategy for a meaningful linear solver. This is a small work around
   * to enable different linear solvers for changing contact solving strategies.
   *                                                       06/18 -- hiermeier */
  CORE::LINALG::Solver* linsolver = iConstrPrec_.begin()->second->GetLinearSolver();

  if (not linsolver)
    dserror(
        "Neither the general solver map, nor the constraint preconditioner "
        "knows the correct linear solver. Please check your input file.");

  return Teuchos::rcpFromRef(*linsolver);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::CONTACT::LinearSystem::LinearSubProblem::InsertIntoGlobalLhs(
    Epetra_Vector& glhs) const
{
  if (p_lhs_.is_null() or p_lhs_.get() == &glhs) return;

  CORE::LINALG::AssembleMyVector(0.0, glhs, 1.0, *p_lhs_);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::CONTACT::LinearSystem::LinearSubProblem::SetOriginalSystem(
    CORE::LINALG::SparseOperator& mat, Epetra_Vector& lhs, Epetra_Vector& rhs)
{
  p_jac_ = Teuchos::rcpFromRef(mat);
  p_rhs_ = Teuchos::rcpFromRef(rhs);
  p_lhs_ = Teuchos::rcpFromRef(lhs);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::CONTACT::LinearSystem::LinearSubProblem::ExtractActiveBlocks(
    CORE::LINALG::SparseOperator& mat, Epetra_Vector& lhs, Epetra_Vector& rhs)
{
  CORE::LINALG::BlockSparseMatrix<CORE::LINALG::DefaultBlockMatrixStrategy>& block_mat =
      dynamic_cast<CORE::LINALG::BlockSparseMatrix<CORE::LINALG::DefaultBlockMatrixStrategy>&>(mat);

  const int numrows = block_mat.Rows();
  const int numcols = block_mat.Cols();

  if (numrows != numcols)
    dserror("The number of row blocks must be equal the number of column blocks.");

  std::vector<std::vector<bool>> isempty(numrows, std::vector<bool>(numcols, false));

  std::vector<bool> isdiagonal(numrows, false);

  for (int r = 0; r < numrows; ++r)
  {
    for (int c = 0; c < numcols; ++c)
    {
      if (block_mat(r, c).EpetraMatrix()->GlobalMaxNumEntries() == 0 or
          block_mat(r, c).EpetraMatrix()->NormFrobenius() == 0.0)
        isempty[r][c] = true;
    }

    if (block_mat(r, r).EpetraMatrix()->NumGlobalDiagonals() ==
            block_mat(r, r).EpetraMatrix()->NumGlobalNonzeros() &&
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
    linsys_.applyDiagonalInverse(block_mat(*it, *it), lhs, rhs);

  // build remaining active linear problem
  const unsigned num_remaining_row_col = keep_row_col_index.size();
  const unsigned num_skip_row_col = skip_row_col_index.size();

  // nothing to skip, use the given linear problem
  switch (num_skip_row_col)
  {
    case 0:
    {
      SetOriginalSystem(mat, lhs, rhs);
      return;
    }
    default:
      break;
  }

  switch (num_remaining_row_col)
  {
    case 0:
    {
      dserror(
          "You are trying to solve a pure diagonal matrix. This is currently not"
          "supported, but feel free to extend the functionality.");

      exit(EXIT_FAILURE);
    }
    case 1:
    {
      const unsigned rc_id = keep_row_col_index[0];
      p_jac_ = Teuchos::rcpFromRef(block_mat(rc_id, rc_id));

      Teuchos::RCP<CORE::LINALG::SparseMatrix> active_sparse_mat =
          Teuchos::rcp_dynamic_cast<CORE::LINALG::SparseMatrix>(p_jac_, true);

      p_rhs_ = CORE::LINALG::ExtractMyVector(rhs, active_sparse_mat->RangeMap());
      p_lhs_ = CORE::LINALG::ExtractMyVector(lhs, active_sparse_mat->DomainMap());

      break;
    }
    default:
    {
      p_jac_ = block_mat.Clone(CORE::LINALG::View, keep_row_col_index, keep_row_col_index);
      p_jac_->Complete();

      Teuchos::RCP<CORE::LINALG::BlockSparseMatrixBase> active_block_mat =
          Teuchos::rcp_dynamic_cast<CORE::LINALG::BlockSparseMatrixBase>(p_jac_, true);

      p_rhs_ = CORE::LINALG::ExtractMyVector(rhs, active_block_mat->FullRangeMap());
      p_lhs_ = CORE::LINALG::ExtractMyVector(lhs, active_block_mat->FullDomainMap());

      break;
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void NOX::NLN::CONTACT::LinearSystem::applyDiagonalInverse(
    CORE::LINALG::SparseMatrix& mat, Epetra_Vector& lhs, const Epetra_Vector& rhs) const
{
  if (mat.EpetraMatrix()->NumGlobalDiagonals() != mat.EpetraMatrix()->NumGlobalNonzeros())
    dserror("The given matrix seems to be no diagonal matrix!");

  Epetra_Vector lhs_block(mat.DomainMap(), true);
  CORE::LINALG::ExtractMyVector(lhs, lhs_block);

  Epetra_Vector rhs_block(mat.RangeMap(), true);
  CORE::LINALG::ExtractMyVector(rhs, rhs_block);

  Epetra_Vector diag_mat(mat.RangeMap(), true);
  int err = mat.ExtractDiagonalCopy(diag_mat);
  if (err) dserror("ExtractDiagonalCopy failed! (err=%d)", err);

  err = lhs_block.ReciprocalMultiply(1.0, diag_mat, rhs_block, 0.0);
  if (err) dserror("ReciprocalMultiply failed! (err=%d)", err);

  CORE::LINALG::AssembleMyVector(0.0, lhs, 1.0, lhs_block);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void NOX::NLN::CONTACT::LinearSystem::throwError(
    const std::string& functionName, const std::string& errorMsg) const
{
  if (utils_.isPrintType(::NOX::Utils::Error))
  {
    utils_.out() << "NOX::CONTACT::LinearSystem::" << functionName << " - " << errorMsg
                 << std::endl;
  }
  throw "NOX Error";
}

BACI_NAMESPACE_CLOSE
