// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_amgnxn_smoothers.hpp"

#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_amgnxn_hierarchies.hpp"
#include "4C_linear_solver_amgnxn_vcycle.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.hpp"

#include <MueLu_EpetraOperator.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Xpetra_MultiVectorFactory.hpp>

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::GenericSmoother::richardson(GenericSmoother& Ainv,
    const BlockedMatrix& A, const BlockedVector& X, BlockedVector& Y, int iters, double omega,
    bool InitialGuessIsZero) const
{
  BlockedVector DX = X.deep_copy();
  BlockedVector DY = Y.deep_copy();  // TODO we only need a new vector

  for (int i = 0; i < iters; i++)
  {
    if (i != 0 or not InitialGuessIsZero)
    {
      A.apply(Y, DX);
      DX.update(1.0, X, -1.0);
    }

    // DY.PutScalar(0.0);
    Ainv.solve(DX, DY, true);

    if (i != 0 or not InitialGuessIsZero)
      Y.update(omega, DY, 1.0);
    else
      Y.update(omega, DY, 0.0);
  }
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::BgsSmoother::solve(
    const BlockedVector& X, BlockedVector& Y, bool InitialGuessIsZero) const
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGNxN::BgsSmoother::Solve");

  unsigned NumSuperBlocks = superblocks_.size();

  for (unsigned k = 0; k < iter_; k++)
  {
    for (unsigned i = 0; i < NumSuperBlocks; i++)
    {
      BlockedVector DXi = X.get_blocked_vector(superblocks_[i]).deep_copy();
      BlockedVector DXitmp = DXi.deep_copy();  // TODO we only need a new vector
      for (unsigned j = 0; j < NumSuperBlocks; j++)
      {
        if (k != 0 or not InitialGuessIsZero or j < i)
        {
          BlockedVector Yj = Y.get_blocked_vector(superblocks_[j]);
          BlockedMatrix Aij = a_->get_blocked_matrix(superblocks_[i], superblocks_[j]);
          Aij.apply(Yj, DXitmp);
          DXi.update(-1.0, DXitmp, 1.0);
        }
      }

      BlockedVector Yi = Y.get_blocked_vector(superblocks_[i]);
      BlockedVector DYi = Yi.deep_copy();  // TODO we only need a new vector
      BlockedMatrix Aii = a_->get_blocked_matrix(superblocks_[i], superblocks_[i]);
      richardson(*smoothers_[i], Aii, DXi, DYi, iters_[i], omegas_[i], true);

      if (k != 0 or not InitialGuessIsZero)
        Yi.update(omega_, DYi, 1.0);
      else
        Yi.update(omega_, DYi, 0.0);
    }
  }
}



/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SimpleSmoother::solve(
    const BlockedVector& X, BlockedVector& Y, bool InitialGuessIsZero) const
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGNxN::SimpleSmoother::Solve");

  BlockedVector Yp = Y.get_blocked_vector(blocks_pred_);
  BlockedVector Ys = Y.get_blocked_vector(blocks_schur_);
  const BlockedVector Xp = X.get_blocked_vector(blocks_pred_);
  const BlockedVector Xs = X.get_blocked_vector(blocks_schur_);

  // we need to allocate this dummy vectors only once
  if (xp_tmp_ == Teuchos::null)
  {
    xp_tmp_ = X.get_blocked_vector(blocks_pred_).deep_copy_rcp();
    xs_tmp_ = X.get_blocked_vector(blocks_schur_).deep_copy_rcp();
    yp_tmp_ = Y.get_blocked_vector(blocks_pred_).deep_copy_rcp();
    d_xp_ = X.get_blocked_vector(blocks_pred_).deep_copy_rcp();
    d_xs_ = X.get_blocked_vector(blocks_schur_).deep_copy_rcp();
    d_ys_ = Ys.deep_copy_rcp();
  }

  BlockedMatrix App = a_->get_blocked_matrix(blocks_pred_, blocks_pred_);
  BlockedMatrix Aps = a_->get_blocked_matrix(blocks_pred_, blocks_schur_);
  BlockedMatrix Asp = a_->get_blocked_matrix(blocks_schur_, blocks_pred_);
  BlockedMatrix Ass = a_->get_blocked_matrix(blocks_schur_, blocks_schur_);


  for (unsigned k = 0; k < iter_; k++)
  {
    // Extract blocks
    d_xp_->update(1.0, Xp, 0.0);
    d_xs_->update(1.0, Xs, 0.0);

    // Predictor equation
    if (k != 0 or not InitialGuessIsZero)
    {
      Aps.apply(Ys, *xp_tmp_);
      d_xp_->update(-1.0, *xp_tmp_, 1.0);   // Xp = Xp -  Aps*Ys;
      smoo_app_->solve(*d_xp_, Yp, false);  // Yp  = App^-1(Xp  - Aps*Ys)
    }
    else
      smoo_app_->solve(*d_xp_, Yp, true);  // Yp  = App^-1(Xp)

    // Compute Schur complement equation
    Asp.apply(Yp, *xs_tmp_);
    d_xs_->update(-1.0, *xs_tmp_, 1.0);  // Xs = Xs - Asp*Yp
    if (k != 0 or not InitialGuessIsZero)
    {
      Ass.apply(Ys, *xs_tmp_);
      d_xs_->update(-1.0, *xs_tmp_, 1.0);  // Xs = Xs - Asp*Yp - Ass*Ys
    }
    // DYs.PutScalar(0.0);
    smoo_schur_->solve(*d_xs_, *d_ys_, true);  // DYs = S^-1(Xs - Asp*Yp - Ass*Ys)

    // Schur update
    if (k != 0 or not InitialGuessIsZero)
      Ys.update(alpha_, *d_ys_, 1.0);  // Ys = Ys + alpha*DYs
    else
      Ys.update(alpha_, *d_ys_, 0.0);  // Ys = alpha*DYs

    // Predictor update
    Aps.apply(*d_ys_, *xp_tmp_);              // Xp_tmp = A12*DYs
    inv_app_->apply(*xp_tmp_, *yp_tmp_);      // Yp_tmp = App^-1*Aps*DYs
    Yp.update(-1.0 * alpha_, *yp_tmp_, 1.0);  // Yp = Yp - alpha*App^-1*Aps*DYs
  }
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::MergeAndSolve::setup(BlockedMatrix matrix)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGNxN::MergeAndSolve::Setup");

  if (Core::Communication::my_mpi_rank(
          Core::Communication::unpack_epetra_comm(matrix.get_matrix(0, 0)->Comm())) == 0)
  {
    std::cout
        << "Warning!!!: We are going to build a Core::LinAlg::BlockSparseMatrix. If this is a "
           "coarse level matrix, make sure that you have fixed the coarse maps of your AMG "
           "hierarchies (for all the blocks). Otherwise expect problems."
        << std::endl;
  }

  // Set matrix
  block_sparse_matrix_ = matrix.get_block_sparse_matrix(Core::LinAlg::DataAccess::View);
  sparse_matrix_ = block_sparse_matrix_->merge();
  a_ = std::dynamic_pointer_cast<Epetra_Operator>(sparse_matrix_->epetra_matrix());
  auto crsA = std::dynamic_pointer_cast<Epetra_CrsMatrix>(a_);
  FOUR_C_ASSERT_ALWAYS(crsA, "Houston, something went wrong in merging the matrix");

  // Set sol vector and rhs
  x_ = std::make_shared<Core::LinAlg::MultiVector<double>>(
      a_->OperatorDomainMap(), 1);  // TODO this one might cause problems
  b_ = std::make_shared<Core::LinAlg::MultiVector<double>>(a_->OperatorRangeMap(), 1);

  // Create linear solver
  Teuchos::ParameterList solvparams;
  Core::Utils::add_enum_class_to_parameter_list<Core::LinearSolver::SolverType>(
      "SOLVER", Core::LinearSolver::SolverType::umfpack, solvparams);
  solver_ = Teuchos::make_rcp<Core::LinAlg::Solver>(solvparams,
      Core::Communication::unpack_epetra_comm(a_->Comm()), nullptr,
      Core::IO::Verbositylevel::standard);

  // Set up solver
  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  solver_->setup(a_, x_, b_, solver_params);

  is_set_up_ = true;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::MergeAndSolve::solve(
    const BlockedVector& X, BlockedVector& Y, bool InitialGuessIsZero) const
{
  // TODO the memory allocation in this function can be improved
  // Seems that we are allocating too many vectors

  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGNxN::MergeAndSolve::Solve");
  if (not(is_set_up_))
    FOUR_C_THROW("The MergeAndSolve class should be set up before calling Apply");

  const Core::LinAlg::MultiMapExtractor& range_ex = block_sparse_matrix_->range_extractor();
  const Core::LinAlg::MultiMapExtractor& domain_ex = block_sparse_matrix_->domain_extractor();

  int NV = X.get_vector(0)->NumVectors();
  Core::LinAlg::MultiVector<double> Xmv(*(range_ex.full_map()), NV);
  Core::LinAlg::MultiVector<double> Ymv(*(domain_ex.full_map()), NV);

  for (int i = 0; i < X.get_num_blocks(); i++) range_ex.insert_vector(*(X.get_vector(i)), i, Xmv);

  b_->Update(1., Xmv, 0.);
  Core::LinAlg::SolverParams solver_params;
  solver_->solve_with_multi_vector(a_, x_, b_, solver_params);
  Ymv.Update(1., *x_, 0.);

  for (int i = 0; i < X.get_num_blocks(); i++) domain_ex.extract_vector(Ymv, i, *(Y.get_vector(i)));
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Core::LinearSolver::AMGNxN::CoupledAmg::CoupledAmg(Teuchos::RCP<AMGNxN::BlockedMatrix> A,
    std::vector<int> num_pdes, std::vector<int> null_spaces_dim,
    std::vector<std::shared_ptr<std::vector<double>>> null_spaces_data,
    const Teuchos::ParameterList& amgnxn_params, const Teuchos::ParameterList& smoothers_params,
    const Teuchos::ParameterList& muelu_params)
    : a_(std::move(A)),
      num_pdes_(std::move(num_pdes)),
      null_spaces_dim_(std::move(null_spaces_dim)),
      null_spaces_data_(std::move(null_spaces_data)),
      amgnxn_params_(amgnxn_params),
      smoothers_params_(smoothers_params),
      muelu_params_(muelu_params),
      is_setup_flag_(false)
{
  setup();
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::CoupledAmg::setup()
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGNxN::CoupledAmg::Setup");

  std::string verbosity = amgnxn_params_.get<std::string>("verbosity", "off");



  if (Core::Communication::my_mpi_rank(
          Core::Communication::unpack_epetra_comm(a_->get_matrix(0, 0)->Comm())) != 0)
    verbosity = "off";

  if (verbosity == "on")
  {
    std::cout << "===============================================" << std::endl;
    std::cout << "Core::LinAlg::SOLVER::AMGNxN::CoupledAmg : debug info  (begin)" << std::endl;
    std::cout << std::endl;
    std::cout << ">>>>>> Creating MueLu AMG Hierarchies for each one of the blocks" << std::endl;
    std::cout << std::endl;
  }

  // recover the muelu params
  int NumBlocks = a_->get_num_rows();
  for (int i = 0; i < NumBlocks; i++)
  {
    // recover the name of the list
    std::stringstream ss;
    ss << i;
    std::string param_name = "muelu parameters for block " + ss.str();
    std::string list_name = amgnxn_params_.get<std::string>(param_name, "none");
    if (list_name == "none")
      FOUR_C_THROW("You must specify the parameters for creating the AMG on block {}", i);

    // Parse contents of the list
    Teuchos::ParameterList muelu_list_this_block;
    if (not muelu_params_.isSublist(list_name)) FOUR_C_THROW("list {} not found", list_name);
    std::string xml_file = muelu_params_.sublist(list_name).get<std::string>("xml file", "none");
    if (xml_file != "none")
    {
      // If the xml file is not an absolute path, make it relative wrt the main xml file
      if ((xml_file)[0] != '/')
      {
        std::string tmp = smoothers_params_.get<std::string>("main xml path", "none");
        if (tmp == "none") FOUR_C_THROW("Path of the main xml not found");
        xml_file.insert(xml_file.begin(), tmp.begin(), tmp.end());
      }

      Teuchos::updateParametersFromXmlFile(
          xml_file, Teuchos::Ptr<Teuchos::ParameterList>(&muelu_list_this_block));
    }
    else
      muelu_list_this_block = muelu_params_.sublist(list_name);

    muelu_lists_.push_back(muelu_list_this_block);
  }


  int num_levels_amg = amgnxn_params_.get<int>("number of levels", 20);
  // if(num_levels_amg==-1)
  //  FOUR_C_THROW("Missing \"number of levels\" in your xml file");
  h_ = Teuchos::make_rcp<AMGNxN::Hierarchies>(
      a_, muelu_lists_, num_pdes_, null_spaces_dim_, null_spaces_data_, num_levels_amg, verbosity);

  if (verbosity == "on")
  {
    std::cout << std::endl;
    std::cout << ">>>>>> Creating the monolithic hierarchy" << std::endl;
    std::cout << std::endl;
  }

  m_ = Teuchos::make_rcp<AMGNxN::MonolithicHierarchy>(h_, amgnxn_params_, smoothers_params_);
  v_ = m_->build_v_cycle();

  if (verbosity == "on")
  {
    std::cout << std::endl;
    std::cout << "Core::LinAlg::SOLVER::AMGNxN::CoupledAmg : debug info  (end)" << std::endl;
    std::cout << "===============================================" << std::endl;
  }

  is_setup_flag_ = true;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::CoupledAmg::solve(
    const BlockedVector& X, BlockedVector& Y, bool InitialGuessIsZero) const
{
  if (!is_setup_flag_) FOUR_C_THROW("Solve cannot be called without a previous set up");

  v_->solve(X, Y, InitialGuessIsZero);
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::MueluSmootherWrapper::apply(
    const Core::LinAlg::MultiVector<double>& X, Core::LinAlg::MultiVector<double>& Y,
    bool InitialGuessIsZero) const
{
  if (InitialGuessIsZero)
    Y.PutScalar(0.0);  // I am not sure what is the meaning of InitialGuessIsZero in MueLu. This is
                       // for safety

  // Convert to Tpetra
  Core::LinAlg::MultiVector<double> X_rcp(X);

  Teuchos::RCP<Xpetra::EpetraMultiVectorT<GlobalOrdinal, Node>> Xex =
      Teuchos::make_rcp<Xpetra::EpetraMultiVectorT<GlobalOrdinal, Node>>(
          Teuchos::rcpFromRef(X_rcp.get_epetra_multi_vector()));

  Teuchos::RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> Xx =
      Teuchos::rcp_dynamic_cast<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(
          Xex);
  Core::LinAlg::MultiVector<double> Y_rcp(Y);

  Teuchos::RCP<Xpetra::EpetraMultiVectorT<GlobalOrdinal, Node>> Yex =
      Teuchos::make_rcp<Xpetra::EpetraMultiVectorT<GlobalOrdinal, Node>>(
          Teuchos::rcpFromRef(Y_rcp.get_epetra_multi_vector()));

  Teuchos::RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> Yx =
      Teuchos::rcp_dynamic_cast<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(
          Yex);

  // Apply underlying smoother
  s_->Apply(*Yx, *Xx, InitialGuessIsZero);

  // Convert to Epetra
  const auto& Ye = MueLu::Utilities<double, int, int, Node>::MV2NonConstEpetraMV(Yx);

  Y.Update(1.0, Core::LinAlg::MultiVector<double>(*Ye), 0.0);
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
Core::LinearSolver::AMGNxN::MueluHierarchyWrapper::MueluHierarchyWrapper(
    Teuchos::RCP<MueLu::Hierarchy<Scalar, LocalOrdinal, GlobalOrdinal, Node>> H)
    : h_(std::move(H))
{
  p_ = Teuchos::make_rcp<MueLu::EpetraOperator>(h_);
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::MueluHierarchyWrapper::apply(
    const Core::LinAlg::MultiVector<double>& X, Core::LinAlg::MultiVector<double>& Y,
    bool InitialGuessIsZero) const
{
  if (InitialGuessIsZero)
    Y.PutScalar(0.0);  // TODO Remove when you are sure that ApplyInverse will zero out Y.
  p_->ApplyInverse(X, Y);
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
Core::LinearSolver::AMGNxN::MueluAMGWrapper::MueluAMGWrapper(
    Teuchos::RCP<Core::LinAlg::SparseMatrix> A, int num_pde, int null_space_dim,
    std::shared_ptr<std::vector<double>> null_space_data, const Teuchos::ParameterList& muelu_list)
    : A_(std::move(A)),
      num_pde_(num_pde),
      null_space_dim_(null_space_dim),
      null_space_data_(std::move(null_space_data)),
      muelu_list_(muelu_list)
{
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::MueluAMGWrapper::build_hierarchy()
{
  // Prepare operator for MueLu
  auto A_crs =
      Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(Teuchos::rcpFromRef(*A_->epetra_operator()));
  if (A_crs == Teuchos::null)
    FOUR_C_THROW("Make sure that the input matrix is a Epetra_CrsMatrix (or derived)");
  Teuchos::RCP<Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> mueluA =
      Teuchos::make_rcp<Xpetra::EpetraCrsMatrixT<int, Xpetra::EpetraNode>>(A_crs);

  Teuchos::RCP<Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>> mueluA_wrap =
      Teuchos::make_rcp<Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(mueluA);
  Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> mueluOp =
      Teuchos::rcp_dynamic_cast<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(
          mueluA_wrap);

  // Prepare null space vector for MueLu
  // safety check
  const size_t localNumRows = mueluA->getLocalNumRows();

  if (localNumRows * null_space_dim_ != null_space_data_->size())
    FOUR_C_THROW("Matrix size is inconsistent with length of nullspace vector!");
  Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> rowMap = mueluA->getRowMap();
  Teuchos::RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> nspVector =
      Xpetra::MultiVectorFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(
          rowMap, null_space_dim_, true);
  for (size_t i = 0; i < Teuchos::as<size_t>(null_space_dim_); i++)
  {
    Teuchos::ArrayRCP<Scalar> nspVectori = nspVector->getDataNonConst(i);
    const size_t myLength = nspVector->getLocalLength();
    for (size_t j = 0; j < myLength; j++)
    {
      nspVectori[j] = (*null_space_data_)[i * myLength + j];
    }
  }


  // Input num eq and offset in the final level.
  // The amalgamation factory needs this info!
  int offsetFineLevel(0);
  if (num_pde_ > 1) offsetFineLevel = A_->row_map().min_all_gid();
  mueluOp->SetFixedBlockSize(num_pde_, offsetFineLevel);
  Teuchos::ParameterList& MatrixList = muelu_list_.sublist("Matrix");
  MatrixList.set<int>("DOF offset", offsetFineLevel);
  MatrixList.set<int>("number of equations", num_pde_);
  // std::cout << "offsetFineLevel " << offsetFineLevel << std::endl;



  // Build up hierarchy
  MueLu::ParameterListInterpreter<Scalar, LocalOrdinal, GlobalOrdinal, Node> mueLuFactory(
      muelu_list_);
  H_ = mueLuFactory.CreateHierarchy();
  H_->SetDefaultVerbLevel(MueLu::Extreme);  // TODO sure?
  H_->GetLevel(0)->Set("A", mueluOp);
  H_->GetLevel(0)->Set("Nullspace", nspVector);
  H_->GetLevel(0)->setlib(Xpetra::UseEpetra);
  H_->setlib(Xpetra::UseEpetra);
  mueLuFactory.SetupHierarchy(*H_);
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::MueluAMGWrapper::setup()
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::MueluAMGWrapper::Setup");

  Teuchos::Time timer("", true);
  timer.reset();

  // Create the hierarchy
  build_hierarchy();

  // Create the V-cycle
  p_ = Teuchos::make_rcp<MueLu::EpetraOperator>(H_);

  double elaptime = timer.totalElapsedTime(true);
  if (muelu_list_.sublist("Hierarchy").get<std::string>("verbosity", "None") != "None" and
      Core::Communication::my_mpi_rank(Core::Communication::unpack_epetra_comm(A_->Comm())) == 0)
    std::cout << "       Calling Core::LinAlg::SOLVER::AMGNxN::MueluAMGWrapper::Setup takes "
              << std::setw(16) << std::setprecision(6) << elaptime << " s" << std::endl;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::MueluAMGWrapper::apply(const Core::LinAlg::MultiVector<double>& X,
    Core::LinAlg::MultiVector<double>& Y, bool InitialGuessIsZero) const
{
  if (InitialGuessIsZero)
    Y.PutScalar(0.0);  // TODO Remove when you are sure that ApplyInverse will zero out Y.
  p_->ApplyInverse(X, Y);
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
Core::LinearSolver::AMGNxN::SingleFieldAMG::SingleFieldAMG(
    Teuchos::RCP<Core::LinAlg::SparseMatrix> A, int num_pde, int null_space_dim,
    std::shared_ptr<std::vector<double>> null_space_data, const Teuchos::ParameterList& muelu_list,
    const Teuchos::ParameterList& fine_smoother_list)
    : MueluAMGWrapper(A, num_pde, null_space_dim, null_space_data, muelu_list),
      fine_smoother_list_(fine_smoother_list)
{
  setup();
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::SingleFieldAMG::setup()
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::SingleFieldAMG::Setup");

  using MueLuUtils = MueLu::Utilities<double, int, int, Node>;

  Teuchos::Time timer("", true);
  timer.reset();

  // Create the hierarchy
  build_hierarchy();

  // recover info

  int NumLevels = H_->GetNumLevels();


  bool explicitdirichlet = A_->explicit_dirichlet();
  bool savegraph = A_->save_graph();

  Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> myA = Teuchos::null;
  Teuchos::RCP<Epetra_CrsMatrix> myAcrs = Teuchos::null;
  Teuchos::RCP<Core::LinAlg::SparseMatrix> myAspa = Teuchos::null;
  Teuchos::RCP<MueLu::SmootherBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>> myS = Teuchos::null;
  Teuchos::RCP<Core::LinearSolver::AMGNxN::MueluSmootherWrapper> mySWrap = Teuchos::null;

  std::vector<Teuchos::RCP<Core::LinAlg::SparseMatrix>> Avec(NumLevels, Teuchos::null);
  std::vector<Teuchos::RCP<Core::LinAlg::SparseMatrix>> Pvec(NumLevels - 1, Teuchos::null);
  std::vector<Teuchos::RCP<Core::LinAlg::SparseMatrix>> Rvec(NumLevels - 1, Teuchos::null);
  std::vector<Teuchos::RCP<SingleFieldSmoother>> SvecPre(NumLevels, Teuchos::null);
  std::vector<Teuchos::RCP<SingleFieldSmoother>> SvecPos(NumLevels - 1, Teuchos::null);


  // Get Muelu operators (except smoothers)
  for (int level = 0; level < NumLevels; level++)
  {
    Teuchos::RCP<MueLu::Level> this_level = H_->GetLevel(level);


    if (this_level->IsAvailable("A"))
    {
      myA =
          this_level->Get<Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>>(
              "A");
      myAcrs = MueLuUtils::Op2NonConstEpetraCrs(myA);
      myAspa =
          Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(Core::Utils::shared_ptr_from_ref(*myAcrs),
              Core::LinAlg::DataAccess::Copy, explicitdirichlet, savegraph);
      Avec[level] = myAspa;
    }
    else
      FOUR_C_THROW("Error in extracting A");

    if (level != 0)
    {
      if (this_level->IsAvailable("P"))
      {
        myA =
            this_level
                ->Get<Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>>("P");
        myAcrs = MueLuUtils::Op2NonConstEpetraCrs(myA);
        myAspa =
            Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(Core::Utils::shared_ptr_from_ref(*myAcrs),
                Core::LinAlg::DataAccess::Copy, explicitdirichlet, savegraph);
        Pvec[level - 1] = myAspa;
      }
      else
        FOUR_C_THROW("Error in extracting P");

      if (this_level->IsAvailable("R"))
      {
        myA =
            this_level
                ->Get<Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>>("R");
        myAcrs = MueLuUtils::Op2NonConstEpetraCrs(myA);
        myAspa =
            Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(Core::Utils::shared_ptr_from_ref(*myAcrs),
                Core::LinAlg::DataAccess::Copy, explicitdirichlet, savegraph);
        Rvec[level - 1] = myAspa;
      }
      else
        FOUR_C_THROW("Error in extracting R");
    }


    if (level == NumLevels - 1)
    {
      if (this_level->IsAvailable("PreSmoother"))
      {
        myS =
            this_level
                ->Get<Teuchos::RCP<MueLu::SmootherBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>>>(
                    "PreSmoother");
        mySWrap = Teuchos::make_rcp<Core::LinearSolver::AMGNxN::MueluSmootherWrapper>(myS);
        SvecPre[level] = mySWrap;
      }
      else
        FOUR_C_THROW("Error in extracting PreSmoother");
    }
  }


  // Build smoothers
  for (int level = 0; level < NumLevels - 1; level++)
  {
    SvecPre[level] = Teuchos::make_rcp<AMGNxN::IfpackWrapper>(Avec[level], fine_smoother_list_);
    SvecPos[level] = SvecPre[level];
  }


  // Build vcycle
  int NumSweeps = 1;
  int FirstLevel = 0;
  v_ = Teuchos::make_rcp<VcycleSingle>(NumLevels, NumSweeps, FirstLevel);
  v_->set_operators(Avec);
  v_->set_projectors(Pvec);
  v_->set_restrictors(Rvec);
  v_->set_pre_smoothers(SvecPre);
  v_->set_pos_smoothers(SvecPos);


  double elaptime = timer.totalElapsedTime(true);
  if (Core::Communication::my_mpi_rank(Core::Communication::unpack_epetra_comm(A_->Comm())) == 0)
    std::cout << "       Calling Core::LinAlg::SOLVER::AMGNxN::SingleFieldAMG::Setup takes "
              << std::setw(16) << std::setprecision(6) << elaptime << " s" << std::endl;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::SingleFieldAMG::apply(const Core::LinAlg::MultiVector<double>& X,
    Core::LinAlg::MultiVector<double>& Y, bool InitialGuessIsZero) const
{
  if (InitialGuessIsZero)
    Y.PutScalar(0.0);  // TODO Remove when you are sure that ApplyInverse will zero out Y.
  v_->apply(X, Y, InitialGuessIsZero);
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
Core::LinearSolver::AMGNxN::IfpackWrapper::IfpackWrapper(
    Teuchos::RCP<Core::LinAlg::SparseMatrix> A, Teuchos::ParameterList& list)
    : a_(std::move(A))
{
  // Determine the preconditioner type
  type_ = list.get<std::string>("type", "none");
  if (type_ == "none") FOUR_C_THROW("The type of preconditioner has to be provided.");

  int overlap = list.get<int>("overlap", 0);

  // Extract list of parameters
  if (not(list.isSublist("ParameterList"))) FOUR_C_THROW("The parameter list has to be provided");
  list_ = list.sublist("ParameterList");

  if (list_.isParameter("relaxation: zero starting solution"))
  {
    std::cout << "WARNING!!!!!: don't use the parameter 'relaxation: zero starting solution' this "
                 "is handled in 4C appropriately"
              << std::endl;
  }

  if (list_.isParameter("chebyshev: zero starting solution"))
  {
    std::cout << "WARNING!!!!!: don't use the parameter 'chebyshev: zero starting solution' this "
                 "is handled in 4C appropriately"
              << std::endl;
  }

  // Create smoother
  Ifpack Factory;
  arow_ = Teuchos::rcp_dynamic_cast<Epetra_RowMatrix>(Teuchos::rcpFromRef(*a_->epetra_matrix()));
  if (arow_ == Teuchos::null)
    FOUR_C_THROW("Something wrong. Be sure that the given matrix is not a block matrix");
  prec_ = Factory.Create(type_, arow_.get(), overlap);


  // Set parameter list and setup
  prec_->SetParameters(list_);
  prec_->Initialize();
  prec_->Compute();
}



/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::IfpackWrapper::apply(const Core::LinAlg::MultiVector<double>& X,
    Core::LinAlg::MultiVector<double>& Y, bool InitialGuessIsZero) const
{
  // We assume that ifpack is always setting the initial guess to zero
  // (This can be done using input parameters, but the default behavior is to zero out the initial
  // guess)
  if (InitialGuessIsZero)
  {
    prec_->ApplyInverse(X, Y);
  }
  else
  {
    Core::LinAlg::MultiVector<double> DX(X.get_map(), X.NumVectors(), false);
    a_->Apply(Y, DX);
    DX.Update(1.0, X, -1.0);
    Core::LinAlg::MultiVector<double> DY(Y.get_map(), X.NumVectors(), false);
    prec_->ApplyInverse(DX, DY);
    Y.Update(1.0, DY, 1.0);
  }
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::DirectSolverWrapper::setup(
    Teuchos::RCP<Core::LinAlg::SparseMatrix> matrix, Teuchos::RCP<Teuchos::ParameterList> params)
{
  // Set matrix
  a_ = std::dynamic_pointer_cast<Epetra_CrsMatrix>(matrix->epetra_matrix());

  // Set sol vector and rhs
  x_ = std::make_shared<Core::LinAlg::MultiVector<double>>(a_->OperatorDomainMap(), 1);
  b_ = std::make_shared<Core::LinAlg::MultiVector<double>>(a_->OperatorRangeMap(), 1);

  // Create linear solver. Default solver: UMFPACK
  const auto solvertype = params->get<std::string>("solver", "umfpack");

  if (solvertype == "umfpack")
  {
    Core::Utils::add_enum_class_to_parameter_list<Core::LinearSolver::SolverType>(
        "SOLVER", Core::LinearSolver::SolverType::umfpack, *params);
  }
  else if (solvertype == "superlu")
  {
    Core::Utils::add_enum_class_to_parameter_list<Core::LinearSolver::SolverType>(
        "SOLVER", Core::LinearSolver::SolverType::superlu, *params);
  }
  else
    FOUR_C_THROW("Solver type not supported as direct solver in AMGNXN framework");

  solver_ = std::make_shared<Core::LinAlg::Solver>(*params,
      Core::Communication::unpack_epetra_comm(a_->Comm()), nullptr,
      Core::IO::Verbositylevel::standard);

  // Set up solver
  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  solver_->setup(a_, x_, b_, solver_params);

  is_set_up_ = true;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::DirectSolverWrapper::apply(
    const Core::LinAlg::MultiVector<double>& X, Core::LinAlg::MultiVector<double>& Y,
    bool InitialGuessIsZero) const
{
  if (not(is_set_up_))
    FOUR_C_THROW("The DirectSolverWrapper class should be set up before calling Apply");

  b_->Update(1., X, 0.);
  Core::LinAlg::SolverParams solver_params;
  solver_->solve_with_multi_vector(a_, x_, b_, solver_params);
  Y.Update(1., *x_, 0.);
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


Core::LinearSolver::AMGNxN::SmootherManager::SmootherManager()
    : set_operator_(false),
      set_params_(false),
      set_params_subsolver_(false),
      set_hierarchies_(false),
      set_level_(false),
      set_block_(false),
      set_blocks_(false),
      set_type_(false),
      set_verbosity_(false),
      set_null_space_(false),
      set_null_space_all_blocks_(false)
{
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::BlockedMatrix>
Core::LinearSolver::AMGNxN::SmootherManager::get_operator()
{
  return operator_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::ParameterList Core::LinearSolver::AMGNxN::SmootherManager::get_params() { return params_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::ParameterList Core::LinearSolver::AMGNxN::SmootherManager::get_params_smoother()
{
  return params_subsolver_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::Hierarchies>
Core::LinearSolver::AMGNxN::SmootherManager::get_hierarchies()
{
  return hierarchies_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

int Core::LinearSolver::AMGNxN::SmootherManager::get_level() { return level_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

int Core::LinearSolver::AMGNxN::SmootherManager::get_block() { return block_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

std::vector<int> Core::LinearSolver::AMGNxN::SmootherManager::get_blocks() { return blocks_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

std::string Core::LinearSolver::AMGNxN::SmootherManager::get_smoother_name()
{
  return subsolver_name_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

std::string Core::LinearSolver::AMGNxN::SmootherManager::get_type() { return type_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

std::string Core::LinearSolver::AMGNxN::SmootherManager::get_verbosity() { return verbosity_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Core::LinearSolver::AMGNxN::NullSpaceInfo
Core::LinearSolver::AMGNxN::SmootherManager::get_null_space()
{
  return null_space_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

std::vector<Core::LinearSolver::AMGNxN::NullSpaceInfo>
Core::LinearSolver::AMGNxN::SmootherManager::get_null_space_all_blocks()
{
  return null_space_all_blocks_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_operator(Teuchos::RCP<BlockedMatrix> in)
{
  set_operator_ = true;
  operator_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_params(const Teuchos::ParameterList& in)
{
  set_params_ = true;
  params_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_params_smoother(
    const Teuchos::ParameterList& in)
{
  set_params_subsolver_ = true;
  params_subsolver_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_hierarchies(Teuchos::RCP<Hierarchies> in)
{
  set_hierarchies_ = true;
  hierarchies_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_level(int in)
{
  set_level_ = true;
  level_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_block(int in)
{
  set_block_ = true;
  block_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_blocks(std::vector<int> in)
{
  set_blocks_ = true;
  blocks_ = std::move(in);
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_smoother_name(std::string in)
{
  set_subsolver_name_ = true;
  subsolver_name_ = std::move(in);
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_type(std::string in)
{
  set_type_ = true;
  type_ = std::move(in);
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_verbosity(std::string in)
{
  set_verbosity_ = true;
  verbosity_ = std::move(in);
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_null_space(const NullSpaceInfo& in)
{
  set_null_space_ = true;
  null_space_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_null_space_all_blocks(
    const std::vector<NullSpaceInfo>& in)
{
  set_null_space_all_blocks_ = true;
  null_space_all_blocks_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_operator() { return set_operator_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_params() { return set_params_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_params_smoother()
{
  return set_params_subsolver_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_hierarchies() { return set_hierarchies_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_level() { return set_level_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_block() { return set_block_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_blocks() { return set_blocks_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_smoother_name()
{
  return set_subsolver_name_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_type() { return set_type_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_verbosity() { return set_verbosity_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_null_space() { return set_null_space_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_null_space_all_blocks()
{
  return set_null_space_all_blocks_;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherFactory::set_type_and_params()
{
  // Valid types
  std::vector<std::string> valid_types;
  valid_types.emplace_back("BGS");
  valid_types.emplace_back("IFPACK");
  valid_types.emplace_back("REUSE_MUELU_SMOOTHER");
  valid_types.emplace_back("REUSE_MUELU_AMG");
  valid_types.emplace_back("NEW_MUELU_AMG");
  valid_types.emplace_back("NEW_MUELU_AMG_IFPACK_SMO");
  valid_types.emplace_back("DIRECT_SOLVER");
  valid_types.emplace_back("MERGE_AND_SOLVE");
  valid_types.emplace_back("BLOCK_AMG");
  valid_types.emplace_back("SIMPLE");

  std::string smoother_type;
  Teuchos::ParameterList smoother_params;
  if (get_params_smoother().isSublist(get_smoother_name()))
  {
    smoother_type =
        get_params_smoother().sublist(get_smoother_name()).get<std::string>("type", "none");
    smoother_params = get_params_smoother().sublist(get_smoother_name()).sublist("parameters");
  }
  else if (std::find(valid_types.begin(), valid_types.end(), get_smoother_name()) !=
           valid_types.end())
    smoother_type = get_smoother_name();
  else
    smoother_type = "none";

  set_type(smoother_type);
  set_params(smoother_params);
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::GenericSmoother>
Core::LinearSolver::AMGNxN::SmootherFactory::create()
{
  // Expected parameters in GetParamsSmoother()
  //
  // <ParameterList name="mySmoother">
  //   <Parameter name="type"   type="string"  value="..."/>
  //   <ParameterList name="parameters">
  //
  //    ...    ...   ...   ...   ...   ...
  //
  //   </ParameterList>
  // </ParameterList>
  //
  // Available input?

  if (not is_set_params_smoother()) FOUR_C_THROW("IsSetParamsSmoother() returns false");
  if (not is_set_smoother_name()) FOUR_C_THROW("IsSetSmootherName() returns false");

  // Determine the type of smoother to be constructed and its parameters
  set_type_and_params();

  // Create the corresponding factory
  Teuchos::RCP<SmootherFactoryBase> mySmootherFactory = Teuchos::null;

  if (get_type() == "BGS")
  {
    mySmootherFactory = Teuchos::make_rcp<BgsSmootherFactory>();
    mySmootherFactory->set_operator(get_operator());
    mySmootherFactory->set_params(get_params());
    mySmootherFactory->set_params_smoother(get_params_smoother());
    mySmootherFactory->set_hierarchies(get_hierarchies());
    mySmootherFactory->set_blocks(get_blocks());
    mySmootherFactory->set_null_space_all_blocks(get_null_space_all_blocks());
  }
  else if (get_type() == "BLOCK_AMG")
  {
    mySmootherFactory = Teuchos::make_rcp<CoupledAmgFactory>();
    mySmootherFactory->set_operator(get_operator());
    mySmootherFactory->set_params(get_params());
    mySmootherFactory->set_params_smoother(get_params_smoother());
    mySmootherFactory->set_blocks(get_blocks());
    mySmootherFactory->set_null_space_all_blocks(get_null_space_all_blocks());
  }
  else if (get_type() == "SIMPLE")
  {
    mySmootherFactory = Teuchos::make_rcp<SimpleSmootherFactory>();
    mySmootherFactory->set_operator(get_operator());
    mySmootherFactory->set_params(get_params());
    mySmootherFactory->set_params_smoother(get_params_smoother());
    mySmootherFactory->set_hierarchies(get_hierarchies());
    mySmootherFactory->set_blocks(get_blocks());
    mySmootherFactory->set_null_space_all_blocks(get_null_space_all_blocks());
  }
  else if (get_type() == "MERGE_AND_SOLVE")
  {
    mySmootherFactory = Teuchos::make_rcp<MergeAndSolveFactory>();
    mySmootherFactory->set_operator(get_operator());
    mySmootherFactory->set_blocks(get_blocks());
    mySmootherFactory->set_level(get_level());
  }
  else if (get_type() == "DIRECT_SOLVER")
  {
    mySmootherFactory = Teuchos::make_rcp<DirectSolverWrapperFactory>();
    mySmootherFactory->set_operator(get_operator());
    mySmootherFactory->set_params(get_params());
    mySmootherFactory->set_block(get_block());
  }
  else if (get_type() == "IFPACK")
  {
    mySmootherFactory = Teuchos::make_rcp<IfpackWrapperFactory>();
    mySmootherFactory->set_operator(get_operator());
    mySmootherFactory->set_params(get_params());
    mySmootherFactory->set_block(get_block());
  }
  else if (get_type() == "REUSE_MUELU_SMOOTHER")
  {
    mySmootherFactory = Teuchos::make_rcp<MueluSmootherWrapperFactory>();
    mySmootherFactory->set_hierarchies(get_hierarchies());
    mySmootherFactory->set_block(get_block());
  }
  else if (get_type() == "REUSE_MUELU_AMG")
  {
    mySmootherFactory = Teuchos::make_rcp<HierarchyRemainderWrapperFactory>();
    mySmootherFactory->set_hierarchies(get_hierarchies());
    mySmootherFactory->set_block(get_block());
  }
  else if (get_type() == "NEW_MUELU_AMG")
  {
    mySmootherFactory = Teuchos::make_rcp<MueluAMGWrapperFactory>();
    mySmootherFactory->set_operator(get_operator());
    mySmootherFactory->set_hierarchies(get_hierarchies());
    mySmootherFactory->set_block(get_block());
    mySmootherFactory->set_params(get_params());
    mySmootherFactory->set_params_smoother(get_params_smoother());
    mySmootherFactory->set_null_space(get_null_space());
  }
  else if (get_type() == "NEW_MUELU_AMG_IFPACK_SMO")
  {
    mySmootherFactory = Teuchos::make_rcp<SingleFieldAMGFactory>();
    mySmootherFactory->set_operator(get_operator());
    mySmootherFactory->set_hierarchies(get_hierarchies());
    mySmootherFactory->set_block(get_block());
    mySmootherFactory->set_params(get_params());
    mySmootherFactory->set_params_smoother(get_params_smoother());
    mySmootherFactory->set_null_space(get_null_space());
  }
  else
    FOUR_C_THROW("Unknown smoother type. Fix your xml file");

  // Build the smoother
  mySmootherFactory->set_verbosity(get_verbosity());
  mySmootherFactory->set_level(get_level());
  return mySmootherFactory->create();
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::GenericSmoother>
Core::LinearSolver::AMGNxN::IfpackWrapperFactory::create()
{
  // Expected parameters with default values
  //
  // <ParameterList name="parameters">
  //   <Parameter name="type"                           type="string"  value="point relaxation"/>
  //   <ParameterList name="ParameterList">
  //     <Parameter name="relaxation: type"             type="string"  value="Gauss-Seidel"/>
  //     <Parameter name="relaxation: backward mode"    type="bool"    value="false"/>
  //     <Parameter name="relaxation: sweeps"           type="int"     value="2"/>
  //     <Parameter name="relaxation: damping factor"   type="double"  value="1.0"/>
  //   </ParameterList>
  // </ParameterList>
  //

  // Check input
  if (not is_set_params()) FOUR_C_THROW("IsSetParams() returns false");
  if (not is_set_operator()) FOUR_C_THROW("IsSetOperator() returns false");

  //  we don't want defaults.
  // Fill myparams with default values where required
  // Teuchos::ParameterList defaults;
  // defaults.set<std::string>("type","point relaxation");
  // defaults.sublist("ParameterList").set<std::string>("relaxation: type", "Gauss-Seidel");
  // defaults.sublist("ParameterList").set<bool>("relaxation: backward mode",false);
  // defaults.sublist("ParameterList").set<int>("relaxation: sweeps",2);
  // defaults.sublist("ParameterList").set<double>("relaxation: damping factor",1.0);
  // Teuchos::ParameterList myParamsAux = GetParams();
  // myParamsAux.setParametersNotAlreadySet(defaults);
  // SetParams(myParamsAux);

  // Some checks
  std::string ifpack_type = get_params().get<std::string>("type", "none");
  if (ifpack_type == "none")
    FOUR_C_THROW("The ifpack type has to be given for the ifpack smoother");

  if (not(get_params().isSublist("ParameterList")))
    FOUR_C_THROW("The ifpack ParameterList has to be provided");

  // Some output
  if (get_verbosity() == "on")
  {
    std::cout << std::endl;
    std::cout << "Creating an IFPACK smoother for block " << get_block() << " at level "
              << get_level() << std::endl;

    std::cout << "The Ifpack type is: " << get_params().get<std::string>("type") << std::endl;
    int overlap = get_params().get<int>("overlap", 0);
    std::cout << "The overlap is: " << overlap << std::endl;
    std::cout << "The parameters are: " << std::endl;
    std::cout << get_params().sublist("ParameterList");
    // std::cout << std::endl;
  }


  // Build the smoother
  if (not get_operator()->has_only_one_block())
    FOUR_C_THROW("This smoother can be built only for single block matrices");
  Teuchos::RCP<Core::LinAlg::SparseMatrix> Op = get_operator()->get_matrix(0, 0);
  if (Op == Teuchos::null) FOUR_C_THROW("I dont want a null pointer here");
  Teuchos::ParameterList myParams = get_params();
  return Teuchos::make_rcp<IfpackWrapper>(Op, myParams);
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::GenericSmoother>
Core::LinearSolver::AMGNxN::MueluSmootherWrapperFactory::create()
{
  // Check input
  if (not is_set_level()) FOUR_C_THROW("IsSetLevel() returns false");
  if (not is_set_block()) FOUR_C_THROW("IsSetBlock() returns false");
  if (not is_set_hierarchies()) FOUR_C_THROW("IsSetHierarchies() returns false");

  if (get_verbosity() == "on")
  {
    std::cout << std::endl;
    std::cout << "Creating a REUSE_MUELU_SMOOTHER smoother for block " << get_block();
    std::cout << " at level " << get_level() << std::endl;
  }
  return get_hierarchies()->get_s_pre(get_block(), get_level());
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::GenericSmoother>
Core::LinearSolver::AMGNxN::MueluAMGWrapperFactory::create()
{
  // Expected parameters (example)
  // <ParameterList name="parameters">
  //   <Parameter name="xml file"      type="string"  value="myfile.xml"/>
  // </ParameterList>
  //
  // TODO or
  //
  // <ParameterList name="parameters">
  //   <Parameter name="parameter list"      type="string"  value="NameOfTheParameterList"/>
  // </ParameterList>
  //  In that case we look in GetParamsSmoother() for a list called "NameOfTheParameterList"
  //  which has to contain all the parameters defining a muelu hierarchy
  //
  // TODO or
  //
  // <ParameterList name="parameters">
  //  ... ... list defining the muelue hierarchy (i.e.) the contents of the xml file
  // </ParameterList>
  //
  //
  // Priority: first "xml file", then "parameter list", then other:
  // If the parameter "xml file" is found, then all other parameters are ignored
  // else, if "parameter list is found", then other parameters are ignored

  // Check input
  if (not is_set_level()) FOUR_C_THROW("IsSetLevel() returns false");
  if (not is_set_operator()) FOUR_C_THROW("IsSetOperator() returns false");
  if (not is_set_block()) FOUR_C_THROW("IsSetBlock() returns false");
  if (not is_set_hierarchies()) FOUR_C_THROW("IsSetHierarchies() returns false");
  if (not is_set_params()) FOUR_C_THROW("IsSetParams() returns false");
  if (not is_set_null_space()) FOUR_C_THROW("IsSetNullSpace() returns false");
  if (not is_set_params_smoother()) FOUR_C_THROW("IsSetSmoothersParams() returns false");

  if (get_verbosity() == "on")
  {
    std::cout << std::endl;
    std::cout << "Creating a NEW_MUELU_AMG smoother for block " << get_block();
    std::cout << " at level " << get_level() << std::endl;
  }

  Teuchos::ParameterList myList;
  std::string xml_filename = get_params().get<std::string>("xml file", "none");
  std::string list_name = get_params().get<std::string>("parameter list", "none");
  if (xml_filename != "none")
  {
    // If the xml file is not an absolute path, make it relative wrt the main xml file
    if ((xml_filename)[0] != '/')
    {
      std::string tmp = get_params_smoother().get<std::string>("main xml path", "none");
      if (tmp == "none") FOUR_C_THROW("Path of the main xml not found");
      xml_filename.insert(xml_filename.begin(), tmp.begin(), tmp.end());
    }

    Teuchos::updateParametersFromXmlFile(
        xml_filename, Teuchos::Ptr<Teuchos::ParameterList>(&myList));

    if (get_verbosity() == "on")
    {
      std::cout << "The chosen parameters are:" << std::endl;
      std::cout << "xml file = : " << xml_filename << std::endl;
    }
  }
  else if (list_name != "none")
  {
    myList = get_params_smoother().sublist(list_name);
    if (get_verbosity() == "on")
    {
      std::cout << "The chosen parameters are:" << std::endl;
      std::cout << "parameter list = : " << list_name << std::endl;
    }
  }
  else
    myList = get_params();



  // TODO now we use the null space generated by 4C, which only makes sense for the finest level.
  // We can obtain null spaces for other levels from inside the muelu hierarchies.
  if (get_level() != 0)
  {
    FOUR_C_THROW(
        "Trying to create a NEW_MUELU_AMG smoother at a level > 0. Sorry, but this is not possible "
        "yet.");
  }

  // Recover info
  if (not get_operator()->has_only_one_block())
    FOUR_C_THROW("This smoother can be built only for single block matrices");
  Teuchos::RCP<Core::LinAlg::SparseMatrix> Op2 = get_operator()->get_matrix(0, 0);
  if (Op2 == Teuchos::null) FOUR_C_THROW("I dont want a null pointer here");
  int num_pde = get_null_space().get_num_pd_es();
  int null_space_dim = get_null_space().get_null_space_dim();
  auto null_space_data = get_null_space().get_null_space_data();

  Teuchos::RCP<MueluAMGWrapper> PtrOut =
      Teuchos::make_rcp<MueluAMGWrapper>(Op2, num_pde, null_space_dim, null_space_data, myList);
  PtrOut->setup();

  return Teuchos::rcp_dynamic_cast<Core::LinearSolver::AMGNxN::GenericSmoother>(PtrOut);
}



/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::GenericSmoother>
Core::LinearSolver::AMGNxN::SingleFieldAMGFactory::create()
{
  // Expected parameters (example)
  // <ParameterList name="parameters">
  //   <Parameter name="xml file"          type="string"  value="myfile.xml"/>
  //   <Parameter name="fine smoother"     type="string"  value="myfinesmoother"/>
  // </ParameterList>
  //
  //
  //
  // <ParameterList name="myfinesmoother">
  //   <Parameter name="type"                           type="string"  value="point relaxation"/>
  //   <ParameterList name="ParameterList">
  //     <Parameter name="relaxation: type"             type="string"  value="Gauss-Seidel"/>
  //     <Parameter name="relaxation: backward mode"    type="bool"    value="false"/>
  //     <Parameter name="relaxation: sweeps"           type="int"     value="2"/>
  //     <Parameter name="relaxation: damping factor"   type="double"  value="1.0"/>
  //   </ParameterList>
  // </ParameterList>

  // Check input
  if (not is_set_level()) FOUR_C_THROW("IsSetLevel() returns false");
  if (not is_set_operator()) FOUR_C_THROW("IsSetOperator() returns false");
  if (not is_set_block()) FOUR_C_THROW("IsSetBlock() returns false");
  if (not is_set_hierarchies()) FOUR_C_THROW("IsSetHierarchies() returns false");
  if (not is_set_params()) FOUR_C_THROW("IsSetParams() returns false");
  if (not is_set_null_space()) FOUR_C_THROW("IsSetNullSpace() returns false");
  if (not is_set_params_smoother()) FOUR_C_THROW("IsSetSmoothersParams() returns false");


  if (get_verbosity() == "on")
  {
    std::cout << std::endl;
    std::cout << "Creating a NEW_MUELU_AMG_IFPACK_SMO smoother for block " << get_block();
    std::cout << " at level " << get_level() << std::endl;
  }

  Teuchos::ParameterList myList;
  std::string xml_filename = get_params().get<std::string>("xml file", "none");
  if (xml_filename != "none")
  {
    // If the xml file is not an absolute path, make it relative wrt the main xml file
    if ((xml_filename)[0] != '/')
    {
      std::string tmp = get_params_smoother().get<std::string>("main xml path", "none");
      if (tmp == "none") FOUR_C_THROW("Path of the main xml not found");
      xml_filename.insert(xml_filename.begin(), tmp.begin(), tmp.end());
    }


    Teuchos::updateParametersFromXmlFile(
        xml_filename, Teuchos::Ptr<Teuchos::ParameterList>(&myList));
    if (get_verbosity() == "on")
    {
      std::cout << "The chosen parameters are:" << std::endl;
      std::cout << "xml file = : " << xml_filename << std::endl;
    }
  }
  else
    FOUR_C_THROW("Not xml file name found");

  std::string fine_smoother = get_params().get<std::string>("fine smoother", "none");
  if (fine_smoother == "none") FOUR_C_THROW("You have to set: fine smoother");
  if (not get_params_smoother().isSublist(fine_smoother))
    FOUR_C_THROW("Not found a list named {}", fine_smoother);
  Teuchos::ParameterList fine_smoother_list = get_params_smoother().sublist(fine_smoother);

  if (get_verbosity() == "on")
  {
    std::cout << "fine smoother:" << std::endl;
    std::cout << "  The Ifpack type is: " << fine_smoother_list.get<std::string>("type")
              << std::endl;
    int overlap = fine_smoother_list.get<int>("overlap", 0);
    std::cout << "  The overlap is: " << overlap << std::endl;
    std::cout << "  The parameters are: " << std::endl;
    std::cout << fine_smoother_list.sublist("ParameterList");
    std::cout << "coarsest smoother:"
              << "the one you have defined in Muelu" << std::endl;
  }


  // Recover info
  if (not get_operator()->has_only_one_block())
    FOUR_C_THROW("This smoother can be built only for single block matrices");
  Teuchos::RCP<Core::LinAlg::SparseMatrix> Op2 = get_operator()->get_matrix(0, 0);
  if (Op2 == Teuchos::null) FOUR_C_THROW("I dont want a null pointer here");
  int num_pde = get_null_space().get_num_pd_es();
  int null_space_dim = get_null_space().get_null_space_dim();
  auto null_space_data = get_null_space().get_null_space_data();

  return Teuchos::make_rcp<SingleFieldAMG>(
      Op2, num_pde, null_space_dim, null_space_data, myList, fine_smoother_list);
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::GenericSmoother>
Core::LinearSolver::AMGNxN::HierarchyRemainderWrapperFactory::create()
{
  // Check input
  if (not is_set_level()) FOUR_C_THROW("IsSetLevel() returns false");
  if (not is_set_block()) FOUR_C_THROW("IsSetBlock() returns false");
  if (not is_set_hierarchies()) FOUR_C_THROW("IsSetHierarchies() returns false");

  if (get_verbosity() == "on")
  {
    std::cout << std::endl;
    std::cout << "Creating a REUSE_MUELU_AMG smoother for block " << get_block();
    std::cout << " at level " << get_level() << std::endl;
  }

  // Pick up info and cast
  int NumLevels = get_hierarchies()->get_num_levels(get_block());
  std::vector<Teuchos::RCP<BlockedMatrix>> Avec(NumLevels, Teuchos::null);
  std::vector<Teuchos::RCP<BlockedMatrix>> Pvec(NumLevels - 1, Teuchos::null);
  std::vector<Teuchos::RCP<BlockedMatrix>> Rvec(NumLevels - 1, Teuchos::null);
  std::vector<Teuchos::RCP<GenericSmoother>> SvecPre(NumLevels, Teuchos::null);
  std::vector<Teuchos::RCP<GenericSmoother>> SvecPos(NumLevels - 1, Teuchos::null);
  for (int level = 0; level < NumLevels; level++)
  {
    Avec[level] = Teuchos::make_rcp<BlockedMatrix>(1, 1);
    Avec[level]->set_matrix(get_hierarchies()->get_a(get_block(), level), 0, 0);
    SvecPre[level] = get_hierarchies()->get_s_pre(get_block(), level);
  }
  for (int level = 0; level < (NumLevels - 1); level++)
  {
    Pvec[level] = Teuchos::make_rcp<BlockedMatrix>(1, 1);
    Pvec[level]->set_matrix(get_hierarchies()->get_p(get_block(), level), 0, 0);
    Rvec[level] = Teuchos::make_rcp<BlockedMatrix>(1, 1);
    Rvec[level]->set_matrix(get_hierarchies()->get_r(get_block(), level), 0, 0);
    SvecPos[level] = get_hierarchies()->get_s_pos(get_block(), level);
  }

  // Construct the V cycle
  int NumSweeps = 1;  // Hard coded
  Teuchos::RCP<Vcycle> V = Teuchos::make_rcp<Vcycle>(NumLevels, NumSweeps, get_level());
  V->set_operators(Avec);
  V->set_projectors(Pvec);
  V->set_restrictors(Rvec);
  V->set_pre_smoothers(SvecPre);
  V->set_pos_smoothers(SvecPos);


  return V;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::GenericSmoother>
Core::LinearSolver::AMGNxN::MergeAndSolveFactory::create()
{
  // Check input
  if (not is_set_operator()) FOUR_C_THROW("IsSetOperator() returns false");

  if (get_verbosity() == "on")
  {
    std::cout << std::endl;
    std::cout << "Creating a MERGE_AND_SOLVE smoother for blocks (";
    for (size_t i = 0; i < get_blocks().size(); i++)
    {
      std::cout << get_blocks()[i];
      if (i < get_blocks().size() - 1)
        std::cout << ", ";
      else
        std::cout << ") ";
    }
    std::cout << " at level " << get_level() << std::endl;
  }

  Teuchos::RCP<MergeAndSolve> S = Teuchos::make_rcp<MergeAndSolve>();
  Teuchos::RCP<BlockedMatrix> matrix = get_operator();
  if (matrix == Teuchos::null) FOUR_C_THROW("We expect here a block sparse matrix");
  S->setup(*matrix);

  return S;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::GenericSmoother>
Core::LinearSolver::AMGNxN::CoupledAmgFactory::create()
{
  //<ParameterList name="parameters">
  //
  //  <Parameter name="number of levels"                 type="int"  value="..."/>
  //
  //  <Parameter name="smoother: all but coarsest level" type="string"  value="myFinestSmoother"/>
  //
  //  <Parameter name="smoother: coarsest level"         type="string"  value="myCoarsestSmoother"/>
  //
  //  <Parameter name="verbosity"                        type="string"  value="on"/>
  //
  //  <Parameter name="muelu parameters for block 0"       type="string"  value="myMuelu0"/>
  //
  //  <Parameter name="muelu parameters for block 1"       type="string"  value="myMuelu1"/>
  //
  //   ....
  //
  //  <Parameter name="muelu parameters for block N"       type="string"  value="myMueluN"/>
  //
  //</ParameterList>
  // WARNING: here the blocks are in local numeration of the submatrix passed to this factory


  // Recover the null space info
  unsigned nBlocks = get_blocks().size();
  const std::vector<int>& Blocks = get_blocks();
  int b = 0;
  std::vector<int> num_pdes(nBlocks, 0);
  std::vector<int> null_spaces_dim(nBlocks, 0);
  std::vector<std::shared_ptr<std::vector<double>>> null_spaces_data(nBlocks);
  for (unsigned i = 0; i < nBlocks; i++)
  {
    b = Blocks[i];
    num_pdes[i] = get_null_space_all_blocks()[b].get_num_pd_es();
    null_spaces_dim[i] = get_null_space_all_blocks()[b].get_null_space_dim();
    null_spaces_data[i] = get_null_space_all_blocks()[b].get_null_space_data();
  }

  // Recover the lists
  const Teuchos::ParameterList& amgnxn_params = get_params();
  const Teuchos::ParameterList& smoothers_params = get_params_smoother();


  return Teuchos::make_rcp<CoupledAmg>(get_operator(), num_pdes, null_spaces_dim, null_spaces_data,
      amgnxn_params, smoothers_params, smoothers_params);
}



/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::GenericSmoother>
Core::LinearSolver::AMGNxN::BgsSmootherFactory::create()
{
  // Expected parameters (example)
  // <ParameterList name="parameters">
  //   <Parameter name="blocks"      type="string"  value="(1,2),(3,4),(5)"/>
  //   <Parameter name="smoothers"   type="string"  value="myBGS,mySIMPLE,IFPACK"/>
  //   <Parameter name="sweeps"      type="int"     value="3"/>
  //   <Parameter name="omega"       type="double"  value="1.0"/>
  //   <Parameter name="local sweeps"  type="string"     value="3,2,4"/>
  //   <Parameter name="local omegas"  type="string"  value="1.0,1.2,0.9"/>
  // </ParameterList>

  // TODO Check that all required data is set

  // =============================================================
  // Parse parameters
  // =============================================================

  // determine how the blocks are grouped
  std::string blocks_string = get_params().get<std::string>("blocks", "none");
  std::vector<std::vector<int>> SuperBlocks2Blocks;
  std::vector<std::vector<int>> SuperBlocks2BlocksLocal;
  get_operator()->parse_blocks(
      blocks_string, get_blocks(), SuperBlocks2Blocks, SuperBlocks2BlocksLocal);

  // std::cout << "======================" << std::endl;
  // for(size_t i=0;i<SuperBlocks2Blocks.size();i++)
  // {
  //   for(size_t j=0;j<SuperBlocks2Blocks[i].size();j++)
  //     std::cout << SuperBlocks2Blocks[i][j] << ", ";
  //   std::cout << std::endl;
  // }


  // Determine the subsolver names
  std::string smoothers_string = get_params().get<std::string>("smoothers", "none");
  std::vector<std::string> SubSolverNames;
  parse_smoother_names(smoothers_string, SubSolverNames, SuperBlocks2Blocks);

  // sweeps and damping
  unsigned iter = static_cast<unsigned>(get_params().get<int>("sweeps", 1));
  double omega = get_params().get<double>("omega", 1.0);
  std::string local_sweeps = get_params().get<std::string>("local sweeps", "none");
  std::string local_omegas = get_params().get<std::string>("local omegas", "none");
  unsigned NumSuperBlocks = SuperBlocks2Blocks.size();
  std::vector<double> omegas(NumSuperBlocks, 1.0);
  std::vector<unsigned> iters(NumSuperBlocks, 1);
  if (local_sweeps != "none")
  {
    std::istringstream ss(local_sweeps);
    std::string token;
    unsigned ib = 0;
    while (std::getline(ss, token, ','))
    {
      if (ib >= NumSuperBlocks) FOUR_C_THROW("too many comas in {}", local_sweeps);
      iters[ib++] = atoi(token.c_str());
    }
    if (ib < NumSuperBlocks) FOUR_C_THROW("too less comas in {}", local_sweeps);
  }
  if (local_omegas != "none")
  {
    std::istringstream ss(local_omegas);
    std::string token;
    unsigned ib = 0;
    while (std::getline(ss, token, ','))
    {
      if (ib >= NumSuperBlocks) FOUR_C_THROW("too many comas in {}", local_omegas);
      omegas[ib++] = atof(token.c_str());
    }
    if (ib < NumSuperBlocks) FOUR_C_THROW("too less comas in {}", local_omegas);
  }



  // =============================================================
  // Some output
  // =============================================================
  if (get_verbosity() == "on")
  {
    std::cout << std::endl;
    std::cout << "Creating a BGS smoother for blocks (";
    for (size_t i = 0; i < get_blocks().size(); i++)
    {
      std::cout << get_blocks()[i];
      if (i < get_blocks().size() - 1)
        std::cout << ", ";
      else
        std::cout << ") ";
    }
    std::cout << " at level " << get_level() << std::endl;
    std::cout << "The chosen parameters are" << std::endl;
    std::cout << "blocks = ";
    for (size_t k = 0; k < SuperBlocks2Blocks.size(); k++)
    {
      std::cout << "(";
      for (size_t j = 0; j < SuperBlocks2Blocks[k].size(); j++)
      {
        std::cout << SuperBlocks2Blocks[k][j];
        if (j < (SuperBlocks2Blocks[k].size() - 1)) std::cout << ",";
      }
      if (k < (SuperBlocks2Blocks.size() - 1))
        std::cout << "),";
      else
        std::cout << ")" << std::endl;
    }
    std::cout << "smoothers = ";
    for (size_t k = 0; k < SubSolverNames.size(); k++)
    {
      std::cout << SubSolverNames[k];
      if (k < (SubSolverNames.size() - 1))
        std::cout << ",";
      else
        std::cout << std::endl;
    }
    std::cout << "sweeps = " << iter << std::endl;
    std::cout << "omega = " << omega << std::endl;
    std::cout << "local sweeps = ";
    for (int i : iters) std::cout << i << ",";
    std::cout << std::endl;
    std::cout << "local omegas = ";
    for (double o : omegas) std::cout << o << ",";
    std::cout << std::endl;
    // std::cout << std::endl;
  }


  // =============================================================
  // Construct smoothers for diagonal superblocks
  // =============================================================

  std::vector<Teuchos::RCP<GenericSmoother>> SubSmoothers(NumSuperBlocks, Teuchos::null);
  for (unsigned scol = 0; scol < NumSuperBlocks; scol++)
  {
    SmootherFactory mySmootherCreator;
    mySmootherCreator.set_smoother_name(SubSolverNames[scol]);
    mySmootherCreator.set_params_smoother(get_params_smoother());
    mySmootherCreator.set_hierarchies(get_hierarchies());
    mySmootherCreator.set_level(get_level());
    const std::vector<int>& scols = SuperBlocks2BlocksLocal[scol];
    mySmootherCreator.set_operator(get_operator()->get_blocked_matrix_rcp(scols, scols));
    mySmootherCreator.set_verbosity(get_verbosity());
    if (SuperBlocks2Blocks[scol].size() == 1)
    {
      int thisblock = SuperBlocks2Blocks[scol][0];
      mySmootherCreator.set_block(thisblock);
      mySmootherCreator.set_null_space(get_null_space_all_blocks()[thisblock]);
    }
    else
    {
      mySmootherCreator.set_blocks(SuperBlocks2Blocks[scol]);
      mySmootherCreator.set_null_space_all_blocks(get_null_space_all_blocks());
    }

    SubSmoothers[scol] = mySmootherCreator.create();
  }

  // =============================================================
  // Construct BGS smoother
  // =============================================================

  return Teuchos::make_rcp<BgsSmoother>(
      get_operator(), SubSmoothers, SuperBlocks2BlocksLocal, iter, omega, iters, omegas);
}



/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::BgsSmootherFactory::parse_smoother_names(
    const std::string& smoothers_string, std::vector<std::string>& smoothers_vector,
    std::vector<std::vector<int>> superblocks)
{
  if (smoothers_string == "none")
  {
    unsigned NumSuperBlocks = superblocks.size();
    smoothers_vector.resize(0);
    for (unsigned i = 0; i < NumSuperBlocks; i++)
    {
      if (0 == (superblocks[i].size()))
        FOUR_C_THROW("Something wrong related with how the blocks are set in your xml file");
      else if (1 == (superblocks[i].size()))
        smoothers_vector.emplace_back("IFPACK");
      else
        smoothers_vector.emplace_back("BGS");
    }
  }
  else
  {
    smoothers_vector.resize(0);
    std::string buf = "";
    for (char i : smoothers_string)
    {
      std::string ch(1, i);
      if (ch == ",")
      {
        smoothers_vector.push_back(buf);
        buf = "";
      }
      else
        buf += ch;
    }
    if (not(buf == "")) smoothers_vector.push_back(buf);
    buf = "";
  }

  if (smoothers_vector.size() != superblocks.size())
    FOUR_C_THROW("Not given enough subsmoothers! Fix your xml file.");
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::GenericSmoother>
Core::LinearSolver::AMGNxN::SimpleSmootherFactory::create()
{
  // Expected parameters (example)
  // <ParameterList name="parameters">
  //   <Parameter name="predictor block"     type="string"  value="(1,2)"/>
  //   <Parameter name="predictor smoother"  type="string"  value="BGS"/>
  //   <Parameter name="predictor inverse"   type="string"  value="diagonal"/>
  //   <Parameter name="predictor inverse"   type="string"  value="row sums"/>
  //   <Parameter name="predictor inverse"   type="string"  value="row sums diagonal blocks"/>
  //   <Parameter name="schur block"         type="string"  value="(3)"/>
  //   <Parameter name="schur smoother"      type="string"  value="IFPACK"/>
  //   <Parameter name="correction"          type="string"  value="smoother"/>
  //   <Parameter name="correction"          type="string"  value="approximated inverse"/>
  //   <Parameter name="sweeps"              type="int"     value="3"/>
  //   <Parameter name="alpha"               type="double"  value="1.0"/>
  //     <!-- Damping of the "pressure" correction-->
  //   <Parameter name="beta"                type="double"  value="1.0"/>
  //     <!-- Coefficient that multiplies the approximate inverse of the predictor block
  //     in the Schur complement matrixCoefficient that multiplies the approximate inverse
  //     of the predictor block in the Schur complement matrix, i.e.
  //     S = A_22 - beta*A21*A11inv*A12-->
  // </ParameterList>

  // TODO Check that all required data is set

  // =============================================================
  // Parse parameters
  // =============================================================

  // determine how the blocks are grouped
  std::string predictor_block_string = get_params().get<std::string>("predictor block", "none");
  std::string schur_block_string = get_params().get<std::string>("schur block", "none");
  if (predictor_block_string == "none")
    FOUR_C_THROW(
        "The field \"predictor block\" is mandatory for the SIMPLE smoother.Fix your xml file");
  if (schur_block_string == "none")
    FOUR_C_THROW(
        "The field \"schur block\" is mandatory for the SIMPLE smoother. Fix your xml file");
  std::string blocks_string = predictor_block_string + "," + schur_block_string;
  std::vector<std::vector<int>> SuperBlocks2Blocks;
  std::vector<std::vector<int>> SuperBlocks2BlocksLocal;
  get_operator()->parse_blocks(
      blocks_string, get_blocks(), SuperBlocks2Blocks, SuperBlocks2BlocksLocal);
  int pred = 0;
  int schur = 1;


  // Smoother names
  std::string predictor_smoother = get_params().get<std::string>("predictor smoother", "none");
  std::string schur_smoother = get_params().get<std::string>("schur smoother", "none");
  if (predictor_smoother == "none")
    FOUR_C_THROW(
        "The field \"predictor smoother\" is mandatory for the SIMPLE smoother. Fix your xml file");
  if (schur_smoother == "none")
    FOUR_C_THROW(
        "The field \"schur smoother\" is mandatory for the SIMPLE smoother. Fix your xml file");
  if (schur_smoother == "REUSE_MUELU_SMOOTHER" or schur_smoother == "REUSE_MUELU_AMG")
    FOUR_C_THROW(
        "Invalid smoother for the schur block. We cannot reuse the smoothers generated by Muelu.");

  // other params
  int iter = get_params().get<int>("sweeps", 1);
  double alpha = get_params().get<double>("alpha", 1.0);
  double beta = get_params().get<double>("beta", 1.0);
  std::string inverse_method =
      get_params().get<std::string>("predictor inverse", "row sums diagonal blocks");
  // std::string correction = GetParams().get<std::string>("correction","approximated inverse");

  // =============================================================
  // Some output
  // =============================================================

  if (get_verbosity() == "on")
  {
    std::cout << std::endl;
    std::cout << "Creating a SIMPLE smoother for blocks (";
    for (size_t i = 0; i < get_blocks().size(); i++)
    {
      std::cout << get_blocks()[i];
      if (i < get_blocks().size() - 1)
        std::cout << ", ";
      else
        std::cout << ") ";
    }
    std::cout << " at level " << get_level() << std::endl;
    std::cout << "The chosen parameters are" << std::endl;
    std::cout << "predictor block = ";
    std::cout << "(";
    for (size_t j = 0; j < SuperBlocks2Blocks[pred].size(); j++)
    {
      std::cout << SuperBlocks2Blocks[pred][j];
      if (j < (SuperBlocks2Blocks[pred].size() - 1)) std::cout << ",";
    }
    std::cout << ")" << std::endl;
    std::cout << "predictor smoother = " << predictor_smoother << std::endl;
    std::cout << "predictor inverse = " << inverse_method << std::endl;
    std::cout << "schur block = ";
    std::cout << "(";
    for (size_t j = 0; j < SuperBlocks2Blocks[schur].size(); j++)
    {
      std::cout << SuperBlocks2Blocks[schur][j];
      if (j < (SuperBlocks2Blocks[schur].size() - 1)) std::cout << ",";
    }
    std::cout << ")" << std::endl;
    std::cout << "schur smoother = " << schur_smoother << std::endl;
    std::cout << "sweeps = " << iter << std::endl;
    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "beta = " << beta << std::endl;
    // std::cout << std::endl;
  }

  // =============================================================
  // Rearrange blocks
  // =============================================================

  if (get_operator()->has_only_one_block()) FOUR_C_THROW("I expect a block matrix here");

  const std::vector<int>& pred_vec = SuperBlocks2BlocksLocal[pred];
  const std::vector<int>& schur_vec = SuperBlocks2BlocksLocal[schur];

  Teuchos::RCP<BlockedMatrix> App = get_operator()->get_blocked_matrix_rcp(pred_vec, pred_vec);
  Teuchos::RCP<BlockedMatrix> Aps = get_operator()->get_blocked_matrix_rcp(pred_vec, schur_vec);
  Teuchos::RCP<BlockedMatrix> Asp = get_operator()->get_blocked_matrix_rcp(schur_vec, pred_vec);
  Teuchos::RCP<BlockedMatrix> Ass = get_operator()->get_blocked_matrix_rcp(schur_vec, schur_vec);

  //{
  //  Core::LinAlg::Map myRange  = App->OperatorRangeMap();
  //  Core::LinAlg::Map myDomain = App->OperatorDomainMap();
  //  std::cout << "Matrix App" << std::endl;
  //  std::cout << "   Range   MinAllGID = " << myRange.MinAllGID()  << std::endl;
  //  std::cout << "   Range   MaxAllGID = " << myRange.MaxAllGID()  << std::endl;
  //  std::cout << "   Domain  MinAllGID = " << myDomain.MinAllGID()  << std::endl;
  //  std::cout << "   Domain  MaxAllGID = " << myDomain.MaxAllGID()  << std::endl;
  //}
  //{
  //  Core::LinAlg::Map myRange  = Ass->OperatorRangeMap();
  //  Core::LinAlg::Map myDomain = Ass->OperatorDomainMap();
  //  std::cout << "Matrix App" << std::endl;
  //  std::cout << "   Range   MinAllGID = " << myRange.MinAllGID()  << std::endl;
  //  std::cout << "   Range   MaxAllGID = " << myRange.MaxAllGID()  << std::endl;
  //  std::cout << "   Domain  MinAllGID = " << myDomain.MinAllGID()  << std::endl;
  //  std::cout << "   Domain  MaxAllGID = " << myDomain.MaxAllGID()  << std::endl;
  //}


  // =============================================================
  // Approximate the schur complement
  // =============================================================

  // Approximate the inverse of App
  Teuchos::RCP<BlockedMatrix> invApp = Teuchos::null;
  {
    int num_rows = App->get_num_rows();
    if (num_rows != App->get_num_cols()) FOUR_C_THROW("We spect here a square matrix");
    invApp = Teuchos::make_rcp<DiagonalBlockedMatrix>(num_rows);
    for (int i = 0; i < num_rows; i++)
      invApp->set_matrix(approximate_inverse(*(App->get_matrix(i, i)), inverse_method), i, i);
  }

  // Compute the schur complement
  Teuchos::RCP<BlockedMatrix> S = compute_schur_complement(*invApp, *Aps, *Asp, *Ass);

  // =============================================================
  // Construct smoothers for diagonal superblocks
  // =============================================================

  SmootherFactory mySmootherCreator;
  mySmootherCreator.set_params_smoother(get_params_smoother());
  mySmootherCreator.set_hierarchies(get_hierarchies());
  mySmootherCreator.set_level(get_level());
  mySmootherCreator.set_verbosity(get_verbosity());

  // For predictor
  if (SuperBlocks2Blocks[pred].size() == 1)
  {
    int thisblock = SuperBlocks2Blocks[pred][0];
    mySmootherCreator.set_block(thisblock);
    mySmootherCreator.set_null_space(get_null_space_all_blocks()[thisblock]);
  }
  else
  {
    mySmootherCreator.set_blocks(SuperBlocks2Blocks[pred]);
    mySmootherCreator.set_null_space_all_blocks(get_null_space_all_blocks());
  }
  mySmootherCreator.set_smoother_name(predictor_smoother);
  mySmootherCreator.set_operator(App);
  Teuchos::RCP<GenericSmoother> Smoother_App = mySmootherCreator.create();

  // For schur
  if (SuperBlocks2Blocks[schur].size() == 1)
  {
    int thisblock = SuperBlocks2Blocks[schur][0];
    mySmootherCreator.set_block(thisblock);
    mySmootherCreator.set_null_space(get_null_space_all_blocks()[thisblock]);
  }
  else
  {
    mySmootherCreator.set_blocks(SuperBlocks2Blocks[schur]);
    mySmootherCreator.set_null_space_all_blocks(get_null_space_all_blocks());
  }

  mySmootherCreator.set_smoother_name(schur_smoother);
  mySmootherCreator.set_operator(S);
  Teuchos::RCP<GenericSmoother> Smoother_S = mySmootherCreator.create();



  // =============================================================
  // Construct SIMPLE smoother
  // =============================================================

  return Teuchos::make_rcp<SimpleSmoother>(
      get_operator(), invApp, S, Smoother_App, Smoother_S, pred_vec, schur_vec, iter, alpha);
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
Teuchos::RCP<Core::LinearSolver::AMGNxN::BlockedMatrix>
Core::LinearSolver::AMGNxN::SimpleSmootherFactory::compute_schur_complement(
    const BlockedMatrix& invApp, const BlockedMatrix& Aps, const BlockedMatrix& Asp,
    const BlockedMatrix& Ass)
{
  Teuchos::RCP<BlockedMatrix> Sout = Teuchos::null;

  if (invApp.has_only_one_block())
  {
    if (Ass.has_only_one_block())
    {
      Teuchos::RCP<Core::LinAlg::SparseMatrix> invApp_sp = invApp.get_matrix(0, 0);
      Teuchos::RCP<Core::LinAlg::SparseMatrix> Asp_sp = Asp.get_matrix(0, 0);
      Teuchos::RCP<Core::LinAlg::SparseMatrix> Aps_sp = Aps.get_matrix(0, 0);
      Teuchos::RCP<Core::LinAlg::SparseMatrix> Ass_sp = Ass.get_matrix(0, 0);
      auto temp =
          Core::LinAlg::matrix_multiply(*Asp_sp, false, *invApp_sp, false, false, false, true);
      auto S_sp = Core::LinAlg::matrix_multiply(*temp, false, *Aps_sp, false, false, false, false);
      S_sp->add(*Ass_sp, false, 1.0, -1.0);
      S_sp->complete();
      Sout = Teuchos::make_rcp<BlockedMatrix>(1, 1);
      Sout->set_matrix(Teuchos::RCP(S_sp.release()), 0, 0);
      return Sout;
    }
    else if (not Ass.has_only_one_block())
    {
      FOUR_C_THROW("TODO: Branch not implemented yet");
    }
    else
      FOUR_C_THROW("Something went wrong");
  }
  else if (not invApp.has_only_one_block())
  {
    if (Ass.has_only_one_block())
    {
      int NumBlocks_pp = invApp.get_num_rows();
      Teuchos::RCP<Core::LinAlg::SparseMatrix> S_sp = Teuchos::null;
      for (int b = 0; b < NumBlocks_pp; b++)
      {
        auto temp = Core::LinAlg::matrix_multiply(
            *(Asp.get_matrix(0, b)), false, *(invApp.get_matrix(b, b)), false, true);
        if (b == 0)
          S_sp = Teuchos::RCP(
              Core::LinAlg::matrix_multiply(*temp, false, *(Aps.get_matrix(b, 0)), false, false)
                  .release());
        else
        {
          auto S_sp_tmp =
              Core::LinAlg::matrix_multiply(*temp, false, *(Aps.get_matrix(b, 0)), false, true);
          S_sp->add(*S_sp_tmp, false, 1.0, 1.0);
        }
      }
      Teuchos::RCP<Core::LinAlg::SparseMatrix> Ass_sp = Ass.get_matrix(0, 0);
      S_sp->add(*Ass_sp, false, 1.0, -1.0);
      S_sp->complete();
      Sout = Teuchos::make_rcp<BlockedMatrix>(1, 1);
      Sout->set_matrix(S_sp, 0, 0);
      return Sout;
    }
    else if (not Ass.has_only_one_block())
    {
      FOUR_C_THROW("TODO: Branch not implemented yet");
    }
    else
      FOUR_C_THROW("Something went wrong");
  }
  else
    FOUR_C_THROW("Something went wrong");


  return Sout;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinAlg::SparseMatrix>
Core::LinearSolver::AMGNxN::SimpleSmootherFactory::approximate_inverse(
    const Core::LinAlg::SparseMatrix& A, const std::string& method)
{
  Core::LinAlg::Vector<double> invAVector(A.row_map());
  if (method == "diagonal")
  {
    A.extract_diagonal_copy(invAVector);
    int err = invAVector.reciprocal(invAVector);
    if (err)
      FOUR_C_THROW(
          "Core::LinAlg::MultiVector<double>::Reciprocal returned {}, are we dividing by 0?", err);
  }
  else if (method == "row sums" or method == "row sums diagonal blocks")
  {
    int err = A.inv_row_sums(invAVector);
    if (err) FOUR_C_THROW("inv_row_sums() returned {}, are we dividing by 0?", err);
  }
  else
    FOUR_C_THROW("Invalid value for \"predictor inverse\". Fix your xml file.");
  Teuchos::RCP<Core::LinAlg::SparseMatrix> S =
      Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(invAVector);
  S->complete(A.row_map(), A.row_map());
  return S;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::GenericSmoother>
Core::LinearSolver::AMGNxN::DirectSolverWrapperFactory::create()
{
  // Check input
  if (not is_set_operator()) FOUR_C_THROW("IsSetOperator() returns false");

  if (get_verbosity() == "on")
  {
    std::cout << std::endl;
    std::cout << "Creating a DIRECT_SOLVER for block " << get_block();
    std::cout << " at level " << get_level() << std::endl;
  }

  Teuchos::RCP<DirectSolverWrapper> S = Teuchos::make_rcp<DirectSolverWrapper>();
  if (not get_operator()->has_only_one_block())
    FOUR_C_THROW("We spect here a matrix with only one block");
  Teuchos::RCP<Core::LinAlg::SparseMatrix> matrix = get_operator()->get_matrix(0, 0);
  if (matrix == Teuchos::null) FOUR_C_THROW("We expect here a sparse matrix");
  S->setup(matrix, Teuchos::make_rcp<Teuchos::ParameterList>(get_params()));


  return S;
}

FOUR_C_NAMESPACE_CLOSE
