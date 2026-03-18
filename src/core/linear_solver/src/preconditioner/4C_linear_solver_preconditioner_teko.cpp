// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_preconditioner_teko.hpp"

#include "4C_comm_utils.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_linear_solver_thyra_utils.hpp"

#include <Stratimikos_DefaultLinearSolverBuilder.hpp>
#include <Stratimikos_MueLuHelpers.hpp>
#include <Teko_EpetraInverseOpWrapper.hpp>
#include <Teko_GaussSeidelPreconditionerFactory.hpp>
#include <Teko_InverseLibrary.hpp>
#include <Teko_LU2x2PreconditionerFactory.hpp>
#include <Teko_StratimikosFactory.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Xpetra_MultiVectorFactory.hpp>

#include <filesystem>

FOUR_C_NAMESPACE_OPEN

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Core::LinearSolver::TekoPreconditioner::TekoPreconditioner(Teuchos::ParameterList& tekolist)
    : tekolist_(tekolist)
{
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::TekoPreconditioner::setup(
    Core::LinAlg::SparseOperator& matrix, Core::LinAlg::MultiVector<double>& b)
{
  using EpetraMultiVector = Xpetra::EpetraMultiVectorT<GlobalOrdinal, Node>;
  using XpetraMultiVector = Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  if (!tekolist_.sublist("Teko Parameters").isParameter("TEKO_XML_FILE"))
    FOUR_C_THROW("TEKO_XML_FILE parameter not set!");
  auto xmlFileName = tekolist_.sublist("Teko Parameters").get<std::string>("TEKO_XML_FILE");

  Teuchos::ParameterList tekoParams;
  auto comm = Core::Communication::to_teuchos_comm<int>(matrix.get_comm());
  Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, Teuchos::Ptr(&tekoParams), *comm);

  auto A = std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(
      Core::Utils::shared_ptr_from_ref(matrix));

  // Reorder/split the linear operator into the block structure desired
  if (tekolist_.sublist("Teko Parameters").isParameter("reorder: maps"))
  {
    auto maps = tekolist_.sublist("Teko Parameters")
                    .get<std::vector<std::shared_ptr<const Core::LinAlg::Map>>>("reorder: maps");
    Core::LinAlg::MultiMapExtractor extractor;
    std::shared_ptr<Core::LinAlg::SparseMatrix> A_sparse;

    // If we have a sparse matrix at hand and given reorder maps, we try to split the linear
    // operator into a block matrix. If it is a block matrix, we first merge and split afterwards.
    if (!A)
    {
      A_sparse = std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(
          Core::Utils::shared_ptr_from_ref(matrix));
    }
    else
    {
      // TODO: How to do this properly?? If last diagonal block is zero we have a saddlepoint system
      // with the Lagrange multiplier block most likely not in the maps yet.
      if (A->matrix(A->rows() - 1, A->cols() - 1).norm_inf() < 1e-12)
      {
        maps.push_back(
            Core::Utils::shared_ptr_from_ref(A->matrix(A->rows() - 1, A->cols() - 1).domain_map()));
      }

      A_sparse = A->merge();
    }

    extractor = Core::LinAlg::MultiMapExtractor(A_sparse->row_map(), maps);
    A = Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
        *A_sparse, extractor, extractor);
    A->complete();
  }

  // wrap linear operators
  if (!A)
  {
    auto A_crs = Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(Teuchos::rcpFromRef(matrix));
    pmatrix_ = Utils::create_thyra_linear_op(*A_crs, LinAlg::DataAccess::Copy);
  }
  else
  {
    pmatrix_ = Utils::create_thyra_linear_op(*A, LinAlg::DataAccess::Copy);

    // check if multigrid is used as preconditioner for single field inverse approximation and
    // attach nullspace and coordinate information to the respective inverse parameter list.
    for (int block = 0; block < A->rows(); block++)
    {
      std::string inverse = "Inverse" + std::to_string(block + 1);

      if (tekolist_.isSublist(inverse))
      {
        // get the single field preconditioner sub-list of a matrix block hardwired under
        // "Inverse<1...n>".
        Teuchos::ParameterList& inverseList = tekolist_.sublist(inverse);

        if (tekoParams.sublist("Inverse Factory Library")
                .sublist(inverse)
                .get<std::string>("Type") == "MueLu")
        {
          const int number_of_equations = inverseList.get<int>("PDE equations");

          Teuchos::RCP<XpetraMultiVector> nullspace =
              Teuchos::make_rcp<EpetraMultiVector>(Teuchos::rcpFromRef(
                  inverseList.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace")
                      ->get_epetra_multi_vector()));

          Teuchos::RCP<XpetraMultiVector> coordinates =
              Teuchos::make_rcp<EpetraMultiVector>(Teuchos::rcpFromRef(inverseList
                      .get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("Coordinates")
                      ->get_epetra_multi_vector()));

          tekoParams.sublist("Inverse Factory Library")
              .sublist(inverse)
              .set("number of equations", number_of_equations);
          Teuchos::ParameterList& userParamList =
              tekoParams.sublist("Inverse Factory Library").sublist(inverse).sublist("user data");
          userParamList.set("Nullspace", nullspace);
          userParamList.set("Coordinates", coordinates);
        }
      }
    }
  }

  // setup preconditioner builder and enable relevant packages
  Stratimikos::LinearSolverBuilder<double> builder;

  // enable block preconditioning and multigrid
  Stratimikos::enableMueLu<Scalar, LocalOrdinal, GlobalOrdinal, Node>(builder);
  Teko::addTekoToStratimikosBuilder(builder);

  // add special in-house block preconditioning methods
  Teuchos::RCP<Teko::Cloneable> clone = Teuchos::make_rcp<Teko::AutoClone<LU2x2SpaiStrategy>>();
  Teko::LU2x2PreconditionerFactory::addStrategy("Spai Strategy", clone);

  Teuchos::RCP<Teko::Cloneable> arrowhead_clone =
      Teuchos::make_rcp<Teko::AutoClone<ArrowHeadPreconditionerFactory>>();
  Teko::PreconditionerFactory::addPreconditionerFactory(
      "Arrowhead Preconditioner", arrowhead_clone);

  // get preconditioner parameter list
  Teuchos::RCP<Teuchos::ParameterList> stratimikos_params =
      Teuchos::make_rcp<Teuchos::ParameterList>(*builder.getValidParameters());
  Teuchos::ParameterList& tekoList =
      stratimikos_params->sublist("Preconditioner Types").sublist("Teko");
  tekoList.setParameters(tekoParams);
  builder.setParameterList(stratimikos_params);

  // construct preconditioning operator
  Teuchos::RCP<Thyra::PreconditionerFactoryBase<double>> precFactory =
      builder.createPreconditioningStrategy("Teko");
  Teuchos::RCP<Thyra::PreconditionerBase<double>> prec =
      Thyra::prec<double>(*precFactory, pmatrix_);
  Teko::LinearOp inverseOp = prec->getUnspecifiedPrecOp();

  p_ = std::make_shared<Teko::Epetra::EpetraInverseOpWrapper>(inverseOp);
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
const Teko::LinearOp Core::LinearSolver::LU2x2SpaiStrategy::getHatInvA00(
    const Teko::BlockedLinearOp& A, Teko::BlockPreconditionerState& state) const
{
  initialize_state(A, state);

  return state.getModifiableOp("invA00");
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
const Teko::LinearOp Core::LinearSolver::LU2x2SpaiStrategy::getTildeInvA00(
    const Teko::BlockedLinearOp& A, Teko::BlockPreconditionerState& state) const
{
  initialize_state(A, state);

  return state.getModifiableOp("invA00");
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
const Teko::LinearOp Core::LinearSolver::LU2x2SpaiStrategy::getInvS(
    const Teko::BlockedLinearOp& A, Teko::BlockPreconditionerState& state) const
{
  initialize_state(A, state);

  return state.getModifiableOp("invS");
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::LU2x2SpaiStrategy::initialize_state(
    const Teko::BlockedLinearOp& A, Teko::BlockPreconditionerState& state) const
{
  if (state.isInitialized()) return;

  Teko::LinearOp F = Teko::getBlock(0, 0, A);
  Teko::LinearOp Bt = Teko::getBlock(0, 1, A);
  Teko::LinearOp B = Teko::getBlock(1, 0, A);
  Teko::LinearOp C = Teko::getBlock(1, 1, A);

  // build the Schur complement
  Teko::ModifiableLinearOp& S = state.getModifiableOp("S");
  {
    auto A_op = Teuchos::rcp_dynamic_cast<const Thyra::EpetraLinearOp>(F);
    auto A_crs = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(A_op->epetra_op(), true);
    const Core::LinAlg::SparseMatrix A_sparse(
        Core::Utils::shared_ptr_from_ref(*Teuchos::rcp_const_cast<Epetra_CrsMatrix>(A_crs)),
        Core::LinAlg::DataAccess::Copy);

    // sparse inverse calculation
    std::shared_ptr<Core::LinAlg::SparseMatrix> A_thresh =
        Core::LinAlg::threshold_matrix(A_sparse, drop_tol_);
    std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern_enriched =
        Core::LinAlg::enrich_matrix_graph(*A_thresh, fill_level_);
    std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse =
        Core::LinAlg::matrix_sparse_inverse(A_sparse, sparsity_pattern_enriched);
    A_thresh = Core::LinAlg::threshold_matrix(*A_inverse, drop_tol_);
    Teko::LinearOp H = Thyra::epetraLinearOp(Teuchos::rcpFromRef(A_thresh->epetra_matrix()));

    // build Schur-complement
    Teko::LinearOp HBt;
    Teko::ModifiableLinearOp& mHBt = state.getModifiableOp("HBt");
    Teko::ModifiableLinearOp& mhatS = state.getModifiableOp("hatS");
    Teko::ModifiableLinearOp& BHBt = state.getModifiableOp("BHBt");

    // build H*Bt
    mHBt = Teko::explicitMultiply(H, Bt, mHBt);
    HBt = mHBt;

    // build B*H*Bt
    BHBt = Teko::explicitMultiply(B, HBt, BHBt);

    // build C-B*H*Bt
    mhatS = Teko::explicitAdd(C, Teko::scale(-1.0, BHBt), mhatS);
    S = mhatS;
  }

  // build inverse S
  {
    Teko::ModifiableLinearOp& invS = state.getModifiableOp("invS");
    if (invS == Teuchos::null)
      invS = buildInverse(*inv_factory_s_, S);
    else
      rebuildInverse(*inv_factory_s_, S, invS);
  }

  // build inverse A00
  {
    Teko::ModifiableLinearOp& invA00 = state.getModifiableOp("invA00");
    if (invA00 == Teuchos::null)
      invA00 = buildInverse(*inv_factory_f_, F);
    else
      rebuildInverse(*inv_factory_f_, F, invA00);
  }

  state.setInitialized(true);
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::LU2x2SpaiStrategy::initializeFromParameterList(
    const Teuchos::ParameterList& lulist, const Teko::InverseLibrary& invLib)
{
  std::string invStr = "", invA00Str = "", invSStr = "";

  // "parse" the parameter list
  if (lulist.isParameter("Inverse Type")) invStr = lulist.get<std::string>("Inverse Type");
  if (lulist.isParameter("Inverse A00 Type"))
    invA00Str = lulist.get<std::string>("Inverse A00 Type");
  if (lulist.isParameter("Inverse Schur Type"))
    invSStr = lulist.get<std::string>("Inverse Schur Type");

  // Spai parameters
  if (lulist.isParameter("Drop tolerance"))
  {
    drop_tol_ = lulist.get<double>("Drop tolerance");
  }

  if (lulist.isParameter("Fill-in level"))
  {
    fill_level_ = lulist.get<int>("Fill-in level");
  }

  // set defaults as needed
  if (invA00Str == "") invA00Str = invStr;
  if (invSStr == "") invSStr = invStr;

  inv_factory_f_ = invLib.getInverseFactory(invA00Str);

  if (invA00Str == invSStr)
    inv_factory_s_ = inv_factory_f_;
  else
    inv_factory_s_ = invLib.getInverseFactory(invSStr);
}


//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::ArrowHeadPreconditionerFactory::initializeFromParameterList(
    const Teuchos::ParameterList& pl)
{
  const std::string inverse_type = "Inverse Type";
  const std::string preconditioner_type = "Preconditioner Type";
  std::vector<Teuchos::RCP<Teko::InverseFactory>> inverses;
  std::vector<Teuchos::RCP<Teko::InverseFactory>> preconditioners;

  Teuchos::RCP<const Teko::InverseLibrary> invLib = getInverseLibrary();

  // get string specifying default inverse
  std::string invStr = "";
  invStr = "Amesos";
  std::string precStr = "None";
  if (pl.isParameter(inverse_type)) invStr = pl.get<std::string>(inverse_type);
  if (pl.isParameter(preconditioner_type)) precStr = pl.get<std::string>(preconditioner_type);
  solveType_ = Teko::GS_UseLowerTriangle;

  Teuchos::RCP<Teko::InverseFactory> defaultInverse = invLib->getInverseFactory(invStr);
  Teuchos::RCP<Teko::InverseFactory> defaultPrec;
  if (precStr != "None") defaultPrec = invLib->getInverseFactory(precStr);

  // now check individual solvers
  Teuchos::ParameterList::ConstIterator itr;
  for (itr = pl.begin(); itr != pl.end(); ++itr)
  {
    std::string fieldName = itr->first;

    // figure out what the integer is
    if (fieldName.compare(0, inverse_type.length(), inverse_type) == 0 && fieldName != inverse_type)
    {
      int position = -1;
      std::string inverse, type;

      // figure out position
      std::stringstream ss(fieldName);
      ss >> inverse >> type >> position;

      // inserting inverse factory into vector
      std::string invStr2 = pl.get<std::string>(fieldName);
      if (position > (int)inverses.size())
      {
        inverses.resize(position, defaultInverse);
        inverses[position - 1] = invLib->getInverseFactory(invStr2);
      }
      else
        inverses[position - 1] = invLib->getInverseFactory(invStr2);
    }
    else if (fieldName.compare(0, preconditioner_type.length(), preconditioner_type) == 0 &&
             fieldName != preconditioner_type)
    {
      int position = -1;
      std::string preconditioner, type;

      // figure out position
      std::stringstream ss(fieldName);
      ss >> preconditioner >> type >> position;

      // inserting preconditioner factory into vector
      std::string precStr2 = pl.get<std::string>(fieldName);
      if (position > (int)preconditioners.size())
      {
        preconditioners.resize(position, defaultPrec);
        preconditioners[position - 1] = invLib->getInverseFactory(precStr2);
      }
      else
        preconditioners[position - 1] = invLib->getInverseFactory(precStr2);
    }
  }

  // use default inverse
  if (inverses.size() == 0) inverses.push_back(defaultInverse);

  // based on parameter type build a strategy
  invOpsStrategy_ = Teuchos::rcp(
      new ArrowHeadInvDiagonalStrategy(inverses, preconditioners, defaultInverse, defaultPrec));
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Core::LinearSolver::ArrowHeadInvDiagonalStrategy::ArrowHeadInvDiagonalStrategy(
    const std::vector<Teuchos::RCP<Teko::InverseFactory>>& inverseFactories,
    const std::vector<Teuchos::RCP<Teko::InverseFactory>>& preconditionerFactories,
    const Teuchos::RCP<Teko::InverseFactory>& defaultInverseFact,
    const Teuchos::RCP<Teko::InverseFactory>& defaultPreconditionerFact)
    : Teko::InvFactoryDiagStrategy(
          inverseFactories, preconditionerFactories, defaultInverseFact, defaultPreconditionerFact)
{
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::ArrowHeadInvDiagonalStrategy::getInvD(const Teko::BlockedLinearOp& A,
    Teko::BlockPreconditionerState& state, std::vector<Teko::LinearOp>& invDiag) const
{
  Teko_DEBUG_SCOPE("InvFactoryDiagSchurStrategy::getInvD", 10);

  // loop over diagonals, build an inverse operator for each
  size_t num_blocks = A->productRange()->numBlocks();
  const std::string opPrefix = "BlockDiagOp";

  FOUR_C_ASSERT(num_blocks == 3,
      "The ArrowHeadInvDiagonalStrategy currently only works for 3x3 block matrices.");

  for (size_t i = 0; i < num_blocks; i++)
  {
    auto precFact = ((i < precDiagFact_.size()) && (!precDiagFact_[i].is_null()))
                        ? precDiagFact_[i]
                        : defaultPrecFact_;
    auto invFact = (i < invDiagFact_.size()) ? invDiagFact_[i] : defaultInvFact_;

    if (i == num_blocks - 1)
    {
      // 1. get the Schur complement contribution from the solid part (without augmentation?)
      auto A20 = Teko::getBlock(2, 0, A);
      auto A00 = Teko::getBlock(0, 0, A);
      auto A02 = Teko::getBlock(0, 2, A);

      auto diagonalType00 = Teko::getDiagonalType("Diagonal");
      auto invA00 = getInvDiagonalOp(A00, diagonalType00);

      auto triple00 = Teko::explicitMultiply(A20, Teko::explicitMultiply(invA00, A02));

      // 2. get the Schur complement contributino from the beam part (without augmentation?)
      auto A21 = Teko::getBlock(2, 1, A);
      auto A11 = Teko::getBlock(1, 1, A);
      auto A12 = Teko::getBlock(1, 2, A);

      // sparse inverse calculation
      double drop_tol = 1e-12;
      int fill_level = 4;

      auto A_op = Teuchos::rcp_dynamic_cast<const Thyra::EpetraLinearOp>(A11);
      auto A_crs = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(A_op->epetra_op(), true);

      const Core::LinAlg::SparseMatrix A_sparse(
          Core::Utils::shared_ptr_from_ref(*Teuchos::rcp_const_cast<Epetra_CrsMatrix>(A_crs)),
          Core::LinAlg::DataAccess::Copy);

      std::shared_ptr<Core::LinAlg::SparseMatrix> A_thresh =
          Core::LinAlg::threshold_matrix(A_sparse, drop_tol);
      std::shared_ptr<Core::LinAlg::Graph> sparsity_pattern_enriched =
          Core::LinAlg::enrich_matrix_graph(*A_thresh, fill_level);
      std::shared_ptr<Core::LinAlg::SparseMatrix> A_inverse =
          Core::LinAlg::matrix_sparse_inverse(A_sparse, sparsity_pattern_enriched);
      A_thresh = Core::LinAlg::threshold_matrix(*A_inverse, drop_tol);

      auto invA11 =
          Thyra::epetraLinearOp(Teuchos::make_rcp<Epetra_CrsMatrix>(A_thresh->epetra_matrix()));

      auto triple11 = Teko::explicitMultiply(A21, Teko::explicitMultiply(invA11, A12));
      auto schur = Teko::explicitAdd(triple00, triple11);
      auto schur_scaled_2 = Teko::explicitScale(-1.0, schur);

      auto A22 = Teko::getBlock(2, 2, A);
      Teko::LinearOp complete_schur = schur_scaled_2;
      if (!Teko::isZeroOp(A22)) complete_schur = Teko::explicitAdd(A22, schur_scaled_2);

      // 4. Get Schur complement
      auto inverse_2 = buildInverse(*invFact, precFact, schur_scaled_2, state, opPrefix, i);

      invDiag.push_back(inverse_2);
    }
    else
    {
      auto block = Teko::getBlock(i, i, A);
      invDiag.push_back(buildInverse(*invFact, precFact, block, state, opPrefix, i));
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
