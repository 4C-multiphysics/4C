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

#include <Stratimikos_DefaultLinearSolverBuilder.hpp>
#include <Stratimikos_MueLuHelpers.hpp>
#include <Teko_EpetraInverseOpWrapper.hpp>
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
void Core::LinearSolver::TekoPreconditioner::setup(Core::LinAlg::SparseOperator& matrix,
    const Core::LinAlg::MultiVector<double>& x, Core::LinAlg::MultiVector<double>& b)
{
  using EpetraMultiVector = Xpetra::EpetraMultiVectorT<GlobalOrdinal, Node>;
  using XpetraMultiVector = Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

  if (!tekolist_.sublist("Teko Parameters").isParameter("TEKO_XML_FILE"))
    FOUR_C_THROW("TEKO_XML_FILE parameter not set!");
  auto xmlFileName = tekolist_.sublist("Teko Parameters").get<std::string>("TEKO_XML_FILE");

  Teuchos::ParameterList tekoParams;
  auto comm = Core::Communication::to_teuchos_comm<int>(
      Core::Communication::unpack_epetra_comm(matrix.Comm()));
  Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, Teuchos::Ptr(&tekoParams), *comm);

  auto A = std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(
      Core::Utils::shared_ptr_from_ref(matrix));

  if (!A)
  {
    if (tekolist_.sublist("Teko Parameters").isParameter("extractor"))
    {
      std::shared_ptr<Core::LinAlg::MultiMapExtractor> extractor =
          tekolist_.sublist("Teko Parameters")
              .get<std::shared_ptr<Core::LinAlg::MultiMapExtractor>>("extractor");

      auto A_sparse = std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(
          Core::Utils::shared_ptr_from_ref(matrix));

      A = Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          *A_sparse, *extractor, *extractor);

      A->complete();
    }
  }

  // wrap linear operators
  if (!A)
  {
    auto A_crs = Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(Teuchos::rcpFromRef(matrix));
    pmatrix_ = A_crs->thyra_operator(LinAlg::DataAccess::Copy);
  }
  else
  {
    pmatrix_ = A->thyra_operator(LinAlg::DataAccess::Copy);

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
Core::LinearSolver::LU2x2SpaiStrategy::LU2x2SpaiStrategy(
    const Teuchos::RCP<Teko::InverseFactory>& invFA, const Teuchos::RCP<Teko::InverseFactory>& invS)
    : inv_factory_f_(invFA), inv_factory_s_(invS)
{
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

FOUR_C_NAMESPACE_CLOSE
