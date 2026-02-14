// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_preconditioner_muelu.hpp"

#include "4C_comm_utils.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_linear_solver_thyra_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <MueLu_CreateXpetraPreconditioner.hpp>
#include <MueLu_EpetraOperator.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <MueLu_UseDefaultTypes.hpp>
#include <Stratimikos_MueLuHelpers.hpp>
#include <Teko_EpetraInverseOpWrapper.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPStdSharedPtrConversions.hpp>
#include <Thyra_XpetraLinearOp_decl.hpp>
#include <Xpetra_BlockedCrsMatrix.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_EpetraCrsMatrix.hpp>
#include <Xpetra_EpetraMap.hpp>
#include <Xpetra_EpetraMultiVector.hpp>
#include <Xpetra_MatrixUtils.hpp>
#include <Xpetra_ThyraUtils.hpp>

#include <filesystem>

FOUR_C_NAMESPACE_OPEN

using SC = Scalar;
using LO = LocalOrdinal;
using GO = GlobalOrdinal;
using NO = Node;

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
Core::LinearSolver::MueLuPreconditioner::MueLuPreconditioner(Teuchos::ParameterList& muelulist)
    : muelulist_(muelulist)
{
}

//----------------------------------------------------------------------------------
//----------------------------------------------------------------------------------
void Core::LinearSolver::MueLuPreconditioner::setup(
    Core::LinAlg::SparseOperator& matrix, Core::LinAlg::MultiVector<double>& b)
{
  using EpetraMultiVector = Xpetra::EpetraMultiVectorT<GO, NO>;

  if (!muelulist_.sublist("MueLu Parameters").isParameter("MUELU_XML_FILE"))
    FOUR_C_THROW("MUELU_XML_FILE parameter not set!");
  auto xmlFileName = muelulist_.sublist("MueLu Parameters").get<std::string>("MUELU_XML_FILE");

  Teuchos::ParameterList muelu_params;
  auto comm = Core::Communication::to_teuchos_comm<int>(matrix.get_comm());
  Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, Teuchos::Ptr(&muelu_params), *comm);

  Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> A =
      Teuchos::rcp_dynamic_cast<Core::LinAlg::BlockSparseMatrixBase>(Teuchos::rcpFromRef(matrix));

  if (A.is_null())
  {
    auto A_crs = Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(Teuchos::rcpFromRef(matrix));
    pmatrix_ =
        Core::LinearSolver::Utils::create_thyra_linear_op(*A_crs, Core::LinAlg::DataAccess::Copy);

    const Teuchos::ParameterList& inverseList = muelulist_.sublist("MueLu Parameters");
    const int number_of_equations = inverseList.get<int>("PDE equations");

    const auto epetra_map = A_crs->row_map().get_epetra_block_map();
    const Teuchos::RCP<const Xpetra::Map<int, int, Xpetra::EpetraNode>> row_map =
        Xpetra::toXpetra<int, Xpetra::EpetraNode>(epetra_map);

    Teuchos::RCP<Xpetra::MultiVector<SC, LO, GO, NO>> nullspace =
        Core::LinearSolver::Parameters::extract_nullspace_from_parameterlist(*row_map, inverseList);

    Teuchos::RCP<Xpetra::MultiVector<SC, LO, GO, NO>> coordinates =
        Teuchos::make_rcp<EpetraMultiVector>(Teuchos::rcpFromRef(
            inverseList.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("Coordinates")
                ->get_epetra_multi_vector()));

    muelu_params.set("number of equations", number_of_equations);
    Teuchos::ParameterList& user_param_list = muelu_params.sublist("user data");
    user_param_list.set("Nullspace", nullspace);
    user_param_list.set("Coordinates", coordinates);

    Teuchos::RCP<Xpetra::MultiVector<SC, LO, GO, NO>> material;
    if (muelulist_.isParameter("Material"))
    {
      material = Teuchos::make_rcp<EpetraMultiVector>(Teuchos::rcpFromRef(
          muelulist_.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("Material")
              ->get_epetra_multi_vector()));
      user_param_list.set("Material", material);
    }

    // setup preconditioner builder and enable relevant packages
    Stratimikos::LinearSolverBuilder<double> builder;

    // enable multigrid
    Stratimikos::enableMueLu<Scalar, LocalOrdinal, GlobalOrdinal, Node>(builder);

    // set preconditioner parameter list
    Teuchos::ParameterList stratimikos_params;
    Teuchos::ParameterList& muelu_list =
        stratimikos_params.sublist("Preconditioner Types").sublist("MueLu");
    muelu_list.setParameters(muelu_params);
    builder.setParameterList(
        Teuchos::make_rcp<Teuchos::ParameterList>(std::move(stratimikos_params)));

    // construct preconditioning operator
    Teuchos::RCP<Thyra::PreconditionerFactoryBase<double>> precFactory =
        builder.createPreconditioningStrategy("MueLu");
    Teuchos::RCP<Thyra::PreconditionerBase<double>> prec =
        Thyra::prec<double>(*precFactory, pmatrix_);
    p_ = prec->getUnspecifiedPrecOp();
  }
  else
  {
    using EpetraCrsMatrix = Xpetra::EpetraCrsMatrixT<GO, NO>;
    using EpetraMap = Xpetra::EpetraMapT<GO, NO>;

    std::vector<Teuchos::RCP<const Xpetra::Map<LO, GO, NO>>> maps;

    for (int block = 0; block < A->rows(); block++)
    {
      Teuchos::RCP<Xpetra::CrsMatrix<SC, LO, GO, NO>> crsA = Teuchos::make_rcp<EpetraCrsMatrix>(
          Teuchos::rcpFromRef(A->matrix(block, block).epetra_matrix()));

      const std::string inverse = "Inverse" + std::to_string(block + 1);
      const Teuchos::ParameterList& inverseList =
          muelulist_.sublist(inverse).sublist("MueLu Parameters");
      const int number_of_equations = inverseList.get<int>("PDE equations");

      std::vector<size_t> striding;
      striding.emplace_back(number_of_equations);

      Teuchos::RCP<const Xpetra::StridedMap<LO, GO, NO>> map =
          Teuchos::make_rcp<Xpetra::StridedMap<LO, GO, NO>>(crsA->getRowMap()->lib(),
              crsA->getRowMap()->getGlobalNumElements(), crsA->getRowMap()->getLocalElementList(),
              crsA->getRowMap()->getIndexBase(), striding, crsA->getRowMap()->getComm(), -1);

      maps.emplace_back(map);
    }

    Teuchos::RCP<const Xpetra::Map<LO, GO, NO>> fullrangemap =
        Xpetra::MapUtils<LO, GO, NO>::concatenateMaps(maps);

    Teuchos::RCP<const Xpetra::MapExtractor<SC, LO, GO, NO>> map_extractor =
        Xpetra::MapExtractorFactory<SC, LO, GO, NO>::Build(fullrangemap, maps);

    auto bOp = Teuchos::make_rcp<Xpetra::BlockedCrsMatrix<SC, LO, GO, NO>>(
        map_extractor, map_extractor, 42);

    for (int row = 0; row < A->rows(); row++)
    {
      for (int col = 0; col < A->cols(); col++)
      {
        Teuchos::RCP<Xpetra::CrsMatrix<SC, LO, GO, NO>> crsA = Teuchos::make_rcp<EpetraCrsMatrix>(
            Teuchos::rcpFromRef(A->matrix(row, col).epetra_matrix()));
        Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO>> mat =
            Xpetra::MatrixFactory<SC, LO, GO, NO>::BuildCopy(
                Teuchos::make_rcp<Xpetra::CrsMatrixWrap<SC, LO, GO, NO>>(crsA));
        bOp->setMatrix(row, col, mat);
      }
    }

    bOp->fillComplete();
    pmatrix_ = Xpetra::ThyraUtils<SC>::toThyra(bOp);

    MueLu::ParameterListInterpreter<SC, LO, GO, NO> mueLuFactory(
        xmlFileName, *bOp->getRowMap()->getComm());
    H_ = mueLuFactory.CreateHierarchy();
    H_->GetLevel(0)->Set("A", Teuchos::rcp_dynamic_cast<Xpetra::Matrix<SC, LO, GO, NO>>(bOp));

    for (int block = 0; block < A->rows(); block++)
    {
      const std::string inverse = "Inverse" + std::to_string(block + 1);
      const Teuchos::ParameterList& inverse_list =
          muelulist_.sublist(inverse).sublist("MueLu Parameters");

      Teuchos::RCP<Xpetra::MultiVector<SC, LO, GO, NO>> nullspace =
          Core::LinearSolver::Parameters::extract_nullspace_from_parameterlist(
              *maps.at(block), inverse_list);

      H_->GetLevel(0)->Set("Nullspace" + std::to_string(block + 1), nullspace);
    }

    if (muelulist_.sublist("Belos Parameters").isParameter("contact slaveDofMap"))
    {
      Teuchos::RCP<Epetra_Map> ep_slave_dof_map =
          muelulist_.sublist("Belos Parameters")
              .get<Teuchos::RCP<Epetra_Map>>("contact slaveDofMap");

      if (ep_slave_dof_map.is_null()) FOUR_C_THROW("Interface contact map is not available!");

      Teuchos::RCP<EpetraMap> x_slave_dof_map = Teuchos::make_rcp<EpetraMap>(ep_slave_dof_map);

      H_->GetLevel(0)->Set("Primal interface DOF map",
          Teuchos::rcp_dynamic_cast<const Xpetra::Map<LO, GO, NO>>(x_slave_dof_map, true));
    }

    if (muelulist_.sublist("Belos Parameters").isParameter("Interface DualNodeID to PrimalNodeID"))
    {
      Teuchos::RCP<std::map<LO, LO>> dual2primal_map =
          muelulist_.sublist("Belos Parameters")
              .get<Teuchos::RCP<std::map<LO, LO>>>("Interface DualNodeID to PrimalNodeID");

      if (dual2primal_map.is_null())
        FOUR_C_THROW("'Interface DualNodeID to PrimalNodeID' map is not available!");

      H_->GetLevel(0)->Set("DualNodeID2PrimalNodeID",
          Teuchos::rcp_dynamic_cast<std::map<int, int>>(dual2primal_map, true));
    }

    mueLuFactory.SetupHierarchy(*H_);
    // p_ = std::make_shared<MueLu::EpetraOperator>(H_);
  }
}

FOUR_C_NAMESPACE_CLOSE