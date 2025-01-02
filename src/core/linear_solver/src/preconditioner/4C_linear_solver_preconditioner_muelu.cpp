// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_preconditioner_muelu.hpp"

#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_utils_exceptions.hpp"

#include <MueLu_CreateXpetraPreconditioner.hpp>
#include <MueLu_EpetraOperator.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <MueLu_UseDefaultTypes.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_RCPStdSharedPtrConversions.hpp>
#include <Xpetra_BlockedCrsMatrix.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_EpetraCrsMatrix.hpp>
#include <Xpetra_EpetraMap.hpp>
#include <Xpetra_EpetraMultiVector.hpp>
#include <Xpetra_Map.hpp>
#include <Xpetra_MapExtractor.hpp>
#include <Xpetra_MapExtractorFactory.hpp>
#include <Xpetra_MatrixUtils.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_StridedMap.hpp>

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
void Core::LinearSolver::MueLuPreconditioner::setup(bool create, Epetra_Operator* matrix,
    Core::LinAlg::MultiVector<double>* x, Core::LinAlg::MultiVector<double>* b)
{
  using EpetraCrsMatrix = Xpetra::EpetraCrsMatrixT<GO, NO>;
  using EpetraMap = Xpetra::EpetraMapT<GO, NO>;
  using EpetraMultiVector = Xpetra::EpetraMultiVectorT<GO, NO>;

  if (create)
  {
    Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> A =
        Teuchos::rcp_dynamic_cast<Core::LinAlg::BlockSparseMatrixBase>(
            Teuchos::rcpFromRef(*matrix));

    if (A.is_null())
    {
      Teuchos::RCP<Epetra_CrsMatrix> crsA =
          Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(Teuchos::rcpFromRef(*matrix));

      Teuchos::RCP<Xpetra::CrsMatrix<SC, LO, GO, NO>> mueluA =
          Teuchos::make_rcp<EpetraCrsMatrix>(crsA);
      pmatrix_ = Xpetra::MatrixFactory<SC, LO, GO, NO>::BuildCopy(
          Teuchos::make_rcp<Xpetra::CrsMatrixWrap<SC, LO, GO, NO>>(mueluA));

      Teuchos::ParameterList& inverseList = muelulist_.sublist("MueLu Parameters");

      std::string xmlFileName = inverseList.get<std::string>("MUELU_XML_FILE");
      if (xmlFileName == "none") FOUR_C_THROW("MUELU_XML_FILE parameter not set!");

      Teuchos::RCP<Teuchos::ParameterList> muelu_params =
          Teuchos::make_rcp<Teuchos::ParameterList>();
      auto comm = pmatrix_->getRowMap()->getComm();
      Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, muelu_params.ptr(), *comm);

      const int number_of_equations = inverseList.get<int>("PDE equations");
      pmatrix_->SetFixedBlockSize(number_of_equations);

      Teuchos::RCP<const Xpetra::Map<LO, GO, NO>> row_map = mueluA->getRowMap();
      Teuchos::RCP<Xpetra::MultiVector<SC, LO, GO, NO>> nullspace =
          Core::LinearSolver::Parameters::extract_nullspace_from_parameterlist(
              *row_map, inverseList);

      Teuchos::RCP<Xpetra::MultiVector<SC, LO, GO, NO>> coordinates =
          Teuchos::make_rcp<EpetraMultiVector>(Teuchos::rcpFromRef(
              *inverseList.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("Coordinates")
                   ->get_ptr_of_Epetra_MultiVector()));

      muelu_params->set("number of equations", number_of_equations);
      Teuchos::ParameterList& user_param_list = muelu_params->sublist("user data");
      user_param_list.set("Nullspace", nullspace);
      user_param_list.set("Coordinates", coordinates);

      H_ = MueLu::CreateXpetraPreconditioner(pmatrix_, *muelu_params);
      P_ = Teuchos::make_rcp<MueLu::EpetraOperator>(H_);
    }
    else
    {
      std::vector<Teuchos::RCP<const Xpetra::Map<LO, GO, NO>>> maps;

      for (int block = 0; block < A->rows(); block++)
      {
        EpetraCrsMatrix crsA(Teuchos::rcpFromRef(*A->matrix(block, block).epetra_matrix()));

        std::string inverse = "Inverse" + std::to_string(block + 1);
        Teuchos::ParameterList& inverseList =
            muelulist_.sublist(inverse).sublist("MueLu Parameters");
        const int number_of_equations = inverseList.get<int>("PDE equations");

        std::vector<size_t> striding;
        striding.emplace_back(number_of_equations);

        Teuchos::RCP<const Xpetra::StridedMap<LO, GO, NO>> map =
            Teuchos::make_rcp<Xpetra::StridedMap<LO, GO, NO>>(crsA.getRowMap()->lib(),
                crsA.getRowMap()->getGlobalNumElements(), crsA.getRowMap()->getLocalElementList(),
                crsA.getRowMap()->getIndexBase(), striding, crsA.getRowMap()->getComm(), -1);

        maps.emplace_back(map);
      }

      Teuchos::RCP<const EpetraMap> fullrangemap =
          Teuchos::make_rcp<EpetraMap>(Teuchos::rcpFromRef(A->full_range_map()));
      Teuchos::RCP<const Xpetra::MapExtractor<SC, LO, GO, NO>> map_extractor =
          Xpetra::MapExtractorFactory<SC, LO, GO, NO>::Build(fullrangemap, maps);

      auto bOp = Teuchos::make_rcp<Xpetra::BlockedCrsMatrix<SC, LO, GO, NO>>(
          map_extractor, map_extractor, 42);

      for (int row = 0; row < A->rows(); row++)
      {
        for (int col = 0; col < A->cols(); col++)
        {
          Teuchos::RCP<Xpetra::CrsMatrix<SC, LO, GO, NO>> crsA = Teuchos::make_rcp<EpetraCrsMatrix>(
              Teuchos::rcpFromRef(*A->matrix(row, col).epetra_matrix()));
          Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO>> mat =
              Xpetra::MatrixFactory<SC, LO, GO, NO>::BuildCopy(
                  Teuchos::make_rcp<Xpetra::CrsMatrixWrap<SC, LO, GO, NO>>(crsA));
          bOp->setMatrix(row, col, mat);
        }
      }

      bOp->fillComplete();
      pmatrix_ = bOp;

      if (!muelulist_.sublist("MueLu Parameters").isParameter("MUELU_XML_FILE"))
        FOUR_C_THROW("MUELU_XML_FILE parameter not set!");

      std::string xmlFileName =
          muelulist_.sublist("MueLu Parameters").get<std::string>("MUELU_XML_FILE");
      Teuchos::RCP<Teuchos::ParameterList> mueluParams =
          Teuchos::make_rcp<Teuchos::ParameterList>();
      auto comm = pmatrix_->getRowMap()->getComm();
      Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, mueluParams.ptr(), *comm);

      MueLu::ParameterListInterpreter<SC, LO, GO, NO> mueLuFactory(xmlFileName, *comm);
      Teuchos::RCP<MueLu::Hierarchy<SC, LO, GO, NO>> H = mueLuFactory.CreateHierarchy();
      H->GetLevel(0)->Set("A", Teuchos::rcp_dynamic_cast<Xpetra::Matrix<SC, LO, GO, NO>>(pmatrix_));

      for (int block = 0; block < A->rows(); block++)
      {
        std::string inverse = "Inverse" + std::to_string(block + 1);
        Teuchos::ParameterList& inverse_list =
            muelulist_.sublist(inverse).sublist("MueLu Parameters");

        Teuchos::RCP<Xpetra::MultiVector<SC, LO, GO, NO>> nullspace =
            Teuchos::make_rcp<EpetraMultiVector>(Teuchos::rcpFromRef(
                *inverse_list.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace")
                     ->get_ptr_of_Epetra_MultiVector()));

        H->GetLevel(0)->Set("Nullspace" + std::to_string(block + 1), nullspace);
      }

      if (muelulist_.sublist("Belos Parameters").isParameter("contact slaveDofMap"))
      {
        Teuchos::RCP<Epetra_Map> ep_slave_dof_map =
            muelulist_.sublist("Belos Parameters")
                .get<Teuchos::RCP<Epetra_Map>>("contact slaveDofMap");
        Teuchos::RCP<EpetraMap> x_slave_dof_map = Teuchos::make_rcp<EpetraMap>(ep_slave_dof_map);

        H->GetLevel(0)->Set("Primal interface DOF map",
            Teuchos::rcp_dynamic_cast<const Xpetra::Map<LO, GO, NO>>(x_slave_dof_map, true));
      }

      mueLuFactory.SetupHierarchy(*H);
      P_ = Teuchos::make_rcp<MueLu::EpetraOperator>(H);
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
