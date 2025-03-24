// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_comm_utils.hpp"

#include "4C_io_pstream.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_CrsMatrix.h>

#include <memory>
#include <stdexcept>

FOUR_C_NAMESPACE_OPEN

namespace
{
  std::shared_ptr<Core::Communication::Communicators> mock_up_communicators()
  {
    // mock up for command line to create communicators
    std::vector<std::string> argv{
        "dummyEntryInputFile", "-nptype=separateDatFiles", "-ngroup=2", "-glayout=1,2"};

    return Core::Communication::create_comm(argv);
  };

  /**
   * Class to setup parallel vectors which are compared.
   */
  class SetupCompareParallelVectorsTest : public ::testing::Test

  {
   public:
    /**
     * \brief Set up the testing environment.
     */
    SetupCompareParallelVectorsTest()
    {
      communicators_ = mock_up_communicators();
      Core::IO::cout.setup(
          false, false, false, Core::IO::standard, communicators_->local_comm(), 0, 0, "dummy");

      // create arbitrary distributed map within each group
      std::shared_ptr<Core::LinAlg::Map> map =
          std::make_shared<Core::LinAlg::Map>(numberOfElementsToDistribute_, 0,
              Core::Communication::as_epetra_comm(communicators_->local_comm()));
      epetraVector_ = std::make_shared<Core::LinAlg::Vector<double>>(*map, false);

      // fill test Core::LinAlg::Vector<double> with entry equals gid
      int numMyEles = map->NumMyElements();
      double* values = new double[numMyEles];
      int* indices = new int[numMyEles];
      for (int lid = 0; lid < numMyEles; ++lid)
      {
        indices[lid] = lid;
        values[lid] = map->GID(lid);
      }
      epetraVector_->replace_local_values(numMyEles, values, indices);
    }

    void TearDown() override { Core::IO::cout.close(); }

   public:
    std::shared_ptr<Core::Communication::Communicators> communicators_;
    const int numberOfElementsToDistribute_ = 791;
    std::shared_ptr<Core::LinAlg::Vector<double>> epetraVector_;
  };

  /**
   * Class to setup parallel matrices which are compared.
   */
  class SetupCompareParallelMatricesTest : public ::testing::Test

  {
   public:
    /**
     * \brief Set up the testing environment.
     */
    SetupCompareParallelMatricesTest()
    {
      communicators_ = mock_up_communicators();
      Core::IO::cout.setup(
          false, false, false, Core::IO::standard, communicators_->local_comm(), 0, 0, "dummy");

      // create arbitrary distributed map within each group
      std::shared_ptr<Core::LinAlg::Map> rowmap =
          std::make_shared<Core::LinAlg::Map>(numberOfElementsToDistribute_, 0,
              Core::Communication::as_epetra_comm(communicators_->local_comm()));
      int approximateNumberOfNonZeroesPerRow = 3;
      epetraCrsMatrix_ = std::make_shared<Epetra_CrsMatrix>(
          ::Copy, *rowmap, approximateNumberOfNonZeroesPerRow, false);

      // fill tri-diagonal Epetra_CrsMatrix
      double* values = new double[3];
      int* columnIndices = new int[3];
      int numMyEles = rowmap->NumMyElements();
      for (int lid = 0; lid < numMyEles; ++lid)
      {
        int rowgid = rowmap->GID(lid);
        if (rowgid == 0)  // first global row
        {
          int colIndicesFirstRow[2] = {0, 1};
          double valuesFirstRow[2] = {static_cast<double>(rowgid) + colIndicesFirstRow[0],
              static_cast<double>(rowgid) + colIndicesFirstRow[1]};
          epetraCrsMatrix_->InsertGlobalValues(rowgid, 2, valuesFirstRow, colIndicesFirstRow);
        }
        else if (rowgid == numberOfElementsToDistribute_ - 1)  // last global row
        {
          rowgid = rowmap->GID(numMyEles - 1);
          int colIndicesLastRow[2] = {rowgid - 1, rowgid};
          double valuesLastRow[2] = {static_cast<double>(rowgid) + colIndicesLastRow[0],
              static_cast<double>(rowgid) + colIndicesLastRow[1]};
          epetraCrsMatrix_->InsertGlobalValues(rowgid, 2, valuesLastRow, colIndicesLastRow);
        }
        else  // all rows in between
        {
          columnIndices[0] = rowgid - 1;
          columnIndices[1] = rowgid;
          columnIndices[2] = rowgid + 1;
          values[0] = static_cast<double>(rowgid) + columnIndices[0];
          values[1] = static_cast<double>(rowgid) + columnIndices[1];
          values[2] = static_cast<double>(rowgid) + columnIndices[2];
          epetraCrsMatrix_->InsertGlobalValues(rowgid, 3, values, columnIndices);
        }
      }

      epetraCrsMatrix_->FillComplete(false);
    }

    void TearDown() override { Core::IO::cout.close(); }

   public:
    std::shared_ptr<Core::Communication::Communicators> communicators_;
    const int numberOfElementsToDistribute_ = 673;
    std::shared_ptr<Epetra_CrsMatrix> epetraCrsMatrix_;
  };

  /**
   * Class to setup parallel rectangular matrices which are compared.
   */
  class SetupCompareParallelRectangularMatricesTest : public ::testing::Test

  {
   public:
    /**
     * \brief Set up the testing environment.
     */
    SetupCompareParallelRectangularMatricesTest()
    {
      communicators_ = mock_up_communicators();
      Core::IO::cout.setup(
          false, false, false, Core::IO::standard, communicators_->local_comm(), 0, 0, "dummy");

      // create arbitrary distributed map within each group
      std::shared_ptr<Core::LinAlg::Map> rowmap =
          std::make_shared<Core::LinAlg::Map>(numberOfElementsToDistribute_, 0,
              Core::Communication::as_epetra_comm(communicators_->local_comm()));
      Core::LinAlg::Map colmap(2 * numberOfElementsToDistribute_, 0,
          Core::Communication::as_epetra_comm(communicators_->local_comm()));
      int approximateNumberOfNonZeroesPerRow = 6;
      epetraCrsMatrix_ = std::make_shared<Epetra_CrsMatrix>(
          ::Copy, *rowmap, approximateNumberOfNonZeroesPerRow, false);

      // fill rectangular Epetra_CrsMatrix
      double* values = new double[6];
      int* columnIndices = new int[6];
      int numMyEles = rowmap->NumMyElements();
      for (int lid = 0; lid < numMyEles; ++lid)
      {
        int rowgid = rowmap->GID(lid);
        if (rowgid == 0)  // first global row
        {
          int colIndicesFirstRow[4] = {
              0, 1, numberOfElementsToDistribute_, numberOfElementsToDistribute_ + 1};
          double valuesFirstRow[4] = {static_cast<double>(rowgid) + colIndicesFirstRow[0],
              static_cast<double>(rowgid) + colIndicesFirstRow[1],
              static_cast<double>(rowgid) + colIndicesFirstRow[0],
              static_cast<double>(rowgid) + colIndicesFirstRow[1]};
          epetraCrsMatrix_->InsertGlobalValues(rowgid, 4, valuesFirstRow, colIndicesFirstRow);
        }
        else if (rowgid == numberOfElementsToDistribute_ - 1)  // last global row
        {
          rowgid = rowmap->GID(numMyEles - 1);
          int colIndicesLastRow[4] = {rowgid - 1, rowgid,
              rowgid - 1 + numberOfElementsToDistribute_, rowgid + numberOfElementsToDistribute_};
          double valuesLastRow[4] = {static_cast<double>(rowgid) + colIndicesLastRow[0],
              static_cast<double>(rowgid) + colIndicesLastRow[1],
              static_cast<double>(rowgid) + colIndicesLastRow[0],
              static_cast<double>(rowgid) + colIndicesLastRow[1]};
          epetraCrsMatrix_->InsertGlobalValues(rowgid, 4, valuesLastRow, colIndicesLastRow);
        }
        else  // all rows in between
        {
          columnIndices[0] = rowgid - 1;
          columnIndices[1] = rowgid;
          columnIndices[2] = rowgid + 1;
          columnIndices[3] = rowgid - 1 + numberOfElementsToDistribute_;
          columnIndices[4] = rowgid + numberOfElementsToDistribute_;
          columnIndices[5] = rowgid + 1 + numberOfElementsToDistribute_;
          values[0] = static_cast<double>(rowgid) + columnIndices[0];
          values[1] = static_cast<double>(rowgid) + columnIndices[1];
          values[2] = static_cast<double>(rowgid) + columnIndices[2];
          values[3] = values[0];
          values[4] = values[1];
          values[5] = values[2];
          epetraCrsMatrix_->InsertGlobalValues(rowgid, 6, values, columnIndices);
        }
      }
      epetraCrsMatrix_->FillComplete(colmap, *rowmap);
    }

    void TearDown() override { Core::IO::cout.close(); }

   public:
    std::shared_ptr<Core::Communication::Communicators> communicators_;
    const int numberOfElementsToDistribute_ = 673;
    std::shared_ptr<Epetra_CrsMatrix> epetraCrsMatrix_;
  };

  TEST_F(SetupCompareParallelVectorsTest, PositiveTestCompareVectors)
  {
    bool success = Core::Communication::are_distributed_vectors_identical(*communicators_,
        *epetraVector_,  //
        "epetraVector");
    EXPECT_EQ(success, true);
  }

  TEST_F(SetupCompareParallelVectorsTest, NegativeTestCompareVectors)
  {
    // disturb one value on each proc which leads to a failure of the comparison
    const int lastLocalIndex = epetraVector_->local_length() - 1;
    double disturbedValue = static_cast<double>(lastLocalIndex);
    epetraVector_->replace_local_values(1, &disturbedValue, &lastLocalIndex);

    EXPECT_THROW(Core::Communication::are_distributed_vectors_identical(
                     *communicators_, *epetraVector_, "epetraVector"),
        Core::Exception);
  }

  TEST_F(SetupCompareParallelMatricesTest, PositiveTestCompareMatrices)
  {
    bool success = Core::Communication::are_distributed_sparse_matrices_identical(
        *communicators_, *epetraCrsMatrix_, "epetraCrsMatrix");
    EXPECT_EQ(success, true);
  }

  TEST_F(SetupCompareParallelMatricesTest, NegativeTestCompareMatrices)
  {
    // disturb one value on each proc which leads to a failure of the comparison
    const int myLastLid[1] = {epetraCrsMatrix_->RowMap().NumMyElements() - 1};
    const double value[1] = {static_cast<double>(myLastLid[0])};

    epetraCrsMatrix_->InsertMyValues(myLastLid[0], 1, &value[0], &myLastLid[0]);
    epetraCrsMatrix_->FillComplete(false);

    EXPECT_THROW(Core::Communication::are_distributed_sparse_matrices_identical(
                     *communicators_, *epetraCrsMatrix_, "epetraCrsMatrix"),
        Core::Exception);
  }

  TEST_F(SetupCompareParallelRectangularMatricesTest, PositiveTestCompareRectangularMatrices)
  {
    bool success = Core::Communication::are_distributed_sparse_matrices_identical(
        *communicators_, *epetraCrsMatrix_, "rectangularEpetraCrsMatrix");
    EXPECT_EQ(success, true);
  }

}  // namespace

FOUR_C_NAMESPACE_CLOSE
