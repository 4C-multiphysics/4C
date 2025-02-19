// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_graph.hpp"

// Do not lint the file for identifier names, since the naming of the Wrapper functions follow the
// naming of the Epetra_Vector

// NOLINTBEGIN(readability-identifier-naming)

FOUR_C_NAMESPACE_OPEN


Core::LinAlg::Graph::Graph(const Epetra_CrsGraph& Source)
    : graph_(std::make_shared<Epetra_CrsGraph>(Source))
{
}

Core::LinAlg::Graph::Graph(const Epetra_FECrsGraph& Source)
    : graph_(std::make_shared<Epetra_CrsGraph>(Source))
{
}

Core::LinAlg::Graph::Graph(Epetra_DataAccess CV, const Epetra_BlockMap& RowMap,
    const int* NumIndicesPerRow, bool StaticProfile)
    : graph_(std::make_shared<Epetra_CrsGraph>(CV, RowMap, NumIndicesPerRow, StaticProfile))
{
}

Core::LinAlg::Graph::Graph(const Graph& other)
    : graph_(std::make_shared<Epetra_CrsGraph>(other.get_ref_of_Epetra_CrsGraph()))
{
}

Core::LinAlg::Graph& Core::LinAlg::Graph::operator=(const Graph& other)
{
  *graph_ = other.get_ref_of_Epetra_CrsGraph();
  return *this;
}


Core::LinAlg::Graph::Graph(
    Epetra_DataAccess CV, const Epetra_BlockMap& RowMap, int NumIndicesPerRow, bool StaticProfile)
    : graph_(std::make_shared<Epetra_CrsGraph>(CV, RowMap, NumIndicesPerRow, StaticProfile))
{
}

int Core::LinAlg::Graph::InsertGlobalIndices(int GlobalRow, int NumIndices, int* Indices)
{
  return graph_->InsertGlobalIndices(GlobalRow, NumIndices, Indices);
}

int Core::LinAlg::Graph::InsertGlobalIndices(
    long long GlobalRow, int NumIndices, long long* Indices)
{
  return graph_->InsertGlobalIndices(GlobalRow, NumIndices, Indices);
}

int Core::LinAlg::Graph::RemoveGlobalIndices(int GlobalRow, int NumIndices, int* Indices)
{
  return graph_->RemoveGlobalIndices(GlobalRow, NumIndices, Indices);
}

int Core::LinAlg::Graph::RemoveGlobalIndices(
    long long GlobalRow, int NumIndices, long long* Indices)
{
  return graph_->RemoveGlobalIndices(GlobalRow, NumIndices, Indices);
}


FOUR_C_NAMESPACE_CLOSE

// NOLINTEND(readability-identifier-naming)