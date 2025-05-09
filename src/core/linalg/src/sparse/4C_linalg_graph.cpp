// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_graph.hpp"

#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN


Core::LinAlg::Graph::Graph(const Epetra_CrsGraph& Source)
    : graphtype_(CRS_GRAPH), graph_(std::make_unique<Epetra_CrsGraph>(Source))
{
}

Core::LinAlg::Graph::Graph(const Epetra_FECrsGraph& Source)
    : graphtype_(CRS_GRAPH), graph_(std::make_unique<Epetra_CrsGraph>(Source))
{
}

Core::LinAlg::Graph::Graph(Epetra_DataAccess CV, const Epetra_BlockMap& RowMap,
    const int* NumIndicesPerRow, bool StaticProfile, GraphType graphtype)
    : graphtype_(graphtype)
{
  if (graphtype_ == CRS_GRAPH)
    graph_ = std::make_unique<Epetra_CrsGraph>(CV, RowMap, NumIndicesPerRow, StaticProfile);
  else if (graphtype_ == FE_GRAPH)
    graph_ = std::make_unique<Epetra_FECrsGraph>(
        CV, RowMap, const_cast<int*>(NumIndicesPerRow), StaticProfile);
}

Core::LinAlg::Graph::Graph(Epetra_DataAccess CV, const Map& RowMap, const int* NumIndicesPerRow,
    bool StaticProfile, GraphType graphtype)
    : graphtype_(graphtype)
{
  if (graphtype_ == CRS_GRAPH)
    graph_ = std::make_unique<Epetra_CrsGraph>(
        CV, RowMap.get_epetra_map(), NumIndicesPerRow, StaticProfile);
  else if (graphtype_ == FE_GRAPH)
    graph_ = std::make_unique<Epetra_FECrsGraph>(
        CV, RowMap.get_epetra_map(), const_cast<int*>(NumIndicesPerRow), StaticProfile);
}

Core::LinAlg::Graph::Graph(Epetra_DataAccess CV, const Epetra_BlockMap& RowMap,
    int NumIndicesPerRow, bool StaticProfile, GraphType graphtype)
    : graphtype_(graphtype)
{
  if (graphtype_ == CRS_GRAPH)
    graph_ = std::make_unique<Epetra_CrsGraph>(CV, RowMap, NumIndicesPerRow, StaticProfile);
  else if (graphtype_ == FE_GRAPH)
    graph_ = std::make_unique<Epetra_FECrsGraph>(CV, RowMap, NumIndicesPerRow, StaticProfile);
}

Core::LinAlg::Graph::Graph(Epetra_DataAccess CV, const Map& RowMap, int NumIndicesPerRow,
    bool StaticProfile, GraphType graphtype)
    : graphtype_(graphtype)
{
  if (graphtype_ == CRS_GRAPH)
    graph_ = std::make_unique<Epetra_CrsGraph>(
        CV, RowMap.get_epetra_map(), NumIndicesPerRow, StaticProfile);
  else if (graphtype_ == FE_GRAPH)
    graph_ = std::make_unique<Epetra_FECrsGraph>(
        CV, RowMap.get_epetra_map(), NumIndicesPerRow, StaticProfile);
}

Core::LinAlg::Graph::Graph(const Graph& other)
    : graphtype_(CRS_GRAPH), graph_(std::make_unique<Epetra_CrsGraph>(other.get_epetra_crs_graph()))
{
}

Core::LinAlg::Graph& Core::LinAlg::Graph::operator=(const Graph& other)
{
  *graph_ = other.get_epetra_crs_graph();
  return *this;
}

int Core::LinAlg::Graph::insert_global_indices(int GlobalRow, int NumIndices, int* Indices)
{
  return graph_->InsertGlobalIndices(GlobalRow, NumIndices, Indices);
}

int Core::LinAlg::Graph::insert_global_indices(
    int numRows, const int* rows, int numCols, const int* cols)
{
  int err = 0;

  if (graphtype_ == CRS_GRAPH)
    FOUR_C_THROW("This type of insert_global_indices() only available for FE_GRAPH type.");
  else if (graphtype_ == FE_GRAPH)
    err = static_cast<Epetra_FECrsGraph*>(graph_.get())
              ->InsertGlobalIndices(numRows, rows, numCols, cols);

  return err;
}

int Core::LinAlg::Graph::remove_global_indices(int GlobalRow, int NumIndices, int* Indices)
{
  return graph_->RemoveGlobalIndices(GlobalRow, NumIndices, Indices);
}

FOUR_C_NAMESPACE_CLOSE
