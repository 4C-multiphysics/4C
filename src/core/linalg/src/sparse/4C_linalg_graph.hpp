// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_GRAPH_HPP
#define FOUR_C_LINALG_GRAPH_HPP


#include "4C_config.hpp"

#include <Epetra_CrsGraph.h>
#include <Epetra_Export.h>
#include <Epetra_FECrsGraph.h>

#include <memory>

// Do not lint the file for identifier names, since the naming of the Wrapper functions follow the
// naming of the Epetra_Vector

// NOLINTBEGIN(readability-identifier-naming)

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{

  class Graph
  {
   public:
    //! Creates a Epetra_CrsGraph object and allocates storage.
    Graph(Epetra_DataAccess CV, const Epetra_BlockMap& RowMap, const int* NumIndicesPerRow,
        bool StaticProfile = false);

    Graph(Epetra_DataAccess CV, const Epetra_BlockMap& RowMap, int NumIndicesPerRow,
        bool StaticProfile = false);

    Graph(const Graph& other);

    Graph& operator=(const Graph& other);

    ~Graph() = default;

    /// Copy constructor from Epetra_CrsGraph to vector
    explicit Graph(const Epetra_CrsGraph& Source);

    /// Copy constructor from Epetra_FECrsGraph to vector
    explicit Graph(const Epetra_FECrsGraph& Source);


    //! (Implicit) conversions: they all return references or RCPs, never copies
    const Epetra_CrsGraph& get_ref_of_Epetra_CrsGraph() const { return *graph_; }

    Epetra_CrsGraph& get_ref_of_Epetra_CrsGraph() { return *graph_; }

    std::shared_ptr<Epetra_CrsGraph> get_ptr_of_Epetra_CrsGraph() { return graph_; }


    // Functions

    //! Returns the Column Map associated with this graph.
    const Epetra_BlockMap& ColMap() const { return (graph_->ColMap()); }

    //! Returns a pointer to the Epetra_Comm communicator associated with this graph.
    const Epetra_Comm& Comm() const { return (graph_->Comm()); }

    //! Extract a list of elements in a specified global row of the graph. Put into storage
    //! allocated by calling
    int ExtractGlobalRowCopy(int GlobalRow, int LenOfIndices, int& NumIndices, int* Indices) const
    {
      return graph_->ExtractGlobalRowCopy(GlobalRow, LenOfIndices, NumIndices, Indices);
    }

    int ExtractGlobalRowCopy(
        long long GlobalRow, int LenOfIndices, int& NumIndices, long long* Indices) const
    {
      return graph_->ExtractGlobalRowCopy(GlobalRow, LenOfIndices, NumIndices, Indices);
    }

    int ExtractMyRowView(int LocalRow, int& NumIndices, int*& Indices) const
    {
      return graph_->ExtractMyRowView(LocalRow, NumIndices, Indices);
    }

    const Epetra_Export* Exporter() { return graph_->Exporter(); };

    int Export(const Epetra_SrcDistObject& A, const Epetra_Export& Exporter,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = 0)
    {
      return graph_->Export(A, Exporter, CombineMode, Indexor);
    }

    //! Transform to local index space. Perform other operations to allow optimal matrix operations.
    int FillComplete() const { return graph_->FillComplete(); }

    //! If FillComplete() has been called, this query returns true, otherwise it returns false.
    bool Filled() const { return (graph_->Filled()); }

    //! Imports an Epetra_DistObject using the Epetra_Import object.
    int Import(const Epetra_SrcDistObject& A, const Epetra_Import& Importer,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = 0)
    {
      return graph_->Import(A, Importer, CombineMode, Indexor);
    }

    //! Enter a list of elements in a specified global row of the graph.
    int InsertGlobalIndices(int GlobalRow, int NumIndices, int* Indices);

    int InsertGlobalIndices(long long GlobalRow, int NumIndices, long long* Indices);

    //! Get a view of the elements in a specified global row of the graph.
    int ExtractGlobalRowView(int GlobalRow, int& NumIndices, int*& Indices) const
    {
      return graph_->ExtractGlobalRowView(GlobalRow, NumIndices, Indices);
    }

    int ExtractGlobalRowView(long long GlobalRow, int& NumIndices, long long*& Indices) const
    {
      return graph_->ExtractGlobalRowView(GlobalRow, NumIndices, Indices);
    }

    //! Returns the allocated number of nonzero entries in specified local row on this processor.
    int NumMyIndices(int Row) const { return graph_->NumMyIndices(Row); }

    //! Returns the allocated number of nonzero entries in specified local row on this processor.
    int NumAllocatedMyIndices(int Row) const { return graph_->NumAllocatedMyIndices(Row); }

    //! Returns the current number of nonzero entries in specified global row on this processor.
    int NumGlobalIndices(long long Row) const { return graph_->NumGlobalIndices(Row); }

    //! Returns the number of matrix rows on this processor.
    int NumMyRows() const { return graph_->NumMyRows(); }

    //! Make consecutive row index sections contiguous, minimize internal storage used for
    //! constructing graph
    int OptimizeStorage() const { return graph_->OptimizeStorage(); }

    //! Returns the number of indices in the global graph.
    int NumGlobalNonzeros() const { return graph_->NumGlobalNonzeros(); }

    //! Remove a list of elements from a specified global row of the graph.
    int RemoveGlobalIndices(int GlobalRow, int NumIndices, int* Indices);

    int RemoveGlobalIndices(long long GlobalRow, int NumIndices, long long* Indices);


    const Epetra_BlockMap& RowMap() const { return graph_->RowMap(); }

    explicit Graph(const std::shared_ptr<Core::LinAlg::Graph>& graph);


   private:
    //! The actual Epetra_CrsGraph object.
    std::shared_ptr<Epetra_CrsGraph> graph_;
  };
}  // namespace Core::LinAlg
FOUR_C_NAMESPACE_CLOSE

// NOLINTEND(readability-identifier-naming)

#endif