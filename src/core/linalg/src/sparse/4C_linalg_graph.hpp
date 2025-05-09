// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_GRAPH_HPP
#define FOUR_C_LINALG_GRAPH_HPP


#include "4C_config.hpp"

#include "4C_linalg_map.hpp"

#include <Epetra_CrsGraph.h>
#include <Epetra_Export.h>
#include <Epetra_FECrsGraph.h>

#include <memory>



FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{

  class Graph
  {
   public:
    //! Type of the underlying graph object
    enum GraphType
    {
      CRS_GRAPH,
      FE_GRAPH
    };

    //! Creates a Epetra_CrsGraph object and allocates storage.
    Graph(Epetra_DataAccess CV, const Epetra_BlockMap& RowMap, const int* NumIndicesPerRow,
        bool StaticProfile = false, GraphType graphtype = CRS_GRAPH);

    Graph(Epetra_DataAccess CV, const Epetra_BlockMap& RowMap, int NumIndicesPerRow,
        bool StaticProfile = false, GraphType graphtype = CRS_GRAPH);

    Graph(Epetra_DataAccess CV, const Core::LinAlg::Map& RowMap, const int* NumIndicesPerRow,
        bool StaticProfile = false, GraphType graphtype = CRS_GRAPH);

    Graph(Epetra_DataAccess CV, const Core::LinAlg::Map& RowMap, int NumIndicesPerRow,
        bool StaticProfile = false, GraphType graphtype = CRS_GRAPH);

    Graph(const Graph& other);

    Graph& operator=(const Graph& other);

    ~Graph() = default;

    /// Copy constructor from Epetra_CrsGraph to Epetra_CrsGraph
    explicit Graph(const Epetra_CrsGraph& Source);

    /// Copy constructor from Epetra_FECrsGraph to Epetra_CrsGraph
    explicit Graph(const Epetra_FECrsGraph& Source);

    //! return the reference to the Epetra_CrsGraph
    const Epetra_CrsGraph& get_epetra_crs_graph() const { return *graph_; }

    Epetra_CrsGraph& get_epetra_crs_graph() { return *graph_; }

    //! Returns the Column Map associated with this graph.
    const Epetra_BlockMap& col_map() const { return (graph_->ColMap()); }

    //! Returns a pointer to the Epetra_Comm communicator associated with this graph.
    const Epetra_Comm& get_comm() const { return (graph_->Comm()); }

    //! Extract a list of elements in a specified global row of the graph. Put into storage
    //! allocated by calling
    int extract_global_row_copy(
        int GlobalRow, int LenOfIndices, int& NumIndices, int* Indices) const
    {
      return graph_->ExtractGlobalRowCopy(GlobalRow, LenOfIndices, NumIndices, Indices);
    }

    int extract_global_row_copy(
        long long GlobalRow, int LenOfIndices, int& NumIndices, long long* Indices) const
    {
      return graph_->ExtractGlobalRowCopy(GlobalRow, LenOfIndices, NumIndices, Indices);
    }

    int extract_local_row_view(int LocalRow, int& NumIndices, int*& Indices) const
    {
      return graph_->ExtractMyRowView(LocalRow, NumIndices, Indices);
    }

    const Epetra_Export* get_epetra_exporter() { return graph_->Exporter(); };

    int export_to(const Epetra_SrcDistObject& A, const Epetra_Export& Exporter,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return graph_->Export(A, Exporter, CombineMode, Indexor);
    }

    //! Transform to local index space. Perform other operations to allow optimal matrix operations.
    int fill_complete()
    {
      int err = 0;

      if (graphtype_ == CRS_GRAPH)
        err = graph_->FillComplete();
      else if (graphtype_ == FE_GRAPH)
        err = static_cast<Epetra_FECrsGraph*>(graph_.get())->GlobalAssemble();

      return err;
    }

    int fill_complete(const Map& domain_map, const Map& range_map)
    {
      int err = 0;

      if (graphtype_ == CRS_GRAPH)
        err = graph_->FillComplete(domain_map.get_epetra_map(), range_map.get_epetra_map());
      else if (graphtype_ == FE_GRAPH)
        err = static_cast<Epetra_FECrsGraph*>(graph_.get())
                  ->GlobalAssemble(domain_map.get_epetra_map(), range_map.get_epetra_map());

      return err;
    }

    //! If FillComplete() has been called, this query returns true, otherwise it returns false.
    bool filled() const { return (graph_->Filled()); }

    //! Imports an Epetra_DistObject using the Epetra_Import object.
    int import_from(const Epetra_SrcDistObject& A, const Epetra_Import& Importer,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr)
    {
      return graph_->Import(A, Importer, CombineMode, Indexor);
    }

    //! Enter a list of elements in a specified global row of the graph.
    int insert_global_indices(int GlobalRow, int NumIndices, int* Indices);

    int insert_global_indices(int numRows, const int* rows, int numCols, const int* cols);

    //! Get a view of the elements in a specified global row of the graph.
    int extract_global_row_view(int GlobalRow, int& NumIndices, int*& Indices) const
    {
      return graph_->ExtractGlobalRowView(GlobalRow, NumIndices, Indices);
    }

    //! Returns the allocated number of nonzero entries in specified local row on this processor.
    int num_local_indices(int Row) const { return graph_->NumMyIndices(Row); }

    //! Returns the allocated number of nonzero entries in specified local row on this processor.
    int num_allocated_local_indices(int Row) const { return graph_->NumAllocatedMyIndices(Row); }

    //! Returns the current number of nonzero entries in specified global row on this processor.
    int num_global_indices(long long Row) const { return graph_->NumGlobalIndices(Row); }

    //! Returns the number of matrix rows on this processor.
    int num_local_rows() const { return graph_->NumMyRows(); }

    //! Make consecutive row index sections contiguous, minimize internal storage used for
    //! constructing graph
    int optimize_storage() { return graph_->OptimizeStorage(); }

    //! Returns the number of indices in the global graph.
    int num_global_nonzeros() const { return graph_->NumGlobalNonzeros(); }

    //! Remove a list of elements from a specified global row of the graph.
    int remove_global_indices(int GlobalRow, int NumIndices, int* Indices);

    const Epetra_BlockMap& row_map() const { return graph_->RowMap(); }

   private:
    GraphType graphtype_;

    //! The actual Epetra_CrsGraph object.
    std::unique_ptr<Epetra_CrsGraph> graph_;
  };
}  // namespace Core::LinAlg
FOUR_C_NAMESPACE_CLOSE


#endif