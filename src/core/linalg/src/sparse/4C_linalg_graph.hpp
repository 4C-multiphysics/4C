// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_GRAPH_HPP
#define FOUR_C_LINALG_GRAPH_HPP


#include "4C_config.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_transfer.hpp"

#include <Epetra_CrsGraph.h>
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

    //! Return the communicator associated with this Graph.
    MPI_Comm get_comm() const { return Core::Communication::unpack_epetra_comm(graph_->Comm()); }

    //! Transform to local index space. Perform other operations to allow optimal matrix operations.
    void fill_complete();

    void fill_complete(const Map& domain_map, const Map& range_map);

    //! If FillComplete() has been called, this query returns true, otherwise it returns false.
    bool filled() const { return (graph_->Filled()); }

    //! Make consecutive row index sections contiguous, minimize internal storage used for
    //! constructing graph
    void optimize_storage();

    void export_to(const Epetra_SrcDistObject& A, const Core::LinAlg::Export& Exporter,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr);

    //! Imports an Epetra_DistObject using the Core::LinAlg::Import object.
    void import_from(const Epetra_SrcDistObject& A, const Core::LinAlg::Import& Importer,
        Epetra_CombineMode CombineMode, const Epetra_OffsetIndex* Indexor = nullptr);

    //! Enter a list of elements in a specified global row of the graph.
    void insert_global_indices(int GlobalRow, int NumIndices, int* Indices);

    void insert_global_indices(int numRows, const int* rows, int numCols, const int* cols);

    void extract_local_row_view(int LocalRow, int& NumIndices, int*& Indices) const;

    //! Get a view of the elements in a specified global row of the graph.
    void extract_global_row_view(int GlobalRow, int& NumIndices, int*& Indices) const;

    //! Extract a list of elements in a specified global row of the graph. Put into storage
    //! allocated by calling
    void extract_global_row_copy(
        int GlobalRow, int LenOfIndices, int& NumIndices, int* Indices) const;

    //! Returns the allocated number of nonzero entries in specified local row on this processor.
    int num_local_indices(int Row) const { return graph_->NumMyIndices(Row); }

    //! Returns the allocated number of nonzero entries in specified local row on this processor.
    int num_allocated_local_indices(int Row) const { return graph_->NumAllocatedMyIndices(Row); }

    //! Returns the current number of nonzero entries in specified global row on this processor.
    int num_global_indices(long long Row) const { return graph_->NumGlobalIndices(Row); }

    //! Returns the number of matrix rows on this processor.
    int num_local_rows() const { return graph_->NumMyRows(); }

    //! Returns the number of indices in the global graph.
    int num_global_nonzeros() const { return graph_->NumGlobalNonzeros(); }

    //! Returns the Row Map associated with this graph.
    const Map& row_map() const { return row_map_.sync(graph_->RowMap()); }

    //! Returns the Column Map associated with this graph.
    const Map& col_map() const { return col_map_.sync(graph_->ColMap()); }

   private:
    GraphType graphtype_;

    //! The actual Epetra_CrsGraph object.
    std::unique_ptr<Epetra_CrsGraph> graph_;
    mutable View<const Map> row_map_;
    mutable View<const Map> col_map_;
  };
}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif