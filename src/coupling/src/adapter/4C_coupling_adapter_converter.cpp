// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_coupling_adapter_converter.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_constraint_framework_embeddedmesh_solid_to_solid_mortar_manager.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/

std::shared_ptr<Core::LinAlg::Vector<double>>
Coupling::Adapter::CouplingMasterConverter::src_to_dst(
    std::shared_ptr<const Core::LinAlg::Vector<double>> source_vector) const
{
  return coup_.master_to_slave(*source_vector);
}

std::shared_ptr<Core::LinAlg::Vector<double>>
Coupling::Adapter::CouplingMasterConverter::dst_to_src(
    std::shared_ptr<const Core::LinAlg::Vector<double>> destination_vector) const
{
  return coup_.slave_to_master(*destination_vector);
}

std::shared_ptr<const Core::LinAlg::Map> Coupling::Adapter::CouplingMasterConverter::src_map() const
{
  return coup_.master_dof_map();
}

std::shared_ptr<const Core::LinAlg::Map> Coupling::Adapter::CouplingMasterConverter::dst_map() const
{
  return coup_.slave_dof_map();
}

std::shared_ptr<const Core::LinAlg::Map> Coupling::Adapter::CouplingMasterConverter::perm_src_map()
    const
{
  return coup_.perm_master_dof_map();
}

std::shared_ptr<const Core::LinAlg::Map> Coupling::Adapter::CouplingMasterConverter::perm_dst_map()
    const
{
  return coup_.perm_slave_dof_map();
}

void Coupling::Adapter::CouplingMasterConverter::fill_src_to_dst_map(
    std::map<int, int>& rowmap) const
{
  coup_.fill_master_to_slave_map(rowmap);
}


std::shared_ptr<Core::LinAlg::Vector<double>> Coupling::Adapter::CouplingSlaveConverter::src_to_dst(
    std::shared_ptr<const Core::LinAlg::Vector<double>> source_vector) const
{
  return coup_.slave_to_master(*source_vector);
}

std::shared_ptr<Core::LinAlg::Vector<double>> Coupling::Adapter::CouplingSlaveConverter::dst_to_src(
    std::shared_ptr<const Core::LinAlg::Vector<double>> destination_vector) const
{
  return coup_.master_to_slave(*destination_vector);
}

std::shared_ptr<const Core::LinAlg::Map> Coupling::Adapter::CouplingSlaveConverter::src_map() const
{
  return coup_.slave_dof_map();
}

std::shared_ptr<const Core::LinAlg::Map> Coupling::Adapter::CouplingSlaveConverter::dst_map() const
{
  return coup_.master_dof_map();
}

std::shared_ptr<const Core::LinAlg::Map> Coupling::Adapter::CouplingSlaveConverter::perm_src_map()
    const
{
  return coup_.perm_slave_dof_map();
}

std::shared_ptr<const Core::LinAlg::Map> Coupling::Adapter::CouplingSlaveConverter::perm_dst_map()
    const
{
  return coup_.perm_master_dof_map();
}

void Coupling::Adapter::CouplingSlaveConverter::fill_src_to_dst_map(
    std::map<int, int>& rowmap) const
{
  coup_.fill_slave_to_master_map(rowmap);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool Coupling::Adapter::MatrixLogicalSplitAndTransform::operator()(
    const Core::LinAlg::SparseMatrix& src, const Core::LinAlg::Map& logical_range_map,
    const Core::LinAlg::Map& logical_domain_map, double scale,
    const CouplingConverter* row_converter, const CouplingConverter* col_converter,
    Core::LinAlg::SparseMatrix& dst, bool exactmatch, bool addmatrix)
{
  auto esrc = std::make_shared<Core::LinAlg::SparseMatrix>(src);
  const Core::LinAlg::Map* final_range_map = &logical_range_map;
  const Core::LinAlg::Map* matching_dst_rows = &logical_range_map;

  if (row_converter)
  {
    const Core::LinAlg::Map& permsrcmap = *row_converter->perm_src_map();

    // check if the permuted map is simply a subset of the current rowmap (no communication)
    int subset = 1;
    for (int i = 0; i < permsrcmap.num_my_elements(); ++i)
      if (!src.row_map().my_gid(permsrcmap.gid(i)))
      {
        subset = 0;
        break;
      }

    int gsubset = 0;
    gsubset = Core::Communication::min_all(subset, logical_range_map.get_comm());

    // need communication -> call import on permuted map
    if (!gsubset)
    {
      if (exporter_ == nullptr)
      {
        exporter_ = std::make_shared<Core::LinAlg::Export>(permsrcmap, src.row_map());
      }

      auto permsrc = std::make_shared<Core::LinAlg::SparseMatrix>(permsrcmap, 0);
      permsrc->import(src, *exporter_, Insert);
      permsrc->complete(src.domain_map(), permsrcmap);
      esrc = permsrc;
    }

    final_range_map = &permsrcmap;
    matching_dst_rows = row_converter->dst_map().get();
  }

  setup_gid_map(col_converter ? *col_converter->src_map() : Core::LinAlg::Map(esrc->row_map()),
      Core::LinAlg::Map(esrc->col_map()), col_converter,
      Core::Communication::unpack_epetra_comm(src.Comm()));

  if (!addmatrix) dst.zero();

  internal_add(*esrc, *final_range_map,
      col_converter ? *col_converter->src_map() : logical_domain_map, *matching_dst_rows, dst,
      exactmatch, scale);

  return true;
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::MatrixLogicalSplitAndTransform::setup_gid_map(
    const Core::LinAlg::Map& rowmap, const Core::LinAlg::Map& colmap,
    const CouplingConverter* converter, MPI_Comm comm)
{
  if (not havegidmap_)
  {
    if (converter != nullptr)
    {
      Core::Communication::Exporter ex(rowmap, colmap, comm);
      converter->fill_src_to_dst_map(gidmap_);
      ex.do_export(gidmap_);
    }
    else
      for (int i = 0; i < colmap.num_my_elements(); ++i) gidmap_[colmap.gid(i)] = colmap.gid(i);
    havegidmap_ = true;
  }
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::MatrixLogicalSplitAndTransform::internal_add(
    Core::LinAlg::SparseMatrix& esrc, const Core::LinAlg::Map& logical_range_map,
    const Core::LinAlg::Map& logical_domain_map, const Core::LinAlg::Map& matching_dst_rows,
    Core::LinAlg::SparseMatrix& edst, bool exactmatch, double scale)
{
  if (not esrc.filled()) FOUR_C_THROW("filled source matrix expected");

  Core::LinAlg::Vector<double> dselector(esrc.domain_map());
  for (int i = 0; i < dselector.local_length(); ++i)
  {
    const int gid = esrc.domain_map().gid(i);
    if (logical_domain_map.my_gid(gid))
      dselector.get_values()[i] = 1.;
    else
      dselector.get_values()[i] = 0.;
  }
  Core::LinAlg::Vector<double> selector(esrc.col_map());
  Core::LinAlg::export_to(dselector, selector);

  if (edst.filled())
    add_into_filled(esrc, logical_range_map, logical_domain_map, selector, matching_dst_rows, edst,
        exactmatch, scale);
  else
    add_into_unfilled(esrc, logical_range_map, logical_domain_map, selector, matching_dst_rows,
        edst, exactmatch, scale);
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::MatrixLogicalSplitAndTransform::add_into_filled(
    Core::LinAlg::SparseMatrix& esrc, const Core::LinAlg::Map& logical_range_map,
    const Core::LinAlg::Map& logical_domain_map, const Core::LinAlg::Vector<double>& selector,
    const Core::LinAlg::Map& matching_dst_rows, Core::LinAlg::SparseMatrix& edst, bool exactmatch,
    double scale)
{
  const Core::LinAlg::Map& srccolmap = Core::LinAlg::Map(esrc.col_map());
  const Core::LinAlg::Map& dstrowmap = Core::LinAlg::Map(edst.row_map());

  // If the destination matrix is filled, we can add in local indices. This code is similar
  // to what is done in Core::LinAlg::Add(SparseMatrix, SparseMatrix) for the filled case.
  // We perform four steps:
  // 1. Identify the local column index mapping from the source to the destination matrix from
  //    on the global IDs
  // 2. Loop over the input matrix rows, extract row view in two matrices
  // 3. Match columns of row i in source matrix (called A) to the columns in the destination
  //    matrix (called B)
  // 4. Perform addition
  if (int(lidvector_.size()) != srccolmap.num_my_elements())
  {
    lidvector_.clear();
    lidvector_.resize(srccolmap.num_my_elements(), -1);
    for (std::map<int, int>::const_iterator iter = gidmap_.begin(); iter != gidmap_.end(); ++iter)
    {
      const int lid = srccolmap.lid(iter->first);
      if (lid != -1) lidvector_[lid] = edst.col_map().lid(iter->second);
    }
  }

  int rows = logical_range_map.num_my_elements();
  for (int i = 0; i < rows; ++i)
  {
    int NumEntriesA, NumEntriesB;
    double *ValuesA, *ValuesB;
    int *IndicesA, *IndicesB;
    const int rowA = esrc.row_map().lid(logical_range_map.gid(i));
    if (rowA == -1) FOUR_C_THROW("Internal error");
    esrc.extract_my_row_view(rowA, NumEntriesA, ValuesA, IndicesA);

    // identify the local row index in the destination matrix corresponding to i
    const int rowB = dstrowmap.lid(matching_dst_rows.gid(i));
    edst.extract_my_row_view(rowB, NumEntriesB, ValuesB, IndicesB);

    // loop through the columns in source matrix and find respective place in destination
    for (int jA = 0, jB = 0; jA < NumEntriesA; ++jA)
    {
      // skip entries belonging to a different block of the logical block matrix
      if (selector.local_values_as_span()[IndicesA[jA]] == 0.) continue;

      const int col = lidvector_[IndicesA[jA]];
      if (col == -1)
      {
        if (exactmatch)
          FOUR_C_THROW("gid {} not found in map for lid {} at {}", srccolmap.gid(IndicesA[jA]),
              IndicesA[jA], jA);
        else
          continue;
      }

      // try linear search in B
      while (jB < NumEntriesB && IndicesB[jB] < col) ++jB;

      // did not find index in linear search (re-indexing from A.ColMap() to B.ColMap()
      // might pass through the indices differently), try binary search
      if (jB == NumEntriesB || IndicesB[jB] != col)
        jB = std::lower_bound(IndicesB, IndicesB + NumEntriesB, col) - IndicesB;

      // not found, sparsity pattern of B does not contain the index from A -> terminate
      if (jB == NumEntriesB || IndicesB[jB] != col)
      {
        FOUR_C_THROW(
            "Source matrix entry with global row ID {} and global column ID {} couldn't be added to"
            " destination matrix entry with global row ID {} and unknown global column ID {}!",
            esrc.row_map().gid(i), srccolmap.gid(IndicesA[jA]), matching_dst_rows.gid(i),
            edst.col_map().gid(col));
      }

      ValuesB[jB] += ValuesA[jA] * scale;
    }
  }
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Coupling::Adapter::MatrixLogicalSplitAndTransform::add_into_unfilled(
    Core::LinAlg::SparseMatrix& esrc, const Core::LinAlg::Map& logical_range_map,
    const Core::LinAlg::Map& logical_domain_map, const Core::LinAlg::Vector<double>& selector,
    const Core::LinAlg::Map& matching_dst_rows, Core::LinAlg::SparseMatrix& edst, bool exactmatch,
    double scale)
{
  const Core::LinAlg::Map& srccolmap = Core::LinAlg::Map(esrc.col_map());

  // standard code for the unfilled case
  std::vector<int> idx;
  std::vector<double> vals;
  int rows = logical_range_map.num_my_elements();
  for (int i = 0; i < rows; ++i)
  {
    int NumEntries;
    double* Values;
    int* Indices;
    esrc.extract_my_row_view(
        esrc.row_map().lid(logical_range_map.gid(i)), NumEntries, Values, Indices);

    idx.clear();
    vals.clear();

    for (int j = 0; j < NumEntries; ++j)
    {
      // skip entries belonging to a different block of the logical block matrix
      if (selector.local_values_as_span()[Indices[j]] == 0.) continue;

      int gid = srccolmap.gid(Indices[j]);
      std::map<int, int>::const_iterator iter = gidmap_.find(gid);
      if (iter != gidmap_.end())
      {
        idx.push_back(iter->second);
        vals.push_back(Values[j] * scale);
      }
      else
      {
        // only complain if an exact match is demanded
        if (exactmatch)
          FOUR_C_THROW("gid {} not found in map for lid {} at {}", gid, Indices[j], j);
      }
    }

    NumEntries = vals.size();
    const int globalRow = matching_dst_rows.gid(i);

    // put row into matrix
    //
    // We might want to preserve a Dirichlet row in our destination matrix
    // here as well. Skip for now.

    if (edst.num_allocated_global_entries(globalRow) == 0)
    {
      int err = edst.insert_global_values(globalRow, NumEntries, vals.data(), idx.data());
      if (err < 0) FOUR_C_THROW("InsertGlobalValues error: {}", err);
    }
    else
      for (int j = 0; j < NumEntries; ++j)
      {
        // add all values, including zeros, as we need a proper matrix graph
        int err = edst.sum_into_global_values(globalRow, 1, &vals[j], &idx[j]);
        if (err > 0)
        {
          err = edst.insert_global_values(globalRow, 1, &vals[j], &idx[j]);
          if (err < 0) FOUR_C_THROW("InsertGlobalValues error: {}", err);
        }
        else if (err < 0)
          FOUR_C_THROW("SumIntoGlobalValues error: {}", err);
      }
  }
}



bool Coupling::Adapter::MatrixRowTransform::operator()(const Core::LinAlg::SparseMatrix& src,
    double scale, const CouplingConverter& converter, Core::LinAlg::SparseMatrix& dst,
    bool addmatrix)
{
  return transformer_(
      src, src.range_map(), src.domain_map(), scale, &converter, nullptr, dst, false, addmatrix);
}



bool Coupling::Adapter::MatrixColTransform::operator()(const Core::LinAlg::Map&,
    const Core::LinAlg::Map&, const Core::LinAlg::SparseMatrix& src, double scale,
    const CouplingConverter& converter, Core::LinAlg::SparseMatrix& dst, bool exactmatch,
    bool addmatrix)
{
  return transformer_(src, src.range_map(), src.domain_map(), scale, nullptr, &converter, dst,
      exactmatch, addmatrix);
}



bool Coupling::Adapter::MatrixRowColTransform::operator()(const Core::LinAlg::SparseMatrix& src,
    double scale, const CouplingConverter& rowconverter, const CouplingConverter& colconverter,
    Core::LinAlg::SparseMatrix& dst, bool exactmatch, bool addmatrix)
{
  return transformer_(src, src.range_map(), src.domain_map(), scale, &rowconverter, &colconverter,
      dst, exactmatch, addmatrix);
}

FOUR_C_NAMESPACE_CLOSE
