// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_UTILS_SPARSE_ALGEBRA_MANIPULATION_HPP
#define FOUR_C_LINALG_UTILS_SPARSE_ALGEBRA_MANIPULATION_HPP

#include "4C_config.hpp"

#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_graph.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"

#include <Epetra_CrsMatrix.h>
#include <Epetra_Export.h>
#include <Epetra_Import.h>
#include <Epetra_Map.h>

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /*!
   \brief Communicate a vector to a different map

   Values of source are copied to target where maps don't have to match.
   Prerequisite: Either the map of source OR the map of target has to be unique
   (will be tested)
   \warning When source is overlapping (and therefore target is unique), values
   in the overlapping region are inserted into the target on a first come
   first serve basis, meaning they should be equal in the source to
   be deterministic
   \param source (in) : source vector values are taken from
   \param target (out): target vector values will be inserted in
   */
  void export_to(
      const Core::LinAlg::MultiVector<double>& source, Core::LinAlg::MultiVector<double>& target);

  /*!
   \brief Communicate a vector to a different map

   Values of source are copied to target where maps don't have to match.
   Prerequisite: Either the map of source OR the map of target has to be unique
   (will be tested)
   \warning When source is overlapping (and therefore target is unique), values
   in the overlapping region are inserted into the target on a first come
   first serve basis, meaning they should be equal in the source to
   be deterministic
   \param source (in) : source vector values are taken from
   \param target (out): target vector values will be inserted in
   */
  void export_to(const Core::LinAlg::Vector<int>& source, Core::LinAlg::Vector<int>& target);

  /*! \brief Extract a partial Core::LinAlg::Vector<double> from a given source vector
   *         on each proc without communication
   *
   *  This methods uses a given partial map to create the partial target vector.
   *
   *  \param source     (in) : source vector ( read-only )
   *  \param target_map (in) : map of the new target vector ( read-only )
   *
   *  \return the extracted partial Core::LinAlg::Vector<double> as RCP
   *
   *  \author hiermeier \date 03/17 */
  std::unique_ptr<Core::LinAlg::Vector<double>> extract_my_vector(
      const Core::LinAlg::Vector<double>& source, const Epetra_Map& target_map);

  /*! \brief Extract a partial Eptra_Vector from a given source vector
   *         on each proc without communication
   *
   *  \param source (in) : source vector ( read-only )
   *  \param target (out): this target vector is going to be filled
   *
   *  \author hiermeier \date 03/17 */
  void extract_my_vector(
      const Core::LinAlg::Vector<double>& source, Core::LinAlg::Vector<double>& target);

  /*! \brief Filter a sparse matrix based on a threshold value.
   *
   *  \param A         (in) : Matrix to filter
   *  \param threshold (in) : Filter value
   *
   *  \return Returned the filtered sparse matrix.
   */
  std::unique_ptr<Core::LinAlg::SparseMatrix> threshold_matrix(
      const Core::LinAlg::SparseMatrix& A, const double threshold);

  /*! \brief Filter the graph of a sparse matrix based on a threshold value and diagonal Jacobi
   *         scaling.
   *
   * E. Chow: Parallel implementation and practical use of sparse approximate inverse
   * preconditioners with a priori sparsity patterns.
   * The International Journal of High Performance Computing Applications, 15(1):56-74, 2001,
   * https://doi.org/10.1177/109434200101500106
   *
   *  \param A         (in) : Matrix to filter
   *  \param threshold (in) : Filter value
   *
   *  \return Returned the filtered sparse matrix graph.
   */
  std::shared_ptr<Core::LinAlg::Graph> threshold_matrix_graph(
      const Core::LinAlg::SparseMatrix& A, const double threshold);

  /*! \brief Enrich a matrix graph based on it's powers.
   *
   *  \param A     (in) : Sparse matrix, which graph needs to be enriched
   *  \param power (in) : Power value
   *
   *  \return Returned the enriched graph G(A^(power))
   */
  std::shared_ptr<Core::LinAlg::Graph> enrich_matrix_graph(const SparseMatrix& A, int power);

  /*!
   \brief split a matrix into a 2x2 block system where the rowmap of one of the blocks is given
          and return a block matrix

   Splits a given matrix into a 2x2 block system where the rowmap of one of the blocks is given
   on input. Blocks A11 and A22 are assumed to be square.
   All values on entry have to be nullptr except the given rowmap and matrix A.
   Note that either A11rowmap or A22rowmap or both have to be nonzero. In case
   both rowmaps are supplied they have to be an exact and nonoverlapping split of A->RowMap().
   Matrix blocks are fill_complete() on exit.

   \param A         : Matrix A on input
   \param Ablock    : Blockmatrix version of A to be calculated
   \param A11rowmap : rowmap of A11 or null
   \param A22rowmap : rowmap of A22 or null
   */
  bool split_matrix2x2(std::shared_ptr<Epetra_CrsMatrix> A,
      std::shared_ptr<BlockSparseMatrix<DefaultBlockMatrixStrategy>>& Ablock,
      std::shared_ptr<Epetra_Map>& A11rowmap, std::shared_ptr<Epetra_Map>& A22rowmap);

  /*!
   \brief split a matrix into a 2x2 block system

   Splits a given matrix into a 2x2 block system. All values on entry have to be
   nullptr except the given rowmap(s) / domainmap(s) and matrix A.
   Note that either A11rowmap or A22rowmap or both have to be nonzero!
   Note that either A11domainmap or A22domainmap or both have to be nonzero!
   In case both rowmaps / domainmaps are supplied they have to be an exact and
   nonoverlapping split of A->RowMap() / A->DomainMap().
   Matrix blocks are fill_complete() on exit.

   \param A            : Matrix A on input
   \param A11rowmap    : rowmap of A11 or null
   \param A22rowmap    : rowmap of A22 or null
   \param A11domainmap : domainmap of A11 or null
   \param A22domainmap : domainmap of A22 or null
   \param A11          : on exit matrix block A11
   \param A12          : on exit matrix block A12
   \param A21          : on exit matrix block A21
   \param A22          : on exit matrix block A22
   */
  bool split_matrix2x2(std::shared_ptr<Core::LinAlg::SparseMatrix> A,
      std::shared_ptr<Epetra_Map>& A11rowmap, std::shared_ptr<Epetra_Map>& A22rowmap,
      std::shared_ptr<Epetra_Map>& A11domainmap, std::shared_ptr<Epetra_Map>& A22domainmap,
      std::shared_ptr<Core::LinAlg::SparseMatrix>& A11,
      std::shared_ptr<Core::LinAlg::SparseMatrix>& A12,
      std::shared_ptr<Core::LinAlg::SparseMatrix>& A21,
      std::shared_ptr<Core::LinAlg::SparseMatrix>& A22);

  /*! \brief Split matrix in 2x2 blocks, where main diagonal blocks have to be square
   *
   *   Used by split interface method, does not call Complete() on output matrix.
   */
  void split_matrix2x2(
      const Core::LinAlg::SparseMatrix& ASparse, Core::LinAlg::BlockSparseMatrixBase& ABlock);

  /*! \brief Split matrix in MxN blocks
   *
   *   Used by split interface method, does not call Complete() on output matrix.
   */
  void split_matrixmxn(
      const Core::LinAlg::SparseMatrix& ASparse, Core::LinAlg::BlockSparseMatrixBase& ABlock);

  /*! \brief Split matrix in either 2x2 or NxN blocks (with N>2)

    Split given sparse matrix into 2x2 or NxN block matrix and return result as templated
    BlockSparseMatrix. The MultiMapExtractor's provided have to be 2x2 or NxN maps, otherwise
    this method will throw an error.

    \warning This is an expensive operation!

    \note This method will NOT call Complete() on the output BlockSparseMatrix.
   */
  template <class Strategy>
  std::shared_ptr<Core::LinAlg::BlockSparseMatrix<Strategy>> split_matrix(
      const Core::LinAlg::SparseMatrix& ASparse, const MultiMapExtractor& domainmaps,
      const MultiMapExtractor& rangemaps)
  {
    // initialize resulting BlockSparseMatrix. no need to provide estimates of nonzeros because
    // all entries will be inserted at once anyway
    std::shared_ptr<BlockSparseMatrix<Strategy>> blockA =
        std::make_shared<Core::LinAlg::BlockSparseMatrix<Strategy>>(
            domainmaps, rangemaps, 0, ASparse.explicit_dirichlet(), ASparse.save_graph());

    if (domainmaps.num_maps() == 2 && rangemaps.num_maps() == 2)
      split_matrix2x2(ASparse, *blockA);
    else if (domainmaps.num_maps() > 0 && rangemaps.num_maps() > 0)
      split_matrixmxn(ASparse, *blockA);
    else
      FOUR_C_THROW(
          "Invalid number %d of row blocks or %d of column blocks for splitting operation!",
          rangemaps.num_maps(), domainmaps.num_maps());

    return blockA;
  }

  /** \brief Insert a diagonal row vector into a unfilled SparseMatrix
   *         on each proc without communication
   *
   *  \param mat (out) : Unfilled matrix
   *  \param diag (in) : Given diagonal (row-layout)
   *
   *  Return 0, if successful. If the given matrix is already filled, the method
   *  returns -1. In this case you should use replace_diagonal_values(), instead.
   *
   *  \author hiermeier \date 03/17 */
  int insert_my_row_diagonal_into_unfilled_matrix(
      Core::LinAlg::SparseMatrix& mat, const Core::LinAlg::Vector<double>& diag);

  /*!
   \brief Split an Epetra_Map and return the part complementary to \c Agiven

   Splits \c Amap into 2 maps, where one is given on input and the other map
   is created as complementary map. The complementary map is returned.

   \param[in] Amap      : Map to split on input
   \param[in] Agiven    : on entry submap that is given and part of Amap
   \return the remainder map of Amap that is not overlapping with Agiven
   */
  std::shared_ptr<Epetra_Map> split_map(const Epetra_Map& Amap, const Epetra_Map& Agiven);

  /*!
   \brief merges two given Epetra_Maps

   merges input map1 and input map2, both of which have to be unique,
   but may be overlapping, to a new map and returns std::shared_ptr to it.

   \param map1         : one map to be merged
   \param map2         : the other map to be merged
   \param allowoverlap : when set to false, an error is thrown if the result
   map is overlapping (default = true, overlap allowed)
   \return the (sorted) merged map of input maps map1 and map2
   */
  std::shared_ptr<Epetra_Map> merge_map(
      const Epetra_Map& map1, const Epetra_Map& map2, bool overlap = true);

  /*!
   \brief find the intersection set of two given Epetra_Maps

   Find the insection set of input map1 and input map2.

   \param map1         : first map
   \param map2         : second map
   \return the (sorted) intersection map of input maps map1 and map2
   */
  std::shared_ptr<Epetra_Map> intersect_map(const Epetra_Map& map1, const Epetra_Map& map2);


  /*!
   \brief merges two given Epetra_Maps

   merges input map1 and input map2 (given as std::shared_ptr), both of which
   have to be unique, but may be overlapping, to a new map and returns
   std::shared_ptr to it. The case that one or both input std::shared_ptrs are null is
   detected and handled appropriately.

   \param map1         : one map to be merged
   \param map2         : the other map to be merged
   \param allowoverlap : when set to false, an error is thrown if the result
   map is overlapping (default = true, overlap allowed)
   \return the (sorted) merged map of input maps map1 and map2
   */
  std::shared_ptr<Epetra_Map> merge_map(const std::shared_ptr<const Epetra_Map>& map1,
      const std::shared_ptr<const Epetra_Map>& map2, bool overlap = true);

  /*!
     \brief split a vector into 2 non-overlapping pieces (std::shared_ptr version)

     \param xmap    : map of vector to be split
     \param x       : vector to be split
     \param x1map   : map of first vector to be extracted
     \param x1      : first vector to be extracted
     \param x2map   : map of second vector to be extracted
     \param x2      : second vector to be extracted

     */
  bool split_vector(const Epetra_Map& xmap, const Core::LinAlg::Vector<double>& x,
      std::shared_ptr<Epetra_Map>& x1map, std::shared_ptr<Core::LinAlg::Vector<double>>& x1,
      std::shared_ptr<Epetra_Map>& x2map, std::shared_ptr<Core::LinAlg::Vector<double>>& x2);

  /*!
   \brief split a vector into 2 non-overlapping pieces (std::shared_ptr version)

   \param xmap    : map of vector to be split
   \param x       : vector to be split
   \param x1map   : map of first vector to be extracted
   \param x1      : first vector to be extracted
   \param x2map   : map of second vector to be extracted
   \param x2      : second vector to be extracted

   */
  bool split_vector(const Epetra_Map& xmap, const Core::LinAlg::Vector<double>& x,
      std::shared_ptr<const Epetra_Map>& x1map, std::shared_ptr<Core::LinAlg::Vector<double>>& x1,
      std::shared_ptr<const Epetra_Map>& x2map, std::shared_ptr<Core::LinAlg::Vector<double>>& x2);

  /*! \brief Write values from a std::vector to a Core::LinAlg::MultiVector<double>
   *
   *  The data layout in the std::vector is consecutively ordered. The
   * Core::LinAlg::MultiVector<double> consists of several single vectors put together after each
   * other.
   *
   *  \param(in) stdVector:         A std::vector<double> to read data from.
   *  \param(in) epetraMultiVector: A Core::LinAlg::MultiVector<double> to write data to.
   *  \param(in) blockSize:         Block size of the Core::LinAlg::MultiVector<double>.
   */
  void std_vector_to_epetra_multi_vector(const std::vector<double>& stdVector,
      Core::LinAlg::MultiVector<double>& epetraMultiVector, const int blockSize);

  /*! \brief Write values from a std::vector to a Core::LinAlg::MultiVector<double>
   *
   *  The data layout in the std::vector is consecutively ordered. The
   * Core::LinAlg::MultiVector<double> consists of several single vectors put together after each
   * other.
   *
   *  \param(in) epetraMultiVector: A Core::LinAlg::MultiVector<double> to read data from.
   *  \param(in) stdVector:         A std::vector<double> to read data to.
   *  \param(in) blockSize:         Block size of the Core::LinAlg::MultiVector<double>.
   */
  void epetra_multi_vector_to_std_vector(const Core::LinAlg::MultiVector<double>& epetraMultiVector,
      std::vector<double>& stdVector, const int blockSize);


}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
