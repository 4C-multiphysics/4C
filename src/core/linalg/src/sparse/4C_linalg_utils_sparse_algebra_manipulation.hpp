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
#include "4C_linalg_map.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_vector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /*!
   * \brief Communicate a vector to a different map
   *
   * Values of source are copied to target where maps don't have to match.
   * Prerequisite: Either the map of source OR the map of target has to be unique
   * (will be tested)
   * \warning When source is overlapping (and therefore target is unique), values
   * in the overlapping region are inserted into the target on a first come
   * first serve basis, meaning they should be equal in the source to
   * be deterministic
   * \param source (in) : source vector values are taken from
   * \param target (out): target vector values will be inserted in
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
   *  */
  std::unique_ptr<Core::LinAlg::Vector<double>> extract_my_vector(
      const Core::LinAlg::Vector<double>& source, const Core::LinAlg::Map& target_map);

  /*! \brief Extract a partial Eptra_Vector from a given source vector
   *         on each proc without communication
   *
   *  \param source (in) : source vector ( read-only )
   *  \param target (out): this target vector is going to be filled
   *
   *  */
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
   * \brief split a matrix into a 2x2 block system
   *
   * Splits a given matrix into a 2x2 block system. All values on entry have to be
   * nullptr except the given rowmap(s) / domainmap(s) and matrix A.
   * Note that either A11rowmap or A22rowmap or both have to be nonzero!
   * Note that either A11domainmap or A22domainmap or both have to be nonzero!
   * In case both rowmaps / domainmaps are supplied they have to be an exact and
   * nonoverlapping split of A->RowMap() / A->DomainMap().
   * Matrix blocks are fill_complete() on exit.
   *
   * \param A            : Matrix A on input
   * \param A11rowmap    : rowmap of A11 or null
   * \param A22rowmap    : rowmap of A22 or null
   * \param A11domainmap : domainmap of A11 or null
   * \param A22domainmap : domainmap of A22 or null
   * \param A11          : on exit matrix block A11
   * \param A12          : on exit matrix block A12
   * \param A21          : on exit matrix block A21
   * \param A22          : on exit matrix block A22
   */
  bool split_matrix2x2(std::shared_ptr<Core::LinAlg::SparseMatrix> A,
      std::shared_ptr<Core::LinAlg::Map>& A11rowmap, std::shared_ptr<Core::LinAlg::Map>& A22rowmap,
      std::shared_ptr<Core::LinAlg::Map>& A11domainmap,
      std::shared_ptr<Core::LinAlg::Map>& A22domainmap,
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

  /*!
   * \brief Split matrix in either 2x2 or NxN blocks (with N>2)
   *
   * Split given sparse matrix into 2x2 or NxN block matrix and return result as templated
   * BlockSparseMatrix. The MultiMapExtractor's provided have to be 2x2 or NxN maps, otherwise
   * this method will throw an error.
   *
   * \warning This is an expensive operation!
   *
   * \note This method will NOT call Complete() on the output BlockSparseMatrix.
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
          "Invalid number {} of row blocks or {} of column blocks for splitting operation!",
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
   *  */
  int insert_my_row_diagonal_into_unfilled_matrix(
      Core::LinAlg::SparseMatrix& mat, const Core::LinAlg::Vector<double>& diag);

  /*!
   * \brief Split an Core::LinAlg::Map and return the part complementary to \c Agiven
   *
   * Splits \c Amap into 2 maps, where one is given on input and the other map
   * is created as complementary map. The complementary map is returned.
   *
   * \param[in] Amap      : Map to split on input
   * \param[in] Agiven    : on entry submap that is given and part of Amap
   * \return the remainder map of Amap that is not overlapping with Agiven
   */
  std::shared_ptr<Core::LinAlg::Map> split_map(
      const Core::LinAlg::Map& Amap, const Core::LinAlg::Map& Agiven);

  /*!
   * \brief Merge two given Core::LinAlg::Maps into one
   *
   * Merges input map1 and input map2, both of which have to be unique,
   * but may be overlapping, to a new map and returns std::shared_ptr to it.
   *
   * \param map1         : one map to be merged
   * \param map2         : the other map to be merged
   * \param allowoverlap : when set to false, an error is thrown if the result
   * map is overlapping (default = true, overlap allowed)
   * \return the (sorted) merged map of input maps map1 and map2
   */
  std::shared_ptr<Core::LinAlg::Map> merge_map(
      const Core::LinAlg::Map& map1, const Core::LinAlg::Map& map2, bool overlap = true);

  /*!
   * \brief find the intersection set of two given Core::LinAlg::Maps
   *
   * Find the insection set of input map1 and input map2.
   *
   * \param map1         : first map
   * \param map2         : second map
   * \return the (sorted) intersection map of input maps map1 and map2
   */
  std::shared_ptr<Core::LinAlg::Map> intersect_map(
      const Core::LinAlg::Map& map1, const Core::LinAlg::Map& map2);


  /*!
   * \brief merges two given Core::LinAlg::Maps
   *
   * merges input map1 and input map2 (given as std::shared_ptr), both of which
   * have to be unique, but may be overlapping, to a new map and returns
   * std::shared_ptr to it. The case that one or both input std::shared_ptrs are null is
   * detected and handled appropriately.
   *
   * \param map1         : one map to be merged
   * \param map2         : the other map to be merged
   * \param allowoverlap : when set to false, an error is thrown if the result map is overlapping
   * (default = true, overlap allowed)
   * \return the (sorted) merged map of input maps map1 and map2
   */
  std::shared_ptr<Core::LinAlg::Map> merge_map(const std::shared_ptr<const Core::LinAlg::Map>& map1,
      const std::shared_ptr<const Core::LinAlg::Map>& map2, bool overlap = true);

  /*!
   * \brief split a vector into 2 non-overlapping pieces (std::shared_ptr version)
   *
   * \param xmap    : map of vector to be split
   * \param x       : vector to be split
   * \param x1map   : map of first vector to be extracted
   * \param x1      : first vector to be extracted
   * \param x2map   : map of second vector to be extracted
   * \param x2      : second vector to be extracted
   */
  bool split_vector(const Core::LinAlg::Map& xmap, const Core::LinAlg::Vector<double>& x,
      std::shared_ptr<Core::LinAlg::Map>& x1map, std::shared_ptr<Core::LinAlg::Vector<double>>& x1,
      std::shared_ptr<Core::LinAlg::Map>& x2map, std::shared_ptr<Core::LinAlg::Vector<double>>& x2);

  /*!
   * \brief split a vector into 2 non-overlapping pieces (std::shared_ptr version)
   *
   * \param xmap    : map of vector to be split
   * \param x       : vector to be split
   * \param x1map   : map of first vector to be extracted
   * \param x1      : first vector to be extracted
   * \param x2map   : map of second vector to be extracted
   * \param x2      : second vector to be extracted
   */
  bool split_vector(const Core::LinAlg::Map& xmap, const Core::LinAlg::Vector<double>& x,
      std::shared_ptr<const Core::LinAlg::Map>& x1map,
      std::shared_ptr<Core::LinAlg::Vector<double>>& x1,
      std::shared_ptr<const Core::LinAlg::Map>& x2map,
      std::shared_ptr<Core::LinAlg::Vector<double>>& x2);

  /*! \brief Write values from a std::vector to a Core::LinAlg::MultiVector<double>
   *
   *  The data layout in the std::vector is consecutively ordered. The
   * Core::LinAlg::MultiVector<double> consists of several single vectors put together after each
   * other.
   *
   *  \param(in) std_vector:   A std::vector<double> to read data from.
   *  \param(in) multi_vector: A Core::LinAlg::MultiVector<double> to write data to.
   *  \param(in) block_size:   Block size of the Core::LinAlg::MultiVector<double>.
   */
  void std_vector_to_multi_vector(const std::vector<double>& std_vector,
      Core::LinAlg::MultiVector<double>& multi_vector, const int block_size);

  /*! \brief Write values from a std::vector to a Core::LinAlg::MultiVector<double>
   *
   *  The data layout in the std::vector is consecutively ordered. The
   * Core::LinAlg::MultiVector<double> consists of several single vectors put together after each
   * other.
   *
   *  \param(in) multi_vector: A Core::LinAlg::MultiVector<double> to read data from.
   *  \param(in) std_vector:   A std::vector<double> to read data to.
   *  \param(in) block_size:   Block size of the Core::LinAlg::MultiVector<double>.
   */
  void multi_vector_to_std_vector(const Core::LinAlg::MultiVector<double>& multi_vector,
      std::vector<double>& std_vector, const int block_size);

  /*! \brief  Transform the row map of a matrix (parallel distribution)
   *
   *
   * This method changes the row map of an input matrix to new row map with identical GIDs but
   * different parallel distribution.
   *
   * \param(in) inmat:  A Core::LinAlg::SparseMatrix that needs to be transformed.
   * \param(in) newrowmap: A Core::LinAlg::Map providing the rowmap of the new distribution.
   *
   * \returns A Core::LinAlg::SparseMatrix with the rowmap given by newrowmap
   */
  std::shared_ptr<Core::LinAlg::SparseMatrix> matrix_row_transform(
      const Core::LinAlg::SparseMatrix& inmat, const Core::LinAlg::Map& newrowmap);

  /*! \brief  Transform the column map of a matrix (parallel distribution)
   *
   *
   *  This method changes the column map of an input matrix to new column map with identical GIDs
   * but different parallel distribution. (and the domain map, accordingly).
   *
   *  \param(in) inmat:  A Core::LinAlg::SparseMatrix that needs to be transformed.
   *  \param(in) newrowmap: A Core::LinAlg::Map providing the rowmap of the new distribution.
   *
   *  \returns A Core::LinAlg::SparseMatrix with the domainmap given by newdomainmap
   */
  std::shared_ptr<Core::LinAlg::SparseMatrix> matrix_col_transform(
      const Core::LinAlg::SparseMatrix& inmat, const Core::LinAlg::Map& newdomainmap);

  /*! \brief  Transform the row and column maps of a matrix (parallel distribution)
   *
   *
   * This method changes the row and column maps of an input matrix to new row and column maps with
   * identical GIDs but different parallel distribution (and the domain map, accordingly).
   *
   * \param(in) inmat:  A Core::LinAlg::SparseMatrix that needs to be transformed.
   * \param(in) newrowmap: A Core::LinAlg::Map providing the rowmap of the new distribution.
   * \param(in) newdomainmap: A Core::LinAlg::Map providing the domainmap of the new distribution.
   *
   * \returns A Core::LinAlg::SparseMatrix with the rowmap given by newrowmap and domainmap
   * given by newdomainmap
   */
  std::shared_ptr<Core::LinAlg::SparseMatrix> matrix_row_col_transform(
      const Core::LinAlg::SparseMatrix& inmat, const Core::LinAlg::Map& newrowmap,
      const Core::LinAlg::Map& newdomainmap);

  /*! \brief Parallel redistribution of a sparse matrix
   *  Helper method for the MatrixTransform() methods above.
   */
  std::shared_ptr<Core::LinAlg::SparseMatrix> redistribute(const Core::LinAlg::SparseMatrix& src,
      const Core::LinAlg::Map& permrowmap, const Core::LinAlg::Map& permdomainmap);

  /*!
   * \brief Transform the row map of a matrix (only GIDs)
   *
   * This method changes the row map of an input matrix to a new row map
   * with different GID numbering. However, the parallel distribution of
   * this new row map is exactly the same as in the old row map. Thus, this
   * is simply a processor-local 1:1 matching of old and new GIDs.
   *
   * @param inmat Matrix on which the row and column maps will be transformed
   * @param newrowmap New row map used for the given input matrix
   *
   * \post Output matrix will be fill_complete()
   */
  std::shared_ptr<Core::LinAlg::SparseMatrix> matrix_row_transform_gids(
      const Core::LinAlg::SparseMatrix& inmat, const Core::LinAlg::Map& newrowmap);

  /*!
   * \brief Transform the column map of a matrix (only GIDs)
   *
   * This method changes the column map of an input matrix to a new column
   * map with different GID numbering (and the domain map, accordingly).
   * However, the parallel distribution of the new domain map is exactly
   * the same as in the old domain map. Thus, this is simply a processor-local
   * 1:1 matching of old and new GIDs.
   *
   * @param inmat Matrix on which the row and column maps will be transformed
   * @param newdomainmap New domain map used for the given input matrix, which will indirectly
   * transform the column map of the given input matrix
   *
   * \post Output matrix will be fill_complete()
   */
  std::shared_ptr<Core::LinAlg::SparseMatrix> matrix_col_transform_gids(
      const Core::LinAlg::SparseMatrix& inmat, const Core::LinAlg::Map& newdomainmap);


  /*!
   * \brief Transform the row and column maps of a matrix (only GIDs)
   *
   * This method changes the row and column maps of an input matrix to new
   * row and column maps with different GID numbering (and the domain map,
   * accordingly). However, the parallel distribution of the new row and
   * domain maps is exactly the same as in the old ones. Thus, this is simply
   * a processor-local 1:1 matching of old and new GIDs.
   *
   * @param inmat Matrix on which the row and column maps will be transformed
   * @param newrowmap New row map used for the given input matrix
   * @param newdomainmap New domain map used for the given input matrix, which will indirectly
   * transform the column map of the given input matrix
   *
   * \post Output matrix will be fill_complete()
   */
  std::shared_ptr<Core::LinAlg::SparseMatrix> matrix_row_col_transform_gids(
      const Core::LinAlg::SparseMatrix& inmat, const Core::LinAlg::Map& newrowmap,
      const Core::LinAlg::Map& newdomainmap);


}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
