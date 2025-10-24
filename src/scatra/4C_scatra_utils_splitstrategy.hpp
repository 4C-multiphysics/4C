// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_UTILS_SPLITSTRATEGY_HPP
#define FOUR_C_SCATRA_UTILS_SPLITSTRATEGY_HPP

#include "4C_config.hpp"

#include "4C_linalg_blocksparsematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace ScaTra
{
  /*!
  \brief concentrations-el.potential split strategy

   */
  class SplitStrategy
  {
   public:
    /// construct with a block matrix base
    explicit SplitStrategy(Core::LinAlg::BlockSparseMatrixBase& mat)
        : mat_(mat),
          matrix00_(mat_.matrix(0, 0)),
          matrix01_(mat_.matrix(0, 1)),
          matrix10_(mat_.matrix(1, 0)),
          matrix11_(mat_.matrix(1, 1)),
          numscal_(-1),
          numdofpernode_(-1)
    {
    }

    /// find row block to a given row gid
    int row_block(int lrow, int rgid)
    {
      if ((lrow % numdofpernode_) < numscal_) return 0;
      return 1;
    }

    /// find column block to a given column gid
    int col_block(int rblock, int lcol, int cgid)
    {
      if ((lcol % numdofpernode_) < numscal_) return 0;
      return 1;
    }

    /// assemble into the given block
    /* For Electrochemistry-applications the upper-left block matrix
     * A00 obeys a sparse block-diagonal substructure. We implement here a special
     * assembly strategy in order to avoid assembling zeros that make up at
     * least 22% of the entries in the element matrix.
     * The well-known VelPresSplitStrategy formerly used for this purpose
     * shows this drawback.
     * Effects: Faster assembly and a more sparse global matrix (and graph)
     */
    void assemble(int eid, int myrank, const std::vector<int>& lmstride,
        const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
        const std::vector<int>& lmrowowner, const std::vector<int>& lmcol)
    {
      const int lrowdim = (int)lmrow.size();
      const int lcoldim = (int)lmcol.size();

      if (mat_.filled())
      {
        // We use the maps of the matrix to gain fast access to the LID's.
        // Assembling with SumIntoMyValues based on LID's is two times faster
        // than inserting single values based on the global row and column ids.

        // There is the case of nodes without dofs (XFEM).
        // If no row dofs are present on this proc, their is nothing to assemble.
        // However, the subsequent check for coldofs (in DEBUG mode) would incorrectly fail.
        bool doit = false;
        for (int lrow = 0; lrow < lrowdim; ++lrow)
          if (lmrowowner[lrow] == myrank)
          {
            doit = true;  // This proc owns at least one row of interest
            break;
          }
        if (!doit) return;

        // get the maps
        const Core::LinAlg::Map& colmap00 = mat_.matrix(0, 0).col_map();
        const Core::LinAlg::Map& colmap01 = mat_.matrix(0, 1).col_map();
        const Core::LinAlg::Map& colmap10 = mat_.matrix(1, 0).col_map();
        const Core::LinAlg::Map& colmap11 = mat_.matrix(1, 1).col_map();
        const Core::LinAlg::Map& rowmap00 = mat_.matrix(0, 0).row_map();
        const Core::LinAlg::Map& rowmap01 = mat_.matrix(0, 1).row_map();
        const Core::LinAlg::Map& rowmap10 = mat_.matrix(1, 0).row_map();
        const Core::LinAlg::Map& rowmap11 = mat_.matrix(1, 1).row_map();

        // prepare vectors for holding column local ids and the values to be assembled
        const int nnode = lcoldim / numdofpernode_;
        std::vector<double> values0(numscal_ * nnode);
        std::vector<double> values1(nnode);
        std::vector<double> values00(nnode);
        std::map<int, std::vector<int>> localcol00map;
        for (int i = 0; i < numscal_; i++)
        {
          std::vector<int> vec(nnode);
          localcol00map.insert(std::pair<int, std::vector<int>>(i, vec));
        }
        std::vector<int> localcol01(nnode);
        std::vector<int> localcol10(numscal_ * nnode);
        std::vector<int> localcol11(nnode);

        // fill vectors with the column map LID's
        for (int inode = 0; inode < nnode; ++inode)
        {
          const int p = inode * numdofpernode_;
          // the concentrations
          for (int sclarid = 0; sclarid < numscal_; ++sclarid)
          {
            const int lcol = p + sclarid;
            const int cgid = lmcol[lcol];
            (localcol00map[sclarid])[inode] = (colmap00.lid(cgid));
            const int pos = inode * numscal_ + sclarid;
            localcol10[pos] = (colmap10.lid(cgid));
          }
          // the el. potential
          const int lcol = p + numscal_;
          const int cgid = lmcol[lcol];
          localcol01[inode] = (colmap01.lid(cgid));
          localcol11[inode] = (colmap11.lid(cgid));
        }

        int rlid0;
        int rlid1;

        // loop rows of local matrix and assemble each of them
        for (int lrow = 0; lrow < lrowdim; ++lrow)
        {
          // check ownership of row
          if (lmrowowner[lrow] != myrank) continue;

          // get global id and block number for current row
          const int rgid = lmrow[lrow];
          const int scalarid = lrow % numdofpernode_;

          if (scalarid < numscal_)
          {
            // the current row belongs to the transport equation of species 'scalarid'
            rlid0 = rowmap00.lid(rgid);
            rlid1 = rowmap01.lid(rgid);
#ifdef FOUR_C_ENABLE_ASSERTIONS
            if (rlid0 < 0) FOUR_C_THROW("Sparse matrix A00 does not have global row {}", rgid);
            if (rlid1 < 0) FOUR_C_THROW("Sparse matrix A01 does not have global row {}", rgid);
#endif

            // separate the (non-zero!) values of the current row
            for (int j = 0; j < nnode; ++j)
            {
              int lcol = j * numdofpernode_ + scalarid;
              double val = Aele(lrow, lcol);
              values00[j] = val;

              lcol = j * numdofpernode_ + numscal_;
              val = Aele(lrow, lcol);
              values1[j] = val;
            }

            // assemble
            matrix00_.sum_into_my_values(
                rlid0, nnode, values00.data(), localcol00map[scalarid].data());
            matrix01_.sum_into_my_values(rlid1, nnode, values1.data(), localcol01.data());
          }
          else
          {
            // the current row belongs to the equation for the electric potential
            rlid0 = rowmap10.lid(rgid);
            rlid1 = rowmap11.lid(rgid);
#ifdef FOUR_C_ENABLE_ASSERTIONS
            if (rlid0 < 0) FOUR_C_THROW("Sparse matrix A10 does not have global row {}", rgid);
            if (rlid1 < 0) FOUR_C_THROW("Sparse matrix A11 does not have global row {}", rgid);
#endif
            // separate the values of the current row
            int nodespassed = 0;
            for (int lcol = 0; lcol < lcoldim; ++lcol)
            {
              double val = Aele(lrow, lcol);
              const int rest = lcol % numdofpernode_;
              if (rest < numscal_)
              {
                int pos = nodespassed * numscal_ + rest;
                values0[pos] = val;
              }
              else
              {
                values1[nodespassed] = val;
                nodespassed++;
              }
            }

            // assemble
            matrix10_.sum_into_my_values(
                rlid0, nnode * numscal_, values0.data(), localcol10.data());
            matrix11_.sum_into_my_values(rlid1, nnode, values1.data(), localcol11.data());
          }
        }  // for (int lrow=0; lrow<ldim; ++lrow)
      }
      else
      {
        // the maps of the blockmatrices are not yet available; use global ids instead

        // loop rows of local matrix
        for (int lrow = 0; lrow < lrowdim; ++lrow)
        {
          // check ownership of row
          if (lmrowowner[lrow] != myrank) continue;

          int rgid = lmrow[lrow];
          int rblock = row_block(lrow, rgid);
          const int scalarid = lrow % numdofpernode_;

          if (scalarid < numscal_)
          {
            // special treatment for block matrix A00:
            for (int lcol = scalarid; lcol < lcoldim; lcol += numdofpernode_)
            {
              double val = Aele(lrow, lcol);
              int cgid = lmcol[lcol];
              int cblock = col_block(rblock, lcol, cgid);

              Core::LinAlg::SparseMatrix& matrix = mat_.matrix(rblock, cblock);
              matrix.assemble(val, rgid, cgid);
            }
            // values for block matrix A01:
            for (int lcol = numscal_; lcol < lcoldim; lcol += numdofpernode_)
            {
              double val = Aele(lrow, lcol);
              int cgid = lmcol[lcol];
              int cblock = col_block(rblock, lcol, cgid);

              Core::LinAlg::SparseMatrix& matrix = mat_.matrix(rblock, cblock);
              matrix.assemble(val, rgid, cgid);
            }
          }
          else  // rblock == 1, assemble everything
          {
            for (int lcol = 0; lcol < lcoldim; ++lcol)
            {
              double val = Aele(lrow, lcol);
              int cgid = lmcol[lcol];
              int cblock = col_block(rblock, lcol, cgid);

              Core::LinAlg::SparseMatrix& matrix = mat_.matrix(rblock, cblock);
              matrix.assemble(val, rgid, cgid);
            }
          }
        }
      }
    }

    /// assemble into the given block
    void assemble(double val, int rgid, int cgid)
    {
      int rblock = row_block(0, rgid);
      int cblock = col_block(rblock, 0, cgid);
      Core::LinAlg::SparseMatrix& matrix = mat_.matrix(rblock, cblock);
      matrix.assemble(val, rgid, cgid);
    }

    /// assemble the remaining ghost entries
    void complete() {}

    /// set number of concentration dofs
    void set_num_scal(const int numscal)
    {
      numscal_ = numscal;
      numdofpernode_ = numscal + 1;
    }

   private:
    /// my block matrix base
    Core::LinAlg::BlockSparseMatrixBase& mat_;

    // the four sub-matrices of the whole matrix
    Core::LinAlg::SparseMatrix& matrix00_;
    Core::LinAlg::SparseMatrix& matrix01_;
    Core::LinAlg::SparseMatrix& matrix10_;
    Core::LinAlg::SparseMatrix& matrix11_;

    /// number of concentration dofs
    int numscal_;

    /// number of dofs per node (= numdim_ +1)
    int numdofpernode_;

  };  // class SplitStrategy

}  // namespace ScaTra

FOUR_C_NAMESPACE_CLOSE

#endif
