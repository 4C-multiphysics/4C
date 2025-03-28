// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_FIXEDSIZEBLOCKMATRIX_HPP
#define FOUR_C_LINALG_FIXEDSIZEBLOCKMATRIX_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  template <class ValueType, unsigned int brows, unsigned int bcols>
  class BlockMatrix
  {
   public:
    typedef typename ValueType::scalar_type scalar_type;

    BlockMatrix() { std::fill(blocks_, blocks_ + brows * bcols, static_cast<ValueType*>(nullptr)); }

    ~BlockMatrix()
    {
      for (unsigned i = 0; i < brows * bcols; ++i)
      {
        delete blocks_[i];
      }
    }

    bool is_used(unsigned row, unsigned col) const
    {
      return blocks_[position(row, col)] != nullptr;
    }

    bool is_used(unsigned pos) const { return blocks_[pos] != nullptr; }

    void clear(unsigned row, unsigned col)
    {
      int p = position(row, col);
      delete blocks_[p];
      blocks_[p] = nullptr;
    }

    void add_view(unsigned row, unsigned col, ValueType& matrix)
    {
      clear(row, col);
      int p = position(row, col);
      blocks_[p] = new ValueType(matrix, true);
    }

    ValueType* operator()(unsigned row, unsigned col)
    {
      int p = position(row, col);
      ValueType* b = blocks_[p];
      if (b == nullptr)
      {
        b = new ValueType(Core::LinAlg::Initialization::zero);
        blocks_[p] = b;
      }
      return b;
    }

    const ValueType* operator()(unsigned row, unsigned col) const
    {
      const ValueType* b = blocks_[position(row, col)];
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (b == nullptr) FOUR_C_THROW("null block access");
#endif
      return b;
    }

    ValueType* operator()(unsigned pos)
    {
      ValueType* b = blocks_[pos];
      if (b == nullptr)
      {
        b = new ValueType(true);
        blocks_[pos] = b;
      }
      return b;
    }

    const ValueType* operator()(unsigned pos) const
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (pos >= brows * bcols) FOUR_C_THROW("block index out of range");
#endif
      const ValueType* b = blocks_[pos];
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (b == nullptr) FOUR_C_THROW("null block access");
#endif
      return b;
    }

    template <class Lhs, class Rhs, unsigned int inner>
    inline void multiply(
        const BlockMatrix<Lhs, brows, inner>& left, const BlockMatrix<Rhs, inner, bcols>& right)
    {
      put_scalar(0.);
      for (unsigned int ic = 0; ic < bcols; ++ic)
      {
        for (unsigned int i = 0; i < inner; ++i)
        {
          if (right.is_used(i, ic))
          {
            for (unsigned int ir = 0; ir < brows; ++ir)
            {
              if (left.is_used(ir, i))
              {
                ValueType* b = (*this)(ir, ic);
                b->multiply(1, *left(ir, i), *right(i, ic), 1);
              }
            }
          }
        }
      }
    }

    void put_scalar(scalar_type scalar)
    {
      for (unsigned i = 0; i < brows * bcols; ++i)
      {
        if (blocks_[i] != nullptr)
        {
          blocks_[i]->put_scalar(scalar);
        }
      }
    }

    template <int rows, int cols>
    void assemble_to(Core::LinAlg::Matrix<rows, cols, scalar_type>& dest, double f)
    {
      const int local_rows = rows / brows;
      const int local_cols = cols / bcols;
      const unsigned row_blocks = brows;
      const unsigned col_blocks = bcols;
      for (unsigned icb = 0; icb < col_blocks; ++icb)
      {
        for (unsigned irb = 0; irb < row_blocks; ++irb)
        {
          if (is_used(irb, icb))
          {
            Core::LinAlg::Matrix<local_rows, local_cols>& local = *(*this)(irb, icb);
            for (int ic = 0; ic < local_cols; ++ic)
            {
              unsigned c = col_blocks * ic + icb;
              for (int ir = 0; ir < local_rows; ++ir)
              {
                unsigned r = row_blocks * ir + irb;
                dest(r, c) += f * local(ir, ic);
              }
            }
          }
        }
      }
    }

    friend std::ostream& operator<<(
        std::ostream& stream, const BlockMatrix<ValueType, brows, bcols>& matrix)
    {
      stream << "BlockMatrix<" << brows << "," << bcols << ">\n";
      for (unsigned int ir = 0; ir < brows; ++ir)
      {
        for (unsigned int ic = 0; ic < bcols; ++ic)
        {
          int p = matrix.position(ir, ic);
          ValueType* b = matrix.blocks_[p];
          stream << "[" << ir << "," << ic << "] = ";
          if (b == nullptr)
          {
            stream << "nullptr\n";
          }
          else
          {
            stream << (*b);
          }
        }
      }
      return stream;
    }

   private:
    int position(unsigned row, unsigned col) const
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (row >= brows or col >= bcols) FOUR_C_THROW("block index out of range");
#endif
      return row + brows * col;
    }

    ValueType* blocks_[brows * bcols];
  };

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
