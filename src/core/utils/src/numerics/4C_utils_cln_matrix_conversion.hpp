#ifndef FOUR_C_UTILS_CLN_MATRIX_CONVERSION_HPP
#define FOUR_C_UTILS_CLN_MATRIX_CONVERSION_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_utils_clnwrapper.hpp"
#include "4C_utils_exceptions.hpp"

#include <cstddef>
#include <sstream>
#include <unordered_map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::CLN
{
  //! convert LINALG matrix with CLN values to LINALG matrix with double values
  template <unsigned int num_row, unsigned int num_col>
  void conv_cln_double(const LinAlg::Matrix<num_row, num_col, ClnWrapper>& in,
      LinAlg::Matrix<num_row, num_col, double>& out)
  {
    for (unsigned int idx = 0; idx < in.num_rows() * in.num_cols(); ++idx)
    {
      out.data()[idx] = cln::double_approx(in.data()[idx].Value());
    }
  }

  //! convert LINALG matrix with double values to LINALG matrix with CLN values
  template <unsigned int num_row, unsigned int num_col>
  void conv_double_cln(const LinAlg::Matrix<num_row, num_col, double>& in,
      LinAlg::Matrix<num_row, num_col, ClnWrapper>& out, const int precision = 20)
  {
    for (unsigned int idx = 0; idx < in.num_rows() * in.num_cols(); ++idx)
    {
      ClnWrapper clnnum;
      // zeros do not convert properly to CLN (loss of precision)
      if ((in.data()[idx] == 0.0))
      {
        // returning the cached value from the ClnWrapper cln table
        clnnum = 0.0;
      }
      else
        clnnum = cln::cl_float(in.data()[idx], cln::float_format(precision));
      out.data()[idx] = clnnum;
    }
  }

  //! convert LINALG matrix with CLN values to a new LINALG matrix with CLN values with different
  //! precision
  template <unsigned int num_row, unsigned int num_col>
  void update_presicion(const LinAlg::Matrix<num_row, num_col, ClnWrapper>& in,
      LinAlg::Matrix<num_row, num_col, ClnWrapper>& out, const int precision = 20)
  {
    for (unsigned int idx = 0; idx < in.num_rows() * in.num_cols(); ++idx)
    {
      ClnWrapper clnnum;
      // zeros do not convert properly to CLN (loss of precision)
      if ((in.data()[idx] == 0.0))
      {
        // returning the cached value from the ClnWrapper cln table
        clnnum = 0.0;
      }
      else
        clnnum = cln::cl_float(in.data()[idx].Value(), cln::float_format(precision));
      out.data()[idx] = clnnum;
    }
  }

}  // namespace Core::CLN

FOUR_C_NAMESPACE_CLOSE

#endif
