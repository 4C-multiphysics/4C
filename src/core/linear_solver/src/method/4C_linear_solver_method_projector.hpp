// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_METHOD_PROJECTOR_HPP
#define FOUR_C_LINEAR_SOLVER_METHOD_PROJECTOR_HPP

#include "4C_config.hpp"

#include "4C_io_pstream.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_box.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

#include <functional>
#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{

  /*!
   * @brief Abstract base class for linear system projectors.
   *
   * This class defines the interface for projectors that can be applied to linear systems.
   */
  class LinearSystemProjector
  {
   public:
    virtual ~LinearSystemProjector() = default;

    /*!
     * @brief Apply the projector to a sparse matrix, i.e., transforming the sparse matrix from the
     * original space to the projected space.
     *
     * @param A The input sparse matrix.
     * @return LinAlg::SparseMatrix The projected sparse matrix.
     */
    [[nodiscard]] virtual std::unique_ptr<LinAlg::SparseOperator> to_reduced(
        const LinAlg::SparseOperator& A) const = 0;

    /*!
     * @brief Apply the projector to a vector, i.e., transforming the vector from the original space
     * to the projected space.
     *
     * @param b The input vector.
     */
    [[nodiscard]] virtual Core::LinAlg::Vector<double> to_reduced(
        const LinAlg::Vector<double>& b) const = 0;

    /*!
     * @brief Apply the transpose of the projector to a vector, i.e., transforming the vector from
     * the projected space back to the original space.
     *
     * @param b The input vector.
     */
    [[nodiscard]] virtual Core::LinAlg::Vector<double> to_full(
        const LinAlg::Vector<double>& b) const = 0;
  };

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
