// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_INTERFACE_JACOBIAN_BASE_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_INTERFACE_JACOBIAN_BASE_HPP

#include "4C_config.hpp"

#include "4C_linalg_sparseoperator.hpp"

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    namespace Interface
    {
      class JacobianBase
      {
       public:
        JacobianBase() = default;

        virtual ~JacobianBase() = default;

        /*! Compute Jacobian given the specified input vector x.
         * Returns true if computation was successful.
         */
        [[nodiscard]] virtual bool compute_jacobian(
            const Core::LinAlg::Vector<double>& x, Core::LinAlg::SparseOperator& jac) = 0;
      };
    }  // namespace Interface
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
