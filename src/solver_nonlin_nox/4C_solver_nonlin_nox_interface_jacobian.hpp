// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_INTERFACE_JACOBIAN_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_INTERFACE_JACOBIAN_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_solver_nonlin_nox_forward_decl.hpp"
#include "4C_solver_nonlin_nox_interface_jacobian_base.hpp"

#include <Teuchos_RCPDecl.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::LinAlg
{
  class SparseMatrix;
  class SparseOperator;
}  // namespace Core::LinAlg

namespace NOX
{
  namespace Nln
  {
    namespace Interface
    {
      class Jacobian : public JacobianBase
      {
       public:
        //! Constructor.
        Jacobian() {};

        /*! \brief Compute RHS and Jacobian at once.
         *
         *  \return TRUE if computation was successful. */
        [[nodiscard]] virtual bool compute_f_and_jacobian(const Core::LinAlg::Vector<double>& x,
            Core::LinAlg::Vector<double>& rhs, Core::LinAlg::SparseOperator& jac) = 0;

        virtual Teuchos::RCP<Core::LinAlg::SparseMatrix>
        calc_jacobian_contributions_from_element_level_for_ptc() = 0;
      };
    }  // end namespace Interface
  }  // end namespace Nln
}  // end namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
