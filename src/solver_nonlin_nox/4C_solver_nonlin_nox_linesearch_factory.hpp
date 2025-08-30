// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_LINESEARCH_FACTORY_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_LINESEARCH_FACTORY_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_forward_decl.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    namespace Inner
    {
      namespace StatusTest
      {
        class Generic;
      }  // namespace StatusTest
    }  // namespace Inner
    namespace LineSearch
    {
      class Factory
      {
       public:
        //! Constructor
        Factory();


        /*! \brief Factory to build a line search object.

            @param gd A global data pointer that contains the top level parameter list.  Without
           storing this inside the line searchobject, there is no guarantee that the second
           parameter \c params will still exist.  It can be deleted by the top level RCP.
            @param params General nln parameterlist.

        */
        Teuchos::RCP<::NOX::LineSearch::Generic> build_line_search(
            const Teuchos::RCP<::NOX::GlobalData>& gd,
            const Teuchos::RCP<::NOX::StatusTest::Generic> outerTests,
            const Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic> innerTests,
            Teuchos::ParameterList& params) const;

       private:
        // checks if the inner status test pointer is initialized
        void inner_status_test_is_required(
            const Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic>& innerTests) const;
      };
      /*! Nonmember function to build a line search object.

      \relates NOX::NlnSol::Constraint::LineSearch::Factory

      */
      Teuchos::RCP<::NOX::LineSearch::Generic> build_line_search(
          const Teuchos::RCP<::NOX::GlobalData>& gd,
          const Teuchos::RCP<::NOX::StatusTest::Generic> outerTests,
          const Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic> innerTests,
          Teuchos::ParameterList& params);

    }  // namespace LineSearch
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
