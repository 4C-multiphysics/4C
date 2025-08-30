// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_INNER_STATUSTEST_FACTORY_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_INNER_STATUSTEST_FACTORY_HPP

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

        class Factory
        {
         public:
          //! Constructor.
          Factory();

          //! Destructor.
          virtual ~Factory() = default;

          //! Returns a inner status test set from a parameter list.
          Teuchos::RCP<Generic> build_inner_status_tests(Teuchos::ParameterList& p,
              const ::NOX::Utils& utils,
              std::map<std::string, Teuchos::RCP<Generic>>* tagged_tests) const;

         protected:
          //! Build the Armijo sufficient decrease test
          Teuchos::RCP<Generic> build_armijo_test(
              Teuchos::ParameterList& p, const ::NOX::Utils& u) const;

          //! Build the filter test
          Teuchos::RCP<Generic> build_filter_test(
              Teuchos::ParameterList& p, const ::NOX::Utils& u) const;

          //! Build the upper bound test
          Teuchos::RCP<Generic> build_upper_bound_test(
              Teuchos::ParameterList& p, const ::NOX::Utils& u) const;

          bool check_and_tag_test(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Generic>& test,
              std::map<std::string, Teuchos::RCP<Generic>>* tagged_tests) const;

         private:
          //! Throws formatted error
          void throw_error(const std::string& functionName, const std::string& errorMsg) const;
        };

        /*! \brief Nonmember helper function for the NOX::Nln::Inner::StatusTest::Factory.

        \relates NOX::Nln::Inner::StatusTest::Factory

        */
        Teuchos::RCP<Generic> build_inner_status_tests(Teuchos::ParameterList& p,
            const ::NOX::Utils& utils,
            std::map<std::string, Teuchos::RCP<Generic>>* tagged_tests = nullptr);
      }  // namespace StatusTest
    }  // namespace Inner
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
