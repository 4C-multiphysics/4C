// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_LINESEARCH_BACKTRACK_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_LINESEARCH_BACKTRACK_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_floating_point_exception.hpp"
#include "4C_solver_nonlin_nox_forward_decl.hpp"
#include "4C_solver_nonlin_nox_inner_statustest_generic.hpp"
#include "4C_solver_nonlin_nox_linesearch_generic.hpp"  // base class

#include <NOX_StatusTest_Generic.H>

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
      class Backtrack : public Generic
      {
       public:
        //! Constructor
        Backtrack(const Teuchos::RCP<::NOX::GlobalData>& gd,
            const Teuchos::RCP<::NOX::StatusTest::Generic> outerTests,
            const Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic> innerTests,
            Teuchos::ParameterList& params);

        /// hard reset
        void reset(const Teuchos::RCP<::NOX::GlobalData>& gd, Teuchos::ParameterList& params);

        /// weak reset
        void reset();

        bool compute(::NOX::Abstract::Group& newgrp, double& step,
            const ::NOX::Abstract::Vector& dir, const ::NOX::Solver::Generic& s) override;

        [[nodiscard]] NOX::Nln::Inner::StatusTest::StatusType check_inner_status(
            const ::NOX::Solver::Generic& solver, const ::NOX::Abstract::Group& grp,
            ::NOX::StatusTest::CheckType checkType) const override;

        //! @name Access functionality
        //@{
        //! get the number of line search iterations
        int get_num_iterations() const override;

        //! get the merit function
        const ::NOX::MeritFunction::Generic& get_merit_function() const override;

        //! get the current search direction
        const ::NOX::Abstract::Vector& get_search_direction() const override;

        //! get current step length
        double get_step_length() const override;

        //!@}

        //! @name Mutator functionality
        //! @{
        //! set current step length
        void set_step_length(double step) override;
        //! @}

       protected:
        //! print inner status test results
        void print_update(std::ostream& os) const;

       private:
        //! throw NOX error
        void throw_error(const std::string& functionName, const std::string& errorMsg) const;

       private:
        //! handle floating point exceptions
        FloatingPointException fp_except_;

        //! inner iteration counter
        int ls_iters_;

        //! Printing Utilities
        Teuchos::RCP<::NOX::Utils> utils_;

        //! Merit function
        Teuchos::RCP<::NOX::MeritFunction::Generic> merit_function_ptr_;

        //! Current step pointer, points to the step variable of the
        //! NOX::Nln::Solver::LineSearchBased class
        double* step_ptr_;

        //! Default step
        double default_step_;

        //! line search reduction factor
        double reduction_factor_;

        //! inner status test checktype
        ::NOX::StatusTest::CheckType check_type_;

        //! inner status type
        NOX::Nln::Inner::StatusTest::StatusType status_;

        //! search direction
        Teuchos::RCP<const ::NOX::Abstract::Vector> search_direction_ptr_;

        //! outer convergence tests
        Teuchos::RCP<::NOX::StatusTest::Generic> outer_tests_ptr_;

        //! line search stopping tests
        Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic> inner_tests_ptr_;
      };
    }  // namespace LineSearch
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
