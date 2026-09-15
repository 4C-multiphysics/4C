// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_INNER_STATUSTEST_GENERIC_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_INNER_STATUSTEST_GENERIC_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_forward_decl.hpp"

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
        namespace Interface
        {
          class Required;
        }  // namespace Interface

        enum StatusType : int
        {
          //! Unevaluated
          status_unevaluated = -4,
          //! Failed
          status_failed = -3,
          // No descent direction
          status_no_descent_direction = -2,
          //! Step too short
          status_step_too_short = -1,
          //! Step too long
          status_step_too_long = 0,
          //! Converged
          status_converged = 1
        };

        inline std::string status_type_to_string(StatusType status)
        {
          switch (status)
          {
            case status_unevaluated:
              return "status_unevaluated";
            case status_failed:
              return "status_failed";
            case status_no_descent_direction:
              return "status_no_descent_direction";
            case status_step_too_short:
              return "status_step_too_short";
            case status_step_too_long:
              return "status_step_too_long";
            case status_converged:
              return "status_converged";
            default:
              return "Unknown NOX:NLN:Inner::StatusTest::StatusType";
          }
        }

        class Generic
        {
         public:
          //! constructor
          Generic() {};

          //! destructor
          virtual ~Generic() = default;

          /** \brief %Test the inner stopping criterion
           *
           *  The test can (and should, if possible) be skipped if
           *  checkType is NOX::StatusType::None.  If the test is skipped, then
           *  the status should be set to ::NOX::StatusTest::Unevaluated. */
          [[nodiscard]] virtual StatusType check_status(const Interface::Required& interface,
              const ::NOX::Solver::Generic& solver, const ::NOX::Abstract::Group& grp,
              ::NOX::StatusTest::CheckType checkType) = 0;

          //! Return the result of the most recent inner checkStatus call
          [[nodiscard]] virtual StatusType get_status() const = 0;

          //! Output formatted description of inner stopping test to output stream.
          virtual std::ostream& print(std::ostream& stream, int indent = 0) const = 0;
        };

        // non-member function
        std::ostream& operator<<(std::ostream& os, StatusType type);
      }  // namespace StatusTest
    }  // namespace Inner
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
