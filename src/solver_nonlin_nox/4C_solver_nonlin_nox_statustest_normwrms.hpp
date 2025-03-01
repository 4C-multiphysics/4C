// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_STATUSTEST_NORMWRMS_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_STATUSTEST_NORMWRMS_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_enum_lists.hpp"

#include <NOX_StatusTest_Generic.H>  // base class
#include <Teuchos_RCP.hpp>

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    namespace StatusTest
    {
      class NormWRMS : public ::NOX::StatusTest::Generic
      {
       public:
        /*! \brief Constructor
         *  At the moment we support only scalar \c ATOL input values for each quantity.
         *  Extensions to a vector-based ATOL are possible. See \c ::NOX::StatusTest::NormWRMS
         *  for more information. */
        NormWRMS(const std::vector<NOX::Nln::StatusTest::QuantityType>& checkList,
            const std::vector<double>& rtol, const std::vector<double>& atol,
            const std::vector<double>& BDFMultiplier, const std::vector<double>& tolerance,
            const double& alpha, const double& beta,
            const std::vector<bool>& disable_implicit_weighting);

        ::NOX::StatusTest::StatusType checkStatus(
            const ::NOX::Solver::Generic& problem, ::NOX::StatusTest::CheckType checkType) override;

        //! Check for the given quantity
        bool is_quantity(const NOX::Nln::StatusTest::QuantityType& qType) const;

        ::NOX::StatusTest::StatusType getStatus() const override;

        //! returns the absolute tolerance of the given quantity
        double get_absolute_tolerance(const NOX::Nln::StatusTest::QuantityType& qType) const;

        //! returns the relative tolerance of the given quantity
        double get_relative_tolerance(const NOX::Nln::StatusTest::QuantityType& qType) const;

        std::ostream& print(std::ostream& stream, int indent) const override;

       private:
        //! calculated norm for the different quantities
        Teuchos::RCP<std::vector<double>> norm_wrms_;

        //! number of quantities to check
        std::size_t n_checks_;

        //! nox_nln_statustest quantities which are checked
        std::vector<NOX::Nln::StatusTest::QuantityType> check_list_;

        //! relative tolerance
        std::vector<double> rtol_;

        //! absolute tolerance
        std::vector<double> atol_;

        //! Time integration method multiplier (BDF Multiplier)
        std::vector<double> factor_;

        //! Required tolerance for the NormWRMS to be declared converged.
        std::vector<double> tol_;

        //! Minimum step size allowed during a line search for WRMS norm to be flagged as converged.
        double alpha_;

        //! Actual step size used during line search.
        double computed_step_size_;

        //! Maximum linear solve tolerance allowed for WRMS norm to be flagged as converged.
        double beta_;

        //! Actual tolerance achieved by the linear solver during the last linear solve.
        double achieved_tol_;

        //! Global status
        ::NOX::StatusTest::StatusType g_status_;

        //! Status of each quantity
        std::vector<::NOX::StatusTest::StatusType> status_;

        //! Flag that tells the print method whether to print the criteria 2 information.
        bool print_criteria2_info_;

        //! Flag that tells the print method whether to print the criteria 3 information.
        bool print_criteria3_info_;

        //! If true, the implicit weighting will be disabled during norm calculation
        std::vector<bool> disable_implicit_weighting_;
      };
    }  // namespace StatusTest
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
