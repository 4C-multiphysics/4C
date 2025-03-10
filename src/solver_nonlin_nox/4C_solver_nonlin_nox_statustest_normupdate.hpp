// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_STATUSTEST_NORMUPDATE_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_STATUSTEST_NORMUPDATE_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_solver_nonlin_nox_forward_decl.hpp"

#include <NOX_Abstract_Vector.H>
#include <NOX_StatusTest_Generic.H>
#include <NOX_Utils.H>

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    namespace StatusTest
    {
      //! \brief Absolute and relative convergence tests based on the norm of the increment.
      /*!
       I recommend to use the weighted root mean square test (normwrms), instead or additionally
       (AND combination)!

       Nevertheless, I want to give you the possibility to check the absolute and relative norm
       of the solution increment corresponding to the different quantities as well. This class is
       leaned on the NOX::Nln::StatusTest::NormF and NOX::Nln::StatusTest::NormWRMS test and has
       an equivalent structure. It is capable to handle a std::vector of increment norms. In this
       way it's easily possible to check the norms of different quantities at once in a very
       general way.

       One difference to the NormF case is the calculation of the reference value. We use
       the following equation:

                 $\frac{\| x_i^{(k+1)} - x_i^{(k)} \|}{\|x_i^{(k)}\|} \leq RTOL$

       */
      class NormUpdate : public ::NOX::StatusTest::Generic
      {
       public:
        //! Type that determines whether to scale the norm by the problem size.
        enum ScaleType
        {
          //! No norm scaling
          Unscaled,
          //! Scale the norm by the length of the vector
          Scaled
        };

        //! Type that determines whether the check is absolute or relative
        enum ToleranceType
        {
          //! Relative convergence check
          Relative,
          //! Absolute
          Absolute
        };

       public:
        //! Constructor.
        NormUpdate(const std::vector<NOX::Nln::StatusTest::QuantityType>& checkList,
            const std::vector<NormUpdate::ToleranceType>& toltype,
            const std::vector<double>& tolerance,
            const std::vector<::NOX::Abstract::Vector::NormType>& ntype,
            const std::vector<NormUpdate::ScaleType>& stype, const double& alpha,
            const double& beta, const ::NOX::Utils* u = nullptr);

        ::NOX::StatusTest::StatusType checkStatus(
            const ::NOX::Solver::Generic& problem, ::NOX::StatusTest::CheckType checkType) override;

        /*! \brief Returns the norm type as <int> of the desired quantity
         *
         *  If the given quantity cannot be found a default value of -100 is returned. */
        int get_norm_type(const NOX::Nln::StatusTest::QuantityType& qType) const;

        //! Check for the given quantity
        bool is_quantity(const NOX::Nln::StatusTest::QuantityType& qType) const;

        //! NOTE: returns the global status of all normF tests
        ::NOX::StatusTest::StatusType getStatus() const override;

        std::ostream& print(std::ostream& stream, int indent = 0) const override;

       protected:
        virtual void compute_norm(
            const ::NOX::Abstract::Group& grp, const ::NOX::Solver::Generic& problem);

       private:
        //! check status in first nonlinear iteration step
        virtual ::NOX::StatusTest::StatusType check_status_first_iter();

        //! throws an NOX error
        void throw_error(const std::string& functionName, const std::string& errorMsg) const;

       protected:
        //! number of status tests
        const std::size_t nChecks_;

        //! computed step size
        double computedStepSize_;

        //! Minimum step size allowed during a line search for increment norm to be flagged as
        //! converged.
        double alpha_;

        //! Actual tolerance achieved by the linear solver during the last linear solve.
        double achievedTol_;

        //! Maximum linear solve tolerance allowed for incr norm to be flagged as converged.
        double beta_;

        //! enums of the quantities we want check
        std::vector<NOX::Nln::StatusTest::QuantityType> checkList_;

        //! global status
        ::NOX::StatusTest::StatusType gStatus_;

        //! Status
        std::vector<::NOX::StatusTest::StatusType> status_;

        //! Type of norm to use
        std::vector<::NOX::Abstract::Vector::NormType> normType_;

        //! Scaling to use
        std::vector<NormUpdate::ScaleType> scaleType_;

        //! Tolerance type (i.e., relative or absolute)
        std::vector<NormUpdate::ToleranceType> toleranceType_;

        //! Tolerance required for convergence.
        std::vector<double> specifiedTolerance_;

        //! reference norm of the solution vector (relative only)
        Teuchos::RCP<const std::vector<double>> normRefSol_;

        //! True tolerance value, i.e., specifiedTolerance * initialTolerance
        std::vector<double> trueTolerance_;

        //! Norm of F to be compared to trueTolerance
        Teuchos::RCP<const std::vector<double>> normUpdate_;

        //! Ostream used to print errors
        ::NOX::Utils utils_;

        //! Flag that tells the print method whether to print the criteria 2 information.
        bool printCriteria2Info_;

        //! Flag that tells the print method whether to print the criteria 3 information.
        bool printCriteria3Info_;
      };  // class NormUpdate


      //! \brief Absolute and relative convergence tests based on the norm of the increment,
      //! skipping the first nonlinear iteration.
      /*!
       * This derived class mostly does exactly the same as the parent class, with only one
       * exception: In the very first nonlinear iteration, the status check is skipped, i.e., the
       * return value always indicates convergence. Only the second and subsequent nonlinear
       * iterations are actually checked.
       *
       * Exemplary application:
       * ----------------------
       * During outer iterations in partitioned multi-field simulations, the structural field might
       * no longer change when outer convergence is almost achieved, and thus skipping the status
       * test in the first nonlinear iteration might save unnecessary calls of the linear solver.
       * This requires, however, that the structural residual is checked instead of the structural
       * increment, and that outer non-convergence is reliably indicated thereby. Experience has
       * shown that this is usually the case when standard nonlinear solution techniques, such as
       * the plain Newton-Raphson method, are employed, but one should still be aware thereof.
       *

       */
      class NormUpdateSkipFirstIter : public NormUpdate
      {
       public:
        //! Constructor.
        NormUpdateSkipFirstIter(const std::vector<NOX::Nln::StatusTest::QuantityType>& checkList,
            const std::vector<NormUpdate::ToleranceType>& toltype,
            const std::vector<double>& tolerance,
            const std::vector<::NOX::Abstract::Vector::NormType>& ntype,
            const std::vector<NormUpdate::ScaleType>& stype, const double& alpha,
            const double& beta, const ::NOX::Utils* u = nullptr)
            :  // call base class constructor
              NormUpdate(checkList, toltype, tolerance, ntype, stype, alpha, beta, u)
        {
          return;
        };

       private:
        //! check status in first nonlinear iteration step
        ::NOX::StatusTest::StatusType check_status_first_iter() override;
      };  // class NormUpdateSkipFirstIter
    }  // namespace StatusTest
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
