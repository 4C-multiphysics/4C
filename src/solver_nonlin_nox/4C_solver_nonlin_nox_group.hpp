// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_GROUP_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_GROUP_HPP

/*----------------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_linalg_serialdensevector.hpp"
#include "4C_solver_nonlin_nox_forward_decl.hpp"
#include "4C_solver_nonlin_nox_group_base.hpp"
#include "4C_solver_nonlin_nox_statustest_normupdate.hpp"

#include <NOX_StatusTest_NormF.H>

#include <set>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::LinAlg
{
  template <typename T>
  class Vector;
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace NOX
{
  namespace Nln
  {
    namespace Solver
    {
      class PseudoTransient;
    }  // namespace Solver
    namespace Interface
    {
      class Required;
    }  // namespace Interface
    namespace GROUP
    {
      class PrePostOperator;
    }  // namespace GROUP

    class Group : public GroupBase
    {
     public:
      //! Standard Constructor
      Group(Teuchos::ParameterList& printParams,    //!< printing parameters
          Teuchos::ParameterList& grpOptionParams,  //!< group option parameters
          const Teuchos::RCP<::NOX::Epetra::Interface::Required>&
              i,                           //!< basically the NOXified user interface
          const ::NOX::Epetra::Vector& x,  //!< current solution vector
          const Teuchos::RCP<NOX::Nln::LinearSystemBase>&
              linSys  //!< linear system, matrix and RHS etc.
      );

      /*! \brief Copy constructor. If type is DeepCopy, takes ownership of
        valid shared linear system. */
      Group(const NOX::Nln::Group& source, ::NOX::CopyType type = ::NOX::DeepCopy);

      /// assign operator
      ::NOX::Abstract::Group& operator=(const ::NOX::Abstract::Group& source) override;

      Teuchos::RCP<::NOX::Abstract::Group> clone(
          ::NOX::CopyType type = ::NOX::DeepCopy) const override;

      //! compute/update the current state variables
      void computeX(const NOX::Nln::Group& grp, const ::NOX::Epetra::Vector& d, double step);
      void computeX(const ::NOX::Abstract::Group& grp, const ::NOX::Abstract::Vector& d,
          double step) override;

      ::NOX::Abstract::Group::ReturnType computeF() override;

      ::NOX::Abstract::Group::ReturnType applyJacobianInverse(Teuchos::ParameterList& p,
          const ::NOX::Abstract::Vector& input, ::NOX::Abstract::Vector& result) const override;

      //! Compute and store \f$F(x)\f$ and the jacobian \f$\frac{\partial F(x)}{\partial x}\f$ at
      //! the same time. This can result in a huge performance gain in some special cases, e.g.
      //! contact problems.
      virtual ::NOX::Abstract::Group::ReturnType compute_f_and_jacobian();

      //! set right hand side
      ::NOX::Abstract::Group::ReturnType set_f(Teuchos::RCP<::NOX::Epetra::Vector> Fptr);

#if !(FOUR_C_TRILINOS_INTERNAL_VERSION_GE(2025, 4))
      //! set the solution vector to zero
      void reset_x();
#endif

      //! set flag whether update of x vector should be skipped (because it has already be done in
      //! preComputeX)
      void set_skip_update_x(bool skipUpdateX);

      /* Check the isValidJacobian flag and the ownership of the linear system
       * separately and get the ownership, if necessary. This prevents unnecessary
       * evaluation calls of the expensive computeJacobian() routines! Afterwards
       * the base class function is called.                   hiermeier 03/2016 */
      bool isJacobian() const override;

      //! returns the nox_nln_interface_required pointer
      Teuchos::RCP<const NOX::Nln::Interface::Required> get_nln_req_interface_ptr() const;

      //! returns the primary rhs norms
      virtual Teuchos::RCP<const std::vector<double>> get_rhs_norms(
          const std::vector<::NOX::Abstract::Vector::NormType>& type,
          const std::vector<NOX::Nln::StatusTest::QuantityType>& chQ,
          Teuchos::RCP<const std::vector<::NOX::StatusTest::NormF::ScaleType>> scale =
              Teuchos::null) const;

      //! returns the Root Mean Squares (abbr.: RMS) of the primary solution updates
      virtual Teuchos::RCP<std::vector<double>> get_solution_update_rms(
          const ::NOX::Abstract::Vector& xOld, const std::vector<double>& aTol,
          const std::vector<double>& rTol,
          const std::vector<NOX::Nln::StatusTest::QuantityType>& chQ,
          const std::vector<bool>& disable_implicit_weighting) const;

      double get_trial_update_norm(const ::NOX::Abstract::Vector& dir,
          const ::NOX::Abstract::Vector::NormType normtype, const StatusTest::QuantityType quantity,
          const StatusTest::NormUpdate::ScaleType scale = StatusTest::NormUpdate::Unscaled) const;

      //! returns the desired norm of the primary solution updates
      virtual Teuchos::RCP<std::vector<double>> get_solution_update_norms(
          const ::NOX::Abstract::Vector& xOld,
          const std::vector<::NOX::Abstract::Vector::NormType>& type,
          const std::vector<StatusTest::QuantityType>& chQ,
          Teuchos::RCP<const std::vector<StatusTest::NormUpdate::ScaleType>> scale =
              Teuchos::null) const;

      //! returns the desired norm of the previous solution
      virtual Teuchos::RCP<std::vector<double>> get_previous_solution_norms(
          const ::NOX::Abstract::Vector& xOld,
          const std::vector<::NOX::Abstract::Vector::NormType>& type,
          const std::vector<StatusTest::QuantityType>& chQ,
          Teuchos::RCP<const std::vector<StatusTest::NormUpdate::ScaleType>> scale) const;

      //! @name reset the pre/post operator wrapper objects
      //! @{
      /*! \brief Resets the pre/post operator for the nln group
       *  Default call to the two parameter version, without resetting the isValid flags.
       *  @param[in] grpOptionParams   ParameterList which holds the new pre/post operator. */
      void reset_pre_post_operator(Teuchos::ParameterList& grpOptionParams)
      {
        reset_pre_post_operator(grpOptionParams, false);
      };

      /*! \brief Resets the pre/post operator wrapper for the nln group
       *  @param[in] grpOptionsParams   ParameterList which holds the new pre/post operator
       *  @param[in] resetIsValidFlag   If true, this forces the computeJacobian(), computeF() etc.
       * routines to reevaluate the linear system after setting a new pre/post operator. */
      void reset_pre_post_operator(
          Teuchos::ParameterList& grpOptionParams, const bool& resetIsValidFlags);

      /*! \brief Resets the pre/post operator wrapper for the nln linear system
       *  Default call to the two parameter version, without resetting the isValid flags.
       *  @param[in] linearSolverParams ParameterList which holds the new pre/post operator. */
      void reset_lin_sys_pre_post_operator(Teuchos::ParameterList& linearSolverParams)
      {
        reset_lin_sys_pre_post_operator(linearSolverParams, false);
      };

      /*! \brief Resets the pre/post operator wrapper for the nln linear system
       *  @param[in] linearSolverParams   ParameterList which holds the new pre/post operator.
       *  @param[in] resetIsValidFlag     If true, this forces the computeJacobian(), computeF()
       * etc. routines to reevaluate the linear system after setting a new pre/post operator. */
      void reset_lin_sys_pre_post_operator(
          Teuchos::ParameterList& linearSolverParams, const bool& resetIsValidFlags);
      //! @}

      //! @name PTC related methods
      //! @{
      //! adjust the pseudo time step length for the ptc nln solver
      void adjust_pseudo_time_step(double& delta, const double& stepSize,
          const ::NOX::Abstract::Vector& dir, const NOX::Nln::Solver::PseudoTransient& ptcsolver);
      void adjust_pseudo_time_step(double& delta, const double& stepSize,
          const ::NOX::Epetra::Vector& dir, const NOX::Nln::Solver::PseudoTransient& ptcsolver);

      // Get element based scaling operator
      Teuchos::RCP<Core::LinAlg::SparseMatrix> get_contributions_from_element_level();
      //! @}

#if !(FOUR_C_TRILINOS_INTERNAL_VERSION_GE(2025, 4))
      /// allow to set isValidNewton manually
      inline void set_is_valid_newton(const bool value) { isValidNewton = value; };

      /// allow to set isValidRHS manually
      inline void set_is_valid_rhs(const bool value) { isValidRHS = value; };
#endif

     private:
      //! Throw an NOX_error
      void throw_error(const std::string& functionName, const std::string& errorMsg) const;

      mutable Teuchos::RCP<Epetra_Vector> tmp_vector_ptr_;

     protected:
      /*! flag whether update of x vector should be skipped
       *  (e.g. because it has already be done in preComputeX as might be the case if we
       *  need a multiplicative update of some beam elements' rotation (pseudo-)vector DOFs) */
      bool skipUpdateX_;

      //! pointer to an user defined wrapped NOX::Nln::Abstract::PrePostOperator object.
      Teuchos::RCP<NOX::Nln::GROUP::PrePostOperator> prePostOperatorPtr_;
    };  // class Group
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
