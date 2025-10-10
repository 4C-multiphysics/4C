// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_GROUP_BASE_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_GROUP_BASE_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_linearsystem_base.hpp"

#include <NOX_Abstract_Group.H>
#include <NOX_Epetra_Interface_Required.H>
#include <NOX_Epetra_Vector.H>
#include <NOX_Utils.H>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    class GroupBase : public ::NOX::Abstract::Group
    {
     public:
      GroupBase(Teuchos::ParameterList& printParams,
          const Teuchos::RCP<::NOX::Epetra::Interface::Required>& i, const ::NOX::Epetra::Vector& x,
          const Teuchos::RCP<NOX::Nln::LinearSystemBase>& linSys);

      GroupBase(const NOX::Nln::GroupBase& source, ::NOX::CopyType type);

      ::NOX::Abstract::Group& operator=(const ::NOX::Abstract::Group& source) override;

      //! Clone the group (deep or shallow copy).
      Teuchos::RCP<::NOX::Abstract::Group> clone(
          ::NOX::CopyType type = ::NOX::DeepCopy) const override;

      /** @name Compute functions. */
      //@{

      void setX(const ::NOX::Abstract::Vector& y) override;

      void computeX(const ::NOX::Abstract::Group& grp, const ::NOX::Abstract::Vector& d,
          double step) override;

      ::NOX::Abstract::Group::ReturnType computeF() override;

      ::NOX::Abstract::Group::ReturnType computeJacobian() override;

      ::NOX::Abstract::Group::ReturnType computeGradient() override;

      ::NOX::Abstract::Group::ReturnType computeNewton(Teuchos::ParameterList& params) override;

      //@}

      /** @name Jacobian operations. */
      //@{

      ::NOX::Abstract::Group::ReturnType applyJacobian(
          const ::NOX::Abstract::Vector& input, ::NOX::Abstract::Vector& result) const override;

      ::NOX::Abstract::Group::ReturnType applyJacobianTranspose(
          const ::NOX::Abstract::Vector& input, ::NOX::Abstract::Vector& result) const override;

      ::NOX::Abstract::Group::ReturnType applyJacobianInverse(Teuchos::ParameterList& params,
          const ::NOX::Abstract::Vector& input, ::NOX::Abstract::Vector& result) const override;

      //@}

      /** @name These functions check if various objects have been computed.
       */
      //@{

      bool isF() const override;
      bool isJacobian() const override;
      bool isGradient() const override;
      bool isNewton() const override;

      //@}

      /** @name Get functions.
       *
       * Note that these function do not check whether or not the vectors are valid. Distinctly from
       * the original Epetra group, all of these function to do not check that the corresponding
       * object has been evaluated.
       */
      //@{

      const ::NOX::Abstract::Vector& getX() const override;

      const ::NOX::Abstract::Vector& getF() const override;

      const ::NOX::Abstract::Vector& getGradient() const override;

      const ::NOX::Abstract::Vector& getNewton() const override;

      Teuchos::RCP<const ::NOX::Abstract::Vector> getXPtr() const override;

      Teuchos::RCP<const ::NOX::Abstract::Vector> getFPtr() const override;

      Teuchos::RCP<const ::NOX::Abstract::Vector> getGradientPtr() const override;

      Teuchos::RCP<const ::NOX::Abstract::Vector> getNewtonPtr() const override;

      double getNormF() const override;

      //@}

      //! Return the user interface object.
      Teuchos::RCP<::NOX::Epetra::Interface::Required> get_required_interface();

      //! Return the Linear System.
      Teuchos::RCP<const NOX::Nln::LinearSystemBase> get_linear_system() const;

      //! Return the Linear System.
      Teuchos::RCP<NOX::Nln::LinearSystemBase> get_linear_system();

     protected:
      //! Resets the isValid flags to false
      void reset_is_valid();

      //! Write last linear solve stats into the provided NOX::SolverStats object
      void logLastLinearSolveStats(::NOX::SolverStats& stats) const override;

     protected:
      //! Printing Utilities object
      const ::NOX::Utils utils;

      /** @name Vectors */
      //@{
      //! Solution vector.
      ::NOX::Epetra::Vector xVector;
      //! Right-hand-side vector (function evaluation).
      ::NOX::Epetra::Vector RHSVector;
      //! Gradient vector (steepest descent vector).
      ::NOX::Epetra::Vector gradVector;
      //! Newton direction vector.
      ::NOX::Epetra::Vector NewtonVector;
      //@}

      /** @name IsValid flags
       *
       * True if a particular object is up-to-date with respect to the
       * current xVector. */
      //@{
      bool isValidRHS;
      bool isValidJacobian;
      bool isValidGrad;
      bool isValidNewton;
      //@}

      /** @name Operators */
      //@{
      //! Pointer to Jacobian matrix
      Teuchos::RCP<NOX::Nln::LinearSystemBase> linearSystemPtr;

      //! Pointer to the user supplied interface functions
      Teuchos::RCP<::NOX::Epetra::Interface::Required> userInterfacePtr;
      //@}

      // Linear solver stats
      mutable bool lastLinearSolveConverged;
      mutable int lastNumIterations;
      mutable double lastAchievedTol;
    };  // class GroupBase
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
