// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_SINGLESTEP_GROUP_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_SINGLESTEP_GROUP_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_constraint_interface_required.hpp"
#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_solver_nonlin_nox_group.hpp"

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    namespace SINGLESTEP
    {
      class Group : public NOX::Nln::Group
      {
       public:
        //! Standard constructor
        Group(Teuchos::ParameterList& printParams,  //!< printing parameters
            Teuchos::ParameterList& grpOptionParams,
            const Teuchos::RCP<::NOX::Epetra::Interface::Required>&
                i,                           //!< basically the NOXified time integrator
            const ::NOX::Epetra::Vector& x,  //!< current solution vector
            const Teuchos::RCP<NOX::Nln::LinearSystemBase>&
                linSys  //!< linear system, matrix and RHS etc.
        );

        /*! \brief Copy constructor. If type is DeepCopy, takes ownership of
          valid shared linear system. */
        Group(const NOX::Nln::SINGLESTEP::Group& source, ::NOX::CopyType type = ::NOX::DeepCopy);

        //! generate a clone of the given object concerning the given \c CopyType
        Teuchos::RCP<::NOX::Abstract::Group> clone(::NOX::CopyType type) const override;

        //! compute/update the current state variables
        void computeX(
            const NOX::Nln::SINGLESTEP::Group& grp, const ::NOX::Epetra::Vector& d, double step);
        void computeX(const ::NOX::Abstract::Group& grp, const ::NOX::Abstract::Vector& d,
            double step) override;

       private:
        //! Throw an NOX_error
        void throw_error(const std::string& functionName, const std::string& errorMsg) const;
      };
    }  // namespace SINGLESTEP
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
