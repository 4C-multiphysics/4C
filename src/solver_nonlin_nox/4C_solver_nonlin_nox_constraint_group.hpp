// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_CONSTRAINT_GROUP_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_CONSTRAINT_GROUP_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_constraint_interface_required.hpp"
#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Solid
{
  class TimIntImpl;
}

namespace NOX
{
  namespace Nln
  {
    namespace CONSTRAINT
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
                linSys,  //!< linear system, matrix and RHS etc.
            const std::map<enum NOX::Nln::SolutionType,
                Teuchos::RCP<NOX::Nln::CONSTRAINT::Interface::Required>>&
                iConstr  //!< constraint interfaces
        );

        /*! \brief Copy constructor. If type is DeepCopy, takes ownership of
          valid shared linear system. */
        Group(const NOX::Nln::CONSTRAINT::Group& source, ::NOX::CopyType type = ::NOX::DeepCopy);

        //! generate a clone of the given object concerning the given \c CopyType
        Teuchos::RCP<::NOX::Abstract::Group> clone(::NOX::CopyType type) const override;

        ::NOX::Abstract::Group& operator=(const ::NOX::Abstract::Group& source) override;

        //! Returns the interface map
        const ReqInterfaceMap& get_constraint_interfaces() const;

        //! Returns a pointer to the given soltype. If the solution type is not found an error is
        //! thrown.
        Teuchos::RCP<const NOX::Nln::CONSTRAINT::Interface::Required> get_constraint_interface_ptr(
            const NOX::Nln::SolutionType soltype) const;

        //! If the \c errflag is set to true, a error is thrown as soon as we cannot find the
        //! corresponding entry in the stl_map. Otherwise a Teuchos::null pointer is returned.
        Teuchos::RCP<const NOX::Nln::CONSTRAINT::Interface::Required> get_constraint_interface_ptr(
            const NOX::Nln::SolutionType soltype, const bool errflag) const;

        // @name "Get" functions
        //@{

        //! Returns the right-hand-side norms of the primary and constraint quantities
        Teuchos::RCP<const std::vector<double>> get_rhs_norms(
            const std::vector<::NOX::Abstract::Vector::NormType>& type,
            const std::vector<NOX::Nln::StatusTest::QuantityType>& chQ,
            Teuchos::RCP<const std::vector<::NOX::StatusTest::NormF::ScaleType>> scale =
                Teuchos::null) const override;

        //! Returns the root mean square norm of the primary and Lagrange multiplier updates
        Teuchos::RCP<std::vector<double>> get_solution_update_rms(
            const ::NOX::Abstract::Vector& xOld, const std::vector<double>& aTol,
            const std::vector<double>& rTol,
            const std::vector<NOX::Nln::StatusTest::QuantityType>& chQ,
            const std::vector<bool>& disable_implicit_weighting) const override;

        //! Returns the desired norm of the primary solution updates and Lagrange multiplier updates
        Teuchos::RCP<std::vector<double>> get_solution_update_norms(
            const ::NOX::Abstract::Vector& xOld,
            const std::vector<::NOX::Abstract::Vector::NormType>& type,
            const std::vector<StatusTest::QuantityType>& chQ,
            Teuchos::RCP<const std::vector<StatusTest::NormUpdate::ScaleType>> scale)
            const override;

        //! Returns the desired norm of the previous primary solution and Lagrange multiplier
        //! solution
        Teuchos::RCP<std::vector<double>> get_previous_solution_norms(
            const ::NOX::Abstract::Vector& xOld,
            const std::vector<::NOX::Abstract::Vector::NormType>& type,
            const std::vector<StatusTest::QuantityType>& chQ,
            Teuchos::RCP<const std::vector<StatusTest::NormUpdate::ScaleType>> scale)
            const override;
        //! @}

        //! @name Handle active set strategies
        //! @{
        //! Returns the current active set map (only needed for inequality constraint problems)
        Teuchos::RCP<const Core::LinAlg::Map> get_current_active_set_map(
            const enum NOX::Nln::StatusTest::QuantityType& qtype) const;

        //! Returns the active set map of the previous Newton step (only needed for inequality
        //! constraint problems)
        Teuchos::RCP<const Core::LinAlg::Map> get_old_active_set_map(
            const enum NOX::Nln::StatusTest::QuantityType& qtype) const;

        //! Returns basic information about the active set status (no Core::LinAlg::Maps needed!)
        enum ::NOX::StatusTest::StatusType get_active_set_info(
            const enum NOX::Nln::StatusTest::QuantityType& qtype, int& activeset_size) const;

        //@}

       private:
        //! throw Nox error
        void throw_error(const std::string& functionName, const std::string& errorMsg) const;

       private:
        // constraint interface map
        ReqInterfaceMap user_constraint_interfaces_;
      };  // class Group
    }  // end namespace CONSTRAINT
  }  // namespace Nln
}  // end namespace  NOX

FOUR_C_NAMESPACE_CLOSE

#endif
