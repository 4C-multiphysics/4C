// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_TIMINT_NOXGROUP_HPP
#define FOUR_C_STRUCTURE_TIMINT_NOXGROUP_HPP

/*----------------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_structure_timint_impl.hpp"

#include <NOX_Epetra_Group.H>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/* forward declarations */
namespace Solid
{
  class TimIntImpl;
}

/*----------------------------------------------------------------------------*/
/* belongs to NOX name space of Trilinos */
namespace NOX
{
  /*--------------------------------------------------------------------------*/
  namespace Solid
  {
    /*==================================================================*/
    /*!
     * \brief Special ::NOX::Epetra::Group that always //ToDo (mayr) That's not correct anymore.
     *        sets Jacobian and RHS at the same time.
     *

     */
    class Group : public ::NOX::Epetra::Group
    {
     public:
      //! Constructor
      Group(FourC::Solid::TimIntImpl& sti,      //!< time integrator
          Teuchos::ParameterList& printParams,  //!< printing parameters
          const Teuchos::RCP<::NOX::Epetra::Interface::Required>&
              i,                           //!< basically the NOXified time integrator
          const ::NOX::Epetra::Vector& x,  //!< current solution vector
          const Teuchos::RCP<::NOX::Epetra::LinearSystem>&
              linSys  //!< linear system, matrix and RHS etc.
      );

      //! Compute and store \f$F(x)\f$
      ::NOX::Abstract::Group::ReturnType computeF() override;

      //! Compute and store Jacobian \f$\frac{\partial F(x)}{\partial x}\f$
      ::NOX::Abstract::Group::ReturnType computeJacobian() override;

     private:
      //! structural time integrator
      //! HINT: Currently this variable is unused within the group
      //::Solid::TimIntImpl& sti_;
    };

  }  // namespace Solid

}  // namespace NOX

/*----------------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE

#endif
