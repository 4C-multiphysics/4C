// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_PROBLEM_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_PROBLEM_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_solver_nonlin_nox_forward_decl.hpp"
#include "4C_solver_nonlin_nox_linearsystem_base.hpp"
#include "4C_utils_exceptions.hpp"

#include <NOX_StatusTest_Generic.H>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::LinAlg
{
  class Solver;
  class SparseOperator;
}  // namespace Core::LinAlg

namespace NOX
{
  namespace Nln
  {
    class GlobalData;
    namespace Inner
    {
      namespace StatusTest
      {
        class Generic;
      }  // namespace StatusTest
    }  // namespace Inner

    class Problem
    {
     public:
      //! minimal constructor
      Problem(const Teuchos::RCP<NOX::Nln::GlobalData>& noxNlnGlobalData);

      //! standard constructor
      Problem(const Teuchos::RCP<NOX::Nln::GlobalData>& noxNlnGlobalData,
          const Teuchos::RCP<::NOX::Epetra::Vector>& x,
          const Teuchos::RCP<Core::LinAlg::SparseOperator>& A);

      //! initialize stuff
      void initialize(const Teuchos::RCP<::NOX::Epetra::Vector>& x,
          const Teuchos::RCP<Core::LinAlg::SparseOperator>& A);

      //! create the linear system for the NOX framework
      Teuchos::RCP<NOX::Nln::LinearSystemBase> create_linear_system() const;

      //! create a nox group
      Teuchos::RCP<::NOX::Abstract::Group> create_group(
          const Teuchos::RCP<NOX::Nln::LinearSystemBase>& linSys) const;

      void create_outer_status_test(Teuchos::RCP<::NOX::StatusTest::Generic>& outerTests) const;

      void create_status_tests(Teuchos::RCP<::NOX::StatusTest::Generic>& outerTest,
          Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic>& innerTest) const;

      //! check final status of the non-linear solving procedure
      void check_final_status(const ::NOX::StatusTest::StatusType& finalStatus) const;

      /// access the global data object
      NOX::Nln::GlobalData& nln_global_data() { return *nox_global_data_; }

      /// access the global data object ptr
      Teuchos::RCP<NOX::Nln::GlobalData> nln_global_data_ptr() { return nox_global_data_; }

     private:
      inline void check_init() const
      {
        if (not isinit_)
          FOUR_C_THROW(
              "You have to call initialize() first, before you can use this"
              " function!");
      }

      inline bool is_jacobian() const { return isjac_; };

      bool isinit_;

      bool isjac_;

      Teuchos::RCP<NOX::Nln::GlobalData> nox_global_data_;

      /** ptr to the state vector RCP. In this way the strong_count is neither lost
       *  nor increased. */
      const Teuchos::RCP<::NOX::Epetra::Vector>* x_vector_;

      /** ptr to the state matrix RCP. In this way the strong_count is neither lost
       *  nor increased. */
      const Teuchos::RCP<Core::LinAlg::SparseOperator>* jac_;

      Teuchos::RCP<Core::LinAlg::SparseOperator> preconditionner_;
    };
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
