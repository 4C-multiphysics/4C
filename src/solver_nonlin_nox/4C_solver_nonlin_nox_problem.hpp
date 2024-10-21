#ifndef FOUR_C_SOLVER_NONLIN_NOX_PROBLEM_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_PROBLEM_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_solver_nonlin_nox_forward_decl.hpp"
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
    }    // namespace Inner

    class Problem
    {
     public:
      //! minimal constructor
      Problem(const Teuchos::RCP<NOX::Nln::GlobalData>& noxNlnGlobalData);

      //! standard constructor
      Problem(const Teuchos::RCP<NOX::Nln::GlobalData>& noxNlnGlobalData,
          const Teuchos::RCP<::NOX::Epetra::Vector>& x,
          const Teuchos::RCP<Core::LinAlg::SparseOperator>& A);

      //! destructor
      virtual ~Problem() = default;

      //! initialize stuff (can be overloaded in derived classes)
      virtual void initialize(const Teuchos::RCP<::NOX::Epetra::Vector>& x,
          const Teuchos::RCP<Core::LinAlg::SparseOperator>& A);

      //! create the linear system for the NOX framework
      virtual Teuchos::RCP<::NOX::Epetra::LinearSystem> create_linear_system() const;

      //! create a nox group
      virtual Teuchos::RCP<::NOX::Abstract::Group> create_group(
          const Teuchos::RCP<::NOX::Epetra::LinearSystem>& linSys) const;

      void create_outer_status_test(Teuchos::RCP<::NOX::StatusTest::Generic>& outerTests) const;

      virtual void create_status_tests(Teuchos::RCP<::NOX::StatusTest::Generic>& outerTest,
          Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic>& innerTest) const;

      //! check final status of the non-linear solving procedure
      virtual void check_final_status(const ::NOX::StatusTest::StatusType& finalStatus) const;

      /// access the global data object
      NOX::Nln::GlobalData& nln_global_data() { return *noxNlnGlobalData_; }

      /// access the global data object ptr
      Teuchos::RCP<NOX::Nln::GlobalData> nln_global_data_ptr() { return noxNlnGlobalData_; }

     protected:
      inline void check_init() const
      {
        if (not isinit_)
          FOUR_C_THROW(
              "You have to call initialize() first, before you can use this"
              " function!");
      }

      inline bool is_jacobian() const { return isjac_; };

     protected:
      bool isinit_;

      bool isjac_;

      Teuchos::RCP<NOX::Nln::GlobalData> noxNlnGlobalData_;

      /** ptr to the state vector RCP. In this way the strong_count is neither lost
       *  nor increased. */
      const Teuchos::RCP<::NOX::Epetra::Vector>* xVector_;

      /** ptr to the state matrix RCP. In this way the strong_count is neither lost
       *  nor increased. */
      const Teuchos::RCP<Core::LinAlg::SparseOperator>* jac_;

      Teuchos::RCP<Core::LinAlg::SparseOperator> precMat_;
    };
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
