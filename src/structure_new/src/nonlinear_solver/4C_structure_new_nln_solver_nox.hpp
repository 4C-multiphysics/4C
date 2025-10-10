// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_NLN_SOLVER_NOX_HPP
#define FOUR_C_STRUCTURE_NEW_NLN_SOLVER_NOX_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_linearsystem_base.hpp"
#include "4C_structure_new_nln_solver_generic.hpp"  // base class

#include <NOX_StatusTest_Generic.H>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>

namespace NOX
{
  namespace Abstract
  {
    class Group;
  }  // namespace Abstract
  namespace Epetra
  {
    class LinearSystem;
  }  // namespace Epetra
}  // namespace NOX

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    class GlobalData;
    class Problem;
    namespace Inner
    {
      namespace StatusTest
      {
        class Generic;
      }  // namespace StatusTest
    }  // namespace Inner
  }  // namespace Nln
}  // namespace NOX

namespace Core::LinAlg
{
  class Solver;
}  // namespace Core::LinAlg

namespace Solid
{
  namespace Nln
  {
    namespace SOLVER
    {
      /*! Interface to NOX as nonlinear solver in structural dynamics
       *
       */
      class Nox : public Generic
      {
       public:
        //! constructor
        Nox(const Teuchos::ParameterList& default_params,
            const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate,
            const std::shared_ptr<Solid::TimeInt::BaseDataSDyn>& sdyn,
            const std::shared_ptr<Solid::TimeInt::NoxInterface>& noxinterface,
            const std::shared_ptr<Solid::Integrator>& integr,
            const std::shared_ptr<const Solid::TimeInt::Base>& timint);

        //! derived from the base class
        void reset() override;

        //! derived from the base class
        Inpar::Solid::ConvergenceStatus solve() override;

        //! returns the outer status test object pointer
        const ::NOX::StatusTest::Generic& get_outer_status_test() const
        {
          FOUR_C_ASSERT(ostatus_, "The outer status test object is not defined!");
          return *ostatus_;
        }

        //! get number of nonlinear iterations (derived)
        int get_num_nln_iterations() const override;

       protected:
        //! Reset the non-linear solver parameters and variables
        virtual void reset_params();

        //! Convert the final nox status into a structural status
        enum Inpar::Solid::ConvergenceStatus convert_final_status(
            const ::NOX::StatusTest::StatusType& finalstatus) const;

       protected:
        //! pointer to the nox nln global data container
        Teuchos::RCP<NOX::Nln::GlobalData> nlnglobaldata_;

        //! NOX non-linear solver
        Teuchos::RCP<::NOX::Solver::Generic> nlnsolver_;

        //! Default solver parameters
        const Teuchos::ParameterList default_params_;

       private:
        //! @name variables which are reset in each solve() call
        //!@{

        /*! \brief NOX non-linear problem class
         *
         *  The main task is to manage the non-linear solver creation
         */
        Teuchos::RCP<NOX::Nln::Problem> problem_;

        //! linear system class
        Teuchos::RCP<NOX::Nln::LinearSystemBase> linsys_;

        //! outer status test
        Teuchos::RCP<::NOX::StatusTest::Generic> ostatus_;

        //! inner status test
        Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic> istatus_;

        //!@}
      };  // class Nox
    }  // namespace SOLVER
  }  // namespace Nln
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
