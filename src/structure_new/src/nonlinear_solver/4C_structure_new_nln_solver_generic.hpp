// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_NLN_SOLVER_GENERIC_HPP
#define FOUR_C_STRUCTURE_NEW_NLN_SOLVER_GENERIC_HPP

#include "4C_config.hpp"

#include "4C_structure_new_input.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_RCPDecl.hpp>

#include <memory>

// forward declaration
namespace NOX
{
  namespace Abstract
  {
    class Group;
  }  // namespace Abstract
}  // namespace NOX

FOUR_C_NAMESPACE_OPEN

namespace Solid
{
  class Integrator;
  namespace TimeInt
  {
    class Implicit;
    class BaseDataGlobalState;
    class BaseDataSDyn;
    class Base;
    class NoxInterface;
  }  // namespace TimeInt
  namespace Nln
  {
    namespace SOLVER
    {
      /*! \brief Base class of all nonlinear solvers for structural dynamcis
       *
       */
      class Generic
      {
       public:
        //! constructor
        Generic(const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate,
            const std::shared_ptr<Solid::TimeInt::BaseDataSDyn>& sdyn,
            const std::shared_ptr<Solid::TimeInt::NoxInterface>& noxinterface,
            const std::shared_ptr<Solid::Integrator>& integr,
            const std::shared_ptr<const Solid::TimeInt::Base>& timint);

        //! destructor
        virtual ~Generic() = default;

        /*! \brief Reset internal storage before the nonlinear solution starts
         *
         *  We actually (re-)build the nonlinear solver object here.
         *
         *  \warning It is not fully clear how rebuilding the nonlinear solver affects a possible
         *           re-use of the preconditioner for the linear system.
         */
        virtual void reset() = 0;

        //! Refresh solver-side map-dependent state after a redistribution.
        virtual void refresh_after_redistribution() = 0;

        //! Solve the non-linear problem
        virtual Solid::ConvergenceStatus solve() = 0;

        /*! returns the nox group for external and internal use
         *
         *  The nox group has to be initialized in one of the derived setup() routines beforehand.
         */
        ::NOX::Abstract::Group& get_solution_group();
        const ::NOX::Abstract::Group& get_solution_group() const;

        //! Get the number of nonlinear iterations
        virtual int get_num_nln_iterations() const = 0;

       protected:
        //! Returns the global state data container pointer
        std::shared_ptr<Solid::TimeInt::BaseDataGlobalState> data_global_state_ptr()
        {
          return gstate_ptr_;
        }

        //! Returns the global state data container (read-only)
        const Solid::TimeInt::BaseDataGlobalState& data_global_state() const
        {
          return *gstate_ptr_;
        }

        //! Returns the global state data container (read and write)
        Solid::TimeInt::BaseDataGlobalState& data_global_state() { return *gstate_ptr_; }

        //! Returns the structural dynamics data container pointer
        std::shared_ptr<Solid::TimeInt::BaseDataSDyn> data_s_dyn_ptr() { return sdyn_ptr_; }

        //! Returns the structural dynamics data container (read-only)
        const Solid::TimeInt::BaseDataSDyn& data_s_dyn() const { return *sdyn_ptr_; }

        //! Returns the structural dynamics data container (read and write)
        Solid::TimeInt::BaseDataSDyn& data_sdyn() { return *sdyn_ptr_; }

        //! Returns the non-linear solver implicit time integration interface pointer
        std::shared_ptr<Solid::TimeInt::NoxInterface> nox_interface_ptr()
        {
          return noxinterface_ptr_;
        }

        //! Returns the non-linear solver implicit time integration interface (read-only)
        const Solid::TimeInt::NoxInterface& nox_interface() const { return *noxinterface_ptr_; }

        //! Returns the non-linear solver implicit time integration interface (read and write)
        Solid::TimeInt::NoxInterface& nox_interface() { return *noxinterface_ptr_; }

        Solid::Integrator& integrator() { return *int_ptr_; }

        const Solid::Integrator& integrator() const { return *int_ptr_; }

        //! Returns the underlying time integration strategy
        const Solid::TimeInt::Base& tim_int() const { return *timint_ptr_; }

        /*! returns the nox group (pointer) (only for internal use)
         *
         *  The nox group has to be initialized in one of the derived setup() routines. */
        Teuchos::RCP<::NOX::Abstract::Group>& group_ptr();

       private:
        //! global state data container of the time integrator
        std::shared_ptr<Solid::TimeInt::BaseDataGlobalState> gstate_ptr_;

        //! structural dynamics data container of the time integrator
        std::shared_ptr<Solid::TimeInt::BaseDataSDyn> sdyn_ptr_;

        //! required interface pointer to the implicit time integrator (call back)
        std::shared_ptr<Solid::TimeInt::NoxInterface> noxinterface_ptr_;

        //! pointer to the current time integrator
        std::shared_ptr<Solid::Integrator> int_ptr_;

        //! pointer to the time integration strategy
        std::shared_ptr<const Solid::TimeInt::Base> timint_ptr_;

        //! nox group
        Teuchos::RCP<::NOX::Abstract::Group> group_ptr_;

      };  // namespace SOLVER
    }  // namespace SOLVER
  }  // namespace Nln
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
