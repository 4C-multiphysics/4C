// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_ABSTRACT_PREPOSTOPERATOR_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_ABSTRACT_PREPOSTOPERATOR_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_forward_decl.hpp"

#include <NOX_Observer.hpp>  // base class

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::LinAlg
{
  template <typename T>
  class Vector;
  class SparseOperator;
}  // namespace Core::LinAlg

namespace NOX
{
  namespace Nln
  {
    class Group;
    class LinearSystem;
    namespace Abstract
    {
      /*!
        \brief %NOX::NLN's pure virtual class to allow users to insert pre and post
        operations into different NOX::NLN classes.

        The user should implement their own concrete implementation of this class
        and register it as a Teuchos::RCP<NOX::Nln::Abstract::PrePostOperator>
        in the corresponding sublist.

        For example: To create and use a user defined pre/post operator for the linear system you
        have to insert the object in the "Linear Solver" sub-sublist:

        <ol>

        <li> Create a pre/post operator that derives from
        NOX::Nln::Abstract::PrePostOperator. For example, the pre/post operator \c
        Foo might be defined as shown below.

        \code
        class Foo : public NOX::Nln::Abstract::PrePostOperator
        {
        // Insert class definition here
        }
        \endcode

        <li> Create the appropriate entries in the linear solver parameter list which belongs to
        the current direction method, as follows.

        \code
        Teuchos::RCP<Foo> foo = Teuchos::rcp(new Foo);
        const std::string& dir_str = paramsPtr->sublist("Direction").get<std::string>("Method");
        Teuchos::ParameterList& p_linsolver = paramsPtr->sublist("Direction").sublist(dir_str).
            sublist("Linear Solver").set<Teuchos::RCP<NOX::Nln::Abstract::PrePostOperator> >
            ("User Defined Pre/Post Operator",foo);
        \endcode

        <li> See also the nox_nln_solver_ptc implementation for a short example.

        </ol>
       */

      class PrePostOperator : public ::NOX::Observer
      {
       public:
        //! constructor (does nothing)
        PrePostOperator() {};

        //! Copy constructor (does nothing)
        PrePostOperator(const NOX::Nln::Abstract::PrePostOperator& source) {};

        /** @name Solver Pre/Post Operator
         *  Non-linear solver pre/post-operator functions. See the  ::NOX::Solver::PrePostOperator
         *  class and its derived classes for more information. The virtual functions can be found
         * in the base class.
         */
        ///@{
        /** User defined method that will be executed at the start of a call to
        ::NOX::Solver::Generic::iterate(). virtual void runPreIterate(const ::NOX::Solver::Generic&
        solver); */

        /** User defined method that will be executed at the end of a call to
        ::NOX::Solver::Generic::iterate(). virtual void runPostIterate(const ::NOX::Solver::Generic&
        solver); */

        /** User defined method that will be executed at the start of a call to
        ::NOX::Solver::Generic::solve().
        virtual void runPreSolve(const ::NOX::Solver::Generic& solver); */

        /** User defined method that will be executed at the end of a call to
        ::NOX::Solver::Generic::solve(). virtual void runPostSolve(const ::NOX::Solver::Generic&
        solver); */
        ///@}

        /** @name Nln::LinearSystem Pre/Post Operator
         *  This pre/post operator is used in the NOX::Nln::LinearSystem class and its derived
         * classes.
         */
        ///@{
        /** User defined method that will be executed at the start
         *  of a call to NOX::Nln::LinearSystem::apply_jacobian_inverse().
         *
         * \param rhs    : full access to the rhs vector
         * \param jac    : full access to the jacobian
         * \param linsys : read only access to the linear system object
         */
        virtual void run_pre_apply_jacobian_inverse(::NOX::Abstract::Vector& rhs,
            Core::LinAlg::SparseOperator& jac, const NOX::Nln::LinearSystem& linsys)
        {
        }

        /** User defined method that will be executed at the end
         *  of a call to NOX::Nln::LinearSystem::apply_jacobian_inverse().
         *
         * \param result : full access to the result vector
         * \param rhs    : full access to the rhs vector
         * \param jac    : full access to the jacobian
         * \param linsys : read only access to the linear system object
         */
        virtual void run_post_apply_jacobian_inverse(::NOX::Abstract::Vector& result,
            ::NOX::Abstract::Vector& rhs, Core::LinAlg::SparseOperator& jac,
            const NOX::Nln::LinearSystem& linsys)
        {
        }

        /** User defined method that will be executed at the start of a call to
         * NOX::Nln::LinearSystem::apply_jacobian_inverse().
         *
         * \param jac    : full access to the jacobian operator
         * \param x      : read only access to the current solution point
         * \param linsys : read only access to the linear system object
         */
        virtual void run_pre_compute_jacobian(Core::LinAlg::SparseOperator& jac,
            const Core::LinAlg::Vector<double>& x, const NOX::Nln::LinearSystem& linsys)
        {
        }

        /** User defined method that will be executed at the end of a call to
         * NOX::Nln::LinearSystem::apply_jacobian_inverse().
         *
         * \param jac    : full access to the jacobian operator
         * \param x      : read only access to the current solution point
         * \param linsys : read only access to the linear system object
         */
        virtual void run_post_compute_jacobian(Core::LinAlg::SparseOperator& jac,
            const Core::LinAlg::Vector<double>& x, const NOX::Nln::LinearSystem& linsys)
        {
        }

        /** User defined method that will be executed at the start of a call to
         * NOX::Nln::LinearSystem::compute_f_and_jacobian().
         *
         * \param rhs    : full access to the right-hand-side vector
         * \param jac    : full access to the jacobian operator
         * \param x      : read only access to the current solution point
         * \param linsys : read only access to the linear system object
         */
        virtual void run_pre_compute_f_and_jacobian(Core::LinAlg::Vector<double>& rhs,
            Core::LinAlg::SparseOperator& jac, const Core::LinAlg::Vector<double>& x,
            const NOX::Nln::LinearSystem& linsys)
        {
        }

        /** User defined method that will be executed at the end of a call to
         * NOX::Nln::LinearSystem::compute_f_and_jacobian().
         *
         * \param rhs    : full access to the right-hand-side vector
         * \param jac    : full access to the jacobian operator
         * \param x      : read only access to the current solution point
         * \param linsys : read only access to the linear system object
         */
        virtual void run_post_compute_f_and_jacobian(Core::LinAlg::Vector<double>& rhs,
            Core::LinAlg::SparseOperator& jac, const Core::LinAlg::Vector<double>& x,
            const NOX::Nln::LinearSystem& linsys)
        {
        }

        ///@}

        /** @name Nln::Group Pre/Post Operator
         *  This pre/post operator is used in the NOX::Nln::Group class and its derived classes.
         */
        ///@{
        /** User defined method that will be executed at the start of a call to
         * NOX::Nln::Group::computeF().
         *
         * \param F        : full access to the right hand side vector of the NOX::Nln::Group.
         * \param grp      : read only access to the NOX::Nln::Group object.
         */
        virtual void run_pre_compute_f(Core::LinAlg::Vector<double>& F, const NOX::Nln::Group& grp)
        {
        }

        /** User defined method that will be executed at the end of a call to
         * NOX::Nln::Group::computeF().
         *
         * \param F        : full access to the right hand side vector of the NOX::Nln::Group.
         * \param grp      : read only access to the NOX::Nln::Group object.
         */
        virtual void run_post_compute_f(Core::LinAlg::Vector<double>& F, const NOX::Nln::Group& grp)
        {
        }

        /** User defined method that will be executed at the start of a call to
         * NOX::Nln::Group::computeX().
         *
         * \param input_grp: read only access to the input group (holds the old X).
         * \param dir      : read only access to the direction vector (step length equal 1.0).
         * \param step     : read only access to the current step length (line search).
         * \param curr_grp : read only access to the called/current group (will hold the new X).
         */
        virtual void run_pre_compute_x(const NOX::Nln::Group& input_grp,
            const Core::LinAlg::Vector<double>& dir, const double& step,
            const NOX::Nln::Group& curr_grp)
        {
        }

        /** User defined method that will be executed at the end of a call to
         * NOX::Nln::Group::computeX().
         *
         * \param input_grp: read only access to the input group (holds the old X).
         * \param dir      : read only access to the direction vector (step length equal 1.0).
         * \param step     : read only access to the current step length (line search).
         * \param curr_grp : read only access to the called/current group (holds the new X).
         */
        virtual void run_post_compute_x(const NOX::Nln::Group& input_grp,
            const Core::LinAlg::Vector<double>& dir, const double& step,
            const NOX::Nln::Group& curr_grp)
        {
        }

        /*! User defined method that will be executed at the beginning
         *  of a call to NOX::Nln::Group::applyJacobianInverse().
         *
         *  \param rhs    : read-only access to the rhs vector
         *  \param result : full access to the result vector
         *  \param xold   : read-only access to the jacobian
         *  \param grp    : read only access to the group object
         */
        virtual void run_pre_apply_jacobian_inverse(const ::NOX::Abstract::Vector& rhs,
            ::NOX::Abstract::Vector& result, const ::NOX::Abstract::Vector& xold,
            const NOX::Nln::Group& grp)
        {
        }

        /*! User defined method that will be executed at the end
         *  of a call to NOX::Nln::Group::applyJacobianInverse().
         *
         *  \param rhs    : read-only access to the rhs vector
         *  \param result : full access to the result vector
         *  \param xold   : read-only access to the jacobian
         *  \param grp    : read only access to the group object
         */
        virtual void run_post_apply_jacobian_inverse(const ::NOX::Abstract::Vector& rhs,
            ::NOX::Abstract::Vector& result, const ::NOX::Abstract::Vector& xold,
            const NOX::Nln::Group& grp)
        {
        }

        ///@}

        /** @name Nln::LineSearch Pre/Post Operator
         *  This pre/post operator is used in the NOX::Nln::LineSearch classes.
         */
        ///@{
        /** User defined method that will be executed before the step is modified in
         *  the line search routine.
         *
         * \param solver     : Access to the underlying solver object.
         * \param linesearch : Access to the line search object. */
        virtual void run_pre_modify_step_length(
            const ::NOX::Solver::Generic& solver, const ::NOX::LineSearch::Generic& linesearch)
        {
        }

        ///@}
      };  // class PrePostOperator
    }  // namespace Abstract
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
