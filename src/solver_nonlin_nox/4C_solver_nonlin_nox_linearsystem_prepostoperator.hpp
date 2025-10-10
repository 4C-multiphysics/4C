// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_LINEARSYSTEM_PREPOSTOPERATOR_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_LINEARSYSTEM_PREPOSTOPERATOR_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_abstract_prepostoperator.hpp"  // wrapped abstract class
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SparseOperator;
}  // namespace Core::LinAlg

namespace NOX
{
  namespace Nln
  {
    class LinearSystem;
    namespace LinSystem
    {
      //! currently supported pre/post operators
      enum PrePostOpType
      {
        prepost_ptc,  //!< pseudo transient continuation pre/post operator
        prepost_dbc   //!< dirichlet boundary condition pre/post operator
      };

      /*!
        @brief Functor to process the pre/post operator object in the parameter list for the linear
        system.

        This is a wrapper class for a user derived  NOX::Nln::Abstract::PrePostOperator (ppo)
        object. All NOX::NLN linear systems use this class so we don't have to repeat all parsing
        code in each NOX::NLN linear system class. This class searches the "Linear Solver" parameter
        list passed into the constructor and if a ppo is found will wrap the object.

        For instructions on how to implement a PrePostOperator, see
        NOX::Nln::Abstract::PrePostOperator.
      */
      class PrePostOperator
      {
       public:
        using Map = std::map<enum PrePostOpType, Teuchos::RCP<NOX::Nln::Abstract::PrePostOperator>>;

       private:
        //! Disallow default constructor.
        PrePostOperator();

        //! Disallow copy constructor.
        PrePostOperator(const PrePostOperator& p);

        //! Disallow assignment operator.
        PrePostOperator& operator=(const PrePostOperator& ppo);

       public:
        //! Allowed constructor.
        PrePostOperator(Teuchos::ParameterList& linearSolverSubList);

        //! Destructor.
        virtual ~PrePostOperator() = default;

        //! Resets the pre/post operator.
        virtual void reset(Teuchos::ParameterList& linearSolverSublist);

        /*! User defined method that will be executed at the start
            of a call to NOX::Nln::LinearSystem::apply_jacobian_inverse().

           \param rhs    : full access to the rhs vector
           \param jac    : full access to the jacobian
           \param linsys : read only access to the linear system object
         */
        virtual void run_pre_apply_jacobian_inverse(::NOX::Abstract::Vector& rhs,
            Core::LinAlg::SparseOperator& jac, const NOX::Nln::LinearSystem& linsys);

        /*! User defined method that will be executed at the end
            of a call to NOX::Nln::LinearSystem::apply_jacobian_inverse().

           \param result : full access to the result vector
           \param rhs    : full access to the rhs vector
           \param jac    : full access to the jacobian
           \param linsys : read only access to the linear system object
         */
        virtual void run_post_apply_jacobian_inverse(::NOX::Abstract::Vector& result,
            ::NOX::Abstract::Vector& rhs, Core::LinAlg::SparseOperator& jac,
            const NOX::Nln::LinearSystem& linsys);

        /** User defined method that will be executed at the start of a call to
         * NOX::Nln::LinearSystem::apply_jacobian_inverse().
         *
         * \param jac    : full access to the jacobian operator
         * \param x      : read only access to the current solution point
         * \param linsys : read only access to the linear system object
         */
        virtual void run_pre_compute_jacobian(Core::LinAlg::SparseOperator& jac,
            const Core::LinAlg::Vector<double>& x, const NOX::Nln::LinearSystem& linsys);

        /** User defined method that will be executed at the end of a call to
         * NOX::Nln::LinearSystem::apply_jacobian_inverse().
         *
         * \param jac    : full access to the jacobian operator
         * \param x      : read only access to the current solution point
         * \param linsys : read only access to the linear system object
         */
        virtual void run_post_compute_jacobian(Core::LinAlg::SparseOperator& jac,
            const Core::LinAlg::Vector<double>& x, const NOX::Nln::LinearSystem& linsys);

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
            const NOX::Nln::LinearSystem& linsys);

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
            const NOX::Nln::LinearSystem& linsys);

       protected:
        //! Flag that determines if a pre/post operator has been supplied by user.
        bool havePrePostOperator_;

        //! Points to user defined pre/post operator for the linear system.
        Teuchos::RCP<Map> prePostOperatorMapPtr_;
      };  // class PrePostOperator
      namespace PrePostOp
      {
        // non-member function
        /*! Returns the inherent pre/post operator std::map of the "Linear Solver" sublist.
         *  If the corresponding parameter called "User Defined Pre/Post Operator" is not yet
         *  defined, a empty std::map is generated and set into the parameter list first. */
        NOX::Nln::LinSystem::PrePostOperator::Map& get_map(Teuchos::ParameterList& p_linsolver);
      }  // namespace PrePostOp
    }  // namespace LinSystem
  }  // namespace Nln
}  // namespace NOX

inline void NOX::Nln::LinSystem::PrePostOperator::run_pre_apply_jacobian_inverse(
    ::NOX::Abstract::Vector& rhs, Core::LinAlg::SparseOperator& jac,
    const NOX::Nln::LinearSystem& linsys)
{
  if (havePrePostOperator_)
  {
    Map::iterator it;
    for (it = prePostOperatorMapPtr_->begin(); it != prePostOperatorMapPtr_->end(); ++it)
      it->second->run_pre_apply_jacobian_inverse(rhs, jac, linsys);
  }
}

inline void NOX::Nln::LinSystem::PrePostOperator::run_post_apply_jacobian_inverse(
    ::NOX::Abstract::Vector& result, ::NOX::Abstract::Vector& rhs,
    Core::LinAlg::SparseOperator& jac, const NOX::Nln::LinearSystem& linsys)
{
  if (havePrePostOperator_)
  {
    Map::iterator it;
    for (it = prePostOperatorMapPtr_->begin(); it != prePostOperatorMapPtr_->end(); ++it)
      it->second->run_post_apply_jacobian_inverse(result, rhs, jac, linsys);
  }
}

inline void NOX::Nln::LinSystem::PrePostOperator::run_pre_compute_jacobian(
    Core::LinAlg::SparseOperator& jac, const Core::LinAlg::Vector<double>& x,
    const NOX::Nln::LinearSystem& linsys)
{
  if (havePrePostOperator_)
  {
    Map::iterator it;
    for (it = prePostOperatorMapPtr_->begin(); it != prePostOperatorMapPtr_->end(); ++it)
      it->second->run_pre_compute_jacobian(jac, x, linsys);
  }
}

inline void NOX::Nln::LinSystem::PrePostOperator::run_post_compute_jacobian(
    Core::LinAlg::SparseOperator& jac, const Core::LinAlg::Vector<double>& x,
    const NOX::Nln::LinearSystem& linsys)
{
  if (havePrePostOperator_)
  {
    Map::iterator it;
    for (it = prePostOperatorMapPtr_->begin(); it != prePostOperatorMapPtr_->end(); ++it)
      it->second->run_post_compute_jacobian(jac, x, linsys);
  }
}

inline void NOX::Nln::LinSystem::PrePostOperator::run_pre_compute_f_and_jacobian(
    Core::LinAlg::Vector<double>& rhs, Core::LinAlg::SparseOperator& jac,
    const Core::LinAlg::Vector<double>& x, const NOX::Nln::LinearSystem& linsys)
{
  if (havePrePostOperator_)
  {
    Map::iterator it;
    for (it = prePostOperatorMapPtr_->begin(); it != prePostOperatorMapPtr_->end(); ++it)
      it->second->run_pre_compute_f_and_jacobian(rhs, jac, x, linsys);
  }
}

inline void NOX::Nln::LinSystem::PrePostOperator::run_post_compute_f_and_jacobian(
    Core::LinAlg::Vector<double>& rhs, Core::LinAlg::SparseOperator& jac,
    const Core::LinAlg::Vector<double>& x, const NOX::Nln::LinearSystem& linsys)
{
  if (havePrePostOperator_)
  {
    Map::iterator it;
    for (it = prePostOperatorMapPtr_->begin(); it != prePostOperatorMapPtr_->end(); ++it)
      it->second->run_post_compute_f_and_jacobian(rhs, jac, x, linsys);
  }
}

FOUR_C_NAMESPACE_CLOSE

#endif
