// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_GROUP_PREPOSTOPERATOR_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_GROUP_PREPOSTOPERATOR_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_abstract_prepostoperator.hpp"
#include "4C_solver_nonlin_nox_forward_decl.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    namespace GROUP
    {
      //! Currently supported group pre/post operators.
      enum PrePostOpType
      {
        prepost_ptc,           //!< pseudo transient continuation pre/post operator
        prepost_tangdis,       //!< tangdis (predictor) pre/post operator
        prepost_impl_generic,  //!< implicit generic (time integrator) pre/post operator
        prepost_rotvecupdate   //!< update of non-additive rotation vector DoFs
      };

      /*!
        @brief Functor to process the pre/post operator object in the parameter list for the
        NOX::Nln::Group objects.

        This is a wrapper class for a user derived  NOX::Nln::Abstract::PrePostOperator (ppo)
        object. All NOX::NLN groups use this class so we don't have to repeat all parsing code in
        each NOX::NLN group class. This class searches the "Group Options" parameter list passed
        into the constructor and if a ppo is found will wrap the object.

        For instructions on how to implement a PrePostOperator, see
        NOX::Nln::Abstract::PrePostOperator or one of the currently supported implementations (enum
        list).
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
        PrePostOperator(Teuchos::ParameterList& groupOptionsSublist);

        //! Destructor.
        virtual ~PrePostOperator() = default;

        //! Resets the pre/post operator.
        virtual void reset(Teuchos::ParameterList& groupOptionsSublist);

        /** User defined method that will be executed at the start of a call to
         * NOX::Nln::Group::computeF().
         *
         * \param F        : full access to the right hand side vector of the NOX::Nln::Group.
         * \param grp      : read only access to the NOX::Nln::Group object.
         */
        virtual void run_pre_compute_f(Core::LinAlg::Vector<double>& F, const NOX::Nln::Group& grp);

        /** User defined method that will be executed at the end of a call to
         * NOX::Nln::Group::computeF().
         *
         * \param F        : full access to the right hand side vector of the NOX::Nln::Group.
         * \param grp      : read only access to the NOX::Nln::Group object.
         */
        virtual void run_post_compute_f(
            Core::LinAlg::Vector<double>& F, const NOX::Nln::Group& grp);

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
            const NOX::Nln::Group& curr_grp);

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
            const NOX::Nln::Group& curr_grp);

        /*! User defined method that will be executed at the beginning
         *  of a call to NOX::Nln::Group::applyJacobianInverse().
         *
         *  \param rhs    : read-only access to the rhs vector
         *  \param result : full access to the result vector
         *  \param xold   : read-only access to the state vector
         *  \param grp    : read only access to the group object
         */
        virtual void run_pre_apply_jacobian_inverse(const ::NOX::Abstract::Vector& rhs,
            ::NOX::Abstract::Vector& result, const ::NOX::Abstract::Vector& xold,
            const NOX::Nln::Group& grp);

        /*! User defined method that will be executed at the end
         *  of a call to NOX::Nln::Group::applyJacobianInverse().
         *
         *  \param rhs    : read-only access to the rhs vector
         *  \param result : full access to the result vector
         *  \param xold   : read-only access to the old state vector
         *  \param grp    : read only access to the group object
         */
        virtual void run_post_apply_jacobian_inverse(const ::NOX::Abstract::Vector& rhs,
            ::NOX::Abstract::Vector& result, const ::NOX::Abstract::Vector& xold,
            const NOX::Nln::Group& grp);

       protected:
        //! Flag that determines if a pre/post operator has been supplied by user.
        bool havePrePostOperator_;

        //! Points to user defined pre/post operator for the linear system.
        Teuchos::RCP<Map> prePostOperatorMapPtr_;
      };  // class PrePostOperator
      namespace PrePostOp
      {
        // non-member function
        /*! Returns the inherent pre/post operator std::map of the "Group Options" sublist.
         *  If the corresponding parameter called "User Defined Pre/Post Operator" is not yet
         *  defined, a empty std::map is generated and set into the parameter list first. */
        NOX::Nln::GROUP::PrePostOperator::Map& get_map(Teuchos::ParameterList& p_grp_opt);
      }  // namespace PrePostOp
    }  // namespace GROUP
  }  // namespace Nln
}  // namespace NOX

inline void NOX::Nln::GROUP::PrePostOperator::run_pre_compute_f(
    Core::LinAlg::Vector<double>& F, const NOX::Nln::Group& grp)
{
  if (havePrePostOperator_)
  {
    Map::iterator it;
    for (it = prePostOperatorMapPtr_->begin(); it != prePostOperatorMapPtr_->end(); ++it)
      it->second->run_pre_compute_f(F, grp);
  }
}

inline void NOX::Nln::GROUP::PrePostOperator::run_post_compute_f(
    Core::LinAlg::Vector<double>& F, const NOX::Nln::Group& grp)
{
  if (havePrePostOperator_)
  {
    Map::iterator it;
    for (it = prePostOperatorMapPtr_->begin(); it != prePostOperatorMapPtr_->end(); ++it)
      it->second->run_post_compute_f(F, grp);
  }
}

inline void NOX::Nln::GROUP::PrePostOperator::run_pre_compute_x(const NOX::Nln::Group& input_grp,
    const Core::LinAlg::Vector<double>& dir, const double& step, const NOX::Nln::Group& curr_grp)
{
  if (havePrePostOperator_)
  {
    Map::iterator it;
    for (it = prePostOperatorMapPtr_->begin(); it != prePostOperatorMapPtr_->end(); ++it)
      it->second->run_pre_compute_x(input_grp, dir, step, curr_grp);
  }
}

inline void NOX::Nln::GROUP::PrePostOperator::run_post_compute_x(const NOX::Nln::Group& input_grp,
    const Core::LinAlg::Vector<double>& dir, const double& step, const NOX::Nln::Group& curr_grp)
{
  if (havePrePostOperator_)
  {
    Map::iterator it;
    for (it = prePostOperatorMapPtr_->begin(); it != prePostOperatorMapPtr_->end(); ++it)
      it->second->run_post_compute_x(input_grp, dir, step, curr_grp);
  }
}

inline void NOX::Nln::GROUP::PrePostOperator::run_pre_apply_jacobian_inverse(
    const ::NOX::Abstract::Vector& rhs, ::NOX::Abstract::Vector& result,
    const ::NOX::Abstract::Vector& xold, const NOX::Nln::Group& grp)
{
  if (havePrePostOperator_)
  {
    Map::iterator it;
    for (it = prePostOperatorMapPtr_->begin(); it != prePostOperatorMapPtr_->end(); ++it)
      it->second->run_pre_apply_jacobian_inverse(rhs, result, xold, grp);
  }
}

inline void NOX::Nln::GROUP::PrePostOperator::run_post_apply_jacobian_inverse(
    const ::NOX::Abstract::Vector& rhs, ::NOX::Abstract::Vector& result,
    const ::NOX::Abstract::Vector& xold, const NOX::Nln::Group& grp)
{
  if (havePrePostOperator_)
  {
    Map::iterator it;
    for (it = prePostOperatorMapPtr_->begin(); it != prePostOperatorMapPtr_->end(); ++it)
      it->second->run_post_apply_jacobian_inverse(rhs, result, xold, grp);
  }
}

FOUR_C_NAMESPACE_CLOSE

#endif
