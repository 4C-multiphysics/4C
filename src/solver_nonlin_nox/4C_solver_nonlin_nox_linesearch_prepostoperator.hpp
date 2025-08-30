// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_LINESEARCH_PREPOSTOPERATOR_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_LINESEARCH_PREPOSTOPERATOR_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_abstract_prepostoperator.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    namespace LineSearch
    {
      enum PrePostOpType
      {
        prepost_output_every_iter
      };

      class PrePostOperator
      {
       public:
        using map = std::map<enum PrePostOpType, Teuchos::RCP<NOX::Nln::Abstract::PrePostOperator>>;

        /// disallow the following
        PrePostOperator() = delete;
        PrePostOperator(const PrePostOperator& ppo) = delete;
        PrePostOperator& operator=(const PrePostOperator& ppo) = delete;

        /// allowed constructor
        PrePostOperator(Teuchos::ParameterList& linesearchSublist);

        virtual ~PrePostOperator() = default;

        //! Resets the pre/post operator.
        virtual void reset(Teuchos::ParameterList& linesearchSublist);

        /*! Returns the inherent pre/post operator std::map of the "Line Search"
         *  sublist. If the corresponding parameter called "User Defined Pre/Post Operator"
         *  is not yet defined, a empty std::map is generated and set into the
         *  parameter list first. */
        static map& get_map(Teuchos::ParameterList& p_ls_list);

        /** User defined method that will be executed before the step is modified in
         *  the line search routine.
         *
         * \param solver     : Access to the underlying solver object.
         * \param linesearch : Access to the line search object. */
        virtual void run_pre_modify_step_length(
            const ::NOX::Solver::Generic& solver, const ::NOX::LineSearch::Generic& linesearch);

       protected:
        //! Flag that determines if a pre/post operator has been supplied by user.
        bool havePrePostOperator_ = false;

        //! Points to user defined pre/post operator for the linear system.
        Teuchos::RCP<map> prePostOperatorMapPtr_ = Teuchos::null;
      };

      inline void PrePostOperator::run_pre_modify_step_length(
          const ::NOX::Solver::Generic& solver, const ::NOX::LineSearch::Generic& linesearch)
      {
        if (havePrePostOperator_)
        {
          map::iterator it;
          for (auto& it : *prePostOperatorMapPtr_)
            it.second->run_pre_modify_step_length(solver, linesearch);
        }
      }
    }  // namespace LineSearch
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
