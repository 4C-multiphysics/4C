// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MORTAR_PARAMSINTERFACE_HPP
#define FOUR_C_MORTAR_PARAMSINTERFACE_HPP

#include "4C_config.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

namespace Mortar
{
  //! Actions to be performed by mortar/contact framework
  enum ActionType : int
  {
    eval_none,        /*!< No evaluation type has been chosen */
    eval_force_stiff, /*!< Evaluation of the contact/meshtying right-hand-side and the
                         evaluate_force_stiff contact/meshtying jacobian. We call this method also,
                         when we are only interested in the jacobian, since the created overhead is
                         negligible. */
    eval_force,       /*!< Evaluation of the contact/meshtying right-hand-side only. Necessary and
                         meaningful for line search strategies for example. */
    eval_run_pre_evaluate,   /*!< Run at the very beginning of a call to
                                Solid::ModelEvaluatorManager::EvaluteForce/Stiff/ForceStiff */
    eval_run_post_evaluate,  /*!< Run in the end of a call to
                                Solid::ModelEvaluatorManager::EvaluteForce/Stiff/ForceStiff */
    eval_run_post_compute_x, /*!< recover internal quantities, e.g. Lagrange multipliers */
    eval_reset, /*!< reset internal quantities, e.g. displacement state and/or Lagrange multipliers
                 */
    eval_run_pre_compute_x, /*!< augment the solution direction at the very beginning of a ComputeX
                             */
    eval_run_post_iterate,  /*!< run in the end of a ::NOX::Solver::Step() call */
    eval_run_pre_solve,     /*!< run at the beginning of a ::NOX::Solver::solve() call */
    eval_contact_potential, /*!< Evaluate the contact potential */
    eval_wgap_gradient_error,   /*!< Evaluate the error of the weighted gap gradient */
    eval_static_constraint_rhs, /*!< Evaluate only the contributions to the constraint rhs. The
                                   active set is not updated during the evaluation. */
    eval_run_post_apply_jacobian_inverse,       /*!< run in the end of a
                                                   NOX::Nln::LinearSystem::apply_jacobian_inverse()
                                                   call */
    remove_condensed_contributions_from_str_rhs /*!< remove any condensed contact contributions from
                                                   the structural rhs */
  };

  /*! \brief Convert Mortar::ActionType enum to string
   *
   * @param[in] act ActionType encoded as enum
   * @return String describing the action type
   */
  static inline std::string action_type_to_string(const enum ActionType& act)
  {
    switch (act)
    {
      case eval_none:
        return "eval_none";
      case eval_force_stiff:
        return "eval_force_stiff";
      case eval_force:
        return "eval_force";
      case eval_run_pre_evaluate:
        return "eval_run_pre_evaluate";
      case eval_run_post_evaluate:
        return "eval_run_post_evaluate";
      case Mortar::eval_reset:
        return "eval_reset";
      case Mortar::eval_run_post_compute_x:
        return "eval_run_post_compute_x";
      case Mortar::eval_run_pre_compute_x:
        return "eval_run_pre_compute_x";
      case Mortar::eval_run_post_iterate:
        return "eval_run_post_iterate";
      case Mortar::eval_contact_potential:
        return "eval_contact_potential";
      case Mortar::eval_wgap_gradient_error:
        return "eval_wgap_gradient_error";
      case Mortar::eval_static_constraint_rhs:
        return "eval_static_constraint_rhs";
      case Mortar::eval_run_post_apply_jacobian_inverse:
        return "eval_run_post_apply_jacobian_inverse";
      case remove_condensed_contributions_from_str_rhs:
        return "remove_condensed_contributions_from_str_rhs";
      case eval_run_pre_solve:
        return "eval_run_pre_solve";
      default:
        return "unknown [enum = " + std::to_string(act) + "]";
    }
    return "";
  };

  /*! \brief Mortar parameter interface
   *
   * Necessary for the communication between the structural time integration framework and the
   * mortar strategies.
   */
  class ParamsInterface
  {
   public:
    //! destructor
    virtual ~ParamsInterface() = default;

    //! Return the mortar/contact action type
    virtual enum ActionType get_action_type() const = 0;

    //! Get the nonlinear iteration number
    virtual int get_nln_iter() const = 0;

    //! Get the current time step counter \f$(n+1)\f$
    virtual int get_step_np() const = 0;

    /*! \brief Get time step number from which the current simulation has been restarted
     *
     * Equal to 0 if no restart has been performed.
     */
    virtual int get_restart_step() const = 0;

  };  // class ParamsInterface
}  // namespace Mortar


FOUR_C_NAMESPACE_CLOSE

#endif
