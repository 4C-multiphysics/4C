// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_SUBMODELEVALUATOR_BASE_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_SUBMODELEVALUATOR_BASE_HPP

#include "4C_config.hpp"

#include "4C_constraint_framework_equation.hpp"
#include "4C_constraint_framework_input.hpp"
#include "4C_structure_new_enum_lists.hpp"
#include "4C_structure_new_model_evaluator_generic.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Constraints::SubmodelEvaluator
{
  /*! \brief Interface class of all submodel evaluators managing
   *  constraint terms
   *
   * This is the abstract base class of all submodel evaluators for constraint
   * problems
   *
   *  This class summarizes the functionality which all submodel evaluators share
   *  and/or have to implement. Look in the derived classes for examples.
   *
   * Definitions:
   * \f$ \begin{bmatrix} K_{sys} + Q_{dd} & Q_{dL} \\ Q_{Ld} & 0 \end{bmatrix} +
   * \begin{bmatrix}  \Delta D \\ \lambda \end{bmatrix} =
   * \begin{bmatrix}  r_d \\  r_L \end{bmatrix}\f$
   */
  class ConstraintBase
  {
   public:
    //! constructor
    ConstraintBase();

    //! destructor
    virtual ~ConstraintBase() = default;

    /*! Evaluate the current right-hand-side vector and tangential stiffness matrix at \f$t_{n+1}\f$
     */
    virtual bool evaluate_force_stiff(const Core::LinAlg::Vector<double>& displacement_vector,
        std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr,
        std::shared_ptr<Core::LinAlg::SparseMatrix> me_stiff_ptr,
        std::shared_ptr<Core::LinAlg::Vector<double>> me_force_ptr);

    //! Evaluate the matrices of the saddle-point system
    virtual void evaluate_coupling_terms(Solid::TimeInt::BaseDataGlobalState& gstate);

    //! Reset the sub model evaluator
    virtual void reset() = 0;

    //! Return the Penalty-Parameter
    double& get_penalty_parameter_ptr() { return penalty_parameter_; }

    //! Set the constraint rows owned by this rank
    void set_owned_constraint_row_ids(std::vector<int> row_ids);

    //! Generate the runtime output
    virtual void runtime_output_step_state(std::pair<double, int> output_time_and_step) {};

    [[nodiscard]] virtual std::map<Solid::EnergyType, double> get_energy() const = 0;

   private:
    //! Column Map
    std::shared_ptr<Core::LinAlg::Map> n_condition_map_;

    //! Penalty parameter
    double penalty_parameter_;

    //! Constraint rows owned by this rank
    std::vector<int> owned_constraint_row_ids_;

    //! Use owned_constraint_row_ids_ for the constraint map
    bool use_explicit_constraint_row_ids_ = false;

   protected:
    //! Vector containing all constraint equation objects
    std::vector<std::shared_ptr<ConstraintEquationBase>> constraint_equations_;

    //! Pointer to the structural stiffness matrix \f$ K_{sys} \f$
    Core::LinAlg::SparseMatrix* stiff_ptr_;

    //! Enforcement Strategy
    EnforcementStrategy strategy_;

    //! Pointer to the discretization
    std::shared_ptr<const Core::FE::Discretization> discret_ptr_ = nullptr;

    //! System Matrix \f$ Q_{dd} \f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> Q_dd_ = nullptr;

    //! System Matrix \f$ Q_{dL} \f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> Q_dL_ = nullptr;

    //! System Matrix \f$ Q_{Ld} \f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> Q_Ld_ = nullptr;

    //! coupling conditions evaluated at current displacements
    std::shared_ptr<Core::LinAlg::Vector<double>> constraint_residual_;
  };

}  // namespace Constraints::SubmodelEvaluator

FOUR_C_NAMESPACE_CLOSE
#endif
