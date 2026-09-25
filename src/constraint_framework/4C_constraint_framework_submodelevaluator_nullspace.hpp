// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_SUBMODELEVALUATOR_NULLSPACE_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_SUBMODELEVALUATOR_NULLSPACE_HPP

#include "4C_config.hpp"

#include "4C_constraint_framework_submodelevaluator_base.hpp"
#include "4C_fem_discretization.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Constraints::SubmodelEvaluator
{
  /* !\brief Nullspace constraint submodel evaluator for the solution of pure Neumann problems
   *
   * Pure Neumann problems are of singular nature, due to the missing Dirichlet boundary condition.
   * Such type of systems can be solved by using the kernel / nullspace B as constraint, imposing
   * Bu = 0 for any solution vector u. The approach implemented here enforces this constraint by
   * adding the given condition by the means of a Lagrange Multiplier. This results in a sparse
   * saddle point system to be solved.
   *
   * P. Bochev and R. B. Lehoucq: On the Finite Element Solution of the Pure Neumann Problem,
   * SIAM Review, 47(1):50-66, 2005, http://dx.doi.org/10.1137/S0036144503426074
   *
   * P. Bochev and R. B. Lehoucq: Energy Principles and Finite Element Methods for Pure Traction
   * Linear Elasticity, Computational Methods in Applied Mathematics, 11(2):173-191, 2011,
   * https://doi.org/10.2478/cmam-2011-0009
   *
   * M. Kutcha, K. A. Mardal and M. Mortensen: On the singular Neumann problem in linear elasticity,
   * Numerical Linear Algebra with Applications, 26(1):e2212, 2019, https://doi.org/10.1002/nla.2212
   *
   */
  class NullspaceConstraintManager : public ConstraintBase
  {
   public:
    /*!
    \brief Standard Constructor
    */
    NullspaceConstraintManager(std::shared_ptr<Core::FE::Discretization> discret_ptr);

    //! @name Public evaluation methods

    /*!
     * \brief Reset the constraint stiffness matrix and delete node pairs
     */
    void reset() override
    {
      // Nothing implemented
    }

    /*! Evaluate the current right-hand-side vector and tangential stiffness matrix at \f$t_{n+1}\f$
     */
    void evaluate_force_stiff(const Core::LinAlg::Vector<double>& displacement_vector,
        std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr,
        std::shared_ptr<Core::LinAlg::SparseMatrix> me_stiff_ptr,
        std::shared_ptr<Core::LinAlg::Vector<double>> me_force_ptr) override;

    //! Evaluate the off-diagonal matrix blocks of the saddle-point system
    void evaluate_coupling_terms(Solid::TimeInt::BaseDataGlobalState& gstate) override;

    std::map<Solid::EnergyType, double> get_energy() const override;

    //! Return the map used for constraint enforcement
    std::shared_ptr<Core::LinAlg::Map> get_constraint_map() { return constraint_map_; }
    //@}

   private:
    //! Row map of the additional constraint degrees of freedom
    std::shared_ptr<Core::LinAlg::Map> constraint_map_;

    //! Row map of the degrees of freedom constrained by the nullspace condition
    std::shared_ptr<Core::LinAlg::Map> dof_condition_map_;

    //! Dimension of the nullspace used for constraint enforcement
    int nullspace_dimension_;

    //! The nullspace components to be used as constraint
    std::vector<int> active_mode_ids_;
  };
}  // namespace Constraints::SubmodelEvaluator

FOUR_C_NAMESPACE_CLOSE

#endif  // FOUR_C_CONSTRAINT_FRAMEWORK_SUBMODELEVALUATOR_NULLSPACE_HPP
