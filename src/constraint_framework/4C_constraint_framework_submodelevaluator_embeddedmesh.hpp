// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_SUBMODELEVALUATOR_EMBEDDEDMESH_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_SUBMODELEVALUATOR_EMBEDDEDMESH_HPP

#include "4C_config.hpp"

#include "4C_constraint_framework_submodelevaluator_base.hpp"
#include "4C_fem_discretization.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Constraints::EmbeddedMesh
{
  class SolidToSolidMortarManager;
}

namespace Constraints::SubmodelEvaluator
{
  class EmbeddedMeshConstraintManager : public ConstraintBase
  {
   public:
    /*!
    \brief Standard Constructor
    */
    EmbeddedMeshConstraintManager(std::shared_ptr<Core::FE::Discretization> discret_ptr,
        const Core::LinAlg::Vector<double>& dispnp);

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
    bool evaluate_force_stiff(const Core::LinAlg::Vector<double>& displacement_vector,
        std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr,
        std::shared_ptr<Core::LinAlg::SparseMatrix> me_stiff_ptr,
        std::shared_ptr<Core::LinAlg::Vector<double>> me_force_ptr) override;

    //! Evaluate the matrices of the saddle-point system
    void evaluate_coupling_terms(Solid::TimeInt::BaseDataGlobalState& gstate) override
    {
      // Nothing implemented
    }

    //! derived
    void runtime_output_step_state(std::pair<double, int> output_time_and_step) override;

    [[nodiscard]] std::map<Solid::EnergyType, double> get_energy() const override;
    //@}

   private:
    //! Pointer to the mortar manager. This object stores the relevant mortar matrices.
    std::shared_ptr<Constraints::EmbeddedMesh::SolidToSolidMortarManager> mortar_manager_;
  };
}  // namespace Constraints::SubmodelEvaluator

FOUR_C_NAMESPACE_CLOSE

#endif  // FOUR_C_CONSTRAINT_FRAMEWORK_SUBMODELEVALUATOR_EMBEDDEDMESH_HPP
