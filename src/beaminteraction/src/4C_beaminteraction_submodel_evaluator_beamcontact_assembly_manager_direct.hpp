// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_SUBMODEL_EVALUATOR_BEAMCONTACT_ASSEMBLY_MANAGER_DIRECT_HPP
#define FOUR_C_BEAMINTERACTION_SUBMODEL_EVALUATOR_BEAMCONTACT_ASSEMBLY_MANAGER_DIRECT_HPP


#include "4C_config.hpp"

#include "4C_beaminteraction_submodel_evaluator_beamcontact_assembly_manager.hpp"

FOUR_C_NAMESPACE_OPEN


namespace BeamInteraction
{
  namespace SubmodelEvaluator
  {
    /**
     * \brief This class collects local force and stiffness terms of the pairs and adds them
     * directly into the global force vector and stiffness matrix.
     */
    class BeamContactAssemblyManagerDirect : public BeamContactAssemblyManager
    {
     public:
      /**
       * \brief Constructor.
       * @param assembly_contact_elepairs (in) Vector with element pairs to be evaluated by this
       * class.
       */
      BeamContactAssemblyManagerDirect(
          const std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>&
              assembly_contact_elepairs);


      /**
       * \brief Evaluate all force and stiffness terms and add them to the global matrices.
       * @param discret (in) Pointer to the disretization.
       * @param data_state (in) Beam interaction data state.
       * @param fe_sysvec (out) Global force vector.
       * @param fe_sysmat (out) Global stiffness matrix.
       */
      void evaluate_force_stiff(std::shared_ptr<Core::FE::Discretization> discret,
          const std::shared_ptr<const Solid::ModelEvaluator::BeamInteractionDataState>& data_state,
          std::shared_ptr<Core::LinAlg::FEVector<double>> fe_sysvec,
          std::shared_ptr<Core::LinAlg::SparseMatrix> fe_sysmat) override;

      /**
       * \brief Return a const reference to the contact pairs in this assembly manager.
       * @return Reference to the pair vector.
       */
      const std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>& get_contact_pairs()
          const
      {
        return assembly_contact_elepairs_;
      }

     protected:
      //! Vector of pairs to be evaluated by this class.
      std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>> assembly_contact_elepairs_;
    };
  }  // namespace SubmodelEvaluator
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
