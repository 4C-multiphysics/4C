/*-----------------------------------------------------------*/
/*! \file

\brief Class to assemble pair based contributions into global matrices. The pairs in this class can
be directly assembled into the global matrices.


\level 3

*/
/*-----------------------------------------------------------*/


#ifndef BACI_BEAMINTERACTION_SUBMODEL_EVALUATOR_BEAMCONTACT_ASSEMBLY_MANAGER_DIRECT_HPP
#define BACI_BEAMINTERACTION_SUBMODEL_EVALUATOR_BEAMCONTACT_ASSEMBLY_MANAGER_DIRECT_HPP


#include "baci_config.hpp"

#include "baci_beaminteraction_submodel_evaluator_beamcontact_assembly_manager.hpp"

BACI_NAMESPACE_OPEN


namespace BEAMINTERACTION
{
  namespace SUBMODELEVALUATOR
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
          const std::vector<Teuchos::RCP<BEAMINTERACTION::BeamContactPair>>&
              assembly_contact_elepairs);


      /**
       * \brief Evaluate all force and stiffness terms and add them to the global matrices.
       * @param discret (in) Pointer to the disretization.
       * @param data_state (in) Beam interaction data state.
       * @param fe_sysvec (out) Global force vector.
       * @param fe_sysmat (out) Global stiffness matrix.
       */
      void EvaluateForceStiff(Teuchos::RCP<DRT::Discretization> discret,
          const Teuchos::RCP<const STR::MODELEVALUATOR::BeamInteractionDataState>& data_state,
          Teuchos::RCP<Epetra_FEVector> fe_sysvec,
          Teuchos::RCP<CORE::LINALG::SparseMatrix> fe_sysmat) override;

      /**
       * \brief Return a const reference to the contact pairs in this assembly manager.
       * @return Reference to the pair vector.
       */
      const std::vector<Teuchos::RCP<BEAMINTERACTION::BeamContactPair>>& GetContactPairs() const
      {
        return assembly_contact_elepairs_;
      }

     protected:
      //! Vector of pairs to be evaluated by this class.
      std::vector<Teuchos::RCP<BEAMINTERACTION::BeamContactPair>> assembly_contact_elepairs_;
    };
  }  // namespace SUBMODELEVALUATOR
}  // namespace BEAMINTERACTION

BACI_NAMESPACE_CLOSE

#endif