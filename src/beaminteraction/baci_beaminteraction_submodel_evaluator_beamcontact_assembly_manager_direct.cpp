/*-----------------------------------------------------------*/
/*! \file

\brief Class to assemble pair based contributions into global matrices. The pairs in this class can
be directly assembled into the global matrices.


\level 3

*/


#include "baci_beaminteraction_submodel_evaluator_beamcontact_assembly_manager_direct.H"

#include "baci_beaminteraction_calc_utils.H"
#include "baci_beaminteraction_contact_pair.H"
#include "baci_beaminteraction_str_model_evaluator_datastate.H"
#include "baci_lib_element.H"
#include "baci_linalg_serialdensematrix.H"
#include "baci_linalg_serialdensevector.H"


/**
 *
 */
BEAMINTERACTION::SUBMODELEVALUATOR::BeamContactAssemblyManagerDirect::
    BeamContactAssemblyManagerDirect(
        const std::vector<Teuchos::RCP<BEAMINTERACTION::BeamContactPair>>&
            assembly_contact_elepairs)
    : BeamContactAssemblyManager(), assembly_contact_elepairs_(assembly_contact_elepairs)
{
}


/**
 *
 */
void BEAMINTERACTION::SUBMODELEVALUATOR::BeamContactAssemblyManagerDirect::EvaluateForceStiff(
    Teuchos::RCP<::DRT::Discretization> discret,
    const Teuchos::RCP<const STR::MODELEVALUATOR::BeamInteractionDataState>& data_state,
    Teuchos::RCP<Epetra_FEVector> fe_sysvec, Teuchos::RCP<CORE::LINALG::SparseMatrix> fe_sysmat)
{
  // resulting discrete element force vectors of the two interacting elements
  std::vector<CORE::LINALG::SerialDenseVector> eleforce(2);

  // resulting discrete force vectors (centerline DOFs only!) of the two
  // interacting elements
  std::vector<CORE::LINALG::SerialDenseVector> eleforce_centerlineDOFs(2);

  // linearizations
  std::vector<std::vector<CORE::LINALG::SerialDenseMatrix>> elestiff(
      2, std::vector<CORE::LINALG::SerialDenseMatrix>(2));

  // linearizations (centerline DOFs only!)
  std::vector<std::vector<CORE::LINALG::SerialDenseMatrix>> elestiff_centerlineDOFs(
      2, std::vector<CORE::LINALG::SerialDenseMatrix>(2));

  // element gids of interacting elements
  std::vector<int> elegids(2);

  // are non-zero stiffness values returned which need assembly?
  bool pair_is_active = false;

  for (auto& elepairptr : assembly_contact_elepairs_)
  {
    // Evaluate the pair and check if there is active contact
    pair_is_active =
        elepairptr->Evaluate(&(eleforce_centerlineDOFs[0]), &(eleforce_centerlineDOFs[1]),
            &(elestiff_centerlineDOFs[0][0]), &(elestiff_centerlineDOFs[0][1]),
            &(elestiff_centerlineDOFs[1][0]), &(elestiff_centerlineDOFs[1][1]));

    if (pair_is_active)
    {
      elegids[0] = elepairptr->Element1()->Id();
      elegids[1] = elepairptr->Element2()->Id();

      // assemble force vector and stiffness matrix affecting the centerline DoFs only
      // into element force vector and stiffness matrix ('all DoFs' format, as usual)
      BEAMINTERACTION::UTILS::AssembleCenterlineDofForceStiffIntoElementForceStiff(*discret,
          elegids, eleforce_centerlineDOFs, elestiff_centerlineDOFs, &eleforce, &elestiff);


      // Fixme
      eleforce[0].scale(-1.0);
      eleforce[1].scale(-1.0);

      // assemble the contributions into force vector class variable
      // f_crosslink_np_ptr_, i.e. in the DOFs of the connected nodes
      BEAMINTERACTION::UTILS::FEAssembleEleForceStiffIntoSystemVectorMatrix(
          *discret, elegids, eleforce, elestiff, fe_sysvec, fe_sysmat);
    }

    // Each pair can also directly assembles terms into the global force vector and system matrix.
    elepairptr->EvaluateAndAssemble(discret, fe_sysvec, fe_sysmat, data_state->GetDisColNp());
  }
}
