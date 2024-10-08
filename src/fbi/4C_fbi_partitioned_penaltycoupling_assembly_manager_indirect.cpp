/*-----------------------------------------------------------*/
/*! \file

\brief Class to assemble pair based contributions into global matrices. The pairs in this class can
not be directly assembled into the global matrices. They have to be assembled into the global
coupling matrices M and D first.


\level 3

*/


#include "4C_fbi_partitioned_penaltycoupling_assembly_manager_indirect.hpp"

#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_params.hpp"
#include "4C_fbi_beam_to_fluid_mortar_manager.hpp"
#include "4C_fbi_calc_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"

#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
BEAMINTERACTION::SUBMODELEVALUATOR::PartitionedBeamInteractionAssemblyManagerIndirect::
    PartitionedBeamInteractionAssemblyManagerIndirect(
        std::vector<Teuchos::RCP<BEAMINTERACTION::BeamContactPair>>& assembly_contact_elepairs,
        Teuchos::RCP<const Core::FE::Discretization>& discretization1,
        Teuchos::RCP<const Core::FE::Discretization>& discretization2,
        Teuchos::RCP<FBI::BeamToFluidMeshtyingParams> beam_contact_params_ptr)
    : PartitionedBeamInteractionAssemblyManager(assembly_contact_elepairs)
{
  // Create the mortar manager.
  mortar_manager_ = Teuchos::make_rcp<BEAMINTERACTION::BeamToFluidMortarManager>(discretization1,
      discretization2, beam_contact_params_ptr, discretization1->dof_row_map()->MaxAllGID());

  // Setup the mortar manager.
  mortar_manager_->setup();
  mortar_manager_->set_local_maps(assembly_contact_elepairs_);
}


/**
 *
 */
void BEAMINTERACTION::SUBMODELEVALUATOR::PartitionedBeamInteractionAssemblyManagerIndirect::
    evaluate_force_stiff(const Core::FE::Discretization& discretization1,
        const Core::FE::Discretization& discretization2, Teuchos::RCP<Epetra_FEVector>& ff,
        Teuchos::RCP<Epetra_FEVector>& fb, Teuchos::RCP<Core::LinAlg::SparseOperator> cff,
        Teuchos::RCP<Core::LinAlg::SparseMatrix>& cbb,
        Teuchos::RCP<Core::LinAlg::SparseMatrix>& cfb,
        Teuchos::RCP<Core::LinAlg::SparseMatrix>& cbf,
        Teuchos::RCP<const Core::LinAlg::Vector<double>> fluid_vel,
        Teuchos::RCP<const Core::LinAlg::Vector<double>> beam_vel)
{
  Teuchos::RCP<Teuchos::Time> t =
      Teuchos::TimeMonitor::getNewTimer("FBI::PartitionedAssemblyManagerIndirect");
  Teuchos::TimeMonitor monitor(*t);

  for (auto& elepairptr : assembly_contact_elepairs_)
  {
    // pre_evaluate the pair
    elepairptr->pre_evaluate();
  }
  // Evaluate the global mortar matrices.
  mortar_manager_->evaluate_global_dm(assembly_contact_elepairs_);

  // Add the global mortar matrices to the force vector and stiffness matrix.
  mortar_manager_->add_global_force_stiffness_contributions(ff, fb, cbb, cbf,
      Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(cff, true), cfb, beam_vel, fluid_vel);
}

FOUR_C_NAMESPACE_CLOSE
