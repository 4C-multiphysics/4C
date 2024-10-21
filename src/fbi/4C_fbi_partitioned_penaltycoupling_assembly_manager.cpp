#include "4C_fbi_partitioned_penaltycoupling_assembly_manager.hpp"

#include "4C_beaminteraction_contact_pair.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
BEAMINTERACTION::SUBMODELEVALUATOR::PartitionedBeamInteractionAssemblyManager::
    PartitionedBeamInteractionAssemblyManager(
        std::vector<Teuchos::RCP<BEAMINTERACTION::BeamContactPair>>& assembly_contact_elepairs)
    : assembly_contact_elepairs_(assembly_contact_elepairs)
{
}

FOUR_C_NAMESPACE_CLOSE
