// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fbi_beam_to_fluid_assembly_manager_factory.hpp"

#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_params.hpp"
#include "4C_fbi_fluid_assembly_strategy.hpp"
#include "4C_fbi_input.hpp"
#include "4C_fbi_partitioned_penaltycoupling_assembly_manager_direct.hpp"
#include "4C_fbi_partitioned_penaltycoupling_assembly_manager_indirect.hpp"
#include "4C_fem_discretization.hpp"

FOUR_C_NAMESPACE_OPEN
/**
 *
 */
std::shared_ptr<BeamInteraction::SUBMODELEVALUATOR::PartitionedBeamInteractionAssemblyManager>
BeamInteraction::BeamToFluidAssemblyManagerFactory::create_assembly_manager(
    std::shared_ptr<const Core::FE::Discretization> discretization1,
    std::shared_ptr<const Core::FE::Discretization> discretization2,
    std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>> interaction_pairs,
    const std::shared_ptr<FBI::BeamToFluidMeshtyingParams> params_ptr,
    std::shared_ptr<FBI::Utils::FBIAssemblyStrategy> assemblystrategy)
{
  // Get the meshtying discretization method.
  FBI::BeamToFluidDiscretization meshtying_discretization =
      params_ptr->get_contact_discretization();

  switch (meshtying_discretization)
  {
    case FBI::BeamToFluidDiscretization::mortar:
      return std::make_shared<
          BeamInteraction::SUBMODELEVALUATOR::PartitionedBeamInteractionAssemblyManagerIndirect>(

          interaction_pairs, discretization1, discretization2, params_ptr);
      break;
    case FBI::BeamToFluidDiscretization::gauss_point_to_segment:
      return std::make_shared<
          BeamInteraction::SUBMODELEVALUATOR::PartitionedBeamInteractionAssemblyManagerDirect>(

          interaction_pairs, assemblystrategy);
      break;
    default:
      FOUR_C_THROW("Beam To Fluid Meshtying discretization Type not supported!");
      return nullptr;
  }
}

FOUR_C_NAMESPACE_CLOSE
