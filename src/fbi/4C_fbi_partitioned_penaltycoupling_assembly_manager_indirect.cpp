// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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
BeamInteraction::SubmodelEvaluator::PartitionedBeamInteractionAssemblyManagerIndirect::
    PartitionedBeamInteractionAssemblyManagerIndirect(
        std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>& assembly_contact_elepairs,
        std::shared_ptr<const Core::FE::Discretization>& discretization1,
        std::shared_ptr<const Core::FE::Discretization>& discretization2,
        std::shared_ptr<FBI::BeamToFluidMeshtyingParams> beam_contact_params_ptr)
    : PartitionedBeamInteractionAssemblyManager(assembly_contact_elepairs)
{
  // Create the mortar manager.
  mortar_manager_ = std::make_shared<BeamInteraction::BeamToFluidMortarManager>(discretization1,
      discretization2, beam_contact_params_ptr, discretization1->dof_row_map()->max_all_gid());

  // Setup the mortar manager.
  mortar_manager_->setup();
  mortar_manager_->set_local_maps(assembly_contact_elepairs_);
}


/**
 *
 */
void BeamInteraction::SubmodelEvaluator::PartitionedBeamInteractionAssemblyManagerIndirect::
    evaluate_force_stiff(const Core::FE::Discretization& discretization1,
        const Core::FE::Discretization& discretization2,
        std::shared_ptr<Core::LinAlg::FEVector<double>>& ff,
        std::shared_ptr<Core::LinAlg::FEVector<double>>& fb,
        std::shared_ptr<Core::LinAlg::SparseOperator> cff,
        std::shared_ptr<Core::LinAlg::SparseMatrix>& cbb,
        std::shared_ptr<Core::LinAlg::SparseMatrix>& cfb,
        std::shared_ptr<Core::LinAlg::SparseMatrix>& cbf,
        std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_vel,
        std::shared_ptr<const Core::LinAlg::Vector<double>> beam_vel)
{
  auto t = Teuchos::TimeMonitor::getNewTimer("FBI::PartitionedAssemblyManagerIndirect");
  Teuchos::TimeMonitor monitor(*t);

  for (auto& elepairptr : assembly_contact_elepairs_)
  {
    // pre_evaluate the pair
    elepairptr->pre_evaluate();
  }
  // Evaluate the global mortar matrices.
  mortar_manager_->evaluate_global_dm(assembly_contact_elepairs_);

  // Add the global mortar matrices to the force vector and stiffness matrix.
  mortar_manager_->add_global_force_stiffness_contributions(ff, *fb, cbb, cbf,
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(cff), cfb, beam_vel, fluid_vel);
}

FOUR_C_NAMESPACE_CLOSE
