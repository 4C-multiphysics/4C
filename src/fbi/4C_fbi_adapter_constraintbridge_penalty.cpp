// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fbi_adapter_constraintbridge_penalty.hpp"

#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_fbi_beam_to_fluid_assembly_manager_factory.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_params.hpp"
#include "4C_fbi_partitioned_penaltycoupling_assembly_manager_direct.hpp"
#include "4C_fbi_partitioned_penaltycoupling_assembly_manager_indirect.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"

FOUR_C_NAMESPACE_OPEN

void Adapter::FBIConstraintBridgePenalty::setup(const Core::LinAlg::Map* beam_map,
    const Core::LinAlg::Map* fluid_map, std::shared_ptr<Core::LinAlg::SparseOperator> fluidmatrix,
    bool fluidmeshtying)
{
  // Initialize all necessary vectors and matrices
  FBIConstraintBridge::setup(beam_map, fluid_map, fluidmatrix, fluidmeshtying);
  fs_ = std::make_shared<Core::LinAlg::FEVector<double>>(beam_map->get_epetra_block_map());
  ff_ = std::make_shared<Core::LinAlg::FEVector<double>>(fluid_map->get_epetra_block_map());
  cff_ = fluidmatrix;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FBIConstraintBridgePenalty::evaluate(
    std::shared_ptr<const Core::FE::Discretization> discretization1,
    std::shared_ptr<const Core::FE::Discretization> discretization2,
    std::shared_ptr<const Core::LinAlg::Vector<double>> fluid_vel,
    std::shared_ptr<const Core::LinAlg::Vector<double>> beam_vel)
{
  // Create assembly manager..
  std::shared_ptr<BeamInteraction::SubmodelEvaluator::PartitionedBeamInteractionAssemblyManager>
      assembly_manager =
          BeamInteraction::BeamToFluidAssemblyManagerFactory::create_assembly_manager(
              discretization1, discretization2, *(get_pairs()), get_params(), assemblystrategy_);
  // compute and assembly the coupling matrices and vectors
  assembly_manager->evaluate_force_stiff(
      *discretization1, *discretization2, ff_, fs_, cff_, css_, csf_, cfs_, fluid_vel, beam_vel);
  cff_->complete();

  // Unset the dirichlet flag in case we were doing a fluid solve
  unset_weak_dirichlet_flag();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FBIConstraintBridgePenalty::reset_bridge()
{
  fs_->put_scalar(0.0);
  cff_->reset();
  ff_->put_scalar(0.0);
  fluid_scaled_ = false;
  structure_scaled_ = false;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FBIConstraintBridgePenalty::set_weak_dirichlet_flag()
{
  beam_interaction_params_->set_weak_dirichlet_flag();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FBIConstraintBridgePenalty::unset_weak_dirichlet_flag()
{
  beam_interaction_params_->unset_weak_dirichlet_flag();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FBIConstraintBridgePenalty::scale_penalty_structure_contributions()
{
  if (!structure_scaled_)
  {
    if (fs_->scale(get_params()->get_penalty_parameter()))
      FOUR_C_THROW("Scaling of the penalty force was unsuccessful!\n");
    structure_scaled_ = true;
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Adapter::FBIConstraintBridgePenalty::scale_penalty_fluid_contributions()
{
  if (!fluid_scaled_)
  {
    if (cff_->scale(get_params()->get_penalty_parameter()) ||
        ff_->scale(get_params()->get_penalty_parameter()))
      FOUR_C_THROW("Scaling of the penalty force was unsuccessful!\n");
    fluid_scaled_ = true;
  }
}

FOUR_C_NAMESPACE_CLOSE
