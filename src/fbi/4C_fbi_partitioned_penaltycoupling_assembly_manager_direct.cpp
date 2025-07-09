// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fbi_partitioned_penaltycoupling_assembly_manager_direct.hpp"

#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_fbi_calc_utils.hpp"
#include "4C_fbi_fluid_assembly_strategy.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_sparseoperator.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
BeamInteraction::SubmodelEvaluator::PartitionedBeamInteractionAssemblyManagerDirect::
    PartitionedBeamInteractionAssemblyManagerDirect(
        std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>> assembly_contact_elepairs,
        std::shared_ptr<FBI::Utils::FBIAssemblyStrategy> assemblystrategy)
    : PartitionedBeamInteractionAssemblyManager(assembly_contact_elepairs),
      assemblystrategy_(assemblystrategy)
{
}


/**
 *
 */
void BeamInteraction::SubmodelEvaluator::PartitionedBeamInteractionAssemblyManagerDirect::
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
  // resulting discrete element force vectors of the two interacting elements
  std::vector<Core::LinAlg::SerialDenseVector> eleforce(2);

  // resulting discrete force vectors (centerline DOFs only!) of the two
  // interacting elements
  std::vector<Core::LinAlg::SerialDenseVector> eleforce_centerlineDOFs(2);

  // linearizations
  std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> elestiff(
      2, std::vector<Core::LinAlg::SerialDenseMatrix>(2));

  // linearizations (centerline DOFs only!)
  std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> elestiff_centerlineDOFs(
      2, std::vector<Core::LinAlg::SerialDenseMatrix>(2));

  // element gids of interacting elements
  std::vector<int> elegids(2);

  // are non-zero stiffness values returned which need assembly?
  bool pair_is_active = false;


  for (auto& elepairptr : assembly_contact_elepairs_)
  {
    // pre_evaluate the pair
    elepairptr->pre_evaluate();
  }

  for (auto& elepairptr : assembly_contact_elepairs_)
  {
    // Evaluate the pair and check if there is active contact
    pair_is_active =
        elepairptr->evaluate(&(eleforce_centerlineDOFs[0]), &(eleforce_centerlineDOFs[1]),
            &(elestiff_centerlineDOFs[0][0]), &(elestiff_centerlineDOFs[0][1]),
            &(elestiff_centerlineDOFs[1][0]), &(elestiff_centerlineDOFs[1][1]));

    if (pair_is_active)
    {
      elegids[0] = elepairptr->element1()->id();
      elegids[1] = elepairptr->element2()->id();

      // assemble force vector and stiffness matrix affecting the centerline DoFs only
      // into element force vector and stiffness matrix ('all DoFs' format, as usual)
      FBI::Utils::assemble_centerline_dof_force_stiff_into_fbi_element_force_stiff(discretization1,
          discretization2, elegids, eleforce_centerlineDOFs, elestiff_centerlineDOFs, &eleforce,
          &elestiff);

      // assemble the contributions into force and stiffness matrices
      assemblystrategy_->assemble(discretization1, discretization2, elegids, eleforce, elestiff, fb,
          ff, cbb, cff, cbf, cfb);
    }
  }
  int err = fb->global_assemble();
  if (err) printf("Global assembly failed with error %i", err);
  err = ff->global_assemble();
  if (err) printf("Global assembly failed with error %i", err);
}

FOUR_C_NAMESPACE_CLOSE
