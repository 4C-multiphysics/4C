// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_submodel_evaluator_beamcontact_assembly_manager_direct.hpp"

#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_beaminteraction_str_model_evaluator_datastate.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"

FOUR_C_NAMESPACE_OPEN


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
void BEAMINTERACTION::SUBMODELEVALUATOR::BeamContactAssemblyManagerDirect::evaluate_force_stiff(
    Teuchos::RCP<Core::FE::Discretization> discret,
    const Teuchos::RCP<const Solid::ModelEvaluator::BeamInteractionDataState>& data_state,
    Teuchos::RCP<Epetra_FEVector> fe_sysvec, Teuchos::RCP<Core::LinAlg::SparseMatrix> fe_sysmat)
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
      BEAMINTERACTION::Utils::assemble_centerline_dof_force_stiff_into_element_force_stiff(*discret,
          elegids, eleforce_centerlineDOFs, elestiff_centerlineDOFs, &eleforce, &elestiff);


      // Fixme
      eleforce[0].scale(-1.0);
      eleforce[1].scale(-1.0);

      // assemble the contributions into force vector class variable
      // f_crosslink_np_ptr_, i.e. in the DOFs of the connected nodes
      BEAMINTERACTION::Utils::fe_assemble_ele_force_stiff_into_system_vector_matrix(
          *discret, elegids, eleforce, elestiff, fe_sysvec, fe_sysmat);
    }

    // Each pair can also directly assembles terms into the global force vector and system matrix.
    elepairptr->evaluate_and_assemble(discret, fe_sysvec, fe_sysmat, data_state->get_dis_col_np());
  }
}

FOUR_C_NAMESPACE_CLOSE
