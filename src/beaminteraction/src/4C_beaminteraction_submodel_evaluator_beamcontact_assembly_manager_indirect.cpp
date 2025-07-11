// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_submodel_evaluator_beamcontact_assembly_manager_indirect.hpp"

#include "4C_beaminteraction_beam_to_solid_mortar_manager.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_beaminteraction_str_model_evaluator_datastate.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
void BeamInteraction::SubmodelEvaluator::BeamContactAssemblyManagerInDirect::evaluate_force_stiff(
    std::shared_ptr<Core::FE::Discretization> discret,
    const std::shared_ptr<const Solid::ModelEvaluator::BeamInteractionDataState>& data_state,
    std::shared_ptr<Core::LinAlg::FEVector<double>> fe_sysvec,
    std::shared_ptr<Core::LinAlg::SparseMatrix> fe_sysmat)
{
  mortar_manager_->evaluate_force_stiff_penalty_regularization(data_state, fe_sysmat, fe_sysvec);
}


double BeamInteraction::SubmodelEvaluator::BeamContactAssemblyManagerInDirect::get_energy(
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& disp) const
{
  const double global_mortar_energy = mortar_manager_->get_energy();

  // The value we returned here is summed up over all processors. Since we already have the global
  // energy here, we only return it on rank 0.
  if (Core::Communication::my_mpi_rank(disp->get_comm()) == 0)
    return global_mortar_energy;
  else
    return 0.0;
}

FOUR_C_NAMESPACE_CLOSE
