// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_contact_submodel_evaluator_assembly_manager_indirect.hpp"

#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_beam_to_solid_mortar_manager.hpp"
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_beaminteraction_str_model_evaluator_datastate.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_vector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
void BeamInteraction::SubmodelEvaluator::BeamContactAssemblyManagerInDirect::evaluate_force_stiff(
    std::shared_ptr<Core::FE::Discretization> discret,
    const std::shared_ptr<const Solid::ModelEvaluator::BeamInteractionDataState>& data_state,
    std::shared_ptr<Core::LinAlg::FEVector<double>> fe_sysvec,
    std::shared_ptr<Core::LinAlg::SparseMatrix> fe_sysmat,
    const std::shared_ptr<BeamInteraction::BeamToSolidVolumeMeshtyingParams>
        beam_to_solid_volume_meshtying_params)
{
  if (mortar_manager_->have_lagrange_dofs())
  {
    mortar_manager_->evaluate_coupling_terms_lagrange(data_state, fe_sysmat, fe_sysvec);

    if (mortar_manager_->get_lagrange_formulation() ==
        BeamToSolid::BeamToSolidLagrangeFormulation::augmented)
    {
      const auto* dof_row_map = discret->dof_row_map();
      const auto* solid_map = &mortar_manager_->get_solid_dof_row_map();
      const auto* beam_map = &mortar_manager_->get_beam_dof_row_map();

      // Create matrix and vector for the penalty regularization contributions.
      std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_penalty = nullptr;
      if (fe_sysmat != nullptr)
      {
        sysmat_penalty = std::make_shared<Core::LinAlg::SparseMatrix>(
            *dof_row_map, 81, true, true, Core::LinAlg::SparseMatrix::FE_MATRIX);
      }
      std::shared_ptr<Core::LinAlg::FEVector<double>> sysvec_penalty = nullptr;
      if (fe_sysvec != nullptr)
      {
        sysvec_penalty = std::make_shared<Core::LinAlg::FEVector<double>>(*dof_row_map);
      }

      // Evaluate the penalty coupling contributions.
      mortar_manager_->evaluate_force_stiff_penalty_regularization(
          data_state, sysmat_penalty, sysvec_penalty);
      if (sysmat_penalty != nullptr) sysmat_penalty->complete();
      if (sysvec_penalty != nullptr) sysvec_penalty->complete();

      // Create the scaling vector holding the diagonal entries for the augmentation scaling
      // matrix.
      Core::LinAlg::Vector<double> augmentation_scaling_vector(*dof_row_map);
      augmentation_scaling_vector.put_scalar(1.0);

      auto add_scaling_values_to_vector = [&](const Core::LinAlg::Map& sub_map, double val)
      {
        for (int i = 0; i < sub_map.num_my_elements(); ++i)
        {
          int gid = sub_map.gid(i);
          int lid_dof_row_map = dof_row_map->lid(gid);
          if (lid_dof_row_map != -1)
          {
            augmentation_scaling_vector.replace_local_value(lid_dof_row_map, val);
          }
          else
          {
            FOUR_C_THROW(
                "Global ID {} from sub map is not found in dof row map. This should not happen.",
                gid);
          }
        }
      };

      if (beam_to_solid_volume_meshtying_params != nullptr)
      {
        add_scaling_values_to_vector(*solid_map,
            beam_to_solid_volume_meshtying_params->get_augmentation_scaling_parameter_solid());
        add_scaling_values_to_vector(*beam_map,
            beam_to_solid_volume_meshtying_params->get_augmentation_scaling_parameter_beam());
      }

      // Scale the penalty contributions and add them to the system matrix and vector.
      if (fe_sysmat != nullptr)
      {
        sysmat_penalty->left_scale(augmentation_scaling_vector);
        fe_sysmat->add(*sysmat_penalty, false, 1.0, 1.0);
      }
      if (fe_sysvec != nullptr)
      {
        fe_sysvec->multiply(1.0, augmentation_scaling_vector, *sysvec_penalty, 1.0);
      }
    }
  }
  else
  {
    mortar_manager_->evaluate_force_stiff_penalty_regularization(data_state, fe_sysmat, fe_sysvec);
  }
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
