// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_surface_visualization_output_writer_contact.hpp"

#include "4C_beaminteraction_beam_to_solid_conditions.hpp"
#include "4C_beaminteraction_beam_to_solid_mortar_manager.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_visualization_output_params.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_base.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_utils.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_visualization.hpp"
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_beaminteraction_str_model_evaluator_datastate.hpp"
#include "4C_beaminteraction_submodel_evaluator_beamcontact.hpp"
#include "4C_beaminteraction_submodel_evaluator_beamcontact_assembly_manager_direct.hpp"
#include "4C_beaminteraction_submodel_evaluator_beamcontact_assembly_manager_indirect.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_geometry_pair_line_to_surface_evaluation_data.hpp"
#include "4C_io_visualization_parameters.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"

#include <unordered_set>
#include <utility>

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
BeamInteraction::BeamToSolidSurfaceVisualizationOutputWriterContact::
    BeamToSolidSurfaceVisualizationOutputWriterContact(
        Core::IO::VisualizationParameters visualization_params,
        std::shared_ptr<const BeamInteraction::BeamToSolidSurfaceVisualizationOutputParams>
            output_params_ptr)
    : output_params_ptr_(output_params_ptr),
      output_writer_base_ptr_(nullptr),
      visualization_params_(std::move(visualization_params))
{
  // Initialize the writer base object and add the desired visualizations.
  output_writer_base_ptr_ =
      std::make_shared<BeamInteraction::BeamToSolidVisualizationOutputWriterBase>(

          "beam-to-solid-surface-contact", visualization_params_);

  // Whether or not to write unique cell and node IDs.
  const bool write_unique_ids = output_params_ptr_->get_write_unique_ids_flag();

  // Depending on the selected input parameters, create the needed writers. All node / cell data
  // fields that should be output eventually have to be defined here. This helps to prevent issues
  // with ranks that do not contribute to a certain writer.
  {
    if (output_params_ptr_->get_nodal_force_output_flag())
    {
      std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization> visualization_writer =
          output_writer_base_ptr_->add_visualization_writer(
              "nodal-forces", "btss-contact-nodal-forces");
      auto& visualization_data = visualization_writer->get_visualization_data();
      visualization_data.register_point_data<double>("displacement", 3);
      visualization_data.register_point_data<double>("force_beam", 3);
      visualization_data.register_point_data<double>("force_solid", 3);
      if (write_unique_ids) visualization_data.register_point_data<int>("uid_0_node_id", 1);
    }

    if (output_params_ptr_->get_averaged_normals_output_flag())
    {
      std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization> visualization_writer =
          output_writer_base_ptr_->add_visualization_writer(
              "averaged-normals", "btss-contact-averaged-normals");
      auto& visualization_data = visualization_writer->get_visualization_data();
      visualization_data.register_point_data<double>("displacement", 3);
      visualization_data.register_point_data<double>("normal_averaged", 3);
      visualization_data.register_point_data<double>("normal_element", 3);
      visualization_data.register_point_data<int>("coupling_id", 1);
      if (write_unique_ids) visualization_data.register_point_data<int>("uid_0_face_id", 1);
    }

    if (output_params_ptr_->get_mortar_lambda_continuous_output_flag())
    {
      std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization> visualization_writer =
          output_writer_base_ptr_->add_visualization_writer(
              "mortar-continuous", "btss-contact-mortar-continuous");
      auto& visualization_data = visualization_writer->get_visualization_data();
      visualization_data.register_point_data<double>("displacement", 3);
      visualization_data.register_point_data<double>("lambda", 1);
      visualization_data.register_point_data<double>("lambda_times_normal", 3);
      visualization_data.register_point_data<double>("surface_normal", 3);
      visualization_data.register_point_data<double>("gap", 1);
      if (write_unique_ids)
      {
        visualization_data.register_point_data<int>("uid_0_pair_beam_id", 1);
        visualization_data.register_point_data<int>("uid_1_pair_solid_id", 1);
        visualization_data.register_cell_data<int>("uid_0_pair_beam_id", 1);
        visualization_data.register_cell_data<int>("uid_1_pair_solid_id", 1);
      }
    }

    // We have the same fields for integration point and segmentation output, so add them in this
    // lambda
    auto register_visualization_data = [&](const std::string& name, const std::string& type)
    {
      std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization> visualization_writer =
          output_writer_base_ptr_->add_visualization_writer(name, type);
      auto& visualization_data = visualization_writer->get_visualization_data();
      visualization_data.register_point_data<double>("displacement", 3);
      visualization_data.register_point_data<double>("surface_normal", 3);
      visualization_data.register_point_data<double>("gap", 1);
      visualization_data.register_point_data<double>("force", 3);
      if (write_unique_ids)
      {
        visualization_data.register_point_data<int>("uid_0_pair_beam_id", 1);
        visualization_data.register_point_data<int>("uid_1_pair_solid_id", 1);
      }
    };

    if (output_params_ptr_->get_integration_points_output_flag())
    {
      register_visualization_data("integration-points", "btss-contact-integration-points");
    }

    if (output_params_ptr_->get_segmentation_output_flag())
    {
      register_visualization_data("segmentation", "btss-contact-segmentation");
    }
  }
}

/**
 *
 */
void BeamInteraction::BeamToSolidSurfaceVisualizationOutputWriterContact::write_output_runtime(
    const BeamInteraction::SubmodelEvaluator::BeamContact* beam_contact) const
{
  // Get the time step and time for the output file. If output is desired at every iteration, the
  // values are padded. The runtime output is written when the time step is already set to the next
  // step.
  auto [output_time, output_step] =
      Core::IO::get_time_and_time_step_index_for_output(visualization_params_,
          beam_contact->g_state().get_time_n(), beam_contact->g_state().get_step_n());
  write_output_beam_to_solid_surface(beam_contact, output_step, output_time);
}

/**
 *
 */
void BeamInteraction::BeamToSolidSurfaceVisualizationOutputWriterContact::
    write_output_runtime_iteration(
        const BeamInteraction::SubmodelEvaluator::BeamContact* beam_contact, int i_iteration) const
{
  if (output_params_ptr_->get_output_every_iteration())
  {
    auto [output_time, output_step] = Core::IO::get_time_and_time_step_index_for_output(
        visualization_params_, beam_contact->g_state().get_time_n(),
        beam_contact->g_state().get_step_n(), i_iteration);
    write_output_beam_to_solid_surface(beam_contact, output_step, output_time);
  }
}

/**
 *
 */
void BeamInteraction::BeamToSolidSurfaceVisualizationOutputWriterContact::
    write_output_beam_to_solid_surface(
        const BeamInteraction::SubmodelEvaluator::BeamContact* beam_contact, int i_step,
        double time) const
{
  // Parameter list that will be passed to all contact pairs when they create their visualization.
  Teuchos::ParameterList visualization_params;
  visualization_params.set<std::shared_ptr<const BeamToSolidSurfaceVisualizationOutputParams>>(
      "btss-output_params_ptr", output_params_ptr_);


  // Add the averaged nodal normal output.
  std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization>
      visualization_averaged_normals =
          output_writer_base_ptr_->get_visualization_writer("btss-contact-averaged-normals");
  if (visualization_averaged_normals != nullptr)
  {
    const std::vector<std::shared_ptr<BeamInteractionConditionBase>>& surface_condition_vector =
        beam_contact->get_conditions()->get_condition_map().at(
            Inpar::BeamInteraction::BeamInteractionConditions::beam_to_solid_surface_contact);
    for (const auto& condition : surface_condition_vector)
    {
      // Get the line-to-surface evaluation data for the current condition.
      auto beam_to_surface_condition =
          std::dynamic_pointer_cast<const BeamToSolidConditionSurface>(condition);
      std::shared_ptr<const GeometryPair::LineToSurfaceEvaluationData> surface_evaluation_data =
          std::dynamic_pointer_cast<const GeometryPair::LineToSurfaceEvaluationData>(
              beam_to_surface_condition->get_geometry_evaluation_data());

      // Get the coupling ID for the current condition.
      const int coupling_id =
          beam_to_surface_condition->get_other_condition().parameters().get<int>("COUPLING_ID");

      // Create the output for the averaged normal field.
      add_averaged_nodal_normals(*visualization_averaged_normals,
          surface_evaluation_data->get_face_elements(), coupling_id,
          output_params_ptr_->get_write_unique_ids_flag());
    }
  }


  // Add the nodal forces resulting from beam contact. The forces are split up into beam and
  // solid nodes.
  std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization> nodal_force_visualization =
      output_writer_base_ptr_->get_visualization_writer("btss-contact-nodal-forces");
  if (nodal_force_visualization != nullptr)
    add_beam_interaction_nodal_forces(nodal_force_visualization, beam_contact->discret_ptr(),
        beam_contact->beam_interaction_data_state().get_dis_np()->as_multi_vector(),
        Core::LinAlg::MultiVector<double>(beam_contact->beam_interaction_data_state()
                .get_force_np()
                ->get_ref_of_epetra_fevector()),
        output_params_ptr_->get_write_unique_ids_flag());

  // Loop over the assembly managers and add the visualization for the pairs contained in the
  // assembly managers.
  for (auto& assembly_manager : beam_contact->get_assembly_managers())
  {
    // Add pair specific output for direct assembly managers.
    auto direct_assembly_manager = std::dynamic_pointer_cast<
        BeamInteraction::SubmodelEvaluator::BeamContactAssemblyManagerDirect>(assembly_manager);
    if (not(direct_assembly_manager == nullptr))
    {
      for (const auto& pair : direct_assembly_manager->get_contact_pairs())
        pair->get_pair_visualization(output_writer_base_ptr_, visualization_params);
    }

    // Add pair specific output for indirect assembly managers.
    auto indirect_assembly_manager = std::dynamic_pointer_cast<
        BeamInteraction::SubmodelEvaluator::BeamContactAssemblyManagerInDirect>(assembly_manager);
    if (not(indirect_assembly_manager == nullptr))
    {
      // Get the global vector with the Lagrange Multiplier values and add it to the parameter list
      // that will be passed to the pairs.
      std::shared_ptr<Core::LinAlg::Vector<double>> lambda =
          indirect_assembly_manager->get_mortar_manager()->get_global_lambda_col();
      visualization_params.set<std::shared_ptr<Core::LinAlg::Vector<double>>>("lambda", lambda);

      // The pairs will need the mortar manager to extract their Lambda DOFs.
      visualization_params.set<std::shared_ptr<const BeamInteraction::BeamToSolidMortarManager>>(
          "mortar_manager", indirect_assembly_manager->get_mortar_manager());

      // This map is used to ensure, that each discrete Lagrange multiplier is only written once per
      // beam element.
      std::shared_ptr<std::unordered_set<int>> beam_tracker =
          std::make_shared<std::unordered_set<int>>();
      visualization_params.set<std::shared_ptr<std::unordered_set<int>>>(
          "beam_tracker", beam_tracker);

      // Add the pair specific output.
      for (const auto& pair : indirect_assembly_manager->get_mortar_manager()->get_contact_pairs())
        pair->get_pair_visualization(output_writer_base_ptr_, visualization_params);

      // Reset assembly manager specific values in the parameter list passed to the individual
      // pairs.
      visualization_params.remove("lambda");
      visualization_params.remove("mortar_manager");
      visualization_params.remove("beam_tracker");
    }
  }


  // Write the data to disc. The data will be cleared in this method.
  output_writer_base_ptr_->write(i_step, time);
}

FOUR_C_NAMESPACE_CLOSE
