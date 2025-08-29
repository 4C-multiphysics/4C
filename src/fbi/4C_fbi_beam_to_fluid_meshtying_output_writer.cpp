// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fbi_beam_to_fluid_meshtying_output_writer.hpp"

#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_base.hpp"
#include "4C_beaminteraction_beam_to_solid_visualization_output_writer_visualization.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_fbi_adapter_constraintbridge_penalty.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_output_params.hpp"
#include "4C_fbi_beam_to_fluid_meshtying_params.hpp"
#include "4C_fbi_constraintenforcer.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
BeamInteraction::BeamToFluidMeshtyingVtkOutputWriter::BeamToFluidMeshtyingVtkOutputWriter(
    const Core::IO::VisualizationParameters& visualization_params,
    std::shared_ptr<const FBI::BeamToFluidMeshtyingVtkOutputParams> output_params_ptr)
    : output_params_ptr_(output_params_ptr),
      output_writer_base_ptr_(nullptr),
      visualization_params_(visualization_params)
{
  // Initialize the writer base object and add the desired visualizations.
  output_writer_base_ptr_ =
      std::make_shared<BeamInteraction::BeamToSolidVisualizationOutputWriterBase>(

          "beam-to-fluid", visualization_params);

  // Depending on the selected input parameters, create the needed writers. All node / cell data
  // fields that should be output eventually have to be defined here. This helps to prevent issues
  // with ranks that do not contribute to a certain writer.
  {
    if (output_params_ptr_->get_nodal_force_output_flag())
    {
      std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization> visualization_writer =
          output_writer_base_ptr_->add_visualization_writer("nodal-forces");
      auto& visualization_data = visualization_writer->get_visualization_data();
      visualization_data.register_point_data<double>("velocity", 3);
      visualization_data.register_point_data<double>("displacement", 3);
      visualization_data.register_point_data<double>("force", 3);
    }

    if (output_params_ptr_->get_mortar_lambda_discret_output_flag())
    {
      FOUR_C_THROW("Mortar discretization not implemented for beam-to-fluid meshtying!\n");
    }

    if (output_params_ptr_->get_mortar_lambda_continuous_output_flag())
    {
      FOUR_C_THROW("Mortar discretization not implemented for beam-to-fluid meshtying!\n");
    }

    if (output_params_ptr_->get_integration_points_output_flag())
    {
      std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization> visualization_writer =
          output_writer_base_ptr_->add_visualization_writer("integration-points");
      auto& visualization_data = visualization_writer->get_visualization_data();
      visualization_data.register_point_data<double>("displacement", 3);
    }
  }
}

/**
 *
 */
void BeamInteraction::BeamToFluidMeshtyingVtkOutputWriter::write_output_runtime(
    Adapter::FBIConstraintenforcer& couplingenforcer, int i_step, double time) const
{
  auto [output_time, output_step] =
      Core::IO::get_time_and_time_step_index_for_output(visualization_params_, time, i_step);
  write_output_beam_to_fluid_mesh_tying(couplingenforcer, output_step, output_time);
}

/**
 *
 */
void BeamInteraction::BeamToFluidMeshtyingVtkOutputWriter::write_output_beam_to_fluid_mesh_tying(
    Adapter::FBIConstraintenforcer& couplingenforcer, int i_step, double time) const
{
  // Parameter list that will be passed to all contact pairs when they create their visualization.
  Teuchos::ParameterList visualization_params;
  visualization_params
      .set<std::shared_ptr<const BeamToSolidVolumeMeshtyingVisualizationOutputParams>>(
          "output_params_ptr", output_params_ptr_);


  // Add the nodal forces resulting from beam contact. The forces are split up into beam and solid
  // nodes.
  std::shared_ptr<BeamInteraction::BeamToSolidOutputWriterVisualization> visualization =
      output_writer_base_ptr_->get_visualization_writer("nodal-forces");
  if (visualization != nullptr)
  {
    // Add the reference geometry and displacement to the visualization.
    visualization->add_discretization_nodal_reference_position(
        couplingenforcer.get_structure()->get_discretization());
    visualization->add_discretization_nodal_data(
        "velocity", *couplingenforcer.get_structure()->velnp());
    visualization->add_discretization_nodal_data(
        "displacement", *couplingenforcer.get_structure()->dispnp());

    // Create maps with the GIDs of beam nodes
    std::vector<int> gid_beam_dof;
    std::vector<int> gid_node;
    for (int i_lid = 0;
        i_lid < couplingenforcer.get_structure()->get_discretization()->num_my_row_nodes(); i_lid++)
    {
      gid_node.clear();
      Core::Nodes::Node* current_node =
          couplingenforcer.get_structure()->get_discretization()->l_row_node(i_lid);
      couplingenforcer.get_structure()->get_discretization()->dof(current_node, gid_node);
      if (BeamInteraction::Utils::is_beam_node(*current_node))
        for (unsigned int dim = 0; dim < 3; ++dim) gid_beam_dof.push_back(gid_node[dim]);
    }
    Core::LinAlg::Map beam_dof_map(-1, gid_beam_dof.size(), gid_beam_dof.data(), 0,

        couplingenforcer.get_structure()->get_discretization()->get_comm());

    // Extract the forces and add them to the discretization.
    Core::LinAlg::Vector<double> force_beam(beam_dof_map, true);
    Core::LinAlg::export_to(*couplingenforcer.assemble_structure_coupling_residual(), force_beam);


    visualization->add_discretization_nodal_data("force", force_beam);
  }

  // Add the pair specific visualization by looping over the individual contact pairs.
  for (const auto& pair : *couplingenforcer.get_bridge()->get_pairs())
    pair->get_pair_visualization(output_writer_base_ptr_, visualization_params);


  // Write the data to disc. The data will be cleared in this method.
  output_writer_base_ptr_->write(i_step, time);
}


FOUR_C_NAMESPACE_CLOSE
