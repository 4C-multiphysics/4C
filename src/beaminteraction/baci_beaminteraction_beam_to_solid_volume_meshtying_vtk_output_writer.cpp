/*----------------------------------------------------------------------*/
/*! \file

\brief Object to handle beam to solid volume meshtying output creation.

\level 3

*/


#include "baci_beaminteraction_beam_to_solid_volume_meshtying_vtk_output_writer.H"

#include "baci_beaminteraction_beam_to_solid_mortar_manager.H"
#include "baci_beaminteraction_beam_to_solid_volume_meshtying_vtk_output_params.H"
#include "baci_beaminteraction_beam_to_solid_vtu_output_writer_base.H"
#include "baci_beaminteraction_beam_to_solid_vtu_output_writer_utils.H"
#include "baci_beaminteraction_beam_to_solid_vtu_output_writer_visualization.H"
#include "baci_beaminteraction_calc_utils.H"
#include "baci_beaminteraction_contact_pair.H"
#include "baci_beaminteraction_str_model_evaluator_datastate.H"
#include "baci_beaminteraction_submodel_evaluator_beamcontact.H"
#include "baci_beaminteraction_submodel_evaluator_beamcontact_assembly_manager_direct.H"
#include "baci_beaminteraction_submodel_evaluator_beamcontact_assembly_manager_indirect.H"
#include "baci_lib_discret.H"
#include "baci_linalg_utils_sparse_algebra_manipulation.H"
#include "baci_structure_new_timint_basedataglobalstate.H"

#include <Epetra_FEVector.h>

#include <unordered_set>


/**
 *
 */
BEAMINTERACTION::BeamToSolidVolumeMeshtyingVtkOutputWriter::
    BeamToSolidVolumeMeshtyingVtkOutputWriter()
    : isinit_(false),
      issetup_(false),
      output_params_ptr_(Teuchos::null),
      output_writer_base_ptr_(Teuchos::null)
{
}

/**
 *
 */
void BEAMINTERACTION::BeamToSolidVolumeMeshtyingVtkOutputWriter::Init()
{
  issetup_ = false;
  isinit_ = true;
}

/**
 *
 */
void BEAMINTERACTION::BeamToSolidVolumeMeshtyingVtkOutputWriter::Setup(
    Teuchos::RCP<const STR::TIMINT::ParamsRuntimeVtkOutput> vtk_params,
    Teuchos::RCP<const BEAMINTERACTION::BeamToSolidVolumeMeshtyingVtkOutputParams>
        output_params_ptr,
    double restart_time)
{
  CheckInit();

  // Set beam to solid volume mesh tying output parameters.
  output_params_ptr_ = output_params_ptr;

  // Initialize the writer base object and add the desired visualizations.
  output_writer_base_ptr_ = Teuchos::rcp<BEAMINTERACTION::BeamToSolidVtuOutputWriterBase>(
      new BEAMINTERACTION::BeamToSolidVtuOutputWriterBase(
          "beam-to-solid-volume", vtk_params, restart_time));

  // Whether or not to write unique cell and node IDs.
  const bool write_unique_ids = output_params_ptr_->GetWriteUniqueIDsFlag();

  // Depending on the selected input parameters, create the needed writers. All node / cell data
  // fields that should be output eventually have to be defined here. This helps to prevent issues
  // with ranks that do not contribute to a certain writer.
  {
    if (output_params_ptr_->GetNodalForceOutputFlag())
    {
      Teuchos::RCP<BEAMINTERACTION::BeamToSolidVtuOutputWriterVisualization> visualization_writer =
          output_writer_base_ptr_->AddVisualizationWriter("nodal-forces", "btsvc-nodal-forces");
      auto& visualization_data = visualization_writer->GetVisualizationDataMutable();
      visualization_data.RegisterPointData<double>("displacement", 3);
      visualization_data.RegisterPointData<double>("force_beam", 3);
      visualization_data.RegisterPointData<double>("force_solid", 3);
      if (write_unique_ids) visualization_data.RegisterPointData<double>("uid_0_node_id", 1);
    }

    if (output_params_ptr_->GetMortarLambdaDiscretOutputFlag())
    {
      Teuchos::RCP<BEAMINTERACTION::BeamToSolidVtuOutputWriterVisualization> visualization_writer =
          output_writer_base_ptr_->AddVisualizationWriter("mortar", "btsvc-mortar");
      auto& visualization_data = visualization_writer->GetVisualizationDataMutable();
      visualization_data.RegisterPointData<double>("displacement", 3);
      visualization_data.RegisterPointData<double>("lambda", 3);
      if (write_unique_ids)
      {
        visualization_data.RegisterPointData<double>("uid_0_pair_beam_id", 1);
        visualization_data.RegisterPointData<double>("uid_1_pair_solid_id", 1);
      }
    }

    if (output_params_ptr_->GetMortarLambdaContinuousOutputFlag())
    {
      Teuchos::RCP<BEAMINTERACTION::BeamToSolidVtuOutputWriterVisualization> visualization_writer =
          output_writer_base_ptr_->AddVisualizationWriter(
              "mortar-continuous", "btsvc-mortar-continuous");
      auto& visualization_data = visualization_writer->GetVisualizationDataMutable();
      visualization_data.RegisterPointData<double>("displacement", 3);
      visualization_data.RegisterPointData<double>("lambda", 3);
      if (write_unique_ids)
      {
        visualization_data.RegisterPointData<double>("uid_0_pair_beam_id", 1);
        visualization_data.RegisterPointData<double>("uid_1_pair_solid_id", 1);
        visualization_data.RegisterCellData<double>("uid_0_pair_beam_id", 1);
        visualization_data.RegisterCellData<double>("uid_1_pair_solid_id", 1);
      }
    }

    if (output_params_ptr_->GetIntegrationPointsOutputFlag())
    {
      Teuchos::RCP<BEAMINTERACTION::BeamToSolidVtuOutputWriterVisualization> visualization_writer =
          output_writer_base_ptr_->AddVisualizationWriter(
              "integration-points", "btsvc-integration-points");
      auto& visualization_data = visualization_writer->GetVisualizationDataMutable();
      visualization_data.RegisterPointData<double>("displacement", 3);
      visualization_data.RegisterPointData<double>("force", 3);
      if (write_unique_ids)
      {
        visualization_data.RegisterPointData<double>("uid_0_pair_beam_id", 1);
        visualization_data.RegisterPointData<double>("uid_1_pair_solid_id", 1);
      }
    }

    if (output_params_ptr_->GetSegmentationOutputFlag())
    {
      Teuchos::RCP<BEAMINTERACTION::BeamToSolidVtuOutputWriterVisualization> visualization_writer =
          output_writer_base_ptr_->AddVisualizationWriter("segmentation", "btsvc-segmentation");
      auto& visualization_data = visualization_writer->GetVisualizationDataMutable();
      visualization_data.RegisterPointData<double>("displacement", 3);
      if (write_unique_ids)
      {
        visualization_data.RegisterPointData<double>("uid_0_pair_beam_id", 1);
        visualization_data.RegisterPointData<double>("uid_1_pair_solid_id", 1);
      }
    }
  }

  issetup_ = true;
}

/**
 *
 */
void BEAMINTERACTION::BeamToSolidVolumeMeshtyingVtkOutputWriter::WriteOutputRuntime(
    const BEAMINTERACTION::SUBMODELEVALUATOR::BeamContact* beam_contact) const
{
  CheckInitSetup();

  // Get the time step and time for the output file. If output is desired at every iteration, the
  // values are padded. The runtime output is written when the time step is already set to the
  // next step.
  auto [output_time, output_step] =
      IO::GetTimeAndTimeStepIndexForOutput(output_params_ptr_->GetVisualizationParameters(),
          beam_contact->GState().GetTimeN(), beam_contact->GState().GetStepN());
  WriteOutputBeamToSolidVolumeMeshTying(beam_contact, output_step, output_time);
}

/**
 *
 */
void BEAMINTERACTION::BeamToSolidVolumeMeshtyingVtkOutputWriter::WriteOutputRuntimeIteration(
    const BEAMINTERACTION::SUBMODELEVALUATOR::BeamContact* beam_contact, int i_iteration) const
{
  CheckInitSetup();

  if (output_params_ptr_->GetOutputEveryIteration())
  {
    auto [output_time, output_step] =
        IO::GetTimeAndTimeStepIndexForOutput(output_params_ptr_->GetVisualizationParameters(),
            beam_contact->GState().GetTimeN(), beam_contact->GState().GetStepN(), i_iteration);
    WriteOutputBeamToSolidVolumeMeshTying(beam_contact, output_step, output_time);
  }
}

/**
 *
 */
void BEAMINTERACTION::BeamToSolidVolumeMeshtyingVtkOutputWriter::
    WriteOutputBeamToSolidVolumeMeshTying(
        const BEAMINTERACTION::SUBMODELEVALUATOR::BeamContact* beam_contact, int i_step,
        double time) const
{
  // Parameter list that will be passed to all contact pairs when they create their visualization.
  Teuchos::ParameterList visualization_params;
  visualization_params.set<Teuchos::RCP<const BeamToSolidVolumeMeshtyingVtkOutputParams>>(
      "btsvc-output_params_ptr", output_params_ptr_);


  // Add the nodal forces resulting from beam contact. The forces are split up into beam and solid
  // nodes.
  Teuchos::RCP<BEAMINTERACTION::BeamToSolidVtuOutputWriterVisualization> visualization =
      output_writer_base_ptr_->GetVisualizationWriter("btsvc-nodal-forces");
  if (visualization != Teuchos::null)
    AddBeamInteractionNodalForces(visualization, beam_contact->DiscretPtr(),
        beam_contact->BeamInteractionDataState().GetDisNp(),
        beam_contact->BeamInteractionDataState().GetForceNp(),
        output_params_ptr_->GetWriteUniqueIDsFlag());


  // Loop over the assembly managers and add the visualization for the pairs contained in the
  // assembly managers.
  for (auto& assembly_manager : beam_contact->GetAssemblyManagers())
  {
    // Add pair specific output for direct assembly managers.
    auto direct_assembly_manager = Teuchos::rcp_dynamic_cast<
        BEAMINTERACTION::SUBMODELEVALUATOR::BeamContactAssemblyManagerDirect>(assembly_manager);
    if (not(direct_assembly_manager == Teuchos::null))
    {
      for (const auto& pair : direct_assembly_manager->GetContactPairs())
        pair->GetPairVisualization(output_writer_base_ptr_, visualization_params);
    }

    // Add pair specific output for indirect assembly managers.
    auto indirect_assembly_manager = Teuchos::rcp_dynamic_cast<
        BEAMINTERACTION::SUBMODELEVALUATOR::BeamContactAssemblyManagerInDirect>(assembly_manager);
    if (not(indirect_assembly_manager == Teuchos::null))
    {
      // Get the global vector with the Lagrange Multiplier values and add it to the parameter
      // list that will be passed to the pairs.
      Teuchos::RCP<Epetra_Vector> lambda =
          indirect_assembly_manager->GetMortarManager()->GetGlobalLambdaCol();
      visualization_params.set<Teuchos::RCP<Epetra_Vector>>("lambda", lambda);

      // The pairs will need the mortar manager to extract their Lambda DOFs.
      visualization_params.set<Teuchos::RCP<const BEAMINTERACTION::BeamToSolidMortarManager>>(
          "mortar_manager", indirect_assembly_manager->GetMortarManager());

      // This map is used to ensure, that each discrete Lagrange multiplier is only written once
      // per beam element.
      Teuchos::RCP<std::unordered_set<int>> beam_tracker =
          Teuchos::rcp(new std::unordered_set<int>());
      visualization_params.set<Teuchos::RCP<std::unordered_set<int>>>("beam_tracker", beam_tracker);

      // Add the pair specific output.
      for (const auto& pair : indirect_assembly_manager->GetMortarManager()->GetContactPairs())
        pair->GetPairVisualization(output_writer_base_ptr_, visualization_params);

      // Reset assembly manager specific values in the parameter list passed to the individual
      // pairs.
      visualization_params.remove("lambda");
      visualization_params.remove("mortar_manager");
      visualization_params.remove("beam_tracker");
    }
  }

  // Write the data to disc. The data will be cleared in this method.
  output_writer_base_ptr_->Write(i_step, time);
}


/**
 * \brief Checks the init and setup status.
 */
void BEAMINTERACTION::BeamToSolidVolumeMeshtyingVtkOutputWriter::CheckInitSetup() const
{
  if (!isinit_ or !issetup_) dserror("Call Init() and Setup() first!");
}

/**
 * \brief Checks the init status.
 */
void BEAMINTERACTION::BeamToSolidVolumeMeshtyingVtkOutputWriter::CheckInit() const
{
  if (!isinit_) dserror("Init() has not been called, yet!");
}
