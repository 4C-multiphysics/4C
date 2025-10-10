// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_submodel_evaluator_beamcontact.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beamcontact_input.hpp"
#include "4C_beaminteraction_beam_to_solid_edge_contact_params.hpp"
#include "4C_beaminteraction_beam_to_solid_mortar_manager.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_contact_params.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_meshtying_params.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_visualization_output_params.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_visualization_output_writer.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_visualization_output_writer_contact.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_params.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_visualization_output_params.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_visualization_output_writer.hpp"
#include "4C_beaminteraction_calc_utils.hpp"
#include "4C_beaminteraction_conditions.hpp"
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_beaminteraction_contact_runtime_visualization_output_params.hpp"
#include "4C_beaminteraction_data.hpp"
#include "4C_beaminteraction_str_model_evaluator_datastate.hpp"
#include "4C_beaminteraction_submodel_evaluator_beamcontact_assembly_manager_direct.hpp"
#include "4C_beaminteraction_submodel_evaluator_beamcontact_assembly_manager_indirect.hpp"
#include "4C_binstrategy.hpp"
#include "4C_comm_mpi_utils.hpp"
#include "4C_geometric_search_bvh.hpp"
#include "4C_geometric_search_params.hpp"
#include "4C_geometric_search_visualization.hpp"
#include "4C_geometry_pair_input.hpp"
#include "4C_geometry_pair_line_to_3D_evaluation_data.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_beam_to_solid.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_io_visualization_manager.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_rigidsphere.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"
#include "4C_structure_new_timint_basedataio.hpp"
#include "4C_utils_exceptions.hpp"

#include <NOX_Solver_Generic.H>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
BeamInteraction::SubmodelEvaluator::BeamContact::BeamContact()
{
  // clear stl stuff
  nearby_elements_map_.clear();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::setup()
{
  check_init();

  // build a new data container to manage beam interaction parameters
  beam_interaction_params_ptr_ = std::make_shared<BeamInteraction::BeamInteractionParams>();
  beam_interaction_params_ptr_->init();
  beam_interaction_params_ptr_->setup();

  // build a new data container to manage geometric search parameters
  geometric_search_params_ptr_ = std::make_shared<Core::GeometricSearch::GeometricSearchParams>(
      Global::Problem::instance()->geometric_search_params(),
      Global::Problem::instance()->io_params());
  if (beam_interaction_params_ptr_->get_search_strategy() ==
          Inpar::BeamInteraction::SearchStrategy::bounding_volume_hierarchy &&
      geometric_search_params_ptr_->get_write_visualization_flag())
  {
    geometric_search_visualization_ptr_ =
        std::make_shared<Core::GeometricSearch::GeometricSearchVisualization>(
            Core::IO::visualization_parameters_factory(
                Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                *Global::Problem::instance()->output_control_file(), g_state().get_time_n()),
            discret_ptr()->get_comm(), "beam-interaction-geometric-search");
  }

  // build a new data container to manage beam contact parameters
  beam_contact_params_ptr_ = std::make_shared<BeamInteraction::BeamContactParams>();

  // build runtime visualization writer if desired
  if (Global::Problem::instance()
          ->beam_contact_params()
          .sublist("RUNTIME VTK OUTPUT")
          .get<bool>("VTK_OUTPUT_BEAM_CONTACT"))
  {
    beam_contact_params_ptr_->build_beam_contact_runtime_output_params(g_state().get_time_n());

    init_output_runtime_beam_contact();
  }


  contactelementtypes_.clear();

  if (Teuchos::getIntegralValue<Inpar::BeamInteraction::Strategy>(
          Global::Problem::instance()->beam_interaction_params().sublist("BEAM TO BEAM CONTACT"),
          "STRATEGY") != Inpar::BeamInteraction::bstr_none)
  {
    contactelementtypes_.push_back(Core::Binstrategy::Utils::BinContentType::Beam);

    beam_contact_params_ptr_->build_beam_to_beam_contact_params();
  }

  // conditions for beam penalty point coupling
  std::vector<const Core::Conditions::Condition*> beampenaltycouplingconditions;
  discret().get_condition("PenaltyPointCouplingCondition", beampenaltycouplingconditions);
  if (beampenaltycouplingconditions.size() > 0)
  {
    contactelementtypes_.push_back(Core::Binstrategy::Utils::BinContentType::Beam);
  }

  if (Teuchos::getIntegralValue<Inpar::BeamInteraction::Strategy>(
          Global::Problem::instance()->beam_interaction_params().sublist("BEAM TO SPHERE CONTACT"),
          "STRATEGY") != Inpar::BeamInteraction::bstr_none)
  {
    contactelementtypes_.push_back(Core::Binstrategy::Utils::BinContentType::RigidSphere);

    beam_contact_params_ptr_->build_beam_to_sphere_contact_params();
  }

  // Check if beam-to-solid volume mesh tying is present.
  const Teuchos::ParameterList& beam_to_solid_volume_parameters =
      Global::Problem::instance()->beam_interaction_params().sublist(
          "BEAM TO SOLID VOLUME MESHTYING");
  if (Teuchos::getIntegralValue<Inpar::BeamToSolid::BeamToSolidContactDiscretization>(
          beam_to_solid_volume_parameters, "CONTACT_DISCRETIZATION") !=
      Inpar::BeamToSolid::BeamToSolidContactDiscretization::none)
  {
    contactelementtypes_.push_back(Core::Binstrategy::Utils::BinContentType::Solid);

    beam_contact_params_ptr_->build_beam_to_solid_volume_meshtying_params();

    // Build the beam to solid volume meshtying output writer if desired.
    if (beam_contact_params_ptr_->beam_to_solid_volume_meshtying_params()
            ->get_visualization_output_params_ptr()
            ->get_output_flag())
    {
      beam_to_solid_volume_meshtying_visualization_output_writer_ptr_ =
          std::make_shared<BeamInteraction::BeamToSolidVolumeMeshtyingVisualizationOutputWriter>(

              Core::IO::visualization_parameters_factory(
                  Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                  *Global::Problem::instance()->output_control_file(), g_state().get_time_n()),
              beam_contact_params_ptr_->beam_to_solid_volume_meshtying_params()
                  ->get_visualization_output_params_ptr());
    }
  }

  // Check if beam-to-solid surface mesh tying is present.
  const Teuchos::ParameterList& beam_to_solid_surface_parameters =
      Global::Problem::instance()->beam_interaction_params().sublist(
          "BEAM TO SOLID SURFACE MESHTYING");
  if (Teuchos::getIntegralValue<Inpar::BeamToSolid::BeamToSolidContactDiscretization>(
          beam_to_solid_surface_parameters, "CONTACT_DISCRETIZATION") !=
      Inpar::BeamToSolid::BeamToSolidContactDiscretization::none)
  {
    contactelementtypes_.push_back(Core::Binstrategy::Utils::BinContentType::Solid);

    beam_contact_params_ptr_->build_beam_to_solid_surface_meshtying_params();

    // Build the beam to solid surface output writer if desired.
    if (beam_contact_params_ptr_->beam_to_solid_surface_meshtying_params()
            ->get_visualization_output_params_ptr()
            ->get_output_flag())
    {
      beam_to_solid_surface_visualization_output_writer_ptr_ =
          std::make_shared<BeamInteraction::BeamToSolidSurfaceVisualizationOutputWriter>(

              Core::IO::visualization_parameters_factory(
                  Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                  *Global::Problem::instance()->output_control_file(), g_state().get_time_n()),
              beam_contact_params_ptr_->beam_to_solid_surface_meshtying_params()
                  ->get_visualization_output_params_ptr());
    }
  }

  // Check if beam-to-solid surface contact is present.
  const Teuchos::ParameterList& beam_to_solid_surface_contact_parameters =
      Global::Problem::instance()->beam_interaction_params().sublist(
          "BEAM TO SOLID SURFACE CONTACT");
  if (Teuchos::getIntegralValue<Inpar::BeamToSolid::BeamToSolidContactDiscretization>(
          beam_to_solid_surface_contact_parameters, "CONTACT_DISCRETIZATION") !=
      Inpar::BeamToSolid::BeamToSolidContactDiscretization::none)
  {
    contactelementtypes_.push_back(Core::Binstrategy::Utils::BinContentType::Solid);

    beam_contact_params_ptr_->build_beam_to_solid_surface_contact_params();

    // Build the beam to solid surface contact output writer if desired.
    if (beam_contact_params_ptr_->beam_to_solid_surface_contact_params()
            ->get_visualization_output_params_ptr()
            ->get_output_flag())
    {
      beam_to_solid_surface_visualization_output_writer_contact_ptr_ =
          std::make_shared<BeamInteraction::BeamToSolidSurfaceVisualizationOutputWriterContact>(

              Core::IO::visualization_parameters_factory(
                  Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                  *Global::Problem::instance()->output_control_file(), g_state().get_time_n()),
              beam_contact_params_ptr_->beam_to_solid_surface_contact_params()
                  ->get_visualization_output_params_ptr());
    }
  }


  // Check if beam-to-solid edge contact is present.
  if (Global::Problem::instance()->parameters().isParameter(
          "BEAM INTERACTION/BEAM TO SOLID EDGE CONTACT"))
  {
    contactelementtypes_.push_back(Core::Binstrategy::Utils::BinContentType::Solid);
    beam_contact_params_ptr_->build_beam_to_solid_edge_contact_params();
  }

  // Build the container to manage beam-to-solid conditions and get all coupling conditions.
  beam_interaction_conditions_ptr_ = std::make_shared<BeamInteraction::BeamInteractionConditions>();
  beam_interaction_conditions_ptr_->set_beam_interaction_conditions(
      *discret_ptr(), *beam_contact_params_ptr_);

  // set flag
  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const BeamInteraction::SubmodelEvaluator::BeamContactAssemblyManagerInDirect>
BeamInteraction::SubmodelEvaluator::BeamContact::get_lagrange_multiplier_assembly_manager() const
{
  if (!std::dynamic_pointer_cast<BeamContactAssemblyManagerInDirect>(assembly_managers_[0]))
    FOUR_C_THROW("Expected a BeamContact Assembly Manager");
  if (assembly_managers_.size() != 1) FOUR_C_THROW("Only working for single assembly manager");
  return std::dynamic_pointer_cast<BeamContactAssemblyManagerInDirect>(assembly_managers_[0]);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::post_setup()
{
  check_init_setup();

  // Todo really needed here? maybe find better place
  // ensure that contact is evaluated correctly at beginning of first time step (initial overlap)
  nearby_elements_map_.clear();
  find_and_store_neighboring_elements();
  create_beam_contact_element_pairs();

  // The following section is specific to lagrange multiplier constraint enforcement
  if (beam_interaction_data_state().get_lambda() == nullptr &&
      beam_contact_params_ptr()->beam_to_solid_volume_meshtying_params() &&
      beam_contact_params_ptr()
              ->beam_to_solid_volume_meshtying_params()
              ->get_constraint_enforcement() ==
          Inpar::BeamToSolid::BeamToSolidConstraintEnforcement::lagrange)
  {
    auto lambda_dof_row_map =
        get_lagrange_multiplier_assembly_manager()->get_mortar_manager()->get_lambda_dof_row_map();

    beam_interaction_data_state().get_lambda() =
        std::make_shared<Core::LinAlg::FEVector<double>>(*lambda_dof_row_map);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::init_submodel_dependencies(
    std::shared_ptr<Solid::ModelEvaluator::BeamInteraction::Map> const submodelmap)
{
  check_init_setup();
  // no active influence on other submodels
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::reset()
{
  check_init_setup();

  std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>::const_iterator iter;
  for (iter = contact_elepairs_.begin(); iter != contact_elepairs_.end(); ++iter)
  {
    std::shared_ptr<BeamInteraction::BeamContactPair> elepairptr = *iter;

    std::vector<const Core::Elements::Element*> element_ptr(2);

    element_ptr[0] = elepairptr->element1();
    element_ptr[1] = elepairptr->element2();

    // element Dof values relevant for centerline interpolation
    std::vector<std::vector<double>> element_posdofvec_absolutevalues(2);

    for (unsigned int ielement = 0; ielement < 2; ++ielement)
    {
      // extract the Dof values of this element from displacement vector
      BeamInteraction::Utils::extract_pos_dof_vec_absolute_values(discret(), element_ptr[ielement],
          *beam_interaction_data_state_ptr()->get_dis_col_np(),
          element_posdofvec_absolutevalues[ielement]);
    }

    // update positional Dof values in the interaction element pair object
    elepairptr->reset_state(
        element_posdofvec_absolutevalues[0], element_posdofvec_absolutevalues[1]);

    // update rotational Dof values in the interaction pair object
    elepairptr->reset_rotation_state(
        discret(), beam_interaction_data_state_ptr()->get_dis_col_np());
  }

  // Set restart displacements in the pairs.
  set_restart_displacement_in_pairs();

  // Update the geometry pair evaluation data.
  beam_interaction_conditions_ptr_->set_state(discret_ptr(), beam_interaction_data_state_ptr());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool BeamInteraction::SubmodelEvaluator::BeamContact::evaluate_force()
{
  check_init_setup();

  pre_evaluate();

  // Loop over the assembly manager and assemble contributions into the global force vector.
  for (auto& assembly_manager : assembly_managers_)
  {
    assembly_manager->evaluate_force_stiff(discret_ptr(), beam_interaction_data_state_ptr(),
        beam_interaction_data_state_ptr()->get_force_np(), nullptr);
  }

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool BeamInteraction::SubmodelEvaluator::BeamContact::evaluate_stiff()
{
  check_init_setup();

  pre_evaluate();

  // Loop over the assembly manager and assemble contributions into the global stiffness matrix.
  for (auto& assembly_manager : assembly_managers_)
  {
    assembly_manager->evaluate_force_stiff(discret_ptr(), beam_interaction_data_state_ptr(),
        nullptr, beam_interaction_data_state_ptr()->get_stiff());
  }

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool BeamInteraction::SubmodelEvaluator::BeamContact::evaluate_force_stiff()
{
  check_init_setup();

  pre_evaluate();

  // Loop over the assembly manager and assemble contributions into the global force vector and
  // stiffness matrix.
  for (auto& assembly_manager : assembly_managers_)
    assembly_manager->evaluate_force_stiff(discret_ptr(), beam_interaction_data_state_ptr(),
        beam_interaction_data_state_ptr()->get_force_np(),
        beam_interaction_data_state_ptr()->get_stiff());

  print_active_beam_contact_set(Core::IO::cout.os(Core::IO::verbose));

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::pre_evaluate()
{
  for (auto& elepairptr : contact_elepairs_) elepairptr->pre_evaluate();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::update_step_state(const double& timefac_n)
{
  check_init_setup();

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool BeamInteraction::SubmodelEvaluator::BeamContact::pre_update_step_element(bool beam_redist)
{
  check_init_setup();

  /* Fixme
   * writing vtk output needs to be done BEFORE updating (and thus clearing
   * element pairs)
   * move this to runtime_output_step_state as soon as we keep element pairs
   * from previous time step */
  /* Fixme
   * writing this output also must be done BEFORE re-distribution which
   * currently happens in Solid::ModelEvaluator::BeamInteraction::update_step_element()
   * before calling update_step_element() on all submodels.
   * Hence, the only option currently is to call it from pre_update_step_element() */
  /* Note: another option would be to not use any data from state vectors or elements and only
   * write previously computed and (locally) stored data at this point. Like
   * this, it works in SubmodelEvaluator::BeamPotential */
  if (visualization_manager_ptr_ != nullptr and
      g_state().get_step_n() % beam_contact_params()
                                   .beam_contact_runtime_visualization_output_params()
                                   ->output_interval_in_steps() ==
          0)
  {
    write_time_step_output_runtime_beam_contact();
  }
  if (beam_to_solid_volume_meshtying_visualization_output_writer_ptr_ != nullptr)
    beam_to_solid_volume_meshtying_visualization_output_writer_ptr_->write_output_runtime(this);
  if (beam_to_solid_surface_visualization_output_writer_ptr_ != nullptr)
    beam_to_solid_surface_visualization_output_writer_ptr_->write_output_runtime(this);
  if (beam_to_solid_surface_visualization_output_writer_contact_ptr_ != nullptr)
    beam_to_solid_surface_visualization_output_writer_contact_ptr_->write_output_runtime(this);

  // not repartition of binning discretization necessary
  return false;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::update_step_element(bool repartition_was_done)
{
  check_init_setup();

  nearby_elements_map_.clear();
  find_and_store_neighboring_elements();
  create_beam_contact_element_pairs();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::post_update_step_element()
{
  check_init_setup();

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::map<Solid::EnergyType, double> BeamInteraction::SubmodelEvaluator::BeamContact::get_energy()
    const
{
  check_init_setup();

  std::map<Solid::EnergyType, double> contact_penalty_potential;

  for (auto& elepairptr : contact_elepairs_)
  {
    contact_penalty_potential[Solid::beam_contact_penalty_potential] += elepairptr->get_energy();
  }

  for (const auto& assembly_manager : assembly_managers_)
  {
    contact_penalty_potential[Solid::beam_contact_penalty_potential] +=
        assembly_manager->get_energy(g_state().get_dis_np());
  }

  return contact_penalty_potential;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::output_step_state(
    Core::IO::DiscretizationWriter& iowriter) const
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::runtime_output_step_state() const {}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::init_output_runtime_beam_contact()
{
  check_init();

  visualization_manager_ptr_ = std::make_shared<Core::IO::VisualizationManager>(
      beam_contact_params()
          .beam_contact_runtime_visualization_output_params()
          ->get_visualization_parameters(),
      discret().get_comm(), "beam-contact");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::write_time_step_output_runtime_beam_contact()
    const
{
  check_init_setup();

  auto [output_time, output_step] = Core::IO::get_time_and_time_step_index_for_output(
      beam_contact_params()
          .beam_contact_runtime_visualization_output_params()
          ->get_visualization_parameters(),
      g_state().get_time_n(), g_state().get_step_n());
  write_output_runtime_beam_contact(output_step, output_time);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::write_iteration_output_runtime_beam_contact(
    int iteration_number) const
{
  check_init_setup();

  auto [output_time, output_step] = Core::IO::get_time_and_time_step_index_for_output(
      beam_contact_params()
          .beam_contact_runtime_visualization_output_params()
          ->get_visualization_parameters(),
      g_state().get_time_n(), g_state().get_step_n(), iteration_number);
  write_output_runtime_beam_contact(output_step, output_time);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::write_output_runtime_beam_contact(
    int timestep_number, double time) const
{
  check_init_setup();

  const unsigned int num_spatial_dimensions = 3;

  // number of active point contact point pairs * 2 = number of row points for writer object
  unsigned int num_row_points = 0;

  // loop over contact pairs and retrieve number of all active contact point pairs
  std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>::const_iterator pair_iter;
  for (pair_iter = contact_elepairs_.begin(); pair_iter != contact_elepairs_.end(); ++pair_iter)
  {
    if ((*pair_iter)->get_type() == ContactPairType::beam_to_beam_contact)
    {
      num_row_points += 2 * (*pair_iter)->get_num_all_active_contact_point_pairs();
    }
  }

  // get and prepare storage for point coordinate values
  std::vector<double>& point_coordinates =
      visualization_manager_ptr_->get_visualization_data().get_point_coordinates();
  point_coordinates.clear();
  point_coordinates.reserve(num_spatial_dimensions * num_row_points);

  // contact force values: collect data and append to visualization results if desired
  std::vector<double> contact_force_vector;
  contact_force_vector.reserve(num_spatial_dimensions * num_row_points);

  // gap values: collect data and append to visualization results if desired
  std::vector<double> gaps;
  gaps.reserve(num_row_points);

  std::vector<double> angles;
  angles.reserve(num_row_points);
  std::vector<int> types;
  types.reserve(num_row_points);

  // loop over my points and collect the geometry/grid data, i.e. contact points
  std::vector<Core::LinAlg::Matrix<3, 1, double>> coordinates_ele1_this_pair;
  std::vector<Core::LinAlg::Matrix<3, 1, double>> coordinates_ele2_this_pair;

  std::vector<double> contact_forces_this_pair;
  std::vector<double> angles_this_pair;
  std::vector<int> types_this_pair;
  std::vector<double> gaps_this_pair;

  // loop over contact pairs and retrieve all active contact point coordinates
  for (pair_iter = contact_elepairs_.begin(); pair_iter != contact_elepairs_.end(); ++pair_iter)
  {
    // ensure that
    if ((*pair_iter)->get_contact_flag() == true &&
        (*pair_iter)->get_type() == ContactPairType::beam_to_beam_contact)
    {
      // active contact points of element 1 and element 2
      (*pair_iter)->get_all_active_contact_point_coords_element1(coordinates_ele1_this_pair);
      (*pair_iter)->get_all_active_contact_point_coords_element2(coordinates_ele2_this_pair);
      (*pair_iter)
          ->get_all_active_beam_to_beam_visualization_values(
              contact_forces_this_pair, gaps_this_pair, angles_this_pair, types_this_pair);

      const unsigned int num_active_point_pairs = (unsigned int)coordinates_ele1_this_pair.size();

      FOUR_C_ASSERT(num_active_point_pairs == (unsigned int)coordinates_ele2_this_pair.size(),
          "number of active points on element 1 does not match number of active points "
          "on element 2!");

      FOUR_C_ASSERT(num_active_point_pairs == (unsigned int)contact_forces_this_pair.size(),
          "number of active points on element 1 does not match number of contact forces!");

      FOUR_C_ASSERT(num_active_point_pairs == (unsigned int)gaps_this_pair.size(),
          "number of active points on element 1 does not match number of gap values!");


      for (unsigned int ipointpair = 0; ipointpair < num_active_point_pairs; ++ipointpair)
      {
        Core::LinAlg::Matrix<3, 1, double> normal_vector;
        normal_vector.update(1.0, coordinates_ele1_this_pair[ipointpair], -1.0,
            coordinates_ele2_this_pair[ipointpair]);

        // contact point on first element
        for (unsigned int idim = 0; idim < num_spatial_dimensions; ++idim)
        {
          point_coordinates.push_back(coordinates_ele1_this_pair[ipointpair](idim));

          contact_force_vector.push_back(
              contact_forces_this_pair[ipointpair] * normal_vector(idim));
        }
        gaps.push_back(gaps_this_pair[ipointpair]);
        angles.push_back(angles_this_pair[ipointpair]);
        types.push_back(types_this_pair[ipointpair]);

        // contact point on second element
        for (unsigned int idim = 0; idim < num_spatial_dimensions; ++idim)
        {
          point_coordinates.push_back(coordinates_ele2_this_pair[ipointpair](idim));

          contact_force_vector.push_back(
              -1.0 * contact_forces_this_pair[ipointpair] * normal_vector(idim));
        }
        gaps.push_back(gaps_this_pair[ipointpair]);
        angles.push_back(angles_this_pair[ipointpair]);
        types.push_back(types_this_pair[ipointpair]);
      }
    }
  }


  // append all desired output data to the writer object's storage
  if (beam_contact_params()
          .beam_contact_runtime_visualization_output_params()
          ->is_write_contact_forces())
  {
    visualization_manager_ptr_->get_visualization_data().set_point_data_vector(
        "force", contact_force_vector, num_spatial_dimensions);
  }

  if (beam_contact_params().beam_contact_runtime_visualization_output_params()->is_write_gaps())
  {
    visualization_manager_ptr_->get_visualization_data().set_point_data_vector("gap", gaps, 1);
  }

  if (beam_contact_params().beam_contact_runtime_visualization_output_params()->is_write_angles())
  {
    visualization_manager_ptr_->get_visualization_data().set_point_data_vector("angle", angles, 1);
  }

  if (beam_contact_params().beam_contact_runtime_visualization_output_params()->is_write_types())
  {
    visualization_manager_ptr_->get_visualization_data().set_point_data_vector("type", types, 1);
  }

  // finalize everything and write all required vtk files to filesystem
  visualization_manager_ptr_->write_to_disk(time, timestep_number);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::reset_step_state() { check_init_setup(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::write_restart(
    Core::IO::DiscretizationWriter& ia_writer, Core::IO::DiscretizationWriter& bin_writer) const
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::pre_read_restart()
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::read_restart(
    Core::IO::DiscretizationReader& ia_reader, Core::IO::DiscretizationReader& bin_reader)
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::post_read_restart()
{
  check_init_setup();
  nearby_elements_map_.clear();
  find_and_store_neighboring_elements();
  create_beam_contact_element_pairs();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::run_post_iterate(
    const ::NOX::Solver::Generic& solver)
{
  check_init_setup();

  if (visualization_manager_ptr_ != nullptr and
      beam_contact_params()
          .beam_contact_runtime_visualization_output_params()
          ->output_every_iteration())
  {
    write_iteration_output_runtime_beam_contact(solver.getNumIterations());
  }
  if (beam_to_solid_volume_meshtying_visualization_output_writer_ptr_ != nullptr)
  {
    beam_to_solid_volume_meshtying_visualization_output_writer_ptr_->write_output_runtime_iteration(
        this, solver.getNumIterations());
  }
  if (beam_to_solid_surface_visualization_output_writer_ptr_ != nullptr)
  {
    beam_to_solid_surface_visualization_output_writer_ptr_->write_output_runtime_iteration(
        this, solver.getNumIterations());
  }
  if (beam_to_solid_surface_visualization_output_writer_contact_ptr_ != nullptr)
  {
    beam_to_solid_surface_visualization_output_writer_contact_ptr_->write_output_runtime_iteration(
        this, solver.getNumIterations());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::add_bins_to_bin_col_map(
    std::set<int>& colbins)
{
  check_init_setup();
  // nothing to do
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::
    add_bins_with_relevant_content_for_ia_discret_col_map(std::set<int>& colbins) const
{
  check_init_setup();
  // nothing to do
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::get_half_interaction_distance(
    double& half_interaction_distance)
{
  check_init_setup();

  // todo: choose meaningful safety factor
  double safe_fac = 1.5;

  // loop over all beams to get largest interaction radius
  double locmax_ia_distance = 0.0;
  double curr_ia_distance = 0.0;
  int const numroweles = ele_type_map_extractor_ptr()->beam_map()->num_my_elements();
  for (int rowele_i = 0; rowele_i < numroweles; ++rowele_i)
  {
    int const elegid = ele_type_map_extractor_ptr()->beam_map()->gid(rowele_i);
    Discret::Elements::Beam3Base* currele =
        dynamic_cast<Discret::Elements::Beam3Base*>(discret_ptr()->g_element(elegid));

    curr_ia_distance = currele->get_circular_cross_section_radius_for_interactions();

    if (curr_ia_distance > locmax_ia_distance) locmax_ia_distance = curr_ia_distance;
  }

  // get global maximum
  double globalmax_beam_ia_distance = 0.0;
  // build sum over all procs
  MPI_Allreduce(&locmax_ia_distance, &globalmax_beam_ia_distance, 1, MPI_DOUBLE, MPI_MAX,
      discret().get_comm());

  // i) beam to beam contact
  if (have_contact_type(Core::Binstrategy::Utils::BinContentType::Beam))
  {
    // safety factor
    globalmax_beam_ia_distance *= safe_fac;

    half_interaction_distance = (globalmax_beam_ia_distance > half_interaction_distance)
                                    ? globalmax_beam_ia_distance
                                    : half_interaction_distance;

    // some screen output
    if (g_state().get_my_rank() == 0)
      std::cout << " beam to beam contact half interaction distance " << globalmax_beam_ia_distance
                << std::endl;
  }

  // ii) beam to sphere contact
  if (have_contact_type(Core::Binstrategy::Utils::BinContentType::RigidSphere))
  {
    // loop over all spheres
    double curr_ia_dist = 0.0;
    double loc_max_ia_dist = 0.0;
    int unsigned const numrowsphereeles = ele_type_map_extractor().sphere_map()->num_my_elements();
    for (unsigned int rowele_i = 0; rowele_i < numrowsphereeles; ++rowele_i)
    {
      int const elegid = ele_type_map_extractor().sphere_map()->gid(rowele_i);
      Discret::Elements::Rigidsphere* sphere =
          dynamic_cast<Discret::Elements::Rigidsphere*>(discret().g_element(elegid));

      curr_ia_dist = sphere->radius() + globalmax_beam_ia_distance;

      // update distance
      loc_max_ia_dist = (curr_ia_dist > loc_max_ia_dist) ? curr_ia_dist : loc_max_ia_dist;
    }

    // get global maximum
    double spherebeamlinking_half_interaction_distance_global = 0.0;
    // build sum over all procs
    MPI_Allreduce(&loc_max_ia_dist, &spherebeamlinking_half_interaction_distance_global, 1,
        MPI_DOUBLE, MPI_MAX, discret().get_comm());

    half_interaction_distance =
        (spherebeamlinking_half_interaction_distance_global > half_interaction_distance)
            ? spherebeamlinking_half_interaction_distance_global
            : half_interaction_distance;

    // some screen output
    if (g_state().get_my_rank() == 0)
      Core::IO::cout(Core::IO::verbose)
          << " sphere to beam contact half interaction distance "
          << spherebeamlinking_half_interaction_distance_global << Core::IO::endl;
  }

  // iii) beam to solid contact
  if (have_contact_type(Core::Binstrategy::Utils::BinContentType::Solid))
  {
    FOUR_C_THROW("Not yet implemented for beam to solid contact");
  }
}

std::shared_ptr<const FourC::Core::LinAlg::Map>
BeamInteraction::SubmodelEvaluator::BeamContact::get_lagrange_map() const
{
  return get_lagrange_multiplier_assembly_manager()->get_mortar_manager()->get_lambda_dof_row_map();
}

void BeamInteraction::SubmodelEvaluator::BeamContact::assemble_force(
    Core::LinAlg::Vector<double>& f) const
{
  get_lagrange_multiplier_assembly_manager()->get_mortar_manager()->assemble_force(
      g_state(), f, *beam_interaction_data_state_ptr());
};

void BeamInteraction::SubmodelEvaluator::BeamContact::assemble_stiff(
    Core::LinAlg::SparseOperator& jac) const
{
  get_lagrange_multiplier_assembly_manager()->get_mortar_manager()->assemble_stiff(
      g_state(), jac, *beam_interaction_data_state_ptr());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BeamInteraction::SubmodelEvaluator::BeamContact::have_contact_type(
    Core::Binstrategy::Utils::BinContentType const& contacttype) const
{
  check_init();
  return (std::find(contactelementtypes_.begin(), contactelementtypes_.end(), contacttype) !=
          contactelementtypes_.end());
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::find_and_store_neighboring_elements()
{
  // measure time for evaluating this function
  TEUCHOS_FUNC_TIME_MONITOR(
      "BeamInteraction::SubmodelEvaluator::BeamContact::find_and_store_neighboring_elements");

  check_init();

  // Build the ids of the elements for the beam-to-solid conditions.
  beam_interaction_conditions_ptr_->build_id_sets(discret_ptr());

  if (beam_interaction_params_ptr_->get_search_strategy() ==
      Inpar::BeamInteraction::SearchStrategy::bruteforce_with_binning)
  {
    // loop over all row beam elements
    // note: like this we ensure that first element of pair is always a beam element, also only
    // beam to something contact considered
    int const numroweles = ele_type_map_extractor_ptr()->beam_map()->num_my_elements();
    for (int rowele_i = 0; rowele_i < numroweles; ++rowele_i)
    {
      int const elegid = ele_type_map_extractor_ptr()->beam_map()->gid(rowele_i);
      Core::Elements::Element* currele = discret_ptr()->g_element(elegid);

      // (unique) set of neighboring bins for all col bins assigned to current element
      std::set<int> neighboring_binIds;

      // loop over all bins touched by currele
      std::set<int>::const_iterator biniter;
      for (biniter = beam_interaction_data_state_ptr()->get_row_ele_to_bin_set(elegid).begin();
          biniter != beam_interaction_data_state_ptr()->get_row_ele_to_bin_set(elegid).end();
          ++biniter)
      {
        std::vector<int> loc_neighboring_binIds;
        // in three-dimensional space: 26 neighbouring-bins + 1 self
        loc_neighboring_binIds.reserve(27);

        // do not check on existence here -> shifted to GetBinContent
        bin_strategy_ptr()->get_neighbor_and_own_bin_ids(*biniter, loc_neighboring_binIds);

        // build up comprehensive unique set of neighboring bins
        neighboring_binIds.insert(loc_neighboring_binIds.begin(), loc_neighboring_binIds.end());
      }
      // get set of elements that reside in neighboring bins
      std::vector<int> glob_neighboring_binIds(
          neighboring_binIds.begin(), neighboring_binIds.end());
      std::set<Core::Elements::Element*> neighboring_elements;
      bin_strategy_ptr()->get_bin_content(
          neighboring_elements, contactelementtypes_, glob_neighboring_binIds);

      // sort out elements that should not be considered in contact evaluation
      select_eles_to_be_considered_for_contact_evaluation(currele, neighboring_elements);

      nearby_elements_map_[elegid] = neighboring_elements;
    }
  }
  else if (beam_interaction_params_ptr_->get_search_strategy() ==
           Inpar::BeamInteraction::SearchStrategy::bounding_volume_hierarchy)
  {
    // Get vector of all beam element bounding boxes.
    int const numroweles = ele_type_map_extractor_ptr()->beam_map()->num_my_elements();
    std::vector<std::pair<int, Core::GeometricSearch::BoundingVolume>> beam_bounding_boxes;
    for (int rowele_i = 0; rowele_i < numroweles; ++rowele_i)
    {
      int const elegid = ele_type_map_extractor_ptr()->beam_map()->gid(rowele_i);
      Core::Elements::Element* currele = discret().g_element(elegid);

      beam_bounding_boxes.emplace_back(
          std::make_pair(elegid, currele->get_bounding_volume(discret(),
                                     *beam_interaction_data_state_ptr()->get_dis_col_np(),
                                     *geometric_search_params_ptr_)));
    }

    // Get vector of the bounding boxes of all possible interacting elements (also including beams
    // if beam-to-beam contact is activated).
    int const numcoleles = discret().num_my_col_elements();
    std::vector<std::pair<int, Core::GeometricSearch::BoundingVolume>> other_bounding_boxes;
    for (int colele_i = 0; colele_i < numcoleles; ++colele_i)
    {
      // Check if the current element is relevant for beam-to-xxx contact.
      Core::Elements::Element* currele = discret().l_col_element(colele_i);
      const Core::Binstrategy::Utils::BinContentType contact_type =
          BeamInteraction::Utils::convert_element_to_bin_content_type(currele);
      if (std::find(contactelementtypes_.begin(), contactelementtypes_.end(), contact_type) !=
          contactelementtypes_.end())
      {
        other_bounding_boxes.emplace_back(
            std::make_pair(currele->id(), currele->get_bounding_volume(discret(),
                                              *beam_interaction_data_state_ptr()->get_dis_col_np(),
                                              *geometric_search_params_ptr_)));
      }
    }

    // Get colliding pairs.
    const auto& collision_pairs = collision_search(other_bounding_boxes, beam_bounding_boxes,
        discret().get_comm(), geometric_search_params_ptr_->verbosity_);

    // Create the beam-to-xxx pair pointers according to the search.
    for (const auto& pair : collision_pairs)
    {
      nearby_elements_map_[pair.gid_predicate].insert(discret().g_element(pair.gid_primitive));
    }

    // Pre-filter some pairs
    for (auto& [beam_gid, neighboring_elements] : nearby_elements_map_)
    {
      const Core::Elements::Element* currele = discret().g_element(beam_gid);
      select_eles_to_be_considered_for_contact_evaluation(currele, neighboring_elements);
    }

    // Check if the primitives and predicates should be output
    if (geometric_search_visualization_ptr_ != nullptr)
    {
      // Output is desired, so create it right here, because we only search the pairs once per time
      // step anyways.
      geometric_search_visualization_ptr_->write_primitives_and_predicates_to_disk(
          g_state().get_time_n(), g_state().get_step_n(), other_bounding_boxes,
          beam_bounding_boxes);
    }
  }
  else
    FOUR_C_THROW("No appropriate search strategy for beam interaction chosen!");
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::
    select_eles_to_be_considered_for_contact_evaluation(
        const Core::Elements::Element* currele, std::set<Core::Elements::Element*>& neighbors) const
{
  check_init();

  // sort out elements that should not be considered in contact evaluation
  std::set<Core::Elements::Element*>::iterator either;
  for (either = neighbors.begin(); either != neighbors.end();)
  {
    bool toerase = false;
    // 1) ensure that an element will not be in contact with it self
    if (currele->id() == (*either)->id())
    {
      toerase = true;
    }
    // 2) ensure each contact only evaluated once (keep in mind that we are
    //    using FEMatrices and FEvectors -> || (*either)->Owner() != myrank not necessary)
    // note: as we are only looping over beam elements, only beam to beam contact needs id check
    // here
    else if (dynamic_cast<Discret::Elements::Beam3Base*>(*either) != nullptr and
             not(currele->id() < (*either)->id()))
    {
      toerase = true;
    }
    // 3) ensure that two elements sharing the same node do not get into contact
    else
    {
      for (int i = 0; i < (*either)->num_node(); ++i)
        for (int j = 0; j < currele->num_node(); ++j)
          if ((*either)->node_ids()[i] == currele->node_ids()[j]) toerase = true;
    }

    if (toerase)
      neighbors.erase(either++);
    else
      ++either;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::create_beam_contact_element_pairs()
{
  // Todo maybe keep existing pairs and reuse them ?
  contact_elepairs_.clear();
  assembly_managers_.clear();

  // clear the geometry evaluation data
  beam_interaction_conditions_ptr_->clear();

  std::map<int, std::set<Core::Elements::Element*>>::const_iterator nearbyeleiter;

  for (nearbyeleiter = nearby_elements_map_.begin(); nearbyeleiter != nearby_elements_map_.end();
      ++nearbyeleiter)
  {
    const int elegid = nearbyeleiter->first;
    std::vector<Core::Elements::Element const*> ele_ptrs(2);
    ele_ptrs[0] = discret_ptr()->g_element(elegid);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (dynamic_cast<Discret::Elements::Beam3Base const*>(ele_ptrs[0]) == nullptr)
      FOUR_C_THROW("first element of element pair must be a beam element");
#endif

    std::set<Core::Elements::Element*>::const_iterator secondeleiter;
    for (secondeleiter = nearbyeleiter->second.begin();
        secondeleiter != nearbyeleiter->second.end(); ++secondeleiter)
    {
      ele_ptrs[1] = *secondeleiter;

      // construct, init and setup contact pairs
      std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>> newbeaminteractionpairs =
          BeamInteraction::BeamContactPair::create(ele_ptrs, *beam_interaction_conditions_ptr_);

      for (auto& pair : newbeaminteractionpairs)
      {
        pair->init(beam_contact_params_ptr_, ele_ptrs);
        pair->setup();

        // add to list of current contact pairs
        contact_elepairs_.push_back(pair);
      }
    }
  }

  // Setup the geometry evaluation data.
  beam_interaction_conditions_ptr_->setup(discret_ptr());

  // Get the pairs that can be assembled directly.
  std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>> assembly_pairs_direct;
  for (auto& elepairptr : contact_elepairs_)
    if (elepairptr->is_assembly_direct()) assembly_pairs_direct.push_back(elepairptr);

  // Check if there are any processors that require a direct element assembly method.
  // We need to do this as in some assembly methods MPI communications are needed and the
  // simulation crashes if the assembly manager is not on all ranks.
  int my_direct_pairs = assembly_pairs_direct.size();
  int global_direct_pairs = 0;
  global_direct_pairs = Core::Communication::sum_all(my_direct_pairs, discret().get_comm());

  // Create the needed assembly manager.
  if (global_direct_pairs > 0)
    assembly_managers_.push_back(
        std::make_shared<BeamInteraction::SubmodelEvaluator::BeamContactAssemblyManagerDirect>(

            assembly_pairs_direct));

  // Each indirect assembly manager depends on a beam interaction.
  beam_interaction_conditions_ptr_->create_indirect_assembly_managers(
      discret_ptr(), assembly_managers_);

  Core::IO::cout(Core::IO::standard)
      << "PID " << std::setw(2) << std::right << g_state().get_my_rank() << " currently monitors "
      << std::setw(5) << std::right << contact_elepairs_.size() << " beam contact pairs"
      << Core::IO::endl;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::set_restart_displacement_in_pairs()
{
  check_init_setup();

  if (beam_interaction_data_state_ptr()->get_restart_coupling_flag())
  {
    for (auto& pair : contact_elepairs_)
    {
      // Element Dof values at the restart state.
      std::vector<std::vector<double>> element_restart_displacement_(2);

      for (unsigned int i_element = 0; i_element < 2; ++i_element)
      {
        // Extract the Dof values of this element from the restart vector
        BeamInteraction::Utils::extract_pos_dof_vec_values(discret(), pair->get_element(i_element),
            *beam_interaction_data_state_ptr()->get_dis_restart_col(),
            element_restart_displacement_[i_element]);
      }

      // Set the displacement state in the pair.
      pair->set_restart_displacement(element_restart_displacement_);
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::print_all_beam_contact_element_pairs(
    std::ostream& out) const
{
  out << "\n\nCurrent BeamContactElementPairs: ";
  std::vector<std::shared_ptr<BeamInteraction::BeamContactPair>>::const_iterator iter;
  for (iter = contact_elepairs_.begin(); iter != contact_elepairs_.end(); ++iter)
    (*iter)->print(out);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BeamInteraction::SubmodelEvaluator::BeamContact::print_active_beam_contact_set(
    std::ostream& out) const
{
  bool atleastoneactivepair = false;

  for (auto& elepairptr : contact_elepairs_)
    if (elepairptr->get_contact_flag() == true) atleastoneactivepair = true;


  if (atleastoneactivepair)
  {
    out << "\n    Active Beam-To-? Contact Set (PID " << g_state().get_my_rank()
        << "):-----------------------------------------\n";
    out << "    ID1            ID2              T    xi       eta      angle    gap         "
           "force\n";


    for (auto& elepairptr : contact_elepairs_)
      elepairptr->print_summary_one_line_per_active_segment_pair(out);

    out << std::endl;
  }
}

FOUR_C_NAMESPACE_CLOSE
