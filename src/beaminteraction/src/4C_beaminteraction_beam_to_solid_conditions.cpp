// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_solid_conditions.hpp"

#include "4C_beam3_euler_bernoulli.hpp"
#include "4C_beam3_reissner.hpp"
#include "4C_beaminteraction_beam_to_solid_mortar_manager_contact.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_contact_pair.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_contact_pair_mortar.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_contact_params.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_meshtying_pair_gauss_point.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_meshtying_pair_gauss_point_FAD.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_meshtying_pair_mortar.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_meshtying_pair_mortar_FAD.hpp"
#include "4C_beaminteraction_beam_to_solid_surface_meshtying_params.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_2d-3d_full.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_2d-3d_mortar.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_2d-3d_plane.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_gauss_point.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_mortar.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_pair_mortar_rotation.hpp"
#include "4C_beaminteraction_beam_to_solid_volume_meshtying_params.hpp"
#include "4C_beaminteraction_contact_pair.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_beaminteraction_str_model_evaluator_datastate.hpp"
#include "4C_beaminteraction_submodel_evaluator_beamcontact_assembly_manager_indirect.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_faces.hpp"
#include "4C_geometry_pair_line_to_3D_evaluation_data.hpp"
#include "4C_geometry_pair_line_to_surface_evaluation_data.hpp"
#include "4C_inpar_beam_to_solid.hpp"
#include "4C_utils_std23_unreachable.hpp"

FOUR_C_NAMESPACE_OPEN


/**
 *
 */
BeamInteraction::BeamToSolidCondition::BeamToSolidCondition(
    const Core::Conditions::Condition& condition_line,
    const Core::Conditions::Condition& condition_other,
    const std::shared_ptr<const BeamToSolidParamsBase>& beam_to_solid_params)
    : BeamInteractionConditionBase(condition_line),
      geometry_evaluation_data_(nullptr),
      condition_other_(&condition_other),
      condition_contact_pairs_(),
      beam_to_solid_params_(beam_to_solid_params)
{
}

/**
 *
 */
bool BeamInteraction::BeamToSolidCondition::ids_in_condition(
    const int id_line, const int id_other) const
{
  if (line_ids_.find(id_line) != line_ids_.end())
    if (id_in_other(id_other)) return true;
  return false;
}

/**
 *
 */
void BeamInteraction::BeamToSolidCondition::clear()
{
  BeamInteractionConditionBase::clear();
  geometry_evaluation_data_->clear();
  condition_contact_pairs_.clear();
}

/**
 *
 */
std::shared_ptr<BeamInteraction::BeamContactPair>
BeamInteraction::BeamToSolidCondition::create_contact_pair(
    const std::vector<Core::Elements::Element const*>& ele_ptrs)
{
  // Check if the given elements are in this condition.
  if (!ids_in_condition(ele_ptrs[0]->id(), ele_ptrs[1]->id())) return nullptr;

  // Create the beam contact pair.
  std::shared_ptr<BeamInteraction::BeamContactPair> contact_pair =
      create_contact_pair_internal(ele_ptrs);

  if (contact_pair != nullptr)
  {
    // Create the geometry pair on the beam contact pair.
    contact_pair->create_geometry_pair(ele_ptrs[0], ele_ptrs[1], geometry_evaluation_data_);

    // Add to the internal vector which keeps track of the created contact pairs.
    condition_contact_pairs_.push_back(contact_pair);
  }
  else
    FOUR_C_THROW(
        "No contact pair was created. This is fatal, since we want to create a geometry pair on "
        "the contact pair here.");

  // Return the newly created pair.
  return contact_pair;
}

/**
 *
 */
std::shared_ptr<BeamInteraction::SubmodelEvaluator::BeamContactAssemblyManager>
BeamInteraction::BeamToSolidCondition::create_indirect_assembly_manager(
    const std::shared_ptr<const Core::FE::Discretization>& discret)
{
  if (beam_to_solid_params_->get_contact_discretization() ==
          Inpar::BeamToSolid::BeamToSolidContactDiscretization::mortar ||
      beam_to_solid_params_->get_contact_discretization() ==
          Inpar::BeamToSolid::BeamToSolidContactDiscretization::mortar_cross_section)
  {
    // Create the mortar manager. We add 1 to the MaxAllGID since this gives the maximum GID and NOT
    // the length of the GIDs.
    const int start_gid_lambda = discret->dof_row_map()->max_all_gid() + 1;
    std::shared_ptr<BeamToSolidMortarManager> mortar_manager = nullptr;
    if (std::dynamic_pointer_cast<const BeamToSolidSurfaceContactParams>(beam_to_solid_params_) ==
        nullptr)
    {
      mortar_manager = std::make_shared<BeamInteraction::BeamToSolidMortarManager>(
          discret, beam_to_solid_params_, start_gid_lambda);
    }
    else
    {
      mortar_manager = std::make_shared<BeamInteraction::BeamToSolidMortarManagerContact>(
          discret, beam_to_solid_params_, start_gid_lambda);
    }

    // Setup the mortar manager.
    mortar_manager->setup();
    mortar_manager->set_local_maps(condition_contact_pairs_);

    // Create the indirect assembly manager with the mortar manager
    return std::make_shared<SubmodelEvaluator::BeamContactAssemblyManagerInDirect>(mortar_manager);
  }
  return nullptr;
}


/**
 *
 */
BeamInteraction::BeamToSolidConditionVolumeMeshtying::BeamToSolidConditionVolumeMeshtying(
    const Core::Conditions::Condition& condition_line,
    const Core::Conditions::Condition& condition_other,
    const std::shared_ptr<const BeamToSolidParamsBase>& beam_to_solid_params)
    : BeamToSolidCondition(condition_line, condition_other, beam_to_solid_params)
{
  // Get the input parameter list that will be passed to the geometry pair.
  const Teuchos::ParameterList& input_parameter_list =
      Global::Problem::instance()->beam_interaction_params().sublist(
          "BEAM TO SOLID VOLUME MESHTYING");

  // Create the geometry evaluation data for this condition.
  geometry_evaluation_data_ =
      std::make_shared<GeometryPair::LineTo3DEvaluationData>(input_parameter_list);
}

/**
 *
 */
void BeamInteraction::BeamToSolidConditionVolumeMeshtying::build_id_sets(
    const std::shared_ptr<const Core::FE::Discretization>& discretization)
{
  // Call the parent method to build the line maps.
  BeamToSolidCondition::build_id_sets(discretization);

  // Build the volume map.
  std::vector<int> volume_ids;
  condition_to_element_ids(*condition_other_, volume_ids);
  volume_ids_ = std::set<int>(volume_ids.begin(), volume_ids.end());
}

/**
 *
 */
template <template <typename...> class BtsClass, typename... BtsTemplateArguments>
std::shared_ptr<BeamInteraction::BeamContactPair>
BeamInteraction::create_beam_to_solid_volume_pair_shape(const Core::FE::CellType shape)
{
  switch (shape)
  {
    case Core::FE::CellType::hex8:
      return std::make_shared<
          BtsClass<GeometryPair::t_hermite, GeometryPair::t_hex8, BtsTemplateArguments...>>();
    case Core::FE::CellType::hex20:
      return std::make_shared<
          BtsClass<GeometryPair::t_hermite, GeometryPair::t_hex20, BtsTemplateArguments...>>();
    case Core::FE::CellType::hex27:
      return std::make_shared<
          BtsClass<GeometryPair::t_hermite, GeometryPair::t_hex27, BtsTemplateArguments...>>();
    case Core::FE::CellType::tet4:
      return std::make_shared<
          BtsClass<GeometryPair::t_hermite, GeometryPair::t_tet4, BtsTemplateArguments...>>();
    case Core::FE::CellType::tet10:
      return std::make_shared<
          BtsClass<GeometryPair::t_hermite, GeometryPair::t_tet10, BtsTemplateArguments...>>();
    case Core::FE::CellType::nurbs27:
      return std::make_shared<
          BtsClass<GeometryPair::t_hermite, GeometryPair::t_nurbs27, BtsTemplateArguments...>>();
    default:
      FOUR_C_THROW("Wrong element type for solid element.");
      return nullptr;
  }
}

/**
 *
 */
template <template <typename...> class BtsClass, typename... BtsTemplateArguments>
std::shared_ptr<BeamInteraction::BeamContactPair>
BeamInteraction::create_beam_to_solid_volume_pair_shape_no_nurbs(const Core::FE::CellType shape)
{
  switch (shape)
  {
    case Core::FE::CellType::hex8:
      return std::make_shared<
          BtsClass<GeometryPair::t_hermite, GeometryPair::t_hex8, BtsTemplateArguments...>>();
    case Core::FE::CellType::hex20:
      return std::make_shared<
          BtsClass<GeometryPair::t_hermite, GeometryPair::t_hex20, BtsTemplateArguments...>>();
    case Core::FE::CellType::hex27:
      return std::make_shared<
          BtsClass<GeometryPair::t_hermite, GeometryPair::t_hex27, BtsTemplateArguments...>>();
    case Core::FE::CellType::tet4:
      return std::make_shared<
          BtsClass<GeometryPair::t_hermite, GeometryPair::t_tet4, BtsTemplateArguments...>>();
    case Core::FE::CellType::tet10:
      return std::make_shared<
          BtsClass<GeometryPair::t_hermite, GeometryPair::t_tet10, BtsTemplateArguments...>>();
    default:
      FOUR_C_THROW("Wrong element type for solid element.");
      return nullptr;
  }
}

/**
 *
 */
template <template <typename...> class BtsClass, typename... BtsMortarTemplateArguments,
    typename... BtsMortarShape>
std::shared_ptr<BeamInteraction::BeamContactPair>
BeamInteraction::create_beam_to_solid_volume_pair_mortar(const Core::FE::CellType shape,
    const Inpar::BeamToSolid::BeamToSolidMortarShapefunctions mortar_shape_function,
    BtsMortarShape... other_mortar_shape_function)
{
  switch (mortar_shape_function)
  {
    case Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::line2:
      return create_beam_to_solid_volume_pair_mortar<BtsClass, BtsMortarTemplateArguments...,
          GeometryPair::t_line2>(shape, other_mortar_shape_function...);
    case Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::line3:
      return create_beam_to_solid_volume_pair_mortar<BtsClass, BtsMortarTemplateArguments...,
          GeometryPair::t_line3>(shape, other_mortar_shape_function...);
    case Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::line4:
      return create_beam_to_solid_volume_pair_mortar<BtsClass, BtsMortarTemplateArguments...,
          GeometryPair::t_line4>(shape, other_mortar_shape_function...);
    default:
      FOUR_C_THROW("Wrong mortar shape function.");
      return nullptr;
  }
}

/**
 *
 */
template <template <typename...> class BtsClass, typename... BtsMortarTemplateArguments>
std::shared_ptr<BeamInteraction::BeamContactPair>
BeamInteraction::create_beam_to_solid_volume_pair_mortar(const Core::FE::CellType shape)
{
  return create_beam_to_solid_volume_pair_shape<BtsClass, BtsMortarTemplateArguments...>(shape);
}

/**
 *
 */
std::shared_ptr<BeamInteraction::BeamContactPair>
BeamInteraction::BeamToSolidConditionVolumeMeshtying::create_contact_pair_internal(
    const std::vector<Core::Elements::Element const*>& ele_ptrs)
{
  const Core::FE::CellType shape = ele_ptrs[1]->shape();
  const auto beam_to_volume_params =
      std::dynamic_pointer_cast<const BeamToSolidVolumeMeshtyingParams>(beam_to_solid_params_);
  const Inpar::BeamToSolid::BeamToSolidContactDiscretization contact_discretization =
      beam_to_volume_params->get_contact_discretization();

  if (contact_discretization ==
      Inpar::BeamToSolid::BeamToSolidContactDiscretization::gauss_point_to_segment)
  {
    // Create the Gauss point to segment pairs.
    return create_beam_to_solid_volume_pair_shape<BeamToSolidVolumeMeshtyingPairGaussPoint>(shape);
  }
  else if (contact_discretization == Inpar::BeamToSolid::BeamToSolidContactDiscretization::mortar)
  {
    const Inpar::BeamToSolid::BeamToSolidMortarShapefunctions mortar_shape_function =
        beam_to_volume_params->get_mortar_shape_function_type();
    const Inpar::BeamToSolid::BeamToSolidMortarShapefunctions mortar_shape_function_rotation =
        beam_to_volume_params->get_mortar_shape_function_rotation_type();

    if (mortar_shape_function_rotation == Inpar::BeamToSolid::BeamToSolidMortarShapefunctions::none)
    {
      // Create the positional mortar pairs.
      return create_beam_to_solid_volume_pair_mortar<BeamToSolidVolumeMeshtyingPairMortar>(
          shape, mortar_shape_function);
    }
    else
    {
      // Create the rotational mortart pairs.
      return create_beam_to_solid_volume_pair_mortar<BeamToSolidVolumeMeshtyingPairMortarRotation>(
          shape, mortar_shape_function, mortar_shape_function_rotation);
    }
  }
  else if (contact_discretization ==
           Inpar::BeamToSolid::BeamToSolidContactDiscretization::gauss_point_cross_section)
  {
    // Depending on the type of beam element we create the correct beam-to-solid pair here.
    const auto sr_beam = dynamic_cast<const Discret::Elements::Beam3r*>(ele_ptrs[0]);
    const auto eb_beam = dynamic_cast<const Discret::Elements::Beam3eb*>(ele_ptrs[0]);
    if (sr_beam != nullptr)
      return create_beam_to_solid_volume_pair_shape_no_nurbs<
          BeamToSolidVolumeMeshtyingPair2D3DFull>(shape);
    else if (eb_beam != nullptr)
      return create_beam_to_solid_volume_pair_shape_no_nurbs<
          BeamToSolidVolumeMeshtyingPair2D3DPlane>(shape);
    else
      FOUR_C_THROW(
          "2D-3D coupling is only implemented for Simo-Reissner and torsion free beam elements.");
  }
  else if (contact_discretization ==
           Inpar::BeamToSolid::BeamToSolidContactDiscretization::mortar_cross_section)
  {
    return create_beam_to_solid_volume_pair_mortar_cross_section(shape,
        beam_to_volume_params->get_mortar_shape_function_type(),
        beam_to_volume_params->get_number_of_fourier_modes());
  }
  else
  {
    FOUR_C_THROW("Got unexpected contact discretization.");
  }

  // Default return value.
  return nullptr;
}

/**
 *
 */
BeamInteraction::BeamToSolidConditionSurface::BeamToSolidConditionSurface(
    const Core::Conditions::Condition& condition_line,
    const Core::Conditions::Condition& condition_other,
    const std::shared_ptr<const BeamToSolidParamsBase>& beam_to_solid_params,
    const bool is_mesh_tying_in)
    : BeamToSolidCondition(condition_line, condition_other, beam_to_solid_params),
      is_mesh_tying_(is_mesh_tying_in)
{
  // Get the input parameter list that will be passed to the geometry pair.
  std::string condition_name;
  if (is_mesh_tying())
    condition_name = "BEAM TO SOLID SURFACE MESHTYING";
  else
    condition_name = "BEAM TO SOLID SURFACE CONTACT";

  const Teuchos::ParameterList& input_parameter_list =
      Global::Problem::instance()->beam_interaction_params().sublist(condition_name);

  // Create the geometry evaluation data for this condition.
  geometry_evaluation_data_ =
      std::make_shared<GeometryPair::LineToSurfaceEvaluationData>(input_parameter_list);
}

/**
 *
 */
void BeamInteraction::BeamToSolidConditionSurface::build_id_sets(
    const std::shared_ptr<const Core::FE::Discretization>& discretization)
{
  // Call the parent method to build the line maps.
  BeamToSolidCondition::build_id_sets(discretization);

  // Build the surface map.
  surface_ids_ = condition_to_element_id_map(*condition_other_);
}

/**
 *
 */
void BeamInteraction::BeamToSolidConditionSurface::setup(
    const std::shared_ptr<const Core::FE::Discretization>& discret)
{
  // Call the parent method.
  BeamToSolidCondition::setup(discret);

  // Cast the geometry evaluation data to the correct type.
  auto line_to_surface_evaluation_data =
      std::dynamic_pointer_cast<GeometryPair::LineToSurfaceEvaluationData>(
          geometry_evaluation_data_);
  if (line_to_surface_evaluation_data == nullptr)
    FOUR_C_THROW("Could not cast to GeometryPair::LineToSurfaceEvaluationData.");

  // If the pairs are FAD, i.e., if the averaged normals have to be evaluated using FAD.
  int fad_order = 0;
  if (condition_contact_pairs_.size() > 0)
  {
    if (is_mesh_tying())
    {
      fad_order = condition_contact_pairs_[0]
                      ->params()
                      ->beam_to_solid_surface_meshtying_params()
                      ->get_fad_order();
    }
    else
    {
      fad_order = condition_contact_pairs_[0]
                      ->params()
                      ->beam_to_solid_surface_contact_params()
                      ->get_fad_order();
    }
  }

  // Loop over all pairs and add the needed face elements.
  std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>> pair_face_elements;
  pair_face_elements.clear();
  for (const auto& pair : condition_contact_pairs_)
  {
    const int solid_id = pair->element2()->id();
    auto find_in_condition = surface_ids_.find(solid_id);
    if (find_in_condition != surface_ids_.end())
    {
      // Check if the face is already in the pair_face_elements map.
      auto find_in_pair = pair_face_elements.find(solid_id);
      if (find_in_pair == pair_face_elements.end())
      {
        // The face element has to be created and added to the contact pair.
        std::shared_ptr<GeometryPair::FaceElement> new_face_element =
            GeometryPair::face_element_factory(find_in_condition->second, fad_order,
                line_to_surface_evaluation_data->get_surface_normal_strategy());
        new_face_element->set_part_of_pair(true);
        pair_face_elements[solid_id] = new_face_element;
        pair->set_face_element(new_face_element);
      }
      else
      {
        // Add the existing face element to the contact pair.
        pair->set_face_element(find_in_pair->second);
      }
    }
    else
    {
      FOUR_C_THROW("The face of the solid element {} is not in the current condition!",
          pair->element2()->id());
    }
  }

  // Now all faces of contact pairs are in pair_face_elements, we still need to add faces that are
  // needed for averaged normal calculation, but are not contained in any pair.
  std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>> face_elements_needed;
  face_elements_needed = pair_face_elements;
  for (const auto& face_element_iterator : pair_face_elements)
  {
    // Loop over the nodes of the face element.
    const Core::Nodes::Node* const* nodes = face_element_iterator.second->get_element()->nodes();
    for (int i_node = 0; i_node < face_element_iterator.second->get_element()->num_node(); i_node++)
    {
      // Loop over the elements connected to that node and check if they are in this condition.
      for (auto ele : nodes[i_node]->adjacent_elements())
      {
        const int element_id = ele.global_id();
        auto find_in_condition = surface_ids_.find(element_id);
        if (find_in_condition != surface_ids_.end())
        {
          // The element exists in this condition, check if it is already in the needed faces map.
          auto find_in_needed = face_elements_needed.find(element_id);
          if (find_in_needed == face_elements_needed.end())
          {
            // It is not already in the needed faces -> add it.
            face_elements_needed[element_id] =
                GeometryPair::face_element_factory(find_in_condition->second, fad_order,
                    line_to_surface_evaluation_data->get_surface_normal_strategy());
          }
        }
        else
        {
          // The element is not part of this condition, i.e. it will not be used for the
          // calculation of averaged normals. This allows for 'sharp' corners.
        }
      }
    }
  }

  // Setup the geometry data for the surface patch.
  line_to_surface_evaluation_data->setup(discret, face_elements_needed);
}

/**
 *
 */
void BeamInteraction::BeamToSolidConditionSurface::set_state(
    const std::shared_ptr<const Core::FE::Discretization>& discret,
    const std::shared_ptr<const Solid::ModelEvaluator::BeamInteractionDataState>&
        beaminteraction_data_state)
{
  // For contact we reset the evaluation data in each iteration (we don't call clear() here, since
  // we want to keep the contact pairs).
  if (is_contact())
  {
    auto line_to_other_evaluation_data =
        std::dynamic_pointer_cast<GeometryPair::LineTo3DEvaluationData>(geometry_evaluation_data_);
    line_to_other_evaluation_data->reset_tracker();
  }

  // Cast the geometry evaluation data to the correct type.
  auto line_to_surface_evaluation_data =
      std::dynamic_pointer_cast<GeometryPair::LineToSurfaceEvaluationData>(
          geometry_evaluation_data_);
  if (line_to_surface_evaluation_data == nullptr)
    FOUR_C_THROW("Could not cast to GeometryPair::LineToSurfaceEvaluationData.");

  // Setup the geometry data for the surface patch.
  line_to_surface_evaluation_data->set_state(beaminteraction_data_state->get_dis_col_np());
}

/**
 *
 */
std::shared_ptr<BeamInteraction::BeamContactPair>
BeamInteraction::BeamToSolidConditionSurface::create_contact_pair_internal(
    const std::vector<Core::Elements::Element const*>& ele_ptrs)
{
  using namespace GeometryPair;

  const auto* beam_element = dynamic_cast<const Discret::Elements::Beam3Base*>(ele_ptrs[0]);
  const bool beam_is_hermite = beam_element->hermite_centerline_interpolation();

  const auto& core_element = surface_ids_[ele_ptrs[1]->id()];
  const auto shape = core_element->shape();

  auto line_to_surface_evaluation_data =
      std::dynamic_pointer_cast<GeometryPair::LineToSurfaceEvaluationData>(
          geometry_evaluation_data_);
  if (line_to_surface_evaluation_data == nullptr)
    FOUR_C_THROW("Could not cast to GeometryPair::LineToSurfaceEvaluationData.");
  auto surface_normal_strategy = line_to_surface_evaluation_data->get_surface_normal_strategy();

  if (is_mesh_tying())
  {
    // Create beam-to-surface pairs for mesh tying.
    auto beam_to_surface_params =
        std::dynamic_pointer_cast<const BeamToSolidSurfaceMeshtyingParams>(beam_to_solid_params_);

    Inpar::BeamToSolid::BeamToSolidSurfaceCoupling coupling_type =
        beam_to_surface_params->get_coupling_type();

    Inpar::BeamToSolid::BeamToSolidContactDiscretization coupling_discretization =
        beam_to_surface_params->get_contact_discretization();

    bool rotational_coupling = beam_to_surface_params->get_is_rotational_coupling();

    switch (coupling_discretization)
    {
      case Inpar::BeamToSolid::BeamToSolidContactDiscretization::gauss_point_to_segment:
      {
        if (rotational_coupling)
        {
          FOUR_C_THROW(
              "Beam-to-solid surface coupling with a Gauss-point-to-segment approach is not "
              "implemented for rotational coupling");
        }

        switch (coupling_type)
        {
          case Inpar::BeamToSolid::BeamToSolidSurfaceCoupling::
              reference_configuration_forced_to_zero:
          case Inpar::BeamToSolid::BeamToSolidSurfaceCoupling::displacement:
          {
            switch (shape)
            {
              case Core::FE::CellType::tri3:
                return std::make_shared<
                    BeamToSolidSurfaceMeshtyingPairGaussPoint<t_hermite, t_tri3>>();
              case Core::FE::CellType::tri6:
                return std::make_shared<
                    BeamToSolidSurfaceMeshtyingPairGaussPoint<t_hermite, t_tri6>>();
              case Core::FE::CellType::quad4:
                return std::make_shared<
                    BeamToSolidSurfaceMeshtyingPairGaussPoint<t_hermite, t_quad4>>();
              case Core::FE::CellType::quad8:
                return std::make_shared<
                    BeamToSolidSurfaceMeshtyingPairGaussPoint<t_hermite, t_quad8>>();
              case Core::FE::CellType::quad9:
                return std::make_shared<
                    BeamToSolidSurfaceMeshtyingPairGaussPoint<t_hermite, t_quad9>>();
              case Core::FE::CellType::nurbs9:
                return std::make_shared<
                    BeamToSolidSurfaceMeshtyingPairGaussPoint<t_hermite, t_nurbs9>>();
              default:
                FOUR_C_THROW("Wrong element type for surface element.");
            }
            break;
          }

          case Inpar::BeamToSolid::BeamToSolidSurfaceCoupling::
              reference_configuration_forced_to_zero_fad:
          case Inpar::BeamToSolid::BeamToSolidSurfaceCoupling::displacement_fad:
          case Inpar::BeamToSolid::BeamToSolidSurfaceCoupling::consistent_fad:
          {
            if (surface_normal_strategy == GeometryPair::SurfaceNormals::standard)
            {
              switch (shape)
              {
                case Core::FE::CellType::tri3:
                  return std::make_shared<BeamToSolidSurfaceMeshtyingPairGaussPointFAD<
                      line_to_surface_patch_scalar_type, t_hermite, t_tri3>>();
                case Core::FE::CellType::tri6:
                  return std::make_shared<BeamToSolidSurfaceMeshtyingPairGaussPointFAD<
                      line_to_surface_patch_scalar_type, t_hermite, t_tri6>>();
                case Core::FE::CellType::quad4:
                  return std::make_shared<BeamToSolidSurfaceMeshtyingPairGaussPointFAD<
                      line_to_surface_patch_scalar_type, t_hermite, t_quad4>>();
                case Core::FE::CellType::quad8:
                  return std::make_shared<BeamToSolidSurfaceMeshtyingPairGaussPointFAD<
                      line_to_surface_patch_scalar_type, t_hermite, t_quad8>>();
                case Core::FE::CellType::quad9:
                  return std::make_shared<BeamToSolidSurfaceMeshtyingPairGaussPointFAD<
                      line_to_surface_patch_scalar_type, t_hermite, t_quad9>>();
                case Core::FE::CellType::nurbs9:
                  return std::make_shared<BeamToSolidSurfaceMeshtyingPairGaussPointFAD<
                      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_nurbs9>, t_hermite,
                      t_nurbs9>>();
                default:
                  FOUR_C_THROW("Wrong element type for surface element.");
              }
            }
            else if (surface_normal_strategy == GeometryPair::SurfaceNormals::extended_volume)
            {
              switch (shape)
              {
                case Core::FE::CellType::quad4:
                  return std::make_shared<BeamToSolidSurfaceMeshtyingPairGaussPointFAD<
                      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex8>, t_hermite,
                      t_quad4>>();
                case Core::FE::CellType::quad8:
                  return std::make_shared<BeamToSolidSurfaceMeshtyingPairGaussPointFAD<
                      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex20>, t_hermite,
                      t_quad8>>();
                case Core::FE::CellType::quad9:
                  return std::make_shared<BeamToSolidSurfaceMeshtyingPairGaussPointFAD<
                      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex27>, t_hermite,
                      t_quad9>>();
                default:
                  FOUR_C_THROW("Wrong element type for surface element.");
              }
            }
            FOUR_C_THROW("Unknown surface normal strategy.");
          }

          default:
            FOUR_C_THROW("Wrong coupling type.");
        }
        break;
      }
      case Inpar::BeamToSolid::BeamToSolidContactDiscretization::mortar:
      {
        Inpar::BeamToSolid::BeamToSolidMortarShapefunctions mortar_shapefunction =
            beam_to_surface_params->get_mortar_shape_function_type();

        switch (coupling_type)
        {
          case Inpar::BeamToSolid::BeamToSolidSurfaceCoupling::
              reference_configuration_forced_to_zero:
          case Inpar::BeamToSolid::BeamToSolidSurfaceCoupling::displacement:
            return beam_to_solid_surface_meshtying_pair_mortar_factory(shape, mortar_shapefunction);
          case Inpar::BeamToSolid::BeamToSolidSurfaceCoupling::
              reference_configuration_forced_to_zero_fad:
          case Inpar::BeamToSolid::BeamToSolidSurfaceCoupling::displacement_fad:
          case Inpar::BeamToSolid::BeamToSolidSurfaceCoupling::consistent_fad:
            return beam_to_solid_surface_meshtying_pair_mortar_fad_factory(
                shape, mortar_shapefunction, rotational_coupling, surface_normal_strategy);
          default:
            FOUR_C_THROW("Wrong coupling type.");
        }
        break;
      }
      default:
        FOUR_C_THROW("Wrong coupling discretization.");
    }
  }
  else
  {
    // Create beam-to-surface pairs for contact.
    const auto beam_to_surface_contact_params =
        std::dynamic_pointer_cast<const BeamToSolidSurfaceContactParams>(beam_to_solid_params_);

    Inpar::BeamToSolid::BeamToSolidContactDiscretization contact_discretization =
        beam_to_surface_contact_params->get_contact_discretization();

    if (beam_is_hermite)
    {
      switch (contact_discretization)
      {
        case Inpar::BeamToSolid::BeamToSolidContactDiscretization::gauss_point_to_segment:
        {
          Inpar::BeamToSolid::BeamToSolidSurfaceContact contact_type =
              beam_to_surface_contact_params->get_contact_type();

          switch (contact_type)
          {
            case Inpar::BeamToSolid::BeamToSolidSurfaceContact::gap_variation:
              switch (shape)
              {
                case Core::FE::CellType::tri3:
                  return std::make_shared<BeamToSolidSurfaceContactPairGapVariation<
                      line_to_surface_patch_scalar_type_1st_order, t_hermite, t_tri3>>();
                case Core::FE::CellType::tri6:
                  return std::make_shared<BeamToSolidSurfaceContactPairGapVariation<
                      line_to_surface_patch_scalar_type_1st_order, t_hermite, t_tri6>>();
                case Core::FE::CellType::quad4:
                  return std::make_shared<BeamToSolidSurfaceContactPairGapVariation<
                      line_to_surface_patch_scalar_type_1st_order, t_hermite, t_quad4>>();
                case Core::FE::CellType::quad8:
                  return std::make_shared<BeamToSolidSurfaceContactPairGapVariation<
                      line_to_surface_patch_scalar_type_1st_order, t_hermite, t_quad8>>();
                case Core::FE::CellType::quad9:
                  return std::make_shared<BeamToSolidSurfaceContactPairGapVariation<
                      line_to_surface_patch_scalar_type_1st_order, t_hermite, t_quad9>>();
                case Core::FE::CellType::nurbs9:
                  return std::make_shared<BeamToSolidSurfaceContactPairGapVariation<
                      line_to_surface_patch_scalar_type_fixed_size_1st_order<t_hermite, t_nurbs9>,
                      t_hermite, t_nurbs9>>();
                default:
                  FOUR_C_THROW("Wrong element type for surface element.");
              }
              break;
            case Inpar::BeamToSolid::BeamToSolidSurfaceContact::potential:
              switch (shape)
              {
                case Core::FE::CellType::tri3:
                  return std::make_shared<BeamToSolidSurfaceContactPairPotential<
                      line_to_surface_patch_scalar_type, t_hermite, t_tri3>>();
                case Core::FE::CellType::tri6:
                  return std::make_shared<BeamToSolidSurfaceContactPairPotential<
                      line_to_surface_patch_scalar_type, t_hermite, t_tri6>>();
                case Core::FE::CellType::quad4:
                  return std::make_shared<BeamToSolidSurfaceContactPairPotential<
                      line_to_surface_patch_scalar_type, t_hermite, t_quad4>>();
                case Core::FE::CellType::quad8:
                  return std::make_shared<BeamToSolidSurfaceContactPairPotential<
                      line_to_surface_patch_scalar_type, t_hermite, t_quad8>>();
                case Core::FE::CellType::quad9:
                  return std::make_shared<BeamToSolidSurfaceContactPairPotential<
                      line_to_surface_patch_scalar_type, t_hermite, t_quad9>>();
                case Core::FE::CellType::nurbs9:
                  return std::make_shared<BeamToSolidSurfaceContactPairPotential<
                      line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_nurbs9>, t_hermite,
                      t_nurbs9>>();
                default:
                  FOUR_C_THROW("Wrong element type for surface element.");
              }
              break;
            default:
              FOUR_C_THROW("Wrong contact type.");
          }
          break;
        }
        case Inpar::BeamToSolid::BeamToSolidContactDiscretization::mortar:
        {
          return beam_to_solid_surface_contact_pair_mortar_factory(
              *beam_to_surface_contact_params, shape, beam_is_hermite);
        }
        default:
          FOUR_C_THROW("Wrong contact discretization.");
      }
    }
    else
    {
      switch (contact_discretization)
      {
        case Inpar::BeamToSolid::BeamToSolidContactDiscretization::gauss_point_to_segment:
        {
          Inpar::BeamToSolid::BeamToSolidSurfaceContact contact_type =
              beam_to_surface_contact_params->get_contact_type();

          switch (contact_type)
          {
            case Inpar::BeamToSolid::BeamToSolidSurfaceContact::gap_variation:
              switch (shape)
              {
                case Core::FE::CellType::tri3:
                  return std::make_shared<BeamToSolidSurfaceContactPairGapVariation<
                      line_to_surface_patch_scalar_type_1st_order, t_line2, t_tri3>>();
                case Core::FE::CellType::tri6:
                  return std::make_shared<BeamToSolidSurfaceContactPairGapVariation<
                      line_to_surface_patch_scalar_type_1st_order, t_line2, t_tri6>>();
                case Core::FE::CellType::quad4:
                  return std::make_shared<BeamToSolidSurfaceContactPairGapVariation<
                      line_to_surface_patch_scalar_type_1st_order, t_line2, t_quad4>>();
                case Core::FE::CellType::quad8:
                  return std::make_shared<BeamToSolidSurfaceContactPairGapVariation<
                      line_to_surface_patch_scalar_type_1st_order, t_line2, t_quad8>>();
                case Core::FE::CellType::quad9:
                  return std::make_shared<BeamToSolidSurfaceContactPairGapVariation<
                      line_to_surface_patch_scalar_type_1st_order, t_line2, t_quad9>>();
                case Core::FE::CellType::nurbs9:
                  return std::make_shared<BeamToSolidSurfaceContactPairGapVariation<
                      line_to_surface_patch_scalar_type_fixed_size_1st_order<t_line2, t_nurbs9>,
                      t_line2, t_nurbs9>>();
                default:
                  FOUR_C_THROW("Wrong element type for surface element.");
              }
              break;
            case Inpar::BeamToSolid::BeamToSolidSurfaceContact::potential:
              switch (shape)
              {
                case Core::FE::CellType::tri3:
                  return std::make_shared<BeamToSolidSurfaceContactPairPotential<
                      line_to_surface_patch_scalar_type, t_line2, t_tri3>>();
                case Core::FE::CellType::tri6:
                  return std::make_shared<BeamToSolidSurfaceContactPairPotential<
                      line_to_surface_patch_scalar_type, t_line2, t_tri6>>();
                case Core::FE::CellType::quad4:
                  return std::make_shared<BeamToSolidSurfaceContactPairPotential<
                      line_to_surface_patch_scalar_type, t_line2, t_quad4>>();
                case Core::FE::CellType::quad8:
                  return std::make_shared<BeamToSolidSurfaceContactPairPotential<
                      line_to_surface_patch_scalar_type, t_line2, t_quad8>>();
                case Core::FE::CellType::quad9:
                  return std::make_shared<BeamToSolidSurfaceContactPairPotential<
                      line_to_surface_patch_scalar_type, t_line2, t_quad9>>();
                case Core::FE::CellType::nurbs9:
                  return std::make_shared<BeamToSolidSurfaceContactPairPotential<
                      line_to_surface_patch_scalar_type_fixed_size<t_line2, t_nurbs9>, t_line2,
                      t_nurbs9>>();
                default:
                  FOUR_C_THROW("Wrong element type for surface element.");
              }
              break;
            default:
              FOUR_C_THROW("Wrong contact type.");
          }
          break;
        }
        default:
          FOUR_C_THROW("Wrong contact discretization.");
      }
    }
  }
  std23::unreachable();
}

FOUR_C_NAMESPACE_CLOSE
