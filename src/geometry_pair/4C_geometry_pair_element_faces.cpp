// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_geometry_pair_element_faces.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_input.hpp"
#include "4C_geometry_pair_scalar_types.hpp"
#include "4C_utils_fad.hpp"

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
template <typename Surface, typename ScalarType>
void GeometryPair::FaceElementTemplate<Surface, ScalarType>::setup(
    const std::shared_ptr<const Core::FE::Discretization>& discret,
    const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>& face_elements)
{
  // Get the DOF GIDs of this face.
  patch_dof_gid_.clear();
  std::vector<int> lmrowowner;
  std::vector<int> lmstride;
  this->get_element()->location_vector(*discret, this->patch_dof_gid_, lmrowowner, lmstride);

  // Set the reference position from the nodes connected to this face.
  // At the moment we need to get the structure discretization at this point since the beam
  // interaction discretization is a copy - without the nurbs information
  face_reference_position_ =
      GeometryPair::InitializeElementData<Surface, double>::initialize(this->core_element_);
  const Core::Nodes::Node* const* nodes = core_element_->nodes();
  for (unsigned int i_node = 0; i_node < Surface::n_nodes_; i_node++)
    for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
      face_reference_position_.element_position_(i_node * 3 + i_dim) = nodes[i_node]->x()[i_dim];
}

/**
 *
 */
template <typename Surface, typename ScalarType>
void GeometryPair::FaceElementTemplate<Surface, ScalarType>::set_state(
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement,
    const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>& face_elements)
{
  // Get all displacements for the current face / patch.
  std::vector<double> patch_displacement = Core::FE::extract_values(*displacement, patch_dof_gid_);

  // Create the full length FAD types.
  face_position_ =
      GeometryPair::InitializeElementData<Surface, ScalarType>::initialize(this->core_element_);
  const unsigned int n_patch_dof = patch_dof_gid_.size();
  std::vector<ScalarType> patch_displacement_fad(n_patch_dof);
  for (unsigned int i_dof = 0; i_dof < n_patch_dof; i_dof++)
  {
    patch_displacement_fad[i_dof] =
        Core::FADUtils::HigherOrderFadValue<ScalarType>::apply(n_patch_dof + n_dof_other_element_,
            n_dof_other_element_ + i_dof, patch_displacement[i_dof]);
    if (i_dof < Surface::n_dof_)
      face_position_.element_position_(i_dof) =
          face_reference_position_.element_position_(i_dof) + patch_displacement_fad[i_dof];
  }
}

/**
 *
 */
template <typename Surface, typename ScalarType>
void GeometryPair::FaceElementTemplate<Surface, ScalarType>::evaluate_face_position_double(
    const Core::LinAlg::Matrix<2, 1, double>& xi, Core::LinAlg::Matrix<3, 1, double>& r,
    bool reference) const
{
  if (reference)
  {
    evaluate_position<Surface>(xi, face_reference_position_, r);
  }
  else
  {
    Core::LinAlg::Matrix<3, 1, ScalarType> r_ad;
    evaluate_position<Surface>(xi, face_position_, r_ad);
    r = Core::FADUtils::cast_to_double(r_ad);
  }
}

/**
 *
 */
template <typename Surface, typename ScalarType>
void GeometryPair::FaceElementTemplate<Surface, ScalarType>::evaluate_face_normal_double(
    const Core::LinAlg::Matrix<2, 1, double>& xi, Core::LinAlg::Matrix<3, 1, double>& n,
    const bool reference, const bool averaged_normal) const
{
  if (averaged_normal)
  {
    n.put_scalar(0.0);
    return;
  }
  else
  {
    // Calculate the normals on the face geometry.
    auto get_position_double = [&]()
    {
      if (reference)
        return ElementDataToDouble<Surface>::to_double(face_reference_position_);
      else
        return ElementDataToDouble<Surface>::to_double(face_position_);
    };

    const auto position_double = get_position_double();
    evaluate_face_normal<Surface>(xi, position_double, n);
  }
}


/**
 *
 */
template <typename Surface, typename ScalarType>
void GeometryPair::FaceElementPatchTemplate<Surface, ScalarType>::setup(
    const std::shared_ptr<const Core::FE::Discretization>& discret,
    const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>& face_elements)
{
  // Call setup of the base class.
  base_class::setup(discret, face_elements);

  // We need to get a UID for the surface element. If the surface is a face element (at the moment
  // the only supported case), we simply take the GID of the parent element.
  const auto* face_element = dynamic_cast<const Core::Elements::FaceElement*>(this->core_element_);
  if (face_element == nullptr)
    FOUR_C_THROW("For FaceElementPatchTemplate the surface must be a Core::Elements::FaceElement");
  const int element_uid = face_element->parent_element_id();

  // Initialize class variables.
  connected_faces_.clear();

  // Initialize variables for this method.
  std::vector<int> my_node_gid, other_faces_node_gid;
  my_node_gid.clear();
  my_node_gid.reserve(this->core_element_->num_node());
  other_faces_node_gid.clear();

  // Temporary variables.
  std::vector<int> temp_node_dof_gid(3);

  // First add the node GIDs of this face.
  for (int i_node = 0; i_node < this->core_element_->num_node(); i_node++)
  {
    const Core::Nodes::Node* my_node = this->core_element_->nodes()[i_node];
    my_node_gid.push_back(my_node->id());
    discret->dof(my_node, 0, temp_node_dof_gid);
  }

  // Add the node GIDs of the connected faces.
  ConnectedFace temp_connected_face;
  for (int i_node = 0; i_node < this->core_element_->num_node(); i_node++)
  {
    // Loop over all elements connected to a node of this face.
    const Core::Nodes::Node* node = this->core_element_->nodes()[i_node];
    for (auto element : node->adjacent_elements())
    {
      auto element_id = element.global_id();

      if (element_id == element_uid) continue;

      // Check if the element was already searched for.
      if (connected_faces_.find(element_id) == connected_faces_.end())
      {
        temp_connected_face.node_lid_map_.clear();
        temp_connected_face.my_node_patch_lid_.clear();

        // Check if the element is part of the surface condition.
        auto find_in_faces = face_elements.find(element_id);
        if (find_in_faces != face_elements.end())
        {
          // Add the node GIDs of this element.
          for (int i_node_connected_element = 0;
              i_node_connected_element < find_in_faces->second->get_element()->num_node();
              i_node_connected_element++)
          {
            const Core::Nodes::Node* other_node =
                find_in_faces->second->get_element()->nodes()[i_node_connected_element];
            const int other_node_id = other_node->id();

            // Check if the other node is part of this face element.
            auto it = std::find(my_node_gid.begin(), my_node_gid.end(), other_node_id);
            if (it == my_node_gid.end())
            {
              // The other node is not part of this face element. Check if this other node was
              // already processed for this patch.
              auto it_other = std::find(
                  other_faces_node_gid.begin(), other_faces_node_gid.end(), other_node_id);
              if (it_other == other_faces_node_gid.end())
              {
                // This node was not processed yet, so add it to other_faces_node_gid.
                other_faces_node_gid.push_back(other_node_id);
                discret->dof(other_node, 0, temp_node_dof_gid);
                for (const auto& value : temp_node_dof_gid) this->patch_dof_gid_.push_back(value);

                // Add the patch id of this other node to the face element tracker.
                temp_connected_face.my_node_patch_lid_.push_back(
                    my_node_gid.size() + other_faces_node_gid.size() - 1);
              }
              else
              {
                // Get the patch index of the other node and add it to the face element tracker.
                const int index_other_node = std::distance(other_faces_node_gid.begin(), it_other);
                temp_connected_face.my_node_patch_lid_.push_back(
                    my_node_gid.size() + index_other_node);
              }
            }
            else
            {
              // The node is part of this face element, add to the map entry.
              const int index_my_node = std::distance(my_node_gid.begin(), it);
              temp_connected_face.node_lid_map_[i_node_connected_element] = index_my_node;
              temp_connected_face.my_node_patch_lid_.push_back(index_my_node);
            }
          }

          // Add this element to the already searched connected elements.
          connected_faces_[element_id] = temp_connected_face;
        }
      }
    }
  }

  // If we only want to calculate dependencies on the face DOF and not the patch, we do not need the
  // GID of the connected faces in the GID vector of this face. The connectivity to the other patch
  // faces is still required for the calculation of the averaged reference normals.
  if (not evaluate_current_normals_)
    this->patch_dof_gid_.resize(3 * this->core_element_->num_node());
}


/**
 *
 */
template <typename Surface, typename ScalarType>
void GeometryPair::FaceElementPatchTemplate<Surface, ScalarType>::set_state(
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement,
    const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>& face_elements)
{
  // Get all displacements for the current face / patch.
  std::vector<double> patch_displacement =
      Core::FE::extract_values(*displacement, this->patch_dof_gid_);

  // Create the full length FAD types.
  this->face_position_ =
      GeometryPair::InitializeElementData<Surface, ScalarType>::initialize(this->core_element_);
  const unsigned int n_patch_dof = this->patch_dof_gid_.size();
  std::vector<ScalarType> patch_displacement_fad(n_patch_dof);
  for (unsigned int i_dof = 0; i_dof < n_patch_dof; i_dof++)
  {
    patch_displacement_fad[i_dof] = Core::FADUtils::HigherOrderFadValue<ScalarType>::apply(
        n_patch_dof + this->n_dof_other_element_, this->n_dof_other_element_ + i_dof,
        patch_displacement[i_dof]);
    if (i_dof < Surface::n_dof_)
      this->face_position_.element_position_(i_dof) =
          this->face_reference_position_.element_position_(i_dof) + patch_displacement_fad[i_dof];
  }

  if (evaluate_current_normals_)
  {
    // Parameter coordinates corresponding to LIDs of nodes.
    Core::LinAlg::Matrix<3, 1, double> xi(Core::LinAlg::Initialization::zero);
    Core::LinAlg::SerialDenseMatrix nodal_coordinates =
        Core::FE::get_ele_node_numbering_nodes_paramspace(Surface::discretization_);

    // Loop over the connected faces and evaluate their nodal normals.
    Core::LinAlg::Matrix<Surface::n_nodes_, 1, Core::LinAlg::Matrix<3, 1, ScalarType>> normals;
    Core::LinAlg::Matrix<3, 1, ScalarType> temp_normal;
    for (unsigned int i_node = 0; i_node < Surface::n_nodes_; i_node++)
    {
      for (unsigned int i_dim = 0; i_dim < 2; i_dim++) xi(i_dim) = nodal_coordinates(i_dim, i_node);
      evaluate_face_normal<Surface>(xi, this->face_position_, normals(i_node));
    }
    for (const auto& value : connected_faces_)
    {
      // Get the connected face element.
      const std::shared_ptr<const my_type>& face_element =
          std::dynamic_pointer_cast<const my_type>(face_elements.at(value.first));

      // Setup an element data container for the other element, but with the FAD type and ordering
      // for this patch
      auto q_other_face =
          InitializeElementData<Surface, ScalarType>::initialize(face_element->get_element());
      for (unsigned int i_node = 0; i_node < Surface::n_nodes_; i_node++)
      {
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        {
          // We need to add the reference position of the other element.
          q_other_face.element_position_(i_dim + 3 * i_node) =
              face_element->face_reference_position_.element_position_(i_dim + 3 * i_node) +
              patch_displacement_fad[i_dim + 3 * value.second.my_node_patch_lid_[i_node]];
        }
      }

      // Evaluate the normals at the shared nodes.
      for (const auto& node_map_iterator : value.second.node_lid_map_)
      {
        for (unsigned int i_dim = 0; i_dim < 2; i_dim++)
          xi(i_dim) = nodal_coordinates(i_dim, node_map_iterator.first);
        evaluate_face_normal<Surface>(xi, q_other_face, temp_normal);
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          normals(node_map_iterator.second)(i_dim) += temp_normal(i_dim);
      }
    }
    average_nodal_normals(normals, this->face_position_.nodal_normals_);
  }
}

/**
 *
 */
template <typename Surface, typename ScalarType>
void GeometryPair::FaceElementPatchTemplate<Surface,
    ScalarType>::calculate_averaged_reference_normals(const std::unordered_map<int,
    std::shared_ptr<GeometryPair::FaceElement>>& face_elements)
{
  // Parameter coordinates corresponding to LIDs of nodes.
  Core::LinAlg::Matrix<2, 1, double> xi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::SerialDenseMatrix nodal_coordinates =
      Core::FE::get_ele_node_numbering_nodes_paramspace(Surface::discretization_);

  // Loop over the connected faces and evaluate their nodal normals.
  Core::LinAlg::Matrix<Surface::n_nodes_, 1, Core::LinAlg::Matrix<3, 1, double>> normals;
  Core::LinAlg::Matrix<3, 1, double> temp_normal;
  for (unsigned int i_node = 0; i_node < Surface::n_nodes_; i_node++)
  {
    for (unsigned int i_dim = 0; i_dim < 2; i_dim++) xi(i_dim) = nodal_coordinates(i_dim, i_node);
    evaluate_face_normal<Surface>(xi, this->face_reference_position_, normals(i_node));
  }
  for (const auto& value : connected_faces_)
  {
    // Get the connected face element.
    const std::shared_ptr<const my_type>& face_element =
        std::dynamic_pointer_cast<const my_type>(face_elements.at(value.first));

    // Evaluate the normals at the shared nodes.
    for (const auto& node_map_iterator : value.second.node_lid_map_)
    {
      for (unsigned int i_dim = 0; i_dim < 2; i_dim++)
        xi(i_dim) = nodal_coordinates(i_dim, node_map_iterator.first);
      evaluate_face_normal<Surface>(xi, face_element->face_reference_position_, temp_normal);
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        normals(node_map_iterator.second)(i_dim) += temp_normal(i_dim);
    }
  }
  average_nodal_normals(normals, this->face_reference_position_.nodal_normals_);
}

/**
 *
 */
template <typename Surface, typename ScalarType>
void GeometryPair::FaceElementPatchTemplate<Surface, ScalarType>::evaluate_face_normal_double(
    const Core::LinAlg::Matrix<2, 1, double>& xi, Core::LinAlg::Matrix<3, 1, double>& n,
    const bool reference, const bool averaged_normal) const
{
  if (averaged_normal)
  {
    if (reference)
    {
      evaluate_surface_normal<Surface>(xi, this->face_reference_position_, n);
    }
    else
    {
      if (evaluate_current_normals_)
      {
        Core::LinAlg::Matrix<3, 1, ScalarType> n_ad;
        evaluate_surface_normal<Surface>(xi, this->face_position_, n_ad);
        n = Core::FADUtils::cast_to_double(n_ad);
      }
      else
      {
        n.put_scalar(0.0);
      }
    }
  }
  else
  {
    // If no averaged normals should be calculated we can call the base method here.
    base_class::evaluate_face_normal_double(xi, n, reference, false);
  }
}

/**
 *
 */
template <typename Surface, typename ScalarType>
template <typename T>
void GeometryPair::FaceElementPatchTemplate<Surface, ScalarType>::average_nodal_normals(
    Core::LinAlg::Matrix<Surface::n_nodes_, 1, Core::LinAlg::Matrix<3, 1, T>>& normals,
    Core::LinAlg::Matrix<3 * Surface::n_nodes_, 1, T>& averaged_normals) const
{
  averaged_normals.put_scalar(0.0);
  for (unsigned int i_node = 0; i_node < Surface::n_nodes_; i_node++)
  {
    normals(i_node).scale(1.0 / Core::FADUtils::vector_norm(normals(i_node)));
    for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
    {
      averaged_normals(i_dim + 3 * i_node) = normals(i_node)(i_dim);
    }
  }
}

/**
 *
 */
template <typename Surface, typename ScalarType, typename Volume>
void GeometryPair::FaceElementTemplateExtendedVolume<Surface, ScalarType, Volume>::setup(
    const std::shared_ptr<const Core::FE::Discretization>& discret,
    const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>& face_elements)
{
  const auto* face_element = dynamic_cast<const Core::Elements::FaceElement*>(this->core_element_);
  if (face_element == nullptr)
    FOUR_C_THROW("For ExtendedVolume coupling the surface must be a Core::Elements::FaceElement");

  // Get the DOF GIDs of this face and volume element.
  this->patch_dof_gid_.clear();
  std::vector<int> surface_gid;
  std::vector<int> lmrowowner;
  std::vector<int> lmstride;
  face_element->location_vector(*discret, surface_gid, lmrowowner, lmstride);
  face_element->parent_element()->location_vector(
      *discret, this->patch_dof_gid_, lmrowowner, lmstride);

  // Safety checks.
  if (this->patch_dof_gid_.size() != Volume::n_dof_ or surface_gid.size() != Surface::n_dof_)
    FOUR_C_THROW("Mismatch in GID sizes!");

  // Calculate the face to volume DOF map.
  for (unsigned i_dof_surf = 0; i_dof_surf < Surface::n_dof_; i_dof_surf++)
  {
    auto dof_iterator =
        find(this->patch_dof_gid_.begin(), this->patch_dof_gid_.end(), surface_gid[i_dof_surf]);
    if (dof_iterator != this->patch_dof_gid_.end())
    {
      // Calculating the index.
      surface_dof_lid_map_(i_dof_surf) = dof_iterator - this->patch_dof_gid_.begin();
    }
    else
      FOUR_C_THROW("Could not find the surface DOF {} in the volume DOFs", surface_gid[i_dof_surf]);
  }

  // Set the reference position.
  volume_reference_position_ =
      GeometryPair::InitializeElementData<Volume, double>::initialize(nullptr);
  this->face_reference_position_ =
      GeometryPair::InitializeElementData<Surface, double>::initialize(nullptr);
  const Core::Nodes::Node* const* nodes = face_element->parent_element()->nodes();
  for (unsigned int i_node = 0; i_node < Volume::n_nodes_; i_node++)
    for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
      volume_reference_position_.element_position_(i_node * 3 + i_dim) = nodes[i_node]->x()[i_dim];
  for (unsigned int i_dof_surf = 0; i_dof_surf < Surface::n_dof_; i_dof_surf++)
    this->face_reference_position_.element_position_(i_dof_surf) =
        volume_reference_position_.element_position_(surface_dof_lid_map_(i_dof_surf));

  // Surface node to volume node map.
  Core::LinAlg::Matrix<Surface::n_nodes_, 1, int> surface_node_lid_map;
  for (unsigned i_node_surf = 0; i_node_surf < Surface::n_nodes_; i_node_surf++)
    surface_node_lid_map(i_node_surf) = surface_dof_lid_map_(i_node_surf * 3) / 3;

  if (surface_node_lid_map(0) == 0 and surface_node_lid_map(1) == 4 and
      surface_node_lid_map(2) == 7 and surface_node_lid_map(3) == 3)
  {
    face_to_volume_coordinate_axis_map_(0) = 2;
    face_to_volume_coordinate_axis_map_(1) = 1;
    face_to_volume_coordinate_axis_factor_(0) = 1;
    face_to_volume_coordinate_axis_factor_(1) = 1;
    third_direction_ = 0;
    third_direction_factor_ = -1;
  }
  else if (surface_node_lid_map(0) == 1 and surface_node_lid_map(1) == 2 and
           surface_node_lid_map(2) == 6 and surface_node_lid_map(3) == 5)
  {
    face_to_volume_coordinate_axis_map_(0) = 1;
    face_to_volume_coordinate_axis_map_(1) = 2;
    face_to_volume_coordinate_axis_factor_(0) = 1;
    face_to_volume_coordinate_axis_factor_(1) = 1;
    third_direction_ = 0;
    third_direction_factor_ = 1;
  }
  else if (surface_node_lid_map(0) == 0 and surface_node_lid_map(1) == 1 and
           surface_node_lid_map(2) == 5 and surface_node_lid_map(3) == 4)
  {
    face_to_volume_coordinate_axis_map_(0) = 0;
    face_to_volume_coordinate_axis_map_(1) = 2;
    face_to_volume_coordinate_axis_factor_(0) = 1;
    face_to_volume_coordinate_axis_factor_(1) = 1;
    third_direction_ = 1;
    third_direction_factor_ = -1;
  }
  else if (surface_node_lid_map(0) == 2 and surface_node_lid_map(1) == 3 and
           surface_node_lid_map(2) == 7 and surface_node_lid_map(3) == 6)
  {
    face_to_volume_coordinate_axis_map_(0) = 0;
    face_to_volume_coordinate_axis_map_(1) = 2;
    face_to_volume_coordinate_axis_factor_(0) = -1;
    face_to_volume_coordinate_axis_factor_(1) = 1;
    third_direction_ = 1;
    third_direction_factor_ = 1;
  }
  else if (surface_node_lid_map(0) == 0 and surface_node_lid_map(1) == 3 and
           surface_node_lid_map(2) == 2 and surface_node_lid_map(3) == 1)
  {
    face_to_volume_coordinate_axis_map_(0) = 1;
    face_to_volume_coordinate_axis_map_(1) = 0;
    face_to_volume_coordinate_axis_factor_(0) = 1;
    face_to_volume_coordinate_axis_factor_(1) = 1;
    third_direction_ = 2;
    third_direction_factor_ = -1;
  }
  else if (surface_node_lid_map(0) == 4 and surface_node_lid_map(1) == 5 and
           surface_node_lid_map(2) == 6 and surface_node_lid_map(3) == 7)
  {
    face_to_volume_coordinate_axis_map_(0) = 0;
    face_to_volume_coordinate_axis_map_(1) = 1;
    face_to_volume_coordinate_axis_factor_(0) = 1;
    face_to_volume_coordinate_axis_factor_(1) = 1;
    third_direction_ = 2;
    third_direction_factor_ = 1;
  }
  else
    FOUR_C_THROW("Could not map face to volume.");

  // Calculate the reference normals.
  calculate_normals(volume_reference_position_, this->face_reference_position_,
      this->face_reference_position_.nodal_normals_);
}


/**
 *
 */
template <typename Surface, typename ScalarType, typename Volume>
void GeometryPair::FaceElementTemplateExtendedVolume<Surface, ScalarType, Volume>::set_state(
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& displacement,
    const std::unordered_map<int, std::shared_ptr<GeometryPair::FaceElement>>& face_elements)
{
  // Get all displacements for the current face / volume.
  std::vector<double> volume_displacement =
      Core::FE::extract_values(*displacement, this->patch_dof_gid_);

  // Create the full length FAD types.
  std::vector<ScalarType> patch_displacement_fad(Volume::n_dof_);

  volume_position_ = GeometryPair::InitializeElementData<Volume, ScalarType>::initialize(nullptr);
  for (unsigned int i_dof = 0; i_dof < Volume::n_dof_; i_dof++)
  {
    volume_position_.element_position_(i_dof) =
        Core::FADUtils::HigherOrderFadValue<ScalarType>::apply(
            Volume::n_dof_ + this->n_dof_other_element_, this->n_dof_other_element_ + i_dof,
            volume_displacement[i_dof] + volume_reference_position_.element_position_(i_dof));
  }
  this->face_position_ =
      GeometryPair::InitializeElementData<Surface, ScalarType>::initialize(nullptr);
  for (unsigned int i_dof = 0; i_dof < Surface::n_dof_; i_dof++)
    this->face_position_.element_position_(i_dof) =
        volume_position_.element_position_(surface_dof_lid_map_(i_dof));

  calculate_normals(volume_position_, this->face_position_, this->face_position_.nodal_normals_);
}

/**
 *
 */
template <typename Surface, typename ScalarType, typename Volume>
template <typename ScalarTypeNormal>
void GeometryPair::FaceElementTemplateExtendedVolume<Surface, ScalarType,
    Volume>::calculate_normals(const GeometryPair::ElementData<Volume, ScalarTypeNormal>&
                                   volume_position,
    const GeometryPair::ElementData<Surface, ScalarTypeNormal>& surface_position,
    Core::LinAlg::Matrix<3 * Surface::n_nodes_, 1, ScalarTypeNormal>& normals) const
{
  // Parameter coordinates corresponding to LIDs of nodes.
  Core::LinAlg::Matrix<2, 1, double> xi_surface(Core::LinAlg::Initialization::zero);
  Core::LinAlg::SerialDenseMatrix nodal_coordinates =
      Core::FE::get_ele_node_numbering_nodes_paramspace(Surface::discretization_);

  // Loop over the faces and evaluate the "normals" at the nodes.
  Core::LinAlg::Matrix<3, 1, double> xi_volume(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, ScalarTypeNormal> r_surface;
  Core::LinAlg::Matrix<3, 1, ScalarTypeNormal> r_volume;
  Core::LinAlg::Matrix<3, 3, ScalarTypeNormal> dr_volume;
  for (unsigned int i_node = 0; i_node < Surface::n_nodes_; i_node++)
  {
    for (unsigned int i_dim = 0; i_dim < 2; i_dim++)
      xi_surface(i_dim) = nodal_coordinates(i_dim, i_node);
    xi_face_to_xi_volume(xi_surface, xi_volume);

    evaluate_position<Surface>(xi_surface, surface_position, r_surface);
    evaluate_position<Volume>(xi_volume, volume_position, r_volume);
    r_volume -= r_surface;
    if (Core::FADUtils::vector_norm(r_volume) > 1e-10)
      FOUR_C_THROW("Nodal positions for face and volume do not match.");

    evaluate_position_derivative1<Volume>(xi_volume, volume_position, dr_volume);
    for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
      normals(3 * i_node + i_dim) = third_direction_factor_ * dr_volume(i_dim, third_direction_);
  }
}

/**
 *
 */
template <typename Surface, typename ScalarType, typename Volume>
void GeometryPair::FaceElementTemplateExtendedVolume<Surface, ScalarType,
    Volume>::evaluate_face_normal_double(const Core::LinAlg::Matrix<2, 1, double>& xi,
    Core::LinAlg::Matrix<3, 1, double>& n, const bool reference, const bool averaged_normal) const
{
  if (averaged_normal)
  {
    if (reference)
    {
      evaluate_surface_normal<Surface>(xi, this->face_reference_position_, n);
    }
    else
    {
      Core::LinAlg::Matrix<3, 1, ScalarType> n_ad;
      evaluate_surface_normal<Surface>(xi, this->face_position_, n_ad);
      n = Core::FADUtils::cast_to_double(n_ad);
    }
  }
  else
  {
    // If no averaged normals should be calculated we can call the base method here.
    base_class::evaluate_face_normal_double(xi, n, reference, false);
  }
}

/**
 *
 */
template <typename Surface, typename ScalarType, typename Volume>
void GeometryPair::FaceElementTemplateExtendedVolume<Surface, ScalarType,
    Volume>::xi_face_to_xi_volume(const Core::LinAlg::Matrix<2, 1, double>& xi_face,
    Core::LinAlg::Matrix<3, 1, double>& xi_volume) const
{
  xi_volume(face_to_volume_coordinate_axis_map_(0)) =
      xi_face(0) * face_to_volume_coordinate_axis_factor_(0);
  xi_volume(face_to_volume_coordinate_axis_map_(1)) =
      xi_face(1) * face_to_volume_coordinate_axis_factor_(1);
  xi_volume(third_direction_) = third_direction_factor_;
}


/**
 *
 */
std::shared_ptr<GeometryPair::FaceElement> GeometryPair::face_element_factory(
    const Core::Elements::Element* core_element, const int fad_order,
    const GeometryPair::SurfaceNormals surface_normal_strategy)
{
  const bool is_fad = fad_order > 0;
  if (not is_fad)
  {
    switch (core_element->shape())
    {
      case Core::FE::CellType::quad4:
        return std::make_shared<
            FaceElementPatchTemplate<t_quad4, line_to_surface_scalar_type<t_hermite, t_quad4>>>(

            core_element, false);
      case Core::FE::CellType::quad8:
        return std::make_shared<
            FaceElementPatchTemplate<t_quad8, line_to_surface_scalar_type<t_hermite, t_quad8>>>(

            core_element, false);
      case Core::FE::CellType::quad9:
        return std::make_shared<
            FaceElementPatchTemplate<t_quad9, line_to_surface_scalar_type<t_hermite, t_quad9>>>(

            core_element, false);
      case Core::FE::CellType::tri3:
        return std::make_shared<
            FaceElementPatchTemplate<t_tri3, line_to_surface_scalar_type<t_hermite, t_tri3>>>(

            core_element, false);
      case Core::FE::CellType::tri6:
        return std::make_shared<
            FaceElementPatchTemplate<t_tri6, line_to_surface_scalar_type<t_hermite, t_tri6>>>(

            core_element, false);
      case Core::FE::CellType::nurbs9:
        return std::make_shared<
            FaceElementTemplate<t_nurbs9, line_to_surface_scalar_type<t_hermite, t_nurbs9>>>(

            core_element);
      default:
        FOUR_C_THROW("Wrong discretization type given.");
    }
  }
  else
  {
    if (surface_normal_strategy == GeometryPair::SurfaceNormals::standard)
    {
      switch (fad_order)
      {
        case 1:
        {
          switch (core_element->shape())
          {
            case Core::FE::CellType::quad4:
              return std::make_shared<
                  FaceElementPatchTemplate<t_quad4, line_to_surface_patch_scalar_type_1st_order>>(
                  core_element, true);
            case Core::FE::CellType::quad8:
              return std::make_shared<
                  FaceElementPatchTemplate<t_quad8, line_to_surface_patch_scalar_type_1st_order>>(
                  core_element, true);
            case Core::FE::CellType::quad9:
              return std::make_shared<
                  FaceElementPatchTemplate<t_quad9, line_to_surface_patch_scalar_type_1st_order>>(
                  core_element, true);
            case Core::FE::CellType::tri3:
              return std::make_shared<
                  FaceElementPatchTemplate<t_tri3, line_to_surface_patch_scalar_type_1st_order>>(

                  core_element, true);
            case Core::FE::CellType::tri6:
              return std::make_shared<
                  FaceElementPatchTemplate<t_tri6, line_to_surface_patch_scalar_type_1st_order>>(

                  core_element, true);
            case Core::FE::CellType::nurbs9:
              return std::make_shared<FaceElementTemplate<t_nurbs9,
                  line_to_surface_patch_scalar_type_fixed_size_1st_order<t_hermite, t_nurbs9>>>(
                  core_element);
            default:
              FOUR_C_THROW("Wrong discretization type given.");
          }
          break;
        }
        case 2:
        {
          switch (core_element->shape())
          {
            case Core::FE::CellType::quad4:
              return std::make_shared<
                  FaceElementPatchTemplate<t_quad4, line_to_surface_patch_scalar_type>>(

                  core_element, true);
            case Core::FE::CellType::quad8:
              return std::make_shared<
                  FaceElementPatchTemplate<t_quad8, line_to_surface_patch_scalar_type>>(

                  core_element, true);
            case Core::FE::CellType::quad9:
              return std::make_shared<
                  FaceElementPatchTemplate<t_quad9, line_to_surface_patch_scalar_type>>(

                  core_element, true);
            case Core::FE::CellType::tri3:
              return std::make_shared<
                  FaceElementPatchTemplate<t_tri3, line_to_surface_patch_scalar_type>>(

                  core_element, true);
            case Core::FE::CellType::tri6:
              return std::make_shared<
                  FaceElementPatchTemplate<t_tri6, line_to_surface_patch_scalar_type>>(

                  core_element, true);
            case Core::FE::CellType::nurbs9:
              return std::make_shared<FaceElementTemplate<t_nurbs9,
                  line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_nurbs9>>>(core_element);
            default:
              FOUR_C_THROW("Wrong discretization type given.");
          }
          break;
        }
        default:
          FOUR_C_THROW("Got unexpected fad order.");
      }
    }
    else if (surface_normal_strategy == GeometryPair::SurfaceNormals::extended_volume)
    {
      switch (core_element->shape())
      {
        case Core::FE::CellType::quad4:
          return std::make_shared<FaceElementTemplateExtendedVolume<t_quad4,
              line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex8>, t_hex8>>(
              core_element);
        case Core::FE::CellType::quad8:
          return std::make_shared<FaceElementTemplateExtendedVolume<t_quad8,
              line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex20>, t_hex20>>(
              core_element);
        case Core::FE::CellType::quad9:
          return std::make_shared<FaceElementTemplateExtendedVolume<t_quad9,
              line_to_surface_patch_scalar_type_fixed_size<t_hermite, t_hex27>, t_hex27>>(
              core_element);
        default:
          FOUR_C_THROW("Got unexpected face type for extended volume coupling.");
      }
    }
    else
      FOUR_C_THROW("Surface normal strategy not recognized.");
  }

  FOUR_C_THROW("Could not create a face element.");
  return nullptr;
}

FOUR_C_NAMESPACE_CLOSE
