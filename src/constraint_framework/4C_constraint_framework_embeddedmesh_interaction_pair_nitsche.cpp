// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_framework_embeddedmesh_interaction_pair_nitsche.hpp"

#include "4C_constraint_framework_embeddedmesh_solid_to_solid_nitsche_manager.hpp"
#include "4C_constraint_framework_embeddedmesh_solid_to_solid_utils.hpp"
#include "4C_cut_boundarycell.hpp"
#include "4C_cut_cutwizard.hpp"
#include "4C_cut_elementhandle.hpp"
#include "4C_cut_sidehandle.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_to_surface.hpp"
#include "4C_geometry_pair_line_to_volume.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_solid_3D_ele_calc_lib_nitsche.hpp"

FOUR_C_NAMESPACE_OPEN


template <typename Interface, typename Background>
Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairNitsche<Interface,
    Background>::SurfaceToBackgroundCouplingPairNitsche(std::shared_ptr<Core::Elements::Element>
                                                            element1,
    Core::Elements::Element* element2, Constraints::EmbeddedMesh::EmbeddedMeshParams& params_ptr,
    std::shared_ptr<Cut::CutWizard>& cutwizard_ptr,
    std::vector<std::shared_ptr<Cut::BoundaryCell>>& boundary_cells)
    : SolidInteractionPair(element1, element2, params_ptr, cutwizard_ptr, boundary_cells)
{
  // Add parameters
  params_ = params_ptr;

  // Initialize the element positions and displacement containers
  ele1pos_ = GeometryPair::InitializeElementData<Interface, double>::initialize(&this->element_1());
  ele2pos_ =
      GeometryPair::InitializeElementData<Background, double>::initialize(&this->element_2());

  ele1pos_current_ =
      GeometryPair::InitializeElementData<Interface, double>::initialize(&this->element_1());
  ele2pos_current_ =
      GeometryPair::InitializeElementData<Background, double>::initialize(&this->element_2());

  ele1dis_ = GeometryPair::InitializeElementData<Interface, double>::initialize(&this->element_1());
  ele2dis_ =
      GeometryPair::InitializeElementData<Background, double>::initialize(&this->element_2());

  // Write the initial position of the elements
  for (int node_ele1 = 0; node_ele1 < element_1().num_point(); node_ele1++)
  {
    // nodal positions
    ele1pos_.element_position_(0 + 3 * node_ele1) = element_1().nodes()[node_ele1]->x()[0];
    ele1pos_.element_position_(1 + 3 * node_ele1) = element_1().nodes()[node_ele1]->x()[1];
    ele1pos_.element_position_(2 + 3 * node_ele1) = element_1().nodes()[node_ele1]->x()[2];
  }

  // For the surface elements, evaluate the normal vectors on the nodes
  Constraints::EmbeddedMesh::evaluate_interface_element_nodal_normals(ele1pos_);

  for (int node_ele2 = 0; node_ele2 < element_2().num_point(); node_ele2++)
  {
    // nodal positions
    ele2pos_.element_position_(0 + 3 * node_ele2) = element_2().nodes()[node_ele2]->x()[0];
    ele2pos_.element_position_(1 + 3 * node_ele2) = element_2().nodes()[node_ele2]->x()[1];
    ele2pos_.element_position_(2 + 3 * node_ele2) = element_2().nodes()[node_ele2]->x()[2];
  }

  // From the gauss rule of the boundary cells related to this pair, set the gauss rule for the
  // interface and background element
  set_gauss_rule_for_interface_and_background();
}


template <typename Interface, typename Background>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairNitsche<Interface,
    Background>::set_current_element_position(Core::FE::Discretization const& discret,
    const Core::LinAlg::Vector<double>& displacement_vector)
{
  std::vector<double> interface_dofvec_timestep = std::vector<double>();
  std::vector<double> background_dofvec_timestep = std::vector<double>();

  Constraints::EmbeddedMesh::get_current_element_displacement(
      discret, &element_1(), displacement_vector, interface_dofvec_timestep);
  Constraints::EmbeddedMesh::get_current_element_displacement(
      discret, &element_2(), displacement_vector, background_dofvec_timestep);

  // Save the displacements of the parent element related to the interface element
  const auto* face_element = dynamic_cast<const Core::Elements::FaceElement*>(&this->element_1());
  if (!face_element) FOUR_C_THROW("Cast to FaceElement failed!");
  Constraints::EmbeddedMesh::get_current_element_displacement(
      discret, face_element->parent_element(), displacement_vector, ele1_parent_dis_);

  // Set the displacements and current position of the first element
  for (int node_ele1 = 0; node_ele1 < element_1().num_point(); node_ele1++)
  {
    // nodal displacements
    ele1dis_.element_position_(0 + 3 * node_ele1) = interface_dofvec_timestep[0 + 3 * node_ele1];
    ele1dis_.element_position_(1 + 3 * node_ele1) = interface_dofvec_timestep[1 + 3 * node_ele1];
    ele1dis_.element_position_(2 + 3 * node_ele1) = interface_dofvec_timestep[2 + 3 * node_ele1];

    ele1pos_current_.element_position_(0 + 3 * node_ele1) =
        element_1().nodes()[node_ele1]->x()[0] + interface_dofvec_timestep[0 + 3 * node_ele1];
    ele1pos_current_.element_position_(1 + 3 * node_ele1) =
        element_1().nodes()[node_ele1]->x()[1] + interface_dofvec_timestep[1 + 3 * node_ele1];
    ele1pos_current_.element_position_(2 + 3 * node_ele1) =
        element_1().nodes()[node_ele1]->x()[2] + interface_dofvec_timestep[2 + 3 * node_ele1];
  }

  // Set the displacements and current position of the second element
  for (int node_ele2 = 0; node_ele2 < element_2().num_point(); node_ele2++)
  {
    // nodal displacements
    ele2dis_.element_position_(0 + 3 * node_ele2) = background_dofvec_timestep[0 + 3 * node_ele2];
    ele2dis_.element_position_(1 + 3 * node_ele2) = background_dofvec_timestep[1 + 3 * node_ele2];
    ele2dis_.element_position_(2 + 3 * node_ele2) = background_dofvec_timestep[2 + 3 * node_ele2];

    ele2pos_current_.element_position_(0 + 3 * node_ele2) =
        element_2().nodes()[node_ele2]->x()[0] + background_dofvec_timestep[0 + 3 * node_ele2];
    ele2pos_current_.element_position_(1 + 3 * node_ele2) =
        element_2().nodes()[node_ele2]->x()[1] + background_dofvec_timestep[1 + 3 * node_ele2];
    ele2pos_current_.element_position_(2 + 3 * node_ele2) =
        element_2().nodes()[node_ele2]->x()[2] + background_dofvec_timestep[2 + 3 * node_ele2];
  }
}


template <typename Interface, typename Background>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairNitsche<Interface,
    Background>::set_gauss_rule_for_interface_and_background()
{
  // Variables before iterating over boundary cells
  Core::LinAlg::Matrix<3, 1> interface_reference_position;
  Core::LinAlg::Matrix<3, 1> interface_position;
  Core::LinAlg::Matrix<3, 1> interface_displacement;
  int current_numpoints = interface_integration_points_.size();

  // Iterate over the boundary cells, get their gauss points and save them
  auto boundary_cells = get_boundary_cells();
  for (auto it_boundarycell = boundary_cells.begin(); it_boundarycell != boundary_cells.end();
      ++it_boundarycell)
  {
    // Check if the shape of the boundary cell is CellType::tri3
    FOUR_C_ASSERT(it_boundarycell->get()->shape() == Core::FE::CellType::tri3,
        "The current implementation only works for boundary cells with shape "
        "Core::FE::CellType::tri3.");

    // Project the gauss points of the boundary cell segment to the interface
    const std::shared_ptr<Core::FE::GaussPoints> gps_boundarycell =
        Constraints::EmbeddedMesh::project_boundary_cell_gauss_rule_on_interface<Interface,
            Core::FE::CellType::tri3>(it_boundarycell->get(), ele1pos_);

    // Save the number of gauss points per boundary cell. The check is done only in
    // the first boundary cell since all the boundary cells have the same cubature
    // degree and have the same shape (tri3)
    if (it_boundarycell == boundary_cells.begin())
      num_gauss_points_boundary_cell_ = gps_boundarycell->num_points();

    // Add the gauss points of the boundary cell to interface_integration_points
    interface_integration_points_.resize(current_numpoints + gps_boundarycell->num_points());

    for (int it_gp = 0; it_gp < gps_boundarycell->num_points(); it_gp++)
    {
      auto& [xi_interface, xi_background, weight] =
          interface_integration_points_[current_numpoints + it_gp];

      // Write the gauss points over the interface
      xi_interface(0) = gps_boundarycell->point(it_gp)[0];
      xi_interface(1) = gps_boundarycell->point(it_gp)[1];

      // Project gauss points on the background element and write them
      GeometryPair::evaluate_position(xi_interface, ele1pos_, interface_position);

      GeometryPair::ProjectionResult temp_projection_result;
      GeometryPair::project_point_to_volume(
          interface_position, ele2pos_, xi_background, temp_projection_result);

      // Write the weight
      weight = gps_boundarycell->weight(it_gp);
    }
    current_numpoints += gps_boundarycell->num_points();
  }
}

template <typename Interface, typename Background>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairNitsche<Interface,
    Background>::evaluate_and_assemble_nitsche_contributions(const Core::FE::Discretization&
                                                                 discret,
    const Constraints::EmbeddedMesh::SolidToSolidNitscheManager* nitsche_manager,
    Core::LinAlg::SparseMatrix& global_penalty_interface,
    Core::LinAlg::SparseMatrix& global_penalty_background,
    Core::LinAlg::SparseMatrix& global_penalty_interface_background,
    Core::LinAlg::SparseMatrix& global_disp_interface_stress_interface,
    Core::LinAlg::SparseMatrix& global_disp_interface_stress_background,
    Core::LinAlg::SparseMatrix& global_disp_background_stress_interface,
    Core::LinAlg::SparseMatrix& global_disp_background_stress_background,
    Core::LinAlg::FEVector<double>& global_constraint, double& nitsche_stabilization_param,
    double& nitsche_average_weight_param)
{
  // Initialize variables for local penalty contributions.
  Core::LinAlg::Matrix<Interface::n_dof_, Interface::n_dof_, double>
      local_stiffness_penalty_interface(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Background::n_dof_, Background::n_dof_, double>
      local_stiffness_penalty_background(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Interface::n_dof_, Background::n_dof_, double>
      local_stiffness_penalty_interface_background(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Interface::n_dof_ + Background::n_dof_, 1, double> local_constraint_penalty(
      Core::LinAlg::Initialization::uninitialized);

  // Initialize variables for local stress contributions.
  Core::LinAlg::Matrix<Interface::n_dof_, Interface::n_dof_, double>
      local_stiffness_disp_interface_stress_interface(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Interface::n_dof_, Background::n_dof_, double>
      local_stiffness_disp_interface_stress_background(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Background::n_dof_, Interface::n_dof_, double>
      local_stiffness_disp_background_stress_interface(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Background::n_dof_, Background::n_dof_, double>
      local_stiffness_disp_background_stress_background(
          Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Interface::n_dof_ + Background::n_dof_, 1, double> local_constraint_stresses(
      Core::LinAlg::Initialization::uninitialized);

  // Evaluate the local penalty contributions of Nitsche method
  evaluate_penalty_contributions_nitsche(local_stiffness_penalty_interface,
      local_stiffness_penalty_background, local_stiffness_penalty_interface_background,
      local_constraint_penalty, nitsche_stabilization_param);

  evaluate_stress_contributions_nitsche(discret, local_stiffness_disp_interface_stress_interface,
      local_stiffness_disp_interface_stress_background,
      local_stiffness_disp_background_stress_interface,
      local_stiffness_disp_background_stress_background, local_constraint_stresses,
      nitsche_average_weight_param);

  // Assemble into global matrices.
  assemble_local_nitsche_contributions<Interface, Background>(this, discret,
      global_penalty_interface, global_penalty_background, global_penalty_interface_background,
      global_disp_interface_stress_interface, global_disp_interface_stress_background,
      global_disp_background_stress_interface, global_disp_background_stress_background,
      global_constraint, local_stiffness_penalty_interface, local_stiffness_penalty_background,
      local_stiffness_penalty_interface_background, local_stiffness_disp_interface_stress_interface,
      local_stiffness_disp_interface_stress_background,
      local_stiffness_disp_background_stress_interface,
      local_stiffness_disp_background_stress_background, local_constraint_penalty,
      local_constraint_stresses);
}


/*!
 * \brief Return the Gauss points of a face element in its parent element
 */
template <int dim>
void interface_element_gp_in_solid(Core::Elements::FaceElement& face_element, const double wgt,
    const double* gp_coord_face, Core::LinAlg::Matrix<dim, 1>& gp_coord_parent,
    Core::LinAlg::Matrix<dim, dim>& derivtrafo)
{
  Core::FE::CollectedGaussPoints intpoints =
      Core::FE::CollectedGaussPoints(1);  // reserve just for 1 entry ...
  intpoints.append(gp_coord_face[0], gp_coord_face[1], 0.0, wgt);

  // get coordinates of gauss point w.r.t. local parent coordinate system
  Core::LinAlg::SerialDenseMatrix temp_gp_coord_parent(1, dim);
  derivtrafo.clear();

  Core::FE::boundary_gp_to_parent_gp<dim>(temp_gp_coord_parent, derivtrafo, intpoints,
      face_element.parent_element()->shape(), face_element.shape(),
      face_element.face_parent_number());

  // coordinates of the current integration point in parent coordinate system
  for (int idim = 0; idim < dim; idim++) gp_coord_parent(idim) = temp_gp_coord_parent(0, idim);
}

template <int dim>
Core::LinAlg::Tensor<double, dim, dim> calculate_deformation_gradient(
    Core::LinAlg::Matrix<3, 1>& xi, Core::Elements::Element& element,
    const Core::FE::Discretization& discret)
{
  const Core::LinAlg::Tensor<double, dim> xi_tensor = Core::LinAlg::reinterpret_as_tensor<dim>(xi);

  // calculate defgrad based on element discretization type
  Core::LinAlg::Tensor<double, dim, dim> deformation_gradient = Core::FE::cell_type_switch<
      Core::FE::CelltypeSequence<Core::FE::CellType::hex8, Core::FE::CellType::nurbs27>>(
      element.shape(),
      [&](auto celltype_t)
      {
        constexpr Core::FE::CellType celltype = celltype_t();

        const Discret::Elements::ElementNodes<celltype> element_nodes =
            Discret::Elements::evaluate_element_nodes<celltype, dim>(discret, element);

        const Discret::Elements::ShapeFunctionsAndDerivatives<celltype> shape_functions =
            evaluate_shape_functions_and_derivs<celltype>(xi_tensor, element_nodes);

        const Discret::Elements::JacobianMapping<celltype> jacobian_mapping =
            Discret::Elements::evaluate_jacobian_mapping(shape_functions, element_nodes);

        return Discret::Elements::evaluate_deformation_gradient(jacobian_mapping, element_nodes);
      });

  return deformation_gradient;
}

void evaluate_cauchy_stress_tensor_at_xi(const Core::FE::Discretization& discret,
    std::vector<double>& ele_displacement, Core::Elements::Element& element,
    Core::LinAlg::Matrix<3, 1>& xi, Core::LinAlg::Matrix<3, 1, double>& normal_vector,
    Core::LinAlg::Matrix<3, 1, double>& traction_vector,
    std::vector<Core::LinAlg::SerialDenseMatrix>& d_cauchyndir_dd)
{
  // Define directions
  Core::LinAlg::Matrix<3, 1, double> e_x(Core::LinAlg::Initialization::zero),
      e_y(Core::LinAlg::Initialization::zero), e_z(Core::LinAlg::Initialization::zero);
  e_x(0, 0) = 1.0;
  e_y(1, 0) = 1.0;
  e_z(2, 0) = 1.0;
  const std::array<Core::LinAlg::Matrix<3, 1, double>, 3> dirs = {e_x, e_y, e_z};

  // Obtain the traction vector at xi
  if (auto* solid_ele = dynamic_cast<Discret::Elements::Solid*>(&element); solid_ele != nullptr)
  {
    // Fill out the Cauchy stress tensor of the interface
    for (int i_dir = 0; i_dir < 3; ++i_dir)
    {
      Discret::Elements::CauchyNDirLinearizations<3> cauchy_linearizations{};
      cauchy_linearizations.d_cauchyndir_dd = &d_cauchyndir_dd[i_dir];

      traction_vector(i_dir) = solid_ele->get_normal_cauchy_stress_at_xi(ele_displacement,
          Core::LinAlg::reinterpret_as_tensor<3>(xi),
          Core::LinAlg::reinterpret_as_tensor<3>(normal_vector),
          Core::LinAlg::reinterpret_as_tensor<3>(dirs[i_dir]), cauchy_linearizations, &discret);
    }
  }
  else
  {
    FOUR_C_THROW("Unsupported solid element type");
  }
}

template <typename Interface, typename Background>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairNitsche<Interface,
    Background>::evaluate_stress_contributions_nitsche(const Core::FE::Discretization& discret,
    Core::LinAlg::Matrix<Interface::n_dof_, Interface::n_dof_, double>&
        local_stiffness_disp_interface_stress_interface,
    Core::LinAlg::Matrix<Interface::n_dof_, Background::n_dof_, double>&
        local_stiffness_disp_interface_stress_background,
    Core::LinAlg::Matrix<Background::n_dof_, Interface::n_dof_, double>&
        local_stiffness_disp_background_stress_interface,
    Core::LinAlg::Matrix<Background::n_dof_, Background::n_dof_, double>&
        local_stiffness_disp_background_stress_background,
    Core::LinAlg::Matrix<Interface::n_dof_ + Background::n_dof_, 1, double>&
        local_constraint_stresses,
    double& nitsche_average_weight_param)
{
  // Initialize the local stress matrices.
  local_stiffness_disp_interface_stress_interface.put_scalar(0.0);
  local_stiffness_disp_interface_stress_background.put_scalar(0.0);
  local_stiffness_disp_background_stress_interface.put_scalar(0.0);
  local_stiffness_disp_background_stress_background.put_scalar(0.0);
  local_constraint_stresses.put_scalar(0.0);

  // Initialize variables for shape function values.
  Core::LinAlg::Matrix<1, Interface::n_nodes_ * Interface::n_val_, double> N_interface(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, Background::n_nodes_ * Background::n_val_, double> N_background(
      Core::LinAlg::Initialization::zero);

  // The traction vector is the result of multiplying the Cauchy stress with the normal vector of
  // the interface
  Core::LinAlg::Matrix<3, 1, double> traction_vector_interface(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, double> traction_vector_background(Core::LinAlg::Initialization::zero);

  // Calculate the stress contributions to stiffness.
  // Gauss point loop
  for (size_t it_gp = 0; it_gp < interface_integration_points_.size(); it_gp++)
  {
    auto& [xi_interface, xi_background, weight] = interface_integration_points_[it_gp];
    double determinant_interface = Constraints::EmbeddedMesh::get_determinant_interface_element(
        xi_interface, this->element_1());

    // Clear the shape function matrices
    N_interface.clear();
    N_background.clear();
    traction_vector_interface.clear();
    traction_vector_background.clear();

    GeometryPair::EvaluateShapeFunction<Interface>::evaluate(
        N_interface, xi_interface, ele1pos_.shape_function_data_);
    GeometryPair::EvaluateShapeFunction<Background>::evaluate(
        N_background, xi_background, ele2pos_.shape_function_data_);

    // Obtain the normal in the current configuration of the interface at xi
    Core::LinAlg::Matrix<3, 1, double> normal_interface;
    GeometryPair::evaluate_face_normal<Interface>(xi_interface, ele1pos_current_, normal_interface);

    // To evaluate the Cauchy stresses in the parent element of the face element, we need to obtain
    // the corresponding coordinates of the Gauss points of the face element projected into its
    // parent element
    auto face_element = dynamic_cast<Core::Elements::FaceElement*>(&element_1());
    if (!face_element) FOUR_C_THROW("Cast to FaceElement failed!");

    Core::LinAlg::Matrix<3, 1> xi_interface_in_solid(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 3> temp;
    interface_element_gp_in_solid<Interface::spatial_dim_>(
        *face_element, 1., xi_interface.data(), xi_interface_in_solid, temp);

    // The following calculations are meant for 3D elements, therefore check the dimension of the
    // interface parent element and the background element.
    FOUR_C_ASSERT(Interface::spatial_dim_ + 1 != 3 or Background::spatial_dim_ != 3,
        "The following implementation for obtaining the Cauchy stresses is only for 3d elements.");
    FOUR_C_ASSERT(Interface::spatial_dim_ + 1 != 3 or Background::spatial_dim_ != 3,
        "The following implementation for obtaining the Cauchy stresses is only for 3d elements.");

    // define linearizations for each direction
    std::vector<Core::LinAlg::SerialDenseMatrix> d_cauchyndir_dd_interface(3);
    std::vector<Core::LinAlg::SerialDenseMatrix> d_cauchyndir_dd_background(3);

    evaluate_cauchy_stress_tensor_at_xi(discret, ele1_parent_dis_, *face_element->parent_element(),
        xi_interface_in_solid, normal_interface, traction_vector_interface,
        d_cauchyndir_dd_interface);

    // obtain the displacements of the background elements
    std::vector<double> background_displacement;
    for (unsigned int i_n_dof = 0; i_n_dof < Background::n_dof_; i_n_dof++)
      background_displacement.push_back(ele2dis_.element_position_(i_n_dof));

    evaluate_cauchy_stress_tensor_at_xi(discret, background_displacement, element_2(),
        xi_background, normal_interface, traction_vector_background, d_cauchyndir_dd_background);

    // As the calculations of the Cauchy stresses are done in the parent element of the face
    // element, we need the locations of the dofs of the interface in its parent element.
    // These locations are saved in a vector and used for calculating the contributions to the
    // stiffness matrix.
    std::vector<int> lm_interface;
    std::vector<int> lm_parent_interface;
    std::vector<int> dummy_1;
    std::vector<int> dummy_2;
    face_element->location_vector(discret, lm_interface, dummy_1, dummy_2);
    face_element->parent_element()->location_vector(discret, lm_parent_interface, dummy_1, dummy_2);

    std::unordered_map<int, int> interface_solid_index;
    interface_solid_index.reserve(lm_parent_interface.size());
    for (int i = 0; i < static_cast<int>(lm_parent_interface.size()); ++i)
    {
      interface_solid_index[lm_parent_interface[i]] = i;
    }

    std::vector<int> dofs_interface_locations;
    dofs_interface_locations.reserve(Interface::n_dof_);

    for (int dof : lm_interface)
    {
      auto it = interface_solid_index.find(dof);
      if (it != interface_solid_index.end())
      {
        dofs_interface_locations.push_back(it->second);
      }
    }

    // Fill in the local matrix K_nitsche_disp_interface_stress_interface.
    for (unsigned int i_interface_node = 0; i_interface_node < Interface::n_nodes_;
        i_interface_node++)
      for (unsigned int j_interface_node = 0; j_interface_node < Interface::n_nodes_;
          j_interface_node++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          local_stiffness_disp_interface_stress_interface(
              i_interface_node * 3 + i_dim, j_interface_node * 3 + i_dim) +=
              N_interface(i_interface_node) *
              d_cauchyndir_dd_interface[i_dim](dofs_interface_locations[j_interface_node], 0) *
              weight * determinant_interface;

    // Fill in the local matrix K_nitsche_disp_interface_stress_background.
    for (unsigned int i_interface_node = 0; i_interface_node < Interface::n_nodes_;
        i_interface_node++)
      for (unsigned int j_background_node = 0; j_background_node < Background::n_nodes_;
          j_background_node++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          local_stiffness_disp_interface_stress_background(
              i_interface_node * 3 + i_dim, j_background_node * 3 + i_dim) +=
              N_interface(i_interface_node) *
              d_cauchyndir_dd_background[i_dim](j_background_node, 0) * weight *
              determinant_interface;

    // Fill in the local matrix K_nitsche_disp_background_stress_interface.
    for (unsigned int i_background_node = 0; i_background_node < Background::n_nodes_;
        i_background_node++)
      for (unsigned int j_interface_node = 0; j_interface_node < Interface::n_nodes_;
          j_interface_node++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          local_stiffness_disp_background_stress_interface(
              i_background_node * 3 + i_dim, j_interface_node * 3 + i_dim) +=
              N_background(i_background_node) *
              d_cauchyndir_dd_interface[i_dim](dofs_interface_locations[j_interface_node], 0) *
              weight * determinant_interface;

    // Fill in the local matrix K_nitsche_disp_background_stress_background.
    for (unsigned int i_background_node = 0; i_background_node < Background::n_nodes_;
        i_background_node++)
      for (unsigned int j_background_node = 0; j_background_node < Background::n_nodes_;
          j_background_node++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          local_stiffness_disp_background_stress_background(
              i_background_node * 3 + i_dim, j_background_node * 3 + i_dim) +=
              N_background(i_background_node) *
              d_cauchyndir_dd_background[i_dim](j_background_node, 0) * weight *
              determinant_interface;

    // Add the local constraint contributions of Nitsche contributions
    for (unsigned int i_interface_node = 0; i_interface_node < Interface::n_nodes_;
        i_interface_node++)
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        local_constraint_stresses(i_interface_node * 3 + i_dim) -=
            (nitsche_average_weight_param * traction_vector_interface(i_dim) +
                (1.0 - nitsche_average_weight_param) * traction_vector_background(i_dim)) *
            N_interface(i_interface_node) * weight * determinant_interface;

    for (unsigned int i_background_node = 0; i_background_node < Background::n_nodes_;
        i_background_node++)
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        local_constraint_stresses(Interface::n_dof_ + i_background_node * 3 + i_dim) +=
            (nitsche_average_weight_param * traction_vector_interface(i_dim) +
                (1.0 - nitsche_average_weight_param) * traction_vector_background(i_dim)) *
            N_background(i_background_node) * weight * determinant_interface;
  }
}

template <typename Interface, typename Background>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairNitsche<Interface,
    Background>::evaluate_penalty_contributions_nitsche(Core::LinAlg::Matrix<Interface::n_dof_,
                                                            Interface::n_dof_, double>&
                                                            local_stiffness_penalty_interface,
    Core::LinAlg::Matrix<Background::n_dof_, Background::n_dof_, double>&
        local_stiffness_penalty_background,
    Core::LinAlg::Matrix<Interface::n_dof_, Background::n_dof_, double>&
        local_stiffness_penalty_interface_background,
    Core::LinAlg::Matrix<Interface::n_dof_ + Background::n_dof_, 1, double>& local_constraint,
    double& nitsche_stabilization_param)
{
  // Initialize the local penalty matrices.
  local_stiffness_penalty_interface.put_scalar(0.0);
  local_stiffness_penalty_background.put_scalar(0.0);
  local_stiffness_penalty_interface_background.put_scalar(0.0);
  local_constraint.put_scalar(0.0);

  // Initialize variables for shape function values.
  Core::LinAlg::Matrix<1, Interface::n_nodes_ * Interface::n_val_, double> N_interface(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, Background::n_nodes_ * Background::n_val_, double> N_background(
      Core::LinAlg::Initialization::zero);

  // Calculate the penalty matrices.
  // Gauss point loop
  for (size_t it_gp = 0; it_gp < interface_integration_points_.size(); it_gp++)
  {
    auto& [xi_interface, xi_background, weight] = interface_integration_points_[it_gp];

    double determinant_interface = Constraints::EmbeddedMesh::get_determinant_interface_element(
        xi_interface, this->element_1());

    // Get the shape function matrices
    N_interface.clear();
    N_background.clear();

    GeometryPair::EvaluateShapeFunction<Interface>::evaluate(
        N_interface, xi_interface, ele1pos_.shape_function_data_);
    GeometryPair::EvaluateShapeFunction<Background>::evaluate(
        N_background, xi_background, ele2pos_.shape_function_data_);

    // Fill in the local penalty matrix K_penalty_interface.
    for (unsigned int i_interface_node = 0; i_interface_node < Interface::n_nodes_;
        i_interface_node++)
      for (unsigned int i_interface_val = 0; i_interface_val < Interface::n_val_; i_interface_val++)
        for (unsigned int j_interface_node = 0; j_interface_node < Interface::n_nodes_;
            j_interface_node++)
          for (unsigned int j_interface_val = 0; j_interface_val < Interface::n_val_;
              j_interface_val++)
            for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
              local_stiffness_penalty_interface(
                  i_interface_node * Interface::n_val_ * 3 + i_interface_val * 3 + i_dim,
                  j_interface_node * Interface::n_val_ * 3 + j_interface_val * 3 + i_dim) +=
                  N_interface(i_interface_node * Interface::n_val_ + i_interface_val) *
                  N_interface(j_interface_node * Interface::n_val_ + j_interface_val) * weight *
                  determinant_interface;

    // Fill in the local penalty matrix K_penalty_background.
    for (unsigned int i_solid_node = 0; i_solid_node < Background::n_nodes_; i_solid_node++)
      for (unsigned int i_solid_val = 0; i_solid_val < Background::n_val_; i_solid_val++)
        for (unsigned int j_solid_node = 0; j_solid_node < Background::n_nodes_; j_solid_node++)
          for (unsigned int j_solid_val = 0; j_solid_val < Background::n_val_; j_solid_val++)
            for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
              local_stiffness_penalty_background(
                  i_solid_node * Background::n_val_ * 3 + i_solid_val * 3 + i_dim,
                  j_solid_node * Background::n_val_ * 3 + j_solid_val * 3 + i_dim) +=
                  N_background(i_solid_node * Background::n_val_ + i_solid_val) *
                  N_background(j_solid_node * Background::n_val_ + j_solid_val) * weight *
                  determinant_interface;

    // Fill in the local penalty matrix K_penalty_interface_background.
    for (unsigned int i_interface_node = 0; i_interface_node < Interface::n_nodes_;
        i_interface_node++)
      for (unsigned int i_interface_val = 0; i_interface_val < Interface::n_val_; i_interface_val++)
        for (unsigned int i_solid_node = 0; i_solid_node < Background::n_nodes_; i_solid_node++)
          for (unsigned int i_solid_val = 0; i_solid_val < Background::n_val_; i_solid_val++)
            for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
              local_stiffness_penalty_interface_background(
                  i_interface_node * Interface::n_val_ * 3 + i_interface_val * 3 + i_dim,
                  i_solid_node * Background::n_val_ * 3 + i_solid_val * 3 + i_dim) +=
                  N_interface(i_interface_node * Interface::n_val_ + i_interface_val) *
                  N_background(i_solid_node * Background::n_val_ + i_solid_val) * weight *
                  determinant_interface;
  }

  // Add the local constraint contributions.
  for (unsigned int i_interface = 0; i_interface < Interface::n_dof_; i_interface++)
  {
    for (unsigned int j_interface = 0; j_interface < Interface::n_dof_; j_interface++)
      local_constraint(i_interface) += local_stiffness_penalty_interface(i_interface, j_interface) *
                                       this->ele1dis_.element_position_(j_interface);
    for (unsigned int i_background = 0; i_background < Background::n_dof_; i_background++)
      local_constraint(i_interface) -=
          local_stiffness_penalty_interface_background(i_interface, i_background) *
          this->ele2dis_.element_position_(i_background);
  }

  for (unsigned int i_background = 0; i_background < Background::n_dof_; i_background++)
  {
    for (unsigned int j_background = 0; j_background < Background::n_dof_; j_background++)
      local_constraint(Interface::n_dof_ + i_background) +=
          local_stiffness_penalty_background(i_background, j_background) *
          this->ele2dis_.element_position_(j_background);
    for (unsigned int i_interface = 0; i_interface < Interface::n_dof_; i_interface++)
      local_constraint(Interface::n_dof_ + i_background) -=
          local_stiffness_penalty_interface_background(i_interface, i_background) *
          this->ele1dis_.element_position_(i_interface);
  }

  local_constraint.scale(nitsche_stabilization_param);
}

/**
 * Explicit template initialization of template class.
 */
namespace Constraints::EmbeddedMesh
{
  using namespace GeometryPair;

  template class SurfaceToBackgroundCouplingPairNitsche<t_quad4, t_hex8>;
  template class SurfaceToBackgroundCouplingPairNitsche<t_nurbs9, t_hex8>;
  template class SurfaceToBackgroundCouplingPairNitsche<t_nurbs9, t_wedge6>;
  template class SurfaceToBackgroundCouplingPairNitsche<t_quad4, t_nurbs27>;
  template class SurfaceToBackgroundCouplingPairNitsche<t_nurbs9, t_nurbs27>;

}  // namespace Constraints::EmbeddedMesh

FOUR_C_NAMESPACE_CLOSE
