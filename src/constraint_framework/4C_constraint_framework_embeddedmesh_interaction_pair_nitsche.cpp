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


template <typename Interface, typename ParentInterface, typename Background>
Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairNitsche<Interface, ParentInterface,
    Background>::SurfaceToBackgroundCouplingPairNitsche(std::shared_ptr<Core::Elements::Element>
                                                            element1,
    Core::Elements::Element* element2, Constraints::EmbeddedMesh::EmbeddedMeshParams& params_ptr,
    std::shared_ptr<Cut::CutWizard>& cutwizard_ptr,
    std::vector<std::shared_ptr<Cut::BoundaryCell>>& boundary_cells)
    : SolidInteractionPair(element1, element2, params_ptr, cutwizard_ptr, boundary_cells)
{
  // Add parameters
  params_ = params_ptr;

  // Get a pointer to the parent element of element_1()
  auto face_element = dynamic_cast<Core::Elements::FaceElement*>(&this->element_1());
  if (!face_element) FOUR_C_THROW("Cast to FaceElement failed!");
  parent_element1_ = (face_element->parent_element());

  // Initialize the element positions and displacement containers
  ele1pos_ = GeometryPair::InitializeElementData<Interface, double>::initialize(&this->element_1());
  ele1parentpos_ = GeometryPair::InitializeElementData<ParentInterface, double>::initialize(
      &this->parent_element_1());
  ele2pos_ =
      GeometryPair::InitializeElementData<Background, double>::initialize(&this->element_2());

  ele1dis_ = GeometryPair::InitializeElementData<Interface, double>::initialize(&this->element_1());
  ele1parentdis_ = GeometryPair::InitializeElementData<ParentInterface, double>::initialize(
      &this->parent_element_1());
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

  for (int node_ele1parent = 0; node_ele1parent < parent_element_1().num_point(); node_ele1parent++)
  {
    // nodal positions
    ele1parentpos_.element_position_(0 + 3 * node_ele1parent) =
        parent_element_1().nodes()[node_ele1parent]->x()[0];
    ele1parentpos_.element_position_(1 + 3 * node_ele1parent) =
        parent_element_1().nodes()[node_ele1parent]->x()[1];
    ele1parentpos_.element_position_(2 + 3 * node_ele1parent) =
        parent_element_1().nodes()[node_ele1parent]->x()[2];
  }

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


template <typename Interface, typename ParentInterface, typename Background>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairNitsche<Interface, ParentInterface,
    Background>::set_current_element_position(Core::FE::Discretization const& discret,
    const Core::LinAlg::Vector<double>& displacement_vector)
{
  std::vector<double> interface_dofvec_timestep = std::vector<double>();
  std::vector<double> parent_dofvec_timestep = std::vector<double>();
  std::vector<double> background_dofvec_timestep = std::vector<double>();

  Constraints::EmbeddedMesh::get_current_element_displacement(
      discret, &element_1(), displacement_vector, interface_dofvec_timestep);
  Constraints::EmbeddedMesh::get_current_element_displacement(
      discret, &parent_element_1(), displacement_vector, parent_dofvec_timestep);
  Constraints::EmbeddedMesh::get_current_element_displacement(
      discret, &element_2(), displacement_vector, background_dofvec_timestep);

  // Set the displacements of the first element
  for (int node_ele1 = 0; node_ele1 < element_1().num_point(); node_ele1++)
  {
    // nodal displacements
    ele1dis_.element_position_(0 + 3 * node_ele1) = interface_dofvec_timestep[0 + 3 * node_ele1];
    ele1dis_.element_position_(1 + 3 * node_ele1) = interface_dofvec_timestep[1 + 3 * node_ele1];
    ele1dis_.element_position_(2 + 3 * node_ele1) = interface_dofvec_timestep[2 + 3 * node_ele1];
  }

  // Set the displacements of the parent element
  for (int node_ele1parent = 0; node_ele1parent < parent_element_1().num_point(); node_ele1parent++)
  {
    // nodal positions
    ele1parentdis_.element_position_(0 + 3 * node_ele1parent) =
        parent_dofvec_timestep[0 + 3 * node_ele1parent];
    ele1parentdis_.element_position_(1 + 3 * node_ele1parent) =
        parent_dofvec_timestep[1 + 3 * node_ele1parent];
    ele1parentdis_.element_position_(2 + 3 * node_ele1parent) =
        parent_dofvec_timestep[2 + 3 * node_ele1parent];
  }

  // Set the displacements of the second element
  for (int node_ele2 = 0; node_ele2 < element_2().num_point(); node_ele2++)
  {
    // nodal displacements
    ele2dis_.element_position_(0 + 3 * node_ele2) = background_dofvec_timestep[0 + 3 * node_ele2];
    ele2dis_.element_position_(1 + 3 * node_ele2) = background_dofvec_timestep[1 + 3 * node_ele2];
    ele2dis_.element_position_(2 + 3 * node_ele2) = background_dofvec_timestep[2 + 3 * node_ele2];
  }
}


template <typename Interface, typename ParentInterface, typename Background>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairNitsche<Interface, ParentInterface,
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

template <typename Interface, typename ParentInterface, typename Background>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairNitsche<Interface, ParentInterface,
    Background>::evaluate_and_assemble_nitsche_contributions(const Core::FE::Discretization&
                                                                 discret,
    const Constraints::EmbeddedMesh::SolidToSolidNitscheManager* nitsche_manager,
    Core::LinAlg::SparseMatrix& global_penalty_interface,
    Core::LinAlg::SparseMatrix& global_penalty_background,
    Core::LinAlg::SparseMatrix& global_penalty_interface_background,
    Core::LinAlg::SparseMatrix& global_nitsche_parent,
    Core::LinAlg::SparseMatrix& global_nitsche_background,
    Core::LinAlg::SparseMatrix& global_nitsche_parent_background,
    Core::LinAlg::FEVector<double>& global_penalty_constraint,
    Core::LinAlg::FEVector<double>& global_nitsche_constraint, double& nitsche_average_weight_param)
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
  Core::LinAlg::Matrix<ParentInterface::n_dof_, ParentInterface::n_dof_, double>
      local_stiffness_nitsche_parent_interface(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Background::n_dof_, Background::n_dof_, double>
      local_stiffness_nitsche_background(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<ParentInterface::n_dof_, Background::n_dof_, double>
      local_stiffness_nitsche_parent_background(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<ParentInterface::n_dof_ + Background::n_dof_, 1, double>
      local_constraint_nitsche(Core::LinAlg::Initialization::uninitialized);

  // Evaluate the local penalty contributions of Nitsche method
  evaluate_penalty_contributions_nitsche(local_stiffness_penalty_interface,
      local_stiffness_penalty_background, local_stiffness_penalty_interface_background,
      local_constraint_penalty);

  // Evaluate stress contributions of Nitsche method
  evaluate_stress_contributions_nitsche(discret, local_stiffness_nitsche_parent_interface,
      local_stiffness_nitsche_background, local_stiffness_nitsche_parent_background,
      local_constraint_nitsche, nitsche_average_weight_param);

  // Assemble into global matrices.
  assemble_local_nitsche_contributions<Interface, ParentInterface, Background>(this, discret,
      global_penalty_interface, global_penalty_background, global_penalty_interface_background,
      global_nitsche_parent, global_nitsche_background, global_nitsche_parent_background,
      global_penalty_constraint, global_nitsche_constraint, local_stiffness_penalty_interface,
      local_stiffness_penalty_background, local_stiffness_penalty_interface_background,
      local_stiffness_nitsche_parent_interface, local_stiffness_nitsche_background,
      local_stiffness_nitsche_parent_background, local_constraint_penalty,
      local_constraint_nitsche);
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

template <Core::FE::CellType celltype>
void evaluate_cauchy_stress_tensor_at_xi(const Core::FE::Discretization& discret,
    std::vector<double>& ele_displacement, Core::Elements::Element& element,
    Core::LinAlg::Matrix<3, 1>& xi, Core::LinAlg::Matrix<3, 1, double>& normal_vector,
    Core::LinAlg::Matrix<3, 1, double>& traction_vector,
    Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>&
        deformation_gradient,
    Core::LinAlg::Matrix<3, 3, double>& cauchy_stress_tensor,
    std::vector<Core::LinAlg::SerialDenseMatrix>& d_cauchyndir_dd
    // Discret::Elements::CauchyNDirLinearizationDependencies<celltype>& linearization_dependencies
)
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
    Discret::Elements::CauchyNDirLinearizations<3> cauchy_linearizations{};

    // Fill out the traction vector and Cauchy stress tensor of the interface
    for (int i_dir = 0; i_dir < 3; ++i_dir)
    {
      cauchy_linearizations.d_cauchyndir_dd = &d_cauchyndir_dd[i_dir];

      traction_vector(i_dir) = solid_ele->get_normal_cauchy_stress_at_xi(ele_displacement,
          Core::LinAlg::reinterpret_as_tensor<3>(xi),
          Core::LinAlg::reinterpret_as_tensor<3>(normal_vector),
          Core::LinAlg::reinterpret_as_tensor<3>(dirs[i_dir]), cauchy_linearizations, &discret);

      for (int j_dir = 0; j_dir < 3; ++j_dir)
      {
        Discret::Elements::CauchyNDirLinearizations<3> cauchy_linearizations_dummy{};
        cauchy_stress_tensor(i_dir, j_dir) = solid_ele->get_normal_cauchy_stress_at_xi(
            ele_displacement, Core::LinAlg::reinterpret_as_tensor<3>(xi),
            Core::LinAlg::reinterpret_as_tensor<3>(dirs[i_dir]),
            Core::LinAlg::reinterpret_as_tensor<3>(dirs[j_dir]), cauchy_linearizations_dummy,
            &discret);
      }
    }
  }
  else
  {
    FOUR_C_THROW("Unsupported solid element type");
  }
}

template <Core::FE::CellType celltype>
struct QuantitiesDomainEvaluatedAtXi
{
  /// Derivative of the shape functions w.r.t. the reference coordinates
  const Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::num_nodes(celltype), double> N_XYZ_;

  /// Deformation gradient
  const Core::LinAlg::Matrix<3, 3> defgrad_;

  /// First Piola-Kirchhoff stresses
  const Core::LinAlg::Matrix<3, 3> pk1_;

  /// Second Piola-Kirchhoff stresses
  Discret::Elements::Stress<celltype> pk2_;
};


template <Core::FE::CellType celltype>
struct LinearizationDependencies
{
  Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::dim<celltype>> d_defgrad_;

  Core::LinAlg::Matrix<3, 3> d_pk2_;
};

// Helper function to evaluate necessary quantities of a domain at xi (like stress, deformation
// gradient and so on...)
template <Core::FE::CellType celltype>
QuantitiesDomainEvaluatedAtXi<celltype> evaluate_domain_quantities_at_xi(
    const Core::FE::Discretization& discret, const Core::Elements::Element& element,
    const Core::LinAlg::Matrix<Core::FE::num_nodes(celltype), 3, double> initial_position_ele,
    const std::vector<double>& ele_displacement, Core::LinAlg::Matrix<3, 1>& xi,
    Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::num_nodes(celltype), double> dxi_N)
{
  const Core::LinAlg::Tensor<double, Core::FE::dim<celltype>> xi_tensor =
      Core::LinAlg::reinterpret_as_tensor<Core::FE::dim<celltype>>(xi);

  const Discret::Elements::ElementNodes<celltype> element_nodes =
      Discret::Elements::evaluate_element_nodes<celltype>(element, ele_displacement, &discret);

  const Discret::Elements::ShapeFunctionsAndDerivatives<celltype> shape_functions =
      Discret::Elements::evaluate_shape_functions_and_derivs<celltype>(xi_tensor, element_nodes);

  // std::cout << "------Results in evaluate_domain_quantities_at_xi" << std::endl;

  // Core::LinAlg::Matrix<3, 1> displ_solid;
  // for (int i = 0; i < 3 ; ++i)
  //   for (int j = 0; j < Core::FE::num_nodes(celltype); ++j)
  //     displ_solid(i) += shape_functions.shapefunctions_(j) * ele_displacement[j * 3 + i];

  // std::cout << "Solid displacement " << std::endl;
  // displ_solid.print(std::cout);
  //
  // std::cout << "shape functions " << std::endl;
  // for (int j = 0; j < Core::FE::num_nodes(celltype); ++j)
  //   std::cout << shape_functions.shapefunctions_(j) << ", ";
  // std::cout << std::endl;

  const Discret::Elements::JacobianMapping<celltype> jacobian_mapping =
      Discret::Elements::evaluate_jacobian_mapping<celltype>(shape_functions, element_nodes);

  // std::cout << "derivatives shape functions wrt the reference configuration" << std::endl;
  // for (int i_dim = 0; i_dim < 3; ++i_dim)
  // {
  //   for (int k = 0; k < Core::FE::num_nodes(celltype); ++k)
  //   {
  //     std::cout << jacobian_mapping.N_XYZ[k](i_dim) << ", ";
  //   }
  //   std::cout << std::endl;
  // }

  // auto jacobian = Core::LinAlg::make_matrix_view(jacobian_mapping.jacobian_);
  // std::cout << "jacobian " << std::endl;
  // jacobian.print(std::cout);

  // auto inverse_jacobian = Core::LinAlg::make_matrix_view(jacobian_mapping.inverse_jacobian_);
  // std::cout << "inverse jacobian " << std::endl;
  // inverse_jacobian.print(std::cout);

  Core::LinAlg::Tensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>
      deformation_gradient =
          Discret::Elements::evaluate_deformation_gradient(jacobian_mapping, element_nodes);

  const Discret::Elements::SpatialMaterialMapping<celltype> spatial_material_mapping =
      evaluate_spatial_material_mapping(jacobian_mapping, element_nodes);

  const Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>
      cauchygreen = evaluate_cauchy_green(spatial_material_mapping);

  const Core::LinAlg::SymmetricTensor<double, Core::FE::dim<celltype>, Core::FE::dim<celltype>>
      gl_strain = Discret::Elements::evaluate_green_lagrange_strain(cauchygreen);

  auto* solid_ele = dynamic_cast<const Discret::Elements::Solid*>(&element);
  if (!solid_ele) FOUR_C_THROW("Cast to Discret::Elements::Solid failed!");

  // Setting a parameter list (I think this is actually not used in the evaluate_material_stress
  // function, therefore I have it empty)
  Teuchos::ParameterList params;
  Mat::EvaluationContext context{.total_time = nullptr,  // Do not have the time here
      .time_step_size = nullptr,                         // Do not have the time-step here
      .xi = &xi_tensor,
      .ref_coords = nullptr};
  // Setting the number of Gauss point (gp) to zero (this is later used for a warning, but not
  // relevant for calculations)
  const int gp = 0;
  auto pk2 = Discret::Elements::evaluate_material_stress<celltype>(*(solid_ele->solid_material()),
      deformation_gradient, gl_strain, params, context, gp, element.id());

  // Calculate the 1st Piola-Kirchhoff stresses
  Core::LinAlg::Matrix<3, 3> def_gradient = Core::LinAlg::make_matrix_view(deformation_gradient);
  Core::LinAlg::Matrix<3, 3> def_gradient_inv(Core::LinAlg::Initialization::zero);
  def_gradient_inv.invert(def_gradient);

  Core::LinAlg::Matrix<3, 3> pk2_matrix(Core::LinAlg::Initialization::zero);
  pk2_matrix(0, 0) = pk2.pk2_(0, 0);
  pk2_matrix(1, 1) = pk2.pk2_(1, 1);
  pk2_matrix(2, 2) = pk2.pk2_(2, 2);
  pk2_matrix(0, 1) = pk2.pk2_(0, 1);
  pk2_matrix(1, 0) = pk2.pk2_(0, 1);
  pk2_matrix(1, 2) = pk2.pk2_(1, 2);
  pk2_matrix(2, 1) = pk2.pk2_(1, 2);
  pk2_matrix(0, 2) = pk2.pk2_(0, 2);
  pk2_matrix(2, 0) = pk2.pk2_(0, 2);
  Core::LinAlg::Matrix<3, 3> pk1(Core::LinAlg::Initialization::zero);
  pk1.multiply(def_gradient, pk2_matrix);

  // Evaluate the derivatives of the shape functions with respect to the reference coordinates XYZ
  Core::LinAlg::Matrix<3, 3, double> jacobian_background(Core::LinAlg::Initialization::zero);
  jacobian_background.multiply(dxi_N, initial_position_ele);

  Core::LinAlg::Matrix<3, 3, double> jacobian_background_inv(Core::LinAlg::Initialization::zero);
  jacobian_background_inv.invert(jacobian_background);

  // std::cout << "jacobian " << std::endl;
  // jacobian_background.print(std::cout);
  // std::cout << "jacobian inv " << std::endl;
  // jacobian_background_inv.print(std::cout);

  Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::num_nodes(celltype), double> dX_N(
      Core::LinAlg::Initialization::uninitialized);
  dX_N.multiply(jacobian_background_inv, dxi_N);

  // std::cout << "dX_N" << std::endl;
  // dX_N.print(std::cout);

  // Return quantities
  QuantitiesDomainEvaluatedAtXi<celltype> quantities_domain_at_xi{dX_N, def_gradient, pk1, pk2};

  // std::cout << "------------------------------------" << std::endl;
  return quantities_domain_at_xi;
}


// Calculate the variation of the 1st. Piola-Kirchhoff stresses at a node in a certain dimensional
// direction (in 3D, dir = 0, 1, 2)
template <Core::FE::CellType celltype>
std::tuple<Core::LinAlg::Matrix<3, 3>, LinearizationDependencies<celltype>>
evaluate_variation_first_pk_at_node_dir(unsigned int& num_node, unsigned int& n_dir,
    QuantitiesDomainEvaluatedAtXi<celltype>& quantities_domain_at_xi)
{
  LinearizationDependencies<celltype> linearization_dependencies;

  // Get the variation of the deformation gradient at a node and direction
  linearization_dependencies.d_defgrad_.put_scalar(0.0);
  for (int i_dim = 0; i_dim < Core::FE::dim<celltype>; i_dim++)
    linearization_dependencies.d_defgrad_(n_dir, i_dim) =
        quantities_domain_at_xi.N_XYZ_(i_dim, num_node);

  // std::cout << "In evaluate_variation_first_pk_at_node_dir: " << std::endl;
  //
  // std::cout << "defgrad_" << std::endl;
  // quantities_domain_at_xi.defgrad_.print(std::cout);
  //
  // std::cout << "d_defgrad_" << std::endl;
  // linearization_dependencies.d_defgrad_.print(std::cout);


  // Get the variation of the Green-Lagrange strains at a node and direction
  Core::LinAlg::Matrix<3, 3> d_glstrain(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 3> d_defgradT_defgrad(Core::LinAlg::Initialization::zero);
  d_defgradT_defgrad.multiply_tn(
      linearization_dependencies.d_defgrad_, quantities_domain_at_xi.defgrad_);
  d_glstrain.update(0.5, d_defgradT_defgrad, 1.0);

  Core::LinAlg::Matrix<3, 3> defgradT_d_defgrad(Core::LinAlg::Initialization::zero);
  defgradT_d_defgrad.multiply_tn(
      quantities_domain_at_xi.defgrad_, linearization_dependencies.d_defgrad_);
  d_glstrain.update(0.5, defgradT_d_defgrad, 1.0);

  Core::LinAlg::Matrix<6, 1> d_glstrain_voigt(Core::LinAlg::Initialization::zero);
  d_glstrain_voigt(0) = d_glstrain(0, 0);
  d_glstrain_voigt(1) = d_glstrain(1, 1);
  d_glstrain_voigt(2) = d_glstrain(2, 2);
  d_glstrain_voigt(3) = 2 * d_glstrain(0, 1);
  d_glstrain_voigt(4) = 2 * d_glstrain(1, 2);
  d_glstrain_voigt(5) = 2 * d_glstrain(0, 2);

  // std::cout << "d_glstrain_voigt" << std::endl;
  // d_glstrain_voigt.print(std::cout);

  // Get the variation of the second Piola-Kirchhoff stress at a node and direction
  Core::LinAlg::Matrix<6, 1> d_pk2_voigt(Core::LinAlg::Initialization::zero);
  d_pk2_voigt.multiply(
      make_stress_like_voigt_view(quantities_domain_at_xi.pk2_.cmat_), d_glstrain_voigt);

  // std::cout << "d_pk2_voigt" << std::endl;
  // d_pk2_voigt.print(std::cout);

  linearization_dependencies.d_pk2_.put_scalar(0.0);
  linearization_dependencies.d_pk2_(0, 0) = d_pk2_voigt(0);
  linearization_dependencies.d_pk2_(1, 1) = d_pk2_voigt(1);
  linearization_dependencies.d_pk2_(2, 2) = d_pk2_voigt(2);
  linearization_dependencies.d_pk2_(0, 1) = d_pk2_voigt(3);
  linearization_dependencies.d_pk2_(1, 0) = d_pk2_voigt(3);
  linearization_dependencies.d_pk2_(1, 2) = d_pk2_voigt(4);
  linearization_dependencies.d_pk2_(2, 1) = d_pk2_voigt(4);
  linearization_dependencies.d_pk2_(0, 2) = d_pk2_voigt(5);
  linearization_dependencies.d_pk2_(2, 0) = d_pk2_voigt(5);

  // Get the variation of the first Piola-Kirchhoff stresses at a node and direction
  Core::LinAlg::Matrix<3, 3> d_defgrad_pk2(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> pk2_matrix(Core::LinAlg::Initialization::zero);
  for (int i_dim = 0; i_dim < Core::FE::dim<celltype>; i_dim++)
    for (int j_dim = 0; j_dim < Core::FE::dim<celltype>; j_dim++)
      pk2_matrix(i_dim, j_dim) = quantities_domain_at_xi.pk2_.pk2_(i_dim, j_dim);
  d_defgrad_pk2.multiply(linearization_dependencies.d_defgrad_, pk2_matrix);

  Core::LinAlg::Matrix<3, 3> defgrad_d_pk2(Core::LinAlg::Initialization::zero);
  defgrad_d_pk2.multiply(quantities_domain_at_xi.defgrad_, linearization_dependencies.d_pk2_);

  // set d_pk1_node_dir to zero for safety
  Core::LinAlg::Matrix<3, 3> d_pk1_node_dir(Core::LinAlg::Initialization::zero);
  d_pk1_node_dir.update(1.0, d_defgrad_pk2, 1.0);
  d_pk1_node_dir.update(1.0, defgrad_d_pk2, 1.0);

  // std::cout << "d_pk1_node_dir" << std::endl;
  // d_pk1_node_dir.print(std::cout);

  return {d_pk1_node_dir, linearization_dependencies};
}

template <Core::FE::CellType celltype>
Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::dim<celltype>>
evaluate_lin_variation_first_pk(LinearizationDependencies<celltype>& lin_dependencies_d_dof,
    LinearizationDependencies<celltype>& lin_dependencies_dof,
    QuantitiesDomainEvaluatedAtXi<celltype>& quantities_domain_at_xi)
{
  // Get the linearization of the virtual Green-Lagrange strains
  Core::LinAlg::Matrix<3, 3> lin_d_glstrain(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3> d_defgradT_lin_defgrad(Core::LinAlg::Initialization::zero);
  d_defgradT_lin_defgrad.multiply_tn(
      lin_dependencies_d_dof.d_defgrad_, lin_dependencies_dof.d_defgrad_);
  lin_d_glstrain.update(0.5, d_defgradT_lin_defgrad, 1.0);

  Core::LinAlg::Matrix<3, 3> lin_defgradT_d_defgrad(Core::LinAlg::Initialization::zero);
  lin_defgradT_d_defgrad.multiply_tn(
      lin_dependencies_dof.d_defgrad_, lin_dependencies_d_dof.d_defgrad_);
  lin_d_glstrain.update(0.5, lin_defgradT_d_defgrad, 1.0);

  Core::LinAlg::Matrix<6, 1> lin_d_glstrain_voigt(Core::LinAlg::Initialization::zero);
  lin_d_glstrain_voigt(0) = lin_d_glstrain(0, 0);
  lin_d_glstrain_voigt(1) = lin_d_glstrain(1, 1);
  lin_d_glstrain_voigt(2) = lin_d_glstrain(2, 2);
  lin_d_glstrain_voigt(3) = 2 * lin_d_glstrain(0, 1);
  lin_d_glstrain_voigt(4) = 2 * lin_d_glstrain(1, 2);
  lin_d_glstrain_voigt(5) = 2 * lin_d_glstrain(0, 2);

  // Get the linearization of the variation of the second Piola-Kirchhoff stress
  Core::LinAlg::Matrix<6, 1> lin_d_pk2_voigt(Core::LinAlg::Initialization::zero);
  lin_d_pk2_voigt.multiply(
      make_stress_like_voigt_view(quantities_domain_at_xi.pk2_.cmat_), lin_d_glstrain_voigt);

  Core::LinAlg::Matrix<3, 3> lin_d_pk2(Core::LinAlg::Initialization::zero);
  lin_d_pk2.put_scalar(0.0);
  lin_d_pk2(0, 0) = lin_d_pk2_voigt(0);
  lin_d_pk2(1, 1) = lin_d_pk2_voigt(1);
  lin_d_pk2(2, 2) = lin_d_pk2_voigt(2);
  lin_d_pk2(0, 1) = lin_d_pk2_voigt(3);
  lin_d_pk2(1, 0) = lin_d_pk2_voigt(3);
  lin_d_pk2(1, 2) = lin_d_pk2_voigt(4);
  lin_d_pk2(2, 1) = lin_d_pk2_voigt(4);
  lin_d_pk2(0, 2) = lin_d_pk2_voigt(5);
  lin_d_pk2(2, 0) = lin_d_pk2_voigt(5);

  // Calculate the linearization of the variation of the first Piola-Kirchhoff stresses
  Core::LinAlg::Matrix<Core::FE::dim<celltype>, Core::FE::dim<celltype>> lin_d_pk1(
      Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 3> d_defgrad_lin_pk2(Core::LinAlg::Initialization::zero);
  d_defgrad_lin_pk2.multiply(lin_dependencies_d_dof.d_defgrad_, lin_dependencies_dof.d_pk2_);
  lin_d_pk1.update(1.0, d_defgrad_lin_pk2, 1.0);

  Core::LinAlg::Matrix<3, 3> lin_defgrad_d_pk2(Core::LinAlg::Initialization::zero);
  lin_defgrad_d_pk2.multiply(lin_dependencies_dof.d_defgrad_, lin_dependencies_d_dof.d_pk2_);
  lin_d_pk1.update(1.0, lin_defgrad_d_pk2, 1.0);

  Core::LinAlg::Matrix<3, 3> defgrad_lin_d_pk2(Core::LinAlg::Initialization::zero);
  defgrad_lin_d_pk2.multiply(quantities_domain_at_xi.defgrad_, lin_d_pk2);
  lin_d_pk1.update(1.0, defgrad_lin_d_pk2, 1.0);

  return lin_d_pk1;
}


template <typename Interface, typename ParentInterface, typename Background>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairNitsche<Interface, ParentInterface,
    Background>::evaluate_stress_contributions_nitsche(const Core::FE::Discretization& discret,
    Core::LinAlg::Matrix<ParentInterface::n_dof_, ParentInterface::n_dof_, double>&
        local_stiffness_nitsche_parent,
    Core::LinAlg::Matrix<Background::n_dof_, Background::n_dof_, double>&
        local_stiffness_nitsche_background,
    Core::LinAlg::Matrix<ParentInterface::n_dof_, Background::n_dof_, double>&
        local_stiffness_nitsche_parent_background,
    Core::LinAlg::Matrix<ParentInterface::n_dof_ + Background::n_dof_, 1, double>&
        local_constraint_nitsche,
    double& nitsche_average_weight_param)
{
  // The following calculations are meant for 3D elements, therefore check the dimension of the
  // interface parent element and the background element.
  FOUR_C_ASSERT(Interface::spatial_dim_ + 1 != 3 or Background::spatial_dim_ != 3,
      "The following implementation is only for 3d elements.");

  // Initialize the local stress matrices.
  local_stiffness_nitsche_parent.put_scalar(0.0);
  local_stiffness_nitsche_background.put_scalar(0.0);
  local_stiffness_nitsche_parent_background.put_scalar(0.0);
  local_constraint_nitsche.put_scalar(0.0);

  // Initialize variables for shape function values.
  Core::LinAlg::Matrix<1, Interface::n_nodes_, double> N_interface(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, ParentInterface::n_nodes_, double> N_parent(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, Background::n_nodes_, double> N_background(
      Core::LinAlg::Initialization::zero);

  // Matrix for derivatives of the shape functions with respect to the parametric coordinates
  Core::LinAlg::Matrix<Interface::element_dim_, Interface::n_nodes_, double> dN_interface(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<ParentInterface::element_dim_, ParentInterface::n_nodes_, double> dN_parent(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Background::element_dim_, Background::n_nodes_, double> dN_background(
      Core::LinAlg::Initialization::zero);

  // Get nodal initial position of parent and background elements
  Core::LinAlg::Matrix<Interface::n_nodes_, 3, double> nodal_initpos_interface(
      Core::LinAlg::Initialization::zero);
  for (int i_dim = 0; i_dim < 3; i_dim++)
    for (int i_interface_node = 0; i_interface_node < Interface::n_nodes_; i_interface_node++)
      nodal_initpos_interface(i_interface_node, i_dim) =
          ele1pos_.element_position_(i_dim + 3 * i_interface_node);

  Core::LinAlg::Matrix<ParentInterface::n_nodes_, 3, double> nodal_initpos_parent(
      Core::LinAlg::Initialization::zero);
  for (int i_dim = 0; i_dim < 3; i_dim++)
    for (int i_parent_node = 0; i_parent_node < ParentInterface::n_nodes_; i_parent_node++)
      nodal_initpos_parent(i_parent_node, i_dim) =
          ele1parentpos_.element_position_(i_dim + 3 * i_parent_node);

  // std::cout << "Initial position parent" << std::endl;
  // nodal_initpos_parent.print(std::cout);

  Core::LinAlg::Matrix<Background::n_nodes_, 3, double> nodal_initpos_background(
      Core::LinAlg::Initialization::zero);
  for (int i_dim = 0; i_dim < 3; i_dim++)
    for (int i_background_node = 0; i_background_node < Background::n_nodes_; i_background_node++)
      nodal_initpos_background(i_background_node, i_dim) =
          ele2pos_.element_position_(i_dim + 3 * i_background_node);

  // std::cout << "Initial position background" << std::endl;
  // nodal_initpos_background.print(std::cout);

  // Get nodal displacement of parent and background elements
  std::vector<double> nodal_disp_parent;
  for (unsigned int i_n_dof = 0; i_n_dof < ParentInterface::n_dof_; i_n_dof++)
    nodal_disp_parent.push_back(ele1parentdis_.element_position_(i_n_dof));

  std::vector<double> nodal_disp_background;
  for (unsigned int i_n_dof = 0; i_n_dof < Background::n_dof_; i_n_dof++)
    nodal_disp_background.push_back(ele2dis_.element_position_(i_n_dof));

  // Calculate the stress contributions to stiffness.
  // Gauss point loop
  for (size_t it_gp = 0; it_gp < interface_integration_points_.size(); it_gp++)
  {
    auto& [xi_interface, xi_background, weight] = interface_integration_points_[it_gp];
    double determinant_interface = Constraints::EmbeddedMesh::get_determinant_interface_element(
        xi_interface, this->element_1());

    // To evaluate the first Piola-Kirchhoff stresses in the parent element of the face element, we
    // need to obtain the corresponding coordinates of the Gauss points of the face element
    // projected into its parent element
    auto face_element = dynamic_cast<Core::Elements::FaceElement*>(&element_1());
    if (!face_element) FOUR_C_THROW("Cast to FaceElement failed!");
    Core::LinAlg::Matrix<3, 1> xi_parent(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 3> temp;
    interface_element_gp_in_solid<Interface::spatial_dim_>(
        *face_element, 1., xi_interface.data(), xi_parent, temp);

    // std::cout << "xi interface : " << std::endl;
    // xi_interface.print(std::cout);
    // std::cout << "xi parent : " << std::endl;
    // xi_parent.print(std::cout);

    // Clear the shape function matrices
    N_interface.clear();
    N_parent.clear();
    N_background.clear();
    dN_parent.clear();
    dN_background.clear();

    GeometryPair::EvaluateShapeFunction<Interface>::evaluate(
        N_interface, xi_interface, ele1pos_.shape_function_data_);
    GeometryPair::EvaluateShapeFunction<ParentInterface>::evaluate(
        N_parent, xi_parent, ele1parentpos_.shape_function_data_);
    GeometryPair::EvaluateShapeFunction<Background>::evaluate(
        N_background, xi_background, ele2pos_.shape_function_data_);

    GeometryPair::EvaluateShapeFunction<Interface>::evaluate_deriv1(
        dN_interface, xi_interface, ele1pos_.shape_function_data_);
    GeometryPair::EvaluateShapeFunction<ParentInterface>::evaluate_deriv1(
        dN_parent, xi_parent, ele1parentpos_.shape_function_data_);
    GeometryPair::EvaluateShapeFunction<Background>::evaluate_deriv1(
        dN_background, xi_background, ele2pos_.shape_function_data_);



    // std::cout << "----------------\n Shape functions interface " << std::endl;
    // N_interface.print(std::cout);
    //
    // std::cout << "****************\n Shape functions parent " << std::endl;
    // N_parent.print(std::cout);
    //
    // std::cout << "////////////////\n Shape functions background " << std::endl;
    // N_background.print(std::cout);
    //
    // std::cout << "Derivative shape functions background " << std::endl;
    // dN_background.print(std::cout);

    // std::cout << "Jacobian of the interface" << std::endl;
    // Core::LinAlg::Matrix<3, 3, double> jacobian_interface;
    // jacobian_interface.multiply(dN_interface, nodal_initpos_interface);
    // jacobian_interface.print(std::cout);


    // Obtain the normal in the reference configuration of the interface at xi
    Core::LinAlg::Matrix<3, 1, double> unit_normal_interface;
    GeometryPair::evaluate_face_normal<Interface>(xi_interface, ele1pos_, unit_normal_interface);
    // std::cout << "Normal interface " << std::endl;
    // unit_normal_interface.print(std::cout);


    // evaluate first Piola-Kirchhoff stress and other necessary quantities on the interface at xi
    // std::cout << "----- Parent" << std::endl;
    auto quantities_at_xi_parent =
        evaluate_domain_quantities_at_xi<ParentInterface::discretization_>(discret,
            parent_element_1(), nodal_initpos_parent, nodal_disp_parent, xi_parent, dN_parent);

    // std::cout << "shape functions parent " << std::endl;
    // N_parent.print(std::cout);
    //
    // std::cout << "Derivative shape functions parent wrt xi" << std::endl;
    // dN_parent.print(std::cout);
    //
    // std::cout << "Interface: Derivatives of shape functions wrt to the reference configuration "
    // << std::endl; quantities_at_xi_parent.N_XYZ_.print(std::cout); std::cout << "Interface: PK1
    // stress " << std::endl; quantities_at_xi_parent.pk1_.print(std::cout); std::cout <<
    // "Interface: PK2 stress " << std::endl; Core::LinAlg::Matrix<3, 3>
    // pk2_matrix_interface(Core::LinAlg::Initialization::zero); pk2_matrix_interface(0, 0) =
    // quantities_at_xi_parent.pk2_.pk2_(0, 0); pk2_matrix_interface(1, 1) =
    // quantities_at_xi_parent.pk2_.pk2_(1, 1); pk2_matrix_interface(2, 2) =
    // quantities_at_xi_parent.pk2_.pk2_(2, 2); pk2_matrix_interface(0, 1) =
    // quantities_at_xi_parent.pk2_.pk2_(0, 1); pk2_matrix_interface(1, 0) =
    // quantities_at_xi_parent.pk2_.pk2_(0, 1); pk2_matrix_interface(1, 2) =
    // quantities_at_xi_parent.pk2_.pk2_(1, 2); pk2_matrix_interface(2, 1) =
    // quantities_at_xi_parent.pk2_.pk2_(1, 2); pk2_matrix_interface(0, 2) =
    // quantities_at_xi_parent.pk2_.pk2_(0, 2); pk2_matrix_interface(2, 0) =
    // quantities_at_xi_parent.pk2_.pk2_(0, 2); pk2_matrix_interface.print(std::cout);



    // evaluate first Piola-Kirchhoff stress and other necessary quantities on the background at xi
    // std::cout << "----- Background" << std::endl;
    auto quantities_at_xi_background =
        evaluate_domain_quantities_at_xi<Background::discretization_>(discret, element_2(),
            nodal_initpos_background, nodal_disp_background, xi_background, dN_background);

    // std::cout << "Derivative shape functions background " << std::endl;
    // dN_background.print(std::cout);
    //
    // std::cout << "Background: Derivatives of shape functions wrt to the reference configuration "
    // << std::endl; quantities_at_xi_background.N_XYZ_.print(std::cout); std::cout << "Background:
    // PK1 stress " << std::endl; quantities_at_xi_background.pk1_.print(std::cout); std::cout <<
    // "Background: PK2 stress " << std::endl; Core::LinAlg::Matrix<3, 3>
    // pk2_matrix_background(Core::LinAlg::Initialization::zero); pk2_matrix_background(0, 0) =
    // quantities_at_xi_background.pk2_.pk2_(0, 0); pk2_matrix_background(1, 1) =
    // quantities_at_xi_background.pk2_.pk2_(1, 1); pk2_matrix_background(2, 2) =
    // quantities_at_xi_background.pk2_.pk2_(2, 2); pk2_matrix_background(0, 1) =
    // quantities_at_xi_background.pk2_.pk2_(0, 1); pk2_matrix_background(1, 0) =
    // quantities_at_xi_background.pk2_.pk2_(0, 1); pk2_matrix_background(1, 2) =
    // quantities_at_xi_background.pk2_.pk2_(1, 2); pk2_matrix_background(2, 1) =
    // quantities_at_xi_background.pk2_.pk2_(1, 2); pk2_matrix_background(0, 2) =
    // quantities_at_xi_background.pk2_.pk2_(0, 2); pk2_matrix_background(2, 0) =
    // quantities_at_xi_background.pk2_.pk2_(0, 2); pk2_matrix_background.print(std::cout);

    // evaluate the weighted difference between the first Piola-Kirchhoff stresses of the interface
    // and the background
    Core::LinAlg::Matrix<3, 1> traction_pk1_interface(Core::LinAlg::Initialization::zero);
    traction_pk1_interface.multiply(quantities_at_xi_parent.pk1_, unit_normal_interface);

    // std::cout << "traction interface " << std::endl;
    // traction_pk1_interface.print(std::cout);

    Core::LinAlg::Matrix<3, 1> traction_pk1_background(Core::LinAlg::Initialization::zero);
    traction_pk1_background.multiply(quantities_at_xi_background.pk1_, unit_normal_interface);

    // std::cout << "traction background " << std::endl;
    // traction_pk1_background.print(std::cout);

    Core::LinAlg::Matrix<3, 1> traction_pk1_weighted(Core::LinAlg::Initialization::zero);
    for (int i_dim = 0; i_dim < 3; ++i_dim)
      traction_pk1_weighted(i_dim) =
          nitsche_average_weight_param * traction_pk1_interface(i_dim) +
          (1.0 - nitsche_average_weight_param) * traction_pk1_background(i_dim);

    // std::cout << "traction weighted " << std::endl;
    // traction_pk1_weighted.print(std::cout);

    // Evaluate gap between interface and background at this gauss point
    Core::LinAlg::Matrix<3, 1> displacement_at_xi_interface(Core::LinAlg::Initialization::zero);
    for (unsigned int i_dir = 0; i_dir < 3; ++i_dir)
    {
      for (unsigned int i_interface = 0; i_interface < Interface::n_nodes_; ++i_interface)
        displacement_at_xi_interface(i_dir) +=
            N_interface(i_interface) * this->ele1dis_.element_position_(i_interface * 3 + i_dir);
    }
    // std::cout << "displacement interface " << std::endl;
    // displacement_at_xi_interface.print(std::cout);

    Core::LinAlg::Matrix<3, 1> displacement_at_xi_background(Core::LinAlg::Initialization::zero);
    for (unsigned int i_dir = 0; i_dir < 3; ++i_dir)
    {
      for (unsigned int i_background = 0; i_background < Background::n_nodes_; ++i_background)
        displacement_at_xi_background(i_dir) +=
            N_background(i_background) * this->ele2dis_.element_position_(i_background * 3 + i_dir);
    }
    // std::cout << "displacement background " << std::endl;
    // displacement_at_xi_background.print(std::cout);

    Core::LinAlg::Matrix<3, 1> gap_at_xi(Core::LinAlg::Initialization::zero);
    for (unsigned int i_dir = 0; i_dir < 3; ++i_dir)
      gap_at_xi(i_dir) = displacement_at_xi_interface(i_dir) - displacement_at_xi_background(i_dir);

    std::cout << "gap between interface background" << std::endl;
    gap_at_xi.print(std::cout);

    // As the calculations of the first Piola-Kirchhoff stresses are done in the parent element of
    // the face element, we need the locations of the dofs of the interface in its parent element.
    // These locations are saved in a vector and used for calculating the contributions to the
    // stiffness matrix.
    std::vector<int> id_nodes_parent_ele;
    auto nodes_parent_ele = face_element->parent_element()->nodes();
    for (int inode = 0; inode < face_element->parent_element()->num_node(); ++inode)
      id_nodes_parent_ele.push_back(nodes_parent_ele[inode]->id());

    std::unordered_map<int, int> interface_solid_index;
    interface_solid_index.reserve(id_nodes_parent_ele.size());
    for (int i = 0; i < id_nodes_parent_ele.size(); ++i)
      interface_solid_index[id_nodes_parent_ele[i]] = i;

    std::vector<int> id_nodes_face_ele;
    auto nodes_face_ele = face_element->nodes();
    for (int inode = 0; inode < face_element->num_node(); ++inode)
      id_nodes_face_ele.push_back(nodes_face_ele[inode]->id());

    std::vector<unsigned int> nodes_interface_locations;
    nodes_interface_locations.reserve(Interface::n_nodes_);

    for (int node : id_nodes_face_ele)
    {
      auto it = interface_solid_index.find(node);
      if (it != interface_solid_index.end())
      {
        nodes_interface_locations.push_back(it->second);
      }
    }

    // std::cout << "Nodes interface locations" << std::endl;
    // for (int i = 0; i < nodes_interface_locations.size(); ++i)
    //   std::cout << nodes_interface_locations[i] << std::endl;

    // Build helper stiffness matrices
    /// Terms in \delta \Pi_{Nitsche}^{interface} , displacement_{interface}
    // Core::LinAlg::Matrix<ParentInterface::n_dof_, ParentInterface::n_dof_, double>
    //     lin_d_pk1_parent(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<ParentInterface::n_dof_, ParentInterface::n_dof_, double>
        d_pk1_parent_disp_interface(Core::LinAlg::Initialization::zero);
    // Core::LinAlg::Matrix<ParentInterface::n_dof_, ParentInterface::n_dof_, double>
    //     d_disp_interface_pk1_interface(Core::LinAlg::Initialization::zero);

    for (unsigned int i_parent_node = 0; i_parent_node < ParentInterface::n_nodes_; i_parent_node++)
      for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
      {
        auto [d_pk1_i_interface, _] =
            evaluate_variation_first_pk_at_node_dir(i_parent_node, i_dir, quantities_at_xi_parent);

        Core::LinAlg::Matrix<3, 1> traction_d_pk1_i_interface(Core::LinAlg::Initialization::zero);
        traction_d_pk1_i_interface.multiply(d_pk1_i_interface, unit_normal_interface);

        for (unsigned int j_interface_node = 0; j_interface_node < Interface::n_nodes_;
            j_interface_node++)
        {
          for (unsigned int j_dir = 0; j_dir < 3; j_dir++)
          {
            d_pk1_parent_disp_interface(i_parent_node * 3 + i_dir,
                nodes_interface_locations[j_interface_node] * 3 + j_dir) -=
                traction_d_pk1_i_interface(j_dir) * N_interface(j_interface_node) *
                nitsche_average_weight_param * weight * determinant_interface;
          }
        }
      }
    local_stiffness_nitsche_parent.update(1.0, d_pk1_parent_disp_interface, 1.0);
    // std::cout << "local_stiffness_nitsche_parent 1 " << std::endl;
    // local_stiffness_nitsche_parent.print(std::cout);
    local_stiffness_nitsche_parent.update_t(1.0, d_pk1_parent_disp_interface, 1.0);
    // std::cout << "local_stiffness_nitsche_parent 2 " << std::endl;
    // local_stiffness_nitsche_parent.print(std::cout);


    for (unsigned int i_parent_node = 0; i_parent_node < ParentInterface::n_nodes_; i_parent_node++)
      for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
      {
        auto [_, lin_dependencies_i_interface] =
            evaluate_variation_first_pk_at_node_dir(i_parent_node, i_dir, quantities_at_xi_parent);

        for (unsigned int j_parent_node = 0; j_parent_node < ParentInterface::n_nodes_;
            j_parent_node++)
        {
          for (unsigned int j_dir = 0; j_dir < 3; j_dir++)
          {
            auto [_, lin_dependencies_j_interface] = evaluate_variation_first_pk_at_node_dir(
                j_parent_node, j_dir, quantities_at_xi_parent);

            auto lin_d_pk1 = evaluate_lin_variation_first_pk(lin_dependencies_i_interface,
                lin_dependencies_j_interface, quantities_at_xi_parent);
            Core::LinAlg::Matrix<3, 1> traction_lin_var_pk1(Core::LinAlg::Initialization::zero);
            traction_lin_var_pk1.multiply(lin_d_pk1, unit_normal_interface);

            local_stiffness_nitsche_parent(i_parent_node * 3 + i_dir, j_parent_node * 3 + j_dir) -=
                (traction_lin_var_pk1(0) * gap_at_xi(0) + traction_lin_var_pk1(1) * gap_at_xi(1) +
                    traction_lin_var_pk1(2) * gap_at_xi(2)) *
                nitsche_average_weight_param * weight * determinant_interface;
          }
        }
      }

    // std::cout << "local_stiffness_nitsche_parent 3 " << std::endl;
    // local_stiffness_nitsche_parent.print(std::cout);



    // for (unsigned int i_interface_node = 0; i_interface_node < ParentInterface::n_nodes_;
    //     i_interface_node++)
    //   for (unsigned int j_interface_node = 0; j_interface_node < ParentInterface::n_nodes_;
    //       j_interface_node++)
    //   {
    //     for (unsigned int j_dir = 0; j_dir < 3; j_dir++)
    //     {
    //       auto [d_pk1_j_interface, _] = evaluate_variation_first_pk_at_node_dir(
    //           j_interface_node, j_dir, quantities_at_xi_parent);
    //
    //       Core::LinAlg::Matrix<3, 1>
    //       traction_d_pk1_j_interface(Core::LinAlg::Initialization::zero);
    //       traction_d_pk1_j_interface.multiply(d_pk1_j_interface, unit_normal_interface);
    //
    //       for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
    //       {
    //         d_disp_interface_pk1_interface(
    //             i_interface_node * 3 + i_dir, j_interface_node * 3 + j_dir) -=
    //             N_parent(i_interface_node) * traction_d_pk1_j_interface(i_dir) *
    //             nitsche_average_weight_param * weight * determinant_interface;
    //       }
    //     }
    //   }
    //
    // //// Add contributions to matrix
    // local_stiffness_nitsche_interface.update(1.0, lin_d_pk1_interface, 1.0);
    // local_stiffness_nitsche_interface.update(1.0, d_pk1_interface_disp_interface, 1.0);
    // local_stiffness_nitsche_interface.update(1.0, d_disp_interface_pk1_interface, 1.0);


    /// Terms in \delta \Pi_{Nitsche}^{interface} , displacement_{background}
    // Core::LinAlg::Matrix<ParentInterface::n_dof_, Background::n_dof_, double>
    //     d_pk1_interface_disp_background(Core::LinAlg::Initialization::zero);
    // Core::LinAlg::Matrix<ParentInterface::n_dof_, Background::n_dof_, double>
    //     d_disp_interface_pk1_background(Core::LinAlg::Initialization::zero);

    for (unsigned int i_parent_node = 0; i_parent_node < ParentInterface::n_nodes_; i_parent_node++)
      for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
      {
        auto [d_pk1_i_parent, _] =
            evaluate_variation_first_pk_at_node_dir(i_parent_node, i_dir, quantities_at_xi_parent);

        Core::LinAlg::Matrix<3, 1> traction_d_pk1_i_parent(Core::LinAlg::Initialization::zero);
        traction_d_pk1_i_parent.multiply(d_pk1_i_parent, unit_normal_interface);

        for (unsigned int j_background_node = 0; j_background_node < Background::n_nodes_;
            j_background_node++)
          for (unsigned int j_dir = 0; j_dir < 3; j_dir++)
          {
            local_stiffness_nitsche_parent_background(
                i_parent_node * 3 + i_dir, j_background_node * 3 + j_dir) +=
                traction_d_pk1_i_parent(j_dir) * N_background(j_background_node) *
                nitsche_average_weight_param * weight * determinant_interface;
          }
      }
    // std::cout << "local_stiffness_nitsche_parent_background 1 " << std::endl;
    // local_stiffness_nitsche_parent_background.print(std::cout);

    for (unsigned int i_interface_node = 0; i_interface_node < Interface::n_nodes_;
        i_interface_node++)
      for (unsigned int j_background_node = 0; j_background_node < Background::n_nodes_;
          j_background_node++)
        for (unsigned int j_dir = 0; j_dir < 3; j_dir++)
        {
          auto [d_pk1_j_background, _] = evaluate_variation_first_pk_at_node_dir(
              j_background_node, j_dir, quantities_at_xi_background);
          Core::LinAlg::Matrix<3, 1> traction_d_pk1_j_background(
              Core::LinAlg::Initialization::zero);
          traction_d_pk1_j_background.multiply(d_pk1_j_background, unit_normal_interface);

          for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
          {
            local_stiffness_nitsche_parent_background(
                nodes_interface_locations[i_interface_node] * 3 + i_dir,
                j_background_node * 3 + j_dir) -=
                N_interface(i_interface_node) * traction_d_pk1_j_background(i_dir) *
                (1.0 - nitsche_average_weight_param) * weight * determinant_interface;
          }
        }
    // std::cout << "local_stiffness_nitsche_parent_background 2 " << std::endl;
    // local_stiffness_nitsche_parent_background.print(std::cout);

    //// Add contributions to matrices
    // local_stiffness_nitsche_parent_background.update(1.0, d_pk1_interface_disp_background, 1.0);
    // local_stiffness_nitsche_parent_background.update(1.0, d_disp_interface_pk1_background, 1.0);
    //
    // local_stiffness_nitsche_background_interface.update_t(
    //     local_stiffness_nitsche_parent_background);


    // Terms in \delta \Pi_{Nitsche}^{background} , displacement_{background}
    // Core::LinAlg::Matrix<Background::n_dof_, Background::n_dof_, double> lin_d_pk1_background(
    //     Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<Background::n_dof_, Background::n_dof_, double>
        d_pk1_background_disp_background(Core::LinAlg::Initialization::zero);
    // Core::LinAlg::Matrix<Background::n_dof_, Background::n_dof_, double>
    //     d_disp_background_pk1_background(Core::LinAlg::Initialization::zero);

    for (unsigned int i_background_node = 0; i_background_node < Background::n_nodes_;
        i_background_node++)
      for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
      {
        auto [d_pk1_i_background, lin_dependencies_i_background] =
            evaluate_variation_first_pk_at_node_dir(
                i_background_node, i_dir, quantities_at_xi_background);

        Core::LinAlg::Matrix<3, 1> traction_i_background(Core::LinAlg::Initialization::zero);
        traction_i_background.multiply(d_pk1_i_background, unit_normal_interface);

        for (unsigned int j_background_node = 0; j_background_node < Background::n_nodes_;
            j_background_node++)
        {
          for (unsigned int j_dir = 0; j_dir < 3; j_dir++)
          {
            d_pk1_background_disp_background(
                i_background_node * 3 + i_dir, j_background_node * 3 + j_dir) +=
                traction_i_background(j_dir) * N_background(j_background_node) *
                (1.0 - nitsche_average_weight_param) * weight * determinant_interface;
          }
        }
      }
    local_stiffness_nitsche_background.update(1.0, d_pk1_background_disp_background, 1.0);
    // std::cout << "local_stiffness_nitsche_background 1 " << std::endl;
    // local_stiffness_nitsche_background.print(std::cout);
    local_stiffness_nitsche_background.update_t(1.0, d_pk1_background_disp_background, 1.0);
    // std::cout << "local_stiffness_nitsche_background 2 " << std::endl;
    // local_stiffness_nitsche_background.print(std::cout);

    for (unsigned int i_background_node = 0; i_background_node < Background::n_nodes_;
        i_background_node++)
      for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
      {
        auto [_, lin_dependencies_i_background] = evaluate_variation_first_pk_at_node_dir(
            i_background_node, i_dir, quantities_at_xi_background);

        for (unsigned int j_background_node = 0; j_background_node < Background::n_nodes_;
            j_background_node++)
        {
          for (unsigned int j_dir = 0; j_dir < 3; j_dir++)
          {
            auto [_, lin_dependencies_j_background] = evaluate_variation_first_pk_at_node_dir(
                j_background_node, j_dir, quantities_at_xi_background);

            auto lin_d_pk1 = evaluate_lin_variation_first_pk(lin_dependencies_i_background,
                lin_dependencies_j_background, quantities_at_xi_background);
            Core::LinAlg::Matrix<3, 1> traction_lin_d_pk1(Core::LinAlg::Initialization::zero);
            traction_lin_d_pk1.multiply(lin_d_pk1, unit_normal_interface);

            local_stiffness_nitsche_background(
                i_background_node * 3 + i_dir, j_background_node * 3 + j_dir) -=
                (traction_lin_d_pk1(0) * gap_at_xi(0) + traction_lin_d_pk1(1) * gap_at_xi(1) +
                    traction_lin_d_pk1(2) * gap_at_xi(2)) *
                (1.0 - nitsche_average_weight_param) * weight * determinant_interface;
          }
        }
      }
    // std::cout << "local_stiffness_nitsche_background 3 " << std::endl;
    // local_stiffness_nitsche_background.print(std::cout);



    // for (unsigned int i_background_node = 0; i_background_node < Background::n_nodes_;
    //     i_background_node++)
    //   for (unsigned int j_background_node = 0; j_background_node < Background::n_nodes_;
    //       j_background_node++)
    //   {
    //     for (unsigned int j_dir = 0; j_dir < 3; j_dir++)
    //     {
    //       auto [d_pk1_j_background, _] = evaluate_variation_first_pk_at_node_dir(
    //           j_background_node, j_dir, quantities_at_xi_background);
    //
    //       Core::LinAlg::Matrix<3, 1> traction_d_pk1_j_background(
    //           Core::LinAlg::Initialization::zero);
    //       traction_d_pk1_j_background.multiply(d_pk1_j_background, unit_normal_interface);
    //
    //       for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
    //       {
    //         d_disp_background_pk1_background(
    //             i_background_node * 3 + i_dir, j_background_node * 3 + j_dir) +=
    //             N_background(i_background_node) * traction_d_pk1_j_background(i_dir) *
    //             (1.0 - nitsche_average_weight_param) * weight * determinant_interface;
    //       }
    //     }
    //   }

    // local_stiffness_nitsche_background.update(1.0, lin_d_pk1_background, 1.0);
    // local_stiffness_nitsche_background.update(1.0, d_pk1_background_disp_background, 1.0);
    // local_stiffness_nitsche_background.update(1.0, d_disp_background_pk1_background, 1.0);


    // Obtain the contributions to the local constraint
    for (unsigned int i_parent_node = 0; i_parent_node < ParentInterface::n_nodes_; i_parent_node++)
      for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
      {
        auto [d_pk1_i_parent, _] =
            evaluate_variation_first_pk_at_node_dir(i_parent_node, i_dir, quantities_at_xi_parent);

        Core::LinAlg::Matrix<3, 1> d_traction_i_parent(Core::LinAlg::Initialization::zero);
        d_traction_i_parent.multiply(d_pk1_i_parent, unit_normal_interface);

        local_constraint_nitsche(i_parent_node * 3 + i_dir) -=
            (d_traction_i_parent(0) * gap_at_xi(0) + d_traction_i_parent(1) * gap_at_xi(1) +
                d_traction_i_parent(2) * gap_at_xi(2)) *
            nitsche_average_weight_param * weight * determinant_interface;
      }
    // std::cout << "local_constraint_nitsche 1 " << std::endl;
    // local_constraint_nitsche.print(std::cout);


    for (unsigned int i_background_node = 0; i_background_node < Background::n_nodes_;
        i_background_node++)
      for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
      {
        auto [d_pk1_i_background, _] = evaluate_variation_first_pk_at_node_dir(
            i_background_node, i_dir, quantities_at_xi_background);

        Core::LinAlg::Matrix<3, 1> d_traction_i_background(Core::LinAlg::Initialization::zero);
        d_traction_i_background.multiply(d_pk1_i_background, unit_normal_interface);

        local_constraint_nitsche(ParentInterface::n_dof_ + i_background_node * 3 + i_dir) -=
            (d_traction_i_background(0) * gap_at_xi(0) + d_traction_i_background(1) * gap_at_xi(1) +
                d_traction_i_background(2) * gap_at_xi(2)) *
            (1.0 - nitsche_average_weight_param) * weight * determinant_interface;
      }
    // std::cout << "local_constraint_nitsche 2 " << std::endl;
    // local_constraint_nitsche.print(std::cout);


    for (unsigned int i_interface_node = 0; i_interface_node < Interface::n_nodes_;
        i_interface_node++)
      for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
      {
        local_constraint_nitsche(nodes_interface_locations[i_interface_node] * 3 + i_dir) -=
            traction_pk1_weighted(i_dir) * N_interface(i_interface_node) * weight *
            determinant_interface;
      }
    // std::cout << "local_constraint_nitsche 3 " << std::endl;
    // local_constraint_nitsche.print(std::cout);


    for (unsigned int i_background_node = 0; i_background_node < Background::n_nodes_;
        i_background_node++)
      for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
      {
        local_constraint_nitsche(ParentInterface::n_dof_ + i_background_node * 3 + i_dir) +=
            traction_pk1_weighted(i_dir) * N_background(i_background_node) * weight *
            determinant_interface;
      }
    // std::cout << "local_constraint_nitsche 1 " << std::endl;
    // local_constraint_nitsche.print(std::cout);
  }

  std::cout << "local_stiffness_nitsche_parent " << std::endl;
  local_stiffness_nitsche_parent.print(std::cout);
  std::cout << "local_stiffness_nitsche_background " << std::endl;
  local_stiffness_nitsche_background.print(std::cout);
  std::cout << "local_stiffness_nitsche_parent_background " << std::endl;
  local_stiffness_nitsche_parent_background.print(std::cout);
  std::cout << "local constraint nitsche " << std::endl;
  local_constraint_nitsche.print(std::cout);
}

template <typename Interface, typename ParentInterface, typename Background>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairNitsche<Interface, ParentInterface,
    Background>::evaluate_penalty_contributions_nitsche(Core::LinAlg::Matrix<Interface::n_dof_,
                                                            Interface::n_dof_, double>&
                                                            local_stiffness_penalty_interface,
    Core::LinAlg::Matrix<Background::n_dof_, Background::n_dof_, double>&
        local_stiffness_penalty_background,
    Core::LinAlg::Matrix<Interface::n_dof_, Background::n_dof_, double>&
        local_stiffness_penalty_interface_background,
    Core::LinAlg::Matrix<Interface::n_dof_ + Background::n_dof_, 1, double>& local_constraint)
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
}

/**
 * Explicit template initialization of template class.
 */
namespace Constraints::EmbeddedMesh
{
  using namespace GeometryPair;

  template class SurfaceToBackgroundCouplingPairNitsche<t_quad4, t_hex8, t_hex8>;
  template class SurfaceToBackgroundCouplingPairNitsche<t_nurbs9, t_nurbs27, t_hex8>;
  template class SurfaceToBackgroundCouplingPairNitsche<t_nurbs9, t_nurbs27, t_wedge6>;
  template class SurfaceToBackgroundCouplingPairNitsche<t_quad4, t_hex8, t_nurbs27>;
  template class SurfaceToBackgroundCouplingPairNitsche<t_nurbs9, t_nurbs27, t_nurbs27>;

}  // namespace Constraints::EmbeddedMesh

FOUR_C_NAMESPACE_CLOSE
