// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_framework_embeddedmesh_interaction_pair_mortar.hpp"

#include "4C_constraint_framework_embeddedmesh_solid_to_solid_mortar_manager.hpp"
#include "4C_constraint_framework_embeddedmesh_solid_to_solid_utils.hpp"
#include "4C_cut_boundarycell.hpp"
#include "4C_cut_cutwizard.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_line_to_volume.hpp"

FOUR_C_NAMESPACE_OPEN


template <typename Interface, typename Background, typename Mortar>
Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairMortar<Interface, Background,
    Mortar>::SurfaceToBackgroundCouplingPairMortar(std::shared_ptr<Core::Elements::Element>
                                                       element1,
    Core::Elements::Element* element2, Constraints::EmbeddedMesh::EmbeddedMeshParams& params_ptr,
    std::shared_ptr<Cut::CutWizard>& cutwizard_ptr,
    std::vector<std::shared_ptr<Cut::BoundaryCell>>& boundary_cells)
    : SolidInteractionPair(element1, element2, params_ptr, cutwizard_ptr, boundary_cells)
{
  // Check that the shape functions in the parameters are of the same shape as the element
  FOUR_C_ASSERT(params_ptr.mortar_shape_function_ ==
                    define_shape_functions_lagrange_multipliers(this->element_1().shape()),
      "The interface element in the coupling pair doesn't have the same shape as defined in the "
      "input parameter MORTAR_SHAPE_FUNCTION.");
  params_ = params_ptr;

  // Initialize the element positions and displacement containers
  ele1pos_ = GeometryPair::InitializeElementData<Interface, double>::initialize(&this->element_1());
  ele2pos_ =
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
  Constraints::EmbeddedMesh::set_gauss_rule_for_interface_and_background(get_boundary_cells(),
      ele1pos_, ele2pos_, interface_integration_points_, num_gauss_points_boundary_cell_);
}

template <typename Interface, typename Background, typename Mortar>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairMortar<Interface, Background,
    Mortar>::get_pair_visualization(Core::IO::VisualizationData& visualization_data,
    std::shared_ptr<Core::LinAlg::Vector<double>> vector_to_visualize,
    const Constraints::EmbeddedMesh::SolidToSolidCouplingManager* coupling_manager,
    std::shared_ptr<std::unordered_set<int>> interface_tracker)
{
  // Cast coupling manager to mortar manager
  const auto* mortar_manager =
      dynamic_cast<const Constraints::EmbeddedMesh::SolidToSolidMortarManager*>(coupling_manager);
  if (mortar_manager == nullptr) FOUR_C_THROW("This manager is not a SolidToSolidMortarManager.");

  // Get the visualization vectors.
  std::vector<double>& point_coordinates = visualization_data.get_point_coordinates();
  std::vector<double>& lambda_vis = visualization_data.get_point_data<double>("lambda");

  // Setup variables.
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> q_lambda;
  Core::LinAlg::Matrix<2, 1, double> xi_mortar_node;

  // Get the lambda GIDs of this pair.
  std::vector<int> lambda_row;
  get_mortar_gid(mortar_manager, this, Mortar::n_dof_, &lambda_row);
  std::vector<double> lambda_pair = Core::FE::extract_values(*vector_to_visualize, lambda_row);
  for (unsigned int i_dof = 0; i_dof < Mortar::n_dof_; i_dof++)
    q_lambda(i_dof) = lambda_pair[i_dof];

  // Write output if this interface element has not written output before
  const auto* face_element = dynamic_cast<const Core::Elements::FaceElement*>(&this->element_1());
  if (face_element == nullptr) FOUR_C_THROW("A face element pointer is needed.");

  auto it = interface_tracker->find(face_element->parent_element_id());
  if (it == interface_tracker->end())
  {
    // Add this element Id to the tracker.
    interface_tracker->insert(face_element->parent_element_id());

    for (unsigned int i_node = 0; i_node < Mortar::n_nodes_; i_node++)
    {
      // Setup variables
      Core::LinAlg::Matrix<3, 1, double> X;
      Core::LinAlg::Matrix<3, 1, double> lambda_discret;

      // Get the local coordinate of this node.
      auto node_coordinates = Core::FE::get_node_coordinates(i_node, Mortar::discretization_);
      xi_mortar_node(0) = node_coordinates(0);
      xi_mortar_node(1) = node_coordinates(1);

      Constraints::EmbeddedMesh::map_from_parametric_to_physical_space<Mortar>(
          ele1pos_, xi_mortar_node, this->ele1pos_.element_position_, X);

      Constraints::EmbeddedMesh::map_from_parametric_to_physical_space<Mortar>(
          ele1pos_, xi_mortar_node, q_lambda, lambda_discret);

      // Add to output data.
      for (unsigned int dim = 0; dim < 3; dim++)
      {
        point_coordinates.push_back(X(dim));
        lambda_vis.push_back(lambda_discret(dim));
      }
    }
  }
}

template <typename Interface, typename Background, typename Mortar>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairMortar<Interface, Background,
    Mortar>::set_current_element_position(Core::FE::Discretization const& discret,
    const Core::LinAlg::Vector<double>& displacement_vector)
{
  std::vector<double> interface_dofvec_timestep = std::vector<double>();
  std::vector<double> background_dofvec_timestep = std::vector<double>();

  Constraints::EmbeddedMesh::get_current_element_displacement(
      discret, &element_1(), displacement_vector, interface_dofvec_timestep);
  Constraints::EmbeddedMesh::get_current_element_displacement(
      discret, &element_2(), displacement_vector, background_dofvec_timestep);

  // Get the initial positions of the first element
  for (int node_ele1 = 0; node_ele1 < element_1().num_point(); node_ele1++)
  {
    // nodal displacements
    ele1dis_.element_position_(0 + 3 * node_ele1) = interface_dofvec_timestep[0 + 3 * node_ele1];
    ele1dis_.element_position_(1 + 3 * node_ele1) = interface_dofvec_timestep[1 + 3 * node_ele1];
    ele1dis_.element_position_(2 + 3 * node_ele1) = interface_dofvec_timestep[2 + 3 * node_ele1];
  }

  // Get the initial positions of the second element
  for (int node_ele2 = 0; node_ele2 < element_2().num_point(); node_ele2++)
  {
    // nodal displacements
    ele2dis_.element_position_(0 + 3 * node_ele2) = background_dofvec_timestep[0 + 3 * node_ele2];
    ele2dis_.element_position_(1 + 3 * node_ele2) = background_dofvec_timestep[1 + 3 * node_ele2];
    ele2dis_.element_position_(2 + 3 * node_ele2) = background_dofvec_timestep[2 + 3 * node_ele2];
  }
}


template <typename Interface, typename Background, typename Mortar>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairMortar<Interface, Background,
    Mortar>::evaluate_and_assemble_coupling_contributions(const Core::FE::Discretization& discret,
    const Constraints::EmbeddedMesh::SolidToSolidCouplingManager* coupling_manager,
    std::map<std::string, std::shared_ptr<Core::LinAlg::SparseMatrix>>& global_matrices,
    std::map<std::string, std::shared_ptr<Core::LinAlg::FEVector<double>>>& global_vectors)
{
  const auto* mortar_manager =
      dynamic_cast<const Constraints::EmbeddedMesh::SolidToSolidMortarManager*>(coupling_manager);
  if (mortar_manager == nullptr)
  {
    FOUR_C_THROW("This coupling manager is not a SolidToSolidMortarManager.");
  }

  evaluate_and_assemble_mortar_contributions(discret, mortar_manager,
      *global_matrices.at("global_g_bl_"), *global_matrices.at("global_g_bg_"),
      *global_matrices.at("global_fbl_l_"), *global_matrices.at("global_fbg_l_"),
      *global_vectors.at("global_constraint_"), *global_vectors.at("global_kappa_"),
      *global_vectors.at("global_active_lambda_"));
}

template <typename Interface, typename Background, typename Mortar>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairMortar<Interface, Background,
    Mortar>::evaluate_and_assemble_mortar_contributions(const Core::FE::Discretization& discret,
    const Constraints::EmbeddedMesh::SolidToSolidMortarManager* mortar_manager,
    Core::LinAlg::SparseMatrix& global_g_bl, Core::LinAlg::SparseMatrix& global_g_bg,
    Core::LinAlg::SparseMatrix& global_fbl_l, Core::LinAlg::SparseMatrix& global_fbg_l,
    Core::LinAlg::FEVector<double>& global_constraint, Core::LinAlg::FEVector<double>& global_kappa,
    Core::LinAlg::FEVector<double>& global_lambda_active)
{
  // Initialize variables for local mortar matrices.
  Core::LinAlg::Matrix<Mortar::n_dof_, Interface::n_dof_, double> local_D(
      Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Mortar::n_dof_, Background::n_dof_, double> local_M(
      Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> local_kappa(
      Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> local_constraint(
      Core::LinAlg::Initialization::uninitialized);

  // Evaluate the local mortar contributions
  evaluate_dm(local_D, local_M, local_kappa, local_constraint);

  // Assemble into global matrices.
  assemble_local_mortar_contributions<Interface, Background, Mortar>(this, discret, mortar_manager,
      global_g_bl, global_g_bg, global_fbl_l, global_fbg_l, global_constraint, global_kappa,
      global_lambda_active, local_D, local_M, local_kappa, local_constraint);
}


template <typename Interface, typename Background, typename Mortar>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairMortar<Interface, Background,
    Mortar>::get_gauss_rule_in_cut_element_for_visualization(Core::IO::VisualizationData&
        cut_element_integration_points_visualization_data)
{
  Constraints::EmbeddedMesh::get_gauss_rule_in_cut_element_for_visualization<Background>(
      element_2(), this->ele2pos_, *get_cutwizard(),
      cut_element_integration_points_visualization_data);
}

template <typename Interface, typename Background, typename Mortar>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairMortar<Interface, Background,
    Mortar>::
    get_projected_gauss_rule_on_interface_and_background_ele_for_visualization(
        Core::IO::VisualizationData& background_integration_points_visualization_data,
        Core::IO::VisualizationData& interface_integration_points_visualization_data)
{
  Constraints::EmbeddedMesh::
      get_projected_gauss_rule_on_interface_and_background_ele_for_visualization<Interface,
          Background>(element_1(), element_2(), ele1pos_, ele2pos_,
          background_integration_points_visualization_data,
          interface_integration_points_visualization_data, interface_integration_points_,
          num_gauss_points_boundary_cell_);
}

template <typename Interface, typename Background, typename Mortar>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairMortar<Interface, Background,
    Mortar>::evaluate_dm(Core::LinAlg::Matrix<Mortar::n_dof_, Interface::n_dof_, double>& local_D,
    Core::LinAlg::Matrix<Mortar::n_dof_, Background::n_dof_, double>& local_M,
    Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_kappa,
    Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_constraint)
{
  // Initialize the local mortar matrices.
  local_D.put_scalar(0.0);
  local_M.put_scalar(0.0);
  local_kappa.put_scalar(0.0);
  local_constraint.put_scalar(0.0);

  // Initialize variables for shape function values.
  Core::LinAlg::Matrix<1, Mortar::n_nodes_ * Mortar::n_val_, double> N_mortar(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, Interface::n_nodes_ * Interface::n_val_, double> N_interface(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, Background::n_nodes_ * Background::n_val_, double> N_background(
      Core::LinAlg::Initialization::zero);

  // Calculate the mortar matrices.
  // Gauss point loop
  for (size_t it_gp = 0; it_gp < interface_integration_points_.size(); it_gp++)
  {
    auto& [xi_interface, xi_background, weight] = interface_integration_points_[it_gp];

    double determinant_interface = Constraints::EmbeddedMesh::get_determinant_interface_element(
        xi_interface, this->element_1());

    // Get the shape function matrices
    N_mortar.clear();
    N_interface.clear();
    N_background.clear();

    GeometryPair::EvaluateShapeFunction<Mortar>::evaluate(
        N_mortar, xi_interface, ele1pos_.shape_function_data_);
    GeometryPair::EvaluateShapeFunction<Interface>::evaluate(
        N_interface, xi_interface, ele1pos_.shape_function_data_);
    GeometryPair::EvaluateShapeFunction<Background>::evaluate(
        N_background, xi_background, ele2pos_.shape_function_data_);

    // Fill in the local templated mortar matrix D.
    for (unsigned int i_mortar_node = 0; i_mortar_node < Mortar::n_nodes_; i_mortar_node++)
      for (unsigned int i_mortar_val = 0; i_mortar_val < Mortar::n_val_; i_mortar_val++)
        for (unsigned int i_interface_node = 0; i_interface_node < Interface::n_nodes_;
            i_interface_node++)
          for (unsigned int i_interface_val = 0; i_interface_val < Interface::n_val_;
              i_interface_val++)
            for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
              local_D(i_mortar_node * Mortar::n_val_ * 3 + i_mortar_val * 3 + i_dim,
                  i_interface_node * Interface::n_val_ * 3 + i_interface_val * 3 + i_dim) +=
                  N_mortar(i_mortar_node * Mortar::n_val_ + i_mortar_val) *
                  N_interface(i_interface_node * Interface::n_val_ + i_interface_val) * weight *
                  determinant_interface;

    // Fill in the local templated mortar matrix M.
    for (unsigned int i_mortar_node = 0; i_mortar_node < Mortar::n_nodes_; i_mortar_node++)
      for (unsigned int i_mortar_val = 0; i_mortar_val < Mortar::n_val_; i_mortar_val++)
        for (unsigned int i_solid_node = 0; i_solid_node < Background::n_nodes_; i_solid_node++)
          for (unsigned int i_solid_val = 0; i_solid_val < Background::n_val_; i_solid_val++)
            for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
              local_M(i_mortar_node * Mortar::n_val_ * 3 + i_mortar_val * 3 + i_dim,
                  i_solid_node * Background::n_val_ * 3 + i_solid_val * 3 + i_dim) +=
                  N_mortar(i_mortar_node * Mortar::n_val_ + i_mortar_val) *
                  N_background(i_solid_node * Background::n_val_ + i_solid_val) * weight *
                  determinant_interface;

    // Fill in the local templated mortar scaling vector kappa.
    for (unsigned int i_mortar_node = 0; i_mortar_node < Mortar::n_nodes_; i_mortar_node++)
      for (unsigned int i_mortar_val = 0; i_mortar_val < Mortar::n_val_; i_mortar_val++)
        for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
          local_kappa(i_mortar_node * Mortar::n_val_ * 3 + i_mortar_val * 3 + i_dim) +=
              N_mortar(i_mortar_node * Mortar::n_val_ + i_mortar_val) * weight *
              determinant_interface;
  }

  // Add the local constraint contributions.
  for (unsigned int i_lambda = 0; i_lambda < Mortar::n_dof_; i_lambda++)
  {
    for (unsigned int i_interface = 0; i_interface < Interface::n_dof_; i_interface++)
      local_constraint(i_lambda) +=
          local_D(i_lambda, i_interface) * this->ele1dis_.element_position_(i_interface);
    for (unsigned int i_background = 0; i_background < Background::n_dof_; i_background++)
      local_constraint(i_lambda) -=
          local_M(i_lambda, i_background) * this->ele2dis_.element_position_(i_background);
  }
}

/**
 * Explicit template initialization of template class.
 */
namespace Constraints::EmbeddedMesh
{
  using namespace GeometryPair;

  template class SurfaceToBackgroundCouplingPairMortar<t_quad4, t_hex8, t_quad4>;
  template class SurfaceToBackgroundCouplingPairMortar<t_nurbs9, t_hex8, t_nurbs9>;
  template class SurfaceToBackgroundCouplingPairMortar<t_nurbs9, t_wedge6, t_nurbs9>;
  template class SurfaceToBackgroundCouplingPairMortar<t_quad4, t_nurbs27, t_quad4>;
  template class SurfaceToBackgroundCouplingPairMortar<t_nurbs9, t_nurbs27, t_nurbs9>;

}  // namespace Constraints::EmbeddedMesh

FOUR_C_NAMESPACE_CLOSE
