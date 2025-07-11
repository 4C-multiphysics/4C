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
#include "4C_cut_elementhandle.hpp"
#include "4C_cut_sidehandle.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_geometry_pair_line_to_surface.hpp"
#include "4C_geometry_pair_line_to_volume.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  //! Helper function to map a point from the element's parametric space to the
  //! physical space
  template <typename Pointtype>
  void map_from_parametric_to_physical_space(
      GeometryPair::ElementData<Pointtype, double> element_data,
      Core::LinAlg::Matrix<Pointtype::element_dim_, 1>& point_param_space,
      Core::LinAlg::Matrix<Pointtype::n_dof_, 1, double> nodal_values,
      Core::LinAlg::Matrix<Pointtype::spatial_dim_, 1, double>& point_physical_space)
  {
    // Evaluate the shape functions on the given point
    Core::LinAlg::Matrix<1, Pointtype::n_nodes_ * Pointtype::n_val_, double> shape_fun(
        Core::LinAlg::Initialization::zero);

    GeometryPair::EvaluateShapeFunction<Pointtype>::evaluate(
        shape_fun, point_param_space, element_data.shape_function_data_);

    // Map the point to the physical system by multiplying the shape
    // functions with the element nodes
    for (unsigned int node = 0; node < Pointtype::n_nodes_; node++)
      for (unsigned int dim = 0; dim < Pointtype::spatial_dim_; dim++)
        for (unsigned int val = 0; val < Pointtype::n_val_; val++)
          point_physical_space(dim) += nodal_values(3 * Pointtype::n_val_ * node + 3 * val + dim) *
                                       shape_fun(Pointtype::n_val_ * node + val);
  }
}  // namespace

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
  FOUR_C_ASSERT(params_ptr.embedded_mesh_mortar_shape_function_ ==
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
  evaluate_interface_element_nodal_normals(ele1pos_);

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

template <typename Interface, typename Background, typename Mortar>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairMortar<Interface, Background,
    Mortar>::get_pair_visualization(Core::IO::VisualizationData&
                                        lagrange_multipliers_visualization_data,
    std::shared_ptr<Core::LinAlg::Vector<double>> lambda,
    const Constraints::EmbeddedMesh::SolidToSolidMortarManager* mortar_manager,
    std::shared_ptr<std::unordered_set<int>> interface_tracker)
{
  // Get the visualization vectors.
  std::vector<double>& point_coordinates =
      lagrange_multipliers_visualization_data.get_point_coordinates();
  std::vector<double>& lambda_vis =
      lagrange_multipliers_visualization_data.get_point_data<double>("lambda");

  // Setup variables.
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> q_lambda;
  Core::LinAlg::Matrix<2, 1, double> xi_mortar_node;

  // Get the lambda GIDs of this pair.
  std::vector<int> lambda_row;
  get_mortar_gid(mortar_manager, this, Mortar::n_dof_, &lambda_row);
  std::vector<double> lambda_pair = Core::FE::extract_values(*lambda, lambda_row);
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

      map_from_parametric_to_physical_space<Mortar>(
          ele1pos_, xi_mortar_node, this->ele1pos_.element_position_, X);

      map_from_parametric_to_physical_space<Mortar>(
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

template <typename Surface, Core::FE::CellType boundarycell_distype>
std::shared_ptr<Core::FE::GaussPoints> project_boundary_cell_gauss_rule_on_interface(
    Cut::BoundaryCell* boundary_cell, GeometryPair::ElementData<Surface, double>& ele1pos)
{
  // Get the coordinates of the vertices of the boundary cell
  const Core::LinAlg::SerialDenseMatrix vertices_boundary_cell = boundary_cell->coordinates();
  const unsigned num_vertices = Core::FE::num_nodes(boundarycell_distype);
  Core::LinAlg::Matrix<2, num_vertices> projected_vertices_xi;

  for (unsigned i_vertex = 0; i_vertex < num_vertices; i_vertex++)
  {
    Core::LinAlg::Matrix<3, 1> vertex_to_project;
    Core::LinAlg::Matrix<3, 1> xi_interface;

    for (int i_dim = 0; i_dim < 3; i_dim++)
      vertex_to_project(i_dim) = vertices_boundary_cell(i_dim, i_vertex);

    GeometryPair::ProjectionResult temp_projection_result;
    GeometryPair::project_point_to_surface(
        vertex_to_project, ele1pos, xi_interface, temp_projection_result);

    if (temp_projection_result == GeometryPair::ProjectionResult::projection_not_found)
      FOUR_C_THROW("No projection was found. ");
    else if (temp_projection_result == GeometryPair::ProjectionResult::projection_found_not_valid)
      std::cout << "WARNING: a projection was found but it is not valid\n";

    projected_vertices_xi(0, i_vertex) = xi_interface(0);
    projected_vertices_xi(1, i_vertex) = xi_interface(1);
  }

  // Check if the points are arranged in a counterclockwise. This is to avoid
  // getting negative gauss points weights in CreateProjected(..). This is the case if:
  // (y3 - y1)(x2 - x1) > (y2 - y1)(x3 - x1). If this is not the case, the points
  // should be rearranged
  if ((projected_vertices_xi(1, 2) - projected_vertices_xi(1, 0)) *
          (projected_vertices_xi(0, 1) - projected_vertices_xi(0, 0)) <
      (projected_vertices_xi(1, 1) - projected_vertices_xi(1, 0)) *
          (projected_vertices_xi(0, 2) - projected_vertices_xi(0, 0)))
  {
    Core::LinAlg::Matrix<2, num_vertices> temp_xie;

    for (size_t i = 0; i < num_vertices; ++i)
    {
      temp_xie(0, i) = projected_vertices_xi(0, num_vertices - 1 - i);
      temp_xie(1, i) = projected_vertices_xi(1, num_vertices - 1 - i);
    }

    projected_vertices_xi = temp_xie;
  }

  std::shared_ptr<Core::FE::GaussPoints> gp =
      Core::FE::GaussIntegration::create_projected<boundarycell_distype>(
          projected_vertices_xi, boundary_cell->get_cubature_degree());

  // Check if the weights of the obtained Gauss Points are positive
  for (int it_gp = 0; it_gp < gp->num_points(); it_gp++)
  {
    FOUR_C_ASSERT_ALWAYS(
        gp->weight(it_gp) > 0.0, "The Gauss rule for this boundary cell has negative weights.");
  }

  return gp;
}

template <typename Interface, typename Background, typename Mortar>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairMortar<Interface, Background,
    Mortar>::set_gauss_rule_for_interface_and_background()
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
        project_boundary_cell_gauss_rule_on_interface<Interface, Core::FE::CellType::tri3>(
            it_boundarycell->get(), ele1pos_);

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
    Mortar>::get_projected_gauss_rule_in_cut_element(Core::IO::VisualizationData&
        cut_element_integration_points_visualization_data)
{
  // Get the vectors to be filled related to the integration of cut elements
  std::vector<double>& cutelement_point_coordinates =
      cut_element_integration_points_visualization_data.get_point_coordinates();
  std::vector<double>& cutelement_weights =
      cut_element_integration_points_visualization_data.get_point_data<double>("weights");
  std::vector<int>& cutelement_integration_cell_id =
      cut_element_integration_points_visualization_data.get_point_data<int>("integration_cell_id");

  // Get the gauss rule of the background cut element
  std::vector<Core::FE::GaussIntegration> gp_intpoints_cut;

  Cut::ElementHandle* element_handle = get_cutwizard()->get_element(&element_2());

  element_handle->get_gauss_rule_integration_cells(
      gp_intpoints_cut, get_cutwizard()->do_inside_cells_have_physical_meaning());

  for (auto& iter : gp_intpoints_cut)
  {
    for (Core::FE::GaussIntegration::iterator gp = iter.begin(); gp != iter.end(); ++gp)
    {
      Core::LinAlg::Matrix<3, 1> gp_projected_cutelement;

      gp_projected_cutelement(0, 0) = gp.point()[0];
      gp_projected_cutelement(1, 0) = gp.point()[1];
      gp_projected_cutelement(2, 0) = gp.point()[2];

      Core::LinAlg::Matrix<3, 1, double> point_coord(Core::LinAlg::Initialization::zero);

      map_from_parametric_to_physical_space<Background>(
          ele2pos_, gp_projected_cutelement, this->ele2pos_.element_position_, point_coord);

      // Write gauss point coordinates
      for (int i_dim = 0; i_dim < 3; ++i_dim)
        cutelement_point_coordinates.push_back(point_coord(i_dim));

      // Write the weight of the gauss point
      cutelement_weights.push_back(gp.weight());

      // Write an id() for this point for the background and the interface gauss points
      int id = element_2().id();
      cutelement_integration_cell_id.push_back(id);
    }
  }
}

template <typename Interface, typename Background, typename Mortar>
void Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairMortar<Interface, Background,
    Mortar>::
    get_projected_gauss_rule_on_interface(
        Core::IO::VisualizationData& background_integration_points_visualization_data,
        Core::IO::VisualizationData& interface_integration_points_visualization_data)
{
  // Get the vectors to be filled related to the background integration rule
  std::vector<double>& background_point_coordinates =
      background_integration_points_visualization_data.get_point_coordinates();
  std::vector<double>& background_weights =
      background_integration_points_visualization_data.get_point_data<double>("weights");
  std::vector<int>& background_integration_interface_id =
      background_integration_points_visualization_data.get_point_data<int>("interface_id");
  std::vector<int>& background_integration_background_id =
      background_integration_points_visualization_data.get_point_data<int>("background_id");
  std::vector<int>& background_integration_boundarycell_id =
      background_integration_points_visualization_data.get_point_data<int>("boundary_cell_id");

  // Get the vectors to be filled related to the interface integration rule
  std::vector<double>& interface_point_coordinates =
      interface_integration_points_visualization_data.get_point_coordinates();
  std::vector<double>& interface_weights =
      interface_integration_points_visualization_data.get_point_data<double>("weights");
  std::vector<int>& interface_integration_interface_id =
      interface_integration_points_visualization_data.get_point_data<int>("interface_id");
  std::vector<int>& interface_integration_background_id =
      interface_integration_points_visualization_data.get_point_data<int>("background_id");
  std::vector<int>& interface_integration_boundarycell_id =
      interface_integration_points_visualization_data.get_point_data<int>("boundary_cell_id");

  // The interface_integration_points_ are stored in a sequential order for each boundary cell.
  // To know from which boundary cell they were generated, we create an id for each boundary cell
  // generated by counting the gauss points in the following loop.
  int id_boundary_cell = 0;
  int gps_count = 0;

  // Start with the loop of the interface gauss points
  for (size_t it_gp = 0; it_gp < interface_integration_points_.size(); it_gp++)
  {
    auto& [xi_interface, xi_background, weight] = interface_integration_points_[it_gp];

    Core::LinAlg::Matrix<3, 1, double> interface_point_coord(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, double> background_point_coord(Core::LinAlg::Initialization::zero);

    map_from_parametric_to_physical_space<Interface>(
        ele1pos_, xi_interface, this->ele1pos_.element_position_, interface_point_coord);

    map_from_parametric_to_physical_space<Background>(
        ele2pos_, xi_background, this->ele2pos_.element_position_, background_point_coord);

    // Do check to see if the physical position of the gauss point of interface
    // and background are the same
    Core::LinAlg::Matrix<3, 1, double> vec(Core::LinAlg::Initialization::zero);
    for (int i_dim = 0; i_dim < 3; ++i_dim)
      vec(i_dim) = interface_point_coord(i_dim) - background_point_coord(i_dim);
    double norm2 = vec.norm2();

    if (norm2 > 1e-9)
    {
      FOUR_C_THROW(
          "The physical coordinates of the gauss points of the "
          "interface and background do not "
          "coincide. The difference is {}.",
          norm2);
    }

    // Write gauss point coordinates of interface
    for (int i_dim = 0; i_dim < 3; ++i_dim)
      interface_point_coordinates.push_back(interface_point_coord(i_dim));

    // Write gauss point coordinates of background
    for (int i_dim = 0; i_dim < 3; ++i_dim)
      background_point_coordinates.push_back(background_point_coord(i_dim));

    // Write the weight of the gauss point
    interface_weights.push_back(weight);

    // Write the weight of the gauss point
    background_weights.push_back(weight);

    // For the obtained Gauss points, write the global ids of the background and the interface
    // elements they come from and the number of their boundary cell
    const auto* face_element = dynamic_cast<const Core::Elements::FaceElement*>(&this->element_1());
    if (!face_element) FOUR_C_THROW("Cast to FaceElement failed!");

    background_integration_interface_id.push_back(face_element->parent_element_id());
    background_integration_background_id.push_back(element_2().id());
    background_integration_boundarycell_id.push_back(id_boundary_cell);

    interface_integration_interface_id.push_back(face_element->parent_element_id());
    interface_integration_background_id.push_back(element_2().id());
    interface_integration_boundarycell_id.push_back(id_boundary_cell);

    // Increase the gauss point count. If it's bigger than the number of gauss
    // points per boundary cell, increase the boundary cell id.
    gps_count += 1;
    if (gps_count == num_gauss_points_boundary_cell_)
    {
      gps_count = 0;
      id_boundary_cell += 1;
    }
  }
}

void get_nurbs_information(const Core::Elements::Element& interface_element,
    Core::LinAlg::Matrix<9, 1, double>& cp_weights,
    std::vector<Core::LinAlg::SerialDenseVector>& myknots,
    std::vector<Core::LinAlg::SerialDenseVector>& mypknots)
{
  const auto* face_element = dynamic_cast<const Core::Elements::FaceElement*>(&interface_element);
  if (!face_element) FOUR_C_THROW("Cast to FaceElement failed!");

  // Factor for surface orientation.
  double normalfac = 1.0;

  // Get the knots and weights for this element.
  const bool zero_size = Core::FE::Nurbs::get_knot_vector_and_weights_for_nurbs_boundary(
      &interface_element, face_element->face_master_number(), face_element->parent_element_id(),
      *(Global::Problem::instance()->get_dis("structure")), mypknots, myknots, cp_weights,
      normalfac);
  if (zero_size)
    FOUR_C_THROW(
        "get_knot_vector_and_weights_for_nurbs_boundary has to return a non "
        "zero size.");
}

template <Core::FE::CellType celldistype>
double calculate_determinant_interface_element(
    const Core::LinAlg::Matrix<2, 1>& eta, const Core::Elements::Element& interface_element)
{
  const int numnodes = Core::FE::num_nodes(celldistype);
  Core::LinAlg::Matrix<3, numnodes> xyze;

  // Get the position of the nodes of the interface element
  for (int i_dim = 0; i_dim < 3; ++i_dim)
  {
    for (int i_node = 0; i_node < numnodes; ++i_node)
      xyze(i_dim, i_node) = (interface_element.nodes()[i_node])->x()[i_dim];
  }

  Core::LinAlg::Matrix<numnodes, 1> funct;
  Core::LinAlg::Matrix<2, numnodes> deriv;

  // Evaluate the shape functions and its derivatives on eta
  if (celldistype == Core::FE::CellType::nurbs9)
  {
    Core::LinAlg::Matrix<9, 1, double> cp_weights(Core::LinAlg::Initialization::zero);
    std::vector<Core::LinAlg::SerialDenseVector> myknots(2);
    std::vector<Core::LinAlg::SerialDenseVector> mypknots(3);

    get_nurbs_information(interface_element, cp_weights, myknots, mypknots);

    Core::FE::Nurbs::nurbs_get_2d_funct_deriv(funct, deriv, eta, myknots, cp_weights, celldistype);
  }
  else
  {
    Core::FE::shape_function_2d(funct, eta(0), eta(1), celldistype);
    Core::FE::shape_function_2d_deriv1(deriv, eta(0), eta(1), celldistype);
  }

  // Calculate the metric tensor and obtain its determinant
  Core::LinAlg::Matrix<2, 2> metrictensor;
  Core::LinAlg::Matrix<2, 3> dxyzdrs;
  dxyzdrs.multiply_nt(deriv, xyze);
  metrictensor.multiply_nt(dxyzdrs, dxyzdrs);
  double determinant =
      std::sqrt(metrictensor(0, 0) * metrictensor(1, 1) - metrictensor(0, 1) * metrictensor(1, 0));

  return determinant;
}

double get_determinant_interface_element(
    Core::LinAlg::Matrix<2, 1> eta, const Core::Elements::Element& element)
{
  double determinant_interface;

  switch (element.shape())
  {
    case Core::FE::CellType::nurbs9:
    {
      determinant_interface =
          calculate_determinant_interface_element<Core::FE::CellType::nurbs9>(eta, element);
      break;
    }
    case Core::FE::CellType::quad4:
    {
      determinant_interface =
          calculate_determinant_interface_element<Core::FE::CellType::quad4>(eta, element);
      break;
    }
    default:
      FOUR_C_THROW(
          "The evaluation of the determinant hasn't been implemented "
          "for other type of "
          "elements. ");
      break;
  }

  return determinant_interface;
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

    double determinant_interface =
        get_determinant_interface_element(xi_interface, this->element_1());

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
