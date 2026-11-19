// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_framework_embeddedmesh_solid_to_solid_utils.hpp"

#include "4C_constraint_framework_embeddedmesh_interaction_pair_mortar.hpp"
#include "4C_constraint_framework_embeddedmesh_solid_to_solid_mortar_manager.hpp"
#include "4C_constraint_framework_input.hpp"
#include "4C_cut_boundarycell.hpp"
#include "4C_cut_combintersection.hpp"
#include "4C_cut_cutwizard.hpp"
#include "4C_cut_element.hpp"
#include "4C_cut_mesh.hpp"
#include "4C_cut_point.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_line_to_volume.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_solid_ele.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_xfem_condition_manager.hpp"

FOUR_C_NAMESPACE_OPEN

/*
  Free function that prepares and performs the cut.
*/
void Constraints::EmbeddedMesh::prepare_and_perform_cut(std::shared_ptr<Cut::CutWizard> cutwizard,
    std::shared_ptr<Core::FE::Discretization>& discret,
    Constraints::EmbeddedMesh::EmbeddedMeshParams& embedded_mesh_coupling_params)
{
  //! vector of all coupling discretizations, the background mesh is coupled with
  std::map<std::string, int> dofset_coupling_map;
  std::vector<std::shared_ptr<Core::FE::Discretization>> meshcoupl_dis;
  meshcoupl_dis.push_back(discret);
  std::vector<std::shared_ptr<Core::FE::Discretization>> levelsetcoupl_dis;
  const double time = 0.0;
  const int step = 1;

  // create a condition manager and set options of the cutwizard
  auto condition_manager = std::make_shared<XFEM::ConditionManager>(
      dofset_coupling_map, discret, meshcoupl_dis, levelsetcoupl_dis, time, step);

  condition_manager->init();
  condition_manager->setup();

  cutwizard->set_options(embedded_mesh_coupling_params.cut_params_,
      embedded_mesh_coupling_params.xfem_nodal_dof_set_strategy_,
      embedded_mesh_coupling_params.xfem_volume_cell_gauss_point_by_,
      embedded_mesh_coupling_params.xfem_bcell_gauss_point_by_, "invalid_file",
      embedded_mesh_coupling_params.gmsh_cut_out_, true, true,
      embedded_mesh_coupling_params.cut_screen_output_);

  // loop all mesh coupling objects
  for (int mc_idx = 0; mc_idx < condition_manager->num_mesh_coupling(); mc_idx++)
  {
    std::shared_ptr<XFEM::MeshCoupling> mc_coupl = condition_manager->get_mesh_coupling(mc_idx);

    if (!mc_coupl->cut_geometry()) continue;  // If don't cut the background mesh.

    // In this step, the information of the cutting interface is delivered to the cutwizard.
    // The information that is given is the cutter discretization (nodes) and its displacement
    // column (the IDs of the DOFs).
    cutwizard->add_cutter_state(mc_idx, mc_coupl->get_cutter_dis(), mc_coupl->get_cutter_disp_col(),
        condition_manager->get_mesh_coupling_start_gid(mc_idx));
  }

  // creating some variables for setting the background state
  std::shared_ptr<const Core::LinAlg::Vector<double>> back_disp_col;

  // set background state
  cutwizard->set_background_state(
      back_disp_col,  //!< col vector holding background ALE displacements for backdis
      condition_manager->get_level_set_field_col(),  //!< col vector holding nodal level-set values
                                                     //!< based on backdis
      condition_manager->get_level_set_coupling_gid()  //!< global side id for level-set coupling
  );

  // Initialize cut objects into the cut
  cutwizard->prepare();

  // Loop all mesh coupling objects:
  // Find corresponding marked surfaces loaded into the cut.
  for (int mc_idx = 0; mc_idx < condition_manager->num_mesh_coupling(); mc_idx++)
  {
    std::shared_ptr<XFEM::MeshCoupling> mc_coupl = condition_manager->get_mesh_coupling(mc_idx);

    if (mc_coupl->is_marked_geometry())
    {
      cutwizard->set_marked_condition_sides(
          *mc_coupl->get_cutter_dis(), condition_manager->get_mesh_coupling_start_gid(mc_idx));
    }
  }

  // performs the "CUT"
  bool include_inner = false;
  cutwizard->cut(include_inner);
}

template <typename Interface, typename Background>
void Constraints::EmbeddedMesh::set_gauss_rule_for_interface_and_background(
    const std::vector<std::shared_ptr<Cut::BoundaryCell>>& boundary_cells,
    GeometryPair::ElementData<Interface, double>& ele1pos,
    GeometryPair::ElementData<Background, double>& ele2pos,
    std::vector<std::tuple<Core::LinAlg::Matrix<2, 1>, Core::LinAlg::Matrix<3, 1>, double>>&
        interface_integration_points,
    int& num_gauss_points_boundary_cell)
{
  // Variables before iterating over boundary cells
  Core::LinAlg::Matrix<3, 1> interface_reference_position;
  Core::LinAlg::Matrix<3, 1> interface_position;
  Core::LinAlg::Matrix<3, 1> interface_displacement;
  int current_numpoints = interface_integration_points.size();

  // Iterate over the boundary cells, get their gauss points and save them
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
            Core::FE::CellType::tri3>(it_boundarycell->get(), ele1pos);

    // Save the number of gauss points per boundary cell. The check is done only in
    // the first boundary cell since all the boundary cells have the same cubature
    // degree and have the same shape (tri3)
    if (it_boundarycell == boundary_cells.begin())
      num_gauss_points_boundary_cell = gps_boundarycell->num_points();

    // Add the gauss points of the boundary cell to interface_integration_points
    interface_integration_points.resize(current_numpoints + gps_boundarycell->num_points());

    for (int it_gp = 0; it_gp < gps_boundarycell->num_points(); it_gp++)
    {
      auto& [xi_interface, xi_background, weight] =
          interface_integration_points[current_numpoints + it_gp];

      // Write the gauss points over the interface
      xi_interface(0) = gps_boundarycell->point(it_gp)[0];
      xi_interface(1) = gps_boundarycell->point(it_gp)[1];

      // Project gauss points on the background element and write them
      GeometryPair::evaluate_position(xi_interface, ele1pos, interface_position);

      GeometryPair::ProjectionResult temp_projection_result;
      GeometryPair::project_point_to_volume(
          interface_position, ele2pos, xi_background, temp_projection_result);

      // Write the weight
      weight = gps_boundarycell->weight(it_gp);
    }
    current_numpoints += gps_boundarycell->num_points();
  }
}

template <typename Pointtype>
void Constraints::EmbeddedMesh::map_from_parametric_to_physical_space(
    GeometryPair::ElementData<Pointtype, double> element_data,
    const Core::LinAlg::Matrix<Pointtype::element_dim_, 1>& point_param_space,
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

template <typename Surface, Core::FE::CellType boundarycell_distype>
std::shared_ptr<Core::FE::GaussPoints>
Constraints::EmbeddedMesh::project_boundary_cell_gauss_rule_on_interface(
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


// Helper function to obtain the information of a NURBS element, such as its knot and control points
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
      &interface_element, face_element->face_target_number(), face_element->parent_element_id(),
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

double Constraints::EmbeddedMesh::get_determinant_interface_element(
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


std::shared_ptr<Constraints::EmbeddedMesh::SolidInteractionPair> coupling_pair_mortar_factory(
    std::shared_ptr<Core::Elements::Element> interfaceele_real,
    Core::Elements::Element* background_ele,
    Constraints::EmbeddedMesh::EmbeddedMeshParams& params_ptr,
    std::shared_ptr<Cut::CutWizard>& cutwizard_ptr,
    std::vector<std::shared_ptr<Cut::BoundaryCell>>& boundary_cells)
{
  switch (interfaceele_real->shape())
  {
    case Core::FE::CellType::quad4:
    {
      switch (background_ele->shape())
      {
        case Core::FE::CellType::hex8:
        {
          return std::make_shared<Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairMortar<
              GeometryPair::t_quad4, GeometryPair::t_hex8, GeometryPair::t_quad4>>(
              interfaceele_real, background_ele, params_ptr, cutwizard_ptr, boundary_cells);
          break;
        }
        default:
          FOUR_C_THROW(
              "The interaction pairs with background element of type {} not yet implemented",
              (Core::FE::cell_type_to_string(background_ele->shape())).c_str());
          break;
      }
      break;
    }
    case Core::FE::CellType::nurbs9:
    {
      switch (background_ele->shape())
      {
        case Core::FE::CellType::hex8:
        {
          return std::make_shared<Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairMortar<
              GeometryPair::t_nurbs9, GeometryPair::t_hex8, GeometryPair::t_nurbs9>>(
              interfaceele_real, background_ele, params_ptr, cutwizard_ptr, boundary_cells);
          break;
        }
        case Core::FE::CellType::wedge6:
        {
          return std::make_shared<Constraints::EmbeddedMesh::SurfaceToBackgroundCouplingPairMortar<
              GeometryPair::t_nurbs9, GeometryPair::t_wedge6, GeometryPair::t_nurbs9>>(
              interfaceele_real, background_ele, params_ptr, cutwizard_ptr, boundary_cells);
          break;
        }
        default:
          FOUR_C_THROW(
              "The interaction pairs with background element of type {} not yet implemented",
              (Core::FE::cell_type_to_string(background_ele->shape())).c_str());
          break;
      }
      break;
    }
    default:
      FOUR_C_THROW("The interaction pairs with interface element of type {} not yet implemented",
          (Core::FE::cell_type_to_string(background_ele->shape())));
  }
}

std::shared_ptr<Constraints::EmbeddedMesh::SolidInteractionPair> generate_coupling_pair(
    std::shared_ptr<Core::Elements::Element> surface_ele, Core::Elements::Element* background_ele,
    Constraints::EmbeddedMesh::EmbeddedMeshParams& params_ptr,
    std::shared_ptr<Cut::CutWizard>& cutwizard,
    std::vector<std::shared_ptr<Cut::BoundaryCell>> coupling_pair_boundary_cells)
{
  switch (params_ptr.coupling_strategy_)
  {
    case Constraints::EmbeddedMesh::CouplingStrategy::mortar:
    {
      return coupling_pair_mortar_factory(
          surface_ele, background_ele, params_ptr, cutwizard, coupling_pair_boundary_cells);
      break;
    }
    default:
      FOUR_C_THROW("Unsupported interaction pair.");
  }
}

std::vector<Constraints::EmbeddedMesh::BackgroundInterfaceInfo>
Constraints::EmbeddedMesh::get_information_background_and_interface_elements(
    Cut::CutWizard& cutwizard, Core::FE::Discretization& discret,
    std::vector<int>& ids_cut_elements_col,
    std::vector<Core::Elements::Element*>& cut_elements_col_vector)
{
  // Declare object to store information of the background elements and their interface elements
  std::vector<Constraints::EmbeddedMesh::BackgroundInterfaceInfo> information_coupling_pairs;
  cut_elements_col_vector.clear();
  std::set<int> ids_cut_elements;

  // Perform checks before building the coupling pairs
  cutwizard.check_if_mesh_intersection_and_cut();

  // Get the mesh that represents the background mesh
  Cut::Mesh background_mesh = (cutwizard.get_intersection())->normal_mesh();

  // Get the elements inside the background mesh
  const std::map<int, std::shared_ptr<Cut::Element>>& background_elements =
      background_mesh.get_mesh_elements();

  // Do a loop to check all the elements of the background mesh, if the element is cut, then
  // create the coupling pair
  for (auto background_ele_iter = background_elements.begin();
      background_ele_iter != background_elements.end(); background_ele_iter++)
  {
    const std::shared_ptr<Cut::Element> background_element = background_ele_iter->second;

    if (background_element->is_cut())
    {
      // Get the element of the background mesh
      auto background_ele = discret.g_element(background_element->id());

      // Add this element into the vector of column cut elements
      ids_cut_elements.insert(background_element->id());

      // Add this element into the vector of cut elements if it hasn't been stored yet
      bool is_cut_ele_included =
          std::find(cut_elements_col_vector.begin(), cut_elements_col_vector.end(),
              background_ele) != cut_elements_col_vector.end();
      if (!is_cut_ele_included) cut_elements_col_vector.push_back(background_ele);

      // Check if the background element is owned by this processor, if this is not the case,
      // continue with the next element.
      if (background_ele->owner() != Core::Communication::my_mpi_rank(discret.get_comm())) continue;

      // Create a multimap of the global ids of interface elements and their corresponding
      // boundary cells for this background element
      std::set<int> unique_interface_ele_global_ids;
      std::multimap<int, Cut::BoundaryCell*> boundarycells_ids_multimap;

      for (auto volume_cell : background_element->volume_cells())
      {
        // Check if the position of the volume cell is in the outside direction of the
        // cutting interface. As the boundary cells are the same for the inside and outside
        // volume cells, they will get repeated if we save all the boundary sides of the inside
        // and outside volume cells. Therefore, we take only the boundary cells of the outside
        // volume cells, although we could also check for the inside cells.
        if (volume_cell->position() == Cut::Point::PointPosition::outside)
        {
          Cut::plain_boundarycell_set bc_temp = volume_cell->boundary_cells();

          for (auto it_boundarycell = bc_temp.begin(); it_boundarycell != bc_temp.end();
              ++it_boundarycell)
          {
            unique_interface_ele_global_ids.insert(
                (*it_boundarycell)->get_global_boundary_cell_id());
            boundarycells_ids_multimap.insert(
                {(*it_boundarycell)->get_global_boundary_cell_id(), *it_boundarycell});
          }
        }
      }

      // Save the obtained background and interface information
      BackgroundInterfaceInfo background_interface_info;
      background_interface_info.background_element_ptr = background_ele;
      background_interface_info.interface_element_global_ids = unique_interface_ele_global_ids;
      background_interface_info.interface_ele_to_boundarycells = boundarycells_ids_multimap;

      information_coupling_pairs.push_back(background_interface_info);
    }
  }

  // Allocate enough space in the vector and copy the information of the std::set into it
  ids_cut_elements_col.reserve(ids_cut_elements.size());
  std::copy(
      ids_cut_elements.begin(), ids_cut_elements.end(), std::back_inserter(ids_cut_elements_col));

  return information_coupling_pairs;
}

void Constraints::EmbeddedMesh::get_coupling_pairs_and_background_elements(
    std::vector<BackgroundInterfaceInfo>& info_background_interface_elements,
    std::shared_ptr<Cut::CutWizard>& cutwizard,
    Constraints::EmbeddedMesh::EmbeddedMeshParams& params_ptr, Core::FE::Discretization& discret,
    std::vector<std::shared_ptr<Constraints::EmbeddedMesh::SolidInteractionPair>>&
        embeddedmesh_coupling_pairs)
{
  // Iterate over the information of the background elements and their corresponding interface
  // elements
  for (auto& background_interface_info : info_background_interface_elements)
  {
    Core::Elements::Element* background_ele = background_interface_info.background_element_ptr;

    for (auto iter_interface_ele : background_interface_info.interface_element_global_ids)
    {
      Core::Elements::Element* interfaceEle = discret.g_element(iter_interface_ele);
      std::vector<std::shared_ptr<Core::Elements::Element>> interfaceEleSurfaces =
          interfaceEle->surfaces();
      if (interfaceEleSurfaces[0] == nullptr)
        FOUR_C_THROW("The interface element doesn't have surfaces defined. ");

      std::shared_ptr<Core::Elements::Element> surface_ele;
      for (const auto& interfaceEleSurface : interfaceEleSurfaces)
      {
        if (Constraints::EmbeddedMesh::is_interface_element_surface(discret, *interfaceEleSurface))
          surface_ele = interfaceEleSurface;
      }

      if (surface_ele == nullptr) FOUR_C_THROW("No face/surface was found");

      // Get the boundary cells of background element related to this coupling pair
      std::vector<std::shared_ptr<Cut::BoundaryCell>> coupling_pair_boundary_cells;
      auto& boundarycells_ids_multimap = background_interface_info.interface_ele_to_boundarycells;
      auto boundarycells = boundarycells_ids_multimap.equal_range(iter_interface_ele);

      for (auto iter = boundarycells.first; iter != boundarycells.second; ++iter)
      {
        std::shared_ptr<Cut::BoundaryCell> boundaryCell =
            Core::Utils::shared_ptr_from_ref(*iter->second);
        coupling_pair_boundary_cells.push_back(boundaryCell);
      }

      // Check that the vector of boundary cells is not empty, if it's not empty, create the
      // coupling pair
      if (coupling_pair_boundary_cells.size() != 0)
      {
        std::shared_ptr<Constraints::EmbeddedMesh::SolidInteractionPair>
            embeddedmesh_coupling_pair = generate_coupling_pair(
                surface_ele, background_ele, params_ptr, cutwizard, coupling_pair_boundary_cells);

        // Add coupling pair to the vector of coupling pairs
        embeddedmesh_coupling_pairs.push_back(embeddedmesh_coupling_pair);
      }
    }
  }
}

void Constraints::EmbeddedMesh::change_gauss_rule_of_cut_elements(
    std::vector<Core::Elements::Element*> cut_elements_vector, Cut::CutWizard& cutwizard)
{
  // loop over column elements
  for (Core::Elements::Element* cut_ele : cut_elements_vector)
  {
    // Construct the gauss rule of the cut element
    Discret::Elements::Solid<3>* solid_ele = dynamic_cast<Discret::Elements::Solid<3>*>(cut_ele);
    if (!solid_ele)
      FOUR_C_THROW(
          "This implementation of the embedded mesh method only works for new solid elements.");

    // Get the element handle. If this element is a background element, this will exist
    Cut::ElementHandle* elementHandle = cutwizard.get_element(cut_ele);
    if (!elementHandle) FOUR_C_THROW("No element handle found for this cut element");

    std::vector<Core::FE::GaussIntegration> gp_intpoints_cut;
    elementHandle->get_gauss_rule_integration_cells(
        gp_intpoints_cut, cutwizard.do_inside_cells_have_physical_meaning());

    Core::FE::GaussIntegration integration_rule =
        create_gauss_integration_from_collection(gp_intpoints_cut);

    solid_ele->set_integration_rule(integration_rule);
  }
}


void Constraints::EmbeddedMesh::mortar_shape_functions_to_number_of_lagrange_values(
    const Constraints::EmbeddedMesh::SolidToSolidMortarShapefunctions shape_function,
    unsigned int& n_lambda_node)
{
  switch (shape_function)
  {
    case Constraints::EmbeddedMesh::SolidToSolidMortarShapefunctions::quad4:
    {
      n_lambda_node = 1 * 3;
      return;
    }
    case Constraints::EmbeddedMesh::SolidToSolidMortarShapefunctions::nurbs9:
    {
      n_lambda_node = 1 * 3;
      return;
    }
    default:
      FOUR_C_THROW("Mortar shape function not implemented!");
  }
}

bool Constraints::EmbeddedMesh::is_interface_node(
    const Core::FE::Discretization& discretization, Core::Nodes::Node const& node)
{
  std::vector<const Core::Conditions::Condition*> conditions;
  discretization.get_condition("EmbeddedMeshSolidSurfCoupling", conditions);

  return std::ranges::any_of(conditions, [&node](const Core::Conditions::Condition* condition)
      { return condition->contains_node(node.id()); });
}

bool Constraints::EmbeddedMesh::is_interface_element_surface(
    const Core::FE::Discretization& discretization, const Core::Elements::Element& ele)
{
  for (int i = 0; i < ele.num_point(); i++)
    if (!Constraints::EmbeddedMesh::is_interface_node(discretization, *(ele.nodes()[i])))
      return false;

  return true;
}

void Constraints::EmbeddedMesh::get_current_element_displacement(
    Core::FE::Discretization const& discret, Core::Elements::Element const* ele,
    const Core::LinAlg::Vector<double>& displacement_vector, std::vector<double>& eledisp)
{
  std::vector<int> lm, lmowner, lmstride;

  ele->location_vector(discret, lm, lmowner, lmstride);
  eledisp = Core::FE::extract_values(displacement_vector, lm);
}

void Constraints::EmbeddedMesh::get_mortar_gid(
    const Constraints::EmbeddedMesh::SolidToSolidMortarManager* mortar_manager,
    const Constraints::EmbeddedMesh::SolidInteractionPair* interaction_pair,
    const unsigned int n_mortar_pos, std::vector<int>* lambda_gid_pos)
{
  mortar_manager->location_vector(interaction_pair, *lambda_gid_pos);
}

template <typename Interface, typename Background, typename Mortar>
void Constraints::EmbeddedMesh::assemble_local_mortar_contributions(
    const Constraints::EmbeddedMesh::SolidInteractionPair* pair,
    const Core::FE::Discretization& discret,
    const Constraints::EmbeddedMesh::SolidToSolidMortarManager* mortar_manager,
    Core::LinAlg::SparseMatrix& global_g_bl, Core::LinAlg::SparseMatrix& global_g_bg,
    Core::LinAlg::SparseMatrix& global_fbl_l, Core::LinAlg::SparseMatrix& global_fbg_l,
    Core::LinAlg::FEVector<double>& global_constraint, Core::LinAlg::FEVector<double>& global_kappa,
    Core::LinAlg::FEVector<double>& global_lambda_active,
    const Core::LinAlg::Matrix<Mortar::n_dof_, Interface::n_dof_, double>& local_D,
    const Core::LinAlg::Matrix<Mortar::n_dof_, Background::n_dof_, double>& local_M,
    const Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_kappa,
    const Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_constraint)
{
  // Get the GIDs of the Lagrange multipliers.
  std::vector<int> lambda_row;
  get_mortar_gid(mortar_manager, pair, Mortar::n_dof_, &lambda_row);

  // Get the GIDs of the interface
  std::vector<int> interface_row;
  std::vector<int> dummy_1;
  std::vector<int> dummy_2;

  pair->element_1().location_vector(discret, interface_row, dummy_1, dummy_2);

  // Get the GIDs of the background
  std::vector<int> background_row;
  dummy_1.clear();
  dummy_2.clear();
  pair->element_2().location_vector(discret, background_row, dummy_1, dummy_2);

  // Assemble into the global matrices. All contributions here are assumed to be symmetric.
  for (unsigned int i_lambda = 0; i_lambda < Mortar::n_dof_; ++i_lambda)
  {
    for (unsigned int i_interface = 0; i_interface < Interface::n_dof_; ++i_interface)
    {
      global_g_bl.fe_assemble(
          local_D(i_lambda, i_interface), lambda_row[i_lambda], interface_row[i_interface]);
      global_fbl_l.fe_assemble(
          local_D(i_lambda, i_interface), interface_row[i_interface], lambda_row[i_lambda]);
    }
    for (unsigned int i_background = 0; i_background < Background::n_dof_; ++i_background)
    {
      global_g_bg.fe_assemble(
          -local_M(i_lambda, i_background), lambda_row[i_lambda], background_row[i_background]);
      global_fbg_l.fe_assemble(
          -local_M(i_lambda, i_background), background_row[i_background], lambda_row[i_lambda]);
    }
  }

  global_kappa.sum_into_global_values(Mortar::n_dof_, lambda_row.data(), local_kappa.data());
  global_constraint.sum_into_global_values(
      Mortar::n_dof_, lambda_row.data(), local_constraint.data());

  // Set all entries in the local kappa vector to 1 and add them to the active vector
  Core::LinAlg::Matrix<Mortar::n_dof_, 1, double> local_kappa_active;
  local_kappa_active.put_scalar(1.0);

  global_lambda_active.sum_into_global_values(
      Mortar::n_dof_, lambda_row.data(), local_kappa_active.data());
}


Core::FE::GaussIntegration Constraints::EmbeddedMesh::create_gauss_integration_from_collection(
    std::vector<Core::FE::GaussIntegration>& intpoints_vector)
{
  // format as Core::FE::GaussIntegration
  std::shared_ptr<Core::FE::CollectedGaussPoints> gp =
      std::make_shared<Core::FE::CollectedGaussPoints>();

  for (auto& i_intpoints : intpoints_vector)
  {
    for (int i = 0; i < i_intpoints.num_points(); ++i)
    {
      gp->append(i_intpoints.point(i)[0], i_intpoints.point(i)[1], i_intpoints.point(i)[2],
          i_intpoints.weight(i));
    }
  }

  return Core::FE::GaussIntegration(gp);
}

template <typename Interface, typename Background>
void Constraints::EmbeddedMesh::
    get_projected_gauss_rule_on_interface_and_background_ele_for_visualization(
        const Core::Elements::Element& interface_element,
        const Core::Elements::Element& background_element,
        const GeometryPair::ElementData<Interface, double>& interface_ele_pos,
        const GeometryPair::ElementData<Background, double>& background_ele_pos,
        Core::IO::VisualizationData& background_integration_points_visualization_data,
        Core::IO::VisualizationData& interface_integration_points_visualization_data,
        const std::vector<std::tuple<Core::LinAlg::Matrix<2, 1>, Core::LinAlg::Matrix<3, 1>,
            double>>& interface_integration_points,
        int num_gauss_points_boundary_cell)
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
  for (size_t it_gp = 0; it_gp < interface_integration_points.size(); it_gp++)
  {
    auto& [xi_interface, xi_background, weight] = interface_integration_points[it_gp];

    Core::LinAlg::Matrix<3, 1, double> interface_point_coord(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, double> background_point_coord(Core::LinAlg::Initialization::zero);

    Constraints::EmbeddedMesh::map_from_parametric_to_physical_space<Interface>(interface_ele_pos,
        xi_interface, interface_ele_pos.element_position_, interface_point_coord);

    Constraints::EmbeddedMesh::map_from_parametric_to_physical_space<Background>(background_ele_pos,
        xi_background, background_ele_pos.element_position_, background_point_coord);

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
    const auto* face_element = dynamic_cast<const Core::Elements::FaceElement*>(&interface_element);
    if (!face_element) FOUR_C_THROW("Cast to FaceElement failed!");

    background_integration_interface_id.push_back(face_element->parent_element_id());
    background_integration_background_id.push_back(background_element.id());
    background_integration_boundarycell_id.push_back(id_boundary_cell);

    interface_integration_interface_id.push_back(face_element->parent_element_id());
    interface_integration_background_id.push_back(background_element.id());
    interface_integration_boundarycell_id.push_back(id_boundary_cell);

    // Increase the gauss point count. If it's bigger than the number of gauss
    // points per boundary cell, increase the boundary cell id.
    gps_count += 1;
    if (gps_count == num_gauss_points_boundary_cell)
    {
      gps_count = 0;
      id_boundary_cell += 1;
    }
  }
}

template <typename Background>
void Constraints::EmbeddedMesh::get_gauss_rule_in_cut_element_for_visualization(
    const Core::Elements::Element& background_element,
    const GeometryPair::ElementData<Background, double>& background_element_pos,
    Cut::CutWizard& cutwizard,
    Core::IO::VisualizationData& cut_element_integration_points_visualization_data)
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

  Cut::ElementHandle* element_handle = cutwizard.get_element(&background_element);

  element_handle->get_gauss_rule_integration_cells(
      gp_intpoints_cut, cutwizard.do_inside_cells_have_physical_meaning());

  for (auto& iter : gp_intpoints_cut)
  {
    for (Core::FE::GaussIntegration::iterator gp = iter.begin(); gp != iter.end(); ++gp)
    {
      Core::LinAlg::Matrix<3, 1> gp_projected_cutelement;

      gp_projected_cutelement(0, 0) = gp.point()[0];
      gp_projected_cutelement(1, 0) = gp.point()[1];
      gp_projected_cutelement(2, 0) = gp.point()[2];

      Core::LinAlg::Matrix<3, 1, double> point_coord(Core::LinAlg::Initialization::zero);

      Constraints::EmbeddedMesh::map_from_parametric_to_physical_space<Background>(
          background_element_pos, gp_projected_cutelement, background_element_pos.element_position_,
          point_coord);

      // Write gauss point coordinates
      for (int i_dim = 0; i_dim < 3; ++i_dim)
        cutelement_point_coordinates.push_back(point_coord(i_dim));

      // Write the weight of the gauss point
      cutelement_weights.push_back(gp.weight());

      // Write an id() for this point for the background and the interface gauss points
      int id = background_element.id();
      cutelement_integration_cell_id.push_back(id);
    }
  }
}


Constraints::EmbeddedMesh::SolidToSolidMortarShapefunctions
Constraints::EmbeddedMesh::define_shape_functions_lagrange_multipliers(Core::FE::CellType celltype)
{
  switch (celltype)
  {
    case Core::FE::CellType::quad4:
      return SolidToSolidMortarShapefunctions::quad4;
    case Core::FE::CellType::quad9:
      return SolidToSolidMortarShapefunctions::quad9;
    case Core::FE::CellType::nurbs9:
      return SolidToSolidMortarShapefunctions::nurbs9;
    default:
      FOUR_C_THROW("Shape functions not implemented for this type of cell.");
  }
}

/**
 * Explicit template initialization of template functions.
 */
namespace Constraints::EmbeddedMesh
{
  using namespace GeometryPair;

#define initialize_template_set_gauss_rule_for_interface_and_background(Interface, Background) \
  template void set_gauss_rule_for_interface_and_background<Interface, Background>(            \
      const std::vector<std::shared_ptr<Cut::BoundaryCell>>& boundary_cells,                   \
      GeometryPair::ElementData<Interface, double>& ele1pos,                                   \
      GeometryPair::ElementData<Background, double>& ele2pos,                                  \
      std::vector<std::tuple<Core::LinAlg::Matrix<2, 1>, Core::LinAlg::Matrix<3, 1>, double>>& \
          interface_integration_points,                                                        \
      int& num_gauss_points_boundary_cell)

  initialize_template_set_gauss_rule_for_interface_and_background(t_quad4, t_hex8);
  initialize_template_set_gauss_rule_for_interface_and_background(t_quad4, t_nurbs27);
  initialize_template_set_gauss_rule_for_interface_and_background(t_nurbs9, t_hex8);
  initialize_template_set_gauss_rule_for_interface_and_background(t_nurbs9, t_nurbs27);
  initialize_template_set_gauss_rule_for_interface_and_background(t_nurbs9, t_wedge6);

#define initialize_template_assemble_local_mortar_contributions(Interface, Background, Mortar) \
  template void assemble_local_mortar_contributions<Interface, Background, Mortar>(            \
      const Constraints::EmbeddedMesh::SolidInteractionPair* pair,                             \
      const Core::FE::Discretization& discret,                                                 \
      const Constraints::EmbeddedMesh::SolidToSolidMortarManager* mortar_manager,              \
      Core::LinAlg::SparseMatrix& global_g_bl, Core::LinAlg::SparseMatrix& global_g_bg,        \
      Core::LinAlg::SparseMatrix& global_fbl_l, Core::LinAlg::SparseMatrix& global_fbg_l,      \
      Core::LinAlg::FEVector<double>& global_constraint,                                       \
      Core::LinAlg::FEVector<double>& global_kappa,                                            \
      Core::LinAlg::FEVector<double>& global_lambda_active,                                    \
      const Core::LinAlg::Matrix<Mortar::n_dof_, Interface::n_dof_, double>& local_D,          \
      const Core::LinAlg::Matrix<Mortar::n_dof_, Background::n_dof_, double>& local_M,         \
      const Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_kappa,                      \
      const Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_constraint);

  initialize_template_assemble_local_mortar_contributions(t_quad4, t_hex8, t_quad4);
  initialize_template_assemble_local_mortar_contributions(t_nurbs9, t_hex8, t_nurbs9);
  initialize_template_assemble_local_mortar_contributions(t_quad4, t_nurbs27, t_quad4);
  initialize_template_assemble_local_mortar_contributions(t_nurbs9, t_nurbs27, t_nurbs9);
  initialize_template_assemble_local_mortar_contributions(t_nurbs9, t_wedge6, t_nurbs9);

#define initialize_template_template_project_boundary_cell_gauss_rule_on_interface( \
    Surface, boundarycell_distype)                                                  \
  template std::shared_ptr<Core::FE::GaussPoints>                                   \
      project_boundary_cell_gauss_rule_on_interface<Surface, boundarycell_distype>( \
          Cut::BoundaryCell * boundary_cell,                                        \
          GeometryPair::ElementData<Surface, double> & ele1pos);

  initialize_template_template_project_boundary_cell_gauss_rule_on_interface(
      t_quad4, Core::FE::CellType::tri3);
  initialize_template_template_project_boundary_cell_gauss_rule_on_interface(
      t_nurbs9, Core::FE::CellType::tri3);

#define initialize_template_map_from_parametric_to_physical_space(Pointtype)     \
  template void map_from_parametric_to_physical_space<Pointtype>(                \
      GeometryPair::ElementData<Pointtype, double> element_data,                 \
      const Core::LinAlg::Matrix<Pointtype::element_dim_, 1>& point_param_space, \
      Core::LinAlg::Matrix<Pointtype::n_dof_, 1, double> nodal_values,           \
      Core::LinAlg::Matrix<Pointtype::spatial_dim_, 1, double>& point_physical_space)

  initialize_template_map_from_parametric_to_physical_space(t_quad4);
  initialize_template_map_from_parametric_to_physical_space(t_nurbs9);
  initialize_template_map_from_parametric_to_physical_space(t_hex8);
  initialize_template_map_from_parametric_to_physical_space(t_wedge6);
  initialize_template_map_from_parametric_to_physical_space(t_nurbs27);

#define initialize_template_get_gauss_rule_in_cut_element_for_visualization(Background)     \
  template void get_gauss_rule_in_cut_element_for_visualization<Background>(                \
      const Core::Elements::Element&, const GeometryPair::ElementData<Background, double>&, \
      Cut::CutWizard&, Core::IO::VisualizationData&);

  initialize_template_get_gauss_rule_in_cut_element_for_visualization(t_hex8);
  initialize_template_get_gauss_rule_in_cut_element_for_visualization(t_nurbs27);
  initialize_template_get_gauss_rule_in_cut_element_for_visualization(t_wedge6);

#define initialize_template_get_projected_gauss_rule_on_interface_and_background_ele_for_visualization( \
    Interface, Background)                                                                              \
  template void get_projected_gauss_rule_on_interface_and_background_ele_for_visualization<             \
      Interface, Background>(const Core::Elements::Element&, const Core::Elements::Element&,            \
      const GeometryPair::ElementData<Interface, double>&,                                              \
      const GeometryPair::ElementData<Background, double>&, Core::IO::VisualizationData&,               \
      Core::IO::VisualizationData&,                                                                     \
      const std::vector<                                                                                \
          std::tuple<Core::LinAlg::Matrix<2, 1>, Core::LinAlg::Matrix<3, 1>, double>>&,                 \
      int);

  initialize_template_get_projected_gauss_rule_on_interface_and_background_ele_for_visualization(
      t_quad4, t_hex8);
  initialize_template_get_projected_gauss_rule_on_interface_and_background_ele_for_visualization(
      t_quad4, t_nurbs27);
  initialize_template_get_projected_gauss_rule_on_interface_and_background_ele_for_visualization(
      t_nurbs9, t_hex8);
  initialize_template_get_projected_gauss_rule_on_interface_and_background_ele_for_visualization(
      t_nurbs9, t_nurbs27);
  initialize_template_get_projected_gauss_rule_on_interface_and_background_ele_for_visualization(
      t_nurbs9, t_wedge6);

}  // namespace Constraints::EmbeddedMesh

FOUR_C_NAMESPACE_CLOSE