// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_SOLID_TO_SOLID_UTILS_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_SOLID_TO_SOLID_UTILS_HPP

#include "4C_config.hpp"

#include "4C_constraint_framework_embeddedmesh_params.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_line_to_surface.hpp"
#include "4C_io_visualization_data.hpp"
#include "4C_linalg_fevector.hpp"

#include <memory>
#include <vector>


FOUR_C_NAMESPACE_OPEN

namespace Cut
{
  class CutWizard;
  class BoundaryCell;
  class Element;
  class Mesh;
  class Point;
  class Side;
  class VolumeCell;
}  // namespace Cut


namespace Core::LinAlg
{
  template <unsigned int rows, unsigned int cols, class ValueType>
  class Matrix;
}  // namespace Core::LinAlg

namespace Core::LinAlg
{
  class SparseMatrix;
}

namespace Constraints::EmbeddedMesh
{
  class SolidToSolidMortarManager;
  class SolidInteractionPair;

  struct BackgroundInterfaceInfo
  {
    Core::Elements::Element* background_element_ptr;
    std::set<int> interface_element_global_ids;
    std::multimap<int, Cut::BoundaryCell*> interface_ele_to_boundarycells;
  };

  /**
   * \brief Free function that prepares and performs the cut.
   * @param cutwizard (in) object of the cut library that performs the cut operation.
   * @param discret (in) Discretization
   */
  void prepare_and_perform_cut(std::shared_ptr<Cut::CutWizard> cutwizard,
      std::shared_ptr<Core::FE::Discretization>& discret,
      Constraints::EmbeddedMesh::EmbeddedMeshParams& embedded_mesh_coupling_params);

  /**
   * \brief Free function that obtains the information of a background element and
   * its interface elements and returns them in a BackgroundInterfaceInfo object.
   * The background elements in this object are owned by the current processor.
   * For further calculations, we need to obtain the background elements that
   * are ghosted in this processor. Therefore, their ids and pointers to this
   * elements are given in ids_cut_elements_col and cut_elements_col_vector, respectively.
   * @param cutwizard (in) object of the cut library that performs the cut operation.
   * @param discret (in) Discretization
   * @param ids_cut_elements_col (out) vector of global ids of column cut elements
   * @param cut_elements_col_vector (out) vector of column cut elements
   */
  std::vector<BackgroundInterfaceInfo> get_information_background_and_interface_elements(
      Cut::CutWizard& cutwizard, Core::FE::Discretization& discret,
      std::vector<int>& ids_cut_elements_col,
      std::vector<Core::Elements::Element*>& cut_elements_col_vector);

  /**
   * \brief Free function to get coupling pairs and background elements
   * @param info_background_interface_elements (out) struct that stores the information of
   * background elements and their interface elements
   * @param cutwizard (in) object of the cut library that performs the cut operation.
   * @param params_ptr (in) pointer to the embeddedmesh parameters
   * @param discret (in) solid discretization
   * @param embedded_mesh_solid_interaction_pairs (out) embedded mesh coupling pairs
   */
  void get_coupling_pairs_and_background_elements(
      std::vector<BackgroundInterfaceInfo>& info_background_interface_elements,
      std::shared_ptr<Cut::CutWizard>& cutwizard,
      Constraints::EmbeddedMesh::EmbeddedMeshParams& params_ptr, Core::FE::Discretization& discret,
      std::vector<std::shared_ptr<Constraints::EmbeddedMesh::SolidInteractionPair>>&
          embeddedmesh_coupling_pairs);

  /**
   * \brief Change integration rule of cut background elements
   * @param cut_elements_vector (in) vector of cut elements
   * @param cutwizard (in) object of the cut library that performs the cut operation.
   */
  void change_gauss_rule_of_cut_elements(
      std::vector<Core::Elements::Element*> cut_elements_vector, Cut::CutWizard& cutwizard);

  /**
   * \brief Get the number of Lagrange multiplier values corresponding to the solid nodes and
   * solid element.
   * @param shape_function (in) Mortar shape function.
   * @param n_lambda_node (out) Number of Lagrange multiplicators per node.
   */
  void mortar_shape_functions_to_number_of_lagrange_values(
      const Constraints::EmbeddedMesh::SolidToSolidMortarShapefunctions shape_function,
      unsigned int& n_lambda_node);

  /**
   * \brief Assemble local mortar contributions from the classical mortar matrices D and M into
   * the global matrices.
   *
   * This function assumes that the mortar contributions are symmetric, i.e. global_g_b =
   * global_fb_l^T and global_g_s = global_fs_l^T.
   *
   * @param pair (in) The interface-to-solid pair.
   * @param discret (in) Discretization
   * @param mortar_manager (in) Mortar manager for the interface-to-solid condition
   * @param global_g_bl (in/out) Constraint equations derived w.r.t the interface (from the
   * boundary layer) DOFs
   * @param global_g_bg (in/out) Constraint equations derived w.r.t the background solid DOFs
   * @param global_fbl_l (in/out) Interface force vector derived w.r.t the Lagrange multipliers
   * @param global_fbg_l (in/out) Background force vector derived w.r.t the Lagrange multipliers
   * @param global_constraint (in/out) Global constraint equations
   * @param global_kappa (in/out) Global penalty scaling vector equations
   * @param global_lambda_active (in/out) Global vector keeping track of active lagrange
   * multipliers
   * @param local_D (in) Local D matrix of the pair.
   * @param local_M (in) Local M matrix of the pair.
   * @param local_kappa (in) Local scaling vector of the pair.
   * @param local_constraint (in) Local constraint contributions of the pair.
   */
  template <typename Interface, typename Background, typename Mortar>
  void assemble_local_mortar_contributions(
      const Constraints::EmbeddedMesh::SolidInteractionPair* pair,
      const Core::FE::Discretization& discret,
      const Constraints::EmbeddedMesh::SolidToSolidMortarManager* mortar_manager,
      Core::LinAlg::SparseMatrix& global_g_bl, Core::LinAlg::SparseMatrix& global_g_bg,
      Core::LinAlg::SparseMatrix& global_fbl_l, Core::LinAlg::SparseMatrix& global_fbg_l,
      Core::LinAlg::FEVector<double>& global_constraint,
      Core::LinAlg::FEVector<double>& global_kappa,
      Core::LinAlg::FEVector<double>& global_lambda_active,
      const Core::LinAlg::Matrix<Mortar::n_dof_, Interface::n_dof_, double>& local_D,
      const Core::LinAlg::Matrix<Mortar::n_dof_, Background::n_dof_, double>& local_M,
      const Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_kappa,
      const Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_constraint);

  /**
   * \brief Map a point from the element's parametric space to the physical space
   */
  template <typename Pointtype>
  void map_from_parametric_to_physical_space(
      GeometryPair::ElementData<Pointtype, double> element_data,
      const Core::LinAlg::Matrix<Pointtype::element_dim_, 1>& point_param_space,
      Core::LinAlg::Matrix<Pointtype::n_dof_, 1, double> nodal_values,
      Core::LinAlg::Matrix<Pointtype::spatial_dim_, 1, double>& point_physical_space);

  double get_determinant_interface_element(
      Core::LinAlg::Matrix<2, 1> eta, const Core::Elements::Element& element);

  /**
   * \brief Evaluate the normal vector at the nodes of an interface element
   */
  template <typename ElementType>
  typename std::enable_if<GeometryPair::IsSurfaceAveragedNormalsElement<ElementType>::value_>::type
  evaluate_interface_element_nodal_normals(
      GeometryPair::ElementData<ElementType, double>& element_data_surface)
  {
    Core::LinAlg::SerialDenseMatrix nodal_coordinates =
        Core::FE::get_ele_node_numbering_nodes_paramspace(ElementType::discretization_);
    Core::LinAlg::Matrix<3, 1, double> xi(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<3, 1, double> temp_normal;
    Core::LinAlg::Matrix<ElementType::n_nodes_, 1, Core::LinAlg::Matrix<3, 1, double>> normals;

    for (size_t iter_node = 0; iter_node < ElementType::n_nodes_; iter_node++)
    {
      for (unsigned int i_dim = 0; i_dim < 2; i_dim++)
        xi(i_dim) = nodal_coordinates(i_dim, iter_node);
      GeometryPair::evaluate_face_normal<ElementType>(xi, element_data_surface, temp_normal);
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        normals(iter_node)(i_dim) += temp_normal(i_dim);
    }

    for (size_t iter_node = 0; iter_node < ElementType::n_nodes_; iter_node++)
    {
      normals(iter_node).scale(1.0 / Core::FADUtils::vector_norm(normals(iter_node)));
      element_data_surface.nodal_normals_(0 + 3 * iter_node) = normals(iter_node)(0);
      element_data_surface.nodal_normals_(1 + 3 * iter_node) = normals(iter_node)(1);
      element_data_surface.nodal_normals_(2 + 3 * iter_node) = normals(iter_node)(2);
    }
  }

  template <typename ElementType>
  std::enable_if_t<!GeometryPair::IsSurfaceAveragedNormalsElement<ElementType>::value_>
  evaluate_interface_element_nodal_normals(
      GeometryPair::ElementData<ElementType, double>& element_data_surface)
  {
  }

  /**
   * \brief Get the GIDs of the Lagrange multiplicator unknowns for a beam-to-solid pair.
   * @param mortar_manager (in) Mortar manager for the interface-to-background condition
   * @param interaction_pair (in) interface-to-background interaction pair
   * @param n_mortar_pos (in) Number of positional mortar DOFs associated with the pair
   * @param lambda_gid_pos (out) GIDs of positional mortar DOFs associated with the pair
   */
  void get_mortar_gid(const Constraints::EmbeddedMesh::SolidToSolidMortarManager* mortar_manager,
      const Constraints::EmbeddedMesh::SolidInteractionPair* interaction_pair,
      const unsigned int n_mortar_pos, std::vector<int>* lambda_gid_pos);

  bool is_interface_node(
      const Core::FE::Discretization& discretization, const Core::Nodes::Node& node);

  bool is_interface_element_surface(
      const Core::FE::Discretization& discretization, const Core::Elements::Element& ele);

  void get_current_element_displacement(Core::FE::Discretization const& discret,
      Core::Elements::Element const* ele, const Core::LinAlg::Vector<double>& displacement_vector,
      std::vector<double>& eledisp);

  Core::FE::GaussIntegration create_gauss_integration_from_collection(
      std::vector<Core::FE::GaussIntegration>& intpoints_vector);

  /**
   * \brief Returns the shape function for the mortar Lagrange multipliers.
   */
  SolidToSolidMortarShapefunctions define_shape_functions_lagrange_multipliers(
      Core::FE::CellType celltype);

  /**
   * \brief Set the Gauss points of the boundary cells onto the interface element and the
   * background element.
   * @param boundary_cells (in) Boundary cells of the interface element, obtained from the cut
   * @param ele1pos (in) Reference nodal positions of the interface element
   * @param ele2pos (in) Reference nodal positions of the background element
   * @param interface_integration_points (out) Tuples (xi_interface, xi_background, weight)
   * @param num_gauss_points_boundary_cell (out) Number of Gauss points per boundary cell
   */
  template <typename Interface, typename Background>
  void set_gauss_rule_for_interface_and_background(
      const std::vector<std::shared_ptr<Cut::BoundaryCell>>& boundary_cells,
      GeometryPair::ElementData<Interface, double>& ele1pos,
      GeometryPair::ElementData<Background, double>& ele2pos,
      std::vector<std::tuple<Core::LinAlg::Matrix<2, 1>, Core::LinAlg::Matrix<3, 1>, double>>&
          interface_integration_points,
      int& num_gauss_points_boundary_cell);

  /**
   * \brief Project the Gauss points of the boundary cells onto the interface element.
   * @param boundary_cells (in) Boundary cells of the interface element, obtained from the cut
   * @param ele1pos (in) Reference nodal positions of the interface element
   */
  template <typename Surface, Core::FE::CellType boundarycell_distype>
  std::shared_ptr<Core::FE::GaussPoints> project_boundary_cell_gauss_rule_on_interface(
      Cut::BoundaryCell* boundary_cell, GeometryPair::ElementData<Surface, double>& ele1pos);

  /**
   * \brief Write the Gauss points of a coupling pair on the interface and background elements
   * for visualization.
   *
   * The Gauss points of the boundary cells, given in the parameter space of the interface element
   * and of the background element, are mapped to physical space using the reference nodal
   * positions. An error is thrown if both mappings don't give the same point.

   * @param interface_element (in) Interface element of the pair, must be a FaceElement
   * @param background_element (in) Background element of the pair
   * @param interface_ele_pos (in) Reference nodal positions of the interface element
   * @param background_ele_pos (in) Reference nodal positions of the background element
   * @param background_integration_points_visualization_data (in/out) Visualization data to
   * which the Gauss points mapped with the background element are appended
   * @param interface_integration_points_visualization_data (in) Visualization data to which
   * the Gauss points mapped with the interface element are appended
   * @param interface_integration_points (in) Gauss points of the pair
   * @param num_gauss_points_boundary_cell (in) Number of Gauss points per boundary cell
   */
  template <typename Interface, typename Background>
  void get_projected_gauss_rule_on_interface_and_background_ele_for_visualization(
      const Core::Elements::Element& interface_element,
      const Core::Elements::Element& background_element,
      const GeometryPair::ElementData<Interface, double>& interface_ele_pos,
      const GeometryPair::ElementData<Background, double>& background_ele_pos,
      Core::IO::VisualizationData& background_integration_points_visualization_data,
      Core::IO::VisualizationData& interface_integration_points_visualization_data,
      const std::vector<std::tuple<Core::LinAlg::Matrix<2, 1>, Core::LinAlg::Matrix<3, 1>, double>>&
          interface_integration_points,
      int num_gauss_points_boundary_cell);

  /**
   * \brief Write the Gauss points of the integration cells of a cut background element for
   * visualization.
   *
   * The Gauss points of the integration cells, given by the cut library in the parameter space
   * of the background element, are mapped to physical space using the reference nodal positions.
   *
   * @param background_element (in) Background element that is cut by the interface
   * @param background_element_pos (in) Reference nodal positions of the background element
   * @param cutwizard (in) object of the cut library that performs the cut operation
   * @param cut_element_integration_points_visualization_data (in/out) Visualization data to which
   * the Gauss points are appended
   */
  template <typename Background>
  void get_gauss_rule_in_cut_element_for_visualization(
      const Core::Elements::Element& background_element,
      const GeometryPair::ElementData<Background, double>& background_element_pos,
      Cut::CutWizard& cutwizard,
      Core::IO::VisualizationData& cut_element_integration_points_visualization_data);

}  // namespace Constraints::EmbeddedMesh

FOUR_C_NAMESPACE_CLOSE

#endif