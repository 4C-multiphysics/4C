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
   * \brief Assemble local penalty contributions of the Nitsche method into
   * the global matrices.
   *
   * @param pair (in) The interface-to-solid pair.
   * @param discret (in) Discretization
   * @param global_penalty_interface (in/out) Global penalty contributions from the interface (from
   * the boundary layer) DOFs
   * @param global_penalty_background (in/out) Global penalty contributions from the background
   * solid DOFs
   * @param global_penalty_interface_background (in/out) Global penalty contributions from the
   * interface and background DOFs multipliers
   * @param local_stiffness_penalty_interface (in) Local penalty contributions from the interface of
   * the pair.
   * @param local_stiffness_penalty_background (in) Local penalty contributions from the background
   * of the pair.
   * @param local_stiffness_penalty_interface_background (in) Local penalty contributions from both
   * interface and background of the pair.
   */
  template <typename Interface, typename Background>
  void assemble_local_nitsche_contributions(
      const Constraints::EmbeddedMesh::SolidInteractionPair* pair,
      const Core::FE::Discretization& discret, Core::LinAlg::SparseMatrix& global_penalty_interface,
      Core::LinAlg::SparseMatrix& global_penalty_background,
      Core::LinAlg::SparseMatrix& global_penalty_interface_background,
      Core::LinAlg::SparseMatrix& global_nitsche_interface,
      Core::LinAlg::SparseMatrix& global_nitsche_background,
      Core::LinAlg::SparseMatrix& global_nitsche_interface_background,
      Core::LinAlg::FEVector<double>& global_penalty_constraint,
      Core::LinAlg::FEVector<double>& global_nitsche_constraint,
      const Core::LinAlg::Matrix<Interface::n_dof_, Interface::n_dof_, double>&
          local_stiffness_penalty_interface,
      const Core::LinAlg::Matrix<Background::n_dof_, Background::n_dof_, double>&
          local_stiffness_penalty_background,
      const Core::LinAlg::Matrix<Interface::n_dof_, Background::n_dof_, double>&
          local_stiffness_penalty_interface_background,
      const Core::LinAlg::Matrix<Interface::n_dof_, Interface::n_dof_, double>&
          local_stiffness_nitsche_interface,
      const Core::LinAlg::Matrix<Background::n_dof_, Background::n_dof_, double>&
          local_stiffness_nitsche_background,
      const Core::LinAlg::Matrix<Interface::n_dof_, Background::n_dof_, double>&
          local_stiffness_nitsche_interface_background,
      const Core::LinAlg::Matrix<Interface::n_dof_ + Background::n_dof_, 1, double>&
          local_constraint_penalty,
      const Core::LinAlg::Matrix<Interface::n_dof_ + Background::n_dof_, 1, double>&
          local_constraint_nitsche);


  template <typename Interface, typename T>
  void calculate_derivative_unit_normal_d_displacement(const T& xi,
      const GeometryPair::ElementData<Interface, double>& element_data_surface,
      Core::LinAlg::Matrix<3, Interface::n_nodes_ * 3, double>& d_unit_normal_d_disp)
  {
    // Define variables
    Core::LinAlg::Matrix<3, 1, double> normal;
    Core::LinAlg::Matrix<3, Interface::n_nodes_ * 3, double> d_normal_d_disp(
        Core::LinAlg::Initialization::zero);

    // Evaluate derivative on position w.r.t. xi
    Core::LinAlg::Matrix<3, 2, double> dinterface;
    Core::LinAlg::Matrix<3, 1, double> dinterface_dxi;
    Core::LinAlg::Matrix<3, 1, double> dinterface_deta;
    GeometryPair::evaluate_position_derivative1<Interface>(xi, element_data_surface, dinterface);
    for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
    {
      dinterface_dxi(i_dir) = dinterface(i_dir, 0);
      dinterface_deta(i_dir) = dinterface(i_dir, 1);
    }

    // Calculate the non-normalized and normalized normal on xi
    normal.cross_product(dinterface_dxi, dinterface_deta);
    double length_normal = Core::FADUtils::vector_norm(normal);

    if constexpr (std::is_same<Interface, GeometryPair::t_nurbs9>::value)
    {
      // In the NURBS case some normals have to be flipped to point outward of the volume element
      normal.scale(element_data_surface.shape_function_data_.surface_normal_factor_);
    }
    Core::LinAlg::Matrix<3, 1, double> unit_normal(normal);
    unit_normal.scale(1.0 / Core::FADUtils::vector_norm(normal));

    // Initialize matrix of shape function derivatives
    Core::LinAlg::Matrix<Interface::element_dim_, Interface::n_nodes_, double> dN(
        Core::LinAlg::Initialization::zero);
    GeometryPair::EvaluateShapeFunction<Interface>::evaluate_deriv1(
        dN, xi, element_data_surface.shape_function_data_);

    // Define directions
    Core::LinAlg::Matrix<3, 1, double> e_x(Core::LinAlg::Initialization::zero),
        e_y(Core::LinAlg::Initialization::zero), e_z(Core::LinAlg::Initialization::zero);
    e_x(0, 0) = 1.0;
    e_y(1, 0) = 1.0;
    e_z(2, 0) = 1.0;
    const std::array<Core::LinAlg::Matrix<3, 1, double>, 3> dirs = {e_x, e_y, e_z};

    // Calculate derivative of non-normalized normal w.r.t. the displacements
    for (unsigned int i_interface_node = 0; i_interface_node < Interface::n_nodes_;
        ++i_interface_node)
    {
      for (unsigned int i_dir = 0; i_dir < 3; ++i_dir)
      {
        Core::LinAlg::Matrix<3, 1, double> d_xi_vector;
        Core::LinAlg::Matrix<3, 1, double> d_eta_vector;
        d_xi_vector.cross_product(dirs[i_dir], dinterface_deta);
        d_xi_vector.scale(dN(0, i_interface_node));

        d_eta_vector.cross_product(dinterface_dxi, dirs[i_dir]);
        d_eta_vector.scale(dN(1, i_interface_node));

        d_normal_d_disp(0, i_interface_node * 3 + i_dir) = d_xi_vector(0) + d_eta_vector(0);
        d_normal_d_disp(1, i_interface_node * 3 + i_dir) = d_xi_vector(1) + d_eta_vector(1);
        d_normal_d_disp(2, i_interface_node * 3 + i_dir) = d_xi_vector(2) + d_eta_vector(2);
      }
    }

    // Define weighting matrix
    Core::LinAlg::Matrix<3, 3, double> weighting_matrix(Core::LinAlg::Initialization::zero);
    weighting_matrix(0, 0) = 1.0 - unit_normal(0) * unit_normal(0);
    weighting_matrix(1, 1) = 1.0 - unit_normal(1) * unit_normal(1);
    weighting_matrix(2, 2) = 1.0 - unit_normal(2) * unit_normal(2);
    weighting_matrix(0, 1) = unit_normal(0) * unit_normal(1);
    weighting_matrix(0, 2) = unit_normal(0) * unit_normal(2);
    weighting_matrix(1, 0) = unit_normal(1) * unit_normal(0);
    weighting_matrix(1, 2) = unit_normal(1) * unit_normal(2);
    weighting_matrix(2, 0) = unit_normal(2) * unit_normal(0);
    weighting_matrix(2, 1) = unit_normal(2) * unit_normal(1);

    // Calculate derivative of normalized normal w.r.t. the displacements
    d_unit_normal_d_disp.multiply(weighting_matrix, d_normal_d_disp);
    d_unit_normal_d_disp.scale(1 / length_normal);
  }

  /**
   * \brief Map a point from the element's parametric space to the physical space
   */
  template <typename Pointtype>
  void map_from_parametric_to_physical_space(
      GeometryPair::ElementData<Pointtype, double> element_data,
      Core::LinAlg::Matrix<Pointtype::element_dim_, 1>& point_param_space,
      Core::LinAlg::Matrix<Pointtype::n_dof_, 1, double> nodal_values,
      Core::LinAlg::Matrix<Pointtype::spatial_dim_, 1, double>& point_physical_space);

  template <typename Surface, Core::FE::CellType boundarycell_distype>
  std::shared_ptr<Core::FE::GaussPoints> project_boundary_cell_gauss_rule_on_interface(
      Cut::BoundaryCell* boundary_cell, GeometryPair::ElementData<Surface, double>& ele1pos);

  double get_determinant_interface_element(
      Core::LinAlg::Matrix<2, 1> eta, const Core::Elements::Element& element);

  template <typename Interface>
  double get_determinant_interface_element_current_conf(Core::LinAlg::Matrix<2, 1> eta,
      const Core::Elements::Element& element,
      const GeometryPair::ElementData<Interface, double>& element_data_surface);

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

}  // namespace Constraints::EmbeddedMesh

FOUR_C_NAMESPACE_CLOSE

#endif