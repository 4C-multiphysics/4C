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
      const Inpar::Constraints::SolidToSolidMortarShapefunctions shape_function,
      unsigned int& n_lambda_node);

  /**
   * \brief Assemble local mortar contributions from the classical mortar matrices D and M into
   * the global matrices.
   *
   * This function assumes that the mortar contributions are symmetric, i.e. global_g_b =
   * global_fb_l^T and global_g_s = global_fs_l^T.
   *
   * @param pair (in) The beam-to-solid pair.
   * @param discret (in) Discretization
   * @param mortar_manager (in) Mortar manager for the solid-to-solid condition
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
  Inpar::Constraints::SolidToSolidMortarShapefunctions define_shape_functions_lagrange_multipliers(
      Core::FE::CellType celltype);

}  // namespace Constraints::EmbeddedMesh

FOUR_C_NAMESPACE_CLOSE

#endif