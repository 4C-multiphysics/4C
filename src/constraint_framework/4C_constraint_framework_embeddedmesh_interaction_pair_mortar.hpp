// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_INTERACTION_PAIR_MORTAR_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_INTERACTION_PAIR_MORTAR_HPP

#include "4C_config.hpp"

#include "4C_constraint_framework_embeddedmesh_interaction_pair.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Constraints::EmbeddedMesh
{
  class SolidToSolidMortarManager;

  template <typename Interface, typename Background, typename Mortar>
  class SurfaceToBackgroundCouplingPairMortar : public SolidInteractionPair
  {
   public:
    /**
     * \brief Standard Constructor
     */
    SurfaceToBackgroundCouplingPairMortar(std::shared_ptr<Core::Elements::Element> element1,
        Core::Elements::Element* element2,
        Constraints::EmbeddedMesh::EmbeddedMeshParams& params_ptr,
        std::shared_ptr<Cut::CutWizard>& cutwizard_ptr,
        std::vector<std::shared_ptr<Cut::BoundaryCell>>& boundary_cells);

    /**
     * \brief Destructor.
     */
    ~SurfaceToBackgroundCouplingPairMortar() override {};

    //! @name Visualization methods
    void get_projected_gauss_rule_in_cut_element(
        Core::IO::VisualizationData& cut_element_integration_points_visualization_data) override;

    void get_projected_gauss_rule_on_interface(
        Core::IO::VisualizationData& background_integration_points_visualization_data,
        Core::IO::VisualizationData& interface_integration_points_visualization_data) override;

    void get_pair_visualization(
        Core::IO::VisualizationData& lagrange_multipliers_visualization_data,
        std::shared_ptr<Core::LinAlg::Vector<double>> lambda,
        const Constraints::EmbeddedMesh::SolidToSolidMortarManager* mortar_manager,
        std::shared_ptr<std::unordered_set<int>> interface_tracker) override;

    //! @name Evaluation methods
    /**
     * \brief Evaluate the global matrices and vectors resulting from mortar coupling.
     * @param discret (in) Discretization, used to get the interface GIDs.
     * @param mortar_manager (in) Mortar manager, used to get the Lagrange multiplier GIDs.
     * @param global_g_bl (in/out) Constraint equations derived w.r.t the interface DOFs.
     * @param global_g_bg (in/out) Constraint equations derived w.r.t the background DOFs.
     * @param global_fbl_l (in/out) Interface force vector derived w.r.t the Lagrange multipliers.
     * @param global_fbg_l (in/out) Background force vector derived w.r.t the Lagrange multipliers.
     * @param global_constraint (in/out) Global constraint vector.
     * @param global_kappa (in/out) Global scaling matrix.
     * @param global_lambda_active (in/out) Global vector with active Lagrange multipliers.
     * @param displacement_vector (in) Global displacement vector.
     */
    void evaluate_and_assemble_mortar_contributions(const Core::FE::Discretization& discret,
        const Constraints::EmbeddedMesh::SolidToSolidMortarManager* mortar_manager,
        Core::LinAlg::SparseMatrix& global_g_bl, Core::LinAlg::SparseMatrix& global_g_bg,
        Core::LinAlg::SparseMatrix& global_fbl_l, Core::LinAlg::SparseMatrix& global_fbg_l,
        Core::LinAlg::FEVector<double>& global_constraint,
        Core::LinAlg::FEVector<double>& global_kappa,
        Core::LinAlg::FEVector<double>& global_lambda_active) override;

    /**
     * \brief Set the Gauss rule over the interface for element1_ and element2_.
     */
    void set_gauss_rule_for_interface_and_background();

    /**
     * \brief Update the current displacement of the interface and background elements
     */
    void set_current_element_position(Core::FE::Discretization const& discret,
        const Core::LinAlg::Vector<double>& displacement_vector) override;

   private:
    /**
     * \brief Evaluate the local mortar matrices for this contact element pair.
     */
    void evaluate_dm(Core::LinAlg::Matrix<Mortar::n_dof_, Interface::n_dof_, double>& local_D,
        Core::LinAlg::Matrix<Mortar::n_dof_, Background::n_dof_, double>& local_M,
        Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_kappa,
        Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_constraint);

    //! Current nodal positions (and tangents) of the interface element.
    GeometryPair::ElementData<Interface, double> ele1pos_;

    //! Current nodal positions (and tangents) of the background element.
    GeometryPair::ElementData<Background, double> ele2pos_;

    //! Displacements of the interface element.
    GeometryPair::ElementData<Interface, double> ele1dis_;

    //! Displacements of the background element.
    GeometryPair::ElementData<Background, double> ele2dis_;

    //! integration rule over the interface for element1_ and element2_
    std::vector<std::tuple<Core::LinAlg::Matrix<2, 1>, Core::LinAlg::Matrix<3, 1>, double>>
        interface_integration_points_;
  };

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

}  // namespace Constraints::EmbeddedMesh

FOUR_C_NAMESPACE_CLOSE

#endif
