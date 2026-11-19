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
    /**
     * \brief Write the Gauss points of the integration cells of a cut background element for
     * visualization.
     *
     * The Gauss points of the integration cells, given by the cut library in the parameter space
     * of the background element, are mapped to physical space using the reference nodal positions.
     *
     * @param cut_element_integration_points_visualization_data (in/out) Visualization data to which
     * the Gauss points are appended
     */
    void get_gauss_rule_in_cut_element_for_visualization(
        Core::IO::VisualizationData& cut_element_integration_points_visualization_data) override;

    /**
     * \brief Write the Gauss points of a coupling pair on the interface and background elements
     * for visualization.
     *
     * The Gauss points of the boundary cells, given in the parameter space of the interface element
     * and of the background element, are mapped to physical space using the reference nodal
     * positions. An error is thrown if both mappings don't give the same point.
     *
     * @param background_integration_points_visualization_data (in/out) Visualization data to
     * which the Gauss points mapped with the background element are appended
     * @param interface_integration_points_visualization_data (in) Visualization data to which
     * the Gauss points mapped with the interface element are appended
     */
    void get_projected_gauss_rule_on_interface_and_background_ele_for_visualization(
        Core::IO::VisualizationData& background_integration_points_visualization_data,
        Core::IO::VisualizationData& interface_integration_points_visualization_data) override;

    /*!
    \brief Get special visualization for a pair. In the case of the Mortar manager, the
    vector_to_visualize is the vector of Lagrange multipliers.
    */
    void get_pair_visualization(Core::IO::VisualizationData& visualization_data,
        std::shared_ptr<Core::LinAlg::Vector<double>> vector_to_visualize,
        const Constraints::EmbeddedMesh::SolidToSolidCouplingManager* coupling_manager,
        std::shared_ptr<std::unordered_set<int>> interface_tracker) override;

    //! @name Evaluation methods
    /**
     * \brief Evaluate the global matrices and vectors resulting from the coupling method.
     * @param discret (in) Discretization, used to get the boundary layer GIDs.
     * @param coupling_manager (in) Manager for the specific type of coupling technique
     * @param global_matrices (in) A map of sparse matrices resulting from the coupling method.
     * @param global_vectors (in) A map of FEVectors resulting from the coupling method.
     */
    void evaluate_and_assemble_coupling_contributions(const Core::FE::Discretization& discret,
        const Constraints::EmbeddedMesh::SolidToSolidCouplingManager* coupling_manager,
        std::map<std::string, std::shared_ptr<Core::LinAlg::SparseMatrix>>& global_matrices,
        std::map<std::string, std::shared_ptr<Core::LinAlg::FEVector<double>>>& global_vectors)
        override;

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
     */
    void evaluate_and_assemble_mortar_contributions(const Core::FE::Discretization& discret,
        const Constraints::EmbeddedMesh::SolidToSolidMortarManager* mortar_manager,
        Core::LinAlg::SparseMatrix& global_g_bl, Core::LinAlg::SparseMatrix& global_g_bg,
        Core::LinAlg::SparseMatrix& global_fbl_l, Core::LinAlg::SparseMatrix& global_fbg_l,
        Core::LinAlg::FEVector<double>& global_constraint,
        Core::LinAlg::FEVector<double>& global_kappa,
        Core::LinAlg::FEVector<double>& global_lambda_active);

    /**
     * \brief Update the current displacement of the interface and background elements
     */
    void set_current_element_position(Core::FE::Discretization const& discret,
        const Core::LinAlg::Vector<double>& displacement_vector) override;

   private:
    /**
     * \brief Evaluate the local mortar matrices for this coupling element pair.
     */
    void evaluate_dm(Core::LinAlg::Matrix<Mortar::n_dof_, Interface::n_dof_, double>& local_D,
        Core::LinAlg::Matrix<Mortar::n_dof_, Background::n_dof_, double>& local_M,
        Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_kappa,
        Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_constraint);

    //! Initial nodal positions (and tangents) of the interface element.
    GeometryPair::ElementData<Interface, double> ele1pos_;

    //! Initial nodal positions (and tangents) of the background element.
    GeometryPair::ElementData<Background, double> ele2pos_;

    //! Displacements of the interface element.
    GeometryPair::ElementData<Interface, double> ele1dis_;

    //! Displacements of the background element.
    GeometryPair::ElementData<Background, double> ele2dis_;

    //! integration rule over the interface for element1_ and element2_
    std::vector<std::tuple<Core::LinAlg::Matrix<2, 1>, Core::LinAlg::Matrix<3, 1>, double>>
        interface_integration_points_;
  };
}  // namespace Constraints::EmbeddedMesh

FOUR_C_NAMESPACE_CLOSE

#endif
