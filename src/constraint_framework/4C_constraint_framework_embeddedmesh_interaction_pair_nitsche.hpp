// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_INTERACTION_PAIR_NITSCHE_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_INTERACTION_PAIR_NITSCHE_HPP

#include "4C_config.hpp"

#include "4C_constraint_framework_embeddedmesh_interaction_pair.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Constraints::EmbeddedMesh
{
  class SolidToSolidNitscheManager;
  class SolidToSolidNitscheManager;

  template <typename Interface, typename Background>
  class SurfaceToBackgroundCouplingPairNitsche : public SolidInteractionPair
  {
   public:
    /**
     * \brief Standard Constructor
     */
    SurfaceToBackgroundCouplingPairNitsche(std::shared_ptr<Core::Elements::Element> element1,
        Core::Elements::Element* element2,
        Constraints::EmbeddedMesh::EmbeddedMeshParams& params_ptr,
        std::shared_ptr<Cut::CutWizard>& cutwizard_ptr,
        std::vector<std::shared_ptr<Cut::BoundaryCell>>& boundary_cells);

    /**
     * \brief Destructor.
     */
    ~SurfaceToBackgroundCouplingPairNitsche() override {};


    //! @name Evaluation methods
    /**
     * \brief Evaluate the global matrices and vectors resulting from Nitsche coupling.
     * @param discret (in) Discretization, used to get the interface GIDs.
     * @param nitsche_manager (in) Nitsche manager, used to get the Lagrange multiplier GIDs.
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
        Core::LinAlg::FEVector<double>& global_lambda_active) override
    {
      FOUR_C_THROW("The evaluation of Mortar contributions cannot be called from a Nitsche pair.");
    }

    void evaluate_and_assemble_nitsche_contributions(const Core::FE::Discretization& discret,
        const Constraints::EmbeddedMesh::SolidToSolidNitscheManager* nitsche_manager,
        Core::LinAlg::SparseMatrix& global_penalty_boundarylayer,
        Core::LinAlg::SparseMatrix& global_penalty_background,
        Core::LinAlg::SparseMatrix& global_penalty_boundarylayer_background,
        Core::LinAlg::SparseMatrix& global_virtual_disp_interface_stress_interface,
        Core::LinAlg::SparseMatrix& global_virtual_disp_interface_stress_background,
        Core::LinAlg::SparseMatrix& global_virtual_disp_background_stress_interface,
        Core::LinAlg::SparseMatrix& global_virtual_disp_background_stress_background,
        Core::LinAlg::FEVector<double>& global_constraint, double& nitsche_stabilization_param,
        double& nitsche_average_weight_param) override;

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
     * \brief Evaluate the local penalty contributions of the Nitsche method for this coupling
     * element pair.
     */
    void evaluate_penalty_contributions_nitsche(
        Core::LinAlg::Matrix<Interface::n_dof_, Interface::n_dof_, double>&
            local_stiffness_penalty_interface,
        Core::LinAlg::Matrix<Background::n_dof_, Background::n_dof_, double>&
            local_stiffness_penalty_background,
        Core::LinAlg::Matrix<Interface::n_dof_, Background::n_dof_, double>&
            local_stiffness_penalty_interface_background,
        Core::LinAlg::Matrix<Interface::n_dof_ + Background::n_dof_, 1, double>& local_constraint,
        double& nitsche_stabilization_param);

    void evaluate_stress_contributions_nitsche(const Core::FE::Discretization& discret,
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
        double& nitsche_average_weight_param);

    //! @name Visualization methods
    /*!
    \brief Get the Gauss points of element2_ after cut for visualization.
    */
    void get_projected_gauss_rule_in_cut_element(
        Core::IO::VisualizationData& cut_element_integration_points_visualization_data) override
    {
      FOUR_C_THROW(
          "This function is not implemented for the SurfaceToBackgroundCouplingPairNitsche.");
    }

    /*!
    \brief Get the Gauss points of element1_ and element2_ for the evaluation of mortar matrices.
    */
    void get_projected_gauss_rule_on_interface(
        Core::IO::VisualizationData& background_integration_points_visualization_data,
        Core::IO::VisualizationData& interface_integration_points_visualization_data) override
    {
      FOUR_C_THROW(
          "This function is not implemented for the SurfaceToBackgroundCouplingPairNitsche.");
    }

    /*!
    \brief Get the Lagrange multiplier field evaluated on the interface nodes for visualization.
    */
    void get_pair_visualization(
        Core::IO::VisualizationData& lagrange_multipliers_visualization_data,
        std::shared_ptr<Core::LinAlg::Vector<double>> lambda,
        const Constraints::EmbeddedMesh::SolidToSolidMortarManager* mortar_manager,
        std::shared_ptr<std::unordered_set<int>> interface_tracker) override
    {
      FOUR_C_THROW(
          "This function is not implemented for the SurfaceToBackgroundCouplingPairNitsche.");
    }

    //! Initial nodal positions (and tangents) of the interface element.
    GeometryPair::ElementData<Interface, double> ele1pos_;

    //! Initial nodal positions (and tangents) of the background element.
    GeometryPair::ElementData<Background, double> ele2pos_;

    //! Current nodal positions (and tangents) of the interface element.
    GeometryPair::ElementData<Interface, double> ele1pos_current_;

    //! Current nodal positions (and tangents) of the background element.
    GeometryPair::ElementData<Background, double> ele2pos_current_;

    //! Displacements of the interface element.
    GeometryPair::ElementData<Interface, double> ele1dis_;

    //! Displacements of the parent element of the interface element
    std::vector<double> ele1_parent_dis_;

    //! Displacements of the background element.
    GeometryPair::ElementData<Background, double> ele2dis_;

    //! integration rule over the interface for element1_ and element2_
    std::vector<std::tuple<Core::LinAlg::Matrix<2, 1>, Core::LinAlg::Matrix<3, 1>, double>>
        interface_integration_points_;
  };
}  // namespace Constraints::EmbeddedMesh

FOUR_C_NAMESPACE_CLOSE

#endif
