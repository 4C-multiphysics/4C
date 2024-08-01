/*----------------------------------------------------------------------*/
/*! \file

\brief Coupling pair element for enforcing Dirichlet boundary conditions between a 3D
surface (interface) and a 3D solid element (background mesh) using mortar shape functions for the
coupling.

\level 3
*/
/*-----------------------------------------------------------*/

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_INTERACTION_PAIR_MORTAR_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_INTERACTION_PAIR_MORTAR_HPP

#include "4C_config.hpp"

#include "4C_constraint_framework_embeddedmesh_interaction_pair.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_geometry_pair_element.hpp"
#include "4C_geometry_pair_element_evaluation_functions.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace CONSTRAINTS::EMBEDDEDMESH
{
  class SolidToSolidMortarManager;

  template <typename Interface, typename Background, typename Mortar>
  class SurfaceToBackgroundCouplingPairMortar : public SolidInteractionPair
  {
   public:
    /**
     * \brief Standard Constructor
     */
    SurfaceToBackgroundCouplingPairMortar(Teuchos::RCP<Core::Elements::Element> element1,
        Core::Elements::Element* element2,
        CONSTRAINTS::EMBEDDEDMESH::EmbeddedMeshParams& params_ptr,
        Teuchos::RCP<Core::Geo::CutWizard>& cutwizard_ptr,
        std::vector<Teuchos::RCP<Core::Geo::Cut::BoundaryCell>>& boundary_cells);

    /**
     * \brief Destructor.
     */
    ~SurfaceToBackgroundCouplingPairMortar() override{};

    //! @name Visualization methods
    void get_projected_gauss_rule_in_cut_element(
        const Core::IO::VisualizationData& cut_element_integration_points_visualization_data)
        override;

    void get_projected_gauss_rule_on_interface(int num_segment,
        const Core::IO::VisualizationData& background_integration_points_visualization_data,
        const Core::IO::VisualizationData& interface_integration_points_visualization_data)
        override;

    void get_pair_visualization(
        const Core::IO::VisualizationData& lagrange_multipliers_visualization_data,
        Teuchos::RCP<Epetra_Vector> lambda,
        const CONSTRAINTS::EMBEDDEDMESH::SolidToSolidMortarManager* mortar_manager,
        Teuchos::RCP<std::unordered_set<int>> interface_tracker) override;

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
        const CONSTRAINTS::EMBEDDEDMESH::SolidToSolidMortarManager* mortar_manager,
        Core::LinAlg::SparseMatrix& global_g_bl, Core::LinAlg::SparseMatrix& global_g_bg,
        Core::LinAlg::SparseMatrix& global_fbl_l, Core::LinAlg::SparseMatrix& global_fbg_l,
        Epetra_FEVector& global_constraint, Epetra_FEVector& global_kappa,
        Epetra_FEVector& global_lambda_active) override;

    /**
     * \brief Set the Gauss rule over the interface for element1_ and element2_.
     */
    void set_gauss_rule_for_interface_and_background();

    /**
     * \brief Update the current displacement of the interface and background elements
     */
    void set_current_element_position(
        Core::FE::Discretization const& discret, const Epetra_Vector& displacement_vector) override;

   private:
    /**
     * \brief Evaluate the local mortar matrices for this contact element pair.
     */
    void evaluate_dm(Core::LinAlg::Matrix<Mortar::n_dof_, Interface::n_dof_, double>& local_D,
        Core::LinAlg::Matrix<Mortar::n_dof_, Background::n_dof_, double>& local_M,
        Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_kappa,
        Core::LinAlg::Matrix<Mortar::n_dof_, 1, double>& local_constraint);

    //! Current nodal positions (and tangents) of the interface element.
    GEOMETRYPAIR::ElementData<Interface, double> ele1pos_;

    //! Current nodal positions (and tangents) of the background element.
    GEOMETRYPAIR::ElementData<Background, double> ele2pos_;

    //! Displacements of the interface element.
    GEOMETRYPAIR::ElementData<Interface, double> ele1dis_;

    //! Displacements of the background element.
    GEOMETRYPAIR::ElementData<Background, double> ele2dis_;

    //! integration rule over the interface for element1_ and element2_
    std::vector<std::tuple<Core::LinAlg::Matrix<2, 1>, Core::LinAlg::Matrix<3, 1>, double>>
        interface_integration_points_;
  };

  /**
   * \brief Evaluate the normal vector at the nodes of an interface element
   */
  template <typename ElementType>
  typename std::enable_if<GEOMETRYPAIR::IsSurfaceAveragedNormalsElement<ElementType>::value_>::type
  evaluate_interface_element_nodal_normals(
      GEOMETRYPAIR::ElementData<ElementType, double>& element_data_surface)
  {
    Core::LinAlg::SerialDenseMatrix nodal_coordinates =
        Core::FE::getEleNodeNumbering_nodes_paramspace(ElementType::discretization_);
    Core::LinAlg::Matrix<3, 1, double> xi(true);
    Core::LinAlg::Matrix<3, 1, double> temp_normal;
    Core::LinAlg::Matrix<ElementType::n_nodes_, 1, Core::LinAlg::Matrix<3, 1, double>> normals;

    for (size_t iter_node = 0; iter_node < ElementType::n_nodes_; iter_node++)
    {
      for (unsigned int i_dim = 0; i_dim < 2; i_dim++)
        xi(i_dim) = nodal_coordinates(i_dim, iter_node);
      GEOMETRYPAIR::EvaluateFaceNormal<ElementType>(xi, element_data_surface, temp_normal);
      for (unsigned int i_dim = 0; i_dim < 3; i_dim++)
        normals(iter_node)(i_dim) += temp_normal(i_dim);
    }

    for (size_t iter_node = 0; iter_node < ElementType::n_nodes_; iter_node++)
    {
      normals(iter_node).scale(1.0 / Core::FADUtils::VectorNorm(normals(iter_node)));
      element_data_surface.nodal_normals_(0 + 3 * iter_node) = normals(iter_node)(0);
      element_data_surface.nodal_normals_(1 + 3 * iter_node) = normals(iter_node)(1);
      element_data_surface.nodal_normals_(2 + 3 * iter_node) = normals(iter_node)(2);
    }
  }

  template <typename ElementType>
  std::enable_if_t<!GEOMETRYPAIR::IsSurfaceAveragedNormalsElement<ElementType>::value_>
  evaluate_interface_element_nodal_normals(
      GEOMETRYPAIR::ElementData<ElementType, double>& element_data_surface)
  {
  }

}  // namespace CONSTRAINTS::EMBEDDEDMESH

FOUR_C_NAMESPACE_CLOSE

#endif
