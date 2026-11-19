// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_INTERACTION_PAIR_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_INTERACTION_PAIR_HPP

#include "4C_config.hpp"

#include "4C_constraint_framework_embeddedmesh_params.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_io_visualization_data.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>
#include <unordered_set>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SerialDenseVector;
  class SerialDenseMatrix;
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Cut
{
  class CutWizard;
  class BoundaryCell;
}  // namespace Cut
namespace GeometryPair
{
  class GeometryPairBase;
}  // namespace GeometryPair

namespace Constraints::EmbeddedMesh
{
  class SolidToSolidCouplingManager;

  class SolidInteractionPair
  {
   public:
    /*!
    \brief Constructor
    */
    SolidInteractionPair(std::shared_ptr<Core::Elements::Element> element1,
        Core::Elements::Element* element2,
        Constraints::EmbeddedMesh::EmbeddedMeshParams& params_ptr,
        std::shared_ptr<Cut::CutWizard> cutwizard_ptr,
        std::vector<std::shared_ptr<Cut::BoundaryCell>>& boundary_cells);

    /*!
    \brief Destructor
    */
    virtual ~SolidInteractionPair() = default;

    //! @name Access methods
    inline Constraints::EmbeddedMesh::EmbeddedMeshParams params() const { return params_; }

    /*!
    \brief Get first element
    */
    const Core::Elements::Element& element_1() const { return *element1_.get(); };

    /*!
    \brief Get second element
    */
    const Core::Elements::Element& element_2() const { return *element2_; };

    /*!
    \brief Get the number of segments
    */
    inline unsigned int get_num_segments() { return boundary_cells_.size(); }

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
    virtual void get_gauss_rule_in_cut_element_for_visualization(
        Core::IO::VisualizationData& cut_element_integration_points_visualization_data) = 0;

    /*!
    \brief Get special visualization for a pair. In the case of the Mortar manager, the
    vector_to_visualize is the vector of Lagrange multipliers.
    */
    virtual void get_projected_gauss_rule_on_interface_and_background_ele_for_visualization(
        Core::IO::VisualizationData& background_integration_points_visualization_data,
        Core::IO::VisualizationData& interface_integration_points_visualization_data) = 0;

    /*!
    \brief Get special visualization for a pair.
    */
    virtual void get_pair_visualization(Core::IO::VisualizationData& visualization_data,
        std::shared_ptr<Core::LinAlg::Vector<double>> vector_to_visualize,
        const Constraints::EmbeddedMesh::SolidToSolidCouplingManager* coupling_manager,
        std::shared_ptr<std::unordered_set<int>> interface_tracker) = 0;


    //! @name Evaluation methods
    /**
     * \brief Evaluate the global matrices and vectors resulting from the coupling method.
     * @param discret (in) Discretization, used to get the boundary layer GIDs.
     * @param coupling_manager (in) Manager for the specific type of coupling technique
     * @param global_matrices (in) A map of sparse matrices resulting from the coupling method.
     * @param global_vectors (in) A map of FEVectors resulting from the coupling method.
     */
    virtual void evaluate_and_assemble_coupling_contributions(
        const Core::FE::Discretization& discret,
        const Constraints::EmbeddedMesh::SolidToSolidCouplingManager* coupling_manager,
        std::map<std::string, std::shared_ptr<Core::LinAlg::SparseMatrix>>& global_matrices,
        std::map<std::string, std::shared_ptr<Core::LinAlg::FEVector<double>>>& global_vectors) = 0;

    /**
     * \brief Set the current element displacement.
     * @param discret (in) Discretization, used to get the boundary layer GIDs.
     * @param dispnp (in) Displacement at \f$t_{n}\f$
     */
    virtual void set_current_element_position(Core::FE::Discretization const& discret,
        const Core::LinAlg::Vector<double>& displacement_vector) = 0;

    //! Get the cutwizard
    std::shared_ptr<Cut::CutWizard> get_cutwizard()
    {
      if (cutwizard_ptr_ == nullptr) FOUR_C_THROW("The cut wizard hasn't been initialized!");
      return cutwizard_ptr_;
    }

    //! Get the boundary cells
    std::vector<std::shared_ptr<Cut::BoundaryCell>> get_boundary_cells() { return boundary_cells_; }

   protected:
    //! embedded mesh parameter data container
    Constraints::EmbeddedMesh::EmbeddedMeshParams params_;

    //! number of gauss points in the boundary cell
    int num_gauss_points_boundary_cell_ = 0;

   private:
    //! @name member variables
    //! first element of interacting pair
    const std::shared_ptr<Core::Elements::Element> element1_;

    //! second element of interacting pair
    const Core::Elements::Element* element2_;

    //! pointer to the cutwizard
    std::shared_ptr<Cut::CutWizard> cutwizard_ptr_ = nullptr;

    //! boundary cells that are related to this coupling pair
    std::vector<std::shared_ptr<Cut::BoundaryCell>> boundary_cells_;
  };
}  // namespace Constraints::EmbeddedMesh

FOUR_C_NAMESPACE_CLOSE

#endif