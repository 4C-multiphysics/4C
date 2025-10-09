// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_SOLID_TO_SOLID_NITSCHE_MANAGER_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_SOLID_TO_SOLID_NITSCHE_MANAGER_HPP

#include "4C_config.hpp"

#include "4C_constraint_framework_embeddedmesh_params.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>

// Forward declarations.
class Map;

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SparseMatrix;
  class FE_Vector;
}  // namespace Core::LinAlg
namespace Core::IO
{
  class VisualizationManager;
}
namespace Solid
{
  namespace TimeInt
  {
    class BaseDataGlobalState;
  }
}  // namespace Solid

namespace LinAlg
{
  class SparseMatrix;
}  // namespace LinAlg

namespace Constraints::EmbeddedMesh
{
  class SolidInteractionPair;

  class SolidToSolidNitscheManager
  {
   public:
    /**
     * \brief Standard Constructor
     *
     * @param discret (in) Pointer to the discretization.
     * @param displacement_vector (in) global displacement vector.
     */
    SolidToSolidNitscheManager(std::shared_ptr<Core::FE::Discretization>& discret,
        const Core::LinAlg::Vector<double>& displacement_vector,
        Constraints::EmbeddedMesh::EmbeddedMeshParams& embedded_mesh_coupling_params,
        std::shared_ptr<Core::IO::VisualizationManager> visualization_manager);

    /**
     * \brief This method builds the global maps for the global node / element IDs to the Lagrange
     * multiplier DOFs.
     * @param displacement_vector (in) global displacement vector.
     */
    void setup(const Core::LinAlg::Vector<double>& displacement_vector);

    /**
     * \brief Evaluate mortar coupling contributions on all pairs and assemble them into the
     * global matrices.
     * @param displacement_vector (in) global displacement vector.
     */
    void evaluate_global_coupling_contributions(
        const Core::LinAlg::Vector<double>& displacement_vector);

    /**
     *
     */
    void add_global_force_stiffness_contributions(Solid::TimeInt::BaseDataGlobalState& displacement,
        std::shared_ptr<Core::LinAlg::SparseMatrix> stiff,
        std::shared_ptr<Core::LinAlg::Vector<double>> force) const;

    /**
     * \brief Sets the current position of the elements of the embedded mesh coupling pairs
     */
    void set_state(const Core::LinAlg::Vector<double>& displacement_vector);

    /**
     * \brief Write output obtained in the embedded mesh
     */
    // void write_output(double time, int timestep_number);

    /**
     * \brief Write the integration points on the interface elements and cut elements
     * after the cut operation and save it in the visualization manager
     */
    void collect_output_integration_points();

    /**
     * \brief Scale penalty contributions from the Nitsche method with the penalty parameter
     */
    void scale_contributions_penalty_stiffness_matrices() const;

    /**
     * \brief Get the communicator associated to the mortar manager
     */
    MPI_Comm get_my_comm();

   protected:
    /**
     * \brief Throw an error if setup was not called on the object prior to this function call.
     */
    inline void check_setup() const
    {
      if (!is_setup_) FOUR_C_THROW("Setup not called on SolidToSolidMortarManager!");
    }

    /**
     * \brief Throw an error if the global maps were not build.
     */
    inline void check_global_maps() const
    {
      if (!is_global_maps_build_)
        FOUR_C_THROW("Global maps are not build in SolidToSolidMortarManager!");
    }

    /**
     * \brief Check if this node is in a cut element
     */
    bool is_cut_node(Core::Nodes::Node const& node);

   private:
    /**
     * \brief Calculate the maps for the solid interface and background dofs. The calculated
     * maps are used to complete the mortar matrices.
     */
    void set_global_maps();

    /**
     * \brief This method builds the local maps from the global multi vector created in Setup. The
     * global mortar matrices are also created.
     *
     * Since some nodes of this pair, that have Lagrange multipliers may not be on this processor,
     * we need to get the node ID to Lagrange multiplier ID form the processor that holds the
     * node. All relevant global node / element to global Lagrange multiplier maps for the given
     * contact pairs are stored in a standard maps in this object. The keys in those maps are the
     * global node / element id and the value is a vector with the corresponding Lagrange
     * multiplier gids. By doing so we only have to communicate between the ranks once per
     * timestep (to be more precise: only once for each set of contact pairs. If they do not
     * change between timesteps and do not switch rank, we can keep the created maps).
     *
     * @param displacement_vector (in) global displacement vector.
     */
    // void set_local_maps(const Core::LinAlg::Vector<double>& displacement_vector);

    //! Pointer to the discretization containing the solid and interface elements.
    std::shared_ptr<Core::FE::Discretization> discret_;

    //! Flag if setup was called.
    bool is_setup_ = false;

    //! Flag if global maps were build.
    bool is_global_maps_build_;

    //! Embedded mesh parameters.
    Constraints::EmbeddedMesh::EmbeddedMeshParams embedded_mesh_coupling_params_;

    //! Vector to background row elements that are cut
    std::vector<Core::Elements::Element*> cut_elements_col_vector_;

    //! Id of background column elements that are cut
    std::vector<int> ids_cut_elements_col_;

    //! Row map of the solid interface DOFs.
    std::shared_ptr<Core::LinAlg::Map> interface_dof_rowmap_;

    //! Row map of the solid background DOFs.
    std::shared_ptr<Core::LinAlg::Map> background_dof_rowmap_;

    //! Row map of both interface and solid background DOFs.
    std::shared_ptr<Core::LinAlg::Map> interface_and_background_dof_rowmap_;

    //! Global contributions of the penalty term associated with the interface DOFs
    std::shared_ptr<Core::LinAlg::SparseMatrix> global_penalty_interface_ = nullptr;

    //! Global contributions of the penalty term associated with the background DOFs
    std::shared_ptr<Core::LinAlg::SparseMatrix> global_penalty_background_ = nullptr;

    //! Global contributions of the penalty term associated with both interface and background DOFs
    std::shared_ptr<Core::LinAlg::SparseMatrix> global_penalty_interface_background_ = nullptr;

    //! Global constraint vector.
    std::shared_ptr<Core::LinAlg::FEVector<double>> global_constraint_ = nullptr;

    //! Vector with all contact pairs to be evaluated by this mortar manager.
    std::vector<std::shared_ptr<Constraints::EmbeddedMesh::SolidInteractionPair>>
        embedded_mesh_solid_pairs_;

    std::shared_ptr<Core::IO::VisualizationManager> visualization_manager_;
  };

}  // namespace Constraints::EmbeddedMesh
FOUR_C_NAMESPACE_CLOSE
#endif
