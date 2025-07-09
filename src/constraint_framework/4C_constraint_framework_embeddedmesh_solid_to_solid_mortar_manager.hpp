// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_SOLID_TO_SOLID_MORTAR_MANAGER_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_SOLID_TO_SOLID_MORTAR_MANAGER_HPP

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

  /**
   * \brief In solid to solid interactions with mortar coupling, we need to create a map with the
   * Lagrange multiplier DOFs. For now, the created DOFs can be done by:
   *  - Lagrange multiplier DOFs on nodes that are physical nodes of the system. They do not need
   * to have the same number of nodal values as the physical node or even the same dimension
   * (although in most cases the Lagrange multiplier has 3 component for each nodal value).
   *
   * By defining the Lagrange multipliers like described above, each additional DOF can be
   * identified by either the global id of the physical node it is defined on or by the global id
   * of the element it is defined on
   *
   * The start value for the Lagrange multiplier global IDs can be explicitly given. This is
   * actually the number of solid DOFs + Lagrange multipliers from other solid-to-solid couplings
   * preceding this mortar manager in the model. The Lagrange multiplier DOFs are then numbered
   * the following way, and used in \ref lambda_dof_rowmap_.
   *   - Lagrange multiplier DOFs on nodes of processor 0
   *   - Lagrange multiplier DOFs on elements of processor 0
   *   - Lagrange multiplier DOFs on nodes of processor 1
   *   - Lagrange multiplier DOFs on elements of processor 1
   *   - ...
   *
   * This class manages the connection between the created nodes and the global node / element
   * DOFs.
   */
  class SolidToSolidMortarManager
  {
   public:
    /**
     * \brief Standard Constructor
     *
     * @param discret (in) Pointer to the discretization.
     * @param displacement_vector (in) global displacement vector.
     * @param start_value_lambda_gid (in) Start value for the Lagrange multiplier global IDs.
     */
    SolidToSolidMortarManager(std::shared_ptr<Core::FE::Discretization>& discret,
        const Core::LinAlg::Vector<double>& displacement_vector,
        Constraints::EmbeddedMesh::EmbeddedMeshParams& embedded_mesh_coupling_params,
        std::shared_ptr<Core::IO::VisualizationManager> visualization_manager,
        int start_value_lambda_gid);

    /**
     * \brief This method builds the global maps for the global node / element IDs to the Lagrange
     * multiplier DOFs.
     * @param displacement_vector (in) global displacement vector.
     */
    void setup(const Core::LinAlg::Vector<double>& displacement_vector);

    /**
     * \brief Get the global IDs of all Lagrange multipliers for the interaction pair.
     * @param interaction_pair (in) pointer to interaction pair.
     * @param lambda_row (out) Standard vector with the global IDs of the Lagrange multipliers for
     * this pair.
     */
    void location_vector(const Constraints::EmbeddedMesh::SolidInteractionPair* interaction_pair,
        std::vector<int>& lambda_row) const;

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
    void add_global_force_stiffness_penalty_contributions(
        Solid::TimeInt::BaseDataGlobalState& data_state,
        std::shared_ptr<Core::LinAlg::SparseMatrix> stiff,
        std::shared_ptr<Core::LinAlg::Vector<double>> force) const;

    /**
     *
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> penalty_invert_kappa() const;

    /**
     *
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> get_global_lambda() const;

    /**
     *
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> get_global_lambda_col() const;

    /**
     * \brief Sets the current position of the elements of the embedded mesh coupling pairs
     */
    void set_state(const Core::LinAlg::Vector<double>& displacement_vector);

    /**
     * \brief Throw an error if the local maps were not build.
     */
    inline void check_local_maps() const
    {
      if (!is_local_maps_build_)
        FOUR_C_THROW("Local maps are not build in BeamToSolidMortarManager!");
    }

    /**
     * \brief Write output obtained in the embedded mesh
     */
    void write_output(double time, int timestep_number);

    /**
     * \brief Collect the results of Lagrange multipliers as runtime output for the visualization
     * manager
     */
    void collect_output_lagrange_multipliers();

    /**
     * \brief Write the integration points on the boundary elements and cut elements
     * after the cut operation and save it in the visualization manager
     */
    void collect_output_integration_points();

    /**
     * \brief Get the communicator associated to the mortar manager
     */
    MPI_Comm get_my_comm();

    /**
     * \brief Obtain the energy contribution of the embedded mesh method
     */
    double get_energy() const;

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
     * \brief Calculate the maps for the solid boundary layer and background dofs. The calculated
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
    void set_local_maps(const Core::LinAlg::Vector<double>& displacement_vector);

    //! Pointer to the discretization containing the solid and interface elements.
    std::shared_ptr<Core::FE::Discretization> discret_;

    //! Flag if setup was called.
    bool is_setup_ = false;

    //! Flag if local maps were build.
    bool is_local_maps_build_ = false;

    //! The start value for the Lagrange multiplier global IDs.
    int start_value_lambda_gid_;

    //! Flag if global maps were build.
    bool is_global_maps_build_;

    //! Embedded mesh parameters.
    Constraints::EmbeddedMesh::EmbeddedMeshParams embedded_mesh_coupling_params_;

    //! Vector to background row elements that are cut
    std::vector<Core::Elements::Element*> cut_elements_col_vector_;

    //! Id of background column elements that are cut
    std::vector<int> ids_cut_elements_col_;

    //! Global constraint vector.
    std::shared_ptr<Core::LinAlg::FEVector<double>> global_constraint_ = nullptr;

    //! Number of Lagrange multiplier DOFs on a node.
    unsigned int n_lambda_node_ = 0;

    //! Row map of the additional Lagrange multiplier DOFs.
    std::shared_ptr<Core::LinAlg::Map> lambda_dof_rowmap_;

    //! Column map of the additional Lagrange multiplier DOFs.
    std::shared_ptr<Core::LinAlg::Map> lambda_dof_colmap_;

    //! Row map of the solid boundary layer DOFs.
    std::shared_ptr<Core::LinAlg::Map> boundary_layer_interface_dof_rowmap_;

    //! Row map of the solid background DOFs.
    std::shared_ptr<Core::LinAlg::Map> background_dof_rowmap_;

    //! Multivector that connects the global node IDs with the Lagrange multiplier DOF IDs.
    //! The global row ID of the multi vector is the global ID of the node that a Lagrange
    //! multiplier is defined on. The columns hold the corresponding global IDs of the Lagrange
    //! multipliers.
    std::shared_ptr<Core::LinAlg::MultiVector<double>> node_gid_to_lambda_gid_;

    //! Standard map from global node ids to global Lagrange multiplier ids, for all
    //! nodes used on this rank.
    std::map<int, std::vector<int>> node_gid_to_lambda_gid_map_;

    //! Derivative of constraint vector w.r.t the solid boundary layer DOF.
    std::shared_ptr<Core::LinAlg::SparseMatrix> global_g_bl_ = nullptr;

    //! Derivative of constraint vector w.r.t the solid background DOF.
    std::shared_ptr<Core::LinAlg::SparseMatrix> global_g_bg_ = nullptr;

    //! Derivative of the solid boundary layer coupling forces w.r.t the Lagrange multipliers.
    std::shared_ptr<Core::LinAlg::SparseMatrix> global_fbl_l_ = nullptr;

    //! Derivative of the solid background coupling forces w.r.t the Lagrange multipliers.
    std::shared_ptr<Core::LinAlg::SparseMatrix> global_fbg_l_ = nullptr;

    //! Global \f$\kappa\f$ vector. This vector is used to scale the mortar matrices. See Yang et
    //! al: Two dimensional mortar contact methods for large deformation frictional sliding (eq.
    //! 37). With this scaling correct units and pass patch tests are achieved (in the penalty
    //! case).
    std::shared_ptr<Core::LinAlg::FEVector<double>> global_kappa_ = nullptr;

    //! This vector keeps tack of all Lagrange multipliers that are active. This is needed when
    //! the kappa vector is inverted and some entries are zero, because no active contributions
    //! act on that Lagrange multiplier.
    std::shared_ptr<Core::LinAlg::FEVector<double>> global_active_lambda_ = nullptr;

    //! Vector with all contact pairs to be evaluated by this mortar manager.
    std::vector<std::shared_ptr<Constraints::EmbeddedMesh::SolidInteractionPair>>
        embedded_mesh_solid_pairs_;

    std::shared_ptr<Core::IO::VisualizationManager> visualization_manager_;
  };
}  // namespace Constraints::EmbeddedMesh

FOUR_C_NAMESPACE_CLOSE
#endif