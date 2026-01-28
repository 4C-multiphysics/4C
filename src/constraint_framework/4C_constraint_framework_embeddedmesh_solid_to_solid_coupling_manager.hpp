// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_SOLID_TO_SOLID_COUPLING_MANAGER_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_EMBEDDEDMESH_SOLID_TO_SOLID_COUPLING_MANAGER_HPP

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

  class SolidToSolidCouplingManager
  {
   public:
    /**
     * \brief Standard Constructor
     *
     * @param discret (in) Pointer to the discretization.
     * @param embedded_mesh_coupling_params (in) embedded mesh coupling parameters
     * @param visualization_manager (in) visualization manager
     */
    SolidToSolidCouplingManager(std::shared_ptr<Core::FE::Discretization>& discret,
        Constraints::EmbeddedMesh::EmbeddedMeshParams& embedded_mesh_coupling_params,
        std::shared_ptr<Core::IO::VisualizationManager> visualization_manager);

    /**
     * \brief Destructor
     */
    virtual ~SolidToSolidCouplingManager() = default;

    /**
     * \brief This method builds the global maps
     * @param displacement_vector (in) global displacement vector.
     */
    virtual void setup(const Core::LinAlg::Vector<double>& displacement_vector) = 0;

    /**
     * \brief Evaluate coupling contributions on all pairs and assemble them into the
     * global matrices.
     * @param displacement_vector (in) global displacement vector.
     */
    virtual void evaluate_global_coupling_contributions(
        const Core::LinAlg::Vector<double>& displacement_vector) = 0;

    /**
     *
     */
    virtual void add_global_force_stiffness_contributions(
        Solid::TimeInt::BaseDataGlobalState& data_state,
        std::shared_ptr<Core::LinAlg::SparseMatrix> stiff,
        std::shared_ptr<Core::LinAlg::Vector<double>> force) const = 0;

    /**
     * \brief Write output of this coupling strategy
     */
    virtual void write_output(double time, int timestep_number) = 0;

    /**
     * \brief Sets the current position of the elements of the embedded mesh coupling pairs
     */
    void set_state(const Core::LinAlg::Vector<double>& displacement_vector);

    /**
     * \brief Write the integration points on the boundary elements and cut elements
     * after the cut operation and save it in the visualization manager
     */
    void collect_output_integration_points();

    /**
     * \brief Get the communicator associated to the coupling manager
     */
    MPI_Comm get_my_comm();

    /**
     * \brief Obtain the energy contribution of the embedded mesh method
     */
    virtual double get_energy() const = 0;

   protected:
    /**
     * \brief Calculate the maps for the solid boundary layer and background dofs.
     */
    virtual void set_global_maps() = 0;

    /**
     * \brief This method builds the local maps from the global multi vector created in Setup.
     *
     * @param displacement_vector (in) global displacement vector.
     */
    virtual void set_local_maps(const Core::LinAlg::Vector<double>& displacement_vector) = 0;

    /**
     * \brief Check if this node is in a cut element
     */
    bool is_cut_node(Core::Nodes::Node const& node);

    /**
     * \brief Throw an error if setup was not called on the object prior to this function call.
     */
    virtual void check_setup() const = 0;

    /**
     * \brief Throw an error if the local maps were not build.
     */
    virtual void check_local_maps() const = 0;

    /**
     * \brief Throw an error if the global maps were not build.
     */
    virtual void check_global_maps() const = 0;

    //! Pointer to the discretization containing the solid and interface elements.
    std::shared_ptr<Core::FE::Discretization> discret_;

    //! Flag if setup was called.
    bool is_setup_ = false;

    //! Flag if local maps were build.
    bool is_local_maps_build_ = false;

    //! Flag if global maps were build.
    bool is_global_maps_build_;

    //! Embedded mesh parameters.
    Constraints::EmbeddedMesh::EmbeddedMeshParams embedded_mesh_coupling_params_;

    //! Vector to background row elements that are cut
    std::vector<Core::Elements::Element*> cut_elements_col_vector_;

    //! Id of background column elements that are cut
    std::vector<int> ids_cut_elements_col_;

    //! Row map of the solid boundary layer DOFs.
    std::shared_ptr<Core::LinAlg::Map> boundary_layer_interface_dof_rowmap_;

    //! Row map of the solid background DOFs.
    std::shared_ptr<Core::LinAlg::Map> background_dof_rowmap_;

    //! Vector with all contact pairs to be evaluated by the manager
    std::vector<std::shared_ptr<Constraints::EmbeddedMesh::SolidInteractionPair>>
        embedded_mesh_solid_pairs_;

    std::shared_ptr<Core::IO::VisualizationManager> visualization_manager_;
  };
}  // namespace Constraints::EmbeddedMesh

FOUR_C_NAMESPACE_CLOSE
#endif