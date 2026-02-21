// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_MESHREADER_HPP
#define FOUR_C_IO_MESHREADER_HPP

#include "4C_config.hpp"

#include "4C_linalg_graph.hpp"
#include "4C_rebalance.hpp"

#include <Teuchos_ParameterList.hpp>

#include <map>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace Core::IO
{
  class InputFile;

  namespace MeshInput
  {
    template <unsigned dim>
    class Mesh;
  }

  namespace Internal
  {
    struct MeshReader;
  }


  /**
   * A class that reads a mesh from an input file and fills the given discretization objects with
   * the mesh data.
   */
  class MeshReader
  {
   public:
    /**
     * Destructor.
     */
    ~MeshReader();

    /**
     * Construct a MeshReader that reads the mesh from the given @p input file.
     * The optional @p parameters can be used to set additional options for the reader.
     * Note that you need to call attach_discretization() before calling read_and_partition().
     */
    MeshReader(
        const Core::IO::InputFile& input, Core::Rebalance::RebalanceParameters parameters = {});

    /**
     * Add a discretization to be filled with the mesh data. The @p section_prefix is used to
     * identify sections in the input file that contain mesh data.
     */
    void attach_discretization(
        std::shared_ptr<Core::FE::Discretization> dis, const std::string& section_prefix);

    /**
     * Read the data from the input file and partition the mesh.
     *
     * After calling this functions, the Discretization objects that were added to this reader
     * are filled with the mesh data.
     *
     * We can read mesh information from a separate file, directly from the input file or generate a
     * mesh.
     */
    void read_and_partition();

    /**
     * Get MPI communicator of this mesh reader.
     */
    [[nodiscard]] MPI_Comm get_comm() const;

    /**
     * Access the external mesh on rank 0. This is only available if an external mesh was actually
     * read. On ranks other than 0, this will always return a nullptr.
     */
    [[nodiscard]] const MeshInput::Mesh<3>* get_external_mesh_on_rank_zero() const;

    /**
     * Access the filtered external mesh on rank 0 that belongs to a certain discretization. This is
     * only available if an external mesh was actually read. On ranks other than 0, this will always
     * return a nullptr.
     */
    [[nodiscard]] const MeshInput::Mesh<3>* get_filtered_external_mesh_on_rank_zero(
        const Core::FE::Discretization& dis) const;

    /**
     * @brief Get the node sets of the external mesh read by this mesh reader.
     *
     * @param node_sets map of node set IDs to vector of node IDs
     * @param node_sets_names map of node set names to the node_set_ids associated with that name
     */
    void get_node_sets(std::map<int, std::vector<int>>& node_sets,
        std::map<std::string, std::vector<int>>& node_sets_names) const;

    void get_element_block_nodes(std::map<int, std::vector<int>>& element_block_nodes) const;

   private:
    /// Communicator for this mesh reader.
    MPI_Comm comm_;

    /// Internal mesh readers.
    std::vector<std::unique_ptr<Internal::MeshReader>> mesh_readers_;

    /// The input file to read the mesh from.
    const Core::IO::InputFile& input_;

    /// Additional parameters for rebalancing meshes.
    Core::Rebalance::RebalanceParameters parameters_;

    /// The discretizations to be filled. The key is an identifier for the sections in the input.
    /// Multiple discretizations might be filled from the same section.
    std::vector<std::pair<std::string, std::shared_ptr<Core::FE::Discretization>>>
        target_discretizations_;
  };
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif
