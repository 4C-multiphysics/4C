// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_EXODUS_HPP
#define FOUR_C_IO_EXODUS_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"

#include <filesystem>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO::Exodus
{
  struct ElementBlock;
  struct NodeSet;
  struct SideSet;

  /**
   * Additional parameters that are used in the constructor.
   */
  struct MeshParameters
  {
    /**
     * The ID of the first node in the mesh. This defaults to 1, since this is the default in
     * the Exodus II mesh format.
     */
    int node_start_id{1};
  };

  /**
   * A class that stores the mesh information read from an Exodus file.
   */
  class Mesh
  {
   public:
    /**
     * @brief Read a mesh from an Exodus file.
     *
     * Read the data from the Exodus file and store it in the class. The optional @p
     * mesh_data can be used to set options documented in the MeshParameters struct.
     */
    Mesh(std::filesystem::path exodus_file, bool exodus_verbosity,
        MeshParameters mesh_parameters = {});

    //! Print mesh info
    void print(std::ostream& os, bool verbose = false) const;

    //! Get number of nodes in mesh
    [[nodiscard]] std::size_t get_num_nodes() const { return nodes_.size(); }

    //! Get number of elements in mesh
    [[nodiscard]] std::size_t get_num_elements() const { return num_elem_; }

    //! Get ElementBlock map
    [[nodiscard]] const std::map<int, ElementBlock>& get_element_blocks() const
    {
      return element_blocks_;
    }

    //! Get Number of ElementBlocks
    [[nodiscard]] std::size_t get_num_element_blocks() const { return element_blocks_.size(); }

    //! Get one ElementBlock
    [[nodiscard]] const ElementBlock& get_element_block(const int id) const;

    //! Get NodeSet map
    [[nodiscard]] const std::map<int, NodeSet>& get_node_sets() const { return node_sets_; }

    //! Get one NodeSet
    [[nodiscard]] NodeSet get_node_set(const int id) const;

    //! Get SideSet map
    [[nodiscard]] std::map<int, SideSet> get_side_sets() const { return side_sets_; }

    //! Get one SideSet
    [[nodiscard]] const SideSet& get_side_set(const int id) const;

    //! Get Node map
    [[nodiscard]] const std::map<int, std::vector<double>>& get_nodes() const { return nodes_; }

    //! Get one nodal coords
    [[nodiscard]] const std::vector<double>& get_node(const int node_id) const;

   private:
    MeshParameters mesh_parameters_;

    std::map<int, std::vector<double>> nodes_;

    std::map<int, ElementBlock> element_blocks_;

    std::map<int, NodeSet> node_sets_;

    std::map<int, SideSet> side_sets_;

    //! Number of spatial dimensions in the mesh file.
    int spatial_dimension_{};

    //! number of elements
    std::size_t num_elem_{};

    //! title
    std::string title_;
  };


  /**
   * An EXODUS element block. This encodes a collection of elements of the same type.
   */
  struct ElementBlock
  {
    /**
     * The type of the elements in the element block.
     */
    FE::CellType cell_type;

    /**
     * Elements in this block. The keys are the element IDs, and the values are the IDs of the nodes
     * making up the element.
     */
    std::map<int, std::vector<int>> elements;

    /**
     * The name of the element block. This may be an empty string.
     */
    std::string name;

    /**
     * Pretty-print information about the element block.
     */
    void print(std::ostream& os, bool verbose = false) const;
  };

  /**
   * An EXODUS node set. This encodes a collection of nodes.
   */
  struct NodeSet
  {
    /**
     *  The IDs of the nodes in the node set.
     */
    std::vector<int> node_ids;

    /**
     * The name of the node set. This may be an empty string.
     */
    std::string name;

    /**
     * Pretty-print information about the node set.
     */
    void print(std::ostream& os, bool verbose = false) const;
  };

  /**
   * An EXODUS side set. This encodes a collection of sides/faces of elements.
   */
  struct SideSet
  {
    /**
     * The IDs of the nodes making up the sides of the side set.
     */
    std::map<int, std::vector<int>> sides;

    /**
     * The name of the side set. This may be an empty string.
     */
    std::string name;

    /**
     * Pretty-print information about the side set.
     */
    void print(std::ostream& os, bool verbose = false) const;
  };
}  // namespace Core::IO::Exodus

FOUR_C_NAMESPACE_CLOSE

#endif
