// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_io_gmsh_reader.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_io_mesh.hpp"
#include "4C_utils_exceptions.hpp"

#include <array>
#include <iostream>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef FOUR_C_WITH_GMSH
#include <gmsh.h>
#endif

FOUR_C_NAMESPACE_OPEN

#ifdef FOUR_C_WITH_GMSH

namespace GmshImplementation
{

  const int gmsh_numbering_offset = 1;  // gmsh uses 1-based numbering

  /**
   * @brief struct to hold element data for a specific element type
   */
  struct ElementData
  {
    int gmsh_cell_type;             ///< the gmsh element type id (int)
    std::vector<int> element_tags;  ///< the gmsh element tags (ids) for all elements of that type
    std::vector<int> element_connectivities;  ///< the gmsh node connectivities for all elements of
                                              ///< that type (flattened)
  };

  /**
   * @brief Get the elements for a gmsh physical group
   *
   * @param dim The dimension of the physical group.
   * @param tag The tag of the physical group.
   * @param elementType_to_element_data_map An empty map from gmsh element type (int) to ElementData
   * struct to be filled. The resulting map collects for each element type in that group the
   * corresponding element data.
   */
  void get_elements_for_physical_group(
      const int dim, const int tag, std::map<int, ElementData>& element_type_to_element_data_map)
  {
    std::vector<int> entity_tags;
    gmsh::model::getEntitiesForPhysicalGroup(dim, tag, entity_tags);

    // loop over all entities in the physical group
    for (const auto& entity_tag : entity_tags)
    {
      // element types in the current entity
      std::vector<int> element_types;
      // element tags for each element type in the current entity
      std::vector<std::vector<size_t>> element_tags_by_type;
      // (flattened) element node tags for each element type in the current entity
      std::vector<std::vector<size_t>> node_tags_by_type;

      gmsh::model::mesh::getElements(
          element_types, element_tags_by_type, node_tags_by_type, dim, entity_tag);

      // Loop over each element type in this entity
      for (std::size_t i = 0; i < element_types.size(); ++i)
      {
        // Get reference to ElementData in the map (default-constructed if missing)
        auto& element_data = element_type_to_element_data_map[element_types[i]];

        // Reserve space in the tag and connectivity vectors
        element_data.element_tags.reserve(
            element_data.element_tags.size() + element_tags_by_type[i].size());
        element_data.element_connectivities.reserve(
            element_data.element_connectivities.size() + node_tags_by_type[i].size());

        // Append the data to the map
        element_data.element_tags.insert(element_data.element_tags.end(),
            std::make_move_iterator(element_tags_by_type[i].begin()),
            std::make_move_iterator(element_tags_by_type[i].end()));

        element_data.element_connectivities.insert(element_data.element_connectivities.end(),
            std::make_move_iterator(node_tags_by_type[i].begin()),
            std::make_move_iterator(node_tags_by_type[i].end()));
      }
    }
  }

  /**
   * @brief Convert a Gmsh element type to a 4C Core::FE::CellType.
   *
   * @param gmsh_element_type The Gmsh element type id.
   * @return Core::FE::CellType The corresponding 4C Core::FE::CellType.
   * @throws if the gmsh element type is unsupported.
   */
  Core::FE::CellType gmsh_element_type_to_core_cell_type(const int gmsh_element_type)
  {
    switch (gmsh_element_type)
    {
      case 1:
        return Core::FE::CellType::line2;
      case 2:
        return Core::FE::CellType::tri3;
      case 3:
        return Core::FE::CellType::quad4;
      case 4:
        return Core::FE::CellType::tet4;
      case 5:
        return Core::FE::CellType::hex8;
      case 6:
        return Core::FE::CellType::wedge6;
      case 7:
        return Core::FE::CellType::pyramid5;
      case 8:
        return Core::FE::CellType::line3;
      case 9:
        return Core::FE::CellType::tri6;
      case 10:
        return Core::FE::CellType::quad9;
      case 11:
        return Core::FE::CellType::tet10;
      case 12:
        return Core::FE::CellType::hex27;
      case 13:
        FOUR_C_THROW(
            "Detected a 18-node prism (wedge) element in the mesh.\n"
            "4C does not support this element type.\n"
            "You may try to use the gmsh option 'Mesh.SecondOrderIncomplete = 1' to use wedge15 "
            "instead.\n");
      case 14:
        // pyramid14, not implemented
        FOUR_C_THROW(
            "Detected a 14-node pyramid element in the mesh.\n"
            "4C does not support this element type.\n");
      case 15:
        return Core::FE::CellType::point1;
      case 16:
        return Core::FE::CellType::quad8;
      case 17:
        return Core::FE::CellType::hex20;
      case 18:
        return Core::FE::CellType::wedge15;
      case 26:
        return Core::FE::CellType::line4;
      case 27:
        return Core::FE::CellType::line5;
      case 28:
        return Core::FE::CellType::line6;
      default:
        // all other element types are not implemented
        FOUR_C_THROW("Unsupported Gmsh element type: {}", gmsh_element_type);
    }
  }
  constexpr std::array<int, 10> tet10_gmsh_node_order_in_four_c = {
      0, 1, 2, 3, 4,  // corner nodes
      5, 6, 7, 9, 8   // edge nodes
  };
  constexpr std::array<int, 20> hex20_gmsh_node_order_in_four_c = {
      0, 1, 2, 3, 4, 5, 6, 7,  // corner nodes
      8, 11, 13, 9,            // edge nodes
      10, 12, 14, 15,          //
      16, 18, 19, 17,          //
  };
  constexpr std::array<int, 27> hex27_gmsh_node_order_in_four_c = {
      0, 1, 2, 3, 4, 5, 6, 7,  // corner nodes
      8, 11, 13, 9,            // edge nodes
      10, 12, 14, 15,          //
      16, 18, 19, 17,          //
      20, 21, 23, 24, 22, 25,  // face nodes
      26                       // center node
  };
  constexpr std::array<int, 15> wedge15_gmsh_node_order_in_four_c = {
      0, 1, 2, 3, 4, 5, 6,         // corner nodes
      9, 7, 8, 10, 11, 12, 14, 13  // edge nodes
  };

  /**
   * @brief Reorders the connectivity from Gmsh node ordering to 4C node ordering.
   *
   * @param cell_type
   * @param gmsh_connectivity
   * @return std::vector<int> 4C connectivity
   */
  std::vector<int> translate_gmsh_connectivity(
      const Core::FE::CellType cell_type, const std::span<int>& gmsh_connectivity)
  {
    auto reorder = [&](const auto& gmsh_node_order_in_four_c) -> std::vector<int>
    {
      // sanity check
      FOUR_C_ASSERT(gmsh_node_order_in_four_c.size() == gmsh_connectivity.size(),
          "Mismatch in node count for cell type {}: expected {}, got {}",
          Core::FE::cell_type_to_string(cell_type), gmsh_node_order_in_four_c.size(),
          gmsh_connectivity.size());

      // initialize output connectivity
      std::vector<int> four_c_connectivity(gmsh_node_order_in_four_c.size());
      // reorder connectivity
      for (std::size_t i = 0; i < gmsh_node_order_in_four_c.size(); ++i)
      {
        four_c_connectivity[i] = gmsh_connectivity[gmsh_node_order_in_four_c[i]];
      }
      return four_c_connectivity;
    };

    switch (cell_type)
    {
      case Core::FE::CellType::tet10:
        return reorder(tet10_gmsh_node_order_in_four_c);
        break;
      case Core::FE::CellType::hex20:
        return reorder(hex20_gmsh_node_order_in_four_c);
        break;
      case Core::FE::CellType::hex27:
        return reorder(hex27_gmsh_node_order_in_four_c);
        break;
      case Core::FE::CellType::wedge15:
        return reorder(wedge15_gmsh_node_order_in_four_c);
        break;
      default:
        // in all other cases, no reordering is needed. Directly return the input connectivity.
        return {gmsh_connectivity.begin(), gmsh_connectivity.end()};
    }
  }

  /**
   * @brief Fill the nodes (coordinates) of the gmsh mesh into RawMesh.
   *
   * @param mesh The RawMesh to be filled.
   */
  void fill_nodes(Core::IO::MeshInput::RawMesh<3>& mesh)
  {
    // get all nodes before renumbering
    std::vector<std::size_t> externalNodeTags;
    std::vector<double> coord;
    size_t node_count = 0;

    {
      std::vector<double> parametricCoord;
      std::vector<std::size_t> nodeTags;
      // get node tags before renumbering
      gmsh::model::mesh::getNodes(externalNodeTags, coord, parametricCoord);
      node_count = externalNodeTags.size();
      // reorder nodes and elements to have contiguous numbering starting from 1
      gmsh::model::mesh::renumberNodes();
      // get renumbered nodes
      gmsh::model::mesh::getNodes(nodeTags, coord, parametricCoord);
    }

    mesh.external_ids = std::vector<int>(externalNodeTags.begin(), externalNodeTags.end());

    // fill points
    FOUR_C_ASSERT(coord.size() == 3 * node_count,
        "Mismatch in coordinate array size: expected {}, got {}", 3 * node_count, coord.size());
    mesh.points = std::vector<std::array<double, 3>>(node_count);
    for (size_t i = 0; i < node_count; ++i)
    {
      mesh.points[i][0] = coord[3 * i];
      mesh.points[i][1] = coord[3 * i + 1];
      mesh.points[i][2] = coord[3 * i + 2];
    }
  }

  /**
   * @brief Fill the element block corresponding to a gmsh physical group into RawMesh.
   * If multiple element types are detected in the physical group, a warning is printed and the
   * physical group is not added to the element blocks (but can still be used as a point set).
   *
   * @param dim The dimension of the physical group.
   * @param tag The tag of the physical group.
   * @param group_name The name of the physical group.
   * @param mesh The RawMesh to be filled.
   */
  void fill_element_block(const int dim, const int tag, const std::string& group_name,
      Core::IO::MeshInput::RawMesh<3>& mesh)
  {
    // fill elements blocks
    std::map<int, GmshImplementation::ElementData> elementType_to_element_data_map;

    GmshImplementation::get_elements_for_physical_group(dim, tag, elementType_to_element_data_map);

    // fill element blocks
    // loop over element types; if multiple types are found for the same physical group,
    // a warning is issued and the corresponding element block is removed
    for (auto& [element_type, element_data] : elementType_to_element_data_map)
    {
      // get cell type from Gmsh element type
      Core::FE::CellType cell_type =
          GmshImplementation::gmsh_element_type_to_core_cell_type(element_type);
      Core::IO::MeshInput::CellBlock<3> cell_block(cell_type);
      cell_block.name = group_name;
      cell_block.external_ids_ = element_data.element_tags;
      cell_block.reserve(element_data.element_tags.size());

      // decrease node tags by 1 to have 0-based indexing
      for (auto& node_id : element_data.element_connectivities)
      {
        node_id -= gmsh_numbering_offset;
      }

      for (size_t element_id = 0; element_id < element_data.element_tags.size(); ++element_id)
      {
        size_t nodes_per_element = Core::FE::num_nodes(cell_type);

        std::span<int> connectivity(
            element_data.element_connectivities.data() + element_id * nodes_per_element,
            element_data.element_connectivities.data() + (element_id + 1) * nodes_per_element);

        cell_block.add_cell(
            GmshImplementation::translate_gmsh_connectivity(cell_type, connectivity));
      }

      // tag is already a unique tag for element blocks. In the future, we might want to make the
      // internal block_id independent of the physical group tag and use a mapping instead.
      int block_id = tag;

      auto [pair_in_map, inserted] = mesh.cell_blocks.emplace(block_id, cell_block);
      if (!inserted)
      {
        std::cout << "Warning: Multiple element types detected "
                  << "in physical group " << (group_name.empty() ? "(unnamed)" : group_name)
                  << " (dim=" << dim << ", tag=" << tag << "). "
                  << "Detected at least " << Core::FE::cell_type_to_string(cell_type) << " and "
                  << Core::FE::cell_type_to_string(pair_in_map->second.cell_type) << ".\n"
                  << "This physical group is removed from element blocks, but can still be used as "
                     "a point set.\n";
        mesh.cell_blocks.erase(pair_in_map);
      }
    }
  }

  /**
   * @brief Fill the node set corresponding to a gmsh physical group into RawMesh.
   *
   * @param dim The dimension of the physical group.
   * @param tag The tag of the physical group.
   * @param group_name The name of the physical group.
   * @param mesh The RawMesh to be filled.
   */
  void fill_node_set(const int dim, const int tag, const std::string& group_name,
      Core::IO::MeshInput::RawMesh<3>& mesh)
  {
    std::vector<std::size_t> nodeTags_in_physical_group;
    {
      std::vector<double> coord_temp;  // not needed here but expected by Gmsh API
      gmsh::model::mesh::getNodesForPhysicalGroup(dim, tag, nodeTags_in_physical_group, coord_temp);
    }

    // decrease all node tags by 1 to have 0-based indexing (gmsh nodes are already contiguously
    // ordered)
    for (auto& node_id : nodeTags_in_physical_group)
    {
      node_id -= gmsh_numbering_offset;
    }

    // tag is already a unique tag for node sets. In the future, we might want to make the internal
    // node_set_id independent of the physical group tag and use a mapping instead.
    int node_set_id = tag;

    // actually fill the point set of the mesh
    mesh.point_sets[node_set_id] = Core::IO::MeshInput::PointSet{};
    mesh.point_sets[node_set_id].name = group_name;
    mesh.point_sets[node_set_id].point_ids.reserve(nodeTags_in_physical_group.size());
    mesh.point_sets[node_set_id].point_ids = std::unordered_set<int>(
        nodeTags_in_physical_group.begin(), nodeTags_in_physical_group.end());
  }

}  // namespace GmshImplementation

Core::IO::MeshInput::RawMesh<3> Core::IO::Gmsh::read_msh_file(const std::filesystem::path& msh_file)
{
  FOUR_C_ASSERT_ALWAYS(
      std::filesystem::exists(msh_file), "File {} does not exist.", msh_file.string());

  Core::IO::MeshInput::RawMesh<3> mesh{};

  gmsh::initialize();

  try
  {
    gmsh::open(msh_file.string());
  }
  catch (...)
  {  // catch all exceptions from Gmsh
    FOUR_C_THROW("Failed to open Gmsh file {}\nIs the file corrupted?\nGmsh version: {}",
        msh_file.string(), GMSH_API_VERSION);
  }

  GmshImplementation::fill_nodes(mesh);

  // read physical groups
  gmsh::vectorpair physical_group_dim_tags;
  gmsh::model::getPhysicalGroups(physical_group_dim_tags);

  // ensure uniqueness of tags or names.
  std::unordered_set<int> unique_tags;

  for (const auto& [dim, tag] : physical_group_dim_tags)
  {
    // check uniqueness of tags
    FOUR_C_ASSERT_ALWAYS(unique_tags.find(tag) == unique_tags.end(),
        "Multiple physical groups with tag={} found.\n"
        "4C requires unique physical group tags across dimensions in gmsh mesh files.",
        tag);
    // insert tag
    unique_tags.insert(tag);

    std::string group_name;
    gmsh::model::getPhysicalName(dim, tag, group_name);

    GmshImplementation::fill_node_set(dim, tag, group_name, mesh);
    GmshImplementation::fill_element_block(dim, tag, group_name, mesh);
  }

  gmsh::finalize();

  MeshInput::assert_valid(mesh);
  return mesh;
}

#else
Core::IO::MeshInput::RawMesh<3> Core::IO::Gmsh::read_msh_file(const std::filesystem::path& msh_file)
{
  FOUR_C_THROW(
      "You have to enable Gmsh to support msh file input. Reconfigure 4C with the CMake option "
      "'FOUR_C_WITH_GMSH' set to 'ON'.");
}
#endif

FOUR_C_NAMESPACE_CLOSE