// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_exodus.hpp"

#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_io_pstream.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_std23_unreachable.hpp"

#include <exodusII.h>
#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <fstream>
#include <ranges>
#include <set>

FOUR_C_NAMESPACE_OPEN

// Helper macro to always check the return value of an Exodus call.
#define CHECK_EXODUS_CALL(call)                                                                \
  do                                                                                           \
  {                                                                                            \
    int error = call;                                                                          \
    if (error < 0)                                                                             \
    {                                                                                          \
      /* Use EnumTools for a direct interpretation of the error value. */                      \
      auto error_enum = static_cast<ex_error_return_code>(error);                              \
      FOUR_C_THROW("Exodus II failed with error code {}: {}", error_enum, #call);              \
    }                                                                                          \
    if (error > 0)                                                                             \
    {                                                                                          \
      Core::IO::cout << std::format("Exodus II returned a warning code {}: {}", error, #call); \
    }                                                                                          \
  } while (false)


namespace
{
  [[maybe_unused]] constexpr std::size_t exodus_max_str_length = MAX_STR_LENGTH;
  [[maybe_unused]] constexpr std::size_t exodus_max_line_length = MAX_LINE_LENGTH;

  Core::FE::CellType cell_type_from_exodus_string(const std::string& shape_exodus)
  {
    if (shape_exodus == "SPHERE")
      return Core::FE::CellType::point1;
    else if (shape_exodus == "QUAD4")
      return Core::FE::CellType::quad4;
    else if (shape_exodus == "QUAD8")
      return Core::FE::CellType::quad8;
    else if (shape_exodus == "QUAD9")
      return Core::FE::CellType::quad9;
    else if (shape_exodus == "SHELL4")
      return Core::FE::CellType::quad4;
    else if (shape_exodus == "SHELL8")
      return Core::FE::CellType::quad8;
    else if (shape_exodus == "SHELL9")
      return Core::FE::CellType::quad9;
    else if (shape_exodus == "TRI3")
      return Core::FE::CellType::tri3;
    else if (shape_exodus == "TRI6")
      return Core::FE::CellType::tri6;
    else if (shape_exodus == "HEX8")
      return Core::FE::CellType::hex8;
    else if (shape_exodus == "HEX20")
      return Core::FE::CellType::hex20;
    else if (shape_exodus == "HEX27")
      return Core::FE::CellType::hex27;
    else if (shape_exodus == "HEX")
      return Core::FE::CellType::hex8;
    else if (shape_exodus == "TET4")
      return Core::FE::CellType::tet4;
    else if (shape_exodus == "TETRA4")
      return Core::FE::CellType::tet4;
    else if (shape_exodus == "TETRA10")
      return Core::FE::CellType::tet10;
    else if (shape_exodus == "TETRA")
      return Core::FE::CellType::tet4;
    else if (shape_exodus == "WEDGE6")
      return Core::FE::CellType::wedge6;
    else if (shape_exodus == "WEDGE15")
      return Core::FE::CellType::wedge15;
    else if (shape_exodus == "WEDGE")
      return Core::FE::CellType::wedge6;
    else if (shape_exodus == "PYRAMID5")
      return Core::FE::CellType::pyramid5;
    else if (shape_exodus == "PYRAMID")
      return Core::FE::CellType::pyramid5;
    else if (shape_exodus == "BAR2")
      return Core::FE::CellType::line2;
    else if (shape_exodus == "BAR3")
      return Core::FE::CellType::line3;
    else
    {
      FOUR_C_THROW("Unknown Exodus Element Shape Name!");
    }
  }
  void reorder_nodes_exodus_to_four_c(std::vector<int>& nodes, Core::FE::CellType cell_type)
  {
    FOUR_C_ASSERT(nodes.size() == static_cast<std::size_t>(Core::FE::num_nodes(cell_type)),
        "Number of nodes (={}) does not match the number of nodes for the cell type {}.",
        nodes.size(), cell_type);
    switch (cell_type)
    {
      case Core::FE::CellType::hex27:
      {
        auto old = nodes;
        nodes[20] = old[21];
        nodes[21] = old[25];
        nodes[22] = old[24];
        nodes[23] = old[26];
        nodes[24] = old[23];
        nodes[25] = old[22];
        nodes[26] = old[20];
        break;
      }
      default:
      {
        // do nothing
      }
    }
  }
}  // namespace



Core::IO::MeshInput::Mesh Core::IO::Exodus::read_exodus_file(
    const std::filesystem::path& exodus_file, MeshParameters mesh_parameters)
{
  Core::IO::MeshInput::Mesh mesh{};

  int CPU_word_size, IO_word_size;
  float exoversion;               /* version of exodus */
  CPU_word_size = sizeof(double); /* size of a double */
  IO_word_size = 0;               /* use what is stored in file */

  if (!std::filesystem::exists(exodus_file))
    FOUR_C_THROW("File {} does not exist.", exodus_file.string());

  int exo_handle =
      ex_open(exodus_file.c_str(), EX_READ, &CPU_word_size, &IO_word_size, &exoversion);
  if (exo_handle <= 0) FOUR_C_THROW("Error while opening EXODUS II file {}.", exodus_file.string());

  // read database parameters
  int num_elem_blk, num_node_sets, num_side_sets, num_nodes, spatial_dimension, num_elements;
  char title[MAX_LINE_LENGTH + 1];
  CHECK_EXODUS_CALL(ex_get_init(exo_handle, title, &spatial_dimension, &num_nodes, &num_elements,
      &num_elem_blk, &num_node_sets, &num_side_sets));

  // get nodal coordinates
  {
    std::vector<double> x(num_nodes);
    std::vector<double> y(num_nodes);
    std::vector<double> z(num_nodes);
    CHECK_EXODUS_CALL(ex_get_coord(exo_handle, x.data(), y.data(), z.data()));

    FOUR_C_ASSERT_ALWAYS(
        mesh_parameters.node_start_id >= 0, "Node start id must be greater than or equal to 0");

    for (int i = 0; i < num_nodes; ++i)
    {
      mesh.points[i + mesh_parameters.node_start_id] = {x[i], y[i], z[i]};
    }
  }

  // Get all ElementBlocks
  {
    std::vector<int> epropID(num_elem_blk);
    std::vector<int> ebids(num_elem_blk);
    CHECK_EXODUS_CALL(ex_get_ids(exo_handle, EX_ELEM_BLOCK, ebids.data()));
    for (int i = 0; i < num_elem_blk; ++i)
    {
      // Read Element Blocks into Map
      char mychar[exodus_max_str_length + 1];
      int num_el_in_blk, num_nod_per_elem, num_attr;
      CHECK_EXODUS_CALL(ex_get_block(exo_handle, EX_ELEM_BLOCK, ebids[i], mychar, &num_el_in_blk,
          &num_nod_per_elem, nullptr, nullptr, &num_attr));
      // prefer std::string to store element type
      std::string ele_type(mychar);

      // get ElementBlock name
      CHECK_EXODUS_CALL(ex_get_name(exo_handle, EX_ELEM_BLOCK, ebids[i], mychar));

      // prefer std::string to store name
      std::string blockname(mychar);

      // get element elements
      std::vector<int> allconn(num_nod_per_elem * num_el_in_blk);
      CHECK_EXODUS_CALL(
          ex_get_conn(exo_handle, EX_ELEM_BLOCK, ebids[i], allconn.data(), nullptr, nullptr));

      std::map<int, std::vector<int>> eleconn;

      // Compare the desired start ID to Exodus' one-based indexing to get the offset.
      const int node_offset = mesh_parameters.node_start_id - 1;
      for (int j = 0; j < num_el_in_blk; ++j)
      {
        std::vector<int>& actconn = eleconn[j];
        actconn.reserve(num_nod_per_elem);
        for (int k = 0; k < num_nod_per_elem; ++k)
        {
          actconn.push_back(allconn[k + j * num_nod_per_elem] + node_offset);
        }
        reorder_nodes_exodus_to_four_c(actconn, cell_type_from_exodus_string(ele_type));
      }

      mesh.cell_blocks.emplace(ebids[i], MeshInput::CellBlock{
                                             .cell_type = cell_type_from_exodus_string(ele_type),
                                             .cell_connectivities = eleconn,
                                         });
    }
  }  // end of element section

  // get all NodeSets
  {
    std::vector<int> npropID(num_node_sets);
    CHECK_EXODUS_CALL(ex_get_prop_array(exo_handle, EX_NODE_SET, "ID", npropID.data()));
    for (int i = 0; i < num_node_sets; ++i)
    {
      // Read NodeSet params
      int num_nodes_in_set, num_df_in_set;

      CHECK_EXODUS_CALL(
          ex_get_set_param(exo_handle, EX_NODE_SET, npropID[i], &num_nodes_in_set, &num_df_in_set));

      // get NodeSet name
      char mychar[exodus_max_str_length + 1];
      CHECK_EXODUS_CALL(ex_get_name(exo_handle, EX_NODE_SET, npropID[i], mychar));
      // prefer std::string to store name
      std::string nodesetname(mychar);

      // get nodes in node set
      std::vector<int> node_set_node_list(num_nodes_in_set);
      CHECK_EXODUS_CALL(
          ex_get_set(exo_handle, EX_NODE_SET, npropID[i], node_set_node_list.data(), nullptr));
      std::set<int> nodes_in_set;
      // Compare the desired start ID to Exodus' one-based indexing to get the offset.
      const int node_offset = mesh_parameters.node_start_id - 1;
      for (int j = 0; j < num_nodes_in_set; ++j)
        nodes_in_set.insert(node_set_node_list[j] + node_offset);
      std::vector<int> nodes_in_set_vec(nodes_in_set.begin(), nodes_in_set.end());
      MeshInput::PointSet actNodeSet{.point_ids = nodes_in_set_vec};

      mesh.point_sets.insert(std::pair<int, MeshInput::PointSet>(npropID[i], actNodeSet));
    }
  }  // end of nodeset section
  // ***************************************************************************

  // get all SideSets
  if (num_side_sets > 0)
  {
    std::vector<int> spropID(num_side_sets);
    CHECK_EXODUS_CALL(ex_get_prop_array(exo_handle, EX_SIDE_SET, "ID", spropID.data()));
    for (int i = 0; i < num_side_sets; ++i)
    {
      // get SideSet name
      char mychar[exodus_max_str_length + 1];
      CHECK_EXODUS_CALL(ex_get_name(exo_handle, EX_SIDE_SET, spropID[i], mychar));
      // prefer std::string to store name
      std::string sidesetname(mychar);

      // Read SideSet params
      int num_side_in_set, num_dist_fact_in_set;
      CHECK_EXODUS_CALL(ex_get_set_param(
          exo_handle, EX_SIDE_SET, spropID[i], &num_side_in_set, &num_dist_fact_in_set));

      // get SideSet
      std::vector<int> side_set_elem_list(num_side_in_set);
      std::vector<int> side_set_side_list(num_side_in_set);
      CHECK_EXODUS_CALL(ex_get_set(exo_handle, EX_SIDE_SET, spropID[i], side_set_elem_list.data(),
          side_set_side_list.data()));
      std::map<int, std::vector<int>> sides_in_set;
      for (int j = 0; j < num_side_in_set; ++j)
      {
        std::vector<int> side(2);  // first entry is element, second side
        side[0] = side_set_elem_list[j];
        side[1] = side_set_side_list[j];
        sides_in_set.insert(std::pair<int, std::vector<int>>(j, side));
      }

      MeshInput::SideSet actSideSet{.sides = sides_in_set};

      // Add this SideSet into Mesh map
      mesh.side_sets.insert(std::pair<int, MeshInput::SideSet>(spropID[i], actSideSet));
    }
  }  // end of sideset section

  CHECK_EXODUS_CALL(ex_close(exo_handle));

  return mesh;
}

FOUR_C_NAMESPACE_CLOSE
