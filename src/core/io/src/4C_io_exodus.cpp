// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_exodus.hpp"

#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_utils_enum.hpp"

#include <exodusII.h>
#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <fstream>
#include <ranges>

FOUR_C_NAMESPACE_OPEN

namespace
{

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

/*----------------------------------------------------------------------*
 |  ctor (public)                                              maf 12/07|
 *----------------------------------------------------------------------*/
Core::IO::Exodus::Mesh::Mesh(const std::string exofilename)
{
  int error;
  int CPU_word_size, IO_word_size;
  float exoversion;               /* version of exodus */
  CPU_word_size = sizeof(double); /* size of a double */
  IO_word_size = 0;               /* use what is stored in file */


  // open EXODUS II file
  int exo_handle =
      ex_open(exofilename.c_str(), EX_READ, &CPU_word_size, &IO_word_size, &exoversion);
  if (exo_handle <= 0) FOUR_C_THROW("Error while opening EXODUS II file {}", exofilename);

  // read database parameters
  int num_elem_blk, num_node_sets, num_side_sets, num_nodes;
  char title[MAX_LINE_LENGTH + 1];
  error = ex_get_init(exo_handle, title, &four_c_dim_, &num_nodes, &num_elem_, &num_elem_blk,
      &num_node_sets, &num_side_sets);
  title_ = std::string(title);

  num_dim_ = 3;

  // get nodal coordinates
  {
    std::vector<double> x(num_nodes);
    std::vector<double> y(num_nodes);
    std::vector<double> z(num_nodes);
    error = ex_get_coord(exo_handle, x.data(), y.data(), z.data());
    if (error != 0) FOUR_C_THROW("exo error returned");

    for (int i = 0; i < num_nodes; ++i)
    {
      // Node numbering starts at one in 4C
      nodes_[i + 1] = {x[i], y[i], z[i]};
    }
  }

  // Get all ElementBlocks
  {
    std::vector<int> epropID(num_elem_blk);
    std::vector<int> ebids(num_elem_blk);
    error = ex_get_ids(exo_handle, EX_ELEM_BLOCK, ebids.data());
    if (error != 0) FOUR_C_THROW("exo error returned");
    error = ex_get_prop_array(exo_handle, EX_ELEM_BLOCK, "ID", epropID.data());
    if (error != 0) FOUR_C_THROW("exo error returned");
    for (int i = 0; i < num_elem_blk; ++i)
    {
      // Read Element Blocks into Map
      char mychar[MAX_STR_LENGTH + 1];
      int num_el_in_blk, num_nod_per_elem, num_attr;
      error = ex_get_block(exo_handle, EX_ELEM_BLOCK, ebids[i], mychar, &num_el_in_blk,
          &num_nod_per_elem, nullptr, nullptr, &num_attr);
      if (error != 0) FOUR_C_THROW("exo error returned");
      // prefer std::string to store element type
      std::string ele_type(mychar);

      // get ElementBlock name
      error = ex_get_name(exo_handle, EX_ELEM_BLOCK, ebids[i], mychar);
      if (error != 0) FOUR_C_THROW("exo error returned");
      // prefer std::string to store name
      std::string blockname(mychar);

      // get element connectivity
      std::vector<int> allconn(num_nod_per_elem * num_el_in_blk);
      error = ex_get_conn(exo_handle, EX_ELEM_BLOCK, ebids[i], allconn.data(), nullptr, nullptr);
      if (error != 0) FOUR_C_THROW("exo error returned");
      std::map<int, std::vector<int>> eleconn;
      for (int j = 0; j < num_el_in_blk; ++j)
      {
        std::vector<int>& actconn = eleconn[j];
        actconn.reserve(num_nod_per_elem);
        for (int k = 0; k < num_nod_per_elem; ++k)
        {
          actconn.push_back(allconn[k + j * num_nod_per_elem]);
        }
        reorder_nodes_exodus_to_four_c(actconn, cell_type_from_exodus_string(ele_type));
      }

      element_blocks_.emplace(
          ebids[i], ElementBlock{cell_type_from_exodus_string(ele_type), eleconn, blockname});
    }
  }  // end of element section

  // get all NodeSets
  {
    std::vector<int> npropID(num_node_sets);
    error = ex_get_prop_array(exo_handle, EX_NODE_SET, "ID", npropID.data());
    for (int i = 0; i < num_node_sets; ++i)
    {
      // Read NodeSet params
      int num_nodes_in_set, num_df_in_set;
      error =
          ex_get_set_param(exo_handle, EX_NODE_SET, npropID[i], &num_nodes_in_set, &num_df_in_set);

      // get NodeSet name
      char mychar[MAX_STR_LENGTH + 1];
      error = ex_get_name(exo_handle, EX_NODE_SET, npropID[i], mychar);
      // prefer std::string to store name
      std::string nodesetname(mychar);

      // get nodes in node set
      std::vector<int> node_set_node_list(num_nodes_in_set);
      error = ex_get_set(exo_handle, EX_NODE_SET, npropID[i], node_set_node_list.data(), nullptr);
      if (error > 0)
        std::cout << "'ex_get_set' for EX_NODE_SET returned warning while reading node set "
                  << npropID[i] << std::endl;
      else if (error < 0)
        FOUR_C_THROW("error reading node set");
      std::set<int> nodes_in_set;
      for (int j = 0; j < num_nodes_in_set; ++j) nodes_in_set.insert(node_set_node_list[j]);
      NodeSet actNodeSet(nodes_in_set, nodesetname);

      // Add this NodeSet into Mesh map (here prelim due to pro names)
      node_sets_.insert(std::pair<int, NodeSet>(npropID[i], actNodeSet));
    }
  }  // end of nodeset section
  // ***************************************************************************

  // get all SideSets
  if (num_side_sets > 0)
  {
    std::vector<int> spropID(num_side_sets);
    error = ex_get_prop_array(exo_handle, EX_SIDE_SET, "ID", spropID.data());
    for (int i = 0; i < num_side_sets; ++i)
    {
      // get SideSet name
      char mychar[MAX_STR_LENGTH + 1];
      error = ex_get_name(exo_handle, EX_SIDE_SET, spropID[i], mychar);
      // prefer std::string to store name
      std::string sidesetname(mychar);

      // Read SideSet params
      int num_side_in_set, num_dist_fact_in_set;
      error = ex_get_set_param(
          exo_handle, EX_SIDE_SET, spropID[i], &num_side_in_set, &num_dist_fact_in_set);

      // get SideSet
      std::vector<int> side_set_elem_list(num_side_in_set);
      std::vector<int> side_set_side_list(num_side_in_set);
      error = ex_get_set(exo_handle, EX_SIDE_SET, spropID[i], side_set_elem_list.data(),
          side_set_side_list.data());
      if (error != 0) FOUR_C_THROW("error reading side set");
      std::map<int, std::vector<int>> sides_in_set;
      for (int j = 0; j < num_side_in_set; ++j)
      {
        std::vector<int> side(2);  // first entry is element, second side
        side[0] = side_set_elem_list[j];
        side[1] = side_set_side_list[j];
        sides_in_set.insert(std::pair<int, std::vector<int>>(j, side));
      }

      SideSet actSideSet(sides_in_set, sidesetname);

      // Add this SideSet into Mesh map
      side_sets_.insert(std::pair<int, SideSet>(spropID[i], actSideSet));
    }
  }  // end of sideset section

  error = ex_close(exo_handle);
  if (error < 0) FOUR_C_THROW("error while closing exodus II file");
}


/*----------------------------------------------------------------------*
 |  Print method (public)                                      maf 12/07|
 *----------------------------------------------------------------------*/
void Core::IO::Exodus::Mesh::print(std::ostream& os, bool verbose) const
{
  os << "Mesh consists of ";
  os << get_num_nodes() << " Nodes, ";
  os << num_elem_ << " Elements, organized in " << std::endl;
  os << get_num_element_blocks() << " ElementBlocks, ";
  os << get_num_node_sets() << " NodeSets, ";
  os << get_num_side_sets() << " SideSets ";
  os << std::endl << std::endl;
  if (verbose)
  {
    os << "ElementBlocks\n";
    for (const auto& [eb_id, eb] : get_element_blocks())
    {
      os << eb_id << ": ";
      eb.print(os);
    }

    os << "\nNodeSets\n";
    for (const auto& [ns_id, ns] : get_node_sets())
    {
      os << "NodeSet " << ns_id << ": ";
      ns.print(os);
    }

    os << "\nSideSets\n";
    for (const auto& [ss_id, ss] : get_side_sets())
    {
      os << "SideSet " << ss_id << ": ";
      ss.print(os);
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const Core::IO::Exodus::ElementBlock& Core::IO::Exodus::Mesh::get_element_block(const int id) const
{
  if (element_blocks_.find(id) == element_blocks_.end())
    FOUR_C_THROW("ElementBlock {} not found.", id);

  return (element_blocks_.find(id))->second;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::Exodus::NodeSet Core::IO::Exodus::Mesh::get_node_set(const int id) const
{
  if (node_sets_.find(id) == node_sets_.end()) FOUR_C_THROW("NodeSet {} not found.", id);

  return (node_sets_.find(id))->second;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::Exodus::SideSet Core::IO::Exodus::Mesh::get_side_set(const int id) const
{
  if (side_sets_.find(id) == side_sets_.end()) FOUR_C_THROW("SideSet {} not found.", id);

  return (side_sets_.find(id))->second;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::vector<double> Core::IO::Exodus::Mesh::get_node(const int NodeID) const
{
  return nodes_.at(NodeID);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::Exodus::Mesh::set_node(const int NodeID, const std::vector<double> coord)
{
  nodes_[NodeID] = coord;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::Exodus::Mesh::set_nsd(const int nsd)
{
  if (nsd != 2 && nsd != 3) return;

  four_c_dim_ = nsd;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::vector<double> Core::IO::Exodus::Mesh::normal(
    const int head1, const int origin, const int head2) const
{
  std::vector<double> normal(3);
  std::vector<double> h1 = get_node(head1);
  std::vector<double> h2 = get_node(head2);
  std::vector<double> o = get_node(origin);

  normal[0] = ((h1[1] - o[1]) * (h2[2] - o[2]) - (h1[2] - o[2]) * (h2[1] - o[1]));
  normal[1] = -((h1[0] - o[0]) * (h2[2] - o[2]) - (h1[2] - o[2]) * (h2[0] - o[0]));
  normal[2] = ((h1[0] - o[0]) * (h2[1] - o[1]) - (h1[1] - o[1]) * (h2[0] - o[0]));

  double length = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
  normal[0] = normal[0] / length;
  normal[1] = normal[1] / length;
  normal[2] = normal[2] / length;

  return normal;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::vector<double> Core::IO::Exodus::Mesh::node_vec(const int tail, const int head) const
{
  std::vector<double> nv(3);
  std::vector<double> t = get_node(tail);
  std::vector<double> h = get_node(head);
  nv[0] = h[0] - t[0];
  nv[1] = h[1] - t[1];
  nv[2] = h[2] - t[2];
  double length = sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]);
  nv[0] = nv[0] / length;
  nv[1] = nv[1] / length;
  nv[2] = nv[2] / length;
  return nv;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::Exodus::ElementBlock::ElementBlock(
    FE::CellType cell_type, std::map<int, std::vector<int>> eleconn, std::string name)
    : cell_type_(cell_type), eleconn_(std::move(eleconn)), name_(name.c_str())
{
  // do a sanity check
  for (const auto& elem : eleconn_ | std::views::values)
  {
    // check if the number of nodes in the element block is correct
    if (static_cast<int>(elem.size()) != Core::FE::get_number_of_element_nodes(cell_type))
      FOUR_C_THROW("number of read nodes does not fit the distype");
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const std::vector<int>& Core::IO::Exodus::ElementBlock::get_ele_nodes(int i) const
{
  return eleconn_.at(i);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::Exodus::ElementBlock::print(std::ostream& os, bool verbose) const
{
  using EnumTools::operator<<;

  os << "Element Block, named: " << name_ << std::endl
     << "of Shape: " << cell_type_ << std::endl
     << "has " << get_num_ele() << " Elements" << std::endl;
  if (verbose)
  {
    for (const auto& [id, conn] : eleconn_)
    {
      os << "Ele " << id << ": ";
      for (const int i : conn)
      {
        os << i << ",";
      }
      os << std::endl;
    }
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::Exodus::NodeSet::NodeSet(const std::set<int>& nodeids, const std::string& name)
    : nodeids_(nodeids), name_(name.c_str())
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::Exodus::NodeSet::print(std::ostream& os, bool verbose) const
{
  os << "Node Set, named: " << name_ << std::endl
     << "has " << get_num_nodes() << " Nodes" << std::endl;
  if (verbose)
  {
    os << "Contains Nodes:" << std::endl;
    std::set<int>::iterator it;
    for (it = nodeids_.begin(); it != nodeids_.end(); it++) os << *it << ",";
    os << std::endl;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::Exodus::SideSet::SideSet(
    const std::map<int, std::vector<int>>& sides, const std::string& name)
    : sides_(sides), name_(name.c_str())
{
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::Exodus::SideSet::print(std::ostream& os, bool verbose) const
{
  os << "SideSet, named: " << name_ << std::endl
     << "has " << get_num_sides() << " Sides" << std::endl;
  if (verbose)
  {
    std::map<int, std::vector<int>>::const_iterator it;
    for (it = sides_.begin(); it != sides_.end(); it++)
    {
      os << "Side " << it->first << ": ";
      os << "Ele: " << it->second.at(0) << ", Side: " << it->second.at(1) << std::endl;
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
