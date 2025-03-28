// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_nodereader.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element_definition.hpp"
#include "4C_fem_general_fiber_node.hpp"
#include "4C_fem_nurbs_discretization_control_point.hpp"
#include "4C_io_input_file.hpp"
#include "4C_io_value_parser.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  std::vector<std::shared_ptr<Core::FE::Discretization>> find_dis_node(
      const std::vector<Core::IO::ElementReader>& element_readers, int global_node_id)
  {
    std::vector<std::shared_ptr<Core::FE::Discretization>> list_of_discretizations;
    for (const auto& element_reader : element_readers)
      if (element_reader.has_node(global_node_id))
        list_of_discretizations.emplace_back(element_reader.get_dis());

    return list_of_discretizations;
  }

}  // namespace


void Core::IO::read_nodes(Core::IO::InputFile& input, const std::string& node_section_name,
    std::vector<ElementReader>& element_readers, int& max_node_id)
{
  const int myrank = Core::Communication::my_mpi_rank(input.get_comm());
  if (myrank > 0) return;

  int line_count = 0;
  for (const auto& node_line : input.in_section_rank_0_only(node_section_name))
  {
    Core::IO::ValueParser parser{
        node_line.get_as_dat_style_string(), {.user_scope_message = "While reading node data: "}};
    auto type = parser.read<std::string>();

    if (type == "NODE")
    {
      int nodeid = parser.read<int>() - 1;
      parser.consume("COORD");
      auto coords = parser.read<std::vector<double>>(3);

      max_node_id = std::max(max_node_id, nodeid) + 1;
      std::vector<std::shared_ptr<Core::FE::Discretization>> dis =
          find_dis_node(element_readers, nodeid);

      for (const auto& di : dis)
      {
        // create node and add to discretization
        std::shared_ptr<Core::Nodes::Node> node =
            std::make_shared<Core::Nodes::Node>(nodeid, coords, myrank);
        di->add_node(node);
      }
    }
    // this node is a Nurbs control point
    else if (type == "CP")
    {
      int cpid = parser.read<int>() - 1;
      parser.consume("COORD");
      auto coords = parser.read<std::vector<double>>(3);
      double weight = parser.read<double>();

      max_node_id = std::max(max_node_id, cpid) + 1;
      if (cpid != line_count)
        FOUR_C_THROW(
            "Reading of control points {} failed: They must be numbered consecutive!!", cpid);
      std::vector<std::shared_ptr<Core::FE::Discretization>> diss =
          find_dis_node(element_readers, cpid);

      for (auto& dis : diss)
      {
        // create node/control point and add to discretization
        std::shared_ptr<Core::FE::Nurbs::ControlPoint> node =
            std::make_shared<Core::FE::Nurbs::ControlPoint>(cpid, coords, weight, myrank);
        dis->add_node(node);
      }
    }
    // this is a special node with additional fiber information
    else if (type == "FNODE")
    {
      enum class FiberType
      {
        Unknown,
        Angle,
        Fiber,
        CosyDirection
      };

      // read fiber node
      std::map<Core::Nodes::CoordinateSystemDirection, std::array<double, 3>> cosyDirections;
      std::vector<std::array<double, 3>> fibers;
      std::map<Core::Nodes::AngleType, double> angles;

      int nodeid = parser.read<int>() - 1;
      parser.consume("COORD");
      auto coords = parser.read<std::vector<double>>(3);
      max_node_id = std::max(max_node_id, nodeid) + 1;

      while (!parser.at_end())
      {
        auto next = parser.read<std::string>();

        if (next == "FIBER" + std::to_string(1 + fibers.size()))
        {
          fibers.emplace_back(parser.read<std::array<double, 3>>());
        }
        else if (next == "CIR")
        {
          cosyDirections[Nodes::CoordinateSystemDirection::Circular] =
              parser.read<std::array<double, 3>>();
        }
        else if (next == "TAN")
        {
          cosyDirections[Nodes::CoordinateSystemDirection::Tangential] =
              parser.read<std::array<double, 3>>();
        }
        else if (next == "RAD")
        {
          cosyDirections[Nodes::CoordinateSystemDirection::Radial] =
              parser.read<std::array<double, 3>>();
        }
        else if (next == "HELIX")
        {
          angles[Nodes::AngleType::Helix] = parser.read<double>();
        }
        else if (next == "TRANS")
        {
          angles[Nodes::AngleType::Transverse] = parser.read<double>();
        }
      }

      // add fiber information to node
      std::vector<std::shared_ptr<Core::FE::Discretization>> discretizations =
          find_dis_node(element_readers, nodeid);
      for (auto& dis : discretizations)
      {
        auto node = std::make_shared<Core::Nodes::FiberNode>(
            nodeid, coords, cosyDirections, fibers, angles, myrank);
        dis->add_node(node);
      }
    }
    else
      FOUR_C_THROW("Unknown node type '{}'", type);

    ++line_count;
  }
}

FOUR_C_NAMESPACE_CLOSE
