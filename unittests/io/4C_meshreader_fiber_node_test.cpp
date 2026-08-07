// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_general_fiber_node.hpp"
#include "4C_fem_general_fiber_node_holder.hpp"
#include "4C_fem_general_fiber_node_utils.hpp"
#include "4C_io_meshreader.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"

namespace
{
  using namespace FourC;

  TEST(MeshReaderFiberNodeTest, RejectsFiberNumberingGap)
  {
    Core::IO::ValueParser parser("FIBER2 0.0 1.0 0.0");

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        Core::IO::Internal::read_fiber_node_data(parser), Core::Exception, "Expected 'FIBER1'");
  }

  TEST(MeshReaderFiberNodeTest, RejectsFiberAndCoordinateSystem)
  {
    Core::IO::ValueParser parser(
        "FIBER1 1.0 0.0 0.0 CIR 1.0 0.0 0.0 TAN 0.0 1.0 0.0 HELIX 0.0 TRANS 0.0");

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(Core::IO::Internal::read_fiber_node_data(parser),
        Core::Exception, "either by CIR/TAN/HELIX/TRANS or by FIBER1");
  }

  TEST(MeshReaderFiberNodeTest, RejectsIncompleteCardiacFiberDirections)
  {
    for (const std::string input : {
             "TAN 0.0 1.0 0.0 HELIX 0.0 TRANS 0.0",
             "CIR 1.0 0.0 0.0 HELIX 0.0 TRANS 0.0",
             "CIR 1.0 0.0 0.0 TAN 0.0 1.0 0.0 TRANS 0.0",
             "CIR 1.0 0.0 0.0 TAN 0.0 1.0 0.0 HELIX 0.0",
         })
    {
      Core::IO::ValueParser parser(input);

      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Core::IO::Internal::read_fiber_node_data(parser),
          Core::Exception, "require all of CIR, TAN, HELIX, and TRANS");
    }
  }

  TEST(MeshReaderFiberNodeTest, AcceptsConsecutiveFibers)
  {
    Core::IO::ValueParser parser("FIBER1 1.0 0.0 0.0 FIBER2 0.0 1.0 0.0 FIBER3 0.0 0.0 1.0");

    const auto data = Core::IO::Internal::read_fiber_node_data(parser);

    EXPECT_EQ(data.fibers.size(), 3);
    EXPECT_TRUE(data.coordinate_system_directions.empty());
  }

  TEST(MeshReaderFiberNodeTest, AcceptsCardiacFiberDirections)
  {
    Core::IO::ValueParser parser("CIR 1.0 0.0 0.0 TAN 0.0 1.0 0.0 HELIX 0.0 TRANS 0.0");

    const auto data = Core::IO::Internal::read_fiber_node_data(parser);

    EXPECT_EQ(data.coordinate_system_directions.size(), 2);
    EXPECT_EQ(data.angles.size(), 2);
    EXPECT_TRUE(data.fibers.empty());
  }

  TEST(MeshReaderFiberNodeTest, ReportsMissingNodalFiberAngle)
  {
    Core::Nodes::NodalFiberHolder holder;

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(holder.get_angle(Core::Nodes::AngleType::Helix),
        Core::Exception, "does not contain the requested HELIX angle");
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(holder.get_angle(Core::Nodes::AngleType::Transverse),
        Core::Exception, "does not contain the requested TRANS angle");
  }

  TEST(MeshReaderFiberNodeTest, RejectsDifferentFiberCountsWithinElement)
  {
    const std::array<double, 3> coordinates{};
    const std::array<double, 3> fiber{1.0, 0.0, 0.0};
    std::array<Core::Nodes::FiberNode, 4> fiber_nodes{
        Core::Nodes::FiberNode(0, coordinates, {}, {fiber}, {}, 0),
        Core::Nodes::FiberNode(1, coordinates, {}, {fiber, fiber}, {}, 0),
        Core::Nodes::FiberNode(2, coordinates, {}, {fiber}, {}, 0),
        Core::Nodes::FiberNode(3, coordinates, {}, {fiber}, {}, 0)};
    std::array<const Core::Nodes::Node*, 4> nodes{
        &fiber_nodes[0], &fiber_nodes[1], &fiber_nodes[2], &fiber_nodes[3]};
    Core::Nodes::NodalFiberHolder holder;
    const std::vector<Core::LinAlg::Matrix<4, 1>> shape_functions;

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        Core::Nodes::project_fibers_to_gauss_points<Core::FE::CellType::tet4>(
            nodes.data(), shape_functions, holder),
        Core::Exception, "same number of FIBER directions");
  }
}  // namespace
