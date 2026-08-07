// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_solid_ele_fibers.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"

namespace
{
  using namespace FourC;

  Core::IO::InputParameterContainer direction_input_with_fiber_and_coordinate_system()
  {
    Core::IO::InputParameterContainer input;
    input.add("FIBER1", std::optional<std::vector<double>>{{1.0, 0.0, 0.0}});
    input.add("RAD", std::optional<std::vector<double>>{{1.0, 0.0, 0.0}});
    input.add("AXI", std::optional<std::vector<double>>{{0.0, 1.0, 0.0}});
    input.add("CIR", std::optional<std::vector<double>>{{0.0, 0.0, 1.0}});
    return input;
  }

  void add_empty_optional_direction_defaults(Core::IO::InputParameterContainer& input)
  {
    for (const char* direction : {"RAD", "AXI", "CIR", "FIBER1", "FIBER2", "FIBER3"})
      input.add(direction, std::optional<std::vector<double>>{std::vector<double>{}});
  }

  TEST(SolidElementFiberTest, RejectsFiberAndCoordinateSystemWhenReadingFibers)
  {
    const auto input = direction_input_with_fiber_and_coordinate_system();

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(Discret::Elements::read_fibers(input), Core::Exception,
        "either by RAD/AXI/CIR or by FIBER1, FIBER2, FIBER3");
  }

  TEST(SolidElementFiberTest, RejectsFiberAndCoordinateSystemWhenReadingCoordinateSystem)
  {
    const auto input = direction_input_with_fiber_and_coordinate_system();

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(Discret::Elements::read_coordinate_system(input),
        Core::Exception, "either by RAD/AXI/CIR or by FIBER1, FIBER2, FIBER3");
  }

  TEST(SolidElementFiberTest, RejectsFiber2WithoutFiber1)
  {
    Core::IO::InputParameterContainer input;
    input.add("FIBER2", std::optional<std::vector<double>>{{0.0, 1.0, 0.0}});

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        Discret::Elements::read_fibers(input), Core::Exception, "FIBER2 requires FIBER1");
  }

  TEST(SolidElementFiberTest, RejectsFiber3WithoutFiber2)
  {
    Core::IO::InputParameterContainer input;
    input.add("FIBER1", std::optional<std::vector<double>>{{1.0, 0.0, 0.0}});
    input.add("FIBER3", std::optional<std::vector<double>>{{0.0, 0.0, 1.0}});

    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        Discret::Elements::read_fibers(input), Core::Exception, "FIBER3 requires FIBER2");
  }

  TEST(SolidElementFiberTest, RejectsIncompleteCoordinateSystem)
  {
    for (const char* direction : {"RAD", "AXI", "CIR"})
    {
      Core::IO::InputParameterContainer input;
      add_empty_optional_direction_defaults(input);
      input.add(direction, std::optional<std::vector<double>>{{1.0, 0.0, 0.0}});

      FOUR_C_EXPECT_THROW_WITH_MESSAGE(Discret::Elements::read_coordinate_system(input),
          Core::Exception, "need to define all of RAD, AXI and CIR");
    }
  }

  TEST(SolidElementFiberTest, IgnoresEmptyOptionalFiberDefaults)
  {
    Core::IO::InputParameterContainer input;
    add_empty_optional_direction_defaults(input);
    input.add("RAD", std::optional<std::vector<double>>{{1.0, 0.0, 0.0}});
    input.add("AXI", std::optional<std::vector<double>>{{0.0, 1.0, 0.0}});
    input.add("CIR", std::optional<std::vector<double>>{{0.0, 0.0, 1.0}});

    EXPECT_NO_THROW(Discret::Elements::read_coordinate_system(input));
    EXPECT_TRUE(Discret::Elements::read_fibers(input).element_fibers.empty());
  }
}  // namespace
