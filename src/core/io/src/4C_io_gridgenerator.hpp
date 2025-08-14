// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_GRIDGENERATOR_HPP
#define FOUR_C_IO_GRIDGENERATOR_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_io_input_spec.hpp"

#include <array>
#include <string>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::IO::GridGenerator
{
  /// forward declaration
  struct RectangularCuboidInputs;

  /// create rectangle cuboid domain (parallel distributed discretization) which may be rotated
  /// based on given parameters
  /*!
    \param dis                    (i/o) discretization to be filled with nodes and elements
    \param inputData              (i) class holding all input parameters
    \param outputFlag             (i) triggers output writing
  */
  void create_rectangular_cuboid_discretization(
      Core::FE::Discretization& dis, const RectangularCuboidInputs& inputData, bool outputFlag);

  /// storage for input parameters for CreateRectangularCuboidDiscretization
  struct RectangularCuboidInputs
  {
    /// first point of the cuboid which is diagonally opposite of top_corner_point_
    /// (expectation: all values are smaller than the values in top_corner_point_)
    std::array<double, 3> bottom_corner_point_;

    /// second point of the cuboid which is diagonally opposite of bottom_corner_point_
    /// (expectation: all values are larger than the values in bottom_corner_point_)
    std::array<double, 3> top_corner_point_;

    /// intervals of the domain to be created
    std::array<int, 3> interval_;

    /// rotation angles of the box geometry
    std::array<double, 3> rotation_angle_{0.0, 0.0, 0.0};

    /// element type of the domain to be created
    std::string elementtype_;

    /// discretization type of the domain to be created
    Core::FE::CellType cell_type;

    /// further arguments to the elements to be created
    InputParameterContainer element_arguments;

    /// global id of the first newly created node
    int node_gid_of_first_new_node_{0};

    /// decide on partitioning strategy
    bool autopartition_{false};

    /**
     * Return the InputSpec containing the parameters encoded in this struct.
     */
    static InputSpec spec();

    /**
     * Given an InputParameterContainer that matches the spec(), create a
     * RectangularCuboidInputs object.
     */
    static RectangularCuboidInputs from_input(const Core::IO::InputParameterContainer& input);
  };

}  // namespace Core::IO::GridGenerator
   // namespace Discret
FOUR_C_NAMESPACE_CLOSE

#endif
