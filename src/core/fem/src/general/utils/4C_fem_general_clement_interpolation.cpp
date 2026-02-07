// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_clement_interpolation.hpp"

#include "4C_fem_geometry_element_volume.hpp"

FOUR_C_NAMESPACE_OPEN

std::shared_ptr<Core::LinAlg::MultiVector<double>> Core::FE::compute_nodal_clement_interpolation(
    const Core::FE::Discretization& dis, const Core::LinAlg::MultiVector<double>& values)
{
  const size_t dimensions = dis.n_dim();
  auto nodal_values = std::make_shared<Core::LinAlg::MultiVector<double>>(
      *dis.node_row_map(), values.num_vectors(), true);

  for (const auto& node : dis.my_row_node_range())
  {
    auto element_patch = node.adjacent_elements();

    for (int column = 0; column < values.num_vectors(); ++column)
    {
      double weight = 0.0;
      double node_value = 0.0;

      Core::LinAlg::Vector<double> element_value_vector(values.get_vector(column));

      for (const auto& element : element_patch)
      {
        const auto* element_nodes = element.user_element()->nodes();
        const int number_of_nodes = element.user_element()->num_node();

        Core::LinAlg::SerialDenseMatrix node_coordinates(dimensions, number_of_nodes);
        for (int node_number = 0; node_number < number_of_nodes; node_number++)
          for (size_t dim = 0; dim < dimensions; ++dim)
            node_coordinates(dim, node_number) = element_nodes[node_number]->x()[dim];

        const double volume =
            Core::Geo::element_volume(element.user_element()->shape(), node_coordinates);

        const int global_element_id = element.global_id();
        const int local_element_id = element_value_vector.get_map().lid(global_element_id);

        node_value += element_value_vector.local_values_as_span()[local_element_id] * volume;
        weight += volume;
      }

      double interpolated_nodal_value = node_value / weight;

      nodal_values->replace_global_value(node.global_id(), column, interpolated_nodal_value);
    }
  }

  return nodal_values;
}

FOUR_C_NAMESPACE_CLOSE
