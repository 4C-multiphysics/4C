// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_clement_interpolation.hpp"

#include "4C_fem_geometry_element_volume.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"

FOUR_C_NAMESPACE_OPEN

std::shared_ptr<Core::LinAlg::MultiVector<double>> Core::FE::compute_nodal_clement_interpolation(
    const Core::FE::Discretization& dis, const Core::LinAlg::MultiVector<double>& values)
{
  const size_t dimensions = dis.n_dim();
  const size_t spatial_dimensions = 3;
  auto nodal_values = std::make_shared<Core::LinAlg::MultiVector<double>>(
      *dis.node_row_map(), values.num_vectors(), true);

  const auto nurbsdis = dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(&dis);

  for (const auto& node : dis.my_row_node_range())
  {
    const auto element_patch = node.adjacent_elements();

    struct ElementData
    {
      int local_element_id;
      double volume;
    };

    std::vector<ElementData> patch_data;
    patch_data.reserve(element_patch.size());

    double total_weight = 0.0;

    for (const auto& element : element_patch)
    {
      const auto* user_element = element.user_element();
      const auto* element_nodes = user_element->nodes();
      const int number_of_nodes = user_element->num_node();

      Core::LinAlg::SerialDenseMatrix node_coordinates(spatial_dimensions, number_of_nodes, true);
      for (int node_number = 0; node_number < number_of_nodes; ++node_number)
        for (size_t dim = 0; dim < dimensions; ++dim)
          node_coordinates(dim, node_number) = element_nodes[node_number]->x()[dim];

      std::vector<Core::LinAlg::SerialDenseVector> myknots(dis.n_dim());
      Core::LinAlg::SerialDenseVector myweights(number_of_nodes);

      if (nurbsdis != nullptr)
      {
        Core::FE::Nurbs::get_my_nurbs_knots_and_weights(
            *nurbsdis, user_element, myknots, myweights);
      }

      const double volume =
          Core::Geo::element_volume(user_element->shape(), node_coordinates, myknots, myweights);

      const int global_element_id = element.global_id();
      const int local_element_id = values.get_vector(0).get_map().lid(global_element_id);

      FOUR_C_ASSERT(local_element_id >= 0, "Global element id {} not owned by this process",
          global_element_id);

      patch_data.push_back({local_element_id, volume});
      total_weight += volume;
    }

    for (int column = 0; column < values.num_vectors(); ++column)
    {
      const auto& element_value_vector = values.get_vector(column);
      const auto values_span = element_value_vector.local_values_as_span();

      double node_value = 0.0;
      for (const auto& data : patch_data)
        node_value += values_span[data.local_element_id] * data.volume;

      const double interpolated_nodal_value = node_value / total_weight;
      nodal_values->replace_global_value(node.global_id(), column, interpolated_nodal_value);
    }
  }

  return nodal_values;
}

FOUR_C_NAMESPACE_CLOSE
