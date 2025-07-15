// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_deal_ii_context.hpp"

FOUR_C_NAMESPACE_OPEN


namespace DealiiWrappers
{

  namespace Internal
  {
    void get_dof_indices_with_local_reorder(const Core::FE::Discretization& discretization,
        const Core::Elements::Element* element, const std::span<const int>& local_reorder,
        std::vector<dealii::types::global_dof_index>& dof_indices)
    {
      Core::Elements::LocationArray location_array(discretization.num_dof_sets());
      element->location_vector(discretization, location_array);
      dof_indices.resize(location_array[0].lm_.size());
      for (unsigned int i = 0; i < location_array[0].lm_.size(); ++i)
      {
        dof_indices[i] = location_array[0].lm_[local_reorder[i]];
      }
    }

    void get_dof_indices(const Core::FE::Discretization& discretization,
        const Core::Elements::Element* element,
        std::vector<dealii::types::global_dof_index>& dof_indices)
    {
      Core::Elements::LocationArray location_array(discretization.num_dof_sets());
      element->location_vector(discretization, location_array);
      dof_indices.resize(location_array[0].lm_.size());
      std::copy(location_array[0].lm_.begin(), location_array[0].lm_.end(), dof_indices.begin());
    }
  }  // namespace Internal
}  // namespace DealiiWrappers


FOUR_C_NAMESPACE_CLOSE