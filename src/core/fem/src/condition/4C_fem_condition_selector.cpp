// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition_selector.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Core::Conditions::Selector::Selector(std::string condition_name) : condition_name_(condition_name)
{
}



Core::Conditions::Selector::Selector(std::string condition_name, int start_pos, int end_pos)
    : condition_name_(condition_name), start_pos_(start_pos), end_pos_(end_pos)
{
}



Core::Conditions::Selector::Selector(
    const std::vector<const Core::Conditions::Condition*>& conditions)
    : conditions_(conditions)
{
  FOUR_C_ASSERT_ALWAYS(conditions_.size() > 0, "Empty condition list");
}



void Core::Conditions::setup_extractor(const Core::FE::Discretization& dis,
    Core::LinAlg::MultiMapExtractor& extractor, const std::vector<Selector>& selectors,
    bool is_overlapping)
{
  setup_extractor(dis, *dis.dof_row_map(), extractor, selectors, is_overlapping);
}



void Core::Conditions::setup_extractor(const Core::FE::Discretization& dis,
    const Core::LinAlg::Map& full_map, Core::LinAlg::MultiMapExtractor& extractor,
    const std::vector<Selector>& selectors, bool is_overlapping)
{
  std::vector<std::set<int>> conditioned_dof_sets(selectors.size());

  std::vector<std::vector<const Core::Conditions::Condition*>> conditions_for_selector;
  for (const auto& selector : selectors)
  {
    if (selector.condition_name_ != "")
    {
      dis.get_condition(selector.condition_name_, conditions_for_selector.emplace_back());
    }
    else
    {
      conditions_for_selector.emplace_back(selector.conditions_);
    }
  }

  const auto select_dofs = [&](const Selector& selector,
                               std::span<const Core::Conditions::Condition*> conditions,
                               FE::ConstNodeRef node, std::set<int>& conddofset) -> bool
  {
    const bool contains_node = std::ranges::any_of(conditions,
        [gid = node.global_id()](const auto* cond) { return cond->contains_node(gid); });

    // put all conditioned dofs into conddofset
    if (contains_node)
    {
      std::vector<int> dof = dis.dof(0, node);
      // Insert the dofs that are within the given range.
      const auto first_dof = dof.begin() + selector.start_pos_;
      const auto last_dof = dof.begin() + std::min(selector.end_pos_, static_cast<int>(dof.size()));
      conddofset.insert(first_dof, last_dof);
      return std::distance(first_dof, last_dof) > 0;
    }
    return false;
  };

  for (const auto& node : dis.my_row_node_range())
  {
    for (unsigned j = 0; j < selectors.size(); ++j)
    {
      const Selector& selector = selectors[j];

      // if the selector applies, we are done
      if (select_dofs(selector, conditions_for_selector[j], node, conditioned_dof_sets[j]))
        if (!is_overlapping) break;
    }
  }

  // Find all non-conditioned dofs by subtracting all conditioned ones.
  std::set<int> otherdofset(
      full_map.my_global_elements(), full_map.my_global_elements() + full_map.num_my_elements());

  for (auto& conddofset : conditioned_dof_sets)
  {
    for (const auto& dof : conddofset)
    {
      otherdofset.erase(dof);
    }
  }

  // Setup all maps. The "other" map goes first so it becomes the zeroth map
  // of the MultiMapExtractor.

  std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps;
  maps.reserve(conditioned_dof_sets.size() + 1);

  maps.emplace_back(Core::LinAlg::create_map(otherdofset, dis.get_comm()));
  for (auto& conddofset : conditioned_dof_sets)
  {
    maps.emplace_back(Core::LinAlg::create_map(conddofset, dis.get_comm()));
  }

  // MultiMapExtractor setup
  extractor.setup(full_map, maps);
}

FOUR_C_NAMESPACE_CLOSE
