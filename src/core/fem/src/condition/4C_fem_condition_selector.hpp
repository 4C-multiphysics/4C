// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_CONDITION_SELECTOR_HPP
#define FOUR_C_FEM_CONDITION_SELECTOR_HPP

#include "4C_config.hpp"

#include <limits>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class Map;
  class MultiMapExtractor;
}  // namespace Core::LinAlg

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Nodes
{
  class Node;
}

namespace Core::Conditions
{
  class Condition;

  class Selector;

  /**
   * Oftentimes the DOFs of a field need to be split into a set of disjoint maps. The
   * Core::LinAlg::MultiMapExtractor class takes care of these splits. This function helps you to
   * set up the @p extractor by providing the necessary information. It takes a number of @p
   * selectors that encapsulate the conditions to be selected from the given Discretization @p dis.
   * Refer to the Selector class for more information on how conditions can be selected.
   *
   * The ordering of the maps in the extractor is as follows:
   *
   * - The first map is the "other" map, which contains all DOFs that are not selected by any of the
   *   conditions.
   * - The subsequent maps are the selected maps, which contain the DOFs of the selected conditions.
   *   The order of the maps is determined by the order of the selectors in the @p selectors
   *   vector.
   *
   * The optional @p is_overlapping flag indicates whether the selected conditions may have
   * overlapping dofs. By default, this is false.
   *
   * @note Each node ends up in one condition only, independent of whether all its dofs are selected
   * or not.
   */
  void setup_extractor(const Core::FE::Discretization& dis,
      Core::LinAlg::MultiMapExtractor& extractor, const std::vector<Selector>& selectors,
      bool is_overlapping = false);

  /**
   * This function is a variant of the above setup_extractor() function. It takes an additional
   * @p full_map parameter. The extracted DOFs are subtracted from that map to create the
   * "other" map of the extractor. In the other functions, the @p full_map is equal to the DOF row
   * map of the discretization.
   */
  void setup_extractor(const Core::FE::Discretization& dis, const Core::LinAlg::Map& full_map,
      Core::LinAlg::MultiMapExtractor& extractor, const std::vector<Selector>& selectors,
      bool is_overlapping = false);

  /**
   * A class to select Condition whose DOFs should be extracted. It works in conjunction with
   * @p setup_extractor() to set up a Core::LinAlg::MultiMapExtractor object.
   */
  class Selector
  {
   public:
    /**
     * Selects all conditions that match the given @p condition_name.
     */
    Selector(std::string condition_name);

    /**
     * Selects all conditions that match the given @p condition_name.
     * The dof positions to be selected are given by the  @p start_pos (inclusive) and @p end_pos
     * (exclusive) parameters.
     */
    Selector(std::string condition_name, int start_pos, int end_pos);

    /**
     * Selects the given @p conditions.
     */
    Selector(const std::vector<Condition*>& conditions);

   private:
    /// The name of the condition to extract.
    std::string condition_name_;

    /// The conditions to extract.
    std::vector<Condition*> conditions_;

    /// The first dof position of the node to select.
    int start_pos_{0};

    /// The past-the-end dof position of the node to select.
    int end_pos_{std::numeric_limits<int>::max()};

    friend void setup_extractor(const Core::FE::Discretization& dis,
        const Core::LinAlg::Map& full_map, Core::LinAlg::MultiMapExtractor& extractor,
        const std::vector<Selector>& selectors, bool is_overlapping);
  };
}  // namespace Core::Conditions

FOUR_C_NAMESPACE_CLOSE

#endif
