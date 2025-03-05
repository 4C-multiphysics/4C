// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_inpar_searchtree.hpp"

#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void Inpar::Geo::set_valid_parameters(std::map<std::string, Core::IO::InputSpec>& list)
{
  using Teuchos::tuple;
  using namespace Core::IO::InputSpecBuilders;

  Core::Utils::SectionSpecs search_tree{"SEARCH TREE"};

  Core::Utils::string_to_integral_parameter<Inpar::Geo::TreeType>("TREE_TYPE", "notree",
      "set tree type", tuple<std::string>("notree", "octree3d", "quadtree3d", "quadtree2d"),
      tuple<Inpar::Geo::TreeType>(
          Inpar::Geo::Notree, Inpar::Geo::Octree3D, Inpar::Geo::Quadtree3D, Inpar::Geo::Quadtree2D),
      search_tree);

  search_tree.move_into_collection(list);
}

FOUR_C_NAMESPACE_CLOSE
