// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_ELEMENT_CONVERSION_HPP
#define FOUR_C_DEAL_II_ELEMENT_CONVERSION_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/hp/fe_collection.h>

FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers::ElementConversion
{

  /**
   * Returns the reindexing of deal.II vertices to 4C vertices for a given cell type. This means
   * that the i-th vertex of a deal.II cell corresponds to the reindex[i]-th vertex of the
   * corresponding 4C cell.
   */
  inline std::span<const int> reindex_dealii_to_four_c(Core::FE::CellType cell_type)
  {
    switch (cell_type)
    {
      case Core::FE::CellType::line2:
      {
        static constexpr std::array reindex{0, 1};
        return reindex;
      }
      case Core::FE::CellType::tet4:
      {
        static constexpr std::array reindex{0, 1, 2, 3};
        return reindex;
      }
      case Core::FE::CellType::hex8:
      {
        static constexpr std::array reindex{0, 1, 3, 2, 4, 5, 7, 6};
        return reindex;
      }
      case Core::FE::CellType::hex27:
      {
        static constexpr std::array reindex{// vertices
            0, 1, 3, 2, 4, 5, 7, 6,
            // lines
            11, 9, 8, 10, 19, 17, 16, 18, 12, 13, 15, 14,
            // faces
            24, 22, 21, 23, 20, 25,
            // center
            26};
        return reindex;
      }
      default:
      {
        FOUR_C_THROW(
            "Unsupported cell type '{}'.", Core::FE::cell_type_to_string(cell_type).c_str());
      }
    }
  }

  /**
   * Given a 4C element, extract the GIDs of its nodes and rearrange them to be compatible with
   * deal.II. Also, return the element center which we assume uniquely identifies the element.
   */
  template <int spacedim>
  dealii::Point<spacedim> vertices_to_dealii(
      const Core::Elements::Element* element, std::vector<unsigned>& vertex_gids)
  {
    auto reindexing = reindex_dealii_to_four_c(element->shape());

    switch (element->shape())
    {
      case Core::FE::CellType::line2:
      case Core::FE::CellType::tet4:
      case Core::FE::CellType::hex8:
      {
        dealii::Point<spacedim> element_center;
        vertex_gids.resize(element->num_node());

        for (int lid = 0; lid < element->num_node(); ++lid)
        {
          const auto& node = element->nodes()[reindexing[lid]];
          vertex_gids[lid] = node->id();
          for (unsigned d = 0; d < spacedim; ++d) element_center[d] += node->x()[d];
        }

        // Normalize the center
        element_center /= element->num_node();
        return element_center;
      }
      case Core::FE::CellType::hex27:
      {
        dealii::Point<spacedim> element_center;
        vertex_gids.resize(8);

        // Only require the first 8 nodes for deal.II
        for (int lid = 0; lid < 8; ++lid)
        {
          const auto& node = element->nodes()[reindexing[lid]];
          vertex_gids[lid] = node->id();
          for (unsigned d = 0; d < spacedim; ++d) element_center[d] += node->x()[d];
        }
        // Normalize the center
        element_center *= 0.125;
        return element_center;
      }
      default:
        FOUR_C_THROW(
            "Unsupported cell type '{}'.", Core::FE::cell_type_to_string(element->shape()).c_str());
    }
  }


  /**
   * The name of the deal.II FiniteElement that corresponds to the given 4C cell type.
   */
  constexpr std::string dealii_fe_name(Core::FE::CellType cell_type)
  {
    switch (cell_type)
    {
      case Core::FE::CellType::line2:
        return "FE_Q(1)";
      case Core::FE::CellType::tet4:
        return "FE_SimplexP(1)";
      case Core::FE::CellType::hex8:
        return "FE_Q(1)";
      case Core::FE::CellType::hex27:
        return "FE_Q(2)";
      default:
        FOUR_C_THROW(
            "Unsupported cell type '{}'.", Core::FE::cell_type_to_string(cell_type).c_str());
    }
  }

  /**
   * Create the dealii::hp::FECollection with all FE types that appear in the given @p
   * discretization. This also included FEs that appear only on other MPI ranks. In addition, this
   * function returns the names of these elements in the same order as in the FECollection.
   *
   * @note The ordering of the FEs in the collection is the same on all ranks.
   */
  template <int dim, int spacedim>
  std::pair<dealii::hp::FECollection<dim, spacedim>, std::vector<std::string>>
  create_required_finite_element_collection(const Core::FE::Discretization& discretization)
  {
    // First, determine all FEs we require locally
    int max_num_dof_per_node{};
    std::set<std::string> local_dealii_fes;

    const MPI_Comm comm = discretization.get_comm();

    for (int i = 0; i < discretization.num_my_row_elements(); ++i)
    {
      const auto* four_c_element = discretization.l_row_element(i);
      max_num_dof_per_node = std::max(
          max_num_dof_per_node, four_c_element->num_dof_per_node(*four_c_element->nodes()[0]));
      local_dealii_fes.emplace(dealii_fe_name(four_c_element->shape()));
    }

    max_num_dof_per_node = dealii::Utilities::MPI::max(max_num_dof_per_node, comm);

    // Communicate the required deal.II FEs
    const auto all_dealii_fe_names = std::invoke(
        [&]()
        {
          std::vector<std::string> local_dealii_fes_vector(
              local_dealii_fes.begin(), local_dealii_fes.end());
          std::vector<std::vector<std::string>> all_dealii_fes_vector =
              dealii::Utilities::MPI::all_gather(comm, local_dealii_fes_vector);

          std::set<std::string> all_dealii_fes;
          for (const auto& my : all_dealii_fes_vector)
          {
            for (const auto& fe : my)
            {
              all_dealii_fes.emplace(fe);
            }
          }
          return std::vector<std::string>(all_dealii_fes.begin(), all_dealii_fes.end());
        });

    // create the deal.II FiniteElement as a collection
    dealii::hp::FECollection<dim, spacedim> fe_collection;

    for (const auto& fe_string : all_dealii_fe_names)
    {
      const auto fe = std::invoke(
          [&]() -> std::unique_ptr<dealii::FiniteElement<dim, spacedim>>
          {
            // NOTE: work around a limitation in deal.II: the convenience getter is not
            // implemented for simplex
            if (fe_string == "FE_SimplexP(1)")
            {
              return std::make_unique<dealii::FE_SimplexP<dim, spacedim>>(1);
            }
            else
              return dealii::FETools::get_fe_by_name<dim, spacedim>(fe_string);
          });

      if (max_num_dof_per_node == 1)
        fe_collection.push_back(*fe);
      else
        fe_collection.push_back(dealii::FESystem<dim, spacedim>(*fe, max_num_dof_per_node));
    }

    return {fe_collection, all_dealii_fe_names};
  }

}  // namespace DealiiWrappers::ElementConversion

FOUR_C_NAMESPACE_CLOSE

#endif
