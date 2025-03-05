// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_deal_ii_triangulation.hpp"

#include "4C_deal_ii_context_implementation.hpp"
#include "4C_deal_ii_element_conversion.hpp"
#include "4C_utils_exceptions.hpp"

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>

FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers
{
  template <int dim, int spacedim>
  Context<dim, spacedim> create_triangulation(
      dealii::Triangulation<dim, spacedim>& tria, const Core::FE::Discretization& discretization)
  {
    static_assert(spacedim == 3);

    FOUR_C_ASSERT_ALWAYS(discretization.filled(), "Discretization must be filled.");

    const MPI_Comm comm = discretization.get_comm();


    // Step 1)
    //
    // dealii::Triangulation and dealii::p:d:Triangulation expect the full coarse mesh. Thus, we
    // need to gather all the data from the distributed Discretization.
    dealii::TriangulationDescription::Description<dim, spacedim> construction_data;
    construction_data.comm = comm;

    // Step 1a)
    // copy the node coordinates and gids
    {
      std::vector<dealii::Point<spacedim>> my_coarse_cell_vertices(
          discretization.num_my_row_nodes());
      std::vector<int> my_node_gids(discretization.num_my_row_nodes());

      for (int i = 0; i < discretization.num_my_row_nodes(); ++i)
      {
        const auto* node = discretization.l_row_node(i);

        for (unsigned k = 0; k < spacedim; ++k) my_coarse_cell_vertices[i][k] = node->x()[k];
        my_node_gids[i] = node->id();
      }

      // communicate vertices and gids and sort them into the local data structure

      const auto all_coarse_cell_vertices =
          dealii::Utilities::MPI::all_gather(MPI_COMM_WORLD, my_coarse_cell_vertices);
      const auto all_node_gids = dealii::Utilities::MPI::all_gather(MPI_COMM_WORLD, my_node_gids);

      FOUR_C_ASSERT(all_coarse_cell_vertices.size() == all_node_gids.size(),
          "The number of communicated vertices does not match the number of GIDs.");

      [[maybe_unused]] const auto communicated_vertices =
          std::accumulate(all_coarse_cell_vertices.begin(), all_coarse_cell_vertices.end(), 0,
              [](int n, auto& v) { return n + v.size(); });
      FOUR_C_ASSERT(communicated_vertices == discretization.num_global_nodes(),
          "The number of communicated vertices does not match the number of global nodes.");


      // Fill the gathered data into the data structure for deal.II
      construction_data.coarse_cell_vertices.resize(discretization.num_global_nodes());
      for (unsigned rank = 0; rank < all_coarse_cell_vertices.size(); ++rank)
      {
        const auto& gids_for_rank = all_node_gids[rank];
        for (unsigned gid_index = 0; gid_index < gids_for_rank.size(); ++gid_index)
        {
          const int gid = gids_for_rank[gid_index];
          construction_data.coarse_cell_vertices[gid] = all_coarse_cell_vertices[rank][gid_index];
        }
      }
    }

    // Step 1b)
    // Copy the element connectivity, owning rank and center
    // This information will be necessary to partition a fullydistributed Triangulation in the same
    // manner as the input Core::FE::Discretization

    std::vector<std::vector<unsigned>> my_cell_vertices(discretization.num_my_row_elements());

    // communicate additionally: GID and center of element to later construct the mapping from
    // elements to deal.II cells
    std::vector<unsigned> my_element_gids(discretization.num_my_row_elements());
    std::vector<dealii::Point<spacedim>> my_element_centers(discretization.num_my_row_elements());

    for (int i_ele = 0; i_ele < discretization.num_my_row_elements(); ++i_ele)
    {
      const auto* element = discretization.l_row_element(i_ele);
      my_element_gids[i_ele] = element->id();


      my_element_centers[i_ele] =
          ElementConversion::vertices_to_dealii<spacedim>(element, my_cell_vertices[i_ele]);
    }

    const auto all_cell_vertices =
        dealii::Utilities::MPI::all_gather(MPI_COMM_WORLD, my_cell_vertices);

    const auto all_element_gids =
        dealii::Utilities::MPI::all_gather(MPI_COMM_WORLD, my_element_gids);

    const auto all_element_centers =
        dealii::Utilities::MPI::all_gather(MPI_COMM_WORLD, my_element_centers);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    {
      const auto communicated_cells = std::accumulate(all_cell_vertices.begin(),
          all_cell_vertices.end(), 0, [](int n, auto& v) { return n + v.size(); });
      FOUR_C_ASSERT(communicated_cells == discretization.num_global_elements(),
          "The number of communicated cells does not match the number of global elements.");

      const auto communicated_gids = std::accumulate(all_element_gids.begin(),
          all_element_gids.end(), 0, [](int n, auto& v) { return n + v.size(); });
      FOUR_C_ASSERT(communicated_gids == discretization.num_global_elements(),
          "The number of communicated GIDs does not match the number of global elements.");

      const auto communicated_centers = std::accumulate(all_element_centers.begin(),
          all_element_centers.end(), 0, [](int n, auto& v) { return n + v.size(); });
      FOUR_C_ASSERT(communicated_centers == discretization.num_global_elements(),
          "The number of communicated centers does not match the number of global elements.");
    }
#endif

    {
      construction_data.coarse_cells.reserve(discretization.num_global_elements());
      for (const auto& cell_vertices_for_rank : all_cell_vertices)
      {
        for (auto& vertices : cell_vertices_for_rank)
        {
          auto& cell_data = construction_data.coarse_cells.emplace_back(dealii::CellData<dim>{});
          cell_data.vertices = std::move(vertices);
        }
      }
    }


    Context<dim, spacedim> context{
        std::make_shared<Internal::ContextImplementation<dim, spacedim>>()};

    // Step 2)
    //
    // At this point we gathered all cells and vertices. We use them to create a Triangulation
    // Note: we can only create the Context that maps between the two discretizations when using pfT
    {
      // Next we have to sort them into a specific order expected by deal.II
      dealii::GridTools::invert_all_negative_measure_cells(
          construction_data.coarse_cell_vertices, construction_data.coarse_cells);

      dealii::GridTools::consistently_order_cells(construction_data.coarse_cells);

      // Use the gathered data depending on the exact type of Triangulation we want to create.
      if (const auto distributed_tria =
              dynamic_cast<dealii::parallel::distributed::Triangulation<dim, spacedim>*>(&tria))
      {
        // For a distributed Triangulation, the coarse mesh is necessary on all processes.
        distributed_tria->create_triangulation(construction_data.coarse_cell_vertices,
            construction_data.coarse_cells, dealii::SubCellData{});
      }
      else if (const auto fully_distributed_tria =
                   dynamic_cast<dealii::parallel::fullydistributed::Triangulation<dim, spacedim>*>(
                       &tria))
      {
        // Define the function that creates the triangulation on the root process in a group
        const auto serial_grid_generator = [&construction_data](auto& tria_serial)
        { tria_serial.create_triangulation(construction_data); };

        // Define the function that partitions the triangulation on the root process in a group
        const auto serial_grid_partitioner = [&](dealii::Triangulation<dim, spacedim>& tria_serial,
                                                 const MPI_Comm& /*mpi_comm*/,
                                                 const unsigned int /*group_size*/)
        {
          for (const auto& cell : tria_serial.active_cell_iterators())
          {
            FOUR_C_ASSERT(
                cell->is_locally_owned(), "The cell is not locally owned, but should be.");
            const auto& cell_center = cell->center();

            for (unsigned rank = 0; rank < all_element_centers.size(); ++rank)
            {
              const auto found =
                  std::find_if(all_element_centers[rank].begin(), all_element_centers[rank].end(),
                      [&](const auto& center) { return center.distance(cell_center) < 1e-14; });
              if (found != all_element_centers[rank].end())
              {
                cell->set_subdomain_id(rank);
              }
            }
          }
        };

        // Create and partition the Triangulation only once and communicate the resulting
        // description data to all processes...
        const auto fully_partitioned_description = dealii::TriangulationDescription::Utilities::
            create_description_from_triangulation_in_groups<dim, spacedim>(serial_grid_generator,
                serial_grid_partitioner, fully_distributed_tria->get_communicator(),
                /*group_size*/ dealii::Utilities::MPI::n_mpi_processes(comm));

        // ... and construct the actual fully distributed Triangulation with the correct data
        fully_distributed_tria->create_triangulation(fully_partitioned_description);

        // This only works for a p:f:T with identical partitioning of cells
        for (const auto& cell : tria.active_cell_iterators())
        {
          if (!cell->is_locally_owned()) continue;

          const auto& cell_center = cell->center();
          const auto found = std::find_if(my_element_centers.begin(), my_element_centers.end(),
              [&](const auto& center) { return center.distance(cell_center) < 1e-14; });

          FOUR_C_ASSERT(found != my_element_centers.end(),
              "The cell center does not match any of the element centers. This should not happen.");

          const auto local_index = std::distance(my_element_centers.begin(), found);

          context.pimpl_->cell_index_to_element_lid[cell->index()] = local_index;
        }

        // Determine all FiniteElement objects that are required.
        std::tie(context.pimpl_->finite_elements, context.pimpl_->finite_element_names) =
            ElementConversion::create_required_finite_element_collection<dim, spacedim>(
                discretization);

        // Create Mapping for potentially curved boundary approximation
        Internal::fill_mapping(context, discretization);
      }
      else
      {
        // If we have a plain serial Triangulation, just pass the fully redundant data.
        tria.create_triangulation(construction_data);
      }
    }
    FOUR_C_ASSERT(tria.n_global_coarse_cells() ==
                      static_cast<std::size_t>(discretization.num_global_elements()),
        "The number of active cells in the triangulation does not match the number of elements.");

    return context;
  }

  // --- explicit instantiations --- //

  template Context<3, 3> create_triangulation<3, 3>(
      dealii::Triangulation<3, 3>&, const Core::FE::Discretization&);

  template Context<1, 3> create_triangulation<1, 3>(
      dealii::Triangulation<1, 3>&, const Core::FE::Discretization&);

}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE
