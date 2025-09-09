// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_deal_ii_triangulation.hpp"

#include "4C_deal_ii_context.hpp"
#include "4C_deal_ii_element_conversion.hpp"
#include "4C_utils_exceptions.hpp"

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/grid/grid_tools.h>

FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers
{
  template <int dim, int spacedim>
  Context<dim, spacedim> create_triangulation(
      dealii::Triangulation<dim, spacedim>& tria, const Core::FE::Discretization& discretization)
  {
    FOUR_C_ASSERT_ALWAYS(discretization.filled(), "Discretization must be filled.");
    const MPI_Comm comm = discretization.get_comm();

    // Now build up the actual context object that will be used to map between the
    // Core::FE::Discretization and the deal.II Triangulation.
    Context<dim, spacedim> context(tria, discretization);

    // Determine all FiniteElement objects that are required.
    std::vector<std::string> finite_element_names;
    std::tie(context.finite_elements_, finite_element_names) =
        create_required_finite_element_collection<dim, spacedim>(discretization);


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


      my_element_centers[i_ele] = ConversionTools::FourCToDeal::vertices_to_dealii<spacedim>(
          element, my_cell_vertices[i_ele]);
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

    // add bool to check if we only have standard hex cells
    // since otherwise some functionality in deal.II does not work
    bool only_standard_hex_cells = true;
    {
      construction_data.coarse_cells.reserve(discretization.num_global_elements());
      for (const auto& cell_vertices_for_rank : all_cell_vertices)
      {
        for (auto& vertices : cell_vertices_for_rank)
        {
          auto& cell_data = construction_data.coarse_cells.emplace_back(dealii::CellData<dim>{});
          cell_data.vertices = std::move(vertices);
          if (cell_data.vertices.size() != dealii::GeometryInfo<dim>::vertices_per_cell)
          {
            only_standard_hex_cells = false;
          }
        }
      }
    }



    // Step 2)
    //
    // At this point we gathered all cells and vertices. We use them to create a Triangulation
    // Note: we can only create the Context that maps between the two discretizations when using pfT
    {
      // Next we have to sort them into a specific order expected by deal.II
      dealii::GridTools::invert_all_negative_measure_cells(
          construction_data.coarse_cell_vertices, construction_data.coarse_cells);


      // check if we have a hex mesh and if so, consistently order the cells.
      // The deal.II function below only works for hexahedral meshes
      // TODO: handle tet meshes as well
      if (only_standard_hex_cells)
        dealii::GridTools::consistently_order_cells(construction_data.coarse_cells);

      if (const auto fully_distributed_tria =
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

        context.active_cell_index_to_element_lid_.resize(tria.n_active_cells());
        context.active_fe_indices_.resize(tria.n_active_cells());

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
          context.active_cell_index_to_element_lid_[cell->active_cell_index()] = local_index;

          const auto* local_element = discretization.l_row_element(local_index);
          auto fe_iter = std::find(finite_element_names.begin(), finite_element_names.end(),
              ConversionTools::FourCToDeal::dealii_fe_name(local_element->shape()));
          context.active_fe_indices_[cell->active_cell_index()] =
              static_cast<unsigned int>(std::distance(finite_element_names.begin(), fe_iter));
        }
      }
      // We cannot handle any other parallel Triangulation types yet.
      else if (dynamic_cast<dealii::parallel::TriangulationBase<dim, spacedim>*>(&tria) != nullptr)
      {
        FOUR_C_THROW(
            "The Triangulation is parallel but not a parallel::fullydistributed::Triangulation. "
            "This is not yet implemented.");
      }
      else
      {
        // If we have a plain serial Triangulation, just pass the fully redundant data.
        tria.create_triangulation(construction_data);

        context.active_cell_index_to_element_lid_.resize(tria.n_active_cells());
        context.active_fe_indices_.resize(tria.n_active_cells());

        // Now setup the cell data for the serial Triangulation.
        // Since the 4C discretization is partitioned it does not make sens to set global indices
        // on the cells that correspond to elements that are not locally owned, instead we set all
        // the relevant data there to invalid values.
        // namely this concerns:
        // - the active_cell_index_to_element_lid_ (-1)
        // - the active_fe_indices_ (dealii::numbers::invalid_unsigned_int)
        for (const auto& cell : tria.active_cell_iterators())
        {
          FOUR_C_ASSERT(cell->is_locally_owned(),
              "The cell is not locally owned, but should be since we have a serial Triangulation.");
          const auto& cell_center = cell->center();
          const auto found = std::find_if(my_element_centers.begin(), my_element_centers.end(),
              [&](const auto& center) { return center.distance(cell_center) < 1e-14; });

          // If we dont find the cell center as a local element we have to set the
          // data to invalid values.
          if (found == my_element_centers.end())
          {
            // If the cell center does not match any of the element centers, we set the
            // cell_index_to_element_lid to -1 and the active_fe_indices to
            // dealii::numbers::invalid_unsigned_int.
            context.active_cell_index_to_element_lid_[cell->active_cell_index()] = -1;
            context.active_fe_indices_[cell->active_cell_index()] =
                dealii::numbers::invalid_unsigned_int;
            continue;
          }
          // Assert that the cell center is physically located in the discretization
#ifdef FOUR_C_ENABLE_ASSERTIONS
          bool found_center = false;
          for (unsigned rank = 0; rank < all_element_centers.size(); ++rank)
          {
            const auto found_on_rank =
                std::find_if(all_element_centers[rank].begin(), all_element_centers[rank].end(),
                    [&](const auto& center) { return center.distance(cell_center) < 1e-14; });
            if (found_on_rank != all_element_centers[rank].end())
            {
              found_center = true;
              break;
            }
          }
          FOUR_C_ASSERT(found_center,
              "The cell center does not match any of the element centers on any rank. This should "
              "not happen.");
#endif

          // now set the data for the cell
          const auto local_index = std::distance(my_element_centers.begin(), found);
          context.active_cell_index_to_element_lid_[cell->active_cell_index()] = local_index;

          const auto* local_element = discretization.l_row_element(local_index);
          auto fe_iter = std::find(finite_element_names.begin(), finite_element_names.end(),
              ConversionTools::FourCToDeal::dealii_fe_name(local_element->shape()));
          context.active_fe_indices_[cell->active_cell_index()] =
              static_cast<unsigned int>(std::distance(finite_element_names.begin(), fe_iter));
        }
      }
    }
    FOUR_C_ASSERT(tria.n_global_coarse_cells() ==
                      static_cast<std::size_t>(discretization.num_global_elements()),
        "The number of active cells in the triangulation does not match the number of elements.");

    return context;
  }



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
      local_dealii_fes.emplace(
          ConversionTools::FourCToDeal::dealii_fe_name(four_c_element->shape()));
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
            {
              return dealii::FETools::get_fe_by_name<dim, spacedim>(fe_string);
            }
          });

      if (max_num_dof_per_node == 1)
        fe_collection.push_back(*fe);
      else
        fe_collection.push_back(dealii::FESystem<dim, spacedim>(*fe, max_num_dof_per_node));
    }

    return {fe_collection, all_dealii_fe_names};
  }



  // --- explicit instantiations --- //

  template Context<2, 2> create_triangulation<2, 2>(
      dealii::Triangulation<2, 2>&, const Core::FE::Discretization&);

  template Context<3, 3> create_triangulation<3, 3>(
      dealii::Triangulation<3, 3>&, const Core::FE::Discretization&);

  template Context<1, 3> create_triangulation<1, 3>(
      dealii::Triangulation<1, 3>&, const Core::FE::Discretization&);

  template std::pair<dealii::hp::FECollection<2, 2>, std::vector<std::string>>
  create_required_finite_element_collection(const Core::FE::Discretization& discretization);
  template std::pair<dealii::hp::FECollection<3, 3>, std::vector<std::string>>
  create_required_finite_element_collection(const Core::FE::Discretization& discretization);
  template std::pair<dealii::hp::FECollection<1, 3>, std::vector<std::string>>
  create_required_finite_element_collection(const Core::FE::Discretization& discretization);

}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE
