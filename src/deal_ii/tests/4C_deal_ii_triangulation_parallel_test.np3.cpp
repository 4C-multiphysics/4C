// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_deal_ii_context.hpp"
#include "4C_deal_ii_create_discretization_helper_test.hpp"
#include "4C_deal_ii_triangulation.hpp"

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/numerics/data_out.h>

namespace
{
  using namespace FourC;

  // To better understand and debug this test suite this flag can be activated locally.
  const bool do_output = false;

  template <int dim, int spacedim>
  void output_triangulation(
      dealii::Triangulation<dim, spacedim>& triangulation, const std::string& file_name)
  {
    if (do_output)
    {
      dealii::DataOut<dim, spacedim> data_out;
      data_out.attach_triangulation(triangulation);

      dealii::Vector<float> partitioning(triangulation.n_active_cells());
      for (unsigned int i = 0; i < partitioning.size(); ++i)
        partitioning(i) = triangulation.locally_owned_subdomain();
      data_out.add_data_vector(partitioning, "partitioning");

      data_out.build_patches();
      data_out.write_vtu_in_parallel(file_name, MPI_COMM_WORLD);
    }
  }

  TEST(CreateTriangulation, SerialTriaFromPartitionedDiscretization)
  {
    constexpr int dim = 3;
    dealii::Triangulation<dim> tria;
    const auto comm = MPI_COMM_WORLD;

    Core::FE::Discretization discret{"empty", comm, dim};
    TESTING::fill_discretization_hyper_cube(discret, 2, comm);

    DealiiWrappers::create_triangulation(tria, discret);

    EXPECT_EQ(tria.n_active_cells(), 8);
    EXPECT_EQ(tria.n_used_vertices(), 27);

    output_triangulation(tria, "SerialTriaFromPartitionedDiscretization.vtu");
  }


  TEST(CreateTriangulation, FullyDistributedTria64Cells)
  {
    constexpr int dim = 3;
    dealii::parallel::fullydistributed::Triangulation<dim> tria(MPI_COMM_WORLD);
    const auto comm = MPI_COMM_WORLD;

    Core::FE::Discretization discret{"empty", comm, dim};
    TESTING::fill_discretization_hyper_cube(discret, 4, comm);

    DealiiWrappers::create_triangulation(tria, discret);

    EXPECT_EQ(tria.n_global_active_cells(), 64);

    output_triangulation(tria, "FullyDistributedTria64Cells.vtu");
  }

  TEST(CreateTriangulation, FullyDistributedTriaTree)
  {
    dealii::parallel::fullydistributed::Triangulation<1, 3> tria(MPI_COMM_WORLD);
    const auto comm = MPI_COMM_WORLD;

    Core::FE::Discretization discret{"tree", comm, 3};
    TESTING::fill_tree_1d_lines(discret, 5, comm);

    DealiiWrappers::create_triangulation(tria, discret);
    EXPECT_EQ(tria.n_global_active_cells(), 31);

    output_triangulation(tria, "FullyDistributedTriaTree.vtu");
  }

  TEST(CreateTriangulation, UncurvedHex27)
  {
    constexpr int dim = 3;
    dealii::Triangulation<dim> tria;
    const auto comm = MPI_COMM_WORLD;

    Core::FE::Discretization discret{"empty", comm, dim};
    TESTING::fill_undeformed_hex27(discret, comm);

    DealiiWrappers::create_triangulation(tria, discret);
    EXPECT_EQ(tria.n_active_cells(), 1);
    EXPECT_EQ(tria.n_used_vertices(), 27);

    output_triangulation(tria, "UncurvedHex27.vtu");
  }

  TEST(CreateTriangulation, DistortedHex27)
  {
    constexpr int dim = 3;
    dealii::Triangulation<dim> tria;
    const auto comm = MPI_COMM_WORLD;

    Core::FE::Discretization discret{"empty", comm, dim};
    TESTING::fill_deformed_hex27(discret, comm);

    DealiiWrappers::create_triangulation(tria, discret);
    EXPECT_EQ(tria.n_active_cells(), 1);
    EXPECT_EQ(tria.n_used_vertices(), 27);

    output_triangulation(tria, "DistortedHex27.vtu");
  }

  TEST(CreateTriangulation, CylindricalHex27)
  {
    constexpr int dim = 3;
    dealii::Triangulation<dim> tria;
    const auto comm = MPI_COMM_WORLD;

    Core::FE::Discretization discret{"empty", comm, dim};
    TESTING::fill_cylindrical_hex27(discret, comm);

    DealiiWrappers::create_triangulation(tria, discret);
    EXPECT_EQ(tria.n_active_cells(), 1);
    EXPECT_EQ(tria.n_used_vertices(), 27);

    output_triangulation(tria, "CylindricalHex27.vtu");
  }

}  // namespace
