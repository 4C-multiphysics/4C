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
#include "4C_fem_discretization.hpp"

#include <deal.II/numerics/data_out.h>
#include <Epetra_SerialComm.h>

namespace
{
  using namespace FourC;

  // To better understand and debug this test suite this flag can be activated locally.
  const bool do_output = true;

  template <int dim, int spacedim>
  void output_triangulation(
      dealii::Triangulation<dim, spacedim>& triangulation, const std::string& file_name)
  {
    if (do_output)
    {
      dealii::DataOut<dim, spacedim> data_out;
      data_out.attach_triangulation(triangulation);
      data_out.build_patches();

      data_out.write_vtu_in_parallel(file_name, MPI_COMM_WORLD);
    }
  }


  TEST(CreateTriangulation, SerialTriaOneCell)
  {
    constexpr int dim = 3;
    dealii::Triangulation<dim> tria;

    const auto comm = MPI_COMM_WORLD;
    Core::FE::Discretization discret{"one_cell", comm, 3};
    TESTING::fill_discretization_hyper_cube(discret, 1, comm);

    DealiiWrappers::create_triangulation(tria, discret);
    EXPECT_EQ(tria.n_active_cells(), 1);
    EXPECT_EQ(tria.n_used_vertices(), 8);

    output_triangulation(tria, "SerialTriaOneCell.vtu");
  }

  TEST(CreateTriangulation, SerialTriaEightCells)
  {
    constexpr int dim = 3;
    dealii::Triangulation<dim> tria;
    const auto comm = MPI_COMM_WORLD;

    Core::FE::Discretization discret{"eight_cells", comm, dim};
    TESTING::fill_discretization_hyper_cube(discret, 2, comm);

    DealiiWrappers::create_triangulation(tria, discret);
    EXPECT_EQ(tria.n_active_cells(), 8);
    EXPECT_EQ(tria.n_used_vertices(), 27);

    output_triangulation(tria, "SerialTriaEightCells.vtu");
  }

  TEST(CreateTriangulation, SerialTriaOneTet4)
  {
    constexpr int dim = 3;
    dealii::Triangulation<dim> tria;
    const auto comm = MPI_COMM_WORLD;

    Core::FE::Discretization discret{"one_tet", comm, dim};
    TESTING::fill_single_tet(discret);

    DealiiWrappers::create_triangulation(tria, discret);
    EXPECT_EQ(tria.n_active_cells(), 1);
    EXPECT_EQ(tria.n_used_vertices(), 4);

    output_triangulation(tria, "SerialTriaOneTet.vtu");
  }
}  // namespace
