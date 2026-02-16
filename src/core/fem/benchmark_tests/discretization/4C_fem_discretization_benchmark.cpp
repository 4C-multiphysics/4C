// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_unittest_utils_create_discretization_helper_test.hpp"

#include <benchmark/benchmark.h>

namespace
{
  using namespace FourC;

  constexpr int dim = 3;
  constexpr int subdivisions = 10;


  void discretization_loop_reference(benchmark::State& state)
  {
    auto comm = MPI_COMM_WORLD;
    Core::FE::Discretization discret{"hypercube", comm, dim};
    TESTING::fill_discretization_hyper_cube(discret, subdivisions, comm);

    for (auto _ : state)
    {
      int count = 0;
      for (auto node : discret.my_row_node_range())
      {
        count += node.local_id();
        count += node.global_id();
      }
      benchmark::DoNotOptimize(count);
    }
  }
  BENCHMARK(discretization_loop_reference);


  void discretization_loop_reference_access_user_pointer(benchmark::State& state)
  {
    auto comm = MPI_COMM_WORLD;
    Core::FE::Discretization discret{"hypercube", comm, dim};
    TESTING::fill_discretization_hyper_cube(discret, subdivisions, comm);

    for (auto _ : state)
    {
      int count = 0;
      for (auto node : discret.my_row_node_range())
      {
        auto* user_node = node.user_node();
        count += user_node->lid();
        count += user_node->id();
      }
      benchmark::DoNotOptimize(count);
    }
  }
  BENCHMARK(discretization_loop_reference_access_user_pointer);


  void discretization_loop_user_pointer(benchmark::State& state)
  {
    auto comm = MPI_COMM_WORLD;
    Core::FE::Discretization discret{"hypercube", comm, dim};
    TESTING::fill_discretization_hyper_cube(discret, subdivisions, comm);

    for (auto _ : state)
    {
      int count = 0;
      // Important to pull the size query out of the loop. Not implemented efficiently.
      const int n = discret.num_my_row_nodes();
      for (int i = 0; i < n; ++i)
      {
        const auto* node = discret.l_row_node(i);
        count += node->lid();
        count += node->id();
      }
      benchmark::DoNotOptimize(count);
    }
  }
  BENCHMARK(discretization_loop_user_pointer);


  void discretization_loop_reference_nested(benchmark::State& state)
  {
    auto comm = MPI_COMM_WORLD;
    Core::FE::Discretization discret{"hypercube", comm, dim};
    TESTING::fill_discretization_hyper_cube(discret, subdivisions, comm);

    for (auto _ : state)
    {
      int count = 0;
      for (auto node : discret.my_row_node_range())
      {
        for (auto element : node.adjacent_elements())
        {
          count += element.local_id();
          count += element.global_id();
        }
      }
      benchmark::DoNotOptimize(count);
    }
  }
  BENCHMARK(discretization_loop_reference_nested);


  void discretization_loop_user_pointer_nested(benchmark::State& state)
  {
    auto comm = MPI_COMM_WORLD;
    Core::FE::Discretization discret{"hypercube", comm, dim};
    TESTING::fill_discretization_hyper_cube(discret, subdivisions, comm);

    for (auto _ : state)
    {
      int count = 0;
      const int n_nodes = discret.num_my_row_nodes();
      for (int i = 0; i < n_nodes; ++i)
      {
        const auto* node = discret.l_row_node(i);
        {
          // Use the new interface which return a range of ElementRef objects
          for (auto ele : node->adjacent_elements())
          {
            count += ele.local_id();
            count += ele.global_id();
          }
        }
      }
      benchmark::DoNotOptimize(count);
    }
  }
  BENCHMARK(discretization_loop_user_pointer_nested);
}  // namespace
