// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_utils_singleton_owner.hpp"

#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include <mpi.h>

int main(int argc, char** argv)
{
  using namespace FourC;

  MPI_Init(&argc, &argv);
  struct CleanUpMPI
  {
    ~CleanUpMPI() { MPI_Finalize(); }
  } cleanup;
  // Kokkos should be initialized right after MPI.
  Kokkos::ScopeGuard kokkos_guard{};

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}