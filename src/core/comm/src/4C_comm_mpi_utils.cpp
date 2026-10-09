// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_comm_mpi_utils.hpp"

#include "4C_utils_singleton_owner.hpp"

#include <Epetra_MpiComm.h>

FOUR_C_NAMESPACE_OPEN

MPI_Comm Core::Communication::unpack_epetra_comm(const Epetra_Comm& comm)
{
  const Epetra_MpiComm& mpi_comm = dynamic_cast<const Epetra_MpiComm&>(comm);
  return mpi_comm.Comm();
}

bool Core::Communication::same_mpi_comm(const MPI_Comm& mpi_comm_a, const MPI_Comm& mpi_comm_b)
{
  int result = -1;
  MPI_Comm_compare(mpi_comm_a, mpi_comm_b, &result);
  return result == MPI_IDENT || result == MPI_CONGRUENT;
}

const Epetra_Comm& Core::Communication::as_epetra_comm(MPI_Comm comm)
{
  static auto epetra_comm_cache = Core::Utils::make_singleton_map<MPI_Comm>(
      [](MPI_Comm comm_in) { return std::unique_ptr<Epetra_Comm>(new Epetra_MpiComm(comm_in)); });

  return *epetra_comm_cache[comm].instance(Core::Utils::SingletonAction::create, comm);
}

int Core::Communication::my_mpi_rank(MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

int Core::Communication::num_mpi_ranks(MPI_Comm comm)
{
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}

void Core::Communication::barrier(MPI_Comm comm) { MPI_Barrier(comm); }

FOUR_C_NAMESPACE_CLOSE