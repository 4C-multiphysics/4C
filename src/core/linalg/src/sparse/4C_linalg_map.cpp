// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_linalg_map.hpp"

#include "4C_comm_utils.hpp"


// Do not lint the file for identifier names, since the naming of the Wrapper functions follow the
// naming of the Core::LinAlg::Map

// NOLINTBEGIN(readability-identifier-naming)

FOUR_C_NAMESPACE_OPEN

Core::LinAlg::Map::Map(int NumGlobalElements, int IndexBase, const Epetra_Comm& Comm)
    : map_(std::make_unique<Epetra_Map>(NumGlobalElements, IndexBase, Comm))
{
}

Core::LinAlg::Map::Map(int NumGlobalElements, int IndexBase, const MPI_Comm& Comm)
    : map_(std::make_unique<Epetra_Map>(
          NumGlobalElements, IndexBase, Core::Communication::as_epetra_comm(Comm)))
{
}

Core::LinAlg::Map::Map(
    int NumGlobalElements, int NumMyElements, int IndexBase, const Epetra_Comm& Comm)
    : map_(std::make_unique<Epetra_Map>(NumGlobalElements, NumMyElements, IndexBase, Comm))
{
}

Core::LinAlg::Map::Map(
    int NumGlobalElements, int NumMyElements, int IndexBase, const MPI_Comm& Comm)
    : map_(std::make_unique<Epetra_Map>(
          NumGlobalElements, NumMyElements, IndexBase, Core::Communication::as_epetra_comm(Comm)))
{
}

Core::LinAlg::Map::Map(int NumGlobalElements, int NumMyElements, const int* MyGlobalElements,
    int IndexBase, const Epetra_Comm& Comm)
    : map_(std::make_unique<Epetra_Map>(
          NumGlobalElements, NumMyElements, MyGlobalElements, IndexBase, Comm))
{
}

Core::LinAlg::Map::Map(int NumGlobalElements, int NumMyElements, const int* MyGlobalElements,
    int IndexBase, const MPI_Comm& Comm)
    : map_(std::make_unique<Epetra_Map>(NumGlobalElements, NumMyElements, MyGlobalElements,
          IndexBase, Core::Communication::as_epetra_comm(Comm)))
{
}

Core::LinAlg::Map::Map(const Map& Source)
    : map_(std::make_unique<Epetra_Map>(Source.get_epetra_map()))
{
}

Core::LinAlg::Map& Core::LinAlg::Map::operator=(const Map& other)
{
  *map_ = *other.map_;
  return *this;
}

std::unique_ptr<Core::LinAlg::Map> Core::LinAlg::Map::create_view(Epetra_Map& view)
{
  std::unique_ptr<Map> ret;
  ret->map_ = ConstPropagatingUniquePtr<Epetra_Map>(&view, [](Epetra_Map* ptr) {});
  return ret;
}

std::unique_ptr<const Core::LinAlg::Map> Core::LinAlg::Map::create_view(const Epetra_Map& view)
{
  std::unique_ptr<Map> ret;
  // We may safely cast away const here, since the returned object is const and we use a
  // ConstPropagatingUniquePtr to ensure that we only ever access the viewed map through
  // const methods.
  ret->map_ =
      ConstPropagatingUniquePtr<Epetra_Map>(const_cast<Epetra_Map*>(&view), [](Epetra_Map* ptr) {});
  return ret;
}

Core::LinAlg::Map::Map(const Epetra_Map& Source) : map_(std::make_unique<Epetra_Map>(Source)) {}

Core::LinAlg::Map::Map(const Epetra_BlockMap& Source)
{
  map_.obj = std::make_unique<Epetra_Map>(Source.NumGlobalElements(), Source.NumMyElements(),
      Source.MyGlobalElements(), Source.IndexBase(), Source.Comm());
}

// NOLINTEND(readability-identifier-naming)

FOUR_C_NAMESPACE_CLOSE
