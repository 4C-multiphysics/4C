// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_linalg_map.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils.hpp"
#include "4C_utils_epetra_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

Core::LinAlg::Map::Map(int NumGlobalElements, int IndexBase, const MPI_Comm& Comm)
    : map_(MapVariant(std::in_place_type<Utils::OwnerOrView<Epetra_Map>>,
          Utils::make_owner<Epetra_Map>(
              NumGlobalElements, IndexBase, Core::Communication::as_epetra_comm(Comm))))
{
}

Core::LinAlg::Map::Map(
    int NumGlobalElements, int NumMyElements, int IndexBase, const MPI_Comm& Comm)
    : map_(MapVariant(std::in_place_type<Utils::OwnerOrView<Epetra_Map>>,
          Utils::make_owner<Epetra_Map>(NumGlobalElements, NumMyElements, IndexBase,
              Core::Communication::as_epetra_comm(Comm))))
{
}

Core::LinAlg::Map::Map(int NumGlobalElements, int NumMyElements, const int* MyGlobalElements,
    int IndexBase, const MPI_Comm& Comm)
    : map_(MapVariant(std::in_place_type<Utils::OwnerOrView<Epetra_Map>>,
          Utils::make_owner<Epetra_Map>(NumGlobalElements, NumMyElements, MyGlobalElements,
              IndexBase, Core::Communication::as_epetra_comm(Comm))))
{
}

Core::LinAlg::Map::Map::Map(const Map& Source)
{
  if (std::holds_alternative<Utils::OwnerOrView<Epetra_Map>>(Source.map_))
  {
    const auto& map_view = std::get<Utils::OwnerOrView<Epetra_Map>>(Source.map_);
    map_ = Utils::make_owner<Epetra_Map>(*map_view);
  }
  else if (std::holds_alternative<Utils::OwnerOrView<Epetra_BlockMap>>(Source.map_))
  {
    const auto& block_view = std::get<Utils::OwnerOrView<Epetra_BlockMap>>(Source.map_);
    map_ = Utils::make_owner<Epetra_BlockMap>(*block_view);
  }
  else
  {
    FOUR_C_THROW("Map::Map(const Map&) - Unknown type in variant.");
  }
}

Core::LinAlg::Map& Core::LinAlg::Map::operator=(const Map& other)
{
  if (this != &other)
  {
    map_ = std::visit(
        [](const auto& wrapped) -> MapVariant
        {
          using T = std::decay_t<decltype(*wrapped)>;
          return Utils::make_owner<T>(*wrapped);
        },
        other.map_);
  }
  return *this;
}

MPI_Comm Core::LinAlg::Map::get_comm() const
{
  return Core::Communication::unpack_epetra_comm(wrapped().Comm());
}

std::unique_ptr<Core::LinAlg::Map> Core::LinAlg::Map::create_view(Epetra_Map& view)
{
  std::unique_ptr<Map> ret(new Map);

  ret->map_ =
      MapVariant(std::in_place_type<Utils::OwnerOrView<Epetra_Map>>, Utils::make_view(&view));

  return ret;
}

std::unique_ptr<const Core::LinAlg::Map> Core::LinAlg::Map::create_view(const Epetra_Map& view)
{
  std::unique_ptr<Map> ret(new Map);

  ret->map_ = MapVariant(std::in_place_type<Utils::OwnerOrView<Epetra_Map>>,
      Utils::make_view(const_cast<Epetra_Map*>(&view)));

  return ret;
}

std::unique_ptr<Core::LinAlg::Map> Core::LinAlg::Map::create_view(Epetra_BlockMap& view)
{
  std::unique_ptr<Map> ret(new Map);

  ret->map_ =
      MapVariant(std::in_place_type<Utils::OwnerOrView<Epetra_BlockMap>>, Utils::make_view(&view));

  return ret;
}

std::unique_ptr<const Core::LinAlg::Map> Core::LinAlg::Map::create_view(const Epetra_BlockMap& view)
{
  std::unique_ptr<Map> ret(new Map);

  ret->map_ = MapVariant(std::in_place_type<Utils::OwnerOrView<Epetra_BlockMap>>,
      Utils::make_view(const_cast<Epetra_BlockMap*>(&view)));

  return ret;
}

Core::LinAlg::Map::Map(const Epetra_Map& Source)
    : map_(MapVariant(std::in_place_type<Utils::OwnerOrView<Epetra_Map>>,
          Utils::make_owner<Epetra_Map>(Source)))
{
}

Core::LinAlg::Map::Map(const Epetra_BlockMap& Source)
    : map_(MapVariant(std::in_place_type<Utils::OwnerOrView<Epetra_BlockMap>>,
          Utils::make_owner<Epetra_BlockMap>(Source)))
{
}

void Core::LinAlg::Map::my_global_elements(int* MyGlobalElementList) const
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  CHECK_EPETRA_CALL(wrapped().MyGlobalElements(MyGlobalElementList));
#else
  wrapped().MyGlobalElements(MyGlobalElementList);
#endif
}

FOUR_C_NAMESPACE_CLOSE
