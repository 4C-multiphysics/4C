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
#include "4C_linalg_utils_exceptions.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"

#include <functional>


FOUR_C_NAMESPACE_OPEN

Core::LinAlg::Map::Map(int NumGlobalElements, int IndexBase, const MPI_Comm& Comm,
    const Core::LinAlg::LocalGlobal mode)
{
  if (mode == Core::LinAlg::LocalGlobal::globally_distributed)
  {
    map_ = MapVariant(std::in_place_type<Utils::OwnerOrView<Epetra_Map>>,
        Utils::make_owner<Epetra_Map>(
            NumGlobalElements, IndexBase, Core::Communication::as_epetra_comm(Comm)));
  }
  else if (mode == Core::LinAlg::LocalGlobal::locally_replicated)
  {
    map_ = MapVariant(std::in_place_type<Utils::OwnerOrView<Epetra_LocalMap>>,
        Utils::make_owner<Epetra_LocalMap>(
            NumGlobalElements, IndexBase, Core::Communication::as_epetra_comm(Comm)));
  }
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

void Core::LinAlg::Map::my_global_elements(std::span<int> myGlobalElementList) const
{
  ASSERT_EPETRA_CALL(wrapped().MyGlobalElements(myGlobalElementList.data()));
}

bool Core::LinAlg::Map::unique_my_gids() const
{
  const int n = this->num_my_elements();
  std::vector<int> gids(this->my_global_elements(), this->my_global_elements() + n);
  std::sort(gids.begin(), gids.end());
  return std::adjacent_find(gids.begin(), gids.end()) == gids.end();
}


bool Core::LinAlg::Map::contains_all_global_ids_of_map(const Map& map) const
{
  FOUR_C_ASSERT_ALWAYS(map.element_size() == 1 && element_size() == 1,
      "This comparison is currently only supported for element size 1! The given map has element "
      "size {}, and this map has element size {}",
      map.element_size(), this->element_size());
  FOUR_C_ASSERT_ALWAYS(
      this->unique_my_gids(), "This map must be valid: no duplicated gids on the same rank");
  FOUR_C_ASSERT_ALWAYS(
      map.unique_my_gids(), "Given map must be valid: no duplicated gids on the same rank");

  if (map.num_global_elements() == 0)
  {
    return true;  // empty maps are always subsets of this map
  }
  if (this->num_global_elements() == 0)
  {
    return false;
  }

  // create one-to-one maps from the underlying maps to ensure unambiguous vector import below
  const auto this_map_one_to_one = create_one_to_one();
  const auto map_one_to_one = map.create_one_to_one();

  // preliminary global size check
  if (map_one_to_one.num_global_elements() > this_map_one_to_one.num_global_elements())
  {
    return false;
  }

  // preliminary max/min global id check: the global ids of the given map must be within the bounds
  // imposed by this map
  if (map_one_to_one.min_all_gid() < this_map_one_to_one.min_all_gid() ||
      map_one_to_one.max_all_gid() > this_map_one_to_one.max_all_gid())
  {
    return false;
  }

  // verify map communicators
  FOUR_C_ASSERT_ALWAYS(
      Core::Communication::same_mpi_comm(this_map_one_to_one.get_comm(), map_one_to_one.get_comm()),
      "To use MPI_Allreduce below, the given maps must have at least congruent or identical "
      "communicators");

  // We perform the superset verification using vector import: from a vector using this map
  // into a vector using the given map. Thereby, we track the import count of each vector
  // component -> at the end, the vector using the given map should be filled with ones
  Core::LinAlg::Vector<int> this_map_gid_counts(this_map_one_to_one);
  this_map_gid_counts.put_value(1);

  Core::LinAlg::Vector<int> map_gid_counts(map_one_to_one);
  map_gid_counts.put_value(0);

  Core::LinAlg::Import importer(map_one_to_one, this_map_one_to_one);
  map_gid_counts.import(this_map_gid_counts, importer, Core::LinAlg::CombineMode::add);

  bool locally_contained = true;
  for (int i = 0; i < map_gid_counts.local_length(); ++i)
  {
    if (map_gid_counts.get_local_values()[i] < 1)
    {
      locally_contained = false;
      break;
    }
  }

  bool globally_contained =
      Core::Communication::all_reduce(locally_contained, std::logical_and<>{}, get_comm());
  return globally_contained;
}

FOUR_C_NAMESPACE_CLOSE
