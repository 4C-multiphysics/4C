// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_MAP_HPP
#define FOUR_C_LINALG_MAP_HPP

#include "4C_config.hpp"

#include "4C_linalg.hpp"
#include "4C_linalg_view.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_owner_or_view.hpp"

#include <Epetra_BlockMap.h>
#include <Epetra_LocalMap.h>
#include <Epetra_Map.h>
#include <mpi.h>

#include <memory>
#include <span>
#include <variant>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class Map
  {
   public:
    /**
     * @brief Construct a Map describing the distribution of global indices.
     *
     * Creates a mapping of a contiguous range of global indices onto the
     * processes of the given MPI communicator.
     *
     * @param NumGlobalElements Total number of global elements in the map.
     * @param IndexBase The global index of the first element (typically 0 or 1).
     * @param Comm The MPI communicator over which the map is defined.
     * @param mode Specifies whether the map is globally distributed or locally replicated. Defaults
     * to LocalGlobal::globally_distributed.
     *
     * @note In globally distributed mode, each rank owns a subset of the global index range. In
     * locally replicated mode, all ranks conceptually own all global indices. This is intended for
     * small data structures where replication is acceptable.
     */
    Map(int NumGlobalElements, int IndexBase, const MPI_Comm& Comm,
        const Core::LinAlg::LocalGlobal mode = Core::LinAlg::LocalGlobal::globally_distributed);

    Map(int NumGlobalElements, int NumMyElements, int IndexBase, const MPI_Comm& Comm);

    Map(int NumGlobalElements, std::span<const int> myGlobalElements, int IndexBase,
        const MPI_Comm& Comm);

    Map(const Map& Source);

    /// Copy constructor from Epetra_Map
    explicit Map(const Epetra_Map& Source);

    /// Copy constructor from Epetra_BlockMap
    explicit Map(const Epetra_BlockMap& Source);

    ~Map() = default;

    Map& operator=(const Map& other);

    //! Print object to the output stream
    void print(std::ostream& os) const { wrapped().Print(os); }

    //! Returns a reference of the Epetra_Map if available.
    const Epetra_Map& get_epetra_map() const
    {
      auto* map = std::get_if<Utils::OwnerOrView<Epetra_Map>>(&map_);
      if (map == nullptr)
      {
        FOUR_C_THROW(
            "This Map is based on an Epetra_BlockMap, not an Epetra_Map. This cast is not "
            "possible.");
      }
      return **map;
    }

    //! Returns a reference of the Epetra_Map if available.
    Epetra_Map& get_epetra_map()
    {
      auto* map = std::get_if<Utils::OwnerOrView<Epetra_Map>>(&map_);
      if (map == nullptr)
      {
        FOUR_C_THROW(
            "This Map is based on an Epetra_BlockMap, not an Epetra_Map. This cast is not "
            "possible.");
      }
      return **map;
    }

    //! Returns a const reference to the underlying Epetra_BlockMap.
    const Epetra_BlockMap& get_epetra_block_map() const { return wrapped(); }

    //! Returns a reference to the underlying Epetra_BlockMap.
    Epetra_BlockMap& get_epetra_block_map() { return wrapped(); }

    //! Returns true if this and Map are identical maps
    bool same_as(const Map& other) const { return wrapped().SameAs((other.wrapped())); }

    //! Returns true if this and Map have identical point-wise structure
    bool point_same_as(const Map& Map) const
    {
      return wrapped().PointSameAs(Map.get_epetra_block_map());
    }

    //! Number of elements across all processors.
    int num_global_elements() const { return wrapped().NumGlobalElements(); }

    //! Number of elements on the calling processor.
    int num_my_elements() const { return wrapped().NumMyElements(); }

    //! returns the index base for this map.
    int index_base() const { return wrapped().IndexBase(); }

    //! Returns true if map is defined across more than one processor.
    bool distributed_global() const { return wrapped().DistributedGlobal(); }

    //! Returns true if this and Map are identical maps
    bool same_as(const Epetra_Map& other) const { return wrapped().SameAs(other); }

    //! Returns true if this and Map are identical maps
    bool same_as(const Epetra_BlockMap& other) const { return wrapped().SameAs(other); }

    //! Returns true if the GID passed in belongs to the calling processor in this map, otherwise
    //! returns false.
    bool my_gid(int GID_in) const { return wrapped().MyGID(GID_in); }

    //! Returns global ID of local ID, return IndexBase-1 if not found on this processor.
    int gid(int LID) const { return wrapped().GID(LID); }

    //! Returns the size of elements in the map; only valid if map has constant element size.
    int element_size() const { return wrapped().ElementSize(); }

    //! Returns the size of elements in the map; only valid if map has constant element size.
    int element_size(int LID) const { return wrapped().ElementSize(LID); }

    //! Returns the maximum global ID across the entire map.
    int max_all_gid() const { return wrapped().MaxAllGID(); }

    //! Returns the minimum global ID across the entire map.
    int min_all_gid() const { return wrapped().MinAllGID(); }

    //! Returns local ID of global ID, return -1 if not found on this processor.
    int lid(int GID) const { return wrapped().LID(GID); }

    //! Returns the processor IDs and corresponding local index value for a given list of global
    //! indices
    int remote_id_list(
        std::span<const int> gidList, std::span<int> pidList, std::span<int> lidList) const
    {
      FOUR_C_ASSERT(gidList.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
          "gidList.size() exceeds INT_MAX");

      FOUR_C_ASSERT(pidList.size() == gidList.size(), "pidList must have same size as gidList");

      FOUR_C_ASSERT(lidList.empty() || lidList.size() == gidList.size(),
          "lidList must be empty or have same size as gidList");

      return wrapped().RemoteIDList(static_cast<int>(gidList.size()), gidList.data(),
          pidList.data(), lidList.empty() ? nullptr : lidList.data());
    }

    //! Returns the minimum global ID owned by this processor.
    int min_my_gid(void) const { return wrapped().MinMyGID(); }

    //! Access function for Epetra_Comm communicator.
    MPI_Comm get_comm() const;

    //! Returns true if map GIDs are 1-to-1.
    bool unique_gids(void) const { return wrapped().UniqueGIDs(); }

    //! Pointer to internal array containing list of global IDs assigned to the calling processor.
    int* my_global_elements(void) const { return wrapped().MyGlobalElements(); }

    //! Maximum element size across all processors.
    int max_element_size(void) const { return wrapped().MaxElementSize(); }

    //! Puts list of global elements on this processor into the user-provided array.
    void my_global_elements(std::span<int> myGlobalElementList) const;

    //! Number of points across all processors.
    int num_global_points() const { return wrapped().NumGlobalPoints(); }

    //! Number of local points for this map; equals the sum of all element sizes on the calling
    //! processor.
    int num_my_points() const { return wrapped().NumMyPoints(); }

    //! Returns a pointer to the BlockMapData instance this BlockMap uses.
    const Epetra_BlockMapData* get_data_ptr() const { return wrapped().DataPtr(); }


    [[nodiscard]] static std::unique_ptr<Map> create_view(Epetra_Map& view);
    [[nodiscard]] static std::unique_ptr<const Map> create_view(const Epetra_Map& view);
    [[nodiscard]] static std::unique_ptr<Map> create_view(Epetra_BlockMap& view);
    [[nodiscard]] static std::unique_ptr<const Map> create_view(const Epetra_BlockMap& view);

   private:
    Map() = default;

    // This wrapper may have two different variants
    // Either it holds an Epetra_Map, Epetra_LocalMap or Epetra_BlockMap
    using MapVariant = std::variant<Utils::OwnerOrView<Epetra_Map>,
        Utils::OwnerOrView<Epetra_LocalMap>, Utils::OwnerOrView<Epetra_BlockMap>>;

    Epetra_BlockMap& wrapped()
    {
      if (auto* distr_ptr = std::get_if<Utils::OwnerOrView<Epetra_Map>>(&map_))
        return **distr_ptr;
      else if (auto* local_ptr = std::get_if<Utils::OwnerOrView<Epetra_LocalMap>>(&map_))
        return **local_ptr;
      else
        return *std::get<Utils::OwnerOrView<Epetra_BlockMap>>(map_);
    }

    const Epetra_BlockMap& wrapped() const
    {
      return const_cast<const Epetra_BlockMap&>(const_cast<Map*>(this)->wrapped());
    }

    //! stores an Epetra_BlockMap, Epetra_LocalMap or Epetra_Map
    MapVariant map_;
  };

  inline std::ostream& operator<<(std::ostream& os, const Map& m)
  {
    os << m.get_epetra_block_map();
    return os;
  }

  template <>
  struct EnableViewFor<Epetra_LocalMap>
  {
    using type = Map;
  };

  template <>
  struct EnableViewFor<Epetra_Map>
  {
    using type = Map;
  };

  template <>
  struct EnableViewFor<Epetra_BlockMap>
  {
    using type = Map;
  };

}  // namespace Core::LinAlg


FOUR_C_NAMESPACE_CLOSE

#endif
