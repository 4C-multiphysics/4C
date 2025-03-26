// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_MAP_HPP
#define FOUR_C_LINALG_MAP_HPP


#include "4C_config.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_view.hpp"

#include <Epetra_Comm.h>
#include <Epetra_Map.h>

#include <memory>

FOUR_C_NAMESPACE_OPEN

// Do not lint the file for identifier names, since the naming of the Wrapper functions follow the
// naming of the Epetra_Map

// NOLINTBEGIN(readability-identifier-naming)

namespace Core::LinAlg
{

  class Map
  {
   public:
    //! Creates a Epetra_Map object based on Epetra_Comm
    Map(int NumGlobalElements, int IndexBase, const Epetra_Comm& Comm);

    Map(int NumGlobalElements, int IndexBase, const MPI_Comm& Comm);

    Map(int NumGlobalElements, int NumMyElements, int IndexBase, const Epetra_Comm& Comm);

    Map(int NumGlobalElements, int NumMyElements, int IndexBase, const MPI_Comm& Comm);

    Map(int NumGlobalElements, int NumMyElements, const int* MyGlobalElements, int IndexBase,
        const Epetra_Comm& Comm);

    Map(int NumGlobalElements, int NumMyElements, const int* MyGlobalElements, int IndexBase,
        const MPI_Comm& Comm);

    Map(const Map& Source);

    /// Copy constructor from Epetra_Map
    explicit Map(const Epetra_Map& Source);

    /// Copy constructor from Epetra_Map
    explicit Map(const Epetra_BlockMap& Source);

    ~Map() = default;


    Map& operator=(const Map& other);
    void Print(std::ostream& os) const { map_->Print(os); }

    //! return the reference to the Epetra_Map
    const Epetra_Map& get_epetra_map() const { return *map_; }
    Epetra_Map& get_epetra_map() { return *map_; }

    //! Returns true if this and Map are identical maps
    bool SameAs(const Map& other) const { return map_->SameAs(*(other.map_)); }

    //! Returns true if this and Map have identical point-wise structure
    bool PointSameAs(const Map& Map) const { return map_->PointSameAs(Map.get_epetra_map()); }

    //! Number of elements across all processors.
    int NumGlobalElements() const { return map_->NumGlobalElements(); }

    //! Number of elements on the calling processor.
    int NumMyElements() const { return map_->NumMyElements(); }

    //! Index base for this map.
    int IndexBase() const { return map_->IndexBase(); }

    //! Returns true if map is defined across more than one processor.
    bool DistributedGlobal() const { return map_->DistributedGlobal(); }

    //! Returns true if this and Map are identical maps
    bool SameAs(const Epetra_Map& other) const { return map_->SameAs(other); }
    bool SameAs(const Epetra_BlockMap& other) const { return map_->SameAs(other); }

    //! Returns true if the GID passed in belongs to the calling processor in this map,
    //! otherwise returns false.
    bool MyGID(int GID_in) const { return map_->MyGID(GID_in); }

    //! Returns global ID of local ID, return IndexBase-1 if not found on this processor.
    int GID(int LID) const { return map_->GID(LID); }

    //! Returns the size of elements in the map; only valid if map has constant element size.
    int ElementSize() const { return map_->ElementSize(); }

    //! Size of element for specified LID.
    int ElementSize(int LID) const { return map_->ElementSize(LID); }

    //! Returns the maximum global ID across the entire map
    int MaxAllGID() const { return map_->MaxAllGID(); }

    //! Returns the minimum global ID across the entire map.
    int MinAllGID() const { return map_->MinAllGID(); }

    //! Returns local ID of global ID, return -1 if not found on this processor.
    int LID(int GID) const { return map_->LID(GID); }

    //! Returns true if this and Map have identical point-wise structure
    bool PointSameAs(const Epetra_Map& Map) const { return map_->PointSameAs(Map); }
    bool PointSameAs(const Epetra_BlockMap& Map) const { return map_->PointSameAs(Map); }

    //! Returns the processor IDs and corresponding local index value for a given list of global
    //! indices.
    int RemoteIDList(int NumIDs, int* GIDList, int* PIDList, int* LIDList) const
    {
      return map_->RemoteIDList(NumIDs, GIDList, PIDList, LIDList);
    }

    int RemoteIDList(
        int NumIDs, const int* GIDList, int* PIDList, int* LIDList, int* SizeList) const
    {
      return map_->RemoteIDList(NumIDs, GIDList, PIDList, LIDList, SizeList);
    }
    int MinMyGID(void) const { return map_->MinMyGID(); }

    //! Access function for Epetra_Comm communicator.
    const Epetra_Comm& Comm() const { return map_->Comm(); }

    //! Returns true if map GIDs are 1-to-1.
    bool UniqueGIDs(void) const { return map_->UniqueGIDs(); }

    //! Pointer to internal array containing list of global IDs assigned to the calling processor.
    int* MyGlobalElements(void) const { return map_->MyGlobalElements(); }

    //! Maximum element size across all processors.
    int MaxElementSize(void) const { return map_->MaxElementSize(); }

    //! Puts list of global elements on this processor into the user-provided array.
    int MyGlobalElements(int* MyGlobalElementList) const
    {
      return map_->MyGlobalElements(MyGlobalElementList);
    }

    //! Number of local points for this map; equals the sum of all element sizes on the calling
    //! processor.
    int NumMyPoints() const { return map_->NumMyPoints(); }

    //! Returns a pointer to the BlockMapData instance this BlockMap uses.
    const Epetra_BlockMapData* DataPtr() const { return map_->DataPtr(); }


    /**
     * View a given Epetra_Map object under our own Vector wrapper.
     */
    [[nodiscard]] static std::shared_ptr<Map> create_view(Epetra_Map& view)
    {
      std::shared_ptr<Map> ret(new Map);
      ret->map_ = std::make_shared<Epetra_Map>(view);
      return ret;
    }

    [[nodiscard]] static std::shared_ptr<const Map> create_view(const Epetra_Map& view)
    {
      std::shared_ptr<Map> ret(new Map);
      ret->map_ = std::make_shared<Epetra_Map>(view);
      return ret;
    }

    /**
     * View the Epetra_BlockMap object under our own Vector wrapper.
     */

    [[nodiscard]] static std::shared_ptr<const Map> create_view(Epetra_BlockMap& view)
    {
      std::shared_ptr<Map> ret(new Map);
      ret->map_ = std::make_shared<Epetra_Map>(view.NumGlobalElements(), view.NumMyElements(),
          view.MyGlobalElements(), view.IndexBase(), view.Comm());
      return ret;
    }

    [[nodiscard]] static std::shared_ptr<const Map> create_view(const Epetra_BlockMap& view)
    {
      std::shared_ptr<Map> ret(new Map);
      ret->map_ = std::make_shared<Epetra_Map>(view.NumGlobalElements(), view.NumMyElements(),
          view.MyGlobalElements(), view.IndexBase(), view.Comm());
      return ret;
    }

   private:
    Map() = default;
    //! The actual Epetra_Map object.
    std::shared_ptr<Epetra_Map> map_ = nullptr;
  };

  inline std::ostream& operator<<(std::ostream& os, const Map& m)
  {
    os << m.get_epetra_map();
    return os;
  }

  template <>
  struct WrapperFor<Epetra_Map>
  {
    using type = Map;
  };

  template <>
  struct WrapperFor<Epetra_BlockMap>
  {
    using type = Map;
  };
}  // namespace Core::LinAlg


// NOLINTEND(readability-identifier-naming)

FOUR_C_NAMESPACE_CLOSE


#endif
