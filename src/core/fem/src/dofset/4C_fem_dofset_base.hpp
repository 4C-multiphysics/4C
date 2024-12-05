// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_DOFSET_BASE_HPP
#define FOUR_C_FEM_DOFSET_BASE_HPP

#include "4C_config.hpp"

#include "4C_fem_dofset_interface.hpp"

#include <list>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::DOFSets
{

  /*! \brief Base class set of degrees of freedom

    This base class manages the static list, all DofSets are
    written into and the list of DofSets the current DofSet is connected to.

    \author tk    */
  class DofSetBase : public DofSetInterface
  {
   public:
    /// Standard Constructor
    DofSetBase();

    /// Destructor
    ~DofSetBase() override;


    //! @name Utility Methods

    /// Print this class
    void print(std::ostream& os) const override;

    /// Get Max of all GID assigned in the DofSets in front of current one in the list
    /// #static_dofsets_
    int max_gi_din_list(MPI_Comm comm) const override;

    //@}

    //! @name Construction

    /// Add Dof Set to list #static_dofsets_
    void add_dof_setto_list() override;

    /// Replace a Dof Set in list #static_dofsets_ with this
    void replace_in_static_dofsets(std::shared_ptr<DofSetInterface> olddofset) override;

    //@}

    /// Print the dofsets in the static_dofsets_ list
    void print_all_dofsets(MPI_Comm comm) const override;


    //! @name DofSet management
    /// Registered DofSets need to know about changes to the DofSet.

    /// Notify proxies of new dofs
    void notify_assigned() override;

    /// Notify proxies of reset
    void notify_reset() override;

    /// Register new dofset to notify
    void register_proxy(DofSetInterface* dofset) override;

    /// Remove dofset from list
    void unregister(DofSetInterface* dofset) override;

    //@}

   private:
    /*! \brief list of registered dof sets

      Whenever you request a proxy of any specific DofSet implementation, this proxy
      has to be registered in the list registered_dofsets_ . See also \ref DofSetProxy.
      Also other, special implementations of a DofSet, which are linked to another
      DofSet in some way may register here, see e.g. \ref DofSetDefinedMappingWrapper and
      \ref DofSetGIDBasedWrapper. This way the registered DofSet will be notified
      of any state changes of the original DofSet by calling \ref NotifyAssigned and \ref
      NotifyReset .*/
    std::list<DofSetInterface*> registered_dofsets_;

    /*! \brief store dofset in static list, if derived class chooses so using AddDofSettoList()

        This is hack to get unique dof numbers on all dof sets. With this in place we
        can combine the maps from any dof sets to form block systems and the like.    */
    static std::list<DofSetInterface*> static_dofsets_;


  };  // class DofSetBase
}  // namespace Core::DOFSets

FOUR_C_NAMESPACE_CLOSE

#endif
