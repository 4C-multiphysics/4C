/*---------------------------------------------------------------------*/
/*! \file

\brief A modified set of degrees of freedom for periodic boundary
       conditions. This class is inherited from dofset and replaces the
       method assign_degrees_of_freedom by a version which uses a map
       of coupled nodes provided by the periodic boundary conditions to
       assign the same degrees of freedom to coupled pairs of master and
       slavenodes. It is absolutely mandatory that for each slave node
       on this proc the master is (ghosted) on this proc, too!

\level 0


*/
/*---------------------------------------------------------------------*/

#ifndef FOUR_C_FEM_DOFSET_PBC_HPP
#define FOUR_C_FEM_DOFSET_PBC_HPP

#include "4C_config.hpp"

#include "4C_fem_dofset.hpp"

#include <Epetra_Comm.h>
#include <Epetra_Map.h>
#include <Teuchos_RCP.hpp>

#include <map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::DOFSets
{


  /*!
  \brief A set of degrees of freedom

  \author gammi
  */
  class PBCDofSet : virtual public DofSet
  {
   public:
    /*!
    \brief Standard Constructor



    create a dofset that allows coupled nodes for periodic boundary
    conditions                                         gammi 05/07


    \param couplednodes (i) list of coupled nodes


    \return void

    */
    PBCDofSet(Teuchos::RCP<std::map<int, std::vector<int>>> couplednodes);


    /// Get maximum GID of degree of freedom row map
    int max_all_gid() const override;

    /// Get minimum GID of degree of freedom row map
    int min_all_gid() const override;

    /// create a copy of this object
    Teuchos::RCP<DofSet> clone() override { return Teuchos::rcp(new PBCDofSet(*this)); }

    /// Assign dof numbers using all elements and nodes of the discretization.
    int assign_degrees_of_freedom(
        const Core::FE::Discretization& dis, const unsigned dspos, const int start) override;

    /// Update the coupled nodes map of dofset
    virtual void set_coupled_nodes(Teuchos::RCP<std::map<int, std::vector<int>>> couplednodes);

    /// Get coupled nodes map (corresponding col format)
    std::map<int, std::vector<int>>* get_coupled_nodes() { return perbndcouples_.get(); }

    /// Get connectivity map between slave node and its master node
    virtual Teuchos::RCP<std::map<int, int>> get_slave_to_master_node_connectivity()
    {
      return perbnd_slavetomaster_;
    };

   protected:
    /// get number of nodal dofs for this element at this node
    int num_dof_per_node(const Core::Nodes::Node& node) const override
    {
      if (slavenodeids_->count(node.id()) == 0)
      {
        return DofSet::num_dof_per_node(node);
      }
      return 0;
    }

    //!\brief master and slave node connectivity for periodic boundary conditions
    Teuchos::RCP<std::map<int, std::vector<int>>> perbndcouples_;

    //!\brief the largest original GID, to stop the dofset from 'shrinking'
    int myMaxGID_;

    //!\brief the smallest original GID, to stop the dofset from 'shrinking'
    int myMinGID_;

   private:
    /// Build the connectivity between slave node and its master node
    void build_slave_to_master_node_connectivity();

    Teuchos::RCP<std::set<int>> slavenodeids_;

    //!\brief slave node to master node connectivity for periodic boundary conditions (key=slave
    //! nid, value=master nid)
    Teuchos::RCP<std::map<int, int>> perbnd_slavetomaster_;

  };  // class PBCDofSet

}  // namespace Core::DOFSets

FOUR_C_NAMESPACE_CLOSE

#endif
