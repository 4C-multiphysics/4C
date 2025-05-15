// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_XFEM_DISCRETIZATION_HPP
#define FOUR_C_XFEM_DISCRETIZATION_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization_faces.hpp"
#include "4C_fem_dofset.hpp"
#include "4C_fem_dofset_interface.hpp"
#include "4C_fem_dofset_proxy.hpp"

FOUR_C_NAMESPACE_OPEN

namespace XFEM
{
  class XFEMDofSet;

  /*!
  \brief A class to manage a discretization in parallel with changing dofs

  */
  class DiscretizationXFEM : public Core::FE::DiscretizationFaces
  {
   public:
    /*!
    \brief Standard Constructor

    \param name (in): name of this discretization
    \param comm (in): An comm object associated with this discretization
    */
    DiscretizationXFEM(const std::string name, MPI_Comm comm, unsigned int n_dim);

    /*!
    \brief Complete construction of a discretization  (Filled()==true NOT prerequisite)

    This call is done at the initial state of the discretisation, therefore the initial dofset
    is stored!

    After adding or deleting nodes or elements or redistributing them in parallel,
    or adding/deleting boundary conditions, this method has to be called to (re)construct
    pointer topologies.<br>
    It builds in this order:<br>
    - row map of nodes
    - column map of nodes
    - row map of elements
    - column map of elements
    - pointers from elements to nodes
    - pointers from nodes to elements
    - assigns degrees of freedoms
    - map of element register classes
    - calls all element register initialize methods
    - build geometries of all Dirichlet and Neumann boundary conditions

    \param nds (in) :  vector of dofset numbers to be initialized as initialdofset

    \param assigndegreesoffreedom (in) : if true, resets existing dofsets and performs
                                         assigning of degrees of freedoms to nodes and
                                         elements.
    \param initelements (in) : if true, build element register classes and call initialize()
                               on each type of finite element present
    \param doboundaryconditions (in) : if true, build geometry of boundary conditions
                                       present.

    \note In order to receive a fully functional discretization, this method must be called
          with all parameters set to true (at least once). The parameters though can be
          used to turn off specific tasks to allow for more flexibility in the
          construction of a discretization, where it is known that this method will
          be called more than once.

    \note Sets Filled()=true
    */
    int initial_fill_complete(const std::vector<int>& nds, bool assigndegreesoffreedom = true,
        bool initelements = true, bool doboundaryconditions = true);

    /// Export Vector with initialdofrowmap (all nodes have one dofset) - to Vector with all active
    /// dofs
    void export_initialto_active_vector(
        const Core::LinAlg::Vector<double>& initialvec, Core::LinAlg::Vector<double>& activevec);

    void export_activeto_initial_vector(
        const Core::LinAlg::Vector<double>& activevec, Core::LinAlg::Vector<double>& initialvec);



    /*!
    \brief Get the gid of all initial dofs of a node.

    Ask the initial DofSet for the gids of the dofs of this node. The
    required vector is created and filled on the fly. So better keep it
    if you need more than one dof gid.
    - HaveDofs()==true prerequisite (produced by call to assign_degrees_of_freedom()))
    \param nds (in)       : number of dofset
    \param node (in)      : the node
    */
    std::vector<int> initial_dof(unsigned nds, const Core::Nodes::Node* node) const
    {
      FOUR_C_ASSERT(nds < initialdofsets_.size(), "undefined dof set");
      FOUR_C_ASSERT(initialized_, "no initial dofs assigned");
      return initialdofsets_[nds]->dof(node);
    }

    /*!
    \brief Get the gid of all initial dofs of a node.

    Ask the initial DofSet for the gids of the dofs of this node. The
    required vector is created and filled on the fly. So better keep it
    if you need more than one dof gid.
    - HaveDofs()==true prerequisite (produced by call to assign_degrees_of_freedom()))
    \param node (in)      : the node
    */
    std::vector<int> initial_dof(const Core::Nodes::Node* node) const
    {
      FOUR_C_ASSERT(initialdofsets_.size() == 1, "expect just one dof set");
      FOUR_C_ASSERT(initialized_, "no initial dofs assigned");
      return initial_dof(0, node);
    }

    /*!
    \brief Get the gid of all initial dofs of a node.

    Ask the initial DofSet for the gids of the dofs of this node. The
    required vector is created and filled on the fly. So better keep it
    if you need more than one dof gid.
    - HaveDofs()==true prerequisite (produced by call to assign_degrees_of_freedom()))
    \param nds (in)       : number of dofset
    \param node (in)      : the node
    \param lm (in/out)    : lm vector the dofs are appended to
    */

    void initial_dof(unsigned nds, const Core::Nodes::Node* node, std::vector<int>& lm) const
    {
      FOUR_C_ASSERT(nds < initialdofsets_.size(), "undefined dof set");
      FOUR_C_ASSERT(initialized_, "no initial dofs assigned");
      initialdofsets_[nds]->dof(node, lm);
    }

    /*!
    \brief Get the gid of all initial dofs of a node.

    Ask the initial DofSet for the gids of the dofs of this node. The
    required vector is created and filled on the fly. So better keep it
    if you need more than one dof gid.
    - HaveDofs()==true prerequisite (produced by call to assign_degrees_of_freedom()))
    \param node (in)      : the node
    \param lm (in/out)    : lm vector the dofs are appended to
    */
    void initial_dof(const Core::Nodes::Node* node, std::vector<int>& lm) const
    {
      FOUR_C_ASSERT(initialdofsets_.size() == 1, "expect just one dof set");
      FOUR_C_ASSERT(initialized_, "no initial dofs assigned");
      initial_dof((unsigned)0, node, lm);
    }


    /// Access to initial dofset
    const Core::DOFSets::DofSetInterface& initial_dof_set(unsigned int nds = 0) const
    {
      initialized();
      return *initialdofsets_[nds];
    }


    std::shared_ptr<Core::DOFSets::DofSetInterface> get_initial_dof_set_proxy(int nds)
    {
      FOUR_C_ASSERT(nds < (int)initialdofsets_.size(), "undefined dof set");
      return std::make_shared<Core::DOFSets::DofSetProxy>(&*initialdofsets_[nds]);
    }

    /*!
    \brief Get initial degree of freedom row map (Initialized()==true prerequisite)

    Return ptr to initial degree of freedom row distribution map of this discretization.
    If it does not exist yet, build it.

    - Initialized()==true prerequisite

    */
    const Core::LinAlg::Map* initial_dof_row_map(unsigned nds = 0) const;

    /*!
    \brief Get initial degree of freedom column map (Initialized()==true prerequisite)

    Return ptr to initial degree of freedom column distribution map of this discretization.
    If it does not exist yet, build it.

    - Initialized()==true prerequisite

    */
    const Core::LinAlg::Map* initial_dof_col_map(unsigned nds = 0) const;

    /// checks if discretization is initialized
    bool initialized() const;


    /*!
    \brief Set a reference to a data vector

    Using this method, a reference to a vector can
    be supplied to the discretization. The elements can access
    this vector by using the name of that vector.
    The method expects state to be either of dof row map or of
    dof column map.
    If the vector is supplied in DofColMap() a reference to it will be stored.
    If the vector is NOT supplied in DofColMap(), but in dof_row_map(),
     a vector with column map is allocated and the supplied vector is exported to it.
    Everything is stored/referenced using std::shared_ptr.

    \param nds (in): number of dofset
    \param name (in): Name of data
    \param state (in): vector of some data

    \note This class will not take ownership or in any way modify the solution vector.
    */
    void set_initial_state(unsigned nds, const std::string& name,
        std::shared_ptr<const Core::LinAlg::Vector<double>> state);

    /** \brief Get number of standard (w/o enrichment) dofs for given node.
     *
     *  For the XFEM discretization the number of elements of the first
     *  nodal dof set is returned.
     *
     *  \param nds  (in) : number of dofset
     *  \param node (in) : the node those number of dofs are requested
     *
     *  */
    int num_standard_dof(const unsigned& nds, const Core::Nodes::Node* node) const override
    {
      std::vector<int> dofs;
      // get the first dofs of the node (not enriched)
      dof(dofs, node, nds, 0, nullptr);
      return static_cast<int>(dofs.size());
    }

    bool is_equal_x_dof_set(int nds, const XFEM::XFEMDofSet& xdofset_new) const;

   private:
    /// Store Initial Dofs
    void store_initial_dofs(const std::vector<int>& nds);

    /*!
    ///Extend initialdofrowmap
    \param srcmap (in) : Sourcemap used as base
    \param numdofspernode (in) : Number of degrees of freedom per node
    \param numdofsets (in) : Number of XFEM-Dofsets per node
    \param uniquenumbering (in) : Assign unique number to additional dofsets

    */
    std::shared_ptr<Core::LinAlg::Map> extend_map(const Core::LinAlg::Map* srcmap,
        int numdofspernodedofset, int numdofsets, bool uniquenumbering);

    /// initial set of dofsets
    std::vector<std::shared_ptr<Core::DOFSets::DofSetInterface>> initialdofsets_;

    /// bool if discretisation is initialized
    bool initialized_;

    /// full (with all reserved dofs) dof row map of initial state
    std::shared_ptr<Core::LinAlg::Map> initialfulldofrowmap_;

    /// permuted (with duplicated gids of first dofset - to all other dofsets) dof row map of
    /// initial state
    std::shared_ptr<Core::LinAlg::Map> initialpermdofrowmap_;


  };  // class DiscretizationXFEM
}  // namespace XFEM

FOUR_C_NAMESPACE_CLOSE

#endif
