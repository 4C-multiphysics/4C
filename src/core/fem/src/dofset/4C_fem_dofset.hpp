// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_DOFSET_HPP
#define FOUR_C_FEM_DOFSET_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset_base.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_map.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>
#include <ranges>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE
/*!
\brief A set of degrees of freedom

\note This is an internal class of the discretization module that one
should not need to touch on an ordinary day. It is here to support the
discretization class. Everyday use should happen via the
discretization class only.

<h3>Purpose</h3>

This class represents one set of degrees of freedom for the
discretization class in the usual parallel fashion. That is there is a
dof_row_map() and a DofColMap() that return the maps of the global FE
system of equation in row and column setting respectively. These maps
are used by the algorithm's Core::LinAlg::Vector<double> classes among others.

There can be dofs in nodes, faces, and elements. And additionally to the
above maps this class needs to know the global dof ids of all nodes
and elements. In order to provide this information nodal and elemental
column vectors are provided that store the number of dofs and the
local column map id of the first dof for each node and element. Since
dof numbers to one object are always consecutive this is all that's
needed. So the methods NumDof() and Dof() can be provided for nodes,
faces, and elements.

One has to keep in mind, however, that the lookup of one dof gid
involves some table lookups. Therefore there is a special version of
Dof() that gathers and returns all dof gids to one node or element at
once. This is to be preferred if more that one lookup is needed.

The point of holding these maps and vectors in a class of its own is
to enable multiple sets of dofs on the same mesh. But judging from
past experience this feature will not be used that often. So effort
has been made to hide the possibility of multiple DofSets.

The setup is done by assign_degrees_of_freedom(). This method uses two
redundant nodal and elemental vectors. It would be hard to avoid
those. Lets hope we can always afford them.

\note It is guaranteed that the same mesh (nodes and elements are all
the same with the same global ids) is assigned the same set of dofs
all the time independent of its parallel distribution. That's crucial
to be able to redistribute a mesh without losing the old vectors.

<h3>Invariants</h3>

There are two possible states in this class: Reset and setup. To
change back and forth use assign_degrees_of_freedom() and reset().

<h3>Dof number uniqueness</h3>

Each DofSet assigns unique dof numbers that do not occur in any other
DofSet. This is true as long as the number of dofs per DofSet does not
change. To achieve this we keep a list of dof sets internally.

<h3>Copying behaviour</h3>

Please note that even though Michael does not like it this class
contains neither copy constructor nor assignment operator. This is
intended. It is legal to copy this objects of class. The internal
variables (all std::shared_ptrs) know how to copy themselves. So the
default versions will do just fine. (Far better than buggy hand
written versions.) And due to the two possible states there is no
reason to deep copy any of the local map and vector variables.
*/

namespace Core::DOFSets
{
  class DofSet : public DofSetBase
  {
   public:
    /// Standard Constructor
    DofSet();


    /// create a copy of this object
    virtual std::shared_ptr<DofSet> clone() { return std::make_shared<DofSet>(*this); }

    //! @name Access methods

    /// Get number of dofs for given node
    int num_dof(const Core::Nodes::Node* node) const override
    {
      int lid = node->lid();
      if (lid == -1) return 0;
      return (numdfcolnodes_->get_local_values())[lid];
    }

    /// Get number of dofs for given element
    int num_dof(const Core::Elements::Element* element) const override
    {
      // check if this is a face element
      int lid = element->lid();
      if (lid == -1) return 0;
      if (element->is_face_element())
        return (numdfcolfaces_ != nullptr) ? (numdfcolfaces_->get_local_values())[lid] : 0;
      else
        return (numdfcolelements_->get_local_values())[lid];
    }

    /// Get the gid of a dof for given node
    int dof(const Core::Nodes::Node* node, int dof) const override
    {
      int lid = node->lid();
      if (lid == -1) return -1;
      if (pccdofhandling_)
        return dofscolnodes_->gid((shiftcolnodes_->get_local_values())[lid] + dof);
      else
        return (idxcolnodes_->get_local_values())[lid] + dof;
    }

    /// Get the gid of a dof for given element
    int dof(const Core::Elements::Element* element, int dof) const override
    {
      int lid = element->lid();
      if (lid == -1) return -1;
      if (element->is_face_element())
        return (idxcolfaces_ != nullptr) ? (idxcolfaces_->get_local_values())[lid] + dof : -1;
      else
        return (idxcolelements_->get_local_values())[lid] + dof;
    }

    /// Get the gid of all dofs of a node
    std::vector<int> dof(const Core::Nodes::Node* node) const override
    {
      const int lid = node->lid();
      if (lid == -1) return std::vector<int>();
      const int idx = (idxcolnodes_->get_local_values())[lid];
      std::vector<int> dof((numdfcolnodes_->get_local_values())[lid]);
      for (unsigned i = 0; i < dof.size(); ++i)
      {
        if (pccdofhandling_)
          dof[i] = dofscolnodes_->gid((shiftcolnodes_->get_local_values())[lid] + i);
        else
          dof[i] = idx + i;
      }
      return dof;
    }

    /// Get the gid of all dofs of a node
    void dof(std::vector<int>& global_dof_index,  ///< vector of dof gids (to be filled)
        const Core::Nodes::Node* node,            ///< the node
        unsigned nodaldofset  ///< number of nodal dof set of the node (currently !=0 only for XFEM)
    ) const override
    {
      FOUR_C_ASSERT(nodaldofset == 0, "only one nodal dofset supported!");
      global_dof_index = dof(node);
    }

    /// Get the gid of all dofs of a element
    std::vector<int> dof(const Core::Elements::Element* element) const override
    {
      int lid = element->lid();
      if (lid == -1) return std::vector<int>();

      if (element->is_face_element() && idxcolfaces_ == nullptr) return std::vector<int>();

      int idx = element->is_face_element() ? (idxcolfaces_->get_local_values())[lid]
                                           : (idxcolelements_->get_local_values())[lid];
      std::vector<int> dof(element->is_face_element()
                               ? (numdfcolfaces_->get_local_values())[lid]
                               : (numdfcolelements_->get_local_values())[lid]);
      for (unsigned i = 0; i < dof.size(); ++i) dof[i] = idx + i;
      return dof;
    }

    /// Get the gid of all dofs of a node
    void dof(const Core::Nodes::Node* node, std::vector<int>& lm) const override
    {
      int lid = node->lid();
      if (lid == -1) return;
      int idx = (idxcolnodes_->get_local_values())[lid];
      int size = (numdfcolnodes_->get_local_values())[lid];
      for (int i = 0; i < size; ++i)
      {
        if (pccdofhandling_)
          lm.push_back(dofscolnodes_->gid((shiftcolnodes_->get_local_values())[lid] + i));
        else
          lm.push_back(idx + i);
      }
    }

    /// Get the gid of all dofs of a node
    void dof(const Core::Nodes::Node* node,  ///< node, for which you want the dof positions
        const unsigned startindex,  ///< first index of vector at which will be written to end
        std::vector<int>& lm        ///< already allocated vector to be filled with dof positions
    ) const override
    {
      const int lid = node->lid();
      if (lid == -1) return;
      const int idx = (idxcolnodes_->get_local_values())[lid];
      const int size = (numdfcolnodes_->get_local_values())[lid];
      FOUR_C_ASSERT(lm.size() >= (startindex + size), "vector<int> lm too small");
      for (int i = 0; i < size; ++i)
      {
        if (pccdofhandling_)
          lm[startindex + i] = dofscolnodes_->gid((shiftcolnodes_->get_local_values())[lid] + i);
        else
          lm[startindex + i] = idx + i;
      }
    }

    /// Get the gid of all dofs of a element
    void dof(const Core::Elements::Element* element, std::vector<int>& lm) const override
    {
      int lid = element->lid();
      if (lid == -1) return;

      if (element->is_face_element() && idxcolfaces_ == nullptr) return;

      int idx = element->is_face_element() ? (idxcolfaces_->get_local_values())[lid]
                                           : (idxcolelements_->get_local_values())[lid];
      int size = element->is_face_element() ? (numdfcolfaces_->get_local_values())[lid]
                                            : (numdfcolelements_->get_local_values())[lid];
      for (int i = 0; i < size; ++i) lm.push_back(idx + i);
    }

    /// Get the GIDs of the first DOFs of a node of which the associated element is interested in
    void dof(const Core::Elements::Element*
                 element,  ///< element which provides its expected number of DOFs per node
        const Core::Nodes::Node* node,  ///< node, for which you want the DOF positions
        std::vector<int>& lm  ///< already allocated vector to be filled with DOF positions
    ) const override
    {
      const int lid = node->lid();
      if (lid == -1) return;
      const int idx = (idxcolnodes_->get_local_values())[lid];
      // this method is used to setup the vector of number of dofs, so we cannot ask
      // numdfcolelements_ here as above. Instead we have to ask the node itself
      const int size = num_dof_per_node(*element, *node);
      for (int i = 0; i < size; ++i)
      {
        if (pccdofhandling_)
          lm.push_back(dofscolnodes_->gid((shiftcolnodes_->get_local_values())[lid] + i));
        else
          lm.push_back(idx + i);
      }
    }

    /// are the dof maps already initialized?
    bool initialized() const override;

    /// Get degree of freedom row map
    const Core::LinAlg::Map* dof_row_map() const override;

    /// Get degree of freedom column map
    const Core::LinAlg::Map* dof_col_map() const override;

    //! Print this class
    void print(std::ostream& os) const override;

    //! Return true if \ref assign_degrees_of_freedom was called
    bool filled() const override { return filled_; }

    //@}

    //! @name Construction

    /*!
    \brief Assign dof numbers using all elements and nodes of the discretization.

    @param[in] dis discretization of a mortar interface
    @param[in] dspos Position of DOfSet inside its discretization
    @param[in] start User-defined offset for DOF numbering [currently not supported]

    @return Maximum dof number of this dofset
    */
    int assign_degrees_of_freedom(
        const Core::FE::Discretization& dis, const unsigned dspos, const int start) override;

    /// reset all internal variables
    void reset() override;

    //@}

    //! @name Proxy management
    /// Proxies need to know about changes to the DofSet.

    /// our original DofSet dies
    void disconnect(DofSetInterface* dofset) override { return; };

    //@}

    /// Get Number of Global Elements of degree of freedom row map
    int num_global_elements() const override;

    /// Get maximum GID of degree of freedom row map
    int max_all_gid() const override;

    /// Get minimum GID of degree of freedom row map
    int min_all_gid() const override;


   protected:
    /// get number of nodal dofs
    int num_dof_per_node(const Core::Nodes::Node& node) const override
    {
      auto adjacent_elements = node.adjacent_elements();
      if (adjacent_elements.empty()) return 0;

      return std::ranges::max(
          adjacent_elements | std::views::transform([&](auto ele)
                                  { return num_dof_per_node(*ele.user_element(), node); }));
    }

    /// get number of nodal dofs for this element at this node
    [[nodiscard]] virtual int num_dof_per_node(
        const Core::Elements::Element& element, const Core::Nodes::Node& node) const
    {
      return element.num_dof_per_node(node);
    }

    /// get number of element dofs for this element
    virtual int num_dof_per_element(const Core::Elements::Element& element) const
    {
      return element.num_dof_per_element();
    }

    virtual int num_dof_per_face(const Core::Elements::Element& element, int face) const
    {
      return element.num_dof_per_face(face);
    }

    /// Get Reserved Max Number Dofs per Node
    virtual void get_reserved_max_num_dofper_node(int& maxnodenumdf) { return; };

    /// get first number to be used as Dof GID in assign_degrees_of_freedom
    virtual int get_first_gid_number_to_be_used(const Core::FE::Discretization& dis) const;

    /// get minimal node GID to be used in assign_degrees_of_freedom
    virtual int get_minimal_node_gid_if_relevant(const Core::FE::Discretization& dis) const;

    /// filled flag
    bool filled_;

    /// position of dofset inside its discretization
    unsigned dspos_;

    /// unique row map of degrees of freedom (node, face, and element dofs))
    std::shared_ptr<Core::LinAlg::Map> dofrowmap_;

    /// unique column map of degrees of freedom (node, face, and element dofs)
    std::shared_ptr<Core::LinAlg::Map> dofcolmap_;

    /// number of dofs for each node
    std::shared_ptr<Core::LinAlg::Vector<int>> numdfcolnodes_;

    /// number of dofs for each face
    std::shared_ptr<Core::LinAlg::Vector<int>> numdfcolfaces_;

    /// number of dofs for each element
    std::shared_ptr<Core::LinAlg::Vector<int>> numdfcolelements_;

    /// column map gid of first dof for each node
    std::shared_ptr<Core::LinAlg::Vector<int>> idxcolnodes_;

    /// column map gid of first dof for each face
    std::shared_ptr<Core::LinAlg::Vector<int>> idxcolfaces_;

    /// column map gid of first dof for each element
    std::shared_ptr<Core::LinAlg::Vector<int>> idxcolelements_;

    /*!
    \brief Activate special dof handling due to point coupling conditions?

    \note In the logic outlined above and implemented in assign_degrees_of_freedom() we
    assume that each nodal dof only appears once and therefore that we can
    define a unique access to all dofs of one node by knowing (1) its number of
    nodal dofs and (2) its first dof gid. The second, third,... dof gid are
    then simply defined by incrementation +1 starting from the first dof gid.
    However, in the case where we have point coupling conditions, e.g. due to
    joints in structural mechanics (where certain dofs of two nodes at the same
    geometric position are to be coupled and certain others not), this basic
    assumption does not hold anymore. Instead, we want to assign the same(!)
    dof to both nodes in this case, i.e. the same dof is then shared by more
    than one node and the numbering logic mentioned above is not sufficient
    anymore. For such cases in particular (but maybe also in general?), we
    should store not only the first dof for each node, but all(!) dofs for
    each node. The #idxcolnodes_ vector is then replaced by the
    two following data containers:
       #dofscolnodes_ (Core::LinAlg::Map) and #shiftcolnodes_ (Core::LinAlg::Vector<int>)

    Nevertheless, for the time being, the old version with #idxcolnodes_ stays
    around, since the new version is only implemented for the standard DofSet
    class declared here (but not for special TransparentDofSets, Proxy, etc.).
    */
    bool pccdofhandling_;

    /// column map of all dofs for each node (possibly non-unique)
    std::shared_ptr<Core::LinAlg::Map> dofscolnodes_;

    /// shift value for access to column map of all dofs for each node
    std::shared_ptr<Core::LinAlg::Vector<int>> shiftcolnodes_;
    //***************************************************************************

  };  // class DofSet

  /**
   * An exception thrown when an error in nodal distribution occurs.
   */
  class NodalDistributionException : public Core::Exception
  {
   public:
    explicit NodalDistributionException(const std::string& message);
  };
}  // namespace Core::DOFSets

// << operator
std::ostream& operator<<(std::ostream& os, const Core::DOFSets::DofSet& dofset);


FOUR_C_NAMESPACE_CLOSE

#endif
