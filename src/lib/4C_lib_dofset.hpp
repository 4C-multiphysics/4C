/*---------------------------------------------------------------------*/
/*! \file

\brief A set of degrees of freedom

\level 0


*/
/*---------------------------------------------------------------------*/

#ifndef FOUR_C_LIB_DOFSET_HPP
#define FOUR_C_LIB_DOFSET_HPP

#include "4C_config.hpp"

#include "4C_lib_dofset_base.hpp"
#include "4C_lib_element.hpp"
#include "4C_lib_node.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_Comm.h>
#include <Epetra_IntVector.h>
#include <Epetra_Map.h>
#include <Teuchos_RCP.hpp>

#include <list>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace DRT
{
  class Discretization;

  /*!
  \brief A set of degrees of freedom

  \note This is an internal class of the discretization module that one
  should not need to touch on an ordinary day. It is here to support the
  Discretization class. Everyday use should happen via the
  Discretization class only.

  <h3>Purpose</h3>

  This class represents one set of degrees of freedom for the
  Discretization class in the usual parallel fashion. That is there is a
  DofRowMap() and a DofColMap() that return the maps of the global FE
  system of equation in row and column setting respectively. These maps
  are used by the algorithm's Epetra_Vector classes among others.

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

  The setup is done by AssignDegreesOfFreedom(). This method uses two
  redundant nodal and elemental vectors. It would be hard to avoid
  those. Lets hope we can always afford them.

  \note It is guaranteed that the same mesh (nodes and elements are all
  the same with the same global ids) is assigned the same set of dofs
  all the time independent of its parallel distribution. That's crucial
  to be able to redistribute a mesh without losing the old vectors.

  <h3>Invariants</h3>

  There are two possible states in this class: Reset and setup. To
  change back and forth use AssignDegreesOfFreedom() and Reset().

  <h3>Dof number uniqueness</h3>

  Each DofSet assigns unique dof numbers that do not occur in any other
  DofSet. This is true as long as the number of dofs per DofSet does not
  change. To achieve this we keep a list of dof sets internally.

  <h3>Copying behaviour</h3>

  Please note that even though Michael does not like it this class
  contains neither copy constructor nor assignment operator. This is
  intended. It is legal to copy this objects of class. The internal
  variables (all Teuchos::RCPs) know how to copy themselves. So the
  default versions will do just fine. (Far better than buggy hand
  written versions.) And due to the two possible states there is no
  reason to deep copy any of the local map and vector variables.

  \author u.kue
  */
  class DofSet : public DofSetBase
  {
   public:
    /// Standard Constructor
    DofSet();


    /// create a copy of this object
    virtual Teuchos::RCP<DofSet> Clone() { return Teuchos::rcp(new DofSet(*this)); }

    //! @name Access methods

    /// Get number of dofs for given node
    int NumDof(const Node* node) const override
    {
      int lid = node->LID();
      if (lid == -1) return 0;
      return (*numdfcolnodes_)[lid];
    }

    /// Get number of dofs for given element
    int NumDof(const Element* element) const override
    {
      // check if this is a face element
      int lid = element->LID();
      if (lid == -1) return 0;
      if (element->IsFaceElement())
        return (numdfcolfaces_ != Teuchos::null) ? (*numdfcolfaces_)[lid] : 0;
      else
        return (*numdfcolelements_)[lid];
    }

    /// Get the gid of a dof for given node
    int Dof(const Node* node, int dof) const override
    {
      int lid = node->LID();
      if (lid == -1) return -1;
      if (pccdofhandling_)
        return dofscolnodes_->GID((*shiftcolnodes_)[lid] + dof);
      else
        return (*idxcolnodes_)[lid] + dof;
    }

    /// Get the gid of a dof for given element
    int Dof(const Element* element, int dof) const override
    {
      int lid = element->LID();
      if (lid == -1) return -1;
      if (element->IsFaceElement())
        return (idxcolfaces_ != Teuchos::null) ? (*idxcolfaces_)[lid] + dof : -1;
      else
        return (*idxcolelements_)[lid] + dof;
    }

    /// Get the gid of all dofs of a node
    std::vector<int> Dof(const Node* node) const override
    {
      const int lid = node->LID();
      if (lid == -1) return std::vector<int>();
      const int idx = (*idxcolnodes_)[lid];
      std::vector<int> dof((*numdfcolnodes_)[lid]);
      for (unsigned i = 0; i < dof.size(); ++i)
      {
        if (pccdofhandling_)
          dof[i] = dofscolnodes_->GID((*shiftcolnodes_)[lid] + i);
        else
          dof[i] = idx + i;
      }
      return dof;
    }

    /// Get the gid of all dofs of a node
    void Dof(std::vector<int>& dof,  ///< vector of dof gids (to be filled)
        const Node* node,            ///< the node
        unsigned nodaldofset  ///< number of nodal dof set of the node (currently !=0 only for XFEM)
    ) const override
    {
      FOUR_C_ASSERT(nodaldofset == 0, "only one nodal dofset supported!");
      dof = Dof(node);
    }

    /// Get the gid of all dofs of a element
    std::vector<int> Dof(const Element* element) const override
    {
      int lid = element->LID();
      if (lid == -1) return std::vector<int>();

      if (element->IsFaceElement() && idxcolfaces_ == Teuchos::null) return std::vector<int>();

      int idx = element->IsFaceElement() ? (*idxcolfaces_)[lid] : (*idxcolelements_)[lid];
      std::vector<int> dof(
          element->IsFaceElement() ? (*numdfcolfaces_)[lid] : (*numdfcolelements_)[lid]);
      for (unsigned i = 0; i < dof.size(); ++i) dof[i] = idx + i;
      return dof;
    }

    /// Get the gid of all dofs of a node
    void Dof(const Node* node, std::vector<int>& lm) const override
    {
      int lid = node->LID();
      if (lid == -1) return;
      int idx = (*idxcolnodes_)[lid];
      int size = (*numdfcolnodes_)[lid];
      for (int i = 0; i < size; ++i)
      {
        if (pccdofhandling_)
          lm.push_back(dofscolnodes_->GID((*shiftcolnodes_)[lid] + i));
        else
          lm.push_back(idx + i);
      }
    }

    /// Get the gid of all dofs of a node
    void Dof(const Node* node,      ///< node, for which you want the dof positions
        const unsigned startindex,  ///< first index of vector at which will be written to end
        std::vector<int>& lm        ///< already allocated vector to be filled with dof positions
    ) const override
    {
      const int lid = node->LID();
      if (lid == -1) return;
      const int idx = (*idxcolnodes_)[lid];
      const int size = (*numdfcolnodes_)[lid];
      FOUR_C_ASSERT(lm.size() >= (startindex + size), "vector<int> lm too small");
      for (int i = 0; i < size; ++i)
      {
        if (pccdofhandling_)
          lm[startindex + i] = dofscolnodes_->GID((*shiftcolnodes_)[lid] + i);
        else
          lm[startindex + i] = idx + i;
      }
    }

    /// Get the gid of all dofs of a element
    void Dof(const Element* element, std::vector<int>& lm) const override
    {
      int lid = element->LID();
      if (lid == -1) return;

      if (element->IsFaceElement() && idxcolfaces_ == Teuchos::null) return;

      int idx = element->IsFaceElement() ? (*idxcolfaces_)[lid] : (*idxcolelements_)[lid];
      int size = element->IsFaceElement() ? (*numdfcolfaces_)[lid] : (*numdfcolelements_)[lid];
      for (int i = 0; i < size; ++i) lm.push_back(idx + i);
    }

    /// Get the GIDs of the first DOFs of a node of which the associated element is interested in
    void Dof(
        const Element* element,  ///< element which provides its expected number of DOFs per node
        const Node* node,        ///< node, for which you want the DOF positions
        std::vector<int>& lm     ///< already allocated vector to be filled with DOF positions
    ) const override
    {
      const int lid = node->LID();
      if (lid == -1) return;
      const int idx = (*idxcolnodes_)[lid];
      // this method is used to setup the vector of number of dofs, so we cannot ask
      // numdfcolelements_ here as above. Instead we have to ask the node itself
      const int size = NumDofPerNode(*element, *node);
      for (int i = 0; i < size; ++i)
      {
        if (pccdofhandling_)
          lm.push_back(dofscolnodes_->GID((*shiftcolnodes_)[lid] + i));
        else
          lm.push_back(idx + i);
      }
    }

    /// are the dof maps already initialized?
    bool Initialized() const override;

    /// Get degree of freedom row map
    const Epetra_Map* DofRowMap() const override;

    /// Get degree of freedom column map
    const Epetra_Map* DofColMap() const override;

    //! Print this class
    void Print(std::ostream& os) const override;

    //! Return true if \ref AssignDegreesOfFreedom was called
    bool Filled() const override { return filled_; }

    //@}

    //! @name Construction

    /*!
    \brief Assign dof numbers using all elements and nodes of the discretization.

    @param[in] dis Discretization of a mortar interface
    @param[in] dspos Position of DOfSet inside its discretization
    @param[in] start User-defined offset for DOF numbering [currently not supported]

    @return Maximum dof number of this dofset
    */
    int AssignDegreesOfFreedom(
        const Discretization& dis, const unsigned dspos, const int start) override;

    /// reset all internal variables
    void Reset() override;

    //@}

    //! @name Proxy management
    /// Proxies need to know about changes to the DofSet.

    /// our original DofSet dies
    void Disconnect(DofSetInterface* dofset) override { return; };

    //@}

    /// Get Number of Global Elements of degree of freedom row map
    int NumGlobalElements() const override;

    /// Get maximum GID of degree of freedom row map
    int MaxAllGID() const override;

    /// Get minimum GID of degree of freedom row map
    int MinAllGID() const override;


   protected:
    /// get number of nodal dofs
    int NumDofPerNode(const Node& node) const override
    {
      const int numele = node.NumElement();
      const DRT::Element* const* myele = node.Elements();
      int numdf = 0;
      for (int j = 0; j < numele; ++j) numdf = std::max(numdf, NumDofPerNode(*myele[j], node));
      return numdf;
    }

    /// get number of nodal dofs for this element at this node
    virtual int NumDofPerNode(const Element& element, const Node& node) const
    {
      return element.NumDofPerNode(node);
    }

    /// get number of element dofs for this element
    virtual int NumDofPerElement(const Element& element) const
    {
      return element.NumDofPerElement();
    }

    virtual int NumDofPerFace(const Element& element, int face) const
    {
      return element.NumDofPerFace(face);
    }

    /// Get Reserved Max Number Dofs per Node
    virtual void GetReservedMaxNumDofperNode(int& maxnodenumdf) { return; };

    /// get first number to be used as Dof GID in AssignDegreesOfFreedom
    virtual int GetFirstGIDNumberToBeUsed(const Discretization& dis) const;

    /// get minimal node GID to be used in AssignDegreesOfFreedom
    virtual int GetMinimalNodeGIDIfRelevant(const Discretization& dis) const;

    /// filled flag
    bool filled_;

    /// position of dofset inside its discretization
    unsigned dspos_;

    /// unique row map of degrees of freedom (node, face, and element dofs))
    Teuchos::RCP<Epetra_Map> dofrowmap_;

    /// unique column map of degrees of freedom (node, face, and element dofs)
    Teuchos::RCP<Epetra_Map> dofcolmap_;

    /// number of dofs for each node
    Teuchos::RCP<Epetra_IntVector> numdfcolnodes_;

    /// number of dofs for each face
    Teuchos::RCP<Epetra_IntVector> numdfcolfaces_;

    /// number of dofs for each element
    Teuchos::RCP<Epetra_IntVector> numdfcolelements_;

    /// column map gid of first dof for each node
    Teuchos::RCP<Epetra_IntVector> idxcolnodes_;

    /// column map gid of first dof for each face
    Teuchos::RCP<Epetra_IntVector> idxcolfaces_;

    /// column map gid of first dof for each element
    Teuchos::RCP<Epetra_IntVector> idxcolelements_;

    /*!
    \brief Activate special dof handling due to point coupling conditions?

    \note In the logic outlined above and implemented in AssignDegreesOfFreedom() we
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
       #dofscolnodes_ (Epetra_Map) and #shiftcolnodes_ (Epetra_IntVector)

    Nevertheless, for the time being, the old version with #idxcolnodes_ stays
    around, since the new version is only implemented for the standard DofSet
    class declared here (but not for special TransparentDofSets, Proxy, etc.).

    \author popp \date 02/2016
    */
    bool pccdofhandling_;

    /// column map of all dofs for each node (possibly non-unique)
    Teuchos::RCP<Epetra_Map> dofscolnodes_;

    /// shift value for access to column map of all dofs for each node
    Teuchos::RCP<Epetra_IntVector> shiftcolnodes_;
    //***************************************************************************

  };  // class DofSet
}  // namespace DRT


// << operator
std::ostream& operator<<(std::ostream& os, const DRT::DofSet& dofset);


FOUR_C_NAMESPACE_CLOSE

#endif
