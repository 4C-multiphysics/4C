// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_NODE_HPP
#define FOUR_C_FEM_GENERAL_NODE_HPP


#include "4C_config.hpp"

#include "4C_comm_parobject.hpp"
#include "4C_comm_parobjectfactory.hpp"
#include "4C_fem_discretization_iterator.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::Elements
{
  class Element;
}

namespace Core::Conditions
{
  class Condition;
}


namespace Core::FE
{
  class ConstElementRef;
  class ElementRef;
  class Discretization;
}  // namespace Core::FE

namespace Core::Nodes
{
  class NodeType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "NodeType"; }

    static NodeType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static NodeType instance_;
  };

  /*!
  \brief A virtual class all nodes that are used in the discretization management module have to
  implement

  */
  class Node : public Core::Communication::ParObject
  {
   public:
    //! @name Enums and Friends

    /*!
    \brief The discretization is a friend of Node
    */
    friend class Core::FE::Discretization;

    //@}

    //! @name Constructors and destructors and related methods

    /*!
    \brief Standard Constructor

    \param id     (in): A globally unique node id
    \param coords (in): vector of nodal coordinates
    \param owner  (in): Owner of this node.
    */
    Node(int id, std::span<const double> coords, int owner);

    /*!
    \brief Deep copy the derived class and return pointer to it

    */
    virtual Node* clone() const;


    /*!
    \brief Return unique ParObject id

    every class imploementing ParObject needs a unique id defined at the
    top of this file.
    */
    int unique_par_object_id() const override
    {
      return NodeType::instance().unique_par_object_id();
    }

    /*!
    \brief Pack this class so it can be communicated

    \ref pack and \ref unpack are used to communicate this node

    */
    void pack(Core::Communication::PackBuffer& data) const override;

    /*!
    \brief Unpack data from a char vector into this class

    \ref pack and \ref unpack are used to communicate this node

    */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    //! @name Access methods

    /*!
    \brief Return global id
    */
    inline int id() const { return id_; }

    /*!
    \brief Return processor local col map id
    */
    inline int lid() const { return lid_; }

    /*!
    \brief Return owner of this node
    */
    inline int owner() const { return owner_; }

    /*!
    \brief Return coordinates vector
    */
    inline std::span<const double> x() const { return std::span<const double>(x_); }

    /*!
    \brief return spatial dimension of node coordinates
    */
    inline int n_dim() const { return x_.size(); }

    /**
     * Return the number of elements adjacent to this node.
     */
    [[nodiscard]] int num_element() const;

    /**
     * A range containing a ElementRef to all elements adjacent to this node.
     */
    [[nodiscard]] FE::IteratorRange<FE::DiscretizationIterator<FE::ElementRef>> adjacent_elements();

    /**
     * A range containing a ConstElementRef to all elements adjacent to this node.
     */
    [[nodiscard]] FE::IteratorRange<FE::DiscretizationIterator<FE::ConstElementRef>>
    adjacent_elements() const;


    /// Get minimum distance to the nodes contained in the adjacent elements
    [[nodiscard]] double minimum_distance_to_adjacent_nodes() const;

    /*!
    \brief Print this node
    */
    virtual void print(std::ostream& os) const;

    //@}

    //! @name Construction

    /*!
      \brief Set processor local col id
      \param lid: processor local col id
     */
    inline void set_lid(int lid) { lid_ = lid; }

    /*!
    \brief Set ownership

    \param owner: Proc owning this node
    */
    inline void set_owner(const int owner) { owner_ = owner; }

    /*!
    \brief Change reference position by adding input vector to position
    */
    void change_pos(std::vector<double> nvector);

    /*!
    \brief Change reference position by setting input vector to position
    */
    void set_pos(std::vector<double> nvector);

    //@}

    /**
     * Access the discretization managing this node. This may be a nullptr if the node is not
     * part of a discretization.
     */
    const FE::Discretization* discretization() const { return discretization_; }
    FE::Discretization* discretization() { return discretization_; }

   protected:
    //! a unique global id
    int id_;
    //! local col map id
    int lid_;
    //! proc owning this node
    int owner_;
    //! nodal coords
    std::vector<double> x_;

    //! Refer to discretization managing this node
    FE::Discretization* discretization_{};
  };  // class Node
}  // namespace Core::Nodes


// << operator
std::ostream& operator<<(std::ostream& os, const Core::Nodes::Node& node);



FOUR_C_NAMESPACE_CLOSE

#endif
