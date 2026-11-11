// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_ABSTRACT_BINARYTREE_HPP
#define FOUR_C_CONTACT_ABSTRACT_BINARYTREE_HPP

#include "4C_config.hpp"

#include "4C_linalg_serialdensematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  /*!
  \brief An abstract interface for binary trees

  */
  class AbstractBinaryTreeNode
  {
   public:
    /*!
    \brief Standard constructor

    */
    AbstractBinaryTreeNode() { return; };

    /*!
    \brief Destructor

    */
    virtual ~AbstractBinaryTreeNode() = default;

    //! @name Evaluation methods

    /*!
    \brief Calculate slabs of dop

    */
    virtual void calculate_slabs_dop() = 0;

    /*!
    \brief Update slabs of current treenode in bottom up way

    */
    virtual void update_slabs_bottom_up(double& eps) = 0;

    /*!
    \brief Enlarge geometry of a Treenode by an offset, dependent on size

    */
    virtual void enlarge_geometry(double& eps) = 0;
  };  // class AbstractBinaryTreeNode


  /*!
  \brief An abstract interface for binary trees

  */
  class AbstractBinaryTree
  {
   public:
    /*!
    \brief Standard constructor

    */
    AbstractBinaryTree() { return; };

    /*!
    \brief Destructor

    */
    virtual ~AbstractBinaryTree() = default;

    //! @name Query methods

    /*!
    \brief Evaluate search tree to get corresponding master elements for the slave elements

    */
    virtual void evaluate_search() = 0;

    /*!
    \brief initialize the binary tree

    */
    virtual void init() = 0;
  };  // class AbstractBinaryTree
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
