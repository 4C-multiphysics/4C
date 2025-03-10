// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MORTAR_BASE_BINARYTREE_HPP
#define FOUR_C_MORTAR_BASE_BINARYTREE_HPP

#include "4C_config.hpp"

#include "4C_mortar_abstract_binarytree.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE
namespace CONTACT
{
  class SelfBinaryTree;
  class SelfDualEdge;
  class UnbiasedSelfBinaryTree;
}  // namespace CONTACT

namespace Mortar
{
  /*!
  \brief A base class for binary tree nodes

  */
  class BaseBinaryTreeNode : public AbstractBinaryTreeNode
  {
    // these classes need access to protected methods of this class and are therefore defined friend
    // classes here
    friend class BinaryTree;
    friend class CONTACT::SelfBinaryTree;
    friend class CONTACT::SelfDualEdge;
    friend class CONTACT::UnbiasedSelfBinaryTree;

   public:
    /*!
    \brief Standard constructor of a base binary tree node

    \param [in] discret:     interface discretization
    \param [in] elelist:     list of all elements in BaseBinaryTreeNode
    \param [in] dopnormals:  reference to DOP normals
    \param [in] kdop:        reference to no. of vertices
    \param [in] dim:         dimension of the problem
    \param [in] useauxpos:   bool whether auxiliary position is used when computing dops
    \param [in] layer:       current layer of tree node

    */
    BaseBinaryTreeNode(Core::FE::Discretization& discret, std::vector<int> elelist,
        const Core::LinAlg::SerialDenseMatrix& dopnormals, const int& kdop, const int& dim,
        const bool& useauxpos, const int layer);


    //! @name Evaluation methods

    /*!
    \brief Calculate slabs of dop

    */
    void calculate_slabs_dop() override;

    /*!
    \brief Update slabs of current tree node in bottom up way

    */
    void update_slabs_bottom_up(double& eps) override = 0;

    /*!
    \brief Enlarge geometry of a tree node by an offset, dependent on size

    */
    void enlarge_geometry(double& enlarge) override;
    //@}

    //! @name Print methods
    // please note: these methods do not get called and are therefore untested. However, they might
    // be of interest for debug and development purposes
    /*!
    \brief Print type of tree node to std::cout

    */
    virtual void print_type() = 0;

    /*!
    \brief Print slabs to std::cout

    */
    void print_slabs();
    //@}

   protected:
    //! @name Access and modification methods
    /*!
    \brief Return dim of Problem

    */
    const int& n_dim() const { return dim_; }

    /*!
    \brief Get discretization of the interface

    */
    Core::FE::Discretization& discret() const { return idiscret_; }

    /*!
    \brief Return pointer to normals of DOP

    */
    const Core::LinAlg::SerialDenseMatrix& dopnormals() const { return dopnormals_; }

    /*!
    \brief Return pointer to element list of tree node

    */
    std::vector<int> elelist() const { return elelist_; }

    /*!
    \brief Return no. of vertices

    */
    const int& kdop() const { return kdop_; }

    /*!
    \brief Return layer of current tree node

    */
    int get_layer() const { return layer_; }

    /*!
    \brief Set layer of current tree node

    */
    void set_layer(int layer) { layer_ = layer; }

    /*!
    \brief Return pointer to slabs of DOP

    */
    Core::LinAlg::SerialDenseMatrix& slabs() { return slabs_; }

    /*!
    \brief Return bool indicating whether auxiliary position is used when computing dops

    */
    const bool& use_aux_pos() const { return useauxpos_; }
    //@}

   private:
    //! dimension of the problem
    const int dim_;
    //! reference to DOP normals
    const Core::LinAlg::SerialDenseMatrix& dopnormals_;
    //! list containing the gids of all elements of the tree node
    std::vector<int> elelist_;
    //! interface discretization
    Core::FE::Discretization& idiscret_;
    //! number of vertices
    const int kdop_;
    //! layer of tree node in tree (0=root node!)
    int layer_;
    //! geometry slabs of tree node, saved as Min|Max
    Core::LinAlg::SerialDenseMatrix slabs_;
    //! auxiliary position is used when computing dops
    const bool useauxpos_;
  };  // class BaseBinaryTreeNode

  /*!
  \brief A base class for binary trees

  */
  class BaseBinaryTree : public AbstractBinaryTree
  {
   public:
    /*!
    \brief Standard constructor

    \param [in] discret: interface discretization
    \param [in] dim:     dimension of the problem
    \param [in] eps:     factor used to enlarge dops
    */
    BaseBinaryTree(Core::FE::Discretization& discret, int dim, double eps);


    //! @name Evaluation methods

    /*!
    \brief Evaluate search tree

    */
    void evaluate_search() override = 0;

    /*!
    \brief Initialize the base binary tree

    */
    void init() override;

    /*!
    \brief Calculate minimal element length / inflation factor "enlarge"

    */
    virtual void set_enlarge() = 0;
    //@}

   protected:
    //! @name Access and modification methods
    /*!
    \brief Return dim of the problem

    */
    const int& n_dim() const { return dim_; }

    /*!
    \brief Get discretization of the interface

    */
    Core::FE::Discretization& discret() const { return idiscret_; }

    /*!
    \brief Get matrix of DOP normals

    */
    const Core::LinAlg::SerialDenseMatrix& dop_normals() const { return dopnormals_; }

    /*!
    \brief Return factor "enlarge" to enlarge dops

    */
    double& enlarge() { return enlarge_; }

    /*!
    \brief Return factor "eps" to set "enlarge"

    */
    const double& eps() const { return eps_; }

    /*!
    \brief Get number of vertices of DOP

    */
    const int& kdop() const { return kdop_; }
    //@}

   private:
    /*!
    \brief Initialize internal variables

     */
    virtual void init_internal_variables() = 0;

    //! interface discretization
    Core::FE::Discretization& idiscret_;
    //! problem dimension (2D or 3D)
    const int dim_;
    //! normals of DOP
    Core::LinAlg::SerialDenseMatrix dopnormals_;
    //! needed to enlarge dops
    double enlarge_;
    //! epsilon for enlarging dops (of user)
    const double eps_;
    //! set k for DOP (8 for 2D, 18 for 3D)
    int kdop_;
  };  // class BaseBinaryTree
}  // namespace Mortar

FOUR_C_NAMESPACE_CLOSE

#endif
