// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_SELFCONTACT_BINARYTREE_UNBIASED_HPP
#define FOUR_C_CONTACT_SELFCONTACT_BINARYTREE_UNBIASED_HPP

#include "4C_config.hpp"

#include "4C_contact_selfcontact_binarytree.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  /*!
  \brief A class for performing unbiased self contact search in 2D / 3D based on a binary search
  tree and processor specific dual graphs to enhance parallel search

  */

  class UnbiasedSelfBinaryTree : public SelfBinaryTree
  {
   public:
    /*!
    \brief Standard constructor

    \param discret (in):    The contact interface discretization
    \param iparams (in):    interface specific parameter list
    \param elements (in):   All elements on self contact interface (fully overlapping map)
    \param dim (in):        The problem dimension

    */
    UnbiasedSelfBinaryTree(Core::FE::Discretization& discret, const Teuchos::ParameterList& iparams,
        std::shared_ptr<Core::LinAlg::Map> elements, int dim, double eps);


    /*!
    \brief Initialize the unbiased self binary tree

    */
    void init() final;

   private:
    //! @name Evaluation methods

    /*!
    \brief Decide whether tree nodes need to be added to contact pairs

    \param [in]  treenode1:  first tree node
    \param [in]  treenode2:  second tree node

    */
    void add_tree_nodes_to_contact_pairs(std::shared_ptr<SelfBinaryTreeNode> treenode1,
        std::shared_ptr<SelfBinaryTreeNode> treenode2) final;

    /*!
    \brief Calculate the processor specific dual graph

    \param [out] dualgraph:  construction of binary tree is based on this data
    \param [in]  elelist:    list (gids) of all contact elements of the surface
    \param [in]  p:          number of current (not necessarily calling) processor

    */
    void calculate_proc_specific_dual_graph(
        std::map<std::shared_ptr<SelfDualEdge>, std::vector<std::shared_ptr<SelfDualEdge>>>*
            dualGraph,
        const std::vector<int>& elelist, const int p);

    /*!
    \brief Use list of possible contact pairs to define the search elements

    */
    void define_search_elements();

    /*!
    \brief Get the mortar element specific number of first order nodes

    \param [in]      contractedEdge:  dual edge that is contracted
    \param [in,out]  contractedNode:  node that consists of both nodes of contracted edge

    */
    void get_contracted_node(std::shared_ptr<SelfDualEdge>& contractedEdge,
        std::shared_ptr<SelfBinaryTreeNode>& contractedNode) final;

    /*!
    \brief Initialize Tree in a bottom up way based on dual graph

    \param [in] procdualgraph:  processor specific dual graph, construction of binary tree is based
    on this data

    */
    void initialize_tree_bottom_up(std::map<int,
        std::map<std::shared_ptr<SelfDualEdge>, std::vector<std::shared_ptr<SelfDualEdge>>>>*
            procdualGraph);

    /*!
    \brief Checks roughly whether self contact of two elements shall be evaluated (3D)

    This method checks if the normal at the slave element center and the vector connecting this
    slave element center and the center of the element it is projected to (master element) point in
    the same direction. All these vectors are evaluated in the reference configuration to check the
    initial state. Only if both vectors point in the same direction integration shall be performed.

    */
    bool rough_check_ref_config(int ele1gid, int ele2gid);

    /*!
    \brief Evaluate unbiased binary search tree

    */
    void search_contact() final;

    /*!
    \brief Communicate the Search Elements to all processors

    */
    void communicate_search_elements_all_procs();
    //@}

    // don't want = operator and cctor
    UnbiasedSelfBinaryTree operator=(const SelfBinaryTree& old) = delete;
    UnbiasedSelfBinaryTree(const SelfBinaryTree& old) = delete;

    //! use two half pass approach
    const bool two_half_pass_;
    //! perform reference configuration check for non-smooth self contact
    const bool check_nonsmooth_selfcontactsurface_;
    //! the contact pairs are communicated to all processors
    const bool searchele_all_proc_;
  };  // class UnbiasedSelfBinaryTree

  /**
   * \brief Calculate the global position in the reference configuration for a given element at a
   * given position in parameter space \f$ \vec{\xi} \f$
   *
   *  \param [in]   element: element for which position shall be calculated
   *  \param [in]        xi: position in parameter space
   *  \param [out]    coord: global position in reference configuration for element at parameter
   * space position xi
   */
  template <int probdim, Core::FE::CellType distype>
  static void local_to_global_position_at_xi_ref_config(const Core::Elements::Element* element,
      const Core::LinAlg::Matrix<Core::FE::dim<distype>, 1>& xi,
      Core::LinAlg::Matrix<probdim, 1>& coord)
  {
    static Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, 1> funct(
        Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<probdim, Core::FE::num_nodes<distype>> nodecoords(
        Core::LinAlg::Initialization::zero);

    Core::FE::shape_function<distype>(xi, funct);

    const Core::Nodes::Node* const* nodes = element->nodes();
    const int nodedim = nodes[0]->n_dim();

    FOUR_C_ASSERT_ALWAYS(nodes, "ERROR: Did not get nodes of element!");
    FOUR_C_ASSERT_ALWAYS(probdim == nodedim,
        "Problem dimension: {} and dimension of nodes: {} does not match!", probdim, nodedim);

    for (int i = 0; i < Core::FE::num_nodes<distype>; ++i)
    {
      for (int j = 0; j < nodedim; ++j)
      {
        nodecoords(j, i) = nodes[i]->x()[j];
      }
    }

    coord.multiply(1.0, nodecoords, funct, 0.0);
  }

  /**
   * \brief Calculate the normal in the reference configuration for a given element at a given
   * position in parameter space \f$ \vec{\xi} \f$
   *
   *  \param [in]    element: element for which position shall be calculated
   *  \param [in]         xi: position in parameter space
   *  \param [out]    normal: normal in reference configuration for element at parameter space
   * position xi
   */
  template <Core::FE::CellType distype>
  static void compute_unit_normal_at_xi_ref_config(const Core::Elements::Element* element,
      const Core::LinAlg::Matrix<Core::FE::dim<distype>, 1>& xi, Core::LinAlg::Matrix<3, 1>& normal)
  {
    static Core::LinAlg::Matrix<3, Core::FE::dim<distype>> gxieta(
        Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::num_nodes<distype>> deriv(
        Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, Core::FE::num_nodes<distype>> nodecoords(
        Core::LinAlg::Initialization::zero);

    Core::FE::shape_function_deriv1<distype>(xi, deriv);

    const Core::Nodes::Node* const* nodes = element->nodes();
    const int nodedim = nodes[0]->n_dim();

    FOUR_C_ASSERT_ALWAYS(nodes, "ERROR: Did not get nodes of element!");
    FOUR_C_ASSERT_ALWAYS(nodedim == 3, "ERROR: Only implemented for 3D cases so far!");

    for (int i = 0; i < Core::FE::num_nodes<distype>; ++i)
    {
      for (int j = 0; j < nodedim; ++j)
      {
        nodecoords(j, i) = nodes[i]->x()[j];
      }
    }

    gxieta.multiply_nt(1.0, nodecoords, deriv, 0.0);
    static Core::LinAlg::Matrix<3, 1> gxi(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<3, 1> geta(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<2, 1> first(Core::LinAlg::Initialization::zero);
    static Core::LinAlg::Matrix<2, 1> second(Core::LinAlg::Initialization::zero);
    first(0, 0) = 1.0;
    second(1, 0) = 1.0;
    gxi.multiply(1.0, gxieta, first, 0.0);
    geta.multiply(1.0, gxieta, second, 0.0);

    // clear, calculate and scale normal
    normal.clear();
    normal.cross_product(gxi, geta);
    const double normnormal = normal.norm2();
    normal.scale(1.0 / normnormal);
  }

}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
