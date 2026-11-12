// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_SELFCONTACT_BINARYTREE_HPP
#define FOUR_C_CONTACT_SELFCONTACT_BINARYTREE_HPP

#include "4C_config.hpp"

#include "4C_contact_base_binarytree.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_map.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace CONTACT
{
  // forward declarations

  //! @name Enums and Friends
  enum SelfBinaryTreeNodeType
  {
    SELFCO_INNER,        ///< indicates an inner node (has children)
    SELFCO_LEAF,         ///< indicates a leaf node (no further children)
    SELFCO_NO_ELEMENTS,  ///< indicates that there are no elements on this (root) treenode
    SELFCO_UNDEFINED     ///< indicates an undefined tree node
  };

  //@}

  /*!
  \brief A class representing one tree node of the binary tree for self contact

  Refer also to the Semesterarbeit of Anh-Tu Vuong, 2009

  */
  class SelfBinaryTreeNode : public BaseBinaryTreeNode
  {
   public:
    /*!
    \brief Constructor of a tree node

    \param type           type of SelfBinaryTreeNode
    \param discret        contact interface discretization
    \param parent         points to parent tree node
    \param elelist        list of all elements in SelfBinaryTreeNode
    \param dopnormals     reference to DOP normals
    \param samplevectors  reference to sample vectors
    \param kdop           reference to no. of vertices
    \param dim            dimension of problem
    \param layer          current layer of tree node
    \param nonsmoothsurf  non-smooth self contact surface
    \param treenodes      references to tree nodes storage scheme

    */
    SelfBinaryTreeNode(SelfBinaryTreeNodeType type, Core::FE::Discretization& discret,
        std::shared_ptr<SelfBinaryTreeNode> parent, std::vector<int> elelist,
        const Core::LinAlg::SerialDenseMatrix& dopnormals,
        const Core::LinAlg::SerialDenseMatrix& samplevectors, const int& kdop, const int& dim,
        const int& nvectors, const int layer, const bool nonsmoothsurf,
        std::vector<std::vector<std::shared_ptr<SelfBinaryTreeNode>>>& treenodes);


    //! @name Evaluation methods

    /*!
    \brief Update slabs of current tree node in bottom up way

    */
    void update_slabs_bottom_up(double& enlarge) final;

    /*!
    \brief Calculate the logical array of qualified sample vectors for leaf nodes

    */
    void calculate_qualified_vectors();

    /*!
    \brief Update the logical array of qualified sample vectors for non-leaf nodes

    */
    void update_qualified_vectors_bottom_up();

    /*!
    \brief Return logical array of qualified sample vectors

    */
    std::vector<bool> qualified_vectors() const { return qualifiedvectors_; }

    /*!
    \brief Print type of tree node to std::cout

    */
    void print_type() final;
    //@}

    //! @name Access methods

    /*!
    \brief Get communicator

    */
    MPI_Comm get_comm() const;

    /*!
    \brief Complete tree by filling tree node storage scheme

    */
    void complete_tree(int layer, double& enlarge);

    /*!
    \brief Return pointer to type of treenode

    */
    SelfBinaryTreeNodeType type() const { return type_; }

    /*!
    \brief Return pointer to adjacent tree nodes

    */
    std::vector<std::shared_ptr<SelfBinaryTreeNode>> adjacent_treenodes()
    {
      return adjacent_treenodes_;
    }

    /*!
    \brief set adjacent tree nodes

    */
    void set_adjacent_tnodes(std::vector<std::shared_ptr<SelfBinaryTreeNode>> adjTnodes)
    {
      adjacent_treenodes_ = adjTnodes;
    }

    /*!
    \brief Return list of endnodes

    */
    std::vector<int> endnodes() const { return endnodes_; }

    /*!
    \brief Set list of endnodes

    */
    void set_endnodes(std::vector<int> endnodes) { endnodes_ = endnodes; }

    /*!
    \brief Update list of endnodes with endnodes of children

    */
    void update_endnodes();

    /*!
    \brief Return pointer to right child

    */
    std::shared_ptr<SelfBinaryTreeNode> rightchild() const { return rightchild_; }

    /*!
    \brief Return pointer to left child

    */
    std::shared_ptr<SelfBinaryTreeNode> leftchild() const { return leftchild_; }

    /*!
    \brief set children of a Binary Tree Node

    */
    void set_children(std::shared_ptr<SelfBinaryTreeNode> leftchild,
        std::shared_ptr<SelfBinaryTreeNode> rightchild);

    /*!
    \brief Return pointer to parent

    */
    std::shared_ptr<SelfBinaryTreeNode> parent() const { return parent_; }

    /*!
    \brief set parent of tree node

    */
    void set_parent(std::shared_ptr<SelfBinaryTreeNode> parent) { parent_ = parent; }

    /*!
    \brief Return owner of current tree node

    */
    int owner() const { return owner_; }

    /*!
    \brief set owner of tree node

    */
    void set_owner(int treenodeowner) { owner_ = treenodeowner; }

    /*!
    \brief set owner of parent according to owner of children

    */
    void set_parent_owner(int leftchildowner, int rightchildowner);
    //@}

   private:
    // don't want = operator and cctor
    SelfBinaryTreeNode operator=(const SelfBinaryTreeNode& old) = delete;
    SelfBinaryTreeNode(const SelfBinaryTreeNode& old) = delete;

    //! type of SelfBinaryTreeNode
    SelfBinaryTreeNodeType type_;

    // the pointers to the parent as well as to the left and right child are not moved to the
    // BaseBinaryTreeNode as this would require a lot of dynamic casting and thereby complicating
    // the readability of the code
    //! pointer to the parent SelfBinaryTreeNode
    std::shared_ptr<SelfBinaryTreeNode> parent_;

    //! pointer to the left child TreeNode
    std::shared_ptr<SelfBinaryTreeNode> leftchild_;

    //! pointer to the right child TreeNode
    std::shared_ptr<SelfBinaryTreeNode> rightchild_;

    //! logical array of qualified sample vectors of current tree node
    std::vector<bool> qualifiedvectors_;

    //! vector with global IDs of end nodes of a surface (2D), -1 if there are no end-nodes
    std::vector<int> endnodes_;

    //! vector pointers to adjacent treenodes on the same layer
    std::vector<std::shared_ptr<SelfBinaryTreeNode>> adjacent_treenodes_;

    //! reference to sample vectors
    const Core::LinAlg::SerialDenseMatrix& samplevectors_;

    //! reference to number of sample vectors
    const int& nvectors_;

    //! owner of the tree node
    int owner_;

    //! tree node is part of a non-smooth self contact surface
    const bool nonsmoothsurf_;

    //! reference to storage scheme of all tree nodes, sorted by layer
    std::vector<std::vector<std::shared_ptr<SelfBinaryTreeNode>>>& treenodes_;

    // relational operators for binary tree nodes

    //! operator <
    friend bool operator<(SelfBinaryTreeNode& node1, SelfBinaryTreeNode& node2)
    {
      if (node1.elelist().size() < node2.elelist().size())
        return true;
      else if (node1.elelist().size() == node2.elelist().size() and
               node1.elelist()[0] < node2.elelist()[0])
        return true;
      else
        return false;
    }

    //! operator >
    friend bool operator>(SelfBinaryTreeNode& node1, SelfBinaryTreeNode& node2)
    {
      return operator<(node2, node1);
    }

    //! operator <=
    friend bool operator<=(SelfBinaryTreeNode& node1, SelfBinaryTreeNode& node2)
    {
      return !operator>(node1, node2);
    }

    //! operator >=
    friend bool operator>=(SelfBinaryTreeNode& node1, SelfBinaryTreeNode& node2)
    {
      return !operator<(node1, node2);
    }

    //! operator ==
    friend bool operator==(SelfBinaryTreeNode& node1, SelfBinaryTreeNode& node2)
    {
      if (node1.elelist().size() != node2.elelist().size())
        return false;
      else if (node1.elelist()[0] == node2.elelist()[0])
        return true;
      else
        return false;
    }

    //! operator !=
    friend bool operator!=(SelfBinaryTreeNode& node1, SelfBinaryTreeNode& node2)
    {
      return !operator==(node1, node2);
    }

  };  // class SelfBinaryTreeNode


  /*!
  \brief A class representing one edge of the dual graph for self contact search

  Refer also to the Semesterarbeit of Anh-Tu Vuong, 2009

  */
  class SelfDualEdge
  {
   public:
    // relational operators for dual edges

    //! operator ==
    friend bool operator==(SelfDualEdge& edge1, SelfDualEdge& edge2)
    {
      if ((*edge1.node1_ == *edge2.node1_) and (*edge1.node2_ == *edge2.node2_))
        return true;
      else if ((*edge1.node2_ == *edge2.node1_) and (*edge1.node1_ == *edge2.node2_))
        return true;
      else
        return false;
    }

    //! operator !=
    friend bool operator!=(SelfDualEdge& edge1, SelfDualEdge& edge2)
    {
      return !operator==(edge1, edge2);
    }

    //! operator <
    friend bool operator<(
        const std::shared_ptr<SelfDualEdge> edge1, const std::shared_ptr<SelfDualEdge> edge2)
    {
      // Use the RCPs to avoid auto-removing them. Since this function is used for sorting a vector
      // of RCPs, the signature needs to stay like this.
      (void)edge1;
      (void)edge2;

      if (edge1->costs_ < edge2->costs_)
        return true;
      else if (edge1->costs_ > edge2->costs_)
        return false;
      else if (*edge1 != *edge2)
      {
        if (*edge1->greater_node() < *edge2->greater_node())
          return true;
        else if (*edge1->greater_node() == *edge2->greater_node())
        {
          if (*edge1->lesser_node() < *edge2->lesser_node())
            return true;
          else
            return false;
        }
        else
          return false;
      }
      else
        return false;
    }

    //! operator >
    friend bool operator>(
        const std::shared_ptr<SelfDualEdge> edge1, const std::shared_ptr<SelfDualEdge> edge2)
    {
      return operator<(edge2, edge1);
    }

    //! operator <=
    friend bool operator<=(
        const std::shared_ptr<SelfDualEdge> edge1, const std::shared_ptr<SelfDualEdge> edge2)
    {
      return !operator>(edge1, edge2);
    }

    //! operator >=
    friend bool operator>=(
        const std::shared_ptr<SelfDualEdge> edge1, const std::shared_ptr<SelfDualEdge> edge2)
    {
      return !operator<(edge1, edge2);
    }

    /*!
    \brief Constructor of a dual edge

    */
    SelfDualEdge(std::shared_ptr<SelfBinaryTreeNode> node1_,
        std::shared_ptr<SelfBinaryTreeNode> node2_, const int& dim);

    /*!
    \brief Destructor

    */
    virtual ~SelfDualEdge() = default;

    /*!
    \brief Return costs

    */
    double costs() const { return costs_; }

    /*!
    \brief Return first node of dual edge

    */
    std::shared_ptr<SelfBinaryTreeNode> get_node1() const { return node1_; }

    /*!
    \brief Return second node of dual edge

    */
    std::shared_ptr<SelfBinaryTreeNode> get_node2() const { return node2_; }

    /*!
    \brief Return common tree node of two dual edges

    */
    std::shared_ptr<SelfBinaryTreeNode> common_node(SelfDualEdge& treenode)
    {
      std::shared_ptr<SelfBinaryTreeNode> node1 = treenode.get_node1();
      std::shared_ptr<SelfBinaryTreeNode> node2 = treenode.get_node2();

      if (*get_node1() == *node1 or *get_node2() == *node1)
        return node1;
      else if (*get_node1() == *node2 or *get_node2() == *node2)
        return node2;
      else
        return nullptr;
    }

   private:
    /*!
    \brief Calculate the cost function of a dual edge

    */
    void calculate_costs();

    /*!
    \brief return greater node of dual edge

    */
    std::shared_ptr<SelfBinaryTreeNode> greater_node()
    {
      if (*node1_ > *node2_)
        return node1_;
      else if (*node2_ > *node1_)
        return node2_;
      else
        return node1_;
    }

    std::shared_ptr<SelfBinaryTreeNode> lesser_node()
    {
      if (*node1_ > *node2_)
        return node2_;
      else if (*node2_ > *node1_)
        return node1_;
      else
        return node1_;
    }

    // don't want = operator and cctor
    SelfDualEdge operator=(const SelfDualEdge& old) = delete;
    SelfDualEdge(const SelfDualEdge& old) = delete;

    //! first node of dual edge
    std::shared_ptr<SelfBinaryTreeNode> node1_;

    //! second node of dual edge
    std::shared_ptr<SelfBinaryTreeNode> node2_;

    //! cost function value of dual edge
    double costs_;

    //! reference to dim. of problem
    const int& dim_;

  };  // class SelfDualEdge


  /*!
  \brief A class for performing self contact search in 2D / 3D based
         on a binary search tree and dual graphs

  Refer also to the Semesterarbeit of Anh-Tu Vuong, 2009

  */

  class SelfBinaryTree : public BaseBinaryTree
  {
   public:
    /*!
    \brief Standard constructor

    Constructs an instance of this class.<br>
    For now, we only consider the serial case (1 processor) here!!!

    \param discret (in):    The contact interface discretization
    \param iparams (in):    interface specific parameter list
    \param elements (in):   All elements on self contact interface (fully overlapping map)
    \param dim (in):        The problem dimension

    */
    SelfBinaryTree(Core::FE::Discretization& discret, const Teuchos::ParameterList& iparams,
        std::shared_ptr<Core::LinAlg::Map> elements, int dim, double eps);


    //! @name Evaluation methods

    /*!
    \brief Evaluate search self binary tree: call set_enlarge() and search_contact()

    */
    void evaluate_search() final;

    /*!
    \brief Initialize the self binary tree

    */
    void init() override;

   protected:
    /*!
    \brief Initialize the leaf nodes and related map

    \param elelist:  gids of all contact elements of current surface

    */
    void init_leaf_nodes_and_map(std::vector<int>& elelist);
    //@}

    //! @name Access methods

    /*!
    \brief Get write access to the adjacency matrix

    */
    std::map<int, std::vector<std::shared_ptr<SelfBinaryTreeNode>>>& set_adjacencymatrix()
    {
      return adjacencymatrix_;
    }

    /*!
    \brief Get communicator

    */
    MPI_Comm get_comm() const;

    /*!
    \brief Get access to the contact pairs

    */
    std::map<int, std::vector<int>> contact_pairs() const { return contactpairs_; }

    /*!
    \brief Get write access to the contact pairs

    */
    std::map<int, std::vector<int>>& set_contact_pairs() { return contactpairs_; }

    /*!
    \brief Get map of leaf nodes

    */
    std::map<int, std::shared_ptr<SelfBinaryTreeNode>> leafsmap() const { return leafsmap_; }

    /*!
    \brief Get write access to map of leaf nodes

    */
    std::map<int, std::shared_ptr<SelfBinaryTreeNode>>& set_leafsmap() { return leafsmap_; }

    /*!
    \brief Return no. of sample vectors

    */
    const int& nvectors() const { return nvectors_; }

    /*!
    \brief Get root nodes

    */
    std::vector<std::shared_ptr<SelfBinaryTreeNode>> roots() const { return roots_; }

    /*!
    \brief Get write access to root nodes

    */
    std::vector<std::shared_ptr<SelfBinaryTreeNode>>& set_roots() { return roots_; }

    /*!
    \brief Get matrix of sample vectors

    */
    const Core::LinAlg::SerialDenseMatrix& sample_vectors() const { return samplevectors_; }

    /*!
    \brief Return reference to storage scheme of all tree nodes

    */
    std::vector<std::vector<std::shared_ptr<SelfBinaryTreeNode>>> treenodes() const
    {
      return treenodes_;
    }
    //@}

    //! @name Evaluation methods

    /*!
    \brief Decide whether tree nodes need to be added to contact pairs

    \param [in]  treenode1:       first tree node
    \param [in]  treenode2:       second tree node

    */
    virtual void add_tree_nodes_to_contact_pairs(std::shared_ptr<SelfBinaryTreeNode> treenode1,
        std::shared_ptr<SelfBinaryTreeNode> treenode2);
    /*!
    \brief Set the vector of adjacent tree nodes for leaf-nodes in the lowest layer

    */
    void calculate_adjacent_leaves();

    /*!
    \brief Calculate the vector of adjacent tree nodes of inner tree nodes

    */
    void calculate_adjacent_tnodes();

    /*!
    \brief Calculate the adjacent tree nodes and adjacent dual edges of current element

    \param [in]  possadjids:   vector of global IDs of possible adjacent elements
    \param [in]  gid:          global id of current element
    \param [in]  adjElementk:  k-th adjacent element of current element
    \param [in]  node1:        leaf node with global id "gid"
    \param [out] adjtreenodes: vector of adjacent tree nodes (elements) of current element
    \param [out] adjdualedges: vector of adjacent dual edges containing current element

     */
    void calculate_adjacent_tree_nodes_and_dual_edges(std::vector<int>& possadjids, const int gid,
        Core::Elements::Element* adjElementk, std::shared_ptr<SelfBinaryTreeNode>& node1,
        std::vector<std::shared_ptr<SelfBinaryTreeNode>>& adjtreenodes,
        std::vector<std::shared_ptr<SelfDualEdge>>& adjdualedges);

    /*!
    \brief Get the mortar element specific number of first order nodes

    \param [in] element:  element of which number of first order nodes shall be determined

    */
    int get_ele_specific_num_nodes(Core::Elements::Element* element);

    /*!
    \brief Get the (contracted) node that combines the nodes of the contracted edge

    \param [in]      contractedEdge:  dual edge that is contracted
    \param [in,out]  contractedNode:  node that consists of both nodes of contracted edge

    */
    virtual void get_contracted_node(std::shared_ptr<SelfDualEdge>& contractedEdge,
        std::shared_ptr<SelfBinaryTreeNode>& contractedNode);

    /*!
    \brief Initialize internal variables

     */
    void init_internal_variables() final;

    /*!
    \brief Target/Source sorting for self contact

    */
    void target_source_sorting(int eleID, bool issource);

    /*!
    \brief Evaluate Binary search tree for self contact search

    */
    void search_self_contact(SelfBinaryTreeNode& treenode);

    /*!
    \brief Evaluate Binary search tree for contact search between separate roots
           (this is more or less identical to two-body contact search)

    */
    void search_root_contact(std::shared_ptr<SelfBinaryTreeNode> treenode1,
        std::shared_ptr<SelfBinaryTreeNode> treenode2);

    /*!
    \brief Calculate minimal element length / inflation factor "enlarge"

    */
    void set_enlarge() final;

    /*!
    \brief Update the dual graph and determine the root nodes

    \param [in] contractedEdge: contracted dual edge
    \param [in] adjEdges:       vector of all adjacent edges of the contracted edge
    \param [in] newnode:        node that represents the contracted self dual edge
    \param [in/out] dualgraph:  construction of binary tree is based on this data

    */
    void update_dual_graph(std::shared_ptr<SelfDualEdge>& contractedEdge,
        std::vector<std::shared_ptr<SelfDualEdge>>& adjEdges,
        std::shared_ptr<SelfBinaryTreeNode>& newNode,
        std::map<std::shared_ptr<SelfDualEdge>, std::vector<std::shared_ptr<SelfDualEdge>>>*
            dualgraph);

    /*!
    \brief Update normals and qualified sample vectors of the whole tree

    */
    void update_normals();

   private:
    /*!
    \brief Calculate the dual graph

    \param [out] dualgraph:  construction of binary tree is based on this data
    \param [in]  elelist:    list (gids) of all contact elements of the surface

     */
    void calculate_dual_graph(
        std::map<std::shared_ptr<SelfDualEdge>, std::vector<std::shared_ptr<SelfDualEdge>>>*
            dualGraph,
        const std::vector<int>& elelist);

    /*!
    \brief Calculate number of slabs intersections of treenode 1 and 2

    \param [in] treenode1:  self binary tree node
    \param [in] treenode2:  self binary tree node

     */
    int calculate_slabs_intercepts(SelfBinaryTreeNode& treenode1, SelfBinaryTreeNode& treenode2);

    /*!
    \brief Initialize Tree in a bottom up way based on dual graph

    \param [in] dualgraph:  construction of binary tree is based on this data

    */
    void initialize_tree_bottom_up(
        std::map<std::shared_ptr<SelfDualEdge>, std::vector<std::shared_ptr<SelfDualEdge>>>*
            dualGraph);

    /*!
    \brief Evaluate binary search tree

    \note Search and update is carried out in a separate way. There has also been a combined
    approach, but this has been removed as part of GitLab Issue 181, as it is outperformed by the
    separate approach for large self contact problems!

    */
    virtual void search_contact();

    /*!
    \brief Find contact of adjacent surfaces

    */
    void evaluate_contact_and_adjacency(std::shared_ptr<SelfBinaryTreeNode> treenode1,
        std::shared_ptr<SelfBinaryTreeNode> treenode2, bool isadjacent);

    /*!
    \brief Test for adjacency (2D)

    */
    bool test_adjacent_2d(SelfBinaryTreeNode& treenode1, SelfBinaryTreeNode& treenode2);

    /*!
    \brief Test for adjacency (2D)

    */
    bool test_adjacent_3d(std::shared_ptr<SelfBinaryTreeNode> treenode1,
        std::shared_ptr<SelfBinaryTreeNode> treenode2);
    //@}

    //! @name Print methods for debug and development purposes

    /*!
    \brief Plot the adjacency matrix

     */
    void plot_adjacency_matrix() const;

    /*!
    \brief Plot the dual graph

     */
    void plot_dual_graph(
        const std::map<std::shared_ptr<SelfDualEdge>, std::vector<std::shared_ptr<SelfDualEdge>>>&
            dualgraph) const;

    /*!
    \brief Plot root nodes and the self binary tree

     */
    void plot_roots_and_tree() const;
    //@}

    // don't want = operator and cctor
    SelfBinaryTree operator=(const SelfBinaryTree& old) = delete;
    SelfBinaryTree(const SelfBinaryTree& old) = delete;

    //! All contact elements on surface (full map)
    std::shared_ptr<Core::LinAlg::Map> elements_;

    //! Interface-specific parameter list
    const Teuchos::ParameterList& iparams_;

    /*!
    \brief Defining number of sample vectors

    \todo What are sample vectors?
    */
    int nvectors_;

    /*!
    \brief Defining sample vectors

    \todo What are sample vectors?
    */
    Core::LinAlg::SerialDenseMatrix samplevectors_;

    //! root treenodes
    std::vector<std::shared_ptr<SelfBinaryTreeNode>> roots_;

    //! storage of all treenodes, sorted by layers
    std::vector<std::vector<std::shared_ptr<SelfBinaryTreeNode>>> treenodes_;

    //! storage of all treenodes, sorted by layers
    std::map<int, std::vector<int>> contactpairs_;

    //! map of adjacent elements, sorted by global id (only needed in 3D)
    std::map<int, std::vector<std::shared_ptr<SelfBinaryTreeNode>>> adjacencymatrix_;

    //! map of all leaf nodes, sorted by global id
    std::map<int, std::shared_ptr<SelfBinaryTreeNode>> leafsmap_;

  };  // class SelfBinaryTree
}  // namespace CONTACT


FOUR_C_NAMESPACE_CLOSE

#endif
