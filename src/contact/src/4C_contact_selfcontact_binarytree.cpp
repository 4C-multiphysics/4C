// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_selfcontact_binarytree.hpp"

#include "4C_contact_element.hpp"
#include "4C_contact_node.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor BinaryTreeNode for self contact (public)             popp 11/09|
 *----------------------------------------------------------------------*/
CONTACT::SelfBinaryTreeNode::SelfBinaryTreeNode(SelfBinaryTreeNodeType type,
    Core::FE::Discretization& discret, std::shared_ptr<SelfBinaryTreeNode> parent,
    std::vector<int> elelist, const Core::LinAlg::SerialDenseMatrix& dopnormals,
    const Core::LinAlg::SerialDenseMatrix& samplevectors, const int& kdop, const int& dim,
    const int& nvectors, const int layer, const bool nonsmoothsurf,
    std::vector<std::vector<std::shared_ptr<SelfBinaryTreeNode>>>& treenodes)
    : Mortar::BaseBinaryTreeNode::BaseBinaryTreeNode(
          discret, elelist, dopnormals, kdop, dim, true, layer),
      // useauxpos_ is always true for contact problems, at least this was the case so far
      type_(type),
      parent_(parent),
      samplevectors_(samplevectors),
      nvectors_(nvectors),
      owner_(-1),
      nonsmoothsurf_(nonsmoothsurf),
      treenodes_(treenodes)
{
}

/*----------------------------------------------------------------------*
 |  get communicator (public)                                 popp 11/09|
 *----------------------------------------------------------------------*/
MPI_Comm CONTACT::SelfBinaryTreeNode::get_comm() const { return discret().get_comm(); }

/*----------------------------------------------------------------------*
 | complete the tree storage in a top down way (public)       popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::complete_tree(int layer, double& enlarge)
{
  // calculate bounding volume
  calculate_slabs_dop();
  enlarge_geometry(enlarge);

  // check root node (layer 0) for sanity
  if (layer == 0)
  {
    set_layer(0);
    if (type_ == SELFCO_INNER)
      treenodes_[layer].push_back(Core::Utils::shared_ptr_from_ref(*this));
    else
      FOUR_C_THROW("root must be inner node in treenodes scheme");
  }

  // build tree node storage recursively
  if (type_ == SELFCO_INNER)
  {
    leftchild_->set_layer(get_layer() + 1);
    rightchild_->set_layer(get_layer() + 1);

    // if map of tree nodes does not have enough rows-->resize!
    if ((int)(treenodes_.size()) <= (get_layer() + 1)) treenodes_.resize((get_layer() + 2));

    // put new pointers to children into map
    treenodes_[(get_layer() + 1)].push_back(leftchild_);
    treenodes_[(get_layer() + 1)].push_back(rightchild_);

    rightchild_->complete_tree(get_layer() + 1, enlarge);
    leftchild_->complete_tree(get_layer() + 1, enlarge);
  }

  // do nothing if arrived at leaf level
}

/*----------------------------------------------------------------------*
 | Calculate booleans for qualified sample vectors (public)   popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::calculate_qualified_vectors()
{
  if (type_ != SELFCO_LEAF) FOUR_C_THROW("Calculate qual. vec. called for non-leaf node!");

  // resize qualified vectors
  qualifiedvectors_.resize(nvectors_);

  // we first need the element center:
  // for line2, line3, quad4, quad8, quad9 elements: xi = eta = 0.0
  // for tri3, tri6 elements: xi = eta = 1/3
  Element* celement = dynamic_cast<Element*>(discret().g_element(elelist()[0]));
  double loccenter[2];

  Core::FE::CellType dt = celement->shape();
  if (dt == Core::FE::CellType::tri3 || dt == Core::FE::CellType::tri6)
  {
    loccenter[0] = 1.0 / 3.0;
    loccenter[1] = 1.0 / 3.0;
  }
  else if (dt == Core::FE::CellType::line2 || dt == Core::FE::CellType::line3 ||
           dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
           dt == Core::FE::CellType::quad9)
  {
    loccenter[0] = 0.0;
    loccenter[1] = 0.0;
  }
  else
    FOUR_C_THROW("calculate_qualified_vectors called for unknown element type");

  // now get the element center normal
  double normal[3] = {0.0, 0.0, 0.0};
  celement->compute_unit_normal_at_xi(loccenter, normal);

  // bound according to curvature criterion (cf. Semesterarbeit of Anh-Tu Vuong, 2009)
  double bound(0.0);
  // as above criterion is only valid for smooth surfaces, we adapt it for non-smooth surfaces
  // such that possible self contact is detected, if the angle between two surfaces is smaller than
  // 90 degrees
  if (nonsmoothsurf_) bound = 1.0 / sqrt(2.0);

  // check normal against sample vectors
  for (int i = 0; i < (int)qualifiedvectors_.size(); ++i)
  {
    double scalar = (double)samplevectors_(i, 0) * normal[0] +
                    (double)samplevectors_(i, 1) * normal[1] +
                    (double)samplevectors_(i, 2) * normal[2];
    if (scalar > bound)
      qualifiedvectors_[i] = true;
    else
      qualifiedvectors_[i] = false;
  }
}

/*----------------------------------------------------------------------*
 | Update qualified sample vectors for inner node (public)    popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::update_qualified_vectors_bottom_up()
{
  if (type_ == SELFCO_LEAF) FOUR_C_THROW("Update qual. vec. called for leaf node!");

  // calculate the qualified vectors (= valid sample vectors) of an inner tree node by comparing the
  // qualified vectors of the children
  qualifiedvectors_.resize(nvectors_);

  for (int i = 0; i < (int)qualifiedvectors_.size(); ++i)
    qualifiedvectors_.at(i) =
        ((rightchild_->qualified_vectors()).at(i) && (leftchild_->qualified_vectors()).at(i));
}

/*----------------------------------------------------------------------*
 | Update endnodes of one tree node (only 2D) (public)        popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::update_endnodes()
{
  if (type_ == SELFCO_LEAF) FOUR_C_THROW("Update endnodes. called for leaf node!");

  // reset endnodes
  endnodes_.clear();

  // find out which nodes the children have in common, save others as endnodes
  if (leftchild_->endnodes_[0] == rightchild_->endnodes_[0] &&
      leftchild_->endnodes_[1] != rightchild_->endnodes_[1])
  {
    endnodes_.push_back(rightchild_->endnodes_[1]);
    endnodes_.push_back(leftchild_->endnodes_[1]);
  }

  else if (leftchild_->endnodes_[1] == rightchild_->endnodes_[1] &&
           leftchild_->endnodes_[0] != rightchild_->endnodes_[0])
  {
    endnodes_.push_back(rightchild_->endnodes_[0]);
    endnodes_.push_back(leftchild_->endnodes_[0]);
  }

  else if (leftchild_->endnodes_[0] == rightchild_->endnodes_[1] &&
           leftchild_->endnodes_[1] != rightchild_->endnodes_[0])
  {
    endnodes_.push_back(rightchild_->endnodes_[0]);
    endnodes_.push_back(leftchild_->endnodes_[1]);
  }

  else if (leftchild_->endnodes_[1] == rightchild_->endnodes_[0] &&
           leftchild_->endnodes_[0] != rightchild_->endnodes_[1])
  {
    endnodes_.push_back(rightchild_->endnodes_[1]);
    endnodes_.push_back(leftchild_->endnodes_[0]);
  }

  else  // the treenode is a closed surface (ring) -> no endnodes
  {
    endnodes_.push_back(-1);
    endnodes_.push_back(-1);
  }
}

/*----------------------------------------------------------------------*
 | Update slabs bottom-up (public)                            popp 10/08|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::update_slabs_bottom_up(double& enlarge)
{
  // if current treenode is inner node
  if (type_ == SELFCO_INNER)
  {
    for (int k = 0; k < kdop() / 2; ++k)
    {
      // for minimum
      if (leftchild_->slabs()(k, 0) <= rightchild_->slabs()(k, 0))
        slabs()(k, 0) = leftchild_->slabs()(k, 0);
      else
        slabs()(k, 0) = rightchild_->slabs()(k, 0);

      // for maximum
      if (leftchild_->slabs()(k, 1) >= rightchild_->slabs()(k, 1))
        slabs()(k, 1) = leftchild_->slabs()(k, 1);
      else
        slabs()(k, 1) = rightchild_->slabs()(k, 1);
    }
  }

  // if current treenode is leaf node
  if (type_ == SELFCO_LEAF)
  {
    calculate_slabs_dop();

    enlarge_geometry(enlarge);

    // Prints slabs to std::cout
    // PrintSlabs();
  }
}

/*----------------------------------------------------------------------*
 | Print type of treenode to std::cout (public)               popp 10/08|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::print_type()
{
  if (type_ == SELFCO_INNER)
    std::cout << std::endl << "SELFCO_INNER ";
  else if (type_ == SELFCO_LEAF)
    std::cout << std::endl << "SELFCO_LEAF ";
  else if (type_ == SELFCO_NO_ELEMENTS)
    std::cout << std::endl << "TreeNode contains no elements = SELFCO_NO_ELEMENTS ";
  else
    std::cout << std::endl << "SELFCO_UNDEFINED ";
}

/*----------------------------------------------------------------------*
 | Set children of current treenode (public)                  popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::set_children(
    std::shared_ptr<SelfBinaryTreeNode> leftchild, std::shared_ptr<SelfBinaryTreeNode> rightchild)
{
  leftchild_ = leftchild;
  rightchild_ = rightchild;
}

/*----------------------------------------------------------------------*
 | Set owner of parent tree node (public)                  schmidt 01/19|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::set_parent_owner(int leftchildowner, int rightchildowner)
{
  // safety checks
  if (leftchildowner != rightchildowner)
    FOUR_C_THROW(
        "It is only allowed to combine tree nodes of the same processor to a parent node for the "
        "unbiased self contact tree implementation!");
  if (leftchildowner < 0) FOUR_C_THROW("Something went wrong! Owner can not be smaller than 0");

  owner_ = leftchildowner;
}

/*----------------------------------------------------------------------*
 | Constructor SelfDualEdge (public)                              popp 11/09|
 *----------------------------------------------------------------------*/
CONTACT::SelfDualEdge::SelfDualEdge(std::shared_ptr<SelfBinaryTreeNode> node1,
    std::shared_ptr<SelfBinaryTreeNode> node2, const int& dim)
    : node1_(node1), node2_(node2), dim_(dim)
{
  // directly move on to cost function
  calculate_costs();
}

/*----------------------------------------------------------------------*
 | Calculate cost function value (public)                     popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfDualEdge::calculate_costs()
{
  // update slabs of both dual edge nodes
  double enlarge = 0.0;
  node1_->update_slabs_bottom_up(enlarge);
  node2_->update_slabs_bottom_up(enlarge);

  // build parent slab for dual edge
  Core::LinAlg::SerialDenseMatrix parentslabs;
  const int n = node1_->kdop();
  parentslabs.reshape(n / 2, 2);

  for (int k = 0; k < n / 2; ++k)
  {
    // for minimum
    if (node1_->slabs()(k, 0) <= node2_->slabs()(k, 0))
      parentslabs(k, 0) = node1_->slabs()(k, 0);
    else
      parentslabs(k, 0) = node2_->slabs()(k, 0);

    // for maximum
    if (node1_->slabs()(k, 1) >= node2_->slabs()(k, 1))
      parentslabs(k, 1) = node1_->slabs()(k, 1);
    else
      parentslabs(k, 1) = node2_->slabs()(k, 1);
  }

  // compute maximal k-DOP length for dual edge
  double lmaxdop = 0.0;
  int slab = 0;
  for (int i = 0; i < n / 2; ++i)
  {
    double lcurrent = abs(parentslabs(i, 1) - parentslabs(i, 0));
    if (lmaxdop < lcurrent)
    {
      lmaxdop = lcurrent;
      slab = i;
    }
  }

  // two-dimensional case
  if (dim_ == 2)
  {
    // compute total length of dual edge
    double lele = 0.0;

    for (int l = 0; l < (int)(node1_->elelist().size()); ++l)
    {
      int gid = (node1_->elelist()).at(l);
      Element* celement = dynamic_cast<Element*>(node1_->discret().g_element(gid));
      lele = lele + celement->max_edge_size();
    }

    for (int l = 0; l < (int)(node2_->elelist().size()); ++l)
    {
      int gid = node2_->elelist()[l];
      Element* celement = dynamic_cast<Element*>(node2_->discret().g_element(gid));
      lele = lele + celement->max_edge_size();
    }

    // cost function = nele * ( L / L_max )
    int nele = node1_->elelist().size() + node2_->elelist().size();
    costs_ = nele * (lele / lmaxdop);
  }

  // three-dimensional case
  else
  {
    // compute total area of dual edge
    double area = 0.0;

    for (int l = 0; l < (int)(node1_->elelist().size()); ++l)
    {
      int gid = (node1_->elelist()).at(l);
      Element* celement = dynamic_cast<Element*>(node1_->discret().g_element(gid));
      area = area + celement->mo_data().area();
    }

    for (int l = 0; l < (int)(node2_->elelist().size()); ++l)
    {
      int gid = node2_->elelist()[l];
      Element* celement = dynamic_cast<Element*>(node2_->discret().g_element(gid));
      area = area + celement->mo_data().area();
    }

    // compute maximal k-DOP area for dual edge
    double lmaxdop2 = 0.0;
    Core::LinAlg::SerialDenseMatrix dopnormals = node1_->dopnormals();
    for (int j = 0; j < n / 2; ++j)
    {
      double scalar = dopnormals(j, 0) * dopnormals(slab, 0) +
                      dopnormals(j, 1) * dopnormals(slab, 1) +
                      dopnormals(j, 2) * dopnormals(slab, 2);

      if (scalar == 0)
      {
        const double lcurrent2 = abs(parentslabs(j, 1) - parentslabs(j, 0));
        if (lmaxdop2 < lcurrent2) lmaxdop2 = lcurrent2;
      }
    }
    const double doparea = lmaxdop * lmaxdop2;

    // cost function = nele * ( A / A_max )
    int nele = node1_->elelist().size() + node2_->elelist().size();
    costs_ = nele * nele * (area / doparea);
  }
}

/*----------------------------------------------------------------------*
 |  ctor SelfBinaryTree (public)                              popp 11/09|
 *----------------------------------------------------------------------*/
CONTACT::SelfBinaryTree::SelfBinaryTree(Core::FE::Discretization& discret,
    const Teuchos::ParameterList& iparams, std::shared_ptr<Core::LinAlg::Map> elements, int dim,
    double eps)
    : Mortar::BaseBinaryTree(discret, dim, eps),
      elements_(elements),
      iparams_(iparams),
      nvectors_(-1)
{
}

/*----------------------------------------------------------------------*
 | initialize the self binary tree                         schmidt 12/18|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::init()
{
  // call initialization method of the base class
  Mortar::BaseBinaryTree::init();

  // initialize internal variables
  init_internal_variables();

  // calculate min. element length and set enlargement accordingly
  set_enlarge();

  // initialize binary tree leaf nodes and create element list
  std::vector<int> elelist;
  init_leaf_nodes_and_map(elelist);

  // initialize and calculate dual graph
  std::map<std::shared_ptr<SelfDualEdge>, std::vector<std::shared_ptr<SelfDualEdge>>> dualgraph;
  calculate_dual_graph(&dualgraph, elelist);

  // plots for debug purposes
  // plot adjacency matrix
  // plot_adjacency_matrix();
  // plot dual graph
  // plot_dual_graph(dualgraph);

  // now initialize SelfBinaryTree in a bottom-up way based on dual graph
  initialize_tree_bottom_up(&dualgraph);
}


/*----------------------------------------------------------------------*
 |  Initialize internal variables (protected)              schmidt 01/19|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::init_internal_variables()
{
  // initialize sizes
  treenodes_.resize(1);
  leafsmap_.clear();

  switch (n_dim())
  {
    // two-dimensional case
    case 2:
    {
      // set number of sample vectors
      nvectors_ = 16;

      // setup sample vectors
      samplevectors_.reshape(16, 3);
      samplevectors_(0, 0) = 1.0;
      samplevectors_(0, 1) = 0.0;
      samplevectors_(0, 2) = 0.0;
      samplevectors_(8, 0) = -1.0;
      samplevectors_(8, 1) = 0.0;
      samplevectors_(8, 2) = 0.0;

      for (int i = 1; i < 4; ++i)
      {
        samplevectors_(i, 0) = cos(std::numbers::pi * (double)i / 8);
        samplevectors_(i, 1) = sin(std::numbers::pi * (double)i / 8);
        samplevectors_(i, 2) = 0;

        samplevectors_(i + 8, 0) = cos(std::numbers::pi * (double)i / 8);
        samplevectors_(i + 8, 1) = -1 * sin(std::numbers::pi * (double)i / 8);
        samplevectors_(i + 8, 2) = 0;
      }

      samplevectors_(4, 0) = 0.0;
      samplevectors_(4, 1) = 1.0;
      samplevectors_(4, 2) = 0.0;
      samplevectors_(12, 0) = 0.0;
      samplevectors_(12, 1) = -1.0;
      samplevectors_(12, 2) = 0.0;

      for (int i = 5; i < 8; ++i)
      {
        samplevectors_(i, 0) = cos(std::numbers::pi * (double)i / 8);
        samplevectors_(i, 1) = sin(std::numbers::pi * (double)i / 8);
        samplevectors_(i, 2) = 0;

        samplevectors_(i + 8, 0) = cos(std::numbers::pi * (double)i / 8);
        samplevectors_(i + 8, 1) = -1 * sin(std::numbers::pi * (double)i / 8);
        samplevectors_(i + 8, 2) = 0;
      }
      break;
    }
    // three-dimensional case
    case 3:
    {
      // set number of sample vectors
      nvectors_ = 50;

      // setup sample vectors
      samplevectors_.reshape(50, 3);
      samplevectors_(0, 0) = 0;
      samplevectors_(0, 1) = 0.0;
      samplevectors_(0, 2) = 1.0;
      samplevectors_(1, 0) = 0;
      samplevectors_(1, 1) = 0.0;
      samplevectors_(1, 2) = -1.0;

      for (int i = 0; i < 8; ++i)
      {
        for (int j = 1; j < 4; ++j)
        {
          samplevectors_(1 + 6 * i + j, 0) =
              sin(std::numbers::pi * (double)j / 8) * cos(std::numbers::pi * (double)i / 8);
          samplevectors_(1 + 6 * i + j, 1) =
              sin(std::numbers::pi * (double)j / 8) * sin(std::numbers::pi * (double)i / 8);
          samplevectors_(1 + 6 * i + j, 2) = cos(std::numbers::pi * (double)j / 8);
        }
        for (int j = 5; j < 8; ++j)
        {
          samplevectors_(6 * i + j, 0) =
              sin(std::numbers::pi * (double)j / 8) * cos(std::numbers::pi * (double)i / 8);
          samplevectors_(6 * i + j, 1) =
              sin(std::numbers::pi * (double)j / 8) * sin(std::numbers::pi * (double)i / 8);
          samplevectors_(6 * i + j, 2) = cos(std::numbers::pi * (double)j / 8);
        }
      }
      break;
    }
    // not 2D or 3D
    default:
    {
      FOUR_C_THROW("Problem dimension must be 2D or 3D!");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 |  Initialize the leaf nodes and related map (protected)  schmidt 01/19|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::init_leaf_nodes_and_map(std::vector<int>& elelist)
{
  // build global element list
  for (int i = 0; i < elements_->num_my_elements(); ++i)
  {
    int gid = elements_->gid(i);
    elelist.push_back(gid);
  }

  if (elelist.size() <= 1) FOUR_C_THROW("Less than 2 elements for binary tree initialization!");

  // check for non-smooth contact surface
  bool nonsmoothsurface(false);
  if (iparams_.get<bool>("NONSMOOTH_CONTACT_SURFACE")) nonsmoothsurface = true;

  // build local element list and create leaf nodes
  std::vector<int> localelelist;
  for (unsigned i = 0; i < elelist.size(); ++i)
  {
    localelelist.clear();
    localelelist.push_back(elelist[i]);
    std::shared_ptr<SelfBinaryTreeNode> leaf = std::make_shared<SelfBinaryTreeNode>(SELFCO_LEAF,
        discret(), nullptr, localelelist, dop_normals(), sample_vectors(), kdop(), n_dim(),
        nvectors(), -1, nonsmoothsurface, treenodes_);
    leaf->set_owner((discret().g_element(elelist[i]))->owner());
    leafsmap_[elelist[i]] = leaf;
  }

  // double-check if there is at the least one leaf node in tree now
  if (leafsmap_.size() == 0) FOUR_C_THROW("SelfBinaryTree: No contact elements defined!");
}

/*----------------------------------------------------------------------*
 |  Get number of first order nodes of element (protected) schmidt 01/19|
 *----------------------------------------------------------------------*/
int CONTACT::SelfBinaryTree::get_ele_specific_num_nodes(Core::Elements::Element* element)
{
  // find all first-order nodes of current element we exclude higher-order nodes (i.e. edge and
  // center nodes) in both 2D and 3D as they do not bring in any additional information about
  // connectivity / adjacency
  int numnode = 0;
  Mortar::Element* target_elem = dynamic_cast<Mortar::Element*>(element);

  switch (target_elem->shape())
  {
    case Core::FE::CellType::line2:
    case Core::FE::CellType::line3:
    {
      numnode = 2;
      break;
    }
    case Core::FE::CellType::tri3:
    case Core::FE::CellType::tri6:
    {
      numnode = 3;
      break;
    }
    case Core::FE::CellType::quad4:
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
    {
      numnode = 4;
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown mortar element type");
      break;
    }
  }

  return numnode;
}

/*----------------------------------------------------------------------*
 |  Get the contracted node (protected)                    schmidt 01/19|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::get_contracted_node(std::shared_ptr<SelfDualEdge>& contractedEdge,
    std::shared_ptr<SelfBinaryTreeNode>& contractedNode)
{
  std::shared_ptr<SelfBinaryTreeNode> node1 = contractedEdge->get_node1();
  std::shared_ptr<SelfBinaryTreeNode> node2 = contractedEdge->get_node2();

  // combine list of elements of both tree nodes to get new list
  std::vector<int> list = node1->elelist();
  std::vector<int> list2 = node2->elelist();
  for (unsigned i = 0; i < node2->elelist().size(); ++i) list.push_back(list2[i]);

  // define new (contracted) tree node
  contractedNode = std::make_shared<SelfBinaryTreeNode>(SELFCO_INNER, discret(), nullptr, list,
      dop_normals(), sample_vectors(), kdop(), n_dim(), nvectors(), -1, false, treenodes_);
  contractedNode->set_children(node1, node2);
  node1->set_parent(contractedNode);
  node2->set_parent(contractedNode);

  // in 2D we simply save the end nodes as adjacency criterion
  if (n_dim() == 2) contractedNode->update_endnodes();
}

/*----------------------------------------------------------------------*
 |  Calculate adjacent tree nodes & dual edges (protected) schmidt 01/19|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::calculate_adjacent_tree_nodes_and_dual_edges(
    std::vector<int>& possadjids, const int gid, Core::Elements::Element* adjElementk,
    std::shared_ptr<SelfBinaryTreeNode>& node1,
    std::vector<std::shared_ptr<SelfBinaryTreeNode>>& adjtreenodes,
    std::vector<std::shared_ptr<SelfDualEdge>>& adjdualedges)
{
  const int eleID = adjElementk->id();

  // if eleID isn't the currently considered element, it could be a neighbor
  if (eleID != gid)
  {
    // if there are not yet any possibly adjacent elements
    if (possadjids.size() == 0)
    {
      // in 2D one common node implies adjacency
      if (n_dim() == 2)
      {
        // get second node from leaves map
        std::shared_ptr<SelfBinaryTreeNode> node2 = leafsmap_[eleID];
        if (node2 == nullptr) FOUR_C_THROW("adjacent leaf tree node not found in leaves map!!");

        // get the finite element nodes of the element equal to tree node 2 and save them as end
        // nodes of the tree node
        std::vector<int> nodeIds;
        nodeIds.clear();
        nodeIds.push_back(adjElementk->node_ids()[0]);
        nodeIds.push_back(adjElementk->node_ids()[1]);
        node2->set_endnodes(nodeIds);

        // add tree node to list of adjacent tree nodes of current tree node
        adjtreenodes.push_back(node2);

        // create edge and add it to the list
        std::shared_ptr<SelfDualEdge> edge = std::make_shared<SelfDualEdge>(node1, node2, n_dim());
        adjdualedges.push_back(edge);
      }
      // in 3D adjacency is more complicated
      else
      {
        // get second node from leaves map
        std::shared_ptr<SelfBinaryTreeNode> node2 = leafsmap_[eleID];
        adjtreenodes.push_back(node2);
        possadjids.push_back(eleID);
      }
    }
    // if there are already possible adjacent elements in possadjids
    else
    {
      bool saved = false;

      // in 2D one common node implies adjacency
      if (n_dim() == 2)
      {
        // get second node from leaves map
        std::shared_ptr<SelfBinaryTreeNode> node2 = leafsmap_[eleID];
        if (node2 == nullptr) FOUR_C_THROW("adjacent tree node not found in leaves map!!");

        // get the finite element nodes of the element equal to tree node 2 and save them as end
        // nodes of the tree node
        std::vector<int> nodeIds;
        nodeIds.push_back(adjElementk->node_ids()[0]);
        nodeIds.push_back(adjElementk->node_ids()[1]);
        node2->set_endnodes(nodeIds);

        // add tree node in list of adjacent tree nodes of current tree node
        adjtreenodes.push_back(node2);

        // create edge and add it to the list
        std::shared_ptr<SelfDualEdge> edge = std::make_shared<SelfDualEdge>(node1, node2, n_dim());
        adjdualedges.push_back(edge);
      }
      // in 3D adjacency is more complicated (adjacent elements have at least 2 common nodes)
      else
      {
        // get second node from leaves map
        std::shared_ptr<SelfBinaryTreeNode> node2 = leafsmap_[eleID];

        for (unsigned l = 0; l < possadjids.size(); ++l)
        {
          // check if possible adjacent element is already in the list. If true, there are 2 common
          // nodes, which means it is a neighbor
          if (eleID == possadjids[l])
          {
            saved = true;
            if (node2 == nullptr) FOUR_C_THROW("adjacent tree node not found in leaves map!!");

            // create edge and add it to the list
            std::shared_ptr<SelfDualEdge> edge =
                std::make_shared<SelfDualEdge>(node1, node2, n_dim());
            adjdualedges.push_back(edge);
            break;
          }
        }
        // possible adjacent element is not yet in the list --> add
        if (!saved)
        {
          possadjids.push_back(eleID);
          adjtreenodes.push_back(node2);
        }
      }  // else 3D
    }  // else possadjids empty
  }  // if eleID!=gid
}

/*----------------------------------------------------------------------*
 |  Calculate the dual graph (private)                     schmidt 12/18|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::calculate_dual_graph(
    std::map<std::shared_ptr<SelfDualEdge>, std::vector<std::shared_ptr<SelfDualEdge>>>* dualGraph,
    const std::vector<int>& elelist)
{
  // loop over all self contact elements
  for (unsigned i = 0; i < elelist.size(); ++i)
  {
    // global id of current element
    const int gid = elelist[i];

    // initialize
    // ... vector of adjacent tree nodes (elements) of current element
    // ... vector of adjacent dual edges containing current element
    // ... vector of global IDs of possible adjacent elements
    std::vector<std::shared_ptr<SelfBinaryTreeNode>> adjtreenodes;
    std::vector<std::shared_ptr<SelfDualEdge>> adjdualedges;
    std::vector<int> possadjids;

    // get current elements and its nodes
    Core::Elements::Element* element = discret().g_element(gid);
    if (!element) FOUR_C_THROW("Cannot find element with gid {}", gid);
    Core::Nodes::Node** nodes = element->nodes();
    if (!nodes) FOUR_C_THROW("Null pointer!");

    // first tree node of one dual edge which includes current element is the element itself saved
    // as a tree node
    std::shared_ptr<SelfBinaryTreeNode> node1 = leafsmap_[gid];

    // for 2D only: get the finite element nodes of the element and save them as end nodes of the
    // tree node
    if (n_dim() == 2)
    {
      std::vector<int> nodeIds;
      nodeIds.push_back(element->node_ids()[0]);
      nodeIds.push_back(element->node_ids()[1]);
      node1->set_endnodes(nodeIds);
    }

    // get element specific first order nodes
    const int numnode = get_ele_specific_num_nodes(element);

    // loop over all first-order nodes of current element (here we make use of the fact that
    // first-order nodes are always stored before higher-order nodes)
    for (int j = 0; j < numnode; ++j)
    {
      Core::Nodes::Node* node = nodes[j];
      if (!node) FOUR_C_THROW("Null pointer!");

      // loop over all adjacent elements of current node
      for (auto ele : node->adjacent_elements())
      {
        calculate_adjacent_tree_nodes_and_dual_edges(
            possadjids, gid, ele.user_element(), node1, adjtreenodes, adjdualedges);
      }  // all adjacent elements
    }  // all nodes

    // add the vector of adjacent tree nodes to the adjacency matrix. We only need the matrix in 3D,
    // because in 2D the adjacency test works by comparing end nodes only
    if (n_dim() == 3) adjacencymatrix_[gid] = adjtreenodes;

    // get adjacent dual edges
    for (unsigned k = 0; k < adjdualedges.size(); ++k)
      for (unsigned j = 0; j < adjdualedges.size(); ++j)
        if (j != k) (*dualGraph)[adjdualedges[k]].push_back(adjdualedges[j]);
  }  // all elements
}  // calculate_dual_graph


/*----------------------------------------------------------------------*
 |  Calculate number of slabs intersections (private)      schmidt 12/18|
 *----------------------------------------------------------------------*/
int CONTACT::SelfBinaryTree::calculate_slabs_intercepts(
    SelfBinaryTreeNode& treenode1, SelfBinaryTreeNode& treenode2)
{
  int nintercepts = 0;

  for (int i = 0; i < kdop() / 2; ++i)
  {
    if (treenode1.slabs()(i, 0) <= treenode2.slabs()(i, 0))
    {
      if (treenode1.slabs()(i, 1) >= treenode2.slabs()(i, 0))
        nintercepts++;
      else if (treenode1.slabs()(i, 1) >= treenode2.slabs()(i, 1))
        nintercepts++;
    }
    else if (treenode1.slabs()(i, 0) >= treenode2.slabs()(i, 0))
    {
      if (treenode2.slabs()(i, 1) >= treenode1.slabs()(i, 1))
        nintercepts++;
      else if (treenode2.slabs()(i, 1) >= treenode1.slabs()(i, 0))
        nintercepts++;
    }
  }

  return nintercepts;
}



/*----------------------------------------------------------------------*
 |  Evaluate search self binary tree (public)                farah 02/16|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::evaluate_search()
{
  // calculate minimal element length
  set_enlarge();

  // update and search for contact
  search_contact();
}


/*----------------------------------------------------------------------*
 |  get communicator (protected)                              popp 11/09|
 *----------------------------------------------------------------------*/
MPI_Comm CONTACT::SelfBinaryTree::get_comm() const { return discret().get_comm(); }

/*----------------------------------------------------------------------*
 | Find minimal length of contact elements (protected)        popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::set_enlarge()
{
  // minimal length of finite elements
  double lmin = 1.0e12;

  // calculate minimal length
  for (int i = 0; i < elements_->num_my_elements(); ++i)
  {
    int gid = elements_->gid(i);
    Core::Elements::Element* element = discret().g_element(gid);
    if (!element) FOUR_C_THROW("Cannot find element with gid {}", gid);
    CONTACT::Element* celement = dynamic_cast<Element*>(element);
    double mincurrent = celement->min_edge_size();
    if (mincurrent < lmin) lmin = mincurrent;
  }

  if (lmin <= 0.0) FOUR_C_THROW("Minimal element length < 0!");

  // set the class variable
  enlarge() = eps() * lmin;
}

/*----------------------------------------------------------------------*
 | Initialize tree bottom-up based on dual graph (private)    popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::initialize_tree_bottom_up(
    std::map<std::shared_ptr<SelfDualEdge>, std::vector<std::shared_ptr<SelfDualEdge>>>* dualGraph)
{
  // vector collecting root nodes
  roots_.resize(0);

  //**********************************************************************
  // the idea is to empty the dual graph step by step
  //**********************************************************************
  while (!(*dualGraph).empty())
  {
    // get the edge with lowest costs (= the first edge in the dual graph as the map is
    // automatically sorted by the costs)  to contract it
    std::map<std::shared_ptr<SelfDualEdge>, std::vector<std::shared_ptr<SelfDualEdge>>>::iterator
        iter = (*dualGraph).begin();
    std::shared_ptr<SelfDualEdge> contractedEdge = (*iter).first;
    std::shared_ptr<SelfBinaryTreeNode> newNode(nullptr);
    get_contracted_node(contractedEdge, newNode);

    // update the dual graph
    // this means we have to create new edges, which include the new tree node and delete the edges,
    // which are adjacent to the contracted edge and update their neighbors additionally we have to
    // check if the tree is nearly complete

    // get the adjacent edges of the contracted edge
    std::vector<std::shared_ptr<SelfDualEdge>> adjEdges = (*iter).second;

    // check if the new tree node includes the whole self contact-surface in this case the tree node
    // has saved itself as adjacent edge
    if (*adjEdges[0] == *contractedEdge)
    {
      // save the tree node as root and continue the loop
      roots_.push_back(newNode);
      (*dualGraph).erase(contractedEdge);
      continue;
    }

    update_dual_graph(contractedEdge, adjEdges, newNode, dualGraph);
  }  // while(!(*dualGraph).empty())
  //**********************************************************************

  // complete the tree starting from its roots (top-down)
  if (roots_.size() == 0) FOUR_C_THROW("No root tree node found!");
  for (unsigned k = 0; k < roots_.size(); ++k) roots_[k]->complete_tree(0, enlarge());

  // output to screen
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << "\nFound " << roots_.size() << " root node(s) for self binary tree." << std::endl;

  // in 3D we have to calculate adjacent tree nodes
  if (n_dim() == 3)
  {
    calculate_adjacent_leaves();
    calculate_adjacent_tnodes();
  }
}

/*----------------------------------------------------------------------*
 | Add tree nodes to contact pairs (protected)             schmidt 01/19|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::add_tree_nodes_to_contact_pairs(
    std::shared_ptr<SelfBinaryTreeNode> treenode1, std::shared_ptr<SelfBinaryTreeNode> treenode2)
{
  bool isadjacent(true);

  if (n_dim() == 2) isadjacent = test_adjacent_2d(*treenode1, *treenode2);
  if (n_dim() == 3) isadjacent = test_adjacent_3d(treenode1, treenode2);
  if (!isadjacent)
  {
    contactpairs_[treenode1->elelist()[0]].push_back(treenode2->elelist()[0]);
    contactpairs_[treenode2->elelist()[0]].push_back(treenode1->elelist()[0]);
  }
}
/*----------------------------------------------------------------------*
 | Set adjacent treenodes of leaf-nodes in lowest layer (3D)  popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::calculate_adjacent_leaves()
{
  // get the adjacent treenodes of each treenode in the lowest layer
  // and save the adjacent leaves which are in the same layer
  int maxlayer = treenodes_.size() - 1;
  std::map<int, std::shared_ptr<SelfBinaryTreeNode>>::iterator leafiter = leafsmap_.begin();
  std::map<int, std::shared_ptr<SelfBinaryTreeNode>>::iterator leafiter_end = leafsmap_.end();

  // loop over all leaf treenodes
  while (leafiter != leafiter_end)
  {
    // do only if in lowest layer
    if (leafiter->second->get_layer() == maxlayer)
    {
      std::vector<std::shared_ptr<SelfBinaryTreeNode>> adjtnodessamelayer;
      std::vector<std::shared_ptr<SelfBinaryTreeNode>> adjtnodes =
          adjacencymatrix_[leafiter->first];

      // search for adjacent treenodes in lowest layer
      for (int i = 0; i < (int)adjtnodes.size(); ++i)
      {
        if (adjtnodes[i]->get_layer() == maxlayer) adjtnodessamelayer.push_back(adjtnodes[i]);
      }

      // store in current treenode
      leafiter->second->set_adjacent_tnodes(adjtnodessamelayer);
    }

    // increment iterator
    ++leafiter;
  }
}

/*----------------------------------------------------------------------*
 | Set adjacent treenodes of the whole tree (3D)              popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::calculate_adjacent_tnodes()
{
  // calculate adjacent treenodes in the same layer of the each treenode
  // above leaf-layer in a bottom up way, so that the adjacent treenodes of
  // the lowest layer MUST have been calculated before (see above)
  int maxlayer = treenodes_.size();

  // loop over all layers (bottom-up, starting in 2nd lowest layer)
  for (int i = maxlayer - 2; i >= 0; --i)
  {
    // loop over all treenodes of this layer
    for (int j = 0; j < (int)treenodes_.at(i).size(); j++)
    {
      // vector of adjacent treenodes
      std::vector<std::shared_ptr<SelfBinaryTreeNode>> adjtnodes;

      //******************************************************************
      // CASE 1: treenode is an inner node
      //******************************************************************
      if (treenodes_[i][j]->type() != SELFCO_LEAF)
      {
        // get the adjacent treenodes of the children
        std::vector<std::shared_ptr<SelfBinaryTreeNode>> adjofleftchild =
            treenodes_[i][j]->leftchild()->adjacent_treenodes();
        std::vector<std::shared_ptr<SelfBinaryTreeNode>> adjofrightchild =
            treenodes_[i][j]->rightchild()->adjacent_treenodes();

        // check the adjacent treenodes of the left child
        for (int k = 0; k < (int)adjofleftchild.size(); ++k)
        {
          // check if the parent of the adjacent node of the child has already been saved
          if (*adjofleftchild[k] != *treenodes_[i][j]->rightchild())
          {
            bool issaved = false;
            for (int l = 0; l < (int)adjtnodes.size(); ++l)
            {
              if (adjtnodes[l] == nullptr) FOUR_C_THROW("nullptr pointer");
              if (*adjofleftchild[k]->parent() == *adjtnodes[l])
              {
                issaved = true;
                break;
              }
            }

            if (!issaved) adjtnodes.push_back(adjofleftchild[k]->parent());
          }
        }

        // check the adjacent treenodes of the right child
        for (int k = 0; k < (int)adjofrightchild.size(); ++k)
        {
          // check if the parent of the adjacent node of the child has already been saved
          if (*adjofrightchild[k] != *treenodes_[i][j]->leftchild())
          {
            bool issaved = false;
            for (int m = 0; m < (int)adjtnodes.size(); ++m)
            {
              if (adjtnodes[m] == nullptr) FOUR_C_THROW("nullptr pointer");
              if (*adjofrightchild[k]->parent() == *adjtnodes[m])
              {
                issaved = true;
                break;
              }
            }

            if (!issaved) adjtnodes.push_back(adjofrightchild[k]->parent());
          }
        }

        // finally set adjacent treenodes of current treenode
        treenodes_[i][j]->set_adjacent_tnodes(adjtnodes);
      }

      //******************************************************************
      // CASE 2: treenode is a leaf node above the lowest layer
      //******************************************************************
      else
      {
        // get the adjacent leaf nodes from the adjacencymatrix
        int gid = treenodes_[i][j]->elelist()[0];
        if (adjacencymatrix_.find(gid) == adjacencymatrix_.end())
          FOUR_C_THROW("element not in adjacencymatrix!!");
        std::vector<std::shared_ptr<SelfBinaryTreeNode>> adjleafs = adjacencymatrix_[gid];

        // loop over all adjacent leaf nodes
        for (int n = 0; n < (int)adjleafs.size(); ++n)
        {
          // for each adjacent leaf find the parent, which is on the
          // same layer as the current treenode
          std::shared_ptr<SelfBinaryTreeNode> adjtnode = adjleafs[n];
          int diff = adjleafs[n]->get_layer() - i;

          // go through layers
          if (diff >= 0)
          {
            while (diff > 0)
            {
              adjtnode = adjtnode->parent();
              --diff;
            }

            // check if the treenode has already been saved as adjacent node
            bool issaved = false;
            for (int p = 0; p < (int)adjtnodes.size(); ++p)
            {
              if (adjtnode == nullptr) FOUR_C_THROW("nullptr vector!!");
              if (*adjtnode == *adjtnodes[p])
              {
                issaved = true;
                break;
              }
            }

            if (!issaved) adjtnodes.push_back(adjtnode);
          }
        }

        // finally set adjacent treenodes of current treenode
        treenodes_[i][j]->set_adjacent_tnodes(adjtnodes);
      }
    }  // all treenodes of current layer
  }  // all tree layers
}

/*----------------------------------------------------------------------*
 | Search for self contact (protected)                        popp 01/11|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::search_self_contact(SelfBinaryTreeNode& treenode)
{
  if (treenode.qualified_vectors().size() == 0) FOUR_C_THROW("no test vectors defined!");

  // if there is a qualified sample vector, there is no self contact
  for (int i = 0; i < (int)treenode.qualified_vectors().size(); i++)
    if (treenode.qualified_vectors()[i] == true)
    {
      return;
    }

  if (treenode.type() != SELFCO_LEAF)
  {
    search_self_contact(*treenode.leftchild());
    search_self_contact(*treenode.rightchild());
    evaluate_contact_and_adjacency(treenode.leftchild(), treenode.rightchild(), true);
  }
}

/*----------------------------------------------------------------------*
 | Search for root contact (protected)                        popp 01/11|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::search_root_contact(
    std::shared_ptr<SelfBinaryTreeNode> treenode1, std::shared_ptr<SelfBinaryTreeNode> treenode2)
{
  // check if tree nodes intercept (they only intercept if ALL slabs intersect!)
  int nintercepts = 0;

  nintercepts = calculate_slabs_intercepts(*treenode1, *treenode2);

  // tree nodes intercept
  if (nintercepts == kdop() / 2)
  {
    // both tree nodes are inner nodes
    if (treenode1->type() != SELFCO_LEAF && treenode2->type() != SELFCO_LEAF)
    {
      search_root_contact(treenode1->leftchild(), treenode2->leftchild());
      search_root_contact(treenode1->leftchild(), treenode2->rightchild());
      search_root_contact(treenode1->rightchild(), treenode2->leftchild());
      search_root_contact(treenode1->rightchild(), treenode2->rightchild());
    }

    // tree node 1 is inner, tree node 2 is leaf
    if (treenode1->type() != SELFCO_LEAF && treenode2->type() == SELFCO_LEAF)
    {
      search_root_contact(treenode1->leftchild(), treenode2);
      search_root_contact(treenode1->rightchild(), treenode2);
    }

    // tree node 1 is leaf, tree node 2 is inner
    if (treenode1->type() == SELFCO_LEAF && treenode2->type() != SELFCO_LEAF)
    {
      search_root_contact(treenode1, treenode2->leftchild());
      search_root_contact(treenode1, treenode2->rightchild());
    }

    // both tree nodes are leaf --> feasible pair
    if (treenode1->type() == SELFCO_LEAF && treenode2->type() == SELFCO_LEAF)
    {
      int gid1 = (int)treenode1->elelist()[0];  // global id of first element
      int gid2 = (int)treenode2->elelist()[0];  // global id of second element
      contactpairs_[gid1].push_back(gid2);
      contactpairs_[gid2].push_back(gid1);
    }
  }
}

/*----------------------------------------------------------------------*
 | find contact and test adjacency (public)                    popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::evaluate_contact_and_adjacency(
    std::shared_ptr<SelfBinaryTreeNode> treenode1, std::shared_ptr<SelfBinaryTreeNode> treenode2,
    bool isadjacent)
{
  // check if treenodes intercept
  // (they only intercept if ALL slabs intersect!)
  int nintercepts = 0;

  nintercepts = calculate_slabs_intercepts(*treenode1, *treenode2);

  if (nintercepts == kdop() / 2)
  {
    // teenodes intersect
    if (isadjacent)
    {
      if (n_dim() == 2)
        isadjacent = test_adjacent_2d(*treenode1, *treenode2);
      else
      {
        isadjacent = test_adjacent_3d(treenode1, treenode2);
      }

      if (isadjacent)
      {
        std::vector<bool> qualifiedvectors1 = treenode1->qualified_vectors();
        std::vector<bool> qualifiedvectors2 = treenode2->qualified_vectors();

        if ((int)qualifiedvectors1.size() == 0 or (int) qualifiedvectors2.size() == 0)
          FOUR_C_THROW("no test vectors defined!");

        if ((int)qualifiedvectors1.size() != (int)qualifiedvectors2.size())
          FOUR_C_THROW("not the same number of test vectors!");

        for (int i = 0; i < (int)qualifiedvectors1.size(); i++)
        {
          if (qualifiedvectors1[i] and qualifiedvectors2.at(i))
          {
            return;
          }
        }
      }
    }

    if ((int)treenode1->elelist().size() > (int)treenode2->elelist().size())
    {
      if (treenode1->type() != SELFCO_LEAF)
      {
        treenode1->leftchild()->calculate_slabs_dop();
        treenode1->leftchild()->enlarge_geometry(enlarge());
        treenode1->rightchild()->calculate_slabs_dop();
        treenode1->rightchild()->enlarge_geometry(enlarge());
        evaluate_contact_and_adjacency(treenode1->leftchild(), treenode2, isadjacent);
        evaluate_contact_and_adjacency(treenode1->rightchild(), treenode2, isadjacent);
      }
    }

    else
    {
      if (treenode2->type() != SELFCO_LEAF)
      {
        treenode2->leftchild()->calculate_slabs_dop();
        treenode2->leftchild()->enlarge_geometry(enlarge());
        treenode2->rightchild()->calculate_slabs_dop();
        treenode2->rightchild()->enlarge_geometry(enlarge());
        evaluate_contact_and_adjacency(treenode2->leftchild(), treenode1, isadjacent);
        evaluate_contact_and_adjacency(treenode2->rightchild(), treenode1, isadjacent);
      }
      else  // both tree nodes are leaves
        add_tree_nodes_to_contact_pairs(treenode1, treenode2);
    }
  }
  else  // dops do not intercept;
    return;
}

/*----------------------------------------------------------------------*
 | find contact and test adjacency (public)                    popp 06/09|
 *----------------------------------------------------------------------*/
bool CONTACT::SelfBinaryTree::test_adjacent_2d(
    SelfBinaryTreeNode& treenode1, SelfBinaryTreeNode& treenode2)
{
  if (n_dim() != 2) FOUR_C_THROW("test_adjacent_2d: problem must be 2D!!\n");

  std::vector<int> endnodes1 = treenode1.endnodes();
  std::vector<int> endnodes2 = treenode2.endnodes();

  if (endnodes1.size() != 2 or endnodes2.size() != 2)
    FOUR_C_THROW("treenode has not 2 endnodes!!\n");

  for (int i = 0; i < (int)endnodes1.size(); i++)
  {
    if (endnodes1[i] == -1)
    {
      // treenode is a closed surface -> has no endnodes;
      return false;
    }

    for (int j = 0; j < (int)endnodes2.size(); j++)
    {
      if (endnodes2[j] == -1)
      {
        // treenode is a closed surface -> has no endnodes;
        return false;
      }

      else if (endnodes1[i] == endnodes2[j])
      {
        // treenodes are adjacent
        return true;
      }
    }
  }
  // treenodes are not adjacent
  return false;
}

/*----------------------------------------------------------------------*
 | find contact and test adjacency (public)                    popp 06/09|
 *----------------------------------------------------------------------*/
bool CONTACT::SelfBinaryTree::test_adjacent_3d(
    std::shared_ptr<SelfBinaryTreeNode> treenode1, std::shared_ptr<SelfBinaryTreeNode> treenode2)
{
  if (n_dim() != 3) FOUR_C_THROW("test_adjacent_3d: problem must be 3D!!\n");

  // if the treenodes are in the same layer check the vector of adjacent treenodes
  if (treenode1->get_layer() == treenode2->get_layer())
  {
    std::vector<std::shared_ptr<SelfBinaryTreeNode>> adjtnodes = treenode1->adjacent_treenodes();
    for (int i = 0; i < (int)adjtnodes.size(); i++)
    {
      if (*adjtnodes[i] == *treenode2)
      {
        //   treenodes are adjacent
        return true;
      }
    }
  }

  else
  {
    // check if bounding volumes overlap
    // (they only intercept if ALL slabs intercept!)
    int nintercepts = 0;

    nintercepts = calculate_slabs_intercepts(*treenode1, *treenode2);

    // if the bounding voumes overlap
    if (nintercepts == kdop() / 2)
    {
      if (treenode1->type() == SELFCO_LEAF and treenode2->type() == SELFCO_LEAF)
      {
        // two leaves
        std::vector<std::shared_ptr<SelfBinaryTreeNode>> adjleafs =
            adjacencymatrix_[treenode1->elelist()[0]];
        for (int i = 0; i < (int)adjleafs.size(); i++)
        {
          if (*treenode2 == *adjleafs[i])
          {
            // leaves are adjacent
            return true;
          }
        }
      }

      // one leaf and one inner treenode
      else if (treenode1->type() == SELFCO_LEAF and treenode2->type() != SELFCO_LEAF)
        return (test_adjacent_3d(treenode1, treenode2->leftchild()) or
                test_adjacent_3d(treenode1, treenode2->rightchild()));

      else if (treenode1->type() != SELFCO_LEAF and treenode2->type() == SELFCO_LEAF)
        return (test_adjacent_3d(treenode1->leftchild(), treenode2) or
                test_adjacent_3d(treenode1->rightchild(), treenode2));

      else if ((treenode1->get_layer()) > treenode2->get_layer())
        return (test_adjacent_3d(treenode1, treenode2->leftchild()) or
                test_adjacent_3d(treenode1, treenode2->rightchild()));

      else if ((treenode1->get_layer()) < treenode2->get_layer())
        return (test_adjacent_3d(treenode1->leftchild(), treenode2) or
                test_adjacent_3d(treenode1->rightchild(), treenode2));
    }
  }
  // treenodes do not overlap;
  return false;
}

/*----------------------------------------------------------------------*
 | do Target/Self facet sorting (self contact) (public)        popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::target_source_sorting(int eleID, bool issource)
{
  // do as long as there are still contact pairs
  if (contactpairs_.find(eleID) != contactpairs_.end() && !contactpairs_.empty())
  {
    // set the current element to content of "issource"
    Core::Elements::Element* element = discret().g_element(eleID);
    CONTACT::Element* celement = dynamic_cast<CONTACT::Element*>(element);
    celement->set_source() = issource;

    // if the element is a source, set its node to source (otherwise the nodes
    // are target-nodes already and nodes between a target and a source element
    // should be source nodes)
    if (celement->is_source())
      for (int i = 0; i < (int)element->num_node(); i++)
      {
        Core::Nodes::Node* node = element->nodes()[i];
        CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
        cnode->set_source() = issource;
      }

    // get the ID of elements in contact with current one
    std::vector<int> contacteleID = contactpairs_[eleID];

    // erase the current element from list of contact pairs
    contactpairs_.erase(eleID);

    // loop over all contact partners of current element
    for (int i = 0; i < (int)contacteleID.size(); i++)
    {
      // add to list of search candidates if current element is source
      if (celement->is_source()) celement->add_search_elements(contacteleID[i]);

      // recursively call this function again
      if (contactpairs_.find(contacteleID[i]) != contactpairs_.end())
        target_source_sorting(contacteleID[i], !issource);
    }
  }
}

/*----------------------------------------------------------------------*
 | Update, contact search and target/source sorting            popp 01/11|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::search_contact()
{
  // check is root node available
  if ((int)roots_.size() == 0) FOUR_C_THROW("No root node for search!");
  if (roots_[0] == nullptr) FOUR_C_THROW("No root node for search!");

  // reset contact pairs from last iteration
  contactpairs_.clear();

  //**********************************************************************
  // STEP 1: update geometry (DOPs and sample vectors) bottom-up
  //**********************************************************************
  // update tree bottom up (for every tree layer)
  for (int i = ((int)(treenodes_.size() - 1)); i >= 0; --i)
  {
    for (int j = 0; j < (int)(treenodes_[i].size()); j++)
      treenodes_[i][j]->update_slabs_bottom_up(enlarge());
  }
  update_normals();

  //**********************************************************************
  // STEP 2: distribute roots among all processors
  //**********************************************************************
  // introduce some parallelization for multibody contact
  std::vector<int> myroots;
  int nproc = Core::Communication::num_mpi_ranks(get_comm());
  int nroot = (int)roots_.size();
  int ratio = nroot / nproc;
  int rest = nroot % nproc;

  // give 'ratio+1' roots to the first 'rest' procs
  if (Core::Communication::my_mpi_rank(get_comm()) < rest)
    for (int k = 0; k < ratio + 1; ++k)
      myroots.push_back(Core::Communication::my_mpi_rank(get_comm()) * (ratio + 1) + k);

  // give 'ratio' roots to the remaining procs
  else
    for (int k = 0; k < ratio; ++k)
      myroots.push_back(
          rest * (ratio + 1) + (Core::Communication::my_mpi_rank(get_comm()) - rest) * ratio + k);

  //**********************************************************************
  // STEP 3: search for self contact starting at root nodes
  //**********************************************************************
  for (int k = 0; k < (int)myroots.size(); ++k) search_self_contact(*roots_[myroots[k]]);

  //**********************************************************************
  // STEP 4: search for two-body contact between different roots
  //**********************************************************************
  for (int k = 0; k < (int)myroots.size(); ++k)
    for (int m = myroots[k] + 1; m < (int)roots_.size(); ++m)
      search_root_contact(roots_[myroots[k]], roots_[m]);

  //**********************************************************************
  // STEP 5: source and target facet sorting
  //**********************************************************************
  std::map<int, std::shared_ptr<SelfBinaryTreeNode>>::iterator leafiterNew = leafsmap_.begin();
  std::map<int, std::shared_ptr<SelfBinaryTreeNode>>::iterator leafiter = leafsmap_.begin();
  std::map<int, std::shared_ptr<SelfBinaryTreeNode>>::iterator leafiter_end = leafsmap_.end();

  // first (re)set all contact elements and nodes to target
  while (leafiter != leafiter_end)
  {
    int gid = leafiter->first;
    Core::Elements::Element* element = discret().g_element(gid);
    CONTACT::Element* celement = dynamic_cast<CONTACT::Element*>(element);

    if (celement->is_source() == true)
    {
      // reset element to target
      celement->set_source() = false;

      // reset nodes to target
      for (int i = 0; i < (int)element->num_node(); ++i)
      {
        Core::Nodes::Node* node = element->nodes()[i];
        CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
        cnode->set_source() = false;
      }
    }

    // increment iterator
    ++leafiter;
  }

  // set all non-smooth entities to source
  while (leafiterNew != leafiter_end)
  {
    int gid = leafiterNew->first;
    Core::Elements::Element* element = discret().g_element(gid);
    CONTACT::Element* celement = dynamic_cast<CONTACT::Element*>(element);

    // reset nodes to target
    for (int i = 0; i < (int)element->num_node(); ++i)
    {
      Core::Nodes::Node* node = element->nodes()[i];
      CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
      if (cnode->is_on_corner_edge())
      {
        cnode->set_source() = true;
        celement->set_source() = true;
      }
    }

    // increment iterator
    ++leafiterNew;
  }

  // make contact pairs information redundant on all procs
  std::vector<int> locdata;
  for (int i = 0; i < elements_->num_my_elements(); ++i)
  {
    int gid = elements_->gid(i);
    if (contactpairs_.find(gid) != contactpairs_.end()) locdata.push_back(gid);
  }
  Core::LinAlg::Map mymap(-1, std::span<const int>(locdata), 0, get_comm());
  std::shared_ptr<Core::LinAlg::Map> redmap = Core::LinAlg::allreduce_e_map(mymap);
  Core::Communication::Exporter ex(mymap, *redmap, get_comm());
  ex.do_export(contactpairs_);

  // now do new source and target sorting
  while (!contactpairs_.empty())
  {
    Core::Elements::Element* element = discret().g_element(contactpairs_.begin()->first);
    CONTACT::Element* celement = dynamic_cast<CONTACT::Element*>(element);
    target_source_sorting(contactpairs_.begin()->first, celement->is_source());
  }

  //**********************************************************************
  // STEP 6: check consistency of source and target facet sorting
  //**********************************************************************
  for (int i = 0; i < elements_->num_my_elements(); ++i)
  {
    int gid1 = elements_->gid(i);
    Core::Elements::Element* ele1 = discret().g_element(gid1);
    if (!ele1) FOUR_C_THROW("Cannot find element with gid %", gid1);
    Mortar::Element* element1 = dynamic_cast<Mortar::Element*>(ele1);

    // only source elements store search candidates
    if (!element1->is_source()) continue;

    // loop over the search candidates of elements1
    for (int j = 0; j < element1->mo_data().num_search_elements(); ++j)
    {
      int gid2 = element1->mo_data().search_elements()[j];
      Core::Elements::Element* ele2 = discret().g_element(gid2);
      if (!ele2) FOUR_C_THROW("Cannot find element with gid %", gid2);
      Mortar::Element* element2 = dynamic_cast<Mortar::Element*>(ele2);

      // error if this is a source element (this happens if individual self contact patches are
      // connected, because our sorting algorithm still fails in that case)
      if (element2->is_source()) FOUR_C_THROW("Source / target inconsistency in self contact");
    }
  }
}

/*----------------------------------------------------------------------*
 | Update normals and qualified sample vectors                popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::update_normals()
{
  // first update normals and sample vectors of all leaf-nodes
  std::map<int, std::shared_ptr<SelfBinaryTreeNode>>::iterator iter = leafsmap_.begin();
  std::map<int, std::shared_ptr<SelfBinaryTreeNode>>::iterator iter_end = leafsmap_.end();

  while (iter != iter_end)
  {
    iter->second->calculate_qualified_vectors();
    ++iter;
  }

  // now update the rest of the tree layer by layer in a bottom-up way
  for (int i = (int)treenodes_.size() - 1; i >= 0; --i)
  {
    for (int j = 0; j < (int)treenodes_[i].size(); ++j)
      if (treenodes_[i][j]->type() != SELFCO_LEAF)
        treenodes_[i][j]->update_qualified_vectors_bottom_up();
  }
}

/*----------------------------------------------------------------------*
 |  Update the dual graph (protected)                      schmidt 01/19|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::update_dual_graph(std::shared_ptr<SelfDualEdge>& contractedEdge,
    std::vector<std::shared_ptr<SelfDualEdge>>& adjEdges,
    std::shared_ptr<SelfBinaryTreeNode>& newNode,
    std::map<std::shared_ptr<SelfDualEdge>, std::vector<std::shared_ptr<SelfDualEdge>>>* dualgraph)
{
  // get nodes of contracted edge
  std::shared_ptr<SelfBinaryTreeNode> node1 = contractedEdge->get_node1();
  std::shared_ptr<SelfBinaryTreeNode> node2 = contractedEdge->get_node2();

  // vector of all new edges
  std::vector<std::shared_ptr<SelfDualEdge>> newAdjEdges;

  // when contracting an edge, we need to update all adjacent edges
  for (unsigned j = 0; j < adjEdges.size(); ++j)
  {
    // define new edge
    std::shared_ptr<SelfDualEdge> newEdge(nullptr);

    if (*adjEdges[j]->get_node1() != *node1 && *adjEdges[j]->get_node1() != *node2)
      newEdge = std::make_shared<SelfDualEdge>(newNode, adjEdges[j]->get_node1(), n_dim());

    else if (*adjEdges[j]->get_node2() != *node1 && *adjEdges[j]->get_node2() != *node2)
      newEdge = std::make_shared<SelfDualEdge>(newNode, adjEdges[j]->get_node2(), n_dim());

    else
      FOUR_C_THROW("Tried to contract identical tree nodes!!");

    // get the neighbors of the new edges
    std::vector<std::shared_ptr<SelfDualEdge>> adjEdgesOfNeighbor = (*dualgraph)[adjEdges[j]];

    // if there are two adjacent surfaces left, the new edge contains the whole self-contact
    // surface. in this case we save the edge as neighbor of itself
    if (adjEdges.size() == 1 && adjEdgesOfNeighbor.size() == 1)
      (*dualgraph)[newEdge].push_back(newEdge);

    // otherwise we need to update the neighbors of the adjacent edges: first add all neighbors of
    // the current adjacent edge -except the contracted edge- as neighbor of the new edge and delete
    // the old adjacent edge
    else
    {
      if (n_dim() == 2)
      {
        // in 2D every edge has 2 neighbors at the most
        newAdjEdges.push_back(newEdge);
        for (unsigned k = 0; k < adjEdgesOfNeighbor.size(); ++k)
        {
          if (*adjEdgesOfNeighbor[k] != *contractedEdge)
          {
            // save the neighbors of the old edge as neighbors of the new one
            (*dualgraph)[newEdge].push_back(adjEdgesOfNeighbor[k]);

            // we have to update all neighbors of our old edge
            std::map<std::shared_ptr<SelfDualEdge>,
                std::vector<std::shared_ptr<SelfDualEdge>>>::iterator edge_iter =
                dualgraph->find(adjEdgesOfNeighbor[k]);

            // find the old edge in the list of neighbors and replace it by the new edge
            for (unsigned a = 0; a < edge_iter->second.size(); ++a)
            {
              if (*edge_iter->second.at(a) == *adjEdges[j]) edge_iter->second.at(a) = newEdge;
            }
          }
        }
      }
      // three-dimensional case
      else
      {
        // in 3D every edge can have a arbitrary number of neighbors which could also be adjacent to
        // each other (3 edges form a "ring"). We have to check if the new edge has already been
        // created. (in 2D this occurs only when the tree is almost finished, in 3D this could
        // happen any time)

        bool issaved = false;
        for (unsigned n = 0; n < newAdjEdges.size(); ++n)
        {
          // here we make use of the customized == operator in dual edge
          if (*newAdjEdges[n] == *newEdge)
          {
            issaved = true;
            break;
          }
        }
        // save the neighbors of the old edge as neighbors of the new one
        if (!issaved) newAdjEdges.push_back(newEdge);

        // we have to update all neighbors of the old/new edge; we just update those edges which
        // aren't adjacent edges of the contracted edge, as we have to update those separately

        // get the common (tree)node of the contracted edge and the old edge
        std::shared_ptr<SelfBinaryTreeNode> commonnode = contractedEdge->common_node(*adjEdges[j]);

        // loop over all neighbors of old edge
        for (unsigned k = 0; k < adjEdgesOfNeighbor.size(); ++k)
        {
          // check if the current edge is not the contracted edge or a neighbor of the contracted
          // edge (we do not need to update these edges now)
          if (*adjEdgesOfNeighbor[k] != *contractedEdge &&
              *adjEdges[j]->common_node(*adjEdgesOfNeighbor[k]) != *commonnode)
          {
            // if the current edge (=neighbor of the new and old edge) is not a neighbor of the
            // contracted edge, they do not have a common node
            std::shared_ptr<SelfBinaryTreeNode> commonnode2 =
                contractedEdge->common_node(*adjEdgesOfNeighbor[k]);

            // now we want to save the current edge as neighbor of the new edge
            if (commonnode2 == nullptr)
            {
              // first find the new edge in the dual graph
              std::map<std::shared_ptr<SelfDualEdge>,
                  std::vector<std::shared_ptr<SelfDualEdge>>>::iterator edge_iter1 =
                  dualgraph->find(newEdge);
              std::map<std::shared_ptr<SelfDualEdge>,
                  std::vector<std::shared_ptr<SelfDualEdge>>>::iterator end = dualgraph->end();

              // if the edge has been found, check if the current edge has already been saved as
              // neighbor
              if (edge_iter1 != end)
              {
                bool edgesaved = false;
                for (unsigned z = 0; z < edge_iter1->second.size(); ++z)
                {
                  if (*edge_iter1->second.at(z) == *adjEdgesOfNeighbor[k]) edgesaved = true;
                }
                // if not yet saved, save it
                if (!edgesaved) edge_iter1->second.push_back(adjEdgesOfNeighbor[k]);
              }

              // if the new edge itself hasn't been saved yet, the neighbor hasn't been saved either
              else
                (*dualgraph)[newEdge].push_back(adjEdgesOfNeighbor[k]);

              // find the old edge in the list of neighbors and replace it by the new edge
              // find the current edge in the dual graph
              std::map<std::shared_ptr<SelfDualEdge>,
                  std::vector<std::shared_ptr<SelfDualEdge>>>::iterator edge_iter2 =
                  dualgraph->find(adjEdgesOfNeighbor[k]);

              bool egdeerased = false;
              bool newedgesaved = false;

              // loop over all neighbors of the current edge
              std::vector<std::shared_ptr<SelfDualEdge>>::iterator adjIter =
                  edge_iter2->second.begin();
              while (adjIter != edge_iter2->second.end())
              {
                if (**adjIter == *adjEdges[j])
                {
                  // erase the old edge
                  adjIter = edge_iter2->second.erase(adjIter);
                  egdeerased = true;
                }
                else
                {
                  if (**adjIter == *newEdge) newedgesaved = true;
                  ++adjIter;
                }
              }
              // as we could update the same edge several times (only in 3D), we only save the new
              // edge as neighbor if not already done so
              if (egdeerased && !newedgesaved) edge_iter2->second.push_back(newEdge);
            }
          }
        }  // loop over all adjacent edges of neighbors
      }  // 3D
    }  // else-block (not 2 adjacent tree nodes left)
  }  // loop over all adjacent edges

  // check for a ring of three dual edges in 3D (i.e. the edge to be contracted and two adjacent
  // edges) (in 2D there is no need to treat this case separately)
  if (adjEdges.size() == 2 && n_dim() == 3)
  {
    // pointers to adjacent edge nodes
    std::shared_ptr<SelfBinaryTreeNode> anode1 = adjEdges[0]->get_node1();
    std::shared_ptr<SelfBinaryTreeNode> anode2 = adjEdges[0]->get_node2();
    std::shared_ptr<SelfBinaryTreeNode> bnode1 = adjEdges[1]->get_node1();
    std::shared_ptr<SelfBinaryTreeNode> bnode2 = adjEdges[1]->get_node2();

    // check for ring (eight possible combinations)
    if ((*node1 == *anode1 && *node2 == *bnode1 && *anode2 == *bnode2) ||
        (*node1 == *anode2 && *node2 == *bnode1 && *anode1 == *bnode2) ||
        (*node1 == *anode1 && *node2 == *bnode2 && *anode2 == *bnode1) ||
        (*node1 == *anode2 && *node2 == *bnode2 && *anode1 == *bnode1) ||
        (*node1 == *bnode1 && *node2 == *anode1 && *bnode2 == *anode2) ||
        (*node1 == *bnode2 && *node2 == *anode1 && *bnode1 == *anode2) ||
        (*node1 == *bnode1 && *node2 == *anode2 && *bnode2 == *anode1) ||
        (*node1 == *bnode2 && *node2 == *anode2 && *bnode1 == *anode1))
    {
      // check for inconsistency
      if (newAdjEdges.size() != 1) FOUR_C_THROW("Inconsistent 3D ring in dual graph");

      // check if the resulting edge already exists in dual graph
      std::map<std::shared_ptr<SelfDualEdge>, std::vector<std::shared_ptr<SelfDualEdge>>>::iterator
          edge_iter3 = dualgraph->find(newAdjEdges[0]);
      std::map<std::shared_ptr<SelfDualEdge>, std::vector<std::shared_ptr<SelfDualEdge>>>::iterator
          end = dualgraph->end();

      // add if not so
      if (edge_iter3 == end) (*dualgraph)[newAdjEdges[0]].push_back(newAdjEdges[0]);
    }
  }

  // delete all adjacent edges of contracted edge
  for (unsigned j = 0; j < adjEdges.size(); ++j) dualgraph->erase(adjEdges[j]);

  // now all new adjacent edges have been created. Save all adjacent edges as neighbor, respectively
  for (unsigned l = 0; l < newAdjEdges.size(); ++l)
    for (unsigned m = 0; m < newAdjEdges.size(); ++m)
      if (l != m && *newAdjEdges[l] != *newAdjEdges[m])
        (*dualgraph)[newAdjEdges[l]].push_back(newAdjEdges[m]);

  // delete the contracted edge
  dualgraph->erase(contractedEdge);
}  // update_dual_graph

/*----------------------------------------------------------------------*
 | Plot the adjacency matrix                                  popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::plot_adjacency_matrix() const
{
  std::map<int, std::vector<std::shared_ptr<SelfBinaryTreeNode>>>::const_iterator iter2 =
      adjacencymatrix_.begin();
  std::map<int, std::vector<std::shared_ptr<SelfBinaryTreeNode>>>::const_iterator iter2_end =
      adjacencymatrix_.end();

  std::cout << "\n" << leafsmap_.size() << " elements in leaves map\n";
  std::cout << adjacencymatrix_.size() << " elements in adjacency matrix\n";

  while (iter2 != iter2_end)
  {
    std::cout << "element " << (*iter2).first << ": ";

    std::vector<std::shared_ptr<SelfBinaryTreeNode>> adj_ = (*iter2).second;
    std::cout << adj_.size() << " elements: ";
    for (unsigned i = 0; i < adj_.size(); ++i) std::cout << adj_[i]->elelist()[0] << " ";
    std::cout << "\n";
    ++iter2;
  }
}

/*----------------------------------------------------------------------*
 | Plot the dual graph                                        popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::plot_dual_graph(
    const std::map<std::shared_ptr<SelfDualEdge>, std::vector<std::shared_ptr<SelfDualEdge>>>&
        dualgraph) const
{
  std::cout << "\n" << leafsmap_.size() << " elements in leafmap\n";
  std::cout << dualgraph.size() << " edges in dual graph\n";

  std::map<std::shared_ptr<SelfDualEdge>,
      std::vector<std::shared_ptr<SelfDualEdge>>>::const_iterator iter3 = dualgraph.begin();
  std::map<std::shared_ptr<SelfDualEdge>,
      std::vector<std::shared_ptr<SelfDualEdge>>>::const_iterator iter3_end = dualgraph.end();

  std::cout << dualgraph.max_size() << " maximal\n";
  int cnt = 0;

  while (iter3 != iter3_end)
  {
    std::cout << "\n Kante " << cnt << ": " << ((*iter3).first)->get_node1()->elelist()[0] << " "
              << ((*iter3).first)->get_node2()->elelist()[0] << "\n";
    std::cout << "Kosten: " << ((*iter3).first)->costs() << "\n";

    std::vector<std::shared_ptr<SelfDualEdge>> edges = (*iter3).second;
    std::cout << edges.size() << " NachbarKanten:\n ";
    for (unsigned i = 0; i < edges.size(); ++i)
    {
      std::cout << edges[i]->get_node1()->elelist()[0] << " ";
      std::cout << edges[i]->get_node2()->elelist()[0] << " ";
      std::cout << "Kosten: " << edges[i]->costs() << " \n";
    }

    std::cout << "\n";
    ++iter3;
    ++cnt;
  }
}

/*----------------------------------------------------------------------*
 | Plot the root nodes and the self binary tree               popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::plot_roots_and_tree() const
{
  // debug output
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    // print roots
    for (unsigned k = 0; k < roots_.size(); ++k)
    {
      std::vector<int> rootElelist = roots_[k]->elelist();
      std::cout << "\nRoot " << k << " (size " << (int)rootElelist.size() << "): ";
      for (int d = 0; d < (int)rootElelist.size(); ++d) std::cout << rootElelist[d] << " ";
      std::cout << "\n";
    }

    // print tree
    for (unsigned i = 0; i < treenodes_.size(); ++i)
    {
      std::cout << "\n Tree at layer: " << i << " Elements: ";
      for (unsigned k = 0; k < treenodes_[i].size(); ++k)
      {
        std::shared_ptr<SelfBinaryTreeNode> currentnode = treenodes_[i][k];
        std::cout << " (";
        for (unsigned l = 0; l < currentnode->elelist().size(); ++l)
        {
          std::cout << currentnode->elelist().at(l) << " ";
          if (currentnode->type() == SELFCO_LEAF) std::cout << "(Leaf) ";
        }
        std::cout << ") ";
      }
    }
    std::cout << std::endl;
  }
}

FOUR_C_NAMESPACE_CLOSE
