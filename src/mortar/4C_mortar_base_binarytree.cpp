// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mortar_base_binarytree.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_node.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor BaseBinaryTree (public)                           schmidt 01/19|
 *----------------------------------------------------------------------*/
Mortar::BaseBinaryTree::BaseBinaryTree(Core::FE::Discretization& discret, int dim, double eps)
    : Mortar::AbstractBinaryTree::AbstractBinaryTree(),
      idiscret_(discret),
      dim_(dim),
      enlarge_(-1.0),
      eps_(eps),
      kdop_(-1)
{
  // keep the constructor clean
  return;
}

/*----------------------------------------------------------------------*
 | initialize the binary tree (public)                     schmidt 01/19|
 *----------------------------------------------------------------------*/
void Mortar::BaseBinaryTree::init()
{
  switch (dim_)
  {
    case 2:
    {
      // set number of DOP sides to 8
      kdop_ = 8;

      // setup normals for DOP
      dopnormals_.reshape(4, 3);
      dopnormals_(0, 0) = 1;
      dopnormals_(0, 1) = 0;
      dopnormals_(0, 2) = 0;
      dopnormals_(1, 0) = 0;
      dopnormals_(1, 1) = 1;
      dopnormals_(1, 2) = 0;
      dopnormals_(2, 0) = 1;
      dopnormals_(2, 1) = 1;
      dopnormals_(2, 2) = 0;
      dopnormals_(3, 0) = -1;
      dopnormals_(3, 1) = 1;
      dopnormals_(3, 2) = 0;
    }
    break;
    case 3:
    {
      // set number of DOP sides to  18
      kdop_ = 18;

      // setup normals for DOP
      dopnormals_.reshape(9, 3);
      dopnormals_(0, 0) = 1;
      dopnormals_(0, 1) = 0;
      dopnormals_(0, 2) = 0;
      dopnormals_(1, 0) = 0;
      dopnormals_(1, 1) = 1;
      dopnormals_(1, 2) = 0;
      dopnormals_(2, 0) = 0;
      dopnormals_(2, 1) = 0;
      dopnormals_(2, 2) = 1;
      dopnormals_(3, 0) = 1;
      dopnormals_(3, 1) = 1;
      dopnormals_(3, 2) = 0;
      dopnormals_(4, 0) = 1;
      dopnormals_(4, 1) = 0;
      dopnormals_(4, 2) = 1;
      dopnormals_(5, 0) = 0;
      dopnormals_(5, 1) = 1;
      dopnormals_(5, 2) = 1;
      dopnormals_(6, 0) = 1;
      dopnormals_(6, 1) = 0;
      dopnormals_(6, 2) = -1;
      dopnormals_(7, 0) = 1;
      dopnormals_(7, 1) = -1;
      dopnormals_(7, 2) = 0;
      dopnormals_(8, 0) = 0;
      dopnormals_(8, 1) = 1;
      dopnormals_(8, 2) = -1;
    }
    break;
    default:
      FOUR_C_THROW("ERROR: Problem dimension must be 2D or 3D!");
      break;
  }

  return;
}

/*----------------------------------------------------------------------*
 |  ctor BaseBinaryTreeNode (public)                       schmidt 01/19|
 *----------------------------------------------------------------------*/
Mortar::BaseBinaryTreeNode::BaseBinaryTreeNode(Core::FE::Discretization& discret,
    std::vector<int> elelist, const Core::LinAlg::SerialDenseMatrix& dopnormals, const int& kdop,
    const int& dim, const bool& useauxpos, const int layer)
    : Mortar::AbstractBinaryTreeNode::AbstractBinaryTreeNode(),
      dim_(dim),
      dopnormals_(dopnormals),
      elelist_(elelist),
      idiscret_(discret),
      kdop_(kdop),
      layer_(layer),
      useauxpos_(useauxpos)
{
  switch (dim_)
  {
    case 2:
    case 3:
    {
      slabs_.reshape(kdop_ / 2, 2);
    }
    break;
    default:
      FOUR_C_THROW("ERROR: Problem dimension must be 2D or 3D!");
      break;
  }

  return;
}

/*----------------------------------------------------------------------*
 | Calculate slabs of DOP out of current node positions       popp 10/08|
 *----------------------------------------------------------------------*/
void Mortar::BaseBinaryTreeNode::calculate_slabs_dop()
{
  // initialize slabs
  for (int j = 0; j < kdop() / 2; ++j)
  {
    slabs()(j, 0) = 1.0e12;
    slabs()(j, 1) = -1.0e12;
  }

  // calculate slabs for every element
  for (int i = 0; i < (int)elelist().size(); ++i)
  {
    int gid = elelist()[i];
    Core::Elements::Element* element = discret().g_element(gid);
    if (!element) FOUR_C_THROW("ERROR: Cannot find element with gid %\n", gid);
    Mortar::Element* mrtrelement = dynamic_cast<Mortar::Element*>(element);
    Core::Nodes::Node** nodes = mrtrelement->points();
    if (!nodes) FOUR_C_THROW("ERROR: Null pointer!");

    // calculate slabs for every node on every element
    for (int k = 0; k < mrtrelement->num_point(); ++k)
    {
      Node* mrtrnode = dynamic_cast<Node*>(nodes[k]);
      if (!mrtrnode) FOUR_C_THROW("ERROR: Null pointer!");

      // get current node position
      std::array<double, 3> pos = {0.0, 0.0, 0.0};
      for (int j = 0; j < n_dim(); ++j) pos[j] = mrtrnode->xspatial()[j];

      // calculate slabs
      for (int j = 0; j < kdop() / 2; ++j)
      {
        //= ax+by+cz=d/sqrt(aa+bb+cc)
        double num =
            dopnormals()(j, 0) * pos[0] + dopnormals()(j, 1) * pos[1] + dopnormals()(j, 2) * pos[2];
        double denom = sqrt((dopnormals()(j, 0) * dopnormals()(j, 0)) +
                            (dopnormals()(j, 1) * dopnormals()(j, 1)) +
                            (dopnormals()(j, 2) * dopnormals()(j, 2)));
        double dcurrent = num / denom;

        if (dcurrent > slabs()(j, 1)) slabs()(j, 1) = dcurrent;
        if (dcurrent < slabs()(j, 0)) slabs()(j, 0) = dcurrent;
      }

      // enlarge slabs with auxiliary position
      if (useauxpos_)
      {
        // calculate element normal at current node
        double xi[2] = {0.0, 0.0};
        double normal[3] = {0.0, 0.0, 0.0};
        mrtrelement->local_coordinates_of_node(k, xi);
        mrtrelement->compute_unit_normal_at_xi(xi, normal);

        // now the auxiliary position
        std::array<double, 3> auxpos = {0.0, 0.0, 0.0};
        double scalar = 0.0;
        for (int j = 0; j < n_dim(); ++j)
          scalar = scalar +
                   (mrtrnode->x()[j] + mrtrnode->uold()[j] - mrtrnode->xspatial()[j]) * normal[j];

        for (int j = 0; j < n_dim(); ++j) auxpos[j] = mrtrnode->xspatial()[j] + scalar * normal[j];

        for (int j = 0; j < kdop() / 2; ++j)
        {
          //= ax+by+cz=d/sqrt(aa+bb+cc)
          double num = dopnormals()(j, 0) * auxpos[0] + dopnormals()(j, 1) * auxpos[1] +
                       dopnormals()(j, 2) * auxpos[2];
          double denom = sqrt((dopnormals()(j, 0) * dopnormals()(j, 0)) +
                              (dopnormals()(j, 1) * dopnormals()(j, 1)) +
                              (dopnormals()(j, 2) * dopnormals()(j, 2)));
          double dcurrent = num / denom;

          if (dcurrent > slabs()(j, 1)) slabs()(j, 1) = dcurrent;
          if (dcurrent < slabs()(j, 0)) slabs()(j, 0) = dcurrent;
        }
      }
    }
  }
  // Prints Slabs to std::cout
  // PrintSlabs();

  return;
}

/*----------------------------------------------------------------------*
 | Enlarge geometry of treenode (public)                      popp 10/08|
 *----------------------------------------------------------------------*/
void Mortar::BaseBinaryTreeNode::enlarge_geometry(double& enlarge)
{
  // scale slabs with scalar enlarge
  for (int i = 0; i < kdop_ / 2; ++i)
  {
    slabs_(i, 0) -= enlarge;
    slabs_(i, 1) += enlarge;
  }

  return;
}

/*----------------------------------------------------------------------*
 | Print slabs to std::cout (public)                          popp 10/08|
 *----------------------------------------------------------------------*/
void Mortar::BaseBinaryTreeNode::print_slabs()
{
  std::cout << std::endl
            << Core::Communication::my_mpi_rank(discret().get_comm())
            << "************************************************************";
  print_type();
  std::cout << "slabs:";
  for (int i = 0; i < slabs_.numRows(); ++i)
    std::cout << "\nslab: " << i << " min: " << slabs_.operator()(i, 0)
              << " max: " << slabs_.operator()(i, 1);
  std::cout << "\n**********************************************************\n";

  return;
}


FOUR_C_NAMESPACE_CLOSE
