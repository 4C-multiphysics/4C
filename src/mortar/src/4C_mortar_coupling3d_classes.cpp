// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mortar_coupling3d_classes.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mortar_coupling3d.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_integrator.hpp"
#include "4C_mortar_projector.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 03/09|
 *----------------------------------------------------------------------*/
Mortar::IntElement::IntElement(int lid, int id, int owner, Mortar::Element* parele,
    const Core::FE::CellType& shape, const int numnode, const int* nodeids,
    std::vector<Core::Nodes::Node*> nodes, const bool isslave, const bool rewind)
    : Mortar::Element(id, owner, shape, numnode, nodeids, isslave),
      lid_(lid),
      rewind_(rewind),
      parele_(parele)
{
  if ((int)nodes.size() != numnode) FOUR_C_THROW("some inconsistency");

  // check for consistency of nodeids and nodes
  // for nurbs, the nodes are not actual nodes in the
  // discretization, so just skip that part.
  if (par_shape() != Core::FE::CellType::nurbs9)
    for (int i = 0; i < numnode; ++i)
      if (nodes[i]->id() != nodeids[i])
        FOUR_C_THROW("IntElement: Inconsistency Nodes and NodeIds!");

  nodes_.clear();
  nodes_ptr_.clear();
  std::vector<int> empty_dofs(3, -2);

  for (int i = 0; i < numnode; ++i)
    nodes_.push_back(Node(nodeids[i], nodes[i]->x(), nodes[i]->owner(), empty_dofs, isslave));
  for (int i = 0; i < numnode; ++i) nodes_ptr_.push_back(&(nodes_[i]));

  if (par_shape() == Core::FE::CellType::nurbs9)
    build_nodal_pointers(nodes_ptr_.data());
  else
    build_nodal_pointers(nodes.data());


  // as discretization is already evaluated, compute area
  // (data container has to be initialized first)
  initialize_data_container();
  mo_data().area() = compute_area();

  return;
}

/*----------------------------------------------------------------------*
 |  map IntElement coords to Element coords (public)          popp 03/09|
 *----------------------------------------------------------------------*/
bool Mortar::IntElement::map_to_parent(const double* xi, double* parxi)
{
  // outdated (popp 05/2016)
  // - affine mapping is only correct for undistorted planar elements
  // - in general we need nonlinear projection procedure
  FOUR_C_THROW("MapToParent() function is outdated");

  // *********************************************************************
  // do mapping for given IntElement and Element
  // *********************************************************** quad9 ***
  if (par_shape() == Core::FE::CellType::quad9)
  {
    // do mapping according to sub-element id
    switch (lid())
    {
      case 0:
      {
        parxi[0] = 0.5 * xi[0] - 0.5;
        parxi[1] = 0.5 * xi[1] - 0.5;
        break;
      }
      case 1:
      {
        parxi[0] = 0.5 * xi[0] + 0.5;
        parxi[1] = 0.5 * xi[1] - 0.5;
        break;
      }
      case 2:
      {
        parxi[0] = 0.5 * xi[0] + 0.5;
        parxi[1] = 0.5 * xi[1] + 0.5;
        break;
      }
      case 3:
      {
        parxi[0] = 0.5 * xi[0] - 0.5;
        parxi[1] = 0.5 * xi[1] + 0.5;
        break;
      }
      default:
      {
        FOUR_C_THROW("MapToParent: Invalid local IntElement Id!");
        break;
      }
    }
  }
  // *********************************************************** quad8 ***
  else if (par_shape() == Core::FE::CellType::quad8)
  {
    // do mapping according to sub-element id
    switch (lid())
    {
      case 0:
      {
        parxi[0] = xi[0] - 1.0;
        parxi[1] = xi[1] - 1.0;
        break;
      }
      case 1:
      {
        parxi[0] = -xi[1] + 1.0;
        parxi[1] = xi[0] - 1.0;
        break;
      }
      case 2:
      {
        parxi[0] = -xi[0] + 1.0;
        parxi[1] = -xi[1] + 1.0;
        break;
      }
      case 3:
      {
        parxi[0] = xi[1] - 1.0;
        parxi[1] = -xi[0] + 1.0;
        break;
      }
      case 4:
      {
        parxi[0] = 0.5 * xi[0] - 0.5 * xi[1];
        parxi[1] = 0.5 * xi[0] + 0.5 * xi[1];
        break;
      }
      default:
      {
        FOUR_C_THROW("MapToParent: Invalid local IntElement Id!");
        break;
      }
    }
  }
  // ************************************************************ tri6 ***
  else if (par_shape() == Core::FE::CellType::tri6)
  {
    // do mapping according to sub-element id
    switch (lid())
    {
      case 0:
      {
        parxi[0] = 0.5 * xi[0];
        parxi[1] = 0.5 * xi[1];
        break;
      }
      case 1:
      {
        parxi[0] = 0.5 * xi[0] + 0.5;
        parxi[1] = 0.5 * xi[1];
        break;
      }
      case 2:
      {
        parxi[0] = 0.5 * xi[0];
        parxi[1] = 0.5 * xi[1] + 0.5;
        break;
      }
      case 3:
      {
        parxi[0] = -0.5 * xi[0] + 0.5;
        parxi[1] = -0.5 * xi[1] + 0.5;
        break;
      }
      default:
      {
        FOUR_C_THROW("MapToParent: Invalid local IntElement Id!");
        break;
      }
    }
  }
  // *********************************************************** quad4 ***
  else if (par_shape() == Core::FE::CellType::quad4)
  {
    // do mapping according to sub-element id
    switch (lid())
    {
      case 0:
      {
        parxi[0] = xi[0];
        parxi[1] = xi[1];
        break;
      }
      default:
      {
        FOUR_C_THROW("MapToParent: Invalid local IntElement Id!");
        break;
      }
    }
  }
  // ************************************************************ tri3 ***
  else if (par_shape() == Core::FE::CellType::tri3)
  {
    // do mapping according to sub-element id
    switch (lid())
    {
      case 0:
      {
        parxi[0] = xi[0];
        parxi[1] = xi[1];
        break;
      }
      default:
      {
        FOUR_C_THROW("MapToParent: Invalid local IntElement Id!");
        break;
      }
    }
  }
  // ************************************************************ nurbs9 ***
  else if (par_shape() == Core::FE::CellType::nurbs9)
  {
    if (lid() != 0) FOUR_C_THROW("nurbs9 should only have one integration element");
    // TODO: There is not necessarily a constant mapping from the IntEle
    // to the parent ele. Actually, we want to have the GP at a certain
    // spatial location, which is determined by the integration element.
    // However, for higher order Elements, the projection from the (bi-)
    // linear IntEle to a distorted higher order (e.g. NURBS) ele
    // might be more complicated. It is still to be seen, if this has
    // a notable effect.
    if (!rewind_)
    {
      parxi[0] = xi[0];
      parxi[1] = xi[1];
    }
    else
    {
      parxi[0] = xi[1];
      parxi[1] = xi[0];
    }
  }
  // ************************************************************ nurbs9 ***

  // ********************************************************* invalid ***
  else
    FOUR_C_THROW("MapToParent called for invalid parent element type!");
  // *********************************************************************

  return true;
}

/*----------------------------------------------------------------------*
 |  map IntElement coord derivatives to Element (public)      popp 03/09|
 *----------------------------------------------------------------------*/
bool Mortar::IntElement::map_to_parent(const std::vector<Core::Gen::Pairedvector<int, double>>& dxi,
    std::vector<Core::Gen::Pairedvector<int, double>>& dparxi)
{
  // outdated (popp 05/2016)
  // - affine mapping is only correct for undistorted planar elements
  // - in general we need nonlinear projection procedure
  FOUR_C_THROW("MapToParent() function is outdated");

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // *********************************************************************
  // do mapping for given IntElement and Element
  // *********************************************************** quad9 ***
  if (par_shape() == Core::FE::CellType::quad9)
  {
    // do mapping according to sub-element id
    switch (lid())
    {
      case 0:
      {
        for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p)
          dparxi[0][p->first] += 0.5 * (p->second);
        for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p)
          dparxi[1][p->first] += 0.5 * (p->second);
        break;
      }
      case 1:
      {
        for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p)
          dparxi[0][p->first] += 0.5 * (p->second);
        for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p)
          dparxi[1][p->first] += 0.5 * (p->second);
        break;
      }
      case 2:
      {
        for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p)
          dparxi[0][p->first] += 0.5 * (p->second);
        for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p)
          dparxi[1][p->first] += 0.5 * (p->second);
        break;
      }
      case 3:
      {
        for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p)
          dparxi[0][p->first] += 0.5 * (p->second);
        for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p)
          dparxi[1][p->first] += 0.5 * (p->second);
        break;
      }
      default:
      {
        FOUR_C_THROW("MapToParent: Invalid local IntElement Id!");
        break;
      }
    }
  }
  // *********************************************************** quad8 ***
  else if (par_shape() == Core::FE::CellType::quad8)
  {
    // do mapping according to sub-element id
    switch (lid())
    {
      case 0:
      {
        for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p) dparxi[0][p->first] += (p->second);
        for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p) dparxi[1][p->first] += (p->second);
        break;
      }
      case 1:
      {
        for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p) dparxi[1][p->first] += (p->second);
        for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p) dparxi[0][p->first] -= (p->second);
        break;
      }
      case 2:
      {
        for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p) dparxi[0][p->first] -= (p->second);
        for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p) dparxi[1][p->first] -= (p->second);
        break;
      }
      case 3:
      {
        for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p) dparxi[1][p->first] -= (p->second);
        for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p) dparxi[0][p->first] += (p->second);
        break;
      }
      case 4:
      {
        for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p)
        {
          dparxi[0][p->first] += 0.5 * (p->second);
          dparxi[1][p->first] += 0.5 * (p->second);
        }
        for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p)
        {
          dparxi[0][p->first] -= 0.5 * (p->second);
          dparxi[1][p->first] += 0.5 * (p->second);
        }
        break;
      }
      default:
      {
        FOUR_C_THROW("MapToParent: Invalid local IntElement Id!");
        break;
      }
    }
  }
  // ************************************************************ tri6 ***
  else if (par_shape() == Core::FE::CellType::tri6)
  {
    // do mapping according to sub-element id
    switch (lid())
    {
      case 0:
      {
        for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p)
          dparxi[0][p->first] += 0.5 * (p->second);
        for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p)
          dparxi[1][p->first] += 0.5 * (p->second);
        break;
      }
      case 1:
      {
        for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p)
          dparxi[0][p->first] += 0.5 * (p->second);
        for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p)
          dparxi[1][p->first] += 0.5 * (p->second);
        break;
      }
      case 2:
      {
        for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p)
          dparxi[0][p->first] += 0.5 * (p->second);
        for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p)
          dparxi[1][p->first] += 0.5 * (p->second);
        break;
      }
      case 3:
      {
        for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p)
          dparxi[0][p->first] -= 0.5 * (p->second);
        for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p)
          dparxi[1][p->first] -= 0.5 * (p->second);
        break;
      }
      default:
      {
        FOUR_C_THROW("MapToParent: Invalid local IntElement Id!");
        break;
      }
    }
  }
  // *********************************************************** quad4 ***
  else if (par_shape() == Core::FE::CellType::quad4)
  {
    // do mapping according to sub-element id
    switch (lid())
    {
      case 0:
      {
        for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p) dparxi[0][p->first] = (p->second);
        for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p) dparxi[1][p->first] = (p->second);
        break;
      }
      default:
      {
        FOUR_C_THROW("MapToParent: Invalid local IntElement Id!");
        break;
      }
    }
  }
  // ************************************************************ tri3 ***
  else if (par_shape() == Core::FE::CellType::tri3)
  {
    // do mapping according to sub-element id
    switch (lid())
    {
      case 0:
      {
        for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p) dparxi[0][p->first] = (p->second);
        for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p) dparxi[1][p->first] = (p->second);
        break;
      }
      default:
      {
        FOUR_C_THROW("MapToParent: Invalid local IntElement Id!");
        break;
      }
    }
  }
  // ************************************************************ nurbs9 ***
  else if (par_shape() == Core::FE::CellType::nurbs9)
  {
    if (lid() != 0) FOUR_C_THROW("nurbs9 should only have one integration element");
    if (!rewind_)
    {
      for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p) dparxi[0][p->first] = (p->second);
      for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p) dparxi[1][p->first] = (p->second);
    }
    else
    {
      for (CI p = dxi[1].begin(); p != dxi[1].end(); ++p) dparxi[0][p->first] = (p->second);
      for (CI p = dxi[0].begin(); p != dxi[0].end(); ++p) dparxi[1][p->first] = (p->second);
    }
  }
  // ************************************************************ nurbs9 ***

  // ********************************************************* invalid ***
  else
    FOUR_C_THROW("MapToParent called for invalid parent element type!");
  // *********************************************************************

  return true;
}

void Mortar::IntElement::node_linearization(
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& nodelin) const
{
  switch (parele_->shape())
  {
    // for all Lagrange Finite elements we can associate them directly with
    // the interpolatory nodes of the parent element
    case Core::FE::CellType::quad4:
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
    case Core::FE::CellType::tri3:
    case Core::FE::CellType::tri6:
    {
      // resize the linearizations
      nodelin.resize(num_node(), std::vector<Core::Gen::Pairedvector<int, double>>(3, 1));

      // loop over all intEle nodes
      for (int in = 0; in < num_node(); ++in)
      {
        const Mortar::Node* mrtrnode = dynamic_cast<const Mortar::Node*>(nodes()[in]);
        for (int dim = 0; dim < 3; ++dim) nodelin[in][dim][mrtrnode->dofs()[dim]] += 1.;
      }
      break;
    }
    case Core::FE::CellType::nurbs9:
    {
      // resize the linearizations
      nodelin.resize(num_node(),
          std::vector<Core::Gen::Pairedvector<int, double>>(3, 3 * (parele_->num_node())));

      // parameter space coords of pseudo nodes
      double pseudo_nodes_param_coords[4][2];

      if (rewind_)
      {
        pseudo_nodes_param_coords[0][0] = -1.;
        pseudo_nodes_param_coords[0][1] = -1.;
        pseudo_nodes_param_coords[1][0] = -1.;
        pseudo_nodes_param_coords[1][1] = +1.;
        pseudo_nodes_param_coords[2][0] = +1.;
        pseudo_nodes_param_coords[2][1] = +1.;
        pseudo_nodes_param_coords[3][0] = +1.;
        pseudo_nodes_param_coords[3][1] = -1.;
      }
      else
      {
        pseudo_nodes_param_coords[0][0] = -1.;
        pseudo_nodes_param_coords[0][1] = -1.;
        pseudo_nodes_param_coords[1][0] = +1.;
        pseudo_nodes_param_coords[1][1] = -1.;
        pseudo_nodes_param_coords[2][0] = +1.;
        pseudo_nodes_param_coords[2][1] = +1.;
        pseudo_nodes_param_coords[3][0] = -1.;
        pseudo_nodes_param_coords[3][1] = +1.;
      }

      // loop over all pseudo-nodes
      for (int on = 0; on < num_node(); ++on)
      {
        double xi[2] = {pseudo_nodes_param_coords[on][0], pseudo_nodes_param_coords[on][1]};

        // evaluate shape functions at pseudo node param coords
        Core::LinAlg::SerialDenseVector sval(9);
        Core::LinAlg::SerialDenseMatrix sderiv(9, 2);
        parele_->evaluate_shape(xi, sval, sderiv, 9, true);

        // loop over all parent element control points
        for (int cp = 0; cp < parele_->num_node(); ++cp)
        {
          Mortar::Node* mrtrcp = dynamic_cast<Mortar::Node*>(parele_->nodes()[cp]);

          // loop over all dimensions
          for (int dim = 0; dim < 3; ++dim) nodelin.at(on).at(dim)[mrtrcp->dofs()[dim]] += sval(cp);
        }
      }
      break;
    }
    default:
    {
      FOUR_C_THROW("unknown type of parent element shape");
      break;
    }
  }
}



/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 11/08|
 *----------------------------------------------------------------------*/
Mortar::IntCell::IntCell(int id, int nvertices, Core::LinAlg::Matrix<3, 3>& coords, double* auxn,
    const Core::FE::CellType& shape, std::vector<Core::Gen::Pairedvector<int, double>>& linv1,
    std::vector<Core::Gen::Pairedvector<int, double>>& linv2,
    std::vector<Core::Gen::Pairedvector<int, double>>& linv3,
    std::vector<Core::Gen::Pairedvector<int, double>>& linauxn)
    : id_(id), slaveId_(-1), masterId_(-1), nvertices_(nvertices), coords_(coords), shape_(shape)
{
  // store auxiliary plane normal
  for (int k = 0; k < 3; ++k) IntCell::auxn()[k] = auxn[k];

  if (shape == Core::FE::CellType::tri3)
  {
    // compute area of IntCell
    std::array<double, 3> t1 = {0.0, 0.0, 0.0};
    std::array<double, 3> t2 = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k)
    {
      t1[k] = IntCell::coords()(k, 1) - IntCell::coords()(k, 0);
      t2[k] = IntCell::coords()(k, 2) - IntCell::coords()(k, 0);
    }

    std::array<double, 3> t1xt2 = {0.0, 0.0, 0.0};
    t1xt2[0] = t1[1] * t2[2] - t1[2] * t2[1];
    t1xt2[1] = t1[2] * t2[0] - t1[0] * t2[2];
    t1xt2[2] = t1[0] * t2[1] - t1[1] * t2[0];
    area_ = 0.5 * sqrt(t1xt2[0] * t1xt2[0] + t1xt2[1] * t1xt2[1] + t1xt2[2] * t1xt2[2]);
  }
  else if (shape == Core::FE::CellType::line2)
  {
    // compute length of int_line
    std::array<double, 3> v = {0.0, 0.0, 0.0};
    v[0] = IntCell::coords()(0, 0) - IntCell::coords()(0, 1);
    v[1] = IntCell::coords()(1, 0) - IntCell::coords()(1, 1);
    v[2] = IntCell::coords()(2, 0) - IntCell::coords()(2, 1);

    area_ = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (area_ < 1e-12)
    {
      std::cout << "v0 = " << IntCell::coords()(0, 0) << "  " << IntCell::coords()(1, 0) << "  "
                << IntCell::coords()(2, 0) << std::endl;
      std::cout << "v1 = " << IntCell::coords()(0, 1) << "  " << IntCell::coords()(1, 1) << "  "
                << IntCell::coords()(2, 1) << std::endl;
      FOUR_C_THROW("INTCELL has no length!");
    }
  }

  // store vertex linearizations and auxn linearization
  linvertex_.resize(3);
  linvertex_[0] = linv1;
  linvertex_[1] = linv2;
  linvertex_[2] = linv3;  // dummy for line2
  linauxn_ = linauxn;

  return;
}


/*----------------------------------------------------------------------*
 |  Get global coords for given local coords (IntCell)        popp 11/08|
 *----------------------------------------------------------------------*/
bool Mortar::IntCell::local_to_global(const double* xi, double* globcoord, int inttype)
{
  // check input
  if (!xi) FOUR_C_THROW("local_to_global called with xi=nullptr");
  if (!globcoord) FOUR_C_THROW("local_to_global called with globcoord=nullptr");

  if (shape() == Core::FE::CellType::tri3 or shape() == Core::FE::CellType::line2)
  {
    // collect fundamental data
    Core::LinAlg::Matrix<3, 1> val;
    Core::LinAlg::Matrix<3, 2> deriv;

    // Evaluate shape, get nodal coords and interpolate global coords
    evaluate_shape(xi, val, deriv);
    for (int i = 0; i < 3; ++i) globcoord[i] = 0.0;

    for (int i = 0; i < num_vertices(); ++i)
    {
      if (inttype == 0)
      {
        // use shape function values for interpolation
        globcoord[0] += val(i) * coords()(0, i);
        globcoord[1] += val(i) * coords()(1, i);
        globcoord[2] += val(i) * coords()(2, i);
      }
      else if (inttype == 1)
      {
        // use shape function derivatives xi for interpolation
        globcoord[0] += deriv(i, 0) * coords()(0, i);
        globcoord[1] += deriv(i, 0) * coords()(1, i);
        globcoord[2] += deriv(i, 0) * coords()(2, i);
      }
      else if (inttype == 2)
      {
        if (shape() == Core::FE::CellType::line2)
          FOUR_C_THROW("for line2 elements only 1 parameter space coordinate");

        // use shape function derivatives eta for interpolation
        globcoord[0] += deriv(i, 1) * coords()(0, i);
        globcoord[1] += deriv(i, 1) * coords()(1, i);
        globcoord[2] += deriv(i, 1) * coords()(2, i);
      }
      else
        FOUR_C_THROW("Invalid interpolation type requested, only 0,1,2!");
    }
  }


  return true;
}

/*----------------------------------------------------------------------*
 |  output for integration cell                              farah 01/16|
 *----------------------------------------------------------------------*/
void Mortar::IntCell::print()
{
  std::cout << "Slave  ID= " << get_slave_id() << std::endl;
  std::cout << "Master ID= " << get_master_id() << std::endl;
  std::cout << "Coordinates for vertex 0 = " << coords()(0, 0) << " " << coords()(1, 0) << " "
            << coords()(2, 0) << std::endl;
  std::cout << "Coordinates for vertex 1 = " << coords()(0, 1) << " " << coords()(1, 1) << " "
            << coords()(2, 1) << std::endl;
  std::cout << "Coordinates for vertex 2 = " << coords()(0, 2) << " " << coords()(1, 2) << " "
            << coords()(2, 2) << std::endl;

  return;
}


/*----------------------------------------------------------------------*
 |  Evaluate shape functions (IntCell)                        popp 11/08|
 *----------------------------------------------------------------------*/
bool Mortar::IntCell::evaluate_shape(
    const double* xi, Core::LinAlg::Matrix<3, 1>& val, Core::LinAlg::Matrix<3, 2>& deriv)
{
  if (!xi) FOUR_C_THROW("evaluate_shape (IntCell) called with xi=nullptr");

  // 3noded triangular element
  if (shape() == Core::FE::CellType::tri3)
  {
    val(0) = 1.0 - xi[0] - xi[1];
    val(1) = xi[0];
    val(2) = xi[1];
    deriv(0, 0) = -1.0;
    deriv(0, 1) = -1.0;
    deriv(1, 0) = 1.0;
    deriv(1, 1) = 0.0;
    deriv(2, 0) = 0.0;
    deriv(2, 1) = 1.0;
  }
  else if (shape() == Core::FE::CellType::line2)
  {
    val(0) = 0.5 * (1 - xi[0]);
    val(1) = 0.5 * (1 + xi[0]);
    deriv(0, 0) = -0.5;
    deriv(1, 0) = 0.5;
  }

  // unknown case
  else
    FOUR_C_THROW("evaluate_shape (IntCell) called for type != tri3/line2");

  return true;
}

/*----------------------------------------------------------------------*
 |  Evaluate Jacobian determinant (IntCell)                   popp 11/08|
 *----------------------------------------------------------------------*/
double Mortar::IntCell::jacobian()
{
  double jac = 0.0;

  // 2D linear case (2noded line element)
  if (shape() == Core::FE::CellType::tri3)
    jac = area() * 2.0;
  else if (shape() == Core::FE::CellType::line2)
    jac = area() * 0.5;
  // unknown case
  else
    FOUR_C_THROW("Jacobian (IntCell) called for unknown ele type!");

  return jac;
}

/*----------------------------------------------------------------------*
 |  Evaluate directional deriv. of Jacobian det. AuxPlane     popp 03/09|
 *----------------------------------------------------------------------*/
void Mortar::IntCell::deriv_jacobian(Core::Gen::Pairedvector<int, double>& derivjac)
{
  // define iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // 1d line element
  if (shape() == Core::FE::CellType::line2)
  {
    // compute length of int_line
    std::array<double, 3> v = {0.0, 0.0, 0.0};
    v[0] = coords()(0, 0) - coords()(0, 1);
    v[1] = coords()(1, 0) - coords()(1, 1);
    v[2] = coords()(2, 0) - coords()(2, 1);

    double l = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    double linv = 1.0 / l;
    double fac = 0.25 * linv;

    // linearizarion of v
    std::vector<Core::Gen::Pairedvector<int, double>> vg(3, 1000);

    // first entry (x component lin)
    for (CI p = get_deriv_vertex(0)[0].begin(); p != get_deriv_vertex(0)[0].end(); ++p)
      vg[0][p->first] += (p->second);
    for (CI p = get_deriv_vertex(1)[0].begin(); p != get_deriv_vertex(1)[0].end(); ++p)
      vg[0][p->first] -= (p->second);

    // first entry (y component lin)
    for (CI p = get_deriv_vertex(0)[1].begin(); p != get_deriv_vertex(0)[1].end(); ++p)
      vg[1][p->first] += (p->second);
    for (CI p = get_deriv_vertex(1)[1].begin(); p != get_deriv_vertex(1)[1].end(); ++p)
      vg[1][p->first] -= (p->second);

    // first entry (z component lin)
    for (CI p = get_deriv_vertex(0)[2].begin(); p != get_deriv_vertex(0)[2].end(); ++p)
      vg[2][p->first] += (p->second);
    for (CI p = get_deriv_vertex(1)[2].begin(); p != get_deriv_vertex(1)[2].end(); ++p)
      vg[2][p->first] -= (p->second);

    // linearizarion of v^t * v
    Core::Gen::Pairedvector<int, double> vv(1000);

    // delta v^T * v
    for (CI p = vg[0].begin(); p != vg[0].end(); ++p) vv[p->first] += v[0] * (p->second);
    for (CI p = vg[1].begin(); p != vg[1].end(); ++p) vv[p->first] += v[1] * (p->second);
    for (CI p = vg[2].begin(); p != vg[2].end(); ++p) vv[p->first] += v[2] * (p->second);

    // v^T * delta v
    for (CI p = vg[0].begin(); p != vg[0].end(); ++p) vv[p->first] += v[0] * (p->second);
    for (CI p = vg[1].begin(); p != vg[1].end(); ++p) vv[p->first] += v[1] * (p->second);
    for (CI p = vg[2].begin(); p != vg[2].end(); ++p) vv[p->first] += v[2] * (p->second);

    // fac * vv
    for (CI p = vv.begin(); p != vv.end(); ++p) derivjac[p->first] += fac * (p->second);
  }
  // 2D linear case (2noded line element)
  else if (shape() == Core::FE::CellType::tri3)
  {
    // metrics routine gives local basis vectors
    static std::vector<double> gxi(3);
    static std::vector<double> geta(3);

    for (int k = 0; k < 3; ++k)
    {
      gxi[k] = coords()(k, 1) - coords()(k, 0);
      geta[k] = coords()(k, 2) - coords()(k, 0);
    }

    // cross product of gxi and geta
    std::array<double, 3> cross = {0.0, 0.0, 0.0};
    cross[0] = gxi[1] * geta[2] - gxi[2] * geta[1];
    cross[1] = gxi[2] * geta[0] - gxi[0] * geta[2];
    cross[2] = gxi[0] * geta[1] - gxi[1] * geta[0];

    // inverse jacobian
    const double jacinv =
        1.0 / sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);


    // *********************************************************************
    // compute Jacobian derivative
    // *********************************************************************
    // first vertex (Coords(k,0)) is part of gxi and geta
    for (CI p = get_deriv_vertex(0)[0].begin(); p != get_deriv_vertex(0)[0].end(); ++p)
    {
      derivjac[p->first] -= jacinv * cross[1] * gxi[2] * (p->second);
      derivjac[p->first] += jacinv * cross[1] * geta[2] * (p->second);
      derivjac[p->first] += jacinv * cross[2] * gxi[1] * (p->second);
      derivjac[p->first] -= jacinv * cross[2] * geta[1] * (p->second);
    }
    for (CI p = get_deriv_vertex(0)[1].begin(); p != get_deriv_vertex(0)[1].end(); ++p)
    {
      derivjac[p->first] += jacinv * cross[0] * gxi[2] * (p->second);
      derivjac[p->first] -= jacinv * cross[0] * geta[2] * (p->second);
      derivjac[p->first] -= jacinv * cross[2] * gxi[0] * (p->second);
      derivjac[p->first] += jacinv * cross[2] * geta[0] * (p->second);
    }
    for (CI p = get_deriv_vertex(0)[2].begin(); p != get_deriv_vertex(0)[2].end(); ++p)
    {
      derivjac[p->first] -= jacinv * cross[0] * gxi[1] * (p->second);
      derivjac[p->first] += jacinv * cross[0] * geta[1] * (p->second);
      derivjac[p->first] += jacinv * cross[1] * gxi[0] * (p->second);
      derivjac[p->first] -= jacinv * cross[1] * geta[0] * (p->second);
    }

    // second vertex (Coords(k,1)) is part of gxi
    for (CI p = get_deriv_vertex(1)[0].begin(); p != get_deriv_vertex(1)[0].end(); ++p)
    {
      derivjac[p->first] -= jacinv * cross[1] * geta[2] * (p->second);
      derivjac[p->first] += jacinv * cross[2] * geta[1] * (p->second);
    }
    for (CI p = get_deriv_vertex(1)[1].begin(); p != get_deriv_vertex(1)[1].end(); ++p)
    {
      derivjac[p->first] += jacinv * cross[0] * geta[2] * (p->second);
      derivjac[p->first] -= jacinv * cross[2] * geta[0] * (p->second);
    }
    for (CI p = get_deriv_vertex(1)[2].begin(); p != get_deriv_vertex(1)[2].end(); ++p)
    {
      derivjac[p->first] -= jacinv * cross[0] * geta[1] * (p->second);
      derivjac[p->first] += jacinv * cross[1] * geta[0] * (p->second);
    }

    // third vertex (Coords(k,2)) is part of geta
    for (CI p = get_deriv_vertex(2)[0].begin(); p != get_deriv_vertex(2)[0].end(); ++p)
    {
      derivjac[p->first] += jacinv * cross[1] * gxi[2] * (p->second);
      derivjac[p->first] -= jacinv * cross[2] * gxi[1] * (p->second);
    }
    for (CI p = get_deriv_vertex(2)[1].begin(); p != get_deriv_vertex(2)[1].end(); ++p)
    {
      derivjac[p->first] -= jacinv * cross[0] * gxi[2] * (p->second);
      derivjac[p->first] += jacinv * cross[2] * gxi[0] * (p->second);
    }
    for (CI p = get_deriv_vertex(2)[2].begin(); p != get_deriv_vertex(2)[2].end(); ++p)
    {
      derivjac[p->first] += jacinv * cross[0] * gxi[1] * (p->second);
      derivjac[p->first] -= jacinv * cross[1] * gxi[0] * (p->second);
    }
  }

  // unknown case
  else
    FOUR_C_THROW("DerivJacobian (IntCell) called for unknown ele type!");

  return;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 11/08|
 *----------------------------------------------------------------------*/
Mortar::Vertex::Vertex(
    std::vector<double> coord, Vertex::VType type, std::vector<int> nodeids, Vertex* next)
    : coord_(std::move(coord)), type_(type), nodeids_(std::move(nodeids)), next_(next)
{
  // empty constructor body
  return;
}

/*----------------------------------------------------------------------*
 |  cctor (public)                                            popp 11/08|
 *----------------------------------------------------------------------*/
Mortar::Vertex::Vertex(const Vertex& old)
    : coord_(old.coord_), type_(old.type_), nodeids_(old.nodeids_), next_(old.next_)
{
  // empty copy constructor body
  return;
}

FOUR_C_NAMESPACE_CLOSE
