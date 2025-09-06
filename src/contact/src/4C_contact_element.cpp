// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_element.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_contact_node.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"

#include <array>

FOUR_C_NAMESPACE_OPEN
CONTACT::ElementType CONTACT::ElementType::instance_;

CONTACT::ElementType& CONTACT::ElementType::instance() { return instance_; }

Core::Communication::ParObject* CONTACT::ElementType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  CONTACT::Element* ele =
      new CONTACT::Element(0, 0, Core::FE::CellType::dis_none, 0, nullptr, false);
  ele->unpack(buffer);
  return ele;
}

std::shared_ptr<Core::Elements::Element> CONTACT::ElementType::create(const int id, const int owner)
{
  // return Teuchos::rcp( new Element( id, owner ) );
  return nullptr;
}

void CONTACT::ElementType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
}

Core::LinAlg::SerialDenseMatrix CONTACT::ElementType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  Core::LinAlg::SerialDenseMatrix nullspace;
  FOUR_C_THROW("method ComputeNullSpace not implemented!");
  return nullspace;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 10/07|
 *----------------------------------------------------------------------*/
CONTACT::Element::Element(int id, int owner, const Core::FE::CellType& shape, const int numnode,
    const int* nodeids, const bool isslave, bool isnurbs)
    : Mortar::Element(id, owner, shape, numnode, nodeids, isslave, isnurbs)
{
  // empty constructor
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                        mwgee 10/07|
 *----------------------------------------------------------------------*/
CONTACT::Element::Element(const CONTACT::Element& old) : Mortar::Element(old)
{
  // empty copy-constructor
}

/*----------------------------------------------------------------------*
 |  clone-ctor (public)                                      mwgee 10/07|
 *----------------------------------------------------------------------*/
Core::Elements::Element* CONTACT::Element::clone() const
{
  CONTACT::Element* newele = new CONTACT::Element(*this);
  return newele;
}

/*----------------------------------------------------------------------*
 |  << operator                                              mwgee 10/07|
 *----------------------------------------------------------------------*/
std::ostream& operator<<(std::ostream& os, const CONTACT::Element& element)
{
  element.print(os);
  return os;
}

/*----------------------------------------------------------------------*
 |  print element (public)                                   mwgee 10/07|
 *----------------------------------------------------------------------*/
void CONTACT::Element::print(std::ostream& os) const
{
  os << "Contact ";
  Mortar::Element::print(os);
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                           mwgee 10/07|
 *----------------------------------------------------------------------*/
void CONTACT::Element::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // add base class Mortar::Element
  Mortar::Element::pack(data);
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                           mwgee 10/07|
 *----------------------------------------------------------------------*/
void CONTACT::Element::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Mortar::Element
  Mortar::Element::unpack(buffer);
}

/*----------------------------------------------------------------------*
 |  number of dofs per node (public)                         mwgee 10/07|
 *----------------------------------------------------------------------*/
int CONTACT::Element::num_dof_per_node(const Core::Nodes::Node& node) const
{
  const CONTACT::Node* cnode = dynamic_cast<const CONTACT::Node*>(&node);
  if (!cnode) FOUR_C_THROW("Node is not a Node");
  return cnode->num_dof();
}

/*----------------------------------------------------------------------*
 |  evaluate element (public)                                mwgee 10/07|
 *----------------------------------------------------------------------*/
int CONTACT::Element::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  FOUR_C_THROW("CONTACT::Element::Evaluate not implemented!");
  return -1;
}

/*----------------------------------------------------------------------*
 |  Build element normal derivative at node                   popp 05/08|
 *----------------------------------------------------------------------*/
void CONTACT::Element::deriv_normal_at_node(int nid, int& i, Core::LinAlg::SerialDenseMatrix& elens,
    std::vector<Core::Gen::Pairedvector<int, double>>& derivn)
{
  // find this node in my list of nodes and get local numbering
  int lid = get_local_node_id(nid);

  // get local coordinates for this node
  double xi[2];
  local_coordinates_of_node(lid, xi);

  // build normal derivative at xi and return it
  deriv_normal_at_xi(xi, i, elens, derivn);
}

/*----------------------------------------------------------------------*
 |  Compute element normal derivative at loc. coord. xi       popp 09/08|
 *----------------------------------------------------------------------*/
void CONTACT::Element::deriv_normal_at_xi(double* xi, int& i,
    Core::LinAlg::SerialDenseMatrix& elens,
    std::vector<Core::Gen::Pairedvector<int, double>>& derivn)
{
  // initialize variables
  const int nnodes = num_node();
  Core::Nodes::Node** mynodes = nodes();
  if (!mynodes) FOUR_C_THROW("deriv_normal_at_xi: Null pointer!");
  Core::LinAlg::SerialDenseVector val(nnodes);
  Core::LinAlg::SerialDenseMatrix deriv(nnodes, 2, true);

  double gxi[3];
  double geta[3];

  // get shape function values and derivatives at xi
  evaluate_shape(xi, val, deriv, nnodes);

  // get local element basis vectors
  metrics(xi, gxi, geta);

  // derivative weighting matrix for current element
  Core::LinAlg::Matrix<3, 3> W;
  const double lcubeinv = 1.0 / (elens(4, i) * elens(4, i) * elens(4, i));

  for (int j = 0; j < 3; ++j)
  {
    for (int k = 0; k < 3; ++k)
    {
      W(j, k) = -lcubeinv * elens(j, i) * elens(k, i);
      if (j == k) W(j, k) += 1 / elens(4, i);
    }
  }

  // now loop over all element nodes for derivatives
  for (int n = 0; n < nnodes; ++n)
  {
    Node* mycnode = dynamic_cast<Node*>(mynodes[n]);
    if (!mycnode) FOUR_C_THROW("deriv_normal_at_xi: Null pointer!");
    int ndof = mycnode->num_dof();

    // derivative weighting matrix for current node
    static Core::LinAlg::Matrix<3, 3> F;
    F(0, 0) = 0.0;
    F(1, 1) = 0.0;
    F(2, 2) = 0.0;
    F(0, 1) = geta[2] * deriv(n, 0) - gxi[2] * deriv(n, 1);
    F(0, 2) = gxi[1] * deriv(n, 1) - geta[1] * deriv(n, 0);
    F(1, 0) = gxi[2] * deriv(n, 1) - geta[2] * deriv(n, 0);
    F(1, 2) = geta[0] * deriv(n, 0) - gxi[0] * deriv(n, 1);
    F(2, 0) = geta[1] * deriv(n, 0) - gxi[1] * deriv(n, 1);
    F(2, 1) = gxi[0] * deriv(n, 1) - geta[0] * deriv(n, 0);

    // total weighting matrix
    static Core::LinAlg::Matrix<3, 3> WF;
    WF.multiply_nn(W, F);

    // create directional derivatives
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < ndof; ++k) (derivn[j])[mycnode->dofs()[k]] += WF(j, k) * normal_fac();
  }
}

/*----------------------------------------------------------------------*
 |  Compute element normal of last time step at xi          seitz 05/17 |
 *----------------------------------------------------------------------*/
void CONTACT::Element::old_unit_normal_at_xi(
    const double* xi, Core::LinAlg::Matrix<3, 1>& n_old, Core::LinAlg::Matrix<3, 2>& d_n_old_dxi)
{
  const int nnodes = num_node();
  Core::LinAlg::SerialDenseVector val(nnodes);
  Core::LinAlg::SerialDenseMatrix deriv(nnodes, 2, true);

  // get shape function values and derivatives at xi
  evaluate_shape(xi, val, deriv, nnodes);

  n_old.clear();
  d_n_old_dxi.clear();

  Core::LinAlg::Matrix<3, 1> tmp_n;
  Core::LinAlg::Matrix<3, 2> tmp_n_deriv;
  for (int i = 0; i < nnodes; ++i)
  {
    Node* cnode = dynamic_cast<CONTACT::Node*>(nodes()[i]);
    if (!cnode) FOUR_C_THROW("this is not a FriNode!");

    for (int d = 0; d < n_dim(); ++d)
    {
      if (Core::LinAlg::Matrix<3, 1>(cnode->data().normal_old(), true).norm2() < 0.9)
        FOUR_C_THROW("where's my old normal");
      tmp_n(d) += val(i) * cnode->data().normal_old()[d];
      for (int x = 0; x < n_dim() - 1; ++x)
        tmp_n_deriv(d, x) += deriv(i, x) * cnode->data().normal_old()[d];
    }
  }
  const double l = tmp_n.norm2();
  n_old.update(1. / l, tmp_n, 0.);

  Core::LinAlg::Matrix<2, 1> dli_dxi;
  dli_dxi.multiply_tn(-1. / (l * l * l), tmp_n_deriv, tmp_n, 0.);
  d_n_old_dxi.update(1. / l, tmp_n_deriv, 0.);
  d_n_old_dxi.multiply_nt(1., tmp_n, dli_dxi, 1.);
}

/*----------------------------------------------------------------------*
 |  Evaluate derivative J,xi of Jacobian determinant          popp 05/08|
 *----------------------------------------------------------------------*/
void CONTACT::Element::d_jac_d_xi(
    double* djacdxi, double* xi, const Core::LinAlg::SerialDenseMatrix& secderiv)
{
  // the derivative dJacdXi
  djacdxi[0] = 0.0;
  djacdxi[1] = 0.0;
  Core::FE::CellType dt = shape();

  // 2D linear case (2noded line element)
  // 3D linear case (3noded triangular element)
  if (dt == Core::FE::CellType::line2 || dt == Core::FE::CellType::tri3)
  {
    // do nothing
  }

  // 2D quadratic case (3noded line element)
  else if (dt == Core::FE::CellType::line3 || dt == Core::FE::CellType::nurbs2 ||
           dt == Core::FE::CellType::nurbs3)
  {
    // get nodal coords for 2nd deriv. evaluation
    Core::LinAlg::SerialDenseMatrix coord(3, num_node());
    get_nodal_coords(coord);

    // metrics routine gives local basis vectors
    double gxi[3];
    double geta[3];
    metrics(xi, gxi, geta);

    std::array<double, 3> gsec = {0.0, 0.0, 0.0};
    for (int i = 0; i < num_node(); ++i)
      for (int k = 0; k < 3; ++k) gsec[k] += secderiv(i, 0) * coord(k, i);

    // the Jacobian itself
    const double jacinv = 1.0 / sqrt(gxi[0] * gxi[0] + gxi[1] * gxi[1] + gxi[2] * gxi[2]);

    // compute dJacdXi (1 component in 2D)
    for (int dim = 0; dim < 3; ++dim) djacdxi[0] += gxi[dim] * gsec[dim] * jacinv;
  }

  // 3D bilinear case    (4noded quadrilateral element)
  // 3D quadratic case   (6noded triangular element)
  // 3D serendipity case (8noded quadrilateral element)
  // 3D biquadratic case (9noded quadrilateral element)
  else if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::tri6 ||
           dt == Core::FE::CellType::quad8 || dt == Core::FE::CellType::quad9 ||
           dt == Core::FE::CellType::nurbs4 || dt == Core::FE::CellType::nurbs8 ||
           dt == Core::FE::CellType::nurbs9)
  {
    // get nodal coords for 2nd deriv. evaluation
    Core::LinAlg::SerialDenseMatrix coord(3, num_node());
    get_nodal_coords(coord);

    // metrics routine gives local basis vectors
    double gxi[3];
    double geta[3];
    metrics(xi, gxi, geta);

    // cross product of gxi and geta
    std::array<double, 3> cross = {0.0, 0.0, 0.0};
    cross[0] = gxi[1] * geta[2] - gxi[2] * geta[1];
    cross[1] = gxi[2] * geta[0] - gxi[0] * geta[2];
    cross[2] = gxi[0] * geta[1] - gxi[1] * geta[0];

    // the Jacobian itself
    const double jacinv =
        1.0 / sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);

    // 2nd deriv. evaluation
    Core::LinAlg::Matrix<3, 3> gsec(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < num_node(); ++i)
      for (int k = 0; k < 3; ++k)
        for (int d = 0; d < 3; ++d) gsec(k, d) += secderiv(i, d) * coord(k, i);

    // compute dJacdXi (2 components in 3D)
    djacdxi[0] += jacinv * (cross[2] * geta[1] - cross[1] * geta[2]) * gsec(0, 0);
    djacdxi[0] += jacinv * (cross[0] * geta[2] - cross[2] * geta[0]) * gsec(1, 0);
    djacdxi[0] += jacinv * (cross[1] * geta[0] - cross[0] * geta[1]) * gsec(2, 0);
    djacdxi[0] += jacinv * (cross[1] * gxi[2] - cross[2] * gxi[1]) * gsec(0, 2);
    djacdxi[0] += jacinv * (cross[2] * gxi[0] - cross[0] * gxi[2]) * gsec(1, 2);
    djacdxi[0] += jacinv * (cross[0] * gxi[1] - cross[1] * gxi[0]) * gsec(2, 2);
    djacdxi[1] += jacinv * (cross[2] * geta[1] - cross[1] * geta[2]) * gsec(0, 2);
    djacdxi[1] += jacinv * (cross[0] * geta[2] - cross[2] * geta[0]) * gsec(1, 2);
    djacdxi[1] += jacinv * (cross[1] * geta[0] - cross[0] * geta[1]) * gsec(2, 2);
    djacdxi[1] += jacinv * (cross[1] * gxi[2] - cross[2] * gxi[1]) * gsec(0, 1);
    djacdxi[1] += jacinv * (cross[2] * gxi[0] - cross[0] * gxi[2]) * gsec(1, 1);
    djacdxi[1] += jacinv * (cross[0] * gxi[1] - cross[1] * gxi[0]) * gsec(2, 1);
  }

  // unknown case
  else
    FOUR_C_THROW("d_jac_d_xi called for unknown element type!");
}


void CONTACT::Element::prepare_dderiv(const std::vector<Mortar::Element*>& meles)
{
  // number of dofs that may appear in the linearization
  int numderiv = 0;
  numderiv += num_node() * 3 * 12;
  for (unsigned m = 0; m < meles.size(); ++m) numderiv += (meles.at(m))->num_node() * 3;
  d_matrix_deriv_ = std::make_shared<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>(
      numderiv, 0, Core::LinAlg::SerialDenseMatrix(num_node(), num_node()));
}

void CONTACT::Element::prepare_mderiv(const std::vector<Mortar::Element*>& meles, const int m)
{
  // number of dofs that may appear in the linearization
  int numderiv = 0;
  numderiv += num_node() * 3 * 12;
  for (unsigned i = 0; i < meles.size(); ++i) numderiv += meles[i]->num_node() * 3;
  m_matrix_deriv_ = std::make_shared<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>(
      numderiv, 0, Core::LinAlg::SerialDenseMatrix(num_node(), meles.at(m)->num_node()));
}


void CONTACT::Element::assemble_dderiv_to_nodes(bool dual)
{
  if (d_matrix_deriv_ == nullptr)
    FOUR_C_THROW("assemble_dderiv_to_nodes called w/o prepare_dderiv first");

  if (d_matrix_deriv_->size() == 0) return;

  for (int j = 0; j < num_node(); ++j)
  {
    CONTACT::Node* cnode_j = dynamic_cast<CONTACT::Node*>(nodes()[j]);

    if (!dual)
    {
      for (int k = 0; k < num_node(); ++k)
      {
        CONTACT::Node* cnode_k = dynamic_cast<CONTACT::Node*>(nodes()[k]);
        std::map<int, double>& ddmap_jk = cnode_j->data().get_deriv_d()[cnode_k->id()];

        for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
                 d_matrix_deriv_->begin();
            p != d_matrix_deriv_->end(); ++p)
          ddmap_jk[p->first] += (p->second)(j, k);
      }
    }
    else
    {
      std::map<int, double>& ddmap_jj = cnode_j->data().get_deriv_d()[cnode_j->id()];

      for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
               d_matrix_deriv_->begin();
          p != d_matrix_deriv_->end(); ++p)
        ddmap_jj[p->first] += (p->second)(j, j);
    }
  }
  d_matrix_deriv_ = nullptr;
}

void CONTACT::Element::assemble_mderiv_to_nodes(Mortar::Element& mele)
{
  if (m_matrix_deriv_ == nullptr)
    FOUR_C_THROW("assemble_mderiv_to_nodes called w/o prepare_mderiv first");
  if (m_matrix_deriv_->size() == 0) return;

  for (int j = 0; j < num_node(); ++j)
  {
    CONTACT::Node* cnode_j = dynamic_cast<CONTACT::Node*>(nodes()[j]);

    for (int k = 0; k < mele.num_node(); ++k)
    {
      CONTACT::Node* cnode_k = dynamic_cast<CONTACT::Node*>(mele.nodes()[k]);
      std::map<int, double>& dmmap_jk = cnode_j->data().get_deriv_m()[cnode_k->id()];

      for (Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>::const_iterator p =
               m_matrix_deriv_->begin();
          p != m_matrix_deriv_->end(); ++p)
        dmmap_jk[p->first] += (p->second)(j, k);
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
