// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mortar_element.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_contact_nitsche_utils.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_material_base.hpp"
#include "4C_mortar_node.hpp"
#include "4C_solid_3D_ele_surface.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN



Mortar::ElementType Mortar::ElementType::instance_;

Mortar::ElementType& Mortar::ElementType::instance() { return instance_; }

Core::Communication::ParObject* Mortar::ElementType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Mortar::Element* ele = new Mortar::Element(0, 0, Core::FE::CellType::dis_none, 0, nullptr, false);
  ele->unpack(buffer);
  return ele;
}


std::shared_ptr<Core::Elements::Element> Mortar::ElementType::create(const int id, const int owner)
{
  // return Teuchos::rcp( new Mortar::Element( id, owner ) );
  return nullptr;
}


void Mortar::ElementType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
}

Core::LinAlg::SerialDenseMatrix Mortar::ElementType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  Core::LinAlg::SerialDenseMatrix nullspace;
  FOUR_C_THROW("method ComputeNullSpace not implemented!");
  return nullspace;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 12/10|
 *----------------------------------------------------------------------*/
Mortar::MortarEleDataContainer::MortarEleDataContainer()
{
  // initialize area
  area() = 0.0;
  dualshapecoeff_ = nullptr;
  derivdualshapecoeff_ = nullptr;
  trafocoeff_ = nullptr;

  return;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                            popp 12/10|
 *----------------------------------------------------------------------*/
void Mortar::MortarEleDataContainer::pack(Core::Communication::PackBuffer& data) const
{
  // add area_
  add_to_pack(data, area_);

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                            mgit 02/10|
 *----------------------------------------------------------------------*/
void Mortar::MortarEleDataContainer::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // area_
  extract_from_pack(buffer, area_);

  dualshapecoeff_ = nullptr;
  derivdualshapecoeff_ = nullptr;
  return;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 10/07|
 *----------------------------------------------------------------------*/
Mortar::Element::Element(int id, int owner, const Core::FE::CellType& shape, const int numnode,
    const int* nodeids, const bool isslave, bool isnurbs)
    : Core::Elements::FaceElement(id, owner),
      shape_(shape),
      isslave_(isslave),
      attached_(false),
      nurbs_(isnurbs),
      normalfac_(1.0),    // normal factor for nurbs
      zero_sized_(false)  // information for nurbs integration
{
  set_node_ids(numnode, nodeids);
  return;
}

/*----------------------------------------------------------------------*
 |  ctor (protected)                                   kronbichler 03/15|
 *----------------------------------------------------------------------*/
Mortar::Element::Element(int id, int owner)
    : Core::Elements::FaceElement(id, owner),
      shape_(Core::FE::CellType::dis_none),
      isslave_(false),
      attached_(false),
      nurbs_(false),
      normalfac_(1.0),    // normal factor for nurbs
      zero_sized_(false)  // information for nurbs integration
{
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       mwgee 10/07|
 *----------------------------------------------------------------------*/
Mortar::Element::Element(const Mortar::Element& old)
    : Core::Elements::FaceElement(old), shape_(old.shape_), isslave_(old.isslave_)
{
  // not yet used and thus not necessarily consistent
  FOUR_C_THROW("Mortar::Element copy-ctor not yet implemented");

  return;
}

/*----------------------------------------------------------------------*
 |  clone-ctor (public)                                      mwgee 10/07|
 *----------------------------------------------------------------------*/
Core::Elements::Element* Mortar::Element::clone() const
{
  Mortar::Element* newele = new Mortar::Element(*this);
  return newele;
}


/*----------------------------------------------------------------------*
 |  << operator                                              mwgee 10/07|
 *----------------------------------------------------------------------*/
std::ostream& operator<<(std::ostream& os, const Mortar::Element& element)
{
  element.print(os);
  return os;
}


/*----------------------------------------------------------------------*
 |  print element (public)                                   mwgee 10/07|
 *----------------------------------------------------------------------*/
void Mortar::Element::print(std::ostream& os) const
{
  os << "Mortar Element ";
  Core::Elements::Element::print(os);
  if (isslave_)
    os << " Slave  ";
  else
    os << " Master ";

  return;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                           mwgee 10/07|
 *----------------------------------------------------------------------*/
void Mortar::Element::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Core::Elements::FaceElement
  Core::Elements::FaceElement::pack(data);
  // add shape_
  add_to_pack(data, shape_);
  // add isslave_
  add_to_pack(data, isslave_);
  // add nurbs_
  add_to_pack(data, nurbs_);

  // for nurbs:
  if (nurbs_)
  {
    // add normalfac
    add_to_pack(data, normalfac_);
    // add zero_sized_
    add_to_pack(data, zero_sized_);
    // knots
    int nr = mortarknots_.size();
    add_to_pack(data, nr);
    if (nr != 0)
    {
      for (int i = 0; i < nr; i++) add_to_pack(data, (mortarknots_[i]));
    }
  }

  // add modata_
  bool hasdata = (modata_ != nullptr);
  add_to_pack(data, hasdata);
  if (hasdata) modata_->pack(data);

  // add physicaltype
  add_to_pack(data, physicaltype_);

  // mesh size
  add_to_pack(data, traceHE_);
  add_to_pack(data, traceHCond_);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                           mwgee 10/07|
 *----------------------------------------------------------------------*/
void Mortar::Element::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Core::Elements::FaceElement
  Core::Elements::FaceElement::unpack(buffer);
  // shape_
  extract_from_pack(buffer, shape_);
  // isslave_
  extract_from_pack(buffer, isslave_);
  // nurbs_
  extract_from_pack(buffer, nurbs_);

  // for nurbs:
  if (nurbs_)
  {
    // normalfac_
    extract_from_pack(buffer, normalfac_);
    // zero_sized_
    extract_from_pack(buffer, zero_sized_);
    // knots
    int nr;
    extract_from_pack(buffer, nr);

    if (nr != 0)
    {
      mortarknots_.resize(nr);
      for (int i = 0; i < nr; i++) extract_from_pack(buffer, mortarknots_[i]);
    }
  }

  // modata_
  bool hasdata;
  extract_from_pack(buffer, hasdata);
  if (hasdata)
  {
    modata_ = std::make_shared<Mortar::MortarEleDataContainer>();
    modata_->unpack(buffer);
  }
  else
  {
    modata_ = nullptr;
  }

  // physical type
  extract_from_pack(buffer, physicaltype_);

  // mesh size
  extract_from_pack(buffer, traceHE_);
  extract_from_pack(buffer, traceHCond_);
}

/*----------------------------------------------------------------------*
 |  number of dofs per node (public)                         mwgee 10/07|
 *----------------------------------------------------------------------*/
int Mortar::Element::num_dof_per_node(const Core::Nodes::Node& node) const
{
  const Mortar::Node* mnode = dynamic_cast<const Mortar::Node*>(&node);
  if (!mnode) FOUR_C_THROW("Node is not a Node");
  return mnode->num_dof();
}

/*----------------------------------------------------------------------*
 |  evaluate element (public)                                mwgee 10/07|
 *----------------------------------------------------------------------*/
int Mortar::Element::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  FOUR_C_THROW("Mortar::Element::Evaluate not implemented!");
  return -1;
}

/*----------------------------------------------------------------------*
 |  Get local coordinates for local node id                   popp 12/07|
 *----------------------------------------------------------------------*/
bool Mortar::Element::local_coordinates_of_node(int lid, double* xi) const
{
  // 2D linear case (2noded line element)
  // 2D quadratic case (3noded line element)
  switch (shape())
  {
    case Core::FE::CellType::line2:
    case Core::FE::CellType::line3:
    {
      switch (lid)
      {
        case 0:
          xi[0] = -1.0;
          break;
        case 1:
          xi[0] = 1.0;
          break;
        case 2:
          xi[0] = 0.0;
          break;
        default:
          FOUR_C_THROW(
              "ERROR: local_coordinates_of_node: Node number {} in segment {} out of range", lid,
              id());
      }
      // we are in the 2D case here!
      xi[1] = 0.0;

      break;
    }

    // 3D linear case (2noded triangular element)
    // 3D quadratic case (3noded triangular element)
    case Core::FE::CellType::tri3:
    case Core::FE::CellType::tri6:
    {
      switch (lid)
      {
        case 0:
        {
          xi[0] = 0.0;
          xi[1] = 0.0;
          break;
        }
        case 1:
        {
          xi[0] = 1.0;
          xi[1] = 0.0;
          break;
        }
        case 2:
        {
          xi[0] = 0.0;
          xi[1] = 1.0;
          break;
        }
        case 3:
        {
          xi[0] = 0.5;
          xi[1] = 0.0;
          break;
        }
        case 4:
        {
          xi[0] = 0.5;
          xi[1] = 0.5;
          break;
        }
        case 5:
        {
          xi[0] = 0.0;
          xi[1] = 0.5;
          break;
        }
        default:
          FOUR_C_THROW("LocCoordsOfNode: Node number {} in segment {} out of range", lid, id());
      }

      break;
    }

    // 3D bilinear case (4noded quadrilateral element)
    // 3D serendipity case (8noded quadrilateral element)
    // 3D biquadratic case (9noded quadrilateral element)
    case Core::FE::CellType::quad4:
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
    {
      switch (lid)
      {
        case 0:
        {
          xi[0] = -1.0;
          xi[1] = -1.0;
          break;
        }
        case 1:
        {
          xi[0] = 1.0;
          xi[1] = -1.0;
          break;
        }
        case 2:
        {
          xi[0] = 1.0;
          xi[1] = 1.0;
          break;
        }
        case 3:
        {
          xi[0] = -1.0;
          xi[1] = 1.0;
          break;
        }
        case 4:
        {
          xi[0] = 0.0;
          xi[1] = -1.0;
          break;
        }
        case 5:
        {
          xi[0] = 1.0;
          xi[1] = 0.0;
          break;
        }
        case 6:
        {
          xi[0] = 0.0;
          xi[1] = 1.0;
          break;
        }
        case 7:
        {
          xi[0] = -1.0;
          xi[1] = 0.0;
          break;
        }
        case 8:
        {
          xi[0] = 0.0;
          xi[1] = 0.0;
          break;
        }
        default:
          FOUR_C_THROW("LocCoordsOfNode: Node number {} in segment {} out of range", lid, id());
      }

      break;
    }

    //==================================================
    //                     NURBS
    case Core::FE::CellType::nurbs2:
    {
      if (lid == 0)
        xi[0] = -1.0;
      else if (lid == 1)
        xi[0] = 1.0;
      else
        FOUR_C_THROW("ERROR: local_coordinates_of_node: Node number {} in segment {} out of range",
            lid, id());

      // we are in the 2D case here!
      xi[1] = 0.0;

      break;
    }
    case Core::FE::CellType::nurbs3:
    {
      if (lid == 0)
        xi[0] = -1.0;
      else if (lid == 1)
        xi[0] = 0.0;
      else if (lid == 2)
        xi[0] = 1.0;
      else
        FOUR_C_THROW("ERROR: local_coordinates_of_node: Node number {} in segment {} out of range",
            lid, id());

      // we are in the 2D case here!
      xi[1] = 0.0;

      break;
    }
    case Core::FE::CellType::nurbs9:
    {
      switch (lid)
      {
        case 0:
        {
          xi[0] = -1.0;
          xi[1] = -1.0;
          break;
        }
        case 1:
        {
          xi[0] = 0.0;
          xi[1] = -1.0;
          break;
        }
        case 2:
        {
          xi[0] = 1.0;
          xi[1] = -1.0;
          break;
        }
        case 3:
        {
          xi[0] = -1.0;
          xi[1] = 0.0;
          break;
        }
        case 4:
        {
          xi[0] = 0.0;
          xi[1] = 0.0;
          break;
        }
        case 5:
        {
          xi[0] = 1.0;
          xi[1] = 0.0;
          break;
        }
        case 6:
        {
          xi[0] = -1.0;
          xi[1] = 1.0;
          break;
        }
        case 7:
        {
          xi[0] = 0.0;
          xi[1] = 1.0;
          break;
        }
        case 8:
        {
          xi[0] = 1.0;
          xi[1] = 1.0;
          break;
        }
        default:
          FOUR_C_THROW("LocCoordsOfNode: Node number {} in segment {} out of range", lid, id());
      }

      break;
    }
    // unknown case
    default:
      FOUR_C_THROW("local_coordinates_of_node called for unknown element type");
  }
  return true;
}

/*----------------------------------------------------------------------*
 |  Get local numbering for global node id                    popp 12/07|
 *----------------------------------------------------------------------*/
int Mortar::Element::get_local_node_id(int nid) const
{
  int lid = -1;

  // look for global ID nid in element's nodes
  for (int i = 0; i < num_node(); ++i)
    if (node_ids()[i] == nid)
    {
      lid = i;
      break;
    }

  if (lid < 0) FOUR_C_THROW("Cannot find node {} in segment {}", nid, id());

  return lid;
}

/*----------------------------------------------------------------------*
 |  Build element normal at node                              popp 12/07|
 *----------------------------------------------------------------------*/
void Mortar::Element::build_normal_at_node(
    int nid, int& i, Core::LinAlg::SerialDenseMatrix& elens) const
{
  // find this node in my list of nodes and get local numbering
  int lid = get_local_node_id(nid);

  // get local coordinates for this node
  double xi[2];
  local_coordinates_of_node(lid, xi);

  // build an outward unit normal at xi and return it
  compute_normal_at_xi(xi, i, elens);
}

/*----------------------------------------------------------------------*
 |  Compute element normal at loc. coord. xi                  popp 09/08|
 *----------------------------------------------------------------------*/
void Mortar::Element::compute_normal_at_xi(
    const double* xi, int& i, Core::LinAlg::SerialDenseMatrix& elens) const
{
  // empty local basis vectors
  double gxi[3];
  double geta[3];

  // metrics routine gives local basis vectors
  metrics(xi, gxi, geta);

  // n is cross product of gxi and geta
  elens(0, i) = (gxi[1] * geta[2] - gxi[2] * geta[1]) * normal_fac();
  elens(1, i) = (gxi[2] * geta[0] - gxi[0] * geta[2]) * normal_fac();
  elens(2, i) = (gxi[0] * geta[1] - gxi[1] * geta[0]) * normal_fac();

  // store length of normal and other information into elens
  elens(4, i) =
      sqrt(elens(0, i) * elens(0, i) + elens(1, i) * elens(1, i) + elens(2, i) * elens(2, i));
  if (elens(4, i) < 1e-12) FOUR_C_THROW("ComputeNormalAtXi gives normal of length 0!");
  elens(3, i) = id();
  elens(5, i) = mo_data().area();
}

/*----------------------------------------------------------------------*
 |  Compute element normal at loc. coord. xi                  popp 11/08|
 *----------------------------------------------------------------------*/
double Mortar::Element::compute_unit_normal_at_xi(const double* xi, double* n) const
{
  // check input
  if (!xi) FOUR_C_THROW("compute_unit_normal_at_xi called with xi=nullptr");
  if (!n) FOUR_C_THROW("compute_unit_normal_at_xi called with n=nullptr");

  // empty local basis vectors
  double gxi[3];
  double geta[3];

  // metrics routine gives local basis vectors
  metrics(xi, gxi, geta);

  // n is cross product of gxi and geta
  n[0] = (gxi[1] * geta[2] - gxi[2] * geta[1]) * normal_fac();
  n[1] = (gxi[2] * geta[0] - gxi[0] * geta[2]) * normal_fac();
  n[2] = (gxi[0] * geta[1] - gxi[1] * geta[0]) * normal_fac();

  // build unit normal
  const double length = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (length < 1e-12) FOUR_C_THROW("Normal of length zero!");
  for (int i = 0; i < 3; ++i) n[i] /= length;

  return length;
}


/*----------------------------------------------------------------------*
 |  Compute nodal averaged normal at xi                      farah 06/16|
 *----------------------------------------------------------------------*/
double Mortar::Element::compute_averaged_unit_normal_at_xi(const double* xi, double* n) const
{
  // check input
  if (!xi) FOUR_C_THROW("compute_unit_normal_at_xi called with xi=nullptr");
  if (!n) FOUR_C_THROW("compute_unit_normal_at_xi called with n=nullptr");

  int nnodes = num_point();
  Core::LinAlg::SerialDenseVector val(nnodes);
  Core::LinAlg::SerialDenseMatrix deriv(nnodes, 2, true);

  // get shape function values and derivatives at xi
  evaluate_shape(xi, val, deriv, nnodes, false);

  // initialize n
  n[0] = 0.0;
  n[1] = 0.0;
  n[2] = 0.0;

  // loop over all nodes of this element
  for (int i = 0; i < num_node(); ++i)
  {
    const Node* mymrtrnode = dynamic_cast<const Node*>(nodes()[i]);
    n[0] = val[i] * mymrtrnode->mo_data().n()[0];
    n[1] = val[i] * mymrtrnode->mo_data().n()[1];
    n[2] = val[i] * mymrtrnode->mo_data().n()[2];
  }

  const double length = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (length < 1e-12) FOUR_C_THROW("Normal of length zero!");
  for (int i = 0; i < 3; ++i) n[i] /= length;

  return length;
}

/*----------------------------------------------------------------------*
 |  Compute unit normal derivative at loc. coord. xi          popp 03/09|
 *----------------------------------------------------------------------*/
void Mortar::Element::deriv_unit_normal_at_xi(
    const double* xi, std::vector<Core::Gen::Pairedvector<int, double>>& derivn) const
{
  // initialize variables
  const int nnodes = num_node();
  const Core::Nodes::Node* const* mynodes = nodes();
  if (!mynodes) FOUR_C_THROW("DerivUnitNormalAtXi: Null pointer!");

  Core::LinAlg::SerialDenseVector val(nnodes);
  Core::LinAlg::SerialDenseMatrix deriv(nnodes, 2, true);

  double gxi[3];
  double geta[3];

  // get shape function values and derivatives at xi
  evaluate_shape(xi, val, deriv, nnodes);

  // get local element basis vectors
  metrics(xi, gxi, geta);

  // n is cross product of gxi and geta
  std::array<double, 3> n = {0.0, 0.0, 0.0};
  n[0] = gxi[1] * geta[2] - gxi[2] * geta[1] * normalfac_;
  n[1] = gxi[2] * geta[0] - gxi[0] * geta[2] * normalfac_;
  n[2] = gxi[0] * geta[1] - gxi[1] * geta[0] * normalfac_;

  // build unit normal
  const double length = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (length < 1e-12) FOUR_C_THROW("Normal of length zero!");
  for (int i = 0; i < 3; ++i) n[i] /= length;

  // check if this mortar ele is an IntEle
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> nodelin;
  node_linearization(nodelin);

  int nderiv = nnodes * 3;
  // to be safe if it is a IntEle for a nurbs9
  if (shape() == Core::FE::CellType::quad4) nderiv = 9 * 3;

  // resize derivn
  derivn.resize(3, nderiv);

  // non-unit normal derivative
  std::vector<Core::Gen::Pairedvector<int, double>> derivnnu(
      3, nderiv);  // assume that each node has 3 dofs...
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // now the derivative
  for (int n = 0; n < nnodes; ++n)
  {
    const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[n]);
    if (!mymrtrnode) FOUR_C_THROW("DerivUnitNormalAtXi: Null pointer!");
    int ndof = mymrtrnode->num_dof();

    // derivative weighting matrix for current node
    Core::LinAlg::Matrix<3, 3> F;
    F(0, 0) = 0.0;
    F(1, 1) = 0.0;
    F(2, 2) = 0.0;
    F(0, 1) = geta[2] * deriv(n, 0) - gxi[2] * deriv(n, 1);
    F(0, 2) = gxi[1] * deriv(n, 1) - geta[1] * deriv(n, 0);
    F(1, 0) = gxi[2] * deriv(n, 1) - geta[2] * deriv(n, 0);
    F(1, 2) = geta[0] * deriv(n, 0) - gxi[0] * deriv(n, 1);
    F(2, 0) = geta[1] * deriv(n, 0) - gxi[1] * deriv(n, 1);
    F(2, 1) = gxi[0] * deriv(n, 1) - geta[0] * deriv(n, 0);

    // create directional derivatives
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < ndof; ++k)
        for (CI p = nodelin[n][k].begin(); p != nodelin[n][k].end(); ++p)
          (derivnnu[j])[p->first] += F(j, k) * p->second;
  }

  const double ll = length * length;
  const double linv = 1.0 / length;
  const double lllinv = 1.0 / (length * length * length);
  const double sxsx = n[0] * n[0] * ll;
  const double sxsy = n[0] * n[1] * ll;
  const double sxsz = n[0] * n[2] * ll;
  const double sysy = n[1] * n[1] * ll;
  const double sysz = n[1] * n[2] * ll;
  const double szsz = n[2] * n[2] * ll;

  for (CI p = derivnnu[0].begin(); p != derivnnu[0].end(); ++p)
  {
    derivn[0][p->first] += linv * (p->second) * normalfac_;
    derivn[0][p->first] -= lllinv * sxsx * (p->second) * normalfac_;
    derivn[1][p->first] -= lllinv * sxsy * (p->second) * normalfac_;
    derivn[2][p->first] -= lllinv * sxsz * (p->second) * normalfac_;
  }

  for (CI p = derivnnu[1].begin(); p != derivnnu[1].end(); ++p)
  {
    derivn[1][p->first] += linv * (p->second) * normalfac_;
    derivn[1][p->first] -= lllinv * sysy * (p->second) * normalfac_;
    derivn[0][p->first] -= lllinv * sxsy * (p->second) * normalfac_;
    derivn[2][p->first] -= lllinv * sysz * (p->second) * normalfac_;
  }

  for (CI p = derivnnu[2].begin(); p != derivnnu[2].end(); ++p)
  {
    derivn[2][p->first] += linv * (p->second) * normalfac_;
    derivn[2][p->first] -= lllinv * szsz * (p->second) * normalfac_;
    derivn[0][p->first] -= lllinv * sxsz * (p->second) * normalfac_;
    derivn[1][p->first] -= lllinv * sysz * (p->second) * normalfac_;
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Get nodal coordinates of the element                      popp 01/08|
 *----------------------------------------------------------------------*/
void Mortar::Element::get_nodal_coords(Core::LinAlg::SerialDenseMatrix& coord) const
{
  const int nnodes = num_point();
  const Core::Nodes::Node* const* mynodes = nodes();
  if (!mynodes) FOUR_C_THROW("GetNodalCoords: Null pointer!");
  if (coord.numRows() != 3 || coord.numCols() != nnodes)
    FOUR_C_THROW("GetNodalCoords: Dimensions!");

  for (int i = 0; i < nnodes; ++i)
  {
    const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);
    if (!mymrtrnode) FOUR_C_THROW("GetNodalCoords: Null pointer!");

    const double* x = mymrtrnode->xspatial();
    std::copy(x, x + 3, &coord(0, i));
  }
}

/*----------------------------------------------------------------------*
 |  Get old nodal coordinates of the element               gitterle 08/10|
 *----------------------------------------------------------------------*/
void Mortar::Element::get_nodal_coords_old(
    Core::LinAlg::SerialDenseMatrix& coord, bool isinit) const
{
  const int nnodes = num_point();
  const Core::Nodes::Node* const* mynodes = nodes();
  if (!mynodes) FOUR_C_THROW("GetNodalCoordsOld: Null pointer!");
  if (coord.numRows() != 3 || coord.numCols() != nnodes)
    FOUR_C_THROW("GetNodalCoordsOld: Dimensions!");

  for (int i = 0; i < nnodes; ++i)
  {
    const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);
    if (!mymrtrnode) FOUR_C_THROW("GetNodalCoordsOld: Null pointer!");

    coord(0, i) = mymrtrnode->x()[0] + mymrtrnode->uold()[0];
    coord(1, i) = mymrtrnode->x()[1] + mymrtrnode->uold()[1];
    if (n_dim() == 3) coord(2, i) = mymrtrnode->x()[2] + mymrtrnode->uold()[2];
  }
}

/*----------------------------------------------------------------------*
 |  Get lagrange multipliers of the element                gitterle 08/10|
 *----------------------------------------------------------------------*/
void Mortar::Element::get_nodal_lag_mult(
    Core::LinAlg::SerialDenseMatrix& lagmult, bool isinit) const
{
  int nnodes = num_node();
  const Core::Nodes::Node* const* mynodes = nodes();
  if (!mynodes) FOUR_C_THROW("GetNodalLagMult: Null pointer!");
  if (lagmult.numRows() != 3 || lagmult.numCols() != nnodes)
    FOUR_C_THROW("GetNodalLagMult: Dimensions!");

  for (int i = 0; i < nnodes; ++i)
  {
    const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);
    if (!mymrtrnode) FOUR_C_THROW("GetNodalCoords: Null pointer!");

    lagmult(0, i) = mymrtrnode->mo_data().lm()[0];
    lagmult(1, i) = mymrtrnode->mo_data().lm()[1];
    lagmult(2, i) = mymrtrnode->mo_data().lm()[2];
  }
}

/*----------------------------------------------------------------------*
 |  Evaluate element metrics (local basis vectors)            popp 08/08|
 *----------------------------------------------------------------------*/
void Mortar::Element::metrics(const double* xi, double* gxi, double* geta) const
{
  std::fill(gxi, gxi + 3, 0.0);
  std::fill(geta, geta + 3, 0.0);

  int nnodes = num_point();

  int dim = 0;
  Core::FE::CellType dt = shape();
  switch (dt)
  {
    case Core::FE::CellType::line2:
    case Core::FE::CellType::line3:
    case Core::FE::CellType::nurbs2:
    case Core::FE::CellType::nurbs3:
    {
      dim = 2;
      break;
    }
    case Core::FE::CellType::tri3:
    case Core::FE::CellType::quad4:
    case Core::FE::CellType::tri6:
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
    case Core::FE::CellType::nurbs4:
    case Core::FE::CellType::nurbs8:
    case Core::FE::CellType::nurbs9:
    {
      dim = 3;
      break;
    }
    default:
      FOUR_C_THROW("Metrics called for unknown element type");
  }

  Core::LinAlg::SerialDenseVector val(nnodes);
  Core::LinAlg::SerialDenseMatrix deriv(nnodes, 2, true);

  // get shape function values and derivatives at xi
  evaluate_shape(xi, val, deriv, nnodes, false);

  // get coordinates of element nodes
  Core::LinAlg::SerialDenseMatrix coord(3, nnodes);
  get_nodal_coords(coord);

  // build basis vectors gxi and geta
  for (int i = 0; i < nnodes; ++i)
  {
    // first local basis vector
    gxi[0] += deriv(i, 0) * coord(0, i);
    gxi[1] += deriv(i, 0) * coord(1, i);
    gxi[2] += deriv(i, 0) * coord(2, i);

    // second local basis vector
    geta[0] += deriv(i, 1) * coord(0, i);
    geta[1] += deriv(i, 1) * coord(1, i);
    geta[2] += deriv(i, 1) * coord(2, i);
  }

  // reset geta to (0,0,1) in 2D case
  if (dim == 2)
  {
    geta[0] = 0.0;
    geta[1] = 0.0;
    geta[2] = 1.0;
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate Jacobian determinant                             popp 12/07|
 *----------------------------------------------------------------------*/
double Mortar::Element::jacobian(const double* xi) const
{
  double jac = 0.0;
  double gxi[3];
  double geta[3];
  Core::FE::CellType dt = shape();

  // 2D linear case (2noded line element)
  if (dt == Core::FE::CellType::line2) jac = mo_data().area() * 0.5;

  // 3D linear case (3noded triangular element)
  else if (dt == Core::FE::CellType::tri3)
    jac = mo_data().area() * 2.0;

  // 2D quadratic case (3noded line element)
  // 3D bilinear case (4noded quadrilateral element)
  // 3D quadratic case (6noded triangular element)
  // 3D serendipity case (8noded quadrilateral element)
  // 3D biquadratic case (9noded quadrilateral element)
  else if (dt == Core::FE::CellType::line3 || dt == Core::FE::CellType::quad4 ||
           dt == Core::FE::CellType::tri6 || dt == Core::FE::CellType::quad8 ||
           dt == Core::FE::CellType::quad9 || dt == Core::FE::CellType::nurbs2 ||
           dt == Core::FE::CellType::nurbs3 || dt == Core::FE::CellType::nurbs4 ||
           dt == Core::FE::CellType::nurbs8 || dt == Core::FE::CellType::nurbs9)
  {
    // metrics routine gives local basis vectors
    metrics(xi, gxi, geta);

    // cross product of gxi and geta
    std::array<double, 3> cross = {0.0, 0.0, 0.0};
    cross[0] = gxi[1] * geta[2] - gxi[2] * geta[1];
    cross[1] = gxi[2] * geta[0] - gxi[0] * geta[2];
    cross[2] = gxi[0] * geta[1] - gxi[1] * geta[0];
    jac = sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
  }

  // unknown case
  else
    FOUR_C_THROW("Jacobian called for unknown element type!");

  return jac;
}

/*----------------------------------------------------------------------*
 |  Evaluate directional deriv. of Jacobian det.              popp 05/08|
 *----------------------------------------------------------------------*/
void Mortar::Element::deriv_jacobian(
    const double* xi, Core::Gen::Pairedvector<int, double>& derivjac) const
{
  // get element nodes
  int nnodes = num_node();

  const Core::Nodes::Node* const* mynodes = nullptr;  // Nodes();
  mynodes = nodes();

  if (!mynodes) FOUR_C_THROW("DerivJacobian: Null pointer!");

  // the inverse Jacobian
  double jacinv = 0.0;
  double gxi[3];
  double geta[3];

  // evaluate shape functions
  Core::LinAlg::SerialDenseVector val(nnodes);
  Core::LinAlg::SerialDenseMatrix deriv(nnodes, 2, true);
  evaluate_shape(xi, val, deriv, nnodes, false);

  // metrics routine gives local basis vectors
  metrics(xi, gxi, geta);

  // cross product of gxi and geta
  std::array<double, 3> cross = {0.0, 0.0, 0.0};
  cross[0] = gxi[1] * geta[2] - gxi[2] * geta[1];
  cross[1] = gxi[2] * geta[0] - gxi[0] * geta[2];
  cross[2] = gxi[0] * geta[1] - gxi[1] * geta[0];

  Core::FE::CellType dt = shape();

  // 2D linear case (2noded line element)
  switch (dt)
  {
    // 3D linear case (3noded triangular element)
    case Core::FE::CellType::tri3:
    {
      jacinv = 1.0 / (mo_data().area() * 2.0);
      break;
    }
    // default 2-D case
    case Core::FE::CellType::line2:
    {
      jacinv = 2.0 / mo_data().area();
      break;
    }
    // 2D quadratic case (3noded line element)
    // 3D bilinear case (4noded quadrilateral element)
    // 3D quadratic case (6noded triangular element)
    // 3D serendipity case (8noded quadrilateral element)
    // 3D biquadratic case (9noded quadrilateral element)
    /* no break (upper case) */
    case Core::FE::CellType::line3:
    case Core::FE::CellType::quad4:
    case Core::FE::CellType::tri6:
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
    case Core::FE::CellType::nurbs2:
    case Core::FE::CellType::nurbs3:
    case Core::FE::CellType::nurbs4:
    case Core::FE::CellType::nurbs8:
    case Core::FE::CellType::nurbs9:
    {
      jacinv = 1.0 / sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
      break;
    }
    default:
      FOUR_C_THROW("Jac. derivative not implemented for this type of Element");
  }

  // *********************************************************************
  // compute Jacobian derivative
  // *********************************************************************
  // (loop over all nodes and over all nodal dofs to capture all
  // potential dependencies of the Jacobian. Note that here we only
  // need to compute the DIRECT derivative of Lin(J), as the current
  // GP coordinate does not change! The derivative d_jac_d_xi is done in
  // a special function (see above)!
  // *********************************************************************
  for (int i = 0; i < nnodes; ++i)
  {
    const Mortar::Node* mymrtrnode = dynamic_cast<const Mortar::Node*>(mynodes[i]);
    if (!mymrtrnode) FOUR_C_THROW("DerivJacobian: Null pointer!");

    derivjac[mymrtrnode->dofs()[0]] +=
        jacinv * (cross[2] * geta[1] - cross[1] * geta[2]) * deriv(i, 0);
    derivjac[mymrtrnode->dofs()[0]] +=
        jacinv * (cross[1] * gxi[2] - cross[2] * gxi[1]) * deriv(i, 1);
    derivjac[mymrtrnode->dofs()[1]] +=
        jacinv * (cross[0] * geta[2] - cross[2] * geta[0]) * deriv(i, 0);
    derivjac[mymrtrnode->dofs()[1]] +=
        jacinv * (cross[2] * gxi[0] - cross[0] * gxi[2]) * deriv(i, 1);

    if (mymrtrnode->num_dof() == 3)
    {
      derivjac[mymrtrnode->dofs()[2]] +=
          jacinv * (cross[1] * geta[0] - cross[0] * geta[1]) * deriv(i, 0);
      derivjac[mymrtrnode->dofs()[2]] +=
          jacinv * (cross[0] * gxi[1] - cross[1] * gxi[0]) * deriv(i, 1);
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Compute length / area of the element                      popp 12/07|
 *----------------------------------------------------------------------*/
double Mortar::Element::compute_area() const
{
  double area = 0.0;
  Core::FE::CellType dt = shape();

  // 2D linear case (2noded line element)
  if (dt == Core::FE::CellType::line2)
  {
    // no integration necessary (constant Jacobian)
    Core::LinAlg::SerialDenseMatrix coord(3, num_point());
    get_nodal_coords(coord);

    // build vector between the two nodes
    std::array<double, 3> tang = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k)
    {
      tang[k] = coord(k, 1) - coord(k, 0);
    }
    area = sqrt(tang[0] * tang[0] + tang[1] * tang[1] + tang[2] * tang[2]);
  }

  // 3D linear case (3noded triangular element)
  else if (dt == Core::FE::CellType::tri3)
  {
    // no integration necessary (constant Jacobian)
    Core::LinAlg::SerialDenseMatrix coord(3, num_point());
    get_nodal_coords(coord);

    // build vectors between the three nodes
    std::array<double, 3> t1 = {0.0, 0.0, 0.0};
    std::array<double, 3> t2 = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k)
    {
      t1[k] = coord(k, 1) - coord(k, 0);
      t2[k] = coord(k, 2) - coord(k, 0);
    }

    // cross product of t1 and t2
    std::array<double, 3> t1xt2 = {0.0, 0.0, 0.0};
    t1xt2[0] = t1[1] * t2[2] - t1[2] * t2[1];
    t1xt2[1] = t1[2] * t2[0] - t1[0] * t2[2];
    t1xt2[2] = t1[0] * t2[1] - t1[1] * t2[0];
    area = 0.5 * sqrt(t1xt2[0] * t1xt2[0] + t1xt2[1] * t1xt2[1] + t1xt2[2] * t1xt2[2]);
  }

  // 2D quadratic case   (3noded line element)
  // 3D bilinear case    (4noded quadrilateral element)
  // 3D quadratic case   (6noded triangular element)
  // 3D serendipity case (8noded quadrilateral element)
  // 3D biquadratic case (9noded quadrilateral element)
  else if (dt == Core::FE::CellType::line3 || dt == Core::FE::CellType::quad4 ||
           dt == Core::FE::CellType::tri6 || dt == Core::FE::CellType::quad8 ||
           dt == Core::FE::CellType::quad9 || dt == Core::FE::CellType::nurbs2 ||
           dt == Core::FE::CellType::nurbs3 || dt == Core::FE::CellType::nurbs4 ||
           dt == Core::FE::CellType::nurbs8 || dt == Core::FE::CellType::nurbs9)
  {
    // Gauss quadrature with correct num_gp and Dim
    Mortar::ElementIntegrator integrator(dt);
    double detg = 0.0;

    // loop over all Gauss points, build Jacobian and compute area
    for (int j = 0; j < integrator.n_gp(); ++j)
    {
      double gpc[2] = {integrator.coordinate(j, 0), integrator.coordinate(j, 1)};
      detg = jacobian(gpc);
      area += integrator.weight(j) * detg;
    }
  }

  // other cases not implemented yet
  else
    FOUR_C_THROW("Area computation not implemented for this type of Mortar::Element");

  return area;
}


/*----------------------------------------------------------------------*
 |  Compute length / area of the element                     seitz 09/17|
 *----------------------------------------------------------------------*/
double Mortar::Element::compute_area_deriv(Core::Gen::Pairedvector<int, double>& area_deriv) const
{
  double area = 0.0;
  Core::FE::CellType dt = shape();

  // 2D linear case (2noded line element)
  if (dt == Core::FE::CellType::line2)
  {
    // no integration necessary (constant Jacobian)
    Core::LinAlg::SerialDenseMatrix coord(3, num_point());
    get_nodal_coords(coord);

    // build vector between the two nodes
    std::array<double, 3> tang = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k)
    {
      tang[k] = coord(k, 1) - coord(k, 0);
    }
    area = sqrt(tang[0] * tang[0] + tang[1] * tang[1] + tang[2] * tang[2]);
  }

  // 3D linear case (3noded triangular element)
  else if (dt == Core::FE::CellType::tri3)
  {
    // no integration necessary (constant Jacobian)
    Core::LinAlg::SerialDenseMatrix coord(3, num_point());
    get_nodal_coords(coord);

    // build vectors between the three nodes
    std::array<double, 3> t1 = {0.0, 0.0, 0.0};
    std::array<double, 3> t2 = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k)
    {
      t1[k] = coord(k, 1) - coord(k, 0);
      t2[k] = coord(k, 2) - coord(k, 0);
    }

    // cross product of t1 and t2
    std::array<double, 3> t1xt2 = {0.0, 0.0, 0.0};
    t1xt2[0] = t1[1] * t2[2] - t1[2] * t2[1];
    t1xt2[1] = t1[2] * t2[0] - t1[0] * t2[2];
    t1xt2[2] = t1[0] * t2[1] - t1[1] * t2[0];
    area = 0.5 * sqrt(t1xt2[0] * t1xt2[0] + t1xt2[1] * t1xt2[1] + t1xt2[2] * t1xt2[2]);
  }

  // 2D quadratic case   (3noded line element)
  // 3D bilinear case    (4noded quadrilateral element)
  // 3D quadratic case   (6noded triangular element)
  // 3D serendipity case (8noded quadrilateral element)
  // 3D biquadratic case (9noded quadrilateral element)
  else if (dt == Core::FE::CellType::line3 || dt == Core::FE::CellType::quad4 ||
           dt == Core::FE::CellType::tri6 || dt == Core::FE::CellType::quad8 ||
           dt == Core::FE::CellType::quad9 || dt == Core::FE::CellType::nurbs2 ||
           dt == Core::FE::CellType::nurbs3 || dt == Core::FE::CellType::nurbs4 ||
           dt == Core::FE::CellType::nurbs8 || dt == Core::FE::CellType::nurbs9)
  {
    // Gauss quadrature with correct num_gp and Dim
    Mortar::ElementIntegrator integrator(dt);
    double detg = 0.0;

    // loop over all Gauss points, build Jacobian and compute area
    for (int j = 0; j < integrator.n_gp(); ++j)
    {
      double gpc[2] = {integrator.coordinate(j, 0), integrator.coordinate(j, 1)};
      detg = jacobian(gpc);
      area += integrator.weight(j) * detg;

      Core::Gen::Pairedvector<int, double> derivjac(num_node() * n_dim());
      deriv_jacobian(gpc, derivjac);
      for (Core::Gen::Pairedvector<int, double>::const_iterator p = derivjac.begin();
          p != derivjac.end(); ++p)
        area_deriv[p->first] += integrator.weight(j) * p->second;
    }
  }

  // other cases not implemented yet
  else
    FOUR_C_THROW("Area computation not implemented for this type of Mortar::Element");

  return area;
}


/*----------------------------------------------------------------------*
 |  Get global coords for given local coords                  popp 01/08|
 *----------------------------------------------------------------------*/
bool Mortar::Element::local_to_global(const double* xi, double* globcoord, int inttype) const
{
  // check input
  if (!xi) FOUR_C_THROW("local_to_global called with xi=nullptr");
  if (!globcoord) FOUR_C_THROW("local_to_global called with globcoord=nullptr");

  // collect fundamental data
  const int nnodes = num_node();

  const Core::Nodes::Node* const* mynodes = nodes();
  if (!mynodes) FOUR_C_THROW("local_to_global: Null pointer!");
  Core::LinAlg::SerialDenseMatrix coord(3, nnodes);
  Core::LinAlg::SerialDenseVector val(nnodes);
  Core::LinAlg::SerialDenseMatrix deriv(nnodes, 2, true);

  // Evaluate shape, get nodal coords  and interpolate global coords
  evaluate_shape(xi, val, deriv, nnodes, false);
  get_nodal_coords(coord);

  // init globcoords
  for (int i = 0; i < 3; ++i) globcoord[i] = 0.0;

  for (int i = 0; i < nnodes; ++i)
  {
    if (inttype == 0)
    {
      // use shape function values for interpolation
      globcoord[0] += val[i] * coord(0, i);
      globcoord[1] += val[i] * coord(1, i);
      globcoord[2] += val[i] * coord(2, i);
    }
    else if (inttype == 1)
    {
      // use shape function derivatives xi for interpolation
      globcoord[0] += deriv(i, 0) * coord(0, i);
      globcoord[1] += deriv(i, 0) * coord(1, i);
      globcoord[2] += deriv(i, 0) * coord(2, i);
    }
    else if (inttype == 2)
    {
      // use shape function derivatives eta for interpolation
      globcoord[0] += deriv(i, 1) * coord(0, i);
      globcoord[1] += deriv(i, 1) * coord(1, i);
      globcoord[2] += deriv(i, 1) * coord(2, i);
    }
    else
      FOUR_C_THROW("Invalid interpolation type requested, only 0,1,2!");
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Compute minimal edge size of Mortar::Element                popp 11/08|
 *----------------------------------------------------------------------*/
double Mortar::Element::min_edge_size() const
{
  double minedgesize = 1.0e12;

  // get coordinates of element nodes
  Core::LinAlg::SerialDenseMatrix coord(3, num_point());
  get_nodal_coords(coord);

  switch (shape())
  {
    case Core::FE::CellType::line2:
    case Core::FE::CellType::line3:
    {
      // there is only one edge
      // (we approximate the quadratic case as linear)
      std::array<double, 3> diff = {0.0, 0.0, 0.0};
      for (int dim = 0; dim < 3; ++dim) diff[dim] = coord(dim, 1) - coord(dim, 0);
      minedgesize = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

      break;
    }
    case Core::FE::CellType::tri3:
    case Core::FE::CellType::tri6:
    {
      // there are three edges
      // (we approximate the quadratic case as linear)
      for (int edge = 0; edge < 3; ++edge)
      {
        std::array<double, 3> diff = {0.0, 0.0, 0.0};
        for (int dim = 0; dim < 3; ++dim)
        {
          if (edge == 2)
            diff[dim] = coord(dim, 0) - coord(dim, edge);
          else
            diff[dim] = coord(dim, edge + 1) - coord(dim, edge);
        }
        double dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
        if (dist < minedgesize) minedgesize = dist;
      }

      break;
    }
    case Core::FE::CellType::quad4:
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
    {
      // there are four edges
      // (we approximate the quadratic case as linear)
      for (int edge = 0; edge < 4; ++edge)
      {
        std::array<double, 3> diff = {0.0, 0.0, 0.0};
        for (int dim = 0; dim < 3; ++dim)
        {
          if (edge == 3)
            diff[dim] = coord(dim, 0) - coord(dim, edge);
          else
            diff[dim] = coord(dim, edge + 1) - coord(dim, edge);
        }
        double dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
        if (dist < minedgesize) minedgesize = dist;
      }

      break;
    }
    case Core::FE::CellType::nurbs3:
    {
      double sxi0[2] = {-1.0, 0.0};
      double sxi1[2] = {1.0, 0.0};
      const int nrow = num_node();
      Core::LinAlg::SerialDenseVector sval0(nrow);
      Core::LinAlg::SerialDenseVector sval1(nrow);
      Core::LinAlg::SerialDenseMatrix sderiv(nrow, 1);
      evaluate_shape(sxi0, sval0, sderiv, nrow);
      evaluate_shape(sxi1, sval1, sderiv, nrow);

      std::array<double, 3> gpx0 = {0.0, 0.0, 0.0};
      std::array<double, 3> gpx1 = {0.0, 0.0, 0.0};

      for (int j = 0; j < nrow; ++j)
      {
        for (int i = 0; i < 3; ++i)
        {
          gpx0[i] += sval0(j) * coord(i, j);
          gpx1[i] += sval1(j) * coord(i, j);
        }
      }

      std::array<double, 3> diff = {0.0, 0.0, 0.0};
      for (int dim = 0; dim < 3; ++dim) diff[dim] = gpx1[dim] - gpx0[dim];
      minedgesize = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

      break;
    }
    case Core::FE::CellType::nurbs9:
    {
      const int nrow = num_node();

      // get real point data
      Core::LinAlg::SerialDenseMatrix coordnurbs(3, nrow, true);

      // parameter space coordinates
      double sxi0[2] = {-1.0, -1.0};
      double sxi1[2] = {1.0, -1.0};
      double sxi2[2] = {1.0, 1.0};
      double sxi3[2] = {-1.0, 1.0};

      // evaluate shape functions at these coordinates
      Core::LinAlg::SerialDenseVector sval0(nrow);
      Core::LinAlg::SerialDenseVector sval1(nrow);
      Core::LinAlg::SerialDenseVector sval2(nrow);
      Core::LinAlg::SerialDenseVector sval3(nrow);
      Core::LinAlg::SerialDenseMatrix sderiv(nrow, 2);
      evaluate_shape(sxi0, sval0, sderiv, nrow);
      evaluate_shape(sxi1, sval1, sderiv, nrow);
      evaluate_shape(sxi2, sval2, sderiv, nrow);
      evaluate_shape(sxi3, sval3, sderiv, nrow);

      std::array<double, 3> gpx0 = {0.0, 0.0, 0.0};
      std::array<double, 3> gpx1 = {0.0, 0.0, 0.0};

      for (int j = 0; j < nrow; ++j)
      {
        for (int i = 0; i < 3; ++i)
        {
          gpx0[i] += sval0(j) * coord(i, j);
          gpx1[i] += sval1(j) * coord(i, j);
        }
      }

      // there are four edges
      // (we approximate the quadratic case as linear)
      for (int edge = 0; edge < 4; ++edge)
      {
        std::array<double, 3> diff = {0.0, 0.0, 0.0};
        for (int dim = 0; dim < 3; ++dim)
        {
          if (edge == 0)
            diff[dim] = coord(dim, 0) - coord(dim, 2);
          else if (edge == 1)
            diff[dim] = coord(dim, 2) - coord(dim, 8);
          else if (edge == 2)
            diff[dim] = coord(dim, 8) - coord(dim, 6);
          else if (edge == 3)
            diff[dim] = coord(dim, 6) - coord(dim, 0);
          else
            FOUR_C_THROW("Wrong edge size!");
        }
        double dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
        if (dist < minedgesize) minedgesize = dist;
      }

      break;
    }
    default:
    {
      FOUR_C_THROW("{} is not implemented for discretization type '{}' of Mortar::Element.",
          __PRETTY_FUNCTION__, Core::FE::cell_type_to_string(shape()));
      break;
    }
  }

  if (minedgesize == 1.0e12) FOUR_C_THROW("{} went wrong...!", __FUNCTION__);
  return minedgesize;
}

/*----------------------------------------------------------------------*
 |  Compute maximal edge size of Mortar::Element                popp 11/08|
 *----------------------------------------------------------------------*/
double Mortar::Element::max_edge_size() const
{
  double maxedgesize = 0.0;
  // get coordinates of element nodes
  Core::LinAlg::SerialDenseMatrix coord(3, num_point());
  get_nodal_coords(coord);

  switch (shape())
  {
    case Core::FE::CellType::line2:
    case Core::FE::CellType::line3:
    {
      // there is only one edge
      std::array<double, 3> diff = {0.0, 0.0, 0.0};
      for (int dim = 0; dim < 3; ++dim) diff[dim] = coord(dim, 1) - coord(dim, 0);
      maxedgesize = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

      break;
    }
    case Core::FE::CellType::tri3:
    case Core::FE::CellType::tri6:
    {
      // there are three edges
      for (int edge = 0; edge < 3; ++edge)
      {
        std::array<double, 3> diff = {0.0, 0.0, 0.0};
        for (int dim = 0; dim < 3; ++dim)
        {
          if (edge == 2)
            diff[dim] = coord(dim, 0) - coord(dim, edge);
          else
            diff[dim] = coord(dim, edge + 1) - coord(dim, edge);
        }
        double dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
        if (dist > maxedgesize) maxedgesize = dist;
      }

      break;
    }
    case Core::FE::CellType::quad4:
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
    {
      // there are four edges
      for (int edge = 0; edge < 4; ++edge)
      {
        std::array<double, 3> diff = {0.0, 0.0, 0.0};
        for (int dim = 0; dim < 3; ++dim)
        {
          if (edge == 3)
            diff[dim] = coord(dim, 0) - coord(dim, edge);
          else
            diff[dim] = coord(dim, edge + 1) - coord(dim, edge);
        }
        double dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
        if (dist > maxedgesize) maxedgesize = dist;
      }

      break;
    }
    default:
    {
      FOUR_C_THROW("{} is not implemented for discretization type '{}' of Mortar::Element.",
          __PRETTY_FUNCTION__, Core::FE::cell_type_to_string(shape()));
      break;
    }
  }

  if (maxedgesize < 1e-12) FOUR_C_THROW("MaxEdgeSize() went wrong...!");
  return maxedgesize;
}

/*----------------------------------------------------------------------*
 |  Initialize data container                                 popp 12/10|
 *----------------------------------------------------------------------*/
void Mortar::Element::initialize_data_container()
{
  // only initialize if not yet done
  if (modata_ == nullptr) modata_ = std::make_shared<Mortar::MortarEleDataContainer>();

  if (parent_element() != nullptr)
  {
    int numdof = parent_element()->num_node() *
                 parent_element()->num_dof_per_node(*parent_element()->nodes()[0]);
    mo_data().parent_disp() = std::vector<double>(numdof);
    for (int i = 0; i < numdof; ++i) mo_data().parent_disp()[i] = 0.0;
  }
}

/*----------------------------------------------------------------------*
 |  Reset data container                                      popp 12/10|
 *----------------------------------------------------------------------*/
void Mortar::Element::reset_data_container()
{
  // reset to nullptr
  modata_ = nullptr;

  return;
}

/*----------------------------------------------------------------------*
 |  Add one Mortar::Element to potential contact partners       popp 01/08|
 *----------------------------------------------------------------------*/
bool Mortar::Element::add_search_elements(const int& gid)
{
  // check calling element type
  if (!is_slave()) FOUR_C_THROW("AddSearchElements called for infeasible Mortar::Element!");

  // add new gid to vector of search candidates
  mo_data().search_elements().push_back(gid);

  return true;
}

/*----------------------------------------------------------------------*
 |  reset found search elements                              farah 10/13|
 *----------------------------------------------------------------------*/
void Mortar::Element::delete_search_elements()
{
  // check calling element type
  if (!is_slave()) FOUR_C_THROW("delete_search_elements called for infeasible Mortar::Element!");

  // add new gid to vector of search candidates
  mo_data().search_elements().clear();

  return;
}

/*----------------------------------------------------------------------*
 |  Derivatives of nodal spatial coords                      seitz 03/15|
 *----------------------------------------------------------------------*/
void Mortar::Element::node_linearization(
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& nodelin) const
{
  // resize the linearizations
  nodelin.resize(num_node(), std::vector<Core::Gen::Pairedvector<int, double>>(3, 1));

  // loop over all intEle nodes
  for (int in = 0; in < num_node(); ++in)
  {
    const Mortar::Node* mrtrnode = dynamic_cast<const Mortar::Node*>(nodes()[in]);
    for (int dim = 0; dim < n_dim(); ++dim) nodelin[in][dim][mrtrnode->dofs()[dim]] += 1.;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Element::estimate_nitsche_trace_max_eigenvalue()
{
  FOUR_C_ASSERT(n_dim() == 3,
      "Contact using Nitsche's method is only supported for 3D problems. We do not intend to "
      "support 2D problems.");

  auto surf_ele = parent_element()->surfaces()[face_parent_number()];
  auto* surf = dynamic_cast<Discret::Elements::SolidSurface*>(surf_ele.get());

  if (mo_data().parent_scalar().empty())
    traceHE_ = 1.0 / surf->estimate_nitsche_trace_max_eigenvalue(mo_data().parent_disp());
  else
    traceHE_ = 1.0 / surf->estimate_nitsche_trace_max_eigenvalue(
                         mo_data().parent_disp(), mo_data().parent_scalar());

  if (parent_element()->num_material() > 1)
    if (parent_element()->material(1)->material_type() == Core::Materials::m_thermo_fourier)
      traceHCond_ = 1.0 / surf->estimate_nitsche_trace_max_eigenvalue_tsi(mo_data().parent_disp());
}


/*----------------------------------------------------------------------*
 |                                                           seitz 10/16|
 *----------------------------------------------------------------------*/
Mortar::ElementNitscheContainer& Mortar::Element::get_nitsche_container()
{
  if (!parent_element()) FOUR_C_THROW("parent element pointer not set");

  if (nitsche_container_ == nullptr)
  {
    switch (parent_element()->shape())
    {
      case Core::FE::CellType::hex8:
        nitsche_container_ =
            std::make_shared<Mortar::ElementNitscheData<Core::FE::CellType::hex8>>();
        break;
      case Core::FE::CellType::tet4:
        nitsche_container_ =
            std::make_shared<Mortar::ElementNitscheData<Core::FE::CellType::tet4>>();
        break;
      case Core::FE::CellType::hex27:
        nitsche_container_ =
            std::make_shared<Mortar::ElementNitscheData<Core::FE::CellType::hex27>>();
        break;
      case Core::FE::CellType::nurbs27:
        nitsche_container_ =
            std::make_shared<Mortar::ElementNitscheData<Core::FE::CellType::nurbs27>>();
        break;
      default:
        FOUR_C_THROW("Nitsche data container not ready. Just add it here...");
    }
  }
  return *nitsche_container_;
}

FOUR_C_NAMESPACE_CLOSE
