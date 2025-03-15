// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_line_coupling.hpp"

#include "4C_contact_defines.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_integrator.hpp"
#include "4C_contact_integrator_factory.hpp"
#include "4C_contact_node.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_projector.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  ctor for lts/stl (public)                                farah 07/16|
 *----------------------------------------------------------------------*/
CONTACT::LineToSurfaceCoupling3d::LineToSurfaceCoupling3d(Core::FE::Discretization& idiscret,
    int dim, Teuchos::ParameterList& params, Element& pEle, std::shared_ptr<Mortar::Element>& lEle,
    std::vector<Element*> surfEles, LineToSurfaceCoupling3d::IntType type)
    : idiscret_(idiscret),
      dim_(dim),
      p_ele_(pEle),
      l_ele_(lEle),
      surf_eles_(surfEles),
      curr_ele_(-1),
      imortar_(params),
      int_type_(type)
{
  // empty constructor

  return;
}

/*----------------------------------------------------------------------*
 |  eval (public)                                            farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToSurfaceCoupling3d::evaluate_coupling()
{
  // clear entries of master vertices
  done_before().clear();

  // loop over all found master elements
  for (int nele = 0; nele < number_surface_elements(); ++nele)
  {
    // set internal counter
    curr_ele() = nele;

    // 1. init internal data
    initialize();

    // 2. create aux plane for master ele
    auxiliary_plane();  //--> build everything based on line element

    // 3. create aux line for slave ele
    auxiliary_line();

    // 4. check orientation
    if (!check_orientation()) return;

    // 5. project master nodes onto auxplane
    project_master();

    // 6. project slave line elements onto auxplane
    project_slave();

    // 7. perform line clipping
    line_clipping();

    // 8. intersections found?
    if ((int)inter_sections().size() == 0 or (int) inter_sections().size() == 1) continue;

    // 9. check length of Integration Line
    bool check = check_length();
    if (check == false) continue;

    // create empty lin vector
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> linvertex(
        2, std::vector<Core::Gen::Pairedvector<int, double>>(
               3, 3 * line_element()->num_node() + 3 * surface_element().num_node() + linsize_));

    // 10. linearize vertices
    linearize_vertices(linvertex);

    // 11. create intlines
    create_integration_lines(linvertex);

    // 12. consistent dual shape
    consist_dual_shape();

    // 13. integration
    integrate_line();
  }  // end loop

  return;
}

/*----------------------------------------------------------------------*
 |  init internal variables                                  farah 08/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToSurfaceCoupling3d::initialize()
{
  // reset auxplane normal, center and length
  auxn()[0] = 0.0;
  auxn()[1] = 0.0;
  auxn()[2] = 0.0;

  auxc()[0] = 0.0;
  auxc()[1] = 0.0;
  auxc()[2] = 0.0;

  lauxn() = 0.0;
  get_deriv_auxn().clear();
  get_deriv_auxc().clear();

  // reset normal of aux line
  //  AuxnLine()[0] = 0.0;
  //  AuxnLine()[1] = 0.0;
  //  AuxnLine()[2] = 0.0;
  //  GetDerivAuxnLine().clear();

  // clear all slave and master vertices
  slave_vertices().clear();
  master_vertices().clear();

  // clear previously found intersections
  inter_sections().clear();
  temp_inter_sections().clear();

  // clear integration line
  int_line() = nullptr;

  return;
}

/*----------------------------------------------------------------------*
 |  check orientation of line and surface element            farah 07/16|
 *----------------------------------------------------------------------*/
bool CONTACT::LineToSurfaceCoupling3d::check_orientation()
{
  // check if surface normal and line ele are parallel!

  // tolerance for line clipping
  const double sminedge = parent_element().min_edge_size();
  const double mminedge = surface_element().min_edge_size();
  const double tol = 0.001 * std::min(sminedge, mminedge);

  // -------------------------------------------
  // CHECK LINE TO SURFACE ORIENTATION!
  // calculate line ele vector
  std::array<double, 3> lvec = {0.0, 0.0, 0.0};
  Node* ns1 = dynamic_cast<Node*>(line_element()->nodes()[0]);
  Node* ns2 = dynamic_cast<Node*>(line_element()->nodes()[1]);
  lvec[0] = ns1->xspatial()[0] - ns2->xspatial()[0];
  lvec[1] = ns1->xspatial()[1] - ns2->xspatial()[1];
  lvec[2] = ns1->xspatial()[2] - ns2->xspatial()[2];

  // calculate lengths
  const double lengthS = sqrt(lvec[0] * lvec[0] + lvec[1] * lvec[1] + lvec[2] * lvec[2]);
  const double lengthA = sqrt(auxn_surf()[0] * auxn_surf()[0] + auxn_surf()[1] * auxn_surf()[1] +
                              auxn_surf()[2] * auxn_surf()[2]);
  const double prod = lengthS * lengthA;
  if (prod < 1e-12) return false;

  // calculate scalar product
  double scaprod = lvec[0] * auxn_surf()[0] + lvec[1] * auxn_surf()[1] + lvec[2] * auxn_surf()[2];
  scaprod = scaprod / (prod);
  double diff = abs(scaprod) - 1.0;

  if (abs(diff) < tol) return false;

  return true;
}


/*----------------------------------------------------------------------*
 |  calculate dual shape functions                           farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToSurfaceCoupling3d::consist_dual_shape()
{
  auto shapefcn = Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(imortar_, "LM_SHAPEFCN");
  auto consistent =
      Teuchos::getIntegralValue<Inpar::Mortar::ConsistentDualType>(imortar_, "LM_DUAL_CONSISTENT");

  if (shapefcn != Inpar::Mortar::shape_dual && shapefcn != Inpar::Mortar::shape_petrovgalerkin)
    return;

  if (consistent == Inpar::Mortar::consistent_none) return;

  if (i_type() == LineToSurfaceCoupling3d::lts) return;
  FOUR_C_THROW("consistent dual shapes for stl is experimental!");

  // slave nodes and dofs
  const int max_nnodes = 9;
  const int nnodes = surface_element().num_node();
  if (nnodes > max_nnodes)
    FOUR_C_THROW(
        "this function is not implemented to handle elements with that many nodes. Just adjust "
        "max_nnodes above");
  const int ndof = 3;

  // get number of master nodes
  int mnodes = line_element()->num_node();

  // Dual shape functions coefficient matrix and linearization
  Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes, true);
  surface_element().mo_data().deriv_dual_shape() =
      std::make_shared<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>(
          (nnodes + mnodes) * ndof, 0, Core::LinAlg::SerialDenseMatrix(nnodes, nnodes));
  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& derivae =
      *(surface_element().mo_data().deriv_dual_shape());

  // various variables
  double detg = 0.0;
  typedef Core::Gen::Pairedvector<int, double>::const_iterator _CI;

  // initialize matrices de and me
  Core::LinAlg::SerialDenseMatrix me(nnodes, nnodes, true);
  Core::LinAlg::SerialDenseMatrix de(nnodes, nnodes, true);

  Core::Gen::Pairedvector<int, Core::LinAlg::Matrix<max_nnodes + 1, max_nnodes>> derivde_new(
      (nnodes + mnodes) * ndof);

  // two-dim arrays of maps for linearization of me/de
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> derivme(
      nnodes, std::vector<Core::Gen::Pairedvector<int, double>>(nnodes, (nnodes + mnodes) * ndof));
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> derivde(
      nnodes, std::vector<Core::Gen::Pairedvector<int, double>>(nnodes, (nnodes + mnodes) * ndof));

  double A_tot = 0.;

  // get number of master nodes
  const int ncol = line_element()->num_node();

  std::shared_ptr<Mortar::IntCell> currcell = int_line();

  A_tot += currcell->area();

  // create an integrator for this cell
  CONTACT::Integrator integrator(imortar_, currcell->shape(), get_comm());

  // check if the cells are tri3
  // there's nothing wrong about other shapes, but as long as they are all
  // tri3 we can perform the jacobian calculation ( and its deriv) outside
  // the Gauss point loop
  if (currcell->shape() != Core::FE::CellType::line2)
    FOUR_C_THROW("only line2 integration cells at the moment. See comment in the code");

  detg = currcell->jacobian();
  // directional derivative of cell Jacobian
  Core::Gen::Pairedvector<int, double> derivjaccell((nnodes + ncol) * ndof);
  currcell->deriv_jacobian(derivjaccell);

  for (int gp = 0; gp < integrator.n_gp(); ++gp)
  {
    // coordinates and weight
    double eta[2] = {integrator.coordinate(gp, 0), integrator.coordinate(gp, 1)};
    const double wgt = integrator.weight(gp);

    // get global Gauss point coordinates
    double globgp[3] = {0.0, 0.0, 0.0};
    currcell->local_to_global(eta, globgp, 0);

    // project Gauss point onto slave integration element
    double sxi[2] = {0.0, 0.0};
    double sprojalpha = 0.0;
    Mortar::Projector::impl(surface_element())
        ->project_gauss_point_auxn_3d(globgp, auxn(), surface_element(), sxi, sprojalpha);

    // project Gauss point onto slave (parent) element
    double psxi[2] = {0., 0.};
    for (int i = 0; i < 2; ++i) psxi[i] = sxi[i];

    // create vector for shape function evaluation
    Core::LinAlg::SerialDenseVector sval(nnodes);
    Core::LinAlg::SerialDenseMatrix sderiv(nnodes, 2, true);

    // evaluate trace space shape functions at Gauss point
    surface_element().evaluate_shape(psxi, sval, sderiv, nnodes);

    // additional data for contact calculation (i.e. incl. derivative of dual shape functions
    // coefficient matrix) GP slave coordinate derivatives
    std::vector<Core::Gen::Pairedvector<int, double>> dsxigp(2, (nnodes + ncol) * ndof);
    // GP slave coordinate derivatives
    std::vector<Core::Gen::Pairedvector<int, double>> dpsxigp(2, (nnodes + ncol) * ndof);
    // global GP coordinate derivative on integration element
    Core::Gen::Pairedvector<int, Core::LinAlg::Matrix<3, 1>> lingp((nnodes + ncol) * ndof);

    // compute global GP coordinate derivative
    static Core::LinAlg::Matrix<3, 1> svalcell;
    static Core::LinAlg::Matrix<3, 2> sderivcell;
    currcell->evaluate_shape(eta, svalcell, sderivcell);

    for (int v = 0; v < 2; ++v)
      for (int d = 0; d < 3; ++d)
        for (_CI p = (currcell->get_deriv_vertex(v))[d].begin();
            p != (currcell->get_deriv_vertex(v))[d].end(); ++p)
          lingp[p->first](d) += svalcell(v) * (p->second);

    // compute GP slave coordinate derivatives
    integrator.deriv_xi_gp_3d_aux_plane(surface_element(), sxi, currcell->auxn(), dsxigp,
        sprojalpha, currcell->get_deriv_auxn(), lingp);

    // compute GP slave coordinate derivatives (parent element)
    dpsxigp = dsxigp;

    double fac = 0.;
    for (_CI p = derivjaccell.begin(); p != derivjaccell.end(); ++p)
    {
      Core::LinAlg::Matrix<max_nnodes + 1, max_nnodes>& dtmp = derivde_new[p->first];
      const double& ps = p->second;
      for (int j = 0; j < nnodes; ++j)
      {
        fac = wgt * sval[j] * ps;
        dtmp(nnodes, j) += fac;
        for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * sval[k];
      }
    }

    for (int i = 0; i < 2; ++i)
      for (_CI p = dpsxigp[i].begin(); p != dpsxigp[i].end(); ++p)
      {
        Core::LinAlg::Matrix<max_nnodes + 1, max_nnodes>& dtmp = derivde_new[p->first];
        const double& ps = p->second;
        for (int j = 0; j < nnodes; ++j)
        {
          fac = wgt * sderiv(j, i) * detg * ps;
          dtmp(nnodes, j) += fac;
          for (int k = 0; k < nnodes; ++k)
          {
            dtmp(k, j) += fac * sval[k];
            dtmp(j, k) += fac * sval[k];
          }
        }
      }

    // computing de, derivde and me, derivme and kappa, derivkappa
    for (int j = 0; j < nnodes; ++j)
    {
      double fac;
      fac = sval[j] * wgt;
      // computing de
      de(j, j) += fac * detg;

      for (int k = 0; k < nnodes; ++k)
      {
        // computing me
        fac = wgt * sval[j] * sval[k];
        me(j, k) += fac * detg;
      }
    }
  }

  // in case of no overlap just return, as there is no integration area
  // and therefore the consistent dual shape functions are not defined.
  // This doesn't matter, as there is no associated integration domain anyway
  if (A_tot < 1.e-12) return;

  // invert bi-ortho matrix me
  //  Core::LinAlg::SerialDenseMatrix meinv =
  //  Core::LinAlg::invert_and_multiply_by_cholesky(me,de,ae);

  Core::LinAlg::Matrix<4, 4, double> me_tmatrix(me, true);
  Core::LinAlg::inverse(me_tmatrix);
  Core::LinAlg::SerialDenseMatrix meinv = me;

  // build linearization of ae and store in derivdual
  // (this is done according to a quite complex formula, which
  // we get from the linearization of the biorthogonality condition:
  // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )
  typedef Core::Gen::Pairedvector<int,
      Core::LinAlg::Matrix<max_nnodes + 1, max_nnodes>>::const_iterator _CIM;
  for (_CIM p = derivde_new.begin(); p != derivde_new.end(); ++p)
  {
    Core::LinAlg::Matrix<max_nnodes + 1, max_nnodes>& dtmp = derivde_new[p->first];
    Core::LinAlg::SerialDenseMatrix& pt = derivae[p->first];
    for (int i = 0; i < nnodes; ++i)
      for (int j = 0; j < nnodes; ++j)
      {
        pt(i, j) += meinv(i, j) * dtmp(nnodes, i);

        for (int k = 0; k < nnodes; ++k)
          for (int l = 0; l < nnodes; ++l) pt(i, j) -= ae(i, k) * meinv(l, j) * dtmp(l, k);
      }
  }

  // store ae matrix in slave element data container
  surface_element().mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);

  return;
}

/*----------------------------------------------------------------------*
 |  integration for LTS (public)                             farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToSurfaceCoupling3d::integrate_line()
{
  // get solution strategy
  auto sol = Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(imortar_, "STRATEGY");

  // create integrator object
  std::shared_ptr<CONTACT::Integrator> integrator =
      CONTACT::INTEGRATOR::build_integrator(sol, imortar_, int_line()->shape(), get_comm());

  // perform integration
  if (i_type() == LineToSurfaceCoupling3d::lts)
  {
    integrator->integrate_deriv_cell_3d_aux_plane_lts(
        parent_element(), *line_element(), surface_element(), int_line(), auxn(), get_comm());
  }
  else if (i_type() == LineToSurfaceCoupling3d::stl)
  {
    integrator->integrate_deriv_cell_3d_aux_plane_stl(
        parent_element(), *line_element(), surface_element(), int_line(), auxn(), get_comm());
  }
  else
    FOUR_C_THROW("wrong integration type for line coupling!");

  return;
}

/*----------------------------------------------------------------------*
 |  check if all vertices are along one line                 farah 09/16|
 *----------------------------------------------------------------------*/
bool CONTACT::LineToSurfaceCoupling3d::check_line_on_line(Mortar::Vertex& edgeVertex1,
    Mortar::Vertex& edgeVertex0, Mortar::Vertex& lineVertex1, Mortar::Vertex& lineVertex0)
{
  // tolerance for line clipping
  const double sminedge = parent_element().min_edge_size();
  const double mminedge = surface_element().min_edge_size();
  const double tol = MORTARCLIPTOL * std::min(sminedge, mminedge);

  // check if point of edge is on line
  bool lineOnLine = false;
  std::array<double, 3> line = {0.0, 0.0, 0.0};
  std::array<double, 3> edgeLine = {0.0, 0.0, 0.0};

  for (int k = 0; k < 3; ++k)
  {
    line[k] = lineVertex1.coord()[k] - lineVertex0.coord()[k];
    edgeLine[k] = edgeVertex1.coord()[k] - lineVertex0.coord()[k];
  }

  double lengthLine = sqrt(line[0] * line[0] + line[1] * line[1] + line[2] * line[2]);
  double lengthedge =
      sqrt(edgeLine[0] * edgeLine[0] + edgeLine[1] * edgeLine[1] + edgeLine[2] * edgeLine[2]);

  if (lengthLine < tol) FOUR_C_THROW("Line Element is of zero length!");

  if (lengthedge < tol)
  {
    lineOnLine = true;
  }
  else
  {
    // calc scalar product
    double scaprod = line[0] * edgeLine[0] + line[1] * edgeLine[1] + line[2] * edgeLine[2];
    scaprod /= (lengthLine * lengthedge);

    if ((abs(scaprod) - tol < 1.0) and (abs(scaprod) + tol > 1.0)) lineOnLine = true;
  }

  if (!lineOnLine) return false;

  return true;
}

/*----------------------------------------------------------------------*
 |  geometric stuff (private)                                farah 08/16|
 *----------------------------------------------------------------------*/
bool CONTACT::LineToSurfaceCoupling3d::line_to_line_clipping(Mortar::Vertex& edgeVertex1,
    Mortar::Vertex& edgeVertex0, Mortar::Vertex& lineVertex1, Mortar::Vertex& lineVertex0)
{
  // output bool
  const bool out = false;

  // tolerance for line clipping
  const double sminedge = parent_element().min_edge_size();
  const double mminedge = surface_element().min_edge_size();
  const double tol = MORTARCLIPTOL * std::min(sminedge, mminedge);

  bool lineOnLine = check_line_on_line(edgeVertex1, edgeVertex0, lineVertex1, lineVertex0);

  if (!lineOnLine) FOUR_C_THROW("vertices not along a line, but already checked!");

  std::array<double, 3> line = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k) line[k] = lineVertex1.coord()[k] - lineVertex0.coord()[k];

  // LINE ON LINE!!! go on with real line to line clipping
  bool e0v0 = false;
  bool e0v1 = false;
  bool e1v0 = false;
  bool e1v1 = false;
  double prod0 = 0.0;
  double prod1 = 0.0;
  double prod2 = 0.0;
  double prod3 = 0.0;
  // check of both master vertices are out of line in 0 direction
  std::array<double, 3> lineEdge0Vert0 = {0.0, 0.0, 0.0};
  std::array<double, 3> lineEdge1Vert0 = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k)
  {
    lineEdge0Vert0[k] = edgeVertex0.coord()[k] - lineVertex0.coord()[k];
    lineEdge1Vert0[k] = edgeVertex1.coord()[k] - lineVertex0.coord()[k];
  }

  for (int k = 0; k < 3; ++k)
  {
    prod0 += lineEdge0Vert0[k] * line[k];
    prod1 += lineEdge1Vert0[k] * line[k];
  }

  if (prod0 < 0.0)
    e0v0 = false;
  else
    e0v0 = true;
  if (prod1 < 0.0)
    e1v0 = false;
  else
    e1v0 = true;


  // check of both master vertices are out of line in 1 direction
  std::array<double, 3> lineEdge0Vert1 = {0.0, 0.0, 0.0};
  std::array<double, 3> lineEdge1Vert1 = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k)
  {
    lineEdge0Vert1[k] = edgeVertex0.coord()[k] - lineVertex1.coord()[k];
    lineEdge1Vert1[k] = edgeVertex1.coord()[k] - lineVertex1.coord()[k];
  }

  for (int k = 0; k < 3; ++k)
  {
    prod2 -= lineEdge0Vert1[k] * line[k];
    prod3 -= lineEdge1Vert1[k] * line[k];
  }

  if (prod2 < 0.0)
    e0v1 = false;
  else
    e0v1 = true;
  if (prod3 < 0.0)
    e1v1 = false;
  else
    e1v1 = true;

  // check if vertices are lying on each other
  bool e0isV0 = true;
  bool e0isV1 = true;
  bool e1isV0 = true;
  bool e1isV1 = true;

  std::array<double, 3> test0 = {0.0, 0.0, 0.0};
  std::array<double, 3> test1 = {0.0, 0.0, 0.0};
  std::array<double, 3> test2 = {0.0, 0.0, 0.0};
  std::array<double, 3> test3 = {0.0, 0.0, 0.0};

  for (int k = 0; k < 3; ++k)
  {
    test0[k] = edgeVertex0.coord()[k] - lineVertex0.coord()[k];
    test1[k] = edgeVertex0.coord()[k] - lineVertex1.coord()[k];
    test2[k] = edgeVertex1.coord()[k] - lineVertex0.coord()[k];
    test3[k] = edgeVertex1.coord()[k] - lineVertex1.coord()[k];
  }

  double l0 = sqrt(test0[0] * test0[0] + test0[1] * test0[1] + test0[2] * test0[2]);
  double l1 = sqrt(test1[0] * test1[0] + test1[1] * test1[1] + test1[2] * test1[2]);
  double l2 = sqrt(test2[0] * test2[0] + test2[1] * test2[1] + test2[2] * test2[2]);
  double l3 = sqrt(test3[0] * test3[0] + test3[1] * test3[1] + test3[2] * test3[2]);

  if (abs(l0) > tol) e0isV0 = false;
  if (abs(l1) > tol) e0isV1 = false;
  if (abs(l2) > tol) e1isV0 = false;
  if (abs(l3) > tol) e1isV1 = false;

  // ========================================================
  // 1.: nodes on each other
  if (e0isV0 and e1isV1)
  {
    if (out) std::cout << "CASE 1" << std::endl;
    inter_sections().push_back(Mortar::Vertex(edgeVertex0.coord(), Mortar::Vertex::master,
        edgeVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(edgeVertex1.coord(), Mortar::Vertex::master,
        edgeVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 2.: nodes on each other
  else if (e0isV1 and e1isV0)
  {
    if (out) std::cout << "CASE 2" << std::endl;

    inter_sections().push_back(Mortar::Vertex(edgeVertex0.coord(), Mortar::Vertex::master,
        edgeVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(edgeVertex1.coord(), Mortar::Vertex::master,
        edgeVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 3.: e0 on v0 and e1 valid
  else if (e0isV0 and e1v0 and e1v1)
  {
    if (out) std::cout << "CASE 3" << std::endl;

    inter_sections().push_back(Mortar::Vertex(edgeVertex0.coord(), Mortar::Vertex::master,
        edgeVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(edgeVertex1.coord(), Mortar::Vertex::master,
        edgeVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 4.: e1 on v0 and e0 valid
  else if (e1isV0 and e0v0 and e0v1)
  {
    if (out) std::cout << "CASE 4" << std::endl;

    inter_sections().push_back(Mortar::Vertex(edgeVertex0.coord(), Mortar::Vertex::master,
        edgeVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(edgeVertex1.coord(), Mortar::Vertex::master,
        edgeVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 5.: e0 on v1 and e1 valid
  else if (e0isV1 and e1v0 and e1v1)
  {
    if (out) std::cout << "CASE 5" << std::endl;

    inter_sections().push_back(Mortar::Vertex(edgeVertex0.coord(), Mortar::Vertex::master,
        edgeVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(edgeVertex1.coord(), Mortar::Vertex::master,
        edgeVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 6.: e1 on v1 and e0 valid
  else if (e1isV1 and e0v0 and e0v1)
  {
    if (out) std::cout << "CASE 6" << std::endl;

    inter_sections().push_back(Mortar::Vertex(edgeVertex0.coord(), Mortar::Vertex::master,
        edgeVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(edgeVertex1.coord(), Mortar::Vertex::master,
        edgeVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 7.: e0 on v0 and e1 out of v1 but in v0
  else if (e0isV0 and e1v0 and !e1v1)
  {
    if (out) std::cout << "CASE 7" << std::endl;

    inter_sections().push_back(Mortar::Vertex(edgeVertex0.coord(), Mortar::Vertex::master,
        edgeVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(lineVertex1.coord(), Mortar::Vertex::projslave,
        lineVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 8.: e1 on v0 and e0 out of v1 but in v0
  else if (e1isV0 and e0v0 and !e0v1)
  {
    if (out) std::cout << "CASE 8" << std::endl;

    inter_sections().push_back(Mortar::Vertex(edgeVertex1.coord(), Mortar::Vertex::master,
        edgeVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(lineVertex1.coord(), Mortar::Vertex::projslave,
        lineVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 9.: e1 on v1 and e0 out of v0 but in v1
  else if (e1isV1 and !e0v0 and e0v1)
  {
    if (out) std::cout << "CASE 9" << std::endl;

    inter_sections().push_back(Mortar::Vertex(edgeVertex1.coord(), Mortar::Vertex::master,
        edgeVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(lineVertex0.coord(), Mortar::Vertex::projslave,
        lineVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 10.: e0 on v1 and e1 out of v0 but in v1
  else if (e0isV1 and !e1v0 and e1v1)
  {
    if (out) std::cout << "CASE 10" << std::endl;

    inter_sections().push_back(Mortar::Vertex(edgeVertex0.coord(), Mortar::Vertex::master,
        edgeVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(lineVertex0.coord(), Mortar::Vertex::projslave,
        lineVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 11.: e0 on v0 and e1 out of v0 but in v1
  else if (e0isV0 and !e1v0 and e1v1)
  {
    if (out) std::cout << "CASE 11" << std::endl;

    // true because no more intersections expected
    return true;
  }
  // ========================================================
  // 12.: e1 on v0 and e0 out of v0 but in v1
  else if (e1isV0 and !e0v0 and e0v1)
  {
    if (out) std::cout << "CASE 12" << std::endl;

    // true because no more intersections expected
    return true;
  }
  // ========================================================
  // 13.: e0 on v1 and e1 out of v1 but in v0
  else if (e0isV1 and !e1v1 and e1v0)
  {
    if (out) std::cout << "CASE 13" << std::endl;

    // true because no more intersections expected
    return true;
  }
  // ========================================================
  // 14.: e1 on v1 and e0 out of v1 but in v0
  else if (e1isV1 and !e0v1 and e0v0)
  {
    if (out) std::cout << "CASE 14" << std::endl;

    // true because no more intersections expected
    return true;
  }
  // ========================================================
  // 15.: all true --> both intersections master nodes
  else if (e0v1 and e1v1 and e0v0 and e1v0)
  {
    if (out) std::cout << "CASE 15" << std::endl;

    inter_sections().push_back(Mortar::Vertex(edgeVertex0.coord(), Mortar::Vertex::master,
        edgeVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(edgeVertex1.coord(), Mortar::Vertex::master,
        edgeVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 16.: all slave nodes are projected: E0 out of V0  and E1 out of V1
  else if (!e0v0 and e1v0 and e0v1 and !e1v1)
  {
    if (out) std::cout << "CASE 16" << std::endl;

    inter_sections().push_back(Mortar::Vertex(lineVertex0.coord(), Mortar::Vertex::projslave,
        lineVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(lineVertex1.coord(), Mortar::Vertex::projslave,
        lineVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 17.: all slave nodes are projected: E1 out of V0  and E0 out of V1
  else if (e0v0 and !e1v0 and !e0v1 and e1v1)
  {
    if (out) std::cout << "CASE 17" << std::endl;

    inter_sections().push_back(Mortar::Vertex(lineVertex0.coord(), Mortar::Vertex::projslave,
        lineVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(lineVertex1.coord(), Mortar::Vertex::projslave,
        lineVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 18.: mixed: E0 and E1 pos to V0 and E0 pos to V1
  else if (e0v0 and e1v0 and e0v1 and !e1v1)
  {
    if (out) std::cout << "CASE 18" << std::endl;

    inter_sections().push_back(Mortar::Vertex(edgeVertex0.coord(), Mortar::Vertex::master,
        edgeVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(lineVertex1.coord(), Mortar::Vertex::projslave,
        lineVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 19.: mixed: E0 and E1 pos to V0 and E1 pos to V1
  else if (e0v0 and e1v0 and !e0v1 and e1v1)
  {
    if (out) std::cout << "CASE 19" << std::endl;

    inter_sections().push_back(Mortar::Vertex(edgeVertex1.coord(), Mortar::Vertex::master,
        edgeVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(lineVertex1.coord(), Mortar::Vertex::projslave,
        lineVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 20.: mixed: E0 neg and E1 pos to V0 and E0 and E1 pos to V1
  else if (!e0v0 and e1v0 and e0v1 and e1v1)
  {
    if (out) std::cout << "CASE 20" << std::endl;

    inter_sections().push_back(Mortar::Vertex(edgeVertex1.coord(), Mortar::Vertex::master,
        edgeVertex1.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(lineVertex0.coord(), Mortar::Vertex::projslave,
        lineVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 21.: mixed: E1 neg and E0 pos to V0 and E0 and E1 pos to V1
  else if (e0v0 and !e1v0 and e0v1 and e1v1)
  {
    if (out) std::cout << "CASE 21" << std::endl;

    inter_sections().push_back(Mortar::Vertex(edgeVertex0.coord(), Mortar::Vertex::master,
        edgeVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));

    inter_sections().push_back(Mortar::Vertex(lineVertex0.coord(), Mortar::Vertex::projslave,
        lineVertex0.nodeids(), nullptr, nullptr, false, false, nullptr, -1));
  }
  // ========================================================
  // 22.: out: E0 and E1 in V0 and out of V1
  else if (e0v0 and e1v0 and !e0v1 and !e1v1)
  {
    if (out) std::cout << "CASE 22" << std::endl;

    // true because no more intersections expected
    return true;
  }
  // ========================================================
  // 23.: out: E0 and E1 in V1 and out of V0
  else if (!e0v0 and !e1v0 and e0v1 and e1v1)
  {
    if (out) std::cout << "CASE 23" << std::endl;

    // true because no more intersections expected
    return true;
  }
  // ========================================================
  // no valid intersection
  else
  {
    std::cout << "e0isV0 = " << e0isV0 << std::endl;
    std::cout << "e0isV1 = " << e0isV1 << std::endl;
    std::cout << "e1isV0 = " << e1isV0 << std::endl;
    std::cout << "e1isV1 = " << e1isV1 << std::endl;

    std::cout << "e0v0 = " << e0v0 << std::endl;
    std::cout << "e1v0 = " << e1v0 << std::endl;
    std::cout << "e0v1 = " << e0v1 << std::endl;
    std::cout << "e1v1 = " << e1v1 << std::endl;

    FOUR_C_THROW("Something went terribly wrong!");
  }

  return true;
}


/*----------------------------------------------------------------------*
 |  geometric stuff (private)                                farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToSurfaceCoupling3d::line_clipping()
{
  // output variable
  const bool out = false;

  // tolerance for line clipping
  const double sminedge = parent_element().min_edge_size();
  const double mminedge = surface_element().min_edge_size();
  const double tol = MORTARCLIPTOL * std::min(sminedge, mminedge);

  // vector with vertices
  inter_sections().clear();
  temp_inter_sections().clear();

  // safety
  if (master_vertices().size() < 3) FOUR_C_THROW("Invalid number of Master Vertices!");
  if (slave_vertices().size() != 2) FOUR_C_THROW("Invalid number of Slave Vertices!");

  // set previous and next Vertex pointer for all elements in lists
  for (int i = 0; i < (int)master_vertices().size(); ++i)
  {
    // standard case
    if (i != 0 && i != (int)master_vertices().size() - 1)
    {
      master_vertices()[i].assign_next(&master_vertices()[i + 1]);
      master_vertices()[i].assign_prev(&master_vertices()[i - 1]);
    }
    // first element in list
    else if (i == 0)
    {
      master_vertices()[i].assign_next(&master_vertices()[i + 1]);
      master_vertices()[i].assign_prev(&master_vertices()[(int)master_vertices().size() - 1]);
    }
    // last element in list
    else
    {
      master_vertices()[i].assign_next(master_vertices().data());
      master_vertices()[i].assign_prev(&master_vertices()[i - 1]);
    }
  }

  // flip ordering
  std::reverse(slave_vertices().begin(), slave_vertices().end());

  // create line from slave vertices
  std::array<double, 3> slaveLine = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k)
    slaveLine[k] = slave_vertices()[1].coord()[k] - slave_vertices()[0].coord()[k];


  // check for parallelity of line and edges and perform line to line clipping
  bool foundValidParallelity = false;

  // loop over master vertices to create master polygon lines
  for (int j = 0; j < (int)master_vertices().size(); ++j)
  {
    // we need one edge first
    std::array<double, 3> edge = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k)
      edge[k] = (master_vertices()[j].next())->coord()[k] - master_vertices()[j].coord()[k];

    // outward edge normals of polygon and slave line
    std::array<double, 3> np = {0.0, 0.0, 0.0};
    std::array<double, 3> nl = {0.0, 0.0, 0.0};
    np[0] = edge[1] * auxn_surf()[2] - edge[2] * auxn_surf()[1];
    np[1] = edge[2] * auxn_surf()[0] - edge[0] * auxn_surf()[2];
    np[2] = edge[0] * auxn_surf()[1] - edge[1] * auxn_surf()[0];
    nl[0] = slaveLine[1] * auxn_surf()[2] - slaveLine[2] * auxn_surf()[1];
    nl[1] = slaveLine[2] * auxn_surf()[0] - slaveLine[0] * auxn_surf()[2];
    nl[2] = slaveLine[0] * auxn_surf()[1] - slaveLine[1] * auxn_surf()[0];

    if (out)
    {
      std::cout << "==============================================" << std::endl;
      std::cout << "SLine= " << slaveLine[0] << "  " << slaveLine[1] << "  " << slaveLine[2]
                << std::endl;
      std::cout << "Pos1= " << slave_vertices()[0].coord()[0] << "  "
                << slave_vertices()[0].coord()[1] << "  " << slave_vertices()[0].coord()[2]
                << std::endl;
      std::cout << "Pos2= " << slave_vertices()[1].coord()[0] << "  "
                << slave_vertices()[1].coord()[1] << "  " << slave_vertices()[1].coord()[2]
                << std::endl;
      std::cout << "N slave= " << nl[0] << "  " << nl[1] << "  " << nl[2] << std::endl;


      std::cout << "==============================================" << std::endl;
      std::cout << "MEdge= " << edge[0] << "  " << edge[1] << "  " << edge[2] << std::endl;
      std::cout << "Pos1= " << (master_vertices()[j].next())->coord()[0] << "  "
                << (master_vertices()[j].next())->coord()[1] << "  "
                << (master_vertices()[j].next())->coord()[2] << std::endl;
      std::cout << "Pos2= " << master_vertices()[j].coord()[0] << "  "
                << master_vertices()[j].coord()[1] << "  " << master_vertices()[j].coord()[2]
                << std::endl;
      std::cout << "N master= " << np[0] << "  " << np[1] << "  " << np[2] << std::endl;
    }

    // check for parallelity of edges
    double parallel = edge[0] * nl[0] + edge[1] * nl[1] + edge[2] * nl[2];
    if (abs(parallel) < tol)
    {
      // safety checks
      if (master_vertices()[j].next()->nodeids().size() > 1)
        FOUR_C_THROW("Only one node id per master vertex allowed!");
      if (master_vertices()[j].nodeids().size() > 1)
        FOUR_C_THROW("Only one node id per master vertex allowed!");

      // store master node ids in set to guarantee uniqueness
      std::pair<int, int> actIDs = std::pair<int, int>(
          master_vertices()[j].next()->nodeids()[0], master_vertices()[j].nodeids()[0]);
      std::pair<int, int> actIDsTw = std::pair<int, int>(
          master_vertices()[j].nodeids()[0], master_vertices()[j].next()->nodeids()[0]);

      // check if edge on line element
      foundValidParallelity = check_line_on_line(*master_vertices()[j].next(), master_vertices()[j],
          slave_vertices()[1], slave_vertices()[0]);

      // check if processed before
      std::set<std::pair<int, int>>::iterator iter = done_before().find(actIDs);
      std::set<std::pair<int, int>>::iterator itertw = done_before().find(actIDsTw);

      // if not perform clipping of lines
      if (iter == done_before().end() and itertw == done_before().end())
      {
        // add to set of processed nodes
        done_before().insert(actIDs);
        done_before().insert(actIDsTw);

        if (foundValidParallelity)
        {
          // perform line-line clipping
          line_to_line_clipping(*master_vertices()[j].next(), master_vertices()[j],
              slave_vertices()[1], slave_vertices()[0]);

          if (out)
            std::cout << "MASTER IDS = " << master_vertices()[j].next()->nodeids()[0] << "  "
                      << master_vertices()[j].nodeids()[0] << std::endl;
          break;
        }
        else
          continue;
      }
    }
  }  // end master vertex loop

  // if there is a line to line setting --> jump to node check
  if (!foundValidParallelity)
  {
    // loop over master vertices to create master polygon lines
    for (int j = 0; j < (int)master_vertices().size(); ++j)
    {
      // we need two edges first
      std::array<double, 3> edge = {0.0, 0.0, 0.0};
      for (int k = 0; k < 3; ++k)
        edge[k] = (master_vertices()[j].next())->coord()[k] - master_vertices()[j].coord()[k];

      // outward edge normals of polygon and slave line
      std::array<double, 3> np = {0.0, 0.0, 0.0};
      std::array<double, 3> nl = {0.0, 0.0, 0.0};
      np[0] = edge[1] * auxn_surf()[2] - edge[2] * auxn_surf()[1];
      np[1] = edge[2] * auxn_surf()[0] - edge[0] * auxn_surf()[2];
      np[2] = edge[0] * auxn_surf()[1] - edge[1] * auxn_surf()[0];
      nl[0] = slaveLine[1] * auxn_surf()[2] - slaveLine[2] * auxn_surf()[1];
      nl[1] = slaveLine[2] * auxn_surf()[0] - slaveLine[0] * auxn_surf()[2];
      nl[2] = slaveLine[0] * auxn_surf()[1] - slaveLine[1] * auxn_surf()[0];

      if (out)
      {
        std::cout << "==============================================" << std::endl;
        std::cout << "SLine= " << slaveLine[0] << "  " << slaveLine[1] << "  " << slaveLine[2]
                  << std::endl;
        std::cout << "Pos1= " << slave_vertices()[0].coord()[0] << "  "
                  << slave_vertices()[0].coord()[1] << "  " << slave_vertices()[0].coord()[2]
                  << std::endl;
        std::cout << "Pos2= " << slave_vertices()[1].coord()[0] << "  "
                  << slave_vertices()[1].coord()[1] << "  " << slave_vertices()[1].coord()[2]
                  << std::endl;
        std::cout << "N slave= " << nl[0] << "  " << nl[1] << "  " << nl[2] << std::endl;


        std::cout << "==============================================" << std::endl;
        std::cout << "MEdge= " << edge[0] << "  " << edge[1] << "  " << edge[2] << std::endl;
        std::cout << "Pos1= " << (master_vertices()[j].next())->coord()[0] << "  "
                  << (master_vertices()[j].next())->coord()[1] << "  "
                  << (master_vertices()[j].next())->coord()[2] << std::endl;
        std::cout << "Pos2= " << master_vertices()[j].coord()[0] << "  "
                  << master_vertices()[j].coord()[1] << "  " << master_vertices()[j].coord()[2]
                  << std::endl;
        std::cout << "N master= " << np[0] << "  " << np[1] << "  " << np[2] << std::endl;
      }

      // check for parallelity of edges
      double parallel = edge[0] * nl[0] + edge[1] * nl[1] + edge[2] * nl[2];
      if (abs(parallel) < tol)
      {
        continue;
      }

      // check for intersection of non-parallel edges
      double wec_p1 = 0.0;
      double wec_p2 = 0.0;
      for (int k = 0; k < 3; ++k)
      {
        wec_p1 += (slave_vertices()[0].coord()[k] - master_vertices()[j].coord()[k]) * np[k];
        wec_p2 += (slave_vertices()[1].coord()[k] - master_vertices()[j].coord()[k]) * np[k];
      }

      if (out)
      {
        std::cout << "WecP1 = " << wec_p1 << std::endl;
        std::cout << "WecP2 = " << wec_p2 << std::endl;
      }


      // change of sign means we have an intersection!
      if (wec_p1 * wec_p2 <= 0.0)
      {
        double wec_q1 = 0.0;
        double wec_q2 = 0.0;
        for (int k = 0; k < 3; ++k)
        {
          wec_q1 += (master_vertices()[j].coord()[k] - slave_vertices()[0].coord()[k]) * nl[k];
          wec_q2 +=
              ((master_vertices()[j].next())->coord()[k] - slave_vertices()[0].coord()[k]) * nl[k];
        }

        if (out)
        {
          std::cout << "WecQ1 = " << wec_q1 << std::endl;
          std::cout << "WecQ2 = " << wec_q2 << std::endl;
        }

        if (wec_q1 * wec_q2 <= 0.0)
        {
          double alpha = wec_p1 / (wec_p1 - wec_p2);
          double alphaq = wec_q1 / (wec_q1 - wec_q2);

          if (alpha < 0.0 or alpha > 1.0) continue;
          if (alphaq < 0.0 or alphaq > 1.0) continue;

          std::vector<double> coords(3);
          for (int k = 0; k < 3; ++k)
          {
            coords[k] = (1 - alpha) * slave_vertices()[0].coord()[k] +
                        alpha * slave_vertices()[1].coord()[k];
            if (abs(coords[k]) < tol) coords[k] = 0.0;
          }

          if (out)
          {
            std::cout << "Found intersection! (" << j << ") " << alpha << std::endl;
            std::cout << "coords 1: " << coords[0] << " " << coords[1] << " " << coords[2]
                      << std::endl;
          }

          // generate vectors of underlying node ids for lineclip (2x slave, 2x master)
          std::vector<int> lcids(4);
          lcids[0] = (int)(slave_vertices()[0].nodeids()[0]);
          lcids[1] = (int)(slave_vertices()[1].nodeids()[0]);
          lcids[2] = (int)(master_vertices()[j].nodeids()[0]);
          lcids[3] = (int)((master_vertices()[j].next())->nodeids()[0]);

          // store intersection points
          temp_inter_sections().push_back(Mortar::Vertex(coords, Mortar::Vertex::lineclip, lcids,
              &(slave_vertices()[1]), &(slave_vertices()[0]), true, false, nullptr, alpha));
        }
      }
    }  // end vertex loop

    // ===================================================
    // find interior node intersections
    //    if((int)temp_inter_sections().size()!=2)
    {
      for (int i = 0; i < (int)slave_vertices().size(); ++i)
      {
        // keep track of inside / outside status
        bool outside = false;

        // check against all poly1 (slave) edges
        for (int j = 0; j < (int)master_vertices().size(); ++j)
        {
          // we need diff vector and edge2 first
          std::array<double, 3> diff = {0.0, 0.0, 0.0};
          std::array<double, 3> edge = {0.0, 0.0, 0.0};
          for (int k = 0; k < 3; ++k)
          {
            diff[k] = slave_vertices()[i].coord()[k] - master_vertices()[j].coord()[k];
            edge[k] = (master_vertices()[j].next())->coord()[k] - master_vertices()[j].coord()[k];
          }

          // compute distance from point on poly1 to edge
          std::array<double, 3> n = {0.0, 0.0, 0.0};
          n[0] = edge[1] * auxn_surf()[2] - edge[2] * auxn_surf()[1];
          n[1] = edge[2] * auxn_surf()[0] - edge[0] * auxn_surf()[2];
          n[2] = edge[0] * auxn_surf()[1] - edge[1] * auxn_surf()[0];
          double ln = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
          for (int k = 0; k < 3; ++k) n[k] /= ln;

          double dist = diff[0] * n[0] + diff[1] * n[1] + diff[2] * n[2];

          // only keep point if not in outside halfspace
          if (dist - tol > 0.0)  // tends to include nodes
          {
            outside = true;
            break;
          }
        }  // end master loop

        if (outside)
        {
          // next slave vertex
          continue;
        }
        else
        {
          temp_inter_sections().push_back(
              Mortar::Vertex(slave_vertices()[i].coord(), Mortar::Vertex::projslave,
                  slave_vertices()[i].nodeids(), nullptr, nullptr, false, false, nullptr, -1));
        }
      }
    }  // if intersections != 2

    // check positions of all found intersections
    std::vector<int> redundantLocalIDs;
    for (int i = 0; i < (int)temp_inter_sections().size(); ++i)
    {
      for (int j = i; j < (int)temp_inter_sections().size(); ++j)
      {
        // do not check same intersections
        if (i == j) continue;

        // distance vector
        std::array<double, 3> diff = {0.0, 0.0, 0.0};
        for (int k = 0; k < 3; ++k)
          diff[k] = temp_inter_sections()[i].coord()[k] - temp_inter_sections()[j].coord()[k];
        double dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
        if (dist < tol)
        {
          // store redundant id
          redundantLocalIDs.push_back(j);
        }
      }
    }

    std::vector<Mortar::Vertex> aux;
    for (int i = 0; i < (int)temp_inter_sections().size(); ++i)
    {
      bool vanish = false;
      for (int j = 0; j < (int)redundantLocalIDs.size(); ++j)
      {
        if (i == redundantLocalIDs[j]) vanish = true;
      }

      if (!vanish) aux.push_back(temp_inter_sections()[i]);
    }

    // store right vector to TempIntersections
    temp_inter_sections().clear();
    for (int i = 0; i < (int)aux.size(); ++i) temp_inter_sections().push_back(aux[i]);

    // ===================================================
    // check if intersection is close to a node
    for (int i = 0; i < (int)temp_inter_sections().size(); ++i)
    {
      // keep track of comparisons
      bool close = false;

      // check against all poly1 (slave) points
      for (int j = 0; j < (int)slave_vertices().size(); ++j)
      {
        // distance vector
        std::array<double, 3> diff = {0.0, 0.0, 0.0};
        for (int k = 0; k < 3; ++k)
          diff[k] = temp_inter_sections()[i].coord()[k] - slave_vertices()[j].coord()[k];
        double dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

        // only keep intersection point if not close
        if (dist <= tol)
        {
          // intersection is close to slave vertex!
          close = true;

          // store slave vertex as intersection point
          inter_sections().push_back(
              Mortar::Vertex(slave_vertices()[j].coord(), Mortar::Vertex::projslave,
                  slave_vertices()[j].nodeids(), nullptr, nullptr, false, false, nullptr, -1));
          break;
        }
      }

      // do only if no close slave point found
      if (!close)
      {
        // check against all poly2 (master) points
        for (int j = 0; j < (int)master_vertices().size(); ++j)
        {
          // distance vector
          std::array<double, 3> diff = {0.0, 0.0, 0.0};
          for (int k = 0; k < 3; ++k)
            diff[k] = temp_inter_sections()[i].coord()[k] - master_vertices()[j].coord()[k];
          double dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

          // only keep intersection point if not close
          if (dist <= tol)
          {
            // intersection is close to master vertex!
            close = true;

            inter_sections().push_back(
                Mortar::Vertex(master_vertices()[j].coord(), Mortar::Vertex::master,
                    master_vertices()[j].nodeids(), nullptr, nullptr, false, false, nullptr, -1));
            break;
          }
        }
      }

      // keep intersection point only if not close to any Slave/Master point
      if (!close) inter_sections().push_back(temp_inter_sections()[i]);
    }
  }  // end found valid parallity

  // 2. check plausibility
  if (inter_sections().size() > 2)
  {
    std::cout << "Intersections= " << inter_sections().size() << std::endl;
    FOUR_C_THROW("intersections not possible!!!");
  }

  return;
}

/*----------------------------------------------------------------------*
 |  create integration lines                                 farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToSurfaceCoupling3d::create_integration_lines(
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& linvertex)
{
  // get coordinates
  Core::LinAlg::Matrix<3, 3> coords;

  for (int i = 0; i < 3; ++i)
  {
    coords(i, 0) = inter_sections()[0].coord()[i];
    coords(i, 1) = inter_sections()[1].coord()[i];
    coords(i, 2) = -1;  // dummy;
  }

  // create Integration Line
  int_line() = std::make_shared<Mortar::IntCell>(parent_element().id(), 2, coords, auxn(),
      Core::FE::CellType::line2, linvertex[0], linvertex[1],
      linvertex[1],  // dummy
      get_deriv_auxn());

  return;
}

/*----------------------------------------------------------------------*
 |  linearize vertices                                       farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToSurfaceCoupling3d::linearize_vertices(
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& linvertex)
{
  // linearize all aux.plane slave and master nodes only ONCE
  // and use these linearizations later during lineclip linearization
  // (this speeds up the vertex linearizations in most cases, as we
  // never linearize the SAME slave or master vertex more than once)

  // number of nodes
  const int nsrows = line_element()->num_node();
  const int nmrows = surface_element().num_node();

  // prepare storage for slave and master linearizations
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> linsnodes(nsrows,
      std::vector<Core::Gen::Pairedvector<int, double>>(
          3, 100 + linsize_ + 3 * line_element()->num_node() + 3 * surface_element().num_node()));
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> linmnodes(nmrows,
      std::vector<Core::Gen::Pairedvector<int, double>>(
          3, 100 + linsize_ + 3 * line_element()->num_node() + 3 * surface_element().num_node()));

  // compute slave linearizations (nsrows)
  slave_vertex_linearization(linsnodes);

  // compute master linearizations (nmrows)
  master_vertex_linearization(linmnodes);

  //**********************************************************************
  // Line vertex linearization
  //**********************************************************************
  // loop over all clip Intersections vertices
  for (int i = 0; i < (int)inter_sections().size(); ++i)
  {
    // references to current vertex and its linearization
    Mortar::Vertex& currv = inter_sections()[i];
    std::vector<Core::Gen::Pairedvector<int, double>>& currlin = linvertex[i];

    // decision on vertex type (slave, projmaster, linclip)
    if (currv.v_type() == Mortar::Vertex::projslave)
    {
      // get corresponding slave id
      int sid = currv.nodeids()[0];

      // find corresponding slave node linearization
      int k = 0;
      while (k < nsrows)
      {
        if (line_element()->node_ids()[k] == sid) break;
        ++k;
      }

      // FOUR_C_THROW if not found
      if (k == nsrows) FOUR_C_THROW("Slave Id not found!");

      // get the correct slave node linearization
      currlin = linsnodes[k];
    }
    else if (currv.v_type() == Mortar::Vertex::master)
    {
      // get corresponding master id
      int mid = currv.nodeids()[0];

      // find corresponding master node linearization
      int k = 0;
      while (k < nmrows)
      {
        if (surface_element().node_ids()[k] == mid) break;
        ++k;
      }

      // FOUR_C_THROW if not found
      if (k == nmrows) FOUR_C_THROW("Master Id not found!");

      // get the correct master node linearization
      currlin = linmnodes[k];
    }
    else if (currv.v_type() == Mortar::Vertex::lineclip)
    {
      // get references to the two slave vertices
      int sindex1 = -1;
      int sindex2 = -1;
      for (int j = 0; j < (int)slave_vertices().size(); ++j)
      {
        if (slave_vertices()[j].nodeids()[0] == currv.nodeids()[0]) sindex1 = j;
        if (slave_vertices()[j].nodeids()[0] == currv.nodeids()[1]) sindex2 = j;
      }
      if (sindex1 < 0 || sindex2 < 0 || sindex1 == sindex2)
        FOUR_C_THROW("Lineclip linearization: (S) Something went wrong!");

      Mortar::Vertex* sv1 = &slave_vertices()[sindex1];
      Mortar::Vertex* sv2 = &slave_vertices()[sindex2];

      // get references to the two master vertices
      int mindex1 = -1;
      int mindex2 = -1;
      for (int j = 0; j < (int)master_vertices().size(); ++j)
      {
        if (master_vertices()[j].nodeids()[0] == currv.nodeids()[2]) mindex1 = j;
        if (master_vertices()[j].nodeids()[0] == currv.nodeids()[3]) mindex2 = j;
      }
      if (mindex1 < 0 || mindex2 < 0 || mindex1 == mindex2)
        FOUR_C_THROW("Lineclip linearization: (M) Something went wrong!");

      Mortar::Vertex* mv1 = &master_vertices()[mindex1];
      Mortar::Vertex* mv2 = &master_vertices()[mindex2];

      // do lineclip vertex linearization
      lineclip_vertex_linearization(currv, currlin, sv1, sv2, mv1, mv2, linsnodes, linmnodes);
    }

    else
      FOUR_C_THROW("VertexLinearization: Invalid Vertex Type!");
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Linearization of lineclip vertex (3D) AuxPlane            popp 03/09|
 *----------------------------------------------------------------------*/
void CONTACT::LineToSurfaceCoupling3d::lineclip_vertex_linearization(Mortar::Vertex& currv,
    std::vector<Core::Gen::Pairedvector<int, double>>& currlin, Mortar::Vertex* sv1,
    Mortar::Vertex* sv2, Mortar::Vertex* mv1, Mortar::Vertex* mv2,
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& linsnodes,
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& linmnodes)
{
  // number of nodes
  const int nsrows = line_element()->num_node();
  const int nmrows = surface_element().num_node();

  // iterator
  typedef Core::Gen::Pairedvector<int, double>::const_iterator _CI;

  // compute factor Z
  std::array<double, 3> crossZ = {0.0, 0.0, 0.0};
  crossZ[0] = (sv1->coord()[1] - mv1->coord()[1]) * (mv2->coord()[2] - mv1->coord()[2]) -
              (sv1->coord()[2] - mv1->coord()[2]) * (mv2->coord()[1] - mv1->coord()[1]);
  crossZ[1] = (sv1->coord()[2] - mv1->coord()[2]) * (mv2->coord()[0] - mv1->coord()[0]) -
              (sv1->coord()[0] - mv1->coord()[0]) * (mv2->coord()[2] - mv1->coord()[2]);
  crossZ[2] = (sv1->coord()[0] - mv1->coord()[0]) * (mv2->coord()[1] - mv1->coord()[1]) -
              (sv1->coord()[1] - mv1->coord()[1]) * (mv2->coord()[0] - mv1->coord()[0]);
  double Zfac = crossZ[0] * auxn()[0] + crossZ[1] * auxn()[1] + crossZ[2] * auxn()[2];

  // compute factor N
  std::array<double, 3> crossN = {0.0, 0.0, 0.0};
  crossN[0] = (sv2->coord()[1] - sv1->coord()[1]) * (mv2->coord()[2] - mv1->coord()[2]) -
              (sv2->coord()[2] - sv1->coord()[2]) * (mv2->coord()[1] - mv1->coord()[1]);
  crossN[1] = (sv2->coord()[2] - sv1->coord()[2]) * (mv2->coord()[0] - mv1->coord()[0]) -
              (sv2->coord()[0] - sv1->coord()[0]) * (mv2->coord()[2] - mv1->coord()[2]);
  crossN[2] = (sv2->coord()[0] - sv1->coord()[0]) * (mv2->coord()[1] - mv1->coord()[1]) -
              (sv2->coord()[1] - sv1->coord()[1]) * (mv2->coord()[0] - mv1->coord()[0]);
  double Nfac = crossN[0] * auxn()[0] + crossN[1] * auxn()[1] + crossN[2] * auxn()[2];

  // slave edge vector
  std::array<double, 3> sedge = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k) sedge[k] = sv2->coord()[k] - sv1->coord()[k];

  // prepare linearization derivZ
  std::array<double, 3> crossdZ1 = {0.0, 0.0, 0.0};
  std::array<double, 3> crossdZ2 = {0.0, 0.0, 0.0};
  std::array<double, 3> crossdZ3 = {0.0, 0.0, 0.0};
  crossdZ1[0] = (mv2->coord()[1] - mv1->coord()[1]) * auxn()[2] -
                (mv2->coord()[2] - mv1->coord()[2]) * auxn()[1];
  crossdZ1[1] = (mv2->coord()[2] - mv1->coord()[2]) * auxn()[0] -
                (mv2->coord()[0] - mv1->coord()[0]) * auxn()[2];
  crossdZ1[2] = (mv2->coord()[0] - mv1->coord()[0]) * auxn()[1] -
                (mv2->coord()[1] - mv1->coord()[1]) * auxn()[0];
  crossdZ2[0] = auxn()[1] * (sv1->coord()[2] - mv1->coord()[2]) -
                auxn()[2] * (sv1->coord()[1] - mv1->coord()[1]);
  crossdZ2[1] = auxn()[2] * (sv1->coord()[0] - mv1->coord()[0]) -
                auxn()[0] * (sv1->coord()[2] - mv1->coord()[2]);
  crossdZ2[2] = auxn()[0] * (sv1->coord()[1] - mv1->coord()[1]) -
                auxn()[1] * (sv1->coord()[0] - mv1->coord()[0]);
  crossdZ3[0] = (sv1->coord()[1] - mv1->coord()[1]) * (mv2->coord()[2] - mv1->coord()[2]) -
                (sv1->coord()[2] - mv1->coord()[2]) * (mv2->coord()[1] - mv1->coord()[1]);
  crossdZ3[1] = (sv1->coord()[2] - mv1->coord()[2]) * (mv2->coord()[0] - mv1->coord()[0]) -
                (sv1->coord()[0] - mv1->coord()[0]) * (mv2->coord()[2] - mv1->coord()[2]);
  crossdZ3[2] = (sv1->coord()[0] - mv1->coord()[0]) * (mv2->coord()[1] - mv1->coord()[1]) -
                (sv1->coord()[1] - mv1->coord()[1]) * (mv2->coord()[0] - mv1->coord()[0]);

  // prepare linearization derivN
  std::array<double, 3> crossdN1 = {0.0, 0.0, 0.0};
  std::array<double, 3> crossdN2 = {0.0, 0.0, 0.0};
  std::array<double, 3> crossdN3 = {0.0, 0.0, 0.0};
  crossdN1[0] = (mv2->coord()[1] - mv1->coord()[1]) * auxn()[2] -
                (mv2->coord()[2] - mv1->coord()[2]) * auxn()[1];
  crossdN1[1] = (mv2->coord()[2] - mv1->coord()[2]) * auxn()[0] -
                (mv2->coord()[0] - mv1->coord()[0]) * auxn()[2];
  crossdN1[2] = (mv2->coord()[0] - mv1->coord()[0]) * auxn()[1] -
                (mv2->coord()[1] - mv1->coord()[1]) * auxn()[0];
  crossdN2[0] = auxn()[1] * (sv2->coord()[2] - sv1->coord()[2]) -
                auxn()[2] * (sv2->coord()[1] - sv1->coord()[1]);
  crossdN2[1] = auxn()[2] * (sv2->coord()[0] - sv1->coord()[0]) -
                auxn()[0] * (sv2->coord()[2] - sv1->coord()[2]);
  crossdN2[2] = auxn()[0] * (sv2->coord()[1] - sv1->coord()[1]) -
                auxn()[1] * (sv2->coord()[0] - sv1->coord()[0]);
  crossdN3[0] = (sv2->coord()[1] - sv1->coord()[1]) * (mv2->coord()[2] - mv1->coord()[2]) -
                (sv2->coord()[2] - sv1->coord()[2]) * (mv2->coord()[1] - mv1->coord()[1]);
  crossdN3[1] = (sv2->coord()[2] - sv1->coord()[2]) * (mv2->coord()[0] - mv1->coord()[0]) -
                (sv2->coord()[0] - sv1->coord()[0]) * (mv2->coord()[2] - mv1->coord()[2]);
  crossdN3[2] = (sv2->coord()[0] - sv1->coord()[0]) * (mv2->coord()[1] - mv1->coord()[1]) -
                (sv2->coord()[1] - sv1->coord()[1]) * (mv2->coord()[0] - mv1->coord()[0]);

  // slave vertex linearization (2x)
  int sid1 = currv.nodeids()[0];
  int sid2 = currv.nodeids()[1];

  // find corresponding slave node linearizations
  int k = 0;
  while (k < nsrows)
  {
    if (line_element()->node_ids()[k] == sid1) break;
    ++k;
  }

  // FOUR_C_THROW if not found
  if (k == nsrows) FOUR_C_THROW("Slave Id1 not found!");

  // get the correct slave node linearization
  std::vector<Core::Gen::Pairedvector<int, double>>& slavelin0 = linsnodes[k];

  k = 0;
  while (k < nsrows)
  {
    if (line_element()->node_ids()[k] == sid2) break;
    ++k;
  }

  // FOUR_C_THROW if not found
  if (k == nsrows) FOUR_C_THROW("Slave Id2 not found!");

  // get the correct slave node linearization
  std::vector<Core::Gen::Pairedvector<int, double>>& slavelin1 = linsnodes[k];

  // master vertex linearization (2x)
  int mid1 = currv.nodeids()[2];
  int mid2 = currv.nodeids()[3];

  // find corresponding master node linearizations
  k = 0;
  while (k < nmrows)
  {
    if (surface_element().node_ids()[k] == mid1) break;
    ++k;
  }

  // FOUR_C_THROW if not found
  if (k == nmrows) FOUR_C_THROW("Master Id1 not found!");

  // get the correct master node linearization
  std::vector<Core::Gen::Pairedvector<int, double>>& masterlin0 = linmnodes[k];

  k = 0;
  while (k < nmrows)
  {
    if (surface_element().node_ids()[k] == mid2) break;
    ++k;
  }

  // FOUR_C_THROW if not found
  if (k == nmrows) FOUR_C_THROW("Master Id2 not found!");

  // get the correct master node linearization
  std::vector<Core::Gen::Pairedvector<int, double>>& masterlin1 = linmnodes[k];

  // linearization of element normal Auxn()
  std::vector<Core::Gen::Pairedvector<int, double>>& linauxn = get_deriv_auxn();

  const double ZNfac = Zfac / Nfac;
  const double ZNNfac = Zfac / (Nfac * Nfac);
  const double Nfacinv = 1.0 / Nfac;

  // bring everything together -> lineclip vertex linearization
  for (int k = 0; k < 3; ++k)
  {
    for (_CI p = slavelin0[k].begin(); p != slavelin0[k].end(); ++p)
    {
      currlin[k][p->first] += (p->second);
      currlin[k][p->first] += ZNfac * (p->second);
      for (int dim = 0; dim < 3; ++dim)
      {
        currlin[dim][p->first] -= sedge[dim] * Nfacinv * crossdZ1[k] * (p->second);
        currlin[dim][p->first] -= sedge[dim] * ZNNfac * crossdN1[k] * (p->second);
      }
    }
    for (_CI p = slavelin1[k].begin(); p != slavelin1[k].end(); ++p)
    {
      currlin[k][p->first] -= ZNfac * (p->second);
      for (int dim = 0; dim < 3; ++dim)
      {
        currlin[dim][p->first] += sedge[dim] * ZNNfac * crossdN1[k] * (p->second);
      }
    }
    for (_CI p = masterlin0[k].begin(); p != masterlin0[k].end(); ++p)
    {
      for (int dim = 0; dim < 3; ++dim)
      {
        currlin[dim][p->first] += sedge[dim] * Nfacinv * crossdZ1[k] * (p->second);
        currlin[dim][p->first] += sedge[dim] * Nfacinv * crossdZ2[k] * (p->second);
        currlin[dim][p->first] -= sedge[dim] * ZNNfac * crossdN2[k] * (p->second);
      }
    }
    for (_CI p = masterlin1[k].begin(); p != masterlin1[k].end(); ++p)
    {
      for (int dim = 0; dim < 3; ++dim)
      {
        currlin[dim][p->first] -= sedge[dim] * Nfacinv * crossdZ2[k] * (p->second);
        currlin[dim][p->first] += sedge[dim] * ZNNfac * crossdN2[k] * (p->second);
      }
    }
    for (_CI p = linauxn[k].begin(); p != linauxn[k].end(); ++p)
    {
      for (int dim = 0; dim < 3; ++dim)
      {
        currlin[dim][p->first] -= sedge[dim] * Nfacinv * crossdZ3[k] * (p->second);
        currlin[dim][p->first] += sedge[dim] * ZNNfac * crossdN3[k] * (p->second);
      }
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  compute and check length of intLine                      farah 07/16|
 *----------------------------------------------------------------------*/
bool CONTACT::LineToSurfaceCoupling3d::check_length()
{
  // tolerance
  const double sminedge = parent_element().min_edge_size();
  const double mminedge = surface_element().min_edge_size();
  const double tol = MORTARCLIPTOL * std::min(sminedge, mminedge);

  // compute distance vector
  std::array<double, 3> v = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i)
    v[i] = inter_sections()[0].coord()[i] - inter_sections()[1].coord()[i];

  // compute length
  double length = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

  // check
  if (length < tol) return false;

  return true;
}

/*----------------------------------------------------------------------*
 |  eval (public)                                            farah 07/16|
 *----------------------------------------------------------------------*/
bool CONTACT::LineToSurfaceCoupling3d::auxiliary_plane()
{
  // we first need the element center:
  // for quad4, quad8, quad9 elements: xi = eta = 0.0
  // for tri3, tri6 elements: xi = eta = 1/3
  double loccenter[2] = {0.0, 0.0};

  Core::FE::CellType dt = surface_element().shape();
  if (dt == Core::FE::CellType::tri3 || dt == Core::FE::CellType::tri6)
  {
    loccenter[0] = 1.0 / 3.0;
    loccenter[1] = 1.0 / 3.0;
  }
  else if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
           dt == Core::FE::CellType::quad9)
  {
    loccenter[0] = 0.0;
    loccenter[1] = 0.0;
  }
  else
    FOUR_C_THROW("auxiliary_plane called for unknown element type");

  // compute element center via shape fct. interpolation
  surface_element().local_to_global(loccenter, auxc(), 0);

  // we then compute the unit normal vector at the element center
  lauxn() = surface_element().compute_unit_normal_at_xi(loccenter, auxn_surf());
  //
  //  // compute aux normal linearization
  //  surface_element().DerivUnitNormalAtXi(loccenter, get_deriv_auxn());

  // bye
  return true;
}

/*----------------------------------------------------------------------*
 |  create auxiliary line + normal                           farah 08/16|
 *----------------------------------------------------------------------*/
bool CONTACT::LineToSurfaceCoupling3d::auxiliary_line()
{
  typedef Core::Gen::Pairedvector<int, double>::const_iterator _CI;

  int nnodes = line_element()->num_node();
  if (nnodes != 2) FOUR_C_THROW("Auxiliary line calculation only for line2 elements!");

  // average nodal normals of line element
  linsize_ = 0;
  for (int i = 0; i < nnodes; ++i)
  {
    Core::Nodes::Node* node = idiscret_.g_node(line_element()->node_ids()[i]);
    if (!node) FOUR_C_THROW("Cannot find slave element with gid %", line_element()->node_ids()[i]);
    Node* mycnode = dynamic_cast<Node*>(node);
    if (!mycnode) FOUR_C_THROW("project_slave: Null pointer!");

    linsize_ += mycnode->get_linsize();
  }

  // TODO: this is a safety scaling. Correct linsize
  //      should be predicted
  linsize_ *= 100;

  // auxiliary normal
  get_deriv_auxn().resize(3, linsize_ * 10);

  // auxiliary center
  get_deriv_auxc().resize(3, linsize_ * 10);

  auxc()[0] = 0.0;
  auxc()[1] = 0.0;
  auxc()[2] = 0.0;

  std::vector<Core::Gen::Pairedvector<int, double>> dauxn(3, 100);

  // average nodal normals of line element
  for (int i = 0; i < nnodes; ++i)
  {
    Core::Nodes::Node* node = idiscret_.g_node(line_element()->node_ids()[i]);
    if (!node) FOUR_C_THROW("Cannot find slave element with gid %", line_element()->node_ids()[i]);
    Node* mycnode = dynamic_cast<Node*>(node);
    if (!mycnode) FOUR_C_THROW("project_slave: Null pointer!");

    auxn()[0] += 0.5 * mycnode->mo_data().n()[0];
    auxn()[1] += 0.5 * mycnode->mo_data().n()[1];
    auxn()[2] += 0.5 * mycnode->mo_data().n()[2];

    for (_CI p = mycnode->data().get_deriv_n()[0].begin();
        p != mycnode->data().get_deriv_n()[0].end(); ++p)
      (dauxn[0])[p->first] += 0.5 * (p->second);
    for (_CI p = mycnode->data().get_deriv_n()[1].begin();
        p != mycnode->data().get_deriv_n()[1].end(); ++p)
      (dauxn[1])[p->first] += 0.5 * (p->second);
    for (_CI p = mycnode->data().get_deriv_n()[2].begin();
        p != mycnode->data().get_deriv_n()[2].end(); ++p)
      (dauxn[2])[p->first] += 0.5 * (p->second);

    // new aux center
    for (int d = 0; d < n_dim(); ++d) auxc()[d] += 0.5 * mycnode->xspatial()[d];

    (get_deriv_auxc()[0])[mycnode->dofs()[0]] += 0.5;
    (get_deriv_auxc()[1])[mycnode->dofs()[1]] += 0.5;
    (get_deriv_auxc()[2])[mycnode->dofs()[2]] += 0.5;
  }

  // create tangent of line element
  std::array<double, 3> tangent = {0.0, 0.0, 0.0};
  Core::Nodes::Node* node = idiscret_.g_node(line_element()->node_ids()[0]);
  if (!node) FOUR_C_THROW("Cannot find slave element with gid %", line_element()->node_ids()[0]);
  Node* mycnode = dynamic_cast<Node*>(node);
  if (!mycnode) FOUR_C_THROW("project_slave: Null pointer!");

  tangent[0] += mycnode->xspatial()[0];
  tangent[1] += mycnode->xspatial()[1];
  tangent[2] += mycnode->xspatial()[2];

  Core::Nodes::Node* node2 = idiscret_.g_node(line_element()->node_ids()[1]);
  if (!node2) FOUR_C_THROW("Cannot find slave element with gid %", line_element()->node_ids()[1]);
  Node* mycnode2 = dynamic_cast<Node*>(node2);
  if (!mycnode2) FOUR_C_THROW("project_slave: Null pointer!");

  tangent[0] -= mycnode2->xspatial()[0];
  tangent[1] -= mycnode2->xspatial()[1];
  tangent[2] -= mycnode2->xspatial()[2];

  Core::LinAlg::SerialDenseMatrix tanplane(3, 3);
  tanplane(0, 0) = 1 - (tangent[0] * tangent[0]);
  tanplane(0, 1) = -(tangent[0] * tangent[1]);
  tanplane(0, 2) = -(tangent[0] * tangent[2]);
  tanplane(1, 0) = -(tangent[1] * tangent[0]);
  tanplane(1, 1) = 1 - (tangent[1] * tangent[1]);
  tanplane(1, 2) = -(tangent[1] * tangent[2]);
  tanplane(2, 0) = -(tangent[2] * tangent[0]);
  tanplane(2, 1) = -(tangent[2] * tangent[1]);
  tanplane(2, 2) = 1 - (tangent[2] * tangent[2]);

  std::array<double, 3> finalauxn = {0.0, 0.0, 0.0};
  finalauxn[0] =
      tanplane(0, 0) * auxn()[0] + tanplane(0, 1) * auxn()[1] + tanplane(0, 2) * auxn()[2];
  finalauxn[1] =
      tanplane(1, 0) * auxn()[0] + tanplane(1, 1) * auxn()[1] + tanplane(1, 2) * auxn()[2];
  finalauxn[2] =
      tanplane(2, 0) * auxn()[0] + tanplane(2, 1) * auxn()[1] + tanplane(2, 2) * auxn()[2];

  // lin tangent
  std::vector<Core::Gen::Pairedvector<int, double>> dnmap_unit(3, linsize_ * 10);
  for (int i = 0; i < n_dim(); ++i)
  {
    dnmap_unit[i][mycnode->dofs()[i]] += 1;
    dnmap_unit[i][mycnode2->dofs()[i]] -= 1;
  }


  std::vector<Core::Gen::Pairedvector<int, double>> tplanex(3, linsize_ * 10);
  std::vector<Core::Gen::Pairedvector<int, double>> tplaney(3, linsize_ * 10);
  std::vector<Core::Gen::Pairedvector<int, double>> tplanez(3, linsize_ * 10);

  for (_CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    tplanex[0][p->first] -= tangent[0] * p->second;
  for (_CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    tplanex[1][p->first] -= tangent[1] * p->second;
  for (_CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    tplanex[2][p->first] -= tangent[2] * p->second;

  for (_CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    tplaney[0][p->first] -= tangent[0] * p->second;
  for (_CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    tplaney[1][p->first] -= tangent[1] * p->second;
  for (_CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    tplaney[2][p->first] -= tangent[2] * p->second;

  for (_CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    tplanez[0][p->first] -= tangent[0] * p->second;
  for (_CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    tplanez[1][p->first] -= tangent[1] * p->second;
  for (_CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    tplanez[2][p->first] -= tangent[2] * p->second;

  //------------
  for (_CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    tplanex[0][p->first] -= tangent[0] * p->second;
  for (_CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    tplanex[1][p->first] -= tangent[0] * p->second;
  for (_CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    tplanex[2][p->first] -= tangent[0] * p->second;

  for (_CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    tplaney[0][p->first] -= tangent[1] * p->second;
  for (_CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    tplaney[1][p->first] -= tangent[1] * p->second;
  for (_CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    tplaney[2][p->first] -= tangent[1] * p->second;

  for (_CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    tplanez[0][p->first] -= tangent[2] * p->second;
  for (_CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    tplanez[1][p->first] -= tangent[2] * p->second;
  for (_CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    tplanez[2][p->first] -= tangent[2] * p->second;



  for (_CI p = dauxn[0].begin(); p != dauxn[0].end(); ++p)
    get_deriv_auxn()[0][p->first] += tanplane(0, 0) * p->second;
  for (_CI p = dauxn[1].begin(); p != dauxn[1].end(); ++p)
    get_deriv_auxn()[0][p->first] += tanplane(0, 1) * p->second;
  for (_CI p = dauxn[2].begin(); p != dauxn[2].end(); ++p)
    get_deriv_auxn()[0][p->first] += tanplane(0, 2) * p->second;

  for (_CI p = dauxn[0].begin(); p != dauxn[0].end(); ++p)
    get_deriv_auxn()[1][p->first] += tanplane(1, 0) * p->second;
  for (_CI p = dauxn[1].begin(); p != dauxn[1].end(); ++p)
    get_deriv_auxn()[1][p->first] += tanplane(1, 1) * p->second;
  for (_CI p = dauxn[2].begin(); p != dauxn[2].end(); ++p)
    get_deriv_auxn()[1][p->first] += tanplane(1, 2) * p->second;

  for (_CI p = dauxn[0].begin(); p != dauxn[0].end(); ++p)
    get_deriv_auxn()[2][p->first] += tanplane(2, 0) * p->second;
  for (_CI p = dauxn[1].begin(); p != dauxn[1].end(); ++p)
    get_deriv_auxn()[2][p->first] += tanplane(2, 1) * p->second;
  for (_CI p = dauxn[2].begin(); p != dauxn[2].end(); ++p)
    get_deriv_auxn()[2][p->first] += tanplane(2, 2) * p->second;

  //-----------------------------
  for (_CI p = tplanex[0].begin(); p != tplanex[0].end(); ++p)
    get_deriv_auxn()[0][p->first] += auxn()[0] * p->second;
  for (_CI p = tplanex[1].begin(); p != tplanex[1].end(); ++p)
    get_deriv_auxn()[0][p->first] += auxn()[1] * p->second;
  for (_CI p = tplanex[2].begin(); p != tplanex[2].end(); ++p)
    get_deriv_auxn()[0][p->first] += auxn()[2] * p->second;

  for (_CI p = tplaney[0].begin(); p != tplaney[0].end(); ++p)
    get_deriv_auxn()[1][p->first] += auxn()[0] * p->second;
  for (_CI p = tplaney[1].begin(); p != tplaney[1].end(); ++p)
    get_deriv_auxn()[1][p->first] += auxn()[1] * p->second;
  for (_CI p = tplaney[2].begin(); p != tplaney[2].end(); ++p)
    get_deriv_auxn()[1][p->first] += auxn()[2] * p->second;

  for (_CI p = tplanez[0].begin(); p != tplanez[0].end(); ++p)
    get_deriv_auxn()[2][p->first] += auxn()[0] * p->second;
  for (_CI p = tplanez[1].begin(); p != tplanez[1].end(); ++p)
    get_deriv_auxn()[2][p->first] += auxn()[1] * p->second;
  for (_CI p = tplanez[2].begin(); p != tplanez[2].end(); ++p)
    get_deriv_auxn()[2][p->first] += auxn()[2] * p->second;


  auxn()[0] = finalauxn[0];
  auxn()[1] = finalauxn[1];
  auxn()[2] = finalauxn[2];

  auxn_surf()[0] = -auxn()[0];
  auxn_surf()[1] = -auxn()[1];
  auxn_surf()[2] = -auxn()[2];

  return true;
}

/*----------------------------------------------------------------------*
 |  eval (public)                                            farah 07/16|
 *----------------------------------------------------------------------*/
bool CONTACT::LineToSurfaceCoupling3d::has_proj_status() { return true; }


/*----------------------------------------------------------------------*
 |  eval (public)                                            farah 07/16|
 *----------------------------------------------------------------------*/
bool CONTACT::LineToSurfaceCoupling3d::project_slave()
{
  // project slave nodes onto auxiliary plane
  int nnodes = line_element()->num_node();

  // initialize storage for slave coords + their ids
  std::vector<double> vertices(3);
  std::vector<int> snodeids(1);

  for (int i = 0; i < nnodes; ++i)
  {
    Core::Nodes::Node* node = idiscret_.g_node(line_element()->node_ids()[i]);
    if (!node) FOUR_C_THROW("Cannot find slave element with gid %", line_element()->node_ids()[i]);
    Node* mycnode = dynamic_cast<Node*>(node);
    if (!mycnode) FOUR_C_THROW("project_slave: Null pointer!");

    // first build difference of point and element center
    // and then dot product with unit normal at center
    const double dist = (mycnode->xspatial()[0] - auxc()[0]) * auxn()[0] +
                        (mycnode->xspatial()[1] - auxc()[1]) * auxn()[1] +
                        (mycnode->xspatial()[2] - auxc()[2]) * auxn()[2];

    // compute projection
    for (int k = 0; k < 3; ++k) vertices[k] = mycnode->xspatial()[k] - dist * auxn()[k];

    // get node id, too
    snodeids[0] = mycnode->id();

    // store into vertex data structure
    slave_vertices().push_back(Mortar::Vertex(vertices, Mortar::Vertex::projslave, snodeids,
        nullptr, nullptr, false, false, nullptr, -1.0));
  }
  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of slave vertex (3D) AuxPlane              farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToSurfaceCoupling3d::slave_vertex_linearization(
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& currlin)
{
  // we first need the slave element center:
  // for quad4, quad8, quad9 elements: xi = eta = 0.0
  // for tri3, tri6 elements: xi = eta = 1/3
  double scxi[2];

  Core::FE::CellType dt = surface_element().shape();
  if (dt == Core::FE::CellType::tri3 || dt == Core::FE::CellType::tri6)
  {
    scxi[0] = 1.0 / 3.0;
    scxi[1] = 1.0 / 3.0;
  }
  else if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
           dt == Core::FE::CellType::quad9)
  {
    scxi[0] = 0.0;
    scxi[1] = 0.0;
  }
  else
    FOUR_C_THROW("master_vertex_linearization called for unknown element type");

  // evlauate shape functions + derivatives at scxi
  int nrow = surface_element().num_node();
  Core::LinAlg::SerialDenseVector sval(nrow);
  Core::LinAlg::SerialDenseMatrix sderiv(nrow, 2, true);
  surface_element().evaluate_shape(scxi, sval, sderiv, nrow);

  // we need all participating slave nodes
  Core::Nodes::Node** snodes = surface_element().nodes();
  std::vector<Mortar::Node*> smrtrnodes(nrow);

  for (int i = 0; i < nrow; ++i)
  {
    smrtrnodes[i] = dynamic_cast<Mortar::Node*>(snodes[i]);
    if (!smrtrnodes[i]) FOUR_C_THROW("master_vertex_linearization: Null pointer!");
  }

  // linearization of the SlaveIntEle spatial coords
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> snodelin(0);

  // resize the linearizations
  snodelin.resize(nrow, std::vector<Core::Gen::Pairedvector<int, double>>(3, 1));

  // loop over all intEle nodes
  for (int in = 0; in < nrow; ++in)
    for (int dim = 0; dim < 3; ++dim) snodelin[in][dim][smrtrnodes[in]->dofs()[dim]] += 1.;

  // map iterator
  typedef Core::Gen::Pairedvector<int, double>::const_iterator
      _CI;  // linearization of element center Auxc()
  //  std::vector<Core::Gen::Pairedvector<int  ,double> >
  //  linauxc(3,10*surface_element().num_node());
  //  // assume 3 dofs per node

  //  for (int i = 0; i < nrow; ++i)
  //      for (int dim=0; dim<3; ++dim)
  //        for (_CI p=snodelin[i][dim].begin(); p!=snodelin[i][dim].end(); ++p)
  //          linauxc[dim][p->first] = sval[i]*p->second;

  // linearization of element normal Auxn()
  std::vector<Core::Gen::Pairedvector<int, double>>& linauxn = get_deriv_auxn();

  // linearization of the MasterIntEle spatial coords
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> mnodelin(0);

  // resize the linearizations
  mnodelin.resize(
      line_element()->num_node(), std::vector<Core::Gen::Pairedvector<int, double>>(3, 1));

  // loop over all intEle nodes
  for (int in = 0; in < line_element()->num_node(); ++in)
  {
    Mortar::Node* mrtrmnode =
        dynamic_cast<Mortar::Node*>(idiscret_.g_node(line_element()->node_ids()[in]));
    if (mrtrmnode == nullptr) FOUR_C_THROW("dynamic cast to mortar node went wrong");

    for (int dim = 0; dim < 3; ++dim) mnodelin[in][dim][mrtrmnode->dofs()[dim]] += 1.;
  }

  // put everything together for slave vertex linearization
  // loop over all vertices
  for (int i = 0; i < line_element()->num_node(); ++i)
  {
    Mortar::Node* mrtrmnode =
        dynamic_cast<Mortar::Node*>(idiscret_.g_node(line_element()->node_ids()[i]));
    if (!mrtrmnode) FOUR_C_THROW("cast to mortar node failed");

    // (1) slave node coordinates part
    for (_CI p = mnodelin[i][0].begin(); p != mnodelin[i][0].end(); ++p)
    {
      currlin[i][0][p->first] += (1.0 - auxn()[0] * auxn()[0]) * p->second;
      currlin[i][1][p->first] -= (auxn()[0] * auxn()[1]) * p->second;
      currlin[i][2][p->first] -= (auxn()[0] * auxn()[2]) * p->second;
    }
    for (_CI p = mnodelin[i][1].begin(); p != mnodelin[i][1].end(); ++p)
    {
      currlin[i][0][p->first] -= (auxn()[0] * auxn()[1]) * p->second;
      currlin[i][1][p->first] += (1.0 - auxn()[1] * auxn()[1]) * p->second;
      currlin[i][2][p->first] -= (auxn()[1] * auxn()[2]) * p->second;
    }
    for (_CI p = mnodelin[i][2].begin(); p != mnodelin[i][2].end(); ++p)
    {
      currlin[i][0][p->first] -= (auxn()[2] * auxn()[0]) * p->second;
      currlin[i][1][p->first] -= (auxn()[2] * auxn()[1]) * p->second;
      currlin[i][2][p->first] += (1.0 - auxn()[2] * auxn()[2]) * p->second;
    }

    // (2) slave element center coordinates (Auxc()) part
    for (_CI p = get_deriv_auxc()[0].begin(); p != get_deriv_auxc()[0].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[0] * auxn()[k] * (p->second);

    for (_CI p = get_deriv_auxc()[1].begin(); p != get_deriv_auxc()[1].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[1] * auxn()[k] * (p->second);

    for (_CI p = get_deriv_auxc()[2].begin(); p != get_deriv_auxc()[2].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[2] * auxn()[k] * (p->second);

    // (3) slave element normal (Auxn()) part
    double xdotn = (mrtrmnode->xspatial()[0] - auxc()[0]) * auxn()[0] +
                   (mrtrmnode->xspatial()[1] - auxc()[1]) * auxn()[1] +
                   (mrtrmnode->xspatial()[2] - auxc()[2]) * auxn()[2];

    for (_CI p = linauxn[0].begin(); p != linauxn[0].end(); ++p)
    {
      currlin[i][0][p->first] -= xdotn * (p->second);
      for (int k = 0; k < 3; ++k)
        currlin[i][k][p->first] -= (mrtrmnode->xspatial()[0] - auxc()[0]) * auxn()[k] * (p->second);
    }

    for (_CI p = linauxn[1].begin(); p != linauxn[1].end(); ++p)
    {
      currlin[i][1][p->first] -= xdotn * (p->second);
      for (int k = 0; k < 3; ++k)
        currlin[i][k][p->first] -= (mrtrmnode->xspatial()[1] - auxc()[1]) * auxn()[k] * (p->second);
    }

    for (_CI p = linauxn[2].begin(); p != linauxn[2].end(); ++p)
    {
      currlin[i][2][p->first] -= xdotn * (p->second);
      for (int k = 0; k < 3; ++k)
        currlin[i][k][p->first] -= (mrtrmnode->xspatial()[2] - auxc()[2]) * auxn()[k] * (p->second);
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 |  eval (public)                                            farah 07/16|
 *----------------------------------------------------------------------*/
bool CONTACT::LineToSurfaceCoupling3d::project_master()
{
  // project master nodes onto auxiliary plane
  int nnodes = surface_element().num_node();
  Core::Nodes::Node** mynodes = surface_element().nodes();
  if (!mynodes) FOUR_C_THROW("project_master: Null pointer!");

  // initialize storage for master coords + their ids
  std::vector<double> vertices(3);
  std::vector<int> mnodeids(1);

  for (int i = 0; i < nnodes; ++i)
  {
    Node* mycnode = dynamic_cast<Node*>(mynodes[i]);
    if (!mycnode) FOUR_C_THROW("project_master: Null pointer!");

    // first build difference of point and element center
    // and then dot product with unit normal at center
    const double dist = (mycnode->xspatial()[0] - auxc()[0]) * auxn()[0] +
                        (mycnode->xspatial()[1] - auxc()[1]) * auxn()[1] +
                        (mycnode->xspatial()[2] - auxc()[2]) * auxn()[2];

    // compute projection
    for (int k = 0; k < 3; ++k) vertices[k] = mycnode->xspatial()[k] - dist * auxn()[k];

    // get node id, too
    mnodeids[0] = mycnode->id();

    // store into vertex data structure
    master_vertices().push_back(Mortar::Vertex(
        vertices, Mortar::Vertex::master, mnodeids, nullptr, nullptr, false, false, nullptr, -1.0));

    // std::cout << "->RealNode(M) " << mycnode->Id() << ": " << mycnode->xspatial()[0] << " " <<
    // mycnode->xspatial()[1] << " " << mycnode->xspatial()[2] << std::endl; std::cout <<
    // "->ProjNode(M) " << mycnode->Id() << ": " << vertices[0] << " " << vertices[1] << " " <<
    // vertices[2] << std::endl;
  }
  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of slave vertex (3D) AuxPlane               farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToSurfaceCoupling3d::master_vertex_linearization(
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& currlin)
{
  // we first need the slave element center:
  // for quad4, quad8, quad9 elements: xi = eta = 0.0
  // for tri3, tri6 elements: xi = eta = 1/3
  double scxi[2];

  Core::FE::CellType dt = surface_element().shape();
  if (dt == Core::FE::CellType::tri3 || dt == Core::FE::CellType::tri6)
  {
    scxi[0] = 1.0 / 3.0;
    scxi[1] = 1.0 / 3.0;
  }
  else if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
           dt == Core::FE::CellType::quad9)
  {
    scxi[0] = 0.0;
    scxi[1] = 0.0;
  }
  else
    FOUR_C_THROW("slave_vertex_linearization called for unknown element type");

  // evlauate shape functions + derivatives at scxi
  const int nrow = surface_element().num_node();
  Core::LinAlg::SerialDenseVector sval(nrow);
  Core::LinAlg::SerialDenseMatrix sderiv(nrow, 2, true);
  surface_element().evaluate_shape(scxi, sval, sderiv, nrow);

  // we need all participating slave nodes
  Core::Nodes::Node** snodes = surface_element().nodes();
  std::vector<Mortar::Node*> smrtrnodes(nrow);

  for (int i = 0; i < nrow; ++i)
  {
    smrtrnodes[i] = dynamic_cast<Mortar::Node*>(snodes[i]);
    if (!smrtrnodes[i]) FOUR_C_THROW("slave_vertex_linearization: Null pointer!");
  }

  // linearization of the IntEle spatial coords
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> nodelin(0);

  // resize the linearizations
  nodelin.resize(nrow, std::vector<Core::Gen::Pairedvector<int, double>>(3, 1));

  // loop over all intEle nodes
  for (int in = 0; in < nrow; ++in)
    for (int dim = 0; dim < 3; ++dim) nodelin[in][dim][smrtrnodes[in]->dofs()[dim]] += 1.;

  // map iterator
  typedef Core::Gen::Pairedvector<int, double>::const_iterator
      _CI;  // linearization of element center Auxc()
  //  std  ::vector<Core::Gen::Pairedvector<int  ,double> > linauxc(3,surface_element().num_node());
  //  // assume 3 dofs per node
  //
  //  for (int i = 0; i < nrow; ++i)
  //      for (int dim=0; dim<3; ++dim)
  //        for (_CI p=nodelin[i][dim].begin(); p!=nodelin[i][dim].end(); ++p)
  //          linauxc[dim][p->first] = sval[i]*p->second;

  // linearization of element normal Auxn()
  std::vector<Core::Gen::Pairedvector<int, double>>& linauxn = get_deriv_auxn();

  // put everything together for slave vertex linearization
  // loop over all vertices
  for (int i = 0; i < surface_element().num_node(); ++i)
  {
    Mortar::Node* mrtrsnode = dynamic_cast<Mortar::Node*>(surface_element().nodes()[i]);
    if (!mrtrsnode) FOUR_C_THROW("cast to mortar node failed");

    // (1) slave node coordinates part
    for (_CI p = nodelin[i][0].begin(); p != nodelin[i][0].end(); ++p)
    {
      currlin[i][0][p->first] += (1.0 - auxn()[0] * auxn()[0]) * p->second;
      currlin[i][1][p->first] -= (auxn()[0] * auxn()[1]) * p->second;
      currlin[i][2][p->first] -= (auxn()[0] * auxn()[2]) * p->second;
    }
    for (_CI p = nodelin[i][1].begin(); p != nodelin[i][1].end(); ++p)
    {
      currlin[i][0][p->first] -= (auxn()[0] * auxn()[1]) * p->second;
      currlin[i][1][p->first] += (1.0 - auxn()[1] * auxn()[1]) * p->second;
      currlin[i][2][p->first] -= (auxn()[1] * auxn()[2]) * p->second;
    }
    for (_CI p = nodelin[i][2].begin(); p != nodelin[i][2].end(); ++p)
    {
      currlin[i][0][p->first] -= (auxn()[2] * auxn()[0]) * p->second;
      currlin[i][1][p->first] -= (auxn()[2] * auxn()[1]) * p->second;
      currlin[i][2][p->first] += (1.0 - auxn()[2] * auxn()[2]) * p->second;
    }

    // (2) slave element center coordinates (Auxc()) part
    for (_CI p = get_deriv_auxc()[0].begin(); p != get_deriv_auxc()[0].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[0] * auxn()[k] * (p->second);

    for (_CI p = get_deriv_auxc()[1].begin(); p != get_deriv_auxc()[1].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[1] * auxn()[k] * (p->second);

    for (_CI p = get_deriv_auxc()[2].begin(); p != get_deriv_auxc()[2].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[2] * auxn()[k] * (p->second);

    // (3) slave element normal (Auxn()) part
    double xdotn = (mrtrsnode->xspatial()[0] - auxc()[0]) * auxn()[0] +
                   (mrtrsnode->xspatial()[1] - auxc()[1]) * auxn()[1] +
                   (mrtrsnode->xspatial()[2] - auxc()[2]) * auxn()[2];

    for (_CI p = linauxn[0].begin(); p != linauxn[0].end(); ++p)
    {
      currlin[i][0][p->first] -= xdotn * (p->second);
      for (int k = 0; k < 3; ++k)
        currlin[i][k][p->first] -= (mrtrsnode->xspatial()[0] - auxc()[0]) * auxn()[k] * (p->second);
    }

    for (_CI p = linauxn[1].begin(); p != linauxn[1].end(); ++p)
    {
      currlin[i][1][p->first] -= xdotn * (p->second);
      for (int k = 0; k < 3; ++k)
        currlin[i][k][p->first] -= (mrtrsnode->xspatial()[1] - auxc()[1]) * auxn()[k] * (p->second);
    }

    for (_CI p = linauxn[2].begin(); p != linauxn[2].end(); ++p)
    {
      currlin[i][2][p->first] -= xdotn * (p->second);
      for (int k = 0; k < 3; ++k)
        currlin[i][k][p->first] -= (mrtrsnode->xspatial()[2] - auxc()[2]) * auxn()[k] * (p->second);
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  get communicator                                         farah 07/16|
 *----------------------------------------------------------------------*/
MPI_Comm CONTACT::LineToSurfaceCoupling3d::get_comm() const { return idiscret_.get_comm(); }


/*----------------------------------------------------------------------*
 |  ctor for ltl (public)                                    farah 07/16|
 *----------------------------------------------------------------------*/
CONTACT::LineToLineCouplingPoint3d::LineToLineCouplingPoint3d(Core::FE::Discretization& idiscret,
    int dim, Teuchos::ParameterList& params, std::shared_ptr<Mortar::Element>& lsele,
    std::shared_ptr<Mortar::Element>& lmele)
    : idiscret_(idiscret), dim_(dim), imortar_(params), l_sele_(lsele), l_mele_(lmele)
{
  // empty constructor
}


/*----------------------------------------------------------------------*
 |  get communicator                                         farah 07/16|
 *----------------------------------------------------------------------*/
MPI_Comm CONTACT::LineToLineCouplingPoint3d::get_comm() const { return idiscret_.get_comm(); }

/*----------------------------------------------------------------------*
 |  eval                                                     farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToLineCouplingPoint3d::evaluate_coupling()
{
  // 1. check parallelity
  bool parallel = check_parallelity();
  if (parallel) return;

  // 2. calc intersection
  // create empty points
  double sxi = 0.0;
  double mxi = 0.0;

  // create empty lin vectors
  Core::Gen::Pairedvector<int, double> dsxi(
      100 + 3 * line_master_element()->num_node() + 3 * line_slave_element()->num_node());
  Core::Gen::Pairedvector<int, double> dmxi(
      100 + 3 * line_master_element()->num_node() + 3 * line_slave_element()->num_node());
  line_intersection(&sxi, &mxi, dsxi, dmxi);

  // 3. check solution
  bool valid = check_intersection(&sxi, &mxi);
  if (!valid) return;

  // 4. check if intersection was already done!
  for (int i = 0; i < line_slave_element()->num_node(); ++i)
  {
    if (dynamic_cast<Node*>(line_slave_element()->nodes()[i])->mo_data().get_dltl().size() > 0)
      return;
  }

  // 5. evaluate terms
  evaluate_terms(&sxi, &mxi, dsxi, dmxi);

  return;
}


/*----------------------------------------------------------------------*
 |  line-line intersection                                   farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToLineCouplingPoint3d::evaluate_terms(double* sxi, double* mxi,
    Core::Gen::Pairedvector<int, double>& dsxi, Core::Gen::Pairedvector<int, double>& dmxi)
{
  bool friction = false;
  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(interface_params(), "FRICTION");
  if (ftype != CONTACT::FrictionType::None) friction = true;

  // get slave element nodes themselves for normal evaluation
  Core::Nodes::Node** mynodes = line_slave_element()->nodes();
  if (!mynodes) FOUR_C_THROW("integrate_deriv_cell_3d_aux_plane_lts: Null pointer!");
  Core::Nodes::Node** mnodes = line_master_element()->nodes();
  if (!mnodes) FOUR_C_THROW("integrate_deriv_cell_3d_aux_plane_lts: Null pointer!");

  int nnodes = 2;
  int ndof = 3;
  int nrow = line_slave_element()->num_node();
  int ncol = line_master_element()->num_node();

  // slave values
  Core::LinAlg::SerialDenseVector sval(nnodes);
  Core::LinAlg::SerialDenseMatrix sderiv(nnodes, 1);
  line_slave_element()->evaluate_shape(sxi, sval, sderiv, nnodes);

  // master values
  Core::LinAlg::SerialDenseVector mval(nnodes);
  Core::LinAlg::SerialDenseMatrix mderiv(nnodes, 1);
  line_master_element()->evaluate_shape(mxi, mval, mderiv, nnodes);

  // map iterator
  typedef Core::Gen::Pairedvector<int, double>::const_iterator _CI;
  typedef std::map<int, double>::const_iterator CI;

  int linsize = 0;
  for (int i = 0; i < nrow; ++i)
  {
    Node* cnode = dynamic_cast<Node*>(mynodes[i]);
    linsize += cnode->get_linsize();
  }

  // TODO: this is for safety. Correct linsize
  //       should be predicted
  linsize *= 100;

  //**********************************************************************
  // geometric quantities
  //**********************************************************************
  std::array<double, 3> gpn = {0.0, 0.0, 0.0};
  Core::Gen::Pairedvector<int, double> dgapgp(
      (ncol * ndof) + 10 * linsize);  // gap lin. without lm and jac.
  double gap = 0.0;
  std::vector<Core::Gen::Pairedvector<int, double>> dnmap_unit(
      3, 10 * linsize);  // deriv of x,y and z comp. of gpn (unit)

  //**********************************************************************
  // evaluate at GP and lin char. quantities
  //**********************************************************************
  std::array<double, 3> sgpx = {0.0, 0.0, 0.0};
  std::array<double, 3> mgpx = {0.0, 0.0, 0.0};

  for (int i = 0; i < nrow; ++i)
  {
    Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[i]);
    gpn[0] += sval[i] * mymrtrnode->mo_data().n()[0];
    gpn[1] += sval[i] * mymrtrnode->mo_data().n()[1];
    gpn[2] += sval[i] * mymrtrnode->mo_data().n()[2];

    sgpx[0] += sval[i] * line_slave_element()->get_nodal_coords(0, i);
    sgpx[1] += sval[i] * line_slave_element()->get_nodal_coords(1, i);
    sgpx[2] += sval[i] * line_slave_element()->get_nodal_coords(2, i);
  }

  // build interpolation of master GP coordinates
  for (int i = 0; i < ncol; ++i)
  {
    mgpx[0] += mval[i] * line_master_element()->get_nodal_coords(0, i);
    mgpx[1] += mval[i] * line_master_element()->get_nodal_coords(1, i);
    mgpx[2] += mval[i] * line_master_element()->get_nodal_coords(2, i);
  }

  // normalize interpolated GP normal back to length 1.0 !!!
  double lengthn = sqrt(gpn[0] * gpn[0] + gpn[1] * gpn[1] + gpn[2] * gpn[2]);
  if (lengthn < 1.0e-12) FOUR_C_THROW("IntegrateAndDerivSegment: Divide by zero!");

  for (int i = 0; i < 3; ++i) gpn[i] /= lengthn;

  // build gap function at current GP
  for (int i = 0; i < n_dim(); ++i) gap += (mgpx[i] - sgpx[i]) * gpn[i];

  // build directional derivative of slave GP normal (non-unit)
  Core::Gen::Pairedvector<int, double> dmap_nxsl_gp(linsize);
  Core::Gen::Pairedvector<int, double> dmap_nysl_gp(linsize);
  Core::Gen::Pairedvector<int, double> dmap_nzsl_gp(linsize);

  for (int i = 0; i < nrow; ++i)
  {
    Node* cnode = dynamic_cast<Node*>(mynodes[i]);

    Core::Gen::Pairedvector<int, double>& dmap_nxsl_i = cnode->data().get_deriv_n()[0];
    Core::Gen::Pairedvector<int, double>& dmap_nysl_i = cnode->data().get_deriv_n()[1];
    Core::Gen::Pairedvector<int, double>& dmap_nzsl_i = cnode->data().get_deriv_n()[2];

    for (_CI p = dmap_nxsl_i.begin(); p != dmap_nxsl_i.end(); ++p)
      dmap_nxsl_gp[p->first] += sval[i] * (p->second);
    for (_CI p = dmap_nysl_i.begin(); p != dmap_nysl_i.end(); ++p)
      dmap_nysl_gp[p->first] += sval[i] * (p->second);
    for (_CI p = dmap_nzsl_i.begin(); p != dmap_nzsl_i.end(); ++p)
      dmap_nzsl_gp[p->first] += sval[i] * (p->second);

    for (_CI p = dsxi.begin(); p != dsxi.end(); ++p)
    {
      double valx = sderiv(i, 0) * cnode->mo_data().n()[0];
      dmap_nxsl_gp[p->first] += valx * (p->second);
      double valy = sderiv(i, 0) * cnode->mo_data().n()[1];
      dmap_nysl_gp[p->first] += valy * (p->second);
      double valz = sderiv(i, 0) * cnode->mo_data().n()[2];
      dmap_nzsl_gp[p->first] += valz * (p->second);
    }
  }

  const double ll = lengthn * lengthn;
  const double linv = 1.0 / (lengthn);
  const double lllinv = 1.0 / (lengthn * lengthn * lengthn);
  const double sxsx = gpn[0] * gpn[0] * ll;
  const double sxsy = gpn[0] * gpn[1] * ll;
  const double sxsz = gpn[0] * gpn[2] * ll;
  const double sysy = gpn[1] * gpn[1] * ll;
  const double sysz = gpn[1] * gpn[2] * ll;
  const double szsz = gpn[2] * gpn[2] * ll;

  for (_CI p = dmap_nxsl_gp.begin(); p != dmap_nxsl_gp.end(); ++p)
  {
    dnmap_unit[0][p->first] += linv * (p->second);
    dnmap_unit[0][p->first] -= lllinv * sxsx * (p->second);
    dnmap_unit[1][p->first] -= lllinv * sxsy * (p->second);
    dnmap_unit[2][p->first] -= lllinv * sxsz * (p->second);
  }

  for (_CI p = dmap_nysl_gp.begin(); p != dmap_nysl_gp.end(); ++p)
  {
    dnmap_unit[1][p->first] += linv * (p->second);
    dnmap_unit[1][p->first] -= lllinv * sysy * (p->second);
    dnmap_unit[0][p->first] -= lllinv * sxsy * (p->second);
    dnmap_unit[2][p->first] -= lllinv * sysz * (p->second);
  }

  for (_CI p = dmap_nzsl_gp.begin(); p != dmap_nzsl_gp.end(); ++p)
  {
    dnmap_unit[2][p->first] += linv * (p->second);
    dnmap_unit[2][p->first] -= lllinv * szsz * (p->second);
    dnmap_unit[0][p->first] -= lllinv * sxsz * (p->second);
    dnmap_unit[1][p->first] -= lllinv * sysz * (p->second);
  }

  // add everything to dgapgp
  for (_CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    dgapgp[p->first] += (mgpx[0] - sgpx[0]) * (p->second);

  for (_CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    dgapgp[p->first] += (mgpx[1] - sgpx[1]) * (p->second);

  for (_CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    dgapgp[p->first] += (mgpx[2] - sgpx[2]) * (p->second);

  // lin slave nodes
  for (int z = 0; z < nrow; ++z)
  {
    Node* cnode = dynamic_cast<Node*>(mynodes[z]);
    for (int k = 0; k < 3; ++k) dgapgp[cnode->dofs()[k]] -= sval[z] * gpn[k];
  }

  for (_CI p = dsxi.begin(); p != dsxi.end(); ++p)
  {
    double& dg = dgapgp[p->first];
    const double& ps = p->second;
    for (int z = 0; z < nrow; ++z)
    {
      Node* cnode = dynamic_cast<Node*>(mynodes[z]);
      for (int k = 0; k < 3; ++k) dg -= gpn[k] * sderiv(z, 0) * cnode->xspatial()[k] * ps;
    }
  }

  //        MASTER
  // lin master nodes
  for (int z = 0; z < ncol; ++z)
  {
    Node* cnode = dynamic_cast<Node*>(mnodes[z]);
    for (int k = 0; k < 3; ++k) dgapgp[cnode->dofs()[k]] += mval[z] * gpn[k];
  }

  for (_CI p = dmxi.begin(); p != dmxi.end(); ++p)
  {
    double& dg = dgapgp[p->first];
    const double& ps = p->second;
    for (int z = 0; z < ncol; ++z)
    {
      Node* cnode = dynamic_cast<Node*>(mnodes[z]);
      for (int k = 0; k < 3; ++k) dg += gpn[k] * mderiv(z, 0) * cnode->xspatial()[k] * ps;
    }
  }

  // gap
  CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(mynodes[0]);

  // do not process slave side boundary nodes
  // (their row entries would be zero anyway!)
  if (cnode->is_on_bound()) return;

  if (gap >= 0.0) return;

  double value[3] = {0.0, 0.0, 0.0};
  value[0] = (mgpx[0] - sgpx[0]);  // gap*gpn[0];
  value[1] = (mgpx[1] - sgpx[1]);  // gap*gpn[1];
  value[2] = (mgpx[2] - sgpx[2]);  // gap*gpn[2];

  // add current Gauss point's contribution to gseg
  cnode->addltl_gap_value(value);

  double lengthv = 0.0;
  lengthv = sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
  if (lengthv < 1e-12) FOUR_C_THROW("zero length!");
  value[0] /= lengthv;
  value[1] /= lengthv;
  value[2] /= lengthv;

  std::vector<std::map<int, double>>& dgmap = cnode->data().get_deriv_gltl();

  for (_CI p = dgapgp.begin(); p != dgapgp.end(); ++p)
  {
    dgmap[0][p->first] += gpn[0] * (p->second);
    dgmap[1][p->first] += gpn[1] * (p->second);
    dgmap[2][p->first] += gpn[2] * (p->second);
  }

  for (_CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    dgmap[0][p->first] += gap * (p->second);
  for (_CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    dgmap[1][p->first] += gap * (p->second);
  for (_CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    dgmap[2][p->first] += gap * (p->second);

  //*****************************************
  // integrate D and M matrix
  // integrate dseg
  for (int k = 0; k < nrow; ++k)
  {
    CONTACT::Node* mnode = dynamic_cast<CONTACT::Node*>(mynodes[k]);

    // multiply the two shape functions
    double prod = sval[k];  // this reduces to sval[k]

    if (abs(prod) > MORTARINTTOL) cnode->add_dltl_value(mnode->id(), prod);
    //    if(abs(prod)>MORTARINTTOL) cnode->AddSNode(mnode->Id()); // only for friction!
  }

  // integrate mseg
  for (int k = 0; k < ncol; ++k)
  {
    CONTACT::Node* mnode = dynamic_cast<CONTACT::Node*>(mnodes[k]);

    // multiply the two shape functions
    double prod = mval[k];  // this reduces to mval[k]

    if (abs(prod) > MORTARINTTOL) cnode->add_mltl_value(mnode->id(), prod);
    //    if(abs(prod)>MORTARINTTOL) cnode->AddMNode(mnode->Id());  // only for friction!
  }

  // integrate LinD
  for (int k = 0; k < nrow; ++k)
  {
    // global master node ID
    int mgid = line_slave_element()->nodes()[k]->id();
    static double fac = 0.0;

    // get the correct map as a reference
    std::map<int, double>& ddmap_jk = cnode->data().get_deriv_dltl()[mgid];

    // (3) Lin(NMaster) - master GP coordinates
    fac = sderiv(k, 0);
    for (_CI p = dsxi.begin(); p != dsxi.end(); ++p)
    {
      ddmap_jk[p->first] += fac * (p->second);
    }
  }  // loop over slave nodes

  // integrate LinM
  for (int k = 0; k < ncol; ++k)
  {
    // global master node ID
    int mgid = line_master_element()->nodes()[k]->id();
    static double fac = 0.0;

    // get the correct map as a reference
    std::map<int, double>& dmmap_jk = cnode->data().get_deriv_mltl()[mgid];

    // (1) Lin(Phi) - dual shape functions
    // this vanishes here since there are no deformation-dependent dual functions

    // (3) Lin(NMaster) - master GP coordinates
    fac = mderiv(k, 0);
    for (_CI p = dmxi.begin(); p != dmxi.end(); ++p)
    {
      dmmap_jk[p->first] += fac * (p->second);
    }
  }  // loop over master nodes


  //  std::cout << "element is evaluated!" << std::endl;

  //***************************************************************************
  if (friction)
  {
    // tangent:
    // first jump:
    std::array<double, 3> jump = {0.0, 0.0, 0.0};
    std::array<double, 3> sgpxold = {0.0, 0.0, 0.0};
    std::array<double, 3> mgpxold = {0.0, 0.0, 0.0};

    int oldID = -1;

    // loop over all slave nodes
    for (int i = 0; i < idiscret_.node_col_map()->NumMyElements(); ++i)
    {
      int gid1 = idiscret_.node_col_map()->GID(i);
      Core::Nodes::Node* node1 = idiscret_.g_node(gid1);
      if (!node1) FOUR_C_THROW("Cannot find node with gid %", gid1);
      Node* contactnode = dynamic_cast<Node*>(node1);

      // here only slave nodes
      if (!contactnode->is_slave()) continue;

      // check if dold is present
      if (dynamic_cast<FriNode*>(contactnode)->fri_data().get_d_old_ltl().size() < 1) continue;

      // store id
      oldID = gid1;

      // if we are here, break
      break;
    }

    // linearizations
    Core::Gen::Pairedvector<int, double> sgpxoldlinx(linsize);
    Core::Gen::Pairedvector<int, double> sgpxoldliny(linsize);
    Core::Gen::Pairedvector<int, double> sgpxoldlinz(linsize);

    Core::Gen::Pairedvector<int, double> mgpxoldlinx(linsize);
    Core::Gen::Pairedvector<int, double> mgpxoldliny(linsize);
    Core::Gen::Pairedvector<int, double> mgpxoldlinz(linsize);

    if (oldID > -1)
    {
      Core::Nodes::Node* node1 = idiscret_.g_node(oldID);
      if (!node1) FOUR_C_THROW("Cannot find node with gid %", oldID);
      Node* contactnode = dynamic_cast<Node*>(node1);

      // check if we have dold
      if (dynamic_cast<FriNode*>(contactnode)->fri_data().get_d_old_ltl().size() > 0)
      {
        for (_CI p = dynamic_cast<FriNode*>(contactnode)->fri_data().get_d_old_ltl().begin();
            p != dynamic_cast<FriNode*>(contactnode)->fri_data().get_d_old_ltl().end(); ++p)
        {
          // node id
          int gid3 = p->first;
          Core::Nodes::Node* snode = idiscret_.g_node(gid3);
          if (!snode) FOUR_C_THROW("Cannot find node with gid");
          Node* csnode = dynamic_cast<Node*>(snode);

          for (int d = 0; d < n_dim(); ++d)
          {
            sgpxold[d] += p->second * csnode->xspatial()[d];
          }
          sgpxoldlinx[csnode->dofs()[0]] += p->second;
          sgpxoldliny[csnode->dofs()[1]] += p->second;
          sgpxoldlinz[csnode->dofs()[2]] += p->second;
        }

        // safety
        if (dynamic_cast<FriNode*>(contactnode)->fri_data().get_m_old_ltl().size() < 1)
          FOUR_C_THROW("something went wrong!");

        for (CI p = dynamic_cast<FriNode*>(contactnode)->fri_data().get_m_old_ltl().begin();
            p != dynamic_cast<FriNode*>(contactnode)->fri_data().get_m_old_ltl().end(); ++p)
        {
          // node id
          int gid3 = p->first;
          Core::Nodes::Node* mnode = idiscret_.g_node(gid3);
          if (!mnode) FOUR_C_THROW("Cannot find node with gid");
          Node* cmnode = dynamic_cast<Node*>(mnode);

          for (int d = 0; d < n_dim(); ++d)
          {
            mgpxold[d] += p->second * cmnode->xspatial()[d];
          }
          mgpxoldlinx[cmnode->dofs()[0]] += p->second;
          mgpxoldliny[cmnode->dofs()[1]] += p->second;
          mgpxoldlinz[cmnode->dofs()[2]] += p->second;
        }
      }
    }

    // create slip
    for (int d = 0; d < n_dim(); ++d) jump[d] = mgpx[d] - mgpxold[d] - (sgpx[d] - sgpxold[d]);

    Core::LinAlg::SerialDenseMatrix tanplane(3, 3);
    tanplane(0, 0) = 1 - (value[0] * value[0]);
    tanplane(0, 1) = -(value[0] * value[1]);
    tanplane(0, 2) = -(value[0] * value[2]);
    tanplane(1, 0) = -(value[1] * value[0]);
    tanplane(1, 1) = 1 - (value[1] * value[1]);
    tanplane(1, 2) = -(value[1] * value[2]);
    tanplane(2, 0) = -(value[2] * value[0]);
    tanplane(2, 1) = -(value[2] * value[1]);
    tanplane(2, 2) = 1 - (value[2] * value[2]);

    double finaljump[3] = {0.0, 0.0, 0.0};
    finaljump[0] = tanplane(0, 0) * jump[0] + tanplane(0, 1) * jump[1] + tanplane(0, 2) * jump[2];
    finaljump[1] = tanplane(1, 0) * jump[0] + tanplane(1, 1) * jump[1] + tanplane(1, 2) * jump[2];
    finaljump[2] = tanplane(2, 0) * jump[0] + tanplane(2, 1) * jump[1] + tanplane(2, 2) * jump[2];

    cnode->addltl_jump_value(finaljump);
    //    std::cout << "jump = " << sqrt(finaljump[0]*finaljump[0] + finaljump[1]*finaljump[1]
    //    +finaljump[2]*finaljump[2]) << std::endl;

    std::vector<std::map<int, double>>& djmapfinal = cnode->data().get_deriv_jumpltl();

    std::vector<Core::Gen::Pairedvector<int, double>> djmap(3, 100);

    // lin slave nodes
    for (int z = 0; z < nrow; ++z)
    {
      Node* node = dynamic_cast<Node*>(mynodes[z]);
      for (int k = 0; k < 3; ++k) djmap[k][node->dofs()[k]] -= sval[z];
    }


    for (int k = 0; k < nrow; ++k)
    {
      Node* node = dynamic_cast<Node*>(mynodes[k]);
      for (_CI p = dsxi.begin(); p != dsxi.end(); ++p)
      {
        for (int z = 0; z < 3; ++z)
          djmap[z][p->first] -= sderiv(k, 0) * (p->second) * node->xspatial()[z];
      }
    }  // loop over slave nodes


    // lin master nodes
    for (int z = 0; z < ncol; ++z)
    {
      Node* node = dynamic_cast<Node*>(mnodes[z]);
      for (int k = 0; k < 3; ++k) djmap[k][node->dofs()[k]] += mval[z];
    }

    for (int k = 0; k < ncol; ++k)
    {
      Node* node = dynamic_cast<Node*>(mnodes[k]);
      for (_CI p = dmxi.begin(); p != dmxi.end(); ++p)
      {
        for (int z = 0; z < 3; ++z)
          djmap[z][p->first] += mderiv(k, 0) * (p->second) * node->xspatial()[z];
      }
    }  // loop over slave nodes

    // sgpx and mgpx old
    for (_CI p = mgpxoldlinx.begin(); p != mgpxoldlinx.end(); ++p) djmap[0][p->first] -= p->second;
    for (_CI p = mgpxoldliny.begin(); p != mgpxoldliny.end(); ++p) djmap[1][p->first] -= p->second;
    for (_CI p = mgpxoldlinz.begin(); p != mgpxoldlinz.end(); ++p) djmap[2][p->first] -= p->second;

    for (_CI p = sgpxoldlinx.begin(); p != sgpxoldlinx.end(); ++p) djmap[0][p->first] += p->second;
    for (_CI p = sgpxoldliny.begin(); p != sgpxoldliny.end(); ++p) djmap[1][p->first] += p->second;
    for (_CI p = sgpxoldlinz.begin(); p != sgpxoldlinz.end(); ++p) djmap[2][p->first] += p->second;



    std::vector<Core::Gen::Pairedvector<int, double>> tplanex(3, 100);
    std::vector<Core::Gen::Pairedvector<int, double>> tplaney(3, 100);
    std::vector<Core::Gen::Pairedvector<int, double>> tplanez(3, 100);

    for (_CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      tplanex[0][p->first] -= gpn[0] * p->second;
    for (_CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      tplanex[1][p->first] -= gpn[1] * p->second;
    for (_CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      tplanex[2][p->first] -= gpn[2] * p->second;

    for (_CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      tplaney[0][p->first] -= gpn[0] * p->second;
    for (_CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      tplaney[1][p->first] -= gpn[1] * p->second;
    for (_CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      tplaney[2][p->first] -= gpn[2] * p->second;

    for (_CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
      tplanez[0][p->first] -= gpn[0] * p->second;
    for (_CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
      tplanez[1][p->first] -= gpn[1] * p->second;
    for (_CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
      tplanez[2][p->first] -= gpn[2] * p->second;

    //------------
    for (_CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      tplanex[0][p->first] -= gpn[0] * p->second;
    for (_CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      tplanex[1][p->first] -= gpn[0] * p->second;
    for (_CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
      tplanex[2][p->first] -= gpn[0] * p->second;

    for (_CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      tplaney[0][p->first] -= gpn[1] * p->second;
    for (_CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      tplaney[1][p->first] -= gpn[1] * p->second;
    for (_CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
      tplaney[2][p->first] -= gpn[1] * p->second;

    for (_CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      tplanez[0][p->first] -= gpn[2] * p->second;
    for (_CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      tplanez[1][p->first] -= gpn[2] * p->second;
    for (_CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
      tplanez[2][p->first] -= gpn[2] * p->second;

    //-----------------------------

    for (_CI p = djmap[0].begin(); p != djmap[0].end(); ++p)
      djmapfinal[0][p->first] += tanplane(0, 0) * p->second;
    for (_CI p = djmap[1].begin(); p != djmap[1].end(); ++p)
      djmapfinal[0][p->first] += tanplane(0, 1) * p->second;
    for (_CI p = djmap[2].begin(); p != djmap[2].end(); ++p)
      djmapfinal[0][p->first] += tanplane(0, 2) * p->second;

    for (_CI p = djmap[0].begin(); p != djmap[0].end(); ++p)
      djmapfinal[1][p->first] += tanplane(1, 0) * p->second;
    for (_CI p = djmap[1].begin(); p != djmap[1].end(); ++p)
      djmapfinal[1][p->first] += tanplane(1, 1) * p->second;
    for (_CI p = djmap[2].begin(); p != djmap[2].end(); ++p)
      djmapfinal[1][p->first] += tanplane(1, 2) * p->second;

    for (_CI p = djmap[0].begin(); p != djmap[0].end(); ++p)
      djmapfinal[2][p->first] += tanplane(2, 0) * p->second;
    for (_CI p = djmap[1].begin(); p != djmap[1].end(); ++p)
      djmapfinal[2][p->first] += tanplane(2, 1) * p->second;
    for (_CI p = djmap[2].begin(); p != djmap[2].end(); ++p)
      djmapfinal[2][p->first] += tanplane(2, 2) * p->second;

    //-----------------------------
    for (_CI p = tplanex[0].begin(); p != tplanex[0].end(); ++p)
      djmapfinal[0][p->first] += jump[0] * p->second;
    for (_CI p = tplanex[1].begin(); p != tplanex[1].end(); ++p)
      djmapfinal[0][p->first] += jump[1] * p->second;
    for (_CI p = tplanex[2].begin(); p != tplanex[2].end(); ++p)
      djmapfinal[0][p->first] += jump[2] * p->second;

    for (_CI p = tplaney[0].begin(); p != tplaney[0].end(); ++p)
      djmapfinal[1][p->first] += jump[0] * p->second;
    for (_CI p = tplaney[1].begin(); p != tplaney[1].end(); ++p)
      djmapfinal[1][p->first] += jump[1] * p->second;
    for (_CI p = tplaney[2].begin(); p != tplaney[2].end(); ++p)
      djmapfinal[1][p->first] += jump[2] * p->second;

    for (_CI p = tplanez[0].begin(); p != tplanez[0].end(); ++p)
      djmapfinal[2][p->first] += jump[0] * p->second;
    for (_CI p = tplanez[1].begin(); p != tplanez[1].end(); ++p)
      djmapfinal[2][p->first] += jump[1] * p->second;
    for (_CI p = tplanez[2].begin(); p != tplanez[2].end(); ++p)
      djmapfinal[2][p->first] += jump[2] * p->second;

  }  // end friction

  return;
}

/*----------------------------------------------------------------------*
 |  line-line intersection                                   farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToLineCouplingPoint3d::line_intersection(double* sxi, double* mxi,
    Core::Gen::Pairedvector<int, double>& dsxi, Core::Gen::Pairedvector<int, double>& dmxi)
{
  // flag for debug output
  const bool out = false;

  // only for line 2
  const int nnodes = 2;

  // prepare linearizations
  typedef Core::Gen::Pairedvector<int, double>::const_iterator _CI;

  // calculate slave vector
  Node* ns1 = dynamic_cast<Node*>(line_slave_element()->nodes()[0]);
  Node* ns2 = dynamic_cast<Node*>(line_slave_element()->nodes()[1]);
  ns1->build_averaged_edge_tangent();
  ns2->build_averaged_edge_tangent();

  // calculate slave vector
  Node* nm1 = dynamic_cast<Node*>(line_master_element()->nodes()[0]);
  Node* nm2 = dynamic_cast<Node*>(line_master_element()->nodes()[1]);
  nm1->build_averaged_edge_tangent();
  nm2->build_averaged_edge_tangent();

  double lengths1 = sqrt(ns1->mo_data().edge_tangent()[0] * ns1->mo_data().edge_tangent()[0] +
                         ns1->mo_data().edge_tangent()[1] * ns1->mo_data().edge_tangent()[1] +
                         ns1->mo_data().edge_tangent()[2] * ns1->mo_data().edge_tangent()[2]);
  double lengths2 = sqrt(ns2->mo_data().edge_tangent()[0] * ns2->mo_data().edge_tangent()[0] +
                         ns2->mo_data().edge_tangent()[1] * ns2->mo_data().edge_tangent()[1] +
                         ns2->mo_data().edge_tangent()[2] * ns2->mo_data().edge_tangent()[2]);
  double lengthm1 = sqrt(nm1->mo_data().edge_tangent()[0] * nm1->mo_data().edge_tangent()[0] +
                         nm1->mo_data().edge_tangent()[1] * nm1->mo_data().edge_tangent()[1] +
                         nm1->mo_data().edge_tangent()[2] * nm1->mo_data().edge_tangent()[2]);
  double lengthm2 = sqrt(nm2->mo_data().edge_tangent()[0] * nm2->mo_data().edge_tangent()[0] +
                         nm2->mo_data().edge_tangent()[1] * nm2->mo_data().edge_tangent()[1] +
                         nm2->mo_data().edge_tangent()[2] * nm2->mo_data().edge_tangent()[2]);
  if (lengths1 < 1e-12 or lengths2 < 1e-12 or lengthm1 < 1e-12 or lengthm2 < 1e-12)
    FOUR_C_THROW("tangents zero length");

  // calc angle between tangents
  std::array<double, 3> ts1 = {0.0, 0.0, 0.0};
  std::array<double, 3> ts2 = {0.0, 0.0, 0.0};
  std::array<double, 3> tm1 = {0.0, 0.0, 0.0};
  std::array<double, 3> tm2 = {0.0, 0.0, 0.0};

  ts1[0] = ns1->mo_data().edge_tangent()[0];
  ts1[1] = ns1->mo_data().edge_tangent()[1];
  ts1[2] = ns1->mo_data().edge_tangent()[2];

  ts2[0] = ns2->mo_data().edge_tangent()[0];
  ts2[1] = ns2->mo_data().edge_tangent()[1];
  ts2[2] = ns2->mo_data().edge_tangent()[2];

  tm1[0] = nm1->mo_data().edge_tangent()[0];
  tm1[1] = nm1->mo_data().edge_tangent()[1];
  tm1[2] = nm1->mo_data().edge_tangent()[2];

  tm2[0] = nm2->mo_data().edge_tangent()[0];
  tm2[1] = nm2->mo_data().edge_tangent()[1];
  tm2[2] = nm2->mo_data().edge_tangent()[2];

  if (out)
  {
    std::cout << "slave 1 = " << ts1[0] << "  " << ts1[1] << "  " << ts1[2] << std::endl;
    std::cout << "slave 2 = " << ts2[0] << "  " << ts2[1] << "  " << ts2[2] << std::endl;
  }

  double test = ts1[0] * ts2[0] + ts1[1] * ts2[1] + ts1[2] * ts2[2];
  if (test < 1e-8)
  {
    ns2->mo_data().edge_tangent()[0] *= -1.0;
    ns2->mo_data().edge_tangent()[1] *= -1.0;
    ns2->mo_data().edge_tangent()[2] *= -1.0;

    ts2[0] *= -1.0;
    ts2[1] *= -1.0;
    ts2[2] *= -1.0;

    for (_CI p = ns2->data().get_deriv_tangent()[0].begin();
        p != ns2->data().get_deriv_tangent()[0].end(); ++p)
      ns2->data().get_deriv_tangent()[0][p->first] *= -1.0;
    for (_CI p = ns2->data().get_deriv_tangent()[1].begin();
        p != ns2->data().get_deriv_tangent()[1].end(); ++p)
      ns2->data().get_deriv_tangent()[1][p->first] *= -1.0;
    for (_CI p = ns2->data().get_deriv_tangent()[2].begin();
        p != ns2->data().get_deriv_tangent()[2].end(); ++p)
      ns2->data().get_deriv_tangent()[2][p->first] *= -1.0;
  }
  if (out)
  {
    std::cout << "----------------" << std::endl;
    std::cout << "slave 1 = " << ts1[0] << "  " << ts1[1] << "  " << ts1[2] << std::endl;
    std::cout << "slave 2 = " << ts2[0] << "  " << ts2[1] << "  " << ts2[2] << std::endl;

    std::cout << "master 1 = " << tm1[0] << "  " << tm1[1] << "  " << tm1[2] << std::endl;
    std::cout << "master 2 = " << tm2[0] << "  " << tm2[1] << "  " << tm2[2] << std::endl;
  }

  test = tm1[0] * tm2[0] + tm1[1] * tm2[1] + tm1[2] * tm2[2];
  if (test < 1e-8)
  {
    nm2->mo_data().edge_tangent()[0] *= -1.0;
    nm2->mo_data().edge_tangent()[1] *= -1.0;
    nm2->mo_data().edge_tangent()[2] *= -1.0;

    tm2[0] *= -1.0;
    tm2[1] *= -1.0;
    tm2[2] *= -1.0;

    for (_CI p = nm2->data().get_deriv_tangent()[0].begin();
        p != nm2->data().get_deriv_tangent()[0].end(); ++p)
      nm2->data().get_deriv_tangent()[0][p->first] *= -1.0;
    for (_CI p = nm2->data().get_deriv_tangent()[1].begin();
        p != nm2->data().get_deriv_tangent()[1].end(); ++p)
      nm2->data().get_deriv_tangent()[1][p->first] *= -1.0;
    for (_CI p = nm2->data().get_deriv_tangent()[2].begin();
        p != nm2->data().get_deriv_tangent()[2].end(); ++p)
      nm2->data().get_deriv_tangent()[2][p->first] *= -1.0;
  }
  if (out)
  {
    std::cout << "----------------" << std::endl;
    std::cout << "master 1 = " << tm1[0] << "  " << tm1[1] << "  " << tm1[2] << std::endl;
    std::cout << "master 2 = " << tm2[0] << "  " << tm2[1] << "  " << tm2[2] << std::endl;
  }

  // res norm
  double conv = 0.0;

  // start in the element center
  double xiS = 0.0;  // xi_slave
  double xiM = 0.0;  // xi_master

  // function f (vector-valued)
  std::array<double, 2> f = {0.0, 0.0};

  // gradient of f (df/deta[0], df/deta[1])
  Core::LinAlg::Matrix<2, 2> df;

  // Newton
  for (int k = 0; k < MORTARMAXITER; ++k)
  {
    //**********************************************
    //  F CALCULATION                             //
    //**********************************************
    // slave values
    Core::LinAlg::SerialDenseVector sval(nnodes);
    Core::LinAlg::SerialDenseMatrix sderiv(nnodes, 1);
    line_slave_element()->evaluate_shape(&xiS, sval, sderiv, nnodes);

    // master values
    Core::LinAlg::SerialDenseVector mval(nnodes);
    Core::LinAlg::SerialDenseMatrix mderiv(nnodes, 1);
    line_master_element()->evaluate_shape(&xiM, mval, mderiv, nnodes);

    std::array<double, 3> xs = {0.0, 0.0, 0.0};
    std::array<double, 3> xm = {0.0, 0.0, 0.0};

    for (int i = 0; i < 3; ++i)
      xs[i] += sval[0] * ns1->xspatial()[i] + sval[1] * ns2->xspatial()[i];
    for (int i = 0; i < 3; ++i)
      xm[i] += mval[0] * nm1->xspatial()[i] + mval[1] * nm2->xspatial()[i];

    std::array<double, 3> xdiff = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) xdiff[i] = xs[i] - xm[i];

    // calculate tangents:
    std::array<double, 3> vs = {0.0, 0.0, 0.0};
    std::array<double, 3> vm = {0.0, 0.0, 0.0};

    for (int i = 0; i < 3; ++i) vs[i] += sval[0] * ts1[i] + sval[1] * ts2[i];
    for (int i = 0; i < 3; ++i) vm[i] += mval[0] * tm1[i] + mval[1] * tm2[i];

    f[0] = xdiff[0] * vs[0] + xdiff[1] * vs[1] + xdiff[2] * vs[2];
    f[1] = xdiff[0] * vm[0] + xdiff[1] * vm[1] + xdiff[2] * vm[2];

    // check for convergence
    conv = sqrt(f[0] * f[0] + f[1] * f[1]);
    if (conv <= MORTARCONVTOL) break;

    //**********************************************
    //   F GRADIENT CALCULATION                   //
    //**********************************************

    std::array<double, 3> xsderiv = {0.0, 0.0, 0.0};
    std::array<double, 3> xmderiv = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i)
      xsderiv[i] += sderiv(0, 0) * ns1->xspatial()[i] + sderiv(1, 0) * ns2->xspatial()[i];
    for (int i = 0; i < 3; ++i)
      xmderiv[i] += mderiv(0, 0) * nm1->xspatial()[i] + mderiv(1, 0) * nm2->xspatial()[i];

    std::array<double, 3> vsderiv = {0.0, 0.0, 0.0};
    std::array<double, 3> vmderiv = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) vsderiv[i] += sderiv(0, 0) * ts1[i] + sderiv(1, 0) * ts2[i];
    for (int i = 0; i < 3; ++i) vmderiv[i] += mderiv(0, 0) * tm1[i] + mderiv(1, 0) * tm2[i];

    df(0, 0) = xsderiv[0] * vs[0] + xsderiv[1] * vs[1] + xsderiv[2] * vs[2] +
               vsderiv[0] * xdiff[0] + vsderiv[1] * xdiff[1] + vsderiv[2] * xdiff[2];

    df(0, 1) = -xmderiv[0] * vs[0] - xmderiv[1] * vs[1] - xmderiv[2] * vs[2];

    df(1, 0) = xsderiv[0] * vm[0] + xsderiv[1] * vm[1] + xsderiv[2] * vm[2];

    df(1, 1) = -xmderiv[0] * vm[0] - xmderiv[1] * vm[1] - xmderiv[2] * vm[2] +
               vmderiv[0] * xdiff[0] + vmderiv[1] * xdiff[1] + vmderiv[2] * xdiff[2];

    //**********************************************
    //   solve deta = - inv(dF) * F               //
    //**********************************************
    double jacdet = df.invert();
    if (abs(jacdet) < 1.0e-12)
    {
      sxi[0] = 1e12;
      mxi[0] = 1e12;
      return;
      FOUR_C_THROW("Singular Jacobian for projection");
    }

    // update eta and alpha
    xiS += -df(0, 0) * f[0] - df(0, 1) * f[1];
    xiM += -df(1, 0) * f[0] - df(1, 1) * f[1];
  }

  // Newton iteration unconverged
  if (conv > MORTARCONVTOL) FOUR_C_THROW("LTL intersection not converged!");

  //**********************************************
  //  Linearization                             //
  //**********************************************
  // slave values
  Core::LinAlg::SerialDenseVector sval(nnodes);
  Core::LinAlg::SerialDenseMatrix sderiv(nnodes, 1);
  line_slave_element()->evaluate_shape(&xiS, sval, sderiv, nnodes);

  // master values
  Core::LinAlg::SerialDenseVector mval(nnodes);
  Core::LinAlg::SerialDenseMatrix mderiv(nnodes, 1);
  line_master_element()->evaluate_shape(&xiM, mval, mderiv, nnodes);

  std::array<double, 3> xs = {0.0, 0.0, 0.0};
  std::array<double, 3> xm = {0.0, 0.0, 0.0};

  for (int i = 0; i < 3; ++i) xs[i] += sval[0] * ns1->xspatial()[i] + sval[1] * ns2->xspatial()[i];
  for (int i = 0; i < 3; ++i) xm[i] += mval[0] * nm1->xspatial()[i] + mval[1] * nm2->xspatial()[i];

  std::array<double, 3> xdiff = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i) xdiff[i] = xs[i] - xm[i];

  // calculate tangents:
  std::array<double, 3> vs = {0.0, 0.0, 0.0};
  std::array<double, 3> vm = {0.0, 0.0, 0.0};

  for (int i = 0; i < 3; ++i) vs[i] += sval[0] * ts1[i] + sval[1] * ts2[i];
  for (int i = 0; i < 3; ++i) vm[i] += mval[0] * tm1[i] + mval[1] * tm2[i];

  std::vector<Core::Gen::Pairedvector<int, double>> xLin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> vsLin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> vmLin(3, 1000);

  // global position difference
  for (int i = 0; i < 3; ++i) (xLin[i])[ns1->dofs()[i]] += sval(0);
  for (int i = 0; i < 3; ++i) (xLin[i])[ns2->dofs()[i]] += sval(1);

  for (int i = 0; i < 3; ++i) (xLin[i])[nm1->dofs()[i]] -= mval(0);
  for (int i = 0; i < 3; ++i) (xLin[i])[nm2->dofs()[i]] -= mval(1);

  // TODO: this would be the correct linearization! however, the old one works better. no idea
  // why!?!?!? tangent vector slave
  for (int i = 0; i < 3; ++i)
  {
    for (_CI p = ns1->data().get_deriv_tangent()[i].begin();
        p != ns1->data().get_deriv_tangent()[i].end(); ++p)
      (vsLin[i])[p->first] += sval[0] * p->second;

    for (_CI p = ns2->data().get_deriv_tangent()[i].begin();
        p != ns2->data().get_deriv_tangent()[i].end(); ++p)
      (vsLin[i])[p->first] += sval[1] * p->second;
  }

  // tangent vector master
  for (int i = 0; i < 3; ++i)
  {
    for (_CI p = nm1->data().get_deriv_tangent()[i].begin();
        p != nm1->data().get_deriv_tangent()[i].end(); ++p)
      (vmLin[i])[p->first] += mval[0] * p->second;

    for (_CI p = nm2->data().get_deriv_tangent()[i].begin();
        p != nm2->data().get_deriv_tangent()[i].end(); ++p)
      (vmLin[i])[p->first] += mval[1] * p->second;
  }

  // TODO: this is the old linearization:
  // tangent vector slave
  //  for(int i=0; i<3;++i)
  //    (vsLin[i])[ns1->Dofs()[i]] += 1;
  //  for(int i=0; i<3;++i)
  //    (vsLin[i])[ns2->Dofs()[i]] -= 1;
  //
  //  // tangent vector master
  //  for(int i=0; i<3;++i)
  //    (vmLin[i])[nm1->Dofs()[i]] += 1;
  //  for(int i=0; i<3;++i)
  //    (vmLin[i])[nm2->Dofs()[i]] -= 1;

  Core::Gen::Pairedvector<int, double> f0(1000);
  Core::Gen::Pairedvector<int, double> f1(1000);

  // lin xdiff * tangent + xdiff * lin tangent
  for (_CI p = xLin[0].begin(); p != xLin[0].end(); ++p) f0[p->first] += (p->second) * vs[0];
  for (_CI p = xLin[1].begin(); p != xLin[1].end(); ++p) f0[p->first] += (p->second) * vs[1];
  for (_CI p = xLin[2].begin(); p != xLin[2].end(); ++p) f0[p->first] += (p->second) * vs[2];

  for (_CI p = vsLin[0].begin(); p != vsLin[0].end(); ++p) f0[p->first] += (p->second) * xdiff[0];
  for (_CI p = vsLin[1].begin(); p != vsLin[1].end(); ++p) f0[p->first] += (p->second) * xdiff[1];
  for (_CI p = vsLin[2].begin(); p != vsLin[2].end(); ++p) f0[p->first] += (p->second) * xdiff[2];

  // lin xdiff * tangent + xdiff * lin tangent
  for (_CI p = xLin[0].begin(); p != xLin[0].end(); ++p) f1[p->first] += (p->second) * vm[0];
  for (_CI p = xLin[1].begin(); p != xLin[1].end(); ++p) f1[p->first] += (p->second) * vm[1];
  for (_CI p = xLin[2].begin(); p != xLin[2].end(); ++p) f1[p->first] += (p->second) * vm[2];

  for (_CI p = vmLin[0].begin(); p != vmLin[0].end(); ++p) f1[p->first] += (p->second) * xdiff[0];
  for (_CI p = vmLin[1].begin(); p != vmLin[1].end(); ++p) f1[p->first] += (p->second) * xdiff[1];
  for (_CI p = vmLin[2].begin(); p != vmLin[2].end(); ++p) f1[p->first] += (p->second) * xdiff[2];

  // end
  for (_CI p = f0.begin(); p != f0.end(); ++p) dsxi[p->first] -= (p->second) * df(0, 0);
  for (_CI p = f1.begin(); p != f1.end(); ++p) dsxi[p->first] -= (p->second) * df(0, 1);

  for (_CI p = f0.begin(); p != f0.end(); ++p) dmxi[p->first] -= (p->second) * df(1, 0);
  for (_CI p = f1.begin(); p != f1.end(); ++p) dmxi[p->first] -= (p->second) * df(1, 1);

  sxi[0] = xiS;
  mxi[0] = xiM;

  return;
}

/*----------------------------------------------------------------------*
 |  check if intersection is in para space interval          farah 07/16|
 *----------------------------------------------------------------------*/
bool CONTACT::LineToLineCouplingPoint3d::check_intersection(double* sxi, double* mxi)
{
  if (sxi[0] >= -1.0 - 1e-12 and sxi[0] <= 1.0 + 1e-12 and mxi[0] >= -1.0 - 1e-12 and
      mxi[0] <= 1.0 + 1e-12)
    return true;

  return false;
}

/*----------------------------------------------------------------------*
 |  check if line eles parallel                              farah 07/16|
 *----------------------------------------------------------------------*/
bool CONTACT::LineToLineCouplingPoint3d::check_parallelity()
{
  // tolerance for line clipping
  const double sminedge = line_slave_element()->min_edge_size();
  const double mminedge = line_master_element()->min_edge_size();
  const double tol = MORTARCLIPTOL * std::min(sminedge, mminedge);

  std::array<double, 3> vs = {0.0, 0.0, 0.0};
  std::array<double, 3> vm = {0.0, 0.0, 0.0};

  // calculate slave vector
  Node* ns1 = dynamic_cast<Node*>(line_slave_element()->nodes()[0]);
  Node* ns2 = dynamic_cast<Node*>(line_slave_element()->nodes()[1]);

  vs[0] = ns1->xspatial()[0] - ns2->xspatial()[0];
  vs[1] = ns1->xspatial()[1] - ns2->xspatial()[1];
  vs[2] = ns1->xspatial()[2] - ns2->xspatial()[2];

  // calculate slave vector
  Node* nm1 = dynamic_cast<Node*>(line_master_element()->nodes()[0]);
  Node* nm2 = dynamic_cast<Node*>(line_master_element()->nodes()[1]);

  vm[0] = nm1->xspatial()[0] - nm2->xspatial()[0];
  vm[1] = nm1->xspatial()[1] - nm2->xspatial()[1];
  vm[2] = nm1->xspatial()[2] - nm2->xspatial()[2];

  // calculate lengths
  const double lengthS = sqrt(vs[0] * vs[0] + vs[1] * vs[1] + vs[2] * vs[2]);
  const double lengthM = sqrt(vm[0] * vm[0] + vm[1] * vm[1] + vm[2] * vm[2]);

  // calculate scalar product
  const double scaprod = vs[0] * vm[0] + vs[1] * vm[1] + vs[2] * vm[2];

  // proof if scalar product equals length product --> parallelity
  const double diff = abs(scaprod) - (lengthS * lengthM);

  if (abs(diff) < tol) return true;

  return false;
}



/*----------------------------------------------------------------------*
 |  calc current angle (rad) between edges                   farah 09/16|
 *----------------------------------------------------------------------*/
double CONTACT::LineToLineCouplingPoint3d::calc_current_angle(
    Core::Gen::Pairedvector<int, double>& lineAngle)
{
  // define iterator for linerization
  typedef Core::Gen::Pairedvector<int, double>::const_iterator CI;

  // slave edge vector and master vector edge
  std::array<double, 3> vs = {0.0, 0.0, 0.0};
  std::array<double, 3> vm = {0.0, 0.0, 0.0};

  // calculate slave vector
  Node* ns1 = dynamic_cast<Node*>(line_slave_element()->nodes()[0]);
  Node* ns2 = dynamic_cast<Node*>(line_slave_element()->nodes()[1]);

  vs[0] = ns1->xspatial()[0] - ns2->xspatial()[0];
  vs[1] = ns1->xspatial()[1] - ns2->xspatial()[1];
  vs[2] = ns1->xspatial()[2] - ns2->xspatial()[2];

  // calculate slave vector
  Node* nm1 = dynamic_cast<Node*>(line_master_element()->nodes()[0]);
  Node* nm2 = dynamic_cast<Node*>(line_master_element()->nodes()[1]);

  vm[0] = nm1->xspatial()[0] - nm2->xspatial()[0];
  vm[1] = nm1->xspatial()[1] - nm2->xspatial()[1];
  vm[2] = nm1->xspatial()[2] - nm2->xspatial()[2];

  // calculate lengths
  const double lengthS = sqrt(vs[0] * vs[0] + vs[1] * vs[1] + vs[2] * vs[2]);
  const double lengthM = sqrt(vm[0] * vm[0] + vm[1] * vm[1] + vm[2] * vm[2]);

  // safety
  if (lengthS < 1e-12 or lengthM < 1e-12) FOUR_C_THROW("line elements of zero length!");

  // calculate scalar product
  double scaprod = vs[0] * vm[0] + vs[1] * vm[1] + vs[2] * vm[2];
  double scaledScaprod = scaprod / (lengthS * lengthM);
  double angleRad = acos(scaledScaprod);

  // check if we used the right angle
  bool switchSign = false;
  if (angleRad > 0.5 * M_PI)  // if angle is > 90 degrees
  {
    switchSign = true;

    // change sign of master vector
    vm[0] = -vm[0];
    vm[1] = -vm[1];
    vm[2] = -vm[2];

    scaprod = vs[0] * vm[0] + vs[1] * vm[1] + vs[2] * vm[2];
    scaledScaprod = scaprod / (lengthS * lengthM);
    angleRad = acos(scaledScaprod);
  }

  //===============================================================
  // linearization

  // delta lengthM
  std::vector<Core::Gen::Pairedvector<int, double>> DlM(3, 1000);
  Core::Gen::Pairedvector<int, double> DlengthM(1000);

  // change sign of master vectors linearization
  if (switchSign)
  {
    DlM[0][nm1->dofs()[0]] -= 1;
    DlM[0][nm2->dofs()[0]] += 1;
    DlM[1][nm1->dofs()[1]] -= 1;
    DlM[1][nm2->dofs()[1]] += 1;
    DlM[2][nm1->dofs()[2]] -= 1;
    DlM[2][nm2->dofs()[2]] += 1;
  }
  else
  {
    DlM[0][nm1->dofs()[0]] += 1;
    DlM[0][nm2->dofs()[0]] -= 1;
    DlM[1][nm1->dofs()[1]] += 1;
    DlM[1][nm2->dofs()[1]] -= 1;
    DlM[2][nm1->dofs()[2]] += 1;
    DlM[2][nm2->dofs()[2]] -= 1;
  }


  for (CI p = DlM[0].begin(); p != DlM[0].end(); ++p)
    (DlengthM)[p->first] += (p->second) * vm[0] * 1.0 / (lengthM);
  for (CI p = DlM[1].begin(); p != DlM[1].end(); ++p)
    (DlengthM)[p->first] += (p->second) * vm[1] * 1.0 / (lengthM);
  for (CI p = DlM[2].begin(); p != DlM[2].end(); ++p)
    (DlengthM)[p->first] += (p->second) * vm[2] * 1.0 / (lengthM);

  // delta lengthS
  std::vector<Core::Gen::Pairedvector<int, double>> DlS(3, 1000);
  Core::Gen::Pairedvector<int, double> DlengthS(1000);

  DlS[0][ns1->dofs()[0]] += 1;
  DlS[0][ns2->dofs()[0]] -= 1;
  DlS[1][ns1->dofs()[1]] += 1;
  DlS[1][ns2->dofs()[1]] -= 1;
  DlS[2][ns1->dofs()[2]] += 1;
  DlS[2][ns2->dofs()[2]] -= 1;

  for (CI p = DlS[0].begin(); p != DlS[0].end(); ++p)
    (DlengthS)[p->first] += (p->second) * vs[0] * 1.0 / (lengthS);
  for (CI p = DlS[1].begin(); p != DlS[1].end(); ++p)
    (DlengthS)[p->first] += (p->second) * vs[1] * 1.0 / (lengthS);
  for (CI p = DlS[2].begin(); p != DlS[2].end(); ++p)
    (DlengthS)[p->first] += (p->second) * vs[2] * 1.0 / (lengthS);

  // lin lengthS * lengthM
  Core::Gen::Pairedvector<int, double> prodLength(1000);

  for (CI p = DlengthS.begin(); p != DlengthS.end(); ++p)
    (prodLength)[p->first] += (p->second) * lengthM;
  for (CI p = DlengthM.begin(); p != DlengthM.end(); ++p)
    (prodLength)[p->first] += (p->second) * lengthS;

  // lin scaprod
  Core::Gen::Pairedvector<int, double> scaProdlin(1000);

  for (CI p = DlS[0].begin(); p != DlS[0].end(); ++p) (scaProdlin)[p->first] += (p->second) * vm[0];
  for (CI p = DlS[1].begin(); p != DlS[1].end(); ++p) (scaProdlin)[p->first] += (p->second) * vm[1];
  for (CI p = DlS[2].begin(); p != DlS[2].end(); ++p) (scaProdlin)[p->first] += (p->second) * vm[2];

  for (CI p = DlM[0].begin(); p != DlM[0].end(); ++p) (scaProdlin)[p->first] += (p->second) * vs[0];
  for (CI p = DlM[1].begin(); p != DlM[1].end(); ++p) (scaProdlin)[p->first] += (p->second) * vs[1];
  for (CI p = DlM[2].begin(); p != DlM[2].end(); ++p) (scaProdlin)[p->first] += (p->second) * vs[2];

  // lin scaprod/lengthprod
  Core::Gen::Pairedvector<int, double> scaProdnormalizedLin(1000);
  for (CI p = scaProdlin.begin(); p != scaProdlin.end(); ++p)
    (scaProdnormalizedLin)[p->first] += (p->second) * 1.0 / (lengthS * lengthM);
  for (CI p = prodLength.begin(); p != prodLength.end(); ++p)
    (scaProdnormalizedLin)[p->first] -=
        (p->second) * scaprod * 1.0 / (lengthS * lengthM * lengthS * lengthM);

  // lin acos(scaledscaprod)
  double fac = (-1.0 / (sqrt(1.0 - scaledScaprod * scaledScaprod)));
  //  if(switchSign)
  //    fac *= -1.0;

  for (CI p = scaProdnormalizedLin.begin(); p != scaProdnormalizedLin.end(); ++p)
    (lineAngle)[p->first] += (p->second) * fac;

  // bye bye
  return angleRad;
}

FOUR_C_NAMESPACE_CLOSE
