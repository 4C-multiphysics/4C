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
  // clear entries of target vertices
  done_before().clear();

  // loop over all found target elements
  for (int nele = 0; nele < number_surface_elements(); ++nele)
  {
    // set internal counter
    curr_ele() = nele;

    // 1. init internal data
    initialize();

    // 2. create aux plane for target ele
    auxiliary_plane();  //--> build everything based on line element

    // 3. create aux line for source ele
    auxiliary_line();

    // 4. check orientation
    if (!check_orientation()) return;

    // 5. project target nodes onto auxplane
    project_target();

    // 6. project source line elements onto auxplane
    project_source();

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

  // clear all source and target vertices
  source_vertices().clear();
  target_vertices().clear();

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
  const double source_min_edge_size = parent_element().min_edge_size();
  const double target_min_edge_size = surface_element().min_edge_size();
  const double tol = 0.001 * std::min(source_min_edge_size, target_min_edge_size);

  // -------------------------------------------
  // CHECK LINE TO SURFACE ORIENTATION!
  // calculate line ele vector
  std::array<double, 3> lvec = {0.0, 0.0, 0.0};
  Node* source_node_1 = dynamic_cast<Node*>(line_element()->nodes()[0]);
  Node* source_node_2 = dynamic_cast<Node*>(line_element()->nodes()[1]);
  lvec[0] = source_node_1->xspatial()[0] - source_node_2->xspatial()[0];
  lvec[1] = source_node_1->xspatial()[1] - source_node_2->xspatial()[1];
  lvec[2] = source_node_1->xspatial()[2] - source_node_2->xspatial()[2];

  // calculate lengths
  const double length_source = sqrt(lvec[0] * lvec[0] + lvec[1] * lvec[1] + lvec[2] * lvec[2]);
  const double lengthA = sqrt(auxn_surf()[0] * auxn_surf()[0] + auxn_surf()[1] * auxn_surf()[1] +
                              auxn_surf()[2] * auxn_surf()[2]);
  const double prod = length_source * lengthA;
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
  auto shapefcn = Teuchos::getIntegralValue<Mortar::ShapeFcn>(imortar_, "LM_SHAPEFCN");
  auto consistent =
      Teuchos::getIntegralValue<Mortar::ConsistentDualType>(imortar_, "LM_DUAL_CONSISTENT");

  if (shapefcn != Mortar::shape_dual && shapefcn != Mortar::shape_petrovgalerkin) return;

  if (consistent == Mortar::consistent_none) return;

  if (i_type() == LineToSurfaceCoupling3d::lts) return;
  FOUR_C_THROW("consistent dual shapes for stl is experimental!");

  // source nodes and dofs
  const int max_nnodes = 9;
  const int nnodes = surface_element().num_node();
  if (nnodes > max_nnodes)
    FOUR_C_THROW(
        "this function is not implemented to handle elements with that many nodes. Just adjust "
        "max_nnodes above");
  const int ndof = 3;

  // get number of target nodes
  int target_nodes = line_element()->num_node();

  // Dual shape functions coefficient matrix and linearization
  Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes, true);
  surface_element().mo_data().deriv_dual_shape() =
      std::make_shared<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>(
          (nnodes + target_nodes) * ndof, 0, Core::LinAlg::SerialDenseMatrix(nnodes, nnodes));
  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& derivae =
      *(surface_element().mo_data().deriv_dual_shape());

  // various variables
  double detg = 0.0;
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // initialize matrices de and me
  Core::LinAlg::SerialDenseMatrix me(nnodes, nnodes, true);
  Core::LinAlg::SerialDenseMatrix de(nnodes, nnodes, true);

  Core::Gen::Pairedvector<int, Core::LinAlg::Matrix<max_nnodes + 1, max_nnodes>> derivde_new(
      (nnodes + target_nodes) * ndof);

  // two-dim arrays of maps for linearization of me/de
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> derivme(nnodes,
      std::vector<Core::Gen::Pairedvector<int, double>>(nnodes, (nnodes + target_nodes) * ndof));
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> derivde(nnodes,
      std::vector<Core::Gen::Pairedvector<int, double>>(nnodes, (nnodes + target_nodes) * ndof));

  double A_tot = 0.;

  // get number of target nodes
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

    // project Gauss point onto source integration element
    double source_xi[2] = {0.0, 0.0};
    double sprojalpha = 0.0;
    Mortar::Projector::impl(surface_element())
        ->project_gauss_point_auxn_3d(globgp, auxn(), surface_element(), source_xi, sprojalpha);

    // project Gauss point onto source (parent) element
    double p_source_xi[2] = {0., 0.};
    for (int i = 0; i < 2; ++i) p_source_xi[i] = source_xi[i];

    // create vector for shape function evaluation
    Core::LinAlg::SerialDenseVector source_val(nnodes);
    Core::LinAlg::SerialDenseMatrix source_deriv(nnodes, 2, true);

    // evaluate trace space shape functions at Gauss point
    surface_element().evaluate_shape(p_source_xi, source_val, source_deriv, nnodes);

    // additional data for contact calculation (i.e. incl. derivative of dual shape functions
    // coefficient matrix) GP source coordinate derivatives
    std::vector<Core::Gen::Pairedvector<int, double>> d_source_xi_gp(2, (nnodes + ncol) * ndof);
    // GP source coordinate derivatives
    std::vector<Core::Gen::Pairedvector<int, double>> dp_source_xigp(2, (nnodes + ncol) * ndof);
    // global GP coordinate derivative on integration element
    Core::Gen::Pairedvector<int, Core::LinAlg::Matrix<3, 1>> lingp((nnodes + ncol) * ndof);

    // compute global GP coordinate derivative
    static Core::LinAlg::Matrix<3, 1> svalcell;
    static Core::LinAlg::Matrix<3, 2> sderivcell;
    currcell->evaluate_shape(eta, svalcell, sderivcell);

    for (int v = 0; v < 2; ++v)
      for (int d = 0; d < 3; ++d)
        for (CI p = (currcell->get_deriv_vertex(v))[d].begin();
            p != (currcell->get_deriv_vertex(v))[d].end(); ++p)
          lingp[p->first](d) += svalcell(v) * (p->second);

    // compute GP source coordinate derivatives
    integrator.deriv_xi_gp_3d_aux_plane(surface_element(), source_xi, currcell->auxn(),
        d_source_xi_gp, sprojalpha, currcell->get_deriv_auxn(), lingp);

    // compute GP source coordinate derivatives (parent element)
    dp_source_xigp = d_source_xi_gp;

    double fac = 0.;
    for (CI p = derivjaccell.begin(); p != derivjaccell.end(); ++p)
    {
      Core::LinAlg::Matrix<max_nnodes + 1, max_nnodes>& dtmp = derivde_new[p->first];
      const double& ps = p->second;
      for (int j = 0; j < nnodes; ++j)
      {
        fac = wgt * source_val[j] * ps;
        dtmp(nnodes, j) += fac;
        for (int k = 0; k < nnodes; ++k) dtmp(k, j) += fac * source_val[k];
      }
    }

    for (int i = 0; i < 2; ++i)
      for (CI p = dp_source_xigp[i].begin(); p != dp_source_xigp[i].end(); ++p)
      {
        Core::LinAlg::Matrix<max_nnodes + 1, max_nnodes>& dtmp = derivde_new[p->first];
        const double& ps = p->second;
        for (int j = 0; j < nnodes; ++j)
        {
          fac = wgt * source_deriv(j, i) * detg * ps;
          dtmp(nnodes, j) += fac;
          for (int k = 0; k < nnodes; ++k)
          {
            dtmp(k, j) += fac * source_val[k];
            dtmp(j, k) += fac * source_val[k];
          }
        }
      }

    // computing de, derivde and me, derivme and kappa, derivkappa
    for (int j = 0; j < nnodes; ++j)
    {
      double fac;
      fac = source_val[j] * wgt;
      // computing de
      de(j, j) += fac * detg;

      for (int k = 0; k < nnodes; ++k)
      {
        // computing me
        fac = wgt * source_val[j] * source_val[k];
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
  using CIM = Core::Gen::Pairedvector<int,
      Core::LinAlg::Matrix<max_nnodes + 1, max_nnodes>>::const_iterator;
  for (CIM p = derivde_new.begin(); p != derivde_new.end(); ++p)
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

  // store ae matrix in source element data container
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
  const double source_min_edge_size = parent_element().min_edge_size();
  const double target_min_edge_size = surface_element().min_edge_size();
  const double tol = MORTARCLIPTOL * std::min(source_min_edge_size, target_min_edge_size);

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
  const double source_min_edge_size = parent_element().min_edge_size();
  const double target_min_edge_size = surface_element().min_edge_size();
  const double tol = MORTARCLIPTOL * std::min(source_min_edge_size, target_min_edge_size);

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
  // check of both target vertices are out of line in 0 direction
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


  // check of both target vertices are out of line in 1 direction
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
    inter_sections().emplace_back(
        edgeVertex0.coord(), Mortar::Vertex::target, edgeVertex0.nodeids(), nullptr);

    inter_sections().emplace_back(
        edgeVertex1.coord(), Mortar::Vertex::target, edgeVertex1.nodeids(), nullptr);
  }
  // ========================================================
  // 2.: nodes on each other
  else if (e0isV1 and e1isV0)
  {
    if (out) std::cout << "CASE 2" << std::endl;

    inter_sections().emplace_back(
        edgeVertex0.coord(), Mortar::Vertex::target, edgeVertex0.nodeids(), nullptr);

    inter_sections().emplace_back(
        edgeVertex1.coord(), Mortar::Vertex::target, edgeVertex1.nodeids(), nullptr);
  }
  // ========================================================
  // 3.: e0 on v0 and e1 valid
  else if (e0isV0 and e1v0 and e1v1)
  {
    if (out) std::cout << "CASE 3" << std::endl;

    inter_sections().emplace_back(
        edgeVertex0.coord(), Mortar::Vertex::target, edgeVertex0.nodeids(), nullptr);

    inter_sections().emplace_back(
        edgeVertex1.coord(), Mortar::Vertex::target, edgeVertex1.nodeids(), nullptr);
  }
  // ========================================================
  // 4.: e1 on v0 and e0 valid
  else if (e1isV0 and e0v0 and e0v1)
  {
    if (out) std::cout << "CASE 4" << std::endl;

    inter_sections().emplace_back(
        edgeVertex0.coord(), Mortar::Vertex::target, edgeVertex0.nodeids(), nullptr);

    inter_sections().emplace_back(
        edgeVertex1.coord(), Mortar::Vertex::target, edgeVertex1.nodeids(), nullptr);
  }
  // ========================================================
  // 5.: e0 on v1 and e1 valid
  else if (e0isV1 and e1v0 and e1v1)
  {
    if (out) std::cout << "CASE 5" << std::endl;

    inter_sections().emplace_back(
        edgeVertex0.coord(), Mortar::Vertex::target, edgeVertex0.nodeids(), nullptr);

    inter_sections().emplace_back(
        edgeVertex1.coord(), Mortar::Vertex::target, edgeVertex1.nodeids(), nullptr);
  }
  // ========================================================
  // 6.: e1 on v1 and e0 valid
  else if (e1isV1 and e0v0 and e0v1)
  {
    if (out) std::cout << "CASE 6" << std::endl;

    inter_sections().emplace_back(
        edgeVertex0.coord(), Mortar::Vertex::target, edgeVertex0.nodeids(), nullptr);

    inter_sections().emplace_back(
        edgeVertex1.coord(), Mortar::Vertex::target, edgeVertex1.nodeids(), nullptr);
  }
  // ========================================================
  // 7.: e0 on v0 and e1 out of v1 but in v0
  else if (e0isV0 and e1v0 and !e1v1)
  {
    if (out) std::cout << "CASE 7" << std::endl;

    inter_sections().emplace_back(
        edgeVertex0.coord(), Mortar::Vertex::target, edgeVertex0.nodeids(), nullptr);

    inter_sections().emplace_back(
        lineVertex1.coord(), Mortar::Vertex::projsource, lineVertex1.nodeids(), nullptr);
  }
  // ========================================================
  // 8.: e1 on v0 and e0 out of v1 but in v0
  else if (e1isV0 and e0v0 and !e0v1)
  {
    if (out) std::cout << "CASE 8" << std::endl;

    inter_sections().emplace_back(
        edgeVertex1.coord(), Mortar::Vertex::target, edgeVertex1.nodeids(), nullptr);

    inter_sections().emplace_back(
        lineVertex1.coord(), Mortar::Vertex::projsource, lineVertex1.nodeids(), nullptr);
  }
  // ========================================================
  // 9.: e1 on v1 and e0 out of v0 but in v1
  else if (e1isV1 and !e0v0 and e0v1)
  {
    if (out) std::cout << "CASE 9" << std::endl;

    inter_sections().emplace_back(
        edgeVertex1.coord(), Mortar::Vertex::target, edgeVertex1.nodeids(), nullptr);

    inter_sections().emplace_back(
        lineVertex0.coord(), Mortar::Vertex::projsource, lineVertex0.nodeids(), nullptr);
  }
  // ========================================================
  // 10.: e0 on v1 and e1 out of v0 but in v1
  else if (e0isV1 and !e1v0 and e1v1)
  {
    if (out) std::cout << "CASE 10" << std::endl;

    inter_sections().emplace_back(
        edgeVertex0.coord(), Mortar::Vertex::target, edgeVertex0.nodeids(), nullptr);

    inter_sections().emplace_back(
        lineVertex0.coord(), Mortar::Vertex::projsource, lineVertex0.nodeids(), nullptr);
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
  // 15.: all true --> both intersections target nodes
  else if (e0v1 and e1v1 and e0v0 and e1v0)
  {
    if (out) std::cout << "CASE 15" << std::endl;

    inter_sections().emplace_back(
        edgeVertex0.coord(), Mortar::Vertex::target, edgeVertex0.nodeids(), nullptr);

    inter_sections().emplace_back(
        edgeVertex1.coord(), Mortar::Vertex::target, edgeVertex1.nodeids(), nullptr);
  }
  // ========================================================
  // 16.: all source nodes are projected: E0 out of V0  and E1 out of V1
  else if (!e0v0 and e1v0 and e0v1 and !e1v1)
  {
    if (out) std::cout << "CASE 16" << std::endl;

    inter_sections().emplace_back(
        lineVertex0.coord(), Mortar::Vertex::projsource, lineVertex0.nodeids(), nullptr);

    inter_sections().emplace_back(
        lineVertex1.coord(), Mortar::Vertex::projsource, lineVertex1.nodeids(), nullptr);
  }
  // ========================================================
  // 17.: all source nodes are projected: E1 out of V0  and E0 out of V1
  else if (e0v0 and !e1v0 and !e0v1 and e1v1)
  {
    if (out) std::cout << "CASE 17" << std::endl;

    inter_sections().emplace_back(
        lineVertex0.coord(), Mortar::Vertex::projsource, lineVertex0.nodeids(), nullptr);

    inter_sections().emplace_back(
        lineVertex1.coord(), Mortar::Vertex::projsource, lineVertex1.nodeids(), nullptr);
  }
  // ========================================================
  // 18.: mixed: E0 and E1 pos to V0 and E0 pos to V1
  else if (e0v0 and e1v0 and e0v1 and !e1v1)
  {
    if (out) std::cout << "CASE 18" << std::endl;

    inter_sections().emplace_back(
        edgeVertex0.coord(), Mortar::Vertex::target, edgeVertex0.nodeids(), nullptr);

    inter_sections().emplace_back(
        lineVertex1.coord(), Mortar::Vertex::projsource, lineVertex1.nodeids(), nullptr);
  }
  // ========================================================
  // 19.: mixed: E0 and E1 pos to V0 and E1 pos to V1
  else if (e0v0 and e1v0 and !e0v1 and e1v1)
  {
    if (out) std::cout << "CASE 19" << std::endl;

    inter_sections().emplace_back(
        edgeVertex1.coord(), Mortar::Vertex::target, edgeVertex1.nodeids(), nullptr);

    inter_sections().emplace_back(
        lineVertex1.coord(), Mortar::Vertex::projsource, lineVertex1.nodeids(), nullptr);
  }
  // ========================================================
  // 20.: mixed: E0 neg and E1 pos to V0 and E0 and E1 pos to V1
  else if (!e0v0 and e1v0 and e0v1 and e1v1)
  {
    if (out) std::cout << "CASE 20" << std::endl;

    inter_sections().emplace_back(
        edgeVertex1.coord(), Mortar::Vertex::target, edgeVertex1.nodeids(), nullptr);

    inter_sections().emplace_back(
        lineVertex0.coord(), Mortar::Vertex::projsource, lineVertex0.nodeids(), nullptr);
  }
  // ========================================================
  // 21.: mixed: E1 neg and E0 pos to V0 and E0 and E1 pos to V1
  else if (e0v0 and !e1v0 and e0v1 and e1v1)
  {
    if (out) std::cout << "CASE 21" << std::endl;

    inter_sections().emplace_back(
        edgeVertex0.coord(), Mortar::Vertex::target, edgeVertex0.nodeids(), nullptr);

    inter_sections().emplace_back(
        lineVertex0.coord(), Mortar::Vertex::projsource, lineVertex0.nodeids(), nullptr);
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
  const double source_min_edge_size = parent_element().min_edge_size();
  const double target_min_edge_size = surface_element().min_edge_size();
  const double tol = MORTARCLIPTOL * std::min(source_min_edge_size, target_min_edge_size);

  // vector with vertices
  inter_sections().clear();
  temp_inter_sections().clear();

  // safety
  if (target_vertices().size() < 3) FOUR_C_THROW("Invalid number of Target Vertices!");
  if (source_vertices().size() != 2) FOUR_C_THROW("Invalid number of Source Vertices!");

  // set next Vertex pointer for all elements in lists
  for (int i = 0; i < static_cast<int>(target_vertices().size()); ++i)
    target_vertices()[i].assign_next(
        &target_vertices()[(i + 1) % static_cast<int>(target_vertices().size())]);

  // flip ordering
  std::reverse(source_vertices().begin(), source_vertices().end());

  // create line from source vertices
  std::array<double, 3> sourceLine = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k)
    sourceLine[k] = source_vertices()[1].coord()[k] - source_vertices()[0].coord()[k];


  // check for parallelity of line and edges and perform line to line clipping
  bool foundValidParallelity = false;

  // loop over target vertices to create target polygon lines
  for (int j = 0; j < (int)target_vertices().size(); ++j)
  {
    // we need one edge first
    std::array<double, 3> edge = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k)
      edge[k] = (target_vertices()[j].next())->coord()[k] - target_vertices()[j].coord()[k];

    // outward edge normals of polygon and source line
    std::array<double, 3> np = {0.0, 0.0, 0.0};
    std::array<double, 3> nl = {0.0, 0.0, 0.0};
    np[0] = edge[1] * auxn_surf()[2] - edge[2] * auxn_surf()[1];
    np[1] = edge[2] * auxn_surf()[0] - edge[0] * auxn_surf()[2];
    np[2] = edge[0] * auxn_surf()[1] - edge[1] * auxn_surf()[0];
    nl[0] = sourceLine[1] * auxn_surf()[2] - sourceLine[2] * auxn_surf()[1];
    nl[1] = sourceLine[2] * auxn_surf()[0] - sourceLine[0] * auxn_surf()[2];
    nl[2] = sourceLine[0] * auxn_surf()[1] - sourceLine[1] * auxn_surf()[0];

    if (out)
    {
      std::cout << "==============================================" << std::endl;
      std::cout << "SLine= " << sourceLine[0] << "  " << sourceLine[1] << "  " << sourceLine[2]
                << std::endl;
      std::cout << "Pos1= " << source_vertices()[0].coord()[0] << "  "
                << source_vertices()[0].coord()[1] << "  " << source_vertices()[0].coord()[2]
                << std::endl;
      std::cout << "Pos2= " << source_vertices()[1].coord()[0] << "  "
                << source_vertices()[1].coord()[1] << "  " << source_vertices()[1].coord()[2]
                << std::endl;
      std::cout << "N source= " << nl[0] << "  " << nl[1] << "  " << nl[2] << std::endl;


      std::cout << "==============================================" << std::endl;
      std::cout << "MEdge= " << edge[0] << "  " << edge[1] << "  " << edge[2] << std::endl;
      std::cout << "Pos1= " << (target_vertices()[j].next())->coord()[0] << "  "
                << (target_vertices()[j].next())->coord()[1] << "  "
                << (target_vertices()[j].next())->coord()[2] << std::endl;
      std::cout << "Pos2= " << target_vertices()[j].coord()[0] << "  "
                << target_vertices()[j].coord()[1] << "  " << target_vertices()[j].coord()[2]
                << std::endl;
      std::cout << "N target= " << np[0] << "  " << np[1] << "  " << np[2] << std::endl;
    }

    // check for parallelity of edges
    double parallel = edge[0] * nl[0] + edge[1] * nl[1] + edge[2] * nl[2];
    if (abs(parallel) < tol)
    {
      // safety checks
      if (target_vertices()[j].next()->nodeids().size() > 1)
        FOUR_C_THROW("Only one node id per target vertex allowed!");
      if (target_vertices()[j].nodeids().size() > 1)
        FOUR_C_THROW("Only one node id per target vertex allowed!");

      // store target node ids in set to guarantee uniqueness
      std::pair<int, int> actIDs = std::pair<int, int>(
          target_vertices()[j].next()->nodeids()[0], target_vertices()[j].nodeids()[0]);
      std::pair<int, int> actIDsTw = std::pair<int, int>(
          target_vertices()[j].nodeids()[0], target_vertices()[j].next()->nodeids()[0]);

      // check if edge on line element
      foundValidParallelity = check_line_on_line(*target_vertices()[j].next(), target_vertices()[j],
          source_vertices()[1], source_vertices()[0]);

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
          line_to_line_clipping(*target_vertices()[j].next(), target_vertices()[j],
              source_vertices()[1], source_vertices()[0]);

          if (out)
            std::cout << "TARGET IDS = " << target_vertices()[j].next()->nodeids()[0] << "  "
                      << target_vertices()[j].nodeids()[0] << std::endl;
          break;
        }
        else
          continue;
      }
    }
  }  // end target vertex loop

  // if there is a line to line setting --> jump to node check
  if (!foundValidParallelity)
  {
    // loop over target vertices to create target polygon lines
    for (int j = 0; j < (int)target_vertices().size(); ++j)
    {
      // we need two edges first
      std::array<double, 3> edge = {0.0, 0.0, 0.0};
      for (int k = 0; k < 3; ++k)
        edge[k] = (target_vertices()[j].next())->coord()[k] - target_vertices()[j].coord()[k];

      // outward edge normals of polygon and source line
      std::array<double, 3> np = {0.0, 0.0, 0.0};
      std::array<double, 3> nl = {0.0, 0.0, 0.0};
      np[0] = edge[1] * auxn_surf()[2] - edge[2] * auxn_surf()[1];
      np[1] = edge[2] * auxn_surf()[0] - edge[0] * auxn_surf()[2];
      np[2] = edge[0] * auxn_surf()[1] - edge[1] * auxn_surf()[0];
      nl[0] = sourceLine[1] * auxn_surf()[2] - sourceLine[2] * auxn_surf()[1];
      nl[1] = sourceLine[2] * auxn_surf()[0] - sourceLine[0] * auxn_surf()[2];
      nl[2] = sourceLine[0] * auxn_surf()[1] - sourceLine[1] * auxn_surf()[0];

      if (out)
      {
        std::cout << "==============================================" << std::endl;
        std::cout << "SLine= " << sourceLine[0] << "  " << sourceLine[1] << "  " << sourceLine[2]
                  << std::endl;
        std::cout << "Pos1= " << source_vertices()[0].coord()[0] << "  "
                  << source_vertices()[0].coord()[1] << "  " << source_vertices()[0].coord()[2]
                  << std::endl;
        std::cout << "Pos2= " << source_vertices()[1].coord()[0] << "  "
                  << source_vertices()[1].coord()[1] << "  " << source_vertices()[1].coord()[2]
                  << std::endl;
        std::cout << "N source= " << nl[0] << "  " << nl[1] << "  " << nl[2] << std::endl;


        std::cout << "==============================================" << std::endl;
        std::cout << "MEdge= " << edge[0] << "  " << edge[1] << "  " << edge[2] << std::endl;
        std::cout << "Pos1= " << (target_vertices()[j].next())->coord()[0] << "  "
                  << (target_vertices()[j].next())->coord()[1] << "  "
                  << (target_vertices()[j].next())->coord()[2] << std::endl;
        std::cout << "Pos2= " << target_vertices()[j].coord()[0] << "  "
                  << target_vertices()[j].coord()[1] << "  " << target_vertices()[j].coord()[2]
                  << std::endl;
        std::cout << "N target= " << np[0] << "  " << np[1] << "  " << np[2] << std::endl;
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
        wec_p1 += (source_vertices()[0].coord()[k] - target_vertices()[j].coord()[k]) * np[k];
        wec_p2 += (source_vertices()[1].coord()[k] - target_vertices()[j].coord()[k]) * np[k];
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
          wec_q1 += (target_vertices()[j].coord()[k] - source_vertices()[0].coord()[k]) * nl[k];
          wec_q2 +=
              ((target_vertices()[j].next())->coord()[k] - source_vertices()[0].coord()[k]) * nl[k];
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
            coords[k] = (1 - alpha) * source_vertices()[0].coord()[k] +
                        alpha * source_vertices()[1].coord()[k];
            if (abs(coords[k]) < tol) coords[k] = 0.0;
          }

          if (out)
          {
            std::cout << "Found intersection! (" << j << ") " << alpha << std::endl;
            std::cout << "coords 1: " << coords[0] << " " << coords[1] << " " << coords[2]
                      << std::endl;
          }

          // generate vectors of underlying node ids for lineclip (2x source, 2x target)
          std::vector<int> lcids(4);
          lcids[0] = (int)(source_vertices()[0].nodeids()[0]);
          lcids[1] = (int)(source_vertices()[1].nodeids()[0]);
          lcids[2] = (int)(target_vertices()[j].nodeids()[0]);
          lcids[3] = (int)((target_vertices()[j].next())->nodeids()[0]);

          // store intersection points
          temp_inter_sections().emplace_back(
              coords, Mortar::Vertex::lineclip, lcids, &(source_vertices()[1]));
        }
      }
    }  // end vertex loop

    // ===================================================
    // find interior node intersections
    //    if((int)temp_inter_sections().size()!=2)
    {
      for (int i = 0; i < (int)source_vertices().size(); ++i)
      {
        // keep track of inside / outside status
        bool outside = false;

        // check against all poly1 (source) edges
        for (int j = 0; j < (int)target_vertices().size(); ++j)
        {
          // we need diff vector and edge2 first
          std::array<double, 3> diff = {0.0, 0.0, 0.0};
          std::array<double, 3> edge = {0.0, 0.0, 0.0};
          for (int k = 0; k < 3; ++k)
          {
            diff[k] = source_vertices()[i].coord()[k] - target_vertices()[j].coord()[k];
            edge[k] = (target_vertices()[j].next())->coord()[k] - target_vertices()[j].coord()[k];
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
        }  // end target loop

        if (outside)
        {
          // next source vertex
          continue;
        }
        else
        {
          temp_inter_sections().emplace_back(source_vertices()[i].coord(),
              Mortar::Vertex::projsource, source_vertices()[i].nodeids(), nullptr);
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

      // check against all poly1 (source) points
      for (int j = 0; j < (int)source_vertices().size(); ++j)
      {
        // distance vector
        std::array<double, 3> diff = {0.0, 0.0, 0.0};
        for (int k = 0; k < 3; ++k)
          diff[k] = temp_inter_sections()[i].coord()[k] - source_vertices()[j].coord()[k];
        double dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

        // only keep intersection point if not close
        if (dist <= tol)
        {
          // intersection is close to source vertex!
          close = true;

          // store source vertex as intersection point
          inter_sections().emplace_back(source_vertices()[j].coord(), Mortar::Vertex::projsource,
              source_vertices()[j].nodeids(), nullptr);
          break;
        }
      }

      // do only if no close source point found
      if (!close)
      {
        // check against all poly2 (target) points
        for (int j = 0; j < (int)target_vertices().size(); ++j)
        {
          // distance vector
          std::array<double, 3> diff = {0.0, 0.0, 0.0};
          for (int k = 0; k < 3; ++k)
            diff[k] = temp_inter_sections()[i].coord()[k] - target_vertices()[j].coord()[k];
          double dist = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

          // only keep intersection point if not close
          if (dist <= tol)
          {
            // intersection is close to target vertex!
            close = true;

            inter_sections().emplace_back(target_vertices()[j].coord(), Mortar::Vertex::target,
                target_vertices()[j].nodeids(), nullptr);
            break;
          }
        }
      }

      // keep intersection point only if not close to any Source/Target point
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
}

/*----------------------------------------------------------------------*
 |  linearize vertices                                       farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToSurfaceCoupling3d::linearize_vertices(
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& linvertex)
{
  // linearize all aux.plane source and target nodes only ONCE
  // and use these linearizations later during lineclip linearization
  // (this speeds up the vertex linearizations in most cases, as we
  // never linearize the SAME source or target vertex more than once)

  // number of nodes
  const int num_source_nodes = line_element()->num_node();
  const int num_target_nodes = surface_element().num_node();

  // prepare storage for source and target linearizations
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> lin_source_nodes(num_source_nodes,
      std::vector<Core::Gen::Pairedvector<int, double>>(
          3, 100 + linsize_ + 3 * line_element()->num_node() + 3 * surface_element().num_node()));
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> lin_target_nodes(num_target_nodes,
      std::vector<Core::Gen::Pairedvector<int, double>>(
          3, 100 + linsize_ + 3 * line_element()->num_node() + 3 * surface_element().num_node()));

  // compute source linearizations (num_source_nodes)
  source_vertex_linearization(lin_source_nodes);

  // compute target linearizations (num_target_nodes)
  target_vertex_linearization(lin_target_nodes);

  //**********************************************************************
  // Line vertex linearization
  //**********************************************************************
  // loop over all clip Intersections vertices
  for (int i = 0; i < (int)inter_sections().size(); ++i)
  {
    // references to current vertex and its linearization
    Mortar::Vertex& currv = inter_sections()[i];
    std::vector<Core::Gen::Pairedvector<int, double>>& currlin = linvertex[i];

    // decision on vertex type (source, projtarget, linclip)
    if (currv.v_type() == Mortar::Vertex::projsource)
    {
      // get corresponding source id
      int source_id = currv.nodeids()[0];

      // find corresponding source node linearization
      int k = 0;
      while (k < num_source_nodes)
      {
        if (line_element()->node_ids()[k] == source_id) break;
        ++k;
      }

      // FOUR_C_THROW if not found
      if (k == num_source_nodes) FOUR_C_THROW("Source Id not found!");

      // get the correct source node linearization
      currlin = lin_source_nodes[k];
    }
    else if (currv.v_type() == Mortar::Vertex::target)
    {
      // get corresponding target id
      int target_id = currv.nodeids()[0];

      // find corresponding target node linearization
      int k = 0;
      while (k < num_target_nodes)
      {
        if (surface_element().node_ids()[k] == target_id) break;
        ++k;
      }

      // FOUR_C_THROW if not found
      if (k == num_target_nodes) FOUR_C_THROW("Target Id not found!");

      // get the correct target node linearization
      currlin = lin_target_nodes[k];
    }
    else if (currv.v_type() == Mortar::Vertex::lineclip)
    {
      // get references to the two source vertices
      int source_index_1 = -1;
      int source_index_2 = -1;
      for (int j = 0; j < (int)source_vertices().size(); ++j)
      {
        if (source_vertices()[j].nodeids()[0] == currv.nodeids()[0]) source_index_1 = j;
        if (source_vertices()[j].nodeids()[0] == currv.nodeids()[1]) source_index_2 = j;
      }
      if (source_index_1 < 0 || source_index_2 < 0 || source_index_1 == source_index_2)
        FOUR_C_THROW("Lineclip linearization: (S) Something went wrong!");

      Mortar::Vertex* source_vertex_1 = &source_vertices()[source_index_1];
      Mortar::Vertex* source_vertex_2 = &source_vertices()[source_index_2];

      // get references to the two target vertices
      int target_index_1 = -1;
      int target_index_2 = -1;
      for (int j = 0; j < (int)target_vertices().size(); ++j)
      {
        if (target_vertices()[j].nodeids()[0] == currv.nodeids()[2]) target_index_1 = j;
        if (target_vertices()[j].nodeids()[0] == currv.nodeids()[3]) target_index_2 = j;
      }
      if (target_index_1 < 0 || target_index_2 < 0 || target_index_1 == target_index_2)
        FOUR_C_THROW("Lineclip linearization: (M) Something went wrong!");

      Mortar::Vertex* target_vertex_1 = &target_vertices()[target_index_1];
      Mortar::Vertex* target_vertex_2 = &target_vertices()[target_index_2];

      // do lineclip vertex linearization
      lineclip_vertex_linearization(currv, currlin, source_vertex_1, source_vertex_2,
          target_vertex_1, target_vertex_2, lin_source_nodes, lin_target_nodes);
    }

    else
      FOUR_C_THROW("VertexLinearization: Invalid Vertex Type!");
  }
}

/*----------------------------------------------------------------------*
 |  Linearization of lineclip vertex (3D) AuxPlane            popp 03/09|
 *----------------------------------------------------------------------*/
void CONTACT::LineToSurfaceCoupling3d::lineclip_vertex_linearization(const Mortar::Vertex& currv,
    std::vector<Core::Gen::Pairedvector<int, double>>& currlin,
    const Mortar::Vertex* source_vertex_1, const Mortar::Vertex* source_vertex_2,
    const Mortar::Vertex* target_vertex_1, const Mortar::Vertex* target_vertex_2,
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& lin_source_nodes,
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& lin_target_nodes)
{
  // number of nodes
  const int num_source_nodes = line_element()->num_node();
  const int num_target_nodes = surface_element().num_node();

  // iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // compute factor Z
  std::array<double, 3> crossZ = {0.0, 0.0, 0.0};
  crossZ[0] = (source_vertex_1->coord()[1] - target_vertex_1->coord()[1]) *
                  (target_vertex_2->coord()[2] - target_vertex_1->coord()[2]) -
              (source_vertex_1->coord()[2] - target_vertex_1->coord()[2]) *
                  (target_vertex_2->coord()[1] - target_vertex_1->coord()[1]);
  crossZ[1] = (source_vertex_1->coord()[2] - target_vertex_1->coord()[2]) *
                  (target_vertex_2->coord()[0] - target_vertex_1->coord()[0]) -
              (source_vertex_1->coord()[0] - target_vertex_1->coord()[0]) *
                  (target_vertex_2->coord()[2] - target_vertex_1->coord()[2]);
  crossZ[2] = (source_vertex_1->coord()[0] - target_vertex_1->coord()[0]) *
                  (target_vertex_2->coord()[1] - target_vertex_1->coord()[1]) -
              (source_vertex_1->coord()[1] - target_vertex_1->coord()[1]) *
                  (target_vertex_2->coord()[0] - target_vertex_1->coord()[0]);
  double Zfac = crossZ[0] * auxn()[0] + crossZ[1] * auxn()[1] + crossZ[2] * auxn()[2];

  // compute factor N
  std::array<double, 3> crossN = {0.0, 0.0, 0.0};
  crossN[0] = (source_vertex_2->coord()[1] - source_vertex_1->coord()[1]) *
                  (target_vertex_2->coord()[2] - target_vertex_1->coord()[2]) -
              (source_vertex_2->coord()[2] - source_vertex_1->coord()[2]) *
                  (target_vertex_2->coord()[1] - target_vertex_1->coord()[1]);
  crossN[1] = (source_vertex_2->coord()[2] - source_vertex_1->coord()[2]) *
                  (target_vertex_2->coord()[0] - target_vertex_1->coord()[0]) -
              (source_vertex_2->coord()[0] - source_vertex_1->coord()[0]) *
                  (target_vertex_2->coord()[2] - target_vertex_1->coord()[2]);
  crossN[2] = (source_vertex_2->coord()[0] - source_vertex_1->coord()[0]) *
                  (target_vertex_2->coord()[1] - target_vertex_1->coord()[1]) -
              (source_vertex_2->coord()[1] - source_vertex_1->coord()[1]) *
                  (target_vertex_2->coord()[0] - target_vertex_1->coord()[0]);
  double Nfac = crossN[0] * auxn()[0] + crossN[1] * auxn()[1] + crossN[2] * auxn()[2];

  // source edge vector
  std::array<double, 3> sedge = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k) sedge[k] = source_vertex_2->coord()[k] - source_vertex_1->coord()[k];

  // prepare linearization derivZ
  std::array<double, 3> crossdZ1 = {0.0, 0.0, 0.0};
  std::array<double, 3> crossdZ2 = {0.0, 0.0, 0.0};
  std::array<double, 3> crossdZ3 = {0.0, 0.0, 0.0};
  crossdZ1[0] = (target_vertex_2->coord()[1] - target_vertex_1->coord()[1]) * auxn()[2] -
                (target_vertex_2->coord()[2] - target_vertex_1->coord()[2]) * auxn()[1];
  crossdZ1[1] = (target_vertex_2->coord()[2] - target_vertex_1->coord()[2]) * auxn()[0] -
                (target_vertex_2->coord()[0] - target_vertex_1->coord()[0]) * auxn()[2];
  crossdZ1[2] = (target_vertex_2->coord()[0] - target_vertex_1->coord()[0]) * auxn()[1] -
                (target_vertex_2->coord()[1] - target_vertex_1->coord()[1]) * auxn()[0];
  crossdZ2[0] = auxn()[1] * (source_vertex_1->coord()[2] - target_vertex_1->coord()[2]) -
                auxn()[2] * (source_vertex_1->coord()[1] - target_vertex_1->coord()[1]);
  crossdZ2[1] = auxn()[2] * (source_vertex_1->coord()[0] - target_vertex_1->coord()[0]) -
                auxn()[0] * (source_vertex_1->coord()[2] - target_vertex_1->coord()[2]);
  crossdZ2[2] = auxn()[0] * (source_vertex_1->coord()[1] - target_vertex_1->coord()[1]) -
                auxn()[1] * (source_vertex_1->coord()[0] - target_vertex_1->coord()[0]);
  crossdZ3[0] = (source_vertex_1->coord()[1] - target_vertex_1->coord()[1]) *
                    (target_vertex_2->coord()[2] - target_vertex_1->coord()[2]) -
                (source_vertex_1->coord()[2] - target_vertex_1->coord()[2]) *
                    (target_vertex_2->coord()[1] - target_vertex_1->coord()[1]);
  crossdZ3[1] = (source_vertex_1->coord()[2] - target_vertex_1->coord()[2]) *
                    (target_vertex_2->coord()[0] - target_vertex_1->coord()[0]) -
                (source_vertex_1->coord()[0] - target_vertex_1->coord()[0]) *
                    (target_vertex_2->coord()[2] - target_vertex_1->coord()[2]);
  crossdZ3[2] = (source_vertex_1->coord()[0] - target_vertex_1->coord()[0]) *
                    (target_vertex_2->coord()[1] - target_vertex_1->coord()[1]) -
                (source_vertex_1->coord()[1] - target_vertex_1->coord()[1]) *
                    (target_vertex_2->coord()[0] - target_vertex_1->coord()[0]);

  // prepare linearization derivN
  std::array<double, 3> crossdN1 = {0.0, 0.0, 0.0};
  std::array<double, 3> crossdN2 = {0.0, 0.0, 0.0};
  std::array<double, 3> crossdN3 = {0.0, 0.0, 0.0};
  crossdN1[0] = (target_vertex_2->coord()[1] - target_vertex_1->coord()[1]) * auxn()[2] -
                (target_vertex_2->coord()[2] - target_vertex_1->coord()[2]) * auxn()[1];
  crossdN1[1] = (target_vertex_2->coord()[2] - target_vertex_1->coord()[2]) * auxn()[0] -
                (target_vertex_2->coord()[0] - target_vertex_1->coord()[0]) * auxn()[2];
  crossdN1[2] = (target_vertex_2->coord()[0] - target_vertex_1->coord()[0]) * auxn()[1] -
                (target_vertex_2->coord()[1] - target_vertex_1->coord()[1]) * auxn()[0];
  crossdN2[0] = auxn()[1] * (source_vertex_2->coord()[2] - source_vertex_1->coord()[2]) -
                auxn()[2] * (source_vertex_2->coord()[1] - source_vertex_1->coord()[1]);
  crossdN2[1] = auxn()[2] * (source_vertex_2->coord()[0] - source_vertex_1->coord()[0]) -
                auxn()[0] * (source_vertex_2->coord()[2] - source_vertex_1->coord()[2]);
  crossdN2[2] = auxn()[0] * (source_vertex_2->coord()[1] - source_vertex_1->coord()[1]) -
                auxn()[1] * (source_vertex_2->coord()[0] - source_vertex_1->coord()[0]);
  crossdN3[0] = (source_vertex_2->coord()[1] - source_vertex_1->coord()[1]) *
                    (target_vertex_2->coord()[2] - target_vertex_1->coord()[2]) -
                (source_vertex_2->coord()[2] - source_vertex_1->coord()[2]) *
                    (target_vertex_2->coord()[1] - target_vertex_1->coord()[1]);
  crossdN3[1] = (source_vertex_2->coord()[2] - source_vertex_1->coord()[2]) *
                    (target_vertex_2->coord()[0] - target_vertex_1->coord()[0]) -
                (source_vertex_2->coord()[0] - source_vertex_1->coord()[0]) *
                    (target_vertex_2->coord()[2] - target_vertex_1->coord()[2]);
  crossdN3[2] = (source_vertex_2->coord()[0] - source_vertex_1->coord()[0]) *
                    (target_vertex_2->coord()[1] - target_vertex_1->coord()[1]) -
                (source_vertex_2->coord()[1] - source_vertex_1->coord()[1]) *
                    (target_vertex_2->coord()[0] - target_vertex_1->coord()[0]);

  // source vertex linearization (2x)
  int source_id_1 = currv.nodeids()[0];
  int source_id_2 = currv.nodeids()[1];

  // find corresponding source node linearizations
  int k = 0;
  while (k < num_source_nodes)
  {
    if (line_element()->node_ids()[k] == source_id_1) break;
    ++k;
  }

  // FOUR_C_THROW if not found
  if (k == num_source_nodes) FOUR_C_THROW("Source Id1 not found!");

  // get the correct source node linearization
  std::vector<Core::Gen::Pairedvector<int, double>>& sourcelin0 = lin_source_nodes[k];

  k = 0;
  while (k < num_source_nodes)
  {
    if (line_element()->node_ids()[k] == source_id_2) break;
    ++k;
  }

  // FOUR_C_THROW if not found
  if (k == num_source_nodes) FOUR_C_THROW("Source Id2 not found!");

  // get the correct source node linearization
  std::vector<Core::Gen::Pairedvector<int, double>>& sourcelin1 = lin_source_nodes[k];

  // target vertex linearization (2x)
  int target_id_1 = currv.nodeids()[2];
  int target_id_2 = currv.nodeids()[3];

  // find corresponding target node linearizations
  k = 0;
  while (k < num_target_nodes)
  {
    if (surface_element().node_ids()[k] == target_id_1) break;
    ++k;
  }

  // FOUR_C_THROW if not found
  if (k == num_target_nodes) FOUR_C_THROW("Target Id1 not found!");

  // get the correct target node linearization
  std::vector<Core::Gen::Pairedvector<int, double>>& targetlin0 = lin_target_nodes[k];

  k = 0;
  while (k < num_target_nodes)
  {
    if (surface_element().node_ids()[k] == target_id_2) break;
    ++k;
  }

  // FOUR_C_THROW if not found
  if (k == num_target_nodes) FOUR_C_THROW("Target Id2 not found!");

  // get the correct target node linearization
  std::vector<Core::Gen::Pairedvector<int, double>>& targetlin1 = lin_target_nodes[k];

  // linearization of element normal Auxn()
  std::vector<Core::Gen::Pairedvector<int, double>>& linauxn = get_deriv_auxn();

  const double ZNfac = Zfac / Nfac;
  const double ZNNfac = Zfac / (Nfac * Nfac);
  const double Nfacinv = 1.0 / Nfac;

  // bring everything together -> lineclip vertex linearization
  for (int k = 0; k < 3; ++k)
  {
    for (CI p = sourcelin0[k].begin(); p != sourcelin0[k].end(); ++p)
    {
      currlin[k][p->first] += (p->second);
      currlin[k][p->first] += ZNfac * (p->second);
      for (int dim = 0; dim < 3; ++dim)
      {
        currlin[dim][p->first] -= sedge[dim] * Nfacinv * crossdZ1[k] * (p->second);
        currlin[dim][p->first] -= sedge[dim] * ZNNfac * crossdN1[k] * (p->second);
      }
    }
    for (CI p = sourcelin1[k].begin(); p != sourcelin1[k].end(); ++p)
    {
      currlin[k][p->first] -= ZNfac * (p->second);
      for (int dim = 0; dim < 3; ++dim)
      {
        currlin[dim][p->first] += sedge[dim] * ZNNfac * crossdN1[k] * (p->second);
      }
    }
    for (CI p = targetlin0[k].begin(); p != targetlin0[k].end(); ++p)
    {
      for (int dim = 0; dim < 3; ++dim)
      {
        currlin[dim][p->first] += sedge[dim] * Nfacinv * crossdZ1[k] * (p->second);
        currlin[dim][p->first] += sedge[dim] * Nfacinv * crossdZ2[k] * (p->second);
        currlin[dim][p->first] -= sedge[dim] * ZNNfac * crossdN2[k] * (p->second);
      }
    }
    for (CI p = targetlin1[k].begin(); p != targetlin1[k].end(); ++p)
    {
      for (int dim = 0; dim < 3; ++dim)
      {
        currlin[dim][p->first] -= sedge[dim] * Nfacinv * crossdZ2[k] * (p->second);
        currlin[dim][p->first] += sedge[dim] * ZNNfac * crossdN2[k] * (p->second);
      }
    }
    for (CI p = linauxn[k].begin(); p != linauxn[k].end(); ++p)
    {
      for (int dim = 0; dim < 3; ++dim)
      {
        currlin[dim][p->first] -= sedge[dim] * Nfacinv * crossdZ3[k] * (p->second);
        currlin[dim][p->first] += sedge[dim] * ZNNfac * crossdN3[k] * (p->second);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  compute and check length of intLine                      farah 07/16|
 *----------------------------------------------------------------------*/
bool CONTACT::LineToSurfaceCoupling3d::check_length()
{
  // tolerance
  const double source_min_edge_size = parent_element().min_edge_size();
  const double target_min_edge_size = surface_element().min_edge_size();
  const double tol = MORTARCLIPTOL * std::min(source_min_edge_size, target_min_edge_size);

  // compute distance vector
  std::array<double, 3> v = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i)
    v[i] = inter_sections()[0].coord()[i] - inter_sections()[1].coord()[i];

  // compute length
  double length = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

  // check
  return length >= tol;
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
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  int nnodes = line_element()->num_node();
  if (nnodes != 2) FOUR_C_THROW("Auxiliary line calculation only for line2 elements!");

  // average nodal normals of line element
  linsize_ = 0;
  for (int i = 0; i < nnodes; ++i)
  {
    Core::Nodes::Node* node = idiscret_.g_node(line_element()->node_ids()[i]);
    if (!node) FOUR_C_THROW("Cannot find source element with gid %", line_element()->node_ids()[i]);
    Node* mycnode = dynamic_cast<Node*>(node);
    if (!mycnode) FOUR_C_THROW("project_source: Null pointer!");

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
    if (!node) FOUR_C_THROW("Cannot find source element with gid %", line_element()->node_ids()[i]);
    Node* mycnode = dynamic_cast<Node*>(node);
    if (!mycnode) FOUR_C_THROW("project_source: Null pointer!");

    auxn()[0] += 0.5 * mycnode->mo_data().n()[0];
    auxn()[1] += 0.5 * mycnode->mo_data().n()[1];
    auxn()[2] += 0.5 * mycnode->mo_data().n()[2];

    for (CI p = mycnode->data().get_deriv_n()[0].begin();
        p != mycnode->data().get_deriv_n()[0].end(); ++p)
      (dauxn[0])[p->first] += 0.5 * (p->second);
    for (CI p = mycnode->data().get_deriv_n()[1].begin();
        p != mycnode->data().get_deriv_n()[1].end(); ++p)
      (dauxn[1])[p->first] += 0.5 * (p->second);
    for (CI p = mycnode->data().get_deriv_n()[2].begin();
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
  if (!node) FOUR_C_THROW("Cannot find source element with gid %", line_element()->node_ids()[0]);
  Node* mycnode = dynamic_cast<Node*>(node);
  if (!mycnode) FOUR_C_THROW("project_source: Null pointer!");

  tangent[0] += mycnode->xspatial()[0];
  tangent[1] += mycnode->xspatial()[1];
  tangent[2] += mycnode->xspatial()[2];

  Core::Nodes::Node* node2 = idiscret_.g_node(line_element()->node_ids()[1]);
  if (!node2) FOUR_C_THROW("Cannot find source element with gid %", line_element()->node_ids()[1]);
  Node* mycnode2 = dynamic_cast<Node*>(node2);
  if (!mycnode2) FOUR_C_THROW("project_source: Null pointer!");

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

  for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    tplanex[0][p->first] -= tangent[0] * p->second;
  for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    tplanex[1][p->first] -= tangent[1] * p->second;
  for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    tplanex[2][p->first] -= tangent[2] * p->second;

  for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    tplaney[0][p->first] -= tangent[0] * p->second;
  for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    tplaney[1][p->first] -= tangent[1] * p->second;
  for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    tplaney[2][p->first] -= tangent[2] * p->second;

  for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    tplanez[0][p->first] -= tangent[0] * p->second;
  for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    tplanez[1][p->first] -= tangent[1] * p->second;
  for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    tplanez[2][p->first] -= tangent[2] * p->second;

  //------------
  for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    tplanex[0][p->first] -= tangent[0] * p->second;
  for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    tplanex[1][p->first] -= tangent[0] * p->second;
  for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    tplanex[2][p->first] -= tangent[0] * p->second;

  for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    tplaney[0][p->first] -= tangent[1] * p->second;
  for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    tplaney[1][p->first] -= tangent[1] * p->second;
  for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    tplaney[2][p->first] -= tangent[1] * p->second;

  for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    tplanez[0][p->first] -= tangent[2] * p->second;
  for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    tplanez[1][p->first] -= tangent[2] * p->second;
  for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    tplanez[2][p->first] -= tangent[2] * p->second;



  for (CI p = dauxn[0].begin(); p != dauxn[0].end(); ++p)
    get_deriv_auxn()[0][p->first] += tanplane(0, 0) * p->second;
  for (CI p = dauxn[1].begin(); p != dauxn[1].end(); ++p)
    get_deriv_auxn()[0][p->first] += tanplane(0, 1) * p->second;
  for (CI p = dauxn[2].begin(); p != dauxn[2].end(); ++p)
    get_deriv_auxn()[0][p->first] += tanplane(0, 2) * p->second;

  for (CI p = dauxn[0].begin(); p != dauxn[0].end(); ++p)
    get_deriv_auxn()[1][p->first] += tanplane(1, 0) * p->second;
  for (CI p = dauxn[1].begin(); p != dauxn[1].end(); ++p)
    get_deriv_auxn()[1][p->first] += tanplane(1, 1) * p->second;
  for (CI p = dauxn[2].begin(); p != dauxn[2].end(); ++p)
    get_deriv_auxn()[1][p->first] += tanplane(1, 2) * p->second;

  for (CI p = dauxn[0].begin(); p != dauxn[0].end(); ++p)
    get_deriv_auxn()[2][p->first] += tanplane(2, 0) * p->second;
  for (CI p = dauxn[1].begin(); p != dauxn[1].end(); ++p)
    get_deriv_auxn()[2][p->first] += tanplane(2, 1) * p->second;
  for (CI p = dauxn[2].begin(); p != dauxn[2].end(); ++p)
    get_deriv_auxn()[2][p->first] += tanplane(2, 2) * p->second;

  //-----------------------------
  for (CI p = tplanex[0].begin(); p != tplanex[0].end(); ++p)
    get_deriv_auxn()[0][p->first] += auxn()[0] * p->second;
  for (CI p = tplanex[1].begin(); p != tplanex[1].end(); ++p)
    get_deriv_auxn()[0][p->first] += auxn()[1] * p->second;
  for (CI p = tplanex[2].begin(); p != tplanex[2].end(); ++p)
    get_deriv_auxn()[0][p->first] += auxn()[2] * p->second;

  for (CI p = tplaney[0].begin(); p != tplaney[0].end(); ++p)
    get_deriv_auxn()[1][p->first] += auxn()[0] * p->second;
  for (CI p = tplaney[1].begin(); p != tplaney[1].end(); ++p)
    get_deriv_auxn()[1][p->first] += auxn()[1] * p->second;
  for (CI p = tplaney[2].begin(); p != tplaney[2].end(); ++p)
    get_deriv_auxn()[1][p->first] += auxn()[2] * p->second;

  for (CI p = tplanez[0].begin(); p != tplanez[0].end(); ++p)
    get_deriv_auxn()[2][p->first] += auxn()[0] * p->second;
  for (CI p = tplanez[1].begin(); p != tplanez[1].end(); ++p)
    get_deriv_auxn()[2][p->first] += auxn()[1] * p->second;
  for (CI p = tplanez[2].begin(); p != tplanez[2].end(); ++p)
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
bool CONTACT::LineToSurfaceCoupling3d::project_source()
{
  // project source nodes onto auxiliary plane
  int nnodes = line_element()->num_node();

  // initialize storage for source coords + their ids
  std::vector<double> vertices(3);
  std::vector<int> source_node_ids(1);

  for (int i = 0; i < nnodes; ++i)
  {
    Core::Nodes::Node* node = idiscret_.g_node(line_element()->node_ids()[i]);
    if (!node) FOUR_C_THROW("Cannot find source element with gid %", line_element()->node_ids()[i]);
    Node* mycnode = dynamic_cast<Node*>(node);
    if (!mycnode) FOUR_C_THROW("project_source: Null pointer!");

    // first build difference of point and element center
    // and then dot product with unit normal at center
    const double dist = (mycnode->xspatial()[0] - auxc()[0]) * auxn()[0] +
                        (mycnode->xspatial()[1] - auxc()[1]) * auxn()[1] +
                        (mycnode->xspatial()[2] - auxc()[2]) * auxn()[2];

    // compute projection
    for (int k = 0; k < 3; ++k) vertices[k] = mycnode->xspatial()[k] - dist * auxn()[k];

    // get node id, too
    source_node_ids[0] = mycnode->id();

    // store into vertex data structure
    source_vertices().emplace_back(vertices, Mortar::Vertex::projsource, source_node_ids, nullptr);
  }
  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of source vertex (3D) AuxPlane              farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToSurfaceCoupling3d::source_vertex_linearization(
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& currlin)
{
  // we first need the source element center:
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
    FOUR_C_THROW("target_vertex_linearization called for unknown element type");

  // evlauate shape functions + derivatives at scxi
  int nrow = surface_element().num_node();
  Core::LinAlg::SerialDenseVector source_val(nrow);
  Core::LinAlg::SerialDenseMatrix source_deriv(nrow, 2, true);
  surface_element().evaluate_shape(scxi, source_val, source_deriv, nrow);

  // we need all participating source nodes
  Core::Nodes::Node** source_nodes = surface_element().nodes();
  std::vector<Mortar::Node*> source_mortar_nodes(nrow);

  for (int i = 0; i < nrow; ++i)
  {
    source_mortar_nodes[i] = dynamic_cast<Mortar::Node*>(source_nodes[i]);
    if (!source_mortar_nodes[i]) FOUR_C_THROW("target_vertex_linearization: Null pointer!");
  }

  // linearization of the SourceIntEle spatial coords
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> source_node_lin;

  // resize the linearizations
  source_node_lin.resize(nrow, std::vector<Core::Gen::Pairedvector<int, double>>(3, 1));

  // loop over all intEle nodes
  for (int in = 0; in < nrow; ++in)
    for (int dim = 0; dim < 3; ++dim)
      source_node_lin[in][dim][source_mortar_nodes[in]->dofs()[dim]] += 1.;

  // map iterator
  using CI = Core::Gen::Pairedvector<int,
      double>::const_iterator;  // linearization of element center Auxc()
  //  std::vector<Core::Gen::Pairedvector<int  ,double> >
  //  linauxc(3,10*surface_element().num_node());
  //  // assume 3 dofs per node

  //  for (int i = 0; i < nrow; ++i)
  //      for (int dim=0; dim<3; ++dim)
  //        for (CI p=source_node_lin[i][dim].begin(); p!=source_node_lin[i][dim].end(); ++p)
  //          linauxc[dim][p->first] = source_val[i]*p->second;

  // linearization of element normal Auxn()
  std::vector<Core::Gen::Pairedvector<int, double>>& linauxn = get_deriv_auxn();

  // linearization of the TargetIntEle spatial coords
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> target_node_lin;

  // resize the linearizations
  target_node_lin.resize(
      line_element()->num_node(), std::vector<Core::Gen::Pairedvector<int, double>>(3, 1));

  // loop over all intEle nodes
  for (int in = 0; in < line_element()->num_node(); ++in)
  {
    Mortar::Node* mortar_target_node =
        dynamic_cast<Mortar::Node*>(idiscret_.g_node(line_element()->node_ids()[in]));
    if (mortar_target_node == nullptr) FOUR_C_THROW("dynamic cast to mortar node went wrong");

    for (int dim = 0; dim < 3; ++dim)
      target_node_lin[in][dim][mortar_target_node->dofs()[dim]] += 1.;
  }

  // put everything together for source vertex linearization
  // loop over all vertices
  for (int i = 0; i < line_element()->num_node(); ++i)
  {
    Mortar::Node* mortar_target_node =
        dynamic_cast<Mortar::Node*>(idiscret_.g_node(line_element()->node_ids()[i]));
    if (!mortar_target_node) FOUR_C_THROW("cast to mortar node failed");

    // (1) source node coordinates part
    for (CI p = target_node_lin[i][0].begin(); p != target_node_lin[i][0].end(); ++p)
    {
      currlin[i][0][p->first] += (1.0 - auxn()[0] * auxn()[0]) * p->second;
      currlin[i][1][p->first] -= (auxn()[0] * auxn()[1]) * p->second;
      currlin[i][2][p->first] -= (auxn()[0] * auxn()[2]) * p->second;
    }
    for (CI p = target_node_lin[i][1].begin(); p != target_node_lin[i][1].end(); ++p)
    {
      currlin[i][0][p->first] -= (auxn()[0] * auxn()[1]) * p->second;
      currlin[i][1][p->first] += (1.0 - auxn()[1] * auxn()[1]) * p->second;
      currlin[i][2][p->first] -= (auxn()[1] * auxn()[2]) * p->second;
    }
    for (CI p = target_node_lin[i][2].begin(); p != target_node_lin[i][2].end(); ++p)
    {
      currlin[i][0][p->first] -= (auxn()[2] * auxn()[0]) * p->second;
      currlin[i][1][p->first] -= (auxn()[2] * auxn()[1]) * p->second;
      currlin[i][2][p->first] += (1.0 - auxn()[2] * auxn()[2]) * p->second;
    }

    // (2) source element center coordinates (Auxc()) part
    for (CI p = get_deriv_auxc()[0].begin(); p != get_deriv_auxc()[0].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[0] * auxn()[k] * (p->second);

    for (CI p = get_deriv_auxc()[1].begin(); p != get_deriv_auxc()[1].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[1] * auxn()[k] * (p->second);

    for (CI p = get_deriv_auxc()[2].begin(); p != get_deriv_auxc()[2].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[2] * auxn()[k] * (p->second);

    // (3) source element normal (Auxn()) part
    double xdotn = (mortar_target_node->xspatial()[0] - auxc()[0]) * auxn()[0] +
                   (mortar_target_node->xspatial()[1] - auxc()[1]) * auxn()[1] +
                   (mortar_target_node->xspatial()[2] - auxc()[2]) * auxn()[2];

    for (CI p = linauxn[0].begin(); p != linauxn[0].end(); ++p)
    {
      currlin[i][0][p->first] -= xdotn * (p->second);
      for (int k = 0; k < 3; ++k)
        currlin[i][k][p->first] -=
            (mortar_target_node->xspatial()[0] - auxc()[0]) * auxn()[k] * (p->second);
    }

    for (CI p = linauxn[1].begin(); p != linauxn[1].end(); ++p)
    {
      currlin[i][1][p->first] -= xdotn * (p->second);
      for (int k = 0; k < 3; ++k)
        currlin[i][k][p->first] -=
            (mortar_target_node->xspatial()[1] - auxc()[1]) * auxn()[k] * (p->second);
    }

    for (CI p = linauxn[2].begin(); p != linauxn[2].end(); ++p)
    {
      currlin[i][2][p->first] -= xdotn * (p->second);
      for (int k = 0; k < 3; ++k)
        currlin[i][k][p->first] -=
            (mortar_target_node->xspatial()[2] - auxc()[2]) * auxn()[k] * (p->second);
    }
  }
}


/*----------------------------------------------------------------------*
 |  eval (public)                                            farah 07/16|
 *----------------------------------------------------------------------*/
bool CONTACT::LineToSurfaceCoupling3d::project_target()
{
  // project target nodes onto auxiliary plane
  int nnodes = surface_element().num_node();
  Core::Nodes::Node** mynodes = surface_element().nodes();
  if (!mynodes) FOUR_C_THROW("project_target: Null pointer!");

  // initialize storage for target coords + their ids
  std::vector<double> vertices(3);
  std::vector<int> target_node_ids(1);

  for (int i = 0; i < nnodes; ++i)
  {
    Node* mycnode = dynamic_cast<Node*>(mynodes[i]);
    if (!mycnode) FOUR_C_THROW("project_target: Null pointer!");

    // first build difference of point and element center
    // and then dot product with unit normal at center
    const double dist = (mycnode->xspatial()[0] - auxc()[0]) * auxn()[0] +
                        (mycnode->xspatial()[1] - auxc()[1]) * auxn()[1] +
                        (mycnode->xspatial()[2] - auxc()[2]) * auxn()[2];

    // compute projection
    for (int k = 0; k < 3; ++k) vertices[k] = mycnode->xspatial()[k] - dist * auxn()[k];

    // get node id, too
    target_node_ids[0] = mycnode->id();

    // store into vertex data structure
    target_vertices().emplace_back(vertices, Mortar::Vertex::target, target_node_ids, nullptr);

    // std::cout << "->RealNode(M) " << mycnode->Id() << ": " << mycnode->xspatial()[0] << " " <<
    // mycnode->xspatial()[1] << " " << mycnode->xspatial()[2] << std::endl; std::cout <<
    // "->ProjNode(M) " << mycnode->Id() << ": " << vertices[0] << " " << vertices[1] << " " <<
    // vertices[2] << std::endl;
  }
  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of source vertex (3D) AuxPlane               farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToSurfaceCoupling3d::target_vertex_linearization(
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& currlin)
{
  // we first need the source element center:
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
    FOUR_C_THROW("source_vertex_linearization called for unknown element type");

  // evlauate shape functions + derivatives at scxi
  const int nrow = surface_element().num_node();
  Core::LinAlg::SerialDenseVector source_val(nrow);
  Core::LinAlg::SerialDenseMatrix source_deriv(nrow, 2, true);
  surface_element().evaluate_shape(scxi, source_val, source_deriv, nrow);

  // we need all participating source nodes
  Core::Nodes::Node** source_nodes = surface_element().nodes();
  std::vector<Mortar::Node*> source_mortar_nodes(nrow);

  for (int i = 0; i < nrow; ++i)
  {
    source_mortar_nodes[i] = dynamic_cast<Mortar::Node*>(source_nodes[i]);
    if (!source_mortar_nodes[i]) FOUR_C_THROW("source_vertex_linearization: Null pointer!");
  }

  // linearization of the IntEle spatial coords
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> nodelin;

  // resize the linearizations
  nodelin.resize(nrow, std::vector<Core::Gen::Pairedvector<int, double>>(3, 1));

  // loop over all intEle nodes
  for (int in = 0; in < nrow; ++in)
    for (int dim = 0; dim < 3; ++dim) nodelin[in][dim][source_mortar_nodes[in]->dofs()[dim]] += 1.;

  // map iterator
  using CI = Core::Gen::Pairedvector<int,
      double>::const_iterator;  // linearization of element center Auxc()
  //  std  ::vector<Core::Gen::Pairedvector<int  ,double> > linauxc(3,surface_element().num_node());
  //  // assume 3 dofs per node
  //
  //  for (int i = 0; i < nrow; ++i)
  //      for (int dim=0; dim<3; ++dim)
  //        for (CI p=nodelin[i][dim].begin(); p!=nodelin[i][dim].end(); ++p)
  //          linauxc[dim][p->first] = source_val[i]*p->second;

  // linearization of element normal Auxn()
  std::vector<Core::Gen::Pairedvector<int, double>>& linauxn = get_deriv_auxn();

  // put everything together for source vertex linearization
  // loop over all vertices
  for (int i = 0; i < surface_element().num_node(); ++i)
  {
    Mortar::Node* mortar_source_node = dynamic_cast<Mortar::Node*>(surface_element().nodes()[i]);
    if (!mortar_source_node) FOUR_C_THROW("cast to mortar node failed");

    // (1) source node coordinates part
    for (CI p = nodelin[i][0].begin(); p != nodelin[i][0].end(); ++p)
    {
      currlin[i][0][p->first] += (1.0 - auxn()[0] * auxn()[0]) * p->second;
      currlin[i][1][p->first] -= (auxn()[0] * auxn()[1]) * p->second;
      currlin[i][2][p->first] -= (auxn()[0] * auxn()[2]) * p->second;
    }
    for (CI p = nodelin[i][1].begin(); p != nodelin[i][1].end(); ++p)
    {
      currlin[i][0][p->first] -= (auxn()[0] * auxn()[1]) * p->second;
      currlin[i][1][p->first] += (1.0 - auxn()[1] * auxn()[1]) * p->second;
      currlin[i][2][p->first] -= (auxn()[1] * auxn()[2]) * p->second;
    }
    for (CI p = nodelin[i][2].begin(); p != nodelin[i][2].end(); ++p)
    {
      currlin[i][0][p->first] -= (auxn()[2] * auxn()[0]) * p->second;
      currlin[i][1][p->first] -= (auxn()[2] * auxn()[1]) * p->second;
      currlin[i][2][p->first] += (1.0 - auxn()[2] * auxn()[2]) * p->second;
    }

    // (2) source element center coordinates (Auxc()) part
    for (CI p = get_deriv_auxc()[0].begin(); p != get_deriv_auxc()[0].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[0] * auxn()[k] * (p->second);

    for (CI p = get_deriv_auxc()[1].begin(); p != get_deriv_auxc()[1].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[1] * auxn()[k] * (p->second);

    for (CI p = get_deriv_auxc()[2].begin(); p != get_deriv_auxc()[2].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[2] * auxn()[k] * (p->second);

    // (3) source element normal (Auxn()) part
    double xdotn = (mortar_source_node->xspatial()[0] - auxc()[0]) * auxn()[0] +
                   (mortar_source_node->xspatial()[1] - auxc()[1]) * auxn()[1] +
                   (mortar_source_node->xspatial()[2] - auxc()[2]) * auxn()[2];

    for (CI p = linauxn[0].begin(); p != linauxn[0].end(); ++p)
    {
      currlin[i][0][p->first] -= xdotn * (p->second);
      for (int k = 0; k < 3; ++k)
        currlin[i][k][p->first] -=
            (mortar_source_node->xspatial()[0] - auxc()[0]) * auxn()[k] * (p->second);
    }

    for (CI p = linauxn[1].begin(); p != linauxn[1].end(); ++p)
    {
      currlin[i][1][p->first] -= xdotn * (p->second);
      for (int k = 0; k < 3; ++k)
        currlin[i][k][p->first] -=
            (mortar_source_node->xspatial()[1] - auxc()[1]) * auxn()[k] * (p->second);
    }

    for (CI p = linauxn[2].begin(); p != linauxn[2].end(); ++p)
    {
      currlin[i][2][p->first] -= xdotn * (p->second);
      for (int k = 0; k < 3; ++k)
        currlin[i][k][p->first] -=
            (mortar_source_node->xspatial()[2] - auxc()[2]) * auxn()[k] * (p->second);
    }
  }
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
  double source_xi = 0.0;
  double target_xi = 0.0;

  // create empty lin vectors
  Core::Gen::Pairedvector<int, double> d_source_xi(
      100 + 3 * line_target_element()->num_node() + 3 * line_source_element()->num_node());
  Core::Gen::Pairedvector<int, double> d_target_xi(
      100 + 3 * line_target_element()->num_node() + 3 * line_source_element()->num_node());
  line_intersection(&source_xi, &target_xi, d_source_xi, d_target_xi);

  // 3. check solution
  bool valid = check_intersection(&source_xi, &target_xi);
  if (!valid) return;

  // 4. check if intersection was already done!
  for (int i = 0; i < line_source_element()->num_node(); ++i)
  {
    if (dynamic_cast<Node*>(line_source_element()->nodes()[i])->mo_data().get_dltl().size() > 0)
      return;
  }

  // 5. evaluate terms
  evaluate_terms(&source_xi, &target_xi, d_source_xi, d_target_xi);
}


/*----------------------------------------------------------------------*
 |  line-line intersection                                   farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToLineCouplingPoint3d::evaluate_terms(double* source_xi, double* target_xi,
    Core::Gen::Pairedvector<int, double>& d_source_xi,
    Core::Gen::Pairedvector<int, double>& d_target_xi)
{
  bool friction = false;
  auto ftype = Teuchos::getIntegralValue<CONTACT::FrictionType>(interface_params(), "FRICTION");
  if (ftype != CONTACT::FrictionType::none) friction = true;

  // get source element nodes themselves for normal evaluation
  Core::Nodes::Node** mynodes = line_source_element()->nodes();
  if (!mynodes) FOUR_C_THROW("integrate_deriv_cell_3d_aux_plane_lts: Null pointer!");
  Core::Nodes::Node** target_nodes = line_target_element()->nodes();
  if (!target_nodes) FOUR_C_THROW("integrate_deriv_cell_3d_aux_plane_lts: Null pointer!");

  int nnodes = 2;
  int ndof = 3;
  int nrow = line_source_element()->num_node();
  int ncol = line_target_element()->num_node();

  // source values
  Core::LinAlg::SerialDenseVector source_val(nnodes);
  Core::LinAlg::SerialDenseMatrix source_deriv(nnodes, 1);
  line_source_element()->evaluate_shape(source_xi, source_val, source_deriv, nnodes);

  // target values
  Core::LinAlg::SerialDenseVector target_val(nnodes);
  Core::LinAlg::SerialDenseMatrix target_deriv(nnodes, 1);
  line_target_element()->evaluate_shape(target_xi, target_val, target_deriv, nnodes);

  // map iterator
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

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
  std::array<double, 3> source_gp_x = {0.0, 0.0, 0.0};
  std::array<double, 3> target_gp_x = {0.0, 0.0, 0.0};

  for (int i = 0; i < nrow; ++i)
  {
    Mortar::Node* mymrtrnode = dynamic_cast<Mortar::Node*>(mynodes[i]);
    gpn[0] += source_val[i] * mymrtrnode->mo_data().n()[0];
    gpn[1] += source_val[i] * mymrtrnode->mo_data().n()[1];
    gpn[2] += source_val[i] * mymrtrnode->mo_data().n()[2];

    source_gp_x[0] += source_val[i] * line_source_element()->get_nodal_coords(0, i);
    source_gp_x[1] += source_val[i] * line_source_element()->get_nodal_coords(1, i);
    source_gp_x[2] += source_val[i] * line_source_element()->get_nodal_coords(2, i);
  }

  // build interpolation of target GP coordinates
  for (int i = 0; i < ncol; ++i)
  {
    target_gp_x[0] += target_val[i] * line_target_element()->get_nodal_coords(0, i);
    target_gp_x[1] += target_val[i] * line_target_element()->get_nodal_coords(1, i);
    target_gp_x[2] += target_val[i] * line_target_element()->get_nodal_coords(2, i);
  }

  // normalize interpolated GP normal back to length 1.0 !!!
  double lengthn = sqrt(gpn[0] * gpn[0] + gpn[1] * gpn[1] + gpn[2] * gpn[2]);
  if (lengthn < 1.0e-12) FOUR_C_THROW("IntegrateAndDerivSegment: Divide by zero!");

  for (int i = 0; i < 3; ++i) gpn[i] /= lengthn;

  // build gap function at current GP
  for (int i = 0; i < n_dim(); ++i) gap += (target_gp_x[i] - source_gp_x[i]) * gpn[i];

  // build directional derivative of source GP normal (non-unit)
  Core::Gen::Pairedvector<int, double> dmap_nxs_gp(linsize);
  Core::Gen::Pairedvector<int, double> dmap_nys_gp(linsize);
  Core::Gen::Pairedvector<int, double> dmap_nzs_gp(linsize);

  for (int i = 0; i < nrow; ++i)
  {
    Node* cnode = dynamic_cast<Node*>(mynodes[i]);

    Core::Gen::Pairedvector<int, double>& dmap_nxsl_i = cnode->data().get_deriv_n()[0];
    Core::Gen::Pairedvector<int, double>& dmap_nysl_i = cnode->data().get_deriv_n()[1];
    Core::Gen::Pairedvector<int, double>& dmap_nzsl_i = cnode->data().get_deriv_n()[2];

    for (CI p = dmap_nxsl_i.begin(); p != dmap_nxsl_i.end(); ++p)
      dmap_nxs_gp[p->first] += source_val[i] * (p->second);
    for (CI p = dmap_nysl_i.begin(); p != dmap_nysl_i.end(); ++p)
      dmap_nys_gp[p->first] += source_val[i] * (p->second);
    for (CI p = dmap_nzsl_i.begin(); p != dmap_nzsl_i.end(); ++p)
      dmap_nzs_gp[p->first] += source_val[i] * (p->second);

    for (CI p = d_source_xi.begin(); p != d_source_xi.end(); ++p)
    {
      double valx = source_deriv(i, 0) * cnode->mo_data().n()[0];
      dmap_nxs_gp[p->first] += valx * (p->second);
      double valy = source_deriv(i, 0) * cnode->mo_data().n()[1];
      dmap_nys_gp[p->first] += valy * (p->second);
      double valz = source_deriv(i, 0) * cnode->mo_data().n()[2];
      dmap_nzs_gp[p->first] += valz * (p->second);
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

  for (CI p = dmap_nxs_gp.begin(); p != dmap_nxs_gp.end(); ++p)
  {
    dnmap_unit[0][p->first] += linv * (p->second);
    dnmap_unit[0][p->first] -= lllinv * sxsx * (p->second);
    dnmap_unit[1][p->first] -= lllinv * sxsy * (p->second);
    dnmap_unit[2][p->first] -= lllinv * sxsz * (p->second);
  }

  for (CI p = dmap_nys_gp.begin(); p != dmap_nys_gp.end(); ++p)
  {
    dnmap_unit[1][p->first] += linv * (p->second);
    dnmap_unit[1][p->first] -= lllinv * sysy * (p->second);
    dnmap_unit[0][p->first] -= lllinv * sxsy * (p->second);
    dnmap_unit[2][p->first] -= lllinv * sysz * (p->second);
  }

  for (CI p = dmap_nzs_gp.begin(); p != dmap_nzs_gp.end(); ++p)
  {
    dnmap_unit[2][p->first] += linv * (p->second);
    dnmap_unit[2][p->first] -= lllinv * szsz * (p->second);
    dnmap_unit[0][p->first] -= lllinv * sxsz * (p->second);
    dnmap_unit[1][p->first] -= lllinv * sysz * (p->second);
  }

  // add everything to dgapgp
  for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    dgapgp[p->first] += (target_gp_x[0] - source_gp_x[0]) * (p->second);

  for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    dgapgp[p->first] += (target_gp_x[1] - source_gp_x[1]) * (p->second);

  for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    dgapgp[p->first] += (target_gp_x[2] - source_gp_x[2]) * (p->second);

  // lin source nodes
  for (int z = 0; z < nrow; ++z)
  {
    Node* cnode = dynamic_cast<Node*>(mynodes[z]);
    for (int k = 0; k < 3; ++k) dgapgp[cnode->dofs()[k]] -= source_val[z] * gpn[k];
  }

  for (CI p = d_source_xi.begin(); p != d_source_xi.end(); ++p)
  {
    double& dg = dgapgp[p->first];
    const double& ps = p->second;
    for (int z = 0; z < nrow; ++z)
    {
      Node* cnode = dynamic_cast<Node*>(mynodes[z]);
      for (int k = 0; k < 3; ++k) dg -= gpn[k] * source_deriv(z, 0) * cnode->xspatial()[k] * ps;
    }
  }

  //        TARGET
  // lin target nodes
  for (int z = 0; z < ncol; ++z)
  {
    Node* cnode = dynamic_cast<Node*>(target_nodes[z]);
    for (int k = 0; k < 3; ++k) dgapgp[cnode->dofs()[k]] += target_val[z] * gpn[k];
  }

  for (CI p = d_target_xi.begin(); p != d_target_xi.end(); ++p)
  {
    double& dg = dgapgp[p->first];
    const double& ps = p->second;
    for (int z = 0; z < ncol; ++z)
    {
      Node* cnode = dynamic_cast<Node*>(target_nodes[z]);
      for (int k = 0; k < 3; ++k) dg += gpn[k] * target_deriv(z, 0) * cnode->xspatial()[k] * ps;
    }
  }

  // gap
  CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(mynodes[0]);

  // do not process source side boundary nodes
  // (their row entries would be zero anyway!)
  if (cnode->is_on_bound()) return;

  if (gap >= 0.0) return;

  double value[3] = {0.0, 0.0, 0.0};
  value[0] = (target_gp_x[0] - source_gp_x[0]);  // gap*gpn[0];
  value[1] = (target_gp_x[1] - source_gp_x[1]);  // gap*gpn[1];
  value[2] = (target_gp_x[2] - source_gp_x[2]);  // gap*gpn[2];

  // add current Gauss point's contribution to gseg
  cnode->addltl_gap_value(value);

  double lengthv = 0.0;
  lengthv = sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
  if (lengthv < 1e-12) FOUR_C_THROW("zero length!");
  value[0] /= lengthv;
  value[1] /= lengthv;
  value[2] /= lengthv;

  std::vector<std::map<int, double>>& dgmap = cnode->data().get_deriv_gltl();

  for (CI p = dgapgp.begin(); p != dgapgp.end(); ++p)
  {
    dgmap[0][p->first] += gpn[0] * (p->second);
    dgmap[1][p->first] += gpn[1] * (p->second);
    dgmap[2][p->first] += gpn[2] * (p->second);
  }

  for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
    dgmap[0][p->first] += gap * (p->second);
  for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
    dgmap[1][p->first] += gap * (p->second);
  for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
    dgmap[2][p->first] += gap * (p->second);

  //*****************************************
  // integrate D and M matrix
  // integrate dseg
  for (int k = 0; k < nrow; ++k)
  {
    CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(mynodes[k]);

    // multiply the two shape functions
    double prod = source_val[k];  // this reduces to source_val[k]

    if (abs(prod) > MORTARINTTOL) cnode->add_dltl_value(target_node->id(), prod);
    //    if(abs(prod)>MORTARINTTOL) cnode->AddSNode(target_node->Id()); // only for friction!
  }

  // integrate mseg
  for (int k = 0; k < ncol; ++k)
  {
    CONTACT::Node* target_node = dynamic_cast<CONTACT::Node*>(target_nodes[k]);

    // multiply the two shape functions
    double prod = target_val[k];  // this reduces to target_val[k]

    if (abs(prod) > MORTARINTTOL) cnode->add_mltl_value(target_node->id(), prod);
    //    if(abs(prod)>MORTARINTTOL) cnode->AddMNode(target_node->Id());  // only for friction!
  }

  // integrate LinD
  for (int k = 0; k < nrow; ++k)
  {
    // global target node ID
    int t_gid = line_source_element()->nodes()[k]->id();
    static double fac = 0.0;

    // get the correct map as a reference
    std::map<int, double>& ddmap_jk = cnode->data().get_deriv_dltl()[t_gid];

    // (3) Lin(NTarget) - target GP coordinates
    fac = source_deriv(k, 0);
    for (CI p = d_source_xi.begin(); p != d_source_xi.end(); ++p)
    {
      ddmap_jk[p->first] += fac * (p->second);
    }
  }  // loop over source nodes

  // integrate LinM
  for (int k = 0; k < ncol; ++k)
  {
    // global target node ID
    int t_gid = line_target_element()->nodes()[k]->id();
    static double fac = 0.0;

    // get the correct map as a reference
    std::map<int, double>& dmmap_jk = cnode->data().get_deriv_mltl()[t_gid];

    // (1) Lin(Phi) - dual shape functions
    // this vanishes here since there are no deformation-dependent dual functions

    // (3) Lin(NTarget) - target GP coordinates
    fac = target_deriv(k, 0);
    for (CI p = d_target_xi.begin(); p != d_target_xi.end(); ++p)
    {
      dmmap_jk[p->first] += fac * (p->second);
    }
  }  // loop over target nodes


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

    // loop over all source nodes
    for (int i = 0; i < idiscret_.node_col_map()->num_my_elements(); ++i)
    {
      int gid1 = idiscret_.node_col_map()->gid(i);
      Core::Nodes::Node* node1 = idiscret_.g_node(gid1);
      if (!node1) FOUR_C_THROW("Cannot find node with gid %", gid1);
      Node* contactnode = dynamic_cast<Node*>(node1);

      // here only source nodes
      if (!contactnode->is_source()) continue;

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
        for (CI p = dynamic_cast<FriNode*>(contactnode)->fri_data().get_d_old_ltl().begin();
            p != dynamic_cast<FriNode*>(contactnode)->fri_data().get_d_old_ltl().end(); ++p)
        {
          // node id
          int gid3 = p->first;
          Core::Nodes::Node* source_node = idiscret_.g_node(gid3);
          if (!source_node) FOUR_C_THROW("Cannot find node with gid");
          Node* c_source_node = dynamic_cast<Node*>(source_node);

          for (int d = 0; d < n_dim(); ++d)
          {
            sgpxold[d] += p->second * c_source_node->xspatial()[d];
          }
          sgpxoldlinx[c_source_node->dofs()[0]] += p->second;
          sgpxoldliny[c_source_node->dofs()[1]] += p->second;
          sgpxoldlinz[c_source_node->dofs()[2]] += p->second;
        }

        // safety
        if (dynamic_cast<FriNode*>(contactnode)->fri_data().get_m_old_ltl().size() < 1)
          FOUR_C_THROW("something went wrong!");

        for (auto p = dynamic_cast<FriNode*>(contactnode)->fri_data().get_m_old_ltl().begin();
            p != dynamic_cast<FriNode*>(contactnode)->fri_data().get_m_old_ltl().end(); ++p)
        {
          // node id
          int gid3 = p->first;
          Core::Nodes::Node* target_node = idiscret_.g_node(gid3);
          if (!target_node) FOUR_C_THROW("Cannot find node with gid");
          Node* c_target_node = dynamic_cast<Node*>(target_node);

          for (int d = 0; d < n_dim(); ++d)
          {
            mgpxold[d] += p->second * c_target_node->xspatial()[d];
          }
          mgpxoldlinx[c_target_node->dofs()[0]] += p->second;
          mgpxoldliny[c_target_node->dofs()[1]] += p->second;
          mgpxoldlinz[c_target_node->dofs()[2]] += p->second;
        }
      }
    }

    // create slip
    for (int d = 0; d < n_dim(); ++d)
      jump[d] = target_gp_x[d] - mgpxold[d] - (source_gp_x[d] - sgpxold[d]);

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

    // lin source nodes
    for (int z = 0; z < nrow; ++z)
    {
      Node* node = dynamic_cast<Node*>(mynodes[z]);
      for (int k = 0; k < 3; ++k) djmap[k][node->dofs()[k]] -= source_val[z];
    }


    for (int k = 0; k < nrow; ++k)
    {
      Node* node = dynamic_cast<Node*>(mynodes[k]);
      for (CI p = d_source_xi.begin(); p != d_source_xi.end(); ++p)
      {
        for (int z = 0; z < 3; ++z)
          djmap[z][p->first] -= source_deriv(k, 0) * (p->second) * node->xspatial()[z];
      }
    }  // loop over source nodes


    // lin target nodes
    for (int z = 0; z < ncol; ++z)
    {
      Node* node = dynamic_cast<Node*>(target_nodes[z]);
      for (int k = 0; k < 3; ++k) djmap[k][node->dofs()[k]] += target_val[z];
    }

    for (int k = 0; k < ncol; ++k)
    {
      Node* node = dynamic_cast<Node*>(target_nodes[k]);
      for (CI p = d_target_xi.begin(); p != d_target_xi.end(); ++p)
      {
        for (int z = 0; z < 3; ++z)
          djmap[z][p->first] += target_deriv(k, 0) * (p->second) * node->xspatial()[z];
      }
    }  // loop over source nodes

    // source_gp_x and target_gp_x old
    for (CI p = mgpxoldlinx.begin(); p != mgpxoldlinx.end(); ++p) djmap[0][p->first] -= p->second;
    for (CI p = mgpxoldliny.begin(); p != mgpxoldliny.end(); ++p) djmap[1][p->first] -= p->second;
    for (CI p = mgpxoldlinz.begin(); p != mgpxoldlinz.end(); ++p) djmap[2][p->first] -= p->second;

    for (CI p = sgpxoldlinx.begin(); p != sgpxoldlinx.end(); ++p) djmap[0][p->first] += p->second;
    for (CI p = sgpxoldliny.begin(); p != sgpxoldliny.end(); ++p) djmap[1][p->first] += p->second;
    for (CI p = sgpxoldlinz.begin(); p != sgpxoldlinz.end(); ++p) djmap[2][p->first] += p->second;



    std::vector<Core::Gen::Pairedvector<int, double>> tplanex(3, 100);
    std::vector<Core::Gen::Pairedvector<int, double>> tplaney(3, 100);
    std::vector<Core::Gen::Pairedvector<int, double>> tplanez(3, 100);

    for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      tplanex[0][p->first] -= gpn[0] * p->second;
    for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      tplanex[1][p->first] -= gpn[1] * p->second;
    for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      tplanex[2][p->first] -= gpn[2] * p->second;

    for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      tplaney[0][p->first] -= gpn[0] * p->second;
    for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      tplaney[1][p->first] -= gpn[1] * p->second;
    for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      tplaney[2][p->first] -= gpn[2] * p->second;

    for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
      tplanez[0][p->first] -= gpn[0] * p->second;
    for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
      tplanez[1][p->first] -= gpn[1] * p->second;
    for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
      tplanez[2][p->first] -= gpn[2] * p->second;

    //------------
    for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      tplanex[0][p->first] -= gpn[0] * p->second;
    for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      tplanex[1][p->first] -= gpn[0] * p->second;
    for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
      tplanex[2][p->first] -= gpn[0] * p->second;

    for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      tplaney[0][p->first] -= gpn[1] * p->second;
    for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      tplaney[1][p->first] -= gpn[1] * p->second;
    for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
      tplaney[2][p->first] -= gpn[1] * p->second;

    for (CI p = dnmap_unit[0].begin(); p != dnmap_unit[0].end(); ++p)
      tplanez[0][p->first] -= gpn[2] * p->second;
    for (CI p = dnmap_unit[1].begin(); p != dnmap_unit[1].end(); ++p)
      tplanez[1][p->first] -= gpn[2] * p->second;
    for (CI p = dnmap_unit[2].begin(); p != dnmap_unit[2].end(); ++p)
      tplanez[2][p->first] -= gpn[2] * p->second;

    //-----------------------------

    for (CI p = djmap[0].begin(); p != djmap[0].end(); ++p)
      djmapfinal[0][p->first] += tanplane(0, 0) * p->second;
    for (CI p = djmap[1].begin(); p != djmap[1].end(); ++p)
      djmapfinal[0][p->first] += tanplane(0, 1) * p->second;
    for (CI p = djmap[2].begin(); p != djmap[2].end(); ++p)
      djmapfinal[0][p->first] += tanplane(0, 2) * p->second;

    for (CI p = djmap[0].begin(); p != djmap[0].end(); ++p)
      djmapfinal[1][p->first] += tanplane(1, 0) * p->second;
    for (CI p = djmap[1].begin(); p != djmap[1].end(); ++p)
      djmapfinal[1][p->first] += tanplane(1, 1) * p->second;
    for (CI p = djmap[2].begin(); p != djmap[2].end(); ++p)
      djmapfinal[1][p->first] += tanplane(1, 2) * p->second;

    for (CI p = djmap[0].begin(); p != djmap[0].end(); ++p)
      djmapfinal[2][p->first] += tanplane(2, 0) * p->second;
    for (CI p = djmap[1].begin(); p != djmap[1].end(); ++p)
      djmapfinal[2][p->first] += tanplane(2, 1) * p->second;
    for (CI p = djmap[2].begin(); p != djmap[2].end(); ++p)
      djmapfinal[2][p->first] += tanplane(2, 2) * p->second;

    //-----------------------------
    for (CI p = tplanex[0].begin(); p != tplanex[0].end(); ++p)
      djmapfinal[0][p->first] += jump[0] * p->second;
    for (CI p = tplanex[1].begin(); p != tplanex[1].end(); ++p)
      djmapfinal[0][p->first] += jump[1] * p->second;
    for (CI p = tplanex[2].begin(); p != tplanex[2].end(); ++p)
      djmapfinal[0][p->first] += jump[2] * p->second;

    for (CI p = tplaney[0].begin(); p != tplaney[0].end(); ++p)
      djmapfinal[1][p->first] += jump[0] * p->second;
    for (CI p = tplaney[1].begin(); p != tplaney[1].end(); ++p)
      djmapfinal[1][p->first] += jump[1] * p->second;
    for (CI p = tplaney[2].begin(); p != tplaney[2].end(); ++p)
      djmapfinal[1][p->first] += jump[2] * p->second;

    for (CI p = tplanez[0].begin(); p != tplanez[0].end(); ++p)
      djmapfinal[2][p->first] += jump[0] * p->second;
    for (CI p = tplanez[1].begin(); p != tplanez[1].end(); ++p)
      djmapfinal[2][p->first] += jump[1] * p->second;
    for (CI p = tplanez[2].begin(); p != tplanez[2].end(); ++p)
      djmapfinal[2][p->first] += jump[2] * p->second;

  }  // end friction
}

/*----------------------------------------------------------------------*
 |  line-line intersection                                   farah 07/16|
 *----------------------------------------------------------------------*/
void CONTACT::LineToLineCouplingPoint3d::line_intersection(double* source_xi, double* target_xi,
    Core::Gen::Pairedvector<int, double>& d_source_xi,
    Core::Gen::Pairedvector<int, double>& d_target_xi)
{
  // flag for debug output
  const bool out = false;

  // only for line 2
  const int nnodes = 2;

  // prepare linearizations
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // calculate source vector
  Node* source_node_1 = dynamic_cast<Node*>(line_source_element()->nodes()[0]);
  Node* source_node_2 = dynamic_cast<Node*>(line_source_element()->nodes()[1]);
  source_node_1->build_averaged_edge_tangent();
  source_node_2->build_averaged_edge_tangent();

  // calculate target vector
  Node* target_node_1 = dynamic_cast<Node*>(line_target_element()->nodes()[0]);
  Node* target_node_2 = dynamic_cast<Node*>(line_target_element()->nodes()[1]);
  target_node_1->build_averaged_edge_tangent();
  target_node_2->build_averaged_edge_tangent();

  double lengths1 =
      sqrt(source_node_1->mo_data().edge_tangent()[0] * source_node_1->mo_data().edge_tangent()[0] +
           source_node_1->mo_data().edge_tangent()[1] * source_node_1->mo_data().edge_tangent()[1] +
           source_node_1->mo_data().edge_tangent()[2] * source_node_1->mo_data().edge_tangent()[2]);
  double lengths2 =
      sqrt(source_node_2->mo_data().edge_tangent()[0] * source_node_2->mo_data().edge_tangent()[0] +
           source_node_2->mo_data().edge_tangent()[1] * source_node_2->mo_data().edge_tangent()[1] +
           source_node_2->mo_data().edge_tangent()[2] * source_node_2->mo_data().edge_tangent()[2]);
  double lengthm1 =
      sqrt(target_node_1->mo_data().edge_tangent()[0] * target_node_1->mo_data().edge_tangent()[0] +
           target_node_1->mo_data().edge_tangent()[1] * target_node_1->mo_data().edge_tangent()[1] +
           target_node_1->mo_data().edge_tangent()[2] * target_node_1->mo_data().edge_tangent()[2]);
  double lengthm2 =
      sqrt(target_node_2->mo_data().edge_tangent()[0] * target_node_2->mo_data().edge_tangent()[0] +
           target_node_2->mo_data().edge_tangent()[1] * target_node_2->mo_data().edge_tangent()[1] +
           target_node_2->mo_data().edge_tangent()[2] * target_node_2->mo_data().edge_tangent()[2]);
  if (lengths1 < 1e-12 or lengths2 < 1e-12 or lengthm1 < 1e-12 or lengthm2 < 1e-12)
    FOUR_C_THROW("tangents zero length");

  // calc angle between tangents
  std::array<double, 3> source_tangent_1 = {0.0, 0.0, 0.0};
  std::array<double, 3> source_tangent_2 = {0.0, 0.0, 0.0};
  std::array<double, 3> target_tangent_1 = {0.0, 0.0, 0.0};
  std::array<double, 3> target_tangent_2 = {0.0, 0.0, 0.0};

  source_tangent_1[0] = source_node_1->mo_data().edge_tangent()[0];
  source_tangent_1[1] = source_node_1->mo_data().edge_tangent()[1];
  source_tangent_1[2] = source_node_1->mo_data().edge_tangent()[2];

  source_tangent_2[0] = source_node_2->mo_data().edge_tangent()[0];
  source_tangent_2[1] = source_node_2->mo_data().edge_tangent()[1];
  source_tangent_2[2] = source_node_2->mo_data().edge_tangent()[2];

  target_tangent_1[0] = target_node_1->mo_data().edge_tangent()[0];
  target_tangent_1[1] = target_node_1->mo_data().edge_tangent()[1];
  target_tangent_1[2] = target_node_1->mo_data().edge_tangent()[2];

  target_tangent_2[0] = target_node_2->mo_data().edge_tangent()[0];
  target_tangent_2[1] = target_node_2->mo_data().edge_tangent()[1];
  target_tangent_2[2] = target_node_2->mo_data().edge_tangent()[2];

  if (out)
  {
    std::cout << "source 1 = " << source_tangent_1[0] << "  " << source_tangent_1[1] << "  "
              << source_tangent_1[2] << std::endl;
    std::cout << "source 2 = " << source_tangent_2[0] << "  " << source_tangent_2[1] << "  "
              << source_tangent_2[2] << std::endl;
  }

  double test = source_tangent_1[0] * source_tangent_2[0] +
                source_tangent_1[1] * source_tangent_2[1] +
                source_tangent_1[2] * source_tangent_2[2];
  if (test < 1e-8)
  {
    source_node_2->mo_data().edge_tangent()[0] *= -1.0;
    source_node_2->mo_data().edge_tangent()[1] *= -1.0;
    source_node_2->mo_data().edge_tangent()[2] *= -1.0;

    source_tangent_2[0] *= -1.0;
    source_tangent_2[1] *= -1.0;
    source_tangent_2[2] *= -1.0;

    for (CI p = source_node_2->data().get_deriv_tangent()[0].begin();
        p != source_node_2->data().get_deriv_tangent()[0].end(); ++p)
      source_node_2->data().get_deriv_tangent()[0][p->first] *= -1.0;
    for (CI p = source_node_2->data().get_deriv_tangent()[1].begin();
        p != source_node_2->data().get_deriv_tangent()[1].end(); ++p)
      source_node_2->data().get_deriv_tangent()[1][p->first] *= -1.0;
    for (CI p = source_node_2->data().get_deriv_tangent()[2].begin();
        p != source_node_2->data().get_deriv_tangent()[2].end(); ++p)
      source_node_2->data().get_deriv_tangent()[2][p->first] *= -1.0;
  }
  if (out)
  {
    std::cout << "----------------" << std::endl;
    std::cout << "source 1 = " << source_tangent_1[0] << "  " << source_tangent_1[1] << "  "
              << source_tangent_1[2] << std::endl;
    std::cout << "source 2 = " << source_tangent_2[0] << "  " << source_tangent_2[1] << "  "
              << source_tangent_2[2] << std::endl;

    std::cout << "target 1 = " << target_tangent_1[0] << "  " << target_tangent_1[1] << "  "
              << target_tangent_1[2] << std::endl;
    std::cout << "target 2 = " << target_tangent_2[0] << "  " << target_tangent_2[1] << "  "
              << target_tangent_2[2] << std::endl;
  }

  test = target_tangent_1[0] * target_tangent_2[0] + target_tangent_1[1] * target_tangent_2[1] +
         target_tangent_1[2] * target_tangent_2[2];
  if (test < 1e-8)
  {
    target_node_2->mo_data().edge_tangent()[0] *= -1.0;
    target_node_2->mo_data().edge_tangent()[1] *= -1.0;
    target_node_2->mo_data().edge_tangent()[2] *= -1.0;

    target_tangent_2[0] *= -1.0;
    target_tangent_2[1] *= -1.0;
    target_tangent_2[2] *= -1.0;

    for (CI p = target_node_2->data().get_deriv_tangent()[0].begin();
        p != target_node_2->data().get_deriv_tangent()[0].end(); ++p)
      target_node_2->data().get_deriv_tangent()[0][p->first] *= -1.0;
    for (CI p = target_node_2->data().get_deriv_tangent()[1].begin();
        p != target_node_2->data().get_deriv_tangent()[1].end(); ++p)
      target_node_2->data().get_deriv_tangent()[1][p->first] *= -1.0;
    for (CI p = target_node_2->data().get_deriv_tangent()[2].begin();
        p != target_node_2->data().get_deriv_tangent()[2].end(); ++p)
      target_node_2->data().get_deriv_tangent()[2][p->first] *= -1.0;
  }
  if (out)
  {
    std::cout << "----------------" << std::endl;
    std::cout << "target 1 = " << target_tangent_1[0] << "  " << target_tangent_1[1] << "  "
              << target_tangent_1[2] << std::endl;
    std::cout << "target 2 = " << target_tangent_2[0] << "  " << target_tangent_2[1] << "  "
              << target_tangent_2[2] << std::endl;
  }

  // res norm
  double conv = 0.0;

  // start in the element center
  double xi_source = 0.0;  // xi_source
  double xi_target = 0.0;  // xi_target

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
    // source values
    Core::LinAlg::SerialDenseVector source_val(nnodes);
    Core::LinAlg::SerialDenseMatrix source_deriv(nnodes, 1);
    line_source_element()->evaluate_shape(&xi_source, source_val, source_deriv, nnodes);

    // target values
    Core::LinAlg::SerialDenseVector target_val(nnodes);
    Core::LinAlg::SerialDenseMatrix target_deriv(nnodes, 1);
    line_target_element()->evaluate_shape(&xi_target, target_val, target_deriv, nnodes);

    std::array<double, 3> x_source = {0.0, 0.0, 0.0};
    std::array<double, 3> x_target = {0.0, 0.0, 0.0};

    for (int i = 0; i < 3; ++i)
      x_source[i] += source_val[0] * source_node_1->xspatial()[i] +
                     source_val[1] * source_node_2->xspatial()[i];
    for (int i = 0; i < 3; ++i)
      x_target[i] += target_val[0] * target_node_1->xspatial()[i] +
                     target_val[1] * target_node_2->xspatial()[i];

    std::array<double, 3> xdiff = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) xdiff[i] = x_source[i] - x_target[i];

    // calculate tangents:
    std::array<double, 3> v_source = {0.0, 0.0, 0.0};
    std::array<double, 3> v_target = {0.0, 0.0, 0.0};

    for (int i = 0; i < 3; ++i)
      v_source[i] += source_val[0] * source_tangent_1[i] + source_val[1] * source_tangent_2[i];
    for (int i = 0; i < 3; ++i)
      v_target[i] += target_val[0] * target_tangent_1[i] + target_val[1] * target_tangent_2[i];

    f[0] = xdiff[0] * v_source[0] + xdiff[1] * v_source[1] + xdiff[2] * v_source[2];
    f[1] = xdiff[0] * v_target[0] + xdiff[1] * v_target[1] + xdiff[2] * v_target[2];

    // check for convergence
    conv = sqrt(f[0] * f[0] + f[1] * f[1]);
    if (conv <= MORTARCONVTOL) break;

    //**********************************************
    //   F GRADIENT CALCULATION                   //
    //**********************************************

    std::array<double, 3> x_source_deriv = {0.0, 0.0, 0.0};
    std::array<double, 3> x_target_deriv = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i)
      x_source_deriv[i] += source_deriv(0, 0) * source_node_1->xspatial()[i] +
                           source_deriv(1, 0) * source_node_2->xspatial()[i];
    for (int i = 0; i < 3; ++i)
      x_target_deriv[i] += target_deriv(0, 0) * target_node_1->xspatial()[i] +
                           target_deriv(1, 0) * target_node_2->xspatial()[i];

    std::array<double, 3> v_source_deriv = {0.0, 0.0, 0.0};
    std::array<double, 3> v_target_deriv = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i)
      v_source_deriv[i] +=
          source_deriv(0, 0) * source_tangent_1[i] + source_deriv(1, 0) * source_tangent_2[i];
    for (int i = 0; i < 3; ++i)
      v_target_deriv[i] +=
          target_deriv(0, 0) * target_tangent_1[i] + target_deriv(1, 0) * target_tangent_2[i];

    df(0, 0) = x_source_deriv[0] * v_source[0] + x_source_deriv[1] * v_source[1] +
               x_source_deriv[2] * v_source[2] + v_source_deriv[0] * xdiff[0] +
               v_source_deriv[1] * xdiff[1] + v_source_deriv[2] * xdiff[2];

    df(0, 1) = -x_target_deriv[0] * v_source[0] - x_target_deriv[1] * v_source[1] -
               x_target_deriv[2] * v_source[2];

    df(1, 0) = x_source_deriv[0] * v_target[0] + x_source_deriv[1] * v_target[1] +
               x_source_deriv[2] * v_target[2];

    df(1, 1) = -x_target_deriv[0] * v_target[0] - x_target_deriv[1] * v_target[1] -
               x_target_deriv[2] * v_target[2] + v_target_deriv[0] * xdiff[0] +
               v_target_deriv[1] * xdiff[1] + v_target_deriv[2] * xdiff[2];

    //**********************************************
    //   solve deta = - inv(dF) * F               //
    //**********************************************
    double jacdet = df.invert();
    if (abs(jacdet) < 1.0e-12)
    {
      source_xi[0] = 1e12;
      target_xi[0] = 1e12;
      return;
      FOUR_C_THROW("Singular Jacobian for projection");
    }

    // update eta and alpha
    xi_source += -df(0, 0) * f[0] - df(0, 1) * f[1];
    xi_target += -df(1, 0) * f[0] - df(1, 1) * f[1];
  }

  // Newton iteration unconverged
  if (conv > MORTARCONVTOL) FOUR_C_THROW("LTL intersection not converged!");

  //**********************************************
  //  Linearization                             //
  //**********************************************
  // source values
  Core::LinAlg::SerialDenseVector source_val(nnodes);
  Core::LinAlg::SerialDenseMatrix source_deriv(nnodes, 1);
  line_source_element()->evaluate_shape(&xi_source, source_val, source_deriv, nnodes);

  // target values
  Core::LinAlg::SerialDenseVector target_val(nnodes);
  Core::LinAlg::SerialDenseMatrix target_deriv(nnodes, 1);
  line_target_element()->evaluate_shape(&xi_target, target_val, target_deriv, nnodes);

  std::array<double, 3> x_source = {0.0, 0.0, 0.0};
  std::array<double, 3> x_target = {0.0, 0.0, 0.0};

  for (int i = 0; i < 3; ++i)
    x_source[i] +=
        source_val[0] * source_node_1->xspatial()[i] + source_val[1] * source_node_2->xspatial()[i];
  for (int i = 0; i < 3; ++i)
    x_target[i] +=
        target_val[0] * target_node_1->xspatial()[i] + target_val[1] * target_node_2->xspatial()[i];

  std::array<double, 3> xdiff = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i) xdiff[i] = x_source[i] - x_target[i];

  // calculate tangents:
  std::array<double, 3> v_source = {0.0, 0.0, 0.0};
  std::array<double, 3> v_target = {0.0, 0.0, 0.0};

  for (int i = 0; i < 3; ++i)
    v_source[i] += source_val[0] * source_tangent_1[i] + source_val[1] * source_tangent_2[i];
  for (int i = 0; i < 3; ++i)
    v_target[i] += target_val[0] * target_tangent_1[i] + target_val[1] * target_tangent_2[i];

  std::vector<Core::Gen::Pairedvector<int, double>> x_lin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> v_source_lin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> v_target_lin(3, 1000);

  // global position difference
  for (int i = 0; i < 3; ++i) (x_lin[i])[source_node_1->dofs()[i]] += source_val(0);
  for (int i = 0; i < 3; ++i) (x_lin[i])[source_node_2->dofs()[i]] += source_val(1);

  for (int i = 0; i < 3; ++i) (x_lin[i])[target_node_1->dofs()[i]] -= target_val(0);
  for (int i = 0; i < 3; ++i) (x_lin[i])[target_node_2->dofs()[i]] -= target_val(1);

  // TODO: this would be the correct linearization! however, the old one works better. no idea
  // why!?!?!? tangent vector source
  for (int i = 0; i < 3; ++i)
  {
    for (CI p = source_node_1->data().get_deriv_tangent()[i].begin();
        p != source_node_1->data().get_deriv_tangent()[i].end(); ++p)
      (v_source_lin[i])[p->first] += source_val[0] * p->second;

    for (CI p = source_node_2->data().get_deriv_tangent()[i].begin();
        p != source_node_2->data().get_deriv_tangent()[i].end(); ++p)
      (v_source_lin[i])[p->first] += source_val[1] * p->second;
  }

  // tangent vector target
  for (int i = 0; i < 3; ++i)
  {
    for (CI p = target_node_1->data().get_deriv_tangent()[i].begin();
        p != target_node_1->data().get_deriv_tangent()[i].end(); ++p)
      (v_target_lin[i])[p->first] += target_val[0] * p->second;

    for (CI p = target_node_2->data().get_deriv_tangent()[i].begin();
        p != target_node_2->data().get_deriv_tangent()[i].end(); ++p)
      (v_target_lin[i])[p->first] += target_val[1] * p->second;
  }

  // TODO: this is the old linearization:
  // tangent vector source
  //  for(int i=0; i<3;++i)
  //    (v_source_lin[i])[source_node_1->Dofs()[i]] += 1;
  //  for(int i=0; i<3;++i)
  //    (v_source_lin[i])[source_node_2->Dofs()[i]] -= 1;
  //
  //  // tangent vector target
  //  for(int i=0; i<3;++i)
  //    (v_target_lin[i])[target_node_1->Dofs()[i]] += 1;
  //  for(int i=0; i<3;++i)
  //    (v_target_lin[i])[target_node_2->Dofs()[i]] -= 1;

  Core::Gen::Pairedvector<int, double> f0(1000);
  Core::Gen::Pairedvector<int, double> f1(1000);

  // lin xdiff * tangent + xdiff * lin tangent
  for (CI p = x_lin[0].begin(); p != x_lin[0].end(); ++p) f0[p->first] += (p->second) * v_source[0];
  for (CI p = x_lin[1].begin(); p != x_lin[1].end(); ++p) f0[p->first] += (p->second) * v_source[1];
  for (CI p = x_lin[2].begin(); p != x_lin[2].end(); ++p) f0[p->first] += (p->second) * v_source[2];

  for (CI p = v_source_lin[0].begin(); p != v_source_lin[0].end(); ++p)
    f0[p->first] += (p->second) * xdiff[0];
  for (CI p = v_source_lin[1].begin(); p != v_source_lin[1].end(); ++p)
    f0[p->first] += (p->second) * xdiff[1];
  for (CI p = v_source_lin[2].begin(); p != v_source_lin[2].end(); ++p)
    f0[p->first] += (p->second) * xdiff[2];

  // lin xdiff * tangent + xdiff * lin tangent
  for (CI p = x_lin[0].begin(); p != x_lin[0].end(); ++p) f1[p->first] += (p->second) * v_target[0];
  for (CI p = x_lin[1].begin(); p != x_lin[1].end(); ++p) f1[p->first] += (p->second) * v_target[1];
  for (CI p = x_lin[2].begin(); p != x_lin[2].end(); ++p) f1[p->first] += (p->second) * v_target[2];

  for (CI p = v_target_lin[0].begin(); p != v_target_lin[0].end(); ++p)
    f1[p->first] += (p->second) * xdiff[0];
  for (CI p = v_target_lin[1].begin(); p != v_target_lin[1].end(); ++p)
    f1[p->first] += (p->second) * xdiff[1];
  for (CI p = v_target_lin[2].begin(); p != v_target_lin[2].end(); ++p)
    f1[p->first] += (p->second) * xdiff[2];

  // end
  for (CI p = f0.begin(); p != f0.end(); ++p) d_source_xi[p->first] -= (p->second) * df(0, 0);
  for (CI p = f1.begin(); p != f1.end(); ++p) d_source_xi[p->first] -= (p->second) * df(0, 1);

  for (CI p = f0.begin(); p != f0.end(); ++p) d_target_xi[p->first] -= (p->second) * df(1, 0);
  for (CI p = f1.begin(); p != f1.end(); ++p) d_target_xi[p->first] -= (p->second) * df(1, 1);

  source_xi[0] = xi_source;
  target_xi[0] = xi_target;
}

/*----------------------------------------------------------------------*
 |  check if intersection is in para space interval          farah 07/16|
 *----------------------------------------------------------------------*/
bool CONTACT::LineToLineCouplingPoint3d::check_intersection(double* source_xi, double* target_xi)
{
  return source_xi[0] >= -1.0 - 1e-12 and source_xi[0] <= 1.0 + 1e-12 and
         target_xi[0] >= -1.0 - 1e-12 and target_xi[0] <= 1.0 + 1e-12;
}

/*----------------------------------------------------------------------*
 |  check if line eles parallel                              farah 07/16|
 *----------------------------------------------------------------------*/
bool CONTACT::LineToLineCouplingPoint3d::check_parallelity()
{
  // tolerance for line clipping
  const double source_min_edge_size = line_source_element()->min_edge_size();
  const double target_min_edge_size = line_target_element()->min_edge_size();
  const double tol = MORTARCLIPTOL * std::min(source_min_edge_size, target_min_edge_size);

  std::array<double, 3> v_source = {0.0, 0.0, 0.0};
  std::array<double, 3> v_target = {0.0, 0.0, 0.0};

  // calculate source vector
  Node* source_node_1 = dynamic_cast<Node*>(line_source_element()->nodes()[0]);
  Node* source_node_2 = dynamic_cast<Node*>(line_source_element()->nodes()[1]);

  v_source[0] = source_node_1->xspatial()[0] - source_node_2->xspatial()[0];
  v_source[1] = source_node_1->xspatial()[1] - source_node_2->xspatial()[1];
  v_source[2] = source_node_1->xspatial()[2] - source_node_2->xspatial()[2];

  // calculate source vector
  Node* target_node_1 = dynamic_cast<Node*>(line_target_element()->nodes()[0]);
  Node* target_node_2 = dynamic_cast<Node*>(line_target_element()->nodes()[1]);

  v_target[0] = target_node_1->xspatial()[0] - target_node_2->xspatial()[0];
  v_target[1] = target_node_1->xspatial()[1] - target_node_2->xspatial()[1];
  v_target[2] = target_node_1->xspatial()[2] - target_node_2->xspatial()[2];

  // calculate lengths
  const double length_source =
      sqrt(v_source[0] * v_source[0] + v_source[1] * v_source[1] + v_source[2] * v_source[2]);
  const double length_target =
      sqrt(v_target[0] * v_target[0] + v_target[1] * v_target[1] + v_target[2] * v_target[2]);

  // calculate scalar product
  const double scaprod =
      v_source[0] * v_target[0] + v_source[1] * v_target[1] + v_source[2] * v_target[2];

  // proof if scalar product equals length product --> parallelity
  const double diff = abs(scaprod) - (length_source * length_target);

  return abs(diff) < tol;
}



/*----------------------------------------------------------------------*
 |  calc current angle (rad) between edges                   farah 09/16|
 *----------------------------------------------------------------------*/
double CONTACT::LineToLineCouplingPoint3d::calc_current_angle(
    Core::Gen::Pairedvector<int, double>& lineAngle)
{
  // define iterator for linerization
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // source edge vector and target vector edge
  std::array<double, 3> v_source = {0.0, 0.0, 0.0};
  std::array<double, 3> v_target = {0.0, 0.0, 0.0};

  // calculate source vector
  Node* source_node_1 = dynamic_cast<Node*>(line_source_element()->nodes()[0]);
  Node* source_node_2 = dynamic_cast<Node*>(line_source_element()->nodes()[1]);

  v_source[0] = source_node_1->xspatial()[0] - source_node_2->xspatial()[0];
  v_source[1] = source_node_1->xspatial()[1] - source_node_2->xspatial()[1];
  v_source[2] = source_node_1->xspatial()[2] - source_node_2->xspatial()[2];

  // calculate source vector
  Node* target_node_1 = dynamic_cast<Node*>(line_target_element()->nodes()[0]);
  Node* target_node_2 = dynamic_cast<Node*>(line_target_element()->nodes()[1]);

  v_target[0] = target_node_1->xspatial()[0] - target_node_2->xspatial()[0];
  v_target[1] = target_node_1->xspatial()[1] - target_node_2->xspatial()[1];
  v_target[2] = target_node_1->xspatial()[2] - target_node_2->xspatial()[2];

  // calculate lengths
  const double length_source =
      sqrt(v_source[0] * v_source[0] + v_source[1] * v_source[1] + v_source[2] * v_source[2]);
  const double length_target =
      sqrt(v_target[0] * v_target[0] + v_target[1] * v_target[1] + v_target[2] * v_target[2]);

  // safety
  if (length_source < 1e-12 or length_target < 1e-12) FOUR_C_THROW("line elements of zero length!");

  // calculate scalar product
  double scaprod =
      v_source[0] * v_target[0] + v_source[1] * v_target[1] + v_source[2] * v_target[2];
  double scaledScaprod = scaprod / (length_source * length_target);
  double angleRad = acos(scaledScaprod);

  // check if we used the right angle
  bool switchSign = false;
  if (angleRad > 0.5 * std::numbers::pi)  // if angle is > 90 degrees
  {
    switchSign = true;

    // change sign of target vector
    v_target[0] = -v_target[0];
    v_target[1] = -v_target[1];
    v_target[2] = -v_target[2];

    scaprod = v_source[0] * v_target[0] + v_source[1] * v_target[1] + v_source[2] * v_target[2];
    scaledScaprod = scaprod / (length_source * length_target);
    angleRad = acos(scaledScaprod);
  }

  //===============================================================
  // linearization

  // delta length_target
  std::vector<Core::Gen::Pairedvector<int, double>> DlT(3, 1000);
  Core::Gen::Pairedvector<int, double> DlengthT(1000);

  // change sign of target vectors linearization
  if (switchSign)
  {
    DlT[0][target_node_1->dofs()[0]] -= 1;
    DlT[0][target_node_2->dofs()[0]] += 1;
    DlT[1][target_node_1->dofs()[1]] -= 1;
    DlT[1][target_node_2->dofs()[1]] += 1;
    DlT[2][target_node_1->dofs()[2]] -= 1;
    DlT[2][target_node_2->dofs()[2]] += 1;
  }
  else
  {
    DlT[0][target_node_1->dofs()[0]] += 1;
    DlT[0][target_node_2->dofs()[0]] -= 1;
    DlT[1][target_node_1->dofs()[1]] += 1;
    DlT[1][target_node_2->dofs()[1]] -= 1;
    DlT[2][target_node_1->dofs()[2]] += 1;
    DlT[2][target_node_2->dofs()[2]] -= 1;
  }


  for (CI p = DlT[0].begin(); p != DlT[0].end(); ++p)
    (DlengthT)[p->first] += (p->second) * v_target[0] * 1.0 / (length_target);
  for (CI p = DlT[1].begin(); p != DlT[1].end(); ++p)
    (DlengthT)[p->first] += (p->second) * v_target[1] * 1.0 / (length_target);
  for (CI p = DlT[2].begin(); p != DlT[2].end(); ++p)
    (DlengthT)[p->first] += (p->second) * v_target[2] * 1.0 / (length_target);

  // delta length_source
  std::vector<Core::Gen::Pairedvector<int, double>> DlS(3, 1000);
  Core::Gen::Pairedvector<int, double> DlengthS(1000);

  DlS[0][source_node_1->dofs()[0]] += 1;
  DlS[0][source_node_2->dofs()[0]] -= 1;
  DlS[1][source_node_1->dofs()[1]] += 1;
  DlS[1][source_node_2->dofs()[1]] -= 1;
  DlS[2][source_node_1->dofs()[2]] += 1;
  DlS[2][source_node_2->dofs()[2]] -= 1;

  for (CI p = DlS[0].begin(); p != DlS[0].end(); ++p)
    (DlengthS)[p->first] += (p->second) * v_source[0] * 1.0 / (length_source);
  for (CI p = DlS[1].begin(); p != DlS[1].end(); ++p)
    (DlengthS)[p->first] += (p->second) * v_source[1] * 1.0 / (length_source);
  for (CI p = DlS[2].begin(); p != DlS[2].end(); ++p)
    (DlengthS)[p->first] += (p->second) * v_source[2] * 1.0 / (length_source);

  // lin length_source * length_target
  Core::Gen::Pairedvector<int, double> prodLength(1000);

  for (CI p = DlengthS.begin(); p != DlengthS.end(); ++p)
    (prodLength)[p->first] += (p->second) * length_target;
  for (CI p = DlengthT.begin(); p != DlengthT.end(); ++p)
    (prodLength)[p->first] += (p->second) * length_source;

  // lin scaprod
  Core::Gen::Pairedvector<int, double> scaProdlin(1000);

  for (CI p = DlS[0].begin(); p != DlS[0].end(); ++p)
    (scaProdlin)[p->first] += (p->second) * v_target[0];
  for (CI p = DlS[1].begin(); p != DlS[1].end(); ++p)
    (scaProdlin)[p->first] += (p->second) * v_target[1];
  for (CI p = DlS[2].begin(); p != DlS[2].end(); ++p)
    (scaProdlin)[p->first] += (p->second) * v_target[2];

  for (CI p = DlT[0].begin(); p != DlT[0].end(); ++p)
    (scaProdlin)[p->first] += (p->second) * v_source[0];
  for (CI p = DlT[1].begin(); p != DlT[1].end(); ++p)
    (scaProdlin)[p->first] += (p->second) * v_source[1];
  for (CI p = DlT[2].begin(); p != DlT[2].end(); ++p)
    (scaProdlin)[p->first] += (p->second) * v_source[2];

  // lin scaprod/lengthprod
  Core::Gen::Pairedvector<int, double> scaProdnormalizedLin(1000);
  for (CI p = scaProdlin.begin(); p != scaProdlin.end(); ++p)
    (scaProdnormalizedLin)[p->first] += (p->second) * 1.0 / (length_source * length_target);
  for (CI p = prodLength.begin(); p != prodLength.end(); ++p)
    (scaProdnormalizedLin)[p->first] -=
        (p->second) * scaprod * 1.0 /
        (length_source * length_target * length_source * length_target);

  // lin acos(scaledscaprod)
  double fac = (-1.0 / (sqrt(1.0 - scaledScaprod * scaledScaprod)));
  //  if(switchSign)
  //    fac *= -1.0;

  for (CI p = scaProdnormalizedLin.begin(); p != scaProdnormalizedLin.end(); ++p)
    (lineAngle)[p->first] += (p->second) * fac;

  return angleRad;
}

FOUR_C_NAMESPACE_CLOSE
