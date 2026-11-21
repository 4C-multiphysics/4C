// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_coupling3d.hpp"

#include "4C_contact_element.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_integrator.hpp"
#include "4C_contact_integrator_factory.hpp"
#include "4C_contact_interpolator.hpp"
#include "4C_contact_node.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_mortar_coupling3d_classes.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_projector.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 11/08|
 *----------------------------------------------------------------------*/
CONTACT::Coupling3d::Coupling3d(Core::FE::Discretization& idiscret, int dim, bool quad,
    Teuchos::ParameterList& params, Mortar::Element& source_elem, Mortar::Element& target_elem)
    : Mortar::Coupling3d(idiscret, dim, quad, params, source_elem, target_elem),
      stype_(Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(params, "STRATEGY"))
{
  // empty constructor

  return;
}

/*----------------------------------------------------------------------*
 |  Build auxiliary plane from source element (public)         popp 11/08|
 *----------------------------------------------------------------------*/
bool CONTACT::Coupling3d::auxiliary_plane()
{
  // we first need the element center:
  // for quad4, quad8, quad9 elements: xi = eta = 0.0
  // for tri3, tri6 elements: xi = eta = 1/3
  double loccenter[2];

  Core::FE::CellType dt = source_int_element().shape();
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
  source_int_element().local_to_global(loccenter, auxc(), 0);

  // we then compute the unit normal vector at the element center
  lauxn() = source_int_element().compute_unit_normal_at_xi(loccenter, auxn());

  // THIS IS CONTACT-SPECIFIC!
  // also compute linearization of the unit normal vector
  source_int_element().deriv_unit_normal_at_xi(loccenter, get_deriv_auxn());

  // std::cout << "Source Element: " << SourceIntElement().Id() << std::endl;
  // std::cout << "->Center: " << Auxc()[0] << " " << Auxc()[1] << " " << Auxc()[2] << std::endl;
  // std::cout << "->Normal: " << Auxn()[0] << " " << Auxn()[1] << " " << Auxn()[2] << std::endl;

  return true;
}

/*----------------------------------------------------------------------*
 |  Integration of cells (3D)                                 popp 11/08|
 *----------------------------------------------------------------------*/
bool CONTACT::Coupling3d::integrate_cells(
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  /**********************************************************************/
  /* INTEGRATION                                                        */
  /* Integrate the Mortar matrix M and the weighted gap function g~ on  */
  /* the current integration cell of the source / target element pair    */
  /**********************************************************************/

  static const auto algo = Teuchos::getIntegralValue<Mortar::AlgorithmType>(imortar_, "ALGORITHM");

  // do nothing if there are no cells
  if (cells().size() == 0) return false;

  // create a CONTACT integrator instance with correct num_gp and Dim
  // it is sufficient to do this once as all IntCells are triangles
  std::shared_ptr<CONTACT::Integrator> integrator =
      CONTACT::INTEGRATOR::build_integrator(stype_, imortar_, cells()[0]->shape(), get_comm());
  // loop over all integration cells
  for (int i = 0; i < (int)(cells().size()); ++i)
  {
    // integrate cell only if it has a non-zero area
    if (cells()[i]->area() < MORTARINTLIM * source_element_area()) continue;

    // set segmentation status of all source nodes
    // (hassegment_ of a source node is true if ANY segment/cell
    // is integrated that contributes to this source node)
    int nnodes = source_int_element().num_node();
    Core::Nodes::Node** mynodes = source_int_element().nodes();
    if (!mynodes) FOUR_C_THROW("Null pointer!");
    for (int k = 0; k < nnodes; ++k)
    {
      Mortar::Node* mycnode = dynamic_cast<Mortar::Node*>(mynodes[k]);
      if (!mycnode) FOUR_C_THROW("Null pointer!");
      mycnode->has_segment() = true;
    }

    // *******************************************************************
    // different options for mortar integration
    // *******************************************************************
    // (1) no quadratic element(s) involved -> linear LM interpolation
    // (2) quadratic element(s) involved -> quadratic LM interpolation
    // (3) quadratic element(s) involved -> linear LM interpolation
    // (4) quadratic element(s) involved -> piecew. linear LM interpolation
    // *******************************************************************
    Mortar::LagMultQuad lmtype = lag_mult_quad();

    // *******************************************************************
    // case (1)
    // *******************************************************************
    if (!quad())
    {
      integrator->integrate_deriv_cell_3d_aux_plane(
          source_element(), target_element(), cells()[i], auxn(), get_comm(), mparams_ptr);
    }
    // *******************************************************************
    // cases (2) and (3)
    // *******************************************************************
    else if ((quad() and (lmtype == Mortar::lagmult_quad or lmtype == Mortar::lagmult_lin or
                             lmtype == Mortar::lagmult_const)) or
             algo == Mortar::algorithm_gpts)
    {
      // check for standard shape functions and quadratic LM interpolation
      if (shape_fcn() == Mortar::shape_standard && lmtype == Mortar::lagmult_quad &&
          (source_element().shape() == Core::FE::CellType::quad8 ||
              source_element().shape() == Core::FE::CellType::tri6))
        FOUR_C_THROW(
            "Quad. LM interpolation for STANDARD 3D quadratic contact only feasible for "
            "quad9");

      // dynamic_cast to make sure to pass in IntElement&
      Mortar::IntElement& sintref = dynamic_cast<Mortar::IntElement&>(source_int_element());
      Mortar::IntElement& mintref = dynamic_cast<Mortar::IntElement&>(target_int_element());

      // call integrator
      integrator->integrate_deriv_cell_3d_aux_plane_quad(
          source_element(), target_element(), sintref, mintref, cells()[i], auxn());
    }

    // *******************************************************************
    // case (4)
    // *******************************************************************
    else if (quad() && lmtype == Mortar::lagmult_pwlin)
    {
      // check for dual shape functions
      if (shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin)
        FOUR_C_THROW(
            "Piecewise linear LM interpolation not yet implemented for DUAL 3D quadratic "
            "contact");

      // dynamic_cast to make sure to pass in IntElement&
      Mortar::IntElement& sintref = dynamic_cast<Mortar::IntElement&>(source_int_element());
      Mortar::IntElement& mintref = dynamic_cast<Mortar::IntElement&>(target_int_element());

      // call integrator
      integrator->integrate_deriv_cell_3d_aux_plane_quad(
          source_element(), target_element(), sintref, mintref, cells()[i], auxn());
    }

    // *******************************************************************
    // undefined case
    // *******************************************************************
    else if (quad() && lmtype == Mortar::lagmult_undefined)
    {
      FOUR_C_THROW(
          "Lagrange multiplier interpolation for quadratic elements undefined\n"
          "If you are using 2nd order mortar elements, you need to specify LM_QUAD in MORTAR "
          "COUPLING section");
    }

    // *******************************************************************
    // other cases
    // *******************************************************************
    else
      FOUR_C_THROW("IntegrateCells: Invalid case for 3D mortar contact LM interpolation");
    // *******************************************************************
  }  // cell loop

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of clip polygon vertices (3D)               popp 02/09|
 *----------------------------------------------------------------------*/
bool CONTACT::Coupling3d::vertex_linearization(
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& linvertex,
    std::map<int, double>& projpar, bool printderiv) const
{
  // linearize all aux.plane source and target nodes only ONCE
  // and use these linearizations later during lineclip linearization
  // (this speeds up the vertex linearizations in most cases, as we
  // never linearize the SAME source or target vertex more than once)

  // number of nodes
  const int num_source_rows = source_int_element().num_node();
  const int num_target_rows = target_int_element().num_node();

  // prepare storage for source and target linearizations
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> lin_source_nodes(num_source_rows,
      std::vector<Core::Gen::Pairedvector<int, double>>(3, 3 * source_element().num_node()));
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> lin_target_nodes(
      num_target_rows, std::vector<Core::Gen::Pairedvector<int, double>>(
                           3, 3 * source_element().num_node() + 3 * target_element().num_node()));

  // compute source linearizations (num_source_rows)
  source_vertex_linearization(lin_source_nodes);

  // compute target linearizations (num_target_rows)
  target_vertex_linearization(lin_target_nodes);

  //**********************************************************************
  // Clip polygon vertex linearization
  //**********************************************************************
  // loop over all clip polygon vertices
  for (int i = 0; i < (int)clip().size(); ++i)
  {
    // references to current vertex and its linearization
    const Mortar::Vertex& currv = clip()[i];
    std::vector<Core::Gen::Pairedvector<int, double>>& currlin = linvertex[i];

    // decision on vertex type (source, projtarget, linclip)
    if (currv.v_type() == Mortar::Vertex::source)
    {
      // get corresponding source id
      int sid = currv.nodeids()[0];

      // find corresponding source node linearization
      int k = 0;
      while (k < num_source_rows)
      {
        if (source_int_element().node_ids()[k] == sid) break;
        ++k;
      }

      // FOUR_C_THROW if not found
      if (k == num_source_rows) FOUR_C_THROW("Source Id not found!");

      // get the correct source node linearization
      currlin = lin_source_nodes[k];
    }
    else if (currv.v_type() == Mortar::Vertex::projtarget)
    {
      // get corresponding target id
      int tid = currv.nodeids()[0];

      // find corresponding target node linearization
      int k = 0;
      while (k < num_target_rows)
      {
        if (target_int_element().node_ids()[k] == tid) break;
        ++k;
      }

      // FOUR_C_THROW if not found
      if (k == num_target_rows) FOUR_C_THROW("Target Id not found!");

      // get the correct target node linearization
      currlin = lin_target_nodes[k];
    }
    else if (currv.v_type() == Mortar::Vertex::lineclip)
    {
      // get references to the two source vertices
      int sindex1 = -1;
      int sindex2 = -1;
      for (int j = 0; j < (int)source_vertices().size(); ++j)
      {
        if (source_vertices()[j].nodeids()[0] == currv.nodeids()[0]) sindex1 = j;
        if (source_vertices()[j].nodeids()[0] == currv.nodeids()[1]) sindex2 = j;
      }
      if (sindex1 < 0 || sindex2 < 0 || sindex1 == sindex2)
        FOUR_C_THROW("Lineclip linearization: (S) Something went wrong!");

      const Mortar::Vertex* source_vertex_1 = &source_vertices()[sindex1];
      const Mortar::Vertex* source_vertex_2 = &source_vertices()[sindex2];

      // get references to the two target vertices
      int tindex1 = -1;
      int tindex2 = -1;
      for (int j = 0; j < (int)target_vertices().size(); ++j)
      {
        if (target_vertices()[j].nodeids()[0] == currv.nodeids()[2]) tindex1 = j;
        if (target_vertices()[j].nodeids()[0] == currv.nodeids()[3]) tindex2 = j;
      }
      if (tindex1 < 0 || tindex2 < 0 || tindex1 == tindex2)
        FOUR_C_THROW("Lineclip linearization: (M) Something went wrong!");

      const Mortar::Vertex* target_vertex_1 = &target_vertices()[tindex1];
      const Mortar::Vertex* target_vertex_2 = &target_vertices()[tindex2];

      // do lineclip vertex linearization
      lineclip_vertex_linearization(currv, currlin, source_vertex_1, source_vertex_2,
          target_vertex_1, target_vertex_2, lin_source_nodes, lin_target_nodes);
    }

    else
      FOUR_C_THROW("VertexLinearization: Invalid Vertex Type!");
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of source vertex (3D) AuxPlane               popp 03/09|
 *----------------------------------------------------------------------*/
bool CONTACT::Coupling3d::source_vertex_linearization(
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& currlin) const
{
  // we first need the source element center:
  // for quad4, quad8, quad9 elements: xi = eta = 0.0
  // for tri3, tri6 elements: xi = eta = 1/3
  double scxi[2];

  Core::FE::CellType dt = source_int_element().shape();
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
  const int nrow = source_int_element().num_node();
  Core::LinAlg::SerialDenseVector source_val(nrow);
  Core::LinAlg::SerialDenseMatrix source_deriv(nrow, 2, true);
  source_int_element().evaluate_shape(scxi, source_val, source_deriv, nrow);

  // we need all participating source nodes
  Core::Nodes::Node** source_nodes = source_int_element().nodes();
  std::vector<Mortar::Node*> source_mortar_nodes(nrow);

  for (int i = 0; i < nrow; ++i)
  {
    source_mortar_nodes[i] = dynamic_cast<Mortar::Node*>(source_nodes[i]);
    if (!source_mortar_nodes[i]) FOUR_C_THROW("source_vertex_linearization: Null pointer!");
  }

  // linearization of the IntEle spatial coords
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> nodelin;
  Mortar::IntElement* source_int_ele = dynamic_cast<Mortar::IntElement*>(&source_int_element());

  if (source_int_ele == nullptr)
  {
    // resize the linearizations
    nodelin.resize(nrow, std::vector<Core::Gen::Pairedvector<int, double>>(3, 1));

    // loop over all intEle nodes
    for (int in = 0; in < nrow; ++in)
      for (int dim = 0; dim < 3; ++dim)
        nodelin[in][dim][source_mortar_nodes[in]->dofs()[dim]] += 1.;
  }
  else
    source_int_ele->node_linearization(nodelin);

  // map iterator
  using CI = Core::Gen::Pairedvector<int,
      double>::const_iterator;  // linearization of element center Auxc()
  std ::vector<Core::Gen::Pairedvector<int, double>> linauxc(
      3, source_element().num_node());  // assume 3 dofs per node

  for (int i = 0; i < nrow; ++i)
    for (int dim = 0; dim < 3; ++dim)
      for (CI p = nodelin[i][dim].begin(); p != nodelin[i][dim].end(); ++p)
        linauxc[dim][p->first] = source_val[i] * p->second;

  // linearization of element normal Auxn()
  const std::vector<Core::Gen::Pairedvector<int, double>>& linauxn = get_deriv_auxn();

  // put everything together for source vertex linearization
  // loop over all vertices
  for (int i = 0; i < source_int_element().num_node(); ++i)
  {
    Mortar::Node* mortar_source_node = dynamic_cast<Mortar::Node*>(source_int_element().nodes()[i]);
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
    for (CI p = linauxc[0].begin(); p != linauxc[0].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[0] * auxn()[k] * (p->second);

    for (CI p = linauxc[1].begin(); p != linauxc[1].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[1] * auxn()[k] * (p->second);

    for (CI p = linauxc[2].begin(); p != linauxc[2].end(); ++p)
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

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of source vertex (3D) AuxPlane               popp 03/09|
 *----------------------------------------------------------------------*/
bool CONTACT::Coupling3d::target_vertex_linearization(
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& currlin) const
{
  // we first need the source element center:
  // for quad4, quad8, quad9 elements: xi = eta = 0.0
  // for tri3, tri6 elements: xi = eta = 1/3
  double scxi[2];

  Core::FE::CellType dt = source_int_element().shape();
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
  int nrow = source_int_element().num_node();
  Core::LinAlg::SerialDenseVector source_val(nrow);
  Core::LinAlg::SerialDenseMatrix source_deriv(nrow, 2, true);
  source_int_element().evaluate_shape(scxi, source_val, source_deriv, nrow);

  // we need all participating source nodes
  Core::Nodes::Node** source_nodes = source_int_element().nodes();
  std::vector<Mortar::Node*> source_mortar_nodes(nrow);

  for (int i = 0; i < nrow; ++i)
  {
    source_mortar_nodes[i] = dynamic_cast<Mortar::Node*>(source_nodes[i]);
    if (!source_mortar_nodes[i]) FOUR_C_THROW("target_vertex_linearization: Null pointer!");
  }

  // linearization of the SourceIntEle spatial coords
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> source_node_lin;
  Mortar::IntElement* source_int_ele = dynamic_cast<Mortar::IntElement*>(&source_int_element());

  if (source_int_ele == nullptr)
  {
    // resize the linearizations
    source_node_lin.resize(nrow, std::vector<Core::Gen::Pairedvector<int, double>>(3, 1));

    // loop over all intEle nodes
    for (int in = 0; in < nrow; ++in)
      for (int dim = 0; dim < 3; ++dim)
        source_node_lin[in][dim][source_mortar_nodes[in]->dofs()[dim]] += 1.;
  }
  else
    source_int_ele->node_linearization(source_node_lin);

  // map iterator
  using CI = Core::Gen::Pairedvector<int,
      double>::const_iterator;  // linearization of element center Auxc()
  std ::vector<Core::Gen::Pairedvector<int, double>> linauxc(
      3, source_element().num_node());  // assume 3 dofs per node

  for (int i = 0; i < nrow; ++i)
    for (int dim = 0; dim < 3; ++dim)
      for (CI p = source_node_lin[i][dim].begin(); p != source_node_lin[i][dim].end(); ++p)
        linauxc[dim][p->first] = source_val[i] * p->second;

  // linearization of element normal Auxn()
  const std::vector<Core::Gen::Pairedvector<int, double>>& linauxn = get_deriv_auxn();

  // linearization of the TargetIntEle spatial coords
  std::vector<std::vector<Core::Gen::Pairedvector<int, double>>> target_node_lin;
  Mortar::IntElement* target_int_ele = dynamic_cast<Mortar::IntElement*>(&target_int_element());

  if (target_int_ele == nullptr)
  {
    // resize the linearizations
    target_node_lin.resize(
        target_int_element().num_node(), std::vector<Core::Gen::Pairedvector<int, double>>(3, 1));

    // loop over all intEle nodes
    for (int in = 0; in < target_int_element().num_node(); ++in)
    {
      Mortar::Node* mortar_target_node =
          dynamic_cast<Mortar::Node*>(target_int_element().nodes()[in]);
      if (mortar_target_node == nullptr) FOUR_C_THROW("dynamic cast to mortar node went wrong");

      for (int dim = 0; dim < 3; ++dim)
        target_node_lin[in][dim][mortar_target_node->dofs()[dim]] += 1.;
    }
  }
  else
    target_int_ele->node_linearization(target_node_lin);

  // put everything together for source vertex linearization
  // loop over all vertices
  for (int i = 0; i < target_int_element().num_node(); ++i)
  {
    Mortar::Node* mortar_target_node = dynamic_cast<Mortar::Node*>(target_int_element().nodes()[i]);
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
    for (CI p = linauxc[0].begin(); p != linauxc[0].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[0] * auxn()[k] * (p->second);

    for (CI p = linauxc[1].begin(); p != linauxc[1].end(); ++p)
      for (int k = 0; k < 3; ++k) currlin[i][k][p->first] += auxn()[1] * auxn()[k] * (p->second);

    for (CI p = linauxc[2].begin(); p != linauxc[2].end(); ++p)
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

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of lineclip vertex (3D) AuxPlane            popp 03/09|
 *----------------------------------------------------------------------*/
bool CONTACT::Coupling3d::lineclip_vertex_linearization(const Mortar::Vertex& currv,
    std::vector<Core::Gen::Pairedvector<int, double>>& currlin,
    const Mortar::Vertex* source_vertex_1, const Mortar::Vertex* source_vertex_2,
    const Mortar::Vertex* target_vertex_1, const Mortar::Vertex* target_vertex_2,
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& lin_source_nodes,
    std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& lin_target_nodes) const
{
  // number of nodes
  const int num_source_rows = source_int_element().num_node();
  const int num_target_rows = target_int_element().num_node();

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
  int sid1 = currv.nodeids()[0];
  int sid2 = currv.nodeids()[1];

  // find corresponding source node linearizations
  int k = 0;
  while (k < num_source_rows)
  {
    if (source_int_element().node_ids()[k] == sid1) break;
    ++k;
  }

  // FOUR_C_THROW if not found
  if (k == num_source_rows) FOUR_C_THROW("Source Id1 not found!");

  // get the correct source node linearization
  std::vector<Core::Gen::Pairedvector<int, double>>& sourcelin0 = lin_source_nodes[k];

  k = 0;
  while (k < num_source_rows)
  {
    if (source_int_element().node_ids()[k] == sid2) break;
    ++k;
  }

  // FOUR_C_THROW if not found
  if (k == num_source_rows) FOUR_C_THROW("Source Id2 not found!");

  // get the correct source node linearization
  std::vector<Core::Gen::Pairedvector<int, double>>& sourcelin1 = lin_source_nodes[k];

  // target vertex linearization (2x)
  int tid1 = currv.nodeids()[2];
  int tid2 = currv.nodeids()[3];

  // find corresponding target node linearizations
  k = 0;
  while (k < num_target_rows)
  {
    if (target_int_element().node_ids()[k] == tid1) break;
    ++k;
  }

  // FOUR_C_THROW if not found
  if (k == num_target_rows) FOUR_C_THROW("Target Id1 not found!");

  // get the correct target node linearization
  std::vector<Core::Gen::Pairedvector<int, double>>& targetlin0 = lin_target_nodes[k];

  k = 0;
  while (k < num_target_rows)
  {
    if (target_int_element().node_ids()[k] == tid2) break;
    ++k;
  }

  // FOUR_C_THROW if not found
  if (k == num_target_rows) FOUR_C_THROW("Target Id2 not found!");

  // get the correct target node linearization
  std::vector<Core::Gen::Pairedvector<int, double>>& targetlin1 = lin_target_nodes[k];

  // linearization of element normal Auxn()
  const std::vector<Core::Gen::Pairedvector<int, double>>& linauxn = get_deriv_auxn();

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

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of clip polygon center (3D)                 popp 02/09|
 *----------------------------------------------------------------------*/
bool CONTACT::Coupling3d::center_linearization(
    const std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& linvertex,
    std::vector<Core::Gen::Pairedvector<int, double>>& lincenter) const
{
  // preparations
  int clipsize = (int)(clip().size());
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // number of nodes
  const int num_source_rows = source_element().num_node();
  const int num_target_rows = target_element().num_node();

  std::vector<double> clipcenter(3);
  for (int k = 0; k < 3; ++k) clipcenter[k] = 0.0;
  double fac = 0.0;

  // first we need node averaged center
  std::array<double, 3> nac = {0.0, 0.0, 0.0};
  for (int i = 0; i < clipsize; ++i)
    for (int k = 0; k < 3; ++k) nac[k] += (clip()[i].coord()[k] / clipsize);

  // loop over all triangles of polygon (1st round: preparations)
  for (int i = 0; i < clipsize; ++i)
  {
    std::array<double, 3> xi_i = {0.0, 0.0, 0.0};
    std::array<double, 3> xi_ip1 = {0.0, 0.0, 0.0};

    // standard case
    if (i < clipsize - 1)
    {
      for (int k = 0; k < 3; ++k) xi_i[k] = clip()[i].coord()[k];
      for (int k = 0; k < 3; ++k) xi_ip1[k] = clip()[i + 1].coord()[k];
    }
    // last vertex of clip polygon
    else
    {
      for (int k = 0; k < 3; ++k) xi_i[k] = clip()[clipsize - 1].coord()[k];
      for (int k = 0; k < 3; ++k) xi_ip1[k] = clip()[0].coord()[k];
    }

    // triangle area
    std::array<double, 3> diff1 = {0.0, 0.0, 0.0};
    std::array<double, 3> diff2 = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) diff1[k] = xi_ip1[k] - xi_i[k];
    for (int k = 0; k < 3; ++k) diff2[k] = xi_i[k] - nac[k];

    std::array<double, 3> cross = {0.0, 0.0, 0.0};
    cross[0] = diff1[1] * diff2[2] - diff1[2] * diff2[1];
    cross[1] = diff1[2] * diff2[0] - diff1[0] * diff2[2];
    cross[2] = diff1[0] * diff2[1] - diff1[1] * diff2[0];

    double Atri = 0.5 * sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);

    // add contributions to clipcenter and fac
    fac += Atri;
    for (int k = 0; k < 3; ++k) clipcenter[k] += 1.0 / 3.0 * (xi_i[k] + xi_ip1[k] + nac[k]) * Atri;
  }

  // build factors for linearization
  std::array<double, 3> z = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k) z[k] = clipcenter[k];
  double n = fac;

  // first we need linearization of node averaged center
  std::vector<Core::Gen::Pairedvector<int, double>> linnac(
      3, 3 * (num_source_rows + num_target_rows));
  const double clipsizeinv = 1.0 / clipsize;

  for (int i = 0; i < clipsize; ++i)
    for (int k = 0; k < 3; ++k)
      for (CI p = linvertex[i][k].begin(); p != linvertex[i][k].end(); ++p)
        linnac[k][p->first] += clipsizeinv * (p->second);

  // loop over all triangles of polygon (2nd round: linearization)
  for (int i = 0; i < clipsize; ++i)
  {
    std::array<double, 3> xi_i = {0.0, 0.0, 0.0};
    std::array<double, 3> xi_ip1 = {0.0, 0.0, 0.0};
    int iplus1 = 0;

    // standard case
    if (i < clipsize - 1)
    {
      for (int k = 0; k < 3; ++k) xi_i[k] = clip()[i].coord()[k];
      for (int k = 0; k < 3; ++k) xi_ip1[k] = clip()[i + 1].coord()[k];
      iplus1 = i + 1;
    }
    // last vertex of clip polygon
    else
    {
      for (int k = 0; k < 3; ++k) xi_i[k] = clip()[clipsize - 1].coord()[k];
      for (int k = 0; k < 3; ++k) xi_ip1[k] = clip()[0].coord()[k];
      iplus1 = 0;
    }

    // triangle area
    std::array<double, 3> diff1 = {0.0, 0.0, 0.0};
    std::array<double, 3> diff2 = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) diff1[k] = xi_ip1[k] - xi_i[k];
    for (int k = 0; k < 3; ++k) diff2[k] = xi_i[k] - nac[k];

    std::array<double, 3> cross = {0.0, 0.0, 0.0};
    cross[0] = diff1[1] * diff2[2] - diff1[2] * diff2[1];
    cross[1] = diff1[2] * diff2[0] - diff1[0] * diff2[2];
    cross[2] = diff1[0] * diff2[1] - diff1[1] * diff2[0];

    double Atri = 0.5 * sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);

    // linearization of cross
    std::vector<Core::Gen::Pairedvector<int, double>> lincross(
        3, 3 * (num_source_rows + num_target_rows));

    for (CI p = linvertex[i][0].begin(); p != linvertex[i][0].end(); ++p)
    {
      lincross[1][p->first] += diff1[2] * (p->second);
      lincross[1][p->first] += diff2[2] * (p->second);
      lincross[2][p->first] -= diff1[1] * (p->second);
      lincross[2][p->first] -= diff2[1] * (p->second);
    }
    for (CI p = linvertex[i][1].begin(); p != linvertex[i][1].end(); ++p)
    {
      lincross[0][p->first] -= diff1[2] * (p->second);
      lincross[0][p->first] -= diff2[2] * (p->second);
      lincross[2][p->first] += diff1[0] * (p->second);
      lincross[2][p->first] += diff2[0] * (p->second);
    }
    for (CI p = linvertex[i][2].begin(); p != linvertex[i][2].end(); ++p)
    {
      lincross[0][p->first] += diff1[1] * (p->second);
      lincross[0][p->first] += diff2[1] * (p->second);
      lincross[1][p->first] -= diff1[0] * (p->second);
      lincross[1][p->first] -= diff2[0] * (p->second);
    }

    for (CI p = linvertex[iplus1][0].begin(); p != linvertex[iplus1][0].end(); ++p)
    {
      lincross[1][p->first] -= diff2[2] * (p->second);
      lincross[2][p->first] += diff2[1] * (p->second);
    }
    for (CI p = linvertex[iplus1][1].begin(); p != linvertex[iplus1][1].end(); ++p)
    {
      lincross[0][p->first] += diff2[2] * (p->second);
      lincross[2][p->first] -= diff2[0] * (p->second);
    }
    for (CI p = linvertex[iplus1][2].begin(); p != linvertex[iplus1][2].end(); ++p)
    {
      lincross[0][p->first] -= diff2[1] * (p->second);
      lincross[1][p->first] += diff2[0] * (p->second);
    }

    for (CI p = linnac[0].begin(); p != linnac[0].end(); ++p)
    {
      lincross[1][p->first] -= diff1[2] * (p->second);
      lincross[2][p->first] += diff1[1] * (p->second);
    }
    for (CI p = linnac[1].begin(); p != linnac[1].end(); ++p)
    {
      lincross[0][p->first] += diff1[2] * (p->second);
      lincross[2][p->first] -= diff1[0] * (p->second);
    }
    for (CI p = linnac[2].begin(); p != linnac[2].end(); ++p)
    {
      lincross[0][p->first] -= diff1[1] * (p->second);
      lincross[1][p->first] += diff1[0] * (p->second);
    }

    // linearization of triangle area
    Core::Gen::Pairedvector<int, double> linarea(3 * (num_source_rows + num_target_rows));
    for (int k = 0; k < 3; ++k)
      for (CI p = lincross[k].begin(); p != lincross[k].end(); ++p)
        linarea[p->first] += 0.25 / Atri * cross[k] * (p->second);

    const double fac1 = 1.0 / (3.0 * n);

    // put everything together
    for (int k = 0; k < 3; ++k)
    {
      for (CI p = linvertex[i][k].begin(); p != linvertex[i][k].end(); ++p)
        lincenter[k][p->first] += fac1 * Atri * (p->second);

      for (CI p = linvertex[iplus1][k].begin(); p != linvertex[iplus1][k].end(); ++p)
        lincenter[k][p->first] += fac1 * Atri * (p->second);

      for (CI p = linnac[k].begin(); p != linnac[k].end(); ++p)
        lincenter[k][p->first] += fac1 * Atri * (p->second);

      for (CI p = linarea.begin(); p != linarea.end(); ++p)
      {
        lincenter[k][p->first] += fac1 * (xi_i[k] + xi_ip1[k] + nac[k]) * (p->second);
        lincenter[k][p->first] -= z[k] / (n * n) * (p->second);
      }
    }
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 11/08|
 *----------------------------------------------------------------------*/
CONTACT::Coupling3dQuad::Coupling3dQuad(Core::FE::Discretization& idiscret, int dim, bool quad,
    Teuchos::ParameterList& params, Mortar::Element& source_elem, Mortar::Element& target_elem,
    Mortar::IntElement& sintele, Mortar::IntElement& mintele)
    : CONTACT::Coupling3d(idiscret, dim, quad, params, source_elem, target_elem),
      source_int_ele_(sintele),
      target_int_ele_(mintele)
{
  //  3D quadratic coupling only for quadratic ansatz type
  if (!Coupling3dQuad::quad()) FOUR_C_THROW("Coupling3dQuad called for non-quadratic ansatz!");
}

/*----------------------------------------------------------------------*
 |  get communicator  (public)                               farah 01/13|
 *----------------------------------------------------------------------*/
MPI_Comm CONTACT::Coupling3dManager::get_comm() const { return idiscret_.get_comm(); }

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 11/08|
 *----------------------------------------------------------------------*/
CONTACT::Coupling3dManager::Coupling3dManager(Core::FE::Discretization& idiscret, int dim,
    bool quad, Teuchos::ParameterList& params, Mortar::Element* source_elem,
    std::vector<Mortar::Element*> target_elem)
    : idiscret_(idiscret),
      dim_(dim),
      quad_(quad),
      imortar_(params),
      source_elem_(source_elem),
      target_elem_(target_elem),
      ncells_(0),
      stype_(Teuchos::getIntegralValue<CONTACT::SolvingStrategy>(params, "STRATEGY"))
{
  return;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                            farah 01/13|
 *----------------------------------------------------------------------*/
CONTACT::Coupling3dQuadManager::Coupling3dQuadManager(Core::FE::Discretization& idiscret, int dim,
    bool quad, Teuchos::ParameterList& params, Mortar::Element* source_elem,
    std::vector<Mortar::Element*> target_elem)
    : Mortar::Coupling3dQuadManager(idiscret, dim, quad, params, source_elem, target_elem),
      CONTACT::Coupling3dManager(idiscret, dim, quad, params, source_elem, target_elem),
      source_target_int_pairs_(-1),
      intcells_(-1)
{
  return;
}


/*----------------------------------------------------------------------*
 |  Evaluate mortar coupling pairs                            popp 03/09|
 *----------------------------------------------------------------------*/
void CONTACT::Coupling3dManager::integrate_coupling(
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  // get algorithm
  auto algo = Teuchos::getIntegralValue<Mortar::AlgorithmType>(imortar_, "ALGORITHM");

  // prepare linearizations
  if (algo == Mortar::algorithm_mortar)
    dynamic_cast<CONTACT::Element&>(source_element()).prepare_dderiv(target_elements());

  // decide which type of numerical integration scheme

  //**********************************************************************
  // STANDARD INTEGRATION (SEGMENTS)
  //**********************************************************************
  if (int_type() == Mortar::inttype_segments)
  {
    // loop over all target elements associated with this source element
    for (int t = 0; t < (int)target_elements().size(); ++t)
    {
      // create Coupling3d object and push back
      coupling().push_back(std::make_shared<Coupling3d>(
          idiscret_, dim_, false, imortar_, source_element(), target_element(t)));

      // do coupling
      coupling()[t]->evaluate_coupling();

      // store number of intcells
      ncells_ += (int)(coupling()[t]->cells()).size();
    }

    // special treatment of boundary elements
    consistent_dual_shape();

    // TODO modification of boundary shapes!

    // integrate cells
    for (int i = 0; i < (int)coupling().size(); ++i)
    {
      // temporary t-matrix linearization of this source/target pair
      if (algo == Mortar::algorithm_mortar)
        dynamic_cast<CONTACT::Element&>(source_element()).prepare_mderiv(target_elements(), i);

      // integrate cells
      coupling()[i]->integrate_cells(mparams_ptr);

      // assemble t-matrix for this source/target pair
      if (algo == Mortar::algorithm_mortar)
        dynamic_cast<CONTACT::Element&>(source_element())
            .assemble_mderiv_to_nodes(coupling()[i]->target_element());
    }
  }

  //**********************************************************************
  // ELEMENT-BASED INTEGRATION
  //**********************************************************************
  else if (int_type() == Mortar::inttype_elements || int_type() == Mortar::inttype_elements_BS)
  {
    if ((int)target_elements().size() == 0) return;

    if (!quad())
    {
      bool boundary_ele = false;
      bool proj = false;

      /* find all feasible target elements (this check is inherent in the
       * segment based integration)                    hiermeier 04/16 */
      std::vector<Mortar::Element*> feasible_ma_eles(target_elements().size());
      find_feasible_target_elements(feasible_ma_eles);

      // create an integrator instance with correct num_gp and Dim
      std::shared_ptr<CONTACT::Integrator> integrator = CONTACT::INTEGRATOR::build_integrator(
          stype_, imortar_, source_element().shape(), get_comm());

      // Perform integration and linearization
      integrator->integrate_deriv_ele_3d(
          source_element(), feasible_ma_eles, &boundary_ele, &proj, get_comm(), mparams_ptr);


      if (int_type() == Mortar::inttype_elements_BS)
      {
        if (boundary_ele == true)
        {
          // loop over all target elements associated with this source element
          for (int t = 0; t < (int)target_elements().size(); ++t)
          {
            // create Coupling3d object and push back
            coupling().push_back(std::make_shared<Coupling3d>(
                idiscret_, dim_, false, imortar_, source_element(), target_element(t)));

            // do coupling
            coupling()[t]->evaluate_coupling();

            // store number of intcells
            ncells_ += (int)(coupling()[t]->cells()).size();
          }

          // special treatment of boundary elements
          consistent_dual_shape();

          // integrate cells
          for (int i = 0; i < (int)coupling().size(); ++i)
          {
            // temporary t-matrix linearization of this source/target pair
            if (algo == Mortar::algorithm_mortar)
              dynamic_cast<CONTACT::Element&>(source_element())
                  .prepare_mderiv(target_elements(), i);

            // integrate cells
            coupling()[i]->integrate_cells(mparams_ptr);

            // assemble t-matrix for this source/target pair
            if (algo == Mortar::algorithm_mortar)
              dynamic_cast<CONTACT::Element&>(source_element())
                  .assemble_mderiv_to_nodes(coupling()[i]->target_element());
          }
        }
      }
    }
    else
    {
      FOUR_C_THROW(
          "You should not be here! This coupling manager is not able to perform mortar coupling "
          "for high-order elements.");
    }
  }
  //**********************************************************************
  // INVALID TYPE OF NUMERICAL INTEGRATION
  //**********************************************************************
  else
  {
    FOUR_C_THROW("Invalid type of numerical integration!");
  }

  // free memory of dual shape function coefficient matrix
  source_element().mo_data().reset_dual_shape();
  source_element().mo_data().reset_deriv_dual_shape();

  // assemble element contribution to nodes
  if (algo == Mortar::algorithm_mortar)
  {
    bool dual =
        (shape_fcn() == Mortar::shape_dual) || (shape_fcn() == Mortar::shape_petrovgalerkin);
    dynamic_cast<CONTACT::Element&>(source_element()).assemble_dderiv_to_nodes(dual);
  }
  return;
}


/*----------------------------------------------------------------------*
 |  Evaluate coupling pairs                                 farah 09/14 |
 *----------------------------------------------------------------------*/
bool CONTACT::Coupling3dManager::evaluate_coupling(
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  // decide which type of coupling should be evaluated
  auto algo = Teuchos::getIntegralValue<Mortar::AlgorithmType>(imortar_, "ALGORITHM");

  //*********************************
  // Mortar Contact
  //*********************************
  if (algo == Mortar::algorithm_mortar || algo == Mortar::algorithm_gpts)
    integrate_coupling(mparams_ptr);

  //*********************************
  // Error
  //*********************************
  else
    FOUR_C_THROW("chose contact algorithm not supported!");

  // interpolate temperatures in TSI case
  if (imortar_.get<CONTACT::Problemtype>("PROBTYPE") == CONTACT::Problemtype::tsi)
    NTS::Interpolator(imortar_, dim_)
        .interpolate_target_temp_3d(source_element(), target_elements());

  return true;
}


/*----------------------------------------------------------------------*
 |  Evaluate mortar coupling pairs for Quad-coupling         farah 09/14|
 *----------------------------------------------------------------------*/
void CONTACT::Coupling3dQuadManager::integrate_coupling(
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  // get algorithm type
  auto algo = Teuchos::getIntegralValue<Mortar::AlgorithmType>(
      Mortar::Coupling3dQuadManager::imortar_, "ALGORITHM");

  // prepare linearizations
  if (algo == Mortar::algorithm_mortar)
    dynamic_cast<CONTACT::Element&>(source_element()).prepare_dderiv(target_elements());

  // decide which type of numerical integration scheme

  //**********************************************************************
  // STANDARD INTEGRATION (SEGMENTS)
  //**********************************************************************
  if (int_type() == Mortar::inttype_segments)
  {
    coupling().resize(0);

    // build linear integration elements from quadratic Mortar::Elements
    std::vector<std::shared_ptr<Mortar::IntElement>> sauxelements;
    std::vector<std::vector<std::shared_ptr<Mortar::IntElement>>> mauxelements(
        target_elements().size());
    split_int_elements(source_element(), sauxelements);

    // loop over all target elements associated with this source element
    for (int t = 0; t < (int)target_elements().size(); ++t)
    {
      // build linear integration elements from quadratic Mortar::Elements
      mauxelements[t].resize(0);
      split_int_elements(*target_elements()[t], mauxelements[t]);

      // loop over all IntElement pairs for coupling
      for (int i = 0; i < (int)sauxelements.size(); ++i)
      {
        for (int j = 0; j < (int)mauxelements[t].size(); ++j)
        {
          coupling().push_back(std::make_shared<Coupling3dQuad>(discret(), n_dim(), true, params(),
              source_element(), *target_elements()[t], *sauxelements[i], *mauxelements[t][j]));

          coupling()[coupling().size() - 1]->evaluate_coupling();

          // increase counter of source/target integration pairs and intcells
          source_target_int_pairs_ += 1;
          intcells_ += (int)coupling()[coupling().size() - 1]->cells().size();
        }  // for maux
      }  // for saux
    }  // for t

    consistent_dual_shape();

    // integrate cells
    for (int i = 0; i < (int)coupling().size(); ++i)
    {
      if (algo == Mortar::algorithm_mortar)
        dynamic_cast<CONTACT::Element&>(source_element())
            .prepare_mderiv(target_elements(), i % mauxelements.size());

      coupling()[i]->integrate_cells(mparams_ptr);

      if (algo == Mortar::algorithm_mortar)
        dynamic_cast<CONTACT::Element&>(source_element())
            .assemble_mderiv_to_nodes(coupling()[i]->target_element());
    }
  }

  //**********************************************************************
  // FAST INTEGRATION (ELEMENTS)
  //**********************************************************************
  else if (int_type() == Mortar::inttype_elements || int_type() == Mortar::inttype_elements_BS)
  {
    // check for standard shape functions and quadratic LM interpolation
    if (shape_fcn() == Mortar::shape_standard && lag_mult_quad() == Mortar::lagmult_quad &&
        (source_element().shape() == Core::FE::CellType::quad8 ||
            source_element().shape() == Core::FE::CellType::tri6))
      FOUR_C_THROW(
          "Quad. LM interpolation for STANDARD 3D quadratic contact only feasible for "
          "quad9");

    if ((int)target_elements().size() == 0) return;

    // create an integrator instance with correct num_gp and Dim
    std::shared_ptr<CONTACT::Integrator> integrator = CONTACT::INTEGRATOR::build_integrator(
        stype_, params(), source_element().shape(), get_comm());

    bool boundary_ele = false;
    bool proj = false;

    // Perform integration and linearization
    integrator->integrate_deriv_ele_3d(
        source_element(), target_elements(), &boundary_ele, &proj, get_comm(), mparams_ptr);

    if (int_type() == Mortar::inttype_elements_BS)
    {
      if (boundary_ele == true)
      {
        coupling().resize(0);

        // build linear integration elements from quadratic Mortar::Elements
        std::vector<std::shared_ptr<Mortar::IntElement>> sauxelements;
        std::vector<std::vector<std::shared_ptr<Mortar::IntElement>>> mauxelements(
            target_elements().size());
        split_int_elements(source_element(), sauxelements);

        // loop over all target elements associated with this source element
        for (int t = 0; t < (int)target_elements().size(); ++t)
        {
          // build linear integration elements from quadratic Mortar::Elements
          mauxelements[t].resize(0);
          split_int_elements(*target_elements()[t], mauxelements[t]);

          // loop over all IntElement pairs for coupling
          for (int i = 0; i < (int)sauxelements.size(); ++i)
          {
            for (int j = 0; j < (int)mauxelements[t].size(); ++j)
            {
              coupling().push_back(std::make_shared<Coupling3dQuad>(discret(), n_dim(), true,
                  params(), source_element(), *target_elements()[t], *sauxelements[i],
                  *mauxelements[t][j]));

              coupling()[coupling().size() - 1]->evaluate_coupling();

              // increase counter of source/target integration pairs and intcells
              source_target_int_pairs_ += 1;
              intcells_ += (int)coupling()[coupling().size() - 1]->cells().size();
            }  // for maux
          }  // for saux
        }  // for t

        consistent_dual_shape();

        for (int i = 0; i < (int)coupling().size(); ++i)
        {
          if (algo == Mortar::algorithm_mortar)
            dynamic_cast<CONTACT::Element&>(source_element())
                .prepare_mderiv(target_elements(), i % mauxelements.size());
          coupling()[i]->integrate_cells(mparams_ptr);
          if (algo == Mortar::algorithm_mortar)
            dynamic_cast<CONTACT::Element&>(source_element())
                .assemble_mderiv_to_nodes(coupling()[i]->target_element());
        }
      }
    }
  }
  //**********************************************************************
  // INVALID
  //**********************************************************************
  else
  {
    FOUR_C_THROW("Invalid type of numerical integration");
  }

  // free memory of consistent dual shape function coefficient matrix
  source_element().mo_data().reset_dual_shape();
  source_element().mo_data().reset_deriv_dual_shape();

  if (algo == Mortar::algorithm_mortar)
    dynamic_cast<CONTACT::Element&>(source_element())
        .assemble_dderiv_to_nodes(
            (shape_fcn() == Mortar::shape_dual || shape_fcn() == Mortar::shape_petrovgalerkin));

  return;
}


/*----------------------------------------------------------------------*
 |  Evaluate coupling pairs for Quad-coupling                farah 01/13|
 *----------------------------------------------------------------------*/
bool CONTACT::Coupling3dQuadManager::evaluate_coupling(
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  // decide which type of coupling should be evaluated
  auto algo = Teuchos::getIntegralValue<Mortar::AlgorithmType>(params(), "ALGORITHM");

  //*********************************
  // Mortar Contact
  //*********************************
  if (algo == Mortar::algorithm_mortar || algo == Mortar::algorithm_gpts)
    integrate_coupling(mparams_ptr);

  //*********************************
  // Error
  //*********************************
  else
    FOUR_C_THROW("chosen contact algorithm not supported!");

  return true;
}


/*----------------------------------------------------------------------*
 |  Calculate dual shape functions                           seitz 07/13|
 *----------------------------------------------------------------------*/
void CONTACT::Coupling3dManager::consistent_dual_shape()
{
  static const auto algo = Teuchos::getIntegralValue<Mortar::AlgorithmType>(imortar_, "ALGORITHM");
  if (algo != Mortar::algorithm_mortar) return;

  // For standard shape functions no modification is necessary
  // A switch earlier in the process improves computational efficiency
  auto consistent =
      Teuchos::getIntegralValue<Mortar::ConsistentDualType>(imortar_, "LM_DUAL_CONSISTENT");
  if (shape_fcn() == Mortar::shape_standard || consistent == Mortar::consistent_none) return;

  // Consistent modification not yet checked for constant LM interpolation
  if (quad() == true && lag_mult_quad() == Mortar::lagmult_const &&
      consistent != Mortar::consistent_none)
    FOUR_C_THROW("Consistent dual shape functions not yet checked for constant LM interpolation!");

  if (consistent == Mortar::consistent_all && int_type() != Mortar::inttype_segments)
    FOUR_C_THROW(
        "Consistent dual shape functions on all elements only for segment-based "
        "integration");

  // do nothing if there are no coupling pairs
  if (coupling().size() == 0) return;

  // check for boundary elements in segment-based integration
  // (fast integration already has this check, so that consistent_dual_shape()
  // is only called for boundary elements)
  //
  // For NURBS elements, always compute consistent dual functions.
  // This improves robustness, since the duality is enforced at exactly
  // the same quadrature points, that the mortar integrals etc. are evaluated.
  // For Lagrange FE, the calculation of dual shape functions for fully
  // projecting elements is ok, since the integrands are polynomials (except
  // the jacobian)
  if (int_type() == Mortar::inttype_segments && consistent == Mortar::consistent_boundary)
  {
    // check, if source element is fully projecting
    // for convenience, we don't check each quadrature point
    // but only the element nodes. This usually does the job.
    bool boundary_ele = false;

    Core::FE::CellType dt_s = source_element().shape();

    double source_xi_test[2] = {0.0, 0.0};
    double alpha_test = 0.0;
    bool proj_test = false;

    Core::Nodes::Node** mynodes_test = source_element().nodes();
    if (!mynodes_test) FOUR_C_THROW("has_proj_status: Null pointer!");

    if (dt_s == Core::FE::CellType::quad4 || dt_s == Core::FE::CellType::quad8 ||
        dt_s == Core::FE::CellType::nurbs9)
    {
      for (int s_test = 0; s_test < source_element().num_node(); ++s_test)
      {
        if (s_test == 0)
        {
          source_xi_test[0] = -1.0;
          source_xi_test[1] = -1.0;
        }
        else if (s_test == 1)
        {
          source_xi_test[0] = -1.0;
          source_xi_test[1] = 1.0;
        }
        else if (s_test == 2)
        {
          source_xi_test[0] = 1.0;
          source_xi_test[1] = -1.0;
        }
        else if (s_test == 3)
        {
          source_xi_test[0] = 1.0;
          source_xi_test[1] = 1.0;
        }
        else if (s_test == 4)
        {
          source_xi_test[0] = 1.0;
          source_xi_test[1] = 0.0;
        }
        else if (s_test == 5)
        {
          source_xi_test[0] = 0.0;
          source_xi_test[1] = 1.0;
        }
        else if (s_test == 6)
        {
          source_xi_test[0] = -1.0;
          source_xi_test[1] = 0.0;
        }
        else if (s_test == 7)
        {
          source_xi_test[0] = 0.0;
          source_xi_test[1] = -1.0;
        }

        proj_test = false;
        for (int bs_test = 0; bs_test < (int)coupling().size(); ++bs_test)
        {
          double target_xi_test[2] = {0.0, 0.0};
          Mortar::Projector::impl(source_element(), coupling()[bs_test]->target_int_element())
              ->project_gauss_point_3d(source_element(), source_xi_test,
                  coupling()[bs_test]->target_int_element(), target_xi_test, alpha_test);

          Core::FE::CellType dt = coupling()[bs_test]->target_int_element().shape();
          if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
              dt == Core::FE::CellType::quad9)
          {
            if (target_xi_test[0] >= -1.0 && target_xi_test[1] >= -1.0 &&
                target_xi_test[0] <= 1.0 && target_xi_test[1] <= 1.0)
              proj_test = true;
          }
          else if (dt == Core::FE::CellType::tri3 || dt == Core::FE::CellType::tri6)
          {
            if (target_xi_test[0] >= 0.0 && target_xi_test[1] >= 0.0 && target_xi_test[0] <= 1.0 &&
                target_xi_test[1] <= 1.0 && target_xi_test[0] + target_xi_test[1] <= 1.0)
              proj_test = true;
          }
          else
          {
            FOUR_C_THROW("Non valid element type for target discretization!");
          }
        }
        if (proj_test == false) boundary_ele = true;
      }
    }

    else if (dt_s == Core::FE::CellType::tri3 || dt_s == Core::FE::CellType::tri6)
    {
      for (int s_test = 0; s_test < source_element().num_node(); ++s_test)
      {
        if (s_test == 0)
        {
          source_xi_test[0] = 0.0;
          source_xi_test[1] = 0.0;
        }
        else if (s_test == 1)
        {
          source_xi_test[0] = 1.0;
          source_xi_test[1] = 0.0;
        }
        else if (s_test == 2)
        {
          source_xi_test[0] = 0.0;
          source_xi_test[1] = 1.0;
        }
        else if (s_test == 3)
        {
          source_xi_test[0] = 0.5;
          source_xi_test[1] = 0.0;
        }
        else if (s_test == 4)
        {
          source_xi_test[0] = 0.5;
          source_xi_test[1] = 0.5;
        }
        else if (s_test == 5)
        {
          source_xi_test[0] = 0.0;
          source_xi_test[1] = 0.5;
        }

        proj_test = false;
        for (int bs_test = 0; bs_test < (int)coupling().size(); ++bs_test)
        {
          double target_xi_test[2] = {0.0, 0.0};
          Mortar::Projector::impl(source_element(), coupling()[bs_test]->target_element())
              ->project_gauss_point_3d(source_element(), source_xi_test,
                  coupling()[bs_test]->target_element(), target_xi_test, alpha_test);

          Core::FE::CellType dt = coupling()[bs_test]->target_element().shape();
          if (dt == Core::FE::CellType::quad4 || dt == Core::FE::CellType::quad8 ||
              dt == Core::FE::CellType::quad9)
          {
            if (target_xi_test[0] >= -1.0 && target_xi_test[1] >= -1.0 &&
                target_xi_test[0] <= 1.0 && target_xi_test[1] <= 1.0)
              proj_test = true;
          }
          else if (dt == Core::FE::CellType::tri3 || dt == Core::FE::CellType::tri6)
          {
            if (target_xi_test[0] >= 0.0 && target_xi_test[1] >= 0.0 && target_xi_test[0] <= 1.0 &&
                target_xi_test[1] <= 1.0 && target_xi_test[0] + target_xi_test[1] <= 1.0)
              proj_test = true;
          }
          else
          {
            FOUR_C_THROW("Non valid element type for target discretization!");
          }
        }
        if (proj_test == false) boundary_ele = true;
      }
    }

    else
      FOUR_C_THROW(
          "Calculation of consistent dual shape functions called for non-valid source element "
          "shape!");

    if (boundary_ele == false) return;
  }

  // source nodes and dofs
  const int max_nnodes = 9;
  const int nnodes = source_element().num_node();
  if (nnodes > max_nnodes)
    FOUR_C_THROW(
        "this function is not implemented to handle elements with that many nodes. Just adjust "
        "max_nnodes above");
  const int ndof = 3;
  const int msize = target_elements().size();

  // get number of target nodes
  int target_nodes = 0;
  for (int t = 0; t < msize; ++t) target_nodes += target_elements()[t]->num_node();

  // Dual shape functions coefficient matrix and linearization
  source_element().mo_data().deriv_dual_shape() =
      std::make_shared<Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>>(
          (nnodes + target_nodes) * ndof, 0, Core::LinAlg::SerialDenseMatrix(nnodes, nnodes));
  Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& derivae =
      *(source_element().mo_data().deriv_dual_shape());

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
  // loop over all target elements associated with this source element
  for (int t = 0; t < (int)coupling().size(); ++t)
  {
    if (!coupling()[t]->rough_check_centers()) continue;
    if (!coupling()[t]->rough_check_orient()) continue;
    if (!coupling()[t]->rough_check_centers()) continue;

    // get number of target nodes
    const int ncol = coupling()[t]->target_element().num_node();

    // loop over all integration cells
    for (int c = 0; c < (int)coupling()[t]->cells().size(); ++c)
    {
      std::shared_ptr<Mortar::IntCell> currcell = coupling()[t]->cells()[c];

      A_tot += currcell->area();

      // create an integrator for this cell
      CONTACT::Integrator integrator(imortar_, currcell->shape(), get_comm());

      // check if the cells are tri3
      // there's nothing wrong about other shapes, but as long as they are all
      // tri3 we can perform the jacobian calculation ( and its deriv) outside
      // the Gauss point loop
      if (currcell->shape() != Core::FE::CellType::tri3)
        FOUR_C_THROW("only tri3 integration cells at the moment. See comment in the code");

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
        Mortar::Projector::impl(coupling()[t]->source_int_element())
            ->project_gauss_point_auxn_3d(globgp, coupling()[t]->auxn(),
                coupling()[t]->source_int_element(), source_xi, sprojalpha);

        // project Gauss point onto source (parent) element
        double p_source_xi[2] = {0., 0.};
        double psprojalpha = 0.0;
        if (quad())
        {
          Mortar::IntElement* ie =
              dynamic_cast<Mortar::IntElement*>(&(coupling()[t]->source_int_element()));
          if (ie == nullptr) FOUR_C_THROW("nullptr pointer");
          Mortar::Projector::impl(source_element())
              ->project_gauss_point_auxn_3d(
                  globgp, coupling()[t]->auxn(), source_element(), p_source_xi, psprojalpha);
          // ie->MapToParent(source_xi,p_source_xi); // old way of doing it via affine map... wrong
          // (popp 05/2016)
        }
        else
          for (int i = 0; i < 2; ++i) p_source_xi[i] = source_xi[i];

        // create vector for shape function evaluation
        Core::LinAlg::SerialDenseVector source_val(nnodes);
        Core::LinAlg::SerialDenseMatrix source_deriv(nnodes, 2, true);

        // evaluate trace space shape functions at Gauss point
        if (lag_mult_quad() == Mortar::lagmult_lin)
          source_element().evaluate_shape_lag_mult_lin(
              Mortar::shape_standard, p_source_xi, source_val, source_deriv, nnodes);
        else
          source_element().evaluate_shape(p_source_xi, source_val, source_deriv, nnodes);

        // additional data for contact calculation (i.e. incl. derivative of dual shape functions
        // coefficient matrix) GP source coordinate derivatives
        std::vector<Core::Gen::Pairedvector<int, double>> d_source_xi_gp(2, (nnodes + ncol) * ndof);
        // GP source coordinate derivatives
        std::vector<Core::Gen::Pairedvector<int, double>> dp_source_xigp(2, (nnodes + ncol) * ndof);
        // global GP coordinate derivative on integration element
        Core::Gen::Pairedvector<int, Core::LinAlg::Matrix<3, 1>> lingp((nnodes + ncol) * ndof);

        // compute global GP coordinate derivative
        static Core::LinAlg::Matrix<3, 1> svalcell;
        static Core::LinAlg::Matrix<2, 3> sderivcell;
        currcell->evaluate_shape(eta, svalcell, sderivcell);

        for (int v = 0; v < 3; ++v)
          for (int d = 0; d < 3; ++d)
            for (CI p = (currcell->get_deriv_vertex(v))[d].begin();
                p != (currcell->get_deriv_vertex(v))[d].end(); ++p)
              lingp[p->first](d) += svalcell(v) * (p->second);

        // compute GP source coordinate derivatives
        integrator.deriv_xi_gp_3d_aux_plane(coupling()[t]->source_int_element(), source_xi,
            currcell->auxn(), d_source_xi_gp, sprojalpha, currcell->get_deriv_auxn(), lingp);

        // compute GP source coordinate derivatives (parent element)
        if (quad())
        {
          Mortar::IntElement* ie =
              dynamic_cast<Mortar::IntElement*>(&(coupling()[t]->source_int_element()));
          if (ie == nullptr) FOUR_C_THROW("wtf");
          integrator.deriv_xi_gp_3d_aux_plane(source_element(), p_source_xi, currcell->auxn(),
              dp_source_xigp, psprojalpha, currcell->get_deriv_auxn(), lingp);
          // ie->MapToParent(d_source_xi_gp,dp_source_xigp); // old way of doing it via affine
          // map... wrong (popp 05/2016)
        }
        else
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
    }  // cells
  }  // target elements

  // in case of no overlap just return, as there is no integration area
  // and therefore the consistent dual shape functions are not defined.
  // This doesn't matter, as there is no associated integration domain anyway
  if (A_tot < 1.e-12) return;

  // declare dual shape functions coefficient matrix and
  // inverse of matrix M_e
  Core::LinAlg::SerialDenseMatrix ae(nnodes, nnodes, true);
  Core::LinAlg::SerialDenseMatrix meinv(nnodes, nnodes, true);

  // compute matrix A_e and inverse of matrix M_e for
  // linear interpolation of quadratic element
  if (lag_mult_quad() == Mortar::lagmult_lin)
  {
    // declare and initialize to zero inverse of Matrix M_e
    Core::LinAlg::SerialDenseMatrix meinv(nnodes, nnodes, true);

    if (source_element().shape() == Core::FE::CellType::tri6)
    {
      // reduce me to non-zero nodes before inverting
      Core::LinAlg::Matrix<3, 3> melin;
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) melin(j, k) = me(j, k);

      // invert bi-ortho matrix melin
      Core::LinAlg::inverse(melin);

      // re-inflate inverse of melin to full size
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) meinv(j, k) = melin(j, k);
    }
    else if (source_element().shape() == Core::FE::CellType::quad8 ||
             source_element().shape() == Core::FE::CellType::quad9)
    {
      // reduce me to non-zero nodes before inverting
      Core::LinAlg::Matrix<4, 4> melin;
      for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 4; ++k) melin(j, k) = me(j, k);

      // invert bi-ortho matrix melin
      Core::LinAlg::inverse(melin);

      // re-inflate inverse of melin to full size
      for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 4; ++k) meinv(j, k) = melin(j, k);
    }
    else
      FOUR_C_THROW("incorrect element shape for linear interpolation of quadratic element!");

    // get solution matrix with dual parameters
    Core::LinAlg::multiply(ae, de, meinv);
  }
  // compute matrix A_e and inverse of matrix M_e for all other cases
  else
    meinv = Core::LinAlg::invert_and_multiply_by_cholesky(me, de, ae);

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
  source_element().mo_data().dual_shape() = std::make_shared<Core::LinAlg::SerialDenseMatrix>(ae);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::Coupling3dManager::find_feasible_target_elements(
    std::vector<Mortar::Element*>& feasible_ma_eles) const
{
  // feasibility counter
  std::size_t fcount = 0;
  for (std::size_t t = 0; t < target_elements().size(); ++t)
  {
    // Build a instance of the Mortar::Coupling3d object (no linearization needed).
    Mortar::Coupling3d coup(idiscret_, dim_, false, imortar_, source_element(), target_element(t));

    // Building the target element normals and check the angles.
    if (coup.rough_check_orient())
    {
      feasible_ma_eles[fcount] = &target_element(t);
      ++fcount;
    }
  }
  feasible_ma_eles.resize(fcount);

  return;
}

FOUR_C_NAMESPACE_CLOSE
