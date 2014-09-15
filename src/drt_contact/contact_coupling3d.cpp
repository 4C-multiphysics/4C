/*!----------------------------------------------------------------------
 \file contact_coupling3d.cpp
 \brief Classes for mortar contact coupling in 3D.

 <pre>
 -------------------------------------------------------------------------
 BACI Contact library
 Copyright (2008) Technical University of Munich

 Under terms of contract T004.008.000 there is a non-exclusive license for use
 of this work by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

 This library is proprietary software. It must not be published, distributed,
 copied or altered in any form or any media without written permission
 of the copyright holder. It may be used under terms and conditions of the
 above mentioned license by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

 This library contains and makes use of software copyrighted by Sandia Corporation
 and distributed under LGPL licence. Licensing does not apply to this or any
 other third party software used here.

 Questions? Contact Dr. Michael W. Gee (gee@lnm.mw.tum.de)
 or
 Prof. Dr. Wolfgang A. Wall (wall@lnm.mw.tum.de)

 http://www.lnm.mw.tum.de

 -------------------------------------------------------------------------
 </pre>

 <pre>
 Maintainer: Alexander Popp
 popp@lnm.mw.tum.de
 http://www.lnm.mw.tum.de
 089 - 289-15238
 </pre>

 *----------------------------------------------------------------------*/

#include "contact_coupling3d.H"
#include "contact_integrator.H"
#include "../drt_contact_aug/contact_augmented_integrator.H"
#include "contact_node.H"
#include "contact_defines.H"
#include "../drt_mortar/mortar_coupling3d_classes.H"
#include "../drt_mortar/mortar_defines.H"
#include "../drt_mortar/mortar_projector.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_inpar/inpar_contact.H"
#include "../linalg/linalg_serialdensevector.H"
#include "../linalg/linalg_serialdensematrix.H"
#include "../linalg/linalg_utils.H"

#include "../drt_mortar/mortar_calc_utils.H"

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 11/08|
 *----------------------------------------------------------------------*/
CONTACT::CoCoupling3d::CoCoupling3d(DRT::Discretization& idiscret, int dim, bool quad,
             Teuchos::ParameterList& params, MORTAR::MortarElement& sele, MORTAR::MortarElement& mele) :
MORTAR::Coupling3d(idiscret,dim,quad,params,sele,mele),
stype_(DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(params,"STRATEGY"))
{
  // empty constructor

  return;
}

/*----------------------------------------------------------------------*
 |  Build auxiliary plane from slave element (public)         popp 11/08|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3d::AuxiliaryPlane()
{
  // we first need the element center:
  // for quad4, quad8, quad9 elements: xi = eta = 0.0
  // for tri3, tri6 elements: xi = eta = 1/3
  double loccenter[2];

  DRT::Element::DiscretizationType dt = SlaveIntElement().Shape();
  if (dt == MORTAR::MortarElement::tri3 || dt == MORTAR::MortarElement::tri6)
  {
    loccenter[0] = 1.0 / 3.0;
    loccenter[1] = 1.0 / 3.0;
  }
  else if (dt == MORTAR::MortarElement::quad4
      || dt == MORTAR::MortarElement::quad8
      || dt == MORTAR::MortarElement::quad9)
  {
    loccenter[0] = 0.0;
    loccenter[1] = 0.0;
  }
  else
    dserror("ERROR: AuxiliaryPlane called for unknown element type");

  // compute element center via shape fct. interpolation
  SlaveIntElement().LocalToGlobal(loccenter, Auxc(), 0);

  // we then compute the unit normal vector at the element center
  Lauxn() = SlaveIntElement().ComputeUnitNormalAtXi(loccenter, Auxn());

  // THIS IS CONTACT-SPECIFIC!
  // also compute linearization of the unit normal vector
  SlaveIntElement().DerivUnitNormalAtXi(loccenter, GetDerivAuxn());

  //std::cout << "Slave Element: " << SlaveIntElement().Id() << std::endl;
  //std::cout << "->Center: " << Auxc()[0] << " " << Auxc()[1] << " " << Auxc()[2] << std::endl;
  //std::cout << "->Normal: " << Auxn()[0] << " " << Auxn()[1] << " " << Auxn()[2] << std::endl;

  return true;
}

/*----------------------------------------------------------------------*
 |  Integration of cells (3D)                                 popp 11/08|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3d::IntegrateCells()
{
  /**********************************************************************/
  /* INTEGRATION                                                        */
  /* Integrate the Mortar matrix M and the weighted gap function g~ on  */
  /* the current integration cell of the slave / master element pair    */
  /**********************************************************************/

  // do nothing if there are no cells
  if (Cells().size() == 0)
    return false;

  // create a CONTACT integrator instance with correct NumGP and Dim
  // it is sufficient to do this once as all IntCells are triangles
  Teuchos::RCP<CONTACT::CoIntegrator> integrator = Teuchos::null;
  if (stype_ == INPAR::CONTACT::solution_augmented)
    integrator = Teuchos::rcp(new CONTACT::AugmentedIntegrator(imortar_,Cells()[0]->Shape(),Comm(),Teuchos::null));
  else
    integrator = Teuchos::rcp(new CONTACT::CoIntegrator(imortar_,Cells()[0]->Shape(),Comm()));

  // loop over all integration cells
  for (int i = 0; i < (int) (Cells().size()); ++i)
  {
    // integrate cell only if it has a non-zero area
    if (Cells()[i]->Area() < MORTARINTLIM * SlaveElementArea())
      continue;

    // debug output of integration cells in GMSH
#ifdef MORTARGMSHCELLS
    GmshOutputCells(i);
#endif // #ifdef MORTARGMSHCELLS
    // set segmentation status of all slave nodes
    // (hassegment_ of a slave node is true if ANY segment/cell
    // is integrated that contributes to this slave node)
    int nnodes = SlaveIntElement().NumNode();
    DRT::Node** mynodes = SlaveIntElement().Nodes();
    if (!mynodes)
      dserror("ERROR: Null pointer!");
    for (int k = 0; k < nnodes; ++k)
    {
      MORTAR::MortarNode* mycnode = dynamic_cast<MORTAR::MortarNode*>(mynodes[k]);
      if (!mycnode)
        dserror("ERROR: Null pointer!");
      mycnode->HasSegment() = true;
    }

    // *******************************************************************
    // different options for mortar integration
    // *******************************************************************
    // (1) no quadratic element(s) involved -> linear LM interpolation
    // (2) quadratic element(s) involved -> quadratic LM interpolation
    // (3) quadratic element(s) involved -> linear LM interpolation
    // (4) quadratic element(s) involved -> piecew. linear LM interpolation
    // *******************************************************************
    INPAR::MORTAR::LagMultQuad lmtype = LagMultQuad();

    // *******************************************************************
    // case (1)
    // *******************************************************************
    if (!Quad())
    {
      if (stype_ == INPAR::CONTACT::solution_augmented)
        Teuchos::rcp_dynamic_cast<CONTACT::AugmentedIntegrator>(integrator)->IntegrateDerivCell3DAuxPlane(SlaveElement(),MasterElement(),Cells()[i],Auxn(),Comm());
      else
        integrator->IntegrateDerivCell3DAuxPlane(SlaveElement(),MasterElement(),Cells()[i],Auxn(),Comm());
    }

    // *******************************************************************
    // cases (2) and (3)
    // *******************************************************************
    else if (stype_ != INPAR::CONTACT::solution_augmented && Quad() && (lmtype==INPAR::MORTAR::lagmult_quad_quad || lmtype==INPAR::MORTAR::lagmult_lin_lin))
    {
      // check for dual shape functions and linear LM interpolation
      if ((ShapeFcn() == INPAR::MORTAR::shape_dual
          || ShapeFcn() == INPAR::MORTAR::shape_petrovgalerkin)
          && lmtype == INPAR::MORTAR::lagmult_lin_lin)
        dserror("ERROR: Linear LM interpolation not yet implemented for DUAL 3D quadratic contact");

      // check for standard shape functions and quadratic LM interpolation
      if (ShapeFcn() == INPAR::MORTAR::shape_standard
          && lmtype == INPAR::MORTAR::lagmult_quad_quad
          && (SlaveElement().Shape() == DRT::Element::quad8
              || SlaveElement().Shape() == DRT::Element::tri6))
        dserror("ERROR: Quad. LM interpolation for STANDARD 3D quadratic contact only feasible for quad9");

      // dynamic_cast to make sure to pass in IntElement&
      MORTAR::IntElement& sintref =
          dynamic_cast<MORTAR::IntElement&>(SlaveIntElement());
      MORTAR::IntElement& mintref =
          dynamic_cast<MORTAR::IntElement&>(MasterIntElement());

      // call integrator
      integrator->IntegrateDerivCell3DAuxPlaneQuad(SlaveElement(),MasterElement(),sintref,mintref,Cells()[i],Auxn());
    }

    // *******************************************************************
    // case (4)
    // *******************************************************************
    else if (stype_ != INPAR::CONTACT::solution_augmented && Quad() && lmtype==INPAR::MORTAR::lagmult_pwlin_pwlin)
    {
      // check for dual shape functions
      if (ShapeFcn() == INPAR::MORTAR::shape_dual
          || ShapeFcn() == INPAR::MORTAR::shape_petrovgalerkin)
        dserror("ERROR: Piecewise linear LM interpolation not yet implemented for DUAL 3D quadratic contact");

      // dynamic_cast to make sure to pass in IntElement&
      MORTAR::IntElement& sintref =
          dynamic_cast<MORTAR::IntElement&>(SlaveIntElement());
      MORTAR::IntElement& mintref =
          dynamic_cast<MORTAR::IntElement&>(MasterIntElement());

      // call integrator
      integrator->IntegrateDerivCell3DAuxPlaneQuad(SlaveElement(),MasterElement(),sintref,mintref,Cells()[i],Auxn());
    }

    // *******************************************************************
    // other cases
    // *******************************************************************
    else
      dserror("ERROR: IntegrateCells: Invalid case for 3D mortar contact LM interpolation");
    // *******************************************************************
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of clip polygon vertices (3D)               popp 02/09|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3d::VertexLinearization(
    std::vector<std::vector<GEN::pairedvector<int, double> > >& linvertex,
    std::map<int, double>& projpar, bool printderiv)
{
  // linearize all aux.plane slave and master nodes only ONCE
  // and use these linearizations later during lineclip linearization
  // (this speeds up the vertex linearizations in most cases, as we
  // never linearize the SAME slave or master vertex more than once)

  // number of nodes
  const int nsrows = SlaveIntElement().NumNode();
  const int nmrows = MasterIntElement().NumNode();

  // prepare storage for slave and master linearizations
  std::vector<std::vector<GEN::pairedvector<int, double> > > linsnodes(nsrows,
      std::vector<GEN::pairedvector<int, double> >(3, 3 * nsrows));
  std::vector<std::vector<GEN::pairedvector<int, double> > > linmnodes(nmrows,
      std::vector<GEN::pairedvector<int, double> >(3, 3 * nsrows + 3 * nmrows));

  // compute slave linearizations (nsrows)
  for (int i = 0; i < nsrows; ++i)
  {
    int sid = SlaveIntElement().NodeIds()[i];
    SlaveVertexLinearization(linsnodes[i], sid);
  }

  // compute master linearizations (nmrows)
  for (int i = 0; i < nmrows; ++i)
  {
    int mid = MasterIntElement().NodeIds()[i];
    MasterVertexLinearization(linmnodes[i], mid);
  }

  //**********************************************************************
  // Clip polygon vertex linearization
  //**********************************************************************
  // loop over all clip polygon vertices
  for (int i = 0; i < (int) Clip().size(); ++i)
  {
    // references to current vertex and its linearization
    MORTAR::Vertex& currv = Clip()[i];
    std::vector<GEN::pairedvector<int, double> >& currlin = linvertex[i];

    // decision on vertex type (slave, projmaster, linclip)
    if (currv.VType() == MORTAR::Vertex::slave)
    {
      // get corresponding slave id
      int sid = currv.Nodeids()[0];

      // find corresponding slave node linearization
      int k = 0;
      while (k < nsrows)
      {
        if (SlaveIntElement().NodeIds()[k] == sid)
          break;
        ++k;
      }

      // dserror if not found
      if (k == nsrows)
        dserror("ERROR: Slave Id not found!");

      // get the correct slave node linearization
      currlin = linsnodes[k];
    }
    else if (currv.VType() == MORTAR::Vertex::projmaster)
    {
      // get corresponding master id
      int mid = currv.Nodeids()[0];

      // find corresponding master node linearization
      int k = 0;
      while (k < nmrows)
      {
        if (MasterIntElement().NodeIds()[k] == mid)
          break;
        ++k;
      }

      // dserror if not found
      if (k == nmrows)
        dserror("ERROR: Master Id not found!");

      // get the correct master node linearization
      currlin = linmnodes[k];
    }
    else if (currv.VType() == MORTAR::Vertex::lineclip)
    {
      // get references to the two slave vertices
      int sindex1 = -1;
      int sindex2 = -1;
      for (int j = 0; j < (int) SlaveVertices().size(); ++j)
      {
        if (SlaveVertices()[j].Nodeids()[0] == currv.Nodeids()[0])
          sindex1 = j;
        if (SlaveVertices()[j].Nodeids()[0] == currv.Nodeids()[1])
          sindex2 = j;
      }
      if (sindex1 < 0 || sindex2 < 0 || sindex1 == sindex2)
        dserror("ERROR: Lineclip linearization: (S) Something went wrong!");

      MORTAR::Vertex* sv1 = &SlaveVertices()[sindex1];
      MORTAR::Vertex* sv2 = &SlaveVertices()[sindex2];

      // get references to the two master vertices
      int mindex1 = -1;
      int mindex2 = -1;
      for (int j = 0; j < (int) MasterVertices().size(); ++j)
      {
        if (MasterVertices()[j].Nodeids()[0] == currv.Nodeids()[2])
          mindex1 = j;
        if (MasterVertices()[j].Nodeids()[0] == currv.Nodeids()[3])
          mindex2 = j;
      }
      if (mindex1 < 0 || mindex2 < 0 || mindex1 == mindex2)
        dserror("ERROR: Lineclip linearization: (M) Something went wrong!");

      MORTAR::Vertex* mv1 = &MasterVertices()[mindex1];
      MORTAR::Vertex* mv2 = &MasterVertices()[mindex2];

      // do lineclip vertex linearization
      LineclipVertexLinearization(currv, currlin, sv1, sv2, mv1, mv2, linsnodes,
          linmnodes);
    }

    else
      dserror("ERROR: VertexLinearization: Invalid Vertex Type!");
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of slave vertex (3D) AuxPlane               popp 03/09|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3d::SlaveVertexLinearization(
    std::vector<GEN::pairedvector<int, double> >& currlin,
    int sid)
{
  // we first need the slave element center:
  // for quad4, quad8, quad9 elements: xi = eta = 0.0
  // for tri3, tri6 elements: xi = eta = 1/3
  double scxi[2];

  DRT::Element::DiscretizationType dt = SlaveIntElement().Shape();
  if (dt == MORTAR::MortarElement::tri3 || dt == MORTAR::MortarElement::tri6)
  {
    scxi[0] = 1.0 / 3.0;
    scxi[1] = 1.0 / 3.0;
  }
  else if (dt == MORTAR::MortarElement::quad4
      || dt == MORTAR::MortarElement::quad8
      || dt == MORTAR::MortarElement::quad9)
  {
    scxi[0] = 0.0;
    scxi[1] = 0.0;
  }
  else
    dserror("ERROR: SlaveVertexLinearization called for unknown element type");

  // evlauate shape functions + derivatives at scxi
  const int nrow = SlaveIntElement().NumNode();
  LINALG::SerialDenseVector sval(nrow);
  LINALG::SerialDenseMatrix sderiv(nrow,2,true);
  SlaveIntElement().EvaluateShape(scxi,sval,sderiv,nrow);

  // we need all participating slave nodes
  DRT::Node** snodes = SlaveIntElement().Nodes();
  std::vector<MORTAR::MortarNode*> smrtrnodes(nrow);

  for (int i=0;i<nrow;++i)
  {
    smrtrnodes[i] = dynamic_cast<MORTAR::MortarNode*>(snodes[i]);
    if (!smrtrnodes[i]) dserror("ERROR: SlaveVertexLinearization: Null pointer!");
  }

  // we also need the corresponding slave node
  DRT::Node* snode = Discret().gNode(sid);
  if (!snode) dserror("ERROR: Cannot find node with gid %",sid);
  MORTAR::MortarNode* mrtrsnode = dynamic_cast<MORTAR::MortarNode*>(snode);

  // map iterator
  typedef GEN::pairedvector<int, double>  :: const_iterator _CI;    // linearization of element center Auxc()
  std  ::vector<GEN::pairedvector<int  ,double> > linauxc(3,nrow); // assume 3 dofs per node

  for (int i = 0; i < nrow; ++i)
  {
    linauxc[0][smrtrnodes[i]->Dofs()[0]] += sval[i];
    linauxc[1][smrtrnodes[i]->Dofs()[1]] += sval[i];
    linauxc[2][smrtrnodes[i]->Dofs()[2]] += sval[i];
  }

  // linearization of element normal Auxn()
  std::vector<GEN::pairedvector<int, double> >& linauxn = GetDerivAuxn();

  // put everything together for slave vertex linearization

  // (1) slave node coordinates part
  currlin[0][mrtrsnode->Dofs()[0]] += 1.0 - Auxn()[0] * Auxn()[0];
  currlin[0][mrtrsnode->Dofs()[1]] -= Auxn()[1] * Auxn()[0];
  currlin[0][mrtrsnode->Dofs()[2]] -= Auxn()[2] * Auxn()[0];
  currlin[1][mrtrsnode->Dofs()[0]] -= Auxn()[0] * Auxn()[1];
  currlin[1][mrtrsnode->Dofs()[1]] += 1.0 - Auxn()[1] * Auxn()[1];
  currlin[1][mrtrsnode->Dofs()[2]] -= Auxn()[2] * Auxn()[1];
  currlin[2][mrtrsnode->Dofs()[0]] -= Auxn()[0] * Auxn()[2];
  currlin[2][mrtrsnode->Dofs()[1]] -= Auxn()[1] * Auxn()[2];
  currlin[2][mrtrsnode->Dofs()[2]] += 1.0 - Auxn()[2] * Auxn()[2];

  // (2) slave element center coordinates (Auxc()) part
  for (_CI p = linauxc[0].begin(); p != linauxc[0].end(); ++p)
    for (int k = 0; k < 3; ++k)
      currlin[k][p->first] += Auxn()[0] * Auxn()[k] * (p->second);

  for (_CI p = linauxc[1].begin(); p != linauxc[1].end(); ++p)
    for (int k = 0; k < 3; ++k)
      currlin[k][p->first] += Auxn()[1] * Auxn()[k] * (p->second);

  for (_CI p = linauxc[2].begin(); p != linauxc[2].end(); ++p)
    for (int k = 0; k < 3; ++k)
      currlin[k][p->first] += Auxn()[2] * Auxn()[k] * (p->second);

  // (3) slave element normal (Auxn()) part
  double xdotn = (mrtrsnode->xspatial()[0] - Auxc()[0]) * Auxn()[0]
      + (mrtrsnode->xspatial()[1] - Auxc()[1]) * Auxn()[1]
      + (mrtrsnode->xspatial()[2] - Auxc()[2]) * Auxn()[2];

  for (_CI p = linauxn[0].begin(); p != linauxn[0].end(); ++p)
  {
    currlin[0][p->first] -= xdotn * (p->second);
    for (int k = 0; k < 3; ++k)
      currlin[k][p->first] -= (mrtrsnode->xspatial()[0] - Auxc()[0]) * Auxn()[k]
          * (p->second);
  }

  for (_CI p = linauxn[1].begin(); p != linauxn[1].end(); ++p)
  {
    currlin[1][p->first] -= xdotn * (p->second);
    for (int k = 0; k < 3; ++k)
      currlin[k][p->first] -= (mrtrsnode->xspatial()[1] - Auxc()[1]) * Auxn()[k]
          * (p->second);
  }

  for (_CI p = linauxn[2].begin(); p != linauxn[2].end(); ++p)
  {
    currlin[2][p->first] -= xdotn * (p->second);
    for (int k = 0; k < 3; ++k)
      currlin[k][p->first] -= (mrtrsnode->xspatial()[2] - Auxc()[2]) * Auxn()[k]
          * (p->second);
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of projmaster vertex (3D) AuxPlane          popp 03/09|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3d::MasterVertexLinearization(
    std::vector<GEN::pairedvector<int, double> >& currlin,
    int mid)
{
  // we first need the slave element center:
  // for quad4, quad8, quad9 elements: xi = eta = 0.0
  // for tri3, tri6 elements: xi = eta = 1/3
  double scxi[2];

  DRT::Element::DiscretizationType dt = SlaveIntElement().Shape();
  if (dt == MORTAR::MortarElement::tri3 || dt == MORTAR::MortarElement::tri6)
  {
    scxi[0] = 1.0 / 3.0;
    scxi[1] = 1.0 / 3.0;
  }
  else if (dt == MORTAR::MortarElement::quad4
      || dt == MORTAR::MortarElement::quad8
      || dt == MORTAR::MortarElement::quad9)
  {
    scxi[0] = 0.0;
    scxi[1] = 0.0;
  }
  else
    dserror("ERROR: MasterVertexLinearization called for unknown element type");

  // evlauate shape functions + derivatives at scxi
  int nrow = SlaveIntElement().NumNode();
  LINALG::SerialDenseVector sval(nrow);
  LINALG::SerialDenseMatrix sderiv(nrow,2,true);
  SlaveIntElement().EvaluateShape(scxi,sval,sderiv,nrow);

  // we need all participating slave nodes
  DRT::Node** snodes = SlaveIntElement().Nodes();
  std::vector<MORTAR::MortarNode*> smrtrnodes(nrow);

  for (int i=0;i<nrow;++i)
  {
    smrtrnodes[i] = dynamic_cast<MORTAR::MortarNode*>(snodes[i]);
    if (!smrtrnodes[i]) dserror("ERROR: MasterVertexLinearization: Null pointer!");
  }

  // we also need the corresponding master node
  DRT::Node* mnode = Discret().gNode(mid);
  if (!mnode) dserror("ERROR: Cannot find node with gid %",mid);
  MORTAR::MortarNode* mrtrmnode = dynamic_cast<MORTAR::MortarNode*>(mnode);

  // map iterator
  typedef GEN::pairedvector<int, double>::const_iterator _CI;  // linearization of element center Auxc()
  std  ::vector<GEN::pairedvector<int,double> > linauxc(3,nrow);

  for (int i = 0; i < nrow; ++i)
  {
    linauxc[0][smrtrnodes[i]->Dofs()[0]] += sval[i];
    linauxc[1][smrtrnodes[i]->Dofs()[1]] += sval[i];
    linauxc[2][smrtrnodes[i]->Dofs()[2]] += sval[i];
  }

  // linearization of element normal Auxn()
  std::vector<GEN::pairedvector<int, double> >& linauxn = GetDerivAuxn();

  // put everything together for master vertex linearization

  // (1) master node coordinates part
  currlin[0][mrtrmnode->Dofs()[0]] += 1.0 - Auxn()[0] * Auxn()[0];
  currlin[0][mrtrmnode->Dofs()[1]] -= Auxn()[1] * Auxn()[0];
  currlin[0][mrtrmnode->Dofs()[2]] -= Auxn()[2] * Auxn()[0];
  currlin[1][mrtrmnode->Dofs()[0]] -= Auxn()[0] * Auxn()[1];
  currlin[1][mrtrmnode->Dofs()[1]] += 1.0 - Auxn()[1] * Auxn()[1];
  currlin[1][mrtrmnode->Dofs()[2]] -= Auxn()[2] * Auxn()[1];
  currlin[2][mrtrmnode->Dofs()[0]] -= Auxn()[0] * Auxn()[2];
  currlin[2][mrtrmnode->Dofs()[1]] -= Auxn()[1] * Auxn()[2];
  currlin[2][mrtrmnode->Dofs()[2]] += 1.0 - Auxn()[2] * Auxn()[2];

  // (2) slave element center coordinates (Auxc()) part
  for (_CI p = linauxc[0].begin(); p != linauxc[0].end(); ++p)
    for (int k = 0; k < 3; ++k)
      currlin[k][p->first] += Auxn()[0] * Auxn()[k] * (p->second);

  for (_CI p = linauxc[1].begin(); p != linauxc[1].end(); ++p)
    for (int k = 0; k < 3; ++k)
      currlin[k][p->first] += Auxn()[1] * Auxn()[k] * (p->second);

  for (_CI p = linauxc[2].begin(); p != linauxc[2].end(); ++p)
    for (int k = 0; k < 3; ++k)
      currlin[k][p->first] += Auxn()[2] * Auxn()[k] * (p->second);

  // (3) slave element normal (Auxn()) part
  double xdotn = (mrtrmnode->xspatial()[0] - Auxc()[0]) * Auxn()[0]
      + (mrtrmnode->xspatial()[1] - Auxc()[1]) * Auxn()[1]
      + (mrtrmnode->xspatial()[2] - Auxc()[2]) * Auxn()[2];

  for (_CI p = linauxn[0].begin(); p != linauxn[0].end(); ++p)
  {
    currlin[0][p->first] -= xdotn * (p->second);
    for (int k = 0; k < 3; ++k)
      currlin[k][p->first] -= (mrtrmnode->xspatial()[0] - Auxc()[0]) * Auxn()[k]
          * (p->second);
  }

  for (_CI p = linauxn[1].begin(); p != linauxn[1].end(); ++p)
  {
    currlin[1][p->first] -= xdotn * (p->second);
    for (int k = 0; k < 3; ++k)
      currlin[k][p->first] -= (mrtrmnode->xspatial()[1] - Auxc()[1]) * Auxn()[k]
          * (p->second);
  }

  for (_CI p = linauxn[2].begin(); p != linauxn[2].end(); ++p)
  {
    currlin[2][p->first] -= xdotn * (p->second);
    for (int k = 0; k < 3; ++k)
      currlin[k][p->first] -= (mrtrmnode->xspatial()[2] - Auxc()[2]) * Auxn()[k]
          * (p->second);
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of lineclip vertex (3D) AuxPlane            popp 03/09|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3d::LineclipVertexLinearization(MORTAR::Vertex& currv,
    std::vector<GEN::pairedvector<int, double> >& currlin,
    MORTAR::Vertex* sv1, MORTAR::Vertex* sv2,
    MORTAR::Vertex* mv1, MORTAR::Vertex* mv2,
    std::vector<std::vector<GEN::pairedvector<int, double> > >& linsnodes,
    std::vector<std::vector<GEN::pairedvector<int, double> > >& linmnodes)
{
  // number of nodes
  const int nsrows = SlaveIntElement().NumNode();
  const int nmrows = MasterIntElement().NumNode();

  // iterator
  typedef GEN::pairedvector<int, double>::const_iterator _CI;

  // compute factor Z
  double crossZ[3] =
  { 0.0, 0.0, 0.0 };
  crossZ[0] =   (sv1->Coord()[1] - mv1->Coord()[1])
              * (mv2->Coord()[2] - mv1->Coord()[2])
              - (sv1->Coord()[2] - mv1->Coord()[2])
              * (mv2->Coord()[1] - mv1->Coord()[1]);
  crossZ[1] =   (sv1->Coord()[2] - mv1->Coord()[2])
              * (mv2->Coord()[0] - mv1->Coord()[0])
              - (sv1->Coord()[0] - mv1->Coord()[0])
              * (mv2->Coord()[2] - mv1->Coord()[2]);
  crossZ[2] =   (sv1->Coord()[0] - mv1->Coord()[0])
              * (mv2->Coord()[1] - mv1->Coord()[1])
              - (sv1->Coord()[1] - mv1->Coord()[1])
          * (mv2->Coord()[0] - mv1->Coord()[0]);
  double Zfac = crossZ[0] * Auxn()[0] + crossZ[1] * Auxn()[1]
      + crossZ[2] * Auxn()[2];

  // compute factor N
  double crossN[3] = { 0.0, 0.0, 0.0 };
  crossN[0] =   (sv2->Coord()[1] - sv1->Coord()[1])
              * (mv2->Coord()[2] - mv1->Coord()[2])
              - (sv2->Coord()[2] - sv1->Coord()[2])
              * (mv2->Coord()[1] - mv1->Coord()[1]);
  crossN[1] =   (sv2->Coord()[2] - sv1->Coord()[2])
              * (mv2->Coord()[0] - mv1->Coord()[0])
              - (sv2->Coord()[0] - sv1->Coord()[0])
              * (mv2->Coord()[2] - mv1->Coord()[2]);
  crossN[2] =   (sv2->Coord()[0] - sv1->Coord()[0])
              * (mv2->Coord()[1] - mv1->Coord()[1])
              - (sv2->Coord()[1] - sv1->Coord()[1])
              * (mv2->Coord()[0] - mv1->Coord()[0]);
  double Nfac =   crossN[0] * Auxn()[0] + crossN[1] * Auxn()[1]
                + crossN[2] * Auxn()[2];

  // slave edge vector
  double sedge[3] = { 0.0, 0.0, 0.0 };
  for (int k = 0; k < 3; ++k)
    sedge[k] = sv2->Coord()[k] - sv1->Coord()[k];

  // prepare linearization derivZ
  double crossdZ1[3] = { 0.0, 0.0, 0.0 };
  double crossdZ2[3] = { 0.0, 0.0, 0.0 };
  double crossdZ3[3] = { 0.0, 0.0, 0.0 };
  crossdZ1[0] = (mv2->Coord()[1] - mv1->Coord()[1]) * Auxn()[2]
      - (mv2->Coord()[2] - mv1->Coord()[2]) * Auxn()[1];
  crossdZ1[1] = (mv2->Coord()[2] - mv1->Coord()[2]) * Auxn()[0]
      - (mv2->Coord()[0] - mv1->Coord()[0]) * Auxn()[2];
  crossdZ1[2] = (mv2->Coord()[0] - mv1->Coord()[0]) * Auxn()[1]
      - (mv2->Coord()[1] - mv1->Coord()[1]) * Auxn()[0];
  crossdZ2[0] = Auxn()[1] * (sv1->Coord()[2] - mv1->Coord()[2])
      - Auxn()[2] * (sv1->Coord()[1] - mv1->Coord()[1]);
  crossdZ2[1] = Auxn()[2] * (sv1->Coord()[0] - mv1->Coord()[0])
      - Auxn()[0] * (sv1->Coord()[2] - mv1->Coord()[2]);
  crossdZ2[2] = Auxn()[0] * (sv1->Coord()[1] - mv1->Coord()[1])
      - Auxn()[1] * (sv1->Coord()[0] - mv1->Coord()[0]);
  crossdZ3[0] = (sv1->Coord()[1] - mv1->Coord()[1])
      * (mv2->Coord()[2] - mv1->Coord()[2])
      - (sv1->Coord()[2] - mv1->Coord()[2])
          * (mv2->Coord()[1] - mv1->Coord()[1]);
  crossdZ3[1] = (sv1->Coord()[2] - mv1->Coord()[2])
      * (mv2->Coord()[0] - mv1->Coord()[0])
      - (sv1->Coord()[0] - mv1->Coord()[0])
          * (mv2->Coord()[2] - mv1->Coord()[2]);
  crossdZ3[2] = (sv1->Coord()[0] - mv1->Coord()[0])
      * (mv2->Coord()[1] - mv1->Coord()[1])
      - (sv1->Coord()[1] - mv1->Coord()[1])
          * (mv2->Coord()[0] - mv1->Coord()[0]);

  // prepare linearization derivN
  double crossdN1[3] = { 0.0, 0.0, 0.0 };
  double crossdN2[3] = { 0.0, 0.0, 0.0 };
  double crossdN3[3] = { 0.0, 0.0, 0.0 };
  crossdN1[0] =   (mv2->Coord()[1] - mv1->Coord()[1]) * Auxn()[2]
                - (mv2->Coord()[2] - mv1->Coord()[2]) * Auxn()[1];
  crossdN1[1] =   (mv2->Coord()[2] - mv1->Coord()[2]) * Auxn()[0]
                - (mv2->Coord()[0] - mv1->Coord()[0]) * Auxn()[2];
  crossdN1[2] =   (mv2->Coord()[0] - mv1->Coord()[0]) * Auxn()[1]
                - (mv2->Coord()[1] - mv1->Coord()[1]) * Auxn()[0];
  crossdN2[0] =   Auxn()[1] * (sv2->Coord()[2] - sv1->Coord()[2])
                - Auxn()[2] * (sv2->Coord()[1] - sv1->Coord()[1]);
  crossdN2[1] =   Auxn()[2] * (sv2->Coord()[0] - sv1->Coord()[0])
                - Auxn()[0] * (sv2->Coord()[2] - sv1->Coord()[2]);
  crossdN2[2] =   Auxn()[0] * (sv2->Coord()[1] - sv1->Coord()[1])
                - Auxn()[1] * (sv2->Coord()[0] - sv1->Coord()[0]);
  crossdN3[0] =   (sv2->Coord()[1] - sv1->Coord()[1])
                * (mv2->Coord()[2] - mv1->Coord()[2])
                - (sv2->Coord()[2] - sv1->Coord()[2])
                * (mv2->Coord()[1] - mv1->Coord()[1]);
  crossdN3[1] =   (sv2->Coord()[2] - sv1->Coord()[2])
                * (mv2->Coord()[0] - mv1->Coord()[0])
                - (sv2->Coord()[0] - sv1->Coord()[0])
                * (mv2->Coord()[2] - mv1->Coord()[2]);
  crossdN3[2] =   (sv2->Coord()[0] - sv1->Coord()[0])
                * (mv2->Coord()[1] - mv1->Coord()[1])
                - (sv2->Coord()[1] - sv1->Coord()[1])
                * (mv2->Coord()[0] - mv1->Coord()[0]);

  // slave vertex linearization (2x)
  int sid1 = currv.Nodeids()[0];
  int sid2 = currv.Nodeids()[1];

  // find corresponding slave node linearizations
  int k = 0;
  while (k < nsrows)
  {
    if (SlaveIntElement().NodeIds()[k] == sid1)
      break;
    ++k;
  }

  // dserror if not found
  if (k == nsrows)
    dserror("ERROR: Slave Id1 not found!");

  // get the correct slave node linearization
  std::vector<GEN::pairedvector<int, double> >& slavelin0 = linsnodes[k];

  k = 0;
  while (k < nsrows)
  {
    if (SlaveIntElement().NodeIds()[k] == sid2)
      break;
    ++k;
  }

  // dserror if not found
  if (k == nsrows)
    dserror("ERROR: Slave Id2 not found!");

  // get the correct slave node linearization
  std::vector<GEN::pairedvector<int, double> >& slavelin1 = linsnodes[k];

  // master vertex linearization (2x)
  int mid1 = currv.Nodeids()[2];
  int mid2 = currv.Nodeids()[3];

  // find corresponding master node linearizations
  k = 0;
  while (k < nmrows)
  {
    if (MasterIntElement().NodeIds()[k] == mid1)
      break;
    ++k;
  }

  // dserror if not found
  if (k == nmrows)
    dserror("ERROR: Master Id1 not found!");

  // get the correct master node linearization
  std::vector<GEN::pairedvector<int,double> >& masterlin0 = linmnodes[k];

  k = 0;
  while (k < nmrows)
  {
    if (MasterIntElement().NodeIds()[k] == mid2)
      break;
    ++k;
  }

  // dserror if not found
  if (k == nmrows)
    dserror("ERROR: Master Id2 not found!");

  // get the correct master node linearization
  std::vector<GEN::pairedvector<int,double> >& masterlin1 = linmnodes[k];

  // linearization of element normal Auxn()
  std::vector<GEN::pairedvector<int,double> >& linauxn = GetDerivAuxn();

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
        currlin[dim][p->first] -= sedge[dim] * Nfacinv * crossdZ1[k]
            * (p->second);
        currlin[dim][p->first] -= sedge[dim] * ZNNfac * crossdN1[k]
            * (p->second);

      }
    }
    for (_CI p = slavelin1[k].begin(); p != slavelin1[k].end(); ++p)
    {
      currlin[k][p->first] -= ZNfac * (p->second);
      for (int dim = 0; dim < 3; ++dim)
      {
        currlin[dim][p->first] += sedge[dim] * ZNNfac * crossdN1[k]
            * (p->second);
      }
    }
    for (_CI p = masterlin0[k].begin(); p != masterlin0[k].end(); ++p)
    {
      for (int dim = 0; dim < 3; ++dim)
      {
        currlin[dim][p->first] += sedge[dim] * Nfacinv * crossdZ1[k]
            * (p->second);
        currlin[dim][p->first] += sedge[dim] * Nfacinv * crossdZ2[k]
            * (p->second);
        currlin[dim][p->first] -= sedge[dim] * ZNNfac * crossdN2[k]
            * (p->second);
      }
    }
    for (_CI p = masterlin1[k].begin(); p != masterlin1[k].end(); ++p)
    {
      for (int dim = 0; dim < 3; ++dim)
      {
        currlin[dim][p->first] -= sedge[dim] * Nfacinv * crossdZ2[k]
            * (p->second);
        currlin[dim][p->first] += sedge[dim] * ZNNfac * crossdN2[k]
            * (p->second);
      }
    }
    for (_CI p = linauxn[k].begin(); p != linauxn[k].end(); ++p)
    {
      for (int dim = 0; dim < 3; ++dim)
      {
        currlin[dim][p->first] -= sedge[dim] * Nfacinv * crossdZ3[k]
            * (p->second);
        currlin[dim][p->first] += sedge[dim] * ZNNfac * crossdN3[k]
            * (p->second);
      }
    }
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of clip polygon center (3D)                 popp 02/09|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3d::CenterLinearization(
    const std::vector<std::vector<GEN::pairedvector<int, double> > >& linvertex,
    std::vector<GEN::pairedvector<int, double> >& lincenter)
{
  // preparations
  int clipsize = (int) (Clip().size());
  typedef GEN::pairedvector<int, double>::const_iterator CI;

  // number of nodes
  const int nsrows = SlaveIntElement().NumNode();
  const int nmrows = MasterIntElement().NumNode();

  std::vector<double> clipcenter(3);
  for (int k = 0; k < 3; ++k)
    clipcenter[k] = 0.0;
  double fac = 0.0;

  // first we need node averaged center
  double nac[3] = { 0.0, 0.0, 0.0 };
  for (int i = 0; i < clipsize; ++i)
    for (int k = 0; k < 3; ++k)
      nac[k] += (Clip()[i].Coord()[k] / clipsize);

  // loop over all triangles of polygon (1st round: preparations)
  for (int i = 0; i < clipsize; ++i)
  {
    double xi_i[3] =   { 0.0, 0.0, 0.0 };
    double xi_ip1[3] = { 0.0, 0.0, 0.0 };

    // standard case
    if (i < clipsize - 1)
    {
      for (int k = 0; k < 3; ++k)
        xi_i[k] = Clip()[i].Coord()[k];
      for (int k = 0; k < 3; ++k)
        xi_ip1[k] = Clip()[i + 1].Coord()[k];
    }
    // last vertex of clip polygon
    else
    {
      for (int k = 0; k < 3; ++k)
        xi_i[k] = Clip()[clipsize - 1].Coord()[k];
      for (int k = 0; k < 3; ++k)
        xi_ip1[k] = Clip()[0].Coord()[k];
    }

    // triangle area
    double diff1[3] = { 0.0, 0.0, 0.0 };
    double diff2[3] = { 0.0, 0.0, 0.0 };
    for (int k = 0; k < 3; ++k)
      diff1[k] = xi_ip1[k] - xi_i[k];
    for (int k = 0; k < 3; ++k)
      diff2[k] = xi_i[k] - nac[k];

    double cross[3] =
    { 0.0, 0.0, 0.0 };
    cross[0] = diff1[1] * diff2[2] - diff1[2] * diff2[1];
    cross[1] = diff1[2] * diff2[0] - diff1[0] * diff2[2];
    cross[2] = diff1[0] * diff2[1] - diff1[1] * diff2[0];

    double Atri = 0.5
        * sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);

    // add contributions to clipcenter and fac
    fac += Atri;
    for (int k = 0; k < 3; ++k)
      clipcenter[k] += 1.0 / 3.0 * (xi_i[k] + xi_ip1[k] + nac[k]) * Atri;
  }

  // build factors for linearization
  double z[3] = { 0.0, 0.0, 0.0 };
  for (int k = 0; k < 3; ++k)
    z[k] = clipcenter[k];
  double n = fac;

  // first we need linearization of node averaged center
  std::vector<GEN::pairedvector<int, double> > linnac(3, 3 * (nsrows + nmrows));
  const double clipsizeinv = 1.0 / clipsize;

  for (int i = 0; i < clipsize; ++i)
    for (int k = 0; k < 3; ++k)
      for (CI p = linvertex[i][k].begin(); p != linvertex[i][k].end(); ++p)
        linnac[k][p->first] += clipsizeinv * (p->second);

  // loop over all triangles of polygon (2nd round: linearization)
  for (int i = 0; i < clipsize; ++i)
  {
    double xi_i[3]   = { 0.0, 0.0, 0.0 };
    double xi_ip1[3] = { 0.0, 0.0, 0.0 };
    int iplus1 = 0;

    // standard case
    if (i < clipsize - 1)
    {
      for (int k = 0; k < 3; ++k)
        xi_i[k] = Clip()[i].Coord()[k];
      for (int k = 0; k < 3; ++k)
        xi_ip1[k] = Clip()[i + 1].Coord()[k];
      iplus1 = i + 1;
    }
    // last vertex of clip polygon
    else
    {
      for (int k = 0; k < 3; ++k)
        xi_i[k] = Clip()[clipsize - 1].Coord()[k];
      for (int k = 0; k < 3; ++k)
        xi_ip1[k] = Clip()[0].Coord()[k];
      iplus1 = 0;
    }

    // triangle area
    double diff1[3] = { 0.0, 0.0, 0.0 };
    double diff2[3] = { 0.0, 0.0, 0.0 };
    for (int k = 0; k < 3; ++k)
      diff1[k] = xi_ip1[k] - xi_i[k];
    for (int k = 0; k < 3; ++k)
      diff2[k] = xi_i[k] - nac[k];

    double cross[3] = { 0.0, 0.0, 0.0 };
    cross[0] = diff1[1] * diff2[2] - diff1[2] * diff2[1];
    cross[1] = diff1[2] * diff2[0] - diff1[0] * diff2[2];
    cross[2] = diff1[0] * diff2[1] - diff1[1] * diff2[0];

    double Atri = 0.5
        * sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);

    // linearization of cross
    std::vector<GEN::pairedvector<int, double> > lincross(3,
        3 * (nsrows + nmrows));

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
    GEN::pairedvector<int, double> linarea(3 * (nsrows + nmrows));
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
CONTACT::CoCoupling3dQuad::CoCoupling3dQuad(DRT::Discretization& idiscret,
    int dim, bool quad, Teuchos::ParameterList& params,
    MORTAR::MortarElement& sele, MORTAR::MortarElement& mele,
    MORTAR::IntElement& sintele, MORTAR::IntElement& mintele) :
    CONTACT::CoCoupling3d(idiscret, dim, quad, params, sele, mele), sintele_(
        sintele), mintele_(mintele)
{
  //  3D quadratic coupling only for quadratic ansatz type
  if (!Quad())
    dserror("ERROR: CoCoupling3dQuad called for non-quadratic ansatz!");

  return;
}

/*----------------------------------------------------------------------*
 |  get communicator  (public)                               farah 01/13|
 *----------------------------------------------------------------------*/
const Epetra_Comm& CONTACT::CoCoupling3dManager::Comm() const
{
  return idiscret_.Comm();
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 11/08|
 *----------------------------------------------------------------------*/
CONTACT::CoCoupling3dManager::CoCoupling3dManager(DRT::Discretization& idiscret,
    int dim, bool quad, Teuchos::ParameterList& params,
    MORTAR::MortarElement* sele, std::vector<MORTAR::MortarElement*> mele) :
    idiscret_(idiscret),
    dim_(dim),
    quad_(quad),
    imortar_(params),
    sele_(sele),
    mele_(mele),
    ncells_(0),
    stype_(DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(params,"STRATEGY"))
{
  // evaluate coupling
  EvaluateCoupling();

  return;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                            farah 01/13|
 *----------------------------------------------------------------------*/
CONTACT::CoCoupling3dQuadManager::CoCoupling3dQuadManager(
    DRT::Discretization& idiscret, int dim, bool quad,
    Teuchos::ParameterList& params, MORTAR::MortarElement* sele,
    std::vector<MORTAR::MortarElement*> mele) :
    MORTAR::Coupling3dQuadManager(idiscret, dim, quad, params, sele, mele, true)
{
  // evaluate coupling
  EvaluateCoupling();

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate coupling pairs                                   popp 03/09|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3dManager::EvaluateCoupling()
{
  // decide which type of numerical integration scheme

  //**********************************************************************
  // STANDARD INTEGRATION (SEGMENTS)
  //**********************************************************************
  if (IntType() == INPAR::MORTAR::inttype_segments)
  {
    // switch, if consistent boundary modification chosen
    if (DRT::INPUT::IntegralValue<int>(imortar_, "LM_DUAL_CONSISTENT") == true
        && ShapeFcn() != INPAR::MORTAR::shape_standard // so for petrov-Galerkin and dual
        )
    {
      // loop over all master elements associated with this slave element
      for (int m = 0; m < (int) MasterElements().size(); ++m)
      {
        // create CoCoupling3d object and push back
        Coupling().push_back(
            Teuchos::rcp(
                new CoCoupling3d(idiscret_, dim_, false, imortar_,
                    SlaveElement(), MasterElement(m))));

        // do coupling
        Coupling()[m]->EvaluateCoupling();

        // store number of intcells
        ncells_ += (int) (Coupling()[m]->Cells()).size();
      }

      // special treatment of boundary elements
      ConsistDualShape();

      // integrate cells
      for (int m = 0; m < (int) MasterElements().size(); ++m)
        Coupling()[m]->IntegrateCells();
    }
    else
    {
      // loop over all master elements associated with this slave element
      for (int m = 0; m < (int) MasterElements().size(); ++m)
      {
        // create CoCoupling3d object and push back
        Coupling().push_back(
            Teuchos::rcp(
                new CoCoupling3d(idiscret_, dim_, false, imortar_,
                    SlaveElement(), MasterElement(m))));

        // do coupling
        Coupling()[m]->EvaluateCoupling();

        // integrate cells
        Coupling()[m]->IntegrateCells();

        // store number of intcells
        ncells_ += (int) (Coupling()[m]->Cells()).size();
      }
    }
  }
  //**********************************************************************
  // ELEMENT-BASED INTEGRATION
  //**********************************************************************
  else if (IntType() == INPAR::MORTAR::inttype_elements
      || IntType() == INPAR::MORTAR::inttype_elements_BS)
  {
    if ((int) MasterElements().size() == 0)
      return false;

    if (!Quad())
    {
      bool boundary_ele = false;
      bool proj = false;

      // create an integrator instance with correct NumGP and Dim
      Teuchos::RCP<CONTACT::CoIntegrator> integrator = Teuchos::null;
      if (stype_==INPAR::CONTACT::solution_augmented)
      {
        integrator = Teuchos::rcp(new CONTACT::AugmentedIntegrator(imortar_,SlaveElement().Shape(),Comm(),
            Teuchos::null));
        //Perform integration and linearization
        Teuchos::rcp_dynamic_cast<CONTACT::AugmentedIntegrator>(integrator)->IntegrateDerivEle3D(SlaveElement(), MasterElements(),&boundary_ele, &proj, Comm());
      }
      else
      {
        integrator = Teuchos::rcp(new CONTACT::CoIntegrator(imortar_, SlaveElement().Shape(),Comm()));
        //Perform integration and linearization
        integrator->IntegrateDerivEle3D(SlaveElement(), MasterElements(),&boundary_ele, &proj, Comm());
      }


      if (IntType() == INPAR::MORTAR::inttype_elements_BS)
      {
        if (boundary_ele == true)
        {
          // switch, if consistent boundary modification chosen
          if (DRT::INPUT::IntegralValue<int>(imortar_, "LM_DUAL_CONSISTENT")
              == true && ShapeFcn() != INPAR::MORTAR::shape_standard // so for petrov-Galerkin and dual
              )
          {
            // loop over all master elements associated with this slave element
            for (int m = 0; m < (int) MasterElements().size(); ++m)
            {
              // create CoCoupling3d object and push back
              Coupling().push_back(Teuchos::rcp(new CoCoupling3d(idiscret_, dim_, false, imortar_,
                  SlaveElement(), MasterElement(m))));

              // do coupling
              Coupling()[m]->EvaluateCoupling();

              // store number of intcells
              ncells_ += (int) (Coupling()[m]->Cells()).size();
            }

            // special treatment of boundary elements
            ConsistDualShape();

            // integrate cells
            for (int m = 0; m < (int) MasterElements().size(); ++m)
              Coupling()[m]->IntegrateCells();

            // free memory of consistent dual shape function coefficient matrix
            SlaveElement().MoData().ResetDualShape();
            SlaveElement().MoData().ResetDerivDualShape();
          }
          else
          {
            // loop over all master elements associated with this slave element
            for (int m = 0; m < (int) MasterElements().size(); ++m)
            {
              // create CoCoupling3d object and push back
              Coupling().push_back(
                  Teuchos::rcp(
                      new CoCoupling3d(idiscret_, dim_, false, imortar_,
                          SlaveElement(), MasterElement(m))));

              // do coupling
              Coupling()[m]->EvaluateCoupling();

              // integrate cells
              Coupling()[m]->IntegrateCells();

              // store number of intcells
              ncells_ += (int) (Coupling()[m]->Cells()).size();
            }
          }
        }
      }
    }
    else
    {
      dserror(
          "You should not be here! This coupling manager is not able to perform mortar coupling for high-order elements.");
    }
  }
  //**********************************************************************
  // INVALID TYPE OF NUMERICAL INTEGRATION
  //**********************************************************************
  else
  {
    dserror("ERROR: Invalid type of numerical integration!");
  }

  // free memory of consistent dual shape function coefficient matrix
  SlaveElement().MoData().ResetDualShape();
  SlaveElement().MoData().ResetDerivDualShape();

  return true;
}

/*----------------------------------------------------------------------*
 |  Evaluate coupling pairs for Quad-coupling                farah 01/13|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3dQuadManager::EvaluateCoupling()
{
  // check
  if (DRT::INPUT::IntegralValue<int>(imortar_, "LM_NODAL_SCALE"))
    dserror("no nodal scaling for quad elements.");

  // decide which type of numerical integration scheme

  //**********************************************************************
  // STANDARD INTEGRATION (SEGMENTS)
  //**********************************************************************
  if (IntType() == INPAR::MORTAR::inttype_segments)
  {
    // loop over all master elements associated with this slave element
    for (int m = 0; m < (int) MasterElements().size(); ++m)
    {
      // build linear integration elements from quadratic MortarElements
      std::vector<Teuchos::RCP<MORTAR::IntElement> > sauxelements(0);
      std::vector<Teuchos::RCP<MORTAR::IntElement> > mauxelements(0);
      SplitIntElements(SlaveElement(), sauxelements);
      SplitIntElements(*MasterElements()[m], mauxelements);

      // loop over all IntElement pairs for coupling
      for (int i = 0; i < (int) sauxelements.size(); ++i)
      {
        for (int j = 0; j < (int) mauxelements.size(); ++j)
        {
          // create instance of coupling class
          CoCoupling3dQuad coup(idiscret_, dim_, true, imortar_, SlaveElement(),
              *MasterElements()[m], *sauxelements[i], *mauxelements[j]);
          // do coupling
          coup.EvaluateCoupling();

          // integrate cells
          coup.IntegrateCells();

          // increase counter of slave/master integration pairs and intcells
          smintpairs_ += 1;
          intcells_ += (int) coup.Cells().size();
        } // for maux
      } // for saux
    } // for m
  }
  //**********************************************************************
  // FAST INTEGRATION (ELEMENTS)
  //**********************************************************************
  else if (IntType() == INPAR::MORTAR::inttype_elements
      || IntType() == INPAR::MORTAR::inttype_elements_BS)
  {
    // check for standard shape functions and quadratic LM interpolation
    if (ShapeFcn() == INPAR::MORTAR::shape_standard
        && LagMultQuad() == INPAR::MORTAR::lagmult_quad_quad
        && (SlaveElement().Shape() == DRT::Element::quad8
            || SlaveElement().Shape() == DRT::Element::tri6))
      dserror(
          "ERROR: Quad. LM interpolation for STANDARD 3D quadratic contact only feasible for quad9");

    if ((int) MasterElements().size() == 0)
      return false;

    // create an integrator instance with correct NumGP and Dim
    CONTACT::CoIntegrator integrator(imortar_, SlaveElement().Shape(), Comm());

    bool boundary_ele = false;
    bool proj = false;

    //Perform integration and linearization
    integrator.IntegrateDerivEle3D(SlaveElement(), MasterElements(),
        &boundary_ele, &proj, Comm());

    if (IntType() == INPAR::MORTAR::inttype_elements_BS)
    {
      if (boundary_ele == true)
      {
        // loop over all master elements associated with this slave element
        for (int m = 0; m < (int) MasterElements().size(); ++m)
        {
          // build linear integration elements from quadratic MortarElements
          std::vector<Teuchos::RCP<MORTAR::IntElement> > sauxelements(0);
          std::vector<Teuchos::RCP<MORTAR::IntElement> > mauxelements(0);
          SplitIntElements(SlaveElement(), sauxelements);
          SplitIntElements(*MasterElements()[m], mauxelements);

          // loop over all IntElement pairs for coupling
          for (int i = 0; i < (int) sauxelements.size(); ++i)
          {
            for (int j = 0; j < (int) mauxelements.size(); ++j)
            {
              // create instance of coupling class
              CoCoupling3dQuad coup(idiscret_, dim_, true, imortar_,
                  SlaveElement(), *MasterElements()[m], *sauxelements[i],
                  *mauxelements[j]);
              // do coupling
              coup.EvaluateCoupling();

              // integrate cells
              coup.IntegrateCells();

              // increase counter of slave/master integration pairs and intcells
              smintpairs_ += 1;
              intcells_ += (int) coup.Cells().size();
            } // for maux
          } // for saux
        } // for m
      }
    }
  }
  //**********************************************************************
  // INVALID
  //**********************************************************************
  else
  {
    dserror("ERROR: Invalid type of numerical integration");
  }

  // free memory of consistent dual shape function coefficient matrix
  SlaveElement().MoData().ResetDualShape();
  SlaveElement().MoData().ResetDerivDualShape();

  return true;
}

/*----------------------------------------------------------------------*
 |  Calculate dual shape functions                           seitz 07/13|
 *----------------------------------------------------------------------*/
void CONTACT::CoCoupling3dManager::ConsistDualShape()
{
  // For standard shape functions no modification is necessary
  // A switch earlier in the process improves computational efficiency
  if (ShapeFcn() == INPAR::MORTAR::shape_standard)
    dserror("ConsistentDualShape() called for standard LM interpolation.");

    // Consistent modification only for linear LM interpolation
    if (Quad()==true && DRT::INPUT::IntegralValue<int>(imortar_,"LM_DUAL_CONSISTENT")==true)
    dserror("Consistent dual shape functions in boundary elements only for linear LM interpolation");

    // you should not be here
    if (DRT::INPUT::IntegralValue<int>(imortar_,"LM_DUAL_CONSISTENT")==false)
    dserror("You should not be here: ConsistDualShape() called but LM_DUAL_CONSISTENT is set NO");

    // do nothing if there are no coupling pairs
    if (Coupling().size()==0)
    return;

    // check for boundary elements in segment-based integration
    // (fast integration already has this check, so that ConsistDualShape()
    // is only called for boundary elements)
    if (IntType()==INPAR::MORTAR::inttype_segments)
    {
      // check, if slave element is fully projecting
      // for convenience, we don't check each quadrature point
      // but only the element nodes. This usually does the job.
      bool boundary_ele=false;

      DRT::Element::DiscretizationType dt_s = SlaveElement().Shape();

      double sxi_test[2] = { 0.0, 0.0};
      double alpha_test=0.0;
      bool proj_test=false;

      DRT::Node** mynodes_test = SlaveElement().Nodes();
      if (!mynodes_test) dserror("ERROR: HasProjStatus: Null pointer!");

      if (dt_s==DRT::Element::quad4 ) //|| dt_s==DRT::Element::quad8 || dt_s==DRT::Element::quad9)
      {
        for (int s_test=0;s_test<4;++s_test)
        {
          if (s_test==0)      { sxi_test[0]=-1.0;sxi_test[1]=-1.0;}
          else if (s_test==1) { sxi_test[0]=-1.0;sxi_test[1]= 1.0;}
          else if (s_test==2) { sxi_test[0]= 1.0;sxi_test[1]=-1.0;}
          else if (s_test==3) { sxi_test[0]= 1.0;sxi_test[1]= 1.0;}

          proj_test=false;
          for (int bs_test=0;bs_test<(int)Coupling().size();++bs_test)
          {
            double mxi_test[2] =
            { 0.0, 0.0};
            MORTAR::MortarProjector::Impl(SlaveElement(),Coupling()[bs_test]->MasterElement())->ProjectGaussPoint3D(SlaveElement(),sxi_test,Coupling()[bs_test]->MasterElement(),mxi_test,alpha_test);

            DRT::Element::DiscretizationType dt = Coupling()[bs_test]->MasterElement().Shape();
            if (dt==DRT::Element::quad4 || dt==DRT::Element::quad8 || dt==DRT::Element::quad9)
            {
              if (mxi_test[0]>=-1.0 && mxi_test[1]>=-1.0 && mxi_test[0]<=1.0 && mxi_test[1]<=1.0)
              proj_test=true;
            }
            else if(dt==DRT::Element::tri3 || dt==DRT::Element::tri6)
            {
              if (mxi_test[0]>=0.0 && mxi_test[1]>=0.0 && mxi_test[0]<=1.0 && mxi_test[1]<=1.0 && mxi_test[0]+mxi_test[1]<=1.0)
              proj_test=true;
            }
            else
            {
              dserror("Non valid element type for master discretization!");
            }
          }
          if(proj_test==false) boundary_ele=true;
        }
      }

      else if(dt_s==DRT::Element::tri3)
      {
        for (int s_test=0;s_test<3;++s_test)
        {
          if      (s_test==0) { sxi_test[0]=0.0;sxi_test[1]=0.0;}
          else if (s_test==1) { sxi_test[0]=1.0;sxi_test[1]=0.0;}
          else if (s_test==2) { sxi_test[0]=0.0;sxi_test[1]=1.0;}

          proj_test=false;
          for (int bs_test=0;bs_test<(int)Coupling().size();++bs_test)
          {
            double mxi_test[2] =
            { 0.0, 0.0};
            MORTAR::MortarProjector::Impl(SlaveElement(),Coupling()[bs_test]->MasterElement())->ProjectGaussPoint3D(SlaveElement(),sxi_test,Coupling()[bs_test]->MasterElement(),mxi_test,alpha_test);

            DRT::Element::DiscretizationType dt = Coupling()[bs_test]->MasterElement().Shape();
            if (dt==DRT::Element::quad4 || dt==DRT::Element::quad8 || dt==DRT::Element::quad9)
            {
              if (mxi_test[0]>=-1.0 && mxi_test[1]>=-1.0 && mxi_test[0]<=1.0 && mxi_test[1]<=1.0)
              proj_test=true;
            }
            else if(dt==DRT::Element::tri3 || dt==DRT::Element::tri6)
            {
              if (mxi_test[0]>=0.0 && mxi_test[1]>=0.0 && mxi_test[0]<=1.0 && mxi_test[1]<=1.0 && mxi_test[0]+mxi_test[1]<=1.0)
              proj_test=true;
            }
            else
            {
              dserror("Non valid element type for master discretization!");
            }
          }
          if(proj_test==false) boundary_ele=true;
        }
      }

      else
      dserror("Calculation of consistent dual shape functions called for non-valid slave element shape.\n"
          "Currently this is only supported for linear FE, i.e. quad4 and tri3. Sorry.");

      if (boundary_ele==false)
      return;
    }

    // slave nodes and dofs
    const int nnodes = SlaveElement().NumNode();
    const int ndof = 3;
    const int msize = MasterElements().size();

    // get number of master nodes
    int mnodes = 0;
    for (int m=0;m<msize;++m)
    mnodes += MasterElements()[m]->NumNode();

    // Dual shape functions coefficient matrix and linearization
    LINALG::SerialDenseMatrix ae(nnodes,nnodes,true);
    std::vector<std::vector<GEN::pairedvector<int, double> > > derivae(nnodes,std::vector<GEN::pairedvector<int,double> >(nnodes,(nnodes+mnodes)*ndof));

    // various variables
    double detg=0.0;
    typedef GEN::pairedvector<int,double>::const_iterator _CI;

    // initialize matrices de and me
    LINALG::SerialDenseMatrix me(nnodes,nnodes,true);
    LINALG::SerialDenseMatrix de(nnodes,nnodes,true);

    // two-dim arrays of maps for linearization of me/de
    std::vector<std::vector<GEN::pairedvector<int,double> > > derivme(nnodes,std::vector<GEN::pairedvector<int,double> >(nnodes,(nnodes+mnodes)*ndof));
    std::vector<std::vector<GEN::pairedvector<int,double> > > derivde(nnodes,std::vector<GEN::pairedvector<int,double> >(nnodes,(nnodes+mnodes)*ndof));

    // loop over all master elements associated with this slave element
    for (int m=0;m<msize;++m)
    {
      // get number of master nodes
      const int ncol = MasterElements()[m]->NumNode();

      // loop over all integration cells
      for (int c=0;c<(int)Coupling()[m]->Cells().size();++c)
      {
        Teuchos::RCP<MORTAR::IntCell> currcell = Coupling()[m]->Cells()[c];

        // create an integrator for this cell
        CONTACT::CoIntegrator integrator(imortar_,currcell->Shape(),Comm());
        for (int gp=0;gp<integrator.nGP(); ++gp)
        {
          // coordinates and weight
          double eta[2] =
          { integrator.Coordinate(gp,0), integrator.Coordinate(gp,1)};
          const double wgt = integrator.Weight(gp);

          // get global Gauss point coordinates
          double globgp[3] = { 0.0, 0.0, 0.0};
          currcell->LocalToGlobal(eta, globgp,0);

          // project Gauss point onto slave integration element
          double sxi[2] = { 0.0, 0.0};
          double sprojalpha = 0.0;
          MORTAR::MortarProjector::Impl(SlaveElement())->ProjectGaussPointAuxn3D(globgp, Coupling()[m]->Auxn(), SlaveElement(), sxi, sprojalpha);

          // create vector for shape function evaluation
          LINALG::SerialDenseVector sval (nnodes);
          LINALG::SerialDenseMatrix sderiv(nnodes,2,true);

          // evaluate trace space shape functions at Gauss point
          SlaveElement().EvaluateShape(sxi, sval, sderiv, nnodes);
          detg=currcell->Jacobian(eta);

          // additional data for contact calculation (i.e. incl. derivative of dual shape functions coefficient matrix)
          // directional derivative of cell Jacobian
          GEN::pairedvector<int,double> derivjaccell((nnodes+ncol)*ndof);
          // GP slave coordinate derivatives
          std::vector<GEN::pairedvector<int,double> > dsxigp(2,(nnodes+ncol)*ndof);
          // global GP coordinate derivative
          std::vector<GEN::pairedvector<int,double> > lingp(3,(nnodes+ncol)*ndof);

          // compute directional derivative of cell Jacobian
          currcell->DerivJacobian(eta, derivjaccell);

          // compute global GP coordinate derivative
          static LINALG::Matrix<3,1> svalcell;
          static LINALG::Matrix<3,2> sderivcell;
          currcell->EvaluateShape(eta,svalcell,sderivcell);

          for (int v=0;v<3;++v)
          {
            for (_CI p=(currcell->GetDerivVertex(v))[0].begin();p!=(currcell->GetDerivVertex(v))[0].end();++p)
              lingp[0][p->first] += svalcell(v) * (p->second);
            for (_CI p=(currcell->GetDerivVertex(v))[1].begin();p!=(currcell->GetDerivVertex(v))[1].end();++p)
              lingp[1][p->first] += svalcell(v) * (p->second);
            for (_CI p=(currcell->GetDerivVertex(v))[2].begin();p!=(currcell->GetDerivVertex(v))[2].end();++p)
              lingp[2][p->first] += svalcell(v) * (p->second);
          }

          // compute GP slave coordinate derivatives
          integrator.DerivXiGP3DAuxPlane(SlaveElement(),sxi,currcell->Auxn(),dsxigp,sprojalpha,currcell->GetDerivAuxn(),lingp);

          // computing de, derivde and me, derivme and kappa, derivkappa
          for (int j=0; j<nnodes; ++j)
          {
            double fac;
            fac = sval[j]*wgt;
            // computing de
            de(j,j)+=fac*detg;

            // linearization of de
            // linearization of cell jacobian
            for (_CI p=derivjaccell.begin();p!=derivjaccell.end();++p)
            derivde[j][j][p->first] += fac*(p->second);

            // linearization of slave gp coordinates in ansatz function j for derivate of de
            fac=wgt*sderiv(j,0)*detg;
            for (_CI p=dsxigp[0].begin(); p!=dsxigp[0].end(); ++p)
              derivde[j][j][p->first] += fac * (p->second);
            fac=wgt*sderiv(j,1)*detg;
            for (_CI p=dsxigp[1].begin(); p!=dsxigp[1].end(); ++p)
              derivde[j][j][p->first] += fac*(p->second);

            for (int k=0; k<nnodes; ++k)
            {
              // computing me
              fac = wgt*sval[j]*sval[k];
              me(j,k)+=fac*detg;

              // linearization of me
              // linearization of cell Jacobian
              for (_CI p=derivjaccell.begin(); p!=derivjaccell.end(); ++p)
              derivme[j][k][p->first] += fac*(p->second);

              // linearizaion of gp coordinates in ansatz function
              fac=wgt*sderiv(j,0)*sval[k]*detg;
              for (_CI p=dsxigp[0].begin(); p!=dsxigp[0].end(); ++p)
              {
                derivme[j][k][p->first] += fac*(p->second);
                derivme[k][j][p->first] += fac*(p->second);
              }

              fac=wgt*sderiv(j,1)*sval[k]*detg;
              for (_CI p=dsxigp[1].begin(); p!=dsxigp[1].end(); ++p)
              {
                derivme[j][k][p->first] += fac*(p->second);
                derivme[k][j][p->first] += fac*(p->second);
              }
            }
          }
        }
      } // cells
    } // master elements

    // in case of no overlap just return, as there is no integration area
    // and therefore the consistent dual shape functions are not defined.
    // This doesn't matter, as there is no associated integration domain anyway
    if (me.Det_long()==0) return;

    // invert bi-ortho matrix me
    LINALG::SerialDenseMatrix meinv = LINALG::InvertAndMultiplyByCholesky(me,de,ae);

    // build linearization of ae and store in derivdual
    // (this is done according to a quite complex formula, which
    // we get from the linearization of the biorthogonality condition:
    // Lin (Me * Ae = De) -> Lin(Ae)=Lin(De)*Inv(Me)-Ae*Lin(Me)*Inv(Me) )

    // loop over all entries of ae (index i,j)
    for (int i=0;i<nnodes;++i)
    {
      for (int j=0;j<nnodes;++j)
      {
        // compute Lin(Ae) according to formula above
        for (int l=0;l<nnodes;++l)// loop over sum l
        {
          // part1: Lin(De)*Inv(Me)
          for (_CI p=derivde[i][l].begin();p!=derivde[i][l].end();++p)
            derivae[i][j][p->first] += meinv(l,j)*(p->second);

          // part2: Ae*Lin(Me)*Inv(Me)
          for (int k=0;k<nnodes;++k)// loop over sum k
            for (_CI p=derivme[k][l].begin();p!=derivme[k][l].end();++p)
              derivae[i][j][p->first] -= ae(i,k)*meinv(l,j)*(p->second);
        }
      }
    }

    // store ae matrix in slave element data container
    SlaveElement().MoData().DualShape() = Teuchos::rcp(new LINALG::SerialDenseMatrix(ae));
//
    // store derivae into element
    SlaveElement().MoData().DerivDualShape() = Teuchos::rcp(new std::vector<std::vector<GEN::pairedvector<int,double> > >(derivae));

    return;
  }

