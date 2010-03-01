/*!----------------------------------------------------------------------
\file contact_coupling3d.cpp
\brief A class for mortar coupling of ONE slave element and ONE master
       element of a contact interface in 3D.

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
            089 - 289-15264
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "contact_coupling3d.H"
#include "contact_integrator.H"
#include "contact_node.H"
#include "../drt_mortar/mortar_defines.H"
#include "contact_defines.H"

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 11/08|
 *----------------------------------------------------------------------*/
CONTACT::CoCoupling3d::CoCoupling3d(DRT::Discretization& idiscret, int dim, bool quad,
              bool auxplane, MORTAR::MortarElement& sele, MORTAR::MortarElement& mele) :
MORTAR::Coupling3d(idiscret,dim,quad,auxplane,sele,mele)
{
  // empty constructor
  
  return;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 06/09|
 *----------------------------------------------------------------------*/
CONTACT::CoCoupling3d::CoCoupling3d(const INPAR::MORTAR::ShapeFcn shapefcn,
                               DRT::Discretization& idiscret, int dim, bool quad,
         bool auxplane, MORTAR::MortarElement& sele, MORTAR::MortarElement& mele) :
MORTAR::Coupling3d(shapefcn,idiscret,dim,quad,auxplane,sele,mele)
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
  if (dt==MORTAR::MortarElement::tri3 ||
      dt==MORTAR::MortarElement::tri6)
  {
    loccenter[0] = 1.0/3;
    loccenter[1] = 1.0/3;
  }
  else if (dt==MORTAR::MortarElement::quad4 ||
           dt==MORTAR::MortarElement::quad8 ||
           dt==MORTAR::MortarElement::quad9)
  {
    loccenter[0] = 0.0;
    loccenter[1] = 0.0;
  }
  else dserror("ERROR: AuxiliaryPlane called for unknown element type");

  // compute element center via shape fct. interpolation
  SlaveIntElement().LocalToGlobal(loccenter,Auxc(),0);

  // we then compute the unit normal vector at the element center
  Lauxn() = SlaveIntElement().ComputeUnitNormalAtXi(loccenter,Auxn());

  // THIS IS CONTACT-SPECIFIC!
  // also compute linearization of the unit normal vector
  SlaveIntElement().DerivUnitNormalAtXi(loccenter,GetDerivAuxn());
  
  //cout << "Slave Element: " << SlaveIntElement().Id() << endl;
  //cout << "->Center: " << Auxc()[0] << " " << Auxc()[1] << " " << Auxc()[2] << endl;
  //cout << "->Normal: " << Auxn()[0] << " " << Auxn()[1] << " " << Auxn()[2] << endl;

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

  // create a CONTACT integrator instance with correct NumGP and Dim
  // it is sufficient to do this once as all Intcells are triangles
  CONTACT::CoIntegrator integrator(shapefcn_,Cells()[0]->Shape());

  // loop over all integration cells
  for (int i=0;i<(int)(Cells().size());++i)
  {
    // compare intcell area with slave integration element area
    double intcellarea = Cells()[i]->Area();
    double selearea = 0.0;
    if (!CouplingInAuxPlane())
      selearea = SlaveIntElement().Area();
    else
    {
      DRT::Element::DiscretizationType dt = SlaveIntElement().Shape();
      if (dt==DRT::Element::quad4 || dt==DRT::Element::quad8 || dt==DRT::Element::quad9)
        selearea = 4.0;
      else if (dt==DRT::Element::tri3 || dt==DRT::Element::tri6)
        selearea = 0.5;
      else dserror("ERROR: IntegrateCells: Invalid 3D slave element type");
    }

    // integrate cell only if not neglectable
    if (intcellarea < MORTARINTLIM*selearea) continue;

    // *******************************************************************
    // different options for mortar integration
    // *******************************************************************
    // (1) no quadratic element(s) involved -> linear LM interpolation
    // (2) quadratic element(s) involved -> quadratic LM interpolation
    // (3) quadratic element(s) involved -> piecew. linear LM interpolation
    // (4) quadratic element(s) involved -> linear LM interpolation
    // *******************************************************************
    INPAR::MORTAR::LagMultQuad3D lmtype = LagMultQuad3D();
    
    // *******************************************************************
    // case (1)
    // *******************************************************************
    if (!Quad())
    {
      // prepare integration and linearization of M, g (and possibly D) on intcells
      int nrow = SlaveElement().NumNode();
      int ncol = MasterElement().NumNode();
      RCP<Epetra_SerialDenseMatrix> dseg = rcp(new Epetra_SerialDenseMatrix(nrow*Dim(),nrow*Dim()));
      RCP<Epetra_SerialDenseMatrix> mseg = rcp(new Epetra_SerialDenseMatrix(nrow*Dim(),ncol*Dim()));
      RCP<Epetra_SerialDenseVector> gseg = rcp(new Epetra_SerialDenseVector(nrow));
  
      if (CouplingInAuxPlane())
        integrator.IntegrateDerivCell3DAuxPlane(SlaveElement(),MasterElement(),Cells()[i],Auxn(),dseg,mseg,gseg);
      else /*(!CouplingInAuxPlane()*/
        integrator.IntegrateDerivCell3D(SlaveElement(),MasterElement(),Cells()[i],dseg,mseg,gseg);
  
      // do the assembly into the slave nodes
#ifdef MORTARONELOOP
      integrator.AssembleD(Comm(),SlaveElement(),*dseg);
#endif // #ifdef MORTARONELOOP
      integrator.AssembleM(Comm(),SlaveElement(),MasterElement(),*mseg);
      integrator.AssembleG(Comm(),SlaveElement(),*gseg);
    }
    
    // *******************************************************************
    // case (2)
    // *******************************************************************
    else if (Quad() && lmtype==INPAR::MORTAR::lagmult_quad_quad)
    {
      // check whether this is feasible (ONLY for quad9 surfaces)
      if (SlaveElement().Shape()==DRT::Element::quad8 || SlaveElement().Shape()==DRT::Element::tri6)
        dserror("ERROR: Quadratic/Quadratic LM interpolation for 3D quadratic contact only feasible for quad9-surfaces");
      
      // prepare integration and linearization of M, g (and possibly D) on intcells
      int nrow = SlaveElement().NumNode();
      int ncol = MasterElement().NumNode();
      RCP<Epetra_SerialDenseMatrix> dseg = rcp(new Epetra_SerialDenseMatrix(nrow*Dim(),nrow*Dim()));
      RCP<Epetra_SerialDenseMatrix> mseg = rcp(new Epetra_SerialDenseMatrix(nrow*Dim(),ncol*Dim()));
      RCP<Epetra_SerialDenseVector> gseg = rcp(new Epetra_SerialDenseVector(nrow));
      
      // static_cast to make sure to pass in IntElement&
      MORTAR::IntElement& sintref = static_cast<MORTAR::IntElement&>(SlaveIntElement());
      MORTAR::IntElement& mintref = static_cast<MORTAR::IntElement&>(MasterIntElement());
      
      // check whether aux. plane coupling or not
      if (CouplingInAuxPlane())
        integrator.IntegrateDerivCell3DAuxPlaneQuad(SlaveElement(),MasterElement(),sintref,mintref,
            Cells()[i],Auxn(),lmtype,dseg,mseg,gseg);
      else /*(!CouplingInAuxPlane()*/
        dserror("ERROR: Only aux. plane version implemented for 3D quadratic contact");
      
      // do the assembly into the slave nodes
#ifdef MORTARONELOOP
      integrator.AssembleD(Comm(),SlaveElement(),*dseg);
#endif // #ifdef MORTARONELOOP
      integrator.AssembleM(Comm(),SlaveElement(),MasterElement(),*mseg);
      integrator.AssembleG(Comm(),SlaveElement(),*gseg);
    }
    
    // *******************************************************************
    // case (3)
    // *******************************************************************
    else if (Quad() && lmtype==INPAR::MORTAR::lagmult_pwlin_pwlin)
    {
      // prepare integration and linearization of M, g (and possibly D) on intcells
      int nrow = SlaveElement().NumNode();
      int ncol = MasterElement().NumNode();
      int nintrow = SlaveIntElement().NumNode();
      RCP<Epetra_SerialDenseMatrix> dseg = rcp(new Epetra_SerialDenseMatrix(nintrow*Dim(),nrow*Dim()));
      RCP<Epetra_SerialDenseMatrix> mseg = rcp(new Epetra_SerialDenseMatrix(nintrow*Dim(),ncol*Dim()));
      RCP<Epetra_SerialDenseVector> gseg = rcp(new Epetra_SerialDenseVector(nintrow));
      
      // static_cast to make sure to pass in IntElement&
      MORTAR::IntElement& sintref = static_cast<MORTAR::IntElement&>(SlaveIntElement());
      MORTAR::IntElement& mintref = static_cast<MORTAR::IntElement&>(MasterIntElement());
      
      // check whether aux. plane coupling or not
      if (CouplingInAuxPlane())
        integrator.IntegrateDerivCell3DAuxPlaneQuad(SlaveElement(),MasterElement(),sintref,mintref,
            Cells()[i],Auxn(),lmtype,dseg,mseg,gseg);
      else /*(!CouplingInAuxPlane()*/
        dserror("ERROR: Only aux. plane version implemented for 3D quadratic contact");
      
      // do the assembly into the slave nodes
      // (NOTE THAT THESE ARE SPECIAL VERSIONS HERE FOR PIECEWISE LINEAR INTERPOLATION)
#ifdef MORTARONELOOP
      integrator.AssembleD(Comm(),SlaveElement(),sintref,*dseg);
#endif // #ifdef MORTARONELOOP
      integrator.AssembleM(Comm(),sintref,MasterElement(),*mseg);
      integrator.AssembleG(Comm(),sintref,*gseg);
    }
    
    // *******************************************************************
    // case (4)
    // *******************************************************************
    else if (Quad() && lmtype==INPAR::MORTAR::lagmult_lin_lin)
    {
      dserror("ERROR: lin contact not yet implemented");
    }
    
    // *******************************************************************
    // other cases
    // *******************************************************************
    else
    {
      dserror("ERROR: IntegrateCells: Invalid case for 3D mortar contact LM interpolation");
    }
    // *******************************************************************
  }
  
  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of clip polygon vertices (3D)               popp 02/09|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3d::VertexLinearization(vector<vector<map<int,double> > >& linvertex,
                                                map<int,double>& projpar, bool printderiv)
{
  typedef map<int,double>::const_iterator CI;

  // linearize all aux.plane slave and master nodes only ONCE
  // and use these linearizations later during lineclip linearization
  // (this speeds up the vertex linearizations in most cases, as we
  // never linearize the SAME slave or master vertex more than once)

  // number of nodes
  int nsrows = SlaveIntElement().NumNode();
  int nmrows = MasterIntElement().NumNode();

  // prepare storage for slave and master linearizations
  vector<vector<map<int,double> > > linsnodes(nsrows,vector<map<int,double> >(3));
  vector<vector<map<int,double> > > linmnodes(nmrows,vector<map<int,double> >(3));

  if (CouplingInAuxPlane())
  {
    // compute slave linearizations (nsrows)
    for (int i=0;i<nsrows;++i)
    {
      int sid = SlaveIntElement().NodeIds()[i];
      SlaveVertexLinearization(linsnodes[i],sid);
    }

    // compute master linearizations (nmrows)
    for (int i=0;i<nmrows;++i)
    {
      int mid = MasterIntElement().NodeIds()[i];
      MasterVertexLinearization(linmnodes[i],mid);
    }
  }

  //**********************************************************************
  // Clip polygon vertex linearization
  //**********************************************************************
  // loop over all clip polygon vertices
  for (int i=0;i<(int)Clip().size();++i)
  {
    // references to current vertex and its linearization
    MORTAR::Vertex& currv = Clip()[i];
    vector<map<int,double> >& currlin = linvertex[i];

    // decision on vertex type (slave, projmaster, linclip)
    if (currv.VType()==MORTAR::Vertex::slave)
    {
      if (CouplingInAuxPlane())
      {
        // get corresponding slave id
        int sid = currv.Nodeids()[0];

        // find corresponding slave node linearization
        int k=0;
        while (k<nsrows){
          if (SlaveIntElement().NodeIds()[k]==sid) break;
          ++k;
        }

        // dserror if not found
        if (k==nsrows) dserror("ERROR: Slave Id not found!");

        // get the correct slave node linearization
        currlin = linsnodes[k];
      }
      else //(!CouplingInAuxPlane())
      {
        // Vertex = slave node -> Linearization = 0
        // this is the easy case with nothing to do
      }
    }
    else if (currv.VType()==MORTAR::Vertex::projmaster)
    {
      if (CouplingInAuxPlane())
      {
        // get corresponding master id
        int mid = currv.Nodeids()[0];

        // find corresponding master node linearization
        int k=0;
        while (k<nmrows){
          if (MasterIntElement().NodeIds()[k]==mid) break;
          ++k;
        }

        // dserror if not found
        if (k==nmrows) dserror("ERROR: Master Id not found!");

        // get the correct master node linearization
        currlin = linmnodes[k];
      }
      else //(!CouplingInAuxPlane())
      {
        // get corresponding master id and projection alpha
        int mid = currv.Nodeids()[0];
        double alpha = projpar[mid];

        //cout << "Coords: " << currv.Coord()[0] << " " << currv.Coord()[1] << endl;

        // do master vertex linearization
        MasterVertexLinearization(currv,currlin,mid,alpha);
      }
    }
    else if (currv.VType()==MORTAR::Vertex::lineclip)
    {
      // get references to the two slave vertices
      int sindex1 = -1;
      int sindex2 = -1;
      for (int j=0;j<(int)SlaveVertices().size();++j)
      {
        if (SlaveVertices()[j].Nodeids()[0]==currv.Nodeids()[0])
          sindex1 = j;
        if (SlaveVertices()[j].Nodeids()[0]==currv.Nodeids()[1])
          sindex2 = j;
      }
      if (sindex1 < 0 || sindex2 < 0 || sindex1==sindex2)
        dserror("ERROR: Lineclip linearization: (S) Something went wrong!");

      MORTAR::Vertex* sv1 = &SlaveVertices()[sindex1];
      MORTAR::Vertex* sv2 = &SlaveVertices()[sindex2];

      // get references to the two master vertices
      int mindex1 = -1;
      int mindex2 = -1;
      for (int j=0;j<(int)MasterVertices().size();++j)
      {
        if (MasterVertices()[j].Nodeids()[0]==currv.Nodeids()[2])
          mindex1 = j;
        if (MasterVertices()[j].Nodeids()[0]==currv.Nodeids()[3])
          mindex2 = j;
      }
      if (mindex1 < 0 || mindex2 < 0 || mindex1==mindex2)
        dserror("ERROR: Lineclip linearization: (M) Something went wrong!");

      MORTAR::Vertex* mv1 = &MasterVertices()[mindex1];
      MORTAR::Vertex* mv2 = &MasterVertices()[mindex2];

      // do lineclip vertex linearization
      if (CouplingInAuxPlane())
        LineclipVertexLinearization(currv,currlin,sv1,sv2,mv1,mv2,linsnodes,linmnodes);
      else
        LineclipVertexLinearization(currv,currlin,sv1,sv2,mv1,mv2,projpar);
    }

    else dserror("ERROR: VertexLinearization: Invalid Vertex Type!");
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of slave vertex (3D) AuxPlane               popp 03/09|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3d::SlaveVertexLinearization(vector<map<int,double> >& currlin,
                                                     int sid)
{
  // we first need the slave element center:
  // for quad4, quad8, quad9 elements: xi = eta = 0.0
  // for tri3, tri6 elements: xi = eta = 1/3
  double scxi[2];

  DRT::Element::DiscretizationType dt = SlaveIntElement().Shape();
  if (dt==MORTAR::MortarElement::tri3 ||
      dt==MORTAR::MortarElement::tri6)
  {
    scxi[0] = 1.0/3;
    scxi[1] = 1.0/3;
  }
  else if (dt==MORTAR::MortarElement::quad4 ||
           dt==MORTAR::MortarElement::quad8 ||
           dt==MORTAR::MortarElement::quad9)
  {
    scxi[0] = 0.0;
    scxi[1] = 0.0;
  }
  else dserror("ERROR: SlaveVertexLinearization called for unknown element type");

  // evlauate shape functions + derivatives at scxi
  int nrow = SlaveIntElement().NumNode();
  LINALG::SerialDenseVector sval(nrow);
  LINALG::SerialDenseMatrix sderiv(nrow,2,true);
  SlaveIntElement().EvaluateShape(scxi,sval,sderiv,nrow);

  // we need all participating slave nodes
  DRT::Node** snodes = SlaveIntElement().Nodes();
  vector<MORTAR::MortarNode*> smrtrnodes(nrow);

  for (int i=0;i<nrow;++i)
  {
    smrtrnodes[i] = static_cast<MORTAR::MortarNode*>(snodes[i]);
    if (!smrtrnodes[i]) dserror("ERROR: SlaveVertexLinearization: Null pointer!");
  }

  // we also need the corresponding slave node
  DRT::Node* snode = Discret().gNode(sid);
  if (!snode) dserror("ERROR: Cannot find node with gid %",sid);
  MORTAR::MortarNode* mrtrsnode = static_cast<MORTAR::MortarNode*>(snode);

  // map iterator
  typedef map<int,double>::const_iterator CI;

  // linearization of element center Auxc()
  vector<map<int,double> > linauxc(3);

  for (int i=0;i<nrow;++i)
  {
    linauxc[0][smrtrnodes[i]->Dofs()[0]] += sval[i];
    linauxc[1][smrtrnodes[i]->Dofs()[1]] += sval[i];
    linauxc[2][smrtrnodes[i]->Dofs()[2]] += sval[i];
  }

  // linearization of element normal Auxn()
  vector<map<int,double> >& linauxn = GetDerivAuxn();

  // put everything together for slave vertex linearization

  // (1) slave node coordinates part
  currlin[0][mrtrsnode->Dofs()[0]] += 1.0 - Auxn()[0] * Auxn()[0];
  currlin[0][mrtrsnode->Dofs()[1]] -=       Auxn()[1] * Auxn()[0];
  currlin[0][mrtrsnode->Dofs()[2]] -=       Auxn()[2] * Auxn()[0];
  currlin[1][mrtrsnode->Dofs()[0]] -=       Auxn()[0] * Auxn()[1];
  currlin[1][mrtrsnode->Dofs()[1]] += 1.0 - Auxn()[1] * Auxn()[1];
  currlin[1][mrtrsnode->Dofs()[2]] -=       Auxn()[2] * Auxn()[1];
  currlin[2][mrtrsnode->Dofs()[0]] -=       Auxn()[0] * Auxn()[2];
  currlin[2][mrtrsnode->Dofs()[1]] -=       Auxn()[1] * Auxn()[2];
  currlin[2][mrtrsnode->Dofs()[2]] += 1.0 - Auxn()[2] * Auxn()[2];

  // (2) slave element center coordinates (Auxc()) part
  for (CI p=linauxc[0].begin();p!=linauxc[0].end();++p)
    for (int k=0;k<3;++k)
      currlin[k][p->first] += Auxn()[0] * Auxn()[k] * (p->second);

  for (CI p=linauxc[1].begin();p!=linauxc[1].end();++p)
    for (int k=0;k<3;++k)
      currlin[k][p->first] += Auxn()[1] * Auxn()[k] * (p->second);

  for (CI p=linauxc[2].begin();p!=linauxc[2].end();++p)
    for (int k=0;k<3;++k)
      currlin[k][p->first] += Auxn()[2] * Auxn()[k] * (p->second);

  // (3) slave element normal (Auxn()) part
  double xdotn = (mrtrsnode->xspatial()[0]-Auxc()[0]) * Auxn()[0]
               + (mrtrsnode->xspatial()[1]-Auxc()[1]) * Auxn()[1]
               + (mrtrsnode->xspatial()[2]-Auxc()[2]) * Auxn()[2];

  for (CI p=linauxn[0].begin();p!=linauxn[0].end();++p)
  {
    currlin[0][p->first] -= xdotn * (p->second);
    for (int k=0;k<3;++k)
      currlin[k][p->first] -= (mrtrsnode->xspatial()[0]-Auxc()[0]) * Auxn()[k] * (p->second);
  }

  for (CI p=linauxn[1].begin();p!=linauxn[1].end();++p)
  {
    currlin[1][p->first] -= xdotn * (p->second);
    for (int k=0;k<3;++k)
      currlin[k][p->first] -= (mrtrsnode->xspatial()[1]-Auxc()[1]) * Auxn()[k] * (p->second);
  }

  for (CI p=linauxn[2].begin();p!=linauxn[2].end();++p)
  {
    currlin[2][p->first] -= xdotn * (p->second);
    for (int k=0;k<3;++k)
      currlin[k][p->first] -= (mrtrsnode->xspatial()[2]-Auxc()[2]) * Auxn()[k] * (p->second);
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of projmaster vertex (3D) AuxPlane          popp 03/09|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3d::MasterVertexLinearization(vector<map<int,double> >& currlin,
                                                    int mid)
{
  // we first need the slave element center:
  // for quad4, quad8, quad9 elements: xi = eta = 0.0
  // for tri3, tri6 elements: xi = eta = 1/3
  double scxi[2];

  DRT::Element::DiscretizationType dt = SlaveIntElement().Shape();
  if (dt==MORTAR::MortarElement::tri3 ||
      dt==MORTAR::MortarElement::tri6)
  {
    scxi[0] = 1.0/3;
    scxi[1] = 1.0/3;
  }
  else if (dt==MORTAR::MortarElement::quad4 ||
           dt==MORTAR::MortarElement::quad8 ||
           dt==MORTAR::MortarElement::quad9)
  {
    scxi[0] = 0.0;
    scxi[1] = 0.0;
  }
  else dserror("ERROR: MasterVertexLinearization called for unknown element type");

  // evlauate shape functions + derivatives at scxi
  int nrow = SlaveIntElement().NumNode();
  LINALG::SerialDenseVector sval(nrow);
  LINALG::SerialDenseMatrix sderiv(nrow,2,true);
  SlaveIntElement().EvaluateShape(scxi,sval,sderiv,nrow);

  // we need all participating slave nodes
  DRT::Node** snodes = SlaveIntElement().Nodes();
  vector<MORTAR::MortarNode*> smrtrnodes(nrow);

  for (int i=0;i<nrow;++i)
  {
    smrtrnodes[i] = static_cast<MORTAR::MortarNode*>(snodes[i]);
    if (!smrtrnodes[i]) dserror("ERROR: MasterVertexLinearization: Null pointer!");
  }

  // we also need the corresponding master node
  DRT::Node* mnode = Discret().gNode(mid);
  if (!mnode) dserror("ERROR: Cannot find node with gid %",mid);
  MORTAR::MortarNode* mrtrmnode = static_cast<MORTAR::MortarNode*>(mnode);

  // map iterator
  typedef map<int,double>::const_iterator CI;

  // linearization of element center Auxc()
  vector<map<int,double> > linauxc(3);

  for (int i=0;i<nrow;++i)
  {
    linauxc[0][smrtrnodes[i]->Dofs()[0]] += sval[i];
    linauxc[1][smrtrnodes[i]->Dofs()[1]] += sval[i];
    linauxc[2][smrtrnodes[i]->Dofs()[2]] += sval[i];
  }

  // linearization of element normal Auxn()
  vector<map<int,double> >& linauxn = GetDerivAuxn();

  // put everything together for master vertex linearization

  // (1) master node coordinates part
  currlin[0][mrtrmnode->Dofs()[0]] += 1.0 - Auxn()[0] * Auxn()[0];
  currlin[0][mrtrmnode->Dofs()[1]] -=       Auxn()[1] * Auxn()[0];
  currlin[0][mrtrmnode->Dofs()[2]] -=       Auxn()[2] * Auxn()[0];
  currlin[1][mrtrmnode->Dofs()[0]] -=       Auxn()[0] * Auxn()[1];
  currlin[1][mrtrmnode->Dofs()[1]] += 1.0 - Auxn()[1] * Auxn()[1];
  currlin[1][mrtrmnode->Dofs()[2]] -=       Auxn()[2] * Auxn()[1];
  currlin[2][mrtrmnode->Dofs()[0]] -=       Auxn()[0] * Auxn()[2];
  currlin[2][mrtrmnode->Dofs()[1]] -=       Auxn()[1] * Auxn()[2];
  currlin[2][mrtrmnode->Dofs()[2]] += 1.0 - Auxn()[2] * Auxn()[2];

  // (2) slave element center coordinates (Auxc()) part
  for (CI p=linauxc[0].begin();p!=linauxc[0].end();++p)
    for (int k=0;k<3;++k)
      currlin[k][p->first] += Auxn()[0] * Auxn()[k] * (p->second);

  for (CI p=linauxc[1].begin();p!=linauxc[1].end();++p)
    for (int k=0;k<3;++k)
      currlin[k][p->first] += Auxn()[1] * Auxn()[k] * (p->second);

  for (CI p=linauxc[2].begin();p!=linauxc[2].end();++p)
    for (int k=0;k<3;++k)
      currlin[k][p->first] += Auxn()[2] * Auxn()[k] * (p->second);

  // (3) slave element normal (Auxn()) part
  double xdotn = (mrtrmnode->xspatial()[0]-Auxc()[0]) * Auxn()[0]
               + (mrtrmnode->xspatial()[1]-Auxc()[1]) * Auxn()[1]
               + (mrtrmnode->xspatial()[2]-Auxc()[2]) * Auxn()[2];

  for (CI p=linauxn[0].begin();p!=linauxn[0].end();++p)
  {
    currlin[0][p->first] -= xdotn * (p->second);
    for (int k=0;k<3;++k)
      currlin[k][p->first] -= (mrtrmnode->xspatial()[0]-Auxc()[0]) * Auxn()[k] * (p->second);
  }

  for (CI p=linauxn[1].begin();p!=linauxn[1].end();++p)
  {
    currlin[1][p->first] -= xdotn * (p->second);
    for (int k=0;k<3;++k)
      currlin[k][p->first] -= (mrtrmnode->xspatial()[1]-Auxc()[1]) * Auxn()[k] * (p->second);
  }

  for (CI p=linauxn[2].begin();p!=linauxn[2].end();++p)
  {
    currlin[2][p->first] -= xdotn * (p->second);
    for (int k=0;k<3;++k)
      currlin[k][p->first] -= (mrtrmnode->xspatial()[2]-Auxc()[2]) * Auxn()[k] * (p->second);
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of projmaster vertex (3D)                   popp 02/09|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3d::MasterVertexLinearization(MORTAR::Vertex& currv,
                                                      vector<map<int,double> >& currlin,
                                                      int mid, double alpha)
{
  // get current vertex coordinates (in slave param. space)
  double sxi[2] = {0.0, 0.0};
  sxi[0] = currv.Coord()[0];
  sxi[1] = currv.Coord()[1];
  
  // evlauate shape functions + derivatives at sxi
  int nrow = SlaveIntElement().NumNode();
  LINALG::SerialDenseVector sval(nrow);
  LINALG::SerialDenseMatrix sderiv(nrow,2,true);
  SlaveIntElement().EvaluateShape(sxi,sval,sderiv,nrow);
  
  // build 3x3 factor matrix L
  LINALG::Matrix<3,3> lmatrix(true);
  
  for (int z=0;z<nrow;++z)
  {
    int gid = SlaveIntElement().NodeIds()[z];
    DRT::Node* node = Discret().gNode(gid);
    if (!node) dserror("ERROR: Cannot find node with gid %",gid);
    MORTAR::MortarNode* snode = static_cast<MORTAR::MortarNode*>(node);
  
    lmatrix(0,0) += sderiv(z,0) * snode->xspatial()[0];
    lmatrix(1,0) += sderiv(z,0) * snode->xspatial()[1];
    lmatrix(2,0) += sderiv(z,0) * snode->xspatial()[2];
  
    lmatrix(0,0) += alpha * sderiv(z,0) * snode->MoData().n()[0];
    lmatrix(1,0) += alpha * sderiv(z,0) * snode->MoData().n()[1];
    lmatrix(2,0) += alpha * sderiv(z,0) * snode->MoData().n()[2];
  
    lmatrix(0,1) += sderiv(z,1) * snode->xspatial()[0];
    lmatrix(1,1) += sderiv(z,1) * snode->xspatial()[1];
    lmatrix(2,1) += sderiv(z,1) * snode->xspatial()[2];
  
    lmatrix(0,1) += alpha * sderiv(z,1) * snode->MoData().n()[0];
    lmatrix(1,1) += alpha * sderiv(z,1) * snode->MoData().n()[1];
    lmatrix(2,1) += alpha * sderiv(z,1) * snode->MoData().n()[2];
  
    lmatrix(0,2) += sval[z] * snode->MoData().n()[0];
    lmatrix(1,2) += sval[z] * snode->MoData().n()[1];
    lmatrix(2,2) += sval[z] * snode->MoData().n()[2];
  }
  
  // get inverse of the 3x3 matrix L (in place)
  lmatrix.Invert();
  
  // start to fill linearization maps for current vertex
  typedef map<int,double>::const_iterator CI;
  
  // (1) master node coordinates part
  DRT::Node* mnode = Discret().gNode(mid);
  if (!mnode) dserror("ERROR: Cannot find node with gid %",mid);
  MORTAR::MortarNode* mrtrmnode = static_cast<MORTAR::MortarNode*>(mnode);
  
  currlin[0][mrtrmnode->Dofs()[0]] += lmatrix(0,0);
  currlin[0][mrtrmnode->Dofs()[1]] += lmatrix(0,1);
  currlin[0][mrtrmnode->Dofs()[2]] += lmatrix(0,2);
  currlin[1][mrtrmnode->Dofs()[0]] += lmatrix(1,0);
  currlin[1][mrtrmnode->Dofs()[1]] += lmatrix(1,1);
  currlin[1][mrtrmnode->Dofs()[2]] += lmatrix(1,2);
  
  // (2) all slave nodes coordinates part
  for (int z=0;z<nrow;++z)
  {
    int gid = SlaveIntElement().NodeIds()[z];
    DRT::Node* node = Discret().gNode(gid);
    if (!node) dserror("ERROR: Cannot find node with gid %",gid);
    MORTAR::MortarNode* snode = static_cast<MORTAR::MortarNode*>(node);
  
    currlin[0][snode->Dofs()[0]] -= sval[z] * lmatrix(0,0);
    currlin[0][snode->Dofs()[1]] -= sval[z] * lmatrix(0,1);
    currlin[0][snode->Dofs()[2]] -= sval[z] * lmatrix(0,2);
    currlin[1][snode->Dofs()[0]] -= sval[z] * lmatrix(1,0);
    currlin[1][snode->Dofs()[1]] -= sval[z] * lmatrix(1,1);
    currlin[1][snode->Dofs()[2]] -= sval[z] * lmatrix(1,2);
  
    // (3) all slave nodes normals part
    // get nodal normal derivative maps (x,y and z components)
    vector<map<int,double> >& derivn = static_cast<CONTACT::CoNode*>(snode)->CoData().GetDerivN();
  
    for (CI p=derivn[0].begin();p!=derivn[0].end();++p)
    {
      currlin[0][p->first] -= alpha * sval[z] * lmatrix(0,0) * (p->second);
      currlin[1][p->first] -= alpha * sval[z] * lmatrix(1,0) * (p->second);
    }
    for (CI p=derivn[1].begin();p!=derivn[1].end();++p)
    {
      currlin[0][p->first] -= alpha * sval[z] * lmatrix(0,1) * (p->second);
      currlin[1][p->first] -= alpha * sval[z] * lmatrix(1,1) * (p->second);
    }
    for (CI p=derivn[2].begin();p!=derivn[2].end();++p)
    {
      currlin[0][p->first] -= alpha * sval[z] * lmatrix(0,2) * (p->second);
      currlin[1][p->first] -= alpha * sval[z] * lmatrix(1,2) * (p->second);
    }
  }
  //**********************************************************************

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of lineclip vertex (3D) AuxPlane            popp 03/09|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3d::LineclipVertexLinearization(MORTAR::Vertex& currv,
                                           vector<map<int,double> >& currlin,
                                           MORTAR::Vertex* sv1, MORTAR::Vertex* sv2,
                                           MORTAR::Vertex* mv1, MORTAR::Vertex* mv2,
                                           vector<vector<map<int,double> > >& linsnodes,
                                           vector<vector<map<int,double> > >& linmnodes)
{
  // number of nodes
  int nsrows = SlaveIntElement().NumNode();
  int nmrows = MasterIntElement().NumNode();

  // iterator
  typedef map<int,double>::const_iterator CI;

  // compute factor Z
  double crossZ[3] = {0.0, 0.0, 0.0};
  crossZ[0] = (sv1->Coord()[1]-mv1->Coord()[1])*(mv2->Coord()[2]-mv1->Coord()[2])
           - (sv1->Coord()[2]-mv1->Coord()[2])*(mv2->Coord()[1]-mv1->Coord()[1]);
  crossZ[1] = (sv1->Coord()[2]-mv1->Coord()[2])*(mv2->Coord()[0]-mv1->Coord()[0])
           - (sv1->Coord()[0]-mv1->Coord()[0])*(mv2->Coord()[2]-mv1->Coord()[2]);
  crossZ[2] = (sv1->Coord()[0]-mv1->Coord()[0])*(mv2->Coord()[1]-mv1->Coord()[1])
           - (sv1->Coord()[1]-mv1->Coord()[1])*(mv2->Coord()[0]-mv1->Coord()[0]);
  double Zfac = crossZ[0]*Auxn()[0]+crossZ[1]*Auxn()[1]+crossZ[2]*Auxn()[2];

  // compute factor N
  double crossN[3] = {0.0, 0.0, 0.0};
  crossN[0] = (sv2->Coord()[1]-sv1->Coord()[1])*(mv2->Coord()[2]-mv1->Coord()[2])
           - (sv2->Coord()[2]-sv1->Coord()[2])*(mv2->Coord()[1]-mv1->Coord()[1]);
  crossN[1] = (sv2->Coord()[2]-sv1->Coord()[2])*(mv2->Coord()[0]-mv1->Coord()[0])
           - (sv2->Coord()[0]-sv1->Coord()[0])*(mv2->Coord()[2]-mv1->Coord()[2]);
  crossN[2] = (sv2->Coord()[0]-sv1->Coord()[0])*(mv2->Coord()[1]-mv1->Coord()[1])
           - (sv2->Coord()[1]-sv1->Coord()[1])*(mv2->Coord()[0]-mv1->Coord()[0]);
  double Nfac = crossN[0]*Auxn()[0]+crossN[1]*Auxn()[1]+crossN[2]*Auxn()[2];

  // slave edge vector
  double sedge[3] = {0.0, 0.0, 0.0};
  for (int k=0;k<3;++k) sedge[k] = sv2->Coord()[k] - sv1->Coord()[k];

  // prepare linearization derivZ
  double crossdZ1[3] = {0.0, 0.0, 0.0};
  double crossdZ2[3] = {0.0, 0.0, 0.0};
  double crossdZ3[3] = {0.0, 0.0, 0.0};
  crossdZ1[0] = (mv2->Coord()[1]-mv1->Coord()[1])*Auxn()[2]-(mv2->Coord()[2]-mv1->Coord()[2])*Auxn()[1];
  crossdZ1[1] = (mv2->Coord()[2]-mv1->Coord()[2])*Auxn()[0]-(mv2->Coord()[0]-mv1->Coord()[0])*Auxn()[2];
  crossdZ1[2] = (mv2->Coord()[0]-mv1->Coord()[0])*Auxn()[1]-(mv2->Coord()[1]-mv1->Coord()[1])*Auxn()[0];
  crossdZ2[0] = Auxn()[1]*(sv1->Coord()[2]-mv1->Coord()[2])-Auxn()[2]*(sv1->Coord()[1]-mv1->Coord()[1]);
  crossdZ2[1] = Auxn()[2]*(sv1->Coord()[0]-mv1->Coord()[0])-Auxn()[0]*(sv1->Coord()[2]-mv1->Coord()[2]);
  crossdZ2[2] = Auxn()[0]*(sv1->Coord()[1]-mv1->Coord()[1])-Auxn()[1]*(sv1->Coord()[0]-mv1->Coord()[0]);
  crossdZ3[0] = (sv1->Coord()[1]-mv1->Coord()[1])*(mv2->Coord()[2]-mv1->Coord()[2])-(sv1->Coord()[2]-mv1->Coord()[2])*(mv2->Coord()[1]-mv1->Coord()[1]);
  crossdZ3[1] = (sv1->Coord()[2]-mv1->Coord()[2])*(mv2->Coord()[0]-mv1->Coord()[0])-(sv1->Coord()[0]-mv1->Coord()[0])*(mv2->Coord()[2]-mv1->Coord()[2]);
  crossdZ3[2] = (sv1->Coord()[0]-mv1->Coord()[0])*(mv2->Coord()[1]-mv1->Coord()[1])-(sv1->Coord()[1]-mv1->Coord()[1])*(mv2->Coord()[0]-mv1->Coord()[0]);

  // prepare linearization derivN
  double crossdN1[3] = {0.0, 0.0, 0.0};
  double crossdN2[3] = {0.0, 0.0, 0.0};
  double crossdN3[3] = {0.0, 0.0, 0.0};
  crossdN1[0] = (mv2->Coord()[1]-mv1->Coord()[1])*Auxn()[2]-(mv2->Coord()[2]-mv1->Coord()[2])*Auxn()[1];
  crossdN1[1] = (mv2->Coord()[2]-mv1->Coord()[2])*Auxn()[0]-(mv2->Coord()[0]-mv1->Coord()[0])*Auxn()[2];
  crossdN1[2] = (mv2->Coord()[0]-mv1->Coord()[0])*Auxn()[1]-(mv2->Coord()[1]-mv1->Coord()[1])*Auxn()[0];
  crossdN2[0] = Auxn()[1]*(sv2->Coord()[2]-sv1->Coord()[2])-Auxn()[2]*(sv2->Coord()[1]-sv1->Coord()[1]);
  crossdN2[1] = Auxn()[2]*(sv2->Coord()[0]-sv1->Coord()[0])-Auxn()[0]*(sv2->Coord()[2]-sv1->Coord()[2]);
  crossdN2[2] = Auxn()[0]*(sv2->Coord()[1]-sv1->Coord()[1])-Auxn()[1]*(sv2->Coord()[0]-sv1->Coord()[0]);
  crossdN3[0] = (sv2->Coord()[1]-sv1->Coord()[1])*(mv2->Coord()[2]-mv1->Coord()[2])-(sv2->Coord()[2]-sv1->Coord()[2])*(mv2->Coord()[1]-mv1->Coord()[1]);
  crossdN3[1] = (sv2->Coord()[2]-sv1->Coord()[2])*(mv2->Coord()[0]-mv1->Coord()[0])-(sv2->Coord()[0]-sv1->Coord()[0])*(mv2->Coord()[2]-mv1->Coord()[2]);
  crossdN3[2] = (sv2->Coord()[0]-sv1->Coord()[0])*(mv2->Coord()[1]-mv1->Coord()[1])-(sv2->Coord()[1]-sv1->Coord()[1])*(mv2->Coord()[0]-mv1->Coord()[0]);

  // slave vertex linearization (2x)
  int sid1 = currv.Nodeids()[0];
  int sid2 = currv.Nodeids()[1];

  // find corresponding slave node linearizations
  int k=0;
  while (k<nsrows){
    if (SlaveIntElement().NodeIds()[k]==sid1) break;
    ++k;
  }

  // dserror if not found
  if (k==nsrows) dserror("ERROR: Slave Id1 not found!");

  // get the correct slave node linearization
  vector<map<int,double> >& slavelin0 = linsnodes[k];

  k=0;
  while (k<nsrows){
    if (SlaveIntElement().NodeIds()[k]==sid2) break;
    ++k;
  }

  // dserror if not found
  if (k==nsrows) dserror("ERROR: Slave Id2 not found!");

  // get the correct slave node linearization
  vector<map<int,double> >& slavelin1 = linsnodes[k];

  // master vertex linearization (2x)
  int mid1 = currv.Nodeids()[2];
  int mid2 = currv.Nodeids()[3];

  // find corresponding master node linearizations
  k=0;
  while (k<nmrows){
    if (MasterIntElement().NodeIds()[k]==mid1) break;
    ++k;
  }

  // dserror if not found
  if (k==nmrows) dserror("ERROR: Master Id1 not found!");

  // get the correct master node linearization
  vector<map<int,double> >& masterlin0 = linmnodes[k];

  k=0;
  while (k<nmrows){
    if (MasterIntElement().NodeIds()[k]==mid2) break;
    ++k;
  }

  // dserror if not found
  if (k==nmrows) dserror("ERROR: Master Id2 not found!");

  // get the correct master node linearization
  vector<map<int,double> >& masterlin1 = linmnodes[k];

  // linearization of element normal Auxn()
  vector<map<int,double> >& linauxn = GetDerivAuxn();

  // bring everything together -> lineclip vertex linearization
  for (int k=0;k<3;++k)
  {
    for (CI p=slavelin0[k].begin();p!=slavelin0[k].end();++p)
    {
      currlin[k][p->first] += (p->second);
      currlin[k][p->first] += Zfac/Nfac * (p->second);
      for (int dim=0;dim<3;++dim)
      {
        currlin[dim][p->first] -= sedge[dim] * 1/Nfac * crossdZ1[k] * (p->second);
        currlin[dim][p->first] -= sedge[dim] * Zfac/(Nfac*Nfac) * crossdN1[k] * (p->second);

      }
    }
    for (CI p=slavelin1[k].begin();p!=slavelin1[k].end();++p)
    {
      currlin[k][p->first] -= Zfac/Nfac * (p->second);
      for (int dim=0;dim<3;++dim)
      {
        currlin[dim][p->first] += sedge[dim] * Zfac/(Nfac*Nfac) * crossdN1[k] * (p->second);
      }
    }
    for (CI p=masterlin0[k].begin();p!=masterlin0[k].end();++p)
    {
      for (int dim=0;dim<3;++dim)
      {
      currlin[dim][p->first] += sedge[dim] * 1/Nfac * crossdZ1[k] * (p->second);
      currlin[dim][p->first] += sedge[dim] * 1/Nfac * crossdZ2[k] * (p->second);
      currlin[dim][p->first] -= sedge[dim] * Zfac/(Nfac*Nfac) * crossdN2[k] * (p->second);
      }
    }
    for (CI p=masterlin1[k].begin();p!=masterlin1[k].end();++p)
    {
      for (int dim=0;dim<3;++dim)
      {
      currlin[dim][p->first] -= sedge[dim] * 1/Nfac * crossdZ2[k] * (p->second);
      currlin[dim][p->first] += sedge[dim] * Zfac/(Nfac*Nfac) * crossdN2[k] * (p->second);
      }
    }
    for (CI p=linauxn[k].begin();p!=linauxn[k].end();++p)
    {
      for (int dim=0;dim<3;++dim)
      {
      currlin[dim][p->first] -= sedge[dim] * 1/Nfac * crossdZ3[k] * (p->second);
      currlin[dim][p->first] += sedge[dim] * Zfac/(Nfac*Nfac) * crossdN3[k] * (p->second);
      }
    }
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of lineclip vertex (3D)                     popp 02/09|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3d::LineclipVertexLinearization(MORTAR::Vertex& currv,
                                           vector<map<int,double> >& currlin,
                                           MORTAR::Vertex* sv1, MORTAR::Vertex* sv2,
                                           MORTAR::Vertex* mv1, MORTAR::Vertex* mv2,
                                           map<int,double>& projpar)
{
  // compute factor Z
  double crossZ[3] = {0.0, 0.0, 0.0};
  crossZ[0] = (sv1->Coord()[1]-mv1->Coord()[1])*(mv2->Coord()[2]-mv1->Coord()[2])
           - (sv1->Coord()[2]-mv1->Coord()[2])*(mv2->Coord()[1]-mv1->Coord()[1]);
  crossZ[1] = (sv1->Coord()[2]-mv1->Coord()[2])*(mv2->Coord()[0]-mv1->Coord()[0])
           - (sv1->Coord()[0]-mv1->Coord()[0])*(mv2->Coord()[2]-mv1->Coord()[2]);
  crossZ[2] = (sv1->Coord()[0]-mv1->Coord()[0])*(mv2->Coord()[1]-mv1->Coord()[1])
           - (sv1->Coord()[1]-mv1->Coord()[1])*(mv2->Coord()[0]-mv1->Coord()[0]);
  double Zfac = crossZ[0]*Auxn()[0]+crossZ[1]*Auxn()[1]+crossZ[2]*Auxn()[2];

  // compute factor N
  double crossN[3] = {0.0, 0.0, 0.0};
  crossN[0] = (sv2->Coord()[1]-sv1->Coord()[1])*(mv2->Coord()[2]-mv1->Coord()[2])
           - (sv2->Coord()[2]-sv1->Coord()[2])*(mv2->Coord()[1]-mv1->Coord()[1]);
  crossN[1] = (sv2->Coord()[2]-sv1->Coord()[2])*(mv2->Coord()[0]-mv1->Coord()[0])
           - (sv2->Coord()[0]-sv1->Coord()[0])*(mv2->Coord()[2]-mv1->Coord()[2]);
  crossN[2] = (sv2->Coord()[0]-sv1->Coord()[0])*(mv2->Coord()[1]-mv1->Coord()[1])
           - (sv2->Coord()[1]-sv1->Coord()[1])*(mv2->Coord()[0]-mv1->Coord()[0]);
  double Nfac = crossN[0]*Auxn()[0]+crossN[1]*Auxn()[1]+crossN[2]*Auxn()[2];

  // slave edge vector
  double sedge[3] = {0.0, 0.0, 0.0};
  for (int k=0;k<3;++k) sedge[k] = sv1->Coord()[k] - sv2->Coord()[k];

  // prepare linearization derivZ
  double crossdZ1[3] = {0.0, 0.0, 0.0};
  double crossdZ2[3] = {0.0, 0.0, 0.0};
  crossdZ1[0] = Auxn()[1]*(mv2->Coord()[2]-mv1->Coord()[2])-Auxn()[2]*(mv2->Coord()[1]-mv1->Coord()[1]);
  crossdZ1[1] = Auxn()[2]*(mv2->Coord()[0]-mv1->Coord()[0])-Auxn()[0]*(mv2->Coord()[2]-mv1->Coord()[2]);
  crossdZ1[2] = Auxn()[0]*(mv2->Coord()[1]-mv1->Coord()[1])-Auxn()[1]*(mv2->Coord()[0]-mv1->Coord()[0]);
  crossdZ2[0] = Auxn()[1]*(sv1->Coord()[2]-mv1->Coord()[2])-Auxn()[2]*(sv1->Coord()[1]-mv1->Coord()[1]);
  crossdZ2[1] = Auxn()[2]*(sv1->Coord()[0]-mv1->Coord()[0])-Auxn()[0]*(sv1->Coord()[2]-mv1->Coord()[2]);
  crossdZ2[2] = Auxn()[0]*(sv1->Coord()[1]-mv1->Coord()[1])-Auxn()[1]*(sv1->Coord()[0]-mv1->Coord()[0]);

  // prepare linearization derivN
  double crossdN1[3] = {0.0, 0.0, 0.0};
  crossdN1[0] = Auxn()[1]*(sv2->Coord()[2]-sv1->Coord()[2])-Auxn()[2]*(sv2->Coord()[1]-sv1->Coord()[1]);
  crossdN1[1] = Auxn()[2]*(sv2->Coord()[0]-sv1->Coord()[0])-Auxn()[0]*(sv2->Coord()[2]-sv1->Coord()[2]);
  crossdN1[2] = Auxn()[0]*(sv2->Coord()[1]-sv1->Coord()[1])-Auxn()[1]*(sv2->Coord()[0]-sv1->Coord()[0]);

  // master vertex linearization (2x)
  vector<vector<map<int,double> > > masterlin(2,vector<map<int,double> >(3));

  int mid1 = currv.Nodeids()[2];
  double alpha1 = projpar[mid1];

  bool found1 = false;
  MORTAR::Vertex* masterv1 = &MasterVertices()[0];
  for (int j=0;j<(int)MasterVertices().size();++j)
  {
    if (MasterVertices()[j].Nodeids()[0]==mid1)
    {
      found1=true;
      masterv1 = &MasterVertices()[j];
      break;
    }
  }
  if (!found1) dserror("ERROR: Lineclip linearization, Master vertex 1 not found!");

  MasterVertexLinearization(*masterv1,masterlin[0],mid1,alpha1);

  int mid2 = currv.Nodeids()[3];
  double alpha2 = projpar[mid2];

  bool found2 = false;
  MORTAR::Vertex* masterv2 = &MasterVertices()[0];
  for (int j=0;j<(int)MasterVertices().size();++j)
  {
    if (MasterVertices()[j].Nodeids()[0]==mid2)
    {
      found2=true;
      masterv2 = &MasterVertices()[j];
      break;
    }
  }
  if (!found2) dserror("ERROR: Lineclip linearization, Master vertex 2 not found!");

  MasterVertexLinearization(*masterv2,masterlin[1],mid2,alpha2);

  // bring everything together -> lineclip vertex linearization
  typedef map<int,double>::const_iterator CI;
  for (int k=0;k<3;++k)
  {
    for (CI p=masterlin[0][k].begin();p!=masterlin[0][k].end();++p)
    {
      for (int dim=0;dim<2;++dim)
      {
      currlin[dim][p->first] += sedge[dim] * 1/Nfac * crossdZ1[k] * (p->second);
      currlin[dim][p->first] -= sedge[dim] * 1/Nfac * crossdZ2[k] * (p->second);
      currlin[dim][p->first] += sedge[dim] * Zfac/(Nfac*Nfac) * crossdN1[k] * (p->second);
      }
    }
    for (CI p=masterlin[1][k].begin();p!=masterlin[1][k].end();++p)
    {
      for (int dim=0;dim<2;++dim)
      {
      currlin[dim][p->first] += sedge[dim] * 1/Nfac * crossdZ2[k] * (p->second);
      currlin[dim][p->first] -= sedge[dim] * Zfac/(Nfac*Nfac) * crossdN1[k] * (p->second);
      }
    }
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Linearization of clip polygon center (3D)                 popp 02/09|
 *----------------------------------------------------------------------*/
bool CONTACT::CoCoupling3d::CenterLinearization(const vector<vector<map<int,double> > >& linvertex,
                                                vector<map<int,double> >& lincenter)
{
  // preparations
  int clipsize = (int)(Clip().size());
  typedef map<int,double>::const_iterator CI;

  vector<double> clipcenter(3);
  for (int k=0;k<3;++k) clipcenter[k] = 0.0;
  double fac = 0.0;

  // first we need node averaged center
  double nac[3] = {0.0, 0.0, 0.0};
  for (int i=0;i<clipsize;++i)
    for (int k=0;k<3;++k)
      nac[k] += (Clip()[i].Coord()[k] / clipsize);

  // loop over all triangles of polygon (1st round: preparations)
  for (int i=0; i<clipsize; ++i)
  {
    double xi_i[3] = {0.0, 0.0, 0.0};
    double xi_ip1[3] = {0.0, 0.0, 0.0};

    // standard case
    if (i<clipsize-1)
    {
      for (int k=0;k<3;++k) xi_i[k] = Clip()[i].Coord()[k];
      for (int k=0;k<3;++k) xi_ip1[k] = Clip()[i+1].Coord()[k];
    }
    // last vertex of clip polygon
    else
    {
      for (int k=0;k<3;++k) xi_i[k] = Clip()[clipsize-1].Coord()[k];
      for (int k=0;k<3;++k) xi_ip1[k] = Clip()[0].Coord()[k];
    }

    // triangle area
    double diff1[3] = {0.0, 0.0, 0.0};
    double diff2[3] = {0.0, 0.0, 0.0};
    for (int k=0;k<3;++k) diff1[k] = xi_ip1[k] - xi_i[k];
    for (int k=0;k<3;++k) diff2[k] = xi_i[k] - nac[k];

    double cross[3] = {0.0, 0.0, 0.0};
    cross[0] = diff1[1]*diff2[2] - diff1[2]*diff2[1];
    cross[1] = diff1[2]*diff2[0] - diff1[0]*diff2[2];
    cross[2] = diff1[0]*diff2[1] - diff1[1]*diff2[0];

    double Atri = 0.5 * sqrt(cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2]);

    // add contributions to clipcenter and fac
    fac += Atri;
    for (int k=0;k<3;++k) clipcenter[k] += 1.0/3.0 * (xi_i[k] + xi_ip1[k] + nac[k]) * Atri;
  }

  // build factors for linearization
  double z[3] = {0.0, 0.0, 0.0};
  for (int k=0;k<3;++k) z[k] = clipcenter[k];
  double n = fac;

  // first we need linearization of node averaged center
  vector<map<int,double> > linnac(3);

  for (int i=0;i<clipsize;++i)
    for (int k=0;k<3;++k)
      for (CI p=linvertex[i][k].begin();p!=linvertex[i][k].end();++p)
        linnac[k][p->first] += 1.0/clipsize * (p->second);

  // loop over all triangles of polygon (2nd round: linearization)
  for (int i=0; i<clipsize; ++i)
  {
    double xi_i[3] = {0.0, 0.0, 0.0};
    double xi_ip1[3] = {0.0, 0.0, 0.0};
    int iplus1 = 0;

    // standard case
    if (i<clipsize-1)
    {
      for (int k=0;k<3;++k) xi_i[k] = Clip()[i].Coord()[k];
      for (int k=0;k<3;++k) xi_ip1[k] = Clip()[i+1].Coord()[k];
      iplus1 = i+1;
    }
    // last vertex of clip polygon
    else
    {
      for (int k=0;k<3;++k) xi_i[k] = Clip()[clipsize-1].Coord()[k];
      for (int k=0;k<3;++k) xi_ip1[k] = Clip()[0].Coord()[k];
      iplus1 = 0;
    }

    // triangle area
    double diff1[3] = {0.0, 0.0, 0.0};
    double diff2[3] = {0.0, 0.0, 0.0};
    for (int k=0;k<3;++k) diff1[k] = xi_ip1[k] - xi_i[k];
    for (int k=0;k<3;++k) diff2[k] = xi_i[k] - nac[k];

    double cross[3] = {0.0, 0.0, 0.0};
    cross[0] = diff1[1]*diff2[2] - diff1[2]*diff2[1];
    cross[1] = diff1[2]*diff2[0] - diff1[0]*diff2[2];
    cross[2] = diff1[0]*diff2[1] - diff1[1]*diff2[0];

    double Atri = 0.5 * sqrt(cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2]);

    // linearization of cross
    vector<map<int,double> > lincross(3);

    for (CI p=linvertex[i][0].begin();p!=linvertex[i][0].end();++p)
    {
      lincross[1][p->first] += diff1[2] * (p->second);
      lincross[1][p->first] += diff2[2] * (p->second);
      lincross[2][p->first] -= diff1[1] * (p->second);
      lincross[2][p->first] -= diff2[1] * (p->second);
    }
    for (CI p=linvertex[i][1].begin();p!=linvertex[i][1].end();++p)
    {
      lincross[0][p->first] -= diff1[2] * (p->second);
      lincross[0][p->first] -= diff2[2] * (p->second);
      lincross[2][p->first] += diff1[0] * (p->second);
      lincross[2][p->first] += diff2[0] * (p->second);
    }
    for (CI p=linvertex[i][2].begin();p!=linvertex[i][2].end();++p)
    {
      lincross[0][p->first] += diff1[1] * (p->second);
      lincross[0][p->first] += diff2[1] * (p->second);
      lincross[1][p->first] -= diff1[0] * (p->second);
      lincross[1][p->first] -= diff2[0] * (p->second);
    }

    for (CI p=linvertex[iplus1][0].begin();p!=linvertex[iplus1][0].end();++p)
    {
      lincross[1][p->first] -= diff2[2] * (p->second);
      lincross[2][p->first] += diff2[1] * (p->second);
    }
    for (CI p=linvertex[iplus1][1].begin();p!=linvertex[iplus1][1].end();++p)
    {
      lincross[0][p->first] += diff2[2] * (p->second);
      lincross[2][p->first] -= diff2[0] * (p->second);
    }
    for (CI p=linvertex[iplus1][2].begin();p!=linvertex[iplus1][2].end();++p)
    {
      lincross[0][p->first] -= diff2[1] * (p->second);
      lincross[1][p->first] += diff2[0] * (p->second);
    }

    for (CI p=linnac[0].begin();p!=linnac[0].end();++p)
    {
      lincross[1][p->first] -= diff1[2] * (p->second);
      lincross[2][p->first] += diff1[1] * (p->second);
    }
    for (CI p=linnac[1].begin();p!=linnac[1].end();++p)
    {
      lincross[0][p->first] += diff1[2] * (p->second);
      lincross[2][p->first] -= diff1[0] * (p->second);
    }
    for (CI p=linnac[2].begin();p!=linnac[2].end();++p)
    {
      lincross[0][p->first] -= diff1[1] * (p->second);
      lincross[1][p->first] += diff1[0] * (p->second);
    }

    // linearization of triangle area
    map<int,double> linarea;
    for (int k=0;k<3;++k)
      for (CI p=lincross[k].begin();p!=lincross[k].end();++p)
        linarea[p->first] += 0.25 / Atri * cross[k] * (p->second);

    // put everything together
    for (int k=0;k<3;++k)
    {
      for (CI p=linvertex[i][k].begin();p!=linvertex[i][k].end();++p)
        lincenter[k][p->first] += 1.0/(3.0*n) * Atri * (p->second);

      for (CI p=linvertex[iplus1][k].begin();p!=linvertex[iplus1][k].end();++p)
        lincenter[k][p->first] += 1.0/(3.0*n) * Atri * (p->second);

      for (CI p=linnac[k].begin();p!=linnac[k].end();++p)
        lincenter[k][p->first] += 1.0/(3.0*n) * Atri * (p->second);

      for (CI p=linarea.begin();p!=linarea.end();++p)
      {
        lincenter[k][p->first] += 1.0/n * 1.0/3.0 * (xi_i[k] + xi_ip1[k] + nac[k]) * (p->second);
        lincenter[k][p->first] -= z[k]/(n*n) * (p->second);
      }
    }
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 11/08|
 *----------------------------------------------------------------------*/
CONTACT::CoCoupling3dQuad::CoCoupling3dQuad(DRT::Discretization& idiscret,
                                int dim, bool quad, bool auxplane,
                                MORTAR::MortarElement& sele, MORTAR::MortarElement& mele,
                                MORTAR::IntElement& sintele,
                                MORTAR::IntElement& mintele,
                                INPAR::MORTAR::LagMultQuad3D& lmtype) :
CONTACT::CoCoupling3d(INPAR::MORTAR::shape_undefined,idiscret,dim,quad,auxplane,sele,mele),
sintele_(sintele),
mintele_(mintele),
lmtype_(lmtype)
{
  // 3D quadratic coupling only for aux. plane case
  if (!CouplingInAuxPlane())
    dserror("ERROR: CoCoupling3dQuad only for auxiliary plane case!");

  //  3D quadratic coupling only for quadratic ansatz type
  if (!Quad())
    dserror("ERROR: CoCoupling3dQuad called for non-quadratic ansatz!");

  return;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                             popp 06/09|
 *----------------------------------------------------------------------*/
CONTACT::CoCoupling3dQuad::CoCoupling3dQuad(const INPAR::MORTAR::ShapeFcn shapefcn,
                                DRT::Discretization& idiscret,
                                int dim, bool quad, bool auxplane,
                                MORTAR::MortarElement& sele, MORTAR::MortarElement& mele,
                                MORTAR::IntElement& sintele,
                                MORTAR::IntElement& mintele,
                                INPAR::MORTAR::LagMultQuad3D& lmtype) :
CONTACT::CoCoupling3d(shapefcn,idiscret,dim,quad,auxplane,sele,mele),
sintele_(sintele),
mintele_(mintele),
lmtype_(lmtype)
{
  // 3D quadratic coupling only for aux. plane case
  if (!CouplingInAuxPlane())
    dserror("ERROR: CoCoupling3dQuad only for auxiliary plane case!");
  
  //  3D quadratic coupling only for quadratic ansatz type
  if (!Quad())
    dserror("ERROR: CoCoupling3dQuad called for non-quadratic ansatz!");
  
  return;
}

#endif //#ifdef CCADISCRET
