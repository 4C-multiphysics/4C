/*!----------------------------------------------------------------------
\file contact_element.cpp
\brief

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

#include "contact_element.H"
#include "contact_node.H"
#include "../drt_lib/drt_utils.H"
#include "../linalg/linalg_utils.H"
#include "../linalg/linalg_serialdensevector.H"
#include "../linalg/linalg_serialdensematrix.H"

CONTACT::CoElementType CONTACT::CoElementType::instance_;


DRT::ParObject* CONTACT::CoElementType::Create( const std::vector<char> & data )
{
  CONTACT::CoElement* ele = new CONTACT::CoElement(0,
                                                   0,DRT::Element::dis_none,
                                                   0,NULL,false);
  ele->Unpack(data);
  return ele;
}


Teuchos::RCP<DRT::Element> CONTACT::CoElementType::Create( const int id, const int owner )
{
  //return Teuchos::rcp( new CoElement( id, owner ) );
  return Teuchos::null;
}


void CONTACT::CoElementType::NodalBlockInformation( DRT::Element * dwele, int & numdf, int & dimns, int & nv, int & np )
{
}


void CONTACT::CoElementType::ComputeNullSpace( DRT::Discretization & dis, std::vector<double> & ns, const double * x0, int numdf, int dimns )
{
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 10/07|
 *----------------------------------------------------------------------*/
CONTACT::CoElement::CoElement(int id, int owner,
                            const DRT::Element::DiscretizationType& shape,
                            const int numnode,
                            const int* nodeids,
                            const bool isslave) :
MORTAR::MortarElement(id,owner,shape,numnode,nodeids,isslave)
{
  // empty constructor

  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                        mwgee 10/07|
 *----------------------------------------------------------------------*/
CONTACT::CoElement::CoElement(const CONTACT::CoElement& old) :
MORTAR::MortarElement(old)
{
  // empty copy-constructor

  return;
}

/*----------------------------------------------------------------------*
 |  clone-ctor (public)                                      mwgee 10/07|
 *----------------------------------------------------------------------*/
CONTACT::CoElement* CONTACT::CoElement::Clone() const
{
  CONTACT::CoElement* newele = new CONTACT::CoElement(*this);
  return newele;
}


/*----------------------------------------------------------------------*
 |  << operator                                              mwgee 10/07|
 *----------------------------------------------------------------------*/
ostream& operator << (ostream& os, const CONTACT::CoElement& element)
{
  element.Print(os);
  return os;
}


/*----------------------------------------------------------------------*
 |  print element (public)                                   mwgee 10/07|
 *----------------------------------------------------------------------*/
void CONTACT::CoElement::Print(ostream& os) const
{
  os << "Contact ";
  MORTAR::MortarElement::Print(os);

  return;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                           mwgee 10/07|
 *----------------------------------------------------------------------*/
void CONTACT::CoElement::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);

  // add base class MORTAR::MortarElement
  MORTAR::MortarElement::Pack(data);

  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                           mwgee 10/07|
 *----------------------------------------------------------------------*/
void CONTACT::CoElement::Unpack(const vector<char>& data)
{
  vector<char>::size_type position = 0;

  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");

  // extract base class MORTAR::MortarElement
  vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  MORTAR::MortarElement::Unpack(basedata);

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}

/*----------------------------------------------------------------------*
 |  number of dofs per node (public)                         mwgee 10/07|
 *----------------------------------------------------------------------*/
int CONTACT::CoElement::NumDofPerNode(const DRT::Node& node) const
{
  const CONTACT::CoNode* cnode = dynamic_cast<const CONTACT::CoNode*>(&node);
  if (!cnode) dserror("Node is not a CoNode");
  return cnode->NumDof();
}

/*----------------------------------------------------------------------*
 |  evaluate element (public)                                mwgee 10/07|
 *----------------------------------------------------------------------*/
int CONTACT::CoElement::Evaluate(ParameterList&            params,
                                DRT::Discretization&      discretization,
                                vector<int>&              lm,
                                Epetra_SerialDenseMatrix& elemat1,
                                Epetra_SerialDenseMatrix& elemat2,
                                Epetra_SerialDenseVector& elevec1,
                                Epetra_SerialDenseVector& elevec2,
                                Epetra_SerialDenseVector& elevec3)
{
  dserror("CONTACT::CoElement::Evaluate not implemented!");
  return -1;
}

/*----------------------------------------------------------------------*
 |  Build element normal derivative at node                   popp 05/08|
 *----------------------------------------------------------------------*/
void CONTACT::CoElement::DerivNormalAtNode(int nid, int& i,
                                          Epetra_SerialDenseMatrix& elens,
                                          vector<map<int,double> >& derivn)
{
  // find this node in my list of nodes and get local numbering
  int lid = GetLocalNodeId(nid);

  // get local coordinates for this node
  double xi[2];
  LocalCoordinatesOfNode(lid,xi);

  // build normal derivative at xi and return it
  DerivNormalAtXi(xi,i,elens,derivn);

  return;
}

/*----------------------------------------------------------------------*
 |  Compute element normal derivative at loc. coord. xi       popp 09/08|
 *----------------------------------------------------------------------*/
void CONTACT::CoElement::DerivNormalAtXi(double* xi, int& i,
                                        Epetra_SerialDenseMatrix& elens,
                                        vector<map<int,double> >& derivn)
{
  // initialize variables
  int nnodes = NumNode();
  DRT::Node** mynodes = Nodes();
  if (!mynodes) dserror("ERROR: DerivNormalAtXi: Null pointer!");
  LINALG::SerialDenseVector val(nnodes);
  LINALG::SerialDenseMatrix deriv(nnodes,2,true);
  vector<double> gxi(3);
  vector<double> geta(3);

  // get shape function values and derivatives at xi
  EvaluateShape(xi, val, deriv, nnodes);

  // get local element basis vectors
  Metrics(xi, gxi, geta);

  // derivative weighting matrix for current element
  LINALG::Matrix<3,3> W;
  double lcube  = elens(4,i)*elens(4,i)*elens(4,i);

  for (int j=0;j<3;++j)
  {
    for (int k=0;k<3;++k)
    {
      W(j,k) = -1/lcube * elens(j,i) * elens(k,i);
      if (j==k) W(j,k) += 1/elens(4,i);
    }
  }

  // now loop over all element nodes for derivatives
  for (int n=0;n<nnodes;++n)
  {
    CoNode* mycnode = static_cast<CoNode*> (mynodes[n]);
    if (!mycnode) dserror("ERROR: DerivNormalAtXi: Null pointer!");
    int ndof = mycnode->NumDof();

    // derivative weighting matrix for current node
    LINALG::Matrix<3,3> F;
    F(0,0) = 0.0;
    F(1,1) = 0.0;
    F(2,2) = 0.0;
    F(0,1) = geta[2] * deriv(n,0) - gxi[2]  * deriv(n,1);
    F(0,2) = gxi[1]  * deriv(n,1) - geta[1] * deriv(n,0);
    F(1,0) = gxi[2]  * deriv(n,1) - geta[2] * deriv(n,0);
    F(1,2) = geta[0] * deriv(n,0) - gxi[0]  * deriv(n,1);
    F(2,0) = geta[1] * deriv(n,0) - gxi[1]  * deriv(n,1);
    F(2,1) = gxi[0]  * deriv(n,1) - geta[0] * deriv(n,0);

    // total weighting matrix
    LINALG::Matrix<3,3> WF;
    WF.MultiplyNN(W,F);

    //create directional derivatives
    for (int j=0;j<3;++j)
      for (int k=0;k<ndof;++k)
        (derivn[j])[mycnode->Dofs()[k]] += WF(j,k);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate derivative J,xi of Jacobian determinant          popp 05/08|
 *----------------------------------------------------------------------*/
void CONTACT::CoElement::DJacDXi(double* djacdxi, double* xi,
                                const LINALG::SerialDenseMatrix& secderiv)
{
  // the derivative dJacdXi
  djacdxi[0] = 0.0;
  djacdxi[1] = 0.0;
  DRT::Element::DiscretizationType dt = Shape();

  // 2D linear case (2noded line element)
  // 3D linear case (3noded triangular element)
  if (dt==line2 || dt==tri3)
  {
    // do nothing
  }

  // 2D quadratic case (3noded line element)
  else if (dt==line3)
  {
    // get nodal coords for 2nd deriv. evaluation
    LINALG::SerialDenseMatrix coord(3,NumNode());
    GetNodalCoords(coord);

    // metrics routine gives local basis vectors
    vector<double> gxi(3);
    vector<double> geta(3);
    Metrics(xi,gxi,geta);

    double gsec[3] = {0.0, 0.0, 0.0};
    for (int i=0;i<NumNode();++i)
      for (int k=0;k<3;++k)
        gsec[k] += secderiv(i,0)*coord(k,i);

    // the Jacobian itself
    double jac = sqrt(gxi[0]*gxi[0]+gxi[1]*gxi[1]+gxi[2]*gxi[2]);

    // compute dJacdXi (1 component in 2D)
    for (int dim=0;dim<3;++dim)
      djacdxi[0] += gxi[dim]*gsec[dim]/jac;
  }

  // 3D bilinear case    (4noded quadrilateral element)
  // 3D quadratic case   (6noded triangular element)
  // 3D serendipity case (8noded quadrilateral element)
  // 3D biquadratic case (9noded quadrilateral element)
  else if (dt==quad4 || dt==tri6 || dt==quad8 || dt==quad9)
  {
    // get nodal coords for 2nd deriv. evaluation
    LINALG::SerialDenseMatrix coord(3,NumNode());
    GetNodalCoords(coord);

    // metrics routine gives local basis vectors
    vector<double> gxi(3);
    vector<double> geta(3);
    Metrics(xi,gxi,geta);

    // cross product of gxi and geta
    double cross[3] = {0.0, 0.0, 0.0};
    cross[0] = gxi[1]*geta[2]-gxi[2]*geta[1];
    cross[1] = gxi[2]*geta[0]-gxi[0]*geta[2];
    cross[2] = gxi[0]*geta[1]-gxi[1]*geta[0];

    // the Jacobian itself
    double jac = sqrt(cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2]);

    // 2nd deriv. evaluation
    LINALG::Matrix<3,3> gsec(true);
    for (int i=0;i<NumNode();++i)
      for (int k=0;k<3;++k)
        for (int d=0;d<3;++d)
          gsec(k,d) += secderiv(i,d)*coord(k,i);

    // compute dJacdXi (2 components in 3D)
    djacdxi[0] += 1/jac * (cross[2]*geta[1]-cross[1]*geta[2]) * gsec(0,0);
    djacdxi[0] += 1/jac * (cross[0]*geta[2]-cross[2]*geta[0]) * gsec(1,0);
    djacdxi[0] += 1/jac * (cross[1]*geta[0]-cross[0]*geta[1]) * gsec(2,0);
    djacdxi[0] += 1/jac * (cross[1]*gxi[2]-cross[2]*gxi[1])   * gsec(0,2);
    djacdxi[0] += 1/jac * (cross[2]*gxi[0]-cross[0]*gxi[2])   * gsec(1,2);
    djacdxi[0] += 1/jac * (cross[0]*gxi[1]-cross[1]*gxi[0])   * gsec(2,2);
    djacdxi[1] += 1/jac * (cross[2]*geta[1]-cross[1]*geta[2]) * gsec(0,2);
    djacdxi[1] += 1/jac * (cross[0]*geta[2]-cross[2]*geta[0]) * gsec(1,2);
    djacdxi[1] += 1/jac * (cross[1]*geta[0]-cross[0]*geta[1]) * gsec(2,2);
    djacdxi[1] += 1/jac * (cross[1]*gxi[2]-cross[2]*gxi[1])   * gsec(0,1);
    djacdxi[1] += 1/jac * (cross[2]*gxi[0]-cross[0]*gxi[2])   * gsec(1,1);
    djacdxi[1] += 1/jac * (cross[0]*gxi[1]-cross[1]*gxi[0])   * gsec(2,1);
  }

  // unknown case
  else
    dserror("ERROR: DJacDXi called for unknown element type!");

  return;
}

#endif  // #ifdef CCADISCRET
