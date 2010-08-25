/*!----------------------------------------------------------------------
\file mortar_element.cpp
\brief A mortar coupling element

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

#include "mortar_element.H"
#include "mortar_node.H"
#include "../drt_lib/drt_utils.H"
#include "../linalg/linalg_utils.H"
#include "../linalg/linalg_serialdensevector.H"
#include "../linalg/linalg_serialdensematrix.H"

MORTAR::MortarElementType MORTAR::MortarElementType::instance_;


DRT::ParObject* MORTAR::MortarElementType::Create( const std::vector<char> & data )
{
  MORTAR::MortarElement* ele = new MORTAR::MortarElement(0,
                                                         0,DRT::Element::dis_none,
                                                         0,NULL,false);
  ele->Unpack(data);
  return ele;
}


Teuchos::RCP<DRT::Element> MORTAR::MortarElementType::Create( const int id, const int owner )
{
  //return Teuchos::rcp( new MortarElement( id, owner ) );
  return Teuchos::null;
}


void MORTAR::MortarElementType::NodalBlockInformation( DRT::Element * dwele, int & numdf, int & dimns, int & nv, int & np )
{
}

void MORTAR::MortarElementType::ComputeNullSpace( DRT::Discretization & dis, std::vector<double> & ns, const double * x0, int numdf, int dimns )
{
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 10/07|
 *----------------------------------------------------------------------*/
MORTAR::MortarElement::MortarElement(int id, int owner,
                           const DRT::Element::DiscretizationType& shape,
                           const int numnode,
                           const int* nodeids,
                           const bool isslave) :
DRT::Element(id,owner),
shape_(shape),
isslave_(isslave)
{
  SetNodeIds(numnode,nodeids);
  Area()=0.0;
  return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                        mwgee 10/07|
 *----------------------------------------------------------------------*/
MORTAR::MortarElement::MortarElement(const MORTAR::MortarElement& old) :
DRT::Element(old),
shape_(old.shape_),
isslave_(old.isslave_),
area_(old.area_),
searchelements_(old.searchelements_)
{
  return;
}

/*----------------------------------------------------------------------*
 |  clone-ctor (public)                                      mwgee 10/07|
 *----------------------------------------------------------------------*/
MORTAR::MortarElement* MORTAR::MortarElement::Clone() const
{
  MORTAR::MortarElement* newele = new MORTAR::MortarElement(*this);
  return newele;
}


/*----------------------------------------------------------------------*
 |  << operator                                              mwgee 10/07|
 *----------------------------------------------------------------------*/
ostream& operator << (ostream& os, const MORTAR::MortarElement& element)
{
  element.Print(os);
  return os;
}


/*----------------------------------------------------------------------*
 |  print element (public)                                   mwgee 10/07|
 *----------------------------------------------------------------------*/
void MORTAR::MortarElement::Print(ostream& os) const
{
  os << "Mortar Element ";
  DRT::Element::Print(os);
  if (isslave_) os << " Slave  ";
  else          os << " Master ";

  return;
}

/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                           mwgee 10/07|
 *----------------------------------------------------------------------*/
void MORTAR::MortarElement::Pack(vector<char>& data) const
{
  data.resize(0);

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class DRT::Element
  vector<char> basedata(0);
  DRT::Element::Pack(basedata);
  AddtoPack(data,basedata);
  // add shape_
  AddtoPack(data,shape_);
  // add isslave_
  AddtoPack(data,isslave_);
  // add area_
  AddtoPack(data,area_);
  // add searchelements_
  AddtoPack(data,&searchelements_,(int)searchelements_.size());

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                           mwgee 10/07|
 *----------------------------------------------------------------------*/
void MORTAR::MortarElement::Unpack(const vector<char>& data)
{
  vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // extract base class DRT::Element
  vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  DRT::Element::Unpack(basedata);
  // shape_
  ExtractfromPack(position,data,shape_);
  // isslave_
  ExtractfromPack(position,data,isslave_);
  // area_
  ExtractfromPack(position,data,area_);
  // searchelements_
  ExtractfromPack(position,data,&searchelements_,(int)searchelements_.size());

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}

/*----------------------------------------------------------------------*
 |  number of dofs per node (public)                         mwgee 10/07|
 *----------------------------------------------------------------------*/
int MORTAR::MortarElement::NumDofPerNode(const DRT::Node& node) const
{
	 const MORTAR::MortarNode* mnode = dynamic_cast<const MORTAR::MortarNode*>(&node);
	 if (!mnode) dserror("Node is not a MortarNode");
	 return mnode->NumDof();
}

/*----------------------------------------------------------------------*
 |  evaluate element (public)                                mwgee 10/07|
 *----------------------------------------------------------------------*/
int MORTAR::MortarElement::Evaluate(ParameterList&            params,
                                DRT::Discretization&      discretization,
                                vector<int>&              lm,
                                Epetra_SerialDenseMatrix& elemat1,
                                Epetra_SerialDenseMatrix& elemat2,
                                Epetra_SerialDenseVector& elevec1,
                                Epetra_SerialDenseVector& elevec2,
                                Epetra_SerialDenseVector& elevec3)
{
  dserror("MORTAR::MortarElement::Evaluate not implemented!");
  return -1;
}

/*----------------------------------------------------------------------*
 |  Get local coordinates for local node id                   popp 12/07|
 *----------------------------------------------------------------------*/
bool MORTAR::MortarElement::LocalCoordinatesOfNode(int& lid, double* xi)
{
  // 2D linear case (2noded line element)
  // 2D quadratic case (3noded line element)
  if (Shape()==line2 || Shape()==line3)
  {
    if (lid==0)
      xi[0]=-1.0;
    else if (lid==1)
      xi[0]= 1.0;
    else if (lid==2)
      xi[0]= 0.0;
    else
      dserror("ERROR: LocalCoordinatesOfNode: Node number % in segment % out of range",lid,Id());

    // we are in the 2D case here!
    xi[1]=0.0;
  }

  // 3D linear case (2noded triangular element)
  // 3D quadratic case (3noded triangular element)
  else if (Shape()==tri3 || Shape()==tri6)
  {
    switch(lid)
    {
    case 0:
    {
      xi[0]=0.0; xi[1]=0.0;
      break;
    }
    case 1:
    {
      xi[0]=1.0; xi[1]=0.0;
      break;
    }
    case 2:
    {
      xi[0]=0.0; xi[1]=1.0;
      break;
    }
    case 3:
    {
      xi[0]=0.5; xi[1]=0.0;
      break;
    }
    case 4:
    {
      xi[0]=0.5; xi[1]=0.5;
      break;
    }
    case 5:
    {
      xi[0]=0.0; xi[1]=0.5;
      break;
    }
    default:
      dserror("ERROR: LocCoordsOfNode: Node number % in segment % out of range",lid,Id());
    }
  }

  // 3D bilinear case (4noded quadrilateral element)
  // 3D serendipity case (8noded quadrilateral element)
  // 3D biquadratic case (9noded quadrilateral element)
  else if (Shape()==quad4 || Shape()==quad8 || Shape()==quad9)
  {
    switch(lid)
    {
    case 0:
    {
      xi[0]=-1.0; xi[1]=-1.0;
      break;
    }
    case 1:
    {
      xi[0]=1.0; xi[1]=-1.0;
      break;
    }
    case 2:
    {
      xi[0]=1.0; xi[1]=1.0;
      break;
    }
    case 3:
    {
      xi[0]=-1.0; xi[1]=1.0;
      break;
    }
    case 4:
    {
      xi[0]=0.0; xi[1]=-1.0;
      break;
    }
    case 5:
    {
      xi[0]=1.0; xi[1]=0.0;
      break;
    }
    case 6:
    {
      xi[0]=0.0; xi[1]=1.0;
      break;
    }
    case 7:
    {
      xi[0]=-1.0; xi[1]=0.0;
      break;
    }
    case 8:
    {
      xi[0]=0.0; xi[1]=0.0;
      break;
    }
    default:
      dserror("ERROR: LocCoordsOfNode: Node number % in segment % out of range",lid,Id());
    }
  }

  // unknown case
  else
    dserror("ERROR: LocalCoordinatesOfNode called for unknown element type");

  return true;
}

/*----------------------------------------------------------------------*
 |  Get local numbering for global node id                    popp 12/07|
 *----------------------------------------------------------------------*/
int MORTAR::MortarElement::GetLocalNodeId(int& nid)
{
  int lid = -1;

  // look for global ID nid in element's nodes
  for (int i=0;i<NumNode();++i)
    if (NodeIds()[i] == nid)
    {
      lid=i;
      break;
    }

  if (lid<0)
    dserror("ERROR: Cannot find node % in segment %", nid, Id());

  return lid;
}

/*----------------------------------------------------------------------*
 |  Build element normal at node                              popp 12/07|
 *----------------------------------------------------------------------*/
void MORTAR::MortarElement::BuildNormalAtNode(int nid, int& i,
                                          Epetra_SerialDenseMatrix& elens)
{
  // find this node in my list of nodes and get local numbering
  int lid = GetLocalNodeId(nid);

  // get local coordinates for this node
  double xi[2];
  LocalCoordinatesOfNode(lid,xi);

  // build an outward unit normal at xi and return it
  ComputeNormalAtXi(xi,i,elens);

  return;
}

/*----------------------------------------------------------------------*
 |  Compute element normal at loc. coord. xi                  popp 09/08|
 *----------------------------------------------------------------------*/
void MORTAR::MortarElement::ComputeNormalAtXi(double* xi, int& i,
                                          Epetra_SerialDenseMatrix& elens)
{
  // empty local basis vectors
  vector<double> gxi(3);
  vector<double> geta(3);

  // metrics routine gives local basis vectors
  Metrics(xi,gxi,geta);

  // n is cross product of gxi and geta
  elens(0,i) = gxi[1]*geta[2]-gxi[2]*geta[1];
  elens(1,i) = gxi[2]*geta[0]-gxi[0]*geta[2];
  elens(2,i) = gxi[0]*geta[1]-gxi[1]*geta[0];

  // store length of normal and other information into elens
  elens(4,i) = sqrt(elens(0,i)*elens(0,i)+elens(1,i)*elens(1,i)+elens(2,i)*elens(2,i));
  if (elens(4,i)==0.0) dserror("ERROR: ComputeNormalAtXi gives normal of length 0!");
  elens(3,i) = Id();
  elens(5,i) = Area();

  return;
}

/*----------------------------------------------------------------------*
 |  Compute element normal at loc. coord. xi                  popp 11/08|
 *----------------------------------------------------------------------*/
double MORTAR::MortarElement::ComputeUnitNormalAtXi(double* xi, double* n)
{
  // check input
  if (!xi) dserror("ERROR: ComputeUnitNormalAtXi called with xi=NULL");
  if (!n)  dserror("ERROR: ComputeUnitNormalAtXi called with n=NULL");

  // empty local basis vectors
  vector<double> gxi(3);
  vector<double> geta(3);

  // metrics routine gives local basis vectors
  Metrics(xi,gxi,geta);

  // n is cross product of gxi and geta
  n[0] = gxi[1]*geta[2]-gxi[2]*geta[1];
  n[1] = gxi[2]*geta[0]-gxi[0]*geta[2];
  n[2] = gxi[0]*geta[1]-gxi[1]*geta[0];

  // build unit normal
  double length = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
  if (length==0.0) dserror("ERROR: Normal of length zero!");
  for (int i=0;i<3;++i) n[i] /= length;

  return length;
}

/*----------------------------------------------------------------------*
 |  Compute unit normal derivative at loc. coord. xi          popp 03/09|
 *----------------------------------------------------------------------*/
void MORTAR::MortarElement::DerivUnitNormalAtXi(double* xi, vector<map<int,double> >& derivn)
{
  // resize derivn
  if ((int)derivn.size()!=3) derivn.resize(3);

  // initialize variables
  int nnodes = NumNode();
  DRT::Node** mynodes = Nodes();
  if (!mynodes) dserror("ERROR: DerivUnitNormalAtXi: Null pointer!");
  LINALG::SerialDenseVector val(nnodes);
  LINALG::SerialDenseMatrix deriv(nnodes,2,true);
  vector<double> gxi(3);
  vector<double> geta(3);

  // get shape function values and derivatives at xi
  EvaluateShape(xi, val, deriv, nnodes);

  // get local element basis vectors
  Metrics(xi, gxi, geta);

  // n is cross product of gxi and geta
  double n[3] = {0.0, 0.0, 0.0};
  n[0] = gxi[1]*geta[2]-gxi[2]*geta[1];
  n[1] = gxi[2]*geta[0]-gxi[0]*geta[2];
  n[2] = gxi[0]*geta[1]-gxi[1]*geta[0];

  // build unit normal
  double length = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
  if (length==0.0) dserror("ERROR: Normal of length zero!");
  for (int i=0;i<3;++i) n[i] /= length;

  // non-unit normal derivative
  vector<map<int,double> > derivnnu(3);
  typedef map<int,double>::const_iterator CI;

  // now the derivative
  for (int n=0;n<nnodes;++n)
  {
    MortarNode* mymrtrnode = static_cast<MortarNode*> (mynodes[n]);
    if (!mymrtrnode) dserror("ERROR: DerivUnitNormalAtXi: Null pointer!");
    int ndof = mymrtrnode->NumDof();

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

    //create directional derivatives
    for (int j=0;j<3;++j)
      for (int k=0;k<ndof;++k)
        (derivnnu[j])[mymrtrnode->Dofs()[k]] += F(j,k);
  }

  double ll = length*length;
  double sxsx = n[0]*n[0]*ll;
  double sxsy = n[0]*n[1]*ll;
  double sxsz = n[0]*n[2]*ll;
  double sysy = n[1]*n[1]*ll;
  double sysz = n[1]*n[2]*ll;
  double szsz = n[2]*n[2]*ll;

  for (CI p=derivnnu[0].begin();p!=derivnnu[0].end();++p)
  {
    derivn[0][p->first] += 1/length*(p->second);
    derivn[0][p->first] -= 1/(length*length*length)*sxsx*(p->second);
    derivn[1][p->first] -= 1/(length*length*length)*sxsy*(p->second);
    derivn[2][p->first] -= 1/(length*length*length)*sxsz*(p->second);
  }

  for (CI p=derivnnu[1].begin();p!=derivnnu[1].end();++p)
  {
    derivn[1][p->first] += 1/length*(p->second);
    derivn[1][p->first] -= 1/(length*length*length)*sysy*(p->second);
    derivn[0][p->first] -= 1/(length*length*length)*sxsy*(p->second);
    derivn[2][p->first] -= 1/(length*length*length)*sysz*(p->second);
  }

  for (CI p=derivnnu[2].begin();p!=derivnnu[2].end();++p)
  {
    derivn[2][p->first] += 1/length*(p->second);
    derivn[2][p->first] -= 1/(length*length*length)*szsz*(p->second);
    derivn[0][p->first] -= 1/(length*length*length)*sxsz*(p->second);
    derivn[1][p->first] -= 1/(length*length*length)*sysz*(p->second);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Get nodal coordinates of the element                      popp 01/08|
 *----------------------------------------------------------------------*/
void MORTAR::MortarElement::GetNodalCoords(LINALG::SerialDenseMatrix& coord,
                                       bool isinit)
{
  int nnodes = NumNode();
  DRT::Node** mynodes = Nodes();
  if (!mynodes) dserror("ERROR: GetNodalCoords: Null pointer!");
  if (coord.M()!=3 || coord.N()!=nnodes) dserror("ERROR: GetNodalCoords: Dimensions!");

  for (int i=0;i<nnodes;++i)
  {
    MortarNode* mymrtrnode = static_cast<MortarNode*> (mynodes[i]);
    if (!mymrtrnode) dserror("ERROR: GetNodalCoords: Null pointer!");
    if (isinit)
    {
      coord(0,i) = mymrtrnode->X()[0];
      coord(1,i) = mymrtrnode->X()[1];
      coord(2,i) = mymrtrnode->X()[2];
    }
    else
    {
      coord(0,i) = mymrtrnode->xspatial()[0];
      coord(1,i) = mymrtrnode->xspatial()[1];
      coord(2,i) = mymrtrnode->xspatial()[2];
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Get old nodal coordinates of the element               gitterle 08/10|
 *----------------------------------------------------------------------*/
void MORTAR::MortarElement::GetNodalCoordsOld(LINALG::SerialDenseMatrix& coord,
                                              bool isinit)
{
  int nnodes = NumNode();
  DRT::Node** mynodes = Nodes();
  if (!mynodes) dserror("ERROR: GetNodalCoords: Null pointer!");
  if (coord.M()!=3 || coord.N()!=nnodes) dserror("ERROR: GetNodalCoords: Dimensions!");

  for (int i=0;i<nnodes;++i)
  {
    MortarNode* mymrtrnode = static_cast<MortarNode*> (mynodes[i]);
    if (!mymrtrnode) dserror("ERROR: GetNodalCoords: Null pointer!");
    
    coord(0,i) = mymrtrnode->X()[0] + mymrtrnode->uold()[0];
    coord(1,i) = mymrtrnode->X()[1] + mymrtrnode->uold()[1];
    coord(2,i) = mymrtrnode->X()[2] + mymrtrnode->uold()[2];
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Get lagrange multipliers of the element                gitterle 08/10|
 *----------------------------------------------------------------------*/
void MORTAR::MortarElement::GetNodalLagMult(LINALG::SerialDenseMatrix& lagmult,
                                       bool isinit)
{
  int nnodes = NumNode();
  DRT::Node** mynodes = Nodes();
  if (!mynodes) dserror("ERROR: GetNodalCoords: Null pointer!");
  if (lagmult.M()!=3 || lagmult.N()!=nnodes) dserror("ERROR: GetNodalCoords: Dimensions!");

  for (int i=0;i<nnodes;++i)
  {
    MortarNode* mymrtrnode = static_cast<MortarNode*> (mynodes[i]);
    if (!mymrtrnode) dserror("ERROR: GetNodalCoords: Null pointer!");
    
    lagmult(0,i) = mymrtrnode->MoData().lm()[0];
    lagmult(1,i) = mymrtrnode->MoData().lm()[1];
    lagmult(2,i) = mymrtrnode->MoData().lm()[2];
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Evaluate element metrics (local basis vectors)            popp 08/08|
 *----------------------------------------------------------------------*/
void MORTAR::MortarElement::Metrics(double* xi, vector<double>& gxi,
                                vector<double>& geta)
{
  int nnodes = NumNode();
  int dim = 0;
  DRT::Element::DiscretizationType dt = Shape();
  if (dt==line2 || dt==line3) dim = 2;
  else if (dt==tri3 || dt==quad4 || dt==tri6 || dt==quad8 || dt==quad9) dim = 3;
  else dserror("ERROR: Metrics called for unknown element type");

  LINALG::SerialDenseVector val(nnodes);
  LINALG::SerialDenseMatrix deriv(nnodes,2,true);

  // get shape function values and derivatives at xi
  EvaluateShape(xi, val, deriv, nnodes);

  // get coordinates of element nodes
  LINALG::SerialDenseMatrix coord(3,nnodes);
  GetNodalCoords(coord);

  // build basis vectors gxi and geta
  for (int i=0;i<nnodes;++i)
  {
    // first local basis vector
    gxi[0] += deriv(i,0)*coord(0,i);
    gxi[1] += deriv(i,0)*coord(1,i);
    gxi[2] += deriv(i,0)*coord(2,i);

    // second local basis vector
    geta[0] += deriv(i,1)*coord(0,i);
    geta[1] += deriv(i,1)*coord(1,i);
    geta[2] += deriv(i,1)*coord(2,i);
  }

  // reset geta to (0,0,1) in 2D case
  if (dim==2)
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
double MORTAR::MortarElement::Jacobian(double* xi)
{
  double jac = 0.0;
  vector<double> gxi(3);
  vector<double> geta(3);
  DRT::Element::DiscretizationType dt = Shape();

  // 2D linear case (2noded line element)
  if (dt==line2)
    jac = Area()/2;

  // 3D linear case (3noded triangular element)
  else if (dt==tri3)
    jac = Area()*2;

  // 2D quadratic case (3noded line element)
  // 3D bilinear case (4noded quadrilateral element)
  // 3D quadratic case (6noded triangular element)
  // 3D serendipity case (8noded quadrilateral element)
  // 3D biquadratic case (9noded quadrilateral element)
  else if (dt==line3 || dt==quad4 || dt==tri6 || dt==quad8 || dt==quad9)
  {
    // metrics routine gives local basis vectors
    Metrics(xi,gxi,geta);

    // cross product of gxi and geta
    double cross[3] = {0.0, 0.0, 0.0};
    cross[0] = gxi[1]*geta[2]-gxi[2]*geta[1];
    cross[1] = gxi[2]*geta[0]-gxi[0]*geta[2];
    cross[2] = gxi[0]*geta[1]-gxi[1]*geta[0];
    jac = sqrt(cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2]);
  }

  // unknown case
  else
    dserror("ERROR: Jacobian called for unknown element type!");

  return jac;
}

/*----------------------------------------------------------------------*
 |  Evaluate directional deriv. of Jacobian det.              popp 05/08|
 *----------------------------------------------------------------------*/
void MORTAR::MortarElement::DerivJacobian(double* xi, map<int,double>& derivjac)
{
  // get element nodes
  int nnodes = NumNode();
  DRT::Node** mynodes = Nodes();
  if (!mynodes) dserror("ERROR: DerivJacobian: Null pointer!");

  // the Jacobian itself
  double jac = 0.0;
  vector<double> gxi(3);
  vector<double> geta(3);

  // evaluate shape functions
  LINALG::SerialDenseVector val(nnodes);
  LINALG::SerialDenseMatrix deriv(nnodes,2,true);
  EvaluateShape(xi, val, deriv, nnodes);

  // metrics routine gives local basis vectors
  Metrics(xi,gxi,geta);

  // cross product of gxi and geta
  double cross[3] = {0.0, 0.0, 0.0};
  cross[0] = gxi[1]*geta[2]-gxi[2]*geta[1];
  cross[1] = gxi[2]*geta[0]-gxi[0]*geta[2];
  cross[2] = gxi[0]*geta[1]-gxi[1]*geta[0];

  DRT::Element::DiscretizationType dt = Shape();

  // 2D linear case (2noded line element)
  if (dt==line2) jac = Area()/2;

  // 3D linear case (3noded triangular element)
  else if (dt==tri3) jac = Area()*2;

  // 2D quadratic case (3noded line element)
  // 3D bilinear case (4noded quadrilateral element)
  // 3D quadratic case (6noded triangular element)
  // 3D serendipity case (8noded quadrilateral element)
  // 3D biquadratic case (9noded quadrilateral element)
  else if (dt==line3 || dt==quad4 || dt==tri6 || dt==quad8 || dt==quad9)
    jac = sqrt(cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2]);

  else
    dserror("ERROR: Jac. derivative not implemented for this type of CoElement");

  // *********************************************************************
  // compute Jacobian derivative
  // *********************************************************************
  // (loop over all nodes and over all nodal dofs to capture all
  // potential dependencies of the Jacobian. Note that here we only
  // need to compute the DIRECT derivative of Lin(J), as the current
  // GP coordinate does not change! The derivative DJacDXi is done in
  // a special function (see above)!
  // *********************************************************************
  for (int i=0;i<nnodes;++i)
  {
    MORTAR::MortarNode* mymrtrnode = static_cast<MORTAR::MortarNode*>(mynodes[i]);
    if (!mymrtrnode) dserror("ERROR: DerivJacobian: Null pointer!");

    derivjac[mymrtrnode->Dofs()[0]] += 1/jac*(cross[2]*geta[1]-cross[1]*geta[2])*deriv(i,0);
    derivjac[mymrtrnode->Dofs()[0]] += 1/jac*(cross[1]*gxi[2]-cross[2]*gxi[1])*deriv(i,1);
    derivjac[mymrtrnode->Dofs()[1]] += 1/jac*(cross[0]*geta[2]-cross[2]*geta[0])*deriv(i,0);
    derivjac[mymrtrnode->Dofs()[1]] += 1/jac*(cross[2]*gxi[0]-cross[0]*gxi[2])*deriv(i,1);

    if (mymrtrnode->NumDof()==3)
    {
      derivjac[mymrtrnode->Dofs()[2]] += 1/jac*(cross[1]*geta[0]-cross[0]*geta[1])*deriv(i,0);
      derivjac[mymrtrnode->Dofs()[2]] += 1/jac*(cross[0]*gxi[1]-cross[1]*gxi[0])*deriv(i,1);
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Compute length / area of the element                      popp 12/07|
 *----------------------------------------------------------------------*/
double MORTAR::MortarElement::ComputeArea()
{
  double area = 0.0;
  DRT::Element::DiscretizationType dt = Shape();

  // 2D linear case (2noded line element)
  if (dt==line2)
  {
    // no integration necessary (constant Jacobian)
    LINALG::SerialDenseMatrix coord(3,NumNode());
    GetNodalCoords(coord);

    // build vector between the two nodes
    double tang[3] = {0.0, 0.0, 0.0};
    for (int k=0;k<3;++k)
    {
      tang[k]=coord(k,1)-coord(k,0);
    }
    area=sqrt(tang[0]*tang[0]+tang[1]*tang[1]+tang[2]*tang[2]);
  }

  // 3D linear case (3noded triangular element)
  else if (dt==tri3)
  {
    // no integration necessary (constant Jacobian)
    LINALG::SerialDenseMatrix coord(3,NumNode());
    GetNodalCoords(coord);

    // build vectors between the three nodes
    double t1[3] = {0.0, 0.0, 0.0};
    double t2[3] = {0.0, 0.0, 0.0};
    for (int k=0;k<3;++k)
    {
      t1[k]=coord(k,1)-coord(k,0);
      t2[k]=coord(k,2)-coord(k,0);
    }

    // cross product of t1 and t2
    double t1xt2[3] = {0.0, 0.0, 0.0};
    t1xt2[0] = t1[1]*t2[2]-t1[2]*t2[1];
    t1xt2[1] = t1[2]*t2[0]-t1[0]*t2[2];
    t1xt2[2] = t1[0]*t2[1]-t1[1]*t2[0];
    area=0.5*sqrt(t1xt2[0]*t1xt2[0]+t1xt2[1]*t1xt2[1]+t1xt2[2]*t1xt2[2]);
  }

  // 2D quadratic case   (3noded line element)
  // 3D bilinear case    (4noded quadrilateral element)
  // 3D quadratic case   (6noded triangular element)
  // 3D serendipity case (8noded quadrilateral element)
  // 3D biquadratic case (9noded quadrilateral element)
  else if (dt==line3 || dt==quad4 || dt==tri6 || dt==quad8 || dt==quad9)
  {
    // Gauss quadrature with correct NumGP and Dim
    MORTAR::ElementIntegrator integrator(dt);
    double detg = 0.0;

    // loop over all Gauss points, build Jacobian and compute area
    for (int j=0;j<integrator.nGP();++j)
    {
      double gpc[2] = {integrator.Coordinate(j,0), integrator.Coordinate(j,1)};
      detg = Jacobian(gpc);
      area+= integrator.Weight(j)*detg;
    }
  }

  // other cases not implemented yet
  else
    dserror("ERROR: Area computation not implemented for this type of MortarElement");

  return area;
}

/*----------------------------------------------------------------------*
 |  Get global coords for given local coords                  popp 01/08|
 *----------------------------------------------------------------------*/
bool MORTAR::MortarElement::LocalToGlobal(const double* xi, double* globcoord,
                                      int inttype)
{
  // check input
  if (!xi) dserror("ERROR: LocalToGlobal called with xi=NULL");
  if (!globcoord) dserror("ERROR: LocalToGlobal called with globcoord=NULL");

  // collect fundamental data
  int nnodes = NumNode();
  DRT::Node** mynodes = Nodes();
  if (!mynodes) dserror("ERROR: LocalToGlobal: Null pointer!");
  LINALG::SerialDenseMatrix coord(3,nnodes);
  LINALG::SerialDenseVector val(nnodes);
  LINALG::SerialDenseMatrix deriv(nnodes,2,true);

  // Evaluate shape, get nodal coords  and interpolate global coords
  EvaluateShape(xi, val, deriv, nnodes);
  for (int i=0;i<3;++i) globcoord[i]=0.0;

  for (int i=0;i<nnodes;++i)
  {
    MortarNode* mymrtrnode = static_cast<MortarNode*> (mynodes[i]);
    if (!mymrtrnode) dserror("ERROR: LocalToGlobal: Null pointer!");
    coord(0,i) = mymrtrnode->xspatial()[0];
    coord(1,i) = mymrtrnode->xspatial()[1];
    coord(2,i) = mymrtrnode->xspatial()[2];

    if (inttype==0)
    {
      // use shape function values for interpolation
      globcoord[0]+=val[i]*coord(0,i);
      globcoord[1]+=val[i]*coord(1,i);
      globcoord[2]+=val[i]*coord(2,i);
    }
    else if (inttype==1)
    {
      // use shape function derivatives xi for interpolation
      globcoord[0]+=deriv(i,0)*coord(0,i);
      globcoord[1]+=deriv(i,0)*coord(1,i);
      globcoord[2]+=deriv(i,0)*coord(2,i);
    }
    else if (inttype==2)
    {
      // use shape function derivatives eta for interpolation
      globcoord[0]+=deriv(i,1)*coord(0,i);
      globcoord[1]+=deriv(i,1)*coord(1,i);
      globcoord[2]+=deriv(i,1)*coord(2,i);
    }
    else
      dserror("ERROR: Invalid interpolation type requested, only 0,1,2!");
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Compute minimal edge size of MortarElement                popp 11/08|
 *----------------------------------------------------------------------*/
double MORTAR::MortarElement::MinEdgeSize(bool isinit)
{
  double minedgesize = 1.0e12;
  DRT::Element::DiscretizationType dt = Shape();

  // get coordinates of element nodes
  LINALG::SerialDenseMatrix coord(3,NumNode());
  GetNodalCoords(coord,isinit);

  // 2D case (2noded and 3noded line elements)
  if (dt==line2 || dt==line3)
  {
    // there is only one edge
    // (we approximate the quadratic case as linear)
    double diff[3] = {0.0, 0.0, 0.0};
    for (int k=0;k<3;++k)
      diff[k] = coord(k,1)-coord(k,0);
    minedgesize = sqrt(diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2]);
  }

  // 3D tri case (3noded and 6noded triangular elements)
  else if (dt==tri3 || dt==tri6)
  {
    // there are three edges
    // (we approximate the quadratic case as linear)
    for (int edge=0;edge<3;++edge)
    {
      double diff[3] = {0.0, 0.0, 0.0};
      for (int k=0;k<3;++k)
      {
        if (edge==2) diff[k] = coord(k,0)-coord(k,edge);
        else diff[k] = coord(k,edge+1)-coord(k,edge);
      }
      double dist = sqrt(diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2]);
      if (dist<minedgesize) minedgesize = dist;
    }
  }

  // 3D quad case (4noded, 8noded and 9noded quadrilateral elements)
  else if (dt==quad4 || dt==quad8 || dt==quad9)
  {
    // there are four edges
    // (we approximate the quadratic case as linear)
    for (int edge=0;edge<4;++edge)
    {
      double diff[3] = {0.0, 0.0, 0.0};
      for (int k=0;k<3;++k)
      {
        if (edge==3) diff[k] = coord(k,0)-coord(k,edge);
        else diff[k] = coord(k,edge+1)-coord(k,edge);
      }
      double dist = sqrt(diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2]);
      if (dist<minedgesize) minedgesize = dist;
    }
  }

  // invalid case
  else
    dserror("ERROR: MinEdgeSize not implemented for this type of MortarElement");

  if (minedgesize==1.0e12) dserror("ERROR: MinEdgeSize went wrong...!");
  return minedgesize;
}

/*----------------------------------------------------------------------*
 |  Compute maximal edge size of MortarElement                popp 11/08|
 *----------------------------------------------------------------------*/
double MORTAR::MortarElement::MaxEdgeSize(bool isinit)
{
  double maxedgesize = 0.0;
  DRT::Element::DiscretizationType dt = Shape();

  // get coordinates of element nodes
  LINALG::SerialDenseMatrix coord(3,NumNode());
  GetNodalCoords(coord,isinit);

  // 2D case (2noded and 3noded line elements)
  if (dt==line2 || dt==line3)
  {
    // there is only one edge
    // (we approximate the quadratic case as linear)
    double diff[3] = {0.0, 0.0, 0.0};
    for (int k=0;k<3;++k)
      diff[k] = coord(k,1)-coord(k,0);
    maxedgesize = sqrt(diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2]);
  }

  // 3D tri case (3noded and 6noded triangular elements)
  else if (dt==tri3 || dt==tri6)
  {
    // there are three edges
    // (we approximate the quadratic case as linear)
    for (int edge=0;edge<3;++edge)
    {
      double diff[3] = {0.0, 0.0, 0.0};
      for (int k=0;k<3;++k)
      {
        if (edge==2) diff[k] = coord(k,0)-coord(k,edge);
        else diff[k] = coord(k,edge+1)-coord(k,edge);
      }
      double dist = sqrt(diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2]);
      if (dist>maxedgesize) maxedgesize = dist;
    }
  }

  // 3D quad case (4noded, 8noded and 9noded quadrilateral elements)
  else if (dt==quad4 || dt==quad8 || dt==quad9)
  {
    // there are four edges
    // (we approximate the quadratic case as linear)
    for (int edge=0;edge<4;++edge)
    {
      double diff[3] = {0.0, 0.0, 0.0};
      for (int k=0;k<3;++k)
      {
        if (edge==3) diff[k] = coord(k,0)-coord(k,edge);
        else diff[k] = coord(k,edge+1)-coord(k,edge);
      }
      double dist = sqrt(diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2]);
      if (dist>maxedgesize) maxedgesize = dist;
    }
  }

  // invalid case
  else
    dserror("ERROR: MaxEdgeSize not implemented for this type of MortarElement");

  if (maxedgesize==0.0) dserror("ERROR: MaxEdgeSize went wrong...!");
  return maxedgesize;
}

/*----------------------------------------------------------------------*
 |  Add MortarElements to potential contact partners          popp 01/08|
 *----------------------------------------------------------------------*/
bool MORTAR::MortarElement::AddSearchElements(const vector<int>& gids)
{
  // check input data and calling element type
  if ((int)gids.size()==0)
    dserror("ERROR: AddSearchElements called with vec of length zero!");
  if (!IsSlave())
    dserror("ERROR: AddSearchElements called for non-slave MortarElement!");

  // loop over all input gids
  for (int i=0;i<(int)gids.size();++i)
  {
    // loop over all search candidates already known
    bool found = false;
    for (int j=0;j<NumSearchElements();++j)
      if (gids[i]==searchelements_[j])
        found = true;

    // add new gid to vector of search candidates
    if (!found) SearchElements().push_back(gids[i]);
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Add one MortarElement to potential contact partners       popp 01/08|
 *----------------------------------------------------------------------*/
bool MORTAR::MortarElement::AddSearchElements(const int & gid)
{
  // check input data and calling element type
  if (!IsSlave())
    dserror("ERROR: AddSearchElements called for non-slave MortarElement!");

  // add new gid to vector of search candidates
  SearchElements().push_back(gid);

  return true;
}

#endif  // #ifdef CCADISCRET
