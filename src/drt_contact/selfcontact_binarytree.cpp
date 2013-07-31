/*!----------------------------------------------------------------------
\file selfcontact_binarytree.cpp
\brief A class for performing self contact search in 2D / 3D based
       on binary search trees and dual graphs

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

#include "selfcontact_binarytree.H"
#include "contact_node.H"
#include "contact_element.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../linalg/linalg_fixedsizematrix.H"
#include "../linalg/linalg_utils.H"
#include <Teuchos_Time.hpp>

/*----------------------------------------------------------------------*
 |  ctor BinaryTreeNode for self contact (public)             popp 11/09|
 *----------------------------------------------------------------------*/
CONTACT::SelfBinaryTreeNode::SelfBinaryTreeNode(
                       SelfBinaryTreeNodeType type,
                       DRT::Discretization& discret,
                       Teuchos::RCP<SelfBinaryTreeNode> parent,
                       std::vector<int> elelist,
                       const Epetra_SerialDenseMatrix& dopnormals,
                       const Epetra_SerialDenseMatrix& samplevectors,
                       const int& kdop, const int& dim,
                       const int& nvectors, const int layer,
                       std::vector<std::vector<Teuchos::RCP<SelfBinaryTreeNode> > > & treenodes) :
type_(type),
idiscret_(discret),
parent_(parent),
elelist_(elelist),
dopnormals_(dopnormals),
samplevectors_(samplevectors),
kdop_(kdop),
dim_(dim),
nvectors_(nvectors),
layer_(layer),
treenodes_(treenodes)
{
  // reshape slabs matrix
  if (dim_==2)      slabs_.Reshape(kdop_/2,2);
  else if (dim_==3) slabs_.Reshape(kdop_/2,2);
  else              dserror("ERROR: Problem dimension must be 2D or 3D!");

  return;
}

/*----------------------------------------------------------------------*
 |  get communicator (public)                                 popp 11/09|
 *----------------------------------------------------------------------*/
const Epetra_Comm& CONTACT::SelfBinaryTreeNode::Comm() const
{
  return idiscret_.Comm();
}

/*----------------------------------------------------------------------*
 | complete the tree storage in a top down way (public)       popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::CompleteTree(int layer,double& enlarge)
{
  // calculate bounding volume
  CalculateSlabsDop(true);
  EnlargeGeometry(enlarge);

  // check root node (layer 0) for sanity
  if (layer == 0)
  {
    SetLayer(0);
    if (type_ == SELFCO_INNER) treenodes_[layer].push_back(Teuchos::rcp(this,false));
    else dserror("ERROR: root must be inner node in treenodes scheme");
  }

  // build treenode storage recursively
  if (type_==SELFCO_INNER)
  {
    leftchild_->SetLayer(layer_+1);
    rightchild_->SetLayer(layer_+1);

    // if map of treenodes does not have enogh rows-->resize!
    if ((int)(treenodes_.size()) <= (layer_+1))
      treenodes_.resize((layer_+2));

    // put new pointers to children into map
    treenodes_[(layer_+1)].push_back(leftchild_);
    treenodes_[(layer_+1)].push_back(rightchild_);

    rightchild_->CompleteTree(layer_+1,enlarge);
    leftchild_->CompleteTree(layer_+1,enlarge);
  }

  // do nothing if arrived at leaf level

 return;
}

/*----------------------------------------------------------------------*
 | Calculate booleans for qualified sample vectors (public)   popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::CalculateQualifiedVectors()
{
  if (type_!=SELFCO_LEAF) dserror("ERROR: Calculate qual. vec. called for non-leaf node!");

  // resize qualified vectors
  qualifiedvectors_.resize(nvectors_);

  // we first need the element center:
  // for line2, line3, quad4, quad8, quad9 elements: xi = eta = 0.0
  // for tri3, tri6 elements: xi = eta = 1/3
  CoElement* celement= static_cast<CoElement*>(idiscret_.gElement(elelist_[0]));
  double loccenter[2];

  DRT::Element::DiscretizationType dt = celement->Shape();
  if (dt==CoElement::tri3 || dt==CoElement::tri6)
  {
    loccenter[0] = 1.0/3;
    loccenter[1] = 1.0/3;
  }
  else if (dt==CoElement::line2 || dt==CoElement::line3 || dt==CoElement::quad4 ||
           dt==CoElement::quad8 || dt==CoElement::quad9)
  {
    loccenter[0] = 0.0;
    loccenter[1] = 0.0;
  }
  else dserror("ERROR: CalculateQualifiedVectors called for unknown element type");

  // now get the element center normal
  double normal[3] = {0.0, 0.0, 0.0};
  celement->ComputeUnitNormalAtXi(loccenter,normal);

  // check normal against sample vectors
  for (int i=0;i<(int)qualifiedvectors_.size();++i)
  {
    double scalar = (double)samplevectors_(i,0)*normal[0]
                  + (double)samplevectors_(i,1)*normal[1]
                  + (double)samplevectors_(i,2)*normal[2];
    if (scalar > 0) qualifiedvectors_[i] = true;
    else            qualifiedvectors_[i] = false;
  }

  return;
}

/*----------------------------------------------------------------------*
 | Update qualified sample vectors for inner node (public)    popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::UpdateQualifiedVectorsBottomUp()
{
  if(type_==SELFCO_LEAF) dserror("ERROR: Update qual. vec. called for leaf node!");

  // calculate the qualified vectors (= valid samplevectors) of an inner
  // treenode by comparing the qualified vectors of the children
  qualifiedvectors_.resize(nvectors_);

  for (int i=0;i<(int)qualifiedvectors_.size();++i)
    qualifiedvectors_.at(i) = ( (rightchild_->QualifiedVectors()).at(i)
                             && (leftchild_->QualifiedVectors()).at(i));

  return;
}

/*----------------------------------------------------------------------*
 | Update endnodes of one tree node (only 2D) (public)        popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::UpdateEndnodes()
{
  if(type_==SELFCO_LEAF) dserror("ERROR: Update endnodes. called for leaf node!");

  // reset endnodes
  endnodes_.clear();

  // find out which nodes the children have in commen, save others as endnodes
  if (leftchild_->endnodes_[0] == rightchild_->endnodes_[0]
   && leftchild_->endnodes_[1] != rightchild_->endnodes_[1])
  {
    endnodes_.push_back(rightchild_->endnodes_[1]);
    endnodes_.push_back(leftchild_->endnodes_[1]);
  }

  else if (leftchild_->endnodes_[1] == rightchild_->endnodes_[1]
        && leftchild_->endnodes_[0] != rightchild_->endnodes_[0])
  {
    endnodes_.push_back(rightchild_->endnodes_[0]);
    endnodes_.push_back(leftchild_->endnodes_[0]);
  }

  else if (leftchild_->endnodes_[0] == rightchild_->endnodes_[1]
        && leftchild_->endnodes_[1] != rightchild_->endnodes_[0])
  {
    endnodes_.push_back(rightchild_->endnodes_[0]);
    endnodes_.push_back(leftchild_->endnodes_[1]);
  }

  else if (leftchild_->endnodes_[1] == rightchild_->endnodes_[0]
        && leftchild_->endnodes_[0] != rightchild_->endnodes_[1] )
  {
    endnodes_.push_back(rightchild_->endnodes_[1]);
    endnodes_.push_back(leftchild_->endnodes_[0]);
  }

  else // the treenode is a closed surface (ring) -> no endnodes
  {
    endnodes_.push_back(-1);
    endnodes_.push_back(-1);
  }

  return;
}

/*----------------------------------------------------------------------*
 | Calculate slabs of DOP out of node postions (public)       popp 10/08|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::CalculateSlabsDop(bool isinit)
{
  // initialize slabs
  for (int j=0; j<kdop_/2; j++)
  {
    slabs_(j,0) =  1.0e12;
    slabs_(j,1) = -1.0e12;
  }

  // calculate slabs for every element
  for (int i=0; i<(int)elelist_.size();++i)
  {
    int gid = Elelist()[i];
    DRT::Element* element= idiscret_.gElement(gid);
    if (!element) dserror("ERROR: Cannot find element with gid %\n",gid);
    CoElement* celement=static_cast<CoElement*>(element);
    DRT::Node** nodes = celement->Nodes();
    if (!nodes) dserror("ERROR: Null pointer!");

    // calculate slabs for every node on every element
    for (int k=0;k<celement->NumNode();++k)
    {
      CoNode* cnode=static_cast<CoNode*>(nodes[k]);
      if (!cnode) dserror("ERROR: Null pointer!");

      // decide which position is relevant (initial or current)
      double pos[3] = {0.0, 0.0, 0.0};
      for (int j=0;j<dim_;++j)
      {
        if (isinit) pos[j] = cnode->X()[j];
        else        pos[j] = cnode->xspatial()[j];
      }

      // calculate slabs
      for(int j=0;j<kdop_/2;++j)
      {
        //= ax+by+cz=d/sqrt(aa+bb+cc)
        double num = dopnormals_(j,0)*pos[0]
                   + dopnormals_(j,1)*pos[1]
                   + dopnormals_(j,2)*pos[2];
        double denom = sqrt((dopnormals_(j,0)*dopnormals_(j,0))
                           +(dopnormals_(j,1)*dopnormals_(j,1))
                           +(dopnormals_(j,2)*dopnormals_(j,2)));
        double dcurrent = num/denom;

        if (dcurrent > slabs_(j,1)) slabs_(j,1) = dcurrent;
        if (dcurrent < slabs_(j,0)) slabs_(j,0) = dcurrent;
      }

      // if update for contactsearch --> add auxiliary positions
      if (!isinit)
      {
        // calculate element normal at current node
        double xi[2] = {0.0, 0.0};
        double normal[3] = {0.0, 0.0, 0.0};
        celement->LocalCoordinatesOfNode(k,xi);
        celement->ComputeUnitNormalAtXi(xi,normal);

        // now the auxiliary position
        double auxpos[3] = {0.0, 0.0, 0.0};
        double scalar = 0.0;
        for (int j=0;j<dim_;++j)
          scalar = scalar + (cnode->X()[j]+cnode->uold()[j]-cnode->xspatial()[j])*normal[j];
        for (int j=0;j<dim_;++j)
          auxpos[j] = cnode->xspatial()[j] + scalar*normal[j];

        for (int j=0;j<kdop_/2;++j)
        {
          //= ax+by+cz=d/sqrt(aa+bb+cc)
          double num = dopnormals_(j,0)*auxpos[0]
                     + dopnormals_(j,1)*auxpos[1]
                     + dopnormals_(j,2)*auxpos[2];
          double denom = sqrt((dopnormals_(j,0)*dopnormals_(j,0))
                             +(dopnormals_(j,1)*dopnormals_(j,1))
                             +(dopnormals_(j,2)*dopnormals_(j,2)));
          double dcurrent = num/denom;

          if (dcurrent > slabs_(j,1)) slabs_(j,1) = dcurrent;
          if (dcurrent < slabs_(j,0)) slabs_(j,0) = dcurrent;
        }
      }
    }
  }

  //Prints slabs to std::cout
  //PrintSlabs();

  return;
}

/*----------------------------------------------------------------------*
 | Update slabs bottom-up (public)                            popp 10/08|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::UpdateSlabsBottomUp(double & enlarge)
{
  // if current treenode is inner node
  if (type_==SELFCO_INNER)
  {
    for (int k=0;k<kdop_/2;++k)
    {
      // for minimum
      if (leftchild_->Slabs()(k,0) <= rightchild_->Slabs()(k,0))
        slabs_(k,0) = leftchild_->Slabs()(k,0);
      else
        slabs_(k,0) = rightchild_->Slabs()(k,0);

      // for maximum
      if (leftchild_->Slabs()(k,1) >= rightchild_->Slabs()(k,1))
        slabs_(k,1) = leftchild_->Slabs()(k,1);
      else
        slabs_(k,1) = rightchild_->Slabs()(k,1);
    }
  }

  // if current treenode is leaf node
  if (type_==SELFCO_LEAF)
  {
    // initialize slabs
    for (int j=0;j<kdop_/2;++j)
    {
      slabs_(j,0) =  1.0e12;
      slabs_(j,1) = -1.0e12;
    }

    int gid = Elelist()[0];
    DRT::Element* element= idiscret_.gElement(gid);
    if (!element) dserror("ERROR: Cannot find element with gid %\n",gid);
    CoElement* celement=static_cast<CoElement*>(element);
    DRT::Node** nodes = celement->Nodes();
    if (!nodes) dserror("ERROR: Null pointer!");

    // update slabs for every node
    for (int k=0;k<celement->NumNode();++k)
    {
      CoNode* cnode=static_cast<CoNode*>(nodes[k]);
      if (!cnode) dserror("ERROR: Null pointer!");

      // decide which position is relevant (initial or current)
      double pos[3] = {0.0, 0.0, 0.0};
      for (int j=0;j<dim_;++j) pos[j] = cnode->xspatial()[j];

      // calculate slabs
      for (int j=0;j<kdop_/2;++j)
      {
        //= ax+by+cz=d/sqrt(aa+bb+cc)
        double num = dopnormals_(j,0)*pos[0]
                   + dopnormals_(j,1)*pos[1]
                   + dopnormals_(j,2)*pos[2];
        double denom = sqrt((dopnormals_(j,0)*dopnormals_(j,0))
                           +(dopnormals_(j,1)*dopnormals_(j,1))
                           +(dopnormals_(j,2)*dopnormals_(j,2)));
        double dcurrent = num/denom;

        if (dcurrent > slabs_(j,1)) slabs_(j,1) = dcurrent;
        if (dcurrent < slabs_(j,0)) slabs_(j,0) = dcurrent;
      }

      // enlarge slabs with aux. position
      // first calculate element normal at current node
      double xi[2] = {0.0, 0.0};
      double normal[3] = {0.0, 0.0, 0.0};
      celement->LocalCoordinatesOfNode(k,xi);
      celement->ComputeUnitNormalAtXi(xi,normal);

      // now the auxiliary position
      double auxpos[3] = {0.0, 0.0, 0.0};
      double scalar = 0.0;
      for (int j=0;j<dim_;++j)
        scalar = scalar + (cnode->X()[j]+cnode->uold()[j]-cnode->xspatial()[j])*normal[j];
      for (int j=0;j<dim_;++j)
        auxpos[j] = cnode->xspatial()[j] + scalar*normal[j];

      for (int j=0;j<kdop_/2;++j)
      {
        //= ax+by+cz=d/sqrt(aa+bb+cc)
        double num = dopnormals_(j,0)*auxpos[0]
                   + dopnormals_(j,1)*auxpos[1]
                   + dopnormals_(j,2)*auxpos[2];
        double denom = sqrt((dopnormals_(j,0)*dopnormals_(j,0))
                           +(dopnormals_(j,1)*dopnormals_(j,1))
                           +(dopnormals_(j,2)*dopnormals_(j,2)));
        double dcurrent = num/denom;

        if (dcurrent > slabs_(j,1)) slabs_(j,1) = dcurrent;
        if (dcurrent < slabs_(j,0)) slabs_(j,0) = dcurrent;
      }
    }

    for (int i=0;i<kdop_/2;++i)
    {
      slabs_(i,0) = slabs_(i,0) - enlarge;
      slabs_(i,1) = slabs_(i,1) + enlarge;
    }

    //Prints slabs to std::cout
    //PrintSlabs();
  }

  return;
}

/*----------------------------------------------------------------------*
 | Print type of treenode to std::cout (public)               popp 10/08|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::PrintType()
{
  if (type_ == SELFCO_INNER)
    std::cout << std::endl << "SELFCO_INNER ";
  else if (type_ == SELFCO_LEAF)
    std::cout << std::endl << "SELFCO_LEAF ";
  else if (type_ == SELFCO_NO_ELEMENTS)
    std::cout << std::endl << "TreeNode contains no elements = SELFCO_NO_ELEMENTS ";
  else
    std::cout << std::endl << "SELFCO_UNDEFINED ";

  return;
}

/*----------------------------------------------------------------------*
 | Print slabs to std::cout (public)                          popp 10/08|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::PrintSlabs()
{
   std::cout << std::endl << Comm().MyPID() << "************************************************************";
   PrintType();
   std::cout << "slabs:";
   for (int i=0;i<slabs_.M();++i)
     std::cout << "\nslab: " << i << " min: " << slabs_.operator ()(i,0) << " max: " << slabs_.operator ()(i,1);
   std::cout << "\n**********************************************************\n";

   return;
}

/*----------------------------------------------------------------------*
 | Set slabs of current treenode with new slabs (public)      popp 10/08|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::SetSlabs(Epetra_SerialDenseMatrix& newslabs)
{
  for (int i=0;i<kdop_/2;++i)
  {
    slabs_(i,0) = newslabs(i,0);
    slabs_(i,1) = newslabs(i,1);
  }

  return;
}

/*----------------------------------------------------------------------*
 | Set children of current treenode (public)                  popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::SetChildren(Teuchos::RCP<SelfBinaryTreeNode> leftchild,
                                              Teuchos::RCP<SelfBinaryTreeNode> rightchild)
{
  leftchild_= leftchild;
  rightchild_ = rightchild;

  return;
}

/*----------------------------------------------------------------------*
 | Enlarge geometry of treenode (public)                      popp 10/08|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTreeNode::EnlargeGeometry(double& enlarge)
{
  // scale slabs with scalar enlarge
  for (int i=0;i<kdop_/2;++i)
  {
    slabs_(i,0) = slabs_(i,0) - enlarge;
    slabs_(i,1) = slabs_(i,1) + enlarge;
  }

  return;
}

/*----------------------------------------------------------------------*
 | Constructor SelfDualEdge (public)                              popp 11/09|
 *----------------------------------------------------------------------*/
CONTACT::SelfDualEdge::SelfDualEdge(Teuchos::RCP<SelfBinaryTreeNode> node1,
                            Teuchos::RCP<SelfBinaryTreeNode> node2,
                            const int& dim) :
node1_(node1),
node2_(node2),
dim_(dim)
{
  // directly move on to cost function
  CalculateCosts();

  return;
}

/*----------------------------------------------------------------------*
 | Calculate cost function value (public)                     popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfDualEdge::CalculateCosts()
{
  // update slabs of both dual edge nodes
  double enlarge = 0.0;
  node1_->UpdateSlabsBottomUp(enlarge);
  node2_->UpdateSlabsBottomUp(enlarge);

  // build parent slab for dual edge
  Epetra_SerialDenseMatrix parentslabs;
  int n = node1_->Kdop();
  parentslabs.Reshape(n/2,2);

  for (int k=0;k<n/2;++k)
  {
    //for minimum
    if (node1_->Slabs()(k,0) <= node2_->Slabs()(k,0))
      parentslabs(k,0) = node1_->Slabs()(k,0);
    else
      parentslabs(k,0) = node2_->Slabs()(k,0);

    // for maximum
    if (node1_->Slabs()(k,1) >= node2_->Slabs()(k,1))
      parentslabs(k,1) = node1_->Slabs()(k,1);
    else
      parentslabs(k,1) = node2_->Slabs()(k,1);
  }

  // compute maximal k-DOP length for dual edge
  double lmaxdop = 0.0;
  int slab = 0;
  for (int i=0;i<n/2;++i)
  {
     double lcurrent = abs(parentslabs(i,1)-parentslabs(i,0));
     if (lmaxdop < lcurrent)
     {
        lmaxdop = lcurrent;
        slab=i;
     }
  }

  // two-dimensional case
  if (dim_==2)
  {
    // compute total length of dual edge
    double lele = 0.0;

    for (int l=0;l<(int)(node1_->Elelist().size());++l)
    {
      int gid = (node1_->Elelist()).at(l);
      CoElement* celement= static_cast<CoElement*> (node1_->Discret().gElement(gid));
      lele = lele + celement->MaxEdgeSize();
    }

    for (int l=0;l<(int)(node2_->Elelist().size());++l)
    {
      int gid = node2_->Elelist()[l];
      CoElement* celement= static_cast<CoElement*> (node2_->Discret().gElement(gid));
      lele = lele + celement->MaxEdgeSize();
    }

    // cost function = nele * ( L / L_max )
    int nele = node1_->Elelist().size() + node2_->Elelist().size();
    costs_= nele * (lele/lmaxdop);
  }

  // three-dimensional case
  else
  {
    // compute total area of dual edge
    double area = 0.0;

    for (int l=0;l<(int)(node1_->Elelist().size());++l)
    {
      int gid = (node1_->Elelist()).at(l);
      CoElement* celement= static_cast<CoElement*> (node1_->Discret().gElement(gid));
      area = area + celement->MoData().Area();
    }

    for (int l=0;l<(int)(node2_->Elelist().size());++l)
    {
      int gid = node2_->Elelist()[l];
      CoElement* celement= static_cast<CoElement*> (node2_->Discret().gElement(gid));
      area = area + celement->MoData().Area();
    }

    // compute maximal k-DOP area for dual edge
    double lmaxdop2 = 0.0;
    Epetra_SerialDenseMatrix dopnormals = node1_->Dopnormals();
    for (int j=0;j<n/2;++j)
    {
      double scalar = dopnormals(j,0)*dopnormals(slab,0)
                    + dopnormals(j,1)*dopnormals(slab,1)
                    + dopnormals(j,2)*dopnormals(slab,2);

      if (scalar==0)
      {
         double lcurrent2 = abs(parentslabs(j,1)-parentslabs(j,0));
         if (lmaxdop2 < lcurrent2) lmaxdop2 = lcurrent2;
      }
    }
    double doparea = lmaxdop * lmaxdop2;

    // cost function = nele * ( A / A_max )
    int nele = node1_->Elelist().size() + node2_->Elelist().size();
    costs_ = nele * nele * (area/doparea);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  ctor SelfBinaryTree (public)                              popp 11/09|
 *----------------------------------------------------------------------*/
CONTACT::SelfBinaryTree::SelfBinaryTree(DRT::Discretization& discret,
                                        const Epetra_Comm* lcomm,
                                        Teuchos::RCP<Epetra_Map> elements,
                                        int dim, double eps) :
idiscret_(discret),
lcomm_(lcomm),
elements_(elements),
dim_(dim),
eps_(eps)
{
  // get out of here if not participating in interface
  if (!lComm()) return;

  // initialize sizes
  treenodes_.resize(1);
  leafsmap_.clear();

  // calculate min. element length and set enlargement accordingly
  SetEnlarge(true);

  //**********************************************************************
  // check for problem dimension
  //**********************************************************************
  // two-dimensional case
  if (dim_==2)
  {
    // set number of DOP sides
    kdop_=8;

    // set number of sample vectors
    nvectors_=16;

    // setup normals for DOP
    dopnormals_.Reshape(4,3);
    dopnormals_(0,0)= 1; dopnormals_(0,1)= 0; dopnormals_(0,2)= 0;
    dopnormals_(1,0)= 0; dopnormals_(1,1)= 1; dopnormals_(1,2)= 0;
    dopnormals_(2,0)= 1; dopnormals_(2,1)= 1; dopnormals_(2,2)= 0;
    dopnormals_(3,0)=-1; dopnormals_(3,1)= 1; dopnormals_(3,2)= 0;

    // setup sample vectors
    samplevectors_.Reshape(16,3);
    samplevectors_(0,0)= 1.0; samplevectors_(0,1)= 0.0; samplevectors_(0,2)= 0.0;
    samplevectors_(8,0)=-1.0; samplevectors_(8,1)= 0.0; samplevectors_(8,2)= 0.0;

    for (int i=1;i<4;++i)
    {
      samplevectors_(i,0)= cos(M_PI*(double)i/8);
      samplevectors_(i,1)= sin(M_PI*(double)i/8);
      samplevectors_(i,2)= 0;

      samplevectors_(i+8,0)=   cos(M_PI*(double)i/8);
      samplevectors_(i+8,1)=-1*sin(M_PI*(double)i/8);
      samplevectors_(i+8,2)= 0;
    }

    samplevectors_(4,0)=  0.0; samplevectors_(4,1)=  1.0; samplevectors_(4,2)=  0.0;
    samplevectors_(12,0)= 0.0; samplevectors_(12,1)=-1.0; samplevectors_(12,2)= 0.0;

    for (int i=5;i<8;++i)
    {
       samplevectors_(i,0)= cos(M_PI*(double)i/8);
       samplevectors_(i,1)= sin(M_PI*(double)i/8);
       samplevectors_(i,2)= 0;

       samplevectors_(i+8,0)=   cos(M_PI*(double)i/8);
       samplevectors_(i+8,1)=-1*sin(M_PI*(double)i/8);
       samplevectors_(i+8,2)= 0;
    }
  }

  // three-dimensional case
  else if (dim_==3)
  {
    // set number of DOP sides
    kdop_=18;

    // set number of sample vectors
    nvectors_=50;

    // setup normals for DOP
    dopnormals_.Reshape(9,3);
    dopnormals_(0,0)= 1; dopnormals_(0,1)= 0; dopnormals_(0,2)= 0;
    dopnormals_(1,0)= 0; dopnormals_(1,1)= 1; dopnormals_(1,2)= 0;
    dopnormals_(2,0)= 0; dopnormals_(2,1)= 0; dopnormals_(2,2)= 1;
    dopnormals_(3,0)= 1; dopnormals_(3,1)= 1; dopnormals_(3,2)= 0;
    dopnormals_(4,0)= 1; dopnormals_(4,1)= 0; dopnormals_(4,2)= 1;
    dopnormals_(5,0)= 0; dopnormals_(5,1)= 1; dopnormals_(5,2)= 1;
    dopnormals_(6,0)= 1; dopnormals_(6,1)= 0; dopnormals_(6,2)=-1;
    dopnormals_(7,0)= 1; dopnormals_(7,1)=-1; dopnormals_(7,2)= 0;
    dopnormals_(8,0)= 0; dopnormals_(8,1)= 1; dopnormals_(8,2)=-1;

    // setup sample vectors
    samplevectors_.Reshape(50,3);
    samplevectors_(0,0)= 0; samplevectors_(0,1)= 0.0; samplevectors_(0,2)= 1.0;
    samplevectors_(1,0)= 0; samplevectors_(1,1)= 0.0; samplevectors_(1,2)=-1.0;

    for (int i=0;i<8;++i)
    {
      for (int j=1;j<4;++j)
      {
        samplevectors_(1+6*i+j,0)= sin(M_PI*(double)j/8)*cos(M_PI*(double)i/8);
        samplevectors_(1+6*i+j,1)= sin(M_PI*(double)j/8)*sin(M_PI*(double)i/8);
        samplevectors_(1+6*i+j,2)= cos(M_PI*(double)j/8);
      }
      for (int j=5;j<8;++j)
      {
        samplevectors_(6*i+j,0)= sin(M_PI*(double)j/8)*cos(M_PI*(double)i/8);
        samplevectors_(6*i+j,1)= sin(M_PI*(double)j/8)*sin(M_PI*(double)i/8);
        samplevectors_(6*i+j,2)= cos(M_PI*(double)j/8);
      }
    }
  }

  else dserror("ERROR: Problem dimension must be 2D or 3D!");

  //**********************************************************************
  // initialize binary tree leaf nodes
  //**********************************************************************
  // create element lists
  std::vector<int> elelist;
  std::vector<int> localelelist;

  // build global element list
  for (int i=0;i<elements_->NumMyElements();++i)
  {
    int gid = elements_->GID(i);
    elelist.push_back(gid);
  }

  if (elelist.size()<=1) dserror("ERROR: Less than 2 elements for binary tree initialization!");
  //std::cout << "Elelistsize: " << (int)elelist.size() << std::endl;


  // build local element list and create leaf nodes
  for (int i=0;i<(int)(elelist.size());++i)
  {
    localelelist.clear();
    localelelist.push_back(elelist[i]);
    Teuchos::RCP<SelfBinaryTreeNode> leaf = Teuchos::rcp(new SelfBinaryTreeNode(SELFCO_LEAF,idiscret_,Teuchos::null,localelelist,
       DopNormals(),SampleVectors(),Kdop(),Dim(),Nvectors(),-1,treenodes_));
    leafsmap_[elelist[i]]=leaf;
  }

  // double-check if there is at the least one leaf node in tree now
  if (leafsmap_.size()==0) dserror("ERROR: SelfBinaryTree: No contact elements defined!");

  //**********************************************************************
  // find adjacent tree nodes and build dual graph
  //**********************************************************************
  // initialize dual graph
  std::map<Teuchos::RCP<SelfDualEdge>, std::vector<Teuchos::RCP<SelfDualEdge> > > dualgraph;

  // loop over all self contact elements
  for (int i=0;i<(int)elelist.size();++i)
  {
    // global id of current element
    int gid = elelist[i];

    // initialize
    // ... vector of adjacent treenodes (elements) of current element
    // ... vector of global IDs of adjacent elements of current element
    // ... vector of adjacent dual edges containing current element
    // ... vector of global IDs of possible adjacent elements
    std::vector<Teuchos::RCP<SelfBinaryTreeNode> >  adjtreenodes;
    std::vector<int>                       adjids;
    std::vector<Teuchos::RCP<SelfDualEdge> >        adjdualedges;
    std::vector<int>                       possadjids;

    // get curent elements and its nodes
    DRT::Element* element= idiscret_.gElement(gid);
    if (!element) dserror("ERROR: Cannot find element with gid %\n",gid);
    DRT::Node** nodes = element->Nodes();
    if (!nodes) dserror("ERROR: Null pointer!");

    // first treenode of one dual edge which includes current element
    // is the element itself saved as a treenode
    Teuchos::RCP<SelfBinaryTreeNode> node1=leafsmap_[gid];

    // for 2D only: get the finite element nodes of the element
    // and save them as endnodes of the treenode
    if (dim_==2)
    {
      std::vector<int> nodeIds;
      nodeIds.push_back(element->NodeIds()[0]);
      nodeIds.push_back(element->NodeIds()[1]);
      node1->SetEndnodes(nodeIds);
    }

    // find all first-order nodes of current element
    // we exclude higher-order nodes (i.e. edge and center nodes)
    // in both 2D and 3D as they do not bring in any additional
    // information about connectivity / adjacency
    int numnode = 0;
    MORTAR::MortarElement* mele = static_cast<MORTAR::MortarElement*>(element);

    switch(mele->Shape())
    {
      case DRT::Element::line2:
      case DRT::Element::line3:
      {
        numnode = 2;
        break;
      }
      case DRT::Element::tri3:
      case DRT::Element::tri6:
      {
        numnode = 3;
        break;
      }
      case DRT::Element::quad4:
      case DRT::Element::quad8:
      case DRT::Element::quad9:
      {
        numnode = 4;
        break;
      }
      default:
      {
        dserror("ERROR: Unknown mortar element type");
        break;
      }
    }

    // loop over all first-order nodes of current element
    // (here we make use of the fact that first-order nodes
    // are always stored before higher-order nodes)
    for(int j=0; j<numnode;j++)
    {
      DRT::Node* node =nodes[j];
      if (!node) dserror("ERROR: Null pointer!");

      // adjacent elements of current node
      int numE =node->NumElement();
      DRT::Element** adjElements = node->Elements();
      if (!adjElements) dserror("ERROR: Null pointer!");

      // loop over all adjacent elements of current node
      for(int k=0;k<numE;++k)
      {
        int eleID=adjElements[k]->Id();

        // if eleID isn't the currently considered element, it could be a neighbour
        if (eleID!=gid)
        {
          // if there are not yet any possibly adjacent elements
          if (possadjids.size()==0)
          {
            // in 2D one common node implies adjacency
            if(dim_==2)
            {
              // get second node from leafsmap
              Teuchos::RCP<SelfBinaryTreeNode> node2 = leafsmap_[eleID];
              if (node2 == Teuchos::null) dserror("adjacent leaf tree node not found in leafsmap!!");

              // get the finite element nodes of the element equal to treenode 2
              // and save them as endnodes of the treenode
              std::vector<int> nodeIds;
              nodeIds.clear();
              nodeIds.push_back(adjElements[k]->NodeIds()[0]);
              nodeIds.push_back(adjElements[k]->NodeIds()[1]);
              node2->SetEndnodes( nodeIds );

              // add treenode in list of adjacent treenodes of current treenode
              adjtreenodes.push_back(node2);
              adjids.push_back(eleID);

              // create edge and add it to the list
              Teuchos::RCP<SelfDualEdge> edge = Teuchos::rcp(new SelfDualEdge(node1,node2,Dim()));
              adjdualedges.push_back(edge);
            }
            // in 3D adjacency is more complicated
            else
            {
              // get second node from leafsmap
              Teuchos::RCP<SelfBinaryTreeNode> node2 = leafsmap_[eleID];
              adjtreenodes.push_back(node2);
              possadjids.push_back(eleID);
            }
          }

          // if there are already possibly adjacent elements in possasdjids
          else
          {
            bool saved=false;

            // in 2D one common node implies adjacency
            if(dim_==2)
            {
              //get second node from leafsmap
              Teuchos::RCP<SelfBinaryTreeNode> node2 = leafsmap_[eleID];
              if (node2 == Teuchos::null) dserror("adjacent tree node not found in leafsmap!!");

              // get the finite element nodes of the element equal to treenode 2
              // and save them as endnodes of the treenode
              std::vector<int> nodeIds;
              nodeIds.push_back(adjElements[k]->NodeIds()[0]);
              nodeIds.push_back(adjElements[k]->NodeIds()[1]);
              node2->SetEndnodes(nodeIds);

              // add treenode in list of adjacent treenodes of current treenode
              adjtreenodes.push_back(node2);
              adjids.push_back(eleID);

              // create edge and add it to the list
              Teuchos::RCP<SelfDualEdge> edge = Teuchos::rcp(new SelfDualEdge(node1,node2,Dim()));
              adjdualedges.push_back(edge);
            }
            // in 3D adjacency is more complicated
            // (adjacent elements have at least 2 common nodes)
            else
            {
              // get second node from leafsmap
              Teuchos::RCP<SelfBinaryTreeNode> node2 = leafsmap_[eleID];

              for (int l=0;l<(int)possadjids.size();++l)
              {
                // check if possibly adjacent element is already in the list
                // if true, there are 2 common nodes, which means it is a neighbour
                if (eleID==possadjids[l])
                {
                  saved=true;
                  if (node2 == Teuchos::null) dserror("adjacent tree node not found in leafsmap!!");

                  // add treenode in list of adjacent treenodes of current treenode
                  // adjtreenodes.push_back(node2);
                  adjids.push_back(eleID);

                  // create edge and add it to the list
                  Teuchos::RCP<SelfDualEdge> edge = Teuchos::rcp(new SelfDualEdge(node1,node2,Dim()));
                  adjdualedges.push_back(edge);
                  break;
                }
              }

              // possibly adjacent element is not yet in the list --> add
              if (!saved)
              {
                possadjids.push_back(eleID);
                adjtreenodes.push_back(node2);
              }
            } // else 3D
          } // else possadjids empty
        } // if eleID!=gid
      } // all adjacent elements
    } // all nodes

    // add the vector of adjacent tree nodes to the adjcencymatrix
    // we only need the matrix in 3D, because in 2D the adjacencytest works by
    // comparing endnodes only
    if (dim_==3) adjacencymatrix_[gid]=adjtreenodes;

    //get adjacent dual edges
    for(int k=0;k<(int)adjdualedges.size();++k)
    {
      for(int j=0;j<(int)adjdualedges.size();++j)
      {
        if (j!=k)
        {
          adjdualedges[k]->CommonNode(adjdualedges[j]);
          dualgraph[adjdualedges[k]].push_back(adjdualedges[j]);
        }
      }
    }
  } // all elements

  // plot adjacencymatrix
  /*
  std::map<int,std::vector<Teuchos::RCP<SelfBinaryTreeNode> > > ::iterator iter2 = adjacencymatrix_.begin();
  std::map<int,std::vector<Teuchos::RCP<SelfBinaryTreeNode> > > ::iterator iter2_end = adjacencymatrix_.end();

  std::cout << "\n" << leafsmap_.size() << " elements in leafsmap\n";
  std::cout << adjacencymatrix_.size() << " elements in adjazencymatrix\n";


  while (iter2 != iter2_end)
  {
    std::cout << "element " << (*iter2).first << ": ";
    //int adj_cnt= 0 ;

    std::vector<Teuchos::RCP<SelfBinaryTreeNode > >  adj_ = (*iter2).second;
    std::cout << adj_.size() << " elements: ";
    for (int i=0;i<(int)adj_.size();++i)
      std::cout << adj_[i]->Elelist()[0] << " ";
    std::cout << "\n";
    ++iter2;
  }
  */

  // plot dual graph
  /*
  std::cout << "\n" <<leafsmap_.size() << " elements in leafmap\n";
  std::cout << dualgraph.size() << " edges in dual graph\n";

  std::map<Teuchos::RCP<SelfDualEdge>,std::vector<Teuchos::RCP<SelfDualEdge> > > ::iterator iter3 = dualgraph.begin();
  std::map<Teuchos::RCP<SelfDualEdge>,std::vector<Teuchos::RCP<SelfDualEdge> > > ::iterator iter3_end = dualgraph.end();

  std::cout << dualgraph.max_size() << " maximal\n";
  int cnt = 0;

  while (iter3 != iter3_end)
  {
    std::cout << "\n Kante " << cnt << ": " << ((*iter3).first)->GetNode1()->Elelist()[0] << " "
         << ((*iter3).first)->GetNode2()->Elelist()[0] << "\n";
    std::cout << "Kosten: " << ((*iter3).first)->Costs() << "\n";

    std::vector<Teuchos::RCP<SelfDualEdge> > edges = (*iter3).second;
    std::cout << edges.size() << " NachbarKanten:\n ";
    for (int i=0;i<(int)edges.size();++i)
    {
      std::cout << edges[i]->GetNode1()->Elelist()[0] << " ";
      std::cout << edges[i]->GetNode2()->Elelist()[0] << " ";
      std::cout << "Kosten: " << edges[i]->Costs() << " \n";
    }

    std::cout << "\n";
    ++iter3;
    ++cnt;
  }
  */

  // now initialze SelfBinaryTree in a bottom-up way based on dualgraph
  InitializeTreeBottomUp(&dualgraph);

  return;
}

/*----------------------------------------------------------------------*
 |  get communicator (public)                                 popp 11/09|
 *----------------------------------------------------------------------*/
const Epetra_Comm& CONTACT::SelfBinaryTree::Comm() const
{
  return idiscret_.Comm();
}

/*----------------------------------------------------------------------*
 | Find minimal length of contact elements (public)           popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::SetEnlarge(bool isinit)
{
  // get out of here if not participating in interface
  if (!lComm()) return;

  // minimal length of finite elements
  double lmin = 1.0e12;

  // calculate minimal length
  for (int i=0;i<elements_->NumMyElements();++i)
  {
    int gid = elements_->GID(i);
    DRT::Element* element = idiscret_.gElement(gid);
    if (!element) dserror("ERROR: Cannot find element with gid %\n",gid);
    CONTACT::CoElement* celement = (CoElement*) element;
    double mincurrent=celement->MinEdgeSize(isinit);
    if (mincurrent < lmin) lmin = mincurrent;
  }

  if (lmin<=0.0) dserror("ERROR: Minimal element length < 0!");

  // set the class variable
  enlarge_ = eps_ * lmin;

  return;
}

/*----------------------------------------------------------------------*
 | Initialize tree bottum-up based on dual graph (public)     popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::InitializeTreeBottomUp(std::map<Teuchos::RCP<SelfDualEdge>,std::vector<Teuchos::RCP<SelfDualEdge> > > *dualGraph)
{
  // vector collecting root nodes
  roots_.resize(0);
  //std::cout << "Initial dual graph size: " << (*dualGraph).size() << std::endl;

  //**********************************************************************
  // the idea is to empty the dual graph step by step
  //**********************************************************************
  while (!(*dualGraph).empty())
  {
    // get the edge with lowest costs (= the first edge in the dual graph as
    // the map is automatically sorted by the costs)  to contract it
    std::map <Teuchos::RCP<SelfDualEdge>,std::vector <Teuchos::RCP<SelfDualEdge> > > ::iterator iter = (*dualGraph).begin();
    Teuchos::RCP<SelfDualEdge> contractedEdge = (*iter).first;
    Teuchos::RCP<SelfBinaryTreeNode> node1 = contractedEdge->GetNode1();
    Teuchos::RCP<SelfBinaryTreeNode> node2 = contractedEdge->GetNode2();

    // combine list of elements of both tree nodes to get new list
    std::vector<int> list = node1->Elelist();
    std::vector<int> list2 = node2->Elelist();
    for (int i=0;i<(int)(node2->Elelist().size());++i)
      list.push_back(list2[i]);

    // define new tree node
    Teuchos::RCP<SelfBinaryTreeNode> newNode = Teuchos::rcp(new SelfBinaryTreeNode(SELFCO_INNER,idiscret_,Teuchos::null,list,
                            DopNormals(),SampleVectors(),Kdop(),Dim(),Nvectors(),-1,treenodes_));
    newNode->SetChildren(node1,node2);
    node1->SetParent(newNode);
    node2->SetParent(newNode);

    // in 2D we simply save the endnodes as adjacency criterion
    if (dim_==2) newNode->UpdateEndnodes();

    // update dualGraph
    // this means we have to create new edges, which include the new treenode and
    // delete the edges, which are adjacent to the contracted edge and update
    // their neighbours
    // additionally we have to check if the tree is nearly complete

    // get the adjacent edges of the contracted edge
    std::vector<Teuchos::RCP<SelfDualEdge> > adjEdges = (*iter).second;

    // check if the new treenode includes whole selfcontact-surface
    // in this case the treenode has saved itself as adjacent edge
    if (adjEdges[0] == contractedEdge)
    {
      //std::cout << "Found root node (size " << (int)newNode->Elelist().size() << ")" << std::endl;
      //std::cout << "Elelist: ";
      //for (int a=0;a<(int)newNode->Elelist().size();++a)
      //  std::cout << newNode->Elelist()[a] << " ";
      //std::cout << std::endl;

      // save the tree node as root and continue the loop
      roots_.push_back(newNode);
      (*dualGraph).erase(contractedEdge);
      continue;
    }

    //std::cout << "Found non-root node (size " << (int)newNode->Elelist().size() << ")" << std::endl;
    //std::cout << "Elelist: ";
    //for (int a=0;a<(int)newNode->Elelist().size();++a)
    //  std::cout << newNode->Elelist()[a] << " ";
    //std::cout << std::endl;

    // vector of all new edges
    std::vector<Teuchos::RCP<SelfDualEdge> > newAdjEdges;

    // when contracting an edge, we need to update all adjacent edges
    for (int j=0;j<(int)adjEdges.size();++j)
    {
      // define new edge
      Teuchos::RCP<SelfDualEdge> newEdge;

      if (adjEdges[j]->GetNode1() != node1 && adjEdges[j]->GetNode1() !=  node2)
        newEdge = Teuchos::rcp(new SelfDualEdge(newNode,adjEdges[j]->GetNode1(),Dim()));

      else if (adjEdges[j]->GetNode2() != node1 && adjEdges[j]->GetNode2() !=  node2)
        newEdge = Teuchos::rcp(new SelfDualEdge(newNode,adjEdges[j]->GetNode2(),Dim()));

      else dserror("ERROR: Tried to contract identical tree nodes!!");

      // get the neighbours of the new edges
      std::vector<Teuchos::RCP<SelfDualEdge> > adjEdgesOfNeighbour = (*dualGraph)[adjEdges[j]];

      // if there are two adjacent surfaces left,
      // the new edge contains the whole self-contact surface.
      // in this case we save the edge as neighbour of itself
      if (adjEdges.size()==1 && adjEdgesOfNeighbour.size()==1)
        (*dualGraph)[newEdge].push_back(newEdge);

      // otherwise we need to update the neighbours of the adjacent edges:
      // first add all neighbours of the current adjacent edge -except the contracted edge-
      // as neighbour of newedge and delete the old adjacent edge
      else
      {
        // two-dimensional case
        if (dim_==2)
        {
          // in 2D every edge has 2 neighbours at the most
          newAdjEdges.push_back(newEdge);
          for (int k=0;k<(int)adjEdgesOfNeighbour.size();++k)
          {
            if (adjEdgesOfNeighbour[k] != contractedEdge)
            {
              // save the neighbours of the old edge as neighbours of the new one
              (*dualGraph)[newEdge].push_back(adjEdgesOfNeighbour[k]);

              // we have to update all neighbours of our old edge
              std::map<Teuchos::RCP<SelfDualEdge>,std::vector<Teuchos::RCP<SelfDualEdge> > > ::iterator edge_iter = (*dualGraph).find(adjEdgesOfNeighbour[k]);

              // find the old edge in the list of neighbours and replace it by the new edge
              for (int a=0;a<(int)edge_iter->second.size();++a)
              {
                if (edge_iter->second.at(a) == adjEdges[j])
                  edge_iter->second.at(a) = newEdge;
              }
            }
          }
        }
        // three-dimensional case
        else
        {
          // in 3D every edge can have a arbirary number of neighbours which
          // could also be adjacent to each other (3 edges form a "ring").
          // We have to check if the new edge has already been created.
          // (in 2D this occurs only when the tree is almost finished, in 3D
          // this could happen any time)

          bool issaved = false;
          for (int n=0;n<(int)newAdjEdges.size();++n)
          {
            // here we make use of the customized == operator in dual edge
            if (newAdjEdges[n]==newEdge)
            {
              issaved=true;
              break;
            }
          }

          // save the neighbours of the old edge as neighbours of the new one
          if (!issaved) newAdjEdges.push_back(newEdge);

          // we have to update all neighbours of the old/new edge
          // we just update those edges wich aren't adjacent edges of the contracted
          // edge, as we have to update those seperately

          // get the common (tree)node of the contracted edge and the old edge
          Teuchos::RCP<SelfBinaryTreeNode> commonnode = contractedEdge->CommonNode(adjEdges[j]);

          // loop over all neighbours of old edge
          for(int k=0;k<(int)adjEdgesOfNeighbour.size();++k)
          {
            // check if it the current edge is not the contracted edge
            // or a neigbour of the contracted edge (we do not need to update
            // these edges now)
            if (adjEdgesOfNeighbour[k] != contractedEdge &&
                adjEdges[j]->CommonNode(adjEdgesOfNeighbour[k]) != commonnode )
            {
              // if the current edge (=neighbour of the new and old edge) is not a
              // neighbour of the contracted edge, they do not have a common node
              Teuchos::RCP<SelfBinaryTreeNode> commonnode2 = contractedEdge->CommonNode(adjEdgesOfNeighbour[k]);

              // now we want to save the current edge as neighbour of the new edge
              if (commonnode2==Teuchos::null)
              {
                // first find the new edge in the dual graph
                std::map<Teuchos::RCP<SelfDualEdge>,std::vector<Teuchos::RCP<SelfDualEdge> > > ::iterator edge_iter1 = (*dualGraph).find(newEdge);
                std::map<Teuchos::RCP<SelfDualEdge>,std::vector<Teuchos::RCP<SelfDualEdge> > > ::iterator end =(*dualGraph).end();

                // if the edge has been found, check if the current edge has
                // already been saved as neighbour
                if (edge_iter1 != end)
                {
                  bool edgesaved = false;
                  for(int z=0;z<(int)edge_iter1->second.size();++z)
                  {
                    if (edge_iter1->second.at(z) == adjEdgesOfNeighbour[k])
                      edgesaved=true;
                  }
                  // if not yet saved, save it
                  if (!edgesaved) edge_iter1->second.push_back(adjEdgesOfNeighbour[k]);
                }

                // if the new edge itself hasn't been saved yet,
                // the neighbour hasn't been saved either
                else  (*dualGraph)[newEdge].push_back(adjEdgesOfNeighbour[k]);

                // find the old edge in the list of neighbours and replace it
                // by the new edge

                // find the current edge in the dual graph
                std::map<Teuchos::RCP<SelfDualEdge>,std::vector<Teuchos::RCP<SelfDualEdge> > > ::iterator edge_iter2 = (*dualGraph).find(adjEdgesOfNeighbour[k]);

                bool egdeerased = false;
                bool newedgesaved = false;

                // loop over all neighbours of the current edge
                std::vector<Teuchos::RCP<SelfDualEdge> > ::iterator adjIter = edge_iter2->second.begin();
                while (adjIter != edge_iter2->second.end())
                {
                  if (*adjIter == adjEdges[j])
                  {
                    // erase the old edge
                    adjIter = edge_iter2->second.erase(adjIter);
                    egdeerased = true;
                  }
                  else
                  {
                    if (*adjIter == newEdge) newedgesaved=true;
                    ++adjIter;
                  }
                }

                // as we could update the same edge several times (only in 3D),
                // we only save the new edge as neighbour if not already done so
                if (egdeerased && !newedgesaved) edge_iter2->second.push_back(newEdge);
              }
            }
          } // loop over all adjacent edges of neighbours
        } // 3D
      } // else-block (not 2 adjacent treenodes left)
    } // loop over all adjacent edges

    // check for a ring of three dual edges in 3D
    // (i.e. the edge to be contracted and two adjacent edges)
    // (in 2D there is no need to treat this case separately)
    if (adjEdges.size()==2 && dim_==3)
    {
      // pointers to adjacent edge nodes
      Teuchos::RCP<SelfBinaryTreeNode> anode1 = adjEdges[0]->GetNode1();
      Teuchos::RCP<SelfBinaryTreeNode> anode2 = adjEdges[0]->GetNode2();
      Teuchos::RCP<SelfBinaryTreeNode> bnode1 = adjEdges[1]->GetNode1();
      Teuchos::RCP<SelfBinaryTreeNode> bnode2 = adjEdges[1]->GetNode2();

      // check for ring (eight possible combinations)
      if ((node1==anode1 && node2==bnode1 && anode2==bnode2) ||
          (node1==anode2 && node2==bnode1 && anode1==bnode2) ||
          (node1==anode1 && node2==bnode2 && anode2==bnode1) ||
          (node1==anode2 && node2==bnode2 && anode1==bnode1) ||
          (node1==bnode1 && node2==anode1 && bnode2==anode2) ||
          (node1==bnode2 && node2==anode1 && bnode1==anode2) ||
          (node1==bnode1 && node2==anode2 && bnode2==anode1) ||
          (node1==bnode2 && node2==anode2 && bnode1==anode1))
      {
        // check for inconsistency
        if (newAdjEdges.size()!=1) dserror("ERROR: Inconsistent 3D ring in dual graph");

        // check if the resulting edge already exists in dual graph
        std::map<Teuchos::RCP<SelfDualEdge>,std::vector<Teuchos::RCP<SelfDualEdge> > > ::iterator edge_iter3 = (*dualGraph).find(newAdjEdges[0]);
        std::map<Teuchos::RCP<SelfDualEdge>,std::vector<Teuchos::RCP<SelfDualEdge> > > ::iterator end =(*dualGraph).end();

        // add if not so
        if (edge_iter3 == end) (*dualGraph)[newAdjEdges[0]].push_back(newAdjEdges[0]);
      }
    }

    // delete all adjacent edges of contracted edge
    for (int j=0;j<(int)adjEdges.size();++j)
      (*dualGraph).erase(adjEdges[j]);

    // now all new adjacent edges have been created.
    // save all adjacent edges as neigbour, respectively
    for (int l=0;l<(int)newAdjEdges.size();++l)
    {
      for (int m=0;m<(int)newAdjEdges.size();++m)
        if (l!=m && newAdjEdges[l]!=newAdjEdges[m])
          (*dualGraph)[ newAdjEdges[l] ].push_back(newAdjEdges[m]);
    }

    // delete the contracted edge
    (*dualGraph).erase(contractedEdge);

  } // while(!(*dualGraph).empty())
  //**********************************************************************

  // complete the tree starting from its roots (top-down)
  if ((int)roots_.size() == 0) dserror("ERROR: No root treenode found!");
  for (int k=0;k<(int)roots_.size();++k)
    roots_[k]->CompleteTree(0,enlarge_);

  // output to screen
  if (Comm().MyPID()==0)
    std::cout << "\nFound " << (int)roots_.size() << " root node(s) for binary tree." << std::endl;

  // in 3D we have to calculate adjacent treenodes
  if (dim_== 3)
  {
    CalculateAdjacentLeaves();
    CalculateAdjacentTnodes();
  }

  /*
  // debug output
  if (Comm().MyPID()==0)
  {
    // print roots
    for (int k=0;k<(int)roots_.size();++k)
    {
      std::vector<int> rootElelist = roots_[k]->Elelist();
      std::cout << "\nRoot " << k << " (size " << (int)rootElelist.size() << "): ";
      for (int d=0;d<(int)rootElelist.size();++d)
        std::cout << rootElelist[d] << " ";
      std::cout << "\n";
    }

    // print tree
    for (int i=0;i<(int)(treenodes_.size());++i)
    {
      std::cout << "\n Tree at layer: " << i << " Elements: ";
      for (int k=0;k<(int)(treenodes_[i].size());++k)
      {
        Teuchos::RCP<SelfBinaryTreeNode> currentnode = treenodes_[i][k];
        std::cout << " (";
        for (int l=0;l<(int)(currentnode->Elelist().size());++l)
        {
          std::cout << currentnode->Elelist().at(l) << " ";
          if (currentnode->Type()== SELFCO_LEAF) std::cout << "(Leaf) ";
        }
        std::cout << ") ";
      }
    }
    std::cout << std::endl;
  }
  */

  return;
}

/*----------------------------------------------------------------------*
 | Set adjacent treenodes of leaf-nodes in lowest layer (3D)  popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::CalculateAdjacentLeaves()
{
  // get the adjacent treenodes of each treenode in the lowest layer
  // and save the adjacent leafs which are in the same layer
  int maxlayer = treenodes_.size()-1;
  std::map<int, Teuchos::RCP<SelfBinaryTreeNode> > ::iterator leafiter = leafsmap_.begin();
  std::map<int, Teuchos::RCP<SelfBinaryTreeNode> > ::iterator leafiter_end = leafsmap_.end();

  // loop over all leaf treenodes
  while (leafiter != leafiter_end)
  {
    // do only if in lowest layer
    if (leafiter->second->Layer() == maxlayer)
    {
      std::vector<Teuchos::RCP<SelfBinaryTreeNode> > adjtnodessamelayer;
      std::vector<Teuchos::RCP<SelfBinaryTreeNode> > adjtnodes=adjacencymatrix_[leafiter->first];

      // search for adjacent treenodes in lowest layer
      for (int i=0;i<(int)adjtnodes.size();++i)
      {
        if (adjtnodes[i]->Layer() == maxlayer)
          adjtnodessamelayer.push_back(adjtnodes[i]);
      }

      // store in current treenode
      leafiter->second->SetAdjacentTnodes(adjtnodessamelayer);
    }

    // increment iterator
    ++leafiter;
  }

  return;
}

/*----------------------------------------------------------------------*
 | Set adjacent treenodes of the whole tree (3D)              popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::CalculateAdjacentTnodes()
{
  // calculate adjacent treenodes in the same layer of the each treenode
  // above leaf-layer in a bottom up way, so that the adjacent treenodes of
  // the lowest layer MUST have been calculated before (see above)
  int maxlayer = treenodes_.size();

  // loop over all layers (bottom-up, starting in 2nd lowest layer)
  for (int i=maxlayer-2; i>=0; --i)
  {
    // loop over all treenodes of this layer
    for(int j=0; j< (int)treenodes_.at(i).size();j++)
    {
      // vector of adjacent treenodes
      std::vector<Teuchos::RCP<SelfBinaryTreeNode> > adjtnodes;

      //******************************************************************
      // CASE 1: treenode is an inner node
      //******************************************************************
      if (treenodes_[i][j]->Type() != SELFCO_LEAF)
      {
        // get the adjacent treenodes of the children
        std::vector<Teuchos::RCP<SelfBinaryTreeNode> > adjofleftchild = treenodes_[i][j]->Leftchild()->AdjacentTreenodes();
        std::vector<Teuchos::RCP<SelfBinaryTreeNode> > adjofrightchild = treenodes_[i][j]->Rightchild()->AdjacentTreenodes();

        // check the adjacent treenodes of the left child
        for (int k=0;k<(int)adjofleftchild.size();++k)
        {
          // check if the parent of the adjacent node of the child has already been saved
          if (adjofleftchild[k] != treenodes_[i][j]->Rightchild())
          {
            bool issaved = false;
            for (int l=0;l<(int)adjtnodes.size();++l)
            {
              if (adjtnodes[l] == Teuchos::null) dserror("Teuchos::null pointer");
              if (adjofleftchild[k]->Parent() == adjtnodes[l])
              {
                issaved=true;
                break;
              }
            }

            if (!issaved) adjtnodes.push_back(adjofleftchild[k]->Parent());
          }
        }

        // check the adjacent treenodes of the right child
        for (int k=0;k<(int)adjofrightchild.size();++k)
        {
          // check if the parent of the adjacent node of the child has already been saved
          if (adjofrightchild[k] != treenodes_[i][j]->Leftchild())
          {
            bool issaved = false;
            for (int m=0;m<(int)adjtnodes.size();++m)
            {
              if (adjtnodes[m] == Teuchos::null) dserror("Teuchos::null pointer");
              if (adjofrightchild[k]->Parent() == adjtnodes[m])
              {
                issaved=true;
                break;
              }
            }

            if(!issaved) adjtnodes.push_back(adjofrightchild[k]->Parent());
          }
        }

        // finally set adjacent treenodes of current treenode
        treenodes_[i][j]->SetAdjacentTnodes(adjtnodes);
      }

      //******************************************************************
      // CASE 2: treenode is a leaf node above the lowest layer
      //******************************************************************
      else
      {
        // get the adjacent leaf nodes from the adjacencymatrix
        int gid = treenodes_[i][j]->Elelist()[0];
        if (adjacencymatrix_.find(gid)==adjacencymatrix_.end()) dserror("element not in adjacencymatrix!!");
        std::vector<Teuchos::RCP<SelfBinaryTreeNode> > adjleafs = adjacencymatrix_[gid];

        // loop over all adjacent leaf nodes
        for (int n=0;n<(int)adjleafs.size();++n)
        {
          // for each adjacent leaf find the parent, which is on the
          // same layer as the current treenode
          Teuchos::RCP<SelfBinaryTreeNode>  adjtnode = adjleafs[n];
          int diff = adjleafs[n]->Layer() - i;

          // go through layers
          if (diff >= 0)
          {
            while (diff > 0)
            {
              adjtnode = adjtnode->Parent();
              --diff;
            }

            // check if the treenode has already been saved as adjacent node
            bool issaved = false;
            for (int p=0;p<(int)adjtnodes.size();++p)
            {
              if (adjtnode==Teuchos::null) dserror("Teuchos::null vector!!");
              if (adjtnode == adjtnodes[p])
              {
                issaved=true;
                break;
              }
            }

            if (!issaved) adjtnodes.push_back(adjtnode);
          }
        }

        // finally set adjacent treenodes of current treenode
        treenodes_[i][j]->SetAdjacentTnodes(adjtnodes);
      }
    } // all treenodes of current layer
  } // all tree layers

  return;
}

/*----------------------------------------------------------------------*
 | Search for self contact (public)                           popp 01/11|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::SearchSelfContactSeparate(Teuchos::RCP<SelfBinaryTreeNode> treenode)
{

  if (treenode->QualifiedVectors().size()==0)
    dserror("no test vectors defined!");

  //if there is a qualified sample vector, there is no self contact
  for(int i=0; i < (int)treenode->QualifiedVectors().size();i++)
    if(treenode->QualifiedVectors()[i] == true)
    {
      return;
    }

  if (treenode->Type() != SELFCO_LEAF)
  {
    SearchSelfContactSeparate(treenode->Leftchild());
    SearchSelfContactSeparate(treenode->Rightchild());
    EvaluateContactAndAdjacency(treenode->Leftchild(),
                                treenode->Rightchild(),true);
  }

  return;
}

/*----------------------------------------------------------------------*
 | Search for self contact (public)                           popp 05/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::SearchSelfContactCombined(Teuchos::RCP<SelfBinaryTreeNode> treenode)
{

  if (treenode->QualifiedVectors().size()==0)
    dserror("no test vectors defined!");

  //if there is a qualified sample vector, there is no self contact
  for(int i=0; i < (int)treenode->QualifiedVectors().size();i++)
    if(treenode->QualifiedVectors()[i] == true)
    {
      return;
    }

  if (treenode->Type() != SELFCO_LEAF)
  {
    treenode->Leftchild()->CalculateSlabsDop(false);
    treenode->Leftchild()->EnlargeGeometry(enlarge_);
    treenode->Rightchild()->CalculateSlabsDop(false);
    treenode->Rightchild()->EnlargeGeometry(enlarge_);
    SearchSelfContactCombined(treenode->Leftchild());
    SearchSelfContactCombined(treenode->Rightchild());
    EvaluateContactAndAdjacency(treenode->Leftchild(),
                                treenode->Rightchild(),true);
  }

  return;
}

/*----------------------------------------------------------------------*
 | Search for root contact (public)                           popp 01/11|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::SearchRootContactSeparate(Teuchos::RCP<SelfBinaryTreeNode> treenode1,
                                                Teuchos::RCP<SelfBinaryTreeNode> treenode2)
{
  // check if treenodes intercept
  // (they only intercept if ALL slabs intersect!)
  int nintercepts = 0;

  for (int i=0;i<kdop_/2;++i)
  {
    if (treenode1->Slabs()(i,0) <= treenode2->Slabs()(i,0))
    {
      if (treenode1->Slabs()(i,1) >= treenode2->Slabs()(i,0))
        nintercepts++;
      else if (treenode1->Slabs()(i,1) >= treenode2->Slabs()(i,1))
        nintercepts++;
    }
    else if (treenode1->Slabs()(i,0) >= treenode2->Slabs()(i,0))
    {
      if (treenode2->Slabs()(i,1) >= treenode1->Slabs()(i,1))
        nintercepts++;
      else if (treenode2->Slabs()(i,1) >= treenode1->Slabs()(i,0))
        nintercepts++;
    }
  }

  // treenodes intercept
  if (nintercepts==kdop_/2)
  {
    // both treenodes are inner nodes
    if (treenode1->Type() != SELFCO_LEAF && treenode2->Type() != SELFCO_LEAF)
    {
      SearchRootContactSeparate(treenode1->Leftchild(),treenode2->Leftchild());
      SearchRootContactSeparate(treenode1->Leftchild(),treenode2->Rightchild());
      SearchRootContactSeparate(treenode1->Rightchild(),treenode2->Leftchild());
      SearchRootContactSeparate(treenode1->Rightchild(),treenode2->Rightchild());
    }

    // treenode1 is inner, treenode2 is leaf
    if (treenode1->Type() != SELFCO_LEAF && treenode2->Type() == SELFCO_LEAF)
    {
      SearchRootContactSeparate(treenode1->Leftchild(),treenode2);
      SearchRootContactSeparate(treenode1->Rightchild(),treenode2);
    }

    // treenode1 is leaf, treenode3 is inner
    if (treenode1->Type() == SELFCO_LEAF && treenode2->Type() != SELFCO_LEAF)
    {
      SearchRootContactSeparate(treenode1,treenode2->Leftchild());
      SearchRootContactSeparate(treenode1,treenode2->Rightchild());
    }

    // both treenodes are leaf --> feasible pair
    if (treenode1->Type() == SELFCO_LEAF && treenode2->Type() == SELFCO_LEAF)
    {
      int gid1 = (int)treenode1->Elelist()[0]; //global id of first element
      int gid2 = (int)treenode2->Elelist()[0]; //global id of second element
      contactpairs_[gid1].push_back(gid2);
      contactpairs_[gid2].push_back(gid1);
     }
  }

  return;
}

/*----------------------------------------------------------------------*
 | Search for root contact (public)                           popp 01/11|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::SearchRootContactCombined(Teuchos::RCP<SelfBinaryTreeNode> treenode1,
                                                Teuchos::RCP<SelfBinaryTreeNode> treenode2)
{
  // check if treenodes intercept
  // (they only intercept if ALL slabs intersect!)
  int nintercepts = 0;

  for (int i=0;i<kdop_/2;++i)
  {
    if (treenode1->Slabs()(i,0) <= treenode2->Slabs()(i,0))
    {
      if (treenode1->Slabs()(i,1) >= treenode2->Slabs()(i,0))
        nintercepts++;
      else if (treenode1->Slabs()(i,1) >= treenode2->Slabs()(i,1))
        nintercepts++;
    }
    else if (treenode1->Slabs()(i,0) >= treenode2->Slabs()(i,0))
    {
      if (treenode2->Slabs()(i,1) >= treenode1->Slabs()(i,1))
        nintercepts++;
      else if (treenode2->Slabs()(i,1) >= treenode1->Slabs()(i,0))
        nintercepts++;
    }
  }

  // treenodes intercept
  if (nintercepts==kdop_/2)
  {
    // both treenodes are inner nodes
    if (treenode1->Type() != SELFCO_LEAF && treenode2->Type() != SELFCO_LEAF)
    {
      //std::cout <<"\n"<< Comm().MyPID() << " 2 inner nodes!";
      treenode1->Leftchild()->CalculateSlabsDop(false);
      treenode1->Leftchild()->EnlargeGeometry(enlarge_);
      treenode1->Rightchild()->CalculateSlabsDop(false);
      treenode1->Rightchild()->EnlargeGeometry(enlarge_);
      treenode2->Leftchild()->CalculateSlabsDop(false);
      treenode2->Leftchild()->EnlargeGeometry(enlarge_);
      treenode2->Rightchild()->CalculateSlabsDop(false);
      treenode2->Rightchild()->EnlargeGeometry(enlarge_);

      SearchRootContactCombined(treenode1->Leftchild(),treenode2->Leftchild());
      SearchRootContactCombined(treenode1->Leftchild(),treenode2->Rightchild());
      SearchRootContactCombined(treenode1->Rightchild(),treenode2->Leftchild());
      SearchRootContactCombined(treenode1->Rightchild(),treenode2->Rightchild());
    }

    // treenode1 is inner, treenode2 is leaf
    if (treenode1->Type() != SELFCO_LEAF && treenode2->Type() == SELFCO_LEAF)
    {
      treenode1->Leftchild()->CalculateSlabsDop(false);
      treenode1->Leftchild()->EnlargeGeometry(enlarge_);
      treenode1->Rightchild()->CalculateSlabsDop(false);
      treenode1->Rightchild()->EnlargeGeometry(enlarge_);

      SearchRootContactCombined(treenode1->Leftchild(),treenode2);
      SearchRootContactCombined(treenode1->Rightchild(),treenode2);
    }

    // treenode1 is leaf, treenode3 is inner
    if (treenode1->Type() == SELFCO_LEAF && treenode2->Type() != SELFCO_LEAF)
    {
      treenode2->Leftchild()->CalculateSlabsDop(false);
      treenode2->Leftchild()->EnlargeGeometry(enlarge_);
      treenode2->Rightchild()->CalculateSlabsDop(false);
      treenode2->Rightchild()->EnlargeGeometry(enlarge_);

      SearchRootContactCombined(treenode1,treenode2->Leftchild());
      SearchRootContactCombined(treenode1,treenode2->Rightchild());
    }

    // both treenodes are leaf --> feasible pair
    if (treenode1->Type() == SELFCO_LEAF && treenode2->Type() == SELFCO_LEAF)
    {
      int gid1 = (int)treenode1->Elelist()[0]; //global id of first element
      int gid2 = (int)treenode2->Elelist()[0]; //global id of second element
      contactpairs_[gid1].push_back(gid2);
      contactpairs_[gid2].push_back(gid1);
     }
  }

  return;
}

/*----------------------------------------------------------------------*
 | find contact and test adjacency (public)                    popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::EvaluateContactAndAdjacency(Teuchos::RCP<SelfBinaryTreeNode> treenode1,
                                                      Teuchos::RCP<SelfBinaryTreeNode> treenode2,
                                                       bool isadjacent)
{

  // check if treenodes intercept
  // (they only intercept if ALL slabs intersect!)
  int nintercepts = 0;

  for (int i=0;i<kdop_/2;++i)
  {
    if (treenode1->Slabs()(i,0) <= treenode2->Slabs()(i,0))
    {
      if (treenode1->Slabs()(i,1) >= treenode2->Slabs()(i,0))
        nintercepts++;
      else if (treenode1->Slabs()(i,1) >= treenode2->Slabs()(i,1))
        nintercepts++;
    }
    else if (treenode1->Slabs()(i,0) >= treenode2->Slabs()(i,0))
    {
      if (treenode2->Slabs()(i,1) >= treenode1->Slabs()(i,1))
        nintercepts++;
      else if (treenode2->Slabs()(i,1) >= treenode1->Slabs()(i,0))
        nintercepts++;
    }
  }

  if (nintercepts==kdop_/2)
  {
    //teenodes intersect
      if (isadjacent)
      {
        if(dim_==2)
          isadjacent = TestAdjacent2D(treenode1,treenode2);
        else
        {
          isadjacent = TestAdjacent3D(treenode1,treenode2);
        }

        if(isadjacent)
        {
            std::vector<bool> qualifiedvectors1 = treenode1->QualifiedVectors();
            std::vector<bool> qualifiedvectors2 = treenode2->QualifiedVectors();

            if((int)qualifiedvectors1.size() == 0 or (int)qualifiedvectors2.size()==0)
                 dserror("no test vectors defined!");

            if((int)qualifiedvectors1.size() != (int)qualifiedvectors2.size())
              dserror("not the same number of test vectors!");

            for(int i=0; i<(int)qualifiedvectors1.size();i++)
            {
              if(qualifiedvectors1[i] and qualifiedvectors2.at(i))
              {
                return;
              }
            }
        }
      }

      if((int)treenode1->Elelist().size() > (int)treenode2->Elelist().size())
      {
        if (treenode1->Type() != SELFCO_LEAF)
        {
          treenode1->Leftchild()->CalculateSlabsDop(false);
          treenode1->Leftchild()->EnlargeGeometry(enlarge_);
          treenode1->Rightchild()->CalculateSlabsDop(false);
          treenode1->Rightchild()->EnlargeGeometry(enlarge_);
          EvaluateContactAndAdjacency(treenode1->Leftchild(),treenode2, isadjacent);
          EvaluateContactAndAdjacency(treenode1->Rightchild(),treenode2, isadjacent);
        }
      }

      else
      {
         if (treenode2->Type() != SELFCO_LEAF)
         {
           treenode2->Leftchild()->CalculateSlabsDop(false);
           treenode2->Leftchild()->EnlargeGeometry(enlarge_);
           treenode2->Rightchild()->CalculateSlabsDop(false);
           treenode2->Rightchild()->EnlargeGeometry(enlarge_);
           EvaluateContactAndAdjacency(treenode2->Leftchild(),treenode1, isadjacent);
           EvaluateContactAndAdjacency(treenode2->Rightchild(),treenode1, isadjacent);
         }
         else //both tree nodes are leaves
         {
           if(dim_==2)
             isadjacent = TestAdjacent2D(treenode1,treenode2);
           if(dim_==3)
             isadjacent = TestAdjacent3D(treenode1,treenode2);
           if(!isadjacent)
           {
           contactpairs_[treenode1->Elelist()[0]].push_back(treenode2->Elelist()[0]);
           contactpairs_[treenode2->Elelist()[0]].push_back(treenode1->Elelist()[0]);
           }
         }
      }

  }
  else    //dops do not intercept;
    return;
}

/*----------------------------------------------------------------------*
 | find contact and test adjacency (public)                    popp 06/09|
 *----------------------------------------------------------------------*/
bool CONTACT::SelfBinaryTree::TestAdjacent2D(Teuchos::RCP<SelfBinaryTreeNode> treenode1,
                                                      Teuchos::RCP<SelfBinaryTreeNode> treenode2)
{
  if(dim_ != 2)
    dserror("TestAdjacent2D: problem must be 2D!!\n");

  std::vector<int> endnodes1 = treenode1->Endnodes();
  std::vector<int> endnodes2 = treenode2->Endnodes();

  if(endnodes1.size() != 2 or endnodes2.size() != 2)
    dserror("treenode has not 2 endnodes!!\n");

  for(int i=0; i<(int)endnodes1.size();i++)
  {
    if(endnodes1[i]== -1)
    {
      //treenode is a closed surface -> has no endnodes;
      return false;
    }

    for(int j=0; j<(int)endnodes2.size();j++)
    {
      if(endnodes2[j] == -1)
      {
        //treenode is a closed surface -> has no endnodes;
        return false;
      }

      else if(endnodes1[i] == endnodes2[j])
      {
        //treenodes are adjacent
        return true;
      }

    }

  }
  //treenodes are not adjacent
  return false;
}

/*----------------------------------------------------------------------*
 | find contact and test adjacency (public)                    popp 06/09|
 *----------------------------------------------------------------------*/
bool CONTACT::SelfBinaryTree::TestAdjacent3D(Teuchos::RCP<SelfBinaryTreeNode> treenode1,
                                                      Teuchos::RCP<SelfBinaryTreeNode> treenode2)
{
  if(dim_ != 3)
    dserror("TestAdjacent3D: problem must be 3D!!\n");

  //if the treenodes are in the same layer check the vector of adjacent treenodes
  if(treenode1->Layer() == treenode2->Layer())
  {

    std::vector<Teuchos::RCP<SelfBinaryTreeNode> > adjtnodes = treenode1->AdjacentTreenodes();
    for(int i=0; i<(int)adjtnodes.size();i++)
    {
      if(adjtnodes[i] == treenode2)
      {
        //   treenodes are adjacent
        return true;
      }
    }
  }

  else
  {
    // check if bounding volumes overlap
    // (they only intercept if ALL slabs intercept!)
    int nintercepts = 0;

    for (int i=0;i<kdop_/2;++i)
    {
      if (treenode1->Slabs()(i,0) <= treenode2->Slabs()(i,0))
      {
        if (treenode1->Slabs()(i,1) >= treenode2->Slabs()(i,0))
          nintercepts++;
        else if (treenode1->Slabs()(i,1) >= treenode2->Slabs()(i,1))
          nintercepts++;
      }
      else if (treenode1->Slabs()(i,0) >= treenode2->Slabs()(i,0))
      {
        if (treenode2->Slabs()(i,1) >= treenode1->Slabs()(i,1))
          nintercepts++;
        else if (treenode2->Slabs()(i,1) >= treenode1->Slabs()(i,0))
          nintercepts++;
      }
    }

    //if the bounding voumes overlap
    if (nintercepts==kdop_/2)
    {
      if (treenode1->Type()==SELFCO_LEAF and treenode2->Type()==SELFCO_LEAF)
      {
     // two leaves
        std::vector<Teuchos::RCP<SelfBinaryTreeNode> > adjleafs = adjacencymatrix_[treenode1->Elelist()[0]];
        for(int i=0; i<(int)adjleafs.size();i++)
        {
          if(treenode2 == adjleafs[i])
          {
           // leaves are adjacent
            return true;
          }
        }
      }

      //one leaf and one inner treenode
      else if (treenode1->Type()==SELFCO_LEAF and treenode2->Type()!=SELFCO_LEAF)
        return ( TestAdjacent3D(treenode1,treenode2->Leftchild())
                  or TestAdjacent3D(treenode1,treenode2->Rightchild()));

      else if(treenode1->Type()!=SELFCO_LEAF and treenode2->Type()==SELFCO_LEAF)
        return ( TestAdjacent3D(treenode1->Leftchild(),treenode2)
                or TestAdjacent3D(treenode1->Rightchild(),treenode2));

      else if((treenode1->Layer()) > treenode2->Layer())
          return ( TestAdjacent3D(treenode1,treenode2->Leftchild())
                    or TestAdjacent3D(treenode1,treenode2->Rightchild()));

      else if((treenode1->Layer()) < treenode2->Layer())
          return ( TestAdjacent3D(treenode1->Leftchild(),treenode2)
                    or TestAdjacent3D(treenode1->Rightchild(),treenode2));
    }
  }
  //treenodes do not overlap;
  return false;
}

/*----------------------------------------------------------------------*
 | do Master/Self facet sorting (self contact) (public)        popp 06/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::MasterSlaveSorting(int eleID,bool isslave)
{
  // do as long as there are still contact pairs
  if(contactpairs_.find(eleID) != contactpairs_.end() && !contactpairs_.empty())
  {
    // set the current element to content of "isslave"
    DRT::Element* element= idiscret_.gElement(eleID);
    CONTACT::CoElement* celement = static_cast<CONTACT::CoElement*>(element);
    celement->SetSlave()=isslave;

    // if the element is a slave, set its node to slave (otherwise the nodes
    // are master-nodes already and nodes between a master and a slave element
    // should be slave nodes)
    if (celement->IsSlave())
      for (int i=0; i<(int)element->NumNode();i++)
      {
        DRT::Node* node = element->Nodes()[i];
        CONTACT::CoNode* cnode = static_cast<CONTACT::CoNode*>(node);
        cnode->SetSlave()=isslave;
      }

    // get the ID of elements in contact with current one
    std::vector<int> contacteleID = contactpairs_[eleID];

    // erase the current element from list of contact pairs
    contactpairs_.erase(eleID);

    // loop over all contact partners of current element
    for(int i =0; i<(int)contacteleID.size();i++)
    {
      // add to list of search candidates if current element is slave
      if (celement->IsSlave())
        celement->AddSearchElements(contacteleID[i]);

      // recursively call this function again
      if (contactpairs_.find(contacteleID[i])!=contactpairs_.end())
        MasterSlaveSorting(contacteleID[i], !isslave);
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 | Separate update, contact search and master/slave sorting   popp 01/11|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::SearchContactSeparate()
{
  // get out of here if not participating in interface
  if (!lComm()) return;

  // check is root node available
  if ((int)roots_.size()==0) dserror("ERROR: No root node for search!");
  if (roots_[0]==Teuchos::null) dserror("ERROR: No root node for search!");

  // reset contact pairs from last iteration
  contactpairs_.clear();

  //lComm()->Barrier();
  //const double t_start1 = Teuchos::Time::wallTime();

  //**********************************************************************
  // STEP 1: update geometry (DOPs and sample vectors) bottom-up
  //**********************************************************************
  // update tree bottom up (for every treelayer)
  for (int i=((int)(treenodes_.size()-1));i>=0;--i)
  {
    for (int j=0;j<(int)(treenodes_[i].size());j++)
      treenodes_[i][j]->UpdateSlabsBottomUp(enlarge_);
  }
  UpdateNormals();

  //lComm()->Barrier();
  //const double t_end1 = Teuchos::Time::wallTime()-t_start1;
  //if (lComm()->MyPID()==0) std::cout << "********* Update:\t" << t_end1 << " seconds\n";

  //**********************************************************************
  // STEP 2: distribute roots among all processors
  //**********************************************************************
  // introduce some parallelization for multibody contact
  std::vector<int> myroots(0);
  int nproc = lComm()->NumProc();
  int nroot = (int)roots_.size();
  int ratio = nroot / nproc;
  int rest  = nroot % nproc;

  // give 'ratio+1' roots to the first 'rest' procs
  if (lComm()->MyPID() < rest)
    for (int k=0;k<ratio+1;++k)
      myroots.push_back(lComm()->MyPID()*(ratio+1) + k);

  // give 'ratio' roots to the remaining procs
  else /*(lComm()->MyPID() >= rest)*/
    for (int k=0;k<ratio;++k)
      myroots.push_back(rest*(ratio+1) + (lComm()->MyPID()-rest)*ratio + k);

  //lComm()->Barrier();
  //const double t_start2 = Teuchos::Time::wallTime();

  //**********************************************************************
  // STEP 3: search for self contact starting at root nodes
  //**********************************************************************
  for (int k=0;k<(int)myroots.size();++k)
  {
    //std::cout << "Self search for root node " << myroots[k] << std::endl;
    SearchSelfContactSeparate(roots_[myroots[k]]);
  }

  //lComm()->Barrier();
  //const double t_end2 = Teuchos::Time::wallTime()-t_start2;
  //if (lComm()->MyPID()==0) std::cout << "********* SelfSe:\t" << t_end2 << " seconds\n";

  //lComm()->Barrier();
  //const double t_start3 = Teuchos::Time::wallTime();

  //**********************************************************************
  // STEP 4: search for two-body contact between different roots
  //**********************************************************************
  for (int k=0;k<(int)myroots.size();++k)
  {
    for (int m=myroots[k]+1;m<(int)roots_.size();++m)
    {
      //std::cout << "-> Root search for pair " << myroots[k] << " " << m << std::endl;
      SearchRootContactSeparate(roots_[myroots[k]],roots_[m]);
    }
  }

  //lComm()->Barrier();
  //const double t_end3 = Teuchos::Time::wallTime()-t_start3;
  //if (lComm()->MyPID()==0) std::cout << "********* RootSe:\t" << t_end3 << " seconds\n";

  //lComm()->Barrier();
  //const double t_start4 = Teuchos::Time::wallTime();

  //**********************************************************************
  // STEP 5: slave and master facet sorting
  //**********************************************************************
  std::map<int, Teuchos::RCP<SelfBinaryTreeNode> > ::iterator leafiter = leafsmap_.begin();
  std::map<int, Teuchos::RCP<SelfBinaryTreeNode> > ::iterator leafiter_end = leafsmap_.end();

  // first (re)set all contact elements and nodes to master
  while (leafiter != leafiter_end)
  {
    int gid = leafiter->first;
    DRT::Element* element= idiscret_.gElement(gid);
    CONTACT::CoElement* celement = static_cast<CONTACT::CoElement*>(element);

    if (celement->IsSlave() == true)
    {
      // reset element to master
      celement->SetSlave() = false;

      // reset nodes to master
      for (int i=0;i<(int)element->NumNode();++i)
      {
        DRT::Node* node = element->Nodes()[i];
        CONTACT::CoNode* cnode = static_cast<CONTACT::CoNode*>(node);
        cnode->SetSlave() = false;
      }
    }

    // increment iterator
    ++leafiter;
  }

  //lComm()->Barrier();
  //const double t_end4 = Teuchos::Time::wallTime()-t_start4;
  //if (lComm()->MyPID()==0) std::cout << "********* ResetE:\t" << t_end4 << " seconds\n";

  //lComm()->Barrier();
  //const double t_start5 = Teuchos::Time::wallTime();

  // make contactpairs information redundant on all procs
  std::vector<int> locdata;
  for (int i=0; i<elements_->NumMyElements();++i)
  {
    int gid = elements_->GID(i);
    if (contactpairs_.find(gid)!=contactpairs_.end())
      locdata.push_back(gid);
  }
  Teuchos::RCP<Epetra_Map> mymap  = Teuchos::rcp(new Epetra_Map(-1,(int)locdata.size(),&locdata[0],0,*lComm()));
  Teuchos::RCP<Epetra_Map> redmap = LINALG::AllreduceEMap(*mymap);
  DRT::Exporter ex(*mymap,*redmap,*lComm());
  ex.Export(contactpairs_);

  //lComm()->Barrier();
  //const double t_end5 = Teuchos::Time::wallTime()-t_start5;
  //if (lComm()->MyPID()==0) std::cout << "********* Commun:\t" << t_end5 << " seconds\n";

  //lComm()->Barrier();
  //const double t_start6 = Teuchos::Time::wallTime();

  // now do new slave and master sorting
  while (!contactpairs_.empty())
  {
    DRT::Element* element = idiscret_.gElement(contactpairs_.begin()->first);
    CONTACT::CoElement* celement = static_cast<CONTACT::CoElement*>(element);
    MasterSlaveSorting(contactpairs_.begin()->first,celement->IsSlave());
  }

  //lComm()->Barrier();
  //const double t_end6 = Teuchos::Time::wallTime()-t_start6;
  //if (lComm()->MyPID()==0) std::cout << "********* SortSM:\t" << t_end6 << " seconds\n";

  //lComm()->Barrier();
  //const double t_start7 = Teuchos::Time::wallTime();

  //**********************************************************************
  // STEP 6: check consistency of slave and master facet sorting
  //**********************************************************************
  for (int i=0; i<elements_->NumMyElements();++i)
  {
    int gid1 = elements_->GID(i);
    DRT::Element* ele1 = idiscret_.gElement(gid1);
    if (!ele1) dserror("ERROR: Cannot find element with gid %",gid1);
    MORTAR::MortarElement* element1 = static_cast<MORTAR::MortarElement*>(ele1);

    // only slave elements store search candidates
    if (!element1->IsSlave())
      continue;

    // loop over the search candidates of elements1
    for (int j=0;j<element1->MoData().NumSearchElements();++j)
    {
      int gid2 = element1->MoData().SearchElements()[j];
      DRT::Element* ele2 = idiscret_.gElement(gid2);
      if (!ele2) dserror("ERROR: Cannot find element with gid %",gid2);
      MORTAR::MortarElement* element2 = static_cast<MORTAR::MortarElement*>(ele2);

      // error if this is a slave element
      // (this happens if individual self contact patches are connected,
      // because our sorting algorithm still fails in that case)
      if (element2->IsSlave())
        dserror("ERROR: Slave / master inconsistency in self contact");
    }
  }

  //lComm()->Barrier();
  //const double t_end7 = Teuchos::Time::wallTime()-t_start7;
  //if (lComm()->MyPID()==0) std::cout << "********* CheckE:\t" << t_end7 << " seconds\n";

  return;
}

/*----------------------------------------------------------------------*
 | Combined update, contact search and master/slave sorting   popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::SearchContactCombined()
{
  // get out of here if not participating in interface
  if (!lComm()) return;

  // check is root node available
  if ((int)roots_.size()==0) dserror("ERROR: No root node for search!");
  if (roots_[0]==Teuchos::null) dserror("ERROR: No root node for search!");

  // reset contact pairs from last iteration
  contactpairs_.clear();

  //**********************************************************************
  // STEP 1: update geometry (DOPs and sample vectors)
  //**********************************************************************
  for (int k=0;k<(int)roots_.size();++k)
  {
    roots_[k]->CalculateSlabsDop(false);
    roots_[k]->EnlargeGeometry(enlarge_);
  }
  UpdateNormals();

  //**********************************************************************
  // STEP 2: distribute roots among all processors
  //**********************************************************************
  // introduce some parallelization for multibody contact
  std::vector<int> myroots(0);
  int nproc = lComm()->NumProc();
  int nroot = (int)roots_.size();
  int ratio = nroot / nproc;
  int rest  = nroot % nproc;

  // give 'ratio+1' roots to the first 'rest' procs
  if (lComm()->MyPID() < rest)
    for (int k=0;k<ratio+1;++k)
      myroots.push_back(lComm()->MyPID()*(ratio+1) + k);

  // give 'ratio' roots to the remaining procs
  else /*(lComm()->MyPID() >= rest)*/
    for (int k=0;k<ratio;++k)
      myroots.push_back(rest*(ratio+1) + (lComm()->MyPID()-rest)*ratio + k);

  //**********************************************************************
  // STEP 3: search for self contact starting at root nodes
  //**********************************************************************
  for (int k=0;k<(int)myroots.size();++k)
  {
    //std::cout << "Self search for root node " << myroots[k] << std::endl;
    SearchSelfContactSeparate(roots_[myroots[k]]);
  }

  //**********************************************************************
  // STEP 4: search for two-body contact between different roots
  //**********************************************************************
  for (int k=0;k<(int)myroots.size();++k)
  {
    for (int m=myroots[k]+1;m<(int)roots_.size();++m)
    {
      //std::cout << "-> Root search for pair " << myroots[k] << " " << m << std::endl;
      SearchRootContactSeparate(roots_[myroots[k]],roots_[m]);
    }
  }

  //**********************************************************************
  // STEP 5: slave and master facet sorting
  //**********************************************************************
  std::map<int, Teuchos::RCP<SelfBinaryTreeNode> > ::iterator leafiter = leafsmap_.begin();
  std::map<int, Teuchos::RCP<SelfBinaryTreeNode> > ::iterator leafiter_end = leafsmap_.end();

  // first (re)set all contact elements and nodes to master
  while (leafiter != leafiter_end)
  {
    int gid = leafiter->first;
    DRT::Element* element= idiscret_.gElement(gid);
    CONTACT::CoElement* celement = static_cast<CONTACT::CoElement*>(element);

    if (celement->IsSlave() == true)
    {
      // reset element to master
      celement->SetSlave() = false;

      // reset nodes to master
      for (int i=0;i<(int)element->NumNode();++i)
      {
        DRT::Node* node = element->Nodes()[i];
        CONTACT::CoNode* cnode = static_cast<CONTACT::CoNode*>(node);
        cnode->SetSlave() = false;
      }
    }

    // increment iterator
    ++leafiter;
  }

  // make contactpairs information redundant on all procs
  std::vector<int> locdata;
  for (int i=0; i<elements_->NumMyElements();++i)
  {
    int gid = elements_->GID(i);
    if (contactpairs_.find(gid)!=contactpairs_.end())
      locdata.push_back(gid);
  }
  Teuchos::RCP<Epetra_Map> mymap  = Teuchos::rcp(new Epetra_Map(-1,(int)locdata.size(),&locdata[0],0,*lComm()));
  Teuchos::RCP<Epetra_Map> redmap = LINALG::AllreduceEMap(*mymap);
  DRT::Exporter ex(*mymap,*redmap,*lComm());
  ex.Export(contactpairs_);

  // now do new slave and master sorting
  while (!contactpairs_.empty())
  {
    DRT::Element* element = idiscret_.gElement(contactpairs_.begin()->first);
    CONTACT::CoElement* celement = static_cast<CONTACT::CoElement*>(element);
    MasterSlaveSorting(contactpairs_.begin()->first,celement->IsSlave());
  }

  //**********************************************************************
  // STEP 6: check consistency of slave and master facet sorting
  //**********************************************************************
  for (int i=0; i<elements_->NumMyElements();++i)
  {
    int gid1 = elements_->GID(i);
    DRT::Element* ele1 = idiscret_.gElement(gid1);
    if (!ele1) dserror("ERROR: Cannot find element with gid %",gid1);
    MORTAR::MortarElement* element1 = static_cast<MORTAR::MortarElement*>(ele1);

    // only slave elements store search candidates
    if (!element1->IsSlave())
      continue;

    // loop over the search candidates of elements1
    for (int j=0;j<element1->MoData().NumSearchElements();++j)
    {
      int gid2 = element1->MoData().SearchElements()[j];
      DRT::Element* ele2 = idiscret_.gElement(gid2);
      if (!ele2) dserror("ERROR: Cannot find element with gid %",gid2);
      MORTAR::MortarElement* element2 = static_cast<MORTAR::MortarElement*>(ele2);

      // error if this is a slave element
      // (this happens if individual self contact patches are connected,
      // because our sorting algorithm still fails in that case)
      if (element2->IsSlave())
        dserror("ERROR: Slave / master inconsistency in self contact");
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 | Update normals and qualified sample vectors                popp 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::SelfBinaryTree::UpdateNormals()
{
  // first update normals and sample vectors of all leaf-nodes
  std::map<int, Teuchos::RCP<SelfBinaryTreeNode> > ::iterator iter = leafsmap_.begin();
  std::map<int, Teuchos::RCP<SelfBinaryTreeNode> > ::iterator iter_end = leafsmap_.end();

  while (iter != iter_end)
  {
    iter->second->CalculateQualifiedVectors();
    ++iter;
  }

  // now update the rest of the tree layer by layer in a bottom-up way
  for (int i=(int)treenodes_.size()-1; i>=0; --i)
  {
    for (int j=0; j<(int)treenodes_[i].size(); ++j)
      if (treenodes_[i][j]->Type() != SELFCO_LEAF)
        treenodes_[i][j]->UpdateQualifiedVectorsBottomUp();
  }

  return;
}

