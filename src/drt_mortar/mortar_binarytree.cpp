/*!----------------------------------------------------------------------
\file mortar_binarytree.cpp
\brief A class for performing mortar search in 2D/3D based on binarytrees

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

#include "mortar_binarytree.H"
#include "mortar_node.H"
#include "mortar_element.H"
#include "mortar_defines.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../linalg/linalg_fixedsizematrix.H"


/*----------------------------------------------------------------------*
 |  ctor BinaryTreeNode (public)                              popp 10/08|
 *----------------------------------------------------------------------*/
MORTAR::BinaryTreeNode::BinaryTreeNode(
                     MORTAR::BinaryTreeNodeType type,
                     DRT::Discretization& discret,
                     Teuchos::RCP<BinaryTreeNode> parent,
                     std::vector<int> elelist,
                     const Epetra_SerialDenseMatrix& dopnormals,
                     const int& kdop, const int& dim,
                     const int layer,
                     std::vector<std::vector<Teuchos::RCP<BinaryTreeNode> > > & streenodesmap,
                     std::vector<std::vector<Teuchos::RCP<BinaryTreeNode> > > & mtreenodesmap,
                     std::vector<std::vector<Teuchos::RCP<BinaryTreeNode> > > & sleafsmap,
                     std::vector<std::vector<Teuchos::RCP<BinaryTreeNode> > > & mleafsmap) :
type_(type),
idiscret_(discret),
parent_(parent),
elelist_(elelist),
dopnormals_(dopnormals),
kdop_(kdop),
dim_(dim),
layer_(layer),
streenodesmap_(streenodesmap),
mtreenodesmap_(mtreenodesmap),
sleafsmap_(sleafsmap),
mleafsmap_(mleafsmap)
{
  // reshape slabs matrix
  if (dim_==2)      slabs_.Reshape(kdop_/2,2);
  else if (dim_==3) slabs_.Reshape(kdop_/2,2);
  else              dserror("ERROR: Problem dimension must be 2D or 3D!");

  return;
}

/*----------------------------------------------------------------------*
 | get communicator (public)                                  popp 10/08|
 *----------------------------------------------------------------------*/
const Epetra_Comm& MORTAR::BinaryTreeNode::Comm() const
{
  return idiscret_.Comm();
}

/*----------------------------------------------------------------------*
 | Initialize tree (public)                                   popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTreeNode::InitializeTree(double& enlarge)
{
  // return if proc. has no elements!
  if (Elelist().size()==0) return;

  // calculate bounding volume
  CalculateSlabsDop(true);
  EnlargeGeometry(enlarge);

  // if current treenode is inner treenode
  if (type_==0 || type_==2)
  {
    // divide treenode
    DivideTreeNode();

    // check what to do with left child
    if (leftchild_->Elelist().size()==0)
    {
      dserror("ERROR: InitializeTree:Processor has no leftchild elements-->return;");
      return;
    }
    else
    {
      //if leftchild is slave leaf
      if (leftchild_->Type()==1)
        sleafsmap_[0].push_back(leftchild_);
      //if leaftchild is master leaf
      if (leftchild_->Type()==3)
        mleafsmap_[0].push_back(leftchild_);

      // recursively initialize the whole tree
      leftchild_->InitializeTree(enlarge);
    }

    // check what to do with right child
    if (rightchild_->Elelist().size()==0)
    {
      dserror("ERROR: InitializeTree:Processor has no rightchild elements-->return;");
      return;
    }
    else
    {
      //if rightchild is slave leaf
      if (rightchild_->Type()==1)
        sleafsmap_[1].push_back(rightchild_);
      // if rightchild is master leaf
      if (rightchild_->Type()==3)
        mleafsmap_[1].push_back(rightchild_);

      // recursively initialize the whole tree
      rightchild_->InitializeTree(enlarge);
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 | Calculate slabs of DOP out of node postions (public)       popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTreeNode::CalculateSlabsDop(bool isinit)
{
  // initialize slabs
  for (int j=0; j<kdop_/2; j++)
  {
    slabs_(j,0) =  1.0e12;
    slabs_(j,1) = -1.0e12;
  }

  // calculate slabs for every element
  for (int i=0; i<(int)Elelist().size();++i)
  {
    int gid = Elelist()[i];
    DRT::Element* element= idiscret_.gElement(gid);
    if (!element) dserror("ERROR: Cannot find element with gid %\n",gid);
    MortarElement* mrtrelement=static_cast<MortarElement*>(element);
    DRT::Node** nodes = mrtrelement->Nodes();
    if (!nodes) dserror("ERROR: Null pointer!");

    // calculate slabs for every node on every element
    for (int k=0;k<mrtrelement->NumNode();k++)
    {
      MortarNode* mrtrnode=static_cast<MortarNode*>(nodes[k]);
      if (!mrtrnode) dserror("ERROR: Null pointer!");

      // decide which position is relevant (initial or current)
      double pos[3] = {0.0, 0.0, 0.0};
      for (int j=0;j<dim_;++j)
      {
        if (isinit) pos[j] = mrtrnode->X()[j];
        else        pos[j] = mrtrnode->xspatial()[j];
      }

      // calculate slabs
      for(int j=0; j<kdop_/2;j++)
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
        mrtrelement->LocalCoordinatesOfNode(k,xi);
        mrtrelement->ComputeUnitNormalAtXi(xi,normal);

        // now the auxiliary position
        double auxpos [3] = {0.0, 0.0, 0.0};
        double scalar=0.0;
        for (int j=0;j<dim_;j++)
          scalar=scalar+(mrtrnode->X()[j]+mrtrnode->uold()[j]-mrtrnode->xspatial()[j])*normal[j];
        for (int j=0;j<dim_;j++)
          auxpos[j]=mrtrnode->xspatial()[j]+scalar*normal[j];

        for(int j=0; j<kdop_/2;j++)
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

  //Prints Slabs to std::cout
  //PrintSlabs();

  return;
}

/*----------------------------------------------------------------------*
 | Update slabs bottom up (public)                            popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTreeNode::UpdateSlabsBottomUp(double & enlarge)
{
  // if current treenode is inner node
  if (type_==0||type_==2)
  {
    //std::cout <<"\n"<< Comm().MyPID() << " Treenode "<< j <<" is a inner treenode!";
    for (int k=0;k<kdop_/2;k++)
    {
      //for minimum
      if (leftchild_->Slabs()(k,0)<=rightchild_->Slabs()(k,0))
        slabs_(k,0)=leftchild_->Slabs()(k,0);
      else
        slabs_(k,0)=rightchild_->Slabs()(k,0);

      // for maximum
      if (leftchild_->Slabs()(k,1)>=rightchild_->Slabs()(k,1))
        slabs_(k,1)=leftchild_->Slabs()(k,1);
      else
        slabs_(k,1)=rightchild_->Slabs()(k,1);
    }
  }

  //if current treenode is leafnode
  if (type_==1||type_==3)
  {
    // initialize slabs
    for (int j=0; j<kdop_/2; j++)
    {
      slabs_(j,0) =  1.0e12;
      slabs_(j,1) = -1.0e12;
    }

    int gid = Elelist()[0];
    DRT::Element* element= idiscret_.gElement(gid);
    if (!element) dserror("ERROR: Cannot find element with gid %\n",gid);
    MortarElement* mrtrelement=static_cast<MortarElement*>(element);
    DRT::Node** nodes = mrtrelement->Nodes();
    if (!nodes) dserror("ERROR: Null pointer!");

    // update slabs for every node
    for (int k=0;k<mrtrelement->NumNode();++k)
    {
      MortarNode* mrtrnode=static_cast<MortarNode*>(nodes[k]);
      if (!mrtrnode) dserror("ERROR: Null pointer!");

      // decide which position is relevant (initial or current)
      double pos[3] = {0.0, 0.0, 0.0};
      for (int j=0;j<dim_;++j) pos[j] = mrtrnode->xspatial()[j];

      // calculate slabs
      for(int j=0; j<kdop_/2;j++)
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

      // enlarge slabs with auxiliary position
      // first calculate element normal at current node
      double xi[2] = {0.0, 0.0};
      double normal[3] = {0.0, 0.0, 0.0};
      mrtrelement->LocalCoordinatesOfNode(k,xi);
      mrtrelement->ComputeUnitNormalAtXi(xi,normal);

      // now the auxiliary position
      double auxpos [3] = {0.0, 0.0, 0.0};
      double scalar=0.0;
      for (int j=0;j<dim_;j++)
        scalar=scalar+(mrtrnode->X()[j]+mrtrnode->uold()[j]-mrtrnode->xspatial()[j])*normal[j];
      for (int j=0;j<dim_;j++)
        auxpos[j]=mrtrnode->xspatial()[j]+scalar*normal[j];

      for(int j=0; j<kdop_/2;j++)
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

    for (int i=0 ; i<kdop_/2 ; i++)
    {
      slabs_(i,0)=slabs_(i,0)-enlarge;
      slabs_(i,1)=slabs_(i,1)+enlarge;
    }

    //Prints Slabs to std::cout
    //PrintSlabs();

  } // current treenode is leaf

  return;
}
/*----------------------------------------------------------------------*
 | Divide treenode (public)                                   popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTreeNode::DivideTreeNode()
{
  // map of elements belonging to left / right child treenode
  std::vector<int> leftelements(0);
  std::vector<int> rightelements(0);

  //if only 2 elements in Treenode, create new treenodes out of them
  if (Elelist().size()==2)
  {
    leftelements.push_back(Elelist()[0]);
    rightelements.push_back(Elelist()[1]);
  }
  // if more than 2 elements in Treenode
  else if (Elelist().size()>2)
  {
    //calculate splitting area (split along longest side)
    double lmax = 0.0;          // max. length of sides of DOP
    int splittingnormal = -1;    // defines side to split
    double xmedian[3] = {0.0, 0.0, 0.0}; // coordinates of centroid

    for(int i=0;i<kdop_/2;++i)
    {
      double lcurrent=abs(slabs_(i,1)-slabs_(i,0));
      if (lmax < lcurrent)
      {
        lmax = lcurrent;
        splittingnormal = i;
      }
    }

    // find median of centroid coordinates to divide area
    // coordinates of median
    for(int i=0; i<dim_;++i)
      xmedian[i]=slabs_(i,1)-((slabs_(i,1)-slabs_(i,0))/2);

    //compute d of ax+by+cz=d of splittingplane
    double d = xmedian[0]*dopnormals_(splittingnormal,0)
             + xmedian[1]*dopnormals_(splittingnormal,1)
             + xmedian[2]*dopnormals_(splittingnormal,2);

    //split treenode into two parts
    for (int i=0; i<((int)Elelist().size());++i)
    {
      bool isright = false; // true, if element should be sorted into right treenode
      bool isleft = false;  // true, if element should be sorted into left treenode

      int gid = Elelist()[i];
      DRT::Element* element= idiscret_.gElement(gid);
      if (!element) dserror("ERROR: Cannot find element with gid %\n",gid);
      DRT::Node** nodes = element->Nodes();

      //vector of values of Hesse-Normalform of nodes of elements
      Epetra_SerialDenseVector  axbycz;
      axbycz.Resize(element->NumNode());

      for (int k=0;k<element->NumNode();++k)
      {
        MortarNode* mrtrnode = static_cast<MortarNode*>(nodes[k]);
        if (!mrtrnode) dserror("ERROR: Null pointer!");
        const double* posnode = mrtrnode->X();

        //split along chosen area
        //ax+by+cz< or > d = criterion
        //compute ax+by+cz for chosen node
        if (dim_==2)
          axbycz[k] = posnode[0]*dopnormals_(splittingnormal,0)
                    + posnode[1]*dopnormals_(splittingnormal,1);
        else if (dim_==3)
          axbycz[k] = posnode[0]*dopnormals_(splittingnormal,0)
                    + posnode[1]*dopnormals_(splittingnormal,1)
                     + posnode[2]*dopnormals_(splittingnormal,2);
        else
          dserror("ERROR: Problem dimension must be 2D or 3D!");

         if (axbycz[k]>=d) isright=true;
         if (axbycz[k]<d)  isleft=true;
      }

      if (isright==false && isleft==false)
        dserror("ERROR: Current element could neither be sorted into left- or right-child node!");

      // if element is split through, it is sorted into left treenode
      if (isright==true && isleft==true) isright=false;

      // sort elements into child treenodes
      if (isright) rightelements.push_back(gid);
      if (isleft)  leftelements.push_back(gid);
    }
  }

  // if treenode splitting algorithm was not able to divide treenode
  // successfully (i.e all elements are in one child treenode),
  // then just put one element into the other treenode
  if (leftelements.size()==0 && rightelements.size()>1)
  {
    leftelements.push_back(rightelements[rightelements.size()-1]);
    rightelements.pop_back();
  }
  if (rightelements.size()==0 && leftelements.size()>1)
  {
    rightelements.push_back(leftelements[leftelements.size()-1]);
    leftelements.pop_back();
  }

  // define type of newly created children treenodes
  if (Elelist().size()>=2)
  {
    //defines type of left and right TreeNode
    BinaryTreeNodeType lefttype = UNDEFINED;
    BinaryTreeNodeType righttype = UNDEFINED;

    // is the new left child treenode a leaf node?
    if (leftelements.size()==1)
    {
      if (type_==0)      lefttype = SLAVE_LEAF;
      else if (type_==2) lefttype = MASTER_LEAF;
      else               dserror("ERROR: Invalid TreeNodeType");
    }
    else
    {
      if (type_==0)      lefttype = SLAVE_INNER;
      else if (type_==2) lefttype = MASTER_INNER;
      else               dserror("ERROR: Invalid TreeNodeType");
    }

    // is the new right child treenode a leaf node?
    if (rightelements.size()==1)
    {
      if (type_==0)      righttype=SLAVE_LEAF;
      else if (type_==2) righttype=MASTER_LEAF;
      else               dserror("ERROR: Invalid TreeNodeType");
    }
    else
    {
      if (type_==0)      righttype=SLAVE_INNER;
      else if (type_==2) righttype=MASTER_INNER;
      else               dserror("ERROR: Invalid TreeNodeType");
    }

    // build left child treenode
    leftchild_= Teuchos::rcp(new BinaryTreeNode(lefttype,idiscret_, Teuchos::rcp(this, false),
                    leftelements, dopnormals_, kdop_, dim_, (layer_+1), streenodesmap_, mtreenodesmap_,
                    sleafsmap_, mleafsmap_));

    // build right child treenode
    rightchild_= Teuchos::rcp(new BinaryTreeNode(righttype,idiscret_, Teuchos::rcp(this, false),
                     rightelements, dopnormals_, kdop_, dim_, (layer_+1), streenodesmap_, mtreenodesmap_,
                     sleafsmap_, mleafsmap_));

    // update slave and mastertreenodes map
    // if parent treenode is slave
    if (type_==0)
    {
      // if map of treenodes does not have enogh rows-->resize!
      if ((int)(streenodesmap_.size())<=(layer_+1))
        streenodesmap_.resize((layer_+2));

      // put new pointers to children into map
      streenodesmap_[(layer_+1)].push_back(leftchild_);
      streenodesmap_[(layer_+1)].push_back(rightchild_);
    }

    // if parent treenode is master
    if (type_==2)
    {
      // if map of treenodes does not have enogh rows-->resize!
      if ((int)(mtreenodesmap_.size())<=(layer_+1))
            mtreenodesmap_.resize((layer_+2));

      // put new pointers to children into map
      mtreenodesmap_[(layer_+1)].push_back(leftchild_);
      mtreenodesmap_[(layer_+1)].push_back(rightchild_);
    }
  }

  else dserror( "ERROR: Only 1 or 0 elements in map-->TreeNode cannot be devided!!");

  return;
}

/*----------------------------------------------------------------------*
 | Print type of treenode to std::cout (public)               popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTreeNode::PrintType()
{
  if (type_==0)
    std::cout << endl << "SLAVE_INNER ";
  else if (type_==1)
    std::cout << endl << "SLAVE_LEAF ";
  else if (type_==2)
    std::cout << endl << "MASTER_INNER ";
  else if (type_==3)
    std::cout << endl << "MASTER_LEAF ";
  else if (type_==4)
    std::cout << endl << "TreeNode contains no Slave-Elements=NO_SLAVEELEMENTS ";
  else if (type_==5)
    std::cout << endl << "TreeNode contains no Master-Elements=NO_MASTERELEMENTS ";
  else
    std::cout << endl << "UNDEFINED ";
}

/*----------------------------------------------------------------------*
 | Print slabs to std::cout (public)                          popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTreeNode::PrintSlabs()
{
   std::cout << endl << Comm().MyPID() << "************************************************************";
   PrintType();
   std::cout << "slabs:";
   for (int i=0;i<slabs_.M();i++)
      std::cout << "\nslab: "<<i<<" min: "<< slabs_.operator ()(i,0) << " max: "<<slabs_.operator ()(i,1);
   std::cout << "\n**********************************************************\n";
}

/*----------------------------------------------------------------------*
 | Print slabs of dop to file for Gmsh (public)               popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTreeNode::PrintDopsForGmsh(std::string filename)
{
  FILE* fp = NULL;
  std::ostringstream currentfilename;

  if (dim_==2)
  {
    fp = fopen(filename.c_str(), "a");
    std::stringstream gmshfilecontent;
    //PrintSlabs();

    // Matrix containing coordinates of points defining kdop (x,y,z)
    Epetra_SerialDenseMatrix position(kdop_,3);

    for (int i=0;i<kdop_;i++) position(i,2)=0.0;

    //point 0
    position(0,0)=(sqrt(2)*slabs_(2,0))-slabs_(1,0);
    position(0,1)=slabs_(1,0);
    //point 1
    position(1,0)=slabs_(1,0)-(sqrt(2)*slabs_(3,0));
    position(1,1)=slabs_(1,0);
    //point 2
    position(2,0)=slabs_(0,1);
    position(2,1)=slabs_(0,1)+(sqrt(2)*slabs_(3,0));
    //point 3
    position(3,0)=slabs_(0,1);
    position(3,1)=-slabs_(0,1)+(sqrt(2)*slabs_(2,1));
    //point 4
    position(4,0)=(sqrt(2)*slabs_(2,1))-slabs_(1,1);
    position(4,1)=slabs_(1,1);
    //point 5
    position(5,0)=slabs_(1,1)-(sqrt(2)*slabs_(3,1));
    position(5,1)=slabs_(1,1);
    //point 6
    position(6,0)=slabs_(0,0);
    position(6,1)=slabs_(0,0)+(sqrt(2)*slabs_(3,1));
    //point 7
    position(7,0)=slabs_(0,0);
    position(7,1)=-slabs_(0,0)+(sqrt(2)*slabs_(2,0));


    for (int i=0;i<(kdop_-1);i++)
    {
      gmshfilecontent <<"SL(" << scientific << position(i,0) << "," << position(i,1) << ","
                              << position(i,2) << "," << position(i+1,0) << "," << position(i+1,1) << ","
                              << position(i+1,2) << ")";
      gmshfilecontent << "{" << scientific << 0.0 << "," << 0.0 << "};" << endl;
    }
    gmshfilecontent << "SL(" << scientific << position(7,0) << "," << position(7,1) << ","
                  << position(7,2) << "," << position(0,0) << "," << position(0,1) << ","
                  << position(0,2) << ")";
    gmshfilecontent << "{" << scientific << 0.0 << "," << 0.0 << "};" << endl;
    fprintf(fp,gmshfilecontent.str().c_str());
    fclose(fp);
  }

  else if (dim_==3)
  {
    //PrintSlabs();
    //plot 3D-DOPs

    //defines coords of points defining k-DOP
    std::vector<std::vector<double> > coords;
    coords.resize(1);

    //trianglepoints[i] contains all needed points (i of coords[i]) to plot triangles
    std::vector<std::vector<int> > trianglepoints ;
    trianglepoints.resize(kdop_);

    double dcurrent;
    LINALG::Matrix<3,3> A;
    for (int i=0;i<kdop_/2;i++)
    {
      //for ismin & ismax of slabs
      for (int imm=0;imm<2;imm++)
      {
        for (int j=0;j<kdop_/2;j++)
        {
          for (int jmm=0;jmm<2;jmm++)
          {
            for (int k=0;k<kdop_/2;k++)
            {
              for (int kmm=0;kmm<2;kmm++)
              {
                double position[3];
                //define matrix A
                double norm0=sqrt((dopnormals_(i,0)*dopnormals_(i,0))+
                    (dopnormals_(i,1)*dopnormals_(i,1))+(dopnormals_(i,2)*dopnormals_(i,2)));
                double norm1=sqrt((dopnormals_(j,0)*dopnormals_(j,0))+
                    (dopnormals_(j,1)*dopnormals_(j,1))+(dopnormals_(j,2)*dopnormals_(j,2)));
                double norm2=sqrt((dopnormals_(k,0)*dopnormals_(k,0))+
                    (dopnormals_(k,1)*dopnormals_(k,1))+(dopnormals_(k,2)*dopnormals_(k,2)));
                // std::cout << endl << "norm0: " << norm0 << " 1: " << norm1 << " 2: " << norm2;
                A(0,0)=(dopnormals_(i,0))/norm0;
                A(0,1)=(dopnormals_(i,1))/norm0;
                A(0,2)=(dopnormals_(i,2))/norm0;
                A(1,0)=(dopnormals_(j,0))/norm1;
                A(1,1)=(dopnormals_(j,1))/norm1;
                A(1,2)=(dopnormals_(j,2))/norm1;
                A(2,0)=(dopnormals_(k,0))/norm2;
                A(2,1)=(dopnormals_(k,1))/norm2;
                A(2,2)=(dopnormals_(k,2))/norm2;

                //only if matrix a is not singular
                if (A.Determinant()!=0)
                {
                  A.Invert();
                  for (int m=0;m<3;m++)
                  {
                    position[m]=A(m,0)*slabs_(i,imm)+A(m,1)*slabs_(j,jmm)+A(m,2)*slabs_(k,kmm);
                  }
                  //check current position if its inside dops defined by slabs
                  bool isoutside=false;

                  for (int m=0;m<kdop_/2;m++)
                  {
                    dcurrent = (dopnormals_(m,0)*position[0]+dopnormals_(m,1)*position[1]+
                        dopnormals_(m,2)*position[2])/sqrt((dopnormals_(m,0)*dopnormals_(m,0))+
                            (dopnormals_(m,1)*dopnormals_(m,1))+(dopnormals_(m,2)*dopnormals_(m,2)));

                    if (dcurrent > (slabs_(m,1)+0.0001))
                      isoutside=true;
                    if (dcurrent < (slabs_(m,0)-0.0001))
                      isoutside=true;

                  }
                  //continue only if position is inside dop
                  if (!isoutside)
                  {
                    bool isinlist=false;
                    //check if current position is in coords-list

                    int currentsize=coords.size();

                    for (int m=0;m<currentsize;m++)
                    {
                      if (coords[m][0] < position[0]+0.0001 && coords[m][0] > position[0]-0.0001)
                      {
                        if (coords[m][1] < position[1]+0.0001 && coords[m][1] > position[1]-0.0001)
                        {
                          if (coords[m][2] < position[2]+0.0001 && coords[m][2] > position[2]-0.0001)
                          {
                            isinlist=true;
                            break;
                          }
                        }
                      }
                    }
                    if (!isinlist)
                    {
                    coords.resize(currentsize+1);
                    coords[currentsize-1].push_back(position[0]);
                    coords[currentsize-1].push_back(position[1]);
                    coords[currentsize-1].push_back(position[2]);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    //plot triangles
    //first look for points that are on max/min slab layer=trianglepoints
    for (int i=0;i<kdop_/2;i++)
    {
      for (int ismin=0;ismin<2;ismin++)
      {
        for (int j=0;j<(int)coords.size()-1;j++)
        {
          bool isonlayer=true;

          double dcurrent = (dopnormals_(i,0)*coords[j][0]+dopnormals_(i,1)*coords[j][1]+
              dopnormals_(i,2)*coords[j][2])/sqrt((dopnormals_(i,0)*dopnormals_(i,0))+
                  (dopnormals_(i,1)*dopnormals_(i,1))+(dopnormals_(i,2)*dopnormals_(i,2)));

          if (dcurrent>slabs_(i,ismin)+0.0001)
            isonlayer=false;
          if (dcurrent<slabs_(i,ismin)-0.0001)
            isonlayer=false;

          if (isonlayer)
            trianglepoints[(2*i)+ismin].push_back(j);
        }
      }
    }


    int count=0;
    //print k-DOP to gmsh-file
    for (int i=0;i<(int) trianglepoints.size();i++)
    {
      // l,m,n to find all possible combinations of points defining triangles
      for (int l=0; l<(int) trianglepoints[i].size(); l++)
      {
        for (int m=0; m<(int) trianglepoints[i].size(); m++)
        {
          for (int n=0; n<(int) trianglepoints[i].size(); n++)
          {
            if ( l!=m && l!=n && m!=n )
            {
              count++;
              //print triangle to gmsh file
              double position0[3],position1[3],position2[3];

              //set coords(vector) to position (double)
              for (int p=0;p<3;p++)
              {
                position0[p]=coords[trianglepoints[i][l]][p];
                position1[p]=coords[trianglepoints[i][m]][p];
                position2[p]=coords[trianglepoints[i][n]][p];
              }
              PlotGmshTriangle(filename,position0,position1,position2);

            }
          }
        }
     }
    }
    //std::cout << endl << "Number needed triangles to plot current treenode: " << count;

    //delete vector coords
    for (int i=0;i<(int)(coords.size())-1;i++)
      coords[i].clear();
    coords.clear();
    //delete vector trianglepoints
    for (int i=0;i<(int)(trianglepoints.size());i++)
      trianglepoints[i].clear();
    trianglepoints.clear();

  } //END 3D-case

  return;
}

/*----------------------------------------------------------------------*
 | Return coords for gmshpoint of 18DOP(public)               popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTreeNode::PlotGmshPoint(  std::string filename, double* position0, int nr)
{
  FILE* fp = NULL;
  fp = fopen(filename.c_str(), "a");
  std::stringstream gmshfilecontent;

  // plot quadrangle 0,1,2,3
  gmshfilecontent << "SP(" << scientific << position0[0] << "," << position0[1] << ","
                  << position0[2] <<  ")";
  gmshfilecontent << "{" << scientific << 0.0 << "," << 0.0 << "," << 0.0 << "," << 0.0 << "};" << endl;

  //plots nr of point
  gmshfilecontent << "T3(" << scientific << position0[0] << "," << position0[1] << ","
                  << position0[2] << "," << 17 << ")";
  gmshfilecontent << "{" << "SK" << nr << "};" << endl;
  fprintf(fp,gmshfilecontent.str().c_str());
  fclose(fp);

  return;
}

/*----------------------------------------------------------------------*
 | Plot quadrangle in gmsh(public)                            popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTreeNode::PlotGmshQuadrangle( std::string filename, double* position0,
                                                              double* position1,double* position2,
                                                              double* position3)
{
  FILE* fp = NULL;
  fp = fopen(filename.c_str(), "a");
  std::stringstream gmshfilecontent;

  // plot quadrangle 0,1,2,3
  gmshfilecontent << "SQ(" << scientific << position0[0] << "," << position0[1] << ","
                  << position0[2] << "," << position1[0] << "," << position1[1] << ","
                  << position1[2] << "," << position2[0] << "," << position2[1] << ","
                  << position2[2] << "," << position3[0] << "," << position3[1] << ","
                  << position3[2] << ")";
  gmshfilecontent << "{" << scientific << 0.0 << "," << 0.0 << "," << 0.0 << "," << 0.0 << "};" << endl;
  fprintf(fp,gmshfilecontent.str().c_str());
  fclose(fp);

  return;
}

/*----------------------------------------------------------------------*
 | Plot triangle in gmsh(public)                              popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTreeNode::PlotGmshTriangle( std::string filename, double* position0,
                                                              double* position1,double* position2)
{
  FILE* fp = NULL;
  fp = fopen(filename.c_str(), "a");
  std::stringstream gmshfilecontent;

  // plot triangle 0,1,2
  gmshfilecontent << "ST(" << scientific << position0[0] << "," << position0[1] << ","
                  << position0[2] << "," << position1[0] << "," << position1[1] << ","
                  << position1[2] << "," << position2[0] << "," << position2[1] << ","
                  << position2[2] << ")";
  gmshfilecontent << "{" << scientific << 0.0 << "," << 0.0 << "," << 0.0 << "};" << endl;
  fprintf(fp,gmshfilecontent.str().c_str());
  fclose(fp);

  return;
}

/*----------------------------------------------------------------------*
 | Set slabs of current treenode with new slabs(public)       popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTreeNode::SetSlabs(Epetra_SerialDenseMatrix& newslabs)
{
  for (int i=0;i<kdop_/2;++i)
  {
    slabs_(i,0)=newslabs(i,0);
    slabs_(i,1)=newslabs(i,1);
  }
}

/*----------------------------------------------------------------------*
 | Enlarge geometry of treenode (public)                      popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTreeNode::EnlargeGeometry(double& enlarge)
{
  //PrintSlabs();
  // scale slabs with Scalar enlarge
  for (int i=0;i<kdop_/2;++i)
  {
      slabs_(i,0)=slabs_(i,0)-enlarge;
      slabs_(i,1)=slabs_(i,1)+enlarge;
  }
  //PrintSlabs();
  return;
}

/*----------------------------------------------------------------------*
 |  ctor BinaryTree(public)                                   popp 10/08|
 *----------------------------------------------------------------------*/
MORTAR::BinaryTree::BinaryTree(DRT::Discretization& discret,
                               Teuchos::RCP<Epetra_Map> selements,
                               Teuchos::RCP<Epetra_Map> melements,
                               int dim, double eps) :
idiscret_(discret),
selements_(selements),
melements_(melements),
dim_(dim),
eps_(eps)
{
  // initialize sizes
  streenodesmap_.resize(1);
  mtreenodesmap_.resize(1);
  couplingmap_.resize(2);
  sleafsmap_.resize(2);
  mleafsmap_.resize(2);

  // claculates minimal element length
  SetEnlarge(true);

  //**********************************************************************
  // check for problem dimension
  //**********************************************************************
  if (dim_==2)
  {
    // set number of DOP sides to 8
    kdop_=8;

    // setup normals for DOP
    dopnormals_.Reshape(4,3);
    dopnormals_(0,0)= 1; dopnormals_(0,1)= 0; dopnormals_(0,2)= 0;
    dopnormals_(1,0)= 0; dopnormals_(1,1)= 1; dopnormals_(1,2)= 0;
    dopnormals_(2,0)= 1; dopnormals_(2,1)= 1; dopnormals_(2,2)= 0;
    dopnormals_(3,0)=-1; dopnormals_(3,1)= 1; dopnormals_(3,2)= 0;
  }
  else if (dim_==3)
  {
    // set number of DOP sides to  18
    kdop_=18;

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
  }
  else
    dserror("ERROR: Problem dimension must be 2D or 3D!");

  //**********************************************************************
  // initialize binary tree root nodes
  //**********************************************************************
  // create element lists
  std::vector<int> slist;
  std::vector<int> mlist;

  for (int i=0;i<selements_->NumMyElements();++i)
  {
    int gid = selements_->GID(i);
    slist.push_back(gid);
  }

  for (int i=0;i<melements_->NumMyElements();++i)
  {
    int gid = melements_->GID(i);
    mlist.push_back(gid);
  }

  // check slave root node case
  if (slist.size()>=2)
  {
    sroot_ = Teuchos::rcp(new BinaryTreeNode(MORTAR::SLAVE_INNER,idiscret_,sroot_ ,slist,DopNormals(),
             Kdop(),Dim(),0,streenodesmap_, mtreenodesmap_, sleafsmap_, mleafsmap_));

    // do initialization
    streenodesmap_[0].push_back(sroot_);
    sroot_->InitializeTree(enlarge_);
  }
  else if (slist.size()==1)
  {
    sroot_ = Teuchos::rcp(new BinaryTreeNode(MORTAR::SLAVE_LEAF,idiscret_,sroot_,slist,DopNormals(),
             Kdop(),Dim(),0,streenodesmap_, mtreenodesmap_, sleafsmap_, mleafsmap_));

    // trivial initialization
    streenodesmap_[0].push_back(sroot_);
    sleafsmap_[0].push_back(sroot_);
  }
  else
  {
    sroot_ = Teuchos::rcp(new BinaryTreeNode(MORTAR::NOSLAVE_ELEMENTS,idiscret_,sroot_,slist,DopNormals(),
             Kdop(),Dim(),0,streenodesmap_, mtreenodesmap_, sleafsmap_, mleafsmap_));

    // trivial initialization
    streenodesmap_[0].push_back(sroot_);
  }

  // check master root node case
  if (mlist.size()>=2)
  {
    mroot_ = Teuchos::rcp(new BinaryTreeNode(MORTAR::MASTER_INNER,idiscret_,mroot_,mlist,DopNormals(),
             Kdop(),Dim(),0,streenodesmap_, mtreenodesmap_, sleafsmap_, mleafsmap_));

    // do initialization
    mtreenodesmap_[0].push_back(mroot_);
    mroot_->InitializeTree(enlarge_);
  }
  else if (mlist.size()==1)
  {
    mroot_ = Teuchos::rcp(new BinaryTreeNode(MORTAR::MASTER_LEAF,idiscret_,mroot_,mlist,DopNormals(),
             Kdop(),Dim(),0,streenodesmap_, mtreenodesmap_, sleafsmap_, mleafsmap_));

    // trivial initialization
    mtreenodesmap_[0].push_back(mroot_);
    mleafsmap_[0].push_back(mroot_);
  }
  else
  {
    mroot_ = Teuchos::rcp(new BinaryTreeNode(MORTAR::NOMASTER_ELEMENTS,idiscret_,mroot_,mlist,DopNormals(),
             Kdop(),Dim(),0,streenodesmap_, mtreenodesmap_, sleafsmap_, mleafsmap_));

    // trivial initialization / error
    mtreenodesmap_[0].push_back(mroot_);
    dserror("No master element for Binarytree initialization on this processor");
  }

  /*
  // print binarytree to std::cout
  for (int k=0;k<Comm().NumProc();++k)
  {
    Comm().Barrier();
    if (Comm().MyPID()==k)
    {
      std::cout << "\n" << Comm().MyPID() << " Print tree with direct print function" << endl;
      std::cout <<"\n" <<Comm().MyPID()<< " Slave Tree:";
      PrintTree(sroot_);
      std::cout <<"\n" <<Comm().MyPID()<< " Master Tree:";
      PrintTree(mroot_);
    }
    Comm().Barrier();
  }

  for (int k=0;k<Comm().NumProc();++k)
  {
    Comm().Barrier();
    if (Comm().MyPID()==k)
    {
      std::cout << "\n" << Comm().MyPID() << " Print tree with print function of slave and master treemap" << endl;
      std::cout <<"\n" <<Comm().MyPID()<< " Slave Tree:";
      PrintTreeOfMap(streenodesmap_);
      std::cout <<"\n" <<Comm().MyPID()<< " Master Tree:";
      PrintTreeOfMap(mtreenodesmap_);
    }
    Comm().Barrier();
  }
  */

  return;
}


/*----------------------------------------------------------------------*
 | get communicator (public)                                  popp 10/08|
 *----------------------------------------------------------------------*/
const Epetra_Comm& MORTAR::BinaryTree::Comm() const
{
  return idiscret_.Comm();
}

/*----------------------------------------------------------------------*
 | Find min. length of master and slave elements (public)     popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTree::SetEnlarge(bool isinit)
{
  double lmin = 1.0e12;

  // calculate mininmal length of slave elements
  for (int i=0;i<selements_->NumMyElements();++i)
  {
    int gid = selements_->GID(i);
    DRT::Element* element = idiscret_.gElement(gid);
    if (!element) dserror("ERROR: Cannot find element with gid %\n",gid);
    MORTAR::MortarElement* mrtrelement = (MORTAR::MortarElement*) element;
    double mincurrent = mrtrelement->MinEdgeSize(isinit);
    if (mincurrent < lmin) lmin = mincurrent;
  }

  // calculate minimal length of master elements
  for (int i=0;i<melements_->NumMyElements();++i)
  {
    int gid = melements_->GID(i);
    DRT::Element* element = idiscret_.gElement(gid);
    if (!element) dserror("ERROR: Cannot find element with gid %\n",gid);
    MORTAR::MortarElement* mrtrelement = (MORTAR::MortarElement*) element;
    double mincurrent=mrtrelement->MinEdgeSize(isinit);
    if (mincurrent < lmin) lmin = mincurrent;
  }

  if (lmin<=0.0) dserror("ERROR: Minimal element length < 0!");

  // set the class variables
  minlengthele_= lmin;
  enlarge_ = eps_ * minlengthele_;

  return;
}

/*----------------------------------------------------------------------*
 | Print tree (public)                                        popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTree::PrintTree(Teuchos::RCP<BinaryTreeNode> treenode)
{
  // if treenode has no elements (NOSLAVE_ELEMENTS,NOMASTER_ELEMENTS)
  if (treenode->Type()==4 || treenode->Type()==5)
  {
    std::cout <<"\n" <<Comm().MyPID()<< " Tree has no element to print";
    return;
  }
  std::cout <<"\n" <<Comm().MyPID()<< " Tree at layer: " << treenode->Layer()<< " Elements: ";
  for (int i=0;i<(int)(treenode->Elelist().size());i++)
    std::cout <<" "<<treenode->Elelist()[i];

  // while treenode is inner node
  if (treenode->Type()==0 || treenode->Type()==2)
  {
    PrintTree(treenode->Leftchild());
    PrintTree(treenode->Rightchild());
  }

  return;
}

/*----------------------------------------------------------------------*
 | Print tree with treenodesmap_ (public)                     popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTree::PrintTreeOfMap(std::vector<std::vector<Teuchos::RCP<BinaryTreeNode> > >& treenodesmap)
{
  // print tree, elements listet in brackets (), belong to one treenode!
  for (int i=0;i<(int)(treenodesmap.size());i++)
  {
    std::cout <<"\n" <<Comm().MyPID()<< " Tree at layer: " << i<< " Elements: ";
    for (int k=0;k<(int)(treenodesmap[i].size());k++)
    {
      Teuchos::RCP<BinaryTreeNode> currentnode=treenodesmap[i][k];
      std::cout << " (";
      for (int l=0;l<(int)(currentnode->Elelist().size());l++)
      {
        std::cout << currentnode->Elelist()[l] << " ";
      }
      std::cout << ") ";
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 | Update tree topdown (public)                               popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTree::EvaluateUpdateTreeTopDown(Teuchos::RCP<BinaryTreeNode> treenode)
{
  //if no slave element on proc-->return
  if (treenode->Elelist().size()==0) return;

  treenode->CalculateSlabsDop(false);
  treenode->EnlargeGeometry(enlarge_);

  if (treenode->Type()==0||treenode->Type()==2)
  {
    EvaluateUpdateTreeTopDown(treenode->Leftchild());
    EvaluateUpdateTreeTopDown(treenode->Rightchild());
  }

  return;
}

/*----------------------------------------------------------------------*
 | Update tree bottom up based on list (public)               popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTree::EvaluateUpdateTreeBottomUp(std::vector<std::vector<Teuchos::RCP<BinaryTreeNode> > >& treenodesmap)
{
  // update tree bottom up (for every treelayer)
  for (int i=((int)(treenodesmap.size()-1));i>=0;i=i-1 )
  {
    for (int j=0;j<(int)(treenodesmap[i].size());j++)
      treenodesmap[i][j]->UpdateSlabsBottomUp(enlarge_);
  }

  return;
}

/*----------------------------------------------------------------------*
 | Search for contact (public)                                popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTree::EvaluateSearchSeparate(Teuchos::RCP<BinaryTreeNode> streenode,
                                                Teuchos::RCP<BinaryTreeNode> mtreenode)
{
  // tree needs to be updated before running contact search!

  // if there are no elements
  if (streenode->Type()==4 || mtreenode->Type()==5) return;

  // check if treenodes intercept
  // (they only intercept if ALL slabs intercept!)
  int nintercepts = 0;

  for (int i=0;i<kdop_/2;++i)
  {
    if (streenode->Slabs()(i,0) <= mtreenode->Slabs()(i,0))
    {
      if (streenode->Slabs()(i,1) >= mtreenode->Slabs()(i,0))
        nintercepts++;
      else if (streenode->Slabs()(i,1) >= mtreenode->Slabs()(i,1))
        nintercepts++;
    }
    else if (streenode->Slabs()(i,0) >= mtreenode->Slabs()(i,0))
    {
      if (mtreenode->Slabs()(i,1) >= streenode->Slabs()(i,1))
        nintercepts++;
      else if (mtreenode->Slabs()(i,1) >= streenode->Slabs()(i,0))
        nintercepts++;
    }
  }

  //treenodes intercept
  if (nintercepts==kdop_/2)
  {
    // slave and master treenodes are inner nodes
    if (streenode->Type()==0 && mtreenode->Type()==2)
    {
      //std::cout <<"\n"<< Comm().MyPID() << " 2 inner nodes!";
      EvaluateSearchSeparate(streenode->Leftchild(),mtreenode->Leftchild());
      EvaluateSearchSeparate(streenode->Leftchild(),mtreenode->Rightchild());
      EvaluateSearchSeparate(streenode->Rightchild(),mtreenode->Leftchild());
      EvaluateSearchSeparate(streenode->Rightchild(),mtreenode->Rightchild());
    }

    // slave treenode is inner, master treenode is leaf
    if (streenode->Type()==0 && mtreenode->Type()==3)
    {
      //std::cout <<"\n"<< Comm().MyPID() << " slafe inner, master leaf!";
      EvaluateSearchSeparate(streenode->Leftchild(),mtreenode);
      EvaluateSearchSeparate(streenode->Rightchild(),mtreenode);
    }

    // slave treenode is leaf,  master treenode is inner
    if (streenode->Type()==1 && mtreenode->Type()==2)
    {
      //std::cout <<"\n"<< Comm().MyPID() << " slave leaf, master inner!";
      EvaluateSearchSeparate(streenode,mtreenode->Leftchild());
      EvaluateSearchSeparate(streenode,mtreenode->Rightchild());
    }

    // both treenodes are leaf --> feasible pair
    if (streenode->Type()==1 && mtreenode->Type()==3)
    {
      int sgid = (int)streenode->Elelist()[0];    //global id of slave element
      int mgid = (int)mtreenode->Elelist()[0];    //global id of masterelement
      //std::cout <<"\n"<< Comm().MyPID() << "TreeDividedContact found between slave-Element: "
      //     << sgid <<"and master-Element: "<< mgid;
      DRT::Element* element= idiscret_.gElement(sgid);
      MORTAR::MortarElement* selement = static_cast<MORTAR::MortarElement*>(element);
      selement->AddSearchElements(mgid);
    }
  }

#ifdef MORTARGMSHCTN // for plotting contacting treenodes
  if (streenode->Type()==1 && mtreenode->Type()==3 && nintercepts==kdop_/2)
  {
    couplingmap_[0].push_back(streenode);
    couplingmap_[1].push_back(mtreenode);
  }
#endif // #ifdef MORTARGMSHCTN

  return;
}

/*----------------------------------------------------------------------*
 | Search for contact (public)                                popp 10/08|
 *----------------------------------------------------------------------*/
void MORTAR::BinaryTree::EvaluateSearchCombined(Teuchos::RCP<BinaryTreeNode> streenode,
                                                Teuchos::RCP<BinaryTreeNode> mtreenode)
{
  // root nodes need to be updated before running combined contact search!

  // if there are no elements
  if (streenode->Type()==4 || mtreenode->Type()==5) return;

  // check if treenodes intercept
  // (they only intercept if ALL slabs intercept!)
  int nintercepts = 0;

  for (int i=0;i<kdop_/2;++i)
  {
    if (streenode->Slabs()(i,0) <= mtreenode->Slabs()(i,0))
    {
      if (streenode->Slabs()(i,1) >= mtreenode->Slabs()(i,0))
        nintercepts++;
      else if (streenode->Slabs()(i,1) >= mtreenode->Slabs()(i,1))
        nintercepts++;
    }
    else if (streenode->Slabs()(i,0) >= mtreenode->Slabs()(i,0))
    {
      if (mtreenode->Slabs()(i,1) >= streenode->Slabs()(i,1))
        nintercepts++;
      else if (mtreenode->Slabs()(i,1) >= streenode->Slabs()(i,0))
        nintercepts++;
    }
  }

  //treenodes intercept
  if (nintercepts==kdop_/2)
  {
    // slave and master treenodes are inner nodes
    if (streenode->Type()==0 && mtreenode->Type()==2)
    {
      //std::cout <<"\n"<< Comm().MyPID() << " 2 inner nodes!";
      streenode->Leftchild()->CalculateSlabsDop(false);
      streenode->Leftchild()->EnlargeGeometry(enlarge_);
      streenode->Rightchild()->CalculateSlabsDop(false);
      streenode->Rightchild()->EnlargeGeometry(enlarge_);
      mtreenode->Leftchild()->CalculateSlabsDop(false);
      mtreenode->Leftchild()->EnlargeGeometry(enlarge_);
      mtreenode->Rightchild()->CalculateSlabsDop(false);
      mtreenode->Rightchild()->EnlargeGeometry(enlarge_);

      EvaluateSearchCombined(streenode->Leftchild(),mtreenode->Leftchild());
      EvaluateSearchCombined(streenode->Leftchild(),mtreenode->Rightchild());
      EvaluateSearchCombined(streenode->Rightchild(),mtreenode->Leftchild());
      EvaluateSearchCombined(streenode->Rightchild(),mtreenode->Rightchild());
    }

    // slave treenode is inner,  master treenode is leaf
    if (streenode->Type()==0 && mtreenode->Type()==3)
    {
      //std::cout <<"\n"<< Comm().MyPID() << " slafe inner, master leaf!";
      streenode->Leftchild()->CalculateSlabsDop(false);
      streenode->Leftchild()->EnlargeGeometry(enlarge_);
      streenode->Rightchild()->CalculateSlabsDop(false);
      streenode->Rightchild()->EnlargeGeometry(enlarge_);

      EvaluateSearchCombined(streenode->Leftchild(),mtreenode);
      EvaluateSearchCombined(streenode->Rightchild(),mtreenode);
    }

    // slave treenode is leaf,  master treenode is inner
    if (streenode->Type()==1 && mtreenode->Type()==2)
    {
      //std::cout <<"\n"<< Comm().MyPID() << " slave leaf, master inner!";
      mtreenode->Leftchild()->CalculateSlabsDop(false);
      mtreenode->Leftchild()->EnlargeGeometry(enlarge_);
      mtreenode->Rightchild()->CalculateSlabsDop(false);
      mtreenode->Rightchild()->EnlargeGeometry(enlarge_);

      EvaluateSearchCombined(streenode,mtreenode->Leftchild());
      EvaluateSearchCombined(streenode,mtreenode->Rightchild());
    }

    // both treenodes are leaf --> feasible pair
    if (streenode->Type()==1 && mtreenode->Type()==3)
    {
      //std::cout <<"\n"<< Comm().MyPID() << " 2 leaf nodes!";
      int sgid = (int)streenode->Elelist()[0]; //global id of slave element
      int mgid = (int)mtreenode->Elelist()[0]; //global id of master element
      //std::cout <<"\n"<< Comm().MyPID() << "TreeCombinedContact found between slave-Element: "
      //     << sgid <<"and master-Element: "<< mgid;
      DRT::Element* element= idiscret_.gElement(sgid);
      MORTAR::MortarElement* selement = static_cast<MORTAR::MortarElement*>(element);
      selement->AddSearchElements(mgid);
     }
  }

#ifdef MORTARGMSHCTN // for plotting contacting treenodes
  if (streenode->Type()==1 && mtreenode->Type()==3 && nintercepts==kdop_/2)
  {
    couplingmap_[0].push_back(streenode);
    couplingmap_[1].push_back(mtreenode);
  }
#endif // #ifdef MORTARGMSHCTN

  return;
}

