/*!----------------------------------------------------------------------
\file truss3.cpp
\brief three dimensional total Lagrange truss element

<pre>
Maintainer: Christian Cyron
            cyron@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15264
</pre>

*----------------------------------------------------------------------*/

#include "truss3.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_utils_nullspace.H"
//#include "../linalg/linalg_fixedsizematrix.H"
#include "../drt_lib/drt_linedefinition.H"

DRT::ELEMENTS::Truss3Type DRT::ELEMENTS::Truss3Type::instance_;


DRT::ParObject* DRT::ELEMENTS::Truss3Type::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::Truss3* object = new DRT::ELEMENTS::Truss3(-1,-1);
  object->Unpack(data);
  return object;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Truss3Type::Create( const std::string eletype,
                                                              const std::string eledistype,
                                                              const int         id,
                                                              const int         owner )
  {
  if ( eletype=="TRUSS3" )
  {
    Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(new DRT::ELEMENTS::Truss3(id,owner));
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Truss3Type::Create( const int id, const int owner )
{
  Teuchos::RCP<DRT::Element> ele = Teuchos::rcp(new DRT::ELEMENTS::Truss3(id,owner));
  return ele;
}


void DRT::ELEMENTS::Truss3Type::NodalBlockInformation( DRT::Element * dwele, int & numdf, int & dimns, int & nv, int & np )
{
  numdf = 3;
  dimns = 6;
  nv = 3;
}

void DRT::ELEMENTS::Truss3Type::ComputeNullSpace( DRT::Discretization & dis, std::vector<double> & ns, const double * x0, int numdf, int dimns )
{
  DRT::UTILS::ComputeStructure3DNullSpace( dis, ns, x0, numdf, dimns );
}

void DRT::ELEMENTS::Truss3Type::SetupElementDefinition( std::map<std::string,std::map<std::string,DRT::INPUT::LineDefinition> > & definitions )
{
  std::map<std::string,DRT::INPUT::LineDefinition>& defs = definitions["TRUSS3"];

  defs["LINE2"]
    .AddIntVector("LINE2",2)
    .AddNamedInt("MAT")
    .AddNamedDouble("CROSS")
    .AddNamedString("KINEM")
    ;

  defs["LIN2"]
    .AddIntVector("LIN2",2)
    .AddNamedInt("MAT")
    .AddNamedDouble("CROSS")
    .AddNamedString("KINEM")
    ;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            cyron 08/08|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Truss3::Truss3(int id, int owner) :
DRT::Element(id,owner),
data_(),
isinit_(false),
material_(0),
lrefe_(0),
crosssec_(0),
kintype_(tr3_totlag),
//note: for corotational approach integration for Neumann conditions only
//hence enough to integrate 3rd order polynomials exactly
gaussrule_(DRT::UTILS::intrule_line_2point)
{
  return;
}
/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       cyron 08/08|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Truss3::Truss3(const DRT::ELEMENTS::Truss3& old) :
 DRT::Element(old),
 data_(old.data_),
 isinit_(old.isinit_),
 X_(old.X_),
 material_(old.material_),
 lrefe_(old.lrefe_),
 jacobimass_(old.jacobimass_),
 jacobinode_(old.jacobinode_),
 crosssec_(old.crosssec_),
 kintype_(old. kintype_),
 gaussrule_(old.gaussrule_)
{
  return;
}
/*----------------------------------------------------------------------*
 |  Deep copy this instance of Truss3 and return pointer to it (public) |
 |                                                            cyron 08/08|
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::Truss3::Clone() const
{
  DRT::ELEMENTS::Truss3* newelement = new DRT::ELEMENTS::Truss3(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  dtor (public)                                            cyron 08/08|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Truss3::~Truss3()
{
  return;
}


/*----------------------------------------------------------------------*
 |  print this element (public)                              cyron 08/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Truss3::Print(std::ostream& os) const
{
  os << "Truss3 ";
  Element::Print(os);
  os << " gaussrule_: " << gaussrule_ << " ";
  return;
}


/*----------------------------------------------------------------------*
 |(public)                                                   cyron 08/08|
 *----------------------------------------------------------------------*/
DRT::Element::DiscretizationType DRT::ELEMENTS::Truss3::Shape() const
{
  return line2;
}


/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                           cyron 08/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Truss3::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class Element
  Element::Pack(data);
  AddtoPack(data,isinit_);
  AddtoPack(data,X_);
  AddtoPack(data,material_);
  AddtoPack(data,lrefe_);
  AddtoPack(data,jacobimass_);
  AddtoPack(data,jacobinode_);
  AddtoPack(data,crosssec_);
  AddtoPack(data,gaussrule_); //implicit conversion from enum to integer
  AddtoPack(data,kintype_);
  AddtoPack(data,data_);

  return;
}


/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                           cyron 08/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Truss3::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // extract base class Element
  std::vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  Element::Unpack(basedata);
  isinit_ = ExtractInt(position,data);
  ExtractfromPack(position,data,X_);
  ExtractfromPack(position,data,material_);
  ExtractfromPack(position,data,lrefe_);
  ExtractfromPack(position,data,jacobimass_);
  ExtractfromPack(position,data,jacobinode_);
  ExtractfromPack(position,data,crosssec_);
  // gaussrule_
  int gausrule_integer;
  ExtractfromPack(position,data,gausrule_integer);
  gaussrule_ = DRT::UTILS::GaussRule1D(gausrule_integer); //explicit conversion from integer to enum
  // kinematic type
  kintype_ = static_cast<KinematicType>( ExtractInt(position,data) );
  std::vector<char> tmp(0);
  ExtractfromPack(position,data,tmp);
  data_.Unpack(tmp);

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                              cyron 08/08|
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<DRT::Element> > DRT::ELEMENTS::Truss3::Lines()
{
  std::vector<Teuchos::RCP<Element> > lines(1);
  lines[0]= Teuchos::rcp(this, false);
  return lines;
}

/*----------------------------------------------------------------------*
 |determine Gauss rule from required type of integration                |
 |                                                   (public)cyron 09/09|
 *----------------------------------------------------------------------*/
DRT::UTILS::GaussRule1D DRT::ELEMENTS::Truss3::MyGaussRule(int nnode, IntegrationType integrationtype)
{
  DRT::UTILS::GaussRule1D gaussrule = DRT::UTILS::intrule1D_undefined;

  switch(nnode)
  {
    case 2:
    {
      switch(integrationtype)
      {
        case gaussexactintegration:
        {
          gaussrule = DRT::UTILS::intrule_line_2point;
          break;
        }
        case gaussunderintegration:
        {
          gaussrule =  DRT::UTILS::intrule_line_1point;
          break;
        }
        case lobattointegration:
        {
          gaussrule =  DRT::UTILS::intrule_line_lobatto2point;
          break;
        }
        default:
          dserror("unknown type of integration");
      }
      break;
    }
    case 3:
    {
      switch(integrationtype)
      {
        case gaussexactintegration:
        {
          gaussrule = DRT::UTILS::intrule_line_3point;
          break;
        }
        case gaussunderintegration:
        {
          gaussrule =  DRT::UTILS::intrule_line_2point;
          break;
        }
        case lobattointegration:
        {
          gaussrule =  DRT::UTILS::intrule_line_lobatto3point;
          break;
        }
        default:
          dserror("unknown type of integration");
      }
      break;
    }
    case 4:
    {
      switch(integrationtype)
      {
        case gaussexactintegration:
        {
          gaussrule = DRT::UTILS::intrule_line_4point;
          break;
        }
        case gaussunderintegration:
        {
          gaussrule =  DRT::UTILS::intrule_line_3point;
          break;
        }
        default:
          dserror("unknown type of integration");
      }
      break;
    }
    case 5:
    {
      switch(integrationtype)
      {
        case gaussexactintegration:
        {
          gaussrule = DRT::UTILS::intrule_line_5point;
          break;
        }
        case gaussunderintegration:
        {
          gaussrule =  DRT::UTILS::intrule_line_4point;
          break;
        }
        default:
          dserror("unknown type of integration");
      }
      break;
    }
    default:
      dserror("Only Line2, Line3, Line4 and Line5 Elements implemented.");
  }

  return gaussrule;
}

void DRT::ELEMENTS::Truss3::SetUpReferenceGeometry(const std::vector<double>& xrefe, const bool secondinit)
{
  /*this method initializes geometric variables of the element; the initilization can usually be applied to elements only once;
   *therefore after the first initilization the flag isinit is set to true and from then on this method does not take any action
   *when called again unless it is called on purpose with the additional parameter secondinit. If this parameter is passed into
   *the method and is true the element is initialized another time with respective xrefe;
   *note: the isinit_ flag is important for avoiding reinitialization upon restart. However, it should be possible to conduct a
   *second initilization in principle (e.g. for periodic boundary conditions*/
  if(!isinit_ || secondinit)
  {
    isinit_ = true;

    //setting reference coordinates
    for(int i=0;i<6;i++)
      X_(i) = xrefe[i];

    //length in reference configuration
    lrefe_ = std::pow(pow(X_(3)-X_(0),2)+pow(X_(4)-X_(1),2)+pow(X_(5)-X_(2),2),0.5);

    //set jacobi determinants for integration of mass matrix and at nodes
    jacobimass_.resize(2);
    jacobimass_[0] = lrefe_ / 2.0;
    jacobimass_[1] = lrefe_ / 2.0;
    jacobinode_.resize(2);
    jacobinode_[0] = lrefe_ / 2.0;
    jacobinode_[1] = lrefe_ / 2.0;
  }

  return;
}


int DRT::ELEMENTS::Truss3Type::Initialize(DRT::Discretization& dis)
{
  //reference node positions
  std::vector<double> xrefe;

  //resize xrefe for the number of coordinates we need to store
  xrefe.resize(3*2);

  //setting beam reference director correctly
  for (int i=0; i<  dis.NumMyColElements(); ++i)
  {
    //in case that current element is not a beam3 element there is nothing to do and we go back
    //to the head of the loop
    if (dis.lColElement(i)->ElementType() != *this) continue;

    //if we get so far current element is a beam3 element and  we get a pointer at it
    DRT::ELEMENTS::Truss3* currele = dynamic_cast<DRT::ELEMENTS::Truss3*>(dis.lColElement(i));
    if (!currele) dserror("cast to Truss3* failed");

    //getting element's nodal coordinates and treating them as reference configuration
    if (currele->Nodes()[0] == NULL || currele->Nodes()[1] == NULL)
      dserror("Cannot get nodes in order to compute reference configuration'");
    else
    {
      for (int k=0; k<2; k++) //element has two nodes
        for(int l= 0; l < 3; l++)
          xrefe[k*3 + l] = currele->Nodes()[k]->X()[l];
    }

    currele->SetUpReferenceGeometry(xrefe);


  } //for (int i=0; i<dis_.NumMyColElements(); ++i)


  return 0;
}


