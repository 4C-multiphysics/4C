/*!----------------------------------------------------------------------
\file beam3ii.cpp
\brief three dimensional nonlinear corotational Timoshenko beam element

<pre>
Maintainer: Christian Cyron
            cyron@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15264
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "beam3ii.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_utils.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_inpar/drt_validparameters.H"
#include "../linalg/linalg_fixedsizematrix.H"
#include "../drt_fem_general/largerotations.H"
#include "../drt_lib/drt_linedefinition.H"

//namespace with utility functions for operations with large rotations used
using namespace LARGEROTATIONS;

DRT::ELEMENTS::Beam3iiType DRT::ELEMENTS::Beam3iiType::instance_;

DRT::ParObject* DRT::ELEMENTS::Beam3iiType::Create( const std::vector<char> & data )
{
  DRT::ELEMENTS::Beam3ii* object = new DRT::ELEMENTS::Beam3ii(-1,-1);
  object->Unpack(data);
  return object;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Beam3iiType::Create( const string eletype,
                                                            const string eledistype,
                                                            const int id,
                                                            const int owner )
{
  if ( eletype=="BEAM3II" )
  {
    Teuchos::RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::Beam3ii(id,owner));
    return ele;
  }
  return Teuchos::null;
}


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Beam3iiType::Create( const int id, const int owner )
{
  Teuchos::RCP<DRT::Element> ele = rcp(new DRT::ELEMENTS::Beam3ii(id,owner));
  return ele;
}


void DRT::ELEMENTS::Beam3iiType::NodalBlockInformation( DRT::Element * dwele, int & numdf, int & dimns, int & nv, int & np )
{
  numdf = 6;
  dimns = 6;
  nv = 6;
}

void DRT::ELEMENTS::Beam3iiType::ComputeNullSpace( DRT::Discretization & dis, std::vector<double> & ns, const double * x0, int numdf, int dimns )
{
  DRT::UTILS::ComputeBeam3DNullSpace( dis, ns, x0, numdf, dimns );
}

void DRT::ELEMENTS::Beam3iiType::SetupElementDefinition( std::map<std::string,std::map<std::string,DRT::INPUT::LineDefinition> > & definitions )
{
  std::map<std::string,DRT::INPUT::LineDefinition>& defs = definitions["BEAM3II"];

  defs["LINE2"]
    .AddIntVector("LINE2",2)
    .AddNamedInt("MAT")
    .AddNamedDouble("CROSS")
    .AddNamedDouble("SHEARCORR")
    .AddNamedDouble("IYY")
    .AddNamedDouble("IZZ")
    .AddNamedDouble("IRR")
    .AddNamedDoubleVector("TRIADS",6)
    ;

  defs["LIN2"]
    .AddIntVector("LIN2",2)
    .AddNamedInt("MAT")
    .AddNamedDouble("CROSS")
    .AddNamedDouble("SHEARCORR")
    .AddNamedDouble("IYY")
    .AddNamedDouble("IZZ")
    .AddNamedDouble("IRR")
    .AddNamedDoubleVector("TRIADS",6)
    ;

  defs["LINE3"]
    .AddIntVector("LINE3",3)
    .AddNamedInt("MAT")
    .AddNamedDouble("CROSS")
    .AddNamedDouble("SHEARCORR")
    .AddNamedDouble("IYY")
    .AddNamedDouble("IZZ")
    .AddNamedDouble("IRR")
    .AddNamedDoubleVector("TRIADS",9)
    ;

  defs["LIN3"]
    .AddIntVector("LIN3",3)
    .AddNamedInt("MAT")
    .AddNamedDouble("CROSS")
    .AddNamedDouble("SHEARCORR")
    .AddNamedDouble("IYY")
    .AddNamedDouble("IZZ")
    .AddNamedDouble("IRR")
    .AddNamedDoubleVector("TRIADS",9)
    ;

  defs["LINE4"]
    .AddIntVector("LINE4",4)
    .AddNamedInt("MAT")
    .AddNamedDouble("CROSS")
    .AddNamedDouble("SHEARCORR")
    .AddNamedDouble("IYY")
    .AddNamedDouble("IZZ")
    .AddNamedDouble("IRR")
    .AddNamedDoubleVector("TRIADS",12)
    ;

  defs["LIN4"]
    .AddIntVector("LIN4",4)
    .AddNamedInt("MAT")
    .AddNamedDouble("CROSS")
    .AddNamedDouble("SHEARCORR")
    .AddNamedDouble("IYY")
    .AddNamedDouble("IZZ")
    .AddNamedDouble("IRR")
    .AddNamedDoubleVector("TRIADS",12)
    ;

  defs["LINE5"]
    .AddIntVector("LINE5",5)
    .AddNamedInt("MAT")
    .AddNamedDouble("CROSS")
    .AddNamedDouble("SHEARCORR")
    .AddNamedDouble("IYY")
    .AddNamedDouble("IZZ")
    .AddNamedDouble("IRR")
    .AddNamedDoubleVector("TRIADS",15)
    ;

  defs["LIN5"]
    .AddIntVector("LIN5",5)
    .AddNamedInt("MAT")
    .AddNamedDouble("CROSS")
    .AddNamedDouble("SHEARCORR")
    .AddNamedDouble("IYY")
    .AddNamedDouble("IZZ")
    .AddNamedDouble("IRR")
    .AddNamedDoubleVector("TRIADS",15)
    ;
}


/*----------------------------------------------------------------------*
 |  ctor (public)                                            cyron 01/08|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Beam3ii::Beam3ii(int id, int owner) :
DRT::Element(id,owner),
isinit_(false),
nodeI_(0),
nodeJ_(0),
crosssec_(0),
crosssecshear_(0),
Iyy_(0),
Izz_(0),
Irr_(0),
jacobi_(0),
jacobimass_(0),
jacobinode_(0)
{
  return;
}
/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       cyron 01/08|
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Beam3ii::Beam3ii(const DRT::ELEMENTS::Beam3ii& old) :
 DRT::Element(old),
 isinit_(old.isinit_),
 Qconv_(old.Qconv_),
 Qold_(old.Qold_),
 Qnew_(old.Qnew_),
 Qconvmass_(old.Qconvmass_),
 Qnewmass_(old.Qnewmass_),
 dispthetaconv_(old.dispthetaconv_),
 dispthetaold_(old.dispthetaold_),
 dispthetanew_(old.dispthetanew_),
 kapparef_(old.kapparef_),
 nodeI_(old.nodeI_),
 nodeJ_(old.nodeJ_),
 crosssec_(old.crosssec_),
 crosssecshear_(old.crosssecshear_),
 Iyy_(old.Iyy_),
 Izz_(old.Izz_),
 Irr_(old.Irr_),
 jacobi_(old.jacobi_),
 jacobimass_(old.jacobimass_),
 jacobinode_(old.jacobinode_)
{
  return;
}
/*----------------------------------------------------------------------*
 |  Deep copy this instance of Beam3ii and return pointer to it (public) |
 |                                                            cyron 01/08 |
 *----------------------------------------------------------------------*/
DRT::Element* DRT::ELEMENTS::Beam3ii::Clone() const
{
  DRT::ELEMENTS::Beam3ii* newelement = new DRT::ELEMENTS::Beam3ii(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 |  dtor (public)                                            cyron 01/08 |
 *----------------------------------------------------------------------*/
DRT::ELEMENTS::Beam3ii::~Beam3ii()
{
  return;
}


/*----------------------------------------------------------------------*
 |  print this element (public)                              cyron 01/08
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Beam3ii::Print(ostream& os) const
{
  return;
}


/*----------------------------------------------------------------------*
 |                                                             (public) |
 |                                                          cyron 01/08 |
 *----------------------------------------------------------------------*/
DRT::Element::DiscretizationType DRT::ELEMENTS::Beam3ii::Shape() const
{
  int numnodes = NumNode();
  switch(numnodes)
  {
  	case 2:
			return line2;
			break;
  	case 3:
			return line3;
			break;
  	case 4:
  		return line4;
  		break;
  	case 5:
  		return line5;
  		break;
  	default:
			dserror("Only Line2, Line3 and Line4 elements are implemented.");
			break;
  }

  return dis_none;
}


/*----------------------------------------------------------------------*
 |  Pack data                                                  (public) |
 |                                                           cyron 01/08/
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Beam3ii::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);
  // add base class Element
  Element::Pack(data);

  //add all class variables of beam2r element
  AddtoPack(data,jacobi_);
  AddtoPack(data,jacobimass_);
  AddtoPack(data,jacobinode_);
  AddtoPack(data,nodeI_);
  AddtoPack(data,nodeJ_);
  AddtoPack(data,crosssec_);
  AddtoPack(data,crosssecshear_);
  AddtoPack(data,isinit_);
  AddtoPack(data,Irr_);
  AddtoPack(data,Iyy_);
  AddtoPack(data,Izz_);
  AddtoPack<4,1>(data,Qconv_);
  AddtoPack<4,1>(data,Qnew_);
  AddtoPack<4,1>(data,Qold_);
  AddtoPack<4,1>(data,Qconvmass_);
  AddtoPack<4,1>(data,Qnewmass_);
  AddtoPack<3,1>(data,dispthetaconv_);
  AddtoPack<3,1>(data,dispthetaold_);
  AddtoPack<3,1>(data,dispthetanew_);
  AddtoPack<3,1>(data,kapparef_);


  return;
}

/*----------------------------------------------------------------------*
 |  Unpack data                                                (public) |
 |                                                           cyron 01/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::Beam3ii::Unpack(const vector<char>& data)
{
  vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // extract base class Element
  vector<char> basedata(0);
  ExtractfromPack(position,data,basedata);
  Element::Unpack(basedata);


  //extract all class variables of beam3ii element
  ExtractfromPack(position,data,jacobi_);
  ExtractfromPack(position,data,jacobimass_);
  ExtractfromPack(position,data,jacobinode_);
  ExtractfromPack(position,data,nodeI_);
  ExtractfromPack(position,data,nodeJ_);
  ExtractfromPack(position,data,crosssec_);
  ExtractfromPack(position,data,crosssecshear_);
  isinit_ = ExtractInt(position,data);
  ExtractfromPack(position,data,Irr_);
  ExtractfromPack(position,data,Iyy_);
  ExtractfromPack(position,data,Izz_);
  ExtractfromPack<4,1>(position,data,Qconv_);
  ExtractfromPack<4,1>(position,data,Qnew_);
  ExtractfromPack<4,1>(position,data,Qold_);
  ExtractfromPack<4,1>(position,data,Qconvmass_);
  ExtractfromPack<4,1>(position,data,Qnewmass_);
  ExtractfromPack<3,1>(position,data,dispthetaconv_);
  ExtractfromPack<3,1>(position,data,dispthetaold_);
  ExtractfromPack<3,1>(position,data,dispthetanew_);
  ExtractfromPack<3,1>(position,data,kapparef_);


  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",(int)data.size(),position);
  return;
}

/*----------------------------------------------------------------------*
 |  get vector of lines (public)                          cyron 01/08|
 *----------------------------------------------------------------------*/
vector<RCP<DRT::Element> > DRT::ELEMENTS::Beam3ii::Lines()
{
  vector<RCP<Element> > lines(1);
  lines[0]= rcp(this, false);
  return lines;
}

/*----------------------------------------------------------------------*
 |determine Gauss rule from required type of integration                |
 |                                                   (public)cyron 09/09|
 *----------------------------------------------------------------------*/
DRT::UTILS::GaussRule1D DRT::ELEMENTS::Beam3ii::MyGaussRule(int nnode, IntegrationType integrationtype)
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


/*----------------------------------------------------------------------*
 |  Initialize (public)                                      cyron 01/08|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::Beam3iiType::Initialize(DRT::Discretization& dis)
{
	  //setting up geometric variables for beam3ii elements

	  for (int num=0; num<  dis.NumMyColElements(); ++num)
	  {
	    //in case that current element is not a beam3ii element there is nothing to do and we go back
	    //to the head of the loop
	    if (dis.lColElement(num)->ElementType() != *this) continue;

	    //if we get so far current element is a beam3ii element and  we get a pointer at it
	    DRT::ELEMENTS::Beam3ii* currele = dynamic_cast<DRT::ELEMENTS::Beam3ii*>(dis.lColElement(num));
	    if (!currele) dserror("cast to Beam3ii* failed");

	    //reference node position
	    vector<double> xrefe;
	    vector<double> rotrefe;
	    const int nnode= currele->NumNode();

	    //resize xrefe for the number of coordinates we need to store
	    xrefe.resize(3*nnode);
	    rotrefe.resize(3*nnode);

	    //getting element's nodal coordinates and treating them as reference configuration
	    if (currele->Nodes()[0] == NULL || currele->Nodes()[1] == NULL)
	      dserror("Cannot get nodes in order to compute reference configuration'");
	    else
	    {
	      for (int node=0; node<nnode; node++) //element has k nodes
	        for(int dof= 0; dof < 3; dof++)// element node has three coordinates x1, x2 and x3
	        {
	        	xrefe[node*3 + dof] = currele->Nodes()[node]->X()[dof];
	        	rotrefe[node*3 + dof]= 0.0;
	        }
	    }

	    //SetUpReferenceGeometry is a templated function
	    switch(nnode)
	    {
	  		case 2:
	  		{
	  			currele->SetUpReferenceGeometry<2>(xrefe,rotrefe);
	  			break;
	  		}
	  		case 3:
	  		{
	  			currele->SetUpReferenceGeometry<3>(xrefe,rotrefe);
	  			break;
	  		}
	  		case 4:
	  		{
	  			currele->SetUpReferenceGeometry<4>(xrefe,rotrefe);
	  			break;
	  		}
	  		case 5:
	  		{
	  			currele->SetUpReferenceGeometry<5>(xrefe,rotrefe);
	  			break;
	  		}
	  		default:
	  			dserror("Only Line2, Line3, Line4 and Line5 Elements implemented.");
	  	}

	  } //for (int num=0; num<dis_.NumMyColElements(); ++num)

	  return 0;
}


#endif  // #ifdef CCADISCRET
