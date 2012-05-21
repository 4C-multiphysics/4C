#include "line_integration.H"
#include "boundarycell_integration.H"

#include "cut_boundarycell.H"
#include "cut_triangulateFacet.H"
#include "cut_kernel.H"

/*-----------------------------------------------------------------------------------------------------*
      compute the equation of the plane Ax+By+Cz=D with the local coordinates of corner points
*------------------------------------------------------------------------------------------------------*/
std::vector<double> GEO::CUT::FacetIntegration::equation_plane(const std::vector<std::vector<double> > cornersLocal)
{
  std::vector<double> eqn_plane(4);

  double x1[3]={0.0,0.0,0.0},y1[3]={0.0,0.0,0.0},z1[3]={0.0,0.0,0.0};
  unsigned mm=0;
  for(std::vector<std::vector<double> >::const_iterator k=cornersLocal.begin();k!=cornersLocal.end();k++)
  {
    const std::vector<double> coords = *k;

    x1[mm] = coords[0];
    y1[mm] = coords[1];
    z1[mm] = coords[2];

//make sure all three points do not lie on a line
//construct a vector pt2-pt1 and pt3-pt1. If the cross product is zero then all are in same line
//go to next point in such cases
    if(mm==2)
    {
      double pt1pt2[3],pt1pt3[3],cross[3];
      pt1pt2[0] = x1[1]-x1[0];pt1pt3[0]=x1[2]-x1[0];
      pt1pt2[1] = y1[1]-y1[0];pt1pt3[1]=y1[2]-y1[0];
      pt1pt2[2] = z1[1]-z1[0];pt1pt3[2]=z1[2]-z1[0];
      cross[0] = fabs(pt1pt2[1]*pt1pt3[2]-pt1pt2[2]*pt1pt3[1]);
      cross[1] = fabs(pt1pt2[0]*pt1pt3[2]-pt1pt2[2]*pt1pt3[0]);
      cross[2] = fabs(pt1pt2[1]*pt1pt3[0]-pt1pt2[0]*pt1pt3[1]);
      if(cross[0]<0.000001 && cross[1]<0.000001 && cross[2]<0.000001)
        continue;
      else
        mm++;
      }
      else
        mm++;
      //3 points are sufficient to find the equation of plane
      if(mm==3)
        break;
      if(mm==cornersLocal.size())
        dserror("All points of a facet are on a line");
  }
  eqn_plane[0] = y1[0]*(z1[1]-z1[2])+y1[1]*(z1[2]-z1[0])+y1[2]*(z1[0]-z1[1]);
  eqn_plane[1] = z1[0]*(x1[1]-x1[2])+z1[1]*(x1[2]-x1[0])+z1[2]*(x1[0]-x1[1]);
  eqn_plane[2] = x1[0]*(y1[1]-y1[2])+x1[1]*(y1[2]-y1[0])+x1[2]*(y1[0]-y1[1]);
  eqn_plane[3] = x1[0]*(y1[1]*z1[2]-y1[2]*z1[1])+x1[1]*(y1[2]*z1[0]-y1[0]*z1[2])+x1[2]*(y1[0]*z1[1]-y1[1]*z1[0]);

  return eqn_plane;
}

/*---------------------------------------------------------------------------------------------------------------------*
          compute only the x-component of unit-normal vector which is used in further computations
    also determine whether the plane is numbered in clockwise or anticlockwise sense when seen away from the face
*----------------------------------------------------------------------------------------------------------------------*/
void GEO::CUT::FacetIntegration::IsClockwise( const std::vector<double> eqn_plane,
                                              const std::vector<std::vector<double> > cornersLocal )
{
#if 0 //new generalized method
  clockwise_ = 0;
  Side* parent = face1_->ParentSide();
  const std::vector<Node*> &par_nodes = parent->Nodes();
  std::vector<vector<double> > corners(par_nodes.size());
  int mm=0;
  //std::cout<<"parent side\t"<<"element id = "<<elem1_->Id()<<"\n";
  for(std::vector<Node*>::const_iterator i=par_nodes.begin();i!=par_nodes.end();i++)
  {
    Node *nod = *i;
    double x1[3];
    nod->Coordinates(x1);
    LINALG::Matrix<3,1> glo,loc;
    std::vector<double> pt_local(3);
    for(int nodno=0;nodno<3;nodno++)
      glo(nodno,0) = x1[nodno];

     elem1_->LocalCoordinates(glo,loc);

     pt_local[0] = loc(0,0);
     pt_local[1] = loc(1,0);
     pt_local[2] = loc(2,0);

    //std::cout<<loc(0,0)<<"\t"<<loc(1,0)<<"\t"<<loc(2,0)<<"\n";

     corners[mm] = pt_local;
     mm++;
    }
//        std::cout<<"\n";
    std::vector<double> eqn_par = equation_plane(corners);

    bool iscut = face1_->OnCutSide();
    if(iscut)
    {
      if(position_==-2)
      {
        if(eqn_plane[0]*eqn_par[0]<0.0)
          clockwise_ = 1;
      }
      else
      {
        if(eqn_plane[0]*eqn_par[0]>0.0)
          clockwise_ = 1;
      }
    }
    else
    {
      if(eqn_plane[0]*eqn_par[0]<0.0)
        clockwise_ = 1;
    }
    //std::cout<<"clockwise = "<<clockwise_<<"\t"<<"is cut side = "<<iscut<<"\n";
#endif

#if 1 //old method of checking the ordering - separate check for different background elements (is this necessary?)
  std::string ordering;

  if( cornersLocal.size()==3 )
  {
    if(eqn_plane[0]>0.0)
      ordering = "acw";
    else
      ordering = "cw";
  }
#if 1
  else
  {
    double crossProd = 0.0;
    for(unsigned i=0;i<cornersLocal.size();i++)
    {
      unsigned j = (i+1)%cornersLocal.size();
      crossProd += (cornersLocal[j][1]-cornersLocal[i][1])*(cornersLocal[j][2]+cornersLocal[i][2]);
    }
    if(crossProd>0)
      ordering = "cw";
    else if(crossProd<0)
      ordering = "acw";
    else
      dserror("the points in the facet are neither ordered in clockwise not anti-clockwise or all collinear"
          " point in this facet");
  }
#else
  else
  {
    double crossProd = 0.0;
    int count=0;
    for(unsigned i=0;i<cornersLocal.size();i++)
    {
      unsigned j = (i+1)%cornersLocal.size();
      unsigned k = (i+2)%cornersLocal.size();
      crossProd = (cornersLocal[j][1]-cornersLocal[i][1])*(cornersLocal[k][2]-cornersLocal[j][2]);
      crossProd -= (cornersLocal[j][2]-cornersLocal[i][2])*(cornersLocal[k][1]-cornersLocal[j][1]);
      std::cout<<"cross = "<<crossProd<<"\n";
      if(crossProd<0)
        count--;
      else if(crossProd>0)
        count++;

    }
    if(count>0)
      ordering = "acw";
    else if(count<0)
      ordering = "cw";
    else
      dserror("the points in the facet are neither ordered in clockwise not anti-clockwise or all collinear"
          " point in this facet");
  }
#endif

  clockwise_ = 0;
  Side* parent = face1_->ParentSide();
  const std::vector<Node*> &par_nodes = parent->Nodes();
  std::vector<vector<double> > corners(par_nodes.size());
  int mm=0;

  for(std::vector<Node*>::const_iterator i=par_nodes.begin();i!=par_nodes.end();i++)
  {
    Node *nod = *i;
    double x1[3];
    nod->Coordinates(x1);
    LINALG::Matrix<3,1> glo,loc;
    std::vector<double> pt_local(3);
    for(int nodno=0;nodno<3;nodno++)
      glo(nodno,0) = x1[nodno];

     elem1_->LocalCoordinates(glo,loc);

     pt_local[0] = loc(0,0);
     pt_local[1] = loc(1,0);
     pt_local[2] = loc(2,0);

     corners[mm] = pt_local;
     mm++;
    }

    std::vector<double> eqn_par = equation_plane(corners);

    // the cut sides are always convex (????)
    // just comparing the sign of coordinates of eqn of plane will do
    bool iscut = face1_->OnCutSide();
    if(iscut)
    {
			if(position_==-2)
			{
        if(eqn_plane[0]*eqn_par[0]<0.0)
          clockwise_ = 1;
			}
			else
			{
        if(eqn_plane[0]*eqn_par[0]>0.0)
          clockwise_ = 1;
			}
    }
    else
    {
			const std::vector<Side*> &ele_sides = elem1_->Sides();

			int parentSideno = 0;
			for(std::vector<Side*>::const_iterator i=ele_sides.begin();i!=ele_sides.end();i++)
			{
        Side*sss = *i;
        if(sss==parent)
        {
          break;
        }
        parentSideno++;

			}

//should check whether this is sufficient or do we need to find the number of side in the element and
//use their orientation to get clockwise ordering
//                std::cout<<"parent side no = "<<parentSideno<<"\n";
//			std::cout<<"parentSideno = "<<parentSideno<<"corner = "<<corners[0][0]<<"\n";
//ParentSideno=1 is x=1 face and 3 is x=-1 face
#if 0 //this is only for hex
			if(parentSideno==1 && eqn_plane_[0]<0.0)
				clockwise_ = 1;
			if(parentSideno==3 && eqn_plane_[0]>0.0)
				clockwise_ = 1;
#endif
    switch ( elem1_->Shape() )
    {
      case DRT::Element::hex8:
      {
        if(parentSideno==1 && ordering=="cw")
          clockwise_ = 1;
        if(parentSideno==3 && ordering=="acw")
          clockwise_ = 1;
        break;
      }
      case DRT::Element::tet4:
      {
        if(parentSideno==1 && ordering=="cw")
         clockwise_ = 1;
       if(parentSideno==2 && ordering=="acw")
         clockwise_ = 1;
        break;
      }
      case DRT::Element::wedge6:
      {
        if(parentSideno==0 && ordering=="cw")
         clockwise_ = 1;
       if(parentSideno==1 && ordering=="acw")
         clockwise_ = 1;
        break;
      }
      case DRT::Element::pyramid5:
      {
        if(parentSideno==1 && ordering=="cw")
          clockwise_ = 1;
        if(parentSideno==3 && ordering=="acw")
          clockwise_ = 1;
        break;
      }
      default:
        throw std::runtime_error( "unsupported integration cell type" );
    }

			/*if(corners[0][0]==1 && eqn_plane_[0]<0.0)
				clockwise_ = 1;
			if(corners[0][0]==-1 && eqn_plane_[0]>0.0)
				clockwise_ = 1;*/
  }

#endif

}

/*-----------------------------------------------------------------------------------------------*
                            computes x=f(y,z) from the plane equation
                  equation of this form is used to replace x in the line integral
*------------------------------------------------------------------------------------------------*/
std::vector<double> GEO::CUT::FacetIntegration::compute_alpha( std::vector<double> eqn_plane,
                                                               std::string intType )
{
  std::vector<double> alfa(3);
  double a = eqn_plane[0];
  double b = eqn_plane[1];
  double c = eqn_plane[2];
  double d = eqn_plane[3];


  if(intType=="x")
  {
    alfa[0] = d/a;
    alfa[1] = -1.0*b/a;
    alfa[2] = -1.0*c/a;
  }
  else if(intType=="y")
  {
    alfa[0] = d/b;
    alfa[1] = -1.0*c/b;
    alfa[2] = -1.0*a/b;
  }
  else if(intType=="z")
  {
    alfa[0] = d/c;
    alfa[1] = -1.0*a/c;
    alfa[2] = -1.0*b/c;
  }
  else
  {
    dserror("The facet integration type undefined");
    exit(1);
  }
  return alfa;
}

/*---------------------------------------------------------------------------------*
      return the absolute normal of the facet in a particular direction
*----------------------------------------------------------------------------------*/
double GEO::CUT::FacetIntegration::getNormal(std::string intType)
{
  double normalScale=0.0;
  for(unsigned i=0;i<3;i++)
    normalScale += eqn_plane_[i]*eqn_plane_[i];
  normalScale = sqrt(normalScale);

  if(intType=="x")
    return eqn_plane_[0]/normalScale;
  else if(intType=="y")
    return eqn_plane_[1]/normalScale;
  else if(intType=="z")
    return eqn_plane_[2]/normalScale;
  else
  {
    dserror("The normal direction is unspecified");
    return 0.0;
  }
}

/*-------------------------------------------------------------------------------------*
                      perform integration over the facet
*--------------------------------------------------------------------------------------*/
double GEO::CUT::FacetIntegration::integrate_facet()
{
		std::vector<std::vector<double> > cornersLocal = face1_->CornerPointsLocal(elem1_);
		if(global_==true)
		{
		  std::vector<Point*>co =  face1_->CornerPoints();
      for(std::vector<Point*>::iterator i=co.begin();i!=co.end();i++)
      {
        Point* po = *i;
        double xo[3];
        po->Coordinates(xo);
        for(unsigned j=0;j<3;j++)
          cornersLocal[i-co.begin()][j] = xo[j];
      }
		}

    eqn_plane_ = equation_plane(cornersLocal);

// the face is in the x-y or in y-z plane which gives zero facet integral
    if(fabs(eqn_plane_[0])<0.0000001 && bcellInt_==false)
      return 0.0;
//x=0 plane which also do not contribute to facet integral
    if(fabs(eqn_plane_[1])<0.0000001 && fabs(eqn_plane_[2])<0.0000001 && fabs(eqn_plane_[3])<0.0000001 && bcellInt_==false)
      return 0.0;

    if(bcellInt_==true) // the integral value of boundarycell will not change w.r.t the ordering of vertices
      clockwise_ = false;
    else
      IsClockwise(eqn_plane_,cornersLocal);

    //integrating over each line of the facet
    double facet_integ = 0.0;
    if(bcellInt_==false)
    {
      std::vector<double> alpha;
      alpha = compute_alpha(eqn_plane_,"x");
			for(std::vector<std::vector<double> >::const_iterator k=cornersLocal.begin();k!=cornersLocal.end();k++)
			{
				const std::vector<double> coords1 = *k;
				std::vector<double> coords2;
				//for the last line the end point is the first point of the facet
				if(k!=(cornersLocal.end()-1))
          coords2 = *(k+1);
				else
          coords2= *(cornersLocal.begin());

  //first index decides the x or y coordinate, second index decides the start point or end point
				LINALG::Matrix<2,2> coordLine;

	//The facet is projected over y-z plane and then the integration is performed
	//so only y- and z-coordinates are passed to make the lines
	//[0]-- indicates y and [1] indicate z because we are now in y-z plane
				coordLine(0,0) = coords1[1];
				coordLine(1,0) = coords1[2];
				coordLine(0,1) = coords2[1];
				coordLine(1,1) = coords2[2];

				LineIntegration line1(coordLine,inte_num_,alpha,bcellInt_);
				facet_integ += line1.integrate_line();
			}
    }
    else
    {
//to reduce the truncation error introduced during the projection of plane,
//the plane, along which the normal component is maximum, is chosen
      std::string projType;
			if(fabs(eqn_plane_[0])<1e-8)
			{
				if(fabs(eqn_plane_[1])<1e-8)
					projType = "z";
				else if(fabs(eqn_plane_[2])<1e-8)
					projType = "y";
				else
				{
					if(fabs(eqn_plane_[1])>fabs(eqn_plane_[2]))
						projType = "y";
					else
						projType = "z";
				}
			}
			else if(fabs(eqn_plane_[1])<1e-8)
			{
				if(fabs(eqn_plane_[2])<1e-8)
					projType = "x";
				else
				{
					if(fabs(eqn_plane_[0])>fabs(eqn_plane_[2]))
						projType = "x";
					else
						projType = "z";
				}
			}
			else if(fabs(eqn_plane_[2])<1e-8)
			{
				if(fabs(eqn_plane_[1])>fabs(eqn_plane_[0]))
					projType = "y";
				else
					projType = "x";
			}
			else
			{
				if(fabs(eqn_plane_[0])>=fabs(eqn_plane_[1]) && fabs(eqn_plane_[0])>=fabs(eqn_plane_[2]))
					projType = "x";
				else if(fabs(eqn_plane_[1])>=fabs(eqn_plane_[2]) && fabs(eqn_plane_[1])>=fabs(eqn_plane_[0]))
					projType = "y";
				else
					projType = "z";
			}

			if(projType!="x" && projType!="y" && projType!="z")
			{
				dserror("projection plane is not defined");
				exit(1);
			}

			BoundaryFacetIntegration(cornersLocal,facet_integ,projType);

    }

//this condition results in negative normal for all the lines in the line integral
    if(clockwise_ && bcellInt_==false)
    {
      facet_integ = -1.0*facet_integ;
      for(int i=0;i<4;i++)
        eqn_plane_[i] = -1.0*eqn_plane_[i];
    }

    return facet_integ;
}

/*-----------------------------------------------------------------------------------------------*
                            Performs integration over the boundarycell
*------------------------------------------------------------------------------------------------*/
void GEO::CUT::FacetIntegration::BoundaryFacetIntegration( const std::vector<std::vector<double> > cornersLocal,
                                                           double &facet_integ,
                                                           std::string intType )
{
	std::vector<double> alpha;
	double abs_normal=0.0;

	for(std::vector<std::vector<double> >::const_iterator k=cornersLocal.begin();k!=cornersLocal.end();k++)
	{
		const std::vector<double> coords1 = *k;
		std::vector<double> coords2;

		//for the last line the end point is the first point of the facet
		if(k!=(cornersLocal.end()-1))
      coords2 = *(k+1);
		else
      coords2= *(cornersLocal.begin());

		//first index decides the x or y coordinate, second index decides the start point or end point
		LINALG::Matrix<2,2> coordLine;
		if(intType=="x")
		{
			if(k==cornersLocal.begin())
			{
				alpha = compute_alpha(eqn_plane_,"x");
				abs_normal = getNormal(intType);
			}
	//The facet is projected over y-z plane and then the integration is performed
	//so only y- and z-coordinates are passed to make the lines
	//(0,i)-- indicates y and (1,i) indicate z because we are now in y-z plane
      coordLine(0,0) = coords1[1];
      coordLine(1,0) = coords1[2];
      coordLine(0,1) = coords2[1];
      coordLine(1,1) = coords2[2];

			LineIntegration line1(coordLine,inte_num_,alpha,bcellInt_);
			line1.set_integ_type("x");
			facet_integ += line1.integrate_line();
		}
		else if(intType=="y")
		{
			if(k==cornersLocal.begin())
			{
				alpha = compute_alpha(eqn_plane_,"y");
				abs_normal = getNormal(intType);
			}
	//The facet is projected over y-z plane and then the integration is performed
	//so only y- and z-coordinates are passed to make the lines
	//(0,i)-- indicates y and (0,i) indicate z because we are now in y-z plane
      coordLine(0,0) = coords1[2];
      coordLine(1,0) = coords1[0];
      coordLine(0,1) = coords2[2];
      coordLine(1,1) = coords2[0];
			LineIntegration line1(coordLine,inte_num_,alpha,bcellInt_);
			line1.set_integ_type("y");
			facet_integ += line1.integrate_line();
		}

		else if(intType=="z")
		{
			if(k==cornersLocal.begin())
			{
				alpha = compute_alpha(eqn_plane_,"z");
				abs_normal = getNormal(intType);
			}
	//The facet is projected over y-z plane and then the integration is performed
	//so only y- and z-coordinates are passed to make the lines
	//(0,i)-- indicates y and (0,i) indicate z because we are now in y-z plane
      coordLine(0,0) = coords1[0];
      coordLine(1,0) = coords1[1];
      coordLine(0,1) = coords2[0];
      coordLine(1,1) = coords2[1];
			LineIntegration line1(coordLine,inte_num_,alpha,bcellInt_);
			line1.set_integ_type("z");
			facet_integ += line1.integrate_line();
		}
		else
		{
			dserror("The facet integration type not supported");
			exit(1);
		}
	}

	facet_integ = facet_integ/abs_normal;
}

/*-------------------------------------------------------------------------------------------------*
      Generate integration rule for the facet if the divergence theorem is used      Sudhakar 03/12
      directly to generate Gauss integration rule for the facet
*--------------------------------------------------------------------------------------------------*/
void GEO::CUT::FacetIntegration::DivergenceIntegrationRule( Mesh &mesh,
                                                            Teuchos::RCP<DRT::UTILS::CollectedGaussPoints> & cgp )
{
  plain_boundarycell_set divCells;

  //the last two parameters has no influence when called from the first parameter is set to true
  GenerateIntegrationRuleDivergence( true, mesh, divCells );

  double normalX = getNormal("x");//make sure eqn of plane is available before calling this

  if(clockwise_) //because if ordering is clockwise the contribution of this facet must be subtracted
    normalX = -1.0*normalX;

  //std::cout<<"isclockwise = "<<clockwise_<<"\n";

  //std::cout<<"size of the divergenceCells = "<<divCells.size()<<"\n";
  //Teuchos::RCP<DRT::UTILS::CollectedGaussPoints> cgp = Teuchos::rcp( new DRT::UTILS::CollectedGaussPoints(0) );

  //DRT::UTILS::GaussIntegration gi_temp;
  for(plain_boundarycell_set::iterator i=divCells.begin();i!=divCells.end();i++)
  {
    BoundaryCell* bcell = *i;

#ifdef DEBUGCUTLIBRARY //write separate file for each bcell along with the distribution of Gauss points
    std::string filename="side";
    std::ofstream file;

    static int facetno = 0;
    facetno++;
    std::stringstream out;
    out <<"facetid"<<facetno<<".pos";
    filename = out.str();
    file.open(filename.c_str());

    const std::vector<Point*> ptl = bcell->Points();
    int mm=0;
    for( unsigned ii=0;ii<ptl.size();ii++ )
    {
      Point* pt1 = ptl[ii];
      LINALG::Matrix<3,1> global1,local1;
      pt1->Coordinates(global1.A());
      elem1_->LocalCoordinates( global1, local1 );
      file<<"Point("<<mm<<")="<<"{"<<local1(0,0)<<","<<local1(1,0)<<","<<local1(2,0)<<"};\n";
      //file<<"Point("<<mm<<")="<<"{"<<global1(0,0)<<","<<global1(1,0)<<","<<global1(2,0)<<"};\n";
      mm++;
    }
    for( unsigned ii=0;ii<ptl.size();ii++ )
    {
      file<<"Line("<<ii<<")="<<"{"<<ii<<","<<(ii+1)%ptl.size()<<"};\n";
    }
#endif

    DRT::UTILS::GaussIntegration gi_temp = DRT::UTILS::GaussIntegration( bcell->Shape(), 7 );

    for ( DRT::UTILS::GaussIntegration::iterator iquad=gi_temp.begin(); iquad!=gi_temp.end(); ++iquad )
    {
      double drs = 0.0;
      LINALG::Matrix<3,1> x_gp_loc(true), normal(true);
      const LINALG::Matrix<2,1> eta( iquad.Point() );

      switch ( bcell->Shape() )
      {
        case DRT::Element::tri3:
        {
          bcell->TransformLocalCoords<DRT::Element::tri3>(elem1_,eta, x_gp_loc, normal, drs);
          //bcell->Transform<DRT::Element::tri3>( eta, x_gp_loc, normal, drs );
          break;
        }
        case DRT::Element::quad4:
        {
          bcell->TransformLocalCoords<DRT::Element::quad4>(elem1_,eta, x_gp_loc, normal, drs);
          //bcell->Transform<DRT::Element::quad4>( eta, x_gp_loc, normal, drs );
          break;
        }
        default:
          throw std::runtime_error( "unsupported integration cell type" );
      }
      double wei = iquad.Weight()*drs*normalX;

      cgp->Append( x_gp_loc, wei );

#ifdef DEBUGCUTLIBRARY //write separate file for each bcell along with the distribution of Gauss points (contd...)
      file<<"Point("<<mm<<")="<<"{"<<x_gp_loc(0,0)<<","<<x_gp_loc(1,0)<<","<<x_gp_loc(2,0)<<"};\n";
      mm++;
#endif
    }
  }
}

/*-----------------------------------------------------------------------------------------------------*
          Generate temporary boundarycells for the considered facet                       Sudhakar 03/12
          if the number of corners of the facet is 3 or 4 this is straightforward
          Else triangulation is performed
*------------------------------------------------------------------------------------------------------*/
void GEO::CUT::FacetIntegration::GenerateIntegrationRuleDivergence( bool divergenceRule, //if called to generate direct divergence rule
                                                                    Mesh &mesh,
                                                                    plain_boundarycell_set & divCells )
{
  std::vector<std::vector<double> > cornersLocal = face1_->CornerPointsLocal(elem1_);

  eqn_plane_ = equation_plane(cornersLocal);

// the face is in the x-y or in y-z plane which will not be considered when divergence theorem is applied
  if(divergenceRule)
  {
    if(fabs(eqn_plane_[0])<0.0000001)
      return;
  }

  if( !divergenceRule && !face1_->OnCutSide() )
    return;

  IsClockwise(eqn_plane_,cornersLocal);

  std::vector<Point*> corners = face1_->CornerPoints();
  if(clockwise_)
    std::reverse(corners.begin(),corners.end());

  if(divergenceRule)
  {
    //-----------------------------------------------------------------
    // only facets with 3 corners are considered as the special case and a Tri3 is created directly.
    // we cannot directly create a Quad4 by checking the number of corners, because it can be a
    // concave facet with 4 corners for which Gaussian rules are not available
    //-----------------------------------------------------------------
    if(corners.size()==3)
    {
      TemporaryTri3(corners, divCells);
    }
    else //split the aribtrary noded facet into cells
    {

      /*std::cout<<"number of corners = "<<corners.size()<<"\n";
      for( std::vector<Point*>::iterator nn=corners.begin();nn!=corners.end();nn++ )
      {
        Point* pt1 = *nn;
        double xx[3];
        pt1->Coordinates(xx);
        std::cout<<xx[0]<<"\t"<<xx[1]<<"\t"<<xx[2]<<"\n";
      }*/

      std::string splitMethod;

#if 1 // split facet
      std::vector<std::vector<GEO::CUT::Point*> > split;
      TriangulateFacet tf( face1_, mesh, corners );
      tf.SplitFacet();
      split = tf.GetSplitCells();
      splitMethod = "split";
#endif

#if 0 // triangulate facet
      std::cout<<"!!! WARNING !!! Facets are triangulated instead of getting splitted ---> more Gauss points\n";
      std::vector<std::vector<GEO::CUT::Point*> > split;
      TriangulateFacet tf( face1_, mesh, corners );

      std::vector<int> ptconc;
      tf.EarClipping( ptconc, true );

      split = tf.GetSplitCells();
      splitMethod = "triangulation";
#endif

      for ( std::vector<std::vector<Point*> >::const_iterator j=split.begin();
                                                              j!=split.end(); ++j )
      {
        std::vector<Point*> tri = *j;
        if(clockwise_)
          std::reverse(tri.begin(),tri.end());
        if(tri.size()==3)
          TemporaryTri3(tri, divCells);
        else if(tri.size()==4)
          TemporaryQuad4(tri, divCells);
        else
        {
          std::cout<<"number of sides = "<<tri.size();
          dserror("Splitting created neither tri3 or quad4");
        }
      }
#ifdef DEBUGCUTLIBRARY // check the area of facet computed from splitting and triangulation
  DebugAreaCheck( divCells, splitMethod, mesh );
#endif
    }
  }

}

/*-------------------------------------------------------------------------------------------*
                            temporarily create a tri3 cell
                    this is temporary because this is not stored for the volumecell
*--------------------------------------------------------------------------------------------*/
void GEO::CUT::FacetIntegration::TemporaryTri3( std::vector<Point*>& corners,
                                                plain_boundarycell_set& divCells )
{
  Epetra_SerialDenseMatrix xyz( 3, 3 );
  for ( int i=0; i<3; ++i )
    corners[i]->Coordinates( &xyz( 0, i ) );
  Tri3BoundaryCell * bc = new Tri3BoundaryCell( xyz, face1_, corners );
  divCells.insert( bc );
}

/*-------------------------------------------------------------------------------------------*
                            temporarily create a quad4 cell
                    this is temporary because this is not stored for the volumecell
*--------------------------------------------------------------------------------------------*/
void GEO::CUT::FacetIntegration::TemporaryQuad4( std::vector<Point*>& corners,
                                                 plain_boundarycell_set& divCells )
{
  Epetra_SerialDenseMatrix xyz( 3, 4 );
  for ( int i=0; i<4; ++i )
    corners[i]->Coordinates( &xyz( 0, i ) );
  Quad4BoundaryCell * bc = new Quad4BoundaryCell( xyz, face1_, corners );
  divCells.insert( bc );
}

/*----------------------------------------------------------------------------------------------*
          Check whether the area of the facet computed by the splitting and
          triangular procedure are the same.                                       sudhakar 05/12
          This is a check for the adopted splitting procedure
*-----------------------------------------------------------------------------------------------*/
void GEO::CUT::FacetIntegration::DebugAreaCheck( const plain_boundarycell_set & divCells,
                                                 std::string alreadyDone,
                                                 Mesh &mesh )
{
  double area1 = 0.0;
  for( plain_boundarycell_set::const_iterator i=divCells.begin();i!=divCells.end();i++ )
  {
    BoundaryCell* bcell = *i;
    area1 += bcell->Area();
  }

  std::vector<Point*> corners = face1_->CornerPoints();

  std::vector<std::vector<GEO::CUT::Point*> > split1;
  if( alreadyDone=="split" )
  {
    TriangulateFacet tf( face1_, mesh, corners );
    std::vector<int> ptconc;

    tf.EarClipping( ptconc, true );
    split1 = tf.GetSplitCells();
  }
  else
  {
    TriangulateFacet tf( face1_, mesh, corners );
    tf.SplitFacet();
    split1 = tf.GetSplitCells();
  }

  plain_boundarycell_set tempCells;

  for ( std::vector<std::vector<Point*> >::const_iterator j=split1.begin();
                                                          j!=split1.end(); ++j )
  {
    std::vector<Point*> tri = *j;
    if(clockwise_)
      std::reverse(tri.begin(),tri.end());
    if(tri.size()==3)
      TemporaryTri3(tri, tempCells);
    else if(tri.size()==4)
      TemporaryQuad4(tri, tempCells);
    else
    {
      std::cout<<"number of sides = "<<tri.size();
      dserror("Splitting created neither tri3 or quad4");
    }
  }

  double area2 = 0.0;
  for( plain_boundarycell_set::const_iterator i=tempCells.begin();i!=tempCells.end();i++ )
  {
    BoundaryCell* bcell = *i;
    area2 += bcell->Area();
  }

  if( fabs(area1-area2)>1e-5 )
  {
    std::cout<<"The coordinates of the facet\n";
    for( unsigned i=0;i<corners.size();i++ )
    {
      Point* pt = corners[i];
      double coo[3];
      pt->Coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }

    std::cout<<"cells produced by Method 1\n";
    for( plain_boundarycell_set::const_iterator i=divCells.begin();i!=divCells.end();i++ )
    {
      BoundaryCell* bcell = *i;
      const std::vector<Point*> pts = bcell->Points();
      std::cout<<"cells\n";
      for( unsigned j=0;j<pts.size();j++ )
      {
        Point *pt1 = pts[j];
        double coo[3];
        pt1->Coordinates(coo);
        std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
      }
    }

    std::cout<<"cells produced by Method 2\n";
    for( plain_boundarycell_set::const_iterator i=tempCells.begin();i!=tempCells.end();i++ )
    {
      BoundaryCell* bcell = *i;
      const std::vector<Point*> pts = bcell->Points();
      std::cout<<"cells\n";
      for( unsigned j=0;j<pts.size();j++ )
      {
        Point *pt1 = pts[j];
        double coo[3];
        pt1->Coordinates(coo);
        std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
      }
    }

    std::cout<<"Area1 = "<<area1<<"\t"<<"Area2 = "<<area2<<"\n";
    std::cout<<"!!!WARNING!!! area predicted by splitting and triangulation are not the same\n";
    dserror( "error: area predicted by splitting and triangulation are not the same" );
  }
}


