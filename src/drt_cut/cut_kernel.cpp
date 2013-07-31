#include "cut_kernel.H"
#include "cut_point.H"
#include "cut_position2d.H"
#include "../drt_io/io_pstream.H"
#include <iostream>

unsigned GEO::CUT::KERNEL::FindNextCornerPoint( const std::vector<Point*> & points,
                                                LINALG::Matrix<3,1> & x1,
                                                LINALG::Matrix<3,1> & x2,
                                                LINALG::Matrix<3,1> & x3,
                                                LINALG::Matrix<3,1> & b1,
                                                LINALG::Matrix<3,1> & b2,
                                                LINALG::Matrix<3,1> & b3,
                                                unsigned i )
{
  unsigned pointsize = points.size();
  unsigned j = ( i+1 ) % pointsize;
  if ( pointsize < 3 )
  {
    return j;
  }

  points[i]->Coordinates( x1.A() );
  points[j]->Coordinates( x2.A() );

  b1.Update( 1, x2, -1, x1, 0 );

  double norm = b1.Norm2();
  if ( norm < std::numeric_limits<double>::min() )
    throw std::runtime_error( "same point in facet not supported" );

  b1.Scale( 1./norm );

  if ( b1.Norm2() < std::numeric_limits<double>::min() )
    throw std::runtime_error( "same point in facet not supported" );

  i = j;
  for ( unsigned k=2; k<pointsize; ++k )
  {
    i = ( i+1 ) % pointsize;
    Point * p = points[i];
    p->Coordinates( x3.A() );

    b2.Update( 1, x3, -1, x1, 0 );

    norm = b2.Norm2();
    if ( norm < std::numeric_limits<double>::min() )
      throw std::runtime_error( "same point in facet not supported" );

    b2.Scale( 1./norm );

    // cross product to get the normal at the point
    b3( 0 ) = b1( 1 )*b2( 2 ) - b1( 2 )*b2( 1 );
    b3( 1 ) = b1( 2 )*b2( 0 ) - b1( 0 )*b2( 2 );
    b3( 2 ) = b1( 0 )*b2( 1 ) - b1( 1 )*b2( 0 );

    if ( b3.Norm2() > PLANARTOL )
    {
      // Found. Return last node on this line.
      return ( i+pointsize-1 ) % pointsize;
    }
  }

  // All on one line. Return first and last point.
  if ( j==0 )
  {
    return 0;
  }
  else
  {
    return pointsize-1;
  }
}

void GEO::CUT::KERNEL::FindCornerPoints( const std::vector<Point*> & points,
                                         std::vector<Point*> & corner_points )
{
  LINALG::Matrix<3,1> x1;
  LINALG::Matrix<3,1> x2;
  LINALG::Matrix<3,1> x3;
  LINALG::Matrix<3,1> b1;
  LINALG::Matrix<3,1> b2;
  LINALG::Matrix<3,1> b3;

  for ( unsigned i = FindNextCornerPoint( points, x1, x2, x3, b1, b2, b3, 0 );
        true;
        i = FindNextCornerPoint( points, x1, x2, x3, b1, b2, b3, i ) )
  {
    Point * p = points[i];
    if ( corner_points.size()>0 and corner_points.front()==p )
      break;
    corner_points.push_back( p );
  }
}

bool GEO::CUT::KERNEL::IsValidQuad4( const std::vector<Point*> & points )
{
  if ( points.size()==4 )
  {
    LINALG::Matrix<3,3> xyze;
    LINALG::Matrix<3,1> xyz;
    for ( int i=0; i<4; ++i )
    {
      points[( i+0 )%4]->Coordinates( &xyze( 0, 0 ) );
      points[( i+1 )%4]->Coordinates( &xyze( 0, 1 ) );
      points[( i+2 )%4]->Coordinates( &xyze( 0, 2 ) );
      points[( i+3 )%4]->Coordinates( &xyz( 0, 0 ) );

      Position2d<DRT::Element::tri3> pos( xyze, xyz );
      if ( pos.Compute() )
      {
        return false;
      }
    }
    return true;
  }
  return false;
}

DRT::Element::DiscretizationType GEO::CUT::KERNEL::CalculateShape( const std::vector<Point*> & points,
                                                                   std::vector<Point*> & line_points )
{
  FindCornerPoints( points, line_points );

  if ( IsValidTri3( line_points ) )
  {
    return DRT::Element::tri3;
  }
  else if ( IsValidQuad4( line_points ) )
  {
    return DRT::Element::quad4;
  }

  return DRT::Element::dis_none;
}

/*-----------------------------------------------------------------------------------------------------*
  Check whether three points are lying on the same line by checking whether the cross product is zero
                                                                                          Sudhakar 04/12
*------------------------------------------------------------------------------------------------------*/
bool GEO::CUT::KERNEL::IsOnLine( Point* & pt1, Point* & pt2, Point* & pt3, bool DeleteInlinePts )
{
  LINALG::Matrix<3,1> x1,x2,x3;
  LINALG::Matrix<3,1> pt1pt2,pt1pt3,cross;
  pt1->Coordinates(x1.A());
  pt2->Coordinates(x2.A());
  pt3->Coordinates(x3.A());

  pt1pt2.Update(1,x2,-1,x1,0);
  pt1pt3.Update(1,x3,-1,x1,0);

  cross(0,0) = pt1pt2(1,0)*pt1pt3(2,0)-pt1pt2(2,0)*pt1pt3(1,0);
  cross(1,0) = pt1pt2(0,0)*pt1pt3(2,0)-pt1pt2(2,0)*pt1pt3(0,0);
  cross(2,0) = pt1pt2(1,0)*pt1pt3(0,0)-pt1pt2(0,0)*pt1pt3(1,0);

  if ( DeleteInlinePts )
  {
    // if the cross product is zero - on the same line
    // increasing this from 1e-10 to 1e-6 shown error in volume prediction
    if(cross.NormInf()<TOL_POINTS_ON_LINE_FOR_DELETING)
      return true;
  }
  else
  {
    if(cross.NormInf()<TOL_POINTS_ON_LINE)
      return true;
  }
  return false;
}

/*---------------------------------------------------------------------------------------------------------*
      Check whether the list of points given forms a convex polygon                         Sudhakar 04/12
      If any 3 points fall along the line, this will delete the middle point and return new pt
      Intially the polygon is projected into the given plane
*----------------------------------------------------------------------------------------------------------*/
std::vector<int> GEO::CUT::KERNEL::CheckConvexity( const std::vector<Point*>& ptlist,
                                                   std::string& geomType,
                                                   bool InSplit,
                                                   bool DeleteInlinePts )
{
  if( InSplit ) // if this function is called while performing facet splitting
  {
    if( ptlist.size()<4 )
    {
      std::cout << "ptlist.size(): " << ptlist.size() << "\n";
      dserror( "The number of points < 4. Is it called for appropriate facet?" );
    }
  }

  std::string projPlane;

  if ( DeleteInlinePts )
  {
    for( unsigned i=0;i<ptlist.size();i++ ) // make sure there are no inline points
    {
      Point* pt2 = ptlist[i];
      Point* pt3 = ptlist[(i+1)%ptlist.size()];
      unsigned ind = i-1;
      if( i==0 )
        ind = ptlist.size()-1;
      Point* pt1 = ptlist[ind];

      bool isline = IsOnLine( pt1, pt2, pt3 );
      if( isline )
      {
        IO::cout<<"the points are\n";
        for( unsigned i=0;i<ptlist.size();i++ )
        {
          Point* ptx = ptlist[i];
          double coox[3];
          ptx->Coordinates(coox);
          IO::cout<<coox[0]<<"\t"<<coox[1]<<"\t"<<coox[2]<<"\n";
        }
        dserror( "Inline checking for facets not done before calling this" );
      }
    }
  }

  bool isClockwise = IsClockwiseOrderedPolygon( ptlist, projPlane, DeleteInlinePts );

  int ind1=0,ind2=0;
  if( projPlane=="x" )
  {
    ind1 = 1;
    ind2 = 2;
  }
  else if( projPlane=="y" )
  {
    ind1 = 2;
    ind2 = 0;
  }
  else if( projPlane=="z" )
  {
    ind1 = 0;
    ind2 = 1;
  }
  else
    dserror( "unspecified projection type" );

  LINALG::Matrix<3,1> x1,x2,x3,xtemp;
  std::vector<int> leftind,rightind;
  std::vector<int> concPts;
  for( unsigned i=0;i<ptlist.size();i++ )
  {
    Point* pt2 = ptlist[i];
    Point* pt3 = ptlist[(i+1)%ptlist.size()];
    unsigned ind = i-1;
    if(i==0)
      ind = ptlist.size()-1;
    Point* pt1 = ptlist[ind];

    pt1->Coordinates(x1.A());
    pt2->Coordinates(x2.A());
    pt3->Coordinates(x3.A());

    xtemp.Update(1.0,x2,-1.0,x1);

    double res = x3(ind1,0)*xtemp(ind2,0)-x3(ind2,0)*xtemp(ind1,0)+
                 xtemp(ind1,0)*x1(ind2,0)-xtemp(ind2,0)*x1(ind1,0);

    if( fabs(res)<TOL_POINTS_ON_LINE ) // this means small angled lines are just eliminated
    {
      if ( not DeleteInlinePts ) // this means small angled lines are just concave
      {
        if( isClockwise )
        {
      	  leftind.push_back(i);
        }
        else
        {
          rightind.push_back(i);
        }
      }
      continue;
    }

    if(res<0.0)
    {
      leftind.push_back(i);
    }
    else
    {
      rightind.push_back(i);
    }
  }

  if( leftind.size()==0 || rightind.size()==0 )
    geomType = "convex";
  else if( leftind.size()==1 || rightind.size()==1 )
    geomType = "1ptConcave";
  else
    geomType = "concave";

  // if the points are ordered acw, right-turning points are concave points, and vice versa
  if( isClockwise )
    return leftind;
  return rightind;
}

/*-----------------------------------------------------------------------------------------------------*
            Find the equation of plane of the polygon defined by these facets
            KERNEL::DeleteInlinePts() must be called before using this function
            This works only for simple polygons (not doubly connected, not self-intersecting)
                                                                                          Sudhakar 01/13
*------------------------------------------------------------------------------------------------------*/
std::vector<double> GEO::CUT::KERNEL::EqnPlanePolygon( const std::vector<Point*>& ptlist, bool DeleteInlinePts )
{
  std::vector<double> eqn_plane(4);
  if( ptlist.size() == 3 )
  {
    Point*p1 = ptlist[0];
    Point*p2 = ptlist[1];
    Point*p3 = ptlist[2];

    //eqn_plane = EqnPlane( ptlist[0], ptlist[1], ptlist[2] );
    eqn_plane = EqnPlane( p1, p2, p3 );
    return eqn_plane;
  }

  std::vector<int> concavePts;
  std::string geoType;
  concavePts = KERNEL::CheckConvexity(  ptlist, geoType, false, DeleteInlinePts ); // find concave points of the polygon

  // for finding equation of convex facet, any 3 points can be used
  if( concavePts.size() == 0 )
  {
    if ( DeleteInlinePts )
    {
      Point*p1 = ptlist[0];
      Point*p2 = ptlist[1];
      Point*p3 = ptlist[2];
      eqn_plane = EqnPlane( p1, p2, p3 );
    }
    else
    {
      std::vector<Point*> pttemp = ptlist;
      std::vector<Point*> preparedPoints = PreparePoints( pttemp );
      eqn_plane = EqnPlane( preparedPoints[0], preparedPoints[1], preparedPoints[2] );
    }
    return eqn_plane;
  }

  // to find equation of plane for a concave facet we choose 3 adjacent points
  // if secondpt is a concave point, normal direction is not computed correctly
  unsigned ncross=0;
  unsigned npts = ptlist.size();
  bool eqndone=false;
  int firstPt=0,secondPt=0,thirdPt=0;

  for( unsigned i=0;i<npts;++i )
  {
    ncross++;

    int concNo = 0;

    if( i==0 )
    {
      firstPt = concavePts[concNo];
      secondPt = (firstPt+1)%npts;
    }
    else
    {
      firstPt = (firstPt+1)%npts;
      secondPt = (firstPt+1)%npts;
    }
    // check whether secondpt is a concave point
    if(std::find(concavePts.begin(), concavePts.end(), secondPt) != concavePts.end())
      continue;

    thirdPt = (secondPt+1)%npts;

    Point*p1 = ptlist[firstPt];
    Point*p2 = ptlist[secondPt];
    Point*p3 = ptlist[thirdPt];

    if ( not DeleteInlinePts )
    {
      if( IsOnLine( p1,p2,p3 ) )
      {
        continue;
      }
    }

    eqn_plane = EqnPlane( p1, p2, p3 );

    eqndone = true;
  }

  if( eqndone == false )
    dserror("equation not computed");

  return eqn_plane;
}

/*-----------------------------------------------------------------------------------------------------*
            Find the equation of plane that contains these non-collinear points
            It must be noted while using this function to find equation of facets,
             none of these 3 points must be a reflex (concave) point
                                                                                          Sudhakar 04/12
*------------------------------------------------------------------------------------------------------*/
std::vector<double> GEO::CUT::KERNEL::EqnPlane( Point* & pt1, Point* & pt2, Point* & pt3 )
{

  bool collinear = IsOnLine( pt1,pt2,pt3 );
  if( collinear )
    dserror(" 3 points lie on a line. Eqn of plane cannot be computed");

  std::vector<double> eqn_plane(4);
  double x1[3],x2[3],x3[3];

  pt1->Coordinates(x1);
  pt2->Coordinates(x2);
  pt3->Coordinates(x3);

  eqn_plane[0] = x1[1]*(x2[2]-x3[2])+x2[1]*(x3[2]-x1[2])+x3[1]*(x1[2]-x2[2]);
  eqn_plane[1] = x1[2]*(x2[0]-x3[0])+x2[2]*(x3[0]-x1[0])+x3[2]*(x1[0]-x2[0]);
  eqn_plane[2] = x1[0]*(x2[1]-x3[1])+x2[0]*(x3[1]-x1[1])+x3[0]*(x1[1]-x2[1]);
  eqn_plane[3] = x1[0]*(x2[1]*x3[2]-x3[1]*x2[2])+x2[0]*(x3[1]*x1[2]-x1[1]*x3[2])+x3[0]*(x1[1]*x2[2]-x2[1]*x1[2]);

  return eqn_plane;
}

/*------------------------------------------------------------------------------------------------------------*
           check whether the point "check" is inside the triangle formed by tri               sudhakar 05/12
                 uses barycentric coordinates as it is faster
*-------------------------------------------------------------------------------------------------------------*/
bool GEO::CUT::KERNEL::PtInsideTriangle( std::vector<Point*> tri, Point* check )
{
  if( tri.size()!=3 )
    dserror("expecting a triangle");

  LINALG::Matrix<3,1> t1,t2,t3,pt, v0(0.0), v1(0.0), v2(0.0);
  tri[0]->Coordinates( t1.A() );
  tri[1]->Coordinates( t2.A() );
  tri[2]->Coordinates( t3.A() );
  check->Coordinates( pt.A() );

  v0.Update(1.0, t3, -1.0, t1);
  v1.Update(1.0, t2, -1.0, t1);
  v2.Update(1.0, pt, -1.0, t1);

  double dot00,dot01,dot02,dot11,dot12;
  dot00 = v0.Dot(v0);
  dot01 = v0.Dot(v1);
  dot02 = v0.Dot(v2);
  dot11 = v1.Dot(v1);
  dot12 = v1.Dot(v2);

  double Det = dot00 * dot11 - dot01 * dot01;

  if( fabs(Det) < 1e-35 )
  {
    std::cout<<"value of det = "<<Det<<"\n";
    std::cout << "triangle: " << "t1 " << t1
              << "t2 " << t2
              << "t3 " << t3 << std::endl;
    std::cout << "point " << pt << std::endl;
    dserror("the triangle is actually on a line. Verify tolerances in cut_tolerance.H\n");
  }

  double invDenom = 1.0 / Det;
  double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  double v = (dot00 * dot12 - dot01 * dot02) * invDenom;

  // if u or v are very small, then set them to zero
  if( fabs(u)<1e-35 )
    u = 0.0;
  if( fabs(v)<1e-35 )
    v = 0.0;

  if( (u >= 0) && (v >= 0) && (u + v < 1) )
    return true;
  return false;
}

/*------------------------------------------------------------------------------------------------------------*
           Check whether the point "check" is inside the Quad formed by quad               sudhakar 07/12
           Splits Quad into 2 Tri and perform the check on each
*-------------------------------------------------------------------------------------------------------------*/
bool GEO::CUT::KERNEL::PtInsideQuad( std::vector<Point*> quad, Point* check )
{
  if( quad.size()!=4 )
    dserror( "expecting a Quad" );

  std::vector<int> concavePts;
  std::string str1;

  concavePts = CheckConvexity(  quad, str1 );

  int concsize = concavePts.size();
  if( concsize > 1 )
  {
    IO::cout<<"The points of the failing Quad are\n";
    for( unsigned i=0;i<4;i++ )
    {
      Point* pt = quad[i];
      double x[3];
      pt->Coordinates(x);
      IO::cout<<x[0]<<"\t"<<x[1]<<"\t"<<x[2]<<"\n";
    }
    dserror( "Quad has more than 1 concave pt --> Selfcut" );
  }

  int indStart = 0;
  if( concsize==1 )
    indStart = concavePts[0];


  std::vector<Point*> tri1(3),tri2(3);

  tri1[0] = quad[indStart];
  tri1[1] = quad[(indStart+1)%4];
  tri1[2] = quad[(indStart+2)%4];

  bool insideTri1 = PtInsideTriangle( tri1, check );
  if( insideTri1 )
    return true;

  tri2[0] = quad[indStart];
  tri2[1] = quad[(indStart+2)%4];
  tri2[2] = quad[(indStart+3)%4];

  bool insideTri2 = PtInsideTriangle( tri2, check );
  if( insideTri2 )
    return true;

  return false;
}

/*------------------------------------------------------------------------------------------------------------*
           Returns true if the points of the polygon are ordered clockwise                        sudhakar 05/12
   Polygon in 3D space is first projected into 2D plane, and the plane of projection is returned in projType
*-------------------------------------------------------------------------------------------------------------*/
bool GEO::CUT::KERNEL::IsClockwiseOrderedPolygon( std::vector<Point*>polyPoints, std::string& projPlane, bool DeleteInlinePts )
{
  if( polyPoints.size()<3 )
    dserror( "polygon with less than 3 corner points" );

  std::vector<double> eqn;

  if ( DeleteInlinePts )
  {
    eqn = EqnPlane( polyPoints[0], polyPoints[1], polyPoints[2] );
  }
  else
  {
    std::vector<Point*> preparedPoints = PreparePoints( polyPoints );
    eqn = EqnPlane( preparedPoints[0], preparedPoints[1], preparedPoints[2] );
  }

  // projection on the plane which has max normal component - reduce round off error
  if( fabs(eqn[0])>fabs(eqn[1]) && fabs(eqn[0])>fabs(eqn[2]) )
    projPlane = "x";
  else if( fabs(eqn[1])>fabs(eqn[2]) && fabs(eqn[1])>fabs(eqn[0]) )
    projPlane = "y";
  else
  {
    if( fabs(eqn[0])==fabs(eqn[1]) )
    {
      if( fabs(eqn[0]) > fabs(eqn[2]) )
        projPlane = "x";
      else
        projPlane = "z";
    }
    else if( fabs(eqn[1])==fabs(eqn[2]) )
    {
      if( fabs(eqn[1]) > fabs(eqn[0]) )
        projPlane = "y";
      else
        projPlane = "x";
    }
    else if( fabs(eqn[0])==fabs(eqn[2]) )
    {
      if( fabs(eqn[0]) > fabs(eqn[1]) )
        projPlane = "z";
      else
        projPlane = "y";
    }
    else
    {
      projPlane = "z";
    }
  }

  int ind1=0,ind2=0;
  if( projPlane=="x" )
  {
    ind1 = 1;
    ind2 = 2;
  }
  else if( projPlane=="y" )
  {
    ind1 = 2;
    ind2 = 0;
  }
  else if( projPlane=="z" )
  {
    ind1 = 0;
    ind2 = 1;
  }

  int numpts = polyPoints.size();
  double crossProd = 0.0;
  for( int i=0;i<numpts;i++ )
  {
    double pt1[3],pt2[3];

    polyPoints[i]->Coordinates(pt1);
    polyPoints[(i+1)%numpts]->Coordinates(pt2);

    crossProd += (pt2[ind1]-pt1[ind1])*(pt2[ind2]+pt1[ind2]);
  }

  if( crossProd>0.0 )
    return true;
  return false;
}

/*--------------------------------------------------------------------------------------*
    If more than two points are on a line, all points except the end points
    are deleted. This is checked for all the lines for a facet. So once this is
    called the facet is free of more than 2 inline points                   Sudhakar 06/12
*---------------------------------------------------------------------------------------*/
void GEO::CUT::KERNEL::DeleteInlinePts( std::vector<Point*>& poly )
{
  bool anyInLine = false;
  unsigned num = poly.size();

  for( unsigned i=0;i<num;i++ )
  {
    Point* pt1 = poly[i];
    Point* pt2 = poly[(i+1)%num];  // next point
    unsigned ind = i-1;
    if(i==0)
      ind = num-1;
    Point* pt3 = poly[ind];       // previous point

    anyInLine = IsOnLine( pt3, pt1, pt2, true );

    if( anyInLine )
    {
      std::vector<Point*>::iterator delPt = poly.begin()+i; //iterator of the point to be deleted
      poly.erase(delPt);
      break;
    }
  }
  if( anyInLine )   // this makes sure the procedure is repeated until all the inline points of the facet are deleted
    DeleteInlinePts( poly );
}

/*--------------------------------------------------------------------------------------*
    Returns true if at least 3 points are collinear                          Wirtz 05/13
*---------------------------------------------------------------------------------------*/
bool GEO::CUT::KERNEL::HaveInlinePts( std::vector<Point*>& poly )
{

  unsigned num = poly.size();
  for( unsigned i=0;i<num;i++ )
  {
    Point* pt1 = poly[i];
    Point* pt2 = poly[(i+1)%num];  // next point
    unsigned ind = i-1;
    if(i==0)
      ind = num-1;
    Point* pt3 = poly[ind];       // previous point
    if ( IsOnLine( pt3, pt1, pt2 ) )
    {
      return true;
    }
  }
  return false;

}

/*--------------------------------------------------------------------------------------*
    Finds tree points of the polygon which are not collinear                 Wirtz 05/13
*---------------------------------------------------------------------------------------*/
std::vector<GEO::CUT::Point*> GEO::CUT::KERNEL::PreparePoints( std::vector<Point*> & polyPoints )
{

  std::vector<Point*> preparedPoints;
  for( unsigned i=0;i<polyPoints.size();i++ )
  {
    Point* pt2 = polyPoints[i];
    Point* pt3 = polyPoints[(i+1)%polyPoints.size()];
    unsigned ind = i-1;
    if(i==0)
      ind = polyPoints.size()-1;
    Point* pt1 = polyPoints[ind];
    bool collinear = IsOnLine( pt1,pt2,pt3 );
    if( collinear )
    {
      continue;
    }
    preparedPoints.push_back( pt1 );
    preparedPoints.push_back( pt2 );
    preparedPoints.push_back( pt3 );
    return preparedPoints;
  }
  dserror( "case with inline points: all points collinear" );
  return preparedPoints;

}
