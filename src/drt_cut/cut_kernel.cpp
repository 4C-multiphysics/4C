#include "cut_kernel.H"
#include "cut_point.H"
#include "cut_position2d.H"
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
bool GEO::CUT::KERNEL::IsOnLine( Point* & pt1, Point* & pt2, Point* & pt3 )
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

  // if the cross product is zero - on the same line
  // increasing this from 1e-10 to 1e-6 shown error in volume prediction
  if(cross.NormInf()<1e-10)
    return true;
  return false;
}

/*---------------------------------------------------------------------------------------------------------*
      Check whether the list of points given forms a convex polygon                         Sudhakar 04/12
      If any 3 points fall along the line, this will delete the middle point and return new pt
      Intially the polygon is projected into the given plane
*----------------------------------------------------------------------------------------------------------*/
std::vector<int> GEO::CUT::KERNEL::CheckConvexity( const std::vector<Point*>& ptlist, std::string& geomType )
{
  if( ptlist.size()<4 )
    dserror( "The number of points < 4. Is it called for appropriate facet?" );

  /**************************************************************/
  /*std::cout<<"points inside the kernel\n";
  for( unsigned i=0;i<ptlist.size();i++ )
  {
    Point* ptt = ptlist[i];
    double z[3];
    ptt->Coordinates(z);
    std::cout<<z[0]<<"\t"<<z[1]<<"\t"<<z[2]<<"\n";
  }*/
  /**************************************************************/

  std::string projPlane;

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
      dserror( "Inline checking for facets not done before calling this" );
  }

  bool isClockwise = IsClockwiseOrderedPolygon( ptlist, projPlane );

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

    /*x1.Print(std::cout);
    x2.Print(std::cout);
    x3.Print(std::cout);
    xtemp.Print(std::cout);*/

    double res = x3(ind1,0)*xtemp(ind2,0)-x3(ind2,0)*xtemp(ind1,0)+
                 xtemp(ind1,0)*x1(ind2,0)-xtemp(ind2,0)*x1(ind1,0);

    /*if( fabs(res)<1e-8 ) //this means small angled lines are just eliminated
      continue;*/

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
            Find the equation of plane that contains these non-collinear points
                                                                                          Sudhakar 04/12
*------------------------------------------------------------------------------------------------------*/
std::vector<double> GEO::CUT::KERNEL::EqnPlane( Point* & pt1, Point* & pt2, Point* & pt3 )
{
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

  double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
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
           Returns true if the points of the polygon are ordered clockwise                        sudhakar 05/12
   Polygon in 3D space is first projected into 2D plane, and the plane of projection is returned in projType
*-------------------------------------------------------------------------------------------------------------*/
bool GEO::CUT::KERNEL::IsClockwiseOrderedPolygon( std::vector<Point*>polyPoints, std::string& projPlane )
{
  if( polyPoints.size()<3 )
    dserror( "polygon with less than 3 corner points" );

  std::vector<double> eqn;
  eqn = EqnPlane( polyPoints[0], polyPoints[1], polyPoints[2] );

  // projection on the plane which has max normal component - reduce round off error
  if( fabs(eqn[0])>fabs(eqn[1]) && fabs(eqn[0])>fabs(eqn[2]) )
    projPlane = "x";
  else if( fabs(eqn[1])>fabs(eqn[2]) && fabs(eqn[1])>fabs(eqn[0]) )
    projPlane = "y";
  else
    projPlane = "z";

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
