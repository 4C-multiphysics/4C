
//#include "../drt_geometry/intersection_templates.H"

#include "cut_position.H"
#include "cut_intersection.H"
#include "cut_facet.H"
#include "cut_point_impl.H"

#include <string>
#include <stack>

#include "cut_edge.H"
#include "cut_levelsetside.H"


bool GEO::CUT::Edge::FindCutPoints( Mesh & mesh,
                                    Element * element,
                                    Side & side,
                                    Side & other )
{
#if 1
  bool cut = false;
  for ( PointPositionSet::iterator i=cut_points_.begin(); i!=cut_points_.end(); ++i )
  {
    Point * p = *i;
    if ( p->IsCut( &other ) )
    {
      cut = true;
      p->AddElement( element );
    }
  }
  if ( cut )
  {
    return true;
  }
#endif

  // test for the cut of edge and side

  PointSet cut_points;
  other.Cut( mesh, *this, cut_points );
  for ( PointSet::iterator i=cut_points.begin();
        i!=cut_points.end();
        ++i )
  {
    Point * p = *i;

    p->AddEdge( this );
    p->AddElement( element );

    // These adds are implicitly done, but for documentation do them all explicitly.

    p->AddSide( &side );
    p->AddSide( &other );
    AddPoint( p );
  }

  return cut_points.size()>0;
}

void GEO::CUT::Edge::GetCutPoints( Element * element,
                                   Side & side,
                                   Side & other,
                                   PointSet & cuts )
{
  for ( PointPositionSet::iterator i=cut_points_.begin(); i!=cut_points_.end(); ++i )
  {
    Point * p = *i;
    if ( p->IsCut( &other ) and p->IsCut( element ) )
    {
      cuts.insert( p );
    }
  }
}

void GEO::CUT::Edge::GetCutPoints( Edge * other, PointSet & cuts )
{
  for ( PointPositionSet::iterator i=cut_points_.begin(); i!=cut_points_.end(); ++i )
  {
    Point * p = *i;
    if ( p->IsCut( other ) )
    {
      cuts.insert( p );
    }
  }
}

void GEO::CUT::Edge::AddPoint( Point* cut_point )
{
  // make sure the position of the point on this edge is known
  cut_point->t( this );

#if 0
  std::vector<Point*>::iterator j = std::lower_bound( cut_points_.begin(), cut_points_.end(), cut_point, PointPositionLess( this ) );
  if ( j==cut_points_.end() or *j!=cut_point )
  {
    cut_points_.push_back( cut_point );
    std::sort( cut_points_.begin(), cut_points_.end(), PointPositionLess( this ) );
  }
#else
  cut_points_.insert( cut_point );
#endif

#if 0
#ifdef DEBUGCUTLIBRARY
  PointSet cp;
  std::copy( cut_points_.begin(), cut_points_.end(), std::inserter( cp, cp.begin() ) );
  if ( cut_points_.size() != cp.size() )
    throw std::runtime_error( "broken cut points" );
#endif
#endif
}

void GEO::CUT::Edge::CutPoint( Node* edge_start, Node* edge_end, std::vector<Point*> & edge_points )
{
  Point * bp = BeginNode()->point();
  Point * ep = EndNode()  ->point();
  if ( bp==edge_start->point() and ep==edge_end->point() )
  {
    edge_points.clear();
    edge_points.assign( cut_points_.begin(), cut_points_.end() );
  }
  else if ( ep==edge_start->point() and bp==edge_end->point() )
  {
    edge_points.clear();
    edge_points.assign( cut_points_.rbegin(), cut_points_.rend() );
  }
  else
  {
    throw std::runtime_error( "not a valid edge" );
  }
}

void GEO::CUT::Edge::CutPoints( Side * side, PointSet & cut_points )
{
  SideCutFilter filter( side );
  for ( std::vector<Point*>::iterator i=cut_points_.begin(); i!=cut_points_.end(); ++i )
  {
    Point * p = *i;
    plain_line_set cut_lines;
    p->CutLines( filter, cut_lines );
    if ( cut_lines.size()>0 )
    {
      cut_points.insert( p );
    }
  }
}

void GEO::CUT::Edge::CutPointsBetween( Point* begin, Point* end, std::vector<Point*> & line )
{
//   PointPositionLess::iterator bi = cut_points_.find( begin );
//   PointPositionLess::iterator ei = cut_points_.find( end );
  std::vector<Point*>::iterator bi = std::lower_bound( cut_points_.begin(), cut_points_.end(), begin, PointPositionLess( this ) );
  std::vector<Point*>::iterator ei = std::lower_bound( cut_points_.begin(), cut_points_.end(), end  , PointPositionLess( this ) );

  if ( *bi != begin )
    bi = cut_points_.end();

  if ( *ei != end )
    ei = cut_points_.end();

  double bt = begin->t( this );
  double et = end->t( this );

  if ( bt < et )
  {
    if ( bi!=cut_points_.end() )
    {
      ++bi;
    }
    //std::copy( bi, ei, std::back_inserter( line ) );
    line.insert( line.end(), bi, ei );
  }
  else if ( bt > et )
  {
    if ( ei!=cut_points_.end() )
    {
      ++ei;
    }
    //std::copy( ei, bi, std::back_inserter( line ) );
    line.insert( line.end(), ei, bi );
  }
  else
  {
    if ( begin!=end )
    {
      throw std::runtime_error( "different points at the same place" );
    }
  }
}

void GEO::CUT::Edge::CutPointsIncluding( Point* begin, Point* end, std::vector<Point*> & line )
{
  std::vector<Point*>::iterator bi = std::lower_bound( cut_points_.begin(), cut_points_.end(), begin, PointPositionLess( this ) );
  std::vector<Point*>::iterator ei = std::lower_bound( cut_points_.begin(), cut_points_.end(), end  , PointPositionLess( this ) );

  if ( *bi != begin )
    throw std::runtime_error( "begin point not on edge" );

  if ( *ei != end )
    throw std::runtime_error( "end point not on edge" );

  double bt = begin->t( this );
  double et = end->t( this );

  if ( bt < et )
  {
    ++ei;
    //std::copy( bi, ei, std::back_inserter( line ) );
    line.insert( line.end(), bi, ei );
  }
  else if ( bt > et )
  {
    ++bi;
    //std::copy( ei, bi, std::inserter( line, line.begin() ) );
    line.insert( line.begin(), ei, bi );
  }
  else
  {
    if ( begin!=end )
    {
      throw std::runtime_error( "different points at the same place" );
    }
  }
}

void GEO::CUT::Edge::CutPointsInside( Element * element, std::vector<Point*> & line )
{
  Point * first = NULL;
  Point * last = NULL;
  for ( std::vector<Point*>::iterator i=cut_points_.begin(); i!=cut_points_.end(); ++i )
  {
    Point * p = *i;
    if ( p->IsCut( element ) )
    {
      if ( first == NULL )
      {
        first = last = p;
      }
      else
      {
        last = p;
      }
    }
  }
  if ( first!=NULL and first!=last )
  {
    CutPointsIncluding( first, last, line );
#if 0
    // Rectify numerical problems that occationally occur. There might be
    // middle points that do not know their position on an element side.
    if ( line.size() > 2 )
    {
      plain_side_set common;
      first->CommonSide( last, common );
      if ( common.size() > 0 )
      {
        for ( std::vector<Point*>::iterator i=line.begin(); i!=line.end(); ++i )
        {
          Point * p = *i;
          for ( plain_side_set::iterator i=common.begin(); i!=common.end(); ++i )
          {
            Side * s = *i;
            p->AddSide( s );
          }
        }
      }
    }
#endif
  }
}

bool GEO::CUT::Edge::IsCut( Side * side )
{
  for ( std::vector<Point*>::iterator i=cut_points_.begin(); i!=cut_points_.end(); ++i )
  {
    Point * p = *i;
    if ( p->IsCut( side ) )
    {
      return true;
    }
  }
  return false;
}

bool GEO::CUT::Edge::ComputeCut( Edge * other, double & pos, LINALG::Matrix<3,1> & x )
{
  LINALG::Matrix<3,1> p1;
  LINALG::Matrix<3,1> p2;
  LINALG::Matrix<3,1> u1;
  LINALG::Matrix<3,1> u2;

  const double * x11 = other->BeginNode()->point()->X();
  const double * x12 = other->EndNode()  ->point()->X();
  const double * x21 =  this->BeginNode()->point()->X();
  const double * x22 =  this->EndNode()  ->point()->X();

  for ( int i=0; i<3; ++i )
  {
    p1( i ) = .5*( x11[i] + x12[i] );
    p2( i ) = .5*( x21[i] + x22[i] );
    u1( i ) = .5*( x12[i] - x11[i] );
    u2( i ) = .5*( x22[i] - x21[i] );
  }

  LINALG::Matrix<3,1> w;
  w.Update( 1, p1, -1, p2, 0 );

  double a = u1.Dot( u1 );
  double b = u1.Dot( u2 );
  double c = u2.Dot( u2 );
  double d = u1.Dot( w );
  double e = u2.Dot( w );

  double denom = a*c - b*b;

  if ( denom >= std::numeric_limits<double>::min() )
  {
    denom = 1./denom;
    double s = denom * ( b*e-c*d );
    double t = denom * ( a*e-b*d );

    pos = t;
    x.Update( 1, p1, s, u1, 0 );

    w.Update( 1, p2, t, u2, 0 );
    w.Update( 1, x, -1 );
    double distance = w.Norm2();

    return ( distance < TOLERANCE and
             s >= -1-TOLERANCE and
             s <=  1+TOLERANCE and
             t >= -1-TOLERANCE and
             t <=  1+TOLERANCE );
  }

  return false;
}

GEO::CUT::Point* GEO::CUT::Edge::NodeInElement( Element * element, Point * other )
{
  Point * p = BeginNode()->point();
  if ( p!=other and p->IsCut( element ) )
  {
    return p;
  }
  p = EndNode()->point();
  if ( p!=other and p->IsCut( element ) )
  {
    return p;
  }
  return NULL;
}


void GEO::CUT::Edge::Cut( Mesh & mesh, ConcreteSide<DRT::Element::tri3> & side, PointSet & cuts )
{
  Intersection<DRT::Element::line2, DRT::Element::tri3> inter( mesh, *this, side );
  inter.Intersect( cuts );
}

void GEO::CUT::Edge::Cut( Mesh & mesh, ConcreteSide<DRT::Element::quad4> & side, PointSet & cuts )
{
  Intersection<DRT::Element::line2, DRT::Element::quad4> inter( mesh, *this, side );
  inter.Intersect( cuts );
}

void GEO::CUT::Edge::LevelSetCut( Mesh & mesh, LevelSetSide & side, PointSet & cuts )
{
  double blsv = BeginNode()->LSV();
  double elsv = EndNode()  ->LSV();

  bool cutfound = false;

  // version for single element cuts, here we need to watch for tolerances on
  // nodal cuts
  if ( fabs( blsv ) <= TOLERANCE )
  {
    cuts.insert( Point::InsertCut( this, &side, BeginNode() ) );
    cutfound = true;
  }
  if ( fabs( elsv ) <= TOLERANCE )
  {
    cuts.insert( Point::InsertCut( this, &side, EndNode() ) );
    cutfound = true;
  }

  if ( not cutfound )
  {
    if ( ( blsv < 0.0 and elsv > 0.0 ) or
         ( blsv > 0.0 and elsv < 0.0 ) )
    {
      double t = blsv / ( blsv-elsv );

      LINALG::Matrix<3,1> x1;
      LINALG::Matrix<3,1> x2;
      BeginNode()->Coordinates( x1.A() );
      EndNode()  ->Coordinates( x2.A() );

      LINALG::Matrix<3,1> x;
      x.Update( -1., x1, 1., x2, 0. );
      x.Update( 1., x1, t );
      Point * p = Point::NewPoint( mesh, x.A(), 2.*t-1., this, &side );
      cuts.insert( p );
    }
  }
}

void GEO::CUT::Edge::RectifyCutNumerics()
{
  if ( cut_points_.size() > 2 )
  {
    // Rectify numerical problems that occationally occur. There might be middle
    // points that do not know their position on an element side.
    //
    // This assumes linear sides. There might be a problem with quad4
    // sides. Those are actually not supported.

    std::map<Side*, std::vector<Point*>::iterator> sidecuts;
    for ( std::vector<Point*>::iterator pi=cut_points_.begin(); pi!=cut_points_.end(); ++pi )
    {
      Point * p = *pi;
      const plain_side_set & cutsides = p->CutSides();
      for ( plain_side_set::const_iterator i=cutsides.begin(); i!=cutsides.end(); ++i )
      {
        Side * s = *i;
        std::map<Side*, std::vector<Point*>::iterator>::iterator j = sidecuts.find( s );
        if ( j!=sidecuts.end() )
        {
          //if ( std::distance( j->second, pi ) > 1 )
          {
            for ( std::vector<Point*>::iterator i=j->second+1; i!=pi; ++i )
            {
              Point * p = *i;
              p->AddSide( s );
            }
          }
          j->second = pi;
        }
        else
        {
          sidecuts[s] = pi;
        }
      }
    }
  }
}
