#include "cut_facet.H"
#include "cut_triangulateFacet.H"
#include "cut_kernel.H"
#include "cut_side.H"
#include "cut_position2d.H"
#include <math.h>

/*-----------------------------------------------------------------------------------------------------------*
              Split the facet into appropriate number of tri and quad                           Sudhakar 04/12
              Work well for both convex and concave facets
*------------------------------------------------------------------------------------------------------------*/
void GEO::CUT::TriangulateFacet::SplitFacet()
{

  // An edge should contain only 2 end points
  // delete all remaining points on the edge
  KERNEL::DeleteInlinePts( ptlist_ );

  // Deal with zero point facet -- this should never occur, but happens in axel10 cut-test
  if( ptlist_.size()==0 )
    return;

  int numpts = ptlist_.size();

  if( numpts<3 )
  {
    dserror("A facet must have atleast 3 corners");
  }
  else if ( numpts==3 ) // form a Tri, and it is done
  {
    split_.push_back( ptlist_ );
    return;
  }
  else
  {
    split_.clear();

    // get concave (reflex) points of polygon
    std::string geoType;
    std::vector<int> ptConcavity = KERNEL::CheckConvexity(  ptlist_, geoType );

    // a convex polygon or a polygon with only one concave point can be
    // very easily split
    if( geoType=="convex" || geoType=="1ptConcave" )
    {
      SplitConvex_1ptConcave_Facet( ptConcavity );
    }
    // facet with more than 1 concave point
    else
    {
      SplitGeneralFacet( ptConcavity );
    }
  }
}

/*-------------------------------------------------------------------------------------*
   If a 4 noded facet is convex, a quad is made. If it is concave,
   it is split into appropriate triangles.
   The Gauss integration rule is available for quad, only if it is convex

                                   /\
                                  / .\
                                 /A .B\                                   sudhakar 04/12
                                /  ++  \
                               / +    + \
                               +        +
*--------------------------------------------------------------------------------------*/
void GEO::CUT::TriangulateFacet::Split4nodeFacet( std::vector<Point*> &poly,
                                                  bool callFromSplitAnyFacet )
{
  if( poly.size()!=4 )
    dserror("This is not a 4 noded facet");

  std::string geoType;
  std::vector<int> ptConcavity = KERNEL::CheckConvexity(  poly, geoType );

  int indStart=0;

  // convex quad can be directly added
  if( geoType=="convex" )
  {
    split_.push_back( poly );
    return;
  }
  // concave quad is split into two tri cells
  else if( geoType=="1ptConcave" )
  {
    indStart = ptConcavity[0];

    std::vector<Point*> temp1(3),temp2(3);
    temp1[0] = poly[indStart];
    temp1[1] = poly[(indStart+1)%4];
    temp1[2] = poly[(indStart+2)%4];

    temp2[0] = poly[indStart];
    temp2[1] = poly[(indStart+2)%4];
    temp2[2] = poly[(indStart+3)%4];

    if( split_.size()==0 )
    {
      split_.resize(2);
      split_[0] = temp1;
      split_[1] = temp2;
    }
    else
    {
      split_.push_back( temp1 );
      split_.push_back( temp2 );
    }
    return;
  }
  else // if there are two concave pts for 4 noded facet --> selfcut
  {
    // this means splitGeneralFacet is failed --> call earclipping
    /*if( callFromSplitAnyFacet )
    {
      std::cout<<"WARNING!!!! calling earclipping because splitanyfacet failed\n";
      ptConcavity = KERNEL::CheckConvexity(  ptlist_, geoType );
      split_.clear();
      EarClipping( ptConcavity, false );
      return;
    }*/
    std::cout<<"the points are\n";
    for( unsigned i=0;i<poly.size();i++ )
    {
      Point* p1 = poly[i];
      double x1[3];
      p1->Coordinates(x1);
      std::cout<<x1[0]<<"\t"<<x1[1]<<"\t"<<x1[2]<<"\n";
    }
    dserror( "a 4 noded facet cannot have more than 1 concave point:"
        "This means that the facet is selfcut" );
  }
}

/*---------------------------------------------------------------------------------------------------*
   A facet that is convex or having one concave point is split into 1 Tri and few Quad cells
   splitting starts with concave pt --- eliminate the need to check whether any other pt is
   inside the newCell formed                                                            Sudhakar 07/12
*----------------------------------------------------------------------------------------------------*/
void GEO::CUT::TriangulateFacet::SplitConvex_1ptConcave_Facet( std::vector<int> ptConcavity )
{
  if( ptConcavity.size() > 1 )
    dserror( "should be called only when the facet has one or no concave points" );

  int num = ptlist_.size();

  if( ptlist_.size()==3 )
  {
    split_.push_back( ptlist_ );
    return;
  }
  else if ( ptlist_.size()==4 )
  {
    Split4nodeFacet( ptlist_, false );
    return;
  }

  bool triDone = false, convex = true;;
  int firstPt=0,secondPt=0,thirdPt=0,lastPt=1,endPt=num-1;
  if( ptConcavity.size()==1 )
  {
    convex = false;
    firstPt = ptConcavity[0];
    lastPt = (firstPt+1)%num; //initialization
    if( firstPt!=0 )
      endPt = firstPt-1;
  }
  std::vector<Point*> newCell;

  while(1)
  {
    newCell.clear();

    secondPt = lastPt;  //properly initialized
    thirdPt = (secondPt+1)%num;

    newCell.push_back(ptlist_[firstPt]);
    newCell.push_back(ptlist_[secondPt]);
    newCell.push_back(ptlist_[thirdPt]);    // tri cell is now formed

    if( thirdPt==endPt )
      triDone = true;
    else if ( !triDone )      // check whether tri can be extended to quad cell
    {
      lastPt = (thirdPt+1)%num;
      newCell.push_back(ptlist_[lastPt]);
      if( lastPt==endPt )
        triDone = true;
    }

    KERNEL::DeleteInlinePts( newCell );

    if( newCell.size()==3 || convex )
      split_.push_back( newCell );
    else if( newCell.size()==4 ) //check the resulting quad is convex, else split into 2 tri
      Split4nodeFacet( newCell, true );
    else
      dserror( "should have either 2 or 3 points" );

    if( triDone )
      break;
  }
}

/*---------------------------------------------------------------------------------------------------*
 * Generalized facet splitting procedure which works for simple facets with any number  sudhakar 08/12
 * of concave points. Involves checking whether a reflex point is inside formed cell
 *---------------------------------------------------------------------------------------------------*/
void GEO::CUT::TriangulateFacet::SplitGeneralFacet( std::vector<int> ptConcavity )
{
  if( ptConcavity.size() < 2 )
    dserror( "Call TriangulateFacet::SplitConvex_1ptConcave_Facet in such cases" );

  if( ptlist_.size()==3 ) // directly form a Tri cell
  {
    split_.push_back( ptlist_ );
    return;
  }
  else if ( ptlist_.size()==4 ) // can be a convex or concave quad
  {
    Split4nodeFacet( ptlist_, false );
    return;
  }

  int num = ptlist_.size();
  int concsize = ptConcavity.size();

  int firstPt=0,secondPt=1,thirdPt=2,fourthPt=3;
  std::vector<Point*> newCell;

  while(1)
  {
    if( (num-concsize) < 4 ) // this means that no Quad cells can be formed for this geometry
    {
      EarClipping( ptConcavity );
      return;
    }

    newCell.clear();
    int ncross = 0;
    for( int i=0;i<num;++i )
    {
      newCell.clear();
      ncross++;
      int concNo = 0;

      if( i==0 )
      {
        firstPt = ptConcavity[concNo];
        secondPt = (firstPt+1)%num;
      }
      else
      {
        firstPt = (firstPt+1)%num;
        secondPt = (firstPt+1)%num;
      }
      if(std::find(ptConcavity.begin(), ptConcavity.end(), secondPt) != ptConcavity.end())
        continue;

      thirdPt = (secondPt+1)%num;

      newCell.push_back(ptlist_[firstPt]);
      newCell.push_back(ptlist_[secondPt]);
      newCell.push_back(ptlist_[thirdPt]);    // tri cell is now formed

      // if 3rd point is not a concave point, then Quad can be formed
      if(std::find(ptConcavity.begin(), ptConcavity.end(), thirdPt) == ptConcavity.end())
      {
        fourthPt = (thirdPt+1)%num;
        newCell.push_back(ptlist_[fourthPt]);
      }

      KERNEL::DeleteInlinePts( newCell );

      bool isEar=true;

      if( newCell.size()==3 )
      {
        for( unsigned j=0;j<ptConcavity.size();j++ )
        {
          unsigned reflInd = ptConcavity[j];

          // considered pt is one of the corners of newCell --> no need to check
          if(std::find(newCell.begin(), newCell.end(), ptlist_[reflInd]) != newCell.end())
            continue;

          if( KERNEL::PtInsideTriangle( newCell, ptlist_[reflInd]) )
          {
            isEar = false;
            break;
          }
        }
        if( isEar )
        {
          ptlist_.erase( ptlist_.begin()+secondPt ); // erase a pt to form new polygon
          split_.push_back( newCell );
          break;
        }
      }
      else if( newCell.size()==4 ) //check the resulting quad is convex, else split into 2 tri
      {
        for( unsigned j=0;j<ptConcavity.size();j++ )
        {
          unsigned reflInd = ptConcavity[j];
          // considered pt is one of the corners of newCell --> no need to check
          if(std::find(newCell.begin(), newCell.end(), ptlist_[reflInd]) != newCell.end())
            continue;

          if( KERNEL::PtInsideQuad( newCell, ptlist_[reflInd]) )
          {
            isEar = false;
            break;
          }
        }

        if( isEar )
        {
          Split4nodeFacet( newCell, true );

          // erase internal points of cell to form new polygon
          // when a point is deleted, all other points are renumbered
          // if-else condition to make sure correct points are deleted
          if( thirdPt==0 )
          {
            ptlist_.erase( ptlist_.begin()+secondPt );
            ptlist_.erase( ptlist_.begin()+thirdPt );
          }
          else
          {
            ptlist_.erase( ptlist_.begin()+thirdPt );
            ptlist_.erase( ptlist_.begin()+secondPt );
          }
          break;
        }
      }
      else
      {
        std::cout<<"number of points in the cell = "<<newCell.size()<<"\n";
        dserror( "neither tri nor quad: Something went wrong in SplitAnyFacet" );
      }

      if( ncross==num )
        dserror("cannot form cell even after making one cycle");
    }

    KERNEL::DeleteInlinePts( ptlist_ );
    num = ptlist_.size();
    if( num==3 )
    {
      split_.push_back( ptlist_ );
      return;
    }
    else if ( num==4 )
    {
      Split4nodeFacet( ptlist_, false );
      return;
    }
    else
    {
      ptConcavity.clear();
      std::string geoType;

      ptConcavity = KERNEL::CheckConvexity(  ptlist_, geoType );

      concsize = ptConcavity.size();
      if( concsize < 2 ) // new ptlist_ forms a convex facet
      {
        SplitConvex_1ptConcave_Facet( ptConcavity );
        return;
      }
    }
  }
}

/*------------------------------------------------------------------------------------------------------*
            check whether the polygon has two continuous concave points.
            At the moment this is unused                                         sudhakar 04/12
*-------------------------------------------------------------------------------------------------------*/
bool GEO::CUT::TriangulateFacet::HasTwoContinuousConcavePts( std::vector<int> ptConcavity )
{
  int siz = ptConcavity.size();
  if( siz<2 )
    return false;

  for( int i=0;i<siz;i++ )
  {
    int firstPt = ptConcavity[i];
    int seconPt = ptConcavity[(i+1)%siz];
    if( firstPt!=siz-1 )
    {
      if( (seconPt-firstPt)==1 )
        return true;
    }
    else if( (seconPt-firstPt)==siz-1 )
      return true;
  }
  return false;
}

/*-------------------------------------------------------------------------------------------------*
    Triangulation by ear clipping. Works for all cases, but costly.                 sudhakar 04/12
    Called when facets have two adjacent concave points
    During the process, if facet is free of adjacent concave points, splitanyfacet() is called
*--------------------------------------------------------------------------------------------------*/
void GEO::CUT::TriangulateFacet::EarClipping( std::vector<int> ptConcavity,   // list of concave points
                                              bool triOnly,                  // whether to create triangles only?
                                              bool DeleteInlinePts )    // how to deal with collinear points?
{
  std::vector<int> convex;

  // just form a Tri cell, and it is done
  if( ptlist_.size()==3 )
  {
    split_.push_back( ptlist_ );
    return;
  }

  // creates only triangles --> do not call SplitGeneralFacet() even if possible
  // when ear clipping is called directly from other functions
  if( triOnly )
  {
    split_.clear();

    if ( DeleteInlinePts )
    {
      KERNEL::DeleteInlinePts( ptlist_ );
      if( ptlist_.size()==3 ) // after deleting the inline points, it may have only 3 points
      {
        split_.push_back( ptlist_ );
        return;
      }
      if ( ptlist_.size() < 3 )
      {
        return;
      }
    }

    ptConcavity.clear();
    std::string geoType;

    ptConcavity = KERNEL::CheckConvexity( ptlist_, geoType, true, DeleteInlinePts );
  }

  while(1)
  {
    unsigned int split_size = split_.size();
    std::vector<int> reflex( ptConcavity );

    int polPts = ptlist_.size();
    convex.resize( polPts-reflex.size() );

    // Find the convex points
    // They are non-reflex points of the polygon
    int conNum=0;

    for( int i=0;i<polPts;i++ )
    {
      if( ptConcavity.size()>0 && i==ptConcavity[0] )
      {
        ptConcavity.erase( ptConcavity.begin() );
        continue;
      }
      else
      {
        convex[conNum] = i;
        conNum++;
      }
    }

    bool haveinlinepts = false;
    if ( not DeleteInlinePts )
    {
      haveinlinepts = KERNEL::HaveInlinePts( ptlist_ );
    }

    // if (i) is an ear, the triangle formed by (i-1),i and (i+1) should be completely within the polygon
    // Find first ear point, and make the triangle and remove this ear from the polygon
    std::vector<Point*> tri(3);
    for( int i=0;i<polPts;i++ )
    {
      unsigned ind0 = i-1;
      if(i==0)
        ind0 = polPts-1;
      unsigned ind2 = (i+1)%polPts;

      if( not DeleteInlinePts and haveinlinepts )
      {
        // if we have inline points, one of the first or third point (ind0 or ind2) should be a reflex point
        // this should work as all inline points are handled like reflex points
        if( std::find(convex.begin(), convex.end(), ind0) != convex.end() and std::find(convex.begin(), convex.end(), ind2) != convex.end() )
          continue;
      }

      // a reflex point cannot be a ear
      if(std::find(reflex.begin(), reflex.end(), i) != reflex.end())
        continue;

      tri[0] = ptlist_[ind0];
      tri[1] = ptlist_[i];
      tri[2] = ptlist_[ind2];

      bool isEar = true;

      // check whether any point of polygon is inside this Tri
      // only reflex points are to be checked
      for( unsigned j=0;j<reflex.size();j++ )
      {
        unsigned reflInd = reflex[j];
        if( reflInd==ind0 || reflInd==ind2 )
          continue;

        if( KERNEL::PtInsideTriangle( tri, ptlist_[reflInd], DeleteInlinePts) )
        {
          if ( ptlist_[reflInd] != ptlist_[ind0] and ptlist_[reflInd] != ptlist_[i] and ptlist_[reflInd] != ptlist_[ind2] )
          {
            isEar = false;
            break;
          }
        }
      }

      if( !isEar )
        continue;

      split_.push_back(tri);
      ptlist_.erase( ptlist_.begin()+i ); // the main pt of ear is removed, and new polygon is formed
      break;
    }

    if( ptlist_.size()<3 )
      dserror( "ear clipping produced 2 vertices polygon" );

    if ( DeleteInlinePts )
    {
      KERNEL::DeleteInlinePts( ptlist_ ); // delete inline points in the new polygon
    }

    if( ptlist_.size()==3 )
    {
      split_.push_back( ptlist_ );
      break;
    }

    else if( ptlist_.size()==4 && !triOnly )
    {
      Split4nodeFacet( ptlist_, false );
      break;
    }

    std::string str1;
    ptConcavity.clear();
    ptConcavity = KERNEL::CheckConvexity(  ptlist_, str1, true, DeleteInlinePts ); // concave points for the new polygon

    if( triOnly==false ) // if possible it shifts to splitGeneralFacet so that no of cells are reduced
    {
      if( ptConcavity.size() < 2 )
      {
        SplitConvex_1ptConcave_Facet( ptConcavity );
        return;
      }
      else if( (ptlist_.size()-ptConcavity.size()) > 3 )
      {
        SplitGeneralFacet( ptConcavity );
        return;
      }
    }
    if ( split_size == split_.size() )
    {
      throw std::runtime_error( "Ear clipping: no progress in the triangulation" );
    }
  }
}


/*-----------------------------------------------------------------------------------------------------------*
    Ear Clipping is a triangulation method for simple polygons (convex, concave, with holes).     wirtz 05/13
    It is simple and robust but not very efficient (O(n^2)).
    As input parameter the outer polygon (ptlist_) and the inner polygons (inlists_) are required.
    Triangles will be generated as output, which are all combined in one vector (split_).
*------------------------------------------------------------------------------------------------------------*/
void GEO::CUT::TriangulateFacet::EarClippingWithHoles( Side * parentside )
{

  while ( inlists_.size() != 0 )
  {
// 1) Transformation in local coordinates
    std::map<int, LINALG::Matrix<3,1> > localmaincyclepoints;
    std::map<std::vector<int>, LINALG::Matrix<3,1> > localholecyclespoints;
    int j = 1;   // index for referencing a maincyclepoint
    std::vector<int> k(2);
    k[0]=1;   // index for referencing a holecycle
    k[1]=1;   // index for referencing a holecyclepoint
    for ( std::vector<Point*>::iterator i=ptlist_.begin(); i!=ptlist_.end(); ++i )
    {
      Point * maincyclepoint = * i;
      LINALG::Matrix<3,1> maincyclepointcoordinates;
      LINALG::Matrix<3,1> localmaincyclepointcoordinates;
      maincyclepoint->Coordinates( maincyclepointcoordinates.A() );
      parentside->LocalCoordinates( maincyclepointcoordinates, localmaincyclepointcoordinates, false );
      localmaincyclepoints[j] = localmaincyclepointcoordinates;
      j++;
    }
    for ( std::list<std::vector<Point*> >::iterator i=inlists_.begin(); i!=inlists_.end(); ++i )
    {
      std::vector<Point*> holecyclepoints = * i;
      for ( std::vector<Point*>::iterator i=holecyclepoints.begin(); i!=holecyclepoints.end(); ++i)
      {
        Point * holecyclepoint = * i;
        LINALG::Matrix<3,1> holecyclepointcoordinates;
        LINALG::Matrix<3,1> localholecyclepointcoordinates;
        holecyclepoint->Coordinates( holecyclepointcoordinates.A() );
        parentside->LocalCoordinates( holecyclepointcoordinates, localholecyclepointcoordinates, false );
        localholecyclespoints[k] = localholecyclepointcoordinates;
        k[1] = k[1] +1;
      }
      k[0] = k[0] + 1;
      k[1] = 1;
    }
// 2) Holecyclepoint with the maximum x-value
    double maximumxvalue = -1;
    std::vector<int> maximumxvalueid(2);
    for( std::map<std::vector<int>, LINALG::Matrix<3,1> >::iterator i=localholecyclespoints.begin(); i!=localholecyclespoints.end(); ++i )
    {
      LINALG::Matrix<3,1> localholecyclespoint = i->second;
      if ( localholecyclespoint(0,0) > maximumxvalue)
      {
        maximumxvalue = localholecyclespoint(0,0);
        maximumxvalueid = i->first;
      }
    }
    double correspondingyvalue = (localholecyclespoints[maximumxvalueid])(1,0);
    LINALG::Matrix<3,1> maximumpoint;
    maximumpoint(0,0) = maximumxvalue;
    maximumpoint(1,0) = correspondingyvalue;
    maximumpoint(2,0) = 0;
// 3) Closest visible point
    int maincyclesize = ptlist_.size();
    double closestedgexvalue = 2;
    std::vector<int> closestedgexvalueid(2);
    int intersectioncount = 0;   // counts the edges which are intersected by the ray to check whether the holecycle is inside or outside the maincycle
    bool intersectioninpoint = false;// the counting with m is disturbed in case the ray hits a point of the maincycle
    for ( int i=1; i<=maincyclesize; ++i )
    {
      double edgepointxvalue1 = (localmaincyclepoints[i])(0,0);
      double edgepointxvalue2 = (localmaincyclepoints[(i%maincyclesize)+1])(0,0);
      double edgepointyvalue1 = (localmaincyclepoints[i])(1,0);
      double edgepointyvalue2 = (localmaincyclepoints[(i%maincyclesize)+1])(1,0);
      if ( ( (edgepointyvalue1 <= correspondingyvalue ) and (edgepointyvalue2 >= correspondingyvalue ) ) or
           ( (edgepointyvalue2 <= correspondingyvalue ) and (edgepointyvalue1 >= correspondingyvalue ) ) )
      {
        double edgepointxvalue = ((edgepointxvalue2 - edgepointxvalue1) * correspondingyvalue + edgepointxvalue1*edgepointyvalue2
                                 - edgepointxvalue2*edgepointyvalue1 )/(edgepointyvalue2 - edgepointyvalue1);
        if ( edgepointxvalue > maximumxvalue )
        {
          if ( edgepointxvalue < closestedgexvalue )
          {
            closestedgexvalue = edgepointxvalue;
            closestedgexvalueid[0] = i;
            closestedgexvalueid[1] = (i%maincyclesize)+1;
          }
          intersectioncount++;
          if ( (abs(edgepointxvalue - edgepointxvalue1) < TOLERANCE and abs(correspondingyvalue - edgepointyvalue1) < TOLERANCE ) or
               (abs(edgepointxvalue - edgepointxvalue2) < TOLERANCE and abs(correspondingyvalue - edgepointyvalue2) < TOLERANCE ) )
          {
            intersectioninpoint = true;
          }
        }
      }
    }
    double epsilon = TOLERANCE;
    while ( intersectioninpoint )
    {
      intersectioninpoint = false;
      for ( int i=1; i<=maincyclesize; ++i )
      {
        double edgepointxvalue1 = (localmaincyclepoints[i])(0,0);
        double edgepointxvalue2 = (localmaincyclepoints[(i%maincyclesize)+1])(0,0);
        double edgepointyvalue1 = (localmaincyclepoints[i])(1,0);
        double edgepointyvalue2 = (localmaincyclepoints[(i%maincyclesize)+1])(1,0);
        double A = edgepointxvalue1 - edgepointxvalue2;
        double B = edgepointyvalue1 - edgepointyvalue2;
        double C = edgepointxvalue1 + edgepointxvalue2 - 2*maximumxvalue - 2;
        double D = edgepointyvalue1 + edgepointyvalue2 - 2*correspondingyvalue - epsilon;
        double N = 2*B - epsilon*A;
        if ( abs(N) > TOLERANCE )
        {
          double eta = ( B*C - A*D )/N;
          double xsi = ( 2*D - epsilon*C )/N;
          if ( eta < 1 and eta > -1 and xsi < 1 and xsi > -1)
          {
            intersectioncount++;
            double xlocalcoord = maximumxvalue + 1 + eta;
            double ylocalcoord = (2*correspondingyvalue + epsilon + epsilon*eta)/2;
            if ( (abs(xlocalcoord - edgepointxvalue1) < TOLERANCE and abs(ylocalcoord - edgepointyvalue1) < TOLERANCE ) or
                 (abs(xlocalcoord - edgepointxvalue2) < TOLERANCE and abs(ylocalcoord - edgepointyvalue2) < TOLERANCE ) )
            {
              intersectioninpoint = true;
              intersectioncount = 0;
              epsilon += TOLERANCE;
              break;
            }
          }
        }
      }
    }
    if ( intersectioncount%2 == 0 )
    {
      int l = 1;   // index for searching the maximumxvalueid
      std::list<std::vector<Point*> >::iterator iter;
      for ( std::list<std::vector<Point*> >::iterator i=inlists_.begin(); i!=inlists_.end(); ++i )
      {
        if ( l == maximumxvalueid[0] )
        {
          iter = i;
        }
        l++;
      }
      inlists_.erase(iter);
      continue;
    }
    LINALG::Matrix<3,1> closestpoint;
    closestpoint(0,0) = closestedgexvalue;
    closestpoint(1,0) = correspondingyvalue;
    closestpoint(2,0) = 0;
  // 4) Closest visible point is node
    int mutuallyvisiblepointid = 0;
    if ( (abs(closestpoint(0,0) - localmaincyclepoints[closestedgexvalueid[0]](0,0)) < TOLERANCE ) and
         (abs(closestpoint(1,0) - localmaincyclepoints[closestedgexvalueid[0]](1,0)) < TOLERANCE ) )
    {
      mutuallyvisiblepointid = closestedgexvalueid[0];
    }
    else if ( (abs(closestpoint(0,0) - localmaincyclepoints[closestedgexvalueid[1]](0,0)) < TOLERANCE ) and
              (abs(closestpoint(1,0) - localmaincyclepoints[closestedgexvalueid[1]](1,0)) < TOLERANCE ) )
    {
      mutuallyvisiblepointid = closestedgexvalueid[1];
    }
    else
    {
  // 5) Closest edge point with the maximum x value
      int potentuallyvisiblepointid;
      LINALG::Matrix<3,1> potentuallyvisiblepoint;
      if ( localmaincyclepoints[closestedgexvalueid[0]](0,0) > localmaincyclepoints[closestedgexvalueid[1]](0,0) )
      {
        potentuallyvisiblepointid = closestedgexvalueid[0];
        potentuallyvisiblepoint(0,0) = localmaincyclepoints[closestedgexvalueid[0]](0,0);
        potentuallyvisiblepoint(1,0) = localmaincyclepoints[closestedgexvalueid[0]](1,0);
        potentuallyvisiblepoint(2,0) = 0;
      }
      else
      {
        potentuallyvisiblepointid = closestedgexvalueid[1];
        potentuallyvisiblepoint(0,0) = localmaincyclepoints[closestedgexvalueid[1]](0,0);
        potentuallyvisiblepoint(1,0) = localmaincyclepoints[closestedgexvalueid[1]](1,0);
        potentuallyvisiblepoint(2,0) = 0;
      }
  // 6) Reflex points in triangle
      std::string geoType;
      std::vector<int> reflexmaincyclepointids = KERNEL::CheckConvexity( ptlist_, geoType, false, false );
      for ( std::vector<int>::iterator i=reflexmaincyclepointids.begin(); i!=reflexmaincyclepointids.end(); ++i )
      {
        int *reflexmaincyclepointid = &*i;
        (*reflexmaincyclepointid)++;   // a little inconsistency here with KERNEL::CheckConvexity
      }
      std::vector<int> insidemaincyclepointids;
      LINALG::Matrix<3,3> triangle;
      for ( int i=0; i<3; ++i )
      {
        triangle(i,0) = maximumpoint(i,0);
        triangle(i,1) = closestpoint(i,0);
        triangle(i,2) = potentuallyvisiblepoint(i,0);
      }
      for ( std::vector<int>::iterator i=reflexmaincyclepointids.begin(); i!=reflexmaincyclepointids.end(); ++i )
      {
        int reflexmaincyclepointid = * i;
        LINALG::Matrix<3,1> reflexmaincyclepoint = localmaincyclepoints[reflexmaincyclepointid];
        Position2d<DRT::Element::tri3> pos( triangle, reflexmaincyclepoint );
        pos.Compute( false );
        if( pos.WithinLimitsTol( 1.0e-8, false, TOLERANCE) )
        {
          insidemaincyclepointids.push_back( reflexmaincyclepointid );
        }
      }
      if ( insidemaincyclepointids.size() == 0 )
      {
        mutuallyvisiblepointid = potentuallyvisiblepointid;
      }
      else
      {
  // 7) Closest angle
        double closestangle = 4;
        std::vector<int> closestanglemaincyclepointids;
        for ( std::vector<int>::iterator i=insidemaincyclepointids.begin(); i!=insidemaincyclepointids.end(); ++i )
        {
          int insidemaincyclepointid = * i;
          LINALG::Matrix<3,1> insidemaincyclepoint = localmaincyclepoints[insidemaincyclepointid];
          double distance1 = sqrt(pow((closestpoint(0,0) - maximumpoint(0,0)),2) + pow((closestpoint(1,0) - maximumpoint(1,0)),2));
          double distance2 = sqrt(pow((insidemaincyclepoint(0,0) - maximumpoint(0,0)),2) + pow((insidemaincyclepoint(1,0) - maximumpoint(1,0)),2));
          double distance3 = sqrt(pow((insidemaincyclepoint(0,0) - closestpoint(0,0)),2) + pow((insidemaincyclepoint(1,0) - closestpoint(1,0)),2));
          if ( (distance1 > 1.0e-8) && (distance2 > 1.0e-8)  )
          {
            double alpha = acos( ( distance1 * distance1 + distance2 * distance2 - distance3 * distance3 ) / ( 2 * distance1 * distance2 ) );
            if ( alpha - closestangle <= 1.0e-16 )
            {
              closestangle = alpha;
              closestanglemaincyclepointids.push_back( insidemaincyclepointid );
            }
          }
        }
        if ( closestanglemaincyclepointids.size() == 1 )
        {
          mutuallyvisiblepointid = closestanglemaincyclepointids[0];
        }
        else
        {
  // 8) Mutually visible point
          double closestdistance = 5;
          for ( std::vector<int>::iterator i=closestanglemaincyclepointids.begin(); i!=closestanglemaincyclepointids.end(); ++i )
          {
            int closestanglemaincyclepointid = * i;
            LINALG::Matrix<3,1> closestanglemaincyclepoint = localmaincyclepoints[closestanglemaincyclepointid];
            double distance = sqrt(pow((closestanglemaincyclepoint(0,0) - maximumpoint(0,0)),2) + pow((closestanglemaincyclepoint(1,0) - maximumpoint(1,0)),2));
            if ( distance < closestdistance )
            {
              mutuallyvisiblepointid = closestanglemaincyclepointid;
            }
          }
        }
      }
    }
  // 9) Make new pointlist
    std::vector<Point*> pttemp( ptlist_ );
    ptlist_.clear();
    for ( int i=0; i<mutuallyvisiblepointid; ++i )
    {
      ptlist_.push_back( pttemp[i] );
    }
    std::vector<Point*> inlist;
    int l = 1;
    std::list<std::vector<Point*> >::iterator iter;
    for ( std::list<std::vector<Point*> >::iterator i=inlists_.begin(); i!=inlists_.end(); ++i )
    {
      if ( l == maximumxvalueid[0] )
      {
        inlist = *i;
        iter = i;
        break;
      }
      l++;
    }
    for ( unsigned int j=(maximumxvalueid[1]-1); j<inlist.size(); ++j )
    {
      ptlist_.push_back( inlist[j] );
    }
    for ( int j=0; j<(maximumxvalueid[1]); ++j )
    {
      ptlist_.push_back( inlist[j] );
    }
    inlists_.erase(iter);
    for ( unsigned int i=mutuallyvisiblepointid-1; i<pttemp.size(); ++i )
    {
      ptlist_.push_back( pttemp[i] );
    }
  }
  std::vector<int> ptConcavity;
  this->EarClipping( ptConcavity, true, false );

}
