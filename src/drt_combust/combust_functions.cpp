/*!-----------------------------------------------------------------------------------------------*
 \file combust_functions.cpp

 \brief Managing and evaluating of spatial functions for combustion and two-phase flow problems

  detailed description in header file combust_interface.H

<pre>
Maintainer: Ursula Rasthofer
            rasthofer@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15236
</pre>
 *------------------------------------------------------------------------------------------------*/

#include "combust_functions.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/standardtypes_cpp.H"


/*----------------------------------------------------------------------*
 | constructor                                          rasthofer 01/14 |
 *----------------------------------------------------------------------*/
DRT::UTILS::BubbleFunction::BubbleFunction() :
Function()
{
}


/*----------------------------------------------------------------------*
 | initial level-set field for two merging bubbles      rasthofer 01/14 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::BubbleFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  //here calculation of distance (sign is already taken in consideration)
  double distance = 0.0;

  // compute phi-value with respect to the lower/small and the upper/large bubble
  double phi_small = std::sqrt((xp[0]-0.5)*(xp[0]-0.5) + (xp[1]-0.3)*(xp[1]-0.3) + xp[2]*xp[2])-0.1;
  double phi_large = std::sqrt((xp[0]-0.5)*(xp[0]-0.5) + (xp[1]-0.58)*(xp[1]-0.58) + xp[2]*xp[2])-0.15;

  // select correct phi-value
  if ( (phi_large <= 0.0) and (phi_small > 0.0) )
  {
    // current position inside large bubble
    distance = phi_large;
  }
  else if ( (phi_small <= 0.0) and (phi_large > 0.0) )
  {
    // current position inside small bubble
    distance = phi_small;
  }
  else if ( (phi_large > 0.0) and (phi_small > 0.0) )
  {
    // current position is not inside the bubbles
    // we take the smaller distance
    if (phi_large >= phi_small)
      distance = phi_small;
    else
      distance = phi_large;
  }
  else
    dserror("Initial level-set field for merging bubbles could not be set correctly!");

  return distance;
}


/*----------------------------------------------------------------------*
 | constructor                                              henke 05/09 |
 *----------------------------------------------------------------------*/
DRT::UTILS::ZalesaksDiskFunction::ZalesaksDiskFunction() :
Function()
{
}


/*----------------------------------------------------------------------*
 | evaluation of level set test function "Zalesak's disk"  schott 06/11 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::ZalesaksDiskFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  // the disk consists of 3 lines and a part of a circle and four points
  // decide if the orthogonal projection of the current point lies on the lines and the circle (four different distances possible)
  // additionally the smallest distance can be between the current point and one of the four corners

  double distance = 99999.0;

  //=====================================
  // distances to the four corners
  //=====================================
  // upper points
  double y_upper = std::sqrt(DSQR(0.15)-DSQR(0.025)) + 0.25;
  // warning: sign must be positive
  double dist_lu = std::sqrt(DSQR(xp[0]+0.025)+DSQR(xp[1]-y_upper));
  if(std::abs(dist_lu) < std::abs(distance)) distance = dist_lu;

  // warning: sign must be positive
  double dist_ru = std::sqrt(DSQR(xp[0]-0.025)+DSQR(xp[1]-y_upper));
  if(std::abs(dist_ru) < std::abs(distance)) distance = dist_ru;

  // under points
  double y_down = 0.15;
  // warning: sign must be negative
  double dist_ld = std::sqrt(DSQR(xp[0]+0.025)+DSQR(xp[1]-y_down));
  if(std::abs(dist_ld) < std::abs(distance)) distance = -dist_ld;

  // warning: sign must be negative
  double dist_rd = std::sqrt(DSQR(xp[0]-0.025)+DSQR(xp[1]-y_down));
  if(std::abs(dist_rd) < std::abs(distance)) distance = -dist_rd;

  //=====================================
  // projection on the 3 lines
  //=====================================
  // decide for vertical lines
  if(xp[1]>= 0.15 && xp[1]<= y_upper)
  {
    // leftVertLine
    if(std::abs(xp[0]+0.025) < std::abs(distance)) distance = xp[0]+0.025;

    // rightVertLine
    if(std::abs(0.025-xp[0]) < std::abs(distance)) distance = 0.025-xp[0];
  }
  // decide for horizontal line
  if(xp[0]>= -0.025 && xp[0]<=0.025)
  {
    // horizontalLine
    if(std::abs(xp[1]-0.15) < std::abs(distance)) distance = xp[1]-0.15;
  }

  //======================================
  // distance to the circle
  //======================================
  // decide for part of circle
  // get radius of sphere for current point
  double s = std::sqrt(DSQR(xp[0]-0.0)+DSQR(xp[1]-0.25));
  // get angle between line form midpoint of circle to current point and vector (0,1,0)
  double y_tmp= std::sqrt(DSQR(0.15)-DSQR(0.025))*s/0.15;
  if((xp[1]-0.25) <= y_tmp)
  {
    if(std::abs(s-0.15) < std::abs(distance)) distance = s-0.15;
  }

  return distance;
}


/*----------------------------------------------------------------------*
 | constructor                                              henke 01/12 |
 *----------------------------------------------------------------------*/
DRT::UTILS::CircularFlame2Function::CircularFlame2Function() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of circular flame test function henke               01/12 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::CircularFlame2Function::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  const double visc_minus = 0.001;
  const double dens_minus = 1.0;
  const double radius_0 = 0.025;
  const double u = 1.0;
  const double sl = 1.0;
  double radius = radius_0 + t*(u+sl);

  // -pres + dynvisc*2*du_x/dx
  double flux = 0.5*dens_minus*radius*radius*u*u/(xp[0]*xp[0]+xp[1]*xp[1])
              + 2.0*dens_minus*sl*u*log(sqrt(xp[0]*xp[0]+xp[1]*xp[1]))
              + visc_minus*2.0*radius*u/(xp[0]*xp[0]+xp[1]*xp[1])*(1.0-2.0*xp[0]*xp[0]/(xp[0]*xp[0]+xp[1]*xp[1]));

  return flux;
}

/*----------------------------------------------------------------------*
 | constructor                                              henke 01/12 |
 *----------------------------------------------------------------------*/
DRT::UTILS::CircularFlame3Function::CircularFlame3Function() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of circular flame test function henke               01/12 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::CircularFlame3Function::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  const double visc_minus = 0.001;
  const double dens_minus = 1.0;
  const double radius_0 = 0.025;
  const double u = 1.0;
  const double sl = 1.0;
  double radius = radius_0 + t*(u+sl);

  // -pres + dynvisc*2*du_y/dy
  double flux = 0.5*dens_minus*radius*radius*u*u/(xp[0]*xp[0]+xp[1]*xp[1])
              + 2.0*dens_minus*sl*u*log(sqrt(xp[0]*xp[0]+xp[1]*xp[1]))
              + visc_minus*2.0*radius*u/(xp[0]*xp[0]+xp[1]*xp[1])*(1.0-2.0*xp[1]*xp[1]/(xp[0]*xp[0]+xp[1]*xp[1]));

  return flux;
}

/*----------------------------------------------------------------------*
 | constructor                                              henke 01/12 |
 *----------------------------------------------------------------------*/
DRT::UTILS::CircularFlame4Function::CircularFlame4Function() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of circular flame test function henke               01/12 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::CircularFlame4Function::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  const double visc_minus = 0.001;
  const double radius_0 = 0.025;
  const double u = 1.0;
  const double sl = 1.0;
  double radius = radius_0 + t*(u+sl);

  // dynvisc*2*du_x/dy
  double flux = -2.0*visc_minus*radius*u*2.0*xp[0]*xp[1]/((xp[0]*xp[0]+xp[1]*xp[1])*(xp[0]*xp[0]+xp[1]*xp[1]));

  return flux;
}


/*----------------------------------------------------------------------*
 | constructor                                          rasthofer 01/14 |
 *----------------------------------------------------------------------*/
DRT::UTILS::DamBreakObstacle::DamBreakObstacle() :
Function()
{
}

/*----------------------------------------------------------------------*
 | 3D dam break with obstacle                           rasthofer 01/14 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::DamBreakObstacle::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  // here calculation of distance (sign is already taken in consideration)
  double distance = 0.0;

  double xp_corner[2];
  double xp_center[2];
  double radius=0.1288;

  xp_corner[0]=1.228; // - 0.0161; // should be 1.228
  xp_corner[1]=0.55;
  xp_center[0]=xp_corner[0]-radius;
  xp_center[1]=xp_corner[1]-radius;

  if (xp[0] <=xp_center[0] and xp[1] >= xp_center[1])
  {
     distance = xp[1]-xp_corner[1] ;
  }
  else if (xp[0] >=xp_center[0] and xp[1] <= xp_center[1] and !(xp[0]==xp_center[0] and xp[1]==xp_center[1]))
  {
      distance= xp[0]-xp_corner[0];
  }
  else if (xp[0] <xp_center[0] and xp[1] < xp_center[1])
  {
      if(xp[1]>(xp_corner[1]+(xp[0]-xp_corner[0])))
      {
          distance = - std::fabs(xp_corner[1] - xp[1]);
      }
      else
      {
          distance = - std::fabs(xp_corner[0] - xp[0]);
      }
  }
  else
  {
      distance = std::sqrt(DSQR(xp[0]-xp_center[0])+DSQR(xp[1]-xp_center[1]))-radius;
  }

  return distance;
}


/*----------------------------------------------------------------------*
 | constructor                                          rasthofer 04/10 |
 *----------------------------------------------------------------------*/
DRT::UTILS::CollapsingWaterColumnFunction::CollapsingWaterColumnFunction() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of two-phase flow test case               rasthofer 04/10 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::CollapsingWaterColumnFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  //here calculation of distance (sign is already taken in consideration)
  double distance = 0.0;

  double xp_corner[2];
  double xp_center[2];
  double radius=0.0; //0.03;

  xp_corner[0]=0.146; //0.144859;
  xp_corner[1]=0.292; //0.290859;
  xp_center[0]=xp_corner[0]-radius;
  xp_center[1]=xp_corner[1]-radius;


  if (xp[0] <=xp_center[0] and xp[1] >= xp_center[1])
  {
     distance = xp[1]-xp_corner[1] ;
  }
  else if (xp[0] >=xp_center[0] and xp[1] <= xp_center[1] and !(xp[0]==xp_center[0] and xp[1]==xp_center[1]))
  {
      distance= xp[0]-xp_corner[0];
  }
  else if (xp[0] <xp_center[0] and xp[1] < xp_center[1])
  {
      if(xp[1]>(xp_corner[1]+(xp[0]-xp_corner[0])))
      {
          distance = - fabs(xp_corner[1] - xp[1]);
      }
      else
      {
          distance = - fabs(xp_corner[0] - xp[0]);
      }
  }
  else
  {
      distance = sqrt(DSQR(xp[0]-xp_center[0])+DSQR(xp[1]-xp_center[1]))-radius;
  }

  return distance;
}


/*----------------------------------------------------------------------*
 | constructor                                          rasthofer 04/10 |
 *----------------------------------------------------------------------*/
DRT::UTILS::CollapsingWaterColumnFunctionCoarse::CollapsingWaterColumnFunctionCoarse() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of two-phase flow test case               rasthofer 04/10 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::CollapsingWaterColumnFunctionCoarse::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  double xp_corner[2];
  double xp_center[2];
    double distance = 0.0;
  double radius=0.03;

  //xp_corner[0]=0.146;//0.06
  xp_corner[0]=0.06;//0.06
  //xp_corner[1]=0.292;//0.06
  xp_corner[1]=0.06;//0.06

  xp_center[0]=xp_corner[0]-radius;
  xp_center[1]=xp_corner[1]-radius;


  if (xp[0] <=xp_center[0] and xp[1] >= xp_center[1])
  {
     distance = xp[1]-xp_corner[1] ;
  }
  else if (xp[0] >=xp_center[0] and xp[1] <= xp_center[1] and !(xp[0]==xp_center[0] and xp[1]==xp_center[1]))
  {
      distance= xp[0]-xp_corner[0];
  }
  else if (xp[0] <xp_center[0] and xp[1] < xp_center[1])
  {
      if(xp[1]>(xp_corner[1]+(xp[0]-xp_corner[0])))
      {
          distance = - fabs(xp_corner[1] - xp[1]);
      }
      else
      {
          distance = - fabs(xp_corner[0] - xp[0]);
      }
  }
  else
  {
      distance = sqrt(DSQR(xp[0]-xp_center[0])+DSQR(xp[1]-xp_center[1]))-radius;
  }

  return distance;
}


/*----------------------------------------------------------------------*
 | constructor                                              henke 10/11 |
 *----------------------------------------------------------------------*/
DRT::UTILS::ORACLESGFunction::ORACLESGFunction() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of level set test function "Zalesak's disk"   henke 05/09 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::ORACLESGFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{

  if (xp[0] > 0.0)
    dserror("invalid coordinate for ORACLES G-function function!");

  const double eps = 0.00152;
  //const double zsing = 0.7525-0.05;//0.0354;

  double gfuncvalue = 0.0;

  // implementation for periodic spanwise boundary
  if (xp[1] >= 0.0)
    gfuncvalue = (xp[1]-0.0354) - eps;
  else
    gfuncvalue = (-0.0354-xp[1]) - eps;

#if 0
  // implementation for spanwise walls
  if ( xp[2] <= -zsing and abs(xp[1]) <= abs(xp[2]+zsing) )
  {
    gfuncvalue = (-0.7525-xp[2]) - eps;
  }
  else if ( xp[2] >= zsing and abs(xp[1]) <= (xp[2]-zsing) )
  {
    gfuncvalue = (xp[2]-0.7525) - eps;
  }
  else if ( xp[1] >= 0.0 and ( xp[1] > abs(xp[2]+zsing) or xp[1] > (xp[2]-zsing) ))
  {
    gfuncvalue = (xp[1]-0.0354) - eps;
  }
  else if ( xp[1] < 0.0 and (-xp[1] > abs(xp[2]+zsing) or -xp[1] > (xp[2]-zsing) ))
  {
    gfuncvalue = (-0.0354-xp[1]) - eps;
  }
  else
    dserror("coordinate out of range of ORACLES G-function function");
#endif

  return gfuncvalue;
}

/*----------------------------------------------------------------------*
 | constructor                                             schott 05/11 |
 *----------------------------------------------------------------------*/
DRT::UTILS::RotatingConeFunction::RotatingConeFunction() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of level set test function "Rotating Cone "  schott 05/11 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::RotatingConeFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  //here calculation of distance (sign is already taken in consideration)
  double distance = 0;


  double x0c = 1.0/6.0;
  double x1c = 1.0/6.0;

  double sigma = 0.2;

  double X0 = (xp[0]-x0c)/sigma;
  double X1 = (xp[1]-x1c)/sigma;

  double radius = sqrt(DSQR(X0)+DSQR(X1));

  if(radius <= 1.0) distance = 0.25*(1.0+cos(PI*X0))*(1.0+cos(PI*X1));
  else distance = 0.0;


  return (distance-1.0);
}


/*----------------------------------------------------------------------*
 | constructor                                              henke 05/11 |
 *----------------------------------------------------------------------*/
DRT::UTILS::LevelSetCutTestFunction::LevelSetCutTestFunction() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of level set test function "Zalesak's disk"   henke 05/11 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::LevelSetCutTestFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  //here calculation of phi (sign is already taken in consideration)
  double phi = 0;

  // column of nodes (x < -0.7)
  if (xp[0] < -0.75)
    phi = xp[0]+0.6;
  // column of nodes (x = -0.7)
  else if ((xp[0] > -0.75) and (xp[0] < -0.65))
    phi = -0.1;
  // column of nodes (x = -0.6)
  else if ((xp[0] > -0.65) and (xp[0] < -0.55))
    phi = 0.0 +1.0E-4;
  // column of nodes (x = -0.5)
  else if ((xp[0] > -0.55) and (xp[0] < -0.45))
    phi = 0.1;
  // column of nodes (x = -0.4)
  else if ((xp[0] > -0.45) and (xp[0] < -0.35))
    phi = 0.0 +1.0E-5;
  // column of nodes (x = -0.3)
  else if ((xp[0] > -0.35) and (xp[0] < -0.25))
    phi = -0.1;
  // column of nodes (x = -0.2)
  else if ((xp[0] > -0.25) and (xp[0] < -0.15))
    phi = 0.0 +1.0E-12;
  // column of nodes (x = -0.1)
  else if ((xp[0] > -0.15) and (xp[0] < -0.05))
    phi = 0.1;
  // column of nodes (x = 0.0)
  else if ((xp[0] > -0.05) and (xp[0] < 0.05))
    phi = 0.0;
  // column of nodes (x = 0.1)
  else if ((xp[0] > 0.05) and (xp[0] < 0.15))
    phi = -0.1;
  // column of nodes (x = 0.2)
  else if ((xp[0] > 0.15) and (xp[0] < 0.25))
    phi = 0.0 -1.0E-12;
  // column of nodes (x = 0.3)
  else if ((xp[0] > 0.25) and (xp[0] < 0.35))
    phi = 0.1;
  // column of nodes (x = 0.4)
  else if ((xp[0] > 0.35) and (xp[0] < 0.45))
    phi = 0.0 -1.0E-5;
  // column of nodes (x = 0.5)
  else if ((xp[0] > 0.45) and (xp[0] < 0.55))
    phi = -0.1;
  // column of nodes (x = 0.6)
  else if ((xp[0] > 0.55) and (xp[0] < 0.65))
    phi = 0.0 +xp[1]*0.0001;
  // column of nodes (x = 0.7)
  else if ((xp[0] > 0.65) and (xp[0] < 0.75))
    phi = 0.1;
  // column of nodes (x = 0.8)
  else if ((xp[0] > 0.75) and (xp[0] < 0.85))
    phi = 0.0 -(xp[1]-0.001)*0.0001;
  // column of nodes (x = 0.9)
  else if ((xp[0] > 0.85) and (xp[0] < 0.95))
    phi = -0.1;
  // column of nodes (x = 1.0)
  else if (xp[0] > 0.95)
    phi = -0.2;
  // something went wrong
  else
   dserror("this node does not exist");
  return phi;
}
