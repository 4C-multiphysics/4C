/*!-----------------------------------------------------------------------------------------------*
 \file xfluid_functions.cpp

 \brief Managing and evaluating of spatial functions for Xfluid problems

  detailed description in header file combust_interface.H

<pre>
Maintainer: Magnus Winter
            winter@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15236
</pre>
 *------------------------------------------------------------------------------------------------*/

#include "xfluid_functions.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/standardtypes_cpp.H"


/*----------------------------------------------------------------------*
 | constructor                                          rasthofer 04/10 |
 *----------------------------------------------------------------------*/
DRT::UTILS::GerstenbergerForwardfacingStep::GerstenbergerForwardfacingStep() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of xfluid level set test case             winter    09/14 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::GerstenbergerForwardfacingStep::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  //  //cube_Gerstenberger:
  //  //1.6x1.6
  //  //=========================================================

  double xp_corner[2];
  double distance = 0.0;

  xp_corner[0]=0.75;
  xp_corner[1]=0.75;

  double xtmp=-(xp_corner[0]-xp[0]);
  double ytmp=-(xp[1]-xp_corner[1]);

  if(xtmp<ytmp)
    distance = -xtmp;
  else
    distance = -ytmp;

  //  //1.6x1.6 (collapsing water column from the right)
  //  //=========================================================
  //  radius = 0.375; //0.375
  //
  //  xp_corner[0]=0.75;
  //  xp_corner[1]=0.75;
  //
  ////  double totallength_x=1.6;
  //
  ////  xp_center[0]= (xp_corner[0]+radius) + totallength_x;
  //  xp_center[0]= xp_corner[0]+radius;
  //  xp_center[1]= xp_corner[1]-radius;
  //
  ////  xp_center[0]=xp_corner[0]-radius;
  ////  xp_center[1]=xp_corner[1]-radius;
  //
  //  if (xp[0] >=xp_center[0] and xp[1] >= xp_center[1])
  //  {
  //     distance = xp[1]-xp_corner[1];
  //  }
  //  else if (xp[0] <=xp_center[0] and xp[1] <= xp_center[1] and !(xp[0]==xp_center[0] and xp[1]==xp_center[1]))
  //  {
  //      distance= -(xp[0]-xp_corner[0]);
  //  }
  //  else if (xp[0] > xp_center[0] and xp[1] < xp_center[1])
  //  {
  //      if(xp[1]>(xp_corner[1]+(xp[0]-xp_corner[0])))
  //      {
  //          distance = - fabs(xp_corner[1] - xp[1]);
  //      }
  //      else
  //      {
  //          distance = - fabs(xp_corner[0] - xp[0]);
  //      }
  //  }
  //  else
  //  {
  //      distance = sqrt(DSQR(xp[0]-xp_center[0])+DSQR(xp[1]-xp_center[1]))-radius;
  //  }
    //=========================================================

  return distance;
}

/*----------------------------------------------------------------------*
 | constructor                                          winter    11/15 |
 *----------------------------------------------------------------------*/
DRT::UTILS::SlipLengthLevelSetManipulator::SlipLengthLevelSetManipulator() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of xfluid level set test case             winter    09/14 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::SlipLengthLevelSetManipulator::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{

  double x_cut = 0.4;
  double scalar;
//  if(xp[0]<x_cut)
//    scalar = 0.0;
//  else
//  {
//    scalar = 1000*(std::pow(1.6,xp[0]-x_cut));
//  }

    if(xp[0]>x_cut)
      scalar = 0.0;
    else
      scalar = 1000*(std::pow(1.6,x_cut-xp[0]));

  return scalar;
}

/*----------------------------------------------------------------------*
 | constructor                                            winter 10/15  |
 *----------------------------------------------------------------------*/
DRT::UTILS::MovingLevelSetCylinder::MovingLevelSetCylinder(std::vector<double>* origin, double radius,
    std::vector<double>* direction, double distance, double maxspeed) :
Function()
{
  // Origin of the geometry
  origin_ = *origin;

  // Radius
  radius_=radius;

  // Orientation of the geometry (symmetry axis)
  direction_= *direction; //needs to be normalized!

  double direction_norm = sqrt(direction_[0]*direction_[0]+direction_[1]*direction_[1]+direction_[2]*direction_[2]);

  direction_[0] /= direction_norm;
  direction_[1] /= direction_norm;
  direction_[2] /= direction_norm;

  // Distance traveled
  distance_= distance;

  // Distance traveled
  maxspeed_= maxspeed;

  //Initialize vector
  midpoint_trajectory_ = origin_;

  //Midpoint of trajectory
  midpoint_trajectory_[0] = 0.5*distance_*direction_[0]+origin_[0];
  midpoint_trajectory_[1] = 0.5*distance_*direction_[1]+origin_[1];
  midpoint_trajectory_[2] = 0.5*distance_*direction_[2]+origin_[2];

  //std::cout << "T_halfcycle: " << (L/2)/maxspeed * PI; //i.e. -d/2 -> d/2

}

/*----------------------------------------------------------------------*
 | evaluation of xfluid level set test case             winter    10/15 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::MovingLevelSetCylinder::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{

  // d = L/2 * sin(f*t-PI/2)
  // v = L*f/2 * cos(f*t-PI/2) = maxspeed * cos( (maxspeed*2/L)*t-PI/2 )

  //maxspeed =  L*f/2 -> f = 2*maxspeed/L
  //T_half   = (L/2)/maxspeed * PI/2

  //coefficient for sinus.
  double sin_coeff = maxspeed_*(2.0/distance_);
  //distance of cylinder viewed from the midpoint
  double dist = 0.5*distance_*sin(sin_coeff*t-PI*0.5);

  double x0_t = midpoint_trajectory_[0] + direction_[0]*dist;
  double x1_t = midpoint_trajectory_[1] + direction_[1]*dist;
//  double x2_t = midpoint_trajectory_[2] + direction_[2]*dist;

  double lsvalue = sqrt( (xp[0]-x0_t)*(xp[0]-x0_t) + (xp[1]-x1_t)*(xp[1]-x1_t) )-radius_;

  return lsvalue;
}

/*----------------------------------------------------------------------*
 | constructor                                            winter 10/15  |
 *----------------------------------------------------------------------*/
DRT::UTILS::TaylorCouetteFlow::TaylorCouetteFlow(
    double radius_inner,
    double radius_outer,
    double vel_theta_inner,
    double vel_theta_outer,
    double sliplength_inner) :
Function()
{
  radius_inner_     = radius_inner;
  radius_outer_     = radius_outer;
//
//  vel_theta_inner_  = vel_theta_inner;
//  vel_theta_outer_  = vel_theta_outer;
//
//  sliplength_inner_ = sliplength_inner;

  //Assume for now always -> krylov projection!

  double alpha=-1.0;

  double tmp = -2.0 * alpha * sliplength_inner/(radius_inner*radius_inner) +1/radius_inner;
  //double tmp_g = alpha * slip_length/mu*g_t;
  double tmp_g = 0.0;

  //A=c1_, B=c2_
  c1_ = (vel_theta_inner + tmp_g - vel_theta_outer*radius_outer*tmp)/(radius_inner - tmp*radius_outer*radius_outer);
  c2_ = vel_theta_outer*radius_outer - c1_*radius_outer*radius_outer;

//  pow(radius_outer,4.0)

  //KRYLOV - int{\Omega} p d\Omega  = 0.0

  double tmp_int_r2 = (c1_*c1_)*(radius_outer*radius_outer*radius_outer*radius_outer)/8.0
        + 2.0*c1_*c2_*(radius_outer*radius_outer)*(log(radius_outer)/2.0 - 1.0/4.0)
        - (c2_*c2_)/2.0*log(radius_outer);  //+C*r_2^2/2

  double tmp_int_r1 = (c1_*c1_)*(radius_inner*radius_inner*radius_inner*radius_inner)/8.0
        + 2.0*c1_*c2_*(radius_inner*radius_inner)*(log(radius_inner)/2.0 - 1.0/4.0)
        - (c2_*c2_)/2.0*log(radius_inner);  //+C*r_1^2/2

  c3_ = (tmp_int_r1-tmp_int_r2)/(0.5*((radius_outer*radius_outer)-(radius_inner*radius_inner)));




}

/*----------------------------------------------------------------------*
 | evaluation of Taylor-Couette analytical solution     winter    10/15 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::TaylorCouetteFlow::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{

  double radius = sqrt(xp[0]*xp[0] + xp[1]*xp[1]);

  // ISSUES WITH POINTS ON INNER CYLINDER! (Don't uncomment!)
//  if(radius > radius_outer_ || radius < radius_inner_)
//  {
//    std::cout << "xp[0]: " << xp[0] << ", xp[1]: " << xp[1] << std::endl;
//    std::cout << "radius: " << radius << ", radius_outer_: " << radius_outer_ << ", radius_inner_: " << radius_inner_  << std::endl;
//    dserror("Illegal radius!");
//  }

// If theta is needed at some point..
//  double theta;
//  if(xp[0]!=0.0)
//  {
//    theta = atan2(xp[1],xp[0]);
//  }
//  else
//  {
//    if(xp[1]>0)
//      theta=0.5*PI;
//    else
//      theta=1.5*PI; // Is this correct? Or -0.5PI?
//  }

  double u_theta = c1_ * radius + c2_/radius;

  switch (index)
  {
  case 0: // u,x
    return  -u_theta*xp[1]/radius;//sin(theta);
  case 1: // u,y
    return  u_theta*xp[0]/radius;//cos(theta);
  case 2: // u,z
    return  0.0;
  case 3: // v,x
    return  (c1_*c1_)*(radius*radius)*0.5 + 2.0*c1_*c2_*log(radius) - (c2_*c2_)/(2.0*(radius*radius)) + c3_;
  default:
    dserror("wrong index %d", index);
    break;
  }


  return 1.0;
}

std::vector<std::vector<double> > DRT::UTILS::TaylorCouetteFlow::FctDer(int index, const double* xp, const double t, DRT::Discretization* dis)
{
  //u_x = -(c1_*r + c2_/r)*y/r = -(c1_*y + c2_*y/(x^2+y^2))
  //d u_x /dx = c2_ * 2*x*y/((x^2+y^2)^2)
  //d u_x /dy = -c1_ - c2_ * (1/(x^2+y^2) - y*2*y/((x^2+y^2)^2))
  //d u_x /dz = 0.0

  //u_y =  (c1_*r + c2_/r)*x/r =  (c1_*x + c2_*x/(x^2+y^2))
  //d u_y /dx = c1_ + c2_ * (1/(x^2+y^2) - x*2*x/((x^2+y^2)^2))
  //d u_y /dy = - c2_ * 2*y*x/((x^2+y^2)^2)
  //d u_y /dz = 0.0

  //u_z = 0.0
  //d u_z / dx = d u_z / dy = d u_z / dz = 0.0

  const unsigned deg =1;
  // resulting vector holding
  std::vector<std::vector<double> > res(deg);
  for(unsigned i=0;i<deg;i++)
    res[i]= std::vector<double> (3,0.0);

  std::vector< double > tmp_vec;
  tmp_vec.reserve(3);

  double r_sqaured  = (xp[0]*xp[0]+xp[1]*xp[1]);   //(x^2+y^2)
  double r_sqaured2 = (xp[0]*xp[0]+xp[1]*xp[1])*(xp[0]*xp[0]+xp[1]*xp[1]); //(x^2+y^2)^2

  switch (index)
  {
  case 0:
  {
    tmp_vec[0] = c2_ * 2.0*xp[0]*xp[1]/(r_sqaured2);
    tmp_vec[1] =-c1_-c2_ *( 1/r_sqaured
                           -2.0*xp[1]*xp[1]/(r_sqaured2)
                          );
    tmp_vec[2] = 0.0;
    break;
  }
  case 1:
  {
    tmp_vec[0] = c1_+c2_ *(1/(r_sqaured)
                           -2.0*xp[0]*xp[0]/(r_sqaured2)
                          );
    tmp_vec[1] = - c2_ * 2.0*xp[1]*xp[0]/(r_sqaured2);
    tmp_vec[2] = 0.0;
    break;
  }
  case 2:
  {
    tmp_vec[0] = 0.0;
    tmp_vec[1] = 0.0;
    tmp_vec[2] = 0.0;
    break;
  }
  default:
    dserror("wrong index %d", index);
    break;
  }

  res[0][0]=tmp_vec[0];
  res[0][1]=tmp_vec[1];
  res[0][2]=tmp_vec[2];

  return res;
}
