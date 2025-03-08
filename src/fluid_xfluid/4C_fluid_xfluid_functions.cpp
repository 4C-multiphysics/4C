// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_xfluid_functions.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_function_manager.hpp"

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace
{
  using namespace Discret::Utils;

  std::shared_ptr<Core::Utils::FunctionOfSpaceTime> create_xfluid_function(
      const std::vector<Core::IO::InputParameterContainer>& parameters)
  {
    if (parameters.size() != 1) return nullptr;

    const auto& function_lin_def = parameters.front();

    const auto type = function_lin_def.get<std::string>("XFLUID_FUNCTION");
    if (type == "FORWARDFACINGSTEP")
    {
      return std::make_shared<GerstenbergerForwardfacingStep>();
    }
    else if (type == "MOVINGLEVELSETCYLINDER")
    {
      auto origin = function_lin_def.get<std::vector<double>>("ORIGIN");

      auto radius = function_lin_def.get<double>("RADIUS");

      auto direction = function_lin_def.get<std::vector<double>>("DIRECTION");

      auto distance = function_lin_def.get<double>("DISTANCE");

      auto maxspeed = function_lin_def.get<double>("MAXSPEED");

      return std::make_shared<MovingLevelSetCylinder>(
          &origin, radius, &direction, distance, maxspeed);
    }
    else if (type == "MOVINGLEVELSETTORUS" or type == "MOVINGLEVELSETTORUSVELOCITY" or
             type == "MOVINGLEVELSETTORUSSLIPLENGTH")
    {
      auto origin = function_lin_def.get<std::vector<double>>("ORIGIN");

      auto orient_vec_torus = function_lin_def.get<std::vector<double>>("ORIENTVEC_TORUS");

      auto radius = function_lin_def.get<double>("RADIUS");

      auto radius_tube = function_lin_def.get<double>("RADIUS_TUBE");

      auto direction = function_lin_def.get<std::vector<double>>("DIRECTION");

      auto distance = function_lin_def.get<double>("DISTANCE");

      auto maxspeed = function_lin_def.get<double>("MAXSPEED");

      auto rot_vec_torus = function_lin_def.get<std::vector<double>>("ROTATION_VEC");

      auto rotspeed = function_lin_def.get<double>("ROTATION_SPEED");  // revolutions per second

      auto rotramptime =
          function_lin_def.get<double>("ROTATION_RAMPTIME");  // revolutions per second

      if (type == "MOVINGLEVELSETTORUS")
        return std::make_shared<MovingLevelSetTorus>(&origin, &orient_vec_torus, radius,
            radius_tube, &direction, distance, maxspeed, &rot_vec_torus, rotspeed, rotramptime);
      else if (type == "MOVINGLEVELSETTORUSVELOCITY")
        return std::make_shared<MovingLevelSetTorusVelocity>(&origin, &orient_vec_torus, radius,
            radius_tube, &direction, distance, maxspeed, &rot_vec_torus, rotspeed, rotramptime);
      else if (type == "MOVINGLEVELSETTORUSSLIPLENGTH")
      {
        auto slipfunct = function_lin_def.get<int>("SLIP_FUNCT");
        return std::make_shared<MovingLevelSetTorusSliplength>(&origin, &orient_vec_torus, radius,
            radius_tube, &direction, distance, maxspeed, &rot_vec_torus, rotspeed, rotramptime,
            slipfunct);
      }
      else
      {
        FOUR_C_THROW("How did you end up here :)?");
        return std::shared_ptr<Core::Utils::FunctionOfSpaceTime>(nullptr);
      }
    }
    else if (type == "TAYLORCOUETTEFLOW")
    {
      auto radius_i = function_lin_def.get<double>("RADIUS_I");
      auto radius_o = function_lin_def.get<double>("RADIUS_O");

      auto vel_theta_i = function_lin_def.get<double>("VEL_THETA_I");
      auto vel_theta_o = function_lin_def.get<double>("VEL_THETA_O");

      auto sliplength_i = function_lin_def.get<double>("SLIPLENGTH_I");
      auto sliplength_o = function_lin_def.get<double>("SLIPLENGTH_O");

      auto traction_theta_i = function_lin_def.get<double>("TRACTION_THETA_I");
      auto traction_theta_o = function_lin_def.get<double>("TRACTION_THETA_O");

      auto viscosity = function_lin_def.get<double>("VISCOSITY");

      return std::make_shared<TaylorCouetteFlow>(radius_i, radius_o, vel_theta_i, vel_theta_o,
          sliplength_i, sliplength_o, traction_theta_i, traction_theta_o, viscosity);
    }
    else if (type == "URQUIZABOXFLOW")
    {
      auto lengthx = function_lin_def.get<double>("LENGTHX");
      auto lengthy = function_lin_def.get<double>("LENGTHY");

      auto rotation = function_lin_def.get<double>("ROTATION");
      auto viscosity = function_lin_def.get<double>("VISCOSITY");
      auto density = function_lin_def.get<double>("DENSITY");

      auto functno = function_lin_def.get<int>("CASE");

      std::vector<double> lin_comb(2, 0.0);
      if (function_lin_def.get_if<std::vector<double>>("COMBINATION") != nullptr)
      {
        lin_comb = function_lin_def.get<std::vector<double>>("COMBINATION");
      }
      else if (functno == 3)
        FOUR_C_THROW(
            "No combination of 2nd and 4th order terms given -> 0 velocity flow. NOT INTERESTING! "
            "DEFINE: COMBINATION C1 C2");

      return std::make_shared<UrquizaBoxFlow>(
          lengthx, lengthy, rotation, viscosity, density, functno, lin_comb);
    }
    else if (type == "URQUIZABOXFLOW_FORCE")
    {
      auto lengthx = function_lin_def.get<double>("LENGTHX");
      auto lengthy = function_lin_def.get<double>("LENGTHY");

      auto rotation = function_lin_def.get<double>("ROTATION");
      auto viscosity = function_lin_def.get<double>("VISCOSITY");
      auto density = function_lin_def.get<double>("DENSITY");

      auto functno = function_lin_def.get<int>("CASE");

      std::vector<double> lin_comb(2, 0.0);
      if (function_lin_def.get_if<std::vector<double>>("COMBINATION") != nullptr)
      {
        lin_comb = function_lin_def.get<std::vector<double>>("COMBINATION");
      }
      else if (functno == 3)
        FOUR_C_THROW(
            "No combination of 2nd and 4th order terms given -> 0 velocity flow. NOT INTERESTING! "
            "DEFINE: COMBINATION C1 C2");

      return std::make_shared<UrquizaBoxFlowForce>(
          lengthx, lengthy, rotation, viscosity, density, functno, lin_comb);
    }
    else if (type == "URQUIZABOXFLOW_TRACTION")
    {
      auto lengthx = function_lin_def.get<double>("LENGTHX");
      auto lengthy = function_lin_def.get<double>("LENGTHY");

      auto rotation = function_lin_def.get<double>("ROTATION");
      auto viscosity = function_lin_def.get<double>("VISCOSITY");
      auto density = function_lin_def.get<double>("DENSITY");

      auto functno = function_lin_def.get<int>("CASE");

      std::vector<double> lin_comb(2, 0.0);
      if (function_lin_def.get_if<std::vector<double>>("COMBINATION") != nullptr)
      {
        lin_comb = function_lin_def.get<std::vector<double>>("COMBINATION");
      }
      else if (functno == 3)
        FOUR_C_THROW(
            "No combination of 2nd and 4th order terms given -> 0 velocity flow. NOT INTERESTING! "
            "DEFINE: COMBINATION C1 C2");

      return std::make_shared<UrquizaBoxFlowTraction>(
          lengthx, lengthy, rotation, viscosity, density, functno, lin_comb);
    }
    else
    {
      return std::shared_ptr<Core::Utils::FunctionOfSpaceTime>(nullptr);
    }
  }
}  // namespace

void Discret::Utils::add_valid_xfluid_functions(Core::Utils::FunctionManager& function_manager)
{
  using namespace Core::IO::InputSpecBuilders;

  auto spec = one_of({
      deprecated_selection<std::string>("XFLUID_FUNCTION", {"FORWARDFACINGSTEP"}),
      all_of({
          deprecated_selection<std::string>("XFLUID_FUNCTION", {"MOVINGLEVELSETCYLINDER"}),
          parameter<std::vector<double>>("ORIGIN", {.size = 3}),
          parameter<double>("RADIUS"),
          parameter<std::vector<double>>("DIRECTION", {.size = 3}),
          parameter<double>("DISTANCE"),
          parameter<double>("MAXSPEED"),
      }),
      all_of({
          deprecated_selection<std::string>(
              "XFLUID_FUNCTION", {"MOVINGLEVELSETTORUS", "MOVINGLEVELSETTORUSVELOCITY"}),
          parameter<std::vector<double>>("ORIGIN", {.size = 3}),
          parameter<std::vector<double>>("ORIENTVEC_TORUS", {.size = 3}),
          parameter<double>("RADIUS"),
          parameter<double>("RADIUS_TUBE"),
          parameter<std::vector<double>>("DIRECTION", {.size = 3}),
          parameter<double>("DISTANCE"),
          parameter<double>("MAXSPEED"),
          parameter<std::vector<double>>("ROTATION_VEC", {.size = 3}),
          parameter<double>("ROTATION_SPEED"),
          parameter<double>("ROTATION_RAMPTIME"),
      }),
      all_of({
          deprecated_selection<std::string>("XFLUID_FUNCTION", {"MOVINGLEVELSETTORUSSLIPLENGTH"}),
          parameter<std::vector<double>>("ORIGIN", {.size = 3}),
          parameter<std::vector<double>>("ORIENTVEC_TORUS", {.size = 3}),
          parameter<double>("RADIUS"),
          parameter<double>("RADIUS_TUBE"),
          parameter<std::vector<double>>("DIRECTION", {.size = 3}),
          parameter<double>("DISTANCE"),
          parameter<double>("MAXSPEED"),
          parameter<std::vector<double>>("ROTATION_VEC", {.size = 3}),
          parameter<double>("ROTATION_SPEED"),
          parameter<double>("ROTATION_RAMPTIME"),
          parameter<int>("SLIP_FUNCT"),
      }),
      all_of({
          deprecated_selection<std::string>("XFLUID_FUNCTION", {"TAYLORCOUETTEFLOW"}),
          parameter<double>("RADIUS_I"),
          parameter<double>("RADIUS_O"),
          parameter<double>("VEL_THETA_I"),
          parameter<double>("VEL_THETA_O"),
          parameter<double>("SLIPLENGTH_I"),
          parameter<double>("SLIPLENGTH_O"),
          parameter<double>("TRACTION_THETA_I"),
          parameter<double>("TRACTION_THETA_O"),
          parameter<double>("VISCOSITY"),
      }),
      all_of({
          deprecated_selection<std::string>("XFLUID_FUNCTION",
              {"URQUIZABOXFLOW", "URQUIZABOXFLOW_TRACTION", "URQUIZABOXFLOW_FORCE"}),
          parameter<double>("LENGTHX"),
          parameter<double>("LENGTHY"),
          parameter<double>("ROTATION"),
          parameter<double>("VISCOSITY"),
          parameter<double>("DENSITY"),
          parameter<int>("CASE"),
          parameter<std::vector<double>>("COMBINATION", {.size = 2}),
      }),
  });

  function_manager.add_function_definition(std::move(spec), create_xfluid_function);
}



Discret::Utils::GerstenbergerForwardfacingStep::GerstenbergerForwardfacingStep() {}

double Discret::Utils::GerstenbergerForwardfacingStep::evaluate(
    const double* xp, const double t, const std::size_t component) const
{
  //  //cube_Gerstenberger:
  //  //1.6x1.6
  //  //=========================================================

  double xp_corner[2];
  double distance = 0.0;

  xp_corner[0] = 0.75;
  xp_corner[1] = 0.75;

  double xtmp = -(xp_corner[0] - xp[0]);
  double ytmp = -(xp[1] - xp_corner[1]);

  if (xtmp < ytmp)
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
  //  else if (xp[0] <=xp_center[0] and xp[1] <= xp_center[1] and !(xp[0]==xp_center[0] and
  //  xp[1]==xp_center[1]))
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
  //      distance = std::sqrt(((xp[0]-xp_center[0]) * (xp[0]-xp_center[0]))+((xp[1]-xp_center[1]) *
  //      (xp[1]-xp_center[1])))-radius;
  //  }
  //=========================================================

  return distance;
}


Discret::Utils::MovingLevelSetCylinder::MovingLevelSetCylinder(std::vector<double>* origin,
    double radius, std::vector<double>* direction, double distance, double maxspeed)
{
  // Origin of the geometry
  origin_ = *origin;

  // Radius
  radius_ = radius;

  // Orientation of the geometry (symmetry axis)
  direction_ = *direction;  // needs to be normalized!

  double direction_norm = std::sqrt(direction_[0] * direction_[0] + direction_[1] * direction_[1] +
                                    direction_[2] * direction_[2]);

  direction_[0] /= direction_norm;
  direction_[1] /= direction_norm;
  direction_[2] /= direction_norm;

  // Distance traveled
  distance_ = distance;

  // Distance traveled
  maxspeed_ = maxspeed;

  // Initialize vector
  midpoint_trajectory_ = origin_;

  // Midpoint of trajectory
  midpoint_trajectory_[0] = 0.5 * distance_ * direction_[0] + origin_[0];
  midpoint_trajectory_[1] = 0.5 * distance_ * direction_[1] + origin_[1];
  midpoint_trajectory_[2] = 0.5 * distance_ * direction_[2] + origin_[2];

  // std::cout << "T_halfcycle: " << (L/2)/maxspeed * PI; //i.e. -d/2 -> d/2
}

double Discret::Utils::MovingLevelSetCylinder::evaluate(
    const double* xp, const double t, const std::size_t component) const
{
  // d = L/2 * sin(f*t-PI/2)
  // v = L*f/2 * cos(f*t-PI/2) = maxspeed * cos( (maxspeed*2/L)*t-PI/2 )

  // maxspeed =  L*f/2 -> f = 2*maxspeed/L
  // T_half   = (L/2)/maxspeed * PI/2

  // coefficient for sinus.
  double sin_coeff = maxspeed_ * (2.0 / distance_);
  // distance of cylinder viewed from the midpoint
  double dist = 0.5 * distance_ * sin(sin_coeff * t - M_PI * 0.5);

  double x0_t = midpoint_trajectory_[0] + direction_[0] * dist;
  double x1_t = midpoint_trajectory_[1] + direction_[1] * dist;
  //  double x2_t = midpoint_trajectory_[2] + direction_[2]*dist;

  double lsvalue =
      std::sqrt((xp[0] - x0_t) * (xp[0] - x0_t) + (xp[1] - x1_t) * (xp[1] - x1_t)) - radius_;

  return lsvalue;
}


Discret::Utils::MovingLSTorus::MovingLSTorus(std::vector<double>* origin,
    std::vector<double>* orientationvec_torus, double radius, double radius_tube,
    std::vector<double>* direction, double distance, double maxspeed,
    std::vector<double>* rotvector, double rotspeed, double rotramptime)
    : eye_(3, std::vector<double>(3, 0.0)),
      rot_joint_(3, std::vector<double>(3, 0.0)),
      rot_cross_(3, std::vector<double>(3, 0.0))
{
  // Origin of the geometry
  origin_ = *origin;

  // Orientation vector of torus
  orientationvec_torus_ = *orientationvec_torus;
  double direction_norm_torus = std::sqrt(orientationvec_torus_[0] * orientationvec_torus_[0] +
                                          orientationvec_torus_[1] * orientationvec_torus_[1] +
                                          orientationvec_torus_[2] * orientationvec_torus_[2]);
  orientationvec_torus_[0] /= direction_norm_torus;
  orientationvec_torus_[1] /= direction_norm_torus;
  orientationvec_torus_[2] /= direction_norm_torus;

  // Radius
  radius_ = radius;

  // Radius of torus tube
  radius_tube_ = radius_tube;

  // Orientation of the geometry (symmetry axis)
  direction_ = *direction;  // needs to be normalized!
  double direction_norm = std::sqrt(direction_[0] * direction_[0] + direction_[1] * direction_[1] +
                                    direction_[2] * direction_[2]);
  direction_[0] /= direction_norm;
  direction_[1] /= direction_norm;
  direction_[2] /= direction_norm;

  // Distance traveled
  distance_ = distance;

  // Distance traveled
  maxspeed_ = maxspeed;

  // Rotation vector
  rotvector_ = *rotvector;
  double rotvector_norm = std::sqrt(rotvector_[0] * rotvector_[0] + rotvector_[1] * rotvector_[1] +
                                    rotvector_[2] * rotvector_[2]);
  rotvector_[0] /= rotvector_norm;
  rotvector_[1] /= rotvector_norm;
  rotvector_[2] /= rotvector_norm;

  // Rotation speed
  rotspeed_ = 2 * M_PI * rotspeed;  // rotation-speed:       revolutions/sec

  // Ramp time
  ramptime_ = rotramptime;  // ADD VALUE HERE!

  // Initialize vector
  midpoint_trajectory_ = origin_;

  // Midpoint of trajectory
  midpoint_trajectory_[0] = 0.5 * distance_ * direction_[0] + origin_[0];
  midpoint_trajectory_[1] = 0.5 * distance_ * direction_[1] + origin_[1];
  midpoint_trajectory_[2] = 0.5 * distance_ * direction_[2] + origin_[2];

  //++++++++++++++++++++++++++++++++++++++++

  // Create identity matrix
  // std::vector< std::vector<double> > eye(3, std::vector<double>(3));
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) (eye_[i])[j] = 0;

  (eye_[0])[0] = 1.0;
  (eye_[1])[1] = 1.0;
  (eye_[2])[2] = 1.0;

  //---------
  // omega (x) omega
  // std::vector< std::vector<double> > rot_joint(3, std::vector<double>(3));
  // rot_joint_(nsd,std::vector<double>(nsd));
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) (rot_joint_[i])[j] = rotvector_[i] * rotvector_[j];

  //---------
  // cross-product matrix
  // w_i x r_j = W_ij * r_j = - r_i * W_ij
  // std::vector< std::vector<double> > rot_cross(3, std::vector<double>(3));
  // rot_cross_(nsd,std::vector<double>(nsd));
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) (rot_cross_[i])[j] = 0.0;

  (rot_cross_[0])[1] = -rotvector_[2];
  (rot_cross_[1])[0] = rotvector_[2];

  (rot_cross_[0])[2] = rotvector_[1];
  (rot_cross_[2])[0] = -rotvector_[1];

  (rot_cross_[1])[2] = -rotvector_[0];
  (rot_cross_[1])[2] = rotvector_[0];


  // std::cout << "T_halfcycle: " << (L/2)/maxspeed * PI; //i.e. -d/2 -> d/2
}


Discret::Utils::MovingLevelSetTorus::MovingLevelSetTorus(std::vector<double>* origin,
    std::vector<double>* orientationvec_torus, double radius, double radius_tube,
    std::vector<double>* direction, double distance, double maxspeed,
    std::vector<double>* rotvector, double rotspeed, double rotramptime)
    : MovingLSTorus(origin, orientationvec_torus, radius, radius_tube, direction, distance,
          maxspeed, rotvector, rotspeed, rotramptime)
{
}

double Discret::Utils::MovingLevelSetTorus::evaluate(
    const double* xp, const double t, const std::size_t component) const
{
  // d = L/2 * sin(f*t-PI/2)
  // v = L*f/2 * cos(f*t-PI/2) = maxspeed * cos( (maxspeed*2/L)*t-PI/2 )

  // maxspeed =  L*f/2 -> f = 2*maxspeed/L
  // T_half   = (L/2)/maxspeed * PI/2

  // coefficient for sinus.
  double dist;
  if (distance_ > 1e-14)
  {
    double sin_coeff = maxspeed_ * (2.0 / distance_);
    // distance of cylinder viewed from the midpoint
    dist = 0.5 * distance_ * sin(sin_coeff * t - M_PI * 0.5);
  }
  else
    dist = 0.0;

  // This is the origin of the torus!
  double x0_t = midpoint_trajectory_[0] + direction_[0] * dist;
  double x1_t = midpoint_trajectory_[1] + direction_[1] * dist;
  double x2_t = midpoint_trajectory_[2] + direction_[2] * dist;

  // Set the frame of reference around this point
  std::vector<double> x_based_trajectory(3);
  x_based_trajectory[0] = xp[0] - x0_t;
  x_based_trajectory[1] = xp[1] - x1_t;
  x_based_trajectory[2] = xp[2] - x2_t;

  std::vector<double> x_rot_based_traject(3);
  std::vector<double> orientvec_torus_t(3);

  x_rot_based_traject = x_based_trajectory;

  // Add the rotation!
  double rotspeed_t = rotspeed_;
  if (rotspeed_ != 0.0)
  {
    if (t < ramptime_ and ramptime_ != 0.0)
      rotspeed_t = 0.5 * (1.0 - cos(M_PI * t / ramptime_)) *
                   rotspeed_;  // Gives zero acceleration in the beginning of used!
    //    else
    //      rotspeed_t = rotspeed_;

    double theta = rotspeed_t * t;


    for (int i = 0; i < 3; i++)
    {
      // x_rot_based_traject[i] = 0.0;
      orientvec_torus_t[i] = 0.0;
      for (int j = 0; j < 3; j++)
      {
        double rotmat_ij = cos(theta) * eye_[i][j] + sin(theta) * rot_cross_[i][j] +
                           (1 - cos(theta)) * rot_joint_[i][j];

        //        double rotmat_ji = cos(theta)*(eye_[j])[i]
        //                           + sin(theta)*(rot_cross_[j])[i]
        //                           + (1.0-cos(theta))*(rot_joint_[j])[i];

        // x^{rot}_i = R_ij x^{un-rot}_j
        //        x_rot_based_traject[i]+=( rotmat_ji ) * x_based_trajectory[j];
        orientvec_torus_t[i] = rotmat_ij * orientationvec_torus_[j];
      }
    }
  }
  else
  {
    // x_rot_based_traject=x_based_trajectory;
    orientvec_torus_t = orientationvec_torus_;
  }

  // Level Set value for torus in x-y plane!
  // double lsvalue = std::sqrt(
  //                 (std::sqrt( (xp[0]-x0_t)*(xp[0]-x0_t) + (xp[1]-x1_t)*(xp[1]-x1_t) )-radius_)
  //                *(std::sqrt( (xp[0]-x0_t)*(xp[0]-x0_t) + (xp[1]-x1_t)*(xp[1]-x1_t) )-radius_)
  //                + (xp[2]-x2_t)*(xp[2]-x2_t)
  //                     ) - radius_tube_ ;
  double orthogonal_val = orientvec_torus_t[0] * x_rot_based_traject[0] +
                          orientvec_torus_t[1] * x_rot_based_traject[1] +
                          orientvec_torus_t[2] * x_rot_based_traject[2];

  std::vector<double> base_vec(3);
  base_vec[0] = x_rot_based_traject[0] - orthogonal_val * orientvec_torus_t[0];
  base_vec[1] = x_rot_based_traject[1] - orthogonal_val * orientvec_torus_t[1];
  base_vec[2] = x_rot_based_traject[2] - orthogonal_val * orientvec_torus_t[2];

  // double lsvalue = 0.0;

  double r_base =
      std::sqrt(base_vec[0] * base_vec[0] + base_vec[1] * base_vec[1] + base_vec[2] * base_vec[2]);
  double lsvalue =
      std::sqrt((radius_ - r_base) * (radius_ - r_base) + orthogonal_val * orthogonal_val) -
      radius_tube_;

  //  if(abs(lsvalue)<0.1)
  //    std::cout << "lsvalue: " << lsvalue << std::endl;

  return lsvalue;
}


Discret::Utils::MovingLevelSetTorusVelocity::MovingLevelSetTorusVelocity(
    std::vector<double>* origin, std::vector<double>* orientationvec_torus, double radius,
    double radius_tube, std::vector<double>* direction, double distance, double maxspeed,
    std::vector<double>* rotvector, double rotspeed, double rotramptime)
    : MovingLSTorus(origin, orientationvec_torus, radius, radius_tube, direction, distance,
          maxspeed, rotvector, rotspeed, rotramptime)
{
}

double Discret::Utils::MovingLevelSetTorusVelocity::evaluate(
    const double* xp, const double t, const std::size_t component) const
{
  // d = L/2 * sin(f*t-PI/2)
  // v = L*f/2 * cos(f*t-PI/2) = maxspeed * cos( (maxspeed*2/L)*t-PI/2 )

  // maxspeed =  L*f/2 -> f = 2*maxspeed/L
  // T_half   = (L/2)/maxspeed * PI/2
  double dist;
  double vel;
  if (distance_ > 1e-14)
  {
    double sin_coeff = maxspeed_ * (2.0 / distance_);
    // distance of cylinder viewed from the midpoint
    dist = 0.5 * distance_ * sin(sin_coeff * t - M_PI * 0.5);
    vel = 0.5 * distance_ * sin_coeff * cos(sin_coeff * t - M_PI * 0.5);
  }
  else
  {
    dist = 0.0;
    vel = 0.0;
  }

  std::vector<double> vel_translation(3);
  // This is the origin of the torus!
  vel_translation[0] = direction_[0] * vel;
  vel_translation[1] = direction_[1] * vel;
  vel_translation[2] = direction_[2] * vel;

  // This is the origin of the torus!
  double x0_t = midpoint_trajectory_[0] + direction_[0] * dist;
  double x1_t = midpoint_trajectory_[1] + direction_[1] * dist;
  double x2_t = midpoint_trajectory_[2] + direction_[2] * dist;

  // Set the frame of reference around this point
  std::vector<double> x_based_trajectory(3);
  x_based_trajectory[0] = xp[0] - x0_t;
  x_based_trajectory[1] = xp[1] - x1_t;
  x_based_trajectory[2] = xp[2] - x2_t;

  // Add the rotation!  (THIS NEEDED?????? PROLLY NO!)
  // double theta = rotspeed_*t;
  //  std::vector<double> x_rot_based_traject;
  //  for(int i=0; i<3; i++)
  //  {
  //    x_rot_based_traject[i] = 0.0;
  //    for(int j=0; j<3; j++)
  //    {
  //      //x^{rot}_i = R_ij x^{un-rot}_j
  //      x_rot_based_traject[i]+=( cos(theta)*eye_[i][j]
  //                              + sin(theta)*rot_cross_[i][j]
  //                              + (1-cos(theta))*rot_joint_[i][j] ) * x_based_trajectory[j];
  //    }
  //  }

  // velocity rotation
  std::vector<double> vel_rotation(3);
  // rotation speed
  double rotspeed_t = rotspeed_;
  if (t < ramptime_ and ramptime_ != 0.0)
    rotspeed_t = 0.5 * (1.0 - cos(M_PI * t / ramptime_)) *
                 rotspeed_;  // Gives zero acceleration in the beginning of used!

  for (int i = 0; i < 3; i++)
  {
    vel_rotation[i] = 0.0;
    for (int j = 0; j < 3; j++)
    {
      // x^{rot}_i = R_ij x^{un-rot}_j
      vel_rotation[i] += rotspeed_t * (rot_cross_[i][j]) *
                         x_based_trajectory[j];  // Or should it be x_rot_based_traject?
    }
  }

  switch (component)
  {
    case 0:
      return (vel_translation[0] + vel_rotation[0]);
    case 1:
      return (vel_translation[1] + vel_rotation[1]);
    case 2:
      return (vel_translation[2] + vel_rotation[2]);
    default:
    {
      FOUR_C_THROW("Only u_x, u_y, u_z are provided! No pressure or other higher indices!");
      return 1.0;
    }
  }
}


Discret::Utils::MovingLevelSetTorusSliplength::MovingLevelSetTorusSliplength(
    std::vector<double>* origin, std::vector<double>* orientationvec_torus, double radius,
    double radius_tube, std::vector<double>* direction, double distance, double maxspeed,
    std::vector<double>* rotvector, double rotspeed, double rotramptime, int slipfunct)
    : MovingLSTorus(origin, orientationvec_torus, radius, radius_tube, direction, distance,
          maxspeed, rotvector, rotspeed, rotramptime),
      slipfunct_(slipfunct)
{
  // Check if the slip function is valid!
  if (slipfunct_ == 0)
    std::cout << "MOVINGLEVELSETTORUSSLIPLENGTH[0]: The slip function is 0.0 everywhere!"
              << std::endl;
  else if (slipfunct_ == 1)
    std::cout << "MOVINGLEVELSETTORUSSLIPLENGTH[1]: You have chosen the Spherical shaped "
                 "increase/decrease of the slip-function! (Inner WDBC (eps ~ 0.0) - Outer NavSlip "
                 "(eps ~ infty))"
              << std::endl;
  else if (slipfunct_ == 2)
    std::cout << "MOVINGLEVELSETTORUSSLIPLENGTH[2]: You have chosen the Spherical shaped "
                 "increase/decrease of the slip-function! (Inner NavSlip (eps ~ infty) - Outer "
                 "WDBC (eps ~ 0.0))"
              << std::endl;
  else if (slipfunct_ == 3)
    std::cout << "MOVINGLEVELSETTORUSSLIPLENGTH[3]: You have chosen the Cylindrical shaped "
                 "increase/decrease of the slip-function! (Inner WDBC (eps ~ 0.0) - Outer NavSlip "
                 "(eps ~ infty))"
              << std::endl;
  else if (slipfunct_ == 4)
    std::cout << "MOVINGLEVELSETTORUSSLIPLENGTH[4]: You have chosen the Cylindrical shaped "
                 "increase/decrease of the slip-function! (Inner NavSlip (eps ~ infty) - Outer "
                 "WDBC (eps ~ 0.0))"
              << std::endl;
  else
  {
    std::cout << slipfunct_ << std::endl;
    FOUR_C_THROW("The chosen function is not supported at the moment!!!");
  }
}

double Discret::Utils::MovingLevelSetTorusSliplength::evaluate(
    const double* xp, const double t, const std::size_t component) const
{
  // coefficient for sinus.
  double dist;
  if (distance_ > 1e-14)
  {
    double sin_coeff = maxspeed_ * (2.0 / distance_);
    // distance of cylinder viewed from the midpoint
    dist = 0.5 * distance_ * sin(sin_coeff * t - M_PI * 0.5);
  }
  else
    dist = 0.0;

  // This is the origin of the torus!
  double x0_t = midpoint_trajectory_[0] + direction_[0] * dist;
  double x1_t = midpoint_trajectory_[1] + direction_[1] * dist;
  double x2_t = midpoint_trajectory_[2] + direction_[2] * dist;

  // Set the frame of reference around this point
  std::vector<double> x_based_trajectory(3);
  x_based_trajectory[0] = xp[0] - x0_t;
  x_based_trajectory[1] = xp[1] - x1_t;
  x_based_trajectory[2] = xp[2] - x2_t;

  std::vector<double> x_rot_based_traject(3);  // actually redundant!
  std::vector<double> orientvec_torus_t(3);

  x_rot_based_traject = x_based_trajectory;

  // Add the rotation!
  double rotspeed_t = rotspeed_;
  if (rotspeed_ != 0.0)
  {
    if (t < ramptime_ and ramptime_ != 0.0)
      rotspeed_t = 0.5 * (1.0 - cos(M_PI * t / ramptime_)) *
                   rotspeed_;  // Gives zero acceleration in the beginning of used!
    //    else
    //      rotspeed_t = rotspeed_;

    double theta = rotspeed_t * t;


    for (int i = 0; i < 3; i++)
    {
      // x_rot_based_traject[i] = 0.0;
      orientvec_torus_t[i] = 0.0;
      for (int j = 0; j < 3; j++)
      {
        double rotmat_ij = cos(theta) * eye_[i][j] + sin(theta) * rot_cross_[i][j] +
                           (1 - cos(theta)) * rot_joint_[i][j];

        //        double rotmat_ji = cos(theta)*(eye_[j])[i]
        //                           + sin(theta)*(rot_cross_[j])[i]
        //                           + (1.0-cos(theta))*(rot_joint_[j])[i];

        // x^{rot}_i = R_ij x^{un-rot}_j
        //        x_rot_based_traject[i]+=( rotmat_ji ) * x_based_trajectory[j];
        orientvec_torus_t[i] = rotmat_ij * orientationvec_torus_[j];
      }
    }
  }
  else
  {
    // x_rot_based_traject=x_based_trajectory;
    orientvec_torus_t = orientationvec_torus_;
  }

  // Level Set value for torus in x-y plane!
  // double lsvalue = std::sqrt(
  //                 (std::sqrt( (xp[0]-x0_t)*(xp[0]-x0_t) + (xp[1]-x1_t)*(xp[1]-x1_t) )-radius_)
  //                *(std::sqrt( (xp[0]-x0_t)*(xp[0]-x0_t) + (xp[1]-x1_t)*(xp[1]-x1_t) )-radius_)
  //                + (xp[2]-x2_t)*(xp[2]-x2_t)
  //                     ) - radius_tube_ ;
  double orthogonal_val = orientvec_torus_t[0] * x_rot_based_traject[0] +
                          orientvec_torus_t[1] * x_rot_based_traject[1] +
                          orientvec_torus_t[2] * x_rot_based_traject[2];

  std::vector<double> base_vec(3);
  base_vec[0] = x_rot_based_traject[0] - orthogonal_val * orientvec_torus_t[0];
  base_vec[1] = x_rot_based_traject[1] - orthogonal_val * orientvec_torus_t[1];
  base_vec[2] = x_rot_based_traject[2] - orthogonal_val * orientvec_torus_t[2];

  double r_base =
      std::sqrt(base_vec[0] * base_vec[0] + base_vec[1] * base_vec[1] + base_vec[2] * base_vec[2]);

  // Do the slip length calculations!!!
  double sliplength = 0.0;
  if (slipfunct_ == 0)
  {
    return sliplength;
  }
  else if (slipfunct_ == 1)
  {
    // Set slip length 2.71.^((r-0.5)/0.01)  [10^(-9),10^(8)]
    double r_sphere = std::sqrt(x_based_trajectory[0] * x_based_trajectory[0] +
                                x_based_trajectory[1] * x_based_trajectory[1] +
                                x_based_trajectory[2] * x_based_trajectory[2]);

    sliplength = pow(2.7182818284590452353602874, 100.0 * (r_sphere - 0.5));
  }
  else if (slipfunct_ == 2)
  {
    // Set slip length 2.71.^(-(r-0.5)/0.01)  [10^(8),10^(-9)]
    double r_sphere = std::sqrt(x_based_trajectory[0] * x_based_trajectory[0] +
                                x_based_trajectory[1] * x_based_trajectory[1] +
                                x_based_trajectory[2] * x_based_trajectory[2]);

    sliplength = pow(2.7182818284590452353602874, -100.0 * (r_sphere - 0.5));
  }
  else if (slipfunct_ == 3)
  {
    // Set slip length 2.71.^((r_base-0.5)/0.01)  [10^(-9),10^(8)]

    sliplength = pow(2.7182818284590452353602874, 100.0 * (r_base - 0.5));
  }
  else if (slipfunct_ == 4)
  {
    // Set slip length 2.71.^(-(r_base-0.5)/0.01)  [10^(8),10^(-9)]

    sliplength = pow(2.7182818284590452353602874, -100.0 * (r_base - 0.5));
  }
  else
    FOUR_C_THROW(
        "Undefined choice of slip-length function for the Moving-Torus case. (So far, 0-4 are "
        "acceptable). And you should have been thrown out earlier than this!!!!");

  return sliplength;
}


Discret::Utils::TaylorCouetteFlow::TaylorCouetteFlow(double radius_inner, double radius_outer,
    double vel_theta_inner, double vel_theta_outer, double sliplength_inner,
    double sliplength_outer, double traction_theta_inner, double traction_theta_outer,
    double viscosity)
{
  radius_inner_ = radius_inner;
  radius_outer_ = radius_outer;

  // Assume for now always -> krylov projection!

  double alpha_i = -1.0;
  double alpha_o = 1.0;
  double mu =
      viscosity;  // For now let the viscosity be unity -> i.e. g_theta is scaled with viscosity!

  // Don't change direction of prescribed traction-vector. Should be
  double g_t1_scaled = mu * traction_theta_inner;  // alpha_i*mu*traction_theta_inner;
  double g_t2_scaled = mu * traction_theta_outer;  // alpha_o*mu*traction_theta_outer;

  double rhs_i = mu * vel_theta_inner + sliplength_inner * g_t1_scaled;
  double rhs_o = mu * vel_theta_outer + sliplength_outer * g_t2_scaled;

  double a_12 = alpha_i * sliplength_inner * mu * radius_inner *
                    (-2.0 * 1.0 / (radius_inner * radius_inner * radius_inner)) +
                mu / radius_inner;
  double a_22 = alpha_o * sliplength_outer * mu * radius_outer *
                    (-2.0 * 1.0 / (radius_outer * radius_outer * radius_outer)) +
                mu / radius_outer;
  double a_11 = mu * radius_inner;
  double a_21 = mu * radius_outer;

  // c2_ = ( rhs_o-rhs_i*a_21/a_11 ) / ( a_22-(a_21/a_11)*a_12 );
  // c1_ = ( rhs_i - a_12*c2_ ) / a_11;

  // Testing this way instead for a possible improvement (according to python this should be all
  // right!)
  double inv_determinant = 1.0 / (a_11 * a_22 - a_12 * a_21);
  c1_ = (rhs_i * a_22 - a_12 * rhs_o) * inv_determinant;
  c2_ = (rhs_o * a_11 - a_21 * rhs_i) * inv_determinant;

  // KRYLOV - int{\Omega} p d\Omega  = 0.0

  double tmp_int_r2 =
      (c1_ * c1_) * (radius_outer * radius_outer * radius_outer * radius_outer) / 8.0 +
      2.0 * c1_ * c2_ * (radius_outer * radius_outer) * (log(radius_outer) / 2.0 - 1.0 / 4.0) -
      (c2_ * c2_) / 2.0 * log(radius_outer);  //+C*r_2^2/2

  double tmp_int_r1 =
      (c1_ * c1_) * (radius_inner * radius_inner * radius_inner * radius_inner) / 8.0 +
      2.0 * c1_ * c2_ * (radius_inner * radius_inner) * (log(radius_inner) / 2.0 - 1.0 / 4.0) -
      (c2_ * c2_) / 2.0 * log(radius_inner);  //+C*r_1^2/2

  c3_ = (tmp_int_r1 - tmp_int_r2) /
        (0.5 * ((radius_outer * radius_outer) - (radius_inner * radius_inner)));
}

double Discret::Utils::TaylorCouetteFlow::evaluate(
    const double* xp, const double t, const std::size_t component) const
{
  double radius = std::sqrt(xp[0] * xp[0] + xp[1] * xp[1]);

  double u_theta = c1_ * radius + c2_ / radius;

  switch (component)
  {
    case 0:                              // u,x
      return -u_theta * xp[1] / radius;  // sin(theta);
    case 1:                              // u,y
      return u_theta * xp[0] / radius;   // cos(theta);
    case 2:                              // u,z
      return 0.0;
    case 3:  // v,x
      return (c1_ * c1_) * (radius * radius) * 0.5 + 2.0 * c1_ * c2_ * log(radius) -
             (c2_ * c2_) / (2.0 * (radius * radius)) + c3_;
    default:
      FOUR_C_THROW("wrong component %d", component);
      break;
  }


  return 1.0;
}

std::vector<double> Discret::Utils::TaylorCouetteFlow::evaluate_spatial_derivative(
    const double* xp, const double t, const std::size_t component) const
{
  // u_x = -(c1_*r + c2_/r)*y/r = -(c1_*y + c2_*y/(x^2+y^2))
  // d u_x /dx = c2_ * 2*x*y/((x^2+y^2)^2)
  // d u_x /dy = -c1_ - c2_ * (1/(x^2+y^2) - y*2*y/((x^2+y^2)^2))
  // d u_x /dz = 0.0

  // u_y =  (c1_*r + c2_/r)*x/r =  (c1_*x + c2_*x/(x^2+y^2))
  // d u_y /dx = c1_ + c2_ * (1/(x^2+y^2) - x*2*x/((x^2+y^2)^2))
  // d u_y /dy = - c2_ * 2*y*x/((x^2+y^2)^2)
  // d u_y /dz = 0.0

  // u_z = 0.0
  // d u_z / dx = d u_z / dy = d u_z / dz = 0.0

  // resulting vector holding
  std::vector<double> res(3, 0.0);
  res.reserve(3);

  double r_squared = (xp[0] * xp[0] + xp[1] * xp[1]);  //(x^2+y^2)
  double r_squared2 =
      (xp[0] * xp[0] + xp[1] * xp[1]) * (xp[0] * xp[0] + xp[1] * xp[1]);  //(x^2+y^2)^2

  switch (component)
  {
    case 0:
    {
      res[0] = c2_ * 2.0 * xp[0] * xp[1] / (r_squared2);
      res[1] = -c1_ - c2_ * (1 / r_squared - 2.0 * xp[1] * xp[1] / (r_squared2));
      res[2] = 0.0;
      break;
    }
    case 1:
    {
      res[0] = c1_ + c2_ * (1 / (r_squared)-2.0 * xp[0] * xp[0] / (r_squared2));
      res[1] = -c2_ * 2.0 * xp[1] * xp[0] / (r_squared2);
      res[2] = 0.0;
      break;
    }
    case 2:
    {
      res[0] = 0.0;
      res[1] = 0.0;
      res[2] = 0.0;
      break;
    }
    default:
      FOUR_C_THROW("wrong component %d", component);
      break;
  }

  return res;
}


Discret::Utils::UrquizaBoxFlow::UrquizaBoxFlow(double lengthx, double lengthy, double rotation,
    double viscosity, double density, int functno, std::vector<double> lincomb)
    : lengthx_(lengthx),
      lengthy_(lengthy),
      rotation_(rotation),
      rotvector_(2, std::vector<double>(2, 0.0)),
      kinvisc_(viscosity / density),
      functno_(functno),
      c1_(lincomb[0]),
      c2_(lincomb[1])
{
  // kinvisc_=viscosity/density;
  if (lengthx_ != 1.0) FOUR_C_THROW("Not tested for other than length 1.0.");

  if (lengthy_ != 1.0) FOUR_C_THROW("Not tested for other than length 1.0.");

  if (rotation_ >= 2 * M_PI or rotation_ < 0.0)
    FOUR_C_THROW("The rotation is not in the predefined interval.");

  if (rotation_ != 0.0)
    FOUR_C_THROW(
        "Needs to be implemented! It's not clear how to calculate traction at the boundaries if "
        "the domain is rotated.");

  if (functno_ == 1)
    std::cout << "Urquiza-box flow -> Quadratic flow" << std::endl;
  else if (functno_ == 2)
    std::cout << "4th order flow!" << std::endl;
  else if (functno_ == 3)  // Velocity: 2nd and 4th order, Press: 5th (x-dir) and 6th (y-dir) order
    std::cout << "Mixed 4th-2nd order flow velocity + 6th and 5th order pressure" << std::endl;
  else if (functno_ == 4)
    std::cout << "Urquiza-box flow modified -> Quadratic flow + 6 and 5th order pressure!"
              << std::endl;
  else if (functno_ == 5)
    std::cout << "Urquiza-box flow modified -> Mixed 4th-2nd order flow + Quadratic pressure!"
              << std::endl;
  else
    FOUR_C_THROW("Function number not supported!");

  // Create rotation matrix
  (rotvector_[0])[0] = cos(rotation_);
  (rotvector_[0])[1] = sin(rotation_);
  (rotvector_[1])[0] = -sin(rotation_);
  (rotvector_[1])[1] = cos(rotation_);
}

double Discret::Utils::UrquizaBoxFlow::evaluate(
    const double* xp, const double t, const std::size_t component) const
{
  // CASE 1:
  //  u =
  //  |    (   2    ) |
  //  |2*y*(- x  + 1) |
  //  |               |
  //  |     (   2    )|
  //  |-2*x*(- y  + 1)|

  // CASE 2:
  //  u =
  //  | 3 (   4    ) |
  //  |y *(- x  + 1) |
  //  |              |
  //  |  3 (   4    )|
  //  |-x *(- y  + 1)|
  //  p=
  //      (       5              3        ) (                6          4              2    )
  //      (2.025*x  - -4.5*(-1)*x  + 3.0*x)*(- -1.0125*(-1)*y  + 3.375*y  - -4.5*(-1)*y  + 1)


  //  Velocity field:
  //  |      3 (   4    )         (   2    ) |
  //  | c2_*y *(- x  + 1) + c1_*y*(- x  + 1) |
  //  |                                      |
  //  |       3 (   4    )         (   2    )|
  //  |- c2_*x *(- y  + 1) - c1_*x*(- y  + 1)|
  //  p=
  //      (       5              3        ) (                6          4              2    )
  //      (2.025*x  - -4.5*(-1)*x  + 3.0*x)*(- -1.0125*(-1)*y  + 3.375*y  - -4.5*(-1)*y  + 1)


  // Initialize
  std::vector<double> rotx(2, 0.0);
  // std::vector< std::vector<double> > rotvector(2,std::vector<double>(2,0.0));

  // Transform to reference configuration
  // x = R^t X
  if (rotation_ != 0.0)
  {
    rotx[0] = (rotvector_[0])[0] * xp[0] + (rotvector_[1])[0] * xp[1];
    rotx[1] = (rotvector_[0])[1] * xp[0] + (rotvector_[1])[1] * xp[1];
  }
  else
  {
    rotx[0] = xp[0];
    rotx[1] = xp[1];
  }

  // Calculate velocity in reference configuration
  std::vector<double> ux(2, 0.0);
  double px = 0.0;

  if (functno_ == 1)
  {
    ux[0] = 2 * rotx[1] * (-rotx[0] * rotx[0] + 1.0);
    ux[1] = -2 * rotx[0] * (-rotx[1] * rotx[1] + 1.0);
    px = 0.0;  //(rotx[0]-1.0)*(rotx[0]-1.0)*(rotx[1]+1.0)*(rotx[1]+1.0) - 64.0/36.0;
  }
  else if (functno_ == 2)
  {
    // double k=3.0;
    double x = rotx[0];
    double y = rotx[1];

    ux[0] = y * y * y * (-x * x * x * x + 1.0);
    ux[1] = -x * x * x * (-y * y * y * y + 1.0);
    px = 0.0;  //( (k*x) - (k*x)*(k*x)*(k*x)/(2.0*3.0) +
               //(k*x)*(k*x)*(k*x)*(k*x)*(k*x)/(2.0*3.0*4.0*5.0) )
               //*( 1.0 - (k*y)*(k*y)/2.0  + (k*y)*(k*y)*(k*y)*(k*y)/(2.0*3.0*4.0) -
               //(k*y)*(k*y)*(k*y)*(k*y)*(k*y)*(k*y)/(2.0*3.0*4.0*5.0*6.0) );
  }
  else if (functno_ == 3)
  {
    double k = 3.0;
    double x = rotx[0];
    double y = rotx[1];

    ux[0] = c2_ * y * y * y * (-x * x * x * x + 1.0) + c1_ * y * (-x * x + 1.0);
    ux[1] = -c2_ * x * x * x * (-y * y * y * y + 1.0) - c1_ * x * (-y * y + 1.0);
    px =
        ((k * x) - (k * x) * (k * x) * (k * x) / (2.0 * 3.0) +
            (k * x) * (k * x) * (k * x) * (k * x) * (k * x) / (2.0 * 3.0 * 4.0 * 5.0)) *
        (1.0 - (k * y) * (k * y) / 2.0 + (k * y) * (k * y) * (k * y) * (k * y) / (2.0 * 3.0 * 4.0) -
            (k * y) * (k * y) * (k * y) * (k * y) * (k * y) * (k * y) /
                (2.0 * 3.0 * 4.0 * 5.0 * 6.0));
  }
  else if (functno_ == 4)
  {
    double k = 3.0;
    double x = rotx[0];
    double y = rotx[1];

    ux[0] = 2 * rotx[1] * (-rotx[0] * rotx[0] + 1.0);
    ux[1] = -2 * rotx[0] * (-rotx[1] * rotx[1] + 1.0);
    px =
        ((k * x) - (k * x) * (k * x) * (k * x) / (2.0 * 3.0) +
            (k * x) * (k * x) * (k * x) * (k * x) * (k * x) / (2.0 * 3.0 * 4.0 * 5.0)) *
        (1.0 - (k * y) * (k * y) / 2.0 + (k * y) * (k * y) * (k * y) * (k * y) / (2.0 * 3.0 * 4.0) -
            (k * y) * (k * y) * (k * y) * (k * y) * (k * y) * (k * y) /
                (2.0 * 3.0 * 4.0 * 5.0 * 6.0));
    // px = (rotx[0]-1.0)*(rotx[0]-1.0)*(rotx[1]+1.0)*(rotx[1]+1.0) - 64.0/36.0;
  }
  else if (functno_ == 5)
  {
    // double k=3.0;
    double x = rotx[0];
    double y = rotx[1];

    ux[0] = c2_ * y * y * y * (-x * x * x * x + 1.0) + c1_ * y * (-x * x + 1.0);
    ux[1] = -c2_ * x * x * x * (-y * y * y * y + 1.0) - c1_ * x * (-y * y + 1.0);
    px = (rotx[0] - 1.0) * (rotx[0] - 1.0) * (rotx[1] + 1.0) * (rotx[1] + 1.0) - 64.0 / 36.0;
  }
  else
    FOUR_C_THROW("This error should have been caught in the constructor.");

  std::vector<double> uX(2, 0.0);
  // Transform the reference velocity back to the true domain.
  // U(x) = R u(x)
  if (rotation_ != 0.0)
  {
    uX[0] = (rotvector_[0])[0] * ux[0] + (rotvector_[0])[1] * ux[1];
    uX[1] = (rotvector_[1])[0] * ux[0] + (rotvector_[1])[1] * ux[1];
  }
  else
  {
    uX[0] = ux[0];
    uX[1] = ux[1];
  }

  // Rotate the traction back!
  switch (component)
  {
    case 0:  // u_x
      return uX[0];
    case 1:  // u_y
      return uX[1];
    case 2:  // u_z
      return 0.0;
    case 3:  // pressure
      return px;
    default:
      FOUR_C_THROW("wrong component %d", component);
      break;
  }

  // Should not end up here unless something went totally wrong!!
  return 0.0;
}

std::vector<double> Discret::Utils::UrquizaBoxFlow::evaluate_spatial_derivative(
    const double* xp, const double t, const std::size_t component) const
{
  //  CASE 1:
  //  du_i/dx_j =
  //  |               2    |
  //  | -4*x*y   - 2*x  + 2|
  //  |                    |
  //  |   2                |
  //  |2*y  - 2    4*x*y   |



  //  CASE 2:
  //  du_i/dx_j=
  //  |        3  3         2 (   4    )|
  //  |    -4*x *y       3*y *(- x  + 1)|
  //  |                                 |
  //  |    2 (   4    )         3  3    |
  //  |-3*x *(- y  + 1)      4*x *y     |


  // CASE 3:
  //  |                         3  3            (   2    )         2 (   4    )|
  //  |       -2*c_1*x*y - 4*c_2*x *y          c_1*(- x  + 1) + 3*c_2*y *(- x  + 1)|
  //  |                                                                        |
  //  |     (   2    )         2 (   4    )                         3  3       |
  //  |- c_1*(- y  + 1) - 3*c_2*x *(- y  + 1)        2*c_1*x*y + 4*c_2*x *y        |


  if (rotation_ != 0.0) FOUR_C_THROW("Needs to be implemented!");

  std::vector<double> tmp_vec(3, 0.0);

  switch (component)
  {
    case 0:
    {
      if (functno_ == 1 or functno_ == 4)
      {
        tmp_vec[0] = -4.0 * xp[0] * xp[1];
        tmp_vec[1] = -2.0 * xp[0] * xp[0] + 2.0;
      }
      else if (functno_ == 2)
      {
        tmp_vec[0] = -4.0 * xp[0] * xp[0] * xp[0] * xp[1] * xp[1] * xp[1];
        tmp_vec[1] = 3.0 * xp[1] * xp[1] * (-xp[0] * xp[0] * xp[0] * xp[0] + 1.0);
      }
      else if (functno_ == 3 or functno_ == 5)
      {
        tmp_vec[0] =
            -c1_ * 2.0 * xp[0] * xp[1] - c2_ * 4.0 * xp[0] * xp[0] * xp[0] * xp[1] * xp[1] * xp[1];
        tmp_vec[1] = c1_ * (-xp[0] * xp[0] + 1.0) +
                     c2_ * 3.0 * xp[1] * xp[1] * (-xp[0] * xp[0] * xp[0] * xp[0] + 1.0);
      }

      break;
    }
    case 1:
    {
      if (functno_ == 1 or functno_ == 4)
      {
        tmp_vec[0] = 2.0 * xp[1] * xp[1] - 2.0;
        tmp_vec[1] = 4.0 * xp[0] * xp[1];
      }
      else if (functno_ == 2)
      {
        tmp_vec[0] = -3.0 * xp[0] * xp[0] * (-xp[1] * xp[1] * xp[1] * xp[1] + 1.0);
        tmp_vec[1] = 4.0 * xp[0] * xp[0] * xp[0] * xp[1] * xp[1] * xp[1];
      }
      else if (functno_ == 3 or functno_ == 5)
      {
        tmp_vec[0] = c1_ * (xp[1] * xp[1] - 1.0) -
                     c2_ * 3.0 * xp[0] * xp[0] * (-xp[1] * xp[1] * xp[1] * xp[1] + 1.0);
        tmp_vec[1] =
            c1_ * 2.0 * xp[0] * xp[1] + c2_ * 4.0 * xp[0] * xp[0] * xp[0] * xp[1] * xp[1] * xp[1];
      }

      break;
    }
    case 2:
    {
      tmp_vec[0] = 0.0;
      tmp_vec[1] = 0.0;
      break;
    }
    default:
      FOUR_C_THROW("wrong component %d", component);
      break;
  }

  return tmp_vec;
}


Discret::Utils::UrquizaBoxFlowForce::UrquizaBoxFlowForce(double lengthx, double lengthy,
    double rotation, double viscosity, double density, int functno, std::vector<double> lincomb)
    : UrquizaBoxFlow(lengthx, lengthy, rotation, viscosity, density, functno, std::move(lincomb))
{
}

double Discret::Utils::UrquizaBoxFlowForce::evaluate(
    const double* xp, const double t, const std::size_t component) const
{
  double x = xp[0];
  double y = xp[1];


  // Body Force (non-simplified) CASE 3:
  // |    (                 2  3          (   4    ))     (     (   2    )       3 (   4    ))   (
  // (   2    )         2 (   4    )) (       (   2    )       3 (   4    )) |- q*(-2*c_1*y -
  // 12*c_2*x *y  + 6*c_2*y*(- x  + 1)) + s*(c_1*y*(- x  + 1) + c_2*y *(- x  + 1)) + (c_1*(- x  + 1)
  // + 3*c_2*y *(- x  + 1))*(- c_1*x*(- y  + 1) - c_2*x *(- y  + 1))

  // |
  // |      (                3  2          (   4    ))     (       (   2    )       3 (   4    )) (
  // (   2    )         2 (   4    )) (     (   2    )       3 (   4    )) |  - q*(2*c_1*x +
  // 12*c_2*x *y  - 6*c_2*x*(- y  + 1)) + s*(- c_1*x*(- y  + 1) - c_2*x *(- y  + 1)) + (- c_1*(- y
  // + 1) - 3*c_2*x *(- y  + 1))*(c_1*y*(- x  + 1) + c_2*y *(- x  + 1))
  //
  //   (                  3  3) (     (   2    )       3 (   4    ))   (       4               2 ) (
  //   6          4              2    )|
  // + (-2*c_1*x*y - 4*c_2*x *y )*(c_1*y*(- x  + 1) + c_2*y *(- x  + 1)) + (10.125*x  - -13.5*(-1)*x
  // + 3.0)*(- -1.0125*(-1)*y  + 3.375*y  - -4.5*(-1)*y  + 1)|
  //                                                                                  |
  //   (                 3  3) (       (   2    )       3 (   4    ))   (       5              3 ) (
  //   5         3              )   |
  // + (2*c_1*x*y + 4*c_2*x *y )*(- c_1*x*(- y  + 1) - c_2*x *(- y  + 1)) + (2.025*x  - -4.5*(-1)*x
  // + 3.0*x)*(- -6.075*(-1)*y  + 13.5*y  - -9.0*(-1)*y)   |


  // Rotate the traction back!
  switch (component)
  {
    case 0:  // (2*nu*D(u)*n)_x
    {
      if (functno_ == 1)
        return 4.0 * kinvisc_ * y - 2.0 * 0.0 * y * (x * x - 1.0) +
               8.0 * x * y * y * (x * x - 1.0) - 4.0 * x * (x * x - 1.0) * (y * y - 1.0);
      //+2.0*(x-1.0)*(y+1.0)*(y+1.0);
      else if (functno_ == 2)
      {
        return -kinvisc_ * (-12.0 * x * x * y * y * y + 6.0 * y * (-x * x * x * x + 1.0)) -
               0.0 * y * y * y * (x * x * x * x - 1.0) -
               4.0 * x * x * x * y * y * y * y * y * y * (-x * x * x * x + 1.0) -
               3.0 * x * x * x * y * y * (-x * x * x * x + 1.0) * (-y * y * y * y + 1.0);
        //+ (10.125*x*x*x*x - 13.5*x*x + 3.0)*(-1.0125*y*y*y*y*y*y + 3.375*y*y*y*y - 4.5*y*y + 1.0);
      }
      else if (functno_ == 3)
      {
        return -kinvisc_ * (-2.0 * c1_ * y - 12.0 * c2_ * x * x * y * y * y +
                               6.0 * c2_ * y * (-x * x * x * x + 1.0)) -
               0.0 * c2_ * y * y * y * (x * x * x * x - 1.0) - 0.0 * c1_ * y * (x * x - 1.0) +
               (c1_ * (-x * x + 1.0) + 3.0 * c2_ * y * y * (-x * x * x * x + 1.0)) *
                   (-c1_ * x * (-y * y + 1.0) - c2_ * x * x * x * (-y * y * y * y + 1.0)) +
               (-2.0 * c1_ * x * y - 4.0 * c2_ * x * x * x * y * y * y) *
                   (c1_ * y * (-x * x + 1.0) + c2_ * y * y * y * (-x * x * x * x + 1.0)) +
               (10.125 * x * x * x * x - 13.5 * x * x + 3.0) *
                   (-1.0125 * y * y * y * y * y * y + 3.375 * y * y * y * y - 4.5 * y * y + 1.0);
      }
      else if (functno_ == 4)
      {
        return 4.0 * kinvisc_ * y - 2.0 * 0.0 * y * (x * x - 1.0) +
               8.0 * x * y * y * (x * x - 1.0) - 4.0 * x * (x * x - 1.0) * (y * y - 1.0) +
               (10.125 * x * x * x * x - 13.5 * x * x + 3.0) *
                   (-1.0125 * y * y * y * y * y * y + 3.375 * y * y * y * y - 4.5 * y * y + 1.0);
      }
      else if (functno_ == 5)
      {
        return -kinvisc_ * (-2.0 * c1_ * y - 12.0 * c2_ * x * x * y * y * y +
                               6.0 * c2_ * y * (-x * x * x * x + 1.0)) -
               0.0 * c2_ * y * y * y * (x * x * x * x - 1.0) - 0.0 * c1_ * y * (x * x - 1.0) +
               (c1_ * (-x * x + 1.0) + 3.0 * c2_ * y * y * (-x * x * x * x + 1.0)) *
                   (-c1_ * x * (-y * y + 1.0) - c2_ * x * x * x * (-y * y * y * y + 1.0)) +
               (-2.0 * c1_ * x * y - 4.0 * c2_ * x * x * x * y * y * y) *
                   (c1_ * y * (-x * x + 1.0) + c2_ * y * y * y * (-x * x * x * x + 1.0)) +
               2.0 * (x - 1.0) * (y + 1.0) * (y + 1.0);
      }
      break;
    }
    case 1:  // (2*nu*D(u)*n)_y
    {
      if (functno_ == 1)
        return -4.0 * kinvisc_ * x + 2.0 * 0.0 * x * (y * y - 1.0) +
               8.0 * x * x * y * (y * y - 1.0) - 4.0 * y * (x * x - 1.0) * (y * y - 1.0);
      //+2.0*(x-1.0)*(x-1.0)*(y+1.0);
      else if (functno_ == 2)
      {
        // Body Force (non-simplified) CASE 2:
        // |    (      2  3       (   4    ))      3 (   4    )      3  6 (   4    )      3  2 (   4
        // ) (   4    )   (        4               2      ) (                6          4 2    )| |-
        // q*(- 12*x *y  + 6*y*(- x  + 1)) + s*y *(- x  + 1) - 4*x *y *(- x  + 1) - 3*x *y *(- x  +
        // 1)*(- y  + 1) + (10.125*x  - -13.5*(-1)*x  + 3.0)*(- -1.0125*(-1)*y  + 3.375*y  -
        // -4.5*(-1)*y  + 1)| | | |        (    3  2       (   4    ))      3 (   4    )      6  3 (
        // 4    )      2  3 (   4    ) (   4    )   (       5              3        ) ( 5         3
        // )     | |    - q*(12*x *y  - 6*x*(- y  + 1)) - s*x *(- y  + 1) - 4*x *y *(- y  + 1) - 3*x
        // *y *(- x  + 1)*(- y  + 1) + (2.025*x  - -4.5*(-1)*x  + 3.0*x)*(- -6.075*(-1)*y  + 13.5*y
        // - -9.0*(-1)*y)     |


        return -kinvisc_ * (12.0 * x * x * x * y * y - 6.0 * x * (-y * y * y * y + 1.0)) -
               0.0 * x * x * x * (y * y * y * y - 1.0) -
               4.0 * x * x * x * x * x * x * y * y * y * (-y * y * y * y + 1.0) -
               3.0 * x * x * y * y * y * (-x * x * x * x + 1.0) * (-y * y * y * y + 1.0);
        //+ (2.025*x*x*x*x*x - 4.5*x*x*x + 3.0*x)*(-6.075*y*y*y*y*y + 13.5*y*y*y - 9.0*y);
      }
      else if (functno_ == 3)
      {
        return -kinvisc_ * (2.0 * c1_ * x + 12.0 * c2_ * x * x * x * y * y -
                               6.0 * c2_ * x * (-y * y * y * y + 1.0)) +
               0.0 * c2_ * x * x * x * (y * y * y * y - 1.0) + 0.0 * c1_ * x * (y * y - 1.0) +
               (-c1_ * (-y * y + 1.0) - 3.0 * c2_ * x * x * (-y * y * y * y + 1.0)) *
                   (c1_ * y * (-x * x + 1.0) + c2_ * y * y * y * (-x * x * x * x + 1.0)) +
               (2.0 * c1_ * x * y + 4.0 * c2_ * x * x * x * y * y * y) *
                   (-c1_ * x * (-y * y + 1.0) - c2_ * x * x * x * (-y * y * y * y + 1.0)) +
               (2.025 * x * x * x * x * x - 4.5 * x * x * x + 3.0 * x) *
                   (-6.075 * y * y * y * y * y + 13.5 * y * y * y - 9.0 * y);
      }
      else if (functno_ == 4)
      {
        return -4.0 * kinvisc_ * x + 2.0 * 0.0 * x * (y * y - 1.0) +
               8.0 * x * x * y * (y * y - 1.0) - 4.0 * y * (x * x - 1.0) * (y * y - 1.0) +
               (2.025 * x * x * x * x * x - 4.5 * x * x * x + 3.0 * x) *
                   (-6.075 * y * y * y * y * y + 13.5 * y * y * y - 9.0 * y);
      }
      else if (functno_ == 5)
      {
        return -kinvisc_ * (2.0 * c1_ * x + 12.0 * c2_ * x * x * x * y * y -
                               6.0 * c2_ * x * (-y * y * y * y + 1.0)) +
               0.0 * c2_ * x * x * x * (y * y * y * y - 1.0) + 0.0 * c1_ * x * (y * y - 1.0) +
               (-c1_ * (-y * y + 1.0) - 3.0 * c2_ * x * x * (-y * y * y * y + 1.0)) *
                   (c1_ * y * (-x * x + 1.0) + c2_ * y * y * y * (-x * x * x * x + 1.0)) +
               (2.0 * c1_ * x * y + 4.0 * c2_ * x * x * x * y * y * y) *
                   (-c1_ * x * (-y * y + 1.0) - c2_ * x * x * x * (-y * y * y * y + 1.0)) +
               2.0 * (x - 1.0) * (x - 1.0) * (y + 1.0);
      }
      break;
    }
    case 2:  // z-dir = 0.0
      return 0.0;
    default:
      FOUR_C_THROW("wrong component %d", component);
      break;
  }

  return 0.0;
}


Discret::Utils::UrquizaBoxFlowTraction::UrquizaBoxFlowTraction(double lengthx, double lengthy,
    double rotation, double viscosity, double density, int functno, std::vector<double> lincomb)
    : UrquizaBoxFlow(lengthx, lengthy, rotation, viscosity, density, functno, std::move(lincomb))
{
}

double Discret::Utils::UrquizaBoxFlowTraction::evaluate(
    const double* xp, const double t, const std::size_t component) const
{
  double tol = 1e-13;

  bool onxsurface = false;
  bool onysurface = false;


  // Initialize
  std::vector<double> normal(2, 0.0);
  std::vector<double> rotx(2, 0.0);
  // std::vector< std::vector<double> > rotvector(2,std::vector<double>(2,0.0));

  // Rotation matrix
  if (rotation_ != 0.0)
  {
    rotx[0] = (rotvector_[0])[0] * xp[0] + (rotvector_[1])[0] * xp[1];
    rotx[1] = (rotvector_[0])[1] * xp[0] + (rotvector_[1])[1] * xp[1];
  }
  else
  {
    rotx[0] = xp[0];
    rotx[1] = xp[1];
  }

  // Is the point on x-surface
  double delx = fabs(fabs(rotx[0]) - lengthx_);
  if (delx < tol) onxsurface = true;

  // Is the point on y-surface
  double dely = fabs(fabs(rotx[1]) - lengthy_);
  if (dely < tol) onysurface = true;

  // Safety checks
  if (onxsurface and onysurface)
  {
    std::cout << "delx: " << delx << std::endl;
    std::cout << "dely: " << dely << std::endl;

    std::cout << "rotx[0]: " << rotx[0] << std::endl;
    std::cout << "rotx[1]: " << rotx[1] << std::endl;
    FOUR_C_THROW("Gauss point in a corner!");
  }
  if (!onxsurface and !onysurface)
  {
    std::cout << "delx: " << delx << std::endl;
    std::cout << "dely: " << dely << std::endl;

    std::cout << "rotx[0]: " << rotx[0] << std::endl;
    std::cout << "rotx[1]: " << rotx[1] << std::endl;
    FOUR_C_THROW("Gauss point not on any surface!");
  }


  // Create normal in reference coordinates
  if (onxsurface)
  {
    if (rotx[0] < 0.0)
      normal[0] = -1.0;
    else
      normal[0] = 1.0;
  }

  if (onysurface)
  {
    if (rotx[1] < 0.0)
      normal[1] = -1.0;
    else
      normal[1] = 1.0;
  }


  //  CASE 1:
  //  2*D(u) =
  //  |                    2      2|
  //  |   -8*x*y      - 2*x  + 2*y |
  //  |                            |
  //  |     2      2               |
  //  |- 2*x  + 2*y       8*x*y    |


  //  CASE 2:
  //  2*D(u) =
  //  |                 3  3                     2 (   4    )      2 (   4    )|
  //  |             -8*x *y                 - 3*x *(- y  + 1) + 3*y *(- x  + 1)|
  //  |                                                                        |
  //  |     2 (   4    )      2 (   4    )                   3  3              |
  //  |- 3*x *(- y  + 1) + 3*y *(- x  + 1)                8*x *y               |

  //  CASE 3:
  //  2*D(u) =
  //  |                                          3  3                              (   2    )      (
  //  2    )         2 (   4    )         2 (   4    )| |                        -4*c_1*x*y -
  //  8*c_2*x *y                            c_1*(- x  + 1) - c_1*(- y  + 1) - 3*c_2*x *(- y  + 1) +
  //  3*c_2*y *(- x  + 1)| | | |   (   2    )      (   2    )         2 (   4    )         2 (   4
  //  ) 3  3 | |c_1*(- x  + 1) - c_1*(- y  + 1) - 3*c_2*x *(- y  + 1) + 3*c_2*y *(- x  + 1)
  //  4*c_1*x*y + 8*c_2*x *y |

  std::vector<std::vector<double>> straintens(2, std::vector<double>(2, 0.0));


  if (functno_ == 1 or functno_ == 4)
  {
    (straintens[0])[0] = -8.0 * rotx[0] * rotx[1];
    (straintens[0])[1] = -2.0 * rotx[0] * rotx[0] + 2.0 * rotx[1] * rotx[1];
    (straintens[1])[0] = -2.0 * rotx[0] * rotx[0] + 2.0 * rotx[1] * rotx[1];
    (straintens[1])[1] = 8.0 * rotx[0] * rotx[1];
  }
  else if (functno_ == 2)
  {
    (straintens[0])[0] = -8.0 * rotx[0] * rotx[0] * rotx[0] * rotx[1] * rotx[1] * rotx[1];
    (straintens[0])[1] = -3.0 * rotx[0] * rotx[0] * (-rotx[1] * rotx[1] * rotx[1] * rotx[1] + 1.0) +
                         3.0 * rotx[1] * rotx[1] * (-rotx[0] * rotx[0] * rotx[0] * rotx[0] + 1.0);
    (straintens[1])[0] = -3.0 * rotx[0] * rotx[0] * (-rotx[1] * rotx[1] * rotx[1] * rotx[1] + 1.0) +
                         3.0 * rotx[1] * rotx[1] * (-rotx[0] * rotx[0] * rotx[0] * rotx[0] + 1.0);
    (straintens[1])[1] = 8.0 * rotx[0] * rotx[0] * rotx[0] * rotx[1] * rotx[1] * rotx[1];
  }
  else if (functno_ == 3 or functno_ == 5)
  {
    (straintens[0])[0] = -c1_ * 4.0 * rotx[0] * rotx[1] -
                         c2_ * 8.0 * rotx[0] * rotx[0] * rotx[0] * rotx[1] * rotx[1] * rotx[1];
    (straintens[0])[1] =
        c1_ * (-rotx[0] * rotx[0] + 1.0) - c1_ * (-rotx[1] * rotx[1] + 1.0) -
        c2_ * 3.0 * rotx[0] * rotx[0] * (-rotx[1] * rotx[1] * rotx[1] * rotx[1] + 1.0) +
        c2_ * 3.0 * rotx[1] * rotx[1] * (-rotx[0] * rotx[0] * rotx[0] * rotx[0] + 1.0);
    (straintens[1])[0] =
        c1_ * (-rotx[0] * rotx[0] + 1.0) - c1_ * (-rotx[1] * rotx[1] + 1.0) -
        c2_ * 3.0 * rotx[0] * rotx[0] * (-rotx[1] * rotx[1] * rotx[1] * rotx[1] + 1.0) +
        c2_ * 3.0 * rotx[1] * rotx[1] * (-rotx[0] * rotx[0] * rotx[0] * rotx[0] + 1.0);
    (straintens[1])[1] = c1_ * 4.0 * rotx[0] * rotx[1] +
                         c2_ * 8.0 * rotx[0] * rotx[0] * rotx[0] * rotx[1] * rotx[1] * rotx[1];
  }
  else
    FOUR_C_THROW("This error should have been caught in the constructor.");


  // Surface viscous traction
  std::vector<double> rotsurfvisctrac(2, 0.0);
  rotsurfvisctrac[0] = (straintens[0])[0] * normal[0] + (straintens[0])[1] * normal[1];
  rotsurfvisctrac[1] = (straintens[1])[0] * normal[0] + (straintens[1])[1] * normal[1];


  std::vector<double> surfvisctrac(2, 0.0);
  // Rotation matrix
  if (rotation_ != 0.0)
  {
    surfvisctrac[0] = (rotvector_[0])[0] * surfvisctrac[0] + (rotvector_[0])[1] * surfvisctrac[1];
    surfvisctrac[1] = (rotvector_[1])[0] * surfvisctrac[0] + (rotvector_[1])[1] * surfvisctrac[1];
  }
  else
  {
    surfvisctrac[0] = rotsurfvisctrac[0];
    surfvisctrac[1] = rotsurfvisctrac[1];
  }


  // Rotate the traction back!
  switch (component)
  {
    case 0:  // (2*nu*D(u)*n)_x
      return kinvisc_ * surfvisctrac[0];
    case 1:  // (2*nu*D(u)*n)_y
      return kinvisc_ * surfvisctrac[1];
    case 2:  // z-dir = 0.0
      return 0.0;
    default:
      FOUR_C_THROW("wrong component %d", component);
      break;
  }

  return 0.0;
}

FOUR_C_NAMESPACE_CLOSE
