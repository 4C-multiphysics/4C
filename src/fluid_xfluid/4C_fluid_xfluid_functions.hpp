// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_utils_function.hpp"

#include <optional>

#ifndef FOUR_C_FLUID_XFLUID_FUNCTIONS_HPP
#define FOUR_C_FLUID_XFLUID_FUNCTIONS_HPP

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{
  class FunctionManager;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Utils
  {
    /// add valid xfluid-specific function lines
    void add_valid_xfluid_functions(Core::Utils::FunctionManager& function_manager);

    /// special implementation level set cut utilizing xfluid
    class GerstenbergerForwardfacingStep : public Core::Utils::FunctionOfSpaceTime
    {
     public:
      /// ctor
      GerstenbergerForwardfacingStep();

      double evaluate(const double* x, double t, std::size_t component) const override;

      [[nodiscard]] std::size_t number_components() const override
      {
        FOUR_C_THROW("Number of components not defined for GerstenbergerForwardfacingStep.");
      };
    };

    /*!
     * @brief Moves a 2D cylinder back and forth along a predetermined direction in a sinus like
     * fashion
     *
     * x_pos = midpoint_trajectory_ + direction_*(L/2)*sin(sin_coeff*t-PI*0.5)
     * The cylinder reaches its turning point distance
     *   at the time t_dist = n*PI/2   n=0,1,2,3,...
     *   and maximum velocity at t_max  = n*PI/4   n=1,2,3,4,...
     */
    class MovingLevelSetCylinder : public Core::Utils::FunctionOfSpaceTime
    {
     public:
      /// ctor
      MovingLevelSetCylinder(std::vector<double>* origin, double radius,
          std::vector<double>* direction, double distance, double maxspeed);

      double evaluate(const double* x, double t, std::size_t component) const override;

      [[nodiscard]] std::size_t number_components() const override
      {
        FOUR_C_THROW("Number of components not defined for MovingLevelSetCylinder.");
      };

     private:
      /// Origin of the geometry
      std::vector<double> origin_;

      /// Radius
      double radius_;

      /// Orientation of the geometry (symmetry axis)
      std::vector<double> direction_;

      /// Midpoint of trajectory
      std::vector<double> midpoint_trajectory_;

      /// Distance traveled
      double distance_;

      /// Maximum speed
      double maxspeed_;
    };

    /*!
     * @brief Moves a 3D torus back and forth along a predetermined direction in a sine like fashion
     * with constant rotation
     *
     * TORUS: The torus is oriented in the orientationvec_torus direction and its midpoint is at
     * origin (for t=0). The radius of the torus is radius, where as the radius for the tube is
     * radius_tube. TRANSLATION: x_pos = midpoint_trajectory_ +
     * direction_*(L/2)*sin(sin_coeff*t-PI*0.5)
     *
     * The cylinder reaches its turning point distance
     *   at the time t_dist = n*PI/2   n=0,1,2,3,...
     *   and maximum velocity at t_max  = n*PI/4   n=1,2,3,4,...
     *
     * ROTATION: The torus rotates around a given rotation vector rotvector, at the constant
     * rotation speed rotspeed
     *
     * R(theta) = cos(theta) I + sin(theta) * [w]_x +(1-cos(theta))*w(x)w
     * theta = rotspeed*t
     * x_rot_pos = R(j,i)*( x(j) - x_pos(j) )   (MAYBE TRANSPOSE R(j,i)!!!!)
     *
     * LEVEL SET:
     *   x_orto = R(i,j) * ( x(j) - x_pos(j) ) * orientationvec_torus(i)
     *   x_base = x_rot_pos(i) - x_orto * orientationvec_torus(i)
     *   phi(theta,t) = sqrt( ( radius - sqrt( x_base*x_base ) )^2 + x_orto*x_orto ) - radius_tube
     *
     * VELOCITY:
     *   TRANSLATION: v_trans(i) = direction_(i)*sin_coeff(L/2)*cos(sin_coeff*t-PI*0.5)
     *   ROTATION: v_rot(i) = rotspeed * [w]_x (i,j) * (x(j) - x_pos(j))   %(or should it be
     * x_rot_pos? TOTAL: v_tot = v_trans + v_rot
     *
     * SLIPLENGTH:
     *   0)
     *   1) Increasing sliplength in shape of sphere.
     */
    class MovingLSTorus : public Core::Utils::FunctionOfSpaceTime
    {
     public:
      /// ctor
      MovingLSTorus(std::vector<double>* origin, std::vector<double>* orientationvec_torus,
          double radius, double radius_tube, std::vector<double>* direction, double distance,
          double maxspeed, std::vector<double>* rotvector, double rotspeed, double rotramptime);

      [[nodiscard]] std::size_t number_components() const override
      {
        FOUR_C_THROW("Number of components not defined for MovingLSTorus.");
      };

     protected:
      /// Origin of the geometry
      std::vector<double> origin_;

      /// Orientation vector
      std::vector<double> orientationvec_torus_;

      /// Radius
      double radius_;

      /// Radius of torus tube
      double radius_tube_;

      /// Orientation of the geometry (symmetry axis)
      std::vector<double> direction_;

      /// Midpoint of trajectory
      std::vector<double> midpoint_trajectory_;

      /// Distance traveled
      double distance_;

      /// Maximum speed
      double maxspeed_;

      /// Rotation vector
      std::vector<double> rotvector_;

      /// Rotation speed (revolutions per second)
      double rotspeed_;

      /// Ramp time for the rotation ( ramping done as: 0.5*cos(PI*t/T_ramp) )
      double ramptime_;

      /// MATRICES!!
      std::vector<std::vector<double>> eye_;
      std::vector<std::vector<double>> rot_joint_;
      std::vector<std::vector<double>> rot_cross_;
    };

    /*!
     * @brief Moving and rotating 3D Torus, returns level set value
     */
    class MovingLevelSetTorus : public MovingLSTorus
    {
     public:
      /// ctor
      MovingLevelSetTorus(std::vector<double>* origin, std::vector<double>* orientationvec_torus,
          double radius, double radius_tube, std::vector<double>* direction, double distance,
          double maxspeed, std::vector<double>* rotvector, double rotspeed, double rotramptime);

      /// evaluate function at given position in space
      double evaluate(const double* x, double t, std::size_t component) const override;
    };

    class MovingLevelSetTorusVelocity : public MovingLSTorus
    {
     public:
      /// ctor
      MovingLevelSetTorusVelocity(std::vector<double>* origin,
          std::vector<double>* orientationvec_torus, double radius, double radius_tube,
          std::vector<double>* direction, double distance, double maxspeed,
          std::vector<double>* rotvector, double rotspeed, double rotramptime);

      /// evaluate function at given position in space
      double evaluate(const double* x, double t, std::size_t component) const override;

      /*!
       * @brief Return the number of components of this spatial function (This is a vector-valued
       * function)
       *
       * \return number of components (u,v,w)
       */
      [[nodiscard]] std::size_t number_components() const override { return (3); };
    };

    class MovingLevelSetTorusSliplength : public MovingLSTorus
    {
     public:
      /// ctor
      MovingLevelSetTorusSliplength(std::vector<double>* origin,
          std::vector<double>* orientationvec_torus, double radius, double radius_tube,
          std::vector<double>* direction, double distance, double maxspeed,
          std::vector<double>* rotvector, double rotspeed, double rotramptime,
          std::optional<int> slipfunct);

      /// evaluate function at given position in space
      double evaluate(const double* x, double t, std::size_t component) const override;

     private:
      std::optional<int> slipfunct_;
    };

    /*!
     * @brief Stationary Taylor-Couette flow with Navier-Slip type boundary condition at inner
     * cylinder.
     */
    class TaylorCouetteFlow : public Core::Utils::FunctionOfSpaceTime
    {
     public:
      /// ctor (for NavSlip at both boundaries)
      TaylorCouetteFlow(double radius_inner, double radius_outer, double vel_theta_inner,
          double vel_theta_outer, double sliplength_inner, double sliplength_outer,
          double traction_theta_inner, double traction_theta_outer, double viscosity);

      /// evaluate function at given position in space,
      /// here: evaluation of Taylor-Couette analytical solution
      double evaluate(const double* x, double t, std::size_t component) const override;

      std::vector<double> evaluate_spatial_derivative(
          const double* x, double t, std::size_t component) const override;

      [[nodiscard]] std::size_t number_components() const override
      {
        FOUR_C_THROW("Number of components not defined for TaylorCouetteFlow.");
      };

     private:
      /// Radi
      double radius_inner_;
      double radius_outer_;

      double c1_;
      double c2_;
      double c3_;
    };

    /*!
     * @brief Urquiza-box flow
     *
     * Oseen test case (can be solved with Nav-Stokes as well as the advective velocity.
     * is chosen as the analytic solution to the Nav-Stokes equations).
     */
    class UrquizaBoxFlow : public Core::Utils::FunctionOfSpaceTime
    {
     public:
      /// ctor
      UrquizaBoxFlow(double lengthx, double lengthy, double rotation, double viscosity,
          double density, int functno, std::vector<double> lincomb);

      double evaluate(const double* x, double t, std::size_t component) const override;

      std::vector<double> evaluate_spatial_derivative(
          const double* x, double t, std::size_t component) const override;

      [[nodiscard]] std::size_t number_components() const override
      {
        FOUR_C_THROW("Number of components not defined for UrquizaBoxFlow.");
      };

     protected:
      double lengthx_;   /// Length lx -> [-lx,lx]
      double lengthy_;   /// Length ly ->[-ly,ly]
      double rotation_;  /// rot    [0,2*pi]
      std::vector<std::vector<double>> rotvector_;
      double kinvisc_;  /// kinvisc = visc/rho
      int functno_;     /// function number

      double c1_;
      double c2_;
    };

    /*!
     * @brief Urquiza-box test case
     */
    class UrquizaBoxFlowForce : public UrquizaBoxFlow
    {
     public:
      /// ctor
      UrquizaBoxFlowForce(double lengthx, double lengthy, double rotation, double viscosity,
          double density, int functno, std::vector<double> lincomb);

      /// evaluate function at given position in space
      double evaluate(const double* x, double t, std::size_t component) const override;

      std::vector<double> evaluate_spatial_derivative(
          const double* x, const double t, const std::size_t component) const override
      {
        FOUR_C_THROW("Derivative not implemented for UrquizaBoxFlowForce");
        return {};
      }
    };

    /*!
     * @brief UrquizaBoxFlowTraction
     */
    class UrquizaBoxFlowTraction : public UrquizaBoxFlow
    {
     public:
      /// ctor
      UrquizaBoxFlowTraction(double lengthx, double lengthy, double rotation, double viscosity,
          double density, int functno, std::vector<double> lincomb);

      /// evaluate function at given position in space
      double evaluate(const double* x, double t, std::size_t component) const override;

      std::vector<double> evaluate_spatial_derivative(
          const double* x, const double t, const std::size_t component) const override
      {
        FOUR_C_THROW("Derivative not implemented for UrquizaBoxFlowTraction");
        return {};
      }
    };
  }  // namespace Utils
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
