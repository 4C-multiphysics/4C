// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MYOCARD_FITZHUGH_NAGUMO_HPP
#define FOUR_C_MAT_MYOCARD_FITZHUGH_NAGUMO_HPP

/*----------------------------------------------------------------------*
 |  headers                                                  cbert 09/13 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_myocard_general.hpp"
#include "4C_mat_myocard_tools.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/// Myocard material according to [1]
///
/// This is a reaction-diffusion law of anisotropic, instationary electric conductivity in cardiac
/// muscle tissue
///
/// d(r)/dt =  b*d*(\phi/d-r)
/// d(\phi)/dt + c1*\phi*(\phi-a)*(\phi-1.0) + c2*\phi*r_ = 0
///
/// with \phi the potential variable and r the internal state
/// </ul>
///



class MyocardFitzhughNagumo : public MyocardGeneral

{
 public:
  /// construct empty material object
  MyocardFitzhughNagumo();

  /// construct empty material object
  explicit MyocardFitzhughNagumo(
      const double eps_deriv_myocard, const std::string tissue, int num_gp);

  /// compute reaction coefficient
  double rea_coeff(const double phi, const double dt) override;

  /// compute reaction coefficient for multiple points per element
  double rea_coeff(const double phi, const double dt, int gp) override;

  ///  returns number of internal state variables of the material
  int get_number_of_internal_state_variables() const override;

  ///  returns current internal state of the material
  double get_internal_state(const int k) const override;

  ///  returns current internal state of the material for multiple points per element
  double get_internal_state(const int k, int gp) const override;

  ///  set internal state of the material
  void set_internal_state(const int k, const double val) override;

  ///  set internal state of the material for multiple points per element
  void set_internal_state(const int k, const double val, int gp) override;

  ///  return number of ionic currents
  int get_number_of_ionic_currents() const override;

  ///  return ionic currents
  double get_ionic_currents(const int k) const override;

  ///  return ionic currents for multiple points per element
  double get_ionic_currents(const int k, int gp) const override;

  /// time update for this material
  void update(const double phi, const double dt) override;

  /// get number of Gauss points
  int get_number_of_gp() const override { return r0_.size(); };

  /// resize internal state variables if number of Gauss point changes
  void resize_internal_state_variables(int gp) override
  {
    r0_.resize(gp);
    r_.resize(gp);
    j1_.resize(gp);
    j2_.resize(gp);
    mechanical_activation_.resize(gp);
  }

 private:
  MyocardTools tools_;

  /// perturbation for numerical approximation of the derivative
  double eps_deriv_;

  /// last gating variables MV
  std::vector<double> r0_;  /// fast inward current

  /// current gating variables MV
  std::vector<double> r_;  /// fast inward current

  /// ionic currents
  std::vector<double> j1_;
  std::vector<double> j2_;

  /// model parameters
  double a_;
  double b_;
  double c1_;
  double c2_;
  double d_;

  // Variables for electromechanical coupling
  std::vector<double>
      mechanical_activation_;  // to store the variable for activation (phi in this case=)
  double act_thres_;  // activation threshold (so that activation = 1.0 if mechanical_activation_ >=
                      // act_thres_)


};  // Myocard_Fitzhugh_Nagumo


FOUR_C_NAMESPACE_CLOSE

#endif
