// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MYOCARD_MINIMAL_HPP
#define FOUR_C_MAT_MYOCARD_MINIMAL_HPP

/*----------------------------------------------------------------------*
 |  headers                                                  cbert 08/13 |
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
/// <h3>References</h3>
/// <ul>
/// <li> [1] A Bueno-Orovio et. al., "Minimal model for human ventricular action potentials in
/// tissue", Journal of Theoretical Biology 253 (2008) 544-560
/// </ul>
///



class MyocardMinimal : public MyocardGeneral

{
 public:
  /// construct empty material object
  MyocardMinimal();

  /// construct empty material object
  explicit MyocardMinimal(const double eps_deriv_myocard, const std::string tissue, int num_gp);

  /// compute reaction coefficient
  double rea_coeff(const double phi, const double dt) override;

  /// compute reaction coefficient for multiple points per element
  double rea_coeff(const double phi, const double dt, int gp) override;

  /// compute reaction coefficient for multiple points per element at timestep n
  double rea_coeff_n(const double phi, const double dt, int gp) override;

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

  /// resize internal state variables if number of Gauss point changes
  void resize_internal_state_variables(int gp) override;

  /// get number of Gauss points
  int get_number_of_gp() const override;


 private:
  MyocardTools tools_;

  /// perturbation for numerical approximation of the derivative
  double eps_deriv_;

  /// last gating variables MV
  std::vector<double> v0_;  /// fast inward current
  std::vector<double> w0_;  /// slow outward current
  std::vector<double> s0_;  /// slow inward current

  /// current gating variables MV
  std::vector<double> v_;  /// fast inward current
  std::vector<double> w_;  /// slow outward current
  std::vector<double> s_;  /// slow inward current

  /// ionic currents MV
  std::vector<double> jfi_;  /// fast inward current
  std::vector<double> jso_;  /// slow outward current
  std::vector<double> jsi_;  /// slow inward current

  /// model parameters
  double u_o_;       // = 0.0;
  double u_u_;       // = 1.55;//1.58;
  double theta_v_;   // = 0.3;
  double theta_w_;   // = 0.13;//0.015;
  double theta_vm_;  // = 0.006;//0.015;
  double theta_o_;   // = 0.006;
  double tau_v1m_;   // = 60.0;
  double tau_v2m_;   // = 1150.0;
  double tau_vp_;    // = 1.4506;
  double tau_w1m_;   // = 60.0;//70.0;
  double tau_w2m_;   // = 15.0;//20.0;
  double k_wm_;      // = 65.0;
  double u_wm_;      // = 0.03;
  double tau_wp_;    // = 200.0;//280.0;
  double tau_fi_;    // = 0.11;
  double tau_o1_;    // = 400.0;//6.0;
  double tau_o2_;    // = 6.0;
  double tau_so1_;   // = 30.0181;//43.0;
  double tau_so2_;   // = 0.9957;//0.2;
  double k_so_;      // = 2.0458;//2.0;
  double u_so_;      // = 0.65;
  double tau_s1_;    // = 2.7342;
  double tau_s2_;    // = 16.0;//3.0;
  double k_s_;       // = 2.0994;
  double u_s_;       // = 0.9087;
  double tau_si_;    // = 1.8875;//2.8723;
  double tau_winf_;  // = 0.07;
  double w_infs_;    // = 0.94;

  // Variables for electromechanical coupling
  double mechanical_activation_;  // to store the variable for activation (phi in this case=)

};  // Myocard_Minimal


FOUR_C_NAMESPACE_CLOSE

#endif
