// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_myocard_minimal.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Constructor                                    (public)  cbert 08/13 |
 *----------------------------------------------------------------------*/
MyocardMinimal::MyocardMinimal() {}


/*----------------------------------------------------------------------*
 |  Constructor                                    (public)  cbert 08/13 |
 *----------------------------------------------------------------------*/
MyocardMinimal::MyocardMinimal(const double eps_deriv_myocard, const std::string tissue, int num_gp)
    : tools_(),
      v0_(num_gp),
      w0_(num_gp),
      s0_(num_gp),
      v_(num_gp),
      w_(num_gp),
      s_(num_gp),
      jfi_(num_gp),
      jso_(num_gp),
      jsi_(num_gp)

{
  for (int i = 0; i < num_gp; ++i)
  {
    // Initial condition
    v0_[i] = 1.0;
    w0_[i] = 1.0;
    s0_[i] = 0.0;
    v_[i] = v0_[i];
    w_[i] = w0_[i];
    s_[i] = s0_[i];
    jfi_[i] = 0.0;  /// fast inward current
    jso_[i] = 0.0;  /// slow outward current
    jsi_[i] = 0.0;  /// slow inward current
  }


  eps_deriv_ = eps_deriv_myocard;
  // Model parameter (To pack later in the parameter list of inputfile!)
  if (tissue == "M")
  {
    u_o_ = 0.0;
    u_u_ = 1.61;
    theta_v_ = 0.3;
    theta_w_ = 0.13;
    theta_vm_ = 0.1;
    theta_o_ = 0.005;
    tau_v1m_ = 80.0;
    tau_v2m_ = 1.4506;
    tau_vp_ = 1.4506;
    tau_w1m_ = 70.0;
    tau_w2m_ = 8.0;
    k_wm_ = 200.0;
    u_wm_ = 0.016;
    tau_wp_ = 280.0;
    tau_fi_ = 0.078;
    tau_o1_ = 410.0;
    tau_o2_ = 7.0;
    tau_so1_ = 91;
    tau_so2_ = 0.8;
    k_so_ = 2.1;
    u_so_ = 0.6;
    tau_s1_ = 2.7342;
    tau_s2_ = 4.0;
    k_s_ = 2.0994;
    u_s_ = 0.9087;
    tau_si_ = 3.3849;
    tau_winf_ = 0.01;
    w_infs_ = 0.5;
  }
  else if (tissue == "ENDO")
  {
    u_o_ = 0.0;
    u_u_ = 1.56;
    theta_v_ = 0.3;
    theta_w_ = 0.13;
    theta_vm_ = 0.2;
    theta_o_ = 0.006;
    tau_v1m_ = 75.0;
    tau_v2m_ = 10.0;
    tau_vp_ = 1.4506;
    tau_w1m_ = 6.0;
    tau_w2m_ = 140.0;
    k_wm_ = 200.0;
    u_wm_ = 0.016;
    tau_wp_ = 280.0;
    tau_fi_ = 0.1;
    tau_o1_ = 470.0;
    tau_o2_ = 6.0;
    tau_so1_ = 40.0;
    tau_so2_ = 1.2;
    k_so_ = 2.0;
    u_so_ = 0.65;
    tau_s1_ = 2.7342;
    tau_s2_ = 2.0;
    k_s_ = 2.0994;
    u_s_ = 0.9087;
    tau_si_ = 2.9013;
    tau_winf_ = 0.0273;
    w_infs_ = 0.78;
  }
  else if (tissue == "EPI")
  {
    u_o_ = 0.0;
    u_u_ = 1.55;
    theta_v_ = 0.3;
    theta_w_ = 0.13;
    theta_vm_ = 0.006;
    theta_o_ = 0.006;
    tau_v1m_ = 60.0;
    tau_v2m_ = 1150;
    tau_vp_ = 1.4506;
    tau_w1m_ = 60.0;
    tau_w2m_ = 15.0;
    k_wm_ = 65.0;
    u_wm_ = 0.03;
    tau_wp_ = 200.0;
    tau_fi_ = 0.11;
    tau_o1_ = 400.0;
    tau_o2_ = 6.0;
    tau_so1_ = 30.0181;
    tau_so2_ = 0.9957;
    k_so_ = 2.0458;
    u_so_ = 0.65;
    tau_s1_ = 2.7342;
    tau_s2_ = 16.0;
    k_s_ = 2.0994;
    u_s_ = 0.9087;
    tau_si_ = 1.8875;
    tau_winf_ = 0.07;
    w_infs_ = 0.94;
  }
  else if (tissue == "Atria")
  {
    u_o_ = 0.0;
    u_u_ = 1.02;  // 1.58;
    theta_v_ = 0.302;
    theta_w_ = 0.33;    // 0.015;
    theta_vm_ = 0.172;  // 0.015;
    theta_o_ = 0.06;
    tau_v1m_ = 65.6;
    tau_v2m_ = 1150.0;
    tau_vp_ = 0.95;
    tau_w1m_ = 170.8;  // 70.0;
    tau_w2m_ = 112.4;  // 20.0;
    k_wm_ = 135.0;
    u_wm_ = 0.0744;
    tau_wp_ = 217.0;  // 280.0;
    tau_fi_ = 0.0678;
    tau_o1_ = 100.0;  // 6.0;
    tau_o2_ = 64.87;
    tau_so1_ = 53.54;  // 43.0;
    tau_so2_ = 8.03;   // 0.2;
    k_so_ = 1.748;     // 2.0;
    u_so_ = 0.644;
    tau_s1_ = 5.406;
    tau_s2_ = 52.91;  // 3.0;
    k_s_ = 1.008;
    u_s_ = 0.814;
    tau_si_ = 6.978;  // 2.8723;
    tau_winf_ = 4.97;
    w_infs_ = 1.0;
  }
  else if (tissue == "pAtria")  // Lenk et al. 2015, "Initiation of atrial fibrillation..." (JTB)
  {
    u_o_ = 0.0;
    u_u_ = 0.9205;
    theta_v_ = 0.35;
    theta_w_ = 0.328;
    theta_vm_ = 0.126;
    theta_o_ = 0.00005;
    tau_v1m_ = 41.857;
    tau_v2m_ = 1150;
    tau_vp_ = 1.7;
    tau_w1m_ = 138.69;
    tau_w2m_ = 62.341;
    k_wm_ = 202.66;
    u_wm_ = 0.055;
    tau_wp_ = 177.41;
    tau_fi_ = 0.045;
    tau_o1_ = 410;
    tau_o2_ = 64.914;
    tau_so1_ = 115;
    tau_so2_ = 6.5;
    k_so_ = 1.386;
    u_so_ = 0.332;
    tau_s1_ = 11.457;
    tau_s2_ = 53.902;
    k_s_ = 1.226;
    u_s_ = 0.792;
    tau_si_ = 7.802;
    tau_winf_ = 0.05;
    w_infs_ = 1.0;
  }
  else if (tissue == "pAtriaRe")  // Lenk et al. 2015, "Initiation of atrial fibrillation..." (JTB)
  {
    u_o_ = 0.0;
    u_u_ = 0.9205;
    theta_v_ = 0.35;
    theta_w_ = 0.328;
    theta_vm_ = 0.126;
    theta_o_ = 0.00005;
    tau_v1m_ = 41.857;
    tau_v2m_ = 1150;
    tau_vp_ = 1.7;
    tau_w1m_ = 138.69;
    tau_w2m_ = 62.341;
    k_wm_ = 202.66;
    u_wm_ = 0.055;
    tau_wp_ = 177.41;
    tau_fi_ = 0.11;
    tau_o1_ = 410;
    tau_o2_ = 64.914;
    tau_so1_ = 115;
    tau_so2_ = 6.5;
    k_so_ = 1.386;
    u_so_ = 0.332;
    tau_s1_ = 11.457;
    tau_s2_ = 53.902;
    k_s_ = 1.226;
    u_s_ = 0.792;
    tau_si_ = 7.802;
    tau_winf_ = 0.05;
    w_infs_ = 1.0;
  }
  else if (tissue == "rAtria")
  // Lenk et al. 2015, "Initiation of atrial fibrillation..." (JTB);
  // Richter et al. 2016, "Anatomical and spiral wave..." (JTB)
  {
    u_o_ = 0.00;
    u_u_ = 1.0089;
    theta_v_ = 0.3;
    theta_w_ = 0.18171;
    theta_vm_ = 0.1007;
    theta_o_ = 0.015473;
    tau_v1m_ = 16.3;
    tau_v2m_ = 1150;
    tau_vp_ = 1.7026;
    tau_w1m_ = 79.963;
    tau_w2m_ = 28.136;
    k_wm_ = 60.219;
    u_wm_ = 0.00991;
    tau_wp_ = 213.55;
    tau_fi_ = 0.083536;
    tau_o1_ = 250.03;
    tau_o2_ = 16.632;
    tau_so1_ = 73.675;
    tau_so2_ = 6.5537;
    k_so_ = 2.9748;
    u_so_ = 0.592093;
    tau_s1_ = 9.876;
    tau_s2_ = 4.2036;
    k_s_ = 2.2268;
    u_s_ = 0.81568;
    tau_si_ = 10.699;
    tau_winf_ = 0.2233;
    w_infs_ = 0.902;
  }
  else
  {
    FOUR_C_THROW(
        "Parameters for tissue type not supported for minimal model (only M, EPI, ENDO, Atria, "
        "pAtria, pAtriaRe, rAtria)");
  }

  // Variables for electromechanical coupling
  mechanical_activation_ = 0.0;  // to store the variable for activation (phi in this case=)
}

double MyocardMinimal::rea_coeff(const double phi, const double dt)
{
  return MyocardMinimal::rea_coeff(phi, dt, 0);
}

double MyocardMinimal::rea_coeff(const double phi, const double dt, int gp)
{
  double reacoeff = 0.0;
  const double p = 1000.0;

  // calculate voltage dependent time constants ([7] page 545)
  double Tau_vm = tools_.gating_function(tau_v1m_, tau_v2m_, p, phi, theta_vm_);
  double Tau_wm = tools_.gating_function(tau_w1m_, tau_w2m_, k_wm_, phi, u_wm_);
  double Tau_so = tools_.gating_function(tau_so1_, tau_so2_, k_so_, phi, u_so_);
  double Tau_s = tools_.gating_function(tau_s1_, tau_s2_, p, phi, theta_w_);
  double Tau_o = tools_.gating_function(tau_o1_, tau_o2_, p, phi, theta_o_);

  // calculate infinity values ([7] page 545)
  double v_inf = tools_.gating_function(1.0, 0.0, p, phi, theta_vm_);
  double w_inf = tools_.gating_function(1.0 - phi / tau_winf_, w_infs_, p, phi, theta_o_);

  // calculate gating variables according to [8]
  double Tau_v = tools_.gating_function(Tau_vm, tau_vp_, p, phi, theta_v_);
  double v_inf_GF = tools_.gating_function(v_inf, 0.0, p, phi, theta_v_);
  v_[gp] = tools_.gating_var_calc(dt, v0_[gp], v_inf_GF, Tau_v);

  double Tau_w = tools_.gating_function(Tau_wm, tau_wp_, p, phi, theta_w_);
  double w_inf_GF = tools_.gating_function(w_inf, 0.0, p, phi, theta_w_);
  w_[gp] = tools_.gating_var_calc(dt, w0_[gp], w_inf_GF, Tau_w);

  const double s_inf = tools_.gating_function(0.0, 1.0, k_s_, phi, u_s_);
  s_[gp] = tools_.gating_var_calc(dt, s0_[gp], s_inf, Tau_s);

  // calculate currents J_fi, J_so and J_si ([7] page 545)
  jfi_[gp] =
      -tools_.gating_function(0.0, v_[gp] * (phi - theta_v_) * (u_u_ - phi) / tau_fi_, p, phi,
          theta_v_);  // fast inward current
  jso_[gp] = tools_.gating_function(
      (phi - u_o_) / Tau_o, 1.0 / Tau_so, p, phi, theta_w_);  // slow outward current
  jsi_[gp] = -tools_.gating_function(
      0.0, w_[gp] * s_[gp] / tau_si_, p, phi, theta_w_);  // slow inward current

  reacoeff = (jfi_[gp] + jso_[gp] + jsi_[gp]);


  // Store necessary variables for mechanical activation and electromechanical coupling
  mechanical_activation_ = phi;

  return reacoeff;
}


double MyocardMinimal::rea_coeff_n(const double phi, const double dt, int gp)
{
  double reacoeff = 0.0;
  const double p = 1000.0;

  // calculate voltage dependent time constants ([7] page 545)

  double Tau_so = tools_.gating_function(tau_so1_, tau_so2_, k_so_, phi, u_so_);
  double Tau_o = tools_.gating_function(tau_o1_, tau_o2_, p, phi, theta_o_);

  // calculate currents J_fi, J_so and J_si ([7] page 545)
  jfi_[gp] = -tools_.gating_function(0.0, v0_[gp] * (phi - theta_v_) * (u_u_ - phi) / tau_fi_, p,
      phi, theta_v_);  // fast inward current
  jso_[gp] = tools_.gating_function(
      (phi - u_o_) / Tau_o, 1.0 / Tau_so, p, phi, theta_w_);  // slow outward current
  jsi_[gp] = -tools_.gating_function(
      0.0, w0_[gp] * s0_[gp] / tau_si_, p, phi, theta_w_);  // slow inward current

  reacoeff = (jfi_[gp] + jso_[gp] + jsi_[gp]);


  // Store necessary variables for mechanical activation and electromechanical coupling
  mechanical_activation_ = phi;

  return reacoeff;
}

/*----------------------------------------------------------------------*
 |  returns number of internal state variables of the material  cbert 08/13 |
 *----------------------------------------------------------------------*/
int MyocardMinimal::get_number_of_internal_state_variables() const { return 3; }

/*----------------------------------------------------------------------*
 |  returns current internal state of the material          cbert 08/13 |
 *----------------------------------------------------------------------*/
double MyocardMinimal::get_internal_state(const int k) const { return get_internal_state(k, 0); }

/*----------------------------------------------------------------------*
 |  returns current internal state of the material       hoermann 09/15 |
 |  for multiple points per element                                     |
 *----------------------------------------------------------------------*/
double MyocardMinimal::get_internal_state(const int k, int gp) const
{
  double val = 0.0;
  switch (k)
  {
    case -1:
    {
      val = mechanical_activation_;
      break;
    }
    case 0:
    {
      val = v_[gp];
      break;
    }
    case 1:
    {
      val = w_[gp];
      break;
    }
    case 2:
    {
      val = s_[gp];
      break;
    }
  }
  return val;
}

/*----------------------------------------------------------------------*
 |  set  internal state of the material                     cbert 08/13 |
 *----------------------------------------------------------------------*/
void MyocardMinimal::set_internal_state(const int k, const double val)
{
  set_internal_state(k, val, 0);
  return;
}

/*----------------------------------------------------------------------*
 |  set  internal state of the material                  hoermann 09/15 |
 |  for multiple points per element                                     |
 *----------------------------------------------------------------------*/
void MyocardMinimal::set_internal_state(const int k, const double val, int gp)
{
  if (v0_.size() < (unsigned)gp) FOUR_C_THROW("Number of gp does not match");
  switch (k)
  {
    case -1:
    {
      mechanical_activation_ = val;
      break;
    }
    case 0:
    {
      v0_[gp] = val;
      v_[gp] = val;
      break;
    }
    case 1:
    {
      w0_[gp] = val;
      w_[gp] = val;
      break;
    }
    case 2:
    {
      s0_[gp] = val;
      s_[gp] = val;
      break;
    }
    default:
    {
      FOUR_C_THROW("There are only 3 internal variables in this myocard material!");
      break;
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 |  returns number of internal state variables of the material  cbert 08/13 |
 *----------------------------------------------------------------------*/
int MyocardMinimal::get_number_of_ionic_currents() const { return 3; }

/*----------------------------------------------------------------------*
 |  returns current internal currents          cbert 08/13 |
 *----------------------------------------------------------------------*/
double MyocardMinimal::get_ionic_currents(const int k) const { return get_ionic_currents(k, 0); }

/*----------------------------------------------------------------------*
 |  returns current internal currents                    hoermann 09/15 |
 |  for multiple points per element                                     |
 *----------------------------------------------------------------------*/
double MyocardMinimal::get_ionic_currents(const int k, int gp) const
{
  double val = 0.0;
  switch (k)
  {
    case 0:
    {
      val = jfi_[gp];
      break;
    }
    case 1:
    {
      val = jso_[gp];
      break;
    }
    case 2:
    {
      val = jsi_[gp];
      break;
    }
  }
  return val;
}

/*----------------------------------------------------------------------*
 |  update of material at the end of a time step             ljag 07/12 |
 *----------------------------------------------------------------------*/
void MyocardMinimal::update(const double phi, const double dt)
{
  // update initial values for next time step
  v0_ = v_;
  w0_ = w_;
  s0_ = s_;

  return;
}

/*----------------------------------------------------------------------*
 |  resize internal state variables                      hoermann 12/16 |
 *----------------------------------------------------------------------*/
void MyocardMinimal::resize_internal_state_variables(int gp)
{
  v0_.resize(gp);
  w0_.resize(gp);
  s0_.resize(gp);
  v_.resize(gp);
  w_.resize(gp);
  s_.resize(gp);
  jfi_.resize(gp);
  jso_.resize(gp);
  jsi_.resize(gp);
  return;
}

/*----------------------------------------------------------------------*
 |  get number of Gauss points                           hoermann 12/16 |
 *----------------------------------------------------------------------*/
int MyocardMinimal::get_number_of_gp() const { return v_.size(); };

FOUR_C_NAMESPACE_CLOSE
