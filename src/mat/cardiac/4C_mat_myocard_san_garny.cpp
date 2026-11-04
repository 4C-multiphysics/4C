// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_myocard_san_garny.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  variables                                                ljag 09/13 |
 *----------------------------------------------------------------------

   There are a total of 52 entries in the a_ variable array.
   There are a total of 15 entries in each of the rate and state variable arrays.
   There are a total of 133 entries in the constant variable array.


 * VOI is time in component environment (second).
 * c_[0] is Version in component membrane (dimensionless).
 * c_[1] is dCell in component membrane (dimensionless).
 * c_[2] is FCellConstant in component membrane (dimensionless).
 * c_[3] is R in component membrane (millijoule_per_mole_kelvin).
 * c_[4] is T in component membrane (kelvin).
 * c_[5] is F in component membrane (coulomb_per_mole).
 * c_[6] is CmCentre in component membrane (microF).
 * c_[7] is CmPeriphery in component membrane (microF).
 * c_[8] is g_Na_Centre_Published in component sodium_current (microlitre_per_second).
 * c_[9] is g_Na_Centre_0DCapable in component sodium_current (microlitre_per_second).
 * c_[10] is g_Na_Centre_1DCapable in component sodium_current (microlitre_per_second).
 * c_[11] is g_Na_Periphery_Published in component sodium_current (microlitre_per_second).
 * c_[12] is g_Na_Periphery_0DCapable in component sodium_current (microlitre_per_second).
 * c_[13] is g_Na_Periphery_1DCapable in component sodium_current (microlitre_per_second).
 * c_[14] is Na_o in component ionic_concentrations (millimolar).
 * c_[15] is g_Ca_L_Centre_Published in component L_type_Ca_channel (microS).
 * c_[16] is g_Ca_L_Centre_0DCapable in component L_type_Ca_channel (microS).
 * c_[17] is g_Ca_L_Centre_1DCapable in component L_type_Ca_channel (microS).
 * c_[18] is g_Ca_L_Periphery_Published in component L_type_Ca_channel (microS).
 * c_[19] is g_Ca_L_Periphery_0DCapable in component L_type_Ca_channel (microS).
 * c_[20] is g_Ca_L_Periphery_1DCapable in component L_type_Ca_channel (microS).
 * c_[21] is E_Ca_L in component L_type_Ca_channel (millivolt).
 * c_[22] is g_Ca_T_Centre_Published in component T_type_Ca_channel (microS).
 * c_[23] is g_Ca_T_Centre_0DCapable in component T_type_Ca_channel (microS).
 * c_[24] is g_Ca_T_Centre_1DCapable in component T_type_Ca_channel (microS).
 * c_[25] is g_Ca_T_Periphery_Published in component T_type_Ca_channel (microS).
 * c_[26] is g_Ca_T_Periphery_0DCapable in component T_type_Ca_channel (microS).
 * c_[27] is g_Ca_T_Periphery_1DCapable in component T_type_Ca_channel (microS).
 * c_[28] is E_Ca_T in component T_type_Ca_channel (millivolt).
 * c_[29] is g_to_Centre_Published in component four_AP_sensitive_currents (microS).
 * c_[30] is g_to_Centre_0DCapable in component four_AP_sensitive_currents (microS).
 * c_[31] is g_to_Centre_1DCapable in component four_AP_sensitive_currents (microS).
 * c_[32] is g_to_Periphery_Published in component four_AP_sensitive_currents (microS).
 * c_[33] is g_to_Periphery_0DCapable in component four_AP_sensitive_currents (microS).
 * c_[34] is g_to_Periphery_1DCapable in component four_AP_sensitive_currents (microS).
 * c_[35] is g_sus_Centre_Published in component four_AP_sensitive_currents (microS).
 * c_[36] is g_sus_Centre_0DCapable in component four_AP_sensitive_currents (microS).
 * c_[37] is g_sus_Centre_1DCapable in component four_AP_sensitive_currents (microS).
 * c_[38] is g_sus_Periphery_Published in component four_AP_sensitive_currents (microS).
 * c_[39] is g_sus_Periphery_0DCapable in component four_AP_sensitive_currents (microS).
 * c_[40] is g_sus_Periphery_1DCapable in component four_AP_sensitive_currents (microS).
 * c_[41] is g_K_r_Centre_Published in component rapid_delayed_rectifying_potassium_current
 (microS).
 * c_[42] is g_K_r_Centre_0DCapable in component rapid_delayed_rectifying_potassium_current
 (microS).
 * c_[43] is g_K_r_Centre_1DCapable in component rapid_delayed_rectifying_potassium_current
 (microS).
 * c_[44] is g_K_r_Periphery_Published in component rapid_delayed_rectifying_potassium_current
 (microS).
 * c_[45] is g_K_r_Periphery_0DCapable in component rapid_delayed_rectifying_potassium_current
 (microS).
 * c_[46] is g_K_r_Periphery_1DCapable in component rapid_delayed_rectifying_potassium_current
 (microS).
 * c_[47] is g_K_s_Centre_Published in component slow_delayed_rectifying_potassium_current (microS).
 * c_[48] is g_K_s_Centre_0DCapable in component slow_delayed_rectifying_potassium_current (microS).
 * c_[49] is g_K_s_Centre_1DCapable in component slow_delayed_rectifying_potassium_current (microS).
 * c_[50] is g_K_s_Periphery_Published in component slow_delayed_rectifying_potassium_current
 (microS).
 * c_[51] is g_K_s_Periphery_0DCapable in component slow_delayed_rectifying_potassium_current
 (microS).
 * c_[52] is g_K_s_Periphery_1DCapable in component slow_delayed_rectifying_potassium_current
 (microS).
 * c_[53] is g_f_Na_Centre_Published in component hyperpolarisation_activated_current (microS).
 * c_[54] is g_f_Na_Centre_0DCapable in component hyperpolarisation_activated_current (microS).
 * c_[55] is g_f_Na_Centre_1DCapable in component hyperpolarisation_activated_current (microS).
 * c_[56] is g_f_Na_Periphery_Published in component hyperpolarisation_activated_current (microS).
 * c_[57] is g_f_Na_Periphery_0DCapable in component hyperpolarisation_activated_current (microS).
 * c_[58] is g_f_Na_Periphery_1DCapable in component hyperpolarisation_activated_current (microS).
 * c_[59] is g_f_K_Centre_Published in component hyperpolarisation_activated_current (microS).
 * c_[60] is g_f_K_Centre_0DCapable in component hyperpolarisation_activated_current (microS).
 * c_[61] is g_f_K_Centre_1DCapable in component hyperpolarisation_activated_current (microS).
 * c_[62] is g_f_K_Periphery_Published in component hyperpolarisation_activated_current (microS).
 * c_[63] is g_f_K_Periphery_0DCapable in component hyperpolarisation_activated_current (microS).
 * c_[64] is g_f_K_Periphery_1DCapable in component hyperpolarisation_activated_current (microS).
 * c_[65] is g_b_Na_Centre_Published in component sodium_background_current (microS).
 * c_[66] is g_b_Na_Centre_0DCapable in component sodium_background_current (microS).
 * c_[67] is g_b_Na_Centre_1DCapable in component sodium_background_current (microS).
 * c_[68] is g_b_Na_Periphery_Published in component sodium_background_current (microS).
 * c_[69] is g_b_Na_Periphery_0DCapable in component sodium_background_current (microS).
 * c_[70] is g_b_Na_Periphery_1DCapable in component sodium_background_current (microS).
 * c_[71] is g_b_K_Centre_Published in component potassium_background_current (microS).
 * c_[72] is g_b_K_Centre_0DCapable in component potassium_background_current (microS).
 * c_[73] is g_b_K_Centre_1DCapable in component potassium_background_current (microS).
 * c_[74] is g_b_K_Periphery_Published in component potassium_background_current (microS).
 * c_[75] is g_b_K_Periphery_0DCapable in component potassium_background_current (microS).
 * c_[76] is g_b_K_Periphery_1DCapable in component potassium_background_current (microS).
 * c_[77] is g_b_Ca_Centre_Published in component calcium_background_current (microS).
 * c_[78] is g_b_Ca_Centre_0DCapable in component calcium_background_current (microS).
 * c_[79] is g_b_Ca_Centre_1DCapable in component calcium_background_current (microS).
 * c_[80] is g_b_Ca_Periphery_Published in component calcium_background_current (microS).
 * c_[81] is g_b_Ca_Periphery_0DCapable in component calcium_background_current (microS).
 * c_[82] is g_b_Ca_Periphery_1DCapable in component calcium_background_current (microS).
 * c_[83] is k_NaCa_Centre_Published in component sodium_calcium_exchanger (nanoA).
 * c_[84] is k_NaCa_Centre_0DCapable in component sodium_calcium_exchanger (nanoA).
 * c_[85] is k_NaCa_Centre_1DCapable in component sodium_calcium_exchanger (nanoA).
 * c_[86] is k_NaCa_Periphery_Published in component sodium_calcium_exchanger (nanoA).
 * c_[87] is k_NaCa_Periphery_0DCapable in component sodium_calcium_exchanger (nanoA).
 * c_[88] is k_NaCa_Periphery_1DCapable in component sodium_calcium_exchanger (nanoA).
 * c_[89] is d_NaCa in component sodium_calcium_exchanger (dimensionless).
 * c_[90] is gamma_NaCa in component sodium_calcium_exchanger (dimensionless).
 * c_[91] is Na_i in component ionic_concentrations (millimolar).
 * c_[92] is Ca_i in component ionic_concentrations (millimolar).
 * c_[93] is Ca_o in component ionic_concentrations (millimolar).
 * c_[94] is K_m_Na in component sodium_potassium_pump (millimolar).
 * c_[95] is K_m_K in component sodium_potassium_pump (millimolar).
 * c_[96] is i_p_max_Centre_Published in component sodium_potassium_pump (nanoA).
 * c_[97] is i_p_max_Centre_0DCapable in component sodium_potassium_pump (nanoA).
 * c_[98] is i_p_max_Centre_1DCapable in component sodium_potassium_pump (nanoA).
 * c_[99] is i_p_max_Periphery_Published in component sodium_potassium_pump (nanoA).
 * c_[100] is i_p_max_Periphery_0DCapable in component sodium_potassium_pump (nanoA).
 * c_[101] is i_p_max_Periphery_1DCapable in component sodium_potassium_pump (nanoA).
 * c_[102] is K_o in component ionic_concentrations (millimolar).
 * c_[103] is i_Ca_p_max_Centre_Published in component persistent_calcium_current (nanoA).
 * c_[104] is i_Ca_p_max_Centre_0DCapable in component persistent_calcium_current (nanoA).
 * c_[105] is i_Ca_p_max_Centre_1DCapable in component persistent_calcium_current (nanoA).
 * c_[106] is i_Ca_p_max_Periphery_Published in component persistent_calcium_current (nanoA).
 * c_[107] is i_Ca_p_max_Periphery_0DCapable in component persistent_calcium_current (nanoA).
 * c_[108] is i_Ca_p_max_Periphery_1DCapable in component persistent_calcium_current (nanoA).
 * c_[109] is K_i in component ionic_concentrations (millimolar).
 * c_[110] is FCell in component membrane (dimensionless).
 * c_[111] is E_Na in component reversal_and_equilibrium_potentials (millivolt).
 * c_[112] is E_K in component reversal_and_equilibrium_potentials (millivolt).
 * c_[113] is tau_P_i in component rapid_delayed_rectifying_potassium_current_P_i_gate (second).
 * c_[114] is E_K_s in component reversal_and_equilibrium_potentials (millivolt).
 * c_[115] is E_Ca in component reversal_and_equilibrium_potentials (millivolt).
 * c_[116] is Cm in component membrane (microF).
 * c_[117] is g_Na in component sodium_current (microlitre_per_second).
 * c_[118] is g_Ca_L in component L_type_Ca_channel (microS).
 * c_[119] is g_Ca_T in component T_type_Ca_channel (microS).
 * c_[120] is g_to in component four_AP_sensitive_currents (microS).
 * c_[121] is g_sus in component four_AP_sensitive_currents (microS).
 * c_[122] is g_K_r in component rapid_delayed_rectifying_potassium_current (microS).
 * c_[123] is g_K_s in component slow_delayed_rectifying_potassium_current (microS).
 * c_[124] is g_f_Na in component hyperpolarisation_activated_current (microS).
 * c_[125] is g_f_K in component hyperpolarisation_activated_current (microS).
 * c_[126] is g_b_Na in component sodium_background_current (microS).
 * c_[127] is g_b_K in component potassium_background_current (microS).
 * c_[128] is g_b_Ca in component calcium_background_current (microS).
 * c_[129] is k_NaCa in component sodium_calcium_exchanger (nanoA).
 * c_[130] is i_p_max in component sodium_potassium_pump (nanoA).
 * c_[131] is i_Ca_p_max in component persistent_calcium_current (nanoA).
 * c_[132] is i_Ca_p in component persistent_calcium_current (nanoA).
 *
 *
 * a_[0] is F_Na in component sodium_current_h_gate (dimensionless).
 * a_[1] is m_infinity in component sodium_current_m_gate (dimensionless).
 * a_[2] is h1_infinity in component sodium_current_h_gate (dimensionless).
 * a_[3] is alpha_d_L in component L_type_Ca_channel_d_gate (per_second).
 * a_[4] is alpha_f_L in component L_type_Ca_channel_f_gate (per_second).
 * a_[5] is alpha_d_T in component T_type_Ca_channel_d_gate (per_second).
 * a_[6] is alpha_f_T in component T_type_Ca_channel_f_gate (per_second).
 * a_[7] is q_infinity in component four_AP_sensitive_currents_q_gate (dimensionless).
 * a_[8] is r_infinity in component four_AP_sensitive_currents_r_gate (dimensionless).
 * a_[9] is P_af_infinity in component rapid_delayed_rectifying_potassium_current_P_af_gate
 (dimensionless).
 * a_[10] is P_i_infinity in component rapid_delayed_rectifying_potassium_current_P_i_gate
 (dimensionless).
 * a_[11] is alpha_xs in component slow_delayed_rectifying_potassium_current_xs_gate (per_second).
 * a_[12] is alpha_y in component hyperpolarisation_activated_current_y_gate (per_second).
 * a_[13] is h in component sodium_current_h_gate (dimensionless).
 * a_[14] is tau_m in component sodium_current_m_gate (second).
 * a_[15] is h2_infinity in component sodium_current_h_gate (dimensionless).
 * a_[16] is tau_h1 in component sodium_current_h_gate (second).
 * a_[17] is beta_d_L in component L_type_Ca_channel_d_gate (per_second).
 * a_[18] is beta_f_L in component L_type_Ca_channel_f_gate (per_second).
 * a_[19] is beta_d_T in component T_type_Ca_channel_d_gate (per_second).
 * a_[20] is beta_f_T in component T_type_Ca_channel_f_gate (per_second).
 * a_[21] is tau_q in component four_AP_sensitive_currents_q_gate (second).
 * a_[22] is tau_r in component four_AP_sensitive_currents_r_gate (second).
 * a_[23] is tau_P_af in component rapid_delayed_rectifying_potassium_current_P_af_gate (second).
 * a_[24] is P_as_infinity in component rapid_delayed_rectifying_potassium_current_P_as_gate
 (dimensionless).
 * a_[25] is beta_xs in component slow_delayed_rectifying_potassium_current_xs_gate (per_second).
 * a_[26] is beta_y in component hyperpolarisation_activated_current_y_gate (per_second).
 * a_[27] is i_Na in component sodium_current (nanoA).
 * a_[28] is tau_h2 in component sodium_current_h_gate (second).
 * a_[29] is tau_d_L in component L_type_Ca_channel_d_gate (second).
 * a_[30] is tau_f_L in component L_type_Ca_channel_f_gate (second).
 * a_[31] is tau_d_T in component T_type_Ca_channel_d_gate (second).
 * a_[32] is tau_f_T in component T_type_Ca_channel_f_gate (second).
 * a_[33] is tau_P_as in component rapid_delayed_rectifying_potassium_current_P_as_gate (second).
 * a_[34] is i_Ca_L in component L_type_Ca_channel (nanoA).
 * a_[35] is d_L_infinity in component L_type_Ca_channel_d_gate (dimensionless).
 * a_[36] is f_L_infinity in component L_type_Ca_channel_f_gate (dimensionless).
 * a_[37] is d_T_infinity in component T_type_Ca_channel_d_gate (dimensionless).
 * a_[38] is f_T_infinity in component T_type_Ca_channel_f_gate (dimensionless).
 * a_[39] is i_Ca_T in component T_type_Ca_channel (nanoA).
 * a_[40] is i_to in component four_AP_sensitive_currents (nanoA).
 * a_[41] is i_sus in component four_AP_sensitive_currents (nanoA).
 * a_[42] is P_a in component rapid_delayed_rectifying_potassium_current (dimensionless).
 * a_[43] is i_K_r in component rapid_delayed_rectifying_potassium_current (nanoA).
 * a_[44] is i_K_s in component slow_delayed_rectifying_potassium_current (nanoA).
 * a_[45] is i_f_Na in component hyperpolarisation_activated_current (nanoA).
 * a_[46] is i_f_K in component hyperpolarisation_activated_current (nanoA).
 * a_[47] is i_b_Na in component sodium_background_current (nanoA).
 * a_[48] is i_b_K in component potassium_background_current (nanoA).
 * a_[49] is i_b_Ca in component calcium_background_current (nanoA).
 * a_[50] is i_NaCa in component sodium_calcium_exchanger (nanoA).
 * a_[51] is i_p in component sodium_potassium_pump (nanoA).
 *
 *
 *
 * s0_[0] is V in component membrane (millivolt).
 * s0_[1] is m in component sodium_current_m_gate (dimensionless).
 * s0_[2] is h1 in component sodium_current_h_gate (dimensionless).
 * s0_[3] is h2 in component sodium_current_h_gate (dimensionless).
 * s0_[4] is d_L in component L_type_Ca_channel_d_gate (dimensionless).
 * s0_[5] is f_L in component L_type_Ca_channel_f_gate (dimensionless).
 * s0_[6] is d_T in component T_type_Ca_channel_d_gate (dimensionless).
 * s0_[7] is f_T in component T_type_Ca_channel_f_gate (dimensionless).
 * s0_[8] is q in component four_AP_sensitive_currents_q_gate (dimensionless).
 * s0_[9] is r in component four_AP_sensitive_currents_r_gate (dimensionless).
 * s0_[10] is P_af in component rapid_delayed_rectifying_potassium_current_P_af_gate
 (dimensionless).
 * s0_[11] is P_as in component rapid_delayed_rectifying_potassium_current_P_as_gate
 (dimensionless).
 * s0_[12] is P_i in component rapid_delayed_rectifying_potassium_current_P_i_gate (dimensionless).
 * s0_[13] is xs in component slow_delayed_rectifying_potassium_current_xs_gate (dimensionless).
 * s0_[14] is y in component hyperpolarisation_activated_current_y_gate (dimensionless).
 *
 * r_[0] is d/dt V in component membrane (millivolt).
 * r_[1] is d/dt m in component sodium_current_m_gate (dimensionless).
 * r_[2] is d/dt h1 in component sodium_current_h_gate (dimensionless).
 * r_[3] is d/dt h2 in component sodium_current_h_gate (dimensionless).
 * r_[4] is d/dt d_L in component L_type_Ca_channel_d_gate (dimensionless).
 * r_[5] is d/dt f_L in component L_type_Ca_channel_f_gate (dimensionless).
 * r_[6] is d/dt d_T in component T_type_Ca_channel_d_gate (dimensionless).
 * r_[7] is d/dt f_T in component T_type_Ca_channel_f_gate (dimensionless).
 * r_[8] is d/dt q in component four_AP_sensitive_currents_q_gate (dimensionless).
 * r_[9] is d/dt r in component four_AP_sensitive_currents_r_gate (dimensionless).
 * r_[10] is d/dt P_af in component rapid_delayed_rectifying_potassium_current_P_af_gate
 (dimensionless).
 * r_[11] is d/dt P_as in component rapid_delayed_rectifying_potassium_current_P_as_gate
 (dimensionless).
 * r_[12] is d/dt P_i in component rapid_delayed_rectifying_potassium_current_P_i_gate
 (dimensionless).
 * r_[13] is d/dt xs in component slow_delayed_rectifying_potassium_current_xs_gate (dimensionless).
 * r_[14] is d/dt y in component hyperpolarisation_activated_current_y_gate (dimensionless).
 */

/*----------------------------------------------------------------------*
 |  Constructor                                    (public)  cbert 08/13 |
 *----------------------------------------------------------------------*/
MyocardSanGarny::MyocardSanGarny() {}


/*----------------------------------------------------------------------*
 |  Constructor                                    (public)  cbert 08/13 |
 *----------------------------------------------------------------------*/
MyocardSanGarny::MyocardSanGarny(const double eps_deriv_myocard, const std::string tissue)
    : tools_(), s0_(15, 0.0), s_(15, 0.0), r_(15, 0.0), a_(52, 0.0), c_(133, 0.0)

{
  eps_deriv_ = eps_deriv_myocard;

  // initial conditions
  s0_[0] = -39.013558536;
  s0_[1] = 0.092361701692;
  s0_[2] = 0.015905380261;
  s0_[3] = 0.01445216109;
  s0_[4] = 0.04804900895;
  s0_[5] = 0.48779845203;
  s0_[6] = 0.42074047435;
  s0_[7] = 0.038968420558;
  s0_[8] = 0.29760539675;
  s0_[9] = 0.064402950262;
  s0_[10] = 0.13034201158;
  s0_[11] = 0.46960956028;
  s0_[12] = 0.87993375273;
  s0_[13] = 0.082293827208;
  s0_[14] = 0.03889291759;

  s_ = s0_;

  // Model constants
  c_[0] = 2;
  c_[1] = 0;
  c_[2] = 1.0309347;
  c_[3] = 8314;
  c_[4] = 310;
  c_[5] = 96845;
  c_[6] = 2e-5;
  c_[7] = 6.5e-5;
  c_[8] = 0;
  c_[9] = 0;
  c_[10] = 0;
  c_[11] = 1.2e-6;  // switched from ul to um^3
  c_[12] = 1.204e-6;
  c_[13] = 3.7e-7;
  c_[14] = 140;
  c_[15] = 0.0058;
  c_[16] = 0.0057938;
  c_[17] = 0.0082;
  c_[18] = 0.0659;
  c_[19] = 0.06588648;
  c_[20] = 0.0659;
  c_[21] = 46.4;
  c_[22] = 0.0043;
  c_[23] = 0.00427806;
  c_[24] = 0.0021;
  c_[25] = 0.0139;
  c_[26] = 0.0138823;
  c_[27] = 0.00694;
  c_[28] = 45;
  c_[29] = 0.00491;
  c_[30] = 0.004905;
  c_[31] = 0.004905;
  c_[32] = 0.03649;
  c_[33] = 0.036495;
  c_[34] = 0.0365;
  c_[35] = 6.65e-5;
  c_[36] = 6.645504e-5;
  c_[37] = 0.000266;
  c_[38] = 0.0114;
  c_[39] = 0.01138376;
  c_[40] = 0.0114;
  c_[41] = 0.000797;
  c_[42] = 0.00079704;
  c_[43] = 0.000738;
  c_[44] = 0.016;
  c_[45] = 0.016;
  c_[46] = 0.0208;
  c_[47] = 0.000518;
  c_[48] = 0.0003445;
  c_[49] = 0.000345;
  c_[50] = 0.0104;
  c_[51] = 0.0104;
  c_[52] = 0.0104;
  c_[53] = 0.000548;
  c_[54] = 0.0005465;
  c_[55] = 0.000437;
  c_[56] = 0.0069;
  c_[57] = 0.006875;
  c_[58] = 0.0055;
  c_[59] = 0.000548;
  c_[60] = 0.0005465;
  c_[61] = 0.000437;
  c_[62] = 0.0069;
  c_[63] = 0.006875;
  c_[64] = 0.0055;
  c_[65] = 5.8e-5;
  c_[66] = 5.81818e-5;
  c_[67] = 5.8e-5;
  c_[68] = 0.000189;
  c_[69] = 0.0001888;
  c_[70] = 0.000189;
  c_[71] = 2.52e-5;
  c_[72] = 2.523636e-5;
  c_[73] = 2.52e-5;
  c_[74] = 8.19e-5;
  c_[75] = 8.1892e-5;
  c_[76] = 8.19e-5;
  c_[77] = 1.32e-5;
  c_[78] = 1.3236e-5;
  c_[79] = 1.323e-5;
  c_[80] = 4.3e-5;
  c_[81] = 4.2952e-5;
  c_[82] = 4.29e-5;
  c_[83] = 2.7e-6;
  c_[84] = 2.7229e-6;
  c_[85] = 2.8e-6;
  c_[86] = 8.8e-6;
  c_[87] = 8.83584e-6;
  c_[88] = 8.8e-6;
  c_[89] = 0.0001;
  c_[90] = 0.5;
  c_[91] = 8;
  c_[92] = 0.0001;
  c_[93] = 2;
  c_[94] = 5.64;
  c_[95] = 0.621;
  c_[96] = 0.0478;
  c_[97] = 0.04782545;
  c_[98] = 0.0478;
  c_[99] = 0.16;
  c_[100] = 0.1551936;
  c_[101] = 0.16;
  c_[102] = 5.4;
  c_[103] = 0;
  c_[104] = 0;
  c_[105] = 0.0042;
  c_[106] = 0;
  c_[107] = 0;
  c_[108] = 0.03339;
  c_[109] = 140;
  c_[110] =
      (c_[0] == 0.0 ? (1.07 * (3.0 * c_[1] - 0.1)) /
                          (3.0 * (1.0 + 0.7745 * (exp((-(3.0 * c_[1] - 2.05) / 0.295)))))
          : c_[0] == 1.0
              ? (c_[2] * c_[1]) / (1.0 + 0.7745 * (exp((-(3.0 * c_[1] - 2.05000) / 0.295000))))
              : (1.07000 * 29.0000 * c_[1]) /
                    (30.0000 * (1.0 + 0.774500 * (exp((-(29.0000 * c_[1] - 24.5000) / 1.95000))))));
  c_[111] = ((c_[3] * c_[4]) / c_[5]) * (log((c_[14] / c_[91])));
  c_[112] = ((c_[3] * c_[4]) / c_[5]) * (log((c_[102] / c_[109])));
  c_[113] = (c_[0] == 0.0 ? 0.002 : c_[0] == 1.0 ? 0.002 : 0.006);
  c_[114] =
      (c_[0] == 0.0 ? ((c_[3] * c_[4]) / c_[5]) *
                          (log(((c_[102] + 0.120000 * c_[14]) / (c_[109] + 0.120000 * c_[91]))))
                    : ((c_[3] * c_[4]) / c_[5]) *
                          (log(((c_[102] + 0.0300000 * c_[14]) / (c_[109] + 0.0300000 * c_[91])))));
  c_[115] = ((c_[3] * c_[4]) / (2.0 * c_[5])) * (log((c_[93] / c_[92])));
  c_[116] = c_[6] + c_[110] * (c_[7] - c_[6]);
  c_[117] = (c_[0] == 0.0   ? c_[8] + c_[110] * (c_[11] - c_[8])
             : c_[0] == 1.0 ? c_[9] + c_[110] * (c_[12] - c_[9])
                            : c_[10] + c_[110] * (c_[13] - c_[10]));
  c_[118] = (c_[0] == 0.0   ? c_[15] + c_[110] * (c_[18] - c_[15])
             : c_[0] == 1.0 ? c_[16] + c_[110] * (c_[19] - c_[16])
                            : c_[17] + c_[110] * (c_[20] - c_[17]));
  c_[119] = (c_[0] == 0.0   ? c_[22] + c_[110] * (c_[25] - c_[22])
             : c_[0] == 1.0 ? c_[23] + c_[110] * (c_[26] - c_[23])
                            : c_[24] + c_[110] * (c_[27] - c_[24]));
  c_[120] = (c_[0] == 0.0   ? c_[29] + c_[110] * (c_[32] - c_[29])
             : c_[0] == 1.0 ? c_[30] + c_[110] * (c_[33] - c_[30])
                            : c_[31] + c_[110] * (c_[34] - c_[31]));
  c_[121] = (c_[0] == 0.0   ? c_[35] + c_[110] * (c_[38] - c_[35])
             : c_[0] == 1.0 ? c_[36] + c_[110] * (c_[39] - c_[36])
                            : c_[37] + c_[110] * (c_[40] - c_[37]));
  c_[122] = (c_[0] == 0.0   ? c_[41] + c_[110] * (c_[44] - c_[41])
             : c_[0] == 1.0 ? c_[42] + c_[110] * (c_[45] - c_[42])
                            : c_[43] + c_[110] * (c_[46] - c_[43]));
  c_[123] = (c_[0] == 0.0   ? c_[47] + c_[110] * (c_[50] - c_[47])
             : c_[0] == 1.0 ? c_[48] + c_[110] * (c_[51] - c_[48])
                            : c_[49] + c_[110] * (c_[52] - c_[49]));
  c_[124] = (c_[0] == 0.0   ? c_[53] + c_[110] * (c_[56] - c_[53])
             : c_[0] == 1.0 ? c_[54] + c_[110] * (c_[57] - c_[54])
                            : c_[55] + c_[110] * (c_[58] - c_[55]));
  c_[125] = (c_[0] == 0.0   ? c_[59] + c_[110] * (c_[62] - c_[59])
             : c_[0] == 1.0 ? c_[60] + c_[110] * (c_[63] - c_[60])
                            : c_[61] + c_[110] * (c_[64] - c_[61]));
  c_[126] = (c_[0] == 0.0   ? c_[65] + c_[110] * (c_[68] - c_[65])
             : c_[0] == 1.0 ? c_[66] + c_[110] * (c_[69] - c_[66])
                            : c_[67] + c_[110] * (c_[70] - c_[67]));
  c_[127] = (c_[0] == 0.0   ? c_[71] + c_[110] * (c_[74] - c_[71])
             : c_[0] == 1.0 ? c_[72] + c_[110] * (c_[75] - c_[72])
                            : c_[73] + c_[110] * (c_[76] - c_[73]));
  c_[128] = (c_[0] == 0.0   ? c_[77] + c_[110] * (c_[80] - c_[77])
             : c_[0] == 1.0 ? c_[78] + c_[110] * (c_[81] - c_[78])
                            : c_[79] + c_[110] * (c_[82] - c_[79]));
  c_[129] = (c_[0] == 0.0   ? c_[83] + c_[110] * (c_[86] - c_[83])
             : c_[0] == 1.0 ? c_[84] + c_[110] * (c_[87] - c_[84])
                            : c_[85] + c_[110] * (c_[88] - c_[85]));
  c_[130] = (c_[0] == 0.0   ? c_[96] + c_[110] * (c_[99] - c_[96])
             : c_[0] == 1.0 ? c_[97] + c_[110] * (c_[100] - c_[97])
                            : c_[98] + c_[110] * (c_[101] - c_[98]));
  c_[131] = (c_[0] == 0.0   ? c_[103] + c_[110] * (c_[106] - c_[103])
             : c_[0] == 1.0 ? c_[104] + c_[110] * (c_[107] - c_[104])
                            : c_[105] + c_[110] * (c_[108] - c_[105]));
  c_[132] = (c_[131] * c_[92]) / (c_[92] + 0.000400000);
}


double MyocardSanGarny::rea_coeff(const double phi, const double dt)
{
  s0_[0] = phi;
  s_[0] = phi;

  // Compute new gating variables
  // ----------------------------

  // s0_[12] is P_i in component rapid_delayed_rectifying_potassium_current_P_i_gate
  // (dimensionless).
  a_[10] = 1.0 / (1.0 + (exp(((s0_[0] + 18.6) / 10.1))));
  r_[12] = (a_[10] - s0_[12]) / c_[113];
  s_[12] = tools_.gating_var_calc(dt, s0_[12], a_[10], c_[113]);

  // s0_[1] is m in component sodium_current_m_gate (dimensionless).
  a_[1] = (c_[0] == 0.0 ? (pow((1.0 / (1.0 + (exp((-s0_[0] / 5.46))))), (1.0 / 3.0)))
                        : (pow((1.0 / (1.0 + (exp((-(s0_[0] + 30.32) / 5.46))))), (1.0 / 3.0))));
  a_[14] = (c_[0] == 0.0 ? 0.0006247 / (0.832 * (exp((-0.335 * (s0_[0] + 56.7)))) +
                                           0.627 * (exp((0.082 * (s0_[0] + 65.0100))))) +
                               4.0e-05
                         : 0.0006247 / (0.832217 * (exp((-0.33566 * (s0_[0] + 56.7062)))) +
                                           0.627400 * (exp((0.0823000 * (s0_[0] + 65.0131))))) +
                               4.56900e-05);
  r_[1] = (a_[1] - s0_[1]) / a_[14];
  s_[1] = tools_.gating_var_calc(dt, s0_[1], a_[1], a_[14]);

  // s0_[2] is h1 in component sodium_current_h_gate (dimensionless).
  a_[2] = 1.0 / (1.0 + (exp(((s0_[0] + 66.1) / 6.4))));
  a_[16] = (3.717e-06 * (exp((-0.2815 * (s0_[0] + 17.11))))) /
               (1.0 + 0.003732 * (exp((-0.342600 * (s0_[0] + 37.7600))))) +
           0.0005977;
  r_[2] = (a_[2] - s0_[2]) / a_[16];
  s_[2] = tools_.gating_var_calc(dt, s0_[2], a_[2], a_[16]);

  // s0_[8] is q in component four_AP_sensitive_currents_q_gate (dimensionless).
  a_[7] = 1.0 / (1.0 + (exp(((s0_[0] + 59.37) / 13.1))));
  a_[21] = (c_[0] == 0.0 ? 0.0101 + 0.06517 / (0.57 * (exp((-0.08 * (s0_[0] + 49.0))))) +
                               2.4e-05 * (exp((0.1 * (s0_[0] + 50.93))))
            : c_[0] == 1.0
                ? (0.001 / 3.0) *
                      (30.3100 +
                          195.500 /
                              (0.5686 * (exp((-0.0816100 * (s0_[0] + 39.0 + 10.0000 * c_[110])))) +
                                  0.717400 * (exp(((0.2719 - 0.1719 * c_[110]) * 1.0 *
                                                   (s0_[0] + 40.93 + 10.0 * c_[110]))))))
                : 0.0101 + 0.06517 / (0.5686 * (exp((-0.08161 * (s0_[0] + 39.0)))) +
                                         0.7174 * (exp((0.2719 * (s0_[0] + 40.93))))));
  r_[8] = (a_[7] - s0_[8]) / a_[21];
  s_[8] = tools_.gating_var_calc(dt, s0_[8], a_[7], a_[21]);

  // s0_[9] is r in component four_AP_sensitive_currents_r_gate (dimensionless).
  a_[8] = 1.0 / (1.0 + (exp((-(s0_[0] - 10.93) / 19.7))));
  a_[22] = (c_[0] == 0.0 ? 0.001 * (2.98 + 15.59 / (1.037 * (exp((0.09 * (s0_[0] + 30.61)))) +
                                                       0.369 * (exp((-0.12 * (s0_[0] + 23.84))))))
            : c_[0] == 1.0
                ? 0.0025 * (1.191 + 7.838 / (1.037 * (exp((0.09012 * (s0_[0] + 30.61)))) +
                                                0.369000 * (exp((-0.119000 * (s0_[0] + 23.84))))))
                : 0.001 * (2.98 + 19.59 / (1.037 * (exp((0.09012 * (s0_[0] + 30.61)))) +
                                              0.369 * (exp((-0.119 * (s0_[0] + 23.84)))))));
  r_[9] = (a_[8] - s0_[9]) / a_[22];
  s_[9] = tools_.gating_var_calc(dt, s0_[9], a_[8], a_[22]);

  // s0_[10] is P_af in component rapid_delayed_rectifying_potassium_current_P_af_gate
  // (dimensionless).
  a_[9] = (c_[0] != 2.0 ? 1.0 / (1.0 + (exp((-(s0_[0] + 14.2) / 10.6))))
                        : 1.0 / (1.0 + (exp((-(s0_[0] + 13.2) / 10.6)))));
  a_[23] = (c_[0] != 2.0 ? 1.0 / (37.2 * (exp(((s0_[0] - 9.0) / 15.9))) +
                                     0.96 * (exp((-(s0_[0] - 9.0) / 22.5))))
                         : 1.0 / (37.2 * (exp(((s0_[0] - 10.0) / 15.9))) +
                                     0.96 * (exp((-(s0_[0] - 10.0) / 22.5)))));
  r_[10] = (a_[9] - s0_[10]) / a_[23];
  s_[10] = tools_.gating_var_calc(dt, s0_[10], a_[9], a_[23]);

  // s0_[13] is xs in component slow_delayed_rectifying_potassium_current_xs_gate (dimensionless).
  a_[11] = 14.0000 / (1.0 + (exp((-(s0_[0] - 40.0) / 9.0))));
  a_[25] = 1.0 * (exp((-s0_[0] / 45.0)));
  r_[13] = a_[11] * (1.0 - s0_[13]) - a_[25] * s0_[13];
  s_[13] = tools_.gating_var_calc(dt, s0_[13], a_[11] / (a_[11] + a_[25]), 1 / (a_[11] + a_[25]));

  // s0_[14] is y in component hyperpolarisation_activated_current_y_gate (dimensionless).
  a_[12] = (c_[0] == 0.0 ? 1.0 * (exp((-(s0_[0] + 78.9100) / 26.6200)))
                         : 1.0 * (exp((-(s0_[0] + 78.9100) / 26.63))));
  a_[26] = 1.0 * (exp(((s0_[0] + 75.1300) / 21.2500)));
  r_[14] = a_[12] * (1.0 - s0_[14]) - a_[26] * s0_[14];
  s_[14] = tools_.gating_var_calc(dt, s0_[14], a_[12] / (a_[12] + a_[26]), 1 / (a_[12] + a_[26]));

  // s0_[3] is h2 in component sodium_current_h_gate (dimensionless).
  a_[15] = a_[2];
  a_[28] = (3.18600e-08 * (exp((-0.621900 * (s0_[0] + 18.8000))))) /
               (1.0 + 7.18900e-05 * (exp((-0.668300 * (s0_[0] + 34.0700))))) +
           0.00355600;
  r_[3] = (a_[15] - s0_[3]) / a_[28];
  s_[3] = tools_.gating_var_calc(dt, s0_[3], a_[15], a_[28]);

  // s0_[11] is P_as in component rapid_delayed_rectifying_potassium_current_P_as_gate
  // (dimensionless).
  a_[24] = a_[9];
  a_[33] = (c_[0] != 2.0 ? 1.0 / (4.20000 * (exp(((s0_[0] - 9.0) / 17.0))) +
                                     0.150000 * (exp((-(s0_[0] - 9.0) / 21.6000))))
                         : 1.0 / (4.20000 * (exp(((s0_[0] - 10.0) / 17.0))) +
                                     0.150000 * (exp((-(s0_[0] - 10.0) / 21.6000)))));
  r_[11] = (a_[24] - s0_[11]) / a_[33];
  s_[11] = tools_.gating_var_calc(dt, s0_[11], a_[24], a_[33]);

  // s0_[4] is d_L in component L_type_Ca_channel_d_gate (dimensionless).
  a_[35] = (c_[0] == 0.0   ? 1.0 / (1.0 + (exp((-(s0_[0] + 23.1000) / 6.0))))
            : c_[0] == 1.0 ? 1.0 / (1.0 + (exp((-(s0_[0] + 22.3000 + 0.800000 * c_[110]) / 6.0))))
                           : 1.0 / (1.0 + (exp((-(s0_[0] + 22.2000) / 6.0)))));
  a_[3] =
      (c_[0] == 0.0 ? (-28.3800 * (s0_[0] + 35.0)) / ((exp((-(s0_[0] + 35.0) / 2.50000))) - 1.0) -
                          (84.9000 * s0_[0]) / ((exp((-0.208000 * s0_[0]))) - 1.0)
          : c_[0] == 1.0
              ? (-28.3900 * (s0_[0] + 35.0)) / ((exp((-(s0_[0] + 35.0) / 2.50000))) - 1.0) -
                    (84.9000 * s0_[0]) / ((exp((-0.208000 * s0_[0]))) - 1.0)
              : (-28.4000 * (s0_[0] + 35.0)) / ((exp((-(s0_[0] + 35.0) / 2.50000))) - 1.0) -
                    (84.9000 * s0_[0]) / ((exp((-0.208000 * s0_[0]))) - 1.0));
  a_[17] = (c_[0] == 1.0 ? (11.4300 * (s0_[0] - 5.0)) / ((exp((0.400000 * (s0_[0] - 5.0)))) - 1.0)
                         : (11.4200 * (s0_[0] - 5.0)) / ((exp((0.400000 * (s0_[0] - 5.0)))) - 1.0));
  a_[29] = 2.0 / (a_[3] + a_[17]);
  r_[4] = (a_[35] - s0_[4]) / a_[29];
  s_[4] = tools_.gating_var_calc(dt, s0_[4], a_[35], a_[29]);

  // s0_[5] is f_L in component L_type_Ca_channel_f_gate (dimensionless).
  a_[36] = 1.0 / (1.0 + (exp(((s0_[0] + 45.0) / 5.0))));
  a_[4] = (c_[0] == 1.0 ? (3.75000 * (s0_[0] + 28.0)) / ((exp(((s0_[0] + 28.0) / 4.0))) - 1.0)
                        : (3.12000 * (s0_[0] + 28.0)) / ((exp(((s0_[0] + 28.0) / 4.0))) - 1.0));
  a_[18] = (c_[0] == 1.0 ? 30.0000 / (1.0 + (exp((-(s0_[0] + 28.0) / 4.0))))
                         : 25.0000 / (1.0 + (exp((-(s0_[0] + 28.0) / 4.0)))));
  a_[30] =
      (c_[0] == 1.0 ? (1.20000 - 0.200000 * c_[110]) / (a_[4] + a_[18]) : 1.0 / (a_[4] + a_[18]));
  r_[5] = (a_[36] - s0_[5]) / a_[30];
  s_[5] = tools_.gating_var_calc(dt, s0_[5], a_[36], a_[30]);

  // s0_[6] is d_T in component T_type_Ca_channel_d_gate (dimensionless).
  a_[37] = 1.0 / (1.0 + (exp((-(s0_[0] + 37.0) / 6.80000))));
  a_[5] = 1068.00 * (exp(((s0_[0] + 26.3000) / 30.0)));
  a_[19] = 1068.00 * (exp((-(s0_[0] + 26.3000) / 30.0)));
  a_[31] = 1.0 / (a_[5] + a_[19]);
  r_[6] = (a_[37] - s0_[6]) / a_[31];
  s_[6] = tools_.gating_var_calc(dt, s0_[6], a_[37], a_[31]);

  // s0_[7] is f_T in component T_type_Ca_channel_f_gate (dimensionless).
  a_[38] = 1.0 / (1.0 + (exp(((s0_[0] + 71.0) / 9.0))));
  a_[6] = (c_[0] == 1.0 ? 15.3 * (exp((-(s0_[0] + 71.0 + 0.7 * c_[110]) / 83.3)))
                        : 15.3 * (exp((-(s0_[0] + 71.7) / 83.3))));
  a_[20] = (c_[0] == 1.0 ? 15.0 * (exp(((s0_[0] + 71.0) / 15.38)))
                         : 15.0 * (exp(((s0_[0] + 71.7) / 15.38))));
  a_[32] = 1.0 / (a_[6] + a_[20]);
  r_[7] = (a_[38] - s0_[7]) / a_[32];
  s_[7] = tools_.gating_var_calc(dt, s0_[7], a_[38], a_[32]);


  // Compute membrane currents
  // -------------------------
  a_[0] = (c_[0] == 0.0 ? (0.0952 * (exp((-0.063 * (s0_[0] + 34.4))))) /
                                  (1.0 + 1.66 * (exp((-0.225 * (s0_[0] + 63.7))))) +
                              0.0869
                        : (0.09518 * (exp((-0.06306 * (s0_[0] + 34.4))))) /
                                  (1.0 + 1.66200 * (exp((-0.225100 * (s0_[0] + 63.7000))))) +
                              0.0869300);
  a_[13] = (1.0 - a_[0]) * s0_[2] + a_[0] * s0_[3];
  a_[27] =
      ((((c_[117] * (pow(s0_[1], 3.0)) * a_[13] * c_[14] * (pow(c_[5], 2.0))) / (c_[3] * c_[4])) *
           ((exp((((s0_[0] - c_[111]) * c_[5]) / (c_[3] * c_[4])))) - 1.0)) /
          ((exp(((s0_[0] * c_[5]) / (c_[3] * c_[4])))) - 1.0)) *
      s0_[0];
  a_[34] = c_[118] * (s0_[5] * s0_[4] + 0.006 / (1.0 + (exp((-(s0_[0] + 14.1) / 6.0))))) *
           (s0_[0] - c_[21]);
  a_[39] = c_[119] * s0_[6] * s0_[7] * (s0_[0] - c_[28]);
  a_[40] = c_[120] * s0_[8] * s0_[9] * (s0_[0] - c_[112]);
  a_[41] = c_[121] * s0_[9] * (s0_[0] - c_[112]);
  a_[42] = 0.6 * s0_[10] + 0.4 * s0_[11];
  a_[43] = c_[122] * a_[42] * s0_[12] * (s0_[0] - c_[112]);
  a_[44] = c_[123] * (pow(s0_[13], 2.0)) * (s0_[0] - c_[114]);
  a_[45] =
      (c_[0] != 2.0 ? c_[124] * s0_[14] * (s0_[0] - c_[111]) : c_[124] * s0_[14] * (s0_[0] - 77.6));
  a_[46] = (c_[0] != 2.0 ? c_[125] * s0_[14] * (s0_[0] - c_[112])
                         : c_[125] * s0_[14] * (s0_[0] + 102.0));
  a_[47] = c_[126] * (s0_[0] - c_[111]);
  a_[49] = c_[128] * (s0_[0] - c_[115]);
  a_[48] = c_[127] * (s0_[0] - c_[112]);
  a_[50] = (c_[0] == 0.0
                ? (c_[129] * ((pow(c_[91], 3.0)) * c_[93] * (exp((0.03743 * s0_[0] * c_[90]))) -
                                 (pow(c_[14], 3.0)) * c_[92] *
                                     (exp((0.0374 * s0_[0] * (c_[90] - 1.0)))))) /
                      (1.0 + c_[89] * (c_[92] * (pow(c_[14], 3.0)) + c_[93] * (pow(c_[91], 3.0))))
                : (c_[129] * ((pow(c_[91], 3.0)) * c_[93] * (exp((0.0374300 * s0_[0] * c_[90]))) -
                                 (pow(c_[14], 3.0)) * c_[92] *
                                     (exp((0.0374300 * s0_[0] * (c_[90] - 1.0)))))) /
                      (1.0 + c_[89] * (c_[92] * (pow(c_[14], 3.0)) + c_[93] * (pow(c_[91], 3.0)))));
  a_[51] = (c_[130] * (pow((c_[91] / (c_[94] + c_[91])), 3.0)) *
               (pow((c_[102] / (c_[95] + c_[102])), 2.0)) * 1.6) /
           (1.5 + (exp((-(s0_[0] + 60.0) / 40.0))));


  // Compute reaction coefficient (I_Na + I_CaL + I_CaT + I_to + I_sus + I_Kr + I_Ks + I_f_Na +
  // I_f_K + I_b_Na + I_b_Ca + I_b_K + I_NaCa + I_p + I_Ca_p)
  // ---------------------------------------------------------------------------------------------------------------------------------------------------
  r_[0] = (a_[27] + a_[34] + a_[39] + a_[40] + a_[41] + a_[43] + a_[44] + a_[45] + a_[46] + a_[47] +
              a_[49] + a_[48] + a_[50] + a_[51] + c_[132]) /
          c_[116];


  double reacoeff = r_[0];

  return reacoeff;
}

/*----------------------------------------------------------------------*
 |  returns number of internal state variables of the material  cbert 08/13 |
 *----------------------------------------------------------------------*/
int MyocardSanGarny::get_number_of_internal_state_variables() const { return 15; }

/*----------------------------------------------------------------------*
 |  returns current internal state of the material          cbert 08/13 |
 *----------------------------------------------------------------------*/
double MyocardSanGarny::get_internal_state(const int k) const
{
  double val = 0.0;
  if (k == -1 && s0_[0] > -20)
  {
    val = 1;
  }
  val = s0_[k];
  return val;
}

/*----------------------------------------------------------------------*
 |  set  internal state of the material                     cbert 08/13 |
 *----------------------------------------------------------------------*/
void MyocardSanGarny::set_internal_state(const int k, const double val)
{
  s0_[k] = val;
  s_[k] = val;
  return;
}

/*----------------------------------------------------------------------*
 |  returns number of internal state variables of the material  cbert 08/13 |
 *----------------------------------------------------------------------*/
int MyocardSanGarny::get_number_of_ionic_currents() const { return 13; }

/*----------------------------------------------------------------------*
 |  returns current internal currents          cbert 08/13 |
 *----------------------------------------------------------------------*/
double MyocardSanGarny::get_ionic_currents(const int k) const
{
  double val = 0.0;
  switch (k)
  {
    case 0:
      val = a_[27];
      break;
    case 1:
      val = a_[34];
      break;
    case 2:
      val = a_[39];
      break;
    case 3:
      val = a_[40];
      break;
    case 4:
      val = a_[41];
      break;
    case 5:
      val = a_[44];
      break;
    case 6:
      val = a_[45];
      break;
    case 7:
      val = a_[46];
      break;
    case 8:
      val = a_[47];
      break;
    case 9:
      val = a_[48];
      break;
    case 10:
      val = a_[49];
      break;
    case 11:
      val = a_[50];
      break;
    case 12:
      val = a_[51];
      break;
  }

  return val;
}

/*----------------------------------------------------------------------*
 |  update of material at the end of a time step             ljag 07/12 |
 *----------------------------------------------------------------------*/
void MyocardSanGarny::update(const double phi, const double dt)
{
  // update initial values for next time step
  for (int i = 0; i < 16; i++) s0_[i] = s_[i];
}

FOUR_C_NAMESPACE_CLOSE
