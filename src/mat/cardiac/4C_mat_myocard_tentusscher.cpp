// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_myocard_tentusscher.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  variables                                                ljag 09/13 |
 *----------------------------------------------------------------------

   There are a total of 70 entries in the a_ variable array.
   There are a total of 19 entries in each of the rate and state variable arrays.
   There are a total of 53 entries in the constant variable array.

 * VOI_ is time in component environment (millisecond).
 * c_[0] is R in component membrane (joule_per_mole_kelvin).
 * c_[1] is T in component membrane (kelvin).
 * c_[2] is F in component membrane (coulomb_per_millimole).
 * c_[3] is Cm in component membrane (microF).
 * c_[4] is V_c in component membrane (micrometre3).
 * c_[5] is stim_start in component membrane (millisecond).
 * c_[6] is stim_period in component membrane (millisecond).
 * c_[7] is stim_duration in component membrane (millisecond).
 * c_[8] is stim_amplitude in component membrane (picoA_per_picoF).
 * c_[9] is P_kna in component reversal_potentials (dimensionless).
 * c_[10] is K_o in component potassium_dynamics (millimolar).
 * c_[11] is Na_o in component sodium_dynamics (millimolar).
 * c_[12] is Ca_o in component calcium_dynamics (millimolar).
 * c_[13] is g_K1 in component inward_rectifier_potassium_current (nanos0_per_picoF).
 * c_[14] is g_Kr in component rapid_time_dependent_potassium_current (nanos0_per_picoF).
 * c_[15] is g_Ks in component slow_time_dependent_potassium_current (nanos0_per_picoF).
 * c_[16] is g_Na in component fast_sodium_current (nanos0_per_picoF).
 * c_[17] is g_bna in component sodium_background_current (nanos0_per_picoF).
 * c_[18] is g_CaL in component L_type_Ca_current (litre_per_farad_second).
 * c_[19] is g_bca in component calcium_background_current (nanos0_per_picoF).
 * c_[20] is g_to in component transient_outward_current (nanos0_per_picoF).
 * c_[21] is P_NaK in component sodium_potassium_pump_current (picoA_per_picoF).
 * c_[22] is K_mk in component sodium_potassium_pump_current (millimolar).
 * c_[23] is K_mNa in component sodium_potassium_pump_current (millimolar).
 * c_[24] is K_NaCa in component sodium_calcium_exchanger_current (picoA_per_picoF).
 * c_[25] is K_sat in component sodium_calcium_exchanger_current (dimensionless).
 * c_[26] is alpha in component sodium_calcium_exchanger_current (dimensionless).
 * c_[27] is gamma in component sodium_calcium_exchanger_current (dimensionless).
 * c_[28] is Km_Ca in component sodium_calcium_exchanger_current (millimolar).
 * c_[29] is Km_Nai in component sodium_calcium_exchanger_current (millimolar).
 * c_[30] is g_pCa in component calcium_pump_current (picoA_per_picoF).
 * c_[31] is K_pCa in component calcium_pump_current (millimolar).
 * c_[32] is g_pK in component potassium_pump_current (nanos_per_picoF).
 * c_[45] is Buf_c in component calcium_dynamics (millimolar).
 * c_[46] is K_buf_c in component calcium_dynamics (millimolar).
 * c_[47] is Buf_sr in component calcium_dynamics (millimolar).
 * c_[48] is K_buf_sr in component calcium_dynamics (millimolar).
 * c_[49] is Buf_ss in component calcium_dynamics (millimolar).
 * c_[50] is K_buf_ss in component calcium_dynamics (millimolar).
 * c_[51] is V_sr in component calcium_dynamics (micrometre3).
 * c_[52] is V_ss in component calcium_dynamics (micrometre3).
 * c_[33] is k1_prime in component calcium_dynamics (per_millimolar2_per_millisecond).
 * c_[34] is k2_prime in component calcium_dynamics (per_millimolar_per_millisecond).
 * c_[35] is k3 in component calcium_dynamics (per_millisecond).
 * c_[36] is k4 in component calcium_dynamics (per_millisecond).
 * c_[37] is EC in component calcium_dynamics (millimolar).
 * c_[38] is max_sr in component calcium_dynamics (dimensionless).
 * c_[39] is min_sr in component calcium_dynamics (dimensionless).
 * c_[40] is V_rel in component calcium_dynamics (per_millisecond).
 * c_[41] is V_xfer in component calcium_dynamics (per_millisecond).
 * c_[42] is K_up in component calcium_dynamics (millimolar).
 * c_[43] is V_leak in component calcium_dynamics (per_millisecond).
 * c_[44] is Vmax_up in component calcium_dynamics (millimolar_per_millisecond).


 * a_[0] is xr1_inf in component rapid_time_dependent_potassium_current_Xr1_gate (dimensionless).
 * a_[1] is xr2_inf in component rapid_time_dependent_potassium_current_Xr2_gate (dimensionless).
 * a_[2] is xs0_inf in component slow_time_dependent_potassium_current_Xs0_gate (dimensionless).
 * a_[3] is m_inf in component fast_sodium_current_m_gate (dimensionless).
 * a_[4] is h_inf in component fast_sodium_current_h_gate (dimensionless).
 * a_[5] is j_inf in component fast_sodium_current_j_gate (dimensionless).
 * a_[6] is d_inf in component L_type_Ca_current_d_gate (dimensionless).
 * a_[7] is f_inf in component L_type_Ca_current_f_gate (dimensionless).
 * a_[8] is f2_inf in component L_type_Ca_current_f2_gate (dimensionless).
 * a_[9] is fCass0_inf in component L_type_Ca_current_fCass0_gate (dimensionless).
 * a_[10] is s0_inf in component transient_outward_current_s0_gate (dimensionless).
 * a_[11] is r_inf in component transient_outward_current_r_gate (dimensionless).
 * a_[12] is i_Stim in component membrane (picoA_per_picoF).
 * a_[13] is alpha_xr1 in component rapid_time_dependent_potassium_current_Xr1_gate (dimensionless).
 * a_[14] is alpha_xr2 in component rapid_time_dependent_potassium_current_Xr2_gate (dimensionless).
 * a_[15] is alpha_xs in component slow_time_dependent_potassium_current_Xs0_gate (dimensionless).
 * a_[16] is alpha_m in component fast_sodium_current_m_gate (dimensionless).
 * a_[17] is alpha_h in component fast_sodium_current_h_gate (per_millisecond).
 * a_[18] is alpha_j in component fast_sodium_current_j_gate (per_millisecond).
 * a_[19] is alpha_d in component L_type_Ca_current_d_gate (dimensionless).
 * a_[20] is tau_f in component L_type_Ca_current_f_gate (millisecond).
 * a_[21] is tau_f2 in component L_type_Ca_current_f2_gate (millisecond).
 * a_[22] is tau_fCass in component L_type_Ca_current_fCass0_gate (millisecond).
 * a_[23] is tau_s in component transient_outward_current_s0_gate (millisecond).
 * a_[24] is tau_r in component transient_outward_current_r_gate (millisecond).
 * a_[25] is E_Na in component reversal_potentials (millivolt).
 * a_[26] is beta_xr1 in component rapid_time_dependent_potassium_current_Xr1_gate (dimensionless).
 * a_[27] is beta_xr2 in component rapid_time_dependent_potassium_current_Xr2_gate (dimensionless).
 * a_[28] is beta_xs in component slow_time_dependent_potassium_current_Xs0_gate (dimensionless).
 * a_[29] is beta_m in component fast_sodium_current_m_gate (dimensionless).
 * a_[30] is beta_h in component fast_sodium_current_h_gate (per_millisecond).
 * a_[31] is beta_j in component fast_sodium_current_j_gate (per_millisecond).
 * a_[32] is beta_d in component L_type_Ca_current_d_gate (dimensionless).
 * a_[33] is E_K in component reversal_potentials (millivolt).
 * a_[34] is tau_xr1 in component rapid_time_dependent_potassium_current_Xr1_gate (millisecond).
 * a_[35] is tau_xr2 in component rapid_time_dependent_potassium_current_Xr2_gate (millisecond).
 * a_[36] is tau_xs in component slow_time_dependent_potassium_current_Xs0_gate (millisecond).
 * a_[37] is tau_m in component fast_sodium_current_m_gate (millisecond).
 * a_[38] is tau_h in component fast_sodium_current_h_gate (millisecond).
 * a_[39] is tau_j in component fast_sodium_current_j_gate (millisecond).
 * a_[40] is gamma_d in component L_type_Ca_current_d_gate (millisecond).
 * a_[41] is E_Ks in component reversal_potentials (millivolt).
 * a_[42] is tau_d in component L_type_Ca_current_d_gate (millisecond).
 * a_[43] is E_Ca in component reversal_potentials (millivolt).
 * a_[44] is alpha_K1 in component inward_rectifier_potassium_current (dimensionless).
 * a_[45] is beta_K1 in component inward_rectifier_potassium_current (dimensionless).
 * a_[46] is xK1_inf in component inward_rectifier_potassium_current (dimensionless).
 * a_[47] is i_K1 in component inward_rectifier_potassium_current (picoA_per_picoF).
 * a_[48] is i_Kr in component rapid_time_dependent_potassium_current (picoA_per_picoF).
 * a_[49] is i_Ks in component slow_time_dependent_potassium_current (picoA_per_picoF).
 * a_[50] is i_Na in component fast_sodium_current (picoA_per_picoF).
 * a_[51] is i_b_Na in component sodium_background_current (picoA_per_picoF).
 * a_[52] is i_CaL in component L_type_Ca_current (picoA_per_picoF).
 * a_[53] is i_b_Ca in component calcium_background_current (picoA_per_picoF).
 * a_[54] is i_to in component transient_outward_current
 (pihttp://iutam.org/the-pan-american-congress-of-applied-mechanics-pacam-2/coA_per_picoF).
 * a_[55] is i_NaK in component sodium_potassium_pump_current (picoA_per_picoF).
 * a_[56] is i_NaCa in component sodium_calcium_exchanger_current (picoA_per_picoF).
 * a_[57] is i_p_Ca in component calcium_pump_current (picoA_per_picoF).
 * a_[58] is i_p_K in component potassium_pump_current (picoA_per_picoF).
 * a_[59] is i_up in component calcium_dynamics (millimolar_per_millisecond).
 * a_[60] is i_leak in component calcium_dynamics (millimolar_per_millisecond).
 * a_[61] is i_xfer in component calcium_dynamics (millimolar_per_millisecond).
 * a_[62] is kcasr in component calcium_dynamics (dimensionless).
 * a_[63] is Ca_i_bufc in component calcium_dynamics (dimensionless).
 * a_[64] is k1 in component calcium_dynamics (per_millimolar2_per_millisecond).
 * a_[65] is k2 in component calcium_dynamics (per_millimolar_per_millisecond).
 * a_[66] is O in component calcium_dynamics (dimensionless).
 * a_[67] is i_rel in component calcium_dynamics (millimolar_per_millisecond).
 * a_[68] is Ca_sr_bufsr in component calcium_dynamics (dimensionless).
 * a_[69] is Ca_ss0_bufss in component calcium_dynamics (dimensionless).

 * s0_[0] is V in component membrane (millivolt).
 * s0_[1] is K_i in component potassium_dynamics (millimolar).
 * s0_[2] is Na_i in component sodium_dynamics (millimolar).
 * s0_[3] is Ca_i in component calcium_dynamics (millimolar).
 * s0_[4] is Xr1 in component rapid_time_dependent_potassium_current_Xr1_gate (dimensionless).
 * s0_[5] is Xr2 in component rapid_time_dependent_potassium_current_Xr2_gate (dimensionless).
 * s0_[6] is Xs in component slow_time_dependent_potassium_current_Xs0_gate (dimensionless).
 * s0_[7] is m in component fast_sodium_current_m_gate (dimensionless).
 * s0_[8] is h in component fast_sodium_current_h_gate (dimensionless).
 * s0_[9] is j in component fast_sodium_current_j_gate (dimensionless).
 * s0_[10] is Ca_ss in component calcium_dynamics (millimolar).
 * s0_[11] is d in component L_type_Ca_current_d_gate (dimensionless).
 * s0_[12] is f in component L_type_Ca_current_f_gate (dimensionless).
 * s0_[13] is f2 in component L_type_Ca_current_f2_gate (dimensionless).
 * s0_[14] is fCass in component L_type_Ca_current_fCass0_gate (dimensionless).
 * s0_[15] is s in component transient_outward_current_s0_gate (dimensionless).
 * s0_[16] is r in component transient_outward_current_r_gate (dimensionless).
 * s0_[17] is Ca_SR in component calcium_dynamics (millimolar).
 * s0_[18] is R_prime in component calcium_dynamics (dimensionless).


 * r_[0] is d/dt V in component membrane (millivolt).
 * r_[1] is d/dt K_i in component potassium_dynamics (millimolar).
 * r_[2] is d/dt Na_i in component sodium_dynamics (millimolar).
 * r_[3] is d/dt Ca_i in component calcium_dynamics (millimolar).
 * r_[4] is d/dt Xr1 in component rapid_time_dependent_potassium_current_Xr1_gate (dimensionless).
 * r_[5] is d/dt Xr2 in component rapid_time_dependent_potassium_current_Xr2_gate (dimensionless).
 * r_[6] is d/dt Xs in component slow_time_dependent_potassium_current_Xs0_gate (dimensionless).
 * r_[7] is d/dt m in component fast_sodium_current_m_gate (dimensionless).
 * r_[8] is d/dt h in component fast_sodium_current_h_gate (dimensionless).
 * r_[9] is d/dt j in component fast_sodium_current_j_gate (dimensionless).
 * r_[10] is d/dt Ca_ss in component calcium_dynamics (millimolar).
 * r_[11] is d/dt d in component L_type_Ca_current_d_gate (dimensionless).
 * r_[12] is d/dt f in component L_type_Ca_current_f_gate (dimensionless).
 * r_[13] is d/dt f2 in component L_type_Ca_current_f2_gate (dimensionless).
 * r_[14] is d/dt fCass in component L_type_Ca_current_fCass0_gate (dimensionless).
 * r_[15] is d/dt s in component transient_outward_current_s0_gate (dimensionless).
 * r_[16] is d/dt r in component transient_outward_current_r_gate (dimensionless).
 * r_[17] is d/dt Ca_SR in component calcium_dynamics (millimolar).
 * r_[18] is d/dt R_prime in component calcium_dynamics (dimensionless).
 */

/*----------------------------------------------------------------------*
 |  Constructor                                    (public)  cbert 08/13 |
 *----------------------------------------------------------------------*/
MyocardTenTusscher::MyocardTenTusscher() {}


/*----------------------------------------------------------------------*
 |  Constructor                                    (public)  cbert 08/13 |
 *----------------------------------------------------------------------*/
MyocardTenTusscher::MyocardTenTusscher(const double eps_deriv_myocard, const std::string tissue)
    : tools_(), s0_(29, 0.0), s_(29, 0.0), r_(29, 0.0), a_(82, 0.0), c_(63, 0.0)

{
  voi_ = 0.0;
  eps_deriv_ = eps_deriv_myocard;

  if (tissue == "M")
  {
    // initial conditions
    s0_[0] = -85.423;
    s0_[1] = 138.52;
    s0_[2] = 10.132;
    s0_[3] = 0.000153;
    s0_[4] = 0.0165;
    s0_[5] = 0.473;
    s0_[6] = 0.0174;
    s0_[7] = 0.00165;
    s0_[8] = 0.749;
    s0_[9] = 0.6788;
    s0_[10] = 0.00042;
    s0_[11] = 3.288e-5;
    s0_[12] = 0.7026;
    s0_[13] = 0.9526;
    s0_[14] = 0.9942;
    s0_[15] = 0.999998;
    s0_[16] = 2.347e-8;
    s0_[17] = 4.272;
    s0_[18] = 0.8978;

    // Model constants
    c_[0] = 8314.472;
    c_[1] = 310;
    c_[2] = 96485.3415;
    c_[3] = 0.185;
    c_[4] = 0.016404;
    c_[5] = 10;
    c_[6] = 1000;
    c_[7] = 1;
    c_[8] = 52;
    c_[9] = 0.03;
    c_[10] = 5.4;
    c_[11] = 140;
    c_[12] = 2;
    c_[13] = 5.405;
    c_[14] = 0.153;
    c_[15] = 0.098;
    c_[16] = 14.838;
    c_[17] = 0.00029;
    c_[18] = 0.0000398;
    c_[19] = 0.000592;
    c_[20] = 0.294;
    c_[21] = 2.724;
    c_[22] = 1;
    c_[23] = 40;
    c_[24] = 1000;
    c_[25] = 0.1;
    c_[26] = 2.5;
    c_[27] = 0.35;
    c_[28] = 1.38;
    c_[29] = 87.5;
    c_[30] = 0.1238;
    c_[31] = 0.0005;
    c_[32] = 0.0146;
    c_[33] = 0.15;
    c_[34] = 0.045;
    c_[35] = 0.06;
    c_[36] = 0.005;
    c_[37] = 1.5;
    c_[38] = 2.5;
    c_[39] = 1;
    c_[40] = 0.102;
    c_[41] = 0.0038;
    c_[42] = 0.00025;
    c_[43] = 0.00036;
    c_[44] = 0.006375;
    c_[45] = 0.2;
    c_[46] = 0.001;
    c_[47] = 10;
    c_[48] = 0.3;
    c_[49] = 0.4;
    c_[50] = 0.00025;
    c_[51] = 0.001094;
    c_[52] = 0.00005468;
  }
  else if (tissue == "EPI")
  {
    // initial conditions
    s0_[0] = -85.23;
    s0_[1] = 136.89;
    s0_[2] = 8.604;
    s0_[3] = 0.000126;
    s0_[4] = 0.00621;
    s0_[5] = 0.4712;
    s0_[6] = 0.0095;
    s0_[7] = 0.00172;
    s0_[8] = 0.7444;
    s0_[9] = 0.7045;
    s0_[10] = 0.00036;
    s0_[11] = 3.373e-5;
    s0_[12] = 0.7888;
    s0_[13] = 0.9755;
    s0_[14] = 0.9953;
    s0_[15] = 0.999998;
    s0_[16] = 2.42e-8;
    s0_[17] = 3.64;
    s0_[18] = 0.9073;

    // Model constants
    c_[0] = 8314.472;
    c_[1] = 310;
    c_[2] = 96485.3415;
    c_[3] = 0.185;
    c_[4] = 0.016404;
    c_[5] = 10;
    c_[6] = 1000;
    c_[7] = 1;
    c_[8] = 52;
    c_[9] = 0.03;
    c_[10] = 5.4;
    c_[11] = 140;
    c_[12] = 2;
    c_[13] = 5.405;
    c_[14] = 0.153;
    c_[15] = 0.392;
    c_[16] = 14.838;
    c_[17] = 0.00029;
    c_[18] = 0.0000398;
    c_[19] = 0.000592;
    c_[20] = 0.294;
    c_[21] = 2.724;
    c_[22] = 1;
    c_[23] = 40;
    c_[24] = 1000;
    c_[25] = 0.1;
    c_[26] = 2.5;
    c_[27] = 0.35;
    c_[28] = 1.38;
    c_[29] = 87.5;
    c_[30] = 0.1238;
    c_[31] = 0.0005;
    c_[32] = 0.0146;
    c_[33] = 0.15;
    c_[34] = 0.045;
    c_[35] = 0.06;
    c_[36] = 0.005;
    c_[37] = 1.5;
    c_[38] = 2.5;
    c_[39] = 1;
    c_[40] = 0.102;
    c_[41] = 0.0038;
    c_[42] = 0.00025;
    c_[43] = 0.00036;
    c_[44] = 0.006375;
    c_[45] = 0.2;
    c_[46] = 0.001;
    c_[47] = 10;
    c_[48] = 0.3;
    c_[49] = 0.4;
    c_[50] = 0.00025;
    c_[51] = 0.001094;
    c_[52] = 0.00005468;
  }
  else if (tissue == "ENDO")
  {
    // initial conditions
    s0_[0] = -86.709;
    s0_[1] = 138.4;
    s0_[2] = 10.355;
    s0_[3] = 0.00013;
    s0_[4] = 0.00448;
    s0_[5] = 0.476;
    s0_[6] = 0.0087;
    s0_[7] = 0.00155;
    s0_[8] = 0.7573;
    s0_[9] = 0.7225;
    s0_[10] = 0.00036;
    s0_[11] = 3.164e-5;
    s0_[12] = 0.8009;
    s0_[13] = 0.9778;
    s0_[14] = 0.9953;
    s0_[15] = 0.3212;
    s0_[16] = 2.235e-8;
    s0_[17] = 3.715;
    s0_[18] = 0.9068;

    // Model constants
    c_[0] = 8314.472;
    c_[1] = 310;
    c_[2] = 96485.3415;
    c_[3] = 0.185;
    c_[4] = 0.016404;
    c_[5] = 10;
    c_[6] = 1000;
    c_[7] = 1;
    c_[8] = 52;
    c_[9] = 0.03;
    c_[10] = 5.4;
    c_[11] = 140;
    c_[12] = 2;
    c_[13] = 5.405;
    c_[14] = 0.153;
    c_[15] = 0.392;
    c_[16] = 14.838;
    c_[17] = 0.00029;
    c_[18] = 0.0000398;
    c_[19] = 0.000592;
    c_[20] = 0.073;
    c_[21] = 2.724;
    c_[22] = 1;
    c_[23] = 40;
    c_[24] = 1000;
    c_[25] = 0.1;
    c_[26] = 2.5;
    c_[27] = 0.35;
    c_[28] = 1.38;
    c_[29] = 87.5;
    c_[30] = 0.1238;
    c_[31] = 0.0005;
    c_[32] = 0.0146;
    c_[33] = 0.15;
    c_[34] = 0.045;
    c_[35] = 0.06;
    c_[36] = 0.005;
    c_[37] = 1.5;
    c_[38] = 2.5;
    c_[39] = 1;
    c_[40] = 0.102;
    c_[41] = 0.0038;
    c_[42] = 0.00025;
    c_[43] = 0.00036;
    c_[44] = 0.006375;
    c_[45] = 0.2;
    c_[46] = 0.001;
    c_[47] = 10;
    c_[48] = 0.3;
    c_[49] = 0.4;
    c_[50] = 0.00025;
    c_[51] = 0.001094;
    c_[52] = 0.00005468;
  }

  s_ = s0_;
}


double MyocardTenTusscher::rea_coeff(const double phi, const double dt)
{
  s0_[0] = phi;
  s_[0] = phi;

  // Compute new gating variables
  // ----------------------------

  // s_[12] is f in component L_type_Ca_current_f_gate (dimensionless).
  a_[7] = 1.0 / (1.0 + (exp(((s0_[0] + 20.0) / 7.0))));
  a_[20] = 1102.50 * (exp((-(pow((s0_[0] + 27.0), 2.0)) / 225.0))) +
           200.0 / (1.0 + (exp(((13.0 - s0_[0]) / 10.0)))) +
           180.0 / (1.0 + (exp(((s0_[0] + 30.0) / 10.0)))) + 20.0000;
  r_[12] = (a_[7] - s0_[12]) / a_[20];
  s_[12] = tools_.gating_var_calc(dt, s0_[12], a_[7], a_[20]);

  // s_[13] is f2 in component L_type_Ca_current_f2_gate (dimensionless).
  a_[8] = 0.67 / (1.0 + (exp(((s0_[0] + 35.0) / 7.0)))) + 0.33;
  a_[21] = 562.0 * (exp((-(pow((s0_[0] + 27.0), 2.0)) / 240.0))) +
           31.0 / (1.0 + (exp(((25.0 - s0_[0]) / 10.0)))) +
           80.0 / (1.0 + (exp(((s0_[0] + 30.0) / 10.0))));
  r_[13] = (a_[8] - s0_[13]) / a_[21];
  s_[13] = tools_.gating_var_calc(dt, s0_[13], a_[8], a_[21]);

  // s_[14] is fCass in component L_type_Ca_current_fCass_gate (dimensionless).
  a_[9] = 0.6 / (1.0 + (pow((s0_[10] / 0.05), 2.0))) + 0.4;
  a_[22] = 80.0 / (1.0 + (pow((s0_[10] / 0.05), 2.0))) + 2.0;
  r_[14] = (a_[9] - s0_[14]) / a_[22];
  s_[14] = tools_.gating_var_calc(dt, s0_[14], a_[9], a_[22]);

  // s_[15] is s in component transient_outward_current_s_gate (dimensionless).
  a_[10] = 1.0 / (1.0 + (exp(((s0_[0] + 20.0) / 5.0))));
  a_[23] = 85.0 * (exp((-(pow((s0_[0] + 45.0), 2.0)) / 320.0))) +
           5.0 / (1.0 + (exp(((s0_[0] - 20.0) / 5.0)))) + 3.0;
  r_[15] = (a_[10] - s0_[15]) / a_[23];
  s_[15] = tools_.gating_var_calc(dt, s0_[15], a_[10], a_[23]);

  // s_[16] is r in component transient_outward_current_r_gate (dimensionless).
  a_[11] = 1.0 / (1.0 + (exp(((20.0 - s0_[0]) / 6.0))));
  a_[24] = 9.5 * (exp((-(pow((s0_[0] + 40.0), 2.0)) / 1800.0))) + 0.8;
  r_[16] = (a_[11] - s0_[16]) / a_[24];
  s_[16] = tools_.gating_var_calc(dt, s0_[16], a_[11], a_[24]);

  // s_[4] is Xr1 in component rapid_time_dependent_potassium_current_Xr1_gate (dimensionless).
  a_[0] = 1.0 / (1.0 + (exp(((-26.0 - s0_[0]) / 7.0))));
  a_[13] = 450.0 / (1.0 + (exp(((-45.0 - s0_[0]) / 10.0))));
  a_[26] = 6.0 / (1.0 + (exp(((s0_[0] + 30.0) / 11.5))));
  a_[34] = 1.0 * a_[13] * a_[26];
  r_[4] = (a_[0] - s0_[4]) / a_[34];
  s_[4] = tools_.gating_var_calc(dt, s0_[4], a_[0], a_[34]);

  // s_[5] is Xr2 in component rapid_time_dependent_potassium_current_Xr2_gate (dimensionless).
  a_[1] = 1.0 / (1.0 + (exp(((s0_[0] + 88.0) / 24.0))));
  a_[14] = 3.0 / (1.0 + (exp(((-60.0 - s0_[0]) / 20.0))));
  a_[27] = 1.12 / (1.0 + (exp(((s0_[0] - 60.0) / 20.0))));
  a_[35] = 1.0 * a_[14] * a_[27];
  r_[5] = (a_[1] - s0_[5]) / a_[35];
  s_[5] = tools_.gating_var_calc(dt, s0_[5], a_[1], a_[35]);

  // s_[6] is Xs in component slow_time_dependent_potassium_current_Xs_gate (dimensionless).
  a_[2] = 1.0 / (1.0 + (exp(((-5.0 - s0_[0]) / 14.0))));
  a_[15] = 1400.0 / pow((1.0 + (exp(((5.0 - s0_[0]) / 6.0)))), 1.0 / 2);
  a_[28] = 1.0 / (1.0 + (exp(((s0_[0] - 35.0) / 15.0))));
  a_[36] = 1.0 * a_[15] * a_[28] + 80.0;
  r_[6] = (a_[2] - s0_[6]) / a_[36];
  s_[6] = tools_.gating_var_calc(dt, s0_[6], a_[2], a_[36]);

  // s_[7] is m in component fast_sodium_current_m_gate (dimensionless).
  a_[3] = 1.0 / (pow((1.0 + (exp(((-56.86 - s0_[0]) / 9.03)))), 2.0));
  a_[16] = 1.0 / (1.0 + (exp(((-60.0 - s0_[0]) / 5.0))));
  a_[29] =
      0.1 / (1.0 + (exp(((s0_[0] + 35.0) / 5.0)))) + 0.1 / (1.0 + (exp(((s0_[0] - 50.0) / 200.0))));
  a_[37] = 1.0 * a_[16] * a_[29];
  r_[7] = (a_[3] - s0_[7]) / a_[37];
  s_[7] = tools_.gating_var_calc(dt, s0_[7], a_[3], a_[37]);

  // s_[8] is h in component fast_sodium_current_h_gate (dimensionless).
  a_[4] = 1.0 / (pow((1.0 + (exp(((s0_[0] + 71.55) / 7.43)))), 2.0));
  a_[17] = (s0_[0] < -40.0 ? 0.057 * (exp((-(s0_[0] + 80.0) / 6.8))) : 0.0);
  a_[30] = (s0_[0] < -40.0 ? 2.7 * (exp((0.079 * s0_[0]))) + 310000.0 * (exp((0.3485 * s0_[0])))
                           : 0.77 / (0.13 * (1.0 + (exp(((s0_[0] + 10.66) / -11.1))))));
  a_[38] = 1.0 / (a_[17] + a_[30]);
  r_[8] = (a_[4] - s0_[8]) / a_[38];
  s_[8] = tools_.gating_var_calc(dt, s0_[8], a_[4], a_[38]);

  // s_[9] is j in component fast_sodium_current_j_gate (dimensionless).
  a_[5] = 1.0 / (pow((1.0 + (exp(((s0_[0] + 71.55) / 7.43)))), 2.0));
  a_[18] = (s0_[0] < -40.0
                ? (((-25428.0 * (exp((0.2444 * s0_[0]))) - 6.948e-06 * (exp((-0.04391 * s0_[0])))) *
                       (s0_[0] + 37.78)) /
                      1.0) /
                      (1.0 + (exp((0.311000 * (s0_[0] + 79.2300)))))
                : 0.0);
  a_[31] =
      (s0_[0] < -40.0
              ? (0.02424 * (exp((-0.01052 * s0_[0])))) / (1.0 + (exp((-0.1378 * (s0_[0] + 40.14)))))
              : (0.6 * (exp((0.057 * s0_[0])))) / (1.0 + (exp((-0.100000 * (s0_[0] + 32.0))))));
  a_[39] = 1.0 / (a_[18] + a_[31]);
  r_[9] = (a_[5] - s0_[9]) / a_[39];
  s_[9] = tools_.gating_var_calc(dt, s0_[9], a_[5], a_[39]);

  // s_[11] is d in component L_type_Ca_current_d_gate (dimensionless).
  a_[6] = 1.0 / (1.0 + (exp(((-8.0 - s0_[0]) / 7.5))));
  a_[19] = 1.4 / (1.0 + (exp(((-35.0 - s0_[0]) / 13.0)))) + 0.25;
  a_[32] = 1.4 / (1.0 + (exp(((s0_[0] + 5.0) / 5.0))));
  a_[40] = 1.0 / (1.0 + (exp(((50.0 - s0_[0]) / 20.0))));
  a_[42] = 1.0 * a_[19] * a_[32] + a_[40];
  r_[11] = (a_[6] - s0_[11]) / a_[42];
  s_[11] = tools_.gating_var_calc(dt, s0_[11], a_[6], a_[42]);

  // Compute membrane currents
  // -------------------------
  a_[55] = ((((c_[21] * c_[10]) / (c_[10] + c_[22])) * s0_[2]) / (s0_[2] + c_[23])) /
           (1.0 + 0.1245 * (exp(((-0.1 * s_[0] * c_[2]) / (c_[0] * c_[1])))) +
               0.0353000 * (exp(((-s_[0] * c_[2]) / (c_[0] * c_[1])))));
  a_[25] = ((c_[0] * c_[1]) / c_[2]) * (log((c_[11] / s0_[2])));
  a_[50] = c_[16] * (pow(s_[7], 3.0)) * s_[8] * s_[9] * (s_[0] - a_[25]);
  a_[51] = c_[17] * (s_[0] - a_[25]);
  a_[56] =
      (c_[24] * ((exp(((c_[27] * s_[0] * c_[2]) / (c_[0] * c_[1])))) * (pow(s0_[2], 3.0)) * c_[12] -
                    (exp((((c_[27] - 1.0) * s_[0] * c_[2]) / (c_[0] * c_[1])))) *
                        (pow(c_[11], 3.0)) * s_[3] * c_[26])) /
      (((pow(c_[29], 3.0)) + (pow(c_[11], 3.0))) * (c_[28] + c_[12]) *
          (1.0 + c_[25] * (exp((((c_[27] - 1.0) * s_[0] * c_[2]) / (c_[0] * c_[1]))))));

  // s_[2] is Na_i in component sodium_dynamics (millimolar).
  r_[2] =
      ((-1.0 * (a_[50] + a_[51] + 3.0 * a_[55] + 3.0 * a_[56])) / (1.0 * c_[4] * c_[2])) * c_[3];
  s_[2] = tools_.gating_var_calc(dt, s0_[2], 0, -s0_[2] / r_[2]);

  a_[33] = ((c_[0] * c_[1]) / c_[2]) * (log((c_[10] / s0_[1])));
  a_[44] = 0.1 / (1.0 + (exp((0.06 * ((s0_[0] - a_[33]) - 200.0)))));
  a_[45] = (3.0 * (exp((0.0002 * ((s0_[0] - a_[33]) + 100.0)))) +
               (exp((0.1 * ((s0_[0] - a_[33]) - 10.0))))) /
           (1.0 + (exp((-0.5 * (s0_[0] - a_[33])))));
  a_[46] = a_[44] / (a_[44] + a_[45]);
  a_[47] = c_[13] * a_[46] * pow((c_[10] / 5.4), 1.0 / 2) * (s_[0] - a_[33]);
  a_[54] = c_[20] * s_[16] * s_[15] * (s_[0] - a_[33]);
  a_[48] = c_[14] * pow((c_[10] / 5.4), 1.0 / 2) * s_[4] * s_[5] * (s_[0] - a_[33]);
  a_[41] =
      ((c_[0] * c_[1]) / c_[2]) * (log(((c_[10] + c_[9] * c_[11]) / (s0_[1] + c_[9] * s_[2]))));
  a_[49] = c_[15] * (pow(s0_[6], 2.0)) * (s0_[0] - a_[41]);
  a_[52] =
      (((c_[18] * s_[11] * s_[12] * s_[13] * s_[14] * 4.0 * (s_[0] - 15.0) * (pow(c_[2], 2.0))) /
           (c_[0] * c_[1])) *
          (0.25 * s_[10] * (exp(((2.0 * (s_[0] - 15.0) * c_[2]) / (c_[0] * c_[1])))) - c_[12])) /
      ((exp(((2.0 * (s_[0] - 15.0) * c_[2]) / (c_[0] * c_[1])))) - 1.0);
  a_[43] = ((0.5 * c_[0] * c_[1]) / c_[2]) * (log((c_[12] / s0_[3])));
  a_[53] = c_[19] * (s_[0] - a_[43]);
  a_[58] = (c_[32] * (s_[0] - a_[33])) / (1.0 + (exp(((25.0 - s_[0]) / 5.98))));
  a_[57] = (c_[30] * s_[3]) / (s_[3] + c_[31]);
  a_[12] = 0.0;  //(VOI_ -  (floor((VOI_/c_[6])))*c_[6]>=c_[5]&&VOI_ -
                 //(floor((VOI_/c_[6])))*c_[6]<=c_[5]+c_[7] ? - c_[8] : 0.0); // EXTERNAL STIMULUS

  // Compute reaction coefficient (I_K1 + I_to + I_Kr + I_Ks + I_CaL + I_NaK + I_Na + I_b_Na +
  // I_NaCa + I_b_Ca + I_p_K + I_stim)
  // ---------------------------------------------------------------------------------------------------------------------------
  r_[0] = (a_[47] + a_[54] + a_[48] + a_[49] + a_[52] + a_[55] + a_[50] + a_[51] + a_[56] + a_[53] +
           a_[58] + a_[57] + a_[12]);

  // s_[1] is K_i in component potassium_dynamics (millimolar).
  r_[1] = ((-1.0 * ((a_[47] + a_[54] + a_[48] + a_[49] + a_[58] + a_[12]) - 2.0 * a_[55])) /
              (1.0 * c_[4] * c_[2])) *
          c_[3];
  s_[1] = tools_.gating_var_calc(dt, s0_[1], 0, -s0_[1] / r_[1]);

  // s_[3] is Ca_i in component calcium_dynamics (millimolar).
  a_[59] = c_[44] / (1.0 + (pow(c_[42], 2.0)) / (pow(s0_[3], 2.0)));
  a_[60] = c_[43] * (s0_[17] - s0_[3]);
  a_[61] = c_[41] * (s0_[10] - s0_[3]);
  a_[63] = 1.0 / (1.0 + (c_[45] * c_[46]) / (pow((s0_[3] + c_[46]), 2.0)));
  r_[3] = a_[63] *
          ((((a_[60] - a_[59]) * c_[51]) / c_[4] + a_[61]) -
              (1.0 * ((a_[53] + a_[57]) - 2.0 * a_[56]) * c_[3]) / (2.0 * 1.0 * c_[4] * c_[2]));
  s_[3] = tools_.gating_var_calc(dt, s0_[3], 0, -s0_[3] / r_[3]);

  // s_[18] is R_prime in component calcium_dynamics (dimensionless).
  a_[62] = c_[38] - (c_[38] - c_[39]) / (1.0 + (pow((c_[37] / s0_[17]), 2.0)));
  a_[65] = c_[34] * a_[62];
  r_[18] = -a_[65] * s0_[10] * s0_[18] + c_[36] * (1.0 - s0_[18]);
  s_[18] = tools_.gating_var_calc(dt, s0_[18], 0, -s0_[18] / r_[18]);

  // s0_[17] is Ca_SR in component calcium_dynamics (millimolar).
  a_[64] = c_[33] / a_[62];
  a_[66] = (a_[64] * (pow(s0_[10], 2.0)) * s0_[18]) / (c_[35] + a_[64] * (pow(s0_[10], 2.0)));
  a_[67] = c_[40] * a_[66] * (s0_[17] - s0_[10]);
  a_[68] = 1.0 / (1.0 + (c_[47] * c_[48]) / (pow((s0_[17] + c_[48]), 2.0)));
  r_[17] = a_[68] * (a_[59] - (a_[67] + a_[60]));
  s_[17] = tools_.gating_var_calc(dt, s0_[17], 0, -s0_[17] / r_[17]);

  // s_[10] is Ca_ss in component calcium_dynamics (millimolar).
  a_[69] = 1.0 / (1.0 + (c_[49] * c_[50]) / (pow((s0_[10] + c_[50]), 2.0)));
  r_[10] = a_[69] *
           (((-1.0 * a_[52] * c_[3]) / (2.0 * 1.0 * c_[52] * c_[2]) + (a_[67] * c_[51]) / c_[52]) -
               (a_[61] * c_[4]) / c_[52]);
  s_[10] = tools_.gating_var_calc(dt, s0_[10], 0, -s0_[10] / r_[10]);

  // Update rest of state variables
  // ------------------------------
  // s_[1] = s0_[1] + dt * r_[1];
  // s_[2] = s0_[2] + dt * r_[2];
  // s_[3] = s0_[3] + dt * r_[3];
  // s_[10] = s0_[10] + dt * r_[10];
  // s_[17] = s0_[17] + dt * r_[17];
  // s_[18] = s0_[18] + dt * r_[18];

  // for (int i=0; i<19; i++)
  //  s_[i] = s0_[i] + dt * r_[i];


  double reacoeff = r_[0];

  return reacoeff;
}

/*----------------------------------------------------------------------*
 |  returns number of internal state variables of the material  cbert 08/13 |
 *----------------------------------------------------------------------*/
int MyocardTenTusscher::get_number_of_internal_state_variables() const { return 19; }

/*----------------------------------------------------------------------*
 |  returns current internal state of the material          cbert 08/13 |
 *----------------------------------------------------------------------*/
double MyocardTenTusscher::get_internal_state(const int k) const
{
  double val = 0.0;
  if (k == -1)
  {
    val = s0_[3];  // Free cytoplasmatic calcium concentration
  }
  else
  {
    val = s0_[k];
  }
  return val;
}

/*----------------------------------------------------------------------*
 |  set  internal state of the material                     cbert 08/13 |
 *----------------------------------------------------------------------*/
void MyocardTenTusscher::set_internal_state(const int k, const double val)
{
  if (k == -1)
  {
    s0_[3] = val;  // Free cytoplasmatic calcium concentration
    s_[3] = val;
  }
  else
  {
    s0_[k] = val;
    s_[k] = val;
  }
  return;
}

/*----------------------------------------------------------------------*
 |  returns number of internal state variables of the material  cbert 08/13 |
 *----------------------------------------------------------------------*/
int MyocardTenTusscher::get_number_of_ionic_currents() const { return 15; }

/*----------------------------------------------------------------------*
 |  returns current internal currents          cbert 08/13 |
 *----------------------------------------------------------------------*/
double MyocardTenTusscher::get_ionic_currents(const int k) const
{
  double val = 0.0;
  val = a_[47 + k];
  return val;
}

/*----------------------------------------------------------------------*
 |  update of material at the end of a time step             ljag 07/12 |
 *----------------------------------------------------------------------*/
void MyocardTenTusscher::update(const double phi, const double dt)
{
  // update initial values for next time step
  for (int i = 0; i < 19; i++) s0_[i] = s_[i];
  voi_ += dt;

  return;
}

FOUR_C_NAMESPACE_CLOSE
