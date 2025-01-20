// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_myocard_inada.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  variables                                                ljag 09/13 |
 *----------------------------------------------------------------------
   There are a total of 82 entries in the algebraic variable array.
   There are a total of 29 entries in each of the rate and state variable arrays.
   There are a total of 63 entries in the constant variable array.


 * VOI is time in component environment (second).
 *
 * c_[0] is R in component membrane (joule_per_kilomole_kelvin).
 * c_[1] is T in component membrane (kelvin).
 * c_[2] is F in component membrane (coulomb_per_mole).
 * c_[3] is C in component membrane (microF).
 * c_[4] is stim_start in component membrane (second).
 * c_[5] is stim_end in component membrane (second).
 * c_[6] is stim_period in component membrane (second).
 * c_[7] is stim_duration in component membrane (second).
 * c_[8] is stim_amplitude in component membrane (nanoA).
 * c_[9] is g_f in component hyperpolarising_activated_current (microS).
 * c_[10] is ACh in component acetylcholine_sensitive_current (millimolar).
 * c_[11] is g_Kr in component rapid_delayed_rectifier_potassium_current (microS).
 * c_[12] is Ki in component intracellular_potassium_concentration (millimolar).
 * c_[13] is Kc in component extracellular_potassium_concentration (millimolar).
 * c_[14] is g_K1 in component time_independent_potassium_current (microS).
 * c_[15] is g_b in component background_current (microS).
 * c_[16] is E_b in component background_current (millivolt).
 * c_[17] is I_p in component sodium_potassium_pump (nanoA).
 * c_[18] is Nai in component intracellular_sodium_concentration (millimolar).
 * c_[19] is kNaCa in component sodium_calcium_exchange_current (nanoA).
 * c_[20] is Qci in component sodium_calcium_exchange_current (dimensionless).
 * c_[21] is Qn in component sodium_calcium_exchange_current (dimensionless).
 * c_[22] is Qco in component sodium_calcium_exchange_current (dimensionless).
 * c_[23] is Kci in component sodium_calcium_exchange_current (millimolar).
 * c_[24] is K1ni in component sodium_calcium_exchange_current (millimolar).
 * c_[25] is K2ni in component sodium_calcium_exchange_current (millimolar).
 * c_[26] is K3ni in component sodium_calcium_exchange_current (millimolar).
 * c_[27] is Kcni in component sodium_calcium_exchange_current (millimolar).
 * c_[28] is K3no in component sodium_calcium_exchange_current (millimolar).
 * c_[29] is K1no in component sodium_calcium_exchange_current (millimolar).
 * c_[30] is K2no in component sodium_calcium_exchange_current (millimolar).
 * c_[31] is Kco in component sodium_calcium_exchange_current (millimolar).
 * c_[32] is Cao in component extracellular_calcium_concentration (millimolar).
 * c_[33] is Nao in component extracellular_sodium_concentration (millimolar).
 * c_[34] is g_Na in component fast_sodium_current (microlitre_per_second).
 * c_[35] is delta_m in component fast_sodium_current_m_gate (millivolt).
 * c_[36] is g_CaL in component L_type_calcium_current (microS).
 * c_[37] is E_CaL in component L_type_calcium_current (millivolt).
 * c_[38] is act_shift in component L_type_calcium_current_d_gate (millivolt).
 * c_[39] is slope_factor_act in component L_type_calcium_current_d_gate (millivolt).
 * c_[40] is inact_shift in component L_type_calcium_current_f_gate (millivolt).
 * c_[41] is inact_shift in component L_type_calcium_current_f2_gate (millivolt).
 * c_[42] is g_to in component transient_outward_potassium_current (microS).
 * c_[43] is E_st in component sustained_outward_potassium_current (millivolt).
 * c_[44] is g_st in component sustained_outward_potassium_current (microS).
 * c_[45] is g_ACh_max in component acetylcholine_sensitive_current (microS).
 * c_[46] is K_ACh in component acetylcholine_sensitive_current (millimolar).
 * c_[47] is alpha_achf in component acetylcholine_sensitive_current_achf_gate (per_second).
 * c_[48] is alpha_achs in component acetylcholine_sensitive_current_achs0_gate (per_second).
 * c_[49] is V_cell in component intracellular_calcium_concentration (micrometre3).
 * c_[50] is P_rel in component intracellular_calcium_concentration (per_second).
 * c_[51] is K_up in component intracellular_calcium_concentration (millimolar).
 * c_[52] is tau_tr in component intracellular_calcium_concentration (second).
 * c_[53] is RTONF in component membrane (millivolt).
 * c_[54] is k34 in component sodium_calcium_exchange_current (dimensionless).
 * c_[55] is V_up in component intracellular_calcium_concentration (micrometre3).
 * c_[56] is V_rel in component intracellular_calcium_concentration (micrometre3).
 * c_[57] is V_sub in component intracellular_calcium_concentration (micrometre3).
 * c_[58] is E_K in component rapid_delayed_rectifier_potassium_current (millivolt).
 * c_[59] is k43 in component sodium_calcium_exchange_current (dimensionless).
 * c_[60] is E_Na in component fast_sodium_current (millivolt).
 * c_[61] is E_K in component transient_outward_potassium_current (millivolt).
 * c_[62] is Vi in component intracellular_calcium_concentration (micrometre3).

 * a_[0] is y_inf in component hyperpolarising_activated_current_y_gate (dimensionless).
 * a_[1] is paf_infinity in component rapid_delayed_rectifier_potassium_current_paf_gate
 (dimensionless).
 * a_[2] is pas0_infinity in component rapid_delayed_rectifier_potassium_current_pas0_gate
 (dimensionless).
 * a_[3] is pik_infinity in component rapid_delayed_rectifier_potassium_current_pik_gate
 (dimensionless).
 * a_[4] is E0_m in component fast_sodium_current_m_gate (millivolt).
 * a_[5] is alpha_h1 in component fast_sodium_current_h1_gate (per_second).
 * a_[6] is alpha_h2 in component fast_sodium_current_h2_gate (per_second).
 * a_[7] is alpha_d in component L_type_calcium_current_d_gate (per_second).
 * a_[8] is f_inf in component L_type_calcium_current_f_gate (dimensionless).
 * a_[9] is f2_inf in component L_type_calcium_current_f2_gate (dimensionless).
 * a_[10] is r_infinity in component transient_outward_potassium_current_r_gate (dimensionless).
 * a_[11] is qfast_infinity in component transient_outward_potassium_current_qfast_gate
 (dimensionless).
 * a_[12] is qslow_infinity in component transient_outward_potassium_current_qslow_gate
 (dimensionless).
 * a_[13] is qa_infinity in component sustained_outward_potassium_current_qa_gate (dimensionless).
 * a_[14] is alpha_qi in component sustained_outward_potassium_current_qi_gate (per_second).
 * a_[15] is beta_achf in component acetylcholine_sensitive_current_achf_gate (per_second).
 * a_[16] is beta_achs in component acetylcholine_sensitive_current_achs0_gate (per_second).
 * a_[17] is i_Stim in component membrane (nanoA).
 * a_[18] is diff_f_TMM in component intracellular_calcium_concentration (per_second).
 * a_[19] is tau_y in component hyperpolarising_activated_current_y_gate (second).
 * a_[20] is tau_paf in component rapid_delayed_rectifier_potassium_current_paf_gate (second).
 * a_[21] is tau_pas in component rapid_delayed_rectifier_potassium_current_pas0_gate (second).
 * a_[22] is alpha_pik in component rapid_delayed_rectifier_potassium_current_pik_gate (per_second).
 * a_[23] is alpha_m in component fast_sodium_current_m_gate (per_second).
 * a_[24] is beta_h1 in component fast_sodium_current_h1_gate (per_second).
 * a_[25] is beta_h2 in component fast_sodium_current_h2_gate (per_second).
 * a_[26] is beta_d in component L_type_calcium_current_d_gate (per_second).
 * a_[27] is tau_f in component L_type_calcium_current_f_gate (second).
 * a_[28] is tau_f2 in component L_type_calcium_current_f2_gate (second).
 * a_[29] is tau_r in component transient_outward_potassium_current_r_gate (second).
 * a_[30] is tau_qfast in component transient_outward_potassium_current_qfast_gate (second).
 * a_[31] is tau_qslow in component transient_outward_potassium_current_qslow_gate (second).
 * a_[32] is alpha_qa in component sustained_outward_potassium_current_qa_gate (per_second).
 * a_[33] is beta_qi in component sustained_outward_potassium_current_qi_gate (per_second).
 * a_[34] is i_f in component hyperpolarising_activated_current (nanoA).
 * a_[35] is beta_pik in component rapid_delayed_rectifier_potassium_current_pik_gate (per_second).
 * a_[36] is beta_m in component fast_sodium_current_m_gate (per_second).
 * a_[37] is h1_inf in component fast_sodium_current_h1_gate (dimensionless).
 * a_[38] is h2_inf in component fast_sodium_current_h2_gate (dimensionless).
 * a_[39] is d_inf in component L_type_calcium_current_d_gate (dimensionless).
 * a_[40] is beta_qa in component sustained_outward_potassium_current_qa_gate (per_second).
 * a_[41] is qi_infinity in component sustained_outward_potassium_current_qi_gate (dimensionless).
 * a_[42] is i_Kr in component rapid_delayed_rectifier_potassium_current (nanoA).
 * a_[43] is tau_pik in component rapid_delayed_rectifier_potassium_current_pik_gate (second).
 * a_[44] is tau_h1 in component fast_sodium_current_h1_gate (second).
 * a_[45] is tau_h2 in component fast_sodium_current_h2_gate (second).
 * a_[46] is tau_d in component L_type_calcium_current_d_gate (second).
 * a_[47] is tau_qa in component sustained_outward_potassium_current_qa_gate (second).
 * a_[48] is tau_qi in component sustained_outward_potassium_current_qi_gate (second).
 * a_[49] is g_K1_prime in component time_independent_potassium_current (microS).
 * a_[50] is i_K1 in component time_independent_potassium_current (nanoA).
 * a_[51] is i_b in component background_current (nanoA).
 * a_[52] is i_p in component sodium_potassium_pump (nanoA).
 * a_[53] is do in component sodium_calcium_exchange_current (dimensionless).
 * a_[54] is k32 in component sodium_calcium_exchange_current (dimensionless).
 * a_[55] is k23 in component sodium_calcium_exchange_current (dimensionless).
 * a_[56] is k21 in component sodium_calcium_exchange_current (dimensionless).
 * a_[57] is di in component sodium_calcium_exchange_current (dimensionless).
 * a_[58] is k41 in component sodium_calcium_exchange_current (dimensionless).
 * a_[59] is k14 in component sodium_calcium_exchange_current (dimensionless).
 * a_[60] is k12 in component sodium_calcium_exchange_current (dimensionless).
 * a_[61] is x1 in component sodium_calcium_exchange_current (dimensionless).
 * a_[62] is x2 in component sodium_calcium_exchange_current (dimensionless).
 * a_[63] is x3 in component sodium_calcium_exchange_current (dimensionless).
 * a_[64] is x4 in component sodium_calcium_exchange_current (dimensionless).
 * a_[65] is i_NaCa in component sodium_calcium_exchange_current (nanoA).
 * a_[66] is i_Na in component fast_sodium_current (nanoA).
 * a_[67] is i_to in component transient_outward_potassium_current (nanoA).
 * a_[68] is i_st in component sustained_outward_potassium_current (nanoA).
 * a_[69] is g_ACh in component acetylcholine_sensitive_current (microS).
 * a_[70] is i_ACh in component acetylcholine_sensitive_current (nanoA).
 * a_[71] is i_CaL in component L_type_calcium_current (nanoA).
 * a_[72] is i_diff in component intracellular_calcium_concentration (millimolar_per_second).
 * a_[73] is i_up in component intracellular_calcium_concentration (millimolar_per_second).
 * a_[74] is i_tr in component intracellular_calcium_concentration (millimolar_per_second).
 * a_[75] is diff_f_TC in component intracellular_calcium_concentration (per_second).
 * a_[76] is i_rel in component intracellular_calcium_concentration (millimolar_per_second).
 * a_[77] is diff_f_TMC in component intracellular_calcium_concentration (per_second).
 * a_[78] is diff_f_CMi in component intracellular_calcium_concentration (per_second).
 * a_[79] is diff_f_CMs in component intracellular_calcium_concentration (per_second).
 * a_[80] is diff_f_CQ in component intracellular_calcium_concentration (per_second).
 * a_[81] is diff_f_CSL in component intracellular_calcium_concentration (per_second).

 * s0_[0] is V in component membrane (millivolt).
 * s0_[1] is y in component hyperpolarising_activated_current_y_gate (dimensionless).
 * s0_[2] is paf in component rapid_delayed_rectifier_potassium_current_paf_gate (dimensionless).
 * s0_[3] is pas in component rapid_delayed_rectifier_potassium_current_pas0_gate (dimensionless).
 * s0_[4] is pik in component rapid_delayed_rectifier_potassium_current_pik_gate (dimensionless).
 * s0_[5] is Casub in component intracellular_calcium_concentration (millimolar).
 * s0_[6] is m in component fast_sodium_current_m_gate (dimensionless).
 * s0_[7] is h1 in component fast_sodium_current_h1_gate (dimensionless).
 * s0_[8] is h2 in component fast_sodium_current_h2_gate (dimensionless).
 * s0_[9] is d in component L_type_calcium_current_d_gate (dimensionless).
 * s0_[10] is f in component L_type_calcium_current_f_gate (dimensionless).
 * s0_[11] is f2 in component L_type_calcium_current_f2_gate (dimensionless).
 * s0_[12] is r in component transient_outward_potassium_current_r_gate (dimensionless).
 * s0_[13] is q_fast in component transient_outward_potassium_current_qfast_gate (dimensionless).
 * s0_[14] is q_slow in component transient_outward_potassium_current_qslow_gate (dimensionless).
 * s0_[15] is qa in component sustained_outward_potassium_current_qa_gate (dimensionless).
 * s0_[16] is qi in component sustained_outward_potassium_current_qi_gate (dimensionless).
 * s0_[17] is achf in component acetylcholine_sensitive_current_achf_gate (dimensionless).
 * s0_[18] is achs in component acetylcholine_sensitive_current_achs0_gate (dimensionless).
 * s0_[19] is Cai in component intracellular_calcium_concentration (millimolar).
 * s0_[20] is Ca_up in component intracellular_calcium_concentration (millimolar).
 * s0_[21] is Ca_rel in component intracellular_calcium_concentration (millimolar).
 * s0_[22] is f_TC in component intracellular_calcium_concentration (dimensionless).
 * s0_[23] is f_TMC in component intracellular_calcium_concentration (dimensionless).
 * s0_[24] is f_TMM in component intracellular_calcium_concentration (dimensionless).
 * s0_[25] is f_CMi in component intracellular_calcium_concentration (dimensionless).
 * s0_[26] is f_CMs in component intracellular_calcium_concentration (dimensionless).
 * s0_[27] is f_CQ in component intracellular_calcium_concentration (dimensionless).
 * s0_[28] is f_CSL in component intracellular_calcium_concentration (dimensionless).

 * r_[0] is d/dt V in component membrane (millivolt).
 * r_[1] is d/dt y in component hyperpolarising_activated_current_y_gate (dimensionless).
 * r_[2] is d/dt paf in component rapid_delayed_rectifier_potassium_current_paf_gate
 (dimensionless).
 * r_[3] is d/dt pas in component rapid_delayed_rectifier_potassium_current_pas0_gate
 (dimensionless).
 * r_[4] is d/dt pik in component rapid_delayed_rectifier_potassium_current_pik_gate
 (dimensionless).
 * r_[5] is d/dt Casub in component intracellular_calcium_concentration (millimolar).
 * r_[6] is d/dt m in component fast_sodium_current_m_gate (dimensionless).
 * r_[7] is d/dt h1 in component fast_sodium_current_h1_gate (dimensionless).
 * r_[8] is d/dt h2 in component fast_sodium_current_h2_gate (dimensionless).
 * r_[9] is d/dt d in component L_type_calcium_current_d_gate (dimensionless).
 * r_[10] is d/dt f in component L_type_calcium_current_f_gate (dimensionless).
 * r_[11] is d/dt f2 in component L_type_calcium_current_f2_gate (dimensionless).
 * r_[12] is d/dt r in component transient_outward_potassium_current_r_gate (dimensionless).
 * r_[13] is d/dt q_fast in component transient_outward_potassium_current_qfast_gate
 (dimensionless).
 * r_[14] is d/dt q_slow in component transient_outward_potassium_current_qslow_gate
 (dimensionless).
 * r_[15] is d/dt qa in component sustained_outward_potassium_current_qa_gate (dimensionless).
 * r_[16] is d/dt qi in component sustained_outward_potassium_current_qi_gate (dimensionless).
 * r_[17] is d/dt achf in component acetylcholine_sensitive_current_achf_gate (dimensionless).
 * r_[18] is d/dt achs in component acetylcholine_sensitive_current_achs0_gate (dimensionless).
 * r_[19] is d/dt Cai in component intracellular_calcium_concentration (millimolar).
 * r_[20] is d/dt Ca_up in component intracellular_calcium_concentration (millimolar).
 * r_[21] is d/dt Ca_rel in component intracellular_calcium_concentration (millimolar).
 * r_[22] is d/dt f_TC in component intracellular_calcium_concentration (dimensionless).
 * r_[23] is d/dt f_TMC in component intracellular_calcium_concentration (dimensionless).
 * r_[24] is d/dt f_TMM in component intracellular_calcium_concentration (dimensionless).
 * r_[25] is d/dt f_CMi in component intracellular_calcium_concentration (dimensionless).
 * r_[26] is d/dt f_CMs in component intracellular_calcium_concentration (dimensionless).
 * r_[27] is d/dt f_CQ in component intracellular_calcium_concentration (dimensionless).
 * r_[28] is d/dt f_CSL in component intracellular_calcium_concentration (dimensionless).
 */


/*----------------------------------------------------------------------*
 |  Constructor                                    (public)  cbert 08/13 |
 *----------------------------------------------------------------------*/
MyocardInada::MyocardInada() {}


/*----------------------------------------------------------------------*
 |  Constructor                                    (public)  cbert 08/13 |
 *----------------------------------------------------------------------*/
MyocardInada::MyocardInada(const double eps0_deriv_myocard, const std::string tissue)
    : tools_(), s0_(29, 0.0), s_(29, 0.0), r_(29, 0.0), a_(82, 0.0), c_(63, 0.0)

{
  voi_ = 0;
  eps0_deriv_ = eps0_deriv_myocard;

  if (tissue == "AN")
  {
    // Initial conditions
    s0_[0] = -71.5535452525735;
    s0_[1] = 0.231892950445813;
    s0_[2] = 0.000907827363439979;
    s0_[3] = 0.00289901267127429;
    s0_[4] = 0.987889953123897;
    s0_[5] = 2.86962804165375e-5;
    s0_[6] = 0.0104794040295793;
    s0_[7] = 0.792210965943567;
    s0_[8] = 0.78834840378483;
    s0_[9] = 3.2286733432613e-5;
    s0_[10] = 0.998822085003546;
    s0_[11] = 0.998815467202695;
    s0_[12] = 0.00802880907824072;
    s0_[13] = 0.995494395556732;
    s0_[14] = 0.547966933708077;
    s0_[15] = 0.0758461021803425;
    s0_[16] = 0.943345608766177;
    s0_[17] = 0.760265641624297;
    s0_[18] = 0.764664867332735;
    s0_[19] = 3.10430405261017e-5;
    s0_[20] = 0.667220893111124;
    s0_[21] = 0.557458998353581;
    s0_[22] = 0.0061557815047835;
    s0_[23] = 0.112200926320798;
    s0_[24] = 0.784309464255762;
    s0_[25] = 0.012895686161953;
    s0_[26] = 0.0119201823967662;
    s0_[27] = 0.400684024879925;
    s0_[28] = 8.91124266400812e-6;

    s_ = s0_;

    // Model constants
    c_[0] = 8314.472;
    c_[1] = 310;
    c_[2] = 96485.3415;
    c_[3] = 4e-5;
    c_[4] = 0.05;
    c_[5] = 99999;
    c_[6] = 1;
    c_[7] = 0.001;
    c_[8] = -2;
    c_[9] = 0;
    c_[10] = 0;
    c_[11] = 0.0015;
    c_[12] = 140;
    c_[13] = 5.4;
    c_[14] = 0.0125;
    c_[15] = 0.0018;
    c_[16] = -52.5;
    c_[17] = 0.0246;
    c_[18] = 8;
    c_[19] = 5.916;
    c_[20] = 0.1369;
    c_[21] = 0.4315;
    c_[22] = 0;
    c_[23] = 0.0207;
    c_[24] = 395.3;
    c_[25] = 2.289;
    c_[26] = 26.44;
    c_[27] = 26.44;
    c_[28] = 4.663;
    c_[29] = 1628;
    c_[30] = 561.4;
    c_[31] = 3.663;
    c_[32] = 2;
    c_[33] = 140;
    c_[34] = 5e-7;
    c_[35] = 1e-5;
    c_[36] = 0.0185;
    c_[37] = 62.5;
    c_[38] = 0;
    c_[39] = -6.61;
    c_[40] = -5;
    c_[41] = -5;
    c_[42] = 0.02;
    c_[43] = -37.4;
    c_[44] = 0;
    c_[45] = 0.0198;
    c_[46] = 0.00035;
    c_[47] = 73.1;
    c_[48] = 3.7;
    c_[49] = 4.39823e-6;
    c_[50] = 1805.6;
    c_[51] = 0.0006;
    c_[52] = 0.06;
    c_[53] = (c_[0] * c_[1]) / c_[2];
    c_[54] = c_[33] / (c_[28] + c_[33]);
    c_[55] = 0.0116000 * c_[49];
    c_[56] = 0.00120000 * c_[49];
    c_[57] = 0.0100000 * c_[49];
    c_[58] = c_[53] * (log((c_[13] / c_[12])));
    c_[59] = c_[18] / (c_[26] + c_[18]);
    c_[60] = c_[53] * (log((c_[33] / c_[18])));
    c_[61] = c_[53] * (log((c_[13] / c_[12])));
    c_[62] = 0.460000 * c_[49] - c_[57];
  }
  else if (tissue == "N")
  {
    // Initial conditions
    s0_[0] = -49.7094187908202;
    s0_[1] = 0.0462303183096481;
    s0_[2] = 0.192515363116553;
    s0_[3] = 0.0797182955833868;
    s0_[4] = 0.949023698965401;
    s0_[5] = 0.000160310601192365;
    s0_[6] = 0.143642247226618;
    s0_[7] = 0.0243210273637729;
    s0_[8] = 0.0157156121147801;
    s0_[9] = 0.00179250298710316;
    s0_[10] = 0.975550840189597;
    s0_[11] = 0.774394220125623;
    s0_[12] = 0.0296516611999521;
    s0_[13] = 0.899732315818241;
    s0_[14] = 0.190111737767474;
    s0_[15] = 0.476404610622697;
    s0_[16] = 0.542303657353244;
    s0_[17] = 0.550559577208797;
    s0_[18] = 0.567277036232041;
    s0_[19] = 0.000184969821581882;
    s0_[20] = 1.11092514657408;
    s0_[21] = 0.296249516481577;
    s0_[22] = 0.0356473236675985;
    s0_[23] = 0.443317425115817;
    s0_[24] = 0.491718960234865;
    s0_[25] = 0.0723007987059414;
    s0_[26] = 0.0630771339141488;
    s0_[27] = 0.261430602900137;
    s0_[28] = 4.1497704886823e-5;

    s_ = s0_;

    // Model constants
    c_[0] = 8314.472;
    c_[1] = 310;
    c_[2] = 96485.3415;
    c_[3] = 4e-5;
    c_[4] = 0.05;
    c_[5] = 9999;
    c_[6] = 1;
    c_[7] = 0.001;
    c_[8] = -2;
    c_[9] = 0.001;
    c_[10] = 0;
    c_[11] = 0.0035;
    c_[12] = 140;
    c_[13] = 5.4;
    c_[14] = 0;
    c_[15] = 0.0012;
    c_[16] = -22.5;
    c_[17] = 0.14268;
    c_[18] = 8;
    c_[19] = 2.14455;
    c_[20] = 0.1369;
    c_[21] = 0.4315;
    c_[22] = 0;
    c_[23] = 0.0207;
    c_[24] = 395.3;
    c_[25] = 2.289;
    c_[26] = 26.44;
    c_[27] = 26.44;
    c_[28] = 4.663;
    c_[29] = 1628;
    c_[30] = 561.4;
    c_[31] = 3.663;
    c_[32] = 2;
    c_[33] = 140;
    c_[34] = 0;
    c_[35] = 1e-5;
    c_[36] = 0.009;
    c_[37] = 62;
    c_[38] = -15;
    c_[39] = -5;
    c_[40] = -5;
    c_[41] = -5;
    c_[42] = 0;
    c_[43] = -37.4;
    c_[44] = 0.0001;
    c_[45] = 0.0198;
    c_[46] = 0.00035;
    c_[47] = 73.1;
    c_[48] = 3.7;
    c_[49] = 3.18872e-6;
    c_[50] = 1500;
    c_[51] = 0.0006;
    c_[52] = 0.06;
    c_[53] = (c_[0] * c_[1]) / c_[2];
    c_[54] = c_[33] / (c_[28] + c_[33]);
    c_[55] = 0.0116000 * c_[49];
    c_[56] = 0.00120000 * c_[49];
    c_[57] = 0.0100000 * c_[49];
    c_[58] = c_[53] * (log((c_[13] / c_[12])));
    c_[59] = c_[18] / (c_[26] + c_[18]);
    c_[60] = c_[53] * (log((c_[33] / c_[18])));
    c_[61] = c_[53] * (log((c_[13] / c_[12])));
    c_[62] = 0.460000 * c_[49] - c_[57];
  }
  else if (tissue == "NH")
  {
    // Initial conditions
    s0_[0] = -69.760276376489;
    s0_[1] = 0.19584111039096;
    s0_[2] = 0.00141762995766447;
    s0_[3] = 0.00539771950846456;
    s0_[4] = 0.98638204514681;
    s0_[5] = 3.27335718697622e-5;
    s0_[6] = 0.0132200747771872;
    s0_[7] = 0.706622937059237;
    s0_[8] = 0.701626826712569;
    s0_[9] = 4.23500474189711e-5;
    s0_[10] = 0.998435073735753;
    s0_[11] = 0.998424216938754;
    s0_[12] = 0.00894826428663828;
    s0_[13] = 0.994837524153424;
    s0_[14] = 0.427382372349565;
    s0_[15] = 0.0910882041816457;
    s0_[16] = 0.890389014175329;
    s0_[17] = 0.74242774587522;
    s0_[18] = 0.746918323597392;
    s0_[19] = 3.73317163732586e-5;
    s0_[20] = 0.818671555213184;
    s0_[21] = 0.682159360306652;
    s0_[22] = 0.00739583869345967;
    s0_[23] = 0.133773505989393;
    s0_[24] = 0.765246381247448;
    s0_[25] = 0.0154714370092264;
    s0_[26] = 0.0135767851016544;
    s0_[27] = 0.449992033196647;
    s0_[28] = 1.21722016147587e-5;

    s_ = s0_;

    // Model constants
    c_[0] = 8314.472;
    c_[1] = 310;
    c_[2] = 96485.3415;
    c_[3] = 2.9e-5;
    c_[4] = 0.05;
    c_[5] = 99999;
    c_[6] = 1;
    c_[7] = 0.001;
    c_[8] = -2;
    c_[9] = 0;
    c_[10] = 0;
    c_[11] = 0.002;
    c_[12] = 140;
    c_[13] = 5.4;
    c_[14] = 0.015;
    c_[15] = 0.002;
    c_[16] = -40;
    c_[17] = 0.1968;
    c_[18] = 8;
    c_[19] = 5.916;
    c_[20] = 0.1369;
    c_[21] = 0.4315;
    c_[22] = 0;
    c_[23] = 0.0207;
    c_[24] = 395.3;
    c_[25] = 2.289;
    c_[26] = 26.44;
    c_[27] = 26.44;
    c_[28] = 4.663;
    c_[29] = 1628;
    c_[30] = 561.4;
    c_[31] = 3.663;
    c_[32] = 2;
    c_[33] = 140;
    c_[34] = 5e-7;
    c_[35] = 1e-5;
    c_[36] = 0.021;
    c_[37] = 62.1;
    c_[38] = 0;
    c_[39] = -6.61;
    c_[40] = -5;
    c_[41] = -5;
    c_[42] = 0.014;
    c_[43] = -37.4;
    c_[44] = 0;
    c_[45] = 0.0198;
    c_[46] = 0.00035;
    c_[47] = 73.1;
    c_[48] = 3.7;
    c_[49] = 4.39823e-6;
    c_[50] = 1805.6;
    c_[51] = 0.0006;
    c_[52] = 0.06;
    c_[53] = (c_[0] * c_[1]) / c_[2];
    c_[54] = c_[33] / (c_[28] + c_[33]);
    c_[55] = 0.0116000 * c_[49];
    c_[56] = 0.00120000 * c_[49];
    c_[57] = 0.0100000 * c_[49];
    c_[58] = c_[53] * (log((c_[13] / c_[12])));
    c_[59] = c_[18] / (c_[26] + c_[18]);
    c_[60] = c_[53] * (log((c_[33] / c_[18])));
    c_[61] = c_[53] * (log((c_[13] / c_[12])));
    c_[62] = 0.460000 * c_[49] - c_[57];
  }
}

/*----------------------------------------------------------------------*
 |  Compute Reaction Coefficient for Material and new State Variables   |
 |                                                (public)  ljag  09/13 |
 *----------------------------------------------------------------------*/
double MyocardInada::rea_coeff(const double phi, const double dt)
{
  s0_[0] = phi;
  s_[0] = phi;

  // Compute new gating variables
  // ----------------------------

  // s_[17] is achf in component acetylcholine_sensitive_current_achf_gate (dimensionless).
  a_[15] = 120.0 / (1.0 + (exp((-(s0_[0] + 50.0) / 15.0))));
  // r_[17] =  c_[47]*(1.0 - s0_[17]) -  a_[15]*s0_[17];
  s_[17] = tools_.gating_var_calc(dt, s0_[17], c_[47] / (a_[15] + c_[47]), 1 / (c_[47] + a_[15]));

  // s_[18] is achs in component acetylcholine_sensitive_current_achs_gate (dimensionless).
  a_[16] = 5.82 / (1.0 + (exp((-(s0_[0] + 50.0) / 15.0))));
  // r_[18] =  c_[48]*(1.0 - s0_[18]) -  a_[16]*s0_[18];
  s_[18] = tools_.gating_var_calc(dt, s0_[18], c_[48] / (a_[16] + c_[48]), 1 / (c_[48] + a_[16]));

  // r_[24] is d/dt f_TMM in component intracellular_calcium_concentration (dimensionless).
  a_[18] = 2277.0 * 2.5 * ((1.0 - s0_[23]) - s0_[24]) - 751.0 * s0_[24];
  // r_[24] = a_[18];
  s_[24] = tools_.gating_var_calc(dt, s0_[24],
      2277.0 * 2.5 * (1.0 - s0_[23]) / (2277.0 * 2.5 * (1.0 - s0_[23]) + 751.0),
      1.0 / (2277.0 * 2.5 * (1.0 - s0_[23]) + 751.0));

  // s_[1] is y in component hyperpolarising_activated_current_y_gate (dimensionless).
  a_[0] =
      1.0 / (1.0 + (exp((((s0_[0] + 83.19) - (-7.2 * (pow(c_[10], 0.69))) /
                                                 ((pow(1.26e-05, 0.69)) + (pow(c_[10], 0.69)))) /
                         13.56))));
  a_[19] = 0.25 + 2.0 * (exp((-(pow((s0_[0] + 70.0), 2.0)) / 500.0)));
  // r_[1] = (a_[0] - s0_[1])/a_[19];
  s_[1] = tools_.gating_var_calc(dt, s0_[1], a_[0], a_[19]);

  // s_[2] is paf in component rapid_delayed_rectifier_potassium_current_paf_gate (dimensionless).
  a_[1] = 1.0 / (1.0 + (exp(((s0_[0] + 10.22) / -8.5))));
  a_[20] = 1.0 / (17.0 * (exp((0.0398 * s0_[0]))) + 0.211 * (exp((-0.051 * s0_[0]))));
  // r_[2] = (a_[1] - s0_[2])/a_[20];
  s_[2] = tools_.gating_var_calc(dt, s0_[2], a_[1], a_[20]);

  // s_[3] is pas in component rapid_delayed_rectifier_potassium_current_pas_gate (dimensionless).
  a_[2] = 1.0 / (1.0 + (exp(((s0_[0] + 10.22) / -8.5))));
  a_[21] = 0.33581 + 0.90673 * (exp((-(pow((s0_[0] + 10.0), 2.0)) / 988.05)));
  // r_[3] = (a_[2] - s0_[3])/a_[21];
  s_[3] = tools_.gating_var_calc(dt, s0_[3], a_[2], a_[21]);

  // s_[10] is f in component L_type_calcium_current_f_gate (dimensionless).
  a_[8] = 1.0 / (1.0 + (exp(((s0_[0] - (-24.0 + c_[40])) / 6.31))));
  a_[27] = 0.01 + 0.1539 * (exp((-(pow((s0_[0] + 40.0), 2.0)) / 185.670)));
  // r_[10] = (a_[8] - s0_[10])/a_[27];
  s_[10] = tools_.gating_var_calc(dt, s0_[10], a_[8], a_[27]);

  // s_[11] is f2 in component L_type_calcium_current_f2_gate (dimensionless).
  a_[9] = 1.0 / (1.0 + (exp(((s0_[0] - (-24.0 + c_[41])) / 6.31))));
  a_[28] = 0.06 + 0.48076 * 2.25 * (exp((-(pow((s0_[0] - -40.0), 2.0)) / 138.04)));
  // r_[11] = (a_[9] - s0_[11])/a_[28];
  s_[11] = tools_.gating_var_calc(dt, s0_[11], a_[9], a_[28]);

  // s_[12] is r in component transient_outward_potassium_current_r_gate (dimensionless).
  a_[29] = 0.000596 + 0.003118 / (1.037 * (exp((0.09 * (s0_[0] + 30.61)))) +
                                     0.396 * (exp((-0.12 * (s0_[0] + 23.84)))));
  a_[10] = 1.0 / (1.0 + (exp(((s0_[0] - 7.44) / -16.4))));
  // r_[12] = (a_[10] - s0_[12])/a_[29];
  s_[12] = tools_.gating_var_calc(dt, s0_[12], a_[10], a_[29]);

  // s_[13] is q_fast in component transient_outward_potassium_current_qfast_gate (dimensionless).
  a_[30] = 0.01266 + 4.72716 / (1.0 + (exp(((s0_[0] + 154.500) / 23.96))));
  a_[11] = 1.0 / (1.0 + (exp(((s0_[0] + 33.8) / 6.12))));
  // r_[13] = (a_[11] - s0_[13])/a_[30];
  s_[13] = tools_.gating_var_calc(dt, s0_[13], a_[11], a_[30]);

  // s_[14] is q_slow in component transient_outward_potassium_current_qslow_gate (dimensionless).
  a_[31] = 0.100000 + 4.0 * (exp((-(pow((s0_[0] + 65.0), 2.0)) / 500.0)));
  a_[12] = 1.0 / (1.0 + (exp(((s0_[0] + 33.8) / 6.12))));
  // r_[14] = (a_[12] - s0_[14])/a_[31];
  s_[14] = tools_.gating_var_calc(dt, s0_[14], a_[12], a_[31]);

  //
  a_[4] = s0_[0] + 44.4000;
  a_[23] = ((fabs(a_[4])) < c_[35] ? (-460.0 * -12.6730) / (exp((a_[4] / -12.6730)))
                                   : (-460.0 * a_[4]) / ((exp((a_[4] / -12.6730))) - 1.0));
  a_[36] = 18400.0 * (exp((a_[4] / -12.6730)));
  // r_[6] =  a_[23]*(1.0 - s0_[6]) -  a_[36]*s0_[6];
  s_[6] = tools_.gating_var_calc(dt, s0_[6], a_[23] / (a_[23] + a_[36]), 1 / (a_[23] + a_[36]));

  // s_[6] is m in component fast_sodium_current_m_gate (dimensionless).
  a_[3] = (1.0 / (1.0 + (exp(((s0_[0] + 4.90000) / 15.1400))))) *
          (1.0 - 0.300000 * (exp((-(pow(s0_[0], 2.0)) / 500.0))));
  a_[22] = 92.0100 * (exp((-0.0183000 * s0_[0])));
  a_[35] = 603.600 * (exp((0.00942000 * s0_[0])));
  a_[43] = 1.0 / (a_[22] + a_[35]);
  // r_[4] = (a_[3] - s0_[4])/a_[43];
  s_[4] = tools_.gating_var_calc(dt, s0_[4], a_[3], a_[43]);

  // s_[5] is Casub in component intracellular_calcium_concentration (millimolar).
  a_[5] = 44.9000 * (exp(((s0_[0] + 66.9000) / -5.57000)));
  a_[24] = 1491.00 / (1.0 + 323.300 * (exp(((s0_[0] + 94.6000) / -12.9000))));
  a_[37] = a_[5] / (a_[5] + a_[24]);
  a_[44] = 0.0300000 / (1.0 + (exp(((s0_[0] + 40.0) / 6.0)))) + 0.000350000;
  // r_[7] = (a_[37] - s0_[7])/a_[44];
  s_[7] = tools_.gating_var_calc(dt, s0_[7], a_[37], a_[44]);

  // s_[8] is h2 in component fast_sodium_current_h2_gate (dimensionless).
  a_[6] = 44.9000 * (exp(((s0_[0] + 66.9000) / -5.57000)));
  a_[25] = 1491.00 / (1.0 + 323.300 * (exp(((s0_[0] + 94.6000) / -12.9000))));
  a_[38] = a_[6] / (a_[6] + a_[25]);
  a_[45] = 0.120000 / (1.0 + (exp(((s0_[0] + 60.0) / 2.0)))) + 0.00295000;
  // r_[8] = (a_[38] - s0_[8])/a_[45];
  s_[8] = tools_.gating_var_calc(dt, s0_[8], a_[38], a_[45]);

  // s_[9] is d in component L_type_calcium_current_d_gate (dimensionless).
  a_[39] = 1.0 / (1.0 + (exp(((s0_[0] - (-3.20000 + c_[38])) / c_[39]))));
  a_[7] = (-26.1200 * (s0_[0] + 35.0)) / ((exp(((s0_[0] + 35.0) / -2.50000))) - 1.0) +
          (-78.1100 * s0_[0]) / ((exp((-0.208000 * s0_[0]))) - 1.0);
  a_[26] = (10.5200 * (s0_[0] - 5.0)) / ((exp((0.400000 * (s0_[0] - 5.0)))) - 1.0);
  a_[46] = 1.0 / (a_[7] + a_[26]);
  // r_[9] = (a_[39] - s0_[9])/a_[46];
  s_[9] = tools_.gating_var_calc(dt, s0_[9], a_[39], a_[46]);

  // s_[15] is qa in component sustained_outward_potassium_current_qa_gate (dimensionless).
  a_[32] = 1.0 / (0.150000 * (exp((-s0_[0] / 11.0))) + 0.200000 * (exp((-s0_[0] / 700.0))));
  a_[40] = 1.0 / (16.0 * (exp((s0_[0] / 8.0))) + 15.0 * (exp((s0_[0] / 50.0))));
  a_[47] = 0.00100000 / (a_[32] + a_[40]);
  a_[13] = 1.0 / (1.0 + (exp(((s0_[0] - -49.1000) / -8.98000))));
  // r_[15] = (a_[13] - s0_[15])/a_[47];
  s_[15] = tools_.gating_var_calc(dt, s0_[15], a_[13], a_[47]);

  // s_[16] is qi in component sustained_outward_potassium_current_qi_gate (dimensionless).
  a_[14] = 0.150400 / (3100.0 * (exp((s0_[0] / 13.0))) + 700.0 * (exp((s0_[0] / 70.0))));
  a_[33] = 0.150400 / (95.0 * (exp((-s0_[0] / 10.0))) + 50.0 * (exp((-s0_[0] / 700.0)))) +
           0.000229000 / (1.0 + (exp((-s0_[0] / 5.0))));
  a_[48] = 0.00100000 / (a_[14] + a_[33]);
  a_[41] = a_[14] / (a_[14] + a_[33]);
  // r_[16] = (a_[41] - s0_[16])/a_[48];
  s_[16] = tools_.gating_var_calc(dt, s0_[16], a_[41], a_[48]);

  // Compute membrane currents
  // -------------------------
  a_[66] =
      (((c_[34] * (pow(s_[6], 3.0)) * (0.635 * s_[7] + 0.365 * s_[8]) * c_[33] * s_[0] * c_[2]) /
           c_[53]) *
          ((exp(((s_[0] - c_[60]) / c_[53]))) - 1.0)) /
      ((exp((s_[0] / c_[53]))) - 1.0);
  a_[69] =
      (c_[45] * s0_[17] * s0_[18] * (pow(c_[10], 1.5))) / ((pow(c_[46], 1.5)) + (pow(c_[10], 1.5)));
  a_[70] = (((a_[69] * c_[13]) / (10.0 + c_[13])) * (s_[0] - c_[58])) /
           (1.0 + (exp((((s_[0] - c_[58]) - 140.0) / (2.5 * c_[53])))));
  a_[71] = c_[36] * s_[9] * (0.675 * s_[10] + 0.325 * s_[11]) * (s_[0] - c_[37]) *
           (1.0 - ((a_[70] * c_[10]) / (9.0e-05 + c_[10])) / 1.0);
  a_[67] = c_[42] * s_[12] * (0.45 * s_[13] + 0.55 * s_[14]) * (s_[0] - c_[61]);
  a_[42] = c_[11] * (0.9 * s_[2] + 0.1 * s_[3]) * s_[4] * (s_[0] - c_[58]);
  a_[34] = s_[1] * c_[9] * (s_[0] + 30.0);
  a_[68] = c_[44] * s_[15] * s_[16] * (s_[0] - c_[43]);
  a_[49] = c_[14] * (0.5 + 0.5 / (1.0 + (exp(((s0_[0] + 30.0) / 5.0)))));
  a_[50] = (a_[49] * (pow((c_[13] / (c_[13] + 0.59)), 3.0)) * (s_[0] + 81.9)) /
           (1.0 + (exp(((1.393 * (s_[0] + 81.9 + 3.6)) / c_[53]))));
  a_[58] = exp(((-c_[21] * s0_[0]) / (2.0 * c_[53])));
  a_[53] = 1.0 + (c_[32] / c_[31]) * (1.0 + (exp(((c_[22] * s0_[0]) / c_[53])))) + c_[33] / c_[29] +
           (pow(c_[33], 2.0)) / (c_[29] * c_[30]) + (pow(c_[33], 3.0)) / (c_[29] * c_[30] * c_[28]);
  a_[55] =
      (((pow(c_[33], 2.0)) / (c_[29] * c_[30]) + (pow(c_[33], 3.0)) / (c_[29] * c_[30] * c_[28])) *
          (exp(((-c_[21] * s0_[0]) / (2.0 * c_[53]))))) /
      a_[53];
  a_[56] = ((c_[32] / c_[31]) * (exp(((-c_[22] * s0_[0]) / c_[53])))) / a_[53];
  a_[54] = exp(((c_[21] * s0_[0]) / (2.0 * c_[53])));
  a_[61] = a_[58] * c_[54] * (a_[55] + a_[56]) + a_[56] * a_[54] * (c_[59] + a_[58]);
  a_[57] = 1.0 +
           (s0_[5] / c_[23]) * (1.0 + (exp(((-c_[20] * s0_[0]) / c_[53]))) + c_[18] / c_[27]) +
           c_[18] / c_[24] + (pow(c_[18], 2.0)) / (c_[24] * c_[25]) +
           (pow(c_[18], 3.0)) / (c_[24] * c_[25] * c_[26]);
  a_[60] = ((s0_[5] / c_[23]) * (exp(((-c_[20] * s0_[0]) / c_[53])))) / a_[57];
  a_[59] =
      (((pow(c_[18], 2.0)) / (c_[24] * c_[25]) + (pow(c_[18], 3.0)) / (c_[24] * c_[25] * c_[26])) *
          (exp(((c_[21] * s0_[0]) / (2.0 * c_[53]))))) /
      a_[57];
  a_[62] = a_[54] * c_[59] * (a_[59] + a_[60]) + a_[58] * a_[60] * (c_[54] + a_[54]);
  a_[63] = a_[59] * c_[59] * (a_[55] + a_[56]) + a_[60] * a_[55] * (c_[59] + a_[58]);
  a_[64] = a_[55] * c_[54] * (a_[59] + a_[60]) + a_[59] * a_[56] * (c_[54] + a_[54]);
  a_[65] = (c_[19] * (a_[62] * a_[56] - a_[61] * a_[60])) / (a_[61] + a_[62] + a_[63] + a_[64]);
  a_[52] = (c_[17] * (pow((c_[18] / (5.64 + c_[18])), 3.0)) *
               (pow((c_[13] / (0.621 + c_[13])), 2.0)) * 1.6) /
           (1.5 + (exp((-(s_[0] + 60.0) / 40.0))));
  a_[51] = c_[15] * (s_[0] - c_[16]);
  a_[17] = 0.0;  //(VOI_>=c_[4]&&VOI_<=c_[5]&&(VOI_ - c_[4]) -  (floor(((VOI_ -
                 // c_[4])/c_[6])))*c_[6]<=c_[7] ? c_[8] : 0.0);

  // Compute reaction coefficient (I_Na + I_CaL + I_to + I_Kr + I_f + I_st + I_K1 + I_NaCa + I_p +
  // I_b + I_ACh)/C
  // ------------------------------------------------------------------------------------------------------------
  r_[0] = (a_[66] + a_[71] + a_[67] + a_[42] + a_[34] + a_[68] + a_[50] + a_[65] + a_[52] + a_[51] +
              a_[70] + a_[17]) /
          c_[3];

  // Calcium dynamics
  // ----------------

  // s0_[20] is Ca_up in component intracellular_calcium_concentration (millimolar).
  a_[73] = 5.0 / (1.0 + c_[51] / s0_[19]);
  a_[74] = (s0_[20] - s0_[21]) / c_[52];
  // r_[20] = a_[73] - ( a_[74]*c_[56])/c_[55];
  double up_inf = (s0_[21] + a_[73] * c_[52] * c_[55] / c_[56]);
  s_[20] = tools_.gating_var_calc(dt, s0_[20], up_inf, c_[52] * c_[55] / c_[56]);

  // s_[22] is f_TC in component intracellular_calcium_concentration (dimensionless).
  a_[75] = 88800.0 * s0_[19] * (1.0 - s0_[22]) - 446.0 * s0_[22];
  // r_[22] = a_[75];
  s_[22] = tools_.gating_var_calc(dt, s0_[22], 88800.0 * s0_[19] / (88800.0 * s0_[19] + 446.0),
      1.0 / (88800.0 * s0_[19] + 446.0));

  // s_[23] is f_TMC in component intracellular_calcium_concentration (dimensionless).
  a_[77] = 227700.0 * s0_[19] * ((1.0 - s0_[23]) - s0_[24]) - 7.51 * s0_[23];
  r_[23] = a_[77];
  s_[23] = tools_.gating_var_calc(dt, s0_[23],
      227700.0 * s0_[19] * (1 - s0_[24]) / (227700.0 * s0_[19] * (1 - s0_[24]) + 7.51),
      1.0 / (227700.0 * s0_[19] * (1 - s0_[24]) + 7.51));

  // s_[21] is Ca_rel in component intracellular_calcium_concentration (millimolar).
  a_[76] = (c_[50] * (s0_[21] - s0_[5])) / (1.0 + (pow((0.0012 / s0_[5]), 2.0)));
  a_[80] = 534.0 * s0_[21] * (1.0 - s0_[27]) - 445.0 * s0_[27];
  // r_[21] = (a_[74] - a_[76]) - 10.0*a_[80];
  double tau_Carel = 1.0 / c_[52] + c_[50] / (1.0 + (pow((0.0012 / s0_[5]), 2.0)));
  double Carel_inf =
      s0_[20] / c_[52] + c_[50] * s0_[5] / (1.0 + (pow((0.0012 / s0_[5]), 2.0))) - 10.0 * a_[80];
  s_[21] = tools_.gating_var_calc(dt, s0_[21], Carel_inf, tau_Carel);

  // s_[19] is Cai in component intracellular_calcium_concentration (millimolar).
  a_[72] = (s0_[5] - s0_[19]) / 4.0e-05;
  a_[78] = 227700.0 * s0_[19] * (1.0 - s0_[25]) - 542.0 * s0_[25];
  r_[19] = (a_[72] * c_[57] - a_[73] * c_[55]) / c_[62] -
           (0.045 * a_[78] + 0.031 * a_[75] + 0.062 * a_[77]);
  s_[19] = tools_.gating_var_calc(dt, s0_[19], 0, -s0_[19] / r_[19]);

  // s_[25] is f_CMi in component intracellular_calcium_concentration (dimensionless).
  // r_[25] = a_[78];
  s_[25] = tools_.gating_var_calc(dt, s0_[25], 227700.0 * s0_[19] / (227700.0 * s0_[19] + 542.0),
      1.0 / (227700.0 * s0_[19] + 542.0));

  // s_[26] is f_CMs in component intracellular_calcium_concentration (dimensionless).
  a_[79] = 227700.0 * s0_[5] * (1.0 - s0_[26]) - 542.0 * s0_[26];
  // r_[26] = a_[79];
  s_[26] = tools_.gating_var_calc(dt, s0_[26], 227700.0 * s0_[5] / (227700.0 * s0_[5] + 542.0),
      1.0 / (227700.0 * s0_[5] + 542.0));

  // s_[27] is f_CQ in component intracellular_calcium_concentration (dimensionless).
  r_[27] = a_[80];
  s_[27] = tools_.gating_var_calc(
      dt, s0_[27], 534.0 * s0_[21] / (534.0 * s0_[21] + 445.0), 1.0 / (534.0 * s0_[21] + 445.0));

  // s_[5] is Casub in component intracellular_calcium_concentration (millimolar).
  a_[81] = (0.115 * s0_[5] * (1.0 - s0_[28]) - 1.0 * s0_[28]);
  r_[5] = (((-(a_[71] - 2.0 * a_[65]) / (2.0 * c_[2]) + a_[76] * c_[56]) / c_[57] - a_[72]) -
              0.045 * a_[79]) -
          (0.031 / 1.2) * a_[81];
  s_[5] = tools_.gating_var_calc(dt, s0_[5], 0, -s0_[5] / r_[5]);

  // s_[28] is f_CSL in component intracellular_calcium_concentration (dimensionless).
  // r_[28] = a_[81];
  s_[28] = tools_.gating_var_calc(
      dt, s0_[28], s0_[5] / (s0_[5] + 1.0 / 0.115), 1.0 / (0.115 * s0_[5] + 1.0));

  // Update rest of state variables
  // ------------------------------
  // s_[5] = s0_[5] + dt*r_[5];
  // s_[19] = s0_[19] + dt*r_[19];
  // s_[20] = s0_[20] + dt*r_[20];
  // s_[21] = s0_[21] + dt*r_[21];

  //  for (int i=19; i<29; i++)
  //    s_[i] = s0_[i] + dt*r_[i];


  double reacoeff = r_[0];

  return reacoeff;
}

/*----------------------------------------------------------------------*
 |  returns number of internal state variables of the material  cbert 08/13 |
 *----------------------------------------------------------------------*/
int MyocardInada::get_number_of_internal_state_variables() const { return 29; }

/*----------------------------------------------------------------------*
 |  returns current internal state of the material          cbert 08/13 |
 *----------------------------------------------------------------------*/
double MyocardInada::get_internal_state(const int k) const
{
  double val = 0.0;
  if (k == -1)
  {
    val = s0_[19];
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
void MyocardInada::set_internal_state(const int k, const double val)
{
  if (k == -1)
  {
    s0_[19] = val;
    s_[19] = val;
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
int MyocardInada::get_number_of_ionic_currents() const { return 11; }

/*----------------------------------------------------------------------*
 |  returns current internal currents          cbert 08/13 |
 *----------------------------------------------------------------------*/
double MyocardInada::get_ionic_currents(const int k) const
{
  double val = 0.0;
  switch (k)
  {
    case 0:
      val = a_[34];
      break;
    case 1:
      val = a_[42];
      break;
    case 2:
      val = a_[50];
      break;
    case 3:
      val = a_[51];
      break;
    case 4:
      val = a_[52];
      break;
    case 5:
      val = a_[65];
      break;
    case 6:
      val = a_[66];
      break;
    case 7:
      val = a_[67];
      break;
    case 8:
      val = a_[68];
      break;
    case 9:
      val = a_[70];
      break;
    case 10:
      val = a_[71];
      break;
  }

  return val;
}

/*----------------------------------------------------------------------*
 |  update of material at the end of a time step             ljag 07/12 |
 *----------------------------------------------------------------------*/
void MyocardInada::update(const double phi, const double dt)
{
  voi_ += dt;
  // update initial values for next time step
  for (int i = 0; i < 29; i++) s0_[i] = s_[i];

  return;
}

FOUR_C_NAMESPACE_CLOSE
